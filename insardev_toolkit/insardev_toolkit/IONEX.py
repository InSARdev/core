# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
"""The CODE final global ionosphere maps (IONEX) of the NASA CDDIS archive, one file per UTC day; each day is stored as published, and its maps are also written as a NetCDF4 file.

The multidimensional VRT index basedir/IONEX.vrt lists the day files present, one source each.
"""
import datetime as dt
import gzip
import os

import numpy as np

from .datagrid import datagrid
from .progressbar_joblib import progressbar_joblib
from .utils_atmosphere import AtmosphereModel


class IONEX(datagrid, progressbar_joblib, AtmosphereModel):
    """The CODE final IONEX maps of the UTC days of a set of scenes, downloaded from CDDIS with an Earthdata Login account.

    Attributes
    ----------
    base_url : str - The CDDIS archive of the IONEX products; the file of a day is <base_url>/<yyyy>/<doy>/<name>.
    earthdata_host : str - The Earthdata Login host.
    source : str - The name of the folder of the files in basedir, and of the index of the maps.
    new_names_from : datetime.date - The first day of the long file names.
    """
    base_url = 'https://cddis.nasa.gov/archive/gnss/products/ionex'
    earthdata_host = 'urs.earthdata.nasa.gov'
    source = 'IONEX'
    new_names_from = dt.date(2022, 11, 27)

    def _phase(self, folder, times, wavelength, factor):
        """The terms of the ionospheric model of a folder on the native nodes, in phase units, lazy.

        Parameters
        ----------
        folder : str - The folder of the downloaded model fields.
        times : list of str or pandas.Timestamp - The acquisition times, UTC.
        wavelength : float - The radar wavelength, metres.
        factor : float - The ionospheric factor, dimensionless.
        """
        from . import utils_iono
        return utils_iono.zenith_tiles(folder, times, wavelength, factor)

    def download(self, basedir, scenes, username=None, password=None, skip_exist=True, n_jobs=8, retries=30,
                 timeout_second=3, min_rate='100KB', min_rate_window=60, debug=False):
        """Download the CODE final IONEX maps of the UTC days of the scenes, and index them in basedir/IONEX.vrt.

        Each day's file is stored as published, and its maps as basedir/IONEX/<yyyymmdd>.nc; the index is written over the day files present after every call.

        Parameters
        ----------
        basedir : str - The download folder (files in basedir/IONEX, index basedir/IONEX.vrt).
        scenes : pandas.DataFrame, object or list - A DataFrame with startTime, an object with such a DataFrame as .df, or Sentinel-1 scene names; only the start time is used.
        username : str - The Earthdata Login username; never printed.
        password : str - The Earthdata Login password.
        skip_exist : bool, optional - True keeps the existing day files and downloads nothing for them. Default True.
        n_jobs : int, optional - Parallel downloads, at most two per CPU. Default 8.
        retries : int, optional - Attempts per file request, the first included. Default 30.
        timeout_second : float, optional - Seconds between attempts. Default 3.
        min_rate : str or float, optional - Bytes per second a download must keep, averaged over min_rate_window. Default '100KB'.
        min_rate_window : float, optional - Seconds over which min_rate is averaged. Default 60.
        debug : bool, optional - Print the failed attempts that are retried too. Default False.

        Returns
        -------
        pandas.DataFrame - One row per UTC day: epoch, tile ('global'), path (the NetCDF4 file), status ('new' or 'existing').

        Raises
        ------
        ValueError: For missing credentials, scenes without a start time, a negative retries, a day without a CODE final map, the Earthdata Login page, or a file that fails the checks.

        Examples
        --------
        >>> ionex = IONEX()
        >>> ionex.download('data', scenes)
        """
        import joblib
        import pandas as pd
        from tqdm.auto import tqdm
        from .HTTP import attempts, download_jobs

        # a negative retries raises before any request
        try:
            attempts(retries)
        except ValueError as e:
            raise ValueError(f'ERROR: {e}') from None

        # the account is the two arguments only: no netrc file is read, before any request
        if not username or not password:
            raise ValueError('ERROR: no Earthdata Login account: give the username and password arguments')

        if isinstance(scenes, pd.DataFrame) or isinstance(getattr(scenes, 'df', None), pd.DataFrame):
            frame = scenes if isinstance(scenes, pd.DataFrame) else scenes.df
            if 'startTime' not in frame.columns:
                raise ValueError('ERROR: the scenes have no startTime column')
            try:
                times = pd.to_datetime(frame['startTime'])
            except (TypeError, ValueError):
                raise ValueError('ERROR: the startTime column of the scenes cannot be read as times') from None
            if times.isna().any():
                raise ValueError(f'ERROR: {int(times.isna().sum())} scene(s) have no startTime')
            if times.dt.tz is not None:
                times = times.dt.tz_convert('UTC').dt.tz_localize(None)
        elif isinstance(scenes, (list, tuple)) and all(isinstance(s, str) for s in scenes):
            from .utils_epochs import _name_start
            starts = []
            for s in scenes:
                try:
                    starts.append(_name_start(s))
                except ValueError:
                    raise ValueError(f'ERROR: no Sentinel-1 start time in the scene name {s}') from None
            times = pd.DatetimeIndex(starts)
        else:
            raise ValueError('ERROR: scenes must be a DataFrame with startTime, an object with .df, or a list of '
                             'Sentinel-1 scene names')

        days = sorted({t.date() for t in times})
        folder = os.path.join(basedir, self.source)
        try:
            if not days:
                return pd.DataFrame(columns=['epoch', 'tile', 'path', 'status'])

            os.makedirs(folder, exist_ok=True)

            if isinstance(min_rate, str):
                from dask.utils import parse_bytes
                min_rate = parse_bytes(min_rate)
            workers = max(1, min(download_jobs(n_jobs) or 1, len(days)))
            with self.progressbar_joblib(tqdm(desc='Downloading IONEX'.ljust(25), total=len(days))):
                outcomes = joblib.Parallel(n_jobs=workers)(
                    joblib.delayed(IONEX._download_day)(day, folder, self.base_url, username, password, skip_exist,
                                                        retries, timeout_second, min_rate, min_rate_window, debug)
                    for day in days)
            rows = [{'epoch': pd.Timestamp(day), 'tile': 'global', 'path': path, 'status': status}
                    for day, path, status in outcomes]
            return pd.DataFrame(rows, columns=['epoch', 'tile', 'path', 'status']).sort_values('epoch',
                                                                                              ignore_index=True)
        finally:
            self._index(basedir)

    def _index(self, basedir):
        """Write the multidimensional VRT index basedir/IONEX.vrt over the day files present; nothing is written when no day file exists.

        Parameters
        ----------
        basedir : str - The download folder; its IONEX subfolder holds the day files.

        Examples
        --------
        >>> IONEX()._index('basedir')
        """
        from .utils_epochs import _DAY_FILE, index_days
        folder = os.path.join(basedir, self.source)
        if not os.path.isdir(folder) or not any(_DAY_FILE.fullmatch(name) for name in os.listdir(folder)):
            return
        index_days(basedir, self.source)

    @staticmethod
    def _day_name(day):
        """The name of the CODE final IONEX file of a UTC day, as the CDDIS archive publishes it.

        The long file name is used from 2022-11-27 on.

        Parameters
        ----------
        day : datetime.date - The UTC day.

        Returns
        -------
        str - The file name.

        Examples
        --------
        >>> IONEX._day_name(dt.date(2016, 5, 11))
        'codg1320.16i.Z'
        """
        doy = day.timetuple().tm_yday
        if day >= IONEX.new_names_from:
            return f'COD0OPSFIN_{day.year}{doy:03d}0000_01D_01H_GIM.INX.gz'
        return f'codg{doy:03d}0.{day.year % 100:02d}i.Z'

    @staticmethod
    def _session(username, password):
        """A requests session for the Earthdata Login account, sent only to the Earthdata Login host and the CDDIS host.

        Parameters
        ----------
        username : str - The Earthdata Login username.
        password : str - The Earthdata Login password.

        Returns
        -------
        requests.Session - The session for the file requests.

        Examples
        --------
        >>> IONEX._session('user', 'password')
        """
        import requests
        from urllib.parse import urlsplit
        hosts = {IONEX.earthdata_host, urlsplit(IONEX.base_url).hostname}

        class _EarthdataSession(requests.Session):
            def rebuild_auth(self, prepared_request, response):
                """Remove the account from a redirect, and set it again only when the redirect goes to an Earthdata or CDDIS host.

                Parameters
                ----------
                prepared_request : requests.PreparedRequest - The request of the redirect.
                response : requests.Response - The redirect answer.

                Examples
                --------
                >>> session.rebuild_auth(prepared_request, response)
                """
                prepared_request.headers.pop('Authorization', None)
                if urlsplit(prepared_request.url).hostname in hosts:
                    prepared_request.prepare_auth((username, password))

        def _no_auth(request):
            # requests reads the netrc entry of a request only when neither the request nor the session has an auth
            return request

        session = _EarthdataSession()
        session.auth = _no_auth
        return session

    @staticmethod
    def _login_check(body, url, day):
        """Raise when the answer is the Earthdata Login page instead of the file. The message names the day and the URL, never the account.

        Parameters
        ----------
        body : bytes or None - The body of the answer, or None for an HTML answer that was not read.
        url : str - The URL requested.
        day : datetime.date - The UTC day of the file.

        Raises
        ------
        ValueError: When the answer is an HTML page.

        Examples
        --------
        >>> IONEX._login_check(b'IONEX', 'https://cddis.nasa.gov/x', dt.date(2016, 5, 11))
        """
        if body is None or body.lstrip()[:1] == b'<':
            raise ValueError(f'ERROR: Earthdata Login refused the account for the IONEX file of {day} at {url}: '
                             f'the answer is the login page, not the file')

    @staticmethod
    def _check_ionex(body, day):
        """The decoded text of a downloaded IONEX file, after the checks of a complete file of the day.

        Parameters
        ----------
        body : bytes - The file as published (gzip, or compress .Z).
        day : datetime.date - The UTC day the file must hold.

        Returns
        -------
        str - The decoded text of the file (ASCII).

        Raises
        ------
        ValueError: When the file is not a complete file of the day.

        Examples
        --------
        >>> IONEX._check_ionex(open('codg1320.16i.Z', 'rb').read(), dt.date(2016, 5, 11))
        """
        if body[:2] == b'\x1f\x8b':
            try:
                raw = gzip.decompress(body)
            except (OSError, EOFError) as e:
                raise ValueError(f'ERROR: the IONEX file of {day} is not a complete gzip stream: {e}') from None
        elif body[:2] == b'\x1f\x9d':
            raw = IONEX._lzw_decode(body)
        else:
            raise ValueError(f'ERROR: the IONEX file of {day} is neither gzip nor compress ({body[:4]!r})')
        text = raw.decode('latin-1')
        first = text.split('\n', 1)[0]
        if 'IONEX VERSION / TYPE' not in first or 'IONOSPHERE MAPS' not in first:
            raise ValueError(f'ERROR: the file of {day} has no IONEX 1.0 header')
        if 'END OF FILE' not in text[-200:]:
            raise ValueError(f'ERROR: the IONEX file of {day} has no END OF FILE: a truncated transfer')
        maps = IONEX._decode_maps(text)
        if maps['epochs'][0] != (day.year, day.month, day.day, 0, 0, 0):
            raise ValueError(f'ERROR: the IONEX file of {day} starts at {maps["epochs"][0]}, not at the day')
        if maps['interval'] <= 0:
            raise ValueError(f'ERROR: the IONEX file of {day} has INTERVAL {maps["interval"]}, not a positive interval')
        expected = 86400 // maps['interval'] + 1
        if len(maps['epochs']) != expected:
            raise ValueError(f'ERROR: the IONEX file of {day} has {len(maps["epochs"])} maps, {expected} expected')
        return text

    @staticmethod
    def _lzw_decode(data):
        """The bytes of a compress (.Z, LZW) stream of 12 to 16 bits.

        Parameters
        ----------
        data : bytes - The whole compress stream.

        Returns
        -------
        bytes - The decoded bytes.

        Raises
        ------
        ValueError: For a stream that is not compress, of another code width, or cut inside a code.

        Examples
        --------
        >>> IONEX._lzw_decode(b'\\x1f\\x9d\\x8c\\x41\\x00')
        b'A'
        """
        if len(data) < 3 or data[:2] != b'\x1f\x9d':
            raise ValueError('ERROR: not a compress (.Z) stream')
        maxbits = data[2] & 0x1f
        block = bool(data[2] & 0x80)
        if not 12 <= maxbits <= 16:
            raise ValueError(f'ERROR: a compress stream of {maxbits} bits, 12 to 16 expected')
        body = data[3:]
        total = len(body) * 8
        first = 257 if block else 256
        size = 1 << maxbits
        prefix = [0] * size
        tail = bytearray(size)
        head = bytearray(size)
        for c in range(256):
            head[c] = c

        def expand(code):
            """The bytes of a code of the table: the string it stands for.

            Parameters
            ----------
            code : int - A literal code (below 256) or a table entry.

            Examples
            --------
            >>> expand(65)
            b'A'
            """
            parts = bytearray()
            while code >= 256:
                parts.append(tail[code])
                code = prefix[code]
            parts.append(code)
            parts.reverse()
            return bytes(parts)

        out = bytearray()
        # a group is 8 codes of the width in use: it starts at gstart and ends at a byte boundary of that width
        width, free, prev, pos, gstart = 9, first, -1, 0, 0
        while True:
            if free > (1 << width) - 1 and width < maxbits:
                group = 8 * width
                pos = gstart + -(-(pos - gstart) // group) * group
                gstart = pos
                width += 1
            if pos + width > total:
                break
            byte, offset = pos >> 3, pos & 7
            chunk = int.from_bytes(body[byte:byte + ((offset + width + 7) >> 3)], 'little')
            code = (chunk >> offset) & ((1 << width) - 1)
            pos += width
            if block and code == 256:
                group = 8 * width
                pos = gstart + -(-(pos - gstart) // group) * group
                gstart = pos
                width, free, prev = 9, first, -1
                continue
            if prev < 0:
                if code > 255:
                    raise ValueError('ERROR: a corrupt compress stream: the first code is not a byte')
                out.append(code)
                prev = code
                continue
            if code < free:
                s = expand(code)
            elif code == free:
                s = expand(prev) + bytes([head[prev]])
            else:
                raise ValueError('ERROR: a corrupt compress stream: a code past the table')
            out += s
            if free < size:
                prefix[free] = prev
                tail[free] = s[0]
                head[free] = head[prev]
                free += 1
            prev = code
        if not 0 <= total - pos <= 7:
            raise ValueError('ERROR: a truncated compress stream: the last code is cut short')
        return bytes(out)

    @staticmethod
    def _decode_maps(text, path=None):
        """The maps of an IONEX text, decoded losslessly: the epochs, the latitudes, the longitudes, the TEC and RMS maps as the integers published, and the header values.

        Parameters
        ----------
        text : str - The decoded IONEX text.
        path : str, optional - The NetCDF4 file to write; None writes nothing. Default None.

        Returns
        -------
        dict - epochs (year, month, day, hour, minute, second tuples), seconds (int64, since 1970-01-01), lat and lon (degrees), tec and rms (integer maps, shape (epoch, lat, lon); rms None when absent), header (header values by attribute name), exponent (EXPONENT of the maps), interval (seconds).

        Raises
        ------
        ValueError: For a header or a map that cannot be decoded.

        Examples
        --------
        >>> IONEX._decode_maps(open('2016.txt', encoding='latin-1').read())['exponent']
        -1
        """
        import io
        from .utils_files import write_file

        lines = text.splitlines()
        header = {}
        i = 0
        while i < len(lines) and lines[i][60:].strip() != 'END OF HEADER':
            label = lines[i][60:].strip()
            if label:
                header.setdefault(label, lines[i][:60].split())
            i += 1

        def values(label, count):
            """The first count values of a header label.

            Parameters
            ----------
            label : str - The header label, such as 'INTERVAL'.
            count : int - The number of values.

            Examples
            --------
            >>> values('INTERVAL', 1)
            ['3600']
            """
            got = header.get(label, [])
            if len(got) < count:
                raise ValueError(f'ERROR: the IONEX header has no {label}')
            return got[:count]

        interval = int(float(values('INTERVAL', 1)[0]))
        nmaps = int(float(values('# OF MAPS IN FILE', 1)[0]))
        exponent = int(float(values('EXPONENT', 1)[0]))
        radius = float(values('BASE RADIUS', 1)[0])
        hgt = [float(v) for v in values('HGT1 / HGT2 / DHGT', 3)]
        lat1, lat2, dlat = [float(v) for v in values('LAT1 / LAT2 / DLAT', 3)]
        lon1, lon2, dlon = [float(v) for v in values('LON1 / LON2 / DLON', 3)]
        if dlat == 0 or dlon == 0:
            raise ValueError('ERROR: the IONEX header has a zero grid step')
        nlat = round((lat2 - lat1) / dlat) + 1
        nlon = round((lon2 - lon1) / dlon) + 1
        lat_pub = [lat1 + k * dlat for k in range(nlat)]
        lon_pub = [lon1 + k * dlon for k in range(nlon)]

        maps = {'TEC': [], 'RMS': []}
        kind, epoch, rows = None, None, None
        i += 1
        while i < len(lines):
            line = lines[i]
            label = line[60:].strip()
            if label in ('START OF TEC MAP', 'START OF RMS MAP'):
                kind, epoch, rows = label.split()[2], None, []
            elif label in ('END OF TEC MAP', 'END OF RMS MAP'):
                if kind is None or label.split()[2] != kind or epoch is None or len(rows) != nlat:
                    raise ValueError(f'ERROR: an IONEX {kind or label} map is not complete')
                maps[kind].append((epoch, rows))
                kind = None
            elif kind is not None and label == 'EPOCH OF CURRENT MAP':
                epoch = tuple(int(float(v)) for v in line[:60].split()[:6])
            elif kind is not None and label == 'LAT/LON1/LON2/DLON/H':
                got, j = [], i + 1
                while len(got) < nlon and j < len(lines):
                    got += [int(v) for v in lines[j].split()]
                    j += 1
                if len(got) != nlon:
                    raise ValueError(f'ERROR: an IONEX map row has {len(got)} values, {nlon} expected')
                rows.append((float(line[:8]), got))
                i = j
                continue
            i += 1

        tec_maps, rms_maps = maps['TEC'], maps['RMS']
        if not tec_maps:
            raise ValueError('ERROR: the IONEX file has no TEC map')
        if len(tec_maps) != nmaps:
            raise ValueError(f'ERROR: the IONEX file has {len(tec_maps)} TEC maps, the header says {nmaps}')
        epochs = [e for e, _ in tec_maps]
        if rms_maps and [e for e, _ in rms_maps] != epochs:
            raise ValueError('ERROR: the RMS maps of the IONEX file do not match its TEC maps')
        for _, rows in tec_maps + rms_maps:
            if not np.allclose([r[0] for r in rows], lat_pub, rtol=0, atol=1e-6):
                raise ValueError('ERROR: an IONEX map has latitudes other than the header grid')

        order_lat = np.argsort(lat_pub)
        order_lon = np.argsort(lon_pub)
        lat = np.array(lat_pub, dtype=np.float64)[order_lat]
        lon = np.array(lon_pub, dtype=np.float64)[order_lon]

        def grid(ms):
            """The integer map array of a list of maps, shape (epoch, lat, lon), with ascending latitudes and longitudes.

            Parameters
            ----------
            ms : list - The maps as (epoch, rows) pairs, the rows as (latitude, values).

            Examples
            --------
            >>> grid(tec_maps)
            """
            arr = np.array([[vals for _, vals in rows] for _, rows in ms], dtype=np.int32)
            return arr[:, order_lat, :][:, :, order_lon]

        tec = grid(tec_maps)
        rms = grid(rms_maps) if rms_maps else None
        base = dt.datetime(1970, 1, 1)
        seconds = np.array([int((dt.datetime(*e) - base).total_seconds()) for e in epochs], dtype=np.int64)
        result = {
            'epochs': epochs,
            'seconds': seconds,
            'lat': lat,
            'lon': lon,
            'tec': tec,
            'rms': rms,
            'header': {'BASE_RADIUS': radius, 'HGT1_HGT2_DHGT': hgt, 'LAT1_LAT2_DLAT': [lat1, lat2, dlat],
                       'LON1_LON2_DLON': [lon1, lon2, dlon], 'EXPONENT': exponent, 'INTERVAL': interval},
            'exponent': exponent,
            'interval': interval,
        }
        if path is None:
            return result

        from . import utils_tiles
        utils_tiles.hdf5_plugins()
        import h5py
        buf = io.BytesIO()
        with h5py.File(buf, 'w') as f:
            dtime = f.create_dataset('time', data=seconds, track_times=False)
            dtime.make_scale('time')
            dtime.attrs.update(units='seconds since 1970-01-01 00:00:00', calendar='standard', standard_name='time',
                               axis='T')
            dlat_ds = f.create_dataset('lat', data=lat, track_times=False)
            dlat_ds.make_scale('lat')
            dlat_ds.attrs.update(units='degrees_north', standard_name='latitude', axis='Y')
            dlon_ds = f.create_dataset('lon', data=lon, track_times=False)
            dlon_ds.make_scale('lon')
            dlon_ds.attrs.update(units='degrees_east', standard_name='longitude', axis='X')
            for name, arr, long_name in [('tec', tec, 'vertical total electron content as published'),
                                         ('rms', rms, 'RMS of the vertical total electron content as published')]:
                if arr is None:
                    continue
                dz = f.create_dataset(name, data=arr, chunks=(1, arr.shape[1], arr.shape[2]), compression='gzip',
                                      track_times=False)
                for d, scale in zip(dz.dims, (dtime, dlat_ds, dlon_ds)):
                    d.attach_scale(scale)
                dz.attrs.update(units='0.1 TECU', long_name=long_name, exponent=exponent)
            f.attrs['Conventions'] = 'CF-1.8'
            f.attrs['source'] = 'CODE final IONEX, maps as published (integers times 10**EXPONENT TECU)'
            for key, value in result['header'].items():
                f.attrs[key] = value
        data = buf.getvalue()
        with h5py.File(io.BytesIO(data), 'r') as f:
            if not np.array_equal(f['tec'][()], tec) or not np.array_equal(f['time'][()], seconds):
                raise ValueError(f'ERROR: the NetCDF4 maps of {path} do not read back as decoded')
            if rms is not None and not np.array_equal(f['rms'][()], rms):
                raise ValueError(f'ERROR: the NetCDF4 maps of {path} do not read back as decoded')
        write_file(path, data)
        return result

    @staticmethod
    def _download_day(day, folder, base_url, username, password, skip_exist, retries, timeout_second, min_rate,
                      min_rate_window, debug):
        """Download the CODE final file of one UTC day, store it, and write its maps as a NetCDF4 file.

        Parameters
        ----------
        day : datetime.date - The UTC day.
        folder : str - The folder of the files (basedir/IONEX).
        base_url : str - The CDDIS archive of the IONEX products.
        username, password : str - The Earthdata Login account.
        skip_exist : bool - True keeps an existing file and requests nothing for its day.
        retries : int - Attempts per file request, the first included.
        timeout_second : float - Seconds between attempts.
        min_rate : str or float - Bytes per second a download must keep.
        min_rate_window : float - Seconds over which min_rate is averaged.
        debug : bool - Print the failed attempts that are retried too.

        Returns
        -------
        tuple - (day, the NetCDF4 path of the day, 'new' or 'existing').

        Raises
        ------
        ValueError: Naming the day and the URL, for a missing day, the Earthdata Login page, or a file that fails after all attempts.

        Examples
        --------
        >>> IONEX._download_day(dt.date(2016, 5, 11), 'basedir/IONEX', IONEX.base_url, 'user', 'password', True, 30, 3,
        ...                     100000, 60, False)
        """
        import time
        import contextlib
        from .HTTP import attempts, answer, final, send, read_body, raise_for_status, reply, returned, NotFound
        from .utils_files import exists

        name = IONEX._day_name(day)
        raw = os.path.join(folder, name)
        nc = os.path.join(folder, f'{day:%Y%m%d}.nc')
        have_raw = skip_exist and exists(raw)
        have_nc = skip_exist and exists(nc)
        if have_nc:
            return day, nc, 'existing'

        if have_raw:
            with open(raw, 'rb') as f:
                body = f.read()
            text = IONEX._check_ionex(body, day)
        else:
            from .utils_files import write_file
            url = f'{base_url}/{day.year}/{day.timetuple().tm_yday:03d}/{name}'
            session = IONEX._session(username, password)
            n = attempts(retries)
            body = text = None
            for attempt in range(n):
                try:
                    response = send(session.get, url, stream=True, timeout=(10, 300), check=False)
                    try:
                        if response.status_code in (404, 410):
                            raise NotFound(f'{url}: {reply(response)}')
                        raise_for_status(response, url)
                        if 'text/html' in response.headers.get('Content-Type', '').lower():
                            # the login page: not the file, raised below without a retry
                            break
                        body = read_body(response, min_rate, min_rate_window)
                    finally:
                        with contextlib.suppress(Exception):
                            response.close()
                    if body.lstrip()[:1] == b'<':
                        break
                    text = IONEX._check_ionex(body, day)
                    break
                except NotFound:
                    raise ValueError(f'ERROR: CODE has no final IONEX map for {day} ({url}); no other product is '
                                     f'used') from None
                except Exception as e:
                    stop = final(e)
                    reason = ' '.join(str(returned(e) or e).split())[:200]
                    if answer(e) and answer(e).split()[0] in ('401', '403'):
                        reason = f'Earthdata Login refused the account ({reason})'
                    if debug or stop or attempt + 1 == n:
                        print(f'ERROR: download attempt {attempt + 1}/{n} failed'
                              f'{" (not retried)" if stop else ""} for {url}: {reason}')
                    if stop or attempt + 1 == n:
                        raise ValueError(f'ERROR: the IONEX file of {day} from {url} failed: {reason}') from None
                    time.sleep(timeout_second)
            IONEX._login_check(body, url, day)
            write_file(raw, body)

        if not have_nc:
            IONEX._decode_maps(text, nc)
        return day, nc, 'new'
