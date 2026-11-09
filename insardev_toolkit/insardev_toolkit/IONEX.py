# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
"""The CODE final global ionosphere maps (IONEX) of the NASA CDDIS archive, one file per UTC day, stored and read as published.

The multidimensional VRT index basedir/IONEX.vrt lists the day files present, one source each.
"""
import datetime as dt
import gzip
import os
import re

import numpy as np
from numba import njit

from .datagrid import datagrid
from .progressbar_joblib import progressbar_joblib
from .utils_atmosphere import AtmosphereModel

# IONEX 1.0: the label of a record is in columns 61 to 80; a map is START OF TEC (or RMS) MAP, EPOCH OF CURRENT MAP, one
# LAT/LON1/LON2/DLON/H record and its values per latitude, END OF TEC (or RMS) MAP
_HEADER_END = re.compile(r'^.{60}END OF HEADER', re.M)
_MARKS = tuple(re.compile(re.escape(label)) for label in ('START OF TEC MAP', 'END OF TEC MAP', 'START OF RMS MAP',
                                                           'END OF RMS MAP', 'EPOCH OF CURRENT MAP'))
# the NOTE of the insar.dev cache, as ASF prints it for its cache proxy (ASF.py)
CACHE_NOTE = ('NOTE: Using insar.dev Cache API. Free for non-commercial use; license required for funded academic, institutional, '
              'or professional use.')
# a map value is an integer of 5 characters (I5), 16 a record: blanks, a minus sign, digits


@njit(nogil=True, cache=True)
def _i5_kernel(chars, out):
    """The integers of right-justified fields of 5 characters, written to out.

    Parameters
    ----------
    chars : numpy.ndarray - uint8, the characters of the fields.
    out : numpy.ndarray - int32, one value per field.

    Returns
    -------
    int - -1, or the index of the first field that is not an integer: blanks, a minus sign, digits up to the last character.

    Examples
    --------
    >>> _i5_kernel(np.frombuffer(b'  123', dtype=np.uint8), np.empty(1, dtype=np.int32))
    -1
    """
    for i in range(chars.size // 5):
        k, j, sign, value = 5 * i, 0, 1, 0
        while j < 4 and chars[k + j] == 32:
            j += 1
        if chars[k + j] == 45:
            sign, j = -1, j + 1
            if j == 5:
                return i
        while j < 5:
            digit = chars[k + j] - 48
            if digit < 0 or digit > 9:
                return i
            value = 10 * value + digit
            j += 1
        out[i] = sign * value
    return -1


def _i5(fields):
    """The integers of a string of right-justified fields of 5 characters (IONEX 1.0, I5).

    Parameters
    ----------
    fields : str - The fields, 5 characters each.

    Returns
    -------
    numpy.ndarray - int32, one value per field.

    Raises
    ------
    ValueError: When a field is not a right-justified integer.

    Examples
    --------
    >>> _i5('  123-1234')
    array([  123, -1234], dtype=int32)
    """
    chars = np.frombuffer(fields.encode('latin-1'), dtype=np.uint8)
    out = np.empty(chars.size // 5, dtype=np.int32)
    if chars.size % 5 or _i5_kernel(chars, out) >= 0:
        raise ValueError('ERROR: an IONEX map holds a value that is not an integer of 5 characters')
    return out


class IONEX(datagrid, progressbar_joblib, AtmosphereModel):
    """The CODE final IONEX maps of the UTC days of a set of scenes: through the insar.dev cache without an account, from CDDIS with one.

    Attributes
    ----------
    cache_url : str - The insar.dev cache of the IONEX products, used without an account; the file of a day is <cache_url>/<yyyy>/<doy>/<name>.
    base_url : str - The CDDIS archive of the IONEX products, used with an Earthdata Login account; the file of a day is <base_url>/<yyyy>/<doy>/<name>.
    earthdata_host : str - The Earthdata Login host.
    source : str - The name of the folder of the files in basedir, and of the index of the maps.
    new_names_from : datetime.date - The first day of the long file names.
    """
    cache_url = 'https://ionex-cache-cddis.insar.dev'
    base_url = 'https://cddis.nasa.gov/archive/gnss/products/ionex'
    earthdata_host = 'urs.earthdata.nasa.gov'
    source = 'IONEX'
    new_names_from = dt.date(2022, 11, 27)

    def __init__(self, username=None, password=None):
        """Initialize the IONEX downloader: with an Earthdata Login account (CDDIS), or without one (the insar.dev cache).

        Parameters
        ----------
        username : str, optional - The Earthdata Login username, for the direct download from CDDIS; never printed. Without an account, the day files come through the insar.dev cache.
        password : str, optional - The Earthdata Login password, with the username.
        """
        self.username = username
        self.password = password

    def _phase(self, folder, times, wavelength, factor, heights=None):
        """The terms of the ionospheric model of a folder on the native nodes, in phase units, lazy.

        Parameters
        ----------
        folder : str - The folder of the downloaded model fields.
        times : list of str or pandas.Timestamp - The acquisition times, UTC.
        wavelength : float - The radar wavelength, metres.
        factor : float - The ionospheric factor, dimensionless.
        heights : ignored (the ionospheric terms have one height axis with identical values along it).
        """
        from . import utils_iono
        return utils_iono.zenith_tiles(folder, times, wavelength, factor)

    def download(self, basedir, scenes, skip_exist=True, n_jobs=8, retries=30,
                 timeout_second=3, min_rate='100KB', min_rate_window=60, debug=False):
        """Download the CODE final IONEX maps of the UTC days of the scenes, and index them in basedir/IONEX.vrt.

        Each day's file is stored as published in basedir/IONEX, and the maps are read from it. A complete index basedir/IONEX.vrt means all done: None at once, no request. Otherwise the missing days are fetched and the index is written over the day files present after they all succeeded: through the insar.dev cache without an account, from CDDIS with an Earthdata Login account.

        Parameters
        ----------
        basedir : str - The download folder (files in basedir/IONEX, index basedir/IONEX.vrt).
        scenes : pandas.DataFrame, object or list - A DataFrame with startTime, an object with such a DataFrame as .df, or Sentinel-1 scene names; only the start time is used.
        skip_exist : bool, optional - True keeps the existing day files and downloads nothing for them. Default True.
        n_jobs : int, optional - Parallel downloads, at most two per CPU. Negative values follow joblib: -1 all CPUs, -2 all but one, and so on; 0 raises ValueError. Default 8.
        retries : int, optional - Attempts per file request, the first included. Default 30.
        timeout_second : float, optional - Seconds between attempts. Default 3.
        min_rate : str or float, optional - Bytes per second a download must keep, averaged over min_rate_window. Default '100KB'.
        min_rate_window : float, optional - Seconds over which min_rate is averaged. Default 60.
        debug : bool, optional - Print the failed attempts that are retried too. Default False.

        Returns
        -------
        pandas.DataFrame or None - One row per UTC day fetched by this call: epoch, tile ('global'), path (the day file); None when nothing is downloaded.

        Raises
        ------
        ValueError: For one of username and password without the other, scenes without a start time, a negative retries, a day without a CODE final map, the Earthdata Login page (with an account) or an HTML page (without one), or a file that fails the checks.

        Examples
        --------
        >>> IONEX().download('data', scenes)                    # the day files through the insar.dev cache
        >>> IONEX('user', 'password').download('data', scenes)  # the day files from CDDIS
        """
        import joblib
        import pandas as pd
        from tqdm.auto import tqdm
        from .HTTP import attempts, download_jobs
        from .utils_epochs import _lists_all
        from .utils_files import exists

        # a negative retries raises before any request
        try:
            attempts(retries)
        except ValueError as e:
            raise ValueError(f'ERROR: {e}') from None

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
        if not days:
            return None
        folder = os.path.join(basedir, self.source)
        paths = [os.path.join(folder, IONEX._day_name(day)) for day in days]
        # a VRT that lists every day means all done: no day file is read, no request, no bar, no pool, no folder
        if skip_exist and _lists_all(os.path.join(basedir, f'{self.source}.vrt'), paths):
            return None

        # a day whose day file is held in folder is kept (skip_exist) and not requested; an account is one username and one
        # password, both or neither, checked before any request (no netrc file is read)
        fetch = [day for day in days if not (skip_exist and exists(os.path.join(folder, IONEX._day_name(day))))]
        if bool(self.username) != bool(self.password):
            raise ValueError('ERROR: IONEX takes both username and password, or neither')
        account = bool(self.username)

        outcomes = []
        if fetch:
            os.makedirs(folder, exist_ok=True)
            # without an account the day files come through the insar.dev cache (as ASF does); the NOTE is printed only for them
            if not account:
                print(CACHE_NOTE)
            if isinstance(min_rate, str):
                from dask.utils import parse_bytes
                min_rate = parse_bytes(min_rate)
            workers = max(1, min(download_jobs(n_jobs) or 1, len(fetch)))
            base = self.base_url if account else self.cache_url
            with self.progressbar_joblib(tqdm(desc='Downloading IONEX'.ljust(25), total=len(fetch))):
                outcomes = joblib.Parallel(n_jobs=workers)(
                    joblib.delayed(IONEX._download_day)(day, folder, base, self.username, self.password, skip_exist,
                                                        retries, timeout_second, min_rate, min_rate_window, debug)
                    for day in fetch)
        # the index over the day files present, written only after every day succeeded
        self._index(basedir)
        rows = [{'epoch': pd.Timestamp(day), 'tile': 'global', 'path': path}
                for day, path, status in outcomes if status == 'new']
        if not rows:
            return None
        return pd.DataFrame(rows, columns=['epoch', 'tile', 'path']).sort_values('epoch', ignore_index=True)

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
        """A requests session that never reads a netrc file: the Earthdata Login account is sent only to the Earthdata Login host and
        the CDDIS host; without an account (the insar.dev cache) no request carries an Authorization header, redirects included.

        Parameters
        ----------
        username : str or None - The Earthdata Login username, None or empty without an account.
        password : str or None - The Earthdata Login password.

        Returns
        -------
        requests.Session - The session for the file requests.

        Examples
        --------
        >>> IONEX._session('user', 'password')
        """
        import requests
        from urllib.parse import urlsplit

        def _no_auth(request):
            # requests reads the netrc entry of a request only when neither the request nor the session has an auth
            return request

        hosts = {IONEX.earthdata_host, urlsplit(IONEX.base_url).hostname}

        class _Session(requests.Session):
            def rebuild_auth(self, prepared_request, response):
                """Remove the Authorization header from a redirect, and set the account again only when the redirect goes to an Earthdata
                or CDDIS host. requests would read the netrc file here, so this never calls it.

                Parameters
                ----------
                prepared_request : requests.PreparedRequest - The request of the redirect.
                response : requests.Response - The redirect answer.

                Examples
                --------
                >>> session.rebuild_auth(prepared_request, response)
                """
                prepared_request.headers.pop('Authorization', None)
                if username and urlsplit(prepared_request.url).hostname in hosts:
                    prepared_request.prepare_auth((username, password))

        session = _Session()
        session.auth = _no_auth
        return session

    @staticmethod
    def _login_check(body, url, day, account=True):
        """Raise when the answer is an HTML page instead of the file: the Earthdata Login page with an account, else the cache's page. The message names the day and the URL, never the account.

        Parameters
        ----------
        body : bytes or None - The body of the answer, or None for an HTML answer that was not read.
        url : str - The URL requested.
        day : datetime.date - The UTC day of the file.
        account : bool, optional - True with an Earthdata Login account (the login page is named). Default True.

        Raises
        ------
        ValueError: When the answer is an HTML page.

        Examples
        --------
        >>> IONEX._login_check(b'IONEX', 'https://cddis.nasa.gov/x', dt.date(2016, 5, 11))
        """
        if body is None or body.lstrip()[:1] == b'<':
            if account:
                raise ValueError(f'ERROR: Earthdata Login refused the account for the IONEX file of {day} at {url}: '
                                 f'the answer is the login page, not the file')
            raise ValueError(f'ERROR: the answer for the IONEX file of {day} at {url} is an HTML page, not the file')

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
        text = IONEX._text(body, f'the IONEX file of {day}')
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
    def _text(body, what):
        """The text of an IONEX file as published: gzip, or compress (.Z) read by ncompress.

        Parameters
        ----------
        body : bytes - The file as published.
        what : str - The file, for the messages.

        Returns
        -------
        str - The decoded text (latin-1).

        Raises
        ------
        ValueError: When the file is neither a complete gzip stream nor a compress stream that ncompress decodes (the message names
        its code width).

        Examples
        --------
        >>> IONEX._text(open('codg1320.16i.Z', 'rb').read(), 'codg1320.16i.Z')
        """
        if body[:2] == b'\x1f\x8b':
            try:
                return gzip.decompress(body).decode('latin-1')
            except (OSError, EOFError) as e:
                raise ValueError(f'ERROR: {what} is not a complete gzip stream: {e}') from None
        if body[:2] == b'\x1f\x9d':
            import ncompress
            try:
                return ncompress.decompress(body).decode('latin-1')
            except ValueError as e:
                width = f' of {body[2] & 0x1f} bits' if len(body) > 2 else ''
                raise ValueError(f'ERROR: {what} is not a compress stream{width}: {e}') from None
        raise ValueError(f'ERROR: {what} is neither gzip nor compress ({body[:4]!r})')

    @staticmethod
    def _read(path):
        """The whole text of a stored day file.

        Parameters
        ----------
        path : str - The day file as published (basedir/IONEX/<name>).

        Returns
        -------
        str - The decoded text.

        Raises
        ------
        ValueError: When the file cannot be read, or is neither a complete gzip stream nor a compress stream.

        Examples
        --------
        >>> IONEX._read('basedir/IONEX/codg1320.16i.Z')
        """
        try:
            with open(path, 'rb') as f:
                body = f.read()
        except OSError as e:
            raise ValueError(f'ERROR: {path} cannot be read: {type(e).__name__}, delete it, then download the IONEX maps again'
                             ) from None
        return IONEX._text(body, path)

    @staticmethod
    def _head(path):
        """The header of a stored day file, up to END OF HEADER; a gzip file is decompressed only that far.

        Parameters
        ----------
        path : str - The day file as published (basedir/IONEX/<name>).

        Returns
        -------
        str - The header text.

        Raises
        ------
        ValueError: When the file cannot be read, is neither gzip nor compress, or has no END OF HEADER.

        Examples
        --------
        >>> IONEX._head('basedir/IONEX/codg1320.16i.Z')
        """
        try:
            with open(path, 'rb') as f:
                if f.read(2) == b'\x1f\x8b':
                    f.seek(0)
                    lines = []
                    with gzip.GzipFile(fileobj=f) as stream:
                        for line in stream:
                            lines.append(line)
                            if line[60:].strip() == b'END OF HEADER':
                                break
                    text = b''.join(lines).decode('latin-1')
                else:
                    f.seek(0)
                    text = IONEX._text(f.read(), path)
        except (OSError, EOFError) as e:
            raise ValueError(f'ERROR: {path} cannot be read: {type(e).__name__}, delete it, then download the IONEX maps again'
                             ) from None
        end = _HEADER_END.search(text)
        if end is None:
            raise ValueError(f'ERROR: {path} has no IONEX header: no END OF HEADER')
        return text[:end.end()]

    @staticmethod
    def _header(text):
        """The values of an IONEX header.

        Parameters
        ----------
        text : str - The IONEX text, its header at least.

        Returns
        -------
        dict - interval (seconds), maps (the count of the header), exponent, radius (km), hgt (HGT1, HGT2, DHGT, km),
        lat_grid and lon_grid (LAT1, LAT2, DLAT and LON1, LON2, DLON, degrees), lat and lon (the nodes in the published order),
        first and last (the epochs of the first and the last map, year to second; None without the label), body (the position
        after END OF HEADER).

        Raises
        ------
        ValueError: For a text without END OF HEADER, a header without one of the values, or a zero grid step.

        Examples
        --------
        >>> IONEX._header(open('2016.txt', encoding='latin-1').read())['exponent']
        -1
        """
        end = _HEADER_END.search(text)
        if end is None:
            raise ValueError('ERROR: the IONEX text has no END OF HEADER')
        header = {}
        for line in text[:end.start()].splitlines():
            label = line[60:].strip()
            if label:
                header.setdefault(label, line[:60].split())
        got = {}
        for key, label, count in (('interval', 'INTERVAL', 1), ('maps', '# OF MAPS IN FILE', 1), ('exponent', 'EXPONENT', 1),
                                  ('radius', 'BASE RADIUS', 1), ('hgt', 'HGT1 / HGT2 / DHGT', 3),
                                  ('lat_grid', 'LAT1 / LAT2 / DLAT', 3), ('lon_grid', 'LON1 / LON2 / DLON', 3)):
            values = header.get(label, [])
            if len(values) < count:
                raise ValueError(f'ERROR: the IONEX header has no {label}')
            got[key] = values[:count]
        lat1, lat2, dlat = [float(v) for v in got['lat_grid']]
        lon1, lon2, dlon = [float(v) for v in got['lon_grid']]
        if dlat == 0 or dlon == 0:
            raise ValueError('ERROR: the IONEX header has a zero grid step')
        nlat = round((lat2 - lat1) / dlat) + 1
        nlon = round((lon2 - lon1) / dlon) + 1
        first, last = (tuple(int(float(v)) for v in header[label][:6]) if len(header.get(label, [])) >= 6 else None
                       for label in ('EPOCH OF FIRST MAP', 'EPOCH OF LAST MAP'))
        return {'interval': int(float(got['interval'][0])), 'maps': int(float(got['maps'][0])),
                'exponent': int(float(got['exponent'][0])), 'radius': float(got['radius'][0]),
                'hgt': [float(v) for v in got['hgt']], 'lat_grid': [lat1, lat2, dlat], 'lon_grid': [lon1, lon2, dlon],
                'lat': [lat1 + k * dlat for k in range(nlat)], 'lon': [lon1 + k * dlon for k in range(nlon)],
                'first': first, 'last': last, 'body': end.end()}

    @staticmethod
    def _decode_maps(text):
        """The maps of an IONEX text, decoded losslessly: the epochs, the latitudes, the longitudes, the TEC and RMS maps as the integers published, and the header values.

        Parameters
        ----------
        text : str - The decoded IONEX text.

        Returns
        -------
        dict - epochs (year, month, day, hour, minute, second tuples), seconds (int64, since 1970-01-01), lat and lon (degrees,
        ascending), tec and rms (int32 maps, shape (epoch, lat, lon); rms None when the file has none), header (BASE_RADIUS,
        HGT1_HGT2_DHGT, LAT1_LAT2_DLAT, LON1_LON2_DLON, EXPONENT, INTERVAL), exponent and interval.

        Raises
        ------
        ValueError: For a header or a map that cannot be decoded.

        Examples
        --------
        >>> IONEX._decode_maps(open('2016.txt', encoding='latin-1').read())['exponent']
        -1
        """
        import bisect
        head = IONEX._header(text)
        lat_pub, lon_pub = head['lat'], head['lon']
        nlat, nlon = len(lat_pub), len(lon_pub)
        # a map row is its LAT/LON1/LON2/DLON/H record and the records of its values, 16 integers of 5 characters a record (IONEX 1.0)
        records, last = -(-nlon // 16), nlon - 16 * (-(-nlon // 16) - 1)
        found = re.compile(r'LAT/LON1/LON2/DLON/H[^\n]*\n((?:[^\n]*\n){%d})' % records).finditer(text, head['body'])
        rows = [(m.start(), m.group(1)) for m in found]
        starts = [pos for pos, _ in rows]
        marks = sorted((m.start(), m.group()) for mark in _MARKS for m in mark.finditer(text, head['body']))
        maps = {'TEC': ([], []), 'RMS': ([], [])}
        lats = []
        kind = epoch = start = None
        for pos, label in marks:
            if label.startswith('START'):
                kind, epoch, start = label.split()[2], None, pos
            elif label.startswith('EPOCH'):
                if kind is not None:
                    epoch = tuple(int(float(v)) for v in text[pos - 60:pos].split()[:6])
            else:
                i, j = (bisect.bisect_left(starts, start), bisect.bisect_left(starts, pos)) if kind else (0, 0)
                if kind is None or label.split()[2] != kind or epoch is None or j - i != nlat:
                    raise ValueError(f'ERROR: an IONEX {kind or label} map is not complete')
                lats.append(' '.join(text[starts[t] - 60:starts[t] - 52] for t in range(i, j)))
                maps[kind][0].append(epoch)
                maps[kind][1].append([rows[t][1] for t in range(i, j)])
                kind = None

        tec_epochs, tec_rows = maps['TEC']
        rms_epochs, rms_rows = maps['RMS']
        if not tec_rows:
            raise ValueError('ERROR: the IONEX file has no TEC map')
        if len(tec_rows) != head['maps']:
            raise ValueError(f'ERROR: the IONEX file has {len(tec_rows)} TEC maps, the header says {head["maps"]}')
        if rms_rows and rms_epochs != tec_epochs:
            raise ValueError('ERROR: the RMS maps of the IONEX file do not match its TEC maps')
        got = np.fromstring(' '.join(lats), dtype=np.float64, sep=' ')
        if got.size != len(lats) * nlat or not np.allclose(got.reshape(-1, nlat), np.asarray(lat_pub)[None, :], rtol=0, atol=1e-6):
            raise ValueError('ERROR: an IONEX map has latitudes other than the header grid')

        order_lat = np.argsort(lat_pub)
        order_lon = np.argsort(lon_pub)

        def grid(ms):
            """The integer maps of the value records of a list of maps, shape (epoch, lat, lon), latitudes and longitudes ascending.

            Parameters
            ----------
            ms : list - The value records of each map, one string per row.

            Examples
            --------
            >>> grid(tec_rows)
            """
            rows = [r for m in ms for r in m]
            fields = ''.join(rows).replace('\r', '').replace('\n', '')
            if len(fields) != 5 * nlon * len(rows):
                # a record padded or cut: each record to its own fields, a row of other length names its count
                fields = []
                for row in rows:
                    lines = [line.rstrip('\r') for line in row.split('\n')[:records]]
                    cut = ''.join(line[:80] if i < records - 1 else line[:5 * last] for i, line in enumerate(lines))
                    if len(row.split()) != nlon:
                        raise ValueError(f'ERROR: an IONEX map row has {len(row.split())} values, {nlon} expected')
                    if len(cut) != 5 * nlon or any(len(line.rstrip()) > (80 if i < records - 1 else 5 * last)
                                                   for i, line in enumerate(lines)):
                        raise ValueError(f'ERROR: an IONEX map row is not in records of 16 values of 5 characters (IONEX 1.0): '
                                         f'{lines[0].strip()[:40]!r}')
                    fields.append(cut)
                fields = ''.join(fields)
            arr = _i5(fields)
            return arr.reshape(len(ms), nlat, nlon)[:, order_lat, :][:, :, order_lon]

        base = dt.datetime(1970, 1, 1)
        seconds = np.array([int((dt.datetime(*e) - base).total_seconds()) for e in tec_epochs], dtype=np.int64)
        return {
            'epochs': tec_epochs,
            'seconds': seconds,
            'lat': np.array(lat_pub, dtype=np.float64)[order_lat],
            'lon': np.array(lon_pub, dtype=np.float64)[order_lon],
            'tec': grid(tec_rows),
            'rms': grid(rms_rows) if rms_rows else None,
            'header': {'BASE_RADIUS': head['radius'], 'HGT1_HGT2_DHGT': head['hgt'], 'LAT1_LAT2_DLAT': head['lat_grid'],
                       'LON1_LON2_DLON': head['lon_grid'], 'EXPONENT': head['exponent'], 'INTERVAL': head['interval']},
            'exponent': head['exponent'],
            'interval': head['interval'],
        }

    @staticmethod
    def _download_day(day, folder, base_url, username, password, skip_exist, retries, timeout_second, min_rate,
                      min_rate_window, debug):
        """Download the CODE final file of one UTC day and store it as published, after the checks of a complete file of the day.

        Parameters
        ----------
        day : datetime.date - The UTC day.
        folder : str - The folder of the files (basedir/IONEX).
        base_url : str - The CDDIS archive of the IONEX products (with an account) or the insar.dev cache (without one).
        username, password : str - The Earthdata Login account, or None for the insar.dev cache.
        skip_exist : bool - True keeps an existing file and requests nothing for its day.
        retries : int - Attempts per file request, the first included.
        timeout_second : float - Seconds between attempts.
        min_rate : str or float - Bytes per second a download must keep.
        min_rate_window : float - Seconds over which min_rate is averaged.
        debug : bool - Print the failed attempts that are retried too.

        Returns
        -------
        tuple - (day, the path of the day file, 'new' or 'existing').

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
        from .utils_files import exists, write_file

        name = IONEX._day_name(day)
        raw = os.path.join(folder, name)
        if skip_exist and exists(raw):
            return day, raw, 'existing'

        url = f'{base_url}/{day.year}/{day.timetuple().tm_yday:03d}/{name}'
        session = IONEX._session(username, password)
        n = attempts(retries)
        body = None
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
                IONEX._check_ionex(body, day)
                break
            except NotFound:
                raise ValueError(f'ERROR: CODE has no final IONEX map for {day} ({url}); no other product is '
                                 f'used') from None
            except Exception as e:
                stop = final(e)
                reason = ' '.join(str(returned(e) or e).split())[:200]
                if username and answer(e) and answer(e).split()[0] in ('401', '403'):
                    reason = f'Earthdata Login refused the account ({reason})'
                if debug or stop or attempt + 1 == n:
                    print(f'ERROR: download attempt {attempt + 1}/{n} failed'
                          f'{" (not retried)" if stop else ""} for {url}: {reason}')
                if stop or attempt + 1 == n:
                    raise ValueError(f'ERROR: the IONEX file of {day} from {url} failed: {reason}') from None
                time.sleep(timeout_second)
        IONEX._login_check(body, url, day, account=bool(username))
        write_file(raw, body)
        return day, raw, 'new'
