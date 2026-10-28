# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
from .datagrid import datagrid
from .progressbar_joblib import progressbar_joblib
from .utils_atmosphere import AtmosphereModel


class ERA5(datagrid, progressbar_joblib, AtmosphereModel):
    """Download the ERA5 pressure levels and surface fields as 1-degree tiles from the public ARCO v3 store, one NetCDF4 file per tile and hour, with a VRT index.

    Values are stored as served, and the store needs no credentials.
    """

    # the ARCO v3 copy of ERA5 on Google Cloud (zarr v2, Blosc chunks, latitude 90 to -90, longitude 0 to 359.75):
    # the 37 pressure levels and the surface fields in one store
    url = 'https://storage.googleapis.com/gcp-public-data-arco-era5/ar/'
    store = 'full_37-1h-0p25deg-chunk-1.zarr-v3/'
    model_variables = ('temperature', 'specific_humidity', 'specific_cloud_liquid_water_content', 'geopotential')
    surface_variables = ('surface_pressure', 'geopotential_at_surface')
    # the grid: pressure levels of the store, node spacing in degrees, nodes per tile side (edge nodes kept)
    levels = 37
    step = 0.25
    nodes = 5
    # concurrent range requests: HTTP.fetch, one session per thread
    workers = 16
    # bytes of the answers to the GET requests of the last download
    bytes_requested = 0

    def _phase(self, folder, times, wavelength, factor):
        """The terms of the tropospheric model of a folder per tile, in phase units, lazy.

        Parameters
        ----------
        folder : str - The folder of the downloaded model fields.
        times : list of str or pandas.Timestamp - The acquisition times, UTC.
        wavelength : float - The radar wavelength, metres.
        factor : float - Not used by ERA5.
        """
        from . import utils_tropo
        return utils_tropo.zenith_tiles(folder, times, wavelength)

    @staticmethod
    def _block_ranges(offsets, cbytes, blocks):
        """The byte ranges of the given Blosc blocks of a chunk; neighbouring blocks form one range.

        Parameters
        ----------
        offsets : numpy.ndarray - The start byte of each block in the chunk.
        cbytes : int - The compressed size of the chunk in bytes.
        blocks : iterable of int - The block indices.

        Returns
        -------
        list of tuple - (first block, block after the last, start byte, end byte) of each range.

        Examples
        --------
        >>> ERA5._block_ranges([16, 120, 300], 400, [0, 2])
        [(0, 1, 16, 120), (2, 3, 300, 400)]
        """
        ends = [int(o) for o in offsets[1:]] + [int(cbytes)]
        runs = []
        for b in sorted(int(b) for b in blocks):
            if runs and runs[-1][1] == b:
                runs[-1] = (runs[-1][0], b + 1, runs[-1][2], ends[b])
            else:
                runs.append((b, b + 1, int(offsets[b]), ends[b]))
        return runs

    def _store_meta(self, store, arrays, net):
        """The metadata of one ARCO store: its attributes and the zarr metadata of the time and the named arrays.

        Parameters
        ----------
        store : str - The store URL with the trailing slash.
        arrays : tuple of str - The array names besides time: the level array and the variables.
        net : dict - The retry and timeout arguments: retries, timeout_second, min_rate, min_rate_window.

        Returns
        -------
        dict - {'store': URL, 'attrs': the store attributes, 'arrays': {name: zarr metadata}}.

        Raises
        ------
        ValueError: When an array has another layout.

        Examples
        --------
        >>> era5 = ERA5()
        >>> meta = era5._store_meta(ERA5.url + ERA5.store, ('level',) + ERA5.model_variables + ERA5.surface_variables,
        ...                         {'retries': 30, 'timeout_second': 3, 'min_rate': '100KB', 'min_rate_window': 60})
        """
        import json
        from .HTTP import fetch, NotFound

        def get(path):
            """The body of one store file.

            Parameters
            ----------
            path : str - The path below the store URL.

            Examples
            --------
            >>> get('.zattrs')
            """
            try:
                body = fetch(store + path, **net)
            except NotFound:
                raise ValueError(f'ERROR: {store + path} is not in the ERA5 store') from None
            self.bytes_requested += len(body)
            return body

        attrs = json.loads(get('.zattrs'))
        meta = {'store': store, 'attrs': attrs, 'arrays': {}}
        for name in ('time',) + tuple(arrays):
            zarray = json.loads(get(f'{name}/.zarray'))
            if zarray.get('zarr_format') != 2 or zarray.get('order') != 'C' or zarray.get('filters') is not None \
                    or (zarray.get('compressor') or {}).get('id') != 'blosc':
                raise ValueError(f'ERROR: {store}{name} is not a Blosc zarr v2 array in C order')
            shape, chunks = zarray['shape'], zarray['chunks']
            if name == 'time':
                ok = len(shape) == 1 and zarray['dtype'] == '<i8'
            elif name == 'level':
                ok = len(shape) == 1 and shape[0] == self.levels and chunks[0] == self.levels
            else:
                ok = shape[-2:] == [721, 1440] and chunks[-2:] == [721, 1440] and zarray['dtype'] == '<f4'
                if name in self.model_variables:
                    ok = ok and len(shape) == 4 and shape[1] == self.levels
                else:
                    ok = ok and len(shape) in (2, 3)
            if not ok:
                raise ValueError(f'ERROR: {store}{name} has the shape {shape}, chunks {chunks} and dtype '
                                 f'{zarray["dtype"]}, not the ERA5 layout')
            meta['arrays'][name] = zarray
        return meta

    def _hour_index(self, meta, hour, net):
        """The index of an hour (UTC) in the time array of a store, checked against the store's validity and its time value.

        Parameters
        ----------
        meta : dict - The metadata of the store.
        hour : datetime.datetime - The hour, naive UTC, on the hour.
        net : dict - The retry and timeout arguments.

        Returns
        -------
        int - The index of the hour in the store (hours since 1900-01-01).

        Raises
        ------
        ValueError: When the hour is outside the validity of the store, or the store holds another time at the index.

        Examples
        --------
        >>> era5 = ERA5()
        >>> era5._hour_index(meta, datetime(2016, 5, 11, 8), {'retries': 30, 'timeout_second': 3})
        """
        from datetime import datetime, timedelta
        import numcodecs
        import numpy as np
        from .HTTP import fetch, NotFound

        attrs = meta['attrs']
        start = datetime.fromisoformat(attrs['valid_time_start'])
        stop = datetime.fromisoformat(attrs['valid_time_stop_era5t']) + timedelta(hours=23)
        if not start <= hour <= stop:
            raise ValueError(f'ERROR: no ERA5 data for {hour:%Y-%m-%dT%H}:00 UTC: the store covers '
                             f'{start:%Y-%m-%dT%H}:00 to {stop:%Y-%m-%dT%H}:00 UTC')
        time = meta['arrays']['time']
        index = int((hour - datetime(1900, 1, 1)).total_seconds()) // 3600
        chunk, pos = divmod(index, time['chunks'][0])
        try:
            body = fetch(meta['store'] + f'time/{chunk}', **net)
        except NotFound:
            raise ValueError(f'ERROR: the time chunk {chunk} of {meta["store"]} is not in the store') from None
        self.bytes_requested += len(body)
        values = np.frombuffer(numcodecs.blosc.decompress(body), dtype=time['dtype'])
        if index >= time['shape'][0] or int(values[pos]) != index:
            held = int(values[pos]) if pos < values.size else None
            raise ValueError(f'ERROR: the time index {index} of {meta["store"]} holds {held}, expected {index}')
        return index

    def _levels(self, meta, net):
        """The pressure levels (hPa) of the store, from its level array.

        Parameters
        ----------
        meta : dict - The metadata of the store.
        net : dict - The retry and timeout arguments.

        Returns
        -------
        numpy.ndarray - The levels, int32, ascending.

        Raises
        ------
        ValueError: When the array does not hold ascending integers.

        Examples
        --------
        >>> era5 = ERA5()
        >>> era5._levels(meta, {'retries': 30, 'timeout_second': 3})
        """
        import numcodecs
        import numpy as np
        from .HTTP import fetch, NotFound

        try:
            body = fetch(meta['store'] + 'level/0', **net)
        except NotFound:
            raise ValueError(f'ERROR: the level array of {meta["store"]} is not in the store') from None
        self.bytes_requested += len(body)
        dtype = meta['arrays']['level']['dtype']
        values = np.frombuffer(numcodecs.blosc.decompress(body), dtype=dtype).astype(np.float64)
        if values.size != self.levels or np.any(values != np.round(values)) or np.any(np.diff(values) <= 0):
            raise ValueError(f'ERROR: the level array of {meta["store"]} is not {self.levels} ascending integers (hPa)')
        return values.astype(np.int32)

    def _read_hour(self, meta, index, tiles, invariant, net, pool):
        """The values of the tiles at one hour: the model variables on the pressure levels and the surface fields.

        Blocks are read by byte range; a chunk whose block route fails is read whole, with a warning.

        Parameters
        ----------
        meta : dict - The metadata of the store.
        index : int - The index of the hour in the store.
        tiles : list of tuple - (lon0, lat0) of the tiles, the lower-left corners in whole degrees.
        invariant : dict - {(lon0, lat0): (5, 5) float32 surface geopotential}, filled for the tiles read now.
        net : dict - The retry and timeout arguments.
        pool : concurrent.futures.Executor - The pool of the range requests.

        Returns
        -------
        dict - {(lon0, lat0): {variable: array}}: the model variables (37, 5, 5) with the level axis first, the surface fields (5, 5); float32 as served.

        Examples
        --------
        >>> era5 = ERA5()
        >>> era5._read_hour(meta, 1019984, [(150, -27)], {}, {'retries': 30}, pool)
        """
        import struct
        import numcodecs
        import numpy as np
        from .HTTP import fetch, NotFound

        n, step = self.nodes, self.step

        def get(url, start=None, stop=None):
            """The body of a store file, or of its byte range [start, stop).

            Parameters
            ----------
            url : str - The URL of the array chunk or metadata file.
            start, stop : int, optional - The first byte and the end (exclusive) of the range; the whole file when None.

            Examples
            --------
            >>> get(url, 0, 16)
            """
            headers = None if start is None else {'Range': f'bytes={start}-{stop - 1}'}
            try:
                body = fetch(url, headers=headers, **net)
            except NotFound:
                raise ValueError(f'ERROR: {url} is not in the ERA5 store') from None
            if start is not None and len(body) != stop - start:
                raise ValueError(f'ERROR: {url}: bytes {start}-{stop - 1} came as {len(body)} bytes')
            self.bytes_requested += len(body)
            return body

        def header(path, dtype, items):
            """The Blosc header and block offset table of one chunk, checked against the array.

            Parameters
            ----------
            path : str - The URL of the chunk.
            dtype : numpy.dtype - The element type of the array.
            items : int - The number of elements of a chunk.

            Returns
            -------
            dict - The Blosc header fields (flags, item size, block size, byte count) and the block byte offsets.

            Raises
            ------
            ValueError: When the header is not that of a chunk of this array (a plain copy, or an item size, block size or byte count that does not match), or the block offsets do not fit the chunk.

            Examples
            --------
            >>> header(url, np.dtype('<f4'), 1000)
            """
            head = get(path, 0, 16)
            ver, verlz, flags, typesize, nbytes, blocksize, cbytes = struct.unpack('<BBBBiii', head)
            if typesize != dtype.itemsize or flags & 0x02 or blocksize <= 0 or blocksize % dtype.itemsize \
                    or nbytes != items * dtype.itemsize:
                raise ValueError('the Blosc header is not one of a chunk of this array')
            nblocks = -(-nbytes // blocksize)
            offsets = np.frombuffer(get(path, 16, 16 + 4 * nblocks), dtype='<i4').astype(np.int64)
            if offsets.size != nblocks or offsets[0] != 16 + 4 * nblocks or np.any(np.diff(offsets) <= 0) \
                    or offsets[-1] >= cbytes:
                raise ValueError('the Blosc block offsets are not as expected')
            return {'ver': ver, 'verlz': verlz, 'flags': flags, 'typesize': typesize, 'nbytes': nbytes,
                    'blocksize': blocksize, 'cbytes': cbytes, 'offsets': offsets, 'nblocks': nblocks}

        def gather(meta, name, selection):
            """The values of an array at the positions given per dimension, broadcast together.

            The chunks are planned and their blocks requested concurrently; a chunk whose block route fails is read whole, with a warning.

            Parameters
            ----------
            meta : dict - The metadata of the store.
            name : str - The array name.
            selection : tuple - One index array per dimension of the array, broadcastable together.

            Examples
            --------
            >>> gather(meta, 'temperature', (1019984, np.arange(37)[:, None], np.array([[400]]), np.array([[600]])))
            """
            zarray = meta['arrays'][name]
            url = meta['store'] + name
            dtype = np.dtype(zarray['dtype'])
            chunks = zarray['chunks']
            items = int(np.prod(chunks))
            shape = np.broadcast_shapes(*[np.shape(s) for s in selection])
            idx = [np.broadcast_to(np.asarray(s, dtype=np.int64), shape).ravel() for s in selection]
            cidx = np.stack([i // c for i, c in zip(idx, chunks)], axis=1)
            flat = np.ravel_multi_index(tuple(i % c for i, c in zip(idx, chunks)), tuple(chunks))
            keys, inverse = np.unique(cidx, axis=0, return_inverse=True)
            inverse = np.asarray(inverse).reshape(-1)
            names = ['.'.join(str(int(v)) for v in c) for c in keys]
            positions = [np.nonzero(inverse == k)[0] for k in range(len(keys))]
            out = np.empty(flat.size, dtype=dtype)
            reason = {}

            def plan(key):
                """The header of one chunk, or the error that makes the block route fail for it.

                Parameters
                ----------
                key : str - The chunk key.

                Examples
                --------
                >>> plan('1019984.0.0.0')
                """
                try:
                    return header(f'{url}/{key}', dtype, items)
                except (ValueError, RuntimeError) as e:
                    return e

            plans = list(pool.map(plan, names))
            tasks = []
            for k, p in enumerate(plans):
                if isinstance(p, Exception):
                    reason[k] = p
                    continue
                per = p['blocksize'] // dtype.itemsize
                blocks = flat[positions[k]] // per
                for run in self._block_ranges(p['offsets'], p['cbytes'], np.unique(blocks)):
                    tasks.append((k, run))

            def read(task):
                """The bytes of one range of a chunk, or the error that makes the block route fail for the chunk.

                Parameters
                ----------
                task : tuple - (chunk index, (first block, block after the last, start byte, end byte)).

                Examples
                --------
                >>> read((0, (0, 1, 16, 120)))
                """
                k, (first, stop, start, end) = task
                try:
                    return get(f'{url}/{names[k]}', start, end)
                except (ValueError, RuntimeError) as e:
                    return e

            bodies = list(pool.map(read, tasks))
            for (k, (first, stop, start, end)), body in zip(tasks, bodies):
                if k in reason:
                    continue
                if isinstance(body, Exception):
                    reason[k] = body
                    continue
                p = plans[k]
                try:
                    flat_k = flat[positions[k]]
                    per = p['blocksize'] // dtype.itemsize
                    blocks_k = flat_k // per
                    for b in range(first, stop):
                        s = int(p['offsets'][b]) - start
                        e = (int(p['offsets'][b + 1]) if b + 1 < p['nblocks'] else int(p['cbytes'])) - start
                        bn = min(p['blocksize'], p['nbytes'] - b * p['blocksize'])
                        # the last block of a chunk is shorter: Blosc stores it unsplit, so its frame says so
                        flags_b, size_b = ((p['flags'] | 0x10, bn) if bn < p['blocksize']
                                            else (p['flags'], p['blocksize']))
                        frame = struct.pack('<BBBBiii', p['ver'], p['verlz'], flags_b, p['typesize'], bn, size_b,
                                            20 + e - s) + struct.pack('<i', 20) + body[s:e]
                        raw = numcodecs.blosc.decompress(frame)
                        if len(raw) != bn:
                            raise ValueError('a Blosc block decodes to another size')
                        sel = blocks_k == b
                        out[positions[k][sel]] = np.frombuffer(raw, dtype=dtype)[flat_k[sel] - b * per]
                except (ValueError, RuntimeError) as e:
                    reason[k] = e
            for k in sorted(reason):
                path = f'{url}/{names[k]}'
                print(f'WARNING: {path}: the block route failed ({reason[k]}); the whole chunk is read')
                raw = numcodecs.blosc.decompress(get(path))
                if len(raw) != items * dtype.itemsize:
                    raise ValueError(f'ERROR: {path} decodes to {len(raw)} bytes, expected {items * dtype.itemsize}')
                out[positions[k]] = np.frombuffer(raw, dtype=dtype)[flat[positions[k]]]
            return out.reshape(shape)

        def positions_of(corners):
            """The store rows and columns of the nodes of the tiles, tile by tile, row by row from the north edge.

            Parameters
            ----------
            corners : list of tuple - (lon0, lat0) of the tiles.

            Examples
            --------
            >>> positions_of([(-2, -27)])
            """
            rows, cols = [], []
            for lon0, lat0 in corners:
                for i in range(n):
                    for j in range(n):
                        rows.append(round((90.0 - (lat0 + step * i)) / step))
                        cols.append(round(((lon0 + step * j) % 360.0) / step))
            return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)

        def surface(name, corners):
            """The surface field of the tiles at one hour, as (tiles, 5, 5) float32.

            Parameters
            ----------
            name : str - The surface variable.
            corners : list of tuple - (lon0, lat0) of the tiles.

            Examples
            --------
            >>> surface('surface_pressure', [(-2, -27)])
            """
            rows, cols = positions_of(corners)
            lead = (np.int64(index),) if len(meta['arrays'][name]['shape']) == 3 else ()
            return gather(meta, name, lead + (rows, cols)).reshape(len(corners), n, n)

        rows, cols = positions_of(tiles)
        out = {t: {} for t in tiles}
        levels = np.arange(self.levels, dtype=np.int64)[:, None]
        for name in self.model_variables:
            values = gather(meta, name, (np.int64(index), levels, rows[None, :], cols[None, :]))
            for k, t in enumerate(tiles):
                out[t][name] = values[:, k * n * n:(k + 1) * n * n].reshape(self.levels, n, n)
        new = [t for t in tiles if t not in invariant]
        if new:
            for t, field in zip(new, surface('geopotential_at_surface', new)):
                invariant[t] = field
        for t in tiles:
            out[t]['geopotential_at_surface'] = invariant[t]
        surf = surface('surface_pressure', tiles)
        for k, t in enumerate(tiles):
            out[t]['surface_pressure'] = surf[k]
        return out

    def _tile_files(self, folder):
        """The tile files of a folder, as (epoch, lon0, lat0, path) sorted by epoch and tile.

        Parameters
        ----------
        folder : str - The tile folder, basedir/ERA5.

        Returns
        -------
        list of tuple - (pandas.Timestamp, int, int, str) per tile file; empty when the folder is missing.

        Examples
        --------
        >>> era5 = ERA5()
        >>> era5._tile_files('era5_tiles/ERA5')
        """
        import os
        import re
        import pandas as pd

        if not os.path.isdir(folder):
            return []
        pattern = re.compile(r'([NS])(\d{2})([EW])(\d{3})_(\d{8}T\d{6})\.nc')
        files = []
        for fname in sorted(os.listdir(folder)):
            m = pattern.fullmatch(fname)
            if m is None:
                continue
            lat0 = int(m.group(2)) * (1 if m.group(1) == 'N' else -1)
            lon0 = int(m.group(4)) * (1 if m.group(3) == 'E' else -1)
            files.append((pd.to_datetime(m.group(5), format='%Y%m%dT%H%M%S'), lon0, lat0, os.path.join(folder, fname)))
        return sorted(files, key=lambda f: (f[0], f[2], f[1]))

    def _tile_levels(self, folder):
        """The pressure levels (hPa) that the tile files of a folder hold.

        Parameters
        ----------
        folder : str - The tile folder, basedir/ERA5.

        Returns
        -------
        numpy.ndarray or None - The levels of the tile files, int32; None when the folder has no tile file.

        Raises
        ------
        ValueError: Naming the first tile file of another layout.

        Examples
        --------
        >>> era5 = ERA5()
        >>> era5._tile_levels('era5_tiles/ERA5')
        """
        import numpy as np
        from .utils_epochs import _read_structure

        levels = None
        for _, _, _, path in self._tile_files(folder):
            # the labels of the pressure variables in the variables attribute (utils_epochs), in hPa
            pressure = [labels for _, _, labels in _read_structure(path) if labels != ['surface']]
            held = np.array([int(p) for p in pressure[0]], dtype=np.int32) if pressure else np.zeros(0, dtype=np.int32)
            if held.size != self.levels:
                raise ValueError(f'ERROR: {path} holds {held.size} levels, not the {self.levels} pressure levels of the '
                                 f'store: remove it')
            if any(labels != pressure[0] for labels in pressure):
                raise ValueError(f'ERROR: {path} holds other pressure levels for its variables: remove it')
            if levels is None:
                levels = held.astype(np.int32)
            elif not np.array_equal(held, levels):
                raise ValueError(f'ERROR: {path} holds other pressure levels than the other tile files: remove it')
        return levels

    def download(self, basedir, scenes, geometry=None, retries=30, timeout_second=3, min_rate='100KB',
                 min_rate_window=60):
        """Download the ERA5 pressure levels and surface fields that the scenes need as 1-degree tiles, and index them in basedir/ERA5.vrt.

        Each scene takes its two bracketing hours; a tile file that exists is kept and listed as existing.

        Parameters
        ----------
        basedir : str - The download folder.
        scenes : pandas.DataFrame or object or list of str - A DataFrame with startTime and geometry columns, an object with such a DataFrame as .df, or Sentinel-1 scene names (the start time is taken from the name; geometry is then required).
        geometry : shapely geometry, optional - The area of the scenes when scenes is a list of names.
        retries : int, optional - The attempts of each request. Default is 30.
        timeout_second : float, optional - Seconds between the attempts. Default is 3.
        min_rate : str or float, optional - Bytes per second a transfer must keep. Default '100KB'.
        min_rate_window : float, optional - Seconds over which the rate is averaged. Default is 60.

        Returns
        -------
        pandas.DataFrame - Columns epoch, tile, path, status; status is 'new' or 'existing'.

        Raises
        ------
        ValueError: When the scenes argument is invalid, a tile file holds another number of levels or cannot be read, or an hour is outside the store; no file is written then.

        Examples
        --------
        >>> era5 = ERA5()
        >>> era5.download('era5_tiles', scenes)
        """
        import os
        import numpy as np
        from concurrent.futures import ThreadPoolExecutor
        from .utils_epochs import scene_items, store, tile_name
        from .utils_files import exists

        net = {'retries': retries, 'timeout_second': timeout_second, 'min_rate': min_rate,
               'min_rate_window': min_rate_window}
        items = scene_items(scenes, 'ERA5', geometry=geometry)
        base = os.path.abspath(basedir)
        folder = os.path.join(base, 'ERA5')
        levels = self._tile_levels(folder)
        rows = []
        for epoch, lon0, lat0 in items.itertuples(index=False):
            lon0, lat0 = int(lon0), int(lat0)
            path = os.path.join(folder, f'{tile_name(lon0, lat0)}_{epoch:%Y%m%dT%H%M%S}.nc')
            rows.append((epoch, lon0, lat0, 'existing' if exists(path) else 'new'))
        self.bytes_requested = 0
        new = [r for r in rows if r[3] == 'new']
        meta, index, values, invariant = None, {}, {}, {}
        if new:
            meta = self._store_meta(self.url + self.store, ('level',) + self.model_variables + self.surface_variables, net)
            store_levels = self._levels(meta, net)
            if levels is not None and not np.array_equal(levels, store_levels):
                raise ValueError(f'ERROR: the store levels differ from the levels of the tile files in {folder}')
            levels = store_levels
            # the index of every new hour before the first write: an hour outside the store raises with no file written
            index = {h: self._hour_index(meta, h.to_pydatetime(), net) for h in sorted({r[0] for r in new})}

        def fetch(epoch, lon0, lat0):
            """The values of one tile at one hour, as {variable: {level: 2-D array}}; the surface fields are labelled 'surface'.

            Parameters
            ----------
            epoch : pandas.Timestamp - The hour of the values, UTC.
            lon0, lat0 : int - The lower-left corner of the tile in whole degrees.

            Examples
            --------
            >>> fetch(pd.Timestamp('2016-05-11 08:00:00'), 150, -27)
            """
            if epoch not in values:
                values.clear()
                corners = [(r[1], r[2]) for r in new if r[0] == epoch]
                values[epoch] = self._read_hour(meta, index[epoch], corners, invariant, net, pool)
            out = values[epoch][(lon0, lat0)]
            served = {var: {str(int(p)): out[var][k] for k, p in enumerate(levels, 0)} for var in self.model_variables}
            served.update({var: {'surface': out[var]} for var in self.surface_variables})
            return served

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            # the fetches run in this process: the hour reads share this pool and the invariant fields
            return store(base, 'ERA5', items, fetch, n_jobs=None)
