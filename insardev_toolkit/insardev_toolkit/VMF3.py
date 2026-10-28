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


class VMF3(datagrid, progressbar_joblib, AtmosphereModel):
    """Download the VMF3 grids of TU Wien as 1-degree tiles, one NetCDF4 file per tile and 6-hour epoch, with a VRT index.

    Only the lines of the requested nodes are fetched, by byte range; a grid file whose layout check fails is fetched whole, with a warning.
    """

    url = 'https://vmf.geo.tuwien.ac.at/trop_products/GRID/1x1/VMF3/VMF3_{product}/{year}/VMF3_{yyyymmdd}.H{hh}'
    orography_url = 'https://vmf.geo.tuwien.ac.at/station_coord_files/orography_ell_1x1'
    products = ('OP', 'EI', 'FC')
    # the grid: rows of latitude, nodes per row, lines per file
    rows = 180
    width = 360
    lines = 64800
    keys = ('ah', 'aw', 'zhd', 'zwd')
    # bytes of the answers to the GET requests of the last download
    bytes_requested = 0

    def _phase(self, folder, times, wavelength, factor):
        """The terms of the tropospheric model of a folder per tile, in phase units, lazy.

        Parameters
        ----------
        folder : str - The folder of the downloaded model fields.
        times : list of str or pandas.Timestamp - The acquisition times, UTC.
        wavelength : float - The radar wavelength, metres.
        factor : float - Not used by VMF3.
        """
        from . import utils_tropo
        return utils_tropo.zenith_tiles(folder, times, wavelength)

    @classmethod
    def _epoch_url(cls, product, epoch):
        """The URL of the grid file of one epoch of a product.

        Parameters
        ----------
        product : str - 'OP' (operational), 'EI' (ERA-Interim based) or 'FC' (forecast).
        epoch : pandas.Timestamp or datetime.datetime - The epoch in UTC, on the 6-hour grid: 00, 06, 12 or 18 UTC.

        Returns
        -------
        str - The URL of the file of the epoch.

        Examples
        --------
        >>> VMF3._epoch_url('OP', pd.Timestamp('2016-05-11 06:00'))
        'https://vmf.geo.tuwien.ac.at/trop_products/GRID/1x1/VMF3/VMF3_OP/2016/VMF3_20160511.H06'
        """
        if product not in cls.products:
            raise ValueError(f'ERROR: VMF3 product must be one of {", ".join(cls.products)}, got {product!r}')
        return cls.url.format(product=product, year=f'{epoch:%Y}', yyyymmdd=f'{epoch:%Y%m%d}', hh=f'{epoch:%H}')

    @classmethod
    def _node_ranges(cls, indices, header, row, total):
        """The byte ranges of the lines of the given nodes of a fixed-width file; neighbouring lines form one range.

        Parameters
        ----------
        indices : list of int - The node indices, row * 360 + longitude index.
        header : int - The bytes before the first node line.
        row : int - The bytes of one node line, its line end included.
        total : int - The length of the file in bytes.

        Returns
        -------
        list of tuple - (start, stop, [indices]) in file order.

        Examples
        --------
        >>> VMF3._node_ranges([0, 1, 360], 350, 53, 3434750)
        [(350, 456, [0, 1]), (19430, 19483, [360])]
        """
        ordered = sorted(set(int(i) for i in indices))
        if not ordered:
            raise ValueError('ERROR: no nodes to read')
        if ordered[0] < 0 or ordered[-1] >= cls.lines:
            raise ValueError(f'ERROR: node index {ordered[0]} or {ordered[-1]} is outside the {cls.lines} lines')
        groups = [[ordered[0]]]
        for i in ordered[1:]:
            if i - groups[-1][-1] == 1:
                groups[-1].append(i)
            else:
                groups.append([i])
        return [(header + g[0] * row, min(header + (g[-1] + 1) * row, total), g) for g in groups]

    def _read_nodes(self, url, nodes, retries=30, timeout_second=3, min_rate='100KB', min_rate_window=60):
        """The values of the grid file nodes: ah, aw, zhd and zwd of each node, read by byte ranges.

        Parameters
        ----------
        url : str - The URL of the grid file of one epoch.
        nodes : list of tuple - (latitude, longitude) of the nodes, at grid node centres (x.5 values), degrees.
        retries : int, optional - The attempts of each request. Default is 30.
        timeout_second : float, optional - Seconds between the attempts. Default is 3.
        min_rate : str or float, optional - Bytes per second a transfer must keep. Default '100KB'.
        min_rate_window : float, optional - Seconds over which the rate is averaged. Default is 60.

        Returns
        -------
        dict - {(latitude, longitude): {'ah', 'aw', 'zhd', 'zwd'}}, float64 values.

        Examples
        --------
        >>> v = VMF3()
        >>> v._read_nodes(VMF3._epoch_url('OP', pd.Timestamp('2016-05-11 06:00')), [(-26.5, 150.5)])
        """
        import math
        import requests
        from .HTTP import send, fetch, raise_for_status, NotFound

        def part(start, stop):
            """The bytes [start, stop) of the grid file, read by one range request.

            Parameters
            ----------
            start, stop : int - The first byte and the end (exclusive) of the range.

            Examples
            --------
            >>> part(0, 1024)
            """
            body = fetch(url, headers={'Range': f'bytes={start}-{stop - 1}'}, retries=retries,
                         timeout_second=timeout_second, min_rate=min_rate, min_rate_window=min_rate_window)
            if len(body) != stop - start:
                raise IOError(f'ERROR: {url}: bytes {start}-{stop - 1} came as {len(body)} bytes')
            self.bytes_requested += len(body)
            return body

        index = {}
        for lat, lon in nodes:
            r, c = round(89.5 - lat), round(lon - 0.5) % self.width
            if abs(89.5 - lat - r) > 1e-9 or abs(lon - 0.5 - round(lon - 0.5)) > 1e-9 or not 0 <= r < self.rows:
                raise ValueError(f'ERROR: {lat}, {lon} is not a node of the VMF3 grid')
            index[(lat, lon)] = r * self.width + c

        head = send(requests.head, url, allow_redirects=True, timeout=(10, 30), check=False)
        if head.status_code in (404, 410):
            raise NotFound(f'{url}: HTTP {head.status_code}')
        raise_for_status(head, url)
        length = head.headers.get('Content-Length')
        if length is None:
            raise IOError(f'ERROR: {url}: no Content-Length in the answer')
        length = int(length)
        # header: the lines that start with '!', then the first node line gives the row length
        first = part(0, min(1024, length))
        pos = 0
        while first[pos:pos + 1] == b'!' and first.find(b'\n', pos) >= 0:
            pos = first.find(b'\n', pos) + 1
        end = first.find(b'\n', pos)
        row = end + 1 - pos if end >= 0 else 0
        ok = row > 0 and pos + self.lines * row == length and first[pos:end].split()[:2] == [b'89.5', b'0.5']
        if ok:
            ok = part(length - row, length).split()[:2] == [b'-89.5', b'359.5']
        want = sorted(set(index.values()))
        if not ok:
            print(f'WARNING: {url}: the layout check failed (header and {self.lines} rows do not add up to the '
                  f'Content-Length, or the row order differs); the whole file is fetched')
            data = fetch(url, retries=retries, timeout_second=timeout_second, min_rate=min_rate,
                         min_rate_window=min_rate_window)
            self.bytes_requested += len(data)
            text = data.split(b'\n')
            head_lines = 0
            while head_lines < len(text) and text[head_lines][:1] == b'!':
                head_lines += 1
            line_of = {i: text[head_lines + i] for i in want}
        else:
            line_of = {}
            for start, stop, group in self._node_ranges(want, pos, row, length):
                block = part(start, stop).split(b'\n')
                for i in group:
                    line_of[i] = block[i - group[0]]
                    if len(line_of[i]) + 1 != row:
                        raise ValueError(f'ERROR: {url}: node line {i} is {len(line_of[i]) + 1} bytes, expected {row}')
        out = {}
        for (lat, lon), i in index.items():
            line = line_of[i].decode('ascii', 'replace')
            tok = line.split()
            if len(tok) != 6 or float(tok[0]) != lat or float(tok[1]) != lon % 360.0:
                raise ValueError(f'ERROR: {url}: the line of node {lat}, {lon} is not that node: {line.strip()[:60]!r}')
            try:
                values = [float(v) for v in tok[2:6]]
            except ValueError:
                raise ValueError(f'ERROR: {url}: the values of node {lat}, {lon} are not numbers') from None
            if not all(math.isfinite(v) for v in values):
                raise ValueError(f'ERROR: {url}: the values of node {lat}, {lon} are not finite')
            out[(lat, lon)] = dict(zip(self.keys, values))
        return out

    def _orography(self, nodes, retries=30, timeout_second=3, min_rate='100KB', min_rate_window=60):
        """The node heights (metres, ellipsoidal) of the orography file, read by byte ranges of its fixed-width lines.

        Parameters
        ----------
        nodes : list of tuple - (latitude, longitude) of the nodes, at grid node centres (x.5 values), degrees.
        retries : int, optional - The attempts of each request. Default is 30.
        timeout_second : float, optional - Seconds between the attempts. Default is 3.
        min_rate : str or float, optional - Bytes per second a transfer must keep. Default '100KB'.
        min_rate_window : float, optional - Seconds over which the rate is averaged. Default is 60.

        Returns
        -------
        dict - {(latitude, longitude): height}, float64 metres.

        Examples
        --------
        >>> v = VMF3()
        >>> v._orography([(-26.5, 150.5)])
        """
        import math
        import requests
        from .HTTP import send, fetch, raise_for_status, NotFound

        def part(start, stop):
            """The bytes [start, stop) of the orography file, read by one range request.

            Parameters
            ----------
            start, stop : int - The first byte and the end (exclusive) of the range.

            Examples
            --------
            >>> part(0, 64)
            """
            body = fetch(self.orography_url, headers={'Range': f'bytes={start}-{stop - 1}'}, retries=retries,
                         timeout_second=timeout_second, min_rate=min_rate, min_rate_window=min_rate_window)
            if len(body) != stop - start:
                raise IOError(f'ERROR: {self.orography_url}: bytes {start}-{stop - 1} came as {len(body)} bytes')
            self.bytes_requested += len(body)
            return body

        index = {}
        for lat, lon in nodes:
            r, c = round(89.5 - lat), round(lon - 0.5) % self.width
            if abs(89.5 - lat - r) > 1e-9 or abs(lon - 0.5 - round(lon - 0.5)) > 1e-9 or not 0 <= r < self.rows:
                raise ValueError(f'ERROR: {lat}, {lon} is not a node of the VMF3 grid')
            index[(lat, lon)] = r * self.width + c

        head = send(requests.head, self.orography_url, allow_redirects=True, timeout=(10, 30), check=False)
        if head.status_code in (404, 410):
            raise NotFound(f'{self.orography_url}: HTTP {head.status_code}')
        raise_for_status(head, self.orography_url)
        length = head.headers.get('Content-Length')
        if length is None:
            raise IOError(f'ERROR: {self.orography_url}: no Content-Length in the answer')
        length = int(length)
        first = part(0, min(64, length))
        width = first.find(b'\n') + 1
        ok = width > 0 and (self.lines - 1) * width < length <= self.lines * width
        want = sorted(set(index.values()))
        if not ok:
            print(f'WARNING: {self.orography_url}: the line width check failed ({self.lines} lines of {width} bytes do '
                  f'not add up to the Content-Length {length}); the whole file is fetched')
            data = fetch(self.orography_url, retries=retries, timeout_second=timeout_second, min_rate=min_rate,
                         min_rate_window=min_rate_window)
            self.bytes_requested += len(data)
            text = data.split(b'\n')
            line_of = {i: text[i] for i in want}
        else:
            line_of = {}
            for start, stop, group in self._node_ranges(want, 0, width, length):
                block = part(start, stop).split(b'\n')
                for i in group:
                    line_of[i] = block[i - group[0]]
                    last = i == self.lines - 1
                    if len(line_of[i]) + 1 != width and not (last and len(line_of[i]) < width):
                        raise ValueError(f'ERROR: {self.orography_url}: line {i + 1} is {len(line_of[i]) + 1} bytes, '
                                         f'expected {width}')
        out = {}
        for (lat, lon), i in index.items():
            try:
                value = float(line_of[i].decode('ascii').strip())
            except (UnicodeDecodeError, ValueError):
                raise ValueError(f'ERROR: {self.orography_url}: the height of node {lat}, {lon} is not a number') \
                    from None
            if not math.isfinite(value):
                raise ValueError(f'ERROR: {self.orography_url}: the height of node {lat}, {lon} is not finite')
            out[(lat, lon)] = value
        return out

    def download(self, basedir, scenes, product='OP', geometry=None, n_jobs=8, joblib_backend='loky', retries=30,
                 timeout_second=3, min_rate='100KB', min_rate_window=60):
        """Download the VMF3 grids that the scenes need, cut to their 1-degree tiles, and index them in basedir/VMF3_<product>.vrt.

        Each scene takes its two bracketing 6-hour epochs, and a tile file that exists is kept.

        Parameters
        ----------
        basedir : str - The download folder.
        scenes : pandas.DataFrame or object or list of str - A DataFrame with startTime and geometry columns, an object with such a DataFrame as .df, or Sentinel-1 scene names (the start time is taken from the name; geometry is then required).
        product : str, optional - 'OP', 'EI' or 'FC'. Default 'OP'.
        geometry : shapely geometry, optional - The area of the scenes when scenes is a list of names.
        n_jobs : int, optional - The number of concurrent download jobs, at most two per CPU. Default is 8.
        joblib_backend : str, optional - The joblib backend. Default is 'loky'.
        retries : int, optional - The attempts of each request. Default is 30.
        timeout_second : float, optional - Seconds between the attempts. Default is 3.
        min_rate : str or float, optional - Bytes per second a transfer must keep. Default '100KB'.
        min_rate_window : float, optional - Seconds over which the rate is averaged. Default is 60.

        Returns
        -------
        pandas.DataFrame - Columns epoch, tile, path, status; status is 'new' or 'existing'.

        Examples
        --------
        >>> v = VMF3()
        >>> v.download('VMF3', scenes, product='OP')  # tiles in VMF3/VMF3_OP/, index VMF3/VMF3_OP.vrt
        """
        import re
        import pandas as pd
        import numpy as np
        from .HTTP import NotFound
        from .utils_epochs import bracket, tile_cells, store

        if product not in self.products:
            raise ValueError(f'ERROR: VMF3 product must be one of {", ".join(self.products)}, got {product!r}')
        if isinstance(scenes, pd.DataFrame) or hasattr(scenes, 'df'):
            df = scenes if isinstance(scenes, pd.DataFrame) else scenes.df
            if not {'startTime', 'geometry'} <= set(df.columns):
                raise ValueError('ERROR: the scenes need startTime and geometry columns')
            pairs = list(zip(df['startTime'], df['geometry']))
        elif isinstance(scenes, (list, tuple)) and scenes and all(isinstance(s, str) for s in scenes):
            if geometry is None:
                raise ValueError('ERROR: scene names need geometry= for the area of the scenes')
            pairs = []
            for name in scenes:
                m = re.search(r'(\d{8}T\d{6})', name)
                if m is None:
                    raise ValueError(f'ERROR: no start time in the scene name {name}')
                pairs.append((pd.to_datetime(m.group(1), format='%Y%m%dT%H%M%S'), geometry))
        else:
            raise ValueError('ERROR: scenes must be a DataFrame with startTime and geometry, an object with .df, '
                             'or a list of scene names')
        if not pairs:
            raise ValueError('ERROR: no scenes')
        rows = set()
        for start, geom in pairs:
            if getattr(geom, 'bounds', None) is None:
                raise ValueError('ERROR: a scene has no geometry')
            west, south, east, north = geom.bounds
            tiles = tile_cells(west, south, east, north, origin=0.5)
            for epoch in bracket(start, 6):
                rows.update((epoch, lon0, lat0) for lon0, lat0 in tiles)
        items = pd.DataFrame(sorted(rows), columns=['epoch', 'lon', 'lat'])

        def fetch(epoch, lon0, lat0):
            """The values of one tile at one epoch: ah, aw, zhd and zwd of its 2 x 2 nodes, and the node heights as the variable orography.

            Parameters
            ----------
            epoch : pandas.Timestamp - The epoch on the 6-hour grid.
            lon0, lat0 : float - The lower-left corner of the tile, at node centres (x.5 values), degrees.

            Examples
            --------
            >>> fetch(pd.Timestamp('2016-05-11 06:00'), 150.5, -27.5)
            """
            nodes = [(la, lo) for la in (lat0, lat0 + 1) for lo in (lon0, lon0 + 1)]
            url = self._epoch_url(product, epoch)
            try:
                grid = self._read_nodes(url, nodes, retries, timeout_second, min_rate, min_rate_window)
            except NotFound:
                raise ValueError(f'ERROR: no VMF3 {product} grid for epoch {epoch:%Y-%m-%d %H:%M:%S}: {url}') from None
            height = self._orography(nodes, retries, timeout_second, min_rate, min_rate_window)
            served = {k: {'surface': np.array([[grid[(la, lo)][k] for lo in (lon0, lon0 + 1)]
                                                for la in (lat0, lat0 + 1)], dtype=np.float64)}
                      for k in self.keys}
            served['orography'] = {'surface': np.array([[height[(la, lo)] for lo in (lon0, lon0 + 1)]
                                                        for la in (lat0, lat0 + 1)], dtype=np.float64)}
            return served

        self.bytes_requested = 0
        return store(basedir, f'VMF3_{product}', items, fetch, n_jobs=n_jobs, joblib_backend=joblib_backend)
