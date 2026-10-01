# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://InSAR.dev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
"""
Sentinel-1 burst SLC measurement stored as compressed HDF5 through h5py, and the legacy GeoTIFF.

A downloaded burst arrives as an uncompressed complex int16 GeoTIFF. It is stored as `<burst>.nc` instead: the
(re, im) int16 pairs zigzag-encoded as uint16 -- small magnitudes of either sign become small unsigned values,
which is what makes speckle compress -- in 256 x 256 chunks compressed by Blosc2 LZ4 with byte shuffle. That is
about 2x smaller than the GeoTIFF (77.8 MB for a 155 MB burst) and a whole burst reads in about 0.1 s.

The file is NetCDF4-structured HDF5 with dimension scales, but its data needs the Blosc2 filter of hdf5plugin: it
reads through h5py with hdf5plugin imported, as everything here does, while netCDF-C, xarray or GDAL open it and
fail at the first read unless that filter is on their HDF5 plugin path.

Bursts are small by design, so a burst is converted and read whole in memory, in bands of rows so that only one
band of temporaries exists at a time: the conversion peaks at about 200 MB above the burst's own bytes. Bursts
downloaded before keep their `<burst>.tiff`, and every reader here takes either file.

`burst_xmls()` writes the annotation, noise and calibration XMLs of a burst, the same files for every source (ASF,
CDSE, a local SAFE).

The path of an IW burst, `path_number()`, follows from its ESA burst ID. `S1_PLATFORMS` and `platform_names()` read
the platform names of a catalog search.
"""
import os
import numpy as np

SLC_CHUNKS = (256, 256, 2)
SLC_ENCODING = 'zigzag'
# rows converted and verified at a time, one chunk of rows
BAND = SLC_CHUNKS[0]


def zigzag_encode(iq, out=None):
    """int16 -> uint16 zigzag: 0, -1, 1, -2, 2 ... -> 0, 1, 2, 3, 4 ...; a foreign byte order is converted first."""
    iq = np.asarray(iq)
    if iq.dtype != np.int16:
        iq = iq.astype(np.int16)
    if out is None:
        out = np.empty(iq.shape, np.uint16)
    np.left_shift(iq.view(np.uint16), 1, out=out)
    sign = np.right_shift(iq, 15)  # 0 or -1
    np.bitwise_xor(out, sign.view(np.uint16), out=out)
    return out


def zigzag_decode(z, out=None):
    """uint16 zigzag -> int16, the inverse of zigzag_encode()."""
    z = np.asarray(z, np.uint16)
    if out is None:
        out = np.empty(z.shape, np.int16)
    u = out.view(np.uint16)
    np.right_shift(z, 1, out=u)
    sign = z & np.uint16(1)
    np.negative(sign, out=sign)  # 0 or 0xFFFF by unsigned wrap-around
    np.bitwise_xor(u, sign, out=u)
    return out


def _to_complex64(source):
    """
    Stored uint16 zigzag (lines, samples, 2) -> complex64 (lines, samples), from an h5py dataset or an array.

    Decoded in bands of rows into the float32 result, so that the result and one band of temporaries are the only
    allocations: reading a burst costs its complex64 size, not several times that.
    """
    lines, samples = source.shape[:2]
    out = np.empty((lines, samples, 2), np.float32)
    for r in range(0, lines, BAND):
        out[r:r + BAND] = zigzag_decode(source[r:r + BAND])
    return out.view(np.complex64)[..., 0]


def _compression():
    import hdf5plugin
    return hdf5plugin.Blosc2(cname='lz4', clevel=5, filters=hdf5plugin.Blosc2.SHUFFLE)


def tiff_pairs(tiff_bytes):
    """
    The int16 (re, im) pairs of a burst GeoTIFF held in memory, and its data offset.

    A Sentinel-1 burst TIFF is uncompressed with contiguous strips, so the pairs are a view of the bytes, with no
    decode and no copy. Any other complex TIFF layout is decoded by tifffile.

    Returns
    -------
    tuple
        (int16 array (lines, samples, 2), data offset of the image in the TIFF).
    """
    import io
    from tifffile import TiffFile
    with TiffFile(io.BytesIO(tiff_bytes)) as tif:
        page = tif.pages[0]
        if page.ndim != 2 or page.dtype != np.complex64:
            raise ValueError(f'ERROR: burst TIFF is {page.dtype} {page.shape}, expected a 2D complex int16 image')
        lines, samples = (int(n) for n in page.shape)
        offsets, counts = page.dataoffsets, page.databytecounts
        data_offset = int(offsets[0])
        contiguous = (int(page.compression) == 1 and sum(counts) == lines * samples * 4
                      and all(offsets[i] + counts[i] == offsets[i + 1] for i in range(len(offsets) - 1)))
        if contiguous:
            iq = np.frombuffer(tiff_bytes, dtype=tif.byteorder + 'i2', count=lines * samples * 2,
                               offset=data_offset).reshape(lines, samples, 2)
        else:
            # complex int16 reads as complex64, whose float32 parts hold every int16 exactly
            c = page.asarray()
            iq = c.view(np.float32).reshape(lines, samples, 2).astype(np.int16)
    return iq, data_offset


# The data offset of pairs_tiff(), the TIFF form of stored (re, im) pairs, and the byteOffset of every stored burst
# annotation and the tiff_data_offset of every stored burst: the same for every source. The TIFF a download delivers
# is not stored, and its header length comes from the server's TIFF writer (22534 bytes for a 1509-line ASF or CDSE
# burst), not from the burst.
PAIRS_TIFF_OFFSET = 8


def pairs_tiff(iq):
    """
    A burst GeoTIFF held in memory from int16 (re, im) pairs (lines, samples, 2): classic little-endian TIFF,
    complex int16 (SampleFormat 5, 32 bits), uncompressed, one strip, the image at offset 8 (PAIRS_TIFF_OFFSET).
    """
    import struct
    iq = np.asarray(iq)
    if iq.ndim != 3 or iq.shape[2] != 2:
        raise ValueError(f'ERROR: expected (lines, samples, 2) int16 pairs, got {iq.shape}')
    lines, samples = (int(n) for n in iq.shape[:2])
    data = np.ascontiguousarray(iq, dtype='<i2').tobytes()
    data_offset = PAIRS_TIFF_OFFSET
    tags = [(256, 4, samples), (257, 4, lines), (258, 3, 32), (259, 3, 1), (262, 3, 1), (273, 4, data_offset),
            (277, 3, 1), (278, 4, lines), (279, 4, len(data)), (339, 3, 5)]
    ifd = struct.pack('<H', len(tags))
    for code, kind, value in tags:
        ifd += struct.pack('<HHII', code, kind, 1, value) if kind == 4 else struct.pack('<HHIHH', code, kind, 1, value, 0)
    ifd += struct.pack('<I', 0)
    return b'II*\x00' + struct.pack('<I', data_offset + len(data)) + data + ifd


def _slc_buffer(iq):
    """The NetCDF4 file content of int16 pairs as a BytesIO, written and verified band by band."""
    import io
    import h5py
    lines, samples = (int(n) for n in iq.shape[:2])
    chunks = tuple(min(c, n) for c, n in zip(SLC_CHUNKS, (lines, samples, 2)))
    buf = io.BytesIO()
    with h5py.File(buf, 'w') as f:
        dims = {}
        for name, n in (('azimuth', lines), ('range', samples), ('component', 2)):
            dims[name] = f.create_dataset(name, data=np.arange(n, dtype=np.int32), track_times=False)
            dims[name].make_scale(name)
        ds = f.create_dataset('slc', shape=(lines, samples, 2), dtype=np.uint16, chunks=chunks,
                              compression=_compression(), track_times=False)
        for i, name in enumerate(('azimuth', 'range', 'component')):
            ds.dims[i].attach_scale(dims[name])
        ds.attrs['encoding'] = SLC_ENCODING
        ds.attrs['description'] = 'complex int16 (re, im) pairs stored as zigzag uint16'
        # the byteOffset of the burst annotation: the data offset of the TIFF form of the pairs, which does not depend
        # on the TIFF the pairs came from (PAIRS_TIFF_OFFSET)
        ds.attrs['tiff_data_offset'] = PAIRS_TIFF_OFFSET
        for r in range(0, lines, BAND):
            ds[r:r + BAND] = zigzag_encode(iq[r:r + BAND])
    # a burst that is written is a burst that reads back: re-opened and compared band by band
    with h5py.File(buf, 'r') as f:
        ds = f['slc']
        if ds.shape != (lines, samples, 2) or ds.dtype != np.uint16 or ds.attrs.get('encoding') != SLC_ENCODING:
            raise ValueError(f'ERROR: burst NetCDF verification failed: {ds.shape} {ds.dtype}')
        for r in range(0, lines, BAND):
            if not np.array_equal(ds[r:r + BAND], zigzag_encode(iq[r:r + BAND])):
                raise ValueError('ERROR: burst NetCDF verification failed: decoded values differ from the TIFF')
    return buf


def slc_tiff_to_nc(tiff_bytes):
    """
    Convert a burst GeoTIFF held in memory into the NetCDF4 file content.

    Parameters
    ----------
    tiff_bytes : bytes
        The burst GeoTIFF (complex int16).

    Returns
    -------
    bytes
        The NetCDF4 file content, verified to decode to exactly the TIFF values.
    """
    iq, _ = tiff_pairs(tiff_bytes)
    return _slc_buffer(iq).getvalue()


def _write_buffer(buf, path):
    """Write a BytesIO to `path` through a temporary file, which is removed when the write fails
    (utils_files.write_atomic)."""
    from .utils_files import write_file
    write_file(path, buf.getbuffer())


def write_slc(tiff_bytes, path):
    """
    Store a burst GeoTIFF held in memory as `path` (.nc), atomically.

    Returns
    -------
    tuple
        (lines, samples) of the burst.
    """
    iq, _ = tiff_pairs(tiff_bytes)
    _write_buffer(_slc_buffer(iq), path)
    return slc_shape(path)


def measurement_path(measurement_dir, burst):
    """
    The measurement file of a burst: `<burst>.nc`, or the legacy `<burst>.tiff` when only that one exists.

    An empty file raises (utils_files.exists). When neither exists, the `.nc` path a download would create is
    returned.
    """
    from .utils_files import exists
    nc = os.path.join(measurement_dir, f'{burst}.nc')
    tiff = os.path.join(measurement_dir, f'{burst}.tiff')
    if exists(nc) or not exists(tiff):
        return nc
    return tiff


def is_nc(path):
    return os.path.splitext(path)[1].lower() == '.nc'


def slc_shape(path):
    """(lines, samples) of a burst measurement file."""
    if is_nc(path):
        import h5py
        with h5py.File(path, 'r') as f:
            return tuple(int(n) for n in f['slc'].shape[:2])
    from tifffile import TiffFile
    with TiffFile(path) as tif:
        return tuple(int(n) for n in tif.pages[0].shape[:2])


def tiff_data_offset(path):
    """The annotation byteOffset of a stored burst: the tiff_data_offset of its `.nc`, PAIRS_TIFF_OFFSET (a file written
    before keeps the data offset of the TIFF it was converted from), or the data offset of a legacy `.tiff`."""
    if is_nc(path):
        import h5py
        with h5py.File(path, 'r') as f:
            return int(f['slc'].attrs['tiff_data_offset'])
    from tifffile import TiffFile
    with TiffFile(path) as tif:
        return int(tif.pages[0].dataoffsets[0])


def read_slc(path):
    """
    Read a whole burst as complex64 (lines, samples).

    Parameters
    ----------
    path : str
        `<burst>.nc` or the legacy `<burst>.tiff`.
    """
    if is_nc(path):
        import h5py
        import hdf5plugin  # noqa: F401  registers the Blosc2 filter
        with h5py.File(path, 'r') as f:
            ds = f['slc']
            if ds.attrs.get('encoding') != SLC_ENCODING:
                raise ValueError(f'ERROR: {path} is not a burst SLC file (encoding {ds.attrs.get("encoding")!r})')
            return _to_complex64(ds)
    from tifffile import imread
    c = imread(path)
    return c if c.dtype == np.complex64 else c.astype(np.complex64)


class SlcReader:
    """
    Patches of a burst `.nc` file, answering the calls the alignment code makes on a rasterio dataset:
    `height`, `width`, `count` and `read(window=...)`, which returns complex64 (1, rows, cols). A window reaching
    past the burst edge is clipped to the burst, as rasterio does.
    """

    def __init__(self, path):
        import h5py
        import hdf5plugin  # noqa: F401  registers the Blosc2 filter
        self._file = h5py.File(path, 'r')
        self._ds = self._file['slc']
        if self._ds.attrs.get('encoding') != SLC_ENCODING:
            self._file.close()
            raise ValueError(f'ERROR: {path} is not a burst SLC file')
        self.height, self.width = (int(n) for n in self._ds.shape[:2])
        self.count = 1

    def read(self, indexes=None, window=None):
        if window is None:
            r0, c0, r1, c1 = 0, 0, self.height, self.width
        else:
            r0, c0 = int(window.row_off), int(window.col_off)
            r1, c1 = r0 + int(window.height), c0 + int(window.width)
        r0, c0 = max(r0, 0), max(c0, 0)
        r1, c1 = min(r1, self.height), min(c1, self.width)
        patch = _to_complex64(self._ds[r0:max(r1, r0), c0:max(c1, c0)])
        # rasterio: an integer band index gives 2D, no index or a list of them gives (bands, rows, cols)
        return patch if isinstance(indexes, (int, np.integer)) else patch[np.newaxis]

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_slc(path):
    """A patch reader for a burst: SlcReader for `.nc`, a rasterio dataset for the legacy `.tiff`."""
    if is_nc(path):
        return SlcReader(path)
    import rasterio
    return rasterio.open(path)


def _as_list(items):
    return items if isinstance(items, list) else [items]


def _filter_azimuth_time(items, start, stop, delta):
    """The items (one or a list) whose azimuthTime is within delta seconds of [start, stop]."""
    from datetime import datetime, timedelta
    low, high = start - timedelta(seconds=delta), stop + timedelta(seconds=delta)
    return [item for item in _as_list(items)
            if low <= datetime.strptime(item['azimuthTime'], '%Y-%m-%dT%H:%M:%S.%f') <= high]


def burst_xmls(annotation, noise, calibration, burst_index, byte_offset, noise_azimuth_line=None):
    """
    The annotation, noise and calibration XML files of one burst, as every source stores them: ASF.download (from
    the burst manifest), CDSE.download (from the burst product) and PyGMTSAR.safeload (from a SAFE).

    The entries of the burst are kept, with their lines counted from the first line of the burst, and the rest of
    the source XMLs is kept as it is, such as the attributes of the noise azimuth vector list.

    Parameters
    ----------
    annotation, noise, calibration : dict
        The <product>, <noise> and <calibration> elements of the source XMLs, parsed by xmltodict; they are changed
        in place. Their lines count from the first line of the source's burst list.
    burst_index : int
        The burst in the annotation burst list.
    byte_offset : int
        The byteOffset of the burst: PAIRS_TIFF_OFFSET, or tiff_data_offset() of a stored measurement.
    noise_azimuth_line : int, optional
        The first line of the burst in the lines of the noise azimuth vector, when that vector counts its lines from
        another first line than the rest of the source XMLs (CDSE). Default: that of the rest.

    Returns
    -------
    tuple of bytes
        The annotation, noise and calibration files (UTF-8).
    """
    import xmltodict
    from datetime import datetime, timedelta

    lines_per_burst = int(annotation['swathTiming']['linesPerBurst'])
    first_line = lines_per_burst * burst_index
    start_utc = _as_list(annotation['swathTiming']['burstList']['burst'])[burst_index]['azimuthTime']
    start = datetime.strptime(start_utc, '%Y-%m-%dT%H:%M:%S.%f')
    interval = float(annotation['imageAnnotation']['imageInformation']['azimuthTimeInterval'])
    stop = start + timedelta(seconds=(lines_per_burst - 1) * interval)
    stop_utc = stop.strftime('%Y-%m-%dT%H:%M:%S.%f')

    def header(element):
        element['startTime'] = start_utc
        element['stopTime'] = stop_utc
        element['imageNumber'] = '001'
        return element

    def vectors(items, delta=3):
        # the entries of the burst time, their lines counted from the burst
        items = _filter_azimuth_time(items, start, stop, delta)
        for item in items:
            item['line'] = str(int(item['line']) - first_line)
        return items

    product = {'adsHeader': header(annotation['adsHeader'])}
    if 'qualityInformation' in annotation:
        product['qualityInformation'] = {key: annotation['qualityInformation'][key]
                                         for key in ('productQualityIndex', 'qualityDataList')
                                         if key in annotation['qualityInformation']}
    if 'generalAnnotation' in annotation:
        product['generalAnnotation'] = annotation['generalAnnotation']
    image = annotation['imageAnnotation']
    image['imageInformation']['productFirstLineUtcTime'] = start_utc
    image['imageInformation']['productLastLineUtcTime'] = stop_utc
    image['imageInformation']['productComposition'] = 'Assembled'
    image['imageInformation']['sliceNumber'] = '0'
    image['imageInformation']['sliceList'] = {'@count': '0'}
    image['imageInformation']['numberOfLines'] = str(lines_per_burst)
    product['imageAnnotation'] = image
    if 'dopplerCentroid' in annotation:
        doppler = annotation['dopplerCentroid']
        items = _filter_azimuth_time(doppler['dcEstimateList']['dcEstimate'], start, stop, 3)
        doppler['dcEstimateList'] = {'@count': len(items), 'dcEstimate': items}
        product['dopplerCentroid'] = doppler
    if 'antennaPattern' in annotation:
        antenna = annotation['antennaPattern']
        items = _filter_azimuth_time(antenna['antennaPatternList']['antennaPattern'], start, stop, 3)
        antenna['antennaPatternList'] = {'@count': len(items), 'antennaPattern': items}
        product['antennaPattern'] = antenna
    timing = annotation['swathTiming']
    items = _filter_azimuth_time(timing['burstList']['burst'], start, start, 1)
    if len(items) != 1:
        raise ValueError(f'ERROR: {len(items)} bursts at {start_utc} in the annotation, expected 1.')
    items[0]['byteOffset'] = byte_offset
    timing['burstList'] = {'@count': len(items), 'burst': items}
    product['swathTiming'] = timing
    grid = annotation['geolocationGrid']
    items = vectors(grid['geolocationGridPointList']['geolocationGridPoint'], 1)
    grid['geolocationGridPointList'] = {'@count': len(items), 'geolocationGridPoint': items}
    product['geolocationGrid'] = grid
    for key in ('coordinateConversion', 'swathMerging'):
        if key in annotation:
            product[key] = annotation[key]

    noise_out = {'adsHeader': header(noise['adsHeader'])}
    for key in ('noiseVector', 'noiseRangeVector'):
        if f'{key}List' in noise:
            items = vectors(noise[f'{key}List'].get(key, []))
            noise_out[f'{key}List'] = {'@count': len(items), key: items}
    if 'noiseAzimuthVectorList' in noise:
        azimuth = noise['noiseAzimuthVectorList']
        vector = azimuth['noiseAzimuthVector']
        line0 = first_line if noise_azimuth_line is None else noise_azimuth_line
        text = vector['line']['#text'] if isinstance(vector['line'], dict) else vector['line']
        lines = [int(line) for line in text.split()]
        # the vector lines from the last one at or before the first line of the burst to the first one at or after
        # its last line (the first or last line of the vector when there is none)
        lower = ([line for line in lines if line <= line0] or [lines[0]])[-1]
        upper = ([line for line in lines if line >= line0 + lines_per_burst - 1] or [lines[-1]])[0]
        mask = [lower <= line <= upper for line in lines]
        lines = [line - line0 for line, keep in zip(lines, mask) if keep]
        vector['firstAzimuthLine'] = lower - line0
        vector['lastAzimuthLine'] = upper - line0
        vector['line'] = {'@count': len(lines), '#text': ' '.join(str(line) for line in lines)}
        lut = vector['noiseAzimuthLut']
        lut = (lut['#text'] if isinstance(lut, dict) else lut).split()
        lut = [value for value, keep in zip(lut, mask) if keep]
        vector['noiseAzimuthLut'] = {'@count': len(lut), '#text': ' '.join(lut)}
        noise_out['noiseAzimuthVectorList'] = azimuth

    calibration_out = {'adsHeader': header(calibration['adsHeader'])}
    if 'calibrationInformation' in calibration:
        calibration_out['calibrationInformation'] = calibration['calibrationInformation']
    if 'calibrationVectorList' in calibration:
        items = vectors(calibration['calibrationVectorList'].get('calibrationVector', []))
        calibration_out['calibrationVectorList'] = {'@count': len(items), 'calibrationVector': items}

    return tuple(xmltodict.unparse({root: content}, pretty=True, indent='  ').encode('utf-8')
                 for root, content in (('product', product), ('noise', noise_out), ('calibration', calibration_out)))


# The Sentinel-1 IW burst ID, ESA Sentinel-1 Level 1 Detailed Algorithm Definition (SEN-TN-52-7445, issue 2/4,
# section 9.25 "Burst ID", eqs. 9-89..9-91): the orbit period of the 12-day repeat cycle of 175 orbits, the preamble
# time and the IW burst cycle time, in seconds.
S1_ORBIT_PERIOD = 12 * 24 * 3600 / 175
S1_IW_PREAMBLE = 2.299849
S1_IW_BURST_CYCLE = 2.758273
# the IW burst IDs of one repeat cycle, 1 to this one
S1_IW_BURST_IDS = 1 + int((175 * S1_ORBIT_PERIOD - S1_IW_PREAMBLE) // S1_IW_BURST_CYCLE)


def path_number(burst_id):
    """Path (relative orbit, 1-175) of a Sentinel-1 IW burst, from its ESA burst ID.

    ESA numbers the IW bursts of the repeat cycle from the ascending node crossing (ANX) of path 1 on,
    burst ID = 1 + floor((dt - Tpre) / Tbeam) for a burst dt seconds after it. A burst ID b so starts at
    dt = Tpre + (b - 1) * Tbeam, on path floor(dt / Torb) + 1. The path is a property of the burst: the same on
    every date and for every satellite (S1A, S1B, S1C, S1D), and the path of the ASF fullBurstID. It is not always
    the relative orbit of the product: a product that starts before an ANX carries the previous path, while its
    bursts past the ANX belong to the next one.

    Parameters
    ----------
    burst_id : int or str
        The ESA burst ID, e.g. 262885 or '038599' as in the burst name 'S1_038599_IW2_...'.

    Returns
    -------
    int
        The path number.

    Raises
    ------
    ValueError
        For a burst ID outside the IW burst IDs of the repeat cycle.
    """
    import math
    burst_id = int(burst_id)
    if not 1 <= burst_id <= S1_IW_BURST_IDS:
        raise ValueError(f'Sentinel-1 IW burst ID {burst_id} is outside 1..{S1_IW_BURST_IDS}.')
    return math.floor((S1_IW_PREAMBLE + (burst_id - 1) * S1_IW_BURST_CYCLE) / S1_ORBIT_PERIOD) + 1



# the Sentinel-1 platform names of the ASF and CDSE catalog searches: 'SENTINEL-1' is every satellite
S1_PLATFORMS = ('SENTINEL-1', 'SENTINEL-1A', 'SENTINEL-1B', 'SENTINEL-1C', 'SENTINEL-1D')


def platform_names(platform):
    """The platform names of one name, a comma-separated string or a list, e.g. 'SENTINEL-1C,SENTINEL-1D' or
    ['SENTINEL-1C', 'SENTINEL-1D'], without surrounding spaces."""
    return [name.strip() for name in (platform.split(',') if isinstance(platform, str) else platform)]


def polarizations(polarization, example="'VV' or ['VV', 'VH']"):
    """
    The polarizations requested from a downloader (ASF, CDSE) as a list of upper-case names, or None for None.

    Parameters
    ----------
    polarization : None, str or list
        The polarization argument of the download.
    example : str, optional
        The valid values named in the error of an empty list. Default: those of Sentinel-1.

    Raises
    ------
    ValueError
        For an empty list, which selects nothing to download.
    """
    if polarization is None:
        return None
    names = [polarization.upper()] if isinstance(polarization, str) else [p.upper() for p in polarization]
    if not names:
        raise ValueError(f'ERROR: polarization={polarization!r} selects no polarization. Use polarization=None or '
                         f'name them, e.g. {example}.')
    return names
