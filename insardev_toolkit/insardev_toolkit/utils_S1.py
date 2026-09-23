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


def pairs_tiff(iq):
    """
    A burst GeoTIFF held in memory from int16 (re, im) pairs (lines, samples, 2): classic little-endian TIFF,
    complex int16 (SampleFormat 5, 32 bits), uncompressed, one strip, the image at offset 8.
    """
    import struct
    iq = np.asarray(iq)
    if iq.ndim != 3 or iq.shape[2] != 2:
        raise ValueError(f'ERROR: expected (lines, samples, 2) int16 pairs, got {iq.shape}')
    lines, samples = (int(n) for n in iq.shape[:2])
    data = np.ascontiguousarray(iq, dtype='<i2').tobytes()
    data_offset = 8
    tags = [(256, 4, samples), (257, 4, lines), (258, 3, 32), (259, 3, 1), (262, 3, 1), (273, 4, data_offset),
            (277, 3, 1), (278, 4, lines), (279, 4, len(data)), (339, 3, 5)]
    ifd = struct.pack('<H', len(tags))
    for code, kind, value in tags:
        ifd += struct.pack('<HHII', code, kind, 1, value) if kind == 4 else struct.pack('<HHIHH', code, kind, 1, value, 0)
    ifd += struct.pack('<I', 0)
    return b'II*\x00' + struct.pack('<I', data_offset + len(data)) + data + ifd


def _slc_buffer(iq, data_offset):
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
        # the burst annotation records the data offset of the measurement TIFF (byteOffset)
        ds.attrs['tiff_data_offset'] = int(data_offset)
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
    iq, data_offset = tiff_pairs(tiff_bytes)
    return _slc_buffer(iq, data_offset).getvalue()


def _write_buffer(buf, path):
    """Write a BytesIO to `path` through a temporary file, which is removed when the write fails."""
    tmp = path + '.tmp'
    try:
        with open(tmp, 'wb') as f:
            f.write(buf.getbuffer())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_slc(tiff_bytes, path):
    """
    Store a burst GeoTIFF held in memory as `path` (.nc), atomically.

    Returns
    -------
    tuple
        (lines, samples) of the burst.
    """
    iq, data_offset = tiff_pairs(tiff_bytes)
    _write_buffer(_slc_buffer(iq, data_offset), path)
    return slc_shape(path)


def measurement_path(measurement_dir, burst):
    """
    The measurement file of a burst: `<burst>.nc`, or the legacy `<burst>.tiff` when only that one exists.

    An empty `.nc`, left by an interrupted copy, counts as absent. When neither exists, the `.nc` path a download
    would create is returned.
    """
    nc = os.path.join(measurement_dir, f'{burst}.nc')
    tiff = os.path.join(measurement_dir, f'{burst}.tiff')
    if (os.path.exists(nc) and os.path.getsize(nc) > 0) or not os.path.exists(tiff):
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
    """Data offset of the burst's measurement TIFF, the annotation byteOffset."""
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
