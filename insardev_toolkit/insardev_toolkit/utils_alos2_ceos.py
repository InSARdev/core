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
ALOS-2 CEOS IMG files (L1.1 SLC): the header of the image file and the cropped image file.

An IMG file is a 720-byte file descriptor followed by one record per image line: a 544-byte prefix and the pixels
as big-endian float32 I, Q pairs (record length 544 + 8 * pixels). The descriptor is record 1, so line 1 is record 2.
A cropped IMG file is the descriptor with its two line counts rewritten, followed by the selected line records with
their record sequence numbers renumbered from 2 (the image line numbers keep the source lines). The downloader
reads the descriptor and the line records of one subswath from the granule ZIP by byte ranges and writes them here.

Byte positions below are 1-based, as in the CEOS format description.
"""
import os
from datetime import date, timedelta

import numpy as np

DESCRIPTOR = 720
PREFIX = 544


def _int32(data, start):
    """
    Read a big-endian int32 field.

    Parameters
    ----------
    data : bytes
        The record bytes.
    start : int
        The 1-based byte position of the field.

    Returns
    -------
    int
        The value of the field.

    Examples
    --------
    >>> _int32(head, 9)
    """
    return int(np.frombuffer(data, dtype='>i4', count=1, offset=start - 1)[0])


def _ascii_int(data, start, stop):
    """
    Read an ASCII integer field.

    Parameters
    ----------
    data : bytes
        The record bytes.
    start : int
        The 1-based byte position of the first character.
    stop : int
        The 1-based byte position of the last character.

    Returns
    -------
    int
        The value of the field.

    Raises
    ------
    ValueError
        If the field is not an integer.

    Examples
    --------
    >>> _ascii_int(head, 181, 186)
    """
    return int(bytes(data[start - 1:stop]).decode('ascii'))


def image_info(head):
    """
    Read the image size, the timing and the near range of an ALOS-2 CEOS IMG file from its first bytes.

    Parameters
    ----------
    head : bytes
        The first 720 + 544 bytes of the IMG file: the file descriptor and the prefix of the first line record.

    Returns
    -------
    dict
        'lines' (number of line records), 'record_length' (bytes per line record), 'pixels' (per line),
        'time' (seconds of day of line 1), 'prf' (Hz), 'near_range' (slant range of the first pixel, m) and
        'date' ('YYYY-MM-DD' of line 1).

    Raises
    ------
    ValueError
        If the bytes are not the head of a consistent IMG file.

    Examples
    --------
    >>> image_info(open('IMG-HH-ALOS2...-F1', 'rb').read(1264))
    """
    if len(head) < DESCRIPTOR + PREFIX:
        raise ValueError(f'IMG head has {len(head)} bytes, expected {DESCRIPTOR + PREFIX}')
    descriptor_length = _int32(head, 9)
    if descriptor_length != DESCRIPTOR:
        raise ValueError(f'IMG descriptor length {descriptor_length}, expected {DESCRIPTOR}')
    lines = _ascii_int(head, 181, 186)
    lines_per_set = _ascii_int(head, 237, 244)
    if lines != lines_per_set:
        raise ValueError(f'IMG line counts differ: {lines} and {lines_per_set}')
    record_length = _ascii_int(head, 187, 192)
    prefix = bytes(head[DESCRIPTOR:DESCRIPTOR + PREFIX])
    pixels = _int32(prefix, 25)
    if record_length != PREFIX + 8 * pixels:
        raise ValueError(f'IMG record length {record_length} does not match {pixels} pixels')
    if _int32(prefix, 9) != record_length:
        raise ValueError(f'IMG first record length {_int32(prefix, 9)}, expected {record_length}')
    if _int32(prefix, 1) != 2:
        raise ValueError(f'IMG first record sequence number {_int32(prefix, 1)}, expected 2')
    day = date(_int32(prefix, 37), 1, 1) + timedelta(days=_int32(prefix, 41) - 1)
    return {'lines': lines,
            'record_length': record_length,
            'pixels': pixels,
            'time': _int32(prefix, 45) / 1000,
            'prf': _int32(prefix, 57) / 1000,
            'near_range': _int32(prefix, 117),
            'date': day.isoformat()}


def prefix_lonlat(prefix):
    """
    Read JAXA's longitude and latitude of the first, mid and last pixel of a line from its record prefix.

    Parameters
    ----------
    prefix : bytes
        The first 544 bytes of a line record (or more).

    Returns
    -------
    tuple of numpy.ndarray
        (lon, lat): float64 arrays of 3 values, degrees on the WGS84 ellipsoid at height 0, of the first, mid and last
        pixel: the big-endian int32 fields at bytes 205, 209, 213 (longitude) and 193, 197, 201 (latitude), in 1e-6
        degrees.

    Raises
    ------
    ValueError
        If the prefix is shorter than 544 bytes.

    Examples
    --------
    >>> prefix_lonlat(open('IMG-HH-ALOS2...-F1', 'rb').read(1264)[720:])
    """
    if len(prefix) < PREFIX:
        raise ValueError(f'IMG line prefix has {len(prefix)} bytes, expected {PREFIX}')
    lat = np.array([_int32(prefix, b) for b in (193, 197, 201)], dtype=np.float64) * 1e-6
    lon = np.array([_int32(prefix, b) for b in (205, 209, 213)], dtype=np.float64) * 1e-6
    return lon, lat


def crop_header(descriptor, lines):
    """
    Rewrite the line counts of an IMG file descriptor for a crop of `lines` line records.

    Parameters
    ----------
    descriptor : bytes
        The 720-byte file descriptor of the source IMG file.
    lines : int
        The number of line records of the crop.

    Returns
    -------
    bytes
        The descriptor with bytes 181-186 set to `lines` (I6) and bytes 237-244 set to `lines` (I8); every other byte
        is unchanged.

    Raises
    ------
    ValueError
        If the descriptor is not 720 bytes or `lines` is not in 1..999999.

    Examples
    --------
    >>> crop_header(descriptor, 114)
    """
    if len(descriptor) != DESCRIPTOR:
        raise ValueError(f'IMG descriptor has {len(descriptor)} bytes, expected {DESCRIPTOR}')
    if lines < 1 or lines > 999999:
        raise ValueError(f'IMG crop line count {lines} not in 1..999999')
    header = bytearray(descriptor)
    header[180:186] = f'{lines:6d}'.encode('ascii')
    header[236:244] = f'{lines:8d}'.encode('ascii')
    return bytes(header)


def renumber(block, first, record_length):
    """
    Renumber the record sequence numbers of a block of IMG line records for their place in a crop.

    Parameters
    ----------
    block : bytes
        A whole number of line records.
    first : int
        The 0-based index of the block's first record inside the crop.
    record_length : int
        The length of one line record in bytes.

    Returns
    -------
    bytes
        A copy of the block where record i (0-based in the block) has the sequence number first + i + 2 in bytes 1-4
        (big-endian int32); every other byte is unchanged, the image line numbers included.

    Raises
    ------
    ValueError
        If the block length is not a multiple of the record length.

    Examples
    --------
    >>> renumber(block, 0, 1056)
    """
    if len(block) % record_length != 0:
        raise ValueError(f'IMG block of {len(block)} bytes is not a whole number of {record_length}-byte records')
    records = np.frombuffer(block, dtype=np.uint8).reshape(-1, record_length).copy()
    sequence = (first + 2 + np.arange(records.shape[0], dtype=np.int64)).astype('>i4')
    records[:, 0:4] = sequence.view(np.uint8).reshape(-1, 4)
    return records.tobytes()


def write_crop(path, descriptor, blocks, lines, record_length):
    """
    Write a cropped IMG file: the descriptor with the crop's line counts, then the line records of the blocks.

    The file is written at `path + '.tmp'`, block by block (the blocks are never all in memory), checked and renamed
    to `path`; on any error the temporary file is removed and the error raised.

    Parameters
    ----------
    path : str
        The cropped IMG file.
    descriptor : bytes
        The 720-byte file descriptor of the source IMG file.
    blocks : iterable of bytes
        The line records of the crop in order, each block a whole number of records.
    lines : int
        The number of line records of the crop.
    record_length : int
        The length of one line record in bytes.

    Returns
    -------
    str
        The path of the cropped IMG file.

    Raises
    ------
    ValueError
        If the blocks do not hold `lines` records or the file size is not 720 + lines * record_length.

    Examples
    --------
    >>> write_crop('IMG-HH-ALOS2...-F1', descriptor, blocks, 114, 1056)
    """
    tmp_path = path + '.tmp'
    try:
        written = 0
        with open(tmp_path, 'wb') as f:
            f.write(crop_header(descriptor, lines))
            for block in blocks:
                f.write(renumber(block, written, record_length))
                written += len(block) // record_length
        if written != lines:
            raise ValueError(f'IMG crop has {written} records, expected {lines}')
        size = os.path.getsize(tmp_path)
        if size != DESCRIPTOR + lines * record_length:
            raise ValueError(f'IMG crop has {size} bytes, expected {DESCRIPTOR + lines * record_length}')
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return path
