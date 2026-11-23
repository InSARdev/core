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
ALOS-1 PALSAR L1.1 IMG files have the same layout with a 412-byte prefix (the first 412 bytes of the ALOS-2 prefix,
the same fields): image_info takes the prefix length.
A cropped IMG file is the descriptor with its two line counts rewritten, followed by the selected line records with
their record sequence numbers renumbered from 2 (the image line numbers keep the source lines). The downloader
reads the descriptor and the line records of one subswath from the granule ZIP by byte ranges and writes them here.
The lines of an ALOS-1 raw (L1.0) crop are selected by the time tag of each line (line_seconds, select_lines).

Byte positions below are 1-based, as in the CEOS format description.
"""
import os
from datetime import date, timedelta

import numpy as np

DESCRIPTOR = 720
PREFIX = 544
# the line prefix of ALOS-1 PALSAR L1.1 IMG files
PREFIX_ALOS1 = 412


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


def image_info(head, prefix_length=PREFIX, raw=False):
    """
    Read the image size, the timing and the near range of an ALOS-2 or ALOS-1 CEOS IMG file from its first bytes.

    The record length must be prefix_length + 8 * pixels (L1.1: float32 I, Q), the pixels of the first line's prefix
    (bytes 25-28). An ALOS-1 L1.0 (raw signal data) IMG file (raw=True) has the descriptor and the 412-byte line prefix
    of L1.1, with the same timing fields (year, day, milliseconds of day, PRF, slant range of the first sample),
    followed by the echo samples as 8-bit I, Q pairs (2 bytes a sample; descriptor bytes 225-228: 2, and 8 for L1.1)
    and fill samples: its record length must be prefix_length + 2 * (left fill + pixels + right fill), the counts of
    the prefix bytes 21-24, 25-28 and 29-32 (0, 10304 and 40 samples in ALPSRP055006360-L1.0, measured 2026-11-23).

    Parameters
    ----------
    head : bytes
        The first 720 + prefix_length bytes of the IMG file: the file descriptor and the prefix of the first line
        record.
    prefix_length : int, optional
        The length of the line prefix: PREFIX (544) for ALOS-2, PREFIX_ALOS1 (412) for ALOS-1. Default PREFIX.
    raw : bool, optional
        True for an ALOS-1 L1.0 (raw signal data) IMG file, False (default) for an L1.1 (SLC) file.

    Returns
    -------
    dict
        'lines' (number of line records), 'record_length' (bytes per line record), 'pixels' (per line; of a raw
        file the echo samples, without the fill),
        'time' (seconds of day of line 1), 'prf' (Hz), 'near_range' (slant range of the first pixel, m) and
        'date' ('YYYY-MM-DD' of line 1).

    Raises
    ------
    ValueError
        If the bytes are not the head of a consistent IMG file.

    Examples
    --------
    >>> image_info(open('IMG-HH-ALOS2...-F1', 'rb').read(1264))
    >>> image_info(open('IMG-HH-ALPSRP131296360-H1.1__A', 'rb').read(1132), PREFIX_ALOS1)
    >>> image_info(open('IMG-HH-ALPSRP055006360-H1.0__A', 'rb').read(1132), PREFIX_ALOS1, raw=True)
    """
    if len(head) < DESCRIPTOR + prefix_length:
        raise ValueError(f'IMG head has {len(head)} bytes, expected {DESCRIPTOR + prefix_length}')
    descriptor_length = _int32(head, 9)
    if descriptor_length != DESCRIPTOR:
        raise ValueError(f'IMG descriptor length {descriptor_length}, expected {DESCRIPTOR}')
    lines = _ascii_int(head, 181, 186)
    lines_per_set = _ascii_int(head, 237, 244)
    if lines != lines_per_set:
        raise ValueError(f'IMG line counts differ: {lines} and {lines_per_set}')
    record_length = _ascii_int(head, 187, 192)
    prefix = bytes(head[DESCRIPTOR:DESCRIPTOR + prefix_length])
    pixels = _int32(prefix, 25)
    if raw:
        fill = _int32(prefix, 21) + _int32(prefix, 29)
        if record_length != prefix_length + 2 * (pixels + fill):
            raise ValueError(f'IMG record length {record_length} does not match {pixels} samples and {fill} fill of the '
                             f'raw data')
    elif record_length != prefix_length + 8 * pixels:
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


def prefix_lonlat(prefix, prefix_length=PREFIX):
    """
    Read JAXA's longitude and latitude of the first, mid and last pixel of a line from its record prefix (the same
    fields in the ALOS-2 and the ALOS-1 prefix).

    Parameters
    ----------
    prefix : bytes
        The first prefix_length bytes of a line record (or more).
    prefix_length : int, optional
        The length of the line prefix: PREFIX (544) for ALOS-2, PREFIX_ALOS1 (412) for ALOS-1. Default PREFIX.

    Returns
    -------
    tuple of numpy.ndarray
        (lon, lat): float64 arrays of 3 values, degrees on the WGS84 ellipsoid at height 0, of the first, mid and last
        pixel: the big-endian int32 fields at bytes 205, 209, 213 (longitude) and 193, 197, 201 (latitude), in 1e-6
        degrees.

    Raises
    ------
    ValueError
        If the prefix is shorter than prefix_length bytes.

    Examples
    --------
    >>> prefix_lonlat(open('IMG-HH-ALOS2...-F1', 'rb').read(1264)[720:])
    >>> prefix_lonlat(open('IMG-HH-ALPSRP131296360-H1.1__A', 'rb').read(1132)[720:], PREFIX_ALOS1)
    """
    if len(prefix) < prefix_length:
        raise ValueError(f'IMG line prefix has {len(prefix)} bytes, expected {prefix_length}')
    lat = np.array([_int32(prefix, b) for b in (193, 197, 201)], dtype=np.float64) * 1e-6
    lon = np.array([_int32(prefix, b) for b in (205, 209, 213)], dtype=np.float64) * 1e-6
    return lon, lat


def line_seconds(records, record_length, origin):
    """
    The times of ALOS-1 PALSAR raw (L1.0) line records, from their line headers: seconds since 00:00 UTC of a day.

    The time of a line is its millisecond tag, the year (prefix bytes 37-40), day of the year (41-44) and
    milliseconds of the day (45-48), big-endian int32 (the UTC millisecond; GMTSAR's ALOS preprocessing reads the same
    three fields), plus its part below the millisecond from the mission auxiliary data of the header (JAXA's ALOS
    Product Format Description, PALSAR Level 1.0, Appendix A-2; the auxiliary data from byte 289): the "1Mpps time" of
    bytes 297-300, 4 dummy bits, 20 bits counting microseconds from the 1PPS pulse (the GPS second) to the line's pulse,
    8 dummy bits. The microseconds of the 1Mpps time minus those of the tag within its second (taken modulo one second)
    is the part below the millisecond, bounded to 0..999 microseconds: a field that disagrees with the tag moves no line
    out of the tag's millisecond. Measured 2026-11-23 on the 34,610 lines of ALPSRP055006320-L1.0: the 1Mpps time minus
    the tag -14..+999 microseconds, its step per line the PRI of the line's PRF (478 and 471 microseconds before and
    after the PRF change at line 14,089), the 1PPS GPS time (bytes 289-296) 14 s after the tag (GPS - UTC).

    Parameters
    ----------
    records : bytes
        A whole number of line records (or of their 412-byte headers).
    record_length : int
        The length of one line record in bytes.
    origin : str
        'YYYY-MM-DD': the day the seconds are counted from.

    Returns
    -------
    numpy.ndarray
        float64 seconds, one per record.

    Raises
    ------
    ValueError
        If the bytes are not a whole number of records, or a record's year and day are not a date.

    Examples
    --------
    >>> line_seconds(head[720:], 21100, '2007-02-05')
    """
    if len(records) % record_length != 0:
        raise ValueError(f'IMG block of {len(records)} bytes is not a whole number of {record_length}-byte records')
    if record_length < PREFIX_ALOS1:
        raise ValueError(f'IMG record length {record_length} is shorter than the {PREFIX_ALOS1}-byte line header')
    heads = np.frombuffer(records, dtype=np.uint8).reshape(-1, record_length)
    year, day, ms = np.ascontiguousarray(heads[:, 36:48]).view('>i4').astype(np.int64).T
    mpps = (np.ascontiguousarray(heads[:, 296:300]).view('>u4').ravel().astype(np.int64) >> 8) & 0xFFFFF
    day0 = date.fromisoformat(origin)
    days = {}
    for y, d in set(zip(year.tolist(), day.tolist())):
        try:
            days[y, d] = (date(y, 1, 1) + timedelta(days=d - 1) - day0).days
        except (ValueError, OverflowError):
            raise ValueError(f'IMG line time tag of year {y}, day {d} is not a date') from None
    offset = np.array([days[y, d] for y, d in zip(year.tolist(), day.tolist())], dtype=np.float64)
    # the microseconds below the tag's millisecond, modulo one second, bounded to the millisecond
    below = np.clip((mpps - 1000 * (ms % 1000) + 500000) % 1000000 - 500000, 0, 999)
    return 86400.0 * offset + (1000 * ms + below) / 1e6


def select_lines(blocks, record_length, lines, origin, start, end, block, window):
    """
    The line records of an ALOS-1 IMG file whose time tags fall in a time span, the window rounded outward to `block`
    lines and clipped to the file's lines: a generator over the file's records from line 1 that stops reading at
    the window's end.

    A line is in the span when its time (line_seconds) lies in ``start..end``. The window starts on the multiple of
    `block` lines at or before the first line at or after `start`, and ends on the multiple of `block` lines at or
    after the first line after `end` (the window's end is exclusive), at most the last line. Each line is placed by
    its own time, so a PRF change inside the file moves no line. The records before the window are dropped as they pass (at
    most `block` - 1 of them held), and `blocks` is not read beyond the window's last line.

    Parameters
    ----------
    blocks : iterable of bytes
        The file's line records in order from line 1 (0-based line 0), each block a whole number of records.
    record_length : int
        The length of one line record in bytes.
    lines : int
        The number of line records of the file.
    origin : str
        'YYYY-MM-DD': the day `start` and `end` count their seconds from.
    start, end : float
        The span in seconds since 00:00 UTC of `origin` (utils_alos2_orbit.raw_window).
    block : int
        The window starts and ends on multiples of `block` lines.
    window : dict
        Receives ``'start'`` and ``'end'`` (0-based lines, end exclusive) when the window is found; stays empty when
        no line lies in the span (nothing is yielded then).

    Yields
    ------
    bytes
        The window's records, in blocks of whole records (at most one block of `blocks` plus `block` - 1 records).

    Examples
    --------
    >>> window = {}
    >>> write_crop('AL1_..._HH.raw', head[:720], select_lines(blocks, 21100, 34820, '2007-02-05', 14900.0, 14910.0, 512, window),
    ...            None, 21100)
    """
    index, first, stop = 0, None, None
    held = []
    for data in blocks:
        seconds = line_seconds(data, record_length, origin)
        view = memoryview(data)
        out = bytearray()
        for k, t in enumerate(seconds.tolist()):
            record = view[k * record_length:(k + 1) * record_length]
            if first is None:
                if index % block == 0:
                    held.clear()
                if t >= start:
                    if t > end:
                        # the span lies before this line: no line meets it
                        return
                    first = (index // block) * block
                    for r in held:
                        out += r
                    held.clear()
                    out += record
                else:
                    held.append(bytes(record))
            else:
                if stop is None and t > end:
                    stop = min(lines, -(-index // block) * block)
                if stop is not None and index >= stop:
                    if out:
                        yield bytes(out)
                    window.update(start=first, end=stop)
                    return
                out += record
            index += 1
        if out:
            yield bytes(out)
    if first is not None:
        window.update(start=first, end=index)


def crop_lines(head, record_length):
    """
    The lines of the source IMG file that a cropped IMG file (write_crop) holds.

    The first line is the image line number of the crop's first record (prefix bytes 13-16, big-endian int32, 1-based:
    write_crop renumbers only the record sequence numbers), the count the crop's line count (descriptor bytes
    181-186). Measured 2026-11-23: the line number of every record of three ALOS-1 raw (L1.0) frames (FBS, FBD, PLR,
    98,412 lines) is its 1-based position in the file.

    Parameters
    ----------
    head : bytes
        The first 720 + 16 bytes of the crop (or more): its descriptor and the start of its first record.
    record_length : int
        The record length of the source file.

    Returns
    -------
    tuple of int
        ``(start, end)``: 0-based lines of the source file, end exclusive.

    Raises
    ------
    ValueError
        If the head is too short, the crop's record length is not `record_length`, or its line count or first line
        number is below 1.

    Examples
    --------
    >>> crop_lines(open('AL1_118_6360_343_20070205_HH.raw', 'rb').read(1132), 21100)
    """
    if len(head) < DESCRIPTOR + 16:
        raise ValueError(f'IMG crop head has {len(head)} bytes, expected {DESCRIPTOR + 16}')
    length = _ascii_int(head, 187, 192)
    if length != record_length:
        raise ValueError(f'IMG crop record length {length}, expected {record_length}')
    lines = _ascii_int(head, 181, 186)
    first = _int32(head[DESCRIPTOR:], 13)
    if lines < 1 or first < 1:
        raise ValueError(f'IMG crop of {lines} lines from line {first}')
    return first - 1, first - 1 + lines


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
    to `path`; on any error the temporary file is removed and the error raised. With lines=None the crop holds the
    records the blocks give, and its line counts are written into the descriptor when the last block is written
    (a crop whose window is found while its records pass, utils_alos2_ceos.select_lines); blocks that give no record
    write nothing.

    Parameters
    ----------
    path : str
        The cropped IMG file.
    descriptor : bytes
        The 720-byte file descriptor of the source IMG file.
    blocks : iterable of bytes
        The line records of the crop in order, each block a whole number of records.
    lines : int or None
        The number of line records of the crop; None: the records the blocks give.
    record_length : int
        The length of one line record in bytes.

    Returns
    -------
    str or None
        The path of the cropped IMG file; None when lines is None and the blocks give no record (nothing written).

    Raises
    ------
    ValueError
        If the blocks do not hold `lines` records or the file size is not 720 + lines * record_length.

    Examples
    --------
    >>> write_crop('IMG-HH-ALOS2...-F1', descriptor, blocks, 114, 1056)
    >>> write_crop('AL1_118_6360_343_20070205_HH.raw', descriptor, select_lines(...), None, 21100)
    """
    tmp_path = path + '.tmp'
    try:
        written = 0
        with open(tmp_path, 'wb') as f:
            # the line counts of an open crop are written after its last block
            f.write(crop_header(descriptor, lines) if lines is not None else bytes(descriptor))
            for block in blocks:
                f.write(renumber(block, written, record_length))
                written += len(block) // record_length
            if lines is None and written:
                lines = written
                f.seek(0)
                f.write(crop_header(descriptor, lines))
        if lines is None:
            os.remove(tmp_path)
            return None
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
