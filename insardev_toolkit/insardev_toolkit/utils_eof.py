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
Sentinel-1 orbit (EOF) files, chosen by their names alone:

    S1A_OPER_AUX_POEORB_OPOD_20210207T122351_V20210117T225942_20210119T005942.EOF

holds the mission, the product (POEORB precise, RESORB restituted), the production time and the validity start
and stop. One mission and day can have several orbit files: a RESORB is produced every orbit and the validity
windows overlap, and a POEORB can be produced again for the same window (reprocessed). A scene takes, among the
files of its own mission whose validity covers its time with the downloader's margins, a precise orbit first,
then the newest production. The orbit downloader (EOF.download) and the Sentinel-1 scanners of the preprocessors
choose the same way, so the scan finds the file that the download selected.
"""
import os
import re
from datetime import datetime, timedelta
from typing import NamedTuple

from .HTTP import NotFound


class OrbitNotFound(NotFound, ValueError):
    """An orbit index was read and lists no orbit file of the mission covering a scene time: a retry cannot change
    it (HTTP.final), and it is a ValueError, as before."""


# the orbit file name, as it is on disk (.EOF) and in the orbit server index (.EOF.zip)
_NAME = re.compile(r'^(?P<mission>S1[A-Z])_OPER_AUX_(?P<product>POEORB|RESORB)_OPOD_(?P<production>\d{8}T\d{6})'
                   r'_V(?P<start>\d{8}T\d{6})_(?P<stop>\d{8}T\d{6})\.EOF(?:\.zip)?$')
_TIME = '%Y%m%dT%H%M%S'


class Orbit(NamedTuple):
    """An orbit file: its name as given and the fields of the name."""
    name: str
    mission: str
    product: str
    production: datetime
    start: datetime
    stop: datetime


def parse(name: str) -> Orbit:
    """
    Parse a Sentinel-1 orbit file name.

    Parameters
    ----------
    name : str
        The orbit file name or path, with the .EOF or .EOF.zip extension.

    Returns
    -------
    Orbit
        The name as given, the mission (S1A, S1B, ...), the product (POEORB or RESORB), the production time and
        the validity start and stop times (UTC, naive).

    Raises
    ------
    ValueError
        If the name is not the name of a POEORB or RESORB orbit file.
    """
    base = os.path.basename(name)
    match = _NAME.match(base)
    if match is None:
        raise ValueError(f'Not a Sentinel-1 POEORB or RESORB orbit file name: {base}')
    production, start, stop = (datetime.strptime(match.group(key), _TIME) for key in ('production', 'start', 'stop'))
    return Orbit(name, match.group('mission'), match.group('product'), production, start, stop)


def parse_files(names, root_dir=None) -> list:
    """
    Parse the orbit file names found in a directory; a name that is not an orbit file name is ignored with a note.

    Parameters
    ----------
    names : iterable of str
        The file names.
    root_dir : str, optional
        The directory of the files: an empty orbit file in it raises (utils_files.exists), as a selected one
        cannot be read. The files are not checked when None.

    Returns
    -------
    list of Orbit
        The parsed orbit files.
    """
    orbits = []
    for name in names:
        try:
            orbit = parse(name)
        except ValueError as e:
            print(f'NOTE: {e}. The file is ignored.')
            continue
        if root_dir is not None:
            from .utils_files import exists
            exists(os.path.join(root_dir, name))
        orbits.append(orbit)
    return orbits


def margins(offset_start=None, offset_end=None) -> tuple:
    """
    Return the margins inside the validity window that a covered time needs.

    The defaults are the downloader's EOF.orbit_offset_start and EOF.orbit_offset_end, read at the call, so that
    a change of them applies to the scan as well as to the download.
    """
    if offset_start is None or offset_end is None:
        from .EOF import EOF
        offset_start = EOF.orbit_offset_start if offset_start is None else offset_start
        offset_end = EOF.orbit_offset_end if offset_end is None else offset_end
    return offset_start, offset_end


def covers(orbit: Orbit, time, offset_start=None, offset_end=None) -> bool:
    """
    Check that the validity of an orbit file covers a time with the margins.

    Parameters
    ----------
    orbit : Orbit
        The parsed orbit file.
    time : datetime.datetime or pandas.Timestamp
        The scene time (UTC, naive).
    offset_start, offset_end : datetime.timedelta, optional
        The margins after the validity start and before the validity stop. Default EOF.orbit_offset_start and
        EOF.orbit_offset_end.

    Returns
    -------
    bool
        True when validity start + offset_start <= time <= validity stop - offset_end.
    """
    offset_start, offset_end = margins(offset_start, offset_end)
    return orbit.start + offset_start <= time <= orbit.stop - offset_end


def select(orbits, mission: str, time, offset_start=None, offset_end=None):
    """
    Select the orbit file of a scene.

    Among the files of the scene's mission that cover the scene time, a precise orbit (POEORB) first, then the
    newest production (then the later validity, as the names sort).

    Parameters
    ----------
    orbits : iterable of Orbit
        The candidate orbit files.
    mission : str
        The mission of the scene (S1A, S1B, ...).
    time : datetime.datetime or pandas.Timestamp
        The scene time (UTC, naive).
    offset_start, offset_end : datetime.timedelta, optional
        The margins, see covers().

    Returns
    -------
    str or None
        The name of the selected file as given, None when no file of the mission covers the time.
    """
    offset_start, offset_end = margins(offset_start, offset_end)
    candidates = [orbit for orbit in orbits
                  if orbit.mission == mission and covers(orbit, time, offset_start, offset_end)]
    if not candidates:
        return None
    return max(candidates, key=lambda orbit: (orbit.product == 'POEORB', orbit.production,
                                              orbit.start, orbit.stop)).name


# the seconds of orbit kept in a burst annotation before the burst start and after the burst stop
ORBIT_MARGIN = 120.0
# the time of an orbit state vector, as an orbit file (without its UTC= prefix) and a burst annotation write it
_OSV_TIME = '%Y-%m-%dT%H:%M:%S.%f'


def osv_records(content, windows) -> list:
    """
    The orbit state vectors of a Sentinel-1 orbit (EOF) file inside time windows, with their texts as the file
    writes them.

    Parameters
    ----------
    content : bytes
        The orbit file.
    windows : iterable of tuple
        The (start, stop) windows, naive UTC datetimes; a state vector at a window end is inside it.

    Returns
    -------
    list of tuple
        The records (time, x, y, z, vx, vy, vz) of the state vectors inside any window, in file order: the UTC
        time without its UTC= prefix and the texts of X, Y, Z, VX, VY and VZ, all str, never converted to numbers.

    Raises
    ------
    ValueError
        If the content holds no Data_Block/List_of_OSVs/OSV records.
    """
    import xml.etree.ElementTree as ET

    osvs = ET.fromstring(content).findall('Data_Block/List_of_OSVs/OSV')
    if not osvs:
        raise ValueError('Not a Sentinel-1 orbit file: no List_of_OSVs records')
    windows = list(windows)
    records = []
    for osv in osvs:
        time = osv.findtext('UTC').removeprefix('UTC=')
        when = datetime.strptime(time, _OSV_TIME)
        if any(start <= when <= stop for start, stop in windows):
            records.append((time,) + tuple(osv.findtext(key) for key in ('X', 'Y', 'Z', 'VX', 'VY', 'VZ')))
    return records


def _bracket(records, start, stop):
    """
    The records from the last one at or before start to the first one at or after stop, ends included.

    Parameters
    ----------
    records : list of tuple
        The records as osv_records() returns them, sorted by time.
    start, stop : datetime.datetime
        The times to bracket (UTC, naive).

    Returns
    -------
    list of tuple or None
        The bracketing records, None when no record is at or before start or none is at or after stop.
    """
    times = [datetime.strptime(record[0], _OSV_TIME) for record in records]
    before = [index for index, time in enumerate(times) if time <= start]
    after = [index for index, time in enumerate(times) if time >= stop]
    if not before or not after:
        return None
    return list(records[before[-1]:after[0] + 1])


def set_orbit(annotation, source, records) -> bytes:
    """
    Store the orbit of a burst in its annotation: the orbit list of the annotation is replaced by the records of
    an orbit file around the burst, ORBIT_MARGIN seconds before its start and after its stop, tagged by the file.

    Parameters
    ----------
    annotation : bytes
        A stored burst annotation: a <product> with exactly one swathTiming/burstList/burst.
    source : str
        The orbit file name (.EOF, without .zip).
    records : list of tuple
        The records of the orbit file as osv_records() returns them, sorted by time.

    Returns
    -------
    bytes
        The annotation with its <orbitList> replaced by <orbitList count="N" source="SOURCE"> and one <orbit> per
        kept record, in the layout of the file; every byte before and after the list is kept.

    Raises
    ------
    utils_eof.OrbitNotFound
        If the records do not cover the burst with the margins (final for HTTP.final, not retried).
    ValueError
        If the source is not an orbit file name, or the annotation has not exactly one orbit list.
    """
    import xml.etree.ElementTree as ET

    parse(source)
    root = ET.fromstring(annotation)
    start = datetime.strptime(root.findtext('swathTiming/burstList/burst/azimuthTime'), _OSV_TIME)
    lines_per_burst = int(root.findtext('swathTiming/linesPerBurst'))
    interval = float(root.findtext('imageAnnotation/imageInformation/azimuthTimeInterval'))
    # the burst stop as utils_S1.burst_xmls computes it
    stop = start + timedelta(seconds=(lines_per_burst - 1) * interval)
    low = start - timedelta(seconds=ORBIT_MARGIN)
    high = stop + timedelta(seconds=ORBIT_MARGIN)
    kept = _bracket(records, low, high)
    if kept is None:
        raise OrbitNotFound(f'ERROR: the orbit {source} does not cover the burst from {low} to {high}.')

    matches = list(re.finditer(rb'<orbitList\b[^>]*>.*?</orbitList>', annotation, re.S))
    if len(matches) != 1:
        raise ValueError(f'Expected one orbitList in the annotation, found {len(matches)}')
    match = matches[0]
    # the indentation of the list when its tag starts a line (pretty xmltodict layout), two levels deep
    begin = match.start()
    while begin > 0 and annotation[begin - 1:begin] in (b' ', b'\t'):
        begin -= 1
    if begin > 0 and annotation[begin - 1:begin] == b'\n':
        newline, pad = '\n', annotation[begin:match.start()].decode()
    else:
        newline, pad = '', ''
    unit = pad[:len(pad) // 2]
    one, two, three = pad + unit, pad + unit * 2, pad + unit * 3
    parts = [f'<orbitList count="{len(kept)}" source="{source}">']
    for time, x, y, z, vx, vy, vz in kept:
        parts += [f'{one}<orbit>',
                  f'{two}<time>{time}</time>',
                  f'{two}<frame>Earth Fixed</frame>',
                  f'{two}<position>',
                  f'{three}<x>{x}</x>',
                  f'{three}<y>{y}</y>',
                  f'{three}<z>{z}</z>',
                  f'{two}</position>',
                  f'{two}<velocity>',
                  f'{three}<x>{vx}</x>',
                  f'{three}<y>{vy}</y>',
                  f'{three}<z>{vz}</z>',
                  f'{two}</velocity>',
                  f'{one}</orbit>']
    parts.append(f'{pad}</orbitList>')
    element = newline.join(parts).encode()
    return annotation[:match.start()] + element + annotation[match.end():]


def stored_orbit(path):
    """
    The orbit stored in a burst annotation by set_orbit().

    Parameters
    ----------
    path : str
        The annotation file.

    Returns
    -------
    tuple or None
        (source, records): the orbit file name and the records (time, x, y, z, vx, vy, vz), all str, the texts
        as stored; None for an annotation as ESA wrote it (its orbit list has no source attribute).

    Raises
    ------
    xml.etree.ElementTree.ParseError
        If the file is not XML.
    """
    import xml.etree.ElementTree as ET

    orbits = ET.parse(path).getroot().find('generalAnnotation/orbitList')
    source = orbits.get('source')
    if source is None:
        return None
    keys = ('time', 'position/x', 'position/y', 'position/z', 'velocity/x', 'velocity/y', 'velocity/z')
    return source, [tuple(orbit.findtext(key) for key in keys) for orbit in orbits.findall('orbit')]
