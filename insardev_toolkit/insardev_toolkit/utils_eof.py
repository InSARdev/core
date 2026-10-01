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
from datetime import datetime
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
