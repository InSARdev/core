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
The existing files that the downloaders and the preprocessors read, reuse or skip.

Such a file is written under a temporary name and renamed when it is complete, so an empty one is never a write in
progress: a disk failure (a power loss, a full disk) or an older version left it, and it raises. Temporary files
(*.tmp) are never read: a stale one, from an interrupted write or from a file manager or backup tool, is ignored
and overwritten.
"""
import os
import stat


class EmptyFileError(OSError):
    """An existing file that must hold data is empty."""


def write_atomic(path, write):
    """
    Write `path` through the temporary file `path`.tmp, renamed to `path` when complete, so that an interrupted
    write leaves no file that passes as data; the temporary file is removed when the write fails.

    Parameters
    ----------
    path : str
        The file.
    write : callable
        write(tmp) writes the whole content to the temporary file tmp.
    """
    tmp = path + '.tmp'
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_file(path, content):
    """
    Write the whole content to `path` through write_atomic: bytes (or a buffer such as a memoryview) in binary
    mode, a str in text mode.
    """
    mode = 'w' if isinstance(content, str) else 'wb'

    def write(tmp):
        with open(tmp, mode) as f:
            f.write(content)
    write_atomic(path, write)


def exists(path, again='download'):
    """
    Whether a file or directory exists, as os.path.exists; an existing empty file raises EmptyFileError.

    Parameters
    ----------
    path : str
        The file.
    again : str, optional
        What makes the file again, named in the error: 'download' (default), or 'run the processing' for a
        processing result.

    Returns
    -------
    bool
        True for an existing file with data or a directory, False when nothing exists at the path.
    """
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return False
    if stat.S_ISREG(st.st_mode) and st.st_size == 0:
        raise EmptyFileError(f'Empty file {path}: check the disk space, then delete the empty files '
                             f'(or the whole folder) and {again} again.')
    return True
