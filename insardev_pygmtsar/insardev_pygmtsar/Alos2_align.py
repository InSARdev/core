# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .Stripmap_align import Stripmap_align
from .Alos2_slc import Alos2_slc


def _alos2_slc_box(img_path: str, row_slice: slice, col_slice: slice):
    """
    The box reader of an ALOS-2 IMG file (the stripmap hook _slc_box): utils_alos2.alos2_slc.

    One byte range per line through a memmap of the line records (offset 720; the pixels after the 544-byte prefix,
    big-endian complex64 converted to native complex64). A top-level function: the xcorr and chunk workers receive it.

    Parameters
    ----------
    img_path : str
        Path to the IMG file.
    row_slice : slice
        0-based lines.
    col_slice : slice
        0-based range bins.

    Returns
    -------
    numpy.ndarray
        complex64 (lines, bins) of the box.

    Examples
    --------
    >>> _alos2_slc_box('129_4450_F4/AL2_129_4450_F4_20150404_HH.img', slice(0, 64), slice(0, 64))
    """
    from .utils_alos2 import alos2_slc
    return alos2_slc(img_path, row_slice=row_slice, col_slice=col_slice)


def _alos2_slc_shape(img_path: str) -> tuple:
    """
    (lines, pixels) of an ALOS-2 IMG file (the stripmap hook _slc_shape), from utils_alos2.alos2_image.

    Parameters
    ----------
    img_path : str
        Path to the IMG file.

    Returns
    -------
    tuple
        (lines, pixels).

    Examples
    --------
    >>> _alos2_slc_shape('129_4450_F4/AL2_129_4450_F4_20150404_HH.img')
    """
    from .utils_alos2 import alos2_image
    info = alos2_image(img_path)
    return info['lines'], info['pixels']


class Alos2_align(Stripmap_align, Alos2_slc):
    """
    ALOS-2 alignment: the reference scene (align_ref) and the repeat scenes (align_rep, the geometry plus the xcorr refinement).

    The ScanSAR L1.1 subswath is a zero-Doppler full-aperture SLC, processed as stripmap: no TOPS deramp/reramp. The
    stripmap alignment (Stripmap_align) with the ALOS-2 hooks.
    """

    _slc_box = staticmethod(_alos2_slc_box)
    _slc_shape = staticmethod(_alos2_slc_shape)

    def _doppler_centroid(self, prm) -> None:
        """
        The Doppler centroid of an ALOS-2 date: None, no demodulation, and the xcorr re-centring places the band about 0.

        No demodulation: the ALOS-2 azimuth spectrum is centred at 0.016 cycles per line in all range blocks, 0.7-0.8 of the
        PRF wide, and demodulating by that centroid does not change the LANCZOS4 interpolation error (median 0.04-0.11 rad;
        measured, m2b): unlike NISAR (about 0.63 cycles per line), the band already sits about 0

        Parameters
        ----------
        prm : PRM
            The PRM of the date.

        Returns
        -------
        None

        Examples
        --------
        >>> alos2._doppler_centroid(prm_ref)
        """
        return None
