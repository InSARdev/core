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
from .Alos1_slc import Alos1_slc
# the box reader and the image shape of the CEOS IMG files, which read the line prefix length (412 bytes for ALOS-1) from
# the file's descriptor: the ALOS-2 hooks as they are
from .Alos2_align import _alos2_slc_box, _alos2_slc_shape


class Alos1_align(Stripmap_align, Alos1_slc):
    """
    ALOS-1 alignment: the reference scene (align_ref) and the repeat scenes (align_rep, the geometry plus the xcorr refinement).

    The PALSAR L1.1 image is a stripmap SLC focused at the leader's Doppler centroid, not at zero Doppler: the stripmap
    alignment (Stripmap_align) with the ALOS-1 hooks, among them the native geometry of each date (_native_geometry).
    """

    # utils_alos2.alos2_slc: the pixels after the line prefix whose length the IMG descriptor gives (412 bytes for ALOS-1)
    _slc_box = staticmethod(_alos2_slc_box)
    _slc_shape = staticmethod(_alos2_slc_shape)

    def _doppler_centroid(self, prm) -> float:
        """
        The Doppler centroid of an ALOS-1 date in cycles per line: the leader's fd1 + fdd1 R at mid swath
        (utils_alos1.alos1_doppler_centroid) divided by the PRF.

        The image is focused at that centroid, so its azimuth spectrum is centred there: the xcorr re-centring places the band
        about it, and the transform's chunk workers demodulate the SLC with it before the LANCZOS4 remap and restore the
        carrier after it (the stripmap demodulation pair).

        Parameters
        ----------
        prm : PRM
            The PRM of the date.

        Returns
        -------
        float
            The centroid in cycles per line.

        Examples
        --------
        >>> alos1._doppler_centroid(prm_ref)
        """
        from .utils_alos1 import alos1_doppler_centroid
        return alos1_doppler_centroid(prm) / float(prm.get('PRF'))

    def _native_height(self, prm) -> float:
        """
        The median WGS84 ellipsoidal DEM height (m) over the footprint of a date's record, computed once per date (its LED
        file) and cached on the object.

        The DEM points are those of the alignment (_get_topo_llt at 12 arc seconds) inside the record's footprint (JAXA's per-line
        positions). The record is the one whose IMG file is the PRM's input_file.

        Parameters
        ----------
        prm : PRM
            The PRM of the date (input_file is read).

        Returns
        -------
        float
            The median height in meters.

        Raises
        ------
        ValueError
            No DEM point inside the footprint.

        Examples
        --------
        >>> alos1._native_height(prm_ref)
        """
        import numpy as np
        import shapely

        record = self.df[self.df['path'] == prm.get('input_file')]
        assert len(record) == 1, f'Record not found for the IMG file {prm.get("input_file")}'
        key = record['orbit'].iloc[0]
        heights = self.__dict__.setdefault('_native_heights', {})
        if key not in heights:
            topo_llt = self._get_topo_llt(record.index.get_level_values(2)[0], degrees=12.0 / 3600)
            inside = shapely.contains_xy(record.geometry.iloc[0], topo_llt[:, 0], topo_llt[:, 1])
            if not inside.any():
                raise ValueError(f'No DEM point inside the footprint of {record.index.get_level_values(2)[0]}: check the DEM')
            heights[key] = float(np.median(topo_llt[inside, 2]))
        return heights[key]

    def _native_geometry(self, prm) -> dict:
        """
        The native-geometry parameters of an ALOS-1 date (utils_alos1.alos1_native_geometry): its Doppler centroid, the
        wavelength, the PRF, the range sampling and the effective velocity at the first and the last bin, at the scene's
        median DEM height over its footprint (_native_height, once per date).

        The alignment and the transform's chunk workers turn the zero-Doppler lines into image lines with them
        (utils_satellite.native_doppler_shift), the transform applies the native-geometry phase (native_doppler_phase), and
        the output extent follows the reference image (utils_satellite.native_doppler_lines).

        Parameters
        ----------
        prm : PRM
            The PRM of the date, with its orbit (orbit_df).

        Returns
        -------
        dict
            The parameters (plain values).

        Examples
        --------
        >>> alos1._native_geometry(prm_ref)
        """
        from .utils_alos1 import alos1_native_geometry
        return alos1_native_geometry(prm, prm.orbit_df, self._native_height(prm))
