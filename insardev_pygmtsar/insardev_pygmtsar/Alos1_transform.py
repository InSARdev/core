# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .Stripmap_transform import Stripmap_transform
from .Alos1_align import Alos1_align
from .Alos1_slc import _alos1_date_groups


class Alos1_transform(Stripmap_transform, Alos1_align):
    """
    ALOS-1 transform: the PALSAR L1.1 SLC geocoded as stripmap from its native-Doppler geometry, the polarizations of a date
    on HH's alignment.

    The stripmap transform (Stripmap_transform) with the ALOS-1 hooks and the default resolution of its transform.
    """

    # value = int16 x 100 (measured, unit A0a of the SLC preprocessor plan, by the ALOS-2 rule: the round scale whose clipped
    # fraction of |I| or |Q| is well below 1e-5; HH clips 2.8e-6 .. 3.9e-6 of the samples on the four Rosamond dates; GMTSAR's
    # ALOS-1 SLC_factor 0.01 too)
    _int16_scale = 100.0
    # the stack attributes: pathNumber from the PRM key 'path', the band and the mission
    _path_attr = 'path'
    _band = 'L'
    _mission = 'ALOS-1'
    # the polarizations of a date share ONE alignment, defined by HH (PLR: all four)
    _date_groups = staticmethod(_alos1_date_groups)

    def transform(self,
                  target: str,
                  ref: str,
                  epsg: str | int | None = 'auto',
                  resolution: tuple[int, int] = (4, 8),
                  chunk: tuple[int, int] | None = None,
                  remove_topo_phase: bool = True,
                  remove_tidal_phase: bool = True,
                  reference_height: float | None = None,
                  dem_vertical_accuracy: float = 0.5,
                  alignment_spacing: float = 12.0 / 3600,
                  xcorr: tuple | int | None = (256, 256),
                  bbox: list | tuple | None = None,
                  overwrite: bool = True,
                  append: bool = False,
                  n_jobs: int | None = None,
                  scheduler: str | None = None,
                  tmpdir: str | None = None,
                  debug: bool = False):
        """
        Transform ALOS-1 PALSAR L1.1 SLC data to geographic coordinates.

        The image of a date is focused at its Doppler centroid, not at zero Doppler: each output pixel is mapped into it by the
        native-Doppler shift of its range (_native_geometry), its phase is turned into the zero-Doppler phase, and the output
        covers the reference image's own lines. The SLC is demodulated about the date's centroid for the interpolation
        (_doppler_centroid). The polarizations of a date share ONE alignment, defined by HH (geometry, xcorr, tide): HV (and
        VH, VV of a PLR date) is resampled on it, all in one pass over the date's chunks. The closest-approach search of the
        baselines uses half the record's own duration (a crop does not carry its scene's duration; decision D3), as a
        Sentinel-1 burst.

        Parameters
        ----------
        target : str
            The output directory where the results are saved.
        ref : str
            The reference scene date (YYYY-MM-DD).
        epsg : str|int|None, optional
            The EPSG code to use for the output data. Use 'auto' for automatic.
            With None, each scene uses the UTM zone of its own centroid.
            Geocoding is always enabled: epsg=0 (radar coordinates) is not supported and raises ValueError.
        resolution : tuple[int, int], optional
            The resolution to use in meters per pixel (y, x). Default (4, 8): FBS's native ground range (32 MHz, about
            7.5 m at the off-nadir angle 34.3 deg), so FBS and FBD (16 MHz) dates mix naturally in one stack; use (4, 16),
            FBD's native ground range (about 15 m), for an FBD-only stack. A multiple of 4 m as the grids of the other
            missions.
        chunk : None, optional
            Not used, kept for backward compatibility (a WARNING is printed when it is set). The stored chunks
            are those of the S1 rule and a worker processes 2 x 2 of them; rechunk a loaded stack with chunk2d().
        remove_topo_phase : bool, optional
            Remove the topographic phase from SLC data for interferometric processing. Set to False
            when creating a DEM from interferograms so the topo phase remains: the flat-earth phase of the
            WGS84 ellipsoid at reference_height (and the tide) is removed instead, like Sentinel-1.
        remove_tidal_phase : bool, optional
            Remove solid Earth tidal displacement phase.
        reference_height : float or None, optional
            Reference height (meters above WGS84 ellipsoid) for flat-earth phase removal.
            All scenes use this same value.
            Set to the elevation of your area of interest for best precision and fewer fringes.
            Default is None (sea level, i.e. 0). Only used when remove_topo_phase=False.
            Raises ValueError if set when remove_topo_phase=True.
        dem_vertical_accuracy : float, optional
            The DEM vertical accuracy in meters.
        alignment_spacing : float, optional
            The alignment spacing in decimal degrees.
        xcorr : tuple | int | None, optional
            Xcorr window (square patch) in pixels, as (height, width) or one int; the height sets the size.
            Default (256, 256). The grid covers the whole reference IMG file, every other window on each axis; each
            accepted patch is re-centred 3 times, and a patch whose re-centring stops early is skipped. The correction
            is a constant shift (rank 1). When that area is too small for the window (fewer than 8 accepted patches), a
            RuntimeError asks for a smaller window, e.g. xcorr=192 or xcorr=128. None disables the refinement: the
            geometry alone.
        bbox : list | tuple | None, optional
            Bounding box [lon_min, lat_min, lon_max, lat_max] in WGS84 to crop
            output grid. Useful when input data was downloaded for a subregion.
        overwrite : bool, optional
            Overwrite existing results. Default True: a completed target is removed and processed again, with a NOTE;
            overwrite=False keeps the completed target.
        append : bool, optional
            Append new scenes to existing results. Requires overwrite=True: the existing target is kept and extended
            (the consolidated zarr.json is written again). With overwrite=False an existing target is never modified and
            the call skips it, with or without append.
        n_jobs : int, optional
            Number of parallel workers of every internal step: the geocoding tiles, the alignment
            (SAT_llt2rat and xcorr) and the SLC chunks. None or -1 (default): all cores.
            Each chunk worker holds one processing chunk of every polarization, 2 x 2 zarr chunks of at most 32 MiB of
            complex64 each (the S1 rule): lower n_jobs to use less RAM.
        scheduler : str, optional
            Not used for ALOS-1 (kept for API compatibility).
        tmpdir : str, optional
            Not used for ALOS-1 (kept for API compatibility).
        debug : bool, optional
            Whether to print debug information.

        Examples
        --------
        >>> alos1.transform('stack', ref='2007-07-03')
        """
        return super().transform(target, ref, resolution=resolution, epsg=epsg, chunk=chunk,
                                 remove_topo_phase=remove_topo_phase, remove_tidal_phase=remove_tidal_phase,
                                 reference_height=reference_height, dem_vertical_accuracy=dem_vertical_accuracy,
                                 alignment_spacing=alignment_spacing, xcorr=xcorr, bbox=bbox, overwrite=overwrite,
                                 append=append, n_jobs=n_jobs, scheduler=scheduler, tmpdir=tmpdir, debug=debug)
