# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .S1_gmtsar import S1_gmtsar
from .PRM import PRM
import numpy as np


def _read_slc_patch(src, cy: int, cx: int, half: int) -> np.ndarray:
    """
    Read a complex SLC patch from an open rasterio dataset.

    Handles both formats:
    - 1 band complex (S1 geotiffs: complex_int16 → complex64)
    - 2 bands real/imag (NISAR: int16 pairs)

    Parameters
    ----------
    src : rasterio.DatasetReader
        Open rasterio dataset.
    cy, cx : int
        Center coordinates of patch.
    half : int
        Half patch size.

    Returns
    -------
    np.ndarray
        Complex64 patch of shape (2*half, 2*half).
    """
    from rasterio.windows import Window
    window = Window(cx - half, cy - half, 2 * half, 2 * half)
    data = src.read(window=window)
    if src.count == 1:
        # Single band complex (S1)
        return data[0].astype(np.complex64)
    else:
        # Two bands: real, imag (NISAR)
        return (data[0] + 1j * data[1]).astype(np.complex64)


def _xcorr_refine_slc(ref_path: str, rep_path: str,
                      ashift: float, rshift: float,
                      stretch_a: float = 0.0, stretch_r: float = 0.0,
                      a_stretch_a: float = 0.0, a_stretch_r: float = 0.0,
                      patch_size: int = 256,
                      min_response: float = 0.1, debug: bool = False) -> dict:
    """
    Run xcorr refinement by reading patches directly from geotiff files.

    Reads only the required patches from disk, avoiding full image load.
    Handles both S1 (1 band complex) and NISAR (2 bands real/imag) formats.

    Parameters
    ----------
    ref_path : str
        Path to reference SLC geotiff.
    rep_path : str
        Path to repeat SLC geotiff.
    ashift, rshift : float
        Geometry-based azimuth and range shifts.
    stretch_a, stretch_r, a_stretch_a, a_stretch_r : float
        Geometry-based stretch parameters.
    patch_size : int
        Xcorr patch size. Default 256.
    min_response : float
        Minimum correlation response.
    debug : bool
        Print debug info.

    Returns
    -------
    dict
        Correction coefficients (same as xcorr_fitoffset output).

    Raises
    ------
    RuntimeError
        If xcorr failed (insufficient valid patches).
    """
    import rasterio
    from .utils_satellite import xcorr_patch, xcorr_fitoffset

    half = patch_size // 2
    hann = np.outer(np.hanning(patch_size), np.hanning(patch_size)).astype(np.float32)
    results = []

    with rasterio.open(ref_path) as src_ref, rasterio.open(rep_path) as src_rep:
        ny_ref, nx_ref = src_ref.height, src_ref.width
        ny_rep, nx_rep = src_rep.height, src_rep.width

        # Auto-compute grid: ~2x patch spacing, minimum 4 patches per dimension
        n_rows = max(4, (ny_ref - patch_size) // (2 * patch_size) + 1)
        n_cols = max(4, (nx_ref - patch_size) // (2 * patch_size) + 1)
        grid = (n_rows, n_cols)

        if debug:
            print(f"Xcorr refinement: {grid[0]}×{grid[1]} = {grid[0]*grid[1]} patches, size {patch_size}")
            print(f"Image sizes: ref={ny_ref}×{nx_ref}, rep={ny_rep}×{nx_rep}")
            print(f"Geometry params: ashift={ashift:.2f}, rshift={rshift:.2f}")

        n_rows, n_cols = grid
        for row in range(n_rows):
            cy1 = int((row + 0.5) * ny_ref / n_rows)
            for col in range(n_cols):
                cx1 = int((col + 0.5) * nx_ref / n_cols)

                # Apply geometry offset - compute float position first
                cy2_float = cy1 + ashift + stretch_a * cx1 + a_stretch_a * cy1
                cx2_float = cx1 + rshift + stretch_r * cx1 + a_stretch_r * cy1

                # Truncate to integer for patch reading
                cy2 = int(cy2_float)
                cx2 = int(cx2_float)

                # Track truncation artifact - phaseCorrelate will "find" this sub-pixel
                # and we need to subtract it to get the TRUE residual
                frac_a = cy2_float - cy2
                frac_r = cx2_float - cx2

                # Bounds check
                if cy1 < half or cy1 > ny_ref - half:
                    continue
                if cy2 < half or cy2 > ny_rep - half:
                    continue
                if cx1 < half or cx1 > nx_ref - half:
                    continue
                if cx2 < half or cx2 > nx_rep - half:
                    continue

                # Read and correlate patches
                patch1 = _read_slc_patch(src_ref, cy1, cx1, half)
                patch2 = _read_slc_patch(src_rep, cy2, cx2, half)

                result = xcorr_patch(patch1, patch2, hann, min_response=min_response)
                if result is not None:
                    # Compensate for int() truncation artifact
                    # phaseCorrelate "finds" the sub-pixel that was lost by truncation
                    # Subtract it to get the TRUE residual beyond the geometry
                    result['dy'] -= frac_a
                    result['dx'] -= frac_r
                    result['cy1'] = cy1
                    result['cx1'] = cx1
                    results.append(result)

    if debug:
        print(f"Xcorr results: {len(results)} with response > {min_response}")

    # Fit bilinear using full radar extent for normalization
    corrections = xcorr_fitoffset(results, nx=nx_ref, ny=ny_ref, debug=debug)

    if corrections is None:
        raise RuntimeError("Xcorr fitoffset failed - insufficient valid patches. Scene alignment cannot continue.")

    if debug:
        print(f"Xcorr fitoffset result:")
        print(f"  ashift={corrections['ashift']:.4f}, stretch_a={corrections['stretch_a']:.8f}, a_stretch_a={corrections['a_stretch_a']:.8f}")
        print(f"  rshift={corrections['rshift']:.4f}, stretch_r={corrections['stretch_r']:.8f}, a_stretch_r={corrections['a_stretch_r']:.8f}")

    return corrections


class S1_align(S1_gmtsar):
    import numpy as np
    import xarray as xr
    import pandas as pd

    @staticmethod
    def _get_k_start(xml_path: str) -> int:
        """
        Get k_start (first valid line in TIFF) from burst XML annotation.

        k_start is the offset between TIFF row 0 and PRM azimuth 0.
        This is needed to convert geometry offsets (computed in PRM space)
        to TIFF offsets (for xcorr which reads from TIFF).

        Parameters
        ----------
        xml_path : str
            Path to burst XML annotation file.

        Returns
        -------
        int
            First valid line index in the TIFF (k_start).
        """
        import xml.etree.ElementTree as ET

        tree = ET.parse(xml_path)
        root = tree.getroot()

        first_valid = root.find('.//burstList/burst/firstValidSample')
        if first_valid is None:
            return 0

        samples = [int(x) for x in first_valid.text.split()]
        for i, s in enumerate(samples):
            if s >= 0:
                return i
        return 0

    @staticmethod
    def _offset2shift(offset_dat: np.ndarray, rmax: int, amax: int) -> tuple:
        """
        Convert offset coordinates to r and a shift grids in one interpolation.

        Uses LinearNDInterpolator to build triangulation once and interpolate
        both r_shift and a_shift values together, which is ~2x faster than
        calling griddata separately for each.

        Parameters
        ----------
        offset_dat : numpy.ndarray
            Array with columns [r, dr, a, da, snr] from offset computation.
        rmax : int
            Maximum range bin.
        amax : int
            Maximum azimuth line.

        Returns
        -------
        tuple
            (r_grd, a_grd) - xarray.DataArrays with shift grids.
        """
        import xarray as xr
        import numpy as np
        from scipy.interpolate import LinearNDInterpolator

        # use center pixel GMT registration mode
        rngs = np.arange(8/2, rmax+8/2, 8)
        azis = np.arange(4/2, amax+4/2, 4)
        grid_r, grid_a = np.meshgrid(rngs, azis)
        grid_points = np.column_stack([grid_r.ravel(), grid_a.ravel()])

        # Input points: (range, azimuth) coordinates
        points = np.column_stack([offset_dat[:, 0], offset_dat[:, 2]])  # [r, a]
        # Shift values: (dr, da)
        shifts = np.column_stack([offset_dat[:, 1], offset_dat[:, 3]])  # [dr, da]

        # Build triangulation once, interpolate both shifts
        interp = LinearNDInterpolator(points, shifts, fill_value=np.nan)
        result = interp(grid_points)  # Shape: (n_points, 2)

        # Reshape to grid
        r_grid = result[:, 0].reshape(grid_r.shape)
        a_grid = result[:, 1].reshape(grid_r.shape)

        r_grd = xr.DataArray(r_grid, coords={'y': azis, 'x': rngs}, name='z')
        a_grd = xr.DataArray(a_grid, coords={'y': azis, 'x': rngs}, name='z')

        return r_grd, a_grd

    def align_ref(self, burst: str, debug: bool = False, return_slc: bool = True) -> tuple:
        """
        Process reference burst - extract PRM, orbit, and optionally deramped SLC.

        All data is returned in-memory, no files are written.

        Parameters
        ----------
        burst : str
            Burst identifier.
        debug : bool, optional
            Enable debug mode. Default is False.
        return_slc : bool, optional
            If True, load and return deramped SLC data. If False, only return PRM.
            Default is True.

        Returns
        -------
        tuple
            (prm, slc_deramped, reramp_params) or (prm, None, None) if return_slc=False
        """
        if return_slc:
            from .PRM import PRM
            prm_dict, orbit_df, slc_data, reramp_params = self._make_burst(burst, mode=2)
            prm = PRM()
            prm.set(**prm_dict)
            prm.orbit_df = orbit_df
            prm.calc_dop_orb(inplace=True, debug=debug)
            return prm, slc_data, reramp_params

        prm, orbit_df = self._make_burst(burst, mode=0, debug=debug)
        prm.calc_dop_orb(inplace=True, debug=debug)
        return prm, None, None

    def align_rep(self, burst_rep: str, burst_ref: str, prm_ref: "PRM",
                  degrees: float = 12.0/3600, debug: bool = False,
                  xcorr: tuple = (128, 128), xcorr_min_response: float = 0.2) -> tuple:
        """
        Process and align secondary burst to reference.

        Returns deramped SLC (no alignment shift, no reramp) with alignment
        offsets stored in the PRM. The alignment and geocoding interpolations
        are merged into a single remap step during the transform phase,
        avoiding double interpolation.

        All data is returned in-memory, no files are written.

        Parameters
        ----------
        burst_rep : str
            Secondary burst identifier.
        burst_ref : str
            Reference burst identifier (used for DEM geometry).
        prm_ref : PRM
            Reference PRM object (from align_ref) with Doppler parameters.
        degrees : float, optional
            Degrees per pixel resolution for the coarse DEM. Default is 12.0/3600.
        debug : bool, optional
            Enable debug mode. Default is False.
        xcorr : tuple or None, optional
            Xcorr patch size as (height, width). Default (256, 256) for S1.
            Set to None to disable xcorr refinement. Grid is auto-computed
            to cover the image with ~2x patch spacing.
        xcorr_min_response : float, optional
            Minimum correlation response. Default 0.2 matches GMTSAR's SNR=20.

        Returns
        -------
        tuple
            (prm, slc_data, reramp_params) where:
            - prm: PRM object with orbit_df attached, alignment, and Doppler parameters
            - slc_data: Deramped SLC data as int16 numpy array (no shift, no reramp)
            - reramp_params: dict with parameters for analytical reramp phase computation

        Examples
        --------
        >>> prm_ref, slc_ref, _ = stack.align_ref(burst_ref)
        >>> prm_rep, slc_rep, reramp_params = stack.align_rep(burst_rep, burst_ref, prm_ref)
        """
        import numpy as np

        # Get reference earth_radius
        earth_radius = prm_ref.get('earth_radius')

        # Prepare coarse DEM for alignment using REFERENCE burst geometry
        # (12 arc seconds is enough for SRTM 90m)
        topo_llt = self._get_topo_llt(burst_ref, degrees=degrees)

        # Extract PRM and orbit for secondary burst (mode=0, no SLC yet)
        prm_rep, orbit_df = self._make_burst(burst_rep, mode=0, debug=debug)

        # Compute time difference between frames
        t1, prf = prm_rep.get('clock_start', 'PRF')
        t2 = prm_rep.get('clock_start')
        nl = int((t2 - t1) * prf * 86400.0 + 0.2)

        # Create shifted reference PRM for SAT_llt2rat
        prm_ref_shifted = PRM(prm_ref)
        prm_ref_shifted.orbit_df = prm_ref.orbit_df
        prm_ref_shifted.set(
            prm_ref.sel('clock_start', 'clock_stop', 'SC_clock_start', 'SC_clock_stop')
            + nl / prf / 86400.0
        )
        prm_ref_shifted.calc_dop_orb(earth_radius, inplace=True, debug=debug)

        # Compute offset from reference to secondary
        tmpm_dat = prm_ref_shifted.SAT_llt2rat(coords=topo_llt, precise=1, debug=debug)

        prm_rep.calc_dop_orb(earth_radius, inplace=True, debug=debug)
        tmp1_dat = prm_rep.SAT_llt2rat(coords=topo_llt, precise=1, debug=debug)

        # Compute r, dr, a, da, SNR table for fitoffset (vectorized)
        offset_dat0 = np.hstack([tmpm_dat, tmp1_dat])
        offset_dat = np.column_stack([
            offset_dat0[:, 0],                      # r_ref
            offset_dat0[:, 5] - offset_dat0[:, 0],  # dr = r_rep - r_ref
            offset_dat0[:, 1],                      # a_ref
            offset_dat0[:, 6] - offset_dat0[:, 1],  # da = a_rep - a_ref
            np.full(len(offset_dat0), 100.0)        # SNR
        ])

        # Get radar coordinates extent
        rmax = prm_rep.get('num_rng_bins')
        amax = prm_rep.get('num_lines')

        # Filter to points inside valid radar extent
        valid_mask = (
            (offset_dat[:, 0] > 0) & (offset_dat[:, 0] < rmax) &
            (offset_dat[:, 2] > 0) & (offset_dat[:, 2] < amax)
        )
        offset_dat_valid = offset_dat[valid_mask]

        # Prepare offset parameters for fitoffset
        par_tmp = offset_dat_valid.copy()
        par_tmp[:, 2] += nl

        # Extract deramped SLC (no shift, no reramp)
        prm_dict, orbit_df_new, slc_data, reramp_params = self._make_burst(burst_rep, mode=2)

        # Build PRM from dict (same as _make_burst does)
        prm_rep = PRM()
        prm_rep.set(**prm_dict)
        prm_rep.orbit_df = orbit_df_new

        # Apply fitoffset parameters (bilinear offset model stored in PRM)
        prm_rep.set(PRM.fitoffset(3, 3, par_tmp))

        # Xcorr refinement: measure actual offsets and correct geometry alignment
        if xcorr is not None:
            import os
            import math

            # Extract patch size from tuple
            xcorr_patch_size = xcorr[0] if isinstance(xcorr, tuple) else int(xcorr)

            if debug:
                print(f"Running xcorr refinement (patch_size={xcorr_patch_size})...")

            # Get tiff file paths (read patches directly, don't load full images)
            prefix_ref = self.fullBurstId(burst_ref)
            prefix_rep = self.fullBurstId(burst_rep)
            ref_tiff = os.path.join(self.datadir, prefix_ref, 'measurement', f'{burst_ref}.tiff')
            rep_tiff = os.path.join(self.datadir, prefix_rep, 'measurement', f'{burst_rep}.tiff')

            # Check files exist
            if not os.path.exists(ref_tiff) or not os.path.exists(rep_tiff):
                if debug:
                    print(f"  WARNING: Xcorr skipped - files not found:")
                    print(f"    ref: {ref_tiff} (exists: {os.path.exists(ref_tiff)})")
                    print(f"    rep: {rep_tiff} (exists: {os.path.exists(rep_tiff)})")
            else:
                if debug:
                    print(f"  ref_tiff: {ref_tiff}")
                    print(f"  rep_tiff: {rep_tiff}")

                # Get geometry alignment from prm_rep (computed by fitoffset above)
                # This is in PRM coordinate space (valid lines only)
                geom_ashift = prm_rep.get('ashift') + prm_rep.get('sub_int_a')
                geom_rshift = prm_rep.get('rshift') + prm_rep.get('sub_int_r')
                geom_stretch_a = prm_rep.get('stretch_a')
                geom_stretch_r = prm_rep.get('stretch_r')
                geom_a_stretch_a = prm_rep.get('a_stretch_a')
                geom_a_stretch_r = prm_rep.get('a_stretch_r')

                # Get k_start values to convert PRM coords to TIFF coords
                # TIFF row 0 = PRM azimuth -k_start, so TIFF row = PRM azi + k_start
                # For xcorr which reads from TIFF, we need to correct the offset:
                # TIFF_offset = PRM_offset + (k_start_rep - k_start_ref)
                ref_xml = os.path.join(self.datadir, prefix_ref, 'annotation', f'{burst_ref}.xml')
                rep_xml = os.path.join(self.datadir, prefix_rep, 'annotation', f'{burst_rep}.xml')
                k_start_ref = self._get_k_start(ref_xml)
                k_start_rep = self._get_k_start(rep_xml)
                k_start_correction = k_start_rep - k_start_ref

                # Convert geometry to TIFF space for xcorr
                geom_ashift_tiff = geom_ashift + k_start_correction

                if debug:
                    print(f"  Geometry (PRM): ashift={geom_ashift:.2f}, rshift={geom_rshift:.2f}")
                    print(f"  k_start: ref={k_start_ref}, rep={k_start_rep}, correction={k_start_correction}")
                    print(f"  Geometry (TIFF): ashift={geom_ashift_tiff:.2f}")

                # Run xcorr WITH geometry pre-applied (in TIFF space) to find small residuals
                xcorr_params = _xcorr_refine_slc(
                    ref_tiff, rep_tiff,
                    ashift=geom_ashift_tiff, rshift=geom_rshift,
                    stretch_a=geom_stretch_a, stretch_r=geom_stretch_r,
                    a_stretch_a=geom_a_stretch_a, a_stretch_r=geom_a_stretch_r,
                    patch_size=xcorr_patch_size,
                    min_response=xcorr_min_response, debug=debug
                )

                # Check if xcorr succeeded
                if xcorr_params is None:
                    # Xcorr failed - keep geometry alignment
                    print(f"WARNING: Xcorr FAILED for {burst_rep} - using geometry")
                    ashift_new = geom_ashift
                    stretch_a_new = geom_stretch_a
                    a_stretch_a_new = geom_a_stretch_a
                    rshift_new = geom_rshift
                    stretch_r_new = geom_stretch_r
                    a_stretch_r_new = geom_a_stretch_r
                else:
                    # xcorr found residuals - only apply RANGE refinement
                    # Azimuth offsets kept from orbit geometry to avoid TOPS
                    # inter-burst range ramp (reramp phase couples ashift × Doppler centroid)
                    ashift_new = geom_ashift
                    stretch_a_new = geom_stretch_a
                    a_stretch_a_new = geom_a_stretch_a
                    rshift_new = geom_rshift + xcorr_params['rshift']
                    stretch_r_new = geom_stretch_r + xcorr_params['stretch_r']
                    a_stretch_r_new = geom_a_stretch_r + xcorr_params['a_stretch_r']

                    if debug:
                        print(f"  Xcorr residuals: da={xcorr_params['ashift']:.4f} (ignored), dr={xcorr_params['rshift']:.4f} (applied)")

                # Update PRM (split into integer and fractional parts)
                prm_rep.set(
                    ashift=int(ashift_new) if ashift_new >= 0 else int(ashift_new) - 1,
                    sub_int_a=math.fmod(ashift_new, 1) if ashift_new >= 0 else math.fmod(ashift_new, 1) + 1,
                    stretch_a=stretch_a_new,
                    a_stretch_a=a_stretch_a_new,
                    rshift=int(rshift_new) if rshift_new >= 0 else int(rshift_new) - 1,
                    sub_int_r=math.fmod(rshift_new, 1) if rshift_new >= 0 else math.fmod(rshift_new, 1) + 1,
                    stretch_r=stretch_r_new,
                    a_stretch_r=a_stretch_r_new,
                )

                if debug:
                    print(f"Xcorr alignment:")
                    print(f"  ashift={ashift_new:.4f}, rshift={rshift_new:.4f}")
                    print(f"  stretch_a={stretch_a_new:.8f}, stretch_r={stretch_r_new:.8f}")
                    print(f"  a_stretch_a={a_stretch_a_new:.8f}, a_stretch_r={a_stretch_r_new:.8f}")

        # Recompute Doppler with earth_radius
        prm_rep.calc_dop_orb(earth_radius, inplace=True, debug=debug)

        return prm_rep, slc_data, reramp_params
