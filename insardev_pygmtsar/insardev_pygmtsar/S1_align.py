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
                      int_ashift: int, int_rshift: int,
                      patch_size: int = 256,
                      min_response: float = 0.1, debug: bool = False) -> dict:
    """
    Measure total alignment offsets via amplitude cross-correlation.

    Uses only integer constant shifts for coarse alignment. xcorr measures
    the full sub-pixel offset at each patch position. The integer shifts
    are added back to get total offsets, which are then fitted with a
    bilinear model to produce the final alignment parameters.

    Parameters
    ----------
    ref_path : str
        Path to reference SLC geotiff.
    rep_path : str
        Path to repeat SLC geotiff.
    int_ashift, int_rshift : int
        Integer azimuth and range shifts for coarse alignment (TIFF space).
    patch_size : int
        Xcorr patch size. Default 256.
    min_response : float
        Minimum correlation response.
    debug : bool
        Print debug info.

    Returns
    -------
    dict
        Total alignment parameters (bilinear model fitted to total offsets):
        {
            'rshift': float, 'stretch_r': float, 'a_stretch_r': float,
            'ashift': float, 'stretch_a': float, 'a_stretch_a': float,
        }

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

        if debug:
            print(f"Xcorr: {n_rows}x{n_cols} = {n_rows*n_cols} patches, size {patch_size}")
            print(f"  Images: ref={ny_ref}x{nx_ref}, rep={ny_rep}x{nx_rep}")
            print(f"  Integer shifts: ashift={int_ashift}, rshift={int_rshift}")

        for row in range(n_rows):
            cy1 = int((row + 0.5) * ny_ref / n_rows)
            for col in range(n_cols):
                cx1 = int((col + 0.5) * nx_ref / n_cols)

                # Coarse alignment: integer shifts only
                cy2 = cy1 + int_ashift
                cx2 = cx1 + int_rshift

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
                    # Total offset = integer shift + xcorr-measured sub-pixel
                    result['dy'] += int_ashift
                    result['dx'] += int_rshift
                    result['cy1'] = cy1
                    result['cx1'] = cx1
                    results.append(result)

    if debug:
        print(f"  Valid patches: {len(results)}")

    # Fit bilinear model to TOTAL offsets
    corrections = xcorr_fitoffset(results, nx=nx_ref, ny=ny_ref, debug=debug)

    if corrections is None:
        raise RuntimeError("Xcorr correction did not converge - insufficient valid high-coherence patches. "
                           "Use xcorr=None for geometry-only alignment in low-coherence areas.")

    if debug:
        print(f"  Fitted total: ashift={corrections['ashift']:.4f}, stretch_a={corrections['stretch_a']:.8f}, a_stretch_a={corrections['a_stretch_a']:.8f}")
        print(f"                rshift={corrections['rshift']:.4f}, stretch_r={corrections['stretch_r']:.8f}, a_stretch_r={corrections['a_stretch_r']:.8f}")

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
                  xcorr: tuple = None, xcorr_min_response: float = 0.2,
                  topo_llt: "np.ndarray | None" = None) -> tuple:
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
        if topo_llt is None:
            topo_llt = self._get_topo_llt(burst_ref, degrees=degrees)

        # Extract PRM, orbit, deramped SLC, and reramp params in one call
        prm_dict, orbit_df, slc_data, reramp_params = self._make_burst(burst_rep, mode=2)
        prm_rep = PRM()
        prm_rep.set(**prm_dict)
        prm_rep.orbit_df = orbit_df

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
                # Compute median integer shifts from raw geometry offsets
                # par_tmp columns: [r, dr, a, da, SNR]
                median_da = np.median(par_tmp[:, 3])
                median_dr = np.median(par_tmp[:, 1])

                # Convert azimuth offset from PRM to TIFF space
                ref_xml = os.path.join(self.datadir, prefix_ref, 'annotation', f'{burst_ref}.xml')
                rep_xml = os.path.join(self.datadir, prefix_rep, 'annotation', f'{burst_rep}.xml')
                k_start_ref = self._get_k_start(ref_xml)
                k_start_rep = self._get_k_start(rep_xml)
                k_start_correction = k_start_rep - k_start_ref

                int_ashift_tiff = int(np.round(median_da + k_start_correction))
                int_rshift = int(np.round(median_dr))

                if debug:
                    print(f"  Median geometry: da={median_da:.2f}, dr={median_dr:.2f}, "
                          f"k_start_corr={k_start_correction}")
                    print(f"  Integer shifts (TIFF): ashift={int_ashift_tiff}, rshift={int_rshift}")

                # xcorr measures total offsets (int_shift + sub-pixel) at each patch
                # then fits bilinear model to get final parameters (in TIFF space)
                try:
                    xcorr_params = _xcorr_refine_slc(
                        ref_tiff, rep_tiff,
                        int_ashift=int_ashift_tiff, int_rshift=int_rshift,
                        patch_size=xcorr_patch_size,
                        min_response=xcorr_min_response, debug=debug
                    )
                except RuntimeError as e:
                    if debug:
                        print(f"  WARNING: {e}")
                    xcorr_params = None

                if xcorr_params is None:
                    # Xcorr failed - keep geometry alignment from fitoffset(3,3) above
                    print(f"WARNING: Xcorr FAILED for {burst_rep} - using geometry")
                else:
                    # Range-only xcorr: orbit geometry for azimuth, xcorr for range.
                    # TOPS reramp phase has range-dependent Doppler centroid; per-burst
                    # ashift noise × fnct(range) creates inter-burst range ramps.
                    # Orbit-based azimuth is smooth across burst boundaries.
                    geom_ashift = prm_rep.get('ashift') + prm_rep.get('sub_int_a')
                    rshift_new = xcorr_params['rshift']

                    prm_rep.set(
                        # Azimuth: orbit geometry (smooth across burst boundaries)
                        ashift=int(geom_ashift) if geom_ashift >= 0 else int(geom_ashift) - 1,
                        sub_int_a=math.fmod(geom_ashift, 1) if geom_ashift >= 0 else math.fmod(geom_ashift, 1) + 1,
                        stretch_a=prm_rep.get('stretch_a'),
                        a_stretch_a=prm_rep.get('a_stretch_a'),
                        # Range: xcorr total offsets (improves coherence, no inter-burst ramp)
                        rshift=int(rshift_new) if rshift_new >= 0 else int(rshift_new) - 1,
                        sub_int_r=math.fmod(rshift_new, 1) if rshift_new >= 0 else math.fmod(rshift_new, 1) + 1,
                        stretch_r=xcorr_params['stretch_r'],
                        a_stretch_r=xcorr_params['a_stretch_r'],
                    )

                    if debug:
                        print(f"  Range-only xcorr:")
                        print(f"    ashift={geom_ashift:.4f} (geometry), rshift={rshift_new:.4f} (xcorr)")

        # Recompute Doppler with earth_radius
        prm_rep.calc_dop_orb(earth_radius, inplace=True, debug=debug)

        return prm_rep, slc_data, reramp_params
