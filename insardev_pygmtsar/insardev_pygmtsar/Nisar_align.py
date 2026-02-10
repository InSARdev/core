# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .Nisar_slc import Nisar_slc
from .PRM import PRM


def _xcorr_batch(h5_path1, h5_path2, slc_path, patches, patch_size, min_response=0.2):
    """Process a batch of xcorr patches - single worker, reuses file handles.

    Each patch dict contains:
    - cy1, cx1: reference patch center (integer)
    - cy2, cx2: secondary patch center (integer, truncated from float)
    - frac_a, frac_r: fractional part lost by truncation (to compensate)

    Parameters
    ----------
    min_response : float
        Minimum correlation response threshold. Default 0.2 matches GMTSAR's SNR=20.
    """
    import numpy as np
    import cv2
    import h5py

    half = patch_size // 2
    hann = np.outer(np.hanning(patch_size), np.hanning(patch_size)).astype(np.float32)

    results = []
    with h5py.File(h5_path1, 'r') as f1, h5py.File(h5_path2, 'r') as f2:
        ds1 = f1[slc_path]
        ds2 = f2[slc_path]

        for p in patches:
            cy1, cx1 = p['cy1'], p['cx1']
            cy2, cx2 = p['cy2'], p['cx2']
            frac_a = p.get('frac_a', 0.0)
            frac_r = p.get('frac_r', 0.0)

            patch1 = ds1[cy1-half:cy1+half, cx1-half:cx1+half]
            patch2 = ds2[cy2-half:cy2+half, cx2-half:cx2+half]

            # Check valid data
            valid = (patch1 != 0) & (patch2 != 0)
            if valid.sum() < 0.5 * valid.size:
                continue

            # Normalize amplitudes
            amp1 = np.abs(patch1).astype(np.float32)
            amp2 = np.abs(patch2).astype(np.float32)
            amp1_norm = ((amp1 - amp1.mean()) / (amp1.std() + 1e-10)).astype(np.float32)
            amp2_norm = ((amp2 - amp2.mean()) / (amp2.std() + 1e-10)).astype(np.float32)

            # Phase correlation
            (dx, dy), response = cv2.phaseCorrelate(amp1_norm * hann, amp2_norm * hann)

            if response > min_response:
                # Compensate for int() truncation - phaseCorrelate "finds" the
                # sub-pixel that was lost, so subtract it to get TRUE residual
                dy_corr = dy - frac_a
                dx_corr = dx - frac_r
                results.append({'cy1': cy1, 'cx1': cx1, 'dy': dy_corr, 'dx': dx_corr, 'response': response})

    return results


class Nisar_align(Nisar_slc):
    """
    Nisar alignment - simpler than S1 (no deramp/reramp needed).

    Nisar uses stripmap mode, so direct SLC interpolation works without
    the complex deramp/reramp procedure required for Sentinel-1 TOPS.
    """
    import numpy as np
    import xarray as xr
    import pandas as pd

    def align_ref(self, scene: str, debug: bool = False, return_slc: bool = True) -> tuple:
        """
        Process reference scene - extract PRM, orbit, and optionally SLC.

        All data is returned in-memory, no files are written.

        Parameters
        ----------
        scene : str
            Scene identifier.
        debug : bool, optional
            Enable debug mode.
        return_slc : bool, optional
            If True, load and return SLC data. If False, only return PRM.

        Returns
        -------
        tuple
            (prm, slc_data, None) or (prm, None, None) if return_slc=False.
            Third element (reramp_params) is always None for Nisar (no TOPS deramp).
        """
        if return_slc:
            prm_dict, orbit_df, slc_data, _ = self._make_scene(scene, mode=2)
            prm = PRM()
            prm.set(**prm_dict)
            prm.orbit_df = orbit_df
            prm.calc_dop_orb(inplace=True, debug=debug)
            return prm, slc_data, None  # No reramp_params for Nisar

        prm, orbit_df = self._make_scene(scene, mode=0, debug=debug)
        prm.calc_dop_orb(inplace=True, debug=debug)
        return prm, None, None  # No slc_data, no reramp_params for Nisar

    def align_rep(self, scene_rep: str, scene_ref: str, prm_ref: "PRM",
                  degrees: float = 12.0 / 3600, debug: bool = False,
                  return_slc: bool = True,
                  xcorr: tuple = (512, 512), xcorr_n_jobs: int = 8,
                  xcorr_min_response: float = 0.2) -> tuple:
        """
        Process and align secondary scene to reference.

        For Nisar stripmap mode, alignment is simpler - no deramp/reramp needed.
        Returns SLC with alignment offsets stored in the PRM.

        Parameters
        ----------
        scene_rep : str
            Secondary scene identifier.
        scene_ref : str
            Reference scene identifier (used for DEM geometry).
        prm_ref : PRM
            Reference PRM object (from align_ref) with Doppler parameters.
        degrees : float, optional
            Degrees per pixel resolution for the coarse DEM.
        debug : bool, optional
            Enable debug mode.
        return_slc : bool, optional
            If True, load and return SLC data. If False, only return PRM.
        xcorr : tuple or None, optional
            Xcorr patch size as (height, width). Default (512, 512) for NISAR.
            Set to None to disable xcorr refinement. Grid is auto-computed.
        xcorr_n_jobs : int, optional
            Number of parallel workers for xcorr. Default 8.
        xcorr_min_response : float, optional
            Minimum correlation response threshold. Default 0.2 matches GMTSAR's SNR=20.

        Returns
        -------
        tuple
            (prm, slc_data, None) or (prm, None) if return_slc=False
        """
        import numpy as np

        earth_radius = prm_ref.get('earth_radius')

        # Prepare coarse DEM for alignment using REFERENCE scene geometry
        topo_llt = self._get_topo_llt(scene_ref, degrees=degrees)

        # Extract PRM and orbit for secondary scene (mode=0, no SLC yet)
        prm_rep, orbit_df = self._make_scene(scene_rep, mode=0, debug=debug)

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

        # Extract PRM (and optionally SLC)
        if return_slc:
            prm_dict, orbit_df_new, slc_data, _ = self._make_scene(scene_rep, mode=2)
        else:
            prm_rep_temp, orbit_df_new = self._make_scene(scene_rep, mode=0)
            prm_dict = {k: v for k, v in prm_rep_temp.df.itertuples()}
            slc_data = None

        # Build PRM from dict
        prm_rep = PRM()
        prm_rep.set(**prm_dict)
        prm_rep.orbit_df = orbit_df_new

        # Apply fitoffset parameters (bilinear offset model stored in PRM)
        prm_rep.set(PRM.fitoffset(3, 3, par_tmp))

        # Xcorr refinement: measure actual offsets and correct geometry alignment
        if xcorr is not None:
            # Extract patch size from tuple
            xcorr_patch_size = xcorr[0] if isinstance(xcorr, tuple) else int(xcorr)

            if debug:
                print(f"Running xcorr refinement (patch_size={xcorr_patch_size})...")

            xcorr_corrections = self._xcorr_refine(
                scene_ref, scene_rep, prm_rep,
                patch_size=xcorr_patch_size,
                n_jobs=xcorr_n_jobs,
                min_response=xcorr_min_response,
                debug=debug
            )

            import math

            # Get geometry from prm_rep
            geom_ashift = prm_rep.get('ashift') + prm_rep.get('sub_int_a')
            geom_rshift = prm_rep.get('rshift') + prm_rep.get('sub_int_r')
            geom_stretch_a = prm_rep.get('stretch_a')
            geom_stretch_r = prm_rep.get('stretch_r')
            geom_a_stretch_a = prm_rep.get('a_stretch_a')
            geom_a_stretch_r = prm_rep.get('a_stretch_r')

            # Check if xcorr succeeded
            if xcorr_corrections is None:
                # Xcorr failed - keep geometry
                print(f"WARNING: Xcorr FAILED for {scene_rep} - using geometry")
                ashift_new = geom_ashift
                stretch_a_new = geom_stretch_a
                a_stretch_a_new = geom_a_stretch_a
                rshift_new = geom_rshift
                stretch_r_new = geom_stretch_r
                a_stretch_r_new = geom_a_stretch_r
            else:
                # xcorr found residuals - ADD to geometry
                ashift_new = geom_ashift + xcorr_corrections['ashift']
                stretch_a_new = geom_stretch_a + xcorr_corrections['stretch_a']
                a_stretch_a_new = geom_a_stretch_a + xcorr_corrections['a_stretch_a']
                rshift_new = geom_rshift + xcorr_corrections['rshift']
                stretch_r_new = geom_stretch_r + xcorr_corrections['stretch_r']
                a_stretch_r_new = geom_a_stretch_r + xcorr_corrections['a_stretch_r']

                if debug:
                    print(f"  Xcorr residuals: da={xcorr_corrections['ashift']:.4f}, dr={xcorr_corrections['rshift']:.4f}")

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

        return prm_rep, slc_data, None  # No reramp_params for Nisar

    def _get_h5_path(self, scene: str) -> str:
        """Get HDF5 file path for a scene."""
        record = self.get_record(scene)
        return record['path'].iloc[0]

    def _get_slc_path(self, scene: str) -> str:
        """Get SLC dataset path within HDF5 file."""
        record = self.get_record(scene)
        pol = record.index.get_level_values(1)[0]
        return f'/science/LSAR/RSLC/swaths/frequency{self.frequency}/{pol}'

    def _xcorr_refine(self, scene_ref: str, scene_rep: str, prm_rep: "PRM",
                      patch_size: int = 512,
                      n_jobs: int = 8, min_response: float = 0.2,
                      debug: bool = False) -> dict:
        """
        Measure xcorr offsets and fit bilinear correction to geometry alignment.

        Uses per-patch parallel reads which are efficient because patch size (512×512)
        matches HDF5 chunk size. Grid is auto-computed based on image size.

        Parameters
        ----------
        scene_ref : str
            Reference scene identifier.
        scene_rep : str
            Repeat scene identifier.
        prm_rep : PRM
            Repeat scene PRM with geometry alignment parameters.
        patch_size : int, optional
            Patch size in pixels. Default 512 (matches HDF5 chunks).
        n_jobs : int, optional
            Number of parallel workers. Default 8.
        debug : bool, optional
            Print debug information.

        Returns
        -------
        dict
            Correction coefficients: {'c0': float, 'c1': float, 'c2': float}
            where dy_correction = c0 + c1*range + c2*azimuth
        """
        import numpy as np
        from joblib import Parallel, delayed

        # Get HDF5 paths and SLC dataset path
        h5_path1 = self._get_h5_path(scene_ref)
        h5_path2 = self._get_h5_path(scene_rep)
        slc_path = self._get_slc_path(scene_ref)

        # Get scene dimensions
        import h5py
        with h5py.File(h5_path1, 'r') as f:
            ny1, nx1 = f[slc_path].shape
        with h5py.File(h5_path2, 'r') as f:
            ny2, nx2 = f[slc_path].shape

        # Auto-compute grid: ~2x patch spacing, minimum 4 patches per dimension
        n_rows = max(4, (ny1 - patch_size) // (2 * patch_size) + 1)
        n_cols = max(4, (nx1 - patch_size) // (2 * patch_size) + 1)
        grid = (n_rows, n_cols)

        # Get geometry from prm_rep (computed by fitoffset)
        ashift = prm_rep.get('ashift') + prm_rep.get('sub_int_a')
        rshift = prm_rep.get('rshift') + prm_rep.get('sub_int_r')
        stretch_a = prm_rep.get('stretch_a')
        stretch_r = prm_rep.get('stretch_r')
        a_stretch_a = prm_rep.get('a_stretch_a')
        a_stretch_r = prm_rep.get('a_stretch_r')

        if debug:
            print(f"Xcorr refinement: {grid[0]}×{grid[1]} = {grid[0]*grid[1]} patches")
            print(f"Geometry: ashift={ashift:.2f}, rshift={rshift:.2f}")

        # Generate patch grid
        half = patch_size // 2
        n_rows, n_cols = grid
        patches = []

        for row in range(n_rows):
            cy1 = int((row + 0.5) * ny1 / n_rows)
            for col in range(n_cols):
                cx1 = int((col + 0.5) * nx1 / n_cols)

                # Apply geometry offset - compute float position first
                cy2_float = cy1 + ashift + stretch_a * cx1 + a_stretch_a * cy1
                cx2_float = cx1 + rshift + stretch_r * cx1 + a_stretch_r * cy1

                # Truncate to integer for patch reading
                cy2 = int(cy2_float)
                cx2 = int(cx2_float)

                # Track truncation artifact - phaseCorrelate will "find" this
                # sub-pixel and we need to subtract it to get TRUE residual
                frac_a = cy2_float - cy2
                frac_r = cx2_float - cx2

                # Bounds check
                if cy1 < half or cy1 > ny1 - half:
                    continue
                if cy2 < half or cy2 > ny2 - half:
                    continue
                if cx1 < half or cx1 > nx1 - half:
                    continue
                if cx2 < half or cx2 > nx2 - half:
                    continue

                patches.append({'cy1': cy1, 'cx1': cx1, 'cy2': cy2, 'cx2': cx2,
                               'frac_a': frac_a, 'frac_r': frac_r})

        if debug:
            print(f"Valid patches: {len(patches)}")

        # Split patches into batches for parallel processing
        # Each worker processes a batch with reused file handles
        n_batches = min(n_jobs, len(patches))
        batch_size = (len(patches) + n_batches - 1) // n_batches
        batches = [patches[i:i+batch_size] for i in range(0, len(patches), batch_size)]

        # Parallel xcorr (batch processing, workers reuse file handles)
        batch_results = Parallel(n_jobs=n_batches)(
            delayed(_xcorr_batch)(h5_path1, h5_path2, slc_path, batch, patch_size, min_response)
            for batch in batches
        )
        results = [r for batch in batch_results for r in batch]

        if debug:
            print(f"Xcorr results: {len(results)} with response > {min_response}")

        # Use shared fitoffset function with full radar extent for normalization
        from .utils_satellite import xcorr_fitoffset
        corrections = xcorr_fitoffset(results, nx=nx1, ny=ny1, debug=debug)

        if corrections is None:
            if debug:
                print("Xcorr fitoffset failed - insufficient valid patches")
            return None

        if debug:
            print(f"Xcorr fitoffset result:")
            print(f"  ashift={corrections['ashift']:.4f}, stretch_a={corrections['stretch_a']:.8f}, a_stretch_a={corrections['a_stretch_a']:.8f}")
            print(f"  rshift={corrections['rshift']:.4f}, stretch_r={corrections['stretch_r']:.8f}, a_stretch_r={corrections['a_stretch_r']:.8f}")

        return corrections
