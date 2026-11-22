# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .Alos2_slc import Alos2_slc
from .PRM import PRM
# the mission-agnostic xcorr estimator of NISAR, as it is
from .Nisar_align import _xcorr_pc, _xcorr_recentre, _xcorr_area_error, XCORR_RECENTRE_PASSES
from insardev_toolkit.utils_pool import release_workers


def _xcorr_grid(shape: tuple, patch_size: int) -> tuple:
    """
    The xcorr grid (rows, columns) over an ALOS-2 IMG file of shape (lines, pixels): every other window on each axis.

    Per axis n = max(4, (size - patch_size) // (2 x patch_size) + 1): NISAR band A's base spacing. An ALOS-2 crop holds whole
    lines, so there is no densification by the fraction of the frame.

    Parameters
    ----------
    shape : tuple
        (lines, pixels) of the reference IMG file.
    patch_size : int
        Edge length of the square xcorr window in pixels.

    Returns
    -------
    tuple
        (rows, columns) of the grid.

    Examples
    --------
    >>> _xcorr_grid((16896, 5670), 256)
    (33, 11)
    """
    return tuple(max(4, (size - patch_size) // (2 * patch_size) + 1) for size in shape)


def _xcorr_batch(img_path1: str, img_path2: str, patches: list, patch_size: int, min_response: float = 0.2,
                 passes: int = XCORR_RECENTRE_PASSES) -> tuple:
    """
    Process a batch of xcorr patches of two ALOS-2 IMG files - single worker, one memmap of each file.

    Each patch dict contains cy1, cx1 (the reference patch centre), cy2, cx2 (the repeat patch centre, truncated from float)
    and frac_a, frac_r (the fractional part lost by the truncation, compensated). A patch is the lines cy - half .. cy + half and
    the pixels cx - half .. cx + half of the line records (offset 720; the pixels after the 544-byte prefix, big-endian complex64
    converted to native complex64; a non-finite pixel is no-data, 0). The patches of a batch are consecutive in (cy1, cx1)
    order, so a batch reads a band of lines.

    The estimator is NISAR's (Nisar_align._xcorr_pc), and every accepted patch is re-centred `passes` times
    (Nisar_align._xcorr_recentre) with the azimuth frequencies placed about 0: the ALOS-2 azimuth spectrum is centred at
    0.016 cycles per line (measured, m2b). A re-centring shift is bounded by 1/8 of the window; a patch whose re-centring
    stops early is skipped, like a patch the first measurement rejects.

    Parameters
    ----------
    img_path1 : str
        Path of the reference IMG file.
    img_path2 : str
        Path of the repeat IMG file.
    patches : list of dict
        The xcorr patches of this batch, each a dict as described above.
    patch_size : int
        Edge length of the square correlation window in pixels.
    min_response : float
        Minimum correlation response threshold. Default 0.2 matches GMTSAR's SNR=20.
    passes : int
        Number of re-centring passes per accepted patch (default XCORR_RECENTRE_PASSES, 3).

    Returns
    -------
    tuple
        (results, stops): the accepted patches as dicts cy1, cx1, dy, dx, response, and the counts of the patches skipped
        because their re-centring stopped early ({'gate': n, 'shift': n}).

    Examples
    --------
    >>> _xcorr_batch('AL2_129_4450_F4_20150404_HH.img', 'AL2_129_4450_F4_20150516_HH.img',
    ...              [{'cy1': 1000, 'cx1': 1000, 'cy2': 1000, 'cx2': 1000, 'frac_a': 0.0, 'frac_r': 0.0}], 256)
    """
    import numpy as np
    from insardev_toolkit.utils_alos2_ceos import DESCRIPTOR, PREFIX
    from .utils_alos2 import alos2_image

    half = patch_size // 2
    hann = np.outer(np.hanning(patch_size), np.hanning(patch_size)).astype(np.float32)
    bound = patch_size // 8

    info1, info2 = alos2_image(img_path1), alos2_image(img_path2)
    records1 = np.memmap(img_path1, dtype=np.uint8, mode='r', offset=DESCRIPTOR,
                         shape=(info1['lines'], info1['record_length']))
    records2 = np.memmap(img_path2, dtype=np.uint8, mode='r', offset=DESCRIPTOR,
                         shape=(info2['lines'], info2['record_length']))

    results = []
    stops = {'gate': 0, 'shift': 0}
    for p in patches:
        cy1, cx1 = p['cy1'], p['cx1']
        patch1, patch2 = [records[cy - half:cy + half, PREFIX + 8 * (cx - half):PREFIX + 8 * (cx + half)]
                          .view('>c8').astype(np.complex64)
                          for records, cy, cx in ((records1, cy1, cx1), (records2, p['cy2'], p['cx2']))]
        # a non-finite pixel is no-data like a zero one (normal radar data)
        if not np.isfinite(patch1).all():
            patch1 = np.where(np.isfinite(patch1), patch1, 0).astype(np.complex64)
        if not np.isfinite(patch2).all():
            patch2 = np.where(np.isfinite(patch2), patch2, 0).astype(np.complex64)

        first = _xcorr_pc(patch1, patch2, hann, min_response)
        if first is None:
            continue
        # fdc 0.0: the ALOS-2 azimuth band is centred at 0.016 cycles per line (m2b)
        (dy, dx, response), stop = _xcorr_recentre(patch1, patch2, first, 0.0, hann, min_response, passes, bound)
        if stop is not None:
            stops[stop] += 1
            continue
        # phaseCorrelate "finds" the sub-pixel lost by the int() truncation: subtract it to get the true residual
        results.append({'cy1': cy1, 'cx1': cx1, 'dy': dy - p.get('frac_a', 0.0), 'dx': dx - p.get('frac_r', 0.0),
                        'response': response})
    del records1, records2
    return results, stops


class Alos2_align(Alos2_slc):
    """
    ALOS-2 alignment: the reference scene (align_ref) and the repeat scenes (align_rep, the geometry plus the xcorr refinement).

    The ScanSAR L1.1 subswath is a zero-Doppler full-aperture SLC, processed as stripmap: no TOPS deramp/reramp.
    """

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
            Third element (reramp_params) is always None for ALOS-2 (no TOPS deramp).

        Examples
        --------
        >>> alos2.align_ref('AL2_129_4450_F4_20150404_HH', return_slc=False)
        """
        if return_slc:
            prm_dict, orbit_df, slc_data, _ = self._make_scene(scene, mode=2)
            prm = PRM()
            prm.set(**prm_dict)
            prm.orbit_df = orbit_df
            prm.calc_dop_orb(inplace=True, debug=debug)
            return prm, slc_data, None  # No reramp_params for ALOS-2

        prm, orbit_df = self._make_scene(scene, mode=0, debug=debug)
        prm.calc_dop_orb(inplace=True, debug=debug)
        return prm, None, None  # No slc_data, no reramp_params for ALOS-2

    def align_rep(self, scene_rep: str, scene_ref: str, prm_ref: "PRM",
                  degrees: float = 12.0 / 3600, debug: bool = False,
                  return_slc: bool = True,
                  xcorr: tuple | int | None = (256, 256), n_jobs: int | None = None,
                  xcorr_min_response: float = 0.2) -> tuple:
        """
        Process and align a repeat scene to the reference.

        The ScanSAR subswath is aligned as stripmap (no deramp/reramp): the geometry offsets of the DEM points
        (SAT_llt2rat of both dates, the bilinear fit), then the xcorr refinement of the residual. Returns the PRM with the
        alignment offsets.

        Parameters
        ----------
        scene_rep : str
            Repeat scene identifier.
        scene_ref : str
            Reference scene identifier (used for DEM geometry and the xcorr).
        prm_ref : PRM
            Reference PRM object (from align_ref).
        degrees : float, optional
            Degrees per pixel resolution for the coarse DEM.
        debug : bool, optional
            Enable debug mode.
        return_slc : bool, optional
            If True, load and return SLC data. If False, only return PRM.
        xcorr : tuple, int or None, optional
            Xcorr window (square patch) in pixels, as (height, width) or one int; the height sets the size.
            Default (256, 256). The grid covers the whole reference IMG file, every other window on each axis
            (_xcorr_grid); each accepted patch is re-centred 3 times, and a patch whose re-centring stops early is
            skipped. The correction is a constant shift (rank 1). When the area is too small for the window (fewer than
            8 accepted patches), a RuntimeError asks for a smaller window, e.g. xcorr=192 or xcorr=128. None disables
            the refinement: the geometry alone.
        n_jobs : int or None, optional
            Number of parallel workers of the geometry offsets (SAT_llt2rat) and the xcorr.
            None or -1 (default): all cores.
        xcorr_min_response : float, optional
            Minimum correlation response threshold. Default 0.2 matches GMTSAR's SNR=20.

        Returns
        -------
        tuple
            (prm, slc_data, None); slc_data is None if return_slc=False.

        Examples
        --------
        >>> prm_ref = alos2.align_ref('AL2_129_4450_F4_20150404_HH', return_slc=False)[0]
        >>> alos2.align_rep('AL2_129_4450_F4_20150516_HH', 'AL2_129_4450_F4_20150404_HH', prm_ref, return_slc=False)
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
            prm_ref.sel('clock_start', 'clock_stop')
            + nl / prf / 86400.0
        )
        prm_ref_shifted.calc_dop_orb(earth_radius, inplace=True, debug=debug)

        # Compute offset from reference to secondary
        tmpm_dat = prm_ref_shifted.SAT_llt2rat(coords=topo_llt, precise=1, n_jobs=n_jobs, debug=debug)

        prm_rep.calc_dop_orb(earth_radius, inplace=True, debug=debug)
        tmp1_dat = prm_rep.SAT_llt2rat(coords=topo_llt, precise=1, n_jobs=n_jobs, debug=debug)

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

        # Filter to points inside valid radar extent, with a finite offset
        from .utils_satellite import offset_valid_mask
        valid_mask = offset_valid_mask(offset_dat, rmax, amax)
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
                n_jobs=n_jobs,
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

        # the idle joblib workers of the SAT_llt2rat and xcorr steps keep their memory: stop them
        release_workers()

        return prm_rep, slc_data, None  # No reramp_params for ALOS-2

    def _xcorr_refine(self, scene_ref: str, scene_rep: str, prm_rep: "PRM",
                      patch_size: int = 256,
                      n_jobs: int | None = None, min_response: float = 0.2,
                      rank: int = 1, debug: bool = False) -> dict:
        """
        Measure xcorr offsets and fit the correction to the geometry alignment.

        The patches cover the whole reference IMG file, every other window on each axis (_xcorr_grid). Every patch
        accepted by cv2.phaseCorrelate is re-centred 3 times (Nisar_align._xcorr_recentre); a patch whose re-centring stops
        early is skipped (with debug=True the skipped counts are printed). The patches are processed in (line, pixel)
        order, in batches of consecutive patches, each worker reading the two IMG files through one memmap each
        (_xcorr_batch). The fit (utils_satellite.xcorr_fitoffset) is a constant shift by default (rank 1).

        Parameters
        ----------
        scene_ref : str
            Reference scene identifier.
        scene_rep : str
            Repeat scene identifier.
        prm_rep : PRM
            Repeat scene PRM with geometry alignment parameters.
        patch_size : int, optional
            Patch size in pixels. Default 256.
        n_jobs : int or None, optional
            Number of parallel workers. None or -1 (default): all cores.
        min_response : float, optional
            Minimum correlation response threshold. Default 0.2 matches GMTSAR's SNR=20.
        rank : int, optional
            Terms of the fitted model per direction (xcorr_fitoffset): 1 a constant shift (default), 3 bilinear.
        debug : bool, optional
            Print debug information.

        Returns
        -------
        dict
            The correction of the geometry offsets: ashift, stretch_a, a_stretch_a, rshift, stretch_r, a_stretch_r
            (the stretches are 0 with rank 1).

        Examples
        --------
        >>> alos2._xcorr_refine('AL2_129_4450_F4_20150404_HH', 'AL2_129_4450_F4_20150516_HH', prm_rep)
        """
        import os
        import time as _time
        from joblib import Parallel, delayed
        from .utils_alos2 import alos2_image
        from .utils_satellite import xcorr_fitoffset, XCORR_MIN_PATCHES

        if n_jobs is None or n_jobs == -1:
            n_jobs = os.cpu_count()

        img_path1 = self.get_record(scene_ref)['path'].iloc[0]
        img_path2 = self.get_record(scene_rep)['path'].iloc[0]
        info1, info2 = alos2_image(img_path1), alos2_image(img_path2)
        ny1, nx1 = info1['lines'], info1['pixels']
        ny2, nx2 = info2['lines'], info2['pixels']

        n_rows, n_cols = grid = _xcorr_grid((ny1, nx1), patch_size)

        # Get geometry from prm_rep (computed by fitoffset)
        ashift = prm_rep.get('ashift') + prm_rep.get('sub_int_a')
        rshift = prm_rep.get('rshift') + prm_rep.get('sub_int_r')
        stretch_a = prm_rep.get('stretch_a')
        stretch_r = prm_rep.get('stretch_r')
        a_stretch_a = prm_rep.get('a_stretch_a')
        a_stretch_r = prm_rep.get('a_stretch_r')

        if debug:
            print(f"Xcorr refinement: {grid[0]}×{grid[1]} = {grid[0]*grid[1]} patches: rank {rank} fit")
            print(f"Geometry: ashift={ashift:.2f}, rshift={rshift:.2f}")

        # Generate patch grid
        half = patch_size // 2
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

        if len(patches) < XCORR_MIN_PATCHES:
            raise _xcorr_area_error(scene_rep, patch_size,
                                    f'{len(patches)} of the {n_rows}x{n_cols} grid patches fit inside both SLCs '
                                    f'({ny1}x{nx1} and {ny2}x{nx2})')

        # in line order: a batch of consecutive patches reads a band of lines of both files
        patches.sort(key=lambda p: (p['cy1'], p['cx1']))
        n_batches = min(n_jobs, len(patches))
        batch_size = (len(patches) + n_batches - 1) // n_batches
        batches = [patches[i:i+batch_size] for i in range(0, len(patches), batch_size)]

        _t0_xcorr = _time.perf_counter()
        batch_results = Parallel(n_jobs=n_batches)(
            delayed(_xcorr_batch)(img_path1, img_path2, batch, patch_size, min_response)
            for batch in batches
        )
        results = [r for batch, _ in batch_results for r in batch]
        stops = {k: sum(s[k] for _, s in batch_results) for k in ('gate', 'shift')}
        _t1_xcorr = _time.perf_counter()

        if debug:
            print(f"PROFILE: xcorr Parallel ({n_batches} batches, {len(patches)} patches) {_t1_xcorr - _t0_xcorr:.3f}s")
            print(f"Xcorr results: {len(results)} with response > {min_response}; skipped, re-centring stopped: "
                  f"{stops['gate']} pass not accepted, {stops['shift']} offset beyond {patch_size // 8} px")

        corrections = xcorr_fitoffset(results, nx=nx1, ny=ny1, rank=rank, debug=debug)

        if corrections is None and len(results) < XCORR_MIN_PATCHES:
            raise _xcorr_area_error(scene_rep, patch_size,
                                    f'{len(results)} of the {len(patches)} patches inside both SLCs are accepted '
                                    f'(response > {min_response})')
        if corrections is None:
            raise RuntimeError("Xcorr fitoffset failed - insufficient valid patches. Scene alignment cannot continue.")

        if debug:
            print(f"Xcorr fitoffset result:")
            print(f"  ashift={corrections['ashift']:.4f}, stretch_a={corrections['stretch_a']:.8f}, a_stretch_a={corrections['a_stretch_a']:.8f}")
            print(f"  rshift={corrections['rshift']:.4f}, stretch_r={corrections['stretch_r']:.8f}, a_stretch_r={corrections['a_stretch_r']:.8f}")

        return corrections
