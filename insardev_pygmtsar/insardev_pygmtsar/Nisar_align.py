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


# re-centring passes after the first measurement of an accepted xcorr patch
XCORR_RECENTRE_PASSES = 3
# the base xcorr patch spacing per band, in windows: band A every other window, band B touching windows
XCORR_BASE_SPACING = {'A': 2, 'B': 1}
# the HDF5 chunk cache of an xcorr worker, per SLC file: the patches are processed in column bands narrow enough for
# the decoded chunks of one band's patch rows in flight to fit in it
XCORR_CACHE_BYTES = 64 * 2**20


def _xcorr_pc(patch1, patch2, hann, min_response):
    """The xcorr estimator on one patch pair: cv2.phaseCorrelate of the normalised, Hann-windowed amplitudes.

    Returns (dy, dx, response), the shift of the repeat content (content at u of patch1 is at u + (dy, dx) of patch2),
    or None when fewer than half of the pixels are valid (non-zero) in both patches or the response is not above
    min_response (GMTSAR's SNR=20 at 0.2).
    """
    import numpy as np
    import cv2

    # Check valid data
    valid = (patch1 != 0) & (patch2 != 0)
    if valid.sum() < 0.5 * valid.size:
        return None

    # Normalize amplitudes
    amp1 = np.abs(patch1).astype(np.float32)
    amp2 = np.abs(patch2).astype(np.float32)
    amp1_norm = ((amp1 - amp1.mean()) / (amp1.std() + 1e-10)).astype(np.float32)
    amp2_norm = ((amp2 - amp2.mean()) / (amp2.std() + 1e-10)).astype(np.float32)

    # Phase correlation
    (dx, dy), response = cv2.phaseCorrelate(amp1_norm * hann, amp2_norm * hann)
    if response > min_response:
        return dy, dx, response
    return None


def _xcorr_recentre(patch1, patch2, first, fdc, hann, min_response, passes, bound):
    """Re-centre an accepted patch: each pass shifts the repeat patch (already in memory) by minus the current
    estimate and measures the residual with the same estimator (_xcorr_pc); the shifts accumulate.

    The shift is band-limited (FFT): the azimuth frequencies are placed about the repeat's Doppler centroid fdc (cycles
    per line; the RSLC azimuth spectrum is centred there), the range ones at baseband. The FFT shift is circular: the
    ceil(|shift|) rows / columns wrapped around to the opposite edge are zeroed in both patches.

    Returns ((dy, dx, response), stop): the estimate of the last accepted pass and why the passes stopped early: None
    (all passes done), 'gate' (a pass not accepted by _xcorr_pc) or 'shift' (the estimate beyond `bound` pixels on
    an axis, too large for the patch). A patch that stopped early is skipped by _xcorr_batch.
    """
    import math
    import numpy as np

    dy, dx, response = first
    na, nr = patch2.shape
    fa = np.fft.fftfreq(na)
    fa = fdc + (((fa - fdc) + 0.5) % 1.0) - 0.5
    fr = np.fft.fftfreq(nr)
    spectrum = np.fft.fft2(patch2)
    for _ in range(passes):
        if abs(dy) > bound or abs(dx) > bound:
            return (dy, dx, response), 'shift'
        # content at u moves to u - (dy, dx)
        shifted = np.fft.ifft2(spectrum * np.exp(2j * np.pi * fa * dy).astype(spectrum.dtype)[:, None]
                               * np.exp(2j * np.pi * fr * dx).astype(spectrum.dtype)[None, :])
        reference = patch1.copy()
        # content moved up (dy > 0) wraps into the last rows, moved down into the first ones; columns alike
        ka, kr = math.ceil(abs(dy)), math.ceil(abs(dx))
        rows = slice(na - ka, na) if dy > 0 else slice(0, ka)
        cols = slice(nr - kr, nr) if dx > 0 else slice(0, kr)
        for patch in (reference, shifted):
            patch[rows, :] = 0
            patch[:, cols] = 0
        step = _xcorr_pc(reference, shifted, hann, min_response)
        if step is None:
            return (dy, dx, response), 'gate'
        dy, dx, response = dy + step[0], dx + step[1], step[2]
    return (dy, dx, response), None


def _xcorr_read(ds, cy, cx, half):
    """A patch of an SLC dataset; non-finite pixels are no-data like the zero ones (normal radar data)."""
    import numpy as np
    patch = ds[cy-half:cy+half, cx-half:cx+half]
    if not np.isfinite(patch).all():
        patch = np.where(np.isfinite(patch), patch, 0).astype(patch.dtype)
    return patch


def _xcorr_batch(h5_path1, h5_path2, slc_path, patches, patch_size, min_response=0.2, cache=None,
                 passes=XCORR_RECENTRE_PASSES):
    """Process a batch of xcorr patches - single worker, reuses file handles.

    Each patch dict contains:
    - cy1, cx1: reference patch center (integer)
    - cy2, cx2: secondary patch center (integer, truncated from float)
    - frac_a, frac_r: fractional part lost by truncation (to compensate)
    - fdc: the repeat's Doppler centroid at its patch, cycles per line (the re-centring)

    Every patch accepted by the estimator is re-centred `passes` times (_xcorr_recentre); a re-centring shift is
    bounded by 1/8 of the window: its zeroed wrap-around strip then holds about 1 % of the Hann window's weight. A
    patch whose re-centring stops early (a pass not accepted, or a shift beyond the bound) is skipped, like a patch
    the first measurement rejects.

    Parameters
    ----------
    min_response : float
        Minimum correlation response threshold. Default 0.2 matches GMTSAR's SNR=20.
    cache : tuple or None
        ((rdcc_nbytes, rdcc_nslots) of the reference file, the same of the repeat file): the HDF5 chunk cache
        (_xcorr_cache); None keeps the h5py default.

    Returns
    -------
    tuple
        (results, stops): the accepted patches as dicts cy1, cx1, dy, dx, response, and the counts of the patches
        skipped because their re-centring stopped early ({'gate': n, 'shift': n}).
    """
    import numpy as np
    import h5py

    half = patch_size // 2
    hann = np.outer(np.hanning(patch_size), np.hanning(patch_size)).astype(np.float32)
    bound = patch_size // 8
    open_kw = [{}, {}] if cache is None else [dict(rdcc_nbytes=int(b), rdcc_nslots=int(s), rdcc_w0=0.0)
                                              for b, s in cache]

    results = []
    stops = {'gate': 0, 'shift': 0}
    with h5py.File(h5_path1, 'r', **open_kw[0]) as f1, h5py.File(h5_path2, 'r', **open_kw[1]) as f2:
        ds1 = f1[slc_path]
        ds2 = f2[slc_path]

        for p in patches:
            cy1, cx1 = p['cy1'], p['cx1']
            cy2, cx2 = p['cy2'], p['cx2']
            frac_a = p.get('frac_a', 0.0)
            frac_r = p.get('frac_r', 0.0)

            patch1 = _xcorr_read(ds1, cy1, cx1, half)
            patch2 = _xcorr_read(ds2, cy2, cx2, half)

            first = _xcorr_pc(patch1, patch2, hann, min_response)
            if first is None:
                continue
            (dy, dx, response), stop = _xcorr_recentre(patch1, patch2, first, p['fdc'], hann, min_response,
                                                       passes, bound)
            if stop is not None:
                stops[stop] += 1
                continue
            # Compensate for int() truncation - phaseCorrelate "finds" the
            # sub-pixel that was lost, so subtract it to get TRUE residual
            results.append({'cy1': cy1, 'cx1': cx1, 'dy': dy - frac_a, 'dx': dx - frac_r, 'response': response})

    return results, stops


def _xcorr_spacing(patch_size: int, frequency: str, fraction: float) -> int:
    """The xcorr patch spacing on one axis: the band's base spacing (XCORR_BASE_SPACING windows), 2x denser when the
    file covers at most 1/2 of the frame on that axis and 2x denser again at most 1/4, never closer than half the
    window."""
    if frequency not in XCORR_BASE_SPACING:
        raise ValueError(f'Unknown NISAR frequency band {frequency!r}: expected one of {list(XCORR_BASE_SPACING)}')
    spacing = XCORR_BASE_SPACING[frequency] * patch_size
    if fraction <= 0.5:
        spacing //= 2
    if fraction <= 0.25:
        spacing //= 2
    return max(spacing, patch_size // 2)


def _xcorr_grid(shape: tuple, patch_size: int, frequency: str, fractions: tuple) -> tuple:
    """The xcorr grid (rows, columns) over a file of shape (lines, bins) covering fractions (azimuth, range) of its
    frame: n = (size - window) // spacing + 1 per axis, at least 4."""
    return tuple(max(4, (size - patch_size) // _xcorr_spacing(patch_size, frequency, fraction) + 1)
                 for size, fraction in zip(shape, fractions))


def _next_prime(n: int) -> int:
    """The smallest prime >= n."""
    n = max(2, int(n))
    while any(n % d == 0 for d in range(2, int(n ** 0.5) + 1)):
        n += 1
    return n


def _xcorr_cache(shape: tuple, chunks: tuple, itemsize: int, patch_size: int, band_width: int) -> tuple:
    """(rdcc_nbytes, rdcc_nslots) of the HDF5 chunk cache of one SLC file for xcorr patches processed row by row in
    column bands of band_width pixels: room for the chunk rows one patch row touches (a window of patch_size lines,
    plus 1 line of slack) across the band plus the window and one chunk of misalignment on each side, so every chunk
    is decoded once while a band's patches are processed; and a hash table over every chunk index of the file
    (HDF5 hashes (row << bits) ^ column, a prime above that range never collides)."""
    cr, cc = chunks
    n_rows, n_cols = -(-shape[0] // cr), -(-shape[1] // cc)
    rows = min(n_rows, -(-(patch_size + 1) // cr) + 1)
    cols = min(n_cols, -(-(band_width + patch_size) // cc) + 2)
    nbytes = rows * cols * cr * cc * itemsize
    nslots = _next_prime((n_rows << max(n_cols - 1, 0).bit_length()) + 1)
    return nbytes, nslots


def _xcorr_area_error(scene: str, patch_size: int, found: str) -> RuntimeError:
    """The error of an SLC area too small for the xcorr window: fewer accepted patches than the fit needs.

    It suggests the smaller windows 192 and 128 (xcorr=None is no option for NISAR: the geometry alone is not accurate).
    """
    from .utils_satellite import XCORR_MIN_PATCHES
    smaller = [f'xcorr={size}' for size in (192, 128) if size < patch_size]
    advice = f'set a smaller xcorr window, e.g. {" or ".join(smaller)}' if smaller else 'use a larger area'
    return RuntimeError(f'ERROR: the area is too small for the selected xcorr window {patch_size}x{patch_size} '
                        f'({scene}): {found}, at least {XCORR_MIN_PATCHES} are needed. Please {advice}.')


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
                  xcorr: tuple | int | None = (256, 256), n_jobs: int | None = None,
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
        xcorr : tuple, int or None, optional
            Xcorr window (square patch) in pixels, as (height, width) or one int; the height sets the size.
            Default (256, 256). Grid is auto-computed over the whole SLC file: band A every other window, band B
            touching windows, 2x denser on an axis the file covers at most 1/2 of its full frame and 4x at most
            1/4 (never closer than half the window); each accepted patch is re-centred 3 times, and a patch
            whose re-centring stops early is skipped. The correction is bilinear on a full frame (over 1/2 of the
            frame on both axes) and a constant shift on a crop.
            When the SLC area is too small for the window (fewer than 8 accepted patches), a RuntimeError asks
            for a smaller window, e.g. xcorr=192 or xcorr=128. None disables the refinement: the geometry alone
            is not accurate for NISAR.
        n_jobs : int or None, optional
            Number of parallel workers of the geometry offsets (SAT_llt2rat) and the xcorr.
            None or -1 (default): all cores.
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
                      patch_size: int = 256,
                      n_jobs: int | None = None, min_response: float = 0.2,
                      debug: bool = False) -> dict:
        """
        Measure xcorr offsets and fit the correction to the geometry alignment.

        The patches cover the whole reference SLC file. Their spacing per axis is the band's base (band A every other
        window, band B touching windows), 2x denser when the file covers at most 1/2 of its full frame on that axis
        and 2x denser again at most 1/4, never closer than half the window (utils_nisar.nisar_frame_fraction: the
        zeroDopplerTime span against a 35 s frame, the slantRange span against the frame's). Every patch accepted by
        cv2.phaseCorrelate is re-centred 3 times (_xcorr_recentre); a patch whose re-centring stops early is skipped
        (with debug=True the skipped counts are printed). The fit is bilinear (rank 3) on a full frame, a file
        covering more than 1/2 of its frame on both axes, and a constant shift (rank 1) on any crop.

        The workers read the patches of column bands row by row through an HDF5 chunk cache of at most about
        XCORR_CACHE_BYTES per file, so each SLC chunk (512×512) is decoded once per band.

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
        debug : bool, optional
            Print debug information.

        Returns
        -------
        dict
            The correction of the geometry offsets: ashift, stretch_a, a_stretch_a, rshift, stretch_r, a_stretch_r
            (the stretches are 0 with rank 1).
        """
        import os
        import numpy as np
        from joblib import Parallel, delayed
        from .utils_nisar import nisar_frame_fraction, nisar_doppler_centroid

        if n_jobs is None or n_jobs == -1:
            n_jobs = os.cpu_count()

        # Get HDF5 paths and SLC dataset path
        h5_path1 = self._get_h5_path(scene_ref)
        h5_path2 = self._get_h5_path(scene_rep)
        slc_path = self._get_slc_path(scene_ref)

        # Get scene dimensions and chunking
        import h5py
        with h5py.File(h5_path1, 'r') as f:
            ny1, nx1 = f[slc_path].shape
            chunks1, itemsize1 = f[slc_path].chunks, f[slc_path].dtype.itemsize
        with h5py.File(h5_path2, 'r') as f:
            ny2, nx2 = f[slc_path].shape
            chunks2, itemsize2 = f[slc_path].chunks, f[slc_path].dtype.itemsize

        # The reference file's extent against its full frame (the grid and the fit are in its coordinates): the
        # patch density per axis, and the fit rank: bilinear on a full frame, a constant shift on any crop
        fractions = nisar_frame_fraction(h5_path1, self.frequency)
        rank = 3 if min(fractions) > 0.5 else 1
        n_rows, n_cols = grid = _xcorr_grid((ny1, nx1), patch_size, self.frequency, fractions)

        # Get geometry from prm_rep (computed by fitoffset)
        ashift = prm_rep.get('ashift') + prm_rep.get('sub_int_a')
        rshift = prm_rep.get('rshift') + prm_rep.get('sub_int_r')
        stretch_a = prm_rep.get('stretch_a')
        stretch_r = prm_rep.get('stretch_r')
        a_stretch_a = prm_rep.get('a_stretch_a')
        a_stretch_r = prm_rep.get('a_stretch_r')

        if debug:
            print(f"Xcorr refinement: {grid[0]}×{grid[1]} = {grid[0]*grid[1]} patches; the file covers "
                  f"{fractions[0]:.3f} x {fractions[1]:.3f} (azimuth x range) of its frame: rank {rank} fit")
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

        from .utils_satellite import xcorr_fitoffset, XCORR_MIN_PATCHES
        if len(patches) < XCORR_MIN_PATCHES:
            raise _xcorr_area_error(scene_rep, patch_size,
                                    f'{len(patches)} of the {n_rows}x{n_cols} grid patches fit inside both SLCs '
                                    f'({ny1}x{nx1} and {ny2}x{nx2})')

        # The repeat's Doppler centroid at every repeat patch centre (the re-centring), one read of its LUT
        fdc = nisar_doppler_centroid(h5_path2, self.frequency, np.array([p['cy2'] - 0.5 for p in patches]),
                                     np.array([p['cx2'] - 0.5 for p in patches]))
        if not np.all(np.isfinite(fdc)):
            raise ValueError(f'{h5_path2}: the dopplerCentroid LUT of frequency{self.frequency} is not finite at '
                             f'{int(np.sum(~np.isfinite(fdc)))} of the {len(patches)} xcorr patches')
        for p, value in zip(patches, fdc):
            p['fdc'] = float(value)

        # Column bands, each processed row by row, narrow enough for the decoded chunks of its patch rows in flight
        # to fit XCORR_CACHE_BYTES per file (chunked files); the batches are consecutive runs of the bands
        cache, n_bands = None, 1
        if chunks1 is not None and chunks2 is not None:
            chunk_rows = -(-(patch_size + 1) // chunks1[0]) + 1
            budget_cols = XCORR_CACHE_BYTES // (chunk_rows * chunks1[0] * chunks1[1] * itemsize1)
            max_width = max(chunks1[1], (budget_cols - 2) * chunks1[1] - patch_size)
            n_bands = -(-nx1 // max_width)
            band_width = -(-nx1 // n_bands)
            cache = (_xcorr_cache((ny1, nx1), chunks1, itemsize1, patch_size, band_width),
                     _xcorr_cache((ny2, nx2), chunks2, itemsize2, patch_size, band_width))
        patches.sort(key=lambda p: (min(p['cx1'] * n_bands // nx1, n_bands - 1), p['cy1'], p['cx1']))

        # Split patches into batches for parallel processing
        # Each worker processes a batch with reused file handles
        n_batches = min(n_jobs, len(patches))
        batch_size = (len(patches) + n_batches - 1) // n_batches
        batches = [patches[i:i+batch_size] for i in range(0, len(patches), batch_size)]

        # Parallel xcorr (batch processing, workers reuse file handles)
        import time as _time
        _t0_xcorr = _time.perf_counter()
        batch_results = Parallel(n_jobs=n_batches)(
            delayed(_xcorr_batch)(h5_path1, h5_path2, slc_path, batch, patch_size, min_response, cache)
            for batch in batches
        )
        results = [r for batch, _ in batch_results for r in batch]
        stops = {k: sum(s[k] for _, s in batch_results) for k in ('gate', 'shift')}
        _t1_xcorr = _time.perf_counter()

        if debug:
            print(f"PROFILE: xcorr Parallel ({n_batches} batches, {n_bands} column bands, {len(patches)} patches) "
                  f"{_t1_xcorr - _t0_xcorr:.3f}s")
            print(f"Xcorr results: {len(results)} with response > {min_response}; skipped, re-centring stopped: "
                  f"{stops['gate']} pass not accepted, {stops['shift']} offset beyond {patch_size // 8} px")

        # Use shared fitoffset function with full radar extent for normalization
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
