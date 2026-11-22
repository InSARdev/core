# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .Alos2_align import Alos2_align
# the half window of the closest-approach search: half the record's own duration (decision D3), as a Sentinel-1 burst
from .utils_satellite import s1_burst_half_window as _record_half_window
# the split of an output chunk into blocks by the radar box of their pixels (shared with insardev_ecef)
from insardev_toolkit.utils_chunks import radar_boxes as _radar_boxes
from insardev_toolkit.utils_pool import release_workers


def _date_groups(items: list, records) -> list:
    """
    The items (sceneId, polarization, scene) of a scene grouped by date and co-pol, the co-pol first.

    The items of one acquisition (the startTime of their record) share one alignment: the co-pol item (HH, VV) defines it and
    the cross-pol item of its transmit polarization (HV with HH, VH with VV) is resampled on it. A cross-pol item without its
    co-pol on that date forms a group of its own (it aligns itself). The groups come in the order of first appearance.

    Parameters
    ----------
    items : list of tuple
        The record indices (sceneId, polarization, scene), as get_repref() gives them.
    records : pandas.DataFrame
        The records (the scan's df): their startTime by the scene (index level 2).

    Returns
    -------
    list of list of tuple
        The groups, each the co-pol item first.

    Examples
    --------
    >>> import pandas as pd
    >>> records = pd.DataFrame({'startTime': [pd.Timestamp('2015-04-04 15:56:18')] * 2},
    ...                        index=pd.MultiIndex.from_tuples([('s', 'HH', 's_HH'), ('s', 'HV', 's_HV')]))
    >>> _date_groups([('s', 'HV', 's_HV'), ('s', 'HH', 's_HH')], records)
    [[('s', 'HH', 's_HH'), ('s', 'HV', 's_HV')]]
    """
    scenes = records.index.get_level_values(2)
    dates = {}
    for item in items:
        dates.setdefault(records.startTime[scenes == item[-1]].iloc[0], []).append(item)
    groups = {}
    for start, date_items in dates.items():
        held = {item[1] for item in date_items}
        for item in date_items:
            copol = item[1][0] * 2
            groups.setdefault((start, copol if copol in held else item[1]), []).append(item)
    return [sorted(group, key=lambda item: item[1][0] != item[1][1]) for group in groups.values()]


def _process_chunk_alos2_worker(args):
    """
    Wrapper for ProcessPoolExecutor - unpacks tuple and calls _process_chunk_alos2.

    Parameters
    ----------
    args : tuple
        The arguments of _process_chunk_alos2.

    Examples
    --------
    >>> _process_chunk_alos2_worker(chunk_args[0])
    """
    return _process_chunk_alos2(*args)


def _process_chunk_alos2(iy, ix, chunk_y, chunk_x, n_y, n_x,
                         outdir, pols,
                         alignment_params, tidal_dt,
                         prm_rep_dict, prm_ref_dict,
                         baseline_params, sc_height_params,
                         num_lines, num_rng_bins,
                         scale, fill_value,
                         remove_topo_phase, topo_path=None, orbit_ref_dict=None,
                         reference_height=0.0, radar_shape=None, tide_nodes=None):
    """
    Process a single output chunk of every polarization of one date - designed for parallel execution.

    All parameters are serializable (no xarray/PRM objects) for ProcessPoolExecutor. Each worker exits after one chunk
    (max_tasks_per_child=1), releasing all memory. The polarizations of the date share one alignment (the co-pol's): every
    block reads the box of each polarization from its IMG file and remaps it with the same maps, and the flat-earth, topo and
    tide phase of a block is computed once and applied to every polarization. tide_nodes are the tidal phase nodes of
    tidal_phase_nodes() (None: no tide) of the whole scene, so the chunk's values do not depend on the chunk size.

    Parameters
    ----------
    iy, ix : int
        The first output row and column of the chunk.
    chunk_y, chunk_x : int
        The processing chunk size.
    n_y, n_x : int
        The output grid size.
    outdir : str
        The scene's output directory (its precise transform).
    pols : list of tuple
        (img_path, zarr_path) of every polarization, the co-pol first.
    alignment_params : tuple or None
        (rshift, ashift, stretch_r, a_stretch_r, stretch_a, a_stretch_a) of the repeat date, None for the reference.
    tidal_dt : tuple or None
        (reference, repeat) datetimes of the tide, None without it.
    prm_rep_dict, prm_ref_dict : dict
        The PRMs of the date (its co-pol) and of the reference.
    baseline_params, sc_height_params : dict or None
        The baselines (None on the reference date: no phase) and the heights of the reference.
    num_lines, num_rng_bins : int
        The SLC size.
    scale : float
        The int16 scale (value = int16 x scale).
    fill_value : int
        The int16 fill value.
    remove_topo_phase : bool
        Remove the topographic phase (else the flat earth at reference_height).
    topo_path : str, optional
        The radar-coordinate topo (remove_topo_phase=True).
    orbit_ref_dict : dict, optional
        The reference orbit (the tide and the reference surface).
    reference_height : float, optional
        The height of the WGS84 ellipsoid of the flat earth (remove_topo_phase=False).
    radar_shape : tuple, optional
        The radar grid of compute_conversion_chunked() (remove_topo_phase=False).
    tide_nodes : optional
        The tidal phase nodes of the whole radar grid.

    Examples
    --------
    >>> _process_chunk_alos2(*chunk_args[0])
    """
    import numpy as np
    import cv2
    import zarr
    import os
    from .utils_alos2 import alos2_slc
    from .utils_satellite import precise_transform_dir

    jy = min(iy + chunk_y, n_y)
    jx = min(ix + chunk_x, n_x)

    # Load ONLY the chunk we need directly from zarr (not full arrays via xarray!): the precise transform of
    # compute_conversion_chunked (float32, NaN outside the swath), not its copy rounded for the stack
    precise_path = precise_transform_dir(outdir)
    trans_root = zarr.open_group(zarr.storage.LocalStore(precise_path), mode='r')
    azi_chunk = trans_root['azi'][iy:jy, ix:jx]
    rng_chunk = trans_root['rng'][iy:jy, ix:jx]
    del trans_root

    # Apply alignment offsets for repeat scenes
    # Use original azi/rng in both equations (bilinear model requires original coords)
    if alignment_params is not None:
        rshift, ashift, stretch_r, a_stretch_r, stretch_a, a_stretch_a = alignment_params
        azi_orig, rng_orig = azi_chunk.copy(), rng_chunk.copy()
        rng_chunk = (rng_orig + rshift + stretch_r * rng_orig + a_stretch_r * azi_orig).astype(np.float32)
        azi_chunk = (azi_orig + ashift + stretch_a * rng_orig + a_stretch_a * azi_orig).astype(np.float32)
        del azi_orig, rng_orig

    # The pixels with radar coordinates; nothing to read for an empty chunk
    valid_mask = np.isfinite(azi_chunk) & np.isfinite(rng_chunk)
    if not valid_mask.any():
        return  # Empty chunk

    # No demodulation: the ALOS-2 azimuth spectrum is centred at 0.016 cycles per line in all range blocks, 0.7-0.8 of the
    # PRF wide, and demodulating by that centroid does not change the LANCZOS4 interpolation error (median 0.04-0.11 rad;
    # measured, m2b): unlike NISAR (about 0.63 cycles per line), the band already sits about 0

    # Geocode the SLC block by block of the output chunk, each with the SLC box of its pixels plus the margin for the
    # interpolation (at most as many cells as the chunk has pixels, as the topo below): the box of a whole chunk can
    # span most of the swath, and its SLC copies of every parallel chunk worker would not fit in memory. Every
    # polarization is read and remapped with the same maps
    margin = 10  # pixels margin for interpolation
    proj_re = [np.full(azi_chunk.shape, np.nan, dtype=np.float32) for _ in pols]
    proj_im = [np.full(azi_chunk.shape, np.nan, dtype=np.float32) for _ in pols]
    for by0, by1, bx0, bx1, box_a0, box_a1, box_r0, box_r1 in _radar_boxes(
            azi_chunk, rng_chunk, valid_mask, num_lines, num_rng_bins, chunk_y * chunk_x):
        azi_min, azi_max = max(0, box_a0 - margin), min(num_lines, box_a1 + margin)
        rng_min, rng_max = max(0, box_r0 - margin), min(num_rng_bins, box_r1 + margin)
        if azi_max <= azi_min or rng_max <= rng_min:
            continue  # the block's pixels are outside of the scene: they stay NaN

        # Local coordinates in the SLC box, the maps of cv2.remap: the transform's azi/rng are the 0-based
        # pixel-centre line and bin, the coordinates cv2.remap reads, so no 0.5 shift (float32 minus a whole line or
        # bin: exact)
        azi_local = azi_chunk[by0:by1, bx0:bx1] - azi_min
        rng_local = rng_chunk[by0:by1, bx0:bx1] - rng_min

        for k, (img_path, _) in enumerate(pols):
            # Read the block's SLC box of the polarization from its IMG file
            slc_chunk = alos2_slc(img_path, row_slice=slice(azi_min, azi_max), col_slice=slice(rng_min, rng_max))
            slc_re = slc_chunk.real.astype(np.float32)
            slc_im = slc_chunk.imag.astype(np.float32)
            del slc_chunk

            proj_re[k][by0:by1, bx0:bx1] = cv2.remap(slc_re, rng_local, azi_local,
                                                     interpolation=cv2.INTER_LANCZOS4,
                                                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            proj_im[k][by0:by1, bx0:bx1] = cv2.remap(slc_im, rng_local, azi_local,
                                                     interpolation=cv2.INTER_LANCZOS4,
                                                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            del slc_re, slc_im
        del azi_local, rng_local
    del azi_chunk, rng_chunk

    # cv2.remap doesn't reliably produce NaN when map values are NaN
    # Explicitly mask pixels where transform was fill (azi/rng was NaN)
    for k in range(len(pols)):
        proj_re[k][~valid_mask] = np.nan
        proj_im[k][~valid_mask] = np.nan
    del valid_mask

    # Apply topo/tidal phase if needed: the flat-earth and topographic phase of flat_earth_topo_phase() and the
    # tide of tidal_phase_radar() on the radar-coordinate topo, geocoded (linear) at each pixel's REFERENCE radar
    # coordinates -- the transform, not the repeat coordinates of the alignment above -- like S1. Block by block of
    # the output chunk, each with the radar box of its pixels (at most as many cells as the chunk has pixels), after
    # the SLC arrays are gone: the box of a whole chunk spans the full swath and would not fit in memory. The phase of a
    # block is computed once and applied to every polarization.
    # remove_topo_phase=False removes the flat earth alone, like S1: the topo is the WGS84 ellipsoid at
    # reference_height of reference_surface_topo(), its nodes on the whole radar grid (radar_shape), so each
    # block holds the whole-grid values; the tide as well
    # Skip for ref bursts: baseline_params=None → drho≈0 (no-op, avoids FP noise)
    if baseline_params is not None:
        from .utils_satellite import flat_earth_topo_phase, tidal_phase_radar
        from .PRM import PRM
        import xarray as xr

        if remove_topo_phase:
            # Load the topo from zarr, block by block below
            if topo_path is None or not os.path.exists(topo_path):
                raise FileNotFoundError(f'Topo not found: {topo_path}')
            topo_store = zarr.storage.LocalStore(topo_path)
            topo_root = zarr.open_group(topo_store, mode='r')

            # Get topo dimensions and scaling
            topo_n_azi = topo_root['topo'].shape[0]
            topo_n_rng = topo_root['topo'].shape[1]
            topo_scale = topo_root['topo'].attrs.get('scale_factor', 1.0)
            topo_fill = topo_root['topo'].attrs.get('_FillValue', 2147483647)
        else:
            # No topo: the radar grid of compute_conversion_chunked(), cell k centred on coordinate k (the 0-based
            # line and bin)
            from .utils_satellite import reference_surface_topo
            topo_store = topo_root = None
            topo_n_azi, topo_n_rng = radar_shape
            grid_a = np.arange(topo_n_azi, dtype=np.float64)
            grid_r = np.arange(topo_n_rng, dtype=np.float64)

        # Reconstruct PRM objects from dicts; the reference one with its orbit for the tide
        prm_rep = PRM()
        prm_rep.set(**prm_rep_dict)
        prm_ref = PRM()
        prm_ref.set(**prm_ref_dict)
        if orbit_ref_dict is not None:
            import pandas as pd
            prm_ref.orbit_df = pd.DataFrame(orbit_ref_dict)

        # Reference radar coordinates of the chunk, read again from the precise transform now that the SLC arrays
        # are gone
        trans_root = zarr.open_group(zarr.storage.LocalStore(precise_path), mode='r')
        azi_ref = trans_root['azi'][iy:jy, ix:jx]
        rng_ref = trans_root['rng'][iy:jy, ix:jx]
        del trans_root
        # the pixels with a value (the polarizations share the maps, so the co-pol's mask is every polarization's)
        need = np.isfinite(azi_ref) & np.isfinite(rng_ref) & np.isfinite(proj_re[0])

        for by0, by1, bx0, bx1, topo_azi_min, topo_azi_max, topo_rng_min, topo_rng_max in _radar_boxes(
                azi_ref, rng_ref, need, topo_n_azi, topo_n_rng, chunk_y * chunk_x):
            if topo_root is not None:
                # Read only the block we need with proper fill handling
                topo_raw = topo_root['topo'][topo_azi_min:topo_azi_max, topo_rng_min:topo_rng_max]
                topo_data = np.where(topo_raw == topo_fill, np.nan, topo_raw * topo_scale).astype(np.float32)
                topo_a_coords = topo_root['a'][topo_azi_min:topo_azi_max]
                topo_r_coords = topo_root['r'][topo_rng_min:topo_rng_max]
                del topo_raw

                # Create minimal xarray DataArray for phase computation
                topo_chunk = xr.DataArray(
                    topo_data,
                    dims=['a', 'r'],
                    coords={'a': topo_a_coords, 'r': topo_r_coords}
                )
                del topo_data
            else:
                # The reference surface of the block (reference_surface_topo reads the coordinates only)
                topo_a_coords = grid_a[topo_azi_min:topo_azi_max]
                topo_r_coords = grid_r[topo_rng_min:topo_rng_max]
                topo_chunk = reference_surface_topo(
                    prm_ref, xr.DataArray(np.broadcast_to(np.float32(0), (topo_a_coords.size, topo_r_coords.size)),
                                          dims=['a', 'r'], coords={'a': topo_a_coords, 'r': topo_r_coords}),
                    reference_height, grid=(grid_a, grid_r))

            phase_chunk = flat_earth_topo_phase(topo_chunk, prm_rep, prm_ref,
                                                 baseline_params=baseline_params,
                                                 sc_height_params=sc_height_params)

            if tidal_dt is not None:
                # between the nodes of the whole radar grid, not the corners of the block (chunk-dependent)
                phase_chunk.values += tidal_phase_radar(topo_chunk, prm_ref, tidal_dt, nodes=tide_nodes).values

            # Geocode phase to the output block at the reference coordinates (cell k is centred on a[k])
            phase_local_a = (azi_ref[by0:by1, bx0:bx1] - topo_a_coords[0]).astype(np.float32)
            phase_local_r = (rng_ref[by0:by1, bx0:bx1] - topo_r_coords[0]).astype(np.float32)
            phase_proj = cv2.remap(phase_chunk.values.astype(np.float32),
                                   phase_local_r, phase_local_a,
                                   interpolation=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
            del phase_chunk, phase_local_a, phase_local_r, topo_chunk, topo_a_coords, topo_r_coords

            # Apply the phase correction to every polarization
            cos_phase = np.cos(phase_proj)
            sin_phase = np.sin(phase_proj)
            del phase_proj
            for k in range(len(pols)):
                re_ = proj_re[k][by0:by1, bx0:bx1]
                im_ = proj_im[k][by0:by1, bx0:bx1]
                corrected_re = re_ * cos_phase + im_ * sin_phase
                corrected_im = im_ * cos_phase - re_ * sin_phase
                proj_re[k][by0:by1, bx0:bx1] = corrected_re
                proj_im[k][by0:by1, bx0:bx1] = corrected_im
                del re_, im_, corrected_re, corrected_im
            del cos_phase, sin_phase
        del azi_ref, rng_ref, need, topo_store, topo_root
        if not remove_topo_phase:
            del grid_a, grid_r

    # Convert to int16: a bright sample past the int16 range would wrap around, so its amplitude is clipped
    # keeping the phase (pack_complex_int16), below fill_value so a saturated sample is not read as NaN; each
    # polarization to its own zarr group (thread-safe region writes)
    from .utils_satellite import pack_complex_int16
    for k, (_, zarr_path) in enumerate(pols):
        re_int16, im_int16 = pack_complex_int16(proj_re[k], proj_im[k], scale, fill_value)
        proj_re[k] = proj_im[k] = None
        root = zarr.open_group(store=zarr.storage.LocalStore(zarr_path), zarr_format=3, mode='r+')
        root['re'][iy:jy, ix:jx] = re_int16
        root['im'][iy:jy, ix:jx] = im_int16
        del re_int16, im_int16, root
    del proj_re, proj_im


def _transform_slc_int16_alos2_chunked(outdir, conversion_dir, prm_rep, prm_ref, pols, epsg,
                                        baseline_params=None, sc_height_params=None,
                                        remove_tidal_phase=True,
                                        remove_topo_phase=True,
                                        n_jobs=None, debug=False,
                                        reference_height=0.0):
    """
    Transform the polarizations of one ALOS-2 date to geocoded int16 zarr groups using chunked I/O.

    The polarizations share the date's alignment (prm_rep, the co-pol's): one ProcessPoolExecutor pass over the output chunks
    writes every polarization (_process_chunk_alos2), each to its own group outdir/<scene>. Memory-efficient: suitable for
    limited RAM systems (e.g., 12GB Colab).

    Parameters
    ----------
    outdir : str
        The scene's output directory.
    conversion_dir : str
        The directory of the radar-coordinate topo.
    prm_rep : PRM
        The PRM of the date (its co-pol), with the alignment offsets of a repeat date; prm_ref itself on the reference date.
    prm_ref : PRM
        The reference PRM, with its orbit.
    pols : list of dict
        {pol, scene_name, img_path, record_dict} of every polarization of the date, the co-pol first.
    epsg : int
        The EPSG code of the output.
    baseline_params, sc_height_params : dict, optional
        The baselines of the date (None on the reference date) and the heights of the reference.
    remove_tidal_phase, remove_topo_phase : bool, optional
        As transform().
    n_jobs : int, optional
        Number of parallel chunk workers. None or -1 (default): all cores.
        Each worker holds one processing chunk of every polarization, 2 x 2 zarr chunks of at most 32 MiB of complex64 each
        (grid_chunks): lower n_jobs to use less RAM.
    debug : bool, optional
        Print debug information.
    reference_height : float, optional
        With remove_topo_phase=False: the height of the WGS84 ellipsoid whose flat-earth phase is removed.

    Examples
    --------
    >>> _transform_slc_int16_alos2_chunked(outdir, conversion_dir, prm, prm_ref, pols, 32718)
    """
    import os
    import time
    import numpy as np
    import pandas as pd
    import zarr
    from .PRM import PRM

    _t0_total = time.perf_counter()

    # Handle n_jobs=-1 (use all cores) - joblib convention
    if n_jobs is None or n_jobs == -1:
        n_jobs = os.cpu_count()
    if debug:
        print(f'Chunk parallelization: n_jobs={n_jobs}')

    # Get transform dimensions directly from zarr (no xarray overhead)
    trans_root = zarr.open_group(zarr.storage.LocalStore(os.path.join(outdir, 'transform')), mode='r')
    out_y = trans_root['y'][:]
    out_x = trans_root['x'][:]
    n_y, n_x = len(out_y), len(out_x)
    del trans_root

    # Check if we need merged transform (repeat scene with alignment offsets)
    has_alignment = prm_rep.get('rshift') is not None
    if has_alignment:
        alignment_params = (
            prm_rep.get('rshift') + prm_rep.get('sub_int_r'),
            prm_rep.get('ashift') + prm_rep.get('sub_int_a'),
            prm_rep.get('stretch_r'),
            prm_rep.get('a_stretch_r'),
            prm_rep.get('stretch_a'),
            prm_rep.get('a_stretch_a')
        )
    else:
        alignment_params = None

    # Compute tidal datetime if needed (differential: ref - rep), in both modes like S1
    is_reference = prm_rep is prm_ref
    tidal_dt = None
    if remove_tidal_phase and not is_reference:
        import datetime as _dt
        def _sc_clock_to_dt(prm):
            """
            The middle time of a PRM's scene as a datetime.

            SC_clock is whole seconds (year x 1000 + 0-based day + second / 86400): only the year is read from it; the day and
            the time of day come from clock_start and clock_stop.

            Examples
            --------
            >>> _sc_clock_to_dt(prm_ref)
            """
            year = int(prm.get('SC_clock_start') // 1000)
            mid = (prm.get('clock_start') + prm.get('clock_stop')) / 2.0
            return _dt.datetime(year, 1, 1) + _dt.timedelta(days=mid)
        tidal_dt = (_sc_clock_to_dt(prm_ref), _sc_clock_to_dt(prm_rep))

    # The workers rebuild the PRMs from dicts, without the orbit: tidal_phase_radar() and reference_surface_topo()
    # need the reference one
    orbit_ref_dict = prm_ref.orbit_df.to_dict('list') if tidal_dt is not None or not remove_topo_phase else None
    topo_path = os.path.join(conversion_dir, 'topo')
    # the radar grid of compute_conversion_chunked(), for the reference surface of remove_topo_phase=False
    a_max, r_max = prm_ref.bounds()
    radar_shape = (len(np.arange(0, a_max, 1, dtype=np.float32)), len(np.arange(0, r_max, 1, dtype=np.float32)))
    # The tide of the repeat dates at fixed nodes of that whole radar grid (cell k centred on k, the 0-based line and
    # bin), once per date: every block of every chunk interpolates the same nodes
    tide_nodes = None
    if tidal_dt is not None and baseline_params is not None:
        from .utils_satellite import tidal_phase_nodes
        tide_nodes = tidal_phase_nodes(prm_ref, tidal_dt, (np.arange(radar_shape[0], dtype=np.float64),
                                                           np.arange(radar_shape[1], dtype=np.float64)))

    # value = int16 x 2.0 (measured, m5: |I|, |Q| median 636 HH and 347 HV, max 123029; about 590 and 330 quanta of the
    # median amplitude, and 1.7e-5 of the samples above the int16 range at a scale of 1.0)
    scale = 2.0
    fill_value = np.iinfo(np.int16).max

    # the stored chunks of the S1 rule (zarr_chunk_shape: at most 32 MiB of complex64, about equal) and a processing
    # chunk of 2 x 2 of them (grid_chunks): no small stripe at the end, and the workers write whole zarr chunks
    from insardev_toolkit.utils_chunks import grid_chunks
    zarr_chunks, (chunk_y, chunk_x) = grid_chunks((n_y, n_x))

    # Pre-create the zarr output of every polarization (scene_name may include sceneId prefix, use basename)
    zarr_paths = []
    for item in pols:
        zarr_path = os.path.join(outdir, os.path.basename(item['scene_name']))
        os.makedirs(zarr_path, exist_ok=True)
        root = zarr.group(store=zarr.storage.LocalStore(zarr_path), zarr_format=3, overwrite=True)
        root.create_array('re', shape=(n_y, n_x), chunks=zarr_chunks, dtype=np.int16,
                          fill_value=fill_value, overwrite=True, dimension_names=['y', 'x'])
        root.create_array('im', shape=(n_y, n_x), chunks=zarr_chunks, dtype=np.int16,
                          fill_value=fill_value, overwrite=True, dimension_names=['y', 'x'])
        del root
        zarr_paths.append(zarr_path)
    worker_pols = [(item['img_path'], zarr_path) for item, zarr_path in zip(pols, zarr_paths)]

    # SLC dimensions for bounds checking
    num_lines = prm_rep.get('num_lines')
    num_rng_bins = prm_rep.get('num_rng_bins')

    # Serialize PRM objects to dicts for the workers
    prm_rep_dict = {k: v for k, v in prm_rep.df.itertuples()}
    prm_ref_dict = {k: v for k, v in prm_ref.df.itertuples()}

    # Generate chunk indices
    chunks = [(iy, ix) for iy in range(0, n_y, chunk_y)
                       for ix in range(0, n_x, chunk_x)]
    n_chunks = len(chunks)

    if debug:
        n_chunks_y = (n_y + chunk_y - 1) // chunk_y
        n_chunks_x = (n_x + chunk_x - 1) // chunk_x
        print(f'Processing {n_chunks_y}x{n_chunks_x} = {n_chunks} chunks of {len(pols)} polarizations with n_jobs={n_jobs}')

    # Build argument tuples for ProcessPoolExecutor
    chunk_args = [
        (iy, ix, chunk_y, chunk_x, n_y, n_x,
         outdir, worker_pols,
         alignment_params, tidal_dt,
         prm_rep_dict, prm_ref_dict,
         baseline_params, sc_height_params,
         num_lines, num_rng_bins,
         scale, fill_value,
         remove_topo_phase, topo_path, orbit_ref_dict,
         reference_height, radar_shape, tide_nodes)
        for iy, ix in chunks
    ]

    # Process chunks using subprocess pool with memory isolation
    # Each worker processes one chunk then exits (max_tasks_per_child=1), releasing memory
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    _t0_chunks = time.perf_counter()
    # the joblib workers of the alignment (the SAT_llt2rat and xcorr workers of this date) stay alive in loky's
    # reusable executor and hold memory the chunk workers need: stop them, the next joblib step starts its own
    release_workers()
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context('spawn'),
                             max_tasks_per_child=1) as executor:
        list(executor.map(_process_chunk_alos2_worker, chunk_args))
    if debug:
        print(f'PROFILE: SLC ProcessPoolExecutor ({n_chunks} chunks, {len(pols)} polarizations, {n_jobs} workers) '
              f'{time.perf_counter() - _t0_chunks:.3f}s')

    del chunk_args

    def _convert_value(v):
        """
        Convert numpy types to Python types for JSON serialization.

        Examples
        --------
        >>> _convert_value(np.float64(1.5))
        1.5
        """
        if isinstance(v, (np.integer,)):
            return int(v)
        elif isinstance(v, (np.floating,)):
            return float(v)
        elif isinstance(v, np.ndarray):
            return v.tolist()
        return v

    # the GMTSAR baseline triples (length m, angle to the horizontal deg, at the reference scene start, centre and
    # end) as SAT_baseline gives them, then BPR: one alignment, so the same on every polarization. The reference date's
    # are 0, as its BPR: SAT_baseline of a date with itself is undefined (NaN lengths from the closest-approach fit)
    triples = ('baseline_start', 'baseline_center', 'baseline_end', 'alpha_start', 'alpha_center', 'alpha_end')
    if is_reference:
        BPR = 0.0
        baselines = dict.fromkeys(triples, 0.0)
    else:
        baseline = prm_ref.SAT_baseline(prm_rep, search_half_window=_record_half_window(prm_rep))
        BPR = baseline.get('B_perpendicular')
        baselines = {name: _convert_value(baseline.get(name) + 0) for name in triples}

    # The exact radar extent polygon of the scene (the crop) from prm_ref, as S1 stores it
    from .utils_satellite import radar_extent_polygon
    geometry = radar_extent_polygon(prm_ref)

    from pyproj import CRS
    spatial_ref = CRS.from_epsg(epsg).to_wkt()

    # Reload the coordinates for the groups (read directly from zarr, not xarray)
    trans_root = zarr.open_group(zarr.storage.LocalStore(os.path.join(outdir, 'transform')), mode='r')
    out_y = trans_root['y'][:]
    out_x = trans_root['x'][:]
    del trans_root

    for item, zarr_path in zip(pols, zarr_paths):
        record_dict = item['record_dict']
        # the PRM of the polarization: the date's PRM with its polarization and input file
        prm_pol = PRM(prm_rep)
        prm_pol.set(polarization=item['pol'], input_file=item['img_path'])

        # Add metadata directly to zarr without loading data
        attrs = {}
        # Add PRM attributes first (technical parameters)
        for name, value in prm_pol.df.itertuples():
            if name not in ['input_file', 'SLC_file', 'led_file']:
                attrs[name] = _convert_value(value)
        # the flat-earth reference height for the downstream elevation, the same on every date, and NaN in DEM mode
        # (the grids give a residual height there; the elevation reads NaN as 0); a technical attribute, before BPR,
        # so to_dataframe() does not list it
        attrs['ref_height'] = float('nan') if remove_topo_phase else float(reference_height)
        attrs.update(baselines)
        # BPR is the cutoff point for to_dataframe
        attrs['BPR'] = BPR + 0

        # === User-facing attributes AFTER BPR (for Stack.to_dataframe) ===
        attrs['geometry'] = geometry
        attrs['pathNumber'] = attrs['path']
        attrs['band'] = 'L'
        attrs['mission'] = 'ALOS-2'
        attrs['flightDirection'] = record_dict['flightDirection']
        attrs['lookDirection'] = record_dict['lookDirection']
        attrs['polarization'] = item['pol']
        if 'startTime' in record_dict:
            st = record_dict['startTime']
            if isinstance(st, (pd.Timestamp, np.datetime64)):
                st = pd.Timestamp(st).strftime('%Y-%m-%d %H:%M:%S')
            attrs['startTime'] = st
        # Add burst (short scene name)
        attrs['burst'] = os.path.basename(item['scene_name'])
        # Add fullBurstID (like S1, this should be last for to_dataframe indices)
        attrs['fullBurstID'] = os.path.basename(outdir)
        attrs['spatial_ref'] = spatial_ref

        store = zarr.storage.LocalStore(zarr_path)
        root = zarr.open_group(store=store, zarr_format=3, mode='r+')
        root.attrs.update(attrs)

        y_arr = root.create_array('y', data=out_y.astype(np.float64), chunks=(len(out_y),), overwrite=True,
                                  dimension_names=['y'])
        x_arr = root.create_array('x', data=out_x.astype(np.float64), chunks=(len(out_x),), overwrite=True,
                                  dimension_names=['x'])

        # Add variable attributes
        for name in ('re', 'im'):
            root[name].attrs['scale_factor'] = scale
            root[name].attrs['add_offset'] = 0
            root[name].attrs['_FillValue'] = int(fill_value)
            root[name].attrs['_ARRAY_DIMENSIONS'] = ['y', 'x']
        y_arr.attrs['_ARRAY_DIMENSIONS'] = ['y']
        x_arr.attrs['_ARRAY_DIMENSIONS'] = ['x']

        # Consolidate metadata
        zarr.consolidate_metadata(store)
        del root, store

    if debug:
        print(f'Total time: {time.perf_counter() - _t0_total:.1f}s')


class Alos2_transform(Alos2_align):
    """ALOS-2 transform: the ScanSAR subswath geocoded as stripmap (no reramp), the polarizations of a date on one alignment."""

    def transform(self,
                  target: str,
                  ref: str,
                  epsg: str | int | None = 'auto',
                  resolution: tuple[int, int] = (30, 30),
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
        Transform ALOS-2 ScanSAR subswath SLC data to geographic coordinates.

        The polarizations of a date share ONE alignment: the co-pol (HH) defines the date's alignment (geometry, xcorr,
        tide) and the cross-pol (HV) is resampled on it, both in one pass over the date's chunks. A cross-pol without its
        co-pol on that date forms its own group and aligns itself. The closest-approach search of the baselines uses half the
        record's own duration (a crop does not carry its scene's duration; decision D3), as a Sentinel-1 burst.

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
            The resolution to use in meters per pixel. Default (30, 30): a bit lower than the data's own, for faster
            processing (on Google Colab); the single-look resolution is about 26 m along track and 17 m across on the
            ground.
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
            Not used for ALOS-2 (kept for API compatibility).
        tmpdir : str, optional
            Not used for ALOS-2 (kept for API compatibility).
        debug : bool, optional
            Whether to print debug information.

        Examples
        --------
        >>> alos2.transform('stack', ref='2015-04-04', bbox=(-72.86, -41.51, -72.36, -41.15))
        """
        from tqdm.auto import tqdm
        import os
        import shutil
        import warnings
        from insardev_toolkit.utils_files import exists

        warnings.filterwarnings('ignore', message='.*Consolidated metadata.*', category=UserWarning)

        if chunk is not None:
            print(f'WARNING: chunk={chunk} is ignored: the stored chunks follow the S1 rule (at most 32 MiB '
                  f'of complex64) and a worker processes 2 x 2 of them.')

        # radar-coordinate output (epsg=0) is not supported: geocoding is always enabled
        if epsg is not None and not isinstance(epsg, str) and epsg == 0:
            raise ValueError("ERROR: epsg=0 (radar coordinates) is not supported, geocoding is always enabled. "
                             "Use epsg='auto' (default) or an explicit EPSG code.")

        # Validate reference_height vs remove_topo_phase
        if remove_topo_phase and reference_height is not None:
            raise ValueError("reference_height is only used when remove_topo_phase=False (flat-earth mode for DEM generation)")
        if reference_height is None:
            reference_height = 0.0

        # Control library threading
        # and put back when the transform ends, also on an error (N156)
        thread_vars = ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                       'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']
        thread_vars_before = {var: os.environ.get(var) for var in thread_vars}
        try:
            for var in thread_vars:
                os.environ[var] = '1'

            if self.DEM is None:
                raise ValueError('ERROR: DEM is not set. Please create a new instance with a DEM.')

            if epsg is None:
                print('NOTE: EPSG code will be computed automatically for each scene.')
            elif isinstance(epsg, str) and epsg == 'auto':
                from .utils_satellite import get_utm_epsg
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='.*geographic CRS.*centroid.*')
                    epsgs = self.to_dataframe().centroid.apply(lambda geom: get_utm_epsg(geom.y, geom.x)).unique()
                if len(epsgs) > 1:
                    raise ValueError(f'ERROR: Multiple UTM zones found: {", ".join(map(str, epsgs))}.')
                epsg = epsgs[0]
                print(f'NOTE: EPSG code computed automatically: {epsg}.')

            # Get reference and repeat scenes as groups
            refrep_dict = self.get_repref(ref=ref)
            refreps = [v for v in refrep_dict.values()]

            assert not os.path.exists(target) or os.path.isdir(target)
            # overwrite=False keeps an existing target as it is, with or without append: nothing is modified or processed
            if os.path.exists(target) and not overwrite:
                print(f'NOTE: target exists and overwrite=False: skipping, nothing is modified.')
                return
            # an orbit that cannot be used raises before anything is removed or written
            self._check_orbits(refreps, target, overwrite, append)
            if overwrite and not append and os.path.exists(target):
                print(f'NOTE: Removing all previous results (overwrite=True is the default now; overwrite=False keeps the target).')
                shutil.rmtree(target)

            metafile = os.path.join(target, 'zarr.json')
            if os.path.exists(target):
                # append (overwrite=True) extends the existing target; an empty metadata file raises
                if not exists(metafile, again='run the processing'):
                    print(f'NOTE: target processing is not completed. Continuing...')
            # an empty metadata file of a scene raises before any scene is processed and before anything is removed
            for scene_refs, _ in refreps:
                exists(os.path.join(target, self.sceneId(scene_refs[0][-1]), 'zarr.json'), again='run the processing')
            if os.path.exists(metafile):
                os.remove(metafile)

            def process_scene_sequential(scenes, target, debug=False):
                """
                Process a single scene group with dates processed sequentially, one alignment per date.

                Examples
                --------
                >>> process_scene_sequential(refreps[0], 'stack')
                """
                scene_refs = scenes[0]
                scene_reps = scenes[1]
                sceneId = self.sceneId(scene_refs[0][-1])
                outdir = os.path.join(target, sceneId)
                metafile_scene = os.path.join(outdir, 'zarr.json')

                # Check if already completed
                if os.path.exists(outdir):
                    assert os.path.isdir(outdir)
                    if exists(metafile_scene, again='run the processing'):
                        return
                    else:
                        print(f'NOTE: {sceneId} incomplete. Removing...')
                        shutil.rmtree(outdir)

                # Phase 1: Compute transform - cache PRMs
                prm_cache = {}
                for scene_ref in scene_refs:
                    prm, _, _ = self.align_ref(scene_ref[-1], debug=debug, return_slc=False)
                    prm_cache[scene_ref[-1]] = prm

                # the reference date's co-pol (the first item of its first group): its PRM and its record give the scene's
                # reference transform
                ref_scene_name = _date_groups(scene_refs, self.df)[0][0][-1]
                prm_ref_main = prm_cache[ref_scene_name]

                # Compute transform and topo tile-by-tile, writing directly to zarr
                # Never builds full arrays in memory - suitable for 12GB Colab
                # Workers read DEM chunks from file - no full DEM in memory
                from .utils_satellite import compute_conversion_chunked, get_utm_epsg, precise_transform_dir
                record = self.get_record(ref_scene_name)
                conversion_dir = os.path.join(outdir, 'conversion')
                # epsg=None: the scene's own UTM zone, from the centroid of its reference record
                _centroid = record.geometry.iloc[0].centroid
                scene_epsg = epsg if epsg is not None else get_utm_epsg(_centroid.y, _centroid.x)

                try:
                    compute_conversion_chunked(
                        prm_ref_main, self.DEM, record.geometry.iloc[0], outdir,
                        scale_factor=1 / dem_vertical_accuracy,
                        epsg=scene_epsg, resolution=resolution, bbox=bbox,
                        compute_topo=remove_topo_phase,
                        n_jobs=n_jobs, debug=debug, datum=self.dem_datum()
                    )

                    # Pre-compute SC_height
                    sc_height_cache = {}
                    for scene_ref in scene_refs:
                        scene_ref_name = scene_ref[-1]
                        prm_ref = prm_cache[scene_ref_name]
                        sc_height_result = prm_ref.SAT_baseline(prm_ref, search_half_window=_record_half_window(prm_ref))
                        sc_height_cache[scene_ref_name] = {
                            'SC_height': sc_height_result.get('SC_height'),
                            'SC_height_start': sc_height_result.get('SC_height_start'),
                            'SC_height_end': sc_height_result.get('SC_height_end')
                        }

                    # Phase 2: Process dates sequentially, one alignment per date: a date's co-pol item and its cross-pol
                    # item are one group, the co-pol first (_date_groups); the group's PRM, xcorr and tide are the co-pol's
                    all_dates = _date_groups(scene_reps, self.df) + _date_groups(scene_refs, self.df)
                    for group in all_dates:
                        scene_item = group[0]
                        is_reference = scene_item in scene_refs
                        # the reference item of the group's defining polarization (the xcorr reads it on both dates)
                        scene_ref = [s for s in scene_refs if s[:2] == scene_item[:2]][0]
                        scene_ref_name = scene_ref[-1]
                        scene_name = scene_item[-1]
                        prm_ref = prm_cache[scene_ref_name]

                        # the polarizations of the date: the IMG path and the record of each item
                        pols = []
                        for item in group:
                            rec = self.get_record(item[-1])
                            record_dict = {}
                            record_reset = rec.reset_index()
                            for col in record_reset.columns:
                                val = record_reset[col].iloc[0]
                                record_dict[col] = val.wkt if hasattr(val, 'wkt') else val
                            pols.append(dict(pol=item[1], scene_name=item[-1], img_path=rec['path'].iloc[0],
                                             record_dict=record_dict))

                        if is_reference:
                            prm = prm_ref
                            baseline_params = None
                        else:
                            prm, _, _ = self.align_rep(scene_name, scene_ref_name, prm_ref,
                                                        degrees=alignment_spacing, debug=debug,
                                                        return_slc=False, xcorr=xcorr, n_jobs=n_jobs)
                            baseline_result = prm_ref.SAT_baseline(prm, search_half_window=_record_half_window(prm))
                            baseline_params = {
                                'baseline_start': baseline_result.get('baseline_start'),
                                'baseline_center': baseline_result.get('baseline_center'),
                                'baseline_end': baseline_result.get('baseline_end'),
                                'alpha_start': baseline_result.get('alpha_start'),
                                'alpha_center': baseline_result.get('alpha_center'),
                                'alpha_end': baseline_result.get('alpha_end'),
                                'B_offset_start': baseline_result.get('B_offset_start'),
                                'B_offset_center': baseline_result.get('B_offset_center'),
                                'B_offset_end': baseline_result.get('B_offset_end')
                            }

                        # one call per date: every polarization of the group
                        _transform_slc_int16_alos2_chunked(
                            outdir=outdir, conversion_dir=conversion_dir,
                            prm_rep=prm, prm_ref=prm_ref, pols=pols,
                            epsg=scene_epsg, baseline_params=baseline_params,
                            sc_height_params=sc_height_cache[scene_ref_name],
                            remove_tidal_phase=remove_tidal_phase,
                            remove_topo_phase=remove_topo_phase,
                            n_jobs=n_jobs, debug=debug,
                            reference_height=reference_height
                        )
                finally:
                    # The radar-coordinate topo and the precise transform are temporary: only the dates above read them
                    # (a completed scene is skipped on a rerun or append, an incomplete one is removed and rebuilt with a
                    # new topo and transform), so they are removed when they are done, and after a failure too, with
                    # their conversion directory
                    for temp_dir in (os.path.join(conversion_dir, 'topo'), precise_transform_dir(outdir)):
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)
                    if os.path.isdir(conversion_dir) and not os.listdir(conversion_dir):
                        os.rmdir(conversion_dir)

                # Cleanup and consolidate
                del prm_cache
                self.consolidate_metadata(target, record_id=all_dates[-1][-1][-1])

            n_scenes = len(refreps)
            # the dates of the first scene: its date groups (the polarizations of a date are one group)
            n_dates = len(_date_groups(refreps[0][0], self.df)) + len(_date_groups(refreps[0][1], self.df))

            # n_jobs of every internal step (tiles, alignment and xcorr, chunks); None or -1: all cores
            if n_jobs is None or n_jobs == -1:
                n_jobs = os.cpu_count()
            print(f'NOTE: Processing {n_scenes} scene(s), {n_dates} dates, chunks parallel with n_jobs={n_jobs}')
            for scenes in tqdm(refreps, desc='Transforming SLC...'.ljust(25)):
                process_scene_sequential(scenes, target, debug=debug)

            # Consolidate zarr metadata
            self.consolidate_metadata(target)
            # the idle joblib workers keep their memory: stop them, the next parallel step starts its own
            release_workers()
        finally:
            for var, value in thread_vars_before.items():
                if value is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = value
