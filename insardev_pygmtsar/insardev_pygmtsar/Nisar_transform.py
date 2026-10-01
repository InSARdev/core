# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .Nisar_align import Nisar_align


def _process_chunk_nisar_worker(args):
    """Wrapper for ProcessPoolExecutor - unpacks tuple and calls _process_chunk_nisar."""
    return _process_chunk_nisar(*args)


def _radar_boxes(azi, rng, mask, n_azi, n_rng, budget, tile=64):
    """Split an output chunk into blocks whose radar-coordinate bounding box holds at most budget cells.

    azi, rng are the reference radar coordinates of the chunk's pixels (cell k of the radar grid is centred on
    coordinate k + 0.5), mask the pixels that need the phase. The extents are taken per tile x tile pixels once; a
    block of tiles is halved along the output axis that makes its larger half's box smaller, until the box fits.
    Returns (y0, y1, x0, x1, a0, a1, r0, r1) per block: the output pixels and the radar cells for linear
    interpolation at them.
    """
    import numpy as np

    n_y, n_x = azi.shape
    ty, tx = -(-n_y // tile), -(-n_x // tile)

    def reduce(arr, fill, op):
        full = np.full((ty * tile, tx * tile), fill, dtype=np.float32)
        full[:n_y, :n_x] = np.where(mask, arr, fill)
        return op(full.reshape(ty, tile, tx, tile), axis=(1, 3))

    a_lo, a_hi = reduce(azi, np.inf, np.min), reduce(azi, -np.inf, np.max)
    r_lo, r_hi = reduce(rng, np.inf, np.min), reduce(rng, -np.inf, np.max)

    def box(t):
        y0, y1, x0, x1 = t
        amin = a_lo[y0:y1, x0:x1].min()
        if not np.isfinite(amin):
            return None
        amax, rmin, rmax = a_hi[y0:y1, x0:x1].max(), r_lo[y0:y1, x0:x1].min(), r_hi[y0:y1, x0:x1].max()
        return (max(0, int(np.floor(amin - 0.5))), min(n_azi, int(np.floor(amax - 0.5)) + 2),
                max(0, int(np.floor(rmin - 0.5))), min(n_rng, int(np.floor(rmax - 0.5)) + 2))

    def cells(b):
        return 0 if b is None else (b[1] - b[0]) * (b[3] - b[2])

    blocks = []
    stack = [((0, ty, 0, tx), box((0, ty, 0, tx)))]
    while stack:
        t, b = stack.pop()
        if b is None:
            continue
        y0, y1, x0, x1 = t
        if cells(b) <= budget or (y1 - y0 < 2 and x1 - x0 < 2):
            blocks.append((y0 * tile, min(y1 * tile, n_y), x0 * tile, min(x1 * tile, n_x)) + b)
            continue
        halves = []
        if y1 - y0 >= 2:
            ym = (y0 + y1) // 2
            halves.append([((y0, ym, x0, x1), box((y0, ym, x0, x1))), ((ym, y1, x0, x1), box((ym, y1, x0, x1)))])
        if x1 - x0 >= 2:
            xm = (x0 + x1) // 2
            halves.append([((y0, y1, x0, xm), box((y0, y1, x0, xm))), ((y0, y1, xm, x1), box((y0, y1, xm, x1)))])
        stack.extend(min(halves, key=lambda h: max(cells(h[0][1]), cells(h[1][1]))))
    return blocks


def _process_chunk_nisar(iy, ix, chunk_y, chunk_x, n_y, n_x,
                         outdir, zarr_path,
                         h5_path, pol, frequency,
                         alignment_params, tidal_dt,
                         prm_rep_dict, prm_ref_dict,
                         baseline_params, sc_height_params,
                         num_lines, num_rng_bins,
                         scale, fill_value,
                         remove_topo_phase, topo_path=None, orbit_ref_dict=None,
                         reference_height=0.0, radar_shape=None, doppler=None, tide_nodes=None):
    """
    Process a single output chunk - designed for parallel execution.

    All parameters are serializable (no xarray/PRM objects) for ProcessPoolExecutor.
    Each worker exits after one chunk (max_tasks_per_child=1), releasing all memory.

    IMPORTANT: Uses zarr directly (not xarray) to avoid loading full arrays.

    doppler is the date's Doppler centroid of the demodulation (cycles per line) and tide_nodes the tidal phase
    nodes of tidal_phase_nodes() (None: no tide), both of the whole scene, so the chunk's values do not depend on
    the chunk size.
    """
    import numpy as np
    import cv2
    import zarr
    import os
    from .utils_nisar import nisar_slc
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

    # Find SLC read bounds (with margin for interpolation)
    valid_mask = np.isfinite(azi_chunk) & np.isfinite(rng_chunk)
    if not valid_mask.any():
        return  # Empty chunk

    margin = 10  # pixels margin for interpolation
    azi_min = max(0, int(np.floor(np.nanmin(azi_chunk))) - margin)
    azi_max = min(num_lines, int(np.ceil(np.nanmax(azi_chunk))) + margin)
    rng_min = max(0, int(np.floor(np.nanmin(rng_chunk))) - margin)
    rng_max = min(num_rng_bins, int(np.ceil(np.nanmax(rng_chunk))) + margin)

    if azi_max <= azi_min or rng_max <= rng_min:
        return  # Empty chunk

    # Read SLC chunk from HDF5
    slc_chunk = nisar_slc(h5_path, pol=pol, frequency=frequency,
                          row_slice=slice(azi_min, azi_max),
                          col_slice=slice(rng_min, rng_max))

    # Local coordinates in the SLC chunk, the maps of cv2.remap: for NISAR the transform's azi/rng are the 0-based
    # pixel-centre line and bin (zeroDopplerTime[i] and slantRange[j] are the centres of line i and bin j), the
    # coordinates cv2.remap reads, so no S1-style 0.5 shift (float32 minus a whole line or bin: exact)
    azi_local = azi_chunk - azi_min
    rng_local = rng_chunk - rng_min
    del azi_chunk, rng_chunk

    # Geocode SLC chunk
    slc_re = slc_chunk.real.astype(np.float32)
    slc_im = slc_chunk.imag.astype(np.float32)
    del slc_chunk

    # The azimuth spectrum of the RSLC is centred on its Doppler centroid f (about 0.63 cycles per line), not on 0,
    # and LANCZOS4 passes only a band around 0: the lines are demodulated with the date's f (doppler, one value for
    # the scene) and the carrier is restored at the output position below. Both carriers are taken at the absolute
    # scene line, and the products are formed part by part in float32, so every sample and output pixel gets the
    # same values in any chunk (an f per chunk and a carrier from the chunk's first line made the SLC depend on the
    # chunk size)
    assert doppler is not None, 'ERROR: the Doppler centroid of the date is required'
    carrier = (2 * np.pi * doppler) * np.arange(azi_min, azi_max, dtype=np.float64)
    cos_k, sin_k = np.cos(carrier).astype(np.float32)[:, None], np.sin(carrier).astype(np.float32)[:, None]
    del carrier
    for r0 in range(0, slc_re.shape[0], 512):
        re_, im_ = slc_re[r0:r0 + 512], slc_im[r0:r0 + 512]
        c_, s_ = cos_k[r0:r0 + 512], sin_k[r0:r0 + 512]
        # times exp(-i carrier): re c + im s, im c - re s, in place (one temporary block at a time)
        t_ = re_ * s_
        re_ *= c_
        re_ += im_ * s_
        im_ *= c_
        im_ -= t_
        del re_, im_, c_, s_, t_
    del cos_k, sin_k

    proj_re = cv2.remap(slc_re, rng_local, azi_local,
                        interpolation=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    proj_im = cv2.remap(slc_im, rng_local, azi_local,
                        interpolation=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    del slc_re, slc_im, rng_local

    # Restore the carrier exp(2 pi i f azi) at the exact output position (the scene line azi_local + azi_min, exact
    # in float64), in row blocks
    for r0 in range(0, proj_re.shape[0], 512):
        phase = (2 * np.pi * doppler) * (azi_local[r0:r0 + 512].astype(np.float64) + azi_min)
        cos_c, sin_c = np.cos(phase).astype(np.float32), np.sin(phase).astype(np.float32)
        re_, im_ = proj_re[r0:r0 + 512], proj_im[r0:r0 + 512]
        re_[:], im_[:] = re_ * cos_c - im_ * sin_c, re_ * sin_c + im_ * cos_c
        del phase, cos_c, sin_c, re_, im_

    # cv2.remap doesn't reliably produce NaN when map values are NaN
    # Explicitly mask pixels where transform was fill (azi/rng was NaN)
    proj_re[~valid_mask] = np.nan
    proj_im[~valid_mask] = np.nan
    del azi_local, valid_mask

    # Apply topo/tidal phase if needed: the flat-earth and topographic phase of flat_earth_topo_phase() and the
    # tide of tidal_phase_radar() on the radar-coordinate topo, geocoded (linear) at each pixel's REFERENCE radar
    # coordinates -- the transform, not the repeat coordinates of the alignment above -- like S1. Block by block of
    # the output chunk, each with the radar box of its pixels (at most an eighth of a chunk of cells), after the SLC
    # arrays are gone: the box of a whole chunk spans the full swath and would not fit in memory.
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
            # No topo: the radar grid of compute_conversion_chunked(), cell k centred on coordinate k + 0.5
            from .utils_satellite import reference_surface_topo
            topo_store = topo_root = None
            topo_n_azi, topo_n_rng = radar_shape
            grid_a = np.arange(topo_n_azi, dtype=np.float64) + 0.5
            grid_r = np.arange(topo_n_rng, dtype=np.float64) + 0.5

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
        need = np.isfinite(azi_ref) & np.isfinite(rng_ref) & np.isfinite(proj_re)

        for by0, by1, bx0, bx1, topo_azi_min, topo_azi_max, topo_rng_min, topo_rng_max in _radar_boxes(
                azi_ref, rng_ref, need, topo_n_azi, topo_n_rng, chunk_y * chunk_x // 8):
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

            # Apply phase correction
            cos_phase = np.cos(phase_proj)
            sin_phase = np.sin(phase_proj)
            del phase_proj

            re_ = proj_re[by0:by1, bx0:bx1]
            im_ = proj_im[by0:by1, bx0:bx1]
            corrected_re = re_ * cos_phase + im_ * sin_phase
            corrected_im = im_ * cos_phase - re_ * sin_phase
            del cos_phase, sin_phase
            proj_re[by0:by1, bx0:bx1] = corrected_re
            proj_im[by0:by1, bx0:bx1] = corrected_im
            del re_, im_, corrected_re, corrected_im
        del azi_ref, rng_ref, need, topo_store, topo_root
        if not remove_topo_phase:
            del grid_a, grid_r

    # Convert to int16: a bright sample past the int16 range would wrap around, so its amplitude is clipped
    # keeping the phase (pack_complex_int16), below fill_value so a saturated sample is not read as NaN
    from .utils_satellite import pack_complex_int16
    re_int16, im_int16 = pack_complex_int16(proj_re, proj_im, scale, fill_value)
    del proj_re, proj_im

    # Write to zarr (thread-safe for region writes)
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, zarr_format=3, mode='r+')
    root['re'][iy:jy, ix:jx] = re_int16
    root['im'][iy:jy, ix:jx] = im_int16
    del re_int16, im_int16


def _transform_slc_int16_nisar_chunked(outdir, conversion_dir, prm_rep, prm_ref,
                                        scene_name, record_dict, epsg,
                                        baseline_params=None, sc_height_params=None,
                                        remove_tidal_phase=True,
                                        remove_topo_phase=True,
                                        h5_path=None, pol=None, frequency=None,
                                        chunk=(8192, 8192), n_jobs=None, debug=False,
                                        reference_height=0.0):
    """
    Transform Nisar SLC to geocoded int16 zarr using chunked I/O.

    Memory-efficient version that processes chunks in parallel with joblib.
    Suitable for large NISAR frequency A data on limited RAM systems (e.g., 12GB Colab).

    Parameters
    ----------
    n_jobs : int, optional
        Number of parallel chunk workers. None or -1 (default): all cores.
        Each worker holds one chunk: lower n_jobs or chunk to use less RAM.
    reference_height : float, optional
        With remove_topo_phase=False: the height of the WGS84 ellipsoid whose flat-earth phase is removed.
    """
    import os
    import time
    import numpy as np
    import pandas as pd
    import zarr
    from insardev_toolkit.datagrid import datagrid

    _t0_total = time.perf_counter()

    # Handle n_jobs=-1 (use all cores) - joblib convention
    if n_jobs is None or n_jobs == -1:
        n_jobs = os.cpu_count()
    if debug:
        print(f'Chunk parallelization: n_jobs={n_jobs}')

    # Get transform dimensions directly from zarr (no xarray overhead)
    trans_path = os.path.join(outdir, 'transform')
    trans_store = zarr.storage.LocalStore(trans_path)
    trans_root = zarr.open_group(trans_store, mode='r')
    out_y = trans_root['y'][:]
    out_x = trans_root['x'][:]
    n_y, n_x = len(out_y), len(out_x)
    # the reference radar extent of the output (the transform's actual_range), for the Doppler centroid below
    azi_range = trans_root['azi'].attrs.get('actual_range')
    rng_range = trans_root['rng'].attrs.get('actual_range')
    del trans_store, trans_root

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

    # The Doppler centroid of the demodulation, one per date: the date's LUT at the centre of the output's radar
    # extent, in this date's coordinates (the alignment of the chunk workers). A value per chunk, at the centre of
    # its SLC box, made the SLC depend on the chunk size
    from .utils_nisar import nisar_doppler_centroid
    if azi_range is not None and rng_range is not None:
        azi_c, rng_c = 0.5 * (azi_range[0] + azi_range[1]), 0.5 * (rng_range[0] + rng_range[1])
    else:
        # no valid output pixel: every chunk is empty, any value does
        azi_c, rng_c = 0.5 * (prm_rep.get('num_lines') - 1), 0.5 * (prm_rep.get('num_rng_bins') - 1)
    if alignment_params is not None:
        rshift, ashift, stretch_r, a_stretch_r, stretch_a, a_stretch_a = alignment_params
        azi_c, rng_c = (azi_c + ashift + stretch_a * rng_c + a_stretch_a * azi_c,
                        rng_c + rshift + stretch_r * rng_c + a_stretch_r * azi_c)
    doppler = nisar_doppler_centroid(h5_path, frequency, azi_c, rng_c)
    if debug:
        print(f'Doppler centroid {doppler:.6f} cycles per line at line {azi_c:.1f}, bin {rng_c:.1f}')

    # Compute tidal datetime if needed (differential: ref - rep), in both modes like S1
    is_reference = prm_rep is prm_ref
    tidal_dt = None
    if remove_tidal_phase and not is_reference:
        import datetime as _dt
        def _sc_clock_to_dt(prm):
            sc_mid = (prm.get('SC_clock_start') + prm.get('SC_clock_stop')) / 2.0
            year = int(sc_mid // 1000)
            doy_frac = sc_mid % 1000
            # SC_clock carries GMTSAR's 0-based day of year: day 0.x is January 1
            return _dt.datetime(year, 1, 1) + _dt.timedelta(days=doy_frac)
        tidal_dt = (_sc_clock_to_dt(prm_ref), _sc_clock_to_dt(prm_rep))

    # The workers rebuild the PRMs from dicts, without the orbit: tidal_phase_radar() and reference_surface_topo()
    # need the reference one
    orbit_ref_dict = prm_ref.orbit_df.to_dict('list') if tidal_dt is not None or not remove_topo_phase else None
    topo_path = os.path.join(conversion_dir, 'topo')
    # the radar grid of compute_conversion_chunked(), for the reference surface of remove_topo_phase=False
    a_max, r_max = prm_ref.bounds()
    radar_shape = (len(np.arange(0.5, a_max, 1, dtype=np.float32)), len(np.arange(0.5, r_max, 1, dtype=np.float32)))
    # The tide of the repeat dates at fixed nodes of that whole radar grid (cell k centred on k + 0.5), once per date:
    # every block of every chunk interpolates the same nodes
    tide_nodes = None
    if tidal_dt is not None and baseline_params is not None:
        from .utils_satellite import tidal_phase_nodes
        tide_nodes = tidal_phase_nodes(prm_ref, tidal_dt, (np.arange(radar_shape[0], dtype=np.float64) + 0.5,
                                                           np.arange(radar_shape[1], dtype=np.float64) + 0.5))

    # Set scale - NISAR L-band has small amplitudes, use 1e-04 like GMTSAR
    # to avoid quantization to zero (with scale=0.5, ~60% of pixels become zero)
    scale = 1e-04
    fill_value = np.iinfo(np.int16).max

    # Pre-create zarr output (scene_name may include sceneId prefix, use basename)
    zarr_path = os.path.join(outdir, os.path.basename(scene_name))
    os.makedirs(zarr_path, exist_ok=True)

    chunk_y, chunk_x = chunk
    zarr_chunks = (min(chunk_y, n_y), min(chunk_x, n_x))
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.group(store=store, zarr_format=3, overwrite=True)

    re_arr = root.create_array('re', shape=(n_y, n_x), chunks=zarr_chunks, dtype=np.int16,
                                fill_value=fill_value, overwrite=True, dimension_names=['y', 'x'])
    im_arr = root.create_array('im', shape=(n_y, n_x), chunks=zarr_chunks, dtype=np.int16,
                                fill_value=fill_value, overwrite=True, dimension_names=['y', 'x'])

    # SLC dimensions for bounds checking
    num_lines = prm_rep.get('num_lines')
    num_rng_bins = prm_rep.get('num_rng_bins')

    # Serialize PRM objects to dicts for joblib
    prm_rep_dict = {k: v for k, v in prm_rep.df.itertuples()}
    prm_ref_dict = {k: v for k, v in prm_ref.df.itertuples()}

    # Generate chunk indices
    chunks = [(iy, ix) for iy in range(0, n_y, chunk_y)
                       for ix in range(0, n_x, chunk_x)]
    n_chunks = len(chunks)

    if debug:
        n_chunks_y = (n_y + chunk_y - 1) // chunk_y
        n_chunks_x = (n_x + chunk_x - 1) // chunk_x
        print(f'Processing {n_chunks_y}x{n_chunks_x} = {n_chunks} chunks with n_jobs={n_jobs}')

    # Build argument tuples for ProcessPoolExecutor
    chunk_args = [
        (iy, ix, chunk_y, chunk_x, n_y, n_x,
         outdir, zarr_path,
         h5_path, pol, frequency,
         alignment_params, tidal_dt,
         prm_rep_dict, prm_ref_dict,
         baseline_params, sc_height_params,
         num_lines, num_rng_bins,
         scale, fill_value,
         remove_topo_phase, topo_path, orbit_ref_dict,
         reference_height, radar_shape, doppler, tide_nodes)
        for iy, ix in chunks
    ]

    # Process chunks using subprocess pool with memory isolation
    # Each worker processes one chunk then exits (max_tasks_per_child=1), releasing memory
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    _t0_chunks = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context('spawn'),
                             max_tasks_per_child=1) as executor:
        list(executor.map(_process_chunk_nisar_worker, chunk_args))
    if debug:
        print(f'PROFILE: SLC ProcessPoolExecutor ({n_chunks} chunks, {n_jobs} workers) {time.perf_counter() - _t0_chunks:.3f}s')

    del chunk_args

    # Add metadata directly to zarr without loading data
    attrs = {}

    def _convert_value(v):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(v, (np.integer,)):
            return int(v)
        elif isinstance(v, (np.floating,)):
            return float(v)
        elif isinstance(v, np.ndarray):
            return v.tolist()
        return v

    # Add PRM attributes first (technical parameters)
    for name, value in prm_rep.df.itertuples():
        if name not in ['input_file', 'SLC_file', 'led_file']:
            attrs[name] = _convert_value(value)
    # the flat-earth reference height for the downstream elevation, the same on every date, and NaN in DEM mode
    # (the grids give a residual height there; the elevation reads NaN as 0); a technical attribute, before BPR,
    # so to_dataframe() does not list it
    attrs['ref_height'] = float('nan') if remove_topo_phase else float(reference_height)

    # Add baseline BPR (this is the cutoff point for to_dataframe)
    if prm_rep is prm_ref:
        BPR = 0.0
    else:
        baseline = prm_ref.SAT_baseline(prm_rep)
        BPR = baseline.get('B_perpendicular')
    attrs['BPR'] = BPR + 0

    # === User-facing attributes AFTER BPR (for Stack.to_dataframe) ===
    # These appear in to_dataframe() output after reversal

    # Add geometry from record
    if 'geometry' in record_dict:
        geom_val = record_dict['geometry']
        attrs['geometry'] = geom_val.wkt if hasattr(geom_val, 'wkt') else str(geom_val)

    # Add pathNumber (from track)
    if 'track' in attrs:
        attrs['pathNumber'] = attrs['track']

    # Band: L-band (LSAR paths are hardcoded in Nisar_slc)
    attrs['band'] = 'L'

    # Add mission
    attrs['mission'] = 'NISAR'

    # Add frequency
    attrs['frequency'] = frequency

    # Add directions from record (already extracted from HDF5 in Nisar_slc)
    attrs['flightDirection'] = record_dict['flightDirection']
    attrs['lookDirection'] = record_dict['lookDirection']

    # Add polarization
    attrs['polarization'] = pol

    # Add startTime from record
    if 'startTime' in record_dict:
        st = record_dict['startTime']
        if isinstance(st, (pd.Timestamp, np.datetime64)):
            st = pd.Timestamp(st).strftime('%Y-%m-%d %H:%M:%S')
        attrs['startTime'] = st

    # Add burst (short scene name)
    attrs['burst'] = os.path.basename(scene_name)

    # Add fullBurstID (like S1, this should be last for to_dataframe indices)
    attrs['fullBurstID'] = os.path.basename(outdir)

    # Add spatial ref
    from pyproj import CRS
    crs = CRS.from_epsg(epsg)
    attrs['spatial_ref'] = crs.to_wkt()

    # Reload root for metadata update (after parallel writes)
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, zarr_format=3, mode='r+')
    root.attrs.update(attrs)

    # Add coordinates as zarr arrays (read directly from zarr, not xarray)
    trans_path = os.path.join(outdir, 'transform')
    trans_store = zarr.storage.LocalStore(trans_path)
    trans_root = zarr.open_group(trans_store, mode='r')
    out_y = trans_root['y'][:]
    out_x = trans_root['x'][:]
    del trans_store, trans_root

    y_arr = root.create_array('y', data=out_y.astype(np.float64), chunks=(len(out_y),), overwrite=True,
                              dimension_names=['y'])
    x_arr = root.create_array('x', data=out_x.astype(np.float64), chunks=(len(out_x),), overwrite=True,
                              dimension_names=['x'])

    # Add variable attributes
    root['re'].attrs['scale_factor'] = scale
    root['re'].attrs['add_offset'] = 0
    root['re'].attrs['_FillValue'] = int(fill_value)
    root['re'].attrs['_ARRAY_DIMENSIONS'] = ['y', 'x']

    root['im'].attrs['scale_factor'] = scale
    root['im'].attrs['add_offset'] = 0
    root['im'].attrs['_FillValue'] = int(fill_value)
    root['im'].attrs['_ARRAY_DIMENSIONS'] = ['y', 'x']

    y_arr.attrs['_ARRAY_DIMENSIONS'] = ['y']
    x_arr.attrs['_ARRAY_DIMENSIONS'] = ['x']

    # Consolidate metadata
    zarr.consolidate_metadata(store)

    if debug:
        print(f'Total time: {time.perf_counter() - _t0_total:.1f}s')


class Nisar_transform(Nisar_align):
    """Nisar transform - simplified version without reramp (stripmap mode)."""
    import pandas as pd
    import xarray as xr
    import numpy as np

    def transform(self,
                  target: str,
                  ref: str,
                  frequency: str | None = None,
                  epsg: str | int | None = 'auto',
                  resolution: tuple[int, int] = (8, 16),
                  chunk: tuple[int, int] = (8192, 8192),
                  remove_topo_phase: bool = True,
                  remove_tidal_phase: bool = True,
                  reference_height: float | None = None,
                  dem_vertical_accuracy: float = 0.5,
                  alignment_spacing: float = 12.0 / 3600,
                  xcorr: tuple | int | None = (256, 256),
                  bbox: list | tuple | None = None,
                  overwrite: bool = False,
                  append: bool = False,
                  n_jobs: int | None = None,
                  scheduler: str | None = None,
                  tmpdir: str | None = None,
                  debug: bool = False):
        """
        Transform Nisar SLC data to geographic coordinates.

        Parameters
        ----------
        target : str
            The output directory where the results are saved.
        ref : str
            The reference scene date (YYYY-MM-DD).
        frequency : str | None, optional
            Frequency band to process: 'A' or 'B'.
            - None: Auto-detect if files have single frequency, error if both present
            - 'A': Process frequencyA (20MHz, ~7m resolution)
            - 'B': Process frequencyB (5MHz, ~25m resolution)
        epsg : str|int|None, optional
            The EPSG code to use for the output data. Use 'auto' for automatic.
            With None, each scene uses the UTM zone of its own centroid.
            Geocoding is always enabled: epsg=0 (radar coordinates) is not supported and raises ValueError.
        resolution : tuple[int, int], optional
            The resolution to use in meters per pixel.
        chunk : tuple[int, int], optional
            Processing chunk size (y, x) in pixels. Default is (8192, 8192).
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
            Default (256, 256). Grid is auto-computed over the whole SLC of the input files: band A every other
            window, band B touching windows, 2x denser on an axis the file covers at most 1/2 of its full frame
            and 4x at most 1/4 (never closer than half the window); each accepted patch is re-centred 3 times,
            and a patch whose re-centring stops early is skipped. The correction is bilinear on a full frame
            (over 1/2 of the frame on both axes) and a constant shift on a crop. When that area is too small for
            the window (fewer than 8 accepted patches), a RuntimeError asks for a smaller window, e.g.
            xcorr=192 or xcorr=128. None disables the refinement: the geometry alone is not accurate for NISAR.
        bbox : list | tuple | None, optional
            Bounding box [lon_min, lat_min, lon_max, lat_max] in WGS84 to crop
            output grid. Useful when input data was downloaded for a subregion.
        overwrite : bool, optional
            Overwrite existing results.
        append : bool, optional
            Append new scenes to existing results.
        n_jobs : int, optional
            Number of parallel workers of every internal step: the geocoding tiles, the alignment
            (SAT_llt2rat and xcorr) and the SLC chunks. None or -1 (default): all cores.
            Each chunk worker holds one chunk: lower n_jobs or chunk to use less RAM.
        scheduler : str, optional
            Not used for NISAR (kept for API compatibility).
        tmpdir : str, optional
            Directory for temporary files.
        debug : bool, optional
            Whether to print debug information.
        """
        from tqdm.auto import tqdm
        import joblib
        import os
        import tempfile
        import shutil
        import warnings
        import pandas as pd
        import numpy as np
        from insardev_toolkit.utils_files import exists

        warnings.filterwarnings('ignore', message='.*Consolidated metadata.*', category=UserWarning)

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
        for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
            os.environ[var] = '1'

        if self.DEM is None:
            raise ValueError('ERROR: DEM is not set. Please create a new instance with a DEM.')

        records = self.to_dataframe(ref=ref)

        # Validate and determine frequency to use
        if frequency is not None:
            if frequency not in ('A', 'B'):
                raise ValueError(f"frequency must be 'A', 'B', or None, got '{frequency}'")
            use_frequency = frequency
        elif self.frequency is not None:
            # Use frequency detected during __init__
            use_frequency = self.frequency
        else:
            # Check what frequencies are available in input files
            import h5py
            from .utils_nisar import nisar_get_frequencies
            sample_path = records['path'].iloc[0]
            available_freqs = nisar_get_frequencies(sample_path)
            if len(available_freqs) == 1:
                use_frequency = available_freqs[0]
            else:
                raise ValueError(
                    f"Input files contain both frequencyA and frequencyB.\n"
                    f"Please specify frequency='A' or frequency='B' parameter:\n"
                    f"  frequency='A': 20MHz bandwidth (~7m resolution)\n"
                    f"  frequency='B': 5MHz bandwidth (~25m resolution)"
                )

        print(f'NOTE: Processing frequency{use_frequency}.')

        if epsg is None:
            print('NOTE: EPSG code will be computed automatically for each scene.')
        elif isinstance(epsg, str) and epsg == 'auto':
            from .utils_satellite import get_utm_epsg
            import warnings
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
        # an orbit that cannot be used raises before anything is removed or written
        self._check_orbits(refreps, target, overwrite, append)
        if overwrite and os.path.exists(target):
            print(f'NOTE: Removing all previous results.')
            shutil.rmtree(target)

        metafile = os.path.join(target, 'zarr.json')
        if os.path.exists(target):
            # an empty metadata file raises
            if not exists(metafile, again='run the processing'):
                print(f'NOTE: target processing is not completed. Continuing...')
            elif not append:
                print(f'NOTE: target processing is completed. Skipping...')
                return
        # an empty metadata file of a scene raises before any scene is processed and before anything is removed
        for scene_refs, _ in refreps:
            exists(os.path.join(target, self.sceneId(scene_refs[0][-1]), 'zarr.json'), again='run the processing')
        if os.path.exists(metafile):
            os.remove(metafile)

        tmpdir_base = tmpdir if tmpdir is not None else tempfile.gettempdir()

        def process_scene_sequential(scenes, target, debug=False):
            """Process a single scene group with dates processed sequentially."""
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

            ref_scene_name = scene_refs[0][-1]
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
                    chunk=chunk, compute_topo=remove_topo_phase,
                    n_jobs=n_jobs, debug=debug, datum=self.dem_datum()
                )

                # Pre-compute SC_height
                sc_height_cache = {}
                for scene_ref in scene_refs:
                    scene_ref_name = scene_ref[-1]
                    prm_ref = prm_cache[scene_ref_name]
                    sc_height_result = prm_ref.SAT_baseline(prm_ref)
                    sc_height_cache[scene_ref_name] = {
                        'SC_height': sc_height_result.get('SC_height'),
                        'SC_height_start': sc_height_result.get('SC_height_start'),
                        'SC_height_end': sc_height_result.get('SC_height_end')
                    }

                # Phase 2: Process dates sequentially
                all_dates = scene_reps + scene_refs
                for scene_item in all_dates:
                    is_reference = scene_item in scene_refs
                    scene_ref = [s for s in scene_refs if s[:2] == scene_item[:2]][0]
                    scene_ref_name = scene_ref[-1]
                    scene_name = scene_item[-1]
                    prm_ref = prm_cache[scene_ref_name]

                    # Get HDF5 path and polarization for this scene
                    rec = self.get_record(scene_name)
                    h5_path = rec['path'].iloc[0]
                    pol = rec.index.get_level_values(1)[0]

                    if is_reference:
                        prm, _, _ = self.align_ref(scene_name, debug=debug, return_slc=False)
                        prm = prm_ref
                        baseline_params = None
                    else:
                        prm, _, _ = self.align_rep(scene_name, scene_ref_name, prm_ref,
                                                    degrees=alignment_spacing, debug=debug,
                                                    return_slc=False, xcorr=xcorr, n_jobs=n_jobs)
                        baseline_result = prm_ref.SAT_baseline(prm)
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

                    # Build record dict
                    record_dict = {}
                    record_reset = rec.reset_index()
                    for col in record_reset.columns:
                        val = record_reset[col].iloc[0]
                        if hasattr(val, 'wkt'):
                            record_dict[col] = val.wkt
                        else:
                            record_dict[col] = val

                    # Use chunked processing for memory efficiency with parallel chunks
                    _transform_slc_int16_nisar_chunked(
                        outdir=outdir, conversion_dir=conversion_dir,
                        prm_rep=prm, prm_ref=prm_ref,
                        scene_name=scene_name, record_dict=record_dict,
                        epsg=scene_epsg, baseline_params=baseline_params,
                        sc_height_params=sc_height_cache[scene_ref_name],
                        remove_tidal_phase=remove_tidal_phase,
                        remove_topo_phase=remove_topo_phase,
                        h5_path=h5_path, pol=pol, frequency=self.frequency,
                        chunk=chunk, n_jobs=n_jobs, debug=debug,
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
            self.consolidate_metadata(target, record_id=all_dates[-1][-1])

        n_scenes = len(refreps)
        n_dates = len(refreps[0][0]) + len(refreps[0][1]) if refreps else 1

        # n_jobs of every internal step (tiles, alignment and xcorr, chunks); None or -1: all cores
        if n_jobs is None or n_jobs == -1:
            n_jobs = os.cpu_count()
        print(f'NOTE: Processing {n_scenes} scene(s), {n_dates} dates, chunks parallel with n_jobs={n_jobs}')
        # the processing reads self.frequency: it is set for this call only, and restored when the call ends or raises
        original_frequency = self.frequency
        self.frequency = use_frequency
        try:
            for scenes in tqdm(refreps, desc='Transforming SLC...'.ljust(25)):
                process_scene_sequential(scenes, target, debug=debug)

            # Consolidate zarr metadata
            self.consolidate_metadata(target)
        finally:
            self.frequency = original_frequency
