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


def _process_chunk_nisar(iy, ix, chunk_y, chunk_x, n_y, n_x,
                         outdir, zarr_path,
                         h5_path, pol, frequency,
                         alignment_params, tidal_dt,
                         prm_rep_dict, prm_ref_dict,
                         baseline_params, sc_height_params,
                         num_lines, num_rng_bins,
                         scale, fill_value, epsg,
                         remove_topo_phase):
    """
    Process a single output chunk - designed for parallel execution.

    All parameters are serializable (no xarray/PRM objects) for ProcessPoolExecutor.
    Each worker exits after one chunk (max_tasks_per_child=1), releasing all memory.

    IMPORTANT: Uses zarr directly (not xarray) to avoid loading full arrays.
    """
    import numpy as np
    import cv2
    import zarr
    import os
    from .utils_nisar import nisar_slc

    jy = min(iy + chunk_y, n_y)
    jx = min(ix + chunk_x, n_x)

    # Load ONLY the chunk we need directly from zarr (not full arrays via xarray!)
    trans_path = os.path.join(outdir, 'transform')
    trans_store = zarr.storage.LocalStore(trans_path)
    trans_root = zarr.open_group(trans_store, mode='r')

    # Read only the specific chunk region (zarr handles this efficiently)
    transform_scale = trans_root['azi'].attrs.get('scale_factor', 1.0)
    transform_fill = trans_root['azi'].attrs.get('_FillValue', 2147483647)

    # Read raw int32 data
    azi_raw = trans_root['azi'][iy:jy, ix:jx]
    rng_raw = trans_root['rng'][iy:jy, ix:jx]

    # Convert to float32 with proper fill value handling
    azi_chunk = np.where(azi_raw == transform_fill, np.nan, azi_raw * transform_scale).astype(np.float32)
    rng_chunk = np.where(rng_raw == transform_fill, np.nan, rng_raw * transform_scale).astype(np.float32)
    del azi_raw, rng_raw, trans_store, trans_root

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

    # Adjust coordinates to local SLC chunk
    azi_local = azi_chunk - azi_min
    rng_local = rng_chunk - rng_min

    # Compute inverse maps for cv2.remap (local chunk coordinates)
    inv_map_a = (azi_local - 0.5).astype(np.float32)
    inv_map_r = (rng_local - 0.5).astype(np.float32)

    # Geocode SLC chunk
    slc_re = slc_chunk.real.astype(np.float32)
    slc_im = slc_chunk.imag.astype(np.float32)
    del slc_chunk

    proj_re = cv2.remap(slc_re, inv_map_r, inv_map_a,
                        interpolation=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    proj_im = cv2.remap(slc_im, inv_map_r, inv_map_a,
                        interpolation=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    del slc_re, slc_im, inv_map_a, inv_map_r

    # cv2.remap doesn't reliably produce NaN when map values are NaN
    # Explicitly mask pixels where transform was fill (azi/rng was NaN)
    proj_re[~valid_mask] = np.nan
    proj_im[~valid_mask] = np.nan

    # Apply topo/tidal phase if needed
    # Skip for ref bursts: baseline_params=None → drho≈0 (no-op, avoids FP noise)
    if remove_topo_phase and epsg != 0 and baseline_params is not None:
        from .utils_satellite import flat_earth_topo_phase, tidal_phase_radar
        from .PRM import PRM
        import xarray as xr

        # Load ONLY the topo chunk we need directly from zarr
        topo_path = os.path.join(outdir, 'topo')
        if os.path.exists(topo_path):
            topo_store = zarr.storage.LocalStore(topo_path)
            topo_root = zarr.open_group(topo_store, mode='r')

            # Get topo dimensions and scaling
            topo_n_azi = topo_root['topo'].shape[0]
            topo_n_rng = topo_root['topo'].shape[1]
            topo_scale = topo_root['topo'].attrs.get('scale_factor', 1.0)
            topo_fill = topo_root['topo'].attrs.get('_FillValue', 2147483647)

            topo_azi_min = max(0, int(np.floor(np.nanmin(azi_chunk))))
            topo_azi_max = min(topo_n_azi, int(np.ceil(np.nanmax(azi_chunk))) + 1)
            topo_rng_min = max(0, int(np.floor(np.nanmin(rng_chunk))))
            topo_rng_max = min(topo_n_rng, int(np.ceil(np.nanmax(rng_chunk))) + 1)

            # Read only the chunk we need with proper fill handling
            topo_raw = topo_root['topo'][topo_azi_min:topo_azi_max, topo_rng_min:topo_rng_max]
            topo_data = np.where(topo_raw == topo_fill, np.nan, topo_raw * topo_scale).astype(np.float32)
            topo_a_coords = topo_root['a'][topo_azi_min:topo_azi_max]
            topo_r_coords = topo_root['r'][topo_rng_min:topo_rng_max]
            del topo_raw, topo_store, topo_root

            # Create minimal xarray DataArray for phase computation
            topo_chunk = xr.DataArray(
                topo_data,
                dims=['a', 'r'],
                coords={'a': topo_a_coords, 'r': topo_r_coords}
            )
            del topo_data, topo_a_coords, topo_r_coords

            # Reconstruct PRM objects from dicts
            prm_rep = PRM()
            prm_rep.set(**prm_rep_dict)
            prm_ref = PRM()
            prm_ref.set(**prm_ref_dict)

            phase_chunk = flat_earth_topo_phase(topo_chunk, prm_rep, prm_ref,
                                                 baseline_params=baseline_params,
                                                 sc_height_params=sc_height_params)

            if tidal_dt is not None:
                phase_chunk.values += tidal_phase_radar(topo_chunk, prm_ref, tidal_dt).values

            # Geocode phase to output chunk
            phase_local_a = (azi_chunk - topo_azi_min - 0.5).astype(np.float32)
            phase_local_r = (rng_chunk - topo_rng_min - 0.5).astype(np.float32)

            phase_proj = cv2.remap(phase_chunk.values.astype(np.float32),
                                   phase_local_r, phase_local_a,
                                   interpolation=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            del phase_chunk, phase_local_a, phase_local_r, topo_chunk

            # Apply phase correction
            cos_phase = np.cos(phase_proj)
            sin_phase = np.sin(phase_proj)
            del phase_proj

            corrected_re = proj_re * cos_phase + proj_im * sin_phase
            corrected_im = proj_im * cos_phase - proj_re * sin_phase
            del cos_phase, sin_phase
            proj_re = corrected_re
            proj_im = corrected_im
            del corrected_re, corrected_im

    # Convert to int16
    with np.errstate(invalid='ignore'):
        re_int16 = np.round(proj_re / scale).astype(np.int16)
        im_int16 = np.round(proj_im / scale).astype(np.int16)

    nan_mask = ~np.isfinite(proj_re)
    del proj_re, proj_im
    re_int16[nan_mask] = fill_value
    im_int16[nan_mask] = fill_value
    del nan_mask

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
                                        remove_thermal_noise=False,
                                        radiometric_calibration=None,
                                        h5_path=None, pol=None, frequency=None,
                                        chunk=(8192, 8192), n_jobs=None, debug=False):
    """
    Transform Nisar SLC to geocoded int16 zarr using chunked I/O.

    Memory-efficient version that processes chunks in parallel with joblib.
    Suitable for large NISAR frequency A data on limited RAM systems (e.g., 12GB Colab).

    Parameters
    ----------
    n_jobs : int, optional
        Number of parallel workers. Default: min(cpu_count, RAM_GB // 2).
        Each worker uses ~1.5 GB RAM.
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

    # Compute tidal datetime if needed (differential: ref - rep)
    is_reference = prm_rep is prm_ref
    tidal_dt = None
    if remove_tidal_phase and remove_topo_phase and not is_reference:
        import datetime as _dt
        def _sc_clock_to_dt(prm):
            sc_mid = (prm.get('SC_clock_start') + prm.get('SC_clock_stop')) / 2.0
            year = int(sc_mid // 1000)
            doy_frac = sc_mid % 1000
            return _dt.datetime(year, 1, 1) + _dt.timedelta(days=doy_frac - 1)
        tidal_dt = (_sc_clock_to_dt(prm_ref), _sc_clock_to_dt(prm_rep))

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
         scale, fill_value, epsg,
         remove_topo_phase)
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
    if epsg != 0:
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


def _transform_slc_int16_nisar(outdir, transform, topo, prm_rep, prm_ref, slc_data,
                               scene_name, record_dict, epsg,
                               baseline_params=None, sc_height_params=None,
                               remove_tidal_phase=True,
                               remove_thermal_noise=False,
                               radiometric_calibration=None,
                               h5_path=None, pol=None, frequency=None,
                               debug=False):
    """
    Transform Nisar SLC to geocoded int16 zarr.

    Simplified version for Nisar - no reramp needed (stripmap mode).
    """
    import os
    import time
    import numpy as np
    import xarray as xr
    import pandas as pd
    from .PRM import PRM
    from .utils_satellite import remap_radar_to_geo, compute_merged_transform
    from insardev_toolkit.datagrid import datagrid

    _t0 = time.perf_counter()
    _timings = {}

    num_lines = prm_rep.get('num_lines')
    num_rng_bins = prm_rep.get('num_rng_bins')

    # Ensure complex64 format
    slc_complex = slc_data.astype(np.complex64) if slc_data.dtype != np.complex64 else slc_data

    # Apply radiometric calibration and/or thermal noise removal for Nisar
    if remove_thermal_noise or radiometric_calibration:
        try:
            from insardev_backscatter.utils_nisar import apply_radiometric_correction_nisar
        except ImportError:
            raise ImportError(
                "Nisar radiometric calibration requires insardev_backscatter extension with Nisar support"
            )
        slc_complex = apply_radiometric_correction_nisar(
            slc_complex,
            h5_path=h5_path,
            pol=pol,
            frequency=frequency,
            calibration_type=radiometric_calibration,
            remove_noise=remove_thermal_noise
        )

    # Set output scale based on whether calibration was applied
    if radiometric_calibration:
        scale = 1e-04
        amp_max = 32767 * scale
        amplitude = np.abs(slc_complex)
        clip_mask = amplitude > amp_max
        if clip_mask.any():
            phase = np.angle(slc_complex[clip_mask])
            slc_complex[clip_mask] = amp_max * np.exp(1j * phase)
    else:
        scale = 0.5

    coords = {'a': np.arange(slc_complex.shape[0]) + 0.5, 'r': np.arange(slc_complex.shape[1]) + 0.5}

    # Mask invalid regions
    nonzero_mask = slc_complex != 0
    col_valid = nonzero_mask.sum(axis=0) > 0.8 * slc_complex.shape[0]
    row_valid = nonzero_mask.sum(axis=1) > 0.8 * slc_complex.shape[1]
    slc_complex = np.where(col_valid[np.newaxis, :] & row_valid[:, np.newaxis], slc_complex, np.nan + 0j)

    slc_xa = xr.DataArray(slc_complex, coords=coords, dims=['a', 'r']).rename('data')
    del slc_complex
    _timings['slc_prep'] = time.perf_counter() - _t0

    # Compute tidal datetime if needed (differential: ref - rep)
    is_reference = prm_rep is prm_ref
    tidal_dt = None
    if remove_tidal_phase and topo is not None and not is_reference:
        import datetime as _dt
        def _sc_clock_to_dt(prm):
            sc_mid = (prm.get('SC_clock_start') + prm.get('SC_clock_stop')) / 2.0
            year = int(sc_mid // 1000)
            doy_frac = sc_mid % 1000
            return _dt.datetime(year, 1, 1) + _dt.timedelta(days=doy_frac - 1)
        tidal_dt = (_sc_clock_to_dt(prm_ref), _sc_clock_to_dt(prm_rep))

    # Nisar: Direct geocoding (no reramp needed - stripmap mode)
    if epsg != 0:
        # Geocoded output path
        _t0 = time.perf_counter()
        if prm_rep.get('rshift') is not None:
            # Rep scene: merged transform with alignment offsets
            azi_map, rng_map = compute_merged_transform(transform, prm_rep)
        else:
            # Ref scene: no alignment offsets, use ref transform directly
            azi_map = transform.azi.values.astype(np.float32)
            rng_map = transform.rng.values.astype(np.float32)
        complex_proj = remap_radar_to_geo(slc_xa, azi_map, rng_map,
                                          transform.y.values, transform.x.values)
        complex_proj = complex_proj.transpose('y', 'x')
        del slc_xa
        _timings['geocode'] = time.perf_counter() - _t0

        # Compute and geocode topo phase
        # Skip for ref bursts: baseline=0 → drho≈0 (no-op, avoids FP noise)
        _t0 = time.perf_counter()
        if not is_reference:
            from .utils_satellite import flat_earth_topo_phase, tidal_phase_radar
            topo_phase = flat_earth_topo_phase(topo, prm_rep, prm_ref,
                                               baseline_params=baseline_params,
                                               sc_height_params=sc_height_params)
            if tidal_dt is not None:
                topo_phase.values += tidal_phase_radar(topo, prm_ref, tidal_dt).values

            topo_phase_xa = xr.DataArray(topo_phase.values, coords=topo_phase.coords,
                                         dims=topo_phase.dims).rename('data')

            # Geocode topo phase
            topo_phase_proj = remap_radar_to_geo(topo_phase_xa, azi_map, rng_map,
                                                 transform.y.values, transform.x.values).transpose('y', 'x').values
            del topo_phase, topo_phase_xa
            _timings['phase_compute'] = time.perf_counter() - _t0

            # Apply topo phase correction
            _t0 = time.perf_counter()
            cos_phase = np.cos(topo_phase_proj)
            sin_phase = np.sin(topo_phase_proj)
            del topo_phase_proj
            proj_re = complex_proj.values.real.copy()
            proj_im = complex_proj.values.imag
            corrected_re = (proj_re * cos_phase + proj_im * sin_phase)
            corrected_im = (proj_im * cos_phase - proj_re * sin_phase)
            del cos_phase, sin_phase, proj_re, proj_im
            complex_proj = xr.DataArray(
                (corrected_re + 1j * corrected_im).astype(np.complex64),
                coords=complex_proj.coords, dims=complex_proj.dims
            )
            del corrected_re, corrected_im
        del azi_map, rng_map
        _timings['phase_apply'] = time.perf_counter() - _t0
    else:
        # Radar coordinates output (epsg=0)
        _t0 = time.perf_counter()
        # Skip for ref bursts: baseline=0 → drho≈0 (no-op, avoids FP noise)
        if not is_reference:
            from .utils_satellite import flat_earth_topo_phase, tidal_phase_radar
            phase = flat_earth_topo_phase(topo, prm_rep, prm_ref,
                                          baseline_params=baseline_params,
                                          sc_height_params=sc_height_params)

            if tidal_dt is not None:
                phase.values += tidal_phase_radar(topo, prm_ref, tidal_dt).values

            phase_aligned = phase.reindex_like(slc_xa, method='nearest').values
            del phase
            cos_phase = np.cos(phase_aligned)
            sin_phase = np.sin(phase_aligned)
            del phase_aligned
            slc_vals = slc_xa.values
            corrected_real = slc_vals.real * cos_phase + slc_vals.imag * sin_phase
            corrected_imag = slc_vals.imag * cos_phase - slc_vals.real * sin_phase
            del cos_phase, sin_phase
            slc_corrected = xr.DataArray(
                (corrected_real + 1j * corrected_imag).astype(np.complex64),
                coords=slc_xa.coords, dims=slc_xa.dims
            )
            del corrected_real, corrected_imag, slc_vals, slc_xa
        else:
            slc_corrected = slc_xa
            del slc_xa
        complex_proj = slc_corrected.rename({'a': 'y', 'r': 'x'})
        _timings['phase_apply'] = time.perf_counter() - _t0

    # Convert to int16
    _t0 = time.perf_counter()
    fill_value = np.iinfo(np.int16).max
    re_vals = complex_proj.values.real
    im_vals = complex_proj.values.imag

    with np.errstate(invalid='ignore'):
        re_int16 = np.round(re_vals / scale).astype(np.int16)
        im_int16 = np.round(im_vals / scale).astype(np.int16)

    nan_mask = ~np.isfinite(re_vals)
    del re_vals, im_vals
    re_int16[nan_mask] = fill_value
    im_int16[nan_mask] = fill_value
    del nan_mask

    y_coords = complex_proj.y.values
    x_coords = complex_proj.x.values
    del complex_proj

    data_proj = xr.Dataset({
        're': xr.DataArray(re_int16, coords={'y': y_coords, 'x': x_coords}, dims=['y', 'x']),
        'im': xr.DataArray(im_int16, coords={'y': y_coords, 'x': x_coords}, dims=['y', 'x'])
    })
    del re_int16, im_int16, y_coords, x_coords
    _timings['int16_convert'] = time.perf_counter() - _t0

    # Add PRM attributes
    for name, value in prm_rep.df.itertuples():
        if name not in ['input_file', 'SLC_file', 'led_file']:
            data_proj.attrs[name] = value

    # Add baseline
    if prm_rep is prm_ref:
        BPR = 0.0
    else:
        baseline = prm_ref.SAT_baseline(prm_rep)
        BPR = baseline.get('B_perpendicular')
    data_proj.attrs['BPR'] = BPR + 0

    # Add record attributes
    for name, value in list(record_dict.items())[::-1]:
        if name not in ['path']:
            if isinstance(value, (pd.Timestamp, np.datetime64)):
                value = pd.Timestamp(value).strftime('%Y-%m-%d %H:%M:%S')
            data_proj.attrs[name] = value

    # Add storage attributes
    for varname in ['re', 'im']:
        data_proj[varname].attrs['scale_factor'] = scale
        data_proj[varname].attrs['add_offset'] = 0
        data_proj[varname].attrs['_FillValue'] = np.iinfo(np.int16).max

    if epsg == 0:
        radar_crs_wkt = '''ENGCRS["Radar Coordinates",EDATUM["Radar datum"],CS[Cartesian,2],AXIS["azimuth",south,ORDER[1],LENGTHUNIT["pixel",1]],AXIS["range",east,ORDER[2],LENGTHUNIT["pixel",1]]]'''
        data_proj.attrs['spatial_ref'] = radar_crs_wkt
    else:
        data_proj = datagrid.spatial_ref(data_proj, epsg)
        data_proj.attrs['spatial_ref'] = data_proj.spatial_ref.attrs['spatial_ref']
        data_proj = data_proj.drop_vars('spatial_ref')
        data_proj = data_proj.drop_vars(['x', 'y'])

    _t0 = time.perf_counter()
    shape = data_proj.re.shape
    encoding = {var: {'chunks': shape} for var in ['re', 'im']}
    data_proj.to_zarr(
        store=os.path.join(outdir, scene_name),
        mode='w',
        zarr_format=3,
        consolidated=True,
        encoding=encoding
    )
    _timings['to_zarr'] = time.perf_counter() - _t0
    del data_proj


class Nisar_transform(Nisar_align):
    """Nisar transform - simplified version without reramp (stripmap mode)."""
    import pandas as pd
    import xarray as xr
    import numpy as np

    def transform(self,
                  target: str,
                  ref: str,
                  records: pd.DataFrame | None = None,
                  frequency: str | None = None,
                  epsg: str | int | None = 'auto',
                  resolution: tuple[int, int] = (8, 16),
                  chunk: tuple[int, int] = (8192, 8192),
                  remove_topo_phase: bool = True,
                  remove_tidal_phase: bool = True,
                  remove_thermal_noise: bool = False,
                  radiometric_calibration: str | None = None,
                  dem_vertical_accuracy: float = 0.5,
                  alignment_spacing: float = 12.0 / 3600,
                  xcorr: tuple | None = (512, 512),
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
        records : pd.DataFrame, optional
            The records to use. By default, all records are used.
        frequency : str | None, optional
            Frequency band to process: 'A' or 'B'.
            - None: Auto-detect if files have single frequency, error if both present
            - 'A': Process frequencyA (20MHz, ~7m resolution)
            - 'B': Process frequencyB (5MHz, ~25m resolution)
        epsg : str|int|None, optional
            The EPSG code to use for the output data. Use 'auto' for automatic.
            Use epsg=0 to disable geocoding and keep radar coordinates.
        resolution : tuple[int, int], optional
            The resolution to use in meters per pixel.
        chunk : tuple[int, int], optional
            Processing chunk size (y, x) in pixels. Default is (8192, 8192).
        remove_topo_phase : bool, optional
            Remove the topographic phase from SLC data.
        remove_tidal_phase : bool, optional
            Remove solid Earth tidal displacement phase.
        remove_thermal_noise : bool, optional
            Apply thermal noise removal (requires insardev_backscatter).
        radiometric_calibration : str | None, optional
            Apply radiometric calibration: 'sigma0', 'beta0', 'gamma0', or None.
        dem_vertical_accuracy : float, optional
            The DEM vertical accuracy in meters.
        alignment_spacing : float, optional
            The alignment spacing in decimal degrees.
        xcorr : tuple | None, optional
            Xcorr patch size as (height, width). Default (512, 512) for NISAR.
            Set to None to disable xcorr refinement. Grid is auto-computed.
        overwrite : bool, optional
            Overwrite existing results.
        append : bool, optional
            Append new scenes to existing results.
        n_jobs : int, optional
            Number of parallel workers for chunk processing. Each worker uses
            ~1.5 GB RAM. Default: auto-detect based on available RAM.
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

        warnings.filterwarnings('ignore', message='.*Consolidated metadata.*', category=UserWarning)

        # Control library threading
        for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
            os.environ[var] = '1'

        if self.DEM is None:
            raise ValueError('ERROR: DEM is not set. Please create a new instance with a DEM.')

        if records is None:
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

        # Store for use in processing (override self.frequency for this transform)
        original_frequency = self.frequency
        self.frequency = use_frequency
        print(f'NOTE: Processing frequency{use_frequency}.')

        if epsg == 0:
            print('NOTE: epsg=0, keeping radar coordinates (no geocoding).')
        elif epsg is None:
            print('NOTE: EPSG code will be computed automatically for each scene.')
        elif isinstance(epsg, str) and epsg == 'auto':
            from .utils_satellite import get_utm_epsg
            epsgs = self.to_dataframe().centroid.apply(lambda geom: get_utm_epsg(geom.y, geom.x)).unique()
            if len(epsgs) > 1:
                raise ValueError(f'ERROR: Multiple UTM zones found: {", ".join(map(str, epsgs))}.')
            epsg = epsgs[0]
            print(f'NOTE: EPSG code computed automatically: {epsg}.')

        assert not os.path.exists(target) or os.path.isdir(target)
        if overwrite and os.path.exists(target):
            print(f'NOTE: Removing all previous results.')
            shutil.rmtree(target)

        metafile = os.path.join(target, 'zarr.json')
        if os.path.exists(target):
            if not os.path.exists(metafile) or os.path.getsize(metafile) == 0:
                print(f'NOTE: target processing is not completed. Continuing...')
            elif not append:
                print(f'NOTE: target processing is completed. Skipping...')
                self.frequency = original_frequency
                return
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
                if os.path.exists(metafile_scene) and os.path.getsize(metafile_scene) > 0:
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
            from .utils_satellite import compute_conversion_chunked
            record = self.get_record(ref_scene_name)

            compute_conversion_chunked(
                prm_ref_main, self.DEM, record.geometry.iloc[0], outdir,
                scale_factor=1 / dem_vertical_accuracy,
                epsg=epsg, resolution=resolution,
                chunk=chunk, compute_topo=remove_topo_phase,
                n_jobs=n_jobs, netcdf_engine=self.netcdf_engine_read, debug=debug
            )
            conversion_dir = os.path.join(outdir, 'conversion')

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
                                                return_slc=False, xcorr=xcorr)
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
                    epsg=epsg, baseline_params=baseline_params,
                    sc_height_params=sc_height_cache[scene_ref_name],
                    remove_tidal_phase=remove_tidal_phase,
                    remove_topo_phase=remove_topo_phase,
                    remove_thermal_noise=remove_thermal_noise,
                    radiometric_calibration=radiometric_calibration,
                    h5_path=h5_path, pol=pol, frequency=self.frequency,
                    chunk=chunk, n_jobs=n_jobs, debug=debug
                )

            # Cleanup and consolidate
            del prm_cache
            self.consolidate_metadata(target, record_id=all_dates[-1][-1])

        # Get reference and repeat scenes as groups
        refrep_dict = self.get_repref(ref=ref)
        refreps = [v for v in refrep_dict.values()]

        n_scenes = len(refreps)
        n_dates = len(refreps[0][0]) + len(refreps[0][1]) if refreps else 1

        # For NISAR: default to all cores for chunk-level parallelization
        if n_jobs is None:
            n_jobs = -1  # joblib convention: use all cores
        print(f'NOTE: Processing {n_scenes} scene(s), {n_dates} dates, chunks parallel with n_jobs={n_jobs}')
        for scenes in tqdm(refreps, desc='Transforming SLC...'.ljust(25)):
            process_scene_sequential(scenes, target, debug=debug)

        # Consolidate zarr metadata
        self.consolidate_metadata(target)

        # Restore original frequency setting
        self.frequency = original_frequency
