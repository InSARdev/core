# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .S1_topo import S1_topo


def _compute_transform_inverse_worker(prm_ref_df, prm_ref_orbit_df, dem_path, geometry_wkt, outdir,
                               scale_factor, epsg, resolution, n_chunks, debug, result_queue):
    """Worker function for compute_transform_inverse in spawned subprocess.

    Must be at module level for multiprocessing spawn to pickle it.
    Computes transform, saves to zarr, sends topo through queue.

    Takes serialized prm_ref from main process - does NOT create S1 instance.
    """
    from insardev_pygmtsar.PRM import PRM
    from insardev_pygmtsar.utils_satellite import compute_transform_inverse, get_dem_wgs84ellipsoid, save_transform
    from shapely import wkt

    # Reconstruct prm_ref from serialized dataframe
    prm_ref = PRM()
    for name, row in prm_ref_df.iterrows():
        prm_ref.set(**{name: row['value']})
    prm_ref.orbit_df = prm_ref_orbit_df

    # Parse geometry and load DEM
    geometry = wkt.loads(geometry_wkt)
    dem = get_dem_wgs84ellipsoid(dem_path, geometry)

    # Compute and save transform
    topo, transform = compute_transform_inverse(
        prm_ref, dem,
        scale_factor=scale_factor, epsg=epsg,
        resolution=resolution, n_chunks=n_chunks, debug=debug
    )
    save_transform(transform, outdir, scale_factor=scale_factor)

    # Send topo and transform (without ele) through queue
    result_queue.put({
        'topo_values': topo.values,
        'topo_a_coords': topo.coords['a'].values,
        'topo_r_coords': topo.coords['r'].values,
        'transform_azi': transform.azi.values,
        'transform_rng': transform.rng.values,
        'transform_y': transform.y.values,
        'transform_x': transform.x.values,
        'transform_attrs': dict(transform.attrs),
    })


def _offset2shift(xyz, rmax, amax, method='linear'):
    """Convert offset coordinates to shift grid."""
    import numpy as np
    import xarray as xr
    from scipy.interpolate import griddata

    rngs = np.arange(8/2, rmax+8/2, 8)
    azis = np.arange(4/2, amax+4/2, 4)
    grid_r, grid_a = np.meshgrid(rngs, azis)
    grid = griddata((xyz[:, 0], xyz[:, 1]), xyz[:, 2], (grid_r, grid_a), method=method)
    return xr.DataArray(grid, coords={'a': azis, 'r': rngs}, dims=['a', 'r'])


def _process_date_worker(args):
    """Worker function for processing a single date in spawned subprocess.

    Must be at module level for multiprocessing spawn to pickle it.
    Each worker processes one date then exits (max_tasks_per_child=1), releasing memory.

    This worker does NOT create S1 instance - uses module-level functions only.
    Computes alignment shifts for repeat dates internally (not passed from main).
    """
    (outdir, burst_item, burst_refs, is_reference,
     xml_file, tiff_file, orbit_file, record_dict,
     topo, transform,
     prm_ref_df, prm_ref_orbit_df, sc_height,
     topo_llt, epsg, remove_tidal_phase, debug) = args

    import warnings
    import numpy as np
    from insardev_pygmtsar.PRM import PRM
    from insardev_pygmtsar.utils_s1 import make_burst

    # Suppress zarr v3 consolidated metadata warnings in worker
    warnings.filterwarnings('ignore', message='.*Consolidated metadata.*', category=UserWarning)

    burst_name = burst_item[-1]

    # Reconstruct prm_ref from dataframe
    prm_ref = PRM()
    for name, row in prm_ref_df.iterrows():
        prm_ref.set(**{name: row['value']})
    prm_ref.orbit_df = prm_ref_orbit_df

    # Load burst data
    if is_reference:
        # Reference: deramped SLC for symmetric geocoding (no alignment offsets)
        from insardev_pygmtsar.utils_s1 import deramped_burst as deramped_burst_func
        _, _, slc, reramp_params = deramped_burst_func(xml_file, tiff_file, orbit_file)
        prm = prm_ref
        baseline_params = None
    else:
        # Repeat: compute alignment offsets + load deramped SLC
        from insardev_pygmtsar.utils_s1 import deramped_burst
        earth_radius = prm_ref.get('earth_radius')

        # First get PRM without SLC to compute offsets
        prm_rep_temp, orbit_df_temp = make_burst(xml_file, tiff_file, orbit_file, mode=0, debug=debug)
        prm_rep_temp.orbit_df = orbit_df_temp

        # Compute time offset
        t1, prf = prm_rep_temp.get('clock_start', 'PRF')
        t2 = prm_rep_temp.get('clock_start')
        nl = int((t2 - t1) * prf * 86400.0 + 0.2)

        # Create shifted reference PRM
        prm_ref_shifted = PRM(prm_ref)
        prm_ref_shifted.orbit_df = prm_ref.orbit_df
        prm_ref_shifted.set(
            prm_ref.sel('clock_start', 'clock_stop', 'SC_clock_start', 'SC_clock_stop')
            + nl / prf / 86400.0
        )
        prm_ref_shifted.calc_dop_orb(earth_radius, inplace=True, debug=debug)

        # Compute offsets
        tmpm_dat = prm_ref_shifted.SAT_llt2rat(coords=topo_llt, precise=1, debug=debug)
        prm_rep_temp.calc_dop_orb(earth_radius, inplace=True, debug=debug)
        tmp1_dat = prm_rep_temp.SAT_llt2rat(coords=topo_llt, precise=1, debug=debug)

        # Compute offset table
        offset_dat0 = np.hstack([tmpm_dat, tmp1_dat])
        func = lambda row: [row[0], row[5] - row[0], row[1], row[6] - row[1], 100]
        offset_dat = np.apply_along_axis(func, 1, offset_dat0)

        # Filter valid points for fitoffset
        rmax = prm_rep_temp.get('num_rng_bins')
        amax = prm_rep_temp.get('num_lines')
        par_tmp = offset_dat[
            (offset_dat[:, 0] > 0) & (offset_dat[:, 0] < rmax) &
            (offset_dat[:, 2] > 0) & (offset_dat[:, 2] < amax)
        ].copy()
        par_tmp[:, 2] += nl

        # Load deramped SLC (no shift, no reramp)
        prm_dict, orbit_df, slc, reramp_params = deramped_burst(
            xml_file, tiff_file, orbit_file
        )
        prm = PRM()
        prm.set(**prm_dict)
        prm.orbit_df = orbit_df

        # Apply fitoffset parameters (bilinear offset model)
        prm.set(PRM.fitoffset(3, 3, par_tmp))
        prm.calc_dop_orb(earth_radius, inplace=True, debug=debug)

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

    # Transform and save
    _transform_slc_int16(
        outdir=outdir, transform=transform, topo=topo,
        prm_rep=prm, prm_ref=prm_ref, slc_data=slc,
        burst_name=burst_name, record_dict=record_dict,
        epsg=epsg, baseline_params=baseline_params, sc_height_params=sc_height,
        reramp_params=reramp_params, remove_tidal_phase=remove_tidal_phase, debug=debug
    )
    return burst_name


def _compute_reramp_phase(azi_rep, rng_rep, reramp_params):
    """Compute TOPS reramp phase analytically at arbitrary radar coordinates.

    Each date's burst has slightly different FM rate and Doppler centroid
    parameters, so the reramp must use each burst's own parameters to restore
    the original TOPS ramp. The ramp does NOT cancel between dates.

    Parameters
    ----------
    azi_rep : np.ndarray
        Azimuth pixel coordinates (float32, 2D). Uses 0.5-based pixel centers.
    rng_rep : np.ndarray
        Range pixel coordinates (float32, 2D). Uses 0.5-based pixel centers.
    reramp_params : dict
        Parameters from deramped_burst(): fka, fnc, ks, dta, dts, ts0, tau0, lpb, k_start.

    Returns
    -------
    phase : np.ndarray
        Reramp phase in radians (float32, same shape as input).
        Apply as: reramped = deramped * exp(-1j * phase)
    """
    import numpy as np

    fka = reramp_params['fka']
    fnc = reramp_params['fnc']
    ks = reramp_params['ks']
    dta = reramp_params['dta']
    dts = reramp_params['dts']
    ts0 = reramp_params['ts0']
    tau0 = reramp_params['tau0']
    lpb = reramp_params['lpb']
    k_start = reramp_params['k_start']

    # Convert pixel coordinates to time coordinates
    # azi_rep/rng_rep use 0.5-based pixel centers (first pixel at 0.5)
    # Subtract 0.5 to get 0-based integer indices matching deramped_burst convention
    azi_full = (azi_rep - 0.5) + k_start
    eta = (azi_full - lpb / 2.0 + 0.5) * dta
    tau = ts0 + (rng_rep - 0.5) * dts - tau0

    # FM rate polynomial at each range
    ka = fka[0] + fka[1] * tau + fka[2] * tau**2
    kt = ka * ks / (ka - ks)

    # Doppler centroid at each range
    fnct = fnc[0] + fnc[1] * tau + fnc[2] * tau**2
    del tau

    # Reference azimuth time
    etaref = -fnct / ka + fnc[0] / fka[0]
    del ka

    # Reramp phase (same formula as deramp but applied as exp(-1j * phase))
    phase = (-np.pi * kt * (eta - etaref)**2 - 2.0 * np.pi * fnct * eta).astype(np.float32)
    del kt, eta, etaref, fnct

    return phase


def _compute_merged_transform(transform, prm_rep):
    """Compute merged transform by adding PRM bilinear offsets to ref transform.

    Parameters
    ----------
    transform : xr.Dataset
        Reference burst transform with azi, rng variables.
    prm_rep : PRM
        Repeat burst PRM with fitoffset parameters (rshift, stretch_r, etc.).

    Returns
    -------
    azi_rep, rng_rep : np.ndarray
        Repeat burst radar coordinates for each output pixel (float32, 2D).
    """
    import numpy as np

    azi_ref = transform.azi.values
    rng_ref = transform.rng.values

    # Bilinear offset model from fitoffset:
    # dr(a, r) = (rshift + sub_int_r) + stretch_r * r + a_stretch_r * a
    # da(a, r) = (ashift + sub_int_a) + stretch_a * r + a_stretch_a * a
    rshift = prm_rep.get('rshift') + prm_rep.get('sub_int_r')
    ashift = prm_rep.get('ashift') + prm_rep.get('sub_int_a')
    stretch_r = prm_rep.get('stretch_r')
    a_stretch_r = prm_rep.get('a_stretch_r')
    stretch_a = prm_rep.get('stretch_a')
    a_stretch_a = prm_rep.get('a_stretch_a')

    rng_rep = (rng_ref + rshift + stretch_r * rng_ref + a_stretch_r * azi_ref).astype(np.float32)
    azi_rep = (azi_ref + ashift + stretch_a * rng_ref + a_stretch_a * azi_ref).astype(np.float32)

    return azi_rep, rng_rep


def _transform_slc_int16(outdir, transform, topo, prm_rep, prm_ref, slc_data,
                               burst_name, record_dict, epsg, scale=0.5,
                               baseline_params=None, sc_height_params=None,
                               reramp_params=None, remove_tidal_phase=True, debug=False):
    """Transform SLC to geocoded int16 zarr.

    Module-level function usable from both class methods and joblib workers.

    All SLCs (ref and rep) are deramped before geocoding. For rep bursts,
    the alignment offsets are merged into the geocoding transform (single
    interpolation). After geocoding, each date's TOPS reramp phase is restored
    analytically (using that date's own burst parameters) along with topo phase.
    The reramp is required because FM rate and Doppler centroid parameters differ
    between acquisitions — the ramp does NOT cancel in interferograms.
    """
    import os
    import time
    import numpy as np
    import xarray as xr
    import pandas as pd
    from insardev_pygmtsar.PRM import PRM
    from insardev_toolkit.datagrid import datagrid

    _t0 = time.perf_counter()
    _timings = {}

    # Convert SLC data to xarray
    num_lines = prm_rep.get('num_lines')
    num_rng_bins = prm_rep.get('num_rng_bins')

    # Input is complex64 (raw DN values)
    slc_complex = slc_data.astype(np.complex64)

    coords = {'a': np.arange(slc_complex.shape[0]) + 0.5, 'r': np.arange(slc_complex.shape[1]) + 0.5}

    nonzero_mask = slc_complex != 0
    col_valid = nonzero_mask.sum(axis=0) > 0.8 * slc_complex.shape[0]
    row_valid = nonzero_mask.sum(axis=1) > 0.8 * slc_complex.shape[1]
    slc_complex = np.where(col_valid[np.newaxis, :] & row_valid[:, np.newaxis], slc_complex, np.nan + 0j)

    slc_xa = xr.DataArray(slc_complex, coords=coords, dims=['a', 'r']).rename('data')
    del slc_complex
    _timings['slc_prep'] = time.perf_counter() - _t0

    # Compute tidal datetime from PRM if tidal correction is enabled
    tidal_dt = None
    if remove_tidal_phase and topo is not None:
        import datetime as _dt
        sc_clock = prm_rep.get('SC_clock_start')
        sc_clock_stop = prm_rep.get('SC_clock_stop')
        sc_mid = (sc_clock + sc_clock_stop) / 2.0
        year = int(sc_mid // 1000)
        doy_frac = sc_mid % 1000
        tidal_dt = _dt.datetime(year, 1, 1) + _dt.timedelta(days=doy_frac - 1)

    if reramp_params is not None and epsg != 0:
        # ====================================================================
        # Merged alignment + geocoding path (projected output)
        # Single interpolation: deramped SLC → projected grid
        # Then analytical reramp + topo phase correction
        # ====================================================================

        # Step 1: Compute transform and geocode SLC (single remap)
        _t0 = time.perf_counter()
        try:
            prm_rep.get('rshift')
            # Rep burst: merged transform with alignment offsets
            azi_map, rng_map = _compute_merged_transform(transform, prm_rep)
        except:
            # Ref burst: no alignment offsets, use ref transform directly
            azi_map = transform.azi.values.astype(np.float32)
            rng_map = transform.rng.values.astype(np.float32)
        complex_proj = _geocode_with_maps(slc_xa, azi_map, rng_map,
                                          transform.y.values, transform.x.values)
        complex_proj = complex_proj.transpose('y', 'x')
        del slc_xa
        _timings['merged_geocode'] = time.perf_counter() - _t0

        # Step 2: Compute reramp phase at geocoded radar coordinates
        _t0 = time.perf_counter()
        phase_reramp = _compute_reramp_phase(azi_map, rng_map, reramp_params)
        del azi_map, rng_map

        # Step 3: Compute topo phase (+ tidal) and geocode to projected coordinates
        topo_phase = _flat_earth_topo_phase(topo, prm_rep, prm_ref,
                                            baseline_params=baseline_params,
                                            sc_height_params=sc_height_params)
        if tidal_dt is not None:
            topo_phase.values += _tidal_phase_radar(topo, prm_ref, tidal_dt).values

        topo_phase_xa = xr.DataArray(topo_phase.values, coords=topo_phase.coords,
                                     dims=topo_phase.dims).rename('data')
        topo_phase_proj = _geocode(transform, topo_phase_xa).transpose('y', 'x').values
        del topo_phase, topo_phase_xa

        # Combine reramp + topo into total phase correction
        phase_reramp += topo_phase_proj
        del topo_phase_proj

        _timings['phase_compute'] = time.perf_counter() - _t0

        # Step 4: Apply combined phase correction exp(-1j * phase)
        _t0 = time.perf_counter()
        cos_phase = np.cos(phase_reramp)
        sin_phase = np.sin(phase_reramp)
        del phase_reramp
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
        _timings['phase_apply'] = time.perf_counter() - _t0
    else:
        # ====================================================================
        # Original path: ref bursts, or rep bursts with epsg=0 (radar coords)
        # ====================================================================

        # Compute topo phase
        _t0 = time.perf_counter()
        phase = _flat_earth_topo_phase(topo, prm_rep, prm_ref,
                                                   baseline_params=baseline_params,
                                                   sc_height_params=sc_height_params)
        _timings['flat_earth_topo_phase'] = time.perf_counter() - _t0

        # Tidal phase correction
        if tidal_dt is not None:
            phase.values += _tidal_phase_radar(topo, prm_ref, tidal_dt).values

        # Apply phase correction and optionally geocode
        _t0 = time.perf_counter()
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

        if epsg == 0:
            complex_proj = slc_corrected.rename({'a': 'y', 'r': 'x'})
            _timings['phase_apply'] = time.perf_counter() - _t0
        else:
            complex_proj = _geocode(transform, slc_corrected)
            complex_proj = complex_proj.transpose('y', 'x')
            _timings['geocode'] = time.perf_counter() - _t0
        del slc_corrected

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

    # Add TOPS-specific parameters
    for name, value in prm_rep.read_tops_params().items():
        data_proj.attrs[name] = value

    # Add baseline
    if prm_rep is prm_ref:
        BPR = 0.0
    else:
        baseline = prm_ref.SAT_baseline(prm_rep)
        BPR = baseline.get('B_perpendicular')
    data_proj.attrs['BPR'] = BPR + 0

    # Add record attributes from dict (reverse order to match transform_slc_int16)
    for name, value in list(record_dict.items())[::-1]:
        if name not in ['orbit', 'path']:
            if isinstance(value, (pd.Timestamp, np.datetime64)):
                value = pd.Timestamp(value).strftime('%Y-%m-%d %H:%M:%S')
            data_proj.attrs[name] = value
    # DEBUG: check what was stored
    if debug:
        print(f'DEBUG: attrs stored: {sorted(data_proj.attrs.keys())}')

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
        store=os.path.join(outdir, burst_name),
        mode='w',
        zarr_format=3,
        consolidated=True,
        encoding=encoding
    )
    _timings['to_zarr'] = time.perf_counter() - _t0
    del data_proj


def _geocode_with_maps(data, azi_map, rng_map, out_y, out_x):
    """Geocode using pre-computed azimuth/range coordinate maps.

    Like _geocode but takes explicit azi/rng maps instead of a
    transform dataset. Used for merged alignment+geocoding where the maps
    include alignment offsets.

    Parameters
    ----------
    data : xr.DataArray
        Input data in radar coordinates with dims ['a', 'r'].
    azi_map : np.ndarray
        Azimuth pixel coordinates for each output pixel (float32, 2D).
    rng_map : np.ndarray
        Range pixel coordinates for each output pixel (float32, 2D).
    out_y, out_x : np.ndarray
        Output grid coordinates (1D).

    Returns
    -------
    xr.DataArray
        Geocoded data with dims ['y', 'x'].
    """
    import cv2
    import numpy as np
    import xarray as xr

    data_vals = data.values
    coord_a = data.a.values
    coord_r = data.r.values

    inv_map_a = ((azi_map - coord_a[0]) / (coord_a[1] - coord_a[0])).astype(np.float32)
    inv_map_r = ((rng_map - coord_r[0]) / (coord_r[1] - coord_r[0])).astype(np.float32)

    n_y, n_x = inv_map_a.shape
    OPENCV_MAX = 32766

    if n_x <= OPENCV_MAX:
        if np.iscomplexobj(data_vals):
            grid_proj_re = cv2.remap(data_vals.real.astype(np.float32), inv_map_r, inv_map_a,
                                     interpolation=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            grid_proj_im = cv2.remap(data_vals.imag.astype(np.float32), inv_map_r, inv_map_a,
                                     interpolation=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            grid_proj = (grid_proj_re + 1j * grid_proj_im).astype(data.dtype)
        else:
            grid_proj = cv2.remap(data_vals.astype(np.float32), inv_map_r, inv_map_a,
                                  interpolation=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    else:
        n_chunks = (n_x + OPENCV_MAX - 1) // OPENCV_MAX
        x_indices = np.arange(n_x)
        chunk_indices = np.array_split(x_indices, n_chunks)

        if np.iscomplexobj(data_vals):
            grid_proj = np.empty((n_y, n_x), dtype=data.dtype)
            data_re = data_vals.real.astype(np.float32)
            data_im = data_vals.imag.astype(np.float32)
            for idx in chunk_indices:
                x_slice = slice(idx[0], idx[-1] + 1)
                re_chunk = cv2.remap(data_re, inv_map_r[:, x_slice], inv_map_a[:, x_slice],
                                     interpolation=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                im_chunk = cv2.remap(data_im, inv_map_r[:, x_slice], inv_map_a[:, x_slice],
                                     interpolation=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                grid_proj[:, x_slice] = (re_chunk + 1j * im_chunk).astype(data.dtype)
            del data_re, data_im
        else:
            grid_proj = np.empty((n_y, n_x), dtype=np.float32)
            data_f32 = data_vals.astype(np.float32)
            for idx in chunk_indices:
                x_slice = slice(idx[0], idx[-1] + 1)
                grid_proj[:, x_slice] = cv2.remap(data_f32, inv_map_r[:, x_slice], inv_map_a[:, x_slice],
                                                  interpolation=cv2.INTER_LANCZOS4,
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            del data_f32

    coords = {'y': out_y, 'x': out_x}
    return xr.DataArray(grid_proj, coords=coords, dims=['y', 'x']).rename(data.name)


def _geocode(transform, data):
    """Geocode data from radar to geographic coordinates using transform."""
    import cv2
    import numpy as np
    import xarray as xr

    trans_azi = transform.azi.values
    trans_rng = transform.rng.values
    out_y = transform.y.values
    out_x = transform.x.values

    data_vals = data.values
    coord_a = data.a.values
    coord_r = data.r.values

    inv_map_a = ((trans_azi - coord_a[0]) / (coord_a[1] - coord_a[0])).astype(np.float32)
    inv_map_r = ((trans_rng - coord_r[0]) / (coord_r[1] - coord_r[0])).astype(np.float32)

    n_y, n_x = inv_map_a.shape
    OPENCV_MAX = 32766  # cv2.remap requires dimensions < 32767

    if n_x <= OPENCV_MAX:
        # Direct remap - no chunking needed
        if np.iscomplexobj(data_vals):
            grid_proj_re = cv2.remap(data_vals.real.astype(np.float32), inv_map_r, inv_map_a,
                                     interpolation=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            grid_proj_im = cv2.remap(data_vals.imag.astype(np.float32), inv_map_r, inv_map_a,
                                     interpolation=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            grid_proj = (grid_proj_re + 1j * grid_proj_im).astype(data.dtype)
        else:
            grid_proj = cv2.remap(data_vals.astype(np.float32), inv_map_r, inv_map_a,
                                  interpolation=cv2.INTER_LANCZOS4,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    else:
        # Chunked remap - split x dimension into minimal equal chunks
        n_chunks = (n_x + OPENCV_MAX - 1) // OPENCV_MAX
        x_indices = np.arange(n_x)
        chunk_indices = np.array_split(x_indices, n_chunks)

        if np.iscomplexobj(data_vals):
            grid_proj = np.empty((n_y, n_x), dtype=data.dtype)
            data_re = data_vals.real.astype(np.float32)
            data_im = data_vals.imag.astype(np.float32)
            for idx in chunk_indices:
                x_slice = slice(idx[0], idx[-1] + 1)
                re_chunk = cv2.remap(data_re, inv_map_r[:, x_slice], inv_map_a[:, x_slice],
                                     interpolation=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                im_chunk = cv2.remap(data_im, inv_map_r[:, x_slice], inv_map_a[:, x_slice],
                                     interpolation=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                grid_proj[:, x_slice] = (re_chunk + 1j * im_chunk).astype(data.dtype)
            del data_re, data_im
        else:
            grid_proj = np.empty((n_y, n_x), dtype=np.float32)
            data_f32 = data_vals.astype(np.float32)
            for idx in chunk_indices:
                x_slice = slice(idx[0], idx[-1] + 1)
                grid_proj[:, x_slice] = cv2.remap(data_f32, inv_map_r[:, x_slice], inv_map_a[:, x_slice],
                                                  interpolation=cv2.INTER_LANCZOS4,
                                                  borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            del data_f32

    coords = {'y': out_y, 'x': out_x}
    return xr.DataArray(grid_proj, coords=coords, dims=['y', 'x']).rename(data.name)


def _tidal_phase_radar(topo, prm_ref, dt):
    """Compute solid Earth tidal phase correction on the radar coordinate grid.

    Uses 2×2 radar-grid corners: computes tidal E,N,U and look vectors at the
    4 corner points, bilinearly interpolates each component separately onto the
    full topo grid, then dot-products to LOS and converts to phase.

    Parameters
    ----------
    topo : xr.DataArray
        Topographic elevation with radar coordinates (a, r).
    prm_ref : PRM
        Reference burst PRM (has orbit_df, clock_start, PRF, etc.).
    dt : datetime.datetime
        Acquisition UTC time.

    Returns
    -------
    xr.DataArray
        Tidal phase correction in radians, same coords as topo.
    """
    import numpy as np
    import xarray as xr
    import cv2
    from insardev_pygmtsar.utils_satellite import satellite_rat2llt, _hermite_interp
    from insardev_pygmtsar.utils_tidal import solid_tide

    n_azi = len(topo.a)
    n_rng = len(topo.r)

    # --- (a) 4 radar corner coordinates ---
    azi_corners = np.array([topo.a.values[0], topo.a.values[0],
                            topo.a.values[-1], topo.a.values[-1]])
    rng_corners = np.array([topo.r.values[0], topo.r.values[-1],
                            topo.r.values[0], topo.r.values[-1]])

    # --- (b) Geocode 4 corners → lat/lon ---
    orbit_df = prm_ref.orbit_df
    orbit_time = orbit_df['isec'].values
    orbit_pos = orbit_df[['px', 'py', 'pz']].values
    orbit_vel = orbit_df[['vx', 'vy', 'vz']].values

    clock_start = (prm_ref.get('clock_start') % 1.0) * 86400
    prf = prm_ref.get('PRF')
    near_range = prm_ref.get('near_range')
    rng_samp_rate = prm_ref.get('rng_samp_rate')
    earth_radius = prm_ref.get('earth_radius')

    lon_corners, lat_corners, _ = satellite_rat2llt(
        azi_corners, rng_corners,
        orbit_time, orbit_pos, orbit_vel,
        clock_start, prf, near_range, rng_samp_rate, earth_radius)

    # --- (c) Compute tidal E, N, U at 4 corners ---
    tide_e, tide_n, tide_u = solid_tide(lon_corners, lat_corners, dt)

    # --- (d) Compute look vectors at 4 corners from orbit ---
    # Satellite ECEF at corner azimuth times
    sat_time = np.float64(clock_start) + azi_corners / np.float64(prf)
    sat_x = _hermite_interp(orbit_time, orbit_pos[:, 0], orbit_vel[:, 0], sat_time)
    sat_y = _hermite_interp(orbit_time, orbit_pos[:, 1], orbit_vel[:, 1], sat_time)
    sat_z = _hermite_interp(orbit_time, orbit_pos[:, 2], orbit_vel[:, 2], sat_time)

    # Ground ECEF from corner lat/lon (WGS84 ellipsoid, height=0)
    ra = 6378137.0
    e2 = 6.69437999014e-3
    lat_rad = np.deg2rad(lat_corners)
    lon_rad = np.deg2rad(lon_corners)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    N_wgs = ra / np.sqrt(1 - e2 * sin_lat**2)
    gx = N_wgs * cos_lat * np.cos(lon_rad)
    gy = N_wgs * cos_lat * np.sin(lon_rad)
    gz = N_wgs * (1 - e2) * sin_lat

    # Look vector (ground → satellite), normalized
    lx = sat_x - gx
    ly = sat_y - gy
    lz = sat_z - gz
    dist = np.sqrt(lx**2 + ly**2 + lz**2)
    lx /= dist;  ly /= dist;  lz /= dist

    # ECEF look → ENU
    # Rotation: b = lat - 90°, g = lon + 90°
    b = lat_rad - np.pi / 2
    g = lon_rad + np.pi / 2
    cos_b = np.cos(b);  sin_b = np.sin(b)
    cos_g = np.cos(g);  sin_g = np.sin(g)
    look_E = cos_g * lx + sin_g * ly
    look_N = -sin_g * cos_b * lx + cos_g * cos_b * ly - sin_b * lz
    look_U = -sin_g * sin_b * lx + cos_g * sin_b * ly + cos_b * lz

    # --- (e) Bilinear interpolation + LOS (memory-efficient) ---
    # Reshape 4-element arrays to 2×2: rows=azimuth (near/far azi), cols=range (near/far rng)
    def _to_2x2(arr):
        return np.array([[arr[0], arr[1]], [arr[2], arr[3]]], dtype=np.float32)

    W, H = n_rng, n_azi  # cv2.resize takes (width, height)
    los = cv2.resize(_to_2x2(tide_e), (W, H), interpolation=cv2.INTER_LINEAR) \
        * cv2.resize(_to_2x2(look_E), (W, H), interpolation=cv2.INTER_LINEAR)
    tmp = cv2.resize(_to_2x2(tide_n), (W, H), interpolation=cv2.INTER_LINEAR)
    tmp *= cv2.resize(_to_2x2(look_N), (W, H), interpolation=cv2.INTER_LINEAR)
    los += tmp
    tmp = cv2.resize(_to_2x2(tide_u), (W, H), interpolation=cv2.INTER_LINEAR)
    tmp *= cv2.resize(_to_2x2(look_U), (W, H), interpolation=cv2.INTER_LINEAR)
    los += tmp
    del tmp

    # --- (f) Convert to phase ---
    wavelength = prm_ref.get('radar_wavelength')
    cnst = -4.0 * np.pi / wavelength
    tidal_phase = (cnst * los).astype(np.float32)

    return xr.DataArray(tidal_phase, coords=topo.coords, dims=topo.dims).rename('tidal_phase')


def _flat_earth_topo_phase(topo, prm_rep, prm_ref, baseline_params=None, sc_height_params=None):
    """Compute flat earth and topographic phase correction."""
    import numpy as np
    import xarray as xr
    from scipy import constants
    from insardev_pygmtsar.PRM import PRM

    is_reference = (prm_rep is prm_ref)

    if topo is None:
        xdim = prm_ref.get('num_rng_bins')
        ydim = prm_ref.get('num_patches') * prm_ref.get('num_valid_az')
        azis = np.arange(0.5, ydim, 1)
        rngs = np.arange(0.5, xdim, 1)
        topo = xr.DataArray(np.zeros((len(azis), len(rngs)), dtype=np.float32),
                            dims=['a', 'r'], coords={'a': azis, 'r': rngs}).rename('topo')

    def calc_drho(rho, topo_vals, earth_radius, height, b, alpha, Bx):
        sina = np.sin(alpha)
        cosa = np.cos(alpha)
        c = earth_radius + height
        ret = earth_radius + topo_vals
        cost = ((rho**2 + c**2 - ret**2) / (2. * rho * c))
        sint = np.sqrt(1. - cost**2)
        term1 = rho**2 + b**2 - 2 * rho * b * (sint * cosa - cost * sina) - Bx**2
        drho = -rho + np.sqrt(term1)
        return drho

    prm1 = PRM().set(prm_ref)
    prm1.orbit_df = prm_ref.orbit_df
    prm2 = PRM().set(prm_rep)
    prm2.orbit_df = prm_rep.orbit_df

    if is_reference:
        prm2.set(
            baseline_start=0, baseline_center=0, baseline_end=0,
            alpha_start=0, alpha_center=0, alpha_end=0,
            B_offset_start=0, B_offset_center=0, B_offset_end=0
        ).fix_aligned()
    elif baseline_params is not None:
        prm2.set(**baseline_params).fix_aligned()
    else:
        prm2.set(prm1.SAT_baseline(prm2).sel(
            'baseline_start', 'baseline_center', 'baseline_end',
            'alpha_start', 'alpha_center', 'alpha_end',
            'B_offset_start', 'B_offset_center', 'B_offset_end'
        )).fix_aligned()

    if sc_height_params is not None:
        prm1.set(**sc_height_params).fix_aligned()
    else:
        prm1.set(prm1.SAT_baseline(prm1).sel('SC_height', 'SC_height_start', 'SC_height_end')).fix_aligned()

    topo_vals = topo.values.copy()
    np.copyto(topo_vals, 0, where=np.isnan(topo_vals))
    y_coords = topo.a.values
    x_coords = topo.r.values

    xdim = prm1.get('num_rng_bins')
    ydim = prm1.get('num_patches') * prm1.get('num_valid_az')

    htc = prm1.get('SC_height')
    ht0 = prm1.get('SC_height_start')
    htf = prm1.get('SC_height_end')

    tspan = 86400 * abs(prm2.get('SC_clock_stop') - prm2.get('SC_clock_start'))

    drange = constants.speed_of_light / (2 * prm2.get('rng_samp_rate'))
    alpha = prm2.get('alpha_start') * np.pi / 180
    cnst = -4 * np.pi / prm2.get('radar_wavelength')

    Bh0 = prm2.get('baseline_start') * np.cos(prm2.get('alpha_start') * np.pi / 180)
    Bv0 = prm2.get('baseline_start') * np.sin(prm2.get('alpha_start') * np.pi / 180)
    Bhf = prm2.get('baseline_end') * np.cos(prm2.get('alpha_end') * np.pi / 180)
    Bvf = prm2.get('baseline_end') * np.sin(prm2.get('alpha_end') * np.pi / 180)
    Bx0 = prm2.get('B_offset_start')
    Bxf = prm2.get('B_offset_end')

    if prm2.get('baseline_center') != 0 or prm2.get('alpha_center') != 0 or prm2.get('B_offset_center') != 0:
        Bhc = prm2.get('baseline_center') * np.cos(prm2.get('alpha_center') * np.pi / 180)
        Bvc = prm2.get('baseline_center') * np.sin(prm2.get('alpha_center') * np.pi / 180)
        Bxc = prm2.get('B_offset_center')

        dBh = (-3 * Bh0 + 4 * Bhc - Bhf) / tspan
        dBv = (-3 * Bv0 + 4 * Bvc - Bvf) / tspan
        ddBh = (2 * Bh0 - 4 * Bhc + 2 * Bhf) / (tspan * tspan)
        ddBv = (2 * Bv0 - 4 * Bvc + 2 * Bvf) / (tspan * tspan)

        dBx = (-3 * Bx0 + 4 * Bxc - Bxf) / tspan
        ddBx = (2 * Bx0 - 4 * Bxc + 2 * Bxf) / (tspan * tspan)
    else:
        dBh = (Bhf - Bh0) / tspan
        dBv = (Bvf - Bv0) / tspan
        dBx = (Bxf - Bx0) / tspan
        ddBh = ddBv = ddBx = 0

    dht = (-3 * ht0 + 4 * htc - htf) / tspan
    ddht = (2 * ht0 - 4 * htc + 2 * htf) / (tspan * tspan)

    x_coords_f64 = x_coords.astype(np.float64)
    y_coords_f64 = y_coords.astype(np.float64)
    near_range = (prm1.get('near_range') + \
        x_coords_f64.reshape(1, -1) * (1 + prm1.get('stretch_r')) * drange) + \
        y_coords_f64.reshape(-1, 1) * prm1.get('a_stretch_r') * drange

    t_arr = y_coords_f64 * tspan / (ydim - 1)
    Bh = Bh0 + dBh * t_arr + ddBh * t_arr**2
    Bv = Bv0 + dBv * t_arr + ddBv * t_arr**2
    Bx = Bx0 + dBx * t_arr + ddBx * t_arr**2
    B = np.sqrt(Bh * Bh + Bv * Bv)
    alpha = np.arctan2(Bv, Bh)
    height = ht0 + dht * t_arr + ddht * t_arr**2

    drho = calc_drho(near_range, topo_vals, prm1.get('earth_radius'),
                     height.reshape(-1, 1), B.reshape(-1, 1), alpha.reshape(-1, 1), Bx.reshape(-1, 1))

    phase_shift = (cnst * drho).astype(np.float32)
    topo_phase = xr.DataArray(phase_shift, topo.coords)
    topo_phase = topo_phase.where(np.isfinite(topo)).rename('phase')

    return topo_phase


class S1_transform(S1_topo):
    import pandas as pd
    import xarray as xr
    import numpy as np

    def consolidate_metadata(self, target: str, resolution: tuple[int, int]=(20, 5), burst: str=None):
        """
        Consolidate metadata for a given resolution and burst.

        Parameters
        ----------
        target : str
            The output directory where the results are saved.
        burst : str
            The burst to use.
        """
        import zarr
        import os
        root_dir = target
        if burst:
            root_dir = os.path.join(target, self.fullBurstId(burst))
        #print ('root_dir', root_dir)
        root_store = zarr.storage.LocalStore(root_dir)
        root_group = zarr.group(store=root_store, zarr_format=3, overwrite=False)
        zarr.consolidate_metadata(root_store)

    def transform(self,
                  target: str,
                  ref: str,
                  records: pd.DataFrame|None=None,
                  epsg: str|int|None='auto',
                  resolution: tuple[int, int]=(20, 5),
                  remove_topo_phase: bool = True,
                  remove_tidal_phase: bool = True,
                  dem_vertical_accuracy: float=0.5,
                  alignment_spacing: float=12.0/3600,
                  overwrite: bool=False,
                  append: bool=False,
                  n_jobs: int|None=None,
                  scheduler: str|None=None,
                  tmpdir: str|None=None,
                  debug: bool=False):
        """
        Transform SLC data to geographic coordinates.

        Parameters
        ----------
        target : str
            The output directory where the results are saved.
        ref : str
            The reference burst data. For multi-path processing only the path with this data is processed.
        records : pd.DataFrame, optional
            The records to use. By default, all records are used.
        epsg : str|int|None, optional
            The EPSG code to use for the output data. By default ('auto'), the EPSG code is computed automatically.
            Use epsg=0 to disable geocoding and keep radar coordinates (y=azimuth, x=range).
        resolution : tuple[int, int], optional
            The resolution to use in meters per pixel in the projected coordinate system.
        remove_topo_phase : bool, optional
            Remove the topographic phase from SLC data for interferometric processing. Set to False
            when creating a DEM from interferograms so the topo phase remains.
        dem_vertical_accuracy : float, optional
            The DEM vertical accuracy in meters.
        alignment_spacing : float, optional
            The alignment spacing in decimal degrees.
        overwrite : bool, optional
            Overwrite existing results and process all bursts.
        append : bool, optional
            Append new burstID processed with the same parameters to the existing results.
        n_jobs : int, optional
            The number of jobs to run in parallel. Default is os.cpu_count().
        scheduler : str, optional
            The parallel scheduler to use: 'loky' (default, multiprocessing), 'threads' (threading),
            or 'sequential' (no parallelism, lowest memory usage). Default is None which uses 'loky'.
        tmpdir : str, optional
            Directory for temporary files. Use fast local storage (e.g., '/mnt' on Google Colab)
            for better performance. Default is system temp directory.
        debug : bool, optional
            Whether to print debug information.

        Notes
        -----
        The processing is parallelized using joblib. All intermediate data is kept in memory.
        Only the final zarr output is written to disk.
        """
        from tqdm.auto import tqdm
        import joblib
        import os
        import tempfile
        import shutil
        import sys
        import warnings
        import pandas as pd
        import numpy as np

        # Suppress zarr v3 consolidated metadata warnings
        warnings.filterwarnings('ignore', message='.*Consolidated metadata.*', category=UserWarning)

        # Control library threading to prevent over-subscription
        # Must be set BEFORE workers spawn (loky inherits env from parent process)
        for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
            os.environ[var] = '1'

        if self.DEM is None:
            raise ValueError('ERROR: DEM is not set. Please create a new instance of S1 with a DEM.')

        if records is None:
            records = self.to_dataframe(ref=ref)

        if epsg == 0:
            print('NOTE: epsg=0, keeping radar coordinates (no geocoding).')
        elif epsg is None:
            print('NOTE: EPSG code will be computed automatically for each burst. These projections can be different.')
        elif isinstance(epsg, str) and epsg == 'auto':
            from .utils_satellite import get_utm_epsg
            epsgs = self.to_dataframe().centroid.apply(lambda geom: get_utm_epsg(geom.y, geom.x)).unique()
            if len(epsgs) > 1:
                raise ValueError(f'ERROR: Multiple UTM zones found: {", ".join(map(str, epsgs))}. Specify the EPSG code manually.')
            epsg = epsgs[0]
            print(f'NOTE: EPSG code is computed automatically for all bursts: {epsg}.')

        # add asserts for the obvious expectations
        assert not os.path.exists(target) or os.path.isdir(target), f'ERROR: target exists but is not a directory'
        if overwrite and os.path.exists(target):
            # remove all previous results and process all bursts
            print(f'NOTE: Removing all previous results and processing all bursts.')
            shutil.rmtree(target)
        # consolidated metadata file zarr.json is saved at the end of the processing
        metafile = os.path.join(target, 'zarr.json')
        assert not os.path.exists(metafile) or os.path.isfile(metafile), f'ERROR: target metadata is not a file'
        # check if the processing is completed
        if os.path.exists(target):
            if not os.path.exists(metafile) or os.path.getsize(metafile) == 0:
                print(f'NOTE: target processing is not completed before. Continuing...')
            elif not append:
                # processing is completed before, nothing to do
                print(f'NOTE: target processing is completed before. Skipping...')
                return
        # remove the consolidated metadata file when appending
        if os.path.exists(metafile):
            os.remove(metafile)

        # Use user-specified tmpdir or fall back to system temp directory
        tmpdir_base = tmpdir if tmpdir is not None else tempfile.gettempdir()

        def process_burst_sequential(bursts, target, debug=False):
            """Process a single burst with dates processed sequentially (efficient - caches prm/transform)."""
            burst_refs = bursts[0]
            burst_reps = bursts[1]
            fullBurstId = self.fullBurstId(burst_refs[0][-1])
            outdir = os.path.join(target, fullBurstId)
            metafile = os.path.join(outdir, 'zarr.json')

            # Check if already completed
            if os.path.exists(outdir):
                assert os.path.isdir(outdir), f'ERROR: {fullBurstId} exists but is not a directory'
                if os.path.exists(metafile) and os.path.getsize(metafile) > 0:
                    return  # Already done
                else:
                    print(f'NOTE: {fullBurstId} directory exists but metadata file is missing. Removing...')
                    shutil.rmtree(outdir)

            # Phase 1: Compute transform - cache PRMs (computed once, reused)
            prm_cache = {}
            for burst_ref in burst_refs:
                prm, _ = self.align_ref(burst_ref[-1], debug=debug, return_slc=False)
                prm_cache[burst_ref[-1]] = prm

            ref_burst_name = burst_refs[0][-1]
            prm_ref_main = prm_cache[ref_burst_name]

            # Load DEM and compute transform
            from .utils_satellite import compute_transform_inverse, get_dem_wgs84ellipsoid, save_transform
            record = self.get_record(ref_burst_name)
            dem = get_dem_wgs84ellipsoid(self.DEM, record.geometry.iloc[0])
            topo, transform = compute_transform_inverse(prm_ref_main, dem, scale_factor=1/dem_vertical_accuracy, epsg=epsg, resolution=resolution, debug=debug)

            # Save transform to zarr
            save_transform(transform, outdir, scale_factor=1/dem_vertical_accuracy)

            if not remove_topo_phase:
                topo = None
            # Drop ele - not needed for geocoding
            transform = transform.drop_vars('ele')

            # Pre-compute SC_height (cached)
            sc_height_cache = {}
            for burst_ref in burst_refs:
                burst_ref_name = burst_ref[-1]
                prm_ref = prm_cache[burst_ref_name]
                sc_height_result = prm_ref.SAT_baseline(prm_ref)
                sc_height_cache[burst_ref_name] = {
                    'SC_height': sc_height_result.get('SC_height'),
                    'SC_height_start': sc_height_result.get('SC_height_start'),
                    'SC_height_end': sc_height_result.get('SC_height_end')
                }

            # Phase 2: Process dates sequentially (efficient - reuses cached data)
            all_dates = burst_reps + burst_refs
            for burst_item in all_dates:
                is_reference = burst_item in burst_refs
                burst_ref = [b for b in burst_refs if b[:2] == burst_item[:2]][0]
                burst_ref_name = burst_ref[-1]
                burst_name = burst_item[-1]
                prm_ref = prm_cache[burst_ref_name]

                if is_reference:
                    # Deramped SLC for symmetric geocoding (same as rep path)
                    _, slc, reramp_params = self.align_ref(burst_name, debug=debug)
                    prm = prm_ref  # use cached PRM (ensures is_reference identity check works)
                    baseline_params = None
                else:
                    prm, slc, reramp_params = self.align_rep(burst_name, burst_ref_name, prm_ref,
                                                              degrees=alignment_spacing, debug=debug)
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

                # Build record_dict for zarr metadata
                record = self.get_record(burst_name)
                record_dict = {}
                for _, row in record.reset_index().iterrows():
                    for name, value in row.items():
                        if isinstance(value, (pd.Timestamp, np.datetime64)):
                            value = pd.Timestamp(value).strftime('%Y-%m-%d %H:%M:%S')
                        elif hasattr(value, 'wkt'):
                            value = value.wkt
                        record_dict[name] = value

                # Call module-level function directly (same as _process_date_worker)
                _transform_slc_int16(outdir, transform, topo, prm, prm_ref, slc,
                                     burst_name=burst_name, record_dict=record_dict, epsg=epsg,
                                     baseline_params=baseline_params,
                                     sc_height_params=sc_height_cache[burst_ref_name],
                                     reramp_params=reramp_params,
                                     remove_tidal_phase=remove_tidal_phase, debug=debug)
                del slc

            # Cleanup and consolidate
            del topo, transform, prm_cache
            self.consolidate_metadata(target, resolution=resolution, burst=all_dates[-1][-1])

        def process_burst_dates_parallel(bursts, target, n_jobs_inner, scheduler_inner=None, debug=False):
            """Process a single burst with dates parallelized across n_jobs_inner processes."""
            import multiprocessing as mp
            import numpy as np
            import xarray as xr
            from .PRM import PRM

            burst_refs = bursts[0]
            burst_reps = bursts[1]
            fullBurstId = self.fullBurstId(burst_refs[0][-1])
            outdir = os.path.join(target, fullBurstId)
            metafile = os.path.join(outdir, 'zarr.json')

            # Check if already completed
            if os.path.exists(outdir):
                assert os.path.isdir(outdir), f'ERROR: {fullBurstId} exists but is not a directory'
                if os.path.exists(metafile) and os.path.getsize(metafile) > 0:
                    return  # Already done
                else:
                    print(f'NOTE: {fullBurstId} directory exists but metadata file is missing. Removing...')
                    shutil.rmtree(outdir)

            ref_burst_name = burst_refs[0][-1]

            # Phase 1: Compute prm_ref with doppler correction (shared with subprocess and date workers)
            prm_ref, _ = self.align_ref(ref_burst_name, debug=debug, return_slc=False)
            prm_ref_df = prm_ref.df
            prm_ref_orbit_df = prm_ref.orbit_df

            # Get geometry for transform subprocess
            record = self.get_record(ref_burst_name)
            geometry_wkt = record.geometry.iloc[0].wkt

            # Phase 2: Compute transform in spawned subprocess (memory released on exit)
            # Use spawn context for true memory isolation (not fork which shares pages)
            ctx = mp.get_context('spawn')
            result_queue = ctx.Queue()
            p = ctx.Process(target=_compute_transform_inverse_worker, args=(
                prm_ref_df, prm_ref_orbit_df, self.DEM, geometry_wkt, outdir,
                1/dem_vertical_accuracy, epsg, resolution, 8, debug, result_queue
            ))
            p.start()
            result = result_queue.get()  # wait for result
            p.join()

            # Reconstruct topo from queue result
            if remove_topo_phase:
                topo = xr.DataArray(result['topo_values'],
                                   coords={'a': result['topo_a_coords'], 'r': result['topo_r_coords']},
                                   dims=['a', 'r'])
            else:
                topo = None

            # Reconstruct transform from queue result (without ele - not needed for geocoding)
            transform = xr.Dataset({
                'rng': xr.DataArray(result['transform_rng'],
                                   coords={'y': result['transform_y'], 'x': result['transform_x']},
                                   dims=['y', 'x']),
                'azi': xr.DataArray(result['transform_azi'],
                                   coords={'y': result['transform_y'], 'x': result['transform_x']},
                                   dims=['y', 'x']),
            }, attrs=result['transform_attrs'])

            # Get SC_height from prm_ref (already computed before subprocess)
            sc_height_result = prm_ref.SAT_baseline(prm_ref)
            sc_height = {
                'SC_height': sc_height_result.get('SC_height'),
                'SC_height_start': sc_height_result.get('SC_height_start'),
                'SC_height_end': sc_height_result.get('SC_height_end')
            }

            # Phase 3: Process dates in parallel using spawned subprocesses
            # Each worker processes one date then exits (max_tasks_per_child=1), releasing memory
            all_dates = burst_reps + burst_refs

            # prm_ref_df and prm_ref_orbit_df already serialized before subprocess

            # Get topo_llt once (small, shared by all workers for alignment computation)
            topo_llt = self._get_topo_llt(ref_burst_name, degrees=alignment_spacing)

            # Build argument tuples for each date (no S1 instance needed in workers)
            worker_args = []
            for burst_item in all_dates:
                is_ref = burst_item in burst_refs
                burst_name = burst_item[-1]

                # Get file paths for this burst
                prefix = self.fullBurstId(burst_name)
                record = self.get_record(burst_name)
                xml_file = os.path.join(self.datadir, prefix, 'annotation', f'{burst_name}.xml')
                tiff_file = os.path.join(self.datadir, prefix, 'measurement', f'{burst_name}.tiff')
                orbit_file = os.path.join(self.datadir, record['orbit'].iloc[0])

                # Build record dict for worker (use reset_index to include index values like polarization)
                record_dict = {}
                record_reset = record.reset_index()
                if debug and burst_item == all_dates[0]:
                    print(f'DEBUG: record_reset.columns = {list(record_reset.columns)}')
                for col in record_reset.columns:
                    val = record_reset[col].iloc[0]
                    if hasattr(val, 'wkt'):  # geometry
                        record_dict[col] = val.wkt
                    else:
                        record_dict[col] = val
                if debug and burst_item == all_dates[0]:
                    print(f'DEBUG: record_dict keys = {list(record_dict.keys())}')

                worker_args.append((
                    outdir, burst_item, burst_refs, is_ref,
                    xml_file, tiff_file, orbit_file, record_dict,
                    topo, transform,
                    prm_ref_df, prm_ref_orbit_df, sc_height,
                    topo_llt, epsg, remove_tidal_phase, debug
                ))

            # Use ProcessPoolExecutor or ThreadPoolExecutor based on scheduler
            from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

            if scheduler_inner == 'threads':
                with ThreadPoolExecutor(max_workers=n_jobs_inner) as executor:
                    list(executor.map(_process_date_worker, worker_args))
            else:
                # Default: ProcessPoolExecutor with max_tasks_per_child=1 for memory isolation
                with ProcessPoolExecutor(max_workers=n_jobs_inner, mp_context=mp.get_context('spawn'),
                                         max_tasks_per_child=1) as executor:
                    list(executor.map(_process_date_worker, worker_args))

            # Cleanup and consolidate
            del topo, transform, prm_ref
            self.consolidate_metadata(target, resolution=resolution, burst=all_dates[-1][-1])

        # Get reference and repeat bursts as groups
        refrep_dict = self.get_repref(ref=ref)
        refreps = [v for v in refrep_dict.values()]

        # Default n_jobs to cpu_count()
        if n_jobs is None:
            n_jobs = os.cpu_count()

        # Auto-select sequential scheduler for single-worker mode (most memory-efficient)
        if n_jobs == 1 and scheduler is None:
            scheduler = 'sequential'
            print(f'NOTE: n_jobs=1, auto-selecting scheduler="sequential" for lowest memory usage.')
            print(f'      Use scheduler="loky" or "threads" to override if needed.')

        n_bursts = len(refreps)
        # n_dates is total dates, n_rep_dates is repeat dates only (excluding reference)
        n_dates = len(refreps[0][0]) + len(refreps[0][1]) if refreps else 1
        n_rep_dates = n_dates - 1  # Only repeat dates matter for parallelization comparison

        # Determine scheduler: 'sequential', 'threads', or 'loky' (default)
        if scheduler == 'sequential' or debug:
            # Sequential execution for debugging or explicit sequential scheduler
            print(f'NOTE: Sequential execution ({n_bursts} bursts, {n_dates} dates).')
            for bursts in tqdm(refreps, desc='Transforming SLC...'.ljust(25)):
                process_burst_sequential(bursts, target, debug=debug)
        elif n_bursts >= n_rep_dates:
            # More bursts than repeat dates: parallelize across bursts
            # e.g., 1000 bursts × 2 dates → burst-parallel
            n_procs = min(n_jobs, n_bursts)
            print(f'NOTE: Using {n_procs} workers for {n_bursts} bursts, {n_dates} dates each (burst-parallel, scheduler={scheduler}).')
            with self.progressbar_joblib(tqdm(desc='Transforming SLC...'.ljust(25), total=len(refreps))) as progress_bar:
                joblib.Parallel(n_jobs=n_procs, backend=scheduler)(
                    joblib.delayed(process_burst_sequential)(bursts, target, debug) for bursts in refreps
                )
        else:
            # More repeat dates than bursts: parallelize dates within each burst
            # e.g., 1 burst × 100 dates → date-parallel
            print(f'NOTE: Processing {n_bursts} bursts sequentially, {n_dates} dates each with {n_jobs} workers (date-parallel, scheduler={scheduler}).')
            for bursts in tqdm(refreps, desc='Transforming SLC...'.ljust(25)):
                process_burst_dates_parallel(bursts, target, n_jobs_inner=n_jobs, scheduler_inner=scheduler, debug=debug)

        # Consolidate zarr metadata for the target directory
        self.consolidate_metadata(target, resolution=resolution)
