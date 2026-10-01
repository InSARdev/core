# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .S1_align import S1_align
from .utils_satellite import remap_radar_to_geo
from insardev_toolkit.utils_S1 import measurement_path
from insardev_toolkit.utils_files import exists

# The GMTSAR pixel (line, bin) of the S1 radar coordinates (a, r), minus (a, r): compute_transform_inverse() stores
# azi = line + 0.5 and rng = bin - 0.5 of SAT_llt2rat (the near_range of S1 is one bin before the first sample), so
# the first sample is centred on (0.5, 0.5). reference_surface_topo() and flat_earth_topo_phase() evaluate GMTSAR's
# expressions at this line and bin, the pixel the SLC is sampled at.
S1_PIXEL_OFFSET = (-0.5, 0.5)


def _process_date_worker(args):
    """Worker function for processing a single date in spawned subprocess.

    Must be at module level for multiprocessing spawn to pickle it.
    Each worker processes one date then exits (max_tasks_per_child=1), releasing memory.

    This worker does NOT create S1 instance - uses module-level functions only.
    Computes alignment shifts for repeat dates internally (not passed from main).
    n_jobs is this worker's share of the transform's n_jobs, for its SAT_llt2rat workers.
    """
    (outdir, burst_item, burst_refs, is_reference,
     xml_file, tiff_file, orbit_file, record_dict,
     topo, transform,
     prm_ref_df, prm_ref_orbit_df, sc_height,
     topo_llt, epsg, remove_tidal_phase,
     reference_height, n_jobs, debug) = args

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
        prm_rep_temp, orbit_df_temp = make_burst(xml_file, tiff_file, orbit_file, debug=debug)
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
        tmpm_dat = prm_ref_shifted.SAT_llt2rat(coords=topo_llt, precise=1, n_jobs=n_jobs, debug=debug)
        prm_rep_temp.calc_dop_orb(earth_radius, inplace=True, debug=debug)
        tmp1_dat = prm_rep_temp.SAT_llt2rat(coords=topo_llt, precise=1, n_jobs=n_jobs, debug=debug)

        # Compute offset table (vectorized)
        offset_dat0 = np.hstack([tmpm_dat, tmp1_dat])
        offset_dat = np.column_stack([
            offset_dat0[:, 0],                      # r_ref
            offset_dat0[:, 5] - offset_dat0[:, 0],  # dr = r_rep - r_ref
            offset_dat0[:, 1],                      # a_ref
            offset_dat0[:, 6] - offset_dat0[:, 1],  # da = a_rep - a_ref
            np.full(len(offset_dat0), 100.0)        # SNR
        ])

        # Filter valid points for fitoffset
        from .utils_satellite import offset_valid_mask
        rmax = prm_rep_temp.get('num_rng_bins')
        amax = prm_rep_temp.get('num_lines')
        par_tmp = offset_dat[offset_valid_mask(offset_dat, rmax, amax)].copy()
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
        reramp_params=reramp_params, remove_tidal_phase=remove_tidal_phase,
        reference_height=reference_height,
        debug=debug
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
    # The same expressions in blocks of output rows, into the float32 output (no full-grid float64 temporaries)
    phase = np.empty(np.shape(azi_rep), dtype=np.float32)
    for r0 in range(0, phase.shape[0], 128):
        azi_full = (azi_rep[r0:r0 + 128] - 0.5) + k_start
        eta = (azi_full - lpb / 2.0 + 0.5) * dta
        tau = ts0 + (rng_rep[r0:r0 + 128] - 0.5) * dts - tau0
        ka = fka[0] + fka[1] * tau + fka[2] * tau**2
        kt = ka * ks / (ka - ks)
        fnct = fnc[0] + fnc[1] * tau + fnc[2] * tau**2
        del tau
        etaref = -fnct / ka + fnc[0] / fka[0]
        del ka
        phase[r0:r0 + 128] = (-np.pi * kt * (eta - etaref)**2 - 2.0 * np.pi * fnct * eta).astype(np.float32)
        del kt, eta, etaref, fnct, azi_full

    return phase


def _transform_slc_int16(outdir, transform, topo, prm_rep, prm_ref, slc_data,
                               burst_name, record_dict, epsg,
                               baseline_params=None, sc_height_params=None,
                               reramp_params=None, remove_tidal_phase=True,
                               reference_height=0.0,
                               debug=False):
    """Transform SLC to geocoded int16 zarr.

    Module-level function usable from both class methods and joblib workers.

    Input: complex64 SLC data from deramped_burst() (raw DN values, no scaling).
    Output: int16 zarr of the raw DN at scale 0.5 (int16 = 2*DN).
    """
    import os
    import time
    import numpy as np
    import xarray as xr
    import pandas as pd
    from insardev_pygmtsar.PRM import PRM
    from insardev_pygmtsar.utils_satellite import remap_radar_to_geo, remap_source, remap_rows, tidal_phase_radar, flat_earth_topo_phase, compute_merged_transform, pack_complex_int16
    from insardev_toolkit.datagrid import datagrid

    _t0 = time.perf_counter()
    _timings = {}

    # Input: complex64 SLC data (raw DN values from deramped_burst, no scaling)
    num_lines = prm_rep.get('num_lines')
    num_rng_bins = prm_rep.get('num_rng_bins')

    # Ensure complex64 format
    slc_complex = slc_data.astype(np.complex64) if slc_data.dtype != np.complex64 else slc_data

    # Output scale: int16 = value / scale, so value = int16 * scale. Raw DN amplitude, typical range 50-5000:
    # int16 = 2*DN (same as old deramped_burst output), DN / scale = 2*DN → scale = 0.5
    scale = 0.5

    coords = {'a': np.arange(slc_complex.shape[0]) + 0.5, 'r': np.arange(slc_complex.shape[1]) + 0.5}

    nonzero_mask = slc_complex != 0
    col_valid = nonzero_mask.sum(axis=0) > 0.8 * slc_complex.shape[0]
    row_valid = nonzero_mask.sum(axis=1) > 0.8 * slc_complex.shape[1]
    del nonzero_mask
    # in place, no second full-burst copy (the caller drops the array afterwards)
    slc_complex[~(col_valid[np.newaxis, :] & row_valid[:, np.newaxis])] = np.nan + 0j

    slc_xa = xr.DataArray(slc_complex, coords=coords, dims=['a', 'r']).rename('data')
    del slc_complex
    _timings['slc_prep'] = time.perf_counter() - _t0

    # Compute differential tidal datetimes for rep bursts only.
    # Both ref and rep times are needed for differential correction:
    # tidal_los = tide(dt_ref)·look - tide(dt_rep)·look
    # Ref bursts get no tidal correction (no differential to compute).
    is_reference = prm_rep is prm_ref
    tidal_dt = None
    if remove_tidal_phase and topo is not None and not is_reference:
        import datetime as _dt
        def _sc_clock_to_dt(prm):
            sc_mid = (prm.get('SC_clock_start') + prm.get('SC_clock_stop')) / 2.0
            year = int(sc_mid // 1000)
            doy_frac = sc_mid % 1000
            # SC_clock carries GMTSAR's 0-based day of year: day 0.x is January 1
            return _dt.datetime(year, 1, 1) + _dt.timedelta(days=doy_frac)
        tidal_dt = (_sc_clock_to_dt(prm_ref), _sc_clock_to_dt(prm_rep))

    # Convert to int16: interpolation overshoot and the reramp can push a component past the int16 range, where
    # it would wrap around, so the amplitude of such a sample is clipped keeping the phase (pack_complex_int16),
    # below fill_value so a saturated sample is not read as NaN
    fill_value = np.iinfo(np.int16).max

    if reramp_params is not None:
        # ====================================================================
        # Merged alignment + geocoding path (projected output)
        # Single interpolation: deramped SLC → projected grid
        # Then analytical reramp + topo phase correction
        # End to end in blocks of output rows: the remap of the SLC, the reramp phase, the geocoded topo phase, the
        # phase correction and the int16 conversion of one block at a time go straight into the two int16 outputs,
        # so no full-grid complex, phase or coordinate-map temporaries exist. Every output pixel is computed on its
        # own, so each block holds the rows of the whole-grid computation, bit for bit.
        # ====================================================================
        _t0 = time.perf_counter()
        try:
            prm_rep.get('rshift')
            # Rep burst: merged transform with alignment offsets
            compute_merged_transform(transform, prm_rep, rows=slice(0, 1))
            merged = True
        except:
            # Ref burst: no alignment offsets, use ref transform directly
            merged = False

        # Step 3 input: the topo phase (+ tidal) on the radar grid, geocoded block by block below
        # Skip for ref bursts: baseline=0 → drho≈0 (no-op, avoids FP noise)
        topo_src = None
        if not is_reference:
            topo_phase = flat_earth_topo_phase(topo, prm_rep, prm_ref, None,
                                                baseline_params=baseline_params,
                                                sc_height_params=sc_height_params,
                                                pixel_offset=S1_PIXEL_OFFSET)
            if tidal_dt is not None:
                topo_phase.values += tidal_phase_radar(topo, prm_ref, tidal_dt).values
            topo_src = remap_source(topo_phase)
            del topo_phase
        slc_src = remap_source(slc_xa)
        del slc_xa
        _timings['phase_compute'] = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        azi_ref = transform.azi.values
        rng_ref = transform.rng.values
        n_y, n_x = azi_ref.shape
        re_int16 = np.empty((n_y, n_x), dtype=np.int16)
        im_int16 = np.empty((n_y, n_x), dtype=np.int16)
        # 256 rows: faster than smaller blocks or the whole grid (a multiple of the SIMD width, as the whole grid is
        # processed); the float64 coordinate expressions go 64 rows at a time
        for r0 in range(0, n_y, 256):
            blk = slice(r0, r0 + 256)
            # Step 1: Compute transform and geocode SLC (single remap)
            if merged:
                n_blk = len(range(n_y)[blk])
                azi_map = np.empty((n_blk, n_x), dtype=np.float32)
                rng_map = np.empty((n_blk, n_x), dtype=np.float32)
                for s0 in range(0, n_blk, 64):
                    azi_map[s0:s0 + 64], rng_map[s0:s0 + 64] = compute_merged_transform(
                        transform, prm_rep, rows=slice(r0 + s0, r0 + min(s0 + 64, n_blk)))
            else:
                azi_map = azi_ref[blk].astype(np.float32)
                rng_map = rng_ref[blk].astype(np.float32)
            proj = remap_rows(slc_src, azi_map, rng_map)

            # Step 2: Compute reramp phase at geocoded radar coordinates
            phase = _compute_reramp_phase(azi_map, rng_map, reramp_params)
            del azi_map, rng_map

            # Step 3: geocoded topo phase, combined with the reramp into the total phase correction
            if topo_src is not None:
                phase += remap_rows(topo_src, azi_ref[blk], rng_ref[blk])

            # Step 4: Apply combined phase correction exp(-1j * phase): the same float32 products, in place
            cos_phase = np.cos(phase)
            sin_phase = np.sin(phase)
            del phase
            proj_re = proj.real.copy()
            proj_im = proj.imag.copy()
            proj.real = proj_re * cos_phase + proj_im * sin_phase
            proj.imag = proj_im * cos_phase - proj_re * sin_phase
            del cos_phase, sin_phase, proj_re, proj_im

            re_int16[blk], im_int16[blk] = pack_complex_int16(proj.real, proj.imag, scale, fill_value)
            del proj
        del slc_src, topo_src, azi_ref, rng_ref
        y_coords = transform.y.values
        x_coords = transform.x.values
        _timings['geocode_phase_int16'] = time.perf_counter() - _t0
    else:
        # ====================================================================
        # Original path: no reramp parameters (transform() always passes them: align_ref/align_rep return them)
        # ====================================================================

        # Compute and apply topo+tidal phase correction
        # Skip for ref bursts: baseline=0 → drho≈0 (no-op, avoids FP noise)
        _t0 = time.perf_counter()
        if not is_reference:
            phase = flat_earth_topo_phase(topo, prm_rep, prm_ref, None,
                                                       baseline_params=baseline_params,
                                                       sc_height_params=sc_height_params,
                                                       pixel_offset=S1_PIXEL_OFFSET)
            _timings['flat_earth_topo_phase'] = time.perf_counter() - _t0

            # Tidal phase correction
            if tidal_dt is not None:
                phase.values += tidal_phase_radar(topo, prm_ref, tidal_dt).values

            # Apply phase correction
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
        else:
            slc_corrected = slc_xa
            del slc_xa

        complex_proj = remap_radar_to_geo(slc_corrected, transform.azi.values, transform.rng.values,
                                                   transform.y.values, transform.x.values)
        complex_proj = complex_proj.transpose('y', 'x')
        _timings['geocode'] = time.perf_counter() - _t0
        del slc_corrected

        _t0 = time.perf_counter()
        re_int16, im_int16 = pack_complex_int16(complex_proj.values.real, complex_proj.values.imag, scale, fill_value)

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

    # The reference height for downstream elevation computation, the same on every date: the flat-earth height
    # in flat mode, NaN in DEM mode (transform() passes it). A technical attribute, before BPR, so
    # to_dataframe() does not list it
    data_proj.attrs['ref_height'] = float(reference_height)

    # Add baseline
    if prm_rep is prm_ref:
        BPR = 0.0
    else:
        baseline = prm_ref.SAT_baseline(prm_rep)
        BPR = baseline.get('B_perpendicular')
    data_proj.attrs['BPR'] = BPR + 0

    # Add record attributes from dict (reverse order to match transform_slc_int16)
    for name, value in list(record_dict.items())[::-1]:
        # NOT BPR: the record carries the scan-time baseline, whose origin is
        # the first date, while the value set above is measured from THIS
        # transform's reference -- which is what makes BPR == 0 name the
        # reference in a stored stack.
        if name not in ['orbit', 'path', 'BPR', 'baseline_model']:
            if isinstance(value, (pd.Timestamp, np.datetime64)):
                # startTime keeps its fraction: core orders the bursts of a merge by it
                value = pd.Timestamp(value).strftime('%Y-%m-%d %H:%M:%S.%f')
            data_proj.attrs[name] = value
    # earth_radius stays the PRM value written above: BPR, SC_height and the topo phase are referenced to it

    # Replace approximate geometry with exact radar extent polygon from prm_ref
    from insardev_pygmtsar.utils_satellite import satellite_rat2llt
    _num_lines = prm_ref.get('num_lines')
    _num_rng = prm_ref.get('num_rng_bins')
    _orbit_ref = prm_ref.orbit_df
    _corner_azi = np.array([0.5, 0.5, _num_lines - 0.5, _num_lines - 0.5], dtype=np.float64)
    _corner_rng = np.array([0.5, _num_rng - 0.5, _num_rng - 0.5, 0.5], dtype=np.float64)
    _clon, _clat, _ = satellite_rat2llt(
        _corner_azi, _corner_rng,
        _orbit_ref['clock'].values, _orbit_ref[['px','py','pz']].values,
        _orbit_ref[['vx','vy','vz']].values,
        86400.0 * prm_ref.get('clock_start'),
        prm_ref.get('PRF'), prm_ref.get('near_range'), prm_ref.get('rng_samp_rate'),
        prm_ref.get('earth_radius'), dem=None, max_iter=1, tol=1.0, n_chunks=1
    )
    from shapely.geometry import Polygon as _Polygon
    _coords = list(zip(_clon.tolist(), _clat.tolist()))
    _coords.append(_coords[0])  # close the ring
    data_proj.attrs['geometry'] = _Polygon(_coords).wkt

    # DEBUG: check what was stored
    if debug:
        print(f'DEBUG: attrs stored: {sorted(data_proj.attrs.keys())}')

    # Add storage attributes
    for varname in ['re', 'im']:
        data_proj[varname].attrs['scale_factor'] = scale
        data_proj[varname].attrs['add_offset'] = 0
        data_proj[varname].attrs['_FillValue'] = np.iinfo(np.int16).max

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


class S1_transform(S1_align):
    import pandas as pd
    import xarray as xr
    import numpy as np

    def transform(self,
                  target: str,
                  ref: str,
                  epsg: str|int|None='auto',
                  resolution: tuple[int, int]=(16, 4),
                  remove_topo_phase: bool = True,
                  remove_tidal_phase: bool = True,
                  reference_height: float|None = None,
                  dem_vertical_accuracy: float=0.5,
                  alignment_spacing: float=12.0/3600,
                  bbox: list|tuple|None = None,
                  overwrite: bool=False,
                  append: bool=False,
                  n_jobs: int|None=None,
                  scheduler: str|None=None,
                  debug: bool=False):
        """
        Transform SLC data to geographic coordinates.

        Parameters
        ----------
        target : str
            The output directory where the results are saved.
        ref : str
            The reference burst data. For multi-path processing only the path with this data is processed.
        epsg : str|int|None, optional
            The EPSG code to use for the output data. By default ('auto'), the EPSG code is computed automatically.
            With None, each burst uses the UTM zone of its own centroid (the projections can differ between bursts).
            Geocoding is always enabled: epsg=0 (radar coordinates) is not supported and raises ValueError.
        resolution : tuple[int, int], optional
            The resolution to use in meters per pixel in the projected coordinate system.
        remove_topo_phase : bool, optional
            Remove the topographic phase from SLC data for interferometric processing. Set to False
            when creating a DEM from interferograms so the topo phase remains.
        remove_tidal_phase : bool, optional
            Remove solid Earth tidal displacement phase. Default is True. Requires GMTSAR solid_tide binary.
        reference_height : float or None, optional
            Reference height (meters above WGS84 ellipsoid) for flat-earth phase removal.
            All bursts use this same value, ensuring consistent phase across burst boundaries.
            Set to the elevation of your area of interest for best precision and fewer fringes.
            Default is None (sea level, i.e. 0). Only used when remove_topo_phase=False.
            Raises ValueError if set when remove_topo_phase=True.
        dem_vertical_accuracy : float, optional
            The DEM vertical accuracy in meters.
        alignment_spacing : float, optional
            The alignment spacing in decimal degrees.
        bbox : list or tuple, optional
            The output area [lon_min, lat_min, lon_max, lat_max] in WGS84. Only the bursts whose footprint overlaps
            it are processed (the others are skipped, with a note), and each burst's output is cropped to it.
            Raises ValueError if it overlaps no burst.
        overwrite : bool, optional
            Overwrite existing results and process all bursts.
        append : bool, optional
            Append new burstID processed with the same parameters to the existing results.
        n_jobs : int, optional
            The number of jobs to run in parallel, for every internal step: the burst or date workers and the
            geocoding and alignment (satellite_llt2rat) workers, which share it inside a burst or date worker
            (each of W concurrent workers takes n_jobs // W, at least 1). None or -1 (default): all cores.
        scheduler : str, optional
            The parallel scheduler to use: 'loky' (default, multiprocessing), 'threads' (threading),
            or 'sequential' (no parallelism, lowest memory usage). Default is None which uses 'loky'.
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
        import shutil
        import sys
        import warnings
        import pandas as pd
        import numpy as np

        # Suppress zarr v3 consolidated metadata warnings
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
        # the stored ref_height, one value on every date: the flat-earth reference height, NaN in DEM mode (the
        # grids give a residual height there, and the elevation reads NaN as 0 as it read the former 0.0)
        ref_height = float('nan') if remove_topo_phase else float(reference_height)

        # Control library threading to prevent over-subscription
        # Must be set BEFORE workers spawn (loky inherits env from parent process)
        for var in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
            os.environ[var] = '1'

        if self.DEM is None:
            raise ValueError('ERROR: DEM is not set. Please create a new instance of S1 with a DEM.')

        records = self.to_dataframe(ref=ref)

        if epsg is None:
            print('NOTE: EPSG code will be computed automatically for each burst. These projections can be different.')
        elif isinstance(epsg, str) and epsg == 'auto':
            from .utils_satellite import get_utm_epsg
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*geographic CRS.*')
                epsgs = self.to_dataframe().centroid.apply(lambda geom: get_utm_epsg(geom.y, geom.x)).unique()
            if len(epsgs) > 1:
                raise ValueError(f'ERROR: Multiple UTM zones found: {", ".join(map(str, epsgs))}. Specify the EPSG code manually.')
            epsg = epsgs[0]
            print(f'NOTE: EPSG code is computed automatically for all bursts: {epsg}.')

        # Get reference and repeat bursts as groups
        refrep_dict = self.get_repref(ref=ref)
        if bbox is not None and refrep_dict:
            # only the bursts whose footprint meets the bbox: the scan geometry of the reference record, the record
            # of the burst transform. Checked before the target is touched, so a wrong bbox removes no results.
            from shapely.geometry import box
            area = box(*bbox)
            skipped = [key for key, (burst_refs, _) in refrep_dict.items()
                       if not self.get_record(burst_refs[0][-1]).geometry.iloc[0].intersects(area)]
            if len(skipped) == len(refrep_dict):
                raise ValueError(f'ERROR: bbox {bbox} does not overlap any of the {len(refrep_dict)} bursts.')
            if skipped:
                print(f'NOTE: bbox does not overlap {len(skipped)} of {len(refrep_dict)} bursts, skipped: {", ".join(skipped)}.')
                refrep_dict = {key: value for key, value in refrep_dict.items() if key not in skipped}
        refreps = [v for v in refrep_dict.values()]

        # add asserts for the obvious expectations
        assert not os.path.exists(target) or os.path.isdir(target), f'ERROR: target exists but is not a directory'
        # an orbit that cannot be used raises before anything is removed or written
        self._check_orbits(refreps, target, overwrite, append)
        if overwrite and os.path.exists(target):
            # remove all previous results and process all bursts
            print(f'NOTE: Removing all previous results and processing all bursts.')
            shutil.rmtree(target)
        # consolidated metadata file zarr.json is saved at the end of the processing
        metafile = os.path.join(target, 'zarr.json')
        assert not os.path.exists(metafile) or os.path.isfile(metafile), f'ERROR: target metadata is not a file'
        # check if the processing is completed; an empty metadata file raises, the bursts' before any processing
        if os.path.exists(target):
            if not exists(metafile, again='run the processing'):
                print(f'NOTE: target processing is not completed before. Continuing...')
            elif not append:
                # processing is completed before, nothing to do
                print(f'NOTE: target processing is completed before. Skipping...')
                return
            for burst_refs, _ in refreps:
                exists(os.path.join(target, self.fullBurstId(burst_refs[0][-1]), 'zarr.json'),
                       again='run the processing')
        # remove the consolidated metadata file when appending
        if os.path.exists(metafile):
            os.remove(metafile)

        def process_burst_sequential(bursts, target, n_jobs_inner, debug=False):
            """Process a single burst with dates processed sequentially (efficient - caches prm/transform).

            n_jobs_inner is this burst worker's share of n_jobs, for its satellite_llt2rat workers.
            """
            burst_refs = bursts[0]
            burst_reps = bursts[1]
            fullBurstId = self.fullBurstId(burst_refs[0][-1])
            outdir = os.path.join(target, fullBurstId)
            metafile = os.path.join(outdir, 'zarr.json')

            # Check if already completed
            if os.path.exists(outdir):
                assert os.path.isdir(outdir), f'ERROR: {fullBurstId} exists but is not a directory'
                if exists(metafile, again='run the processing'):
                    return  # Already done
                else:
                    print(f'NOTE: {fullBurstId} directory exists but metadata file is missing. Removing...')
                    shutil.rmtree(outdir)

            # Phase 1: Compute transform - cache PRMs (computed once, reused)
            prm_cache = {}
            for burst_ref in burst_refs:
                prm, _, _ = self.align_ref(burst_ref[-1], debug=debug, return_slc=False)
                prm_cache[burst_ref[-1]] = prm

            ref_burst_name = burst_refs[0][-1]
            prm_ref_main = prm_cache[ref_burst_name]

            # Load DEM and compute transform
            from .utils_satellite import compute_transform_inverse, get_dem_wgs84ellipsoid, save_transform, get_utm_epsg
            record = self.get_record(ref_burst_name)
            # epsg=None: this burst's own UTM zone, from the centroid of its reference record
            _centroid = record.geometry.iloc[0].centroid
            burst_epsg = epsg if epsg is not None else get_utm_epsg(_centroid.y, _centroid.x)
            dem = get_dem_wgs84ellipsoid(self.DEM, record.geometry.iloc[0], datum=self.dem_datum())
            topo, transform = compute_transform_inverse(prm_ref_main, dem, scale_factor=1/dem_vertical_accuracy, epsg=burst_epsg, resolution=resolution, bbox=bbox, compute_topo=remove_topo_phase, n_jobs=n_jobs_inner, debug=debug)
            del dem

            # Save transform to zarr
            save_transform(transform, outdir, scale_factor=1/dem_vertical_accuracy)

            if not remove_topo_phase:
                # flat-earth reference: the WGS84 ellipsoid at reference_height per radar pixel, referenced to the
                # PRM scalar earth_radius like the DEM topo (a constant height on a per-line sphere drifted across range)
                from .utils_satellite import reference_surface_topo
                topo = reference_surface_topo(prm_ref_main, topo, reference_height, pixel_offset=S1_PIXEL_OFFSET)
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

            # Cache topo_llt per reference burst (same DEM geometry for all dates)
            topo_llt_cache = {}
            for burst_ref in burst_refs:
                topo_llt_cache[burst_ref[-1]] = self._get_topo_llt(burst_ref[-1], degrees=alignment_spacing)

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
                                                              degrees=alignment_spacing, debug=debug,
                                                              topo_llt=topo_llt_cache.get(burst_ref_name),
                                                              n_jobs=n_jobs_inner)
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

                self.transform_slc_int16(outdir, transform, topo, prm, prm_ref, slc, epsg=burst_epsg,
                                        baseline_params=baseline_params,
                                        sc_height_params=sc_height_cache[burst_ref_name],
                                        reramp_params=reramp_params,
                                        remove_tidal_phase=remove_tidal_phase,
                                        reference_height=ref_height)
                del slc

            # Cleanup and consolidate
            del topo, transform, prm_cache
            self.consolidate_metadata(target, record_id=all_dates[-1][-1])

        def process_burst_dates_parallel(bursts, target, n_jobs_inner, scheduler_inner=None, debug=False):
            """Process a single burst with dates parallelized across n_jobs_inner workers."""
            import numpy as np
            import xarray as xr

            burst_refs = bursts[0]
            burst_reps = bursts[1]
            fullBurstId = self.fullBurstId(burst_refs[0][-1])
            outdir = os.path.join(target, fullBurstId)
            metafile = os.path.join(outdir, 'zarr.json')

            # Check if already completed
            if os.path.exists(outdir):
                assert os.path.isdir(outdir), f'ERROR: {fullBurstId} exists but is not a directory'
                if exists(metafile, again='run the processing'):
                    return  # Already done
                else:
                    print(f'NOTE: {fullBurstId} directory exists but metadata file is missing. Removing...')
                    shutil.rmtree(outdir)

            # Phase 1: Compute transform inline (same as process_burst_sequential)
            prm_cache = {}
            for burst_ref in burst_refs:
                prm, _, _ = self.align_ref(burst_ref[-1], debug=debug, return_slc=False)
                prm_cache[burst_ref[-1]] = prm

            ref_burst_name = burst_refs[0][-1]
            prm_ref_main = prm_cache[ref_burst_name]

            from .utils_satellite import compute_transform_inverse, get_dem_wgs84ellipsoid, save_transform, get_utm_epsg
            record = self.get_record(ref_burst_name)
            # epsg=None: this burst's own UTM zone, from the centroid of its reference record
            _centroid = record.geometry.iloc[0].centroid
            burst_epsg = epsg if epsg is not None else get_utm_epsg(_centroid.y, _centroid.x)
            dem = get_dem_wgs84ellipsoid(self.DEM, record.geometry.iloc[0], datum=self.dem_datum())
            topo, transform = compute_transform_inverse(prm_ref_main, dem, scale_factor=1/dem_vertical_accuracy, epsg=burst_epsg, resolution=resolution, bbox=bbox, compute_topo=remove_topo_phase, n_jobs=n_jobs_inner, debug=debug)
            del dem

            save_transform(transform, outdir, scale_factor=1/dem_vertical_accuracy)

            if not remove_topo_phase:
                from .utils_satellite import reference_surface_topo
                topo = reference_surface_topo(prm_ref_main, topo, reference_height, pixel_offset=S1_PIXEL_OFFSET)
            transform = transform.drop_vars('ele')

            # Pre-compute SC_height and topo_llt caches
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

            topo_llt_cache = {}
            for burst_ref in burst_refs:
                topo_llt_cache[burst_ref[-1]] = self._get_topo_llt(burst_ref[-1], degrees=alignment_spacing)

            # Phase 2: Build worker args and process dates in parallel
            all_dates = burst_reps + burst_refs
            prm_ref_df = prm_cache[ref_burst_name].df
            prm_ref_orbit_df = prm_cache[ref_burst_name].orbit_df
            topo_llt = topo_llt_cache[ref_burst_name]
            # each of the concurrent date workers takes its share of n_jobs for its satellite_llt2rat workers
            n_jobs_date = max(1, n_jobs_inner // min(n_jobs_inner, len(all_dates)))

            worker_args = []
            for burst_item in all_dates:
                is_ref = burst_item in burst_refs
                burst_name = burst_item[-1]
                burst_ref = [b for b in burst_refs if b[:2] == burst_item[:2]][0]
                burst_ref_name = burst_ref[-1]

                prefix = self.fullBurstId(burst_name)
                record = self.get_record(burst_name)
                xml_file = os.path.join(self.datadir, prefix, 'annotation', f'{burst_name}.xml')
                tiff_file = measurement_path(os.path.join(self.datadir, prefix, 'measurement'), burst_name)
                orbit_file = self._orbit_file(burst_name, record)

                record_dict = {}
                record_reset = record.reset_index()
                for col in record_reset.columns:
                    val = record_reset[col].iloc[0]
                    if hasattr(val, 'wkt'):
                        record_dict[col] = val.wkt
                    else:
                        record_dict[col] = val

                worker_args.append((
                    outdir, burst_item, burst_refs, is_ref,
                    xml_file, tiff_file, orbit_file, record_dict,
                    topo, transform,
                    prm_ref_df, prm_ref_orbit_df, sc_height_cache[burst_ref_name],
                    topo_llt, burst_epsg, remove_tidal_phase,
                    ref_height, n_jobs_date, debug
                ))

            joblib.Parallel(n_jobs=n_jobs_inner, backend=scheduler_inner)(
                joblib.delayed(_process_date_worker)(args) for args in worker_args
            )

            # Cleanup and consolidate
            del topo, transform, prm_cache
            self.consolidate_metadata(target, record_id=all_dates[-1][-1])

        # Default n_jobs to cpu_count(), -1 too (joblib convention), before any min(n_jobs, ...)
        if n_jobs is None or n_jobs == -1:
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

        # Force sequential scheduler for debug mode
        if debug and scheduler is None:
            scheduler = 'sequential'

        # Determine scheduler: 'sequential', 'threads', or 'loky' (default)
        if n_bursts >= n_rep_dates or scheduler == 'sequential':
            # More bursts than repeat dates: parallelize across bursts
            # e.g., 1000 bursts × 2 dates → burst-parallel
            n_procs = min(n_jobs, n_bursts)
            # each of the concurrent burst workers takes its share of n_jobs for its satellite_llt2rat workers
            n_jobs_burst = n_jobs if scheduler == 'sequential' else max(1, n_jobs // n_procs)
            print(f'NOTE: Using {n_procs} workers for {n_bursts} bursts, {n_dates} dates each (burst-parallel, scheduler={scheduler}).')
            with self.progressbar_joblib(tqdm(desc='Transforming SLC...'.ljust(25), total=len(refreps))) as progress_bar:
                joblib.Parallel(n_jobs=n_procs, backend=scheduler)(
                    joblib.delayed(process_burst_sequential)(bursts, target, n_jobs_burst, debug) for bursts in refreps
                )
        else:
            # More repeat dates than bursts: parallelize dates within each burst
            # e.g., 1 burst × 100 dates → date-parallel
            print(f'NOTE: Processing {n_bursts} bursts sequentially, {n_dates} dates each with {n_jobs} workers (date-parallel, scheduler={scheduler}).')
            for bursts in tqdm(refreps, desc='Transforming SLC...'.ljust(25)):
                process_burst_dates_parallel(bursts, target, n_jobs_inner=n_jobs, scheduler_inner=scheduler, debug=debug)

        # Consolidate zarr metadata for the target directory
        self.consolidate_metadata(target)

    def transform_slc_int16(self,
                            outdir: str,
                            transform: xr.Dataset,
                            topo: xr.DataArray | None,
                            prm_rep: "PRM",
                            prm_ref: "PRM",
                            slc_data: np.ndarray,
                            epsg: int,
                            baseline_params: dict=None,
                            sc_height_params: dict=None,
                            reramp_params: dict=None,
                            remove_tidal_phase: bool=True,
                            reference_height: float=0.0
                            ):
        """
        Perform geocoding from radar to geographic coordinates.

        Input: complex64 SLC data (raw DN values from deramped_burst).
        Output: int16 zarr of the raw DN at scale 0.5.

        Delegates to the module-level _transform_slc_int16 function.
        """
        import os
        import pandas as pd
        import numpy as np

        # Extract burst name and record dict from self
        if 'input_file' in prm_rep.df.index:
            burst_name = os.path.splitext(os.path.basename(prm_rep.get('input_file')))[0]
        else:
            burst_name = 'burst'

        df = self.get_record(burst_name)
        record_dict = {}
        for _, row in df.reset_index().iterrows():
            for name, value in row.items():
                if isinstance(value, (pd.Timestamp, np.datetime64)):
                    # startTime keeps its fraction: core orders the bursts of a merge by it
                    value = pd.Timestamp(value).strftime('%Y-%m-%d %H:%M:%S.%f')
                elif hasattr(value, 'wkt'):
                    value = value.wkt
                record_dict[name] = value

        _transform_slc_int16(
            outdir=outdir, transform=transform, topo=topo,
            prm_rep=prm_rep, prm_ref=prm_ref, slc_data=slc_data,
            burst_name=burst_name, record_dict=record_dict,
            epsg=epsg,
            baseline_params=baseline_params, sc_height_params=sc_height_params,
            reramp_params=reramp_params, remove_tidal_phase=remove_tidal_phase,
            reference_height=reference_height,
            debug=bool(os.environ.get('INSAR_DEBUG'))
        )

    def flat_earth_topo_phase(self, topo: xr.DataArray | None, prm_rep: "PRM", prm_ref: "PRM",
                               baseline_params: dict = None, sc_height_params: dict = None) -> xr.DataArray:
        """
        Compute the combined earth curvature and topographic phase correction.

        Uses the full GMTSAR algorithm with time-varying baseline geometry.

        Parameters
        ----------
        topo : xr.DataArray
            Topographic elevation in radar coordinates (meters).
        prm_rep : PRM
            Repeat burst PRM object.
        prm_ref : PRM
            Reference burst PRM object.
        baseline_params : dict, optional
            Pre-computed baseline parameters (from SAT_baseline). If None, computed on the fly.
        sc_height_params : dict, optional
            Pre-computed SC_height parameters (from SAT_baseline). If None, computed on the fly.

        Returns
        -------
        xr.DataArray
            Combined flat earth and topo phase (radians).
        """
        import numpy as np
        import xarray as xr
        from scipy import constants
        import warnings
        import time
        import os
        warnings.filterwarnings('ignore')

        _t0 = time.perf_counter()
        _timings = {}

        # For reference burst (same as itself), we still compute flat earth correction
        # with baseline=0 to go through the same code path as repeat bursts
        is_reference = (prm_rep is prm_ref)

        # Create topo with zeros if None (flat-earth only correction)
        if topo is None:
            xdim = prm_ref.get('num_rng_bins')
            ydim = prm_ref.get('num_patches') * prm_ref.get('num_valid_az')
            azis = np.arange(0.5, ydim, 1)
            rngs = np.arange(0.5, xdim, 1)
            topo = xr.DataArray(np.zeros((len(azis), len(rngs)), dtype=np.float32),
                                dims=['a', 'r'],
                                coords={'a': azis, 'r': rngs}).rename('topo')

        # Calculate the combined earth curvature and topography correction
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

        # Create copies to avoid modifying cached PRMs
        # fix_aligned() modifies near_range, so calling it multiple times on cached PRMs causes drift
        from .PRM import PRM
        prm1 = PRM().set(prm_ref)  # copy of reference PRM
        prm1.orbit_df = prm_ref.orbit_df  # copy orbit data reference
        prm2 = PRM().set(prm_rep)  # copy of repeat PRM
        prm2.orbit_df = prm_rep.orbit_df  # copy orbit data reference
        _timings['prm_copy'] = time.perf_counter() - _t0

        # Set baseline parameters on PRM copies
        # Order follows GMTSAR: SAT_baseline first, then fix_aligned
        _t0 = time.perf_counter()
        if is_reference:
            # For reference burst, baseline is 0 - set explicitly to avoid SAT_baseline issues
            prm2.set(
                baseline_start=0, baseline_center=0, baseline_end=0,
                alpha_start=0, alpha_center=0, alpha_end=0,
                B_offset_start=0, B_offset_center=0, B_offset_end=0
            ).fix_aligned()
        elif baseline_params is not None:
            # Use cached baseline parameters (avoids expensive SAT_baseline call)
            prm2.set(**baseline_params).fix_aligned()
        else:
            # Only set the 9 baseline geometry parameters on prm2 (equivalent to GMTSAR's tail=9)
            # Do NOT set SC_height values on prm2 - those belong to prm1
            prm2.set(prm1.SAT_baseline(prm2).sel(
                'baseline_start', 'baseline_center', 'baseline_end',
                'alpha_start', 'alpha_center', 'alpha_end',
                'B_offset_start', 'B_offset_center', 'B_offset_end'
            )).fix_aligned()
        _timings['SAT_baseline_prm2'] = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        if sc_height_params is not None:
            # Use cached SC_height parameters (avoids expensive SAT_baseline call)
            prm1.set(**sc_height_params).fix_aligned()
        else:
            prm1.set(prm1.SAT_baseline(prm1).sel('SC_height', 'SC_height_start', 'SC_height_end')).fix_aligned()
        _timings['SAT_baseline_prm1'] = time.perf_counter() - _t0

        # Fill NaNs by 0 (avoid np.where temp array)
        topo_vals = topo.values.copy()
        np.copyto(topo_vals, 0, where=np.isnan(topo_vals))
        y_coords = topo.a.values
        x_coords = topo.r.values

        # Get full dimensions
        xdim = prm1.get('num_rng_bins')
        ydim = prm1.get('num_patches') * prm1.get('num_valid_az')

        # Get heights (from prm1 = reference copy)
        htc = prm1.get('SC_height')
        ht0 = prm1.get('SC_height_start')
        htf = prm1.get('SC_height_end')

        # Compute the time span and the time spacing (from prm2 = repeat copy)
        tspan = 86400 * abs(prm2.get('SC_clock_stop') - prm2.get('SC_clock_start'))
        assert (tspan >= 0.01) and (prm2.get('PRF') >= 0.01), \
            f"ERROR in sc_clock_start={prm2.get('SC_clock_start')}, sc_clock_stop={prm2.get('SC_clock_stop')}, or PRF={prm2.get('PRF')}"

        # Setup the default parameters
        drange = constants.speed_of_light / (2 * prm2.get('rng_samp_rate'))
        alpha = prm2.get('alpha_start') * np.pi / 180
        cnst = -4 * np.pi / prm2.get('radar_wavelength')

        # Calculate initial baselines (from prm2 which has baseline params set)
        Bh0 = prm2.get('baseline_start') * np.cos(prm2.get('alpha_start') * np.pi / 180)
        Bv0 = prm2.get('baseline_start') * np.sin(prm2.get('alpha_start') * np.pi / 180)
        Bhf = prm2.get('baseline_end') * np.cos(prm2.get('alpha_end') * np.pi / 180)
        Bvf = prm2.get('baseline_end') * np.sin(prm2.get('alpha_end') * np.pi / 180)
        Bx0 = prm2.get('B_offset_start')
        Bxf = prm2.get('B_offset_end')

        # First case is quadratic baseline model, second case is default linear model
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

        # Calculate height increment
        dht = (-3 * ht0 + 4 * htc - htf) / tspan
        ddht = (2 * ht0 - 4 * htc + 2 * htf) / (tspan * tspan)

        # Ensure float64 precision for near_range calculation to avoid sqrt precision issues
        # (float32 rho gives sqrt(rho**2) != rho due to precision loss)
        x_coords_f64 = x_coords.astype(np.float64)
        y_coords_f64 = y_coords.astype(np.float64)
        near_range = (prm1.get('near_range') + \
            x_coords_f64.reshape(1, -1) * (1 + prm1.get('stretch_r')) * drange) + \
            y_coords_f64.reshape(-1, 1) * prm1.get('a_stretch_r') * drange

        # Calculate the change in baseline and height along the frame
        t_arr = y_coords_f64 * tspan / (ydim - 1)
        Bh = Bh0 + dBh * t_arr + ddBh * t_arr**2
        Bv = Bv0 + dBv * t_arr + ddBv * t_arr**2
        Bx = Bx0 + dBx * t_arr + ddBx * t_arr**2
        B = np.sqrt(Bh * Bh + Bv * Bv)
        alpha = np.arctan2(Bv, Bh)
        height = ht0 + dht * t_arr + ddht * t_arr**2

        # Calculate the combined earth curvature and topography correction
        _t0 = time.perf_counter()
        er = prm1.get('earth_radius')
        drho = calc_drho(near_range, topo_vals, er,
                         height.reshape(-1, 1), B.reshape(-1, 1), alpha.reshape(-1, 1), Bx.reshape(-1, 1))

        phase_shift = (cnst * drho).astype(np.float32)
        _timings['calc_drho'] = time.perf_counter() - _t0

        _t0 = time.perf_counter()
        topo_phase = xr.DataArray(phase_shift, topo.coords)
        topo_phase = topo_phase.where(np.isfinite(topo)).rename('phase')
        _timings['xarray_wrap'] = time.perf_counter() - _t0

        # Print timing breakdown in debug mode
        if os.environ.get('INSAR_DEBUG'):
            total = sum(_timings.values())
            print(f'  PROFILE flat_earth_topo_phase: ' +
                  ' | '.join(f'{k}={v:.2f}s' for k, v in sorted(_timings.items(), key=lambda x: -x[1])) +
                  f' | total={total:.2f}s')

        return topo_phase
