# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
"""
Satellite geometry utilities for direct radar-to-geo transform.
Pure numpy implementation without disk I/O or external binaries.
"""
import numpy as np


def get_geoid(grid=None, netcdf_engine='netcdf4'):
    """Get EGM96 geoid heights, optionally interpolated to grid.

    Parameters
    ----------
    grid : xarray.DataArray, optional
        If provided, interpolate geoid to this grid's lat/lon coordinates.
    netcdf_engine : str, optional
        NetCDF engine to use: 'netcdf4' or 'h5netcdf'. Default is 'netcdf4'.

    Returns
    -------
    xarray.DataArray
        Geoid heights in meters.
    """
    import xarray as xr
    import importlib.resources as resources

    with resources.as_file(resources.files('insardev_pygmtsar.data') / 'geoid_egm96_icgem.grd') as geoid_filename:
        geoid = xr.open_dataarray(geoid_filename, engine=netcdf_engine)\
            .rename({'y': 'lat', 'x': 'lon'})\
            .astype(np.float32).transpose('lat', 'lon').rename('geoid')
    if grid is not None:
        return geoid.interp(lat=grid.lat, lon=grid.lon, method='linear')
    return geoid


def get_geoid_correction(lat, lon, netcdf_engine='netcdf4'):
    """Get EGM96 geoid correction for given coordinates.

    Memory-efficient version using scipy interpolation on numpy arrays.
    No xarray overhead - suitable for large point arrays.

    Parameters
    ----------
    lat : array-like
        Latitude coordinates (can be 1D flattened array).
    lon : array-like
        Longitude coordinates (same shape as lat).
    netcdf_engine : str, optional
        NetCDF engine to use: 'netcdf4' or 'h5netcdf'. Default is 'netcdf4'.

    Returns
    -------
    numpy.ndarray
        Geoid heights in meters (same shape as input).
    """
    from scipy.interpolate import RegularGridInterpolator
    import importlib.resources as resources

    lat = np.asarray(lat)
    lon = np.asarray(lon)

    with resources.as_file(resources.files('insardev_pygmtsar.data') / 'geoid_egm96_icgem.grd') as geoid_filename:
        if netcdf_engine == 'h5netcdf':
            import h5netcdf
            nc = h5netcdf.File(str(geoid_filename), 'r')
        else:
            from netCDF4 import Dataset
            nc = Dataset(str(geoid_filename), 'r')
        try:
            # Read coordinate arrays (small 1D)
            geoid_lat = nc.variables['y'][:]
            geoid_lon = nc.variables['x'][:]
            # Read geoid data
            geoid_z = nc.variables['z'][:].astype(np.float32)
        finally:
            nc.close()

    # Create interpolator (lat increasing required)
    if geoid_lat[0] > geoid_lat[-1]:
        geoid_lat = geoid_lat[::-1]
        geoid_z = geoid_z[::-1, :]

    interp = RegularGridInterpolator(
        (geoid_lat, geoid_lon), geoid_z,
        method='linear', bounds_error=False, fill_value=0.0
    )

    # Interpolate to requested points
    points = np.column_stack([lat.ravel(), lon.ravel()])
    result = interp(points).astype(np.float32)

    return result.reshape(lat.shape) if lat.ndim > 0 else result


def get_dem_wgs84ellipsoid(dem_path, geometry, buffer_degrees=0.04, netcdf_engine='netcdf4'):
    """Load ellipsoid-corrected DEM cropped to geometry bounds.

    Reads only the needed tile directly from disk using netCDF4 slicing.
    No full load, no lazy/dask overhead.

    Parameters
    ----------
    dem_path : str
        Path to DEM file (.tiff, .tif, .TIF, .nc, .netcdf, .grd).
    geometry : shapely.geometry
        Geometry for cropping (uses bounds).
    buffer_degrees : float, optional
        Buffer around geometry bounds in degrees. Default is 0.04.
    netcdf_engine : str, optional
        NetCDF engine to use: 'netcdf4' or 'h5netcdf'. Default is 'netcdf4'.

    Returns
    -------
    xarray.DataArray
        DEM with WGS84 ellipsoidal heights (orthometric + geoid).
    """
    import xarray as xr

    # Get bounds from geometry
    bounds = geometry.bounds  # (minx, miny, maxx, maxy)
    lon_min, lat_min, lon_max, lat_max = bounds
    lon_min -= buffer_degrees
    lat_min -= buffer_degrees
    lon_max += buffer_degrees
    lat_max += buffer_degrees

    if dem_path.endswith(('.nc', '.netcdf', '.grd')):
        # Use configured engine for direct tile slicing - no full load, no dask
        if netcdf_engine == 'h5netcdf':
            import h5netcdf
            nc = h5netcdf.File(dem_path, 'r')
        else:
            from netCDF4 import Dataset
            nc = Dataset(dem_path, 'r')
        try:
            # Get coordinate arrays
            lat_var = nc.variables.get('lat') or nc.variables.get('y')
            lon_var = nc.variables.get('lon') or nc.variables.get('x')
            lat_coords = lat_var[:].astype(np.float64)
            lon_coords = lon_var[:].astype(np.float64)

            # Find indices for the requested bounds
            lat_idx = np.where((lat_coords >= lat_min) & (lat_coords <= lat_max))[0]
            lon_idx = np.where((lon_coords >= lon_min) & (lon_coords <= lon_max))[0]

            if len(lat_idx) == 0 or len(lon_idx) == 0:
                nc.close()
                return None

            lat_start, lat_end = lat_idx[0], lat_idx[-1] + 1
            lon_start, lon_end = lon_idx[0], lon_idx[-1] + 1

            # Find the data variable (first 2D variable that's not a coordinate)
            data_var = None
            for name, var in nc.variables.items():
                if name not in ('lat', 'lon', 'x', 'y') and len(var.dimensions) == 2:
                    data_var = var
                    break
            if data_var is None:
                raise ValueError(f'No 2D data variable found in {dem_path}')

            # Read only the needed tile from disk
            ortho_vals = data_var[lat_start:lat_end, lon_start:lon_end].astype(np.float32)
            ortho_lat = lat_coords[lat_start:lat_end]
            ortho_lon = lon_coords[lon_start:lon_end]
        finally:
            nc.close()

        # Create xarray for geoid interpolation
        ortho = xr.DataArray(
            ortho_vals,
            coords={'lat': ortho_lat, 'lon': ortho_lon},
            dims=['lat', 'lon']
        )

    elif dem_path.endswith(('.tiff', '.tif', '.TIF')):
        import rioxarray as rio
        # For GeoTIFF, use rioxarray with windowed reading
        with xr.open_dataarray(dem_path, engine='rasterio') as da:
            da = da.squeeze(drop=True).rename({'y': 'lat', 'x': 'lon'})
            if da.lat.diff('lat')[0].item() < 0:
                da = da.reindex(lat=da.lat[::-1])
            ortho = da.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).load()
    else:
        raise ValueError(f'Unrecognized DEM file extension: {dem_path}')

    if ortho is None or ortho.size == 0:
        return None

    # Apply geoid correction (convert orthometric to ellipsoidal heights)
    geoid = get_geoid(ortho, netcdf_engine=netcdf_engine)
    dem_ellipsoid = ortho + geoid

    return dem_ellipsoid.astype(np.float32)


def _process_tile_worker(args):
    """Worker function for processing a single tile in spawned subprocess.

    Must be at module level for multiprocessing spawn to pickle it.
    Each worker processes one tile then exits (max_tasks_per_child=1), releasing memory.

    Like S1 burst processing: fully independent, reads DEM from disk, writes zarr chunks.
    No full arrays from main process - computes coordinates locally from grid params.

    OPTIMIZATION: Pre-computes orbit interpolation ONCE per tile (not per batch).
    This eliminates 240x overhead from repeated Hermite interpolations.
    """
    import numpy as np
    import cv2
    import zarr
    from scipy import constants

    # Unpack arguments - grid_params instead of reading from zarr
    (trans_dir, dem_path, epsg,
     tile_bounds,  # (iy, jy, ix, jx) - indices into output grid
     grid_params,  # (y_min, dy, x_min, dx) - compute coords locally
     orbit_dict, clock_start_days, prf,
     near_range, rng_samp_rate, num_lines, earth_radius,
     n_azi, n_rng, ra, e2,
     scale_factor, fill_value, row_batch, lookdir, netcdf_engine) = args

    iy, jy, ix, jx = tile_bounds
    tile_height = jy - iy
    tile_width = jx - ix
    y_min, dy, x_min, dx = grid_params

    # Debug: verify grid_params are scalars (not arrays)
    assert np.isscalar(y_min), f"y_min is not scalar: {type(y_min)}, shape={getattr(y_min, 'shape', 'N/A')}"
    assert np.isscalar(dy), f"dy is not scalar: {type(dy)}, shape={getattr(dy, 'shape', 'N/A')}"
    assert np.isscalar(x_min), f"x_min is not scalar: {type(x_min)}, shape={getattr(x_min, 'shape', 'N/A')}"
    assert np.isscalar(dx), f"dx is not scalar: {type(dx)}, shape={getattr(dx, 'shape', 'N/A')}"

    # Reconstruct orbit DataFrame
    import pandas as pd
    orbit_df = pd.DataFrame(orbit_dict)

    # Open zarr store for writing
    trans_store = zarr.storage.LocalStore(trans_dir)
    trans_root = zarr.open(trans_store, mode='r+')

    # Get zarr arrays for writing (no reading of full coordinate arrays!)
    azi_arr = trans_root['azi']
    rng_arr = trans_root['rng']
    ele_arr = trans_root['ele']

    def to_int32(arr):
        scaled = (scale_factor * arr).round()
        finite = np.isfinite(scaled)
        # Suppress warning for NaN->int cast (handled by np.where with fill_value)
        with np.errstate(invalid='ignore'):
            return np.where(finite, scaled.astype(np.int32), fill_value)

    # === PRE-COMPUTE ORBIT INTERPOLATION ONCE PER TILE ===
    # This is the key optimization - avoids 16x repeated interpolation per tile
    SOL = constants.speed_of_light
    orbit_time = orbit_df['clock'].values
    px = orbit_df['px'].values
    py = orbit_df['py'].values
    pz = orbit_df['pz'].values
    vx = orbit_df['vx'].values
    vy = orbit_df['vy'].values
    vz = orbit_df['vz'].values

    # Compute acceleration for Hermite interpolation
    dt_orb = orbit_time[1] - orbit_time[0]
    ax = np.gradient(vx, dt_orb)
    ay = np.gradient(vy, dt_orb)
    az_acc = np.gradient(vz, dt_orb)

    # Time range for azimuth lines
    t1 = 86400.0 * clock_start_days + (num_lines - num_lines) / (2.0 * prf)
    npad = 100
    azi_times = t1 + np.arange(-npad, num_lines + npad) / prf
    n_azi_times = len(azi_times)

    # Interpolate orbit at azimuth times - DONE ONCE PER TILE
    orb_x = _hermite_interp(orbit_time, px, vx, azi_times, nval=6)
    orb_y = _hermite_interp(orbit_time, py, vy, azi_times, nval=6)
    orb_z = _hermite_interp(orbit_time, pz, vz, azi_times, nval=6)
    orb_vx = _hermite_interp(orbit_time, vx, ax, azi_times, nval=6)
    orb_vy = _hermite_interp(orbit_time, vy, ay, azi_times, nval=6)
    orb_vz = _hermite_interp(orbit_time, vz, az_acc, azi_times, nval=6)

    # Range conversion constants
    range_pixel_size = SOL / (2.0 * rng_samp_rate)
    e2_wgs = (ra**2 - 6356752.31424518**2) / ra**2

    # === COMPUTE RADAR BOUNDARY POLYGON (once per tile) ===
    # Forward transform radar edges to geocoded coords, build convex hull
    # This allows skipping Doppler computation for pixels outside radar swath
    from shapely.geometry import MultiPoint, Polygon

    # Sample radar boundary: first/last azi rows, first/last rng cols
    bnd_step = 100  # Sample every 100 pixels for speed
    bnd_azi_list, bnd_rng_list = [], []

    # Pre-compute coordinate sample arrays (to ensure matching lengths)
    rng_samples = np.arange(0.5, n_rng, bnd_step, dtype=np.float32)
    azi_samples = np.arange(0.5, n_azi, bnd_step, dtype=np.float32)

    # First and last azimuth rows (all range)
    for azi_val in [0.5, n_azi - 0.5]:
        bnd_azi_list.append(np.full(len(rng_samples), azi_val, dtype=np.float32))
        bnd_rng_list.append(rng_samples.copy())

    # First and last range columns (all azimuth)
    for rng_val in [0.5, n_rng - 0.5]:
        bnd_azi_list.append(azi_samples.copy())
        bnd_rng_list.append(np.full(len(azi_samples), rng_val, dtype=np.float32))

    bnd_azi = np.concatenate(bnd_azi_list)
    bnd_rng = np.concatenate(bnd_rng_list)
    del bnd_azi_list, bnd_rng_list, rng_samples, azi_samples

    # Forward transform boundary to geocoded coords using rat2llt
    bnd_lon, bnd_lat, _ = satellite_rat2llt(
        bnd_azi, bnd_rng,
        orbit_df['clock'].values, orbit_df[['px', 'py', 'pz']].values,
        orbit_df[['vx', 'vy', 'vz']].values,
        86400.0 * clock_start_days, prf, near_range, rng_samp_rate,
        earth_radius, dem=None, max_iter=1, tol=1.0, n_chunks=1, lookdir=lookdir
    )

    # Project to output CRS and build convex hull polygon
    bnd_y, bnd_x = proj(bnd_lat, bnd_lon, from_epsg=4326, to_epsg=epsg)
    bnd_valid = np.isfinite(bnd_y) & np.isfinite(bnd_x)
    if bnd_valid.sum() > 3:
        bnd_points = MultiPoint(np.column_stack([bnd_x[bnd_valid], bnd_y[bnd_valid]]))
        radar_polygon = bnd_points.convex_hull.buffer(max(dy, dx) * 10)  # Buffer for safety
    else:
        radar_polygon = None  # Fallback: process all pixels
    del bnd_azi, bnd_rng, bnd_lon, bnd_lat, bnd_y, bnd_x, bnd_valid

    # Process tile in row batches to limit memory
    for by in range(0, tile_height, row_batch):
        ey = min(by + row_batch, tile_height)
        batch_shape = (ey - by, tile_width)

        # Compute coordinates locally from grid params (no full arrays!)
        # Grid coords are pixel centers: coord[i] = origin + spacing * (i + 0.5)
        _arr = np.arange(iy + by, iy + ey) + 0.5
        if not np.isscalar(dy):
            raise ValueError(f"dy is array! type={type(dy)}, shape={dy.shape}, by={by}, ey={ey}")
        y_batch = (y_min + dy * _arr).astype(np.float32)
        x_batch = (x_min + dx * (np.arange(ix, jx) + 0.5)).astype(np.float32)
        x_grid, y_grid = np.meshgrid(x_batch, y_batch)

        # Project batch to lon/lat
        batch_lat, batch_lon = proj(y_grid.ravel(), x_grid.ravel(), from_epsg=epsg, to_epsg=4326)
        batch_lat = np.asarray(batch_lat).reshape(batch_shape).astype(np.float32)
        batch_lon = np.asarray(batch_lon).reshape(batch_shape).astype(np.float32)

        # Check if batch intersects radar polygon - skip if entirely outside
        if radar_polygon is not None:
            from shapely.geometry import box as shapely_box
            batch_box = shapely_box(
                float(x_batch.min()), float(y_batch.min()),
                float(x_batch.max()), float(y_batch.max())
            )
            if not radar_polygon.intersects(batch_box):
                # Entire batch is outside radar coverage - skip
                del x_grid, y_grid, batch_lat, batch_lon
                continue

            # Create pixel-level mask for pixels inside radar polygon
            from shapely import vectorized
            x_flat = (x_min + dx * (np.arange(ix, jx) + 0.5))
            y_flat = (y_min + dy * (np.arange(iy + by, iy + ey) + 0.5))
            xx, yy = np.meshgrid(x_flat, y_flat)
            inside_mask = vectorized.contains(radar_polygon, xx.ravel(), yy.ravel()).reshape(batch_shape)
            del xx, yy, x_flat, y_flat
        else:
            inside_mask = np.ones(batch_shape, dtype=bool)

        del x_grid, y_grid

        # Read DEM tile from file (netCDF4 direct slicing - no full load)
        from shapely.geometry import box as shapely_box
        buffer_deg = 0.02
        batch_geom = shapely_box(
            float(np.nanmin(batch_lon)) - buffer_deg,
            float(np.nanmin(batch_lat)) - buffer_deg,
            float(np.nanmax(batch_lon)) + buffer_deg,
            float(np.nanmax(batch_lat)) + buffer_deg
        )
        dem_tile = get_dem_wgs84ellipsoid(dem_path, batch_geom, buffer_degrees=0.01, netcdf_engine=netcdf_engine)

        if dem_tile is None or dem_tile.size == 0:
            batch_ele = np.zeros(batch_shape, dtype=np.float32)
        else:
            # Interpolate DEM tile using cv2.remap
            dem_lat = dem_tile.lat.values.astype(np.float64)
            dem_lon = dem_tile.lon.values.astype(np.float64)
            dem_vals = dem_tile.values.astype(np.float32)
            dem_lat0, dem_dlat = dem_lat[0], dem_lat[1] - dem_lat[0]
            dem_lon0, dem_dlon = dem_lon[0], dem_lon[1] - dem_lon[0]

            map_row = ((batch_lat - dem_lat0) / dem_dlat).astype(np.float32)
            map_col = ((batch_lon - dem_lon0) / dem_dlon).astype(np.float32)
            batch_ele = cv2.remap(dem_vals, map_col, map_row,
                                  interpolation=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            del dem_tile, dem_lat, dem_lon, dem_vals, map_row, map_col

        # === INLINE LLT2RAT using pre-computed orbit (no function call overhead) ===
        # Only process pixels inside radar polygon (skip pixels outside for performance)
        inside_flat = inside_mask.ravel()
        n_inside = inside_flat.sum()

        if n_inside == 0:
            # All pixels outside radar coverage - skip batch
            del batch_lat, batch_lon, batch_ele, inside_mask, inside_flat
            continue

        # Extract only inside pixels for processing
        lon_flat = batch_lon.ravel()[inside_flat]
        lat_flat = batch_lat.ravel()[inside_flat]
        ele_flat = batch_ele.ravel()[inside_flat]
        n_points = n_inside

        # Convert geodetic to ECEF
        lon_rad = np.radians(lon_flat)
        lat_rad = np.radians(lat_flat)
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        N = ra / np.sqrt(1 - e2_wgs * sin_lat**2)
        xp = (N + ele_flat) * cos_lat * cos_lon
        yp = (N + ele_flat) * cos_lat * sin_lon
        zp = (N * (1 - e2_wgs) + ele_flat) * sin_lat
        del lon_rad, lat_rad, sin_lat, cos_lat, sin_lon, cos_lon, N

        # Find zero-Doppler using coarse sampling then refine
        chunk_size = 50000
        batch_azi_pix = np.zeros(n_points, dtype=np.float32)
        batch_rng_pix = np.zeros(n_points, dtype=np.float32)

        for ci in range(0, n_points, chunk_size):
            cj = min(ci + chunk_size, n_points)
            chunk_xp = xp[ci:cj]
            chunk_yp = yp[ci:cj]
            chunk_zp = zp[ci:cj]
            n_chunk = cj - ci

            # Coarse Doppler sampling
            sample_step = max(1, n_azi_times // 20)
            sample_idx = np.arange(0, n_azi_times, sample_step)
            doppler_samples = np.zeros((n_chunk, len(sample_idx)), dtype=np.float32)
            for j, idx in enumerate(sample_idx):
                delta_x = chunk_xp - orb_x[idx]
                delta_y = chunk_yp - orb_y[idx]
                delta_z = chunk_zp - orb_z[idx]
                doppler_samples[:, j] = delta_x * orb_vx[idx] + delta_y * orb_vy[idx] + delta_z * orb_vz[idx]

            # Find zero crossing
            sign_change = doppler_samples[:, :-1] * doppler_samples[:, 1:] < 0
            first_crossing = np.argmax(sign_change, axis=1)
            no_crossing = ~np.any(sign_change, axis=1)
            # Points with no Doppler zero crossing are outside radar swath
            # Don't use fallback - mark them as invalid (NaN) later
            if no_crossing.any():
                # Use index 0 as placeholder (will be marked NaN below)
                first_crossing[no_crossing] = 0

            bracket_lo = sample_idx[first_crossing]
            bracket_hi = np.minimum(sample_idx[np.minimum(first_crossing + 1, len(sample_idx) - 1)], n_azi_times - 1)

            # Vectorized Doppler refinement (no Python loop!)
            dx_lo = chunk_xp - orb_x[bracket_lo]
            dy_lo = chunk_yp - orb_y[bracket_lo]
            dz_lo = chunk_zp - orb_z[bracket_lo]
            doppler_lo = dx_lo * orb_vx[bracket_lo] + dy_lo * orb_vy[bracket_lo] + dz_lo * orb_vz[bracket_lo]

            dx_hi = chunk_xp - orb_x[bracket_hi]
            dy_hi = chunk_yp - orb_y[bracket_hi]
            dz_hi = chunk_zp - orb_z[bracket_hi]
            doppler_hi = dx_hi * orb_vx[bracket_hi] + dy_hi * orb_vy[bracket_hi] + dz_hi * orb_vz[bracket_hi]

            denom = doppler_lo - doppler_hi
            denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
            alpha = doppler_lo / denom
            # Mark pixels where alpha is outside [0,1] as invalid (zero crossing outside bracket)
            invalid_alpha = (alpha < 0) | (alpha > 1)
            alpha = np.clip(alpha, 0, 1)
            azi_idx_float = bracket_lo + alpha * (bracket_hi - bracket_lo)
            # Mark invalid pixels as NaN
            azi_idx_float[no_crossing] = np.nan
            azi_idx_float[invalid_alpha] = np.nan
            batch_azi_pix[ci:cj] = azi_idx_float - npad

            # Compute slant range (only for valid azi pixels)
            invalid_azi = np.isnan(azi_idx_float)
            # Use 0 as placeholder for invalid pixels (will be marked NaN later)
            azi_idx_safe = np.where(invalid_azi, 0.0, azi_idx_float)
            azi_idx_int = np.clip(np.floor(azi_idx_safe).astype(np.int32), 0, n_azi_times - 2)
            azi_frac = azi_idx_safe - azi_idx_int
            sat_x = orb_x[azi_idx_int] * (1 - azi_frac) + orb_x[azi_idx_int + 1] * azi_frac
            sat_y = orb_y[azi_idx_int] * (1 - azi_frac) + orb_y[azi_idx_int + 1] * azi_frac
            sat_z = orb_z[azi_idx_int] * (1 - azi_frac) + orb_z[azi_idx_int + 1] * azi_frac
            range_m = np.sqrt((chunk_xp - sat_x)**2 + (chunk_yp - sat_y)**2 + (chunk_zp - sat_z)**2)
            rng_pix = (range_m - near_range) / range_pixel_size
            # Mark range as NaN where azi is invalid
            rng_pix[invalid_azi] = np.nan
            batch_rng_pix[ci:cj] = rng_pix

        del xp, yp, zp

        # Mark out-of-bounds as NaN (in the flat inside arrays)
        out_of_bounds = ((batch_azi_pix < 0.5) | (batch_azi_pix > n_azi - 0.5) |
                        (batch_rng_pix < 0.5) | (batch_rng_pix > n_rng - 0.5))
        batch_azi_pix[out_of_bounds] = np.nan
        batch_rng_pix[out_of_bounds] = np.nan

        # Compute ele_gmtsar for inside pixels only
        sin_lat = np.sin(np.float32(np.pi / 180) * lat_flat)
        cos_lat = np.cos(np.float32(np.pi / 180) * lat_flat)
        sin_lon = np.sin(np.float32(np.pi / 180) * lon_flat)
        cos_lon = np.cos(np.float32(np.pi / 180) * lon_flat)
        N = np.float32(ra) / np.sqrt(1 - np.float32(e2) * sin_lat**2)
        Nh = N + ele_flat
        xp = Nh * cos_lat * cos_lon
        yp = Nh * cos_lat * sin_lon
        zp = (N * np.float32(1 - e2) + ele_flat) * sin_lat
        batch_ele_gmt_inside = np.sqrt(xp**2 + yp**2 + zp**2) - np.float32(earth_radius)
        batch_ele_gmt_inside[out_of_bounds] = np.nan
        del sin_lat, cos_lat, sin_lon, cos_lon, N, Nh, xp, yp, zp, out_of_bounds
        del lat_flat, lon_flat, ele_flat

        # Scatter inside results back to full batch arrays (NaN for outside pixels)
        batch_azi = np.full(batch_shape, np.nan, dtype=np.float32)
        batch_rng = np.full(batch_shape, np.nan, dtype=np.float32)
        batch_ele_gmt = np.full(batch_shape, np.nan, dtype=np.float32)
        batch_azi.ravel()[inside_flat] = batch_azi_pix
        batch_rng.ravel()[inside_flat] = batch_rng_pix
        batch_ele_gmt.ravel()[inside_flat] = batch_ele_gmt_inside
        del batch_azi_pix, batch_rng_pix, batch_ele_gmt_inside, inside_flat, inside_mask
        del batch_lat, batch_lon, batch_ele

        # Write batch to zarr
        azi_arr[iy + by:iy + ey, ix:jx] = to_int32(batch_azi)
        rng_arr[iy + by:iy + ey, ix:jx] = to_int32(batch_rng)
        ele_arr[iy + by:iy + ey, ix:jx] = to_int32(batch_ele_gmt)
        del batch_azi, batch_rng, batch_ele_gmt

    return True


def _process_topo_worker(args):
    """Worker function for computing topo (forward transform: radar → geo) for a single tile.

    Must be at module level for multiprocessing spawn to pickle it.
    Each worker processes one tile then exits (max_tasks_per_child=1), releasing memory.
    """
    import numpy as np
    import cv2
    import zarr

    # Unpack arguments
    (topo_dir, dem_path, tile_bounds, azi_coords_tile, rng_coords_tile,
     orbit_dict, clock_start_days, prf, near_range, rng_samp_rate, earth_radius,
     ra, e2, scale_factor, fill_value, row_batch, lookdir, netcdf_engine) = args

    ia, ja, ir, jr = tile_bounds
    tile_height = ja - ia
    tile_width = jr - ir

    # Reconstruct orbit DataFrame
    import pandas as pd
    orbit_df = pd.DataFrame(orbit_dict)
    orbit_time = orbit_df['clock'].values
    orbit_pos = orbit_df[['px', 'py', 'pz']].values
    orbit_vel = orbit_df[['vx', 'vy', 'vz']].values
    clock_start = 86400.0 * clock_start_days

    # Open zarr store for writing
    topo_store = zarr.storage.LocalStore(topo_dir)
    topo_root = zarr.open(topo_store, mode='r+')
    topo_arr = topo_root['topo']

    # Process tile in row batches to limit memory
    for ba in range(0, tile_height, row_batch):
        ea = min(ba + row_batch, tile_height)
        batch_shape = (ea - ba, tile_width)

        # Create radar coordinate grids for this batch
        batch_azi = azi_coords_tile[ba:ea]
        batch_rng = rng_coords_tile
        azi_grid, rng_grid = np.meshgrid(batch_azi, batch_rng, indexing='ij')

        # Forward transform: radar → lon/lat
        lon, lat, _ = satellite_rat2llt(
            azi_grid, rng_grid,
            orbit_time, orbit_pos, orbit_vel,
            clock_start, prf, near_range, rng_samp_rate, earth_radius,
            dem=None, max_iter=1, tol=0.5, n_chunks=1, lookdir=lookdir
        )
        lat = lat.astype(np.float32)
        lon = lon.astype(np.float32)
        del azi_grid, rng_grid

        # Read DEM chunk from file for this batch
        from shapely.geometry import box as shapely_box
        buffer_deg = 0.02
        batch_geom = shapely_box(
            float(np.nanmin(lon)) - buffer_deg,
            float(np.nanmin(lat)) - buffer_deg,
            float(np.nanmax(lon)) + buffer_deg,
            float(np.nanmax(lat)) + buffer_deg
        )
        dem_chunk = get_dem_wgs84ellipsoid(dem_path, batch_geom, buffer_degrees=0.01, netcdf_engine=netcdf_engine)

        if dem_chunk is None or dem_chunk.size == 0:
            ele = np.zeros(lat.shape, dtype=np.float32).ravel()
        else:
            dem_lat_arr = dem_chunk.lat.values.astype(np.float64)
            dem_lon_arr = dem_chunk.lon.values.astype(np.float64)
            dem_vals = dem_chunk.values.astype(np.float32)
            dem_lat0, dem_dlat = dem_lat_arr[0], dem_lat_arr[1] - dem_lat_arr[0]
            dem_lon0, dem_dlon = dem_lon_arr[0], dem_lon_arr[1] - dem_lon_arr[0]

            map_row = ((lat - dem_lat0) / dem_dlat).astype(np.float32)
            map_col = ((lon - dem_lon0) / dem_dlon).astype(np.float32)
            ele = cv2.remap(dem_vals, map_col, map_row,
                            interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            ele = ele.ravel().astype(np.float32)
            del dem_chunk, dem_lat_arr, dem_lon_arr, dem_vals, map_row, map_col

        lat = lat.ravel()
        lon = lon.ravel()

        # Compute ele_gmtsar
        lat_rad = np.float32(np.pi / 180) * lat
        lon_rad = np.float32(np.pi / 180) * lon
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        N = np.float32(ra) / np.sqrt(1 - np.float32(e2) * sin_lat**2)
        Nh = N + ele
        xp = Nh * cos_lat * cos_lon
        yp = Nh * cos_lat * sin_lon
        zp = (N * np.float32(1 - e2) + ele) * sin_lat
        ele_gmtsar = np.sqrt(xp**2 + yp**2 + zp**2) - np.float32(earth_radius)
        del lat, lon, ele, lat_rad, lon_rad, sin_lat, cos_lat, sin_lon, cos_lon, N, Nh, xp, yp, zp

        # Write batch to zarr
        batch_topo = ele_gmtsar.reshape(batch_shape)
        scaled = (scale_factor * batch_topo).round()
        finite = np.isfinite(scaled)
        int_data = np.where(finite, scaled.astype(np.int32), fill_value)
        topo_arr[ia + ba:ia + ea, ir:jr] = int_data
        del ele_gmtsar, batch_topo, scaled, finite, int_data

    return True


def _process_boundary_worker(args):
    """Worker function for processing a boundary chunk in spawned subprocess.

    Must be at module level for multiprocessing spawn to pickle it.
    Each worker processes one chunk then exits (max_tasks_per_child=1), releasing memory.

    Args now contain pre-sliced arrays (chunk_azi, chunk_rng) instead of full arrays.
    """
    import numpy as np
    from shapely.geometry import box

    # Unpack arguments - chunk_azi and chunk_rng are already sliced
    (chunk_azi, chunk_rng, dem_path,
     orbit_time, orbit_pos, orbit_vel, clock_start, prf,
     near_range, rng_samp_rate, earth_radius, epsg, lookdir, netcdf_engine) = args

    # Fast ellipsoid transform to get approximate lon/lat
    lon_approx, lat_approx, _ = satellite_rat2llt(
        chunk_azi, chunk_rng, orbit_time, orbit_pos, orbit_vel,
        clock_start, prf, near_range, rng_samp_rate, earth_radius,
        dem=None, max_iter=1, tol=1.0, n_chunks=1, lookdir=lookdir
    )

    # Read narrow DEM chunk from file
    buffer_deg = 0.02
    lat_min = np.nanmin(lat_approx) - buffer_deg
    lat_max = np.nanmax(lat_approx) + buffer_deg
    lon_min = np.nanmin(lon_approx) - buffer_deg
    lon_max = np.nanmax(lon_approx) + buffer_deg

    chunk_geom = box(lon_min, lat_min, lon_max, lat_max)
    dem_chunk = get_dem_wgs84ellipsoid(dem_path, chunk_geom, buffer_degrees=0.01, netcdf_engine=netcdf_engine)

    if dem_chunk is None or dem_chunk.size == 0:
        y_proj, x_proj = proj(lat_approx.ravel(), lon_approx.ravel(), from_epsg=4326, to_epsg=epsg)
        return np.asarray(y_proj).ravel().astype(np.float32), np.asarray(x_proj).ravel().astype(np.float32)

    # Refine with DEM chunk
    lon, lat, _ = satellite_rat2llt(
        chunk_azi, chunk_rng, orbit_time, orbit_pos, orbit_vel,
        clock_start, prf, near_range, rng_samp_rate, earth_radius,
        dem=dem_chunk, max_iter=10, tol=0.5, n_chunks=1, lookdir=lookdir
    )
    y_proj, x_proj = proj(lat.ravel(), lon.ravel(), from_epsg=4326, to_epsg=epsg)
    return np.asarray(y_proj).ravel().astype(np.float32), np.asarray(x_proj).ravel().astype(np.float32)


def get_utm_epsg(lat, lon):
    """Get UTM EPSG code for given lat/lon coordinates."""
    zone_num = int((lon + 180) // 6) + 1
    return 32600 + zone_num if lat >= 0 else 32700 + zone_num


def proj(ys, xs, to_epsg, from_epsg):
    """Project coordinates between EPSG codes.

    Parameters
    ----------
    ys : array-like
        Y coordinates (latitude for EPSG:4326)
    xs : array-like
        X coordinates (longitude for EPSG:4326)
    to_epsg : int
        Target EPSG code
    from_epsg : int
        Source EPSG code

    Returns
    -------
    tuple
        (ys_new, xs_new) in target CRS
    """
    from pyproj import CRS, Transformer
    from_crs = CRS.from_epsg(from_epsg)
    to_crs = CRS.from_epsg(to_epsg)
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    xs_new, ys_new = transformer.transform(xs, ys)
    del transformer, from_crs, to_crs
    return ys_new, xs_new


def _hermite_interp(x, y, dy, xp, nval=4):
    """
    Vectorized Hermite interpolation using function values and derivatives.

    Parameters
    ----------
    x : array (N,)
        Sample points (sorted, ascending)
    y : array (N,)
        Function values at sample points
    dy : array (N,)
        Derivative values at sample points
    xp : array (M,)
        Query points
    nval : int
        Number of sample points to use for interpolation (default 4)

    Returns
    -------
    array (M,)
        Interpolated values at query points
    """
    nmax = len(x)
    xp_flat = np.asarray(xp).ravel()
    M = len(xp_flat)

    # Find interpolation window for each query point
    indices = np.searchsorted(x, xp_flat)
    i0 = indices - (nval) // 2
    i0 = np.clip(i0, 0, nmax - nval)

    # Build index array: (M, nval)
    idx = i0[:, np.newaxis] + np.arange(nval)

    # Get values at interpolation points: (M, nval)
    x_vals = x[idx]
    y_vals = y[idx]
    dy_vals = dy[idx]

    # Query points as column: (M, 1)
    xp_col = xp_flat[:, np.newaxis]

    # Compute pairwise differences: (M, nval, nval)
    x_diff = x_vals[:, :, np.newaxis] - x_vals[:, np.newaxis, :]

    # xp - x_vals: (M, nval)
    xp_minus_x = xp_col - x_vals

    # Mask for j != i: (nval, nval)
    mask = ~np.eye(nval, dtype=bool)

    # Lagrange basis: hj = prod_{j!=i} (xp - x[j]) / (x[i] - x[j])
    numer = np.where(mask, xp_minus_x[:, np.newaxis, :], 1.0)
    hj = np.prod(numer, axis=2)

    denom = np.where(mask, x_diff, 1.0)
    hj_denom = np.prod(denom, axis=2)
    hj = hj / hj_denom

    # sj = sum_{j!=i} 1/(x[i] - x[j])
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_diff = np.where(mask, 1.0 / x_diff, 0.0)
    sj = np.sum(inv_diff, axis=2)

    # Hermite formula: yp = sum_i (y[i]*f0 + dy[i]*f1) * hj^2
    f0 = 1.0 - 2.0 * xp_minus_x * sj
    f1 = xp_minus_x
    hj2 = hj * hj

    return np.sum((y_vals * f0 + dy_vals * f1) * hj2, axis=1)


def _ecef_to_geodetic(X, Y, Z):
    """
    Convert ECEF coordinates to geodetic (WGS84).

    Uses Bowring's direct formula with single refinement iteration,
    accurate to ~1mm for near-Earth applications.

    Parameters
    ----------
    X, Y, Z : arrays
        ECEF coordinates in meters

    Returns
    -------
    lon, lat, height : arrays
        Longitude (deg), latitude (deg), height above ellipsoid (m)
    """
    # WGS84 constants
    a = 6378137.0
    f = 1 / 298.257223563
    b = a * (1 - f)
    e2 = (a**2 - b**2) / a**2
    ep2 = (a**2 - b**2) / b**2  # second eccentricity squared

    # Longitude (exact)
    lon = np.degrees(np.arctan2(Y, X))

    # Fast latitude using Bowring's direct formula
    p = np.sqrt(X**2 + Y**2)

    # Initial estimate using Bowring's formula (very accurate, ~0.1mm)
    theta = np.arctan2(Z * a, p * b)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    lat_rad = np.arctan2(Z + ep2 * b * sin_theta**3, p - e2 * a * cos_theta**3)

    # Single refinement iteration for sub-mm accuracy
    sin_lat = np.sin(lat_rad)
    N = a / np.sqrt(1 - e2 * sin_lat**2)
    lat_rad = np.arctan2(Z + e2 * N * sin_lat, p)

    # Height
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    N = a / np.sqrt(1 - e2 * sin_lat**2)

    with np.errstate(invalid='ignore'):
        height = np.where(
            np.abs(cos_lat) > 1e-10,
            p / cos_lat - N,
            np.abs(Z) - b
        )

    return lon, np.degrees(lat_rad), height


def _geodetic_to_ecef(lon, lat, height):
    """
    Convert geodetic coordinates to ECEF (WGS84).

    Parameters
    ----------
    lon, lat : arrays
        Longitude and latitude in degrees
    height : array
        Height above ellipsoid in meters

    Returns
    -------
    X, Y, Z : arrays
        ECEF coordinates in meters
    """
    # WGS84 constants
    a = 6378137.0
    f = 1 / 298.257223563
    e2 = f * (2 - f)

    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)

    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)

    N = a / np.sqrt(1 - e2 * sin_lat**2)

    X = (N + height) * cos_lat * cos_lon
    Y = (N + height) * cos_lat * sin_lon
    Z = (N * (1 - e2) + height) * sin_lat

    return X, Y, Z


def satellite_rat2llt(azi, rng, orbit_time, orbit_pos, orbit_vel,
                      clock_start, prf, near_range, rng_samp_rate,
                      earth_radius, dem=None, max_iter=50, tol=0.01, n_chunks=1,
                      lookdir='R'):
    """
    Direct radar-to-geographic coordinate transform with optional DEM.

    Computes geographic coordinates (lon, lat, height) for each radar pixel
    using zero-Doppler geometry. If DEM is provided, iterates to find the
    intersection with terrain surface using range-based convergence with
    per-pixel adaptive damping for sub-pixel accuracy.

    Parameters
    ----------
    azi : array
        Azimuth coordinates (line numbers, 0-based)
    rng : array
        Range coordinates (pixel numbers, 0-based)
    orbit_time : array (N,)
        Orbit state vector times (seconds from reference epoch)
    orbit_pos : array (N, 3)
        Satellite ECEF positions [x, y, z] in meters
    orbit_vel : array (N, 3)
        Satellite ECEF velocities [vx, vy, vz] in m/s
    clock_start : float
        Image start time (same units as orbit_time)
    prf : float
        Pulse repetition frequency (Hz)
    near_range : float
        Near range distance (meters)
    rng_samp_rate : float
        Range sampling rate (Hz)
    earth_radius : float
        Local Earth radius in meters (used for initial estimate)
    dem : xarray.DataArray, optional
        DEM with coords (lat, lon) and values as height above ellipsoid.
        If None, returns ellipsoidal height based on earth_radius.
    max_iter : int, optional
        Maximum iterations for DEM refinement (default 4)
    tol : float, optional
        Convergence tolerance in meters (default 0.1)

    Returns
    -------
    lon : array
        Longitude in degrees
    lat : array
        Latitude in degrees
    height : array
        Height above WGS84 ellipsoid in meters (from DEM if provided)

    Examples
    --------
    >>> # Without DEM (spherical Earth approximation):
    >>> lon, lat, h = satellite_rat2llt(
    ...     azi_grid, rng_grid, orbit_time, orbit_pos, orbit_vel,
    ...     clock_start, prf, near_range, rng_samp_rate, earth_radius
    ... )
    >>> # With DEM (iterative refinement):
    >>> lon, lat, h = satellite_rat2llt(
    ...     azi_grid, rng_grid, orbit_time, orbit_pos, orbit_vel,
    ...     clock_start, prf, near_range, rng_samp_rate, earth_radius,
    ...     dem=dem_dataarray
    ... )
    """
    import xarray as xr
    import gc

    c = 299792458.0  # Speed of light

    # Ensure float64 for time calculations (float32 causes precision loss)
    azi = np.asarray(azi, dtype=np.float64)
    rng = np.asarray(rng, dtype=np.float64)

    # Chunked processing along azimuth dimension to reduce peak memory
    if n_chunks > 1 and azi.ndim == 2:
        n_azi = azi.shape[0]
        chunk_size = (n_azi + n_chunks - 1) // n_chunks

        lon_chunks = []
        lat_chunks = []
        height_chunks = []

        for i in range(n_chunks):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, n_azi)
            if start >= n_azi:
                break

            # Process chunk
            lon_c, lat_c, h_c = satellite_rat2llt(
                azi[start:end], rng[start:end],
                orbit_time, orbit_pos, orbit_vel,
                clock_start, prf, near_range, rng_samp_rate,
                earth_radius, dem=dem, max_iter=max_iter, tol=tol, n_chunks=1,
                lookdir=lookdir
            )
            lon_chunks.append(lon_c)
            lat_chunks.append(lat_c)
            height_chunks.append(h_c)
            gc.collect()

        return np.concatenate(lon_chunks), np.concatenate(lat_chunks), np.concatenate(height_chunks)
    orbit_time = np.asarray(orbit_time)
    orbit_pos = np.asarray(orbit_pos)
    orbit_vel = np.asarray(orbit_vel)

    # Prepare orbit data for Hermite interpolation
    px, py, pz = orbit_pos[:, 0], orbit_pos[:, 1], orbit_pos[:, 2]
    vx, vy, vz = orbit_vel[:, 0], orbit_vel[:, 1], orbit_vel[:, 2]

    # Compute acceleration (derivative of velocity) for Hermite interpolation
    dt = orbit_time[1] - orbit_time[0]
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)
    az = np.gradient(vz, dt)

    # Convert azimuth to time
    if azi.ndim == 2:
        # 2D grid: azimuth constant along range axis
        azi_unique = azi[:, 0]
        t_unique = clock_start + azi_unique / prf

        # Interpolate orbit at unique azimuth times only
        # Use nval=6 to match GMTSAR SAT_llt2rat.c line 105
        Sx = _hermite_interp(orbit_time, px, vx, t_unique, nval=6)
        Sy = _hermite_interp(orbit_time, py, vy, t_unique, nval=6)
        Sz = _hermite_interp(orbit_time, pz, vz, t_unique, nval=6)
        Vx = _hermite_interp(orbit_time, vx, ax, t_unique, nval=6)
        Vy = _hermite_interp(orbit_time, vy, ay, t_unique, nval=6)
        Vz = _hermite_interp(orbit_time, vz, az, t_unique, nval=6)

        # Broadcast to full grid using float32 to save memory
        num_rng = azi.shape[1]
        Sx = np.broadcast_to(Sx[:, np.newaxis], azi.shape).astype(np.float32).copy()
        Sy = np.broadcast_to(Sy[:, np.newaxis], azi.shape).astype(np.float32).copy()
        Sz = np.broadcast_to(Sz[:, np.newaxis], azi.shape).astype(np.float32).copy()
        Vx = np.broadcast_to(Vx[:, np.newaxis], azi.shape).astype(np.float32).copy()
        Vy = np.broadcast_to(Vy[:, np.newaxis], azi.shape).astype(np.float32).copy()
        Vz = np.broadcast_to(Vz[:, np.newaxis], azi.shape).astype(np.float32).copy()
    else:
        # 1D or scalar input
        # Use nval=6 to match GMTSAR SAT_llt2rat.c line 105
        t = clock_start + azi.ravel() / prf
        Sx = _hermite_interp(orbit_time, px, vx, t, nval=6)
        Sy = _hermite_interp(orbit_time, py, vy, t, nval=6)
        Sz = _hermite_interp(orbit_time, pz, vz, t, nval=6)
        Vx = _hermite_interp(orbit_time, vx, ax, t, nval=6)
        Vy = _hermite_interp(orbit_time, vy, ay, t, nval=6)
        Vz = _hermite_interp(orbit_time, vz, az, t, nval=6)

        if azi.ndim > 0:
            Sx = Sx.reshape(azi.shape)
            Sy = Sy.reshape(azi.shape)
            Sz = Sz.reshape(azi.shape)
            Vx = Vx.reshape(azi.shape)
            Vy = Vy.reshape(azi.shape)
            Vz = Vz.reshape(azi.shape)

    # Convert range pixels to slant range
    dr = c / (2.0 * rng_samp_rate)
    slant_range = near_range + rng * dr

    # Satellite distance from Earth center
    r_sat = np.sqrt(Sx**2 + Sy**2 + Sz**2)

    # Build zero-Doppler coordinate system
    # All vectors must be perpendicular to velocity

    # Velocity unit vector - use float32 throughout
    v_mag = np.sqrt(Vx**2 + Vy**2 + Vz**2).astype(np.float32)
    vx_u = (Vx / v_mag).astype(np.float32)
    vy_u = (Vy / v_mag).astype(np.float32)
    vz_u = (Vz / v_mag).astype(np.float32)
    del v_mag, Vx, Vy, Vz  # Free velocity arrays

    # Radial unit vector (from Earth center to satellite)
    ux = (Sx / r_sat).astype(np.float32)
    uy = (Sy / r_sat).astype(np.float32)
    uz = (Sz / r_sat).astype(np.float32)

    # Project radial onto zero-Doppler plane (perpendicular to velocity)
    radial_dot_vel = (ux * vx_u + uy * vy_u + uz * vz_u).astype(np.float32)
    ux_zd = (ux - radial_dot_vel * vx_u).astype(np.float32)
    uy_zd = (uy - radial_dot_vel * vy_u).astype(np.float32)
    uz_zd = (uz - radial_dot_vel * vz_u).astype(np.float32)
    del ux, uy, uz, radial_dot_vel  # Free radial vectors
    u_zd_mag = np.sqrt(ux_zd**2 + uy_zd**2 + uz_zd**2).astype(np.float32)
    ux_zd /= u_zd_mag
    uy_zd /= u_zd_mag
    uz_zd /= u_zd_mag
    del u_zd_mag

    # Cross-track in zero-Doppler plane (nadir_zd × velocity)
    cx = (uy_zd * vz_u - uz_zd * vy_u).astype(np.float32)
    cy = (uz_zd * vx_u - ux_zd * vz_u).astype(np.float32)
    cz = (ux_zd * vy_u - uy_zd * vx_u).astype(np.float32)
    c_mag = np.sqrt(cx**2 + cy**2 + cz**2).astype(np.float32)
    cx /= c_mag
    cy /= c_mag
    cz /= c_mag
    del c_mag

    # Determine cross-track sign using GMTSAR's geometric approach:
    # det = (cross_track × velocity) · satellite_pos
    # For right-looking radar, det should be positive (cross_track points away from Earth center
    # when crossed with velocity). If negative, flip cross_track.
    # This works for any satellite (ascending/descending, left/right looking).
    # Reference: GMTSAR SAT_llt2rat.c lines 262-270
    # Use first element for arrays (all elements have same orbit geometry)
    vx0 = float(vx_u.flat[0]) if hasattr(vx_u, 'flat') else float(vx_u)
    vy0 = float(vy_u.flat[0]) if hasattr(vy_u, 'flat') else float(vy_u)
    vz0 = float(vz_u.flat[0]) if hasattr(vz_u, 'flat') else float(vz_u)
    sx0 = float(Sx.flat[0]) if hasattr(Sx, 'flat') else float(Sx)
    sy0 = float(Sy.flat[0]) if hasattr(Sy, 'flat') else float(Sy)
    sz0 = float(Sz.flat[0]) if hasattr(Sz, 'flat') else float(Sz)
    cx0 = float(cx.flat[0]) if hasattr(cx, 'flat') else float(cx)
    cy0 = float(cy.flat[0]) if hasattr(cy, 'flat') else float(cy)
    cz0 = float(cz.flat[0]) if hasattr(cz, 'flat') else float(cz)
    det_x = cy0 * vz0 - cz0 * vy0
    det_y = cz0 * vx0 - cx0 * vz0
    det_z = cx0 * vy0 - cy0 * vx0
    det = det_x * sx0 + det_y * sy0 + det_z * sz0
    # lookdir_sign: 1 for right-looking, -1 for left-looking
    lookdir_sign = -1 if lookdir.upper() == 'L' else 1
    if det * lookdir_sign < 0:
        cx, cy, cz = -cx, -cy, -cz
    del vx_u, vy_u, vz_u  # Free velocity unit vectors

    # Initial estimate using spherical Earth approximation
    cos_look = ((r_sat**2 + slant_range**2 - earth_radius**2) / (2.0 * r_sat * slant_range)).astype(np.float32)
    cos_look = np.clip(cos_look, -1.0, 1.0)
    sin_look = np.sqrt(1.0 - cos_look**2).astype(np.float32)

    # Target position in ECEF (zero-Doppler geometry)
    Tx = (Sx - slant_range * cos_look * ux_zd + slant_range * sin_look * cx).astype(np.float32)
    Ty = (Sy - slant_range * cos_look * uy_zd + slant_range * sin_look * cy).astype(np.float32)
    Tz = (Sz - slant_range * cos_look * uz_zd + slant_range * sin_look * cz).astype(np.float32)

    # Convert to geodetic coordinates
    lon, lat, height = _ecef_to_geodetic(Tx, Ty, Tz)
    del Tx, Ty, Tz  # Free ECEF target positions

    # If no DEM provided, return ellipsoidal result
    if dem is None:
        # Free remaining large arrays
        del Sx, Sy, Sz, r_sat, slant_range, ux_zd, uy_zd, uz_zd, cx, cy, cz, cos_look, sin_look
        return lon, lat, height

    # Iterative refinement with DEM
    # The target must satisfy: |T - S| = slant_range AND T is on DEM surface

    # Pre-extract DEM data for fast cv2 interpolation
    import cv2
    dem_values = dem.values.astype(np.float32)
    dem_lat = dem.lat.values
    dem_lon = dem.lon.values
    lat_min, lat_max = dem_lat.min(), dem_lat.max()
    lon_min, lon_max = dem_lon.min(), dem_lon.max()
    lat_step = (lat_max - lat_min) / (len(dem_lat) - 1)
    lon_step = (lon_max - lon_min) / (len(dem_lon) - 1)
    n_lat, n_lon = dem_values.shape

    # Handle both ascending and descending lat coordinates
    lat_ascending = dem_lat[1] > dem_lat[0]

    # Chunk size aligned with HDF5 chunks (512x512) - 8192 = 16x512
    CHUNK_SIZE = 8192

    def fast_dem_interp(lat_pts, lon_pts):
        """Fast cubic interpolation using cv2.remap with 2D chunking for large grids."""
        original_shape = lat_pts.shape
        is_1d = lat_pts.ndim == 1

        # Convert geographic coords to pixel indices
        if lat_ascending:
            map_y = ((lat_pts - lat_min) / lat_step).astype(np.float32)
        else:
            map_y = ((lat_max - lat_pts) / lat_step).astype(np.float32)
        map_x = ((lon_pts - lon_min) / lon_step).astype(np.float32)

        # Mark out-of-bounds
        out_of_bounds = (map_y < 0) | (map_y >= n_lat - 1) | \
                        (map_x < 0) | (map_x >= n_lon - 1) | \
                        ~np.isfinite(map_y) | ~np.isfinite(map_x)

        map_y_safe = np.where(out_of_bounds, 0, map_y)
        map_x_safe = np.where(out_of_bounds, 0, map_x)

        if is_1d:
            n_rows, n_cols = lat_pts.size, 1
        else:
            n_rows, n_cols = original_shape

        needs_chunking = n_rows > CHUNK_SIZE or n_cols > CHUNK_SIZE

        if not needs_chunking:
            # Fast path: small grid
            result = cv2.remap(dem_values, map_x_safe, map_y_safe,
                               interpolation=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # Reshape to original input shape (cv2.remap returns (N,1) for 1D inputs)
            result = result.reshape(original_shape)
        else:
            result = np.empty(original_shape, dtype=np.float32)

            # 2D chunking for large grids
            for iy in range(0, n_rows, CHUNK_SIZE):
                jy = min(iy + CHUNK_SIZE, n_rows)
                if is_1d:
                    chunk_result = cv2.remap(dem_values,
                                             map_x_safe[iy:jy], map_y_safe[iy:jy],
                                             interpolation=cv2.INTER_CUBIC,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    result[iy:jy] = chunk_result.ravel()
                else:
                    for ix in range(0, n_cols, CHUNK_SIZE):
                        jx = min(ix + CHUNK_SIZE, n_cols)
                        chunk_result = cv2.remap(dem_values,
                                                 map_x_safe[iy:jy, ix:jx],
                                                 map_y_safe[iy:jy, ix:jx],
                                                 interpolation=cv2.INTER_CUBIC,
                                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                        result[iy:jy, ix:jx] = chunk_result

        result[out_of_bounds] = np.nan
        return result

    # Damped iteration to converge on slant range
    # The target must satisfy: distance(target, satellite) = slant_range
    # AND target lies on DEM surface
    # Uses fast inline ECEF-to-geodetic, damped updates in geodetic space
    damping = np.full_like(lon, 0.5)
    prev_range_diff = None

    # WGS84 constants for fast ECEF to lon/lat
    a_wgs = 6378137.0
    f_wgs = 1 / 298.257223563
    b_wgs = a_wgs * (1 - f_wgs)
    e2_wgs = (a_wgs**2 - b_wgs**2) / a_wgs**2
    ep2_wgs = (a_wgs**2 - b_wgs**2) / b_wgs**2

    # Track best result
    best_lon = lon.copy()
    best_lat = lat.copy()
    best_diff = np.full_like(lon, np.inf)

    for iteration in range(max_iter):
        # Get DEM height at current (lat, lon)
        dem_height = fast_dem_interp(lat, lon)

        # Compute target ECEF on DEM surface
        Tx_dem, Ty_dem, Tz_dem = _geodetic_to_ecef(lon, lat, dem_height)

        # Compute actual slant range to DEM point
        actual_range = np.sqrt((Tx_dem - Sx)**2 + (Ty_dem - Sy)**2 + (Tz_dem - Sz)**2)

        # Signed range difference for oscillation detection
        range_diff_signed = actual_range - slant_range
        range_diff = np.abs(range_diff_signed)

        # Track best result per pixel
        better = range_diff < best_diff
        best_lon = np.where(better, lon, best_lon)
        best_lat = np.where(better, lat, best_lat)
        best_diff = np.where(better, range_diff, best_diff)

        # Check convergence on RANGE
        valid_mask = np.isfinite(range_diff)
        if not valid_mask.any():
            break  # All pixels outside DEM
        max_diff = range_diff[valid_mask].max()
        if max_diff < tol:
            break

        # Per-pixel adaptive damping: reduce when oscillating (sign change)
        if prev_range_diff is not None:
            oscillating = range_diff_signed * prev_range_diff < 0
            damping = np.where(oscillating, np.maximum(0.1, damping * 0.7), damping)
        prev_range_diff = range_diff_signed.copy()

        # Target distance from Earth center (on DEM)
        r_target = np.sqrt(Tx_dem**2 + Ty_dem**2 + Tz_dem**2)

        # Recompute look angle to place target at exact slant_range
        cos_look = (r_sat**2 + slant_range**2 - r_target**2) / (2.0 * r_sat * slant_range)
        cos_look = np.clip(cos_look, -1.0, 1.0)
        sin_look = np.sqrt(1.0 - cos_look**2)

        # Compute new target position at exact slant_range (in ECEF)
        Tx_new = Sx - slant_range * cos_look * ux_zd + slant_range * sin_look * cx
        Ty_new = Sy - slant_range * cos_look * uy_zd + slant_range * sin_look * cy
        Tz_new = Sz - slant_range * cos_look * uz_zd + slant_range * sin_look * cz

        # Fast inline ECEF to geodetic (Bowring's direct formula)
        p_new = np.sqrt(Tx_new**2 + Ty_new**2)
        lon_new = np.degrees(np.arctan2(Ty_new, Tx_new))
        theta = np.arctan2(Tz_new * a_wgs, p_new * b_wgs)
        lat_new = np.degrees(np.arctan2(
            Tz_new + ep2_wgs * b_wgs * np.sin(theta)**3,
            p_new - e2_wgs * a_wgs * np.cos(theta)**3
        ))

        # Damped update in geodetic space
        lon = lon + damping * (lon_new - lon)
        lat = lat + damping * (lat_new - lat)

    # Use best result
    lon = np.where(best_diff < np.inf, best_lon, lon)
    lat = np.where(best_diff < np.inf, best_lat, lat)
    height = fast_dem_interp(lat, lon)

    # Free all large intermediate arrays before returning
    del Sx, Sy, Sz, r_sat, slant_range
    del ux_zd, uy_zd, uz_zd, cx, cy, cz
    del dem_values, cos_look, sin_look

    return lon, lat, height


def satellite_baseline(orbit_df1: "pd.DataFrame", orbit_df2: "pd.DataFrame",
                       clock_start: float, prf: float,
                       num_valid_az: int, num_patches: int, nrows: int,
                       earth_radius: float = None, SC_height: float = None,
                       near_range: float = None, num_rng_bins: int = None,
                       rng_samp_rate: float = None,
                       clock_start_rep: float = None,
                       num_valid_az_rep: int = None, num_patches_rep: int = None,
                       nrows_rep: int = None, prf_rep: float = None) -> dict:
    """
    Compute satellite baseline between two acquisitions.

    Pure Python replacement for GMTSAR SAT_baseline binary.
    Uses GMTSAR's exact algorithm: search repeat orbit for closest point
    to reference scene start, then compute perpendicular baseline using
    look angle geometry.

    Parameters
    ----------
    orbit_df1 : pd.DataFrame
        Orbit state vectors for reference image.
        Must have columns: px, py, pz, vx, vy, vz, isec
    orbit_df2 : pd.DataFrame
        Orbit state vectors for secondary image.
    clock_start : float
        Reference image start time in days (from PRM clock_start)
    prf : float
        Reference image pulse repetition frequency in Hz
    num_valid_az : int
        Reference image number of valid azimuth lines per patch
    num_patches : int
        Reference image number of patches
    nrows : int
        Reference image total number of rows
    earth_radius : float, optional
        Local Earth radius (meters). Required for GMTSAR-style computation.
    SC_height : float, optional
        Satellite height above Earth radius (meters). Required for GMTSAR-style.
    near_range : float, optional
        Near range distance (meters). Required for GMTSAR-style.
    num_rng_bins : int, optional
        Number of range bins. Required for GMTSAR-style.
    rng_samp_rate : float, optional
        Range sampling rate (Hz). Required for GMTSAR-style.
    clock_start_rep : float, optional
        Repeat image start time in days. If None, uses clock_start.
    num_valid_az_rep : int, optional
        Repeat image num_valid_az. If None, uses num_valid_az.
    num_patches_rep : int, optional
        Repeat image num_patches. If None, uses num_patches.
    nrows_rep : int, optional
        Repeat image nrows. If None, uses nrows.
    prf_rep : float, optional
        Repeat image PRF. If None, uses prf.

    Returns
    -------
    dict
        Dictionary containing:
        - B_parallel: Along-track baseline component (meters)
        - B_perpendicular: Cross-track baseline component (meters)
        - baseline: Total baseline length (meters)

    Examples
    --------
    >>> baseline = satellite_baseline(
    ...     orbit_df_ref, orbit_df_sec,
    ...     prm_ref.get('clock_start'),
    ...     prm_ref.get('PRF'),
    ...     prm_ref.get('num_valid_az'),
    ...     prm_ref.get('num_patches'),
    ...     prm_ref.get('nrows'),
    ...     earth_radius=prm_ref.get('earth_radius'),
    ...     SC_height=prm_ref.get('SC_height'),
    ...     near_range=prm_ref.get('near_range'),
    ...     num_rng_bins=prm_ref.get('num_rng_bins'),
    ...     rng_samp_rate=prm_ref.get('rng_samp_rate')
    ... )
    >>> print(f"Perpendicular baseline: {baseline['B_perpendicular']:.1f} m")
    """
    # Default repeat parameters to reference if not provided
    if clock_start_rep is None:
        clock_start_rep = clock_start
    if num_valid_az_rep is None:
        num_valid_az_rep = num_valid_az
    if num_patches_rep is None:
        num_patches_rep = num_patches
    if nrows_rep is None:
        nrows_rep = nrows
    if prf_rep is None:
        prf_rep = prf

    # Compute reference scene START time (not center - this is key for GMTSAR match)
    time_of_day_start = (clock_start % 1.0) * 86400.0
    scene_duration = num_patches * num_valid_az / prf
    t11 = time_of_day_start + (nrows - num_valid_az) / (2.0 * prf)  # Scene start
    t12 = t11 + scene_duration  # Scene end

    # Compute repeat scene start time
    time_of_day_start_rep = (clock_start_rep % 1.0) * 86400.0
    scene_duration_rep = num_patches_rep * num_valid_az_rep / prf_rep
    t21 = time_of_day_start_rep + (nrows_rep - num_valid_az_rep) / (2.0 * prf_rep)

    orbit_time1 = orbit_df1['isec'].values
    orbit_time2 = orbit_df2['isec'].values

    # Reference satellite position at scene START
    x11 = _hermite_interp(orbit_time1, orbit_df1['px'].values, orbit_df1['vx'].values, np.array([t11]))[0]
    y11 = _hermite_interp(orbit_time1, orbit_df1['py'].values, orbit_df1['vy'].values, np.array([t11]))[0]
    z11 = _hermite_interp(orbit_time1, orbit_df1['pz'].values, orbit_df1['vz'].values, np.array([t11]))[0]

    # Search repeat orbit for closest point to reference start position
    # This is the key GMTSAR algorithm - find minimum distance, not same time
    dt = 0.5 / prf
    ns = int((t12 - t11) / dt)
    ns2 = int(ns * 0.5)  # 50% extension for search

    # Vectorized search: batch all search times and interpolate once
    k_vals = np.arange(-ns2, ns + ns2)
    ts_all = t21 + k_vals * dt
    # Filter to valid orbit time range
    valid_mask = (ts_all >= orbit_time2[0]) & (ts_all <= orbit_time2[-1])
    ts_valid = ts_all[valid_mask]
    k_valid = k_vals[valid_mask]

    # Batch interpolation (3 calls instead of ~6000×3)
    xs_all = _hermite_interp(orbit_time2, orbit_df2['px'].values, orbit_df2['vx'].values, ts_valid, nval=6)
    ys_all = _hermite_interp(orbit_time2, orbit_df2['py'].values, orbit_df2['vy'].values, ts_valid, nval=6)
    zs_all = _hermite_interp(orbit_time2, orbit_df2['pz'].values, orbit_df2['vz'].values, ts_valid, nval=6)

    # Vectorized distance computation
    ds_all = np.sqrt((xs_all - x11)**2 + (ys_all - y11)**2 + (zs_all - z11)**2)
    min_idx = np.argmin(ds_all)
    m1 = k_valid[min_idx]

    # Polynomial refinement for precise minimum (GMTSAR's poly_interp)
    t_coarse = t21 + m1 * dt
    ntt = 100
    times = np.array([(k - ntt/2 + 0.5) * 0.01 / ntt for k in range(ntt)])

    # Batch interpolation for refinement (3 calls instead of 100×3)
    ts_refine = t_coarse + times
    xs_refine = _hermite_interp(orbit_time2, orbit_df2['px'].values, orbit_df2['vx'].values, ts_refine, nval=6)
    ys_refine = _hermite_interp(orbit_time2, orbit_df2['py'].values, orbit_df2['vy'].values, ts_refine, nval=6)
    zs_refine = _hermite_interp(orbit_time2, orbit_df2['pz'].values, orbit_df2['vz'].values, ts_refine, nval=6)
    ds_refine = np.sqrt((xs_refine - x11)**2 + (ys_refine - y11)**2 + (zs_refine - z11)**2)
    bs_sq = ds_refine * ds_refine

    # Polynomial fit: bs_sq = d0 + d1*t + d2*t^2
    coeffs = np.polyfit(times, bs_sq, 2)
    d2, d1, d0 = coeffs[0], coeffs[1], coeffs[2]

    # Minimum at t = -d1/(2*d2), baseline = sqrt(d0 - d1^2/(4*d2))
    t_min_offset = -d1 / (2.0 * d2)
    with np.errstate(invalid='ignore'):
        baseline_start = np.sqrt(d0 - d1**2 / (4.0 * d2))

    # Get repeat position at refined minimum
    t_refined = t_coarse + t_min_offset
    x21 = _hermite_interp(orbit_time2, orbit_df2['px'].values, orbit_df2['vx'].values, np.array([t_refined]))[0]
    y21 = _hermite_interp(orbit_time2, orbit_df2['py'].values, orbit_df2['vy'].values, np.array([t_refined]))[0]
    z21 = _hermite_interp(orbit_time2, orbit_df2['pz'].values, orbit_df2['vz'].values, np.array([t_refined]))[0]

    # Compute radial unit vector at reference start
    r1 = np.sqrt(x11**2 + y11**2 + z11**2)
    xu1, yu1, zu1 = x11/r1, y11/r1, z11/r1

    # Vertical baseline (radial component)
    bv1 = (x21 - x11) * xu1 + (y21 - y11) * yu1 + (z21 - z11) * zu1

    # Get sign (GMTSAR's get_sign function)
    rlnref = np.arctan2(y11, x11)
    rlnrep = np.arctan2(y21, x21)

    # Compute derivatives for hermite interpolation of each velocity component
    dt_orb = orbit_time1[1] - orbit_time1[0]
    ax1 = np.gradient(orbit_df1['vx'].values, dt_orb)  # x-acceleration for vx interpolation
    az1 = np.gradient(orbit_df1['vz'].values, dt_orb)  # z-acceleration for vz interpolation

    # Check orbit direction from z-velocity at scene start
    vz1 = _hermite_interp(orbit_time1, orbit_df1['vz'].values, az1, np.array([t11]))[0]

    sign = 1
    if vz1 < 0:  # Descending orbit
        sign = -sign
    # Note: GMTSAR also checks lookdir for left-looking, but Sentinel-1 is right-looking
    # so this check is omitted (would only flip sign for lookdir == "L")
    sign_after_orb = sign
    if rlnrep < rlnref:
        sign = -sign

    # Debug output
    import os
    if os.environ.get('INSAR_DEBUG'):
        print(f'  DEBUG satellite_baseline: vz1={vz1:.2f}, rlnref={np.degrees(rlnref):.4f}°, '
              f'rlnrep={np.degrees(rlnrep):.4f}°, sign_after_orb={sign_after_orb}, '
              f'rlnrep<rlnref={rlnrep < rlnref}, final_sign={sign}, '
              f'baseline={baseline_start:.2f}, bv={bv1:.2f}')

    # Horizontal baseline with sign
    bh1 = sign * np.sqrt(max(0, baseline_start**2 - bv1**2))

    # Alpha angle (from horizontal)
    alpha1 = np.arctan2(bv1, bh1)

    # Check if we have all parameters for GMTSAR-style look angle computation
    if all(p is not None for p in [earth_radius, SC_height, near_range, num_rng_bins, rng_samp_rate]):
        # GMTSAR-style look angle computation
        c = 299792458.0
        dr = c / (2.0 * rng_samp_rate)
        rc = earth_radius + SC_height
        ra = earth_radius
        far_range = near_range + dr * num_rng_bins

        # Look angle at mid-range (average of near and far)
        arg1 = (near_range**2 + rc**2 - ra**2) / (2.0 * near_range * rc)
        arg2 = (far_range**2 + rc**2 - ra**2) / (2.0 * far_range * rc)
        rlook = np.arccos(np.clip((arg1 + arg2) / 2.0, -1, 1))

        # Add incidence angle correction
        arg1 = (-near_range**2 + rc**2 + ra**2) / (2.0 * ra * rc)
        arg2 = (-far_range**2 + rc**2 + ra**2) / (2.0 * ra * rc)
        rlook = rlook + np.arccos(np.clip((arg1 + arg2) / 2.0, -1, 1))

        # Final GMTSAR-style computation
        B_parallel = baseline_start * np.sin(rlook - alpha1)
        B_perpendicular = baseline_start * np.cos(rlook - alpha1)
    else:
        # Fallback: simple geometric baseline using velocity direction
        vx1 = _hermite_interp(orbit_time1, orbit_df1['vx'].values, ax1, np.array([t11]))[0]
        vy1_val = _hermite_interp(orbit_time1, orbit_df1['vy'].values,
                                   np.gradient(orbit_df1['vy'].values, dt_orb), np.array([t11]))[0]
        vz1_val = _hermite_interp(orbit_time1, orbit_df1['vz'].values, az1, np.array([t11]))[0]

        v1 = np.sqrt(vx1**2 + vy1_val**2 + vz1_val**2)
        uv = np.array([vx1/v1, vy1_val/v1, vz1_val/v1])
        ur = np.array([xu1, yu1, zu1])
        uc = np.cross(ur, uv)
        uc = uc / np.linalg.norm(uc)

        db = np.array([x21 - x11, y21 - y11, z21 - z11])
        B_parallel = np.dot(db, uv)
        B_perpendicular = -np.dot(db, uc)

    # Now compute baseline at center and end positions for full GMTSAR compatibility
    t1c = (t11 + t12) / 2.0  # Scene center
    t1e = t12  # Scene end

    def compute_baseline_at_time(t1_pos):
        """Compute baseline components at a specific reference scene time."""
        # Reference satellite position at this time
        x1 = _hermite_interp(orbit_time1, orbit_df1['px'].values, orbit_df1['vx'].values, np.array([t1_pos]))[0]
        y1 = _hermite_interp(orbit_time1, orbit_df1['py'].values, orbit_df1['vy'].values, np.array([t1_pos]))[0]
        z1 = _hermite_interp(orbit_time1, orbit_df1['pz'].values, orbit_df1['vz'].values, np.array([t1_pos]))[0]

        # Search repeat orbit for closest point - vectorized
        t2_search_start = t21 + (t1_pos - t11)  # Approximate corresponding time in repeat

        # Batch all search times
        k_vals_local = np.arange(-ns2, ns + ns2)
        ts_all_local = t2_search_start + k_vals_local * dt
        valid_mask_local = (ts_all_local >= orbit_time2[0]) & (ts_all_local <= orbit_time2[-1])
        ts_valid_local = ts_all_local[valid_mask_local]
        k_valid_local = k_vals_local[valid_mask_local]

        # Batch interpolation
        xs_all_local = _hermite_interp(orbit_time2, orbit_df2['px'].values, orbit_df2['vx'].values, ts_valid_local, nval=6)
        ys_all_local = _hermite_interp(orbit_time2, orbit_df2['py'].values, orbit_df2['vy'].values, ts_valid_local, nval=6)
        zs_all_local = _hermite_interp(orbit_time2, orbit_df2['pz'].values, orbit_df2['vz'].values, ts_valid_local, nval=6)

        ds_all_local = np.sqrt((xs_all_local - x1)**2 + (ys_all_local - y1)**2 + (zs_all_local - z1)**2)
        min_idx_local = np.argmin(ds_all_local)
        m_best = k_valid_local[min_idx_local]

        # Polynomial refinement - vectorized
        t_coarse = t2_search_start + m_best * dt
        times_local = np.array([(k - ntt/2 + 0.5) * 0.01 / ntt for k in range(ntt)])

        # Batch interpolation for refinement
        ts_refine_local = t_coarse + times_local
        xs_refine_local = _hermite_interp(orbit_time2, orbit_df2['px'].values, orbit_df2['vx'].values, ts_refine_local, nval=6)
        ys_refine_local = _hermite_interp(orbit_time2, orbit_df2['py'].values, orbit_df2['vy'].values, ts_refine_local, nval=6)
        zs_refine_local = _hermite_interp(orbit_time2, orbit_df2['pz'].values, orbit_df2['vz'].values, ts_refine_local, nval=6)
        ds_refine_local = np.sqrt((xs_refine_local - x1)**2 + (ys_refine_local - y1)**2 + (zs_refine_local - z1)**2)
        bs_sq_local = ds_refine_local * ds_refine_local

        coeffs_local = np.polyfit(times_local, bs_sq_local, 2)
        d2_l, d1_l, d0_l = coeffs_local[0], coeffs_local[1], coeffs_local[2]
        t_min_local = -d1_l / (2.0 * d2_l)
        with np.errstate(invalid='ignore'):
            baseline_val = np.sqrt(d0_l - d1_l**2 / (4.0 * d2_l))

        # Get repeat position at refined minimum
        t_ref = t_coarse + t_min_local
        x2 = _hermite_interp(orbit_time2, orbit_df2['px'].values, orbit_df2['vx'].values, np.array([t_ref]))[0]
        y2 = _hermite_interp(orbit_time2, orbit_df2['py'].values, orbit_df2['vy'].values, np.array([t_ref]))[0]
        z2 = _hermite_interp(orbit_time2, orbit_df2['pz'].values, orbit_df2['vz'].values, np.array([t_ref]))[0]

        # Radial unit vector
        r1 = np.sqrt(x1**2 + y1**2 + z1**2)
        xu, yu, zu = x1/r1, y1/r1, z1/r1

        # Vertical baseline
        bv = (x2 - x1) * xu + (y2 - y1) * yu + (z2 - z1) * zu

        # Sign: start from orbit direction only, then apply longitude check for THIS position
        # Bug fix: previously used 'sign' which already included start position's longitude check,
        # causing double-flip if center/end positions have different longitude relationship
        rlnref = np.arctan2(y1, x1)
        rlnrep = np.arctan2(y2, x2)
        sign_local = sign_after_orb  # Start from orbit direction only
        if rlnrep < rlnref:
            sign_local = -sign_local

        # Horizontal baseline
        bh = sign_local * np.sqrt(max(0, baseline_val**2 - bv**2))

        # Alpha angle
        alpha_val = np.arctan2(bv, bh)

        # B_offset (along-track component) - compute using velocity direction
        vx = _hermite_interp(orbit_time1, orbit_df1['vx'].values, ax1, np.array([t1_pos]))[0]
        vy_v = _hermite_interp(orbit_time1, orbit_df1['vy'].values,
                               np.gradient(orbit_df1['vy'].values, dt_orb), np.array([t1_pos]))[0]
        vz_v = _hermite_interp(orbit_time1, orbit_df1['vz'].values, az1, np.array([t1_pos]))[0]
        v_mag = np.sqrt(vx**2 + vy_v**2 + vz_v**2)
        uv = np.array([vx/v_mag, vy_v/v_mag, vz_v/v_mag])
        db = np.array([x2 - x1, y2 - y1, z2 - z1])
        b_offset = np.dot(db, uv)  # Along-track component

        # SC_height at this position
        sc_height = r1 - earth_radius if earth_radius is not None else r1 - 6371000.0

        return baseline_val, alpha_val, b_offset, sc_height

    # Compute at center and end
    baseline_center, alpha_center, b_offset_center, sc_height_center = compute_baseline_at_time(t1c)
    baseline_end, alpha_end, b_offset_end, sc_height_end = compute_baseline_at_time(t1e)

    # Get SC_height at start from original computation
    r1_start = np.sqrt(x11**2 + y11**2 + z11**2)
    sc_height_start = r1_start - earth_radius if earth_radius is not None else r1_start - 6371000.0

    # B_offset at start (using the already computed values)
    vx1 = _hermite_interp(orbit_time1, orbit_df1['vx'].values, ax1, np.array([t11]))[0]
    vy1_val = _hermite_interp(orbit_time1, orbit_df1['vy'].values,
                               np.gradient(orbit_df1['vy'].values, dt_orb), np.array([t11]))[0]
    vz1_val = _hermite_interp(orbit_time1, orbit_df1['vz'].values, az1, np.array([t11]))[0]
    v1 = np.sqrt(vx1**2 + vy1_val**2 + vz1_val**2)
    uv_start = np.array([vx1/v1, vy1_val/v1, vz1_val/v1])
    db_start = np.array([x21 - x11, y21 - y11, z21 - z11])
    b_offset_start = np.dot(db_start, uv_start)

    # Convert alpha to degrees for output
    alpha_start_deg = float(np.degrees(alpha1))
    alpha_center_deg = float(np.degrees(alpha_center))
    alpha_end_deg = float(np.degrees(alpha_end))

    # Validate baseline parameter consistency
    # Alpha should vary smoothly along the scene (< 5° variation is reasonable for ~2.5s burst)
    # Exception: when baseline is nearly vertical (|alpha| near 90°), small bh changes can flip sign
    def alpha_variation(a1, a2):
        """Compute alpha variation, handling ±90° wraparound for vertical baselines."""
        diff = abs(a1 - a2)
        # For nearly vertical baselines, +90° and -90° are effectively the same
        # (both mean bh≈0, just different sign). Check wrapped difference.
        # If both alphas are near ±90° and diff is large, it's a vertical baseline flip
        near_vertical = (abs(abs(a1) - 90) < 15) and (abs(abs(a2) - 90) < 15)
        if near_vertical and diff > 90:
            diff = 180 - diff  # Wrapped difference for vertical baseline flip
        return diff

    # Only validate when baseline is significant (> 1m)
    # For self-baseline (baseline ≈ 0), alpha is numerically undefined (arctan2 of tiny values)
    baseline_mean = (baseline_start + baseline_center + baseline_end) / 3
    if baseline_mean > 1.0:
        # Alpha should vary smoothly along the scene (< 5° variation)
        alpha_var_sc = alpha_variation(alpha_start_deg, alpha_center_deg)
        alpha_var_ce = alpha_variation(alpha_center_deg, alpha_end_deg)
        assert alpha_var_sc < 5.0, (
            f"Alpha varies too much between start and center: {alpha_var_sc:.2f}° "
            f"(start={alpha_start_deg:.2f}°, center={alpha_center_deg:.2f}°). "
            f"This indicates a sign computation bug."
        )
        assert alpha_var_ce < 5.0, (
            f"Alpha varies too much between center and end: {alpha_var_ce:.2f}° "
            f"(center={alpha_center_deg:.2f}°, end={alpha_end_deg:.2f}°). "
            f"This indicates a sign computation bug."
        )
        # Baseline length should vary smoothly (< 10% variation)
        baseline_var_sc = abs(baseline_start - baseline_center) / baseline_mean * 100
        baseline_var_ce = abs(baseline_center - baseline_end) / baseline_mean * 100
        assert baseline_var_sc < 10.0, (
            f"Baseline varies too much between start and center: {baseline_var_sc:.1f}% "
            f"(start={baseline_start:.2f}m, center={baseline_center:.2f}m)"
        )
        assert baseline_var_ce < 10.0, (
            f"Baseline varies too much between center and end: {baseline_var_ce:.1f}% "
            f"(center={baseline_center:.2f}m, end={baseline_end:.2f}m)"
        )

    # SC_height should vary smoothly
    # For S1 bursts (~2.5s): <100m variation
    # For NISAR scenes (~35s): allow more variation (up to 250m for longer acquisitions)
    sc_height_var_sc = abs(sc_height_start - sc_height_center)
    sc_height_var_ce = abs(sc_height_center - sc_height_end)
    sc_height_threshold = 250.0  # meters
    assert sc_height_var_sc < sc_height_threshold, (
        f"SC_height varies too much between start and center: {sc_height_var_sc:.1f}m "
        f"(start={sc_height_start:.1f}m, center={sc_height_center:.1f}m)"
    )
    assert sc_height_var_ce < sc_height_threshold, (
        f"SC_height varies too much between center and end: {sc_height_var_ce:.1f}m "
        f"(center={sc_height_center:.1f}m, end={sc_height_end:.1f}m)"
    )

    return {
        'B_parallel': float(B_parallel),
        'B_perpendicular': float(B_perpendicular),
        'baseline': float(baseline_start),
        # Time-varying baseline parameters (for topo phase computation)
        'baseline_start': float(baseline_start),
        'baseline_center': float(baseline_center),
        'baseline_end': float(baseline_end),
        'alpha_start': alpha_start_deg,
        'alpha_center': alpha_center_deg,
        'alpha_end': alpha_end_deg,
        'B_offset_start': float(b_offset_start),
        'B_offset_center': float(b_offset_center),
        'B_offset_end': float(b_offset_end),
        # SC_height parameters
        'SC_height': float(sc_height_center),  # Center value as main
        'SC_height_start': float(sc_height_start),
        'SC_height_end': float(sc_height_end),
    }


def satellite_llt2rat(lon: np.ndarray, lat: np.ndarray, elevation: np.ndarray,
                      orbit_df: "pd.DataFrame",
                      clock_start: float,
                      prf: float,
                      near_range: float,
                      rng_samp_rate: float,
                      num_valid_az: int,
                      num_patches: int,
                      nrows: int,
                      earth_radius: float,
                      ra: float = 6378137.0,
                      rc: float = 6356752.31424518,
                      precise: int = 1,
                      lookdir: str = 'R',
                      fd1: float = 0.0,
                      fdd1: float = 0.0,
                      wavelength: float = None,
                      vel: float = None,
                      num_rng_bins: int = None,
                      rshift: int = 0,
                      ashift: int = 0,
                      sub_int_r: float = 0.0,
                      sub_int_a: float = 0.0,
                      chirp_ext: int = 0,
                      debug: bool = False) -> np.ndarray:
    """
    Convert geographic coordinates (LLT) to radar coordinates (RAT).

    Replaces GMTSAR SAT_llt2rat binary. Optimized using coarse orbit sampling
    with Doppler-based search instead of brute-force distance minimization.

    Parameters
    ----------
    lon : array
        Longitude in degrees
    lat : array
        Latitude in degrees
    elevation : array
        Elevation above reference ellipsoid in meters
    orbit_df : pd.DataFrame
        Orbit state vectors with columns: clock, px, py, pz, vx, vy, vz
    clock_start : float
        Image start time in days (from PRM clock_start)
    prf : float
        Pulse repetition frequency in Hz
    near_range : float
        Near range distance in meters
    rng_samp_rate : float
        Range sampling rate in Hz
    num_valid_az : int
        Number of valid azimuth lines per patch
    num_patches : int
        Number of patches
    nrows : int
        Total number of rows
    earth_radius : float
        Local earth radius from doppler_centroid()
    ra : float
        Semi-major axis of reference ellipsoid (default WGS84)
    rc : float
        Semi-minor axis of reference ellipsoid (default WGS84)
    precise : int
        Precision level (0=standard, 1=polynomial refinement)
    lookdir : str
        Look direction ('R' for right-looking, 'L' for left-looking)
    fd1 : float
        Doppler centroid (Hz). Set to 0 to disable Doppler correction.
    fdd1 : float
        Doppler centroid rate (Hz/m)
    wavelength : float
        Radar wavelength (m). Required if fd1 != 0.
    vel : float
        Ground velocity (m/s). Required if fd1 != 0.
    num_rng_bins : int
        Number of range bins. Required if fd1 != 0.
    rshift : int
        Range shift in pixels (for aligned images)
    ashift : int
        Azimuth shift in pixels (for aligned images)
    sub_int_r : float
        Sub-integer range shift
    sub_int_a : float
        Sub-integer azimuth shift
    chirp_ext : int
        Chirp extension in pixels

    Returns
    -------
    np.ndarray
        Array of shape (N, 5) with columns [range_pix, azimuth_pix, range_m, azimuth_time, elevation]
    """
    from scipy import constants

    SOL = constants.speed_of_light

    lon = np.asarray(lon)
    lat = np.asarray(lat)
    elevation = np.asarray(elevation)

    lon = lon.ravel()
    lat = lat.ravel()
    elevation = elevation.ravel()
    n_points = len(lon)

    # Prepare orbit data
    orbit_time = orbit_df['clock'].values
    px = orbit_df['px'].values
    py = orbit_df['py'].values
    pz = orbit_df['pz'].values
    vx = orbit_df['vx'].values
    vy = orbit_df['vy'].values
    vz = orbit_df['vz'].values

    # Compute acceleration for Hermite interpolation
    dt_orb = orbit_time[1] - orbit_time[0]
    ax = np.gradient(vx, dt_orb)
    ay = np.gradient(vy, dt_orb)
    az = np.gradient(vz, dt_orb)

    # Time range for azimuth lines
    t1 = 86400.0 * clock_start + (nrows - num_valid_az) / (2.0 * prf)

    # Pre-compute orbit at azimuth line times (coarse grid, ~nrows points)
    # Add padding for targets outside image bounds
    npad = 100  # padding in azimuth lines
    azi_times = t1 + np.arange(-npad, nrows + npad) / prf

    # Interpolate orbit at azimuth times
    orb_x = _hermite_interp(orbit_time, px, vx, azi_times, nval=6)
    orb_y = _hermite_interp(orbit_time, py, vy, azi_times, nval=6)
    orb_z = _hermite_interp(orbit_time, pz, vz, azi_times, nval=6)
    orb_vx = _hermite_interp(orbit_time, vx, ax, azi_times, nval=6)
    orb_vy = _hermite_interp(orbit_time, vy, ay, azi_times, nval=6)
    orb_vz = _hermite_interp(orbit_time, vz, az, azi_times, nval=6)

    # Convert geodetic to ECEF
    e2 = (ra**2 - rc**2) / ra**2
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad)
    cos_lon = np.cos(lon_rad)
    N = ra / np.sqrt(1 - e2 * sin_lat**2)
    xp = (N + elevation) * cos_lat * cos_lon
    yp = (N + elevation) * cos_lat * sin_lon
    zp = (N * (1 - e2) + elevation) * sin_lat

    # For each target, find zero-Doppler azimuth time using Doppler search
    # Doppler = (T - S) · V, we want Doppler = 0
    # Doppler changes monotonically along orbit, so we can use searchsorted

    # Compute Doppler at coarse azimuth times for all targets
    # This is O(n_points × n_azi) but with vectorized operations
    n_azi = len(azi_times)

    # Process in chunks to manage memory
    chunk_size = 10000
    azimuth_pix = np.zeros(n_points, dtype=np.float64)
    range_m = np.zeros(n_points, dtype=np.float64)

    for i in range(0, n_points, chunk_size):
        end = min(i + chunk_size, n_points)
        chunk_xp = xp[i:end]
        chunk_yp = yp[i:end]
        chunk_zp = zp[i:end]
        n_chunk = end - i

        # Compute Doppler at a few sample points to bracket zero
        # Use coarse sampling first, then refine
        sample_step = max(1, n_azi // 20)  # ~20 samples
        sample_idx = np.arange(0, n_azi, sample_step)

        # Compute Doppler at sample points: (T - S) · V
        doppler_samples = np.zeros((n_chunk, len(sample_idx)), dtype=np.float32)
        for j, idx in enumerate(sample_idx):
            dx = chunk_xp - orb_x[idx]
            dy = chunk_yp - orb_y[idx]
            dz = chunk_zp - orb_z[idx]
            doppler_samples[:, j] = dx * orb_vx[idx] + dy * orb_vy[idx] + dz * orb_vz[idx]

        # Find sign change (zero crossing) for each target
        # Doppler typically goes from positive to negative (descending) or vice versa
        sign_change = doppler_samples[:, :-1] * doppler_samples[:, 1:] < 0
        first_crossing = np.argmax(sign_change, axis=1)
        # Handle case where no crossing found (use minimum absolute Doppler)
        no_crossing = ~np.any(sign_change, axis=1)
        first_crossing[no_crossing] = np.argmin(np.abs(doppler_samples[no_crossing]), axis=1)

        # Get bracket indices in original azi_times
        bracket_lo = sample_idx[first_crossing]
        bracket_hi = np.minimum(sample_idx[np.minimum(first_crossing + 1, len(sample_idx) - 1)], n_azi - 1)

        # Refine within bracket using linear interpolation on Doppler
        # Compute exact Doppler at bracket bounds - VECTORIZED (no Python loop!)
        dx_lo = chunk_xp - orb_x[bracket_lo]
        dy_lo = chunk_yp - orb_y[bracket_lo]
        dz_lo = chunk_zp - orb_z[bracket_lo]
        doppler_lo = dx_lo * orb_vx[bracket_lo] + dy_lo * orb_vy[bracket_lo] + dz_lo * orb_vz[bracket_lo]

        dx_hi = chunk_xp - orb_x[bracket_hi]
        dy_hi = chunk_yp - orb_y[bracket_hi]
        dz_hi = chunk_zp - orb_z[bracket_hi]
        doppler_hi = dx_hi * orb_vx[bracket_hi] + dy_hi * orb_vy[bracket_hi] + dz_hi * orb_vz[bracket_hi]
        del dx_lo, dy_lo, dz_lo, dx_hi, dy_hi, dz_hi

        # Linear interpolation to find zero crossing
        denom = doppler_lo - doppler_hi
        denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        alpha = doppler_lo / denom
        alpha = np.clip(alpha, 0, 1)

        # Interpolated azimuth index (relative to azi_times array)
        azi_idx_float = bracket_lo + alpha * (bracket_hi - bracket_lo)

        # Convert to azimuth pixel (relative to image start)
        azimuth_pix[i:end] = azi_idx_float - npad  # subtract padding offset

        # Compute slant range at zero-Doppler time
        # Interpolate satellite position
        azi_idx_int = np.floor(azi_idx_float).astype(np.int32)
        azi_idx_int = np.clip(azi_idx_int, 0, n_azi - 2)
        azi_frac = azi_idx_float - azi_idx_int

        sat_x = orb_x[azi_idx_int] * (1 - azi_frac) + orb_x[azi_idx_int + 1] * azi_frac
        sat_y = orb_y[azi_idx_int] * (1 - azi_frac) + orb_y[azi_idx_int + 1] * azi_frac
        sat_z = orb_z[azi_idx_int] * (1 - azi_frac) + orb_z[azi_idx_int + 1] * azi_frac

        range_m[i:end] = np.sqrt((chunk_xp - sat_x)**2 + (chunk_yp - sat_y)**2 + (chunk_zp - sat_z)**2)

    # Compute azimuth time
    azimuth_time = t1 + azimuth_pix / prf

    # Convert range to pixels
    range_pixel_size = SOL / (2.0 * rng_samp_rate)
    range_pix = (range_m - near_range) / range_pixel_size

    # Apply sub-pixel shift corrections
    range_pix = range_pix - (rshift + sub_int_r) + chirp_ext
    azimuth_pix = azimuth_pix - (ashift + sub_int_a)

    # Doppler centroid correction
    if fd1 != 0.0 and wavelength is not None and vel is not None and num_rng_bins is not None:
        dr = range_pixel_size
        dopc = fd1 + fdd1 * (near_range + dr * num_rng_bins / 2.0)
        rng_abs = np.abs(range_m)
        rdd = (vel * vel) / rng_abs
        daa = -0.5 * (wavelength * dopc) / rdd
        drr = 0.5 * rdd * daa * daa / dr
        daa_pix = prf * daa
        range_pix = range_pix + drr
        azimuth_pix = azimuth_pix + daa_pix

    # Result array: [range_pix, azimuth_pix, range_m, azimuth_time, elevation]
    result = np.zeros((n_points, 5))
    result[:, 0] = range_pix
    result[:, 1] = azimuth_pix
    result[:, 2] = range_m
    result[:, 3] = azimuth_time
    result[:, 4] = elevation

    return result


def save_transform(transform, outdir, scale_factor=2.0):
    """Save transform dataset to zarr with int32 encoding.

    Parameters
    ----------
    transform : xarray.Dataset
        Transform dataset with rng, azi, ele variables.
    outdir : str
        Output directory for the zarr store.
    scale_factor : float, optional
        Scale factor for encoding. Default is 2.0.
    """
    import xarray as xr
    import os

    fill_value = np.iinfo(np.int32).max
    trans_int = xr.Dataset(attrs=transform.attrs)

    # Scale coordinate variables (rng, azi, ele)
    for varname in ['rng', 'azi', 'ele']:
        scaled = (scale_factor * transform[varname]).round()
        finite_mask = np.isfinite(scaled)
        int_data = scaled.fillna(0).astype(np.int32)
        int_data = int_data.where(finite_mask, fill_value)
        trans_int[varname] = int_data
        trans_int[varname].attrs['scale_factor'] = 1/scale_factor
        trans_int[varname].attrs['add_offset'] = 0

    # Scale look vector components (unit vectors, range -1 to 1)
    # Use higher scale factor for precision (1e6 gives ~1e-6 precision)
    look_scale = 1e6
    for varname in ['look_E', 'look_N', 'look_U']:
        if varname in transform:
            scaled = (look_scale * transform[varname]).round()
            finite_mask = np.isfinite(scaled)
            int_data = scaled.fillna(0).astype(np.int32)
            int_data = int_data.where(finite_mask, fill_value)
            trans_int[varname] = int_data
            trans_int[varname].attrs['scale_factor'] = 1/look_scale
            trans_int[varname].attrs['add_offset'] = 0

    # Use 8192 chunk size for memory-efficient reading
    CHUNK_SIZE = 8192
    n_y, n_x = transform.y.size, transform.x.size
    chunk_y = min(CHUNK_SIZE, n_y)
    chunk_x = min(CHUNK_SIZE, n_x)

    all_vars = ['rng', 'azi', 'ele'] + [v for v in ['look_E', 'look_N', 'look_U'] if v in trans_int]
    encoding = {var: {'chunks': (chunk_y, chunk_x), '_FillValue': fill_value} for var in all_vars}
    trans_int.to_zarr(
        store=os.path.join(outdir, 'transform'),
        mode='w',
        zarr_format=3,
        consolidated=True,
        encoding=encoding
    )


def save_topo(topo, outdir, scale_factor=2.0):
    """Save topo DataArray to zarr with int32 encoding.

    Parameters
    ----------
    topo : xarray.DataArray
        Topo array in radar coordinates (a, r).
    outdir : str
        Output directory for the zarr store.
    scale_factor : float, optional
        Scale factor for encoding. Default is 2.0.
    """
    import xarray as xr
    import os

    if topo is None:
        return

    fill_value = np.iinfo(np.int32).max

    # Scale elevation values
    scaled = (scale_factor * topo).round()
    finite_mask = np.isfinite(scaled)
    int_data = scaled.fillna(0).astype(np.int32)
    int_data = int_data.where(finite_mask, fill_value)
    int_data.attrs['scale_factor'] = 1/scale_factor
    int_data.attrs['add_offset'] = 0

    # Use 8192 chunk size for memory-efficient reading
    CHUNK_SIZE = 8192
    n_a, n_r = topo.a.size, topo.r.size
    chunk_a = min(CHUNK_SIZE, n_a)
    chunk_r = min(CHUNK_SIZE, n_r)

    ds = xr.Dataset({'topo': int_data})
    encoding = {'topo': {'chunks': (chunk_a, chunk_r), '_FillValue': fill_value}}
    ds.to_zarr(
        store=os.path.join(outdir, 'topo'),
        mode='w',
        zarr_format=3,
        consolidated=True,
        encoding=encoding
    )


def load_topo(outdir):
    """Load topo from zarr with lazy loading.

    Parameters
    ----------
    outdir : str
        Directory containing the topo zarr store.

    Returns
    -------
    xarray.DataArray
        Topo array in radar coordinates (a, r) with lazy loading.
    """
    import xarray as xr
    import os

    topo_path = os.path.join(outdir, 'topo')
    if not os.path.exists(topo_path):
        return None

    ds = xr.open_zarr(store=topo_path, consolidated=True, zarr_format=3, chunks='auto')

    # Decode int32 to float32
    topo = ds['topo']
    fill_value = topo.attrs.get('_FillValue')
    scale_factor = topo.attrs.get('scale_factor', 1.0)
    if fill_value is not None:
        data = topo.astype('float32')
        data = data.where(ds['topo'] != fill_value)
        topo = data * scale_factor
    else:
        topo = topo.astype('float32')

    return topo


def load_transform(outdir):
    """Load transform from zarr with lazy loading.

    Parameters
    ----------
    outdir : str
        Directory containing the transform zarr store.

    Returns
    -------
    xarray.Dataset
        Transform dataset with rng, azi, ele variables (lazy loaded).
    """
    import xarray as xr
    import os

    trans_path = os.path.join(outdir, 'transform')
    ds = xr.open_zarr(store=trans_path, consolidated=True, zarr_format=3, chunks='auto')

    # Decode int32 to float32 for each variable
    for v in ('rng', 'azi', 'ele'):
        if v not in ds:
            continue
        fill_value = ds[v].attrs.get('_FillValue')
        scale_factor = ds[v].attrs.get('scale_factor', 1.0)
        if fill_value is not None:
            data = ds[v].astype('float32')
            data = data.where(ds[v] != fill_value)
            ds[v] = data * scale_factor
        else:
            data = ds[v].astype('float32')
            ds[v] = data.where(np.abs(data) < 1e8)

    return ds


def remap_radar_to_geo(data, azi_map, rng_map, out_y, out_x):
    """Remap data from radar coordinates to geographic coordinates using cv2.remap.

    Takes radar-coordinate data and pre-computed coordinate maps, and resamples
    to a geographic grid using Lanczos interpolation. Handles both real and
    complex data, and works around OpenCV's 32k pixel limit via chunking.

    Parameters
    ----------
    data : xr.DataArray
        Input data in radar coordinates with dims ['a', 'r'] (azimuth, range).
    azi_map : np.ndarray
        Azimuth coordinates for each output pixel (float32, 2D).
        Maps each (y, x) output pixel to its source azimuth in radar coords.
    rng_map : np.ndarray
        Range coordinates for each output pixel (float32, 2D).
        Maps each (y, x) output pixel to its source range in radar coords.
    out_y : np.ndarray
        Output grid Y coordinates (1D, typically northing or latitude).
    out_x : np.ndarray
        Output grid X coordinates (1D, typically easting or longitude).

    Returns
    -------
    xr.DataArray
        Geocoded data with dims ['y', 'x'] and the same name as input.

    Notes
    -----
    - Uses cv2.INTER_LANCZOS4 for high-quality resampling
    - Pixels outside the radar coverage are filled with NaN
    - For grids wider than 32766 pixels, processes in x-chunks
    """
    import cv2
    import numpy as np
    import xarray as xr

    data_vals = data.values
    coord_a = data.a.values
    coord_r = data.r.values

    # Convert geographic pixel coordinates to radar array indices
    inv_map_a = ((azi_map - coord_a[0]) / (coord_a[1] - coord_a[0])).astype(np.float32)
    inv_map_r = ((rng_map - coord_r[0]) / (coord_r[1] - coord_r[0])).astype(np.float32)

    n_y, n_x = inv_map_a.shape
    OPENCV_MAX = 32766

    if n_x <= OPENCV_MAX:
        # Fast path: single cv2.remap call
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
        # Chunked path: work around OpenCV's 32k pixel limit
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


def compute_merged_transform(transform, prm_rep):
    """Compute merged transform by adding PRM bilinear offsets to ref transform.

    Parameters
    ----------
    transform : xr.Dataset
        Reference scene/burst transform with azi, rng variables.
    prm_rep : PRM
        Repeat scene/burst PRM with fitoffset parameters (rshift, stretch_r, etc.).

    Returns
    -------
    azi_rep, rng_rep : np.ndarray
        Repeat scene/burst radar coordinates for each output pixel (float32, 2D).
    """
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


def tidal_phase_radar(topo, prm, dt):
    """Compute solid Earth tidal phase correction on the radar coordinate grid.

    Uses 2×2 radar-grid corners: computes tidal E,N,U and look vectors at the
    4 corner points, bilinearly interpolates each component separately onto the
    full topo grid, then dot-products to LOS and converts to phase.

    Satellite-agnostic: works for both S1 and NISAR.

    Parameters
    ----------
    topo : xr.DataArray
        Topographic elevation with radar coordinates (a, r).
    prm : PRM
        Reference scene PRM (has orbit_df, clock_start, PRF, etc.).
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
    from .utils_tidal import solid_tide

    n_azi = len(topo.a)
    n_rng = len(topo.r)

    # --- (a) 4 radar corner coordinates ---
    azi_corners = np.array([topo.a.values[0], topo.a.values[0],
                            topo.a.values[-1], topo.a.values[-1]])
    rng_corners = np.array([topo.r.values[0], topo.r.values[-1],
                            topo.r.values[0], topo.r.values[-1]])

    # --- (b) Geocode 4 corners -> lat/lon ---
    orbit_df = prm.orbit_df
    orbit_time = orbit_df['isec'].values
    orbit_pos = orbit_df[['px', 'py', 'pz']].values
    orbit_vel = orbit_df[['vx', 'vy', 'vz']].values

    clock_start = (prm.get('clock_start') % 1.0) * 86400
    prf = prm.get('PRF')
    near_range = prm.get('near_range')
    rng_samp_rate = prm.get('rng_samp_rate')
    earth_radius = prm.get('earth_radius')
    lookdir = prm.get('lookdir') if 'lookdir' in prm.df.index else 'R'

    lon_corners, lat_corners, _ = satellite_rat2llt(
        azi_corners, rng_corners,
        orbit_time, orbit_pos, orbit_vel,
        clock_start, prf, near_range, rng_samp_rate, earth_radius,
        lookdir=lookdir
    )

    # --- (c) Compute tidal E, N, U at 4 corners ---
    tide_e, tide_n, tide_u = solid_tide(lon_corners, lat_corners, dt)

    # --- (d) Compute look vectors at 4 corners from orbit ---
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

    # Look vector (ground -> satellite), normalized
    lx = sat_x - gx
    ly = sat_y - gy
    lz = sat_z - gz
    dist = np.sqrt(lx**2 + ly**2 + lz**2)
    lx /= dist;  ly /= dist;  lz /= dist

    # ECEF look -> ENU
    b = lat_rad - np.pi / 2
    g = lon_rad + np.pi / 2
    cos_b = np.cos(b);  sin_b = np.sin(b)
    cos_g = np.cos(g);  sin_g = np.sin(g)
    look_E = cos_g * lx + sin_g * ly
    look_N = -sin_g * cos_b * lx + cos_g * cos_b * ly - sin_b * lz
    look_U = -sin_g * sin_b * lx + cos_g * sin_b * ly + cos_b * lz

    # --- (e) Bilinear interpolation + LOS ---
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
    wavelength = prm.get('radar_wavelength')
    cnst = -4.0 * np.pi / wavelength
    tidal_phase = (cnst * los).astype(np.float32)

    return xr.DataArray(tidal_phase, coords=topo.coords, dims=topo.dims).rename('tidal_phase')


def flat_earth_topo_phase(topo, prm_rep, prm_ref, baseline_params=None, sc_height_params=None):
    """Compute the combined earth curvature and topographic phase correction.

    Uses the full GMTSAR algorithm with time-varying baseline geometry.
    Satellite-agnostic: works for both S1 and NISAR.

    Parameters
    ----------
    topo : xr.DataArray or None
        Topographic elevation in radar coordinates (meters). If None, uses zero topo.
    prm_rep : PRM
        Repeat scene PRM object.
    prm_ref : PRM
        Reference scene PRM object.
    baseline_params : dict, optional
        Pre-computed baseline parameters.
    sc_height_params : dict, optional
        Pre-computed SC_height parameters.

    Returns
    -------
    xr.DataArray
        Combined flat earth and topo phase (radians).
    """
    import numpy as np
    import xarray as xr
    from scipy import constants
    from .PRM import PRM

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


def compute_transform(prm, dem,
                      scale_factor=2.0,
                      epsg=None,
                      resolution=(16.0, 4.0),
                      n_chunks=8,
                      debug=False):
    """
    Compute geocoding transform using direct radar-to-geo method.

    Creates an INVERSE transform (projected → radar) on a regular geographic grid.
    Uses satellite_rat2llt for fast closed-form radar-to-geographic conversion,
    then inverts via bilinear splatting to create coords (y, x) with vars
    (rng, azi, ele).

    Parameters
    ----------
    prm : PRM
        The reference burst PRM object with orbit_df attached.
    dem : xarray.DataArray
        Pre-loaded ellipsoid-corrected DEM (WGS84 ellipsoidal heights).
    scale_factor : float, optional
        Scale factor for integer compression. Default is 2.0.
    epsg : int, optional
        Target EPSG code. If None, auto-detect UTM zone.
    resolution : tuple[float, float], optional
        Output resolution (dy, dx) in meters. Default is (16.0, 4.0).
    n_chunks : int, optional
        Number of azimuth chunks for memory-efficient processing. Default is 8.
    debug : bool, optional
        If True, prints timing information. Default is False.

    Returns
    -------
    tuple
        (topo, transform) where topo is DataArray and transform is Dataset.
    """
    import cv2
    import xarray as xr
    import time
    import gc
    import warnings
    warnings.filterwarnings('ignore')

    _timings = {} if debug else None

    # Get orbit data from PRM
    orbit_df = prm.orbit_df
    if orbit_df is None:
        raise ValueError("PRM object has no orbit_df attached")

    orbit_time = orbit_df['isec'].values
    orbit_pos = orbit_df[['px', 'py', 'pz']].values
    orbit_vel = orbit_df[['vx', 'vy', 'vz']].values

    # Get PRM parameters
    clock_start = (prm.get('clock_start') % 1.0) * 86400
    prf = prm.get('PRF')
    near_range = prm.get('near_range')
    rng_samp_rate = prm.get('rng_samp_rate')
    earth_radius = prm.get('earth_radius')
    lookdir = prm.get('lookdir') if 'lookdir' in prm.df.index else 'R'
    a_max, r_max = prm.bounds()

    # Create radar grid coordinates
    azi_coords = np.arange(0.5, a_max, 1, dtype=np.float32)
    rng_coords = np.arange(0.5, r_max, 1, dtype=np.float32)
    n_azi = len(azi_coords)
    n_rng = len(rng_coords)

    if debug:
        print(f'Radar grid: {n_azi} x {n_rng} = {n_azi * n_rng:,} points, n_chunks={n_chunks}')

    # WGS84 constants for ECEF conversion
    ra = 6378137.0
    rc = 6356752.31424518
    e2 = np.float32((ra**2 - rc**2) / ra**2)

    # epsg=0: radar coordinates mode - compute topo and identity transform
    if epsg == 0:
        ele_gmtsar_full = np.zeros((n_azi, n_rng), dtype=np.float32)
        chunk_size = (n_azi + n_chunks - 1) // n_chunks
        for chunk_idx in range(n_chunks):
            azi_start = chunk_idx * chunk_size
            azi_end = min((chunk_idx + 1) * chunk_size, n_azi)
            if azi_start >= n_azi:
                break
            azi_chunk = azi_coords[azi_start:azi_end]
            azi_grid, rng_grid = np.meshgrid(azi_chunk, rng_coords, indexing='ij')
            lon, lat, ele_dem = satellite_rat2llt(
                azi_grid, rng_grid,
                orbit_time, orbit_pos, orbit_vel,
                clock_start, prf, near_range, rng_samp_rate, earth_radius,
                dem=dem, max_iter=20, tol=0.1, n_chunks=1, lookdir=lookdir
            )
            # ECEF conversion → ele_gmtsar
            lon_rad = np.radians(lon).astype(np.float32)
            lat_rad = np.radians(lat).astype(np.float32)
            sin_lat = np.sin(lat_rad).astype(np.float32)
            cos_lat = np.cos(lat_rad).astype(np.float32)
            N = (np.float32(ra) / np.sqrt(1 - e2 * sin_lat**2)).astype(np.float32)
            cos_lon = np.cos(lon_rad).astype(np.float32)
            sin_lon = np.sin(lon_rad).astype(np.float32)
            xp = ((N + ele_dem) * cos_lat * cos_lon).astype(np.float32)
            yp = ((N + ele_dem) * cos_lat * sin_lon).astype(np.float32)
            zp = ((N * (1 - e2) + ele_dem) * sin_lat).astype(np.float32)
            ele_gmtsar = (np.sqrt(xp**2 + yp**2 + zp**2) - earth_radius).astype(np.float32)
            ele_gmtsar_full[azi_start:azi_end, :] = ele_gmtsar

        # Create identity transform in radar coordinates (y=azi, x=rng)
        azi_2d, rng_2d = np.meshgrid(azi_coords, rng_coords, indexing='ij')
        trans = xr.Dataset({
            'rng': xr.DataArray(rng_2d.astype(np.float32), coords={'y': azi_coords, 'x': rng_coords}, dims=['y', 'x']),
            'azi': xr.DataArray(azi_2d.astype(np.float32), coords={'y': azi_coords, 'x': rng_coords}, dims=['y', 'x']),
            'ele': xr.DataArray(ele_gmtsar_full, coords={'y': azi_coords, 'x': rng_coords}, dims=['y', 'x']),
        })

        radar_crs_wkt = '''ENGCRS["Radar Coordinates",EDATUM["Radar datum"],CS[Cartesian,2],AXIS["azimuth",south,ORDER[1],LENGTHUNIT["pixel",1]],AXIS["range",east,ORDER[2],LENGTHUNIT["pixel",1]]]'''
        trans.attrs['spatial_ref'] = radar_crs_wkt

        topo = xr.DataArray(ele_gmtsar_full,
                           coords={'a': azi_coords, 'r': rng_coords},
                           dims=['a', 'r']).rename('topo')
        return topo, trans

    # Auto-detect EPSG if not specified
    if epsg is None:
        epsg = get_utm_epsg(float(dem.lat.mean()), float(dem.lon.mean()))

    dy, dx = resolution

    # First pass: determine output grid bounds using coarse sampling
    t0 = time.perf_counter()
    coarse_step = max(1, n_azi // 20)
    azi_coarse = azi_coords[::coarse_step]
    azi_grid_c, rng_grid_c = np.meshgrid(azi_coarse, rng_coords, indexing='ij')
    lon_c, lat_c, _ = satellite_rat2llt(
        azi_grid_c, rng_grid_c,
        orbit_time, orbit_pos, orbit_vel,
        clock_start, prf, near_range, rng_samp_rate, earth_radius,
        dem=dem, max_iter=2, tol=1.0, n_chunks=1, lookdir=lookdir
    )
    y_c, x_c = proj(lat_c, lon_c, from_epsg=4326, to_epsg=epsg)
    del azi_grid_c, rng_grid_c, lon_c, lat_c

    # Output grid bounds with margin
    margin = 100  # pixels
    y_min = dy * (np.floor(np.nanmin(y_c) / dy) - margin)
    y_max = dy * (np.ceil(np.nanmax(y_c) / dy) + margin)
    x_min = dx * (np.floor(np.nanmin(x_c) / dx) - margin)
    x_max = dx * (np.ceil(np.nanmax(x_c) / dx) + margin)
    del y_c, x_c

    out_y = np.arange(y_min + dy/2, y_max, dy)
    out_x = np.arange(x_min + dx/2, x_max, dx)
    n_y, n_x = len(out_y), len(out_x)
    out_size = n_y * n_x

    if debug:
        print(f'Output grid: {n_y} x {n_x} = {out_size:,} points')

    # Initialize output accumulators
    inv_azi = np.zeros(out_size, dtype=np.float32)
    inv_rng = np.zeros(out_size, dtype=np.float32)
    inv_ele = np.zeros(out_size, dtype=np.float32)
    weight_sum = np.zeros(out_size, dtype=np.float32)
    ele_gmtsar_full = np.zeros((n_azi, n_rng), dtype=np.float32)

    if _timings is not None:
        _timings['bounds_scan'] = time.perf_counter() - t0

    # Process in chunks along azimuth
    t0 = time.perf_counter()
    chunk_size = (n_azi + n_chunks - 1) // n_chunks

    for chunk_idx in range(n_chunks):
        azi_start = chunk_idx * chunk_size
        azi_end = min((chunk_idx + 1) * chunk_size, n_azi)
        if azi_start >= n_azi:
            break

        azi_chunk = azi_coords[azi_start:azi_end]
        azi_grid, rng_grid = np.meshgrid(azi_chunk, rng_coords, indexing='ij')

        lon, lat, ele_dem = satellite_rat2llt(
            azi_grid, rng_grid,
            orbit_time, orbit_pos, orbit_vel,
            clock_start, prf, near_range, rng_samp_rate, earth_radius,
            dem=dem, max_iter=20, tol=0.1, n_chunks=1, lookdir=lookdir
        )

        # ECEF conversion → ele_gmtsar
        lon_rad = np.radians(lon).astype(np.float32)
        lat_rad = np.radians(lat).astype(np.float32)
        sin_lat = np.sin(lat_rad).astype(np.float32)
        cos_lat = np.cos(lat_rad).astype(np.float32)
        del lat_rad
        N = (np.float32(ra) / np.sqrt(1 - e2 * sin_lat**2)).astype(np.float32)
        cos_lon = np.cos(lon_rad).astype(np.float32)
        sin_lon = np.sin(lon_rad).astype(np.float32)
        del lon_rad

        xp = ((N + ele_dem) * cos_lat * cos_lon).astype(np.float32)
        yp = ((N + ele_dem) * cos_lat * sin_lon).astype(np.float32)
        del cos_lon, sin_lon
        zp = ((N * (1 - e2) + ele_dem) * sin_lat).astype(np.float32)
        del N, cos_lat, sin_lat, ele_dem

        ele_gmtsar = (np.sqrt(xp**2 + yp**2 + zp**2) - earth_radius).astype(np.float32)
        ele_gmtsar_full[azi_start:azi_end, :] = ele_gmtsar
        del xp, yp, zp

        # Project to EPSG
        y_proj, x_proj = proj(lat, lon, from_epsg=4326, to_epsg=epsg)
        y_proj = y_proj.astype(np.float32)
        x_proj = x_proj.astype(np.float32)
        del lat, lon

        # Bilinear splatting
        # Skip pixels where elevation is NaN (outside DEM coverage or boundary issues)
        valid_mask = np.isfinite(y_proj) & np.isfinite(x_proj) & np.isfinite(ele_gmtsar)
        yi_norm = ((y_proj - y_min) / dy - 0.5).astype(np.float32)
        xi_norm = ((x_proj - x_min) / dx - 0.5).astype(np.float32)
        del y_proj, x_proj

        yi_floor = np.floor(yi_norm).astype(np.int64)
        xi_floor = np.floor(xi_norm).astype(np.int64)
        yi_frac = (yi_norm - yi_floor).astype(np.float32)
        xi_frac = (xi_norm - xi_floor).astype(np.float32)
        del yi_norm, xi_norm

        azi_flat = azi_grid.ravel().astype(np.float32)
        rng_flat = rng_grid.ravel().astype(np.float32)
        ele_flat = ele_gmtsar.ravel()
        del azi_grid, rng_grid, ele_gmtsar
        yi_floor_flat = yi_floor.ravel()
        xi_floor_flat = xi_floor.ravel()
        yi_frac_flat = yi_frac.ravel()
        xi_frac_flat = xi_frac.ravel()
        del yi_floor, xi_floor, yi_frac, xi_frac
        valid_flat = valid_mask.ravel()
        del valid_mask

        for di in [0, 1]:
            for dj in [0, 1]:
                yi = yi_floor_flat + di
                xi = xi_floor_flat + dj
                wy = (1.0 - yi_frac_flat) if di == 0 else yi_frac_flat
                wx = (1.0 - xi_frac_flat) if dj == 0 else xi_frac_flat
                ww = wy * wx
                m = valid_flat & (yi >= 0) & (yi < n_y) & (xi >= 0) & (xi < n_x)
                idx = (yi[m] * n_x + xi[m]).astype(np.int64)
                ww_m = ww[m]
                inv_azi += np.bincount(idx, weights=azi_flat[m] * ww_m, minlength=out_size).astype(np.float32)
                inv_rng += np.bincount(idx, weights=rng_flat[m] * ww_m, minlength=out_size).astype(np.float32)
                inv_ele += np.bincount(idx, weights=ele_flat[m] * ww_m, minlength=out_size).astype(np.float32)
                weight_sum += np.bincount(idx, weights=ww_m, minlength=out_size).astype(np.float32)

        del azi_flat, rng_flat, ele_flat
        del yi_floor_flat, xi_floor_flat, yi_frac_flat, xi_frac_flat, valid_flat
        gc.collect()

    if _timings is not None:
        _timings['chunked_processing'] = time.perf_counter() - t0

    # Reshape to 2D
    inv_azi = inv_azi.reshape(n_y, n_x)
    inv_rng = inv_rng.reshape(n_y, n_x)
    inv_ele = inv_ele.reshape(n_y, n_x)
    weight_sum = weight_sum.reshape(n_y, n_x)

    # Normalize by weights
    valid_weight = weight_sum > 1e-6
    for arr in [inv_azi, inv_rng, inv_ele]:
        arr[valid_weight] /= weight_sum[valid_weight]
        arr[~valid_weight] = np.nan
    del weight_sum

    dem_coverage = np.isfinite(inv_ele)

    # Restore NaN for areas outside DEM coverage
    inv_azi[~dem_coverage] = np.nan
    inv_rng[~dem_coverage] = np.nan
    inv_ele[~dem_coverage] = np.nan

    # Build dataset
    trans = xr.Dataset({
        'rng': xr.DataArray(inv_rng, coords={'y': out_y, 'x': out_x}, dims=['y', 'x']),
        'azi': xr.DataArray(inv_azi, coords={'y': out_y, 'x': out_x}, dims=['y', 'x']),
        'ele': xr.DataArray(inv_ele, coords={'y': out_y, 'x': out_x}, dims=['y', 'x']),
    })

    # Add georeference
    from insardev_toolkit.datagrid import datagrid
    trans = datagrid.spatial_ref(trans, epsg)
    trans.attrs['spatial_ref'] = trans.spatial_ref.attrs['spatial_ref']
    trans = trans.drop_vars('spatial_ref')

    if _timings is not None:
        print(f'PROFILE compute_transform breakdown:')
        for k, v in sorted(_timings.items(), key=lambda x: -x[1]):
            print(f'  {k}: {v:.3f}s')

    topo = xr.DataArray(ele_gmtsar_full,
                       coords={'a': azi_coords, 'r': rng_coords},
                       dims=['a', 'r']).rename('topo')
    return topo, trans


def compute_transform_inverse(prm, dem,
                              scale_factor=2.0,
                              epsg=None,
                              resolution=(16.0, 4.0),
                              n_chunks=8,
                              compute_topo=True,
                              debug=False):
    """
    Compute geocoding transform using optimized inverse method.

    Uses boundary-based valid region detection and inverse transform (llt2rat)
    for faster computation when output grid is comparable to or larger than radar grid.

    Parameters
    ----------
    prm : PRM
        The reference burst PRM object with orbit_df attached.
    dem : xarray.DataArray
        Pre-loaded ellipsoid-corrected DEM (WGS84 ellipsoidal heights).
    scale_factor : float, optional
        Scale factor for integer compression. Default is 2.0.
    epsg : int, optional
        Target EPSG code. If None, auto-detect UTM zone.
    resolution : tuple[float, float], optional
        Output resolution (dy, dx) in meters. Default is (16.0, 4.0).
    n_chunks : int, optional
        Number of azimuth chunks for memory-efficient processing. Default is 8.
    compute_topo : bool, optional
        If True, compute topo phase array (needed for InSAR). If False, skip
        for faster geocoding-only workflows. Default is True.
    debug : bool, optional
        If True, prints timing information. Default is False.

    Returns
    -------
    tuple
        (topo, transform) where topo is DataArray (or None if compute_topo=False)
        and transform is Dataset.
    """
    import cv2
    import xarray as xr
    import time
    import gc
    import pandas as pd
    import warnings
    warnings.filterwarnings('ignore')

    _timings = {} if debug else None

    # Get orbit data from PRM
    orbit_df = prm.orbit_df
    if orbit_df is None:
        raise ValueError("PRM object has no orbit_df attached")

    # Ensure clock column matches isec (satellite_llt2rat uses 'clock' internally)
    orbit_df = orbit_df.copy()
    orbit_df['clock'] = orbit_df['isec']

    orbit_time = orbit_df['isec'].values
    orbit_pos = orbit_df[['px', 'py', 'pz']].values
    orbit_vel = orbit_df[['vx', 'vy', 'vz']].values

    # Get PRM parameters
    clock_start = (prm.get('clock_start') % 1.0) * 86400
    clock_start_days = prm.get('clock_start') % 1.0
    prf = prm.get('PRF')
    near_range = prm.get('near_range')
    rng_samp_rate = prm.get('rng_samp_rate')
    earth_radius = prm.get('earth_radius')
    lookdir = prm.get('lookdir') if 'lookdir' in prm.df.index else 'R'
    a_max, r_max = prm.bounds()
    num_lines = int(a_max)
    num_rng = int(r_max)

    # Create radar grid coordinates
    azi_coords = np.arange(0.5, a_max, 1, dtype=np.float32)
    rng_coords = np.arange(0.5, r_max, 1, dtype=np.float32)
    n_azi = len(azi_coords)
    n_rng = len(rng_coords)

    if debug:
        print(f'Radar grid: {n_azi} x {n_rng} = {n_azi * n_rng:,} points')

    # WGS84 constants
    ra = 6378137.0
    rc = 6356752.31424518
    e2 = np.float32((ra**2 - rc**2) / ra**2)

    # Auto-detect EPSG if not specified
    if epsg is None:
        epsg = get_utm_epsg(float(dem.lat.mean()), float(dem.lon.mean()))
    elif epsg == 0:
        # Radar coordinates mode - fall back to original method
        return compute_transform(prm, dem, scale_factor, epsg, resolution, n_chunks, debug)

    dy, dx = resolution

    # Step 1: Compute bounds from first/last rows only (fast)
    t0 = time.perf_counter()
    for row_azi in [azi_coords[0], azi_coords[-1]]:
        azi_row = np.full(n_rng, row_azi, dtype=np.float32)
        lon, lat, _ = satellite_rat2llt(
            azi_row, rng_coords,
            orbit_time, orbit_pos, orbit_vel,
            clock_start, prf, near_range, rng_samp_rate, earth_radius,
            dem=dem, max_iter=10, tol=0.5, n_chunks=1, lookdir=lookdir
        )
        y_proj, x_proj = proj(lat, lon, from_epsg=4326, to_epsg=epsg)
        if row_azi == azi_coords[0]:
            y_first, x_first = y_proj, x_proj
        else:
            y_last, x_last = y_proj, x_proj

    if _timings is not None:
        _timings['bounds'] = time.perf_counter() - t0
        print(f'  Bounds from first/last rows: {_timings["bounds"]:.2f}s')

    # Step 2: Determine output grid bounds
    t0 = time.perf_counter()
    margin = 100
    y_min = dy * (np.floor(np.nanmin([np.nanmin(y_first), np.nanmin(y_last)]) / dy) - margin)
    y_max = dy * (np.ceil(np.nanmax([np.nanmax(y_first), np.nanmax(y_last)]) / dy) + margin)
    x_min = dx * (np.floor(np.nanmin([np.nanmin(x_first), np.nanmin(x_last)]) / dx) - margin)
    x_max = dx * (np.ceil(np.nanmax([np.nanmax(x_first), np.nanmax(x_last)]) / dx) + margin)
    del y_first, x_first, y_last, x_last

    out_y = np.arange(y_min + dy/2, y_max, dy).astype(np.float32)
    out_x = np.arange(x_min + dx/2, x_max, dx).astype(np.float32)

    # Note: OpenCV cv2.remap has 32767 pixel limit per dimension.
    # Instead of cropping here, _geocode_standalone handles this via x-chunking.

    n_y, n_x = len(out_y), len(out_x)

    if debug:
        print(f'Output grid: {n_y} x {n_x} = {n_y * n_x:,} points')

    # Step 3: Forward transform sparse boundary pixels for valid mask
    # Use multiple rows/cols near edges for robust convex hull
    # azi indices: 0, 3, n_azi-4, n_azi-1 (near top and bottom)
    # rng indices: 0, 3, n_rng-4, n_rng-1 (near left and right)
    azi_edge_idx = np.unique(np.clip([0, 3, n_azi-4, n_azi-1], 0, n_azi-1))
    rng_edge_idx = np.unique(np.clip([0, 3, n_rng-4, n_rng-1], 0, n_rng-1))

    # Build boundary points: specific azi rows (all rng) + specific rng cols (all azi)
    bnd_azi_list, bnd_rng_list = [], []

    # Rows near top/bottom edges (specific azi, all rng)
    for ai in azi_edge_idx:
        bnd_azi_list.append(np.full(n_rng, azi_coords[ai], dtype=np.float32))
        bnd_rng_list.append(rng_coords.copy())

    # Columns near left/right edges (all azi, specific rng)
    for ri in rng_edge_idx:
        bnd_azi_list.append(azi_coords.copy())
        bnd_rng_list.append(np.full(n_azi, rng_coords[ri], dtype=np.float32))

    # Forward transform boundary points in chunks (cv2.remap limit ~32k)
    y_bnd_all, x_bnd_all = [], []
    for bnd_azi, bnd_rng in zip(bnd_azi_list, bnd_rng_list):
        lon_b, lat_b, _ = satellite_rat2llt(
            bnd_azi, bnd_rng,
            orbit_time, orbit_pos, orbit_vel,
            clock_start, prf, near_range, rng_samp_rate, earth_radius,
            dem=dem, max_iter=10, tol=0.5, n_chunks=1, lookdir=lookdir
        )
        y_b, x_b = proj(lat_b, lon_b, from_epsg=4326, to_epsg=epsg)
        y_bnd_all.append(y_b)
        x_bnd_all.append(x_b)
    del bnd_azi_list, bnd_rng_list
    y_bnd_all = np.concatenate(y_bnd_all)
    x_bnd_all = np.concatenate(x_bnd_all)

    if _timings is not None:
        _timings['boundary_forward'] = time.perf_counter() - t0
        print(f'  Boundary forward: {_timings["boundary_forward"]:.2f}s ({len(y_bnd_all):,} pts)')

    # Step 4: Build valid mask using boundary polygon
    t0 = time.perf_counter()

    # Filter valid points
    valid = np.isfinite(y_bnd_all) & np.isfinite(x_bnd_all)
    if not valid.any():
        raise ValueError('compute_transform_inverse: no valid boundary points (DEM coverage issue)')
    bnd_y = y_bnd_all[valid]
    bnd_x = x_bnd_all[valid]
    del y_bnd_all, x_bnd_all

    # Build boundary polygon using shapely convex hull
    import shapely
    points = shapely.MultiPoint(np.column_stack([bnd_x, bnd_y]))
    polygon = points.convex_hull

    # Add margin buffer
    margin = 10 * max(dx, dy)
    polygon = polygon.buffer(margin)

    # Rasterize polygon to valid mask
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    # Create affine transform for output grid
    transform = from_bounds(out_x[0] - dx/2, out_y[-1] - dy/2,
                            out_x[-1] + dx/2, out_y[0] + dy/2, n_x, n_y)

    # Rasterize polygon
    valid_mask = rasterize(
        [(polygon, 1)],
        out_shape=(n_y, n_x),
        transform=transform,
        fill=0,
        dtype=np.uint8
    ).astype(bool)

    n_valid = valid_mask.sum()

    if _timings is not None:
        _timings['valid_mask'] = time.perf_counter() - t0
        print(f'  Valid mask: {_timings["valid_mask"]:.2f}s ({100*n_valid/(n_y*n_x):.1f}% valid)')

    # Step 5: Convert valid output pixels to lon/lat and get DEM elevation
    t0 = time.perf_counter()
    x_grid, y_grid = np.meshgrid(out_x, out_y)
    valid_y = y_grid[valid_mask]
    valid_x = x_grid[valid_mask]
    del x_grid, y_grid

    # Project UTM to lon/lat
    valid_lat, valid_lon = proj(valid_y, valid_x, from_epsg=epsg, to_epsg=4326)
    valid_lat = valid_lat.astype(np.float32)
    valid_lon = valid_lon.astype(np.float32)

    # Get DEM elevation at valid points
    valid_ele = dem.interp(
        lat=xr.DataArray(valid_lat, dims='z'),
        lon=xr.DataArray(valid_lon, dims='z'),
        method='linear'
    ).values.astype(np.float32)

    if _timings is not None:
        _timings['dem_interp'] = time.perf_counter() - t0
        print(f'  DEM interp: {_timings["dem_interp"]:.2f}s')

    # Step 6: Inverse transform on valid pixels only
    t0 = time.perf_counter()
    result = satellite_llt2rat(
        lon=valid_lon, lat=valid_lat, elevation=valid_ele,
        orbit_df=orbit_df, clock_start=clock_start_days, prf=prf,
        near_range=near_range, rng_samp_rate=rng_samp_rate,
        num_valid_az=num_lines, num_patches=1, nrows=num_lines,
        earth_radius=earth_radius, precise=1, fd1=0.0,
        debug=debug
    )
    inv_rng_valid = result[:, 0].astype(np.float32)
    inv_azi_valid = result[:, 1].astype(np.float32)
    del result

    # satellite_llt2rat returns values in same coordinate system as forward transform
    # (0.5-based pixel centers), so no offset correction needed

    # Filter out-of-bounds (valid pixel centers are [0.5, n-0.5])
    out_of_bounds = (inv_azi_valid < 0.5) | (inv_azi_valid > n_azi - 0.5) | (inv_rng_valid < 0.5) | (inv_rng_valid > n_rng - 0.5)
    inv_azi_valid[out_of_bounds] = np.nan
    inv_rng_valid[out_of_bounds] = np.nan

    if _timings is not None:
        _timings['inverse_transform'] = time.perf_counter() - t0
        print(f'  Inverse transform ({n_valid:,} pixels): {_timings["inverse_transform"]:.2f}s')

    # Step 7: Compute ele_gmtsar and look angles for valid output pixels
    # Process in chunks to limit working memory to ~250 MB (20× more chunks than n_chunks)
    t0 = time.perf_counter()
    n_valid_pts = len(valid_lon)
    chunk_size = max(1, n_valid_pts // (20 * n_chunks))  # ~160 chunks for default n_chunks=8

    # Pre-allocate output arrays
    inv_ele_valid = np.empty(n_valid_pts, dtype=np.float32)
    # NOTE: look_E/N/U calculation commented out - incidence is computed from azi/rng in Batch.incidence()
    # Keeping this GMTSAR-compatible code for reference
    # look_E_valid = np.empty(n_valid_pts, dtype=np.float32)
    # look_N_valid = np.empty(n_valid_pts, dtype=np.float32)
    # look_U_valid = np.empty(n_valid_pts, dtype=np.float32)

    for i in range(0, n_valid_pts, chunk_size):
        j = min(i + chunk_size, n_valid_pts)

        # Chunk inputs
        lon_c = valid_lon[i:j]
        lat_c = valid_lat[i:j]
        ele_c = valid_ele[i:j]
        azi_c = inv_azi_valid[i:j]

        # Trig functions
        lon_rad = np.float32(np.pi / 180) * lon_c
        lat_rad = np.float32(np.pi / 180) * lat_c
        sin_lat = np.sin(lat_rad, dtype=np.float32)
        cos_lat = np.cos(lat_rad, dtype=np.float32)
        sin_lon = np.sin(lon_rad, dtype=np.float32)
        cos_lon = np.cos(lon_rad, dtype=np.float32)

        # Ground point ECEF
        N = np.float32(ra) / np.sqrt(1 - np.float32(e2) * sin_lat**2)
        Nh = N + ele_c
        xp = Nh * cos_lat * cos_lon
        yp = Nh * cos_lat * sin_lon
        zp = (N * np.float32(1 - e2) + ele_c) * sin_lat

        # ele_gmtsar = distance from earth center - earth_radius
        inv_ele_valid[i:j] = np.sqrt(xp**2 + yp**2 + zp**2, dtype=np.float32) - np.float32(earth_radius)

        # NOTE: Look vector calculation commented out - incidence is computed from azi/rng in Batch.incidence()
        # Keeping this GMTSAR-compatible code for reference
        # # Satellite ECEF at azimuth time
        # sat_time = np.float32(clock_start) + azi_c / np.float32(prf)
        # sat_x = _hermite_interp(orbit_time, orbit_pos[:, 0], orbit_vel[:, 0], sat_time).astype(np.float32)
        # sat_y = _hermite_interp(orbit_time, orbit_pos[:, 1], orbit_vel[:, 1], sat_time).astype(np.float32)
        # sat_z = _hermite_interp(orbit_time, orbit_pos[:, 2], orbit_vel[:, 2], sat_time).astype(np.float32)
        #
        # # Look vector (ground to satellite), normalize to unit
        # sat_x -= xp; sat_y -= yp; sat_z -= zp
        # dist = np.sqrt(sat_x**2 + sat_y**2 + sat_z**2, dtype=np.float32)
        # sat_x /= dist; sat_y /= dist; sat_z /= dist
        #
        # # Transform ECEF look vector to local ENU
        # lat_rad -= np.float32(np.pi / 2)  # b = lat - 90°
        # lon_rad += np.float32(np.pi / 2)  # g = lon + 90°
        # cos_b = np.cos(lat_rad, dtype=np.float32)
        # sin_b = np.sin(lat_rad, dtype=np.float32)
        # cos_g = np.cos(lon_rad, dtype=np.float32)
        # sin_g = np.sin(lon_rad, dtype=np.float32)
        #
        # look_E_valid[i:j] = cos_g * sat_x + sin_g * sat_y
        # look_N_valid[i:j] = -sin_g * cos_b * sat_x + cos_g * cos_b * sat_y - sin_b * sat_z
        # look_U_valid[i:j] = -sin_g * sin_b * sat_x + cos_g * sin_b * sat_y + cos_b * sat_z

    del valid_lon, valid_lat, valid_ele

    if _timings is not None:
        _timings['ele_computation'] = time.perf_counter() - t0
        print(f'  Elevation computation: {_timings["ele_computation"]:.2f}s')

    # Step 8: Scatter valid pixels to full output grid
    t0 = time.perf_counter()
    inv_azi = np.full((n_y, n_x), np.nan, dtype=np.float32)
    inv_rng = np.full((n_y, n_x), np.nan, dtype=np.float32)
    inv_ele = np.full((n_y, n_x), np.nan, dtype=np.float32)
    # NOTE: look_E/N/U arrays commented out - incidence is computed from azi/rng in Batch.incidence()
    # look_E = np.full((n_y, n_x), np.nan, dtype=np.float32)
    # look_N = np.full((n_y, n_x), np.nan, dtype=np.float32)
    # look_U = np.full((n_y, n_x), np.nan, dtype=np.float32)

    inv_azi[valid_mask] = inv_azi_valid
    inv_rng[valid_mask] = inv_rng_valid
    inv_ele[valid_mask] = inv_ele_valid
    # look_E[valid_mask] = look_E_valid
    # look_N[valid_mask] = look_N_valid
    # look_U[valid_mask] = look_U_valid
    del inv_azi_valid, inv_rng_valid, inv_ele_valid  # , look_E_valid, look_N_valid, look_U_valid

    if _timings is not None:
        _timings['scatter'] = time.perf_counter() - t0
        print(f'  Scatter to grid: {_timings["scatter"]:.2f}s')

    # Step 9: Compute topo by cv2.remap with cubic interpolation
    # inv_azi[y,x], inv_rng[y,x] tell us: output (y,x) → radar (azi,rng)
    # We need the inverse mapping: radar (azi,rng) → output (y,x)
    # Scatter elevation to radar grid using nearest neighbor bincount
    t0 = time.perf_counter()
    if compute_topo:
        valid_mask_flat = valid_mask.ravel()
        valid_azi_flat = inv_azi.ravel()[valid_mask_flat]
        valid_rng_flat = inv_rng.ravel()[valid_mask_flat]
        valid_ele_flat = inv_ele.ravel()[valid_mask_flat]

        # Convert to radar grid indices and round to nearest
        azi_round = np.round(valid_azi_flat - azi_coords[0]).astype(np.int64)
        rng_round = np.round(valid_rng_flat - rng_coords[0]).astype(np.int64)
        del valid_azi_flat, valid_rng_flat

        # Scatter elevation using bincount (average if multiple points hit same cell)
        grid_size = n_azi * n_rng
        m = (azi_round >= 0) & (azi_round < n_azi) & (rng_round >= 0) & (rng_round < n_rng)
        idx = (azi_round[m] * n_rng + rng_round[m]).astype(np.int64)
        del azi_round, rng_round

        ele_sum = np.bincount(idx, weights=valid_ele_flat[m], minlength=grid_size)
        ele_cnt = np.bincount(idx, minlength=grid_size)
        del idx, m, valid_ele_flat, valid_mask_flat

        with np.errstate(invalid='ignore', divide='ignore'):
            ele_gmtsar_full = (ele_sum / np.maximum(ele_cnt, 1)).reshape(n_azi, n_rng).astype(np.float32)
        holes = ele_cnt.reshape(n_azi, n_rng) == 0
        del ele_sum, ele_cnt

        # Fill holes with nearest valid elevation using distance transform (O(n) algorithm)
        if holes.any() and not holes.all():
            from scipy.ndimage import distance_transform_edt
            _, nearest_idx = distance_transform_edt(holes, return_distances=True, return_indices=True)
            ele_gmtsar_full[holes] = ele_gmtsar_full[nearest_idx[0][holes], nearest_idx[1][holes]]
            del nearest_idx
        del holes

        if _timings is not None:
            _timings['topo_scatter'] = time.perf_counter() - t0
            print(f'  Topo scatter+fill: {_timings["topo_scatter"]:.2f}s')
    else:
        ele_gmtsar_full = None

    # Build dataset
    # NOTE: look_E/N/U removed - incidence is computed from azi/rng in Batch.incidence()
    trans = xr.Dataset({
        'rng': xr.DataArray(inv_rng, coords={'y': out_y, 'x': out_x}, dims=['y', 'x']),
        'azi': xr.DataArray(inv_azi, coords={'y': out_y, 'x': out_x}, dims=['y', 'x']),
        'ele': xr.DataArray(inv_ele, coords={'y': out_y, 'x': out_x}, dims=['y', 'x']),
        # 'look_E': xr.DataArray(look_E, coords={'y': out_y, 'x': out_x}, dims=['y', 'x']),
        # 'look_N': xr.DataArray(look_N, coords={'y': out_y, 'x': out_x}, dims=['y', 'x']),
        # 'look_U': xr.DataArray(look_U, coords={'y': out_y, 'x': out_x}, dims=['y', 'x']),
    })

    # Add georeference
    from insardev_toolkit.datagrid import datagrid
    trans = datagrid.spatial_ref(trans, epsg)
    trans.attrs['spatial_ref'] = trans.spatial_ref.attrs['spatial_ref']
    trans = trans.drop_vars('spatial_ref')

    if _timings is not None:
        total = sum(_timings.values())
        print(f'  TOTAL: {total:.2f}s')

    if ele_gmtsar_full is not None:
        topo = xr.DataArray(ele_gmtsar_full,
                           coords={'a': azi_coords, 'r': rng_coords},
                           dims=['a', 'r']).rename('topo')
    else:
        topo = None

    return topo, trans


def compute_conversion_chunked(prm, dem_path, geometry, outdir,
                               scale_factor=2.0,
                               epsg=None,
                               resolution=(16.0, 4.0),
                               chunk=(8192, 8192),
                               compute_topo=True,
                               n_jobs=-1,
                               netcdf_engine='netcdf4',
                               debug=False):
    """
    Compute transform and topo tile-by-tile, writing directly to zarr.

    Memory-efficient version that never builds full arrays in memory.
    Workers read DEM chunks directly from file - no full DEM in memory.
    Suitable for large NISAR data on limited RAM systems (e.g., 12GB Colab).

    Directory structure:
    - outdir/transform/   : Persistent transform zarr (azi, rng, ele) for geocoding
    - outdir/conversion/topo/ : Temporary topo zarr for conversion phase only

    Parameters
    ----------
    prm : PRM
        The reference PRM object with orbit_df attached.
    dem_path : str
        Path to DEM file (GeoTIFF, NetCDF, etc.)
    geometry : shapely.geometry
        Scene geometry for DEM bounds estimation.
    outdir : str
        Scene output directory.
    scale_factor : float, optional
        Scale factor for integer compression. Default is 2.0.
    epsg : int, optional
        Target EPSG code. If None, auto-detect UTM zone.
    resolution : tuple[float, float], optional
        Output resolution (dy, dx) in meters. Default is (16.0, 4.0).
    chunk : tuple[int, int], optional
        Tile size (y, x) for chunked processing. Default is (8192, 8192).
    compute_topo : bool, optional
        If True, compute topo array in radar coords. Default is True.
    n_jobs : int, optional
        Number of parallel workers. Default is -1 (use all cores).
    debug : bool, optional
        If True, prints timing information. Default is False.
    """
    import os
    import time
    import zarr
    import xarray as xr
    import joblib
    import cv2
    import warnings
    warnings.filterwarnings('ignore')

    t0_total = time.perf_counter()

    # Handle n_jobs=-1 (use all cores) - joblib convention
    if n_jobs is None or n_jobs == -1:
        n_jobs = os.cpu_count()
    if debug:
        print(f'Parallel tile processing: n_jobs={n_jobs}')

    # Get orbit data from PRM
    orbit_df = prm.orbit_df
    if orbit_df is None:
        raise ValueError("PRM object has no orbit_df attached")

    orbit_df = orbit_df.copy()
    orbit_df['clock'] = orbit_df['isec']
    orbit_time = orbit_df['isec'].values
    orbit_pos = orbit_df[['px', 'py', 'pz']].values
    orbit_vel = orbit_df[['vx', 'vy', 'vz']].values

    # Get PRM parameters
    clock_start = (prm.get('clock_start') % 1.0) * 86400
    clock_start_days = prm.get('clock_start') % 1.0
    prf = prm.get('PRF')
    near_range = prm.get('near_range')
    rng_samp_rate = prm.get('rng_samp_rate')
    earth_radius = prm.get('earth_radius')
    lookdir = prm.get('lookdir') if 'lookdir' in prm.df.index else 'R'
    a_max, r_max = prm.bounds()
    num_lines = int(a_max)
    num_rng = int(r_max)

    # Radar grid coordinates
    azi_coords = np.arange(0.5, a_max, 1, dtype=np.float32)
    rng_coords = np.arange(0.5, r_max, 1, dtype=np.float32)
    n_azi = len(azi_coords)
    n_rng = len(rng_coords)

    # WGS84 constants
    ra = 6378137.0
    rc = 6356752.31424518
    e2 = np.float32((ra**2 - rc**2) / ra**2)

    # Auto-detect EPSG from geometry centroid
    if epsg is None:
        centroid = geometry.centroid
        epsg = get_utm_epsg(centroid.y, centroid.x)

    dy, dx = resolution

    if debug:
        print(f'Radar grid: {n_azi} x {n_rng} = {n_azi * n_rng:,} points')

    # Step 1: Compute output grid bounds from full boundary (parallel per chunk)
    # Workers read DEM chunks directly from file - NO full DEM in memory
    t0 = time.perf_counter()

    # Boundary task definitions (just indices, no data)
    # line_type: 0=first_row, 1=last_row, 2=first_col, 3=last_col
    bnd_chunk_size = 1000
    tasks = []
    # First row: azi=0, rng varies
    for start in range(0, n_rng, bnd_chunk_size):
        tasks.append((0, start, min(start + bnd_chunk_size, n_rng)))
    # Last row: azi=n_azi-1, rng varies
    for start in range(0, n_rng, bnd_chunk_size):
        tasks.append((1, start, min(start + bnd_chunk_size, n_rng)))
    # First col: azi varies, rng=0
    for start in range(0, n_azi, bnd_chunk_size):
        tasks.append((2, start, min(start + bnd_chunk_size, n_azi)))
    # Last col: azi varies, rng=n_rng-1
    for start in range(0, n_azi, bnd_chunk_size):
        tasks.append((3, start, min(start + bnd_chunk_size, n_azi)))

    # Build worker arguments with PRE-SLICED arrays (not full arrays)
    # This prevents massive memory duplication when pickling for spawn workers
    worker_args = []
    n_bnd = 0
    for line_type, start, end in tasks:
        n_bnd += end - start
        # Pre-slice based on line_type - pass only what each worker needs
        if line_type == 0:  # first row: azi=const, rng varies
            chunk_azi = np.full(end - start, azi_coords[0], dtype=np.float32)
            chunk_rng = rng_coords[start:end].astype(np.float32)
        elif line_type == 1:  # last row: azi=const, rng varies
            chunk_azi = np.full(end - start, azi_coords[-1], dtype=np.float32)
            chunk_rng = rng_coords[start:end].astype(np.float32)
        elif line_type == 2:  # first col: azi varies, rng=const
            chunk_azi = azi_coords[start:end].astype(np.float32)
            chunk_rng = np.full(end - start, rng_coords[0], dtype=np.float32)
        else:  # last col: azi varies, rng=const
            chunk_azi = azi_coords[start:end].astype(np.float32)
            chunk_rng = np.full(end - start, rng_coords[-1], dtype=np.float32)

        worker_args.append((
            chunk_azi, chunk_rng, dem_path,
            orbit_time, orbit_pos, orbit_vel, clock_start, prf,
            near_range, rng_samp_rate, earth_radius, epsg, lookdir, netcdf_engine
        ))
    del tasks

    # Parallel execution using subprocess pool with memory isolation
    # Each worker processes one chunk then exits (max_tasks_per_child=1), releasing memory
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context('spawn'),
                             max_tasks_per_child=1) as executor:
        results = list(executor.map(_process_boundary_worker, worker_args))
    del worker_args

    # Concatenate results
    bnd_y = np.concatenate([r[0] for r in results])
    bnd_x = np.concatenate([r[1] for r in results])
    del results

    # Compute bounds with margin - ensure scalars (not 0-d arrays)
    margin = 100
    valid_bnd = np.isfinite(bnd_y) & np.isfinite(bnd_x)
    y_min = float(dy * (np.floor(np.nanmin(bnd_y[valid_bnd]) / dy) - margin))
    y_max = float(dy * (np.ceil(np.nanmax(bnd_y[valid_bnd]) / dy) + margin))
    x_min = float(dx * (np.floor(np.nanmin(bnd_x[valid_bnd]) / dx) - margin))
    x_max = float(dx * (np.ceil(np.nanmax(bnd_x[valid_bnd]) / dx) + margin))
    dy = float(dy)
    dx = float(dx)

    # Compute grid dimensions without creating full arrays (memory-efficient)
    n_y = int((y_max - y_min) / dy)
    n_x = int((x_max - x_min) / dx)

    if debug:
        print(f'Output grid: {n_y} x {n_x}, bounds from {n_bnd} boundary points in {time.perf_counter() - t0:.1f}s')

    # Free boundary arrays - workers will determine tile validity internally
    del bnd_y, bnd_x, valid_bnd

    # Step 2: Pre-create zarr arrays (lazy - no memory allocation)
    fill_value = np.iinfo(np.int32).max
    chunk_y, chunk_x = chunk
    zarr_chunks = (min(chunk_y, n_y), min(chunk_x, n_x))
    radar_chunks = (min(chunk_y, n_azi), min(chunk_x, n_rng))

    # Transform zarr - persistent at scene level for geocoding
    transform_dir = os.path.join(outdir, 'transform')
    os.makedirs(transform_dir, exist_ok=True)
    trans_store = zarr.storage.LocalStore(transform_dir)
    trans_root = zarr.group(store=trans_store, zarr_format=3, overwrite=True)

    azi_arr = trans_root.create_array('azi', shape=(n_y, n_x), chunks=zarr_chunks,
                                       dtype=np.int32, fill_value=fill_value, overwrite=True,
                                       dimension_names=['y', 'x'])
    rng_arr = trans_root.create_array('rng', shape=(n_y, n_x), chunks=zarr_chunks,
                                       dtype=np.int32, fill_value=fill_value, overwrite=True,
                                       dimension_names=['y', 'x'])
    ele_arr = trans_root.create_array('ele', shape=(n_y, n_x), chunks=zarr_chunks,
                                       dtype=np.int32, fill_value=fill_value, overwrite=True,
                                       dimension_names=['y', 'x'])

    # Topo zarr - temporary in conversion dir for phase correction only
    if compute_topo:
        conversion_dir = os.path.join(outdir, 'conversion')
        topo_dir = os.path.join(conversion_dir, 'topo')
        os.makedirs(topo_dir, exist_ok=True)
        topo_store = zarr.storage.LocalStore(topo_dir)
        topo_root = zarr.group(store=topo_store, zarr_format=3, overwrite=True)
        topo_arr = topo_root.create_array('topo', shape=(n_azi, n_rng), chunks=radar_chunks,
                                           dtype=np.int32, fill_value=fill_value, overwrite=True,
                                           dimension_names=['a', 'r'])

    # Step 4: Compute transform tile-by-tile using subprocess pool
    # Each worker processes one tile then exits (max_tasks_per_child=1), releasing memory
    # This pattern matches S1 processing to prevent memory accumulation
    t0 = time.perf_counter()
    n_tiles = ((n_y + chunk_y - 1) // chunk_y) * ((n_x + chunk_x - 1) // chunk_x)
    row_batch = 512  # Process 512 rows at a time to limit memory

    # Write coordinate arrays (compute directly, don't keep in memory)
    out_y_coords = (y_min + dy * (np.arange(n_y) + 0.5)).astype(np.float64)
    out_x_coords = (x_min + dx * (np.arange(n_x) + 0.5)).astype(np.float64)
    y_arr = trans_root.create_array('y', data=out_y_coords, chunks=(n_y,), overwrite=True,
                                     dimension_names=['y'])
    x_arr = trans_root.create_array('x', data=out_x_coords, chunks=(n_x,), overwrite=True,
                                     dimension_names=['x'])
    y_arr.attrs['_ARRAY_DIMENSIONS'] = ['y']
    x_arr.attrs['_ARRAY_DIMENSIONS'] = ['x']
    del out_y_coords, out_x_coords  # Free immediately after writing to zarr

    # Serialize orbit_df for subprocess pickling
    orbit_dict = orbit_df.to_dict()

    # Build tile arguments - workers determine validity internally
    # No polygon check in main process - workers skip tiles with no valid data
    tile_args = []
    for iy in range(0, n_y, chunk_y):
        jy = min(iy + chunk_y, n_y)
        for ix in range(0, n_x, chunk_x):
            jx = min(ix + chunk_x, n_x)
            tile_args.append((
                transform_dir, dem_path, epsg,
                (iy, jy, ix, jx),  # tile_bounds
                (y_min, dy, x_min, dx),  # grid_params - worker computes coords locally
                orbit_dict, clock_start_days, prf,
                near_range, rng_samp_rate, num_lines, earth_radius,
                n_azi, n_rng, ra, e2,
                scale_factor, fill_value, row_batch, lookdir, netcdf_engine
            ))

    if debug:
        print(f'Computing transform: {len(tile_args)} tiles (row_batch={row_batch})...')

    # Process tiles using subprocess pool with memory isolation
    # Each worker processes one tile then exits, releasing all memory
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context('spawn'),
                             max_tasks_per_child=1) as executor:
        list(executor.map(_process_tile_worker, tile_args))

    del tile_args

    # Add transform metadata
    for arr in [azi_arr, rng_arr, ele_arr]:
        arr.attrs['scale_factor'] = 1/scale_factor
        arr.attrs['add_offset'] = 0
        arr.attrs['_FillValue'] = int(fill_value)
        arr.attrs['_ARRAY_DIMENSIONS'] = ['y', 'x']

    from pyproj import CRS
    trans_root.attrs['spatial_ref'] = CRS.from_epsg(epsg).to_wkt()
    zarr.consolidate_metadata(trans_store)

    if debug:
        print(f'Transform done: {time.perf_counter() - t0:.1f}s')

    # Step 5: Compute topo tile-by-tile using forward transform (radar → geo)
    # Process tiles in parallel with subprocess pool (like transform tiles)
    if compute_topo:
        t0 = time.perf_counter()

        # Build tile arguments for parallel processing
        topo_tile_args = []
        for ia in range(0, n_azi, chunk_y):
            ja = min(ia + chunk_y, n_azi)
            for ir in range(0, n_rng, chunk_x):
                jr = min(ir + chunk_x, n_rng)
                # Pass coordinate slices for this tile (not full arrays!)
                azi_coords_tile = azi_coords[ia:ja].copy()
                rng_coords_tile = rng_coords[ir:jr].copy()
                topo_tile_args.append((
                    topo_dir, dem_path, (ia, ja, ir, jr),
                    azi_coords_tile, rng_coords_tile,
                    orbit_dict, clock_start_days, prf, near_range, rng_samp_rate, earth_radius,
                    ra, e2, scale_factor, fill_value, row_batch, lookdir, netcdf_engine
                ))

        if debug:
            print(f'Computing topo: {len(topo_tile_args)} tiles (row_batch={row_batch})...')

        # Process tiles using subprocess pool with memory isolation
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context('spawn'),
                                 max_tasks_per_child=1) as executor:
            list(executor.map(_process_topo_worker, topo_tile_args))

        del topo_tile_args

        # Add topo metadata
        topo_arr.attrs['scale_factor'] = 1/scale_factor
        topo_arr.attrs['add_offset'] = 0
        topo_arr.attrs['_FillValue'] = int(fill_value)
        topo_arr.attrs['_ARRAY_DIMENSIONS'] = ['a', 'r']

        a_arr = topo_root.create_array('a', data=azi_coords.astype(np.float64), chunks=(len(azi_coords),), overwrite=True,
                                        dimension_names=['a'])
        r_arr = topo_root.create_array('r', data=rng_coords.astype(np.float64), chunks=(len(rng_coords),), overwrite=True,
                                        dimension_names=['r'])
        a_arr.attrs['_ARRAY_DIMENSIONS'] = ['a']
        r_arr.attrs['_ARRAY_DIMENSIONS'] = ['r']
        zarr.consolidate_metadata(topo_store)

        if debug:
            print(f'Topo done: {time.perf_counter() - t0:.1f}s')

    if debug:
        print(f'Total conversion: {time.perf_counter() - t0_total:.1f}s')


# =============================================================================
# XCORR REFINEMENT UTILITIES
# =============================================================================

def xcorr_patch(patch1: np.ndarray, patch2: np.ndarray, hann: np.ndarray,
                min_valid_fraction: float = 0.5, min_response: float = 0.2) -> dict | None:
    """
    Compute amplitude cross-correlation offset between two patches.

    Parameters
    ----------
    patch1 : np.ndarray
        Reference patch (complex64).
    patch2 : np.ndarray
        Repeat patch (complex64).
    hann : np.ndarray
        Hanning window (float32), same size as patches.
    min_valid_fraction : float
        Minimum fraction of non-zero pixels required.
    min_response : float
        Minimum correlation response to accept result.

    Returns
    -------
    dict or None
        {'dy': float, 'dx': float, 'response': float} or None if invalid.
    """
    import cv2

    # Check valid data
    valid = (patch1 != 0) & (patch2 != 0)
    if valid.sum() < min_valid_fraction * valid.size:
        return None

    # Normalize amplitudes
    amp1 = np.abs(patch1).astype(np.float32)
    amp2 = np.abs(patch2).astype(np.float32)
    amp1_norm = ((amp1 - amp1.mean()) / (amp1.std() + 1e-10)).astype(np.float32)
    amp2_norm = ((amp2 - amp2.mean()) / (amp2.std() + 1e-10)).astype(np.float32)

    # Phase correlation
    (dx, dy), response = cv2.phaseCorrelate(amp1_norm * hann, amp2_norm * hann)

    if response < min_response:
        return None

    return {'dy': dy, 'dx': dx, 'response': response}


def xcorr_fitoffset(results: list, nx: int, ny: int, debug: bool = False) -> dict | None:
    """
    Fit bilinear model to xcorr offsets using PRM.fitoffset (robust IRLS with MAD).

    Converts xcorr results to the matrix format expected by PRM.fitoffset,
    which uses iteratively reweighted least squares with MAD-based outlier
    downweighting - proven and consistent with geometry fitting.

    Parameters
    ----------
    results : list
        List of dicts with 'cy1', 'cx1', 'dy', 'dx', 'response'.
    nx : int
        Image width (num_rng_bins) - unused, kept for API compatibility.
    ny : int
        Image height (num_lines) - unused, kept for API compatibility.
    debug : bool
        Print debug info.

    Returns
    -------
    dict or None
        Correction parameters in same format as PRM alignment:
        {
            'rshift': float, 'stretch_r': float, 'a_stretch_r': float,
            'ashift': float, 'stretch_a': float, 'a_stretch_a': float,
        }
        Returns None if insufficient valid patches (xcorr failed).
    """
    from .PRM import PRM

    n_initial = len(results)
    if n_initial < 8:
        if debug:
            print(f"  xcorr_fitoffset: only {n_initial} patches, need >= 8")
        return None  # Insufficient patches - xcorr failed

    # Filter out zero offsets (invalid/failed correlations)
    n_before_zero = len(results)
    results = [r for r in results if abs(r['dx']) > 0.01 or abs(r['dy']) > 0.01]
    n_zeros = n_before_zero - len(results)

    if len(results) < 8:
        if debug:
            print(f"  xcorr_fitoffset: {n_zeros} zero offsets filtered, only {len(results)} remain")
        return None  # Insufficient patches after zero filtering

    # Convert to PRM.fitoffset matrix format: [r, dr, a, da, SNR]
    # r = cx1 (range position), dr = dx (range offset)
    # a = cy1 (azimuth position), da = dy (azimuth offset)
    # SNR = response * 100 (scale to match expected range)
    matrix = np.array([
        [r['cx1'], r['dx'], r['cy1'], r['dy'], r['response'] * 100]
        for r in results
    ])

    if debug:
        dx = matrix[:, 1]
        dy = matrix[:, 3]
        print(f"  xcorr_fitoffset: {n_initial} initial, {n_zeros} zeros, {len(results)} final")
        print(f"  offset range: dx=[{dx.min():.2f}, {dx.max():.2f}], dy=[{dy.min():.2f}, {dy.max():.2f}]")

    # Use PRM.fitoffset - robust IRLS with MAD-based outlier downweighting
    # rank=3 for bilinear model: offset = c0 + c1*r + c2*a
    try:
        prm_result = PRM.fitoffset(3, 3, matrix, SNR=20)
    except Exception as e:
        if debug:
            print(f"  PRM.fitoffset failed: {e}")
        return None

    # Extract coefficients from PRM result
    return {
        'rshift': prm_result.get('rshift') + prm_result.get('sub_int_r'),
        'stretch_r': prm_result.get('stretch_r'),
        'a_stretch_r': prm_result.get('a_stretch_r'),
        'ashift': prm_result.get('ashift') + prm_result.get('sub_int_a'),
        'stretch_a': prm_result.get('stretch_a'),
        'a_stretch_a': prm_result.get('a_stretch_a'),
    }


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


def xcorr_refine_slc(ref_path: str, rep_path: str,
                     ashift: float, rshift: float,
                     stretch_a: float = 0.0, stretch_r: float = 0.0,
                     a_stretch_a: float = 0.0, a_stretch_r: float = 0.0,
                     patch_size: int = 256,
                     min_response: float = 0.1, debug: bool = False) -> dict | None:
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
    dict or None
        Correction coefficients (same as xcorr_fitoffset output).
        Returns None if xcorr failed (insufficient valid patches).
    """
    import rasterio

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
        if debug:
            print("Xcorr fitoffset failed - insufficient valid patches")
        return None

    if debug:
        print(f"Xcorr fitoffset result:")
        print(f"  ashift={corrections['ashift']:.4f}, stretch_a={corrections['stretch_a']:.8f}, a_stretch_a={corrections['a_stretch_a']:.8f}")
        print(f"  rshift={corrections['rshift']:.4f}, stretch_r={corrections['stretch_r']:.8f}, a_stretch_r={corrections['a_stretch_r']:.8f}")

    return corrections
