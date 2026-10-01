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


def precise_transform_dir(outdir):
    """The precise transform of compute_conversion_chunked: azi, rng and ele as the tile workers compute them
    (float32, NaN outside the swath). The processing reads it; outdir/transform is only its rounded copy for the
    stack. Temporary: removed with the conversion directory when the dates are processed."""
    import os
    return os.path.join(outdir, 'conversion', 'transform')


def get_dem_wgs84ellipsoid(dem_path, geometry, buffer_degrees=0.04, geoid_correction=True, datum=None):
    """Load ellipsoid-corrected DEM cropped to geometry bounds.

    Reads only the needed window directly from disk through h5py.
    No full load, no lazy/dask overhead.

    Parameters
    ----------
    dem_path : str
        Path to DEM file: NetCDF4 (.nc, .netcdf, .grd) or VRT of NetCDF4 tiles (.vrt).
    geometry : shapely.geometry
        Geometry for cropping (uses bounds).
    buffer_degrees : float, optional
        Buffer around geometry bounds in degrees. Default is 0.04.
    geoid_correction : bool, optional
        Apply geoid correction. Set False for approximate use (e.g., boundary).
        Default is True.
    datum : str, optional
        The vertical datum of the DEM: 'EGM2008', 'EGM96' or 'ellipsoid', as insardev_toolkit
        utils_geoid.dem_datum() resolves it once per DEM (Satellite.dem_datum()). None resolves it from the file.

    Returns
    -------
    xarray.DataArray
        DEM with WGS84 ellipsoidal heights, float32: DEM height + the geoid height of its datum
        (insardev_toolkit utils_geoid.ellipsoidal_height()).
    """
    import xarray as xr
    from insardev_toolkit import utils_tiles

    # Get bounds from geometry
    bounds = geometry.bounds  # (minx, miny, maxx, maxy)
    lon_min, lat_min, lon_max, lat_max = bounds
    lon_min -= buffer_degrees
    lat_min -= buffer_degrees
    lon_max += buffer_degrees
    lat_max += buffer_degrees

    # Read only the needed window from disk: a NetCDF4 grid or the tiles of a VRT
    window = utils_tiles.read_dem(dem_path, (lon_min, lat_min, lon_max, lat_max))
    if window is None:
        return None
    ortho_vals, ortho_lat, ortho_lon = window

    # Create xarray for geoid interpolation
    ortho = xr.DataArray(
        ortho_vals,
        coords={'lat': ortho_lat, 'lon': ortho_lon},
        dims=['lat', 'lon']
    )

    if ortho.size == 0:
        return None

    if geoid_correction:
        # Apply geoid correction (convert DEM heights to ellipsoidal heights): the geoid of the DEM's vertical datum
        from insardev_toolkit import utils_geoid
        height = utils_geoid.ellipsoidal_height(ortho, utils_geoid.dem_datum(dem_path, datum))
        return height.astype(np.float32, copy=False)
    else:
        return ortho.astype(np.float32)


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
    (trans_dir, precise_dir, dem_path, epsg,
     tile_bounds,  # (iy, jy, ix, jx) - indices into output grid
     grid_params,  # (y_min, dy, x_min, dx) - compute coords locally
     orbit_dict, clock_start_days, prf,
     near_range, rng_samp_rate, num_lines, earth_radius,
     n_azi, n_rng, ra, e2,
     scale_factor, fill_value, row_batch, lookdir, fp_wkb, datum) = args

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
    # per-tile extent of azi, rng and ele, folded as the batches are written
    extent = np.full((3, 2), np.nan, dtype=np.float64)

    azi_arr = trans_root['azi']
    rng_arr = trans_root['rng']
    ele_arr = trans_root['ele']
    # the precise transform (precise_transform_dir), which the processing reads
    precise_root = zarr.open(zarr.storage.LocalStore(precise_dir), mode='r+')

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

    # === THE SLC FOOTPRINT of compute_conversion_chunked (_precise_footprint) ===
    # The Doppler solve skips a row batch outside it and the pixels of a batch across its edge outside it; a batch
    # inside it is solved whole
    import shapely
    from shapely.geometry import box as shapely_box
    radar_polygon = shapely.from_wkb(fp_wkb)
    shapely.prepare(radar_polygon)

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
        # float64, and the ECEF of the pixels below with it (as the ecef tile worker): in float32 a longitude near
        # 100 degrees steps by 0.8 m, and the ECEF computed from float32 values rounds again, which the zero-Doppler
        # and slant-range solve carries into azi/rng (per row batch, so the memory stays bounded)
        batch_lat = np.asarray(batch_lat, dtype=np.float64).reshape(batch_shape)
        batch_lon = np.asarray(batch_lon, dtype=np.float64).reshape(batch_shape)

        # Check if batch intersects radar polygon - skip if entirely outside
        batch_box = shapely_box(
            float(x_batch.min()), float(y_batch.min()),
            float(x_batch.max()), float(y_batch.max())
        )
        if not radar_polygon.intersects(batch_box):
            # Entire batch is outside radar coverage - skip
            del x_grid, y_grid, batch_lat, batch_lon
            continue

        # The pixels inside the footprint: all of a batch whose pixel centres lie inside it, else pixel by pixel
        x_flat = (x_min + dx * (np.arange(ix, jx) + 0.5))
        y_flat = (y_min + dy * (np.arange(iy + by, iy + ey) + 0.5))
        if radar_polygon.contains(shapely_box(float(x_flat.min()), float(y_flat.min()),
                                              float(x_flat.max()), float(y_flat.max()))):
            inside_mask = np.ones(batch_shape, dtype=bool)
        else:
            xx, yy = np.meshgrid(x_flat, y_flat)
            inside_mask = shapely.contains_xy(radar_polygon, xx.ravel(), yy.ravel()).reshape(batch_shape)
            del xx, yy
        del x_flat, y_flat

        del x_grid, y_grid

        # Read DEM tile from file (h5py window read - no full load)
        buffer_deg = 0.02
        batch_geom = shapely_box(
            float(np.nanmin(batch_lon)) - buffer_deg,
            float(np.nanmin(batch_lat)) - buffer_deg,
            float(np.nanmax(batch_lon)) + buffer_deg,
            float(np.nanmax(batch_lat)) + buffer_deg
        )
        dem_tile = get_dem_wgs84ellipsoid(dem_path, batch_geom, buffer_degrees=0.01, datum=datum)

        if dem_tile is None or dem_tile.size == 0 or len(dem_tile.lat) < 2 or len(dem_tile.lon) < 2:
            # DEM tile missing or too small for interpolation - fill with NaN
            batch_ele = np.full(batch_shape, np.nan, dtype=np.float32)
        else:
            # Interpolate DEM tile using cv2.remap at each pixel's post position in the lattice of the whole DEM
            # file, not of this window: the window follows the tile, and a position measured from the window's own
            # first post and step (and rounded to float32 there) took another of cv2's 1/32 fractions in another
            # window, so ele, and azi/rng with it, depended on the chunk size. The file position is rounded to the
            # 1/32 of cv2.remap there and shifted by the window's first post (whole posts: exact in float32)
            from insardev_toolkit import utils_tiles
            dem_grid = utils_tiles.open_grid(dem_path)
            dem_lat = dem_tile.lat.values.astype(np.float64)
            dem_lon = dem_tile.lon.values.astype(np.float64)
            dem_vals = dem_tile.values.astype(np.float32)

            def lattice_map(coords, first, values):
                # the window's first post in the file lattice (read_dem returns a slice of it), and each value's
                # fractional index in the lattice, linear between its two posts
                i0 = int(np.searchsorted(coords, first))
                assert coords[i0] == first, 'ERROR: the DEM window is not a slice of the DEM lattice'
                v = values.astype(np.float64)
                g = np.clip(np.searchsorted(coords, v, side='right') - 1, 0, coords.size - 2)
                pos = g + (v - coords[g]) / (coords[g + 1] - coords[g])
                return (np.round(pos * 32) / 32 - i0).astype(np.float32)

            map_row = lattice_map(dem_grid.lat, dem_lat[0], batch_lat)
            map_col = lattice_map(dem_grid.lon, dem_lon[0], batch_lon)
            batch_ele = cv2.remap(dem_vals, map_col, map_row,
                                  interpolation=cv2.INTER_CUBIC,
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
            sample_idx = _doppler_sample_idx(n_azi_times)
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

        # Mark out-of-bounds as NaN (in the flat inside arrays), and a pixel without a radar position (NaN azi or
        # rng): its ele is not written either
        out_of_bounds = ((batch_azi_pix < 0.5) | (batch_azi_pix > n_azi - 0.5) |
                        (batch_rng_pix < 0.5) | (batch_rng_pix > n_rng - 0.5) |
                        ~np.isfinite(batch_azi_pix) | ~np.isfinite(batch_rng_pix))
        batch_azi_pix[out_of_bounds] = np.nan
        batch_rng_pix[out_of_bounds] = np.nan

        # ele is the WGS84 ellipsoidal height of the DEM at the pixel, as the S1 transform stores it (the topo pass
        # computes the GMTSAR sphere height from it)
        batch_ele_inside = ele_flat
        batch_ele_inside[out_of_bounds] = np.nan
        del out_of_bounds, lat_flat, lon_flat, ele_flat

        # Scatter inside results back to full batch arrays (NaN for outside pixels)
        batch_azi = np.full(batch_shape, np.nan, dtype=np.float32)
        batch_rng = np.full(batch_shape, np.nan, dtype=np.float32)
        batch_ele_out = np.full(batch_shape, np.nan, dtype=np.float32)
        batch_azi.ravel()[inside_flat] = batch_azi_pix
        batch_rng.ravel()[inside_flat] = batch_rng_pix
        batch_ele_out.ravel()[inside_flat] = batch_ele_inside
        del batch_azi_pix, batch_rng_pix, batch_ele_inside, inside_flat, inside_mask
        del batch_lat, batch_lon, batch_ele

        # Write batch to zarr: rounded for the stack, and as computed to the precise transform
        azi_arr[iy + by:iy + ey, ix:jx] = to_int32(batch_azi)
        rng_arr[iy + by:iy + ey, ix:jx] = to_int32(batch_rng)
        ele_arr[iy + by:iy + ey, ix:jx] = to_int32(batch_ele_out)
        precise_root['azi'][iy + by:iy + ey, ix:jx] = batch_azi
        precise_root['rng'][iy + by:iy + ey, ix:jx] = batch_rng
        precise_root['ele'][iy + by:iy + ey, ix:jx] = batch_ele_out
        # how far each variable reaches, folded in while the batch is still in
        # hand and before packing, so it is in physical units. The parent
        # combines the tiles into actual_range; an all-NaN batch gives NaN and
        # drops out of that by itself.
        with np.errstate(invalid='ignore'):
            for _i, _b in enumerate((batch_azi, batch_rng, batch_ele_out)):
                if np.isfinite(_b).any():
                    extent[_i][0] = np.fmin(extent[_i][0], np.nanmin(_b))
                    extent[_i][1] = np.fmax(extent[_i][1], np.nanmax(_b))
        del batch_azi, batch_rng, batch_ele_out

    return extent


def _process_topo_worker(args):
    """Worker function for computing topo (the transform's DEM points gridded in radar coordinates) for a single tile.

    Must be at module level for multiprocessing spawn to pickle it.
    Each worker processes one tile then exits (max_tasks_per_child=1), releasing memory.

    As compute_transform_inverse() builds the Sentinel-1 topo (and GMTSAR dem2topo_ra.csh the topo_ra): the DEM
    points of the output grid, mapped to radar coordinates by the transform (azi, rng) and carrying their radius minus
    the PRM earth_radius (the GMTSAR sphere height, from the pixel's latitude and the transform's WGS84 height ele),
    are averaged in the radar cell each one falls in, and a cell no point falls in takes the value of the nearest cell
    that has one. So each radar pixel holds the terrain height at that pixel. The tile is gridded with a margin of
    cells around it, so its nearest fill at the tile edges sees the points beyond them.
    """
    import numpy as np
    import zarr
    from pyproj import Transformer
    from scipy.ndimage import distance_transform_edt

    # Unpack arguments
    (topo_dir, trans_dir, precise_dir, tile_bounds, window, margin, azi0, rng0, n_azi, n_rng,
     scale_factor, fill_value, epsg, earth_radius) = args

    ia, ja, ir, jr = tile_bounds
    oy0, oy1, ox0, ox1 = window
    # the tile with its margin of cells, the gridding area
    ea0, ea1 = max(0, ia - margin), min(n_azi, ja + margin)
    er0, er1 = max(0, ir - margin), min(n_rng, jr + margin)
    ext_h, ext_w = ea1 - ea0, er1 - er0

    # the output grid coordinates of the transform, the azi, rng and ele of the precise transform (float32, NaN
    # outside the swath), not their copy rounded for the stack
    trans_root = zarr.open_group(zarr.storage.LocalStore(trans_dir), mode='r')
    out_y, out_x = trans_root['y'][:], trans_root['x'][:]
    del trans_root
    trans_root = zarr.open_group(zarr.storage.LocalStore(precise_dir), mode='r')
    azi_arr, rng_arr, ele_arr = trans_root['azi'], trans_root['rng'], trans_root['ele']
    # the latitude of the output pixel centres, for the radius of the points
    to_lonlat = Transformer.from_crs(epsg, 4326, always_xy=True)
    ra, e2 = 6378137.0, 1 - 6356752.31424518**2 / 6378137.0**2

    # Read the transform window one zarr chunk at a time (each read decompresses only its chunk) and grid it in
    # sub-batches of rows, accumulating the points of the gridding area
    ele_sum = np.zeros(ext_h * ext_w, dtype=np.float64)
    ele_cnt = np.zeros(ext_h * ext_w, dtype=np.int64)
    batch_y, batch_x = azi_arr.chunks
    for by in range((oy0 // batch_y) * batch_y, oy1, batch_y):
        y0, y1 = max(by, oy0), min(by + batch_y, oy1)
        for bx in range((ox0 // batch_x) * batch_x, ox1, batch_x):
            x0, x1 = max(bx, ox0), min(bx + batch_x, ox1)
            azi_raw = azi_arr[y0:y1, x0:x1]
            rng_raw = rng_arr[y0:y1, x0:x1]
            ele_raw = ele_arr[y0:y1, x0:x1]
            for r0 in range(0, y1 - y0, 512):
                a_raw, r_raw, e_raw = azi_raw[r0:r0 + 512], rng_raw[r0:r0 + 512], ele_raw[r0:r0 + 512]
                valid = np.isfinite(a_raw) & np.isfinite(r_raw) & np.isfinite(e_raw)
                # radar cell of each point: cell k is centred on coordinate azi0 + k (rng0 + k)
                ca = np.round(a_raw[valid].astype(np.float64) - azi0).astype(np.int64) - ea0
                cr = np.round(r_raw[valid].astype(np.float64) - rng0).astype(np.int64) - er0
                m = (ca >= 0) & (ca < ext_h) & (cr >= 0) & (cr < ext_w)
                if m.any():
                    # the sphere height of each point, |P| - earth_radius, from its latitude and WGS84 height ele
                    # (float64), in pieces of points so that the temporaries stay small
                    flat = np.flatnonzero(valid)[m]
                    ele = e_raw[valid][m].astype(np.float64)
                    for p0 in range(0, ele.size, 65536):
                        k = flat[p0:p0 + 65536]
                        _, lat = to_lonlat.transform(out_x[x0 + k % valid.shape[1]], out_y[y0 + r0 + k // valid.shape[1]])
                        sin_lat = np.sin(np.radians(lat))
                        N = ra / np.sqrt(1 - e2 * sin_lat**2)
                        h = ele[p0:p0 + 65536]
                        ele[p0:p0 + 65536] = np.hypot((N + h) * np.sqrt(1 - sin_lat**2),
                                                      (N * (1 - e2) + h) * sin_lat) - earth_radius
                        del k, lat, sin_lat, N, h
                    del flat
                    idx = ca[m] * ext_w + cr[m]
                    ele_sum += np.bincount(idx, weights=ele, minlength=ext_h * ext_w)
                    ele_cnt += np.bincount(idx, minlength=ext_h * ext_w)
                    del idx, ele
                del a_raw, r_raw, e_raw, valid, ca, cr, m
            del azi_raw, rng_raw, ele_raw
    del trans_root, out_y, out_x

    holes = (ele_cnt == 0).reshape(ext_h, ext_w)
    if holes.all():
        return False
    with np.errstate(invalid='ignore', divide='ignore'):
        topo = (ele_sum / np.maximum(ele_cnt, 1)).reshape(ext_h, ext_w)
    del ele_sum, ele_cnt

    # Fill holes with nearest valid elevation using distance transform (O(n) algorithm)
    if holes.any():
        nearest_idx = distance_transform_edt(holes, return_distances=False, return_indices=True)
        topo[holes] = topo[nearest_idx[0][holes], nearest_idx[1][holes]]
        del nearest_idx
    del holes

    # Write the tile (the margin cells belong to the neighbouring tiles)
    scaled = (scale_factor * topo[ia - ea0:ja - ea0, ir - er0:jr - er0]).round()
    del topo
    finite = np.isfinite(scaled)
    int_data = np.full(scaled.shape, fill_value, dtype=np.int32)
    int_data[finite] = scaled[finite].astype(np.int32)
    topo_root = zarr.open(zarr.storage.LocalStore(topo_dir), mode='r+')
    topo_root['topo'][ia:ja, ir:jr] = int_data
    del scaled, finite, int_data

    return True


def _topo_tile_windows(precise_dir, tiles, margin, azi0, rng0, block=64):
    """Output-grid window of the transform points that fall in each radar tile (with its margin), or None.

    The azi and rng extents of every block x block cell of the output grid in the precise transform (float32, NaN
    outside the swath), read chunk row by chunk row, give for a radar tile the output blocks whose points can reach
    it; the window is their bounding box.
    """
    import numpy as np
    import zarr

    trans_root = zarr.open_group(zarr.storage.LocalStore(precise_dir), mode='r')
    azi_arr, rng_arr = trans_root['azi'], trans_root['rng']
    n_y, n_x = azi_arr.shape
    nby, nbx = -(-n_y // block), -(-n_x // block)
    lo = np.full((2, nby, nbx), np.inf, dtype=np.float64)
    hi = np.full((2, nby, nbx), -np.inf, dtype=np.float64)
    batch = (azi_arr.chunks[0] // block) * block or block
    for y0 in range(0, n_y, batch):
        y1 = min(y0 + batch, n_y)
        b0 = y0 // block
        for k, arr in enumerate((azi_arr, rng_arr)):
            raw = arr[y0:y1, :]
            h = -(-(y1 - y0) // block) * block
            pad = np.full((h, nbx * block), np.nan, dtype=np.float32)
            pad[:y1 - y0, :n_x] = raw
            del raw
            blocks = pad.reshape(h // block, block, nbx, block)
            # NaN drops out of fmin/fmax: a block without a point keeps +inf / -inf and selects no tile
            lo[k, b0:b0 + h // block] = np.fmin.reduce(blocks, axis=(1, 3), initial=np.inf)
            hi[k, b0:b0 + h // block] = np.fmax.reduce(blocks, axis=(1, 3), initial=-np.inf)
            del pad, blocks
    del trans_root
    # the extents in radar cells: cell k takes the coordinates azi0 + k - 0.5 .. azi0 + k + 0.5
    a_lo, a_hi = lo[0] - azi0, hi[0] - azi0
    r_lo, r_hi = lo[1] - rng0, hi[1] - rng0

    windows = []
    for ia, ja, ir, jr in tiles:
        sel = ((a_hi >= ia - margin - 0.5) & (a_lo < ja + margin + 0.5)
               & (r_hi >= ir - margin - 0.5) & (r_lo < jr + margin + 0.5))
        if not sel.any():
            windows.append(None)
            continue
        rows = np.nonzero(sel.any(axis=1))[0]
        cols = np.nonzero(sel.any(axis=0))[0]
        windows.append((int(rows[0]) * block, min(n_y, (int(rows[-1]) + 1) * block),
                        int(cols[0]) * block, min(n_x, (int(cols[-1]) + 1) * block)))
    return windows


def _process_boundary_worker(args):
    """Worker function for processing a boundary chunk in spawned subprocess.

    Must be at module level for multiprocessing spawn to pickle it.

    Args now contain pre-sliced arrays (chunk_azi, chunk_rng) instead of full arrays.
    Returns the edge's y, x in the output CRS (float32) for the grid bounds, and its precise outline for the
    footprint (_precise_outline).
    """
    import numpy as np
    from shapely.geometry import box

    # Unpack arguments - chunk_azi and chunk_rng are already sliced
    (chunk_azi, chunk_rng, dem_path,
     orbit_time, orbit_pos, orbit_vel, clock_start, prf,
     near_range, rng_samp_rate, earth_radius, epsg, lookdir, datum) = args

    # Fast ellipsoid transform to get approximate lon/lat
    lon_approx, lat_approx, _ = satellite_rat2llt(
        chunk_azi, chunk_rng, orbit_time, orbit_pos, orbit_vel,
        clock_start, prf, near_range, rng_samp_rate, earth_radius,
        dem=None, max_iter=1, tol=1.0, n_chunks=1, lookdir=lookdir
    )

    # The precise outline of the edge, for the footprint of the tile workers
    outline = _precise_outline(chunk_azi, chunk_rng, lon_approx, lat_approx, dem_path, orbit_time, orbit_pos,
                               orbit_vel, clock_start, prf, near_range, rng_samp_rate, earth_radius, epsg, lookdir,
                               datum)

    # Read narrow DEM chunk from file
    buffer_deg = 0.02
    lat_min = np.nanmin(lat_approx) - buffer_deg
    lat_max = np.nanmax(lat_approx) + buffer_deg
    lon_min = np.nanmin(lon_approx) - buffer_deg
    lon_max = np.nanmax(lon_approx) + buffer_deg

    chunk_geom = box(lon_min, lat_min, lon_max, lat_max)
    # Skip geoid correction for boundary - approximate positions sufficient for grid bounds
    dem_chunk = get_dem_wgs84ellipsoid(dem_path, chunk_geom, buffer_degrees=0.01,
                                       geoid_correction=False)

    if dem_chunk is None or dem_chunk.size == 0:
        y_proj, x_proj = proj(lat_approx.ravel(), lon_approx.ravel(), from_epsg=4326, to_epsg=epsg)
        return np.asarray(y_proj).ravel().astype(np.float32), np.asarray(x_proj).ravel().astype(np.float32), outline

    # Refine with DEM chunk
    lon, lat, _ = satellite_rat2llt(
        chunk_azi, chunk_rng, orbit_time, orbit_pos, orbit_vel,
        clock_start, prf, near_range, rng_samp_rate, earth_radius,
        dem=dem_chunk, max_iter=10, tol=0.5, n_chunks=1, lookdir=lookdir
    )
    y_proj, x_proj = proj(lat.ravel(), lon.ravel(), from_epsg=4326, to_epsg=epsg)
    return np.asarray(y_proj).ravel().astype(np.float32), np.asarray(x_proj).ravel().astype(np.float32), outline


def _precise_outline(edge_azi, edge_rng, lon0, lat0, dem_path, orbit_time, orbit_pos, orbit_vel, clock_start, prf,
                     near_range, rng_samp_rate, earth_radius, epsg, lookdir, datum):
    """The precise outline of one SLC edge, for the footprint of the transform (_precise_footprint): every edge pixel
    solved by satellite_rat2llt to 0.01 m in slant range on the WGS84 heights the tile workers use (DEM + the geoid of
    its vertical datum, datum), from its ellipsoid point (lon0, lat0). The DEM window is widened until every solution
    lies inside it: terrain moves a pixel off its ellipsoid point by about h / tan(incidence).
    A pixel without a solution on the DEM (its slant-range residual stays above 2 m: outside the DEM, or no
    convergence) keeps its ellipsoid point and is also solved at the lowest and at the highest DEM height of the
    window: its ground point lies between the two (_precise_footprint takes the outer one).

    Returns
    -------
    y, x : arrays (n,)
        The pixels' positions in the output CRS (float64).
    unsolved : array of int
        The indices of the pixels without a solution on the DEM.
    bounds : tuple (y_low, x_low, y_high, x_high) of arrays (len(unsolved),), or None
        Their positions at the lowest and the highest DEM height of the window (None: no DEM height there).
    """
    import numpy as np
    import xarray as xr
    from shapely.geometry import box

    lon0 = np.asarray(lon0, dtype=np.float64).ravel()
    lat0 = np.asarray(lat0, dtype=np.float64).ravel()
    w = [float(np.nanmin(lon0)) - 0.05, float(np.nanmin(lat0)) - 0.05,
         float(np.nanmax(lon0)) + 0.05, float(np.nanmax(lat0)) + 0.05]
    lon, lat, h = lon0, lat0, np.full(lon0.shape, np.nan)
    dem = None
    for _ in range(6):
        dem = get_dem_wgs84ellipsoid(dem_path, box(*w), buffer_degrees=0.0, datum=datum)
        if dem is None or dem.size == 0:
            break
        lon, lat, h = satellite_rat2llt(edge_azi, edge_rng, orbit_time, orbit_pos, orbit_vel, clock_start, prf,
                                        near_range, rng_samp_rate, earth_radius, dem=dem, max_iter=50, tol=0.01,
                                        n_chunks=1, lookdir=lookdir)
        dlon, dlat = dem.lon.values, dem.lat.values
        inside = ((lon > dlon.min() + 0.01) & (lon < dlon.max() - 0.01)
                  & (lat > dlat.min() + 0.01) & (lat < dlat.max() - 0.01))
        if inside.all():
            break
        # the window widened to the solutions: unchanged where the DEM file has no more, and then the loop ends
        wider = [min(w[0], float(np.nanmin(lon)) - 0.05), min(w[1], float(np.nanmin(lat)) - 0.05),
                 max(w[2], float(np.nanmax(lon)) + 0.05), max(w[3], float(np.nanmax(lat)) + 0.05)]
        if wider == w:
            break
        w = wider
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    no_dem = ~np.isfinite(h) | ~np.isfinite(lon) | ~np.isfinite(lat)
    lon = np.where(no_dem, lon0, lon)
    lat = np.where(no_dem, lat0, lat)
    h = np.where(no_dem, 0.0, h)

    # the slant-range residual of each solution, on the orbit of satellite_rat2llt
    t = clock_start + np.asarray(edge_azi, dtype=np.float64) / prf
    S = np.stack([_hermite_interp(orbit_time, orbit_pos[:, k], orbit_vel[:, k], t, nval=6) for k in range(3)], axis=1)
    P = np.stack(_geodetic_to_ecef(lon, lat, h), axis=1)
    rho = near_range + np.asarray(edge_rng, dtype=np.float64) * 299792458.0 / (2.0 * rng_samp_rate)
    resid = np.sqrt(np.sum((P - S) ** 2, axis=1)) - rho
    del S, P
    unsolved = np.nonzero(no_dem | ~(np.abs(resid) <= 2.0))[0]

    y, x = proj(lat, lon, from_epsg=4326, to_epsg=epsg)
    y = np.asarray(y, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()

    bounds = None
    heights = dem.values[np.isfinite(dem.values)] if dem is not None and dem.size else np.empty(0)
    if unsolved.size and heights.size:
        # a constant height surface over the DEM window and 1 degree around it
        glat = np.linspace(float(dem.lat.values.min()) - 1.0, float(dem.lat.values.max()) + 1.0, 64)
        glon = np.linspace(float(dem.lon.values.min()) - 1.0, float(dem.lon.values.max()) + 1.0, 64)
        bounds = ()
        for height in (float(heights.min()), float(heights.max())):
            surface = xr.DataArray(np.full((64, 64), height, dtype=np.float32), coords={'lat': glat, 'lon': glon},
                                   dims=['lat', 'lon'])
            lon_h, lat_h, _ = satellite_rat2llt(np.asarray(edge_azi)[unsolved], np.asarray(edge_rng)[unsolved],
                                                orbit_time, orbit_pos, orbit_vel, clock_start, prf, near_range,
                                                rng_samp_rate, earth_radius, dem=surface, max_iter=50, tol=0.01,
                                                n_chunks=1, lookdir=lookdir)
            y_h, x_h = proj(lat_h, lon_h, from_epsg=4326, to_epsg=epsg)
            bounds += (np.asarray(y_h, dtype=np.float64).ravel(), np.asarray(x_h, dtype=np.float64).ravel())
    return y, x, unsolved, bounds


def _precise_footprint(outlines, dy, dx):
    """The SLC footprint of the transform tile workers (WKB), from the precise outlines of the four SLC edges
    (_precise_outline; first line, last line, first bin, last bin, as compute_conversion_chunked orders them): the
    ring around the SLC (first line, last bin, last line and first bin reversed), not a convex hull, made valid (a
    layover fold becomes a part of its own) and buffered by one output cell for the pixel-centre numerics. An edge
    pixel without a solution on the DEM takes its position at the lowest or the highest DEM height, whichever is
    farther from the centre of the solved outline (its ground point lies between the two), with a warning."""
    import numpy as np
    import shapely

    solved = [np.setdiff1d(np.arange(o[0].size), o[2]) for o in outlines]
    centre_y = np.concatenate([o[0][k] for o, k in zip(outlines, solved)])
    centre_x = np.concatenate([o[1][k] for o, k in zip(outlines, solved)])
    if not centre_y.size:
        centre_y = np.concatenate([o[0] for o in outlines])
        centre_x = np.concatenate([o[1] for o in outlines])
    cy, cx = float(np.nanmean(centre_y)), float(np.nanmean(centre_x))
    edges = []
    n_unsolved = 0
    for y, x, unsolved, bounds in outlines:
        y, x = y.copy(), x.copy()
        n_unsolved += unsolved.size
        if bounds is not None:
            y_low, x_low, y_high, x_high = bounds
            with np.errstate(invalid='ignore'):
                d_low = np.where(np.isfinite(y_low) & np.isfinite(x_low), np.hypot(x_low - cx, y_low - cy), -np.inf)
                d_high = np.where(np.isfinite(y_high) & np.isfinite(x_high), np.hypot(x_high - cx, y_high - cy),
                                  -np.inf)
            high = d_high >= d_low
            y_out, x_out = np.where(high, y_high, y_low), np.where(high, x_high, x_low)
            ok = np.isfinite(y_out) & np.isfinite(x_out)
            y[unsolved[ok]], x[unsolved[ok]] = y_out[ok], x_out[ok]
        edges.append((y, x))
    n_total = sum(e[0].size for e in edges)
    if n_unsolved:
        print(f'WARNING: {n_unsolved} of {n_total} SLC outline pixels have no DEM solution (outside the DEM or not '
              f'converged): the transform footprint is bounded by the DEM height range there.')
    (y0, x0), (y1, x1), (y2, x2), (y3, x3) = edges
    ring_y = np.concatenate([y0, y3, y1[::-1], y2[::-1]])
    ring_x = np.concatenate([x0, x3, x1[::-1], x2[::-1]])
    ok = np.isfinite(ring_y) & np.isfinite(ring_x)
    polygon = shapely.Polygon(np.column_stack([ring_x[ok], ring_y[ok]]))
    if not polygon.is_valid:
        polygon = shapely.make_valid(polygon)
    return shapely.to_wkb(polygon.buffer(max(dy, dx)))


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


def orbit_defect(orbit_df, start=None, stop=None):
    """
    The defect that makes orbit state vectors unusable, or None: the check of every orbit the processing reads
    (Sentinel-1 EOF files, the orbit inside a NISAR scene).

    Parameters
    ----------
    orbit_df : pd.DataFrame
        State vectors in time order: 'clock' (seconds) and the ECEF px, py, pz, vx, vy, vz.
    start, stop : float, optional
        The time the state vectors must cover, on the 'clock' of orbit_df.

    Returns
    -------
    str or None
        The defect, worded for the error messages: 'does not cover' (no state vector at or before start, or none
        at or after stop), 'has a gap in the state vectors for' (a step between state vectors above 1.5 times
        their median step), 'has invalid (not finite) state vectors for'; None when they can be used.
    """
    clock = orbit_df['clock'].to_numpy(dtype=np.float64)
    if (len(clock) == 0 or (start is not None and clock.min() > start)
            or (stop is not None and clock.max() < stop)):
        return 'does not cover'
    steps = np.diff(clock)
    if len(steps) and steps.max() > 1.5 * np.median(steps):
        return 'has a gap in the state vectors for'
    if not np.isfinite(orbit_df[['px', 'py', 'pz', 'vx', 'vy', 'vz']].to_numpy()).all():
        return 'has invalid (not finite) state vectors for'
    return None


def orbit_seconds(orbit_df, clock_start):
    """
    Orbit state-vector times in seconds from 00:00 UTC of the scene day: the clock of the scene times
    (clock_start % 1.0) * 86400.

    'isec' is the second of each vector's own day, so it wraps to 0 at midnight, and the orbit window of a
    scene near 00:00 UTC (1400 s either side) holds vectors of both days. Here t = isec + 86400 * (vector day
    - scene day), the vector days taken as dates from (iy, id) so that Jan 1 is crossed too. GMTSAR keeps the
    day the same way (86400 * id + sec, SAT_llt2rat_sub.c), on a count from Jan 1; these values stay small.
    When every vector is on the scene day the offsets are zero and 'isec' itself is returned, bit for bit.

    Parameters
    ----------
    orbit_df : pd.DataFrame
        State vectors with iy (year), id (0-based day of year) and isec (seconds of that day).
    clock_start : float
        The scene day as PRM clock_start counts it: the 0-based day of the year, its fraction ignored. It has
        no year, so the year is the one that puts that day next to the vectors (they are minutes from the scene,
        years are 365 days apart); a day counted past Dec 31 is read the same way.

    Returns
    -------
    np.ndarray
        float64 orbit times, increasing through midnight.
    """
    isec = orbit_df['isec'].values
    iy = np.asarray(orbit_df['iy'].values, dtype=np.int64)
    # dates as days since 1970-01-01: Jan 1 of the year plus the 0-based day
    days = ((iy - 1970).astype('datetime64[Y]').astype('datetime64[D]').astype(np.int64)
            + np.asarray(orbit_df['id'].values, dtype=np.int64))
    mid = len(days) // 2
    jan1 = (iy[mid] - 1970 + np.arange(-1, 2)).astype('datetime64[Y]').astype('datetime64[D]').astype(np.int64)
    candidates = jan1 + int(np.floor(clock_start))
    offset = days - candidates[np.argmin(np.abs(candidates - days[mid]))]
    if not offset.any():
        return isec
    return isec + 86400.0 * offset


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


def geocentric_radius(lat_rad):
    """Geocentric radius for WGS84 ellipsoid at given geodetic latitude (radians).

    Parameters
    ----------
    lat_rad : float or array
        Geodetic latitude in radians.

    Returns
    -------
    float or array
        Geocentric radius in meters.
    """
    a, b = 6378137.0, 6356752.31424518
    cos, sin = np.cos(lat_rad), np.sin(lat_rad)
    return np.sqrt(((a**2 * cos)**2 + (b**2 * sin)**2) / ((a * cos)**2 + (b * sin)**2))


def reference_surface_topo(prm, topo, height, nodes=(33, 65), iterations=4, grid=None, pixel_offset=(0.0, 0.0)):
    """Flat-earth topo for `remove_topo_phase=False`: the WGS84 ellipsoid at `height`, per radar pixel.

    Returns a DataArray on topo's (a, r) grid holding R_surface - earth_radius, where R_surface is the geocentric
    radius of the point at ellipsoidal height `height` seen at that pixel's zero-Doppler time and slant range, and
    earth_radius is the PRM scalar -- the same convention as the DEM topo of compute_transform_inverse(), so
    flat_earth_topo_phase() treats both modes alike.

    A single radius per azimuth line cannot represent this surface: across the swath the ellipsoid radius follows
    the latitude, and a line's near-range value misplaced the far range by tens of metres. The surface is smooth,
    so it is solved exactly (float64) on a coarse node grid and interpolated bilinearly to every pixel.

    grid : (a, r) coordinates of the whole radar grid when topo is a block of it (NISAR, block by block): the nodes
    span that grid, so the block holds the whole-grid values bit for bit. None, the default, is topo's own grid;
    only topo's coordinates are read, never its values.

    pixel_offset : (line, bin) minus (a, r): the GMTSAR pixel of radar coordinate (a, r), in the SAT_llt2rat
    convention (time clock_start + line / PRF, slant range near_range + bin * dr), as for flat_earth_topo_phase().
    (0, 0), the default, for NISAR, whose radar coordinates are that line and bin; (-0.5, 0.5) for S1.
    """
    import numpy as np
    import xarray as xr

    ra = 6378137.0
    e2 = 6.69437999014e-3
    rb = ra * np.sqrt(1.0 - e2)
    ep2 = (ra * ra - rb * rb) / (rb * rb)

    y = np.asarray(topo.a.values, dtype=np.float64)
    x = np.asarray(topo.r.values, dtype=np.float64)
    gy, gx = (y, x) if grid is None else (np.asarray(grid[0], dtype=np.float64), np.asarray(grid[1], dtype=np.float64))
    ya = np.linspace(gy[0], gy[-1], max(2, min(int(nodes[0]), len(gy))))
    xr_ = np.linspace(gx[0], gx[-1], max(2, min(int(nodes[1]), len(gx))))

    orbit_time = prm.orbit_df['clock'].values
    px, py, pz = (prm.orbit_df[k].values for k in ('px', 'py', 'pz'))
    vx, vy, vz = (prm.orbit_df[k].values for k in ('vx', 'vy', 'vz'))
    dt = orbit_time[1] - orbit_time[0]
    ax, ay, az = np.gradient(vx, dt), np.gradient(vy, dt), np.gradient(vz, dt)
    t = 86400.0 * prm.get('clock_start') + (ya + pixel_offset[0]) / prm.get('PRF')
    S = np.stack([_hermite_interp(orbit_time, p, v, t, nval=6) for p, v in ((px, vx), (py, vy), (pz, vz))], axis=-1)
    V = np.stack([_hermite_interp(orbit_time, v, a, t, nval=6) for v, a in ((vx, ax), (vy, ay), (vz, az))], axis=-1)

    # zero-Doppler frame per line: nadir projected on the plane normal to V, and the cross-track side of the look
    r_sat = np.linalg.norm(S, axis=1)
    vh = V / np.linalg.norm(V, axis=1)[:, None]
    uh = S / r_sat[:, None]
    nd = uh - np.sum(uh * vh, axis=1)[:, None] * vh
    k = np.linalg.norm(nd, axis=1)
    nd /= k[:, None]
    cr = np.cross(nd, vh)
    cr /= np.linalg.norm(cr, axis=1)[:, None]
    lookdir = prm.get('lookdir') if 'lookdir' in prm.df.index else 'R'
    # the same side rule as satellite_rat2llt (GMTSAR SAT_llt2rat)
    det = np.dot(np.cross(cr[0], vh[0]), S[0])
    if det * (-1 if str(lookdir).upper() == 'L' else 1) < 0:
        cr = -cr

    rho = prm.get('near_range') + (xr_ + pixel_offset[1]) * (299792458.0 / (2.0 * prm.get('rng_samp_rate')))
    Sg, ndg, crg = S[:, None, :], nd[:, None, :], cr[:, None, :]
    rs, kg, rhog = r_sat[:, None], k[:, None], rho[None, :]
    R = np.full((len(ya), len(xr_)), prm.get('earth_radius') + height, dtype=np.float64)
    for _ in range(max(1, int(iterations))):
        # |S + rho (-cos b nd + sin b cr)| = R, exactly: S.nd = r_sat k and S.cr = 0
        cosb = np.clip((rs * rs + rhog * rhog - R * R) / (2.0 * rhog * rs * kg), -1.0, 1.0)
        sinb = np.sqrt(1.0 - cosb * cosb)
        G = Sg + rhog[..., None] * (-cosb[..., None] * ndg + sinb[..., None] * crg)
        # geodetic latitude and longitude of the ground point (Bowring)
        p = np.hypot(G[..., 0], G[..., 1])
        th = np.arctan2(G[..., 2] * ra, p * rb)
        lat = np.arctan2(G[..., 2] + ep2 * rb * np.sin(th) ** 3, p - e2 * ra * np.cos(th) ** 3)
        lon = np.arctan2(G[..., 1], G[..., 0])
        # radius of the point at the ellipsoidal height, for the next intersection
        N = ra / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        R = np.sqrt(((N + height) * np.cos(lat)) ** 2 + ((N * (1.0 - e2) + height) * np.sin(lat)) ** 2)

    # bilinear to every pixel: rows first on the node columns, then columns in row blocks so the transients stay small
    Ry = np.stack([np.interp(y, ya, R[:, j]) for j in range(R.shape[1])], axis=1)
    ix = np.clip(np.searchsorted(xr_, x, side='right') - 1, 0, len(xr_) - 2)
    span = xr_[ix + 1] - xr_[ix]
    # a single range column makes both nodes one point: weight 0, not 0/0
    w = np.where(span > 0, (x - xr_[ix]) / np.where(span > 0, span, 1.0), 0.0)[None, :]
    out = np.empty((len(y), len(x)), dtype=np.float32)
    er = float(prm.get('earth_radius'))
    for r0 in range(0, len(y), 256):
        blk = Ry[r0:r0 + 256]
        out[r0:r0 + 256] = blk[:, ix] * (1.0 - w) + blk[:, ix + 1] * w - er
    return xr.DataArray(out, coords={'a': topo.a, 'r': topo.r}, dims=['a', 'r']).rename('topo')


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
                       nrows_rep: int = None, prf_rep: float = None, *, lookdir: str) -> dict:
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
        Must have columns: px, py, pz, vx, vy, vz, iy, id, isec
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
        Repeat image start time in days. If None, uses clock_start's time of day on the repeat orbit's day.
    num_valid_az_rep : int, optional
        Repeat image num_valid_az. If None, uses num_valid_az.
    num_patches_rep : int, optional
        Repeat image num_patches. If None, uses num_patches.
    nrows_rep : int, optional
        Repeat image nrows. If None, uses nrows.
    prf_rep : float, optional
        Repeat image PRF. If None, uses prf.
    lookdir : str
        Look side, 'R' (Sentinel-1) or 'L' (NISAR). Required: GMTSAR flips the horizontal baseline for 'L'.

    Returns
    -------
    dict
        Dictionary containing:
        - B_parallel: Baseline component along the mid-range line of sight, at the scene centre (meters)
        - B_perpendicular: Baseline component normal to that line of sight, positive up, at the scene centre (meters)
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
    ...     rng_samp_rate=prm_ref.get('rng_samp_rate'),
    ...     lookdir=prm_ref.get('lookdir')
    ... )
    >>> print(f"Perpendicular baseline: {baseline['B_perpendicular']:.1f} m")
    """
    lookdir = str(lookdir).strip().upper()
    if lookdir not in ('R', 'L'):
        raise ValueError(f"lookdir must be 'R' or 'L', got {lookdir!r}")

    # Default repeat parameters to reference if not provided. The repeat orbit is on its own date: keep the
    # reference's time of day on the repeat orbit's day (the day-less code did this implicitly)
    if clock_start_rep is None:
        _tod = clock_start % 1.0
        _k = int(np.argmin(np.abs(orbit_df2['isec'].values / 86400.0 - _tod)))
        clock_start_rep = float(orbit_df2['id'].values[_k]) + _tod
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

    # seconds from 00:00 UTC of each scene's own day, the clock of time_of_day_start(_rep)
    orbit_time1 = orbit_seconds(orbit_df1, clock_start)
    orbit_time2 = orbit_seconds(orbit_df2, clock_start_rep)

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
    # GMTSAR get_sign (SAT_baseline.c:533-534): a left-looking radar (NISAR) flips bh, so bh stays positive
    # toward the look side; sign_after_orb carries the flip to the centre and end positions too
    if lookdir == 'L':
        sign = -sign
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

    rlook = None  # mid-range look angle, set below when the geometry parameters are available
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
        # No earth-central angle here. GMTSAR (SAT_baseline.c:586-588) adds it to get the incidence angle, but
        # the baseline is resolved at the satellite, where the line of sight makes the LOOK angle with the
        # vertical. With it B_perpendicular = cos(g)*Bperp - sin(g)*Bpar: 0.91-0.96 x exact on Sentinel-1.
        # B_parallel and B_perpendicular are resolved at the scene centre below
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

    if rlook is not None:
        # At the scene CENTRE, not at GMTSAR's start: on a NISAR scene the start value is up to 1.7% further
        # from the exact B_perp than the centre value; on Sentinel-1 bursts the two differ by under 0.4%
        B_parallel = baseline_center * np.sin(rlook - alpha_center)
        B_perpendicular = baseline_center * np.cos(rlook - alpha_center)

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
        """Compute alpha variation, handling the ±180° angle wraparound and the
        ±90° vertical-baseline sign flip."""
        diff = abs(a1 - a2)
        # alpha is a circular angle in (-180°, 180°]: e.g. 179.99° and -180.00° are
        # 0.01° apart, not 359.99°. Take the shorter arc before any further checks.
        if diff > 180:
            diff = 360 - diff
        # For nearly vertical baselines, +90° and -90° are effectively the same
        # (both mean bh≈0, just different sign): a 180° gap is a benign vertical flip.
        near_vertical = (abs(abs(a1) - 90) < 15) and (abs(abs(a2) - 90) < 15)
        if near_vertical and diff > 90:
            diff = 180 - diff  # Wrapped difference for vertical baseline flip
        return diff

    # Only validate when the baseline is long enough for alpha to be determined.
    # For self-baseline (baseline ≈ 0), alpha is numerically undefined (arctan2 of tiny values),
    # and a short baseline is barely better: the baseline vector moves well under a metre along
    # a burst whatever its length, so the angle it subtends grows as 1/baseline. Measured on
    # Sentinel-1, alpha varies by 5.37° over a 7.5m baseline and by 0.03° over a 400m one, both
    # being the same movement of the vector, and it stays under 1.2° for every baseline above
    # 25m. GMTSAR itself validates none of this and reads bperp from alpha_start alone.
    baseline_mean = (baseline_start + baseline_center + baseline_end) / 3
    if baseline_mean > 25.0:
        # Alpha should vary smoothly along the scene (< 5° variation)
        alpha_var_sc = alpha_variation(alpha_start_deg, alpha_center_deg)
        alpha_var_ce = alpha_variation(alpha_center_deg, alpha_end_deg)
        assert alpha_var_sc < 5.0, (
            f"Alpha varies too much between start and center: {alpha_var_sc:.2f}° "
            f"(start={alpha_start_deg:.2f}°, center={alpha_center_deg:.2f}°, "
            f"baseline={baseline_mean:.2f}m). "
            f"This indicates a sign computation bug."
        )
        assert alpha_var_ce < 5.0, (
            f"Alpha varies too much between center and end: {alpha_var_ce:.2f}° "
            f"(center={alpha_center_deg:.2f}°, end={alpha_end_deg:.2f}°, "
            f"baseline={baseline_mean:.2f}m). "
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


def _doppler_sample_idx(n_azi):
    """The orbit lines of the coarse zero-Doppler search (satellite_llt2rat and the transform tile worker): every
    n_azi // 20 lines from the first, and the last line of the window. A target's zero Doppler is bracketed by the
    two samples between which its Doppler changes sign; the samples must reach the end of the window, or a target
    after the last sample has no bracket (up to n_azi // 20 - 1 lines, inside the image for windows of 2000 lines
    and more)."""
    step = max(1, n_azi // 20)
    return np.append(np.arange(0, n_azi - 1, step), n_azi - 1)


def _satellite_llt2rat_chunk_worker(args):
    """Worker function for parallel satellite_llt2rat chunk processing.

    Must be at module level for joblib pickling.
    """
    (chunk_xp, chunk_yp, chunk_zp, orb_x, orb_y, orb_z, orb_vx, orb_vy, orb_vz, npad, n_azi) = args

    n_chunk = len(chunk_xp)

    # Compute Doppler at a few sample points to bracket zero
    sample_idx = _doppler_sample_idx(n_azi)

    # Compute Doppler at sample points: (T - S) · V
    doppler_samples = np.zeros((n_chunk, len(sample_idx)), dtype=np.float32)
    for j, idx in enumerate(sample_idx):
        dx = chunk_xp - orb_x[idx]
        dy = chunk_yp - orb_y[idx]
        dz = chunk_zp - orb_z[idx]
        doppler_samples[:, j] = dx * orb_vx[idx] + dy * orb_vy[idx] + dz * orb_vz[idx]

    # Find sign change (zero crossing) for each target
    sign_change = doppler_samples[:, :-1] * doppler_samples[:, 1:] < 0
    first_crossing = np.argmax(sign_change, axis=1)
    # A target with no crossing has its zero-Doppler time outside the window (lines -npad to nrows + npad): it has
    # no radar position there, NaN below (bracket 0 until then). Held at the nearest sample, it took a false one
    no_crossing = ~np.any(sign_change, axis=1)
    first_crossing[no_crossing] = 0

    # Get bracket indices in original azi_times
    bracket_lo = sample_idx[first_crossing]
    bracket_hi = np.minimum(sample_idx[np.minimum(first_crossing + 1, len(sample_idx) - 1)], n_azi - 1)

    # Refine within bracket using linear interpolation on Doppler
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
    chunk_azimuth_pix = azi_idx_float - npad  # subtract padding offset

    # Compute slant range at zero-Doppler time
    azi_idx_int = np.floor(azi_idx_float).astype(np.int32)
    azi_idx_int = np.clip(azi_idx_int, 0, n_azi - 2)
    azi_frac = azi_idx_float - azi_idx_int

    sat_x = orb_x[azi_idx_int] * (1 - azi_frac) + orb_x[azi_idx_int + 1] * azi_frac
    sat_y = orb_y[azi_idx_int] * (1 - azi_frac) + orb_y[azi_idx_int + 1] * azi_frac
    sat_z = orb_z[azi_idx_int] * (1 - azi_frac) + orb_z[azi_idx_int + 1] * azi_frac

    chunk_range_m = np.sqrt((chunk_xp - sat_x)**2 + (chunk_yp - sat_y)**2 + (chunk_zp - sat_z)**2)
    chunk_azimuth_pix[no_crossing] = np.nan
    chunk_range_m[no_crossing] = np.nan

    return chunk_azimuth_pix, chunk_range_m


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
                      n_jobs: int | None = None,
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
    n_jobs : int or None
        Number of parallel workers for the point chunks. None or -1 (default): all cores. Inside a worker of an
        outer pool the caller passes its share of the outer n_jobs.

    Returns
    -------
    np.ndarray
        Array of shape (N, 5) with columns [range_pix, azimuth_pix, range_m, azimuth_time, elevation]. A point
        whose zero-Doppler time is outside the searched window (100 lines before the first line to 100 lines after
        the last) has NaN range and azimuth columns.
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

    # Convert geodetic to ECEF per chunk (no full-length ECEF arrays, no copies of them), and the chunk results are
    # streamed into the output instead of concatenated
    e2 = (ra**2 - rc**2) / ra**2

    def _ecef(i, end):
        lon_rad = np.radians(lon[i:end])
        lat_rad = np.radians(lat[i:end])
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        N = ra / np.sqrt(1 - e2 * sin_lat**2)
        el = elevation[i:end]
        xp = (N + el) * cos_lat * cos_lon
        yp = (N + el) * cos_lat * sin_lon
        zp = (N * (1 - e2) + el) * sin_lat
        return xp, yp, zp

    n_azi = len(azi_times)

    import os
    from joblib import Parallel, delayed

    if n_jobs is None or n_jobs == -1:
        n_jobs = os.cpu_count()
    # every point is solved on its own, so the chunking does not change a result: one chunk per worker at least
    chunk_size = max(65536, min(1000000, -(-n_points // n_jobs)))
    starts = list(range(0, n_points, chunk_size))
    tasks = (delayed(_satellite_llt2rat_chunk_worker)(
                 _ecef(i, min(i + chunk_size, n_points)) + (orb_x, orb_y, orb_z, orb_vx, orb_vy, orb_vz, npad, n_azi))
             for i in starts)
    range_pixel_size = SOL / (2.0 * rng_samp_rate)

    # Result array: [range_pix, azimuth_pix, range_m, azimuth_time, elevation]
    result = np.zeros((n_points, 5))
    for i, (azimuth_pix, range_m) in zip(starts, Parallel(n_jobs=n_jobs, return_as='generator')(tasks)):
        end = i + len(azimuth_pix)
        azimuth_time = t1 + azimuth_pix / prf
        range_pix = (range_m - near_range) / range_pixel_size
        range_pix = range_pix - (rshift + sub_int_r) + chirp_ext
        azimuth_pix = azimuth_pix - (ashift + sub_int_a)
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
        result[i:end, 0] = range_pix
        result[i:end, 1] = azimuth_pix
        result[i:end, 2] = range_m
        result[i:end, 3] = azimuth_time
        result[i:end, 4] = elevation[i:end]

    return result


def offset_valid_mask(offset_dat, rmax, amax):
    """The points of an alignment offset table [r_ref, dr, a_ref, da, SNR] that PRM.fitoffset() takes: the reference
    position inside the radar extent (rmax bins, amax lines) and a finite offset. satellite_llt2rat() gives NaN for
    a point outside its window, and a point one date sees inside and the other outside has a NaN offset."""
    return ((offset_dat[:, 0] > 0) & (offset_dat[:, 0] < rmax) &
            (offset_dat[:, 2] > 0) & (offset_dat[:, 2] < amax) &
            np.isfinite(offset_dat[:, 1]) & np.isfinite(offset_dat[:, 3]))


def _scaled_int32(da, scale, fill_value, rows=256):
    """The int32 encoding of save_transform(): round(scale * da), fill_value where that is not finite. The same
    xarray expressions are evaluated on blocks of rows into one int32 array, so only one block's float temporaries
    exist; every element is encoded on its own, so the array is the whole-array result."""
    out = None
    for r0 in range(0, da.shape[0], rows):
        scaled = (scale * da[r0:r0 + rows]).round()
        finite_mask = np.isfinite(scaled)
        int_blk = scaled.fillna(0).astype(np.int32).where(finite_mask, fill_value)
        if out is None:
            out = np.empty(da.shape, dtype=int_blk.dtype)
        out[r0:r0 + rows] = int_blk.values
        del scaled, finite_mask, int_blk
    return out


def _scaled_range(da, scale, rows=256):
    """The minimum and maximum of round(scale * da), NaN skipped, in blocks of rows. They are the whole-array
    minimum and maximum; only the sign of a zero can depend on the reduction order, so a zero (or an all-NaN or
    infinite result) is taken from the whole-array expression instead."""
    vmin = vmax = np.nan
    for r0 in range(0, da.shape[0], rows):
        v = (scale * da[r0:r0 + rows]).round().values
        vmin = np.fmin(vmin, np.fmin.reduce(v, axis=None))
        vmax = np.fmax(vmax, np.fmax.reduce(v, axis=None))
        del v
    if not (np.isfinite(vmin) and vmin != 0 and np.isfinite(vmax) and vmax != 0):
        scaled = (scale * da).round()
        vmin = scaled.min(skipna=True).values
        vmax = scaled.max(skipna=True).values
        del scaled
    return vmin, vmax


def save_transform(transform, outdir, scale_factor=2.0):
    """Save transform dataset to zarr with int32 encoding.

    The store is written in two steps, so no full-size int32 copy of a variable exists: xarray writes the metadata,
    the attributes and the coordinates from placeholders equal to zarr's fill value (zarr stores no such chunk),
    then zarr writes every chunk of every variable, encoded on its own. The files are those of one to_zarr() call
    with the whole int32 arrays.

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
    import zarr
    import os

    fill_value = np.iinfo(np.int32).max
    trans_int = xr.Dataset(attrs=transform.attrs)

    def _placeholder(da, scale):
        # the dtype, attributes and encoding of the int32 expression (the source's attributes, no encoding), from
        # one element; zeros, the zarr fill value of the array xarray creates, with no memory behind them
        one = da.isel({dim: slice(0, 1) for dim in da.dims})
        tmpl = (scale * one).round()
        tmpl = tmpl.fillna(0).astype(np.int32).where(np.isfinite(tmpl), fill_value)
        res = xr.DataArray(np.broadcast_to(tmpl.dtype.type(0), da.shape), coords=da.coords, dims=da.dims,
                           attrs=dict(tmpl.attrs))
        res.encoding = dict(tmpl.encoding)
        return res

    scales = {}
    # Scale coordinate variables (rng, azi, ele)
    for varname in ['rng', 'azi', 'ele']:
        vmin, vmax = _scaled_range(transform[varname], scale_factor)
        trans_int[varname] = _placeholder(transform[varname], scale_factor)
        scales[varname] = scale_factor
        trans_int[varname].attrs['scale_factor'] = 1/scale_factor
        trans_int[varname].attrs['add_offset'] = 0
        # CF actual_range, in the units a reader gets back: from the ROUNDED
        # values, so it is what decoding reconstructs and never the _FillValue
        # sentinel. It spares every later reader a pass over the raster.
        trans_int[varname].attrs['actual_range'] = [
            float(vmin) / scale_factor,
            float(vmax) / scale_factor]
        # the fill value as an attribute: xarray writes it into the store exactly as an encoding _FillValue (the same
        # bytes), and the placeholder is not filled
        trans_int[varname].attrs['_FillValue'] = fill_value

    # Scale look vector components (unit vectors, range -1 to 1)
    # Use higher scale factor for precision (1e6 gives ~1e-6 precision)
    look_scale = 1e6
    for varname in ['look_E', 'look_N', 'look_U']:
        if varname in transform:
            vmin, vmax = _scaled_range(transform[varname], look_scale)
            trans_int[varname] = _placeholder(transform[varname], look_scale)
            scales[varname] = look_scale
            trans_int[varname].attrs['scale_factor'] = 1/look_scale
            trans_int[varname].attrs['add_offset'] = 0
            trans_int[varname].attrs['actual_range'] = [
                float(vmin) / look_scale,
                float(vmax) / look_scale]
            trans_int[varname].attrs['_FillValue'] = fill_value

    for _c in ('y', 'x'):
        if _c in trans_int.coords:
            _v = trans_int[_c].values
            trans_int[_c].attrs['actual_range'] = [float(np.nanmin(_v)),
                                                   float(np.nanmax(_v))]

    # Use 8192 chunk size for memory-efficient reading
    CHUNK_SIZE = 8192
    n_y, n_x = transform.y.size, transform.x.size
    chunk_y = min(CHUNK_SIZE, n_y)
    chunk_x = min(CHUNK_SIZE, n_x)

    all_vars = ['rng', 'azi', 'ele'] + [v for v in ['look_E', 'look_N', 'look_U'] if v in trans_int]
    # _FillValue is set as an attribute above
    encoding = {var: {'chunks': (chunk_y, chunk_x)} for var in all_vars}
    store = os.path.join(outdir, 'transform')
    trans_int.to_zarr(
        store=store,
        mode='w',
        zarr_format=3,
        consolidated=True,
        encoding=encoding
    )
    del trans_int

    # the data, chunk by chunk
    group = zarr.open_group(store, mode='r+', zarr_format=3)
    for varname, scale in scales.items():
        arr = group[varname]
        src = transform[varname]
        for y0 in range(0, n_y, chunk_y):
            for x0 in range(0, n_x, chunk_x):
                arr[y0:y0 + chunk_y, x0:x0 + chunk_x] = _scaled_int32(src[y0:y0 + chunk_y, x0:x0 + chunk_x],
                                                                      scale, fill_value)


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
    - Uses cv2.INTER_LANCZOS4 (8x8 Lanczos) for high-quality SLC resampling
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
            # the two remaps go straight into the complex output (no complex128 temporary)
            grid_proj = np.empty(inv_map_a.shape, dtype=data.dtype)
            grid_proj.real = cv2.remap(data_vals.real.astype(np.float32), inv_map_r, inv_map_a,
                                       interpolation=cv2.INTER_LANCZOS4,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
            grid_proj.imag = cv2.remap(data_vals.imag.astype(np.float32), inv_map_r, inv_map_a,
                                       interpolation=cv2.INTER_LANCZOS4,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
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
                grid_proj.real[:, x_slice] = re_chunk
                grid_proj.imag[:, x_slice] = im_chunk
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


def remap_source(data):
    """The source of remap_rows(): data in radar coordinates (dims a, r) as the float32 image cv2.remap reads,
    with its coordinates and dtype.

    A complex array is one 2-channel image of its real and imaginary parts: for a C-contiguous complex64 array a
    view, with no copy. cv2 interpolates every channel on its own with the same weights, so each channel is bit for
    bit the 1-channel remap of that part that remap_radar_to_geo() does.
    """
    vals = data.values
    if np.iscomplexobj(vals):
        vals = np.ascontiguousarray(vals, dtype=np.complex64)
        src = vals.view(np.float32).reshape(vals.shape + (2,))
    else:
        src = np.ascontiguousarray(vals, dtype=np.float32)
    return src, data.a.values, data.r.values, data.dtype


def remap_rows(source, azi_map, rng_map):
    """remap_radar_to_geo() for a block of output rows, as an array.

    source is remap_source(data); azi_map and rng_map are the radar coordinates of the block's output pixels. The
    inverse maps and the Lanczos-4 remap are those of remap_radar_to_geo(), and every output pixel depends only on
    its own map entry, so the block holds the rows of the whole-grid result, bit for bit.
    """
    import cv2

    src, coord_a, coord_r, dtype = source
    # Convert geographic pixel coordinates to radar array indices: the same expressions, 64 rows at a time (with
    # float64 coordinates they have float64 temporaries)
    n_y, n_x = np.shape(azi_map)
    inv_map_a = np.empty((n_y, n_x), dtype=np.float32)
    inv_map_r = np.empty((n_y, n_x), dtype=np.float32)
    for r0 in range(0, n_y, 64):
        inv_map_a[r0:r0 + 64] = (azi_map[r0:r0 + 64] - coord_a[0]) / (coord_a[1] - coord_a[0])
        inv_map_r[r0:r0 + 64] = (rng_map[r0:r0 + 64] - coord_r[0]) / (coord_r[1] - coord_r[0])

    border = (np.nan, np.nan, np.nan, np.nan) if src.ndim == 3 else np.nan
    OPENCV_MAX = 32766
    if n_x <= OPENCV_MAX:
        out = cv2.remap(src, inv_map_r, inv_map_a, interpolation=cv2.INTER_LANCZOS4,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    else:
        # work around OpenCV's 32k pixel limit, in the column chunks of remap_radar_to_geo()
        out = np.empty((n_y, n_x) + src.shape[2:], dtype=np.float32)
        for idx in np.array_split(np.arange(n_x), (n_x + OPENCV_MAX - 1) // OPENCV_MAX):
            x_slice = slice(idx[0], idx[-1] + 1)
            out[:, x_slice] = cv2.remap(src, inv_map_r[:, x_slice], inv_map_a[:, x_slice],
                                        interpolation=cv2.INTER_LANCZOS4,
                                        borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    del inv_map_a, inv_map_r
    if src.ndim == 3:
        # (rows, cols, 2) float32 as complex64 (rows, cols), no copy
        out = out.reshape(n_y, n_x, 2).view(np.complex64)[..., 0]
        if dtype != np.complex64:
            out = out.astype(dtype)
    return out


def pack_complex_int16(re, im, scale, fill_value):
    """Pack complex samples into int16 real and imaginary parts, int16 = round(value / scale).

    The int16 range clips the AMPLITUDE of a bright sample and never its phase: a sample with a part past
    fill_value - 1 (the largest magnitude stored, for both signs) is scaled down as a whole complex number until
    that part is exactly at it, which keeps the phase to the int16 rounding and clips as little amplitude as
    possible. Clipping each part on its own, as GMTSAR make_slc_nsr does, would bend the phase. Every other sample
    is rounded as it is. A sample with a non-finite part is stored as fill_value in both parts, and no stored
    sample reaches fill_value.

    Parameters
    ----------
    re, im : np.ndarray
        Real and imaginary parts, NaN where there is no data.
    scale : float
        The stored value is int16 * scale.
    fill_value : int
        The int16 no-data value, np.iinfo(np.int16).max.

    Returns
    -------
    tuple of np.ndarray
        The int16 real and imaginary parts.
    """
    cap = fill_value - 1
    with np.errstate(invalid='ignore', over='ignore'):
        re_q = np.divide(re, scale)
        im_q = np.divide(im, scale)
        # A sample saturates when a part rounds past the cap: rounding is half to even, so |part| > cap + 0.5
        limit = cap + 0.5
        saturated = re_q > limit
        saturated |= re_q < -limit
        saturated |= im_q > limit
        saturated |= im_q < -limit
        if saturated.any():
            # The saturated samples only: the larger part lands exactly on the cap, the phase is kept
            sat_re = re_q[saturated].astype(np.float64)
            sat_im = im_q[saturated].astype(np.float64)
            factor = cap / np.maximum(np.abs(sat_re), np.abs(sat_im))
            re_q[saturated] = np.round(sat_re * factor)
            im_q[saturated] = np.round(sat_im * factor)
            del sat_re, sat_im, factor
        del saturated
        np.round(re_q, out=re_q)
        np.round(im_q, out=im_q)
        nodata = ~np.isfinite(re_q)
        nodata |= ~np.isfinite(im_q)
        re_q[nodata] = fill_value
        im_q[nodata] = fill_value
        del nodata
    re_int16 = re_q.astype(np.int16)
    del re_q
    im_int16 = im_q.astype(np.int16)
    return re_int16, im_int16


def compute_merged_transform(transform, prm_rep, rows=None):
    """Compute merged transform by adding PRM bilinear offsets to ref transform.

    Parameters
    ----------
    transform : xr.Dataset
        Reference scene/burst transform with azi, rng variables.
    prm_rep : PRM
        Repeat scene/burst PRM with fitoffset parameters (rshift, stretch_r, etc.).
    rows : slice, optional
        Only these output rows (the same values as those rows of the whole result). Default is all rows.

    Returns
    -------
    azi_rep, rng_rep : np.ndarray
        Repeat scene/burst radar coordinates for each output pixel (float32, 2D).
    """
    azi_ref = transform.azi.values
    rng_ref = transform.rng.values
    if rows is not None:
        azi_ref = azi_ref[rows]
        rng_ref = rng_ref[rows]

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


def _tidal_enu_look(azi_corners, rng_corners, prm, tidal_dt):
    """Differential solid Earth tide (ref - rep) E, N, U [m] and the unit look vector (ground -> satellite) E, N, U
    at radar coordinates of prm's grid (1-D float64 arrays): the point of the WGS84 ellipsoid (height 0) seen there,
    the look from the orbit at its line. The corners of tidal_phase_radar() and the nodes of tidal_phase_nodes().
    """
    import numpy as np
    from .utils_tidal import solid_tide

    dt_ref, dt_rep = tidal_dt

    # --- (b) Geocode 4 corners -> lat/lon ---
    orbit_df = prm.orbit_df
    # seconds from 00:00 UTC of the scene day, the clock of clock_start below
    orbit_time = orbit_seconds(orbit_df, prm.get('clock_start'))
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

    # --- (c) Differential tidal E, N, U at 4 corners (ref - rep) ---
    tide_e_ref, tide_n_ref, tide_u_ref = solid_tide(lon_corners, lat_corners, dt_ref)
    tide_e_rep, tide_n_rep, tide_u_rep = solid_tide(lon_corners, lat_corners, dt_rep)
    tide_e = tide_e_ref - tide_e_rep
    tide_n = tide_n_ref - tide_n_rep
    tide_u = tide_u_ref - tide_u_rep

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
    return tide_e, tide_n, tide_u, look_E, look_N, look_U


def tidal_phase_nodes(prm, tidal_dt, grid, shape=(33, 33)):
    """The differential tidal phase of tidal_phase_radar() at fixed nodes of a whole radar grid, for its `nodes`.

    grid : (a, r) coordinates of the whole radar grid. shape[0] x shape[1] nodes span it evenly, first to last
    coordinate; at each one the tide and look vector of its own ground point give the phase exactly (float64).

    Returns (a_nodes, r_nodes, phase) with phase (len(a_nodes), len(r_nodes)) in radians.
    """
    import numpy as np

    gy, gx = np.asarray(grid[0], dtype=np.float64), np.asarray(grid[1], dtype=np.float64)
    ya = np.linspace(gy[0], gy[-1], max(2, min(int(shape[0]), len(gy))))
    xr_ = np.linspace(gx[0], gx[-1], max(2, min(int(shape[1]), len(gx))))
    A, R = np.meshgrid(ya, xr_, indexing='ij')
    tide_e, tide_n, tide_u, look_E, look_N, look_U = _tidal_enu_look(A.ravel(), R.ravel(), prm, tidal_dt)
    cnst = -4.0 * np.pi / prm.get('radar_wavelength')
    phase = cnst * (tide_e * look_E + tide_n * look_N + tide_u * look_U)
    return ya, xr_, phase.reshape(A.shape)


def tidal_phase_radar(topo, prm, tidal_dt, nodes=None):
    """Compute differential solid Earth tidal phase correction on radar grid.

    Computes tide at both ref and rep epochs, takes the difference
    (ref - rep), projects to LOS, and converts to phase.  This cancels
    the tidal signal in the interferogram when added to the rep burst's
    drho phase.

    Uses 2×2 radar-grid corners: computes tidal E,N,U and look vectors at the
    4 corner points, bilinearly interpolates onto the full topo grid.

    Satellite-agnostic: works for both S1 and NISAR.

    Parameters
    ----------
    topo : xr.DataArray
        Topographic elevation with radar coordinates (a, r).
    prm : PRM
        Reference scene PRM (has orbit_df, clock_start, PRF, etc.).
    tidal_dt : tuple(datetime, datetime)
        (dt_ref, dt_rep) acquisition UTC times for differential correction.
    nodes : tuple, optional
        (a_nodes, r_nodes, phase) of tidal_phase_nodes() when topo is a block of a larger radar grid (NISAR, block
        by block): the phase is interpolated bilinearly between these nodes of the whole grid, so every block holds
        the whole-grid values bit for bit, whatever the blocks (the corners of each block made the phase depend on
        the blocks, and so on the chunk size). None, the default, uses the corners of topo itself.

    Returns
    -------
    xr.DataArray
        Differential tidal phase correction in radians, same coords as topo.
    """
    import numpy as np
    import xarray as xr

    if nodes is not None:
        ya, xr_, P = nodes
        y = np.asarray(topo.a.values, dtype=np.float64)
        x = np.asarray(topo.r.values, dtype=np.float64)
        # bilinear, like reference_surface_topo(): rows first on the node columns, then columns in row blocks
        Py = np.stack([np.interp(y, ya, P[:, j]) for j in range(P.shape[1])], axis=1)
        ix = np.clip(np.searchsorted(xr_, x, side='right') - 1, 0, len(xr_) - 2)
        span = xr_[ix + 1] - xr_[ix]
        w = np.where(span > 0, (x - xr_[ix]) / np.where(span > 0, span, 1.0), 0.0)[None, :]
        # (in blocks of 64 lines: the float64 temporaries of a block stay small next to the chunk worker's arrays)
        tidal_phase = np.empty((len(y), len(x)), dtype=np.float32)
        for r0 in range(0, len(y), 64):
            blk = Py[r0:r0 + 64]
            tidal_phase[r0:r0 + 64] = blk[:, ix] * (1.0 - w) + blk[:, ix + 1] * w
        return xr.DataArray(tidal_phase, coords=topo.coords, dims=topo.dims).rename('tidal_phase')

    n_azi = len(topo.a)
    n_rng = len(topo.r)

    # --- (a) 4 radar corner coordinates ---
    azi_corners = np.array([topo.a.values[0], topo.a.values[0],
                            topo.a.values[-1], topo.a.values[-1]])
    rng_corners = np.array([topo.r.values[0], topo.r.values[-1],
                            topo.r.values[0], topo.r.values[-1]])

    # --- (b)-(d) tide and look vectors at the corners ---
    tide_e, tide_n, tide_u, look_E, look_N, look_U = _tidal_enu_look(azi_corners, rng_corners, prm, tidal_dt)

    # --- (e) Bilinear interpolation between the corners + LOS ---
    # The corner values belong to the first and last line and column. cv2.resize of a 2x2 image aligns pixel
    # CENTRES and puts them at the quarter points instead: flat outer quarters, twice the gradient between
    a0, a1 = float(topo.a.values[0]), float(topo.a.values[-1])
    r0, r1 = float(topo.r.values[0]), float(topo.r.values[-1])
    u_all = ((topo.a.values - a0) / (a1 - a0) if a1 != a0 else np.zeros(n_azi)).astype(np.float32)[:, None]
    v = ((topo.r.values - r0) / (r1 - r0) if r1 != r0 else np.zeros(n_rng)).astype(np.float32)[None, :]

    def _bilinear(arr, u):
        # corner order (a0, r0), (a0, r1), (a1, r0), (a1, r1)
        c = np.asarray(arr, dtype=np.float32)
        first = c[0] + (c[1] - c[0]) * v
        last = c[2] + (c[3] - c[2]) * v
        return first + (last - first) * u

    # --- (f) Convert to phase ---
    wavelength = prm.get('radar_wavelength')
    cnst = -4.0 * np.pi / wavelength
    # in blocks of radar lines, into the float32 output: every element is computed on its own, so no full-grid
    # temporaries of the six bilinear fields are needed
    tidal_phase = np.empty((n_azi, n_rng), dtype=np.float32)
    for i0 in range(0, n_azi, 256):
        u = u_all[i0:i0 + 256]
        los = _bilinear(tide_e, u) * _bilinear(look_E, u)
        los += _bilinear(tide_n, u) * _bilinear(look_N, u)
        los += _bilinear(tide_u, u) * _bilinear(look_U, u)
        tidal_phase[i0:i0 + 256] = (cnst * los).astype(np.float32)
        del los

    return xr.DataArray(tidal_phase, coords=topo.coords, dims=topo.dims).rename('tidal_phase')


def flat_earth_topo_phase(topo, prm_rep, prm_ref, earth_radius_azi=None, baseline_params=None, sc_height_params=None,
                          pixel_offset=(0.0, 0.0)):
    """Compute the combined earth curvature and topographic phase correction.

    Uses the full GMTSAR algorithm with time-varying baseline geometry.
    Satellite-agnostic: works for both S1 and NISAR.

    Parameters
    ----------
    topo : xr.DataArray or None
        Target radius minus the PRM scalar earth_radius, in radar coordinates (meters): the DEM topo of
        compute_transform_inverse(), or reference_surface_topo() for a flat-earth reference. If None, the WGS84
        ellipsoid at height 0.
    prm_rep : PRM
        Repeat scene PRM object.
    prm_ref : PRM
        Reference scene PRM object.
    earth_radius_azi : numpy.ndarray, optional
        Per-azimuth-line radius for a topo referenced to it instead. None, the default, uses the PRM scalar
        earth_radius, which is exact for the topo conventions above: a per-line radius added to a topo referenced
        to the scalar put the target at the wrong radius and jumped the phase at every burst seam.
    baseline_params : dict, optional
        Pre-computed baseline parameters.
    sc_height_params : dict, optional
        Pre-computed SC_height parameters.
    pixel_offset : tuple, optional
        (line, bin) minus (a, r): the GMTSAR pixel of radar coordinate (a, r), in the SAT_llt2rat convention (time
        clock_start + line / PRF, slant range near_range + bin * dr), at which GMTSAR's phasediff expressions are
        evaluated (time line * dt, range near_range + bin * dr). (0, 0), the default, for NISAR, whose radar
        coordinates are that line and bin (compute_conversion_chunked); (-0.5, 0.5) for S1, whose coordinates are
        line + 0.5 and bin - 0.5 (compute_transform_inverse; the S1 near_range is one bin before the first sample),
        so the phase refers to the pixel the SLC is sampled at, not half a bin short and half a line late.

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
        if earth_radius_azi is None:
            topo = reference_surface_topo(prm_ref, topo, 0.0, pixel_offset=pixel_offset)

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

    topo_raw = topo.values
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

    # the GMTSAR line and bin of the radar coordinates (pixel_offset)
    x_coords_f64 = x_coords.astype(np.float64) + pixel_offset[1]
    y_coords_f64 = y_coords.astype(np.float64) + pixel_offset[0]

    t_arr = y_coords_f64 * tspan / (ydim - 1)
    Bh = Bh0 + dBh * t_arr + ddBh * t_arr**2
    Bv = Bv0 + dBv * t_arr + ddBv * t_arr**2
    Bx = Bx0 + dBx * t_arr + ddBx * t_arr**2
    B = np.sqrt(Bh * Bh + Bv * Bv)
    alpha = np.arctan2(Bv, Bh)
    height = ht0 + dht * t_arr + ddht * t_arr**2

    if earth_radius_azi is None:
        # a float64 scalar keeps ret = er + topo (float32 topo) in float64
        er = np.float64(prm1.get('earth_radius'))
    else:
        er = np.asarray(earth_radius_azi, dtype=np.float64).reshape(-1, 1)
    # SC_height was computed relative to the PRM scalar earth_radius.
    # Adjust height so c = earth_radius + height = satellite distance stays constant
    # whatever radius ret = er + topo is referenced to.
    height_adj = height.reshape(-1, 1) + (prm1.get('earth_radius') - er)
    # the same float64 expressions in blocks of radar lines, into the float32 output (no full-grid temporaries)
    B2, alpha2, Bx2 = B.reshape(-1, 1), alpha.reshape(-1, 1), Bx.reshape(-1, 1)
    er_rows = np.ndim(er) == 2
    phase_shift = np.empty(topo_raw.shape, dtype=np.float32)
    for r0 in range(0, topo_raw.shape[0], 128):
        r1 = r0 + 128
        topo_vals = topo_raw[r0:r1].copy()
        np.copyto(topo_vals, 0, where=np.isnan(topo_vals))
        near_range = (prm1.get('near_range') + \
            x_coords_f64.reshape(1, -1) * (1 + prm1.get('stretch_r')) * drange) + \
            y_coords_f64[r0:r1].reshape(-1, 1) * prm1.get('a_stretch_r') * drange
        drho = calc_drho(near_range, topo_vals, er[r0:r1] if er_rows else er,
                         height_adj[r0:r1], B2[r0:r1], alpha2[r0:r1], Bx2[r0:r1])
        phase_shift[r0:r1] = (cnst * drho).astype(np.float32)
        del topo_vals, near_range, drho
    phase_shift[~np.isfinite(topo_raw)] = np.nan
    topo_phase = xr.DataArray(phase_shift, topo.coords).rename('phase')

    return topo_phase


# valid output pixels per block of compute_transform_inverse(): its float64 temporaries (the DEM interpolation
# mostly, about 130 bytes per pixel) and the per-block call overheads are set by it; 4M points were faster than
# the whole-grid steps and 2M points slower (measured on a 30 M-pixel burst)
_TRANSFORM_BLOCK_POINTS = 4_000_000
# points per satellite_llt2rat() call there: all its parallel chunks are in flight at once (inside a burst worker
# process they are threads of that process); 2M points took no more time than 4M and ~540 MB less (16 threads)
_LLT2RAT_CALL_POINTS = 2_000_000


class _CellScatter:
    """Values scattered to radar cells block after block, summed per cell as ONE np.bincount over all blocks in
    their order would sum them, bit for bit.

    np.bincount adds each cell's float64 weights one by one, in input order, and float64 addition is not
    associative, so per-block sums added together could differ in the last bit. Every block's points are instead
    kept in order, as (int32 cell offset, float32 weight), in parts of the cell index range; mean() then takes
    np.bincount part by part, where each cell sees exactly its weights in the input order. The kept points are
    8 bytes each and no full-grid float64 or int64 array exists.
    """

    def __init__(self, n_cells, part_cells=4_194_304):
        self.n_cells = n_cells
        self.n_parts = max(1, -(-n_cells // part_cells))
        self.part_cells = -(-n_cells // self.n_parts)
        self.part_dtype = np.uint8 if self.n_parts <= 256 else np.uint16
        self.idx = [[] for _ in range(self.n_parts)]
        self.w = [[] for _ in range(self.n_parts)]

    def add(self, idx, weights):
        """Append points (cell indices int64, float32 weights) in order."""
        if idx.size == 0:
            return
        part = (idx // self.part_cells).astype(self.part_dtype)
        if self.n_parts == 1:
            self.idx[0].append(idx.astype(np.int32))
            self.w[0].append(weights)
            return
        # stable: the input order within every part (radix sort of the small part numbers)
        order = np.argsort(part, kind='stable')
        ends = np.cumsum(np.bincount(part, minlength=self.n_parts))
        del part
        idx = idx[order]
        weights = weights[order]
        del order
        start = 0
        for p in range(self.n_parts):
            end = int(ends[p])
            if end > start:
                self.idx[p].append((idx[start:end] - p * self.part_cells).astype(np.int32))
                self.w[p].append(weights[start:end])
            start = end

    def mean(self, out, holes):
        """The per-cell float64 sum / max(count, 1) into float32 out, and count == 0 into bool holes (both flat, all
        cells), part by part: the whole-grid bincount, division and float32 conversion, element by element."""
        for p in range(self.n_parts):
            lo = p * self.part_cells
            n = min(self.part_cells, self.n_cells - lo)
            if n <= 0:
                break
            if self.idx[p]:
                rel = np.concatenate(self.idx[p])
                w = np.concatenate(self.w[p])
                self.idx[p] = self.w[p] = None
                ele_sum = np.bincount(rel, weights=w, minlength=n)
                ele_cnt = np.bincount(rel, minlength=n)
                del rel, w
            else:
                # no point in this part: the zero float64 sums and zero counts of the whole-grid bincount (np.bincount
                # of an empty input returns int64 even with weights, and the in-place division below cannot write it)
                self.idx[p] = self.w[p] = None
                ele_sum = np.zeros(n, dtype=np.float64)
                ele_cnt = np.zeros(n, dtype=np.intp)
            holes[lo:lo + n] = ele_cnt == 0
            np.maximum(ele_cnt, 1, out=ele_cnt)
            with np.errstate(invalid='ignore', divide='ignore'):
                np.divide(ele_sum, ele_cnt, out=ele_sum)
            del ele_cnt
            out[lo:lo + n] = ele_sum
            del ele_sum


def compute_transform_inverse(prm, dem,
                              scale_factor=2.0,
                              epsg=None,
                              resolution=(16.0, 4.0),
                              bbox=None,
                              n_chunks=8,
                              compute_topo=True,
                              n_jobs=None,
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
        If True (default), grid the DEM heights into the radar cells (the topo of remove_topo_phase=True).
        If False (flat-earth mode), skip that: the topo holds the radar grid coordinates only, its values NaN.
    n_jobs : int or None, optional
        Number of parallel workers of the inverse transform (satellite_llt2rat). None or -1 (default): all cores.
    debug : bool, optional
        If True, prints timing information. Default is False.

    Returns
    -------
    tuple
        (topo, transform) where topo is a DataArray on the radar grid (a, r), its values NaN and backed
        by no memory if compute_topo=False, and transform is a Dataset.
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

    # satellite_llt2rat uses the 'clock' column: seconds from 00:00 UTC of the scene day, the clock of
    # clock_start_days below
    orbit_df = orbit_df.copy()
    orbit_df['clock'] = orbit_seconds(orbit_df, prm.get('clock_start'))

    orbit_time = orbit_df['clock'].values
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

    # Apply bbox crop if specified (WGS84: [lon_min, lat_min, lon_max, lat_max])
    if bbox is not None:
        bbox_y, bbox_x = proj(
            np.array([bbox[1], bbox[3]]),
            np.array([bbox[0], bbox[2]]),
            from_epsg=4326, to_epsg=epsg
        )
        y_min = max(y_min, dy * np.floor(min(bbox_y) / dy))
        y_max = min(y_max, dy * np.ceil(max(bbox_y) / dy))
        x_min = max(x_min, dx * np.floor(min(bbox_x) / dx))
        x_max = min(x_max, dx * np.ceil(max(bbox_x) / dx))
        if debug:
            print(f'  bbox crop: y=[{y_min:.0f}, {y_max:.0f}], x=[{x_min:.0f}, {x_max:.0f}]')

    out_y = np.arange(y_min + dy/2, y_max, dy).astype(np.float32)
    out_x = np.arange(x_min + dx/2, x_max, dx).astype(np.float32)

    # Note: OpenCV cv2.remap has 32767 pixel limit per dimension.
    # Instead of cropping here, _geocode_standalone handles this via x-chunking.

    n_y, n_x = len(out_y), len(out_x)
    if n_y == 0 or n_x == 0:
        # S1.transform() skips the bursts whose footprint misses the bbox, so this one's footprint meets the bbox
        # while its output grid (the geocoded first and last lines and the margin) does not
        raise ValueError(f'ERROR: bbox {bbox} does not overlap the output grid of the burst, '
                         f'y=[{y_min:.0f}, {y_max:.0f}] x=[{x_min:.0f}, {x_max:.0f}] in EPSG:{epsg}.')

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

    # Fast path: if polygon covers entire grid, all pixels are valid (skip cv2 import)
    grid_box = shapely.box(out_x[0], out_y[0], out_x[-1], out_y[-1])
    if polygon.contains(grid_box):
        valid_mask = np.ones((n_y, n_x), dtype=bool)
    else:
        # Rasterize polygon to valid mask (cv2, no rasterio/GDAL to avoid fork deadlocks)
        import cv2
        poly_coords = np.array(polygon.exterior.coords)
        px = ((poly_coords[:, 0] - out_x[0]) / (out_x[-1] - out_x[0]) * (n_x - 1)).astype(np.int32)
        py = ((poly_coords[:, 1] - out_y[0]) / (out_y[-1] - out_y[0]) * (n_y - 1)).astype(np.int32)
        valid_mask = np.zeros((n_y, n_x), dtype=np.uint8)
        cv2.fillPoly(valid_mask, [np.column_stack([px, py])], 1)
        valid_mask = valid_mask.astype(bool)

    n_valid = valid_mask.sum()

    if _timings is not None:
        _timings['valid_mask'] = time.perf_counter() - t0
        print(f'  Valid mask: {_timings["valid_mask"]:.2f}s ({100*n_valid/(n_y*n_x):.1f}% valid)')

    # Steps 5-9 in blocks of output rows, end to end: for the valid output pixels of one block, the projection to
    # lon/lat, the DEM interpolation, the inverse transform, the elevations, the scatter into the output grids and
    # the points of the radar-grid topo. No array over all valid pixels and no float64 temporary of more than one
    # block is held: the live data are the three output grids, the valid mask and the kept topo points. Every output
    # pixel is computed on its own and the blocks follow the row-major order of the whole-grid boolean indexing, so
    # every value is the one of whole-grid steps (the topo sums too, see _CellScatter).
    t0 = time.perf_counter()
    inv_azi = np.full((n_y, n_x), np.nan, dtype=np.float32)
    inv_rng = np.full((n_y, n_x), np.nan, dtype=np.float32)
    inv_ele = np.full((n_y, n_x), np.nan, dtype=np.float32)
    # NOTE: look_E/N/U arrays commented out - incidence is computed from azi/rng in Batch.incidence()
    if compute_topo:
        # the elevation of every pixel, scattered to its nearest radar cell
        scatter = _CellScatter(n_azi * n_rng)
    row_counts = valid_mask.sum(axis=1)
    block_points = _TRANSFORM_BLOCK_POINTS
    r0 = 0
    while r0 < n_y:
        cum = np.cumsum(row_counts[r0:])
        r1 = min(n_y, r0 + max(1, int(np.searchsorted(cum, block_points, side='right'))))
        blk_mask = valid_mask[r0:r1]
        k = int(cum[r1 - r0 - 1])
        if k == 0:
            r0 = r1
            continue

        # Step 5: the block's valid output pixels to lon/lat, and their DEM elevation
        x_grid, y_grid = np.meshgrid(out_x, out_y[r0:r1])
        vy = y_grid[blk_mask]
        vx = x_grid[blk_mask]
        del x_grid, y_grid
        lat_b, lon_b = proj(vy, vx, from_epsg=epsg, to_epsg=4326)
        del vy, vx
        lat_b = lat_b.astype(np.float32)
        lon_b = lon_b.astype(np.float32)
        ele_b = dem.interp(
            lat=xr.DataArray(lat_b, dims='z'),
            lon=xr.DataArray(lon_b, dims='z'),
            method='linear'
        ).values.astype(np.float32)

        # Step 6: inverse transform, in calls of at most _LLT2RAT_CALL_POINTS points: satellite_llt2rat returns
        # float64 [N, 5] and its parallel chunks are all in flight at once (threads, inside a burst worker process)
        rng_b = np.empty(k, dtype=np.float32)
        azi_b = np.empty(k, dtype=np.float32)
        for j0 in range(0, k, _LLT2RAT_CALL_POINTS):
            j1 = min(k, j0 + _LLT2RAT_CALL_POINTS)
            result = satellite_llt2rat(
                lon=lon_b[j0:j1], lat=lat_b[j0:j1], elevation=ele_b[j0:j1],
                orbit_df=orbit_df, clock_start=clock_start_days, prf=prf,
                near_range=near_range, rng_samp_rate=rng_samp_rate,
                num_valid_az=num_lines, num_patches=1, nrows=num_lines,
                earth_radius=earth_radius, precise=1, fd1=0.0,
                n_jobs=n_jobs, debug=debug
            )
            # Convert satellite_llt2rat pixel indices to SLC 0.5-based coordinates
            # (first pixel center at 0.5) for remap_radar_to_geo: inv_map = (val - 0.5) / 1.0
            # Azimuth: 0-based (first line = 0) → add 0.5
            # Range: effectively 1-based (first sample = 1, due to GMTSAR near_range -= dr) → subtract 0.5
            rng_b[j0:j1] = result[:, 0].astype(np.float32) - 0.5
            azi_b[j0:j1] = result[:, 1].astype(np.float32) + 0.5
            del result
        # Filter out-of-bounds (valid pixel centers are [0.5, n-0.5])
        out_of_bounds = (azi_b < 0.5) | (azi_b > n_azi - 0.5) | (rng_b < 0.5) | (rng_b > n_rng - 0.5)
        azi_b[out_of_bounds] = np.nan
        rng_b[out_of_bounds] = np.nan
        del out_of_bounds

        # Step 7: ele_gmtsar of the block's valid output pixels
        # Trig functions
        lon_rad = np.float32(np.pi / 180) * lon_b
        lat_rad = np.float32(np.pi / 180) * lat_b
        del lon_b, lat_b
        sin_lat = np.sin(lat_rad, dtype=np.float32)
        cos_lat = np.cos(lat_rad, dtype=np.float32)
        sin_lon = np.sin(lon_rad, dtype=np.float32)
        cos_lon = np.cos(lon_rad, dtype=np.float32)
        del lon_rad, lat_rad

        # Ground point ECEF
        N = np.float32(ra) / np.sqrt(1 - np.float32(e2) * sin_lat**2)
        Nh = N + ele_b
        xp = Nh * cos_lat * cos_lon
        yp = Nh * cos_lat * sin_lon
        zp = (N * np.float32(1 - e2) + ele_b) * sin_lat
        del Nh, sin_lon, cos_lon, ele_b

        R_target = np.sqrt(xp**2 + yp**2 + zp**2, dtype=np.float32)
        del xp, yp, zp
        # Per-point local geocentric radius (consistent across all bursts)
        R_local = N * np.sqrt(cos_lat**2 + np.float32((1 - e2)**2) * sin_lat**2, dtype=np.float32)
        del N, sin_lat, cos_lat
        # NOTE: the look vector (GMTSAR-compatible look_E/N/U) is not computed - incidence is computed from azi/rng
        # in Batch.incidence()

        # Step 8: scatter to the output grids (the block's rows, in the row-major order of its mask)
        inv_azi[r0:r1][blk_mask] = azi_b
        inv_rng[r0:r1][blk_mask] = rng_b
        inv_ele[r0:r1][blk_mask] = R_target - R_local
        del R_local

        # Step 9 (scatter): the elevation of every pixel kept for its nearest radar cell
        if compute_topo:
            # For topo phase: constant earth_radius (cancels in calc_drho). Subtracted in float64 because
            # flat_earth_topo_phase() adds the same float64 earth_radius back; float32(er) is up to 0.25 m off
            ele_topo_b = (R_target.astype(np.float64) - np.float64(earth_radius)).astype(np.float32)
            # Convert to radar grid indices and round to nearest. The mask comes before the int64 cast: a NaN azi or
            # rng (out of the radar grid, or over a DEM gap) has no radar cell, and its cast is platform-defined (0 on
            # arm64, the radar cell (0, 0))
            azi_round = np.round(azi_b - azi_coords[0])
            rng_round = np.round(rng_b - rng_coords[0])
            m = (np.isfinite(azi_round) & np.isfinite(rng_round)
                 & (azi_round >= 0) & (azi_round < n_azi) & (rng_round >= 0) & (rng_round < n_rng))
            idx = azi_round[m].astype(np.int64) * n_rng + rng_round[m].astype(np.int64)
            del azi_round, rng_round
            scatter.add(idx, ele_topo_b[m])
            del ele_topo_b, m, idx
        del R_target, azi_b, rng_b
        r0 = r1
    del valid_mask, row_counts

    if _timings is not None:
        _timings['inverse_transform'] = time.perf_counter() - t0
        print(f'  Inverse transform, elevations and scatter ({n_valid:,} pixels): {_timings["inverse_transform"]:.2f}s')

    # Step 9: the topo on the radar grid, the mean elevation per cell (nearest-neighbour scatter), holes filled
    t0 = time.perf_counter()
    if compute_topo:
        # the float64 sum / count per cell as float32, and the cells without a pixel, part by part
        ele_gmtsar_full = np.empty((n_azi, n_rng), dtype=np.float32)
        holes = np.empty((n_azi, n_rng), dtype=bool)
        scatter.mean(ele_gmtsar_full.reshape(-1), holes.reshape(-1))
        del scatter

        # Fill holes with nearest valid elevation using distance transform (O(n) algorithm)
        if holes.any() and not holes.all():
            from scipy.ndimage import distance_transform_edt
            # the feature transform only, the distances were thrown away
            nearest_idx = distance_transform_edt(holes, return_distances=False, return_indices=True)
            # in blocks of radar lines: the nearest cell of a hole is never a hole, so no filled value is read back
            for a0 in range(0, n_azi, 256):
                h = holes[a0:a0 + 256]
                ele_gmtsar_full[a0:a0 + 256][h] = ele_gmtsar_full[nearest_idx[0, a0:a0 + 256][h],
                                                                  nearest_idx[1, a0:a0 + 256][h]]
                del h
            del nearest_idx
        del holes

        if _timings is not None:
            _timings['topo_scatter'] = time.perf_counter() - t0
            print(f'  Topo scatter+fill: {_timings["topo_scatter"]:.2f}s')
    else:
        # the radar grid alone (reference_surface_topo reads only the coordinates): a zero-stride NaN array
        ele_gmtsar_full = np.broadcast_to(np.float32(np.nan), (n_azi, n_rng))

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

    topo = xr.DataArray(ele_gmtsar_full, coords={'a': azi_coords, 'r': rng_coords}, dims=['a', 'r']).rename('topo')

    return topo, trans


def compute_conversion_chunked(prm, dem_path, geometry, outdir,
                               scale_factor=2.0,
                               epsg=None,
                               resolution=(16.0, 4.0),
                               bbox=None,
                               chunk=(8192, 8192),
                               compute_topo=True,
                               n_jobs=-1,
                               debug=False,
                               datum=None):
    """
    Compute transform and topo tile-by-tile, writing directly to zarr.

    Memory-efficient version that never builds full arrays in memory.
    Workers read DEM chunks directly from file - no full DEM in memory.
    Suitable for large NISAR data on limited RAM systems (e.g., 12GB Colab).

    Directory structure:
    - outdir/transform/   : Persistent transform zarr (azi, rng, ele) for geocoding: azi/rng are the 0-based
      pixel-centre line and bin of each output pixel, ele its WGS84 ellipsoidal height, rounded to int32 counts of
      1/scale_factor for the stack
    - outdir/conversion/transform/ : Temporary precise transform (float32, precise_transform_dir), which the
      processing reads (the topo below and the SLC chunks); the caller removes it when the dates are processed
    - outdir/conversion/topo/ : Topo zarr in radar coordinates for the flat-earth and topographic phase: the
      transform's DEM points gridded in the radar cells, as compute_transform_inverse() builds the S1 topo

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
    datum : str, optional
        The vertical datum of the DEM (Satellite.dem_datum()). None resolves it from dem_path here, once for
        all workers.
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

    # seconds from 00:00 UTC of the scene day, the clock of clock_start(_days) below; the tile workers read 'clock'
    orbit_df = orbit_df.copy()
    orbit_df['clock'] = orbit_seconds(orbit_df, prm.get('clock_start'))
    orbit_time = orbit_df['clock'].values
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

    # the vertical datum of the DEM, once for all workers (a fallback warns here, not in every worker)
    from insardev_toolkit import utils_geoid
    datum = utils_geoid.dem_datum(dem_path, datum)

    # 4 boundary edges: first_row, last_row, first_col, last_col
    worker_args = []
    # First row: azi=0, rng varies
    worker_args.append((
        np.full(n_rng, azi_coords[0], dtype=np.float32),
        rng_coords.astype(np.float32),
        dem_path, orbit_time, orbit_pos, orbit_vel, clock_start, prf,
        near_range, rng_samp_rate, earth_radius, epsg, lookdir, datum
    ))
    # Last row: azi=n_azi-1, rng varies
    worker_args.append((
        np.full(n_rng, azi_coords[-1], dtype=np.float32),
        rng_coords.astype(np.float32),
        dem_path, orbit_time, orbit_pos, orbit_vel, clock_start, prf,
        near_range, rng_samp_rate, earth_radius, epsg, lookdir, datum
    ))
    # First col: azi varies, rng=0
    worker_args.append((
        azi_coords.astype(np.float32),
        np.full(n_azi, rng_coords[0], dtype=np.float32),
        dem_path, orbit_time, orbit_pos, orbit_vel, clock_start, prf,
        near_range, rng_samp_rate, earth_radius, epsg, lookdir, datum
    ))
    # Last col: azi varies, rng=n_rng-1
    worker_args.append((
        azi_coords.astype(np.float32),
        np.full(n_azi, rng_coords[-1], dtype=np.float32),
        dem_path, orbit_time, orbit_pos, orbit_vel, clock_start, prf,
        near_range, rng_samp_rate, earth_radius, epsg, lookdir, datum
    ))
    n_bnd = 2 * n_rng + 2 * n_azi

    # Parallel execution - reuse workers (boundary tasks are small)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    _t0_bnd = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context('spawn')) as executor:
        results = list(executor.map(_process_boundary_worker, worker_args))
    if debug:
        print(f'PROFILE: boundary ProcessPoolExecutor ({len(worker_args)} tasks, {n_jobs} workers) {time.perf_counter() - _t0_bnd:.3f}s')
    del worker_args

    # Concatenate results
    bnd_y = np.concatenate([r[0] for r in results])
    bnd_x = np.concatenate([r[1] for r in results])
    # the precise outlines of the four edges, for the footprint of the tile workers
    outlines = [r[2] for r in results]
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

    # Apply bbox crop if specified (WGS84: [lon_min, lat_min, lon_max, lat_max])
    if bbox is not None:
        bbox_y, bbox_x = proj(
            np.array([bbox[1], bbox[3]]),
            np.array([bbox[0], bbox[2]]),
            from_epsg=4326, to_epsg=epsg
        )
        y_min = max(y_min, dy * np.floor(min(bbox_y) / dy))
        y_max = min(y_max, dy * np.ceil(max(bbox_y) / dy))
        x_min = max(x_min, dx * np.floor(min(bbox_x) / dx))
        x_max = min(x_max, dx * np.ceil(max(bbox_x) / dx))
        if debug:
            print(f'  bbox crop: y=[{y_min:.0f}, {y_max:.0f}], x=[{x_min:.0f}, {x_max:.0f}]')

    # Compute grid dimensions without creating full arrays (memory-efficient)
    n_y = int((y_max - y_min) / dy)
    n_x = int((x_max - x_min) / dx)

    if debug:
        print(f'Output grid: {n_y} x {n_x}, bounds from {n_bnd} boundary points in {time.perf_counter() - t0:.1f}s')

    # The SLC footprint for the tile workers, once for all tiles: the ring of the precise outlines
    fp_wkb = _precise_footprint(outlines, dy, dx)

    # Free boundary arrays - workers will determine tile validity internally
    del bnd_y, bnd_x, valid_bnd, outlines

    # Step 2: Pre-create zarr arrays (lazy - no memory allocation)
    fill_value = np.iinfo(np.int32).max
    chunk_y, chunk_x = chunk
    zarr_chunks = (min(chunk_y, n_y), min(chunk_x, n_x))
    # Topo chunks of 1024 x 1024 cells: the chunk workers read the radar box of each part of their output chunk,
    # which is far smaller than a processing tile and rarely aligned to it
    topo_chunk = 1024
    radar_chunks = (min(topo_chunk, n_azi), min(topo_chunk, n_rng))

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

    # The precise transform, which the processing reads (precise_transform_dir): the same variables, as computed.
    # A chunk per row batch of a tile when the tiles are whole batches, so each batch write is a whole chunk (no
    # read-modify-write); else the tile's chunks, as the stored transform (no two tiles share a chunk)
    precise_dir = precise_transform_dir(outdir)
    os.makedirs(precise_dir, exist_ok=True)
    precise_store = zarr.storage.LocalStore(precise_dir)
    precise_root = zarr.group(store=precise_store, zarr_format=3, overwrite=True)
    precise_chunks = (min(row_batch, n_y), zarr_chunks[1]) if chunk_y % row_batch == 0 else zarr_chunks
    for name in ('azi', 'rng', 'ele'):
        precise_root.create_array(name, shape=(n_y, n_x), chunks=precise_chunks, dtype=np.float32, fill_value=np.nan,
                                  overwrite=True, dimension_names=['y', 'x'])
    zarr.consolidate_metadata(precise_store)

    # Write coordinate arrays (compute directly, don't keep in memory)
    out_y_coords = (y_min + dy * (np.arange(n_y) + 0.5)).astype(np.float64)
    out_x_coords = (x_min + dx * (np.arange(n_x) + 0.5)).astype(np.float64)
    y_arr = trans_root.create_array('y', data=out_y_coords, chunks=(n_y,), overwrite=True,
                                     dimension_names=['y'])
    x_arr = trans_root.create_array('x', data=out_x_coords, chunks=(n_x,), overwrite=True,
                                     dimension_names=['x'])
    y_arr.attrs['_ARRAY_DIMENSIONS'] = ['y']
    for _a in (y_arr, x_arr):
        _f, _l = float(_a[0]), float(_a[-1])
        _a.attrs['actual_range'] = [min(_f, _l), max(_f, _l)]
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
                transform_dir, precise_dir, dem_path, epsg,
                (iy, jy, ix, jx),  # tile_bounds
                (y_min, dy, x_min, dx),  # grid_params - worker computes coords locally
                orbit_dict, clock_start_days, prf,
                near_range, rng_samp_rate, num_lines, earth_radius,
                n_azi, n_rng, ra, e2,
                scale_factor, fill_value, row_batch, lookdir, fp_wkb, datum
            ))

    if debug:
        print(f'Computing transform: {len(tile_args)} tiles (row_batch={row_batch})...')

    # Process tiles using subprocess pool with memory isolation
    # Each worker processes one tile then exits, releasing all memory
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    _t0_tiles = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context('spawn'),
                             max_tasks_per_child=1) as executor:
        tile_extents = list(executor.map(_process_tile_worker, tile_args))
    if debug:
        print(f'PROFILE: transform ProcessPoolExecutor ({len(tile_args)} tiles, {n_jobs} workers) {time.perf_counter() - _t0_tiles:.3f}s')

    del tile_args

    # how far each variable reaches, combined from the tiles that wrote it --
    # they carried it back, so nothing is re-read. A variable no tile could
    # fill stays NaN and gets no attribute rather than a made-up range.
    _ext = np.stack([e for e in tile_extents if e is not None]) \
        if any(e is not None for e in tile_extents) else np.full((1, 3, 2), np.nan)
    with np.errstate(invalid='ignore'):
        _lo = np.nanmin(_ext[:, :, 0], axis=0)
        _hi = np.nanmax(_ext[:, :, 1], axis=0)

    # Add transform metadata
    for _i, arr in enumerate([azi_arr, rng_arr, ele_arr]):
        arr.attrs['scale_factor'] = 1/scale_factor
        arr.attrs['add_offset'] = 0
        arr.attrs['_FillValue'] = int(fill_value)
        arr.attrs['_ARRAY_DIMENSIONS'] = ['y', 'x']
        if np.isfinite(_lo[_i]) and np.isfinite(_hi[_i]):
            arr.attrs['actual_range'] = [float(_lo[_i]), float(_hi[_i])]

    from pyproj import CRS
    trans_root.attrs['spatial_ref'] = CRS.from_epsg(epsg).to_wkt()
    zarr.consolidate_metadata(trans_store)

    if debug:
        print(f'Transform done: {time.perf_counter() - t0:.1f}s')

    # Step 5: Compute topo tile-by-tile in radar coordinates from the precise transform just written: its DEM points,
    # mapped to radar coordinates (azi, rng) with their radius minus earth_radius (from ele), gridded in the radar cells
    # like the S1 topo of compute_transform_inverse(). Process tiles in parallel with subprocess pool (like transform
    # tiles)
    if compute_topo:
        t0 = time.perf_counter()

        # Tiles of whole topo chunks (no two workers write one chunk), with a margin of cells for the hole fill
        margin = 16
        tile_a = max(radar_chunks[0], (chunk_y // radar_chunks[0]) * radar_chunks[0])
        tile_r = max(radar_chunks[1], (chunk_x // radar_chunks[1]) * radar_chunks[1])
        tiles = [(ia, min(ia + tile_a, n_azi), ir, min(ir + tile_r, n_rng))
                 for ia in range(0, n_azi, tile_a) for ir in range(0, n_rng, tile_r)]
        windows = _topo_tile_windows(precise_dir, tiles, margin, float(azi_coords[0]), float(rng_coords[0]))

        # Build tile arguments for parallel processing: each tile reads its window of the precise transform
        topo_tile_args = [(topo_dir, transform_dir, precise_dir, tile, window, margin,
                           float(azi_coords[0]), float(rng_coords[0]), n_azi, n_rng,
                           scale_factor, fill_value, epsg, earth_radius)
                          for tile, window in zip(tiles, windows) if window is not None]
        del tiles, windows

        if debug:
            print(f'Computing topo: {len(topo_tile_args)} tiles (margin={margin})...')

        # Process tiles using subprocess pool with memory isolation
        _t0_topo = time.perf_counter()
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context('spawn'),
                                 max_tasks_per_child=1) as executor:
            list(executor.map(_process_topo_worker, topo_tile_args))
        if debug:
            print(f'PROFILE: topo ProcessPoolExecutor ({len(topo_tile_args)} tiles, {n_jobs} workers) {time.perf_counter() - _t0_topo:.3f}s')

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

# the fewest correlated patches (with a non-zero offset) that xcorr_fitoffset fits
XCORR_MIN_PATCHES = 8


def xcorr_fitoffset(results: list, nx: int, ny: int, rank: int = 3, debug: bool = False) -> dict | None:
    """
    Fit the offset model to xcorr offsets using PRM.fitoffset (robust IRLS with MAD): bilinear (rank 3) or a
    constant shift (rank 1).

    Converts xcorr results to the matrix format expected by PRM.fitoffset,
    which uses iteratively reweighted least squares with MAD-based outlier
    downweighting - proven and consistent with geometry fitting.

    Parameters
    ----------
    results : list
        List of dicts with 'cy1', 'cx1', 'dy', 'dx', 'response': the patches the xcorr gate (min_response)
        accepted. Every one with a non-zero offset enters the fit: there is no second response cut.
    nx : int
        Image width (num_rng_bins) - unused, kept for API compatibility.
    ny : int
        Image height (num_lines) - unused, kept for API compatibility.
    rank : int
        Terms of the model per direction (PRM.fitoffset rank_rng = rank_azi): 3 = shift plus the stretches in range
        and azimuth (bilinear), 2 = shift plus the stretch in range, 1 = a constant shift (the stretches are 0).
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
    if n_initial < XCORR_MIN_PATCHES:
        if debug:
            print(f"  xcorr_fitoffset: only {n_initial} patches, need >= {XCORR_MIN_PATCHES}")
        return None  # Insufficient patches - xcorr failed

    # Filter out zero offsets (invalid/failed correlations)
    n_before_zero = len(results)
    results = [r for r in results if abs(r['dx']) > 0.01 or abs(r['dy']) > 0.01]
    n_zeros = n_before_zero - len(results)

    if len(results) < XCORR_MIN_PATCHES:
        if debug:
            print(f"  xcorr_fitoffset: {n_zeros} zero offsets filtered, only {len(results)} remain")
        return None  # Insufficient patches after zero filtering

    # Convert to PRM.fitoffset matrix format: [r, dr, a, da, SNR]
    # r = cx1 (range position), dr = dx (range offset)
    # a = cy1 (azimuth position), da = dy (azimuth offset)
    # SNR = 100 for every row, as the geometry offset tables: the xcorr gate (min_response) is the one threshold,
    # so every accepted patch passes the SNR > 20 cut of PRM.fitoffset (response * 100 there silently raised any
    # min_response below 0.2 to 0.2)
    matrix = np.array([
        [r['cx1'], r['dx'], r['cy1'], r['dy'], 100.0]
        for r in results
    ])

    if debug:
        dx = matrix[:, 1]
        dy = matrix[:, 3]
        print(f"  xcorr_fitoffset: {n_initial} initial, {n_zeros} zeros, {len(results)} final")
        print(f"  offset range: dx=[{dx.min():.2f}, {dx.max():.2f}], dy=[{dy.min():.2f}, {dy.max():.2f}]")

    # Use PRM.fitoffset - robust IRLS with MAD-based outlier downweighting
    # rank=3 for bilinear model: offset = c0 + c1*r + c2*a; rank=1: offset = c0
    try:
        prm_result = PRM.fitoffset(rank, rank, matrix, SNR=20, debug=debug)
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


# =============================================================================
# SATELLITE-AGNOSTIC UTILITIES (moved from S1_align.py, S1_transform.py)
# =============================================================================

def _offset2shift(xyz, rmax, amax, method='linear'):
    """Convert offset coordinates to shift grid.

    Parameters
    ----------
    xyz : np.ndarray
        Array of shape (N, 3) with columns [range, azimuth, value].
    rmax : int
        Maximum range coordinate.
    amax : int
        Maximum azimuth coordinate.
    method : str
        Interpolation method for scipy.interpolate.griddata.

    Returns
    -------
    xr.DataArray
        Interpolated shift grid with dims ['a', 'r'].
    """
    import xarray as xr
    from scipy.interpolate import griddata

    rngs = np.arange(8/2, rmax+8/2, 8)
    azis = np.arange(4/2, amax+4/2, 4)
    grid_r, grid_a = np.meshgrid(rngs, azis)
    grid = griddata((xyz[:, 0], xyz[:, 1]), xyz[:, 2], (grid_r, grid_a), method=method)
    return xr.DataArray(grid, coords={'a': azis, 'r': rngs}, dims=['a', 'r'])
