# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .S1_dem import S1_dem
from .PRM import PRM


class S1_align(S1_dem):
    import numpy as np
    import xarray as xr
    import pandas as pd

    @staticmethod
    def _offset2shift(xyz: np.ndarray, rmax: int, amax: int, method: str = 'linear') -> xr.DataArray:
        """
        Convert offset coordinates to shift values on a grid.

        Parameters
        ----------
        xyz : numpy.ndarray
            Array containing the offset coordinates (x, y, z) = (range, azimuth, shift).
        rmax : int
            Maximum range bin.
        amax : int
            Maximum azimuth line.
        method : str, optional
            Interpolation method. Default is 'linear'.

        Returns
        -------
        xarray.DataArray
            Array containing the shift values on a grid.
        """
        import xarray as xr
        import numpy as np
        from scipy.interpolate import griddata

        # use center pixel GMT registration mode
        rngs = np.arange(8/2, rmax+8/2, 8)
        azis = np.arange(4/2, amax+4/2, 4)
        grid_r, grid_a = np.meshgrid(rngs, azis)

        grid = griddata((xyz[:, 0], xyz[:, 1]), xyz[:, 2], (grid_r, grid_a), method=method)
        # No flipud needed - grid row index directly corresponds to azimuth index
        da = xr.DataArray(grid, coords={'y': azis, 'x': rngs}, name='z')
        return da

    @staticmethod
    def _offset2shift_combined(offset_dat: np.ndarray, rmax: int, amax: int) -> tuple:
        """
        Convert offset coordinates to r and a shift grids in one interpolation.

        Uses LinearNDInterpolator to build triangulation once and interpolate
        both r_shift and a_shift values together, which is ~2x faster than
        calling griddata separately for each.

        Parameters
        ----------
        offset_dat : numpy.ndarray
            Array with columns [r, dr, a, da, snr] from offset computation.
        rmax : int
            Maximum range bin.
        amax : int
            Maximum azimuth line.

        Returns
        -------
        tuple
            (r_grd, a_grd) - xarray.DataArrays with shift grids.
        """
        import xarray as xr
        import numpy as np
        from scipy.interpolate import LinearNDInterpolator

        # use center pixel GMT registration mode
        rngs = np.arange(8/2, rmax+8/2, 8)
        azis = np.arange(4/2, amax+4/2, 4)
        grid_r, grid_a = np.meshgrid(rngs, azis)
        grid_points = np.column_stack([grid_r.ravel(), grid_a.ravel()])

        # Input points: (range, azimuth) coordinates
        points = np.column_stack([offset_dat[:, 0], offset_dat[:, 2]])  # [r, a]
        # Shift values: (dr, da)
        shifts = np.column_stack([offset_dat[:, 1], offset_dat[:, 3]])  # [dr, da]

        # Build triangulation once, interpolate both shifts
        interp = LinearNDInterpolator(points, shifts, fill_value=np.nan)
        result = interp(grid_points)  # Shape: (n_points, 2)

        # Reshape to grid
        r_grid = result[:, 0].reshape(grid_r.shape)
        a_grid = result[:, 1].reshape(grid_r.shape)

        r_grd = xr.DataArray(r_grid, coords={'y': azis, 'x': rngs}, name='z')
        a_grd = xr.DataArray(a_grid, coords={'y': azis, 'x': rngs}, name='z')

        return r_grd, a_grd

    def _get_topo_llt(self, burst: str, degrees: float, debug: bool = False) -> tuple:
        """
        Get the topography coordinates (lon, lat, z) for decimated DEM.

        Parameters
        ----------
        burst : str
            Burst identifier.
        degrees : float
            Number of degrees for decimation.
        debug : bool, optional
            Enable debug mode. Default is False.

        Returns
        -------
        numpy.ndarray
            Array containing the topography coordinates (lon, lat, z), NaN filtered.
        """
        import numpy as np
        import warnings
        warnings.filterwarnings('ignore')

        # add buffer around the cropped area for borders interpolation
        record = self.get_record(burst)
        dem_area = self.get_dem_wgs84ellipsoid(geometry=record.geometry)

        ny = int(np.round(degrees/dem_area.lat.diff('lat')[0]))
        nx = int(np.round(degrees/dem_area.lon.diff('lon')[0]))
        if debug:
            print('DEBUG: DEM decimation', 'ny', ny, 'nx', nx)
        dem_area = dem_area.coarsen({'lat': ny, 'lon': nx}, boundary='pad').mean()

        # Extract values directly using meshgrid instead of xr.broadcast
        lat_vals = dem_area.lat.values
        lon_vals = dem_area.lon.values
        z_vals = dem_area.values

        # Create coordinate arrays using meshgrid (keep float64 for lon/lat precision)
        lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
        topo_llt = np.column_stack([
            lon_grid.ravel(),
            lat_grid.ravel(),
            z_vals.ravel()
        ])
        del lon_grid, lat_grid, dem_area, lat_vals, lon_vals, z_vals

        # Filter out records where the elevation (third column) is NaN
        valid_mask = ~np.isnan(topo_llt[:, 2])
        result = topo_llt[valid_mask]
        del topo_llt, valid_mask
        return result

    def align_ref(self, burst: str, debug: bool = False, return_slc: bool = True) -> tuple:
        """
        Process reference burst - extract PRM, orbit, and optionally deramped SLC.

        All data is returned in-memory, no files are written.

        Parameters
        ----------
        burst : str
            Burst identifier.
        debug : bool, optional
            Enable debug mode. Default is False.
        return_slc : bool, optional
            If True, load and return deramped SLC data. If False, only return PRM.
            Default is True.

        Returns
        -------
        tuple
            (prm, slc_deramped, reramp_params) or (prm, None) if return_slc=False
        """
        if return_slc:
            from .PRM import PRM
            prm_dict, orbit_df, slc_data, reramp_params = self._make_burst(burst, mode=2)
            prm = PRM()
            prm.set(**prm_dict)
            prm.orbit_df = orbit_df
            prm.calc_dop_orb(inplace=True, debug=debug)
            return prm, slc_data, reramp_params

        prm, orbit_df = self._make_burst(burst, mode=0, debug=debug)
        prm.calc_dop_orb(inplace=True, debug=debug)
        return prm, None

    def align_rep(self, burst_rep: str, burst_ref: str, prm_ref: "PRM",
                  degrees: float = 12.0/3600, debug: bool = False) -> tuple:
        """
        Process and align secondary burst to reference.

        Returns deramped SLC (no alignment shift, no reramp) with alignment
        offsets stored in the PRM. The alignment and geocoding interpolations
        are merged into a single remap step during the transform phase,
        avoiding double interpolation.

        All data is returned in-memory, no files are written.

        Parameters
        ----------
        burst_rep : str
            Secondary burst identifier.
        burst_ref : str
            Reference burst identifier (used for DEM geometry).
        prm_ref : PRM
            Reference PRM object (from align_ref) with Doppler parameters.
        degrees : float, optional
            Degrees per pixel resolution for the coarse DEM. Default is 12.0/3600.
        debug : bool, optional
            Enable debug mode. Default is False.

        Returns
        -------
        tuple
            (prm, slc_data, reramp_params) where:
            - prm: PRM object with orbit_df attached, alignment, and Doppler parameters
            - slc_data: Deramped SLC data as int16 numpy array (no shift, no reramp)
            - reramp_params: dict with parameters for analytical reramp phase computation

        Examples
        --------
        >>> prm_ref, slc_ref, _ = stack.align_ref(burst_ref)
        >>> prm_rep, slc_rep, reramp_params = stack.align_rep(burst_rep, burst_ref, prm_ref)
        """
        import numpy as np

        # Get reference earth_radius
        earth_radius = prm_ref.get('earth_radius')

        # Prepare coarse DEM for alignment using REFERENCE burst geometry
        # (12 arc seconds is enough for SRTM 90m)
        topo_llt = self._get_topo_llt(burst_ref, degrees=degrees)

        # Extract PRM and orbit for secondary burst (mode=0, no SLC yet)
        prm_rep, orbit_df = self._make_burst(burst_rep, mode=0, debug=debug)

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

        # Compute r, dr, a, da, SNR table for fitoffset
        offset_dat0 = np.hstack([tmpm_dat, tmp1_dat])
        func = lambda row: [row[0], row[5] - row[0], row[1], row[6] - row[1], 100]
        offset_dat = np.apply_along_axis(func, 1, offset_dat0)

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

        # Extract deramped SLC (no shift, no reramp)
        prm_dict, orbit_df_new, slc_data, reramp_params = self._make_burst(burst_rep, mode=2)

        # Build PRM from dict (same as _make_burst does)
        prm_rep = PRM()
        prm_rep.set(**prm_dict)
        prm_rep.orbit_df = orbit_df_new

        # Apply fitoffset parameters (bilinear offset model stored in PRM)
        prm_rep.set(PRM.fitoffset(3, 3, par_tmp))

        # Recompute Doppler with earth_radius
        prm_rep.calc_dop_orb(earth_radius, inplace=True, debug=debug)

        return prm_rep, slc_data, reramp_params
