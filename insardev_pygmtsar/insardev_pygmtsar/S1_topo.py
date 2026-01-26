# ----------------------------------------------------------------------------
# insardev_pygmtsar
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_pygmtsar directory for license terms.
# ----------------------------------------------------------------------------
from .S1_geocode import S1_geocode

class S1_topo(S1_geocode):
    import xarray as xr

    def flat_earth_topo_phase(self, topo: xr.DataArray, burst_rep: str, burst_ref: str, basedir: str) -> xr.DataArray:
        """
        np.arctan2(np.sin(topo_phase), np.cos(topo_phase))[0].plot.imshow()
        """
        import xarray as xr
        import numpy as np
        from scipy import constants
        import warnings
        warnings.filterwarnings('ignore')

        if topo is None:
            # skip topo phase removal, use zero elevation for flat-earth correction
            prm_ref = self.PRM(burst_ref, basedir=basedir)
            xdim = prm_ref.get('num_rng_bins')
            ydim = prm_ref.get('num_patches') * prm_ref.get('num_valid_az')
            # use 0.5-based coordinates matching the SLC grid (same as compute_topo)
            azis = np.arange(0.5, ydim, 1)
            rngs = np.arange(0.5, xdim, 1)
            topo = xr.DataArray(np.zeros((len(azis), len(rngs)), dtype=np.float32),
                                dims=['a', 'r'],
                                coords={'a': azis, 'r': rngs}).rename('topo')

        # calculate the combined earth curvature and topography correction
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

        def prepare_prms(burst_rep, burst_ref):
            if burst_rep == burst_ref:
                return (None, None)
            prm_ref = self.PRM(burst_ref, basedir=basedir)
            prm_rep = self.PRM(burst_rep, basedir=basedir)
            prm_rep.set(prm_ref.SAT_baseline(prm_rep, tail=9)).fix_aligned()
            prm_ref.set(prm_ref.SAT_baseline(prm_ref).sel('SC_height','SC_height_start','SC_height_end')).fix_aligned()
            return (prm_ref, prm_rep)

        prm1, prm2 = prepare_prms(burst_rep, burst_ref)

        if prm1 is None or prm2 is None:
            # Reference burst: no phase correction needed (return zeros)
            return xr.DataArray(np.zeros_like(topo.values, dtype=np.float32), topo.coords).rename('phase')

        # fill NaNs by 0
        topo_vals = np.where(np.isnan(topo.values), 0, topo.values)
        y_coords = topo.a.values
        x_coords = topo.r.values

        # get full dimensions
        xdim = prm1.get('num_rng_bins')
        ydim = prm1.get('num_patches') * prm1.get('num_valid_az')

        # get heights
        htc = prm1.get('SC_height')
        ht0 = prm1.get('SC_height_start')
        htf = prm1.get('SC_height_end')

        # compute the time span and the time spacing
        tspan = 86400 * abs(prm2.get('SC_clock_stop') - prm2.get('SC_clock_start'))
        assert (tspan >= 0.01) and (prm2.get('PRF') >= 0.01), \
            f"ERROR in sc_clock_start={prm2.get('SC_clock_start')}, sc_clock_stop={prm2.get('SC_clock_stop')}, or PRF={prm2.get('PRF')}"

        # setup the default parameters
        drange = constants.speed_of_light / (2 * prm2.get('rng_samp_rate'))
        alpha = prm2.get('alpha_start') * np.pi / 180
        cnst = -4 * np.pi / prm2.get('radar_wavelength')

        # calculate initial baselines
        Bh0 = prm2.get('baseline_start') * np.cos(prm2.get('alpha_start') * np.pi / 180)
        Bv0 = prm2.get('baseline_start') * np.sin(prm2.get('alpha_start') * np.pi / 180)
        Bhf = prm2.get('baseline_end')   * np.cos(prm2.get('alpha_end')   * np.pi / 180)
        Bvf = prm2.get('baseline_end')   * np.sin(prm2.get('alpha_end')   * np.pi / 180)
        Bx0 = prm2.get('B_offset_start')
        Bxf = prm2.get('B_offset_end')

        # first case is quadratic baseline model, second case is default linear model
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

        # calculate height increment
        dht = (-3 * ht0 + 4 * htc - htf) / tspan
        ddht = (2 * ht0 - 4 * htc + 2 * htf) / (tspan * tspan)

        # Ensure float64 precision for near_range calculation to avoid sqrt precision issues
        # (float32 rho gives sqrt(rho**2) != rho due to precision loss)
        x_coords_f64 = x_coords.astype(np.float64)
        y_coords_f64 = y_coords.astype(np.float64)
        near_range = (prm1.get('near_range') + \
            x_coords_f64.reshape(1,-1) * (1 + prm1.get('stretch_r')) * drange) + \
            y_coords_f64.reshape(-1,1) * prm1.get('a_stretch_r') * drange

        # calculate the change in baseline and height along the frame
        time = y_coords_f64 * tspan / (ydim - 1)
        Bh = Bh0 + dBh * time + ddBh * time**2
        Bv = Bv0 + dBv * time + ddBv * time**2
        Bx = Bx0 + dBx * time + ddBx * time**2
        B = np.sqrt(Bh * Bh + Bv * Bv)
        alpha = np.arctan2(Bv, Bh)
        height = ht0 + dht * time + ddht * time**2

        # calculate the combined earth curvature and topography correction
        drho = calc_drho(near_range, topo_vals, prm1.get('earth_radius'),
                         height.reshape(-1, 1), B.reshape(-1, 1), alpha.reshape(-1, 1), Bx.reshape(-1, 1))

        phase_shift = (cnst * drho).astype(np.float32)

        topo_phase = xr.DataArray(phase_shift, topo.coords)
        topo_phase = topo_phase.where(np.isfinite(topo)).rename('phase')
        return topo_phase

    def get_topo(self, burst, basedir: str):
        """
        Retrieve the inverse transform data.

        This function opens a NetCDF dataset, which contains data mapping from radar
        coordinates to geographical coordinates (from azimuth-range to latitude-longitude domain).

        Parameters
        ----------
        burst : str
            The burst name.

        Returns
        -------
        xarray.Dataset
            An xarray dataset with the transform data.

        Examples
        --------
        Get the inverse transform data:
        get_trans_inv()
        """
        import xarray as xr
        import os
        return xr.open_zarr(os.path.join(basedir, 'topo'),
                             zarr_format=3,
                            consolidated=True,
                            chunks="auto")['topo']

    def compute_topo(self, workdir: str, transform: xr.Dataset, burst_ref: str, basedir: str):
        """
        Retrieve or calculate the transform data. This transform data is then saved as
            a NetCDF file for future use.

            This function generates data mapping from radar coordinates to geographical coordinates.
            The function uses the direct transform data.

        Parameters
        ----------
        workdir : str
            The work directory.
        burst_ref : str
            The reference burst name.
        basedir : str
            The basedir directory.
        resolution : tuple[int, int]
            The resolution of the transform data.

        Note
        ----
        This function operates on the 'transform' grid using chunks (specified by 'chunksize') rather than
        larger processing chunks. This approach is effective due to on-the-fly index creation for the NetCDF chunks.

        """
        from scipy.spatial import cKDTree
        import xarray as xr
        import numpy as np
        import os
        import warnings
        warnings.filterwarnings('ignore')

        # get transform data as numpy arrays
        transform_azi = transform.azi.values
        transform_rng = transform.rng.values
        transform_ele = transform.ele.values
        transform_y = transform.y.values
        transform_x = transform.x.values

        # define radar coordinate grid
        prm = self.PRM(burst_ref, basedir)
        a_max, r_max = prm.bounds()
        azis = np.arange(0.5, a_max, 1)
        rngs = np.arange(0.5, r_max, 1)

        # flatten transform arrays for KDTree
        valid_mask = np.isfinite(transform_azi) & np.isfinite(transform_rng)
        trans_azi_flat = transform_azi[valid_mask]
        trans_rng_flat = transform_rng[valid_mask]
        trans_ele_flat = transform_ele[valid_mask]

        # build KDTree for nearest neighbor search
        tree = cKDTree(np.column_stack([trans_azi_flat, trans_rng_flat]), compact_nodes=False, balanced_tree=False)

        # create output grid
        grid_azi, grid_rng = np.meshgrid(azis, rngs, indexing='ij')
        query_points = np.column_stack([grid_azi.ravel(), grid_rng.ravel()])

        # find nearest neighbors
        tolerance = 2
        distances, indices = tree.query(query_points, k=1, workers=1)

        # get elevation values
        grid_ele = trans_ele_flat[indices]
        grid_ele[distances > tolerance] = np.nan
        grid_ele = grid_ele.reshape(azis.size, rngs.size).astype(np.float32)

        # create output DataArray
        coords = {'a': azis, 'r': rngs}
        topo = xr.DataArray(grid_ele, coords=coords, dims=['a', 'r']).rename('topo')

        # use a single chunk for efficient storage
        encoding = {'topo': {'chunks': topo.shape}}
        topo.to_zarr(
            store=os.path.join(basedir, 'topo'),
            mode='w',
            zarr_format=3,
            consolidated=True,
            encoding=encoding
        )
        del topo
