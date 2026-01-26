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
from .utils_satellite import satellite_rat2llt, get_geoid, get_utm_epsg, proj, compute_transform


class S1_geocode(S1_align):
    import pandas as pd
    import xarray as xr
    import numpy as np

    def geocode(self, transform: xr.Dataset, data: xr.DataArray,
                resolution: tuple[float, float] = None) -> xr.DataArray:
        """
        Perform geocoding from radar to projected coordinates using inverse transform.

        The inverse transform has coords (y, x) and vars (rng, azi, ele).
        Uses cv2.remap with Lanczos interpolation directly on the inverse maps.

        Parameters
        ----------
        transform : xarray.Dataset
            The inverse transform with coords (y, x) and vars (rng, azi, ele).
        data : xarray.DataArray
            Grid(s) in radar coordinates (a, r).
        resolution : tuple[float, float], optional
            Output resolution (dy, dx) in meters. If None, uses transform resolution.

        Returns
        -------
        xarray.DataArray
            The geocoded grid(s) in projected coordinates (y, x).
        """
        import cv2
        import xarray as xr
        import numpy as np

        # get transform arrays - inverse transform has coords (y, x), vars (azi, rng)
        trans_azi = transform.azi.values  # 2D: (n_y, n_x)
        trans_rng = transform.rng.values  # 2D: (n_y, n_x)
        out_y = transform.y.values  # 1D
        out_x = transform.x.values  # 1D

        # get data arrays - data is in radar coords (a, r)
        data_vals = data.values
        coord_a = data.a.values
        coord_r = data.r.values

        # Convert transform azi/rng to fractional radar indices
        # inv_map_a[i,j] = row index in radar data, inv_map_r[i,j] = col index
        inv_map_a = ((trans_azi - coord_a[0]) / (coord_a[1] - coord_a[0])).astype(np.float32)
        inv_map_r = ((trans_rng - coord_r[0]) / (coord_r[1] - coord_r[0])).astype(np.float32)

        n_y, n_x = inv_map_a.shape
        OPENCV_MAX = 32766  # cv2.remap requires dimensions < 32767

        # Use cv2.remap with Lanczos to sample radar data at inverse map coordinates
        # map_r is x (column), map_a is y (row) in source image
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

    def get_transform(self, outdir: str, burst: str = None) -> xr.Dataset:
        """
        Retrieve the inverse transform data.

        The inverse transform has coords (y, x) and vars (rng, azi, ele, look_E, look_N, look_U).
        For each projected pixel, stores the corresponding radar coordinates and look vectors.

        Parameters
        ----------
        outdir : str
            Output directory containing transform zarr.
        burst : str, optional
            The burst name (not used, kept for API compatibility).

        Returns
        -------
        xarray.Dataset
            An xarray dataset with the transform data.
        """
        import xarray as xr
        import os
        import numpy as np
        ds = xr.open_zarr(store=os.path.join(outdir, 'transform'),
                         consolidated=True,
                          zarr_format=3,
                         chunks='auto')
        # variables are stored as int32 with _FillValue and scale_factor
        # inverse transform has vars (rng, azi, ele)
        for v in ('rng', 'azi', 'ele'):
            if v not in ds:
                continue
            fill_value = ds[v].attrs.get('_FillValue')
            scale_factor = ds[v].attrs.get('scale_factor', 1.0)
            if fill_value is not None:
                # xarray didn't decode - apply manually
                data = ds[v].astype('float32')
                data = data.where(ds[v] != fill_value)
                ds[v] = data * scale_factor
            else:
                # xarray already decoded - ensure float32 and mask extreme values
                data = ds[v].astype('float32')
                ds[v] = data.where(np.abs(data) < 1e8)
        return ds
