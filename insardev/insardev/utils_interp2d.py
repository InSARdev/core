# ----------------------------------------------------------------------------
# insardev
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev directory for license terms.
# Professional use requires an active per-seat subscription at: https://patreon.com/pechnikov
# ----------------------------------------------------------------------------

# Xarray's interpolation can be inefficient for large grids;
# this custom function handles the task more effectively.
def interp2d_like(data, grid, method='cubic', **kwargs):
    """
    Efficiently interpolate a 2D array using OpenCV interpolation methods.
    
    Args:
        data (xarray.DataArray): The input data array.
        grid (xarray.DataArray): The grid to interpolate onto.
        method (str): Interpolation method ('nearest', 'linear', 'cubic' or 'lanczos').
        **kwargs: Additional arguments for interpolation.

    Returns:
        xarray.DataArray: The interpolated data.
    """
    import cv2
    import numpy as np
    import xarray as xr
    import dask.array as da
    # prevent PerformanceWarning: Increasing number of chunks by factor of ...
    import warnings
    warnings.filterwarnings("ignore", category=da.core.PerformanceWarning)

    dims = grid.dims[-2:]
    dim1, dim2 = dims
    coords = {dim1: grid[dim1], dim2: grid[dim2]}
    #print ('coords', coords)

    # Define interpolation method
    if method == 'nearest':
        interpolation = cv2.INTER_NEAREST
    elif method == 'linear':
        interpolation = cv2.INTER_LINEAR
    elif method == 'cubic':
        interpolation = cv2.INTER_CUBIC
    elif method == 'lanczos':
        interpolation = cv2.INTER_LANCZOS4
    else:
        raise ValueError(f"Unsupported interpolation {method}. Should be 'nearest', 'linear', 'cubic' or 'lanczos'")

    # TBD: can be added to the function parameters
    borderMode = cv2.BORDER_REFLECT

    # define interpolation function using outer variable data
    def interpolate_chunk(out_chunk1, out_chunk2, dim1, dim2, interpolation, borderMode, **kwargs):
        d1 = float(data[dim1].diff(dim1)[0])
        d2 = float(data[dim2].diff(dim2)[0])

        # select the chunk from data with some padding
        chunk = data.sel({
            dim1: slice(out_chunk1[0] - 3 * d1, out_chunk1[-1] + 3 * d1),
            dim2: slice(out_chunk2[0] - 3 * d2, out_chunk2[-1] + 3 * d2)
        }).compute(n_workers=1)

        # Create grid for interpolation
        dst_grid_x, dst_grid_y = np.meshgrid(out_chunk2, out_chunk1)

        # map destination grid coordinates to source pixel indices
        src_x_coords = np.interp(
            dst_grid_x.ravel(),
            chunk[dim2].values,
            np.arange(len(chunk[dim2]))
        )
        src_y_coords = np.interp(
            dst_grid_y.ravel(),
            chunk[dim1].values,
            np.arange(len(chunk[dim1]))
        )

        # reshape the coordinates for remap
        src_x_coords = src_x_coords.reshape(dst_grid_x.shape).astype(np.float32)
        src_y_coords = src_y_coords.reshape(dst_grid_y.shape).astype(np.float32)

        # interpolate using OpenCV
        dst_grid = cv2.remap(
            chunk.values.astype(np.float32),
            src_x_coords,
            src_y_coords,
            interpolation=interpolation,
            borderMode=borderMode
        )
        return dst_grid

    # define chunk sizes
    chunk_sizes = grid.chunks[-2:] if hasattr(grid, 'chunks') else (data.sizes[dim1], data.sizes[dim2])

    # create dask array for parallel processing
    grid_y = da.from_array(grid[dim1].values, chunks=chunk_sizes[0])
    grid_x = da.from_array(grid[dim2].values, chunks=chunk_sizes[1])

    # Perform interpolation
    meta = np.empty((0, 0), dtype=data.dtype)
    dask_out = da.blockwise(
        interpolate_chunk,
        'yx',
        grid_y, 'y',
        grid_x, 'x',
        dtype=data.dtype,
        meta=meta,
        dim1=dim1,
        dim2=dim2,
        interpolation=interpolation,
        borderMode=borderMode,
        **kwargs
    )

    da_out = xr.DataArray(dask_out, coords=coords, dims=dims).rename(data.name)

    # Append all the input coordinates
    return da_out.assign_coords({k: v for k, v in data.coords.items() if k not in coords})
