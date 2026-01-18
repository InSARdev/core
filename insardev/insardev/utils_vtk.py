# ----------------------------------------------------------------------------
# insardev
#
# VTK utilities extracted from Stack_export
# ----------------------------------------------------------------------------
from __future__ import annotations


def as_vtk(dataset):
    """Convert a 2D xarray dataset to a VTK structured grid.

    Expects each data_var to be either (y, x) or (band, y, x). Any other
    dimension mix is rejected to prevent malformed VTK files.
    """
    from vtk import (
        vtkPoints,
        vtkStructuredGrid,
        vtkThreshold,
        vtkDataObject,
        VTK_FLOAT,
        VTK_UNSIGNED_CHAR,
        vtkStringArray,
        vtkFloatArray,
        vtkIntArray,
    )
    from vtk.util import numpy_support as vn
    import numpy as np
    import xarray as xr

    assert isinstance(dataset, xr.Dataset), 'ERROR: Expected "dataset" argument as Xarray Dataset'

    # Work on a copy to avoid mutating user data
    dataset = dataset.copy(deep=True)

    xs = np.asarray(dataset.x.values)
    ys = np.asarray(dataset.y.values)
    xx, yy = np.meshgrid(xs, ys, indexing='xy')
    if 'z' in dataset:
        zs = np.asarray(dataset.z.values)
    else:
        zs = np.zeros_like(xx)

    vtk_points = vtkPoints()
    points = np.column_stack((xx.ravel(order='C'), yy.ravel(order='C'), zs.ravel(order='C')))
    vtk_points.SetData(vn.numpy_to_vtk(points, deep=True))

    sgrid = vtkStructuredGrid()
    sgrid.SetDimensions(len(xs), len(ys), 1)
    sgrid.SetPoints(vtk_points)

    for data_var in dataset.data_vars:
        if data_var == 'z':
            continue

        da = dataset[data_var]
        dims = tuple(da.dims)

        # Only allow (y, x) or (band, y, x)
        if dims not in (('y', 'x'), ('band', 'y', 'x')):
            raise ValueError(
                f'Unsupported dimensions {dims} for variable {data_var}; expected (y, x) or (band, y, x)'
            )

        if dims == ('band', 'y', 'x'):
            # RGB/RGBA data - don't apply fill_value handling (0/255 are valid colors)
            values = np.asarray(da.values)
            bands = values.shape[0]
            if bands in (3, 4):
                # Clip to valid range and convert to uint8
                array = vn.numpy_to_vtk(
                    values[:3].clip(0, 255).round().astype(np.uint8).reshape(3, -1).T,
                    deep=True,
                    array_type=VTK_UNSIGNED_CHAR,
                )
            elif bands == 1:
                # Single band - treat as scalar, apply fill_value handling
                values_1b = values[0]
                if np.issubdtype(da.dtype, np.floating):
                    values_1b = np.asarray(values_1b, dtype=np.float32)
                    fill_value = da.attrs.get('_FillValue')
                    if fill_value is not None and not np.isnan(fill_value):
                        values_1b = np.where(values_1b == fill_value, np.nan, values_1b)
                array = vn.numpy_to_vtk(values_1b.ravel(order='C'), deep=True, array_type=VTK_FLOAT)
            else:
                raise ValueError(
                    f'Unsupported band count {bands} for variable {data_var} (expected 1, 3, or 4)'
                )
        else:
            # Scalar (y, x) data - apply fill_value handling
            if np.issubdtype(da.dtype, np.floating):
                values = np.asarray(da.values, dtype=np.float32)
                fill_value = da.attrs.get('_FillValue')
                if fill_value is not None and not np.isnan(fill_value):
                    values = np.where(values == fill_value, np.nan, values)
            else:
                values = np.asarray(da.values)
            array = vn.numpy_to_vtk(values.ravel(order='C'), deep=True, array_type=VTK_FLOAT)

        array.SetName(da.name)
        sgrid.GetPointData().AddArray(array)

    for coord in dataset.coords:
        if len(dataset[coord].dims) > 0:
            continue
        if np.issubdtype(dataset[coord].dtype, np.datetime64):
            data_array = vtkStringArray()
            data_value = str(dataset[coord].dt.date.values)
        elif np.issubdtype(dataset[coord].dtype, str):
            data_array = vtkStringArray()
            data_value = str(dataset[coord].values)
        elif np.issubdtype(dataset[coord].dtype, np.int64):
            data_array = vtkIntArray()
            data_value = dataset[coord].values.astype(np.int64)
        elif np.issubdtype(dataset[coord].dtype, np.float64):
            data_array = vtkFloatArray()
            data_value = dataset[coord].values.astype(np.float64)
        else:
            continue
        data_array.SetName(coord)
        data_array.InsertNextValue(data_value)
        sgrid.GetFieldData().AddArray(data_array)

    return sgrid
