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


def write_vtm(batch, path, transform=None, mask=True):
    """Write the seamless multiblock VTK export of a batch under `path`: per variable `<var>.vtm`, the index (one block per
    burst, one dataset per dask block), and `<var>/<burst>_<ci>_<cj>.vts`, one structured-grid piece per dask block.

    Each piece holds its block's pixels plus the next row and column (the shared one-pixel edge), all pairs or dates as point
    arrays, and z the transform's ele by nearest pixel (NaN where masked, as to_vtk). A point takes the to_dataset() value of
    its pixel: the most recent burst with finite data wins. The pieces are written inside the tasks of the blocks; the
    driver gathers only the flags of the written pieces.

    Examples
    --------
    >>> write_vtm(velocity, 'velocity', transform=transform)
    """
    import os
    import numpy as np
    import dask
    import dask.array as da
    from xml.sax.saxutils import escape, quoteattr
    from .BatchCore import BatchCore

    keys = BatchCore._acquisition_order(batch, who='to_vtm')
    if transform is not None:
        tf = {k: transform[k][['ele']] for k in keys if k in transform}
        if not tf:
            raise ValueError(f"ERROR: to_vtm(): the transform has none of the bursts "
                             f"({', '.join(keys[:3])}). Pass their transform.")
    else:
        tf = None
    os.makedirs(path, exist_ok=True)

    # the grid variables (y, x): the per-date metadata of a burst (baseline, burst ids) is not exported
    def grid(k, v):
        return v in batch[k].data_vars and {'y', 'x'} <= set(batch[k][v].dims)

    variables = []
    for k in keys:
        variables += [v for v in batch[k].data_vars if v not in variables and grid(k, v)]

    for var in variables:
        vkeys = [k for k in keys if grid(k, var)]
        vdir = os.path.join(path, var)
        os.makedirs(vdir, exist_ok=True)
        stacks = {k: _stack_of(batch[k][var], var) for k in vkeys}
        bbox = {k: (float(s.y.min()), float(s.y.max()), float(s.x.min()), float(s.x.max())) for k, (s, _, _) in stacks.items()}
        computes, meta = [], []
        for k in vkeys:
            g, _, labels = stacks[k]
            y0, x0 = g.y.values, g.x.values
            dy, dx = float(y0[1] - y0[0]), float(x0[1] - x0[0])
            Ey = np.append(y0, y0[-1] + dy)
            Ex = np.append(x0, x0[-1] + dx)
            bE = (float(Ey.min()), float(Ey.max()), float(Ex.min()), float(Ex.max()))
            js = [j for j in vkeys if _overlaps(bE, bbox[j])]
            own = js.index(k)
            tol = min(abs(dy), abs(dx)) / 2
            parts = []
            for j in js:
                gj, _, labels_j = stacks[j]
                if labels_j != labels:
                    raise ValueError(f'to_vtm(): burst {j} has other pairs or dates than burst {k}')
                D = _float32(gj).reindex(y=Ey, x=Ex, method='nearest', tolerance=tol).data
                parts.append(D)
                if tf is not None and j in tf:
                    ti = tf[j]
                    yi = _nearest(ti.y.values, gj.y.values)
                    xi = _nearest(ti.x.values, gj.x.values)
                    z = ti['ele'].isel(y=yi, x=xi).assign_coords(y=gj.y, x=gj.x).astype(np.float32)
                    parts.append(z.reindex(y=Ey, x=Ex, method='nearest', tolerance=tol).data[None])
                else:
                    parts.append(da.full((1, len(Ey), len(Ex)), np.nan if tf is not None else 0, dtype=np.float32))
            Gy = list(_float32(g).data.chunks[1])
            Gx = list(_float32(g).data.chunks[2])
            Ech = {0: -1, 1: tuple(Gy[:-1] + [Gy[-1] + 1]), 2: tuple(Gx[:-1] + [Gx[-1] + 1])}
            s = da.concatenate([p.rechunk(Ech) for p in parts], axis=0).rechunk({0: -1})
            # the shared one-pixel edge: the next block's first row and column, read from the same graph
            so = da.overlap.overlap(s, depth={1: (0, 1), 2: (0, 1)}, boundary={0: 'none', 1: 'none', 2: 'none'})
            # the halo keeps the input's blocks: the output less its edge has the input's block sizes, else overlap_check raises
            from .utils_dask import overlap_check
            trim = so.map_blocks(lambda b: b[:, :-1, :-1], dtype=so.dtype,
                                 chunks=(so.chunks[0], tuple(c - 1 for c in so.chunks[1]), tuple(c - 1 for c in so.chunks[2])))
            overlap_check(trim, da.empty((trim.shape[0], sum(Gy), sum(Gx)), chunks=(trim.chunks[0], tuple(Gy), tuple(Gx))),
                          'to_vtm', (2, 2), (1, 1))
            nby, nbx = len(Gy), len(Gx)
            fn = _piece_writer(vdir, k, labels, len(js), own, Gy, Gx, Ey, Ex, tf is not None and mask)
            computes.append(da.map_blocks(fn, so, drop_axis=0, chunks=((1,) * nby, (1,) * nbx), dtype=np.int8))
            meta.append((k, nby, nbx))
        flags = dask.compute(*computes)

        lines = ['<?xml version="1.0"?>',
                 '<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian" header_type="UInt64">',
                 '  <vtkMultiBlockDataSet>']
        for bi, ((k, nby, nbx), fl) in enumerate(zip(meta, flags)):
            lines.append(f'    <Block index="{bi}" name={quoteattr(str(k))}>')
            for ci in range(nby):
                for cj in range(nbx):
                    if fl[ci, cj]:
                        name = f'{k}_{ci:03d}_{cj:03d}'
                        lines.append(f'      <DataSet index="{ci * nbx + cj}" name={quoteattr(name)} '
                                     f'file={quoteattr(f"{var}/{name}.vts")}/>')
            lines.append('    </Block>')
        lines += ['  </vtkMultiBlockDataSet>', '</VTKFile>']
        with open(os.path.join(path, f'{var}.vtm'), 'w') as f:
            f.write('\n'.join(lines) + '\n')


def _stack_of(g, var):
    """(the variable as (stack, y, x), its stack dimension, its array names): one layer per pair or date, or the variable alone."""
    from . import utils_xarray
    if 'pair' in g.dims:
        stack = 'pair'
        labels = [f'{utils_xarray.format_date(r)}_{utils_xarray.format_date(p)}'
                  for r, p in zip(g.coords['ref'].values, g.coords['rep'].values)]
    elif 'date' in g.dims:
        stack = 'date'
        labels = [utils_xarray.format_date(d) for d in g['date'].values]
    else:
        stack, labels = 'layer', [var]
        g = g.expand_dims(stack)
    if tuple(g.dims) != (stack, 'y', 'x'):
        raise ValueError(f'to_vtm(): {var} has dims {tuple(g.dims)}; expected (pair|date, y, x) or (y, x)')
    return g, stack, labels


def _float32(g):
    """the values as float32 with the _FillValue as NaN, as as_vtk writes them (lazy)."""
    import numpy as np
    import dask.array as da
    d = da.asarray(g.data).astype(np.float32)
    fill = g.attrs.get('_FillValue')
    if fill is not None and not np.isnan(fill):
        d = da.where(d == np.float32(fill), np.float32(np.nan), d)
    return g.copy(data=d)


def _nearest(source, target):
    """the index in `source` of each `target` coordinate (BatchCore.to_vtk's decimation of the transform)."""
    from . import utils_xarray
    return utils_xarray.nearest_indices(source, target)


def _overlaps(a, b):
    """do two boxes (y0, y1, x0, x1) share a pixel edge."""
    return a[0] <= b[1] and b[0] <= a[1] and a[2] <= b[3] and b[2] <= a[3]


def _piece_writer(vdir, burst, labels, nb, own, Gy, Gx, Ey, Ex, masked):
    """The function of one dask block (runs in its task): the piece `<burst>_<ci>_<cj>.vts` of the block's pixels and the
    next row and column of the neighbours, every pair as a point array, z as the first pair's mask (when `masked`).
    Returns 1 when a piece was written (a block with no finite pixel writes none), else 0."""
    import os
    import numpy as np
    oy = np.concatenate([[0], np.cumsum(Gy)])[:-1]
    ox = np.concatenate([[0], np.cumsum(Gx)])[:-1]
    P = len(labels)

    def write_piece(blk, block_info=None):
        from vtk import vtkStructuredGrid, vtkPoints, vtkXMLStructuredGridWriter, VTK_FLOAT
        from vtk.util import numpy_support as vn
        info = block_info[0]
        ci, cj = info['chunk-location'][1], info['chunk-location'][2]
        ya, xa = int(oy[ci]), int(ox[cj])
        own_r, own_c = int(Gy[ci]), int(Gx[cj])
        arr = np.asarray(blk, dtype=np.float32)
        R, C = arr.shape[1], arr.shape[2]
        D = [arr[j * (P + 1): j * (P + 1) + P] for j in range(nb)]
        Zb = [arr[j * (P + 1) + P] for j in range(nb)]
        # the to_dataset() value of every pixel: the most recent burst with finite data wins
        M = np.full((P, R, C), np.nan, dtype=np.float32)
        Zf = np.full((R, C), np.nan, dtype=np.float32)
        for j in range(nb):
            M = np.where(np.isfinite(D[j]), D[j], M)
            Zf = np.where(np.isfinite(Zb[j]), Zb[j], Zf)
        # data priority: a pixel of this burst is dropped where a later burst has finite data
        lost = np.zeros((P, R, C), dtype=bool)
        for j in range(own + 1, nb):
            lost |= np.isfinite(D[j])
        inG = np.zeros((R, C), dtype=bool)
        inG[:own_r, :own_c] = True
        K = inG[None] & ~lost & np.isfinite(M)
        # the points of the block's cells: the kept pixels and their next row and column (the shared edge)
        S = K.copy()
        S[:, 1:, :] |= K[:, :-1, :]
        S[:, :, 1:] |= K[:, :, :-1]
        S[:, 1:, 1:] |= K[:, :-1, :-1]
        V = np.where(S, M, np.float32(np.nan)).astype(np.float32)
        Z = np.where(np.isfinite(M[0]), Zf, np.float32(np.nan)) if masked else Zf
        Sany = S.any(axis=0)
        rows = np.flatnonzero(Sany.any(axis=1))
        cols = np.flatnonzero(Sany.any(axis=0))
        if rows.size == 0:
            return np.zeros((1, 1), dtype=np.int8)
        r0, r1 = int(rows[0]), int(rows[-1]) + 1
        c0, c1 = int(cols[0]), int(cols[-1]) + 1
        xs = Ex[xa + c0: xa + c1]
        ys = Ey[ya + r0: ya + r1]
        xx, yy = np.meshgrid(xs, ys, indexing='xy')
        zc = Z[r0:r1, c0:c1].astype(np.float64)
        points = np.column_stack((xx.ravel(order='C'), yy.ravel(order='C'), zc.ravel(order='C')))
        sg = vtkStructuredGrid()
        sg.SetDimensions(len(xs), len(ys), 1)
        vp = vtkPoints()
        vp.SetData(vn.numpy_to_vtk(points, deep=True))
        sg.SetPoints(vp)
        for p, lab in enumerate(labels):
            arr_p = vn.numpy_to_vtk(V[p, r0:r1, c0:c1].ravel(order='C'), deep=True, array_type=VTK_FLOAT)
            arr_p.SetName(lab)
            sg.GetPointData().AddArray(arr_p)
        w = vtkXMLStructuredGridWriter()
        w.SetFileName(os.path.join(vdir, f'{burst}_{ci:03d}_{cj:03d}.vts'))
        w.SetInputData(sg)
        w.Write()
        return np.ones((1, 1), dtype=np.int8)

    return write_piece
