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
import xarray as xr

def get_spacing(da, coarsen=(1, 1)):
    import numpy as np
    if coarsen is None:
        coarsen = (1, 1)
    if not isinstance(coarsen, (list, tuple, np.ndarray)):
        coarsen = (coarsen, coarsen)
    dy = da.y.diff('y').item(0)
    dx = da.x.diff('x').item(0)
    if coarsen is not None:
        dy *= coarsen[0]
        dx *= coarsen[1]
    return (dy, dx)

def spacing_of(ds):
    """(dy, dx) ground metres per pixel of a burst, positive."""
    import numpy as np
    y = np.asarray(ds['y'].values, dtype=float)
    x = np.asarray(ds['x'].values, dtype=float)
    return (abs(float(y[1] - y[0])) if y.size > 1 else 1.0,
            abs(float(x[1] - x[0])) if x.size > 1 else 1.0)


def meters_to_pixels(value, spacing, minimum=1, odd=False, name='window'):
    """A GROUND SIZE in metres -> samples on THIS grid, per axis.

    Every spatial size a caller states is a distance on the ground, so the
    same call describes the same neighbourhood whatever the posting. The
    conversion happens once, at the public boundary, and everything inside
    the library keeps working in pixels.

    `minimum` is the smallest sensible count per axis; a request that rounds
    below it RAISES rather than collapsing to a degenerate window, and the
    message says what it would take on this grid. `odd` rounds up to the next
    odd count for kernels centred on a pixel.
    """
    import numpy as np
    if not isinstance(value, (list, tuple, np.ndarray)):
        value = (value, value)
    value = tuple(value)
    if len(value) != 2:
        raise ValueError(f'{name} takes one value or two (y, x) in metres, '
                         f'got {value}')
    out = []
    for v, d, ax in zip(value, spacing, ('y', 'x')):
        v, d = float(v), abs(float(d))
        if not v > 0:
            raise ValueError(f'{name} must be positive metres, got {value}')
        n = int(round(v / d))
        if odd and n % 2 == 0:
            n += 1
        if n < minimum:
            raise ValueError(
                f'{name}={v:g} m is under {minimum} pixel(s) on this grid, '
                f'which posts {d:g} m per pixel in {ax}: ask for at least '
                f'{minimum * d:g} m.')
        out.append(n)
    return tuple(out)


def window_meters_to_pixels(window, spacing, name='window'):
    """(DS, PS) or (DSy, DSx, PSy, PSx) in metres -> (wy, wx, pey, pex) pixels.

    Two values are the DS window and the PS extent as GROUND SQUARES, so
    (400, 4000) is a 400 m box inside a 4000 m search on any posting, and the
    anisotropy of the grid is handled by the conversion rather than by the
    caller. Four values state the two axes separately, still in metres.
    """
    import numpy as np
    w = tuple(window) if isinstance(window, (list, tuple, np.ndarray)) else None
    if w is None or len(w) not in (2, 4):
        raise ValueError(
            f'{name} takes 2 values (DS, PS extent) or 4 (DSy, DSx, PSy, PSx), '
            f'in METRES; got {window!r}')
    if len(w) == 2:
        w = (w[0], w[0], w[1], w[1])
    wy, wx = meters_to_pixels((w[0], w[1]), spacing, minimum=2,
                              name=f'{name} (DS box)')
    pey, pex = meters_to_pixels((w[2], w[3]), spacing, minimum=2,
                                name=f'{name} (PS extent)')
    return wy, wx, pey, pex


def coarsen_start(da, name, spacing, grid_factor=1):
    """
    Calculate start coordinate to align coarsened grids.
    
    Parameters
    ----------
    da : xarray.DataArray
        Input data array
    name : str
        Coordinate name to align
    spacing : int
        Coarsening spacing
    grid_factor : int, optional
        Grid factor for alignment, default is 1
        
    Returns
    -------
    int or None
        Start index for optimal alignment, or None if no good alignment found
    """
    import numpy as np
    
    # get coordinate values
    coords = da[name].values
    if len(coords) < spacing:
        print(f'_coarsen_start: Not enough points for spacing {spacing}')
        return None
        
    # calculate coordinate differences
    diffs = np.diff(coords)
    if not np.allclose(diffs, diffs[0], rtol=1e-5):
        print(f'_coarsen_start: Non-uniform spacing detected for {name}')
        return None
        
    # calculate target spacing
    target_spacing = diffs[0] * spacing * grid_factor
    
    # find best alignment point
    best_offset = None
    min_error = float('inf')
    
    for i in range(spacing):
        # get coarsened coordinates
        coarse_coords = coords[i::spacing]
        if len(coarse_coords) < 2:
            continue
            
        # calculate alignment error
        error = np.abs(coarse_coords[0] % target_spacing)
        if error < min_error:
            min_error = error
            best_offset = i
            
    if best_offset is not None:
        #print(f'_coarsen_start: {name} spacing={spacing} grid_factor={grid_factor} => {best_offset} (error={min_error:.2e})')
        return best_offset
        
    print(f'_coarsen_start: No good alignment found for {name}')
    return None

def to_dict(datas: dict[str, xr.Dataset | xr.DataArray] | None = None):
    """
    Convert a list of datasets or dictionaries of datasets to a dictionary.
    """
    if isinstance(datas, xr.Dataset):
        return {'default': datas}
    if isinstance(datas, xr.DataArray):
        return {'default': datas.to_dataset()}
    return datas

def apply(*args, **kwarg):
    """
    Apply a function to multiple datasets or dictionaries of datasets with the same keys.

    Parameters
    ----------
    *args : list of datasets or dictionaries of datasets
        The datasets to apply the function to.
    **kwarg : dict
        The keyword arguments to pass to the function.

    Returns
    -------
    dict or dataset
        The result of applying the function to the datasets.
        If the input is a dictionary or a list of dictionaries, the result is a dictionary.
        If the input is a dataset or a list of datasets, the result is a dataset.
    
    Examples
    --------
    >>> sbas.apply(func=lambda a, b, **kwargs: (a, b))
    >>> sbas.apply(intfs, corrs, func=lambda a, b, **kwargs: (a, b))
    >>> sbas.apply(intfs, corrs, func=lambda a, b, **kwargs: a)
    >>> sbas.apply(intfs['106_226497_IW1'], corrs['106_226497_IW1'], func=lambda a, b, **kwargs: (a,b))
    >>> sbas.apply(intfs['106_226497_IW1'], corrs['106_226497_IW1'], func=lambda a, b, **kwargs: a)
    """
    from insardev_toolkit import progressbar
    import dask

    func = kwarg.pop('func', None)
    if func is None:
        raise ValueError('`func` argument is required')
    compute = kwarg.pop('compute', False)
    #print ('compute', compute)
    add_key = kwarg.pop('add_key', False)
    if not args:
        return
    datas = [to_dict(arg) if arg is not None else None for arg in args]
    keys = list(datas[0].keys())
    if add_key:
        dss = {key: func(*(d[key] if d is not None else None for d in datas), **(kwarg | {'key': key})) for key in keys}
    else:
        dss = {key: func(*(d[key] if d is not None else None for d in datas), **kwarg) for key in keys}
    if compute:
        progressbar(dss := dask.persist(dss)[0], desc=f'Computing...'.ljust(25))
    # detect output type
    sample = next(iter(dss.values()))
    # multiple datasets or dictionaries
    if (isinstance(sample, (tuple, list))):
        n = len(sample)
        dicts = [{key: dss[key][i] for key in keys} for i in range(n)]
        if isinstance(args[0], dict):
            return tuple(dicts)
        return tuple(d['default'] for d in dicts)
    # single dataset or dictionary
    if isinstance(args[0], dict):
        return dss
    return dss['default']

