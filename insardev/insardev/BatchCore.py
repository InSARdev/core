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
from __future__ import annotations
from . import utils_io,  utils_xarray
import operator
import numpy as np
import xarray as xr
from collections.abc import Mapping
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .Batch import Batch, BatchWrap, BatchUnit, BatchComplex, BatchVar
    from .Stack import Stack
    import rasterio as rio
    import pandas as pd
    import matplotlib

class BatchCore(dict):
    """
    This class has 'pair' stack variable for the datasets in the dict and stores real values (correlation and unwrapped phase).
    
    Examples:
    intfs60_detrend = Batch(intfs60) - Batch(intfs60_trend)

    dss = intfs60_detrend.sel(['106_226487_IW2','106_226488_IW2','106_226489_IW2','106_226490_IW2','106_226491_IW2'])
    dss_fixed = dss + {'106_226490_IW2': 2.6, '106_226491_IW2': 3})

    intfs60_detrend.isel(1)
    intfs60_detrend.isel([0, 2])
    intfs60_detrend.isel(slice(1, None))
    """

    class CoordCollection:
        def __init__(self, ds):
            self._ds = ds
        def __getitem__(self, key):
            return self._ds.coords[key]
        def __contains__(self, key):
            return key in self._ds.coords
        def get(self, key, default=None):
            return self._ds.coords.get(key, default)
        def keys(self):
            return self._ds.coords.keys()
        def values(self):
            return self._ds.coords.values()
        def items(self):
            return self._ds.coords.items()

    @staticmethod
    def _get_torch_device(device='auto', debug=False):
        """Get PyTorch device. See utils_gaussian._get_torch_device for full docs."""
        from .utils_gaussian import _get_torch_device
        return _get_torch_device(device, debug)

    @staticmethod
    def _gaussian(data_np, weight_np=None, sigma=None, truncate=4.0, threshold=0.5, device='auto',
                  pixel_sizes=None, resolution=67.0):
        """2D Gaussian convolution. See utils_gaussian.gaussian_numpy for full docs."""
        from .utils_gaussian import gaussian_numpy
        return gaussian_numpy(data_np, weight_np, sigma, truncate, threshold, device, pixel_sizes, resolution)

    def __init__(self, mapping: Mapping[str, xr.Dataset] | Stack | BatchComplex | None = None):
        from .Stack import Stack
        from .Batch import Batch, BatchWrap, BatchUnit, BatchComplex
        #print('BatchCore __init__', 0 if mapping is None else len(mapping))
        # Batch/etc. initialization won't filter out the data when it's a child class of BatchCore
        if isinstance(mapping, (Stack, BatchComplex)) and not isinstance(self, (Batch, BatchWrap, BatchUnit, BatchComplex)):
            real_dict = {}
            for key, ds in mapping.items():
                # pick only the data_vars whose dtype is not complex
                real_vars = [v for v in ds.data_vars if ds[v].dtype.kind != 'c' and tuple(ds[v].dims) == ('y', 'x')]
                real_dict[key] = ds[real_vars]
            mapping = real_dict
        #print('BatchCore __init__ mapping', mapping or {}, '\n')
        super().__init__(mapping or {})

    def from_dataset(self, data: xr.Dataset) -> Batch:
        """
        Create a Batch by selecting each burst's coordinates from a merged Dataset.

        The input Dataset should have been created via to_dataset() or have
        coordinates that are supersets of each burst's coordinates.

        Parameters
        ----------
        data : xr.Dataset
            The input data to split back into per-burst datasets.

        Returns
        -------
        Batch
            A new Batch with the same keys as self, each containing the
            selected subset of the Dataset.

        Examples
        --------
        # Round-trip: merge, process, split
        merged = batch.to_dataset()
        processed = some_processing(merged)
        result = batch.from_dataset(processed)
        """
        from .Batch import Batch

        # Validate input type
        if not isinstance(data, xr.Dataset):
            raise TypeError(f"data must be xr.Dataset, got {type(data).__name__}")

        out = {}
        for key, ds in dict.items(self):
            # Select burst's spatial extent - coordinates match exactly
            selected = data.sel(y=ds.y, x=ds.x)

            # Align non-spatial dims (pair, date) if needed
            for dim in data.dims:
                if dim not in ('y', 'x') and dim in ds.dims:
                    # Select matching size and assign burst's coordinates
                    selected = selected.isel({dim: slice(0, ds.sizes[dim])})
                    selected = selected.assign_coords({dim: ds.coords[dim]})

            out[key] = selected
        return Batch(out)

    # def __repr__(self):
    #     if not self:
    #         return f"{self.__class__.__name__}(empty)"
    #     n = len(self)
    #     if n <= 1:
    #         # delegate to the underlying dict repr
    #         return dict.__repr__(self)
    #     sample = next(iter(self.values()))
    #     if not 'date' in sample and not 'pair' in sample:
    #         return f'{self.__class__.__name__} object containing {len(self)} items'
    #     sample_len = f'{len(sample.date)} date' if 'date' in sample else f'{len(sample.pair)} pair'
    #     keys = list(self.keys())
    #     return f'{self.__class__.__name__} object containing {len(self)} items for {sample_len} ({keys[0]} ... {keys[-1]})'

    # def __repr__(self):
    #     if not self:
    #         return f"{self.__class__.__name__}(empty)"
    #     sample = next(iter(self.values()))  # pick any dataset
    #     # figure out which stack coord we have
    #     if 'date' in sample.coords:
    #         count = sample.coords['date'].size
    #         axis_name = 'date'
    #     elif 'pair' in sample.coords:
    #         count = sample.coords['pair'].size
    #         axis_name = 'pair'
    #     else:
    #         # fallback if neither coord is present
    #         return f"{self.__class__.__name__} containing {len(self)} items"
    #     keys = list(self.keys())
    #     return (
    #         f"{self.__class__.__name__} containing {len(self)} items "
    #         f"for {count} {axis_name} "
    #         f"({keys[0]} … {keys[-1]})"
    #     )

    def __repr__(self):
        # empty case
        if not self:
            return f"{self.__class__.__name__}(empty)"

        n = len(self)
        # single‐item: show the actual Dataset repr
        if n == 1:
            key, ds = next(iter(self.items()))
            return f"{self.__class__.__name__}['{key}']:\n{ds!r}"

        # multi‐item: show summary
        sample = next(iter(self.values()))
        
        # Handle CoordCollection objects
        if isinstance(sample, self.CoordCollection):
            keys = list(self.keys())
            return f"{self.__class__.__name__} coords containing {n} items ({keys[0]} … {keys[-1]})"
        
        if 'date' in sample.coords:
            count = sample.coords['date'].size
            axis = 'date'
        elif 'pair' in sample.coords:
            count = sample.coords['pair'].size
            axis = 'pair'
        else:
            return f"{self.__class__.__name__} containing {n} items"

        keys = list(self.keys())
        return (
            f"{self.__class__.__name__} containing {n} items "
            f"for {count} {axis} "
            f"({keys[0]} … {keys[-1]})"
        )

    def __or__(self, other):
        # Batch | Mapping
        if not isinstance(other, Mapping):
            return NotImplemented
        merged = dict(self)
        merged.update(other)
        return type(self)(merged)

    def __ror__(self, other):
        # Mapping | Batch
        if not isinstance(other, Mapping):
            return NotImplemented
        merged = dict(other)
        merged.update(self)
        return type(self)(merged)

    @property
    def data(self) -> xr.Dataset:
        """
        Return the single Dataset in this Batch.

        Raises
        ------
        ValueError
            if the Batch has zero or more than one item.
        """
        n = len(self)
        if n != 1:
            raise ValueError(f'Batch.data is only available for single-item Batches, but this Batch has {n} items')
        # return the only Dataset
        return next(iter(self.values()))

    # @property
    # def chunks(self) -> tuple[int, int, int]:
    #     sample = next(iter(self.values()))
    #     # for DatasetCoarsen extract the original Dataset
    #     if hasattr(sample, 'obj'):
    #         sample = sample.obj
    #     data_var = [var for var in sample.data_vars if (sample[var].ndim in (2,3) and sample[var].dims[-2:] == ('y','x'))][0]

    #     if sample[data_var].chunks is None:
    #         print ('WARNING: Batch.chunks undefined, i.e. the data is not lazy and parallel chunks processing is not possible.')
    #         return (1, -1, -1)
    #     else:
    #         return tuple(chunks[0] for chunks in sample[data_var].chunks)

    @property
    def crs(self) -> rio.crs.CRS:
        return next(iter(self.values())).rio.crs

    @property
    def chunks(self) -> dict[str, int]:
        try:
            sample = next(iter(self.values()))
        except StopIteration:
            return {}

        # for DatasetCoarsen extract the original Dataset
        if hasattr(sample, 'obj'):
            sample = sample.obj
        data_var = [var for var in sample.data_vars if (sample[var].ndim in (2,3) and sample[var].dims[-2:] == ('y','x'))][0]
        
        chunks = sample[data_var].chunks
        #print ('chunks', chunks)
        if chunks is None:
            print ('WARNING: Batch.chunks undefined, i.e. the data is not lazy and parallel chunks processing is not possible.')
            # use "common" chunking for 2D and 3D data
            return -1 if data_var.ndim == 2 else (1, -1, -1)

        # build dict of first‐chunk sizes, one chunk means chunk size 1 or -1
        return {dim: sizes[0] if len(sizes) > 1 else (1 if sizes[0] == 1 else -1) for dim, sizes in zip(sample[data_var].dims, chunks)}

    def __getitem__(self, key):
        """
        Access coordinates, data variables, or datasets in the batch.
        
        Parameters
        ----------
        key : str, list, or tuple
            If str: access coordinate or data variable across all datasets
            If list/tuple: select subset of datasets
            
        Returns
        -------
        Batch
            Batch of the requested coordinate/variable or selected datasets
        """
        # Handle list/tuple keys for dataset selection
        if isinstance(key, (list, tuple)):
            return type(self)({
                burst_id: ds[key]
                for burst_id, ds in self.items()
            })
            
        # Try to access as a dataset key first
        try:
            return super().__getitem__(key)
        except KeyError:
            # If not a dataset key, try to access as coordinate/variable
            return type(self)({
                k: ds[key] if not isinstance(ds, self.CoordCollection) else ds._ds.coords[key]
                for k, ds in self.items()
                if (isinstance(ds, self.CoordCollection) and key in ds._ds.coords) or 
                   (not isinstance(ds, self.CoordCollection) and (key in ds.coords or key in ds.data_vars))
            })

    def __getattr__(self, name: str):
        """Attribute-style access to coords or data variables (e.g., batch.ele)."""
        if name.startswith('_') or name in ('keys', 'values', 'items', 'get'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if not self:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        def _extract(ds):
            # CoordCollection wrapper
            if isinstance(ds, self.CoordCollection):
                if name in ds._ds.coords:
                    return ds._ds.coords.to_dataset()[[name]]
                return None

            # Dataset: prefer data_vars, then coords
            if hasattr(ds, 'data_vars'):
                if name in ds.data_vars:
                    return ds[[name]]
                if name in ds.coords:
                    return ds.coords.to_dataset()[[name]]
                return None

            # DataArray fallback: match by name
            if hasattr(ds, 'name') and ds.name == name:
                return ds.to_dataset()
            if hasattr(ds, 'coords') and name in ds.coords:
                return ds.coords.to_dataset()[[name]]
            return None

        subset = {k: out for k, ds in self.items() if (out := _extract(ds)) is not None}
        if subset:
            return type(self)(subset)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __add__(self, other):
        # scalar + batch → map scalar + each dataset
        if isinstance(other, (int, float, np.floating, np.integer)):
            return type(self)({k: v + other for k, v in self.items()})
        keys = self.keys()
        return type(self)({k: (self[k] + other[k] if k in other else self[k]) for k in keys})

    def __radd__(self, other):
        # scalar + batch → same as batch + scalar
        return self.__add__(other)

    def __sub__(self, other):
        # scalar - batch → map scalar - each dataset
        if isinstance(other, (int, float, np.floating, np.integer)):
            return type(self)({k: v - other for k, v in self.items()})
        keys = self.keys()
        result = {}
        for k in keys:
            if k not in other:
                result[k] = self[k]
            else:
                val = other[k]
                ds = self[k]
                # Handle per-pair coefficients from burst_polyfit
                if isinstance(val, (list, tuple)):
                    # Get a spatial variable (with y, x dims) to check for pair dimension
                    spatial_vars = [v for v in ds.data_vars if 'y' in ds[v].dims and 'x' in ds[v].dims]
                    sample_var = spatial_vars[0] if spatial_vars else list(ds.data_vars)[0]
                    sample_da = ds[sample_var]
                    has_pair_dim = 'pair' in sample_da.dims
                    n_pairs = sample_da.sizes.get('pair', 1)

                    if len(val) > 0 and isinstance(val[0], (list, tuple)):
                        # Multi-pair degree=1: [[ramp0, off0], [ramp1, off1], ...]
                        # Use polyval for this case
                        result[k] = ds - self[[k]].polyval({k: val})[k]
                    elif has_pair_dim and len(val) == n_pairs:
                        # Multi-pair degree=0: [off0, off1, ...]
                        offsets = xr.DataArray(val, dims=['pair'])
                        result[k] = ds - offsets
                    else:
                        # Single pair degree=1: [ramp, offset] or other
                        result[k] = ds - val
                elif isinstance(val, (int, float, np.floating, np.integer)):
                    # Scalar subtraction: only apply to spatial variables (y, x dims)
                    # to avoid errors with non-spatial variables like (date,) dims
                    new_ds = ds.copy()
                    for var in ds.data_vars:
                        if 'y' in ds[var].dims and 'x' in ds[var].dims:
                            new_ds[var] = ds[var] - val
                    result[k] = new_ds
                else:
                    result[k] = ds - val
        return type(self)(result)

    def __rsub__(self, other):
        # scalar - batch
        if isinstance(other, (int, float, np.floating, np.integer)):
            return type(self)({k: other - v for k, v in self.items()})
        return NotImplemented

    def __mul__(self, other):
        # scalar * batch → map scalar * each dataset
        if isinstance(other, (int, float, np.floating, np.integer)):
            return type(self)({k: v * other for k, v in self.items()})
        keys = self.keys()
        return type(self)({k: (self[k] * other[k] if k in other else self[k]) for k in keys})

    def __rmul__(self, other):
        # scalar * batch  → map scalar * each dataset
        return type(self)({k: other * v for k, v in self.items()})

    def __truediv__(self, other):
        # batch / scalar → map each dataset / scalar
        if isinstance(other, (int, float, np.floating, np.integer)):
            return type(self)({k: v / other for k, v in self.items()})
        keys = self.keys()
        return type(self)({k: (self[k] / other[k] if k in other else self[k]) for k in keys})

    def __rtruediv__(self, other):
        # scalar / batch
        if isinstance(other, (int, float, np.floating, np.integer)):
            return type(self)({k: other / v for k, v in self.items()})
        return NotImplemented

    def __neg__(self):
        # -batch → negate each dataset
        return type(self)({k: -v for k, v in self.items()})

    def _binop(self, other, op):
        """
        generic helper for any binary operator `op(ds, other)` or `op(ds, other_ds)`
        """
        if isinstance(other, (int, float)):
            return type(self)({k: op(ds, other) for k, ds in self.items()})
        elif isinstance(other, BatchCore):
            common = set(self) & set(other)
            return type(self)({k: op(self[k], other[k]) for k in common})
        else:
            return NotImplemented

    def __gt__(self, other):   return self._binop(other, operator.gt)
    def __lt__(self, other):   return self._binop(other, operator.lt)
    def __ge__(self, other):   return self._binop(other, operator.ge)
    def __le__(self, other):   return self._binop(other, operator.le)
    def __eq__(self, other):   return self._binop(other, operator.eq)
    def __ne__(self, other):   return self._binop(other, operator.ne)

    # reversed ops
    __rgt__ = __gt__
    __rlt__ = __lt__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Support numpy ufuncs on Batch objects, e.g.:
        - np.exp(-1j * intfs)
        - np.isfinite(weight)
        - np.abs(batch)
        """
        # only handle the normal call
        if method != "__call__":
            return NotImplemented

        # find the first Batch among inputs
        batch = next((x for x in inputs if isinstance(x, BatchCore)), None)
        if batch is None:
            return NotImplemented

        result = {}
        for k in batch.keys():
            # build the argument list for this key
            args = [
                inp[k] if isinstance(inp, BatchCore) else inp
                for inp in inputs
            ]
            result[k] = ufunc(*args, **kwargs)

        return type(self)(result)

    # def iexp(self):
    #     """
    #     np.exp(-1j * intfs)
    #     """
    #     import numpy as np
    #     return np.exp(1j * self)

    # def conj(self) -> BatchWrap:
    #     """
    #     Return a new BatchWrap in which each complex dataset has been
    #     replaced with its complex conjugate.

    #     Example:
    #     intfs.iexp().conj() for np.exp(-1j * intfs)
    #     """
    #     return type(self)({
    #         k: ds.conj()
    #         for k, ds in self.items()
    #     })

    def map_da(self, func, **kwargs):
        """Apply func(DataArray) → DataArray to every var in every dataset."""
        return type(self)({
            k: ds.map(func, **kwargs)
            for k, ds in self.items()
        })

    def astype(self, dtype, **kwargs):
        return self.map_da(lambda da: da.astype(dtype), **kwargs)
    
    def abs(self, **kwargs):
        return self.map_da(lambda da: xr.ufuncs.abs(da), **kwargs)

    def square(self, **kwargs):
        return self.map_da(lambda da: xr.ufuncs.square(da), **kwargs)

    def sqrt(self, **kwargs):
        return self.map_da(lambda da: xr.ufuncs.sqrt(da), **kwargs)

    def log10(self, **kwargs):
        return self.map_da(lambda da: xr.ufuncs.log10(da), **kwargs)

    def multiply(self, value, **kwargs):
        return self.map_da(lambda da: da * value, **kwargs)

    def divide(self, value, **kwargs):
        return self.map_da(lambda da: da / value, **kwargs)

    def clip(self, min=None, max=None, **kwargs):
        return self.map_da(lambda da: da.clip(min=min, max=max), **kwargs)

    def isfinite(self, **kwargs):
        return self.map_da(lambda da: xr.ufuncs.isfinite(da), **kwargs)

    # def where(self, cond, other=0):
    #     # cond can be a BatchWrap of booleans
    #     if isinstance(cond, BatchWrap):
    #         return type(self)({
    #             k: ds.where(cond[k], other)
    #             for k, ds in self.items()
    #         })
    #     else:
    #         return self.map_da(lambda da: da.where(cond, other), keep_attrs=True)

    # def where(self, cond, other=0, **kwargs):
    #     """
    #     Batch‐wise .where:
        
    #     - if `cond` is a Batch (or BatchWrap) with exactly the same keys:
    #         * when other==0 → do ds * mask  (very fast, no alignment)
    #         * otherwise   → ds.where(mask, other, **kwargs)
    #     - else:
    #         broadcast a single mask or scalar/DataArray
    #         into every var via `map_da(lambda da: da.where(cond, other, **kwargs))`.
    #     """
    #     # per‐burst mask
    #     if hasattr(cond, 'keys') and set(cond.keys()) == set(self.keys()):
    #         print ('X')
    #         out = {}
    #         for k, ds in self.items():
    #             mask = cond[k]
    #             # if mask coords don't exactly match ds, you can
    #             # uncomment the next line to reindex first:
    #             # mask = mask.reindex_like(ds, method='nearest')
                
    #             if other == 0:
    #                 # blaze past .where with a simple multiply
    #                 out[k] = ds * mask
    #             else:
    #                 out[k] = ds.where(mask, other, **kwargs)
    #         return type(self)(out)

    #     # single‐mask/scalar-broadcast case:
    #     return self.map_da(lambda da: da.where(cond, other, **kwargs), **kwargs)


    # def where(self, cond, other=0, **kwargs):
    #     """
    #     Batch-wise .where: if cond is another Batch with exactly the same keys,
    #     do each ds.where(mask, other), otherwise fall back to per-DataArray broadcast.
    #     """
    #     # 1) fast path: cond is a Batch with the same bursts
    #     if isinstance(cond, Batch) and set(cond.keys()) == set(self.keys()):
    #         return type(self)({
    #             k: ds.where(cond[k], other, **kwargs)
    #             for k, ds in self.items()
    #         })

    #     # 2) broadcast a single mask/scalar to every var
    #     return self.map_da(lambda da: da.where(cond, other, **kwargs), **kwargs)

    def where(self, cond, other=np.nan, **kwargs):
        """
        Fast batch-wise .where:

        If `cond` is a Batch (or subclass) with exactly the same keys,
           and each cond[k] is a 1-variable Dataset or a DataArray,
           we extract the single DataArray mask and do either:
             - other==0 → simple multiply ds * mask_da
             - else       → ds.where(mask_da, other, **kwargs)

        Otherwise fall back to per-DataArray map_da (slower).

        keep_attrs=True argument can be used to preserve attributes of the original data.
        """
        # detect same key Batch-like mask
        if hasattr(cond, 'keys') and set(cond.keys()) == set(self.keys()):
            out = {}
            for k, ds in self.items():
                mask_obj = cond[k]
                # extract DataArray from a 1-var Dataset or use it direct
                if isinstance(mask_obj, xr.Dataset):
                    data_vars = list(mask_obj.data_vars)
                    if len(data_vars) != 1:
                        raise ValueError(f"Batch.where: expected 1 var in mask for '{k}', got {data_vars}")
                    mask_da = mask_obj[data_vars[0]]
                else:
                    mask_da = mask_obj
                
                # Align mask to data coordinates (handles different x/y grids)
                # Get reference DataArray from ds for alignment (use spatial variable)
                if isinstance(ds, xr.Dataset):
                    spatial_vars = [v for v in ds.data_vars if 'y' in ds[v].dims and 'x' in ds[v].dims]
                    ref_var = spatial_vars[0] if spatial_vars else list(ds.data_vars)[0]
                    ref_da = ds[ref_var]
                else:
                    ref_da = ds
                mask_da = mask_da.reindex_like(ref_da, method='nearest')
                
                out[k] = ds.where(mask_da, other, **kwargs)
            return type(self)(out)

        # fallback: single scalar or DataArray broadcast
        # DataArray case seems not usefull because Batch datasets differ in shape
        return self.map_da(lambda da: da.where(cond, other, **kwargs), **kwargs)

    def mask(self, mask, other=np.nan):
        """
        Apply a mask to each burst.

        This is memory-efficient for large masks (e.g., landmask, DEM) as it
        reindexes the mask to each burst's coordinates using nearest-neighbor
        interpolation instead of broadcasting the full mask.

        Parameters
        ----------
        mask : xr.DataArray, xr.Dataset, or GeoDataFrame
            The mask to apply. Can be:
            - xr.DataArray: boolean mask reindexed to each burst's coordinates
            - xr.Dataset: first data variable used as mask, reindexed per burst
            - GeoDataFrame: polygon(s) to mask by - pixels inside polygons are kept
        other : scalar, optional
            Value to use for masked elements. Default is np.nan.

        Returns
        -------
        Batch
            Masked batch with same type as self.

        Examples
        --------
        # Apply binary landmask
        land = np.isfinite(xr.open_dataarray('land.nc').rio.reproject(intf.crs))
        masked_intf = intf.mask(land)

        # Mask by AOI polygon
        AOI = gpd.read_file('aoi.geojson')
        masked_velocity = velocity.mask(AOI)
        """
        import geopandas as gpd
        from shapely import Geometry
        import rioxarray

        # Handle GeoDataFrame/GeoSeries/Geometry masking
        if isinstance(mask, (gpd.GeoDataFrame, gpd.GeoSeries, Geometry)):
            # Extract geometry for rio.clip
            if isinstance(mask, gpd.GeoDataFrame):
                geom = mask.geometry
                mask_crs = mask.crs
            elif isinstance(mask, gpd.GeoSeries):
                geom = mask
                mask_crs = mask.crs
            else:
                # Shapely Geometry - no CRS info
                geom = [mask]
                mask_crs = None

            # Reproject to batch CRS if needed
            crs = self.crs
            if crs is not None and mask_crs is not None and mask_crs != crs:
                geom = gpd.GeoSeries(geom, crs=mask_crs).to_crs(crs)

            out = {}
            for key, ds in self.items():
                out[key] = ds.rio.clip(geom, all_touched=False)
            return type(self)(out)

        # Handle xarray mask
        if isinstance(mask, xr.Dataset):
            mask = next(iter(mask.data_vars.values()))

        if not np.issubdtype(mask.dtype, np.bool_):
            raise ValueError('Batch.mask: mask must be a Dataset or DataArray of boolean type, or a GeoDataFrame')

        # auto-chunk if not already chunked to avoid high memory usage
        if not mask.chunks:
            mask = mask.chunk('auto')

        out = {}
        for key, ds in self.items():
            # the fastest way to align mask to burst coordinates
            mask_burst = mask.reindex(y=ds.y, x=ds.x, method='nearest')
            # preserve original chunking structure for lazy computation
            out[key] = ds.where(mask_burst, other)
        return type(self)(out)

    def crop(self, geometry):
        """
        Crop each burst to the bounding rectangle of a geometry.

        Unlike mask() which clips to the exact geometry shape, crop() selects
        the bounding box (rectangular extent) of the geometry.

        Parameters
        ----------
        geometry : GeoDataFrame, GeoSeries, or Shapely Geometry
            Geometry whose bounding box defines the crop extent.

        Returns
        -------
        Batch
            Cropped batch with same type as self.

        Examples
        --------
        # Crop to AOI bounding box
        cropped = velocity.crop(AOI.buffer(500))

        # Compare with mask (exact geometry)
        masked = velocity.mask(AOI.buffer(500))  # clips to exact shape
        cropped = velocity.crop(AOI.buffer(500))  # crops to bounding rectangle
        """
        import geopandas as gpd
        from shapely import Geometry

        # Extract bounds from geometry
        if isinstance(geometry, gpd.GeoDataFrame):
            bounds = geometry.total_bounds  # (minx, miny, maxx, maxy)
            geom_crs = geometry.crs
        elif isinstance(geometry, gpd.GeoSeries):
            bounds = geometry.total_bounds
            geom_crs = geometry.crs
        elif isinstance(geometry, Geometry):
            bounds = geometry.bounds  # (minx, miny, maxx, maxy)
            geom_crs = None
        else:
            raise TypeError(f"geometry must be GeoDataFrame, GeoSeries, or Shapely Geometry, got {type(geometry).__name__}")

        minx, miny, maxx, maxy = bounds

        # Reproject bounds to batch CRS if needed
        crs = self.crs
        if crs is not None and geom_crs is not None and geom_crs != crs:
            from shapely.geometry import box
            bbox = gpd.GeoSeries([box(minx, miny, maxx, maxy)], crs=geom_crs).to_crs(crs)
            minx, miny, maxx, maxy = bbox.total_bounds

        out = {}
        for key, ds in self.items():
            # Determine coordinate order (ascending or descending)
            y_asc = len(ds.y) < 2 or float(ds.y[1]) > float(ds.y[0])
            x_asc = len(ds.x) < 2 or float(ds.x[1]) > float(ds.x[0])

            # Create slices based on coordinate order
            y_slice = slice(miny, maxy) if y_asc else slice(maxy, miny)
            x_slice = slice(minx, maxx) if x_asc else slice(maxx, minx)

            clipped = ds.sel(y=y_slice, x=x_slice)
            if clipped.y.size > 0 and clipped.x.size > 0:
                out[key] = clipped

        return type(self)(out)

    def __pow__(self, exponent, **kwargs):
        return self.map_da(lambda da: da**exponent, **kwargs)

    def power(self, **kwargs):
        """ element-wise |x|², i.e. signal intensity """
        return self.map_da(lambda da: xr.ufuncs.abs(da)**2, **kwargs)

    # def abs(self):
    #     """ element-wise absolute value """
    #     return type(self)({k: ds.map(lambda da: da.abs()) for k, ds in self.items()})

    # def sqrt(self):
    #     """ element-wise square-root """
    #     return type(self)({k: ds.map(lambda da: da.sqrt()) for k, ds in self.items()})

    # def square(self):
    #     """ element-wise square """
    #     return type(self)({k: ds.map(lambda da: da**2) for k, ds in self.items()})

    # def clip(self, min_, max_):
    #     """ element-wise clip to [min_, max_] """
    #     return type(self)({k: ds.map(lambda da: da.clip(min_, max_)) for k, ds in self.items()})

    # def where(self, cond, other=np.nan):
    #     """
    #     like xarray.where: keep ds where cond is True, else fill with other.
    #     `cond` may be a scalar, a DataArray, or another Batch with the same keys.
    #     """
    #     if isinstance(cond, Batch):
    #         return type(self)({
    #             k: ds.where(cond[k], other)
    #             for k, ds in self.items()
    #         })
    #     else:
    #         return type(self)({
    #             k: ds.where(cond, other)
    #             for k, ds in self.items()
    #         })

    # def isfinite(self):
    #     """ element-wise finite mask """
    #     return type(self)({k: ds.map(lambda da: np.isfinite(da)) for k, ds in self.items()})

    # def sel(self, keys: dict|list|str):
    #     if isinstance(keys, str):
    #         keys = [keys]
    #     return type(self)({k: self[k] for k in (keys if isinstance(keys, list) else keys.keys())})

    def sel(self, keys: dict|list|str|pd.DataFrame|None = None, **indexers):
        """
        Select data by burst keys or coordinate values.

        Parameters
        ----------
        keys : str, list, dict, DataFrame, or None
            - str: Single burst key to select
            - list: List of burst keys to select
            - dict/Batch: Align dimensions between batches
            - DataFrame: Complex filtering by dates/polarizations
            - None: Use only keyword indexers
        **indexers : slice or value
            Coordinate-based selection applied to each dataset.
            Example: x=slice(650_000, 700_000), y=slice(4_100_000, 4_150_000)

        Returns
        -------
        Batch
            New Batch with selected data.

        Examples
        --------
        Select by burst keys:
        >>> subset = batch.sel(['burst1', 'burst2'])

        Select by spatial coordinates:
        >>> subset = batch.sel(x=slice(650_000, 700_000))
        >>> subset = batch.sel(x=slice(650_000, 700_000), y=slice(4_100_000, 4_150_000))

        Combine both:
        >>> subset = batch.sel(['burst1'], x=slice(650_000, 700_000))
        """
        import pandas as pd
        import numpy as np

        # Handle coordinate-based selection via keyword indexers
        if indexers:
            result = self if keys is None else self
            # First apply key selection if provided
            if keys is not None:
                result = result.sel(keys)

            # Convert slices to index-based selection (fast, order-agnostic)
            def select_with_slices(ds, indexers):
                for dim, idx in indexers.items():
                    if dim not in ds.coords:
                        continue
                    if isinstance(idx, slice):
                        coord_vals = ds.coords[dim].values
                        # Get bounds from slice, use coord min/max as defaults
                        start = idx.start if idx.start is not None else coord_vals.min()
                        stop = idx.stop if idx.stop is not None else coord_vals.max()
                        min_val, max_val = min(start, stop), max(start, stop)
                        # Find indices within range (order-agnostic)
                        mask = (coord_vals >= min_val) & (coord_vals <= max_val)
                        indices = np.where(mask)[0]
                        if len(indices) > 0:
                            # Use isel with index slice (fast)
                            ds = ds.isel({dim: slice(indices[0], indices[-1] + 1)})
                        else:
                            # No matching coordinates - return empty
                            ds = ds.isel({dim: slice(0, 0)})
                    else:
                        # Non-slice indexer (exact value, list, etc.)
                        ds = ds.sel({dim: idx})
                return ds

            # Apply coordinate selection to each dataset
            # Bursts outside the range get NaN-filled with the target coordinates
            out = {}
            target_coords = {}  # Will store target y/x from first non-empty result

            for k, ds in result.items():
                selected = select_with_slices(ds, indexers)
                if all(selected.sizes[d] > 0 for d in ('y', 'x') if d in selected.sizes):
                    out[k] = selected
                    # Capture target coordinates from first non-empty result
                    if not target_coords:
                        for dim in ('y', 'x'):
                            if dim in selected.coords:
                                target_coords[dim] = selected.coords[dim].values

            # If no burst had data, compute target coords from slice bounds and spacing
            if not target_coords and result:
                sample_ds = next(iter(result.values()))
                for dim in ('y', 'x'):
                    if dim in indexers and isinstance(indexers[dim], slice) and dim in sample_ds.coords:
                        coord_vals = sample_ds.coords[dim].values
                        # Get spacing from original data
                        spacing = abs(coord_vals[1] - coord_vals[0]) if len(coord_vals) > 1 else 1
                        # Get bounds from slice
                        start = indexers[dim].start if indexers[dim].start is not None else coord_vals.min()
                        stop = indexers[dim].stop if indexers[dim].stop is not None else coord_vals.max()
                        min_val, max_val = min(start, stop), max(start, stop)
                        # Create target coordinates
                        is_descending = len(coord_vals) > 1 and coord_vals[0] > coord_vals[-1]
                        if is_descending:
                            target_coords[dim] = np.arange(max_val, min_val - spacing/2, -spacing)
                        else:
                            target_coords[dim] = np.arange(min_val, max_val + spacing/2, spacing)

            # Fill empty bursts with NaN using target coordinates
            if target_coords:
                for k, ds in result.items():
                    if k not in out:
                        # Reindex to target coords with NaN fill
                        out[k] = ds.reindex(**target_coords, fill_value=np.nan)

            return type(result)(out)

        # Original key-based selection logic
        if keys is None:
            return self

        if not isinstance(keys, pd.DataFrame):
            if isinstance(keys, str):
                keys = [keys]
            if isinstance(keys, list):
                return type(self)({k: self[k] for k in keys})

            # keys is dict-like (e.g., BatchWrap, BatchUnit)
            # Select matching burst IDs and align dimensions (like 'pair') per key
            result = {}
            for k in keys.keys():
                if k not in self:
                    continue
                ds = self[k]
                other_ds = keys[k]

                # Align 'pair' dimension if both have it - use minimum size (positional indexing)
                if hasattr(other_ds, 'dims') and 'pair' in getattr(other_ds, 'dims', []):
                    if hasattr(ds, 'dims') and 'pair' in ds.dims:
                        n_pairs = min(ds.sizes['pair'], other_ds.sizes['pair'])
                        if n_pairs < ds.sizes['pair']:
                            ds = ds.isel(pair=slice(n_pairs))

                result[k] = ds
            return type(self)(result)

        dss = {}
        # iterate all burst groups (fullBurstID is the first index level)
        for id in keys.index.get_level_values(0).unique():
            if id not in self:
                continue
            # select all records for the current burst group
            records = keys[keys.index.get_level_values(0)==id]
            ds = self[id]
            
            # Detect dimension type: date for Stack-like, pair for Batch-like
            if 'date' in ds.dims:
                # Stack-like: filter by dates
                dates = records.startTime.values.astype(str)
                ds = ds.sel(date=dates)
            # For pair-based data, we just select the burst if it exists
            # (pair filtering is handled elsewhere or not needed for simple selection)
            
            # filter polarizations
            pols = records.polarization.unique()
            if len(pols) > 1:
                raise ValueError(f'ERROR: Inconsistent polarizations found for the same burst: {id}')
            elif len(pols) == 0:
                raise ValueError(f'ERROR: No polarizations found for the burst: {id}')
            pols = pols[0]
            if ',' in pols:
                pols = pols.split(',')
            if isinstance(pols, str):
                pols = [pols]
            count = 0
            if np.unique(pols).size < len(pols):
                raise ValueError(f'ERROR: defined polarizations {pols} are not unique.')
            if len([pol for pol in pols if pol in ds.data_vars]) < len(pols):
                raise ValueError(f'ERROR: defined polarizations {pols} are not available in the dataset: {id}')
            for pol in [pol for pol in ['VV', 'VH', 'HH', 'HV'] if pol in ds.data_vars]:
                if pol not in pols:
                    ds = ds.drop(pol)
                else:
                    count += 1
            if count == 0:
                raise ValueError(f'ERROR: No valid polarizations found for the burst: {id}')
            dss[id] = ds
        return type(self)(dss)

    # def isel(self, indices):
    #     """Select by integer locations (like xarray .isel)."""
    #     import numpy as np

    #     keys = list(self.keys())
    #     # allow a single integer, a list of ints, or a slice
    #     if isinstance(indices, (int, np.integer)):
    #         idxs = [indices]
    #     elif isinstance(indices, slice):
    #         idxs = list(range(*indices.indices(len(keys))))
    #     else:
    #         idxs = list(indices)
    #     selected = {keys[i]: self[keys[i]] for i in idxs }
    #     return type(self)(selected)

    # def isel(self, indices=None, **indexers):
    #     """
    #     Select by integer locations, either by a single positional index/slice
    #     (applied over the *keys* of the batch) OR by keyword dimension selectors
    #     (delegated to each xarray.Dataset.isel).
    #     """
    #     # xarray‐style keyword isel
    #     if indexers:
    #         return type(self)({
    #             k: ds.isel(**indexers)
    #             for k, ds in self.items()
    #         })

    #     # positional isel over the batch keys (old behavior)
    #     import numpy as np
    #     keys = list(self.keys())
    #     if indices is None:
    #         return type(self)(dict(self))  # no selection
    #     if isinstance(indices, (int, np.integer)):
    #         idxs = [indices]
    #     elif isinstance(indices, slice):
    #         idxs = list(range(*indices.indices(len(keys))))
    #     else:
    #         idxs = list(indices)
    #     return type(self)({
    #         keys[i]: self[keys[i]]
    #         for i in idxs
    #     })

    def isel(self, indices=None, **indexers):
        """
        Select by integer locations, either by:
        keyword dimension selectors (delegated to each xarray.Dataset.isel)
        a single positional index/slice/list over the *keys* of the batch
        (NEW) a single dict positional argument of dimension indexers
        """
        import numpy as np

        # dict as a keyword indexers
        if isinstance(indices, dict):
            indexers = indices
            indices = None

        # xarray‐style keyword isel (including dict-via-positional)
        if indexers:
            return type(self)({
                k: ds.isel(**indexers)
                for k, ds in self.items()
            })

        # fallback: positional isel over the batch keys (old behavior)
        keys = list(self.keys())
        if indices is None:
            # no selection, cast to dict to prevent special logic in the class constructor
            return type(self)(dict(self))
        if isinstance(indices, (int, np.integer)):
            idxs = [indices]
        elif isinstance(indices, slice):
            idxs = list(range(*indices.indices(len(keys))))
        else:
            idxs = list(indices)

        return type(self)({
            keys[i]: self[keys[i]]
            for i in idxs
        })

    @property
    def dims(self):
        return {k: self[k].dims for k in self.keys()}
    
    @property
    def coords(self):
        """Return a Batch of Coordinates for each dataset."""
        return type(self)({k: ds.coords.to_dataset() for k, ds in self.items()})

    def assign_coords(self, coords=None, **coords_kwargs):
        """
        Assign new coordinates to each dataset in the batch.
        Works like xarray.Dataset.assign_coords but handles batch operations.
        
        Parameters
        ----------
        coords : dict-like or Batch, optional
            Dictionary of coordinates to assign or Batch of coordinates
        **coords_kwargs : optional
            Coordinates to assign, specified as keyword arguments
        
        Returns
        -------
        Batch
            New batch with assigned coordinates
        """
        if coords is None:
            coords = {}
        coords = dict(coords, **coords_kwargs)
        
        # Check if any coord is a BatchCore - if so, we need per-burst assignment
        batch_coords = {name: coord for name, coord in coords.items() 
                       if isinstance(coord, tuple) and len(coord) == 2 
                       and isinstance(coord[1], BatchCore)}
        
        if batch_coords:
            # Per-burst coordinate assignment
            result = {}
            for key, ds in self.items():
                ds_coords = {}
                for name, coord in coords.items():
                    if name in batch_coords:
                        dims, batch = coord
                        # Get this burst's values from the batch
                        burst_ds = batch[key]
                        var_name = next(iter(burst_ds.data_vars))
                        data = burst_ds[var_name]
                        # Compute lazy arrays - coordinates should never be lazy
                        values = data.compute().values if hasattr(data.data, 'compute') else data.values
                        ds_coords[name] = (dims, values)
                    else:
                        ds_coords[name] = coord
                result[key] = ds.assign_coords(ds_coords)
            return type(self)(result)
        
        def process_coord(coord):
            if not isinstance(coord, tuple) or len(coord) != 2:
                return coord
                
            dims, data = coord
            
            # Handle DataArray directly
            if isinstance(data, xr.DataArray):
                values = data.values
                return xr.DataArray(values if data.ndim > 0 else np.array([values]), dims=dims)
            
            # Handle BatchComplex
            if isinstance(data, type(self)):
                first_ds = next(iter(data.values()))
                if isinstance(first_ds, xr.DataArray):
                    values = first_ds.values
                    return xr.DataArray(values if first_ds.ndim > 0 else np.array([values]), dims=dims)
                elif isinstance(first_ds, xr.Dataset):
                    coord_name = first_ds.dims[0]
                    values = first_ds.coords[coord_name].values
                    return xr.DataArray(values if not np.isscalar(values) else np.array([values]), dims=dims)
            
            # Handle objects with values attribute
            if hasattr(data, 'values'):
                values = data.values() if callable(data.values) else data.values
                if hasattr(values, '__iter__'):
                    values = next(iter(values))
                    if isinstance(values, xr.DataArray):
                        values = values.values
                values = np.asarray(values)
                return xr.DataArray(values if values.ndim > 0 else np.array([values]), dims=dims)
            
            # Handle array-like inputs
            values = np.asarray(data)
            return xr.DataArray(values if values.ndim > 0 else np.array([values]), dims=dims)
        
        # Get target dimension size from first dataset
        first_ds = next(iter(self.values()))
        target_size = first_ds.dims[list(coords.values())[0][0]]
        
        # Process coordinates
        processed_coords = {name: process_coord(coord) for name, coord in coords.items()}
        
        # Ensure consistent dimension sizes
        for name, coord in processed_coords.items():
            if coord.size != target_size:
                if coord.size == 1 and target_size == 2:
                    processed_coords[name] = xr.DataArray([coord.values[0], coord.values[0]], dims=coord.dims)
                else:
                    raise ValueError(f"Coordinate {name} has size {coord.size} but expected size {target_size}")
        
        return type(self)({
            k: ds.assign_coords(processed_coords)
            for k, ds in self.items()
        })

    def set_index(self, indexes=None, **indexes_kwargs):
        """
        Set Dataset index(es) for each dataset in the batch.
        Works like xarray.Dataset.set_index but handles batch operations.
        
        Parameters
        ----------
        indexes : dict-like or Batch, optional
            Dictionary of indexes to set or Batch of indexes
        **indexes_kwargs : optional
            Indexes to set, specified as keyword arguments
        
        Returns
        -------
        Batch
            New batch with set indexes
        """
        if indexes is None:
            indexes = {}
        indexes = dict(indexes, **indexes_kwargs)
        
        # Handle both dict and Batch inputs
        if isinstance(indexes, type(self)):
            return type(self)({
                k: ds.set_index(indexes[k])
                for k, ds in self.items()
                if k in indexes
            })
        else:
            return type(self)({
                k: ds.set_index(indexes)
                for k, ds in self.items()
            })

    def expand_dims(self, *args, **kw):
        return type(self)({k: ds.expand_dims(*args, **kw) for k, ds in self.items()})

    def drop_vars(self, names):
        """Return a new Batch with those data-vars removed from each dataset."""
        if isinstance(names, str):
            names = [names]
        return type(self)({
            k: ds.drop_vars(names)
            for k, ds in self.items()
        })

    def rename_vars(self, **kw):
        return type(self)({k: ds.rename_vars(**kw) for k, ds in self.items()})
    
    def rename(self, **kw):
        return type(self)({k: ds.rename(**kw) for k, ds in self.items()})

    def reindex(self, **kw):
        return type(self)({k: ds.reindex(**kw) for k, ds in self.items()})

    def interp(self, **kw):
        return type(self)({k: ds.interp(**kw) for k, ds in self.items()})

    def interp_like(self, other: Batch, **interp_kwargs):
        """Regrid each Dataset onto the coords of the *corresponding* Dataset in `other`."""
        return type(self)({k: ds.interp_like(other[k], **interp_kwargs) for k, ds in self.items() if k in other})

    def reindex_like(self, other: Batch, **reindex_kwargs):
        return type(self)({k: ds.reindex_like(other[k], **reindex_kwargs) for k, ds in self.items() if k in other})

    def transpose(self, *dims, **kw):
        return type(self)({k: ds.transpose(*dims, **kw) for k, ds in self.items()})

    def _agg(self, name: str, dim=None, **kwargs):
        """
        Internal helper for aggregation methods.
        If the target object's .<name>() accepts a `dim=` arg, we pass dim, otherwise we just call it without.
        """
        import inspect
        out = {}
        for key, obj in self.items():
            fn = getattr(obj, name)
            sig = inspect.signature(fn)
            if "dim" in sig.parameters:
                out[key] = fn(dim=dim, **kwargs)
            else:
                out[key] = fn(**kwargs)

        # filter out collapsed dimensions
        sample = next(iter(out.values()), None)
        dims = (sample.dims or []) if hasattr(sample, 'dims') else []
        chunks = {d: size for d, size in self.chunks.items() if d in dims}
        result = type(self)(out)
        if chunks:
            return result.chunk(chunks)
        return result

    def mean(self, dim=None, **kwargs):
        return self._agg("mean", dim=dim, **kwargs)

    def sum(self, dim=None, **kwargs):
        return self._agg("sum", dim=dim, **kwargs)

    def min(self, dim=None, **kwargs):
        return self._agg("min", dim=dim, **kwargs)

    def max(self, dim=None, **kwargs):
        return self._agg("max", dim=dim, **kwargs)

    def median(self, dim=None, **kwargs):
        return self._agg("median", dim=dim, **kwargs)

    def std(self, dim=None, **kwargs):
        return self._agg("std", dim=dim, **kwargs)

    def var(self, dim=None, **kwargs):
        return self._agg("var", dim=dim, **kwargs)

    def polyval(self, coeffs: dict[str, list | xr.DataArray], dim: str = 'x') -> BatchCore:
        """
        Evaluate polynomial coefficients for each burst.

        Applies xarray.polyval to evaluate polynomial corrections at each position
        along the specified dimension. Designed to work with polynomial coefficients
        returned by Stack.burst_polyfit.

        Parameters
        ----------
        coeffs : dict[str, list | xr.DataArray]
            Polynomial coefficients per burst. Can be either:
            - list[float]: [ramp, offset] for linear polynomial ramp*x + offset (single pair)
            - list[list[float]]: [[ramp0, offset0], [ramp1, offset1], ...] (multiple pairs)
            - list[float]: [offset0, offset1, ...] for degree=0 with multiple pairs
            - xr.DataArray: with 'degree' dimension [1, 0] (xarray.polyfit format)
            Following xarray.polyfit convention: highest degree first.
        dim : str, optional
            Coordinate dimension to evaluate polynomial on. Default is 'x' for range
            direction corrections.

        Returns
        -------
        BatchCore (or subclass)
            New batch with polynomial evaluated at each position. The result has
            the same structure as self, with polynomial values broadcast to match
            each dataset's shape.

        Examples
        --------
        >>> # Single pair: Estimate offsets and ramps
        >>> coeffs = Stack.burst_polyfit(intfs, degree=1)
        >>> corrections = intfs.polyval(coeffs, dim='x')
        >>> intfs_aligned = intfs - corrections

        >>> # Multiple pairs: coefficients are lists per pair
        >>> offsets = Stack.burst_polyfit(intfs_multi, degree=0)
        >>> # offsets = {'burst1': [off0, off1], 'burst2': [off0, off1]}
        >>> intfs_aligned = intfs_multi - offsets

        See Also
        --------
        xarray.polyval : Underlying polynomial evaluation function
        Stack.burst_polyfit : Function that produces compatible coefficients
        """
        result = {}
        for bid, ds in self.items():
            # Get a spatial variable (with y, x dims)
            spatial_vars = [v for v in ds.data_vars if 'y' in ds[v].dims and 'x' in ds[v].dims]
            sample_var = spatial_vars[0] if spatial_vars else list(ds.data_vars)[0]

            if bid not in coeffs:
                # No coefficients for this burst - zero correction
                result[bid] = xr.zeros_like(ds[sample_var]).to_dataset(name=sample_var)
                continue

            # Get coordinate for evaluation
            coord = ds.coords[dim]
            sample_da = ds[sample_var]

            coeff = coeffs[bid]

            # Check if we have per-pair coefficients (list of lists or list of scalars for multiple pairs)
            has_pair_dim = 'pair' in sample_da.dims
            n_pairs = sample_da.sizes.get('pair', 1)

            if isinstance(coeff, (list, tuple)) and len(coeff) > 0:
                first_elem = coeff[0]

                # Detect format:
                # - Single pair degree=1: [ramp, offset] where both are scalars
                # - Single pair degree=0: scalar (but wrapped in list by caller)
                # - Multi pair degree=0: [off0, off1, ...] list of scalars
                # - Multi pair degree=1: [[ramp0, off0], [ramp1, off1], ...] list of lists

                if isinstance(first_elem, (list, tuple)):
                    # Multi-pair degree=1: [[ramp0, off0], [ramp1, off1], ...]
                    corrections = []
                    for pair_coeff in coeff:
                        corr = pair_coeff[0] * coord + pair_coeff[1]
                        corrections.append(corr)
                    # Stack along pair dimension
                    correction = xr.concat(corrections, dim='pair')

                elif has_pair_dim and len(coeff) == n_pairs and not isinstance(first_elem, (list, tuple)):
                    # Multi-pair degree=0: [off0, off1, ...] - all scalars matching pair count
                    # Check if it looks like [ramp, offset] for single pair (2 elements, no pair dim wouldn't reach here)
                    if len(coeff) == 2 and not has_pair_dim:
                        # Single pair degree=1: [ramp, offset]
                        correction = coeff[0] * coord + coeff[1]
                    else:
                        # Multi-pair degree=0
                        corrections = [xr.full_like(coord, off, dtype=float) for off in coeff]
                        correction = xr.concat(corrections, dim='pair')

                else:
                    # Single pair degree=1: [ramp, offset]
                    correction = coeff[0] * coord + coeff[1]

            elif isinstance(coeff, xr.DataArray):
                # General case using xr.polyval
                correction = xr.polyval(coord, coeff)
            else:
                # Single scalar (degree=0, single pair)
                correction = xr.full_like(coord, float(coeff), dtype=float)

            result[bid] = correction.to_dataset(name=sample_var)

        return type(self)(result)

    # def coarsen(self, window: dict[str,int], **kwargs):
    #     """
    #     intfs.coarsen({'y':2, 'x':8}, boundary='trim').mean().isel(0)
    #     """
    #     return type(self)({
    #         k: ds.coarsen(window, **kwargs)
    #         for k, ds in self.items()
    #     })

    def coarsen(self, window: dict[str, int], **kwargs) -> Batch:
        """
        Coarsen each DataSet in the batch by integer factors and align the 
        blocks so that they fall on "nice" grid boundaries.

        Parameters
        ----------
        window : dict[str,int]
            e.g. {'y': 2, 'x': 8}
        **kwargs
            extra args forwarded into the reduction, e.g. skipna=True.

        Returns
        -------
        Batch
            A new Batch where each Dataset has been sliced for alignment,
            coarsened by `window`, then reduced by `.mean()` (or whichever
            `func` you chose).
        """
        chunks = self.chunks
        out = {}
        # produce unified grid and chunks for all datasets in the batch
        for key, ds in self.items():
            # align each dimension
            for dim, factor in window.items():
                start = utils_xarray.coarsen_start(ds, dim, factor)
                #print ('start', start)
                if start is not None:
                    # rechunk to the original chunk sizes
                    ds = ds.isel({dim: slice(start, None)}).chunk(chunks)
                    # or allow a bit different chunks for coarsening
                    #ds = ds.isel({dim: slice(start, None)})
            # coarsen and revert original chunks
            out[key] = ds.coarsen(window, **kwargs)

        return type(self)(out)

    def chunk(self, chunks):
        return type(self)({k: ds.chunk(chunks) for k, ds in self.items()})

    def pipe(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)

    def map(self, func, *args, **kwargs):
        return type(self)({k: func(ds, *args, **kwargs) for k, ds in self.items()})

    def to_dict(self) -> dict:
        """
        Extract data variables as a dictionary of {burst_key: {var_name: values}}.

        Returns
        -------
        dict
            Dictionary mapping burst keys to dictionaries of variable names to numpy arrays.

        Examples
        --------
        >>> bpr = stack.BPR(stack.isel(date=[1]), stack.isel(date=[0]))
        >>> bpr.to_dict()
        {'123_262885_IW2': {'BPR': array([-158.75])},
         '123_262886_IW2': {'BPR': array([-158.81])},
         '123_262887_IW2': {'BPR': array([-158.87])}}
        """
        result = {}
        for key, ds in self.items():
            result[key] = {var: ds[var].values for var in ds.data_vars}
        return result

    def compute(self):
        import dask
        from insardev_toolkit.progressbar import progressbar
        progressbar(result := dask.persist(dict(self))[0], desc=f'Computing Batch...'.ljust(25))
        # Ensure coordinates are also computed (not lazy dask arrays)
        computed = {}
        for key, ds in result.items():
            # Compute any lazy coordinates, preserving their dims
            new_coords = {}
            for name, coord in ds.coords.items():
                if hasattr(coord, 'data') and hasattr(coord.data, 'compute'):
                    # Preserve original dims when assigning computed values
                    new_coords[name] = (coord.dims, coord.compute().values)
            if new_coords:
                ds = ds.assign_coords(new_coords)
            computed[key] = ds
        return type(self)(computed)

    def to_dataframe(self,
                     crs: str | int | None = 'auto',
                     debug: bool = False) -> pd.DataFrame:
        """
        Return a Pandas/GeoPandas DataFrame for all Batch scenes.
        
        Extracts attributes from each Dataset in the Batch (from .attrs or dim-indexed data vars)
        and combines them into a single DataFrame, matching the Stack.to_dataframe format
        with additional ref/rep columns for pair information.

        Parameters
        ----------
        crs : str | int | None, optional
            Coordinate reference system for the output GeoDataFrame.
            If 'auto', uses CRS from the data. If None, returns without CRS conversion.
        debug : bool, optional
            Print debug information. Default is False.

        Returns
        -------
        pandas.DataFrame or geopandas.GeoDataFrame
            The DataFrame containing Batch scenes with their attributes.
            Index is (fullBurstID, burst) matching Stack.to_dataframe.
            For pair-based data, ref and rep columns are added after the index.

        Examples
        --------
        >>> df = batch.to_dataframe()
        >>> df = batch.to_dataframe(crs=4326)
        """
        import geopandas as gpd
        from shapely import wkt
        import pandas as pd

        if not self:
            return pd.DataFrame()

        # Detect CRS from data if auto
        if crs is not None and isinstance(crs, str) and crs == 'auto':
            sample = next(iter(self.values()))
            crs = sample.attrs.get('crs', 4326)

        # Detect polarizations
        sample = next(iter(self.values()))
        polarizations = [pol for pol in ['VV', 'VH', 'HH', 'HV'] if pol in sample.data_vars]

        # Detect dimension: 'date' for BatchComplex, 'pair' for others
        dim = 'date' if 'date' in sample.dims else 'pair'

        # Define the attribute order matching Stack.to_dataframe
        # Order: fullBurstID, burst, startTime, polarization, flightDirection, pathNumber, subswath, mission, beamModeType, BPR, geometry
        attr_order = ['fullBurstID', 'burst', 'startTime', 'polarization', 'flightDirection', 
                      'pathNumber', 'subswath', 'mission', 'beamModeType', 'BPR', 'geometry']

        # Make attributes dataframe from data
        processed_attrs = []
        for key, ds in self.items():
            for idx in range(ds.dims[dim]):
                processed_attr = {}
                
                # Get ref/rep for pair dimension
                if dim == 'pair':
                    if 'ref' in ds.coords:
                        processed_attr['ref'] = pd.Timestamp(ds['ref'].values[idx])
                    if 'rep' in ds.coords:
                        processed_attr['rep'] = pd.Timestamp(ds['rep'].values[idx])
                else:
                    processed_attr['date'] = pd.Timestamp(ds[dim].values[idx])
                
                # Extract attributes from ds.attrs
                for attr_name in attr_order:
                    if attr_name in ds.attrs and attr_name not in processed_attr:
                        value = ds.attrs[attr_name]
                        if attr_name == 'geometry' and isinstance(value, str):
                            processed_attr[attr_name] = wkt.loads(value)
                        elif attr_name == 'startTime':
                            processed_attr[attr_name] = pd.Timestamp(value)
                        else:
                            processed_attr[attr_name] = value
                
                processed_attrs.append(processed_attr)

        if not processed_attrs:
            return pd.DataFrame()

        # Check if we have geometry column for GeoDataFrame
        has_geometry = 'geometry' in processed_attrs[0]
        
        if has_geometry:
            df = gpd.GeoDataFrame(processed_attrs, crs=4326)
        else:
            df = pd.DataFrame(processed_attrs)

        # Add polarization info if not already present
        if 'polarization' not in df.columns and polarizations:
            df['polarization'] = ','.join(map(str, polarizations))

        # Round BPR for readability
        if 'BPR' in df.columns:
            df['BPR'] = df['BPR'].round(1)

        # Reorder columns to match Stack.to_dataframe format
        # For pair data: fullBurstID, burst (index), then ref, rep, then rest
        if 'fullBurstID' in df.columns and 'burst' in df.columns:
            # Build column order
            if dim == 'pair':
                # ref, rep first after index, then startTime, polarization, etc.
                first_cols = ['fullBurstID', 'burst', 'ref', 'rep']
            else:
                first_cols = ['fullBurstID', 'burst', 'date']
            
            # Rest of columns in attr_order, excluding index columns and ref/rep/date
            other_cols = [c for c in attr_order if c not in first_cols and c in df.columns]
            
            # Reorder
            ordered_cols = [c for c in first_cols if c in df.columns] + other_cols
            df = df[ordered_cols]
            
            # Sort and set index
            df = df.sort_values(by=['fullBurstID', 'burst']).set_index(['fullBurstID', 'burst'])
            
            # Move geometry to end if present
            if has_geometry and 'geometry' in df.columns:
                df = df.loc[:, df.columns.drop("geometry").tolist() + ["geometry"]]

        # Convert CRS if requested and we have a GeoDataFrame
        if has_geometry and crs is not None:
            return df.to_crs(crs)
        return df
    
    def persist(self):
        return type(self)({
            k: ds.chunk(ds.chunks).persist()
            for k, ds in self.items()
        })

    @property
    def spacing(self) -> tuple[float, float]:
        """Return the (y, x) grid spacing."""
        sample = next(iter(self.values()))
        return sample.y.diff('y').item(0), sample.x.diff('x').item(0)
    
    def downsample(self, new_spacing: tuple[float, float] | float | int, debug: bool = False):
        """
        Update the Batch data onto a grid with the given (y, x) spacing.
        Like to coarsening but with cell size in meters instead of pixels:
        intfs.downsample(60)
        intfs.coarsen({'y':2, 'x':2}, boundary='trim').mean()

        If the data is already at or finer than the requested spacing,
        returns the input unchanged.
        """
        if isinstance(new_spacing, (int, float)):
            new_spacing = (new_spacing, new_spacing)
        dy, dx = self.spacing
        yscale, xscale = max(1, int(np.round(new_spacing[0]/dy))), max(1, int(np.round(new_spacing[1]/dx)))
        # If both scale factors are 1, no downsampling needed - return as is
        if yscale == 1 and xscale == 1:
            return self
        if debug:
            print (f'DEBUG: cell size in meters: y={dy:.1f}, x={dx:.1f} -> y={new_spacing[0]:.1f}, x={new_spacing[1]:.1f}')
        return self.coarsen({'y': yscale, 'x': xscale}, boundary='trim').mean()

    def save(self, store: str, storage_options: dict[str, str] | None = None,
                caption: str | None = 'Saving...', n_jobs: int = -1, debug=False):
        return utils_io.save(self, store=store, storage_options=storage_options, compat=False, caption=caption, n_jobs=n_jobs, debug=debug)

    def open(self, store: str, storage_options: dict[str, str] | None = None, n_jobs: int = -1, debug=False):
        data = utils_io.open(store=store, storage_options=storage_options, compat=False, n_jobs=n_jobs, debug=debug)
        if not isinstance(data, dict):
            raise ValueError(f'ERROR: open() returns multiple datasets, you need to use Stack class to open them.')
        return data
    
    def snapshot(self, store: str | None = None, storage_options: dict[str, str] | None = None,
                caption: str | None = 'Snapshotting...', n_jobs: int = -1, debug=False):
        # Only save if this batch has data; otherwise just open existing store
        if len(self) > 0:
            self.save(store=store, storage_options=storage_options, caption=caption, n_jobs=n_jobs, debug=debug)
        return utils_io.open(store=store, storage_options=storage_options, compat=False, n_jobs=n_jobs, debug=debug)

    def to_dataset(self, polarization=None, chunks='auto', compute: bool = False, debug: bool = False):
        """
        Merge multiple burst DataArrays into a single unified grid.

        This function efficiently combines bursts using dask delayed operations
        for lazy evaluation. For each output chunk, it selects the minimal set
        of input bursts needed and combines their data using forward-fill.

        For best results with overlapping bursts, call .dissolve() first to average
        values in overlap regions, then call .to_dataset() to merge into a single grid.

        Parameters
        ----------
        polarization : str, optional
            Specific polarization to process. If None, processes all polarizations.
        chunks : str, int, or tuple, optional
            Spatial chunk size for processing. Options:
            - 'auto' (default): automatically determine chunk size based on memory
            - int: use same chunk size for both y and x dimensions
            - tuple (y_chunk, x_chunk): explicit chunk sizes for each dimension
            Note: Only 2D spatial chunks are supported. The stack dimension
            (date/pair) is always processed one slice at a time.
        compute : bool, optional
            Whether to compute the result immediately. Default is False (lazy).
        debug : bool, optional
            Print debugging/profiling information. Default is False.

        Returns
        -------
        xr.DataArray or xr.Dataset
            Merged data on a unified grid.

        Examples
        --------
        >>> # Merge bursts into single grid (fast, uses ffill for overlaps)
        >>> merged = batch.to_dataset()
        >>>
        >>> # With explicit chunk size for memory-constrained environments
        >>> merged = batch.to_dataset(chunks=1024)
        >>>
        >>> # With different y/x chunk sizes
        >>> merged = batch.to_dataset(chunks=(512, 2048))
        >>>
        >>> # For smooth overlaps, dissolve first then merge
        >>> merged = batch.dissolve().to_dataset()
        >>>
        >>> # Debug mode to see chunk sizes and burst distribution
        >>> merged = batch.to_dataset(debug=True)
        """
        import xarray as xr
        import numpy as np
        import dask
        import dask.array as da
        from insardev_toolkit import progressbar, datagrid

        if not len(self):
            return None

        sample = next(iter(self.values()))
        if len(self) == 1:
            if debug:
                print(f"=== to_dataset() debug ===")
                print(f"Single burst - returning directly (no merge needed)")
                print(f"Burst key: {next(iter(self.keys()))}")
                # Get spatial vars
                spatial_vars = [v for v in sample.data_vars if 'y' in sample[v].dims and 'x' in sample[v].dims]
                for var in spatial_vars[:3]:  # Show first 3 spatial vars
                    da = sample[var]
                    print(f"  {var}: shape={da.shape}, dtype={da.dtype}")
            if compute:
                progressbar(sample := sample.persist(), desc=f'Compute Dataset'.ljust(25))
                return sample
            return sample

        # Determine which polarizations to process
        if polarization is None:
            # Filter for spatial variables (with y, x dims) - excludes converted attributes
            polarizations = [v for v in sample.data_vars
                            if 'y' in sample[v].dims and 'x' in sample[v].dims]
        else:
            polarizations = [polarization]

        # Build data dictionary: {pol: [burst0_data, burst1_data, ...]}
        burst_keys = list(self.keys())
        datas_by_pol = {pol: [self[k][pol] for k in burst_keys] for pol in polarizations}

        # Get grid info from first polarization (all pols have same grid)
        first_pol = polarizations[0]
        first_datas = datas_by_pol[first_pol]

        # Define unified grid from all burst extents
        y_min = min(ds.y.min().item() for ds in first_datas)
        y_max = max(ds.y.max().item() for ds in first_datas)
        x_min = min(ds.x.min().item() for ds in first_datas)
        x_max = max(ds.x.max().item() for ds in first_datas)
        dims = first_datas[0].dims
        stackvar = list(dims)[0] if len(dims) > 2 else None

        # Handle stack dimension - preserve original coordinate values
        if stackvar is not None:
            stackval = first_datas[0][stackvar].values
        else:
            stackvar = 'fake'
            stackval = [0]
            # Expand dims for all polarizations
            for pol in polarizations:
                datas_by_pol[pol] = [d.expand_dims({stackvar: [0]}) for d in datas_by_pol[pol]]
            first_datas = datas_by_pol[first_pol]

        n_stack = len(stackval)
        dy = first_datas[0].y.diff('y').item(0)
        dx = first_datas[0].x.diff('x').item(0)
        ys = np.arange(y_min, y_max + dy/2, dy)
        xs = np.arange(x_min, x_max + dx/2, dx)

        fill_dtype = first_datas[0].dtype

        # Determine chunk sizes for spatial dimensions
        # Stack dimension is always processed one slice at a time (chunked to 1)
        if chunks == 'auto':
            # Use complex128 (16 bytes) to account for memory overhead:
            # output chunk + 2-4 overlapping input bursts to load and merge
            auto_chunks = dask.array.core.normalize_chunks('auto', (ys.size, xs.size), dtype=np.complex128)
            y_chunk_size = auto_chunks[0][0] if auto_chunks[0] else ys.size
            x_chunk_size = auto_chunks[1][0] if auto_chunks[1] else xs.size
        elif isinstance(chunks, (int, np.integer)):
            # Single int: use same size for both dimensions
            y_chunk_size = min(int(chunks), ys.size)
            x_chunk_size = min(int(chunks), xs.size)
        elif isinstance(chunks, (tuple, list)) and len(chunks) == 2:
            # Tuple/list of (y_chunk, x_chunk)
            y_chunk_size = min(int(chunks[0]), ys.size)
            x_chunk_size = min(int(chunks[1]), xs.size)
        else:
            raise ValueError(
                f"chunks must be 'auto', int, or 2-tuple (y, x), got {type(chunks).__name__}: {chunks}. "
                "Note: 3D chunks are not supported; stack dimension is always processed per-slice."
            )

        # Number of chunks in each dimension
        n_y_chunks = (ys.size + y_chunk_size - 1) // y_chunk_size
        n_x_chunks = (xs.size + x_chunk_size - 1) // x_chunk_size

        # Extract extents and build spatial index for O(1) chunk lookup
        burst_info = []
        # chunk_index[(yi, xi)] = list of (burst_idx, coverage) sorted by coverage desc
        from collections import defaultdict
        chunk_index = defaultdict(list)

        for burst_idx, d in enumerate(first_datas):
            info = {
                'y_min': float(d.y.min()),
                'y_max': float(d.y.max()),
                'x_min': float(d.x.min()),
                'x_max': float(d.x.max()),
                'y_coords': d.y.values,
                'x_coords': d.x.values,
            }
            burst_info.append(info)

            # Which output chunks does this burst overlap?
            # Convert burst extent to chunk indices
            y_chunk_start = max(0, int((info['y_min'] - y_min) / (dy * y_chunk_size)))
            y_chunk_end = min(n_y_chunks, int((info['y_max'] - y_min) / (dy * y_chunk_size)) + 1)
            x_chunk_start = max(0, int((info['x_min'] - x_min) / (dx * x_chunk_size)))
            x_chunk_end = min(n_x_chunks, int((info['x_max'] - x_min) / (dx * x_chunk_size)) + 1)

            # Register this burst for all overlapping chunks
            for yi in range(y_chunk_start, y_chunk_end):
                for xi in range(x_chunk_start, x_chunk_end):
                    # Compute coverage for this chunk
                    yb0 = yi * y_chunk_size
                    yb1 = min(yb0 + y_chunk_size, ys.size)
                    xb0 = xi * x_chunk_size
                    xb1 = min(xb0 + x_chunk_size, xs.size)
                    y0, y1 = ys[yb0], ys[yb1-1]
                    x0, x1 = xs[xb0], xs[xb1-1]
                    y_overlap = min(y1, info['y_max']) - max(y0, info['y_min'])
                    x_overlap = min(x1, info['x_max']) - max(x0, info['x_min'])
                    coverage = max(0, y_overlap / dy) * max(0, x_overlap / dx)
                    chunk_index[(yi, xi)].append((burst_idx, coverage))

        # Sort each chunk's burst list by coverage (descending)
        for key in chunk_index:
            chunk_index[key].sort(key=lambda x: -x[1])

        # Debug output
        if debug:
            print(f"=== to_dataset() debug ===")
            print(f"Number of bursts: {len(first_datas)}")
            print(f"Output grid: y={ys.size}, x={xs.size} (total {ys.size * xs.size:,} pixels)")
            print(f"Chunk size: y={y_chunk_size}, x={x_chunk_size} ({y_chunk_size * x_chunk_size:,} pixels/chunk)")
            print(f"Number of chunks: {n_y_chunks} y × {n_x_chunks} x = {n_y_chunks * n_x_chunks} total")
            print(f"Stack dimension: {stackvar}={n_stack} slices")
            print(f"Data type: {fill_dtype} ({np.dtype(fill_dtype).itemsize} bytes/pixel)")

            # Memory estimate per chunk
            chunk_mem_mb = y_chunk_size * x_chunk_size * np.dtype(fill_dtype).itemsize / 1024 / 1024
            print(f"Memory per chunk: ~{chunk_mem_mb:.1f} MB")

            # Burst distribution across chunks
            bursts_per_chunk = [len(chunk_index.get((yi, xi), []))
                               for yi in range(n_y_chunks) for xi in range(n_x_chunks)]
            if bursts_per_chunk:
                # Show distribution histogram
                from collections import Counter
                dist = Counter(bursts_per_chunk)
                dist_str = ", ".join(f"{k} bursts: {v}" for k, v in sorted(dist.items()))
                print(f"Chunk distribution: {dist_str}")

            # Burst extents
            print(f"Burst extents:")
            for idx, info in enumerate(burst_info):
                ny = len(info['y_coords'])
                nx = len(info['x_coords'])
                print(f"  [{idx}] y=[{info['y_min']:.1f}, {info['y_max']:.1f}] "
                      f"x=[{info['x_min']:.1f}, {info['x_max']:.1f}] ({ny}×{nx} pixels)")

        def merge_tiles_numpy(tiles_with_offsets, out_shape, fill_dtype):
            """Merge overlapping tiles using forward-fill (NaN-aware).

            Args:
                tiles_with_offsets: list of (tile_data, y_offset, x_offset)
                    where offsets are positions in output array
                out_shape: (out_ny, out_nx) output shape
                fill_dtype: output dtype

            Returns:
                merged (out_ny, out_nx) array
            """
            out = np.full(out_shape, np.nan, dtype=fill_dtype)
            for tile, y_off, x_off in tiles_with_offsets:
                # Compute valid region considering both tile size and output bounds
                ny, nx = tile.shape
                y_end = min(y_off + ny, out_shape[0])
                x_end = min(x_off + nx, out_shape[1])
                tile_y_end = y_end - y_off
                tile_x_end = x_end - x_off
                # Forward-fill: only write to NaN positions
                view = out[y_off:y_end, x_off:x_end]
                tile_view = tile[:tile_y_end, :tile_x_end]
                mask = np.isnan(view)
                view[mask] = tile_view[mask]
            return out

        # Build result for each polarization
        results = {}
        for pol in polarizations:
            datas = datas_by_pol[pol]

            # Rechunk all bursts to consistent spatial chunks for efficient block access
            # Stack dimension chunked to 1 for per-date processing
            rechunked_datas = []
            for d in datas:
                # xarray .chunk() works for both numpy and dask backed arrays
                rechunked = d.chunk({d.dims[0]: 1, 'y': y_chunk_size, 'x': x_chunk_size}).data
                rechunked_datas.append(rechunked)

            # Build output blocks for each stack slice separately
            stack_mosaics = []
            for s_idx in range(n_stack):
                # Extract this stack slice from all bursts
                slice_arrays = [rd[s_idx] for rd in rechunked_datas]

                # Build 2D mosaic for this stack slice
                blocks_rows = []
                for yi in range(n_y_chunks):
                    yb0 = yi * y_chunk_size
                    yb1 = min(yb0 + y_chunk_size, ys.size)
                    blocks_row = []
                    for xi in range(n_x_chunks):
                        xb0 = xi * x_chunk_size
                        xb1 = min(xb0 + x_chunk_size, xs.size)

                        # O(1) lookup of overlapping bursts via spatial index
                        overlapping = chunk_index.get((yi, xi), [])

                        out_shape = (yb1 - yb0, xb1 - xb0)

                        if len(overlapping) == 0:
                            # No data - create NaN block
                            block = da.full(out_shape, np.nan, dtype=fill_dtype)
                        else:
                            # Collect delayed tiles with their offsets in output chunk
                            tiles_info = []  # list of (delayed_tile, y_offset, x_offset)

                            # Output chunk coordinate bounds
                            out_y0, out_y1 = ys[yb0], ys[yb1-1]
                            out_x0, out_x1 = xs[xb0], xs[xb1-1]

                            for burst_idx, _ in overlapping:
                                info = burst_info[burst_idx]
                                arr = slice_arrays[burst_idx]
                                burst_ys = info['y_coords']
                                burst_xs = info['x_coords']

                                # Find burst indices that fall within output chunk
                                # Use searchsorted for robust coordinate matching
                                burst_y0 = np.searchsorted(burst_ys, out_y0 - dy/2)
                                burst_y1 = np.searchsorted(burst_ys, out_y1 + dy/2)
                                burst_x0 = np.searchsorted(burst_xs, out_x0 - dx/2)
                                burst_x1 = np.searchsorted(burst_xs, out_x1 + dx/2)

                                if burst_y1 > burst_y0 and burst_x1 > burst_x0:
                                    # Compute offset: where in output chunk does this tile start?
                                    # Find where burst_ys[burst_y0] falls in output coordinates
                                    tile_y_in_out = np.searchsorted(ys[yb0:yb1], burst_ys[burst_y0] - dy/2)
                                    tile_x_in_out = np.searchsorted(xs[xb0:xb1], burst_xs[burst_x0] - dx/2)

                                    # Slice the burst array
                                    tile = arr[burst_y0:burst_y1, burst_x0:burst_x1]
                                    # Rechunk to single chunk to ensure to_delayed() returns one object
                                    # (slice may span multiple input chunks)
                                    if tile.npartitions > 1:
                                        tile = tile.rechunk(-1)
                                    tiles_info.append((tile.to_delayed().ravel()[0],
                                                      tile_y_in_out, tile_x_in_out))

                            if len(tiles_info) == 0:
                                block = da.full(out_shape, np.nan, dtype=fill_dtype)
                            elif len(tiles_info) == 1 and tiles_info[0][1] == 0 and tiles_info[0][2] == 0:
                                # Single tile at origin - check if it covers the whole chunk
                                tile_delayed, _, _ = tiles_info[0]
                                # Use merge even for single tile to handle size mismatches
                                block = da.from_delayed(
                                    dask.delayed(merge_tiles_numpy)(tiles_info, out_shape, fill_dtype),
                                    shape=out_shape,
                                    dtype=fill_dtype
                                )
                            else:
                                # Multiple tiles or offset - merge with forward-fill
                                block = da.from_delayed(
                                    dask.delayed(merge_tiles_numpy)(tiles_info, out_shape, fill_dtype),
                                    shape=out_shape,
                                    dtype=fill_dtype
                                )

                        blocks_row.append(block)
                    blocks_rows.append(blocks_row)

                # Assemble 2D mosaic for this stack slice
                mosaic_2d = da.block(blocks_rows)
                stack_mosaics.append(mosaic_2d[np.newaxis, :, :])

            # Stack all slices along axis 0
            data = da.concatenate(stack_mosaics, axis=0)

            result = xr.DataArray(data, coords={stackvar: stackval, 'y': ys, 'x': xs})\
                .rename(pol)\
                .assign_attrs(datas[0].attrs)
            # Preserve ref/rep coordinates along pair dimension
            if stackvar == 'pair':
                if 'ref' in datas[0].coords:
                    result = result.assign_coords(ref=(stackvar, datas[0].coords['ref'].values))
                if 'rep' in datas[0].coords:
                    result = result.assign_coords(rep=(stackvar, datas[0].coords['rep'].values))
            result = datagrid.spatial_ref(result, datas)
            if stackvar == 'fake':
                result = result.isel({stackvar: 0})
            results[pol] = result

        # Return DataArray if single polarization was requested, Dataset otherwise
        if polarization is not None:
            # Single polarization explicitly requested - return DataArray
            output = results[polarization]
        else:
            # All polarizations - return Dataset (even if only one)
            output = xr.merge(list(results.values()))

        if compute:
            progressbar(output := output.persist(), desc=f'Computing Dataset...'.ljust(25))
        return output

    def to_geojson(self, filename: str = None, crs: str = None, decimals: int = 3) -> str:
        """
        Convert batch data to GeoJSON with pixel rectangles.

        Creates a GeoJSON FeatureCollection where each pixel is represented
        as a polygon rectangle. All data variables (VV, VH, etc.) are preserved
        as properties in each feature. Coordinates are rounded to 6 digits.

        Parameters
        ----------
        filename : str, optional
            Path to save the GeoJSON file. If None (default), returns the
            GeoJSON string.
        crs : str, optional
            Target CRS (e.g., 'EPSG:4326'). If None (default), uses the data's
            original CRS.
        decimals : int, optional
            Number of decimal places for rounding values. Default is 3.

        Returns
        -------
        str or None
            GeoJSON string if filename is None, otherwise None (saves to file).

        Examples
        --------
        Save to file in WGS84:

        >>> velocity.to_geojson('velocity.geojson', crs='EPSG:4326')

        Save in original CRS:

        >>> velocity.to_geojson('velocity.geojson')

        Read from file:

        >>> import geopandas as gpd
        >>> gdf = gpd.read_file('velocity.geojson')

        Or get as string:

        >>> geojson_str = velocity.to_geojson()

        Or create GeoDataFrame from string:

        >>> import geopandas as gpd
        >>> gdf = gpd.read_file(velocity.to_geojson(), driver='GeoJSON')

        Or parse to dict:

        >>> import json
        >>> geojson = json.loads(velocity.to_geojson())
        """
        import geopandas as gpd
        import shapely.geometry

        # Merge to single dataset
        ds = self.to_dataset()

        # Get spatial data variables (with y, x dims) - excludes converted attributes
        data_vars = [v for v in ds.data_vars
                    if 'y' in ds[v].dims and 'x' in ds[v].dims]

        # Convert to dataframe and drop NaN rows
        df = ds.to_dataframe().dropna().reset_index()

        if df.empty:
            return None

        # Get pixel spacing
        dy, dx = self.spacing

        # Create rectangles in projected coordinates
        def point_to_rectangle(row, half_y, half_x):
            return shapely.geometry.Polygon([
                (row.x - half_x, row.y - half_y),
                (row.x + half_x, row.y - half_y),
                (row.x + half_x, row.y + half_y),
                (row.x - half_x, row.y + half_y)
            ])

        # Build GeoDataFrame with rectangles
        gdf = gpd.GeoDataFrame(
            df[['y', 'x'] + data_vars],
            geometry=[point_to_rectangle(row, abs(dy)/2, abs(dx)/2) for _, row in df.iterrows()],
            crs=self.crs
        )

        # Drop projected x, y columns and ensure column names are plain strings
        gdf = gdf.drop(columns=['x', 'y'])
        gdf.columns = [str(c) for c in gdf.columns]

        # Reproject if CRS specified, round values to 3 digits and coordinates to 6 digits
        if crs is not None:
            gdf = gdf.to_crs(crs)
        for col in data_vars:
            gdf[col] = gdf[col].astype(float).round(decimals)
        gdf.geometry = shapely.set_precision(gdf.geometry, grid_size=1e-6)

        # Add CRS for GIS software compatibility
        crs_urn = gdf.crs.to_string().replace(':', '::')
        crs_str = f'"crs": {{"type": "name", "properties": {{"name": "urn:ogc:def:crs:{crs_urn}"}}}}, '
        geojson = gdf.to_json(drop_id=True).replace('"type": "FeatureCollection", ', '"type": "FeatureCollection", ' + crs_str)

        if filename is not None:
            with open(filename, 'w') as f:
                f.write(geojson)
            return
        return geojson

    def to_vtk(self, path: str, mask: bool = True):
        """
        Export the batch data to VTK files.

        Converts the batch to a unified dataset using to_dataset(), then saves
        each data variable as a separate VTK file named {varname}.vtk in the
        specified directory.

        Parameters
        ----------
        path : str
            Output directory where VTK files will be saved.
        mask : bool, optional
            If True (default), set z-coordinate to NaN where data values are NaN,
            effectively masking those areas in the VTK visualization.

        Examples
        --------
        >>> intfs20.to_vtk('vtk')  # Creates vtk/VV.vtk, etc.
        >>> intfs20.to_vtk('vtk', mask=False)  # No masking
        """
        import os
        import numpy as np
        from vtk import vtkStructuredGridWriter, VTK_BINARY
        from .utils_vtk import as_vtk

        os.makedirs(path, exist_ok=True)

        ds = self.to_dataset()
        if ds is None:
            return

        # Handle both Dataset and DataArray
        if isinstance(ds, xr.DataArray):
            ds = ds.to_dataset()

        for varname in ds.data_vars:
            if varname == 'spatial_ref':
                continue
            # Select this variable and convert to dataset for as_vtk
            da = ds[varname]
            # Handle 3D data (pair, y, x) - select first pair for VTK
            if 'pair' in da.dims:
                da = da.isel(pair=0)
            
            # Create dataset with z-coordinate for VTK
            ds_var = da.to_dataset()
            if mask:
                # Set z to NaN where data is NaN to mask those areas
                z = xr.zeros_like(da, dtype=np.float32)
                z = z.where(np.isfinite(da))
                ds_var['z'] = z
            
            vtk_grid = as_vtk(ds_var)

            filename = os.path.join(path, f"{varname}.vtk")
            writer = vtkStructuredGridWriter()
            writer.SetFileName(filename)
            writer.SetInputData(vtk_grid)
            writer.SetFileType(VTK_BINARY)
            writer.Write()

    def plot(self,
            cmap: matplotlib.colors.Colormap | str | None = 'viridis',
            alpha: float = 0.7,
            vmin: float | None = None,
            vmax: float | None = None,
            quantile: float | None = None,
            symmetrical: bool = False,
            caption: str = '',
            cols: int = 4,
            rows: int = 4,
            size: float = 4,
            nbins: int = 5,
            aspect: float = 1.02,
            y: float = 1.05,
            flip: bool = False,
            _size: tuple[int, int] | None = None,
            ):
        import xarray as xr
        import numpy as np
        import pandas as pd
        import matplotlib.ticker as mticker
        from matplotlib.ticker import FuncFormatter
        import matplotlib.pyplot as plt
        from .Batch import BatchWrap
        from insardev_toolkit import progressbar

        # no data means no plot and no error
        if not len(self):
            return

        wrap = True if type(self) == BatchWrap else False

        # screen size in pixels (width, height) to estimate reasonable number pixels per plot
        # this is quite large to prevent aliasing on 600dpi plots without additional processing
        if _size is None:
            _size = (8000,4000)

        # use outer variables
        def plot_polarization(polarization):
            stackvar = list(sample[polarization].dims)[0] if len(sample[polarization].dims) > 2 else None
            if stackvar is None:
                stackvar = 'fake'
                da = self[[polarization]].to_dataset()[polarization].expand_dims({stackvar: [0]})
            else:
                da = self[[polarization]].isel({stackvar: slice(0, rows*cols)}).to_dataset()[polarization]
            #print ('da', da)
            if 'stack' in da.dims and isinstance(da.coords['stack'].to_index(), pd.MultiIndex):
                da = da.unstack('stack')
            
            # there is no reason to plot huge arrays much larger than screen size for small plots
            #print ('screen_size', screen_size)
            size_y, size_x = da.shape[-2:]
            #print ('size_x, size_y', size_x, size_y)
            factor_y = int(np.round(size_y / (_size[1] / rows)))
            factor_x = int(np.round(size_x / (_size[0] / cols)))
            #print ('factor_x, factor_y', factor_x, factor_y)
            # decimate for faster plot, do not coarsening without antialiasing
            # maybe data is already smoothed and maybe not, decimation is the only safe option
            da = da[:,::max(1, factor_y), ::max(1, factor_x)]
            # materialize for all the calculations and plotting
            progressbar(da := da.persist(), desc=f'Computing {polarization} Plot'.ljust(25))

            # calculate min, max when needed
            if quantile is not None:
                q = np.nanquantile(da.values, quantile)
                # Handle edge cases: all NaN data returns scalar, empty data, etc.
                if np.ndim(q) == 0:
                    _vmin = _vmax = float(q)
                else:
                    _vmin, _vmax = q[0], q[-1]
            else:
                _vmin, _vmax = vmin, vmax
            # define symmetrical boundaries
            if symmetrical is True and _vmax > 0:
                minmax = max(abs(_vmin), _vmax)
                _vmin = -minmax
                _vmax =  minmax
            
            # note: multi-plots ineffective for linked lazy data
            # Convert coordinates to kilometers for cleaner display
            da_plot = (self.wrap(da) if wrap else da)
            fg = da_plot.plot.imshow(
                col=stackvar,
                col_wrap=min(cols, da[stackvar].size), size=size, aspect=aspect,
                vmin=_vmin, vmax=_vmax,
                cmap=cmap, alpha=alpha,
                interpolation='none',
                cbar_kwargs={'label': caption or polarization},
            )
            fg.set_axis_labels('easting [km]', 'northing [km]')
            fg.set_ticks(max_xticks=nbins, max_yticks=nbins)
            fg.fig.suptitle(f'{polarization} {caption or ''}'.strip(), y=y)

            # fg is the FacetGrid returned by xarray.plot.imshow
            # Get original limits from first axis before any modifications
            if flip:
                first_ax = fg.axes.flatten()[0]
                orig_xlim = first_ax.get_xlim()
                orig_ylim = first_ax.get_ylim()
                # Ensure we flip to reversed order (max, min)
                flipped_xlim = (max(orig_xlim), min(orig_xlim))
                flipped_ylim = (max(orig_ylim), min(orig_ylim))

            for idx, ax in enumerate(fg.axes.flatten()):
                # flip axes if requested (force consistent flipped limits)
                if flip:
                    ax.set_xlim(flipped_xlim)
                    ax.set_ylim(flipped_ylim)
                # format tick labels in km
                km_formatter = FuncFormatter(lambda v, _: f'{v/1000:.0f}')
                ax.xaxis.set_major_formatter(km_formatter)
                ax.yaxis.set_major_formatter(km_formatter)
                if stackvar == 'fake':
                    # remove 'fake = 0' title
                    ax.set_title('')
                elif stackvar in ('pair', 'date') and idx < da[stackvar].size:
                    # Format pair/date titles nicely
                    if stackvar == 'pair':
                        # Get ref/rep from non-dimension coordinates
                        if 'ref' in da.coords and 'rep' in da.coords:
                            ref_val = da.coords['ref'].values[idx]
                            rep_val = da.coords['rep'].values[idx]
                            ref_str = pd.Timestamp(ref_val).strftime('%Y-%m-%d')
                            rep_str = pd.Timestamp(rep_val).strftime('%Y-%m-%d')
                            ax.set_title(f'{ref_str} {rep_str}')
                        else:
                            ax.set_title(f'pair={idx}')
                    elif stackvar == 'date':
                        coord_val = da[stackvar].values[idx]
                        if hasattr(coord_val, 'strftime'):
                            ax.set_title(f'date={coord_val.strftime("%Y-%m-%d")}')
                        else:
                            ax.set_title(f'date={pd.Timestamp(coord_val).strftime("%Y-%m-%d")}')

            return fg

        if quantile is not None:
            assert vmin is None and vmax is None, "ERROR: arguments 'quantile' and 'vmin', 'vmax' cannot be used together"

        sample = next(iter(self.values()))
        # find all variables in the first dataset related to polarizations
        # TODO
        #polarizations = [pol for pol in ['VV','VH','HH','HV'] if pol in sample.data_vars]
        polarizations = list(sample.data_vars)
        #print ('polarizations', polarizations)

        # process polarizations one by one
        fgs = []
        for pol in polarizations:
            fg = plot_polarization(polarization=pol)
            fgs.append(fg)
        return fgs

    def gaussian(
        self,
        weight: BatchUnit | None = None,
        wavelength: float | None = None,
        threshold: float = 0.5,
        device: str = 'auto',
        debug: bool = False
    ) -> Batch:
        """
        2D (yx) Gaussian kernel smoothing (multilook) on each dataset in this Batch.

        Parameters
        ----------
        weight : BatchUnit or None
            A Batch of 2D DataArrays, one per key, matching this Batch's keys.
            If None, no weighting is applied.
        wavelength : float or None
            Gaussian sigma via 5.3 cutoff formula. Must be positive if provided.
        threshold : float
            Drop-off threshold for the kernel.
        device : str, optional
            PyTorch device: 'auto' (default), 'cuda', 'mps', or 'cpu'.
            'auto' uses GPU if Dask client has resources={'gpu': 1}.
        debug : bool
            Print sigma values if True.

        Returns
        -------
        Batch
            A new Batch with the same keys, each smoothed by its corresponding weight.
        """
        import xarray as xr
        import numpy as np
        from .Batch import BatchUnit
        # constant 5.3 defines half-gain at filter_wavelength
        cutoff = 5.3

        # validate weight if provided
        if weight is not None:
            if not isinstance(weight, BatchUnit) or set(weight.keys()) != set(self.keys()):
                raise ValueError('`weight` must be a BatchUnit with the same keys as `self`')

        # precompute pixel sizes for decimation
        dy, dx = self.spacing

        # validate wavelength if provided
        if wavelength is not None:
            if wavelength <= 0:
                raise ValueError('wavelength must be positive')
            sig_y = wavelength / (dy * cutoff)
            sig_x = wavelength / (dx * cutoff)
            if debug:
                print(f'DEBUG: multilooking sigmas ({sig_y:.2f}, {sig_x:.2f}), wavelength {wavelength:.1f}')
            sigmas = (sig_y, sig_x)
        else:
            sigmas = None

        import dask
        import dask.array as da

        # Determine if GPU resource annotation is needed
        use_gpu = device in ('cuda', 'mps')

        out = {}
        # loop over each key
        for key, ds in self.items():
            w = weight[key] if weight is not None else None

            def gaussian_da(data_arr: xr.DataArray, w_da: xr.DataArray | None = None) -> xr.DataArray:
                is_complex = np.issubdtype(data_arr.dtype, np.complexfloating)
                out_dtype = np.complex64 if is_complex else np.float32

                # Get weight numpy array if provided
                weight_np = w_da.values if w_da is not None else None

                # Create wrapper for _gaussian (expects 2D input)
                def apply_gaussian(block):
                    return BatchCore._gaussian(block, weight_np, sigma=sigmas, threshold=threshold,
                                               device=device, pixel_sizes=(dy, dx)).astype(out_dtype)

                # Chunk first dim to 1 so _gaussian gets 2D slices (works for numpy and dask)
                first_dim = data_arr.dims[0]
                data_arr = data_arr.chunk({first_dim: 1})
                dask_data = data_arr.data

                # Build dimension string based on ndim (e.g., 'yx' for 2D, 'pyx' for 3D)
                dim_str = ''.join(chr(ord('a') + i) for i in range(dask_data.ndim))

                with dask.annotate(resources={'gpu': 1} if use_gpu else {}):
                    result_dask = da.blockwise(
                        apply_gaussian, dim_str,
                        dask_data, dim_str,
                        dtype=out_dtype,
                    )

                return xr.DataArray(
                    result_dask,
                    dims=data_arr.dims,
                    coords=data_arr.coords
                )

            # determine weight for each variable:
            # - if w is DataArray: use same weight for all variables
            # - if w is Dataset: use matching variable name from weight
            def get_weight(var):
                if w is None:
                    return None
                if isinstance(w, xr.DataArray):
                    return w
                # Dataset: use matching variable if exists
                return w[var] if var in w.data_vars else None

            # apply to every 2D or 3D (yx) var in the Dataset
            new_ds = xr.Dataset({
                var: gaussian_da(ds[var], get_weight(var))
                for var in ds.data_vars
                if (ds[var].ndim in (2,3) and ds[var].dims[-2:] == ('y','x'))
            })
            # preserve original Dataset attributes
            new_ds.attrs = ds.attrs
            out[key] = new_ds

        return type(self)(out)

    def residuals(self, polarization: str | None = None, debug: bool = False) -> float | list[float]:
        """
        Measure phase offset discrepancy across all burst overlaps.

        Computes the weighted mean of absolute median phase differences
        across all overlapping regions. After offset correction with fit(),
        these median differences should be close to zero.

        Parameters
        ----------
        polarization : str, optional
            Polarization to use for residual computation. Auto-detected if
            only one variable exists, otherwise defaults to 'VV'.
        debug : bool, optional
            Print debug information for each overlap. Default is False.

        Returns
        -------
        float or list[float]
            Single pair: Weighted mean absolute median phase discrepancy in radians.
            Multiple pairs: List of discrepancies, one per pair.
            0.0 = perfect alignment, π = maximum discrepancy (for wrapped phase).

            Practical interpretation:

            - < 0.1 rad: Excellent alignment
            - 0.1 - 0.5 rad: Good alignment
            - 0.5 - 1.0 rad: Moderate misalignment
            - > 1.0 rad: Poor alignment

        Examples
        --------
        >>> # Compare before and after alignment
        >>> before = intfs.residuals()
        >>> aligned = intfs.align()
        >>> after = aligned.residuals()
        >>> print(f'Discrepancy reduced from {before} to {after}')
        """
        from .Batch import Batch, BatchWrap
        import dask

        # Determine if we need circular statistics based on class type
        if isinstance(self, BatchWrap):
            use_circular = True
        elif isinstance(self, Batch):
            use_circular = False
        else:
            raise TypeError(f"residuals() only works with Batch (unwrapped) or BatchWrap (wrapped) phase data, not {type(self).__name__}")

        def maybe_wrap(x):
            """Wrap to [-π, π) for circular stats, identity otherwise."""
            if use_circular:
                return (x + np.pi) % (2*np.pi) - np.pi
            return x

        # Collect burst extents and detect pair dimension
        ids = sorted(self.keys())

        # Auto-detect polarization if not specified
        sample_ds = self[ids[0]]
        # Filter for spatial variables (with y, x dims) - excludes converted attributes like 'num_valid_az'
        available_pols = [v for v in sample_ds.data_vars
                         if 'y' in sample_ds[v].dims and 'x' in sample_ds[v].dims]
        if polarization is None:
            polarization = available_pols[0]
        if polarization not in available_pols:
            raise ValueError(f"Polarization '{polarization}' not found. Available: {available_pols}")

        sample_da = sample_ds[polarization]
        n_pairs = sample_da.sizes.get('pair', 1)
        has_pair_dim = 'pair' in sample_da.dims

        # Extract pathNumber and subswath from burst ID (format: "123_262883_IW2")
        burst_subswath = {}
        burst_track = {}  # pathNumber + subswath for detailed debug output
        for bid in ids:
            parts = bid.split('_')
            if len(parts) < 3:
                raise ValueError(f"Burst '{bid}' has invalid format, expected 'pathNumber_burstNumber_subswath'")
            path_num = parts[0]
            subswath = parts[2]
            burst_subswath[bid] = subswath
            burst_track[bid] = f"{path_num}{subswath}"

        extents = {}
        for bid in ids:
            ds = self[bid]
            da = ds[polarization]
            if 'pair' in da.dims:
                da = da.isel(pair=0)
            # Get coordinates from Dataset if not on DataArray
            y_coords = da.coords['y'].values if 'y' in da.coords else ds.coords['y'].values
            x_coords = da.coords['x'].values if 'x' in da.coords else ds.coords['x'].values
            extents[bid] = (y_coords.min(), y_coords.max(), x_coords.min(), x_coords.max())

        def extents_overlap(e1, e2):
            y1_min, y1_max, x1_min, x1_max = e1
            y2_min, y2_max, x2_min, x2_max = e2
            y_overlap = not (y1_max < y2_min or y2_max < y1_min)
            x_overlap = not (x1_max < x2_min or x2_max < x1_min)
            return y_overlap and x_overlap

        # Find all overlapping burst pairs
        overlap_pairs = []
        for i, id1 in enumerate(ids):
            e1 = extents[id1]
            for j, id2 in enumerate(ids):
                if i >= j:
                    continue
                e2 = extents[id2]
                if extents_overlap(e1, e2):
                    overlap_pairs.append((id1, id2))

        if not overlap_pairs:
            return [0.0] * n_pairs if has_pair_dim else 0.0

        if debug:
            print(f'residuals: found {len(overlap_pairs)} overlap pairs, {n_pairs} pair(s)', flush=True)

        # Build all lazy phase differences (dask graphs)
        jobs = []
        lazy_diffs = []
        for id1, id2 in overlap_pairs:
            i1 = self[id1][polarization]
            i2 = self[id2][polarization]

            for pair_idx in range(n_pairs):
                i1_p = i1.isel(pair=pair_idx) if 'pair' in i1.dims else i1
                i2_p = i2.isel(pair=pair_idx) if 'pair' in i2.dims else i2
                phase_diff = i2_p - i1_p
                jobs.append((id1, id2, pair_idx))
                lazy_diffs.append(phase_diff)

        # Compute all phase differences at once - dask schedules efficiently
        if debug:
            print(f'Computing {len(lazy_diffs)} phase differences...', flush=True)
        computed_diffs = dask.compute(*lazy_diffs)

        # Process computed results
        results = []
        for (id1, id2, pair_idx), phase_diff in zip(jobs, computed_diffs):
            valid = phase_diff.values.ravel()
            valid = valid[np.isfinite(valid)]

            if len(valid) == 0:
                continue

            valid = maybe_wrap(valid)
            median_diff = np.median(valid)
            abs_discrepancy = np.abs(maybe_wrap(median_diff))
            weight = len(valid)

            results.append((pair_idx, abs_discrepancy, weight, id1, id2, median_diff))

        # Aggregate results per pair and per subswath
        total_weights = [0.0] * n_pairs
        weighted_sums = [0.0] * n_pairs

        # Per-subswath tracking for debug
        subswath_stats = {}  # {(subswath, pair_idx): {'sum': float, 'weight': float, 'count': int, 'values': []}}
        per_overlap_discrepancies = {p: [] for p in range(n_pairs)}  # For computing std

        for result in results:
            if result is None:
                continue
            pair_idx, abs_discrepancy, weight, id1, id2, median_diff = result
            weighted_sums[pair_idx] += abs_discrepancy * weight
            total_weights[pair_idx] += weight
            per_overlap_discrepancies[pair_idx].append(abs_discrepancy)

            # Extract track info for debug stats
            if debug:
                track1 = burst_track[id1]
                track2 = burst_track[id2]

                # Categorize: same track or cross-track
                if track1 == track2:
                    track_key = track1
                else:
                    track_key = f'{track1}-{track2}'

                key = (track_key, pair_idx)
                if key not in subswath_stats:
                    subswath_stats[key] = {'sum': 0.0, 'weight': 0.0, 'count': 0, 'values': []}
                subswath_stats[key]['sum'] += abs_discrepancy * weight
                subswath_stats[key]['weight'] += weight
                subswath_stats[key]['count'] += 1
                subswath_stats[key]['values'].append(abs_discrepancy)

        discrepancies = []
        for p in range(n_pairs):
            if total_weights[p] == 0:
                discrepancies.append(0.0)
            else:
                discrepancies.append(round(weighted_sums[p] / total_weights[p], 3))

        if debug:
            # Compute std for overall discrepancy
            for p in range(n_pairs):
                vals = per_overlap_discrepancies[p]
                if len(vals) > 1:
                    std = np.std(vals)
                    print(f'Pair {p} discrepancy: {discrepancies[p]:.3f} ± {std:.3f} rad ({len(vals)} overlaps)', flush=True)
                else:
                    print(f'Pair {p} discrepancy: {discrepancies[p]:.3f} rad ({len(vals)} overlaps)', flush=True)

            # Print per-track stats (only for pair_idx=0 to avoid clutter)
            print('Per-track discrepancy (pair 0):', flush=True)
            for (track, pair_idx), stats in sorted(subswath_stats.items()):
                if pair_idx == 0 and stats['weight'] > 0:
                    track_disc = stats['sum'] / stats['weight']
                    vals = stats['values']
                    if len(vals) > 1:
                        track_std = np.std(vals)
                        print(f'  {track}: {track_disc:.3f} ± {track_std:.3f} rad ({stats["count"]} overlaps)', flush=True)
                    else:
                        print(f'  {track}: {track_disc:.3f} rad ({stats["count"]} overlaps)', flush=True)

        # Return single value for single pair, list for multiple
        if n_pairs == 1 and not has_pair_dim:
            return discrepancies[0]
        return discrepancies

    def fit(self,
            degree: int = 0,
            method: str = 'median',
            polarization: str | None = None,
            debug: bool = False,
            return_residuals: bool = False):
        """
        Estimate per-burst polynomial coefficients using overlap-based least-squares.

        Fits polynomial corrections (offset or offset+ramp) to each burst by analyzing
        phase differences in overlapping regions. Uses global least-squares optimization
        to find consistent coefficients across all bursts.

        Parameters
        ----------
        degree : int, optional
            Polynomial degree:
            - 0 (default): Estimate offsets only.
            - 1: Estimate linear ramp (in x/range direction).
        method : str, optional
            Estimation method: 'median' (robust) or 'mean' (faster).
        polarization : str, optional
            Polarization to use for coefficient estimation. Auto-detected if
            only one variable exists, otherwise defaults to 'VV'.
        debug : bool, optional
            Print debug information. Default is False.
        return_residuals : bool, optional
            If True, also return input residuals (before correction). Default is False.

        Returns
        -------
        dict or tuple
            If return_residuals is False:
                For single pair (no pair dimension):
                    degree=0: {burst_id: offset}
                    degree=1: {burst_id: [ramp, intercept]}
                For multiple pairs:
                    degree=0: {burst_id: [offset_pair0, offset_pair1, ...]}
                    degree=1: {burst_id: [[ramp0, intercept0], [ramp1, intercept1], ...]}
            If return_residuals is True:
                (coefficients_dict, residuals) where residuals is float or list[float]

        Examples
        --------
        >>> # 3-step alignment for best results (0.028 rad discrepancy):
        >>> # Step 1: Estimate offsets
        >>> offsets1 = intfs.fit(degree=0)
        >>> intfs1 = intfs - offsets1
        >>> # Step 2: Estimate ramps
        >>> ramps = intfs1.fit(degree=1)
        >>> intfs2 = intfs1 - intfs1.polyval(ramps)
        >>> # Step 3: Re-estimate offsets
        >>> offsets2 = intfs2.fit(degree=0)
        >>> # Combine coefficients (for single pair)
        >>> coeffs = {b: [ramps[b][0], ramps[b][1] + offsets1[b] + offsets2[b]] for b in offsets1}
        >>> aligned = intfs - intfs.polyval(coeffs)
        """
        from .Batch import Batch, BatchWrap
        import dask
        from scipy import sparse
        from scipy.sparse.linalg import lsqr
        from scipy.sparse.csgraph import connected_components

        # Determine if we need circular statistics based on class type
        if isinstance(self, BatchWrap):
            use_circular = True
        elif isinstance(self, Batch):
            use_circular = False
        else:
            raise TypeError(f"fit() only works with Batch (unwrapped) or BatchWrap (wrapped) phase data, not {type(self).__name__}")

        # Constants
        MIN_OVERLAP_PIXELS = 50
        MIN_ROW_PIXELS = 10
        MIN_VALID_ROWS = 5
        MIN_INLIER_SAMPLES = 10
        MAD_OUTLIER_THRESHOLD = 2.5
        OUTPUT_PRECISION = 3
        RAMP_PRECISION = 9

        def maybe_wrap(x):
            """Wrap to [-π, π) for circular stats, identity otherwise."""
            if use_circular:
                return (x + np.pi) % (2*np.pi) - np.pi
            return x

        def phase_diff(a, center):
            """Circular or linear difference from center."""
            if use_circular:
                return maybe_wrap(a - center)
            return a - center

        def phase_mean(a):
            """Circular or linear mean."""
            if use_circular:
                return np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))
            return np.mean(a)

        def phase_mad(a, center):
            """Circular or linear MAD."""
            return np.median(np.abs(phase_diff(a, center)))

        # Collect burst extents and x-centers
        ids = sorted(self.keys())
        n_bursts = len(ids)
        id_to_idx = {bid: i for i, bid in enumerate(ids)}

        # Auto-detect polarization if not specified
        sample_ds = self[ids[0]]
        # Filter for spatial variables (with y, x dims) - excludes converted attributes like 'num_valid_az'
        available_pols = [v for v in sample_ds.data_vars
                         if 'y' in sample_ds[v].dims and 'x' in sample_ds[v].dims]
        if polarization is None:
            polarization = available_pols[0]
        if polarization not in available_pols:
            raise ValueError(f"Polarization '{polarization}' not found. Available: {available_pols}")

        # Detect number of pairs
        sample_da = sample_ds[polarization]
        n_pairs = sample_da.sizes.get('pair', 1)
        has_pair_dim = 'pair' in sample_da.dims

        if debug:
            print(f'fit(degree={degree}): {n_bursts} bursts, {n_pairs} pair(s), pol={polarization}', flush=True)

        # Extract pathNumber and subswath from burst ID (format: "123_262883_IW2")
        # Used to skip same-path different-subswath overlaps (small x-extent, diagonal connection)
        # but allow cross-path overlaps which can have large x-extent with significant iono ramps
        burst_path = {}  # pathNumber (e.g., '33')
        burst_subswath = {}  # subswath (e.g., 'IW3')
        for bid in ids:
            if degree == 1:
                parts = bid.split('_')
                if len(parts) < 3:
                    raise ValueError(f"Burst '{bid}' has invalid format, expected 'pathNumber_burstNumber_subswath'")
                burst_path[bid] = parts[0]
                burst_subswath[bid] = parts[2]

        extents = {}
        x_centers = {}

        for bid in ids:
            ds = self[bid]
            da = ds[polarization]
            if 'pair' in da.dims:
                da = da.isel(pair=0)
            # Get coordinates from Dataset if not on DataArray
            y_coords = da.coords['y'].values if 'y' in da.coords else ds.coords['y'].values
            x_coords = da.coords['x'].values if 'x' in da.coords else ds.coords['x'].values
            extents[bid] = (y_coords.min(), y_coords.max(), x_coords.min(), x_coords.max())
            x_centers[bid] = float(np.mean(x_coords))

        def extents_overlap(e1, e2):
            y1_min, y1_max, x1_min, x1_max = e1
            y2_min, y2_max, x2_min, x2_max = e2
            y_overlap = not (y1_max < y2_min or y2_max < y1_min)
            x_overlap = not (x1_max < x2_min or x2_max < x1_min)
            return y_overlap and x_overlap

        def process_phase_diff(phase, id1, id2, pair_idx):
            """Process a computed phase difference to extract offset and optionally ramp."""
            all_valid = phase.values.ravel()
            all_valid = all_valid[np.isfinite(all_valid)]

            if len(all_valid) < MIN_OVERLAP_PIXELS:
                return None

            if 'y' not in phase.dims or 'x' not in phase.dims:
                return None

            x_coords = phase.coords['x'].values

            # Row-wise processing
            row_phases = []
            row_x_centroids = []
            row_weights = []

            for y_idx in range(phase.shape[0]):
                row = phase.values[y_idx, :]
                valid_mask = np.isfinite(row)
                n_valid = np.sum(valid_mask)
                if n_valid >= MIN_ROW_PIXELS:
                    x_valid = x_coords[valid_mask]
                    phase_valid = row[valid_mask]
                    # Unwrap for row mean computation (needed for both circular and linear)
                    if use_circular:
                        phase_unwrapped = np.unwrap(phase_valid)
                        row_mean = maybe_wrap(np.mean(phase_unwrapped))
                    else:
                        row_mean = np.mean(phase_valid)
                    row_phases.append(row_mean)
                    row_x_centroids.append(np.mean(x_valid))
                    row_weights.append(n_valid)

            if len(row_phases) < MIN_VALID_ROWS:
                return None

            a = np.array(row_phases)
            x_row = np.array(row_x_centroids)
            weights = np.array(row_weights)
            a = maybe_wrap(a)

            # Outlier rejection
            if method == 'median':
                offset_initial = np.median(a)
                mad = phase_mad(a, offset_initial)
                if mad > 0:
                    inliers = np.abs(phase_diff(a, offset_initial)) <= MAD_OUTLIER_THRESHOLD * mad
                    if np.sum(inliers) >= MIN_INLIER_SAMPLES:
                        a = a[inliers]
                        x_row = x_row[inliers]
                        weights = weights[inliers]

            n_valid = int(np.sum(weights))
            x_centroid = float(np.average(x_row, weights=weights))

            # Compute offset
            if method == 'median':
                sorted_idx = np.argsort(a)
                cumsum = np.cumsum(weights[sorted_idx])
                median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
                offset = a[sorted_idx[median_idx]]
            else:
                offset = phase_mean(a)

            # Compute ramp if degree=1
            ramp_val = None
            if degree == 1 and len(a) >= MIN_VALID_ROWS:
                x_centered = x_row - x_centroid
                x_range = np.max(x_row) - np.min(x_row)
                if x_range > 100:
                    residuals = a - offset
                    Swxx = np.sum(weights * x_centered**2)
                    Swxr = np.sum(weights * x_centered * residuals)
                    if Swxx > 1e-10:
                        ramp_val = Swxr / Swxx

            return (id1, id2, pair_idx, maybe_wrap(offset), ramp_val, x_centroid, n_valid)

        # Find overlapping burst pairs
        # For degree=1 (ramp), skip same-path cross-subswath overlaps (small x-extent, diagonal)
        # but allow cross-path overlaps which have large x-extent with significant iono ramps
        # For degree=0 (offset), use all overlaps including cross-subswath
        all_overlap_pairs = []
        cross_subswath_skipped = 0
        for i, id1 in enumerate(ids):
            e1 = extents[id1]
            for j, id2 in enumerate(ids):
                if i >= j:
                    continue
                if extents_overlap(e1, extents[id2]):
                    if degree == 1:
                        # For ramp estimation, skip same-path cross-subswath overlaps (diagonal, small x-extent)
                        # but allow cross-path overlaps - they have large x-extent with iono ramp differences
                        path1, path2 = burst_path[id1], burst_path[id2]
                        sw1, sw2 = burst_subswath[id1], burst_subswath[id2]
                        if path1 == path2 and sw1 != sw2:
                            # Same path, different subswath: diagonal overlap, skip
                            cross_subswath_skipped += 1
                            continue
                        # Same path + same subswath (along-track) or different paths: allow
                    all_overlap_pairs.append((id1, id2))

        if debug:
            print(f'Found {len(all_overlap_pairs)} overlapping burst pairs', flush=True)
            if degree == 1 and cross_subswath_skipped > 0:
                print(f'  (skipped {cross_subswath_skipped} same-path cross-subswath pairs for ramp estimation)', flush=True)

        # Build all lazy phase differences (dask graphs)
        jobs = []
        lazy_diffs = []
        for id1, id2 in all_overlap_pairs:
            i1 = self[id1][polarization]
            i2 = self[id2][polarization]

            for pair_idx in range(n_pairs):
                i1_p = i1.isel(pair=pair_idx) if 'pair' in i1.dims else i1
                i2_p = i2.isel(pair=pair_idx) if 'pair' in i2.dims else i2
                phase_diff_val = i2_p - i1_p
                jobs.append((id1, id2, pair_idx))
                lazy_diffs.append(phase_diff_val)

        if debug:
            print(f'Computing {len(lazy_diffs)} phase differences...', flush=True)

        # Compute all phase differences at once - dask schedules efficiently
        computed_diffs = dask.compute(*lazy_diffs)

        # Process computed results
        results = []
        rejection_counts = {'too_few_pixels': 0, 'missing_dims': 0, 'too_few_rows': 0, 'no_ramp': 0}
        for (id1, id2, pair_idx), phase in zip(jobs, computed_diffs):
            result = process_phase_diff(phase, id1, id2, pair_idx)
            if result is not None:
                results.append(result)
            else:
                # Count rejections (only for pair_idx==0 to avoid double counting)
                if pair_idx == 0:
                    all_valid = phase.values.ravel()
                    all_valid = all_valid[np.isfinite(all_valid)]
                    if len(all_valid) < MIN_OVERLAP_PIXELS:
                        rejection_counts['too_few_pixels'] += 1
                    elif 'y' not in phase.dims or 'x' not in phase.dims:
                        rejection_counts['missing_dims'] += 1
                    else:
                        rejection_counts['too_few_rows'] += 1

        if debug:
            n_valid = len([r for r in results if r[2] == 0])  # count for pair_idx=0
            n_rejected = len(all_overlap_pairs) - n_valid
            print(f'  Rejections: {rejection_counts}', flush=True)
            if degree == 1:
                # Count how many had no ramp computed
                no_ramp = sum(1 for r in results if r[2] == 0 and r[4] is None)
                print(f'  Valid pairs with no ramp (x_range too small): {no_ramp}', flush=True)

        # Organize results by pair_idx
        pairs_by_pair_idx = {p: [] for p in range(n_pairs)}
        for result in results:
            if result is None:
                continue
            id1, id2, pair_idx, offset, ramp_val, x_centroid, n_used = result
            weight = np.sqrt(n_used)
            if degree == 0:
                pairs_by_pair_idx[pair_idx].append((id1, id2, offset, weight))
            else:
                if ramp_val is not None:
                    pairs_by_pair_idx[pair_idx].append((id1, id2, offset, ramp_val, x_centroid, weight))

        def solve_for_pair(pair_idx):
            """Solve least-squares for a single pair index."""
            pairs = pairs_by_pair_idx[pair_idx]

            if len(pairs) == 0:
                if degree == 0:
                    return {bid: np.float32(0.0) for bid in ids}
                else:
                    return {bid: [np.float32(0.0), np.float32(0.0)] for bid in ids}

            # Build connectivity graph
            adjacency = sparse.lil_matrix((n_bursts, n_bursts))
            for p in pairs:
                id1, id2 = p[0], p[1]
                i, j = id_to_idx[id1], id_to_idx[id2]
                adjacency[i, j] = 1
                adjacency[j, i] = 1

            n_components, labels = connected_components(adjacency.tocsr(), directed=False)

            if debug and pair_idx == 0:  # Only print for first pair to avoid spam
                print(f'  Found {n_components} connected component(s) for {len(pairs)} valid pairs', flush=True)
                for comp in range(n_components):
                    comp_indices = np.where(labels == comp)[0]
                    comp_ids = [ids[i] for i in comp_indices]
                    # Try to extract subswath info from burst IDs
                    subswaths = set()
                    paths = set()
                    for bid in comp_ids:
                        if '_IW' in bid:
                            sw = bid.split('_IW')[1][0]
                            subswaths.add(f'IW{sw}')
                        parts = bid.split('_')
                        if len(parts) >= 1 and parts[0].isdigit():
                            paths.add(parts[0])
                    print(f'    Component {comp}: {len(comp_ids)} bursts, paths={paths}, subswaths={subswaths}', flush=True)

            if degree == 0:
                offsets_out = {}

                for comp in range(n_components):
                    comp_indices = np.where(labels == comp)[0]
                    comp_ids = [ids[i] for i in comp_indices]
                    comp_id_to_local = {bid: i for i, bid in enumerate(comp_ids)}
                    n_comp = len(comp_ids)

                    if n_comp == 1:
                        offsets_out[comp_ids[0]] = np.float32(0.0)
                        continue

                    comp_pairs = [(id1, id2, off, w) for id1, id2, off, w in pairs
                                  if id1 in comp_id_to_local and id2 in comp_id_to_local]

                    if len(comp_pairs) == 0:
                        for bid in comp_ids:
                            offsets_out[bid] = np.float32(0.0)
                        continue

                    n_pairs_comp = len(comp_pairs)
                    A = sparse.lil_matrix((n_pairs_comp + 1, n_comp))
                    b = np.zeros(n_pairs_comp + 1)
                    W = np.zeros(n_pairs_comp + 1)

                    for k, (id1, id2, off, w) in enumerate(comp_pairs):
                        i = comp_id_to_local[id1]
                        j = comp_id_to_local[id2]
                        A[k, i] = -1
                        A[k, j] = +1
                        b[k] = off
                        W[k] = w

                    constraint_weight = np.sum(W[:-1]) * 100 if np.sum(W[:-1]) > 0 else 1e6
                    A[n_pairs_comp, 0] = 1
                    b[n_pairs_comp] = 0
                    W[n_pairs_comp] = constraint_weight

                    sqrt_W = np.sqrt(W)
                    result = lsqr(sparse.diags(sqrt_W) @ A.tocsr(), sqrt_W * b)

                    for i, bid in enumerate(comp_ids):
                        offsets_out[bid] = np.float32(round(float(maybe_wrap(result[0][i])), OUTPUT_PRECISION))

                return offsets_out

            else:  # degree == 1
                ramps_out = {}

                for comp in range(n_components):
                    comp_indices = np.where(labels == comp)[0]
                    comp_ids = [ids[i] for i in comp_indices]
                    comp_id_to_local = {bid: i for i, bid in enumerate(comp_ids)}
                    n_comp = len(comp_ids)

                    if n_comp == 1:
                        ramps_out[comp_ids[0]] = [np.float32(0.0), np.float32(0.0)]
                        continue

                    comp_pairs = [(id1, id2, off, r, xc, w) for id1, id2, off, r, xc, w in pairs
                                  if id1 in comp_id_to_local and id2 in comp_id_to_local]

                    if len(comp_pairs) == 0:
                        for bid in comp_ids:
                            ramps_out[bid] = [np.float32(0.0), np.float32(0.0)]
                        continue

                    n_pairs_comp = len(comp_pairs)
                    A = sparse.lil_matrix((n_pairs_comp + 1, n_comp))
                    b = np.zeros(n_pairs_comp + 1)
                    W = np.zeros(n_pairs_comp + 1)

                    for k, (id1, id2, off, ramp_diff, xc, w) in enumerate(comp_pairs):
                        i = comp_id_to_local[id1]
                        j = comp_id_to_local[id2]
                        A[k, i] = -1
                        A[k, j] = +1
                        b[k] = ramp_diff
                        W[k] = w

                    constraint_weight = np.sum(W[:-1]) * 100 if np.sum(W[:-1]) > 0 else 1e6
                    A[n_pairs_comp, 0] = 1
                    b[n_pairs_comp] = 0
                    W[n_pairs_comp] = constraint_weight

                    sqrt_W = np.sqrt(W)
                    result = lsqr(sparse.diags(sqrt_W) @ A.tocsr(), sqrt_W * b)

                    for i, bid in enumerate(comp_ids):
                        ramp = np.float32(round(float(result[0][i]), RAMP_PRECISION))
                        intercept = np.float32(round(-ramp * x_centers[bid], OUTPUT_PRECISION))
                        ramps_out[bid] = [ramp, intercept]

                return ramps_out

        # Solve for each pair
        results_per_pair = [solve_for_pair(p) for p in range(n_pairs)]

        # Compute residuals from the pairwise offsets (before correction)
        # This uses the same overlap data we already computed
        residuals_out = None
        if return_residuals:
            # Calculate weighted mean absolute offset per pair
            disc_per_pair = []
            for p in range(n_pairs):
                pairs_p = pairs_by_pair_idx[p]
                if len(pairs_p) == 0:
                    disc_per_pair.append(0.0)
                    continue

                # Extract offsets and weights
                if degree == 0:
                    # pairs_p = [(id1, id2, offset, weight), ...]
                    offsets = [abs(maybe_wrap(t[2])) for t in pairs_p]
                    weights = [t[3] for t in pairs_p]
                else:
                    # pairs_p = [(id1, id2, offset, ramp, x_centroid, weight), ...]
                    offsets = [abs(maybe_wrap(t[2])) for t in pairs_p]
                    weights = [t[5] for t in pairs_p]

                total_weight = sum(weights)
                if total_weight > 0:
                    weighted_sum = sum(o * w for o, w in zip(offsets, weights))
                    disc_per_pair.append(round(weighted_sum / total_weight, 3))
                else:
                    disc_per_pair.append(0.0)

            if n_pairs == 1 and not has_pair_dim:
                residuals_out = disc_per_pair[0]
            else:
                residuals_out = disc_per_pair

            if debug:
                print(f'Input residuals: {residuals_out}', flush=True)

        # If single pair, return simple dict
        if n_pairs == 1 and not has_pair_dim:
            coeffs = results_per_pair[0]
            return (coeffs, residuals_out) if return_residuals else coeffs

        # Multiple pairs: combine into list per burst
        combined = {}
        for bid in ids:
            if degree == 0:
                combined[bid] = [results_per_pair[p][bid] for p in range(n_pairs)]
            else:
                combined[bid] = [results_per_pair[p][bid] for p in range(n_pairs)]

        return (combined, residuals_out) if return_residuals else combined

    def align(self,
              degree: int = 0,
              method: str = 'median',
              polarization: str | None = None,
              debug: bool = False,
              return_residuals: bool = False):
        """
        Align burst interferograms by removing phase offsets and optionally ionospheric ramps.

        Uses a multi-step approach for optimal alignment:
        - degree=0: Single-step offset correction
        - degree=1: 3-step correction (offset → ramp → re-offset) for ionospheric ramp removal

        The 3-step approach produces consistent fringes across bursts by removing
        per-track ionospheric ramps, which is essential for deformation analysis.

        Parameters
        ----------
        degree : int, optional
            Correction degree:
            - 0 (default): Offset-only correction (faster, good overlap alignment)
            - 1: Offset + linear ramp correction (better fringe continuity)
        method : str, optional
            Estimation method: 'median' (robust, default) or 'mean' (faster).
        polarization : str, optional
            Polarization to use for coefficient estimation. Auto-detected if
            only one variable exists, otherwise defaults to 'VV'.
            Corrections are applied to all polarizations since phase offsets
            are the same for all polarizations (same geometry).
        debug : bool, optional
            Print debug information. Default is False.
        return_residuals : bool, optional
            If True, also return final residuals. Default is False.

        Returns
        -------
        BatchCore or tuple
            If return_residuals is False:
                Aligned interferograms with phase corrections applied.
            If return_residuals is True:
                (aligned_intfs, residuals) where residuals is float or list[float]

        Examples
        --------
        >>> # Simple offset-only alignment (default)
        >>> aligned = intfs.align()
        >>>
        >>> # Alignment with ramp correction
        >>> aligned = intfs.align(degree=1)
        >>>
        >>> # Use VH polarization for estimation
        >>> aligned = intfs.align(polarization='VH')
        >>>
        >>> # With coherence filtering
        >>> aligned = intfs.where(corr >= 0.3).align()
        >>>
        >>> # Get alignment quality with result
        >>> aligned, res = intfs.align(return_residuals=True)
        >>> print('Residuals:', res)

        Notes
        -----
        For degree=1, the function performs:
        1. Estimate and remove offsets
        2. Estimate and remove ramps (using along-track and cross-path overlaps)
        3. Re-estimate offsets on ramp-corrected data
        4. Combine into final [ramp, offset] coefficients

        Ramp estimation uses:
        - Same-path, same-subswath overlaps (along-track, y-direction)
        - Cross-path overlaps (can have large x-extent with significant iono ramps)

        It skips same-path, cross-subswath overlaps (diagonal, small x-extent).

        This 3-step approach achieves better fringe continuity than single-step
        methods because it separates the offset and ramp estimation, avoiding
        cross-contamination between the two.
        """
        from .Batch import Batch, BatchWrap
        import warnings

        # Validate class type
        if not isinstance(self, (Batch, BatchWrap)):
            raise TypeError(f"align() only works with Batch (unwrapped) or BatchWrap (wrapped) phase data, not {type(self).__name__}")

        # Warn if data is lazy (has dask arrays) - align() triggers expensive recomputation
        if len(self) > 0:
            first_ds = next(iter(self.values()))
            has_dask = any(hasattr(var.data, 'dask') for var in first_ds.data_vars.values())
            if has_dask:
                warnings.warn(
                    "align() called on lazy data. This triggers expensive recomputation. "
                    "Consider calling .compute() before .align() for better performance.",
                    UserWarning
                )

        # Auto-detect polarization if not specified
        if polarization is None:
            ids = list(self.keys())
            sample_ds = self[ids[0]]
            # Filter for spatial variables (with y, x dims) - excludes converted attributes like 'num_valid_az'
            available_pols = [v for v in sample_ds.data_vars
                             if 'y' in sample_ds[v].dims and 'x' in sample_ds[v].dims]
            polarization = available_pols[0]

        if degree == 0:
            # Single-step offset correction
            if debug:
                print('align(degree=0): single-step offset correction', flush=True)
                res_in = self.residuals(polarization=polarization)
                print(f'Input residuals: {res_in}', flush=True)

            offsets = self.fit(degree=0, method=method, polarization=polarization, debug=debug)
            aligned = self - offsets

            if debug or return_residuals:
                res_out = aligned.residuals(polarization=polarization)
                if debug:
                    print(f'Output residuals: {res_out}', flush=True)

            if return_residuals:
                return aligned, res_out
            return aligned

        elif degree == 1:
            # 3-step offset-ramp-offset correction
            if debug:
                print('align(degree=1): 3-step offset-ramp-offset correction', flush=True)
                res_in = self.residuals(polarization=polarization)
                print(f'Input residuals: {res_in}', flush=True)

            # Step 1: Estimate offsets
            if debug:
                print('\nStep 1: Estimate offsets...', flush=True)
            offsets1 = self.fit(degree=0, method=method, polarization=polarization, debug=debug)
            intfs1 = self - offsets1
            if debug:
                res1 = intfs1.residuals(polarization=polarization)
                print(f'Residuals after step 1: {res1}', flush=True)

            # Step 2: Estimate ramps (uses same-track overlaps only)
            if debug:
                print('\nStep 2: Estimate ramps...', flush=True)
            ramps = intfs1.fit(degree=1, method=method, polarization=polarization, debug=debug)
            intfs2 = intfs1 - intfs1.polyval(ramps)
            if debug:
                res2 = intfs2.residuals(polarization=polarization)
                print(f'Residuals after step 2: {res2}', flush=True)

            # Step 3: Re-estimate offsets
            if debug:
                print('\nStep 3: Re-estimate offsets...', flush=True)
            offsets2 = intfs2.fit(degree=0, method=method, polarization=polarization, debug=debug)

            # Combine coefficients: [ramp, offset1 + ramp_intercept + offset2]
            # Detect if multi-pair
            sample_bid = list(offsets1.keys())[0]
            is_multi_pair = isinstance(offsets1[sample_bid], list)

            if is_multi_pair:
                n_pairs = len(offsets1[sample_bid])
                coeffs = {
                    b: [[ramps[b][p][0], ramps[b][p][1] + offsets1[b][p] + offsets2[b][p]]
                        for p in range(n_pairs)]
                    for b in offsets1
                }
            else:
                coeffs = {
                    b: [ramps[b][0], ramps[b][1] + offsets1[b] + offsets2[b]]
                    for b in offsets1
                }

            aligned = self - self.polyval(coeffs)

            if debug or return_residuals:
                res_out = aligned.residuals(polarization=polarization)
                if debug:
                    print(f'Final residuals: {res_out}', flush=True)

            if return_residuals:
                return aligned, res_out
            return aligned

        else:
            raise ValueError(f"degree must be 0 or 1, got {degree}")

    def dissolve(self, extend: bool = False, weight: float = None, debug: bool = False):
        """
        Dissolve burst boundaries by averaging overlapping regions.

        For each burst, this method computes a merged product covering that burst's
        extent, averaging values from all overlapping bursts.

        For wrapped phase data (BatchWrap), circular mean is used.
        For unwrapped phase or other data (Batch, BatchUnit), arithmetic mean is used.

        Parameters
        ----------
        extend : bool, optional
            If True, NaN areas in current burst can be filled by overlapping
            bursts. Good for unwrapping consistency between bursts.
            If False (default), only pixels valid in the current burst are kept (NaN areas remain NaN).
            Better for performance when you don't want to process same pixels in multiple bursts.
        weight : float, optional
            Normalized weight of the current burst in range [0, 1]. Default is None.
            weight=None: equal weights for all bursts (simple average)
            weight=1: only current burst used, overlapping bursts ignored
            weight=0: only overlapping bursts used, current burst ignored
            weight=0.5: current burst has same weight as sum of all overlapping bursts
        debug : bool, optional
            Print debug information. Default is False.

        Returns
        -------
        BatchCore
            New batch with dissolved (averaged) overlap regions (lazy).

        Examples
        --------
        >>> # Dissolve with equal weights (default)
        >>> intfs_dissolved = intfs.dissolve()
        >>>
        >>> # Dissolve without extension (keep original burst footprint)
        >>> intfs_dissolved = intfs.dissolve(extend=False)
        >>>
        >>> # Dissolve with current burst having 70% weight
        >>> intfs_dissolved = intfs.dissolve(weight=0.7)

        Notes
        -----
        - For BatchWrap (wrapped phase): uses circular mean via exp(1j*phase)
        - For Batch/BatchUnit (unwrapped phase, correlation): uses arithmetic mean
        - Returns lazy data, processes per burst replacing polarization variables
        """
        import warnings
        import dask
        import dask.array as da
        from shapely import box, STRtree
        from .Batch import BatchWrap

        if len(self) <= 1:
            return type(self)(self)

        wrap = isinstance(self, BatchWrap)
        burst_ids = list(self.keys())
        sample = self[burst_ids[0]]
        # Filter for spatial variables (with y, x dims) - excludes converted attributes
        polarizations = [v for v in sample.data_vars
                        if 'y' in sample[v].dims and 'x' in sample[v].dims]

        if debug:
            import time
            t0 = time.time()
            print(f'dissolve: {len(self)} bursts, wrap={wrap}, extend={extend}, weight={weight}', flush=True)

        # Build STRtree for fast spatial queries
        first_pol = polarizations[0]
        burst_extents = tuple(
            (float(self[bid][first_pol].y.min()), float(self[bid][first_pol].y.max()),
             float(self[bid][first_pol].x.min()), float(self[bid][first_pol].x.max()))
            for bid in burst_ids
        )
        burst_boxes = [box(xmin, ymin, xmax, ymax) for ymin, ymax, xmin, xmax in burst_extents]
        tree = STRtree(burst_boxes)

        overlapping_map = {
            burst_idx: tuple(int(idx) for idx in tree.query(burst_boxes[burst_idx]) if idx != burst_idx)
            for burst_idx in range(len(burst_ids))
        }

        if debug:
            total_overlaps = sum(len(v) for v in overlapping_map.values())
            print(f'dissolve: STRtree found {total_overlaps} burst overlaps', flush=True)

        def dissolve_pol(da_current, das_others, wrap, extend, weight):
            """Dissolve one polarization - returns numpy array."""
            # Use exact coordinates from da_current - do NOT modify them
            ys = da_current.y.values
            xs = da_current.x.values
            n_others = len(das_others)

            if weight is None:
                w_current, w_other = 1.0, 1.0
            else:
                w_current = weight
                w_other = (1.0 - weight) / n_others if n_others > 0 else 0.0

            # Reindex das_others to match da_current coordinates
            # Use interp for floating point coordinate matching instead of reindex
            das_reindexed = []
            for d in das_others:
                # Use interp with nearest to avoid issues with floating point coordinate matching
                das_reindexed.append(d.interp(y=ys, x=xs, method='nearest', kwargs={'fill_value': np.nan}))

            current_vals = da_current.values
            current_valid = np.isfinite(current_vals)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)

                if wrap:
                    weighted_sum = np.where(current_valid, np.exp(1j * current_vals) * w_current, 0.0)
                    weight_sum = np.where(current_valid, w_current, 0.0)
                    for d in das_reindexed:
                        vals = d.values
                        valid = np.isfinite(vals)
                        weighted_sum += np.where(valid, np.exp(1j * vals) * w_other, 0.0)
                        weight_sum += np.where(valid, w_other, 0.0)
                    valid_weights = weight_sum > 0
                    normalized = np.divide(weighted_sum, weight_sum, out=np.zeros_like(weighted_sum), where=valid_weights)
                    out = np.where(valid_weights, np.arctan2(normalized.imag, normalized.real), np.nan)
                else:
                    weighted_sum = np.where(current_valid, current_vals * w_current, 0.0)
                    weight_sum = np.where(current_valid, w_current, 0.0)
                    for d in das_reindexed:
                        vals = d.values
                        valid = np.isfinite(vals)
                        weighted_sum += np.where(valid, vals * w_other, 0.0)
                        weight_sum += np.where(valid, w_other, 0.0)
                    out = np.divide(weighted_sum, weight_sum, out=np.full_like(weighted_sum, np.nan), where=weight_sum > 0)

                if not extend:
                    out = np.where(current_valid, out, np.nan)

            return out.astype(da_current.dtype)

        # Build output - per burst, replace pol variables with lazy arrays
        output = {}
        for burst_idx, bid in enumerate(burst_ids):
            overlapping_indices = overlapping_map[burst_idx]
            ds_current = self[bid]

            if not overlapping_indices:
                output[bid] = ds_current
                continue

            ds_others = [self[burst_ids[idx]] for idx in overlapping_indices]

            # Copy dataset and replace each pol with lazy dissolved version
            new_ds = ds_current.copy()
            for pol in polarizations:
                da_current = ds_current[pol]
                das_others = [ds[pol] for ds in ds_others]

                # Check if 3D (has stack dimension like 'pair')
                if len(da_current.dims) > 2:
                    stackvar = da_current.dims[0]
                    n_stack = da_current.sizes[stackvar]
                    shape_2d = da_current.shape[1:]

                    def dissolve_pol_3d(da_slice, das_others_slice, wrap, extend, weight):
                        """Wrapper to return 3D array with shape (1, y, x)."""
                        return dissolve_pol(da_slice, das_others_slice, wrap, extend, weight)[np.newaxis, ...]

                    # Create separate delayed array for each stack element for parallelization
                    delayed_slices = []
                    for i in range(n_stack):
                        da_slice = da_current.isel({stackvar: i})
                        das_others_slice = [d.isel({stackvar: i}) for d in das_others]
                        # Create 3D delayed array with shape (1, y, x) and chunks (1, -1, -1)
                        delayed_slice = da.from_delayed(
                            dask.delayed(dissolve_pol_3d)(da_slice, das_others_slice, wrap, extend, weight),
                            shape=(1,) + shape_2d,
                            dtype=da_current.dtype
                        )
                        delayed_slices.append(delayed_slice)

                    # Concatenate along axis 0 - each slice is already (1, y, x)
                    # Rechunk to ensure (1, -1, -1) chunking is preserved
                    delayed_array = da.concatenate(delayed_slices, axis=0).rechunk({0: 1, 1: -1, 2: -1})
                else:
                    # 2D case - single delayed array
                    delayed_array = da.from_delayed(
                        dask.delayed(dissolve_pol)(da_current, das_others, wrap, extend, weight),
                        shape=da_current.shape,
                        dtype=da_current.dtype
                    )
                new_ds[pol] = da_current.copy(data=delayed_array)

            output[bid] = new_ds

        if debug:
            print(f'dissolve: preparation done in {time.time() - t0:.1f}s', flush=True)

        return type(self)(output)
