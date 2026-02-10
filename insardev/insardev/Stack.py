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
from .Stack_plot import Stack_plot
from .BatchCore import BatchCore
from .Batch import Batch, BatchWrap, BatchUnit, BatchComplex, Batches
#from collections.abc import Mapping
from . import utils_io
from . import utils_xarray
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import rasterio as rio
    import pandas as pd
    import xarray as xr

class Stack(Stack_plot, BatchCore):

    def __init__(self, mapping:dict[str, xr.Dataset] | None = None):
        #print('Stack __init__', 0 if mapping is None else len(mapping))
        super().__init__(mapping)

    @staticmethod
    def _batch_type_for_subset(subset: dict) -> type | None:
        """Determine appropriate Batch type based on dtypes and dims in subset.

        Returns
        -------
        type or None
            BatchComplex if all variables are complex with date/pair dimension.
            Batch if all variables are spatial-only (no date/pair dimension).
            None if variables have date/pair dimension but are not complex (should be Stack).
        """
        import numpy as np
        dtypes = set()
        has_temporal_dim = False
        for ds in subset.values():
            for var in ds.data_vars:
                dtypes.add(ds[var].dtype)
                # Check if variable has temporal dimension (date or pair)
                if 'date' in ds[var].dims or 'pair' in ds[var].dims:
                    has_temporal_dim = True
        # If all variables are complex with temporal dim, use BatchComplex
        if dtypes and all(np.issubdtype(dt, np.complexfloating) for dt in dtypes) and has_temporal_dim:
            return BatchComplex
        # If no temporal dimension, return Batch (spatial-only variables)
        if not has_temporal_dim:
            return Batch
        # Has temporal dimension but not complex - should be Stack
        return None

    @property
    def coords(self):
        """Return coordinates from the first dataset in the stack.
        
        All datasets in a Stack share the same coordinate structure,
        so we expose the first one's coords for convenience.
        """
        if not self:
            return None
        first_ds = next(iter(self.values()))
        return first_ds.coords

    def PRM(self, keys: str | list[str] | None = None) -> dict:
        """Return platform parameters per burst.

        Parameters
        ----------
        keys : str | list[str] | None
            Parameter name(s) to extract. If ``None``, all scalar attrs
            and 0-D data_vars are returned per burst.

        Returns
        -------
        dict
            Mapping of burst key -> param dict (or burst key -> single value
            when a single key is requested).
        """
        if not self:
            return {}

        # normalize key selection
        select_all = keys is None
        if isinstance(keys, str):
            keys_list = [keys]
        else:
            keys_list = keys if keys is not None else None

        result: dict[str, object] = {}
        for burst, ds in self.items():
            params: dict[str, object] = dict(getattr(ds, 'attrs', {}))
            for name, var in getattr(ds, 'data_vars', {}).items():
                if var.ndim == 0:
                    try:
                        params.setdefault(name, var.item())
                    except Exception:
                        params.setdefault(name, var.values)

            if not select_all:
                if keys_list is None:
                    params = {}
                else:
                    params = {k: params.get(k) for k in keys_list}
                    if isinstance(keys, str):
                        params = params.get(keys)

            result[burst] = params

        return result

    def __getitem__(self, key):
        """Access by key or variable list."""
        # Handle variable selection like sbas[['ele']] or sbas[['VV', 'VH']]
        if isinstance(key, list):
            if len(key) == 1 and self:
                var = key[0]
                # Build a Batch containing only the requested variable for each scene
                subset = {k: ds[[var]] for k, ds in self.items() if var in ds.data_vars}
                if not subset:
                    raise KeyError(var)
            else:
                # Multiple variables
                subset = {k: ds[key] for k, ds in self.items()}
            # Return appropriate Batch type if spatial-only or complex, else Stack
            batch_type = self._batch_type_for_subset(subset)
            if batch_type is not None:
                return batch_type(subset)
            # Has temporal dimension but not complex - return Stack
            return type(self)(subset)
        return dict.__getitem__(self, key)

    def __getattr__(self, name: str):
        """
        Access variables (e.g., 'ele') from the Stack as Batch.

        This allows accessing variables stored in burst datasets:
            sbas.ele  -> BatchVar containing elevation data
        """
        if name.startswith('_') or name in ('keys', 'values', 'items', 'get'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if self:
            sample = next(iter(self.values()), None)
            if sample is not None and hasattr(sample, 'data_vars'):
                if name in sample.data_vars or name in sample.coords:
                    subset = {k: ds[[name]] for k, ds in self.items() if name in ds.data_vars or name in ds.coords}
                    if subset:
                        batch_type = self._batch_type_for_subset(subset)
                        # For single attribute access, default to Batch if not BatchComplex
                        if batch_type is None:
                            batch_type = Batch
                        return batch_type(subset)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def transform(self) -> Batch:
        """Return a Batch view of this Stack (including 1D/2D non-complex vars)."""
        return Batch(self)

    def adi2(self, angle_coarse: float = 15, angle_fine: float = 5, device: str = 'auto') -> "Batches":
        """
        Polarimetric ADI optimization for dual-pol data (VV/VH or HH/HV).

        NOTE: Requires insardev_polsar extension.

        Finds optimal polarimetric combination that minimizes Amplitude Dispersion
        Index (ADI) for each pixel. Uses coarse-to-fine search for efficiency:
        ~7x faster than exhaustive grid search with <0.1% PS loss.

        Parameters
        ----------
        angle_coarse : float
            Coarse grid step in degrees. Default 15°.
            Smaller = more accurate coarse pass but slower.
        angle_fine : float
            Fine grid step in degrees. Default 5°.
            This is the final angular precision of the result.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', or 'cpu'.

        Returns
        -------
        Batches
            [S_opt, adi, psi] where:

            - **S_opt** (Stack): Optimized SLC stack with mixed VV/VH signal.
              Preserves all original variables (BPR, coords, attrs).
              Use for interferograms: ``S_opt.pairs(baseline).phasediff()``

            - **adi** (Batch): Minimum ADI values per pixel.
              Use for PS selection: ``ps_mask = adi < 0.25``

            - **psi** (Batch): Optimal mixing angle per pixel (radians).
              ψ=0 means VV was optimal, ψ>0 means VH contributed.
              Use to identify where dual-pol helped: ``vh_helped = psi > 0.1``

        Examples
        --------
        Basic PS selection with dual-pol optimization:

        >>> S_opt, adi, psi = stack.adi2()
        >>> ps_mask = adi < 0.25

        Higher precision (slower):

        >>> S_opt, adi, psi = stack.adi2(angle_coarse=10, angle_fine=2)

        Faster but less accurate:

        >>> S_opt, adi, psi = stack.adi2(angle_coarse=20, angle_fine=5)

        Compare with single-pol to see improvement:

        >>> adi_vv = stack.adi()  # Single-pol VV
        >>> new_ps = (adi < 0.25) & (adi_vv > 0.25)  # PS found only by dual-pol
        >>> print(f"New PS from dual-pol: {new_ps.sum().compute()}")

        Use optimized stack for interferograms:

        >>> phase, corr = S_opt.pairs(baseline).phasediff()

        Notes
        -----
        - For pixels where ψ≈0, S_opt≈VV (no benefit from VH)
        - For pixels where ψ>0, S_opt phase should be more stable than VV
        - For DS processing, use ``phasediff2()`` instead (better coherence optimization)
        - Improvement varies by scene type; urban areas may see less benefit

        Algorithm
        ---------
        Uses coarse-to-fine (CTF) search:
        1. Search at angle_coarse step over full (ψ, φ) space
        2. Refine within ±angle_coarse at angle_fine step around best

        Default (15°, 5°) uses ~200 combinations vs ~1400 for full 5° grid.

        References
        ----------
        Ramirez et al. (2023), "Extending polarimetric optimization of multi-temporal
        InSAR techniques on dual polarized Sentinel-1 data"
        https://doi.org/10.1016/j.asr.2023.03.007
        """
        # Detect polarizations
        sample_ds = next(iter(self.values()))
        if 'VV' in sample_ds.data_vars and 'VH' in sample_ds.data_vars:
            pols = ['VV', 'VH']
        elif 'HH' in sample_ds.data_vars and 'HV' in sample_ds.data_vars:
            pols = ['HH', 'HV']
        else:
            raise ValueError("Dual-pol data required (VV+VH or HH+HV)")

        # Import implementation from insardev_polsar
        try:
            from insardev_polsar.adi2 import adi2 as _adi2_impl
        except ImportError:
            raise ImportError("adi2() requires insardev_polsar extension. Install it first.")

        # Get dual-pol subset as BatchComplex
        batch_complex = self[pols]

        # Call internal adi2 implementation
        s_opt_batch, adi_batch, psi_batch = _adi2_impl(batch_complex, angle_coarse, angle_fine, device)

        # Merge S_opt back into original stack structure (preserves BPR, etc.)
        # Replace VV/VH with optimized output, keep all other variables
        output_pol = pols[0]  # VV or HH
        s_opt_dict = {}
        for burst_id, orig_ds in self.items():
            s_opt_ds = s_opt_batch[burst_id]
            # Drop original pols, add optimized output
            merged = orig_ds.drop_vars(pols).assign({output_pol: s_opt_ds[output_pol]})
            s_opt_dict[burst_id] = merged

        s_opt_stack = type(self)(s_opt_dict)

        return Batches([s_opt_stack, adi_batch, psi_batch])

    def adi(self, device: str = 'auto') -> Batch:
        """
        Compute Amplitude Dispersion Index (ADI) for calibrated σ₀ data.

        Wrapper that calls BatchComplex.adi() on complex variables.

        Parameters
        ----------
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', or 'cpu'.

        Returns
        -------
        Batch
            ADI values for each polarization variable present.

        Examples
        --------
        >>> adi = stack.adi()
        >>> ps_mask = adi < 0.25
        """
        # Get complex variables
        sample_ds = next(iter(self.values()))
        complex_vars = [v for v in sample_ds.data_vars
                        if sample_ds[v].dtype.kind == 'c' and 'date' in sample_ds[v].dims]

        if not complex_vars:
            raise ValueError("No complex time-series data found")

        # Get as BatchComplex and call adi()
        batch_complex = self[complex_vars]
        return batch_complex.adi(device)

    def similarity(
        self,
        window: tuple = (5, 5),
        neighbors: tuple = (5, 5),
        valid_threshold: float = 0.5,
        device: str = 'auto'
    ) -> Batch:
        """
        Compute spatial phase similarity using neighbor phase difference variance.

        NOTE: Requires insardev_polsar extension.

        Wrapper that calls BatchComplex.similarity() on complex variables.
        Use with adi2()-optimized stack for best results.

        Parameters
        ----------
        window : tuple of int
            Window size (y, x). Asymmetric to account for different pixel spacing.
        neighbors : tuple of int
            Neighbor count range (min, max):
            - min: Minimum valid neighbors required, else NaN
            - max: Maximum neighbors to use (top N most similar)
        valid_threshold : float
            Minimum fraction of dates with valid (non-NaN) data.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'

        Returns
        -------
        Batch
            Phase similarity values (lower = more spatially similar)

        Examples
        --------
        >>> sim = stack.similarity(window=(5, 5), neighbors=(5, 5))
        >>> stable_mask = sim < 0.5
        """
        # Get complex variables
        sample_ds = next(iter(self.values()))
        complex_vars = [v for v in sample_ds.data_vars
                        if sample_ds[v].dtype.kind == 'c' and 'date' in sample_ds[v].dims]

        if not complex_vars:
            raise ValueError("No complex time-series data found")

        # Get as BatchComplex and call similarity()
        batch_complex = self[complex_vars]
        return batch_complex.similarity(window, neighbors, valid_threshold, device)

    def neighbors(
        self,
        window: tuple = (5, 5),
        neighbors: tuple | None = None,
        valid_threshold: float = 0.5,
        device: str = 'auto'
    ) -> Batch:
        """
        Count valid neighbors per pixel within a spatial window.

        NOTE: Requires insardev_polsar extension.

        Wrapper that calls BatchComplex.neighbors() on complex variables.
        Useful for estimating pixel density before running similarity().

        Parameters
        ----------
        window : tuple of int
            Window size (y, x). Must be odd numbers.
        neighbors : tuple of int or None
            If provided, filter output: (min, max)
            - Pixels with count < min: set to NaN
            - Pixels with count > max: clipped to max
        valid_threshold : float
            Minimum fraction of dates with valid data.
        device : str
            PyTorch device: 'auto', 'cuda', 'mps', 'cpu'

        Returns
        -------
        Batch
            Valid neighbor count per pixel

        Examples
        --------
        >>> counts = stack.neighbors(window=(15, 15))
        >>> dense_mask = counts >= 10
        """
        # Get complex variables
        sample_ds = next(iter(self.values()))
        complex_vars = [v for v in sample_ds.data_vars
                        if sample_ds[v].dtype.kind == 'c' and 'date' in sample_ds[v].dims]

        if not complex_vars:
            raise ValueError("No complex time-series data found")

        # Get as BatchComplex and call neighbors()
        batch_complex = self[complex_vars]
        return batch_complex.neighbors(window, neighbors, valid_threshold, device)

    def to_vtk(self, path: str, data: BatchCore | dict | None = None,
               transform: Batch | None = None, overlay: "xr.DataArray" | None = None, mask: bool = True):
        """Export to VTK.

        Merges bursts using to_dataset() and exports one VTK file per data variable
        (e.g., VV.vtk). Within each file, pairs become separate VTK arrays named
        by date (e.g., 20190708_20190702).

        Parameters
        ----------
        path : str
            Output directory/filename for VTK files.
        data : BatchCore | dict | None, optional
            Data to export. If ``None``, export this Stack. Use ``data=None`` with
            ``overlay`` to export just topography with image overlay.
        transform : Batch | None, optional
            Optional transform Batch providing topography (``ele`` or ``z``).
        overlay : xarray.DataArray | None, optional
            Optional overlay (e.g., imagery). If it lacks a ``band`` dim, one is added.
        mask : bool, optional
            If True, mask topography by valid data pixels.

        Examples
        --------
        # Export data with overlay
        stack.to_vtk('velocity', velocity, overlay=gmap)

        # Export just topography with image overlay (like PyGMTSAR export_vtk)
        stack.to_vtk('gmap', data=None, overlay=gmap)
        """
        import os
        import numpy as np
        import xarray as xr
        import pandas as pd
        from tqdm.auto import tqdm
        from vtk import vtkStructuredGridWriter, VTK_BINARY
        from .utils_vtk import as_vtk

        # Handle data=None case (export just overlay on topography)
        if data is None and overlay is not None:
            tfm = transform if transform is not None else self.transform()
            if tfm is not None and not isinstance(tfm, BatchCore):
                tfm = Batch(tfm)

            if tfm is None:
                raise ValueError("transform is required when data=None")

            # Get topography at native resolution
            topo_merged = tfm[['ele']].to_dataset()
            topo_da = topo_merged['ele'] if 'ele' in topo_merged else None
            if topo_da is None:
                raise ValueError("transform must contain 'ele' variable")

            # Interpolate elevation to overlay grid (preserves image quality)
            ov = overlay
            if 'band' not in ov.dims:
                ov = ov.expand_dims('band')
            # Interpolate elevation to match overlay coordinates
            topo_da = topo_da.interp(y=ov.y, x=ov.x, method='linear')

            if mask:
                # Mask by finite values in overlay
                topo_da = topo_da.where(np.isfinite(ov.isel(band=0)))

            # Build output dataset
            layers = [topo_da.rename('z'), ov.rename('colors')]
            ds_out = xr.merge(layers, compat='override', join='left')
            vtk_grid = as_vtk(ds_out)

            # Determine output filename
            if path.endswith('.vtk'):
                filename = path
            else:
                filename = f'{path}.vtk'
            os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)

            writer = vtkStructuredGridWriter()
            writer.SetFileName(filename)
            writer.SetInputData(vtk_grid)
            writer.SetFileType(VTK_BINARY)
            writer.Write()
            return

        target = data if data is not None else self
        if not isinstance(target, BatchCore):
            target = Batch(target)
        # Default to self.transform() when called on Stack and transform not provided
        tfm_is_default = transform is None
        tfm = transform if transform is not None else self.transform()
        if tfm is not None and not isinstance(tfm, BatchCore):
            tfm = Batch(tfm)

        if not target:
            return

        def _format_dt(val):
            try:
                ts = pd.to_datetime(val)
                if pd.isna(ts):
                    return str(val)
                return ts.strftime('%Y%m%d')
            except Exception:
                return str(val)

        def _format_pair(da, idx):
            """Format pair label from ref/rep coordinates at given index."""
            if 'ref' in da.coords and 'rep' in da.coords:
                ref_val = da.coords['ref'].values[idx]
                rep_val = da.coords['rep'].values[idx]
                return f"{_format_dt(ref_val)}_{_format_dt(rep_val)}"
            return str(idx)

        os.makedirs(path, exist_ok=True)

        # Merge bursts into unified dataset(s) per variable
        merged = target.to_dataset()
        if isinstance(merged, xr.DataArray):
            merged = merged.to_dataset()

        # Get transform elevation merged via to_dataset()
        # Decimate default transform to match input batch resolution for efficiency
        topo_merged = None
        if tfm is not None:
            if tfm_is_default:
                # Decimate each burst's transform to match corresponding input burst
                # Use index-based nearest neighbor selection (much faster than reindex)
                def _nearest_indices(source_coords, target_coords):
                    """Find indices in source_coords nearest to target_coords."""
                    # Handle descending coordinates (e.g., y going north to south)
                    descending = len(source_coords) > 1 and source_coords[0] > source_coords[-1]
                    if descending:
                        source_coords = source_coords[::-1]
                    indices = np.searchsorted(source_coords, target_coords)
                    indices = np.clip(indices, 0, len(source_coords) - 1)
                    # Check if previous index is closer
                    prev_indices = np.clip(indices - 1, 0, len(source_coords) - 1)
                    prev_diff = np.abs(source_coords[prev_indices] - target_coords)
                    curr_diff = np.abs(source_coords[indices] - target_coords)
                    indices = np.where(prev_diff < curr_diff, prev_indices, indices)
                    # Convert back to original order if descending
                    if descending:
                        indices = len(source_coords) - 1 - indices
                    return indices

                decimated = {}
                for k in target.keys():
                    if k not in tfm:
                        continue
                    tfm_ds = tfm[k][['ele']]
                    tgt_ds = target[k]
                    # Find nearest indices for y and x coordinates
                    y_idx = _nearest_indices(tfm_ds.y.values, tgt_ds.y.values)
                    x_idx = _nearest_indices(tfm_ds.x.values, tgt_ds.x.values)
                    # Select using indices and assign target coordinates
                    selected = tfm_ds.isel(y=y_idx, x=x_idx)
                    selected = selected.assign_coords(y=tgt_ds.y, x=tgt_ds.x)
                    decimated[k] = selected
                topo_merged = Batch(decimated).to_dataset()
            else:
                # User-provided transform: use as-is
                topo_merged = tfm[['ele']].to_dataset()

        # Group by data variable (polarization)
        data_vars = list(merged.data_vars)

        with tqdm(total=len(data_vars), desc='Exporting VTK') as pbar:
            for data_var in data_vars:
                da = merged[data_var]

                # Handle pair dimension
                if 'pair' in da.dims:
                    n_pairs = da.sizes['pair']
                    export_items = []
                    for i in range(n_pairs):
                        da_slice = da.isel(pair=i)
                        if 'pair' in da_slice.dims:
                            da_slice = da_slice.squeeze('pair', drop=True)
                        pair_label = _format_pair(da, i)
                        export_items.append((pair_label, da_slice))
                else:
                    export_items = [(None, da)]

                if not export_items:
                    pbar.update(1)
                    continue

                ref_da = export_items[0][1]
                layers = []

                # Determine target grid - use overlay grid if provided (preserves image quality)
                ov = None
                if overlay is not None:
                    if not isinstance(overlay, xr.DataArray):
                        raise TypeError("overlay must be an xarray.DataArray (e.g., an RGB raster)")
                    ov = overlay
                    if 'band' not in ov.dims:
                        ov = ov.expand_dims('band')
                    # Select overlay region matching data extent
                    y_min, y_max = float(ref_da.y.min()), float(ref_da.y.max())
                    x_min, x_max = float(ref_da.x.min()), float(ref_da.x.max())
                    try:
                        ov_y_asc = len(ov.y) < 2 or float(ov.y[1]) > float(ov.y[0])
                        ov_x_asc = len(ov.x) < 2 or float(ov.x[1]) > float(ov.x[0])
                        y_slice = slice(y_min, y_max) if ov_y_asc else slice(y_max, y_min)
                        x_slice = slice(x_min, x_max) if ov_x_asc else slice(x_max, x_min)
                        ov = ov.sel(y=y_slice, x=x_slice)
                    except Exception:
                        pass
                    if ov.size == 0:
                        ov = None

                # Use overlay grid or data grid as target
                target_y = ov.y if ov is not None else ref_da.y
                target_x = ov.x if ov is not None else ref_da.x

                # Add topography from transform
                if topo_merged is not None:
                    topo_da = topo_merged['ele'] if 'ele' in topo_merged else None
                    if topo_da is not None:
                        topo_da = topo_da.interp(y=target_y, x=target_x, method='linear')
                        if mask:
                            ref_for_mask = ref_da.interp(y=target_y, x=target_x, method='nearest') if ov is not None else ref_da
                            topo_da = topo_da.where(np.isfinite(ref_for_mask))
                        layers.append(topo_da.rename('z'))

                # Add overlay at native resolution
                if ov is not None:
                    layers.append(ov.rename('colors'))

                # Add data arrays - interpolate to target grid
                for pair_label, da_item in export_items:
                    var_name = pair_label if pair_label is not None else data_var
                    if ov is not None:
                        da_item = da_item.interp(y=target_y, x=target_x, method='linear')
                    layers.append(da_item.rename(var_name))

                ds_out = xr.merge(layers, compat='override', join='left')
                vtk_grid = as_vtk(ds_out)

                filename = os.path.join(path, f"{data_var}.vtk")

                writer = vtkStructuredGridWriter()
                writer.SetFileName(filename)
                writer.SetInputData(vtk_grid)
                writer.SetFileType(VTK_BINARY)
                writer.Write()

                pbar.update(1)

    def to_vtks(self, path: str, data: BatchCore | dict | None = None,
               transform: Batch | None = None, overlay: "xr.DataArray" | None = None, mask: bool = True):
        """Export to VTK from a Batch.

        Parameters
        ----------
        path : str
            Output directory for VTK files.
        data : BatchCore | dict | None, optional
            Data to export. If ``None``, export this Stack. Accepts any mapping convertible to ``Batch``.
        transform : Batch | None, optional
            Optional transform Batch providing topography (`ele`).
        overlay : xarray.DataArray | None, optional
            Optional overlay (e.g., imagery). If it lacks a ``band`` dim, one is added.
        mask : bool, optional
            If True, mask topography by valid data pixels.
        """
        import os
        import numpy as np
        import xarray as xr
        import pandas as pd
        from tqdm.auto import tqdm
        from vtk import vtkStructuredGridWriter, VTK_BINARY
        from .utils_vtk import as_vtk

        target = data if data is not None else self
        if not isinstance(target, BatchCore):
            target = Batch(target)
        tfm = transform
        if tfm is not None and not isinstance(tfm, BatchCore):
            tfm = Batch(tfm)

        if not target:
            return

        def _interp_to_grid(source: xr.DataArray, target_da: xr.DataArray) -> xr.DataArray:
            if {'y', 'x'}.issubset(source.dims):
                return source.interp(y=target_da.y, x=target_da.x, method='linear')
            if {'lat', 'lon'}.issubset(source.dims):
                if {'lat', 'lon'}.issubset(target_da.coords):
                    return source.interp(lat=target_da.lat, lon=target_da.lon, method='linear')
                return source.rename({'lat': 'y', 'lon': 'x'}).interp(y=target_da.y, x=target_da.x, method='linear')
            return source

        def _format_dt(val):
            try:
                ts = pd.to_datetime(val)
                if pd.isna(ts):
                    return str(val)
                return ts.strftime('%Y%m%d')
            except Exception:
                return str(val)

        def _format_pair(val):
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return f"{_format_dt(val[0])}_{_format_dt(val[1])}"
            return _format_dt(val)

        os.makedirs(path, exist_ok=True)

        with tqdm(total=len(target), desc='Exporting VTK') as pbar:
            for burst, ds in target.items():
                if not ds.data_vars:
                    pbar.update(1)
                    continue

                data_var = next(iter(ds.data_vars))
                base_da = ds[data_var]

                if 'pair' in ds.dims:
                    pair_coord = ds.coords.get('pair')
                    pair_values = pair_coord.values if pair_coord is not None else range(ds.sizes.get('pair', 0))
                    export_items = []
                    for i, pair_val in enumerate(pair_values):
                        ds_slice = ds.isel(pair=i)
                        if 'pair' in ds_slice.dims:
                            ds_slice = ds_slice.squeeze('pair', drop=True)
                        else:
                            ds_slice = ds_slice.squeeze(drop=True)
                        export_items.append((pair_val, ds_slice))
                else:
                    export_items = [(None, ds)]

                for pair_val, ds_item in export_items:
                    base_da_item = ds_item[data_var]
                    layers = [base_da_item.rename(data_var)]

                    if tfm is not None and burst in tfm:
                        tfm_ds = tfm[burst]
                        topo_da = tfm_ds.get('ele') if 'ele' in tfm_ds else tfm_ds.get('z') if 'z' in tfm_ds else None
                        if topo_da is not None:
                            topo_da = _interp_to_grid(topo_da, base_da_item)
                            if mask:
                                topo_da = topo_da.where(np.isfinite(base_da_item))
                            layers.append(topo_da.rename('z'))

                    if overlay is not None:
                        if not isinstance(overlay, xr.DataArray):
                            raise TypeError("overlay must be an xarray.DataArray (e.g., an RGB raster)")

                        ov = overlay
                        if 'band' not in ov.dims:
                            ov = ov.expand_dims('band')
                        try:
                            ov = ov.sel(y=slice(float(base_da_item.y.min()), float(base_da_item.y.max())),
                                        x=slice(float(base_da_item.x.min()), float(base_da_item.x.max())))
                        except Exception:
                            try:
                                ov = ov.sel(lat=slice(float(base_da_item.lat.min()), float(base_da_item.lat.max())),
                                            lon=slice(float(base_da_item.lon.min()), float(base_da_item.lon.max())))
                            except Exception:
                                pass
                        ov = _interp_to_grid(ov, base_da_item)
                        layers.append(ov.rename('colors'))

                    ds_out = xr.merge(layers, compat='override', join='left')
                    vtk_grid = as_vtk(ds_out)

                    pair_suffix = ''
                    if pair_val is not None:
                        pair_suffix = f"_{_format_pair(pair_val)}"

                    filename = os.path.join(path, f"{burst}{pair_suffix}.vtk")

                    writer = vtkStructuredGridWriter()
                    writer.SetFileName(filename)
                    writer.SetInputData(vtk_grid)
                    writer.SetFileType(VTK_BINARY)
                    writer.Write()

                pbar.update(1)

    def compute(self, *batches: BatchCore, allow_rechunk: bool = False) -> tuple:
        """Compute multiple Batch objects together efficiently.

        This method materializes multiple dependent Batch objects in a single
        dask graph execution, which is faster than computing them separately
        because shared computations are only performed once.

        Stack serves as a unified interface - use empty Stack() for utility operations:
        - stack.compute(batch1, batch2) or Stack().compute(batch1, batch2)

        Parameters
        ----------
        *batches : BatchCore
            One or more Batch objects to compute together.
        allow_rechunk : bool, optional
            If True (default), auto-rechunk output for efficient further processing:
            - 3D data: chunk size 1 for first dim (date/pair), auto for y,x
            - 2D data: auto for y,x based on dask.config['array.chunk-size']
            If False, use single spatial chunk (-1 for y,x).

        Returns
        -------
        tuple
            Tuple of computed Batch objects in the same order as input.

        Examples
        --------
        >>> mintf, mcorr = stack.phasediff(pairs, wavelength=200)
        >>> mintf, mcorr = Stack().compute(mintf.downsample(20), mcorr.downsample(20))
        """
        from .Batch import Batches
        return Batches(batches).compute(allow_rechunk=allow_rechunk)

    def snapshot(self, *args, store: str | None = None, storage_options: dict[str, str] | None = None,
                caption: str | None = None, n_jobs: int = -1, debug=False):
        """Save or open a snapshot of the Stack or Batch objects.

        Stack serves as a unified interface - use empty Stack() for utility operations:
        - stack.snapshot('path') on non-empty Stack saves the Stack
        - Stack().snapshot('path') on empty Stack opens existing snapshot
        - stack.snapshot(batch1, batch2, store='path') saves batches

        Parameters
        ----------
        *args : BatchCore or str
            Batch objects to save, or store path string.
        store : str, optional
            Path to the Zarr store (alternative to first positional arg).
        storage_options : dict, optional
            Storage options for cloud stores.
        caption : str, optional
            Progress bar caption.
        n_jobs : int
            Number of parallel jobs (-1 for all cores).
        debug : bool
            Print debug information.

        Returns
        -------
        Stack or tuple
            The saved Stack, or tuple of opened/saved Batch objects.

        Examples
        --------
        >>> # Save stack itself
        >>> stack.snapshot('mystack')
        >>> # Open existing snapshot
        >>> mintf, mcorr = Stack().snapshot('mintf_corr')
        >>> # Save batches
        >>> mintf, mcorr = stack.snapshot(mintf, mcorr, store='mintf_corr')
        """
        from .Batch import Batches
        from . import utils_io

        # Handle case where first arg is store path
        if len(args) == 1 and isinstance(args[0], str):
            store = args[0]
            args = ()

        # If no batch args provided
        if len(args) == 0:
            if store is None:
                raise ValueError("store path is required to save/open snapshot")
            # Empty Stack -> open mode, non-empty Stack -> save mode
            if len(self) == 0:
                result = Batches().snapshot(store=store, storage_options=storage_options,
                                             caption=caption, n_jobs=n_jobs, debug=debug)
                # Unwrap single Stack from tuple
                if len(result) == 1 and isinstance(result[0], Stack):
                    return result[0]
                return result
            utils_io.save(self, store=store, storage_options=storage_options, compat=True,
                         caption=caption or 'Snapshotting...', n_jobs=n_jobs, debug=debug)
            return self

        # Save mode - args are batches
        return Batches(args).snapshot(store=store, storage_options=storage_options, caption=caption, n_jobs=n_jobs, debug=debug)

    def archive(self, *args, store: str | None = None, caption: str | None = None,
                compression: int = 6, n_jobs: int = -1, debug=False):
        """Save or open an archive of the Stack or Batch objects as a single ZIP file.

        Wrapper around snapshot() that uses ZipStore for single-file storage.
        Useful for downloading data from Google Colab or similar environments.

        Stack serves as a unified interface - use empty Stack() for utility operations:
        - stack.archive('path.zip') on non-empty Stack saves the Stack
        - Stack().archive('path.zip') on empty Stack opens existing archive
        - stack.archive(batch1, batch2, store='path.zip') saves batches

        Parameters
        ----------
        *args : BatchCore or str
            Batch objects to save, or store path string.
        store : str, optional
            Path to the ZIP file. Must end with '.zip'.
        caption : str, optional
            Progress bar caption.
        compression : int
            ZIP compression level 0-9 (0=no compression, 9=max). Default 6.
            Higher values produce smaller files but take longer.
        n_jobs : int
            Number of parallel jobs (-1 for all cores).
        debug : bool
            Print debug information.

        Returns
        -------
        Stack or tuple
            The saved Stack, or tuple of opened/saved Batch objects.

        Examples
        --------
        >>> # Save stack itself
        >>> stack.archive('mystack.zip')
        >>> # Save with max compression (for GitHub 100MB limit)
        >>> stack.archive('mystack.zip', compression=9)
        >>> # Save to cloud storage (GCS, S3, etc.)
        >>> stack.archive('gs://bucket/mystack.zip')
        >>> # Open existing archive
        >>> mintf, mcorr = Stack().archive('mintf_corr.zip')
        >>> # Save batches
        >>> mintf, mcorr = stack.archive(mintf, mcorr, store='mintf_corr.zip')
        """
        import zipfile
        import tempfile
        import os
        import fsspec
        import zarr
        from .Batch import Batches
        from . import utils_io

        # Handle case where first arg is store path
        if len(args) == 1 and isinstance(args[0], str):
            store = args[0]
            args = ()

        if store is None:
            raise ValueError("store path is required for archive")

        if not store.endswith('.zip'):
            raise ValueError(f"Archive store must have '.zip' extension, got: {store}")

        # Check if cloud storage path
        is_cloud = '://' in store

        # If no batch args provided
        if len(args) == 0:
            # Empty Stack -> open mode, non-empty Stack -> save mode
            if len(self) == 0:
                # Open mode - check file exists first
                if is_cloud:
                    fs, path = fsspec.core.url_to_fs(store)
                    if not fs.exists(path):
                        raise FileNotFoundError(f"Archive not found: {store}")
                elif not os.path.exists(store):
                    raise FileNotFoundError(f"Archive not found: {store}")
                # Use ZipStore directly for reading
                zip_store = zarr.storage.ZipStore(store, mode='r')
                result = Batches().snapshot(store=zip_store, caption=caption or 'Opening archive...', n_jobs=n_jobs, debug=debug)
                zip_store.close()
                # Unwrap single Stack from tuple
                if len(result) == 1 and isinstance(result[0], Stack):
                    return result[0]
                return result
            # Save self - write to temp directory, then zip
            temp_dir = tempfile.mkdtemp()
            try:
                utils_io.save(self, store=temp_dir, storage_options=None, compat=True,
                             caption=caption or 'Archiving...', n_jobs=n_jobs, debug=debug)
                # Create zip with specified compression level
                # Use fsspec for cloud storage support
                with fsspec.open(store, 'wb') as f:
                    with zipfile.ZipFile(f, 'w', zipfile.ZIP_DEFLATED, compresslevel=compression) as zf:
                        for root, dirs, files in os.walk(temp_dir):
                            for file in files:
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, temp_dir)
                                zf.write(file_path, arcname)
            finally:
                import shutil
                shutil.rmtree(temp_dir)
            return self

        # Save mode - args are batches
        return Batches(args).archive(store, caption=caption, compression=compression, n_jobs=n_jobs, debug=debug)

    # def downsample(self, *args, coarsen=None, resolution=60, func='mean', debug:bool=False):
    #     datas = []
    #     for arg in args:
    #         print ('type(arg)', type(arg))
    #         if isinstance(arg, (Stack, BatchComplex)):
    #             arg = Batch(arg)
    #             print (arg.isel(0)['033_069722_IW3'].data_vars)
    #         wrap = True if isinstance(arg, BatchWrap) else False
    #         print ('\ttype(arg)', type(arg), 'wrap', wrap)
    #         sample = next(iter(arg.values()))
    #         callback = utils_xarray.downsampler(sample, coarsen=coarsen, resolution=resolution, func=func, wrap=wrap, debug=debug)
    #         data = callback(arg)
    #         datas.append(BatchWrap(data) if wrap else Batch(data))
    #     return datas

    def to_dataframe(self,
                     datas: dict[str, xr.Dataset | xr.DataArray] | None = None,
                     crs:str|None='auto',
                     attr_start:str='BPR',
                     debug:bool=False
                     ) -> pd.DataFrame:
        """
        Return a Pandas DataFrame for all Stack scenes.

        Returns
        -------
        pandas.DataFrame
            The DataFrame containing Stack scenes.

        Examples
        --------
        df = stack.to_dataframe()
        """
        import geopandas as gpd
        from shapely import wkt
        import pandas as pd
        import numpy as np

        if datas is not None and not isinstance(datas, dict):
            raise ValueError(f'ERROR: datas is not None or a dict: {type(datas)}')
    
        if crs is not None and isinstance(crs, str) and crs == 'auto':
            crs = self.crs

        if datas is None:
            datas = self

        polarizations = [pol for pol in ['VV', 'VH', 'HH', 'HV'] if pol in next(iter(datas.values())).data_vars]
        #print ('polarizations', polarizations)

        # make attributes dataframe from datas
        processed_attrs = []
        for ds in datas.values():
            #print (data.id)
            attrs = [data_var for data_var in ds if ds[data_var].dims==('date',)][::-1]
            attr_start_idx = attrs.index(attr_start)
            for date_idx, date in enumerate(ds.date.values):
                processed_attr = {}
                for attr in attrs[:attr_start_idx+1]:
                    # Use isel + values to handle both numpy and dask arrays
                    value = ds[attr].isel(date=date_idx).values
                    # Compute if dask array
                    if hasattr(value, 'compute'):
                        value = value.compute()
                    # Extract scalar from 0-d array
                    if hasattr(value, 'item'):
                        value = value.item()
                    # Parse geometry WKT string
                    if attr == 'geometry':
                        processed_attr[attr] = wkt.loads(value)
                    else:
                        processed_attr[attr] = value
                processed_attrs.append(processed_attr)
                #print (processed_attr)
        df = gpd.GeoDataFrame(processed_attrs, crs=4326)
        #del df['date']
        #df['polarization'] = ','.join(polarizations)
        # convert polarizations to strings like "VV,VH" to pevent confusing with tuples in the dataframe
        df = df.assign(polarization=','.join(map(str, polarizations)))
        # reorder columns to the same order as preprocessor uses
        pol = df.pop("polarization")
        df.insert(3, "polarization", pol)
        # round for human readability
        df['BPR'] = df['BPR'].round(1)

        group_col = df.columns[0]
        burst_col = df.columns[1]
        #print ('df.columns[0]', df.columns[0])
        #print ('df.columns[:2][::-1].tolist()', df.columns[:2][::-1].tolist())
        df['startTime'] = pd.to_datetime(df['startTime'])
        #df['date'] = df['startTime'].dt.date.astype(str)
        df = df.sort_values(by=[group_col, burst_col]).set_index([group_col, burst_col])
        # move geometry to the end of the dataframe to be the most similar to insar_pygmtsar output
        df = df.loc[:, df.columns.drop("geometry").tolist() + ["geometry"]]

        # Skip CRS transformation for Engineering CRS (radar coordinates mode)
        # since burst geometry is always in WGS84 from metadata
        if crs is not None:
            from pyproj import CRS as ProjCRS
            try:
                proj_crs = ProjCRS.from_user_input(crs)
                if proj_crs.type_name == 'Engineering CRS':
                    # Can't transform to engineering CRS, keep WGS84
                    return df
            except Exception:
                pass
            return df.to_crs(crs)
        return df

    @staticmethod
    def _load_zarr_array(zarr_path, group_path, name, storage_options=None):
        """
        Load a single zarr array as numpy with direct file reading.

        Reads one array from a zarr group, applies scale_factor and fill_value
        decoding. File handles are opened and closed within this call —
        no persistent descriptors.

        Uses fsspec for unified local/remote access and numcodecs for
        zstd decompression, bypassing zarr library overhead.

        Parameters
        ----------
        zarr_path : str
            Path to zarr store. Supports local and remote (fsspec) paths:
            - Local: /path/to/data.zarr
            - S3: s3://bucket/path/data.zarr
            - GCS: gs://bucket/path/data.zarr (requires gcsfs)
            - Azure: az://container/path/data.zarr (requires adlfs)
        group_path : str
            Relative path to group within zarr store. Empty string for root.
        name : str
            Array name within the group.
        storage_options : dict, optional
            Options passed to fsspec filesystem.

        Returns
        -------
        np.ndarray
            Float32 2D array with NaN for masked values.
        """
        import numpy as np
        import json
        from numcodecs import Zstd
        import fsspec

        fs, root = fsspec.core.url_to_fs(zarr_path, **(storage_options or {}))
        base_path = f"{root}/{group_path}".rstrip('/') if group_path else root

        arr_meta_path = f"{base_path}/{name}/zarr.json"
        with fs.open(arr_meta_path, 'r') as f:
            meta = json.load(f)
        shape = tuple(meta['shape'])
        assert len(shape) >= 2, f"_load_zarr_array is for 2D+ delayed vars only, got {name} with shape {shape}"
        dtype = meta['data_type']
        attrs = meta.get('attributes', {})
        scale_factor = np.float32(attrs.get('scale_factor', 1.0))
        fill_value = attrs.get('_FillValue')

        # Single chunk only - for multi-chunk use _load_zarr_array_chunk
        chunk_suffix = '/'.join(['0'] * len(shape))
        chunk_path = f"{base_path}/{name}/c/{chunk_suffix}"
        codec = Zstd()
        with fs.open(chunk_path, 'rb') as f:
            raw = codec.decode(f.read())
        arr_int = np.frombuffer(raw, dtype=dtype).reshape(shape)

        arr_f32 = np.empty(shape, dtype=np.float32)
        np.multiply(arr_int, scale_factor, out=arr_f32, casting='unsafe')
        if fill_value is not None:
            np.putmask(arr_f32, arr_int == fill_value, np.nan)
        return arr_f32

    @staticmethod
    def _load_zarr_array_chunk(zarr_path, group_path, name, chunk_idx, chunk_shape,
                                disk_chunk_shape, scale_factor, fill_value, dtype, storage_options=None):
        """Load ONE chunk of a zarr array. Returns NaN array if chunk file doesn't exist."""
        import numpy as np
        from numcodecs import Zstd
        import fsspec

        fs, root = fsspec.core.url_to_fs(zarr_path, **(storage_options or {}))
        base_path = f"{root}/{group_path}".rstrip('/')

        iy, ix = chunk_idx
        chunk_path = f"{base_path}/{name}/c/{iy}/{ix}"

        # Check if chunk exists - return NaN array if not (like native zarr)
        if not fs.exists(chunk_path):
            return np.full(chunk_shape, np.nan, dtype=np.float32)

        codec = Zstd()
        with fs.open(chunk_path, 'rb') as f:
            raw = codec.decode(f.read())
        # Zarr pads edge chunks to full disk_chunk_shape
        arr_full = np.frombuffer(raw, dtype=dtype).reshape(disk_chunk_shape)
        arr_chunk = arr_full[:chunk_shape[0], :chunk_shape[1]].copy()
        del arr_full

        arr_f32 = np.empty(chunk_shape, dtype=np.float32)
        np.multiply(arr_chunk, scale_factor, out=arr_f32, casting='unsafe')
        if fill_value is not None:
            np.putmask(arr_f32, arr_chunk == fill_value, np.nan)
        return arr_f32

    @staticmethod
    def _get_zarr_array_meta(zarr_path, group_path, name, storage_options=None):
        """Get zarr array metadata: shape, chunks, scale, fill_value, dtype."""
        import numpy as np
        import json
        import fsspec

        fs, root = fsspec.core.url_to_fs(zarr_path, **(storage_options or {}))
        base_path = f"{root}/{group_path}".rstrip('/')

        with fs.open(f"{base_path}/{name}/zarr.json", 'r') as f:
            meta = json.load(f)

        shape = tuple(meta['shape'])
        chunks = tuple(meta.get('chunk_grid', {}).get('configuration', {}).get('chunk_shape', shape))
        attrs = meta.get('attributes', {})
        scale_factor = np.float32(attrs.get('scale_factor', 1.0))
        fill_value = attrs.get('_FillValue')
        dtype = meta['data_type']

        return shape, chunks, scale_factor, fill_value, dtype

    @staticmethod
    def _load_zarr_complex_chunk(zarr_path, group_path, chunk_idx, chunk_shape,
                                  disk_chunk_shape, scale, fill_value, storage_options=None):
        """
        Load ONE zarr chunk of complex64 data. Returns NaN array if chunk files don't exist.

        Called separately for each chunk - one reader call per chunk.
        Handles edge chunks where disk_chunk_shape > chunk_shape (zarr pads edges).
        """
        import numpy as np
        from numcodecs import Zstd
        import fsspec

        fs, root = fsspec.core.url_to_fs(zarr_path, **(storage_options or {}))
        base_path = f"{root}/{group_path}".rstrip('/')
        codec = Zstd()
        iy, ix = chunk_idx

        re_path = f"{base_path}/re/c/{iy}/{ix}"
        im_path = f"{base_path}/im/c/{iy}/{ix}"

        # Check if chunk exists - return NaN array if not (like native zarr)
        if not fs.exists(re_path) or not fs.exists(im_path):
            return np.full(chunk_shape, np.nan + 0j, dtype=np.complex64)

        # Read real part (disk has full chunk size, slice to logical size)
        with fs.open(re_path, 'rb') as f:
            re_full = np.frombuffer(codec.decode(f.read()), dtype=np.int16).reshape(disk_chunk_shape)
        re_int16 = re_full[:chunk_shape[0], :chunk_shape[1]]
        del re_full

        # Read imaginary part
        with fs.open(im_path, 'rb') as f:
            im_full = np.frombuffer(codec.decode(f.read()), dtype=np.int16).reshape(disk_chunk_shape)
        im_int16 = im_full[:chunk_shape[0], :chunk_shape[1]]
        del im_full

        # Build complex output
        data = np.empty(chunk_shape, dtype=np.complex64)
        if fill_value is not None:
            mask = (re_int16 == fill_value) | (im_int16 == fill_value)

        np.multiply(re_int16, scale, out=data.real, casting='unsafe')
        del re_int16
        np.multiply(im_int16, scale, out=data.imag, casting='unsafe')
        del im_int16

        if fill_value is not None:
            np.putmask(data, mask, np.nan + 0j)

        return data

    @staticmethod
    def _load_zarr_complex(zarr_path, group_path, storage_options=None):
        """
        Load complex64 array from zarr - single chunk case only (S1 format).
        For multi-chunk NISAR, use _load_zarr_complex_chunk per chunk instead.
        """
        import numpy as np
        import json
        from numcodecs import Zstd
        import fsspec

        fs, root = fsspec.core.url_to_fs(zarr_path, **(storage_options or {}))
        base_path = f"{root}/{group_path}".rstrip('/') if group_path else root

        meta_path = f"{base_path}/re/zarr.json"
        with fs.open(meta_path, 'r') as f:
            meta = json.load(f)
        shape = tuple(meta['shape'])
        scale = np.float32(meta['attributes']['scale_factor'])
        fill_value = meta['attributes'].get('_FillValue')

        codec = Zstd()
        data = np.empty(shape, dtype=np.complex64)

        with fs.open(f"{base_path}/re/c/0/0", 'rb') as f:
            re_int16 = np.frombuffer(codec.decode(f.read()), dtype=np.int16).reshape(shape)
        if fill_value is not None:
            mask = (re_int16 == fill_value)
        np.multiply(re_int16, scale, out=data.real, casting='unsafe')
        del re_int16

        with fs.open(f"{base_path}/im/c/0/0", 'rb') as f:
            im_int16 = np.frombuffer(codec.decode(f.read()), dtype=np.int16).reshape(shape)
        if fill_value is not None:
            mask |= (im_int16 == fill_value)
        np.multiply(im_int16, scale, out=data.imag, casting='unsafe')
        del im_int16

        if fill_value is not None:
            np.putmask(data, mask, np.nan + 0j)

        return data

    @staticmethod
    def _get_zarr_slc_meta(zarr_path, group_path, storage_options=None):
        """Get SLC zarr metadata: shape, chunks, scale, fill_value."""
        import numpy as np
        import json
        import fsspec

        fs, root = fsspec.core.url_to_fs(zarr_path, **(storage_options or {}))
        base_path = f"{root}/{group_path}".rstrip('/')

        with fs.open(f"{base_path}/re/zarr.json", 'r') as f:
            meta = json.load(f)

        shape = tuple(meta['shape'])
        chunks = tuple(meta.get('chunk_grid', {}).get('configuration', {}).get('chunk_shape', shape))
        scale = np.float32(meta['attributes']['scale_factor'])
        fill_value = meta['attributes'].get('_FillValue')

        return shape, chunks, scale, fill_value

    def load(self, urls:str | list | dict[str, str], storage_options:dict[str, str]|None=None,
             allow_rechunk:bool=False, debug:bool=False):
        import numpy as np
        import dask
        import dask.array as da
        import xarray as xr
        import pandas as pd
        import geopandas as gpd
        import zarr
        from shapely import wkt
        import os
        from insardev_toolkit import progressbar_joblib
        from tqdm.auto import tqdm
        import joblib
        import warnings
        # suppress the "Sending large graph of size …"
        warnings.filterwarnings(
            'ignore',
            category=UserWarning,
            module=r'distributed\.client',
            message=r'Sending large graph of size .*'
        )
        from distributed import get_client, WorkerPlugin
        class IgnoreDaskDivide(WorkerPlugin):
            def setup(self, worker):
                # suppress the "RuntimeWarning: invalid value encountered in divide"
                warnings.filterwarnings(
                    "ignore",
                    category=RuntimeWarning,
                    module=r'dask\._task_spec'
                )
        client = get_client()
        client.register_plugin(IgnoreDaskDivide(), name='ignore_divide')

        # Whitelist of scalar attrs actually used by insardev processing
        # Alignment params (sub_int_*, stretch_*, a_stretch_*, ashift, rshift) are
        # excluded - they're only needed for debugging alignment quality in insardev_pygmtsar
        _USED_SCALAR_ATTRS = {
            # Mission/acquisition metadata
            'startTime', 'polarization', 'burst', 'flightDirection',
            'pathNumber', 'subswath', 'mission', 'beamModeType',
            'fullBurstID', 'geometry',  # Used in to_dataframe()
            # Radar parameters for phase2elevation, LOS calculations
            'radar_wavelength', 'near_range', 'PRF',
            'SC_height_start', 'SC_height_end', 'earth_radius',
            'rng_samp_rate', 'num_lines',
            # Baseline
            'BPR', 'BPT', 'B_perpendicular', 'B_parallel',
            # align_elevation() parameters
            'linesPerBurst', 'clock_start', 'ksr', 'ker',
        }

        def store_open_group_delayed(zarr_path, group):
            """
            Open a fullBurstID group using delayed loading for complex data.

            This avoids the complex xarray concat graph by creating simple
            dask.delayed tasks for each date's data. File handles are opened
            and closed cleanly for each load operation.
            """
            import rioxarray
            root = zarr.open_consolidated(zarr_path, zarr_format=3, mode='r')
            grp = root[group]

            # Get burst subgroups (excluding transform)
            burst_keys = [k for k in grp.keys() if k != 'transform']

            # Collect metadata for all bursts (no 2D data loading)
            burst_infos = []
            spatial_ref = None
            for burst_key in burst_keys:
                burst_grp = grp[burst_key]
                burst_path = f"{group}/{burst_key}"

                # Open to get metadata only
                ds = xr.open_zarr(burst_grp.store, group=burst_grp.path,
                                  consolidated=True, zarr_format=3)

                shape = burst_grp['re'].shape
                date = np.datetime64(ds.attrs['startTime'], 's')
                polarization = ds.attrs['polarization']
                burst_name = ds.attrs['burst']

                # Capture spatial_ref from reference burst (BPR=0)
                if spatial_ref is None and ds.attrs.get('BPR', 1) == 0:
                    spatial_ref = ds.attrs.get('spatial_ref')

                # Extract scalar attrs (only whitelisted keys used by insardev)
                scalar_attrs = {}
                array_attrs = {}
                skip_attrs = {'Conventions', 'spatial_ref'}
                for k, v in ds.attrs.items():
                    if k in skip_attrs:
                        continue
                    if isinstance(v, (list, tuple)):
                        array_attrs[k] = np.array(v)
                    elif k in _USED_SCALAR_ATTRS:
                        # Only include whitelisted scalar attrs
                        if isinstance(v, str):
                            scalar_attrs[k] = v
                        else:
                            scalar_attrs[k] = float(v) if isinstance(v, (int, float)) else v

                # Store only what we need (not the full dataset!)
                burst_infos.append({
                    'polarization': polarization,
                    'burst_path': burst_path,
                    'burst_name': burst_name.replace(polarization, 'XX'),
                    'shape': shape,
                    'date': date,
                    'y': ds.y.values,
                    'x': ds.x.values,
                    'scalar_attrs': scalar_attrs,
                    'array_attrs': array_attrs,
                })
                ds.close()  # Close xarray dataset after extracting metadata

            # Group by polarization and sort by date
            polarizations = np.unique([info['polarization'] for info in burst_infos])

            datas = []
            for polarization in polarizations:
                pol_infos = sorted(
                    [info for info in burst_infos if info['polarization'] == polarization],
                    key=lambda x: x['date']
                )

                # Create delayed dask arrays for each date
                # Resource annotation limits concurrent loads to prevent memory accumulation
                # Workers must be configured with resources={'load': N} to limit concurrency
                delayed_arrays = []
                for info in pol_infos:
                    shape = info['shape']
                    # Get chunk metadata
                    _, zarr_chunks, scale, fill_value = Stack._get_zarr_slc_meta(
                        zarr_path, info['burst_path'], storage_options
                    )
                    n_chunks_y = (shape[0] + zarr_chunks[0] - 1) // zarr_chunks[0]
                    n_chunks_x = (shape[1] + zarr_chunks[1] - 1) // zarr_chunks[1]

                    if n_chunks_y == 1 and n_chunks_x == 1:
                        # Single chunk (S1): one reader for entire array
                        with dask.annotate(resources={'load': 1}):
                            delayed_load = dask.delayed(Stack._load_zarr_complex)(
                                zarr_path, info['burst_path'], storage_options
                            )
                        arr = da.from_delayed(delayed_load, shape=shape, dtype=np.complex64)
                    else:
                        # Multi-chunk (NISAR): one reader call per chunk
                        chunk_rows = []
                        for iy in range(n_chunks_y):
                            chunk_cols = []
                            for ix in range(n_chunks_x):
                                # Logical chunk shape (may be smaller at edges)
                                y0, y1 = iy * zarr_chunks[0], min((iy + 1) * zarr_chunks[0], shape[0])
                                x0, x1 = ix * zarr_chunks[1], min((ix + 1) * zarr_chunks[1], shape[1])
                                chunk_shape = (y1 - y0, x1 - x0)

                                with dask.annotate(resources={'load': 1}):
                                    delayed_chunk = dask.delayed(Stack._load_zarr_complex_chunk)(
                                        zarr_path, info['burst_path'], (iy, ix),
                                        chunk_shape, zarr_chunks, scale, fill_value, storage_options
                                    )
                                chunk_arr = da.from_delayed(delayed_chunk, shape=chunk_shape, dtype=np.complex64)
                                chunk_cols.append(chunk_arr)
                            chunk_rows.append(chunk_cols)
                        arr = da.block(chunk_rows)

                    arr = arr[np.newaxis, :, :]  # Add date dim: (y, x) -> (1, y, x)
                    delayed_arrays.append(arr)

                # Stack all dates: (n_dates, y, x)
                stacked = da.concatenate(delayed_arrays, axis=0)

                # Create xarray DataArray
                dates = [info['date'] for info in pol_infos]
                data_arr = xr.DataArray(
                    stacked,
                    dims=['date', 'y', 'x'],
                    coords={
                        'date': np.array(dates),
                        'y': pol_infos[0]['y'],
                        'x': pol_infos[0]['x'],
                    },
                )

                # Create dataset with polarization as variable name
                data_ds = xr.Dataset({polarization: data_arr})

                # Add scalar metadata as variables along date dimension
                data_ds['burst'] = xr.DataArray([info['burst_name'] for info in pol_infos], dims=['date'])

                # Add all scalar attrs as variables (replicated per date)
                # Preserve original order from first burst (to_dataframe expects specific order)
                # Exclude 'burst' (handled above with XX replacement) and 'polarization' (per-pol)
                excluded_keys = {'burst', 'polarization'}
                all_scalar_keys = [k for k in pol_infos[0]['scalar_attrs'].keys() if k not in excluded_keys]
                # Add any keys from other dates that might be missing
                for info in pol_infos[1:]:
                    for k in info['scalar_attrs'].keys():
                        if k not in all_scalar_keys and k not in excluded_keys:
                            all_scalar_keys.append(k)
                for key in all_scalar_keys:
                    vals = [info['scalar_attrs'].get(key, np.nan) for info in pol_infos]
                    # Check if all values are numeric
                    if all(isinstance(v, (int, float, np.number)) for v in vals):
                        data_ds[key] = xr.DataArray(np.array(vals, dtype=np.float64), dims=['date'])
                    else:
                        # String values
                        data_ds[key] = xr.DataArray(vals, dims=['date'])

                # Add array attrs (e.g., polynomial coefficients) - take from first burst
                first_info = pol_infos[0]
                for key, arr in first_info['array_attrs'].items():
                    if arr.ndim == 1:
                        # Stack arrays from all dates: (n_dates, n_coef)
                        stacked = np.stack([info['array_attrs'].get(key, arr) for info in pol_infos])
                        data_ds[key] = xr.DataArray(stacked, dims=['date', 'coef'])

                datas.append(data_ds)

            # Merge polarizations
            ds = xr.merge(datas, compat='no_conflicts', combine_attrs='override')
            del datas

            # Load transform: zarr handles metadata/coords, custom reader for 2D chunks
            grp_transform = grp['transform']
            transform = xr.open_zarr(grp_transform.store, group=grp_transform.path,
                                     consolidated=True, zarr_format=3)

            # Coords eagerly (small 1D arrays)
            ds = ds.assign_coords(x=transform.x.values, y=transform.y.values)

            # 2D vars as lazy dask arrays via custom reader (no persistent file descriptors)
            # One reader call per chunk for memory efficiency
            transform_path = f"{group}/transform"
            for var in transform.data_vars:
                shape, zarr_chunks, scale_factor, fill_value, dtype = Stack._get_zarr_array_meta(
                    zarr_path, transform_path, var, storage_options
                )
                n_chunks_y = (shape[0] + zarr_chunks[0] - 1) // zarr_chunks[0]
                n_chunks_x = (shape[1] + zarr_chunks[1] - 1) // zarr_chunks[1]

                if n_chunks_y == 1 and n_chunks_x == 1:
                    # Single chunk: one reader for entire array
                    delayed_load = dask.delayed(Stack._load_zarr_array)(
                        zarr_path, transform_path, var, storage_options
                    )
                    arr = da.from_delayed(delayed_load, shape=shape, dtype=np.float32)
                else:
                    # Multi-chunk: one reader per chunk
                    chunk_rows = []
                    for iy in range(n_chunks_y):
                        chunk_cols = []
                        for ix in range(n_chunks_x):
                            y0, y1 = iy * zarr_chunks[0], min((iy + 1) * zarr_chunks[0], shape[0])
                            x0, x1 = ix * zarr_chunks[1], min((ix + 1) * zarr_chunks[1], shape[1])
                            chunk_shape = (y1 - y0, x1 - x0)

                            delayed_chunk = dask.delayed(Stack._load_zarr_array_chunk)(
                                zarr_path, transform_path, var, (iy, ix),
                                chunk_shape, zarr_chunks, scale_factor, fill_value, dtype, storage_options
                            )
                            chunk_arr = da.from_delayed(delayed_chunk, shape=chunk_shape, dtype=np.float32)
                            chunk_cols.append(chunk_arr)
                        chunk_rows.append(chunk_cols)
                    arr = da.block(chunk_rows)

                ds[var] = xr.DataArray(arr, dims=['y', 'x'])

            # Set spatial_ref
            if spatial_ref is None:
                spatial_ref = transform.attrs.get('spatial_ref')
            if spatial_ref is None:
                raise KeyError('spatial_ref')
            ds.attrs['spatial_ref'] = spatial_ref
            ds.rio.write_crs(spatial_ref, inplace=True)

            # Close zarr resources (metadata only, no lazy data references)
            transform.close()
            root.store.close()

            return group, ds

        if isinstance(urls, str):
            # note: isinstance(urls, zarr.storage.ZipStore) can be loaded too but it is less efficient
            urls = os.path.expanduser(urls)
            zarr_path = urls  # Store for delayed loading
            root = zarr.open_consolidated(urls, zarr_format=3, mode='r')
            groups = list(root.group_keys())
            del root  # Close the root - we'll reopen in each group loader

            # Use the new delayed loading approach
            with progressbar_joblib.progressbar_joblib(tqdm(desc='Loading Dataset...'.ljust(25), total=len(groups))) as progress_bar:
                dss = joblib.Parallel(n_jobs=-1, backend='loky')\
                    (joblib.delayed(store_open_group_delayed)(zarr_path, group) for group in groups)
            # list of key - dataset converted to dict and appended to the existing dict
            self.update(dss)
        # elif isinstance(urls, FsspecStore):
        #     root = zarr.open_consolidated(urls, zarr_format=3, mode='r')
        #     dss = []
        #     for group in tqdm(list(root.group_keys()), desc='Loading Store'):
        #         dss.append(store_open_group(root, group))
        #     self.dss = dict(dss)
        #     del dss
        elif isinstance(urls, list) or isinstance(urls, pd.DataFrame):
            # load bursts and transform specified by URLs
            # this allows to load from multiple locations with precise control of the data
            if isinstance(urls, list):
                print ('NOTE: urls is a list, using it as is.')
                df = pd.DataFrame(urls, columns=['url'])
                df['fullBurstID'] = df['url'].str.rsplit('/', n=2).str[1]
                df['burst'] = df["url"].str.rsplit("/", n=2).str[2]
                urls = df.sort_values(by=['fullBurstID', 'burst']).set_index(['fullBurstID', 'burst'])
                print (urls.head())
            elif isinstance(urls.index, pd.MultiIndex) and urls.index.nlevels == 2 and len(urls.columns) == 1:
                print ('NOTE: Detected Pandas Dataframe with MultiIndex, using first level as fullBurstID and the first column as URLs.')
                #groups = {key: group.index.get_level_values(1).tolist() for key, group in urls.groupby(level=0)}
                #groups = {key: group[urls.columns[0]].tolist() for key, group in urls.groupby(level=0)}
            else:
                raise ValueError(f'ERROR: urls is not a list, or Pandas Dataframe with multiindex: {type(urls)}')

            dss = {}
            for fullBurstID in tqdm(urls.index.get_level_values(0).unique(), desc='Loading Datasets...'.ljust(25)):
                df = urls[urls.index.get_level_values(0) == fullBurstID]
                burst_urls = df[df.index.get_level_values(1) != 'transform'].iloc[:,0].values
                transform_url = df[df.index.get_level_values(1) == 'transform'].iloc[:,0].values[0]

                # Read burst metadata eagerly from each URL (attrs, shape, coords)
                burst_infos = []
                spatial_ref = None
                for burst_url in burst_urls:
                    bds = xr.open_zarr(burst_url, consolidated=True, zarr_format=3,
                                       storage_options=storage_options)
                    shape = (bds.dims['y'], bds.dims['x'])
                    date = np.datetime64(bds.attrs['startTime'], 's')
                    polarization = bds.attrs['polarization']
                    burst_name = bds.attrs['burst']

                    if spatial_ref is None and bds.attrs.get('BPR', 1) == 0:
                        spatial_ref = bds.attrs.get('spatial_ref')

                    # Extract scalar attrs (only whitelisted keys used by insardev)
                    scalar_attrs = {}
                    array_attrs = {}
                    for k, v in bds.attrs.items():
                        if k in {'Conventions', 'spatial_ref'}:
                            continue
                        if isinstance(v, (list, tuple)):
                            array_attrs[k] = np.array(v)
                        elif k in _USED_SCALAR_ATTRS:
                            # Only include whitelisted scalar attrs
                            if isinstance(v, str):
                                scalar_attrs[k] = v
                            else:
                                scalar_attrs[k] = float(v) if isinstance(v, (int, float)) else v

                    burst_infos.append({
                        'url': burst_url,
                        'polarization': polarization,
                        'burst_name': burst_name.replace(polarization, 'XX'),
                        'shape': shape,
                        'date': date,
                        'y': bds.y.values,
                        'x': bds.x.values,
                        'scalar_attrs': scalar_attrs,
                        'array_attrs': array_attrs,
                    })
                    bds.close()

                # Build dataset same as primary path: delayed complex arrays
                polarizations = np.unique([info['polarization'] for info in burst_infos])
                datas = []
                for polarization in polarizations:
                    pol_infos = sorted(
                        [info for info in burst_infos if info['polarization'] == polarization],
                        key=lambda x: x['date']
                    )

                    # Resource annotation limits concurrent loads to prevent memory accumulation
                    delayed_arrays = []
                    for info in pol_infos:
                        with dask.annotate(resources={'load': 1}):
                            delayed_load = dask.delayed(Stack._load_zarr_complex)(
                                info['url'], '', storage_options
                            )
                        arr = da.from_delayed(delayed_load, shape=info['shape'], dtype=np.complex64)
                        arr = arr[np.newaxis, :, :]
                        delayed_arrays.append(arr)

                    stacked = da.concatenate(delayed_arrays, axis=0)
                    dates = [info['date'] for info in pol_infos]
                    data_arr = xr.DataArray(
                        stacked,
                        dims=['date', 'y', 'x'],
                        coords={
                            'date': np.array(dates),
                            'y': pol_infos[0]['y'],
                            'x': pol_infos[0]['x'],
                        },
                    )
                    data_ds = xr.Dataset({polarization: data_arr})
                    data_ds['burst'] = xr.DataArray([info['burst_name'] for info in pol_infos], dims=['date'])

                    # Exclude 'burst' (handled above with XX replacement) and 'polarization' (per-pol)
                    excluded_keys = {'burst', 'polarization'}
                    all_scalar_keys = [k for k in pol_infos[0]['scalar_attrs'].keys() if k not in excluded_keys]
                    for info in pol_infos[1:]:
                        for k in info['scalar_attrs'].keys():
                            if k not in all_scalar_keys and k not in excluded_keys:
                                all_scalar_keys.append(k)
                    for key in all_scalar_keys:
                        vals = [info['scalar_attrs'].get(key, np.nan) for info in pol_infos]
                        if all(isinstance(v, (int, float, np.number)) for v in vals):
                            data_ds[key] = xr.DataArray(np.array(vals, dtype=np.float64), dims=['date'])
                        else:
                            data_ds[key] = xr.DataArray(vals, dims=['date'])

                    first_info = pol_infos[0]
                    for key, arr in first_info['array_attrs'].items():
                        if arr.ndim == 1:
                            stacked_arr = np.stack([info['array_attrs'].get(key, arr) for info in pol_infos])
                            data_ds[key] = xr.DataArray(stacked_arr, dims=['date', 'coef'])

                    datas.append(data_ds)

                # Merge polarizations
                ds = xr.merge(datas, compat='no_conflicts', combine_attrs='override')
                del datas

                # Load transform: zarr for metadata/coords, custom reader for 2D
                transform = xr.open_zarr(transform_url, consolidated=True, zarr_format=3,
                                         storage_options=storage_options)
                ds = ds.assign_coords(x=transform.x.values, y=transform.y.values)
                for var in transform.data_vars:
                    shape = transform[var].shape
                    delayed_load = dask.delayed(Stack._load_zarr_array)(
                        transform_url, '', var, storage_options
                    )
                    ds[var] = xr.DataArray(
                        da.from_delayed(delayed_load, shape=shape, dtype=np.float32),
                        dims=['y', 'x']
                    )

                if spatial_ref is None:
                    spatial_ref = transform.attrs.get('spatial_ref')
                if spatial_ref is None:
                    raise KeyError('spatial_ref')
                ds.attrs['spatial_ref'] = spatial_ref
                ds.rio.write_crs(spatial_ref, inplace=True)
                transform.close()

                dss[fullBurstID] = ds

            #assert len(np.unique([ds.rio.crs.to_epsg() for ds in dss])) == 1, 'All datasets must have the same coordinate reference system'
            self.update(dss)

        # Check for duplicate dates in each burst
        for key, ds in self.items():
            dates = ds.date.values
            unique, counts = np.unique(dates, return_counts=True)
            if (counts > 1).any():
                duplicates = unique[counts > 1]
                raise ValueError(f'Burst {key} contains duplicate dates: {duplicates}. '
                                 'This may be caused by corrupted data or library issues. '
                                 'Try restarting the runtime and reloading the data.')

        # Rechunk spatial variables only (non-spatial stay as-is)
        from .utils_dask import get_dask_chunk_size_mb, rechunk2d
        dask_chunk_mb = get_dask_chunk_size_mb()
        note_printed = False

        for key, ds in self.items():
            rechunked_vars = {}
            for var in ds.data_vars:
                arr = ds[var]
                # Only touch spatial variables (y, x dims)
                if not (arr.ndim in (2, 3) and arr.dims[-2:] == ('y', 'x')):
                    continue
                if allow_rechunk:
                    # Use rechunk2d for uniform chunk sizes
                    y_size, x_size = arr.shape[-2], arr.shape[-1]
                    element_bytes = arr.dtype.itemsize
                    optimal = rechunk2d((y_size, x_size), element_bytes)
                    if arr.ndim == 3:
                        chunks = {arr.dims[0]: 1, 'y': optimal['y'], 'x': optimal['x']}
                    else:
                        chunks = {'y': optimal['y'], 'x': optimal['x']}
                    rechunked_vars[var] = arr.chunk(chunks)
                # else: preserve existing zarr chunks (don't rechunk)
            if rechunked_vars:
                self[key] = ds.assign(rechunked_vars)
            if not note_printed:
                if allow_rechunk:
                    print(f"NOTE load: rechunking to dask.config['array.chunk-size']={dask_chunk_mb} MB")
                else:
                    print(f"NOTE load: preserving zarr chunks (allow_rechunk=False)")
                note_printed = True

        return self


    def pairs(self,
              pairs: list[tuple[str|int, str|int]] | np.ndarray | pd.DataFrame
              ) -> Batches:
        """
        Select SLC data organized by interferometric pairs.

        Returns reference and repeat SLC data as a Batches, ready for
        interferogram computation via multiplication.

        Parameters
        ----------
        pairs : list, np.ndarray, or pd.DataFrame
            Pairs of dates as [(ref1, rep1), (ref2, rep2), ...].
            Dates can be indices (int) or date strings.

        Returns
        -------
        Batches
            Batches containing [BatchComplex(ref), BatchComplex(rep)]
            with 'pair' dimension and ref/rep date coordinates.

        Examples
        --------
        # Get paired SLC data
        ref, rep = stack.pairs(baseline.tolist())

        # Manual phase difference
        phasediff = ref * rep.conj()

        # With filtering
        intf = (ref * rep.conj()).gaussian(wavelength=30).angle()
        """
        import numpy as np

        pairs = np.array(pairs if isinstance(pairs[0], (list, tuple, np.ndarray)) else [pairs])

        # Check for duplicate pairs
        unique, counts = np.unique(pairs, axis=0, return_counts=True)
        if (counts > 1).any():
            duplicates = unique[counts > 1]
            raise ValueError(f'Input pairs contain duplicates: {duplicates.tolist()}')

        ref_dates = pairs[:, 0]
        rep_dates = pairs[:, 1]
        n_pairs = len(ref_dates)

        # Rename date->pair and reset to integer index
        data1 = self.isel(date=ref_dates).rename(date='pair').map(lambda ds: ds.assign_coords(pair=np.arange(n_pairs)))
        data2 = self.isel(date=rep_dates).rename(date='pair').map(lambda ds: ds.assign_coords(pair=np.arange(n_pairs)))

        # BPR differences aligned with pair dimension: BPR(rep) - BPR(ref)
        # Keep as per-burst dict structure (each burst has its own BPR)
        bpr = data2[['BPR']] - data1[['BPR']]

        # Store original datetime values for ref/rep (already materialized via .values)
        ref_values = self.isel(date=ref_dates).coords['date'].values
        rep_values = self.isel(date=rep_dates).coords['date'].values

        def add_pair_coords(batch):
            # Add ref/rep/BPR as non-dimension coordinates along pair dimension
            return batch.assign_coords(
                ref=('pair', ref_values),
                rep=('pair', rep_values),
                BPR=('pair', bpr)
            )

        ref_batch = add_pair_coords(BatchComplex(data1))
        rep_batch = add_pair_coords(BatchComplex(data2))

        return Batches([ref_batch, rep_batch])

    def elevation(self, phase: Batch | float | list | "np.ndarray", baseline: float | None = None, transform: Batch | None = None) -> "Batch | float | np.ndarray":
        """Compute elevation (meters) from unwrapped phase grids in radar coordinates.

        Parameters
        ----------
        phase : Batch | float | list | np.ndarray
            Unwrapped phase grids (e.g., output of unwrap2d()), or a scalar/array
            of phase values for quick height-per-fringe calculations.
        baseline : float | None, optional
            Perpendicular baseline in meters. Required when phase is scalar/array.
            If ``None`` and phase is Batch, use the burst-specific ``BPR``.
        transform : Batch | None, optional
            Transform batch containing look vectors; its `.incidence()` is used.
            If ``None``, defaults to ``self.transform()``.

        Returns
        -------
        Batch | float | np.ndarray
            Elevation grids as float32 datasets, or scalar/array if input was scalar/array.
        """
        import xarray as xr
        import numpy as np

        # Handle scalar/list/array input for quick calculations
        if not isinstance(phase, BatchCore):
            if baseline is None:
                raise ValueError("baseline is required when phase is a scalar/list/array")
            if transform is None:
                transform = self.transform()

            # Get average parameters from first burst
            first_key = next(iter(transform.keys()))
            tfm = transform[first_key]

            def _scalar_from_ds(ds, name: str):
                if name in ds:
                    var = ds[name]
                    if var.ndim == 0:
                        return float(var.item())
                    # For per-date variables, return mean value
                    return float(var.mean().item())
                return ds.attrs.get(name)

            wavelength = _scalar_from_ds(tfm, 'radar_wavelength')
            slant_start = _scalar_from_ds(tfm, 'SC_height_start')
            slant_end = _scalar_from_ds(tfm, 'SC_height_end')
            if wavelength is None or slant_start is None or slant_end is None:
                raise KeyError(f"Missing parameters in transform for burst {first_key}: radar_wavelength, SC_height_start, SC_height_end")
            slant_range = (slant_start + slant_end) / 2  # average slant range

            # Get average incidence angle
            inc_batch = transform.incidence()
            inc_da = inc_batch[first_key]['incidence']
            incidence = float(inc_da.mean().compute())

            # Convert input to numpy array
            phase_arr = np.asarray(phase)
            is_scalar = phase_arr.ndim == 0

            # Height from phase formula: h = -λ * φ * R * cos(incidence) / (4π * B⊥)
            elev = -(wavelength * phase_arr * slant_range * np.cos(incidence) / (4 * np.pi * baseline))

            # Return same type as input, rounded to 3 decimals
            if is_scalar:
                return round(float(elev), 3)
            return np.round(elev, 3)

        # Default to self.transform() when called on Stack and transform not provided
        if transform is None:
            transform = self.transform()
        incidence_batch = transform.incidence()
        out: dict[str, xr.Dataset] = {}

        for key, phase_ds in phase.items():
            if key not in incidence_batch:
                raise KeyError(f'Missing incidence for key: {key}')

            tfm = transform[key]

            def _scalar_from_ds(ds, name: str):
                if name in ds:
                    var = ds[name]
                    if var.ndim == 0:
                        return float(var.item())
                    # For per-date variables, return mean value
                    return float(var.mean().item())
                return ds.attrs.get(name)

            wavelength = _scalar_from_ds(tfm, 'radar_wavelength')
            slant_start = _scalar_from_ds(tfm, 'SC_height_start')
            slant_end = _scalar_from_ds(tfm, 'SC_height_end')
            if wavelength is None or slant_start is None or slant_end is None:
                raise KeyError(f"Missing parameters in transform for burst {key}: radar_wavelength, SC_height_start, SC_height_end")

            # Get BPR - either scalar or per-pair DataArray for broadcasting
            if baseline is not None:
                bpr = float(baseline)
            elif 'BPR' in phase_ds.coords:
                # Use BPR as DataArray to broadcast correctly across pairs
                bpr = phase_ds.coords['BPR']
            else:
                raise KeyError(f"Missing baseline (BPR) for burst {key}")

            inc_da = incidence_batch[key]['incidence']
            slant_range = xr.DataArray(
                np.linspace(slant_start, slant_end, inc_da.sizes['x']),
                coords={'x': inc_da.coords['x']},
                dims=('x',)
            )

            elev_vars: dict[str, xr.DataArray] = {}
            for var_name, data in phase_ds.data_vars.items():
                if 'y' in data.coords and 'x' in data.coords:
                    incidence = inc_da.interp(y=data.y, x=data.x, method='linear')
                    slant = slant_range.interp(x=data.x)
                else:
                    incidence = inc_da.reindex_like(data, method='nearest')
                    slant = slant_range.reindex_like(data.x, method='nearest')

                # Height from phase formula (PyGMTSAR convention):
                # h = -λ * φ * R * cos(incidence) / (4π * B⊥)
                # BPR broadcasts across pair dimension automatically
                elev = -(wavelength * data * slant * xr.ufuncs.cos(incidence) / (4 * np.pi * bpr))
                name = 'ele' if len(phase_ds.data_vars) == 1 else f'{var_name}_ele'
                elev_vars[name] = elev.astype('float32')

            out[key] = xr.Dataset(elev_vars, coords=phase_ds.coords, attrs=phase_ds.attrs)

        return Batch(out)

    def displacement_los(self, phase: Batch, transform: Batch = None) -> Batch:
        """Compute line-of-sight displacement (meters) from unwrapped phase.

        Parameters
        ----------
        phase : Batch
            Unwrapped phase grids in radar coordinates.
        transform : Batch, optional
            Transform batch providing mission constants; only the first burst is
            queried for ``radar_wavelength``. If None, uses self.transform().

        Returns
        -------
        Batch
            LOS displacement grids (meters), lazily scaled by the mission wavelength.
        """
        import numpy as np
        import xarray as xr

        # Default to self.transform() when not provided
        if transform is None:
            transform = self.transform()

        if not transform:
            raise ValueError('transform must contain at least one burst with radar_wavelength')

        transform_first = next(iter(transform.values()))

        def _scalar_from_ds(ds, name: str):
            if name in ds:
                var = ds[name]
                if var.ndim == 0:
                    return var.item()
                # Handle 1D+ arrays - verify all values are identical
                elif var.ndim >= 1:
                    values = var.values.flatten()
                    unique = np.unique(values)
                    if len(unique) != 1:
                        raise ValueError(f'{name} has multiple distinct values: {unique}')
                    return unique[0]
            return ds.attrs.get(name)

        wavelength = _scalar_from_ds(transform_first, 'radar_wavelength')
        if wavelength is None:
            raise KeyError('Missing radar_wavelength in transform')

        # scale factor from phase in radians to displacement in meters
        # constant is negative to make LOS = -1 * range change
        scale = -float(wavelength) / (4 * np.pi)

        out: dict[str, xr.Dataset] = {}
        for key, phase_ds in phase.items():
            disp_vars: dict[str, xr.DataArray] = {}
            for var_name, data in phase_ds.data_vars.items():
                disp = (data * scale).astype('float32')
                #name = 'los' if len(phase_ds.data_vars) == 1 else f'{var_name}_los'
                disp_vars[var_name] = disp
            out[key] = xr.Dataset(disp_vars, coords=phase_ds.coords, attrs=phase_ds.attrs)

        return Batch(out)

    def _displacement_component(self, phase: Batch, transform: Batch = None, func=None, suffix: str = '') -> Batch:
        """Internal helper to scale LOS displacement by an incidence-based function (e.g., cos/sin)."""
        import xarray as xr
        import numpy as np

        # Default to self.transform() when not provided, decimated to match phase resolution
        tfm_is_default = transform is None
        if transform is None:
            transform = self.transform()

        # Decimate default transform to match input phase resolution for efficiency
        if tfm_is_default and transform is not None:
            transform = Batch({k: transform[k].reindex(y=phase[k].y, x=phase[k].x, method='nearest')
                               for k in phase.keys() if k in transform})

        los_batch = self.displacement_los(phase, transform)
        incidence_batch = transform.incidence()

        out: dict[str, xr.Dataset] = {}

        for key, los_ds in los_batch.items():
            if key not in incidence_batch:
                raise KeyError(f'Missing incidence for key: {key}')

            inc_da = incidence_batch[key]['incidence']
            comp_vars: dict[str, xr.DataArray] = {}

            for var_name, data in los_ds.data_vars.items():
                # align incidence to data grid
                if 'y' in data.coords and 'x' in data.coords:
                    incidence = inc_da.interp(y=data.y, x=data.x, method='linear')
                else:
                    incidence = inc_da.reindex_like(data, method='nearest')

                comp = (data / func(incidence)).astype('float32')

                if len(los_ds.data_vars) == 1:
                    name = suffix
                elif var_name.endswith('_los'):
                    name = var_name[:-4] + f'_{suffix}'
                else:
                    name = f'{var_name}_{suffix}'

                comp_vars[name] = comp

            out[key] = xr.Dataset(comp_vars, coords=los_ds.coords, attrs=los_ds.attrs)

        return Batch(out)

    def displacement_vertical(self, phase: Batch, transform: Batch = None) -> Batch:
        """Compute vertical displacement (meters) from unwrapped phase and incidence."""
        import xarray as xr
        return self._displacement_component(phase, transform, func=xr.ufuncs.cos, suffix='vertical')

    def displacement_eastwest(self, phase: Batch, transform: Batch = None) -> Batch:
        """Compute east-west displacement (meters) from unwrapped phase and incidence."""
        import xarray as xr
        return self._displacement_component(phase, transform, func=xr.ufuncs.sin, suffix='eastwest')

    def align_elevation(self, debug: bool = False) -> "Stack":
        """Apply elevation correction for TOPS burst processing.

        Corrects for satellite height variation along track when processing
        per-burst (as opposed to GMTSAR's approach of stitching before
        interferometry).

        Parameters
        ----------
        debug : bool, optional
            Print debug information including elevation corrections and
            alignment quality metrics.

        Returns
        -------
        Stack
            New Stack with corrected elevation values.

        Notes
        -----
        The satellite height varies along the orbit (30-60m between adjacent
        bursts). This causes an apparent elevation offset between bursts
        when they are processed independently. This function computes and
        applies optimal elevation corrections to align all bursts.

        Uses least-squares optimization over all overlapping burst pairs
        (both within-subswath and cross-subswath) to find globally optimal
        corrections. The elevation correction for each pair is computed
        analytically from satellite height difference using radar geometry.

        For improved accuracy, uses a single reference range per orbital path
        (IW2's near_range or average if IW2 not available) instead of
        per-subswath near_range, reducing systematic biases across subswaths.
        """
        import numpy as np
        from scipy import sparse
        from scipy.sparse.linalg import lsqr
        from collections import defaultdict

        if not self:
            return type(self)({})

        ids = sorted(self.keys())
        n_bursts = len(ids)
        id_to_idx = {bid: i for i, bid in enumerate(ids)}

        if debug:
            print(f"align_elevation(): {n_bursts} bursts")

        # Compute reference near_range per orbital path
        # Using IW2 (middle subswath) provides best accuracy across all subswaths
        path_ref_range = {}
        path_all_ranges = defaultdict(list)

        for bid in ids:
            parts = bid.split('_')
            if len(parts) >= 3:
                path = parts[0]
                sw = parts[2]
                nr = float(self[bid]['near_range'].isel(date=0).values)
                path_all_ranges[path].append(nr)
                if sw == 'IW2':
                    path_ref_range[path] = nr

        # For paths without IW2, use average of available subswaths
        for path, ranges in path_all_ranges.items():
            if path not in path_ref_range:
                path_ref_range[path] = np.mean(ranges)

        if debug:
            print(f"Reference range per path:")
            for path in sorted(path_ref_range.keys()):
                print(f"  Path {path}: {path_ref_range[path]/1000:.1f} km")

        def compute_elevation_correction(ds_prev, ds_curr, ref_range):
            """Compute elevation correction from satellite height difference.

            Returns (dh_elevation, has_overlap) where dh_elevation is the
            apparent elevation offset (meters) to add to curr relative to prev,
            and has_overlap indicates if the bursts have valid azimuth overlap.
            """
            prf = float(ds_prev['PRF'].isel(date=0).values)
            lpb = int(ds_prev['linesPerBurst'].isel(date=0).values)
            clock_start_prev = float(ds_prev['clock_start'].isel(date=0).values)
            clock_start_curr = float(ds_curr['clock_start'].isel(date=0).values)

            ksr_prev = int(ds_prev['ksr'].isel(date=0).values)
            ker_prev = int(ds_prev['ker'].isel(date=0).values)
            ksr_curr = int(ds_curr['ksr'].isel(date=0).values)
            ker_curr = int(ds_curr['ker'].isel(date=0).values)

            # Lines from burst1 start to burst2 start
            nl = int(np.floor((clock_start_curr - clock_start_prev) * 86400.0 * prf + 0.5))

            # Overlap of valid regions
            valid_overlap_start = max(ksr_prev, nl + ksr_curr)
            valid_overlap_end = min(ker_prev, nl + ker_curr)

            # Check if there's valid overlap
            if valid_overlap_end <= valid_overlap_start:
                return 0.0, False

            # Stitch at middle of valid overlap
            azi_prev = round((valid_overlap_start + valid_overlap_end) / 2)
            azi_curr = azi_prev - nl

            # Satellite height at overlap midpoint for each burst
            H_prev_start = float(ds_prev['SC_height_start'].isel(date=0).values)
            H_prev_end = float(ds_prev['SC_height_end'].isel(date=0).values)
            H_curr_start = float(ds_curr['SC_height_start'].isel(date=0).values)
            H_curr_end = float(ds_curr['SC_height_end'].isel(date=0).values)

            H_prev = H_prev_start + (H_prev_end - H_prev_start) * azi_prev / lpb
            H_curr = H_curr_start + (H_curr_end - H_curr_start) * azi_curr / lpb
            dH_satellite = H_prev - H_curr

            # Compute apparent elevation difference from satellite height change
            # Using radar geometry: range, Earth radius, incidence angle
            # Use path-specific reference range for consistent accuracy across subswaths
            earth_radius = float(ds_prev['earth_radius'].isel(date=0).values)
            H_avg = (H_prev + H_curr) / 2
            c = earth_radius + H_avg
            ret = earth_radius
            cos_theta = (c**2 + ret**2 - ref_range**2) / (2 * c * ret)
            drho_dH = (c - ret * cos_theta) / ref_range
            drho_dh = (ret - c * cos_theta) / ref_range
            dh_elevation = -dH_satellite * drho_dH / drho_dh

            return dh_elevation, True

        # Find all overlapping burst pairs and compute analytical corrections
        # - Within-subswath (same IW): compute from satellite height variation along track
        # - Cross-subswath at same burst number: dh=0 (same satellite position, elevations match)
        overlap_pairs = []  # (id1, id2, dh_elevation)

        for i, id1 in enumerate(ids):
            ds1 = self[id1]
            # Extract path, burst number and subswath from burst ID
            parts1 = id1.split('_')
            path1 = parts1[0] if len(parts1) >= 3 else ''
            bn1 = parts1[1] if len(parts1) >= 3 else ''
            sw1 = parts1[2] if len(parts1) >= 3 else ''

            for j, id2 in enumerate(ids):
                if i >= j:
                    continue
                ds2 = self[id2]
                parts2 = id2.split('_')
                path2 = parts2[0] if len(parts2) >= 3 else ''
                bn2 = parts2[1] if len(parts2) >= 3 else ''
                sw2 = parts2[2] if len(parts2) >= 3 else ''

                if sw1 == sw2:
                    # Within-subswath: check azimuth overlap and compute analytical correction
                    ref_range = path_ref_range.get(path1, float(ds1['near_range'].isel(date=0).values))
                    dh, has_overlap = compute_elevation_correction(ds1, ds2, ref_range)
                    if has_overlap:
                        overlap_pairs.append((id1, id2, dh))
                elif bn1 == bn2:
                    # Cross-subswath at SAME burst number: same satellite position, dh=0
                    # These bursts have spatial overlap and same acquisition time
                    overlap_pairs.append((id1, id2, 0.0))

        # Count within-subswath vs cross-subswath
        within_sw = 0
        cross_sw = 0
        for id1, id2, _ in overlap_pairs:
            sw1 = id1.split('_')[2] if len(id1.split('_')) >= 3 else ''
            sw2 = id2.split('_')[2] if len(id2.split('_')) >= 3 else ''
            if sw1 == sw2:
                within_sw += 1
            else:
                cross_sw += 1

        if debug:
            print(f"Overlapping pairs: {len(overlap_pairs)} "
                  f"({within_sw} within-subswath, {cross_sw} cross-subswath)")

        if len(overlap_pairs) == 0:
            if debug:
                print("No overlapping pairs found, returning unchanged")
            return type(self)(dict(self))

        # Build least-squares system: A @ corrections = b
        # For each pair (id1, id2, dh): corr[id2] - corr[id1] = dh
        n_constraints = len(overlap_pairs)
        rows = []
        cols = []
        data = []
        b = []

        for constraint_idx, (id1, id2, dh) in enumerate(overlap_pairs):
            idx1 = id_to_idx[id1]
            idx2 = id_to_idx[id2]
            # corr[id2] - corr[id1] = dh
            rows.extend([constraint_idx, constraint_idx])
            cols.extend([idx1, idx2])
            data.extend([-1.0, 1.0])
            b.append(dh)

        A = sparse.csr_matrix((data, (rows, cols)), shape=(n_constraints, n_bursts))
        b = np.array(b)

        # Solve least-squares (with implicit reference: mean correction = 0)
        result = lsqr(A, b)
        corrections = result[0]

        # Center corrections (mean = 0) to avoid drift
        corrections = corrections - np.mean(corrections)

        if debug:
            # Compute residuals for quality assessment
            residuals = A @ corrections - b

            # Separate within-subswath and cross-subswath residuals
            within_sw_residuals = []
            cross_sw_residuals = []
            sw_residuals = defaultdict(list)

            for idx, (id1, id2, dh) in enumerate(overlap_pairs):
                sw1 = id1.split('_')[2] if len(id1.split('_')) >= 3 else ''
                sw2 = id2.split('_')[2] if len(id2.split('_')) >= 3 else ''
                if sw1 == sw2:
                    within_sw_residuals.append(residuals[idx])
                    sw_residuals[sw1].append(residuals[idx])
                else:
                    cross_sw_residuals.append(residuals[idx])

            within_arr = np.array(within_sw_residuals) if within_sw_residuals else np.array([0.0])
            cross_arr = np.array(cross_sw_residuals) if cross_sw_residuals else np.array([0.0])

            # Print quality summary
            print(f"\nAlignment quality:")
            print(f"  Overall RMS: {np.sqrt(np.mean(residuals**2)):.3f} m")
            print(f"  Within-subswath ({len(within_arr)} pairs): "
                  f"RMS={np.sqrt(np.mean(within_arr**2)):.3f} m, "
                  f"max={np.max(np.abs(within_arr)):.3f} m")
            print(f"  Cross-subswath ({len(cross_arr)} pairs): "
                  f"RMS={np.sqrt(np.mean(cross_arr**2)):.3f} m, "
                  f"max={np.max(np.abs(cross_arr)):.3f} m")

            # Per-subswath breakdown
            print(f"\n  Per-subswath RMS:")
            for sw in sorted(sw_residuals.keys()):
                sw_arr = np.array(sw_residuals[sw])
                print(f"    {sw} ({len(sw_arr)} pairs): "
                      f"RMS={np.sqrt(np.mean(sw_arr**2)):.3f} m, "
                      f"bias={np.mean(sw_arr):.3f} m")

            # Correction range
            print(f"\nCorrections range: {corrections.min():.2f} to {corrections.max():.2f} m")
            print(f"Mean correction between adjacent bursts: "
                  f"{np.mean(np.abs(np.array([dh for _, _, dh in overlap_pairs if dh != 0]))):.2f} m")

        # Apply elevation corrections
        output = {}
        for bid, ds in self.items():
            ele_corr = corrections[id_to_idx[bid]]

            # Skip if no correction needed
            if abs(ele_corr) < 1e-6:
                output[bid] = ds
                continue

            new_ds = ds.copy()

            # Apply elevation correction (preserve float32 dtype)
            if 'ele' in ds.data_vars:
                new_ds['ele'] = ds['ele'] + np.float32(ele_corr)

            output[bid] = new_ds

        return type(self)(output)

    def baseline(self, days: int | None = None, meters: float | None = None,
                 invert: bool = False) -> "Baseline":
        """Generate baseline pairs table from the Stack.

        Creates a Baseline DataFrame containing all valid interferometric pairs
        with their temporal and spatial baselines.

        Parameters
        ----------
        days : int, optional
            Maximum temporal separation in days. If None, no temporal limit.
        meters : float, optional
            Maximum perpendicular baseline difference in meters. If None, no limit.
        invert : bool, optional
            If True, invert reference and repeat dates. Default is False.

        Returns
        -------
        Baseline
            DataFrame subclass with columns: ref, rep, ref_baseline, rep_baseline,
            pair, baseline, duration. Has custom plot() and hist() methods.

        Examples
        --------
        >>> # Get all baseline pairs
        >>> bl = stack.baseline()
        >>> bl.plot()  # Plot baseline network
        >>> bl.hist()  # Plot duration histogram

        >>> # Filter by temporal separation
        >>> bl = stack.baseline(days=48)

        >>> # Filter by baseline
        >>> bl = stack.baseline(meters=100)
        """
        import numpy as np
        import pandas as pd
        from .Baseline import Baseline

        if days is None:
            days = int(1e6)

        # Get baseline table: date -> BPR
        # Extract BPR per date from first burst (all bursts have same dates)
        if not self:
            return Baseline()

        first_key = next(iter(self.keys()))
        first_ds = self[first_key]

        if 'date' not in first_ds.dims:
            raise ValueError("Stack must have 'date' dimension to compute baselines")

        # Normalize to date only (no time component)
        dates = pd.DatetimeIndex(first_ds.coords['date'].values).normalize()

        # Get BPR values - they are stored as a data variable along date dimension
        if 'BPR' in first_ds.data_vars:
            bpr_values = first_ds['BPR'].values
        elif 'BPR' in first_ds.coords:
            bpr_values = first_ds.coords['BPR'].values
        else:
            raise ValueError("Stack must have 'BPR' variable to compute baselines")

        # Build baseline table
        tbl = pd.DataFrame({'date': dates, 'BPR': bpr_values}).set_index('date')

        # Generate pairs
        data = []
        for line1 in tbl.itertuples():
            for line2 in tbl.itertuples():
                if not (line1.Index < line2.Index and (line2.Index - line1.Index).days <= days):
                    continue
                if meters is not None and not (abs(line1.BPR - line2.BPR) <= meters):
                    continue

                if not invert:
                    data.append({
                        'ref': line1.Index,
                        'rep': line2.Index,
                        'ref_baseline': np.round(line1.BPR, 2),
                        'rep_baseline': np.round(line2.BPR, 2)
                    })
                else:
                    data.append({
                        'ref': line2.Index,
                        'rep': line1.Index,
                        'ref_baseline': np.round(line2.BPR, 2),
                        'rep_baseline': np.round(line1.BPR, 2)
                    })

        if not data:
            return Baseline(burst_id=first_key, dates=dates)

        df = pd.DataFrame(data).sort_values(['ref', 'rep']).reset_index(drop=True)
        df = df.assign(
            pair=[f'{ref.date()} {rep.date()}' for ref, rep in zip(df['ref'], df['rep'])],
            baseline=df['rep_baseline'] - df['ref_baseline'],
            duration=(df['rep'] - df['ref']).dt.days
        )

        return Baseline(df, burst_id=first_key, dates=dates)
