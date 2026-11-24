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
from .BatchCore import BatchCore
import numpy as np
import xarray as xr
from .Batch import Batch
from .BatchComplex import BatchComplex
from .BatchUnit import BatchUnit
from .BatchWrap import BatchWrap
from . import utils_complex
from . import utils_baseline
from .utils_dask import graph_check


def _subtract_date_from_pair(first, second):
    """Subtract per-date atmospheric screens from per-pair data.

    Uses BatchComplex.pairs() to select ref/rep screens,
    then: result = data * conj(screen_ref) * screen_rep
    """
    import numpy as np

    key0 = list(first.keys())[0]
    ref_dates = first[key0].coords['ref'].values
    rep_dates = first[key0].coords['rep'].values
    pairs = np.column_stack([ref_dates, rep_dates])

    screen_ref, screen_rep = second.pairs(pairs)
    # Rechunk screens to match first's chunks — isel produces fragmented
    # dim0 chunks (63,63,...,31) while first has merged (1165,) chunks.
    # Without this, the multiply triggers an implicit rechunk of first
    # that re-reads the full upstream graph 19× per spatial tile.
    ref_var = next(v for v in first[key0].data_vars if first[key0][v].ndim >= 3)
    ref_chunks = first[key0][ref_var].data.chunks
    for batch in (screen_ref, screen_rep):
        for key in batch:
            ds = batch[key]
            rechunked = {}
            for v in ds.data_vars:
                da_xr = ds[v]
                if da_xr.ndim >= 3 and hasattr(da_xr.data, 'chunks') and da_xr.data.chunks != ref_chunks:
                    rechunked[v] = da_xr.chunk(dict(zip(da_xr.dims, ref_chunks)))
            if rechunked:
                batch[key] = ds.assign(rechunked)
    return utils_complex.screens(first, screen_ref, screen_rep)


class Batches(tuple):
    """
    A tuple-like container for multiple Batch objects that allows chained operations.

    Enables operations like:
        mintf, mcorr = stack.phasediff(...).downsample(20).compute()
        mintf, mcorr = stack.phasediff(...).downsample(20).snapshot('mintf_corr')
        mintf, mcorr = Batches.open('mintf_corr')

    Instead of:
        mintf, mcorr = stack.phasediff(...)
        mintf, mcorr = stack.compute(mintf.downsample(20), mcorr.downsample(20))
    """

    def __new__(cls, batches=()):
        return super().__new__(cls, batches)

    @staticmethod
    def _preserve_nonspatial(source, target):
        """Copy non-spatial variables (e.g. BPR) from source to target batch."""
        import dask.array as da
        for key in source:
            src_ds = source[key]
            tgt_ds = target[key]
            extra = {}
            for v in src_ds.data_vars:
                if v not in tgt_ds.data_vars:
                    var = src_ds[v]
                    if not isinstance(var.data, da.Array):
                        var = var.chunk()
                    extra[v] = var
            if extra:
                target[key] = tgt_ds.assign(extra)
        return target

    def phase(self) -> 'BatchComplex | BatchWrap | Batch | None':
        """Extract phase from Batches.

        Returns the first BatchComplex, or first BatchWrap,
        or first Batch found. Returns None if none found.
        """
        for b in self:
            if isinstance(b, BatchComplex):
                return b
        for b in self:
            if isinstance(b, BatchWrap):
                return b
        for b in self:
            if isinstance(b, Batch) and not isinstance(b, BatchUnit):
                return b
        return None

    def correlation(self) -> 'BatchUnit | None':
        """Extract correlation weights from Batches.

        Returns the first BatchUnit found, or None.
        """
        for b in self:
            if isinstance(b, BatchUnit):
                return b
        return None

    def snapshot(self, store: str | None = None, storage_options: dict[str, str] | None = None,
                 caption: str | None = None,
                 debug: bool = False, overwrite: bool = True, **kwargs):
        """Save or open a Batches snapshot.

        When called on a Batches with data, saves all batches to Zarr store.
        When called on an empty Batches(), opens an existing store.

        Parameters
        ----------
        store : str
            Path to the Zarr store.
        storage_options : dict, optional
            Storage options for cloud stores.
        caption : str, optional
            Progress bar caption.
        debug : bool
            Print debug information.
        overwrite : bool
            True (default) replaces an existing store; False keeps a complete one and returns it opened.

        Returns
        -------
        tuple
            Tuple of Batch objects for unpacking.

        Examples
        --------
        >>> # Save
        >>> mintf, mcorr = stack.phasediff(...).downsample(20).snapshot('mintf_corr')
        >>> # Open
        >>> mintf, mcorr = Batches().snapshot('mintf_corr')
        """
        from . import utils_io

        if len(self) == 0:
            result = utils_io.snapshot(store=store, storage_options=storage_options,
                                       caption=caption or 'Opening...',
                                       debug=debug)
        else:
            result = utils_io.snapshot(*self, store=store, storage_options=storage_options,
                                       caption=caption or 'Snapshotting...',
                                       debug=debug, overwrite=overwrite, wrapper=Batches)

        if isinstance(result, Batches):
            return result
        return Batches((result,))  # fallback for stores without __wrapper__

    def archive(self, store: str, caption: str | None = None, compression: int = 6,
                debug: bool = False):
        """Save or open a Batches archive as a single ZIP file.

        Wrapper around snapshot() that uses ZipStore for single-file storage.
        Useful for downloading data from Google Colab or similar environments.

        Parameters
        ----------
        store : str
            Path to the ZIP file. Must end with '.zip'.
        caption : str, optional
            Progress bar caption.
        compression : int
            ZIP compression level 0-9 (0=no compression, 9=max). Default 6.
            Higher values produce smaller files but take longer.
        debug : bool
            Print debug information.

        Returns
        -------
        Batches
            In save mode the input batches (the archive is written, nothing is read back);
            in open mode the batches read from the archive.

        Examples
        --------
        >>> # Save to zip
        >>> mintf, mcorr = stack.phasediff(...).downsample(20).archive('mintf_corr.zip')
        >>> # Save with max compression (for GitHub 100MB limit)
        >>> mintf, mcorr = stack.phasediff(...).archive('mintf_corr.zip', compression=9)
        >>> # Save to cloud storage (GCS, S3, etc.)
        >>> mintf, mcorr = stack.phasediff(...).archive('gs://bucket/mintf_corr.zip')
        >>> # Open from zip
        >>> mintf, mcorr = Batches().archive('mintf_corr.zip')
        """
        import zarr
        import zipfile
        import tempfile
        import os
        import fsspec
        from . import utils_io

        if not store.endswith('.zip'):
            raise ValueError(f"Archive store must have '.zip' extension, got: {store}")

        # Check if cloud storage path
        is_cloud = '://' in store

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
            result = self.snapshot(store=zip_store, caption=caption or 'Opening archive...', debug=debug)
            zip_store.close()
            return result
        else:
            # Save mode - write to temp directory, then zip
            # This avoids ZipStore's duplicate entry problem
            temp_dir = tempfile.mkdtemp()
            try:
                # an archive is an export: zarr's default codec ('auto'), not the snapshot codec;
                # the snapshot's result is lazy over temp_dir, deleted below, so it is not returned
                utils_io._snapshot(*self, store=temp_dir, caption=caption or 'Archiving...',
                                   debug=debug, wrapper=Batches, compressors='auto')
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

    def downsample(self, *args, **kwargs):
        """Apply downsample to all batches."""
        return Batches([b.downsample(*args, **kwargs) for b in self])

    def chunk(self, *args, **kwargs):
        """Apply chunk to all batches."""
        return Batches([b.chunk(*args, **kwargs) for b in self])

    def chunk2d(self, *args, **kwargs):
        """Apply chunk2d to all batches."""
        return Batches([b.chunk2d(*args, **kwargs) for b in self])

    def chunk1d(self, *args, **kwargs):
        """Apply chunk1d to all batches."""
        return Batches([b.chunk1d(*args, **kwargs) for b in self])

    def where(self, cond, other=np.nan, **kwargs):
        """Apply where mask to all batches."""
        return Batches([b.where(cond, other, **kwargs) for b in self])

    def crop(self, *args, **kwargs):
        """Apply crop to all batches."""
        return Batches([b.crop(*args, **kwargs) for b in self])

    def sel(self, *args, **kwargs):
        """Apply sel to all batches."""
        return Batches([b.sel(*args, **kwargs) for b in self])

    def isel(self, *args, **kwargs):
        """Apply isel to all batches."""
        return Batches([b.isel(*args, **kwargs) for b in self])

    def filter(self, days=None, meters=None, date=None, pair=None, count=None,
               min_connections=2, cleanup=True):
        """Filter pairs in the baseline network.

        Selects pairs matching the given temporal/spatial criteria. By default,
        removes degraded dates (hanging or single-side connected).

        Parameters
        ----------
        days : int, optional
            Maximum temporal separation in days. If None, no temporal limit.
        meters : float, optional
            Maximum perpendicular baseline in meters. If None, no limit.
        date : str or list, optional
            Date(s) to exclude. Accepts a single date string or a list,
            any format parseable by ``pd.to_datetime``.
        pair : str or list, optional
            Pair(s) to exclude. Each pair is a string ``'YYYY-MM-DD YYYY-MM-DD'``.
            Accepts a single pair string or a list.
        count : int, optional
            Remove dates with fewer than this many connections.
        min_connections : int, optional
            Minimum pairs per date for cleanup. Default is 2.
        cleanup : bool, optional
            If True (default), iteratively remove hanging dates and dates
            connected only to predecessors or only to successors.
            Set to False to keep the raw network for testing.

        Returns
        -------
        Batches
            Filtered Batches with valid pairs only.

        Examples
        --------
        >>> stack.filter(days=100, meters=80).unwrap3d()
        >>> stack.filter(date='2024-12-30').unwrap3d()
        >>> intfcorr.filter(date=['2024-12-30', '2024-06-21'])
        >>> intfcorr.filter(pair='2024-06-21 2024-12-30')
        >>> intfcorr.filter(count=3)  # remove dates with < 3 connections
        >>> intfcorr.filter(days=100, meters=80, cleanup=False)  # raw network
        """
        if days is None and meters is None and date is None and pair is None and count is None:
            return self

        # Get pair coordinates from the first batch element
        first_batch = self[0]
        first_key = next(iter(first_batch.keys()))
        ds = first_batch[first_key]
        valid = utils_baseline.filter_pairs(ds.coords['ref'].values, ds.coords['rep'].values,
                                            ds.coords['BPR'].values, days=days, meters=meters,
                                            exclude_dates=date, exclude_pairs=pair, count=count,
                                            min_connections=min_connections, cleanup=cleanup)
        if not valid.any():
            raise ValueError("No valid pairs remain after filtering. "
                             "Try increasing 'days' or 'meters'.")
        return self.isel(pair=np.flatnonzero(valid))

    def coherent(self, threshold=0.5):
        """Mask low-coherence pixels using mean correlation.

        Computes mean correlation across pairs from the BatchUnit item
        and sets pixels with mean correlation below threshold to NaN
        in all batch items.

        Parameters
        ----------
        threshold : float
            Minimum mean correlation to keep. Default 0.5.

        Returns
        -------
        Batches
            Same structure with NaN where mean correlation < threshold.
        """
        corr = next((b for b in self if isinstance(b, BatchUnit)), None)
        if corr is None:
            raise ValueError('coherent() requires a BatchUnit (correlation) in Batches')
        results = []
        for b in self:
            # the shared weight check (N81): a burst corr lacks, or a DataArray corr without
            # (y, x) (BatchUnit(corr['BPR'])), raises. Not by name: one correlation grid
            # masks every grid of every batch
            BatchCore._weight(corr, b, 'corr', by_name=False)
            out = {}
            for key in b:
                corr_ds = corr[key]
                if isinstance(corr_ds, xr.DataArray):
                    # a DataArray BatchUnit (N81) is its own one grid
                    corr_da = corr_ds
                else:
                    corr_var = next(v for v in corr_ds.data_vars if 'y' in corr_ds[v].dims)
                    corr_da = corr_ds[corr_var]
                mask = corr_da.mean('pair') >= threshold if 'pair' in corr_da.dims else corr_da >= threshold
                ds = b[key]
                # THE GRIDS ONLY (N81): xarray's where() on the whole Dataset turned
                # the string metadata into objects that failed on compute
                out[key] = ds.where(mask) if isinstance(ds, xr.DataArray) else ds.assign(
                    {v: ds[v].where(mask) for v in ds.data_vars if BatchCore._is_grid(ds[v])})
            results.append(type(b)(out))
        graph_check('coherent', results, self)
        return Batches(results)

    def angle(self):
        """Apply angle() to BatchComplex batches, return others unchanged.

        Returns
        -------
        Batches
            Batches with BatchComplex converted to BatchWrap (phase angles),
            other batch types unchanged.

        Examples
        --------
        >>> phase, corr = stack.phasediff2(pairs).angle()
        >>> # phase is now BatchWrap with angles, corr is unchanged BatchUnit
        """
        results = []
        for b in self:
            if isinstance(b, BatchComplex):
                results.append(b.angle())
            else:
                results.append(b)
        return Batches(results)

    def goldstein(self, window: 'float | tuple' = 200, threshold: float = 0.5):
        """Apply Goldstein filter to phase using correlation as weight.

        Expects Batches with [BatchComplex (phase), BatchUnit (correlation)].

        Parameters
        ----------
        window : float or tuple
            Goldstein filter patch size in METRES on the ground, default 200.
            Converted per burst and rounded down to an even pixel count.
        threshold : float
            Minimum fraction of valid (non-NaN) pixels required to process a patch.
            Default 0.5 means at least 50% of pixels must be valid.

        Returns
        -------
        Batches
            Batches with Goldstein-filtered phase and unchanged correlation.

        Examples
        --------
        >>> intf, corr = stack.pairs(pairs).interferogram(30).goldstein(200)
        """
        if len(self) < 2:
            raise ValueError("goldstein() requires Batches with at least 2 elements: [interferogram, correlation]")

        phase, corr = self[0], self[1]

        if not isinstance(phase, BatchComplex):
            raise TypeError(f"First element must be BatchComplex, got {type(phase).__name__}")
        # the correlation is the weight: the shared check and messages (N81)
        BatchCore._weight(corr, phase, 'corr', required=True)

        filtered_phase = phase.goldstein(corr, window, threshold=threshold)
        return Batches([filtered_phase, corr] + list(self[2:]))

    def interferogram(self,
                  wavelength: float | None = None,
                  weight: 'BatchUnit | None' = None,
                  phase: 'BatchComplex | None' = None,
                  gaussian_threshold: float = 0.5) -> 'Batches':
        """
        Compute phase difference from paired SLC data.

        Expects Batches from pairs() with [ref, rep] BatchComplex objects.

        Parameters
        ----------
        wavelength : float or None
            Filter wavelength in metres, passed to gaussian(). FIRST, as
            gaussian() and multilook() take it -- it is the argument that
            decides what comes back, and it used to sit third behind two
            optional objects, so `interferogram(400)` bound 400 to the weight.

            None IS A MODE, not a missing value: the interferogram is returned
            single-look and there is no correlation to report with it, so the
            result holds ONE element instead of two. Unpacking
            `phase, corr = ...` then fails, which is the intended signal that
            a correlation was never computed.
        weight : BatchUnit or None
            Per-burst weights for Gaussian filtering and masking.
        phase : BatchComplex or None
            Optional phase to subtract (e.g., topographic phase).
        gaussian_threshold : float
            Threshold for Gaussian filter (default 0.5).

        Returns
        -------
        Batches
            Batches with [interferogram, correlation]: the interferogram COMPLEX
            (BatchComplex) until angle() is called -- unwrap2d() and
            unwrap2d_irls() call it themselves -- and the correlation a BatchUnit.

        Examples
        --------
        >>> intf, corr = stack.pairs(baseline.tolist()).interferogram(30)
        >>> # single-look, ONE element back -- no correlation without a filter
        >>> intf, = stack.pairs(baseline.tolist()).interferogram()
        """
        if len(self) != 2:
            raise ValueError("interferogram() requires Batches with exactly 2 elements: [ref, rep]")

        ref, rep = self[0], self[1]

        if not isinstance(ref, BatchComplex) or not isinstance(rep, BatchComplex):
            raise TypeError("Both elements must be BatchComplex")

        # A WEIGHT OR A PHASE WHERE THE WAVELENGTH GOES is the old argument
        # order; caught here so it says so rather than failing later on a
        # comparison against a Batch.
        if wavelength is not None and not isinstance(wavelength, (int, float)):
            raise TypeError(
                f'interferogram() takes the wavelength first now, as gaussian() '
                f'does, and got a {type(wavelength).__name__}. Pass the others '
                f'by name: interferogram(wavelength, weight=..., phase=...)')

        # a BatchUnit (N81); a DataArray one weights every grid
        weight = BatchCore._weight(weight, ref)

        phase_in = phase   # the phase as given: a dask input of the call, for the graph rule
        if phase is not None and not isinstance(phase, BatchComplex):
            phase = phase.iexp(-1)
        intf = utils_complex.interferogram(ref, rep, phase)

        if wavelength is not None:
            intf_look = intf.gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold)
            intensity_ref = ref.power().gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold)
            intensity_rep = rep.power().gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold)
            del ref, rep
            corr_look = utils_complex.coherence(intf_look, intensity_ref, intensity_rep)
            del intensity_ref, intensity_rep
        else:
            intf_look = intf
            corr_look = None
            del ref, rep
        del intf

        if weight is not None:
            intf_look = utils_complex.masked(intf_look, weight)
            corr_look = utils_complex.masked(corr_look, weight)

        # the graph rule: every dask input of the call (the weight and the phase, when given) is not added
        sources = [self] + [x for x in (weight, phase_in) if x is not None]
        if corr_look is None:
            graph_check('interferogram', [intf_look], sources)
            return Batches([intf_look])
        graph_check('interferogram', [intf_look, corr_look], sources)
        return Batches([intf_look, corr_look])

    # compute(): the memory one persist of every grid holds, in bytes per full-resolution pixel of every grid it computes:
    # 32 = 8 * 4, the 4 data copies (input, output, temporaries) of one 8-byte complex value (the user's rule, 2026-11-24).
    # The measured peaks (dask's managed bytes at their peak per pixel of _compute_estimate; the ALOS-2 route
    # pairs().interferogram(400).goldstein(800).downsample(100)) stay below it: 5.0 to 24.1 B over (4, 16) and (32, 32) with
    # 4 to 143 pairs, and between runs the same batch varies (14.2 to 19.5 B at (32, 32) with 143 pairs). The share is the
    # part of the available memory that persist may take.
    _COMPUTE_BYTES_PER_PIXEL = 32
    _COMPUTE_MEMORY_SHARE = 0.5

    def _compute_blocks(self, lazy):
        """(client, dim, {burst: chunks of the stacked dimension}) of the lazy batches, or None for one persist.

        A GRID is one 2-D grid of a burst: one index of the stacked (leading) dimension of its 3-D variables, a pair or a
        date. None (one persist): no Client; no lazy 3-D variable; the stacked dimension not the same in every batch and
        burst, or chunked differently between the variables of one burst; or one block along it (the 1-D chunking, a
        pass over the spatial blocks, as snapshot() does it).
        """
        try:
            from dask.distributed import get_client
            client = get_client()
        except (ValueError, ImportError):
            return None
        batches = [batch for batch, l in zip(self, lazy) if l]
        dim = None
        for batch in batches:
            for value in batch.values():
                for arr in BatchCore._vars_of(value).values():
                    if arr.ndim >= 3 and hasattr(arr.data, 'dask'):
                        if dim is None:
                            dim = arr.dims[0]
                        elif arr.dims[0] != dim:
                            return None
        if dim is None:
            return None
        blocks = {}
        for batch in batches:
            for key, value in batch.items():
                for arr in BatchCore._vars_of(value).values():
                    if dim in arr.dims and hasattr(arr.data, 'dask'):
                        chunks = arr.data.chunks[arr.dims.index(dim)]
                        if blocks.setdefault(key, chunks) != chunks:
                            return None
        if all(len(chunks) == 1 for chunks in blocks.values()):
            return None
        return client, dim, blocks

    @staticmethod
    def _compute_estimate(batches, dim, blocks):
        """The memory one persist of the lazy batches would hold, in bytes, from dask's own sizes, before computing.

        One persist of every grid lets dask's order start nearly every grid before it finishes any, and each started grid
        holds about one full-resolution image: the memory grows linearly with the grids. A burst's full-resolution pixels
        per grid are the largest 2-D extent of the arrays its graph reads and makes (the shape of every 3-D layer without
        its leading dimension, and the variable's own); the estimate is _COMPUTE_BYTES_PER_PIXEL times those pixels times
        the burst's grids, summed over the bursts.
        """
        import math
        pixels = 0
        for key, chunks in blocks.items():
            grids = sum(chunks)
            extent = 0
            for batch in batches:
                value = batch.get(key)
                if value is None:
                    continue
                for arr in BatchCore._vars_of(value).values():
                    if dim not in arr.dims or not hasattr(arr.data, 'dask'):
                        continue
                    axis = arr.dims.index(dim)
                    extent = max(extent, math.prod(n for i, n in enumerate(arr.shape) if i != axis))
                    for layer in arr.data.__dask_graph__().layers.values():
                        shape = (layer.collection_annotations or {}).get('shape')
                        if shape and len(shape) == 3 and all(isinstance(n, (int, np.integer)) for n in shape):
                            extent = max(extent, math.prod(shape[1:]))
            pixels += grids * extent
        return Batches._COMPUTE_BYTES_PER_PIXEL * pixels

    @staticmethod
    def _compute_cgroup_free(root='/sys/fs/cgroup'):
        """The free memory of the container's cgroup in bytes, or None when no cgroup memory limit applies.

        cgroup v2: memory.max minus memory.current; v1: memory/memory.limit_in_bytes minus memory/memory.usage_in_bytes
        (the limit a host reports as 'max' or as a number above the host's memory is no limit). root is the cgroup folder
        (a temporary one in the test).
        """
        import os
        import psutil
        total = psutil.virtual_memory().total
        for limit_file, usage_file in (('memory.max', 'memory.current'),
                                       ('memory/memory.limit_in_bytes', 'memory/memory.usage_in_bytes')):
            limit_path, usage_path = os.path.join(root, limit_file), os.path.join(root, usage_file)
            if not (os.path.isfile(limit_path) and os.path.isfile(usage_path)):
                continue
            with open(limit_path) as f:
                limit = f.read().strip()
            with open(usage_path) as f:
                usage = f.read().strip()
            if not limit.isdigit() or not usage.isdigit() or int(limit) >= total:
                continue
            return max(int(limit) - int(usage), 0)
        return None

    @staticmethod
    def _compute_limit(cgroup_root='/sys/fs/cgroup'):
        """The memory one persist may hold, in bytes: _COMPUTE_MEMORY_SHARE of the available memory now, the smaller of the
        host's (psutil reports the host's memory inside a container) and the cgroup's free memory when a cgroup limit
        exists (_compute_cgroup_free). On macOS dask's memory management is off, so the workers' memory_limit bounds
        nothing; the available memory is the limit. It is the memory of the machine this process runs on, the client's:
        the workers of a remote cluster are not measured."""
        import psutil
        available = psutil.virtual_memory().available
        cgroup = Batches._compute_cgroup_free(cgroup_root)
        if cgroup is not None:
            available = min(available, cgroup)
        return Batches._COMPUTE_MEMORY_SHARE * available

    @staticmethod
    def _compute_units(blocks):
        """The units of a windowed compute(): block j of the stacked dimension of every burst together (the same pair
        index of every batch and burst), in order. [{burst: slice}, ...]"""
        units = []
        for j in range(max(len(chunks) for chunks in blocks.values())):
            unit = {}
            for key, chunks in blocks.items():
                if j < len(chunks):
                    start = sum(chunks[:j])
                    unit[key] = slice(start, start + chunks[j])
            units.append(unit)
        return units

    @staticmethod
    def _unit_part(value, dim, part, first):
        """The part of one burst a unit computes: its grids `part` of the stacked dimension, and the lazy variables
        without that dimension in the burst's first unit only."""
        if dim not in value.dims:
            return value if first else None
        if isinstance(value, xr.DataArray):
            return value.isel({dim: part})
        if not first:
            other = [v for v in value.data_vars if dim not in value[v].dims and hasattr(value[v].data, 'dask')]
            value = value.drop_vars(other)
        return value.isel({dim: part})

    def _unit_parts(self, batches, dim, unit, units, seen):
        """The lazy parts of every batch one unit computes; `seen` (the bursts already started) is updated."""
        parts = []
        for batch in batches:
            part = {}
            for key, value in batch.items():
                if key in unit:
                    p = self._unit_part(value, dim, unit[key], key not in seen)
                elif key not in seen and not any(key in u for u in units):
                    p = value          # a burst with no stacked grids: whole, with the first unit
                else:
                    p = None
                if p is not None:
                    part[key] = p
            parts.append(part)
        seen.update(k for part in parts for k in part)
        return parts

    @staticmethod
    def _join_pieces(pieces, dim):
        """Join the held pieces of every burst along the stacked dimension, in order."""
        joined = []
        for plist in pieces:
            out = {}
            for key, ps in plist.items():
                if len(ps) == 1:
                    out[key] = ps[0]
                elif isinstance(ps[0], xr.DataArray):
                    out[key] = xr.concat(ps, dim=dim, coords='minimal', compat='override', join='override')
                else:
                    other = [v for v in ps[0].data_vars if dim not in ps[0][v].dims]
                    stacked = xr.concat([p.drop_vars(other, errors='ignore') for p in ps], dim=dim, data_vars='minimal',
                                        coords='minimal', compat='override', join='override')
                    out[key] = stacked.assign({v: ps[0][v] for v in other})[list(ps[0].data_vars)]
            joined.append(out)
        return joined

    @staticmethod
    def _cull_factory(batches):
        """cull(value): one burst's part on a graph of only the tasks it needs. The graph of everything is built and
        optimised once here, as the scheduler would optimise one persist of it (blockwise and linear fusion); a persist of
        one unit then ships only its own tasks, not the whole graph of every unit (measured: 22 MB pickled per persist at
        Puyehue (4, 16) 143 pairs, about 4 s a unit), and a task two units share keeps one key."""
        import dask.array as da
        from dask._task_spec import convert_legacy_graph
        from dask.base import collections_to_expr
        from dask.core import flatten
        from dask.highlevelgraph import HighLevelGraph

        base_layers, arrays = set(), []
        for batch in batches:
            for value in batch.values():
                for arr in list(BatchCore._vars_of(value).values()) + list(value.coords.values()):
                    if isinstance(arr.data, da.Array):
                        arrays.append(arr.data)
                        base_layers.update(arr.data.__dask_graph__().layers)
        # the output keys stay, so a unit's parts find their inputs in it
        base = convert_legacy_graph(dict(collections_to_expr(arrays, True).optimize().__dask_graph__()))
        del arrays

        class _Keys:
            """the keys of the base graph and of one part's own top layers, for convert_legacy_graph"""
            def __init__(self, top):
                self.top = top

            def __contains__(self, key):
                return key in self.top or key in base

        class _AllKeys:
            """the keys of the base graph, of one part's own top layers and of its whole unoptimised graph"""
            def __init__(self, top, hg):
                self.top, self.hg = top, hg

            def __contains__(self, key):
                if key in self.top or key in base:
                    return True
                try:
                    self.hg[key]
                    return True
                except (KeyError, TypeError):
                    return False

        def cull_array(x):
            hg = x.__dask_graph__()
            top = {}
            for name, layer in hg.layers.items():
                if name not in base_layers:
                    top.update(layer)
            top = convert_legacy_graph(top, all_keys=_Keys(top))
            graph, stack = {}, list(flatten(x.__dask_keys__()))
            while stack:
                k = stack.pop()
                if k in graph:
                    continue
                task = top.get(k)
                if task is None:
                    task = base.get(k)
                if task is None:
                    # an intermediate task the optimised graph fused away (the part needs it directly): its own
                    # unoptimised task, whose inputs are then found the same way
                    task = convert_legacy_graph({k: hg[k]}, all_keys=_AllKeys(top, hg)).get(k)
                    if task is None:
                        # data already on the cluster ({key: its own Future}, a persisted input): the scheduler
                        # holds it, the part depends on it as it is
                        graph[k] = None
                        continue
                graph[k] = task
                stack.extend(d for d in task.dependencies if d not in graph)
            graph = {k: t for k, t in graph.items() if t is not None}
            return da.Array(HighLevelGraph.from_collections(x.name, graph, dependencies=()), x.name, x.chunks,
                            meta=x._meta)

        def cull(value):
            def one(arr):
                return arr.copy(data=cull_array(arr.data)) if isinstance(arr.data, da.Array) else arr
            if isinstance(value, xr.DataArray):
                out = one(value)
            else:
                out = value.assign({v: one(value[v]) for v in value.data_vars})
            lazy_coords = {c: one(out.coords[c]) for c in out.coords if isinstance(out.coords[c].data, da.Array)}
            return out.assign_coords(lazy_coords) if lazy_coords else out
        return cull

    def _compute_window(self, lazy, dim, units, n):
        """The sliding window of compute(): every unit (one index of the stacked dimension of every batch and burst) is
        persisted on its own, n units stay in flight, and the units that finish are replaced at once by as many new
        ones (no barrier: a slow grid holds its own slot only). A refused graph shows as cancelled futures and raises
        then. The lazy coordinates of every unit are fetched in one compute at the end, every unit is held (as
        compute() holds its results), and the pieces are joined along the stacked dimension. Returns the persisted
        batches as dicts, as one dask.persist of the whole would."""
        import itertools
        import dask
        from dask.distributed import as_completed
        from distributed.client import futures_of
        from insardev_toolkit.progressbar import tqdm
        from .utils_dask import wait_known

        batches = [dict(batch) for batch, l in zip(self, lazy) if l]
        nb = len(batches)
        cull = self._cull_factory(batches)
        done = [None] * len(units)
        seen = set()
        todo = iter(range(len(units)))
        ac = as_completed(with_results=False)
        owner, remaining = {}, {}
        pbar = tqdm(desc='Computing bursts'.ljust(25), total=len(units))

        def submit(k):
            """persist the next k units in one call; returns the number of them with nothing to compute"""
            us = list(itertools.islice(todo, k))
            if not us:
                return 0
            flat = []
            for u in us:
                parts = self._unit_parts(batches, dim, units[u], units, seen)
                flat.extend({key: cull(value) for key, value in part.items()} for part in parts)
            # the graph is optimised already (_cull_factory): the scheduler takes it as it is, so a task two units
            # share keeps one key and runs once while both are in flight
            results = list(dask.persist(*flat, optimize_graph=False))
            empty = 0
            for i, u in enumerate(us):
                done[u] = results[i * nb:(i + 1) * nb]
                futures = futures_of(done[u])
                if not futures:
                    empty += 1
                    pbar.update(1)
                    continue
                remaining[u] = len(futures)
                for f in futures:
                    owner.setdefault(f.key, []).append(u)
                    ac.add(f)
            return empty

        free = n
        while free:
            free = submit(free)
        for batch in ac.batches():
            finished = 0
            for f in batch:
                u = owner[f.key].pop(0)
                if f.status == 'error':
                    f.result()
                if f.status == 'cancelled':
                    wait_known(done[u])         # raises the refusal (input data no longer on the cluster)
                remaining[u] -= 1
                if not remaining[u]:
                    del remaining[u]
                    finished += 1
            if finished:
                pbar.update(finished)
                free = finished
                while free:
                    free = submit(free)
        pbar.close()

        # every unit's lazy coordinates in one compute, then every unit held (as compute() holds its results)
        lazy_coords = [(u, bi, key, name, coord) for u, results in enumerate(done) for bi, result in enumerate(results)
                       for key, value in result.items() for name, coord in value.coords.items()
                       if hasattr(coord.data, 'compute')]
        values = dask.compute(*[coord.data for *_, coord in lazy_coords]) if lazy_coords else ()
        done = [[dict(result) for result in results] for results in done]
        for (u, bi, key, name, coord), val in zip(lazy_coords, values):
            done[u][bi][key] = done[u][bi][key].assign_coords({name: (coord.dims, val)})
        pieces = [{key: [] for key in batch} for batch in batches]
        for results in done:
            for bi, result in enumerate(results):
                for key, value in result.items():
                    pieces[bi][key].append(BatchCore._persisted(value, {}))
        return self._join_pieces(pieces, dim)

    def compute(self):
        """Compute all batches; the results stay in the workers' memory.

        All bursts of all batches go in a single scheduler submission (dask.persist), as always, when there is nothing to
        window: no distributed Client; no lazy 3-D variable; the stacked (leading) dimension not the same in every batch and
        burst; chunks along it that differ between the variables of one burst; or one block along it per burst (the 1-D
        chunking, a pass over the spatial blocks).

        With a Client and lazy 3-D variables chunked along their stacked dimension, compute() first estimates, from dask's
        own sizes and before computing, the memory that single submission would hold: one submission of every grid (a pair
        or a date of a burst) lets dask's order start nearly every grid before it finishes any, and each started grid
        holds about one full-resolution image, so the memory grows with the grids. The estimate is 32 bytes per
        full-resolution pixel of every grid (8 * 4: the 4 data copies, input, output and temporaries, of one 8-byte complex
        value), see _compute_estimate. When it fits in a share of the available memory (_compute_limit), the single
        submission runs: it is the fastest. Otherwise the units are computed in a sliding window (_compute_window): a unit
        is one chunk along the stacked dimension of every burst (one grid when the pairs are chunked by 1, the usual case,
        else the grids of that chunk), as many units in flight as the Client has workers, the next unit submitted the moment
        one finishes. The memory is then bounded by the units in flight, not by all of them, and a slow unit does not hold
        up the others.

        The limit is the memory of the machine this process runs on, the Client's: the smaller of the available memory and,
        in a container, the cgroup's free memory. The workers of a remote cluster are not measured.

        Both ways preserve the computation the batches share (e.g., the phase
        and the correlation of one pair run once: the same grid of every batch
        goes in the same submission) and give the same values, chunks and
        coordinates. For results that do not fit in memory, use snapshot().

        A batch of DataArrays is computed as the DataArrays it holds, and a
        batch with nothing lazy comes back as it is (BatchCore.compute, N81).

        Returns
        -------
        Batches
            Computed batches with data in memory.
        """
        import dask
        from .utils_dask import progress_persisted

        lazy = [BatchCore._is_lazy_any(batch) for batch in self]
        if not any(lazy):
            return self

        # Save input chunk structure per batch per burst
        all_input_chunks = [{key: BatchCore._input_chunks(v) for key, v in batch.items()}
                            for batch, l in zip(self, lazy) if l]

        found = self._compute_blocks(lazy)
        if found is not None:
            client, dim, blocks = found
            estimate = self._compute_estimate([batch for batch, l in zip(self, lazy) if l], dim, blocks)
            if estimate <= self._compute_limit():
                found = None
        if found is not None:
            all_results = self._compute_window(lazy, dim, self._compute_units(blocks), max(len(client.nthreads()), 1))
        else:
            # Persist all lazy batches at once — single scheduler submission
            # progress_persisted extracts futures and blocks until completion
            all_results = list(dask.persist(*[dict(batch) for batch, l in zip(self, lazy) if l]))
            progress_persisted(all_results, desc='Computing bursts'.ljust(25))

        # Finalize: materialize coordinates, hold the futures, rechunk to match input
        computed = iter(zip(all_results, all_input_chunks))
        computed_batches = []
        for batch, l in zip(self, lazy):
            if not l:
                computed_batches.append(batch)
                continue
            result, input_chunks = next(computed)
            computed_batches.append(type(batch)({key: BatchCore._persisted(v, input_chunks[key])
                                                 for key, v in result.items()}))
        return Batches(computed_batches)

    def unwrap2d(self, conncomp=False, conncomp_size=1000, conncomp_gap=None,
                 conncomp_linksize=5, conncomp_linkcount=30, union=False,
                 debug=False, **kwargs):
        """
        Unwrap phase: L¹ IRLS with a multigrid-preconditioned solver, on the CPU (numba).

        Expects Batches with [BatchWrap or BatchComplex (phase), BatchUnit (weight, optional)].
        If the first element is BatchComplex, .angle() is called automatically.

        Parameters
        ----------
        conncomp : bool
            If False (default), link disconnected components using ILP.
            If True, keep components separate and return conncomp labels.
        conncomp_size : int
            Minimum pixels for a connected component. Default 1000.
        conncomp_gap : int or None
            Maximum pixel distance between connectable components.
        conncomp_linksize : int
            Pixels on each side for phase offset estimation. Default 5.
        conncomp_linkcount : int
            Max nearest neighbor components to consider. Default 30.
        union : bool
            False (default) solves each burst on its own, which is the form
            that scales. True unions the bursts and solves once over the
            result, so the answer is consistent across burst edges -- viable
            while the merged scene fits. A Batch either way: the merge is
            internal, and each burst comes back holding only its own pixels.
        debug : bool
            Print diagnostic information.
        **kwargs
            Passed to unwrap2d_irls(): accuracy (default 2e-3; 5e-4 near the optimum,
            about 1.9 times longer), max_iter (the safety cap, default 100), epsilon.
            The former tol, cg_max_iter, cg_tol, device, semaphore print one WARNING.

        Returns
        -------
        Batch or tuple
            If conncomp=False: Batch of unwrapped phase.
            If conncomp=True: tuple of (Batch unwrapped, Batch conncomp).

        Examples
        --------
        >>> intf, corr = stack.pairs(baseline.tolist()).interferogram(100).angle()
        >>> unwrapped = intf.unwrap2d()  # Without weights
        >>> unwrapped = Batches([intf, corr]).unwrap2d()  # With weights
        """
        if len(self) < 1:
            raise ValueError("unwrap2d() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        # the second element IS the weight: a BatchUnit or an error, never dropped (N81)
        weight = BatchCore._weight(self[1] if len(self) >= 2 else None)

        # Auto-convert complex phase to wrapped phase
        if isinstance(phase, BatchComplex):
            phase = phase.angle()

        if not isinstance(phase, BatchWrap):
            raise TypeError(f"First element must be BatchWrap or BatchComplex, got {type(phase).__name__}")

        # Delegate to BatchWrap.unwrap2d
        return phase.unwrap2d(weight=weight, conncomp=conncomp, conncomp_size=conncomp_size,
                              conncomp_gap=conncomp_gap, conncomp_linksize=conncomp_linksize,
                              conncomp_linkcount=conncomp_linkcount, union=union,
                              debug=debug, **kwargs)

    def subtract(self):
        """
        Subtract the next same-type batch from the first batch.

        Finds the first batch element, then the next element of the same type,
        subtracts the second from the first, replaces the first with the result,
        and drops the second.

        Type-specific subtraction:
        - BatchComplex: first * conj(second) (phase subtraction on unit circle)
        - Batch: first - second (real subtraction)
        - BatchWrap: wrap(first - second) (wrapped phase subtraction)
        - BatchUnit: not supported (raises error)

        Returns
        -------
        Batches
            With first element replaced by subtracted result, second dropped.

        Examples
        --------
        >>> intfcorr = Batches([intf, model_intf]).subtract()
        """
        if len(self) < 2:
            raise ValueError("subtract() requires at least 2 elements")

        first_type = type(self[0])
        if isinstance(self[0], BatchUnit):
            raise TypeError("subtract() cannot be applied to BatchUnit")

        # Find next element of the same type
        second_idx = None
        for i in range(1, len(self)):
            if type(self[i]) is first_type:
                second_idx = i
                break
        if second_idx is None:
            raise ValueError(f"subtract() requires a second {first_type.__name__} element")

        first = self[0]
        second = self[second_idx]

        # Check if second is per-date and first is per-pair
        is_date_to_pair = False
        for key in first.keys():
            first_ds = first[key]
            second_ds = second[key]
            first_pol = [v for v in first_ds.data_vars if 'y' in first_ds[v].dims][0]
            second_pol = [v for v in second_ds.data_vars if 'y' in second_ds[v].dims][0]
            if 'pair' in first_ds[first_pol].dims and 'date' in second_ds[second_pol].dims:
                is_date_to_pair = True
            break

        if is_date_to_pair:
            result = _subtract_date_from_pair(first, second)
        elif isinstance(first, BatchComplex):
            result = utils_complex.interferogram(first, second)
        else:
            result = utils_complex.difference(first, second)

        result = Batches._preserve_nonspatial(first, result)

        elements = list(self)
        elements[0] = result
        elements.pop(second_idx)
        return Batches(elements)

    def displacement_los(self, transform):
        """
        Convert phase to line-of-sight displacement (meters).

        Applies Batch.displacement_los(transform) to the first element.

        Parameters
        ----------
        transform : Batch or Stack
            Transform batch or Stack providing radar_wavelength.

        Returns
        -------
        Batches
            Batches with [displacement Batch] preserving other elements.

        Examples
        --------
        >>> disp = stack.unwrap3d().displacement_los(stack.transform())
        """
        data = self[0]
        result = data.displacement_los(transform)
        elements = [result] + list(self[1:])
        return Batches(elements)

    def threshold(self, threshold=np.pi/2):
        """
        Filter pixels by circular standard deviation (cstd) of pair phases.

        Pixels with cstd >= threshold are set to NaN. The second element,
        if any, is the correlation weight and must be a BatchUnit.

        Parameters
        ----------
        threshold : float
            Maximum cstd in radians. Default π/2.

        Returns
        -------
        Batches
            Batches with filtered phase, preserving other elements.
        """
        phase = self[0]
        # the second element IS the weight: a BatchUnit or an error, never dropped (N81)
        weight = BatchCore._weight(self[1] if len(self) >= 2 else None)

        if not isinstance(phase, BatchComplex):
            raise TypeError(f"threshold() requires BatchComplex, got {type(phase).__name__}")

        filtered = phase.threshold(weight=weight, threshold=threshold)
        elements = [filtered] + list(self[1:])
        graph_check('threshold', elements, self)
        return Batches(elements)

    def fit1d(self, **kwargs):
        """
        Per-pixel model from the first element, dispatching on its type.

        Returns
        -------
        Batch
            The model: velocity, height, coherence, rmse, and `seasonal` from
            the variants that fit an annual term. Identical in name and unit to
            fit1d_network()'s, so predict() consumes either.
        """
        phase = self[0]
        # the second element IS the weight: a BatchUnit or an error, never dropped (N81)
        weight = BatchCore._weight(self[1] if len(self) >= 2 else None)
        # BatchComplex.fit1d() normalises every sample to a unit phasor, so a
        # magnitude weight has nothing to act on and it takes none -- complex
        # data carries no correlation. Only the unwrapped fit is offered one;
        # a weight given with complex data raises instead of being dropped.
        if weight is not None and isinstance(phase, BatchComplex):
            raise TypeError('ERROR: fit1d() of complex data takes no weight. '
                            'Use Batches([phase]).fit1d().')
        if weight is None:
            return phase.fit1d(**kwargs)
        return phase.fit1d(weight=weight, **kwargs)

    def rmse(self, solution):
        """RMSE of phase vs solution, using correlation weight if present.

        Extracts phase from self[0] and optional weight from self[1] (BatchUnit).
        Weight is automatically passed to the RMSE calculation and reduced
        to (y, x) via mean over the temporal dimension, in one task per
        (y, x) chunk.

        Parameters
        ----------
        solution : Batch
            Velocity (y, x), pair-based, or date-based solution.

        Returns
        -------
        Batches
            [RMSE Batch (y, x), mean weight BatchUnit (y, x)] when weight present,
            [RMSE Batch (y, x)] otherwise.
        """
        from . import utils_stats

        def mean_over(batch, dim):
            """batch.mean(dim): the grids with dim reduced in one task per (y, x) chunk (utils_stats.mean_pairs); the other
            variables carried as _agg carries them; the class of the batch kept."""
            def reduce_grid(grid):
                if dim not in grid.dims:
                    return grid
                grid = grid.transpose(dim, 'y', 'x')
                reduced = utils_stats.mean_pairs(grid.data)
                return xr.DataArray(reduced, dims=('y', 'x'), name=grid.name, attrs=grid.attrs,
                                    coords={k: v for k, v in grid.coords.items() if k in ('y', 'x', 'spatial_ref')})
            out = {}
            for key in batch:
                value = dict.__getitem__(batch, key)
                if isinstance(value, xr.DataArray):
                    out[key] = reduce_grid(value)
                    continue
                grids = BatchCore._grid_vars(value)
                ds = xr.Dataset({v: reduce_grid(value[v]) for v in grids})
                for v in value.data_vars:
                    if v not in grids:
                        ds[v] = value[v]
                ds.attrs = value.attrs
                out[key] = ds
            return batch._view(out)

        if len(self) < 1:
            raise ValueError("rmse() requires Batches with at least 1 element: [phase]")

        phase = self[0]
        # the second element IS the weight: a BatchUnit or an error, never dropped (N81)
        weight = BatchCore._weight(self[1] if len(self) >= 2 else None)

        rmse_result = phase.rmse(solution, weight=weight)

        if weight is not None:
            # Reduce weight to (y, x) — detect temporal dimension
            w_sample = next(iter(dict.values(weight)))
            # a DataArray weight (N81) is its own one grid, taken as it is
            w_spatial = [w_sample] if isinstance(w_sample, xr.DataArray) else \
                [w_sample[v] for v in w_sample.data_vars if 'y' in w_sample[v].dims]
            tdim = next((d for d in ('pair', 'date')
                         if w_spatial and d in w_spatial[0].dims), None)
            reduced_weight = mean_over(weight, tdim) if tdim else weight
            elements = [rmse_result, reduced_weight]
        else:
            elements = [rmse_result]

        graph_check('rmse', elements, [self, weight])
        return Batches(elements)

    def stl(self, freq='W', periods=52, robust=False):
        """
        Perform Seasonal-Trend decomposition using LOESS (STL).

        Expects Batches with [Batch (time series data)]. No weight parameter needed.

        Parameters
        ----------
        freq : str
            Frequency string for resampling (default 'W' for weekly).
        periods : int
            Number of periods for seasonal decomposition (default 52).
        robust : bool
            Whether to use robust fitting. Default False.

        Returns
        -------
        Batch
            Batch of Datasets with the 'trend', 'seasonal' and 'residual' variables
            (one (y, x) variable in, see Batch.stl()).

        Examples
        --------
        >>> stl_result = Batches([displacement]).stl(freq='W', periods=52)
        """
        if len(self) < 1:
            raise ValueError("stl() requires Batches with at least 1 element: [data]")

        data = self[0]

        # Delegate to Batch.stl
        return data.stl(freq=freq, periods=periods, robust=robust)

    def __getattr__(self, name):
        """Proxy unknown attributes to all batches if they're callable."""
        if name.startswith('_'):
            raise AttributeError(f"Batches has no attribute '{name}'")

        # Check if all batches have this attribute and it's callable
        attrs = [getattr(b, name, None) for b in self]
        if all(callable(a) for a in attrs if a is not None):
            def method(*args, **kwargs):
                results = [getattr(b, name)(*args, **kwargs) for b in self]
                # If results are Batch-like, wrap in Batches
                if results and hasattr(results[0], 'keys') and callable(results[0].keys):
                    return Batches(results)
                return tuple(results)
            return method

        raise AttributeError(f"Batches has no attribute '{name}'")
