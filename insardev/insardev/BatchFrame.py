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


class BatchFrame(tuple):
    """
    A (model, table) pair: a Batch and a table of points that come from one calculation.

    Unpacks as a tuple, lazy or computed:
        model, table = stack.fit3d_network(..., dates=True)
        model, table = stack.fit3d_network(..., dates=True).compute()
        stack.fit3d_network(..., dates=True).to_geopackage('insardev.gpkg')

    The lazy table is a dask-geopandas GeoDataFrame, one partition per part; the computed one is a geopandas GeoDataFrame. The
    parts and the model share the tasks of the calculation: compute() runs them once.
    """

    def __new__(cls, model, parts, meta, footprints=None):
        """
        Parameters
        ----------
        model : Batch
            The model, lazy or computed.
        parts : list
            The table's parts: dask.delayed objects that each return a geopandas GeoDataFrame (a lazy frame), or computed
            GeoDataFrames (a computed frame).
        meta : geopandas.GeoDataFrame
            The empty GeoDataFrame of the table's columns, dtypes and CRS.
        footprints : geopandas.GeoSeries, optional
            One footprint per part. They become the spatial_partitions of the lazy table.

        Returns
        -------
        BatchFrame
            The tuple (model, table).

        Examples
        --------
        >>> frame = BatchFrame(model, parts, meta, footprints)
        >>> model, table = frame
        """
        import dask
        parts = list(parts)
        if any(dask.is_dask_collection(part) for part in parts):
            import dask.dataframe as dd
            import dask_geopandas  # noqa: F401  (registers the GeoDataFrame collection of a geopandas meta)
            table = dd.from_delayed(parts, meta=meta, verify_meta=False)
            if footprints is not None:
                table.spatial_partitions = footprints
        else:
            table = BatchFrame._concat(parts, meta)
        self = super().__new__(cls, (model, table))
        self._parts = parts
        self._meta = meta
        self._footprints = footprints
        return self

    @staticmethod
    def _concat(frames, meta):
        """The computed parts as one geopandas GeoDataFrame with the dtypes and the CRS of meta."""
        import pandas as pd
        if not len(frames):
            return meta.copy()
        table = pd.concat(list(frames))
        table = table.astype({name: dtype for name, dtype in meta.dtypes.items() if name != meta.geometry.name})
        if table.crs is None:
            table = table.set_crs(meta.crs)
        return table

    def compute(self):
        """
        Compute the model and the table together.

        Adopted from BatchCore.compute: the model's bursts and the lazy parts are persisted in ONE dask.persist call, so the
        tasks they share run once; the progress is shown as for a Batch; the model keeps its data in the worker memory,
        rechunked to its input chunks; the parts are gathered into one geopandas GeoDataFrame. A frame with nothing lazy comes
        back as it is.

        Returns
        -------
        BatchFrame
            The computed model and the table as a geopandas GeoDataFrame.

        Raises
        ------
        RuntimeError
            The frame needs data the cluster no longer holds (e.g. after client.restart()).

        Examples
        --------
        >>> model, table = frame.compute()
        """
        import dask
        from .utils_dask import progress_persisted

        model = self[0]
        lazy_parts = [part for part in self._parts if dask.is_dask_collection(part)]
        # NOTHING LAZY, NOTHING TO COMPUTE
        if not lazy_parts and not BatchCore._is_lazy_any(model):
            return self

        # Save input chunk structure per burst
        all_input_chunks = {key: BatchCore._input_chunks(v) for key, v in model.items()}

        # One persist of the model's bursts and the parts: the shared tasks run once. optimize_graph=False: dask optimizes the
        # model's arrays and the delayed parts as separate groups; the array optimization fuses a shared task into its getitem
        # and renames it (level1_of_scans-getitem-<id>), while the parts keep _level1_of_scans-<id>. The keys differ, so the
        # task runs once for each (measured on the fit3d_network frame). Unoptimized, the key is shared.
        result, persisted_parts = dask.persist(dict(model), self._parts, optimize_graph=False)
        progress_persisted((result, persisted_parts), desc='Computing BatchFrame...'.ljust(25))

        # Finalize the model: materialize coordinates, hold the futures, rechunk to match input
        model_computed = type(model)({key: BatchCore._persisted(v, all_input_chunks[key]) for key, v in result.items()})
        # Gather the parts, which are on the cluster already
        frames = dask.compute(*persisted_parts) if lazy_parts else persisted_parts
        return BatchFrame(model_computed, [BatchFrame._concat(frames, self._meta)], self._meta)

    def to_geopackage(self, filename: str, crs: str | int | None = 4326, layer: str | None = None,
                      overwrite: bool = True) -> None:
        """
        Write the table to a GeoPackage as ONE layer, with the TRE column names.

        The columns named by an ISO date become d + YYYYMMDD (2018-10-17 becomes d20181017); the other columns keep their
        names. The points are reprojected to crs. A lazy frame computes its parts only, never the model.

        Parameters
        ----------
        filename : str
            The output GeoPackage; the .gpkg extension is added when missing.
        crs : str | int | None, optional
            The CRS of the written points. Default 4326; None keeps the table's own CRS.
        layer : str, optional
            The layer name. Default: the file's stem.
        overwrite : bool, optional
            Replace an existing file. Default True. When False a new layer is added to the file, and an existing layer of the
            same name gets the rows appended.

        Returns
        -------
        None

        Examples
        --------
        >>> frame.to_geopackage('insardev.gpkg')
        >>> frame.to_geopackage('insardev.gpkg', crs=32637, layer='dates')
        """
        import dask
        from . import utils_export

        if any(dask.is_dask_collection(part) for part in self._parts):
            table = BatchFrame._concat(dask.compute(*self._parts), self._meta)
        else:
            table = self[1]
        utils_export._table_to_geopackage(table, filename, crs, layer, overwrite)
