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
from .Stack_base import Stack_base
from insardev_toolkit import progressbar
from .Batch import Batch, BatchWrap, BatchUnit, BatchComplex
from . import utils_xarray

class Stack_phasediff(Stack_base):
    import xarray as xr
    import numpy as np
    import pandas as pd

    # internal method to compute interferogram on single polarization data array(s)
    def phasediff(self,
                       pairs:list[tuple[str|int,str|int]]|np.ndarray|pd.DataFrame|None=None,
                       weight:BatchUnit|None=None,
                       phase:xr.DataArray|None=None,
                       wavelength:float|None=None,
                       gaussian_threshold:float=0.5,
                       multilook:bool=True,
                       goldstein:int|list[int,int]|None=None,
                       complex:bool=False
                       ) -> tuple[xr.DataArray,xr.DataArray]:
        """
        Compute phase difference (interferogram) between pairs of dates.

        Parameters
        ----------
        pairs : list, np.ndarray, or pd.DataFrame
            Pairs of dates to compute interferograms for.
        weight : BatchUnit or None
            Per-burst weights for Gaussian filtering and masking. Use
            BatchUnit(stack.from_dataset(data)) to create from a single DataArray/Dataset.
        phase : xr.DataArray or None
            Optional phase to subtract (e.g., topographic phase).
        wavelength : float or None
            Gaussian filter wavelength for multilooking.
        gaussian_threshold : float
            Threshold for Gaussian filter (default 0.5).
        multilook : bool
            Whether to apply multilooking (default True).
        goldstein : int, list, or None
            Goldstein filter patch size.
        complex : bool
            Return complex values instead of phase angles (default False).

        Returns
        -------
        BatchWrap or tuple[BatchWrap, BatchUnit]
            Phase difference batch, or tuple of (phase, correlation) if wavelength is set.

        Examples
        --------
        # With a single weight raster, first convert to BatchUnit:
        weight = BatchUnit(stack.from_dataset(corr))
        intf, corr = stack.phasediff(pairs, weight=weight, wavelength=200)
        """
        import numpy as np

        if goldstein is not None and wavelength is None:
            raise ValueError('wavelength is required to define spatial correlation for Goldstein filtering')

        # validate weight type
        if weight is not None and not isinstance(weight, BatchUnit):
            raise TypeError(
                f'weight must be a BatchUnit, got {type(weight).__name__}. '
                'Use BatchUnit(stack.from_dataset(data)) to convert a single DataArray.'
            )

        pairs = np.array(pairs if isinstance(pairs[0], (list, tuple, np.ndarray)) else [pairs])
        # Check for duplicate pairs
        unique, counts = np.unique(pairs, axis=0, return_counts=True)
        if (counts > 1).any():
            duplicates = unique[counts > 1]
            raise ValueError(f'Input pairs contain duplicates: {duplicates.tolist()}')
        ref_dates = pairs[:,0]
        rep_dates = pairs[:,1]
        n_pairs = len(ref_dates)

        # Rename date->pair and reset to integer index (dates can have duplicates across pairs)
        data1 = self.isel(date=ref_dates).rename(date='pair')
        data2 = self.isel(date=rep_dates).rename(date='pair')
        data1 = data1.map(lambda ds: ds.assign_coords(pair=np.arange(n_pairs)))
        data2 = data2.map(lambda ds: ds.assign_coords(pair=np.arange(n_pairs)))
        phasediff = BatchComplex(data1) * BatchComplex(data2).conj()
        if phase is not None:
            if isinstance(phase, BatchComplex):
                phasediff = phasediff * phase
            else:
                # Convert phase to complex phasor
                phasediff = phasediff * phase.iexp(-1)

        corr_look = None
        if wavelength is not None:
            # Gaussian filtering with cut-off wavelength on phase difference
            phasediff_look = phasediff.gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold)

            # Gaussian filtering with cut-off wavelength on amplitudes
            # Extract only the needed dates FIRST, then compute power and filter
            data1_power = BatchComplex(data1).power()
            data2_power = BatchComplex(data2).power()
            intensity_look1 = data1_power.gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold)
            intensity_look2 = data2_power.gaussian(weight=weight, wavelength=wavelength, threshold=gaussian_threshold)

            # correlation requires multilooking to detect influence between pixels
            corr_look = (phasediff_look.abs() / (intensity_look1 * intensity_look2).sqrt()).clip(0, 1)

        # keep phase difference without multilooking if multilook=False
        if not multilook or wavelength is None:
            phasediff_look = phasediff
        if goldstein is not None:
            phasediff_look = phasediff_look.goldstein(corr_look, goldstein)

        # filter out not valid pixels (per-burst masking using weight)
        if weight is not None:
            phasediff_look = phasediff_look.where(weight.isfinite())
            corr_look = corr_look.where(weight.isfinite()) if corr_look is not None else None
        
        if not complex:
            phasediff_look = phasediff_look.angle()

        # BPR differences aligned with pair dimension: BPR(rep) - BPR(ref)
        bpr = data2[['BPR']] - data1[['BPR']]
        #print ('bpr', bpr.to_dict())

        # Store original datetime values for ref/rep before they get replaced with integers
        ref_values = self.isel(date=ref_dates).coords['date'].values
        rep_values = self.isel(date=rep_dates).coords['date'].values

        def as_xarray(batch):
            # Add ref/rep/BPR as non-dimension coordinates along pair dimension
            # pair dimension uses positional indexing (no explicit coordinate needed)
            return batch.assign_coords(
                ref=('pair', ref_values),
                rep=('pair', rep_values),
                BPR=('pair', bpr)
            )
        
        if corr_look is None:
            return as_xarray(phasediff_look)
        return (as_xarray(phasediff_look), as_xarray(corr_look))

    def phasediff_singlelook(self, *args, **kwarg):
        from .Batch import BatchComplex
        kwarg['multilook'] = False
        return self.phasediff(*args, **kwarg)
        #intfs, corrs = self.phasediff(**kwarg)
        #return BatchWrap(intfs), BatchUnit(corrs)

    def phasediff_multilook(self, *args, **kwarg):
        from .Batch import BatchComplex
        kwarg['multilook'] = True
        return self.phasediff(*args, **kwarg)
