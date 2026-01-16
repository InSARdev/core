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

def corrcoef(data, mask_diagonal=False):
    """
    Calculate the correlation coefficients matrix for a stack of interferograms.

    Parameters:
    - data (xarray.DataArray): A 3D input grid representing a stack of interferograms.
                                The dimensions should include time, 'y', and 'x'.
    - mask_diagonal (bool): If True, the diagonal elements of the correlation matrix
                            will be set to NaN to exclude self-correlation.

    Returns:
    xarray.DataArray: The cross-correlation matrix.
    
    Example:
    # plot variance-covariance matrix
    corr_stack = corr60m.mean('pair')
    corr = utils.corrcoef(unwrap.phase.where(corr_stack.where(corr_stack>0.7)))
    corr.where((corr)>0.7).plot.imshow(vmin=-1, vmax=1, interpolation='none')
    plt.xticks(rotation=90)
    plt.show()
    
    # find the most correlated interferograms to exclude
    corr = utils.corrcoef(unwrap.phase.where(corr_stack.where(corr_stack>0.7)), mask_diagonal=True)
    df = corr.where(corr>0.7).to_dataframe('').reset_index().dropna()
    exclude = np.unique(np.concatenate([df.ref, df.rep]))
    exclude
    """
    import xarray as xr
    import dask
    import numpy as np
    assert len(data.dims) == 3, 'ERROR: expected 3D input grid'
    stackdim = data.dims[0]
    stackvals = data[stackdim].values
    corr = dask.array.corrcoef(data.stack(points=['y', 'x']).dropna(dim='points').data).round(2)
    corr = xr.DataArray((corr), coords={'ref': stackvals, 'rep': stackvals}).rename('corr')
    corr['ref'] = corr['ref'].astype(str)
    corr['rep'] = corr['rep'].astype(str)
    if mask_diagonal:
        # set diagonal elements to NaN
        diagonal_mask = np.eye(len(stackvals), dtype=bool)
        return corr.where(~diagonal_mask, np.nan)
    return corr
