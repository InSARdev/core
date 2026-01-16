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

def nanconvolve2d_gaussian(data,
                    weight=None,
                    sigma=None,
                    mode='reflect',
                    truncate=4.0,
                    threshold=0.5):
    """
    Convolve a data array with a Gaussian kernel.

    Parameters
    ----------
    data : xarray.DataArray
        The data array to convolve.
    weight : xarray.DataArray, optional
        The weight array to use for the convolution.
    sigma : float or tuple of floats, optional
        The standard deviation of the Gaussian kernel.
    mode : str, optional
        The mode to use for the convolution.
    truncate : float, optional
        The truncation factor for the Gaussian kernel.
    threshold : float, optional
        The threshold for the convolution.

    We use a threshold defined as a fraction of the weight as an indicator that the Gaussian window
    covers enough valid (non-NaN) pixels. When the accumulated weight is below this threshold, we replace
    the output with NaN, since the result is unreliable due to insufficient data within the window.
    This is a simple way to prevent border effects when most of the filter window is empty.
    """
    import numpy as np
    import xarray as xr

    if sigma is None:
        return data

    if not isinstance(sigma, (list, tuple, np.ndarray)):
        sigma = (sigma, sigma)
    depth = [np.ceil(_sigma * truncate).astype(int) for _sigma in sigma]
    #print ('sigma', sigma, 'depth', depth)

    # weighted Gaussian filtering for real floats with NaNs
    def nanconvolve2d_gaussian_floating_dask_chunk(data, weight=None, **kwargs):
        import numpy as np
        from scipy.ndimage import gaussian_filter
        assert not np.issubdtype(data.dtype, np.complexfloating)
        assert np.issubdtype(data.dtype, np.floating)
        if weight is not None:
            assert not np.issubdtype(weight.dtype, np.complexfloating)
            assert np.issubdtype(weight.dtype, np.floating)
        # all other arguments are passed to gaussian_filter
        threshold = kwargs.pop('threshold')
        # replace nan + 1j to to 0.+0.j
        data_complex  = (1j + data) * (weight if weight is not None else 1)
        conv_complex = gaussian_filter(np.nan_to_num(data_complex, 0), **kwargs)
        #conv = conv_complex.real/conv_complex.imag
        # to prevent "RuntimeWarning: invalid value encountered in divide" even when warning filter is defined
        # threshold check: mask pixels where accumulated weight is too low
        # when weight is provided, conv_complex.imag is smoothed(weight), only mask near-zero values
        # (matching PyGMTSAR behavior)
        conv = np.where(conv_complex.imag <= threshold if weight is None else conv_complex.imag < 1e-10, np.nan, conv_complex.real/(conv_complex.imag + 1e-17))
        del data_complex, conv_complex
        return conv

    def nanconvolve2d_gaussian_dask_chunk(data, weight=None, **kwargs):
        import numpy as np
        if np.issubdtype(data.dtype, np.complexfloating):
            #print ('complexfloating')
            real = nanconvolve2d_gaussian_floating_dask_chunk(data.real, weight, **kwargs)
            imag = nanconvolve2d_gaussian_floating_dask_chunk(data.imag, weight, **kwargs)
            conv = real + 1j*imag
            del real, imag
        else:
            #print ('floating')
            conv = nanconvolve2d_gaussian_floating_dask_chunk(data.real, weight, **kwargs)
        return conv

    # weighted Gaussian filtering for real or complex floats
    def nanconvolve2d_gaussian_dask(data, weight, **kwargs):
        import dask.array as da
        # ensure both dask arrays have the same chunk structure
        # use map_overlap with the custom function to handle both arrays
        return da.map_overlap(
            nanconvolve2d_gaussian_dask_chunk,
            *([data, weight] if weight is not None else [data]),
            depth={0: depth[0], 1: depth[1]},
            boundary='none',
            dtype=data.dtype,
            meta=data._meta,
            **kwargs
        )
    #print ('data', data)
    #print ('weight', weight)
    return xr.DataArray(nanconvolve2d_gaussian_dask(data.data,
                                    weight.data if weight is not None else None,
                                    threshold=threshold,
                                    sigma=sigma,
                                    mode=mode,
                                    truncate=truncate),
                        coords=data.coords,
                        name=data.name)
