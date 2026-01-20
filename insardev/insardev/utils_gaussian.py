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

def nanconvolve2d_gaussian_pytorch(data_np, weight_np=None, sigma=None, truncate=4.0, threshold=0.5, device='cpu'):
    """
    2D Gaussian convolution using PyTorch with separable kernels.

    Optimized for GPU (MPS/CUDA). For CPU, use nanconvolve2d_gaussian_scipy instead.

    Parameters
    ----------
    data_np : np.ndarray
        2D numpy array to convolve, can be real or complex.
    weight_np : np.ndarray, optional
        Weight array for weighted convolution.
    sigma : float or tuple of floats
        Standard deviation of Gaussian kernel (y, x).
    truncate : float, optional
        Truncation factor for kernel size (default 4.0).
    threshold : float, optional
        Minimum weight fraction for valid output (default 0.5).
    device : str or torch.device
        PyTorch device: 'cuda', 'mps', or 'cpu'.

    Returns
    -------
    np.ndarray
        Convolved array with same shape as input.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    if sigma is None:
        return data_np

    if not isinstance(sigma, (list, tuple, np.ndarray)):
        sigma = (sigma, sigma)

    dev = torch.device(device) if isinstance(device, str) else device
    is_complex = np.issubdtype(data_np.dtype, np.complexfloating)

    # Create 1D Gaussian kernels
    def make_kernel_1d(s, trunc):
        size = int(2 * np.ceil(s * trunc) + 1)
        x = torch.arange(size, device=dev, dtype=torch.float32) - size // 2
        k = torch.exp(-0.5 * (x / s) ** 2)
        return k / k.sum()

    kernel_y = make_kernel_1d(sigma[0], truncate)
    kernel_x = make_kernel_1d(sigma[1], truncate)
    pad_y = len(kernel_y) // 2
    pad_x = len(kernel_x) // 2

    # Reshape kernels for conv2d: (out_ch, in_ch, H, W)
    kernel_y = kernel_y.view(1, 1, -1, 1)
    kernel_x = kernel_x.view(1, 1, 1, -1)

    def separable_conv(t):
        """Apply separable Gaussian convolution with replicate padding."""
        t = t.unsqueeze(0).unsqueeze(0)
        t = F.pad(t, (0, 0, pad_y, pad_y), mode='replicate')
        t = F.conv2d(t, kernel_y)
        t = F.pad(t, (pad_x, pad_x, 0, 0), mode='replicate')
        t = F.conv2d(t, kernel_x)
        return t.squeeze()

    def convolve_real(arr, wgt=None):
        """Convolve a real 2D array with NaN/weight handling."""
        t_data = torch.from_numpy(arr.astype(np.float32)).to(dev)

        # Valid mask (1 where not NaN)
        valid = (~torch.isnan(t_data)).float()
        t_data = torch.nan_to_num(t_data, 0.0)

        # Combine with external weights
        if wgt is not None:
            t_weight = torch.from_numpy(wgt.astype(np.float32)).to(dev)
            t_weight = torch.nan_to_num(t_weight, 0.0)
            combined_weight = valid * t_weight
        else:
            combined_weight = valid

        # Weighted convolution
        numerator = separable_conv(t_data * combined_weight)
        denominator = separable_conv(combined_weight)

        # Result with threshold masking
        if wgt is None:
            result = torch.where(denominator > threshold, numerator / (denominator + 1e-10), torch.nan)
        else:
            result = torch.where(denominator > 1e-10, numerator / (denominator + 1e-10), torch.nan)

        return result.cpu().numpy()

    if is_complex:
        result = convolve_real(data_np.real, weight_np) + 1j * convolve_real(data_np.imag, weight_np)
    else:
        result = convolve_real(data_np, weight_np)

    # Cleanup GPU memory
    if dev.type == 'mps':
        torch.mps.empty_cache()
    elif dev.type == 'cuda':
        torch.cuda.empty_cache()

    return result


def nanconvolve2d_gaussian_scipy_numpy(data_np, weight_np=None, sigma=None, truncate=4.0, threshold=0.5):
    """
    2D Gaussian convolution using scipy with complex trick for efficiency.

    Memory-efficient implementation for CPU. Uses complex numbers to compute
    both numerator and denominator in a single gaussian_filter call.

    Parameters
    ----------
    data_np : np.ndarray
        2D numpy array to convolve, can be real or complex.
    weight_np : np.ndarray, optional
        Weight array for weighted convolution.
    sigma : float or tuple of floats
        Standard deviation of Gaussian kernel (y, x).
    truncate : float, optional
        Truncation factor for kernel size (default 4.0).
    threshold : float, optional
        Minimum weight fraction for valid output (default 0.5).

    Returns
    -------
    np.ndarray
        Convolved array with same shape as input.
    """
    import numpy as np
    from scipy.ndimage import gaussian_filter

    if sigma is None:
        return data_np

    if not isinstance(sigma, (list, tuple, np.ndarray)):
        sigma = (sigma, sigma)

    is_complex = np.issubdtype(data_np.dtype, np.complexfloating)

    def convolve_real(arr, wgt=None):
        """Single-pass convolution using scipy gaussian_filter."""
        arr = arr.astype(np.float32)
        # Use complex trick: real part = weighted data, imag part = weights
        data_complex = (1j + arr) * (wgt if wgt is not None else 1)
        conv_complex = gaussian_filter(np.nan_to_num(data_complex, 0), sigma=sigma, truncate=truncate, mode='nearest')
        # Threshold check: mask pixels where accumulated weight is too low
        if wgt is None:
            result = np.where(conv_complex.imag <= threshold, np.nan, conv_complex.real / (conv_complex.imag + 1e-17))
        else:
            result = np.where(conv_complex.imag < 1e-10, np.nan, conv_complex.real / (conv_complex.imag + 1e-17))
        return result.astype(np.float32)

    if is_complex:
        result = convolve_real(data_np.real, weight_np) + 1j * convolve_real(data_np.imag, weight_np)
    else:
        result = convolve_real(data_np, weight_np)

    return result


def nanconvolve2d_gaussian_scipy(data,
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
