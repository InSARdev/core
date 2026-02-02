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
        t_data.nan_to_num_(0.0)  # in-place

        # Combine with external weights
        if wgt is not None:
            t_weight = torch.from_numpy(wgt.astype(np.float32)).to(dev)
            t_weight.nan_to_num_(0.0)  # in-place
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


import cv2 as _cv2


def nanconvolve2d_gaussian_numpy(data_np, weight_np=None, sigma=None, truncate=4.0, threshold=0.5):
    """
    2D Gaussian convolution with NaN handling using OpenCV separable filter.

    Memory-optimized with in-place operations. Requires float32 input.

    Parameters
    ----------
    data_np : np.ndarray
        2D numpy array (float32) to convolve, can be real or complex.
    weight_np : np.ndarray, optional
        Weight array (float32) for weighted convolution.
    sigma : float or tuple of floats
        Standard deviation of Gaussian kernel (y, x).
    truncate : float, optional
        Truncation factor for kernel size (default 4.0).
    threshold : float, optional
        Minimum weight fraction for valid output (default 0.5).

    Returns
    -------
    np.ndarray
        Convolved array (float32) with same shape as input.
    """
    import numpy as np

    if sigma is None:
        return data_np

    if not isinstance(sigma, (list, tuple, np.ndarray)):
        sigma = (sigma, sigma)

    is_complex = np.issubdtype(data_np.dtype, np.complexfloating)

    def make_gaussian_kernel(s, trunc):
        size = int(2 * np.ceil(s * trunc) + 1)
        x = np.arange(size) - size // 2
        kernel = np.exp(-0.5 * (x / s) ** 2)
        return (kernel / kernel.sum()).astype(np.float32)

    kernel_y = make_gaussian_kernel(sigma[0], truncate)
    kernel_x = make_gaussian_kernel(sigma[1], truncate)

    def convolve_real(arr, wgt=None):
        """Separable convolution using OpenCV."""
        assert arr.dtype == np.float32, f"Input must be float32, got {arr.dtype}"
        nan_mask = np.isnan(arr)
        arr_clean = arr.copy()
        np.copyto(arr_clean, 0.0, where=nan_mask)

        if wgt is not None:
            assert wgt.dtype == np.float32, f"Weight must be float32, got {wgt.dtype}"
            wgt_clean = wgt.copy()
            # Zero where EITHER wgt or arr has NaN (two calls to avoid temp bool array)
            np.copyto(wgt_clean, 0.0, where=np.isnan(wgt))
            np.copyto(wgt_clean, 0.0, where=nan_mask)
        else:
            # Weight = 1 where data valid, 0 where NaN
            # Use subtract to avoid temp bool array from ~nan_mask (saves 122 MB for 488 MB input)
            wgt_clean = np.subtract(1.0, nan_mask, dtype=np.float32)
        del nan_mask

        # In-place multiply to avoid temp array
        arr_clean *= wgt_clean
        numerator = _cv2.sepFilter2D(arr_clean, -1, kernel_x, kernel_y,
                                     borderType=_cv2.BORDER_REPLICATE)
        assert numerator.dtype == np.float32, f"numerator must be float32, got {numerator.dtype}"
        del arr_clean
        denominator = _cv2.sepFilter2D(wgt_clean, -1, kernel_x, kernel_y,
                                       borderType=_cv2.BORDER_REPLICATE)
        assert denominator.dtype == np.float32, f"denominator must be float32, got {denominator.dtype}"
        del wgt_clean

        # In-place division: reuse numerator array as result
        # Store threshold mask before modifying denominator
        if wgt is None:
            low_weight_mask = denominator <= threshold
        else:
            low_weight_mask = denominator < 1e-10
        denominator += 1e-17  # in-place to avoid temp array
        numerator /= denominator
        del denominator
        # Set NaN where denominator was too low
        np.copyto(numerator, np.nan, where=low_weight_mask)
        del low_weight_mask

        assert numerator.dtype == np.float32, f"Output must be float32, got {numerator.dtype}"
        return numerator

    if is_complex:
        # Pre-allocate and assign real/imag separately to avoid temp arrays
        result = np.empty(data_np.shape, dtype=np.complex64)
        result.real = convolve_real(data_np.real, weight_np)
        result.imag = convolve_real(data_np.imag, weight_np)
    else:
        result = convolve_real(data_np, weight_np)

    return result


def _get_torch_device(device='auto', debug=False):
    """Get PyTorch device. Use Batch._get_torch_device for the canonical implementation."""
    from .Batch import Batch
    return Batch._get_torch_device(device, debug)


def gaussian_numpy(data_np, weight_np=None, sigma=None, truncate=4.0, threshold=0.5, device='auto',
                   pixel_sizes=None, resolution=67.0):
    """
    2D Gaussian convolution with separable kernels.

    Uses scipy.ndimage for CPU (memory-efficient) and PyTorch for GPU.
    Supports complex values, weights, and NaN handling.
    For large sigmas with pixel_sizes provided, automatically decimates like PyGMTSAR.

    Parameters
    ----------
    data_np : np.ndarray
        2D numpy array to convolve, can be real or complex.
    weight_np : np.ndarray, optional
        Weight array for weighted convolution.
    sigma : float or tuple of floats
        Standard deviation of Gaussian kernel (y, x) in pixels.
    truncate : float, optional
        Truncation factor for kernel size (default 4.0).
    threshold : float, optional
        Minimum weight fraction for valid output (default 0.5).
    device : str, optional
        PyTorch device: 'auto', 'cuda', 'mps', or 'cpu'.
    pixel_sizes : tuple of floats, optional
        Pixel sizes (dy, dx) in meters. Required for decimation.
    resolution : float, optional
        Processing resolution in meters for decimation (default 67.0).
        Decimation triggers when scales > 1 and wavelength/resolution > 3.
        With 67m, common 200m wavelength uses direct filtering (200/67=2.98).

    Returns
    -------
    np.ndarray
        Convolved array with same shape as input.
    """
    import numpy as np

    if sigma is None:
        return data_np

    if not isinstance(sigma, (list, tuple, np.ndarray)):
        sigma = (sigma, sigma)
    sigma = np.array(sigma, dtype=np.float64)

    # Handle (1, y, x) arrays from apply_ufunc with core_dims=[['y', 'x']]
    squeeze = False
    if data_np.ndim == 3 and data_np.shape[0] == 1:
        data_np = data_np[0]
        squeeze = True

    # Ensure correct dtypes (nanconvolve2d requires float32/complex64)
    is_complex = np.issubdtype(data_np.dtype, np.complexfloating)
    if is_complex:
        if data_np.dtype != np.complex64:
            data_np = data_np.astype(np.complex64)
    else:
        if data_np.dtype != np.float32:
            data_np = data_np.astype(np.float32)
    if weight_np is not None and weight_np.dtype != np.float32:
        weight_np = weight_np.astype(np.float32)

    # Get device
    dev = _get_torch_device(device)

    # GMTSAR constant
    cutoff = 5.3

    # Check if decimation needed (PyGMTSAR approach with fixed resolution)
    if pixel_sizes is not None:
        dy, dx = pixel_sizes
        scale_y = int(np.round(resolution / dy))
        scale_x = int(np.round(resolution / dx))
        # Recover wavelength from sigma: wavelength = sigma * cutoff * pixel_size
        wavelength = sigma[0] * cutoff * dy
    else:
        scale_y = scale_x = 1
        wavelength = None

    # Decimate when scales > 1 and wavelength large enough (PyGMTSAR approach)
    # CPU: wavelength/resolution > 2 (i.e. wavelength > ~134m) - decimation always faster
    # GPU: wavelength/resolution > 24 (~1600m) - direct filtering is fast for smaller kernels
    use_decimation = (scale_y > 1 and scale_x > 1 and wavelength is not None
                      and wavelength / resolution > (2 if dev.type == 'cpu' else 24))
    if use_decimation:
        original_shape = data_np.shape

        # AA sigma from resolution wavelength (same for both dimensions like PyGMTSAR)
        sigma_aa = (resolution / cutoff / dy, resolution / cutoff / dx)

        # wavelength_dec = sqrt(wavelength^2 - resolution^2)
        wavelength_dec = np.sqrt(wavelength**2 - resolution**2)

        # After coarsen, pixel sizes become (dy*scale_y, dx*scale_x)
        dy_dec = dy * scale_y
        dx_dec = dx * scale_x
        sigma_dec = (wavelength_dec / cutoff / dy_dec, wavelength_dec / cutoff / dx_dec)

        if dev.type == 'cpu':
            # CPU path: numpy operations
            # 1. AA filter
            data_aa = nanconvolve2d_gaussian_numpy(
                data_np, weight_np, sigma=sigma_aa, truncate=truncate, threshold=threshold
            )
            # 2. Coarsen (einsum)
            ny_trim = (data_aa.shape[0] // scale_y) * scale_y
            nx_trim = (data_aa.shape[1] // scale_x) * scale_x
            data_trimmed = data_aa[:ny_trim, :nx_trim]
            del data_aa
            dec_shape = (ny_trim // scale_y, nx_trim // scale_x)
            reshaped = data_trimmed.reshape(dec_shape[0], scale_y, dec_shape[1], scale_x)
            # In-place division avoids temporary array
            data_dec = np.einsum('ijkl->ik', reshaped, dtype=data_trimmed.dtype)
            del data_trimmed, reshaped
            data_dec /= (scale_y * scale_x)
            # 3. Main filter
            result_dec = nanconvolve2d_gaussian_numpy(
                data_dec, None, sigma=sigma_dec, truncate=truncate, threshold=threshold
            )
            del data_dec
            # 4. Upsample (np.repeat) - split to allow earlier del
            result = np.repeat(result_dec, scale_y, axis=0)
            del result_dec
            result = np.repeat(result, scale_x, axis=1)
            pad_y = original_shape[0] - result.shape[0]
            pad_x = original_shape[1] - result.shape[1]
            if pad_y > 0 or pad_x > 0:
                result = np.pad(result, ((0, pad_y), (0, pad_x)), mode='edge')
        else:
            # GPU path: keep data on GPU throughout, avoid CPU transfers
            import torch
            import torch.nn.functional as F

            # Helper: create 1D Gaussian kernel on GPU
            def make_kernel_1d(s, trunc):
                size = int(2 * np.ceil(s * trunc) + 1)
                x = torch.arange(size, device=dev, dtype=torch.float32) - size // 2
                k = torch.exp(-0.5 * (x / s) ** 2)
                return k / k.sum()

            # Helper: separable Gaussian convolution on GPU
            def gpu_gaussian(t_data, sig):
                ky = make_kernel_1d(sig[0], truncate).view(1, 1, -1, 1)
                kx = make_kernel_1d(sig[1], truncate).view(1, 1, 1, -1)
                pad_y, pad_x = ky.shape[2] // 2, kx.shape[3] // 2
                # Valid mask and weighted convolution
                valid = (~torch.isnan(t_data)).float()
                t_data = t_data.nan_to_num(0.0)
                t = (t_data * valid).unsqueeze(0).unsqueeze(0)
                t = F.pad(t, (0, 0, pad_y, pad_y), mode='replicate')
                t = F.conv2d(t, ky)
                t = F.pad(t, (pad_x, pad_x, 0, 0), mode='replicate')
                numer = F.conv2d(t, kx).squeeze()
                w = valid.unsqueeze(0).unsqueeze(0)
                w = F.pad(w, (0, 0, pad_y, pad_y), mode='replicate')
                w = F.conv2d(w, ky)
                w = F.pad(w, (pad_x, pad_x, 0, 0), mode='replicate')
                denom = F.conv2d(w, kx).squeeze()
                return torch.where(denom > threshold, numer / (denom + 1e-10), torch.nan)

            # Convert to GPU tensor
            t_data = torch.from_numpy(data_np.astype(np.float32)).to(dev)

            # 1. AA filter (GPU)
            t_aa = gpu_gaussian(t_data, sigma_aa)
            del t_data

            # 2. Coarsen with avg_pool2d (GPU)
            ny_trim = (t_aa.shape[0] // scale_y) * scale_y
            nx_trim = (t_aa.shape[1] // scale_x) * scale_x
            t_trimmed = t_aa[:ny_trim, :nx_trim]
            del t_aa
            t_dec = F.avg_pool2d(t_trimmed.unsqueeze(0).unsqueeze(0),
                                 kernel_size=(scale_y, scale_x),
                                 stride=(scale_y, scale_x)).squeeze()
            del t_trimmed

            # 3. Main filter (GPU)
            t_result = gpu_gaussian(t_dec, sigma_dec)
            del t_dec

            # 4. Upsample with F.interpolate (GPU)
            t_up = F.interpolate(t_result.unsqueeze(0).unsqueeze(0),
                                 scale_factor=(scale_y, scale_x),
                                 mode='nearest')
            del t_result

            # Pad to original shape (keep 4D for F.pad)
            pad_y = original_shape[0] - t_up.shape[2]
            pad_x = original_shape[1] - t_up.shape[3]
            if pad_y > 0 or pad_x > 0:
                t_up = F.pad(t_up, (0, pad_x, 0, pad_y), mode='replicate')
            result = t_up.squeeze().cpu().numpy()
            del t_up

            # Clear GPU cache
            if dev.type == 'mps':
                torch.mps.empty_cache()
            elif dev.type == 'cuda':
                torch.cuda.empty_cache()
    else:
        # Direct filtering (no decimation)
        if dev.type == 'cpu':
            result = nanconvolve2d_gaussian_numpy(
                data_np, weight_np, sigma=tuple(sigma), truncate=truncate, threshold=threshold
            )
        else:
            result = nanconvolve2d_gaussian_pytorch(
                data_np, weight_np, sigma=tuple(sigma), truncate=truncate, threshold=threshold, device=dev
            )

    if squeeze:
        result = result[np.newaxis, ...]
    return result
