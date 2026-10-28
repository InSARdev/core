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

from .utils_torch import serialize_gpu as _serialize_gpu

@_serialize_gpu
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
    import cv2 as _cv2

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


# GMTSAR constant: the half-gain wavelength is 5.3 sigma
GAUSSIAN_CUTOFF = 5.3


def decimation_plan(sigma, pixel_sizes, device_type, truncate=4.0, resolution=67.0):
    """
    The decimation gaussian_numpy() applies to a filter of `sigma` pixels (y, x), or None when it filters directly.

    Decimation (PyGMTSAR approach) triggers when the decimation scales are above 1 and the wavelength over the
    resolution is above 2 on the CPU (wavelength > ~134 m: decimation is always faster there) or above 24 on a GPU
    (~1600 m: direct filtering is fast for smaller kernels).

    Returns a dict:
      scale   (scale_y, scale_x): the decimation cell in pixels
      sigma_aa, sigma_dec: the sigmas (y, x) of the anti-aliasing filter (pixels) and of the main filter (cells)
      halo    (y, x): the overlap depth in pixels a block needs for its results to be those of the whole array. A
              pixel depends on the cells within the main filter reach of its cell, each of them on the pixels of
              the cell and the anti-aliasing reach beyond; the cell grid starts up to scale - 1 pixels inside the
              block and the last cell of the block may be cut by the same amount (see `skip` of gaussian_numpy).
    """
    import numpy as np

    if pixel_sizes is None:
        return None
    sigma = np.asarray(sigma, dtype=np.float64)
    dy, dx = pixel_sizes
    scale_y = int(np.round(resolution / dy))
    scale_x = int(np.round(resolution / dx))
    # Recover wavelength from sigma: wavelength = sigma * cutoff * pixel_size
    wavelength = sigma[0] * GAUSSIAN_CUTOFF * dy
    if not (scale_y > 1 and scale_x > 1 and wavelength / resolution > (2 if device_type == 'cpu' else 24)):
        return None
    # AA sigma from resolution wavelength (same for both dimensions like PyGMTSAR)
    sigma_aa = (resolution / GAUSSIAN_CUTOFF / dy, resolution / GAUSSIAN_CUTOFF / dx)
    # wavelength_dec = sqrt(wavelength^2 - resolution^2); after coarsen, pixel sizes become (dy*scale_y, dx*scale_x)
    wavelength_dec = np.sqrt(wavelength**2 - resolution**2)
    sigma_dec = (wavelength_dec / GAUSSIAN_CUTOFF / (dy * scale_y), wavelength_dec / GAUSSIAN_CUTOFF / (dx * scale_x))
    halo = tuple(int(s * (np.ceil(sd * truncate) + 2) + np.ceil(sa * truncate))
                 for s, sd, sa in zip((scale_y, scale_x), sigma_dec, sigma_aa))
    return dict(scale=(scale_y, scale_x), sigma_aa=sigma_aa, sigma_dec=sigma_dec, halo=halo)


@_serialize_gpu
def gaussian_numpy(data_np, weight_np=None, sigma=None, truncate=4.0, threshold=0.5, device='auto',
                   pixel_sizes=None, resolution=67.0, skip=(0, 0)):
    """
    2D Gaussian convolution with separable kernels.

    Uses scipy.ndimage for CPU (memory-efficient) and PyTorch for GPU.
    Supports complex values, weights, and NaN handling.
    For large sigmas with pixel_sizes provided, automatically decimates like PyGMTSAR (see decimation_plan()).

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
        Processing resolution in meters for decimation (default 67.0). See decimation_plan() for when it triggers.
    skip : tuple of int, optional
        (rows, columns) at the start of the block before the first decimation cell: the decimation grid starts at
        that row and column, the rows and columns before them are filled by replication and are meant to be trimmed.
        Used for a block of a larger array, whose decimation grid belongs to the whole array. It has no effect
        without decimation. Default (0, 0).

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
    # its (1, y, x) weight block too: left 3-D, it failed against the 2-D data
    # (a per-pair weight on per-pair data, N81)
    if weight_np is not None and weight_np.ndim == 3 and weight_np.shape[0] == 1:
        weight_np = weight_np[0]

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

    # Check if decimation needed (PyGMTSAR approach with fixed resolution)
    plan = decimation_plan(sigma, pixel_sizes, dev.type, truncate, resolution)
    if plan is not None:
        scale_y, scale_x = plan['scale']
        sigma_aa, sigma_dec = plan['sigma_aa'], plan['sigma_dec']

        # Complex input on the GPU decimation path: the rest of this branch works
        # in float32 and would drop the imaginary part. A Gaussian is linear, so
        # filtering the parts separately is exact.
        if is_complex and dev.type != 'cpu':
            kw = dict(weight_np=weight_np, sigma=sigma, truncate=truncate,
                      threshold=threshold, device=device, pixel_sizes=pixel_sizes,
                      resolution=resolution, skip=skip)
            re_ = gaussian_numpy(np.ascontiguousarray(data_np.real), **kw)
            im_ = gaussian_numpy(np.ascontiguousarray(data_np.imag), **kw)
            result = (re_ + 1j * im_).astype(np.complex64)
            # the (1, y, x) block was squeezed above: it comes back as it came
            return result[np.newaxis, ...] if squeeze else result

        # THE DECIMATION GRID BELONGS TO THE WHOLE ARRAY, NOT TO THE BLOCK: the cells start at row sy and column sx of
        # the block (the block of a larger array starts inside a cell of the array's grid), the rows and columns
        # before them are filled by replication below and are overlap that the caller trims
        original_shape = data_np.shape
        sy, sx = skip
        data_np = data_np[sy:, sx:]
        if weight_np is not None:
            weight_np = weight_np[sy:, sx:]

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
            pad_y = original_shape[0] - sy - result.shape[0]
            pad_x = original_shape[1] - sx - result.shape[1]
            if sy or sx or pad_y > 0 or pad_x > 0:
                result = np.pad(result, ((sy, pad_y), (sx, pad_x)), mode='edge')
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

            # Helper: separable Gaussian convolution on GPU. A weight multiplies the valid mask, as on the CPU
            # (nanconvolve2d_gaussian_numpy): a NaN weight counts as 0, and the output is NaN where the weighted
            # denominator is under 1e-10 (without a weight, where it is not above the threshold)
            def gpu_gaussian(t_data, sig, t_weight=None):
                ky = make_kernel_1d(sig[0], truncate).view(1, 1, -1, 1)
                kx = make_kernel_1d(sig[1], truncate).view(1, 1, 1, -1)
                pad_y, pad_x = ky.shape[2] // 2, kx.shape[3] // 2
                # Valid mask and weighted convolution
                valid = (~torch.isnan(t_data)).float()
                if t_weight is not None:
                    valid = valid * t_weight.nan_to_num(0.0)
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
                keep = denom > (threshold if t_weight is None else 1e-10)
                return torch.where(keep, numer / (denom + 1e-10), torch.nan)

            # Convert to GPU tensors
            t_data = torch.from_numpy(data_np.astype(np.float32)).to(dev)
            t_wgt = None if weight_np is None else torch.from_numpy(weight_np.astype(np.float32)).to(dev)

            # 1. AA filter (GPU), weighted as on the CPU; the main filter below works on the decimated result, unweighted
            t_aa = gpu_gaussian(t_data, sigma_aa, t_wgt)
            del t_data, t_wgt

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
            pad_y = original_shape[0] - sy - t_up.shape[2]
            pad_x = original_shape[1] - sx - t_up.shape[3]
            if sy or sx or pad_y > 0 or pad_x > 0:
                t_up = F.pad(t_up, (sx, pad_x, sy, pad_y), mode='replicate')
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
            # Clear GPU cache
            import torch
            if dev.type == 'mps':
                torch.mps.empty_cache()
            elif dev.type == 'cuda':
                torch.cuda.empty_cache()

    if squeeze:
        result = result[np.newaxis, ...]
    return result
