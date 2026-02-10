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

def goldstein_numpy(phase_np, corr_np, psize_y, psize_x, threshold=0.5):
    """
    Goldstein adaptive filter using NumPy with batched FFT.

    Memory usage is controlled by dask chunk size (set via .chunk()).
    Uses scipy.fft for memory-efficient processing. Requires complex64/float32 input.

    Parameters
    ----------
    phase_np : np.ndarray
        2D complex64 numpy array of phase data.
    corr_np : np.ndarray
        2D float32 numpy array of correlation values.
    psize_y : int
        Patch size in y dimension.
    psize_x : int
        Patch size in x dimension.
    threshold : float
        Minimum fraction of valid (non-NaN) pixels required to process a patch.
        Default 0.5 means at least 50% of pixels must be valid.

    Returns
    -------
    np.ndarray
        Filtered complex64 array with same shape as input.
    """
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    from scipy import fft as scipy_fft

    # Enforce input types
    assert phase_np.dtype == np.complex64, f"phase must be complex64, got {phase_np.dtype}"
    assert corr_np.dtype == np.float32, f"corr must be float32, got {corr_np.dtype}"

    ny, nx = phase_np.shape
    step_y, step_x = psize_y // 2, psize_x // 2

    # Save original NaN mask to restore at the end
    nan_mask = np.isnan(phase_np)

    # Pad input to ensure all pixels are covered by at least one patch
    pad_y = psize_y - step_y
    pad_x = psize_x - step_x
    phase_np = np.pad(phase_np, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=np.nan)
    corr_np = np.pad(corr_np, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=np.nan)
    padded_ny, padded_nx = phase_np.shape

    # Create valid mask before filling NaN with zeros
    valid_input = ~np.isnan(phase_np)

    # Create triangular weight matrix
    wx = 1.0 - np.abs(np.arange(psize_x // 2) - (psize_x / 2.0 - 1.0)) / (psize_x / 2.0 - 1.0)
    wy = 1.0 - np.abs(np.arange(psize_y // 2) - (psize_y / 2.0 - 1.0)) / (psize_y / 2.0 - 1.0)
    quadrant = np.outer(wy, wx)
    wgt = np.block([[quadrant, np.flip(quadrant, axis=1)],
                    [np.flip(quadrant, axis=0), np.flip(np.flip(quadrant, axis=0), axis=1)]]).astype(np.float32)
    wgt_sum = wgt.sum()
    del quadrant, wx, wy

    # Fill NaN with 0 (np.pad already created copies)
    np.copyto(phase_np, 0.0, where=~valid_input)
    np.copyto(corr_np, 0.0, where=np.isnan(corr_np))

    # Threshold for valid patches
    min_valid_pixels = int(psize_y * psize_x * threshold)

    # Extract all patches using sliding_window_view
    phase_windows = sliding_window_view(phase_np, (psize_y, psize_x))
    corr_windows = sliding_window_view(corr_np, (psize_y, psize_x))
    valid_windows = sliding_window_view(valid_input, (psize_y, psize_x))

    # Subsample to get patches at step intervals: (n_patches_y, n_patches_x, psize_y, psize_x)
    n_patches_y = (padded_ny - psize_y) // step_y + 1
    n_patches_x = (padded_nx - psize_x) // step_x + 1
    phase_patches = phase_windows[::step_y, ::step_x].reshape(-1, psize_y, psize_x).copy()
    corr_patches = corr_windows[::step_y, ::step_x].reshape(-1, psize_y, psize_x).copy()
    valid_patches = valid_windows[::step_y, ::step_x].reshape(-1, psize_y, psize_x)
    del phase_windows, corr_windows, valid_windows

    # Compute alpha for each patch: alpha = 1 - weighted_corr / wgt_sum
    alpha = np.einsum('ij,kij->k', wgt, corr_patches)
    del corr_patches
    alpha /= -wgt_sum
    alpha += 1.0

    # Valid mask: at least threshold fraction of valid pixels
    valid_mask = valid_patches.sum(axis=(1, 2)) >= min_valid_pixels
    del valid_patches

    # Batched FFT2 on all patches (scipy.fft is memory-efficient)
    fft_patches = scipy_fft.fft2(phase_patches, axes=(1, 2), workers=1, overwrite_x=True)
    del phase_patches

    # Power spectrum filter: magnitude^alpha
    pspec_weight = np.abs(fft_patches, out=np.empty_like(fft_patches, dtype=np.float32))
    np.clip(pspec_weight, 1e-10, None, out=pspec_weight)
    np.power(pspec_weight, alpha[:, np.newaxis, np.newaxis], out=pspec_weight)
    del alpha

    # Batched IFFT2
    fft_patches *= pspec_weight
    del pspec_weight
    filtered = scipy_fft.ifft2(fft_patches, axes=(1, 2), workers=1, overwrite_x=True)
    del fft_patches

    # Apply triangular weight and mask invalid patches
    filtered *= wgt[np.newaxis, :, :]
    filtered[~valid_mask] = 0
    n_patches = filtered.shape[0]
    wgt_expanded = np.zeros((n_patches, psize_y, psize_x), dtype=np.float32)
    np.copyto(wgt_expanded, wgt, where=valid_mask[:, np.newaxis, np.newaxis])
    del valid_mask

    # Accumulate: fold patches back into output
    out = np.zeros((padded_ny, padded_nx), dtype=np.complex64)
    count = np.zeros((padded_ny, padded_nx), dtype=np.float32)
    patch_idx = 0
    for i in range(n_patches_y):
        y_start = i * step_y
        for j in range(n_patches_x):
            x_start = j * step_x
            out[y_start:y_start+psize_y, x_start:x_start+psize_x] += filtered[patch_idx]
            count[y_start:y_start+psize_y, x_start:x_start+psize_x] += wgt_expanded[patch_idx]
            patch_idx += 1

    del filtered, wgt_expanded

    # Mark pixels with no valid contribution as invalid
    insufficient_weight = count < 1e-6

    # Normalize by accumulated weights
    np.clip(count, 1e-10, None, out=count)
    out /= count
    del count

    # Crop to original size
    out = out[:ny, :nx]
    insufficient_weight = insufficient_weight[:ny, :nx]

    # Restore NaN mask (original NaN + insufficient contributions)
    out[nan_mask | insufficient_weight] = np.nan + 1j * np.nan
    assert out.dtype == np.complex64, f"Output must be complex64, got {out.dtype}"
    return out


def goldstein_pytorch(phase_np, corr_np, psize_y, psize_x, device, threshold=0.5):
    """
    Goldstein adaptive filter using PyTorch with vectorized unfold/fold.

    Memory usage is controlled by dask chunk size (set via .chunk()).
    Optimized for GPU (MPS/CUDA). Requires complex64/float32 input.

    Parameters
    ----------
    phase_np : np.ndarray
        2D complex64 numpy array of phase data.
    corr_np : np.ndarray
        2D float32 numpy array of correlation values.
    psize_y : int
        Patch size in y dimension.
    psize_x : int
        Patch size in x dimension.
    device : torch.device
        PyTorch device: 'cuda', 'mps', or 'cpu'.
    threshold : float
        Minimum fraction of valid (non-NaN) pixels required to process a patch.
        Default 0.5 means at least 50% of pixels must be valid.

    Returns
    -------
    np.ndarray
        Filtered complex64 array with same shape as input.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    # Enforce input types
    assert phase_np.dtype == np.complex64, f"phase must be complex64, got {phase_np.dtype}"
    assert corr_np.dtype == np.float32, f"corr must be float32, got {corr_np.dtype}"

    ny, nx = phase_np.shape
    step_y, step_x = psize_y // 2, psize_x // 2

    # Save original NaN mask to restore at the end
    nan_mask = np.isnan(phase_np)

    # Create triangular weight matrix
    wx = 1.0 - np.abs(np.arange(psize_x // 2) - (psize_x / 2.0 - 1.0)) / (psize_x / 2.0 - 1.0)
    wy = 1.0 - np.abs(np.arange(psize_y // 2) - (psize_y / 2.0 - 1.0)) / (psize_y / 2.0 - 1.0)
    quadrant = np.outer(wy, wx)
    wgt_np = np.block([[quadrant, np.flip(quadrant, axis=1)],
                       [np.flip(quadrant, axis=0), np.flip(np.flip(quadrant, axis=0), axis=1)]]).astype(np.float32)
    wgt = torch.from_numpy(wgt_np).to(device)
    wgt_sum = wgt.sum()

    # Create valid mask before filling NaN
    valid_input = ~nan_mask

    # Threshold for valid patches
    min_valid_pixels = int(psize_y * psize_x * threshold)

    with torch.no_grad():
        # Pad to ensure all pixels are covered and dimensions work with unfold/fold
        pad_y = psize_y - step_y
        pad_x = psize_x - step_x
        # Additional padding to make dimensions divisible by step
        extra_pad_h = (step_y - ((ny + pad_y - psize_y) % step_y)) % step_y
        extra_pad_w = (step_x - ((nx + pad_x - psize_x) % step_x)) % step_x
        total_pad_y = pad_y + extra_pad_h
        total_pad_x = pad_x + extra_pad_w

        # Pad with 0 (creates copies)
        phase_np = np.pad(phase_np, ((0, total_pad_y), (0, total_pad_x)), mode='constant', constant_values=0)
        corr_np = np.pad(corr_np, ((0, total_pad_y), (0, total_pad_x)), mode='constant', constant_values=0)
        valid_input = np.pad(valid_input, ((0, total_pad_y), (0, total_pad_x)), mode='constant', constant_values=False)

        # Fill NaN with 0 (np.pad already created copies)
        np.copyto(phase_np, 0.0, where=~valid_input)
        np.copyto(corr_np, 0.0, where=np.isnan(corr_np))

        padded_h, padded_w = phase_np.shape

        # Convert to torch tensors - split complex into 2 channels: (1, 2, H, W)
        phase_t = torch.from_numpy(
            np.stack([phase_np.real, phase_np.imag], axis=0)
        ).unsqueeze(0).float().to(device)
        corr_t = torch.from_numpy(corr_np).unsqueeze(0).unsqueeze(0).to(device)
        valid_t = torch.from_numpy(valid_input.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        del phase_np, corr_np, valid_input

        # Use unfold to extract all patches at once
        phase_patches = F.unfold(phase_t, kernel_size=(psize_y, psize_x), stride=(step_y, step_x))
        corr_patches = F.unfold(corr_t, kernel_size=(psize_y, psize_x), stride=(step_y, step_x))
        valid_patches = F.unfold(valid_t, kernel_size=(psize_y, psize_x), stride=(step_y, step_x))
        del phase_t, corr_t, valid_t

        num_patches = phase_patches.shape[2]

        # Reshape phase: (1, 2*psize*psize, L) -> (L, 2, psize, psize) -> complex (L, psize, psize)
        phase_patches = phase_patches.squeeze(0).view(2, psize_y, psize_x, num_patches).permute(3, 0, 1, 2)
        phase_patches = torch.complex(phase_patches[:, 0], phase_patches[:, 1])

        # Reshape corr: (1, psize*psize, L) -> (L, psize, psize)
        corr_patches = corr_patches.squeeze(0).view(psize_y, psize_x, num_patches).permute(2, 0, 1)

        # Reshape valid: (1, psize*psize, L) -> (L, psize, psize)
        valid_patches = valid_patches.squeeze(0).view(psize_y, psize_x, num_patches).permute(2, 0, 1)

        # Compute alpha for each patch
        weighted_corr = (wgt.unsqueeze(0) * corr_patches).sum(dim=(1, 2))
        alpha = 1.0 - weighted_corr / wgt_sum
        del corr_patches, weighted_corr

        # Check valid patches using threshold
        valid_count = valid_patches.sum(dim=(1, 2))
        valid_mask = valid_count >= min_valid_pixels
        del valid_count, valid_patches

        # Batched FFT and power spectrum filtering
        fft_patches = torch.fft.fft2(phase_patches)
        del phase_patches
        magnitude = torch.abs(fft_patches).clamp(min=1e-10)
        pspec_weight = torch.pow(magnitude, alpha.unsqueeze(1).unsqueeze(2))
        del magnitude, alpha
        fft_filtered = pspec_weight * fft_patches
        del pspec_weight, fft_patches

        # Batched IFFT
        filtered = torch.fft.ifft2(fft_filtered)
        del fft_filtered

        # Apply weight and mask invalid patches
        weighted = wgt.unsqueeze(0) * filtered
        del filtered
        weighted = torch.where(valid_mask.unsqueeze(1).unsqueeze(2), weighted, torch.zeros_like(weighted))

        # Prepare for fold: (1, 2*psize*psize, num_patches)
        weighted_ri = torch.stack([weighted.real, weighted.imag], dim=1)
        del weighted
        weighted_ri = weighted_ri.permute(1, 2, 3, 0).reshape(1, 2 * psize_y * psize_x, num_patches)

        # Fold valid_mask for counting overlaps
        valid_for_fold = valid_mask.float().view(1, 1, num_patches).expand(1, psize_y * psize_x, num_patches)
        del valid_mask

        # Fold back - accumulates overlapping patches
        result_ri = F.fold(weighted_ri, output_size=(padded_h, padded_w),
                           kernel_size=(psize_y, psize_x), stride=(step_y, step_x))
        count = F.fold(valid_for_fold, output_size=(padded_h, padded_w),
                       kernel_size=(psize_y, psize_x), stride=(step_y, step_x))
        del weighted_ri, valid_for_fold

        # Mark pixels with no valid contribution
        insufficient_weight = (count.squeeze(0).squeeze(0) < 1e-6).cpu().numpy()

        # Normalize
        result_ri = result_ri / count.clamp(min=1)
        result_ri = result_ri.squeeze(0)  # (2, H, W)
        del count

        # Crop to original size and combine real/imag
        result = torch.complex(result_ri[0, :ny, :nx], result_ri[1, :ny, :nx])
        result = result.cpu().numpy()
        insufficient_weight = insufficient_weight[:ny, :nx]
        del result_ri

    # Restore NaN mask (original NaN + insufficient contributions)
    result[nan_mask | insufficient_weight] = np.nan + 1j * np.nan

    # Clear GPU cache
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    del wgt

    assert result.dtype == np.complex64, f"Output must be complex64, got {result.dtype}"
    return result
