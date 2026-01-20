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

def goldstein_numpy(phase_np, corr_np, psize_y, psize_x):
    """
    Goldstein adaptive filter using NumPy with loop over patches.

    Memory-efficient implementation for CPU. Uses constant memory regardless
    of image size - only stores input/output arrays, weight matrix, and one
    patch at a time.

    Parameters
    ----------
    phase_np : np.ndarray
        2D complex numpy array of phase data.
    corr_np : np.ndarray
        2D real numpy array of correlation values.
    psize_y : int
        Patch size in y dimension.
    psize_x : int
        Patch size in x dimension.

    Returns
    -------
    np.ndarray
        Filtered complex array with same shape as input.
    """
    import numpy as np

    ny, nx = phase_np.shape
    step_y, step_x = psize_y // 2, psize_x // 2

    # Create triangular weight matrix
    wx = 1.0 - np.abs(np.arange(psize_x // 2) - (psize_x / 2.0 - 1.0)) / (psize_x / 2.0 - 1.0)
    wy = 1.0 - np.abs(np.arange(psize_y // 2) - (psize_y / 2.0 - 1.0)) / (psize_y / 2.0 - 1.0)
    quadrant = np.outer(wy, wx)
    wgt = np.block([[quadrant, np.flip(quadrant, axis=1)],
                    [np.flip(quadrant, axis=0), np.flip(np.flip(quadrant, axis=0), axis=1)]]).astype(np.float32)
    wgt_sum = wgt.sum()

    # Output arrays
    out = np.zeros((ny, nx), dtype=np.complex64)
    count = np.zeros((ny, nx), dtype=np.float32)

    # Clean input (fill NaN with 0)
    phase_clean = np.nan_to_num(phase_np, nan=0.0, posinf=0.0, neginf=0.0, copy=True).astype(np.complex64)
    corr_clean = np.nan_to_num(corr_np, nan=0.0, copy=True).astype(np.float32)

    # Loop over patches (PyGMTSAR style)
    for i in range(0, ny - psize_y + 1, step_y):
        for j in range(0, nx - psize_x + 1, step_x):
            # Extract patch
            phase_patch = phase_clean[i:i+psize_y, j:j+psize_x]
            corr_patch = corr_clean[i:i+psize_y, j:j+psize_x]

            # Skip if mostly empty
            if (phase_patch != 0).sum() < psize_y * psize_x * 0.5:
                continue

            # Compute alpha
            alpha = 1.0 - (wgt * corr_patch).sum() / wgt_sum

            # FFT -> power spectrum filter -> IFFT
            fft_patch = np.fft.fft2(phase_patch)
            magnitude = np.abs(fft_patch)
            magnitude = np.clip(magnitude, 1e-10, None)
            pspec_weight = np.power(magnitude, alpha)
            filtered = np.fft.ifft2(pspec_weight * fft_patch)

            # Accumulate weighted result
            weighted = wgt * filtered
            out[i:i+psize_y, j:j+psize_x] += weighted
            count[i:i+psize_y, j:j+psize_x] += wgt

    # Normalize
    count = np.clip(count, 1e-10, None)
    return (out / count).astype(np.complex64)


def goldstein_pytorch(phase_np, corr_np, psize_y, psize_x, device):
    """
    Goldstein adaptive filter using PyTorch with vectorized unfold/fold.

    Optimized for GPU (MPS/CUDA). Uses batch FFT on all patches at once.

    Parameters
    ----------
    phase_np : np.ndarray
        2D complex numpy array of phase data.
    corr_np : np.ndarray
        2D real numpy array of correlation values.
    psize_y : int
        Patch size in y dimension.
    psize_x : int
        Patch size in x dimension.
    device : torch.device
        PyTorch device: 'cuda', 'mps', or 'cpu'.

    Returns
    -------
    np.ndarray
        Filtered complex array with same shape as input.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    ny, nx = phase_np.shape
    step_y, step_x = psize_y // 2, psize_x // 2

    # Create triangular weight matrix
    wx = 1.0 - np.abs(np.arange(psize_x // 2) - (psize_x / 2.0 - 1.0)) / (psize_x / 2.0 - 1.0)
    wy = 1.0 - np.abs(np.arange(psize_y // 2) - (psize_y / 2.0 - 1.0)) / (psize_y / 2.0 - 1.0)
    quadrant = np.outer(wy, wx)
    wgt_np = np.block([[quadrant, np.flip(quadrant, axis=1)],
                       [np.flip(quadrant, axis=0), np.flip(np.flip(quadrant, axis=0), axis=1)]]).astype(np.float32)
    wgt = torch.from_numpy(wgt_np).to(device)
    wgt_sum = wgt.sum()

    # Prepare data - fill NaN with 0
    phase_clean = np.nan_to_num(phase_np, nan=0.0, posinf=0.0, neginf=0.0, copy=True)
    corr_clean = np.nan_to_num(corr_np, nan=0.0, copy=True)

    with torch.no_grad():
        # Pad to make dimensions work with unfold/fold
        pad_h = (step_y - ((ny - psize_y) % step_y)) % step_y
        pad_w = (step_x - ((nx - psize_x) % step_x)) % step_x

        if pad_h > 0 or pad_w > 0:
            phase_clean = np.pad(phase_clean, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
            corr_clean = np.pad(corr_clean, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

        padded_h, padded_w = phase_clean.shape

        # Convert to torch tensors - split complex into 2 channels: (1, 2, H, W)
        phase_t = torch.from_numpy(
            np.stack([phase_clean.real, phase_clean.imag], axis=0)
        ).unsqueeze(0).float().to(device)
        corr_t = torch.from_numpy(corr_clean).unsqueeze(0).unsqueeze(0).to(device)
        del phase_clean, corr_clean

        # Use unfold to extract all patches at once
        phase_patches = F.unfold(phase_t, kernel_size=(psize_y, psize_x), stride=(step_y, step_x))
        corr_patches = F.unfold(corr_t, kernel_size=(psize_y, psize_x), stride=(step_y, step_x))
        del phase_t, corr_t

        num_patches = phase_patches.shape[2]

        # Reshape phase: (1, 2*psize*psize, L) -> (L, 2, psize, psize) -> complex (L, psize, psize)
        phase_patches = phase_patches.squeeze(0).view(2, psize_y, psize_x, num_patches).permute(3, 0, 1, 2)
        phase_patches = torch.complex(phase_patches[:, 0], phase_patches[:, 1])

        # Reshape corr: (1, psize*psize, L) -> (L, psize, psize)
        corr_patches = corr_patches.squeeze(0).view(psize_y, psize_x, num_patches).permute(2, 0, 1)

        # Compute alpha for each patch
        weighted_corr = (wgt.unsqueeze(0) * corr_patches).sum(dim=(1, 2))
        alpha = 1.0 - weighted_corr / wgt_sum
        del corr_patches, weighted_corr

        # Check valid patches (at least 50% non-zero)
        valid_count = (phase_patches != 0).sum(dim=(1, 2)).float()
        valid_mask = valid_count >= (psize_y * psize_x * 0.5)
        del valid_count

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

        # Normalize
        result_ri = result_ri / count.clamp(min=1)
        result_ri = result_ri.squeeze(0)  # (2, H, W)
        del count

        # Crop to original size and combine real/imag
        result = torch.complex(result_ri[0, :ny, :nx], result_ri[1, :ny, :nx])
        result = result.cpu().numpy()
        del result_ri

    # Clear GPU cache
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    del wgt

    return result
