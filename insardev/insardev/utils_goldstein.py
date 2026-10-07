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
from .utils_torch import serialize_gpu

def goldstein_numpy(phase_np, corr_np, psize_y, psize_x, threshold=0.5, skip=(0, 0)):
    """
    Goldstein adaptive filter using NumPy with batched FFT.

    The patches are filtered in strips of patch rows that fit half of the Dask chunk size
    (dask.config['array.chunk-size']) and accumulated in the same order as one batch of all patches: the same patches,
    the same fold order, so the strip size can only change the rounding of a patch's FFT (about 1 ulp for some patch
    sizes). Every strip pads its own rows, so no padded copy of the block is made. Requires complex64/float32 input.

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
    skip : tuple of int
        (rows, columns) at the start of the block that no patch covers: the patch grid starts at that row and
        column of the block, and those pixels are NaN. Used for a block of a larger array, whose patch grid belongs
        to the whole array (see Batch._apply_goldstein_2d_for_dask). Default (0, 0): the grid starts at the block.

    Returns
    -------
    np.ndarray
        Filtered complex64 array with same shape as input.
    """
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    from scipy import fft as scipy_fft
    from .utils_dask import get_dask_chunk_size_mb

    # Enforce input types
    assert phase_np.dtype == np.complex64, f"phase must be complex64, got {phase_np.dtype}"
    assert corr_np.dtype == np.float32, f"corr must be float32, got {corr_np.dtype}"

    ny, nx = phase_np.shape
    # the patch grid starts at row sy and column sx of the block; the rows and columns before them stay uncovered
    sy, sx = skip
    gny, gnx = ny - sy, nx - sx
    step_y, step_x = psize_y // 2, psize_x // 2

    # The padded grid covers every pixel by at least one patch; the strips below pad their own rows, so no padded
    # copy of the block is made
    pad_y = psize_y - step_y
    pad_x = psize_x - step_x
    padded_ny, padded_nx = gny + pad_y, gnx + pad_x

    # Create triangular weight matrix
    wx = 1.0 - np.abs(np.arange(psize_x // 2) - (psize_x / 2.0 - 1.0)) / (psize_x / 2.0 - 1.0)
    wy = 1.0 - np.abs(np.arange(psize_y // 2) - (psize_y / 2.0 - 1.0)) / (psize_y / 2.0 - 1.0)
    quadrant = np.outer(wy, wx)
    wgt = np.block([[quadrant, np.flip(quadrant, axis=1)],
                    [np.flip(quadrant, axis=0), np.flip(np.flip(quadrant, axis=0), axis=1)]]).astype(np.float32)
    wgt_sum = wgt.sum()
    del quadrant, wx, wy

    # Threshold for valid patches
    min_valid_pixels = int(psize_y * psize_x * threshold)

    # Patches at step intervals: n_patches_y rows of n_patches_x patches
    n_patches_y = (padded_ny - psize_y) // step_y + 1
    n_patches_x = (padded_nx - psize_x) // step_x + 1

    # Patch rows per strip: about 40 bytes per patch pixel (the complex patch, its spectrum and filtered copy, the
    # correlation, power spectrum and weight arrays) within half of the Dask chunk size
    budget = get_dask_chunk_size_mb() * 1024 * 1024 // 2
    rows_per_strip = max(1, budget // (n_patches_x * psize_y * psize_x * 40))

    out = np.zeros((sy + padded_ny, sx + padded_nx), dtype=np.complex64)
    count = np.zeros((sy + padded_ny, sx + padded_nx), dtype=np.float32)
    for row0 in range(0, n_patches_y, rows_per_strip):
        row1 = min(row0 + rows_per_strip, n_patches_y)
        # The strip's rows of the padded grid: NaN padding filled with zeros, as invalid pixels
        y0, y1 = row0 * step_y, (row1 - 1) * step_y + psize_y
        strip_phase = np.zeros((y1 - y0, padded_nx), dtype=np.complex64)
        strip_corr = np.zeros((y1 - y0, padded_nx), dtype=np.float32)
        strip_valid = np.zeros((y1 - y0, padded_nx), dtype=bool)
        # the strip's first row is a row of the block, so the height is positive
        height = min(y1, gny) - y0
        block = phase_np[sy + y0:sy + y0 + height, sx:]
        strip_valid[:height, :gnx] = ~np.isnan(block)
        np.copyto(strip_phase[:height, :gnx], block, where=strip_valid[:height, :gnx])
        block = corr_np[sy + y0:sy + y0 + height, sx:]
        np.copyto(strip_corr[:height, :gnx], block, where=~np.isnan(block))
        del block
        phase_patches = sliding_window_view(strip_phase, (psize_y, psize_x))[::step_y, ::step_x].reshape(-1, psize_y, psize_x).copy()
        corr_patches = sliding_window_view(strip_corr, (psize_y, psize_x))[::step_y, ::step_x].reshape(-1, psize_y, psize_x).copy()
        valid_patches = sliding_window_view(strip_valid, (psize_y, psize_x))[::step_y, ::step_x].reshape(-1, psize_y, psize_x)
        del strip_phase, strip_corr, strip_valid

        # Compute alpha for each patch: alpha = 1 - weighted_corr / wgt_sum
        alpha = np.einsum('ij,kij->k', wgt, corr_patches)
        del corr_patches
        alpha /= -wgt_sum
        alpha += 1.0

        # Valid mask: at least threshold fraction of valid pixels
        valid_mask = valid_patches.sum(axis=(1, 2)) >= min_valid_pixels
        del valid_patches

        # Batched FFT2 on the strip's patches (scipy.fft is memory-efficient)
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
        wgt_expanded = np.zeros((filtered.shape[0], psize_y, psize_x), dtype=np.float32)
        np.copyto(wgt_expanded, wgt, where=valid_mask[:, np.newaxis, np.newaxis])
        del valid_mask

        # Accumulate: fold the strip's patches back into output, in the order of all patches
        patch_idx = 0
        for i in range(row0, row1):
            y_start = i * step_y
            for j in range(n_patches_x):
                x_start = j * step_x
                out[sy+y_start:sy+y_start+psize_y, sx+x_start:sx+x_start+psize_x] += filtered[patch_idx]
                count[sy+y_start:sy+y_start+psize_y, sx+x_start:sx+x_start+psize_x] += wgt_expanded[patch_idx]
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
    out[np.isnan(phase_np) | insufficient_weight] = np.nan + 1j * np.nan
    assert out.dtype == np.complex64, f"Output must be complex64, got {out.dtype}"
    return out


@serialize_gpu
def goldstein_pytorch(phase_np, corr_np, psize_y, psize_x, device, threshold=0.5, skip=(0, 0)):
    """
    Goldstein adaptive filter using PyTorch with vectorized unfold/fold.

    The patches are filtered in strips of patch rows that fit half of the Dask chunk size
    (dask.config['array.chunk-size']), unfolded and folded back a patch row at a time in the order of
    goldstein_numpy, so the strip size does not change a pixel's sum.
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
    skip : tuple of int
        (rows, columns) at the start of the block that no patch covers: the patch grid starts at that row and
        column of the block, and those pixels are NaN. See goldstein_numpy. Default (0, 0).

    Returns
    -------
    np.ndarray
        Filtered complex64 array with same shape as input.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from .utils_dask import get_dask_chunk_size_mb

    # Enforce input types
    assert phase_np.dtype == np.complex64, f"phase must be complex64, got {phase_np.dtype}"
    assert corr_np.dtype == np.float32, f"corr must be float32, got {corr_np.dtype}"

    ny, nx = phase_np.shape
    # the patch grid starts at row sy and column sx of the block; the rows and columns before them stay uncovered
    sy, sx = skip
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
    valid_input = ~nan_mask[sy:, sx:]

    # Threshold for valid patches
    min_valid_pixels = int(psize_y * psize_x * threshold)

    with torch.no_grad():
        # Pad by half a patch: every pixel is covered by a patch, and the patch count is that of goldstein_numpy
        # (unfold and fold take a padded size that the step does not divide, the remainder is not covered). An
        # extra patch row or column would add patches that goldstein_numpy does not have.
        total_pad_y = psize_y - step_y
        total_pad_x = psize_x - step_x

        # Pad with 0 (creates copies)
        phase_np = np.pad(phase_np[sy:, sx:], ((0, total_pad_y), (0, total_pad_x)), mode='constant', constant_values=0)
        corr_np = np.pad(corr_np[sy:, sx:], ((0, total_pad_y), (0, total_pad_x)), mode='constant', constant_values=0)
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

        # Patches at step intervals: n_patches_y rows of n_patches_x patches, as in goldstein_numpy
        n_patches_y = (padded_h - psize_y) // step_y + 1
        n_patches_x = (padded_w - psize_x) // step_x + 1

        # Patch rows per strip: the unfolded patches, their spectra and the filtered copies take about 80 bytes of
        # device memory per patch pixel, within half of the Dask chunk size; unfolding all the patches of the block at
        # once took about 300 bytes per pixel of the block
        budget = get_dask_chunk_size_mb() * 1024 * 1024 // 2
        rows_per_strip = max(1, budget // (n_patches_x * psize_y * psize_x * 80))

        # The sums of the filtered patches and of their weights; the rows and columns before the patch grid (sy, sx)
        # get no patch: no weight, so they are marked below
        result_ri = torch.zeros((1, 2, sy + padded_h, sx + padded_w), device=device)
        count = torch.zeros((1, 1, sy + padded_h, sx + padded_w), device=device)

        for row0 in range(0, n_patches_y, rows_per_strip):
            row1 = min(row0 + rows_per_strip, n_patches_y)
            n_rows = row1 - row0
            # Use unfold to extract all patches of the strip's rows at once
            y0, y1 = row0 * step_y, (row1 - 1) * step_y + psize_y
            phase_patches = F.unfold(phase_t[:, :, y0:y1], kernel_size=(psize_y, psize_x), stride=(step_y, step_x))
            corr_patches = F.unfold(corr_t[:, :, y0:y1], kernel_size=(psize_y, psize_x), stride=(step_y, step_x))
            valid_patches = F.unfold(valid_t[:, :, y0:y1], kernel_size=(psize_y, psize_x), stride=(step_y, step_x))

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

            # Fold the patch weights of the valid patches: the divisor is the SUM OF PATCH WEIGHTS, as in
            # goldstein_numpy (a count of patches scaled the result by the weight sum over the patch count, and left
            # the pixels whose weight sum is 0, the first row and column, finite)
            valid_for_fold = wgt.reshape(1, psize_y * psize_x, 1) * valid_mask.float().view(1, 1, num_patches)
            del valid_mask

            # Fold back - accumulates overlapping patches, ONE PATCH ROW AT A TIME and the rows in order, as
            # goldstein_numpy does: a pixel's sum is then built the same way whatever the strip
            for i in range(n_rows):
                cols = slice(i * n_patches_x, (i + 1) * n_patches_x)
                rows = slice(sy + (row0 + i) * step_y, sy + (row0 + i) * step_y + psize_y)
                result_ri[:, :, rows, sx:] += F.fold(weighted_ri[:, :, cols], output_size=(psize_y, padded_w),
                                                     kernel_size=(psize_y, psize_x), stride=(step_y, step_x))
                count[:, :, rows, sx:] += F.fold(valid_for_fold[:, :, cols], output_size=(psize_y, padded_w),
                                                 kernel_size=(psize_y, psize_x), stride=(step_y, step_x))
            del weighted_ri, valid_for_fold
        del phase_t, corr_t, valid_t

        # Mark pixels with no valid contribution
        insufficient_weight = (count.squeeze(0).squeeze(0) < 1e-6).cpu().numpy()

        # Normalize
        result_ri = result_ri / count.clamp(min=1e-10)
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
