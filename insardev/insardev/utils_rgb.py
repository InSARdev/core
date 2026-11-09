# ----------------------------------------------------------------------------
# insardev
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev directory for license terms.
# Professional use requires an active per-seat subscription at: https://patreon.com/pechnikov
"""RGB composites of a co-pol and a cross-pol channel (utils_rgb, group GKK unit U5): the quantile stretch of each
channel, gamma and brightness, NaN as black. Plain arrays only: the batch classes call rgb_composite()."""
import numpy as np


def rgb_composite(copol, xpol, gamma=1.0, brightness=2.0, quantile=None):
    """RGB composite from co-pol and cross-pol arrays.

    Parameters
    ----------
    copol : np.ndarray
        Co-polarization data (HH or VV), shape (..., y, x)
    xpol : np.ndarray
        Cross-polarization data (HV or VH), shape (..., y, x)
    gamma : float
        Gamma correction (>1 brightens dark areas)
    brightness : float
        Linear brightness multiplier
    quantile : list
        Quantile range for normalization, default [0.02, 0.98]

    Returns
    -------
    np.ndarray
        RGB array as float32 [0-1], shape (..., y, x, 3)
    """
    # Normalize each channel to [0, 1] using quantile stretch
    def normalize_channel(data):
        valid = data[np.isfinite(data)]
        if len(valid) == 0:
            return np.zeros_like(data)
        q_vals = quantile if quantile is not None else [0.02, 0.98]
        if np.isscalar(q_vals):
            q_vals = [q_vals, 1 - q_vals] if q_vals < 0.5 else [1 - q_vals, q_vals]
        q = np.nanquantile(valid, q_vals)
        vmin_ch, vmax_ch = q[0], q[-1]
        if vmax_ch <= vmin_ch:
            vmax_ch = vmin_ch + 1e-10
        normalized = (data - vmin_ch) / (vmax_ch - vmin_ch)
        return np.clip(normalized, 0, 1)

    # R=copol, G=xpol, B=copol
    r_norm = normalize_channel(copol)
    g_norm = normalize_channel(xpol)
    b_norm = normalize_channel(copol)

    # Apply gamma correction
    if gamma != 1.0:
        r_norm = np.power(r_norm, 1.0 / gamma)
        g_norm = np.power(g_norm, 1.0 / gamma)
        b_norm = np.power(b_norm, 1.0 / gamma)

    # Handle NaN (set to 0)
    nan_mask = ~np.isfinite(copol) | ~np.isfinite(xpol)
    r_norm = np.where(nan_mask, 0, r_norm)
    g_norm = np.where(nan_mask, 0, g_norm)
    b_norm = np.where(nan_mask, 0, b_norm)

    # Stack to RGB
    rgb_float = np.stack([r_norm, g_norm, b_norm], axis=-1)

    # Apply brightness
    if brightness != 1.0:
        rgb_float = rgb_float * brightness
        rgb_float = np.clip(rgb_float, 0, 1)

    return rgb_float.astype(np.float32)
