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
"""Complex-phase algebra of the batch classes: intensity, pair products, screens, coherence."""
import xarray as xr


def intensity(x):
    """Signal intensity real**2 + imag**2 (BatchComplex.power)."""
    # Optimized: avoid sqrt in abs() by computing real**2 + imag**2 directly
    return x.real**2 + x.imag**2


def abs_squared(x):
    """|x|**2 as abs(x)**2 (BatchCore.power); a different rounding than intensity(), kept apart."""
    return xr.ufuncs.abs(x)**2


def interferogram(ref, rep, phase=None):
    """ref * conj(rep), times phase when given (Batches.interferogram, Batches.subtract, singlelook)."""
    intf = ref * rep.conj()
    if phase is not None:
        intf = intf * phase
    return intf


def screens(data, screen_ref, screen_rep):
    """data * conj(screen_ref) * screen_rep: the screens of a pair's two dates removed (_subtract_date_from_pair)."""
    return data * screen_ref.conj() * screen_rep


def difference(first, second):
    """first - second, the real subtraction of Batches.subtract and of BatchCore.rmse."""
    return first - second


def coherence(intf, power_ref, power_rep):
    """Correlation clip(|intf| / sqrt(power_ref * power_rep), 0, 1) of Batches (Batches.interferogram)."""
    return (intf.abs() / (power_ref * power_rep).sqrt()).clip(0, 1)


def masked(data, weight):
    """data where weight is finite, NaN elsewhere; None stays None (Batches.interferogram)."""
    return None if data is None else data.where(weight.isfinite())
