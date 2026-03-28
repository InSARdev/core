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
__version__ = '2026.3.21.post2'

# processing functions
from .Stack import Stack
from .BatchCore import BatchCore
from .Batch import Batch, BatchWrap, BatchUnit, BatchComplex, Batches
from .Baseline import Baseline

# Auto-load extensions if available
try:
    import insardev_polsar
except ImportError:
    pass

try:
    import insardev_backscatter
except ImportError:
    pass
