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
__version__ = '2026.10.6'

def _warn_numbagg():
    """numbagg runs parallel numba kernels, and xarray and dask hand reductions to it when it is installed. With numba's
    workqueue threading layer (no TBB or OpenMP layer here) two threads of one dask worker in it at once abort the
    process. Nothing here needs numbagg. It is looked up, never imported, and numba is imported only when it is installed.
    The main process prints, as the insardev_io note does: not every worker. First in this file: it is printed even when
    an import below fails."""
    import importlib
    import importlib.util
    import multiprocessing
    if multiprocessing.current_process().name != 'MainProcess' or importlib.util.find_spec('numbagg') is None:
        return
    for layer in ('tbbpool', 'omppool'):
        try:
            importlib.import_module(f'numba.np.ufunc.{layer}')
            return
        except ImportError:
            pass
    print('WARNING: numbagg is installed and numba has no thread-safe threading layer here: '
          'multi-threaded dask workers can abort. Remove it: pip uninstall numbagg')

_warn_numbagg()
del _warn_numbagg

# Two process-wide dask settings, made once at import. The workers get them too: they import insardev at start, as the
# Dask nanny hands its config to them.
import dask as _dask
# dask arrays culled after their optimization (utils_dask.culled_array_optimize), until the orphan tasks of
# https://github.com/dask/dask/issues/9888 are solved: there is no safe workaround without it.
from .utils_dask import culled_array_optimize as _culled_array_optimize
_dask.config.set(array_optimize=_culled_array_optimize)
# NO p2p RECHUNK, FOR ANY RECHUNK: it writes huge temporaries and takes huge RAM (it crashes even on a 128 GB host),
# and dask picks it by itself under a client when a rechunk graph is large. The method is decided where the rechunk
# graph is built, so this setting covers the rechunks of every function of the library.
_dask.config.set({'array.rechunk.method': 'tasks'})
# Client.restart waits out the lost-worker timers of its previous restart (BUGS N194; distributed deploy/spec.py, the timers of
# _update_worker_status): one wrapper, installed once, see utils_dask.install_safe_restart.
from .utils_dask import install_safe_restart as _install_safe_restart
_install_safe_restart()

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

try:
    import insardev_io
    import multiprocessing
    if multiprocessing.current_process().name == 'MainProcess':
        print('NOTE: insardev_io extension loaded for zarr I/O.')
except ImportError:
    pass
