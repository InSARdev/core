# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
"""The joblib (loky) worker pools of the downloaders, the preprocessors and Stack.load: the release of the idle
workers at the end of a parallel step."""


def release_workers():
    """
    Stop the idle workers of joblib's loky pool of this process, now.

    joblib keeps the workers of a Parallel call alive for 300 s for its next call, with the memory they hold; this
    gives that memory back where the parallel step of an entry point is done, or before a step that needs it (the
    dask workers, the chunk process pools). The next Parallel call starts new workers.

    It is process-global: it stops the loky workers of every joblib user of this process, so it must not run while
    another thread, or a Parallel(return_as='generator') that is not exhausted yet, still has tasks to dispatch:
    that Parallel call fails or hangs. It waits for the tasks the workers still run.
    """
    from joblib.externals.loky import get_reusable_executor
    get_reusable_executor(reuse=True).shutdown(wait=True)
