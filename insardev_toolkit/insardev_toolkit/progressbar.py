# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------

import math
import warnings
import tqdm.auto as _tqdm_auto
import tqdm.std as _tqdm_std
from tqdm.std import tqdm as _tqdm

class tqdm(_tqdm):
    """tqdm subclass that floors the percentage instead of rounding.

    Prevents showing 100% when not all items are complete (e.g. 3536/3537).
    """
    @staticmethod
    def format_meter(n, total, elapsed, **kwargs):
        s = _tqdm.format_meter(n, total, elapsed, **kwargs)
        if total and total > 0 and n < total:
            rounded_pct = int(round(n / total * 100))
            floored_pct = int(math.floor(n / total * 100))
            if rounded_pct != floored_pct:
                s = s.replace(f'{rounded_pct:3d}%|', f'{floored_pct:3d}%|', 1)
        return s

# Patch tqdm.auto to use text mode with floor percentage globally
_tqdm_auto.tqdm = tqdm
_tqdm_std.tqdm = tqdm

from distributed.diagnostics.progressbar import ProgressBar
from distributed.utils import LoopRunner
from tornado.ioloop import IOLoop

class TqdmDaskProgress(ProgressBar):
    """
    Class to create a progress bar for Dask computation using tqdm.

    Inherits from Dask's ProgressBar class.
    """
    __loop = None

    def __init__(
        self,
        keys,
        scheduler=None,
        interval="100ms",
        loop=None,
        complete=True,
        start=True,
        **tqdm_kwargs
    ):
        """
        Initialize the progress bar.

        Parameters
        ----------
        keys : iterable
            Iterable of keys that uniquely identify the tasks for which to track progress.
        scheduler : Scheduler, optional
            Dask scheduler instance. Default is None.
        interval : str, optional
            Time interval for updates. Default is "100ms".
        loop : IOLoop, optional
            Tornado IOLoop to use for updates. Default is None.
        complete : bool, optional
            If True, show completed tasks. Default is True.
        start : bool, optional
            If True, start listening for updates immediately. Default is True.
        **tqdm_kwargs : dict
            Additional keyword arguments for tqdm.
        """
        self._loop_runner = loop_runner = LoopRunner(loop=loop)
        super().__init__(keys, scheduler, interval, complete)
        self.tqdm = tqdm(total=9e6,**tqdm_kwargs)

        if start:
            loop_runner.run_sync(self.listen)

    @property
    def loop(self):
        """
        Get the current Tornado IOLoop.

        Returns
        -------
        IOLoop
            The current Tornado IOLoop.
        """
        loop = self.__loop
        if loop is None:
            # If the loop is not running when this is called, the LoopRunner.loop
            # property will raise a DeprecationWarning
            # However subsequent calls might occur - eg atexit, where a stopped
            # loop is still acceptable - so we cache access to the loop.
            self.__loop = loop = self._loop_runner.loop
        return loop

    @loop.setter
    def loop(self, value):
        """
        Set the current Tornado IOLoop.

        Parameters
        ----------
        value : IOLoop
            Tornado IOLoop instance.
        """
        warnings.warn(
            "setting the loop property is deprecated", DeprecationWarning, stacklevel=2
        )
        self.__loop = value

    def _draw_bar(self, remaining, all, **kwargs):
        """
        Update the progress bar.

        Parameters
        ----------
        remaining : int
            Number of tasks remaining.
        all : int
            Total number of tasks.
        **kwargs : dict
            Additional keyword arguments.
        """
        if self.tqdm.total == 9e6:
            self.tqdm.reset(total=all)
        update_ct = (all - remaining) - self.tqdm.n
        self.tqdm.update(update_ct)

    def _draw_stop(self, **kwargs):
        """
        Close the progress bar.

        **kwargs : dict
            Additional keyword arguments.
        """
        self.tqdm.close()

def _scheduler_knows(keys, dask_scheduler=None):
    return all(k in dask_scheduler.tasks for k in keys)

def _drop_refused(client, futures):
    """
    Detach the futures of a graph the scheduler refused from the client, before raising.

    The raised error holds these futures through its traceback (and a cancelled future's
    own CancelledError, which the client keeps, holds them through the traceback raising it
    gave it). A future state left in client.futures is shared by the next persist of the
    same keys -- the recompute from the source the error asks for -- which then starts as
    'cancelled' and was refused falsely before the scheduler had even seen the new graph.
    The scheduler has already released the refused keys, so nothing is sent to it; the
    futures are released, so a kept traceback holds no cluster data either.
    """
    for f in futures:
        client.futures.pop(f.key, None)
        f.release()

def _refusal_reason(future):
    """The text of a cancelled future's CancelledError, raised and dropped here so that
    it keeps no traceback (and so no frame or future) alive."""
    reason = 'cancelled'
    try:
        future.result()
    except Exception as e:
        reason = str(e)
        e.__traceback__ = None
    return reason

def progressbar(futures, **kwargs):
    """
    Helper function to track progress of Dask computation using tqdm.

    The bar waits until the scheduler has the keys it tracks. A graph that needs data
    the cluster no longer holds is refused ("lost dependencies"), its keys never appear,
    and the bar would wait forever. So first wait until the scheduler has all the keys
    or the futures report the refusal. The refused futures are detached from the client
    before the error is raised (_drop_refused), so the recompute from the source works
    on the first attempt.

    Parameters
    ----------
    futures : list
        List of Dask futures.
    **kwargs : dict
        Additional keyword arguments for TqdmDaskProgress.

    Raises
    ------
    RuntimeError
        The graph needs data the cluster no longer holds.
    """
    import time
    from distributed.client import futures_of
    futures = futures_of(futures)
    if not isinstance(futures, (set, list)):
        futures = [futures]
    if futures:
        client = next(iter(futures)).client
        keys = [f.key for f in futures]
        delay = 0.05
        while not client.run_on_scheduler(_scheduler_knows, keys):
            refused = next((f for f in futures if f.status in ('error', 'cancelled')), None)
            if refused is not None:
                _drop_refused(client, futures)
                if refused.status == 'error':
                    refused.result()
                reason = _refusal_reason(refused)
                raise RuntimeError(f'Input data is no longer on the cluster ({reason}). '
                                   'Recompute it from its source.')
            time.sleep(delay)
            delay = min(2 * delay, 1.0)
    TqdmDaskProgress(futures, **kwargs)
