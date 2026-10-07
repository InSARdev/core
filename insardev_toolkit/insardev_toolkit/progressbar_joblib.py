# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2025, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------

class progressbar_joblib:
    """
    Class to provide a context manager for `joblib` that allows progress reporting via `tqdm`.

    This class is primarily used to facilitate the display of progress information during 
    the execution of parallel tasks with `joblib`.
    """
    import contextlib

    @staticmethod
    @contextlib.contextmanager
    def progressbar_joblib(tqdm_object):
        """
        Context manager to patch `joblib` to report into `tqdm` progress bar given as argument.

        Parameters
        ----------
        tqdm_object : tqdm object
            The `tqdm` progress bar object to be updated.
        """
        import joblib

        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            """
            Subclass of `joblib.parallel.BatchCompletionCallBack` to update the `tqdm` 
            progress bar upon the completion of a batch of tasks.
            """
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def __call__(self, *args, **kwargs):
                """
                Overridden method from `joblib.parallel.BatchCompletionCallBack` to update 
                the `tqdm` progress bar and then call the super class method.
                """
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
            # joblib runs the tasks in this process without the batch callbacks when it has one worker (n_jobs=1
            # or the sequential backend): the bar counts the tasks that are done
            if tqdm_object.total is not None and tqdm_object.n < tqdm_object.total:
                tqdm_object.update(n=tqdm_object.total - tqdm_object.n)
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback
            tqdm_object.close()
