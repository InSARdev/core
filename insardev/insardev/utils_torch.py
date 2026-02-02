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

def get_torch_device(device='auto', debug=False):
    """
    Get PyTorch device for GPU-accelerated operations.

    Checks Dask cluster resources:
    - If workers have resources={'gpu': N} where N >= 1 → use GPU
    - Otherwise (default) → CPU for parallel processing

    Parameters
    ----------
    device : str
        Device specification: 'auto', 'cuda', 'mps', or 'cpu'.
        'auto' uses CPU by default, GPU only if Dask has resources={'gpu': 1}.
    debug : bool
        Print debug information.

    Returns
    -------
    torch.device
        PyTorch device object.
    """
    import torch

    if device == 'auto':
        gpu_enabled = False

        try:
            from dask.distributed import get_client
            client = get_client()
            workers = client.scheduler_info().get('workers', {})
            if workers:
                # Only enable GPU if explicitly set to gpu >= 1
                gpu_enabled = any(w.get('resources', {}).get('gpu', 0) >= 1 for w in workers.values())
        except ValueError:
            # No Dask client active - still default to CPU
            pass

        if gpu_enabled:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        else:
            device = 'cpu'

    if debug:
        print(f"DEBUG: using device={device}")

    return torch.device(device)
