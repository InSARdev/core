# ----------------------------------------------------------------------------
# insardev_toolkit
#
# This file is part of the InSARdev project: https://github.com/AlexeyPechnikov/InSARdev
#
# Copyright (c) 2026, Alexey Pechnikov
#
# See the LICENSE file in the insardev_toolkit directory for license terms.
# ----------------------------------------------------------------------------
"""The chunk rule of the geocoded Zarr stores, shared by the insardev_pygmtsar and insardev_ecef preprocessors: S1 and
NISAR stores of both get the same stored chunks, and the split of a NISAR output chunk into blocks by the radar box of
their pixels (the SLC and topo reads of the chunk workers)."""


def zarr_chunk_shape(shape, budget_mb=32, element_bytes=8):
    """Compute 2D chunk shape by halving the largest dimension until budget is met.

    Uses a fixed element_bytes (default 8, matching complex64) so that all
    variables on the same grid — regardless of actual dtype — share identical
    chunk boundaries.
    """
    budget = budget_mb * 1024 * 1024
    chunks = list(shape)
    while chunks[0] * chunks[1] * element_bytes > budget:
        i = 0 if chunks[0] >= chunks[1] else 1
        chunks[i] = (chunks[i] + 1) // 2
    return tuple(chunks)


def grid_chunks(shape):
    """The (storage, processing) chunk sizes of a 2D grid: the storage chunks of zarr_chunk_shape(), at most 32 MiB of
    complex64 and about equal, and a processing chunk of 2 x 2 of them (1 x 2, 2 x 1 or 1 x 1 at the borders), so
    every processing chunk writes whole storage chunks."""
    storage = zarr_chunk_shape(shape)
    return storage, tuple(min(2 * s, n) for s, n in zip(storage, shape))


def radar_boxes(azi, rng, mask, n_azi, n_rng, budget, tile=64):
    """Split an output chunk into blocks whose radar-coordinate bounding box holds at most budget cells.

    azi, rng are the radar coordinates of the chunk's pixels (cell k of the radar grid is centred on coordinate k, the
    0-based line and bin), mask the pixels to cover. The extents are taken per tile x tile pixels once; a block of
    tiles is halved along the output axis that makes its larger half's box smaller, until the box fits.
    Returns (y0, y1, x0, x1, a0, a1, r0, r1) per block: the output pixels and the radar cells for linear
    interpolation at them.
    """
    import numpy as np

    n_y, n_x = azi.shape
    ty, tx = -(-n_y // tile), -(-n_x // tile)

    def reduce(arr, fill, op):
        full = np.full((ty * tile, tx * tile), fill, dtype=np.float32)
        full[:n_y, :n_x] = np.where(mask, arr, fill)
        return op(full.reshape(ty, tile, tx, tile), axis=(1, 3))

    a_lo, a_hi = reduce(azi, np.inf, np.min), reduce(azi, -np.inf, np.max)
    r_lo, r_hi = reduce(rng, np.inf, np.min), reduce(rng, -np.inf, np.max)

    def box(t):
        y0, y1, x0, x1 = t
        amin = a_lo[y0:y1, x0:x1].min()
        if not np.isfinite(amin):
            return None
        amax, rmin, rmax = a_hi[y0:y1, x0:x1].max(), r_lo[y0:y1, x0:x1].min(), r_hi[y0:y1, x0:x1].max()
        return (int(np.floor(amin)), min(n_azi, int(np.floor(amax)) + 2),
                int(np.floor(rmin)), min(n_rng, int(np.floor(rmax)) + 2))

    def cells(b):
        return 0 if b is None else (b[1] - b[0]) * (b[3] - b[2])

    blocks = []
    stack = [((0, ty, 0, tx), box((0, ty, 0, tx)))]
    while stack:
        t, b = stack.pop()
        if b is None:
            continue
        y0, y1, x0, x1 = t
        if cells(b) <= budget or (y1 - y0 < 2 and x1 - x0 < 2):
            blocks.append((y0 * tile, min(y1 * tile, n_y), x0 * tile, min(x1 * tile, n_x)) + b)
            continue
        halves = []
        if y1 - y0 >= 2:
            ym = (y0 + y1) // 2
            halves.append([((y0, ym, x0, x1), box((y0, ym, x0, x1))), ((ym, y1, x0, x1), box((ym, y1, x0, x1)))])
        if x1 - x0 >= 2:
            xm = (x0 + x1) // 2
            halves.append([((y0, y1, x0, xm), box((y0, y1, x0, xm))), ((y0, y1, xm, x1), box((y0, y1, xm, x1)))])
        stack.extend(min(halves, key=lambda h: max(cells(h[0][1]), cells(h[1][1]))))
    return blocks
