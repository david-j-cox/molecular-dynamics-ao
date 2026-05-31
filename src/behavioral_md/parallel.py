"""Parallel parameter/population sweeps across CPU cores.

Run many independent simulation cells (organism-lives across a parameter grid,
or a population of agents) concurrently. The worker must be a top-level,
importable function so it is picklable by ProcessPoolExecutor.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any


def run_sweep(
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    cells: Iterable[dict[str, Any]],
    max_workers: int | None = None,
    progress_every: int = 0,
) -> list[dict[str, Any]]:
    """Run ``worker(cell)`` for every cell in parallel; return results in order.

    ``max_workers`` defaults to all available cores. Each result is whatever the
    worker returns.
    """
    cells = list(cells)
    if max_workers is None:
        max_workers = os.cpu_count() or 1

    results: list[dict[str, Any] | None] = [None] * len(cells)
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {ex.submit(worker, cell): i for i, cell in enumerate(cells)}
        for fut in as_completed(future_to_idx):
            i = future_to_idx[fut]
            results[i] = fut.result()
            done += 1
            if progress_every and done % progress_every == 0:
                print(f"  ... {done}/{len(cells)} cells done", flush=True)
    return [r for r in results if r is not None]
