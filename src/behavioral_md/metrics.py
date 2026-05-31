"""Death-pattern and survival metrics.

Death is treated as a dependent variable: when an organism dies, why (starvation
vs. danger), and how mortality changes as learning accumulates across lives.
These summaries are computed from the per-life arrays returned by
``jax_engine.run_lives`` (``survived`` = time-to-death in steps, ``cause_of_death``
= 0 alive / 1 starvation / 2 danger), or from any equivalent arrays.
"""

from __future__ import annotations

import numpy as np

CAUSE_LABELS = {0: "alive", 1: "starvation", 2: "danger"}


def time_to_death(survived: np.ndarray, cause: np.ndarray) -> np.ndarray:
    """Flatten to a 1-D array of death times for organisms that actually died."""
    survived = np.asarray(survived).ravel()
    cause = np.asarray(cause).ravel()
    return survived[cause != 0]


def cause_breakdown(cause: np.ndarray) -> dict[str, float]:
    """Fraction of all (life, organism) outcomes by cause (incl. survivors)."""
    cause = np.asarray(cause).ravel()
    if cause.size == 0:
        return {label: 0.0 for label in CAUSE_LABELS.values()}
    return {label: float(np.mean(cause == code)) for code, label in CAUSE_LABELS.items()}


def survival_curve(survived: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Kaplan-Meier-style fraction still alive vs. step, pooled over all lives.

    Survivors (time-to-death == n_steps) are right-censored at n_steps.
    Returns (steps[0..n_steps], fraction_alive).
    """
    survived = np.asarray(survived).ravel()
    steps = np.arange(n_steps + 1)
    frac = np.array([(survived > t).mean() for t in steps])
    return steps, frac


def mortality_by_life(survived: np.ndarray, cause: np.ndarray) -> dict[str, np.ndarray]:
    """Per-life [n_lives] mortality summaries (does survival improve with learning?).

    Returns death_rate (fraction that died), mean_lifespan (steps; survivors count
    as n_steps), and starvation/danger death rates.
    """
    survived = np.asarray(survived)
    cause = np.asarray(cause)
    return {
        "death_rate": (cause != 0).mean(axis=1),
        "mean_lifespan": survived.mean(axis=1),
        "starvation_rate": (cause == 1).mean(axis=1),
        "danger_rate": (cause == 2).mean(axis=1),
    }
