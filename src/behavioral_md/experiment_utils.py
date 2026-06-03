"""Shared utilities for the experiment and demo scripts.

Small analysis/convenience helpers that several experiments had each reimplemented:
mean + 95% CI, the generalized-matching-law line fit, cue-center construction, the
off-grid inert-source trick, the weak-innate atom set, and a uniform results-JSON
writer. They live in the installed package (not under experiments/) so every script
can import them with ``from behavioral_md.experiment_utils import ...`` and still run
standalone -- no PYTHONPATH games. NumPy is eager; JAX is imported lazily in the one
helper that needs it, so the NumPy-only experiments don't pull JAX in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from behavioral_md.atoms import default_atom_set

LOG_DIR = Path("outputs/logs")
FIG_DIR = Path("outputs/figures")


def compute_mean_ci(values, conf_z: float = 1.96) -> tuple[float, float]:
    """Mean and 95% CI half-width (normal approx, 1.96*SEM). NaN mean if empty."""
    a = np.asarray(values, float)
    if a.size == 0:
        return float("nan"), 0.0
    ci = conf_z * a.std(ddof=1) / np.sqrt(a.size) if a.size > 1 else 0.0
    return float(a.mean()), float(ci)


def fit_matching_law(x, y) -> tuple[float, float, float]:
    """Fit y = a*x + log_b by least squares; return (a, log_b, r2).

    For the generalized matching law, x = log reinforcement ratio, y = log behavior
    ratio: a is the sensitivity (slope), log_b the bias (intercept), r2 the fit.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    a, log_b = np.polyfit(x, y, 1)
    ss_res = np.sum((y - (a * x + log_b)) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(a), float(log_b), float(r2)


def make_cue_centers(cfg):
    """Cue receptors tiling the scalar cue dimension v in [0, 1] (JAX array)."""
    import jax.numpy as jnp
    return jnp.linspace(0.0, 1.0, cfg.n_cue_receptors)


def make_inert_source(cfg) -> np.ndarray:
    """An off-grid source position so a stimulus channel is effectively absent."""
    return np.array([cfg.grid_size + 50.0, cfg.grid_size + 50.0])


def weak_innate_atoms(food_sensitivity: float):
    """Default atoms with a WEAK innate approach_food sensitivity.

    The controlled acquisition protocol starts food with little behavioral control
    so the learned history weight, not innate sensitivity, drives the acquisition
    curve. Used by the acquisition demo and its JAX twin.
    """
    atoms = default_atom_set()
    for a in atoms:
        if a.name == "approach_food":
            a.sensitivity["food"] = food_sensitivity
    return atoms


def save_results_json(filename: str, results: dict) -> Path:
    """Write ``results`` as indented JSON under outputs/logs/; return the path."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / filename
    path.write_text(json.dumps(results, indent=2))
    return path
