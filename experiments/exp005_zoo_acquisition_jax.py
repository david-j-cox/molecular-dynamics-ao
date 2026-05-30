"""Experiment 005 -- phenomena zoo (first entry): acquisition on the JAX engine.

Reproduces the across-lives acquisition curve (latency to food falls as the
food-approach association is learned) using the vectorized JAX engine and its
multi-life wrapper. Same controlled protocol as scripts/run_demo.py (fixed
layout, weak innate food sensitivity), so the curve should match the NumPy demo
(~135 -> ~80 steps) -- but for thousands of organisms in seconds.

Run:  python -m experiments.exp005_zoo_acquisition_jax
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from behavioral_md.atoms import default_atom_set
from behavioral_md.config import SimulationConfig
from behavioral_md.jax_engine import build_spec, initial_state, make_simulate, run_lives
from behavioral_md.visualization import plot_acquisition_latency

LAYOUT = {"position": [4, 4], "food": [4, 8], "danger": [9, 0], "light": [0, 9], "cue": [8, 4]}
INNATE_FOOD = 0.2
N_ORG, N_LIVES, N_STEPS = 2000, 40, 300


def _weak_innate_atoms():
    atoms = default_atom_set()
    for a in atoms:
        if a.name == "approach_food":
            a.sensitivity["food"] = INNATE_FOOD
    return atoms


def main() -> None:
    cfg = SimulationConfig(grid_size=10, sensor_range=8.0)
    spec = build_spec(_weak_innate_atoms(), cfg)
    sources_np = np.stack([LAYOUT[k] for k in ("food", "danger", "light", "cue")], dtype=float)
    sources = jnp.broadcast_to(jnp.asarray(sources_np), (N_ORG, 4, 2))
    sim = make_simulate(
        spec, cfg, sources, jnp.ones(N_ORG, bool),
        jnp.zeros(N_ORG), jnp.linspace(0.0, 1.0, cfg.n_cue_receptors),
    )
    state0 = initial_state(spec, cfg, N_ORG, LAYOUT["position"], cfg.n_cue_receptors)

    t0 = time.perf_counter()
    metrics = run_lives(sim, spec, cfg, state0, LAYOUT["position"], N_LIVES, N_STEPS,
                        jax.random.key(0))
    elapsed = time.perf_counter() - t0

    lat = metrics["latency"]  # [n_lives, n_org]
    early, late = lat[:5].mean(), lat[-5:].mean()
    reach = (lat < N_STEPS).mean(axis=1)
    print(f"{N_ORG} organisms x {N_LIVES} lives x {N_STEPS} steps "
          f"({N_ORG*N_LIVES*N_STEPS:,} agent-steps) in {elapsed:.2f}s")
    print(f"Mean latency  early (lives 0-4): {early:.1f}  ->  late: {late:.1f}")
    print(f"Fraction reaching food  early: {reach[:5].mean():.2f}  late: {reach[-5:].mean():.2f}")

    # Reuse the publication-style plot: long DataFrame (organism=seed, life=episode).
    rows = [{"seed": o, "episode": life, "latency": int(lat[life, o])}
            for life in range(N_LIVES) for o in range(N_ORG)]
    out = Path("outputs/figures/zoo_acquisition_jax.png")
    plot_acquisition_latency(pd.DataFrame(rows), out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
