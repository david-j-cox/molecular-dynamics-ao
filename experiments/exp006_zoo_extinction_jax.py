"""Experiment 006 -- phenomena zoo: extinction on the JAX engine.

Each organism lives N_TRAIN reinforced lives at a fixed layout (the food-approach
association forms), then N_EXT lives where food is present but non-reinforcing.
The Rescorla-Wagner omission decay shrinks the learned food weight, and the
behavioral measure (steps in contact with food) falls. Same protocol as
scripts/run_extinction_demo.py, but vectorized for thousands of organisms.

Run:  python -m experiments.exp006_zoo_extinction_jax
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from behavioral_md.config import SimulationConfig
from behavioral_md.experiment_utils import make_cue_centers
from behavioral_md.jax_engine import build_spec, initial_state, make_simulate, run_lives
from behavioral_md.visualization import plot_extinction

LAYOUT = {"position": [4, 4], "food": [4, 8], "danger": [9, 0], "light": [0, 9], "cue": [8, 4]}
N_ORG, N_TRAIN, N_EXT, N_STEPS = 2000, 20, 20, 300


def main() -> None:
    cfg = SimulationConfig(grid_size=10, sensor_range=8.0)
    spec = build_spec(config=cfg)
    sources_np = np.stack([LAYOUT[k] for k in ("food", "danger", "light", "cue")], dtype=float)
    sources = jnp.broadcast_to(jnp.asarray(sources_np), (N_ORG, 4, 2))
    sim = make_simulate(spec, cfg, sources, make_cue_centers(cfg))
    state0 = initial_state(spec, cfg, N_ORG, LAYOUT["position"], cfg.n_cue_receptors)

    schedule = [True] * N_TRAIN + [False] * N_EXT  # reinforce, then extinguish
    t0 = time.perf_counter()
    m = run_lives(sim, spec, cfg, state0, LAYOUT["position"], N_TRAIN + N_EXT, N_STEPS,
                  jax.random.key(0), food_reinforces=schedule)
    elapsed = time.perf_counter() - t0

    hw, contact = m["hw_food"], m["contact"]  # [n_lives, n_org]
    print(f"{N_ORG} organisms x {N_TRAIN+N_EXT} lives in {elapsed:.2f}s")
    print(f"hw_food  end-train (life {N_TRAIN-1}): {hw[N_TRAIN-1].mean():.2f}  "
          f"-> end-extinction: {hw[-1].mean():.2f}")
    print(f"steps at food  last train: {contact[N_TRAIN-1].mean():.0f}  "
          f"first ext: {contact[N_TRAIN].mean():.0f}  last ext: {contact[-1].mean():.0f}")

    rows = [{"seed": o, "episode": life, "hw_food": float(hw[life, o]),
             "steps_at_food": int(contact[life, o])}
            for life in range(N_TRAIN + N_EXT) for o in range(N_ORG)]
    out = Path("outputs/figures/zoo_extinction_jax.png")
    plot_extinction(pd.DataFrame(rows), transition=N_TRAIN, path=out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
