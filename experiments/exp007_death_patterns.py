"""Experiment 007 -- death patterns: when and why organisms die, across lives.

Death is a dependent variable. Using the JAX engine over a population across
lives (learning carries over), this captures: the survival curve (fraction alive
vs. time within a life), the time-to-death distribution, the cause-of-death
breakdown (starvation vs. danger), and how mortality changes as the food-approach
association is learned (does survival improve across lives?).

A randomly placed danger near the start makes danger-deaths non-trivial.

Run:  python -m experiments.exp007_death_patterns
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from behavioral_md.config import SimulationConfig
from behavioral_md.experiment_utils import make_cue_centers
from behavioral_md.jax_engine import build_spec, initial_state, make_simulate, run_lives
from behavioral_md.metrics import (
    CAUSE_LABELS,
    cause_breakdown,
    mortality_by_life,
    survival_curve,
    time_to_death,
)
from behavioral_md.visualization import (
    plot_mortality_by_life,
    plot_survival_curve,
    plot_time_to_death,
)

# Food up the x=4 corridor; danger off to the side (a risk near, but not blocking,
# the food approach) so both death causes occur and survival is still possible.
LAYOUT = {"position": [4, 4], "food": [4, 8], "danger": [6, 7], "light": [0, 9], "cue": [8, 4]}
N_ORG, N_LIVES, N_STEPS = 3000, 40, 300


def main() -> None:
    cfg = SimulationConfig(grid_size=10, sensor_range=8.0)
    spec = build_spec(config=cfg)
    sources_np = np.stack([LAYOUT[k] for k in ("food", "danger", "light", "cue")], dtype=float)
    sources = jnp.broadcast_to(jnp.asarray(sources_np), (N_ORG, 4, 2))
    sim = make_simulate(spec, cfg, sources, make_cue_centers(cfg))
    state0 = initial_state(spec, cfg, N_ORG, LAYOUT["position"], cfg.n_cue_receptors)

    t0 = time.perf_counter()
    m = run_lives(sim, spec, cfg, state0, LAYOUT["position"], N_LIVES, N_STEPS,
                  jax.random.key(0))
    elapsed = time.perf_counter() - t0

    survived, cause = m["survived"], m["cause_of_death"]   # [n_lives, n_org]
    breakdown = cause_breakdown(cause)
    ttd = time_to_death(survived, cause)
    steps, frac = survival_curve(survived, N_STEPS)
    mort = mortality_by_life(survived, cause)

    print(f"{N_ORG} organisms x {N_LIVES} lives in {elapsed:.2f}s\n")
    print("Cause-of-death breakdown (all life-outcomes):")
    for label in CAUSE_LABELS.values():
        print(f"  {label:11s}: {breakdown[label]*100:5.1f}%")
    print(f"\nMedian time-to-death (of those that died): {np.median(ttd):.0f} steps")
    print(f"Death rate  early (lives 0-4): {mort['death_rate'][:5].mean():.2f}  "
          f"-> late: {mort['death_rate'][-5:].mean():.2f}")
    print(f"Mean lifespan  early: {mort['mean_lifespan'][:5].mean():.0f}  "
          f"-> late: {mort['mean_lifespan'][-5:].mean():.0f} steps")

    plot_survival_curve(steps, frac, "outputs/figures/death_survival_curve.png")
    plot_time_to_death(ttd, N_STEPS, "outputs/figures/death_time_to_death.png")
    plot_mortality_by_life(mort, "outputs/figures/death_mortality_by_life.png")
    print("\nWrote death_survival_curve / death_time_to_death / death_mortality_by_life .png")


if __name__ == "__main__":
    main()
