"""Experiment 004 -- JAX vs NumPy engine: speed + behavioral sanity.

Runs one foraging life for a population of organisms on a fixed layout, on both
engines, and reports steps/second and the speedup. Also a behavioral sanity
check that the JAX engine produces sensible dynamics (organisms feed and a
fraction survive), since the per-component equivalence is already validated in
`jax_engine.validate_*`.

Run:  python -m experiments.exp004_jax_benchmark
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from behavioral_md.config import SimulationConfig
from behavioral_md.environments import BehavioralFieldEnv
from behavioral_md.jax_engine import build_spec, initial_state, make_simulate
from behavioral_md.organism import Organism
from behavioral_md.simulation import run_episode

LAYOUT = {"position": [4, 4], "food": [6, 6], "danger": [2, 8], "light": [0, 9], "cue": [8, 3]}
CUE_VALUE = 0.5


def jax_run(n_org: int, n_steps: int, seed: int = 0):
    cfg = SimulationConfig(grid_size=10, sensor_range=8.0, max_steps=n_steps)
    spec = build_spec(config=cfg)
    sources_np = np.stack([LAYOUT[k] for k in ("food", "danger", "light", "cue")], dtype=float)
    sources = jnp.broadcast_to(jnp.asarray(sources_np), (n_org, 4, 2))
    food_reinforces = jnp.ones(n_org, bool)
    cue_value = jnp.full(n_org, CUE_VALUE)
    cue_centers = jnp.linspace(0.0, 1.0, cfg.n_cue_receptors)
    sim = make_simulate(spec, cfg, sources, food_reinforces, cue_value, cue_centers)
    state0 = initial_state(spec, cfg, n_org, LAYOUT["position"], cfg.n_cue_receptors)
    keys = jax.random.split(jax.random.key(seed), n_steps)

    t0 = time.perf_counter()
    final, energy = sim(state0, keys)            # includes JIT compile
    final.energy.block_until_ready()
    t_compile = time.perf_counter() - t0

    t0 = time.perf_counter()
    final, energy = sim(state0, keys)            # steady-state
    final.energy.block_until_ready()
    t_run = time.perf_counter() - t0
    return t_compile, t_run, np.asarray(final.energy), np.asarray(final.alive)


def numpy_run(n_org: int, n_steps: int, seed: int = 0) -> float:
    cfg = SimulationConfig(grid_size=10, sensor_range=8.0, max_steps=n_steps)
    t0 = time.perf_counter()
    for s in range(n_org):
        env = BehavioralFieldEnv(cfg)
        org = Organism(cfg)
        run_episode(env, org, cfg, 0, None, {"layout": LAYOUT, "cue_value": CUE_VALUE}, seed=s)
    return time.perf_counter() - t0


def main() -> None:
    n_org, n_steps = 2000, 600
    print(f"Workload: {n_org} organisms x {n_steps} steps = {n_org*n_steps:,} agent-steps\n")

    t_compile, t_run, energy, alive = jax_run(n_org, n_steps)
    jax_sps = n_org * n_steps / t_run
    print(f"JAX   : compile {t_compile:.2f}s, steady {t_run:.3f}s "
          f"({jax_sps:,.0f} agent-steps/s)")
    print(f"        sanity: {alive.mean()*100:.0f}% alive at end, "
          f"mean final energy {energy.mean():.2f}")

    # NumPy baseline on a smaller population (single-process), then extrapolate.
    n_np = 200
    t_np = numpy_run(n_np, n_steps)
    np_sps = n_np * n_steps / t_np
    print(f"\nNumPy : {n_np} organisms x {n_steps} steps in {t_np:.2f}s "
          f"({np_sps:,.0f} agent-steps/s, single process)")
    print(f"\nSpeedup (steps/s): ~{jax_sps/np_sps:.0f}x")


if __name__ == "__main__":
    main()
