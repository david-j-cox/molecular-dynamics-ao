"""Acquisition demo: does latency to food drop across lives?

Controlled protocol (unlike the random-layout figures): the food sits at a
FIXED location every life, the organism starts at a fixed spot, and danger is
kept far off the approach path. Innate food sensitivity is weakened so that the
approach must be *learned* -- if the two-tier learning works, the approach_food
drive strengthens with feeding and latency to first food should fall across lives.

A population of agents runs in parallel (mean +/- 95% CI). The learning history
persists across an agent's lives; energy/position reset each life.

Run:  python scripts/run_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from behavioral_md.atoms import default_atom_set
from behavioral_md.config import SimulationConfig
from behavioral_md.environments import BehavioralFieldEnv
from behavioral_md.organism import Organism
from behavioral_md.parallel import run_sweep
from behavioral_md.simulation import run_episode
from behavioral_md.visualization import plot_acquisition_latency, plot_learning_curve

FIG_DIR = Path("outputs/figures")
N_AGENTS = 100
N_LIVES = 40
STEPS = 300
SENSOR_RANGE = 8.0
INNATE_FOOD = 0.2  # weak innate approach so acquisition has room to show
# Fixed arena: food a moderate distance from a fixed start (reachable while
# untrained, but slowly); danger parked far off the approach path.
LAYOUT = {"position": [4, 4], "food": [4, 8], "danger": [9, 0], "light": [0, 9], "cue": [8, 4]}


def _weak_innate_atoms():
    atoms = default_atom_set()
    for a in atoms:
        if a.name == "approach_food":
            a.sensitivity["food"] = INNATE_FOOD
    return atoms


def agent_worker(cell: dict[str, Any]) -> dict[str, Any]:
    seed = cell["seed"]
    cfg = SimulationConfig(n_episodes=N_LIVES, max_steps=STEPS, seed=seed,
                           sensor_range=SENSOR_RANGE)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg, atoms=_weak_innate_atoms())
    rows = []
    for ep in range(N_LIVES):
        r = run_episode(env, org, cfg, ep, None, {"layout": LAYOUT}, seed=seed * 1000 + ep)
        rows.append({"seed": seed, "episode": ep, "n_consumed": r["n_consumed"],
                     "steps": r["steps"], "latency": r["latency"]})
    return {"summaries": rows}


def main(n_agents: int = N_AGENTS) -> None:
    print(f"Acquisition demo: {n_agents} agents x {N_LIVES} lives (fixed layout)...")
    results = run_sweep(agent_worker, [{"seed": s} for s in range(n_agents)],
                        progress_every=25)
    summaries = pd.DataFrame([r for res in results for r in res["summaries"]])

    plot_acquisition_latency(summaries, FIG_DIR / "acquisition_latency.png")
    plot_learning_curve(summaries, FIG_DIR / "acquisition_curve.png")

    by_life = summaries.groupby("episode")["latency"].mean()
    early = by_life.iloc[:5].mean()
    late = by_life.iloc[-5:].mean()
    reached = (summaries["latency"] < STEPS).groupby(summaries["episode"]).mean()
    print(f"Mean latency  early: {early:.1f}  ->  late: {late:.1f}")
    print(f"Fraction reaching food  early: {reached.iloc[:5].mean():.2f}  "
          f"late: {reached.iloc[-5:].mean():.2f}")
    print(f"Wrote {FIG_DIR/'acquisition_latency.png'} and {FIG_DIR/'acquisition_curve.png'}")


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--agents", type=int, default=N_AGENTS)
    main(_ap.parse_args().agents)
