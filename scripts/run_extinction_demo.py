"""Extinction demo: a trained approach weakens once food stops reinforcing.

Each agent first lives N_TRAIN reinforced lives at a fixed layout (so
approach_food's food-channel weight grows), then N_EXT lives where the food is
still present and visible but yields no energy (food_reinforces=False). With the
Rescorla-Wagner rule (lambda=0 on non-reinforced exposure), the learned weight
decays and food-directed behavior falls off.

Two measures across lives (population, 95% CI): the learned food weight
(approach_food.hw_food) and a behavioral measure, steps spent in contact with
food. A population is run in parallel.

Run:  python scripts/run_extinction_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from behavioral_md.config import SimulationConfig
from behavioral_md.environments import BehavioralFieldEnv
from behavioral_md.organism import Organism
from behavioral_md.parallel import run_sweep
from behavioral_md.visualization import plot_extinction

FIG_DIR = Path("outputs/figures")
N_AGENTS = 100
N_TRAIN = 20
N_EXT = 20
STEPS = 300
LAYOUT = {"position": [4, 4], "food": [4, 8], "danger": [9, 0], "light": [0, 9], "cue": [8, 4]}


def _life(env, org, cfg, ep, reinforce, seed):
    """Run one life manually; return (end hw_food, steps in contact with food)."""
    obs, info = env.reset(seed=seed, options={"layout": LAYOUT, "food_reinforces": reinforce})
    org.reset(obs)
    afi = org.index["approach_food"]
    steps_at_food = 0
    for _ in range(cfg.max_steps):
        org.step(obs)
        action = org.emit_action()
        obs, _r, term, trunc, info = env.step(action)
        org.update_history(obs, action, info)
        if float(obs["food_contact"][0]) > 0.0:
            steps_at_food += 1
        if (not org.alive) or term or trunc:
            break
    return org.atoms[afi].history_weights["food"], steps_at_food


def agent_worker(cell: dict[str, Any]) -> dict[str, Any]:
    seed = cell["seed"]
    cfg = SimulationConfig(max_steps=STEPS, seed=seed, sensor_range=8.0)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg)
    rows = []
    for ep in range(N_TRAIN + N_EXT):
        reinforce = ep < N_TRAIN
        hw, at_food = _life(env, org, cfg, ep, reinforce, seed * 1000 + ep)
        rows.append({"seed": seed, "episode": ep, "hw_food": hw, "steps_at_food": at_food})
    return {"rows": rows}


def main() -> None:
    print(f"Extinction demo: {N_AGENTS} agents x ({N_TRAIN} train + {N_EXT} extinction) lives...")
    results = run_sweep(agent_worker, [{"seed": s} for s in range(N_AGENTS)], progress_every=25)
    df = pd.DataFrame([r for res in results for r in res["rows"]])

    plot_extinction(df, transition=N_TRAIN, path=FIG_DIR / "extinction.png")

    g = df.groupby("episode")
    hw = g["hw_food"].mean()
    saf = g["steps_at_food"].mean()
    print(f"hw_food   end-of-training (life {N_TRAIN-1}): {hw.iloc[N_TRAIN-1]:.2f}  "
          f"-> end-of-extinction (life {N_TRAIN+N_EXT-1}): {hw.iloc[-1]:.2f}")
    print(f"steps at food  last train life: {saf.iloc[N_TRAIN-1]:.0f}  "
          f"first ext life: {saf.iloc[N_TRAIN]:.0f}  last ext life: {saf.iloc[-1]:.0f}")
    print(f"Wrote {FIG_DIR/'extinction.png'}")


if __name__ == "__main__":
    main()
