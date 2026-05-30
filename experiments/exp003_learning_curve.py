"""Experiment 003 -- does learning tighten the approach to food? (temporal weighting)

Hypothesis (lab notebook): the positive energy consequence arrives when the
organism is AT food, where the movement atoms have ~zero activation, so the
eligibility trace must be slow enough (temporal weighting) for the *approach
trajectory* to still be credited. We measure the across-lives learning curve
(food-contact rate, survival, and the movement atoms' learned food weight) while
sweeping ``eligibility_decay``.

Each (decay, seed) cell is one persistent organism living a sequence of lives
(history carries over across episodes; energy/atoms reset each life). Cells run
in parallel across cores.

Run:  python -m experiments.exp003_learning_curve
Saves: outputs/logs/exp003_results.json
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from behavioral_md.config import SimulationConfig
from behavioral_md.environments import BehavioralFieldEnv
from behavioral_md.organism import Organism
from behavioral_md.simulation import run_episode
from experiments._parallel import run_sweep

N_EPISODES = 60
STEPS = 1000


def _drive_food_weight(org: Organism) -> float:
    """Food weight of the approach_food DRIVE atom (the two-tier learner)."""
    return float(org.atoms[org.index["approach_food"]].history_weights["food"])


def worker(cell: dict[str, Any]) -> dict[str, Any]:
    """One persistent organism living N_EPISODES lives; return per-episode curve."""
    cfg = SimulationConfig(
        max_steps=STEPS,
        seed=cell["seed"],
        eligibility_decay=cell["decay"],
        sensor_range=4.0,
        n_episodes=N_EPISODES,
    )
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg)  # persists across episodes -> history accumulates

    contact_rate, survived, hw_food = [], [], []
    for ep in range(N_EPISODES):
        # New random layout each life (reproducible); tests general approach
        # learning rather than memorizing one food location.
        r = run_episode(env, org, cfg, ep, None, reset_options=None, seed=cell["seed"] * 1000 + ep)
        contact_rate.append(r["n_consumed"] / r["steps"])
        survived.append(r["steps"])
        hw_food.append(_drive_food_weight(org))
    return {
        "decay": cell["decay"],
        "seed": cell["seed"],
        "contact_rate": contact_rate,
        "survived": survived,
        "hw_food": hw_food,
    }


def main() -> None:
    decays = [0.90, 0.95, 0.99]
    seeds = list(range(24))
    cells = [{"decay": d, "seed": s} for d, s in itertools.product(decays, seeds)]
    print(f"Exp 003: {len(cells)} organisms x {N_EPISODES} lives x {STEPS} steps ...")

    results = run_sweep(worker, cells, progress_every=12)

    # Aggregate per decay: mean curve across seeds.
    out: dict[str, Any] = {"n_episodes": N_EPISODES, "steps": STEPS, "by_decay": {}}
    print(f"\n{'decay':>6} | {'contact_rate':^17} | {'survived(steps)':^17} | {'hw_food(drive)':^17}")
    print(f"{'':>6} | {'early':>7} {'late':>7} | {'early':>7} {'late':>7} | {'early':>7} {'late':>7}")
    for d in decays:
        rows = [r for r in results if r["decay"] == d]
        cr = np.array([r["contact_rate"] for r in rows])   # (seeds, episodes)
        sv = np.array([r["survived"] for r in rows])
        hw = np.array([r["hw_food"] for r in rows])
        early = slice(0, 5)
        late = slice(N_EPISODES - 5, N_EPISODES)
        rec = {
            "contact_rate_mean": cr.mean(0).tolist(),
            "survived_mean": sv.mean(0).tolist(),
            "hw_food_mean": hw.mean(0).tolist(),
            "contact_early": float(cr[:, early].mean()),
            "contact_late": float(cr[:, late].mean()),
            "survived_early": float(sv[:, early].mean()),
            "survived_late": float(sv[:, late].mean()),
            "hw_early": float(hw[:, early].mean()),
            "hw_late": float(hw[:, late].mean()),
        }
        out["by_decay"][str(d)] = rec
        print(
            f"{d:6} | {rec['contact_early']:7.3f} {rec['contact_late']:7.3f} | "
            f"{rec['survived_early']:7.0f} {rec['survived_late']:7.0f} | "
            f"{rec['hw_early']:7.3f} {rec['hw_late']:7.3f}"
        )

    path = Path("outputs/logs/exp003_results.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {path}")
    print("(early = mean of first 5 lives, late = mean of last 5 lives)")


if __name__ == "__main__":
    main()
