"""Generalization demo: a trained cue evokes a graded response across cue values.

Each agent is trained with the neutral cue fixed at value v0 and co-located with
food, so the cue->food association forms (its value-tuned receptors near v0 gain
weight). Then a non-spatial probe presents cues across the value range and reads
the evoked conditioned response (no food). The result is a generalization
gradient: response peaked at v0, falling off with cue distance, emerging from the
overlap of the receptor tuning curves.

A population of agents runs in parallel (mean +/- 95% CI).

Run:  python scripts/run_generalization_demo.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from behavioral_md.config import SimulationConfig
from behavioral_md.environments import BehavioralFieldEnv
from behavioral_md.organism import Organism
from behavioral_md.parallel import run_sweep
from behavioral_md.simulation import run_episode
from behavioral_md.visualization import plot_generalization_gradient

FIG_DIR = Path("outputs/figures")
N_AGENTS = 100
N_TRAIN = 30
STEPS = 300
V0 = 0.5  # trained cue value
PROBE_VALUES = np.linspace(0.0, 1.0, 21)
# Cue co-located with food so it is present at reinforcement; danger off-path.
LAYOUT = {"position": [4, 4], "food": [4, 8], "danger": [9, 0], "light": [0, 9], "cue": [4, 8]}


def agent_worker(cell: dict[str, Any]) -> dict[str, Any]:
    seed = cell["seed"]
    cfg = SimulationConfig(max_steps=STEPS, seed=seed, sensor_range=8.0)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg)
    for ep in range(N_TRAIN):
        run_episode(env, org, cfg, ep, None, {"layout": LAYOUT, "cue_value": V0},
                    seed=seed * 1000 + ep)
    # Non-spatial probe of the conditioned response across cue values.
    gradient = [org.cue_field.response(float(v)) for v in PROBE_VALUES]
    return {"seed": seed, "gradient": gradient}


def main(n_agents: int = N_AGENTS) -> None:
    print(f"Generalization demo: {n_agents} agents x {N_TRAIN} training lives (cue v0={V0})...")
    results = run_sweep(agent_worker, [{"seed": s} for s in range(n_agents)], progress_every=25)
    responses = np.array([r["gradient"] for r in results])  # (agents, values)

    plot_generalization_gradient(PROBE_VALUES, responses, V0, FIG_DIR / "generalization.png")

    mean = responses.mean(axis=0)
    peak_v = PROBE_VALUES[int(np.argmax(mean))]
    edge = (mean[0] + mean[-1]) / 2
    print(f"Gradient peak at cue value {peak_v:.2f} (trained {V0}); "
          f"mean response: peak {mean.max():.2f} vs edges {edge:.2f} "
          f"({100*(1-edge/mean.max()):.0f}% drop at the extremes).")
    print(f"Wrote {FIG_DIR/'generalization.png'}")


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--agents", type=int, default=N_AGENTS)
    main(_ap.parse_args().agents)
