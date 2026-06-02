"""Rapid reacquisition demo (dual excitatory/inhibitory vs Rescorla-Wagner).

Each agent lives TRAIN reinforced lives, then EXT non-reinforced lives (extinction),
then REACQ reinforced lives again. Under the Konorski/Bouton dual rule, extinction
builds a separate inhibition (w-) and PRESERVES the original excitation (w+); when
reinforcement returns, the labile inhibition is rapidly cancelled and the preserved
excitation is already in place, so the net association recovers in far fewer lives than
the original acquisition took. The Rescorla-Wagner control erases the weight during
extinction, so its reacquisition is no faster than original acquisition.

Measures across lives (population, 95% CI): the net association the force reads
(approach_food food weight), plus w+ and w- (dual arm). A population runs in parallel.

Run:  python scripts/run_reacquisition_demo.py
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
from behavioral_md.visualization import plot_dual_components

FIG_DIR = Path("outputs/figures")
N_AGENTS = 100
N_TRAIN, N_EXT, N_REACQ = 25, 20, 15
STEPS = 300
# A slow acquisition rate spreads original learning over several lives (so reacquisition
# speed is resolvable); inhibition relaxes fast (config default 0.1, labile), so the dual
# rule removes inhibition and rides the preserved excitation back up far faster than RW
# rebuilds its erased weight from scratch.
LEARNING_RATE = 0.01
THRESHOLD = 0.5  # net association marking "responding reacquired"
LAYOUT = {"position": [4, 4], "food": [4, 8], "danger": [9, 0], "light": [0, 9], "cue": [8, 4]}


def _life(env, org, cfg, reinforce, seed):
    """Run one life; return (net, w_plus, w_minus, steps_at_food) for approach_food."""
    obs, info = env.reset(seed=seed, options={"layout": LAYOUT, "food_reinforces": reinforce})
    org.reset(obs)
    af = org.atoms[org.index["approach_food"]]
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
    return (af.history_weights["food"], af.w_plus["food"], af.w_minus["food"], steps_at_food)


def _run_arm(model: str, seed: int) -> list[dict[str, Any]]:
    cfg = SimulationConfig(max_steps=STEPS, seed=seed, sensor_range=8.0,
                           learning_model=model, learning_rate=LEARNING_RATE)
    env, org = BehavioralFieldEnv(cfg), Organism(cfg)
    rows = []
    for ep in range(N_TRAIN + N_EXT + N_REACQ):
        reinforce = ep < N_TRAIN or ep >= N_TRAIN + N_EXT
        net, wp, wm, saf = _life(env, org, cfg, reinforce, seed * 1000 + ep)
        rows.append({"seed": seed, "episode": ep, "net": net, "w_plus": wp,
                     "w_minus": wm, "steps_at_food": saf})
    return rows


def agent_worker(cell: dict[str, Any]) -> dict[str, Any]:
    seed = cell["seed"]
    return {"dual": _run_arm("dual_exc_inhib", seed), "rw": _run_arm("rescorla_wagner", seed)}


def _lives_to_threshold(df: pd.DataFrame, phase_start: int) -> float:
    """Mean over seeds of lives (from phase_start) until net first reaches THRESHOLD."""
    out = []
    for _seed, g in df.groupby("seed"):
        net = g.sort_values("episode")["net"].to_numpy()[phase_start:]
        hit = np.argmax(net >= THRESHOLD) if (net >= THRESHOLD).any() else len(net)
        out.append(hit)
    return float(np.mean(out))


def main(n_agents: int = N_AGENTS) -> None:
    print(f"Reacquisition demo: {n_agents} agents x "
          f"({N_TRAIN} train + {N_EXT} ext + {N_REACQ} reacq) lives, dual vs RW...")
    results = run_sweep(agent_worker, [{"seed": s} for s in range(n_agents)], progress_every=25)
    dual = pd.DataFrame([r for res in results for r in res["dual"]])
    rw = pd.DataFrame([r for res in results for r in res["rw"]])

    plot_dual_components(dual, transitions=[N_TRAIN, N_TRAIN + N_EXT],
                         path=FIG_DIR / "reacquisition.png",
                         title="Rapid reacquisition (dual exc/inhib)")

    reacq_start = N_TRAIN + N_EXT
    d_acq = _lives_to_threshold(dual, 0)
    d_reacq = _lives_to_threshold(dual, reacq_start)
    r_acq = _lives_to_threshold(rw, 0)
    r_reacq = _lives_to_threshold(rw, reacq_start)

    print("  lives to reach net>=0.5 (mean over agents):")
    print(f"    dual:  acquisition {d_acq:.2f}  ->  reacquisition {d_reacq:.2f}")
    print(f"    RW:    acquisition {r_acq:.2f}  ->  reacquisition {r_reacq:.2f}")
    print(f"  dual reacquisition faster than its acquisition:  "
          f"{'OK' if d_reacq < d_acq else 'FAIL'}")
    print(f"  dual reacquisition faster than RW reacquisition: "
          f"{'OK' if d_reacq < r_reacq else 'FAIL'}")
    print(f"Wrote {FIG_DIR/'reacquisition.png'}")


if __name__ == "__main__":
    import argparse

    _ap = argparse.ArgumentParser()
    _ap.add_argument("--agents", type=int, default=N_AGENTS)
    main(_ap.parse_args().agents)
