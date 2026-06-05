"""exp038 -- should the learning signal be graded by amount eaten, or normalized per event?

Open engine-quality question (ToDo). The default DeltaEnergy normalizes the appetitive teaching
signal to ~1 per feeding event, because the raw per-step intake is tiny. The alternative, still
objective, grades the signal by the AMOUNT eaten (intake / food_intake_rate): a depleted patch
yields less food AND less learning. This compares the two on the controlled acquisition protocol
(fixed layout, weak innate food drive, amplified asymptote -- the run_demo setup), reporting the
acquisition effect (latency drop), reliability (late reach), and mortality.

Default (normalized) is unchanged in the engine; graded is opt-in via DeltaEnergy(graded=True).

Run:  python experiments/exp038_graded_teaching.py
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from behavioral_md.config import SimulationConfig
from behavioral_md.consequence import DeltaEnergy
from behavioral_md.environments import BehavioralFieldEnv
from behavioral_md.experiment_utils import weak_innate_atoms
from behavioral_md.organism import Organism
from behavioral_md.parallel import run_sweep
from behavioral_md.simulation import run_episode

N_AGENTS = 80
N_LIVES = 40
STEPS = 300
SENSOR_RANGE = 8.0
INNATE_FOOD = 0.2
REINF_ASYMPTOTE = 2.0  # the amplified controlled protocol (exp031 / run_demo)
LAYOUT = {"position": [4, 4], "food": [4, 8], "danger": [9, 0], "light": [0, 9], "cue": [8, 4]}


def agent_worker(cell: dict[str, Any]) -> dict[str, Any]:
    seed, graded = cell["seed"], cell["graded"]
    cfg = SimulationConfig(n_episodes=N_LIVES, max_steps=STEPS, seed=seed,
                           sensor_range=SENSOR_RANGE, reinforcement_asymptote=REINF_ASYMPTOTE)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg, atoms=weak_innate_atoms(INNATE_FOOD))
    if graded:
        org.consequence_model = DeltaEnergy(danger_loss=cfg.danger_energy_loss,
                                            graded=True, graded_scale=cfg.food_intake_rate)
    rows = []
    for ep in range(N_LIVES):
        r = run_episode(env, org, cfg, ep, None, {"layout": LAYOUT}, seed=seed * 1000 + ep)
        rows.append({"graded": graded, "episode": ep, "latency": r["latency"],
                     "consumed": bool(r["consumed"]), "alive": bool(r["alive"])})
    return {"rows": rows}


def summarize(df: pd.DataFrame, steps: int) -> dict:
    by_life = df.groupby("episode")
    lat = by_life["latency"].mean()
    reach = by_life["consumed"].mean()
    return {
        "early_lat": round(lat.iloc[:5].mean(), 1),
        "late_lat": round(lat.iloc[-5:].mean(), 1),
        "drop": round(lat.iloc[:5].mean() - lat.iloc[-5:].mean(), 1),
        "late_reach": round(reach.iloc[-5:].mean(), 3),
        "death": round((~df["alive"]).mean(), 3),
    }


def main(n_agents: int = N_AGENTS) -> None:
    cells = [{"seed": s, "graded": g} for g in (False, True) for s in range(n_agents)]
    print(f"exp038: graded vs normalized teaching, {n_agents} agents x {N_LIVES} lives each...")
    results = run_sweep(agent_worker, cells, progress_every=max(1, len(cells) // 6))
    df = pd.DataFrame([r for res in results for r in res["rows"]])

    print("\n                       early_lat  late_lat  drop  late_reach  death")
    for graded, label in [(False, "normalized (default)"), (True, "graded by amount")]:
        s = summarize(df[df["graded"] == graded], STEPS)
        print(f"  {label:22s} {s['early_lat']:8.1f}  {s['late_lat']:8.1f}  {s['drop']:5.1f}  "
              f"{s['late_reach']:9.3f}  {s['death']:.3f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=N_AGENTS)
    main(ap.parse_args().agents)
