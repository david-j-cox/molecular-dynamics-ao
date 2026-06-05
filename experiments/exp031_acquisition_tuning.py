"""exp031 -- acquisition tuning sweep.

Goal (ToDo: TUNING / QUALITY):
  1. amplify the acquisition effect size on the controlled layout, and
  2. reduce bimodal reach-failure (agents that never lock onto food and starve),
  to decide final engine defaults.

The controlled protocol mirrors scripts/run_demo.py: a fixed arena, a weak innate
approach_food sensitivity so the *learned* history weight drives acquisition, and a
population of agents whose learning persists across lives. Here we wrap that protocol
in a parameter grid and report, per combo:

  drop      = early latency - late latency   (the acquisition effect; bigger is better)
  late_reach= fraction of late lives that reach food   (reliability; bigger is better)
  death     = fraction of lives ending in death        (smaller is better)

Each (combo, seed) agent is one parallel cell, so the pool fans out over the whole grid
at once. Pick a grid with --grid and population size with --agents.

Run:  python experiments/exp031_acquisition_tuning.py --grid coarse --agents 40
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import pandas as pd

from behavioral_md.config import SimulationConfig
from behavioral_md.environments import BehavioralFieldEnv
from behavioral_md.experiment_utils import weak_innate_atoms
from behavioral_md.organism import Organism
from behavioral_md.parallel import run_sweep
from behavioral_md.simulation import run_episode

OUT_DIR = Path("outputs/logs")
N_LIVES = 40
STEPS = 300
SENSOR_RANGE = 8.0
LAYOUT = {"position": [4, 4], "food": [4, 8], "danger": [9, 0], "light": [0, 9], "cue": [8, 4]}

# Config fields the grid may override; everything else stays at SimulationConfig defaults.
CONFIG_KEYS = {
    "learning_rate", "reinforcement_asymptote", "extinction_rate", "eligibility_decay",
    "softmax_temperature", "motivational_strength", "deficit_exponent",
    "move_cost", "basal_metabolism", "rest_cost", "food_intake_rate", "energy_init",
}
DEFAULTS = {  # baseline = current engine defaults (the point of comparison)
    "learning_rate": 0.05, "reinforcement_asymptote": 1.0, "softmax_temperature": 0.3,
    "food_intake_rate": 0.05, "motivational_strength": 2.0, "deficit_exponent": 2.0,
    "move_cost": 0.005, "basal_metabolism": 0.005, "innate_food": 0.2,
}

# Named grids. Each value is a list of settings for that axis; the cartesian product
# (overlaid on DEFAULTS) is the swept combos. Keep the baseline reachable in every grid.
GRIDS: dict[str, dict[str, list]] = {
    # axis 1: learning magnitude/speed and choice directedness + intake (survival).
    "coarse": {
        "learning_rate": [0.05, 0.10, 0.20],
        "reinforcement_asymptote": [1.0, 2.0],
        "softmax_temperature": [0.2, 0.3],
        "food_intake_rate": [0.05, 0.10],
    },
    # final: pick the locomotion cost at the amplified setting (asymptote=2.0). move_cost is
    # not load-bearing for the risk DP (own metabolism) or matching (no energy), so it is a
    # safe global lever for reach-failure; check it does not erase the acquisition effect.
    "final": {
        "reinforcement_asymptote": [2.0],
        "move_cost": [0.005, 0.004, 0.003],
    },
    # confirm: higher-N before/after on the SAFE levers (keep deficit_exponent=2.0 so the
    # convex marginal-value / risk mechanism is preserved). baseline = asymptote 1.0.
    "confirm": {
        "reinforcement_asymptote": [1.0, 2.0],
        "food_intake_rate": [0.05, 0.08],
        "innate_food": [0.2, 0.25],
    },
    # axis 2: survival / directedness levers aimed at reach-failure, with the learning
    # amplifier from the coarse grid (reinforcement_asymptote=2.0) held fixed.
    "survival": {
        "reinforcement_asymptote": [2.0],
        "softmax_temperature": [0.2, 0.3],
        "move_cost": [0.002, 0.005],
        "deficit_exponent": [1.0, 2.0, 3.0],
        "innate_food": [0.15, 0.2, 0.3],
    },
}


def _combo_id(combo: dict[str, Any]) -> str:
    return ",".join(f"{k}={combo[k]}" for k in sorted(combo))


def agent_worker(cell: dict[str, Any]) -> dict[str, Any]:
    p = cell["params"]
    seed = cell["seed"]
    overrides = {k: v for k, v in p.items() if k in CONFIG_KEYS}
    cfg = SimulationConfig(n_episodes=N_LIVES, max_steps=STEPS, seed=seed,
                           sensor_range=SENSOR_RANGE, **overrides)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg, atoms=weak_innate_atoms(p.get("innate_food", 0.2)))
    rows = []
    for ep in range(N_LIVES):
        r = run_episode(env, org, cfg, ep, None, {"layout": LAYOUT}, seed=seed * 1000 + ep)
        rows.append({"combo": cell["combo"], "episode": ep, "latency": r["latency"],
                     "consumed": bool(r["consumed"]), "alive": bool(r["alive"]),
                     "cause": r["cause_of_death"] or "alive"})
    return {"rows": rows}


def build_cells(grid_name: str, n_agents: int) -> tuple[list[dict], dict[str, dict]]:
    axes = GRIDS[grid_name]
    keys = sorted(axes)
    combos: dict[str, dict] = {}
    cells = []
    for values in itertools.product(*(axes[k] for k in keys)):
        params = dict(DEFAULTS)
        params.update(dict(zip(keys, values, strict=True)))
        cid = _combo_id({k: params[k] for k in keys})
        combos[cid] = {k: params[k] for k in keys}
        for s in range(n_agents):
            cells.append({"combo": cid, "seed": s, "params": params})
    return cells, combos


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cid, g in df.groupby("combo"):
        by_life = g.groupby("episode")
        lat = by_life["latency"].mean()
        reach = by_life["consumed"].mean()
        early_lat, late_lat = lat.iloc[:5].mean(), lat.iloc[-5:].mean()
        out.append({
            "combo": cid,
            "early_lat": round(early_lat, 1),
            "late_lat": round(late_lat, 1),
            "drop": round(early_lat - late_lat, 1),
            "late_reach": round(reach.iloc[-5:].mean(), 3),
            "death": round((~g["alive"]).mean(), 3),
            "starv": round((g["cause"] == "starvation").mean(), 3),
            "danger": round((g["cause"] == "danger").mean(), 3),
        })
    res = pd.DataFrame(out)
    # composite: reward a large drop and high late reach, penalize death.
    res["score"] = (res["drop"] + 120 * res["late_reach"] - 120 * res["death"]).round(1)
    return res.sort_values("score", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=list(GRIDS), default="coarse")
    ap.add_argument("--agents", type=int, default=40)
    args = ap.parse_args()

    cells, combos = build_cells(args.grid, args.agents)
    print(f"exp031 tuning: grid={args.grid}  {len(combos)} combos x {args.agents} agents "
          f"= {len(cells)} cells x {N_LIVES} lives")
    results = run_sweep(agent_worker, cells, progress_every=max(1, len(cells) // 8))
    df = pd.DataFrame([r for res in results for r in res["rows"]])
    table = summarize(df)

    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.max_colwidth", 80):
        print("\nRanked (score = drop + 120*late_reach - 120*death):\n")
        print(table.to_string(index=False))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"exp031_tuning_{args.grid}.json"
    path.write_text(json.dumps({"grid": args.grid, "n_agents": args.agents,
                                "combos": combos,
                                "table": table.to_dict(orient="records")}, indent=2))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
