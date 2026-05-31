"""Experiment 002 -- verify the Phase-4a controller in CORE code, and tune defaults.

Unlike Exp 001 (which prototyped damping/softmax outside the core), this runs the
actual `Organism` + `BehavioralFieldEnv` now that velocity damping and softmax
emission are built in. It sweeps damping x temperature x food-sensitivity x
sensor-range x seed in parallel across all CPU cores and reports reach rate +
latency per configuration -- both a correctness check and a defaults-tuning pass.

Run:  python experiments/exp002_core_controller_sweep.py
Saves: outputs/logs/exp002_results.json
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np

from behavioral_md.atoms import default_atom_set
from behavioral_md.config import SimulationConfig
from behavioral_md.environments import BehavioralFieldEnv
from behavioral_md.organism import Organism
from behavioral_md.simulation import run_episode
from experiments._parallel import run_sweep

LAYOUT = {
    "position": [1, 1],
    "food": [6, 6],
    "danger": [9, 0],
    "light": [0, 9],
    "cue": [3, 3],
}


def _atoms_with_food_sens(food_sens: float):
    atoms = default_atom_set()
    for a in atoms:
        if a.name.startswith("move_"):
            a.sensitivity["food"] = food_sens
    return atoms


def worker(cell: dict[str, Any]) -> dict[str, Any]:
    """Run one life with the core engine under this cell's parameters."""
    cfg = SimulationConfig(
        grid_size=10,
        max_steps=cell["steps"],
        seed=cell["seed"],
        damping_coef=cell["damping"],
        emission="softmax",
        softmax_temperature=cell["temperature"],
        sensor_range=cell["sensor_range"],
        consume_radius=1.0,
    )
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg, atoms=_atoms_with_food_sens(cell["food_sens"]))
    summary = run_episode(
        env, org, cfg, episode=0, logger=None,
        reset_options={"layout": LAYOUT}, seed=cell["seed"],
    )
    return {**cell, "reached": summary["consumed"], "latency": summary["latency"]}


def main() -> None:
    grid = {
        "damping": [2.0, 10.0, 20.0, 40.0],
        "temperature": [0.2, 0.3, 0.5],
        "food_sens": [0.5, 1.0, 1.5],
        "sensor_range": [5.0, 10.0],
        "seed": list(range(16)),
        "steps": [200],
    }
    keys = list(grid)
    cells = [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*grid.values())]
    print(f"Exp 002: {len(cells)} runs across CPU cores...")

    results = run_sweep(worker, cells)

    # Aggregate reach rate + median latency per (damping, temperature, food_sens,
    # sensor_range), collapsing over seeds.
    agg: dict[tuple, dict] = {}
    for r in results:
        key = (r["damping"], r["temperature"], r["food_sens"], r["sensor_range"])
        a = agg.setdefault(key, {"n": 0, "reached": 0, "latencies": []})
        a["n"] += 1
        if r["reached"]:
            a["reached"] += 1
            a["latencies"].append(r["latency"])

    rows = []
    for key, a in agg.items():
        rows.append(
            {
                "damping": key[0], "temperature": key[1], "food_sens": key[2],
                "sensor_range": key[3], "n": a["n"], "reached": a["reached"],
                "reach_rate": round(a["reached"] / a["n"], 3),
                "median_latency": float(np.median(a["latencies"])) if a["latencies"] else None,
            }
        )
    rows.sort(key=lambda r: (-r["reach_rate"], r["median_latency"] or 1e9))

    out = {"n_runs": len(results), "grid": grid, "aggregated": rows}
    path = Path("outputs/logs/exp002_results.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved {path}\n")

    print("Top 12 configurations by reach rate (over 16 seeds):")
    print(f"  {'damp':>5} {'temp':>5} {'food':>5} {'srange':>7} {'reach':>7} {'med_lat':>8}")
    for r in rows[:12]:
        print(
            f"  {r['damping']:5} {r['temperature']:5} {r['food_sens']:5} "
            f"{r['sensor_range']:7} {r['reached']:3}/{r['n']:<3} {str(r['median_latency']):>8}"
        )
    worst = rows[-1]
    print(f"\nWorst cell: {worst['reached']}/{worst['n']} "
          f"(damp={worst['damping']}, T={worst['temperature']}, "
          f"food={worst['food_sens']}, srange={worst['sensor_range']})")


if __name__ == "__main__":
    main()
