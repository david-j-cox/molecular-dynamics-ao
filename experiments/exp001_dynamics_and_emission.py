"""Experiment 001 -- can the organism reach food, and under what dynamics/emission?

Motivation
----------
With the literal spec (pure Verlet integration + argmax action emission) the
organism never reaches food. This experiment quantifies that, isolates the two
causes, and measures candidate fixes.

It probes variations that are NOT yet in the core code (velocity damping and
softmax emission), by re-implementing the organism's per-step update here. Core
``Organism.step``/``emit_action`` are untouched so this script documents what
*would* happen under each candidate before we change any defaults.

Run:  python experiments/exp001_dynamics_and_emission.py
Saves: outputs/logs/exp001_results.json  (raw numbers for the lab notebook)
"""

from __future__ import annotations

import numpy as np

from behavioral_md.atoms import ACTION_ATOMS, default_atom_set
from behavioral_md.config import SimulationConfig
from behavioral_md.environments import BehavioralFieldEnv
from behavioral_md.experiment_utils import save_results_json
from behavioral_md.forces import sensory_from_observation
from behavioral_md.organism import Organism

# Fixed, reproducible layout: food up-and-right of start; danger in a far
# corner; light and cue off to the sides. Manhattan start->food distance = 10.
LAYOUT = {
    "position": [1, 1],
    "food": [6, 6],
    "danger": [9, 0],
    "light": [0, 9],
    "cue": [3, 3],
}
ACTION_LABELS = ["noop", "up", "down", "left", "right", "eat", "pause"]


def _make_atoms(food_sens: float, danger_sens: float):
    atoms = default_atom_set()
    for a in atoms:
        if a.name.startswith("move_"):
            a.sensitivity["food"] = food_sens
            a.sensitivity["danger"] = danger_sens
    return atoms


def run_episode(
    *,
    damping: float,
    emission: str,          # "argmax" | "softmax"
    temperature: float = 0.5,
    food_sens: float = 0.5,
    danger_sens: float = -0.4,
    sensor_range: float = 10.0,
    consume_radius: float = 1.0,
    dt: float = 0.1,
    steps: int = 200,
    seed: int = 0,
    record_path: bool = False,
):
    """Run one episode under the given dynamics/emission; return summary dict."""
    cfg = SimulationConfig(
        grid_size=10,
        max_steps=steps,
        dt=dt,
        seed=seed,
        consume_radius=consume_radius,
        sensor_range=sensor_range,
    )
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg, atoms=_make_atoms(food_sens, danger_sens))
    rng = np.random.default_rng(seed)

    obs, _ = env.reset(seed=seed, options={"layout": LAYOUT})
    org.reset(obs)
    ids = list(ACTION_ATOMS.keys())
    reached = None
    path = []
    min_dist = np.inf

    for t in range(steps):
        sensory = sensory_from_observation(obs)
        force, comp = org.force_calc.compute(sensory, org.energy)

        # Verlet step, optionally with velocity damping (-c * v).
        for i, atom in enumerate(org.atoms):
            vel = (atom.state[0] - atom.previous_state[0]) / dt
            f = force[i] - damping * vel
            atom.integrate(f, dt, cfg.activation_min, cfg.activation_max)
        org.last_force = force
        org.last_components = comp
        org.eligibility.update(np.array([a.activation for a in org.atoms]))

        # Emission.
        acts = np.array([org.activation(ACTION_ATOMS[i]) for i in ids])
        if emission == "argmax":
            best = acts.max()
            winners = [ids[k] for k in np.flatnonzero(acts >= best - 1e-9)]
            action = int(rng.choice(winners))
        else:  # softmax
            z = acts / temperature
            z -= z.max()
            p = np.exp(z)
            p /= p.sum()
            action = int(rng.choice(ids, p=p))

        obs, r, term, trunc, info = env.step(action)
        min_dist = min(min_dist, info["distance_to_food"])
        if record_path:
            path.append((int(info["position"][0]), int(info["position"][1])))
        if info.get("consumed", False) and reached is None:
            reached = t + 1
        if term or trunc:
            break

    return {
        "reached": reached,
        "min_distance": round(float(min_dist), 3),
        "path": path,
    }


def reach_stats(n_seeds: int = 12, **kw) -> dict:
    """Aggregate reach rate and latency across seeds for one configuration."""
    results = [run_episode(seed=s, **kw) for s in range(n_seeds)]
    latencies = [r["reached"] for r in results if r["reached"] is not None]
    hits = len(latencies)
    return {
        "n_seeds": n_seeds,
        "reached": hits,
        "reach_rate": round(hits / n_seeds, 3),
        "median_latency": float(np.median(latencies)) if latencies else None,
        "min_dist_when_failed": round(
            float(np.median([r["min_distance"] for r in results if r["reached"] is None])), 2
        )
        if hits < n_seeds
        else None,
        "latencies": latencies,
    }


def main() -> None:
    out: dict = {"layout": LAYOUT, "configs": {}}

    # --- A. Sample trajectory under the literal spec (shows wall-pinning). ----
    traj = run_episode(
        damping=0.0, emission="argmax", food_sens=0.5, seed=0, steps=80, record_path=True
    )
    out["literal_spec_sample_trajectory"] = {
        "reached": traj["reached"],
        "min_distance": traj["min_distance"],
        "path": traj["path"],
    }

    # --- B. Reach-rate matrix over candidate fixes. --------------------------
    configs = {
        "pure_verlet_argmax (literal spec)": dict(
            damping=0.0, emission="argmax", food_sens=0.5, sensor_range=10.0
        ),
        "pure_verlet_softmax_T1.0": dict(
            damping=0.0, emission="softmax", temperature=1.0, food_sens=0.5, sensor_range=10.0
        ),
        "damping_c2_argmax": dict(
            damping=2.0, emission="argmax", food_sens=0.5, sensor_range=10.0
        ),
        "overdamped_c20_softmax_T0.3 (tuned)": dict(
            damping=20.0,
            emission="softmax",
            temperature=0.3,
            food_sens=1.5,
            sensor_range=5.0,
        ),
    }
    for name, kw in configs.items():
        out["configs"][name] = {"params": kw, "stats": reach_stats(**kw)}

    # --- Save + print. -------------------------------------------------------
    path = save_results_json("exp001_results.json", out)

    print(f"Saved {path}\n")
    print("A. Literal-spec sample trajectory (food at (6,6), 80 steps):")
    print(f"   reached={traj['reached']}  min_dist={traj['min_distance']}")
    print(f"   path={traj['path']}\n")
    print("B. Reach-rate over 12 seeds:")
    print(f"   {'config':40s} reach   median_lat  min_dist_if_failed")
    for name, d in out["configs"].items():
        s = d["stats"]
        print(
            f"   {name:40s} {s['reached']:2d}/12   "
            f"{str(s['median_latency']):>8s}    {s['min_dist_when_failed']}"
        )


if __name__ == "__main__":
    main()
