"""exp048 -- does the second-order (inertial) Verlet term earn its keep?

Peer reviewers (AI/RL and philosophy of science) noted the manuscript's own analysis concedes the
overdamped Verlet limit "is" a leaky competing accumulator (Usher & McClelland 2001), so the
molecular-dynamics framing may be decorative: a first-order leaky integrator
x <- x + dt*(force/m - leak*x) might reproduce the same behavior as the damped second-order Verlet
update. This ablation tests that on the flagship acquisition phenomenon (run_demo's controlled
layout), reporting the learning curve under each integrator.

If the first-order leaky integrator reproduces the acquisition curve, the inertial term is not
load-bearing and the MD framing should be demoted to a motivating analogy (the engine is a
leaky-accumulator + softmax + Rescorla-Wagner value learner). If not, the term earns its keep.

Run:  python experiments/exp048_integrator_ablation.py
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from behavioral_md.config import SimulationConfig
from behavioral_md.environments.gridworld import BehavioralFieldEnv
from behavioral_md.experiment_utils import weak_innate_atoms
from behavioral_md.organism import Organism
from behavioral_md.parallel import run_sweep
from behavioral_md.simulation import run_episode

N_AGENTS = 60
N_LIVES = 40
STEPS = 300
SENSOR_RANGE = 8.0
INNATE_FOOD = 0.2
REINF_ASYMPTOTE = 2.0
LAYOUT = {"position": [4, 4], "food": [4, 8], "danger": [9, 0], "light": [0, 9], "cue": [8, 4]}


def _worker(cell: dict[str, Any]) -> dict[str, Any]:
    seed = cell["seed"]
    kw = dict(n_episodes=N_LIVES, max_steps=STEPS, seed=seed, sensor_range=SENSOR_RANGE,
              reinforcement_asymptote=REINF_ASYMPTOTE, integrator=cell["integrator"])
    if cell["integrator"] == "leaky":
        kw["leak_coef"] = cell["leak"]
    cfg = SimulationConfig(**kw)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg, atoms=weak_innate_atoms(INNATE_FOOD))
    rows = []
    for ep in range(N_LIVES):
        r = run_episode(env, org, cfg, ep, None, {"layout": LAYOUT}, seed=seed * 1000 + ep)
        rows.append({"episode": ep, "latency": r["latency"], "alive": bool(r["alive"])})
    return {"summaries": rows}


def _curve(integrator: str, leak: float = 1.0) -> dict[str, float]:
    cells = [{"seed": s, "integrator": integrator, "leak": leak} for s in range(N_AGENTS)]
    res = run_sweep(_worker, cells, progress_every=200)
    df = pd.DataFrame([r for x in res for r in x["summaries"]])
    by_life = df.groupby("episode")["latency"].mean()
    early = float(by_life.iloc[:5].mean())
    late = float(by_life.iloc[-5:].mean())
    reach = float((df["latency"] < STEPS).groupby(df["episode"]).mean().iloc[-5:].mean())
    death = float(1.0 - df.groupby("episode")["alive"].mean().iloc[-5:].mean())
    return {"early": early, "late": late, "drop": early - late, "reach": reach, "death": death}


def main() -> None:
    print(f"Integrator ablation: {N_AGENTS} agents x {N_LIVES} lives, controlled acquisition.\n")
    rows = []
    rows.append(("damped Verlet (default)", _curve("verlet")))
    for leak in (0.5, 1.0, 2.0):
        rows.append((f"leaky (leak={leak})", _curve("leaky", leak)))
    print(f"{'integrator':<26}{'latency early->late':<22}{'drop':>7}{'reach':>8}{'death':>8}")
    for name, c in rows:
        print(f"{name:<26}{c['early']:6.1f} -> {c['late']:<11.1f}{c['drop']:7.1f}"
              f"{c['reach']:8.2f}{c['death']:8.2f}")
    v = rows[0][1]
    best = min((r for r in rows[1:]), key=lambda r: abs(r[1]["drop"] - v["drop"]))
    print(f"\nVerlet drop {v['drop']:.1f} vs closest leaky ({best[0]}) drop {best[1]['drop']:.1f}: "
          f"the first-order leaky integrator "
          f"{'reproduces' if abs(best[1]['drop']-v['drop'])<15 else 'does NOT reproduce'} "
          "the acquisition curve.")


if __name__ == "__main__":
    main()
