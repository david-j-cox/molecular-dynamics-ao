"""exp043 -- standing robustness battery for the core phenomena, on the current engine.

Each core phenomenon was demonstrated once at a tuned point (the reproduce harness re-checks those
points). This sweeps each one's key parameter across a range and confirms the qualitative signature
still holds on the engine as it currently sits -- robustness (not knife-edge) plus a freshness check
after the recent additions. Companion to exp042 (the newer phenomena). Each phenomenon is driven
through its validated path/regime.

Signatures (should hold across each range):
- Acquisition: latency to food falls within a life-block (early > late) across learning rate.
- Matching:    log response ratio tracks log reinforcement ratio across VI splits (slope ~ 1).
- Momentum:    the RICH component resists satiation more than the lean one across lean-VI values.
- Risk:        risk-prone below the requirement, averse above (reversal > 0) across e_req values.

Honest boundary found: acquisition is robust for learning rate up to ~0.06 and DESTABILISES above
~0.1 (the weight update overshoots) -- reported so the safe operating range is explicit.

Run:  python experiments/exp043_core_robustness.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from behavioral_md.chamber import (
    ChamberConfig,
    run_concurrent_chamber,
    run_multiple_schedule,
    run_risk_choice,
)
from behavioral_md.config import SimulationConfig
from behavioral_md.environments import BehavioralFieldEnv
from behavioral_md.experiment_utils import weak_innate_atoms
from behavioral_md.organism import Organism
from behavioral_md.parallel import run_sweep
from behavioral_md.simulation import run_episode

FIG = Path("outputs/figures")
LAYOUT = {"position": [4, 4], "food": [4, 8], "danger": [9, 0], "light": [0, 9], "cue": [8, 4]}


def _acq_worker(cell: dict[str, Any]) -> dict[str, Any]:
    lr, seed = cell["lr"], cell["seed"]
    cfg = SimulationConfig(n_episodes=25, max_steps=200, seed=seed, sensor_range=8.0,
                           reinforcement_asymptote=2.0, learning_rate=lr)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg, atoms=weak_innate_atoms(0.2))
    lat = [run_episode(env, org, cfg, ep, None, {"layout": LAYOUT},
                       seed=seed * 1000 + ep)["latency"] for ep in range(25)]
    return {"lr": lr, "drop": float(np.mean(lat[:5]) - np.mean(lat[-5:]))}


def acquisition(lrs):
    cells = [{"lr": lr, "seed": s} for lr in lrs for s in range(24)]
    res = run_sweep(_acq_worker, cells, progress_every=len(cells))
    return [float(np.mean([r["drop"] for r in res if r["lr"] == lr])) for lr in lrs]


def matching(vi_pairs):
    cfg = ChamberConfig(temperature=0.1, emission_bias=0.0, energy_init=0.6)
    log_b, log_r = [], []
    for vl, vr in vi_pairs:
        out = run_concurrent_chamber([vl, vr], cfg, n_org=600, n_steps=4000, seed=0)
        b = out["emit"].sum(axis=0) + 1.0
        rr = out["reinforced"].sum(axis=0) + 1.0
        log_b.append(np.log(b[0] / b[1]))
        log_r.append(np.log(rr[0] / rr[1]))
    slope = float(np.polyfit(log_r, log_b, 1)[0])
    return np.array(log_r), np.array(log_b), slope


def momentum(lean_vis):
    # exp022's validated regime: satiation disruptor, momentum from the context drive.
    cfg = ChamberConfig(motiv_strength=2.0, energy_init=0.5, emission_bias=1.2, temperature=0.5,
                        ctx_drive_gain=0.8, momentum_mass_gain=0.0)
    rich, lean = [], []
    for vlean in lean_vis:
        res = run_multiple_schedule([5.0, vlean], cfg, 300, 300, 12, 10,
                                    disruptor="satiation", seed=0)
        pr, nb = res["press_rate"], res["n_baseline"]
        resist = (pr[nb:] / pr[nb - 3:nb].mean(0)).mean(0)
        rich.append(resist[0])
        lean.append(resist[1])
    return np.array(rich), np.array(lean)


def risk(e_reqs):
    safe = [(1.0, 0.05)]
    risky = [(0.5, 0.0), (0.5, 0.10)]
    cfg = ChamberConfig(temperature=0.02)
    revs = []
    for er in e_reqs:
        r = run_risk_choice(safe, risky, cfg, 4000, 1000, seed=0, cost=0.05, e_init=0.5, e_req=er)
        bins = np.asarray(r["energy_bins"])
        curve = np.asarray(r["risky_by_energy"], float)
        below = np.nanmean(curve[(bins < er) & (bins > er - 0.2)])
        above = np.nanmean(curve[(bins >= er) & (bins < er + 0.2)])
        revs.append(below - above)
    return np.array(revs)


def main() -> None:
    lrs = [0.02, 0.04, 0.06, 0.08]
    acq = acquisition(lrs)
    vi_pairs = [(3, 30), (6, 18), (12, 12), (18, 6), (30, 3)]
    logr, logb, slope = matching(vi_pairs)
    lean_vis = [20.0, 30.0, 45.0]
    rich_res, lean_res = momentum(lean_vis)
    e_reqs = [0.3, 0.4, 0.5, 0.6]
    revs = risk(e_reqs)

    print("Core phenomena robustness on the current engine:\n")
    print(f"Acquisition  latency drop vs lr {lrs}:\n  {np.round(acq, 1)}  (> 0 -> learns)")
    print(f"Matching     log-ratio sensitivity slope = {slope:.2f}  (~1 -> tracks reinforcement)")
    print(f"Momentum     lean VI {lean_vis}:\n  rich {np.round(rich_res, 2)} vs "
          f"lean {np.round(lean_res, 2)}  (rich > lean -> momentum)")
    print(f"Risk         e_req {e_reqs}:\n  reversal {np.round(revs, 2)}  (> 0 -> prone below req)")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    ax = axes[0, 0]
    ax.plot(lrs, acq, "o-", color="tab:blue")
    ax.axhline(0, color="0.5", ls="--", lw=1)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("latency drop (early - late)")
    ax.set_title("Acquisition: learns across learning rate")
    ax = axes[0, 1]
    ax.plot(logr, logb, "o", color="tab:green")
    lo, hi = float(logr.min()), float(logr.max())
    ax.plot([lo, hi], [slope * lo, slope * hi], "-", color="0.5", label=f"slope = {slope:.2f}")
    ax.plot([lo, hi], [lo, hi], ":", color="0.7", label="perfect matching")
    ax.set_xlabel("log reinforcement ratio")
    ax.set_ylabel("log response ratio")
    ax.legend(fontsize=8)
    ax.set_title("Matching: response tracks reinforcement")
    ax = axes[1, 0]
    ax.plot(lean_vis, rich_res, "o-", color="tab:red", label="rich component")
    ax.plot(lean_vis, lean_res, "s-", color="0.6", label="lean component")
    ax.set_xlabel("lean component VI")
    ax.set_ylabel("resistance (disrupt / baseline)")
    ax.legend(fontsize=8)
    ax.set_title("Momentum: rich resists satiation more")
    ax = axes[1, 1]
    ax.plot(e_reqs, revs, "o-", color="tab:purple")
    ax.axhline(0, color="0.5", ls="--", lw=1)
    ax.set_xlabel("requirement e_req")
    ax.set_ylabel("risk reversal (below - above)")
    ax.set_title("Risk: prone below the requirement across e_req")
    fig.suptitle("exp043: core phenomena are robust across parameter ranges (current engine)",
                 fontsize=13)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "exp043_core_robustness.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
