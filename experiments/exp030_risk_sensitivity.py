"""Experiment 030 -- risk-sensitive foraging and the energy-budget rule (Caraco, 1980).

A concurrent choice between a SAFE option (a constant outcome) and a RISKY option (a
variable outcome with the SAME mean), with energy dynamics and a real death boundary
(chamber.run_risk_choice). The organism chooses by a softmax over the EXPECTED SURVIVAL
UTILITY of each option at its CURRENT energy E: U(E) = logistic((E - e_req)/width) ~
P(survive). U is convex below the requirement and concave above, so by Jensen's
inequality the mean-preserving spread of the risky option is favored when starving
(risk-prone) and disfavored when well-fed (risk-averse). The preference reverses at the
energy requirement -- the energy-budget rule. Nothing encodes "gamble when hungry"; it
falls out of maximizing a survival-shaped utility.

Two preparations, each with a LINEAR-utility control (risk-neutral -> no reversal):
  A. reward variance   -- SAFE = +0.05 for sure; RISKY = 0 or +0.10 at p=0.5.
  B. predation variance -- SAFE = lean +0.05; RISKY = rich +0.0875, but a predation
                           strike (p=0.2) costs -0.10. Both matched to mean +0.05.

This is the proper version of the risk-sensitive foraging that the Phase 5 day/night
gridworld could not show: there, "risk" was a stationary, deterministic hazard (no
variance); here risk is genuine outcome variance, which is what the energy-budget rule
is about (see lab_notebook 2026-06-03/04).

Run:   python experiments/exp030_risk_sensitivity.py
Saves: outputs/logs/exp030_risk_sensitivity.json
       outputs/figures/exp030_risk_sensitivity.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from behavioral_md.chamber import ChamberConfig, run_risk_choice
from behavioral_md.experiment_utils import save_results_json
from behavioral_md.visualization import plot_risk_sensitivity

N_ORG = 5000
N_STEPS = 1000
E_REQ = 0.5
COST = 0.05
SEED = 0
MIN_COUNT = 3000          # mask undersampled energy bins

SAFE = [(1.0, 0.05)]
RISKY_REWARD = [(0.5, 0.0), (0.5, 0.10)]          # reward variance (matched mean 0.05)
RISKY_PREDATION = [(0.8, 0.0875), (0.2, -0.10)]   # predation variance (matched mean 0.05)


def _run(risky, util_shape: str) -> tuple[list, list]:
    cfg = ChamberConfig(temperature=0.02, energy_capacity=1.0)
    r = run_risk_choice(SAFE, risky, cfg, N_ORG, N_STEPS, seed=SEED, e_req=E_REQ,
                        util_width=0.08, cost=COST, e_init=0.5, util_shape=util_shape)
    rb = np.asarray(r["risky_by_energy"])
    bc = np.asarray(r["bin_count"])
    masked = np.where(bc >= MIN_COUNT, rb, np.nan).tolist()    # NaN where undersampled
    return masked, r["energy_bins"]


def _version(risky) -> dict:
    surv, bins = _run(risky, "survival")
    lin, _ = _run(risky, "linear")
    return {"survival": surv, "linear": lin, "energy_bins": bins}


def _reversal(curve, bins) -> tuple[float, float]:
    """Mean P(risky) just below vs just above the requirement (the reversal)."""
    c = np.asarray(curve, float)
    b = np.asarray(bins, float)
    below = np.nanmean(c[(b < E_REQ) & (b > E_REQ - 0.2)])
    above = np.nanmean(c[(b > E_REQ) & (b < E_REQ + 0.2)])
    return float(below), float(above)


def main() -> None:
    a = _version(RISKY_REWARD)
    b = _version(RISKY_PREDATION)
    bins = a["energy_bins"]
    results = {"e_req": E_REQ, "energy_bins": bins, "reward_variance": a,
               "predation_variance": b}
    save_results_json("exp030_risk_sensitivity.json", results)

    print("Energy-budget rule: P(choose risky) just below vs above the requirement "
          f"(e_req={E_REQ}):")
    for name, ver in (("reward variance", a), ("predation variance", b)):
        sb, sa = _reversal(ver["survival"], bins)
        lb, la = _reversal(ver["linear"], bins)
        print(f"  {name:20s} survival: below={sb:.2f} above={sa:.2f}  (reversal {sb - sa:+.2f})"
              f"   | linear control: below={lb:.2f} above={la:.2f}")

    p = plot_risk_sensitivity(bins, a, b, E_REQ,
                              Path("outputs/figures/exp030_risk_sensitivity.png"))
    print("\nRisk-prone below the requirement, risk-averse above -- only under the survival "
          "utility; the linear control is flat.")
    print("Saved outputs/logs/exp030_risk_sensitivity.json")
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
