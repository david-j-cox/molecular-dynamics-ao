"""The last gap: the organism LEARNS the option distributions within its life.

So far the distributions have always been handed over -- to the planner (the DP), or as the
fixed support that selection acted on. Here the organism starts ignorant: it estimates each
option's outcome distribution from what it observes, plans survival (the DP) on its current
estimate, forages day/night, and respawns on death keeping what it has learned
(survival.simulate_learning_choice). Nothing tells it which option is risky.

It finds out. With a little (annealing) exploration it samples the risky option, its estimated
variance climbs to the truth, and within a few cycles its planned policy gambles exactly where
the energy-budget rule prescribes (recall 0 -> 1). Survival improves as it learns. The learned
threshold lands on the DP optimum -- the rule, now emerging from EXPERIENCE.

This closes the arc: imposed (exp030) -> derived (the DP) -> executed (the behavioral loop) ->
evolved (selection) -> learned (within life), the same energy-budget rule each time.

Run:   python studies/risk_sensitivity/learning.py
Saves: studies/risk_sensitivity/figures/within_life_learning.png + learning_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md.survival import simulate_learning_choice  # noqa: E402

FIG = Path(__file__).parent / "figures"
DAY, NIGHT, METAB, S = 24, 24, 0.03, 0.05
SAFE = [(1.0, S)]
RISKY = [(0.5, 0.0), (0.5, 2 * S)]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 10})


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    r = simulate_learning_choice(SAFE, RISKY, DAY, NIGHT, METAB, n_org=60, n_cycles=50, seed=0)
    cyc = np.arange(len(r["gamble_recall"]))
    recall = np.array(r["gamble_recall"])
    var_frac = np.array(r["risky_variance"]) / r["true_risky_variance"]
    surv = np.array(r["survival"])
    learned = np.array(r["learned_theta"])
    dp = np.array(r["dp_theta"])
    t = np.arange(DAY)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: learning curves -- discover the variance, learn to gamble where it pays.
    axL.plot(cyc, recall, color="black", lw=1.8, marker="o", ms=3, label="Gamble recall")
    axL.plot(cyc, var_frac, color="0.5", lw=1.8, ls="--",
             label="Estimated risky variance (frac. of true)")
    axL.plot(cyc, surv, color="0.0", lw=1.2, ls=":", label="Survival")
    axL.axhline(1.0, color="0.9", lw=0.6)
    axL.set_xlabel("Cycle (experience)")
    axL.set_ylabel("Fraction")
    axL.set_title("Learning the distribution from experience", fontsize=11)
    axL.set_ylim(0, 1.05)
    axL.legend(loc="center right", frameon=False)

    # Right: the learned threshold lands on the DP optimum.
    axR.plot(learned, t, color="black", lw=1.8, marker="o", ms=4, label="Learned threshold")
    axR.plot(dp, t, color="0.55", lw=1.8, ls="--", label="DP-optimal threshold")
    axR.set_xlabel("Risk threshold θ (gamble below)")
    axR.set_ylabel("Time of day (0 = dawn → dusk)")
    axR.set_title("Learned policy ≈ DP optimum", fontsize=11)
    axR.set_xlim(0, 1)
    axR.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "within_life_learning.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    (FIG / "learning_summary.json").write_text(json.dumps(
        {"gamble_recall": r["gamble_recall"], "risky_variance": r["risky_variance"],
         "true_risky_variance": r["true_risky_variance"], "survival": r["survival"],
         "learned_theta": r["learned_theta"], "dp_theta": r["dp_theta"]}, indent=2))

    print(f"True risky variance: {r['true_risky_variance']:.4f}")
    print("Cycle   gamble_recall   est_risky_var   survival")
    for c in (0, 1, 3, 10, 49):
        print(f"  {c:2d}      {recall[c]:.2f}            {r['risky_variance'][c]:.4f}"
              f"          {surv[c]:.2f}")
    print(f"Learned vs DP threshold at dusk: {learned[-1]:.2f} vs {dp[-1]:.2f}")
    print(f"Saved {FIG/'within_life_learning.png'}")


if __name__ == "__main__":
    main()
