"""The strictest version: a MODEL-FREE learner -- no model, no planning.

The within-life learner (learning.py) still PLANS: it learns the option distributions and
runs the survival DP on them. This one does neither. Each organism holds a tabular value
Q[energy_bin, time_of_day, action] and learns it by Monte-Carlo from the bare SURVIVAL signal
-- after each day/night cycle, every (state, action) it visited is nudged toward 1 if it
survived and 0 if it died (survival.simulate_model_free_choice). There is no model of the
distributions and no planning; survival values are learned directly from living and dying.

The energy-budget rule emerges anyway: the aggregate greedy policy's gamble recall climbs and
its threshold lands on the DP optimum. It is the cost of assuming the least -- markedly slower
and noisier than the model-based learner (which plans on its learned model) -- but it gets
there. The same rule, from nothing but reinforcement.

Run:   python studies/risk_sensitivity/model_free.py
Saves: studies/risk_sensitivity/figures/model_free.png + model_free_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md.survival import (  # noqa: E402
    simulate_learning_choice,
    simulate_model_free_choice,
)

FIG = Path(__file__).parent / "figures"
DAY, NIGHT, METAB, S = 24, 24, 0.03, 0.05
SAFE = [(1.0, S)]
RISKY = [(0.5, 0.0), (0.5, 2 * S)]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 10})


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    mf = simulate_model_free_choice(SAFE, RISKY, DAY, NIGHT, METAB, n_org=300, n_cycles=300,
                                    seed=0)
    mb = simulate_learning_choice(SAFE, RISKY, DAY, NIGHT, METAB, n_org=60, n_cycles=50, seed=0)
    mf_rec = np.array(mf["gamble_recall"])
    mb_rec = np.array(mb["gamble_recall"])
    learned = np.array(mf["learned_theta"])
    dp = np.array(mf["dp_theta"])
    t = np.arange(DAY)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: the cost of assuming less -- model-free is slower and noisier than model-based.
    axL.plot(np.arange(len(mb_rec)), mb_rec, color="0.55", lw=1.8, ls="--",
             label="Model-based (learns + plans)")
    axL.plot(np.arange(len(mf_rec)), mf_rec, color="black", lw=1.8,
             label="Model-free (no model, no planning)")
    axL.axhline(1.0, color="0.9", lw=0.6)
    axL.set_xlabel("Cycle (experience)")
    axL.set_ylabel("Gamble recall")
    axL.set_title("The energy-budget rule emerges from reinforcement", fontsize=11)
    axL.set_ylim(0, 1.05)
    axL.legend(loc="lower right", frameon=False)

    # Right: the model-free learned threshold lands on the DP optimum.
    axR.plot(learned, t, color="black", lw=1.8, marker="o", ms=4, label="Learned threshold")
    axR.plot(dp, t, color="0.55", lw=1.8, ls="--", label="DP-optimal threshold")
    axR.set_xlabel("Risk threshold θ (gamble below)")
    axR.set_ylabel("Time of day (0 = dawn → dusk)")
    axR.set_title("Model-free policy ≈ DP optimum", fontsize=11)
    axR.set_xlim(0, 1)
    axR.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "model_free.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    (FIG / "model_free_summary.json").write_text(json.dumps(
        {"model_free_recall": mf["gamble_recall"], "model_based_recall": mb["gamble_recall"],
         "survival": mf["survival"], "learned_theta": mf["learned_theta"],
         "dp_theta": mf["dp_theta"]}, indent=2))

    print("Gamble recall (energy-budget rule recovered):")
    print(f"  model-free  -> cycle 5: {mf_rec[5]:.2f}, cycle 60: {mf_rec[60]:.2f}, "
          f"final: {mf_rec[-1]:.2f}")
    print(f"  model-based -> cycle 1: {mb_rec[1]:.2f}, cycle 5: {mb_rec[5]:.2f}")
    diff = float(np.nanmean(np.abs(learned - dp)))
    print(f"Model-free learned vs DP threshold: mean |diff| = {diff:.3f}")
    print(f"Saved {FIG/'model_free.png'}")


if __name__ == "__main__":
    main()
