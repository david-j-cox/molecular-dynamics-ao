"""Multi-patch foraging: risk-sensitive patch choice, and finite-horizon MVT.

The risk arc so far was a binary safe-vs-risky choice. A real forager faces a MENU of patches
and depleting patches it must decide when to leave. Both fall out of the same survival objective
(survival.survival_dp_patches, survival.survival_dp_depleting), and both join the marginal-value-
theorem work in the JAX engine (experiments/exp020_patch_leaving_mvt.py), which is rate-maximizing
and so risk-NEUTRAL. Survival makes patch foraging risk-sensitive.

1. Patch choice is a THREE-way energy-budget rule. A menu of a low-variance "safe" patch, a
   high-mean "rich" patch, and a high-variance "wild" patch. The survival-optimal choice over
   (energy, time-of-day):
     - SAFE (low variance) when comfortable -- above the requirement, just hold steady.
     - RICH (high mean) when below the requirement with time to spare -- maximize intake rate to
       climb toward R. This is the classic rate-maximizing / optimal-foraging regime.
     - WILD (high variance) when below the requirement near the deadline -- no time to climb
       steadily, so gamble on variance (the dusk lottery).
   Survival interpolates between rate-maximizing and variance-seeking depending on how much time
   is left to reach the requirement.

2. The giving-up rule is finite-horizon. A depleting patch with a travel cost to reach a fresh
   one. Through the day the organism abandons depleted patches readily (classic MVT relocation),
   but it STOPS leaving in the final stretch -- once fewer than ~travel_steps remain, there is no
   time to reach and exploit a fresh patch before the night fast. The leaving "deadline" tracks
   the travel cost (it moves earlier as travel gets more expensive), a finite-horizon effect that
   infinite-horizon MVT, with its single time-invariant giving-up density, cannot express.

Run:   python studies/risk_sensitivity/multi_patch.py
Saves: studies/risk_sensitivity/figures/multi_patch.png + multi_patch_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from behavioral_md.survival import (  # noqa: E402
    skewed_outcomes,
    survival_dp_depleting,
    survival_dp_patches,
)

FIG = Path(__file__).parent / "figures"
DAY, NIGHT, METAB = 30, 24, 0.03
R = NIGHT * METAB
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 10})


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    # --- Result 1: three-way patch-choice map -----------------------------------------------
    patches = [skewed_outcomes(0.045, 0.02, 0.0),    # 0 safe: low mean, low variance
               skewed_outcomes(0.060, 0.04, 0.0),    # 1 rich: high mean, moderate variance
               skewed_outcomes(0.050, 0.11, 0.0)]    # 2 wild: mid mean, high variance
    res = survival_dp_patches(patches, DAY, NIGHT, METAB)
    e, choice = res["energy"], res["choice"]               # choice[t, e]

    # --- Result 2: finite-horizon giving-up rule --------------------------------------------
    travels = [2, 4, 6]
    p_leave = {}
    last_leave = {}
    for tr in travels:
        d = survival_dp_depleting(0.12, 0.6, tr, DAY, NIGHT, METAB)
        eb, bb, act = d["energy"], d["biomass"], d["action"]
        bsel = (bb >= 0.3) & (bb <= 0.6)                   # a depleted patch: leaving is live
        esel = (eb > 0.2) & (eb < R)                       # below-R foragers (rate-relevant)
        pl = np.array([act[t][np.ix_(esel, bsel)].mean() for t in range(DAY)])
        p_leave[tr] = pl
        last_leave[tr] = int(np.where(pl > 0.3)[0].max()) if (pl > 0.3).any() else -1

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: patch-choice map over (time, energy).
    cmap = ListedColormap(["#4c72b0", "#55a868", "#c44e52"])     # safe, rich, wild
    axL.pcolormesh(np.arange(DAY), e, choice.T, cmap=cmap, vmin=0, vmax=2, shading="nearest")
    axL.axhline(R, color="white", lw=1.2, ls="--")
    axL.text(0.5, R + 0.015, "requirement R", color="white", fontsize=9)
    axL.set_xlabel("Time of day (0 = dawn → dusk)")
    axL.set_ylabel("Energy reserve")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.colors]
    axL.legend(handles, ["Safe (low variance)", "Rich (high mean)", "Wild (high variance)"],
               loc="upper left", frameon=True, framealpha=0.85)

    # Right: P(leave a depleted patch) over the day -- the deadline crash.
    t = np.arange(DAY)
    greys = ["0.0", "0.45", "0.7"]
    for tr, g in zip(travels, greys, strict=True):
        axR.plot(t, p_leave[tr], color=g, lw=2.0, marker="o", ms=3,
                 label=f"Travel cost = {tr} steps")
        axR.axvline(DAY - tr, color=g, lw=1.0, ls=":")
    axR.set_xlabel("Time of day (0 = dawn → dusk)")
    axR.set_ylabel("P(leave a depleting patch)")
    axR.set_ylim(-0.03, 1.05)
    axR.legend(loc="lower left", frameon=False)
    axR.text(DAY - 4.2, 0.5, "leaving stops at\nt ≈ day − travel", color="0.3", fontsize=9,
             ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "multi_patch.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Regime shares for the summary (fraction of the day-step x energy grid in each patch).
    shares = [float((choice == j).mean()) for j in range(3)]
    (FIG / "multi_patch_summary.json").write_text(json.dumps(
        {"night_requirement": R, "patch_shares_safe_rich_wild": shares,
         "p_leave": {str(k): v.tolist() for k, v in p_leave.items()},
         "last_leave_step": {str(k): v for k, v in last_leave.items()},
         "day_steps": DAY}, indent=2))

    print(f"Night requirement R = {R:.2f}.")
    print(f"Patch shares over the (time x energy) grid -- safe {shares[0]:.2f}, "
          f"rich {shares[1]:.2f}, wild {shares[2]:.2f}.")
    print("Finite-horizon MVT: last step the forager still leaves a depleted patch, vs travel:")
    for tr in travels:
        print(f"  travel {tr}: last leave at t = {last_leave[tr]}  (day - travel = {DAY - tr})")
    print(f"Saved {FIG/'multi_patch.png'}")


if __name__ == "__main__":
    main()
