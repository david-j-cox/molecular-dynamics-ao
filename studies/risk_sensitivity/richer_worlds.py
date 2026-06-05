"""Richer worlds: continuous outcomes, and the energy-budget rule extended to skewness.

Every risk result so far used a two-point gamble {mean +/- w}. That is enough to derive the
energy-budget rule, but it has two costs: a discretization artifact (the "reachability comb" --
the requirement can only be hit by whole numbers of identical lucky draws, so the band edges
show a sawtooth), and it cannot ask about the SHAPE of risk beyond variance. A continuous
outcome distribution (survival.skewed_outcomes -- a standardized, warped normal with chosen
mean, std, and skew) fixes both.

Two results:

1. Continuous outcomes remove the comb. The safe-suffices (upper) edge of the risk-prone band,
   over the day, is a sawtooth for the two-point gamble and perfectly smooth for a continuous
   gamble of the same mean and variance -- and lands on the same dusk requirement. The
   energy-budget band is a property of survival, not of the discretization.

2. The energy-budget rule extends to the THIRD moment. At FIXED mean and variance -- where
   mean-variance risk theory predicts indifference -- the survival-optimal policy is NOT
   indifferent to skew, and its preference REVERSES at the requirement, exactly as the variance
   preference does:
     - BELOW the requirement (building the buffer, with time): it prefers NEGATIVE skew --
       frequent small gains that climb steadily toward R beat the all-or-nothing lottery.
     - ABOVE the requirement (already safe): it prefers POSITIVE skew -- i.e. it avoids negative
       skew, the rare catastrophe that is the only thing that can sink a comfortable organism.
   (This is the molar average; in the narrow near-deadline corner the positive-skew lottery can
   still win -- the variance lifeline of the sun-variance study. Here variance is held fixed.)

Run:   python studies/risk_sensitivity/richer_worlds.py
Saves: studies/risk_sensitivity/figures/richer_worlds.png + richer_worlds_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md.survival import (  # noqa: E402
    outcome_moments,
    risk_threshold,
    skewed_outcomes,
    survival_dp,
)

FIG = Path(__file__).parent / "figures"
DAY, NIGHT, METAB, M, SD = 24, 24, 0.03, 0.05, 0.06
N_EGRID = 1601                          # fine grid so the comb is a real signal, not grid noise
R = NIGHT * METAB
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 12, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 10})


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    safe = [(1.0, M)]

    # --- Result 1: continuous outcomes remove the reachability comb -------------------------
    two_point = [(0.5, M - SD), (0.5, M + SD)]
    continuous = skewed_outcomes(M, SD, 0.0)             # symmetric continuous, same mean/var
    edge_tp = risk_threshold(survival_dp(safe, two_point, DAY, NIGHT, METAB, n_egrid=N_EGRID))
    edge_co = risk_threshold(survival_dp(safe, continuous, DAY, NIGHT, METAB, n_egrid=N_EGRID))
    rough_tp = float(np.nanstd(np.diff(edge_tp)))
    rough_co = float(np.nanstd(np.diff(edge_co)))
    t = np.arange(DAY)

    # --- Result 2: skew preference reverses at the requirement ------------------------------
    skews = np.linspace(-1.2, 1.2, 13)
    skewness, below_adv, above_adv = [], [], []
    for sk in skews:
        g = skewed_outcomes(M, SD, sk)
        _, _, sval = outcome_moments(g)
        res = survival_dp(safe, g, DAY, NIGHT, METAB, n_egrid=N_EGRID)
        e = res["energy"]
        adv = res["q_risky"] - res["q_safe"]             # gamble's survival edge over safe
        below = (e > 0.05) & (e < R)
        above = (e > R) & (e < 0.98)
        skewness.append(sval)
        below_adv.append(float(adv[:, below].mean()))
        above_adv.append(float(adv[:, above].mean()))
    skewness = np.array(skewness)
    below_adv = np.array(below_adv)
    above_adv = np.array(above_adv)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))

    # Left: the comb vanishes under a continuous distribution.
    axL.plot(t, edge_tp, color="0.55", lw=1.8, marker="s", ms=4,
             label=f"Two-point gamble (roughness {rough_tp:.3f})")
    axL.plot(t, edge_co, color="black", lw=2.0, marker="o", ms=4,
             label=f"Continuous gamble (roughness {rough_co:.3f})")
    axL.axhline(R, color="0.8", lw=1.0, ls=":", zorder=0)
    axL.text(0.3, R + 0.01, "night requirement R", color="0.5", fontsize=9)
    axL.set_xlabel("Time of day (0 = dawn → dusk)")
    axL.set_ylabel("Safe-suffices edge of the risk-prone band")
    axL.legend(loc="lower right", frameon=False)

    # Right: skew preference reverses at the requirement.
    axR.axhline(0.0, color="0.6", lw=1.0)
    axR.axvline(0.0, color="0.85", lw=1.0, zorder=0)
    axR.plot(skewness, below_adv * 1000, color="black", lw=2.0, marker="o", ms=4,
             label="Below requirement (desperate)")
    axR.plot(skewness, above_adv * 1000, color="0.55", lw=2.0, ls="--", marker="s", ms=4,
             label="Above requirement (comfortable)")
    axR.set_xlabel("Skewness of the gamble (← disaster   lottery →)")
    axR.set_ylabel("Gamble's survival edge over safe\n(×10⁻³, + = gamble preferred)")
    axR.legend(loc="upper center", frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "richer_worlds.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    (FIG / "richer_worlds_summary.json").write_text(json.dumps(
        {"roughness_two_point": rough_tp, "roughness_continuous": rough_co,
         "dusk_edge_two_point": float(np.nanmax(edge_tp[-3:])),
         "dusk_edge_continuous": float(np.nanmax(edge_co[-3:])),
         "skewness": skewness.tolist(), "below_adv": below_adv.tolist(),
         "above_adv": above_adv.tolist(), "night_requirement": R}, indent=2))

    print(f"Comb: upper-edge roughness two-point {rough_tp:.4f} vs continuous {rough_co:.4f} "
          f"(same dusk edge {np.nanmax(edge_tp[-3:]):.3f} / {np.nanmax(edge_co[-3:]):.3f}).")
    print("Skew preference (gamble edge over safe, averaged over the regime):")
    print(f"  below R: left-skew {below_adv[0]*1000:+.2f} -> right-skew {below_adv[-1]*1000:+.2f} "
          "(×10^-3): prefers NEGATIVE skew (steady small gains).")
    print(f"  above R: left-skew {above_adv[0]*1000:+.2f} -> right-skew {above_adv[-1]*1000:+.2f} "
          "(×10^-3): prefers POSITIVE skew (avoids the rare catastrophe).")
    print(f"Saved {FIG/'richer_worlds.png'}")


if __name__ == "__main__":
    main()
