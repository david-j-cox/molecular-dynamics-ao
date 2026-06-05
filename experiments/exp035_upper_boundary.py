"""exp035 -- the upper boundary: where does risk-aversion ABOVE the requirement come from?

Approach B (exp033) reproduces risk-proneness BELOW the requirement, but only weak aversion above:
with a single (starvation) death boundary, survival just saturates at 1 once you clear R, so a bad
draw when well-fed is harmless and there is no positive reason to avoid variance. Real foragers are
risk-averse when fed because there is a SECOND cost to being high: a heavier reserve is slower and
more visible, raising predation (the starvation-predation trade-off; McNamara & Houston 1990).

This test adds predation as a second death source above an upper threshold x_r -- another bare
environmental FACT, not a utility -- and asks whether aversion above R sharpens. The lower
requirement R = night_cost*night_steps and the upper boundary x_r are both real; nothing about the
policy is installed.

Run:  python -m experiments.exp035_upper_boundary
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.exp033_multilevel_reinforcement import run_survival_signal

FIG = Path("outputs/figures")
R = 0.05 * 6        # lower requirement (overnight drain) = 0.30
X_R = 0.45          # upper boundary: predation strikes above this reserve
PRED_PROB = 0.20    # per-step predation probability above x_r


def main():
    single, bins = run_survival_signal(n_cycles=500, seed=0)                       # starvation only
    twob, _ = run_survival_signal(n_cycles=500, seed=0,
                                  predation_threshold=X_R, predation_prob=PRED_PROB)  # + predation

    def band(curve, lo, hi):
        return np.nanmean(curve[(bins >= lo) & (bins < hi)])

    print(f"R={R:.2f} (lower, starvation)   x_r={X_R:.2f} (upper, predation)")
    print("mean P(risky):")
    for name, c in [("single boundary (B)", single), ("two boundaries (B+predation)", twob)]:
        below = band(c, 0.0, R)
        between = band(c, R, X_R)
        above = band(c, X_R, 0.9)
        print(f"  {name:30s} below R: {below:.3f}  between R..x_r: {between:.3f}  "
              f"above x_r: {above:.3f}")

    plt.figure(figsize=(7, 4.5))
    plt.axvline(R, color="0.6", ls="--", lw=1, label=f"requirement R={R:.2f}")
    plt.axvline(X_R, color="0.6", ls=":", lw=1, label=f"upper boundary x_r={X_R:.2f}")
    plt.axhline(0.5, color="0.85", lw=1)
    plt.plot(bins, single, "o-", color="tab:gray", label="single boundary (starvation only)")
    plt.plot(bins, twob, "^-", color="tab:red", label="two boundaries (+ predation when fat)")
    plt.xlabel("current energy reserve E")
    plt.ylabel("P(choose risky)")
    plt.title("exp035: a predation upper boundary sharpens risk-aversion above R")
    plt.ylim(-0.02, 1.02)
    plt.legend(fontsize=8)
    plt.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "exp035_upper_boundary.png"
    plt.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
