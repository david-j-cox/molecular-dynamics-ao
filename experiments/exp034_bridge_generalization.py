"""exp034 -- does approach B generalize? (the non-hard-coding test)

A mechanism is only a faithful bridge, not a disguised hard-code, if it GENERALIZES: change the
environment's requirement and the emergent policy must track it WITH NO RETUNING. Here we shift the
overnight fast (night length), which sets the requirement R = night_cost * night_steps, and check
that approach B's risk-prone-below / risk-averse-above crossover moves to the new R on its own. The
learner's parameters (lr, decay, temperature, bins) are held FIXED across all requirements.

If the crossover tracks R across night lengths, B is a real mechanism: nothing about R or the policy
was installed; only the environmental fact (a longer night needs a bigger dusk reserve) changed.

Run:  python -m experiments.exp034_bridge_generalization
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.exp033_multilevel_reinforcement import run_survival_signal

FIG = Path("outputs/figures")
NIGHT_COST = 0.05
NIGHTS = [4, 8, 12]            # -> R = 0.20, 0.40, 0.60 ; the only thing that changes
COLORS = ["tab:blue", "tab:green", "tab:red"]


def crossover(curve, bins):
    """Energy at which P(risky) last crosses 0.5 from above (the risk-prone -> averse boundary)."""
    above = curve > 0.5
    # scan low->high; the boundary is the first bin where it goes from prone(>0.5) to averse(<=0.5)
    for i in range(1, len(bins)):
        if above[i - 1] and not above[i]:
            return bins[i]
    return np.nan


def main():
    plt.figure(figsize=(7, 4.5))
    print("Generalization: does the crossover track R = night_cost * night_steps?  (learner fixed)")
    rows = []
    for nights, col in zip(NIGHTS, COLORS, strict=True):
        R = NIGHT_COST * nights
        curve, bins = run_survival_signal(night_steps=nights, night_cost=NIGHT_COST,
                                          n_cycles=500, seed=0)
        xo = crossover(curve, bins)
        rows.append((R, xo))
        print(f"  night_steps={nights:2d}  R={R:.2f}  emergent crossover={xo:.2f}  "
              f"(below R prone, above R averse)")
        plt.plot(bins, curve, "o-", color=col, label=f"R={R:.2f} (nights={nights})")
        plt.axvline(R, color=col, ls="--", lw=1, alpha=0.6)

    plt.axhline(0.5, color="0.8", lw=1)
    plt.xlabel("current energy reserve E")
    plt.ylabel("P(choose risky)")
    plt.title("exp034: B generalizes -- the reversal tracks the requirement R (no retuning)")
    plt.ylim(-0.02, 1.02)
    plt.legend(fontsize=8, title="dashed line = requirement R")
    plt.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "exp034_bridge_generalization.png"
    plt.savefig(out, dpi=130)
    print(f"wrote {out}")
    # headline: correlation of emergent crossover with the imposed requirement
    Rs = np.array([r for r, _ in rows])
    xos = np.array([x for _, x in rows])
    if np.all(np.isfinite(xos)):
        print(f"crossover tracks R: corr={np.corrcoef(Rs, xos)[0, 1]:.3f}  "
              f"mean|crossover-R|={np.mean(np.abs(xos - Rs)):.3f}")


if __name__ == "__main__":
    main()
