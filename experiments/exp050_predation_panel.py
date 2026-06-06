"""exp050 -- predation robustness panel: does the rule survive a second death source?

The behavioral-ecology consensus (McNamara & Houston 1990; Houston, McNamara & Hutchinson 1993;
Brodin 2007) is that mass-dependent predation is co-equal with starvation in setting reserves: a
heavier animal is slower and more visible, so a high reserve is dangerous. A reviewer's must-do
check is therefore whether the energy-budget rule and the skew reversal SURVIVE, SPLIT, or INVERT
once a second death source is added. This panel answers it in the survival DP.

Economy: a moderate, winter-like cycle (day 14 < night 16, metabolism 0.03, so the requirement
R = 0.48 of capacity bites without the biologically extreme R = 0.72 of the headline run). Predation
is a per-day-step death probability above an upper reserve boundary x_r = 0.8 (survival.survival_dp
predation_threshold / predation_prob); starvation (E <= 0) is the lower boundary.

Result. Both reversals SURVIVE, and the second death source SPLITS the policy into a band:
  - variance (energy-budget rule): still risk-prone below R; the safe band R..x_r becomes strongly
    risk-averse (predation makes leaving the band lethal, sharpening the aversion); and a risk-prone
    ESCAPE regime appears above x_r, where shedding reserve to drop below the boundary is optimal.
  - skew (prudence): still negative-skew below R, positive-skew in the band, and a strong positive-
    skew preference near x_r (the distribution that mostly drifts away from the lethal boundary),
    flipping to negative skew in the escape regime (reliable downward).
The reversal at R is robust to predation; predation adds a SECOND reversal at x_r. This is the
twin-threshold reserve target (a band between starvation and predation) of the state-dependent
foraging program, recovered here as risk and skew preferences rather than as a target reserve.

Run:  python experiments/exp050_predation_panel.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md.survival import moment_preference_fields, survival_dp  # noqa: E402

FIG = Path("outputs/figures")
DAY, NIGHT, METAB, S = 14, 16, 0.03, 0.05
R = NIGHT * METAB
XR = 0.8
PP = 0.2
NG = 1201


def _regimes(field, e):
    def m(lo, hi):
        sel = (e > lo) & (e < hi)
        return float(np.nanmean(field[:, sel])) * 1e3
    return {"below R": m(0.05, R - 0.02), "band R..x_r": m(R + 0.02, XR - 0.04),
            "near x_r": m(XR - 0.10, XR - 0.01), "above x_r": m(XR + 0.02, 0.95)}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    off = moment_preference_fields(DAY, NIGHT, METAB, n_egrid=NG)
    on = moment_preference_fields(DAY, NIGHT, METAB, n_egrid=NG,
                                  predation_threshold=XR, predation_prob=PP)
    e = off["energy"]
    risky = [(0.5, 0.0), (0.5, 2 * S)]
    v_off = survival_dp([(1.0, S)], risky, DAY, NIGHT, METAB, n_egrid=NG)["value"]
    v_on = survival_dp([(1.0, S)], risky, DAY, NIGHT, METAB, n_egrid=NG,
                       predation_threshold=XR, predation_prob=PP)["value"]

    print(f"Predation panel: day {DAY}<night {NIGHT}, R={R:.2f}, predation>x_r={XR} p={PP}\n")
    for name, fld in [("variance", "variance"), ("skew", "skew")]:
        r0 = _regimes(off[fld], e)
        r1 = _regimes(on[fld], e)
        print(f"{name} preference (x1e-3):")
        print("  no predation : " + "  ".join(f"{k} {v:+.1f}" for k, v in r0.items()))
        print("  + predation  : " + "  ".join(f"{k} {v:+.1f}" for k, v in r1.items()))

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    ax[0].plot(e, v_off, color="0.55", lw=2, label="starvation only")
    ax[0].plot(e, v_on, color="tab:red", lw=2, label="+ predation above $x_r$")
    for xx, lab in [(R, "$R$"), (XR, "$x_r$")]:
        ax[0].axvline(xx, color="0.3", ls=":", lw=1)
        ax[0].text(xx + 0.01, 0.05, lab, fontsize=9)
    ax[0].set_xlabel("reserve $E$")
    ax[0].set_ylabel("survival value $V(E)$ at dawn")
    ax[0].set_title("Predation bounds the reserve: a safe band")
    ax[0].legend(fontsize=8, loc="center left")

    for col, fld, title in [(1, "variance", "Variance preference (+ predation)"),
                            (2, "skew", "Skew preference (+ predation)")]:
        f = on[fld]
        mmax = np.nanmax(np.abs(f[:, (e > 0.03) & (e < 0.95)]))
        im = ax[col].imshow(f.T, origin="lower", aspect="auto", cmap="RdBu_r",
                            vmin=-mmax, vmax=mmax, extent=[0, f.shape[0] - 1, e[0], e[-1]])
        ax[col].axhline(R, color="black", lw=1.2, ls="--")
        ax[col].axhline(XR, color="black", lw=1.2, ls=":")
        ax[col].text(0.5, R + 0.01, "R", fontsize=9)
        ax[col].text(0.5, XR + 0.01, "$x_r$", fontsize=9)
        ax[col].set_xlabel("time of day (0 = dawn → dusk)")
        ax[col].set_ylabel("reserve $E$")
        ax[col].set_title(title, fontsize=10)
        ax[col].set_ylim(0, 1)
        fig.colorbar(im, ax=ax[col])
    fig.suptitle("exp050: both reversals survive a second death source; predation splits the rule "
                 "into a band", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIG / "exp050_predation_panel.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
