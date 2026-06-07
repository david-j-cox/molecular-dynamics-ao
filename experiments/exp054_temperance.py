"""exp054 -- temperance: the SIGN of the kurtosis preference, derived from a survival objective.

Roadmap 1.3. The moment decomposition (sec:moments) showed variance preference (governed by V'') and
skew preference (V''') each reverse at the requirement R. The fourth term is TEMPERANCE, governed by
V'''': a preference over kurtosis (fat tails) at fixed mean, variance, and skew. In economics
temperance is the AXIOM u''''<0 (kurtosis aversion; Eeckhoudt & Schlesinger 2006; measured by Deck &
Schlesinger 2014). Here we DERIVE its sign from survival.

The coarse symmetric three-point gamble of 1.1 could not isolate the fourth moment (its outliers are
too non-local), so the temperance term read as weak and ambiguous. This uses the proper >=5-point
moment-matching generator (survival.kurtosis_outcomes): a symmetric 5-point gamble with exact mean,
variance, and zero skew, varying ONLY kurtosis. The kurtosis preference field is the survival
advantage of a high-kurtosis over a low-kurtosis gamble (both matched in mean/variance/skew),
measured directly off the DP; its sign is the sign of V''''. We exclude the final two (dusk)
day-steps, where the continuation value is a near-step function (the reachability comb) and the
smooth moment expansion does not apply.

Result. The survival objective is robustly TEMPERATE: the kurtosis preference is negative (kurtosis
AVERSION) on BOTH sides of the requirement, of a magnitude comparable to the skew effect, and
vanishes in the deep-safe region (the curvature control). So, unlike variance and skew, kurtosis
preference does NOT reverse at R: the moment ladder is two reversals (V'', V''') plus one
single-signed aversion (V''''). This derives the sign of temperance (matching the economics axiom)
from survival rather than assuming it, and refines the roadmap's guess of a kurtosis reversal: there
is none. Notably the desperate (below-R) forager is variance-SEEKING yet kurtosis-AVERSE: it wants
spread (a real chance to reach R) but not fat tails, whose downside and peakedness do not pay.

Run:  python experiments/exp054_temperance.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md.survival import (  # noqa: E402
    kurtosis_outcomes,
    risk_threshold,
    skewed_outcomes,
    survival_dp,
)

FIG = Path("outputs/figures")
S, DAY, NIGHT, METAB = 0.05, 14, 16, 0.03
R = NIGHT * METAB
SD, NG = 0.03, 1601
DROP = 2                      # exclude the last DROP (dusk) step-function slices


def _adv(out):
    res = survival_dp([(1.0, S)], out, DAY, NIGHT, METAB, n_egrid=NG)
    return res["q_risky"] - res["q_safe"], res


def _regimes(field, e, thr):
    bl, ab, deep = [], [], []
    for t in range(DAY - DROP):
        th = thr[t] if not np.isnan(thr[t]) else R
        b = (e > 0.05) & (e < th - 0.02)
        a = (e > th + 0.02) & (e < 0.85)
        d = e > 0.9
        if b.sum():
            bl.append(field[t, b].mean())
        if a.sum():
            ab.append(field[t, a].mean())
        if d.sum():
            deep.append(field[t, d].mean())
    return np.nanmean(bl) * 1e3, np.nanmean(ab) * 1e3, np.nanmean(deep) * 1e3


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    a_sym, res = _adv(skewed_outcomes(S, SD, 0.0))
    e, thr = res["energy"], risk_threshold(res)
    variance = a_sym
    skew = 0.5 * (_adv(skewed_outcomes(S, SD, 0.9))[0] - _adv(skewed_outcomes(S, SD, -0.9))[0])
    kurt = _adv(kurtosis_outcomes(S, SD, 7.0))[0] - _adv(kurtosis_outcomes(S, SD, 1.5))[0]

    rows = [("variance ($V''$)", variance), ("skew ($V'''$)", skew), ("kurtosis ($V''''$)", kurt)]
    print(f"Moment preferences (x1e-3), smooth interior, R={R}:\n")
    print(f"{'moment':>18}{'below R':>10}{'above R':>10}{'deep-safe':>11}   reading")
    readings = {"variance ($V''$)": "reverses (prone -> averse)",
                "skew ($V'''$)": "reverses (neg -> pos skew)",
                "kurtosis ($V''''$)": "single-signed AVERSION (temperance); no reversal"}
    summary = {}
    for name, f in rows:
        b, a, d = _regimes(f, e, thr)
        summary[name] = (b, a)
        print(f"{name:>18}{b:>+10.2f}{a:>+10.2f}{d:>+11.2f}   {readings[name]}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    mmax = np.nanmax(np.abs(kurt[: DAY - DROP, (e > 0.03) & (e < 0.88)]))
    im = ax[0].imshow(kurt[: DAY - DROP].T, origin="lower", aspect="auto", cmap="RdBu_r",
                      vmin=-mmax, vmax=mmax, extent=[0, DAY - DROP - 1, e[0], e[-1]])
    ax[0].plot(np.arange(DAY - DROP), thr[: DAY - DROP], color="black", lw=1.6, label="threshold")
    ax[0].axhline(R, color="0.3", ls=":", lw=1)
    ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("time of day (0 = dawn → dusk)")
    ax[0].set_ylabel("reserve $E$")
    ax[0].set_title("Kurtosis preference field ($\\propto V''''$):\ntemperance (blue = averse) "
                    "on both sides")
    ax[0].legend(loc="lower right", fontsize=8)
    fig.colorbar(im, ax=ax[0], label="high-kurt edge ($\\times10^{-3}$ scale)")

    labels = ["variance\n($V''$)", "skew\n($V'''$)", "kurtosis\n($V''''$)"]
    below = [summary[n][0] for n, _ in rows]
    above = [summary[n][1] for n, _ in rows]
    x = np.arange(3)
    ax[1].bar(x - 0.18, below, 0.36, label="below $R$", color="tab:red")
    ax[1].bar(x + 0.18, above, 0.36, label="above $R$", color="tab:blue")
    ax[1].axhline(0, color="0.4", lw=1)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("preference ($\\times10^{-3}$)")
    ax[1].set_title("Two moments reverse at $R$; the fourth does not\n(temperance is a sign, "
                    "not a reversal)")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    out = FIG / "exp054_temperance.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
