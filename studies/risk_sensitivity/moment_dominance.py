"""Moment-dominance phase diagram and the "survival => all-moments" theorem.

Every risk result so far reads as a SECOND-moment story (the energy-budget rule = variance
preference reverses at the requirement) with a THIRD-moment extension (the skew reversal of
richer_worlds.py). This study shows both are instances of one fact about the EMERGENT survival
value, and states the general theorem.

The theorem. Foraging is choosing an outcome distribution; the safe option lands the organism
deterministically at the post-decision energy e* = reserve + mean - metabolism, so a mean-
preserving spread's survival advantage over safe is exactly the Taylor (moment) expansion of the
DP's continuation value V about e*:

    adv ~ (1/2) V''(e*) * mu2  +  (1/6) V'''(e*) * mu3  +  (1/24) V''''(e*) * mu4  + ...

So variance preference is governed by V'' (curvature), skew preference by V''' (PRUDENCE), and
kurtosis preference by V'''' (TEMPERANCE). A survival/ruin-minimization objective is a whole-
distribution functional: the optimal policy is therefore sensitive to ALL moments, and which one
governs a given state is set by the sign and size of a successive derivative of one emergent value
function. The energy-budget rule, the skew reversal, and (preliminarily) a temperance reversal are
the SAME phenomenon -- successive derivatives of V changing sign across the requirement.

We measure each moment's preference field directly from the trusted DP by an ISOLATED mean-
preserving spread (no finite differencing of a gridded value), matched in all lower moments:
  - variance: a symmetric gamble                         sign(field) = sign V''
  - skew:     half-difference of +skew and -skew gambles sign(field) = sign V'''  (prudence)
  - kurtosis: high- minus low-kurtosis 3-point gamble    sign(field) = sign V'''' (temperance)

Results:
  1. The variance field reverses sign EXACTLY at the optimal policy threshold (corr 1.00,
     mean|diff| ~ one grid cell), rising through the day to the night requirement R. The energy-
     budget rule IS the inflection (V'' = 0) of the emergent survival value.
  2. The skew field reverses across the same threshold: below the requirement the organism prefers
     NEGATIVE skew (steady small gains that climb toward R), above it prefers POSITIVE skew (avoid
     the rare catastrophe). Prudence is the sign of V'''.
  3. The kurtosis field carries the temperance term; here it is weak and does not cleanly reverse
     (highest-order, smallest signal, and the 3-point spread is only approximately local). The
     controlled moment-matched treatment is roadmap item 1.3.

Must-cite: Eeckhoudt & Schlesinger (2006, AER) -- prudence/temperance as moment preferences via
utility derivatives; Menezes, Geiss & Tressler (1980, AER) -- downside risk = skewness; ruin-theory
framing (Lundberg; Bayraktar, inverse survival function). The bridge foraging <-> ruin theory <->
prudence/temperance appears unstated in the literature; that bridge is the theorem.

Run:   python studies/risk_sensitivity/moment_dominance.py
Saves: studies/risk_sensitivity/figures/moment_dominance.png + moment_dominance_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md.survival import field_zero_crossing, moment_preference_fields  # noqa: E402

FIG = Path(__file__).parent / "figures"
DAY, NIGHT, METAB = 24, 24, 0.03
N_EGRID = 1201
R = NIGHT * METAB
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
                     "legend.fontsize": 9, "axes.titlesize": 11})


def _regime_means(field, e, thr):
    """Field averaged below and above the per-day-step threshold (the reversal readout)."""
    below, above = [], []
    for t in range(field.shape[0]):
        th = thr[t] if not np.isnan(thr[t]) else R
        b = (e > 0.05) & (e < th - 0.02)
        a = (e > th + 0.02) & (e < 0.95)
        if b.sum():
            below.append(field[t, b].mean())
        if a.sum():
            above.append(field[t, a].mean())
    return float(np.mean(below)), float(np.mean(above))


def _field_panel(ax, field, e, thr, title):
    m = np.nanmax(np.abs(field[:, (e > 0.03) & (e < 0.97)]))
    im = ax.imshow(field.T, origin="lower", aspect="auto", cmap="RdBu_r",
                   vmin=-m, vmax=m, extent=[0, field.shape[0] - 1, e[0], e[-1]])
    t = np.arange(field.shape[0])
    ax.plot(t, thr, color="black", lw=1.8, label="optimal threshold")
    ax.axhline(R, color="0.25", lw=1.0, ls=":")
    ax.text(0.4, R + 0.015, "R", color="0.2", fontsize=9)
    ax.set_xlabel("time of day (0 = dawn → dusk)")
    ax.set_ylabel("reserve E")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    return im


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    f = moment_preference_fields(DAY, NIGHT, METAB, n_egrid=N_EGRID)
    e = f["energy"]
    thr = f["threshold"]
    zc_var = field_zero_crossing(f["variance"], e)
    zc_skew = field_zero_crossing(f["skew"], e)

    ok = ~np.isnan(thr) & ~np.isnan(zc_var)
    var_diff = float(np.nanmean(np.abs(zc_var[ok] - thr[ok])))
    var_corr = float(np.corrcoef(zc_var[ok], thr[ok])[0, 1])
    var_bl, var_ab = _regime_means(f["variance"], e, thr)
    skew_bl, skew_ab = _regime_means(f["skew"], e, thr)
    kurt_bl, kurt_ab = _regime_means(f["kurtosis"], e, thr)

    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.0))
    im0 = _field_panel(ax[0, 0], f["variance"], e, thr,
                       "Variance preference  (∝ V″, curvature)\nthe energy-budget rule")
    ax[0, 0].legend(loc="lower right", frameon=False)
    fig.colorbar(im0, ax=ax[0, 0], label="gamble's survival edge\n(red + = risk-prone)")
    im1 = _field_panel(ax[0, 1], f["skew"], e, thr,
                       "Skew preference  (∝ V‴, prudence)\nneg-skew below R, pos-skew above")
    fig.colorbar(im1, ax=ax[0, 1], label="pos-skew edge\n(red + = lottery preferred)")
    im2 = _field_panel(ax[1, 0], f["kurtosis"], e, thr,
                       "Kurtosis preference  (∝ V⁗, temperance)\nweak highest-order term — see 1.3")
    fig.colorbar(im2, ax=ax[1, 0], label="high-kurt edge")

    axv = ax[1, 1]
    t = np.arange(DAY)
    axv.plot(t, thr, color="black", lw=2.2, label="optimal DP threshold")
    axv.plot(t, zc_var, "o", color="tab:red", ms=5,
             label=f"variance-field reversal (r={var_corr:.2f})")
    axv.plot(t, zc_skew, "s", color="0.55", ms=4, label="skew-field reversal")
    axv.axhline(R, color="0.25", lw=1.0, ls=":")
    axv.text(0.4, R + 0.012, "night requirement R", color="0.2", fontsize=9)
    axv.set_xlabel("time of day (0 = dawn → dusk)")
    axv.set_ylabel("reserve E of preference reversal")
    axv.set_ylim(0, 1)
    axv.set_title("Each moment's reversal sits on the optimal threshold\n"
                  "(survival value's inflection ladder)")
    axv.legend(loc="upper left", frameon=False)
    fig.suptitle("Moment-dominance: a survival objective is sensitive to ALL moments, "
                 "each governed by a successive derivative of one emergent value",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / "moment_dominance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    (FIG / "moment_dominance_summary.json").write_text(json.dumps(
        {"night_requirement": R, "variance_reversal_vs_threshold_meanabsdiff": var_diff,
         "variance_reversal_vs_threshold_corr": var_corr,
         "variance_below": var_bl, "variance_above": var_ab,
         "skew_below": skew_bl, "skew_above": skew_ab,
         "kurtosis_below": kurt_bl, "kurtosis_above": kurt_ab,
         "dusk_variance_reversal": float(np.nanmax(zc_var[-4:])),
         "dusk_threshold": float(np.nanmax(thr[-4:]))}, indent=2))

    print("Moment-dominance: each moment's preference is a derivative of the emergent value.")
    print(f"  variance (V''):  reversal tracks the optimal threshold "
          f"corr={var_corr:.3f}, mean|diff|={var_diff:.4f} (rises to R={R:.2f} at dusk).")
    print(f"     below R {var_bl*1e3:+.2f} (risk-prone)  ->  above R {var_ab*1e3:+.2f} "
          "(risk-averse)  [x1e-3]")
    print(f"  skew (V'''):     below R {skew_bl*1e3:+.2f} (prefers NEG skew)  ->  "
          f"above R {skew_ab*1e3:+.2f} (prefers POS skew)  [x1e-3]")
    print(f"  kurtosis (V''''):below R {kurt_bl*1e3:+.2f}  ->  above R {kurt_ab*1e3:+.2f}  "
          "[x1e-3]  (weak; controlled test = roadmap 1.3)")
    print(f"Saved {FIG/'moment_dominance.png'}")


if __name__ == "__main__":
    main()
