"""exp055 -- sequence-based parameter recovery and the identifiability of the requirement R.

Roadmap 2.1. Houston & Rosenstrom (2024) argue the field's evidence on state-dependent
risk sensitivity is mixed because the diagnostic signal is in the SEQUENCE of choices, not the
aggregate preference: "the devil is in the sequence". We make that exact inside the survival DP.

A population chooses risky with probability sigma(beta*(q_risky - q_safe)) at its current (reserve,
time). The latent parameters are the overnight requirement R (fixing WHERE q_risky - q_safe flips
sign, i.e. the reserve at which risk preference reverses), the daytime metabolism m_day (the reserve
drift away from the boundary), and the softmax temperature beta (HOW SHARP the flip is). q depends
only on (m_day, R) via the DP; beta enters only the choice rule. We recover theta two ways from the
SAME behaving population:

  SEQUENCE  -- keep every (reserve, time, choice): the analyst sees the conditional choice curve
               P(risky | reserve); its crossing point gives R (a location), its slope gives beta (a
               scale). Two features of one curve => the parameters separate.
  AGGREGATE -- keep only the overall risky-choice proportion + the reserve-occupancy histogram (drop
               the choice-to-reserve pairing): one scalar prediction, so any (R, beta) on its level
               set fits equally => a ridge.

Results (printed below + figure exp055_identifiability.png):
  A. Conditional choice curve: empirical P(risky | reserve) is a BUMP -- indifferent (0.5) at ruin,
     risk-prone below R, risk-averse approaching R, fading to 0.5 deep-safe. The downward crossing
     or trough localizes R, its steepness localizes beta -- the signature the sequence keeps.
  B. Recovery: ML refits of simulated data recover (R, m_day, beta) almost exactly from sequences
     (true-vs-recovered on the diagonal); the aggregate readout recovers the marginal proportion but
     NOT R (recovered R scatters off the diagonal).
  C. Profile likelihood of R: sharply peaked at R_true from sequences (identifiable); flat across a
     wide R range from aggregate (non-identifiable) -- a mechanistic account of why aggregate-
     preference evidence is mixed.
  D. Fisher information by reserve: the information about R is concentrated at the requirement
     boundary; the information about m_day is broader and sits at lower reserves -- choices near R
     localize R, choices far from it localize metabolism.

Recovery and confusion-matrix framing follow Wilson & Collins 2019; the named open problem is
Houston & Rosenstrom 2024. The companion learning-rate vs softmax-temperature degeneracy is exp056.

Run:  python experiments/exp055_sequence_identifiability.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md import recovery as rc  # noqa: E402

FIG = Path("outputs/figures")
TRUE = {"m_day": 0.04, "R": 0.45, "beta": 10.0}
N_ORG, N_CYCLES = 140, 18


def empirical_bump(data, n_bins=22, min_count=30):
    """Empirical P(risky | reserve), pooled over the whole day, binned by reserve.

    The conditional choice curve is a BUMP, not a sigmoid: indifferent (0.5) at low reserve where
    options mean near-certain death, risk-prone in the band below the threshold, risk-averse as the
    reserve approaches the requirement R, fading back to indifference deep-safe; the downward cross
    and the risk-averse trough localize R; the steepness localizes beta.
    """
    e, c = data["reserve"], data["chose_risky"]
    edges = np.linspace(0, rc.CAP, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    b = np.clip(np.digitize(e, edges) - 1, 0, n_bins - 1)
    frac = np.full(n_bins, np.nan)
    cnt = np.zeros(n_bins)
    for i in range(n_bins):
        sel = b == i
        cnt[i] = sel.sum()
        if cnt[i] >= min_count:
            frac[i] = c[sel].mean()
    return centers, frac, cnt


def model_bump(data, fit, n_bins=60):
    """Fitted model P(risky | reserve), occupancy-weighted over the day (averaging tames comb)."""
    res = rc.solve(fit["m_day"], fit["R"])
    e = res["energy"]
    d = res["q_risky"] - res["q_safe"]
    edges = np.linspace(0, rc.CAP, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    b = np.clip(np.digitize(data["reserve"], edges) - 1, 0, n_bins - 1)
    num = np.zeros(n_bins)
    den = np.zeros(n_bins)
    for t in range(rc.DAY):
        pt = 1.0 / (1.0 + np.exp(-fit["beta"] * np.interp(centers, e, d[t])))
        w = np.bincount(b[data["tstep"] == t], minlength=n_bins).astype(float)
        num += w * pt
        den += w
    curve = np.where(den > 0, num / np.maximum(den, 1), np.nan)
    return centers, np.where(den > 50, curve, np.nan)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    data = rc.simulate_choice_sequences(TRUE["m_day"], TRUE["R"], TRUE["beta"],
                                        n_org=N_ORG, n_cycles=N_CYCLES, seed=1)
    n = data["reserve"].size
    print(f"Generated {n} choices from theta={TRUE}; overall risky fraction "
          f"{data['chose_risky'].mean():.3f}\n")

    # --- A. recover from the full sequence vs the aggregate, on this dataset --------------------
    fs = rc.fit_sequence(data)
    fa = rc.fit_aggregate(data)
    print("Recovery on the reference dataset (true -> sequence | aggregate):")
    for k in ("R", "m_day", "beta"):
        print(f"  {k:6s} {TRUE[k]:7.3f} -> {fs[k]:7.3f} | {fa[k]:7.3f}")

    # --- B. recovery across a grid of true R (x2 beta), sequence vs aggregate -------------------
    R_truth = np.array([0.36, 0.42, 0.48, 0.54])
    betas = [8.0, 16.0]
    rec = {"seq": {b: [] for b in betas}, "agg": {b: [] for b in betas}}
    for b in betas:
        for j, Rt in enumerate(R_truth):
            d = rc.simulate_choice_sequences(TRUE["m_day"], float(Rt), b,
                                             n_org=N_ORG, n_cycles=N_CYCLES, seed=100 + j)
            rec["seq"][b].append(rc.fit_sequence(d)["R"])
            rec["agg"][b].append(rc.fit_aggregate(d)["R"])
    seq_all_t = np.concatenate([R_truth for _ in betas])
    seq_all_r = np.concatenate([rec["seq"][b] for b in betas])
    agg_all_r = np.concatenate([rec["agg"][b] for b in betas])
    corr_seq = np.corrcoef(seq_all_t, seq_all_r)[0, 1]
    corr_agg = np.corrcoef(seq_all_t, agg_all_r)[0, 1]
    rmse_seq = float(np.sqrt(np.mean((seq_all_t - seq_all_r) ** 2)))
    rmse_agg = float(np.sqrt(np.mean((seq_all_t - agg_all_r) ** 2)))
    print(f"\nR recovery across the grid:  sequence r={corr_seq:.3f} rmse={rmse_seq:.3f}"
          f"   aggregate r={corr_agg:.3f} rmse={rmse_agg:.3f}")

    # --- C. profile likelihood of R -------------------------------------------------------------
    R_grid = np.linspace(0.30, 0.62, 33)
    prof_seq = rc.profile_loglik_R(data, R_grid, TRUE, aggregate=False)
    prof_agg = rc.profile_loglik_R(data, R_grid, TRUE, aggregate=True)

    def ci_width(prof):                       # width of the {profile > -1.92} set (95% chi2_1)
        ok = R_grid[prof > -1.92]
        return float(ok.max() - ok.min()) if ok.size else 0.0
    w_seq, w_agg = ci_width(prof_seq), ci_width(prof_agg)
    print(f"\n95% profile-likelihood width on R:  sequence {w_seq:.3f}  |  aggregate {w_agg:.3f}"
          f"   (grid step {R_grid[1]-R_grid[0]:.3f})")

    # --- D. Fisher information by reserve -------------------------------------------------------
    # Sum per-cell information over the smooth interior day-steps (exclude the last 2 dusk steps,
    # where the continuation value is a near-step function -- the reachability comb, as in exp054),
    # and lightly smooth across reserve so the discretization teeth do not dominate the eye.
    def smooth(v, k=3):
        ker = np.ones(k) / k
        return np.convolve(v, ker, mode="same")
    fcurves = {}
    for p in ("R", "m_day"):
        centers, _, info = rc.fisher_info_map(TRUE["m_day"], TRUE["R"], TRUE["beta"], param=p,
                                              n_bins=28)
        marg = smooth(info[: rc.DAY - 2].sum(0))
        fcurves[p] = marg / marg.max()
    cR = (centers * fcurves["R"]).sum() / fcurves["R"].sum()
    cM = (centers * fcurves["m_day"]).sum() / fcurves["m_day"].sum()
    print(f"\nFisher-info centroid reserve:  R-info {cR:.3f} (at the requirement {TRUE['R']:.2f})"
          f"   m_day-info {cM:.3f} (lower / broader)")

    # --- figure ---------------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.2))

    # A. conditional choice curve (the bump) + fitted model overlay
    centers_c, frac, cnt = empirical_bump(data, n_bins=22)
    mc, mp = model_bump(data, fs)
    good = ~np.isnan(frac)
    ax[0, 0].axvspan(0, TRUE["R"], color="tab:green", alpha=0.06)
    ax[0, 0].scatter(centers_c[good], frac[good], s=6 + 45 * cnt[good] / cnt.max(),
                     color="tab:blue", alpha=0.75, label="empirical choices", zorder=3)
    ax[0, 0].plot(mc, mp, color="tab:red", lw=2, label="fitted model")
    ax[0, 0].axvline(TRUE["R"], color="0.3", ls=":", lw=1.4, label=f"$R={TRUE['R']:.2f}$")
    ax[0, 0].axhline(0.5, color="0.7", lw=0.8)
    ax[0, 0].text(0.16, 0.62, "risk-\nprone", fontsize=8, color="tab:green", ha="center")
    ax[0, 0].set_xlim(0, 1)
    ax[0, 0].set_ylim(0.15, 0.75)
    ax[0, 0].set_xlabel("reserve $E$ at choice")
    ax[0, 0].set_ylabel("P(choose risky)")
    ax[0, 0].set_title("A. The conditional choice curve (a bump):\nrisk-prone below $R$, averse "
                       "approaching it -- what the sequence keeps")
    ax[0, 0].legend(fontsize=8, loc="upper right")

    # B. recovery scatter: true R vs recovered R, sequence vs aggregate
    mk = {8.0: "o", 16.0: "s"}
    for b in betas:
        ax[0, 1].scatter(R_truth, rec["seq"][b], marker=mk[b], color="tab:green", s=55,
                         label=f"sequence ($\\beta$={b:g})", zorder=3)
        ax[0, 1].scatter(R_truth, rec["agg"][b], marker=mk[b], color="tab:red", s=55,
                         facecolors="none", label=f"aggregate ($\\beta$={b:g})", zorder=3)
    lim = [0.30, 0.60]
    ax[0, 1].plot(lim, lim, color="0.5", ls="--", lw=1, label="identity")
    ax[0, 1].set_xlim(*lim)
    ax[0, 1].set_ylim(0.28, 0.66)
    ax[0, 1].set_xlabel("true requirement $R$")
    ax[0, 1].set_ylabel("recovered $R$")
    ax[0, 1].set_title(f"B. Recovery of $R$\nsequence $r$={corr_seq:.2f} | "
                       f"aggregate $r$={corr_agg:.2f}")
    ax[0, 1].legend(fontsize=7.5, loc="upper left")

    # C. profile likelihood of R: sharp (sequence) vs flat (aggregate)
    ax[1, 0].plot(R_grid, prof_seq, color="tab:green", lw=2, label="sequence")
    ax[1, 0].plot(R_grid, prof_agg, color="tab:red", lw=2, label="aggregate")
    ax[1, 0].axvline(TRUE["R"], color="0.3", ls=":", lw=1.4)
    ax[1, 0].axhline(-1.92, color="0.7", lw=0.8, ls="--", label="95% (−1.92)")
    ax[1, 0].set_ylim(-25, 1)
    ax[1, 0].set_xlabel("requirement $R$ (others profiled out)")
    ax[1, 0].set_ylabel("profile log-likelihood (rel. max)")
    ax[1, 0].set_title("C. Profile likelihood of $R$\nsharp from sequences, flat from aggregate")
    ax[1, 0].legend(fontsize=8, loc="lower right")

    # D. Fisher information by reserve: R vs m_day
    ax[1, 1].plot(centers, fcurves["R"], color="tab:purple", lw=2, label="info about $R$")
    ax[1, 1].plot(centers, fcurves["m_day"], color="tab:orange", lw=2, label="info about $m_{day}$")
    ax[1, 1].axvline(TRUE["R"], color="0.3", ls=":", lw=1.4, label=f"$R={TRUE['R']:.2f}$")
    ax[1, 1].set_xlim(0, 1)
    ax[1, 1].set_xlabel("reserve $E$ at choice")
    ax[1, 1].set_ylabel("Fisher information (each $\\div$ its max)")
    ax[1, 1].set_title("D. Where each parameter is identifiable\n$R$ at the boundary; "
                       "$m_{day}$ broader / lower")
    ax[1, 1].legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out = FIG / "exp055_identifiability.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
