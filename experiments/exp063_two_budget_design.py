"""exp063 -- the energy-budget x skew design, run as a computational experiment.

Roadmap 2.3, executed (not pre-registered). exp058 established the *logic* that separates the
survival account from fixed-utility rivals: only a survival objective makes the SKEW preference
depend on the energy budget (prefer the negatively skewed gamble when lean/below the requirement R,
the positively skewed gamble when safe/above R), whereas expected-utility and prospect theory
predict one budget-invariant skew preference. exp058 stopped at noise-free predicted probabilities.
This experiment actually runs the study: synthetic foragers, finite trials, the rival models FIT
head-to-head, and the two questions a real design must answer -- can you identify the reversal, and
how much data does it take?

One marginal-forager economy is used throughout (intake ~ metabolism, so the gamble is life-or-death
and the survival reversal is present). Every model scores the SAME gambles; the static utilities
score intakes normalized to unit mean (the utility argument's scale is a free modelling convention),
which is the only way EU/PT curvature bites at the marginal-forager scale. The survival model
(choice_models.fit_survival) scores via the DP with a free requirement R and choice temperature,
and -- unlike the static models -- reads the reserve, so it alone expresses a budget-dependent
reversal.

Panels:
  A. DESIGN SIGNATURE. P(choose +skew over -skew) vs reserve: survival is a rising curve crossing
     0.5 at R (the reversal); EU/PT are flat (budget-invariant); mean-variance sits at 0.5
     (skew-blind). Sampling choice ACROSS reserve is what exposes the survival-only dependence.
  B. IDENTIFIABILITY OF R vs NUMBER OF BUDGET LEVELS. Fitting survival to survival-generated data,
     the recovered R is imprecise with 1-2 budgets (the (R,beta) ridge of exp055: two points barely
     constrain the conditional choice curve) and sharpens as a reserve GRADIENT (3-5+ budgets)
     traces the curve. This turns exp055's "aggregate is a ridge, the state-resolved structure is
     not" into a concrete design requirement.
  C. HOW MUCH DATA to pin R. With the 5-budget gradient, the 90% recovery interval for R narrows as
     trials per cell grow -- the quantified "how much data" deliverable. (Pinning WHERE the reversal
     sits is the hard half of the problem; merely DETECTING that choice depends on the budget is the
     easy half -- see D -- because no fixed-utility model reads the reserve at all.)
  D. DETECTION vs SPECIFICITY. P(survival selected by AIC over every fixed-utility rival) vs trials
     per cell: it rises toward 1 when the truth IS survival (power) and stays near zero when the
     truth is EU or PT (specificity -- no budget-dependence to find), so the selection is a real
     signal and not a parsimony artifact of survival's smaller parameter count.

Run:  python experiments/exp063_two_budget_design.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md import choice_models as cm  # noqa: E402
from behavioral_md.survival import skewed_outcomes  # noqa: E402

FIG = Path("outputs/figures")

# --- one marginal-forager economy (intake ~ metabolism, so survival depends on the gamble) --------
DAY, NIGHT, M_DAY = 14, 16, 0.03
R_TRUE = NIGHT * M_DAY                      # = 0.48, the overnight requirement (the reversal point)
M, SD = 0.05, 0.03                          # gamble mean and std (small: a marginal forager)
SKEWS = (0.3, 0.6, 0.9)                     # |skew| of the matched mean+variance gamble pairs
BETA_S = 200.0                              # generating survival choice temperature: a noisy
#                                             gradient-follower (not a perfect maximizer), strong
#                                             enough that R is identifiable from a reserve gradient

# generating fixed-utility parameters (a clear, budget-invariant +skew preference on these gambles)
EU_GEN = dict(a=0.3, c=1.5, beta=5.0)
PT_GEN = dict(alpha=0.9, gamma=0.55, beta=4.0)

FIT_EGRID = 281                     # DP grid for fitting (coarser=faster; the generator uses 601)
DELTA_AIC = 4.0                     # "strong" evidence threshold (survival over the fixed models)
FIXED = ("ev", "mv", "eu", "pt")    # the budget-invariant rivals


def _norm(g):
    """Static utilities score intakes normalized to unit mean (free utility-argument scale)."""
    return [(p, x / M) for p, x in g]


def _pair(s):
    return skewed_outcomes(M, SD, s), skewed_outcomes(M, SD, -s)


def p_survival(A, B, reserve, R=R_TRUE, beta=BETA_S):
    return cm.survival_choice_prob(A, B, reserve, R, beta=beta, day_steps=DAY, night_steps=NIGHT,
                                   m_day=M_DAY, n_egrid=601)


def p_static(name, A, B):
    kw = EU_GEN if name == "eu" else PT_GEN if name == "pt" else {}
    return cm.predict_choice(cm.STATIC_MODELS[name], _norm(A), _norm(B), **kw)


def make_dataset(gen: str, reserves, n_per_cell: int, rng):
    """One synthetic experiment: choices at every (skew x reserve) cell from generator ``gen``.

    Returns ``(static_trials, surv_trials)`` sharing the same observed counts -- static models score
    normalized gambles (no reserve), the survival model scores raw gambles tagged with the reserve.
    """
    st, vt = [], []
    for s in SKEWS:
        A, B = _pair(s)
        for res in reserves:
            p = p_survival(A, B, res) if gen == "survival" else p_static(gen, A, B)
            k = int(rng.binomial(n_per_cell, p))
            st.append({"A": _norm(A), "B": _norm(B), "n": n_per_cell, "k": k})
            vt.append({"A": A, "B": B, "reserve": res, "n": n_per_cell, "k": k})
    return st, vt


def survival_vs_fixed(static_trials, surv_trials, restarts: int = 3):
    """Fit survival and every fixed rival; return (surv_fit, delta_aic=best_fixed_aic-surv_aic)."""
    surv = cm.fit_survival(surv_trials, n_egrid=FIT_EGRID, restarts=restarts)
    best_fixed = min(cm.fit_model(n, static_trials)["aic"] for n in FIXED)
    return surv, best_fixed - surv["aic"]


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.6))

    # --- A. design signature: only survival's skew preference tracks the reserve ------------------
    res_grid = np.linspace(0.45 * R_TRUE, 1.45 * R_TRUE, 30)
    s_demo = 0.9
    A, B = _pair(s_demo)
    surv_curve = np.array([p_survival(A, B, r) for r in res_grid])
    p_eu = p_static("eu", A, B)
    p_pt = p_static("pt", A, B)
    ax[0, 0].axhline(0.5, color="0.75", lw=0.8)
    ax[0, 0].axvline(R_TRUE, color="0.3", ls=":", lw=1.4, label=f"requirement $R$={R_TRUE:.2f}")
    ax[0, 0].plot(res_grid, surv_curve, color="tab:red", lw=2.4, label="survival-DP")
    ax[0, 0].axhline(p_eu, color="tab:blue", lw=2, ls="--", label="EU (budget-invariant)")
    ax[0, 0].axhline(p_pt, color="tab:green", lw=2, ls="-.", label="prospect (budget-invariant)")
    ax[0, 0].axhline(0.5, color="tab:gray", lw=1.6, ls=":", label="mean-variance (skew-blind)")
    ax[0, 0].text(0.5 * R_TRUE, 0.18, "lean:\nprefers −skew", fontsize=8, color="tab:red")
    ax[0, 0].text(1.2 * R_TRUE, 0.8, "safe:\nprefers +skew", fontsize=8, color="tab:red", ha="left")
    ax[0, 0].set_xlabel("energy reserve / budget")
    ax[0, 0].set_ylabel(f"P(choose +skew over −skew), |skew|={s_demo}")
    ax[0, 0].set_ylim(0, 1)
    ax[0, 0].set_title("A. Design logic: only the survival skew preference\ntracks the reserve "
                       "(crossing 0.5 at $R$)")
    ax[0, 0].legend(fontsize=7.5, loc="center right")

    # --- B. identifiability of R vs number of budget levels ---------------------------------------
    K_levels = [1, 2, 3, 5, 8]
    B_REPS, B_N = 14, 120
    R_mean, R_sd = [], []
    for K in K_levels:
        reserves = (np.array([R_TRUE]) if K == 1
                    else np.linspace(0.5 * R_TRUE, 1.3 * R_TRUE, K))
        rng = np.random.default_rng(10)
        fits = []
        for _ in range(B_REPS):
            _, vt = make_dataset("survival", reserves, B_N, rng)
            fits.append(cm.fit_survival(vt, n_egrid=FIT_EGRID, restarts=3)["params"]["R"])
        R_mean.append(float(np.mean(fits)))
        R_sd.append(float(np.std(fits)))
        print(f"B K={K}: R_fit={R_mean[-1]:.3f} +/- {R_sd[-1]:.3f} (true {R_TRUE})", flush=True)
    ax[0, 1].axhline(R_TRUE, color="0.3", ls=":", lw=1.4, label=f"true $R$={R_TRUE:.2f}")
    ax[0, 1].errorbar(K_levels, R_mean, yerr=R_sd, fmt="o-", color="tab:purple", capsize=4,
                      lw=2, label="recovered $R$ (mean ± sd)")
    ax[0, 1].set_xlabel("number of budget levels sampled (reserve gradient)")
    ax[0, 1].set_ylabel("recovered requirement $R$")
    ax[0, 1].set_xticks(K_levels)
    ax[0, 1].set_title("B. Identifiability of $R$: 1-2 budgets is a ridge\n(exp055); a gradient "
                       "(≥3-5) pins it")
    ax[0, 1].legend(fontsize=8)

    # --- C. how much data to PIN R: precision of recovered R vs trials per cell ----------------
    trials_grid = [15, 30, 60, 120, 240]
    P_REPS = 25
    reserves5 = np.linspace(0.5 * R_TRUE, 1.3 * R_TRUE, 5)   # the 5-budget gradient
    R_mu, R_lo, R_hi = [], [], []
    for n in trials_grid:
        rng = np.random.default_rng(20)
        fits = []
        for _ in range(P_REPS):
            _, vt = make_dataset("survival", reserves5, n, rng)
            fits.append(cm.fit_survival(vt, n_egrid=FIT_EGRID, restarts=3)["params"]["R"])
        fits = np.array(fits)
        R_mu.append(float(fits.mean()))
        R_lo.append(float(np.percentile(fits, 5)))
        R_hi.append(float(np.percentile(fits, 95)))
        print(f"C n={n}: R={R_mu[-1]:.3f} [90% {R_lo[-1]:.3f},{R_hi[-1]:.3f}]", flush=True)
    ax[1, 0].axhline(R_TRUE, color="0.3", ls=":", lw=1.4, label=f"true $R$={R_TRUE:.2f}")
    ax[1, 0].fill_between(trials_grid, R_lo, R_hi, color="tab:purple", alpha=0.18,
                          label="90% recovery interval")
    ax[1, 0].plot(trials_grid, R_mu, "o-", color="tab:purple", lw=2, label="recovered $R$ (mean)")
    ax[1, 0].set_xlabel("trials per cell (5-budget gradient)")
    ax[1, 0].set_ylabel("recovered requirement $R$")
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_xticks(trials_grid)
    ax[1, 0].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax[1, 0].set_title("C. How much data to pin $R$: the recovery\ninterval tightens with "
                       "trials per cell")
    ax[1, 0].legend(fontsize=8, loc="upper right")

    # --- D. detection + specificity: survival selected on survival data, not on fixed data ----
    select = {"survival": [], "eu": [], "pt": []}
    for gen in select:
        for n in trials_grid:
            rng = np.random.default_rng(30)
            hits = 0
            for _ in range(P_REPS):
                st, vt = make_dataset(gen, reserves5, n, rng)
                _, delta = survival_vs_fixed(st, vt)
                hits += delta > DELTA_AIC
            select[gen].append(hits / P_REPS)
            print(f"D truth={gen} n={n}: P(select survival)={select[gen][-1]:.2f}", flush=True)
    styles = {"survival": ("tab:red", "truth = survival (power)"),
              "eu": ("tab:blue", "truth = EU (false positive)"),
              "pt": ("tab:green", "truth = PT (false positive)")}
    ax[1, 1].axhline(0.05, color="0.6", ls="--", lw=1, label="5% reference")
    for gen, (c, lab) in styles.items():
        ax[1, 1].plot(trials_grid, select[gen], "o-", color=c, lw=2, label=lab)
    ax[1, 1].set_xlabel("trials per cell (5-budget gradient)")
    ax[1, 1].set_ylabel(f"P(survival selected, ΔAIC > {DELTA_AIC:.0f})")
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_xticks(trials_grid)
    ax[1, 1].get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax[1, 1].set_ylim(-0.03, 1.03)
    ax[1, 1].set_title("D. Detection vs specificity: selected on\nsurvival data, ~never on "
                       "fixed-utility data")
    ax[1, 1].legend(fontsize=8, loc="center right")

    fig.tight_layout()
    out = FIG / "exp063_two_budget_design.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")

    # headline summary
    print("\nSUMMARY")
    print(f"  B: R recovery sharpens from sd={R_sd[K_levels.index(2)]:.3f} (2 budgets) to "
          f"sd={R_sd[K_levels.index(5)]:.3f} (5-budget gradient), at {B_N} trials/cell.")
    print(f"  C: with the gradient, the 90% recovery interval for R narrows to "
          f"[{R_lo[-1]:.3f}, {R_hi[-1]:.3f}] at {trials_grid[-1]} trials/cell.")
    print(f"  D: at {trials_grid[-1]} trials/cell, P(select survival)={select['survival'][-1]:.2f} "
          f"on survival data vs {max(select['eu'][-1], select['pt'][-1]):.2f} on fixed data.")


if __name__ == "__main__":
    sys.exit(main())
