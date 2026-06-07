"""exp058 -- model comparison: survival vs the standard accounts on skew-controlled choice.

Roadmap 2.2. The risk arc's novel claim is that a survival objective is sensitive to the SKEW of a
gamble and -- uniquely -- that the skew preference REVERSES with the energy budget. We set the
survival account against the standard rivals on gambles matched in mean and variance, so any choice
difference is a pure skew effect, and ask both what the models can express (structural) and what the
data say (Genest, Stauffer & Schultz 2016, the available machine-readable monkey study).

  A. STRUCTURAL + DATA. A pure mean-variance model (and Scalar Utility Theory, whose risk
     sensitivity is scalar/Weber noise on magnitude) is SKEW-BLIND: at matched mean+variance it
     predicts indifference (0.5) whatever the skew. Genest's monkeys preferred the positively skewed
     gamble ~70-80% of the time (matched mean+variance; third-order dominance), refuting the
     skew-blind model; a curved-utility EU (Genest's own, u=ln(x+1)/0.81) and prospect theory fit.
  B. THE DISTINGUISHING PREDICTION. The survival-DP skew preference REVERSES across the requirement
     R (prefer -skew when desperate, +skew when safe); fixed-utility EU and prospect theory predict
     a single, budget-invariant skew preference. This is the falsifiable separation.
  C. MODEL RECOVERY (synthetic). The fit/AIC harness recovers a skew-blind generator as mean-var and
     a skew generator as a skew-sensitive model; but within ONE energy budget the skew-sensitive
     models are mutually confusable -- you cannot tell them apart.
  D. THE DESIGN THAT DISCRIMINATES. Choices at TWO budgets expose a skew x budget interaction:
     non-zero for survival-DP data, ~zero for eu/pt (no state input). This mirrors Genest's
     own logistic-regression test (their Eqs 1-3) and is the design -- matched mean+variance, skew x
     energy budget -- that would settle it (Caraco & Chasin 1984, unavailable; or the exp-2.3
     pre-registration). Genest found variance/skew add nothing beyond utility on their single-budget
     data (P~0.88-0.99), which is exactly why the budget manipulation is needed.

Run:  python experiments/exp058_model_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from behavioral_md import choice_models as cm  # noqa: E402
from behavioral_md.survival import skewed_outcomes  # noqa: E402

FIG = Path("outputs/figures")

# Genest 2016 positive-skew preference (matched EV + variance), digitized from Fig 5D + text.
GENEST = [("monkey A", 0.70, 359), ("monkey B", 0.80, 500)]
GENEST_SCALE = 0.81                       # their fitted utility u(x) = ln(x + 1) / 0.81

# Survival economy for the energy-budget panels.
S, SD, DAY, NIGHT, METAB = 0.05, 0.03, 14, 16, 0.03
R = NIGHT * METAB


def _pair(mean, std, s):
    return skewed_outcomes(mean, std, float(s)), skewed_outcomes(mean, std, -float(s))


def eu_log(gamble):
    """Genest's fitted (log) expected utility of a gamble."""
    p = np.array([q for q, _ in gamble])
    x = np.array([v for _, v in gamble])
    return float((p * np.log(np.clip(x, -0.999, None) + 1.0) / GENEST_SCALE).sum())


def skew_pref_static(model, skews, mean, std, **kw):
    """P(choose +SK over -SK) at matched mean+variance, vs skew magnitude, for a static model."""
    out = []
    for s in skews:
        A, B = _pair(mean, std, s)
        out.append(cm.predict_choice(model, A, B, **kw))
    return np.array(out)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    skews = np.linspace(0.05, 1.4, 16)
    mean_g, std_g = 0.6, 0.2

    # --- A. structural skew-blindness + Genest data --------------------------------------------
    A_g, B_g = _pair(mean_g, std_g, 0.9)
    d_eu = eu_log(A_g) - eu_log(B_g)
    obs = np.array([p for _, p, _ in GENEST])
    ns = np.array([n for *_, n in GENEST])

    def beta_nll(beta):
        p = min(max(cm._sigmoid(beta[0] * d_eu), 1e-9), 1 - 1e-9)
        return -np.sum(ns * (obs * np.log(p) + (1 - obs) * np.log(1 - p)))
    beta_eu = float(minimize(beta_nll, [10.0], method="Nelder-Mead").x[0])

    p_mv = skew_pref_static(cm.mv_value, skews, mean_g, std_g, b=2.0, beta=8.0)
    p_pt = skew_pref_static(cm.pt_value, skews, mean_g, std_g, beta=6.0)
    p_eu = np.array([cm._sigmoid(beta_eu * (eu_log(skewed_outcomes(mean_g, std_g, float(s)))
                                            - eu_log(skewed_outcomes(mean_g, std_g, -float(s)))))
                     for s in skews])
    print("A. skew-blind mean-variance predicts P=0.50 for all skew (matched mean+variance).")
    print(f"   Genest +skew preference {[(m, p) for m, p, _ in GENEST]} refutes mean-variance; "
          f"EU(log) fit beta={beta_eu:.1f} passes through it.")

    # --- B. the distinguishing prediction: survival skew preference reverses with budget -------
    reserves = np.linspace(0.12, 0.95, 40)
    A_s, B_s = _pair(S, SD, 0.9)
    surv_adv = np.array([cm.survival_advantage(A_s, B_s, e, DAY, NIGHT, METAB, n_egrid=801)
                         for e in reserves])
    surv_norm = surv_adv / np.max(np.abs(surv_adv))   # +1 prefers +skew, -1 prefers -skew
    eu_sign = float(np.sign(d_eu))                    # EU(log) prefers +skew at every budget
    below = surv_norm[reserves < R - 0.03].mean()
    above = surv_norm[reserves > R + 0.03].mean()
    print(f"B. survival skew preference (normalized) reverses at R={R:.2f}: {below:+.2f} below R "
          f"(prefers -skew) -> {above:+.2f} above R; EU/PT fixed at {eu_sign:+.0f}.")

    # --- C. model recovery within one energy budget --------------------------------------------
    gen = ["mv", "eu", "pt"]
    gen_params = {"mv": dict(b=3.0), "eu": dict(a=0.3, c=4.0), "pt": dict(alpha=0.9, gamma=0.6)}
    conf = np.zeros((3, 3))
    for gi, gm in enumerate(gen):
        for rep in range(6):
            trials = []
            for s in np.linspace(-1.2, 1.2, 10):
                for dv in (0.0, 0.12, -0.12):
                    A = skewed_outcomes(0.6 + dv, 0.2, float(s))
                    B = skewed_outcomes(0.6, 0.2, -float(s))
                    p = cm.predict_choice(cm.STATIC_MODELS[gm], A, B, beta=9.0, **gen_params[gm])
                    trials.append({"A": A, "B": B, "n": 35, "k": int(rng.binomial(35, p))})
            fits = cm.compare_models(trials, names=tuple(gen), seed=rep)
            conf[gi, gen.index(fits[0]["name"])] += 1
    conf /= conf.sum(1, keepdims=True)
    print("C. recovery confusion (rows=truth, cols=AIC-best):")
    for gi, gm in enumerate(gen):
        print(f"   {gm}: " + " ".join(f"{c:.2f}" for c in conf[gi]))

    # --- D. skew x budget interaction discriminates survival from fixed-utility models ---------
    budgets = [0.6 * R, 1.4 * R]                      # below R, above R
    skew_levels = np.linspace(-0.9, 0.9, 7)
    interaction = {}
    for gm in ("eu", "pt", "survival"):
        s_all, b_all, y_all = [], [], []
        for bi, bud in enumerate(budgets):
            for s in skew_levels:
                if gm == "survival":
                    A, B = _pair(S, SD, s)
                    p = cm._sigmoid(300.0 * cm.survival_advantage(A, B, bud, DAY, NIGHT, METAB,
                                                                  n_egrid=801))
                else:
                    A, B = _pair(mean_g, std_g, s)
                    mdl = cm.eu_value if gm == "eu" else cm.pt_value
                    kw = dict(a=0.3, c=4.0) if gm == "eu" else {}
                    p = cm.predict_choice(mdl, A, B, beta=6.0, **kw)
                s_all.append(s)
                b_all.append(bi)
                y_all.append(p)
        s_arr, b_arr, y = np.array(s_all), np.array(b_all), np.array(y_all)
        x = np.column_stack([np.ones_like(s_arr), s_arr, b_arr, s_arr * b_arr])
        z = np.log(np.clip(y, 1e-3, 1 - 1e-3) / np.clip(1 - y, 1e-3, 1 - 1e-3))
        coef, *_ = np.linalg.lstsq(x, z, rcond=None)
        interaction[gm] = float(coef[3])              # skew x budget coefficient
    print("D. skew x budget interaction coefficient (~0 = budget-invariant skew preference):")
    for gm, c in interaction.items():
        print(f"   {gm}: {c:+.2f}")

    # --- figure --------------------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.3))

    ax[0, 0].axhline(0.5, color="0.7", lw=0.8)
    ax[0, 0].plot(skews, p_mv, color="tab:gray", lw=2.2, label="mean-variance (skew-blind)")
    ax[0, 0].plot(skews, p_eu, color="tab:blue", lw=2, label="EU, curved utility (Genest)")
    ax[0, 0].plot(skews, p_pt, color="tab:green", lw=2, ls="--", label="prospect theory")
    for m, p, n in GENEST:
        ci = 1.96 * np.sqrt(p * (1 - p) / n)
        ax[0, 0].errorbar([0.9], [p], yerr=[ci], fmt="o", color="black", capsize=3,
                          label=f"Genest {m} (n={n})")
    ax[0, 0].set_xlabel("skew magnitude (|skew| of the gambles)")
    ax[0, 0].set_ylabel("P(choose +skew over −skew)")
    ax[0, 0].set_ylim(0.35, 0.95)
    ax[0, 0].set_title("A. Matched mean+variance: mean-variance is skew-blind;\nthe monkey data "
                       "refute it, curved-utility EU fits")
    ax[0, 0].legend(fontsize=7.5, loc="upper left")

    ax[0, 1].axhline(0.0, color="0.7", lw=0.8)
    ax[0, 1].axvline(R, color="0.3", ls=":", lw=1.4, label=f"requirement $R$={R:.2f}")
    ax[0, 1].plot(reserves, surv_norm, color="tab:red", lw=2.2, label="survival-DP")
    ax[0, 1].axhline(eu_sign, color="tab:blue", lw=2, ls="--", label="EU / prospect (fixed)")
    ax[0, 1].fill_between(reserves, 0.0, surv_norm, where=surv_norm < 0, color="tab:red",
                          alpha=0.08)
    ax[0, 1].text(0.18, -0.6, "prefers\n−skew", fontsize=8, color="tab:red", ha="center")
    ax[0, 1].text(0.8, 0.6, "prefers\n+skew", fontsize=8, color="tab:red", ha="center")
    ax[0, 1].set_ylim(-1.15, 1.15)
    ax[0, 1].set_xlabel("energy reserve / budget")
    ax[0, 1].set_ylabel("skew preference (normalized; + = prefers +skew)")
    ax[0, 1].set_title("B. Distinguishing prediction: survival skew\npreference REVERSES at $R$; "
                       "EU/PT are budget-fixed")
    ax[0, 1].legend(fontsize=8, loc="lower right")

    im = ax[1, 0].imshow(conf, cmap="Blues", vmin=0, vmax=1)
    ax[1, 0].set_xticks(range(3))
    ax[1, 0].set_xticklabels(gen)
    ax[1, 0].set_yticks(range(3))
    ax[1, 0].set_yticklabels(gen)
    for i in range(3):
        for j in range(3):
            ax[1, 0].text(j, i, f"{conf[i, j]:.2f}", ha="center", va="center",
                          color="white" if conf[i, j] > 0.5 else "black", fontsize=10)
    ax[1, 0].set_xlabel("recovered (AIC-best)")
    ax[1, 0].set_ylabel("generating model")
    ax[1, 0].set_title("C. Recovery within one budget:\nmean-variance separable, skew models "
                       "confusable")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046, label="P(recovered)")

    names = list(interaction)
    ax[1, 1].bar(names, [interaction[n] for n in names],
                 color=["tab:blue", "tab:green", "tab:red"])
    ax[1, 1].axhline(0, color="0.4", lw=1)
    ax[1, 1].set_ylabel("skew × budget interaction coefficient")
    ax[1, 1].set_title("D. Two budgets discriminate: only survival-DP\nhas a skew×budget "
                       "interaction (the 2.3 design)")
    fig.tight_layout()
    out = FIG / "exp058_model_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
