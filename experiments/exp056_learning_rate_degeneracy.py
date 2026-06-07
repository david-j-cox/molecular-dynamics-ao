"""exp056 -- the learning-rate vs softmax-temperature degeneracy: what sequences canNOT fix.

Companion to exp055. There the survival PLANNER's requirement R is identifiable from choice
sequences and lost in aggregate preference. The natural worry is that "use sequences" is the whole
lesson -- that any model's parameters fall out once you keep the trial-by-trial data. They do not.

On the SAME matched-mean risk choice (sure safe payoff vs a 0/1 risky payoff of equal mean) and the
SAME softmax choice rule, replace the survival planner with a generic reinforcement-learning value-
learner: one scalar value per option, delta-rule updated (rate ``alpha``), chosen by a Boltzmann
policy (temperature ``beta``). The textbook agent behind exp049/exp052. Its two parameters are
partially CONFOUNDED in the choice sequence -- a faster learner with a flatter policy produces
the same choices as a slower learner with a sharper one. This is the well-known bandit alpha-beta
degeneracy (Comput. Brain Behav. 2022; Daw 2011).

Results (printed below + figure exp056_degeneracy.png):
  A. The negative-log-likelihood surface over (alpha, beta) is a long diagonal VALLEY -- a ridge of
     near-equal fit along which alpha and beta trade off (anticorrelated), not a single well.
  B. Refitting many datasets from one true (alpha, beta) gives a recovered cloud stretched
     along that anticorrelated ridge: the COMBINATION is constrained, each parameter alone is not.
  C. Across a grid of true alpha, recovered alpha is imprecise and regressive (wide per-true-value
     scatter, coefficient of variation ~40%), whereas the planner's R recovered at r=1.00 with a
     profile-likelihood 95% interval narrower than the grid step, from the same data (exp055).

The lesson: identifiability comes from the STATE-DEPENDENT STRUCTURE of the survival model (which
option is favored depends on the reserve relative to R, so the reserve-resolved sequence pins R),
NOT from possessing sequence data per se. A structureless value-learner stays degenerate however
long you watch it. This sharpens exp055 and pre-empts the "sequences fix everything" misreading.

Run:  python experiments/exp056_learning_rate_degeneracy.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md import recovery as rc  # noqa: E402

FIG = Path("outputs/figures")
TRUE_ALPHA, TRUE_BETA = 0.2, 6.0
N_TRIALS = 150                 # a realistic experiment-length session (animal/human bandit runs),
#                                where the alpha-beta degeneracy bites (cf. Wilson & Collins)
N_DATASETS = 40
PLANNER_R_CORR = 1.00          # exp055: sequence recovery of R across its grid


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    # --- A. likelihood surface over (alpha, beta) -----------------------------------------------
    ref = rc.rl_simulate(TRUE_ALPHA, TRUE_BETA, n_trials=N_TRIALS, seed=3)
    print(f"Value-learner on matched-mean risk choice; true alpha={TRUE_ALPHA}, beta={TRUE_BETA}.")
    print(f"Reference: risky fraction {ref['choices'].mean():.3f} over {N_TRIALS} trials\n")
    alphas = np.linspace(0.03, 0.75, 60)
    betas = np.linspace(1.0, 18.0, 60)
    surf = rc.rl_loglik_surface(ref["choices"], ref["rewards"], alphas, betas)
    ridge_beta = betas[np.argmin(surf, axis=1)]            # best beta at each alpha
    valley = surf.min(axis=1) < 2.0                        # alphas within ~2 logL of the optimum
    ridge_corr = float(np.corrcoef(alphas[valley], ridge_beta[valley])[0, 1])
    mle = rc.rl_fit(ref["choices"], ref["rewards"])
    print(f"A. surface: MLE alpha={mle['alpha']:.3f}, beta={mle['beta']:.2f}; "
          f"valley spans alpha in [{alphas[valley].min():.2f}, {alphas[valley].max():.2f}]; "
          f"ridge corr(alpha,beta)={ridge_corr:+.3f}")

    # --- B. recovery cloud from many independent datasets at one true theta ---------------------
    rec_a, rec_b = [], []
    for s in range(N_DATASETS):
        d = rc.rl_simulate(TRUE_ALPHA, TRUE_BETA, n_trials=N_TRIALS, seed=1000 + s)
        f = rc.rl_fit(d["choices"], d["rewards"])
        rec_a.append(f["alpha"])
        rec_b.append(f["beta"])
    rec_a, rec_b = np.array(rec_a), np.array(rec_b)
    cloud_corr = float(np.corrcoef(rec_a, rec_b)[0, 1])
    print(f"B. recovery cloud ({N_DATASETS} datasets): recovered alpha {rec_a.mean():.3f}+-"
          f"{rec_a.std():.3f}, beta {rec_b.mean():.2f}+-{rec_b.std():.2f}; "
          f"corr(rec alpha, rec beta)={cloud_corr:+.3f}")

    # --- C. marginal recovery of alpha across a grid of true alpha ------------------------------
    true_alpha_grid = np.array([0.05, 0.12, 0.20, 0.30, 0.45, 0.60])
    rec_alpha_grid = []
    for j, a in enumerate(true_alpha_grid):
        fits = []
        for s in range(6):
            d = rc.rl_simulate(float(a), TRUE_BETA, n_trials=N_TRIALS, seed=5000 + 13 * j + s)
            fits.append(rc.rl_fit(d["choices"], d["rewards"])["alpha"])
        rec_alpha_grid.append(fits)
    rec_alpha_grid = np.array(rec_alpha_grid)              # [grid, reps]
    rep_truth = np.repeat(true_alpha_grid, rec_alpha_grid.shape[1])
    alpha_corr = float(np.corrcoef(rep_truth, rec_alpha_grid.ravel())[0, 1])
    cv = float(np.mean(rec_alpha_grid.std(1) / np.clip(rec_alpha_grid.mean(1), 1e-6, None)))
    print(f"C. alpha recovery across the grid: r={alpha_corr:.3f}, mean within-truth CV={cv:.0%}  "
          f"(vs planner R r={PLANNER_R_CORR:.2f}, sub-grid-step CI, from sequences, exp055)")

    # --- figure ---------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    # A. NLL surface + ridge + truth + MLE
    levels = [0.5, 1, 2, 4, 8, 16, 32]
    cs = ax[0].contourf(alphas, betas, surf.T, levels=levels, cmap="viridis_r", extend="max")
    ax[0].plot(alphas[valley], ridge_beta[valley], color="white", lw=1.4, ls="--",
               label="best-fit ridge")
    ax[0].scatter([TRUE_ALPHA], [TRUE_BETA], color="red", marker="*", s=160, zorder=5,
                  label="true", edgecolor="black", linewidth=0.5)
    ax[0].scatter([mle["alpha"]], [mle["beta"]], color="white", marker="o", s=45, zorder=5,
                  edgecolor="black", label="MLE")
    ax[0].set_xlabel("learning rate $\\alpha$")
    ax[0].set_ylabel("softmax temperature $\\beta$")
    ax[0].set_title(f"A. A degenerate valley, not a well\nridge corr$(\\alpha,\\beta)$="
                    f"{ridge_corr:+.2f}")
    ax[0].legend(fontsize=8, loc="upper right")
    fig.colorbar(cs, ax=ax[0], label="$\\Delta$ neg-log-lik")

    # B. recovery cloud
    ax[1].scatter(rec_a, rec_b, s=30, color="tab:red", alpha=0.7, edgecolor="0.3", linewidth=0.4,
                  zorder=3, label="refits")
    ax[1].scatter([TRUE_ALPHA], [TRUE_BETA], color="black", marker="*", s=170, zorder=5,
                  label="true")
    ax[1].set_xlabel("recovered $\\alpha$")
    ax[1].set_ylabel("recovered $\\beta$")
    ax[1].set_title(f"B. Recovery cloud trades off\ncorr$(\\hat\\alpha,\\hat\\beta)$="
                    f"{cloud_corr:+.2f} ({N_DATASETS} datasets)")
    ax[1].legend(fontsize=8, loc="upper right")

    # C. marginal recovery of alpha vs the planner's clean R
    for j, a in enumerate(true_alpha_grid):
        ax[2].scatter(np.full(rec_alpha_grid.shape[1], a), rec_alpha_grid[j], s=24,
                      color="tab:red", alpha=0.6, zorder=3)
    ax[2].scatter(true_alpha_grid, rec_alpha_grid.mean(1), s=60, color="darkred", marker="_",
                  zorder=4, label="mean recovered")
    lim = [0.0, 0.7]
    ax[2].plot(lim, lim, color="0.5", ls="--", lw=1, label="identity")
    ax[2].set_xlim(*lim)
    ax[2].set_ylim(-0.02, 0.9)
    ax[2].set_xlabel("true learning rate $\\alpha$")
    ax[2].set_ylabel("recovered $\\alpha$")
    ax[2].set_title(f"C. $\\alpha$ recovery imprecise (CV={cv:.0%})\nvs planner $R$ "
                    f"($r$={PLANNER_R_CORR:.2f}, tight, exp055)")
    ax[2].legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    out = FIG / "exp056_degeneracy.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
