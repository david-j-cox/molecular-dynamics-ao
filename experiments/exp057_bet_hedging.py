"""exp057 -- bet-hedging meets the energy-budget rule: two regimes of one survival objective.

Roadmap 3.1. The risk arc so far optimizes survival of a SINGLE day/night cycle. Bet-hedging is an
across-GENERATION phenomenon: in a fluctuating world selection maximizes long-run (geometric-mean)
lineage growth, which can favor a lower-variance strategy even at equal arithmetic-mean fitness. We
put both on one substrate and ask how environmental autocorrelation arbitrates between them.

Days are GOOD or BAD via a symmetric 2-state Markov chain with stay-probability rho (the marginal is
fixed at 50/50 for every rho, so a sweep over rho changes ONLY the clustering -- mean bad-run length
1/(1-rho) -- not the fraction of bad days). A bad day scales foraging intake by ``bad_scale`` (the
refuge quality: high = the safe option still sustains you on bad days, low = it does not). Reserves
carry across a season of n_days; the end-of-season reserve is the fecundity (offspring proxy); the
geometric mean of cohort fecundity over seasons is the long-run growth a lineage maximizes. The
options are the arc's matched-mean pair (safe = sure intake, risky = a mean-preserving spread): the
ONLY thing that differs between them is variance -- any preference is a pure risk preference.

Result -- a phase diagram over (autocorrelation rho x refuge quality bad_scale):
  * REFUGE (bad_scale high): the safe option survives bad runs, so selection is CONSERVATIVE -- play
    safe, reduce fitness variance (classic bet-hedging). Autocorrelation barely matters.
  * NO REFUGE (bad_scale low) with clustered bad runs (rho high): the safe option cannot outlast a
    long bad run, so selection is RISK-PRONE -- gamble (the energy-budget rule extended across days:
    when a long bad streak will kill you anyway, variance is the only hope).
  * INTERMEDIATE refuge: rising autocorrelation FLIPS the optimum from safe to gambling -- a single
    knob (clustering) moving behavior between the two phenomena.
So bet-hedging and the energy-budget rule are two regions of one survival objective. (This corrects
the roadmap's guess of a one-directional conservative bet-hedge: the dominant effect of correlated
scarcity is risk-PRONENESS; conservatism wins only where a refuge exists.) The geometric optimum
(analysis) and the evolved threshold (selection) are two readouts of the same process and agree.

Run:  python experiments/exp057_bet_hedging.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md.survival import bethedge_fitness, evolve_bethedge  # noqa: E402

FIG = Path("outputs/figures")
SAFE = [(1.0, 0.05)]
RISKY = [(0.5, 0.0), (0.5, 0.10)]            # matched mean (0.05); differs from safe in variance
DAY, NIGHT, METAB, NDAYS = 12, 4, 0.025, 24
THETAS = np.linspace(0.0, 1.0, 11)


SEEDS = (5, 6, 7)   # seed-average so the phase diagram / flip is robust, not a single-seed argmax


def avg_fitness(rho: float, bad: float) -> tuple[np.ndarray, np.ndarray]:
    """Geometric and arithmetic fitness over THETAS at (rho, bad_scale), averaged across SEEDS."""
    geom = np.zeros(THETAS.size)
    arith = np.zeros(THETAS.size)
    for sd in SEEDS:
        fits = [bethedge_fitness(SAFE, RISKY, DAY, NIGHT, METAB, th, n_days=NDAYS, rho=rho,
                                 bad_scale=bad, n_seasons=80, seed=sd) for th in THETAS]
        geom += np.array([f["geom"] for f in fits])
        arith += np.array([f["arith"] for f in fits])
    return geom / len(SEEDS), arith / len(SEEDS)


def geom_opt_theta(rho: float, bad: float) -> float:
    """Geometric-mean-optimal constant threshold at (rho, bad_scale), seed-averaged."""
    geom, _ = avg_fitness(rho, bad)
    return float(THETAS[int(np.argmax(geom))])


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    # --- A. phase diagram: geometric-optimal threshold over (rho x refuge) ----------------------
    rhos = np.linspace(0.50, 0.96, 8)
    bads = np.linspace(0.50, 0.82, 8)
    grid = np.empty((bads.size, rhos.size))
    for i, bad in enumerate(bads):
        for j, rho in enumerate(rhos):
            grid[i, j] = geom_opt_theta(rho, bad)
    print("Phase diagram (geom-optimal threshold; low = bet-hedge/safe, high = desperation/gamble)")
    print("  refuge\\rho " + " ".join(f"{r:.2f}" for r in rhos))
    for i, bad in enumerate(bads):
        print(f"  bad={bad:.2f}   " + "    ".join(f"{v:.1f}" for v in grid[i]))

    # --- B. the intermediate-refuge flip: threshold vs autocorrelation --------------------------
    bad_mid = 0.60
    rho_line = np.linspace(0.50, 0.96, 9)
    geom_line, arith_line, evolved_line = [], [], []
    for rho in rho_line:
        geom, arith = avg_fitness(rho, bad_mid)
        geom_line.append(THETAS[int(np.argmax(geom))])
        arith_line.append(THETAS[int(np.argmax(arith))])
        evolved_line.append(evolve_bethedge(SAFE, RISKY, DAY, NIGHT, METAB, n_days=NDAYS, rho=rho,
                                             bad_scale=bad_mid, n_generations=120, seed=0)
                            ["evolved_theta"])
    print(f"\nIntermediate refuge (bad={bad_mid}): threshold vs autocorrelation")
    print("  rho      " + " ".join(f"{r:.2f}" for r in rho_line))
    print("  geom*    " + "  ".join(f"{v:.1f}" for v in geom_line))
    print("  evolved  " + "  ".join(f"{v:.2f}" for v in evolved_line))

    # --- C. the bet-hedging signature at a refuge cell: equal arithmetic, geometric penalizes var
    bad_ref = 0.72
    arith_c, geom_c = {}, {}
    for rho in (0.55, 0.92):
        a = [bethedge_fitness(SAFE, RISKY, DAY, NIGHT, METAB, th, n_days=NDAYS, rho=rho,
                              bad_scale=bad_ref, seed=5) for th in THETAS]
        arith_c[rho] = np.array([x["arith"] for x in a])
        geom_c[rho] = np.array([x["geom"] for x in a])

    # --- figure ---------------------------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.7))

    im = ax[0].imshow(grid, origin="lower", aspect="auto", cmap="RdBu_r", vmin=0, vmax=1,
                      extent=[rhos[0], rhos[-1], bads[0], bads[-1]])
    cs = ax[0].contour(rhos, bads, grid, levels=[0.5], colors="black", linewidths=1.5)
    ax[0].clabel(cs, fmt="flip", fontsize=8)
    ax[0].text(0.55, 0.78, "refuge:\nbet-hedge\n(play safe)", fontsize=8.5, color="navy", va="top")
    ax[0].text(0.82, 0.54, "no refuge:\ndesperation\n(gamble)", fontsize=8.5, color="darkred",
               va="top", ha="center")
    ax[0].set_xlabel("autocorrelation $\\rho$ (mean bad-run $=1/(1-\\rho)$)")
    ax[0].set_ylabel("refuge quality (bad-day intake scale)")
    ax[0].set_title("A. Phase diagram: optimal risk threshold\nbet-hedge vs energy-budget"
                    " desperation")
    fig.colorbar(im, ax=ax[0], label="geometric-optimal threshold $\\theta^*$")

    ax[1].plot(rho_line, geom_line, "o-", color="tab:purple", lw=2, label="geometric optimum")
    ax[1].plot(rho_line, evolved_line, "s--", color="tab:green", lw=1.6,
               label="evolved (selection)")
    ax[1].plot(rho_line, arith_line, "^:", color="tab:gray", lw=1.4, label="arithmetic optimum")
    ax[1].set_ylim(-0.05, 1.05)
    ax[1].set_xlabel("autocorrelation $\\rho$")
    ax[1].set_ylabel("risk threshold $\\theta$ (gamble below)")
    ax[1].set_title(f"B. Autocorrelation flips the optimum\n(intermediate refuge, bad={bad_mid}):"
                    " safe $\\to$ gamble")
    ax[1].legend(fontsize=8, loc="center left")

    for rho, c in [(0.55, "tab:blue"), (0.92, "tab:red")]:
        ax[2].plot(THETAS, arith_c[rho] / arith_c[rho].max(), "--", color=c, lw=1.3, alpha=0.7)
        ax[2].plot(THETAS, geom_c[rho] / geom_c[rho].max(), "-", color=c, lw=2,
                   label=f"$\\rho$={rho} geometric")
    ax[2].set_xlabel("risk threshold $\\theta$ (gamble below)")
    ax[2].set_ylabel("fitness (each $\\div$ its max)")
    ax[2].set_title(f"C. Bet-hedging signature (refuge, bad={bad_ref})\narithmetic flat (--), "
                    "geometric (–) punishes variance")
    ax[2].legend(fontsize=8, loc="lower center")

    fig.tight_layout()
    out = FIG / "exp057_bet_hedging.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
