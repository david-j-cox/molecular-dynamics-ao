"""Experiment 013 -- concatenated matching law: the PROBABILITY term.

Both patches run IDENTICAL VI schedules (equal availability rate) and deliver
equal AMOUNT, but a contact with an armed patch is reinforced only with
probability p_k. Non-reinforced contacts are extinction trials (train the cue
toward 0), so partial reinforcement drives the learned cue value to ~ p_k*lambda;
allocation then tracks relative probability -- the probability term:

    log(B_L/B_R) = a_p * log(p_L/p_R) + log b

Run:  python -m experiments.exp013_matching_probability
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from behavioral_md.experiment_utils import fit_matching_law
from behavioral_md.matching import MatchConfig, make_matching_sim
from behavioral_md.visualization import plot_matching

N_ORG, N_STEPS = 400, 5000
PATCH_POS = np.array([[2.0, 5.0], [8.0, 5.0]])     # odd-grid symmetric layout
PATCH_CUE = np.array([0.2, 0.8])
START = [5.0, 5.0]
ARM = jnp.array([0.12, 0.12])                       # IDENTICAL VI
PROB_PAIRS = [(0.9, 0.1), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.1, 0.9)]


def main() -> None:
    mcfg = MatchConfig()                            # grid_size 11 (symmetric)
    sim, initial_state = make_matching_sim(mcfg, PATCH_POS, PATCH_CUE, START)
    keys = jax.random.split(jax.random.key(0), N_STEPS)
    amount = jnp.ones(2)

    t0 = time.perf_counter()
    log_B, log_P = [], []
    for i, (pL, pR) in enumerate(PROB_PAIRS):
        state0 = initial_state(N_ORG, jax.random.key(300 + i))
        time_at, count, _amt = sim(state0, keys, ARM, amount, jnp.array([pL, pR]))
        B = np.asarray(time_at)
        bL, bR = B[:, 0], B[:, 1]
        ok = (bL > 0) & (bR > 0)
        log_B.append(np.log(bL[ok] / bR[ok]))
        log_P.append(np.full(ok.sum(), np.log(pL / pR)))

    x = np.concatenate(log_P)
    y = np.concatenate(log_B)
    a_p, log_b, r2 = fit_matching_law(x, y)
    print(f"{N_ORG} organisms x {len(PROB_PAIRS)} probability ratios x {N_STEPS} steps "
          f"(identical VI, equal amount) in {time.perf_counter()-t0:.1f}s")
    print(f"Probability term:  log(B_L/B_R) = {a_p:.2f} * log(p_L/p_R) + {log_b:+.2f}")
    print(f"  probability sensitivity a_p = {a_p:.2f}   bias log b = {log_b:+.2f}   R^2 = {r2:.3f}")

    out = Path("outputs/figures/matching_probability.png")
    plot_matching(x, y, a_p, log_b, out,
                  xlabel="log(p$_L$/p$_R$)", ylabel="log(B$_L$/B$_R$)")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
