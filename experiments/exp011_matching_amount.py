"""Experiment 011 -- concatenated matching law: the AMOUNT term.

Both patches run IDENTICAL VI schedules (equal reinforcement RATE), but deliver
different reinforcement AMOUNTS (magnitude per reinforcer). With rate held equal,
allocation should track relative amount -- the amount term of the concatenated
matching law:

    log(B_L/B_R) = a_amt * log(A_L/A_R) + log b

The cue's learned value tracks the magnitude collected, so the richer-amount
patch is approached more. We sweep the amount ratio and fit a_amt.

Run:  python -m experiments.exp011_matching_amount
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from behavioral_md.matching import MatchConfig, make_matching_sim
from behavioral_md.visualization import plot_matching

N_ORG, N_STEPS = 400, 4000
PATCH_POS = np.array([[2.0, 5.0], [8.0, 5.0]])
PATCH_CUE = np.array([0.2, 0.8])
START = [5.0, 5.0]
ARM = jnp.array([0.10, 0.10])              # IDENTICAL VI on both patches
AMOUNT_PAIRS = [                            # (A_L, A_R): swept amount ratios
    (0.3, 1.7), (0.5, 1.5), (0.7, 1.3), (1.0, 1.0),
    (1.3, 0.7), (1.5, 0.5), (1.7, 0.3),
]


def main() -> None:
    mcfg = MatchConfig()
    sim, initial_state = make_matching_sim(mcfg, PATCH_POS, PATCH_CUE, START)
    keys = jax.random.split(jax.random.key(0), N_STEPS)

    t0 = time.perf_counter()
    log_B, log_A = [], []
    for i, (aL, aR) in enumerate(AMOUNT_PAIRS):
        state0 = initial_state(N_ORG, jax.random.key(200 + i))
        time_at, count, _amt = sim(state0, keys, ARM, jnp.array([aL, aR]))
        B = np.asarray(time_at)
        bL, bR = B[:, 0], B[:, 1]
        ok = (bL > 0) & (bR > 0)
        log_B.append(np.log(bL[ok] / bR[ok]))
        log_A.append(np.full(ok.sum(), np.log(aL / aR)))   # programmed amount ratio

    x = np.concatenate(log_A)
    y = np.concatenate(log_B)
    a_amt, log_b = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (a_amt * x + log_b)) ** 2) / np.sum((y - y.mean()) ** 2)
    elapsed = time.perf_counter() - t0

    print(f"{N_ORG} organisms x {len(AMOUNT_PAIRS)} amount ratios x {N_STEPS} steps "
          f"(identical VI) in {elapsed:.1f}s")
    print(f"Amount term:  log(B_L/B_R) = {a_amt:.2f} * log(A_L/A_R) + {log_b:+.2f}")
    print(f"  amount sensitivity a_amt = {a_amt:.2f}   bias log b = {log_b:+.2f}   "
          f"R^2 = {r2:.3f}")

    out = Path("outputs/figures/matching_amount.png")
    plot_matching(x, y, a_amt, log_b, out,
                  xlabel="log(A$_L$/A$_R$)", ylabel="log(B$_L$/B$_R$)")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
