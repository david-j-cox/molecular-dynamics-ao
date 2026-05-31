"""Experiment 008 -- concurrent VI-VI matching via discriminative cues.

Two cue-marked patches run different VI schedules. Across a sweep of VI rate
ratios, measure how the organism allocates behavior (time at each patch) versus
the reinforcement it obtains, and fit the generalized matching law
``log(B_L/B_R) = a*log(R_L/R_R) + log b`` (a = sensitivity, b = bias).

Run:  python -m experiments.exp008_matching
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
PATCH_POS = np.array([[2.0, 5.0], [8.0, 5.0]])   # left, right (separation = COD)
PATCH_CUE = np.array([0.2, 0.8])                  # "green" / "red"
START = [5.0, 5.0]
# VI rate ratios to sweep (left:right), as arming-probability pairs (mean VI ~ 1/p).
RATE_PAIRS = [
    (0.02, 0.18), (0.04, 0.16), (0.06, 0.14), (0.10, 0.10),
    (0.14, 0.06), (0.16, 0.04), (0.18, 0.02),
]


def main() -> None:
    mcfg = MatchConfig()
    sim, initial_state = make_matching_sim(mcfg, PATCH_POS, PATCH_CUE, START)
    keys = jax.random.split(jax.random.key(0), N_STEPS)

    t0 = time.perf_counter()
    log_B, log_R = [], []
    for i, (pL, pR) in enumerate(RATE_PAIRS):
        state0 = initial_state(N_ORG, jax.random.key(100 + i))
        time_at, reinforced = sim(state0, keys, jnp.array([pL, pR]))
        time_at = np.asarray(time_at)
        reinforced = np.asarray(reinforced)
        # Discard the first half (acquisition); use steady-state allocation.
        bL, bR = time_at[:, 0], time_at[:, 1]
        rL, rR = reinforced[:, 0], reinforced[:, 1]
        ok = (bL > 0) & (bR > 0) & (rL > 0) & (rR > 0)
        log_B.append(np.log(bL[ok] / bR[ok]))
        log_R.append(np.log(rL[ok] / rR[ok]))
    elapsed = time.perf_counter() - t0

    x = np.concatenate(log_R)
    y = np.concatenate(log_B)
    a, log_b = np.polyfit(x, y, 1)
    ss_res = np.sum((y - (a * x + log_b)) ** 2)
    r2 = 1 - ss_res / np.sum((y - y.mean()) ** 2)

    print(f"{N_ORG} organisms x {len(RATE_PAIRS)} schedules x {N_STEPS} steps in {elapsed:.1f}s")
    print(f"Generalized matching law fit:  log(B_L/B_R) = {a:.2f} * log(R_L/R_R) + {log_b:+.2f}")
    print(f"  sensitivity a = {a:.2f}   bias log b = {log_b:+.2f}   R^2 = {r2:.3f}")
    print(f"  ({'undermatching' if a < 0.9 else 'overmatching' if a > 1.1 else 'near-matching'})")

    out = Path("outputs/figures/matching.png")
    plot_matching(x, y, a, log_b, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
