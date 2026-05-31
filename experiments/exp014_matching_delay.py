"""Experiment 014 -- concatenated matching law: the DELAY term.

Both patches run IDENTICAL VI schedules and deliver equal AMOUNT with certainty,
but reinforcers are delivered after a patch-specific DELAY. Delay reduces the
reinforcer's efficacy (empirical delay discounting; default hyperbolic 1/(1+kD)),
so the longer-delay patch builds a lower cue value and is approached less -- the
delay term, which enters the concatenated matching law with a NEGATIVE sign:

    log(B_L/B_R) = -a_d * log(D_L/D_R) + log b

(longer delay on the left -> log(D_L/D_R) > 0 -> log(B_L/B_R) < 0). We report the
fitted slope (expected negative) and a_d = -slope.

Run:  python -m experiments.exp014_matching_delay
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
PATCH_POS = np.array([[2.0, 5.0], [8.0, 5.0]])
PATCH_CUE = np.array([0.2, 0.8])
START = [5.0, 5.0]
ARM = jnp.array([0.12, 0.12])                       # IDENTICAL VI
DELAY_PAIRS = [(1.0, 9.0), (2.0, 6.0), (4.0, 4.0), (6.0, 2.0), (9.0, 1.0)]  # steps


def main() -> None:
    mcfg = MatchConfig(delay_discount="hyperbolic", delay_k=0.5)
    sim, initial_state = make_matching_sim(mcfg, PATCH_POS, PATCH_CUE, START)
    keys = jax.random.split(jax.random.key(0), N_STEPS)
    amount, prob = jnp.ones(2), jnp.ones(2)

    t0 = time.perf_counter()
    log_B, log_D = [], []
    for i, (dL, dR) in enumerate(DELAY_PAIRS):
        state0 = initial_state(N_ORG, jax.random.key(400 + i))
        time_at, count, _amt = sim(state0, keys, ARM, amount, prob, jnp.array([dL, dR]))
        B = np.asarray(time_at)
        bL, bR = B[:, 0], B[:, 1]
        ok = (bL > 0) & (bR > 0)
        log_B.append(np.log(bL[ok] / bR[ok]))
        log_D.append(np.full(ok.sum(), np.log(dL / dR)))

    x = np.concatenate(log_D)
    y = np.concatenate(log_B)
    slope, log_b, r2 = fit_matching_law(x, y)
    print(f"{N_ORG} organisms x {len(DELAY_PAIRS)} delay ratios x {N_STEPS} steps "
          f"(identical VI, equal amount, hyperbolic) in {time.perf_counter()-t0:.1f}s")
    print(f"Delay term:  log(B_L/B_R) = {slope:.2f} * log(D_L/D_R) + {log_b:+.2f}")
    print(f"  slope = {slope:.2f} (expected < 0: longer delay -> less preferred)")
    print(f"  delay sensitivity a_d = {-slope:.2f}   bias log b = {log_b:+.2f}   R^2 = {r2:.3f}")

    out = Path("outputs/figures/matching_delay.png")
    plot_matching(x, y, slope, log_b, out,
                  xlabel="log(D$_L$/D$_R$)", ylabel="log(B$_L$/B$_R$)")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
