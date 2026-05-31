"""Experiment 009 -- matching sensitivity vs. changeover delay (patch separation).

The travel time between the two cue-marked patches is the changeover-delay (COD)
analog. Classic concurrent VI-VI finds undermatching when switching is too easy
(no/short COD) and matching restored once an adequate COD (~2 s in pigeons;
Shull & Pliskoff) is imposed. Here we sweep patch separation (=travel steps) and,
at each separation, run the full VI rate-ratio sweep, fit the generalized matching
law per organism (slope a across schedules), and plot a vs. separation with 95% CI.

The grid scales with separation so distant patches fit, and the sensor range is
kept large relative to separation so both patches stay detectable -- isolating the
travel/COD manipulation from a detectability confound.

Run:  python -m experiments.exp009_matching_cod
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from behavioral_md.matching import MatchConfig, make_matching_sim
from behavioral_md.visualization import plot_matching_cod

N_ORG, N_STEPS = 400, 6000
SEPARATIONS = [2, 4, 6, 10, 14, 18]
RATE_PAIRS = [
    (0.02, 0.18), (0.04, 0.16), (0.06, 0.14), (0.10, 0.10),
    (0.14, 0.06), (0.16, 0.04), (0.18, 0.02),
]
PATCH_CUE = np.array([0.2, 0.8])


def sensitivity_for_separation(sep: int, keys) -> np.ndarray:
    """Per-organism GML slope a (across the rate-ratio sweep) at one separation."""
    grid = sep + 7
    center = grid / 2.0
    patch_pos = np.array([[center - sep / 2.0, center], [center + sep / 2.0, center]])
    start = [center, center]
    # Sensor range large vs. separation -> both patches detectable (isolate COD).
    mcfg = MatchConfig(grid_size=grid, sensor_range=max(8.0, 2.0 * sep))
    sim, initial_state = make_matching_sim(mcfg, patch_pos, PATCH_CUE, start)

    logB, logR = [], []  # each: list of [n_org] over schedules
    for i, (pL, pR) in enumerate(RATE_PAIRS):
        state0 = initial_state(N_ORG, jax.random.key(1000 * sep + i))
        time_at, reinforced = sim(state0, keys, jnp.array([pL, pR]))
        time_at = np.asarray(time_at)
        reinforced = np.asarray(reinforced)
        with np.errstate(divide="ignore", invalid="ignore"):
            lb = np.log(time_at[:, 0] / time_at[:, 1])
            lr = np.log(reinforced[:, 0] / reinforced[:, 1])
        logB.append(lb)
        logR.append(lr)
    logB = np.stack(logB, axis=1)  # [n_org, n_schedules]
    logR = np.stack(logR, axis=1)

    # Per-organism slope across schedules (finite points only; need >= 3).
    slopes = []
    for o in range(N_ORG):
        m = np.isfinite(logB[o]) & np.isfinite(logR[o])
        if m.sum() >= 3 and np.ptp(logR[o, m]) > 1e-6:
            slopes.append(np.polyfit(logR[o, m], logB[o, m], 1)[0])
    return np.array(slopes)


def main() -> None:
    keys = jax.random.split(jax.random.key(0), N_STEPS)
    t0 = time.perf_counter()
    means, cis = [], []
    print(f"{'sep':>4} {'n_fit':>6} {'a_mean':>7} {'95% CI':>14}")
    for sep in SEPARATIONS:
        s = sensitivity_for_separation(sep, keys)
        mean = float(np.mean(s))
        ci = 1.96 * float(np.std(s, ddof=1)) / np.sqrt(len(s))
        means.append(mean)
        cis.append(ci)
        print(f"{sep:>4} {len(s):>6} {mean:>7.2f}  [{mean-ci:>5.2f},{mean+ci:>5.2f}]")
    print(f"\n({len(SEPARATIONS)} separations x {len(RATE_PAIRS)} schedules x "
          f"{N_ORG} organisms x {N_STEPS} steps in {time.perf_counter()-t0:.1f}s)")

    out = Path("outputs/figures/matching_cod.png")
    plot_matching_cod(SEPARATIONS, means, cis, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
