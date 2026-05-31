"""Experiment 010 -- multi-alternative matching (N cue-marked patches).

Generalizes the two-patch prep to N concurrent alternatives. N patches sit on a
circle around the start (equal travel/COD to each -> no spatial bias), each marked
by a distinct cue value and running its own VI schedule. Across many schedule
configurations we measure each alternative's behavior allocation vs. obtained
reinforcement and fit the multi-alternative generalized matching law, each
alternative against the pooled rest:

    log(B_i / sum_{j!=i} B_j) = a * log(R_i / sum_{j!=i} R_j) + log b

Run:  python -m experiments.exp010_matching_multi
"""

from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from behavioral_md.matching import MatchConfig, make_matching_sim
from behavioral_md.visualization import plot_matching

N_PATCHES = 5
N_ORG, N_STEPS, N_CONDITIONS = 300, 6000, 16
RADIUS = 6.0
# VI arming probabilities drawn per condition to span ~an order of magnitude.
RATE_CHOICES = np.array([0.02, 0.04, 0.07, 0.11, 0.16])


def circle_layout(n, radius, grid):
    center = grid / 2.0
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = np.stack([center + radius * np.cos(ang), center + radius * np.sin(ang)], axis=1)
    cues = np.linspace(0.1, 0.9, n)        # distinct discriminative cue per patch
    return pos, cues, [center, center]


def main() -> None:
    grid = int(2 * RADIUS + 8)
    pos, cues, start = circle_layout(N_PATCHES, RADIUS, grid)
    mcfg = MatchConfig(grid_size=grid, sensor_range=max(8.0, 2.5 * RADIUS))
    sim, initial_state = make_matching_sim(mcfg, pos, cues, start)
    keys = jax.random.split(jax.random.key(0), N_STEPS)
    rng = np.random.default_rng(0)

    t0 = time.perf_counter()
    x_all, y_all = [], []  # log relative reinforcement / behavior, alt vs rest
    for c in range(N_CONDITIONS):
        arm = rng.choice(RATE_CHOICES, size=N_PATCHES, replace=True)
        state0 = initial_state(N_ORG, jax.random.key(500 + c))
        time_at, reinforced = sim(state0, keys, jnp.asarray(arm))
        B = np.asarray(time_at)        # [O, N]
        R = np.asarray(reinforced)     # [O, N]
        totB, totR = B.sum(axis=1, keepdims=True), R.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            # each alternative vs. the pooled rest
            lb = np.log(B / (totB - B))
            lr = np.log(R / (totR - R))
        m = np.isfinite(lb) & np.isfinite(lr)
        x_all.append(lr[m])
        y_all.append(lb[m])
    x = np.concatenate(x_all)
    y = np.concatenate(y_all)
    a, log_b = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - (a * x + log_b)) ** 2) / np.sum((y - y.mean()) ** 2)
    elapsed = time.perf_counter() - t0

    print(f"{N_PATCHES} alternatives x {N_CONDITIONS} conditions x {N_ORG} organisms "
          f"x {N_STEPS} steps in {elapsed:.1f}s")
    print(f"Multi-alternative GML (alt vs rest): "
          f"log(B_i/SumRest) = {a:.2f} * log(R_i/SumRest) + {log_b:+.2f}")
    print(f"  sensitivity a = {a:.2f}   bias log b = {log_b:+.2f}   R^2 = {r2:.3f}   "
          f"(n={x.size})")

    out = Path("outputs/figures/matching_multi.png")
    plot_matching(x, y, a, log_b, out,
                  xlabel=r"log(R$_i$ / $\Sigma$R$_{rest}$)",
                  ylabel=r"log(B$_i$ / $\Sigma$B$_{rest}$)")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
