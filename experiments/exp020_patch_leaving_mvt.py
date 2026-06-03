"""Experiment 020 -- patch-leaving and the marginal-value theorem (multi-patch JAX world).

Two depleting/regrowing food patches separated by a travel distance D. The
organism senses each patch's salience = exp(-dist/range) * (biomass/K) and heads
to the most salient one (forage.make_forage_sim). While it feeds, the occupied
patch depletes -- its salience falls -- while the distant patch regrows. The
organism leaves once the alternative's distance-attenuated salience overtakes the
current patch, i.e. when

    biomass_frac(current) < exp(-D / sensor_range)   (alternative near full).

So the GIVE-UP DENSITY (patch biomass at the moment of leaving) should fall as
exp(-D/range) and RESIDENCE TIME should rise with D: the classic marginal-value
theorem result -- longer travel justifies depleting a patch further before
leaving. This experiment sweeps D and measures both, with no give-up rule
hand-coded; leaving emerges from the salience dynamics.

Run:   python experiments/exp020_patch_leaving_mvt.py
Saves: outputs/logs/exp020_mvt.json
       outputs/figures/exp020_giveup_density.png
       outputs/figures/exp020_residence_time.png
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from behavioral_md.config import SimulationConfig
from behavioral_md.experiment_utils import (
    compute_mean_ci,
    make_cue_centers,
    make_inert_source,
    save_results_json,
)
from behavioral_md.forage import initial_state, make_forage_sim
from behavioral_md.jax_engine import build_spec
from behavioral_md.visualization import plot_giveup_density, plot_residence_time

# Travel distances are kept within sensor_range (12) so the distant patch is
# actually sensed: leaving is then governed by the salience give-up rule rather
# than by the organism failing to detect the alternative. Beyond ~sensor_range
# the gradient breaks down (few organisms relocate; see lab notebook).
TRAVEL_DISTANCES = [3.0, 4.0, 5.0, 6.0, 7.0]
N_ORG = 128
N_STEPS = 3500
SEED = 0


def _config() -> SimulationConfig:
    """Survivable, perpetually-foraging two-patch world.

    food_min_biomass is dropped to 0.02 so give-up densities below the default
    0.2 floor are reachable (the long-travel regime). Metabolism is cheap so the
    organism survives the whole run, but intake only flows while in contact, so it
    stays hungry enough that the food drive keeps it cycling between patches. A
    low softmax temperature sharpens emission so the organism commits to feeding
    (the ``consume`` atom holds it on a rich patch) instead of random-walking off
    it -- leaving is then driven by salience give-up, not emission noise.
    """
    return SimulationConfig(
        grid_size=35,
        sensor_range=12.0,
        consume_radius=1.0,
        food_carrying_capacity=1.0,
        food_min_biomass=0.02,
        food_regrowth_rate=0.1,
        food_intake_rate=0.06,
        basal_metabolism=0.005,
        move_cost=0.005,
        rest_cost=0.001,
        energy_init=0.6,
        energy_capacity=1.0,
        deficit_exponent=2.0,
        motivational_strength=2.0,
        softmax_temperature=0.15,
    )


def _patches(cfg: SimulationConfig, dist: float) -> np.ndarray:
    """Two patches a distance ``dist`` apart, centered horizontally on the grid."""
    c = (cfg.grid_size - 1) / 2.0
    return np.array([[c - dist / 2.0, c], [c + dist / 2.0, c]], float)


def _residences(on_patch: np.ndarray, biomass: np.ndarray, energy: np.ndarray):
    """Extract (give_up_density, residence_steps, energy_at_giveup) per patch stay.

    ``on_patch`` [T] int (-1 = not in range of any patch), ``biomass`` [T, P],
    ``energy`` [T].

    A residence is defined by patch ALLEGIANCE: the organism is "at" a patch from
    its first in-range contact until it first makes contact with the OTHER patch.
    Brief excursions out of range (random steps that return to the same patch
    before reaching the other) do NOT end a residence -- only arriving at a
    different patch does. The give-up density is the patch's biomass at the last
    in-range contact before the switch; the duration spans first-to-last contact;
    the give-up energy is the organism's reserve at that last contact (it sets the
    deficit gain, hence the tenacity factor kappa). A residence is recorded only
    when the organism actually switches patches.
    """
    giveups, durations, energies = [], [], []
    cur = -1            # patch the organism is currently allied to
    first_t = last_t = -1
    last_bm = last_e = float("nan")
    for t, p in enumerate(on_patch):
        if p < 0:
            continue
        if cur < 0:                     # first contact of the run
            cur, first_t, last_t = p, t, t
            last_bm, last_e = float(biomass[t, p]), float(energy[t])
        elif p == cur:                  # still on the same patch
            last_t, last_bm, last_e = t, float(biomass[t, p]), float(energy[t])
        else:                           # arrived at the other patch -> switched
            giveups.append(last_bm)
            durations.append(last_t - first_t + 1)
            energies.append(last_e)
            cur, first_t, last_t = p, t, t
            last_bm, last_e = float(biomass[t, p]), float(energy[t])
    return giveups, durations, energies


def main() -> None:
    cfg = _config()
    spec = build_spec(config=cfg)
    cue_centers = make_cue_centers(cfg)
    far = make_inert_source(cfg)  # inert danger/light/cue

    # First pass: simulate each travel distance and measure give-up density,
    # residence, and the naive (frictionless) salience-crossover prediction.
    rows = []  # (D, gm, gci, rm, rci, naive, n, mean_E, frac_alive)
    for dist in TRAVEL_DISTANCES:
        patches = _patches(cfg, dist)
        sim = make_forage_sim(spec, cfg, patches, far, far, far, cue_centers)
        state = initial_state(spec, cfg, N_ORG, 2, patches[0], cfg.n_cue_receptors)
        keys = jax.random.split(jax.random.PRNGKey(SEED), N_STEPS)
        fr = jnp.ones(N_ORG, bool)
        cv = jnp.zeros(N_ORG)
        final, (energy, _at_food, on_patch, biomass) = sim(state, keys, fr, cv)

        on_np, bm_np, en_np = np.asarray(on_patch), np.asarray(biomass), np.asarray(energy)
        all_g, all_r = [], []
        for o in range(N_ORG):
            g, r, _e = _residences(on_np[:, o], bm_np[:, o], en_np[:, o])
            all_g += g
            all_r += r
        gm, gci = compute_mean_ci(all_g)
        rm, rci = compute_mean_ci(all_r)
        naive = float(np.exp(-dist / cfg.sensor_range))
        rows.append((dist, gm, gci, rm, rci, naive, len(all_g),
                     float(en_np.mean()), float(np.asarray(final.alive).mean())))

    # Tenacity factor kappa: a SINGLE constant patch-tenacity scale. The model's
    # leave point is the salience crossover exp(-D/range) scaled down by kappa < 1
    # (the organism depletes PAST indifference -- consummatory perseveration: the
    # consume atom's food gain 1.0 exceeds approach_food's 0.5, amplified by
    # integrator lag). The test is that ONE kappa collapses all distances onto the
    # a-priori exp(-D/range) shape -- i.e. observed/naive is constant in D. (The
    # static gain ratio (g_leave+d_gain)/(g_stay+d_gain) brackets kappa: 0.5 when
    # satiated, ~0.82 when starving; the realized value is set by the dynamics.)
    ratios = np.array([gm / naive for _d, gm, _gc, _r, _rc, naive, _n, _e, _a in rows])
    kappa = float(np.mean(ratios))
    kappa_sd = float(np.std(ratios))

    results = {
        "sensor_range": cfg.sensor_range,
        "kappa_fitted": kappa,
        "kappa_sd": kappa_sd,
        "observed_over_naive_by_distance": {
            str(r[0]): round(float(r[1] / r[5]), 4) for r in rows
        },
        "by_distance": {},
    }
    fig_d, fig_r = [], []
    for dist, gm, gci, rm, rci, naive, n, mean_e, frac_alive in rows:
        corrected = kappa * naive
        results["by_distance"][str(dist)] = {
            "n_residences": n,
            "give_up_density_mean": gm, "give_up_density_ci": gci,
            "give_up_density_naive": naive, "give_up_density_corrected": corrected,
            "residence_steps_mean": rm, "residence_steps_ci": rci,
            "mean_energy": mean_e, "frac_alive_end": frac_alive,
        }
        fig_d.append((dist, gm, gci, naive, corrected))
        fig_r.append((dist, rm, rci))
        print(
            f"D={dist:5.1f}  give-up={gm:.3f}+/-{gci:.3f}  naive={naive:.3f}  "
            f"obs/naive={gm/naive:.3f}  kappa*naive={corrected:.3f}  "
            f"residence={rm:5.1f}+/-{rci:4.1f}  n={n}  alive_end={frac_alive:.2f}"
        )
    print(f"\nFitted tenacity kappa = {kappa:.3f} +/- {kappa_sd:.3f} (constant across D)")

    log = save_results_json("exp020_mvt.json", results)

    figdir = Path("outputs/figures")
    d = [x[0] for x in fig_d]
    p1 = plot_giveup_density(d, [x[1] for x in fig_d], [x[2] for x in fig_d],
                             [x[3] for x in fig_d], figdir / "exp020_giveup_density.png",
                             corrected=[x[4] for x in fig_d])
    p2 = plot_residence_time(d, [x[1] for x in fig_r], [x[2] for x in fig_r],
                             figdir / "exp020_residence_time.png")

    print(f"\nSaved {log}")
    print(f"Saved {p1}")
    print(f"Saved {p2}")


if __name__ == "__main__":
    main()
