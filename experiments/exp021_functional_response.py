"""Experiment 021 -- resource model comparison: constant vs functional-response intake.

exp020 showed patch-leaving / MVT under CONSTANT intake (a fixed energy gain per
step in contact). Two issues surfaced: (a) the organism leaves a patch when
SATIATED, not hungry, so high intake satiates it, switches off the food drive, and
it wanders off and starves -- the population is not sustainable; (b) with intake
constant, the within-patch intake RATE never diminishes, so leaving is a pure
salience comparison, not a Charnov marginal-rate rule.

This experiment adds the Holling functional response (config.food_intake_scaling
= 'biomass': intake = food_intake_rate * biomass/K) and compares the two regimes
at the SAME nominal rate. Under the functional response a depleted patch yields
diminishing intake, so (a) hunger re-engages foraging -> the population survives,
and (b) the instantaneous intake rate genuinely falls within a patch, so the
give-up RATE = rate * give-up-density falling with travel distance is a literal
marginal-value-theorem signature.

Run:   python experiments/exp021_functional_response.py
Saves: outputs/logs/exp021_functional_response.json
       outputs/figures/exp021_giveup_regimes.png
       outputs/figures/exp021_giveup_rate.png
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from behavioral_md.experiment_utils import (
    compute_mean_ci,
    make_cue_centers,
    make_inert_source,
    save_results_json,
)
from behavioral_md.forage import initial_state, make_forage_sim
from behavioral_md.jax_engine import build_spec
from behavioral_md.visualization import plot_giveup_rate, plot_giveup_regimes
from experiments.exp020_patch_leaving_mvt import (
    N_ORG,
    N_STEPS,
    SEED,
    TRAVEL_DISTANCES,
    _config,
    _patches,
    _residences,
)


def _run_regime(scaling: str):
    """Run the two-patch sweep under one intake-scaling regime; return per-D rows."""
    cfg = _config().model_copy(update={"food_intake_scaling": scaling})
    spec = build_spec(config=cfg)
    cue_centers = make_cue_centers(cfg)
    far = make_inert_source(cfg)
    rows = []
    for dist in TRAVEL_DISTANCES:
        patches = _patches(cfg, dist)
        sim = make_forage_sim(spec, cfg, patches, far, far, far, cue_centers)
        state = initial_state(spec, cfg, N_ORG, 2, patches[0], cfg.n_cue_receptors)
        keys = jax.random.split(jax.random.PRNGKey(SEED), N_STEPS)
        final, (energy, _af, on_patch, biomass) = sim(
            state, keys, jnp.ones(N_ORG, bool), jnp.zeros(N_ORG)
        )
        on_np, bm_np, en_np = np.asarray(on_patch), np.asarray(biomass), np.asarray(energy)
        gs, rs, es = [], [], []
        for o in range(N_ORG):
            g, r, e = _residences(on_np[:, o], bm_np[:, o], en_np[:, o])
            gs += g
            rs += r
            es += e
        gm, gci = compute_mean_ci(gs)
        # Give-up intake RATE (functional response): rate * biomass_frac at leaving.
        rate_vals = [cfg.food_intake_rate * g for g in gs]  # K = 1
        ratem, rateci = compute_mean_ci(rate_vals)
        rows.append({
            "D": dist, "giveup_mean": gm, "giveup_ci": gci, "n": len(gs),
            "giveup_energy": float(np.mean(es)) if es else float("nan"),
            "frac_alive_end": float(np.asarray(final.alive).mean()),
            "giveup_rate_mean": ratem, "giveup_rate_ci": rateci,
        })
    return cfg.food_intake_rate, rows


def main() -> None:
    rate, const_rows = _run_regime("constant")
    _r2, bio_rows = _run_regime("biomass")

    results = {"food_intake_rate": rate, "constant": const_rows, "biomass": bio_rows}
    save_results_json("exp021_functional_response.json", results)

    print(f"{'D':>5} | {'constant: giveup  alive  E':>32} | {'biomass: giveup  alive  E':>32}")
    for c, b in zip(const_rows, bio_rows, strict=True):
        print(
            f"{c['D']:5.1f} | {c['giveup_mean']:.3f}+/-{c['giveup_ci']:.3f}  "
            f"{c['frac_alive_end']:.2f}   {c['giveup_energy']:.2f}      | "
            f"{b['giveup_mean']:.3f}+/-{b['giveup_ci']:.3f}  "
            f"{b['frac_alive_end']:.2f}   {b['giveup_energy']:.2f}"
        )

    d = [r["D"] for r in const_rows]
    figdir = Path("outputs/figures")
    p1 = plot_giveup_regimes(
        d,
        [r["giveup_mean"] for r in const_rows], [r["giveup_ci"] for r in const_rows],
        [r["giveup_mean"] for r in bio_rows], [r["giveup_ci"] for r in bio_rows],
        figdir / "exp021_giveup_regimes.png",
    )
    # Charnov marginal rate: give-up intake rate vs D, functional-response regime.
    p2 = plot_giveup_rate(
        d, [r["giveup_rate_mean"] for r in bio_rows], [r["giveup_rate_ci"] for r in bio_rows],
        figdir / "exp021_giveup_rate.png",
    )

    ca = np.mean([r["frac_alive_end"] for r in const_rows])
    ba = np.mean([r["frac_alive_end"] for r in bio_rows])
    print(f"\nMean end-of-run survival: constant={ca:.2f}  biomass(functional)={ba:.2f}")
    print(f"Saved {p1}")
    print(f"Saved {p2}")


if __name__ == "__main__":
    main()
