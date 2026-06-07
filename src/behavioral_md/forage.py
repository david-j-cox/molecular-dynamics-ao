"""Multi-patch foraging in the JAX survival world (patch-leaving / marginal value).

Extends the single-patch energy-budget world to P depleting/regrowing food patches.
The organism approaches the most SALIENT patch, where

    salience = exp(-distance / sensor_range) * (biomass / carrying_capacity)

so close, full patches pull hardest. As the organism feeds, the current patch's
biomass (hence its salience) falls; when a fuller or nearer alternative overtakes
it, the organism leaves and travels. This derives the marginal-value theorem -- a
patch is abandoned once its biomass_frac drops below exp(-D/range) for a
competitor patch at distance D, so farther alternatives give a LOWER give-up
density and a LONGER residence time.

This is a faithful generalization of :func:`jax_engine.make_simulate`: identical
two-tier force, damped Verlet, Rescorla-Wagner learning, cue-receptor drive,
energy budget, and death bookkeeping. ONLY the food channel becomes multi-patch:
its intensity/direction/contact are the salience-argmax over P patches instead of
the single source. With P == 1 it reduces exactly to ``make_simulate`` (see
:func:`validate_against_single_patch`). Danger/light/cue stay single sources.
Vectorized over organisms; patch geometry is static, biomass is per-organism [O, P].
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from behavioral_md._kernels import exp_falloff, safe_unit
from behavioral_md.config import SimulationConfig
from behavioral_md.jax_engine import (
    _DELTAS,
    CAUSE_DANGER,
    CAUSE_STARVATION,
    drive_integrate_emit,
    learn_with_cue,
)


class ForageState(NamedTuple):
    """Per-organism state. Mirrors :class:`jax_engine.SimState`, but ``biomass``
    is per-patch [O, P] and ``on_patch`` records the in-range patch (-1 = none)."""

    positions: jnp.ndarray    # [O, 2]
    biomass: jnp.ndarray      # [O, P]  per-patch biomass
    energy: jnp.ndarray       # [O]
    alive: jnp.ndarray        # [O] bool
    activation: jnp.ndarray   # [O, A]
    previous: jnp.ndarray     # [O, A]
    history: jnp.ndarray      # [O, A, C]
    eligibility: jnp.ndarray  # [O, A]
    fatigue: jnp.ndarray      # [O, A]  (0 when fatigue_gain == 0)
    cue_weights: jnp.ndarray  # [O, K]
    on_patch: jnp.ndarray     # [O] int patch in range, -1 if none
    cause_of_death: jnp.ndarray  # [O] int 0 alive / 1 starvation / 2 danger


def initial_state(spec, cfg, n_org, n_patches, position, n_receptors):
    """Fresh state: every patch at carrying capacity, organism at ``position``."""
    a, c = spec.sensitivity.shape
    act0 = jnp.broadcast_to(spec.baseline, (n_org, a))
    pos0 = jnp.broadcast_to(jnp.asarray(position, float), (n_org, 2))
    return ForageState(
        positions=pos0,
        biomass=jnp.full((n_org, n_patches), cfg.food_carrying_capacity),
        energy=jnp.full(n_org, cfg.energy_init),
        alive=jnp.ones(n_org, bool),
        activation=act0, previous=act0,
        history=jnp.zeros((n_org, a, c)),
        eligibility=jnp.zeros((n_org, a)),
        fatigue=jnp.zeros((n_org, a)),
        cue_weights=jnp.zeros((n_org, n_receptors)),
        on_patch=jnp.full(n_org, -1, dtype=jnp.int32),
        cause_of_death=jnp.zeros(n_org, dtype=jnp.int32),
    )


def make_forage_sim(spec, cfg, patches, danger_pos, light_pos, cue_pos, cue_centers):
    """Return a jitted ``sim(state, keys, food_reinforces, cue_value)`` for one life.

    ``patches`` [P,2] and ``danger_pos``/``light_pos``/``cue_pos`` [2] are static
    geometry. ``food_reinforces`` [O] bool and ``cue_value`` [O] are per-life
    runtime args (as in :func:`jax_engine.make_simulate`). Per-step trace returned:
    (energy[T,O], at_food[T,O], on_patch[T,O], biomass[T,O,P]).
    """
    patches = jnp.asarray(patches, float)                       # [P, 2]
    danger_pos = jnp.asarray(danger_pos, float)
    light_pos = jnp.asarray(light_pos, float)
    cue_pos = jnp.asarray(cue_pos, float)
    k_cap = cfg.food_carrying_capacity

    def observe(pos, biomass):
        """Multi-patch sensory observation, returns the C-channel arrays plus the
        in-range patch index. Food channel = salience-argmax over patches."""
        oi = jnp.arange(pos.shape[0])
        # Food: per-patch salience; head to the single best patch.
        diff = patches[None, :, :] - pos[:, None, :]            # [O, P, 2]
        dist = jnp.linalg.norm(diff, axis=2)                    # [O, P]
        unit = safe_unit(diff, dist)
        bm_frac = biomass / k_cap                               # [O, P]
        salience = exp_falloff(dist, cfg.sensor_range) * bm_frac  # [O, P]
        best = jnp.argmax(salience, axis=1)                     # [O]
        food_dir = unit[oi, best]                               # [O, 2]
        food_int = salience[oi, best]                           # [O]
        # In-range patch with the most biomass (patches placed apart -> at most one).
        in_range = dist <= cfg.consume_radius                   # [O, P]
        on = jnp.argmax(jnp.where(in_range, biomass, -1.0), axis=1)
        any_in = in_range.any(axis=1)
        on_patch = jnp.where(any_in, on, -1).astype(jnp.int32)
        food_contact = jnp.where(any_in, bm_frac[oi, on], 0.0)

        def chan(src):
            d = src[None, :] - pos
            di = jnp.linalg.norm(d, axis=1)
            return exp_falloff(di, cfg.sensor_range), safe_unit(d, di)

        dint, ddir = chan(danger_pos)
        lint, ldir = chan(light_pos)
        cint, cdir = chan(cue_pos)
        intensity = jnp.stack([food_int, dint, lint, cint], axis=1)   # [O, C]
        direction = jnp.stack([food_dir, ddir, ldir, cdir], axis=1)   # [O, C, 2]
        return intensity, direction, food_contact, on_patch

    def step(carry, key_t):
        state, food_reinforces, cue_value = carry
        intensity, direction, contact, _on = observe(state.positions, state.biomass)
        new_act, new_prev, new_fatigue, elig, action, cue_act, cue_drive = drive_integrate_emit(
            spec, cfg, state.activation, state.previous, state.history, state.eligibility,
            state.fatigue, intensity, direction, contact, state.energy, state.cue_weights,
            cue_value, cue_centers, key_t,
        )

        # Move, then resolve food at the new position.
        new_pos = jnp.clip(state.positions + _DELTAS[action], 0.0, cfg.grid_size - 1)
        intensity2, _d2, contact2, on_patch = observe(new_pos, state.biomass)

        oi = jnp.arange(new_pos.shape[0])
        feed = (on_patch >= 0) & food_reinforces
        idx = jnp.clip(on_patch, 0, None)
        avail = jnp.clip(state.biomass[oi, idx] - cfg.food_min_biomass, 0.0, None)
        # Functional response: 'constant' = fixed rate while in contact (time-at-
        # patch); 'biomass' = rate scales with biomass fraction (Holling type I),
        # so a depleted patch yields diminishing intake -> hunger re-engages.
        rate = cfg.food_intake_rate
        if cfg.food_intake_scaling == "biomass":
            rate = rate * (state.biomass[oi, idx] / k_cap)
        per = jnp.where(feed, jnp.minimum(rate, avail), 0.0)
        biomass = state.biomass.at[oi, idx].add(-per)
        biomass = biomass + cfg.food_regrowth_rate * biomass * (1.0 - biomass / k_cap)
        biomass = jnp.clip(biomass, cfg.food_min_biomass, k_cap)
        intake = per

        d_danger = jnp.linalg.norm(new_pos - danger_pos[None, :], axis=1)
        danger_c = jnp.where(d_danger <= cfg.consume_radius, 1.0, 0.0)

        moved = (action >= 1) & (action <= 4)
        cost = cfg.basal_metabolism + jnp.where(moved, cfg.move_cost, cfg.rest_cost)
        effort = jnp.clip(new_act[:, spec.action_atom_idx], 0.0, None).sum(axis=1)
        cost = cost + cfg.effort_cost * effort + cfg.fatigue_energy_cost * new_fatigue.sum(axis=1)
        new_energy = jnp.clip(
            state.energy + intake - cfg.danger_energy_loss * danger_c - cost,
            0.0, cfg.energy_capacity,
        )
        appetitive = (intake > 0).astype(jnp.float32)
        aversive = (danger_c > 0).astype(jnp.float32)
        new_hist, new_cue_w = learn_with_cue(
            spec, cfg, state.history, elig, state.cue_weights, intensity2, contact2,
            danger_c, appetitive, aversive, cue_act, cue_drive,
        )

        a1, a2 = state.alive, state.alive[:, None]
        new_alive = state.alive & (new_energy > 0.0)
        just_died = state.alive & (~new_alive)
        cause_now = jnp.where(danger_c > 0, CAUSE_DANGER, CAUSE_STARVATION)
        cause = jnp.where(just_died, cause_now, state.cause_of_death)
        new = ForageState(
            positions=jnp.where(a2, new_pos, state.positions),
            biomass=jnp.where(a1[:, None], biomass, state.biomass),
            energy=jnp.where(a1, new_energy, state.energy),
            alive=new_alive,
            activation=jnp.where(a2, new_act, state.activation),
            previous=jnp.where(a2, new_prev, state.previous),
            history=jnp.where(a1[:, None, None], new_hist, state.history),
            eligibility=jnp.where(a2, elig, state.eligibility),
            fatigue=jnp.where(a2, new_fatigue, state.fatigue),
            cue_weights=jnp.where(a2, new_cue_w, state.cue_weights),
            on_patch=jnp.where(state.alive, on_patch, -1),
            cause_of_death=cause,
        )
        at_food_flag = ((contact2 > 0) & new.alive).astype(jnp.float32)
        return (new, food_reinforces, cue_value), (
            new.energy, at_food_flag, new.on_patch, new.biomass,
        )

    def scanned(state, keys, food_reinforces, cue_value):
        (final, _fr, _cv), out = jax.lax.scan(step, (state, food_reinforces, cue_value), keys)
        return final, out

    return jax.jit(scanned)


def validate_against_single_patch(n_org: int = 6, n_steps: int = 200, seed: int = 0) -> float:
    """With P == 1, the multi-patch sim must reproduce ``jax_engine.make_simulate``.

    Returns the max abs difference in the energy + at-food traces across the run.
    """
    from behavioral_md.jax_engine import build_spec, make_simulate
    from behavioral_md.jax_engine import initial_state as single_initial

    cfg = SimulationConfig(grid_size=10)
    spec = build_spec(config=cfg)
    layout = {"food": [6, 6], "danger": [2, 8], "light": [0, 9], "cue": [8, 3]}
    start = [1, 1]
    cue_centers = jnp.linspace(-2.0, 2.0, 5)
    n_recep = cue_centers.shape[0]

    sources_np = np.stack([layout["food"], layout["danger"], layout["light"], layout["cue"]])
    sources = jnp.asarray(np.broadcast_to(sources_np, (n_org, 4, 2)).astype(float))
    single_sim = make_simulate(spec, cfg, sources, cue_centers)
    s0 = single_initial(spec, cfg, n_org, start, n_recep)

    forage_sim = make_forage_sim(
        spec, cfg, np.asarray([layout["food"]], float),
        layout["danger"], layout["light"], layout["cue"], cue_centers,
    )
    f0 = initial_state(spec, cfg, n_org, 1, start, n_recep)

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_steps)
    fr = jnp.ones(n_org, bool)
    cv = jnp.zeros(n_org)

    _sf, (e1, af1) = single_sim(s0, keys, fr, cv)
    _ff, (e2, af2, _on, _bm) = forage_sim(f0, keys, fr, cv)
    return float(max(np.max(np.abs(np.asarray(e1) - np.asarray(e2))),
                     np.max(np.abs(np.asarray(af1) - np.asarray(af2)))))


if __name__ == "__main__":
    diff = validate_against_single_patch()
    flag = "OK" if diff < 1e-6 else "MISMATCH"
    print(f"P=1 vs make_simulate   max |diff| = {diff:.2e}  {flag}")
