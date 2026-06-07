"""JAX-vectorized engine (foundation): batched force + damped-Verlet step.

This is the fast twin of the object-oriented NumPy engine. The whole population is
held as arrays ([n_organisms, n_atoms, ...]) and one timestep is a pure, batched,
``jit``-able function -- no Python per-atom or per-organism loops. The NumPy
engine (`organism.py`, `forces.py`) remains the canonical, readable reference;
this module is validated to reproduce its force and integration numerically
(`validate_against_numpy`).

Scope: the FULL organism, not just the dynamics. The two-tier force decomposition
(`compute_force`), the integrator (`integrate`: damped Verlet or the first-order leaky
ablation), stochastic softmax emission (`emission_probs`/`sample_actions`), Rescorla-Wagner
learning (`learn_step`) with the cue-receptor field (`learn_with_cue`), the gridworld coupling
(`observe`/`env_step`), and the energy budget are all batched pure functions, assembled into a
jitted one-life rollout (`make_simulate`) and a multi-life loop (`run_lives`). Each component is
validated to reproduce the NumPy organism to floating-point precision (`validate_against_numpy`,
`validate_learning`, `validate_emission`, `validate_env`), and the population runs ~85x faster
(exp004). The NumPy engine (`organism.py`, `forces.py`) remains the canonical, readable reference.

The integrator choice (Verlet / leaky) and the effort->energy / load-bearing fatigue terms (exp059)
ARE supported, all validated against NumPy. Not yet ported: multi-dimensional / oscillator atoms
(exp061; the ModelSpec assumes a scalar state per atom -- a larger refactor, and the multi-dim runs
are small enough not to need the population-scale speed).

Static model structure (atom sensitivities, valences, directions, coupling) is
packed once into a :class:`ModelSpec`; per-step state (activations, history,
sensory input, energy) flows through the pure functions.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from behavioral_md._kernels import exp_falloff, safe_unit
from behavioral_md.atoms import ACTION_ATOMS, STIMULI, default_atom_set
from behavioral_md.config import SimulationConfig
from behavioral_md.forces import default_coupling_matrix

_FOOD = STIMULI.index("food")
_ACTION_IDS = tuple(sorted(ACTION_ATOMS))  # env action ids in emission order


class ModelSpec(NamedTuple):
    """Static, batch-independent description of the atom set (arrays over atoms)."""

    sensitivity: jnp.ndarray      # [A, C]
    direction: jnp.ndarray        # [A, 2]  (0,0 for non-directional)
    valence: jnp.ndarray          # [A]
    stim_channel: jnp.ndarray     # [A] int (channel index of the atom's stimulus)
    contact_exp: jnp.ndarray      # [A]
    readiness: jnp.ndarray        # [A]
    mass: jnp.ndarray             # [A]
    coupling: jnp.ndarray         # [A, A]  (C[i, j] = effect of j on i)
    is_movement: jnp.ndarray      # [A] bool
    is_consummatory: jnp.ndarray  # [A] bool
    is_drive: jnp.ndarray         # [A] bool (non-directional, non-consummatory, valenced)
    action_atom_idx: jnp.ndarray  # [n_actions] atom index of each action atom
    action_ids: jnp.ndarray       # [n_actions] env action id of each action atom
    baseline: jnp.ndarray         # [A] initial/baseline activation
    approach_food_idx: int        # atom index of approach_food (cue drive target)


def build_spec(atoms=None, config: SimulationConfig | None = None) -> ModelSpec:
    """Pack a NumPy atom list (default: `default_atom_set`) into a :class:`ModelSpec`."""
    atoms = atoms if atoms is not None else default_atom_set()
    a = len(atoms)
    c = len(STIMULI)
    sens = np.zeros((a, c))
    direction = np.zeros((a, 2))
    valence = np.zeros(a)
    stim_ch = np.zeros(a, dtype=int)
    cexp = np.ones(a)
    readiness = np.ones(a)
    mass = np.ones(a)
    is_move = np.zeros(a, bool)
    is_consum = np.zeros(a, bool)
    is_drive = np.zeros(a, bool)
    baseline = np.zeros(a)
    for i, atom in enumerate(atoms):
        baseline[i] = atom.baseline_activation
        for j, s in enumerate(STIMULI):
            sens[i, j] = atom.sensitivity.get(s, 0.0)
        cexp[i] = atom.contact_exponent
        readiness[i] = atom.readiness
        mass[i] = atom.mass
        valence[i] = atom.valence
        is_consum[i] = atom.consummatory
        stim_ch[i] = STIMULI.index(atom.stimulus) if atom.stimulus in STIMULI else 0
        if atom.direction is not None:
            direction[i] = np.asarray(atom.direction)
            is_move[i] = True
        is_drive[i] = (atom.direction is None) and (not atom.consummatory) \
            and (atom.stimulus is not None) and (atom.valence != 0.0)
    coupling = default_coupling_matrix(atoms)
    names = [atom.name for atom in atoms]
    action_idx = np.array([names.index(ACTION_ATOMS[i]) for i in _ACTION_IDS])
    return ModelSpec(
        sensitivity=jnp.asarray(sens), direction=jnp.asarray(direction),
        valence=jnp.asarray(valence), stim_channel=jnp.asarray(stim_ch),
        contact_exp=jnp.asarray(cexp), readiness=jnp.asarray(readiness),
        mass=jnp.asarray(mass), coupling=jnp.asarray(coupling),
        is_movement=jnp.asarray(is_move), is_consummatory=jnp.asarray(is_consum),
        is_drive=jnp.asarray(is_drive),
        action_atom_idx=jnp.asarray(action_idx),
        action_ids=jnp.asarray(np.array(_ACTION_IDS)),
        baseline=jnp.asarray(baseline),
        approach_food_idx=names.index("approach_food"),
    )


def deficit_gain(energy: jnp.ndarray, config: SimulationConfig) -> jnp.ndarray:
    """Convex marginal value of the energy deficit, per organism. Shape [O]."""
    deficit = jnp.clip(1.0 - energy / config.energy_capacity, 0.0, None)
    return config.motivational_strength * deficit**config.deficit_exponent


def compute_force(
    spec: ModelSpec,
    activation: jnp.ndarray,   # [O, A]
    history: jnp.ndarray,      # [O, A, C]
    intensity: jnp.ndarray,    # [O, C]
    direction: jnp.ndarray,    # [O, C, 2]  unit vectors to each source
    food_contact: jnp.ndarray, # [O]
    d_gain: jnp.ndarray,       # [O]
) -> jnp.ndarray:
    """Batched two-tier behavioral force; matches `forces.ForceCalculator.compute`."""
    # Effective per-(organism, atom, channel) intensity, contact-gated for the
    # consummatory atoms on the food channel.
    eff = intensity[:, None, :] ** spec.contact_exp[None, :, None]          # [O, A, C]
    eff = eff.at[:, :, _FOOD].set(
        jnp.where(spec.is_consummatory[None, :], food_contact[:, None], eff[:, :, _FOOD])
    )
    s_drive = jnp.einsum("ac,oac->oa", spec.sensitivity, eff)               # [O, A]
    h_drive = jnp.einsum("oac,oac->oa", history, eff)                       # [O, A]

    # Energy-deficit motivational term: food drive atoms (distal) and consummatory.
    is_food_drive = spec.is_drive & (spec.stim_channel == _FOOD)
    m_drive = (
        jnp.where(is_food_drive[None, :], d_gain[:, None] * intensity[:, _FOOD][:, None], 0.0)
        + jnp.where(spec.is_consummatory[None, :], d_gain[:, None] * food_contact[:, None], 0.0)
    )

    # Movement atoms express the drive atoms through the live stimulus geometry:
    # express[o,i] = sum_k is_drive[k]*valence[k]*act[o,k]*(stim_dir[o,chan_k] . dir[i]).
    drive_dir = direction[:, spec.stim_channel, :]                          # [O, A_k, 2]
    dot_ki = jnp.einsum("okd,id->oki", drive_dir, spec.direction)           # [O, A_k, A_i]
    weighted = (spec.is_drive * spec.valence)[None, :, None] * activation[:, :, None]
    express = jnp.sum(weighted * dot_ki, axis=1)                            # [O, A_i]

    sensory = jnp.where(spec.is_movement[None, :], express, s_drive)
    hist = jnp.where(spec.is_movement[None, :], 0.0, h_drive)
    motiv = jnp.where(spec.is_movement[None, :], 0.0, m_drive)
    drive = spec.readiness[None, :] * (sensory + hist + motiv)
    coupling = activation @ spec.coupling.T                                 # [O, A]
    return drive + coupling                                                 # fatigue off by default


def integrate(
    spec: ModelSpec,
    activation: jnp.ndarray,
    previous: jnp.ndarray,
    force: jnp.ndarray,
    config: SimulationConfig,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Integrator step; returns (new_activation, new_previous), both clipped.

    Damped Verlet by default, or the first-order leaky accumulator when
    ``config.integrator == 'leaky'`` (the exp048/exp060 ablation). ``config`` is static at trace
    time, so the branch is resolved when the simulate function is built (no per-step Python branch).
    """
    if config.integrator == "leaky":
        new = activation + config.dt * (force / spec.mass[None, :] - config.leak_coef * activation)
        new = jnp.clip(new, config.activation_min, config.activation_max)
        return new, activation
    velocity = (activation - previous) / config.dt
    net = force - config.damping_coef * velocity
    new = 2.0 * activation - previous + (net / spec.mass[None, :]) * config.dt**2
    new = jnp.clip(new, config.activation_min, config.activation_max)
    return new, activation


# Movement deltas by env action id (0 no-op, 1 up, 2 down, 3 left, 4 right,
# 5 consume, 6 pause), matching gridworld._MOVES.
_DELTAS = jnp.array(
    [[0.0, 0.0], [0.0, 1.0], [0.0, -1.0], [-1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
)


def observe(
    positions: jnp.ndarray,    # [O, 2]
    sources: jnp.ndarray,      # [O, C, 2]  source positions in STIMULI order
    biomass: jnp.ndarray,      # [O]
    config: SimulationConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Batched sensory observation: (intensity [O,C], direction [O,C,2], food_contact [O]).

    Matches `gridworld._build_observation`: exp(-d/range) intensity (food scaled
    by remaining biomass), unit direction to each source, and a biomass-scaled
    food contact signal within consume_radius.
    """
    diff = sources - positions[:, None, :]                          # [O, C, 2]
    dist = jnp.linalg.norm(diff, axis=2)                            # [O, C]
    direction = safe_unit(diff, dist)
    intensity = exp_falloff(dist, config.sensor_range)             # [O, C]
    biomass_frac = biomass / config.food_carrying_capacity         # [O]
    intensity = intensity.at[:, _FOOD].multiply(biomass_frac)
    in_range = dist[:, _FOOD] <= config.consume_radius
    food_contact = jnp.where(in_range, biomass_frac, 0.0)
    return intensity, direction, food_contact


def env_step(
    positions: jnp.ndarray,        # [O, 2]
    action: jnp.ndarray,           # [O] int
    food_pos: jnp.ndarray,         # [O, 2]
    danger_pos: jnp.ndarray,       # [O, 2]
    biomass: jnp.ndarray,          # [O]
    food_reinforces: jnp.ndarray,  # [O] bool
    config: SimulationConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Batched env transition; returns (new_positions, food_intake, danger_contact, new_biomass)."""
    new_pos = jnp.clip(positions + _DELTAS[action], 0.0, config.grid_size - 1)
    d_food = jnp.linalg.norm(new_pos - food_pos, axis=1)
    in_range = (d_food <= config.consume_radius) & food_reinforces
    available = jnp.clip(biomass - config.food_min_biomass, 0.0, None)
    intake = jnp.where(in_range, jnp.minimum(config.food_intake_rate, available), 0.0)
    k = config.food_carrying_capacity
    biomass = biomass - intake
    biomass = biomass + config.food_regrowth_rate * biomass * (1.0 - biomass / k)
    biomass = jnp.clip(biomass, config.food_min_biomass, k)
    d_danger = jnp.linalg.norm(new_pos - danger_pos, axis=1)
    danger_contact = jnp.where(d_danger <= config.consume_radius, 1.0, 0.0)
    return new_pos, intake, danger_contact, biomass


def emission_probs(
    spec: ModelSpec, activation: jnp.ndarray, config: SimulationConfig
) -> jnp.ndarray:
    """Softmax (Luce / matching) probabilities over action atoms. Shape [O, n_actions]."""
    acts = activation[:, spec.action_atom_idx]
    z = acts / config.softmax_temperature
    z = z - z.max(axis=1, keepdims=True)
    p = jnp.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def sample_actions(spec: ModelSpec, probs: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
    """Sample one env action id per organism from the emission probabilities."""
    idx = jax.random.categorical(key, jnp.log(probs))
    return spec.action_ids[idx]


def update_eligibility(eligibility: jnp.ndarray, activation: jnp.ndarray,
                       config: SimulationConfig) -> jnp.ndarray:
    """Recency-weighted trace: e <- decay * e + activation. Shape [O, A]."""
    return config.eligibility_decay * eligibility + activation


def learn_step(
    spec: ModelSpec,
    history: jnp.ndarray,           # [O, A, C]
    eligibility: jnp.ndarray,       # [O, A]
    intensity: jnp.ndarray,         # [O, C]
    appetitive: jnp.ndarray,        # [O]
    aversive: jnp.ndarray,          # [O]
    appetitive_exposure: jnp.ndarray,  # [O] bool
    aversive_exposure: jnp.ndarray,    # [O] bool
    config: SimulationConfig,
) -> jnp.ndarray:
    """Batched valence-split Rescorla-Wagner update (rw_independent); matches
    `learning.RescorlaWagner.update`. Returns new history weights."""
    approach = spec.valence > 0.0                                   # [A]
    is_learner = spec.valence != 0.0                                # [A]
    mag = jnp.where(approach[None, :], appetitive[:, None], aversive[:, None])      # [O, A]
    exposed = jnp.where(approach[None, :], appetitive_exposure[:, None],
                        aversive_exposure[:, None])                  # [O, A]
    gate = is_learner[None, :] & exposed & (eligibility > 0.0)       # [O, A]
    rate = jnp.where(mag > 0.0, config.learning_rate, config.extinction_rate)       # [O, A]
    target = config.reinforcement_asymptote * mag                   # [O, A]
    present = intensity > 1e-6                                       # [O, C]
    dv = (
        gate[:, :, None] * present[:, None, :]
        * rate[:, :, None] * eligibility[:, :, None] * intensity[:, None, :]
        * (target[:, :, None] - history)
    )
    return jnp.clip(history + dv, config.history_weight_min, config.history_weight_max)


_DANGER = STIMULI.index("danger")
_CUE = STIMULI.index("cue")


def drive_integrate_emit(spec, cfg, activation, previous, history, eligibility, fatigue,
                         intensity, direction, contact, energy, cue_weights,
                         cue_value, cue_centers, key):
    """One per-step motor pass, shared by every JAX world (single- and multi-patch).

    Deficit-scaled two-tier force, plus the cue-receptor (Shepard) drive added onto
    approach_food, then damped-Verlet integration, the eligibility-trace update, and
    softmax action emission. Returns
    (new_activation, new_previous, eligibility, action, cue_act, cue_drive); the last
    two are reused by :func:`learn_with_cue`.
    """
    dgain = deficit_gain(energy, cfg)
    force = compute_force(spec, activation, history, intensity, direction, contact, dgain)
    force = force - fatigue                                     # within-bout decrement (0 when off)
    cue_act = intensity[:, _CUE][:, None] * jnp.exp(
        -cfg.cue_generalization_beta * jnp.abs(cue_value[:, None] - cue_centers[None, :])
    )
    cue_drive = jnp.sum(cue_weights * cue_act, axis=1)
    force = force.at[:, spec.approach_food_idx].add(cue_drive)
    new_act, new_prev = integrate(spec, activation, previous, force, cfg)
    # Fatigue accrues with positive activation and recovers (matches organism._update_fatigue);
    # stays identically 0 when fatigue_gain == 0, so the default engine is unchanged.
    new_fatigue = cfg.fatigue_decay * fatigue + cfg.fatigue_gain * jnp.clip(new_act, 0.0, None)
    elig = update_eligibility(eligibility, new_act, cfg)
    action = sample_actions(spec, emission_probs(spec, new_act, cfg), key)
    return new_act, new_prev, new_fatigue, elig, action, cue_act, cue_drive


def learn_with_cue(spec, cfg, history, eligibility, cue_weights, intensity, contact,
                   danger_contact, appetitive, aversive, cue_act, cue_drive):
    """One per-step learning pass, shared by every JAX world.

    Valence-split Rescorla-Wagner on the history weights (:func:`learn_step`) plus the
    cue-receptor weight update (error toward lambda*reinforcement, gated on food
    contact). Returns (new_history, new_cue_weights).
    """
    new_hist = learn_step(spec, history, eligibility, intensity, appetitive, aversive,
                          contact > 0, danger_contact > 0, cfg)
    elig_af = jnp.clip(eligibility[:, spec.approach_food_idx], 0.0, None)
    cue_err = cfg.reinforcement_asymptote * appetitive - cue_drive
    cue_upd = jnp.where((contact > 0)[:, None],
                        cfg.cue_learning_rate * elig_af[:, None] * cue_act * cue_err[:, None], 0.0)
    new_cue_w = jnp.clip(cue_weights + cue_upd, cfg.history_weight_min, cfg.history_weight_max)
    return new_hist, new_cue_w


class SimState(NamedTuple):
    """Per-organism simulation state (one entry per organism in every array)."""

    positions: jnp.ndarray   # [O, 2]
    biomass: jnp.ndarray     # [O]
    energy: jnp.ndarray      # [O]
    alive: jnp.ndarray       # [O] bool
    activation: jnp.ndarray  # [O, A]
    previous: jnp.ndarray    # [O, A]
    history: jnp.ndarray     # [O, A, C]
    eligibility: jnp.ndarray # [O, A]
    fatigue: jnp.ndarray     # [O, A]  within-bout response decrement (0 when fatigue_gain == 0)
    cue_weights: jnp.ndarray # [O, K]
    cause_of_death: jnp.ndarray  # [O] int: 0 alive, 1 starvation, 2 danger

# Cause-of-death codes (shared with metrics).
CAUSE_ALIVE, CAUSE_STARVATION, CAUSE_DANGER = 0, 1, 2


def initial_state(spec, cfg, n_org, position, n_receptors):
    """Construct a fresh SimState for ``n_org`` organisms at a start position."""
    a, c = spec.sensitivity.shape
    act0 = jnp.broadcast_to(spec.baseline, (n_org, a))
    pos0 = jnp.broadcast_to(jnp.asarray(position, float), (n_org, 2))
    return SimState(
        positions=pos0, biomass=jnp.full(n_org, cfg.food_carrying_capacity),
        energy=jnp.full(n_org, cfg.energy_init), alive=jnp.ones(n_org, bool),
        activation=act0, previous=act0, history=jnp.zeros((n_org, a, c)),
        eligibility=jnp.zeros((n_org, a)), fatigue=jnp.zeros((n_org, a)),
        cue_weights=jnp.zeros((n_org, n_receptors)),
        cause_of_death=jnp.zeros(n_org, dtype=jnp.int32),
    )


def make_simulate(spec, cfg, sources, cue_centers):
    """Return a jitted ``sim(state, keys, food_reinforces, cue_value)`` for one life.

    Closes over the static model/config/layout so atom arrays and config scalars
    become compile-time constants; `food_reinforces` and `cue_value` are runtime
    arguments so they can vary across lives (e.g. extinction, discrimination)
    without recompiling. `lax.scan` runs all timesteps for the whole population as
    one fused, batched computation.
    """
    food_pos, danger_pos = sources[:, _FOOD], sources[:, _DANGER]

    def step(carry, key_t):
        state, food_reinforces, cue_value = carry
        intensity, direction, contact = observe(state.positions, sources, state.biomass, cfg)
        new_act, new_prev, new_fatigue, elig, action, cue_act, cue_drive = drive_integrate_emit(
            spec, cfg, state.activation, state.previous, state.history, state.eligibility,
            state.fatigue, intensity, direction, contact, state.energy, state.cue_weights,
            cue_value, cue_centers, key_t,
        )
        new_pos, intake, danger_c, new_bio = env_step(
            state.positions, action, food_pos, danger_pos, state.biomass, food_reinforces, cfg
        )
        intensity2, _d2, contact2 = observe(new_pos, sources, new_bio, cfg)

        moved = (action >= 1) & (action <= 4)
        cost = cfg.basal_metabolism + jnp.where(moved, cfg.move_cost, cfg.rest_cost)
        # Effort -> energy (exp059): summed positive action-atom activation costs energy; fatigue is
        # a metabolic load. Both default to 0, so the default engine is unchanged.
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

        a1, a2 = state.alive, state.alive[:, None]  # freeze dead organisms
        new_alive = state.alive & (new_energy > 0.0)
        # Record cause at the death transition (was alive, now not): danger if in
        # danger contact this step, else starvation. Preserve prior causes.
        just_died = state.alive & (~new_alive)
        cause_now = jnp.where(danger_c > 0, CAUSE_DANGER, CAUSE_STARVATION)
        cause = jnp.where(just_died, cause_now, state.cause_of_death)
        new = SimState(
            positions=jnp.where(a2, new_pos, state.positions),
            biomass=jnp.where(a1, new_bio, state.biomass),
            energy=jnp.where(a1, new_energy, state.energy),
            alive=new_alive,
            activation=jnp.where(a2, new_act, state.activation),
            previous=jnp.where(a2, new_prev, state.previous),
            history=jnp.where(a1[:, None, None], new_hist, state.history),
            eligibility=jnp.where(a2, elig, state.eligibility),
            fatigue=jnp.where(a2, new_fatigue, state.fatigue),
            cue_weights=jnp.where(a2, new_cue_w, state.cue_weights),
            cause_of_death=cause,
        )
        # Emit PRESENCE at the food location (contact2 > 0) for LIVING organisms.
        # Presence is the behavioral approach measure and stays meaningful under
        # extinction (food present but non-reinforcing), where intake is 0 by
        # construction. Masking by `alive` avoids counting an organism frozen at
        # the food location after death.
        at_food_flag = ((contact2 > 0) & new.alive).astype(jnp.float32)
        return (new, food_reinforces, cue_value), (new.energy, at_food_flag)

    def scanned(state: SimState, keys, food_reinforces, cue_value):
        """Run len(keys) timesteps; return (final_state, (energy[T,O], at_food[T,O]))."""
        (final, _fr, _cv), out = jax.lax.scan(step, (state, food_reinforces, cue_value), keys)
        return final, out

    return jax.jit(scanned)


def reset_for_life(state: SimState, spec, cfg, position) -> SimState:
    """Start a new life: reset body/energy/position, KEEP learned history + cue weights."""
    n_org = state.energy.shape[0]
    a = spec.sensitivity.shape[0]
    act0 = jnp.broadcast_to(spec.baseline, (n_org, a))
    pos0 = jnp.broadcast_to(jnp.asarray(position, float), (n_org, 2))
    return state._replace(
        positions=pos0, biomass=jnp.full(n_org, cfg.food_carrying_capacity),
        energy=jnp.full(n_org, cfg.energy_init), alive=jnp.ones(n_org, bool),
        activation=act0, previous=act0, eligibility=jnp.zeros((n_org, a)),
        fatigue=jnp.zeros((n_org, a)),
        cause_of_death=jnp.zeros(n_org, dtype=jnp.int32),
    )


def run_lives(sim, spec, cfg, state0, position, n_lives, n_steps, key,
              food_reinforces=None, cue_value=None):
    """Run ``n_lives`` lives (learning carries over); return per-life metrics.

    ``sim`` is the jitted function from :func:`make_simulate`. ``food_reinforces``
    is a per-life bool schedule (length ``n_lives``; default all True) and
    ``cue_value`` a per-life cue value (default all 0). Returns a dict of
    [n_lives, n_org] arrays: latency, contact count, steps survived, and the
    end-of-life ``hw_food`` (approach_food's learned food-channel weight).
    """
    n_org = state0.energy.shape[0]
    food_idx = STIMULI.index("food")
    af = spec.approach_food_idx
    fr_sched = [True] * n_lives if food_reinforces is None else list(food_reinforces)
    cv_sched = [0.0] * n_lives if cue_value is None else list(cue_value)

    latency, contact, survived, hw_food, cause = [], [], [], [], []
    state = state0
    for life in range(n_lives):
        state = reset_for_life(state, spec, cfg, position)
        keys = jax.random.split(jax.random.fold_in(key, life), n_steps)
        fr = jnp.full(n_org, bool(fr_sched[life]))
        cv = jnp.full(n_org, float(cv_sched[life]))
        state, (energy, at_food) = sim(state, keys, fr, cv)
        reached = at_food.sum(axis=0) > 0
        latency.append(jnp.where(reached, jnp.argmax(at_food, axis=0), n_steps))
        contact.append(at_food.sum(axis=0))
        # Time-to-death = first step energy hits 0; n_steps if it survived.
        died = energy <= 0.0                                   # [T, O]
        ever_died = died.any(axis=0)
        survived.append(jnp.where(ever_died, jnp.argmax(died, axis=0), n_steps))
        hw_food.append(state.history[:, af, food_idx])
        cause.append(state.cause_of_death)                     # 0 alive / 1 starve / 2 danger
    return {
        "latency": np.asarray(jnp.stack(latency)),     # [n_lives, n_org]
        "contact": np.asarray(jnp.stack(contact)),
        "survived": np.asarray(jnp.stack(survived)),   # time-to-death (steps)
        "hw_food": np.asarray(jnp.stack(hw_food)),
        "cause_of_death": np.asarray(jnp.stack(cause)),
    }


def validate_against_numpy(n_org: int = 4, seed: int = 0) -> float:
    """Return the max abs difference in force vs the NumPy engine on random state."""
    from behavioral_md.forces import ForceCalculator, SensoryField

    rng = np.random.default_rng(seed)
    cfg = SimulationConfig()
    atoms = default_atom_set()
    spec = build_spec(atoms, cfg)
    a, c = len(atoms), len(STIMULI)

    act = rng.normal(size=(n_org, a))
    hist = rng.normal(size=(n_org, a, c)) * 0.5
    inten = rng.uniform(0.1, 1.0, size=(n_org, c))
    ang = rng.uniform(0, 2 * np.pi, size=(n_org, c))
    sdir = np.stack([np.cos(ang), np.sin(ang)], axis=-1)  # [O, C, 2] unit vectors
    contact = rng.uniform(0, 1, size=n_org)
    energy = rng.uniform(0, 1, size=n_org)

    jax_force = np.asarray(compute_force(
        spec, jnp.asarray(act), jnp.asarray(hist), jnp.asarray(inten),
        jnp.asarray(sdir), jnp.asarray(contact), deficit_gain(jnp.asarray(energy), cfg),
    ))

    np_force = np.zeros((n_org, a))
    fc = ForceCalculator(atoms, config=cfg)
    for o in range(n_org):
        for i, atom in enumerate(atoms):
            atom.state = np.array([act[o, i]])
            atom.history_weights = {s: hist[o, i, j] for j, s in enumerate(STIMULI)}
        sensory = {
            s: SensoryField(direction=sdir[o, j], intensity=float(inten[o, j]),
                            contact=float(contact[o]) if s == "food" else 0.0)
            for j, s in enumerate(STIMULI)
        }
        np_force[o], _ = fc.compute(sensory, float(energy[o]))
    return float(np.max(np.abs(jax_force - np_force)))


def validate_learning(n_org: int = 4, seed: int = 1) -> float:
    """Max abs difference in updated history weights vs NumPy RescorlaWagner."""
    from behavioral_md.learning import EligibilityTrace, RescorlaWagner

    rng = np.random.default_rng(seed)
    cfg = SimulationConfig()
    atoms = default_atom_set()
    spec = build_spec(atoms, cfg)
    a, c = len(atoms), len(STIMULI)

    elig = rng.uniform(0, 2, size=(n_org, a))
    hist = rng.normal(size=(n_org, a, c)) * 0.3
    inten = rng.uniform(0, 1, size=(n_org, c)) * (rng.uniform(size=(n_org, c)) > 0.3)
    app = (rng.uniform(size=n_org) > 0.5).astype(float)
    avr = (rng.uniform(size=n_org) > 0.5).astype(float)
    app_exp = rng.uniform(size=n_org) > 0.3
    avr_exp = rng.uniform(size=n_org) > 0.3

    jax_hist = np.asarray(learn_step(
        spec, jnp.asarray(hist), jnp.asarray(elig), jnp.asarray(inten),
        jnp.asarray(app), jnp.asarray(avr), jnp.asarray(app_exp), jnp.asarray(avr_exp), cfg,
    ))

    rule = RescorlaWagner(cfg)
    np_hist = hist.copy()
    for o in range(n_org):
        for i, atom in enumerate(atoms):
            atom.history_weights = {s: np_hist[o, i, j] for j, s in enumerate(STIMULI)}
        trace = EligibilityTrace(a, cfg.eligibility_decay)
        trace.trace = elig[o].copy()
        intensities = {s: float(inten[o, j]) for j, s in enumerate(STIMULI)}
        rule.update(atoms, trace, intensities, float(app[o]), float(avr[o]),
                    appetitive_exposure=bool(app_exp[o]), aversive_exposure=bool(avr_exp[o]))
        for i, atom in enumerate(atoms):
            for j, s in enumerate(STIMULI):
                np_hist[o, i, j] = atom.history_weights[s]
    return float(np.max(np.abs(jax_hist - np_hist)))


def validate_emission(n_org: int = 5, seed: int = 2) -> float:
    """Max abs difference in softmax emission probabilities vs the NumPy rule."""
    rng = np.random.default_rng(seed)
    cfg = SimulationConfig()
    atoms = default_atom_set()
    spec = build_spec(atoms, cfg)
    act = rng.normal(size=(n_org, len(atoms))) * 3
    jax_p = np.asarray(emission_probs(spec, jnp.asarray(act), cfg))
    idx = np.asarray(spec.action_atom_idx)
    z = act[:, idx] / cfg.softmax_temperature
    z = z - z.max(axis=1, keepdims=True)
    np_p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    return float(np.max(np.abs(jax_p - np_p)))


def validate_env(n_org: int = 8, seed: int = 3) -> float:
    """Max abs difference vs the NumPy gridworld over random position+action+biomass."""
    from behavioral_md.environments import BehavioralFieldEnv

    rng = np.random.default_rng(seed)
    cfg = SimulationConfig(grid_size=10)
    layout = {"position": [1, 1], "food": [6, 6], "danger": [2, 8], "light": [0, 9], "cue": [8, 3]}
    env = BehavioralFieldEnv(cfg)
    env.reset(seed=seed, options={"layout": layout})
    sources_np = np.stack([env.food_pos, env.danger_pos, env.light_pos, env.cue_pos])  # [C,2]
    sources = jnp.asarray(np.broadcast_to(sources_np, (n_org, *sources_np.shape)))

    pos = rng.integers(0, cfg.grid_size, size=(n_org, 2)).astype(float)
    action = rng.integers(0, 7, size=n_org)
    biomass = rng.uniform(cfg.food_min_biomass, cfg.food_carrying_capacity, size=n_org)
    food_pos = jnp.asarray(np.broadcast_to(env.food_pos, (n_org, 2)))
    danger_pos = jnp.asarray(np.broadcast_to(env.danger_pos, (n_org, 2)))

    new_pos, intake, danger, new_bio = env_step(
        jnp.asarray(pos), jnp.asarray(action), food_pos, danger_pos, jnp.asarray(biomass),
        jnp.ones(n_org, bool), cfg,
    )
    inten, direction, contact = observe(new_pos, sources, new_bio, cfg)

    diffs = []
    for o in range(n_org):
        env.position = pos[o].copy()
        env.food_biomass = float(biomass[o])
        obs, _r, _t, _tr, info = env.step(int(action[o]))
        diffs += [
            np.abs(np.asarray(new_pos[o]) - env.position).max(),
            abs(float(intake[o]) - info["food_intake"]),
            abs(float(danger[o]) - info["danger_contact"]),
            abs(float(new_bio[o]) - info["food_biomass"]),
            abs(float(contact[o]) - float(obs["food_contact"][0])),
        ]
        for j, s in enumerate(STIMULI):
            diffs.append(abs(float(inten[o, j]) - float(obs[f"{s}_intensity"][0])))
            diffs.append(np.abs(np.asarray(direction[o, j]) - obs[f"{s}_vector"]).max())
    return float(np.max(diffs))


if __name__ == "__main__":
    checks = {
        "force": validate_against_numpy(),
        "learning": validate_learning(),
        "emission": validate_emission(),
        "environment": validate_env(),
    }
    for name, diff in checks.items():
        flag = "OK" if diff < 1e-6 else "MISMATCH"
        print(f"{name:10s} max |JAX - NumPy| = {diff:.2e}  {flag}")
