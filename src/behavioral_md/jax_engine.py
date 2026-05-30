"""JAX-vectorized engine (foundation): batched force + damped-Verlet step.

This is the fast twin of the object-oriented NumPy engine. The whole population is
held as arrays ([n_organisms, n_atoms, ...]) and one timestep is a pure, batched,
``jit``-able function -- no Python per-atom or per-organism loops. The NumPy
engine (`organism.py`, `forces.py`) remains the canonical, readable reference;
this module is validated to reproduce its force and integration numerically
(`validate_against_numpy`).

Scope so far: the deterministic dynamics -- the two-tier force decomposition
(`compute_force`) and the damped Verlet update (`integrate`). Stochastic emission,
learning, the cue-receptor field, and the environment are added in later phases.

Static model structure (atom sensitivities, valences, directions, coupling) is
packed once into a :class:`ModelSpec`; per-step state (activations, history,
sensory input, energy) flows through the pure functions.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

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
    for i, atom in enumerate(atoms):
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
    """Damped Verlet step; returns (new_activation, new_previous), both clipped."""
    velocity = (activation - previous) / config.dt
    net = force - config.damping_coef * velocity
    new = 2.0 * activation - previous + (net / spec.mass[None, :]) * config.dt**2
    new = jnp.clip(new, config.activation_min, config.activation_max)
    return new, activation


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


if __name__ == "__main__":
    checks = {
        "force": validate_against_numpy(),
        "learning": validate_learning(),
        "emission": validate_emission(),
    }
    for name, diff in checks.items():
        flag = "OK" if diff < 1e-6 else "MISMATCH"
        print(f"{name:10s} max |JAX - NumPy| = {diff:.2e}  {flag}")
