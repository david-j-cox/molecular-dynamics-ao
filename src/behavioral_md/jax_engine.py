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

import jax.numpy as jnp
import numpy as np

from behavioral_md.atoms import STIMULI, default_atom_set
from behavioral_md.config import SimulationConfig
from behavioral_md.forces import default_coupling_matrix

_FOOD = STIMULI.index("food")


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
    return ModelSpec(
        sensitivity=jnp.asarray(sens), direction=jnp.asarray(direction),
        valence=jnp.asarray(valence), stim_channel=jnp.asarray(stim_ch),
        contact_exp=jnp.asarray(cexp), readiness=jnp.asarray(readiness),
        mass=jnp.asarray(mass), coupling=jnp.asarray(coupling),
        is_movement=jnp.asarray(is_move), is_consummatory=jnp.asarray(is_consum),
        is_drive=jnp.asarray(is_drive),
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


if __name__ == "__main__":
    diff = validate_against_numpy()
    print(f"max |JAX force - NumPy force| = {diff:.2e}")
    print("OK -- JAX force matches NumPy" if diff < 1e-6 else "MISMATCH")
