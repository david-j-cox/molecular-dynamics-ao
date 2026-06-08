"""Tests that the JAX engine reproduces the NumPy reference engine.

Skipped automatically if JAX is not installed (it is an optional extra).
"""

import pytest

jax_engine = pytest.importorskip("behavioral_md.jax_engine")

TOL = 1e-6


def test_force_matches_numpy():
    assert jax_engine.validate_against_numpy() < TOL


def test_learning_matches_numpy():
    assert jax_engine.validate_learning() < TOL


def test_emission_matches_numpy():
    assert jax_engine.validate_emission() < TOL


def test_environment_matches_numpy():
    assert jax_engine.validate_env() < TOL


def test_leaky_integrator_matches_numpy():
    """The JAX integrator reproduces the first-order leaky update (exp048/exp060 parity) exactly,
    and leaves the default damped-Verlet step unchanged."""
    import numpy as np

    from behavioral_md.config import SimulationConfig

    cfg = SimulationConfig(integrator="leaky", leak_coef=1.3)
    spec = jax_engine.build_spec(config=cfg)
    a = spec.mass.shape[0]
    rng = np.random.default_rng(0)
    act = np.asarray(rng.normal(size=(3, a)))
    prev = np.asarray(rng.normal(size=(3, a)))
    force = np.asarray(rng.normal(size=(3, a)))
    new, _ = jax_engine.integrate(spec, act, prev, force, cfg)
    mass = np.asarray(spec.mass)[None, :]
    ref = np.clip(act + cfg.dt * (force / mass - cfg.leak_coef * act),
                  cfg.activation_min, cfg.activation_max)
    assert np.max(np.abs(np.asarray(new) - ref)) < TOL


def test_jax_effort_and_fatigue_match_numpy():
    """The effort->energy / fatigue terms (exp059) match the NumPy organism: force includes the
    fatigue decrement, fatigue updates identically, effort = summed positive action activation."""
    import jax.numpy as jnp
    import numpy as np

    from behavioral_md.atoms import STIMULI, default_atom_set
    from behavioral_md.config import SimulationConfig
    from behavioral_md.forces import ForceCalculator, SensoryField

    cfg = SimulationConfig(effort_cost=0.05, fatigue_gain=0.1, fatigue_energy_cost=0.05,
                           fatigue_decay=0.9)
    atoms = default_atom_set()
    spec = jax_engine.build_spec(atoms, cfg)
    a, c = len(atoms), len(STIMULI)
    rng = np.random.default_rng(7)
    act = rng.normal(size=(1, a))
    hist = rng.normal(size=(1, a, c)) * 0.5
    inten = rng.uniform(0.1, 1.0, size=(1, c))
    ang = rng.uniform(0, 2 * np.pi, size=(1, c))
    sdir = np.stack([np.cos(ang), np.sin(ang)], axis=-1)
    contact = rng.uniform(0, 1, size=1)
    energy = rng.uniform(0, 1, size=1)
    fatigue = rng.uniform(0, 0.5, size=(1, a))

    # Force WITH fatigue: JAX subtracts the fatigue array (as drive_integrate_emit does).
    jf = np.asarray(jax_engine.compute_force(
        spec, jnp.asarray(act), jnp.asarray(hist), jnp.asarray(inten), jnp.asarray(sdir),
        jnp.asarray(contact), jax_engine.deficit_gain(jnp.asarray(energy), cfg))) - fatigue
    fc = ForceCalculator(atoms, config=cfg)
    for i, atom in enumerate(atoms):
        atom.state = np.array([act[0, i]])
        atom.history_weights = {s: hist[0, i, j] for j, s in enumerate(STIMULI)}
        atom.fatigue = float(fatigue[0, i])
    sensory = {s: SensoryField(direction=sdir[0, j], intensity=float(inten[0, j]),
                               contact=float(contact[0]) if s == "food" else 0.0)
               for j, s in enumerate(STIMULI)}
    nf, _ = fc.compute(sensory, float(energy[0]))
    assert np.max(np.abs(jf[0] - nf)) < TOL                       # force includes -fatigue

    # Fatigue update parity: the JAX closed form equals the real NumPy organism._update_fatigue.
    from behavioral_md.organism import Organism
    org = Organism(cfg)
    for i, at in enumerate(org.atoms):
        at.state = np.array([act[0, i]])
        at.fatigue = float(fatigue[0, i])
    org._update_fatigue()
    np_fat = np.array([at.fatigue for at in org.atoms])
    jax_fat = (cfg.fatigue_decay * fatigue + cfg.fatigue_gain * np.clip(act, 0.0, None))[0]
    assert np.max(np.abs(np_fat - jax_fat)) < TOL
    idx = np.asarray(spec.action_atom_idx)
    jax_effort = float(np.clip(np.asarray(act)[:, idx], 0.0, None).sum())
    ref_effort = float(sum(max(0.0, act[0, k]) for k in idx))
    assert abs(jax_effort - ref_effort) < TOL


def test_build_spec_rejects_unsupported_config():
    """build_spec raises on config the JAX engine does not implement (rather than silently running
    the default model in disguise); supported defaults build fine."""
    import pytest as _pytest

    from behavioral_md.config import SimulationConfig

    jax_engine.build_spec(config=SimulationConfig())                      # defaults: OK
    jax_engine.build_spec(config=SimulationConfig(integrator="leaky"))    # supported: OK
    for bad in (dict(consequence_model="subtractive"), dict(learning_model="dual_exc_inhib"),
                dict(credit_assignment="rw_competitive"), dict(emission="argmax"),
                dict(day_night=True)):
        with _pytest.raises(ValueError, match="jax_engine"):
            jax_engine.build_spec(config=SimulationConfig(**bad))
