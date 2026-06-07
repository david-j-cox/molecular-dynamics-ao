"""Tests for the Organism: action emission, energy bookkeeping, death."""

from behavioral_md.config import SimulationConfig
from behavioral_md.environments import ACTIONS, BehavioralFieldEnv
from behavioral_md.organism import Organism


def _fresh(**cfg_kw):
    cfg = SimulationConfig(**cfg_kw)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg)
    obs, _ = env.reset(seed=0)
    org.reset(obs)
    return cfg, env, org, obs


def test_emits_valid_action():
    cfg, env, org, obs = _fresh(grid_size=8)
    org.step(obs)
    action = org.emit_action()
    assert action in ACTIONS


def test_energy_decreases_without_food():
    # Place food and danger far away; metabolism + movement should drain energy.
    cfg = SimulationConfig(grid_size=10, max_steps=30)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg)
    layout = {"position": [5, 5], "food": [9, 9], "danger": [0, 0], "light": [0, 9], "cue": [9, 0]}
    obs, _ = env.reset(seed=0, options={"layout": layout})
    org.reset(obs)
    start = org.energy
    for _ in range(20):
        org.step(obs)
        a = org.emit_action()
        obs, _, _, _, info = env.step(a)
        org.update_history(obs, a, info)
    assert org.energy < start


def test_starvation_causes_death():
    cfg = SimulationConfig(grid_size=10, max_steps=500, energy_init=0.05, basal_metabolism=0.02)
    env = BehavioralFieldEnv(cfg)
    org = Organism(cfg)
    layout = {"position": [5, 5], "food": [9, 9], "danger": [0, 0], "light": [0, 9], "cue": [9, 0]}
    obs, _ = env.reset(seed=0, options={"layout": layout})
    org.reset(obs)
    for _ in range(cfg.max_steps):
        org.step(obs)
        a = org.emit_action()
        obs, _, _, _, info = env.step(a)
        org.update_history(obs, a, info)
        if not org.alive:
            break
    assert not org.alive
    assert org.cause_of_death == "starvation"
    assert org.energy <= 0.0


def test_multidim_cpg_atom_oscillates_in_organism():
    """A multi-dimensional oscillator (CPG) atom dropped into the organism's atom set is integrated
    by the real Organism.step pipeline and free-runs (oscillates); scalar atoms stay scalar."""
    import numpy as np

    from behavioral_md.atoms import default_atom_set, oscillator_atom
    from behavioral_md.config import SimulationConfig
    from behavioral_md.organism import Organism

    atoms = default_atom_set() + [oscillator_atom("rhythm", period=30.0, dt=0.1, amplitude=2.0)]
    obs = {f"{s}_vector": np.zeros(2) for s in ("food", "danger", "light", "cue")}
    obs.update({f"{s}_intensity": np.array([0.0]) for s in ("food", "danger", "light", "cue")})
    obs.update({"food_contact": np.array([0.0]), "cue_value": np.array([0.0]),
                "context": np.array([0.0])})
    org = Organism(SimulationConfig(seed=0, learning_rate=0.0), atoms=atoms,
                   rng=np.random.default_rng(0))
    org.reset(obs)
    trace = []
    for _ in range(90):
        org.step(obs)
        trace.append(org.activation("rhythm"))
    trace = np.array(trace)
    assert int((np.diff(np.sign(trace)) != 0).sum()) >= 4      # oscillates (a rhythm)
    assert all(a.dims == 1 for a in atoms if a.name != "rhythm")


def test_organism_reset_preserves_oscillator_phase():
    """reset() restores a CPG atom's constructed phase/velocity (it resumes oscillating), rather
    than collapsing it to rest; a scalar atom still resets to baseline with zero velocity."""
    import numpy as np

    from behavioral_md.atoms import oscillator_atom
    osc = oscillator_atom("cpg", period=30.0, dt=0.1, amplitude=2.0)
    for _ in range(7):
        osc.integrate(np.zeros(2), 0.1, -10, 10)
    osc.reset()
    moved = []
    for _ in range(10):
        osc.integrate(np.zeros(2), 0.1, -10, 10)
        moved.append(osc.activation)
    assert np.ptp(moved) > 0.1                                 # resumes oscillating after reset
