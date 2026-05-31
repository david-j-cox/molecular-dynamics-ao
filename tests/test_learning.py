"""Tests for valence-split Rescorla-Wagner learning and extinction."""


from behavioral_md.atoms import default_atom_set
from behavioral_md.config import SimulationConfig
from behavioral_md.learning import EligibilityTrace, RescorlaWagner


def _setup(**cfg_kw):
    cfg = SimulationConfig(**cfg_kw)
    atoms = default_atom_set()
    elig = EligibilityTrace(len(atoms), cfg.eligibility_decay)
    elig.trace[:] = 1.0  # all atoms eligible
    rule = RescorlaWagner(cfg)
    idx = {a.name: i for i, a in enumerate(atoms)}
    return cfg, atoms, elig, rule, idx


def test_appetitive_strengthens_approach_food():
    cfg, atoms, elig, rule, idx = _setup()
    af = atoms[idx["approach_food"]]
    intensities = {"food": 1.0, "danger": 0.0, "light": 0.0, "cue": 0.0}
    before = af.history_weights["food"]
    for _ in range(20):
        rule.update(atoms, elig, intensities, appetitive=1.0, aversive=0.0,
                    appetitive_exposure=True)
    assert af.history_weights["food"] > before
    assert af.history_weights["food"] <= cfg.reinforcement_asymptote + 1e-6


def test_aversive_strengthens_avoid_danger_not_approach_food():
    cfg, atoms, elig, rule, idx = _setup()
    af, ad = atoms[idx["approach_food"]], atoms[idx["avoid_danger"]]
    intensities = {"food": 0.0, "danger": 1.0, "light": 0.0, "cue": 0.0}
    for _ in range(20):
        rule.update(atoms, elig, intensities, appetitive=0.0, aversive=1.0,
                    aversive_exposure=True)
    assert ad.history_weights["danger"] > 0.5      # avoidance strengthened
    assert abs(af.history_weights["danger"]) < 1e-9  # appetitive drive untouched


def test_extinction_decays_trained_weight_toward_zero():
    cfg, atoms, elig, rule, idx = _setup()
    af = atoms[idx["approach_food"]]
    af.history_weights["food"] = 1.0  # pretend fully trained
    intensities = {"food": 1.0, "danger": 0.0, "light": 0.0, "cue": 0.0}
    for _ in range(100):  # non-reinforced exposures (appetitive=0, but exposed)
        rule.update(atoms, elig, intensities, appetitive=0.0, aversive=0.0,
                    appetitive_exposure=True)
    assert af.history_weights["food"] < 0.2


def test_no_learning_without_contact_exposure():
    cfg, atoms, elig, rule, idx = _setup()
    af = atoms[idx["approach_food"]]
    intensities = {"food": 1.0, "danger": 0.0, "light": 0.0, "cue": 0.0}
    # appetitive signal present but NOT in contact -> weight should not move
    rule.update(atoms, elig, intensities, appetitive=1.0, aversive=0.0,
                appetitive_exposure=False)
    assert af.history_weights["food"] == 0.0
