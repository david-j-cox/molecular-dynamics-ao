"""Tests for valence-split Rescorla-Wagner learning and extinction, plus the dual
excitatory/inhibitory (Konorski/Bouton) rule."""


from behavioral_md.atoms import default_atom_set
from behavioral_md.config import SimulationConfig
from behavioral_md.learning import (
    DualExcitatoryInhibitory,
    EligibilityTrace,
    RescorlaWagner,
)

_FOOD = {"food": 1.0, "danger": 0.0, "light": 0.0, "cue": 0.0}


def _setup(**cfg_kw):
    cfg = SimulationConfig(**cfg_kw)
    atoms = default_atom_set()
    elig = EligibilityTrace(len(atoms), cfg.eligibility_decay)
    elig.trace[:] = 1.0  # all atoms eligible
    rule = RescorlaWagner(cfg)
    idx = {a.name: i for i, a in enumerate(atoms)}
    return cfg, atoms, elig, rule, idx


def _setup_dual(**cfg_kw):
    cfg = SimulationConfig(learning_model="dual_exc_inhib", **cfg_kw)
    atoms = default_atom_set()
    elig = EligibilityTrace(len(atoms), cfg.eligibility_decay)
    elig.trace[:] = 1.0
    rule = DualExcitatoryInhibitory(cfg)
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


# --- Dual excitatory/inhibitory (Konorski/Bouton) ----------------------------


def _train(rule, atoms, elig, n, mag, **kw):
    for _ in range(n):
        rule.update(atoms, elig, _FOOD, appetitive=mag, aversive=0.0,
                    appetitive_exposure=True, **kw)


def test_dual_extinction_grows_inhibition_preserves_excitation():
    # The defining property: extinction builds w- and leaves w+ intact (the net falls
    # because of new inhibition, not because excitation was erased).
    cfg, atoms, elig, rule, idx = _setup_dual()
    af = atoms[idx["approach_food"]]
    _train(rule, atoms, elig, 40, 1.0)                      # acquire
    wp_trained = af.w_plus["food"]
    net_trained = af.history_weights["food"]
    assert wp_trained > 0.5 and net_trained > 0.5
    _train(rule, atoms, elig, 40, 0.0)                      # extinguish (omission)
    assert abs(af.w_plus["food"] - wp_trained) < 1e-9       # excitation preserved
    assert af.w_minus["food"] > 0.4                         # inhibition grew
    assert af.history_weights["food"] < net_trained         # net suppressed


def test_dual_spontaneous_recovery_via_passive_decay():
    # After extinction, a rest interval (no exposure) lets w- passively decay while w+
    # stays put, so the net recovers -- spontaneous recovery.
    cfg, atoms, elig, rule, idx = _setup_dual(
        inhibition_rate=0.05, inhibition_passive_decay=0.02
    )
    af = atoms[idx["approach_food"]]
    _train(rule, atoms, elig, 40, 1.0)
    _train(rule, atoms, elig, 40, 0.0)
    wp = af.w_plus["food"]
    net_ext = af.history_weights["food"]
    # Rest: no exposure -> only passive w- decay acts.
    for _ in range(80):
        rule.update(atoms, elig, _FOOD, 0.0, 0.0, appetitive_exposure=False)
    assert abs(af.w_plus["food"] - wp) < 1e-9               # excitation unchanged
    assert af.w_minus["food"] < 0.5                         # inhibition decayed
    assert af.history_weights["food"] > net_ext + 0.05      # net recovered


def test_dual_reacquisition_faster_than_acquisition():
    # w+ survives extinction, so re-reinforcement restores the net in fewer steps than
    # the original acquisition took from scratch.
    cfg, atoms, elig, rule, idx = _setup_dual()
    af = atoms[idx["approach_food"]]

    def steps_to(target):
        n = 0
        while af.history_weights["food"] < target and n < 2000:
            rule.update(atoms, elig, _FOOD, 1.0, 0.0, appetitive_exposure=True)
            n += 1
        return n

    n_acq = steps_to(0.5)
    _train(rule, atoms, elig, 80, 0.0)                      # extinguish below threshold
    assert af.history_weights["food"] < 0.5
    n_reacq = steps_to(0.5)
    assert n_reacq < n_acq


def test_dual_context_gate_enables_renewal():
    # Inhibition learned in one context is released in a different context (gate -> 0),
    # so the net returns to ~w+ -- renewal. Excitation is context-general.
    cfg, atoms, elig, rule, idx = _setup_dual(context_gating=True)
    af = atoms[idx["approach_food"]]
    _train(rule, atoms, elig, 40, 1.0, context=0.0)         # acquire in context 0
    wp = af.w_plus["food"]
    _train(rule, atoms, elig, 60, 0.0, context=1.0)         # extinguish in context 1
    rule._write_net(af, "food", context=1.0)
    net_ext_ctx = af.history_weights["food"]                # extinction context: suppressed
    rule._write_net(af, "food", context=0.0)
    net_other_ctx = af.history_weights["food"]              # different context: renewed
    assert net_other_ctx - net_ext_ctx > 0.3               # clear renewal
    assert net_other_ctx > wp - 0.05                        # gate ~ 0 -> net ~ w+


def test_dual_leaves_rescorla_wagner_path_untouched():
    # The dual rule never touches w+/w- for the RW path; a default atom under RW keeps
    # the dual associations at zero and learns only history_weights.
    cfg, atoms, elig, rule, idx = _setup()  # RescorlaWagner
    af = atoms[idx["approach_food"]]
    _train(rule, atoms, elig, 20, 1.0)
    assert af.history_weights["food"] > 0.0
    assert af.w_plus["food"] == 0.0 and af.w_minus["food"] == 0.0
