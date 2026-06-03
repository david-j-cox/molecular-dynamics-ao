"""Tests for the consequence model (delta-energy currency + teaching signals)."""

import numpy as np

from behavioral_md.config import SimulationConfig
from behavioral_md.consequence import (
    ConcatenatedAsymmetric,
    ConsequenceEvent,
    DeltaEnergy,
    Subtractive,
    make_consequence_model,
)


def test_food_intake_adds_energy():
    m = DeltaEnergy(danger_loss=0.15)
    assert np.isclose(m.energy_delta(ConsequenceEvent(food_intake=0.2)), 0.2)


def test_danger_contact_removes_energy():
    m = DeltaEnergy(danger_loss=0.15)
    assert np.isclose(m.energy_delta(ConsequenceEvent(danger_contact=1.0)), -0.15)


def test_learning_signals_split_by_valence():
    m = DeltaEnergy()
    app, avr = m.learning_signals(ConsequenceEvent(food_intake=0.2))
    assert app > 0 and avr == 0
    app, avr = m.learning_signals(ConsequenceEvent(danger_contact=1.0))
    assert app == 0 and avr > 0


def test_no_event_no_signal():
    m = DeltaEnergy()
    assert np.isclose(m.energy_delta(ConsequenceEvent()), 0.0)
    assert m.learning_signals(ConsequenceEvent()) == (0.0, 0.0)


def test_subtractive_scales_aversive_signal():
    """de Villiers: a punisher trains avoidance c times more strongly than a reinforcer
    trains approach; reinforcement is unchanged."""
    m = Subtractive(danger_loss=0.15, c=3.0)
    app, avr = m.learning_signals(ConsequenceEvent(food_intake=0.2, danger_contact=1.0))
    assert app == 1.0
    assert avr == 3.0
    # Energy currency is unchanged (the asymmetry is in the teaching signal).
    assert np.isclose(m.energy_delta(ConsequenceEvent(danger_contact=1.0)), -0.15)


def test_concatenated_separate_sensitivities():
    m = ConcatenatedAsymmetric(reinf_sensitivity=0.5, punish_sensitivity=2.0)
    app, avr = m.learning_signals(ConsequenceEvent(food_intake=0.2, danger_contact=1.0))
    assert app == 0.5
    assert avr == 2.0


def test_make_consequence_model_dispatch():
    cfg = SimulationConfig(consequence_model="subtractive", punishment_weight=2.5)
    assert isinstance(make_consequence_model(cfg), Subtractive)
    cfg2 = SimulationConfig(consequence_model="concatenated_asymmetric",
                            punish_sensitivity=2.0)
    m = make_consequence_model(cfg2)
    assert isinstance(m, ConcatenatedAsymmetric)
    assert m.punish_sensitivity == 2.0
    assert isinstance(make_consequence_model(SimulationConfig()), DeltaEnergy)
