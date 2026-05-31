"""Tests for the consequence model (delta-energy currency + teaching signals)."""

import numpy as np

from behavioral_md.consequence import ConsequenceEvent, DeltaEnergy


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
