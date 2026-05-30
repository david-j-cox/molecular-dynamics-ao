"""Consequence models: how environmental events become energy and learning signals.

A *consequence* is routed through a pluggable model so the organism core never
hard-codes the reinforcement/punishment relationship. Every model maps a
:class:`ConsequenceEvent` (what contingent things happened this step) to:

- ``energy_delta``  -- the change to the organism's physical energy reserve
  (food intake adds energy; danger removes it). This is the objective currency.
- ``learning_signal`` -- the scalar consequence fed to the history-weight update.

The default :class:`DeltaEnergy` treats both as the same quantity (consequence =
change in energy), the maximally objective view: there is no free-floating
"reward", only energy gained and lost in the same units that govern survival.

The reinforcement/punishment *asymmetry* (a punisher subtracts more than a
reinforcer adds; punishment as competitive suppression; injury as a delayed
healing cost plus repertoire impairment) lives in the OTHER models, which are
stubbed here and to be filled in later. See the lab notebook for the literature
(Rasmussen & Newland 2008; de Villiers; Deluty; Klapes & Riley 2018; Klapes &
McDowell 2025).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConsequenceEvent:
    """Contingent events delivered to the organism on a single step."""

    food_intake: float = 0.0      # energy available from consumed food this step
    danger_contact: float = 0.0   # danger intensity contacted this step, in [0, 1]


class ConsequenceModel:
    """Interface mapping a :class:`ConsequenceEvent` to energy + learning signals."""

    def energy_delta(self, event: ConsequenceEvent) -> float:
        raise NotImplementedError

    def learning_signal(self, event: ConsequenceEvent) -> float:
        raise NotImplementedError


@dataclass
class DeltaEnergy(ConsequenceModel):
    """Default: consequence = change in energy. Symmetric currency, no asymmetry.

    ``danger_loss`` is the energy removed per unit danger contact (the punisher
    magnitude). Food intake is delivered directly as energy.
    """

    danger_loss: float = 0.15

    def _delta(self, event: ConsequenceEvent) -> float:
        return event.food_intake - self.danger_loss * event.danger_contact

    def energy_delta(self, event: ConsequenceEvent) -> float:
        return self._delta(event)

    def learning_signal(self, event: ConsequenceEvent) -> float:
        return self._delta(event)


# --- Planned models (asymmetry / competitive suppression / injury) ----------
# These are intentionally not implemented yet; they are the swappable hooks for
# later experiments. Keeping them as named stubs documents the intended design.


class Subtractive(ConsequenceModel):
    """de Villiers (1980) direct/subtractive suppression. TODO."""


class CompetitiveSuppression(ConsequenceModel):
    """Deluty (1976): punishment strengthens competing responses. TODO."""


class ConcatenatedAsymmetric(ConsequenceModel):
    """Concatenated GML with separate reinforcement/punishment sensitivities. TODO."""


class InjuryHealing(ConsequenceModel):
    """Injury = delayed healing energy cost + temporary repertoire impairment. TODO."""


def make_consequence_model(config) -> ConsequenceModel:
    """Construct the consequence model named by ``config.consequence_model``."""
    name = getattr(config, "consequence_model", "delta_energy")
    if name == "delta_energy":
        return DeltaEnergy(danger_loss=config.danger_energy_loss)
    raise NotImplementedError(f"consequence_model '{name}' is not implemented yet")
