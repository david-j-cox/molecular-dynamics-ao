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
        """Change to the physical energy reserve (intake minus injury loss)."""
        raise NotImplementedError

    def learning_signals(self, event: ConsequenceEvent) -> tuple[float, float]:
        """Return ``(appetitive, aversive)`` teaching signals, each >= 0.

        ``appetitive`` trains approach-valence drives (reinforcement);
        ``aversive`` trains avoid-valence drives (punishment). Splitting them
        keeps reinforcement and punishment from contaminating each other and is
        the natural place to encode their asymmetry.
        """
        raise NotImplementedError


@dataclass
class DeltaEnergy(ConsequenceModel):
    """Default: energy reserve changes by intake - injury; teaching signals are
    the occurrence of reinforcement (feeding) and punishment (danger contact).

    The reserve uses the raw (objective) energy units; the learning signals are
    normalized to ~1 per event so associative strength is well-scaled (the tiny
    per-step energy amounts would otherwise produce negligible learning). The
    reinforcement/punishment magnitude/asymmetry will be elaborated in the other
    consequence models.
    """

    danger_loss: float = 0.15

    def energy_delta(self, event: ConsequenceEvent) -> float:
        return event.food_intake - self.danger_loss * event.danger_contact

    def learning_signals(self, event: ConsequenceEvent) -> tuple[float, float]:
        appetitive = 1.0 if event.food_intake > 0.0 else 0.0
        aversive = 1.0 if event.danger_contact > 0.0 else 0.0
        return appetitive, aversive


@dataclass
class Subtractive(ConsequenceModel):
    """de Villiers (1980) subtractive suppression: a punisher cancels ``c`` reinforcers.

    In the two-tier foraging organism, approach (valence +1) and avoidance (valence -1)
    are separate drive atoms that compete at the movement layer, so "subtraction" is
    expressed by training avoidance ``c`` times more strongly than approach: the aversive
    teaching signal is scaled by ``c`` (= ``punishment_weight``). The energy reserve also
    takes an asymmetric hit (a punisher costs more than a reinforcer yields).
    """

    danger_loss: float = 0.15
    c: float = 2.0   # reinforcers cancelled per punisher (punishment_weight)

    def energy_delta(self, event: ConsequenceEvent) -> float:
        return event.food_intake - self.danger_loss * event.danger_contact

    def learning_signals(self, event: ConsequenceEvent) -> tuple[float, float]:
        appetitive = 1.0 if event.food_intake > 0.0 else 0.0
        aversive = self.c if event.danger_contact > 0.0 else 0.0
        return appetitive, aversive


@dataclass
class ConcatenatedAsymmetric(ConsequenceModel):
    """Separate reinforcement vs punishment sensitivities (Critchfield/Klapes).

    The appetitive and aversive teaching signals carry independent gains
    (``reinf_sensitivity`` and ``punish_sensitivity``), so the strength of approach
    training and avoidance training -- and thus the behavioral reinforcement/punishment
    asymmetry -- are tuned separately rather than tied to one ``c``.
    """

    danger_loss: float = 0.15
    reinf_sensitivity: float = 1.0
    punish_sensitivity: float = 1.0

    def energy_delta(self, event: ConsequenceEvent) -> float:
        return event.food_intake - self.danger_loss * event.danger_contact

    def learning_signals(self, event: ConsequenceEvent) -> tuple[float, float]:
        appetitive = self.reinf_sensitivity if event.food_intake > 0.0 else 0.0
        aversive = self.punish_sensitivity if event.danger_contact > 0.0 else 0.0
        return appetitive, aversive


# --- Planned models (not yet implemented) -----------------------------------
# CompetitiveSuppression (Deluty) and concatenated matching are CHOICE/ALLOCATION
# accounts (about the relation BETWEEN responses), not event->energy maps, so they
# live in the concurrent chamber (chamber.run_punishment_choice), not here. InjuryHealing
# is a genuine ConsequenceModel (a delayed, embodied cost) left for a follow-up.


class InjuryHealing(ConsequenceModel):
    """Injury = delayed healing energy cost + temporary repertoire impairment. TODO."""


def make_consequence_model(config) -> ConsequenceModel:
    """Construct the consequence model named by ``config.consequence_model``."""
    name = getattr(config, "consequence_model", "delta_energy")
    if name == "delta_energy":
        return DeltaEnergy(danger_loss=config.danger_energy_loss)
    if name == "subtractive":
        return Subtractive(danger_loss=config.danger_energy_loss,
                           c=config.punishment_weight)
    if name == "concatenated_asymmetric":
        return ConcatenatedAsymmetric(danger_loss=config.danger_energy_loss,
                                      reinf_sensitivity=config.reinf_sensitivity,
                                      punish_sensitivity=config.punish_sensitivity)
    raise NotImplementedError(f"consequence_model '{name}' is not implemented yet")
