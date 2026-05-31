"""Learning history: eligibility traces and pluggable decremental-learning rules.

Learning history is stored as a non-negative weight per (atom, stimulus channel)
on the *drive atoms* (those with a nonzero ``valence``). The weight is the
learned strength of that approach/avoid disposition; the atom's valence supplies
the approach/avoid sign in the force expression, so the weight itself only needs
to grow (reinforcement) and shrink (extinction).

The update rule is pluggable (mirrors :mod:`behavioral_md.consequence`) so
different decremental-learning accounts can be swapped in. Implemented:

``RescorlaWagner`` (default)
    Error-correcting delta rule. On a reinforced exposure the weight moves toward
    ``lambda`` at the acquisition rate; on a non-reinforced exposure (the cue is
    present but no reinforcement/punishment) it decays toward 0 at the extinction
    rate. Extinction is acquisition with the asymptote set to 0
    (Rescorla & Wagner, 1972; Bush & Mosteller, 1955). Asymmetric rates
    (``extinction_rate < learning_rate``) make extinction slower than acquisition.
``LinearOperator``
    Strengthen-only ``dw = lr * mag * elig * intensity`` (the original literal
    rule); does not extinguish on its own.

Planned (stubs, see ToDo): dual excitatory/inhibitory (spontaneous recovery,
renewal), momentum-modulated decay (Nevin & Grace), Pearce-Hall associability,
resurgence-as-choice (Shahan & Craig, 2017).

``credit_assignment`` (rw_independent | rw_competitive | source_only) controls
how credit is shared across the present stimulus channels of a drive atom.
"""

from __future__ import annotations

import numpy as np

from behavioral_md.atoms import STIMULI, BehavioralAtom
from behavioral_md.config import SimulationConfig


class EligibilityTrace:
    """Per-atom decaying trace of recent activation (recency-weighted credit)."""

    def __init__(self, n_atoms: int, decay: float) -> None:
        self.decay = decay
        self.trace = np.zeros(n_atoms, dtype=np.float64)

    def update(self, activations: np.ndarray) -> None:
        """e_i <- decay * e_i + activation_i."""
        self.trace = self.decay * self.trace + np.asarray(activations, dtype=np.float64)

    def reset(self) -> None:
        self.trace[:] = 0.0


class LearningRule:
    """Interface: update drive-atom history weights from teaching signals.

    ``appetitive`` (reinforcement) trains approach-valence atoms; ``aversive``
    (punishment) trains avoid-valence atoms. Both are >= 0; a value of 0 marks a
    non-reinforced exposure (the basis for extinction).
    """

    def update(
        self,
        atoms: list[BehavioralAtom],
        eligibility: EligibilityTrace,
        intensities: dict[str, float],
        appetitive: float,
        aversive: float,
        *,
        appetitive_exposure: bool = True,
        aversive_exposure: bool = True,
        source: str | None = None,
    ) -> None:
        """Update history weights.

        ``appetitive_exposure``/``aversive_exposure`` mark whether the organism
        was actually in contact with the reinforcement/punishment source this
        step (e.g. at the food location). Learning (both strengthening and
        extinction) fires only on such exposures, so the weight is not eroded
        merely by traveling toward a distant source.
        """
        raise NotImplementedError


def _channels_for(atom: BehavioralAtom, intensities: dict[str, float], scheme: str,
                  source: str | None) -> list[str]:
    if scheme == "source_only":
        channels = [atom.stimulus] if atom.stimulus else []
    else:
        channels = [s for s in STIMULI if intensities.get(s, 0.0) > 1e-6]
    if source is not None and source not in channels:
        channels.append(source)
    return channels


class RescorlaWagner(LearningRule):
    """Error-correcting rule with omission decay and asymmetric acq/ext rates."""

    def __init__(self, config: SimulationConfig) -> None:
        self.lr_acq = config.learning_rate
        self.lr_ext = config.extinction_rate
        self.lam = config.reinforcement_asymptote
        self.lo = config.history_weight_min
        self.hi = config.history_weight_max
        self.scheme = config.credit_assignment

    def update(self, atoms, eligibility, intensities, appetitive, aversive, *,
               appetitive_exposure=True, aversive_exposure=True, source=None):
        for i, atom in enumerate(atoms):
            if atom.valence == 0.0:
                continue  # only drive atoms learn
            if atom.valence > 0.0:
                exposed, mag = appetitive_exposure, appetitive
            else:
                exposed, mag = aversive_exposure, aversive
            # Learn only on contact exposure to the source (reinforced -> strengthen,
            # non-reinforced -> extinguish); traveling toward it changes nothing.
            if not exposed and source is None:
                continue
            e_i = float(eligibility.trace[i])
            if e_i <= 0.0:
                continue  # credit only positively-engaged drive atoms
            channels = _channels_for(atom, intensities, self.scheme, source)
            if not channels:
                continue
            # Reinforced exposure -> move toward +lambda at lr_acq; non-reinforced
            # exposure (mag == 0) -> decay toward 0 at lr_ext (extinction).
            rate = self.lr_acq if mag > 0.0 else self.lr_ext
            target = self.lam * mag
            v_pred_shared = (
                sum(atom.history_weights.get(s, 0.0) * intensities.get(s, 0.0) for s in channels)
                if self.scheme == "rw_competitive"
                else None
            )
            for s in channels:
                intensity = intensities.get(s, 0.0)
                if source is not None and s == source and intensity <= 1e-6:
                    intensity = 1.0  # explicitly paired cue treated as present
                if intensity <= 1e-6:
                    continue
                v = atom.history_weights.get(s, 0.0)
                v_pred = v_pred_shared if v_pred_shared is not None else v
                dv = rate * e_i * intensity * (target - v_pred)
                atom.history_weights[s] = float(np.clip(v + dv, self.lo, self.hi))


class LinearOperator(LearningRule):
    """Strengthen-only rule (no extinction); the original literal spec rule."""

    def __init__(self, config: SimulationConfig) -> None:
        self.lr = config.learning_rate
        self.lo = config.history_weight_min
        self.hi = config.history_weight_max
        self.scheme = config.credit_assignment

    def update(self, atoms, eligibility, intensities, appetitive, aversive, *,
               appetitive_exposure=True, aversive_exposure=True, source=None):
        for i, atom in enumerate(atoms):
            if atom.valence == 0.0:
                continue
            mag = appetitive if atom.valence > 0.0 else aversive
            if mag <= 0.0 and source is None:
                continue  # no decremental term -> only updates on reinforcement
            e_i = float(eligibility.trace[i])
            if e_i <= 0.0:
                continue
            for s in _channels_for(atom, intensities, self.scheme, source):
                intensity = intensities.get(s, 0.0)
                if source is not None and s == source and intensity <= 1e-6:
                    intensity = 1.0
                if intensity <= 1e-6:
                    continue
                v = atom.history_weights.get(s, 0.0)
                atom.history_weights[s] = float(
                    np.clip(v + self.lr * mag * e_i * intensity, self.lo, self.hi)
                )


def make_learning_rule(config: SimulationConfig) -> LearningRule:
    """Construct the learning rule named by ``config.learning_model``."""
    name = config.learning_model
    if name == "rescorla_wagner":
        return RescorlaWagner(config)
    if name == "linear":
        return LinearOperator(config)
    raise NotImplementedError(f"learning_model '{name}' is not implemented yet")
