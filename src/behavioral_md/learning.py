"""Learning history: eligibility traces and history-weight updates.

Learning history is stored as a *signed* weight per (atom, stimulus channel).
The sign carries the approach/avoid direction (a positive food weight pulls a
movement atom toward food via the directional projection; a negative danger
weight pushes it away). Weights change only through experienced consequences.

Two orthogonal toggles control the update (both from :class:`SimulationConfig`):

``learning_rule``
    ``"rw"``     error-correcting: ``dV = lr * elig * I * (lambda*c - V_pred)``
                 (negatively accelerated acquisition; non-reinforcement -> decay
                 toward 0, i.e. extinction).
    ``"linear"`` literal spec rule: ``dV = lr * c * elig * I`` (no error term;
                 bounded by clipping; does not extinguish on its own).

``credit_assignment``
    ``"rw_competitive"``  present cues share one prediction ``V_pred = sum_s V_s*I_s``
                          -> cue competition (blocking / overshadowing).
    ``"rw_independent"``  each channel uses its own ``V_pred = V_s`` -> no competition.
    ``"source_only"``     only the channel matching the consequence source updates.

``c`` is the consequence value (e.g. +1 food contact, -1 danger); ``I`` is the
stimulus intensity (how present the cue was); ``elig`` is the atom's eligibility.
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


def update_history(
    atoms: list[BehavioralAtom],
    eligibility: EligibilityTrace,
    intensities: dict[str, float],
    appetitive: float,
    aversive: float,
    config: SimulationConfig,
    *,
    source: str | None = None,
) -> None:
    """Update *drive-atom* history weights from valence-split teaching signals.

    Two-tier model: only drive atoms (nonzero ``valence``) learn. An
    approach-valence atom (``valence > 0``) is taught by ``appetitive``
    (reinforcement); an avoid-valence atom (``valence < 0``) by ``aversive``
    (punishment). Both signals strengthen the disposition (weights grow toward
    +lambda) -- the atom's valence handles approach/avoid direction in the
    expression, so the weight is always a positive "strength of disposition".

    Credit is assigned across present stimulus channels per ``credit_assignment``
    (rw_independent: each present cue independently -> conditioning/generalization;
    rw_competitive: cues share one error -> blocking; source_only: only the atom's
    own stimulus). ``source`` optionally forces a paired cue (e.g. cue->food).
    Updates fire only on consequence events (appetitive or aversive > 0) or when a
    ``source`` is explicitly paired.
    """
    if appetitive <= 0.0 and aversive <= 0.0 and source is None:
        return

    lr = config.learning_rate
    lam = config.reinforcement_asymptote
    lo, hi = config.history_weight_min, config.history_weight_max
    rule = config.learning_rule
    scheme = config.credit_assignment

    for i, atom in enumerate(atoms):
        if atom.valence == 0.0:
            continue  # only drive atoms learn (two-tier)
        mag = appetitive if atom.valence > 0.0 else aversive
        if mag <= 0.0 and source is None:
            continue
        e_i = float(eligibility.trace[i])
        if e_i <= 0.0:
            continue  # only credit positively-engaged drives

        # Channels eligible to update for this atom.
        if scheme == "source_only":
            channels = [atom.stimulus] if atom.stimulus else []
        else:
            channels = [s for s in STIMULI if intensities.get(s, 0.0) > 1e-6]
        if source is not None and source not in channels:
            channels.append(source)
        if not channels:
            continue

        v_pred_shared = (
            sum(atom.history_weights.get(s, 0.0) * intensities.get(s, 0.0) for s in channels)
            if scheme == "rw_competitive"
            else None
        )

        for s in channels:
            intensity = intensities.get(s, 0.0)
            if source is not None and s == source and intensity <= 1e-6:
                intensity = 1.0  # explicitly paired cue treated as present
            if intensity <= 1e-6:
                continue

            v = atom.history_weights.get(s, 0.0)
            if rule == "linear":
                dv = lr * mag * e_i * intensity
            else:  # rw, strengthening toward +lambda*mag
                v_pred = v_pred_shared if v_pred_shared is not None else v
                dv = lr * e_i * intensity * (lam * mag - v_pred)

            atom.history_weights[s] = float(np.clip(v + dv, lo, hi))
