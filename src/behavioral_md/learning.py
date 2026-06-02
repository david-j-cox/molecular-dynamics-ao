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
        context: float = 0.0,
    ) -> None:
        """Update history weights.

        ``appetitive_exposure``/``aversive_exposure`` mark whether the organism
        was actually in contact with the reinforcement/punishment source this
        step (e.g. at the food location). Learning (both strengthening and
        extinction) fires only on such exposures, so the weight is not eroded
        merely by traveling toward a distant source.

        ``context`` is a scalar context signal (default 0.0) used only by the
        dual excitatory/inhibitory rule to gate inhibition for renewal effects;
        other rules ignore it.
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
               appetitive_exposure=True, aversive_exposure=True, source=None,
               context=0.0):
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
               appetitive_exposure=True, aversive_exposure=True, source=None,
               context=0.0):
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


class DualExcitatoryInhibitory(LearningRule):
    """Konorski/Bouton extinction: separate excitatory (w+) and inhibitory (w-) links.

    Extinction does NOT erase the original excitation; it builds a new, separate
    inhibitory association. The net association the force reads is ``w+ - gate*w-``,
    written into ``atom.history_weights`` each step (so the force model is unchanged):

    - Reinforced exposure: w+ moves toward ``lambda`` (acquisition); w- relaxes toward 0
      (reinforcement removes inhibition -> rapid reacquisition, since w+ was preserved).
    - Non-reinforced exposure (omission): w+ is UNCHANGED; w- grows toward ``lambda``
      (inhibition accrues). The preserved w+ is what makes the three post-extinction
      phenomena possible.
    - Passive decay: w- decays multiplicatively every step (even without exposure), so
      after a rest interval the net recovers -> spontaneous recovery.
    - Context gating: w- is tagged with the context it was learned in and, at readout,
      scaled by a Shepard similarity ``exp(-beta*|context_now - context_learned|)``;
      inhibition is context-specific while excitation is context-general -> renewal.

    w- is floored at 0 (inhibition is non-negative); the net is clipped to
    ``[history_weight_min, history_weight_max]`` like any history weight.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.lr_acq = config.learning_rate
        self.lr_inhib = config.inhibition_rate
        self.lr_relax = config.inhibition_relax_rate
        self.passive_decay = config.inhibition_passive_decay
        self.gating = config.context_gating
        self.beta = config.context_beta
        self.ctx_tag_rate = config.ctx_tag_rate
        self.lam = config.reinforcement_asymptote
        self.lo = config.history_weight_min
        self.hi = config.history_weight_max
        self.scheme = config.credit_assignment

    def _write_net(self, atom: BehavioralAtom, s: str, context: float) -> None:
        """Recompute the context-gated net w+ - gate*w- into history_weights."""
        wp = atom.w_plus.get(s, 0.0)
        wm = atom.w_minus.get(s, 0.0)
        if self.gating and wm > 0.0:
            gate = float(np.exp(-self.beta * abs(context - atom.w_minus_ctx.get(s, 0.0))))
        else:
            gate = 1.0
        atom.history_weights[s] = float(np.clip(wp - gate * wm, self.lo, self.hi))

    def update(self, atoms, eligibility, intensities, appetitive, aversive, *,
               appetitive_exposure=True, aversive_exposure=True, source=None,
               context=0.0):
        for i, atom in enumerate(atoms):
            if atom.valence == 0.0:
                continue  # only drive atoms learn

            # Every step (even off-contact): optionally apply passive w- decay (so the
            # net recovers over a rest interval -> spontaneous recovery), and always
            # refresh the context-gated net so the force tracks the CURRENT context
            # immediately when it changes (-> renewal is expressed on approach, before
            # any new contact). With context_gating off this just rewrites w+ - w-.
            for s in STIMULI:
                wm = atom.w_minus.get(s, 0.0)
                if self.passive_decay > 0.0 and wm != 0.0:
                    atom.w_minus[s] = wm * (1.0 - self.passive_decay)
                self._write_net(atom, s, context)

            if atom.valence > 0.0:
                exposed, mag = appetitive_exposure, appetitive
            else:
                exposed, mag = aversive_exposure, aversive
            if not exposed and source is None:
                continue
            e_i = float(eligibility.trace[i])
            if e_i <= 0.0:
                continue
            channels = _channels_for(atom, intensities, self.scheme, source)
            for s in channels:
                intensity = intensities.get(s, 0.0)
                if source is not None and s == source and intensity <= 1e-6:
                    intensity = 1.0
                if intensity <= 1e-6:
                    continue
                # Clip each delta step's result to the weight bounds (as RescorlaWagner
                # does) so a large eligibility*intensity step cannot overshoot and diverge.
                step = e_i * intensity
                if mag > 0.0:
                    # Reinforced: strengthen excitation, relax inhibition toward 0.
                    wp = atom.w_plus.get(s, 0.0)
                    atom.w_plus[s] = float(np.clip(
                        wp + self.lr_acq * step * (self.lam * mag - wp), self.lo, self.hi))
                    wm = atom.w_minus.get(s, 0.0)
                    atom.w_minus[s] = float(np.clip(
                        wm + self.lr_relax * step * (0.0 - wm), 0.0, self.hi))
                else:
                    # Omission: excitation preserved; inhibition grows toward lambda,
                    # tagged with the current context for context-gated renewal.
                    wm = atom.w_minus.get(s, 0.0)
                    atom.w_minus[s] = float(np.clip(
                        wm + self.lr_inhib * step * (self.lam - wm), 0.0, self.hi))
                    ctx = atom.w_minus_ctx.get(s, 0.0)
                    atom.w_minus_ctx[s] = ctx + self.ctx_tag_rate * step * (context - ctx)
                self._write_net(atom, s, context)


def make_learning_rule(config: SimulationConfig) -> LearningRule:
    """Construct the learning rule named by ``config.learning_model``."""
    name = config.learning_model
    if name == "rescorla_wagner":
        return RescorlaWagner(config)
    if name == "linear":
        return LinearOperator(config)
    if name == "dual_exc_inhib":
        return DualExcitatoryInhibitory(config)
    raise NotImplementedError(f"learning_model '{name}' is not implemented yet")
