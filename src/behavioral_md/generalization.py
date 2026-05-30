"""Stimulus generalization over a scalar cue dimension (population of receptors).

The neutral cue varies along an abstract scalar dimension ``v in [0, 1]`` (e.g. a
tone frequency or line tilt -- not a spatial location). It is represented by a
population of ``K`` receptors tiled across the dimension, each tuned to a center
``c_k`` with a Shepard exponential kernel ``exp(-beta * |v - c_k|)``. Each
receptor carries its own learned association weight to reinforcement.

Training a cue near value ``v0`` strengthens the receptors near ``v0`` (those the
cue activates); a later test cue at value ``v`` evokes a response
``sum_k w_k * exp(-beta * |v - c_k|)`` that is peaked at ``v0`` and falls off with
distance. The generalization gradient therefore *emerges* from the overlap of the
receptor tuning curves rather than being inserted at readout (classic elemental
account; Blough, 1975).
"""

from __future__ import annotations

import numpy as np


class CueReceptorField:
    """Population of value-tuned cue receptors with learned association weights."""

    def __init__(self, n_receptors: int, beta: float, lo: float, hi: float) -> None:
        self.centers = np.linspace(0.0, 1.0, n_receptors)
        self.beta = beta
        self.lo = lo
        self.hi = hi
        self.weights = np.zeros(n_receptors, dtype=np.float64)
        self.last_activation = np.zeros(n_receptors, dtype=np.float64)

    def _tuning(self, value: float) -> np.ndarray:
        """Shepard similarity of each receptor to the cue value, in [0, 1]."""
        return np.exp(-self.beta * np.abs(value - self.centers))

    def drive(self, value: float, intensity: float) -> float:
        """Conditioned response to the cue; caches the receptor activations.

        ``intensity`` is the cue's spatial presence (1.0 = fully present). Call
        this once per step; :meth:`learn` then uses the cached activations.
        """
        self.last_activation = intensity * self._tuning(value)
        return float(self.weights @ self.last_activation)

    def response(self, value: float, intensity: float = 1.0) -> float:
        """Probe the conditioned response at a cue value (no side effects)."""
        return float(self.weights @ (intensity * self._tuning(value)))

    def learn(self, eligibility: float, mag: float, rate: float, lam: float) -> None:
        """Rescorla-Wagner update on the receptors active this step.

        ``mag`` is the teaching signal (1 reinforced, 0 omitted); ``rate`` is the
        acquisition or extinction rate; ``lam`` the asymptote. Receptors that were
        active (near the presented value) move toward ``lam*mag``.
        """
        target = lam * mag
        self.weights = np.clip(
            self.weights + rate * eligibility * self.last_activation * (target - self.weights),
            self.lo,
            self.hi,
        )

    def reset_state(self) -> None:
        """Clear per-step activations; learned weights persist across lives."""
        self.last_activation[:] = 0.0
