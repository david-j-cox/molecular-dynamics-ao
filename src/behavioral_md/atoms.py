"""Behavioral atoms and Verlet integration.

A *behavioral atom* is a small response-capable unit (approach_food, move_up,
consume, pause, ...). Its scalar activation is treated as a response tendency /
response strength. Activations evolve as positions in an abstract activation
space under the literal Verlet update from the spec:

    x_i(t + dt) = 2*x_i(t) - x_i(t - dt) + (F_i(t) / m_i) * dt**2

No mentalistic constructs are used: an atom's "activation" is a response
tendency (an input->output relation), "mass" is behavioral inertia / resistance
to change, and "history_weights" are accumulated learning history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Canonical stimulus channels presented by the environment.
STIMULI: tuple[str, ...] = ("food", "danger", "light", "cue")

# Atoms that emit a discrete action, in env action-id order.
ACTION_ATOMS: dict[int, str] = {
    1: "move_up",
    2: "move_down",
    3: "move_left",
    4: "move_right",
    5: "consume",
    6: "pause",
}


def verlet_update(
    current_state: np.ndarray,
    previous_state: np.ndarray,
    force: np.ndarray,
    mass: float,
    dt: float,
) -> np.ndarray:
    """Literal velocity-free Verlet step (see CLAUDE.md).

    Returns the next state; the caller is responsible for clipping and for
    rolling ``previous_state <- current_state`` afterward.
    """
    return 2.0 * current_state - previous_state + (force / mass) * dt**2


@dataclass
class BehavioralAtom:
    """A single response-capable unit.

    Parameters mirror the spec. ``state``/``previous_state`` are arrays so the
    model can later carry multi-dimensional activation; v1 uses shape ``(1,)``
    and treats ``activation`` as the scalar response tendency.

    ``direction`` is set only for the four movement atoms. It is the atom's
    preferred direction in arena space; a stimulus drives the atom in
    proportion to the dot product of the stimulus's unit vector with this
    direction (the behavior-environment relation). Non-directional atoms leave
    it ``None`` and use the scalar drive form.
    """

    name: str
    state: np.ndarray
    previous_state: np.ndarray
    mass: float = 1.0
    fatigue: float = 0.0
    readiness: float = 1.0
    baseline_activation: float = 0.0
    sensitivity: dict[str, float] = field(default_factory=dict)
    history_weights: dict[str, float] = field(default_factory=dict)
    direction: np.ndarray | None = None
    # Exponent applied to stimulus intensity for this atom. 1.0 = linear distal
    # sensing (approach/locomotion). A large value makes the atom respond only
    # near maximal intensity, i.e. on contact -- used for consummatory responses.
    contact_exponent: float = 1.0
    # Consummatory atoms (e.g. consume) are driven by a sharp binary food-contact
    # signal rather than the broad navigation field, so they fire ONLY at food.
    # This keeps the organism stationary at food (consummatory > locomotion there).
    consummatory: bool = False

    @property
    def activation(self) -> float:
        """Scalar response tendency (v1: the single state component)."""
        return float(self.state[0])

    @property
    def is_directional(self) -> bool:
        return self.direction is not None

    def integrate(
        self, force: float, dt: float, activation_min: float, activation_max: float
    ) -> None:
        """Advance one Verlet step under a scalar force, clip, and roll state."""
        f = np.atleast_1d(np.asarray(force, dtype=np.float64))
        new_state = verlet_update(self.state, self.previous_state, f, self.mass, dt)
        new_state = np.clip(new_state, activation_min, activation_max)
        self.previous_state = self.state
        self.state = new_state

    def reset(self) -> None:
        """Return the atom to its baseline activation with zero velocity."""
        self.state = np.array([self.baseline_activation], dtype=np.float64)
        self.previous_state = self.state.copy()
        self.fatigue = 0.0


def _atom(
    name: str,
    *,
    mass: float = 1.0,
    baseline: float = 0.0,
    readiness: float = 1.0,
    sensitivity: dict[str, float] | None = None,
    history: dict[str, float] | None = None,
    direction: tuple[float, float] | None = None,
    contact_exponent: float = 1.0,
    consummatory: bool = False,
) -> BehavioralAtom:
    """Construct an atom initialized at its baseline activation (zero velocity)."""
    state = np.array([baseline], dtype=np.float64)
    return BehavioralAtom(
        name=name,
        state=state.copy(),
        previous_state=state.copy(),
        mass=mass,
        readiness=readiness,
        baseline_activation=baseline,
        sensitivity=dict.fromkeys(STIMULI, 0.0) | (sensitivity or {}),
        history_weights=dict.fromkeys(STIMULI, 0.0) | (history or {}),
        direction=None if direction is None else np.array(direction, dtype=np.float64),
        contact_exponent=contact_exponent,
        consummatory=consummatory,
    )


def default_atom_set() -> list[BehavioralAtom]:
    """The minimum atom set from the spec, with interpretable starting params.

    Innate (sensory) tendencies are deliberately weak so that learning history
    can take over: food has weak innate approach, danger has innate avoidance
    (negative sensitivity on movement atoms), the cue is neutral (trained by the
    demos).
    """
    atoms = [
        # --- Non-directional "drive" / response-class atoms ----------------
        _atom("approach_food", sensitivity={"food": 0.5}),
        _atom("avoid_danger", sensitivity={"danger": 1.0}),
        _atom("approach_light", sensitivity={"light": 0.3}),
        _atom("orient_to_cue", sensitivity={"cue": 0.0}),  # neutral until trained
        # consume is consummatory: driven by a sharp binary food-contact signal,
        # so it fires only AT food and holds the organism there to feed.
        _atom("consume", sensitivity={"food": 1.0}, consummatory=True),
        # Freezing is evoked by *proximal* threat, so danger is contact-gated
        # (otherwise distant danger drives constant freezing under a broad field).
        _atom("pause", mass=1.5, sensitivity={"danger": 0.5}, contact_exponent=4.0),
        _atom("explore", baseline=0.2, sensitivity={}),
        # --- Directional movement atoms ------------------------------------
        # Weak innate food approach (+) and innate danger avoidance (-) enter
        # via sensitivity; the sign of the *history* weight is what learning
        # tunes (approach vs. avoid) per the directional-projection model.
        _atom("move_up", sensitivity={"food": 1.5, "danger": -0.4}, direction=(0.0, 1.0)),
        _atom("move_down", sensitivity={"food": 1.5, "danger": -0.4}, direction=(0.0, -1.0)),
        _atom("move_left", sensitivity={"food": 1.5, "danger": -0.4}, direction=(-1.0, 0.0)),
        _atom("move_right", sensitivity={"food": 1.5, "danger": -0.4}, direction=(1.0, 0.0)),
    ]
    return atoms
