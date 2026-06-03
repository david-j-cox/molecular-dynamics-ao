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
    # Dual excitatory/inhibitory associations (Konorski/Bouton), used only by the
    # DualExcitatoryInhibitory learning rule. ``history_weights`` always holds the NET
    # the force reads; the dual rule maintains w_plus (excitation, preserved through
    # extinction) and w_minus (inhibition, grows on omission, context-specific) and
    # writes history_weights = w_plus - gate*w_minus each step. ``w_minus_ctx`` tags the
    # context value in which the inhibition was learned (for context-gated renewal).
    # Empty by default so other rules and existing atoms are unaffected.
    w_plus: dict[str, float] = field(default_factory=dict)
    w_minus: dict[str, float] = field(default_factory=dict)
    w_minus_ctx: dict[str, float] = field(default_factory=dict)
    direction: np.ndarray | None = None
    # Exponent applied to stimulus intensity for this atom. 1.0 = linear distal
    # sensing (approach/locomotion). A large value makes the atom respond only
    # near maximal intensity, i.e. on contact -- used for consummatory responses.
    contact_exponent: float = 1.0
    # Consummatory atoms (e.g. consume) are driven by a sharp binary food-contact
    # signal rather than the broad navigation field, so they fire ONLY at food.
    # This keeps the organism stationary at food (consummatory > locomotion there).
    consummatory: bool = False
    # Two-tier learning. A *drive* atom is a non-directional, sign-stable
    # response class tagged with the stimulus it tracks and a valence
    # (+1 approach, -1 avoid). Learning lives on drive atoms; their activation is
    # *expressed* through the (topographic) movement atoms via the live stimulus
    # geometry. Movement atoms carry no stimulus/valence -- they only express.
    stimulus: str | None = None
    valence: float = 0.0

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
    stimulus: str | None = None,
    valence: float = 0.0,
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
        w_plus=dict.fromkeys(STIMULI, 0.0),
        w_minus=dict.fromkeys(STIMULI, 0.0),
        w_minus_ctx=dict.fromkeys(STIMULI, 0.0),
        direction=None if direction is None else np.array(direction, dtype=np.float64),
        contact_exponent=contact_exponent,
        consummatory=consummatory,
        stimulus=stimulus,
        valence=valence,
    )


def default_atom_set() -> list[BehavioralAtom]:
    """Two-tier atom set.

    *Drive atoms* (non-directional, sign-stable) carry the learning history: each
    tracks one ``stimulus`` with an approach/avoid ``valence``. Innate
    sensitivities are weak so learning history can take over; the cue starts
    neutral (trained by the demos). *Movement atoms* are topographies that carry
    no sensitivity/history -- they only express the drive atoms through the live
    stimulus geometry (handled in the force calculator). ``consume`` is a
    contact-gated consummatory response; ``pause``/``explore`` modulate movement
    via coupling.
    """
    atoms = [
        # --- Drive atoms (learn; expressed through movement) ----------------
        _atom("approach_food", stimulus="food", valence=1.0, sensitivity={"food": 0.5}),
        _atom("avoid_danger", stimulus="danger", valence=-1.0, sensitivity={"danger": 1.0}),
        _atom("approach_light", stimulus="light", valence=1.0, sensitivity={"light": 0.3}),
        _atom("orient_to_cue", stimulus="cue", valence=1.0, sensitivity={"cue": 0.0}),
        # --- Consummatory + modulatory atoms --------------------------------
        # consume fires only AT food (sharp binary contact) and holds position.
        _atom("consume", sensitivity={"food": 1.0}, consummatory=True),
        # Freezing is evoked by *proximal* threat (contact-gated danger).
        _atom("pause", mass=1.5, sensitivity={"danger": 0.5}, contact_exponent=4.0),
        _atom("explore", baseline=0.2),
        # --- Movement atoms (topographies; express the drive atoms) ---------
        _atom("move_up", direction=(0.0, 1.0)),
        _atom("move_down", direction=(0.0, -1.0)),
        _atom("move_left", direction=(-1.0, 0.0)),
        _atom("move_right", direction=(1.0, 0.0)),
    ]
    return atoms
