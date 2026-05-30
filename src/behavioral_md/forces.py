"""Behavioral force calculation.

The net force on atom ``i`` follows the spec decomposition:

    F_i = sensory_drive + history_drive + motivational_drive + coupling_drive - fatigue

All four drives are computed from the *current* sensory field and the atom's
learning history; the coupling drive lets atoms excite/inhibit one another. The
behavior-environment relation enters through a directional projection: for a
movement atom with preferred direction ``d``, a stimulus with unit vector ``u``
and intensity ``I`` contributes proportionally to ``I * dot(u, d)``. For a
non-directional atom the geometric factor is ``1``.

Grounding (no mentalism):
  - sensory_drive    : stimulus control via psychophysical intensity x sensitivity
  - history_drive    : operant strength from learning history (sign = approach/avoid)
  - motivational_drive: energy deficit drives food-seeking (state-dependent foraging)
  - coupling_drive   : competition/induction among members of a response class
  - fatigue          : within-bout response decrement (subtracted)
  - readiness        : momentary gain on the stimulus-evoked drives
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from behavioral_md.atoms import STIMULI, BehavioralAtom
from behavioral_md.config import SimulationConfig


@dataclass
class SensoryField:
    """One stimulus channel as sensed by the organism this step."""

    direction: np.ndarray  # unit vector toward the source, shape (2,)
    intensity: float       # in [0, 1], distance falloff
    contact: float = 0.0   # sharp binary contact signal (food only), in {0, 1}


def sensory_from_observation(obs: dict[str, np.ndarray]) -> dict[str, SensoryField]:
    """Build per-stimulus :class:`SensoryField`s from an env observation dict."""
    fields: dict[str, SensoryField] = {}
    for s in STIMULI:
        fields[s] = SensoryField(
            direction=np.asarray(obs[f"{s}_vector"], dtype=np.float64),
            intensity=float(np.asarray(obs[f"{s}_intensity"]).ravel()[0]),
        )
    # Food carries a sharp contact signal for consummatory atoms.
    fields["food"].contact = float(np.asarray(obs.get("food_contact", 0.0)).ravel()[0])
    return fields


@dataclass
class ForceComponents:
    """Per-atom breakdown of the net force, for logging/inspection."""

    sensory: np.ndarray
    history: np.ndarray
    motivational: np.ndarray
    coupling: np.ndarray
    fatigue: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.sensory + self.history + self.motivational + self.coupling - self.fatigue


class ForceCalculator:
    """Computes behavioral forces over a fixed ordered list of atoms."""

    def __init__(
        self,
        atoms: list[BehavioralAtom],
        coupling_matrix: np.ndarray | None = None,
        config: SimulationConfig | None = None,
    ) -> None:
        self.atoms = atoms
        self.n = len(atoms)
        self.config = config or SimulationConfig()
        if coupling_matrix is None:
            coupling_matrix = default_coupling_matrix(atoms)
        if coupling_matrix.shape != (self.n, self.n):
            raise ValueError(
                f"coupling_matrix must be ({self.n}, {self.n}), got {coupling_matrix.shape}"
            )
        self.coupling_matrix = coupling_matrix

    def _geometry(self, atom: BehavioralAtom, field: SensoryField) -> float:
        """Directional projection factor (1.0 for non-directional atoms)."""
        if atom.direction is None:
            return 1.0
        return float(np.dot(field.direction, atom.direction))

    def compute(
        self, sensory: dict[str, SensoryField], energy: float
    ) -> tuple[np.ndarray, ForceComponents]:
        """Return ``(net_force, components)`` as arrays aligned with ``self.atoms``.

        ``energy`` is the organism's current reserve; its *deficit* drives the
        motivational term via the convex marginal-value-of-energy coupling.
        """
        sensory_drive = np.zeros(self.n)
        history_drive = np.zeros(self.n)
        motivational_drive = np.zeros(self.n)
        fatigue = np.zeros(self.n)

        # Convex marginal value of energy: low reserve -> strong food drive.
        cap = self.config.energy_capacity
        deficit = max(0.0, 1.0 - energy / cap)
        deficit_gain = self.config.motivational_strength * (
            deficit**self.config.deficit_exponent
        )

        for i, atom in enumerate(self.atoms):
            s_drive = 0.0
            h_drive = 0.0
            for s in STIMULI:
                field = sensory[s]
                if s == "food" and atom.consummatory:
                    # Consummatory atoms use the sharp binary contact signal and
                    # are non-directional (geom=1): they fire only AT food.
                    eff_intensity = field.contact
                    geom = 1.0
                else:
                    geom = self._geometry(atom, field)
                    eff_intensity = field.intensity**atom.contact_exponent
                s_drive += atom.sensitivity.get(s, 0.0) * eff_intensity * geom
                h_drive += atom.history_weights.get(s, 0.0) * eff_intensity * geom

            # Energy deficit drives food-directed behavior (objective: energy is
            # worth more nearer the death boundary -> state-dependent foraging).
            food = sensory["food"]
            if atom.consummatory:
                # Hunger amplifies consummatory behavior only when at food.
                m_drive = deficit_gain * food.contact
            else:
                geom_food = self._geometry(atom, food)
                m_drive = deficit_gain * food.intensity**atom.contact_exponent * geom_food

            # readiness is a momentary gain on stimulus-evoked drive.
            sensory_drive[i] = atom.readiness * s_drive
            history_drive[i] = atom.readiness * h_drive
            motivational_drive[i] = atom.readiness * m_drive
            fatigue[i] = atom.fatigue

        # Coupling: row i receives sum_j C[i, j] * activation_j.
        activations = np.array([a.activation for a in self.atoms])
        coupling_drive = self.coupling_matrix @ activations

        components = ForceComponents(
            sensory=sensory_drive,
            history=history_drive,
            motivational=motivational_drive,
            coupling=coupling_drive,
            fatigue=fatigue,
        )
        return components.total, components


def default_coupling_matrix(atoms: list[BehavioralAtom]) -> np.ndarray:
    """A small interpretable coupling matrix C where C[i, j] is j's effect on i.

    Encodes the qualitative relations from the spec:
      - approach_food excites consume
      - avoid_danger inhibits consume
      - pause inhibits all movement
      - explore excites all movement (gated to low stimulus control elsewhere)
    Magnitudes are kept small so coupling modulates rather than dominates.
    """
    idx = {a.name: i for i, a in enumerate(atoms)}
    n = len(atoms)
    c = np.zeros((n, n))

    def link(target: str, source: str, weight: float) -> None:
        if target in idx and source in idx:
            c[idx[target], idx[source]] = weight

    move_atoms = ("move_up", "move_down", "move_left", "move_right")

    link("consume", "approach_food", 0.30)
    link("consume", "avoid_danger", -0.40)
    for m in move_atoms:
        link(m, "pause", -0.30)
        link(m, "explore", 0.20)
    # Freezing to danger suppresses consumption.
    link("consume", "pause", -0.20)

    return c
