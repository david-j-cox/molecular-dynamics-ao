"""The Organism: a distributed dynamical system of coupled behavioral atoms.

The organism is *not* a policy network. Each step it senses the stimulus field,
computes a behavioral force on every atom, advances atom activations by one
Verlet step, and emits the action whose atom is currently most active. After the
consequence is delivered it updates its learning history. Behavior emerges from
the coupled atom dynamics, not from reward maximization.
"""

from __future__ import annotations

import numpy as np

from behavioral_md.atoms import ACTION_ATOMS, STIMULI, BehavioralAtom, default_atom_set
from behavioral_md.config import SimulationConfig
from behavioral_md.consequence import ConsequenceEvent, make_consequence_model
from behavioral_md.forces import (
    ForceCalculator,
    ForceComponents,
    sensory_from_observation,
)
from behavioral_md.learning import EligibilityTrace, update_history

# Movement actions (cost more energy than resting/consuming).
_MOVE_ACTIONS = frozenset({1, 2, 3, 4})


class Organism:
    """A single organism made of behavioral atoms."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        atoms: list[BehavioralAtom] | None = None,
        coupling_matrix: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.config = config or SimulationConfig()
        self.atoms = atoms if atoms is not None else default_atom_set()
        self.names = [a.name for a in self.atoms]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.force_calc = ForceCalculator(self.atoms, coupling_matrix, self.config)
        self.eligibility = EligibilityTrace(len(self.atoms), self.config.eligibility_decay)
        self.rng = rng or np.random.default_rng(self.config.seed)
        self.consequence_model = make_consequence_model(self.config)

        # Objective physical state.
        self.energy = self.config.energy_init
        self.alive = True
        self.cause_of_death: str | None = None
        # Most-recent step bookkeeping, exposed for logging.
        self.last_force = np.zeros(len(self.atoms))
        self.last_components: ForceComponents | None = None
        self.last_intensities: dict[str, float] = dict.fromkeys(STIMULI, 0.0)
        self.last_action = 0

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def reset(self, observation: dict[str, np.ndarray]) -> None:
        """Reset atom states, eligibility, and energy for a new life."""
        for atom in self.atoms:
            atom.reset()
        self.eligibility.reset()
        self.energy = self.config.energy_init
        self.alive = True
        self.cause_of_death = None
        self.last_force = np.zeros(len(self.atoms))
        self.last_components = None
        self.last_intensities = self._intensities(observation)
        self.last_action = 0

    def step(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Advance all atoms one Verlet step under the current stimulus field."""
        sensory = sensory_from_observation(observation)
        self.last_intensities = {s: sensory[s].intensity for s in STIMULI}

        force, components = self.force_calc.compute(sensory, self.energy)
        self.last_force = force
        self.last_components = components

        cfg = self.config
        for i, atom in enumerate(self.atoms):
            # Velocity damping (-c * v); v is the Verlet velocity (x - x_prev)/dt.
            # c=0 recovers literal pure Verlet.
            velocity = (atom.state[0] - atom.previous_state[0]) / cfg.dt
            net_force = force[i] - cfg.damping_coef * velocity
            atom.integrate(net_force, cfg.dt, cfg.activation_min, cfg.activation_max)

        self._update_fatigue()
        # Eligibility reflects the activations that just produced behavior.
        self.eligibility.update(np.array([a.activation for a in self.atoms]))
        return np.array([a.activation for a in self.atoms])

    def emit_action(self) -> int:
        """Emit an action from the action-atom activations.

        ``emission='softmax'`` samples via the Luce choice rule / matching law
        (P(a) proportional to exp(activation/temperature)); ``'argmax'`` is
        winner-take-all with random tie-breaking (literal spec).
        """
        ids = list(ACTION_ATOMS.keys())
        acts = np.array([self.atoms[self.index[ACTION_ATOMS[a]]].activation for a in ids])

        if self.config.emission == "softmax":
            z = acts / self.config.softmax_temperature
            z -= z.max()
            p = np.exp(z)
            p /= p.sum()
            action = int(self.rng.choice(ids, p=p))
            self.last_action = action
            return action

        # argmax
        best = float(acts.max())
        if best < self.config.emission_threshold:
            self.last_action = 0
            return 0
        winners = [ids[k] for k in np.flatnonzero(acts >= best - 1e-9)]
        action = int(self.rng.choice(winners)) if len(winners) > 1 else int(winners[0])
        self.last_action = action
        return action

    def update_history(
        self,
        observation: dict[str, np.ndarray],
        action: int,
        info: dict,
    ) -> None:
        """Run energy bookkeeping and the learning-history update for one step.

        Energy flows: ``E <- E - basal - move/rest cost + contingent energy
        (food intake - danger loss)``. Depletion to zero is death. The learning
        consequence is the consequence model's signal (delta-E by default).
        """
        cfg = self.config

        # 1. Contingent events -> energy + learning signal (via the model).
        event = ConsequenceEvent(
            food_intake=cfg.food_intake_rate if info.get("at_food", False) else 0.0,
            danger_contact=float(info.get("danger_contact", 0.0)),
        )
        intake = self.consequence_model.energy_delta(event)
        appetitive, aversive = self.consequence_model.learning_signals(event)

        # 2. Objective energy bookkeeping (intake minus metabolic expenditure).
        expenditure = cfg.basal_metabolism + (
            cfg.move_cost if action in _MOVE_ACTIONS else cfg.rest_cost
        )
        self.energy = float(np.clip(self.energy + intake - expenditure, 0.0, cfg.energy_capacity))
        if self.energy <= 0.0 and self.alive:
            self.alive = False
            self.cause_of_death = "danger" if event.danger_contact > 0.0 else "starvation"

        # 3. Learning-history update on the drive atoms (valence-split credit).
        intensities = self._intensities(observation)
        source = info.get("credit_source")  # demos may pair a neutral cue
        update_history(
            self.atoms,
            self.eligibility,
            intensities,
            appetitive,
            aversive,
            cfg,
            source=source,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _intensities(self, observation: dict[str, np.ndarray]) -> dict[str, float]:
        return {
            s: float(np.asarray(observation[f"{s}_intensity"]).ravel()[0]) for s in STIMULI
        }

    def _update_fatigue(self) -> None:
        gain, decay = self.config.fatigue_gain, self.config.fatigue_decay
        if gain == 0.0:
            return
        for atom in self.atoms:
            atom.fatigue = decay * atom.fatigue + gain * max(0.0, atom.activation)

    def activation(self, name: str) -> float:
        return self.atoms[self.index[name]].activation
