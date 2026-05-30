"""Simulation configuration.

A single Pydantic model controls all tunable parameters so that runs are
reproducible and self-documenting. Pass an instance through the environment,
organism, and simulation loop rather than relying on hidden global state.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SimulationConfig(BaseModel):
    """Controls every knob of a simulation run.

    Grouped roughly as: time, episodes, learning, activation/weight bounds,
    motivation, environment geometry, and rendering.
    """

    model_config = {"frozen": False, "extra": "forbid"}

    # --- Time / integration ------------------------------------------------
    dt: float = Field(0.1, gt=0.0, description="Verlet integration timestep.")

    # --- Episodes ----------------------------------------------------------
    n_episodes: int = Field(20, ge=1, description="Number of episodes to run.")
    max_steps: int = Field(200, ge=1, description="Max timesteps per episode.")

    # --- Learning ----------------------------------------------------------
    learning_model: Literal["rescorla_wagner", "linear"] = Field(
        "rescorla_wagner",
        description=(
            "Pluggable decremental-learning rule. 'rescorla_wagner' = "
            "error-correcting with omission decay (lambda=0 on non-reinforced "
            "exposure) and asymmetric acquisition/extinction rates; 'linear' = "
            "strengthen-only w += lr*mag*eligibility*intensity (no extinction)."
        ),
    )
    credit_assignment: Literal["rw_competitive", "rw_independent", "source_only"] = Field(
        "rw_independent",
        description=(
            "How consequence credit is allocated across present stimulus channels. "
            "'rw_competitive' = cues share one error term (blocking/overshadowing); "
            "'rw_independent' = each channel has its own error term (no competition); "
            "'source_only' = only the channel matching the consequence source updates."
        ),
    )
    learning_rate: float = Field(
        0.05, ge=0.0, description="Acquisition rate: history-weight update toward +lambda."
    )
    extinction_rate: float = Field(
        0.02,
        ge=0.0,
        description=(
            "Extinction rate: decay toward 0 on non-reinforced exposure. Usually "
            "< learning_rate (extinction is slower than acquisition; yields PREE)."
        ),
    )
    eligibility_decay: float = Field(
        0.95,
        ge=0.0,
        le=1.0,
        description=(
            "Per-step decay of the eligibility trace (temporal weighting of credit). "
            "Exp 003 sweet spot ~0.9-0.95; 0.99 is too long and inverts learning."
        ),
    )
    reinforcement_asymptote: float = Field(
        1.0,
        gt=0.0,
        description="Lambda: asymptotic associative strength per unit consequence (RW rule).",
    )

    # --- Fatigue (within-bout response decrement) --------------------------
    fatigue_gain: float = Field(
        0.0, ge=0.0, description="Fatigue accrued per step per unit positive activation (0 = off)."
    )
    fatigue_decay: float = Field(
        0.9, ge=0.0, le=1.0, description="Per-step recovery (decay) of fatigue."
    )

    # --- Dynamics: damping -------------------------------------------------
    # Velocity damping turns each atom into a damped (here, overdamped) harmonic
    # oscillator so activation tracks current drive (leaky-accumulator regime)
    # instead of acting as an inertial double-integrator. 0.0 = literal pure
    # Verlet (no dissipation), as in the original spec. Standard in MD (Langevin
    # friction -gamma*v). See lab notebook Exp 001.
    damping_coef: float = Field(
        10.0, ge=0.0, description="Velocity-damping coefficient c in force -= c*velocity."
    )

    # --- Action emission ---------------------------------------------------
    emission: Literal["softmax", "argmax"] = Field(
        "softmax",
        description=(
            "Action-emission rule over action-atom activations. 'softmax' = Luce "
            "choice rule / matching law (temperature ~ matching sensitivity); "
            "'argmax' = winner-take-all (literal spec)."
        ),
    )
    softmax_temperature: float = Field(
        0.3, gt=0.0, description="Temperature for softmax emission (lower = sharper)."
    )
    emission_threshold: float = Field(
        -1e9,
        description="Min action-atom activation to emit a directed action; below it emits no-op.",
    )

    # --- Activation bounds -------------------------------------------------
    activation_min: float = Field(-10.0, description="Lower clip for atom activation.")
    activation_max: float = Field(10.0, description="Upper clip for atom activation.")

    # --- History-weight bounds ---------------------------------------------
    history_weight_min: float = Field(-5.0, description="Lower clip for history weights.")
    history_weight_max: float = Field(5.0, description="Upper clip for history weights.")

    # --- Energy budget (objective; replaces the hunger/MO construct) -------
    # Energy is a conserved physical reserve in [0, energy_capacity]. Behavior
    # shifts with depletion only because energy is real and runs out -- nothing
    # is "valued" or "established". See lab notebook (behavioral-ecology framing).
    energy_capacity: float = Field(1.0, gt=0.0, description="Max energy reserve.")
    energy_init: float = Field(0.5, ge=0.0, description="Starting energy reserve.")
    basal_metabolism: float = Field(
        0.005, ge=0.0, description="Energy spent per step just to keep living."
    )
    move_cost: float = Field(
        0.005, ge=0.0, description="Extra energy per step for a movement action."
    )
    rest_cost: float = Field(
        0.001, ge=0.0, description="Extra energy per step for no-op/pause/consume."
    )
    food_intake_rate: float = Field(
        0.05,
        ge=0.0,
        description="Energy gained per step in contact with food (time-at-patch feeding).",
    )
    danger_energy_loss: float = Field(
        0.15, ge=0.0, description="Energy lost per unit danger contact (punisher magnitude)."
    )
    # Convex marginal value of energy: food-directed drive scales with the
    # deficit raised to this exponent (1 = linear, 2 = quadratic/steep near
    # starvation -> risk-sensitive foraging).
    deficit_exponent: float = Field(
        2.0, ge=1.0, description="Exponent on the energy deficit for the food drive."
    )
    motivational_strength: float = Field(
        2.0, ge=0.0, description="Gain on the energy-deficit-driven food drive."
    )
    consequence_model: Literal["delta_energy"] = Field(
        "delta_energy",
        description="Pluggable consequence model (only delta_energy implemented).",
    )

    # --- Environment geometry ----------------------------------------------
    grid_size: int = Field(10, ge=3, description="Side length of the square grid.")
    sensor_range: float = Field(
        4.0,
        gt=0.0,
        description=(
            "Distance scale for stimulus intensity falloff (exp(-d/range)). "
            "Local enough for a clean gradient on a ~10-cell grid (Exp 003)."
        ),
    )
    consume_radius: float = Field(
        1.0, ge=0.0, description="Max distance to food at which 'consume' succeeds."
    )

    # --- Food as a renewable resource -------------------------------------
    # Food biomass depletes when eaten and regrows logistically toward a
    # carrying capacity. Depleting then waiting for regrowth gives interval-like
    # (VI) availability; food intensity/contact scale with biomass so a spent
    # patch is less visible/edible and the organism leaves until it regrows.
    food_carrying_capacity: float = Field(
        1.0, gt=0.0, description="K: max food biomass at a patch."
    )
    food_regrowth_rate: float = Field(
        0.1, ge=0.0, description="r: logistic regrowth rate per step (dB = r*B*(1-B/K))."
    )
    food_min_biomass: float = Field(
        0.2,
        ge=0.0,
        description="Uneatable remnant; floor from which the patch regrows (logistic).",
    )

    # --- Rendering ---------------------------------------------------------
    render_mode: Literal["human", "rgb_array", "none"] = Field(
        "none", description="Gymnasium render mode."
    )

    # --- Reproducibility ---------------------------------------------------
    seed: int = Field(0, description="Master RNG seed.")

    def validate_bounds(self) -> SimulationConfig:
        """Sanity-check that min bounds are below max bounds."""
        if self.activation_min >= self.activation_max:
            raise ValueError("activation_min must be < activation_max")
        if self.history_weight_min >= self.history_weight_max:
            raise ValueError("history_weight_min must be < history_weight_max")
        return self
