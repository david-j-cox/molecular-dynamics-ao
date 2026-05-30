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
    learning_rate: float = Field(0.05, ge=0.0, description="History-weight update rate.")
    eligibility_decay: float = Field(
        0.9, ge=0.0, le=1.0, description="Per-step decay of the eligibility trace."
    )

    # --- Activation bounds -------------------------------------------------
    activation_min: float = Field(-10.0, description="Lower clip for atom activation.")
    activation_max: float = Field(10.0, description="Upper clip for atom activation.")

    # --- History-weight bounds ---------------------------------------------
    history_weight_min: float = Field(-5.0, description="Lower clip for history weights.")
    history_weight_max: float = Field(5.0, description="Upper clip for history weights.")

    # --- Motivation --------------------------------------------------------
    hunger_init: float = Field(0.5, ge=0.0, description="Initial hunger level.")
    hunger_growth: float = Field(0.01, ge=0.0, description="Hunger gained per step.")
    hunger_drop_on_food: float = Field(
        0.5, ge=0.0, description="Hunger removed on food consumption."
    )
    hunger_max: float = Field(1.0, gt=0.0, description="Hunger saturation ceiling.")

    # --- Environment geometry ----------------------------------------------
    grid_size: int = Field(10, ge=3, description="Side length of the square grid.")
    sensor_range: float = Field(
        10.0, gt=0.0, description="Distance scale for stimulus intensity falloff."
    )
    consume_radius: float = Field(
        0.0, ge=0.0, description="Max distance to food at which 'consume' succeeds."
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
