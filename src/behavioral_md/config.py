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
    learning_model: Literal["rescorla_wagner", "linear", "dual_exc_inhib"] = Field(
        "rescorla_wagner",
        description=(
            "Pluggable decremental-learning rule. 'rescorla_wagner' = "
            "error-correcting with omission decay (lambda=0 on non-reinforced "
            "exposure) and asymmetric acquisition/extinction rates; 'linear' = "
            "strengthen-only w += lr*mag*eligibility*intensity (no extinction); "
            "'dual_exc_inhib' = Konorski/Bouton separate excitatory (w+) and "
            "inhibitory (w-) associations (net = w+ - gate*w-): extinction grows w- "
            "rather than erasing w+, yielding spontaneous recovery, renewal, and "
            "rapid reacquisition (uses the inhibition_*/context_* fields below)."
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
    # --- Dual excitatory/inhibitory rule (learning_model='dual_exc_inhib') ----
    # All read only by DualExcitatoryInhibitory; defaults leave the other rules
    # untouched. w- = inhibition (grows on omission, context-specific, decays slowly).
    inhibition_rate: float = Field(
        0.02, ge=0.0,
        description="Growth rate of inhibition (w-) toward lambda on non-reinforced exposure.",
    )
    inhibition_relax_rate: float = Field(
        0.1, ge=0.0,
        description="Rate at which inhibition (w-) relaxes toward 0 on reinforced exposure. "
        "Larger than the acquisition rate because inhibition is labile (Bouton): "
        "reinforcement rapidly cancels it, so reacquisition is faster than original "
        "acquisition (w+ was preserved through extinction).",
    )
    inhibition_passive_decay: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Per-step multiplicative decay of inhibition (w- *= 1-decay) applied even "
        "off-contact, so the net recovers over a rest interval (spontaneous recovery). "
        "Default 0 (opt-in); keep small so foraging travel does not erode inhibition.",
    )
    context_gating: bool = Field(
        False,
        description="If true, inhibition (w-) is gated at readout by similarity between the "
        "current context and the context in which it was learned (renewal). Excitation (w+) "
        "is context-general.",
    )
    context_beta: float = Field(
        6.0, ge=0.0,
        description="Shepard width for the context gate "
        "exp(-beta*|context_now - context_learned|).",
    )
    ctx_tag_rate: float = Field(
        0.1, ge=0.0, le=1.0,
        description="EMA rate at which a channel's inhibition-context tag tracks the context "
        "while w- is growing.",
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
    # Integrator form. "verlet" is the damped second-order (default; baseline). "leaky" is a
    # first-order leaky integrator x <- x + dt*(force/m - leak*x), i.e. the leaky competing
    # accumulator the overdamped Verlet limit reduces to. Used only for the ablation that asks
    # whether the second-order/inertial term earns its keep (Exp 048). Default keeps Verlet so the
    # reproduce baseline is byte-identical.
    integrator: Literal["verlet", "leaky"] = Field(
        "verlet", description="Activation integrator: damped Verlet (default) or first-order leaky."
    )
    leak_coef: float = Field(
        1.0, ge=0.0, description="Leak rate for the first-order leaky integrator."
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
    # Effort -> energy coupling (closes the force<->energy loop): on top of the fixed per-action
    # cost above, vigorous responding costs energy in proportion to behavioral effort = the summed
    # POSITIVE activation of the action atoms (motor output intensity). 0.0 = off (byte-identical;
    # cost is the fixed per-action constant only); >0 makes effort metabolically constrained, so a
    # strongly driven response (e.g. food-seeking amplified by an energy deficit) is itself costly.
    effort_cost: float = Field(
        0.0, ge=0.0, description="Energy per step per unit behavioral effort (summed positive "
        "action-atom activation). 0 = off."
    )
    # Fatigue as a metabolic load (requires fatigue_gain > 0 to accrue any fatigue): each unit of
    # total atom fatigue costs this much energy per step. 0.0 = off (fatigue only decrements force).
    fatigue_energy_cost: float = Field(
        0.0, ge=0.0, description="Energy per step per unit total atom fatigue (0 = off)."
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

    # --- Cue generalization (population of value-tuned receptors) ----------
    n_cue_receptors: int = Field(
        11, ge=1, description="Receptors tiling the scalar cue dimension v in [0,1]."
    )
    cue_generalization_beta: float = Field(
        6.0,
        ge=0.0,
        description="Shepard tuning width: receptor_k = exp(-beta*|v - c_k|).",
    )
    cue_learning_rate: float = Field(
        0.0005,
        ge=0.0,
        description=(
            "Cue-receptor association rate. Kept small so weights stay "
            "sub-asymptotic and track receptor activation -> a graded, peaked "
            "generalization gradient (large rates saturate and flatten it)."
        ),
    )
    consequence_model: Literal["delta_energy", "subtractive", "concatenated_asymmetric"] = (
        Field(
            "delta_energy",
            description=(
                "Pluggable consequence model (reinforcement/punishment asymmetry). "
                "'delta_energy' = symmetric (consequence == change in energy); "
                "'subtractive' = de Villiers (1980): a punisher cancels punishment_weight "
                "reinforcers -- aversive learning is scaled by that weight so avoidance is "
                "trained more strongly than approach; 'concatenated_asymmetric' = "
                "Critchfield/Klapes: separate reinforcement vs punishment sensitivities "
                "(reinf_sensitivity, punish_sensitivity) on the appetitive/aversive "
                "teaching signals, so the two are independently tunable."
            ),
        )
    )
    punishment_weight: float = Field(
        2.0,
        ge=0.0,
        description=(
            "de Villiers c (subtractive model): reinforcers cancelled per punisher. "
            "Scales the aversive teaching signal so avoidance is trained punishment_weight "
            "times more strongly than approach (the reinforcement/punishment asymmetry)."
        ),
    )
    reinf_sensitivity: float = Field(
        1.0,
        ge=0.0,
        description="Concatenated model: scale on the appetitive (reinforcement) teaching signal.",
    )
    punish_sensitivity: float = Field(
        1.0,
        ge=0.0,
        description="Concatenated model: scale on the aversive (punishment) teaching signal. "
        "punish_sensitivity/reinf_sensitivity is the asymmetry; >1 = punishment dominates.",
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
    food_intake_scaling: Literal["constant", "biomass"] = Field(
        "constant",
        description=(
            "How per-step intake depends on patch biomass. 'constant' = a fixed "
            "food_intake_rate while in contact (time-at-patch feeding; the default, "
            "and what the single-patch engine assumes). 'biomass' = intake scales "
            "with biomass fraction (food_intake_rate * biomass/K), a Holling/"
            "functional-response model: depleted patches yield diminishing intake, "
            "so within-patch returns fall and hunger re-engages foraging."
        ),
    )

    # --- Day/night ambient light (Phase 5) ---------------------------------
    # A global sun L(t) = 0.5*(1 - cos(2*pi*(t mod steps_per_day)/steps_per_day)) in
    # [0,1] (0 = midnight, 1 = noon). Opt-in: when off, L is ignored and behavior is
    # byte-identical. When on, light GRADES PERCEPTION, not the physical consequences:
    # danger is harder to detect and food harder to see at night, and food regrows
    # faster by day. Physical danger contact and food intake still depend on TRUE
    # proximity -- so a starving organism that forages at night accepts a risk it cannot
    # see (risk-sensitive foraging).
    day_night: bool = Field(False, description="Enable the day/night ambient light cycle.")
    steps_per_day: int = Field(96, ge=2, description="Timesteps per full day/night cycle.")
    danger_detect_floor: float = Field(
        0.2, ge=0.0, le=1.0,
        description="Night danger detectability: danger_sensed = danger_true*(floor+(1-floor)*L).",
    )
    food_light_floor: float = Field(
        0.3, ge=0.0, le=1.0,
        description="Night floor on food visibility AND regrowth (scale by floor+(1-floor)*L).",
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
