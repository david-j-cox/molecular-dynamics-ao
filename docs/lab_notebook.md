# Lab Notebook — Behavioral Molecular Dynamics Engine

A running record of what we tried, what the data showed, and why we made each
decision. Newest entries at the bottom. Each experiment that produces numbers
has a reproducible script under `experiments/` and raw output under
`outputs/logs/` (gitignored; regenerate by running the script). Key numbers are
copied into the relevant entry so the notebook stands alone.

Conventions:
- **Q** = question/hypothesis, **M** = method, **R** = results, **I** =
  interpretation, **D** = decision/next step.
- Dates are absolute.

---

## 2026-05-30 — Build log (Phases 1–3)

- **Phase 1**: package scaffold, `SimulationConfig`, `BehavioralFieldEnv`
  (Gymnasium-compliant; passes `check_env`). pygame made optional (no Py3.14
  wheel).
- **Phase 2**: `BehavioralAtom`, `verlet_update`, `ForceCalculator`. Decisions
  recorded with the user:
  - Learning rule: **both** RW and linear, selectable (`learning_rule`).
  - Dynamics: **literal pure Verlet** (no restoring/damping term) — user choice,
    over my flagged concern about stability.
  - Emission: **argmax now**, softmax later — user choice.
  - Mass: **static** for v1.
  - Behavior-environment relation: **directional projection** — a stimulus drives
    a movement atom ∝ `dot(stimulus_unit_vector, atom_direction)`; sign of the
    history weight encodes approach (+) vs. avoid (−).
- **Phase 3**: `Organism`, `learning.py` (eligibility trace + history update),
  `simulation.py` (loop + long-format logging). Credit assignment made
  **selectable**: `rw_competitive | rw_independent | source_only`.

---

## 2026-05-30 — Exp 001: Why doesn't the organism reach food?

**Script:** `experiments/exp001_dynamics_and_emission.py`
**Raw output:** `outputs/logs/exp001_results.json`
**Fixed layout:** start (1,1), food (6,6), danger (9,0), light (0,9), cue (3,3);
10×10 grid, Manhattan start→food = 10.

### Q
The first acquisition run consumed food in **0/30 episodes**; latency censored at
the 120-step cap every episode; all history weights stayed 0 (no consumption →
no consequence → no learning). Why?

### M
Instrument a single episode (print per-step action, position, atom activations),
then sweep candidate dynamics/emission rules over 12 seeds and record reach rate
+ latency. Candidate variations (velocity damping, softmax emission) are
prototyped *outside* core code in the experiment script; core `Organism` is
unchanged.

### R — Bug 1: `consume` dominated from a distance
With `consume.sensitivity[food]=1.0` and a broad sensory field
(`sensor_range=10`), at the start cell (distance 7.07) food intensity ≈ 0.49, so
`consume`'s drive ≈ 0.49 — larger than any movement atom's projected drive
(≈ 0.07). The organism emitted `eat` on every step and never moved:

```
t= 0 act=eat pos=[1,1] consume_act=0.005  (all move atoms ≈ 0)
t=10 act=eat pos=[1,1] consume_act=0.313
t=30 act=eat pos=[1,1] consume_act=1.714   <- still sitting on start cell
```

**Fix applied to core:** consummatory responses are *contact*-gated, not distal.
Added a per-atom `contact_exponent`; `consume` uses `intensity**6`, so its drive
is ≈ 0 at distance and ≈ 1 only on the food cell (where movement atoms get zero
food drive because the food direction vector is zero). Movement atoms keep the
linear distal field (`exponent=1`).

### R — Bug 2: pure Verlet + argmax pins against walls
After fixing Bug 1 the organism moves but **never reaches food**. Sample
trajectory under the literal spec (pure Verlet, argmax), food at (6,6):

```
(1,2)(1,3)(1,4)(1,5)(1,6)(1,7)(1,8)(1,9)  -> runs up the left wall
(2,9)(3,9)(4,9)(5,9)(6,9)(7,9)(8,9)(9,9)  -> across the top wall
(9,9)(9,9)(9,9)... pinned at the corner for the rest of the episode
```

Mechanism: Verlet makes activation the **double integral of force**
(force→acceleration). `move_up` banks a large activation on the way up; when the
organism passes food's row the force on `move_up` reverses, but its *activation*
stays the argmax winner long after, so it keeps emitting "up" into the wall.
Deterministic argmax has no way to shed the banked activation → limit cycle /
wall-pinning.

### R — Reach rate over 12 seeds (the key table)

| Configuration | reaches food | median latency | min dist if failed |
|---|---|---|---|
| `pure_verlet_argmax` (literal spec) | **0 / 12** | — | 3.0 |
| `pure_verlet_softmax` (T=1.0) | 1 / 12 | 29 | 3.16 |
| `damping_c2_argmax` | 0 / 12 | — | 3.0 |
| `overdamped_c20_softmax` (T=0.3, food_sens=1.5, sensor_range=5) | **10 / 12** | 114 | 0.5 |

(Numbers are from `outputs/logs/exp001_results.json`, n=12 seeds, 200-step cap.)

Earlier finer sweeps (not all in the script): softmax alone is seed-fragile
(1–4/12 across T=1–3); damping alone with argmax still pins (0/12 at c=0.5…2);
overdamped + softmax is the only combination that reliably reaches, robust to
food/no-food danger placement (8–10/12).

### I — interpretation (physics ↔ behavior)
- **Velocity damping** `−c·(x−x_prev)/dt` makes each atom a *damped* harmonic
  oscillator. As damping grows the second-order Verlet dynamics become
  effectively first-order ("overdamped"), so activation stops being an inertial
  integrator and becomes a **leaky accumulator of current drive**
  (Usher–McClelland leaky competing accumulator). This is also standard in real
  MD: Langevin dynamics adds exactly a `−γv` friction term — damping is not a
  departure from the MD analogy.
- **Softmax emission** = the **Luce choice rule / matching law** (temperature ↔
  matching sensitivity). It breaks the argmax limit cycle and makes response
  allocation lawful/graded rather than winner-take-all.

### D — OPEN DECISION (with user)
Proposed: change *defaults* to velocity-damping (overdamped) + softmax emission,
keeping pure-Verlet and argmax reachable via config so nothing is lost. Awaiting
user input on the dynamics framing before editing core. Bug-1 fix
(`contact_exponent`) already applied to core and committed.

**RESOLVED 2026-05-30:** user approved overdamped Verlet + softmax as defaults,
literal forms kept as config switches. See the Phase-4 design entry below.

---

## 2026-05-30 — Phase 4 design: long-life energy-budget foraging (PLAN)

Major redirection agreed with the user. This entry is the design of record; all
numeric values marked *(proposed)* are starting points to calibrate, not commitments.

### Settled design decisions (with rationale)

1. **Eating does not end the episode.** Model life in the long run. Lives run
   ~10,000 steps. Episodes end only on **death** (energy reserve depleted).
2. **Timescale:** 1 step = 15 min; **96 steps = 1 day** *(expandable)*. A
   10,000-step life ≈ 104 days.
3. **Motivation -> objective energy budget** (behavioral ecology), replacing the
   hunger/MO construct. Energy is a conserved reserve with literal in/out flows.
   Nothing is "valued" or "established."
4. **Energy -> behavior coupling: convex marginal value of energy.** The
   food-directed force rises steeply as the reserve approaches starvation
   (state-dependent / risk-sensitive foraging). Objective: energy is worth more
   nearer the death boundary, so foraging pressure is a function of the
   measurable reserve, not a posited motivational state.
5. **Light = global ambient sun** `L(t) in [0,1]`, cycling over the 96-step day.
   It is a discriminative *context / setting event*, sensed as a scalar (no
   direction to move toward). Distinct from the localized discriminative stimuli
   (old "light"/"cue" point sources = S+/S-delta/neutral signals) which remain a
   separate construct to develop later.
6. **Darkness raises risk, graded with a low floor.** `danger_sensed =
   danger_true * (floor + (1-floor)*L)`, floor ~0.2 *(proposed)*. Night vision is
   poor but nonzero; a near-starving organism rationally accepts elevated night
   risk to feed (couples to #4).
7. **Food = multiple renewable patches** (lean 2-3) with **logistic regrowth**,
   growth rate proportional to `L` (grows in daylight), drawn down on
   consumption. Food is also sensed more strongly in daylight (same graded form
   as danger). Multiple independently-regrowing patches make patch-foraging
   (marginal-value theorem) and **concurrent-schedule matching** emerge.
8. **Schedules: both, on one engine.** (a) The grid yields interval-like
   schedules and matching *naturally* from concurrent patch regrowth (faithful to
   the molar/foraging reading of operant behavior -- Baum, Rachlin). (b) A
   separate minimal **operant chamber** (feeder + `press` response, no
   navigation) reuses the same Organism/atoms/forces/learning for textbook-clean
   FR/VR/FI/VI curves and conc VI-VI matching.
9. **Dynamics/emission defaults:** overdamped Verlet (velocity damping `-c*v`) +
   softmax emission (Luce/matching). Literal pure-Verlet and argmax stay as
   config switches. (Justified by Exp 001.)
10. **Death is a dependent variable.** Log time-to-death, cause, and the energy
    trajectory leading up to it, so survival/mortality patterns can defend the
    modeling choices in write-up.

### Objective energy model (proposed equations + numbers)

Reserve `E` normalized to [0, 1]; `E_init = 0.5`. Per step:

```
E <- E - basal - move_cost(action) + intake - danger_cost
basal        = 0.01            (proposed; matches user's metabolic anchor)
move_cost    = 0.005 for a move, ~0.001 for pause/no-op   (proposed)
intake       = energy drawn from the patch when consuming  (see patches)
danger_cost  = 0.1 * danger_sensed   (injury = energy loss; objective)  (proposed)
death when E <= 0  -> episode ends
```

- **Consequence = change in energy from contingent events** (food gain +,
  danger loss -). This is what feeds the learning update -- the reinforcer is
  literally energy, not an abstract +/-1. (Baseline metabolism/movement are
  non-contingent costs, not learning consequences.)
- **Convex coupling:** food-directed drive multiplied by `g(E) = (1 - E)**2`
  *(proposed; 1/E variant available)*.

### Food patch dynamics (proposed)

Per patch, biomass `B in [0, K]`, `K = 1`:
```
B <- B + r * L * B * (1 - B/K)        # logistic growth, daylight-gated
r = 0.05 (proposed)
on consume at patch: bite = min(B, 0.2);  B -= bite;  intake = yield * bite
yield = 1.0 (proposed)
food_sensed = B * (floor_f + (1-floor_f)*L),  floor_f ~0.3 (proposed)
```

### Day/night (proposed)

```
L(t) = 0.5 * (1 - cos(2*pi * (t mod 96) / 96))   # 0 at midnight, 1 at midday
```

### Compute / parallelism (proposed)

Start with **multiprocessing across independent lives/conditions** (e.g. 64
lives/condition); near-linear speedup, no rewrite. JAX vectorization deferred.

### Logging / metrics additions

Add to the per-step log: `energy`, `light`, per-patch `biomass`, `cause_of_death`.
New metrics: survival curves, time-to-death distributions, death-cause breakdown,
day/night activity and mortality, patch allocation vs. relative intake (matching),
schedule response patterns (chamber).

### Proposed build order (phases)

- **4a** Dynamics/emission defaults: damping + softmax in core (controller must
  work before a 10k-step life is viable).
- **4b** Energy budget: reserve, costs, food=energy intake, danger=energy loss,
  death; convex coupling replacing `hunger`. Death-pattern logging.
- **4c** Persistent episodes (eating no longer terminates); long lives.
- **4d** Multi-patch logistic food + daylight-gated growth/visibility.
- **5**  Global ambient sun + graded light-gating of danger and food.
- **6**  Operant chamber env + schedule layer; FR/VR/FI/VI + conc VI-VI matching.
- **7**  Parallel runner; metrics + figures incl. death-pattern analysis.

### Open / proposed (awaiting confirmation)
- All *(proposed)* numbers above (energy magnitudes, growth rates, floors).
- `press` atom for the chamber (vs. reusing `consume`).
- Parallelism approach (multiprocessing first vs. JAX now).
- Discriminative-stimulus (S+/S-delta) semantics -- deferred by user.

---

## 2026-05-30 — Reinforcement/punishment asymmetry: literature + modular plan

User pushback: modeling danger as an instantaneous energy debit ("injury") is
too crude and collapses the **fundamental asymmetry between reinforcement and
punishment**. Objectively, injury is not energy spent at the moment of contact;
it is (1) a **delayed healing cost** -- extra energy to heal *plus* keep living,
raising the metabolic burden over a recovery window -- and (2) a **repertoire
impairment** -- specific responses are disabled/weakened while healing (the
"I can only use one arm" point), which is itself a competitive-suppression
mechanism. Decision: keep ΔE for now, but make the consequence/punishment model
**modular** so these can be explored.

### Literature (for write-up defense)
- **Asymmetry is empirical.** A single punisher subtracts more value than a
  single reinforcer adds; punishment also lowers sensitivity to reinforcement and
  biases toward the unpunished alternative (Rasmussen & Newland, 2008, JEAB).
- **Direct/subtractive suppression** (de Villiers, 1980): punisher subtracts from
  the same response's value. `B1/B2 = (R1 - c*P1)/(R2 - c*P2)`.
- **Competitive suppression** (Deluty, 1976): punishing a response strengthens
  *competing* responses; suppression is reallocation, not direct weakening.
  `B1/B2 = (R1 + c*P2)/(R2 + c*P1)`. (Matches the "repertoire impairment" intuition.)
- **Concatenated GML** allows asymmetry via separate sensitivities:
  `log(B1/B2) = a_r*log(R1/R2) - a_p*log(P1/P2) + log b`, `a_r != a_p`.
- **Klapes & Riley (2018, JEAB), "Toward a contemporary quantitative model of
  punishment":** five GML-based models (additive/Deluty x2, subtractive/de
  Villiers x2, concatenated GML); information-theoretic selection casts doubt on
  the subtractive model being the "true" account (it does not convincingly beat
  plain GML).
- **Klapes & McDowell (2025, JEAB):** contemporary model for **continuous choice**
  under combined reinforcing + punishing contingencies.
- Critchfield et al. (2003): direct vs. competitive suppression test, mixed
  support for direct suppression in human choice.

### Design implication: pluggable `ConsequenceModel`
Route every consequence through an interface so the organism core never hard-codes
how reinforcement/punishment act. Planned implementations:
- `DeltaEnergy` (default now): consequence = ΔE; danger = energy loss; symmetric currency.
- `Subtractive` (de Villiers): punisher subtracts from the punished response's drive/weight.
- `CompetitiveSuppression` (Deluty): punisher boosts competing atoms' drive.
- `ConcatenatedAsymmetric` (Klapes): separate reinforcement/punishment sensitivities.
- `InjuryHealing`: contact triggers a delayed healing energy cost (recovery window)
  + temporary impairment of specific atoms (repertoire loss), not an instant debit.

This is added to the Phase-4 open items; ΔE remains the default until we explore
the alternatives.

### Sources
- https://onlinelibrary.wiley.com/doi/10.1002/jeab.70009 (Klapes & McDowell, 2025)
- https://pubmed.ncbi.nlm.nih.gov/29509286/ (Klapes & Riley, 2018)
- https://pmc.ncbi.nlm.nih.gov/articles/PMC2251321/ (Rasmussen & Newland, 2008)
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1284944/ (Critchfield et al., 2003)

---

## 2026-05-30 — Phase 4a DONE + Exp 002 (core controller, parallel sweep)

Implemented in **core**: velocity damping (`config.damping_coef`) in
`Organism.step` and softmax emission (`config.emission`/`softmax_temperature`) in
`Organism.emit_action`. Literal pure-Verlet (`damping_coef=0`) and `argmax` remain
selectable.

**Exp 002** (`experiments/exp002_core_controller_sweep.py`, parallel via
`experiments/_parallel.py`): 1,152 lives = damping{2,10,20,40} x temp{0.2,0.3,0.5}
x food_sens{0.5,1,1.5} x sensor_range{5,10} x 16 seeds, run across 18 cores in
**2.2 s wall** (~15 cores busy). Raw: `outputs/logs/exp002_results.json`.

Findings:
- Best cells reach **15/16** (was 0/12 under literal spec in Exp 001).
- `food_sens=1.5` dominates the top -> promoted to the movement-atom default
  (was 0.5).
- damping 10–20 and temperature 0.2–0.3 are the sweet spot; **damping=40 is
  over-damped (0/16)**.
- Chosen defaults: `damping_coef=10`, `emission=softmax`, `softmax_temperature=0.3`,
  movement `food` sensitivity `1.5`, `consume_radius=1.0`. Pure-default reach on the
  fixed test layout: **12/16, median latency 48** (T=0.5 gives 14/16 if more
  exploration is wanted).

Caveat logged: reach rate on this fixed "go to the food" layout is a weak proxy;
final dynamics tuning happens once the foraging world (4b–4d) is in place.

**Parallel infra** (`experiments/_parallel.run_sweep`) is reusable for the larger
sweeps (1000s of lives across parameter grids) planned for later phases.

Next: **4b** energy budget + pluggable `ConsequenceModel` (ΔE default).

---

## 2026-05-30 — Phase 4b built; Exp 003 reveals a learning architecture problem

**Built (4b + persistent episodes):** objective energy budget (reserve, basal
metabolism, movement/rest costs, contact-based food intake, danger energy loss),
convex marginal-value coupling (`(1-E/Ecap)**deficit_exponent`), death at E<=0
with cause attribution, pluggable `ConsequenceModel` (`DeltaEnergy` default,
consequence = ΔE), eating no longer ends the episode (death/truncation do),
contact-gated consummatory `consume` (sharp binary `food_contact` signal),
proximal-gated `pause`. Energy bookkeeping verified to conserve exactly; convex
coupling verified (starving food drive 1.54 -> 3.16); death/starvation works.

**Exp 003 (`experiments/exp003_learning_curve.py`): learning does NOT tighten
approach -- it is incoherent.** 72 persistent organisms x 60 lives, sweeping
eligibility_decay {0.9,0.95,0.99}. `hw_food` (movement atoms' learned food
weight) stays **negative** (-0.5 to -0.9) across all lives; contact rate flat or
declining. Follow-ups (all with data in this session):
- eligibility_decay (temporal weighting): barely matters (0.99 slightly less
  negative, still wrong).
- sensor_range {2,3,5,10}: `hw_food` negative at every width -> not a broad-field
  miscredit problem alone.
- reinforcement_asymptote (lambda) {1,5,20,50}: does not fix sign; lambda=1 even
  showed mild contact/survival gains with `hw_food` still negative.

**Root cause (controlled no-danger trace): you cannot coherently learn on signed
directional units.** A directional movement atom (e.g. `move_right`) is positive
when food is in its direction and negative when food is opposite, so its
*eligibility* swings sign (observed +0.05 -> -3.2 -> +1.7 within one life).
Crediting a consequence times a sign-flipping eligibility gives self-cancelling
credit. With food in varying locations, no movement atom is consistently "toward
food", so none develops a stable food weight (they average ~0; danger bleed
makes them slightly negative). Secondary: per-step food intake (0.05) is a tiny
RW target, so positive credit is weak even when correctly signed.

The non-directional `approach_food` atom *could* learn cleanly (reliably positive
when food present) -- but in the chosen **single-tier directional-projection**
design it does not drive locomotion.

**OPEN DECISION (architecture):** single-tier directional projection cannot
support coherent learning. Options:
1. **Two-tier** (originally offered, then declined): non-directional drive atoms
   (approach_food, ...) learn cleanly; their activation is projected onto the
   movement atoms via the live stimulus geometry each step. Learning lives on
   sign-stable response-class units.
2. **Sign-corrected single-tier**: learn the food *sensitivity gain* of movement
   atoms using an unsigned engagement signal (e.g. credit |activation| or
   rectified eligibility), not a signed history weight.
Energy/survival mechanics committed as WIP; learning effectiveness pending this
decision.

---

## 2026-05-30 — RESOLVED: two-tier + valence-split learning works

User chose **Option 1 (two-tier)**. Implemented:
- **Drive atoms** (approach_food, avoid_danger, approach_light, orient_to_cue):
  non-directional, sign-stable, tagged with a `stimulus` and `valence`
  (+approach / -avoid). They carry the learning history.
- **Movement atoms**: topographies with no sensitivity/history; their force is
  the *expression* of the drive atoms projected onto each move direction via the
  live stimulus geometry (`sum_k valence_k * activation_k * dot(stim_dir_k, move_dir)`).
- **Valence-split teaching signals**: `ConsequenceModel.learning_signals` returns
  `(appetitive, aversive)`. Appetitive (reinforcement) trains approach-valence
  drives; aversive (punishment) trains avoid-valence drives. Both *strengthen*
  the disposition (weights grow toward +lambda); valence handles direction in the
  expression. This removes the cross-valence contamination (danger no longer
  trains the food drive) and fixes the avoidance sign (punishment strengthens
  avoidance instead of driving its weight negative).
- Learning signals normalized to ~1 per event (raw per-step energy ~0.05 is too
  small an RW target); the energy *reserve* still uses raw objective energy.

**Exp 003 re-run (72 organisms x 60 lives, sensor_range=4):** learning now has the
right sign and rises.

| eligibility_decay | hw_food(drive) early->late | survived early->late |
|---|---|---|
| 0.90 | 0.267 -> 1.000 | 182 -> 178 |
| 0.95 | 0.514 -> 0.785 | 163 -> 156 |
| 0.99 | -0.469 -> -1.957 | 128 -> 114 |

Temporal-weighting result: decay 0.9-0.95 is the sweet spot; **0.99 is too long**
(credits stimuli not actually driving the approach -> weight goes negative).

**Remaining bottleneck (separate from learning):** stronger learned approach does
NOT yet raise contact rate / survival, because the organism cannot reliably
*stay* at food -- the consummatory `consume` holds it only until the next softmax
draw knocks it off, and when sated nothing keeps it near food, so it orbits and
drifts. This staying-at-food / patch-residence problem is the next target (likely
emission stickiness when consummatory, and/or a "stay while feeding" mechanism).
Architecture (two-tier) committed.

---

## 2026-05-30 — Option 1: consummatory inhibits locomotion -> positive learning curve

Staying-at-food fix (user chose Option 1): added strong coupling
`consume -> movement = -1.5` (consummatory behavior competes with / suppresses
locomotion: "don't walk away mid-meal"). Kept `approach_food -> consume = +0.30`
(removing it cut survival 5/16 -> 2/16; it helps survivors lock onto food).

Single long life (3000 steps, food adjacent, 16 seeds): survival 0/16 -> **5/16**
camp at food and live indefinitely. Outcome is bimodal -- organisms that
establish feeding survive; others fail to reach/lock on and die. Reaching+locking
on is now the fragile step.

**Across-lives learning curve (Exp 003 re-run, decay sweep, sensor_range=4):**
the camping mechanism + two-tier learning finally yields a positive acquisition
curve.

| eligibility_decay | contact early->late | survived early->late | hw_food early->late |
|---|---|---|---|
| 0.90 | 0.143 -> 0.136 | 189 -> 191 | 0.509 -> 1.000 |
| 0.95 | 0.117 -> 0.144 | 160 -> 204 | 0.420 -> 0.500 |
| 0.99 | 0.116 -> 0.092 | 162 -> 134 | 0.335 -> -0.748 |

decay **0.95 is the sweet spot**: contact rate, survival, AND the learned food
weight all rise across lives. Promoted to defaults: `eligibility_decay=0.95`,
`sensor_range=4.0` (with `damping_coef=10`, softmax `T=0.3` from Phase 4a).

Effect sizes are modest (survival +28%, contact +23% early->late) but the sign is
right -- the organism gets better at foraging and surviving the longer it lives.
Amplifying the effect (learning rate, reach reliability) and the bimodal
reach-failure are open tuning items; the qualitative acquisition result needed
for the demos is in hand. Phase 4 (energy foraging + learning) is functionally
complete; next is Phase 5 (day/night) or amplifying/again-parallel tuning.

---

## 2026-05-30 — Extinction, generalization, peak shift, and the JAX direction

Since the entries above (committed incrementally on `dev`):
- **Extinction**: pluggable `LearningRule` (RescorlaWagner with omission decay,
  asymmetric acq/ext rates, gated to contact exposure) -> trained `hw_food` 1.0
  extinguishes to ~0; `run_extinction_demo`.
- **Generalization**: `CueReceptorField` (population of value-tuned receptors,
  Shepard tuning, summed/elemental error) -> gradient peaked at the trained
  value; `run_generalization_demo`.
- **Peak shift**: balanced S+/S- discrimination on the receptor field shifts the
  peak past S+ away from S- (Hanson 1959); `run_peak_shift_demo`. (Foraging
  cannot balance S+/S- exposure -- reinforced organisms camp at S+ -- so the demo
  uses a controlled discrimination procedure.)
- All demos parallelized; `--agents` flag added; running at 10,000 agents.

### Next major phase: JAX-vectorized engine
Per-step Python overhead is the bottleneck. Plan (see [[memory]] jax direction):
collapse per-organism loops into batched array ops; pure `jit`/`vmap`/`scan`;
keep the NumPy OO engine as the canonical reference; validate equivalence.
Research roadmap order: **phenomena zoo -> evolution -> model comparison ->
real-data fitting** (autodiff is the enabler).

**Foundation built + validated (`jax_engine.py`):** `build_spec` packs the atom
set into static arrays; `compute_force` is the batched two-tier force;
`integrate` the damped Verlet step. `validate_against_numpy` confirms the JAX
force matches `ForceCalculator.compute` to ~1e-7 over random state. JAX 0.10.1
installs on the Py3.14/arm64 venv (CPU backend; autodiff available).

Remaining for a full vectorized engine: stochastic softmax emission (RNG via
`jax.random`), the valence-split learning update + eligibility, the cue field,
and a vectorized environment (grid/patches/cue) -- then `scan` over time and
`vmap`/batch over organisms, plus the phenomena zoo on top.

---

## 2026-05-31 — Death as a dependent variable (capture + first finding)

Wired death-pattern capture through the JAX engine and added `metrics.py`:
- `SimState.cause_of_death` (0 alive / 1 starvation / 2 danger), set at the death
  transition; `run_lives` returns per-life `survived` (time-to-death) and
  `cause_of_death`.
- `metrics.py`: `time_to_death`, `cause_breakdown`, `survival_curve`
  (Kaplan-Meier-style), `mortality_by_life`.
- `visualization.py`: `plot_survival_curve`, `plot_time_to_death`,
  `plot_mortality_by_life`. `experiments/exp007_death_patterns.py` runs it
  (3000 organisms x 40 lives in ~6s).

**First finding (single depleting patch; food [4,8], danger [6,7] off-corridor):**
near-total mortality (~100%); survival curve shows early danger deaths then a
starvation cliff at ~55 steps (initial energy exhausted). Across lives, a clear
**cause substitution**: danger-deaths fall (0.28 -> 0.19) as `avoid_danger`'s
weight grows, but starvation-deaths rise (0.69 -> 0.81) -- learned caution trades
danger mortality for starvation mortality; total mortality is unchanged. So
faster food approach (acquisition) does NOT improve survival on one depleting
patch. This motivates the multiple-patch economy (next), where leaving a spent
patch for another should make sustained survival, and a learning-driven mortality
drop, attainable.

---

## 2026-05-31 — Matching + COD sweep (cue-signaled concurrent VI-VI)

`matching.py` (vectorized JAX): two patches = same reinforcer under different
discriminative CUES (one cue dimension; green 0.2 / red 0.8), each on its own
Bernoulli-armed VI. A value-tuned receptor population learns cue->rate; that
learned value drives damped-Verlet directional approach; emission is softmax
(matching). Travel between patches = changeover delay (COD). Energy/death omitted
(steady-state allocation prep). Patches are NOT separate food channels (cues
scale; channels don't) -- the user's design.

- **Exp 008 (matching):** sweep VI rate ratios, fit GML log(B_L/B_R) =
  a*log(R_L/R_R) + log b. a=0.69 (undermatching, as in real animals), R^2=0.80;
  400 organisms x 7 schedules x 4000 steps in ~2s. Matching EMERGES from the
  mechanism (not imposed as an equilibrium, unlike the manuscript's top-down model).
- **Exp 009 (COD sweep):** vary patch separation (= travel steps = COD); fit a
  per organism at each. Result reproduces Shull & Pliskoff: a rises with COD --
  0.04 (sep 2) -> 0.16 -> 0.32 -> 0.79 (sep 10) -> 0.88 (sep 14), then a
  sampling-limited point at sep 18 (only 36/400 organisms sampled both patches
  enough; wide CI). Near-zero COD -> rapid switching -> near-indifference;
  adequate COD -> matching. Grid scales with separation; sensor_range kept large
  vs separation to isolate travel/COD from a detectability confound.

Next: ABABABAB reversal -> damped-oscillator return to equilibrium + Brunt-Vaisala
frequency vs mass (tie to manuscript Figs 7-9); then concatenated matching law
(amount/delay/probability).

---

## 2026-05-31 — Multi-alternative matching (N patches)

`matching.py` is already P-patch generic; exp010 places N=5 cue-marked patches on
a circle around the start (equal travel/COD to each -> no spatial bias), sweeps 16
random VI configurations, and fits the multi-alternative GML with each alternative
vs. the pooled rest: log(B_i / Sum_rest) = a*log(R_i / Sum_rest) + log b.

Result: a=0.99, log b=-0.03, R^2=0.75 (5 alts x 16 conditions x 300 organisms x
6000 steps in ~8s) -- near-perfect matching, vs the 2-patch undermatching (a=0.69).
The circle geometry puts COD in the matching regime; multi-alternative concurrent
schedules commonly match well. Data fall in the lower-left (each alternative's
reinforcement relative to the pooled rest is usually <1 with N=5), and the fit is
essentially the identity over that range. Same cue-signaled mechanism as 2-patch,
no new machinery -- just more alternatives.

---

## 2026-05-31 — Concatenated matching law: AMOUNT term

Added a per-patch reinforcer AMOUNT to matching.py (runtime arg): the cue's
learned value is trained toward `lambda * amount` collected, so the learned value
(hence approach) tracks reinforcement magnitude, not just occurrence. sim now
returns (time_at, count, amount_obtained).

**Exp 011** (identical VI on both patches, sweep amount ratio): amount term
`log(B_L/B_R) = a_amt*log(A_L/A_R) + log b` -> a_amt=1.28 (slight overmatching),
R^2=0.79. Rate matching unchanged (regression: exp008 still a=0.69). So the model
has SEPARABLE sensitivities: a_rate=0.69 (undermatch) vs a_amt=1.28 (overmatch) --
exactly what the concatenated law allows (a_r != a_amt). The asymmetry is
mechanistically sensible: obtained-rate is behavior-dependent (feedback ->
undermatching), whereas programmed amount sets the learning target directly ->
value ratio ~ amount ratio, amplified by softmax -> overmatching.

Caveat: a consistent bias toward the cue-0.8 patch (log b = -0.29 rate, -0.63
amount) -- a systematic preference to investigate (likely cue-receptor cross-talk
or an asymmetry between the two cue values), not yet explained.

Next concatenated-law terms: delay and probability (same per-patch runtime-arg
pattern).

---

## 2026-05-31 — Canonical counterbalanced matching; a spurious side bias found+fixed

User flagged that all matching prep should include a 1:1 condition and
counterbalance the rich/lean assignment so only the schedule controls responding.
Built exp012 (canonical conc VI-VI): ratios 9:1, 3:1, 1:1, 1:3, 1:9 (total rate
constant), cue<->side COUNTERBALANCED (half the organisms cue 0.2 left, half cue
0.2 right).

**Diagnosis.** At 1:1, log(B_L/B_R) was identical (-0.52) for BOTH cue
assignments -> intrinsic CUE bias = 0, but a strong SIDE bias toward the right
patch. Cause: the default 10x10 grid (cells 0-9) put the right patch (x=8) 1 cell
from the wall and the left patch (x=2) 2 cells from its wall; the closer wall
traps the organism near the right patch. The layout was not actually symmetric.

**Fix.** Use an odd grid (size 11, center 5) so a centered layout has equal wall
margins. This removed the bias everywhere:
- exp008 (rate, sep 6): a 0.69->0.56, bias -0.29 -> -0.01.
- exp011 (amount): a 1.28->0.97 (now near-perfect amount matching), bias -0.63 -> 0.00.
- exp012 (canonical): bias -0.01, cue bias exactly 0.
So the earlier biases were entirely the even-grid artifact; counterbalancing + the
1:1 control is what exposed it. MatchConfig.grid_size default -> 11.

Note: rate matching at separation 6 is undermatching (a~0.2-0.6), consistent with
the COD sweep (exp009): sep 6 is in the undermatching regime. Pooled GML at large
separation (sep 10) degrades (near-exclusive choice, schedule stops modulating);
exp009's per-organism slope fits with scaled sensing are the systematic COD
characterization. Headline matching quality is a function of COD, not a single number.

---

## 2026-05-31 — Concatenated matching law complete (rate, amount, probability, delay)

Added the PROBABILITY and DELAY dimensions to matching.py (per-patch runtime
args), completing the concatenated GML. All four arise from the same mechanism --
the cue's learned value integrates reinforcement events, each scaled by amount,
gated by probability, discounted by delay, at a frequency set by rate -- and the
softmax (matching) emission converts value to allocation.

- **Probability** (exp013): a contact with an armed patch is reinforced w.p. p_k;
  non-reinforced contacts are extinction trials (target 0). Partial reinforcement
  -> value ~ p*lambda. Equal VI + equal amount, sweep p: a_p=0.78, bias -0.02,
  R^2=0.88.
- **Delay** (exp014): reinforcer delivered after delay D_k reduces its EFFICACY
  (empirical delay discounting), modeled as a discount on the teaching signal --
  NOT a credit-assignment/eligibility effect (per user: the standard procedure
  blacks out during the delay, so attribution is unambiguous; delay just makes
  the reinforcer a weaker strengthener). Discount form selectable
  (hyperbolic 1/(1+kD) default | exponential exp(-D/tau)); efficacy-discount only
  (no procedural blackout). Equal VI + equal amount, sweep D: slope -0.61
  (correct negative sign), a_d=0.61, bias 0.00, R^2=0.82.

Separable sensitivities, all ~0 bias on the odd grid: a_rate~0.56, a_amount~0.97,
a_prob~0.78, a_delay~0.61. The concatenated matching law is reproduced bottom-up.

---

## 2026-05-31 — Operant chamber + schedules: graded responding, but no molecular signatures

Built `chamber.py` (vectorized): single press response, no navigation, schedules
FR/VR/FI/VI; pressing driven by learned press-value + energy-deficit motivation,
damped-Verlet dynamics, logistic emission. Added a RESTORING force (spring toward
baseline): without it a roughly-constant drive ramps activation to the clip and
pressing saturates at 1.0; with it, equilibrium activation ~ drive/restoring so
response rate is GRADED.

**Result (exp015):** graded, reinforcement-maintained pressing (~0.88 resp/step),
but:
- response rates are ~EQUAL across VR/VI/FR/FI at matched reinforcement;
- FI is FLAT within the interval (scallop index +0.001) -- no scallop.
Tried stronger energy dynamics (food/metabolism so the deficit varies within the
interval): still flat, undifferentiated.

**Why (the honest boundary).** The classic molecular signatures are not produced
because this is a MOLAR value model: response strength is a scalar value updated
on reinforcement OCCURRENCE. The signatures require molecular mechanisms it lacks:
- VR>VI rate difference: differential reinforcement of inter-response times (VI
  differentially reinforces long IRTs; VR does not). Needs IRT-level credit.
- FI scallop / FR break-and-run: temporal/count discrimination of proximity to
  reinforcement. Needs a clock/counter discriminative stimulus.

So the engine reproduces molar choice/matching well (concatenated GML, COD) but
molecular within-schedule structure needs added mechanism. FORK for next step:
(a) IRT-level reinforcement (credit the recent inter-response time) -> VR/VI rate
difference and rate structure; (b) a time/count discriminative cue (reuse the
CueReceptorField as a "clock") -> FI scallop / FR break-and-run via the
generalization gradient over elapsed-time. Chamber + finding committed as WIP.

---

## 2026-05-31 — Operant chamber, take 2: effort + unit price (behavioral economics)

User's insight: a single response in isolation is degenerate ("why not jam the
lever") -- behavior is choice, and the single-response rate is set against OTHER
behavior with its own reinforcement (Herrnstein's R_e). Two fixes, kept strictly
first-principles:
- **Effort-opposed value with extinction.** The press value is strengthened by
  reinforced presses and ERODED by unreinforced ones (ordinary RW up/down on the
  response); pressing costs energy (effort). On ratio schedules every press has
  the same reinforcement probability -> value stays high -> high rate; on interval
  schedules, faster pressing makes more presses unreinforced -> value erodes ->
  rate self-limits. The organism never computes unit price; it emerges.
- Added a restoring force earlier so rates are graded (not pinned at the clip).

Results:
- Rate ordering directionally correct but WEAK: VR 0.75 / FR 0.77 > VI 0.71. Full
  molar VR>>VI needs sensitivity to the feedback FUNCTION (how reinf rate changes
  with response rate) -- a known hard problem; not fully captured.
- **Demand curve (exp016): robust.** Sweeping FR size (= unit price =
  responses*effort/magnitude), consumption falls monotonically 0.88 -> 0.005
  reinforcers/step; log-log elasticity ~ -0.96. The Hursh demand relation emerges
  from effort + reinforced-up/unreinforced-down value. (High-FR end is partly an
  emission-ceiling artifact -> presses max; a give-up/breakpoint would need
  stronger extinction. Consumption is the robust DV.)

Concurrent chamber (run_concurrent_chamber) also added (M responses, per-response
effort) but matching/Herrnstein is NOT the goal here per user -- single-response
schedule signatures are.

OPEN (to discuss, not yet built): the time/count piece for FI scallop and the
FR/VR post-reinforcement pause. Constraint: 100% first principles, NO mentalism
(no internal clock that is "read"). See next entry once decided.

---

## 2026-05-31 — Pluggable timing models -> FI scallop (toggleable switches)

Built timing.py: a pluggable TimingModel interface (like ConsequenceModel /
LearningRule), selected by ChamberConfig.timing_model. Four toggles spanning the
theory landscape, all vectorized over organisms:
- none / homeostatic: NO timer (energy carries timing if anything). First-principles.
- set: Scalar Expectancy Theory -- pacemaker-accumulator + reference memory +
  comparator. Cognitivist (an internal represented/read time); included for
  comparison, NOT first-principles.
- bet: Behavioral Theory of Timing (Killeen & Fetterman) -- stochastic pacemaker
  drives transitions through behavioral STATES; operant controlled by the state
  usual at reinforcement.
- let: Learning to Time (Machado) -- serial behavioral-state activation + LEARNED
  state->response links (RW); no comparator/stored duration. Most behavioral /
  least mentalist; maps onto the two-tier atom architecture.

Two chamber fixes were needed to make the timing signal control responding:
1. Press activation made a LEAKY INTEGRATOR (overdamped limit of damped Verlet),
   act_tau~3 steps -- the 2nd-order oscillator was either too sluggish (flat) or
   overshot (post-reinforcement spike) and washed out the within-interval ramp.
2. An emission THRESHOLD (emission_bias) so low drive = a genuine pause, and the
   timing signal drives pressing (separate press-value learning off, to isolate timing).
Also reset press activation on reinforcement (consume -> pause) for a clean PRP.

Result (exp017, FI-25, 400 organisms): "none" is FLAT (no scallop); SET, BeT, LeT
each produce a clear FI scallop (last-first +0.75..+0.88) with DISTINCT shapes --
LeT accelerates earliest, SET intermediate, BeT sharpest break-and-run. Clean
dissociation; the toggle framework shows each model's signature side by side.

Note: these timers tick per STEP (time-based) -> FI scallop. The FR/VR pause and
ratio patterns would use a per-RESPONSE (count-based) tick of the same models --
a natural next extension.

---

## 2026-05-31 — FR/VR patterns via count-based timing + cumulative records

Made the timing tick MASKABLE (tick(advance)): advance=1/step is a TIME clock
(FI); advance=press is a COUNT clock (FR). Added ChamberConfig.timing_clock and
tracked count-since-reinforcer for all schedules.

With the count clock (LeT), FR reproduces the classic pattern:
- Post-reinforcement pause much larger on FR than VR (FR-20 PRP=4.06 vs VR-20=1.36;
  FR-40=4.42 vs VR-40=1.68) -- because FR's count->value function is sharply peaked
  at n (low value at count 0 -> long pause) while VR's is spread (value-at-0 higher).
- Higher response rate on VR than FR (VR-20 0.91 vs FR-20 0.63) -- the FR pause
  drags its average down. exp018.

Cumulative records (exp019, plot_cumulative_record): the classic visualization for
a few AOs with reinforcer pips. FI (time clock) -> SCALLOPS (flat after each pip,
accelerating into the next); FR (count clock) -> BREAK-AND-RUN staircase (flat
pause then steep run). Records stacked with full-height offset (no overlap).

Schedule-signature scorecard now: FI scallop (timing), FR break-and-run + FR>VR
pause + VR>FR rate (count timing), demand-like consumption decline (effort/value).
Still weak: the molar VR>>VI rate difference (needs feedback-function sensitivity).

---

## 2026-05-31 — Multi-patch foraging in the JAX world: patch-leaving + MVT

Added a multi-patch food layer to the JAX SURVIVAL world (forage.py), distinct
from the matching work (matching.py has VI feeders but no biomass/depletion/energy/
death). The organism now senses P depleting/regrowing patches and approaches the
single most SALIENT one:

    salience_p = exp(-dist_p / sensor_range) * (biomass_p / K)

Direction/intensity/contact for the food channel are the salience-argmax over
patches; danger/light/cue stay single sources. Everything else (two-tier force,
damped Verlet, RW learning, cue receptors, energy budget, death) is the SAME
kernel as jax_engine.make_simulate. With P=1 it reproduces make_simulate to
0.0e+00 (validate_against_single_patch; test_forage.py) -- the multi-patch path is
a strict generalization, not a reimplementation.

Mechanism (no give-up rule hand-coded): while the organism feeds, the occupied
patch depletes (its salience falls) while the distant alternative regrows. The
``consume`` atom (sensitivity food 1.0 + energy-deficit gain) holds it on a rich
patch; once biomass drops enough that the alternative's distance-attenuated
salience wins, ``approach_food`` flips direction toward it and the organism leaves.
Predicted give-up density (leave when biomass_frac < exp(-D/range), alternative
near full) and longer residence for longer travel -- the marginal-value theorem.

exp020 (two patches, sweep travel distance D, 128 organisms x 3500 steps):
- Give-up density FALLS monotonically with D: 0.60 -> 0.58 -> 0.54 -> 0.48 -> 0.43
  for D = 3..7, parallel to and below the exp(-D/range) prediction. The offset is
  expected -- the prediction assumes a FULL alternative (frac_B=1), but the
  alternative is still regrowing at switch time, so the realized threshold
  exp(-D/range)*frac_B sits lower.
- Residence time RISES monotonically: 38 -> 46 -> 52 -> 57 -> 62 steps. Longer
  travel justifies depleting a patch further before leaving. Classic MVT.

Tuning notes (two failure modes fixed):
1. First pass STARVED (E~0): metabolism too high relative to intake. Made survival
   cheap (basal 0.005, move 0.005) + low softmax temperature (0.15) so the organism
   COMMITS to consuming a patch instead of random-walking off it -- leaving is then
   salience give-up, not emission noise.
2. The D-dependence only appears when the alternative is actually SENSED. With the
   default sensor_range=4, exp(-D/4) is negligible for D>=6, so leaving was governed
   by a flat hunger-dependent consume floor (no D-dependence). Set sensor_range=12
   and keep D <= sensor_range; beyond ~sensor_range the gradient breaks (few
   organisms detect/relocate to the far patch). Residence definition is by patch
   ALLEGIANCE (first contact until first contact with the OTHER patch), robust to
   brief out-of-range excursions that would otherwise read as spurious leaving.

---

## 2026-05-31 — Patch-leaving follow-ups: tenacity factor + functional response

Two improvements after noticing observed give-up density sits ~20% BELOW the naive
exp(-D/range) salience-crossover.

(1) Tenacity factor (exp020). The discrepancy is a CONSTANT multiplicative offset,
not a shape error: observed/naive = 0.79 +/- 0.018, flat across D=3..7. So the
shape exp(-D/range) is exactly right and one constant closes the gap:

    give_up_density = kappa * exp(-D/range),   kappa ~= 0.79

kappa is PATCH TENACITY / consummatory perseveration -- the organism depletes a
patch PAST the salience-indifference point because the consume atom's food gain
(1.0) exceeds approach_food's (0.5), amplified by leaky-integrator lag. The static
gain ratio (g_leave+d_gain)/(g_stay+d_gain) BRACKETS kappa but doesn't pin it:
diagnostics show the organism actually leaves when SATIATED (give-up energy ~0.85,
so d_gain~=0 -> gain ratio ~0.5, the lower bracket) while hungry would give ~0.82;
the realized 0.79 is set by the dynamics, so kappa is reported as a single fitted
constant validated by its constancy across D. Plot now shows observed, the naive
crossover, and kappa*crossover (which lands on the data within CI).

(2) Functional response (exp021, config.food_intake_scaling). Constant intake has
a flaw: the organism satiates on a rich patch, the food drive switches off, it
wanders away and starves -- 0% survive to end of run (give-up stats still valid,
pooled over each organism's first few visits, but the population isn't viable).
Added the Holling response intake = food_intake_rate * biomass/K ('biomass'
scaling; 'constant' is the default and preserves the P=1 == make_simulate check).
At the SAME nominal rate it fixes both problems:
  - Survival: a depleted patch yields diminishing intake -> hunger re-engages
    foraging -> end-of-run survival 0.00 -> 0.50.
  - Charnov reading: within-patch intake rate now genuinely diminishes (= rate *
    biomass_frac), so the GIVE-UP RATE (rate * give-up density) falling with travel
    distance (0.044 -> 0.034 over D=3..7) is a literal marginal-value signature,
    not just a salience artifact. Under constant intake the within-patch rate is
    flat, so the same give-up-density gradient is purely stimulus-control.
The MVT give-up-density gradient holds in BOTH regimes (functional response is
higher and shallower: 0.74 -> 0.57 vs constant 0.60 -> 0.43). Same phenomenology,
two resource models, only the functional response is Charnov-interpretable and
viable -- a nice dissociation of mechanism from phenomenon.

---

## 2026-05-31 — Behavioral momentum (multiple schedule): satiation yes, extinction no

Added run_multiple_schedule to chamber.py: K components presented successively,
each with its own VI rate and its own Pavlovian context->reinforcer value (the
behavioral-momentum "mass", an omission-RW association settling at a level graded
by that component's reinforcement RATE, NOT by responding). Train to baseline,
disrupt, measure resistance = per-component rate / its own baseline. exp022.

POSITIVE RESULT -- satiation/prefeeding disruptor (energy clamped to capacity ->
deficit ~ 0). The rich component (VI 5, reinf rate 0.17, context value 1.68) keeps
99.6% of baseline; the lean component (VI 40, rate 0.023, ctx 0.76) keeps 80%. So
the rich component is more resistant -- behavioral momentum, and mechanistically
clean: the rich component's higher context value is a larger NON-motivational share
of its press drive, so removing the deficit drive takes a smaller proportional bite.
This is Nevin's logic (resistance tracks the stimulus-reinforcer relation) falling
out of the model. Robust across parameterizations.

NEGATIVE / DIAGNOSED -- extinction is NOT clean momentum here (rich 0.57, lean 0.78
= anti-momentum). Worked through several mechanisms; the obstacles are real and
worth recording:
1. Energy-survival confound: withholding food IS starvation, which raises the
   deficit drive -> responding rises under extinction (proportions > 1), masking/
   reversing response extinction. Fix attempted: maintain deprivation (clamp energy)
   during the test so only the response-reinforcer contingency changes.
2. Response-based vs time-based value decay: the chamber erodes press value on
   unreinforced PRESSES, so a vigorous (rich) response self-extinguishes faster --
   anti-momentum. Switched the momentum runner to TIME-based decay divided by mass
   (Nevin: resistance per unit time, not per response; matches the Verlet mass =
   resistance-to-change metaphor, momentum_mass_gain). This helps in absolute terms.
3. Shared emission floor + the proportion metric: once value decays, both components
   fall to the SAME absolute floor (logistic at zero drive / shared deficit), so
   proportion-of-baseline is dominated by baseline HEIGHT (rich is higher -> lower
   proportion) rather than by mass. Removing the floor trades it for a logistic
   CEILING; a clean monotone "rich more resistant throughout" needs fragile tuning.
Conclusion: momentum is real in the model under a motivational (satiation) disruptor;
extinction-resistance momentum needs reinforcement decoupled from survival and a
floor-free, time-based, mass-scaled decay -- a clean future-work target, not faked.
Satiation/prefeeding is itself a canonical Nevin disruptor, so this counts as a
genuine behavioral-momentum demonstration.

---

## 2026-05-31 — Code sweep / refactor + reproduction harness

Built a reproduction harness (scripts/reproduce.py): runs every experiment + demo
in a fresh subprocess, scrubs volatile lines (wall-clock timings, throughput, abs
paths), and snapshots stdout to outputs/repro/baseline.json. `--check` re-runs and
diffs against the baseline so any numeric drift is explicit. Captured a baseline
(26/26 run cleanly), then refactored against it.

Refactor (verified: harness --check shows 0/26 experiments changed, all JAX
validators still <1e-6, forage P=1 == make_simulate exactly 0.0, 35 tests pass):
- New behavioral_md/experiment_utils.py centralizes helpers each experiment had
  copy-pasted: compute_mean_ci, fit_matching_law (was duplicated in 6 matching
  exps), make_cue_centers, make_inert_source, weak_innate_atoms, save_results_json.
  Lives in the installed package (not under experiments/) so every script imports
  it and still runs standalone -- no PYTHONPATH needed.
- New behavioral_md/_kernels.py: exp_falloff (exp(-d/range)) and safe_unit (0/0-safe
  unit vector), the sensory-geometry primitives that jax_engine, forage, and
  matching had each written inline. Wired into all three.
- Removed the experiments/_parallel.py passthrough; exp002/exp003 now import
  run_sweep from behavioral_md.parallel directly.
- Found and fixed a runnability bug: exp002/exp003/exp021 could not run as
  `python experiments/expNNN.py` (they did `from experiments... import`, which
  needs the repo root on sys.path). exp002/003 are now standalone; exp021 still
  builds on exp020's setup (the harness puts the root on PYTHONPATH).

Deliberately NOT changed (noted for later): the NumPy reference engine vs the JAX
twin keep their parallel implementations on purpose (the validators check
equivalence); the unimplemented consequence-model stubs stay as documented future
work; chamber/matching keep hardcoded activation/weight clip bounds (they use their
own config objects, values already match the SimulationConfig defaults). No
mechanism numerics changed -- this sweep was structural.

Follow-up structural change (same verification: 0/26 drift, validators <1e-6, P=1
== make_simulate 0.0, 38 tests): the near-duplicate per-step loops in
jax_engine.make_simulate and forage.make_forage_sim were factored into two shared
functions in jax_engine -- drive_integrate_emit (deficit-scaled force + cue drive +
damped-Verlet integrate + eligibility + softmax emission) and learn_with_cue
(valence-split RW + cue-receptor update). Both engines' step functions now differ
only where they should: observe (single vs multi-patch salience) and the env
transition (single-patch env_step vs multi-patch deplete/regrow + functional
response). The shared physics/learning lives in one place.

---

## 2026-06-02 — Fitting matching sensitivities (exp023): autodiff doesn't, derivative-free does

Goal: search organism parameters so the emergent generalized-matching-law
sensitivities (a_rate, a_amt) hit chosen targets -- the long-promised payoff of the
JAX engine ("autodiff is the enabler"). It did not go as planned, and the why is the
interesting part.

**Autodiff is unusable here.** The stochastic matching engine is non-differentiable
(categorical action sampling, Bernoulli VI arming), so I built a differentiable
surrogate. Two dead ends first:
- An *expected-value* (mean-field) surrogate -- expected displacement, expected
  arming -- collapses the trajectory and parks the organism at the higher-value patch:
  a_rate ~ 2.7-3.7 (severe overmatching), and the slope no longer tracks the
  stochastic one. Undermatching in this engine is a *sampling-noise* phenomenon (the
  organism keeps wandering to the poorer patch); remove the noise and you lose it.
- A *Gumbel-softmax* surrogate (reparameterized action sampling + relaxed-Bernoulli
  arming, common random numbers) fixes the FORWARD model beautifully: soft a_rate
  0.463 vs stochastic 0.565, a_amt 0.649 vs 0.967, and a beta sweep gives soft-vs-
  stochastic correlation 0.97 -- gradients should transfer. But the reverse-mode
  GRADIENT through the ~1000-step recurrent rollout EXPLODES: d(a_rate)/d(beta) came
  out at -310..+10 across noise seeds (mean -39, std 103) versus a clean finite-
  difference of ~+0.03. Per-step sensitivities compound multiplicatively through the
  fed-back state (learned weights, armed-probability, position) -- the classic
  exploding-gradient / chaos of differentiating long recurrent rollouts. Optimization
  with these gradients does nothing (Adam wanders; even an unreachable target fails to
  move beta).

**Why not a molar closed-form (where autodiff WOULD be clean)?** Because reproducing
the matching law in closed form requires assuming B_k ~ v_k^a -- i.e. writing the
sensitivity a in as a parameter. That forces the very result this engine is supposed
to let *emerge* from the coupled dynamics. Scientifically that defeats the purpose, so
it's out.

**Resolution (exp023, fit.py).** Keep the emergent dynamics; search the smooth,
deterministic-under-CRN Gumbel surrogate derivative-free (Nelder-Mead). Free params
temperature, approach_gain, beta. A per-parameter transfer probe was decisive:
- beta: strong, sign-consistent lever on a_rate (soft 0.37->0.50, stoch 0.34->0.71).
- temperature / approach_gain: weak on a_rate, consistent on a_amt.
- lr_cue: stochastic a_rate falls sharply with it, but the surrogate doesn't reproduce
  that (a noise-accumulation effect) -- EXCLUDED as a free param; its gradient/effect
  does not transfer.

On this two-patch preparation the two sensitivities are positively coupled (beta moves
both; temperature/approach_gain only trim a_amt by ~+-0.06), so the reachable joint
region is narrow and strong decoupling isn't achievable with organism params alone
(the strong independent lever for a_rate is the environmental changeover delay /
patch separation, exp009). So the demonstration tunes in aligned directions:

```
                a_rate                  a_amt
           target soft stoch   |   target soft stoch
  baseline    --  0.46 0.56    |     --   0.65 0.97
  tune-up   0.50  0.50 0.66    |   0.67   0.67 1.06
  tune-down 0.38  0.38 0.36    |   0.56   0.56 0.86
```

The surrogate hits its targets almost exactly; the stochastic engine moves the same
direction (a_rate 0.36 < 0.56 < 0.66; a_amt 0.86 < 0.97 < 1.06 -- both ordered, both
transfers PASS). The surrogate reads systematically lower than the stochastic engine
(the relaxation softens the choice) -- a monotone compression, not a disagreement.

Honest takeaways: (1) for emergent quantities of long stochastic rollouts, the forward
surrogate is the asset, not its gradient; (2) "autodiff is the enabler" needs revising
-- the enabler is the differentiable *forward* model, searched derivative-free, with
autodiff itself deferred to a truncated-backprop (TBPTT) attempt if a gradient method
is ever wanted; (3) on this prep the matching sensitivities are only weakly tunable by
organism parameters and are coupled -- a genuine result, not a fitting failure.
Validation: 6 new tests (test_matching_diff, test_fit); reproduce baseline 27/27
(exp023 added, exp001-022 + demos show 0 drift).

---

## 2026-06-02 (cont.) — Decoupling rate and amount sensitivity (exp024)

exp023 left the two matching sensitivities positively COUPLED: the organism's
discriminability levers (temperature, approach_gain, beta) move a_rate and a_amt
together, so only aligned targets were reachable. Picked up the decoupling thread.

**First, a falsified premise (worth recording).** The plan was to use patch separation
(the changeover-delay lever, exp009) to control a_rate independently. A probe across
separations D=2..14 killed that idea: in the stochastic engine COD raises BOTH
sensitivities (a_rate 0.06->1.06 AND a_amt 0.16->~1.0 as D grows). exp009 had only ever
measured rate vs COD, so the fact that COD also drives amount sensitivity is a NEW
result. Mechanistically obvious in hindsight: separation is a *discriminability* knob,
the same role beta (cue-tuning width) plays -- they are redundant, not orthogonal. (Also
the surrogate only tracks COD up to D~6 then turns over, so it wouldn't have transferred
at large D anyway.) Conclusion: no discriminability lever can decouple the two.

**The fix: an asymmetric lever.** a_rate is driven by how reinforcement *frequency*
maps to value; a_amt by how *magnitude* does. So a knob on the magnitude->value map
decouples them. Added MatchConfig.amount_exponent (rho): the learned-value teaching
signal uses amount**rho instead of amount (utility curvature of reinforcer magnitude --
standard behavioral economics, objective). Because a_rate is measured at equal amounts
(amount=1 -> amount**rho=1 for any rho), rho leaves a_rate EXACTLY untouched and scales
a_amt. The probe is textbook-clean (1500-step soft / 400x4000 stochastic):

```
 rho  soft_rate soft_amt | stoch_rate stoch_amt
 0.4    0.463    0.292   |   0.561     0.502
 1.0    0.463    0.649   |   0.561     0.970
 2.0    0.463    0.942   |   0.561     1.311
```

a_rate is flat to 3 decimals across rho in BOTH engines; a_amt scales monotonically and
transfers. rho=1.0 reproduces the linear-amount baseline exactly (guarded so existing
experiments don't drift -- verified 0 drift on exp008/011/012/013/014/023).

**exp024 demonstration.** With beta setting a_rate and rho setting a_amt, fit two
CROSSING targets (free = temperature/approach_gain/beta/amount_exponent), impossible on
the coupled manifold:
```
                  a_rate (t/soft/stoch)   a_amt (t/soft/stoch)
  baseline          --  0.46 0.56          --  0.65 0.97
  A rate^/amt_v    0.48 0.48 0.59         0.35 0.35 0.61
  B rate_v/amt^    0.40 0.40 0.39         0.85 0.85 1.20
```
Surrogate hits both targets exactly; stochastic transfers and CROSSES -- A has higher
a_rate but lower a_amt than B (decoupling checks both OK). Fitted rho = 0.49 (A) vs 1.74
(B) carries the amount difference; beta = 6.4 (A) vs 4.6 (B) the rate difference. So the
two matching sensitivities are now independently tunable, with the new dimension
(magnitude utility curvature) the orthogonal lever.

Validation: 2 new tests (rho orthogonality in the surrogate; rho fits a low-amount
target). reproduce baseline re-captured with exp024 (exp001-023 + demos 0 drift).
Open: do a_prob / a_delay need their own asymmetric levers too?

---

## 2026-06-02 (cont.) — Probability and delay decoupling (exp025): the pattern generalizes

Extended the decoupling work to the last two concatenated-matching dimensions. Probed
the stochastic engine first (programmed-ratio regression, so absolute a_rate reads
lower than exp008's obtained-ratio 0.56 -- structure is what matters):

```
config        a_rate   a_amt  a_prob a_delay
baseline       0.204   0.964   0.759   0.610
beta=9         0.332   1.118   0.807   0.686   <- beta scales ALL four
rho=1.6        0.204   1.213   0.759   0.610   <- amount_exponent: a_amt only
delay_k=1.0    0.204   0.964   0.759   0.642   <- delay_k: a_delay only
delay_k=0.2    0.204   0.964   0.759   0.453
```

Clean structure: the discriminability levers (beta, COD, ~temperature) scale all four
sensitivities (the overall-sensitivity anchor, read off a_rate), and each GRADED
dimension has its own orthogonal curvature lever:
  amount -> amount_exponent (rho)        value ~ amount^rho
  delay  -> delay_k                      hyperbolic discount steepness  (already existed!)
  prob   -> probability_exponent (sigma) reinforcement gated on prob^sigma  (NEW)
So delay was already decouplable; only probability needed a new knob. sigma is nonlinear
probability weighting (prospect-theory flavor): contacts reinforce with effective prob
prob^sigma. Each lever leaves the other three sensitivities untouched because their
sweeps hold that dimension neutral (amount=1, prob=1, delay=0 -> 1^x=1, discount(0)=1);
sigma==1.0 / rho==1.0 are guarded so existing experiments stay bit-identical.

Surrogate extended to carry prob/delay schedules + the sigma/delay_k levers
(soft_sensitivities_all returns all four; soft_sensitivities keeps the cheaper rate+amt
pair for exp023/024). Verified in the surrogate: sigma=0.5->1.7 moves soft a_prob
0.15->0.52 with rate/amt/delay byte-identical; delay_k moves a_delay only.

exp025 fits two CROSSING (a_prob, a_delay) targets with free = beta/sigma/delay_k
(fit_dims, a dict-targeted Nelder-Mead over soft_sensitivities_all):
```
                a_prob (t/soft/stoch)    a_delay (t/soft/stoch)
  baseline        --  0.37 0.78           --  0.37 0.61
  A prob^/delay_v 0.50 0.50 0.93         0.22 0.22 0.45
  B prob_v/delay^ 0.15 0.14 0.50         0.34 0.37 0.64
```
Surrogate hits targets; stochastic transfers and CROSSES (a_prob A>B, a_delay A<B).
Fitted sigma = 1.44 (A) vs 0.49 (B), delay_k = 0.15 (A) vs 0.66 (B) carry the two
dimensions independently. So all four generalized-matching-law sensitivities are now
independently tunable -- rate as the frequency anchor, amount/probability/delay each
with its own utility-curvature exponent.

Validation: 3 new tests (sigma & delay_k orthogonality; fit_dims sigma ordering), 49
pass; ruff clean; reproduce baseline re-captured with exp025 (exp001-024 + demos 0
drift -- the rho==1/sigma==1 guards hold).

---

## 2026-06-02 (cont.) — Dual excitatory/inhibitory extinction (Konorski/Bouton)

Added a new opt-in learning rule, learning_model="dual_exc_inhib"
(learning.DualExcitatoryInhibitory), to capture what single-weight Rescorla-Wagner
cannot: post-extinction return of responding. The RW rule erases the weight on
omission; the dual rule keeps a separate excitation w+ (preserved) and inhibition w-
(grows on omission). The net the force reads is w+ - gate*w-, written into
atom.history_weights each step, so forces.py is untouched. Three new state dicts on
BehavioralAtom (w_plus, w_minus, w_minus_ctx); empty/zero by default so the other rules
and all existing experiments are unaffected.

Mechanism (per drive atom, per channel, eligibility e, intensity x):
- Reinforced: w+ += lr_acq*e*x*(lam - w+); w- relaxes toward 0 at inhibition_relax_rate
  (0.1 default, > acquisition rate -- inhibition is labile, Bouton).
- Omission: w+ UNCHANGED; w- += inhibition_rate*e*x*(lam - w-); a context tag EMAs
  toward the current context.
- Passive: w- *= (1 - inhibition_passive_decay) every step (off-contact too).
- Readout each step: gate = exp(-context_beta*|context_now - context_learned|) if
  context_gating else 1; net = clip(w+ - gate*w-). Both w+ and w- clipped to bounds
  (the same overshoot guard RW uses -- a large eligibility*intensity step diverged to
  NaN without it; found via seed 9).

A scalar "context" was threaded env -> obs -> rule (gridworld reset option + obs key +
observation_space; organism passes obs["context"] to update via a new context= kwarg
that RW/Linear ignore). The net is refreshed EVERY step at the current context so the
gate tracks context changes immediately (renewal is expressed on approach, before any
new contact), and so passive decay shows up off-contact.

Three demos (population, plot_dual_components overlays w+/w-/net with phase vlines):
- run_reacquisition_demo (dual vs RW control, slow acquisition rate for resolution):
  dual reacquires net>=0.5 in ~0.47 lives vs ~1.98 original and vs RW's ~0.88.
- run_spontaneous_recovery_demo (passive_decay=0.02): SAME arena throughout, food
  withheld during the rest interval via a new env flag food_present=False (distinct from
  food_reinforces=False = food-present-but-unrewarded extinction). Recovery is driven
  purely by the PASSAGE OF TIME (passive w- decay), context held constant and gating off
  -- mechanically distinct from renewal, NOT a context change (caught in review: an
  earlier version relocated food during rest, which read as a context swap; replaced with
  a true food-free interval in the identical arena). net 0.42 (end-ext) -> 1.00 (end-rest)
  -> 0.48 (re-extinguishing at test); w+ flat at 1.0; w- 0.58 -> 0.00; 0 rest contacts.
- run_renewal_demo (ABA vs ABB, context_gating): first test-life net ABA 0.45 (renewed
  in the acquisition context) vs ABB 0.00 (stays suppressed in the extinction context).

Honest caveat: renewal and SR are clean in the NET (the learned association the force
reads). Raw food-contact counts are floored by incidental proximity (food sits near the
start), so they don't discriminate motivation here -- the net is the reported readout.
The defining figures show w+ flat through extinction while w- rises and the net falls,
then the net recovering (rest decay) / renewing (context switch) -- the signature of
inhibition-on-top-of-preserved-excitation.

Validation: 5 new unit tests (extinction preserves w+/grows w-; net recovers under
passive decay; reacquisition faster than acquisition; context gate enables renewal; RW
path untouched), 54 tests pass; ruff clean; reproduce baseline 32/32 with exp001-025 +
prior demos byte-identical (rule fully opt-in).

---

## 2026-06-03 — Resistance to change, PREE, and emergent resurgence (operant chamber)

Three decremental-learning phenomena, all built in the vectorized operant chamber
(`chamber.py`) rather than the gridworld/`learning.py` rule, because all three are
operant response-rate effects and resurgence in particular needs concurrent two-response
CHOICE, which only the chamber has (`run_concurrent_chamber` precedent). They share one
idea: matching/softmax choice over response values plus the extinction dynamics. New
`ChamberConfig` fields (all opt-in, defaults leave existing runs untouched):
`associability_rule`/`ph_eta`/`ph_init`/`ph_floor` (Pearce-Hall), `value_rule`/
`inhib_rate`/`inhib_relax`/`inhib_passive_decay` (dual). exp026-028; +4 chamber tests
(58 total); reproduce baseline recaptured 35/35 (exp001-025 + prior demos byte-identical;
the only library change is an ADDITIVE `value` key on `run_multiple_schedule`'s return).

### Item 1 — behavioral momentum as mass-modulated decay (exp026)

`run_multiple_schedule` already had the mechanism: a Pavlovian context->reinforcer value
(rate-graded "mass") that divides the per-step, TIME-BASED value decay, `dv = -(value_
extinction/mass)*v`. exp022 could only show momentum under SATIATION, not extinction,
because it read resistance from the press RATE, where a saturating emission function +
shared energy-deficit floor confound proportion-of-baseline with baseline height (the
rich component, higher on the logistic, falls more in proportion -> spurious ANTI-momentum;
reproduced here: rich 0.428 vs lean 0.558 retained). 

Fix: read resistance from the VALUE the mass actually protects, with small motivation so
the value (not the deficit floor) carries responding. Because the decay is multiplicative,
sessions-for-value-to-reach-25%-of-baseline is SCALE-FREE -> a clean control: at
`momentum_mass_gain=0` rich and lean reach criterion in the SAME number of sessions (2=2,
no momentum); at gain=3 rich resists more (6 vs 5 sessions). Added an additive `value`
key to `run_multiple_schedule`'s return for this readout. Effect is real but modest
(ctx-mass ratio ~2:1, and the mass itself decays slowly during extinction).

### Item 2 — partial-reinforcement extinction effect from Pearce-Hall (exp027, `run_pree`)

Single response, CRF (p=1) vs PRF (p=0.25) training then extinction. Pearce-Hall
associability: effective rate scaled by a per-organism alpha that EMAs toward recent
|prediction error|. After PRF an omission is partly expected (alpha low) -> slow
extinction; after CRF the first omission is maximally surprising (alpha spikes) -> fast.

The crucial design choice for a CLEAN control: extinction decay is TIME-BASED (per step),
not press-contingent. A first attempt with press-contingent decay showed PREE even under
the 'fixed' control — an artifact, because (a) vigorous responding self-extinguishes
faster and (b) PRF's lower baseline + emission floor inflate its proportion-retained.
With time-based decay the value's extinction rate constant is response-rate- and
baseline-independent, so 'fixed' gives CRF and PRF the IDENTICAL sessions-to-criterion
(7=7) — no PREE — and any PREE under 'pearce_hall' is associability alone: CRF 16 vs PRF
23 sessions, with the diagnostic alpha-at-extinction-onset CRF 0.81 (surprised) vs PRF
0.48 (expected). Readout is on value (baseline-free); the rate figure is illustrative.

### Item 3 — resurgence WITHOUT coding resurgence-as-choice (exp028, `run_resurgence`)

Per the steer: get resurgence to EMERGE, not implement Shahan & Craig's model. Three-phase
concurrent chamber, one of {R1, R2, background "other" (fixed value = Herrnstein R_e)}
emitted per step by softmax (= matching). Phase 1 reinforce R1; phase 2 extinguish R1 +
reinforce R2; phase 3 extinguish both. Resurgence (R1 recovering in phase 3 from its
phase-2 suppressed level) is computed NOWHERE — it falls out of softmax reallocation when
R2's reinforcement is removed. Key realization: the procedure is SYMMETRIC (R2 gets phase-2
training as R1 got phase-1), so R1 and R2 correctly converge to PARITY at test; resurgence
is the RISE of R1, not R1 exceeding R2 (an earlier "R1>R2@test" criterion was wrong-headed).

The control nails causation: with `control_reinforce_r2=True` (R2 stays reinforced in
phase 3), R1 does NOT recover (+0.13 -> -0.00) — it is REMOVAL of the alternative's
reinforcement (choice reallocation), not time or disinhibition, that drives recovery.

How much latent R1 strength survives phase 2 is set by the SAME mechanisms from items 1-2
(nothing resurgence-specific):
- single value (RW), gain 0: R1 -> background floor in phase 2 (endP2 0.02); bare choice
  reallocation, resurgence +0.13 to parity.
- + momentum mass (gain 8): training-history mass (slow-decaying reinforcement trace,
  `mass_grow`/`mass_decay`, mirroring the slow `ctx` in `run_multiple_schedule`) slows R1's
  phase-2 decay -> LARGER resurgence (+0.19). (Required making the mass trace persistent;
  an EMA-toward-current-reinforcement washed out within the long phase.)
- dual exc/inhib (vectorized port of `learning.DualExcitatoryInhibitory`): omission grows a
  separate inhibition and PRESERVES R1's excitation -> R1 stays far less suppressed in
  phase 2 (endP2 0.20 vs 0.02) and resurges from a higher floor (0.20 -> 0.33).

The single-rule figure is textbook: R1 high / R2 floor (P1), crossover (P2), R1 climbs back
as R2 collapses (P3). Resurgence-as-choice, emergent.

### Open / honest scope
- Momentum-under-extinction effect size is modest (mass ratio bound); a dedicated prep
  training rich/lean to EQUAL baseline value (mass the only difference) would sharpen it.
- Pearce-Hall lives only in the chamber value world; porting it to the foraging
  `LearningRule` (gridworld) is straightforward future work but not done.
- Resurgence uses the one-response-per-step allocation measure (+ background option), the
  natural matching framing; a free-operant rate version is a possible refinement.

---

## 2026-06-03 (cont.) — Resurgence model-mimicry study (4th mechanism: Resurgence as Choice)

Context: behavioral momentum's account of resurgence (the augmented BMT model, Shahan &
Sweeney 2011) was rebuked by Craig & Shahan (2016) for mispredicting reinforcement-rate
effects; Shahan & Craig (2017) replaced it with Resurgence as Choice. The interesting
meta-point: MANY mechanisms produce resurgence, so the phenomenon underdetermines the
process. New side study spun out under studies/resurgence_mechanisms/.

Added a 4th, FORMAL mechanism to chamber.run_resurgence: value_rule="rac" (Resurgence as
Choice). Distinct from the other three on BOTH axes -- value is a temporally-weighted
(leaky-integrated) reinforcement tally (vr <- (1-1/rac_tau)*vr + rac_bump*reinforced), and
allocation is power-law MATCHING over relative value (not softmax over a local delta-rule
value). Resurgence needs NO preserved target strength: when the alternative's reinforcement
stops, its integrated value decays and the target's RELATIVE value recovers. The recovery
is transient and its visibility depends on rac_tau vs phase length (had to set tau ~ phase/5,
i.e. 500 vs 2500, so the rise completes within phase 3; tau too long -> R2 decays too slowly,
target never recovers within the phase -- itself a genuine RaC time-scale signature). Tuned:
tau=500, bump=0.04, sensitivity=1.0, floor=0.1 -> endP2 0.03, P3 peak 0.24, resurgence +0.20,
control +0.00. +1 unit test (59 total). exp028 untouched (rac is a new branch; single/dual
paths byte-identical, baseline still valid).

studies/resurgence_mechanisms/compare_mechanisms.py runs all four through the IDENTICAL
preparation:
- mimicry_four_mechanisms.png: all four reproduce the canonical curve (resurgence +0.14 to
  +0.26), differing only in SHAPE (momentum overshoots; dual suppressed only to a high floor;
  RaC smooth/transient). The basic result cannot discriminate them.
- dissociation_reinforcement_rate.png: TWO parametric sweeps.
  * ALTERNATIVE rate (phase 2): all four rise -> NOT diagnostic (and matches the data).
  * TARGET rate (phase 1): ONLY momentum rises (0.24->0.27); local/dual/RaC flat. This is
    the Craig & Shahan (2016) dissociation -- the simulated version of why momentum was
    rejected (it makes resurgence depend on target reinforcement history; the others don't).

README.md is the full writeup: model-mimicry thesis, each mechanism's process/benefits/
drawbacks, the diagnostic-experiment catalog (target-rate isolates momentum; context/renewal
and retention-interval/spontaneous-recovery isolate the dual/inhibition account; multi-
alternative matching and time-scale isolate RaC; local choice is the parsimonious null), the
methodological moral (the canonical curve is not evidence for any one process; the decisive
tests are about WHAT IS RETAINED, not how much), limitations, and references. The study is a
studies/ artifact -- deterministic and runnable but deliberately NOT in the reproduce baseline.

---

## 2026-06-03 (cont.) — Reinforcement/punishment asymmetry (both preparations)

Next ToDo thread (user-picked): the punishment/reinforcement-asymmetry consequence models +
a model-selection study, shown in BOTH the concurrent chamber (choice allocation) and the
open foraging world (survival), per the user's "need to see results in both preparations".

Chamber (chamber.run_punishment_choice): concurrent M-alternative choice, each response on
its own reinforcement VI AND punishment VI; reinforcement and punishment each train a leaky-
integrated value (vr, vp). Three accounts of how a punisher maps to choice:
  subtractive  (de Villiers 1980):  score = vr - pun_c*vp  (cancels own reinforcement)
  competitive  (Deluty 1976):       score_i = vr_i + pun_c*sum_{j!=i} vp_j  (boosts competitors)
  concatenated (Critchfield/Klapes): B_i ~ vr_i^a_r * vp_i^(-a_p)  (separate sensitivities)
All three SUPPRESS the punished response (mimicry). The discriminating result is the de
Villiers vs Deluty DISSOCIATION: punishing the target at a fixed rate and varying the
ALTERNATIVE's reinforcement, the log-odds suppression rises with alternative richness for
subtractive (+0.91 -> +2.58) but FALLS for competitive (+1.59 -> +0.68) -- opposite slopes,
their historic debate, and the punishment analogue of the resurgence target-rate dissociation.
Concatenated recovers a_p log-linearly (set 0.5/1.0/1.5 -> ~0.74/1.52/2.35, R^2~=1.0),
separable from reinforcement. exp029 + study figures.

METHOD CAVEAT (reported): fitting the GML on OBTAINED punishment is confounded by response
feedback -- a heavily suppressed response is rarely emitted, so it collects FEWER punishers
and the obtained-rate axis can invert (a spurious NEGATIVE a_p appeared for the subtractive
model). Suppression curves use SCHEDULED rate / log-odds.

Foraging (consequence.py ConsequenceModel): danger = punisher. Subtractive scales the aversive
teaching signal by c=punishment_weight (avoidance trained c times more than approach);
ConcatenatedAsymmetric exposes reinf_/punish_sensitivity. Demo (studies/.../compare_models.py):
food up column 5, danger off to the side as an avoidable obstacle; sweeping c traces a clean
approach-avoidance gradient -- learned avoidance 0.55 -> 3.57, food 5.7 -> 0.3/life -- and at high
c the organism OVER-AVOIDS the danger guarding the food and starves (asymmetry maladaptive when
overtuned). Layout matters: danger directly ON the single path floors food intake (degenerate);
off-path + local sensor range (5) gives the graded tradeoff.

NOTE on architecture: only the SUBTRACTIVE account fits both worlds. CompetitiveSuppression and
the concatenated law are between-response CHOICE accounts; the event->energy ConsequenceModel
interface cannot express them, so they live only in the chamber. InjuryHealing (the one genuine
embodied ConsequenceModel) deferred.

Validation: +6 tests (run_punishment_choice suppression/dissociation/a_p; Subtractive/
Concatenated/dispatch) -> 65 pass; ruff clean; reproduce baseline recaptured 36/36 (exp029
added; exp001-028 + demos byte-identical -- all changes additive/opt-in).

---

## 2026-06-03 (cont.) — Phase 5 day/night ambient sun: feature done, risk-sensitivity does NOT emerge (and why)

Added a global ambient sun to the gridworld env: L(t) = 0.5*(1 - cos(2*pi*(t mod
steps_per_day)/steps_per_day)) in [0,1] (0 = midnight, 1 = noon),
gridworld.ambient_light. Opt-in (config.day_night, default False -> byte-identical;
run_demo --check clean). When on, light GRADES PERCEPTION, not the physical consequences:
sensed danger = danger_true*(danger_detect_floor + (1-floor)*L) and food
visibility/regrowth scale by (food_light_floor + (1-floor)*L), while actual danger
contact and food intake still use TRUE proximity. New obs key "ambient_light"; logged in
info. 4 env tests; ruff clean.

GOAL was "risk-sensitive foraging: a starving organism accepts night risk." It does NOT
emerge, and the reason is instructive (two parts):

1. TUNING / mechanism. danger_detect_floor is behaviorally INERT -- foraging is
   byte-identical across floors 0.1/0.5/1.0 (day 200, night 160, to the integer). The
   deficit-scaled approach drive (motiv_strength*deficit^2, up to ~2.0) dwarfs the innate
   avoidance (sensitivity*intensity ~= 1.0*small), and the consummatory atom holds the
   organism at the patch, so gating how well it PERCEIVES the danger changes nothing. I
   also under-set the harm: danger_energy_loss=0.05 in the Phase 5 runs = 1 feeding-step =
   trivial (default 0.15 = 3 feeding-steps = 15% of reserve). An earlier 1.93-vs-1.17
   night/day reading was sampling noise across different agent counts; with matched seeds
   the floors are identical.

2. CONCEPTUAL (the deeper reason). Our "risk" is STATIONARY and DETERMINISTIC: a fixed
   location, constant magnitude, contact = 1.0/0.0 (no probability, no variance). Only the
   DETECTABILITY varies with light. But risk-sensitive foraging (Caraco 1980; Stephens &
   Krebs) is about sensitivity to the VARIANCE of outcomes, governed by the energy-budget
   rule (risk-prone below the requirement, risk-averse above). We never had "risk" in that
   sense -- we had a deterministic hazard whose perceptibility changes (an information
   manipulation, not a variance one). The phenomenon cannot emerge because outcome variance
   is not in the model.

What DOES emerge from the feature is a food-visibility foraging tendency (food dim at night
-> less night foraging), but it is extreme (night ~ 0 when the floor is low) and trivial,
so it is not committed as a demo.

PATH FORWARD (for genuine risk-sensitivity): introduce PROBABILISTIC / variance risk -- a
risky patch where predation strikes with probability p and removes a large chunk of energy
(high variance) vs a safe patch with a low steady return (matched mean), and test the
energy-budget rule (risk-prone when starving). This is a patch-CHOICE prep (cf. the
matching/chamber machinery), not the single fixed-danger gridworld. Day/night could then
modulate either the predation probability or its detectability. The env sun feature is
committed and reusable; the risk model is what needs redesigning.

---

## 2026-06-04 — Risk-sensitive foraging done RIGHT: the energy-budget rule (Caraco)

Follow-up to the Phase 5 finding. The reason risk-sensitive foraging didn't emerge there
was conceptual: a stationary DETERMINISTIC hazard has no variance, and risk-sensitive
foraging (Caraco 1980; Stephens & Krebs) is about sensitivity to OUTCOME VARIANCE under the
energy-budget rule. Built the proper version (chamber.run_risk_choice) and it works cleanly.

A concurrent choice between a SAFE option (constant outcome) and a RISKY option (variable,
MATCHED MEAN), with energy dynamics + a real death boundary (E<=0 fatal). The organism
chooses by softmax over the EXPECTED SURVIVAL UTILITY of each option at its CURRENT energy:
U(E) = logistic((E - e_req)/width) ~ P(survive). U is CONVEX below the requirement, CONCAVE
above -> by Jensen the risky option's mean-preserving spread is favored when starving
(risk-prone) and disfavored when fed (risk-averse). The preference reverses at e_req -- the
energy-budget rule, emergent from survival-utility maximization (nothing codes "gamble when
hungry"). util_shape="linear" is the risk-neutral CONTROL.

Two preparations (exp030, studies/risk_sensitivity/), both matched to mean +0.05:
- reward variance:    SAFE +0.05 sure;  RISKY 0 or +0.10 (p=0.5).   reversal +0.31
- predation variance: SAFE lean +0.05;  RISKY rich +0.0875 but predation strike p=0.2
                      costs -0.10.                                  reversal +0.54
Linear control flat at 0.50 in both -> the reversal is the survival-utility curvature, not
the schedules. P(risky) below vs above e_req: reward 0.65/0.35, predation 0.83/0.29.

Tuning that mattered: the Jensen gap scales with the risky option's VARIANCE relative to the
utility WIDTH, so the reward spread has to span a meaningful fraction of util_width (small
spreads gave a ~0.04 swing; spread 0.10 vs width 0.08 gives the full reversal). And the
economy must be near break-even (mean reward ~= cost) so energy DIFFUSES around e_req and
organisms visit BOTH sides; otherwise energy piles at capacity (all well-fed, all risk-
averse) and the reversal is never sampled. The effect is sharpest near e_req where U is most
curved and washes out at the energy extremes where U saturates (indifference) -- correct: risk
attitude matters most near the survival margin.

Caveat: the organism is GIVEN the option distributions (an innate state-dependent rule, as in
Caraco) rather than learning them; e_req/width are parameters, not derived from the horizon.
A fuller model would learn the distributions and compute U from the actual survival problem.

Validation: +2 chamber tests (70 total); ruff clean; reproduce baseline recaptured with
exp030; exp001-029 + demos unchanged.

---

## 2026-06-04 (cont.) — Risk-sensitivity DERIVED from survival (not imposed), via a DP

Methodological correction to exp030. There the choice reads option values through an
ASSUMED survival sigmoid U(E)=logistic((E-e_req)/width); it is that curvature (and the free
e_req) that produces the energy-budget rule -- we installed the result in the utility. New
module behavioral_md.survival derives the rule from the bare dynamics already in the engine
(energy reserve + metabolic drain + hard death at E<=0) with survival as the ONLY objective.

survival_dp: backward DP over one DAY (forage: safe vs risky) + NIGHT (forced fast, no
choice). P(survive the cycle) from every (energy, time-of-day); the optimal risk policy is
read straight off it. The "requirement" is NOT a parameter -- it is night_steps*metabolism
(the reserve needed at dusk to outlast the fast), emergent.

Result (studies/risk_sensitivity/first_principles.py, survival_policy_map.png): the optimal
policy is a risk-prone BAND, both edges emergent:
- upper edge (safe already secures survival -> risk-averse above) rises through the day
  from ~0.19 toward the night requirement R=0.72 at dusk.
- lower edge = RUIN (even gambling can't reach R -> doomed either way -> indifferent); near
  zero early, rises late in the day as recovery time runs out.

This answers the objection to exp030's figure (why doesn't P(risky) keep rising the more
negative things get?). The OPTIMAL policy gambles for EVERY energy inside the band, not a
moderate slice; risk-proneness is bounded BELOW only by genuine ruin (an emergent edge), not
by a saturating utility. exp030's bump is what a BOUNDED-RATIONAL chooser (softmax over a
saturating value) produces -- survival.softmax_policy reproduces it -- but the band and its
bounds come from survival, not from e_req. The day/night structure (Phase 5) supplies the
principled requirement: the sun feature finally earns its keep.

Time-dependence is the emergent richness: early day you gamble only if very low (lots of
recovery time); late day the whole band rises toward R and the ruin floor climbs. A fine
sawtooth on the edges is a real "reachability comb" (the discrete 0/2s gamble reaches R only
via whole numbers of lucky draws); a smoother outcome distribution fills it -- the envelope
is the point.

Next: have the behavioral chamber choose by softmax over the DP-DERIVED survival values
(first-principles behavioral model) instead of the imposed sigmoid; then learn the
distributions / let selection shape the rule.

Validation: behavioral_md.survival + 5 tests (75 total); ruff clean. Study artifact (DP is
deterministic; not added to the reproduce baseline).

---

## 2026-06-04 (cont.) — The energy-budget rule EVOLVES (selection, no utility/DP/learning)

Capstone of the risk arc. exp030 IMPOSED a survival utility; survival_dp DERIVED the policy;
simulate_survival_choice EXECUTED it behaviorally. survival.evolve_risk_policy removes even
the planner: a population carries a heritable state-dependent risk trait theta(t)=a+b*(t/day)
(gamble when E<theta), forages day/night, dies at E<=0, and survivors reproduce with Gaussian
mutation on (a,b). Selection is the bare survival dynamics -- nothing rewards "gamble when
hungry".

The rule emerges anyway (studies/risk_sensitivity/evolution.py, evolved_policy.png): the
time-of-day slope b evolves from ~0 to ~0.31 within ~10 generations, so the evolved threshold
RISES through the day, and it CONVERGES on the DP-optimal threshold at DUSK (evolved 0.73 vs
DP 0.70) where the decision matters most and selection is strongest. It is looser at DAWN
(0.44 vs 0.19) where there is all day to recover and the choice barely affects survival ->
selection sculpts the policy most precisely exactly where it matters. Final survival ~0.26 (real
selection pressure). Economy: day net on safe (24*0.02=0.48) < night drain R=0.72, so safe
alone is insufficient and gambling is forced when behind -- the energy-budget condition.

Arc complete: IMPOSED (exp030) -> DERIVED (DP) -> EXECUTED (behavioral) -> EVOLVED, converging
on the same energy-budget rule from progressively fewer assumptions. This is the first
EVOLUTIONARY result in the engine (opens the long-parked evolution thread). Genome is a linear
threshold (hence the dawn slack vs the curved DP optimum); next is within-life LEARNING of the
distributions.

Validation: evolve_risk_policy + 1 test (76 total); ruff clean. Study artifact; not in the
reproduce baseline.

---

## 2026-06-04 (cont.) — Within-life LEARNING of the option distributions (arc complete)

The last gap: the organism no longer KNOWS the option distributions. survival.
simulate_learning_choice has each organism start ignorant (one pseudo-observation of each
option at the grand mean -> initially indifferent), estimate each option's outcome
distribution from observed outcomes, re-plan the survival DP on its CURRENT estimate each
cycle, forage day/night, and respawn on death keeping what it learned. Nothing tells it which
option is risky.

Exploration was the catch: with both options estimated as point masses at the mean, the DP
always picks safe -> never samples risky -> never learns (locked in). Fixed with decaying
epsilon-greedy (0.45 -> 0.05) so it samples the risky option and discovers the variance.

Result (studies/risk_sensitivity/learning.py, within_life_learning.png): GAMBLE RECALL (of the
states where the true optimum gambles, the fraction the learned plan also gambles) goes 0.00
(cycle 0, ignorant) -> 0.98 (cycle 1) -> 1.00 (cycle 3+); estimated risky variance climbs to
true (0.0025) by ~cycle 3; survival improves 0.50 -> 0.80 as it learns and exploration anneals
(learning has adaptive value). The learned threshold lands on the DP optimum and -- because
learning recovers the actual distributions -- tracks the CURVED DP threshold across the whole
day, MORE faithfully than the evolved linear genome (which only approximated it).

NOTE on metric: full-grid policy accuracy is insensitive (the risk-prone band is a small
fraction of states, so "never gamble" already scores 0.83); gamble RECALL on the true-gamble
states isolates the learning (0 -> 1).

ARC COMPLETE -- the same energy-budget rule at five levels, most assumed to least: IMPOSED
(exp030, a utility) -> DERIVED (DP, only the dynamics) -> EXECUTED (behavioral, the DP values)
-> EVOLVED (selection, the support) -> LEARNED (within life, nothing; discovered from
experience). The learner is model-BASED (plans on its learned model); a model-FREE learner
(survival values from living/dying, no planning) is the strictest remaining version.

Validation: simulate_learning_choice + 1 test (77 total); ruff clean. Study artifact.

---

## 2026-06-04 (cont.) — Model-FREE survival learner (the strictest version)

The within-life learner (learning.py) still PLANS (learns distributions, runs the DP). The
strictest version removes both model and planner: survival.simulate_model_free_choice has each
organism hold a tabular Q[energy_bin, time_of_day, action] and learn it by MONTE-CARLO from the
bare survival signal -- after each day/night cycle, every visited (state, action) is nudged
toward 1 if it survived and 0 if it died (alpha=0.1, decaying epsilon-greedy 0.3->0.05, random
start energy each cycle for state coverage). No model of the distributions, no planning;
survival values learned directly from living and dying.

Result (studies/risk_sensitivity/model_free.py, model_free.png): the AGGREGATE greedy policy's
gamble recall climbs 0.73 -> 0.93 over ~100 cycles and its threshold lands on the DP optimum
(mean |diff| ~= 0.03) across the whole day. The energy-budget rule emerges from nothing but
reinforcement. It is the cost of assuming the least -- markedly slower/noisier than model-based
(recall ~1.0 in ~3 cycles vs ~0.9 in ~100). Honest metric note: INDIVIDUAL Monte-Carlo Q-tables
are high-variance (per-organism recall plateaus ~0.5); the population-MEAN value function is the
clean readout (pooled experience), so the recall reported is on the aggregate greedy policy.

ARC FULLY COMPLETE -- the same energy-budget rule at six levels, most assumed to least: IMPOSED
(a utility) -> DERIVED (DP, only the dynamics) -> EXECUTED (DP values) -> EVOLVED (selection, the
support) -> LEARNED MODEL-BASED (estimate + plan) -> LEARNED MODEL-FREE (reinforcement on the
survival signal, no model, no planning). A behavioral regularity (Caraco's energy-budget rule)
shown to be what survival IMPLIES, SELECTS FOR, and TEACHES -- by planning or by reinforcement.

Validation: simulate_model_free_choice + 1 test (78 total); ruff clean. Study artifact.

---

## 2026-06-04 (cont.) — The day/night SUN as the source of risk (variance, not a hazard)

Closes the Phase 5 loop. Phase 5's sun could not produce risk-sensitivity because its "danger"
was a stationary DETERMINISTIC hazard, and risk-sensitivity is about VARIANCE. So make the sun
set the VARIANCE of foraging: steady in full light (midday), erratic in the dark (dawn/dusk),
mean matched. survival.sun_variance_risky builds per-step risky outcomes {mean-w(t), mean+w(t)}
with spread w(t) = w_min..w_max tracking darkness; survival_dp_timevarying solves the DP with
time-varying option distributions. Control = constant variance with the SAME average spread
(sqrt(mean w^2)), so only the TIMING differs.

Result (studies/risk_sensitivity/sun_variance.py, sun_variance.png): high-variance foraging is a
LIFELINE near the deadline and a LIABILITY far from it. The RUIN edge (lowest reserve from which
gambling can still reach the night requirement) under the sun vs constant:
  t=20 (dusk, getting dark):  sun 0.26  vs  constant 0.38   -> sun LOWER (lifeline)
  t=0  (dawn, dark, far off):  sun 0.10  vs  constant 0.04   -> sun HIGHER (liability)
Near the deadline a big-variance gamble can bridge the gap to R (only hope) -> ruin edge drops;
far from it the downside has all day to bite -> ruin edge rises. The day/night sun puts the high
variance exactly at DUSK, when a behind-schedule organism most needs the gamble. The Phase 5
"starving organism accepts night risk" intuition, finally emerging for the right reason
(variance, not a deterministic hazard). New: survival_dp_timevarying + sun_variance_risky.

Validation: +3 tests (timevarying==constant when fixed; spread peaks in dark, mean matched; dusk
lifeline), 81 total; ruff clean. Study artifact. (Built in the parent repo; the standalone
risk-sensitive-foraging carve-out predates this and stays the frozen six-level version.)
