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
