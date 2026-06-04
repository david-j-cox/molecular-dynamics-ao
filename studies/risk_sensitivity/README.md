# Risk-sensitive foraging and the energy-budget rule

This study is the proper version of a phenomenon the Phase 5 day/night work could not
produce. There, "risk" was a **stationary, deterministic hazard** — a fixed danger at a
fixed place with a constant cost, whose *detectability* varied with the sun. But
risk-sensitive foraging (Caraco, 1980; Stephens & Krebs, 1986) is not about hazards; it is
about sensitivity to the **variance** of outcomes, governed by the **energy-budget rule**.
A deterministic hazard has no variance, so the phenomenon could not emerge (see
`lab_notebook` 2026-06-03/04). Here risk is genuine outcome variance, and it does.

Reproduce: `python experiments/exp030_risk_sensitivity.py`

## The preparation

A concurrent choice (`chamber.run_risk_choice`) between a **SAFE** option (a constant
outcome) and a **RISKY** option (a variable outcome with the **same mean**), with energy
dynamics and a real death boundary (E ≤ 0 is fatal). Two flavors:

- **Reward variance** — SAFE: +0.05 for sure; RISKY: 0 or +0.10 at p = 0.5 (the classic
  Caraco junco design).
- **Predation variance** — SAFE: lean +0.05; RISKY: rich +0.0875, but a predation strike
  (p = 0.2) costs −0.10. Both matched to mean +0.05.

## The mechanism (why it emerges, not coded)

Each step the organism chooses by a softmax over the **expected survival utility** of each
option at its **current energy** E:

```
U(E) = logistic((E - e_req) / width)   ~   P(survive | energy E)
EU(option) = sum_outcomes  p · U(E + delta - cost)
```

`U` is a survival sigmoid: **convex below** the requirement `e_req`, **concave above**. By
Jensen's inequality, a mean-preserving spread (the risky option) raises expected utility
where `U` is convex and lowers it where `U` is concave. So the organism is **risk-prone
when starving** (convex region) and **risk-averse when well-fed** (concave region), and the
preference **reverses at the energy requirement** — the energy-budget rule. Nothing in the
code says "gamble when hungry"; it falls out of maximizing a survival-shaped utility, which
is the principled objective in an energy world where death is real.

## Result

`figures/energy_budget_rule.png`. P(choose risky) vs current energy, both preparations,
survival utility vs. a linear-utility control:

```
                     P(risky) below e_req   above e_req   reversal
reward variance              0.65              0.35         +0.31
predation variance           0.83              0.29         +0.54
linear control (both)        0.50              0.50          0.00
```

The reversal occurs exactly at the requirement and is sharpest in the curved region around
it (far above/below, `U` saturates and the options look equal — indifference). The
**linear-utility control is flat at 0.50**: with matched means and a risk-neutral utility
there is no energy dependence, so the reversal is attributable to the survival-utility
curvature alone, not to the schedules.

## Reading it against Phase 5

| | Phase 5 gridworld | this study |
|---|---|---|
| "risk" | a stationary deterministic hazard (detectability varies) | genuine outcome variance |
| what varies with state | nothing about the hazard | the *value* of variance, via U's curvature at E |
| energy-budget rule | cannot arise (no variance) | emerges, with a clean reversal |

The lesson the two together teach: **risk-sensitivity is about variance, and the state
dependence is about the curvature of the survival utility** — not about how detectable a
fixed danger is.

## Deriving the rule instead of imposing it (`first_principles.py`)

The account above has a circularity worth admitting: the survival sigmoid `U(E)` is
*assumed*, and it is exactly its curvature that produces the energy-budget rule. We did not
derive risk-sensitivity; we installed it in the utility. The first-principles version
removes the utility entirely and derives the rule from the bare dynamics already in the
engine — an energy reserve, a metabolic drain, and a hard death boundary — with **survival
as the only objective** (`behavioral_md.survival.survival_dp`).

A single day/night cycle: by **day** the organism forages (safe vs. risky); by **night** it
cannot forage and simply burns `metabolism` each step, dying if its reserve hits zero.
Backward dynamic programming gives the exact probability of surviving the cycle from every
(energy, time-of-day), and the optimal risk policy is read straight off it. The "requirement"
is not a parameter — it is `night_steps × metabolism`, the reserve you must hold at dusk to
outlast the fast.

`figures/survival_policy_map.png`: the optimal policy is a **risk-prone band**, and *both*
edges are emergent:

- **upper edge** — where the safe option already secures survival (risk-averse above); it
  rises through the day from ≈0.19 toward the night requirement R = 0.72 at dusk.
- **lower edge** — **ruin**, where even the gamble cannot reach R (doomed either way →
  indifferent); it sits near zero early but rises late in the day as recovery time runs out.

This is richer and more correct than the imposed sigmoid, and it answers the obvious
objection to the `exp030` figure — *why doesn't P(risky) keep rising the more negative
things get?* The optimal policy gambles for **every** energy inside the band, not just a
moderate slice; risk-proneness is bounded *below* only by genuine **ruin** (an emergent
edge), not by a utility that happens to saturate. The bump in `exp030` is what a *bounded-
rational* chooser (softmax over a saturating value) produces — `survival.softmax_policy`
reproduces it — but the bound and the band themselves come from survival, not from `e_req`.

## Closing the loop, and the requirement as the one axis (`parametric.py`)

Two follow-ups confirm the derivation is behavior, not just a table.

**Closing the loop** (`figures/closing_the_loop.png`). A population that actually lives and
dies, choosing by a softmax over the **DP-derived** survival values (no imposed utility;
`survival.simulate_survival_choice`), reproduces the energy-budget band in its *realized*
policy, and its realized survival-by-starting-energy matches the planner's `V(E)`. The
first-principles policy is executable behavior that achieves the predicted survival.

**The requirement is the axis** (`figures/requirement_sweep.png`). `R = night_steps ×
metabolism` is the one knob that sets where the band sits. Sweeping the night length, the
dusk safe-suffices edge tracks `R` almost exactly (it sits one forage step below it):

```
R (= night × metabolism):  0.24  0.42  0.60  0.78  0.96
dusk safe-suffices edge:    0.22  0.40  0.58  0.76  0.94
```

A heavier survival burden pushes the whole risk-prone band to higher reserves. The
requirement is derived from the dynamics, and it is the right parametric axis.

## It evolves (`evolution.py`)

The capstone removes the planner too. A population carries a **heritable** state-dependent
risk trait — a threshold linear in time of day, `theta(t) = a + b·(t/day)` — and gambles when
its reserve is below `theta(t)`. Organisms forage through day/night cycles and die at E ≤ 0;
the survivors reproduce with mutation (`survival.evolve_risk_policy`). Selection is the bare
survival dynamics — **nothing rewards "gamble when hungry"**, there is no utility, no DP, and
no learning rule.

The rule emerges anyway (`figures/evolved_policy.png`): the time-of-day slope `b` evolves
positive (from ≈0 to ≈0.31) within ~10 generations, so the evolved threshold **rises through
the day**, and it **converges on the DP-optimal threshold at dusk** (evolved 0.73 vs DP 0.70)
— where the decision is most consequential and selection is strongest. It is looser at dawn
(0.44 vs 0.19), where there is all day to recover and the choice barely affects survival. So
risk-sensitivity is not installed; it is what survival **selects for**, and selection sculpts
it most precisely exactly where it matters.

## It is learned within life (`learning.py`)

The last assumption was that the organism *knows* the option distributions. Here it does not:
it starts ignorant, estimates each option's outcome distribution from what it observes, plans
survival (the DP) on its current estimate, and forages day/night (respawning on death, keeping
what it has learned; `survival.simulate_learning_choice`). Nothing tells it which option is
risky.

It finds out (`figures/within_life_learning.png`). With a little annealing exploration it
samples the risky option, its estimated variance climbs to the truth within a few cycles, and
its planned policy comes to gamble **exactly where the energy-budget rule prescribes** (gamble
recall 0 → 1); survival improves as it learns. Because learning recovers the *actual*
distributions, the learned threshold tracks the curved DP optimum across the whole day — more
faithfully than the evolved *linear* genome, which only approximated it.

## Strictest of all: model-free (`model_free.py`)

The within-life learner still *plans* (it learns the distributions and runs the DP). The
model-free learner does neither: each organism holds a tabular value `Q[energy_bin,
time_of_day, action]` and learns it by Monte-Carlo from the bare survival signal — after each
cycle, every visited (state, action) is nudged toward 1 if it survived and 0 if it died
(`survival.simulate_model_free_choice`). No model, no planning; survival values learned
directly from living and dying.

The rule emerges anyway (`figures/model_free.png`): the aggregate greedy policy's gamble recall
climbs to ~0.93 and its threshold lands on the DP optimum (mean |diff| ≈ 0.03) across the whole
day. It is the cost of assuming the least — markedly slower and noisier than the model-based
learner (recall ~1.0 in ~3 cycles vs ~0.9 in ~100, and individual Monte-Carlo Q-tables stay
high-variance) — but it gets there from nothing but reinforcement.

## The arc

The same energy-budget rule appears at every level of explanation, from most assumed to least:

| stage | source of the rule | what is assumed |
|---|---|---|
| **imposed** (`exp030`) | a survival utility's curvature | the utility + `e_req` |
| **derived** (the DP) | survival-maximizing dynamic programming | only the death dynamics |
| **executed** (the behavioral loop) | softmax over DP-derived survival values | the DP's values |
| **evolved** (`evolution.py`) | selection on a heritable trait | the option support |
| **learned, model-based** (`learning.py`) | within-life learning + planning | discovered, then planned |
| **learned, model-free** (`model_free.py`) | reinforcement on the survival signal | *nothing* — no model, no planning |

## The sun as the source of risk (`sun_variance.py`)

A synthesis with the day/night work elsewhere in the engine (Phase 5). That sun could not
produce risk-sensitivity because its "danger" was a *stationary, deterministic* hazard — and
risk-sensitivity is about variance. So let the sun set the **variance** instead: steady foraging
in full light (midday), erratic in the dark (dawn/dusk), mean matched
(`survival.sun_variance_risky` + `survival_dp_timevarying`). Compared to a control with the same
*average* variance spread evenly, so only the timing differs.

High-variance foraging turns out to be a **lifeline near the deadline and a liability far from
it**. The ruin edge — the lowest reserve from which gambling can still reach the night
requirement — drops near dusk under the sun (sun 0.26 vs. constant 0.38: a desperate forager is
saved by the dark's high variance) but rises at dawn (0.10 vs. 0.04: the downside has all day to
bite). The day/night cycle puts the high variance exactly at dusk, when a behind-schedule
organism most needs the gamble — the "starving organism accepts night risk" intuition, emerging
for the right reason (variance, not a fixed hazard).

### Realized as behavior (`behavioral_sun.py`)

The figure above is the planner's *table* (the DP ruin edge). Does it describe what organisms
actually do, and whether they actually live? A population is dropped into the dark dusk
(day-step 20) holding a range of reserves, forages the remaining day-steps under the optimal
policy drawing **real** intake from the time-varying distributions, and then must outlast the
night (`survival.simulate_dusk_survival`). `figures/behavioral_sun.png`:

```
                              sun (dark dusk)   constant (same avg.)
reserve for >=50% survival         0.54               0.58
peak survival advantage         +0.25 (at reserve 0.56)
advantage once reserve safe        0.00  (>= 0.66)
```

The ruin-edge lifeline is realized as **who actually lives**: behind at dusk, an organism
survives the night from a *lower* reserve under the dark's high variance, and is up to ~25
percentage points more likely to live across the whole desperate band. The advantage is exactly
**zero** once the reserve already outlasts the night — variance helps the desperate, never the
comfortable. (The clean, decision-relevant readout is survival-from-how-little; an aggregate
P(gamble)-by-time-of-day readout is muddier, dominated by the *dawn* liability — desperate
organisms avoiding the high-variance dawn gamble — rather than a sharp dusk peak.)

## Richer worlds: continuous outcomes and skew (`richer_worlds.py`)

Every result above used a two-point gamble `{mean ± w}`. Enough to derive the rule, but it costs
a discretization artifact (the *reachability comb* — the requirement is reachable only by whole
numbers of identical lucky draws, so the band edges show a sawtooth) and it cannot ask about the
*shape* of risk beyond variance. A continuous distribution (`survival.skewed_outcomes` — a
standardized, warped normal with chosen mean, std, and skew) fixes both.

**Continuous outcomes remove the comb** (left panel of `figures/richer_worlds.png`). The
safe-suffices edge over the day is a sawtooth for the two-point gamble (roughness 0.060) and
perfectly smooth for a continuous gamble of the same mean and variance (roughness 0.000), landing
on the same dusk requirement. The energy-budget band is a property of survival, not of the grid.

**The energy-budget rule extends to the third moment** (right panel). At *fixed mean and
variance* — where mean-variance risk theory predicts indifference — the survival-optimal policy is
not skew-indifferent, and its preference **reverses at the requirement**, exactly as the variance
preference does:

```
gamble's survival edge over safe (×10⁻³), averaged over the regime:
                          left-skew (disaster)   right-skew (lottery)
below R (desperate)             +2.18                  −1.94        prefers NEGATIVE skew
above R (comfortable)           −6.28                  −0.53        prefers POSITIVE skew
```

Below R, with time to build the buffer, frequent small gains (negative skew) climb toward R more
reliably than the all-or-nothing lottery. Above R, already safe, the only threat is the rare
catastrophe, so the policy avoids negative skew. The two regime curves run opposite directions and
cross near the symmetric gamble — a genuine reversal, not the flat response mean-variance predicts.
(This is the molar average; in the narrow near-deadline corner the positive-skew lottery can still
win — but that is the *variance* lifeline of the sun-variance study, and here variance is held
fixed, isolating the pure skew effect.)

## Multi-patch foraging: risk-sensitive patch choice and MVT (`multi_patch.py`)

The arc so far was a binary safe-vs-risky choice. A real forager faces a *menu* of patches and
*depleting* patches it must decide when to leave. Both fall out of the same survival objective
(`survival.survival_dp_patches`, `survival.survival_dp_depleting`) and both connect to the
rate-maximizing — and therefore risk-*neutral* — marginal-value-theorem work in the JAX engine
(`experiments/exp020_patch_leaving_mvt.py`). Survival makes patch foraging risk-sensitive.

**Patch choice is a three-way energy-budget rule** (left panel of `figures/multi_patch.png`). A
menu of a low-variance *safe* patch, a high-mean *rich* patch, and a high-variance *wild* patch;
the survival-optimal choice over (energy, time-of-day):

```
                       share of (time × energy) grid
safe  (low variance)            0.29     above R: comfortable, hold steady
rich  (high mean)               0.53     below R with time: maximize rate to climb toward R
wild  (high variance)           0.18     below R near the deadline: gamble on variance
```

Survival **interpolates between rate-maximizing (rich) and variance-seeking (wild)** depending on
how much time is left to reach the requirement — the optimal-foraging regime and the risk-prone
regime are two faces of one policy.

**The giving-up rule is finite-horizon** (right panel). A depleting patch with a travel cost to
reach a fresh one. The organism abandons depleted patches readily through the day (`P(leave)` → 1.0
mid-day: classic MVT relocation), but **stops leaving near dusk**, and the leaving deadline tracks
the travel cost:

```
travel cost (steps):     2     4     6
last step it still leaves:  27    24    22       (day = 30; cutoff ≈ day − travel)
```

Once fewer than ~travel steps remain there is no time to reach and exploit a fresh patch before the
night fast. Infinite-horizon MVT, with its single time-invariant giving-up density, cannot express
this; survival's deadline produces it. Both deviations are survival *refinements* of MVT, not
contradictions — away from the deadline and the ruin edge, survival reduces to the risk-neutral
rate-maximizing MVT (the rich-patch regime, the mid-day leaving plateau). MVT is the not-desperate
limit.

## Limitations

- Model-free convergence is slow and the aggregate readout pools experience across the
  population; individual Q-tables remain high-variance (the price of no model and no planning).
  The energy economy is deliberately minimal. (The day/night sun modulating the variance,
  continuous/skewed outcomes, and multiple patches — choice and depleting/MVT — are now done; see
  above.) The remaining open thread is evolving the multi-patch/depleting policies as heritable
  traits (`survival.evolve_risk_policy` currently evolves only the binary threshold).
- The *upper* (safe-suffices) band edge's sawtooth was a two-point reachability comb and is
  removed by a continuous outcome distribution (`richer_worlds.py`). The *lower* (ruin) edge still
  jitters — it is a near-indifference region (both options ≈ 0 survival), not a discretization
  artifact, so a smoother distribution does not settle it; the envelope is what matters.

## References

- Caraco, T. (1980). On foraging time allocation in a stochastic environment. *Ecology*,
  61, 119–128.
- Caraco, T., Martindale, S., & Whittam, T. S. (1980). An empirical demonstration of risk-
  sensitive foraging preferences. *Animal Behaviour*, 28, 820–830.
- Stephens, D. W., & Krebs, J. R. (1986). *Foraging Theory*. Princeton University Press.
- Kacelnik, A., & Bateson, M. (1996). Risky theories -- the effects of variance on foraging
  decisions. *American Zoologist*, 36, 402–434.
