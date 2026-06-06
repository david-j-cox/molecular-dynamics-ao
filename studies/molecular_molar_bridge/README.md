# The molecular--molar bridge: why the energy-budget rule does not emerge from per-step atom dynamics

## The finding (exp032)

The energy-budget rule (risk-prone below the energy requirement, risk-averse above; Caraco,
Martindale & Whittam 1980) is reproduced by the *imposed-utility* chamber (exp030), where the choice
is a softmax over an installed survival utility `U(E) = logistic((E - R)/w)`. It is **not** reproduced
by the real atom engine -- force decomposition with the convex motivational gain, damped Verlet
integration, eligibility-gated Rescorla--Wagner learning, and energy/death -- regardless of whether
the per-step learning signal is the energy gain or the single-step survival of the realized draw.

Figure: `outputs/figures/exp032_mechanistic_energy_budget.png`. P(choose risky) vs current energy:

| curve | below R | above R | reversal | reproduces the rule? |
|---|---|---|---|---|
| imposed utility (exp030 reference) | 0.70 | 0.30 | **+0.21** | yes (built in via `U(E)`) |
| mechanistic, energy teaching | 0.17 | 0.22 | -0.05 | no (flat/low) |
| mechanistic, single-step survival teaching | 0.50 | 0.49 | +0.01 | no (flat) |

Matched-mean (variance-only) options, harsh economy (per-step cost = mean intake) so the low-energy
regime is visited and lethal. The imposed reference shows the clean reversal; neither mechanistic
variant does.

## Why it does not emerge

1. **The convex motivational gain scales both options equally.** `g = mu * deficit^p` multiplies food
   drive, but it is option-independent, so for two matched-mean food options it cancels in the choice.
   It cannot, by construction, create a preference based on *variance*.
2. **Per-step reinforcement is mean-based.** Rescorla--Wagner on matched-mean outcomes drives both
   history weights to the same value, so learning yields no preference. (The small asymmetry that does
   appear runs *toward* safe, an artifact of death/selection, not risk sensitivity.)
3. **Single-step survival is trivial.** At these reserves a single draw almost never crosses zero, so
   "did this outcome keep me alive" is ~always 1: no signal. The survival contingency only bites over
   a *horizon* (surviving the night), not a step.

## The molecular--molar reading

This is a clean instance of the molecular vs molar debate in behavior analysis (the molecular view:
behavior is controlled by momentary, local contingencies; the molar view: by aggregate relations over
extended periods -- Baum 1973; **Baum 2002**, *From molecular to molar: a paradigm shift in behavior
analysis*, JEAB 78:95-116, doi:10.1901/jeab.2002.78-95; Rachlin's teleological behaviorism).

The atom engine is **molecular**: forces are momentary, integration is local in time, and credit is
assigned per step. It reproduces molar *aggregates* -- matching, demand, momentum -- because those are
sums/ratios of momentary contingencies. But the energy-budget rule is **molar in a stronger sense**:
its contingency is an extended-horizon terminal event (survive the period, defined by the death
boundary `E<=0` after the overnight fast). A purely molecular, mean-based learner cannot see that
contingency, so the rule does not emerge. This is the molecular/molar scaling gap made concrete in a
mechanistic model.

We already know what supplies the missing piece: the model-free survival learner
(`survival.simulate_model_free_choice`) recovers the rule from *cycle* survival via a Monte-Carlo
backup (survived the period -> 1, died -> 0). The missing ingredient is **horizon credit assignment**,
not anything specific to risk.

## The design problem: restore the molar influence without hard-coding the answer

The imposed-utility chamber works precisely because it *installs the answer*: `U(E)` is the survival
value function, so the curvature that produces the reversal is put in by hand. That is
question-begging. The task is to make the molecular engine sensitive to the molar contingency without
installing the rule. We adopt three litmus tests for "not hard-coded":

- **Inject only environmental facts, never the policy.** Legitimate molar inputs are real
  consequences the environment actually imposes: death at `E<=0` over a period, and the passage of
  time within the period (time-to-deadline, already present as the day/night cycle). Illegitimate
  inputs are `U(E)`, a requirement `R`, or "be risk-prone when hungry."
- **Use a general mechanism, not one tuned to this result.** Temporal credit assignment and selection
  are general; a bespoke variance term is not.
- **It must generalize.** Shift the requirement or the option distributions and the policy must track
  without retuning (parameter recovery); the same mechanism should also produce other molar effects.
  If it only works for this one setup, it is a disguised hard-code.

### Candidate mechanisms (all pass the litmus tests)

1. **Within-life horizon-survival credit assignment (TD / eligibility).** Make the terminal signal the
   bare survival of the *period* (1/0) and let a general temporal-difference / eligibility mechanism
   distribute it to the molecular choices that led there. The eligibility trace already in the engine
   is exactly the molecular *implementation* of molar credit; it is currently too short and is tied to
   per-step consequences. Lengthen the horizon and make the terminal signal survival. Only death (a
   fact) is injected; the rule emerges. This is the molecular implementation of molar control.

2. **Time-to-deadline as a sensed channel.** The molar state variable the rule depends on is
   time-to-dusk. Expose it (the env already has an ambient-light cycle) so the molecular policy can
   condition on `(energy, time)`. This does not install the policy; it supplies a real environmental
   state, and the `(reserve, time)` policy (threshold rising toward dusk) can then emerge.

3. **Selection across lives (evolution on atom parameters).** Let many atom organisms with heritable
   variation forage and die; survivors reproduce. The molar fact enters *only* through differential
   survival -- nothing about value is installed at all. This is the least question-begging route; we
   have already evolved the rule at the genome/DP level (`survival.evolve_risk_policy`), and the atom
   substrate is the next target.

### Recommendation

Build (1) + (2) within-life -- horizon-survival credit assignment on the atom history weights, with
time-to-deadline as a sensed channel -- and validate non-question-beggingness by **generalization**:
move the requirement and the option distributions and confirm the emergent threshold tracks them with
no retuning. Use (3) as an independent confirmation that selection alone, installing nothing, yields
the same disposition. The claim to defend is the molecular--molar reconciliation: molar control is
*implemented* molecularly by temporally-extended credit assignment (and by selection), with only real
consequences injected.

## Result (exp033): contact-as-only-currency fails; a daily survival signal succeeds

Two candidate bridges were tested on a day/night economy with energy and death, state-conditioned by
(energy bin x day-phase), measured at the **emergent** requirement R = night_cost*night_steps = 0.30
(NOT a free parameter -- the reserve needed to outlast the overnight fast). Figure:
`outputs/figures/exp033_multilevel_reinforcement.png`.

| mechanism | below R | above R | reversal | rule? |
|---|---|---|---|---|
| imposed utility (reference) | 0.63 | 0.42 | +0.20 | yes (built in) |
| **A**: contact is the only currency (reinforcer-as-currency; survival never represented, death = truncation) | 0.29 | 0.37 | -0.07 | **no** |
| **B**: daily survival signal scales down (alive-at-dawn = 1 / died = 0, credited onto the day's choices via eligibility) | 0.61 | 0.45 | +0.17 | **yes** |

- **A fails.** With contact as the only currency, the safe option's higher contact *rate* (it feeds
  every step; risky only on its good draw) dominates the value, and "followed by contacts" saturates
  without a sharp differential at the death boundary. Long eligibility horizons (decay up to 0.995)
  do not rescue it. So survival cannot be left fully implicit in the reinforcement stream.
- **B succeeds.** When surviving the cycle is a bare daily *fact* (alive at dawn = 1, died = 0) that
  is cashed out onto the day's choices through the eligibility trace, the rule emerges and closely
  tracks the imposed reference below R -- but with an **emergent requirement** (R = the overnight
  drain) and **no utility function anywhere**. This is your "survival has value at a daily level that
  scales down into the many-per-day food contacts": survival is a signal, but only a 0/1 environmental
  fact at the day scale, not an installed `U(E)`.

So the answer to "molar influence without hard-coding the answer": the molar consequence (survival)
must enter as an explicit signal, but only as a **bare fact at the period scale**, scaled down onto
the molecular choices by temporal credit -- not as a value function. The convexity (risk-prone below,
averse above) and the requirement itself are then emergent, not installed. Pure reinforcer-currency
(A) is too weak; an imposed utility (exp030) is too strong; the daily survival fact (B) is the
minimal faithful bridge.

## Validation of B (exp034, exp035)

**Generalization -- the non-hard-coding proof (exp034).** With the learner held fixed (lr, decay,
temperature, bins), shifting the night length (hence R = night_cost*night_steps) moves B's emergent
risk-prone/averse crossover to the new requirement on its own:

| night steps | R | emergent crossover |
|---|---|---|
| 4 | 0.20 | 0.25 |
| 8 | 0.40 | 0.45 |
| 12 | 0.60 | 0.65 |

corr(crossover, R) = **1.000**, mean offset = 0.05 (one energy bin, discretization). Nothing about R
or the policy was installed; only the environmental fact (a longer night needs a bigger dusk reserve)
changed, and the policy followed. This is the test that distinguishes a real mechanism from a
disguised hard-code, and B passes it. Figure: `outputs/figures/exp034_bridge_generalization.png`.

**The upper boundary -- risk-aversion above R (exp035).** With a single (starvation) boundary,
survival saturates at 1 above R, so B is only weakly averse when fed. Adding a second death source
-- predation above an upper reserve x_r (heavier = slower/more visible; McNamara & Houston 1990),
another bare environmental fact -- sharpens aversion in the well-fed band: P(risky) in [R, x_r] drops
from 0.36 (single boundary) to **0.27** (two boundaries), while risk-proneness below R is unchanged
(0.61). This is the twin-threshold energy-budget rule (prone below R, averse above), with aversion
driven by a real cost of being fat, not a utility. (Above x_r the policy is noisy: organisms there
are being eaten regardless of choice, so the bin is rarely and unstably visited.) Figure:
`outputs/figures/exp035_upper_boundary.png`.

## Atom-substrate port (exp036): learning ports, expression needs timescale separation

Putting approach B onto the engine's own dynamics -- the choice produced by the force decomposition
-> `verlet_update` -> softmax, with energy-state-conditioned history weights credited by the
period-scale survival signal -- splits into two outcomes:

- **The learning ports cleanly.** The state-conditioned history weights acquire the rule:
  W(risky)-W(safe) = +0.06 below R and -0.06 just above. The survival-credit mechanism works on the
  atom history weights exactly as in the tabular prototype.
- **The expression does not, in a per-step choice.** P(risky) stays flat (~0.50). The damped-Verlet
  activation needs \emph{sustained} drive to build magnitude (a constant drive settles to a
  weight-ordered activation, e.g.\ 0.72 vs 0.60 -> softmax 0.92/0.08), but in an abstract one-shot
  choice the energy state changes every step, so the activation never reflects the current state and
  the softmax smears to indifference.

This isolates a second molar->molecular problem distinct from credit assignment:

1. **Credit** (molar outcome -> molecular weights): solved by survival-eligibility (B). Works here.
2. **Expression** (molecular dynamics enacting a molar state-dependent policy): requires
   **timescale separation** -- the molecular dynamics must be fast relative to the molar state, or
   the organism must commit to an option over a stretch where the state is ~constant. The spatial
   foraging loop supplies exactly this (approach a patch over many steps; reserve changes slowly); an
   abstract per-step choice does not. So the faithful atom-engine demonstration must be **spatial**.

## Spatial loop (exp037): the rule survives the atom dynamics, but spatial travel inverts it

A 1D spatial forager (SAFE patch at x=-1, RISKY at x=+1), with approach B's survival-credit learning
on energy-state-conditioned weights and a **drive-readout** emission (orient on the pull difference,
the steady-state of the movement atom -- exp036 showed the transient integrated activation
under-builds). Two results:

- **The rule survives the atom dynamics.** With intake per step (no travel-cost concentration), the
  atom-dynamics organism reproduces the energy-budget rule: P(feed risky) reversal **+0.24**
  (risk-prone below R), tracking the imposed reference. So the port works -- given the correct
  drive/steady-state readout, the distributed mechanism + survival credit express the rule.
- **Spatial travel INVERTS it.** When intake comes only on patch contact (real foraging), the rule
  flips: reversal **-0.19** (risk-*averse* below R). Confirmed causal: switching only the
  travel/no-travel flag flips the sign. The reason is structural and new: spatial travel
  **concentrates the cost of a failed gamble** -- a risky 0-draw means a whole trip's travel cost
  spent for nothing, a large one-encounter energy drop that is lethal when already low. So when low,
  the organism takes the reliable immediate intake (safe). The classic energy-budget rule assumes
  per-step matched-mean options with a deadline as the sole forcing; concentrated per-encounter cost
  breaks that assumption and reverses the prediction.

This is a genuine structural finding: **risk-proneness-when-low does not survive spatial foraging
with travel costs** -- the molar->molecular bridge (survival credit) is intact, but the realized
policy depends on the foraging structure (per-step vs per-encounter cost).

## Status

- exp032: negative baseline (the rule does not emerge from per-step atom dynamics).
- exp033: A vs B -- **A insufficient, B reproduces the rule with an emergent requirement.**
- exp034: **B generalizes** -- the reversal tracks R with no retuning (corr 1.0).
- exp035: **a predation upper boundary yields risk-aversion above R** (twin-threshold rule).
- exp036: **B's learning ports to the atom history weights, but a per-step choice cannot express it**
  (timescale mismatch). Credit solved; expression needs timescale separation.
- exp037: **the atom dynamics + survival credit reproduce the rule (drive-readout emission), but
  spatial travel-cost concentration inverts it** -- a structural limit on the energy-budget rule.
- exp045: **approach B in a real 2D arena** (faithful self-contained prep; engine primitives, no
  shared-core refactor). A 2D forager (SAFE + RISKY patches, day/night requirement, energy/death +
  predation) learns from period-scale survival credit conditioned on energy x time. With the
  drive-readout emission the **energy-budget rule is reproduced** (reversal +0.23); spatial travel
  **weakens** it (+0.12; milder than exp037's full 1D inversion at these 2D params); the raw
  `verlet_update`+softmax emission **under-builds** (+0.05) -- the exp036 dt^2 limit persists even
  with spatial commitment, so the drive readout (movement-atom steady state) is the faithful emission.
- Open: the travel-cost result deserves its own writeup (when does spatial risk-sensitivity match vs
  invert the non-spatial rule?); a literal shared-core `BehavioralFieldEnv` + `SurvivalCredit`
  integration is deferred as higher-risk. See `docs/lab_notebook.md`.

## References
- Baum, W. M. (1973). The correlation-based law of effect. *JEAB* 20, 137-153. doi:10.1901/jeab.1973.20-137
- Baum, W. M. (2002). From molecular to molar: a paradigm shift in behavior analysis. *JEAB* 78, 95-116. doi:10.1901/jeab.2002.78-95
- Caraco, T., Martindale, S., & Whittam, T. S. (1980). An empirical demonstration of risk-sensitive foraging preferences. *Anim. Behav.* 28, 820-830. doi:10.1016/S0003-3472(80)80142-4
- Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning* 3, 9-44. doi:10.1007/BF00115009
