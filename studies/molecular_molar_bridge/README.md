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

## Status

- exp032: negative baseline (the rule does not emerge from per-step atom dynamics).
- exp033: A vs B. **A insufficient, B reproduces the rule with an emergent requirement.**
- Next: (1) port B into the real atom engine as a hierarchical operant level (a daily survival
  reinforcer modulating the molecular history weights), since exp033 is a tabular prototype of the
  mechanism; (2) validate non-question-beggingness by **generalization** -- shift the requirement
  (night length) and the option distributions and confirm B's reversal tracks with no retuning;
  (3) add the upper (reproduction/predation) boundary for risk-aversion above. See
  `docs/lab_notebook.md` (2026-06-05).

## References
- Baum, W. M. (1973). The correlation-based law of effect. *JEAB* 20, 137-153. doi:10.1901/jeab.1973.20-137
- Baum, W. M. (2002). From molecular to molar: a paradigm shift in behavior analysis. *JEAB* 78, 95-116. doi:10.1901/jeab.2002.78-95
- Caraco, T., Martindale, S., & Whittam, T. S. (1980). An empirical demonstration of risk-sensitive foraging preferences. *Anim. Behav.* 28, 820-830. doi:10.1016/S0003-3472(80)80142-4
- Sutton, R. S. (1988). Learning to predict by the methods of temporal differences. *Machine Learning* 3, 9-44. doi:10.1007/BF00115009
