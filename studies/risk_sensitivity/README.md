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

This completes the arc — **imposed** (exp030) → **derived** (the DP) → **executed** (the
behavioral loop) → **evolved** — converging on the same energy-budget rule each time, from
progressively fewer assumptions.

## Limitations

- The DP gives the *normative* optimum (full knowledge of the option distributions); the
  evolved genome is a *linear* threshold (hence the dawn slack against the curved DP optimum).
  The remaining step is *within-life learning* of the distributions from experience.
- The band edges show a fine sawtooth — a real "reachability comb" from the discrete
  (0 / 2s) gamble: the requirement can only be reached via whole numbers of lucky draws. A
  smoother outcome distribution fills it in; the envelope is what matters.

## References

- Caraco, T. (1980). On foraging time allocation in a stochastic environment. *Ecology*,
  61, 119–128.
- Caraco, T., Martindale, S., & Whittam, T. S. (1980). An empirical demonstration of risk-
  sensitive foraging preferences. *Animal Behaviour*, 28, 820–830.
- Stephens, D. W., & Krebs, J. R. (1986). *Foraging Theory*. Princeton University Press.
- Kacelnik, A., & Bateson, M. (1996). Risky theories -- the effects of variance on foraging
  decisions. *American Zoologist*, 36, 402–434.
