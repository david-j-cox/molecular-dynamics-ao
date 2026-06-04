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

## Limitations

- The organism is given the option distributions (an evolved/innate state-dependent rule,
  as in Caraco's account) rather than learning them; the survival sigmoid's `e_req`/`width`
  are parameters, not derived from the horizon. A fuller model would learn the
  distributions and compute `U` from the actual survival problem.
- The reversal is strongest near `e_req` (where curvature is greatest) and washes out at
  the energy extremes where the utility saturates; this is a feature of the sigmoid, and
  matches the intuition that risk attitude matters most near the survival margin.

## References

- Caraco, T. (1980). On foraging time allocation in a stochastic environment. *Ecology*,
  61, 119–128.
- Caraco, T., Martindale, S., & Whittam, T. S. (1980). An empirical demonstration of risk-
  sensitive foraging preferences. *Animal Behaviour*, 28, 820–830.
- Stephens, D. W., & Krebs, J. R. (1986). *Foraging Theory*. Princeton University Press.
- Kacelnik, A., & Bateson, M. (1996). Risky theories -- the effects of variance on foraging
  decisions. *American Zoologist*, 36, 402–434.
