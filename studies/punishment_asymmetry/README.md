# The reinforcement/punishment asymmetry, in two preparations

Reinforcement and punishment are not mirror images. A punisher does not simply "undo" a
reinforcer; how punishment controls behavior — whether it subtracts value directly,
works by strengthening competing behavior, or carries its own matching sensitivity — has
been debated for fifty years. This study implements the leading accounts in the
behavioral molecular-dynamics engine and asks two things the field asks: do they make
**distinguishable** predictions, and does the asymmetry show up the same way in **choice**
(concurrent schedules) and in **open foraging** (survival)?

The engine has two worlds, and we exercise both:

- **Concurrent choice (operant chamber):** `chamber.run_punishment_choice` — two responses
  concurrently reinforced and (independently) punished. Three allocation accounts of
  punishment, compared.
- **Open foraging (survival):** the pluggable `consequence.ConsequenceModel` — danger is
  the punisher; the subtractive model trains avoidance more strongly than approach, and
  sweeping that asymmetry traces an approach-avoidance gradient.

Reproduce: `python studies/punishment_asymmetry/compare_models.py`
(the chamber quantities are also in `experiments/exp029_punishment_asymmetry.py`).

---

## Preparation 1 — concurrent choice: three accounts of punishment

Both responses are on equal reinforcement VIs; punishment is added on independent VIs.
Reinforcement and punishment each train a temporally-weighted value (`vr`, `vp`), and one
response is emitted per step by matching over a model-specific score:

| account | score for response *i* | the claim |
|---|---|---|
| **subtractive** (de Villiers, 1980) | `vr_i − c·vp_i` | a punisher cancels *c* reinforcers of the **same** response |
| **competitive** (Deluty, 1976) | `vr_i + c·Σ_{j≠i} vp_j` | a punisher strengthens the **competing** responses (relative suppression) |
| **concatenated** (Critchfield/Klapes) | `vr_i^{a_r} · vp_i^{−a_p}` | power-law matching with a **separate** punishment sensitivity `a_p` |

### All three suppress — so the basic result does not discriminate them

Adding punishment to one response reduces allocation to it in every account (the common
phenomenon). They differ in functional form, but suppression-per-se is mimicry.

### The de Villiers vs. Deluty dissociation (the diagnostic)

The fifty-year question is **how** punishment suppresses: by directly subtracting a
response's own value (de Villiers), or by reallocating to competitors (Deluty)? The two
predict **opposite** dependence on the *alternative's* reinforcement rate. Punishing the
target at a fixed rate and varying the alternative's reinforcement, the log-odds
suppression of the target:

```
alternative reinforcement rate (1/VI):  0.05   0.10   0.17   0.25
subtractive (de Villiers):              +0.91  +1.11  +1.59  +2.58    (rises)
competitive (Deluty):                   +1.59  +1.15  +0.86  +0.68    (falls)
```

- **Subtractive rises** with alternative richness: as the alternative gets richer the
  target is emitted (and reinforced) less, so its small reinforcement value is more easily
  pushed below the floor by a fixed subtraction.
- **Competitive falls**: a punisher works by boosting the competitor, and that boost
  matters less when the competitor is already strong.

This opposite-slope diagnostic is the punishment analogue of the resurgence
target-reinforcement dissociation (`studies/resurgence_mechanisms/`): the basic curve
mimics across mechanisms; a *parametric* manipulation pulls them apart.

### Concatenated matching law with punishment

With **both** responses punished and the punishment ratio swept, the emergent
`log(B1/B2) = a_r·log(vr1/vr2) − a_p·log(vp1/vp2) + bias` is log-linear (R² ≈ 1.0) and
recovers a punishment sensitivity that tracks the set `a_p` (set 0.5/1.0/1.5 → recovered
≈ 0.74/1.52/2.35), separable from the reinforcement term — the Critchfield/Klapes result.

**A measurement caveat worth stating.** Fitting on *obtained* punishment is confounded by
response feedback: a heavily suppressed response is rarely emitted, so it collects fewer
punishers and the obtained-punishment axis can invert (we observed a spurious *negative*
`a_p` for the subtractive model). Suppression functions therefore use **scheduled** rate.
This artifact is formalized as its own finding in
[`studies/obtained_rate_confound/`](../obtained_rate_confound/) (the obtained punishment
rate is inverted-U in the scheduled rate, which flips the sign of the fitted sensitivity:
+2.04 on scheduled vs −2.24 on obtained).

---

## Preparation 2 — open foraging: the asymmetry as approach-avoidance

In the survival world, danger is the punisher, routed through the pluggable
`ConsequenceModel`. The **subtractive** model (`consequence_model="subtractive"`) scales
the aversive teaching signal by `c = punishment_weight`, so a punisher trains the
`avoid_danger` drive *c* times more strongly than a reinforcer trains `approach_food`.
(`concatenated_asymmetric` exposes the two sensitivities separately.)

With a food source up the column and a danger off to the side as an *avoidable* obstacle,
sweeping `c` traces a clean approach-avoidance gradient:

```
punishment sensitivity c:   1     2     3     4     5     6
food consumed / life:       5.7   4.3   2.5   2.1   1.2   0.3
learned avoidance weight:   0.55  1.14  1.71  2.52  3.10  3.57
```

Learned avoidance climbs monotonically, food intake falls, and at high punishment
sensitivity the organism **over-avoids** — it gives the danger a wide berth, fails to reach
the food the danger guards, and starves. Excessive punishment sensitivity is maladaptive:
the asymmetry that protects the organism from harm, pushed too far, costs it the resource.

This is the *same* asymmetry parameter seen from the survival side: in the chamber it sets
how punishment shifts allocation; in foraging it sets where the organism draws the line
between food and safety.

---

## Benefits, drawbacks, and what distinguishes the accounts

| account | what it buys | where it strains |
|---|---|---|
| **subtractive** (de Villiers) | one parameter (`c`); a punisher has a fixed "cost in reinforcers"; clean in the energy world (an objective debit) | suppression depends on alternative reinforcement in a way (rising) some data do not show; treats punishment as pure value subtraction |
| **competitive** (Deluty) | punishment needs no special value channel — it is reallocation; predicts the (falling) alternative-reinforcement dependence | requires competitors to absorb the suppression; awkward when there is no good alternative |
| **concatenated** (Critchfield/Klapes) | separates `a_r` and `a_p` empirically; fits concurrent data well; punishment sensitivity is measurable | descriptive, not a process — it says *that* sensitivities differ, not *why*; needs both responses punished |
| **subtractive ConsequenceModel** (foraging) | makes the asymmetry an embodied, survival-relevant quantity; yields over-avoidance/starvation | in the binary-event foraging world it collapses toward "scale the aversive signal", so it cannot, there, distinguish subtraction from competition |

**The discriminating experiments.** As with resurgence, the canonical result (suppression;
approach-avoidance) does not identify the process. The diagnostics are *parametric*:
- **Alternative reinforcement rate** — subtractive vs. competitive predict opposite slopes
  (simulated above). The single cleanest test.
- **Both-responses-punished ratio sweep** — isolates the concatenated `a_p` (and exposes
  the obtained-rate feedback artifact).
- **Remove the alternative** — competitive suppression should weaken when there is no
  competitor to strengthen; subtraction should not. *(proposed)*
- **Reinforcer/punisher devaluation and magnitude** — whether `c` is fixed or scales with
  punisher magnitude (Rasmussen & Newland, 2008). *(proposed)*

---

## Limitations

- These are minimal, transparent implementations, not the published quantitative models;
  the predictions are qualitative and illustrative.
- The chamber `vr`/`vp` are leaky-integrated values; recovered sensitivities carry a fixed
  gain relative to the set values (they track them monotonically, which is the point).
- The foraging gradient's tail (heavy over-avoidance) is high-variance — the organism
  rarely eats, so food counts are noisy there.
- Competitive suppression and the concatenated law are **choice** accounts and live only in
  the chamber; the foraging `ConsequenceModel` interface (event → energy + teaching signal)
  cannot express them, so the foraging side uses the subtractive model. `InjuryHealing`
  (a delayed, embodied punisher with temporary repertoire impairment) is left for later.

## References

- Critchfield, T. S., Paletz, E. M., MacAleese, K. R., & Newland, M. C. (2003).
  Punishment in human choice: Direct or competitive suppression? *Journal of the
  Experimental Analysis of Behavior*, 80, 1–27.
- Deluty, M. Z. (1976). Choice and the rate of punishment in concurrent schedules.
  *Journal of the Experimental Analysis of Behavior*, 25, 75–80.
- de Villiers, P. A. (1980). Toward a quantitative theory of punishment. *Journal of the
  Experimental Analysis of Behavior*, 33, 15–25.
- Klapes, B., Riley, S., & McDowell, J. J (2018/2020). Toward a contemporary quantitative
  model of punishment. *Journal of the Experimental Analysis of Behavior*.
- Rasmussen, E. B., & Newland, M. C. (2008). Asymmetry of reinforcement and punishment in
  human choice. *Journal of the Experimental Analysis of Behavior*, 89, 157–167.
