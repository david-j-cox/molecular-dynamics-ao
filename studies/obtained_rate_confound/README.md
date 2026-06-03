# Obtained punishment rates are endogenous: a sign-flipping fit artifact

A short methodological finding from the punishment-asymmetry work
(`studies/punishment_asymmetry/`), formalized on its own because it is a general
hazard, not a quirk of one model.

## The claim

The generalized matching law for punishment is conventionally fit on **obtained**
reinforcement and punishment rates:

```
log(B1/B2) = a_r * log(R1/R2) - a_p * log(P1/P2) + log c
```

But a punisher is collected **only when the punished response is emitted**, and
punishment suppresses that very response. So the obtained punishment rate is a function
of the behavior it is being used to predict. Fitting allocation on the **obtained**
punishment rate is therefore endogenous, and when suppression is strong it can recover a
punishment sensitivity of the **wrong sign**. The **scheduled** (programmed) rate is
exogenous and recovers the correct value.

## The mechanism

Let `B_i` be allocation to response *i* and `rho_i` its scheduled punishment (arming)
rate. A punisher is delivered on emission, so the obtained rate is

```
P_i(obtained) ≈ B_i * rho_i        =>   log P_i = log B_i + log rho_i
```

Now raise `rho_target` while holding everything else fixed. Two regimes:

- **Graded regime** (low rho): `rho` rising dominates, so `P(obtained)` rises.
- **Collapse regime** (high rho): suppression drives `B_target -> 0` faster than `rho`
  rises, so `P(obtained)` **falls**.

The obtained rate is therefore **inverted-U in the scheduled rate** (`figures/feedback_function.png`).
On the descending limb, obtained punishment and allocation fall *together*, so a
regression of `log(B_target/B_alt)` on `log P(obtained)` has the predictor and the
response mechanically co-moving — the slope turns positive, i.e. the recovered `a_p`
turns **negative**.

## What the engine shows

Two responses equally reinforced (VI 5); the target punished at a rising scheduled rate,
the alternative unpunished (subtractive model, `chamber.run_punishment_choice`):

```
scheduled rate:  0.025  0.040  0.067  0.100  0.143  0.200  0.250
obtained rate:   0.024  0.038  0.057  0.052  0.018  0.014  0.014   <- peaks at 0.067, then collapses
log(Bt/Ba):      -0.24  -0.41  -0.83  -2.01  -3.85  -4.16  -4.23   <- monotone (real suppression)

recovered a_p  on SCHEDULED rate:  +2.04   (correct: suppression)
recovered a_p  on OBTAINED  rate:  -2.24   (WRONG SIGN: the artifact)
```

`figures/fit_scheduled_vs_obtained.png`: against the scheduled rate the suppression is
clean and monotone; against the obtained rate the points **fold back** (arrows trace the
sweep), and the fitted slope flips sign.

## Why this is specific to punishment

Reinforcement has **positive** feedback: obtained reinforcement rises with emission, and
reinforcement raises emission, so obtained reinforcement increases monotonically with the
scheduled rate — the obtained-rate fit is biased in *magnitude* but not in sign.
Punishment has **negative** feedback: it suppresses the emissions that collect it,
producing the inverted-U and the sign reversal. The artifact is uniquely severe for
punishment, and worst near response elimination — exactly the regime clinical and applied
work cares about.

## The fix

- **Use scheduled / programmed rates** (exogenous) as the independent variable, or
- **jointly model the feedback function** `B(rho)` rather than treating obtained rate as
  exogenous, or
- **restrict the fit to the graded (pre-collapse) regime**, and report where the obtained
  rate turns over.

## Limitation

This is a model-generated demonstration, not a re-analysis of data: the engine's
suppression is steep enough to reach the collapse regime within the swept range. The
qualitative result — obtained punishment is endogenous and inverted-U, so its slope can
mislead — is a property of the response-feedback structure, not of the particular
suppression model. The magnitude of the bias in real data depends on how far punishment
pushes responding toward elimination.

## References

- Baum, W. M. (1981). Optimization and the matching law as accounts of instrumental
  behavior. *Journal of the Experimental Analysis of Behavior*, 36, 387–403.
  (programmed vs obtained rates; feedback functions)
- Baum, W. M. (1973). The correlation-based law of effect. *Journal of the Experimental
  Analysis of Behavior*, 20, 137–153.
- Critchfield, T. S., Paletz, E. M., MacAleese, K. R., & Newland, M. C. (2003).
  Punishment in human choice: Direct or competitive suppression? *Journal of the
  Experimental Analysis of Behavior*, 80, 1–27.
- de Villiers, P. A. (1980). Toward a quantitative theory of punishment. *Journal of the
  Experimental Analysis of Behavior*, 33, 15–25.
