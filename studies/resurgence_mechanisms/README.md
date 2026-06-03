# Four processes, one phenomenon: the mechanisms of resurgence

**Resurgence** is the recovery of a previously reinforced, then extinguished, *target*
response when a more recently reinforced *alternative* response is in turn placed on
extinction. It is a workhorse model of relapse (of operant behavior, of problem behavior
after differential-reinforcement treatment, of drug seeking) and a long-standing
theoretical battleground.

This study makes one point and draws out its consequences: in the behavioral
molecular-dynamics engine, **resurgence is produced by at least four mechanistically
distinct processes**, all running through the *identical* three-phase preparation. The
canonical resurgence curve therefore **underdetermines** the underlying process — a
case of model mimicry. The scientific value is not "we can make resurgence happen" (that
is easy) but: cataloguing the candidate processes, being honest about what each does and
does not buy, and identifying the *manipulations that pull them apart*.

This is a deliberate side study. The mechanisms here are minimal, transparent
implementations in one shared engine (`behavioral_md.chamber.run_resurgence`), not the
published quantitative models; the simulated predictions are qualitative and
illustrative. The aim is conceptual clarity and an experimental roadmap, not a fit to
data.

## Contents

- `compare_mechanisms.py` — runs all four mechanisms through the identical preparation
  (the mimicry figure) and two parametric dissociations (reinforcement-rate sweeps).
- `figures/mimicry_four_mechanisms.png` — all four reproduce the resurgence curve.
- `figures/dissociation_reinforcement_rate.png` — where their predictions diverge.
- `figures/summary.json` — the numbers.

Reproduce: `python studies/resurgence_mechanisms/compare_mechanisms.py`

---

## The preparation

A concurrent operant chamber. Each step the organism emits exactly **one** option — the
target **R1**, the alternative **R2**, or a background "other" behavior maintained by a
fixed extraneous reinforcement source (Herrnstein's *R_e*) — by a choice rule over the
options' current values. Energy/motivation is held constant, so "extinction" withholds
only the response→reinforcer contingency (it is not starvation). Three phases of equal
length:

| phase | R1 (target) | R2 (alternative) |
|---|---|---|
| 1 — training | reinforced (VI) | extinction |
| 2 — alternative reinforcement | extinction | reinforced (VI) |
| 3 — test | extinction | extinction |

**Resurgence** is the rise of R1 responding in phase 3 above its phase-2 suppressed
level. Crucially, *nothing* in the engine computes "resurgence"; it is read off the
emitted-response record. The procedure is symmetric (R2 is trained in phase 2 as R1 was
in phase 1), so R1 and R2 converge toward parity at test — resurgence is the *recovery*
of R1, not R1 overtaking R2.

**The control that isolates the cause.** If R2 stays reinforced in phase 3 instead of
being extinguished (`control_reinforce_r2=True`), allocation does not flow back and
resurgence is abolished in every mechanism. So the recovery is driven by *removal of the
alternative's reinforcement* — choice reallocation — not by the mere passage of time or
by disinhibition. This is the empirical signature (Leitenberg, Rawson, & Mulick, 1975;
Winterbauer & Bouton, 2010) and the anchor for everything below.

---

## The four mechanisms

Each is one line of `value_rule` / parameter choice in the same engine. They differ in
two respects: how a response's *value* changes during extinction, and how value maps to
*choice*.

### 1. Local choice (`value_rule="single"`, `momentum_mass_gain=0`)

**Process.** Each response carries a single value updated by an error-correcting (delta)
rule — toward an asymptote on reinforcement, decaying toward zero on non-reinforced
emissions. Allocation is a **softmax** over the current values plus the fixed extraneous
value (local/molecular matching).

**Why it resurges.** In phase 2, R1's value decays to the floor while R2's grows; in
phase 3, R2's value decays once its reinforcement stops and the softmax reallocates the
freed-up choice probability back toward R1. R1's *own* value is essentially gone — it
returns only to parity with the background.

**Benefits.** Maximally parsimonious; shows resurgence needs almost nothing — two
competing values and a choice rule. Molecular and trial-by-trial. Naturally captures the
alternative-reinforcement-rate effect (a richer R2 suppresses R1 more, so more
reallocation is available).

**Drawbacks.** The target's value is genuinely *erased*, so the account confers no
special status on the target relative to any other once-low response, and offers no
preserved learning to explain the *other* relapse phenomena (renewal, spontaneous
recovery of the same response after a delay, reinstatement, rapid reacquisition). The
softmax temperature and the extraneous-reinforcement floor are doing quiet work.

### 2. Behavioral momentum (`value_rule="single"`, `momentum_mass_gain>0`)

**Process.** A reinforcement-history **mass** — a slowly-decaying Pavlovian
stimulus→reinforcer value, graded by reinforcement *rate* — divides the rate at which
the target's value decays under extinction (Nevin & Grace, 2000). Richer target training
→ more mass → slower phase-2 decay → more latent target strength surviving into the test.

**Benefits.** Unifies resurgence with **resistance to change** and the **partial-
reinforcement extinction effect** (see `exp026`/`exp027` in the main package) and with a
large literature on resistance to disruption. Predicts cross-disruptor generality: a
target's resistance to extinction, satiation/prefeeding, and distraction should all scale
with its reinforcement rate.

**Drawbacks.** This is the **challenged** account. The quantitative momentum model of
resurgence (Shahan & Sweeney, 2011) was shown to **mispredict reinforcement-rate
effects** (Craig & Shahan, 2016), and Shahan & Craig (2017) proposed Resurgence as Choice
in its place. The core problem is visible in our simulation (below): momentum is the only
one of the four whose resurgence **increases with the target's own reinforcement
history**, whereas resurgence empirically is driven far more by the *alternative's*
reinforcement and is not well predicted by target history. Momentum is included here as
the historically important foil, not as the favored account.

### 3. Dual excitatory/inhibitory (`value_rule="dual"`)

**Process.** Extinction does not erase the excitatory association (`w+`); it builds a
*separate*, context-specific **inhibition** (`w-`), and behavior is driven by the net
`w+ − w-` (Konorski, 1967; Bouton, 2004). The target retains a **latent excitatory
strength** through phase 2; resurgence is reallocation *plus* the re-expression of that
preserved excitation as the competing R2 is removed (and as inhibition can passively
decay or be gated by context). This is the same rule used for spontaneous recovery,
renewal, and rapid reacquisition in the main package (`learning.DualExcitatoryInhibitory`).

**Benefits.** Unifies resurgence with the *other* relapse phenomena — **renewal,
spontaneous recovery, reinstatement, rapid reacquisition** — all of which argue that
extinction is new learning, not unlearning (Bouton, 2004; Bouton, Winterbauer, & Todd, 2012). Uniquely
among the four it predicts **context-specificity** (renewal-type effects) and
**time-driven** recovery (spontaneous recovery of the target without any alternative
extinction).

**Drawbacks.** Requires positing inhibition plus context machinery (more structure and
parameters). Its resurgence-specific quantitative predictions for reinforcement-rate
manipulations are less developed than RaC's; in our engine it mimics the choice accounts
on the rate sweeps (its distinctive predictions live in *context* and *time*
manipulations, not reinforcement rate).

### 4. Resurgence as Choice (`value_rule="rac"`)

**Process.** A **molar** account (Shahan & Craig, 2017). Each response's value is a
**temporally-weighted** (leaky-integrated) tally of the reinforcers it has produced, and
allocation **matches** relative value (a power law over R1, R2, and the extraneous
source). There is no response strength and no preserved target association: resurgence is
the recovery of the target's *relative* value when the alternative's integrated
reinforcement **decays** after its reinforcement stops.

**Benefits.** Parsimonious and quantitatively accurate for exactly the reinforcement-rate
effects that sank the momentum model; unifies resurgence with the matching/choice
literature. Predicts a **transient, time-scale-dependent** resurgence: magnitude and
shape depend on the temporal weighting (our simulation: the time constant τ relative to
phase duration). Generates parametric predictions for multiple alternatives, magnitudes,
and session structure.

**Drawbacks.** As a molar choice model it is agnostic about the associative/
representational substrate, so on its own it does not explain renewal, spontaneous
recovery of the *same* response after a delay, reinstatement, or rapid reacquisition —
the phenomena the dual account is built for. The temporal-weighting parameter is
powerful and risks over-flexibility, and treating behavior as pure allocation (no
response strength) is, to some, too abstract.

---

## Why the basic result cannot tell them apart

`figures/mimicry_four_mechanisms.png`: under the identical preparation, **all four
reproduce the canonical resurgence curve** — target high (phase 1), suppressed (phase 2),
recovering (phase 3) — and all four are abolished by the reinforced-alternative control.

```
mechanism               suppressed (end phase 2) -> peak (phase 3)   resurgence
local choice                  0.02  ->  0.16                           +0.14
behavioral momentum           0.02  ->  0.28                           +0.26
dual exc/inhib                0.20  ->  0.34                           +0.14
resurgence as choice          0.03  ->  0.24                           +0.20
```

They differ in *shape* — momentum overshoots sharply; the dual rule is suppressed only to
a high floor (preserved excitation); RaC's transitions are smooth and its resurgence is a
slow, transient hump (the molar temporal integration). Shape is suggestive, but it is
parameter-sensitive and not a clean discriminator. **The phenomenon underdetermines the
process.**

---

## Where they diverge: diagnostic experiments

The way forward is manipulations on which the mechanisms make *different* predictions.
`figures/dissociation_reinforcement_rate.png` runs two of these in the engine; the table
then catalogs the broader set, including manipulations our minimal engine does not yet
implement (marked *proposed*).

### Simulated dissociations (reinforcement rate)

**Vary the ALTERNATIVE's reinforcement rate (phase 2).** All four predict resurgence
**increases** with alternative richness — so this canonical manipulation, though central
to the literature, **does not by itself discriminate** them (and the increase matches the
data: Leitenberg et al., 1975; Sweeney & Shahan, 2013). The mechanisms differ only in
functional form.

**Vary the TARGET's reinforcement rate (phase 1).** Here they split cleanly:

```
mechanism               lean target  ->  rich target     (resurgence)
local choice                0.14   0.14   0.14   0.14   0.14    (flat)
behavioral momentum         0.24   0.25   0.26   0.26   0.27    (INCREASES)
dual exc/inhib              0.14   0.14   0.14   0.14   0.14    (flat)
resurgence as choice        0.21   0.21   0.21   0.20   0.20    (flat / slight decline)
```

**Only behavioral momentum predicts that resurgence grows with the target's own
reinforcement history.** This is precisely the prediction that the empirical record does
not support (Craig & Shahan, 2016), and it is the cleanest single dissociation in the
engine. A target-reinforcement-rate (or amount, or training-duration) manipulation
discriminates momentum from the other three.

### The diagnostic catalog

| Manipulation | local choice | behavioral momentum | dual exc/inhib | resurgence as choice | What it isolates |
|---|---|---|---|---|---|
| **Alternative reinf. rate ↑** (phase 2) | resurgence ↑ | ↑ | ↑ | ↑ | *Not* diagnostic (all ↑); matches data |
| **Target reinf. rate / training ↑** (phase 1) | flat | **↑** | flat | flat | **Momentum vs. the rest** (simulated) |
| **Context switch between phases** (ABA/ABC renewal) | none | none | **renewal: context-specific suppression** | none | **Dual vs. the rest** *(proposed)* |
| **Retention interval before test** (rest, no responding) | none | partial (mass persists) | **spontaneous recovery of target** | both values decay (net small) | **Dual vs. the rest** *(proposed)* |
| **Target resistance to OTHER disruptors** (satiation, distraction) | no special pattern | **scales with target reinf. rate** | weak | no special pattern | **Momentum signature** *(proposed)* |
| **Reinforcer devaluation** (target outcome) | resurgence ↓ | weaker (Pavlovian persistence) | ↓ (excitation is outcome-linked) | resurgence ↓ | Value/choice vs. momentum *(proposed)* |
| **Multiple / serial alternatives** | qualitative only | qualitative only | qualitative only | **quantitative matching predictions** | **RaC's molar claim** *(proposed)* |
| **Time scale / session spacing** (massed vs. spaced, within- vs. between-session) | — | — | — | **τ-dependent transient** | **RaC temporal weighting** (τ simulated) |
| **Resurgence vs. a never-trained response** | only if it had value | yes (mass) | yes (preserved excitation) | yes (had reinforcement history) | Generic reallocation vs. target-specific learning *(proposed)* |

Reading the table: **reinforcement-rate of the *target*** separates momentum; **context
and time** separate the dual/inhibition account; **quantitative matching across multiple
alternatives and time scales** is where RaC stakes its claim; the local-choice account is
the parsimonious null that the others must beat by explaining the *other* relapse
phenomena it cannot.

---

## The moral, and the way forward

1. **The canonical resurgence curve is not evidence for any one process.** At least four
   produce it, and a reinforced-alternative control rules out only the trivial
   alternatives, not the substantive ones.

2. **The reinforcement-rate manipulations are partly degenerate.** Alternative-rate is
   non-diagnostic (all mechanisms predict the same direction); target-rate is the clean
   one and it indicts momentum — consistent with the empirical literature that motivated
   Resurgence as Choice.

3. **The decisive tests are about *what is retained*, not *how much*.** Context (renewal),
   time (spontaneous recovery), reinforcer identity (devaluation), and target-specificity
   ask whether a preserved association is doing the work (dual) or whether resurgence is
   pure value reallocation (local / RaC). These are the experiments worth running.

4. **The four are not mutually exclusive.** The most likely truth is a blend: choice
   dynamics (RaC) operating over preserved associations with context dependence (dual),
   with momentum's resistance-to-change as a real but distinct phenomenon that the
   *augmented* momentum *model of resurgence* simply over-reached to claim. A quantitative
   model-comparison program — fitting the formal versions of all four to a common dataset
   spanning the manipulations above — is the proper next step.

---

## Limitations

- These are **minimal, illustrative** mechanisms in one shared engine, not the published
  quantitative models (the RaC equations, the augmented BMT model, formal Bouton-style
  models). The simulated dissociations show the *engine's* mechanisms diverge; a real
  contribution would fit the actual models to data.
- Magnitudes are parameter-dependent. The dissociations are reported for their
  **direction and qualitative shape**, which are robust to the tuning, not for their
  exact values.
- The engine's "local choice" and "resurgence as choice" arms are close cousins (both
  matching-based); the distinction drawn here — error-correcting value + softmax vs.
  temporally-integrated reinforcement + power-law matching — is real but finer than the
  momentum-vs-choice and association-vs-choice contrasts that the table foregrounds.

## References

- Bouton, M. E. (2004). Context and behavioral processes in extinction. *Learning &
  Memory*, 11, 485–494.
- Bouton, M. E., Winterbauer, N. E., & Todd, T. P. (2012). Relapse processes after the
  extinction of instrumental learning: Renewal, resurgence, and reacquisition.
  *Behavioural Processes*, 90, 130–141.
- Craig, A. R., & Shahan, T. A. (2016). Behavioral momentum theory fails to account for
  the effects of reinforcement rate on resurgence. *Journal of the Experimental Analysis
  of Behavior*, 105, 375–392.
- Epstein, R. (1983). Resurgence of previously reinforced behavior during extinction.
  *Behaviour Analysis Letters*, 3, 391–397.
- Konorski, J. (1967). *Integrative Activity of the Brain*. University of Chicago Press.
- Leitenberg, H., Rawson, R. A., & Mulick, J. A. (1975). Extinction and reinforcement of
  alternative behavior. *Journal of Comparative and Physiological Psychology*, 88,
  640–652.
- Nevin, J. A., & Grace, R. C. (2000). Behavioral momentum and the law of effect.
  *Behavioral and Brain Sciences*, 23, 73–130.
- Shahan, T. A., & Sweeney, M. M. (2011). A model of resurgence based on behavioral
  momentum theory. *Journal of the Experimental Analysis of Behavior*, 95, 91–108.
- Shahan, T. A., & Craig, A. R. (2017). Resurgence as Choice. *Behavioural Processes*,
  141, 100–127.
- Sweeney, M. M., & Shahan, T. A. (2013). Effects of high, low, and thinning rates of
  alternative reinforcement on response elimination and resurgence. *Journal of the
  Experimental Analysis of Behavior*, 100, 102–116.
- Winterbauer, N. E., & Bouton, M. E. (2010). Mechanisms of resurgence of an extinguished
  instrumental behavior. *Journal of Experimental Psychology: Animal Behavior Processes*,
  36, 343–353.
