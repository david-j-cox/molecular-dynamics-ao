# Novelty reconciliation: the two abstract-only papers, now read in full

Date: 2026-06-04. Both PDFs are in `refs/`. This note records what each paper *actually*
claims (full text, not abstract) and what it does to our one novel result -- the
skew-preference reversal at the requirement. Companion to `related_work.md`; where the two
disagree, this note is the corrected version.

Framing note: this is not a priority contest. The goal is to draw the line between what is
already known and what is still open, so we work on the open part and build on the rest.

Bottom line up front (updated after reading H&R Appendix S4): **The skew-preference reversal at
the survival requirement is already KNOWN -- Caraco & Chasin 1984 (single threshold) and H&R
2024 Appendix S4 (single + twin threshold, via the same third-derivative/prudence mechanism).
We reproduce it; we do not claim it. Coricelli et al. addresses a different question and is a
contrast cite. The still-open ground we should lead with: (a) an exact global all-moments phase
diagram, (b) a derived kurtosis/temperance prediction, (c) the cross-mechanism collapse to a
single axis R, and (d) sequence-based recovery / identifiability of R -- which H&R explicitly
name as the field's open problem ("the devil is in the sequence").**

---

## 1. Coricelli, Diecidue & Zaffuto (2018), *J. Risk Uncertain.* 56:193-210

What it is: a human economics experiment (49 subjects, 168 binary choices between
three-outcome, equal-EV prospects, no feedback). Goal: disentangle an **aspiration-level**
simplifying heuristic from a **skewness-preference** compensatory strategy, and MLE-compare
EU / EU+AL / CPT / CPT+AL.

What it actually finds:
- A **uniform preference for positive skewness** (dSkw coefficient positive and significant
  in Models 1, 3, 4, 5; Table 3): subjects prefer the greater positive skew and the lesser
  negative skew. This is a single-sign preference, **not a reversal**.
- An aspiration-level heuristic that operates when only one prospect can reach the target;
  modeled as Diecidue & van de Ven's (2008) value function with discontinuous jumps mu+
  (success) and lambda- (failure) at the aspiration level.
- The two strategies are **decoupled**: skew preference is the compensatory fallback used
  *when aspiration levels are not predictive*. There is no claim that being below vs above
  the aspiration level flips the sign of skew preference.

The one tempting passage and why it does NOT pre-empt us: Model 2 (Table 3) reports a
**negative** dSkw coefficient. But the authors explicitly attribute this to the fact that, in
that subset of conditions, the aspiration-achieving prospect *happens* to have the lowest
skewness -- so the negative sign is an artifact of aspiration-based choosing, not a genuine
below-target preference for negative skew. They never interpret it as a skew reversal.

What this leaves open: their aspiration level is a fitted psychological reference point
producing probability-of-success effects; ours is a survival requirement R that produces a
derived skew **sign reversal** (negative skew preferred below R, positive above). Different
object, different mechanism, different prediction -- so the requirement-crossing reversal is a
question Coricelli does not address. Coricelli is worth citing -- as the closest economics
analog (aspiration/reference point + skewness in one model). Our `related_work` framing
("aspiration-modulated skew," implying they modulate skew by aspiration) was **too generous
to them**, an artifact of reading the abstract; correct it to "two decoupled strategies;
uniform positive-skew preference."

---

## 2. Houston & Rosenstrom (2024), *Biol. Rev.* 99:478-495

What it is: the field's own critical review of risk-sensitive foraging (RSF), functional and
mechanistic. On full reading it maps more of the skew terrain as already known than the
abstract implied.

### 2a. The memory note's characterization was wrong on one point
The note (and `related_work.md`) say H&R "names skewed-distribution RSF an OPEN problem."
**That is not what the full text says.** The open problem they actually name is the *general*
account -- "we lack a simple and empirically defendable general account of it in either
mechanistic or evolutionary terms" -- and the path forward they advocate is **choice
sequences**, not skew ("the devil is in the sequence"; Section VI.3, Section IX.8). Skew is
not their headline gap.

### 2b. What is already known: they say they already derive the skew case
Section V.5 (Reproduction), p.485, verbatim:

> "Naturally, RSF models like the twin thresholds model can be applied to novel situations
> with a little extra effort: in **Appendix S4 we show how to derive RSF model predictions
> for skewed reward distributions** (Caraco & Chasin, 1984; cf. Genest et al., 2016)."

So H&R treat skewed-distribution RSF not as open but as a **worked extension** sitting in
their Supporting Information. And they anchor it to:
- **Caraco & Chasin (1984), "Foraging preferences: response to skew,"** *Anim. Behav.*
  32:76-85 -- an empirical paper, title literally about response to skew, in juncos, from
  the same Caraco lineage that established the variance reversal at the energy budget (Caraco
  1981). This is the most directly relevant prior work on skew in foraging, and we have not
  read it yet -- so what it established vs. left open is the key thing to find out.
- Genest, Stauffer & Schultz (2016), PNAS -- monkeys discriminate skew (already on our list).

### 2c. What this does and does not leave us
It does NOT establish that the *requirement-crossing sign reversal* (negative skew preferred
below R, positive above, as a clean normative statement) has been published. It DOES
establish that skew in RSF is not virgin territory (40-year-old empirical work + a 2024 supp
derivation), and that the earlier "skew RSF is an open problem per H&R" line is false as stated.

### 2c-bis. Appendix S4 now read (the decisive document) -- the reversal at the requirement is known
H&R Supporting Information Appendix S4 ("A formal EBR model of variance and skewness risk near
the survival threshold") is now in `refs/` and read in full. It derives **exactly the
skew-preference reversal at the requirement, by exactly the mechanism we planned to use.**
Specifics, verbatim where it matters:
- They take a **third-order Taylor expansion** of the value function V(x,s), with `Vxx` and
  `Vxxx` (second and third partial derivatives in reserve) as the operative terms -- i.e. the
  third-derivative / prudence mechanism is theirs already, not a bridge we are first to import.
- Result: **positive skewness adds value when `x + gi + gs > xs + nu*sqrt(s)`** (above the
  survival threshold by ~1 SD of background gain); **negative skewness adds value when
  `x + gi + gs < xs + nu*sqrt(s)`.** At `x + gi + gs = xs` the animal is neutral fixed-vs-
  symmetric but **prefers negative skew**; slightly below xs it prefers both variance and
  negative skew. Their intuition is identical to ours: "negative skewness helps to move the
  bulk of the variability above the survival threshold."
- They also state the **rare-event tail**: "when very far below xs -- when surviving depends on
  ... a very rare event -- positive skewness should become a preferred characteristic."
- Twin-threshold extension (Fig. S3B): "**three inflection points in mean reward** where
  preferences change, but **a fourth inflection point** emerges when comparing variance-only to
  variance + skew," including a preference change near the reproduction threshold. They note:
  "Caraco & Chasin (1984) find the optimal choice of skew in the case of a single threshold. We
  are not aware of prior studies investigating the ... twin threshold model for skewed reward
  distributions" -- and then do it themselves in S4.

So the honest, useful conclusion: **our skew-preference-reversal-at-R result reproduces known
theory** -- Caraco & Chasin (1984) for the single threshold (empirical + optimal), and H&R
(2024) Appendix S4 for the single + twin threshold (analytic, same prudence mechanism). This is
good to know: it independently validates our DP (we re-derive a published optimum), and it tells
us precisely where the still-open ground is. It is not the thing to "write up first as novel."

(We still do not have Caraco & Chasin 1984 itself -- no library access. Not blocking: H&R S4
characterizes its result, single-threshold optimal skew choice. Get it eventually for the exact
1984 statement, but it is no longer a gate.)

### 2d. Useful material H&R gives us
- The twin-threshold model (Hurly 2003; intermediate-variance preference) with two boundaries
  -- survival below, reproduction above -- is exactly the structure of our Tier 1.2 predation
  panel. Appendix S3 derives threshold-reaching from mean + variance. Cite both.
- Confirms the EBR's empirical support is mixed: "The EBR makes a clear prediction... risk
  averse when energy budget is positive, risk prone when negative. The data do not match this
  prediction (Kacelnik & El Mouden 2013)" (Conclusion 5). Keep our claims normative.
- Endorses our other distinctive angle indirectly: they call for theoretically-motivated
  study of **choice sequences** to tell mechanisms apart -- which is precisely Tier 2.1
  (sequence-based recovery / identifiability of R). H&R is the citation that motivates 2.1.
- Their Fig. 1A/1D (step -> sigmoid value function as time-to-dusk shrinks) is the same
  emergent value function whose derivatives V'', V''', V'''' we want to read off in Tier 1.1.

---

## Consequences for the build plan

### What is now KNOWN (we build on it, we do not re-claim it)
- The skew-preference **sign reversal at the survival threshold** (negative skew below, positive
  above, positive again in the far rare-event tail) -- Caraco & Chasin 1984 (single threshold);
  H&R 2024 Appendix S4 (single + twin threshold).
- The **mechanism** -- third derivative of the survival value function (`Vxxx`, = prudence) --
  is already used by H&R S4. The variance reversal (`Vxx`, the EBR) is older still (Stephens 1981).
- The twin-threshold structure with multiple inflection points, incl. behavior near a second
  (reproduction) boundary -- H&R S4 / S3.

### What is still UNKNOWN (the part worth our work)
1. **The all-moments theorem + exact global phase diagram.** H&R S4 is a *local* Taylor
   approximation near the threshold (they explicitly flag that its accuracy decays away from the
   expansion point) plus an illustrative ex-Gaussian numerical figure. An **exact, global**
   moment-dominance map over the whole (reserve, time-to-horizon) state space from the full
   survival-DP -- locating every inflection point exactly, not locally -- is not done. Modest
   but real: it turns their local result into an exact picture.
2. **Kurtosis / temperance (the 4th moment), stated generally.** H&R stop at the 3rd moment and
   only gesture ("higher-order statistics ... complex effects," citing Fawcett et al. 2014). A
   systematic "survival objective => sensitivity to ALL moments; which one governs is
   state-dependent; sign set by the sign of the corresponding derivative of V" theorem, with a
   derived **kurtosis** prediction and the explicit bridge to the economics prudence/temperance
   ladder (Eeckhoudt & Schlesinger 2006), is genuinely additive. This is Tier 1.1 + 1.3.
3. **Cross-mechanism collapse.** That the optimal policy, an evolved strategy, a learned
   model-based learner, and a model-free learner all reduce to the same single axis R -- S4 is
   pure optimality, so this is orthogonal and remains a distinctive contribution.
4. **Choice sequences + identifiability of R (Tier 2.1).** H&R's *actual* stated open problem
   ("the devil is in the sequence"; "not aware of prior studies"). The review itself solicits
   this. Strongest genuinely-open contribution; sequence-based recovery of R is the headline.

### Revised priority
The skew reversal is **no longer the "write up first" result** -- it is established. Reorder:
- **Lead with the unknowns**: the all-moments theorem + exact phase diagram (1.1), the kurtosis/
  temperance prediction (1.3), and especially **sequence-based recovery / identifiability of R
  (2.1)**, which H&R explicitly invite.
- **Position the skew reversal as validation**: "the emergent policy independently reproduces the
  Caraco & Chasin / H&R-S4 optimum" -- a correctness check on the mechanistic engine, not a
  discovery claim.
- Tier 1.2 (predation as a *mortality* second boundary) is still worth doing but overlaps H&R's
  twin-threshold (reproduction) boundary; the distinct part is predation-as-multiplicative-
  mortality + mass-dependent risk, not the mere existence of a second threshold. Scope it to the
  genuinely new piece.
