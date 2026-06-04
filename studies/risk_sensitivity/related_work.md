# Related work: where these results sit in the literature

A deep-dive done 2026-06-04 to situate the risk-sensitivity arc against the published
behavioral-ecology and decision-theory literature. Bottom line up front: **most of what we
re-derived is well-established theory, and we should frame it as a mechanistic
reproduction/derivation, not a discovery — with one exception (the skew-preference reversal),
which appears to be anticipated-in-principle but never explicitly stated, and is therefore the
genuinely novel contribution.**

This file is a map of prior art and an honest novelty assessment per result. It is not a claim
of priority; it is the literature we must cite and engage. Two key references (Houston &
Rosenstrom 2024; Coricelli, Diecidue & Zaffuto) were read via abstracts/secondary sources only —
flagged below for full-text verification before any manuscript.

## Novelty map (honest calibration)

| our result | status in the literature | closest prior art |
|---|---|---|
| Energy-budget rule (risk-prone below R, risk-averse above, reversal at R) | **Established.** Reproduction. | Stephens 1981; Caraco 1980/1981; Stephens & Krebs 1986 (z-score) |
| Requirement = reserve to survive an overnight fast | **Established / textbook.** | Stephens & Krebs 1986 z-score model |
| Deriving the rule from survival-maximizing DP (not an imposed utility) | **Established — the signature move of the field.** | McNamara & Houston 1986, 1992; Houston & McNamara 1999; Mangel & Clark 1988 |
| Threshold rises through the day toward dusk | **Established** ("dusk-loading" of reserves). | McNamara, Houston & Lima 1994; Bednekoff & Houston 1994 |
| Sun sets foraging *variance*; lifeline near deadline | Variance-timing is implied by the above; the day/night-variance framing is a fresh illustration. | (interruption/variability models) McNamara, Houston & Hutchinson |
| **Skew-preference reversal at the requirement** (mean-variance is insufficient) | **Anticipated-in-principle, never stated. The novel result.** | Houston & Rosenstrom 2024 (flags it as open); shortfall/z-score lineage assumes normality; Coricelli et al.; Eeckhoudt & Schlesinger 2006 (prudence) |
| Patch choice as a three-way safe/rich/wild rule | **Partially anticipated** (the interpolation is known); the 3-patch realization is fresh. | McNamara & Houston 1992; Stephens 1981 |
| Finite-horizon giving-up density (stop leaving near dusk; cutoff tracks travel) | **Partially anticipated / re-derivation.** Core mechanism already published. | Tenhumberg et al. 2001 (closest); Nonacs 2001; Brown 1988; Charnov 1976 |
| Cross-mechanism collapse (optimal = evolved = learned model-based = model-free), single axis R | **Synthesis is distinctive** — classic single-mechanism SDP papers do not unify these. | (each mechanism separately in the SDP literature) |

## 1. The energy-budget rule (Strand 1)

The core result — risk-prone when the reserve is below a requirement, risk-averse above, preference
reversing *at* the requirement, with equal-mean safe vs. risky options — is the founding result of
risk-sensitive foraging:

- **Caraco, Martindale & Whittam (1980)** and **Caraco (1981)** — the original empirical
  demonstration (yellow-eyed and dark-eyed juncos): risk-averse on a positive 24-h energy budget,
  risk-prone on a negative one. The named "expected energy budget rule."
- **Stephens (1981), "The logic of risk-sensitive foraging preferences"** — the analytic backbone
  and the single most-canonical citation for *our* exact framing: maximizing P(avoid starvation)
  against a fixed requirement deduces the reversal-at-the-requirement.
- **Stephens & Krebs (1986), *Foraging Theory*** — the textbook **z-score model**: a fixed reserve
  R needed to survive an overnight fast, minimize P(reserve < R), risk-averse above R and
  risk-prone below. This *is* our model in its classic form (reserve, hard threshold, overnight
  fast, equal means). Our `R = night_steps x metabolism` is a faithful operationalization.
- **McNamara & Houston (1992), "Risk-sensitive foraging: a review of the theory"** — recasts the
  static rule as state-dependent control: the reversal comes from the *curvature* of the terminal
  survival function (convex below, concave above). The exact switch can sit slightly off R and the
  transition can be gradual; our sharp reversal at R is the idealized hard-death-boundary case.
- Modern consensus is that the rule's *empirical* support is mixed and confounded by amount-vs-delay
  variance (**Kacelnik & Bateson 1996**; **Bateson 2002**), and the field "lacks a simple,
  empirically defendable general account" (**Houston & Rosenstrom 2024**, *Biological Reviews*).
  So our work is best positioned as a normative/optimality demonstration, not an empirical claim.

**Verdict:** reproduction of established theory. Anchor to **Stephens 1981** (theory) + **Caraco
1980** (empirical origin) + **McNamara & Houston 1992** (state-dependent generalization).

## 2. The survival-DP method and the time-of-day routine (Strand 2)

Backward stochastic dynamic programming over an energy-reserve state with a hard starvation
boundary is the *founding paradigm* of state-dependent behavioral ecology, not a novel method:

- **Mangel & Clark (1988), *Dynamic Modeling in Behavioral Ecology*** and **Clark & Mangel (2000),
  *Dynamic State Variable Models in Ecology*** — the founding texts; risk-sensitive foraging via
  survival-maximizing backward induction with a zero-reserve death boundary is the central worked
  example.
- **Houston & McNamara (1999), *Models of Adaptive Behaviour: An Approach Based on State*** — builds
  adaptive behavior on state (reserves), with the starvation-predation trade-off solved by SDP as a
  core example. The most direct founding text for our strand.
- **McNamara & Houston (1986), "The common currency for behavioral decisions"** — the "derive
  behavior from a fitness common currency, don't assume a utility" manifesto. Our refusal to impose
  a utility (the `exp030` -> `first_principles` move) is exactly this orthodoxy.
- **Houston, McNamara & Hutchinson (1993)** and **McNamara (1990)** — the general survival-DP
  starvation-predation trade-off and the optimal reserve that shifts parametrically with
  environmental harshness (our "requirement is the one axis" intuition).
- **Time-of-day rising threshold:** **McNamara, Houston & Lima (1994), "Foraging routines of small
  birds in winter"** and **Bednekoff & Houston (1994)** — optimal reserves are *dusk-loaded*; mass
  is built toward dusk to survive the night. This is precisely our threshold rising toward the
  overnight requirement. **Brodin (2007)** reviews the whole SDP-over-reserves lineage as the field
  standard.

**Verdict:** the method and the rising-threshold result are standard. Our distinctiveness is the
*synthesis* — unifying the survival-DP threshold with evolution of the trait, within-life
model-based learning, and model-free learning, and collapsing the family to a single parametric
axis (R). The classic single-mechanism papers do not make this cross-mechanism collapse.

## 3. The skew-preference reversal (Strand 3) -- the novel result

Classical risk-sensitive foraging is framed in mean and variance. Our finding that the survival
policy is *not* mean-variance-sufficient (it discriminates same-mean-same-variance gambles by
skew) and that the skew preference *reverses at the requirement* (negative skew below, positive
skew / negative-skew-avoidance above) is, on this search, **anticipated-in-principle but never
explicitly derived or stated:**

- The foraging shortfall/z-score objective (**Stephens & Charnov 1982**; **Caraco 1980**;
  **Lim, Wittek & Parkinson 2015**) is a *whole-distribution* functional that *should* be
  skew-sensitive — but the field collapsed it to mean-variance by assuming **normally distributed**
  reserves, so skew was assumed away (identically zero), never studied. Lim et al. (2015) state the
  normality assumption on the record.
- **Houston & Rosenstrom (2024)** (*Biological Reviews*) explicitly flags deriving RSF predictions
  for **skewed** reward distributions as an *open frontier* — confirming the gap is current,
  recognized, and unfilled. This is the citation most capable of weakening a strong novelty claim;
  **read the full text before any manuscript.**
- Economics/psychology establish both halves *separately* but never unify them into a
  survival-objective derivation with our specific sign-flip: aspiration-level kinks make choice
  depend on more than mean-variance (**Friedman & Savage 1948**; **Lopes 1987** SP/A theory;
  **Diecidue & van de Ven 2008**); the **prudence / third-derivative** mechanism explains *why* a
  survival kink induces skew sensitivity (**Eeckhoudt & Schlesinger 2006**); aspiration position
  modulates skew preference behaviorally (**Coricelli, Diecidue & Zaffuto** — closest economics
  analog, verify full text); skewness-seeking is robust in humans (**Golec & Tamarkin 1998**;
  **Astebro et al. 2015**) and animals discriminate skew at matched mean+variance
  (**Genest, Stauffer & Schultz 2016**, *PNAS*, monkeys).
- Defensive cite: **Olschewski et al. 2024** (*PNAS*) — in decisions-from-experience, apparent
  skew preference can be a "frequent-winner" artifact. We must note our effect is a property of the
  *optimal policy under a survival objective*, not an experience-sampling artifact.

**Verdict:** genuinely novel in its precise form. Defensible claim: *"the first explicit
demonstration that a survival/shortfall objective with a hard reserve requirement refutes
mean-variance sufficiency and produces a threshold-tied skew-preference reversal"* — while
crediting the shortfall and aspiration-level literatures as having implied skew sensitivity without
stating it, and the prudence framework as the mechanism. This is the result worth writing up first.

## 4. Multi-patch: three-way choice and finite-horizon MVT (Strand 4)

- **Patch choice (safe / rich-rate-maximizing / wild-variance):** the two endpoints
  (rate-maximizing when comfortable, variance-prone when below R near the deadline) are exactly the
  Stephens (1981) budget rule and its DP generalization (**McNamara & Houston 1992**), where the
  optimal policy provably *interpolates* between rate-maximizing and risk-prone over the
  reserves x time surface. Novel only as an explicit, inspectable three-patch realization in which
  the middle regime is the classic optimal-foraging rate-maximizer. **Partially anticipated.**
- **Finite-horizon giving-up density:** **Tenhumberg, Keller, Possingham & Tyre (2001), "Optimal
  patch time allocation for time-limited foragers"** is the closest prior art — a DP model dropping
  MVT's infinite-horizon assumption shows a time-limited forager holds patches *longer* as the end
  of the period approaches, because there is no time to profit from traveling to a fresh patch;
  verified in a parasitoid. That is our mechanism. **Nonacs (2001), "State dependent behavior and
  the marginal value theorem"** shows survival-currency state-dependence systematically lengthens
  residence beyond MVT. **Brown (1988)** supplies the missed-opportunity-cost term that collapses
  near the deadline (the GUD framing). **Charnov (1976)** is the rate-maximizing infinite-horizon
  baseline we frame as the not-desperate limit. **Re-derivation / clean synthesis, not a novel
  mechanism;** our framing (a finite-horizon GUD reducing to Charnov's MVT in the infinite-horizon
  limit, cutoff pinned to travel cost) is not stated this cleanly in any single source, but
  Tenhumberg et al. 2001 owns the core result and must be cited prominently.
- Recent **risk-MVT** (**Calcagno et al. 2025**) adds *predation* risk to residence time — a
  contrast cite (predation risk, not reward variance or a metabolic horizon).

## How to position the arc

Frame the bulk (energy-budget rule, survival-DP derivation, dusk-loading, three-way patch choice,
finite-horizon GUD) as a **faithful mechanistic reproduction and unification** of the
McNamara-Houston / Mangel-Clark program in a single inspectable agent-based engine, valuable for
its transparency and its cross-mechanism collapse (optimal = evolved = learned), not for the
qualitative predictions. Stake the **novelty** claim on the **skew-preference reversal**, which the
literature anticipates in principle (shortfall objectives, prudence, aspiration kinks) but has never
explicitly derived — with Houston & Rosenstrom (2024) itself naming it an open problem.

## References (with DOIs / stable URLs)

- Brown, J. S. (1988). Patch use as an indicator of habitat preference, predation risk, and
  competition. *Behav. Ecol. Sociobiol.* 22(1): 37-47. doi:10.1007/BF00395696
- Bateson, M. (2002). Recent advances in our understanding of risk-sensitive foraging preferences.
  *Proc. Nutr. Soc.* 61(4): 509-516. doi:10.1079/PNS2002181
- Bednekoff, P. A., & Houston, A. I. (1994). Avian daily foraging patterns: effects of digestive
  constraints and variability. *Evol. Ecol.* 8: 36-52. doi:10.1007/BF01237664
- Brodin, A. (2007). Theoretical models of adaptive energy management in small wintering birds.
  *Phil. Trans. R. Soc. B* 362: 1857-1871. PMC2442386
- Calcagno, V., Grognard, F., Hamelin, F. M., & Mailleret, L. (2025). Taking fear back into the
  marginal value theorem: the risk-MVT and optimal boldness. *Evolution* 79(9). doi:10.1101/2023.10.31.564970 (bioRxiv)
- Caraco, T., Martindale, S., & Whittam, T. S. (1980). An empirical demonstration of risk-sensitive
  foraging preferences. *Anim. Behav.* 28(3): 820-830. doi:10.1016/S0003-3472(80)80142-4
- Caraco, T. (1981). Energy budgets, risk and foraging preferences in dark-eyed juncos.
  *Behav. Ecol. Sociobiol.* 8: 213-217. doi:10.1007/BF00299833
- Charnov, E. L. (1976). Optimal foraging: the marginal value theorem. *Theor. Popul. Biol.* 9(2):
  129-136. doi:10.1016/0040-5809(76)90040-X
- Clark, C. W., & Mangel, M. (2000). *Dynamic State Variable Models in Ecology*. Oxford UP.
- Coricelli, G., Diecidue, E., & Zaffuto, F. D. Aspiration levels and preference for skewness in
  choice under risk. SSRN 2767382 / INSEAD WP 2016/28/DSC. (verify full text)
- Diecidue, E., & van de Ven, J. (2008). Aspiration level, probability of success and failure, and
  expected utility. *Int. Econ. Rev.* 49: 683-700. doi:10.1111/j.1468-2354.2008.00494.x
- Ebert, S., & Strack, P. (2015). Until the bitter end: on prospect theory in a dynamic context.
  *Am. Econ. Rev.* 105: 1618-1633. doi:10.1257/aer.20130896
- Eeckhoudt, L., & Schlesinger, H. (2006). Putting risk in its proper place. *Am. Econ. Rev.* 96:
  280-289. doi:10.1257/000282806776157777
- Friedman, M., & Savage, L. J. (1948). The utility analysis of choices involving risk.
  *J. Polit. Econ.* 56: 279-304.
- Genest, W., Stauffer, W. R., & Schultz, W. (2016). Utility functions predict variance and skewness
  risk preferences in monkeys. *PNAS* 113: 8402-8407. doi:10.1073/pnas.1602217113
- Golec, J., & Tamarkin, M. (1998). Bettors love skewness, not risk, at the horse track.
  *J. Polit. Econ.* 106: 205-225. doi:10.1086/250004
- Houston, A. I., & McNamara, J. M. (1999). *Models of Adaptive Behaviour*. Cambridge UP.
- Houston, A. I., McNamara, J. M., & Hutchinson, J. M. C. (1993). General results concerning the
  trade-off between gaining energy and avoiding predation. *Phil. Trans. R. Soc. B* 341: 375-397.
  doi:10.1098/rstb.1993.0123
- Houston, A. I., & Rosenstrom, T. H. (2024). A critical review of risk-sensitive foraging.
  *Biol. Rev.* 99(2): 478-495. doi:10.1111/brv.13031 (PMID 37987237; verify full text)
- Kacelnik, A., & Bateson, M. (1996). Risky theories -- the effects of variance on foraging
  decisions. *Am. Zool.* 36(4): 402-434. doi:10.1093/icb/36.4.402
- Lim, I. S., Wittek, P., & Parkinson, J. (2015). On the origin of risk sensitivity: the energy
  budget rule revisited. *Anim. Behav.* 110: 69-77. arXiv:1504.04986
- Lopes, L. L. (1987). Between hope and fear: the psychology of risk. *Adv. Exp. Soc. Psychol.* 20:
  255-295. doi:10.1016/S0065-2601(08)60416-5
- Lucas, J. R. (1985). Time constraints and diet choice. *Am. Nat.* 126(5): 680-705. doi:10.1086/284447
- Mangel, M., & Clark, C. W. (1988). *Dynamic Modeling in Behavioral Ecology*. Princeton UP.
- McNamara, J. M. (1990). The value of fat reserves and the trade-off between starvation and
  predation. *Acta Biotheor.* 38: 37-61. doi:10.1007/BF00047272
- McNamara, J. M., & Houston, A. I. (1986). The common currency for behavioral decisions.
  *Am. Nat.* 127: 358-378. doi:10.1086/284489
- McNamara, J. M., & Houston, A. I. (1992). Risk-sensitive foraging: a review of the theory.
  *Bull. Math. Biol.* 54(2-3): 355-378. doi:10.1007/BF02464838
- McNamara, J. M., Houston, A. I., & Lima, S. L. (1994). Foraging routines of small birds in winter:
  a theoretical investigation. *J. Avian Biol.* 25(4): 287-302. doi:10.2307/3677064
- Nonacs, P. (2001). State dependent behavior and the marginal value theorem. *Behav. Ecol.* 12(1):
  71-83. doi:10.1093/oxfordjournals.beheco.a000381
- Olsson, O., & Holmgren, N. M. A. (1998). The survival-rate-maximizing policy for Bayesian
  foragers: wait for good news. *Behav. Ecol.* 9(4): 345-353. doi:10.1093/beheco/9.4.345
- Olschewski, S., Spektor, M. S., & Le Mens, G. (2024). Frequent winners explain apparent skewness
  preferences in experience-based decisions. *PNAS* 121: e2317751121. doi:10.1073/pnas.2317751121
- Real, L., & Caraco, T. (1986). Risk and foraging in stochastic environments. *Annu. Rev. Ecol.
  Syst.* 17: 371-390. doi:10.1146/annurev.es.17.110186.002103
- Stephens, D. W. (1981). The logic of risk-sensitive foraging preferences. *Anim. Behav.* 29(2):
  628-629. doi:10.1016/S0003-3472(81)80128-5
- Stephens, D. W., & Charnov, E. L. (1982). Optimal foraging: some simple stochastic models.
  *Behav. Ecol. Sociobiol.* 10: 251-263. doi:10.1007/BF00302814
- Stephens, D. W., & Krebs, J. R. (1986). *Foraging Theory*. Princeton UP.
- Tenhumberg, B., Keller, M. A., Possingham, H. P., & Tyre, A. J. (2001). Optimal patch time
  allocation for time-limited foragers. *Behav. Ecol. Sociobiol.* 50: 134-141. doi:10.1007/s002650100348
- Astebro, T., Mata, J., & Santos-Pinto, L. (2015). Skewness seeking. *Theory Decis.* 78: 189-208.

## Sourcing caveats

- **Houston & Rosenstrom (2024)** and **Coricelli, Diecidue & Zaffuto** were assessed from
  abstracts/secondary descriptions, not full PDFs. They are the two papers most able to weaken the
  skew-reversal novelty claim; pull the full texts (institutional access) and confirm they do not
  already state the below/above sign-flip before any manuscript.
- The empirical consensus (Kacelnik & Bateson 1996; Houston & Rosenstrom 2024) is that the
  energy-budget rule's data support is mixed; keep our claims normative (optimality), not empirical.
