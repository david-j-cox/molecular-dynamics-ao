# Research roadmap: strengthening the risk-sensitivity arc

Planned extensions, ranked and sequenced, from a 2026-06-04 scoping pass (four parallel
literature/feasibility agents). Companion to `related_work.md` (the novelty map of what we already
have). Nothing here is built yet -- this is the build plan, to be executed after the cited prior
art is checked for accessibility and correctness.

## The three goals every item serves

1. **Protect/deepen the one novel result** (the skew-preference reversal -- the survival objective
   is not mean-variance-sufficient).
2. **Add biology the survival-DP uniquely illuminates** (and that connects to existing engine pieces).
3. **Make empirical contact** -- the field's own review (Houston & Rosenstrom 2024) says this is
   the missing piece.

Convergent signal: all four agents independently landed on the same three judgments -- higher
moments are the novelty lever; adding a predator is the must-do robustness check; empirical/recovery
is the credibility gap.

---

## TIER 1 -- cheap, high-value, deepens the novel result (do first)

### 1.1 Moment-dominance phase diagram + a general "survival => all-moments" theorem  -- DONE (2026-06-06)
Built in `moment_dominance.py` (+ `survival.moment_preference_fields` / `central_moments` /
`mean_preserving_spread_advantage` / `field_zero_crossing`; `survival_dp` now also returns the
per-step continuation value `cont`). Rather than finite-differencing the gridded value (the
piecewise-linear `np.interp` makes high derivatives a reachability comb), each moment's preference
field is measured directly from the trusted DP by an isolated mean-preserving spread. RESULT: the
variance field reverses sign EXACTLY at the optimal threshold (corr 1.00, mean|diff| 0.005, rising
to R) -- the energy-budget rule IS the inflection (V''=0) of the emergent value; the skew field
reverses across the same threshold (prudence, sign of V'''); the kurtosis/temperance term is weak
here and deferred to 1.3 (needs the controlled moment-matched generator). 3 tests; 0 reproduce
drift. Figure: `figures/moment_dominance.png`.

- **What:** At each (reserve, time) cell, decompose the gamble's value change into moment
  contributions. The exact form: a mean-preserving spread changes the option's survival value by
  `(1/2)V''(e*)·var + (1/6)V'''(e*)·skew·sigma^3 + (1/24)V''''(e*)·kurt·sigma^4 + ...`, where `V` is
  the EMERGENT survival value function (the DP's continuation value). So **variance is governed by
  V'' , skew by V''' (prudence), kurtosis by V'''' (temperance)** -- and the energy-budget /
  skew / kurtosis reversals are sign changes of these derivatives across the requirement. Color
  each cell by the dominant moment; overlay the sign of V''' and V''''. State the general theorem:
  a survival/ruin-minimization objective is a whole-distribution functional, so the optimal policy
  is sensitive to all moments, and which one governs is state-dependent.
- **Must-cite:** Eeckhoudt & Schlesinger 2006 (prudence/temperance = moment preferences via utility
  derivatives -- the bridge we import into foraging); Menezes, Geiss & Tressler 1980 (downside risk
  = skewness); ruin-theory framing (Bayraktar inverse-survival-function; Lundberg). Bridge of
  foraging <-> ruin theory <-> prudence appears unstated in the literature = the theorem.
- **Tractability:** ~1 day, nearly free on the existing DP (finite-difference derivatives of the
  value function). Headline figure.
- **Novelty:** High. Generalizes the skew result; produces the memorable figure.

### 1.2 Predation as a second death source (the must-do robustness panel)  -- DONE (2026-06-06)
Built in exp050 + survival_dp predation_threshold/predation_prob (default off; 0 reproduce drift). On
a moderate winter economy (day 14 < night 16, R=0.48, which also defuses the R=0.72 critique) a
predation death boundary above x_r makes the value function develop a bounded safe band; BOTH the
variance (energy-budget) and skew reversals SURVIVE and the rule SPLITS into a band: risk-prone /
neg-skew below R, strongly risk-averse / pos-skew in the band R..x_r, and a risk-prone / neg-skew
escape regime above x_r. The reversal at R is robust; predation adds a second reversal at x_r (not an
inversion). 3 tests; figure exp050_predation_panel.png; manuscript sec:predation. (Used the
threshold-boundary form, not the linear mass-dependent form, which on a forced forager gives geometric
convex decay = escape-when-fat rather than the McNamara-Houston reserve target.)

- **What:** daily survival = `(1 - P_starve) * (1 - P_predation(foraging_effort))`, with
  mass-dependent predation (heavier reserve = slower = higher predation). Two-boundary survival
  problem (starve below, too-fat-predation above). Re-run the skew-matched gambles: does the skew
  reversal **survive, split into two thresholds, or invert**?
- **Must-cite:** McNamara & Houston 1990 (value of fat reserves; starvation-predation trade-off);
  Houston, McNamara & Hutchinson 1993 (general results); Brodin 2007 (review). The bare reserve-
  target result REPRODUCES known theory (validation milestone); the skew interaction is the novel bit.
- **Tractability:** ~2-4 days; predation enters the DP recursion as a second mortality term, no new
  solver.
- **Novelty:** Medium (the trade-off is canonical); the skew x two-boundary interaction is novel and
  pre-empts the reviewer's reflex objection.

### 1.3 Kurtosis / temperance as a derived fourth result  -- DONE (2026-06-06)
Built in exp054 + survival.kurtosis_outcomes (a >=5-point symmetric moment-matching generator: exact
mean/variance, zero skew, chosen kurtosis). RESULT: the survival objective is robustly TEMPERATE
(kurtosis-averse, V''''<0) on BOTH sides of R, magnitude comparable to the skew effect, vanishing
deep-safe (curvature control); measured on the smooth interior (dusk step-function slices excluded).
So, unlike variance and skew, kurtosis preference does NOT reverse at R -- the moment ladder is two
reversals (V'',V''') + one single-signed aversion (V''''). This DERIVES the sign of temperance from
survival (matching the Eeckhoudt-Schlesinger axiom) and CORRECTS this item's conjecture of a reversal
(there is none). Note: the desperate below-R forager is variance-SEEKING yet kurtosis-AVERSE. 2 tests;
figure exp054_temperance.png; manuscript sec:moments + fig:temperance. 0 reproduce drift.

(original plan:)
- **What:** Match mean+variance+skew, vary only kurtosis (needs a >=5-point moment-matching gamble
  generator). Predict kurtosis-aversion (temperance) above R, kurtosis-seeking near/below R --
  deriving the SIGN of temperance and its reversal from a survival objective.
- **Must-cite:** Eeckhoudt & Schlesinger 2006 (temperance = u''''<0 = kurtosis aversion); Deck &
  Schlesinger 2014 (empirical higher-order risk preferences); Lau et al. 2025.
- **Tractability:** ~1-2 days; the one item with non-trivial stimulus construction (moment-matching
  solver, scipy least-squares on >=5 support points). DP unchanged.
- **Novelty:** High -- a mechanistic origin for temperance that economics only measures.

---

## TIER 2 -- credibility / empirical (the field's stated weak point)

### 2.1 Sequence-based parameter recovery + identifiability of R  -- DONE (2026-06-07)
Built in `recovery.py` + exp055 (planner) + exp056 (RL degeneracy); survival_dp gains optional
`metabolism_night` (default None -> byte-identical) so R = night*m_night decouples from the daytime
burn. The behaving population chooses by `sigma(beta*(q_risky - q_safe))` at its current (reserve,
time); the conditional-choice likelihood is exact and closed-form (no intractable DP-likelihood -- so
Beaumont/ABC was not needed). RESULTS: (exp055) from the SEQUENCE of (reserve, choice) pairs R is
recovered at r=1.00 (profile-likelihood 95% width < grid step); from the AGGREGATE risky proportion +
occupancy it is NOT (r=-0.27, flat profile) -- a single scalar fits along a (R, beta) ridge. This is a
mechanistic account of why the field's aggregate-preference evidence is mixed. Fisher info localizes
R at the requirement boundary and m_day broader/lower (choices near R localize R, far from it localize
metabolism). (exp056) identifiability is from the STATE-DEPENDENT STRUCTURE, not from having sequences:
a generic delta-rule value-learner on the SAME choice has a degenerate anticorrelated (alpha, beta)
valley (ridge corr -0.96; the bandit alpha-beta degeneracy), recovering imprecisely (alpha r=0.81, CV
~55%) where the planner's R is sharp. 6 tests; 118 pass; ruff clean. Figures exp055_identifiability.png,
exp056_degeneracy.png. Cites Houston & Rosenstrom 2024, Wilson & Collins 2019, the bandit-recovery
degeneracy (Comput. Brain Behav. 2022).

(original plan:)
- **What:** Recover latent params (requirement R, metabolism, learning rate, softmax temperature)
  from trial-by-trial choice SEQUENCES; show R is identifiable from sequences but LOST in aggregate
  preference (explaining why the field's evidence is mixed). Profile-likelihood / Fisher-information
  map of where each parameter is identifiable (choices near R localize R; far from R localize
  metabolism).
- **Must-cite:** Houston & Rosenstrom 2024 ("the devil is in the sequence"); Wilson & Collins 2019
  (recovery protocol, confusion matrices); bandit recovery (Comput. Brain Behav. 2022) for the
  learning-rate vs softmax-temperature degeneracy; Beaumont 2010 (ABC) where the DP likelihood is
  intractable.
- **Tractability:** High; pure numpy/scipy, no external data.
- **Novelty:** Strong; directly answers the named review.

### 2.2 Model comparison vs SUT / mean-variance / prospect theory on skew-controlled data  -- PARTIAL (2026-06-07)
Built in `choice_models.py` + exp058 (4 panels). DELIVERED: (1) the structural result -- a pure
mean-variance model (and SUT, scalar/Weber noise on magnitude) is skew-BLIND, predicting 0.5 for
matched mean+variance whatever the skew; (2) the fit / AIC / cross-validated / model-recovery harness;
(3) the Genest 2016 contact -- their monkeys' ~70-80% +skew preference (matched mean+variance) refutes
the skew-blind model, and a curved-utility EU (their own u=ln(x+1)/0.81) and prospect theory fit it.
The survival-DP's distinguishing prediction (the skew preference REVERSES across R) is shown, and a
skew x budget interaction (=0 for EU/PT, +7 for survival-DP) is the design that would discriminate.
NOT DELIVERED (data-gated): a decisive empirical WIN of survival over curved-utility EU. Caraco &
Chasin 1984 -- the ideal matched mean+variance x energy-budget x skew set -- is UNAVAILABLE
(paywalled). Genest is single-budget, and Genest's own logistic test (their Eqs 1/2/3) already found
variance/skew add nothing beyond their curved utility (P~0.88-0.99), so survival and EU are
indistinguishable on it. The decisive test needs energy-budget x skew data (Caraco, or the 2.3
pre-reg). CITATION CORRECTION: McDevitt & Kacelnik 2014 is "Magnitude sensitivity: rethinking
decision-making" (magnitude sensitivity), NOT a prospect-theory-vs-RSF paper -- do not cite it as such.
6 tests; 134 pass; ruff clean. Figure exp058_model_comparison.png.

(original plan:)
- **What:** Refit Caraco & Chasin 1984 (sparrows; matched mean+variance, manipulated skew -- the
  ideal legacy dataset) and Genest, Stauffer & Schultz 2016 (monkeys; machine-readable; already
  shows a magnitude-dependent reversal). AIC / cross-validated comparison showing the survival-DP
  captures skew effects the second-moment rivals structurally cannot.
- **Must-cite:** Kacelnik & Bateson 1996 (Scalar Utility Theory -- the rival); McDevitt & Kacelnik
  2014 (prospect theory vs RSF) [CITATION UNVERIFIED -- the DOI below resolves to an unrelated paper
  and the title conflated a 2021 TiCS article; re-identify the intended work before citing];
  Stephens 1981 (z-score). Datasets: Caraco & Chasin 1984; Genest et
  al. 2016.
- **Tractability:** Medium; gated by digitizing legacy data (WebPlotDigitizer); recovers proportions,
  not sequences (which reinforces 2.1's point).
- **Novelty:** High impact -- turns the skew result into an empirical win over the standard rivals.

### 2.3 Pre-registered falsifiable skew-reversal experiment + DP power simulation
- **What:** Closed-economy operant design: matched mean+variance, 3 skew levels x 2 energy budgets,
  titration readout. DP-simulated predicted choice curve + power analysis + explicit refutation
  criteria separating survival-DP from SUT (no skew effect) and prospect theory (budget-invariant
  skew effect).
- **Must-cite:** Genest et al. 2016 (skew-gamble construction recipe); Bautista et al. 2005 (closed
  economy); Bateson & Kacelnik 1999 (starling energy-budget rationing); Kacelnik/Bateson titration.
- **Tractability:** High as a design (all components validated); the deliverable is the pre-reg +
  power sim, not a wet experiment.
- **Novelty:** High -- the experiment the review's "skew is an open frontier" remark implies.

---

## TIER 3 -- new biology, bigger builds

### 3.1 Bet-hedging meets the energy-budget rule  -- DONE (2026-06-07)
Built in `bethedge_fitness` + `evolve_bethedge` + `_markov_day_types` (survival.py) + exp057. Days are
a symmetric 2-state Markov good/bad chain (marginal fixed 50/50, stay-prob rho sets clustering); a bad
day scales intake by `bad_scale` (refuge quality); reserves carry across an `n_days` season with the
cap LIFTED so end-of-season reserve (fecundity) carries the variance bet-hedging acts on; the
geometric mean of cohort fecundity over seasons is the long-run growth selection maximizes. RESULT: a
phase diagram over (rho x bad_scale) -- with a refuge (bad_scale high) selection is CONSERVATIVE (play
safe, bet-hedge); with no refuge + clustered bad runs (bad_scale low, rho high) it is RISK-PRONE
(gamble; the energy-budget rule across days). Intermediate refuge: rising autocorrelation FLIPS the
optimum safe->gamble. The geometric optimum (analysis) and the evolved threshold (selection) agree.
TWO CORRECTIONS: (1) matched-mean options DO show bet-hedging once the cap is lifted (my earlier
"no trade-off" claim was a cap artifact); (2) the dominant effect of autocorrelated scarcity is
risk-PRONENESS, not the conservative bet-hedge this item guessed -- conservatism wins only where a
refuge exists (corrects the roadmap's guessed direction, as 1.3 corrected its guessed reversal).
4 tests; 122 pass; ruff clean. Figure exp057_bet_hedging.png. Cites Childs/Metcalf/Rees 2010, Philippi
& Seger 1989, Stephens 1981.

(original plan:)
- **What:** Markov good/bad days (tunable autocorrelation) fed into the EXISTING evolution +
  within-life simulators. Show the within-life energy-budget threshold and the across-generation
  bet-hedge are two readouts of one selection process; autocorrelation length shifts the evolved
  threshold.
- **Must-cite:** Childs, Metcalf & Rees 2010 (bet-hedging); Philippi & Seger 1989; Evolution 73(2)
  2019 (short-term insurance vs bet-hedging). Stephens 1981 for the within-life rule.
- **Tractability:** Cheap-moderate; mostly a simulator edit (reuses evolve_risk_policy).
- **Novelty:** Clean novel synthesis (two timescales unified).

### 3.2 Predator-prey co-evolutionary risk game (the user's predator/prey idea)
- **What:** Co-evolve a heritable prey risk threshold (we have it) against heritable predator
  effort; prey fitness from the survival DP, predator fitness from capture yield. Measure the joint
  ESS and whether selection responds to the SKEW of attack-success, not just its mean.
- **Must-cite:** Brown, Laundre & Gurung 1999 (ecology of fear); Calcagno et al. 2025 (risk-MVT --
  but rate-maximizing, fixed exogenous risk, no starvation boundary = our opening); Gil et al. 2025.
- **Tractability:** ~1-2 weeks; reuses the evolution harness; cost is stabilizing two-sided dynamics.
- **Novelty:** Genuinely novel in the survival-DP/reserve framing; risk of rediscovering Red Queen
  oscillations unless tied to skew.

### 3.3 State-dependent value of information / explore-exploit under starvation
- **What:** Belief-state DP (3-D: reserve x time x belief) over a hidden patch type. Predict
  information is worth LESS when desperate (horizon collapses -- you may not survive to use it), the
  opposite of "explore when uncertain", and that it interacts with skew (the skewed option is both
  the risk-prone and the information-rich choice).
- **Must-cite:** McNamara 1982 (optimal patch use in a stochastic environment); Olsson & Holmgren
  1998 ("wait for good news"); Dall et al. 2005 (value of information); honeybee energetic-state
  explore-exploit (Behav. Ecol. 2015); Bayesian-foraging meta-uncertainty (PLOS Comp Biol 2025).
- **Tractability:** Expensive (continuous belief grid; prototype with 2 patch types = 1-D belief).
- **Novelty:** High; the info-value-reverses-with-state result appears open.

### 3.4 Lifetime reproductive value currency (robustness control)
- **What:** Replace single-cycle survival with a reproductive-value continuation `V(E)` + a
  reproduce-vs-store action; keep the DP 2-D. Test whether the skew reversal is a property of the
  BOUNDARY or the CURRENCY.
- **Must-cite:** Houston & McNamara 1999; McNamara & Houston 1986 (common currency); Mangel & Clark
  1988.
- **Tractability:** Cheap-moderate; no belief dimension.
- **Novelty:** Mostly robustness, but "the skew reversal is invariant across currencies" (or a
  second reversal appears) is publishable either way.

---

## Molecular-agent bridge -- does the mechanism realize the optimum? (repo-unifying)

- **What:** The parent project's thesis is behavioral MOLECULAR DYNAMICS -- distributed response
  "atoms", stimulus forces, Verlet integration, learning history (`atoms.py`, `forces.py`,
  `organism.py`, `chamber.py`). The entire risk arc has been pure survival DP, divorced from that
  substrate. This bridge: give the atom-based organism an energy/hunger state that modulates its
  forces, and test whether its EMERGENT choice between a safe and a risky option reproduces the
  energy-budget rule (risk-prone when hungry) and ideally the skew sensitivity -- WITHOUT imposing
  the survival utility. I.e. does the messy distributed mechanism IMPLEMENT the normative optimum?
- **Why it matters:** unifies the two halves of the repo (mechanism <-> optimality) in a way no
  external literature touches. "Does the mechanism realize the optimum?" is the deepest internal
  question.
- **Prior art / framing:** internal; conceptually the rate-distortion / "bounded mechanism
  approximating an optimal policy" question. `chamber.run_risk_choice` already does an imposed-utility
  risk choice -- the bridge replaces the imposed utility with emergent atom dynamics + a hunger drive.
- **Tractability:** Medium-high; requires working inside the atom/force/organism code (more than a
  numpy DP edit). Its own thread after the Tier-1 wins.
- **Novelty:** Distinctive to this repo; the strongest "so what makes YOUR engine special" answer.

---

## Low-value / skip
- **4+ patches:** trivial incremental (N is just a loop; we already do a 3-patch menu). No new science.
- **Producer-scrounger game (bare version):** crowded -- Caraco & Giraldeau 1991 already solved P-S
  under a starvation/shortfall currency. Only the skew slice (does the scrounger fraction cross the
  skew-reversal boundary?) is novel, and it's a subtle sell.

## Recommended sequence
1.1 (headline, nearly free) -> 1.2 (must-do robustness) -> 2.1 (credibility foundation) -> 1.3
(temperance) -> 3.1 (bet-hedging, cheap) -> molecular bridge -> 2.2 (data, gated by
digitization) -> 3.2 / 3.3 / 3.4 (bigger builds) -> 2.3 (pre-reg).

## Caveats (verify before building / writing)
- Read via abstract/snippet only -- pull full texts and confirm they do not pre-empt our claims:
  **Houston & Rosenstrom 2024** (Biol. Rev.; most able to weaken the skew novelty), **Coricelli,
  Diecidue & Zaffuto** (aspiration-modulated skew), the **ruin-theory PDFs** (Bayraktar; Lundberg).
- Legacy citation details (Caraco 1981; Caraco & Chasin 1984 exact vol/page/DOI) assembled from
  secondary sources -- verify against PDFs before a reference list.
- Keep claims NORMATIVE (optimality), not empirical -- the energy-budget rule's data support is
  mixed (Kacelnik & Bateson 1996; Houston & Rosenstrom 2024).

## References / links

See the consolidated link list in the session notes; DOIs for every item are in `related_work.md`
plus, new to this roadmap:
- Eeckhoudt & Schlesinger (2006) AER 96(1):280-289. doi:10.1257/000282806776157777
- Menezes, Geiss & Tressler (1980) AER 70(5):921-932.
- Deck & Schlesinger (2014) Econometrica 82(5):1913-1943. SSRN 2198787
- Wilson & Collins (2019) eLife 8:e49547. doi:10.7554/eLife.49547
- Caraco & Chasin (1984) Anim. Behav. 32:76-85. doi:10.1016/S0003-3472(84)80326-7
- Genest, Stauffer & Schultz (2016) PNAS 113:8402-8407. doi:10.1073/pnas.1602217113
- Childs, Metcalf & Rees (2010) Proc. R. Soc. B 277:3055-3064. doi:10.1098/rspb.2010.0707
- Brown, Laundre & Gurung (1999) J. Mammalogy 80:385-399. doi:10.2307/1383287
- Calcagno et al. (2025) Evolution 79(9):1742. doi:10.1093/evolut/qpaf093 (bioRxiv 10.1101/2023.10.31.564970)
- McNamara (1982) Theor. Popul. Biol. 21:269-288. doi:10.1016/0040-5809(82)90018-1
- Olsson & Holmgren (1998) Behav. Ecol. 9:345-353. doi:10.1093/beheco/9.4.345
- Dall et al. (2005) TREE 20:187-193. doi:10.1016/j.tree.2005.01.010
- Bautista et al. (2005) J. Exp. Biol. doi:10.1242/jeb.01910
- McDevitt & Kacelnik (2014) Evol. Hum. Behav. doi:10.1016/j.evolhumbehav.2014.06.003 [UNVERIFIED:
  DOI resolves to an unrelated paper; title belongs to a 2021 TiCS article. Re-identify before citing.]
- Cartar & Dill (1990) Behav. Ecol. Sociobiol. 26:121-127. doi:10.1007/BF00171581
- Caraco & Giraldeau (1991) J. Theor. Biol. 153:559-583. doi:10.1016/S0022-5193(05)80156-0
