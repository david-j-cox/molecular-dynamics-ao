# Behavioral Process Coverage -- the single-organism definition of done

The single organism is "done" when it can account for every known behavioral process of operant and
respondent conditioning, each demonstrated on its own, and -- the load-bearing requirement -- each
EMERGING from the one unmodified substrate (behavioral atoms + forces + Verlet integration + energy
dynamics + the Rescorla-Wagner credit-assignment learning rule). We hard-code a process-specific
mechanism only if we are forced to, and every such case is flagged here as a debt to revisit.

Respondent and operant are not two systems. They are one associative substrate read out under
different procedures, so the tables below are organized by procedure/phenomenon, not by "two boxes."
Higher-order repertoires (stimulus equivalence, relational responding, verbal behavior) are explicitly
OUT of scope for this milestone.

Status:     [x] demonstrated   [~] partial (mechanism present, effect not shown on its own)   [ ] gap
Emergence:  EMERGENT (from the core substrate)   ADDED (uses a process-specific term/structure --
            revisit)   CONFIG (needs a non-default rule variant)   VERIFY (not yet audited)   -- (gap)

The Emergence column is a FIRST PASS from the experiment docstrings; the rows marked VERIFY or ADDED
need a read of the implementation before they can be trusted (see "Emergence audit" at the bottom).

---

## Respondent (Pavlovian) procedures

| Process | Status | Evidence | Emergence |
|---|---|---|---|
| Acquisition of a CS-US association | [x] | exp039, exp044, exp064 | EMERGENT (cue-field RW) |
| Extinction of a CR | [~] | (cue weights decay on omission; not isolated as a CR-extinction demo) | VERIFY |
| Spontaneous recovery (of a CR) | [x] | run_spontaneous_recovery_demo | ADDED (passive inhibition decay over rest) |
| Stimulus generalization | [x] | exp044, run_generalization_demo | EMERGENT (receptor overlap) |
| Generalization gradient + peak shift | [x] | exp044, run_peak_shift_demo | EMERGENT (overlap + summed-error inhibition) |
| Discrimination (CS+ vs CS-) | [x] | run_stimulus_control_demo, exp044 | EMERGENT |
| Blocking (Kamin) | [x] | exp039 | CONFIG (rw_competitive credit) |
| Overshadowing | [x] | exp039 | CONFIG (shared prediction error) |
| Conditioned inhibition (summation + retardation tests) | [ ] | -- | -- |
| Latent inhibition | [ ] | -- | -- |
| Second-order / higher-order conditioning | [ ] | -- | -- |
| Sensory preconditioning | [ ] | -- | -- |
| Overexpectation | [ ] | (summed-error RW predicts it; untested) | -- |
| Trace conditioning | [ ] | -- | -- |
| Delay conditioning / inhibition of delay | [ ] | -- | -- |
| CS-US contingency vs contiguity (truly-random control) | [~] | exp064 (paired/unpaired) | VERIFY |
| Counterconditioning | [ ] | -- | -- |
| Renewal -- ABA | [x] | run_renewal_demo | ADDED (context signal + context-specific inhibition) |
| Renewal -- ABC / AAB | [ ] | -- | -- |
| Reinstatement | [ ] | -- | -- |
| Rapid reacquisition | [x] | run_reacquisition_demo | VERIFY (dual exc/inhib rule) |
| Conditioned suppression (CER) | [ ] | -- | -- |
| Autoshaping / sign-tracking | [ ] | -- | -- |
| Taste aversion / selective associations / one-trial | [ ] | -- | -- |
| US devaluation / revaluation | [ ] | -- | -- |
| Occasion setting (positive / negative) | [ ] | -- | -- |
| External inhibition / disinhibition | [ ] | -- | -- |

## Operant procedures

| Process | Status | Evidence | Emergence |
|---|---|---|---|
| Acquisition (by reinforcement) | [x] | exp005, run_demo | EMERGENT |
| Shaping by successive approximation (novel topography) | [ ] | -- | -- |
| Extinction (operant) | [x] | exp006, run_extinction_demo | EMERGENT |
| Extinction burst | [ ] | -- | -- |
| Spontaneous recovery (operant) | [x] | run_spontaneous_recovery_demo | ADDED (see respondent row) |
| Schedule signature -- FR (break-and-run) | [x] | exp019 | VERIFY |
| Schedule signature -- VR (high steady rate) | [x] | exp018, exp047 | VERIFY |
| Schedule signature -- FI (scallop) | [x] | exp017, exp019 | VERIFY (timing model) |
| Schedule signature -- VI (steady moderate) | [x] | exp008, run_vi_foraging_demo | EMERGENT |
| Post-reinforcement pause (FR > VR) | [x] | exp018 | VERIFY |
| Partial-reinforcement extinction effect (PREE) | [x] | exp027 | ADDED (Pearce-Hall associability) |
| Discriminative stimulus control (three-term) | [x] | run_stimulus_control_demo | EMERGENT |
| Conditional discrimination | [ ] | (= occasion setting) | -- |
| Matching -- concurrent VI-VI | [x] | exp008-012 | EMERGENT (softmax over learned value) |
| Matching -- undermatching / bias | [x] | exp008 | EMERGENT |
| Matching -- overmatching | [~] | (range not shown to cross a=1) | VERIFY |
| Matching -- amount / probability / delay sensitivity | [x] | exp011, exp013, exp014 | EMERGENT |
| Melioration / local choice dynamics | [~] | (matching emerges; melioration not isolated) | VERIFY |
| Behavioral momentum / resistance to change | [x] | exp022, exp026 | EMERGENT (atom mass = inertia) |
| Behavioral contrast (positive / negative) | [x] | exp040 | ADDED (shared hunger term) |
| Anticipatory contrast | [x] | exp041 | ADDED (learned predicted-income term) |
| Punishment -- positive | [x] | exp029 | CONFIG (subtractive/competitive variants) |
| Punishment -- negative (timeout / omission) | [ ] | -- | -- |
| Negative reinforcement -- escape | [ ] | -- | -- |
| Avoidance -- signaled (discriminated) | [ ] | -- | -- |
| Avoidance -- unsignaled (Sidman) | [ ] | -- | -- |
| Differential reinforcement -- DRL | [ ] | -- | -- |
| Differential reinforcement -- DRH | [ ] | -- | -- |
| Differential reinforcement -- DRO | [ ] | -- | -- |
| Differential reinforcement -- DRA / DRI | [ ] | -- | -- |
| Delay of reinforcement / delay discounting | [x] | exp014 | EMERGENT |
| Self-control choice (smaller-sooner vs larger-later) | [ ] | -- | -- |
| Probability discounting | [x] | exp013 | EMERGENT |
| Demand / behavioral economics (unit price) | [x] | exp016 | EMERGENT |
| Resurgence | [x] | exp028 | EMERGENT (choice reallocation) |
| Renewal (operant) | [x] | run_renewal_demo | ADDED (context machinery) |
| Risk sensitivity -- variance (energy-budget rule) | [x] | exp030, exp032-037 | EMERGENT (survival DP) |
| Risk sensitivity -- skew (prudence) | [x] | exp050, exp054, exp058 | EMERGENT |
| Risk sensitivity -- kurtosis (temperance) | [x] | exp054 | EMERGENT |
| Bet-hedging / geometric-mean fitness | [x] | exp057 | EMERGENT (selection) |
| Optimal foraging / patch-leaving (MVT) | [x] | exp020 | EMERGENT |
| Superstition / adventitious reinforcement | [ ] | -- | -- |
| Premack / response-deprivation | [ ] | -- | -- |
| Motivating operations -- deprivation / satiation | [~] | exp022 (satiation disruptor) | VERIFY (carried by the reserve?) |
| Schedule-induced / adjunctive (interim) behavior | [x] | exp062 | ADDED (CPG pacemaker atom) |
| Interval timing / temporal control | [x] | exp017 | VERIFY |
| Temporal stimulus control / food anticipation | [x] | exp064 | EMERGENT (learned light cue) |
| Contingency vs contiguity (correlation-based law of effect) | [~] | exp047, exp064 | VERIFY |
| Conditioned (secondary) reinforcement | [ ] | -- | -- |
| Behavior chaining (homo / heterogeneous) | [ ] | -- | -- |
| Token reinforcement | [ ] | -- | -- |
| Concurrent chains / conditioned-reinforcer value | [ ] | -- | -- |

## Pavlovian-instrumental interactions

| Process | Status | Evidence | Emergence |
|---|---|---|---|
| Autoshaping / negative automaintenance | [ ] | -- | -- |
| Sign-tracking vs goal-tracking | [ ] | -- | -- |
| Pavlovian-instrumental transfer (PIT) | [ ] | -- | -- |
| Two-process avoidance (Pavlovian fear + operant escape) | [ ] | -- | -- |

---

## Summary

Demonstrated `[x]`: ~38 fine-grained processes. Partial `[~]`: ~8. Gap `[ ]`: ~30.

The biggest clearly-in-scope clusters still missing:
1. **Negative reinforcement (whole quadrant):** escape, signaled avoidance, Sidman avoidance,
   negative punishment. The organism has an avoid_danger atom but no demonstrated escape/avoidance
   LEARNING -- a major hole.
2. **Conditioned reinforcement + chaining:** secondary reinforcement, behavior chains, tokens,
   concurrent chains. None demonstrated; foundational for complex operant behavior.
3. **Differential reinforcement:** DRL, DRH, DRO, DRA/DRI. None on their own (DRL is adjacent to the
   FI timing).
4. **The Pavlovian return-of-fear family beyond ABA renewal:** reinstatement, ABC/AAB renewal,
   conditioned inhibition (proper summation/retardation test), second-order conditioning,
   overexpectation, latent inhibition, sensory preconditioning, US devaluation.
5. **Pavlovian-instrumental bridge:** autoshaping, sign-tracking, PIT, two-process avoidance -- the
   natural showcase of the "one substrate" thesis, currently empty.
6. **Misc operant:** extinction burst, superstition, Premack/response-deprivation, shaping of a novel
   topography, self-control choice.

## Emergence audit (the third, deeper bar)

Several demonstrated `[x]` rows are flagged ADDED or CONFIG -- they reproduce the phenomenon but lean
on a process-specific term or a non-default rule, so they do not yet meet the "emerges from the
unmodified substrate" bar. These are debts to revisit (can the same effect fall out of the core
dynamics without the dedicated term?):
- Renewal (context signal + context-specific inhibition machinery)
- Behavioral contrast and anticipatory contrast (shared-hunger / predicted-income terms)
- PREE (Pearce-Hall variable associability)
- Spontaneous recovery (passive inhibition-decay term)
- Adjunctive behavior (a dedicated CPG pacemaker atom)
- Blocking / overshadowing / punishment (require selecting a specific credit-assignment or punishment
  rule variant -- arguably legitimate substrate options, but worth confirming they are not bespoke)

A focused pass should read each of these implementations and decide: legitimate substrate feature,
or process-specific hard-coding to be re-derived. The VERIFY rows (timing models, schedule signatures,
matching overmatching, MO) need the same read before their status/emergence is trusted.

## How to use this file

This is the live definition-of-done tracker. Flip `[ ]` -> `[x]` as each process is demonstrated,
prefer an EMERGENT demonstration, and record the evidence experiment. When the gap list is empty and
the ADDED/VERIFY debts are resolved (or accepted with reasons), the single organism is done and the
work turns to groups / social dynamics (see ToDO.txt).
