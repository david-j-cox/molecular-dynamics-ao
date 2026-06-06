# Spatial travel cost and the energy-budget rule

When does spatial foraging **invert** the energy-budget rule (risk-averse when low) rather than
merely weaken it? This reconciles two earlier results in the molecular-molar arc:

- **exp037** (1D spatial forager): spatial travel **inverts** the rule.
- **exp045** (2D forager): spatial travel only **weakens** it (at those parameters).

## The variable: per-trip travel cost

The responsible quantity is the **travel cost** `t` -- the energy a *failed* risky trip wastes.
`exp046` isolates it in a fast trip-based survival-credit forager (no spatial pathing): each
day-encounter the organism chooses safe or risky, pays `t` (the round trip), then receives intake
(safe `S`; risky `0` or `2S`, matched mean). The two options share the same net mean (`S - t`); the
difference is the **downside** -- a risky 0-draw nets `-t` (the trip is wasted) while a safe trip nets
`S - t` (the trip pays off). Survival to dawn is credited back through an eligibility trace; the night
drain sets the requirement `R`.

## Finding

| travel cost t | reversal (below - above R) | regime |
|---|---|---|
| 0.00 | +0.18 | **rule** (risk-prone below R) |
| 0.02 | -0.07 | **inverted** (risk-averse below R) |
| 0.04 | -0.04 | inverted |
| 0.06 | +0.02 | ~flat |
| 0.08 | +0.13 | desperation (rule-like) |

- **t = 0:** the classic energy-budget rule.
- **small t:** the rule **inverts** -- a wasted risky trip is a real loss, lethal when low, so the
  organism takes the reliable safe option when low. This is exp037's inversion.
- **large t:** a **starvation-desperation** regime re-emerges (the harsh economy forces gambling), so
  the realized reversal is **non-monotonic** in travel cost.

## The honest conclusion

Introducing *any* travel cost flips the rule **toward inversion** -- that is the robust mechanism (the
asymmetric risky-downside loss). But the *realized magnitude* is **economy-dependent**, because travel
cost cannot be varied independently of (a) the overall energy economy and (b) the organism's
self-selected energy distribution -- it shifts both. So there is **no clean single-parameter phase
boundary**; the readouts are confounded (realized rate by visitation; the learned value at unvisited
states is uninformative; forcing uniform energy sampling breaks the survival-credit signal because
choices no longer determine survival). exp037 (full inversion) and exp045 (weakening) are two points
in this economy-coupled space, not a contradiction.

This is itself a useful negative-shaped result: spatial risk-sensitivity is not a clean function of
travel cost alone; the energy economy is inseparable from it.

See `experiments/exp046_travel_cost_study.py`, `docs/lab_notebook.md`, and the parent study
`studies/molecular_molar_bridge/`.
