"""exp039 -- Kamin blocking and overshadowing from competitive credit assignment.

Two classic cue-competition phenomena, both produced by the same mechanism: when several cues are
present at reinforcement they share ONE prediction error (Rescorla-Wagner with
``credit_assignment="rw_competitive"``), so a cue only gains associative strength to the extent the
outcome is still surprising given the OTHER cues.

- Blocking (Kamin 1969): pretrain cue A -> US until A predicts the US, then reinforce the compound
  A+B. B acquires little, because A already predicts the US (error ~ 0). With independent credit
  (each cue its own error) B is learned normally -- no blocking.
- Overshadowing (Pavlov 1927): reinforce the compound A+B from the start (no pretraining). The two
  cues split the available associative strength, so each acquires less than if trained alone.

This is a minimal trial-based conditioning preparation that drives the engine's real Rescorla-Wagner
rule directly: the approach drive atom is engaged (eligibility) and the compound/element cue
intensities are presented on the ``light`` (= A) and ``cue`` (= B) channels with the US delivered as
the appetitive signal. We read the learned history weights (associative strengths) to A and B, and
contrast rw_competitive against rw_independent and against each cue trained alone.

Run:  python experiments/exp039_blocking_overshadowing.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from behavioral_md.atoms import default_atom_set
from behavioral_md.config import SimulationConfig
from behavioral_md.learning import EligibilityTrace, make_learning_rule

FIG = Path("outputs/figures")
A, B = "light", "cue"     # two conditioned stimuli; "food" is the US (appetitive signal)
AF = 0                    # approach_food drive atom (valence +1) -- the conditioned response
N_TRIALS = 60
LR = 0.3                  # fast convergence so 60 trials reach asymptote


def _intens(a: float, b: float) -> dict[str, float]:
    return {A: a, B: b, "food": 0.0, "danger": 0.0}


def _condition(rule, atoms, trials: int, intens: dict[str, float]) -> None:
    """Run `trials` reinforced trials with the given CS intensities present."""
    elig = EligibilityTrace(len(atoms), 0.95)
    for _ in range(trials):
        elig.trace[:] = 0.0
        elig.trace[AF] = 1.0   # the approach drive is engaged when the US arrives
        rule.update(atoms, elig, intens, appetitive=1.0, aversive=0.0, appetitive_exposure=True)


def _train(scheme: str, phases: list[tuple[int, dict[str, float]]]) -> tuple[float, float]:
    """Fresh organism with the given credit scheme; run the phases; return (w_A, w_B)."""
    cfg = SimulationConfig(credit_assignment=scheme, learning_rate=LR, reinforcement_asymptote=1.0)
    atoms = default_atom_set()
    rule = make_learning_rule(cfg)
    for trials, intens in phases:
        _condition(rule, atoms, trials, intens)
    w = atoms[AF].history_weights
    return w[A], w[B]


def main() -> None:
    compound = _intens(1.0, 1.0)
    a_alone = _intens(1.0, 0.0)
    b_alone = _intens(0.0, 1.0)

    # --- Blocking: pretrain A, then reinforce compound A+B. Does B get blocked? ---
    blk = {}
    for scheme in ("rw_competitive", "rw_independent"):
        _, w_b = _train(scheme, [(N_TRIALS, a_alone), (N_TRIALS, compound)])
        blk[scheme] = w_b
    _, b_ctrl = _train("rw_competitive", [(N_TRIALS, b_alone)])  # B trained alone (unblocked ref)

    # --- Overshadowing: reinforce compound A+B from scratch. Is B overshadowed? ---
    osh = {}
    for scheme in ("rw_competitive", "rw_independent"):
        _, w_b = _train(scheme, [(N_TRIALS, compound)])
        osh[scheme] = w_b

    print("Associative strength to cue B (history weight, asymptote = 1.0):\n")
    print("BLOCKING  (pretrain A, then compound A+B)")
    print(f"  rw_competitive : w_B = {blk['rw_competitive']:.3f}   <- blocked (A predicts US)")
    print(f"  rw_independent : w_B = {blk['rw_independent']:.3f}   <- not blocked")
    print(f"  B trained alone: w_B = {b_ctrl:.3f}\n")
    print("OVERSHADOWING  (compound A+B from the start)")
    print(f"  rw_competitive : w_B = {osh['rw_competitive']:.3f}   <- overshadowed (shared)")
    print(f"  rw_independent : w_B = {osh['rw_independent']:.3f}   <- full")
    print(f"  B trained alone: w_B = {b_ctrl:.3f}")

    labels = ["B alone\n(control)", "blocking\ncompetitive", "blocking\nindependent",
              "overshadow\ncompetitive", "overshadow\nindependent"]
    vals = [b_ctrl, blk["rw_competitive"], blk["rw_independent"],
            osh["rw_competitive"], osh["rw_independent"]]
    colors = ["0.4", "tab:red", "0.7", "tab:orange", "0.7"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(1.0, color="0.5", ls="--", lw=1, label="asymptote (full conditioning)")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("associative strength to cue B")
    ax.set_title("exp039: cue competition (blocking & overshadowing) under shared prediction error")
    ax.legend(fontsize=9)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    out = FIG / "exp039_blocking_overshadowing.png"
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
