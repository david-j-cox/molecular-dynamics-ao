"""exp061 -- multi-dimensional & hierarchical atoms: rhythm, gaits, and a muscle->operant stack.

The engine's atoms were scalar (a single response tendency) -- and a scalar atom is a damped
INTEGRATOR with no restoring force, so under a constant drive it ramps (exp060), it cannot produce
temporal structure. Generalizing the atom to a multi-dimensional state with an internal coupling
matrix (``BehavioralAtom.internal_coupling``, ``oscillator_atom``) supplies the missing restoring
force and unlocks dynamics a scalar atom cannot:

  A. RHYTHM (central pattern generator). A 2-D atom with a diagonal spring oscillates: a
     pacemaker / fixed-action-pattern rhythm. The scalar atom under the same drive merely ramps.
  B. GAIT (phase structure). The 2-D atom's two muscles run anti-phase (flexor/extensor stepping) --
     a within-atom phase relationship; coupling across dimensions carries the temporal pattern.
  C. HIERARCHY (muscle -> response -> operant). A slow scalar OPERANT atom (a go/no-go decision)
     GATES the fast muscle CPG; the emitted behavior is the gated muscle output -- only while
     the operant commands it. This is the muscle/response/operant nesting with built-in timescale
     separation (the molar operant slow, the molecular muscle fast), the structure exp059 showed the
     molar policy needs.

Backward compatible: every existing atom is scalar (dims=1, coupling=None) and byte-identical
(reproduce exp001-030 unchanged). The generalization is opt-in.

Run:  python experiments/exp061_multidim_atoms.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from behavioral_md.atoms import _atom, oscillator_atom  # noqa: E402

FIG = Path("outputs/figures")
DT = 0.1
A_MIN, A_MAX = -10.0, 10.0


def scalar_ramp(force: float, steps: int) -> np.ndarray:
    """A scalar atom under constant drive: it ramps (no restoring force)."""
    atom = _atom("scalar")
    out = []
    for _ in range(steps):
        atom.integrate(force, DT, A_MIN, A_MAX)
        out.append(atom.activation)
    return np.array(out)


def cpg_run(steps: int, period: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """A 2-D oscillator atom free-running: returns both muscle traces."""
    atom = oscillator_atom("locomote", period=period, dt=DT, amplitude=1.0)
    m0, m1 = [], []
    for _ in range(steps):
        atom.integrate(np.zeros(2), DT, A_MIN, A_MAX)
        m0.append(atom.state[0])
        m1.append(atom.state[1])
    return np.array(m0), np.array(m1)


def hierarchy_run(steps: int, go_windows, period: float = 30.0) -> dict:
    """Muscle CPG gated by a slow operant atom (a go/no-go decision)."""
    operant = _atom("locomote_operant", mass=1.0)        # slow scalar decision atom
    muscle = oscillator_atom("muscle", period=period, dt=DT, amplitude=1.0)
    op_trace, musc_trace, behavior = [], [], []
    for t in range(steps):
        go = any(lo <= t < hi for lo, hi in go_windows)
        # Operant tracks the go-signal (damped scalar): drive up when go, down otherwise.
        drive = 4.0 if go else -4.0
        vel = (operant.state[0] - operant.previous_state[0]) / DT
        operant.integrate(drive - 8.0 * vel, DT, 0.0, 1.0)
        muscle.integrate(np.zeros(2), DT, A_MIN, A_MAX)
        gate = max(0.0, operant.activation)              # operant gates the muscle output
        op_trace.append(operant.activation)
        musc_trace.append(muscle.activation)
        behavior.append(gate * muscle.activation)        # emitted behavior = gated rhythm
    return {"operant": np.array(op_trace), "muscle": np.array(musc_trace),
            "behavior": np.array(behavior)}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    steps = 200

    ramp = scalar_ramp(0.3, steps)
    m0, m1 = cpg_run(steps)
    go_windows = [(40, 100), (140, 190)]
    hier = hierarchy_run(steps, go_windows, period=28.0)

    zc = int((np.diff(np.sign(m0)) != 0).sum())
    print("Multi-dimensional / hierarchical atoms:\n")
    print(f"A. scalar atom under drive RAMPS to {ramp[-1]:.1f} (no rhythm); a 2-D oscillator "
          f"atom CPGs with ~{2 * steps / max(zc, 1):.0f}-step period (amplitude {m0.max():.2f}).")
    print(f"B. the 2-D atom's two muscles run anti-phase: corr(m0, m1) = "
          f"{np.corrcoef(m0, m1)[0, 1]:+.2f} (flexor/extensor stepping).")
    on = np.array([any(lo <= t < hi for lo, hi in go_windows) for t in range(steps)])
    amp_on = np.abs(hier["behavior"][on]).mean()
    amp_off = np.abs(hier["behavior"][~on]).mean()
    print(f"C. operant-gated CPG: behavior amplitude {amp_on:.2f} when the operant says GO vs "
          f"{amp_off:.3f} otherwise -- the slow operant gates the fast muscle rhythm.")

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.5))
    t = np.arange(steps)

    ax[0].plot(t, ramp, color="tab:gray", lw=2, label="scalar atom (ramps)")
    ax[0].plot(t, m0, color="tab:blue", lw=2, label="2-D oscillator atom (CPG)")
    ax[0].axhline(0, color="0.85", lw=0.8)
    ax[0].set_xlabel("step")
    ax[0].set_ylabel("activation (output dim)")
    ax[0].set_title("A. Multi-dim + internal coupling = rhythm:\nscalar ramps, oscillator CPGs")
    ax[0].legend(fontsize=8, loc="upper left")

    ax[1].plot(t, m0, color="tab:blue", lw=2, label="muscle 0 (flexor)")
    ax[1].plot(t, m1, color="tab:red", lw=2, label="muscle 1 (extensor)")
    ax[1].axhline(0, color="0.85", lw=0.8)
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("muscle activation")
    ax[1].set_title("B. One 2-D atom carries a GAIT:\nanti-phase muscles (stepping)")
    ax[1].legend(fontsize=8, loc="upper right")

    ax[2].fill_between(t, -1.15, 1.15, where=on, color="0.9", label="operant says GO")
    ax[2].plot(t, hier["operant"], color="tab:green", lw=2, label="operant (slow decision)")
    ax[2].plot(t, hier["behavior"], color="tab:purple", lw=1.6, label="behavior (gated rhythm)")
    ax[2].axhline(0, color="0.85", lw=0.8)
    ax[2].set_xlabel("step")
    ax[2].set_ylabel("activation")
    ax[2].set_ylim(-1.2, 1.2)
    ax[2].set_title("C. Hierarchy: a slow OPERANT gates the\nfast muscle CPG (muscle→operant)")
    ax[2].legend(fontsize=7.5, loc="upper right")

    fig.tight_layout()
    out = FIG / "exp061_multidim_atoms.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
