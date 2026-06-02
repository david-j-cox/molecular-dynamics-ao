"""Reproduction harness: run every experiment/demo and snapshot its findings.

Each experiment is run in a FRESH subprocess (so JAX/matplotlib/RNG state cannot
leak between runs, and it mirrors how a person runs them). The captured stdout --
which is where each script prints its key numbers -- is scrubbed of volatile lines
(wall-clock timings, absolute save paths) and stored as the snapshot.

Usage:
    python scripts/reproduce.py                 # capture baseline -> outputs/repro/baseline.json
    python scripts/reproduce.py --check          # re-run, diff against baseline
    python scripts/reproduce.py exp020 exp022     # only matching names (capture or --check)
    python scripts/reproduce.py --check --out cur.json

The intent: before a refactor, capture the baseline; after, run --check and review
every line that changed. Numeric drift is then a deliberate decision, not a surprise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPRO = ROOT / "outputs" / "repro"

# Put the repo root on PYTHONPATH so experiments that do `from experiments... import`
# resolve when run as a standalone script (only `behavioral_md` is pip-installed).
_ENV = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}

# Run order mirrors the build order; scripts/ demos last.
EXPERIMENTS = [
    "experiments/exp001_dynamics_and_emission.py",
    "experiments/exp002_core_controller_sweep.py",
    "experiments/exp003_learning_curve.py",
    "experiments/exp004_jax_benchmark.py",
    "experiments/exp005_zoo_acquisition_jax.py",
    "experiments/exp006_zoo_extinction_jax.py",
    "experiments/exp007_death_patterns.py",
    "experiments/exp008_matching.py",
    "experiments/exp009_matching_cod.py",
    "experiments/exp010_matching_multi.py",
    "experiments/exp011_matching_amount.py",
    "experiments/exp012_matching_canonical.py",
    "experiments/exp013_matching_probability.py",
    "experiments/exp014_matching_delay.py",
    "experiments/exp015_operant_chamber.py",
    "experiments/exp016_demand.py",
    "experiments/exp017_timing_scallop.py",
    "experiments/exp018_fr_vr_pause.py",
    "experiments/exp019_cumulative_records.py",
    "experiments/exp020_patch_leaving_mvt.py",
    "experiments/exp021_functional_response.py",
    "experiments/exp022_behavioral_momentum.py",
    "experiments/exp023_fit_sensitivity.py",
    "experiments/exp024_decoupled_fit.py",
    "scripts/run_demo.py",
    "scripts/run_extinction_demo.py",
    "scripts/run_generalization_demo.py",
    "scripts/run_peak_shift_demo.py",
]

# Volatile lines/fragments to neutralize so the snapshot tracks FINDINGS, not
# run-to-run noise (wall-clock benchmark times, absolute save paths).
_SCRUBS = [
    (re.compile(r"\d+\.\d+\s*s\b"), "<TIME>s"),                  # "1.23s"
    (re.compile(r"\d+\.?\d*\s*x\b"), "<SPEEDUP>x"),               # "84x" / "84.3x"
    (re.compile(r"[\d,]+ agent-steps/s"), "<RATE> agent-steps/s"),  # throughput (wall-clock)
    (re.compile(re.escape(str(ROOT))), "<ROOT>"),                 # absolute paths
]


def _scrub(text: str) -> str:
    lines = []
    for line in text.splitlines():
        for pat, repl in _SCRUBS:
            line = pat.sub(repl, line)
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def _name(path: str) -> str:
    stem = Path(path).stem
    return stem.split("_")[0] if stem.startswith("exp") else stem


def run_all(selected: list[str]) -> dict:
    snap: dict[str, dict] = {}
    for path in EXPERIMENTS:
        name = _name(path)
        if selected and not any(s in name or s in path for s in selected):
            continue
        print(f"running {name} ...", flush=True)
        proc = subprocess.run(
            [sys.executable, path], cwd=ROOT, capture_output=True, text=True, timeout=900,
            env=_ENV,
        )
        snap[name] = {
            "path": path,
            "returncode": proc.returncode,
            "stdout": _scrub(proc.stdout),
        }
        if proc.returncode != 0:
            print(f"  FAILED (rc={proc.returncode}); stderr tail:\n"
                  + "\n".join(proc.stderr.splitlines()[-8:]))
    return snap


def diff(baseline: dict, current: dict) -> int:
    """Print per-experiment diffs; return the number of experiments that changed."""
    changed = 0
    names = sorted(set(baseline) | set(current))
    for name in names:
        b, c = baseline.get(name), current.get(name)
        if b is None:
            print(f"[+] {name}: NEW in current")
            changed += 1
            continue
        if c is None:
            print(f"[-] {name}: MISSING from current")
            changed += 1
            continue
        if b["returncode"] != c["returncode"]:
            print(f"[!] {name}: returncode {b['returncode']} -> {c['returncode']}")
            changed += 1
        if b["stdout"] != c["stdout"]:
            changed += 1
            print(f"[~] {name}: output changed")
            bl, cl = b["stdout"].splitlines(), c["stdout"].splitlines()
            for i in range(max(len(bl), len(cl))):
                lb = bl[i] if i < len(bl) else ""
                lc = cl[i] if i < len(cl) else ""
                if lb != lc:
                    print(f"      - {lb}")
                    print(f"      + {lc}")
    if not changed:
        print("No differences from baseline.")
    return changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="diff against baseline")
    ap.add_argument("--out", default=None, help="snapshot filename under outputs/repro/")
    ap.add_argument("names", nargs="*", help="only run experiments matching these names")
    args = ap.parse_args()

    REPRO.mkdir(parents=True, exist_ok=True)
    snap = run_all(args.names)

    if args.check:
        out = REPRO / (args.out or "current.json")
        out.write_text(json.dumps(snap, indent=2))
        baseline_path = REPRO / "baseline.json"
        if not baseline_path.exists():
            print(f"\nNo baseline at {baseline_path}; capture one first.")
            return
        baseline = json.loads(baseline_path.read_text())
        print("\n=== diff vs baseline ===")
        n = diff(baseline, snap)
        print(f"\n{n} experiment(s) changed.")
    else:
        out = REPRO / (args.out or "baseline.json")
        out.write_text(json.dumps(snap, indent=2))
        n_ok = sum(1 for v in snap.values() if v["returncode"] == 0)
        print(f"\nSaved baseline: {out}  ({n_ok}/{len(snap)} ran cleanly)")


if __name__ == "__main__":
    main()
