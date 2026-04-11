"""One-command reproducibility: python reproduce.py

Regenerates all experiments, figures, and results from raw data.
Requires: pip install -e .
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

scripts = [
    ("Core experiments", [
        "experiments/exp1_dimensionality.py",
        "experiments/exp2_blind_spot.py",
        "experiments/exp3_fix_works.py",
        "experiments/exp4_corollaries.py",
    ]),
    ("Validation & robustness", [
        "experiments/reviewer_fixes.py",
        "experiments/monoculture.py",
        "experiments/paradigm_tests.py",
    ]),
]

failed = []
for group_name, group_scripts in scripts:
    print(f"\n{'='*60}")
    print(f"  {group_name}")
    print(f"{'='*60}")
    for s in group_scripts:
        if not os.path.exists(s):
            print(f"  SKIP (not found): {s}")
            continue
        print(f"\n  Running: {s}")
        result = subprocess.run([sys.executable, s], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAILED: {s}")
            print(result.stderr[-500:] if result.stderr else "")
            failed.append(s)
        else:
            print(f"  OK: {s}")

print(f"\n{'='*60}")
if failed:
    print(f"  {len(failed)} scripts failed: {failed}")
else:
    print("  ALL EXPERIMENTS COMPLETE")
print(f"  Figures: figures/")
print(f"  Results: results/")
print(f"{'='*60}")
