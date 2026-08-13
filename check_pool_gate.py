"""Gate the classification pool before spending a training run on it.

The regression operator failed because its training pool did not cover the
semantic regime the search actually operates in -- it was excellent on pool-like
parents (locality ratio 0.06) and near-random on the functions the search
contained (0.33-0.66). Training on a pool with the same defect would rebuild
that failure at the cost of ~7 GPU-hours.

A classification search was measured operating at semantic norms of roughly
20-75, so that is the range that has to be covered. Exits non-zero if it is not,
which stops the unattended pipeline rather than letting it run on regardless.
"""
import re
import sys

# Norm buckets the classification search actually occupies, and the minimum
# fraction of each that must have a training pair. The pilot at 61k functions
# managed 0.58-0.73; the full pool is ~40x denser, so 0.75 is a real gate rather
# than a formality, but not one a healthy pool should trip.
REQUIRED = {"10-25": 0.75, "25-50": 0.75, "50-100": 0.75}

path = sys.argv[1] if len(sys.argv) > 1 else "data/training_clf/pool_coverage.txt"

try:
    text = open(path).read()
except OSError as e:
    print(f"GATE FAIL: cannot read {path}: {e}")
    sys.exit(2)

print(text)
print("=" * 66)

rows = {}
for line in text.splitlines():
    m = re.match(r"\s*([\d.e+]+-[\d.e+]+)\s+([\d,]+)\s+([\d.]+)", line)
    if m:
        rows[m.group(1)] = (int(m.group(2).replace(",", "")), float(m.group(3)))

if not rows:
    print("GATE FAIL: no coverage table found. Did data generation finish?")
    sys.exit(2)

ok = True
for bucket, need in REQUIRED.items():
    if bucket not in rows:
        print(f"  [WARN] bucket {bucket} absent from the report")
        continue
    n, cov = rows[bucket]
    verdict = "PASS" if cov >= need else "FAIL"
    if cov < need:
        ok = False
    print(f"  [{verdict}] norm {bucket:>10}  n={n:>9,}  coverage {cov:.3f} "
          f"(need >= {need})")

if ok:
    print("\nGATE PASS - the pool covers the regime a classification search "
          "operates in.\nProceeding to training.")
    sys.exit(0)

print("\nGATE FAIL - the pool does not cover the norm range the search lives "
      "in.\nTraining on it would rebuild the out-of-distribution failure "
      "diagnosed on\nthe regression side. Raising SD_MAX_THRESHOLD will NOT "
      "help: coverage was\nmeasured to be identical at SD<100, <250 and <1000 "
      "for every bucket below\nnorm 100. The fix is a denser pool - more "
      "synthetic problems - or a change\nto how pairs are formed.\n\n"
      "Pipeline stopped before the training run.")
sys.exit(1)
