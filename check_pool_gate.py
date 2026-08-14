"""Gate the classification pool before spending a training run on it.

The regression operator failed because its training pool did not cover the
semantic regime the search actually operates in -- excellent on pool-like
parents (locality ratio 0.06), near-random on the functions the search contained
(0.33-0.66). Training on a pool with the same defect wastes ~7 GPU-hours.

The criterion here was wrong on the first attempt and is worth stating so it
does not get reintroduced. It required a *fraction* of each norm bucket to have
a training pair, with a threshold of 0.75 that was invented rather than measured
against the regression pool that produced a working operator. The full pool came
in at 0.51-0.65 and the gate stopped the pipeline -- but the same table showed
over a million functions in the norm 10-100 range being paired, which is
abundant signal. The fraction was the wrong statistic: it falls as the pool
grows more diverse, even while the absolute training signal in every regime
rises.

So the gate now asks the question that matters: are there *enough pairs* whose
input functions sit in the norm range a classification search occupies (measured
at 20-75)? A fraction is still reported, but only a floor low enough to catch a
genuinely broken pool.
"""
import re
import sys

# Norm buckets a classification search occupies, with the minimum number of
# paired functions each must contain. The regression pool trained a operator
# that was demonstrably local (ratio 0.06) on its own regime; these floors are
# set well below what a working pool of this size produces, to catch a pool
# that is broken rather than one that is merely diverse.
MIN_PAIRED = {"10-25": 50_000, "25-50": 50_000, "50-100": 50_000}
MIN_FRACTION = 0.20

path = sys.argv[1] if len(sys.argv) > 1 else "data/training_clf/pool_coverage.txt"

try:
    text = open(path).read()
except OSError as e:
    print(f"GATE FAIL: cannot read {path}: {e}")
    sys.exit(2)

print(text)
print("=" * 70)

# rows look like:  10-25    506,479      281,095     0.555     0.555     0.555
rows = {}
for line in text.splitlines():
    m = re.match(r"\s*([\d.e+]+-[\d.e+]+)\s+([\d,]+)\s+([\d,]+)\s+([\d.]+)",
                 line)
    if m:
        rows[m.group(1)] = (int(m.group(2).replace(",", "")),
                            int(m.group(3).replace(",", "")),
                            float(m.group(4)))

if not rows:
    print("GATE FAIL: no coverage table found, or it predates the "
          "paired-count column.\nRe-run data generation to regenerate "
          "pool_coverage.txt.")
    sys.exit(2)

ok = True
for bucket, need in MIN_PAIRED.items():
    if bucket not in rows:
        print(f"  [WARN] bucket {bucket} absent from the report")
        continue
    n, paired, frac = rows[bucket]
    bad = paired < need or frac < MIN_FRACTION
    ok = ok and not bad
    print(f"  [{'FAIL' if bad else 'PASS'}] norm {bucket:>10}  "
          f"pool {n:>10,}  paired {paired:>10,} (need >= {need:,})  "
          f"fraction {frac:.3f} (need >= {MIN_FRACTION})")

if ok:
    print("\nGATE PASS - the pool supplies substantial training signal across "
          "the norm\nrange a classification search occupies. Proceeding to "
          "training.")
    sys.exit(0)

print("\nGATE FAIL - too little training signal in the range the search lives "
      "in.\nRaising SD_MAX_THRESHOLD will NOT help: coverage was measured "
      "identical at\nSD<100, <250 and <1000 for every bucket below norm 100. "
      "The fix is more\nfunctions per synthetic problem (larger population or "
      "more generations),\nnot more problems -- 50 problems gave a 40x larger "
      "pool than 2 problems at\n*lower* coverage, because each problem "
      "explores its own region.\n\nPipeline stopped before the training run.")
sys.exit(1)
