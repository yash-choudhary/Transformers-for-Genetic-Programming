# Classification extension — final status

Branch `classification`, complete through `b9d324c`. The regression study is
frozen on `regression` and tag `regression-v7-complete`; nothing here touches it.

## The approach

Trees stay real-valued and their output is read as a **decision value**
thresholded at zero, labels encoded −1/+1. That keeps semantics continuous, so
Euclidean semantic distance keeps its meaning and the transformer machinery
transfers unchanged. Fitness is mean **logistic loss** — accuracy is
piecewise-constant and gives selection nothing between boundary crossings.
Accuracy and AUC are reported, never selected on.

## Benchmarks are constructed — say so explicitly

PMLB has exactly two 4-feature binary classification sets and neither is usable
(264 samples at 73% imbalance; 50 samples), and github.com is unreachable from
this machine so nothing else could be downloaded. The five regression
benchmarks are therefore thresholded into binary problems. Two constructions:

- **median** — y above a balanced cut. Chosen from the observed values as the
  one splitting closest to 50/50, because these targets are discrete ordinals
  and a plain median gave LEV a 21/79 split.
- **middle-band** — y inside the central third. **Not linearly representable**,
  which matters: see below.

## The finding that shaped the study

Comparing only against stdGP hid something. On the median task, **logistic
regression with five coefficients matches or beats TSGP on all five data sets**,
and random forest is *worse* than logreg on four — the signature of a near-linear
decision boundary. Thresholding a monotonic target produces exactly that, so the
median task cannot test whether flexible structure is worth anything.

The middle-band task fixes it: logreg falls to the majority baseline on ERA
(0.538 vs 0.497) and pollen (0.6603 vs 0.6603) while a random forest reaches
0.8525 on ESL against logreg's 0.6434.

**It was selected using the ML baselines, before any TSGP run touched it.** That
answers the cherry-picking objection and should be stated in the write-up.

## Results — equal budget (k=1, the paper's operator), n=30 everywhere

| task | accuracy vs stdGP | solution size | notable AUC |
|---|---|---|---|
| median | loses ERA (p=0.0015), LEV (p<1e−4); n.s. ESL, Galaxy, pollen | **smaller 5/5, 1.8–5.1×**, p to 7.4e−11 | Galaxy 0.9907 vs 0.9851 (p=0.014) |
| middle-band | loses ESL (p=0.0002); n.s. ERA, Galaxy, LEV; **wins pollen (p=0.0095)** | **smaller 5/5, 2.1–8.8×** | pollen 0.7421 vs 0.5946 (p<1e−4) |

**The honest headline is a trade, not a dominance:** 2–9× smaller solutions on
every data set at equal budget, equal accuracy on three of five, worse on two.

**The one clear win** is pollen on the middle-band task, and it *strengthened* at
equal budget (p=0.0095, against p=0.042 at k=8). stdGP there sits **below** the
majority baseline (−0.011) while TSGP is above it (+0.007). Where the non-linear
structure is hardest, standard GP finds essentially nothing.

## Did regenerating the pool for the classification regime help?

The most expensive decision in the extension — a full pool rebuild plus ~6h of
training. Verdict, both arms k=1 at n=30:

| task | clf-trained better on | significantly |
|---|---|---|
| median | 4/5 | 2/5 — pollen +0.054 (p<1e−4), ESL +0.010 (p=0.0065); worse on ERA (p=0.030) |
| middle-band | 3/5 | 1/5 — pollen +0.026 (p=0.0004) |

**Real but modest, and concentrated on pollen.** Not a broad improvement.

The motivation was sound and measured: classification drives semantic magnitude
upward without bound (population norm 19.5 → 27.3 → 57.7 across generations
against a regression control near 10), leaving the regression-trained operator
out of distribution. The regression-trained operator nonetheless transfers
better than that predicted.

## Traps worth remembering

- **SD scale differs ~40× between pools** — regression pairs sit at SD median
  0.164, classification at 6.637. The paper's SD_d = 0.1 is *below the 1st
  percentile* of the classification pool, so `TSGP_SD_DESIRED_CLF = 2.0` (its
  p25) is used. Reading classification diagnostics against the regression
  reference makes a healthy operator look broken.
- **n=10 is underpowered here.** Taking the k=1 arms from n=10 to n=30 revealed
  two significant losses that n=10 reported as "no difference".
- The **step-size floor persists in classification** (RESPONSE span 1.87), the
  same structural limit found in regression. No retraining moved it.

## Limitations to carry into the write-up

1. Benchmarks are **derived**, not native classification data.
2. The **k=8 step-control arms remain at n=10** and are underpowered by our own
   demonstration — report them as preliminary, not as findings.
3. Not compared against SLIM_GSGP, DSR or DAE-GP.
4. `TSGP_NUM_FEATURES` stayed at 4; the d=8 widening was abandoned when the
   network made wider benchmarks unreachable.

## Reproducing

```
run_classification_pipeline.cmd   # pool, gate, training, k=8 grids
run_remaining.cmd                 # middle-band grid, first k=1 arm
run_strengthen.cmd                # k=1 arms to n=30
run_final.cmd                     # transfer control at equal budget
python summarise_classification.py [--stdgp DIR --k1 DIR --transfer DIR --tsgp DIR]
python baselines_ml.py            # logistic regression / trees / forest
```

Set `TSGP_CLF_TASK=median|middle` to pick the construction. Every result file
records which one produced it, and the summary refuses to mix them.
