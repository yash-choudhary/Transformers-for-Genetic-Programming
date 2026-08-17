# TSGP Replication — Complete Project Record

**A standalone source of truth.** Everything needed to understand, defend, or
rebuild this study is in this one file: every result, every parameter, every
defect found, every deviation from the paper, and what can and cannot be
recovered if the repository is lost. Numbers were recomputed from the archived
run files on 2026-08-17, not transcribed from earlier documents.

Read §14 first if the repository is already gone.

---

## Contents

1. Identity and provenance
2. The claim under test, and the outcome
3. Environment and hardware
4. Complete parameter inventory
5. Code map
6. Pipeline: stages, commands, costs
7. Datasets
8. Results — regression
9. Results — search dynamics and operator diagnostics
10. Results — classification extension
11. The diagnostic investigation: eight hypotheses
12. Defects found and fixed
13. Deviations from the paper that remain
14. What is not in git, and how to regenerate it
15. The report and its build system
16. Verification status: what is proven, what is not
17. Open questions

---

## 1. Identity and provenance

| | |
|---|---|
| Study | Independent replication of Transformer Semantic Genetic Programming (TSGP) |
| Paper replicated | Anthes, P., Sobania, D. and Rothlauf, F., 2025. *Transformer semantic genetic programming for symbolic regression.* arXiv:2501.18479 |
| Authors | Yash Choudhary (25211648), Pranay Nagpure (25203920), Raj Jigneshbhai Barot (25213741) |
| Supervisor | Assoc. Prof. Miguel Nicolau |
| Institution | Michael Smurfit Graduate School of Business, University College Dublin — MSc Business Analytics |
| Repository | https://github.com/yash-choudhary/Transformers-for-Genetic-Programming |
| Branch of record | `classification` (pushed; tracks `origin/classification`) |
| Regression study frozen at | branch `regression`, tag `regression-v7-complete` |
| Report file | `MIS41170-Capstone-FINAL.docx`, built by `build_report.py` |

**Note on the title page:** the name appears as "Raj Jignesh Bhai Barot" on the
title page and "Raj Jigneshbhai Barot" in the preface. One is wrong; unresolved.

### Commit history (most recent first)

```
985c533 2026-08-17 report: populate the end matter and fix what the proofread found
0c18abd 2026-08-17 figures: draw the genetic programming loop and its operators
aa2d0fb 2026-08-17 report: fact-check every figure against the archived runs
40db51f 2026-08-16 figures: add transformer diagrams; fix overlaps and thin-data plots
8334da3 2026-08-16 figures: script-generated plots plus report builder
aacc244 2026-08-16 classification: final status document
b9d324c 2026-08-16 classification: final equal-budget matrix at n=30
42fe195 2026-08-15 classification: final run — transfer control at equal budget and n=30
3c629e8 2026-08-15 classification: equal-budget arms at n=30
3f931fe 2026-08-15 classification: script to close the two remaining weaknesses
8aa80fc 2026-08-15 classification: complete — equal-budget arm and middle-band task
3b39ae3 2026-08-15 classification: ML baselines expose a benchmark flaw; add middle-band task
```

---

## 2. The claim under test, and the outcome

**The paper's claim.** A transformer, trained once on ~5M semantically similar
expression pairs and conditioned on a desired semantic distance, replaces
crossover and mutation as the sole variation operator in an otherwise standard
GP loop, and beats standard GP, SLIM_GSGP, DSR and DAE-GP on five PMLB
benchmarks while keeping solutions small.

**Outcome, in one paragraph.** The standard GP baseline reproduces the
published figures within 0.010 on four of five datasets, which licenses the
comparison. TSGP does not reproduce: it has the worse median on all five
datasets against our own standard GP, significantly so on three. The cause was
localised to the operator's *step size*: it cannot produce an offspring at the
semantic distance the method requests (floor ≈ 0.7 against a requested 0.1),
and unlike standard GP it cannot refine that step as the population converges.
Supplying the missing step-size control externally (best of k=8 candidates)
beats our k=1 operator significantly on all five datasets and recovers the
paper's direction on three, at 8× the model evaluations.

**Extension (not part of the replication).** The method was applied to binary
classification, which the paper does not address. TSGP produces solutions 1.8
to 8.8× smaller than standard GP on every dataset at equal budget, with
statistically indistinguishable accuracy on three of five in each of two task
constructions, and one clear win on the hardest non-linear problem.

---

## 3. Environment and hardware

| | |
|---|---|
| Machine | Windows 11 Home Single Language, 31.8 GB RAM |
| GPU | NVIDIA RTX 2070, 8 GB |
| `capstone-gpu` env | TensorFlow 2.10.1 **GPU**, numpy 1.23.5, CUDA 11.2, cuDNN 8.1 — transformer training only |
| `capstone2` env | TensorFlow 2.15.1 **CPU**, numpy 1.26.4 — GP search, experiments, diagnostics |
| `capstone` env | python-docx, pymupdf, matplotlib, numpy 1.26.4 — figures and report build |
| Frameworks | DEAP (evolution), FAISS (nearest neighbours), TensorFlow/Keras (model) |

### Environment traps that cost real time

- **The GPU was silently unused for the whole first phase of the project.**
  conda puts the CUDA DLLs in `%CONDA_PREFIX%\Library\bin`, which only reaches
  PATH via `conda activate`. Calling `python.exe` by its full path starts a
  CUDA-built TensorFlow that finds no device and falls back to CPU: **3.7 s per
  batch against ~0.15 s**, i.e. ~20 hours per epoch instead of ~49 minutes.
  Use `gpu_python.cmd`, and verify with `nvidia-smi`, never by TF not erroring.
- **Two TensorFlow installs in one env shadow each other.** `capstone2` had
  both `tensorflow-gpu 2.10.1` and `tensorflow 2.15.1`; the later install
  overwrote the shared package directory and broke GPU training.
- **The conda CUDA runtime lacks the compiler components JIT expects** — fixed
  by `jit_compile=False` on the optimiser.
- **Windows Update terminates long unattended runs.** Active hours were
  08:00–18:00; two multi-hour jobs were killed. Training resumes from
  per-epoch checkpoints, but AdamW optimiser moments are *not* checkpointed,
  so a resumed run restarts them.

---

## 4. Complete parameter inventory

`Paper` = specified by Anthes et al.; `Ours` = the paper is silent and we chose;
`Follows` = fixed by other choices.

### Search and evaluation (both TSGP and the baseline)

| Parameter | Value | Source |
|---|---|---|
| Population size | 100 | Paper |
| Generations | 50 | Paper |
| Initialisation | ramped half-and-half, depth 2–5 | Paper |
| Max tree depth | 17 | Paper |
| Selection | tournament, size 5 | Paper |
| Function set | add, sub, mul, protected div | Paper |
| Terminal set | x0–x3, ERC | Paper |
| ERC grid | −0.5 to 0.5, step 0.1 (11 values) | Paper |
| Fitness | RMSE (regression) / mean logistic loss (classification) | Paper / Ours |
| Crossover | subtree, p=0.9, 10% terminal bias | Paper |
| Mutation | subtree, p=0.1, new subtree depth 0–2 | Paper |
| Runs per dataset | 30 | Paper |
| Split | 50/50, features and target standardised, scaler fit on train only | Paper |
| Replacement | generational, no elitism | Ours |
| Protected division | returns 1.0 when \|denominator\| < 1e-6 | Paper (value ours) |

### Data generation and the model

| Parameter | Value | Source |
|---|---|---|
| Synthetic problems | 50, y = Xβ + ε, standardised, varying noise | Paper |
| Datagen population | 2,000 | Paper |
| Datagen generations | 50 (**was 100 — see §12**) | Paper |
| Datagen selection | double tournament (fitness + parsimony) | Paper |
| Samples per synthetic problem | 200 | Ours |
| Probe inputs for semantics | 100, standard normal | Ours |
| Neighbours per function, k | 3 | Paper |
| Pair filter | 0 < SD < 100 | Paper |
| Target training pairs | 5,000,000 | Paper |
| Encoder / decoder layers | 2 / 2 | Paper |
| Attention heads | 8 | Paper |
| Hidden dimension | 128 | Paper |
| Feed-forward dimension | 512 (4× hidden) | Ours |
| Dropout | 0.1 | Ours |
| Max sequence length | 100 tokens | Paper |
| Vocabulary | 22 tokens (3 special + 4 functions + 4 variables + 11 constants) | Follows |
| Parameters | ~934,000 | Follows |
| Optimiser | AdamW, lr 1e-3 | Paper |
| Weight decay | 0.004 | Ours |
| Epochs | 8 | Paper |
| Batch size | 256 | Ours |
| Loss | masked cross-entropy (padding excluded) | Ours |
| SD conditioning | scalar → learned linear projection → added to token embeddings in **both** encoder and decoder | Paper (mechanism ours) |

### The operator at search time

| Parameter | Value | Source |
|---|---|---|
| Desired distance SD_d, regression | 0.1 | Paper |
| Desired distance SD_d, classification | **2.0** (classification pool's p25) | Ours — see §10 |
| Sampling temperature | 1.0 (no scaling) | Ours |
| Candidates per parent, k | 1 = the paper's operator; 8 = the extension | Paper / extension |
| Step-control target (k=8 arm) | decays 1.0 → 0.02 × ‖y_train‖ across the run | Extension, tuned on ESL at n=10 |

### Environment-variable switches

| Variable | Effect |
|---|---|
| `TSGP_NUM_FEATURES` | terminal-set width (stayed 4; a d=8 widening was abandoned) |
| `TSGP_CLF_TASK` | `median` or `middle` label construction |
| `TSGP_NO_DIVISION=1` | drops protected division from the primitive set (§8.5 diagnostic only) |

---

## 5. Code map

All under `tsgp/`:

| Module | Responsibility |
|---|---|
| `config.py` | every parameter above, in one place |
| `primitives.py` | primitive set, protected division, vectorised evaluation, `evaluate_semantics_fast` |
| `tokenizer.py` | tree ↔ prefix tokens ↔ integer ids |
| `datasets.py` | PMLB fetch/cache, 50/50 split, standardisation |
| `fetch_datasets.py` | stage 1 entry point |
| `data_generator.py` | synthetic problems, GP pool collection, FAISS pairing |
| `transformer_model.py` | encoder-decoder, SD conditioning |
| `train_transformer.py` | teacher-forced training, per-epoch checkpoints |
| `syntax_control.py` | grammar-constrained decoding masks |
| `tsgp_search.py` | the learned operator; batched sampler; step-control (k, annealing) |
| `classification.py` | decision-value objective, logistic loss, both label constructions |
| `classification_datagen.py` | classification-regime pool |
| `experiment_units.py` | one (dataset, method, run) unit, deterministic seeding, atomic write |
| `run_single.py` / `run_experiments.py` | one unit / the restartable grid |
| `aggregate_results.py` | medians and the comparison table |
| `instrument.py` | per-generation logging (reproduces the paper's Figs 2–4) |
| `operator_diagnostics.py` | locality, response, regime gates |
| `step_floor.py` | the achieved-vs-requested distance sweep and min-of-k |

Top level: `run_classification.py`, `summarise_classification.py`,
`baselines_ml.py`, `check_pool_gate.py`, `sweep_step_control.py`,
`make_figures.py`, `build_report.py`, and the `.cmd` drivers.

---

## 6. Pipeline: stages, commands, costs

Every stage skips work already on disk; repeating a command resumes it.

| # | Command | Produces | Cost |
|---|---|---|---|
| 1 | `python -m tsgp.fetch_datasets` | `data/pmlb_cache/` | minutes, needs network once |
| 2 | `python -m tsgp.data_generator --output data/training` | 5M pairs (1.7 GB) | ~3 h |
| 3 | `python -m tsgp.train_transformer --data data/training --checkpoints checkpoints_adamw` | operator weights | 8 × ~49 min = ~6.5 h on GPU |
| 4 | `python -m tsgp.run_experiments --weights checkpoints_adamw/tsgp_final.npy --output results_v7` | the 300-run grid | ~2 h |
| 5 | `python -m tsgp.run_experiments ... --step-k 8 --step-anneal --step-frac-start 1.0 --step-frac-end 0.02 --output results_anneal_b` | step-control arm | ~10.4 h |
| 6 | `set TSGP_NO_DIVISION=1` then `python -m tsgp.run_experiments --methods stdgp --output results_nodiv` | primitive-set test | ~3.4 min |
| 7 | `run_classification_pipeline.cmd`, `run_remaining.cmd`, `run_strengthen.cmd`, `run_final.cmd` | the classification chapter | pool ~2 h + training ~6 h + grids |
| 8 | `python summarise_classification.py`, `python baselines_ml.py` | classification tables, ML baselines | minutes |
| 9 | `python make_figures.py` then `python build_report.py` | 14 figures, the .docx | ~3 min |

**Total to rebuild from nothing: roughly 22 hours of machine time**, dominated
by stages 2, 3, 5 and 7.

### Performance engineering (both verified, not assumed)

- **Batched sampler.** Profiling showed dispatch, not arithmetic: 69.4 ms per
  decode step eager vs 7.3 ms under `tf.function`. The encoder was also re-run
  every decode step, and each pass carried one sequence. Fixing all three
  (encoder hoisted, both passes graph-compiled, whole population decoded in
  lockstep) took **1374.7 ms → 84.0 ms per individual** (~16×); a TSGP run
  116 min → ~4.2 min; the grid 291 h → ~10.5 h.
  *Equivalence:* 100 parents, sequential vs batched — 100/100 valid trees both,
  median offspring size 11 both, **KS D = 0.110, p = 0.583**.
- **Vectorised evaluation** (whole matrix instead of a row loop):
  **bit-identical on 400/400** random trees.
- **Vectorised token sampler** (Gumbel-max over all active sequences):
  **KS D = 0.022, p = 0.846**; median offspring size 37 vs 37.
- `evaluate_semantics_fast` (prefix-stack interpreter): exact, but only 2.1×.
  `gp.compile` was **not** the bottleneck; the decode loop is O(T²) with no KV
  cache. That is where the remaining headroom is.

---

## 7. Datasets

Five PMLB benchmarks, exactly the paper's selection. PMLB requires
identifier-prefixed keys, not the short names the paper prints.

| Name | PMLB key | Instances | Features |
|---|---|---|---|
| ERA | `1030_ERA` | 1,000 | 4 |
| ESL | `1027_ESL` | 488 | 4 |
| Galaxy | `690_visualizing_galaxy` | 323 | 4 |
| LEV | `1029_LEV` | 1,000 | 4 |
| pollen | `529_pollen` | 3,848 | 4 |

All five are natively four-dimensional, so nothing is discarded by the
operator's four-variable terminal set. Four are real-world; **pollen is
synthetic**, per the paper's own description.

Classification benchmarks are **constructed** from these by thresholding the
target, because PMLB contains exactly two four-feature binary classification
sets and neither is usable (264 samples at 73% imbalance; 50 samples).

- **median split** — a cut chosen from observed values to divide the training
  half as evenly as the discrete ordinal targets allow. A plain median gives
  LEV a 21/79 split.
- **middle band** — positive class = central third of the training
  distribution. Not linearly representable. Introduced *after* the ML
  baselines showed the median split has a near-linear boundary, and **selected
  on the baselines alone, before any TSGP run touched it**.

---

## 8. Results — regression

### 8.1 The paper's published values (Table 2 and Table 3 of arXiv:2501.18479)

| Dataset | TSGP | stdGP | SLIM | DSR | DAE | TSGP size | stdGP size |
|---|---|---|---|---|---|---|---|
| ERA | **0.797** | 0.817 | 0.810 | 0.852 | 0.904 | 72 | 60 |
| ESL | **0.379** | 0.502 | 0.418 | 0.507 | 0.595 | 73 | 12 |
| Galaxy | 0.327 | 0.337 | **0.305** | 0.434 | 0.468 | 64 | 48 |
| LEV | **0.672** | 0.703 | 0.681 | 0.773 | 0.842 | 69 | 50 |
| pollen | 0.518 | **0.514** | 0.569 | 0.750 | 0.752 | 58 | 37 |

The paper reports TSGP significantly better than stdGP on **ERA, ESL and LEV
only**; its Galaxy and pollen differences are not significant.

### 8.2 Our grid — median test RMSE, n=30, each run drawing its own split

| Dataset | TSGP (k=1) | stdGP | TSGP+control (k=8) | stdGP no-division |
|---|---|---|---|---|
| ERA | 0.8442 | 0.8070 | 0.8253 | 0.8064 |
| ESL | 0.4748 | 0.4531 | 0.4084 | 0.4157 |
| Galaxy | 0.3501 | 0.3322 | 0.3023 | 0.3184 |
| LEV | 0.7540 | 0.6930 | 0.7198 | 0.6955 |
| pollen | 0.6324 | 0.5075 | 0.5824 | 0.4705 |

### 8.3 Median solution size (nodes)

| Dataset | TSGP (k=1) | stdGP | TSGP+control | stdGP no-division |
|---|---|---|---|---|
| ERA | 15 | 48 | 17 | 71 |
| ESL | 23 | 27 | 25 | 45 |
| Galaxy | 21 | 48 | 21 | 66 |
| LEV | 16 | 38 | 20 | 70 |
| pollen | 24 | 55 | 18 | 63 |

Ours are 15–24 nodes against the paper's 58–73. **No intervention moved this.**

### 8.4 Significance (Wilcoxon rank-sum unless stated)

| Dataset | TSGP vs stdGP | k=8 vs stdGP | k=8 vs k=1 | Δ vs paper (stdGP) |
|---|---|---|---|---|
| ERA | p=0.0020 → stdGP | p=0.337 | p=0.0199 | −0.010 |
| ESL | p=0.859 → n.s. | p=0.0105 → TSGP | p=3.70e-06 | −0.049 |
| Galaxy | p=0.497 → n.s. | p=0.0120 → TSGP | p=6.16e-05 | −0.005 |
| LEV | p=0.00079 → stdGP | p=0.135 | p=0.00035 | −0.010 |
| pollen | p=0.0024 → stdGP | p=0.0256 → stdGP | p=9.90e-07 | −0.007 |

Baseline validity: stdGP within **0.010 on four of five**; ESL out by 0.049.
Direction agrees with the paper on **one** dataset (pollen) at k=1, and on
**three** (ESL, Galaxy, pollen) with step control.

### 8.5 Primitive-set test — standard GP without protected division

Rerun 2026-08-17 as a proper arm (`results_nodiv/`, `TSGP_NO_DIVISION=1`),
30 runs per dataset, seeded to pair run-for-run with `results_v7`.

| Dataset | with division | without | Δ | paired Wilcoxon p |
|---|---|---|---|---|
| ERA | 0.8070 | 0.8064 | −0.0007 | 0.198 |
| ESL | 0.4531 | 0.4157 | −0.0374 | 0.053 |
| Galaxy | 0.3322 | 0.3184 | −0.0138 | 0.109 |
| LEV | 0.6930 | 0.6955 | +0.0025 | 0.416 |
| pollen | 0.5075 | 0.4705 | −0.0370 | 0.177 |

**Nominally better on four of five, none significant.** This is the null result
that justified *not* regenerating the training pool to encourage division:
a primitive that can be removed without measurable cost is not one whose
absence handicaps the operator.

Division usage in best solutions (final grid): TSGP 3–37%, stdGP 33–53%.
Division is 0.7% of tokens in the training targets against ~16% each for
add/sub/mul.

**Supersedes** an earlier screening run whose per-run files were not kept and
whose quoted deltas (0.026 ESL, 0.033 Galaxy, 0.027 LEV, 0.039 pollen) do
**not** reproduce.

---

## 9. Results — search dynamics and operator diagnostics

### 9.1 Per-generation trace, ESL (one instrumented run per method, same seeds)

Target-vector magnitude ‖y‖ = 15.6. Source: `results_instr_base/*run00.json`.

| | Gen 1 | 5 | 10 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|
| stdGP step distance | 28.2 | 13.4 | 8.1 | 2.7 | 1.6 | **1.0** |
| TSGP step distance | 22.0 | 34.5 | 18.5 | 16.8 | 23.1 | **25.7** |
| stdGP best size | 9 | 11 | 17 | 39 | 39 | **103** |
| TSGP best size | 9 | 5 | 21 | 21 | 21 | **21** |
| stdGP best RMSE | 0.707 | 0.530 | 0.446 | 0.422 | 0.408 | **0.393** |
| TSGP best RMSE | 0.506 | 0.463 | 0.448 | 0.448 | 0.448 | **0.448** |

**This is the central finding.** Standard GP's step anneals 28.2 → 1.0 as its
population converges — subtree crossover between similar parents produces a
near-parent offspring for free. TSGP's step never falls: it stays 16–34,
larger than the entire target signal, because it draws from the same
conditional distribution at generation 50 as at generation 1. The search has
an exploration phase and no exploitation phase.

### 9.2 Search stalls (30-run grid, `results_v7`)

| Dataset | last improving generation (median) | improving generations (median) |
|---|---|---|
| ERA | 10 | 3 |
| ESL | 15 | 4 |
| Galaxy | 18.5 | 5 |
| LEV | 25 | 4 |
| pollen | 29 | 4.5 |

stdGP for comparison: last improvement at generation 45–49, with 8–17
improving generations.

### 9.3 The step-size floor (`diagnostics/stepfloor_linear_final.json`, 256 parents)

| Requested SD_d | 0.0001 | 0.001 | 0.01 | 0.1 | 1.0 |
|---|---|---|---|---|---|
| Achieved distance (median) | 0.664 | 0.766 | 0.699 | 0.869 | 1.925 |

Below a request of ~0.1 the achieved distance does not fall. **The method's
operating point (0.1) lies beneath the operator's floor (~0.7).** The
requested step size was never actually honoured, in any version of this work.

Min-of-k, same file: k=1 → 0.79, k=2 → 0.42, k=4 → 0.24, k=8 → 0.15,
k=16 → 0.095. **This is why step control works**: taking the nearest of k
draws goes underneath the floor.

### 9.4 Locality control (`diagnostics/baseline_adamw.json`, 256 pool parents)

| Requested SD_d | 0.01 | 0.1 | 1.0 | 10.0 | 50.0 |
|---|---|---|---|---|---|
| distance to own parent | 0.760 | 0.809 | 1.673 | 6.120 | 35.96 |
| distance to unrelated parent | 13.16 | 13.21 | 13.86 | 14.74 | 35.52 |
| ratio | 0.058 | **0.061** | 0.121 | 0.415 | 1.012 |

At the operating point offspring land **16× closer to their own parent than to
an unrelated one**. The operator is strongly local, and the conditioning *does*
respond (0.81 → 1.67 → 6.12 across a 7.6-fold span). Both facts overturn
earlier claims — see §11.

**Regime gate** (the check the earlier drafts never ran): on functions that
ramped half-and-half produces at generation 0, whose semantic norms are
10–100× the pool's, the ratio degrades to 0.64 (gate 0.6, **fail**). The
operator scores well on pool parents and near-randomly on the population the
search actually starts from.

### 9.5 Measurement noise

The same checkpoint measured 31.7 / 34.9 / 36.2 / 38.1 across repeated
evaluations — about ±20%. **Any achieved-SD difference below ~20% in this
record is not real.** The conclusions above rest on effects far larger, or on
paired comparisons under identical seeds.

---

## 10. Results — classification extension

### 10.1 Design

- Trees stay **real-valued**; output read as a **decision value**, class =
  sign, labels −1/+1. This keeps semantics continuous so Euclidean semantic
  distance retains meaning and the whole transformer apparatus transfers. A
  label-emitting tree would make semantics a binary vector, whose distance is
  a flat Hamming count — the k-NN pairing would then match near-arbitrary
  expressions.
- Fitness = **mean logistic loss** log(1 + exp(−y·f(x))). Accuracy is
  piecewise constant and gives selection no gradient. Accuracy and AUC are
  reported, never selected on.
- **SD_d rescaled to 2.0.** SD is an absolute Euclidean norm, so its scale
  follows the pool's semantic magnitudes. Measured: regression pairs
  p1 0.001 / p25 0.049 / **p50 0.164** / p75 0.667; classification pairs
  p1 0.200 / **p25 2.156** / p50 6.637 / p75 11.323 — roughly 40× apart. The
  paper's 0.1 sits mid-range for regression but **below the 1st percentile**
  for classification. 2.0 is the classification pool's p25. Both TSGP arms
  (retrained and transfer) use it, so comparisons are internally consistent.
  *This is a deviation from the paper's stated value and is recorded as one.*

### 10.2 Median-split task, equal budget (k=1), n=30

| Dataset | TSGP acc | stdGP acc | p | TSGP size | stdGP size | ratio | p (size) | TSGP AUC | stdGP AUC | majority |
|---|---|---|---|---|---|---|---|---|---|---|
| ERA | 0.6870 | 0.7020 | **0.0015** | 9 | 31 | 3.4× | 7.4e-05 | 0.7637 | 0.7672 | 0.596 |
| ESL | 0.9324 | 0.9344 | 0.918 | 29 | 78 | 2.7× | 6.7e-10 | 0.9836 | 0.9816 | 0.547 |
| Galaxy | 0.9444 | 0.9444 | 0.842 | 27 | 137 | 5.1× | 7.4e-11 | **0.9907** | 0.9851 | 0.512 |
| LEV | 0.8060 | 0.8330 | **<1e-4** | 15 | 27 | 1.8× | 1.2e-04 | 0.8720 | 0.8972 | 0.632 |
| pollen | 0.8381 | 0.8363 | 0.647 | 23 | 65 | 2.8× | 4.7e-08 | 0.9187 | 0.9184 | 0.492 |

Galaxy AUC 0.9907 vs 0.9851 is significant (p=0.014).

### 10.3 Middle-band task, equal budget (k=1), n=30

| Dataset | TSGP acc | stdGP acc | p | TSGP size | stdGP size | ratio | p (size) | TSGP AUC | stdGP AUC | majority |
|---|---|---|---|---|---|---|---|---|---|---|
| ERA | 0.5850 | 0.5890 | 0.762 | 9 | 79 | 8.8× | 3.6e-10 | 0.6258 | 0.6310 | 0.504 |
| ESL | 0.7705 | 0.8299 | **0.0002** | 12 | 70 | 5.8× | 6.3e-08 | 0.8365 | 0.9011 | 0.516 |
| Galaxy | 0.8704 | 0.8735 | 0.156 | 28 | 98 | 3.5× | 1.7e-07 | 0.9325 | 0.9405 | 0.660 |
| LEV | 0.6880 | 0.6930 | 0.264 | 10 | 21 | 2.1× | 1.8e-02 | 0.6051 | 0.6262 | 0.682 |
| pollen | **0.6783** | 0.6596 | **0.0095** | 11 | 42 | 3.8× | 9.1e-03 | **0.7421** | 0.5946 | 0.657 |

**The one clear win.** On pollen under the middle band, standard GP does not
beat the majority baseline at all while TSGP does; AUC 0.7421 vs 0.5946
(p≈1e-5). Where the non-linear structure is hardest, standard GP finds
essentially nothing.

### 10.4 Was retraining on classification data necessary? (transfer control)

Both arms k=1, n=30. "Transfer" = the **regression-trained** operator.

| Task | clf-trained better on | significant |
|---|---|---|
| median split | 4 of 5 | 2 of 5 — pollen +0.054 (p=2.3e-10), ESL +0.010 (p=0.0065); **worse** on ERA (p=0.030) |
| middle band | 3 of 5 (Galaxy tied) | 1 of 5 — pollen +0.026 (p=0.00036) |

**Real but modest, and concentrated on pollen.** The motivation was measured,
not assumed: under logistic loss nothing bounds output magnitude, so the
population's median semantic norm climbs 19.5 → 27.3 → 57.7 across
generations against a regression control near 10 — leaving a regression-trained
operator out of distribution for most of the run. It nonetheless transfers
better than that predicted.

**The step-size floor persists in classification** (response span 1.87,
`diagnostics/clf_final.json`). A different task, a different pool, a 40×
different semantic scale — and the floor is unchanged. This is the strongest
evidence that the floor is structural, and it is also the control that covers
the un-regenerated regression pool (§13).

### 10.5 Standard classifiers on the same splits (n=30)

| Task | Dataset | logreg | rf-100 | tree-d3 | tree-full | majority |
|---|---|---|---|---|---|---|
| median | ERA | **0.7150** | 0.6980 | 0.7090 | 0.7000 | 0.5940 |
| median | ESL | **0.9344** | 0.9160 | 0.8975 | 0.9078 | 0.5471 |
| median | Galaxy | 0.9383 | 0.9259 | 0.9043 | 0.9074 | 0.4815 |
| median | LEV | **0.8290** | 0.8110 | 0.8070 | 0.8110 | 0.6240 |
| median | pollen | 0.8373 | 0.8202 | 0.7497 | 0.7669 | 0.5005 |
| middle | ERA | 0.5380 | 0.6230 | **0.6250** | 0.6200 | 0.4970 |
| middle | ESL | 0.6434 | **0.8525** | 0.7869 | 0.8258 | 0.5020 |
| middle | Galaxy | 0.8519 | **0.9259** | 0.8889 | 0.9198 | 0.6605 |
| middle | LEV | 0.7240 | **0.7790** | 0.7480 | 0.7730 | 0.6840 |
| middle | pollen | 0.6603 | **0.7058** | 0.6601 | 0.6583 | 0.6603 |

**This changed the design of the study.** On the median split, logistic
regression with five coefficients matches or beats TSGP on all five datasets
(significantly on ERA and LEV, indistinguishable on the other three) and a
random forest is worse than logreg on all five — the signature of a
near-linear boundary. Thresholding a monotonic target produces exactly that,
so the median split cannot test whether structural flexibility is worth
anything. Hence the middle band, where logreg collapses to the majority
baseline on ERA and pollen while a forest reaches 0.8525 on ESL.

### 10.6 Preliminary k=8 classification arms (n=10 — not relied upon)

median split: ERA 0.7010, ESL 0.9262, Galaxy 0.9383, LEV 0.8220, pollen 0.8410.
middle band: ERA 0.6020, ESL 0.8217, Galaxy 0.8735, LEV 0.6970, pollen 0.6809
(pollen win p=0.042, weaker than the equal-budget k=1 result).
Raising both arms to n=30 would cost ~7 h; the budget went to the
equal-budget arms instead. **n=10 demonstrably hides real differences** —
taking the k=1 arms from 10 to 30 runs surfaced two significant losses that
n=10 reported as "no difference".

---

## 11. The diagnostic investigation: eight hypotheses

All eliminated. This is what turns a bare non-replication into a diagnosed one.

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 1 | Optimiser (Adam vs AdamW) | Eliminated | Retrained with AdamW. ERA, 10 paired seeds: median 0.9480 → 0.9183, 6/10 improved, **Wilcoxon signed-rank p = 0.2324**. Median size fell 20 → 15, wrong direction. The one unambiguous deviation, and fixing it changed nothing. |
| 2 | Undertraining / batch size | Eliminated | Batch-32 probe (156,250 updates ≈ our whole run) gave *smaller* offspring. Median offspring size 11 at **every one of 8 epochs**, no trend. Teacher-forced accuracy plateaus at epoch 3 (ceiling ≈ 0.7997 vs copy baseline ≈ 0.58). |
| 3 | Syntax control distorting lengths | Eliminated | The mask removes **0.0004** of probability mass. |
| 4 | Sampling temperature | Eliminated (harmful) | T=0.5 significantly worse on **5/5** datasets; ERA degrades monotonically as T falls. Temperature is not a calibration knob — it is the operator's only source of randomness, and sharpening collapses population diversity. |
| 5 | Autoregressive drift | Eliminated | Greedy decoding lands at the same distance as sampling (10.90, size 44, 181/200 valid). |
| 6 | Out-of-distribution parents | Eliminated | Within-run distance settles at the in-distribution floor; the population does reach the regime the operator was trained for. |
| 7 | SD encoding (numerics) | Eliminated | A higher-capacity binned encoding (64 log10 bins) hits a floor of the same kind at matched training. The limit is how coarsely one token maps onto behaviour, not how the request is represented. |
| 8 | Training SD range (deliberate deviation) | Eliminated | Training only on SD ≤ 0.2 (2.7M of 5M pairs) made achieved distance **worse**: 50.1 vs 38.1 baseline. *This cuts in the paper's favour* — a departure from its specification degraded the operator. |

**Root cause as far as the evidence reaches:** the operator's semantic step has
a floor well above the requested distance (§9.3), and it cannot anneal that
step as the population converges (§9.1). No parameter the paper specifies
alters this, and the paper publishes no comparable diagnostic — the gap sits
in quantities it does not report.

### Claims withdrawn from earlier drafts (v1–v6), each by adding a control

| Earlier claim | What controlled measurement showed | Verdict |
|---|---|---|
| "The SD conditioning is inert" | Monotonic across a 7.6-fold span: 0.1/1/10 → 0.81/1.67/6.12 | Withdrawn |
| "The failure is token-level accuracy on a one-to-many task" | Offspring land 16× closer to their own parent than to an unrelated one (ratio 0.061) | Withdrawn |
| "Solutions are 3–4× too small; the operator cannot build structure" | Sampled offspring size median 35 = training-target median 35, exactly | Withdrawn |

**The transferable lesson:** every one of these was a measurement without a
control condition. Tree size in prefix notation is a critical branching process
with mean 1/(1−2p): p=0.486 gives mean size 35, p=0.476 gives 21. A one-point
change in a single probability halves the tree. It is a hypersensitive,
low-information metric and it misled three report versions.

---

## 12. Defects found and fixed

### In the research code

1. **Pinned train/test split** (`datasets.py`, `random_state=42`) — all 30
   "independent" runs of a dataset shared one 50/50 split. **The single
   largest error in the study.** Fixing it moved 9 of 10 dataset-method cells
   closer to the published values (ERA TSGP 0.9488 → 0.8442; ERA stdGP
   0.8865 → 0.8070 against a published 0.817). The whole grid was rerun.
2. **Truncated pair generation** (`data_generator.py`) — `find_semantic_pairs`
   early-returned at exactly 5M pairs while walking the pool in problem order,
   so later synthetic problems contributed **zero** training pairs. Now
   shuffles first. *Not reflected in the trained model — see §13.*
3. **`DATAGEN_GP_GENERATIONS` 100 → 50** per Table 1. *Also not reflected in
   the trained model — see §13.*
4. **Dataset loading** — the code used the paper's short names; PMLB requires
   identifier-prefixed keys. Every run aborted on the first fetch. Added a
   local cache with retry/backoff after a transient `WinError 10065` killed a
   multi-hour run.
5. **No checkpointing** — all 300 runs lived in one script, so any failure
   discarded everything. Restructured into independent, deterministically
   seeded, atomically written units.
6. **The GPU was never used** (see §3).

### Paper-fidelity corrections

7. **Best-so-far reporting.** §4.2 of the paper says the best solution *on the
   training set* is used for test error — the best at any point. The code
   reported the best of the final population; with generational replacement
   and no elitism the best is routinely destroyed. On ERA: 0.9657 vs 0.8512.
   Fixed with an archive that never re-enters the population.
8. **Internal node bias.** §4.1 specifies a 10% terminal bias in subtree
   crossover. `GP_INTERNAL_NODE_BIAS = 0.1` existed but was never read — both
   toolboxes used unbiased `cxOnePoint`. Now `cxOnePointLeafBiased`.
9. *Rejected as unfaithful:* adding elitism. Table 1 lists selection only, so
   generational replacement without elitism is the faithful reading.

### In the report (fact-check, 2026-08-17)

Recomputed from the archived runs; every one of these was wrong in the draft:

- RQ1 said TSGP "loses on four of five" — it is five by median, three
  significantly.
- Chapter 1's contributions still asserted the **withdrawn** token-error
  diagnosis.
- "Two datasets that were significant losses" — only Galaxy changed verdict;
  ESL was already n.s. (p=0.906) on the old grid.
- Stall statistics 15–32 and 2.5–6 were from the superseded grid; the final
  grid gives 10–29 and 3–5.
- Table 7.3's sweep 1.15/1.55/6.44 (5.6-fold) does not exist in the shipped
  diagnostics, which give **0.81/1.67/6.12 (7.6-fold)**.
- Division usage "0 to 17%" of TSGP solutions — the final grid gives **3–37%**.
- §7.8's primitive-set deltas did not reproduce (see §8.5).
- "103 nodes at 0.443" was a single run; medians are 27 nodes at 0.453.
- "trained four model variants" — five.
- Table 7.4's p-values were labelled paired; they are rank-sum.
- Two figure cross-references pointed at 6.2 (size) where 6.3 (step) was meant.
- Chapter 8's SD_d rescale to 2.0 was undisclosed.
- pollen described as real-world; the four-feature restriction described as
  discarding features when all five datasets are natively 4-D.
- Chapter cross-references not renumbered after the classification chapter was
  inserted (Threats 8→9, Conclusion 9→10).
- Report's own §3.2/4.1/4.2 cited where the paper's were meant.

### In the report build system

- **`clear_section` deleted the body's `sectPr`** when clearing the last
  section, and python-docx wrote a default one back. Cost the document its
  4 cm binding gutter, its margins and the restart of page numbering at
  Chapter 1 — silently. Now stops at anything that is not a paragraph or table.
- **List of Figures / Tables page numbers were wrong in every earlier build**:
  they recorded the PDF page *index* rather than the number printed on the
  page (out by the length of the front matter), and were measured with the
  lists emptied, so refilling them reflowed the document underneath.
- **The table of contents was never rebuilt**, so it still showed the
  pre-Chapter-8 structure. Word now rebuilds it during the build.
- **Appendix headings inherited chapter numbering** ("10.5 Appendix B").

---

## 13. Deviations from the paper that remain

| Deviation | Status | Cost to close |
|---|---|---|
| **Training pool predates two of its own fixes.** The 5M pairs (and therefore the operator behind every result in Chapters 6–7) were generated before both the shuffle fix and the 100→50 generations fix. | Open, disclosed | ~22 h: regenerate (3 h) + retrain (6.5 h) + rerun the grid (2 h), the step-control grid (10.4 h) and all Chapter 7 diagnostics |
| Same pool also predates the leaf-bias correction | Open, disclosed | as above |
| AdamW weight decay 0.004 is our choice | Open | — |
| SD_d = 2.0 in the classification chapter | Deliberate, justified (§10.1) | — |
| SLIM_GSGP, DSR, DAE-GP not re-implemented | Out of scope | substantial |
| Step-control arms are not equal-budget (8× evaluations) | Stated wherever used | — |
| k=8 classification arms at n=10, not 30 | Stated as preliminary | ~7 h |

**The control that covers the pool deviation:** the classification operator
(§10.4) was trained from scratch on a pool built *after* both corrections, on
a different task at a 40× different semantic scale, and reproduces the
step-size floor unchanged. The central diagnosis therefore rests on a
corrected pool as well as the original one.

---

## 14. What is not in git, and how to regenerate it

**In git (recoverable from the repository alone):** all source, and the run
files behind every number in this record — 1,654 JSON across the twelve
result directories named in §8 and §10, plus `results_instr_base/` (the
working tree holds ~2,164 in total, the remainder being superseded or scratch
grids), `diagnostics/*.json`, `results_clf_ml/*.json`, the
`.cmd` drivers, `EXPERIMENT_LOG.md`, `V7_STATUS.md`,
`CLASSIFICATION_STATUS.md`, `build_report.py`, `make_figures.py`, this file.

**NOT in git** (`.gitignore`) — regenerate as follows:

| Missing | Size | Regenerate with | Cost |
|---|---|---|---|
| `data/training/` (5M pairs) | 1.7 GB | pipeline stage 2 | ~3 h |
| `data/training_clf/` | 2.1 GB | `run_classification_pipeline.cmd` stage 1 | ~2 h |
| `data/pmlb_cache/` | small | stage 1 | minutes, needs network |
| `checkpoints_adamw/` (the operator of record, 9 files) | 33 MB | stage 3 | ~6.5 h |
| `checkpoints_clf/` (classification operator, 9 files) | 33 MB | classification pipeline | ~6 h |
| `checkpoints/`, `_bs32_probe`, `_lowsd`, `_sdbin`, `_sdnorm` (diagnostic variants) | | only needed to rerun §11 | |
| `figures/` (14 figures, png+pdf) | | `python make_figures.py` | seconds |
| `*.log`, `*.png` at top level | | — | not needed |

**A regenerated operator will not be bit-identical** — training is stochastic
and optimiser moments are not checkpointed. Expect the *conclusions* to
reproduce (they survived four independently trained variants), not the exact
figures.

**The report source** `MIS41170-Capstone.docx` lives in
`C:\Users\yc199\Downloads\` and is **not in the repository**. `build_report.py`
reads it and writes `MIS41170-Capstone-FINAL.docx` to `D:\MSBA\Capstone\TSGP\`.
**Without that source file the report cannot be rebuilt.** Back it up.

---

## 15. The report and its build system

`build_report.py` reads the hand-edited source .docx, never modifies it, and
writes the final document in four stages:

0. **Corrections** — 43 named text replacements (the fact-check of §12). Each
   must match exactly once or the build aborts, so a correction that silently
   did nothing cannot pass.
1. **Insertions** — 14 figures at named anchors, the classification chapter,
   the extra threats section, the three appendices, the glossary, the index.
2. **Renumbering** — every figure and table caption renumbered in document
   order, chapter by chapter; appendix captions (A.1–A.3, C.1–C.2) left alone.
3. **Lists, index and contents** — entries written with a fixed-width
   placeholder so the document is at final length, Word rebuilds the table of
   contents, the PDF is rendered, the number *printed* on each page is read
   back, and the entries are filled without reflowing a line. Repeats until
   nothing moves.

Final document: **86 pages, 10 chapters, 14 figures, 19 tables**, three
appendices, 26-term glossary, 37-term generated index.

Appendix C's two results tables are computed from the run files at build time,
so they cannot drift from the data.

---

## 16. Verification status: what is proven, what is not

**Mechanically verified** (recomputed from archived run files, 2026-08-17):
every number in report Tables 6.1, 6.2, 6.3, 7.2, 7.4, 8.2, 8.3, 8.4, C.1 and
C.2; all significance tests; the step floor; the locality control; the ML
baselines; the no-division grid; all 32 caption numbers and 37 index terms;
the reference list (25 entries, all cited, all resolving, no year mismatches).

**Recorded but not independently verifiable from archived artifacts:**

- PMLB containing exactly two four-feature binary classification datasets
  (264 samples at 73% imbalance; 50 samples) — needs network access.
- Token statistics: 0.7% protdiv vs ~16% each add/sub/mul in training targets.
- Sampled offspring median size 35 = training-target median 35.
- Population semantic norms 19.5 → 27.3 → 57.7.
- Batched-sampler timings (69.4 ms, 1374.7 → 84.0 ms) and the equivalence
  tests — logged at the time, not reproducible from stored artifacts.
- SD percentiles of the two pools (the pools themselves are gitignored).

**Known to be unreliable if repeated:** any achieved-SD figure to better than
±20% (§9.5).

---

## 17. Open questions

1. **The solution-size gap.** Ours are 15–24 nodes against the paper's 58–73,
   and *no intervention moved it* — not retraining, not the optimiser, not
   temperature, not step control (which reaches only 17–25). It is the one
   symptom with no mechanism attached. Tree size is also hypersensitive
   (§11), so it may be a red herring twice over.
2. **Does a constant distance target match a decaying one?** Ten runs on ESL
   gave a difference of 0.017, better on 6/10, **p = 0.49** — enough to
   withdraw the annealing claim, not enough to assert equivalence. The tuned
   schedule transferred to ESL, Galaxy and LEV but not ERA or pollen.
3. **A regression pool matched to the search's semantic regime.** Chapter 8
   did this for classification; the equivalent for regression is untested.
4. **The remaining three baselines** — SLIM_GSGP, DSR, DAE-GP — would place
   TSGP against the full field the paper reports.
5. **KV cache in the decode loop** — the real remaining performance headroom;
   would make larger k affordable.

---

*Compiled 2026-08-17 from the archived run files, `tsgp/config.py`,
`EXPERIMENT_LOG.md`, `V7_STATUS.md`, `CLASSIFICATION_STATUS.md` and the git
history. Where this file and any other document disagree, this one was
recomputed from the data and should be preferred.*
