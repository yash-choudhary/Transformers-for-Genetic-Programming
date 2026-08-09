# TSGP Replication — Experiment Log

Running record of the replication of Anthes, Sobania & Rothlauf (2025),
*Transformer Semantic Genetic Programming for Symbolic Regression*
(arXiv:2501.18479v1, PDF in repo root).

**Append new entries at the bottom of the Log section.** Each entry records what
was changed, why, and what the evidence showed — including things that were tried
and rejected, which matter as much as the ones that worked.

---

## Current status

| | |
|---|---|
| stdGP baseline | **Replicates** — matches or beats the paper on 4/5 datasets |
| TSGP | **Does not replicate** — worse than the paper on 5/5; loses to our stdGP on 5/5 |
| Optimizer deviation (Adam vs AdamW) | **Eliminated** — fixed, retrained, no significant effect (p = 0.23) |
| Undertraining / batch size | **Eliminated** — offspring size flat across all 8 epochs; bs32 probe negative |
| Syntax control | **Exonerated** — removes 0.0004 of probability mass, no effect on length |
| Sampling temperature | **Eliminated** — significantly *worse* on 5/5; T=1.0 is both best and faithful |
| Root symptom | Model's P(function) 0.470 vs training 0.488 → offspring half the trained length |
| Model competence | **Sound** — 0.798 token accuracy vs a ~0.58 copy baseline; not undertrained |
| Hypotheses eliminated | **7** (optimizer, batch size, syntax control, temperature, drift, OOD parents, SD encoding) |
| Root cause reached | One-to-many task (42% of tokens change); residual 20% token error puts offspring 77–230× beyond the training distance |
| Not attributable to | Any parameter the paper specifies |
| Recommendation | Consolidate and write up — no remaining lever is both paper-permitted and plausible |

---

## Environment

| | |
|---|---|
| Machine | Windows 11 Home Single Language, 31.8 GB RAM, RTX 2070 8 GB |
| `capstone2` | TF 2.15.1 **CPU-only**, numpy 1.26.4 — used for GP search / experiments |
| `capstone-gpu` | TF 2.10.1 **GPU**, numpy 1.23.5, CUDA 11.2 / cuDNN 8.1 — used for training |

### Operational gotchas (all cost time at least once)

- **Activate, don't invoke.** `...\envs\capstone-gpu\python.exe` directly skips conda
  activation, leaves CUDA DLLs off `PATH`, and TF silently reports **zero GPUs**. Use
  `conda activate` or `conda run`.
- **`conda run` buffers output.** It captures stdout/stderr and releases only at exit,
  so tqdm bars vanish for the whole run. Use `--no-capture-output` (alias `--live-stream`).
- **TF installs shadow each other.** `capstone2` has `tensorflow-gpu 2.10.1` *and*
  `tensorflow 2.15.1` installed; the later install overwrote the shared package dir and
  dragged keras/protobuf past what 2.10 accepts. That is why GPU training "stopped working".
- **conda `cudatoolkit` is a runtime, not the toolkit.** XLA needs `libdevice` (was in
  `Library\bin`, not the `nvvm/libdevice/` tree it looks for) and `ptxas.exe` (absent
  entirely). Resolved by disabling XLA rather than installing `cuda-nvcc`.
- **Windows Update will kill long runs.** Active hours were 08:00–18:00; a GPU job at 83%
  utilisation does not count as "in use" (Windows watches keyboard/mouse). Pause updates
  before any multi-hour run.
- **Pickles carry numpy version dependencies.** See entry 2026-08-08a.

---

## Parameter inventory: specified vs. inferred

Critical for the write-up — the paper is detailed, but not complete.

### Explicitly specified, and we match

Primitive set `{V, ERC, +, −, ×, %}` with protected division → 1; ERC range [−0.5, 0.5]
step 0.1; RHH init depth 2–5; max depth 17; population 100; 50 generations; tournament
size 5; RMSE; 30 runs; 50/50 train/test split; stdGP crossover 90% with 10% terminal
bias and 10% subtree mutation at depth 0–2; data generation over 50 synthetic problems
at population 2,000 with double tournament; k=3; `SD≠0 ∧ SD<100`; 5M pairs; 8 heads;
d_model 128; 2 encoder + 2 decoder stacks; 100-token limit; AdamW at lr 1e-3 for
8 epochs; `SD_d = 0.1`; datasets ERA, ESL, Galaxy, LEV, Pollen.

### Not specified — our choice

| Parameter | Our value | Notes |
|---|---|---|
| Transformer batch size | 256 | Paper silent. Keras default is 32 → 8× more updates. **Leading suspect.** |
| AdamW weight decay | 0.004 | Paper silent. 0.004 is the Keras default; 0.01 is common elsewhere. |
| SD encoding into the model | `Dense(d_model)` on raw scalar, added to embeddings | Paper: "an additional input to the encoder and decoder layer" |
| Feed-forward dim | 4 × d_model | Vaswani default; paper says "adapted from Vaswani et al." |
| Samples per synthetic problem | 200 | Paper silent |
| Gaussian noise magnitude | `uniform(0.05, 0.3)` | Paper: "we apply Gaussian noise" |
| Random inputs for semantics | 100 | Paper silent |
| Sampling temperature | 1.0 | Paper silent (1.0 = no scaling) |
| Replacement / elitism | Generational, no elitism | Table 1 lists selection only |
| Loss | Masked cross-entropy | Implied by "next token prediction", not named |

### Known conflict with the paper

`DATAGEN_GP_GENERATIONS = 100`, but Table 1 says **50 generations**, and §4.1 says data
generation uses "all other GP search parameters... the same as defined in Table 1". The
phrase "until a sufficient number of functions are generated" gives some licence.
**Unresolved.**

---

## Log

### 2026-08-06a — Benchmarks could not load

`DATASETS` used the paper's short names (`ERA`, `ESL`…); PMLB requires ID-prefixed keys.
Every run aborted on the first fetch. Mapped to `1030_ERA`, `1027_ESL`,
`690_visualizing_galaxy`, `1029_LEV`, `529_pollen` — all confirmed 4-feature regression
tasks, matching the paper's d=4 selection. Added a local cache with retry/backoff after a
transient `WinError 10065` killed a run mid-loop.

### 2026-08-06b — Grid restructured into restartable units

One monolithic script with no checkpointing meant any failure discarded all 300 runs.
Now every (dataset, method, run) triple is an independent unit with its own JSON file,
deterministically seeded from its own identity, written atomically. Failures are caught
per unit and the grid continues. Entry points split: `fetch_datasets`, `run_single`,
`run_experiments`, `aggregate_results`.

### 2026-08-06c — Batched the transformer sampler (26× faster)

Profiling found the bottleneck was dispatch, not arithmetic: 69.4 ms/decode-step eager
vs 7.3 ms under `tf.function` on a 934K-parameter model. The encoder was also re-run
every decode step despite being fixed, and each pass carried one sequence.

Fixed all three: encoder hoisted, both passes graph-compiled, whole population decoded
in lockstep. **1374.7 ms → 84.0 ms per individual**; a TSGP run 116 min → ~4.2 min;
the grid 291 h → ~10.5 h.

*Equivalence check* (the operator must be unchanged): 100 parents, sequential vs batched
— valid trees 100/100 both, median offspring size 11.0 both, **KS D=0.110, p=0.583**,
zero malformed trees. Not bit-identical (RNG consumption order differs), distributionally
identical.

### 2026-08-06d — Two paper-fidelity corrections

**Best-so-far reporting.** §4.2: "the best performing solution on the training set is used
to compute the RMSE on the test set" — the best at *any* point. The code reported the best
of the final population; with generational replacement and no elitism the best is routinely
destroyed. On ERA that was 0.9657 vs 0.8512. Fixed with an archive that never re-enters the
population, so search is untouched.

**Internal node bias.** §4.1 specifies 10% terminal bias in subtree crossover.
`GP_INTERNAL_NODE_BIAS = 0.1` existed in config but was never read — both toolboxes used
unbiased `cxOnePoint`. Now `cxOnePointLeafBiased`.

*Rejected as drift:* adding elitism. Table 1 specifies selection only; generational
replacement without elitism is the faithful reading.

### 2026-08-07a — SD conditioning hypothesis: tested and REJECTED

Diagnostics suggested the semantic-distance conditioning was inert: sweeping `SD_d` across
the full training range moved achieved distance only 1.6×, `SD_d=0.0` and `0.1` gave
byte-identical output, and the learned projection carries a bias 33× the signal at the
operating point (‖0.1·W‖ = 0.032 vs ‖bias‖ = 1.065).

**Not acted on.** §4.4 measures semantics on the target problem's *test set*, not the random
inputs our diagnostic used, so magnitudes aren't comparable. The paper never claims achieved
distance ≈ `SD_d`, explicitly expects large distances early, and lists systematic step-size
control as *future work*. Nothing specifies how SD should be encoded. Changing it would have
been inventing a method the authors did not describe.

### 2026-08-07b — Full grid v1 (300/300 units)

| Dataset | Ours TSGP | Ours stdGP | Paper TSGP | Paper stdGP |
|---|---|---|---|---|
| ERA | 0.9488 | 0.8865 | 0.797 | 0.817 |
| ESL | 0.4760 | 0.4512 | 0.379 | 0.502 |
| Galaxy | 0.3366 | 0.3195 | 0.327 | 0.337 |
| LEV | 0.7619 | 0.6716 | 0.672 | 0.703 |
| pollen | 0.6732 | 0.4818 | 0.518 | 0.514 |

stdGP matches or beats the paper on 4/5 — infrastructure is sound. TSGP is worse on 5/5 and
loses to our own stdGP on 5/5, where the paper has it winning 4/5 (Wilcoxon: stdGP
significantly better on ERA, Galaxy, LEV, pollen; ESL n.s.).

**Diagnostic signal:** median best-solution size 17–22 vs the paper's 58–73, and consistently
*smaller* than our stdGP where the paper's is consistently *larger*. TSGP's best-so-far stops
improving at median generation 15–32, with only 2.5–6 of 50 generations producing any
improvement.

Report: `TSGP_Replication_Report.pdf`.

### 2026-08-08a — Environment work for GPU retraining

- **AdamW** added ([`make_optimizer`](tsgp/train_transformer.py)), resolving across namespaces
  (2.10 exposes it under `optimizers.experimental`, 2.11+ under `optimizers`), with
  `jit_compile=False` — TF 2.10's experimental optimizers default to XLA, which needs a full
  CUDA toolkit the conda runtime doesn't ship.
- **Portable pickle.** `sd` was stored as `np.float32`; pickles written under numpy ≥1.26
  reference `numpy._core`, which numpy 1.23 can't import — so the GPU env couldn't read the
  training data. Generator now stores `float(sd)`; existing data converted to
  `data/training/training_pairs_portable.pkl` (verified to contain no numpy references).
- Non-interactive resume: `input()` would block an unattended run, so training now
  auto-resumes when stdin isn't a TTY, with `--fresh` to override.

### 2026-08-08b — AdamW retrain: NEGATIVE result

8 epochs, batch 256, GPU, ~49 min/epoch (interrupted after epoch 4 by a Windows Update
restart; resumed — note optimizer moments are not checkpointed, so AdamW state restarted).
Checkpoints in `checkpoints_adamw/`.

Paired comparison on ERA, identical seeds and starting populations, 10 runs:

| | median | mean |
|---|---|---|
| Old (Adam), same 10 seeds | 0.9480 | 0.9551 |
| New (AdamW) | 0.9183 | 0.9294 |
| Paper | 0.797 | — |

**Wilcoxon signed-rank p = 0.2324; 6/10 runs improved.** Not significant. Median best size
went *down*, 20 → 15, against the paper's 72 — the structural symptom is unchanged.

**Conclusion:** the optimizer is eliminated as the primary cause. This is progress — it was
the one unambiguous deviation from §4.1, and fixing it changed nothing material.

### 2026-08-08c — Batch-size probe (in progress)

**Hypothesis:** the model is undertrained. The paper fixes 8 epochs but never states a batch
size; it also says "systematic hyper-parameter tuning was not feasible", suggesting defaults.
Keras's default batch size is 32.

- Batch 256 × 8 epochs = **156,256** updates
- Batch 32 × 1 epoch = **156,250** updates

So one batch-32 epoch matches our entire current run on update count, isolating batch size
itself. Persistent tiny solutions are characteristic of an undertrained next-token model
falling back on short, high-frequency sequences.

Added `--batch-size` and `--epochs` flags. Probe (~2.6 h):

```bash
conda run --no-capture-output -n capstone-gpu python -m tsgp.train_transformer \
    --data data/training/training_pairs_portable.pkl \
    --checkpoints checkpoints_bs32_probe --batch-size 32 --epochs 1 --fresh
```

Then compare offspring size against `checkpoints/` and `checkpoints_adamw/`. If unchanged,
the hypothesis is dead for ~2.6 h instead of ~21 h for a full batch-32 run.

**Result: NEGATIVE.** Probe ran ~3.1 h (`checkpoints_bs32_probe/`). Batch 32 produced
*smaller* offspring, not larger — three-way comparison on 120 identical parents:

| Checkpoint | median | mean | p90 | median SD |
|---|---|---|---|---|
| Adam, bs256 × 8ep | 11 | 17.8 | 43 | 93.0 |
| AdamW, bs256 × 8ep | 11 | 18.8 | 45 | 82.3 |
| AdamW, bs32 × 1ep | 9 | 14.3 | 25 | 120.6 |

Mann-Whitney "is bs32 larger?" **p = 0.9787** — decisively not.

### 2026-08-08d — Undertraining hypothesis fully eliminated

The probe matched *updates* but not *epochs*, so the strong form ("the paper used bs32 for
8 epochs = 1.25M updates") was still open. Tested for free using the AdamW per-epoch
checkpoints — does more training grow the offspring?

| epoch | updates | median | mean | p90 |
|---|---|---|---|---|
| 1 | 19,532 | 11 | 17.2 | 35 |
| 2 | 39,064 | 11 | 17.2 | 37 |
| 3 | 58,596 | 11 | 13.3 | 23 |
| 4 | 78,128 | 11 | 18.2 | 37 |
| 5 | 97,660 | 12 | 19.0 | 43 |
| 6 | 117,192 | 11 | 19.4 | 44 |
| 7 | 136,724 | 11 | 19.0 | 49 |
| 8 | 156,256 | 11 | 18.8 | 45 |

**Median is 11 at every epoch.** No trend. Behaviour is fixed within one epoch, so a full
batch-32 run (~21 h) would not change it. **Not worth running.**

### 2026-08-08e — New lead: generated length ≠ trained length

The epoch sweep surfaced a mismatch that needs no guessing about unstated hyperparameters:

- Training pairs, output token length: **mean 35.2**, max 97
- Model's sampled offspring: **mean 14–19, median 11**

The model generates sequences roughly half the length of its own training targets, stably
from epoch 1. A well-fit next-token model should reproduce its training length
distribution. Because offspring are also *smaller than their parents* (parent median 15 →
offspring median 11), the search shrinks rather than grows — the opposite of the paper's
Figure 3, where TSGP solution size climbs across generations. This is sufficient on its own
to explain best-solution sizes of 17–22 against the paper's 58–73.

**Candidate causes to separate:**
1. Sampling-time syntax control distorting the length distribution (measurable: sample with
   the mask disabled and compare lengths).
2. The SD encoding. Now correctly classified as **paper-silent, our choice** — the same
   category as batch size — rather than as a bug to fix. Earlier measurements showed it is
   nearly inert (‖0.1·W‖ = 0.032 vs ‖bias‖ = 1.065; `SD_d` 0.0 and 0.1 give identical
   output). Testing an alternative encoding is exploring an unspecified parameter, not drift.

**Next:** diagnostic 1 first — it is minutes of work and requires no retraining.

### 2026-08-08f — Syntax control is NOT the cause

Sampled 100 offspring with and without the mask:

| | with mask | without mask |
|---|---|---|
| generated length | mean 15.9, median 11 | mean 16.7, median 11 |
| probability mass removed by mask | **0.0004** | — |

The mask removes four ten-thousandths of the mass and does not change the length
distribution. **Sampler exonerated.**

What it did show is a calibration gap in the function/terminal ratio:

| | P(function) |
|---|---|
| Training data | 0.4879 |
| Model, masked | 0.4812 |
| Model, unmasked | 0.4621 |

**Correction to an earlier claim in this log.** I previously presented `1/(1−2p)`
reproducing the measured mean size as evidence for a branching-process "amplifier". It is
not evidence — for a binary tree `size = 2·n_func + 1` identically, so that formula recovers
the mean size by algebra for *any* set of trees. "P(function) is low" and "trees are small"
are the same statement in different units, not cause and effect. The calibration gap versus
training is real; the amplification framing was overstated.

### 2026-08-08g — Temperature sweep (paper-silent parameter)

Sampling temperature is not specified by the paper, so tuning it explores an unspecified
choice rather than altering the method. 120 identical parents, `checkpoints_adamw`:

| T | mean size | P(function) | median SD |
|---|---|---|---|
| 1.00 | 16.7 | 0.4700 | 138.5 |
| 0.50 | **24.0** | **0.4792** | 57.5 |
| 0.40 | 21.9 | 0.4771 | 30.6 |
| 0.30 | 22.1 | 0.4774 | 29.1 |
| 0.20 | 22.6 | 0.4779 | **25.3** |
| 0.15 | 22.4 | 0.4777 | 28.8 |
| *training target* | *41.5* | *0.4879* | — |

**Temperature saturates below T≈0.5** and cannot reach the training P(function). But it buys
two things: ~44% larger offspring, and semantic distance collapsing 138.5 → ~25, which lands
inside the paper's Figure 4 range for ERA (0–35) where T=1.0 was far outside it. A single
Galaxy unit at T=0.5 produced a **57-node** solution (paper's Galaxy median: 64).

`--temperature` added to `run_experiments` and `run_single`; the value is recorded in every
result file so a results directory is never ambiguous. Default stays 1.0.

**Result: NEGATIVE — actively harmful.** Full 150-unit grid at T=0.5 (`results_t05`) plus
30 ERA units at T=0.2 (`results_t02`).

| Dataset | TSGP T=1.0 | TSGP T=0.5 | stdGP | paper TSGP |
|---|---|---|---|---|
| ERA | 0.9488 | 1.0008 | 0.8865 | 0.797 |
| ESL | 0.4760 | 0.5088 | 0.4512 | 0.379 |
| Galaxy | 0.3366 | 0.4122 | 0.3195 | 0.327 |
| LEV | 0.7619 | 0.8031 | 0.6716 | 0.672 |
| pollen | 0.6732 | 0.7203 | 0.4818 | 0.518 |

Paired against identical seeds, **every dataset degrades significantly** (p = 9.3e-09 to
0.0024; only 1–10 of 30 runs better). ERA is monotonic in temperature:
T=1.0 → 0.9488, T=0.5 → 1.0008, T=0.2 → 1.0499.

**Why:** temperature is not a calibration knob here — it is the variation operator's only
source of stochasticity, the analogue of mutation/crossover randomness in stdGP. Sharpening
makes offspring more typical *and more alike*, collapsing population diversity. The size
gain measured on random initial populations did not survive an evolutionary loop.

**Methodological lesson:** solution size was the wrong optimisation target. At T=0.5,
Galaxy grew 22 → 34 and pollen 21 → 48 — much closer to the paper's 64 and 58 — and RMSE got
*worse on both*. The paper's large solutions are a consequence of a search that works, not a
cause of it. Chasing the size symptom was a mistake.

**T=1.0 stands** as both the best-performing and the faithful (paper-silent → neutral)
choice. The v1/v2 numbers remain the headline result.

### Hypotheses eliminated to date

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 1 | Optimizer (Adam vs AdamW) | Eliminated | Retrained; paired p = 0.23 |
| 2 | Undertraining / batch size | Eliminated | Offspring size flat across all 8 epochs; bs32 probe negative |
| 3 | Syntax control distorting lengths | Eliminated | Mask removes 0.0004 of probability mass |
| 4 | Sampling temperature | Eliminated (harmful) | Significantly worse on 5/5 datasets |

Still open, all requiring a retrain: SD encoding (paper-silent), AdamW weight decay
(paper-silent), training pool built with unbiased crossover.

### 2026-08-09a — Did the model learn its task? (the check we should have run first)

Prompted by the fair objection that eliminating four hyperparameter hypotheses does not make
a failed replication "complete". Because the stdGP baseline *does* reproduce the paper, a
defect in our TSGP implementation is a priori more likely than an error in the paper. All
four eliminated hypotheses were hyperparameter-level; the model's actual competence was never
verified.

On 300 held-out training pairs, `checkpoints_adamw`:

| Check | Result |
|---|---|
| Teacher-forced next-token accuracy | 0.7997 (first token 0.8933; random ≈ 0.0455) |
| Free generation: target SD → achieved | 0.152 → 32.0 (**210×**) |
| Decoder conditions on encoder? | Yes — 35/40 distinct greedy outputs |

Accuracy across epochs (2,000 pairs) — **plateaued**, so this is the model's ceiling:

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| token acc | .7715 | .7847 | .7906 | .7928 | .7937 | .7958 | .7914 | .7980 |
| seq exact | .0020 | .0015 | .0035 | .0050 | .0040 | .0040 | .0025 | .0060 |

**Interpretation.** The model trains successfully and uses the parent. It is *not* broken and
not undertrained. But free-running generation drifts: at 80% per-token accuracy the chance of
reproducing a ~35-token target intact is 0.8³⁵ ≈ 0.04%, and in a prefix expression one wrong
operator or variable changes the whole function's semantics. Teacher forcing hides this
entirely, which is why training metrics looked healthy throughout.

Note the task is inherently **one-to-many** — with k=3 neighbours, many functions sit at a
given SD from `f_i` — so exact sequence reproduction is the wrong bar and 80% may be near the
achievable ceiling. What is supposed to resolve that ambiguity is the SD conditioning: telling
the model *which* distance is wanted narrows the output distribution.

**This connects to the earlier SD measurement.** `sd_projection` carries ‖0.1·W‖ = 0.032
against ‖bias‖ = 1.065, and `SD_d` = 0.0 and 0.1 give byte-identical output — the conditioning
is numerically inert. So the model cannot use SD to disambiguate, generation is effectively
unconditional on distance, and achieved SD is uncontrolled (210× the target). Controlling
semantic distance is the entire mechanism of TSGP.

**Status revised: NOT complete.** The SD encoding is now the prime suspect, backed by a chain
of measurements rather than speculation. Note the distinction that matters for fidelity: the
paper *does* specify that SD is provided as an input to the encoder and decoder to control
step size (§3.2). Our implementation nominally does this but numerically fails to. Making the
conditioning functional is implementing what the paper describes — though the *specific*
encoding remains our choice, since the paper does not state one.

### 2026-08-09b — SD normalisation variant (log1p + standardised)

Confirmed why raw SD fails numerically: **75% of training SDs are below 0.667** while the tail
reaches 100, so a linear projection sees almost no variation across the bulk while outliers
dominate its scale. After `log1p` + standardisation, p1→p99 spans z = −0.63 → +4.17.

Implemented as a switchable input transform (`--sd-normalize`) that adds **no weights**, so
checkpoints stay interchangeable; verified the existing model is bit-for-bit unaffected and
the SD signal in the logits is 7× stronger (0.61 vs 0.087) even before retraining. Trained
4 epochs (`checkpoints_sdnorm`) — 4 rather than 8 because accuracy plateaus by epoch 3, so
this is a diagnostic, not a replication run.

**Early read at epoch 1 (both models at equal training):**

| model | span (max/min) | log-log corr with SD_d |
|---|---|---|
| raw SD | 1.69× | +0.789 |
| normalised | 1.26× | −0.446 |

Neither controllable; the normalised correlation runs the wrong way. Given the spread
(p25≈40, p75≈500) the difference is likely noise — the honest reading is both are flat.

### 2026-08-09c — Drift hypothesis FALSIFIED; size is a sampling artifact

Greedy decoding removes all sampling noise. If drift caused the semantic gap, greedy should
land far closer. On 200 real training inputs (`checkpoints_adamw`):

| decoding | achieved SD | offspring size | distinct |
|---|---|---|---|
| greedy | 10.90 | **44** | 181/200 |
| sampled | 11.50 | 17 | 200/200 |
| *target / training* | *0.141* | *~41* | — |

**1. Sampling noise is NOT the cause.** Greedy and sampled reach the same distance (77× vs
81× the target). The model does not generate semantically-close functions under any decoding.

**2. The size deficit is purely a sampling artifact.** Greedy gives correctly-sized trees
(44 vs training ~41); sampling halves them. With 4 function tokens sharing ~0.47 of the mass
against 15 terminal tokens sharing ~0.53, argmax usually picks a function while sampling
spreads over the many terminals. **Size and semantic quality are decoupled** — now the third
independent confirmation that chasing solution size was the wrong target.

**3. In-distribution vs out-of-distribution parents matter a lot.** Real training inputs give
achieved SD ≈ 11; random RHH parents give ≈ 83–134. This also confounds the earlier
temperature sweep, which used random parents. §4.4 anticipates the effect: TSGP "is unlikely
to be able to generate solutions of low semantic distance to these random solutions at the
beginning of the search."

*Vicious-cycle theory subsequently rejected:* the earlier instrumented run shows within-run
distance falling 17.5 (gen 1) → ~11 by gen 5 and then plateauing, and ~11 is exactly the
greedy figure on real training inputs. The population **does** reach the in-distribution
regime; ~11 is simply the model's floor there.

### 2026-08-09d — The task itself is hard (structural analysis of the pairs)

Measured how similar `f_i` and `f_o` actually are, over 4,000 random training pairs:

| | |
|---|---|
| identical input == output | **0.00%** |
| normalised edit distance | median **0.421**, mean 0.427 |
| token-set Jaccard overlap | median 0.889 |
| pairs within edit distance 0.10 | 6.8% |
| pairs within edit distance 0.50 | 63.0% |

Semantic neighbours are **structurally dissimilar** (42% of tokens differ) while drawing on
nearly the same vocabulary. The model must transform, not copy — and because many functions
sit at any given SD from `f_i`, the mapping is genuinely one-to-many.

**This recalibrates the 80% accuracy figure.** A pure copy baseline scores ~58% (42% of
tokens differ), so at 0.798 the model is well above copying and has learned real
transformation structure. **It is not a broken or undertrained model.** But the residual 20%
is fatal: semantics are hypersensitive to individual tokens, so the model emits *a* plausible
semantic neighbour rather than *the* target, landing at distance ~11 instead of ~0.14.

**This is the deepest explanation reached.** It is also not attributable to any parameter the
paper specifies, and the paper reports neither token accuracy nor achieved-vs-requested SD,
so there is no published figure to compare against.

### 2026-08-09e — SD-normalisation result: NEGATIVE (hypothesis eliminated)

Evaluated at epoch 3 (accuracy plateaus there, so ≈ final quality), 300 real training inputs:

| | teacher-forced acc | achieved SD (target 0.152) | sweep SD_d 0.01 → 10 |
|---|---|---|---|
| raw SD (baseline) | **0.7997** | 34.9 (230×) | 33.7 → 25.6 |
| sd-normalised | 0.7938 | 44.9 (295×) | 71.2 → 51.9 |

No accuracy gain, worse achieved distance, and **neither model responds to `SD_d`** — both
sweeps trend mildly *backwards*. Giving the projection a well-conditioned input did not make
the conditioning functional.

Confirmed on the completed 4-epoch model: accuracy 0.7950 vs 0.7997 baseline; achieved SD
51.8 (341×) vs 36.2 (238×); sweep 45.4 → 38.5, still flat/backwards.

**Why this is informative rather than just another negative.** It separates two explanations
that were previously confounded. The conditioning is not inert because of bad numerics — it
is inert because **SD is nearly uninformative for token-level prediction**. Many functions sit
at any given distance from `f_i`, so knowing the distance barely constrains the next token,
and cross-entropy therefore applies almost no pressure to use the input. Normalising the
scalar cannot fix an objective that does not reward it.

Incidental: on real training parents, offspring are already 29–41 nodes against a training
distribution of ~41. **Size was never the problem** — fourth independent confirmation.

### 2026-08-09f — Pair quality: difficulty is intrinsic; k=3 exonerated

Tested whether the 42% median edit distance is intrinsic to semantic k-NN pairing or an
artifact of how our function pool was built.

| SD bucket | median SD | median edit distance | median output len |
|---|---|---|---|
| [0, 0.01) | 0.0048 | 0.222 | 39 |
| [0.05, 0.15) | 0.0869 | 0.364 | 37 |
| [0.5, 2) | 0.9401 | 0.566 | 25 |
| [10, ∞) | 19.555 | 0.727 | 13 |

corr(log1p(SD), edit distance) = **+0.444** — the pairing is sensible, not random.

**Intrinsic.** Even at near-zero SD, 22% of tokens differ, and there are **zero identical
pairs**. Semantically indistinguishable functions are written differently — `add(x0,x1)` vs
`add(x1,x0)` alone, since add and mul are commutative. Irreducible under prefix tokens.

**k=3 exonerated.** Neighbour ranks 0/1/2 have median SD 0.23 / 0.31 / 0.38. Dropping to k=1
would barely change the pairs.

**New lead (SD-range mismatch).** The operator is *always* queried at `SD_d = 0.1`, where
pairs have edit distance 0.32, but is *trained* across the full range including a tail at
0.727 that teaches large structural rewrites. Working SD conditioning would separate those
regimes; we proved it cannot. So at inference the model draws from its **marginal over all
SD values** rather than the requested one — which explains achieved distance ~32 against a
training median of 0.164.

### 2026-08-09g — Low-SD training subset: NEGATIVE (deliberate deviation)

**Deviation, deliberately.** Sect. 4.1 keeps every pair with `SD != 0 and SD < 100`; this run
trained only on `SD <= 0.2` (2.7M of 5M pairs, `--max-sd 0.2`, `checkpoints_lowsd`). Run as a
diagnostic of the marginal-distribution theory: the operator is always queried at
`SD_d = 0.1` but trained across the full range, and since the conditioning is inert it should
sample from its marginal rather than the requested distance.

**Prediction:** removing the high-SD tail should pull achieved distance down toward 0.15.

| | achieved SD | teacher-forced acc |
|---|---|---|
| baseline (full range, 8 ep) | 38.1 | 0.7997 |
| low-SD @ep1 | 58.1 | 0.7096 |
| low-SD @ep2 | 43.8 | 0.7235 |
| low-SD @ep3 | 44.2 | 0.7337 |
| low-SD @final (4 ep) | **50.1** | 0.7374 |

**Result: worse, not better.** No convergence toward baseline; the controllability sweep is
non-monotonic noise (52.1 → 48.8 → 45.8 → 27.7 → 58.8). Theory rejected.

**Note this cuts in the paper's favour:** a deliberate departure from the specification made
the operator worse. The paper's stated configuration is not what is holding the result back.

**Measurement-noise caveat.** Across today's runs the *same baseline checkpoint* measured
31.7 / 34.9 / 36.2 / 38.1 — about ±20% run to run, since the diagnostic samples
stochastically. Low-SD measured 43.8–58.1. The ranges do not overlap so the conclusion
stands, but differences below ~20% in any achieved-SD figure in this log should not be read
as real.

### Hypotheses eliminated — final tally

| # | Hypothesis | Verdict | Key evidence |
|---|---|---|---|
| 1 | Optimizer (Adam vs AdamW) | Eliminated | Retrained; paired p = 0.23 |
| 2 | Undertraining / batch size | Eliminated | Accuracy plateaus at epoch 3; bs32 probe negative |
| 3 | Syntax control | Eliminated | Mask removes 0.0004 of probability mass |
| 4 | Sampling temperature | Eliminated (harmful) | Significantly worse on 5/5 datasets |
| 5 | Autoregressive drift | Eliminated | Greedy decoding lands at the same distance as sampled |
| 6 | Out-of-distribution parents | Eliminated | Within-run distance plateaus at the in-distribution floor |
| 7 | SD encoding (numerics) | Eliminated | Normalised variant no better; still no response to SD_d |
| 8 | Training SD range (deviation) | Eliminated | Low-SD subset made achieved distance worse (50.1 vs 38.1) |

**Root cause, as far as the evidence reaches:** the operator must perform a one-to-many
semantic-preserving transformation (42% of tokens change, many valid targets). The model
learns this better than a copy baseline (0.798 vs ~0.58) but the residual 20% token error
places offspring ~77–230× further from the parent than the training pairs. No parameter the
paper specifies alters this, and the paper publishes no comparable diagnostic.

---

## Open questions

1. **Batch size** — probe in progress.
2. **Training pool crossover** — the 5M pairs were generated before the leaf-bias fix. Data
   regeneration (~3 h) was set up but paused; note the paper does not explicitly state that
   data generation uses the biased operator.
3. **`DATAGEN_GP_GENERATIONS`** — 100 in config vs Table 1's 50.
4. **AdamW weight decay** — 0.004 is our choice, not the paper's.

## Reproducing

```bash
python -m tsgp.fetch_datasets                        # cache the five PMLB benchmarks
python -m tsgp.data_generator --output data/training # regenerate the 5M pairs
python -m tsgp.train_transformer --data ... --checkpoints ...
python -m tsgp.run_experiments --weights ... --output ...
python -m tsgp.aggregate_results --output ...
```

Every stage is restartable; completed units and epochs are skipped on re-run.
