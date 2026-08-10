# v7 — search dynamics: status and handoff

Branch: `v7-search-dynamics`. Everything below was produced on 2026-08-10.

## TL;DR

The v1–v6 diagnosis does not survive controlled re-measurement. The operator is
**not** broken. The replication still fails, but for a different, measured
reason: the operator's semantic step has a **floor of ~0.7** while the paper
requests `SD_d = 0.1`, and unlike stdGP it **never anneals**, so the search has
no exploitation phase and freezes around generation 10.

## Results

### Headline grid — `results_v7/` (300 units, k=1, i.e. the paper's operator)

| Dataset | TSGP | stdGP | paper TSGP | paper stdGP | winner | paper | p |
|---|---|---|---|---|---|---|---|
| ERA | 0.8442 | 0.8070 | 0.797 | 0.817 | stdGP | TSGP | 2.0e−03 |
| ESL | 0.4748 | 0.4531 | 0.379 | 0.502 | n.s. | TSGP | 0.86 |
| Galaxy | 0.3501 | 0.3322 | 0.327 | 0.337 | n.s. | TSGP | 0.50 |
| LEV | 0.7540 | 0.6930 | 0.672 | 0.703 | stdGP | TSGP | 7.9e−04 |
| pollen | 0.6324 | 0.5075 | 0.518 | 0.514 | stdGP | stdGP | 2.4e−03 |

stdGP now reproduces the paper within 0.010 on four of five. TSGP does not
replicate: direction agrees on 1/5 (pollen, where the paper also has stdGP
ahead).

Median best-solution size — TSGP 15/23/21/16/24 against the paper's 72/73/64/69/58.

### Annealed step control — `results_anneal/` (IN PROGRESS, 20/150 when saved)

ESL pilot, n=10, paired on identical seeds against k=1:
median 0.4917 → **0.4364**, better on 7/10 seeds, size 17 → 25.
Nominally beats stdGP (0.4531) — the paper's direction — but **p = 0.16**
paired and **p = 0.30** vs stdGP, i.e. not significant.
ERA at n=13 showed **no effect** (0.8433 vs 0.8442), so the ESL gain may be
dataset-specific.

## What the v1–v6 report got wrong

Measured with a control condition the earlier diagnostics lacked:

| Report claim | Measurement |
|---|---|
| "SD conditioning is inert" | Monotonic over 5.6×: SD_d 0.1/1/10 → 1.15/1.55/6.44 |
| "failure localises to token-level accuracy" | Offspring land 16× closer to their own parent than to an unrelated one (ratio 0.061) |
| "solutions 3–4× too small; the operator does not build structure" | Sampled size median 35 vs training targets' median 35 — exact match |

Tree size in prefix notation is a **critical branching process** (mean size
= 1/(1−2p)), so a 1-point change in operator probability halves it. It is a
hypersensitive, low-information metric and was a red herring for three report
versions.

## Bugs fixed

1. **`datasets.py` — `random_state=42`** pinned one train/test split across all
   30 "independent" runs. Fixing it moved **9 of 10 cells closer to the
   published values** (ERA TSGP 0.9488 → 0.8442; ERA stdGP 0.8865 → 0.8070).
   This was the single largest correction in the whole study.
2. **`data_generator.py`** — `find_semantic_pairs` early-returned at exactly
   `TARGET_NUM_PAIRS` while walking the pool in problem order, so later
   synthetic problems contributed zero training pairs. Now shuffles first.
3. **`config.py`** — `DATAGEN_GP_GENERATIONS` 100 → 50, per Table 1.
4. **The GPU was never being used.** conda's CUDA DLLs live in
   `%CONDA_PREFIX%\Library\bin`, which only reaches PATH via `conda activate`;
   calling `python.exe` by path gave a CUDA-built TF that silently fell back to
   CPU at **3.7 s/batch vs ~0.15 s/batch** — 20 h/epoch instead of 54 min.
   Use `gpu_python.cmd`. Verify with `nvidia-smi`, not by TF not erroring.

## Performance work (both verified equivalent, not just faster)

- **Vectorised evaluation** (`primitives.py`): whole-matrix instead of a Python
  row loop. Bit-identical on 400/400 trees.
- **Vectorised sampler** (`tsgp_search.py`): Gumbel-max over all active
  sequences with the four possible syntax masks precomputed, replacing one
  `np.random.choice` per sequence per token. KS D = 0.022, p = 0.846 against
  the original; median offspring size 37 vs 37.
- **`evaluate_semantics_fast`**: prefix-stack interpreter, no `gp.compile`.
  Exact (0.0 relative diff on 1500 trees) but only 2.1× — `gp.compile` was
  **not** the bottleneck. The real cost is the decode loop being O(T²) with no
  KV cache. That is where the remaining headroom is.

## New modules

- `tsgp/instrument.py` — per-generation logging that reproduces the paper's
  Figs 2/3/4. This is what found the mechanism; v1–v6 only ever compared
  endpoint tables.
- `tsgp/operator_diagnostics.py` — gated pass/fail on LOCALITY / RESPONSE /
  REGIME, with the control condition. Run before spending a grid on a model.
- `tsgp/step_floor.py` — measures the achievable-step floor and min-of-k scaling.
- `tsgp/aggregate_results.py` — now scores against Tables 2/3 with Wilcoxon,
  and `--compare-with DIR` isolates a single change.
- `gpu_python.cmd` — launcher that actually exposes the GPU.

## Dead ends (do not repeat without reading why)

- **Binned SD conditioning.** Retrain abandoned at epoch 1: at *matched*
  training the binned encoding reproduced the identical step floor as the
  rank-1 linear one, so the floor is not a conditioning-capacity problem.
  `diagnostics/sdbin_epoch1.json` vs `diagnostics/linear_epoch1.json`.
- **Greedy min-of-k step control.** Breaks the floor (0.79 → 0.095 at k=16) but
  makes results *worse* — ESL paired, k=1 0.4135 vs k=8 0.4246 — because
  minimising distance makes offspring near-copies and starves exploration.
  Also 21 min/unit.

## To resume

```
# annealed grid — resumable, skips completed units
gpu_python.cmd -m tsgp.run_experiments --weights checkpoints_adamw/tsgp_final.npy \
    --methods tsgp --step-k 8 --step-anneal --output results_anneal --quiet

# results
python -m tsgp.aggregate_results --output results_anneal
python -m tsgp.aggregate_results --output results_v7 \
    --compare-with results --label-a fixed-split --label-b per-run-split
```

## Open items for the write-up

- The draft report (`MScBA_Capstone_Report_Draft.pdf`, in Downloads) still
  carries the v1–v6 numbers and the "inert conditioning / one-to-many" account
  in §7.3–7.4 and §9.1. The headline conclusion survives; the diagnosis and
  Tables 6.1–6.3 need replacing.
- Report k=1 as the replication; anything with `--step-anneal` is a separate,
  clearly-labelled extension of the paper's §5 future work. It spends k× the
  model evaluations, so it is **not** an equal-budget comparison against stdGP.
- §4.2 ambiguity: the best-so-far archive reading makes our stdGP stronger than
  the paper's (ESL size 103 at 0.443 vs their 12 at 0.502), raising the bar
  TSGP must clear. Defensible, but state it.
