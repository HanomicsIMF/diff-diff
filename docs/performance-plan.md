# Performance Improvement Plan

This document outlines the strategy for improving diff-diff's performance on large datasets, particularly for BasicDiD/TWFE and CallawaySantAnna estimators.

---

## Practitioner Workflow Baseline (v3.1.3, April 2026)

Earlier sections of this document (v1.4.0, v2.0.3) measured isolated `fit()`
calls on synthetic panels for R-parity. This section measures **end-to-end
practitioner chains** - Bacon decomposition, fit, event-study pre-trend
inspection, HonestDiD sensitivity grids, cross-estimator robustness refits,
and reporting - at data shapes anchored to applied-econ papers and industry
writeups. The six scenarios are defined in
[`docs/performance-scenarios.md`](performance-scenarios.md); scripts live in
`benchmarks/speed_review/bench_*.py`; raw results in
`benchmarks/speed_review/baselines/*.json` and flame profiles in
`benchmarks/speed_review/baselines/profiles/`.

Environment: macOS darwin 25.3 on Apple Silicon M4, Python 3.9,
numpy 2.x, diff_diff 3.1.3. Each multi-scale scenario runs at three data
scales under both `DIFF_DIFF_BACKEND=python` and `DIFF_DIFF_BACKEND=rust`,
with one intentional exception: the SDiD few-markets scenario at its
`large` scale runs Rust only, because the pure-numpy jackknife at n=500
would exceed four minutes per run without changing the already-clear
Python-vs-Rust conclusion established at `small` and `medium`. The
numerical tables below are auto-generated from the committed JSON
baselines by `benchmarks/speed_review/gen_findings_tables.py`; narrative
prose is hand-written and must be re-read when numbers shift.

### Scale sweep - end-to-end wall-clock

Four of the six scenarios run at three scales (small / medium / large). The
small scale matches tutorial data shapes; medium reflects typical
practitioner workloads; large stretches toward the upper end of what an
analyst might bring (1M-row BRFSS microdata, 1,500-unit county-level
staggered panel, 1,000-unit multi-region brand survey, 500-unit zip-level
geo-experiment). Dose-response and reversible-dCDH run at a single mid-range
scale. Data-shape details are in `docs/performance-scenarios.md`.

<!-- TABLE:start scale_sweep_totals -->
| Scenario | Scale | Python (s) | Rust (s) | Py/Rust |
|---|---|---:|---:|---:|
| 1. Staggered campaign | small | 0.50 | 0.50 | 1.0x |
|  | medium | 0.72 | 0.75 | 1.0x |
|  | large | 1.27 | 1.28 | 1.0x |
| 2. Brand awareness survey | small | 0.20 | 0.21 | 1.0x |
|  | medium | 0.80 | 0.51 | 1.6x |
|  | large | 0.96 | 0.88 | 1.1x |
| 3. BRFSS microdata -> CS panel | small | 1.57 | 1.65 | 1.0x |
|  | medium | 5.94 | 6.30 | 0.9x |
|  | large | 23.77 | 26.38 | 0.9x |
| 4. SDiD few markets | small | 3.68 | 0.04 | 88.9x |
|  | medium | 4.01 | 0.12 | 33.5x |
|  | large | skip | 0.27 | - |
| 5. Reversible dCDH | single | 0.84 | 0.79 | 1.1x |
| 6. Pricing dose-response | single | 0.60 | 0.63 | 1.0x |
<!-- TABLE:end scale_sweep_totals -->

### Scaling findings

**Three findings are load-bearing for the optimization priority list:**

1. **BRFSS `aggregate_survey` is the dominant practitioner pain point at
   realistic pooled-multi-year scale.** Scales near-linearly with microdata
   row count. At 1M rows (roughly what a 10-year pooled BRFSS analysis
   looks like) the full chain takes ~24 seconds and essentially all of it
   is inside `_compute_stratified_psu_meat`. Rust does not touch it
   (`aggregate_survey` is entirely Python).
2. **Staggered CS chain stays cheap across scales.** A 10x unit increase
   (150 -> 1,500) is a small-single-digit multiplier on total time.
   ImputationDiD is the dominant phase at most (scale, backend)
   combinations; SunAbraham takes the top spot at Rust medium but the
   two phases together consistently account for ~70-80% of the chain.
3. **SDiD Rust gap is stable across scales, not emergent.** Python SDiD
   has a fixed per-jackknife-refit overhead that dominates even at small
   n. Rust stays sub-second through 500 units.

**Two findings hold across scales:**

4. Brand-awareness survey total scales roughly linearly in n_units, but
   the JK1 replicate path inside it scales closer to
   n_units x n_replicates - faster growth than the chain total, so it
   increasingly dominates at large n.
5. Rust backend gives large uplift only for SDiD (order-of-magnitude
   and up). Elsewhere the gap is modest - under ~1.6x at worst on
   brand-awareness medium, and within noise on the other scenarios
   and scales. The primary bottlenecks live in Python code the Rust
   backend does not touch (`aggregate_survey`, JK1 replicate fit), and
   paths that Rust does touch (CS bootstrap, ImputationDiD, Survey
   TSL) are already well-vectorized in Python.

### Top phases by scenario at largest measured scale

<!-- TABLE:start top_phases_by_scenario -->
| Scenario | Scale | Backend | Top phase (%) | 2nd phase (%) | 3rd phase (%) |
|---|---|---|---|---|---|
| 1. Staggered campaign | large | python | `6_imputation_did_robustness` (51%) | `5_sun_abraham_robustness` (26%) | `2_cs_fit_with_covariates_bootstrap999` (13%) |
| 1. Staggered campaign | large | rust | `6_imputation_did_robustness` (43%) | `5_sun_abraham_robustness` (32%) | `2_cs_fit_with_covariates_bootstrap999` (14%) |
| 2. Brand awareness survey | large | python | `3_replicate_weights_jk1` (54%) | `4_multi_outcome_loop_3_metrics` (26%) | `7_event_study_plus_honest_did` (14%) |
| 2. Brand awareness survey | large | rust | `3_replicate_weights_jk1` (48%) | `4_multi_outcome_loop_3_metrics` (24%) | `7_event_study_plus_honest_did` (16%) |
| 3. BRFSS microdata -> CS panel | large | python | `1_aggregate_survey_microdata_to_panel` (100%) | `5_sun_abraham_robustness` (0%) | `2_cs_fit_with_stage2_survey_design` (0%) |
| 3. BRFSS microdata -> CS panel | large | rust | `1_aggregate_survey_microdata_to_panel` (100%) | `5_sun_abraham_robustness` (0%) | `2_cs_fit_with_stage2_survey_design` (0%) |
| 4. SDiD few markets | medium | python | `5_sensitivity_to_zeta_omega` (43%) | `3_in_time_placebo` (39%) | `2_sdid_bootstrap_variance_200` (9%) |
| 4. SDiD few markets | large | rust | `5_sensitivity_to_zeta_omega` (40%) | `3_in_time_placebo` (29%) | `1_sdid_jackknife_variance` (16%) |
| 5. Reversible dCDH | single | python | `1_dcdh_fit_Lmax3_survey_TSL` (58%) | `4_heterogeneity_refit` (41%) | `3_honest_did_on_placebo` (1%) |
| 5. Reversible dCDH | single | rust | `4_heterogeneity_refit` (51%) | `1_dcdh_fit_Lmax3_survey_TSL` (49%) | `3_honest_did_on_placebo` (1%) |
| 6. Pricing dose-response | single | python | `1_cdid_cubic_spline_bootstrap199` (26%) | `3_cdid_event_study_pretrend` (25%) | `6_spline_sensitivity_num_knots2` (25%) |
| 6. Pricing dose-response | single | rust | `1_cdid_cubic_spline_bootstrap199` (26%) | `3_cdid_event_study_pretrend` (25%) | `6_spline_sensitivity_num_knots2` (24%) |
<!-- TABLE:end top_phases_by_scenario -->

Per-scenario phase narrative (cross-check against the table above after
any rerun):

- **Staggered campaign.** ImputationDiD robustness and SunAbraham are
  the two largest phases at every scale, together accounting for
  ~70-80% of the chain. Their relative order is not stable across
  backend and scale: ImputationDiD is the single largest phase under
  Python at every scale and under Rust at small and large, but at
  Rust medium SunAbraham clearly leads (roughly 1.7x the ImputationDiD
  phase there). CS fit with `n_bootstrap=999` (both with and without
  covariates) is well-vectorized and sits well below both in the
  ranking.
- **Brand awareness survey.** At small scale HonestDiD dominates. At
  medium the backends diverge: on Python JK1 leads clearly (about
  2.2x the multi-outcome loop), while on Rust the multi-outcome loop
  and JK1 come in essentially tied. Medium is also the scale where
  Python and Rust separate the most on total time (~1.6x under
  Python at the time of writing); the analytical TSL path with FPC
  appears to vectorize better under Rust at that shape. At large,
  JK1 becomes the clearly dominant phase under both backends and
  totals re-converge.
- **BRFSS.** `aggregate_survey` share of total grows with scale and is
  effectively 100% of runtime at 1M rows. Downstream phases (CS fit,
  SunAbraham, HonestDiD) are a fraction of a second combined.
- **SDiD few markets.** `sensitivity_to_zeta_omega` and
  `in_time_placebo` are the two largest phases under Python at every
  scale and under Rust at medium/large (together ~70% of the chain).
  At Rust small the absolute cost collapses so far that per-phase
  fixed overhead dominates and `2_sdid_bootstrap_variance_200` slightly
  edges the other two. The difference across backends is absolute:
  under Python these phases drive a multi-second chain, under Rust
  they stay in the top ranks but of a sub-second total runtime. That
  is the Python-vs-Rust story for this scenario.
- **Reversible dCDH.** Main fit and heterogeneity refit are the two
  largest phases by design - together effectively the whole chain. The
  split is not stable across backends: under Python the main fit is
  the larger of the two (roughly 58/41), under Rust the heterogeneity
  refit slightly leads (roughly 51/49). Both fits run under the same
  `SurveyDesign` and rebuild shared TSL scaffolding - that is the
  optimization opportunity.
- **Pricing dose-response.** Four spline fits account for essentially all
  runtime; linear scaling in variant count.

### Top hotspots ranked by total-time contribution

| # | Location | Scenario + scale | Signal | Recommended action |
|---|---|---|---|---|
| 1 | `diff_diff/survey.py:1160` `_compute_stratified_psu_meat` | BRFSS @ 1M rows | dominates BRFSS chain at all scales, ~100% at 1M rows | **Algorithmic fix, highest priority.** Function called once per (state, year) cell (500 calls); per-call work rebuilds stratum-PSU scaffolding every time. Precompute stratum indexes once at `aggregate_survey` top-level and reuse. |
| 2 | `diff_diff/imputation.py` ImputationDiD fit | Staggered CS @ 1,500 units | dominant phase under Python at every scale and under Rust at small/large; at Rust medium SunAbraham takes the top spot. Together ImputationDiD + SunAbraham are ~70-80% of the chain at every scale | **Investigate only after BRFSS fix lands.** Total chain is well under practitioner-perceptible threshold; candidate follow-up. |
| 3 | `diff_diff/utils.py:1434` `_sc_weight_fw_numpy` | SDiD python @ any scale | dominates Python SDiD at all scales | **Already ported to Rust.** Python fallback acceptable as a teaching/safety path; non-production for n > 100. Python skipped at n=500 (jackknife cost would exceed 4 minutes per run). |
| 4 | `diff_diff/chaisemartin_dhaultfoeuille.py` dCDH fit + heterogeneity | Reversible (single scale) | main fit and survey-aware heterogeneity refit each rebuild TSL scaffolding; heterogeneity phase is as expensive as the main fit | **Cache/precompute** - heterogeneity refit duplicates the main fit's TSL setup under the same `SurveyDesign`. Not P0; newer code path (v3.1) never optimization-reviewed. |
| 5 | `diff_diff/continuous_did.py` CDiD spline bootstrap | Dose-response (single scale) | four spline fits ~equal, linear in variant count | **Leave alone** - well under perceptible threshold. |

### Memory analysis

End-to-end peak RSS and per-scenario growth are captured in each JSON
baseline under the `memory` field, recorded via a psutil background
sampler at 10 ms. A standalone `tracemalloc`-based allocator attribution
pass for the BRFSS-1M scenario lives at
`benchmarks/speed_review/mem_profile_brfss.py`; its scrubbed output is
in `benchmarks/speed_review/baselines/mem_profile_brfss_large_<backend>.txt`.

<!-- TABLE:start memory_by_scenario -->
| Scenario | Scale | Py peak RSS (MB) | Py growth (MB) | Rust peak RSS (MB) | Rust growth (MB) |
|---|---|---:|---:|---:|---:|
| 1. Staggered campaign | small | 141 | 27 | 147 | 33 |
|  | medium | 222 | 77 | 252 | 101 |
|  | large | 485 | 262 | 585 | 331 |
| 2. Brand awareness survey | small | 127 | 11 | 128 | 13 |
|  | medium | 189 | 56 | 184 | 50 |
|  | large | 347 | 150 | 342 | 153 |
| 3. BRFSS microdata -> CS panel | small | 134 | 12 | 134 | 13 |
|  | medium | 210 | 18 | 211 | 16 |
|  | large | 426 | 27 | 427 | 29 |
| 4. SDiD few markets | small | 123 | 10 | 115 | 2 |
|  | medium | 146 | 6 | 117 | 0 |
|  | large | skip | skip | 118 | 0 |
| 5. Reversible dCDH | single | 135 | 22 | 135 | 21 |
| 6. Pricing dose-response | single | 122 | 8 | 124 | 10 |
<!-- TABLE:end memory_by_scenario -->

The ~115-130 MB floor is the Python + diff-diff + numpy import footprint;
the "growth" columns are the practitioner-meaningful numbers.

### Memory findings

1. **BRFSS `aggregate_survey` is compute-bound, not memory-bound.** At
   20x data growth (50K -> 1M rows), working-memory growth stays in the
   low tens of MB. The tracemalloc pass confirms: net retained allocation
   after `aggregate_survey` returns is well under 1 MB; the top
   allocation site is `tracemalloc`'s own linecache overhead (a smoking
   gun that nothing else is allocating meaningfully). **The BRFSS cost
   is pure CPU; the function is already memory-efficient.** This
   strengthens the case for the precompute-scaffolding fix: low-risk,
   pure CPU win, fits in any deployment environment including 512 MB
   Lambda.
2. **Staggered CS chain is memory-heavier than wall-clock suggested.** At
   1,500 units the chain's peak RSS sits in the high-400s to high-500s
   MB depending on backend. Fine for workstations, tight for 512 MB
   Lambda tier. Bootstrap-999 in CS and ImputationDiD's saturated
   regression are plausible drivers. Rust uses slightly more memory here
   (likely FFI-held temporary array copies); not worth optimizing.
3. **JK1 replicate path is allocation-heavy at large replicate count.**
   At 1,000 units × 160 replicates the chain's growth during run sits in
   the mid-100s of MB (see memory table). Each replicate refit plus the
   n × n_replicates weight matrix drives this. A Rust port would save
   memory even though time is within noise today - the dual benefit
   strengthens the case for the port if replicate counts grow.
4. **SDiD Rust path is essentially memory-free** (growth at or below a
   single MB across scales). Rust does the work in native memory without
   round-tripping through the Python allocator. Confirms the existing
   Rust port is well-behaved on both axes.
5. **No scenario hits OOM territory at measured scales.** Peak RSS across
   the whole sweep stays under 600 MB. 1 GB is a comfortable ceiling for
   every scenario measured.

### Priority of optimization opportunities

| # | Opportunity | Time upside | Memory upside | Risk | Priority |
|---|---|---|---|---|---|
| 1 | `aggregate_survey` precompute stratum scaffolding | ~-20s at 1M rows | none (already memory-efficient) | Low | **High** |
| 2 | Staggered CS chain working-memory audit (Lambda-oriented) | none | ~200-300 MB at 1,500 units (peak RSS crosses 512 MB Lambda line under Rust) | Medium | Low (bump to Medium if Lambda deployment becomes a concrete ask) |
| 3 | dCDH: cache TSL scaffolding across main fit + heterogeneity refit | ~0.2s per chain | ~20 MB per chain | Low | Low |
| 4 | ImputationDiD fit-loop vectorization audit | ~0.1-0.3s at 1,500 units | unknown | Low | Low |
| 5 | Rust-port JK1 replicate fit loop | ~0.5s at 160 replicates | ~140 MB at 160 replicates | Medium | Low (demoted: Rust is no longer slower than Python on this path after rerun, so the "fix-a-Rust-regression" leg of the original rationale is gone) |

**Bottom line: one clear priority, four optional.** #1 is the single
practitioner-perceptible win identified by this analysis and should be
the next PR. #2-5 are optional polish that should be prioritized by
concrete deployment-environment signal (Lambda OOMs, practitioner
reports of slowness at specific shapes), not proactively.

### Correctness-adjacent observations (not P0, route separately)

These are developer-ergonomics / API-consistency smells surfaced during
scenario development. None are silent-failures and none belong in this PR
or in the silent-failures audit; logging here for awareness.

1. **`aggregate` / `level` parameter naming is inconsistent.** CS accepts
   `aggregate="event_study"`; ContinuousDiD requires
   `aggregate="eventstudy"` on `fit()` **but** `level="event_study"` on
   `to_dataframe()`. Two different spellings within one estimator plus a
   third cross-estimator spelling. Surfaced when the P1 exit-propagation
   fix stopped silently swallowing the resulting `ValueError` in the
   dose-response benchmark. Route: API-consistency cleanup, minor.
2. **`generate_survey_did_data(panel=True)` `treated` column.** Row-level
   active-treatment indicator that is zero in pre-periods, which makes it
   quietly incompatible with `check_parallel_trends` (expects unit-level
   treatment group membership) and pre-period placebo tests. Tutorial 17
   does not hit this because it uses a 2x2 design where `post` discriminates
   the comparison. Suggest adding a `treat_unit` column alongside `treated`
   for generator output clarity. Route: DGP cleanup, minor.
3. **`SurveyDesign.replicate_method` case sensitivity.** `"jk1"` raises
   `ValueError("must be one of {'Fay', 'SDR', 'BRR', 'JKn', 'JK1'}")`;
   `"JK1"` works. Either normalize the input or mention the expected casing
   in the error message. Route: API-ergonomics, minor.

### What this baseline does not answer

- OOM behaviour at the edge: the sweep captures peak RSS up to ~600 MB
  (staggered CS large under Rust). Behaviour under a hard memory ceiling
  (512 MB Lambda, 1 GB container) is not exercised; if deployment signal
  emerges that practitioners hit those ceilings, a ceiling-test pass
  should be added.
- Pure-Rust profiles: scenarios run the Rust backend as a black box.
  Optimizing inside `rust/` is a separate concern owned by the crate
  maintainers and is not in scope here.
- Real-data shapes: the scenarios use synthetic DGPs. The BRFSS scenario
  uses a BRFSS-shaped synthetic panel, not actual BRFSS microdata. If a
  real-data calibration becomes relevant, CDC BRFSS annual files are
  public.

### Reproducing

```bash
pip install pyinstrument                  # one-time, dev-only
python benchmarks/speed_review/run_all.py # both backends, all scenarios

# Single scenario, single backend:
DIFF_DIFF_BACKEND=rust python benchmarks/speed_review/bench_campaign_staggered.py
```

Raw JSON is written under `benchmarks/speed_review/baselines/` for
scenario-level diffing as the library evolves; flame HTMLs are written
alongside under `baselines/profiles/` (gitignored; regenerated on each run).

---

## Results Achieved (v2.0.3)

**v2.0.3 includes Rust backend optimizations** that further improve SyntheticDiD performance:

| Estimator | v2.0 (10K scale) | v2.0.3 (10K scale) | Speedup | vs R |
|-----------|------------------|-------------------|---------|------|
| BasicDiD/TWFE | 0.011s | **0.010s** | 1.1x | **4x faster than R** |
| CallawaySantAnna | 0.109s | **0.145s** | 0.8x | **5x faster than R** |
| SyntheticDiD (Pure) | 19.5s | **19.5s** | 1.0x | 57x faster than R |
| SyntheticDiD (Rust) | 2.6s | **2.6s** | 1.0x | **429x faster than R** |

**20K Scale Results** (new in v2.0.3 benchmarks):

| Estimator | Python Pure (s) | Python Rust (s) | R (s) | Rust vs R |
|-----------|-----------------|-----------------|-------|-----------|
| BasicDiD/TWFE | 0.022 | 0.025 | 0.050 | **2x** |
| CallawaySantAnna | 0.366 | 0.373 | 1.559 | **4x** |
| SyntheticDiD | 137.3 | **10.9** | 2451.0 | **225x** |

### What Changed in v2.0.3

1. **Cholesky factorization** for symmetric positive-definite matrix inversion (~2x faster for well-conditioned matrices)
2. **Reduced bootstrap allocations** - Direct Array2 allocation eliminates Vec<Vec<f64>> intermediate
3. **Vectorized variance computation** - HC1 meat uses BLAS-accelerated matrix operations
4. **Webb lookup table** - Faster Webb distribution weight generation
5. **Rayon chunk size tuning** - Reduced parallel scheduling overhead

---

## Results Achieved (v1.4.0)

**Phase 1 is complete.** Pure Python optimizations exceeded all targets:

| Estimator | v1.3 (10K scale) | v1.4 (10K scale) | Speedup | vs R |
|-----------|------------------|------------------|---------|------|
| BasicDiD/TWFE | 0.835s | **0.011s** | **76x** | **4.2x faster than R** |
| CallawaySantAnna | 2.234s | **0.109s** | **20x** | **7.2x faster than R** |
| SyntheticDiD | 32.6s | N/A | N/A | 37x faster than R |

### What Was Implemented

1. **Unified `linalg.py` backend** (`diff_diff/linalg.py`)
   - `solve_ols()` - scipy lstsq with gelsy LAPACK driver
   - `compute_robust_vcov()` - Vectorized cluster-robust SE via pandas groupby
   - Single optimization point for all estimators

2. **CallawaySantAnna optimizations** (`staggered.py`)
   - `_precompute_structures()` - Pre-computed wide-format outcome matrix, cohort masks
   - `_compute_att_gt_fast()` - Vectorized ATT(g,t) using numpy (23x faster)
   - `_generate_bootstrap_weights_batch()` - Batch weight generation
   - Vectorized bootstrap using matrix operations (26x faster)

3. **TWFE optimization** (`twfe.py`)
   - Cached groupby indexes for within-transformation

4. **All estimators migrated** to unified backend
   - `estimators.py`, `twfe.py`, `staggered.py`, `triple_diff.py`, `synthetic_did.py`, `sun_abraham.py`, `utils.py`

---

## Original Problem Statement

Benchmark comparisons showed that while diff-diff was competitive or faster than R for small datasets, performance degraded significantly at scale:

| Scale | BasicDiD Python | R (fixest) | Ratio |
|-------|-----------------|------------|-------|
| Small (<1K obs) | 0.003s | 0.041s | Python 16x faster |
| 5K (40-200K obs) | 0.180s | 0.046s | R 4x faster |
| 10K (100-500K obs) | 0.835s | 0.049s | R 17x faster |

| Scale | CallawaySantAnna Python | R (did) | Ratio |
|-------|-------------------------|---------|-------|
| Small | 0.048s | 0.077s | Python 1.6x faster |
| 5K | 0.793s | 0.382s | R 2x faster |
| 10K | 2.234s | 0.816s | R 2.7x faster |

Note: SyntheticDiD is already 37-1600x faster than R's synthdid package.

## Root Cause Analysis

### 1. OLS Solver (`estimators.py`)

Current implementation uses `np.linalg.lstsq` with default settings:
- General-purpose LAPACK driver (gelsd) rather than faster alternatives
- Preceded by expensive `matrix_rank()` check (O(min(n,k)^3))
- NumPy may not link to optimized BLAS

### 2. Cluster-Robust Standard Errors (`utils.py`)

Loop-based implementation:
```python
for cluster in unique_clusters:
    mask = cluster_ids == cluster  # O(n) per cluster
    ...
```
- O(n * n_clusters) complexity
- Creates boolean mask array for each cluster
- No vectorization or parallelization

### 3. Within-Transformation (`twfe.py`)

Multiple groupby operations:
```python
for var in variables:
    unit_means = data.groupby(unit)[var].transform("mean")
    time_means = data.groupby(time)[var].transform("mean")
    ...
```
- Multiple passes over data per variable
- No caching of groupby indexes
- Not using alternating projections algorithm

### 4. CallawaySantAnna Nested Loops (`staggered.py`)

```python
for g in treatment_groups:
    for t in valid_periods:
        att_gt = self._compute_att_gt(...)
```
- Repeated DataFrame indexing (`.set_index()`, `.loc[]`, `.isin()`) for each (g,t)
- No pre-computation of outcome changes
- Influence function dictionaries created per (g,t)

## Optimization Strategy

### Phase 1: Pure Python Optimizations (No New Dependencies)

Quick wins that improve performance without adding dependencies.

#### 1.1 Vectorized Cluster-Robust SE

Replace loop with vectorized groupby:
```python
scores = X * residuals[:, np.newaxis]
cluster_scores = pd.DataFrame(scores).groupby(cluster_ids).sum()
meat = cluster_scores.values.T @ cluster_scores.values
```

**Expected speedup:** 5-10x for SE computation

#### 1.2 scipy.linalg.lstsq with Optimized Driver

```python
from scipy.linalg import lstsq
coefficients = lstsq(X, y, lapack_driver='gelsy',
                     overwrite_a=True, overwrite_b=True,
                     check_finite=False)[0]
```

**Expected speedup:** 1.2-1.5x for OLS

#### 1.3 Cache Groupby Indexes

Create groupby objects once and reuse:
```python
unit_grouper = data.groupby(unit, sort=False)
time_grouper = data.groupby(time, sort=False)
```

**Expected speedup:** 1.5-2x for demeaning

#### 1.4 Pre-compute CallawaySantAnna Data Structures

Pivot to wide format once, pre-compute all period changes:
```python
outcome_wide = data.pivot(index=unit, columns=time, values=outcome)
changes = {(t0, t1): outcome_wide[t1] - outcome_wide[t0] for ...}
```

**Expected speedup:** 3-5x for CallawaySantAnna

### Phase 2: Compiled Backend

Implement performance-critical components in a compiled language for maximum speed.

#### Backend Options: Rust vs C++

We have two viable options for a compiled backend. Both can achieve near-identical performance; the choice depends on team expertise and maintenance considerations.

##### Option A: Rust with PyO3

**Pros:**
- **Memory safety by design** - No segfaults, buffer overflows, or data races; compiler catches these at build time
- **Modern tooling** - Cargo package manager + maturin makes wheel building straightforward
- **Zero-copy NumPy interop** - rust-numpy crate provides direct array access without copying
- **Easy parallelism** - rayon crate makes parallel iteration trivial (`.par_iter()`)
- **Growing ecosystem** - Used by polars, pyfixest, cryptography, orjson, ruff
- **Low per-call overhead** - Research shows PyO3 has ~0.14ms overhead vs NumPy's ~3.5ms for simple operations
- **Single toolchain** - `cargo build` works the same on all platforms

**Cons:**
- **Learning curve** - Rust's ownership model takes time to learn
- **Smaller scientific ecosystem** - Fewer numerical libraries than C++ (though ndarray and faer are mature)
- **Slower compilation** - Rust compiles slower than C++
- **Newer language** - Less institutional knowledge, fewer Stack Overflow answers

**Key dependencies:** `pyo3`, `rust-numpy`, `ndarray`, `faer` (linear algebra), `rayon` (parallelism)

##### Option B: C++ with pybind11

**Pros:**
- **Mature ecosystem** - Eigen, Armadillo, Intel MKL, OpenBLAS all native C++
- **Familiar to more developers** - Larger pool of contributors
- **Proven in scientific Python** - NumPy, SciPy, scikit-learn, pandas all use C/C++ extensions
- **Excellent Eigen integration** - pybind11 has built-in support for Eigen matrices
- **Faster compilation** - C++ compiles faster than Rust
- **More optimization resources** - Decades of C++ performance tuning knowledge

**Cons:**
- **Memory safety risks** - Segfaults, buffer overflows, use-after-free possible; harder to debug
- **Manual memory management** - Must carefully manage lifetimes, especially with Python GC interaction
- **Complex build systems** - CMake configuration, compiler flags, platform-specific issues
- **Copy overhead by default** - pybind11 copies arrays unless carefully configured with `py::array_t`
- **Manual GIL management** - Easy to deadlock or corrupt state if GIL not handled correctly
- **Platform differences** - MSVC vs GCC vs Clang have different behaviors and flags

**Key dependencies:** `pybind11`, `Eigen` (linear algebra), `OpenMP` or `TBB` (parallelism)

##### Comparison Summary

| Factor | Rust (PyO3) | C++ (pybind11) |
|--------|-------------|----------------|
| Memory safety | Compile-time guarantees | Runtime risks |
| Build tooling | Cargo + maturin (simple) | CMake + scikit-build (complex) |
| NumPy interop | Zero-copy via rust-numpy | Zero-copy possible but tricky |
| Parallelism | rayon (trivial) | OpenMP/TBB (more boilerplate) |
| Linear algebra | faer, ndarray-linalg | Eigen, MKL, OpenBLAS |
| Ecosystem maturity | Growing | Established |
| Learning curve | Steeper (ownership) | Moderate (but footguns) |
| Wheel building | maturin-action (simple) | cibuildwheel (more config) |
| Debug experience | Good (cargo, clippy) | Variable (platform-dependent) |

##### Recommendation

**Rust with PyO3** is the recommended approach because:

1. **pyfixest validates this for our exact domain** - They use Rust/PyO3 for fixed effects econometrics
2. **Memory safety prevents production bugs** - No risk of segfaults in user code
3. **maturin simplifies distribution** - Single command builds wheels for all platforms
4. **rayon makes parallelization trivial** - Critical for bootstrap and cluster SE

However, **C++ is a viable alternative** if:
- Team has stronger C++ expertise
- Need to integrate with existing C++ econometrics code
- Want to leverage Eigen's mature linear algebra

#### Graceful Degradation

```python
try:
    from diff_diff._rust_backend import solve_ols_clustered
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

def _fit_ols(self, X, y, cluster_ids=None):
    if _HAS_RUST and self.backend == 'rust':
        return solve_ols_clustered(X, y, cluster_ids)
    else:
        # Existing NumPy implementation
        ...
```

#### Components to Implement in Rust

| Component | Current Bottleneck | Rust Benefit |
|-----------|-------------------|--------------|
| Cluster-robust SE | O(n * clusters) loop | rayon parallel iteration |
| Within-transformation | Multiple groupby passes | Single-pass with hash tables |
| OLS solving | NumPy lstsq overhead | faer or direct LAPACK |
| Bootstrap resampling | Sequential iterations | Embarrassingly parallel |
| ATT(g,t) computation | Repeated DataFrame indexing | Pre-indexed sparse structures |

#### Architecture by Backend

##### Rust Layout

```
diff_diff/
├── estimators.py          # Python API (unchanged)
├── _rust_backend/         # Compiled Rust module
│   └── ...
└── _fallback.py           # Pure Python fallback

src/                       # Rust source (Cargo workspace)
├── Cargo.toml
├── lib.rs
├── ols.rs                 # OLS with cluster SE
├── demeaning.rs           # Alternating projections
├── bootstrap.rs           # Parallel bootstrap
└── staggered.rs           # ATT(g,t) computation

pyproject.toml             # maturin build config
```

##### C++ Layout

```
diff_diff/
├── estimators.py          # Python API (unchanged)
├── _cpp_backend/          # Compiled C++ module
│   └── ...
└── _fallback.py           # Pure Python fallback

cpp/                       # C++ source
├── CMakeLists.txt
├── src/
│   ├── module.cpp         # pybind11 bindings
│   ├── ols.cpp            # OLS with cluster SE
│   ├── ols.hpp
│   ├── demeaning.cpp      # Within transformation
│   ├── demeaning.hpp
│   ├── bootstrap.cpp      # Parallel bootstrap
│   └── bootstrap.hpp
└── extern/
    └── eigen/             # Eigen submodule (or system install)

pyproject.toml             # scikit-build-core config
```

#### Distribution

##### Rust (maturin)

```yaml
# .github/workflows/wheels.yml
- uses: PyO3/maturin-action@v1
  with:
    command: build
    args: --release --out dist
```

- Simple single-action CI configuration
- Use abi3 stable ABI for Python version-independent wheels
- Cross-compilation via `--target` flag

##### C++ (cibuildwheel)

```yaml
# .github/workflows/wheels.yml
- uses: pypa/cibuildwheel@v2
  env:
    CIBW_BUILD: "cp39-* cp310-* cp311-* cp312-*"
```

- More configuration required for CMake integration
- Need to handle OpenMP linking per-platform
- Consider vcpkg or conan for dependency management

Both approaches build wheels for:
- Linux (manylinux2014, x86_64 and aarch64)
- macOS (x86_64 and ARM64)
- Windows (x86_64)

## Implementation Roadmap

| Phase | Scope | Effort | Expected Speedup |
|-------|-------|--------|------------------|
| 1.1 | Vectorize cluster SE | 1-2 days | 5-10x (SE only) |
| 1.2 | scipy lstsq optimization | 1 day | 1.2-1.5x (OLS) |
| 1.3 | Cache groupby indexes | 1 day | 1.5-2x (demeaning) |
| 1.4 | Pre-compute CS structures | 2-3 days | 3-5x (CS) |
| 2.1 | Rust cluster SE | 1-2 weeks | 10-50x (SE) |
| 2.2 | Rust parallel bootstrap | 1 week | 5-20x (bootstrap) |
| 2.3 | Rust demeaning | 2 weeks | 3-10x (TWFE) |
| 2.4 | Rust OLS solver | 2 weeks | Match R |
| 2.5 | Rust staggered ATT | 2-3 weeks | 5-10x (CS) |
| 2.6 | CI/CD wheel building | 1 week | N/A |

## Outcomes

### Phase 1 Results (v1.4.0) ✅

**Exceeded all targets:**

- BasicDiD @ 10K: 0.835s → **0.011s** (76x improvement, 4.2x faster than R)
- CallawaySantAnna @ 10K: 2.2s → **0.109s** (20x improvement, 7.2x faster than R)
- Bootstrap inference: 26x faster via vectorization

### Phase 2 (Rust Backend) - Optional Future Work

No longer required for R parity. May be pursued for:
- Further optimization at extreme scales (100K+ units)
- Parallel bootstrap across CPU cores
- Memory efficiency for very large datasets

## References

### Rust Backend

- [PyO3 User Guide](https://pyo3.rs/) - Rust bindings for Python
- [rust-numpy](https://github.com/PyO3/rust-numpy) - Zero-copy NumPy interop
- [maturin](https://github.com/PyO3/maturin) - Build and publish Rust Python packages
- [faer](https://github.com/sarah-ek/faer-rs) - Pure Rust linear algebra (competitive with MKL)
- [Polars](https://github.com/pola-rs/polars) - Example of Rust/Python hybrid architecture
- [pyfixest](https://github.com/py-econometrics/pyfixest) - Rust backend for fixed effects econometrics

### C++ Backend

- [pybind11 documentation](https://pybind11.readthedocs.io/) - C++ bindings for Python
- [pybind11 Eigen integration](https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html) - Zero-copy with Eigen
- [Eigen](https://eigen.tuxfamily.org/) - C++ linear algebra library
- [scikit-build-core](https://scikit-build-core.readthedocs.io/) - CMake integration for Python packages
- [cibuildwheel](https://cibuildwheel.readthedocs.io/) - Build wheels for all platforms

### General

- [fixest demeaning algorithm](https://rdrr.io/cran/fixest/man/demeaning_algo.html) - Reference implementation
- [PyO3 vs C performance comparison](https://www.alphaxiv.org/overview/2507.00264v1) - Academic benchmark
- [Making Python 100x faster with Rust](https://ohadravid.github.io/posts/2023-03-rusty-python/) - Practical tutorial
