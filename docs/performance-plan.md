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
scales under both `DIFF_DIFF_BACKEND=python` and `DIFF_DIFF_BACKEND=rust`.

### Scale sweep - end-to-end wall-clock

Four of the six scenarios run at three scales (small / medium / large). The
small scale matches tutorial data shapes; medium reflects typical
practitioner workloads; large stretches toward the upper end of what an
analyst might bring (1M-row BRFSS microdata, 1,500-unit county-level
staggered panel, 1,000-unit multi-region brand survey, 500-unit zip-level
geo-experiment). Dose-response and reversible-dCDH run at a single mid-range
scale.

| Scenario | Scale | Data shape | Python (s) | Rust (s) | Py/Rust |
|---|---|---|---:|---:|---:|
| **1. Staggered campaign** | small | 150 units × 26 periods | 0.48 | 0.49 | 1.0x |
| (CS + 8-step chain, bootstrap 999) | medium | 500 units × 26 periods | 0.72 | 0.87 | 0.8x |
|  | large | 1,500 units × 26 periods | 1.24 | 1.22 | 1.0x |
| **2. Brand awareness survey** | small | 200 units × 12 periods, 40 JK1 reps | 0.21 | 0.20 | 1.0x |
| (DiD + SurveyDesign + JK1 replicate weights) | medium | 500 units × 12 periods, 90 JK1 reps | 0.52 | 0.46 | 1.1x |
|  | large | 1,000 units × 12 periods, 160 JK1 reps | 0.83 | 0.92 | 0.9x |
| **3. BRFSS microdata → CS panel** | small | 50K rows → 500 cells | 1.59 | 1.61 | 1.0x |
| (`aggregate_survey` + CS + HonestDiD) | medium | 250K rows → 500 cells | 6.11 | 6.20 | 1.0x |
|  | large | **1M rows → 500 cells** | **23.96** | **24.37** | **1.0x** |
| **4. Geo-experiment few markets (SDiD)** | small | 80 units, 5 treated | 3.05 | 0.04 | **76x** |
| (jackknife + bootstrap + sensitivity chain) | medium | 200 units, 15 treated | 3.65 | 0.12 | **31x** |
|  | large | 500 units, 30 treated | skip | 0.26 | - |
| 5. Reversible dCDH (L_max=3 + TSL) | (single) | 120 groups × 10 periods | 0.55 | 0.53 | 1.0x |
| 6. Pricing dose-response (CDiD spline) | (single) | 500 units × 6 periods | 0.58 | 0.59 | 1.0x |

### Scaling findings

**Three findings invert at large scale relative to the tutorial-scale pass:**

1. **BRFSS `aggregate_survey` becomes the dominant practitioner pain point.**
   Scales near-linearly with microdata row count - 50K → 1M rows (20x)
   costs 15x runtime (1.5s → 24s). At 1M rows, 97% of runtime is inside
   `_compute_stratified_psu_meat`, called once per output cell. This is a
   concrete 20-second cost hit on any realistic pooled multi-year BRFSS
   study, and Rust does not touch it (aggregate_survey is entirely Python).
2. **Staggered CS chain remains cheap across scales.** 150 → 1,500 units
   (10x) increases total by only 2.6x (0.48s → 1.24s). ImputationDiD stays
   the dominant phase (46-62%) but scales well; absolute time at
   practitioner scale is still under 1 second.
3. **SDiD Rust gap is stable, not emergent.** Python SDiD at 80 units is
   already 3 seconds; at 200 units it is 3.7 seconds. The cost is
   dominated by fixed-overhead-per-jackknife-refit rather than data size;
   Rust stays sub-second through 500 units. The 76x headline at small scale
   is driven by Python having ~3s of baseline cost, not by bad scaling.

**Two findings hold across scales:**

4. Brand-awareness survey chain scales roughly linearly in n_units
   (0.21s → 0.83s for a 5x unit increase); the JK1 replicate-weight path
   itself scales closer to n_units × n_replicates (40 → 160 replicates
   across the sweep), becoming the dominant phase at large scale.
5. Rust backend gives measurable uplift only for SDiD; for everything else
   backend choice is within noise because the bottlenecks are in Python
   (`aggregate_survey`) or already well-vectorized (CS bootstrap, ImputationDiD,
   Survey TSL/replicate).

### Top hotspots ranked by total-time contribution (at largest measured scale)

| # | Location | Scenario + scale | Time | Recommended action |
|---|---|---|---:|---|
| 1 | `diff_diff/survey.py:1160` `_compute_stratified_psu_meat` | BRFSS @ 1M rows | **20.7s self + 22.6s inclusive** | **Algorithmic fix, raised priority.** Function called once per (state, year) cell (500 calls); per-call work scales with cell size and rebuilds stratum-PSU scaffolding every time. Precompute stratum indexes once at `aggregate_survey` top-level and reuse. Upside at practitioner scale is now 15-20 seconds, not 1.5 seconds. |
| 2 | `diff_diff/imputation.py` ImputationDiD fit | Staggered CS @ 1,500 units | 0.66s (53%) | **Investigate if/when BRFSS fix lands.** Stayed the dominant phase across scales but total chain is ~1.2s at large - not P0. Still a candidate follow-up once the higher-value fix is in. |
| 3 | `diff_diff/utils.py:1434` `_sc_weight_fw_numpy` | SDiD python @ any scale | 3s fixed overhead + scaling | **Already ported to Rust.** Python fallback acceptable as a teaching/safety path; document as non-production for n > 100. Python skipped at n=500 because jackknife 500 refits × ~500ms/refit would exceed 4 minutes. |
| 4 | `diff_diff/chaisemartin_dhaultfoeuille.py` dCDH fit + heterogeneity | Reversible (single scale) | 0.32s main + 0.22s heterogeneity | **Cache/precompute** - heterogeneity refit rebuilds TSL scaffolding the main fit already computed. Not P0 - total is ~550ms - but newer code path (v3.1) never optimization-reviewed. |
| 5 | `diff_diff/continuous_did.py` CDiD spline bootstrap | Dose-response (single scale) | 0.14s per fit × 4 variants | **Leave alone** - linear in variant count, all well under perceptible threshold. |

### Per-scenario phase rankings at each scale

**Scenario 1 - Staggered campaign (CS + 8-step chain).**
ImputationDiD robustness remains the single dominant phase at every scale
(0.30s / 0.33s / 0.66s for small / medium / large). SunAbraham scales at
similar rate. The CS fit with `n_bootstrap=999` at 1,500 units is 0.18s
(15%) - well-vectorized. Action: investigate ImputationDiD only after
higher-upside items land.

**Scenario 2 - Brand awareness survey.**
At small scale HonestDiD dominates (42%); at medium the multi-outcome loop
and the JK1 replicate-weight path are within a factor of 2 (23-36%); at
large the JK1 path becomes the single top phase (~45-50%, 0.37s Python /
0.46s Rust). Replicate count grows with PSU count (40 / 90 / 160 at the
three scales), so the path scales roughly as n_units × n_replicates - a
near-quadratic curve in a design dimension that commonly grows. Note that
Rust is marginally slower than Python here because the JK1 replicate-fit
loop is not yet Rust-accelerated and the FFI crossings cost more than the
per-fit work. Action: leave alone, but flag the JK1 path as a Rust-port
candidate if practitioners regularly run n_replicates >= 160.

**Scenario 3 - BRFSS microdata → CS panel.**
`aggregate_survey` share of total grows with scale: 94% at 50K → 99% at 250K →
100% at 1M. Everything downstream (CS fit, SunAbraham, HonestDiD) stays
under 500 ms combined. Action: fix `aggregate_survey` per-cell loop. This
is now the single most impactful optimization identified.

**Scenario 4 - Geo-experiment few markets (SDiD).**
`sensitivity_to_zeta_omega` and `in_time_placebo` are the dominant
python-backend phases at every scale (together ~70%); Rust eliminates both.
Action: no further optimization needed - Rust port ships the answer.

**Scenario 5 - Reversible treatment (dCDH L_max=3 + TSL).**
Unchanged from single-scale pass: main fit 58% + heterogeneity refit 40%,
both rebuilding shared TSL scaffolding.

**Scenario 6 - Pricing dose-response (ContinuousDiD).**
Unchanged: four spline fits ~140ms each, ~99% of total.

### Memory analysis

End-to-end peak RSS and per-scenario growth are captured in each JSON
baseline under the `memory` field, recorded via a psutil background
sampler at 10 ms. A standalone `tracemalloc`-based allocator attribution
pass for the BRFSS-1M scenario lives at
`benchmarks/speed_review/mem_profile_brfss.py`; its output is in
`benchmarks/speed_review/baselines/mem_profile_brfss_large_<backend>.txt`.

| Scenario | Scale | Peak RSS (Py) | Growth during run (Py) | Peak RSS (Rust) | Growth (Rust) |
|---|---|---:|---:|---:|---:|
| Staggered campaign | small | 141 MB | +26 | 147 MB | +33 |
|  | medium | 226 MB | +81 | 263 MB | +109 |
|  | **large** | **486 MB** | **+252** | **589 MB** | **+322** |
| Brand awareness survey | small | 126 MB | +10 | 130 MB | +13 |
|  | medium | 188 MB | +56 | 185 MB | +52 |
|  | large | 336 MB | +146 | 315 MB | +127 |
| BRFSS microdata -> CS panel | small (50K) | 131 MB | +10 | 134 MB | +11 |
|  | medium (250K) | 206 MB | +19 | 214 MB | +24 |
|  | **large (1M)** | **419 MB** | **+23** | **428 MB** | **+29** |
| SDiD few markets | small | 124 MB | +10 | 115 MB | +1 |
|  | medium | 152 MB | +8 | 117 MB | 0 |
|  | large | skip | skip | 117 MB | 0 |
| Reversible dCDH | single | 131 MB | +18 | 136 MB | +22 |
| Dose-response | single | 120 MB | +6 | 122 MB | +9 |

The ~115-130 MB floor is the Python + diff-diff + numpy import footprint;
the "growth during run" column is the practitioner-meaningful number.

### Memory findings

1. **BRFSS `aggregate_survey` is compute-bound, not memory-bound.** Across
   a 20x data growth (50K → 1M rows), working-memory growth only goes
   10 → 19 → 23 MB. The tracemalloc pass confirms this: net retained
   allocation after `aggregate_survey` returns is 0.6 MB, Python traced
   peak is 84 MB (vs 46 MB input microdata), and the top allocation site
   is `tracemalloc`'s own `linecache.py` overhead - a smoking gun that
   nothing else is allocating meaningfully. **The 24-second cost is pure
   CPU; the function is already memory-efficient.** This strengthens the
   case for the precompute-scaffolding fix: low-risk, pure CPU win, fits
   in any deployment environment including 512 MB Lambda.

2. **Staggered CS chain is memory-heavier than wall-clock suggested.** At
   1,500 units the chain allocates +252 MB Python / +322 MB Rust during
   the run, pushing peak RSS to ~486-589 MB. Fine for workstations,
   tight for 512 MB Lambda tier. The Bootstrap-999 in CS and ImputationDiD's
   saturated regression are the plausible drivers. Not a P0 today but
   worth flagging for future edge / Lambda deployments. Interestingly,
   Rust uses **more** memory here (70 MB more at large scale), likely
   FFI-held temporary array copies; not worth optimizing.

3. **JK1 replicate path is allocation-heavy at scale.** At 1,000 units ×
   160 replicates, +127-146 MB growth. Each replicate refit plus the
   n × n_replicates weight matrix drives this. A Rust port would save
   both time (0.3-0.4s) and memory (~100 MB) - the dual benefit slightly
   strengthens the case for the port.

4. **SDiD Rust path is essentially memory-free** (+0-1 MB across scales).
   Rust does the work in native memory without round-tripping through
   the Python allocator. Confirms the existing Rust port is well-behaved
   on both axes.

5. **No scenario hits OOM territory at measured scales.** Maximum peak
   RSS across the whole sweep is 589 MB (staggered CS large + Rust).
   1 GB is a comfortable ceiling for every scenario measured.

### Priority of optimization opportunities

| # | Opportunity | Time upside | Memory upside | Risk | Priority |
|---|---|---|---|---|---|
| 1 | `aggregate_survey` precompute stratum scaffolding | -15 to -20s at 1M rows | none (already memory-efficient) | Low | **High** |
| 2 | Rust-port JK1 replicate fit loop | -0.3s at 160 replicates | -100 MB at 160 replicates | Medium | Medium |
| 3 | dCDH: cache TSL scaffolding across main fit + heterogeneity refit | -0.2s per chain | -20 MB per chain | Low | Low |
| 4 | ImputationDiD fit-loop vectorization audit | -0.1 to -0.3s at 1,500 units | unknown | Low | Low |
| 5 | Staggered CS chain working-memory audit (Lambda-oriented) | none | -100+ MB at 1,500 units | Medium | Low |

#1 is the single clearest practitioner win. Everything else is optional
polish that should be prioritized by actual deployment-environment signal
(e.g. "our practitioners keep hitting 512 MB Lambda limits on the
staggered chain" → item 5 moves up).

### Correctness-adjacent observations (not P0, route separately)

These are developer-ergonomics / API-consistency smells surfaced during
scenario development. None are silent-failures and none belong in this PR
or in the silent-failures audit; logging here for awareness.

1. **`aggregate` parameter naming.** CS accepts `aggregate="event_study"`;
   ContinuousDiD requires `aggregate="eventstudy"` (no underscore). Both
   estimators expose the same conceptual aggregation but different
   spellings. Route: API-consistency cleanup, minor.
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

- Scaling: each scenario runs at a single data shape. We do not know how
  end-to-end time scales with n, periods, or cohorts. If scaling becomes a
  decision input, add a small per-scenario scale sweep (e.g., n_units in
  {100, 500, 1000}) - the scripts are parameterised to support this.
- Memory: no memory-ceiling measurement. If memory becomes a concern,
  `pyinstrument --output-memory` or `memray` can be wrapped into
  `bench_shared.run_scenario` without restructuring.
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
