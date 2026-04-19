# Performance Improvement Plan

This document outlines the strategy for improving diff-diff's performance on large datasets, particularly for BasicDiD/TWFE and CallawaySantAnna estimators.

---

## Practitioner Workflow Baseline (v3.1.3, April 2026)

Earlier sections of this document (v1.4.0, v2.0.3) measured isolated `fit()`
calls on synthetic panels for R-parity. This section measures **end-to-end
practitioner chains** — Bacon decomposition, fit, event-study pre-trend
inspection, HonestDiD sensitivity grids, cross-estimator robustness refits,
and reporting — at data shapes anchored to applied-econ papers and industry
writeups. The six scenarios are defined in
[`docs/performance-scenarios.md`](performance-scenarios.md); scripts live in
`benchmarks/speed_review/bench_*.py`; raw results in
`benchmarks/results/*.json` and flame profiles in
`benchmarks/results/profiles/`.

Environment: macOS darwin 25.3 on Apple Silicon M4, Python 3.9,
numpy 2.x, diff_diff 3.1.3. Each scenario runs under
`DIFF_DIFF_BACKEND=python` and `DIFF_DIFF_BACKEND=rust`.

### Per-scenario wall-clock totals

| Scenario | Python (s) | Rust (s) | Rust speedup | Dominant phase |
|---|---:|---:|---:|---|
| 1. Staggered campaign (CS + 8-step chain) | 0.48 | 0.48 | 1.0x | ImputationDiD robustness (53%) |
| 2. Brand awareness survey (DiD + SurveyDesign) | 0.18 | 0.20 | 0.9x | Multi-outcome loop + HonestDiD (~70% combined) |
| 3. BRFSS microdata -> CS panel | 1.58 | 1.58 | 1.0x | `aggregate_survey` (93%) |
| 4. Geo-experiment few markets (SDiD) | 2.96 | 0.04 | **76x** | SDiD Frank-Wolfe weight solver |
| 5. Reversible treatment (dCDH L_max=3 + TSL) | 0.49 | 0.55 | 0.9x | dCDH fit (58%) + heterogeneity refit (40%) |
| 6. Pricing dose-response (CDiD spline) | 0.57 | 0.58 | 1.0x | Four spline variants, ~25% each |

At practitioner-realistic scales, the full 8-step Baker chain runs in under
two seconds for 5 of 6 scenarios with or without the Rust backend. The Rust
backend provides dramatic uplift only for SDiD; elsewhere it is at parity
(or marginally slower on small data due to the Python/Rust FFI crossing
overhead).

### Top hotspots ranked by total-time contribution

| # | Location | Scenario | Time | Recommended action |
|---|---|---|---:|---|
| 1 | `diff_diff/survey.py:1160` `_compute_stratified_psu_meat` | BRFSS | 1.0s self + 1.4s inclusive per 50K microdata | **Algorithmic fix** — loop runs per (state, year) cell; precompute stratum scaffolding once at top of `aggregate_survey` and reuse |
| 2 | `diff_diff/utils.py:1434` `_sc_weight_fw_numpy` | Geo few markets (python) | 0.46s | **Already ported to Rust.** Python fallback acceptable for n < 50; document the python-backend ceiling rather than re-optimizing |
| 3 | `diff_diff/imputation.py` ImputationDiD fit chain | Staggered campaign | 0.24s | **Investigate** — 4x slower than CS with `n_bootstrap=999` on identical data; unexpected given CS has the heavier bootstrap and same influence-function path. Likely imputation loop is not vectorized. |
| 4 | `diff_diff/chaisemartin_dhaultfoeuille.py` dCDH fit (`L_max=3` + TSL) | Reversible | 0.32s main + 0.22s heterogeneity refit | **Cache/precompute** — heterogeneity refit repeats TSL setup and data prep already done by main fit. Pass shared precomputed structures through. |
| 5 | `diff_diff/continuous_did.py` CDiD bootstrap loop | Dose response | 0.14s per fit, 4 variants = 0.56s | **Leave alone** — linear scaling with spline variants is expected; total well under practitioner-perceptible threshold |

### Per-scenario findings

**Scenario 1 — Staggered campaign (CS + 8-step chain)**

Top 5 phases (python-backend, ordered by time):

1. `6_imputation_did_robustness` — 234 ms (49%) — **investigate**
2. `5_sun_abraham_robustness` — 149 ms (31%) — expected; SA saturated TWFE
3. `2_cs_fit_with_covariates_bootstrap999` — 59 ms (12%) — expected
4. `7_cs_without_covariates` — 29 ms (6%) — expected
5. `1_bacon_decomposition` — 7 ms (1%) — negligible

Action: flag ImputationDiD for a focused profile comparison against CS on
the same data; total scenario is otherwise already cheap enough.

**Scenario 2 — Brand awareness survey**

Top 5 phases (python-backend, ordered by time):

1. `4_multi_outcome_loop_3_metrics` — 64 ms (36%) — expected; linear in outcome count
2. `7_event_study_plus_honest_did` — 62 ms (35%) — expected; MP fit + 3x HonestDiD
3. `6_placebo_refit_pre_period` — 24 ms (13%) — expected
4. `3_replicate_weights_brr` — 12 ms (7%) — expected; 40 replicate columns
5. `5_check_parallel_trends` — 9 ms (5%) — expected

Action: **leave alone.** Full survey chain is ~200 ms end-to-end.

**Scenario 3 — BRFSS microdata -> CS panel**

Top 5 phases (python-backend, ordered by time):

1. `1_aggregate_survey_microdata_to_panel` — 1480 ms (94%) — **algorithmic fix**
2. `5_sun_abraham_robustness` — 81 ms (5%) — expected
3. `2_cs_fit_with_stage2_survey_design` — 15 ms (1%) — expected
4. `4_honest_did_grid` — 4 ms — negligible
5. `6_practitioner_next_steps` — <1 ms — negligible

Action: **fix `aggregate_survey` per-cell loop.** Profile confirmed the
self-time is concentrated in `_compute_stratified_psu_meat` being called
once per output cell (500 cells for 50 states x 10 years) with redundant
stratum-scaffolding reconstruction per call. A single precomputation of
stratum indexes at the top of `aggregate_survey` should eliminate most of
the 1s self-time without changing numerical output.

**Scenario 4 — Geo-experiment few markets (SDiD)**

Top 5 phases (python vs rust):

| Phase | Python | Rust |
|---|---:|---:|
| `5_sensitivity_to_zeta_omega` | 1059 ms | 11 ms |
| `3_in_time_placebo` | 954 ms | 8 ms |
| `2_sdid_bootstrap_variance_200` | 475 ms | 12 ms |
| `1_sdid_jackknife_variance` | 472 ms | 7 ms |

Profile of python fit: 99% of time is in `_sc_weight_fw_numpy` Frank-Wolfe
solver, split ~evenly between unit-weight and time-weight solves.
`_fw_step` convergence check (`np.allclose`) is half the inner-loop cost.

Action: **no further optimization needed.** Rust port is shipped and
provides 76x on the full chain. The practitioner path defaults to Rust when
available; the python fallback is a developer-safety path and the
performance ceiling is acceptable for the teaching scale
(40-80 units) but documented as non-production for larger n.

**Scenario 5 — Reversible treatment (dCDH L_max=3 + TSL)**

Top 5 phases:

1. `1_dcdh_fit_Lmax3_survey_TSL` — 316 ms (64% python / 58% rust) — **cache candidate**
2. `4_heterogeneity_refit` — 174 ms (35%) — **cache candidate**
3. `3_honest_did_on_placebo` — 4-13 ms — expected

The main fit and heterogeneity refit each independently rebuild TSL
scaffolding (stratum-PSU indexes, influence-function allocators, design-
matrix reshaping). Because heterogeneity always follows an unconditional
fit, the scaffolding is shared and can be passed through.

Action: **investigate shared precomputation.** Not a P0 — total is ~550 ms
end-to-end — but this is a newer code path (v3.1) and has not been
optimization-reviewed.

**Scenario 6 — Pricing dose-response (ContinuousDiD)**

Four spline fits (cubic bootstrap 199, event-study, linear bootstrap 199,
cubic num_knots=2 bootstrap 199) account for ~99% of runtime, ~140 ms each.
Linear scaling in variant count is expected.

Action: **leave alone.** Bootstrap 199 on 500 units x 6 periods with cubic
splines at 140 ms per fit is well within practitioner-acceptable latency.

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
3. **`SurveyDesign.replicate_method` case sensitivity.** `"brr"` raises
   `ValueError("must be one of {'Fay', 'SDR', 'BRR', 'JKn', 'JK1'}")`;
   `"BRR"` works. Either normalize the input or mention the expected casing
   in the error message. Route: API-ergonomics, minor.

### What this baseline does not answer

- Scaling: each scenario runs at a single data shape. We do not know how
  end-to-end time scales with n, periods, or cohorts. If scaling becomes a
  decision input, add a small per-scenario scale sweep (e.g., n_units in
  {100, 500, 1000}) — the scripts are parameterised to support this.
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

Raw JSON and flame HTML are written under `benchmarks/results/` for
scenario-level diffing as the library evolves.

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
