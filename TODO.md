# Development TODO

Internal tracking for technical debt, known limitations, and maintenance tasks.

For the public feature roadmap, see [ROADMAP.md](ROADMAP.md).

---

## Known Limitations

Current limitations that may affect users:

| Issue | Location | Priority | Notes |
|-------|----------|----------|-------|
| MultiPeriodDiD wild bootstrap not supported | `estimators.py:778-784` | Low | Edge case |
| `predict()` raises NotImplementedError | `estimators.py:567-588` | Low | Rarely needed |

For survey-specific limitations (NotImplementedError paths), see the
[consolidated deferred list](docs/survey-roadmap.md#deferred-work-consolidated)
in survey-roadmap.md.

## Code Quality

### Large Module Files

Target: < 1000 lines per module for maintainability. Updated 2026-03-29.

| File | Lines | Action |
|------|-------|--------|
| `power.py` | 2588 | Consider splitting (power analysis + MDE + sample size) |
| `linalg.py` | 2289 | Monitor — unified backend, splitting would hurt cohesion |
| `staggered.py` | 2275 | Monitor — grew with survey support |
| `imputation.py` | 2009 | Monitor |
| `triple_diff.py` | 1921 | Monitor |
| `utils.py` | 1902 | Monitor |
| `two_stage.py` | 1708 | Monitor |
| `survey.py` | 1646 | Monitor — grew with Phase 6 features |
| `continuous_did.py` | 1626 | Monitor |
| `honest_did.py` | 1511 | Acceptable |
| `sun_abraham.py` | 1540 | Acceptable |
| `estimators.py` | 1357 | Acceptable |
| `trop_local.py` | 1261 | Acceptable |
| `trop_global.py` | 1251 | Acceptable |
| `prep.py` | 1225 | Acceptable |
| `pretrends.py` | 1105 | Acceptable |
| `trop.py` | 981 | Split done — trop_global.py + trop_local.py |
| `visualization/` | 4172 | Subpackage (split across 7 files) — OK |

---

### Tech Debt from Code Reviews

Deferred items from PR reviews that were not addressed before merge.

#### Methodology/Correctness

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| CallawaySantAnna: consider materializing NaN entries for non-estimable (g,t) cells in group_time_effects dict (currently omitted with consolidated warning); would require updating downstream consumers (event study, balance_e, aggregation) | `staggered.py` | #256 | Low |
| ImputationDiD dense `(A0'A0).toarray()` scales O((U+T+K)^2), OOM risk on large panels | `imputation.py` | #141 | Medium (deferred — only triggers when sparse solver fails) |
| Multi-absorb weighted demeaning needs iterative alternating projections for N > 1 absorbed FE with survey weights; unweighted multi-absorb also uses single-pass (pre-existing, exact only for balanced panels) | `estimators.py` | #218 | Medium |
| EfficientDiD `control_group="last_cohort"` trims at `last_g - anticipation` but REGISTRY says `t >= last_g`. With `anticipation=0` (default) these are identical. With `anticipation>0`, code is arguably more conservative (excludes anticipation-contaminated periods). Either align REGISTRY with code or change code to `t < last_g` — needs design decision. | `efficient_did.py` | #230 | Low |
| TripleDifference power: `generate_ddd_data` is a fixed 2×2×2 cross-sectional DGP — no multi-period or unbalanced-group support. Add a `generate_ddd_panel_data` for panel DDD power analysis. | `prep_dgp.py`, `power.py` | #208 | Low |
| Survey design resolution/collapse patterns are inconsistent across panel estimators — ContinuousDiD rebuilds unit-level design in SE code, EfficientDiD builds once in fit(), StackedDiD re-resolves on stacked data; extract shared helpers for panel-to-unit collapse, post-filter re-resolution, and metadata recomputation | `continuous_did.py`, `efficient_did.py`, `stacked_did.py` | #226 | Low |
| Survey-weighted Silverman bandwidth in EfficientDiD conditional Omega* — `_silverman_bandwidth()` uses unweighted mean/std for bandwidth selection; survey-weighted statistics would better reflect the population distribution but is a second-order refinement | `efficient_did_covariates.py` | — | Low |
| TROP: `fit()` and `_fit_global()` share ~150 lines of near-identical data setup (panel pivoting, absorbing-state validation, first-treatment detection, effective rank, NaN warnings). Both bootstrap methods also duplicate the stratified resampling loop. Extract shared helpers to eliminate cross-file sync risk. | `trop.py`, `trop_global.py`, `trop_local.py` | — | Low |
| StaggeredTripleDifference R cross-validation: CSV fixtures not committed (gitignored); tests skip without local R + triplediff. Commit fixtures or generate deterministically. | `tests/test_methodology_staggered_triple_diff.py` | #245 | Medium |
| StaggeredTripleDifference R parity: benchmark only tests no-covariate path (xformla=~1). Add covariate-adjusted scenarios and aggregation SE parity assertions. | `benchmarks/R/benchmark_staggered_triplediff.R` | #245 | Medium |
| StaggeredTripleDifference: per-cohort group-effect SEs include WIF (conservative vs R's wif=NULL). Documented in REGISTRY. Could override mixin for exact R match. | `staggered_triple_diff.py` | #245 | Low |
| HonestDiD Delta^RM: uses naive FLCI instead of paper's ARP conditional/hybrid confidence sets (Sections 3.2.1-3.2.2). ARP infrastructure exists but moment inequality transformation needs calibration. CIs are conservative (wider, valid coverage). | `honest_did.py` | #248 | Medium |
| Replicate weight tests use Fay-like BRR perturbations (0.5/1.5), not true half-sample BRR. Add true BRR regressions per estimator family. Existing `test_survey_phase6.py` covers true BRR at the helper level. | `tests/test_replicate_weight_expansion.py` | #253 | Low |
| WooldridgeDiD: QMLE sandwich uses `aweight` cluster-robust adjustment `(G/(G-1))*(n-1)/(n-k)` vs Stata's `G/(G-1)` only. Conservative (inflates SEs). Add `qmle` weight type if Stata golden values confirm material difference. | `wooldridge.py`, `linalg.py` | #216 | Medium |
| WooldridgeDiD: aggregation weights use cell-level n_{g,t} counts. Paper (W2025 Eqs. 7.2-7.4) defines cohort-share weights. Add optional `weights="cohort_share"` parameter to `aggregate()`. | `wooldridge_results.py` | #216 | Medium |
| WooldridgeDiD: canonical link requirement (W2023 Prop 3.1) not enforced — no warning if user applies wrong method to outcome type. Estimator is consistent regardless, but equivalence with imputation breaks. | `wooldridge.py` | #216 | Low |
| WooldridgeDiD: Stata `jwdid` golden value tests — add R/Stata reference script and `TestReferenceValues` class. | `tests/test_wooldridge.py` | #216 | Medium |

#### Performance

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| ImputationDiD event-study SEs recompute full conservative variance per horizon (should cache A0/A1 factorization) | `imputation.py` | #141 | Low |
| Rust faer SVD ndarray-to-faer conversion overhead (minimal vs SVD cost) | `rust/src/linalg.rs:67` | #115 | Low |
| Unrelated label events (e.g., adding `bug` label) re-trigger CI workflows when `ready-for-ci` is already present; filter `labeled`/`unlabeled` events to only `ready-for-ci` transitions | `.github/workflows/rust-test.yml`, `notebooks.yml` | #269 | Low |

#### Testing/Docs

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| R comparison tests spawn separate `Rscript` per test (slow CI) | `tests/test_methodology_twfe.py:294` | #139 | Low |
| CS R helpers hard-code `xformla = ~ 1`; no covariate-adjusted R benchmark for IRLS path | `tests/test_methodology_callaway.py` | #202 | Low |
| ~376 `duplicate object description` Sphinx warnings — restructure `docs/api/*.rst` to avoid duplicate `:members:` + `autosummary` | `docs/api/*.rst` | — | Low |
| Doc-snippet smoke tests only cover `.rst` files; `.txt` AI guides outside CI validation | `tests/test_doc_snippets.py` | #239 | Low |
| Add CI validation for `docs/doc-deps.yaml` integrity (stale paths, unmapped source files) | `docs/doc-deps.yaml` | #269 | Low |

---

### Standard Error Consistency

Different estimators compute SEs differently. Consider unified interface.

| Estimator | Default SE Type |
|-----------|-----------------|
| DifferenceInDifferences | HC1 or cluster-robust |
| TwoWayFixedEffects | Always cluster-robust (unit level) |
| CallawaySantAnna | Simple difference-in-means SE |
| SyntheticDiD | Bootstrap or placebo-based |

**Action**: Consider adding `se_type` parameter for consistency across estimators.

### Type Annotations

Mypy reports 0 errors. All mixin `attr-defined` errors resolved via
`TYPE_CHECKING`-guarded method stubs in bootstrap mixin classes.

## Deprecated Code

Deprecated parameters still present for backward compatibility:

- `lambda_reg` and `zeta` in `SyntheticDiD` (`synthetic_did.py`)
  - Deprecated in favor of `zeta_omega`/`zeta_lambda` parameters
  - Remove in v3.1

---

## Test Coverage

**Note**: 21 visualization tests are skipped when matplotlib unavailable—this is expected.

---

## Honest DiD Improvements

Enhancements for `honest_did.py`:

- [ ] Improved C-LF implementation with direct optimization instead of grid search
  (current implementation uses simplified FLCI approach with estimation uncertainty
  adjustment; see `honest_did.py:947`)
- [x] Support for CallawaySantAnnaResults (implemented in `honest_did.py:612-653`;
  requires `aggregate='event_study'` when calling `CallawaySantAnna.fit()`)
- [ ] Event-study-specific bounds for each post-period
- [ ] Hybrid inference methods
- [ ] Simulation-based power analysis for honest bounds

---

## CallawaySantAnna Bootstrap Improvements

- [ ] Consider aligning p-value computation with R `did` package (symmetric percentile method)

---

## RuntimeWarnings in Linear Algebra Operations

### Apple Silicon M4 BLAS Bug (numpy < 2.3)

Spurious RuntimeWarnings ("divide by zero", "overflow", "invalid value") are emitted by `np.matmul`/`@` on Apple Silicon M4 + macOS Sequoia with numpy < 2.3. The warnings appear for matrices with ≥260 rows but **do not affect result correctness** — coefficients and fitted values are valid (no NaN/Inf), and the design matrices are full rank.

**Root cause**: Apple's BLAS SME (Scalable Matrix Extension) kernels corrupt the floating-point status register, causing spurious FPE signals. Tracked in [numpy#28687](https://github.com/numpy/numpy/issues/28687) and [numpy#29820](https://github.com/numpy/numpy/issues/29820). Fixed in numpy ≥ 2.3 via [PR #29223](https://github.com/numpy/numpy/pull/29223).

**Not reproducible** on M3, Intel, or Linux.

- [ ] `linalg.py:162` - Warnings in fitted value computation (`X @ coefficients`)
  - Caused by M4 BLAS bug, not extreme coefficient values
  - Seen in test_prep.py during treatment effect recovery tests (n > 260)
- [ ] `triple_diff.py:307,323` - Warnings in propensity score computation
  - Occurs in IPW and DR estimation methods with covariates
  - Related to logistic regression overflow in edge cases (separate from BLAS bug)

- **Long-term:** Revert to `@` operator when numpy ≥ 2.3 becomes the minimum supported version.

---

## Feature Gaps (from R `did` package comparison)

Features in R's `did` package that block porting additional tests:

| Feature | R tests blocked | Priority | Status |
|---------|----------------|----------|--------|
| Calendar time aggregation | 1 test in test-att_gt.R | Low | |

---

## Performance Optimizations

Potential future optimizations:

- [ ] JIT compilation for bootstrap loops (numba)
- [ ] Sparse matrix handling for large fixed effects

### QR+SVD Redundancy in Rank Detection

**Background**: The current `solve_ols()` implementation performs both QR (for rank detection) and SVD (for solving) decompositions on rank-deficient matrices. This is technically redundant since SVD can determine rank directly.

**Current approach** (R-style, chosen for robustness):
1. QR with pivoting for rank detection (`_detect_rank_deficiency()`)
2. scipy's `lstsq` with 'gelsd' driver (SVD-based) for solving

**Why we use QR for rank detection**:
- QR with pivoting provides the canonical ordering of linearly dependent columns
- R's `lm()` uses this approach for consistent dropped-column reporting
- Ensures consistent column dropping across runs (SVD column selection can vary)

**Potential optimization** (future work):
- Skip QR when `rank_deficient_action="silent"` since we don't need column names
- Use SVD rank directly in the Rust backend (already implemented)
- Add `skip_rank_check` parameter for hot paths where matrix is known to be full-rank (implemented in v2.2.0)

**Priority**: Low - the QR overhead is minimal compared to SVD solve, and correctness is more important than micro-optimization.

### Incomplete `check_finite` Bypass

**Background**: The `solve_ols()` function accepts a `check_finite=False` parameter intended to skip NaN/Inf validation for performance in hot paths where data is known to be clean.

**Current limitation**: When `check_finite=False`, our explicit validation is skipped, but scipy's internal QR decomposition in `_detect_rank_deficiency()` still validates finite values. This means callers cannot fully bypass all finite checks.

**Impact**: Minimal - the scipy check is fast and only affects edge cases where users explicitly pass `check_finite=False` with non-finite data (which would be a bug in their code anyway).

**Potential fix** (future work):
- Pass `check_finite=False` through to scipy's QR call (requires scipy >= 1.9.0)
- Or skip `_detect_rank_deficiency()` entirely when `check_finite=False` and `_skip_rank_check=True`

**Priority**: Low - this is an edge case optimization that doesn't affect correctness.

