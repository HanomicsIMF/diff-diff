# Survey Data Support Roadmap

This document captures planned future work for survey data support in diff-diff.
Phases 1-4 are implemented. Phase 5 is deferred for future PRs.

## Implemented (Phases 1-2)

- **SurveyDesign** class with weights, strata, PSU, FPC, weight_type, nest, lonely_psu
- **Weighted OLS** via sqrt(w) transformation in `solve_ols()`
- **Weighted sandwich estimator** in `compute_robust_vcov()` (pweight/fweight/aweight)
- **Taylor Series Linearization** (TSL) variance with strata + PSU + FPC
- **Survey degrees of freedom**: n_PSU - n_strata
- **Base estimator integration**: DifferenceInDifferences, TwoWayFixedEffects, MultiPeriodDiD
- **Weighted demeaning**: `demean_by_group()` and `within_transform()` with weights
- **SurveyMetadata** in results objects with effective n, DEFF, weight_range

## Implemented (Phase 3): OLS-Based Standalone Estimators

| Estimator | File | Survey Support | Notes |
|-----------|------|----------------|-------|
| StackedDiD | `stacked_did.py` | pweight only | Q-weights compose multiplicatively with survey weights; TSL vcov on composed weights; fweight/aweight rejected (composition changes weight semantics) |
| SunAbraham | `sun_abraham.py` | Full | Survey weights in LinearRegression + weighted within-transform; bootstrap+survey deferred |
| BaconDecomposition | `bacon.py` | Diagnostic | Weighted cell means, weighted within-transform, weighted group shares; no inference (diagnostic only) |
| TripleDifference | `triple_diff.py` | Full | Regression, IPW, and DR methods with weighted OLS/logit + TSL on influence functions |
| ContinuousDiD | `continuous_did.py` | Analytical | Weighted B-spline OLS + TSL on influence functions; bootstrap+survey deferred |
| EfficientDiD | `efficient_did.py` | Analytical | Weighted means/covariances in Omega* + TSL on EIF scores; bootstrap+survey deferred |

### Phase 3 Deferred Work

The following capabilities were deferred from Phase 3 because they depend on
Phase 5 infrastructure (bootstrap+survey interaction):

| Estimator | Deferred Capability | Blocker |
|-----------|-------------------|---------|
| SunAbraham | Pairs bootstrap + survey | Phase 5: bootstrap+survey interaction |
| ContinuousDiD | Multiplier bootstrap + survey | Phase 5: bootstrap+survey interaction |
| EfficientDiD | Multiplier bootstrap + survey | Phase 5: bootstrap+survey interaction |
| EfficientDiD | Covariates (DR path) + survey | DR nuisance estimation needs survey weight threading |

All blocked combinations raise `NotImplementedError` when attempted, with a
message pointing to the planned phase or describing the limitation.

## Implemented (Phase 4): Complex Standalone Estimators + Weighted Logit

| Estimator | File | Survey Support | Notes |
|-----------|------|----------------|-------|
| ImputationDiD | `imputation.py` | Analytical | Weighted iterative FE, weighted ATT aggregation, weighted conservative variance (Theorem 3); bootstrap+survey deferred |
| TwoStageDiD | `two_stage.py` | Analytical | Weighted iterative FE, weighted Stage 2 OLS, weighted GMM sandwich variance; bootstrap+survey deferred |
| CallawaySantAnna | `staggered.py` | Weights-only | Weights-only SurveyDesign (strata/PSU/FPC rejected); reg supports covariates, IPW/DR no-covariate only; survey-weighted WIF in aggregation; full design SEs, covariates+IPW/DR, and bootstrap+survey deferred |

**Infrastructure**: Weighted `solve_logit()` added to `linalg.py` — survey weights
enter the IRLS working weights as `w_survey * mu * (1 - mu)`. This also unblocked
TripleDifference IPW/DR from Phase 3 deferred work.

### Phase 4 Deferred Work

| Estimator | Deferred Capability | Blocker |
|-----------|-------------------|---------|
| ImputationDiD | Bootstrap + survey | Phase 5: bootstrap+survey interaction |
| TwoStageDiD | Bootstrap + survey | Phase 5: bootstrap+survey interaction |
| CallawaySantAnna | Bootstrap + survey | Phase 5: bootstrap+survey interaction |
| CallawaySantAnna | Strata/PSU/FPC in SurveyDesign | Phase 5: route combined IF/WIF through `compute_survey_vcov()` for design-based aggregation SEs |
| CallawaySantAnna | Covariates + IPW/DR + survey | Phase 5: DRDID panel nuisance IF corrections |
| CallawaySantAnna | Efficient DRDID nuisance IF for reg+covariates | Phase 5: replace conservative plug-in IF with semiparametrically efficient IF |

## Implemented (Phase 5): SyntheticDiD + TROP Survey Support

| Estimator | File | Survey Support | Notes |
|-----------|------|----------------|-------|
| SyntheticDiD | `synthetic_did.py` | pweight | Both sides weighted (WLS interpretation): treated means survey-weighted, ω composed with control survey weights post-optimization. Placebo and bootstrap SE preserve survey weights. Covariates residualized with WLS. |
| TROP | `trop.py` | pweight | ATT aggregation only: population-weighted average of per-observation treatment effects. Model fitting (kernel weights, LOOCV, nuclear norm) unchanged. Rust and Python bootstrap paths both support weighted ATT. |

### Phase 5 Deferred Work

| Estimator | Deferred Capability | Blocker |
|-----------|-------------------|---------|
| SyntheticDiD | strata/PSU/FPC + survey-aware bootstrap | Bootstrap + Survey Interaction |
| TROP | strata/PSU/FPC + survey-aware bootstrap | Bootstrap + Survey Interaction |

## Phase 6: Advanced Features

### Bootstrap + Survey Interaction
Unblock bootstrap + survey for all estimators that currently defer it
(ImputationDiD, TwoStageDiD, CallawaySantAnna, SunAbraham, ContinuousDiD,
EfficientDiD). Requires survey-aware resampling schemes.

### Replicate Weight Variance
Re-run WLS for each replicate weight column, compute variance from distribution
of estimates. Supports BRR, Fay's BRR, JK1, JKn, bootstrap replicate methods.
Add `replicate_weights`, `replicate_type`, `replicate_rho` fields to SurveyDesign.

### DEFF Diagnostics
Compare survey vcov to SRS vcov element-wise. Report design effect per
coefficient. Effective n = n / DEFF.

### Subpopulation Analysis
`SurveyDesign.subpopulation(data, mask)` — zero-out weights for excluded
observations while preserving the full design structure for correct variance
estimation (unlike simple subsetting, which would drop design information).
