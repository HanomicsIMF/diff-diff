# Survey Data Support Roadmap

This document captures the survey data support roadmap for diff-diff.
Phases 1-7 are implemented. Phase 8 (maturity refinements) is planned.

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
| CallawaySantAnna | `staggered.py` | Full | Full SurveyDesign (strata/PSU/FPC/replicate weights); reg supports covariates, IPW/DR no-covariate only; survey-weighted WIF in aggregation; replicate IF variance for analytical SEs |

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

| Estimator | Deferred Capability | Status |
|-----------|-------------------|--------|
| SyntheticDiD | strata/PSU/FPC + survey-aware bootstrap | Resolved in Phase 6 |
| TROP | strata/PSU/FPC + survey-aware bootstrap | Resolved in Phase 6 |

## Phase 6: Advanced Features

### Bootstrap + Survey Interaction ✅ (2026-03-24)
Survey-aware bootstrap for all 8 bootstrap-using estimators. Two strategies:
- **Multiplier at PSU level** (CS, ImputationDiD, TwoStageDiD, ContinuousDiD,
  EfficientDiD): generate multiplier weights at PSU level within strata, FPC-scaled.
- **Rao-Wu rescaled** (SunAbraham, SyntheticDiD, TROP): draw PSUs within strata,
  rescale observation weights per Rao, Wu & Yue (1992).
- **CS analytical expansion**: strata/PSU/FPC now supported for aggregated SEs via
  `compute_survey_if_variance()`.
- **TROP**: cross-classified pseudo-strata (survey_stratum × treatment_group).
- **Rust TROP**: pweight-only in Rust; full design falls to Python path.

### Replicate Weight Variance ✅ (2026-03-26)
Re-run WLS for each replicate weight column, compute variance from distribution
of estimates. Supports BRR, Fay's BRR, JK1, JKn methods.
JKn requires explicit `replicate_strata` (per-replicate stratum assignment).
- `replicate_weights`, `replicate_method`, `fay_rho` fields on SurveyDesign
- `compute_replicate_vcov()` for OLS-based estimators (re-runs WLS per replicate)
- `compute_replicate_if_variance()` for IF-based estimators (reweights IF)
- Dispatch in `LinearRegression.fit()` and `staggered_aggregation.py`
- Replicate weights mutually exclusive with strata/PSU/FPC
- Survey df = rank(replicate_weights) - 1, matching R's `survey::degf()`
- **Limitations**: Supported in CallawaySantAnna, ContinuousDiD, EfficientDiD,
  TripleDifference (analytical only, no bootstrap). Rejected with
  `NotImplementedError` in DifferenceInDifferences, TwoWayFixedEffects,
  MultiPeriodDiD, StackedDiD, SunAbraham, ImputationDiD, TwoStageDiD,
  SyntheticDiD, TROP. Expansion to regression-based estimators (SA,
  Imputation, TwoStage, Stacked) is straightforward but deferred.

### DEFF Diagnostics ✅ (2026-03-26)
Per-coefficient design effects comparing survey vcov to SRS (HC1) vcov.
- `DEFFDiagnostics` dataclass with per-coefficient DEFF, effective n, SRS/survey SE
- `compute_deff_diagnostics()` standalone function
- `LinearRegression.compute_deff()` post-fit method
- Display label "Kish DEFF (weights)" for existing weight-based DEFF

### Subpopulation Analysis ✅ (2026-03-26)
`SurveyDesign.subpopulation(data, mask)` — zero-out weights for excluded
observations while preserving the full design structure for correct variance
estimation (unlike simple subsetting, which would drop design information).
- Mask: bool array/Series, column name, or callable
- Returns (SurveyDesign, DataFrame) pair with synthetic `_subpop_weight` column
- Weight validation relaxed: zero weights allowed (negative still rejected)

---

## Phase 7: Completing the Survey Story

These items close the remaining gaps that matter for practitioners using major
population surveys (ACS, CPS, BRFSS, MEPS) with modern DiD methods. Together
they make diff-diff the only package — R or Python — with full design-based
variance estimation for heterogeneity-robust DiD estimators.

### 7a. CallawaySantAnna Covariates + IPW/DR + Survey ✅

**Implemented.** DRDID panel nuisance-estimation IF corrections (PS + OR)
for both survey and non-survey IPW/DR paths. Survey-weighted propensity
score estimation via `solve_logit()`, survey-weighted outcome regression,
and IFs that account for nuisance parameter estimation uncertainty under
the survey design (Sant'Anna & Zhao 2020, Theorem 3.1).

**Reference:** Sant'Anna, P.H.C. & Zhao, J. (2020). "Doubly Robust
Difference-in-Differences Estimators." *Journal of Econometrics* 219(1).

### 7b. Repeated Cross-Sections ✅

**Priority: High.** Many major surveys (BRFSS, ACS annual cross-sections,
CPS monthly) are not panels — units are not followed over time. The R `did`
package supports `panel=FALSE` for these settings. diff-diff currently
requires panel data for all staggered estimators.

**Implemented.** `CallawaySantAnna(panel=False)` for repeated cross-section
surveys. Uses cross-sectional DRDID (Sant'Anna & Zhao 2020, Section 4):
`reg` matches `DRDID::reg_did_rc`, `dr` matches `DRDID::drdid_rc` (locally
efficient with 4 OLS fits), `ipw` matches `DRDID::std_ipw_did_rc`. Survey
weights, covariates, and all estimation methods supported.

**Reference:** Sant'Anna, P.H.C. & Zhao, J. (2020). Sections 3 (panel) vs
4 (repeated cross-sections). Callaway, B. & Sant'Anna, P.H.C. (2021).
Section 4.1.

### 7c. Survey-Aware DiD Tutorial ✅

**Implemented.** `docs/tutorials/16_survey_did.ipynb` — 35-cell tutorial
framed around a state-level preventive care program evaluated with a
stratified health survey (ACS/BRFSS-like). Covers: why survey design
matters, SurveyDesign setup, basic DiD with survey, staggered DiD
(CallawaySantAnna) with survey, replicate weights (JK1), subpopulation
analysis, DEFF diagnostics, repeated cross-sections, and estimator
support reference table. Uses `generate_survey_did_data()` DGP function
added to `diff_diff.prep`.

### 7d. HonestDiD with Survey Variance ✅

**Implemented.** Survey df and full event-study VCV from IF vectors
propagated to HonestDiD sensitivity analysis. When CallawaySantAnna is fit
with a survey design, HonestDiD uses t-distribution critical values with
survey degrees of freedom. Bootstrap/replicate designs fall back to
diagonal VCV with a warning.

**Reference:** Rambachan, A. & Roth, J. (2023). "A More Credible Approach
to Parallel Trends." *Review of Economic Studies* 90(5).

### 7e. StaggeredTripleDifference Survey Support ✅

**Implemented.** Full survey design support (pweight only) for the
Ortiz-Villavicencio & Sant'Anna (2025) staggered DDD estimator. Survey
weights thread through all three pairwise DiD comparisons: propensity
score estimation (weighted IRLS via `solve_logit`), outcome regression
(WLS via `solve_ols`), and Riesz representer computation (weighted
Hajek normalization). IF combination weights (w1/w2/w3) use
survey-weighted cell sizes. Per-cell SEs use the standard IF formula
(matching CallawaySantAnna); aggregated SEs (overall, event study,
group) use design-based variance via `compute_survey_if_variance()`
(TSL with strata/PSU/FPC) or `compute_replicate_if_variance()`
(replicate weights). Bootstrap uses PSU-level multiplier weights
when survey design is present.

**Note:** The R `triplediff` package does not support survey weights.
diff-diff is the only implementation (R or Python) with design-based
variance estimation for staggered triple differences.

**Reference:** Ortiz-Villavicencio, M. & Sant'Anna, P.H.C. (2025).
"Better Understanding Triple Differences Estimators." arXiv:2505.09942.

---

## Phase 8: Survey Maturity

Refinements to close remaining gaps versus R's `survey` package and improve
practitioner experience. Prioritized by user impact.

### 8a. Successive Difference Replication (SDR)

**Priority: High.** ACS PUMS — the most common US survey dataset for DiD
policy evaluation — provides 80 SDR replicate weight columns. Without SDR
support, these users can't use their provided replicate weights directly.

**What's needed:**
- Add `"SDR"` to `valid_rep_methods` in `SurveyDesign`
- Variance formula: `V = 4/R * sum((theta_r - theta)^2)` — a scaling
  difference from BRR, not a new algorithm
- Wire through `compute_replicate_vcov()` and `compute_replicate_if_variance()`

**Reference:** Fay, R.E. & Train, G.F. (1995). "Aspects of Survey and
Model-Based Postcensal Estimation of Income and Poverty Characteristics
for States and Counties." ASA Proceedings.

### 8b. FPC in ImputationDiD and TwoStageDiD

**Priority: High.** Both estimators now support replicate weights and TSL
with strata/PSU, but reject FPC outright (`NotImplementedError`). Adding
FPC is incremental — thread `fpc` through the existing TSL variance path.
Matters for finite population surveys (common in state-level sampling).

**Current gate:** `imputation.py:280`, `two_stage.py:268`

### 8c. Silent Operation Warnings

**Priority: High.** Add `UserWarning` emissions for operations that
silently alter analysis results:
- TROP lstsq → pseudo-inverse numerical fallback
- TwoStageDiD NaN masking of unidentified fixed effects
- TwoStageDiD always-treated unit removal
- CallawaySantAnna silent (g,t) pair skipping
- TROP missing treatment indicator fill with 0
- Rust → Python backend fallback (currently debug log only)
- Survey weight normalization (pweights rescaled to mean=1)
- `np.inf` → 0 never-treated conversion

### 8d. Lonely PSU "adjust" in Bootstrap

**Priority: Medium.** `lonely_psu="adjust"` works for analytical (TSL)
variance but raises `NotImplementedError` for survey-aware bootstrap
(2 raises in `bootstrap_utils.py`). Real survey data regularly has
singleton strata. Users needing bootstrap inference with such data hit
a wall.

**Reference:** Rust, K.F. & Rao, J.N.K. (1996). "Variance Estimation
for Complex Surveys Using Replication Techniques." Statistical Methods
in Medical Research 5(3).

### 8e. Survey Diagnostics and Utilities

**Priority: Medium.** Small additions that signal maturity to survey
statisticians:
- **CV on estimates**: coefficient of variation (SE/estimate) on results
  objects — trivial to add, used by federal agencies for publication
  standards (NCHS requires CV < 30% for releasable estimates)
- **Weight trimming**: `trim_weights(data, weight_col, upper=None,
  quantile=None)` utility in `prep.py` for capping extreme weights
- **ImputationDiD pretrends + survey**: pre-trends F-test currently
  ignores survey variance (`NotImplementedError` at `imputation.py:240`)

### 8f. Survey Compatibility Matrix

**Priority: Medium.** Users discover survey support limits by hitting
`NotImplementedError` at runtime. Add a table to the survey tutorial
or `choosing_estimator.rst` showing which estimator × survey feature
combinations are supported (weights, strata/PSU, FPC, replicate weights,
bootstrap + survey).

### 8g. Documentation-Only Items

**Priority: Low.** No code changes required:
- **Multi-stage design**: document that single-stage (strata + PSU)
  is sufficient for variance estimation per Lumley (2004) Section 2.2.
  Don't implement multi-stage — it adds complexity without changing
  results for DiD applications.
- **Post-stratification / calibration**: document that `SurveyDesign`
  expects pre-calibrated weights. Point users to `samplics` or R's
  `survey::calibrate()` for weight calibration. This is data prep,
  not DiD estimation — out of scope.

### Deferred

| Estimator | Capability | Reason |
|-----------|-----------|--------|
| SyntheticDiD | Replicate weights | No published theory on replicate weights + unit weight optimization |
| TROP | Replicate weights | No published theory on replicate weights + nuclear norm regularization |
| BaconDecomposition | Replicate weights | Diagnostic tool with no inference — replicate weights don't apply |
| EfficientDiD | Covariates + survey, cluster + survey, bootstrap + survey | Lower demand, newer estimator; 3 `NotImplementedError` paths |
