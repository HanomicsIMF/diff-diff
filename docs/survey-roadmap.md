# Survey Data Support Roadmap

This document captures the survey data support roadmap for diff-diff.
Phases 1-9 are complete. Phase 10 covers the credibility and announcement
readiness work still ahead.

---

## What's Shipped

### Phases 1-2: Core Infrastructure

- `SurveyDesign` class with weights, strata, PSU, FPC, weight_type, nest, lonely_psu
- Taylor Series Linearization (TSL) variance with strata + PSU + FPC
- Weighted OLS, sandwich estimator, demeaning, survey degrees of freedom
- `SurveyMetadata` on results (effective n, DEFF, weight_range)
- Base estimators: DifferenceInDifferences, TwoWayFixedEffects, MultiPeriodDiD

### Phase 3: OLS-Based Standalone Estimators

| Estimator | Survey Support | Notes |
|-----------|----------------|-------|
| StackedDiD | pweight only | Q-weights compose multiplicatively; fweight/aweight rejected |
| SunAbraham | Full | Bootstrap via Rao-Wu rescaled |
| BaconDecomposition | Diagnostic | Weighted descriptives only, no inference |
| TripleDifference | Full | Regression, IPW, and DR methods with TSL on IFs |
| ContinuousDiD | Full | Weighted B-spline OLS + TSL; bootstrap via multiplier at PSU |
| EfficientDiD | Full | No-cov and DR covariate paths both survey-weighted; bootstrap via multiplier at PSU |

### Phase 4: Complex Estimators + Weighted Logit

| Estimator | Survey Support | Notes |
|-----------|----------------|-------|
| ImputationDiD | Full | Weighted iterative FE + conservative variance; bootstrap via multiplier at PSU |
| TwoStageDiD | Full | Weighted FE + GMM sandwich; bootstrap via multiplier at PSU |
| CallawaySantAnna | Full | Strata/PSU/FPC/replicate weights; IPW/DR covariates (Phase 7a); replicate IF variance |

Weighted `solve_logit()` in `linalg.py` — survey weights enter IRLS as
`w_survey * mu * (1 - mu)`.

### Phase 5: SyntheticDiD + TROP

| Estimator | Survey Support | Notes |
|-----------|----------------|-------|
| SyntheticDiD | pweight | Treated means survey-weighted; omega composed with control weights post-optimization |
| TROP | pweight | Population-weighted ATT aggregation; model fitting unchanged |

### Phase 6: Advanced Features (v2.7.6)

- **Survey-aware bootstrap** for all 8 bootstrap-using estimators:
  multiplier at PSU (CS, Imputation, TwoStage, Continuous, Efficient)
  and Rao-Wu rescaled (SA, SyntheticDiD, TROP)
- **Replicate weight variance**: BRR, Fay's BRR, JK1, JKn, SDR.
  12 of 16 estimators supported (not SyntheticDiD, TROP, BaconDecomposition, or WooldridgeDiD)
- **DEFF diagnostics**: per-coefficient design effects vs SRS baseline
- **Subpopulation analysis**: `SurveyDesign.subpopulation()` preserves
  full design structure for correct variance

### Phase 7: Completing the Survey Story (v2.8.0-v2.8.1)

- **7a.** CS IPW/DR covariates + survey: DRDID nuisance IF corrections
  (Sant'Anna & Zhao 2020, Theorem 3.1)
- **7b.** Repeated cross-sections: `CallawaySantAnna(panel=False)` matching
  `DRDID::reg_did_rc`, `drdid_rc`, `std_ipw_did_rc`
- **7c.** Survey tutorial: `docs/tutorials/16_survey_did.ipynb` with full
  workflow (strata, PSU, FPC, replicates, subpopulation, DEFF)
- **7d.** HonestDiD + survey: survey df and event-study VCV propagated
  to sensitivity analysis with t-distribution critical values
- **7e.** StaggeredTripleDifference survey support (only implementation
  in R or Python with design-based DDD variance)

### Phase 8: Survey Maturity (v2.8.3-v2.8.4)

- **8a.** SDR replicate method for ACS PUMS (80 columns)
- **8b.** FPC in ImputationDiD and TwoStageDiD
- **8c.** Silent operation warnings (8 operations now emit `UserWarning`)
- **8d.** Lonely PSU "adjust" in bootstrap (Rust & Rao 1996)
- **8e.** CV on estimates, `trim_weights()`, survey-aware ImputationDiD pretrends
- **8f.** Compatibility matrix in `choosing_estimator.rst`

### Phase 9: Real-Data Validation (v2.9.0)

15 cross-validation tests against R's `survey` package using real federal
survey datasets:

| Dataset | Design | Key result |
|---------|--------|------------|
| API (R `survey`) | Strata + FPC | ATT, SE, df, CI match R (7 variants incl. subpopulation, Fay's BRR) |
| NHANES (CDC/NCHS) | Strata + PSU (nest=TRUE) | ACA DiD matches R for strata+PSU, covariates, subpopulation |
| RECS 2020 (U.S. EIA) | 60 JK1 replicate weights | Coefficients, SEs, df, CI match R |

Files: `benchmarks/R/benchmark_realdata_*.R`, `tests/test_survey_real_data.py`,
`benchmarks/data/real/*_realdata_golden.json`

### Documentation Remaining (Phase 8g)

- **Multi-stage design**: not yet documented. Single-stage (strata + PSU)
  is sufficient per Lumley (2004) Section 2.2.
- **Post-stratification / calibration**: not yet documented. `SurveyDesign`
  expects pre-calibrated weights. `samplics` is the most complete Python
  option (post-stratification, raking, GREG) but is in read-only mode —
  active development has moved to `svy`, which is not yet publicly
  released. `weightipy` is actively maintained for raking. Weight
  calibration is out of scope for diff-diff today, though building this
  capability is a future possibility.

### Phase 10: Survey Completeness (v2.9.0–v3.0)

- **10a.** Survey theory document (`survey-theory.md`) — formal justification for design-based variance with modern DiD influence functions
- **10b.** Research-grade survey DGP — 8 new parameters on `generate_survey_did_data()`
- **10c.** R validation expansion — 8 of 16 estimators cross-validated against R's `survey::svyglm()`
- **10d.** Tutorial rewrite — flat-weight vs design-based comparison with known ground truth
- **10f.** WooldridgeDiD survey support — OLS, logit, Poisson paths with `pweight` + strata/PSU/FPC + TSL variance

---

## Phase 10: Remaining Items

The items below establish further credibility with
practitioners and methodologists.

### 10a. Theory Document (HIGH priority) ✅

`docs/methodology/survey-theory.md` lays out the formal argument for
design-based variance estimation with modern DiD influence functions:

1. Modern heterogeneity-robust DiD estimators (CS, SA, BJS) are smooth
   functionals of the weighted empirical distribution
2. Survey-weighted empirical distribution is design-consistent for the
   finite-population quantity (Hájek/design-weighted estimator)
3. The influence function is a property of the functional, not the
   sampling design — IFs remain valid under survey weighting
4. TSL (stratified cluster sandwich) and replicate-weight methods are
   valid variance estimators for smooth functionals of survey-weighted
   estimating equations (Binder 1983, Rao & Wu 1988, Shao 1996)

This is the short-term deliverable that can be linked from docs and README
immediately.

**Key references:**
- Binder, D.A. (1983). "On the Variances of Asymptotically Normal
  Estimators from Complex Surveys." *International Statistical Review* 51.
- Rao, J.N.K. & Wu, C.F.J. (1988). "Resampling Inference with Complex
  Survey Data." *JASA* 83(401).
- Shao, J. (1996). "Resampling Methods in Sample Surveys." *Statistics* 27.

### 10b. Survey Simulation DGP (HIGH priority) ✅

Enhanced `generate_survey_did_data()` with 8 research-grade parameters:
`icc`, `weight_cv`, `informative_sampling`, `heterogeneous_te_by_strata`,
`te_covariate_interaction`, `covariate_effects`, `strata_sizes`, and
`return_true_population_att`. All backward-compatible. Supports panel
and repeated cross-section modes.

**Remaining gap for 10e:** Conditional parallel trends — the DGP has
unconditional PT by construction. A `conditional_pt` parameter is needed
before the simulation study so that unconditional PT fails but conditional
PT holds after covariate adjustment (DR/IPW recovers truth).

### 10c. Expand R Validation Coverage (HIGH priority) ✅

8 of 16 estimators now cross-validated against R's `survey::svyglm()`:
DifferenceInDifferences, TWFE, CallawaySantAnna, SyntheticDiD,
ImputationDiD, StackedDiD, SunAbraham, TripleDifference.

### 10d. Tutorial: Show the Pain (HIGH priority) ✅

Survey tutorial rewritten with side-by-side flat-weight vs design-based
comparison using the research-grade DGP from 10b, showing known ground
truth, coverage simulation, and false pre-trend detection rates.

### 10e. Position Paper / arXiv Preprint (MEDIUM priority, long-term)

A 15-25 page methodology note targeting JSSAM, simultaneously posted to
arXiv. Theory (~5pp), simulation study using DGP from 10b (~8pp),
empirical illustration with NHANES ACA data (~3pp), software section
(~2pp).

**Simulation study scenarios** (minimum):
1. Unconditional PT with complex survey — coverage of TSL vs flat-weight SEs
2. Informative sampling + heterogeneous TE — weighted ATT bias correction
3. Panel vs repeated cross-section — both design types
4. **Conditional PT** — unconditional PT fails (differential pre-trends
   correlated with X), conditional PT holds after covariate adjustment.
   DR/IPW with covariates recovers truth; no-covariate estimator is biased.
   This is the most novel claim — survey-weighted nuisance estimation
   (propensity scores, outcome regression) produces valid IFs under complex
   sampling. **Requires DGP extension**: add a `conditional_pt` parameter
   to `generate_survey_did_data()` that makes the time trend
   X-dependent (e.g., `trend_i = 0.5*t + delta * x1_i * t`).

**Co-authorship:** A co-author from the DiD methodology community would
strengthen credibility — someone who can vouch that the IFs are valid
under survey weighting. The survey statistics side (Binder 1983, Rao &
Wu 1988) is established and doesn't need a survey methodologist to
co-sign.

### 10f. WooldridgeDiD Survey Support — SHIPPED

WooldridgeDiD (ETWFE) now supports `survey_design` for all three methods
(OLS, logit, Poisson) with `pweight` only (`fweight`/`aweight` rejected).
OLS uses survey-weighted within-transformation + WLS + TSL vcov.
Logit/Poisson use survey-weighted IRLS + X_tilde linearization for TSL
vcov. Replicate-weight designs raise `NotImplementedError`; bootstrap +
survey is rejected.

### 10g. Practitioner Guidance (LOW priority)

A decision flowchart helping practitioners decide whether they need full
survey design or whether flat weights suffice. Key factors: ICC, number
of PSUs, stratification gain, DEFF magnitude. DEFF diagnostics provide
the empirical answer, but practitioners need guidance on interpretation.

---

## Current Limitations

All items below raise an error when attempted, with a message describing
the limitation and suggested alternative.

| Estimator | Limitation | Alternative |
|-----------|-----------|-------------|
| SyntheticDiD | Replicate weights | Use strata/PSU/FPC design with Rao-Wu rescaled bootstrap |
| TROP | Replicate weights | Use strata/PSU/FPC design with Rao-Wu rescaled bootstrap |
| BaconDecomposition | Replicate weights | Diagnostic only, no inference |
| SyntheticDiD | `variance_method='placebo'` + strata/PSU/FPC | Use `variance_method='bootstrap'` |
| ImputationDiD | `pretrends=True` + replicate weights | Use analytical survey design instead |
| ImputationDiD | `pretrend_test()` + replicate weights | Use analytical survey design instead |
| DiD, TWFE | `inference='wild_bootstrap'` + `survey_design` | Use analytical survey inference (default) |
| EfficientDiD | `cluster` + `survey_design` | Use `survey_design` with PSU/strata |
| All bootstrap estimators | Bootstrap + replicate weights | These are alternative variance methods; pick one |

**Warning/fallback (no error):** MultiPeriodDiD with `wild_bootstrap` +
`survey_design` warns and falls back to analytical inference.

**Conservative approach (no error):** CallawaySantAnna `reg`+covariates
uses conservative plug-in IF rather than efficient DRDID nuisance IF
correction (see REGISTRY.md).
