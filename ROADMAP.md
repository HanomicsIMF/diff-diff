# diff-diff Roadmap

This document outlines the feature roadmap for diff-diff, prioritized by practitioner value and academic credibility.

For past changes and release history, see [CHANGELOG.md](CHANGELOG.md).

---

## Current Status (v3.0)

diff-diff is a **production-ready** DiD library with feature parity with R's `did` + `HonestDiD` + `synthdid` ecosystem for core DiD analysis, plus **unique survey support** that no R or Python package matches.

### Estimators

- **Core**: Basic DiD, TWFE, MultiPeriod event study
- **Heterogeneity-robust**: Callaway-Sant'Anna (2021), Sun-Abraham (2021), Borusyak-Jaravel-Spiess Imputation (2024), Two-Stage DiD (Gardner 2022), Stacked DiD (Wing et al. 2024)
- **Specialized**: Synthetic DiD (Arkhangelsky et al. 2021), Triple Difference, Staggered Triple Difference (Ortiz-Villavicencio & Sant'Anna 2025), Continuous DiD (Callaway, Goodman-Bacon & Sant'Anna 2024), TROP
- **Efficient**: EfficientDiD (Chen, Sant'Anna & Xie 2025) — semiparametrically efficient with doubly robust covariates
- **Nonlinear**: WooldridgeDiD / ETWFE (Wooldridge 2023, 2025) — saturated OLS (direct cohort x time coefficients), logit, and Poisson QMLE (ASF-based ATT with delta-method SEs)

### Inference & Diagnostics

- Robust SEs, cluster SEs, wild bootstrap, multiplier bootstrap, placebo-based variance
- Parallel trends tests, placebo tests, Goodman-Bacon decomposition
- Honest DiD sensitivity analysis (Rambachan & Roth 2023), pre-trends power analysis (Roth 2022)
- Power analysis and simulation-based MDE tools
- EPV diagnostics for propensity score estimation

### Survey Support

`SurveyDesign` with strata, PSU, FPC, weight types (pweight/fweight/aweight), lonely PSU handling. All 16 estimators accept `survey_design` (15 inference-level + BaconDecomposition diagnostic); design-based variance estimation varies by estimator:

- **TSL variance** (Taylor Series Linearization) with strata + PSU + FPC
- **Replicate weights**: BRR, Fay's BRR, JK1, JKn, SDR — 12 of 16 estimators (not SyntheticDiD, TROP, BaconDecomposition, WooldridgeDiD)
- **Survey-aware bootstrap**: multiplier at PSU (IF-based) and Rao-Wu rescaled (resampling-based)
- **DEFF diagnostics**, **subpopulation analysis**, **weight trimming**, **CV on estimates**
- **Repeated cross-sections**: `CallawaySantAnna(panel=False)` for BRFSS, ACS, CPS
- **R cross-validation**: 15 tests against R's `survey` package using NHANES, RECS, and API datasets

See [Survey Design Support](docs/choosing_estimator.rst#survey-design-support) for the full compatibility matrix, and [survey-roadmap.md](docs/survey-roadmap.md) for implementation details.

### Infrastructure

- Optional Rust backend for accelerated computation
- Label-gated CI (tests run only when `ready-for-ci` label is added)
- Documentation dependency map (`docs/doc-deps.yaml`) with `/docs-impact` skill
- AI practitioner guardrails based on Baker et al. (2025) 8-step workflow

---

## Survey Academic Credibility (Phase 10)

Phase 10 established the theoretical and empirical foundation for survey support
credibility. See [survey-roadmap.md](docs/survey-roadmap.md) for detailed specs.

| Item | Priority | Status |
|------|----------|--------|
| **10a.** Theory document (`survey-theory.md`) | HIGH | ✅ Shipped (v2.9.1) |
| **10b.** Research-grade survey DGP (enhance `generate_survey_did_data`) | HIGH | ✅ Shipped (v2.9.1) |
| **10c.** Expand R validation (ImputationDiD, StackedDiD, SunAbraham, TripleDifference) | HIGH | ✅ Shipped (v2.9.1) |
| **10d.** Tutorial: flat-weight vs design-based comparison | HIGH | ✅ Shipped (v2.9.1) |
| **10e.** Position paper / arXiv preprint | MEDIUM | Not started — depends on 10b |
| **10f.** WooldridgeDiD survey support (OLS + logit + Poisson) | MEDIUM | ✅ Shipped (v2.9.0) |
| **10g.** Practitioner guidance: when does survey design matter? | LOW | Not started |

---

## Future: Survey Aggregation Helper

**`survey_aggregate()` helper function** for the microdata-to-panel workflow. Bridges individual-level survey data (BRFSS, ACS, CPS) collected as repeated cross-sections to geographic-level (state, city) panel DiD. Computes design-based cell means and precision weights that estimators can consume directly.

---

## Future Estimators

### de Chaisemartin-D'Haultfouille Estimator

Handles treatment that switches on and off (reversible treatments), unlike most other methods.

- Allows units to move into and out of treatment
- Time-varying, heterogeneous treatment effects
- Comparison with never-switchers or flexible control groups

**Reference**: [de Chaisemartin & D'Haultfouille (2020, 2024)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3980758). *American Economic Review*.

### Local Projections DiD

Implements local projections for dynamic treatment effects. Doesn't require specifying full dynamic structure.

- Flexible impulse response estimation
- Robust to misspecification of dynamics
- Natural handling of anticipation effects

**Reference**: Dube, Girardi, Jorda, and Taylor (2023).

### Causal Duration Analysis with DiD

Extends DiD to duration/survival outcomes where standard methods fail (hazard rates, time-to-event).

- Duration analogue of parallel trends on hazard rates
- Avoids distributional assumptions and hazard function specification

**Reference**: [Deaner & Ku (2025)](https://www.aeaweb.org/conference/2025/program/paper/k77Kh8iS). *AEA Conference Paper*.

---

## Long-Term Research Directions

Frontier methods requiring more research investment.

### DiD with Interference / Spillovers

Standard DiD assumes SUTVA; spatial/network spillovers violate this. Two-stage imputation approach estimates treatment AND spillover effects under staggered timing.

**Reference**: [Butts (2024)](https://arxiv.org/abs/2105.03737). *Working Paper*.

### Quantile/Distributional DiD

Recover the full counterfactual distribution and quantile treatment effects (QTT), not just mean ATT.

- Changes-in-Changes (CiC) identification strategy
- QTT(tau) at user-specified quantiles
- Full counterfactual distribution function

**Reference**: [Athey & Imbens (2006)](https://onlinelibrary.wiley.com/doi/10.1111/j.1468-0262.2006.00668.x). *Econometrica*.

### CATT Meta-Learner for Heterogeneous Effects

ML-powered conditional ATT — discover who benefits most from treatment using doubly robust meta-learner.

**Reference**: [Lan, Chang, Dillon & Syrgkanis (2025)](https://arxiv.org/abs/2502.04699). *Working Paper*.

### Causal Forests for DiD

Machine learning methods for discovering heterogeneous treatment effects in DiD settings.

**References**:
- [Kattenberg, Scheer & Thiel (2023)](https://ideas.repec.org/p/cpb/discus/452.html). *CPB Discussion Paper*.
- Athey & Wager (2019). *Annals of Statistics*.

### Matrix Completion Methods

Unified framework encompassing synthetic control and regression approaches.

**Reference**: [Athey et al. (2021)](https://arxiv.org/abs/1710.10251). *Journal of the American Statistical Association*.

### Double/Debiased ML for DiD

For high-dimensional settings with many potential confounders.

**Reference**: Chernozhukov et al. (2018). *The Econometrics Journal*.

### Alternative Inference Methods

- **Randomization inference**: Exact p-values for small samples
- **Bayesian DiD**: Priors on parallel trends violations
- **Conformal inference**: Prediction intervals with finite-sample guarantees

---

## Contributing

Interested in contributing? The Phase 10 items and future estimators are good candidates. See the [GitHub repository](https://github.com/igerber/diff-diff) for open issues.

Key references for implementation:
- [Roth et al. (2023)](https://www.sciencedirect.com/science/article/abs/pii/S0304407623001318). "What's Trending in Difference-in-Differences?" *Journal of Econometrics*.
- [Baker et al. (2025)](https://arxiv.org/pdf/2503.13323). "Difference-in-Differences Designs: A Practitioner's Guide."
