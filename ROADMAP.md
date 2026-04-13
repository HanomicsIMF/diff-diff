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
| **10g.** Practitioner guidance: when does survey design matter? | LOW | Subsumed by B1d |

---

## Data Science Practitioners (Phases B1–B4)

Parallel track targeting data science practitioners — marketing, product, operations — who need DiD for real-world problems but are underserved by the current academic framing. See [business-strategy.md](docs/business-strategy.md) for competitive analysis, personas, and full rationale.

### Phase B1: Foundation (Docs & Positioning)

*Goal: Make diff-diff discoverable and approachable for data science practitioners. Zero code changes.*

| Item | Priority | Status |
|------|----------|--------|
| **B1a.** Brand Awareness Survey DiD tutorial — lead use case showcasing unique survey support | HIGH | Done (Tutorial 17) |
| **B1b.** README "For Data Scientists" section alongside "For Academics" and "For AI Agents" | HIGH | Done |
| **B1c.** Practitioner decision tree — "which method should I use?" framed for business contexts | HIGH | Done |
| **B1d.** "Getting Started" guide for practitioners with business ↔ academic terminology bridge | MEDIUM | Done |

### Phase B2: Practitioner Content

*Goal: End-to-end tutorials for each persona. Ship incrementally, each as its own PR.*

| Item | Priority | Status |
|------|----------|--------|
| **B2a.** Marketing Campaign Lift tutorial (CallawaySantAnna, staggered geo rollout) | HIGH | Not started |
| **B2b.** Geo-Experiment tutorial (SyntheticDiD) | HIGH | Done (Tutorial 18) |
| **B2c.** diff-diff vs GeoLift vs CausalImpact comparison page | MEDIUM | Not started |
| **B2d.** Product Launch Regional Rollout tutorial (staggered estimators) | MEDIUM | Not started |
| **B2e.** Pricing/Promotion Impact tutorial (ContinuousDiD, dose-response) | MEDIUM | Not started |
| **B2f.** Loyalty Program Evaluation tutorial (TripleDifference) | LOW | Not started |

### Phase B3: Convenience Layer

*Goal: Reduce time-to-insight and enable stakeholder communication. Core stays numpy/pandas/scipy only.*

| Item | Priority | Status |
|------|----------|--------|
| **B3a.** `BusinessReport` class — plain-English summaries, markdown export; rich export via optional `[reporting]` extra | HIGH | Not started |
| **B3b.** `DiagnosticReport` — unified diagnostic runner with plain-English interpretation. Includes making `practitioner_next_steps()` context-aware (substitute actual column names from fitted results into code snippets instead of generic placeholders). | HIGH | Not started |
| **B3c.** Practitioner data generator wrappers (thin wrappers around existing generators with business-friendly names) | MEDIUM | Not started |
| **B3d.** `aggregate_survey()` helper (microdata-to-panel bridge for BRFSS/ACS/CPS) | MEDIUM | Shipped (v3.0.1) |

### Phase B4: Platform (Longer-term)

*Goal: Integrate into data science practitioner workflows.*

| Item | Priority | Status |
|------|----------|--------|
| **B4a.** Integration guides (Databricks, Jupyter dashboards, survey platforms) | MEDIUM | Not started |
| **B4b.** Export templates (PowerPoint via optional extra, Confluence/Notion markdown, HTML widget) | MEDIUM | Not started |
| **B4c.** AI agent integration — position B3a/B3b as tools for AI agents assisting practitioners | LOW | Not started |

---

## de Chaisemartin-D'Haultfœuille (dCDH) Estimator

The dCDH estimator is the only modern DiD estimator in the library that handles **non-absorbing (reversible) treatments**. All other staggered estimators (CallawaySantAnna, SunAbraham, ImputationDiD, TwoStageDiD, EfficientDiD, WooldridgeDiD) assume treatment is an absorbing state — once treated, always treated. dCDH is the natural fit for marketing campaigns, seasonal promotions, policy on/off cycles, and any setting where treatment turns on and off over time.

**Implementation strategy.** A single `ChaisemartinDHaultfoeuille` (alias `DCDH`) class evolves across phases via additional `fit()` parameters and additional fields on the results object. Not an estimator family — features land as enhancements to the single class, matching the library's pattern for `CallawaySantAnna`, `ImputationDiD`, `EfficientDiD`, etc.

**Methodology source of truth:** [docs/methodology/REGISTRY.md `## ChaisemartinDHaultfoeuille`](docs/methodology/REGISTRY.md) — assumption checks, estimator equations, edge cases, and all documented deviations from the R `DIDmultiplegtDYN` reference implementation. Consult REGISTRY.md before any methodology change.

**Primary papers** (consulted by the implementer; not committed in-repo as they are upstream sources):
- de Chaisemartin, C. & D'Haultfœuille, X. (2020). Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects. *American Economic Review*, 110(9), 2964-2996. — `DID_M` contemporaneous-switch estimator, TWFE decomposition diagnostics.
- de Chaisemartin, C. & D'Haultfœuille, X. (2022, revised 2024). Difference-in-Differences Estimators of Intertemporal Treatment Effects. NBER Working Paper 29873. — Full dynamic event study `DID_l`, cohort-recentered analytical variance (Web Appendix Section 3.7.3), residualization-style covariates `DID^X`, group-specific linear trends `DID^{fd}`.

The dynamic companion paper subsumes the AER 2020 paper: `DID_1 = DID_M`. The single class implements the dynamic estimator's machinery (`DID_{g,l}` building block, cohort-recentered analytical variance from Web Appendix Section 3.7.3 of the dynamic paper) at horizon `l = 1` for Phase 1, with later phases looping over multiple horizons and adding covariate / extension support.

### Phase 1: Foundation (contemporaneous switch / DID_M)

*Goal: Ship a `ChaisemartinDHaultfoeuille` estimator class for the contemporaneous-switch case (`l = 1`), which is `DID_M` of the AER 2020 paper. Forward-compatible API: parameters and result fields for Phase 2/3 are reserved from day one and raise `NotImplementedError` with phase pointers until they're implemented.*

| Item | Priority | Status |
|------|----------|--------|
| **1a.** `ChaisemartinDHaultfoeuille` class with `fit()` returning per-group `DID_{g,1}` and aggregate `DID_1` / `DID_M` | HIGH | Shipped |
| **1b.** Joiners-only (`DID_+`) and leavers-only (`DID_-`) views on the results object | HIGH | Shipped |
| **1c.** Single-lag placebo `DID_M^pl` (AER 2020 placebo specification = `DID^{pl}_1` of dynamic paper) | HIGH | Shipped (point estimate; analytical SE deferred to Phase 2) |
| **1d.** Analytical SE via cohort-recentered plug-in formula (Web Appendix Section 3.7.3 of dynamic paper, applied at `l = 1`) | HIGH | Shipped |
| **1e.** Multiplier bootstrap clustered at the group level (library extension; matches CS / ImputationDiD / TwoStageDiD convention) | HIGH | Shipped |
| **1f.** TWFE decomposition diagnostic: per-`(g, t)` weights, fraction negative, `sigma_fe` (Theorem 1 of AER 2020 + `twowayfeweights` parity) | MEDIUM | Shipped |
| **1g.** Parity tests vs R `DIDmultiplegtDYN` at `l = 1` | HIGH | Shipped |
| **1h.** REGISTRY.md entry, doc-deps.yaml mapping, README.md section, RST docs, CHANGELOG.md entry | HIGH | Shipped |
| **1i.** Survey compatibility matrix in `docs/choosing_estimator.rst`: explicitly document **NO survey support** for dCDH (separate effort after all phases ship) | HIGH | Shipped |

### Phase 2: Dynamic event study (multiple horizons)

*Goal: Add multi-horizon event study to the same class via the `L_max` parameter. Loops the Phase 1 machinery over horizons `l = 1, ..., L`. No API breakage from Phase 1. No new tutorial - the comprehensive tutorial waits for Phase 3.*

| Item | Priority | Status |
|------|----------|--------|
| **2a.** Multi-horizon `DID_l` via per-group `DID_{g,l}` building block, with `L_max` parameter | HIGH | Shipped |
| **2b.** Multi-horizon analytical SE (cohort-recentered plug-in per horizon) | HIGH | Shipped |
| **2c.** Dynamic placebos `DID^{pl}_l` for pre-trends testing (Web Appendix Section 1.1 of dynamic paper) | HIGH | Shipped (point estimates; SE deferred) |
| **2d.** Normalized estimator `DID^n_l` (Section 3.2 of dynamic paper) | MEDIUM | Shipped |
| **2e.** Cost-benefit aggregate `delta` (Section 3.3 of dynamic paper, Lemma 4) | MEDIUM | Shipped |
| **2f.** Simultaneous (sup-t) confidence bands for event study plots | MEDIUM | Shipped |
| **2g.** `plot_event_study()` integration; `< 50%`-of-switchers warning for far horizons | MEDIUM | Shipped |
| **2h.** Parity tests vs `did_multiplegt_dyn` for multi-horizon designs | HIGH | Shipped (point estimates; SE/placebo parity deferred) |

### Phase 3: Covariates, extensions, and tutorial

*Goal: Add residualization-style covariate adjustment, group-specific linear trends, non-binary treatment support, HonestDiD integration, and a single comprehensive tutorial covering all three phases. This is the phase where dCDH ships as a complete public feature.*

| Item | Priority | Status |
|------|----------|--------|
| **3a.** Residualization-style covariate adjustment `DID^X` (Web Appendix Section 1.2 of dynamic paper). **Note:** NOT doubly-robust, NOT IPW, NOT Callaway-Sant'Anna-style. | HIGH | Shipped (PR B) |
| **3b.** Group-specific linear trends `DID^{fd}` (Web Appendix Section 1.3, Lemma 6) — second-difference estimator with cumulation for level effects | MEDIUM | Shipped (PR B) |
| **3c.** State-set-specific trends (`trends_nonparam` option, Web Appendix Section 1.4) | MEDIUM | Shipped (PR B) |
| **3d.** Heterogeneity testing `beta^{het}_l` (Web Appendix Section 1.5) | LOW | Shipped (PR B) |
| **3e.** Design-2 switch-in / switch-out separation (Web Appendix Section 1.6) | LOW | Shipped (PR B; convenience wrapper) |
| **3f.** Non-binary treatment support (the formula already handles it; this row is documentation + tests) | MEDIUM | Shipped (PR #300; also ships placebo SE, L_max=1 per-group path, parity SE assertions) |
| **3g.** HonestDiD (Rambachan-Roth) integration on `DID^{pl}_l` placebos | MEDIUM | Not started |
| **3h.** **Single comprehensive tutorial notebook** covering all three phases — Favara-Imbs (2015) banking deregulation replication as the headline application, with comparison plots vs LP / TWFE | HIGH | Not started |
| **3i.** Parity tests vs `did_multiplegt_dyn` for covariate and extension specifications | HIGH | Not started |

### Out of scope for the dCDH single-class evolution

These are referenced by the dCDH papers but live in *separate* efforts or *separate* companion papers we don't yet have:

- **Survey design integration** — deferred to a separate effort after all three phases ship. Phase 1 documents "no survey support" in the compatibility matrix; the separate effort revisits when Phase 3 is complete.
- **Fuzzy DiD** (within-cell-varying treatment, Web Appendix Section 1.7 of dynamic paper) → de Chaisemartin & D'Haultfœuille (2018), separate paper not yet reviewed
- **Principled anticipation handling and trimming rules** (footnote 14 of dynamic paper) → de Chaisemartin (2021), separate paper not yet reviewed
- **2SLS DiD** (referenced in AER appendix Section 3.4) → separate paper

These remain in **Future Estimators** below if/when we choose to extend.

### Architectural notes (for plan and PR reviewers)

- **Single `ChaisemartinDHaultfoeuille` class** (alias `DCDH`). Not a family. New features land as `fit()` parameters or fields on the results dataclass. No `DCDHDynamic`, `DCDHCovariate`, etc. Matches the library's idiomatic pattern: `CallawaySantAnna`, `ImputationDiD`, and `EfficientDiD` are all single classes that evolved across many phases.
- **Forward-compatible API from Phase 1.** `fit(aggregate=None, controls=None, trends_linear=None, L_max=None, ...)` accepts the Phase 2/3 parameters from day one and raises `NotImplementedError` with a clear pointer to the relevant phase until they are implemented. No signature changes between phases.
- **Conservative CI** under Assumption 8 (independent groups), exact only under iid sampling. Documented in REGISTRY.md as a `**Note:**` deviation from "default nominal coverage." Theorem 1 of the dynamic paper.
- **Cohort recentering for variance is essential.** Cohorts are defined by the triple `(D_{g,1}, F_g, S_g)`. The plug-in variance subtracts cohort-conditional means, **NOT a single grand mean**. Test fixtures must catch this — a wrong implementation silently produces a smaller, incorrect variance.
- **No Rust acceleration is planned for any phase.** The estimator's hot path is groupby + BLAS-accelerated matrix-vector products, where NumPy already operates near-optimally. If profiling on large panels (`G > 100K`) reveals a bottleneck post-ship, the existing `_rust_bootstrap_weights` helper can be reused for the bootstrap loop without writing new Rust code.
- **No survey design integration in any phase.** Handled as a separate effort after all three phases ship. Phase 1 documents the absence in the compatibility matrix so survey users do not silently apply survey weights and get wrong answers.

---

## Future Estimators

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
