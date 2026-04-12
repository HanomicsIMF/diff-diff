---
title: "diff-diff: Comprehensive Difference-in-Differences Causal Inference for Python"
tags:
  - difference-in-differences
  - causal-inference
  - econometrics
  - Python
  - treatment-effects
  - survey-data
authors:
  - name: Isaac Gerber
    orcid: 0009-0009-3275-5591
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 12 April 2026
bibliography: paper.bib
---

# Summary

`diff-diff` is a Python library for Difference-in-Differences (DiD) causal inference
analysis. It provides 16 estimators covering the full modern DiD toolkit - from classic
two-group/two-period designs through heterogeneity-robust staggered adoption methods,
synthetic control hybrids, and sensitivity analysis - under a consistent scikit-learn-style
API. All estimators accept an optional `SurveyDesign` object for design-based variance
estimation with complex survey data, a capability absent from existing DiD software in any
language. Point estimates and standard errors are validated against established R packages
to machine precision.

# Statement of Need

Difference-in-differences is the most widely used quasi-experimental research design in
applied economics and the social sciences. Since 2018, a wave of methodological advances
has addressed fundamental limitations of the conventional two-way fixed effects (TWFE)
estimator under staggered treatment adoption and heterogeneous effects [@Roth2023]. These
modern methods - including Callaway and Sant'Anna [-@Callaway2021], Sun and Abraham
[-@Sun2021], Borusyak, Jaravel, and Spiess [-@Borusyak2024], and others - are now standard
practice in applied work.

The R ecosystem provides mature implementations across several packages: `did`
[@Callaway2021], `fixest` [@Berge2018], `synthdid` [@Arkhangelsky2021], and `HonestDiD`
[@Rambachan2023]. Stata offers `csdid` and `didregress`. Python, however, lacks a unified
DiD library. Practitioners working in Python-based data science workflows - increasingly
common in industry settings for marketing measurement, product experimentation, and policy
evaluation - must either context-switch to R, reimplement methods from scratch, or rely on
partial implementations scattered across unrelated packages.

`diff-diff` fills this gap by providing a single-import library that covers 16 estimators
with a consistent API, survey-weighted inference, and numerical validation against R to
machine precision. It targets both applied researchers who need rigorous econometric methods
and data science practitioners who need accessible causal inference tools integrated into
Python workflows.

# Key Features

**Breadth of methods.** `diff-diff` implements 16 estimators organized across the modern
DiD taxonomy: classic DiD and TWFE; heterogeneity-robust staggered estimators including
Callaway-Sant'Anna [@Callaway2021], Sun-Abraham [@Sun2021], imputation
[@Borusyak2024], two-stage [@Gardner2022], stacked [@Wing2024], and efficient
[@Chen2025] approaches; extended designs including synthetic DiD [@Arkhangelsky2021],
triple difference [@OrtizVillavicencio2025], continuous treatment [@Callaway2024],
nonlinear ETWFE [@Wooldridge2023], and triply robust panel estimation [@Athey2025];
reversible-treatment DiD for non-absorbing interventions [@deChaisemartin2020]; and
diagnostics including Goodman-Bacon decomposition [@GoodmanBacon2021], Honest DiD
sensitivity analysis [@Rambachan2023], and pre-trends power analysis [@Roth2022]. All
estimators share a consistent `fit()` interface with `get_params()`/`set_params()` for
configuration, R-style formula support, and rich results objects with `summary()` output.
An optional Rust backend via PyO3 accelerates compute-intensive operations.

**Survey-weighted inference.** A `SurveyDesign` class supports stratification, primary
sampling units, finite population corrections, and probability weights. Variance estimation
includes Taylor series linearization, five replicate weight methods (BRR, Fay's BRR, JK1,
JKn, SDR), and survey-aware bootstrap. Survey variance is validated against R's `survey`
package [@Lumley2004] on three federal datasets (NHANES, RECS, API) to machine precision
(differences < 1e-10). No other DiD package in any language provides integrated survey
support.

**Validation against R.** Point estimates match the R `did`, `synthdid`, and `fixest`
packages to machine precision (differences < 1e-10). Standard errors match exactly for
core estimators including Callaway-Sant'Anna and basic DiD. Validation includes the
canonical MPDTA minimum-wage dataset from Callaway and Sant'Anna [-@Callaway2021].

**Practitioner tooling.** Beyond estimation, `diff-diff` includes a practitioner decision
tree for estimator selection, an 8-step diagnostic workflow based on Baker et al.
[-@Baker2025], AI agent integration with structured next-steps guidance, and microdata
aggregation utilities for converting individual-level survey responses into
geographic-period panels suitable for DiD analysis.

# Acknowledgments

Wenli Xu (Faculty of Finance, City University of Macau) implemented the WooldridgeDiD
(ETWFE) estimator, including saturated OLS, logit, and Poisson QMLE paths with ASF-based
ATT and delta-method standard errors. Development was assisted by Claude Code (Anthropic).

# References
