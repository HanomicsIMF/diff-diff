# diff-diff Tutorials

This directory contains Jupyter notebook tutorials demonstrating the features of the `diff-diff` library.

## Notebooks

### 1. Basic DiD (`01_basic_did.ipynb`)
Introduction to Difference-in-Differences with `diff-diff`:
- Basic 2x2 DiD estimation
- Column-name and formula interfaces
- Adding covariates
- Fixed effects (dummy and absorbed)
- Two-Way Fixed Effects (TWFE)
- Cluster-robust standard errors
- Wild cluster bootstrap

### 2. Staggered DiD (`02_staggered_did.ipynb`)
Handling staggered treatment adoption with the Callaway-Sant'Anna estimator:
- Understanding staggered adoption
- Problems with TWFE in staggered settings
- **Goodman-Bacon decomposition**: Diagnosing *why* TWFE fails
- Group-time effects ATT(g,t)
- Aggregation methods (simple, group, event-study)
- Control group specifications
- Visualization

### 3. Synthetic DiD (`03_synthetic_did.ipynb`)
Synthetic Difference-in-Differences for few treated units:
- When to use Synthetic DiD
- Understanding unit and time weights
- Pre-treatment fit diagnostics
- Inference methods (bootstrap, placebo)
- Regularization tuning
- Comparison with standard DiD

### 4. Parallel Trends (`04_parallel_trends.ipynb`)
Testing assumptions and diagnostics:
- Visual inspection of trends
- Simple parallel trends tests
- Robust Wasserstein-based tests
- Equivalence testing (TOST)
- Placebo tests (timing, group, permutation)
- Event study as a diagnostic
- What to do if parallel trends fails

### 15. Efficient DiD (`15_efficient_did.ipynb`)
Efficient Difference-in-Differences (Chen, Sant'Anna & Xie 2025):
- Optimal weighting across comparison groups and baselines
- PT-All vs PT-Post assumptions
- Efficiency gains vs Callaway-Sant'Anna
- Event study and group-level aggregation
- Bootstrap inference and diagnostics

### 16. Wooldridge ETWFE (`16_wooldridge_etwfe.ipynb`)
Wooldridge Extended Two-Way Fixed Effects (ETWFE) for staggered DiD:
- Basic OLS estimation with cohort x time ATT cells
- Aggregation methods: event-study, group, calendar, simple
- Poisson QMLE for count / non-negative outcomes
- Logit for binary outcomes
- Comparison with Callaway-Sant'Anna
- Delta-method standard errors

### Survey-Aware DiD (`16_survey_did.ipynb`)
Survey-aware DiD with complex sampling designs (strata, PSU, FPC, weights):
- Why survey design matters for DiD inference
- Setting up `SurveyDesign` (weights, strata, PSU, FPC)
- Basic DiD and staggered DiD with survey design
- Replicate weights (JK1, BRR, Fay, JKn)
- Subpopulation analysis
- DEFF diagnostics
- Repeated cross-sections with survey design

### 17. Brand Awareness Survey (`17_brand_awareness_survey.ipynb`)
Practitioner walkthrough for measuring brand-campaign lift on survey data with complex sampling:
- The brand-tracker problem framed for marketing analytics
- Naive vs survey-aware DiD comparison (overconfidence under naive)
- `SurveyDesign` setup (strata, PSU, FPC, weights) wired into the fit
- Funnel-metric extension across awareness / consideration / purchase intent
- Diagnostics (parallel trends, placebo, automated `practitioner_next_steps()`)
- Stakeholder communication template

### 18. Geo-Experiment Analysis with SyntheticDiD (`18_geo_experiments.ipynb`)
Practitioner walkthrough for marketing analytics teams measuring geo-experiment lift:
- The geo-experiment problem framed for marketing analytics
- Synthetic panel of 80 markets with simulated campaign launch
- `SyntheticDiD` fit, diagnostics, and inference (placebo + bootstrap)
- Unit weights and time weights interpretation
- Stakeholder communication template (Tutorial 17 Section 9 pattern)

### 19. dCDH Marketing Pulse Campaigns (`19_dcdh_marketing_pulse.ipynb`)
Practitioner walkthrough for measuring lift from on/off promotional pulses across markets, where treatment can switch in both directions:
- The marketing-pulse problem framed for reversible (non-absorbing) treatment
- TWFE decomposition diagnostic (`twowayfeweights`) showing why standard regression misleads on reversible panels (de Chaisemartin & D'Haultfoeuille 2020 Theorem 1)
- `DCDH` Phase 1: DID_M, joiners-vs-leavers decomposition, single-lag placebo
- Multi-horizon event study with `L_max` + multiplier bootstrap
- Stakeholder communication template + drift guards

## Running the Notebooks

1. Install diff-diff with dependencies:
```bash
pip install diff-diff
pip install matplotlib  # for visualizations
pip install jupyter     # to run notebooks
```

2. Start Jupyter:
```bash
jupyter notebook
```

3. Open any notebook and run the cells.

## Requirements

- Python 3.8+
- diff-diff
- numpy
- pandas
- matplotlib (optional, for visualizations)
