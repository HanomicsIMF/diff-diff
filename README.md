# diff-diff

<p align="center">
  <img src="https://raw.githubusercontent.com/igerber/diff-diff/main/diff-diff.png"
       alt="diff-diff: Difference-in-Differences causal inference in Python - sklearn-like API with Callaway-Sant'Anna, Synthetic DiD, Honest DiD, and Event Studies"
       width="800">
</p>

[![PyPI version](https://img.shields.io/pypi/v/diff-diff.svg)](https://pypi.org/project/diff-diff/)
[![Python versions](https://img.shields.io/pypi/pyversions/diff-diff.svg)](https://pypi.org/project/diff-diff/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://img.shields.io/pypi/dm/diff-diff.svg)](https://pypi.org/project/diff-diff/)
[![Documentation](https://readthedocs.org/projects/diff-diff/badge/?version=stable)](https://diff-diff.readthedocs.io/en/stable/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19646175.svg)](https://doi.org/10.5281/zenodo.19646175)

A Python library for Difference-in-Differences (DiD) causal inference - sklearn-like estimators with statsmodels-style outputs, built for econometricians, marketing analysts, and data scientists running campaign-lift, policy, and staggered-rollout analyses.

## Installation

```bash
pip install diff-diff
```

For development:

```bash
git clone https://github.com/igerber/diff-diff.git
cd diff-diff
pip install -e ".[dev]"
```

## Quick Start

```python
import pandas as pd
from diff_diff import DifferenceInDifferences  # or: DiD

data = pd.DataFrame({
    'outcome': [10, 11, 15, 18, 9, 10, 12, 13],
    'treated': [1, 1, 1, 1, 0, 0, 0, 0],
    'post': [0, 0, 1, 1, 0, 0, 1, 1],
})

did = DifferenceInDifferences()
results = did.fit(data, outcome='outcome', treatment='treated', time='post')
print(results)              # DiDResults(ATT=3.0000, SE=1.7321, p=0.1583)
results.print_summary()     # full statsmodels-style table
```

## Documentation

- [Quickstart](https://diff-diff.readthedocs.io/en/stable/quickstart.html) - basic 2x2 DiD with column-name and formula interfaces, covariates, fixed effects, cluster-robust SEs
- [Choosing an Estimator](https://diff-diff.readthedocs.io/en/stable/choosing_estimator.html) - decision flowchart for picking the right estimator
- [Tutorials](https://diff-diff.readthedocs.io/en/stable/tutorials/01_basic_did.html) - hands-on Jupyter notebooks covering every estimator and design pattern
- [Troubleshooting](https://diff-diff.readthedocs.io/en/stable/troubleshooting.html) - common issues and solutions
- [R Comparison](https://diff-diff.readthedocs.io/en/stable/r_comparison.html) | [Python Comparison](https://diff-diff.readthedocs.io/en/stable/python_comparison.html) | [Benchmarks](https://diff-diff.readthedocs.io/en/stable/benchmarks.html) - validation results vs `did`, `synthdid`, `fixest`
- [API Reference](https://diff-diff.readthedocs.io/en/stable/api/index.html) - full API for all estimators, results classes, diagnostics, utilities

## For AI Agents

If you are an AI agent or LLM using this library, call `diff_diff.get_llm_guide()` for a concise API reference with an 8-step practitioner workflow (based on Baker et al. 2025). The workflow ensures rigorous DiD analysis - testing assumptions, running sensitivity analysis, and checking robustness, not just calling `fit()`.

```python
from diff_diff import get_llm_guide

get_llm_guide()                 # concise API reference
get_llm_guide("practitioner")   # 8-step workflow (Baker et al. 2025)
get_llm_guide("full")           # comprehensive documentation
get_llm_guide("autonomous")     # autonomous-agent variant
```

The guides are bundled in the wheel - accessible from a `pip install` with no network access. After estimation, call `practitioner_next_steps(results)` for context-aware guidance on remaining diagnostic steps.

## For Data Scientists

Measuring campaign lift? Evaluating a product launch? Rolling out a policy in waves? diff-diff handles the causal inference so you can focus on the business question.

- [Which method fits my problem?](https://diff-diff.readthedocs.io/en/stable/practitioner_decision_tree.html) - start from your business scenario (campaign in some markets, staggered rollout, survey data) and find the right estimator
- [Getting started for practitioners](https://diff-diff.readthedocs.io/en/stable/practitioner_getting_started.html) - end-to-end walkthrough from marketing campaign to causal estimate to stakeholder-ready result
- [Brand awareness survey tutorial](https://diff-diff.readthedocs.io/en/stable/tutorials/17_brand_awareness_survey.html) - full example with complex survey design, brand funnel analysis, and staggered rollouts
- Have BRFSS/ACS/CPS individual records? Use [`aggregate_survey()`](https://diff-diff.readthedocs.io/en/stable/api/prep.html) to roll respondent-level microdata into a geographic-period panel with inverse-variance precision weights for second-stage DiD

`BusinessReport` and `DiagnosticReport` are experimental preview classes that produce plain-English output and a structured `to_dict()` schema from any fitted result - wording and schema will evolve. See [docs/methodology/REPORTING.md](https://github.com/igerber/diff-diff/blob/main/docs/methodology/REPORTING.md) for usage and stability notes.

## Practitioner Workflow (Baker et al. 2025)

For rigorous DiD analysis, follow these 8 steps. Skipping diagnostic steps produces unreliable results.

1. **Define target parameter** - ATT, group-time ATT(g,t), or event-study ATT_es(e). State whether weighted or unweighted.
2. **State identification assumptions** - which parallel trends variant (unconditional, conditional, PT-GT-Nev, PT-GT-NYT), no-anticipation, overlap.
3. **Test parallel trends** - simple 2x2: `check_parallel_trends()`, `equivalence_test_trends()`; staggered: inspect CS event-study pre-period coefficients (generic PT tests are invalid for staggered designs). Insignificant pre-trends do NOT prove PT holds.
4. **Choose estimator** - staggered adoption -> CS/SA/BJS (NOT plain TWFE); few treated units -> SDiD; factor confounding -> TROP; simple 2x2 -> DiD. Run `BaconDecomposition` to diagnose TWFE bias.
5. **Estimate** - `estimator.fit(data, ...)`. Always print the cluster count first and choose inference method based on the result (cluster-robust if >= 50 clusters, wild bootstrap if fewer).
6. **Sensitivity analysis** - `compute_honest_did(results)` for bounds under PT violations (MultiPeriodDiD, CS, or dCDH), `run_all_placebo_tests()` for 2x2 falsification, specification comparisons for staggered designs.
7. **Heterogeneity** - CS: `aggregate='group'`/`'event_study'`; SA: `results.event_study_effects` / `to_dataframe(level='cohort')`; subgroup re-estimation.
8. **Robustness** - compare 2-3 estimators (CS vs SA vs BJS), report with and without covariates (shows whether conditioning drives identification), present pre-trends and sensitivity bounds.

Full guide: `diff_diff.get_llm_guide("practitioner")`.

## Estimators

- [DifferenceInDifferences](https://diff-diff.readthedocs.io/en/stable/api/estimators.html) - basic 2x2 DiD with robust/cluster-robust SEs, wild bootstrap, formula interface, and fixed effects
- [TwoWayFixedEffects](https://diff-diff.readthedocs.io/en/stable/api/estimators.html) - panel data DiD with unit and time fixed effects via within-transformation or dummies
- [MultiPeriodDiD](https://diff-diff.readthedocs.io/en/stable/api/estimators.html) - event study design with period-specific treatment effects for dynamic analysis
- [CallawaySantAnna](https://diff-diff.readthedocs.io/en/stable/api/staggered.html) - Callaway & Sant'Anna (2021) group-time ATT estimator for staggered adoption
- [ChaisemartinDHaultfoeuille](https://diff-diff.readthedocs.io/en/stable/api/chaisemartin_dhaultfoeuille.html) - de Chaisemartin & D'Haultfœuille (2020/2022) for **reversible (non-absorbing) treatments** with multi-horizon event study, normalized effects, cost-benefit delta, sup-t bands, and dynamic placebos. The only library option for treatments that switch on AND off. Alias `DCDH`.
- [SunAbraham](https://diff-diff.readthedocs.io/en/stable/api/staggered.html) - Sun & Abraham (2021) interaction-weighted estimator for heterogeneity-robust event studies
- [ImputationDiD](https://diff-diff.readthedocs.io/en/stable/api/imputation.html) - Borusyak, Jaravel & Spiess (2024) imputation estimator, most efficient under homogeneous effects
- [TwoStageDiD](https://diff-diff.readthedocs.io/en/stable/api/two_stage.html) - Gardner (2022) two-stage estimator with GMM sandwich variance
- [SyntheticDiD](https://diff-diff.readthedocs.io/en/stable/api/estimators.html) - Synthetic DiD combining standard DiD and synthetic control for few treated units
- [TripleDifference](https://diff-diff.readthedocs.io/en/stable/api/triple_diff.html) - triple difference (DDD) estimator for designs requiring two criteria for treatment eligibility
- [ContinuousDiD](https://diff-diff.readthedocs.io/en/stable/api/continuous_did.html) - Callaway, Goodman-Bacon & Sant'Anna (2024) continuous treatment DiD with dose-response curves
- [HeterogeneousAdoptionDiD](https://diff-diff.readthedocs.io/en/stable/api/had.html) - de Chaisemartin, Ciccia, D'Haultfœuille & Knau (2026) for designs where **no unit remains untreated**; local-linear estimator targeting the Weighted Average Slope (WAS) at the dose support boundary (Design 1' with `d̲ = 0` and Design 1 with `d̲ > 0`), with a multi-period event-study extension (last-treatment cohort, pointwise CIs). Alias `HAD`.
- [StackedDiD](https://diff-diff.readthedocs.io/en/stable/api/stacked_did.html) - Wing, Freedman & Hollingsworth (2024) stacked DiD with Q-weights and sub-experiments
- [EfficientDiD](https://diff-diff.readthedocs.io/en/stable/api/efficient_did.html) - Chen, Sant'Anna & Xie (2025) efficient DiD with optimal weighting for tighter SEs
- [TROP](https://diff-diff.readthedocs.io/en/stable/api/trop.html) - Triply Robust Panel estimator (Athey et al. 2025) with nuclear norm factor adjustment
- [StaggeredTripleDifference](https://diff-diff.readthedocs.io/en/stable/api/staggered.html#staggeredtripledifference) - Ortiz-Villavicencio & Sant'Anna (2025) staggered DDD with group-time ATT
- [WooldridgeDiD](https://diff-diff.readthedocs.io/en/stable/api/wooldridge_etwfe.html) - Wooldridge (2023, 2025) ETWFE: saturated OLS, logit/Poisson QMLE (ASF-based ATT). Alias `ETWFE`.
- [BaconDecomposition](https://diff-diff.readthedocs.io/en/stable/api/bacon.html) - Goodman-Bacon (2021) decomposition for diagnosing TWFE bias in staggered settings

## Diagnostics & Sensitivity

- [Parallel Trends Testing](https://diff-diff.readthedocs.io/en/stable/api/diagnostics.html) - simple and Wasserstein-robust parallel trends tests, equivalence testing (TOST)
- [Placebo Tests](https://diff-diff.readthedocs.io/en/stable/api/diagnostics.html) - placebo timing, group, permutation, leave-one-out
- [Honest DiD](https://diff-diff.readthedocs.io/en/stable/api/honest_did.html) - Rambachan & Roth (2023) sensitivity analysis: robust CI under PT violations, breakdown values
- [Pre-Trends Power Analysis](https://diff-diff.readthedocs.io/en/stable/api/pretrends.html) - Roth (2022) minimum detectable violation and power curves
- [Power Analysis](https://diff-diff.readthedocs.io/en/stable/api/power.html) - analytical and simulation-based MDE, sample size, power curves for study design

## Survey Support

All estimators accept an optional `survey_design` parameter for design-based variance estimation:

- **Design elements**: strata, PSU, FPC, weight types (pweight/fweight/aweight), lonely PSU handling, nest
- **Variance methods**: Taylor Series Linearization (TSL), replicate weights (BRR/Fay/JK1/JKn/SDR), survey-aware bootstrap
- **Diagnostics**: DEFF per coefficient, effective n, subpopulation analysis, weight trimming, CV on estimates
- **Repeated cross-sections**: `CallawaySantAnna(panel=False)` for BRFSS, ACS, CPS
- **Compatibility matrix**: see [Survey Design Support](https://diff-diff.readthedocs.io/en/stable/choosing_estimator.html#survey-design-support)

No other Python or R DiD package offers design-based variance estimation for modern heterogeneity-robust estimators.

## Requirements

- Python 3.9 - 3.14
- numpy >= 1.20
- pandas >= 1.3
- scipy >= 1.7

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black diff_diff tests
ruff check diff_diff tests
```

## References

This library implements methods from a wide body of econometric and causal-inference research. See the full bibliography on [Read the Docs](https://diff-diff.readthedocs.io/en/stable/references.html) for citations spanning DiD foundations, modern staggered estimators, sensitivity analysis, and synthetic controls.

## Citing diff-diff

If you use diff-diff in your research, please cite it:

```bibtex
@software{diff_diff,
  title = {diff-diff: Difference-in-Differences Causal Inference for Python},
  author = {Gerber, Isaac},
  year = {2026},
  url = {https://github.com/igerber/diff-diff},
  doi = {10.5281/zenodo.19646175},
  license = {MIT},
}
```

The DOI above is the Zenodo concept DOI - it always resolves to the latest release. To cite a specific version, look up its versioned DOI on [the Zenodo project page](https://doi.org/10.5281/zenodo.19646175).

See [`CITATION.cff`](https://github.com/igerber/diff-diff/blob/main/CITATION.cff) for the full citation metadata.

**Note on authorship**: academic citation (`CITATION.cff`, the BibTeX above) lists individual authors with ORCIDs per scholarly convention. Package metadata surfaces (`pyproject.toml`, Sphinx docs) list "diff-diff contributors" to acknowledge the collective - see [`CONTRIBUTORS.md`](https://github.com/igerber/diff-diff/blob/main/CONTRIBUTORS.md) for the full list.

## License

MIT License
