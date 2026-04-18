Efficient Difference-in-Differences
====================================

Semiparametrically efficient ATT estimator for staggered adoption designs
from Chen, Sant'Anna & Xie (2025).

This module implements the efficiency-bound-attaining estimator that:

1. **Achieves the semiparametric efficiency bound** for ATT(g,t) estimation on the no-covariate path
2. **Optimally weights** across comparison groups and baselines via the
   inverse covariance matrix Ω*
3. **Supports two PT assumptions**: PT-All (overidentified, tighter SEs) and
   PT-Post (just-identified, matches CS for post-treatment effects)
4. **Uses EIF-based inference** for analytical standard errors and multiplier
   bootstrap

.. note::

   EfficientDiD supports a doubly-robust covariate path: sieve-based
   propensity score ratios combined with a linear OLS outcome regression.
   The DR property ensures consistency if either the OR or the PS ratio is
   correctly specified, but the linear OLS working model for the outcome
   regression does not generically attain the semiparametric efficiency
   bound unless the conditional mean is linear in the covariates. The
   unqualified efficiency-bound claim applies to the no-covariate path
   only. Pass column names to the ``covariates`` parameter on ``fit()``.
   See ``docs/methodology/REGISTRY.md`` for the full contract.

**When to use EfficientDiD:**

- Staggered adoption design where you want **maximum efficiency** on the no-covariate path
- You believe parallel trends holds across all pre-treatment periods (PT-All)
- You want tighter confidence intervals than Callaway-Sant'Anna
- You need a formal efficiency benchmark for comparing estimators

For covariate-adjusted designs, the doubly-robust path is consistent under
either outcome-regression or propensity-ratio correctness but does not
generically attain the efficiency bound under the shipped linear OLS
outcome regression.

**Reference:** Chen, X., Sant'Anna, P. H. C., & Xie, H. (2025). Efficient
Difference-in-Differences and Event Study Estimators.

.. module:: diff_diff.efficient_did

EfficientDiD
-------------

Main estimator class for Efficient Difference-in-Differences.

.. autoclass:: diff_diff.EfficientDiD
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

   .. rubric:: Methods

   .. autosummary::

      ~EfficientDiD.fit
      ~EfficientDiD.get_params
      ~EfficientDiD.set_params

EfficientDiDResults
-------------------

Results container for Efficient DiD estimation.

.. autoclass:: diff_diff.efficient_did_results.EfficientDiDResults
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~EfficientDiDResults.summary
      ~EfficientDiDResults.print_summary
      ~EfficientDiDResults.to_dataframe

EDiDBootstrapResults
--------------------

Bootstrap inference results for Efficient DiD.

.. autoclass:: diff_diff.efficient_did_bootstrap.EDiDBootstrapResults
   :members:
   :undoc-members:
   :show-inheritance:

Example Usage
-------------

Basic usage::

    from diff_diff import EfficientDiD, generate_staggered_data

    data = generate_staggered_data(n_units=300, n_periods=10,
                                    cohort_periods=[4, 6, 8], seed=42)

    edid = EfficientDiD(pt_assumption="all")
    results = edid.fit(data, outcome='outcome', unit='unit',
                       time='period', first_treat='first_treat',
                       aggregate='all')
    results.print_summary()

PT-Post mode (matches CS for post-treatment ATT)::

    edid_post = EfficientDiD(pt_assumption="post")
    results_post = edid_post.fit(data, outcome='outcome', unit='unit',
                                  time='period', first_treat='first_treat',
                                  aggregate='all')
    print(f"PT-All ATT:  {results.overall_att:.4f} (SE={results.overall_se:.4f})")
    print(f"PT-Post ATT: {results_post.overall_att:.4f} (SE={results_post.overall_se:.4f})")

Bootstrap inference::

    edid_boot = EfficientDiD(pt_assumption="all", n_bootstrap=999, seed=42)
    results_boot = edid_boot.fit(data, outcome='outcome', unit='unit',
                                  time='period', first_treat='first_treat',
                                  aggregate='all')
    print(f"Bootstrap SE: {results_boot.overall_se:.4f}")
    print(f"Bootstrap CI: [{results_boot.overall_conf_int[0]:.4f}, "
          f"{results_boot.overall_conf_int[1]:.4f}]")

Comparison with Other Staggered Estimators
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 27 27 26

   * - Feature
     - EfficientDiD
     - CallawaySantAnna
     - ImputationDiD
   * - Approach
     - Optimal EIF-based weighting
     - Separate 2x2 DiD aggregation
     - Impute Y(0) via FE model
   * - PT assumption
     - PT-All (stronger) or PT-Post
     - Conditional PT
     - Strict exogeneity
   * - Efficiency
     - Achieves semiparametric bound
     - Not efficient
     - Efficient under homogeneity
   * - Covariates
     - Supported (doubly robust, sieve-based)
     - Supported (OR, IPW, DR)
     - Supported
   * - Bootstrap
     - Multiplier bootstrap (EIF)
     - Multiplier bootstrap
     - Multiplier bootstrap
   * - PT-Post equivalence
     - Matches CS post-treatment ATT(g,t)
     - Baseline
     - Different framework
