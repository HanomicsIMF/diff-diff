Wooldridge Extended Two-Way Fixed Effects (ETWFE)
===================================================

Extended Two-Way Fixed Effects estimator from Wooldridge (2021, 2023),
based on the Stata ``jwdid`` package specification (Friosavila 2021),
with documented SE/aggregation deviations noted in the Methodology Registry.

This module implements ETWFE via a single saturated regression that:

1. **Estimates ATT(g,t)** for each cohort×time treatment cell simultaneously
2. **Supports linear (OLS), Poisson QMLE, and logit** link functions
3. **Uses ASF-based ATT** for nonlinear models: E[f(η₁)] − E[f(η₀)]
4. **Computes delta-method SEs** for all aggregations (event, group, calendar, simple)
5. **Follows the Stata jwdid specification** for OLS and nonlinear paths (see Methodology Registry for documented SE/aggregation deviations)

**When to use WooldridgeDiD:**

- Staggered adoption design with heterogeneous treatment timing
- Nonlinear outcomes (binary, count, non-negative continuous)
- You want a single-regression approach matching Stata's ``jwdid``
- You need event-study, group, calendar, or simple ATT aggregations

**References:**

- Wooldridge, J. M. (2021). Two-Way Fixed Effects, the Two-Way Mundlak
  Regression, and Difference-in-Differences Estimators. *SSRN 3906345*.
- Wooldridge, J. M. (2023). Simple approaches to nonlinear
  difference-in-differences with panel data. *The Econometrics Journal*,
  26(3), C31–C66.
- Friosavila, F. (2021). ``jwdid``: Stata module for ETWFE. SSC s459114.

.. module:: diff_diff.wooldridge

WooldridgeDiD
--------------

Main estimator class for Wooldridge ETWFE.

.. autoclass:: diff_diff.WooldridgeDiD
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~WooldridgeDiD.fit
      ~WooldridgeDiD.get_params
      ~WooldridgeDiD.set_params

WooldridgeDiDResults
---------------------

Results container returned by ``WooldridgeDiD.fit()``.

.. autoclass:: diff_diff.wooldridge_results.WooldridgeDiDResults
   :members:
   :undoc-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::

      ~WooldridgeDiDResults.aggregate
      ~WooldridgeDiDResults.summary

Example Usage
-------------

Basic OLS (follows Stata ``jwdid y, ivar(unit) tvar(time) gvar(cohort)``)::

    import pandas as pd
    from diff_diff import WooldridgeDiD

    df = pd.read_stata("mpdta.dta")
    df['first_treat'] = df['first_treat'].astype(int)

    m = WooldridgeDiD()
    r = m.fit(df, outcome='lemp', unit='countyreal', time='year', cohort='first_treat')

    r.aggregate('event').aggregate('group').aggregate('simple')
    print(r.summary('event'))
    print(r.summary('group'))
    print(r.summary('simple'))

View cohort×time cell estimates (post-treatment)::

    for (g, t), v in sorted(r.group_time_effects.items()):
        if t >= g:
            print(f"g={g} t={t}  ATT={v['att']:.4f}  SE={v['se']:.4f}")

Poisson QMLE for non-negative outcomes
(follows Stata ``jwdid emp, method(poisson)``)::

    import numpy as np
    df['emp'] = np.exp(df['lemp'])

    m_pois = WooldridgeDiD(method='poisson')
    r_pois = m_pois.fit(df, outcome='emp', unit='countyreal',
                        time='year', cohort='first_treat')
    r_pois.aggregate('event').aggregate('group').aggregate('simple')
    print(r_pois.summary('simple'))

Logit for binary outcomes
(follows Stata ``jwdid y, method(logit)``)::

    m_logit = WooldridgeDiD(method='logit')
    r_logit = m_logit.fit(df, outcome='hi_emp', unit='countyreal',
                          time='year', cohort='first_treat')
    r_logit.aggregate('group').aggregate('simple')
    print(r_logit.summary('group'))

Aggregation Methods
-------------------

Call ``.aggregate(type)`` before ``.summary(type)``:

.. list-table::
   :header-rows: 1
   :widths: 15 30 25

   * - Type
     - Description
     - Stata equivalent
   * - ``'event'``
     - ATT by relative time k = t − g
     - ``estat event``
   * - ``'group'``
     - ATT averaged across post-treatment periods per cohort
     - ``estat group``
   * - ``'calendar'``
     - ATT averaged across cohorts per calendar period
     - ``estat calendar``
   * - ``'simple'``
     - Overall weighted average ATT
     - ``estat simple``

Comparison with Other Staggered Estimators
------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 27 27 26

   * - Feature
     - WooldridgeDiD (ETWFE)
     - CallawaySantAnna
     - ImputationDiD
   * - Approach
     - Single saturated regression
     - Separate 2×2 DiD per cell
     - Impute Y(0) via FE model
   * - Nonlinear outcomes
     - Yes (Poisson, Logit)
     - No
     - No
   * - Covariates
     - Via regression (linear index)
     - OR, IPW, DR
     - Supported
   * - SE for aggregations
     - Delta method
     - Multiplier bootstrap
     - Multiplier bootstrap
   * - Stata equivalent
     - ``jwdid``
     - ``csdid``
     - ``did_imputation``
