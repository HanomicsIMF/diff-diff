.. meta::
   :description: Which Difference-in-Differences method fits your business problem? Match your campaign, product launch, or pricing scenario to the right analysis method in diff-diff.
   :keywords: which DiD method, python marketing campaign analysis, geo experiment method, campaign lift measurement, business causal inference, staggered rollout analysis

Which Analysis Method Fits Your Problem?
========================================

You ran a campaign, launched a product, or changed pricing in some markets. You need
to know whether it worked - and by how much. This guide matches your situation to the
right analysis method. No econometrics background required.

Start Here
----------

Which of these best describes your situation?

1. **My campaign launched in all test markets at the same time**

   Your treatment started on the same date everywhere. Go to
   :ref:`section-simultaneous`.

2. **My campaign rolled out in waves** (some markets in March, more in June, etc.)

   Different markets started at different times. Go to
   :ref:`section-staggered`.

3. **My campaign turned on and off** (always-on with periodic dark periods, seasonal flights, holdout pulses)

   Treatment switches on AND off in the same market over time. Go to
   :ref:`section-reversible`.

4. **I varied spending levels across markets** (e.g., $50K, $100K, $200K)

   You want to know how the effect changes with the amount spent. Go to
   :ref:`section-dose`.

5. **I only have 3-5 test markets**

   Too few treated units for standard methods. Go to
   :ref:`section-few-markets`.

6. **I have survey data** (brand tracking, customer satisfaction, etc.)

   Your outcome comes from a survey with complex sampling. Go to
   :ref:`section-survey`.

.. tip::

   In academic literature, "rolling out in waves" is called *staggered adoption*,
   and markets are called *units*. You will see these terms in the detailed
   documentation, but this guide uses business language throughout.


.. _section-simultaneous:

Campaign Launched Simultaneously
--------------------------------

**Your situation:** You launched the campaign on the same date in all test markets.
You have outcome data (sales, signups, etc.) for both test and control markets, before
and after the launch.

**Recommended method:** :class:`~diff_diff.DifferenceInDifferences`

This is the simplest and most interpretable approach. It compares the before/after
change in your test markets to the before/after change in your control markets.

.. code-block:: python

   from diff_diff import DifferenceInDifferences, generate_did_data

   # 20 markets, 12 months, campaign launches at month 7 in 8 markets
   data = generate_did_data(
       n_units=20, n_periods=12, treatment_effect=5.0,
       treatment_fraction=0.4, treatment_period=7, seed=42,
   )

   did = DifferenceInDifferences()
   results = did.fit(data, outcome="outcome", treatment="treated", time="post")
   print(f"Campaign lift: {results.att:.1f} (p = {results.p_value:.4f})")

.. note::

   **Academic term:** This is the classic *2x2 DiD* design. The estimate is called the
   *ATT* (Average Treatment Effect on the Treated) - it tells you the average lift
   among the markets that received the campaign.

**When to upgrade:**

- If you have many time periods and want unit-level controls:
  :class:`~diff_diff.TwoWayFixedEffects`
- If you want to see how the effect evolves over time (week by week):
  :class:`~diff_diff.MultiPeriodDiD`


.. _section-staggered:

Staggered Rollout
-----------------

**Your situation:** Your campaign launched in some markets in one month, more markets
a few months later, and so on. Different markets were treated at different times.

**Recommended method:** :class:`~diff_diff.CallawaySantAnna`

.. warning::

   Do **not** use basic DiD or TWFE for staggered rollouts. When markets are treated
   at different times, these methods can compare already-active markets to newly-launched
   ones - giving biased (potentially wrong-sign) results. Callaway-Sant'Anna avoids this
   by comparing each wave only to true control markets.

.. code-block:: python

   from diff_diff import CallawaySantAnna, generate_staggered_data

   # Campaign launches in wave 1 markets at month 4, wave 2 at month 7
   data = generate_staggered_data(
       n_units=20, n_periods=12, cohort_periods=[4, 7],
       never_treated_frac=0.4, treatment_effect=5.0, seed=42,
   )

   cs = CallawaySantAnna()
   results = cs.fit(
       data, outcome="outcome", unit="unit",
       time="period", first_treat="first_treat",
   )
   print(f"Overall campaign lift: {results.overall_att:.1f}")

.. note::

   **Academic term:** This is a *staggered adoption* design with *heterogeneous treatment
   timing*. Callaway & Sant'Anna (2021) is the standard method. The *ATT(g,t)* gives you
   the lift for each rollout wave at each time period, and the *overall ATT* aggregates
   them into a single number.


.. _section-reversible:

Reversible Treatment (On/Off Cycles)
------------------------------------

**Your situation:** Your campaign isn't a one-time launch. It runs in some markets,
then pauses for a few weeks, then resumes. Or you have always-on activity with
periodic "dark periods" where you go quiet in some markets to measure incrementality.
Or you run seasonal flights that go on, off, and back on across the year.

The key feature: **the same market goes from treated to untreated to treated again**.
This breaks every other modern staggered estimator (Callaway-Sant'Anna, Sun-Abraham,
Imputation DiD, Two-Stage DiD, Efficient DiD, ETWFE), which all assume that once a
market is treated it stays treated.

**Recommended method:** :class:`~diff_diff.ChaisemartinDHaultfoeuille` (alias :class:`~diff_diff.DCDH`)

This is the **only library estimator** that handles non-absorbing (reversible)
treatments. It compares period-to-period outcome changes in markets that switch
into treatment ("joiners") and markets that switch out ("leavers"), against
simultaneously-stable controls. You get three numbers: the overall lift `DID_M`,
a joiners-only view `DID_+`, and a leavers-only view `DID_-`.

.. code-block:: python

   from diff_diff import ChaisemartinDHaultfoeuille
   from diff_diff.prep import generate_reversible_did_data

   # 80 markets, 6 periods, treatment switches on or off once per market
   data = generate_reversible_did_data(
       n_groups=80, n_periods=6, pattern="single_switch", seed=42,
   )

   est = ChaisemartinDHaultfoeuille()
   results = est.fit(
       data, outcome="outcome", group="group",
       time="period", treatment="treatment",
   )
   results.print_summary()

   print(f"Overall lift (DID_M): {results.overall_att:.2f}")
   print(f"Joiners only (DID_+): {results.joiners_att:.2f}")
   print(f"Leavers only (DID_-): {results.leavers_att:.2f}")

.. note::

   **Academic term:** This is the de Chaisemartin & D'Haultfœuille (2020) `DID_M`
   estimator, equivalently `DID_1` (horizon `l = 1`) of their dynamic companion
   paper (NBER WP 29873). It is the standard method for *non-absorbing* or
   *reversible* treatments. The Python implementation matches the R
   `DIDmultiplegtDYN` reference package maintained by the paper authors.

.. warning::

   By default, the estimator drops markets whose treatment switches more than
   once before estimation (``drop_larger_lower=True``, matching the R reference).
   Each drop emits a warning. If your design has many multi-switch markets and
   you need them all, raise this with the diff-diff maintainers — Phase 2 of the
   estimator will add explicit multi-switch handling via the dynamic event-study
   path.

.. note::

   Single-lag placebo (`DID_M^pl`) is computed automatically and exposed via
   ``results.placebo_effect``. The placebo inference fields (SE, p-value, CI)
   are intentionally ``NaN`` in Phase 1 — and stay ``NaN`` even when
   ``n_bootstrap > 0``. The dynamic companion paper Section 3.7.3 derives
   the cohort-recentered analytical variance for ``DID_l`` only;
   placebo-bootstrap support is deferred to Phase 2.


.. _section-dose:

Varying Spending Levels
-----------------------

**Your situation:** You spent different amounts across markets - $50K in some, $100K in
others, $200K in others. You want to know how the effect changes with spending level.

**Recommended method:** :class:`~diff_diff.ContinuousDiD`

This estimator can show how the average lift varies with spending level, with the
appropriate identification assumptions in place.

.. code-block:: python

   from diff_diff import ContinuousDiD, generate_continuous_did_data

   # Markets with varying spending levels (dose)
   data = generate_continuous_did_data(n_units=100, n_periods=4, seed=42)

   cdid = ContinuousDiD()
   results = cdid.fit(
       data, outcome="outcome", unit="unit",
       time="period", first_treat="first_treat", dose="dose",
   )
   print(f"Average lift across dose levels: {results.overall_att:.1f}")

.. warning::

   Dose-response curves *ATT(d)* and *ACRT(d)* require **Strong Parallel Trends (SPT)** -
   no selection into spending level on the basis of treatment effects. Under standard
   parallel trends, only the binarized average effect (*ATT^loc*) is identified. Your
   data must also include an untreated group (markets with zero spend), a balanced panel,
   and time-invariant dose (each market's spending level fixed across periods).

.. note::

   **Academic term:** This is a *continuous treatment* DiD (Callaway, Goodman-Bacon &
   Sant'Anna 2024). The *dose* is the spending level. Under standard parallel trends,
   the method identifies *ATT(d|d)* - the average lift at dose *d* among markets that
   actually received dose *d*. Cross-dose comparisons and the full *ATT(d)* curve
   require Strong Parallel Trends (see warning above).


.. _section-few-markets:

Few Test Markets
----------------

**Your situation:** You have only 3-5 test markets and 15-50 controls. Standard methods
struggle because there are too few treated units to estimate reliably.

**Recommended method:** :class:`~diff_diff.SyntheticDiD`

This method constructs a weighted blend of control markets that closely tracks your test
markets before the campaign. The "synthetic control" provides a better counterfactual than
a simple average of all controls.

.. code-block:: python

   from diff_diff import SyntheticDiD, generate_did_data

   # Only 3 test markets out of 20 (treatment_fraction=0.15)
   data = generate_did_data(
       n_units=20, n_periods=12, treatment_effect=5.0,
       treatment_fraction=0.15, treatment_period=7, seed=42,
   )

   # Pass post_periods explicitly so the analysis window matches the campaign window.
   # (Without this, SyntheticDiD defaults to the last half of periods.)
   post_periods = sorted(data.loc[data["post"] == 1, "period"].unique())

   sdid = SyntheticDiD()
   results = sdid.fit(
       data, outcome="outcome", unit="unit",
       time="period", treatment="treated", post_periods=post_periods,
   )
   print(f"Campaign lift: {results.att:.1f} (SE = {results.se:.2f})")

.. note::

   **Academic term:** *Synthetic Difference-in-Differences* (Arkhangelsky et al. 2021)
   combines the synthetic control method with DiD. It finds unit weights and time weights
   that minimize pre-treatment differences, then estimates the treatment effect using those
   weights.

.. tip::

   For a full walkthrough including diagnostics, inference, and a stakeholder
   communication template, see `Tutorial 18: Geo-Experiment Analysis with SyntheticDiD
   <tutorials/18_geo_experiments.ipynb>`_.


.. _section-survey:

Survey Data
-----------

**Your situation:** Your outcome comes from a survey - brand tracking, customer
satisfaction, NPS, or similar. The survey uses stratified sampling, clustering (e.g.,
by geography), or probability weights.

**Answer:** Use any of the methods above, combined with
:class:`~diff_diff.SurveyDesign`.

Ignoring survey weights and clustering makes your confidence intervals too narrow -
you will be overconfident about the result. Passing a ``SurveyDesign`` to ``fit()``
corrects for this automatically.

**If your data is individual-level microdata** (e.g., BRFSS, ACS, CPS, or NHANES
respondent records), use :func:`~diff_diff.aggregate_survey` first to roll it up
to a geographic-period panel with inverse-variance precision weights. The
returned second-stage design uses ``weight_type="aweight"``, so it works with
estimators marked **Full** in the :ref:`survey-design-support` matrix (DiD,
TWFE, MultiPeriodDiD, SunAbraham, ContinuousDiD, EfficientDiD) but not with
``pweight``-only estimators like ``CallawaySantAnna`` or ``ImputationDiD``.
See :doc:`practitioner_getting_started` for an end-to-end example.

.. code-block:: python

   from diff_diff import DifferenceInDifferences, SurveyDesign

   # Reference column names in your data; SurveyDesign resolves them at fit time.
   survey = SurveyDesign(
       weights="sample_weight",  # observation-level sampling weight
       strata="stratum",         # stratification variable
       psu="cluster_id",         # primary sampling unit (e.g., geography)
   )

   did = DifferenceInDifferences()
   results = did.fit(
       data, outcome="outcome", treatment="treated",
       time="post", survey_design=survey,
   )

.. tip::

   For a full walkthrough with brand funnel metrics and staggered rollouts, see
   `Tutorial 17: Brand Awareness Survey
   <tutorials/17_brand_awareness_survey.ipynb>`_.


At a Glance
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - Your Situation
     - Recommended Method
     - Key Benefit
   * - Campaign in some markets, all at once
     - ``DifferenceInDifferences``
     - Simple and interpretable
   * - Staggered rollout (waves)
     - ``CallawaySantAnna``
     - Handles different launch dates correctly
   * - On/off cycles (reversible treatment)
     - ``ChaisemartinDHaultfoeuille``
     - Only library option for non-absorbing treatments
   * - Varied spending levels
     - ``ContinuousDiD``
     - Dose-response curve
   * - Only a few test markets
     - ``SyntheticDiD``
     - Optimal with few treated units
   * - Survey data (any design above)
     - Any + ``SurveyDesign``
     - Correct confidence intervals


What About the Other Estimators?
--------------------------------

diff-diff has 17 estimators covering advanced scenarios: Sun-Abraham for
interaction-weighted estimation, Imputation DiD and Two-Stage DiD for alternative
staggered approaches, Stacked DiD, Efficient DiD, Triple Difference, TROP, and more.
The six scenarios above cover the most common business use cases.

For the full academic decision tree with all estimators, see :doc:`choosing_estimator`.


Next Steps
----------

- :doc:`practitioner_getting_started` - Walk through a complete analysis end-to-end
- `Tutorial 17: Brand Awareness Survey <tutorials/17_brand_awareness_survey.ipynb>`_ -
  Full example with survey design, brand funnel metrics, and staggered rollouts
- :doc:`quickstart` - Academic quickstart (if you already know DiD methodology)
- :doc:`api/index` - Full API reference
