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

3. **I varied spending levels across markets** (e.g., $50K, $100K, $200K)

   You want to know how the effect changes with the amount spent. Go to
   :ref:`section-dose`.

4. **I only have 3-5 test markets**

   Too few treated units for standard methods. Go to
   :ref:`section-few-markets`.

5. **I have survey data** (brand tracking, customer satisfaction, etc.)

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


.. _section-dose:

Varying Spending Levels
-----------------------

**Your situation:** You spent different amounts across markets - $50K in some, $100K in
others, $200K in others. You want to know how the effect changes with spending level.

**Recommended method:** :class:`~diff_diff.ContinuousDiD`

Instead of a single "did it work?" answer, this gives you a dose-response curve showing
how the lift changes with the amount spent.

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

.. note::

   **Academic term:** This is a *continuous treatment* DiD (Callaway, Goodman-Bacon &
   Sant'Anna 2024). The *dose* is the spending level. The estimate *ATT(d)* gives
   you the lift at each spending level, not just an average.


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

   sdid = SyntheticDiD()
   results = sdid.fit(
       data, outcome="outcome", unit="unit",
       time="period", treatment="treated",
   )
   print(f"Campaign lift: {results.att:.1f} (SE = {results.se:.2f})")

.. note::

   **Academic term:** *Synthetic Difference-in-Differences* (Arkhangelsky et al. 2021)
   combines the synthetic control method with DiD. It finds unit weights and time weights
   that minimize pre-treatment differences, then estimates the treatment effect using those
   weights.


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

.. code-block:: python

   from diff_diff import DifferenceInDifferences, SurveyDesign

   survey = SurveyDesign(
       data=data,
       strata="stratum",     # sampling strata
       psu="cluster_id",     # primary sampling unit (e.g., geography)
       weight="sample_weight",
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

diff-diff has 16 estimators covering advanced scenarios: Sun-Abraham for
interaction-weighted estimation, Imputation DiD and Two-Stage DiD for alternative
staggered approaches, Stacked DiD, Efficient DiD, Triple Difference, TROP, and more.
The five scenarios above cover the most common business use cases.

For the full academic decision tree with all estimators, see :doc:`choosing_estimator`.


Next Steps
----------

- :doc:`practitioner_getting_started` - Walk through a complete analysis end-to-end
- `Tutorial 17: Brand Awareness Survey <tutorials/17_brand_awareness_survey.ipynb>`_ -
  Full example with survey design, brand funnel metrics, and staggered rollouts
- :doc:`quickstart` - Academic quickstart (if you already know DiD methodology)
- :doc:`api/index` - Full API reference
