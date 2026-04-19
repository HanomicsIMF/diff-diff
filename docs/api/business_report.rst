BusinessReport
==============

``BusinessReport`` wraps any fitted diff-diff result object and produces
stakeholder-ready output:

- ``summary()`` — a short paragraph block suitable for an email or Slack.
- ``full_report()`` — a structured multi-section markdown report.
- ``to_dict()`` — a stable AI-legible structured schema (single source
  of truth; prose renders from this dict).

By default, BusinessReport constructs an internal ``DiagnosticReport``
to surface pre-trends, sensitivity, and other validity checks as part
of the narrative. Pass ``auto_diagnostics=False`` to skip this, or
``diagnostics=<DiagnosticReport>`` to supply an explicit one.

Data-dependent checks (2x2 parallel trends on simple DiD,
Goodman-Bacon decomposition on staggered estimators, the EfficientDiD
Hausman PT-All vs PT-Post pretest) require the raw panel + column
names. Pass ``data``, ``outcome``, ``treatment``, ``unit``, ``time``,
and/or ``first_treat`` to ``BusinessReport`` and they are forwarded
to the auto-constructed ``DiagnosticReport``. Without these kwargs,
those specific checks are skipped with an explicit reason while the
rest of the report still renders.

For survey-weighted fits (any result carrying
``survey_metadata``) pass the original ``SurveyDesign`` via
``survey_design=<design>``. It is threaded through to
``bacon_decompose`` for a fit-faithful Goodman-Bacon replay. When
``survey_metadata`` is set but ``survey_design`` is not supplied,
Bacon is skipped with an explicit reason so the report never emits
an unweighted decomposition for a design that differs from the
estimate. The simple 2x2 parallel-trends helper has no survey-aware
variant and is skipped unconditionally on a survey-backed
``DiDResults`` regardless of ``survey_design``; supply
``precomputed={'parallel_trends': ...}`` with a survey-aware
pretest to opt in.

Methodology deviations (no traffic-light gates, pre-trends verdict
thresholds, power-aware phrasing, unit-translation policy, schema
stability) are documented in :doc:`../methodology/REPORTING`.

Example
-------

.. code-block:: python

   from diff_diff import CallawaySantAnna, BusinessReport

   cs = CallawaySantAnna(base_period="universal").fit(
       df, outcome="revenue", unit="store", time="period",
       first_treat="first_treat", aggregate="event_study",
   )
   report = BusinessReport(
       cs,
       outcome_label="Revenue per store",
       outcome_unit="$",
       business_question="Did the loyalty program lift revenue?",
       treatment_label="the loyalty program",
       # Optional: panel + column names so auto diagnostics can run the
       # data-dependent checks (2x2 PT, Goodman-Bacon, EfficientDiD
       # Hausman). Without these the auto path still runs and just
       # skips those checks.
       data=df,
       outcome="revenue",
       unit="store",
       time="period",
       first_treat="first_treat",
   )
   print(report.summary())

API
---

.. autoclass:: diff_diff.BusinessReport
   :members:
   :show-inheritance:

.. autoclass:: diff_diff.BusinessContext
   :members:
   :show-inheritance:

.. autodata:: diff_diff.BUSINESS_REPORT_SCHEMA_VERSION
