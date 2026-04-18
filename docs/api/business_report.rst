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
