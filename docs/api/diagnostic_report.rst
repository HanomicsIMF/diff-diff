DiagnosticReport
================

``DiagnosticReport`` orchestrates the library's existing diagnostic
functions (parallel trends, pre-trends power, HonestDiD sensitivity,
Goodman-Bacon, design-effect, EPV, heterogeneity, and estimator-native
checks for SyntheticDiD and TROP) into a single report with a stable
AI-legible schema.

Construction is free; ``run_all()`` triggers the compute and caches.
A second call to ``to_dict()`` or ``summary()`` reuses the cached
result.

Methodology deviations (no traffic-light gates, opt-in placebo
battery, estimator-native diagnostic routing, power-aware phrasing
threshold) are documented in :doc:`../methodology/REPORTING`.

Example
-------

.. code-block:: python

   from diff_diff import CallawaySantAnna, DiagnosticReport

   cs = CallawaySantAnna().fit(
       df, outcome="outcome", unit="unit", time="period",
       first_treat="first_treat", aggregate="event_study",
   )
   dr = DiagnosticReport(
       cs,
       data=df,
       outcome="outcome",
       unit="unit",
       time="period",
       first_treat="first_treat",
   )
   print(dr.summary())
   dr.to_dataframe()  # one row per check

API
---

.. autoclass:: diff_diff.DiagnosticReport
   :members:
   :show-inheritance:

.. autoclass:: diff_diff.DiagnosticReportResults
   :members:
   :show-inheritance:

.. autodata:: diff_diff.DIAGNOSTIC_REPORT_SCHEMA_VERSION
