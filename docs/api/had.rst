Heterogeneous Adoption Difference-in-Differences
================================================

Estimator for designs where **no unit remains untreated** at the post period
- every unit `g` receives a strictly positive heterogeneous dose `D_{g,2} > 0`,
and there is no genuinely untreated control group to anchor a standard DiD
contrast.

This module implements the methodology from de Chaisemartin, Ciccia,
D'Haultfœuille & Knau (2026), "Difference-in-Differences Estimators When No
Unit Remains Untreated" (arXiv:2405.04465v6), which:

1. **Targets the Weighted Average Slope (WAS)** as the identified parameter
   when no untreated comparison group exists (paper Equation 2).
2. **Uses local-linear regression at the support boundary** to estimate the
   slope of the dose-outcome relationship at the lowest observed dose.
3. **Provides bias-corrected confidence intervals** (Calonico, Cattaneo & Titiunik
   2014 / `nprobust`-style) and HC2 / Bell-McCaffrey small-sample SEs.
4. **Extends to multi-period event-study settings** (paper Appendix B.2),
   producing per-horizon WAS effects with correlated standard errors and
   sup-t bands.

.. note::

   **When to use HAD.** Use ``HeterogeneousAdoptionDiD`` when your panel has
   no untreated unit at any treatment period (e.g. universal-rollout policies,
   industry-wide tariff changes) but treatment intensity varies across units.
   For panels with a never-treated control group and continuous treatment,
   use :class:`~diff_diff.ContinuousDiD` instead. For binary reversible
   treatments, use :class:`~diff_diff.ChaisemartinDHaultfoeuille`.

HeterogeneousAdoptionDiD
------------------------

.. autoclass:: diff_diff.HeterogeneousAdoptionDiD
   :members:
   :undoc-members:
   :show-inheritance:

HeterogeneousAdoptionDiDResults
-------------------------------

Single-period results container for ``HeterogeneousAdoptionDiD`` estimation.

.. autoclass:: diff_diff.HeterogeneousAdoptionDiDResults
   :members:
   :undoc-members:
   :show-inheritance:

HeterogeneousAdoptionDiDEventStudyResults
-----------------------------------------

Multi-period event-study results container for the Appendix B.2 extension.

.. autoclass:: diff_diff.HeterogeneousAdoptionDiDEventStudyResults
   :members:
   :undoc-members:
   :show-inheritance:
