Heterogeneous Adoption Difference-in-Differences
================================================

Estimator for designs where **no unit remains untreated** at the post period.
Every unit `g` is exposed to treatment at the same single date but adoption
intensity (dose) varies across units; there is no genuinely untreated control
group to anchor a standard DiD contrast.

This module implements the methodology from de Chaisemartin, Ciccia,
D'Haultfœuille & Knau (2026), "Difference-in-Differences Estimators When No
Unit Remains Untreated" (arXiv:2405.04465v6), which:

1. **Targets WAS or WAS_{d̲} depending on design path:** Design 1' (the
   QUG / Quasi-Untreated-Group case with ``d̲ = 0``) identifies the
   Weighted Average Slope (WAS, paper Equation 2); Design 1 (no QUG,
   ``d̲ > 0``) identifies ``WAS_{d̲}`` under Assumption 6, or sign
   identification only under Assumption 5 (neither additional assumption
   is testable via pre-trends). The shipped result classes expose
   ``target_parameter == "WAS"`` versus ``"WAS_d_lower"`` so callers can
   key on the resolved estimand.
2. **Estimates the target via local-linear regression at the dose support
   boundary**, with three concrete fit paths: ``continuous_at_zero`` for
   Design 1', and ``continuous_near_d_lower`` or ``mass_point`` for
   Design 1 (auto-detected from the dose distribution).
3. **Provides bias-corrected confidence intervals** ported from the
   ``nprobust`` machinery for the continuous-dose paths, and a
   structural-residual 2SLS sandwich for the mass-point path.
4. **Extends to multi-period event-study settings** (paper Appendix B.2),
   restricting staggered-timing panels to the last-treatment cohort (which
   retains never-treated units as comparisons) with pointwise per-horizon CIs.

.. note::

   **When to use HAD.** Use ``HeterogeneousAdoptionDiD`` when your panel has
   no untreated unit at the post period (e.g. universal-rollout policies,
   industry-wide tariff changes) but treatment intensity varies across
   units. For panels with a never-treated control group and continuous
   treatment, use :class:`~diff_diff.ContinuousDiD` instead. For binary
   reversible treatments, use :class:`~diff_diff.ChaisemartinDHaultfoeuille`.

.. note::

   **Inference contract.** Per-horizon CIs are always pointwise. There are
   three SE regimes selected by call site:

   - **Unweighted** - continuous paths use the CCT-2014 weighted-robust SE
     from the in-house ``lprobust`` port; the mass-point path uses a
     structural-residual 2SLS sandwich. No cross-horizon covariance.
   - **``survey_design=make_pweight_design(weights)``** (pweight-only
     shortcut) - continuous paths reuse the CCT-2014 SE; the mass-point
     path uses an analytical weighted 2SLS sandwich (``classical`` /
     ``hc1`` only - ``hc2`` / ``hc2_bm`` raise ``NotImplementedError``
     pending a 2SLS-specific leverage derivation).
   - **``survey_design=SurveyDesign(...)``** (full TSL with strata / PSU
     / FPC) - both paths compose Binder (1983) Taylor-series linearization
     with ``df_survey`` threaded into ``safe_inference``.

   The deprecated ``survey=`` and ``weights=`` aliases still resolve to
   the same paths with a ``DeprecationWarning`` (removal queued for the
   next minor release).

   A simultaneous confidence band (sup-t) is available only on the
   **weighted event-study path** via ``cband=True``. Joint cross-horizon
   analytical covariance is not computed in this release; tracked in
   ``TODO.md``.

   **Mass-point ``vcov_type="classical"`` deviation.** The mass-point
   ``survey_design=SurveyDesign(...)`` paths (static and event-study) and
   the ``survey_design=make_pweight_design(weights)`` +
   ``aggregate="event_study"`` + ``cband=True`` path reject
   ``vcov_type="classical"`` with ``NotImplementedError``. The per-unit
   2SLS influence function returned by the mass-point fit is HC1-scaled
   so that ``compute_survey_if_variance`` and the sup-t bootstrap target
   ``V_HC1`` consistently; mixing it with a classical analytical SE
   would silently report a ``V_HC1``-targeted variance under a
   ``classical`` label. Use ``vcov_type="hc1"`` (or leave it unset with
   the default ``robust=True`` mapping); a classical-aligned IF
   derivation is queued for a follow-up PR.

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

HAD Pretests
------------

Diagnostic pretests for the HAD identification assumptions from de Chaisemartin
et al. (2026). The composite orchestrator
:func:`~diff_diff.did_had_pretest_workflow` dispatches to two shapes based on
panel structure: the **overall** path (two-period first-differenced sample)
runs single-period tests; the **event-study** path (three or more periods)
runs joint multi-period tests. Both paths return a unified
:class:`~diff_diff.HADPretestReport`.

.. autofunction:: diff_diff.did_had_pretest_workflow

.. autoclass:: diff_diff.HADPretestReport
   :members:
   :undoc-members:
   :show-inheritance:

Single-period tests (``aggregate="overall"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: diff_diff.qug_test

.. autofunction:: diff_diff.stute_test

.. autofunction:: diff_diff.yatchew_hr_test

.. autoclass:: diff_diff.QUGTestResults
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: diff_diff.StuteTestResults
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: diff_diff.YatchewTestResults
   :members:
   :undoc-members:
   :show-inheritance:

Joint multi-period tests (``aggregate="event_study"``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: diff_diff.stute_joint_pretest

.. autofunction:: diff_diff.joint_pretrends_test

.. autofunction:: diff_diff.joint_homogeneity_test

.. autoclass:: diff_diff.StuteJointResult
   :members:
   :undoc-members:
   :show-inheritance:
