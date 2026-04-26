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
   - **``weights=np.ndarray`` shortcut (deprecated)** - continuous paths
     reuse the CCT-2014 SE; the mass-point path uses an analytical
     weighted 2SLS sandwich (``classical`` / ``hc1``; CR1 when
     ``cluster=`` is supplied, except ``cluster=`` +
     ``aggregate="event_study"`` + ``cband=True`` is rejected outright
     regardless of ``vcov_type`` per the cluster-combination deviation
     below; ``hc2`` / ``hc2_bm`` raise ``NotImplementedError`` pending a
     2SLS-specific leverage derivation). Yields
     ``variance_formula="pweight"`` / ``"pweight_2sls"``.
   - **``survey_design=SurveyDesign(weights="col", ...)``** (canonical;
     accepts strata / PSU / FPC) - both paths compose Binder (1983)
     Taylor-series linearization with ``df_survey`` threaded into
     ``safe_inference``. Yields ``variance_formula="survey_binder_tsl"``
     / ``"survey_binder_tsl_2sls"``.

   The two weighted paths currently produce different SE families on this
   estimator (CCT-2014 / 2SLS pweight-sandwich vs Binder-TSL); the
   deprecated ``weights=`` and ``survey=`` aliases will be removed in the
   next minor release, at which point the long-term unification onto a
   single SE contract under ``survey_design=`` lands. (Tracked in
   ``TODO.md``; the deprecation warning emitted by ``HeterogeneousAdoptionDiD.fit``
   spells the migration out per call site.) On array-in HAD pretest
   helpers (``stute_test``, ``yatchew_hr_test``, ``stute_joint_pretest``)
   the pweight-only shortcut is
   ``survey_design=make_pweight_design(weights)``; data-in surfaces use
   ``survey_design=SurveyDesign(weights="col_name", ...)`` against
   ``data`` instead. ``qug_test`` is the exception: the QUG step has no
   survey-aware migration target (Phase 4.5 C0 decision; see methodology
   REGISTRY) and permanently raises ``NotImplementedError`` on any of
   ``survey_design=`` / ``survey=`` / ``weights=``. The composite
   workflow ``did_had_pretest_workflow`` handles this by skipping QUG
   under survey/weighted dispatch and emitting a ``UserWarning``.

   A simultaneous confidence band (sup-t) is available only on the
   **weighted event-study path** via ``cband=True``. Joint cross-horizon
   analytical covariance is not computed in this release; tracked in
   ``TODO.md``.

   **Mass-point ``vcov_type="classical"`` deviation.** The mass-point
   ``survey_design=SurveyDesign(...)`` paths (static and event-study) and
   the deprecated ``weights=`` + ``aggregate="event_study"`` +
   ``cband=True`` path reject ``vcov_type="classical"`` with
   ``NotImplementedError``. The per-unit 2SLS influence function returned
   by the mass-point fit is HC1-scaled so that
   ``compute_survey_if_variance`` and the sup-t bootstrap target
   ``V_HC1`` consistently; mixing it with a classical analytical SE
   would silently report a ``V_HC1``-targeted variance under a
   ``classical`` label. Use ``vcov_type="hc1"`` or set ``robust=True``
   explicitly (the constructor default ``robust=False`` maps to
   ``vcov_type="classical"``, which triggers the guard); a
   classical-aligned IF derivation is queued for a follow-up PR.

   **Mass-point cluster-combination deviation.** On
   ``design="mass_point"``, two clustered weighted paths are rejected
   outright regardless of ``vcov_type``:

   - ``survey_design=SurveyDesign(...)`` + ``cluster=`` (static and
     event-study): the survey path composes Binder-TSL variance, which
     would silently override the CR1 cluster-robust sandwich.
     Workarounds: ``cluster=`` alone (unweighted CR1), or ``weights=``
     + ``cluster=`` (weighted-CR1 pweight sandwich), or
     ``survey_design=`` alone (Binder-TSL). Combined cluster-robust +
     survey inference is queued for a follow-up PR.
   - Deprecated ``weights=`` shortcut + ``cluster=`` +
     ``aggregate="event_study"`` + ``cband=True``: the sup-t bootstrap
     normalizes HC1-scale perturbations by the CR1 analytical SE,
     mixing variance families. Workarounds: pass ``cband=False`` (keeps
     weighted-CR1 per-horizon), or drop ``cluster=`` (keeps
     weighted-HC1 sup-t).

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
:func:`~diff_diff.did_had_pretest_workflow` is a diagnostic battery only - it
does NOT pick the HAD design path (``continuous_at_zero`` /
``continuous_near_d_lower`` / ``mass_point``); that is auto-detected inside
:meth:`HeterogeneousAdoptionDiD.fit` from the dose support. The workflow has
two explicit modes selected by the caller via the ``aggregate=`` kwarg:
``aggregate="overall"`` (default, two-period first-differenced sample) runs
single-period tests; ``aggregate="event_study"`` (multi-period panel with
three or more periods) runs joint multi-period tests. Both modes return a
unified :class:`~diff_diff.HADPretestReport`.

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
