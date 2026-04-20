"""Shared helpers for BusinessReport and DiagnosticReport.

This module hosts per-estimator dispatch logic that both BR and DR
consume. It lives in its own module rather than in ``business_report``
or ``diagnostic_report`` to avoid a circular-import risk: BR already
imports from DR (for ``DiagnosticReportResults``), so placing shared
helpers here keeps the dependency graph tidy if DR ever needs to
reference a BR symbol.

Current contents:

- ``describe_target_parameter(results)`` — returns the
  ``target_parameter`` block documenting what scalar the headline
  represents. Introduced for BR/DR gap #6 (target-parameter
  clarity); see ``docs/methodology/REPORTING.md`` and per-estimator
  entries in ``docs/methodology/REGISTRY.md`` for the canonical
  wording.
"""

from __future__ import annotations

from typing import Any, Dict


def describe_target_parameter(results: Any) -> Dict[str, Any]:
    """Return the target-parameter block for a fitted result.

    The block names what scalar the headline number actually
    represents (overall ATT, DID_M, dose-response ATT(d|d), etc.) so
    BR/DR output is self-contained. Baker et al. (2025) Step 2 is
    "Define the target parameter"; this helper does that work for the
    user.

    Shape::

        {
            "name": str,                # stakeholder-facing name
            "definition": str,          # plain-English description
            "aggregation": str,         # machine-readable tag
            "headline_attribute": str,  # which raw attribute the scalar comes from
            "reference": str,           # citation pointer
        }

    Each per-estimator branch cites REGISTRY.md as the single source
    of truth for the canonical wording (per
    ``feedback_verify_claims.md``). All wording choices are
    deliberate:

    - ``ImputationDiD`` / ``TwoStageDiD``: the ``aggregate`` fit-time
      kwarg controls which horizon / group tables get populated but
      does NOT change ``overall_att``. The headline is always the
      sample-mean overall ATT (per BJS 2024 Step 3 with
      ``w_it = 1/N_1``); disambiguate via the event-study or group
      aggregate if you need the horizon / group target.
    - ``CallawaySantAnna``: ``overall_att`` is cohort-size-weighted
      across post-treatment ``ATT(g, t)`` cells regardless of the
      fit-time ``aggregate`` kwarg. The event-study / group
      aggregations live on dedicated fields
      (``event_study_effects`` / ``group_effects``).
    - ``ContinuousDiD``: the regime (PT vs. SPT) is a user-level
      assumption, not a library setting. The ``definition`` names
      both regime readings (``ATT^loc`` under PT,
      ``ATT^glob`` under SPT) so the user can pick the
      interpretation that matches their assumption.
    - ``WooldridgeDiD``: ``overall_att`` reports the observation-
      count-weighted ASF-based ATT across cohort x time cells.
      Calling ``.aggregate("event")`` populates additional event-
      study tables but does NOT change the headline scalar.
    """
    name = type(results).__name__

    if name == "DiDResults":
        # Covers both ``DifferenceInDifferences`` (2x2 DiD) and
        # ``TwoWayFixedEffects`` (TWFE with unit + time FE). Both
        # estimators return ``DiDResults``; there is no separate
        # ``TwoWayFixedEffectsResults`` class as of this PR
        # (confirmed in PR #347 R1 review). The description covers
        # both interpretations because the result carries no
        # estimator-provenance marker BR/DR can dispatch on. Adding a
        # dedicated TWFE result class (or persisting provenance on
        # DiDResults) is queued as follow-up so this branch can split
        # in a future PR.
        return {
            "name": "ATT (2x2 or TWFE within-transformed coefficient)",
            "definition": (
                "The average treatment effect on the treated. For "
                "``DifferenceInDifferences``, this is the 2x2 DiD "
                "contrast between treated-unit change and control-unit "
                "change across pre / post. For ``TwoWayFixedEffects``, "
                "this is the coefficient on the treatment-by-post "
                "interaction in a regression with unit and time fixed "
                "effects; under homogeneous treatment effects it is "
                "the ATT, and under heterogeneous effects with staggered "
                "adoption it is a weighted average of 2x2 comparisons "
                "that may include forbidden later-vs-earlier comparisons "
                "(see Goodman-Bacon)."
            ),
            "aggregation": "2x2",
            "headline_attribute": "att",
            "reference": ("REGISTRY.md Sec. DifferenceInDifferences / TwoWayFixedEffects"),
        }

    if name == "MultiPeriodDiDResults":
        return {
            "name": "average ATT over post-treatment periods (event-study average)",
            "definition": (
                "The equally-weighted average of the post-treatment event-study "
                "coefficients. Each coefficient is an ATT at a single event time "
                "relative to treatment onset; the headline averages across "
                "post-treatment horizons."
            ),
            "aggregation": "event_study",
            "headline_attribute": "avg_att",
            "reference": "REGISTRY.md Sec. MultiPeriodDiD",
        }

    if name == "CallawaySantAnnaResults":
        return {
            "name": "overall ATT (cohort-size-weighted average of ATT(g,t))",
            "definition": (
                "A cohort-size-weighted average of group-time ATTs "
                "``ATT(g, t)`` across post-treatment cells (``t >= g``). "
                "``overall_att`` is the simple-aggregation headline regardless "
                "of the fit-time ``aggregate`` kwarg; event-study and group "
                "aggregations populate ``event_study_effects`` / "
                "``group_effects`` fields when requested."
            ),
            "aggregation": "simple",
            "headline_attribute": "overall_att",
            "reference": "Callaway & Sant'Anna (2021); REGISTRY.md Sec. CallawaySantAnna",
        }

    if name == "SunAbrahamResults":
        return {
            "name": "overall ATT (interaction-weighted average of CATT(g, e))",
            "definition": (
                "An interaction-weighted (IW) average of cohort-specific "
                "ATT(g, e) effects estimated via saturated cohort-by-event-time "
                "interactions. Weights are the sample shares of each cohort at "
                "each event time; the headline is the resulting overall ATT."
            ),
            "aggregation": "iw",
            "headline_attribute": "overall_att",
            "reference": "Sun & Abraham (2021); REGISTRY.md Sec. SunAbraham",
        }

    if name == "ImputationDiDResults":
        return {
            "name": "overall ATT (sample-mean imputation average)",
            "definition": (
                "The sample-mean overall ATT ``tau_hat_w = (1/N_1) * sum "
                "tau_hat_it`` across treated observations, where "
                "``tau_hat_it = Y_it - Y_hat_it(0)`` and ``Y_hat_it(0)`` is "
                "imputed from a unit+time fixed-effects model fitted on "
                "untreated observations only (BJS 2024 Step 3). The fit-time "
                "``aggregate`` kwarg populates additional horizon / group "
                "tables but does NOT change ``overall_att`` — for the "
                "horizon or group estimand, consult the event-study or group "
                "aggregate directly."
            ),
            "aggregation": "simple",
            "headline_attribute": "overall_att",
            "reference": "Borusyak-Jaravel-Spiess (2024); REGISTRY.md Sec. ImputationDiD",
        }

    if name == "TwoStageDiDResults":
        return {
            "name": "overall ATT (two-stage residualization average)",
            "definition": (
                "The sample-mean overall ATT recovered from Stage 2 of "
                "Gardner's two-stage procedure: Stage 1 fits unit+time fixed "
                "effects on untreated observations only, and Stage 2 regresses "
                "the residualized outcome on the treatment indicator across "
                "treated observations. Point estimate is algebraically "
                "equivalent to Borusyak-Jaravel-Spiess imputation. As with "
                "ImputationDiD, the fit-time ``aggregate`` kwarg populates "
                "additional tables but does NOT change ``overall_att``."
            ),
            "aggregation": "simple",
            "headline_attribute": "overall_att",
            "reference": "Gardner (2022); REGISTRY.md Sec. TwoStageDiD",
        }

    if name == "StackedDiDResults":
        clean_control = getattr(results, "clean_control", None)
        if clean_control == "never_treated":
            control_clause = "Controls are the never-treated units (``A_s = infinity``)."
        elif clean_control == "strict":
            control_clause = (
                "Controls for each sub-experiment ``a`` are units strictly "
                "untreated across the full pre/post event window "
                "(``A_s > a + kappa_post + kappa_pre``)."
            )
        else:  # "not_yet_treated" or unknown -> default rule
            control_clause = (
                "Controls for each sub-experiment ``a`` are units not yet "
                "treated through the event's post-window "
                "(``A_s > a + kappa_post``)."
            )
        return {
            "name": "overall ATT (average of post-treatment event-study coefficients)",
            "definition": (
                "The average of post-treatment event-study coefficients "
                "``delta_h`` (h >= -anticipation), estimated from the stacked "
                "sub-experiment panel with delta-method SE "
                "(``stacked_did.py`` around line 541). Each sub-experiment "
                "aligns a treated cohort with its clean-control set over the "
                "event window ``[-kappa_pre, +kappa_post]``; each per-horizon "
                "``delta_h`` is the paper's ``theta_kappa^e`` "
                "treated-share-weighted cross-event aggregate. The "
                "``overall_att`` headline is the equally-weighted average of "
                "these per-horizon coefficients, not a separate cross-event "
                "weighted aggregate at the ATT level. " + control_clause
            ),
            "aggregation": "stacked",
            "headline_attribute": "overall_att",
            "reference": "Wing-Freedman-Hollingsworth (2024); REGISTRY.md Sec. StackedDiD",
        }

    if name == "WooldridgeDiDResults":
        return {
            "name": "overall ATT (observation-count-weighted ASF ATT across cohort x time cells)",
            "definition": (
                "The overall ATT under Wooldridge's ETWFE: the average-structural-"
                "function (ASF) contrast between treated and counterfactual "
                "untreated outcomes, averaged across cohort x time cells with "
                'observation-count weights. Calling ``.aggregate("event")`` '
                "populates additional event-study tables but does NOT change "
                "the ``overall_att`` scalar."
            ),
            "aggregation": "simple",
            "headline_attribute": "overall_att",
            "reference": "Wooldridge (2023); REGISTRY.md Sec. WooldridgeDiD",
        }

    if name == "EfficientDiDResults":
        pt_assumption = getattr(results, "pt_assumption", "all")
        if pt_assumption == "post":
            return {
                "name": "overall ATT under PT-Post (single-baseline)",
                "definition": (
                    "The overall ATT identified under the weaker PT-Post "
                    "regime (parallel trends hold only in post-treatment "
                    "periods). The baseline is period ``g - 1`` only; the "
                    "estimator is just-identified and reduces to standard "
                    "single-baseline DiD (Corollary 3.2)."
                ),
                "aggregation": "pt_post_single_baseline",
                "headline_attribute": "overall_att",
                "reference": "Chen-Sant'Anna-Xie (2025) Cor. 3.2; REGISTRY.md Sec. EfficientDiD",
            }
        return {
            "name": "overall ATT under PT-All (over-identified combined)",
            "definition": (
                "The overall ATT identified under the stronger PT-All regime "
                "(parallel trends hold for all groups and all periods). The "
                "estimator is over-identified (Lemma 2.1) and applies "
                "optimal-combination weights to achieve the semiparametric "
                "efficiency bound on the no-covariate path."
            ),
            "aggregation": "pt_all_combined",
            "headline_attribute": "overall_att",
            "reference": "Chen-Sant'Anna-Xie (2025) Lemma 2.1; REGISTRY.md Sec. EfficientDiD",
        }

    if name == "ContinuousDiDResults":
        return {
            "name": "overall ATT (dose-aggregated)",
            "definition": (
                "The overall ATT aggregated across treated dose levels. Under "
                "Parallel Trends (PT) this identifies ``ATT^loc`` — the "
                "binarized ATT for treated-vs-untreated; under Strong Parallel "
                "Trends (SPT) it identifies ``ATT^glob`` — the population "
                "average ATT across dose levels. The regime is a user-level "
                "assumption, not a library setting; the dose-response curve "
                "``ATT(d)`` / marginal effect ``ACRT(d)`` / per-dose "
                "``ATT(d|d)`` are available on the result object for "
                "dose-specific inference."
            ),
            "aggregation": "dose_overall",
            "headline_attribute": "overall_att",
            "reference": "Callaway-Goodman-Bacon-Sant'Anna (2024); REGISTRY.md Sec. ContinuousDiD",
        }

    if name == "TripleDifferenceResults":
        return {
            "name": "ATT (triple-difference)",
            "definition": (
                "The ATT identified via the DDD cancellation "
                "``DDD = DiD_A + DiD_B - DiD_C`` across the Group x Period x "
                "Eligibility cells. Differences out group-specific and period-"
                "specific unobservables without requiring separate parallel-"
                "trends assumptions between each cell pair."
            ),
            "aggregation": "ddd",
            "headline_attribute": "att",
            "reference": "Ortiz-Villavicencio & Sant'Anna (2025); REGISTRY.md Sec. TripleDifference",
        }

    if name == "StaggeredTripleDiffResults":
        return {
            "name": "overall ATT (cohort-weighted staggered triple-difference)",
            "definition": (
                "A cohort-weighted aggregate of per-(g, g_c, t) triple-"
                "difference ATTs ``DDD(g, g_c, t)`` via the GMM optimal-"
                "combination weighting. Extends the DDD cancellation to "
                "staggered adoption timing with a third (eligibility) "
                "dimension."
            ),
            "aggregation": "staggered_ddd",
            "headline_attribute": "overall_att",
            "reference": "Ortiz-Villavicencio & Sant'Anna (2025); REGISTRY.md Sec. StaggeredTripleDifference",
        }

    if name == "ChaisemartinDHaultfoeuilleResults":
        l_max = getattr(results, "L_max", None)
        has_controls = getattr(results, "covariate_residuals", None) is not None
        has_trends = getattr(results, "linear_trends_effects", None) is not None
        if l_max is None:
            # DID_M — period-aggregated contemporaneous-switch ATT.
            return {
                "name": "DID_M (period-aggregated contemporaneous-switch ATT)",
                "definition": (
                    "The contemporaneous-switch ATT averaged across switching "
                    "periods. At each period the estimator contrasts joiners "
                    "(``D: 0 -> 1``), leavers (``D: 1 -> 0``), and stable-"
                    "control cells that share the same treatment state across "
                    "adjacent periods; ``DID_M`` averages these per-period "
                    "contrasts."
                ),
                "aggregation": "M",
                "headline_attribute": "overall_att",
                "reference": (
                    "de Chaisemartin & D'Haultfoeuille (2020); "
                    "REGISTRY.md Sec. ChaisemartinDHaultfoeuille"
                ),
            }
        # L_max >= 1 — dynamic horizon estimand.
        if has_controls and has_trends:
            agg_tag = "l_x_fd"
            headline_name = "DID^{X,fd}_l (covariate-residualized first-differences)"
            extra = (
                " Identification holds conditional on the covariates entering "
                "the first-stage residualization and allowing group-specific "
                "linear trends."
            )
        elif has_controls:
            agg_tag = "l_x"
            headline_name = "DID^X_l (covariate-residualized per-horizon ATT)"
            extra = (
                " Identification holds conditional on the covariates entering "
                "the first-stage residualization."
            )
        elif has_trends:
            agg_tag = "l_fd"
            headline_name = "DID^{fd}_l (first-differenced per-horizon ATT)"
            extra = " The identifying restriction allows group-specific linear " "pre-trends."
        else:
            agg_tag = "l"
            headline_name = "DID_l (per-horizon dynamic ATT)"
            extra = ""
        return {
            "name": headline_name,
            "definition": (
                "A cohort-averaged dynamic ATT at event horizon ``l`` "
                "post-switch. The estimator contrasts joiners and stable "
                "controls that share baseline treatment state at the switching "
                "period; the aggregate averages across cohorts with treated-"
                "share weights." + extra
            ),
            "aggregation": agg_tag,
            "headline_attribute": "overall_att",
            "reference": (
                "de Chaisemartin & D'Haultfoeuille (2020, 2024); "
                "REGISTRY.md Sec. ChaisemartinDHaultfoeuille"
            ),
        }

    if name == "SyntheticDiDResults":
        return {
            "name": "synthetic-DiD ATT (time- and unit-weighted synthetic-control comparison)",
            "definition": (
                "The treatment-effect contrast ``tau_hat^{sdid} = sum_t "
                "lambda_t (Y_bar_{tr, t} - sum_j omega_j Y_{j, t})`` between "
                "the treated group and a synthetic-control composed of "
                "unit-weighted donors (``omega_j``) aggregated with time-"
                "weighted pre-period matching weights (``lambda_t``). "
                "Identification is through the weighted parallel-trends "
                "analogue rather than unconditional PT."
            ),
            "aggregation": "synthetic",
            "headline_attribute": "att",
            "reference": "Arkhangelsky et al. (2021); REGISTRY.md Sec. SyntheticDiD",
        }

    if name == "BaconDecompositionResults":
        # BaconDecompositionResults is a diagnostic, not an estimator:
        # BR refuses it (raises TypeError), but DR accepts it as a
        # read-out and the schema needs a target-parameter block.
        # The "target parameter" of a Bacon decomposition is the TWFE
        # coefficient it is decomposing — named here for schema
        # completeness.
        return {
            "name": "TWFE ATT being decomposed",
            "definition": (
                "Goodman-Bacon decomposition is a diagnostic of the TWFE "
                "coefficient ``twfe_estimate``: it partitions TWFE weights "
                "across treated-vs-never, earlier-vs-later, and "
                "later-vs-earlier (forbidden) 2x2 comparisons. The scalar "
                "``twfe_estimate`` is the weighted sum of these 2x2 DiDs."
            ),
            "aggregation": "twfe",
            "headline_attribute": "twfe_estimate",
            "reference": "Goodman-Bacon (2021); REGISTRY.md Sec. BaconDecomposition",
        }

    if name == "TROPResults":
        return {
            "name": "TROP ATT (factor-model-adjusted weighted average)",
            "definition": (
                "A weighted average of per-(unit, time) treatment effects "
                "``tau_hat_{it}(lambda_hat)`` estimated under low-rank factor-"
                "model identification. Unobserved heterogeneity is captured "
                "through latent factor loadings rather than a parallel-trends "
                "assumption; the aggregate averages across treated cells with "
                "weights ``W_{it}``."
            ),
            "aggregation": "factor_model",
            "headline_attribute": "att",
            "reference": "REGISTRY.md Sec. TROP",
        }

    # Default: unrecognized result class. Fall through with a neutral
    # block — agents / downstream consumers can still dispatch on
    # ``aggregation="unknown"`` and fall back to generic ATT narration.
    return {
        "name": "ATT (unrecognized result class)",
        "definition": (
            f"The target parameter for ``{name}`` is not specifically "
            "documented in BR/DR. Defaulting to generic ATT narration; see "
            "the estimator's own documentation for the canonical estimand."
        ),
        "aggregation": "unknown",
        "headline_attribute": "att",
        "reference": "REGISTRY.md",
    }
