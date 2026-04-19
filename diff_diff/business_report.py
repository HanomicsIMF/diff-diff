"""
BusinessReport — plain-English stakeholder narrative from any diff-diff result.

Wraps any of the 16 fitted result types and produces:

- ``summary()``: a short paragraph block suitable for an email or Slack message.
- ``full_report()``: a multi-section markdown report with headline, assumptions,
  pre-trends, main result, robustness, sample, and an optional academic appendix.
- ``to_dict()``: a stable AI-legible structured schema (single source of truth —
  prose is rendered from this dict, not templated alongside it).

Design principles:

- Plain English, not academic jargon. The library ships this in addition to, not
  in place of, the estimator's existing ``results.summary()`` academic output.
- No estimator fitting and no variance re-derivation. Every effect, SE, p-value,
  CI, and sensitivity bound is either read from ``results`` or produced by an
  existing diff-diff utility. The report layer does compose a few cross-period
  summaries from per-period inputs already on the result (joint-Wald / Bonferroni
  pre-trends p-value, MDV-to-ATT ratio, heterogeneity dispersion over
  post-treatment effects); see ``docs/methodology/REPORTING.md`` for the full
  enumeration.
- Optional business context via keyword args (``outcome_label``, ``outcome_unit``,
  ``business_question``, ``treatment_label``). Without them, BusinessReport uses
  generic fallbacks — the zero-config path works.
- Diagnostic integration is implicit by default: ``BusinessReport(results)``
  auto-constructs a ``DiagnosticReport`` so the summary can mention pre-trends,
  robustness, and design-effect findings. Pass ``auto_diagnostics=False`` or an
  explicit ``diagnostics=`` object to override.

Methodology deviations (no traffic-light gates, pre-trends verdict thresholds,
power-aware phrasing, unit-translation policy, schema stability) are documented
in ``docs/methodology/REPORTING.md``. The ``to_dict()`` schema is marked
experimental in v3.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from diff_diff.diagnostic_report import DiagnosticReport, DiagnosticReportResults

BUSINESS_REPORT_SCHEMA_VERSION = "1.0"

__all__ = [
    "BusinessReport",
    "BusinessContext",
    "BUSINESS_REPORT_SCHEMA_VERSION",
]

# Recognized ``outcome_unit`` values mapped to a coarse "kind" used by the
# formatter. Unrecognized strings are accepted and rendered verbatim without
# arithmetic translation (``unit_kind = "unknown"``).
_UNIT_KINDS: Dict[str, str] = {
    "$": "currency",
    "usd": "currency",
    "%": "percent",
    "pp": "percentage_points",
    "percentage_points": "percentage_points",
    "percent": "percent",
    "log_points": "log_points",
    "log": "log_points",
    "count": "count",
    "users": "count",
}


@dataclass(frozen=True)
class BusinessContext:
    """Frozen bundle of business-framing metadata used when rendering prose.

    Populated from ``BusinessReport`` constructor kwargs. Falls back to
    neutral labels when fields are not supplied.
    """

    outcome_label: str
    outcome_unit: Optional[str]
    outcome_direction: Optional[str]
    business_question: Optional[str]
    treatment_label: str
    alpha: float


class BusinessReport:
    """Produce a stakeholder-ready narrative from any diff-diff results object.

    Parameters
    ----------
    results : Any
        A fitted diff-diff results object. Any of the 16 result types is
        accepted. ``BaconDecompositionResults`` is not a valid input — Bacon
        is a diagnostic, not an estimator; use ``DiagnosticReport`` for that.
    outcome_label : str, optional
        Stakeholder-friendly outcome name (e.g. ``"Revenue per user"``).
    outcome_unit : str, optional
        Unit label: ``"$"`` / ``"%"`` / ``"pp"`` / ``"log_points"`` / ``"count"``
        (recognized for formatting) or any free-form string (used verbatim
        without arithmetic translation).
    outcome_direction : str, optional
        ``"higher_is_better"`` or ``"lower_is_better"``. Drives whether the
        effect is described as "lift" / "drag" rather than just "increase" /
        "decrease".
    business_question : str, optional
        Question the analysis answers (prepended to the summary).
    treatment_label : str, optional
        Stakeholder-friendly treatment name (e.g. ``"the campaign"``).
    alpha : float, optional
        Significance level. Defaults to ``results.alpha`` when not supplied.
        Single knob: drives both CI level and significance phrasing.
    honest_did_results : HonestDiDResults or SensitivityResults, optional
        Pre-computed sensitivity result. When supplied, this is forwarded to
        the internal ``DiagnosticReport`` so sensitivity is not re-computed.
    auto_diagnostics : bool, default True
        When ``True`` and ``diagnostics`` is ``None``, auto-construct a
        ``DiagnosticReport``. Set ``False`` to skip diagnostics entirely.
    diagnostics : DiagnosticReport or DiagnosticReportResults, optional
        Explicit diagnostics object. Takes precedence over ``auto_diagnostics``.
    include_appendix : bool, default True
        Whether ``full_report()`` appends the estimator's academic
        ``results.summary()`` output under a "Technical Appendix" section.
    """

    def __init__(
        self,
        results: Any,
        *,
        outcome_label: Optional[str] = None,
        outcome_unit: Optional[str] = None,
        outcome_direction: Optional[str] = None,
        business_question: Optional[str] = None,
        treatment_label: Optional[str] = None,
        alpha: Optional[float] = None,
        honest_did_results: Optional[Any] = None,
        auto_diagnostics: bool = True,
        diagnostics: Optional[Union[DiagnosticReport, DiagnosticReportResults]] = None,
        include_appendix: bool = True,
        data: Optional[Any] = None,
        outcome: Optional[str] = None,
        treatment: Optional[str] = None,
        unit: Optional[str] = None,
        time: Optional[str] = None,
        first_treat: Optional[str] = None,
    ):
        if type(results).__name__ == "BaconDecompositionResults":
            raise TypeError(
                "BaconDecompositionResults is a diagnostic, not an estimator; "
                "wrap the underlying estimator with BusinessReport and pass the "
                "Bacon object to DiagnosticReport(precomputed={'bacon': ...})."
            )

        if diagnostics is not None and not isinstance(
            diagnostics, (DiagnosticReport, DiagnosticReportResults)
        ):
            raise TypeError(
                "diagnostics= must be a DiagnosticReport or "
                "DiagnosticReportResults instance; "
                f"got {type(diagnostics).__name__}."
            )

        self._results = results
        self._honest_did_results = honest_did_results
        self._auto_diagnostics = auto_diagnostics
        self._diagnostics_arg = diagnostics
        self._include_appendix = include_appendix
        # Raw-data passthrough so the auto-constructed DR can run
        # data-dependent checks (2x2 PT on simple DiD, Bacon-from-
        # scratch on staggered estimators, EfficientDiD Hausman
        # pretest). Without these, the auto path silently skips those
        # checks (round-12 CI review on PR #318).
        self._dr_data = data
        self._dr_outcome = outcome
        self._dr_treatment = treatment
        self._dr_unit = unit
        self._dr_time = time
        self._dr_first_treat = first_treat

        resolved_alpha = alpha if alpha is not None else getattr(results, "alpha", 0.05)
        self._context = BusinessContext(
            outcome_label=outcome_label or "the outcome",
            outcome_unit=outcome_unit,
            outcome_direction=outcome_direction,
            business_question=business_question,
            treatment_label=treatment_label or "the treatment",
            alpha=float(resolved_alpha),
        )

        self._cached_schema: Optional[Dict[str, Any]] = None

    # -- Public API ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the AI-legible structured schema (single source of truth)."""
        if self._cached_schema is None:
            self._cached_schema = self._build_schema()
        return self._cached_schema

    def to_json(self, *, indent: int = 2) -> str:
        """Return ``to_dict()`` serialized as JSON."""
        import json

        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Return a short plain-English paragraph block (6-10 sentences)."""
        return _render_summary(self.to_dict())

    def full_report(self) -> str:
        """Return a structured multi-section markdown report."""
        base = _render_full_report(self.to_dict())
        if self._include_appendix:
            try:
                appendix = self._results.summary()
            except Exception:  # noqa: BLE001
                appendix = None
            if appendix:
                base = base + "\n\n## Technical Appendix\n\n```\n" + str(appendix) + "\n```\n"
        return base

    def export_markdown(self) -> str:
        """Alias for ``full_report()`` (discoverability)."""
        return self.full_report()

    def headline(self) -> str:
        """Return just the headline sentence."""
        return _render_headline_sentence(self.to_dict())

    def caveats(self) -> List[Dict[str, str]]:
        """Return the list of structured caveats (severity + topic + message)."""
        return list(self.to_dict().get("caveats", []))

    def __repr__(self) -> str:
        estimator = type(self._results).__name__
        headline = self.to_dict().get("headline") or {}
        val = headline.get("effect")
        if isinstance(val, (int, float)) and np.isfinite(val):
            return f"BusinessReport(results={estimator}, effect={val:.3g})"
        return f"BusinessReport(results={estimator})"

    def __str__(self) -> str:
        return self.summary()

    # -- Implementation detail ---------------------------------------------

    def _resolve_diagnostics(self) -> Optional[DiagnosticReportResults]:
        """Return the DiagnosticReportResults to embed, or ``None`` if skipped."""
        if self._diagnostics_arg is not None:
            if isinstance(self._diagnostics_arg, DiagnosticReportResults):
                return self._diagnostics_arg
            if isinstance(self._diagnostics_arg, DiagnosticReport):
                return self._diagnostics_arg.run_all()
            raise TypeError("diagnostics= must be a DiagnosticReport or DiagnosticReportResults")
        if not self._auto_diagnostics:
            return None
        precomputed: Dict[str, Any] = {}
        if self._honest_did_results is not None:
            precomputed["sensitivity"] = self._honest_did_results
        dr = DiagnosticReport(
            self._results,
            alpha=self._context.alpha,
            precomputed=precomputed or None,
            outcome_label=self._context.outcome_label,
            treatment_label=self._context.treatment_label,
            data=self._dr_data,
            outcome=self._dr_outcome,
            treatment=self._dr_treatment,
            unit=self._dr_unit,
            time=self._dr_time,
            first_treat=self._dr_first_treat,
        )
        return dr.run_all()

    def _build_schema(self) -> Dict[str, Any]:
        """Assemble the structured schema.

        Pulls validation content (PT, sensitivity, Bacon, DEFF, EPV, ...) from
        the internal ``DiagnosticReport``; extracts the stakeholder-facing
        headline and sample metadata from the fitted result itself.
        """
        estimator_name = type(self._results).__name__
        diagnostics_results = self._resolve_diagnostics()
        dr_schema: Optional[Dict[str, Any]] = (
            diagnostics_results.schema if diagnostics_results is not None else None
        )

        headline = self._extract_headline(dr_schema)
        sample = self._extract_sample()
        heterogeneity = _lift_heterogeneity(dr_schema)
        pre_trends = _lift_pre_trends(dr_schema)
        sensitivity = _lift_sensitivity(dr_schema)
        robustness = _lift_robustness(dr_schema)
        assumption = _describe_assumption(estimator_name, self._results)
        next_steps = (dr_schema or {}).get("next_steps", [])
        caveats = _build_caveats(self._results, headline, sample, dr_schema)
        references = _references_for(estimator_name)

        if diagnostics_results is None:
            diagnostics_block: Dict[str, Any] = {
                "status": "skipped",
                "reason": "auto_diagnostics=False",
            }
        else:
            diagnostics_block = {
                "status": "ran",
                "schema": dr_schema,
                "overall_interpretation": (
                    dr_schema.get("overall_interpretation", "") if dr_schema is not None else ""
                ),
            }

        return {
            "schema_version": BUSINESS_REPORT_SCHEMA_VERSION,
            "estimator": {
                "class_name": estimator_name,
                "display_name": estimator_name,
            },
            "context": {
                "outcome_label": self._context.outcome_label,
                "outcome_unit": self._context.outcome_unit,
                "outcome_direction": self._context.outcome_direction,
                "business_question": self._context.business_question,
                "treatment_label": self._context.treatment_label,
                "alpha": self._context.alpha,
            },
            "headline": headline,
            "assumption": assumption,
            "pre_trends": pre_trends,
            "sensitivity": sensitivity,
            "sample": sample,
            "heterogeneity": heterogeneity,
            "robustness": robustness,
            "diagnostics": diagnostics_block,
            "next_steps": next_steps,
            "caveats": caveats,
            "references": references,
        }

    def _extract_headline(self, dr_schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract the headline effect + CI + p-value from the result."""
        r = self._results
        # Delegate the attribute-alias lookup to the shared helper in the
        # diagnostic_report module so BR and DR agree on which fields a
        # result class exposes for its headline (including
        # ``ContinuousDiDResults`` which uses ``overall_att_se`` /
        # ``overall_att_p_value`` / ``overall_att_conf_int``).
        from diff_diff.diagnostic_report import _extract_scalar_headline

        extracted = _extract_scalar_headline(r, fallback_alpha=self._context.alpha)
        att: Optional[float] = None
        se: Optional[float] = None
        p: Optional[float] = None
        ci: Optional[List[float]] = None
        alpha = self._context.alpha
        result_alpha: Optional[float] = None
        if extracted is not None:
            _name, att, se, p, ci, result_alpha = extracted

        # On any alpha mismatch, preserve the fitted CI at its native
        # level. A faithful CI cannot be recomputed from point estimate
        # and SE alone without reproducing the fit's inference contract
        # (finite-df t-quantile, percentile bootstrap, wild cluster
        # bootstrap, survey replicate quantile, rank-deficient
        # undefined-df, etc.), and the 16 result classes do not expose
        # a uniform descriptor for that. Two separate alpha values:
        # ``display_alpha`` drives ``ci_level`` so the displayed CI
        # label matches the preserved bounds; the caller's requested
        # alpha drives the significance phrasing (``is_significant`` /
        # ``near_threshold``). A caveat records the override.
        display_alpha = alpha
        phrasing_alpha = alpha
        alpha_was_honored = True
        alpha_override_caveat: Optional[str] = None
        if (
            result_alpha is not None
            and not np.isclose(alpha, result_alpha)
            and att is not None
            and se is not None
        ):
            inference_method = getattr(r, "inference_method", "analytical")
            if inference_method == "wild_bootstrap":
                inference_label = "wild cluster bootstrap"
            elif (
                inference_method == "bootstrap" or getattr(r, "bootstrap_results", None) is not None
            ):
                inference_label = "bootstrap"
            elif getattr(r, "bootstrap_distribution", None) is not None:
                inference_label = "bootstrap"
            elif getattr(r, "variance_method", None) in {"bootstrap", "jackknife", "placebo"}:
                variance_method = getattr(r, "variance_method", None)
                inference_label = f"{variance_method} variance"
            else:
                df_survey = getattr(
                    r,
                    "df_survey",
                    getattr(getattr(r, "survey_metadata", None), "df_survey", None),
                )
                if isinstance(df_survey, (int, float)) and df_survey > 0:
                    inference_label = "finite-df survey"
                elif isinstance(df_survey, (int, float)) and df_survey == 0:
                    # Rank-deficient replicate design: the fit deliberately
                    # left inference undefined. Preserve (NaN bounds remain NaN).
                    inference_label = "undefined-df (replicate-weight)"
                else:
                    # Ordinary analytical fit with a finite but unexposed
                    # ``df`` (``DifferenceInDifferences`` / ``MultiPeriodDiD``
                    # / most staggered estimators / TROP). We cannot
                    # reproduce the t-quantile without the fit's ``df``.
                    inference_label = "analytical (native degrees of freedom)"

            display_alpha = float(result_alpha)
            alpha_was_honored = False
            alpha_override_caveat = (
                f"Requested alpha ({phrasing_alpha:.2f}) was not honored "
                f"for the confidence interval because this fit uses "
                f"{inference_label} inference; the displayed CI remains "
                f"at the fit's native level "
                f"({int(round((1.0 - result_alpha) * 100))}%). The "
                f"significance phrasing still uses the requested alpha."
            )

        unit = self._context.outcome_unit
        unit_kind = _UNIT_KINDS.get(unit.lower() if unit else "", "unknown")
        sign = (
            "positive"
            if (att is not None and att > 0)
            else (
                "negative"
                if (att is not None and att < 0)
                else ("null" if att == 0 else "undefined")
            )
        )
        if att is None or not np.isfinite(att):
            sign = "undefined"
        ci_level = int(round((1.0 - display_alpha) * 100))
        is_significant = (
            p is not None and np.isfinite(p) and p < phrasing_alpha if p is not None else False
        )
        near_threshold = (
            p is not None
            and np.isfinite(p)
            and (phrasing_alpha - 0.01) < p < (phrasing_alpha + 0.001)
        )
        # Use DR-computed breakdown_M if available for quick reference.
        breakdown_M: Optional[float] = None
        if dr_schema:
            sens_section = dr_schema.get("sensitivity") or {}
            if sens_section.get("status") == "ran":
                breakdown_M = sens_section.get("breakdown_M")

        return {
            "effect": att,
            "se": se,
            "ci_lower": ci[0] if ci else None,
            "ci_upper": ci[1] if ci else None,
            "alpha_was_honored": alpha_was_honored,
            "alpha_override_caveat": alpha_override_caveat,
            "ci_level": ci_level,
            "p_value": p,
            "is_significant": is_significant,
            "near_significance_threshold": near_threshold,
            "unit": unit,
            "unit_kind": unit_kind,
            "sign": sign,
            "breakdown_M": breakdown_M,
        }

    def _extract_sample(self) -> Dict[str, Any]:
        """Extract sample metadata from the fitted result."""
        r = self._results
        survey = self._extract_survey_block()
        n_treated = _safe_int(getattr(r, "n_treated", getattr(r, "n_treated_units", None)))
        n_control_units = _safe_int(getattr(r, "n_control", getattr(r, "n_control_units", None)))

        # Control-group semantics. For estimators that expose a
        # ``control_group`` kwarg (CS, EfficientDiD), the meaning of
        # ``n_control_units`` depends on it. On CallawaySantAnna with
        # ``control_group="not_yet_treated"``, ``n_control_units`` counts
        # only the never-treated subset, so the actual dynamic
        # comparison group can be non-empty even when this count is 0.
        # Label the exposed count as never-treated and record the
        # active control-group mode so prose can surface the dynamic-
        # comparison context instead of misreporting "0 control"
        # (round-13 CI review on PR #318).
        control_group = getattr(r, "control_group", None)
        n_never_treated: Optional[int] = None
        n_control: Optional[int] = n_control_units
        if isinstance(control_group, str) and control_group == "not_yet_treated":
            n_never_treated = n_control_units
            # Do not populate a fixed ``n_control`` for this mode: the
            # comparison set is dynamic and varies by (g, t) cell.
            n_control = None

        return {
            "n_obs": _safe_int(getattr(r, "n_obs", None)),
            "n_treated": n_treated,
            "n_control": n_control,
            "n_never_treated": n_never_treated,
            "control_group": control_group if isinstance(control_group, str) else None,
            "n_periods": _safe_int(getattr(r, "n_periods", None)),
            "pre_periods": _safe_list_len(getattr(r, "pre_periods", None)),
            "post_periods": _safe_list_len(getattr(r, "post_periods", None)),
            "survey": survey,
        }

    def _extract_survey_block(self) -> Optional[Dict[str, Any]]:
        sm = getattr(self._results, "survey_metadata", None)
        if sm is None:
            return None
        deff = _safe_float(getattr(sm, "design_effect", None))
        return {
            "weight_type": getattr(sm, "weight_type", None),
            "effective_n": _safe_float(getattr(sm, "effective_n", None)),
            "design_effect": deff,
            "is_trivial": deff is not None and 0.95 <= deff <= 1.05,
            "n_strata": _safe_int(getattr(sm, "n_strata", None)),
            "n_psu": _safe_int(getattr(sm, "n_psu", None)),
            "df_survey": _safe_int(getattr(sm, "df_survey", None)),
            "replicate_method": getattr(sm, "replicate_method", None),
        }


# ---------------------------------------------------------------------------
# Schema helpers (module-private)
# ---------------------------------------------------------------------------
def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _safe_ci(ci: Any) -> Optional[List[float]]:
    if ci is None:
        return None
    try:
        lo, hi = ci
    except (TypeError, ValueError):
        return None
    lo_f = _safe_float(lo)
    hi_f = _safe_float(hi)
    if lo_f is None or hi_f is None:
        return None
    return [lo_f, hi_f]


def _safe_list_len(val: Any) -> Optional[int]:
    if val is None:
        return None
    try:
        return int(len(val))
    except TypeError:
        return None


def _lift_pre_trends(dr: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Pull pre-trends + power into a single BR-facing block."""
    if dr is None:
        return {"status": "skipped", "reason": "auto_diagnostics=False"}
    pt = dr.get("parallel_trends") or {}
    pp = dr.get("pretrends_power") or {}
    if pt.get("status") != "ran":
        return {
            "status": pt.get("status", "not_run"),
            "reason": pt.get("reason"),
        }
    return {
        "status": "computed",
        "method": pt.get("method"),
        "joint_p_value": pt.get("joint_p_value"),
        "verdict": pt.get("verdict"),
        "n_pre_periods": pt.get("n_pre_periods"),
        "power_status": pp.get("status"),
        "power_tier": pp.get("tier"),
        "mdv": pp.get("mdv"),
        "mdv_share_of_att": pp.get("mdv_share_of_att"),
        # Carry the covariance-source annotation through so BR can hedge the
        # power-tier phrasing when compute_pretrends_power silently used a
        # diagonal fallback despite event_study_vcov being available.
        "power_covariance_source": pp.get("covariance_source"),
    }


def _lift_sensitivity(dr: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if dr is None:
        return {"status": "skipped", "reason": "auto_diagnostics=False"}
    sens = dr.get("sensitivity") or {}
    if sens.get("status") != "ran":
        return {
            "status": sens.get("status", "not_run"),
            "reason": sens.get("reason"),
        }
    return {
        "status": "computed",
        "method": sens.get("method"),
        "breakdown_M": sens.get("breakdown_M"),
        "conclusion": sens.get("conclusion"),
        "grid": sens.get("grid"),
    }


def _lift_heterogeneity(dr: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if dr is None:
        return None
    het = dr.get("heterogeneity") or {}
    if het.get("status") != "ran":
        return None
    return {
        "source": het.get("source"),
        "n_effects": het.get("n_effects"),
        "min": het.get("min"),
        "max": het.get("max"),
        "cv": het.get("cv"),
        "sign_consistent": het.get("sign_consistent"),
    }


def _lift_robustness(dr: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if dr is None:
        return {"status": "skipped", "reason": "auto_diagnostics=False"}
    bacon = dr.get("bacon") or {}
    native = dr.get("estimator_native_diagnostics") or {}
    return {
        "bacon": {
            "status": bacon.get("status"),
            "forbidden_weight": bacon.get("forbidden_weight"),
            "verdict": bacon.get("verdict"),
        },
        "estimator_native": {
            "status": native.get("status"),
            "pre_treatment_fit": native.get("pre_treatment_fit"),
        },
    }


def _describe_assumption(estimator_name: str, results: Any = None) -> Dict[str, Any]:
    """Return the identifying-assumption block for an estimator."""
    if estimator_name in {
        "SyntheticDiDResults",
    }:
        return {
            "parallel_trends_variant": "weighted_pt",
            "no_anticipation": True,
            "description": (
                "Synthetic-Difference-in-Differences identifies the ATT under a "
                "weighted parallel-trends analogue: the synthetic control is "
                "chosen to match the treated group's pre-period trajectory."
            ),
        }
    if estimator_name in {"TROPResults"}:
        return {
            "parallel_trends_variant": "factor_model",
            "no_anticipation": True,
            "description": (
                "TROP uses low-rank factor-model identification rather than a "
                "parallel-trends assumption; unobserved heterogeneity is "
                "captured through latent factor loadings."
            ),
        }
    if estimator_name == "ContinuousDiDResults":
        # Callaway, Goodman-Bacon & Sant'Anna (2024), two-level PT:
        # REGISTRY.md §ContinuousDiD > Identification.
        return {
            "parallel_trends_variant": "dose_pt_or_strong_pt",
            "no_anticipation": True,
            "description": (
                "ContinuousDiD identifies dose-specific treatment effects "
                "under two possible parallel-trends conditions (Callaway, "
                "Goodman-Bacon & Sant'Anna 2024). Parallel Trends (PT) "
                "assumes untreated potential outcome paths are equal across "
                "all dose groups and the untreated group (conditional on "
                "dose), identifying ATT(d|d) and the binarized ATT^loc but "
                "NOT ATT(d), ACRT, or cross-dose comparisons. Strong "
                "Parallel Trends (SPT) additionally rules out selection "
                "into dose on the basis of treatment effects and is "
                "required to identify the dose-response curve ATT(d), "
                "marginal effect ACRT(d), and cross-dose contrasts."
            ),
        }
    if estimator_name in {"TripleDifferenceResults", "StaggeredTripleDiffResults"}:
        # Ortiz-Villavicencio & Sant'Anna (2025) — identification is the
        # triple-difference cancellation across the 2x2x2 cells, not
        # ordinary DiD parallel trends; see REGISTRY.md §TripleDifference
        # and §StaggeredTripleDifference.
        return {
            "parallel_trends_variant": "triple_difference_cancellation",
            "no_anticipation": True,
            "description": (
                "Triple-difference identification relies on the DDD "
                "decomposition (Ortiz-Villavicencio & Sant'Anna 2025): "
                "the ATT is recovered from `DDD = DiD_A + DiD_B - DiD_C` "
                "across the Group x Period x Eligibility (or Treatment) "
                "cells, which differences out group-specific and "
                "period-specific unobservables without requiring separate "
                "parallel trends to hold between each cell pair. The "
                "identifying restriction is therefore weaker than ordinary "
                "DiD parallel trends but assumes that the residual "
                "unobservable component is additively separable across the "
                "three dimensions; practical overlap and common-support "
                "conditions still apply on the propensity score when "
                "covariates are used."
            ),
        }
    if estimator_name == "ChaisemartinDHaultfoeuilleResults":
        # de Chaisemartin & D'Haultfoeuille (2020, 2024) — identification is
        # transition-based across (joiner, leaver, stable-control) cells
        # around each switching period, not a group-time ATT parallel-
        # trends restriction. Writing up dCDH as "parallel trends across
        # treatment cohorts" was flagged as a source-faithfulness bug in
        # PR #318 review; REGISTRY.md §ChaisemartinDHaultfoeuille is
        # explicit about the transition-set construction.
        return {
            "parallel_trends_variant": "transition_based",
            "no_anticipation": True,
            "description": (
                "Identification is transition-based (de Chaisemartin & "
                "D'Haultfoeuille 2020; dynamic companion 2024). At each "
                "switching period, the estimator contrasts joiners "
                "(D:0->1), leavers (D:1->0), and stable-treated / "
                "stable-untreated control cells that share the same "
                "treatment state across adjacent periods, yielding the "
                "contemporaneous ``DID_M`` and per-horizon ``DID_l`` / "
                "``DID_{g,l}`` building blocks. The identifying "
                "restriction is parallel trends within each transition's "
                "stable-control cell (not a single group-time ATT PT "
                "condition across all cohorts) plus no anticipation; "
                "with non-binary treatment the stable-control match is "
                "additionally on exact baseline dose ``D_{g,1}``. "
                "Reversible treatment is natively supported, unlike the "
                "absorbing-treatment designs that rely on a fixed "
                "treatment-onset cohort."
            ),
        }
    if estimator_name == "EfficientDiDResults":
        # Chen, Sant'Anna & Xie (2025) — identification is parameterized
        # by ``pt_assumption`` ("all" vs "post"). PT-All is the stronger
        # regime (PT across all groups/periods, over-identified — paper
        # Lemma 2.1), PT-Post the weaker (PT only in post-treatment,
        # just-identified reduction to single-baseline DiD per Corollary
        # 3.2). Also read ``control_group`` when present (not_yet_treated
        # vs last_cohort) to be source-faithful to REGISTRY.md §EfficientDiD
        # lines 736-738 and 907.
        pt_assumption = getattr(results, "pt_assumption", "all")
        control_group = getattr(results, "control_group", None)
        # The estimator only accepts ``control_group`` values of
        # ``"never_treated"`` (the default) or ``"last_cohort"``. When
        # ``last_cohort`` is used, the latest treatment cohort is
        # reclassified as a pseudo-never-treated comparison and time
        # periods at/after its onset are dropped; describing such a fit
        # with generic never-treated language would misstate the
        # identifying setup (see REGISTRY.md §EfficientDiD line 908).
        is_last_cohort = control_group == "last_cohort"
        if pt_assumption == "post":
            variant = "pt_post"
            if is_last_cohort:
                control_clause = (
                    "the comparison group is the latest treated cohort "
                    "reclassified as pseudo-never-treated (periods "
                    "at/after that cohort's treatment start are "
                    "dropped)"
                )
            else:
                control_clause = "the comparison group is never-treated"
            description = (
                "Identification under PT-Post (Chen, Sant'Anna & Xie "
                "2025): parallel trends holds only in post-treatment "
                "periods, " + control_clause + ", and the baseline is period g-1 only. This is the "
                "weaker of the two regimes — just-identified and "
                "reducing to standard single-baseline DiD (Corollary "
                "3.2). Also assumes no anticipation (Assumption NA), "
                "overlap (Assumption O), and absorbing / irreversible "
                "treatment."
            )
        else:
            variant = "pt_all"
            if is_last_cohort:
                baseline_clause = (
                    "using the latest treated cohort as a pseudo-never-"
                    "treated comparison (periods at/after that cohort's "
                    "treatment start are dropped); any earlier cohort "
                    "and any pre-treatment period can serve as baseline"
                )
            else:
                baseline_clause = (
                    "using never-treated units as comparison; any "
                    "not-yet-treated cohort and any pre-treatment period "
                    "can serve as baseline"
                )
            description = (
                "Identification under PT-All (Chen, Sant'Anna & Xie "
                "2025): parallel trends holds for all groups and all "
                "periods, "
                + baseline_clause
                + ". The estimator is over-identified (Lemma 2.1), and "
                "the paper's optimal combination weights are applied. "
                "Also assumes no anticipation (Assumption NA), overlap "
                "(Assumption O), and absorbing / irreversible "
                "treatment. The Hausman PT-All vs PT-Post pretest "
                "(operating on the post-treatment event-study vector "
                "ES(e), Theorem A.1) checks whether the stronger "
                "PT-All regime is tenable."
            )
        block: Dict[str, Any] = {
            "parallel_trends_variant": variant,
            "no_anticipation": True,
            "description": description,
        }
        if isinstance(control_group, str):
            block["control_group"] = control_group
        return block
    if estimator_name in {
        "CallawaySantAnnaResults",
        "SunAbrahamResults",
        "ImputationDiDResults",
        "TwoStageDiDResults",
        "StackedDiDResults",
        "WooldridgeDiDResults",
    }:
        return {
            "parallel_trends_variant": "conditional_or_group_time",
            "no_anticipation": True,
            "description": (
                "Identification relies on parallel trends across treatment "
                "cohorts and time periods (group-time ATT), plus no "
                "anticipation."
            ),
        }
    return {
        "parallel_trends_variant": "unconditional",
        "no_anticipation": True,
        "description": (
            "Identification relies on the standard DiD parallel-trends "
            "assumption plus no anticipation of treatment by either group."
        ),
    }


def _build_caveats(
    _results: Any,
    headline: Dict[str, Any],
    sample: Dict[str, Any],
    dr_schema: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Assemble the plain-English caveats list for the headline schema."""
    caveats: List[Dict[str, Any]] = []

    # NaN ATT is the highest-severity caveat.
    if headline.get("sign") == "undefined":
        caveats.append(
            {
                "severity": "warning",
                "topic": "estimation_failure",
                "message": (
                    "Estimation produced a non-finite effect. Inspect data "
                    "preparation and model specification before interpreting."
                ),
            }
        )

    # Alpha override could not be honored (bootstrap / finite-df inference).
    alpha_override_msg = headline.get("alpha_override_caveat")
    if isinstance(alpha_override_msg, str) and alpha_override_msg:
        caveats.append(
            {
                "severity": "info",
                "topic": "alpha_override_preserved",
                "message": alpha_override_msg,
            }
        )

    # Near-threshold p-value.
    if headline.get("near_significance_threshold"):
        caveats.append(
            {
                "severity": "info",
                "topic": "near_significance",
                "message": (
                    "The p-value is close to the conventional significance "
                    "threshold; small changes to the sample or specification "
                    "could move it either way."
                ),
            }
        )

    # Few treated units.
    nt = sample.get("n_treated")
    if nt is not None and nt <= 3:
        caveats.append(
            {
                "severity": "warning",
                "topic": "few_treated",
                "message": (
                    f"Only {nt} treated units in this fit; standard errors "
                    "rely on large-cluster asymptotics and may be unreliable. "
                    "Consider SyntheticDiD or an exact-permutation inference "
                    "alternative."
                ),
            }
        )

    # Non-trivial design effect.
    survey = sample.get("survey")
    if survey and not survey.get("is_trivial"):
        deff = survey.get("design_effect")
        eff_n = survey.get("effective_n")
        if isinstance(deff, (int, float)) and deff >= 5.0:
            caveats.append(
                {
                    "severity": "warning",
                    "topic": "design_effect",
                    "message": (
                        f"Very large survey design effect (DEFF = {deff:.2g}). "
                        "Inspect the weight distribution and consider weight "
                        "trimming if driven by outlier weights."
                    ),
                }
            )
        elif isinstance(deff, (int, float)) and deff >= 1.5:
            if isinstance(eff_n, (int, float)):
                caveats.append(
                    {
                        "severity": "info",
                        "topic": "design_effect",
                        "message": (
                            f"Survey design reduces effective sample size: "
                            f"DEFF = {deff:.2g}; effective n = {eff_n:.0f}."
                        ),
                    }
                )

    # Bacon forbidden comparisons.
    if dr_schema:
        bacon = dr_schema.get("bacon") or {}
        if bacon.get("status") == "ran":
            fw = bacon.get("forbidden_weight")
            if isinstance(fw, (int, float)) and fw > 0.10:
                caveats.append(
                    {
                        "severity": "warning",
                        "topic": "bacon_contamination",
                        "message": (
                            f"Goodman-Bacon decomposition places {fw:.0%} "
                            "of implicit TWFE weight on 'forbidden' "
                            "later-vs-earlier comparisons. TWFE may be "
                            "materially biased under heterogeneous effects. "
                            "Re-estimate with a heterogeneity-robust "
                            "estimator (CS / SA / BJS / Gardner)."
                        ),
                    }
                )

        # Fragile sensitivity.
        sens = dr_schema.get("sensitivity") or {}
        if sens.get("status") == "ran":
            bkd = sens.get("breakdown_M")
            if isinstance(bkd, (int, float)) and bkd < 0.5:
                caveats.append(
                    {
                        "severity": "warning",
                        "topic": "sensitivity_fragility",
                        "message": (
                            f"HonestDiD breakdown value is {bkd:.2g}: the "
                            "result's confidence interval includes zero "
                            "once parallel-trends violations reach less than "
                            "half the observed pre-period variation. Treat "
                            "the headline as tentative."
                        ),
                    }
                )

        # Sensitivity was skipped for methodology reasons (e.g., CS fit with
        # ``base_period='varying'`` — HonestDiD bounds are not interpretable
        # there). Surface the reason as a warning-severity caveat so readers
        # do not assume the headline is robust across the R-R grid.
        if sens.get("status") == "skipped":
            reason = sens.get("reason")
            if isinstance(reason, str) and reason:
                caveats.append(
                    {
                        "severity": "warning",
                        "topic": "sensitivity_skipped",
                        "message": ("HonestDiD sensitivity was not run on this fit. " + reason),
                    }
                )

        # Non-fatal warnings captured from delegated diagnostics
        # (e.g., HonestDiD's bootstrap diag-covariance fallback, dropped
        # non-consecutive horizons on dCDH). DR already records these in
        # ``schema["warnings"]``; mirror the methodology-critical ones
        # into BR's caveat list so summary/full-report prose can surface
        # them without readers having to inspect the DR schema.
        for msg in dr_schema.get("warnings", []) or []:
            if not isinstance(msg, str) or not msg:
                continue
            # Skip alpha-override and design-effect messages already
            # covered by dedicated caveats above.
            lower = msg.lower()
            if "sensitivity:" in lower or "pretrends_power:" in lower:
                caveats.append(
                    {
                        "severity": "info",
                        "topic": "diagnostic_warning",
                        "message": msg,
                    }
                )

    # Unit mismatch caveat (log_points + unit override).
    unit_kind = headline.get("unit_kind")
    if unit_kind == "log_points":
        caveats.append(
            {
                "severity": "info",
                "topic": "unit_policy",
                "message": (
                    "The effect is reported in log-points as estimated; "
                    "BusinessReport does not arithmetically translate log-points "
                    "to percent or level changes. For small effects, log-points "
                    "approximate percentage changes."
                ),
            }
        )
    return caveats


def _pt_method_subject(method: Optional[str]) -> str:
    """Return a source-faithful sentence subject for the PT verdict prose.

    The ``parallel_trends.method`` field distinguishes between the
    2x2 slope-difference check, the pre-period event-study Wald /
    Bonferroni variants, EfficientDiD's Hausman PT-All vs PT-Post
    pretest, SDiD's weighted pre-treatment fit, and TROP's factor-
    model identification. Generic "pre-treatment event-study" wording
    is wrong for the first and third cases. See round-8 CI review on
    PR #318 and REGISTRY.md §EfficientDiD (Hausman pretest).
    """
    if method == "slope_difference":
        return "The pre-period slope-difference test"
    if method == "hausman":
        return "The Hausman PT-All vs PT-Post pretest"
    if method in {"joint_wald", "joint_wald_event_study", "joint_wald_no_vcov", "bonferroni"}:
        return "Pre-treatment event-study coefficients"
    if method == "synthetic_fit":
        return "The synthetic-control pre-treatment fit"
    if method == "factor":
        return "The factor-model pre-treatment fit"
    return "Pre-treatment data"


def _pt_method_stat_label(method: Optional[str]) -> Optional[str]:
    """Return the joint-statistic label appropriate to the PT method.

    Returns ``"joint p"`` for Wald / Bonferroni paths, ``"p"`` for the
    2x2 slope-difference and Hausman paths (which are single-statistic
    tests), and ``None`` for design-enforced paths that have no p-value.
    """
    if method in {"joint_wald", "joint_wald_event_study", "joint_wald_no_vcov", "bonferroni"}:
        return "joint p"
    if method in {"slope_difference", "hausman"}:
        return "p"
    if method in {"synthetic_fit", "factor"}:
        return None
    return "joint p"


def _references_for(estimator_name: str) -> List[Dict[str, str]]:
    """Map the estimator to the appropriate citation references."""
    base = [
        {
            "role": "sensitivity",
            "citation": (
                "Rambachan, A., & Roth, J. (2023). A More Credible Approach "
                "to Parallel Trends. Review of Economic Studies."
            ),
        },
        {
            "role": "workflow",
            "citation": (
                "Baker, A. C., Callaway, B., Cunningham, S., Goodman-Bacon, A., "
                "& Sant'Anna, P. H. C. (2025). Difference-in-Differences "
                "Designs: A Practitioner's Guide."
            ),
        },
    ]
    estimator_refs = {
        "CallawaySantAnnaResults": {
            "role": "estimator",
            "citation": (
                "Callaway, B., & Sant'Anna, P. H. C. (2021). "
                "Difference-in-Differences with multiple time periods. "
                "Journal of Econometrics."
            ),
        },
        "SyntheticDiDResults": {
            "role": "estimator",
            "citation": (
                "Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., "
                "& Wager, S. (2021). Synthetic Difference in Differences."
            ),
        },
        "SunAbrahamResults": {
            "role": "estimator",
            "citation": (
                "Sun, L., & Abraham, S. (2021). Estimating dynamic treatment "
                "effects in event studies. Journal of Econometrics."
            ),
        },
        "ImputationDiDResults": {
            "role": "estimator",
            "citation": (
                "Borusyak, K., Jaravel, X., & Spiess, J. (2024). " "Revisiting event-study designs."
            ),
        },
        "EfficientDiDResults": {
            "role": "estimator",
            "citation": (
                "Chen, X., Sant'Anna, P. H. C., & Xie, H. (2025). "
                "Efficient Estimation of Treatment Effects in Staggered "
                "DiD Designs."
            ),
        },
        "ChaisemartinDHaultfoeuilleResults": {
            "role": "estimator",
            "citation": (
                "de Chaisemartin, C., & D'Haultfœuille, X. (2020). "
                "Two-way fixed effects estimators with heterogeneous "
                "treatment effects. American Economic Review."
            ),
        },
    }
    if estimator_name in estimator_refs:
        return [estimator_refs[estimator_name]] + base
    return base


# ---------------------------------------------------------------------------
# Prose rendering
# ---------------------------------------------------------------------------
def _format_value(value: Optional[float], unit: Optional[str], unit_kind: str) -> str:
    """Format a numeric effect with its unit. No arithmetic translation."""
    if value is None or not np.isfinite(value):
        return "undefined"
    if unit_kind == "currency":
        sign = "-" if value < 0 else ""
        return f"{sign}${abs(value):,.2f}"
    if unit_kind == "percent":
        return f"{value:.2f}%"
    if unit_kind == "percentage_points":
        return f"{value:.2f} pp"
    if unit_kind == "log_points":
        return f"{value:.3g} log-points"
    if unit_kind == "count":
        return f"{value:,.0f}"
    # unknown / free-form
    if unit:
        return f"{value:.3g} {unit}"
    return f"{value:.3g}"


def _significance_phrase(p: Optional[float], alpha: float) -> str:
    """Return a plain-English significance phrase.

    Tiers per ``docs/methodology/REPORTING.md``:
      * p < 0.001: "strongly supported by the data"
      * 0.001 <= p < 0.01: "well-supported"
      * 0.01 <= p < alpha: "statistically significant at the X% level"
      * alpha <= p < 0.10: CI-includes-zero language
      * p >= 0.10: consistent-with-no-effect language
    """
    if p is None or not np.isfinite(p):
        return "statistical significance cannot be assessed (p-value unavailable)"
    ci_level = int(round((1.0 - alpha) * 100))
    if p < 0.001:
        return "the direction of the effect is strongly supported by the data"
    if p < 0.01:
        return "the direction of the effect is well-supported by the data"
    if p < alpha:
        return f"the effect is statistically significant at the {ci_level}% level"
    if p < 0.10:
        return (
            "the confidence interval includes zero; the direction is suggestive "
            "but not statistically significant"
        )
    return "the confidence interval includes zero; the data are consistent with no effect"


def _direction_verb(effect: float, outcome_direction: Optional[str]) -> str:
    """Return a direction-aware verb for the headline sentence.

    When ``outcome_direction`` is unset we use neutral change verbs
    (``increased`` / ``decreased``). When it is supplied, we additionally
    flavor the verb with a value-laden connotation so the stakeholder can
    read off whether the estimated effect points in the desired direction:

    - ``higher_is_better``: positive effect -> "lifted"; negative -> "reduced"
    - ``lower_is_better``:  positive effect -> "worsened"; negative -> "improved"
    - None:                 positive -> "increased"; negative -> "decreased"
    """
    if effect == 0:
        return "did not change"
    if outcome_direction == "higher_is_better":
        return "lifted" if effect > 0 else "reduced"
    if outcome_direction == "lower_is_better":
        return "worsened" if effect > 0 else "improved"
    return "increased" if effect > 0 else "decreased"


def _render_headline_sentence(schema: Dict[str, Any]) -> str:
    """Render the headline sentence from the schema.

    Uses the absolute value in the magnitude slot when the verb already
    conveys direction ("decreased ... by $0.14" rather than "decreased ...
    by -$0.14"). CI bounds are rendered at their natural signed values.
    When ``outcome_direction`` is supplied, the verb picks up a value-laden
    connotation ("lifted" / "reduced" vs neutral "increased" / "decreased").
    """
    ctx = schema.get("context", {})
    h = schema.get("headline", {})
    effect = h.get("effect")
    outcome = ctx.get("outcome_label", "the outcome")
    treatment = ctx.get("treatment_label", "the treatment")
    outcome_direction = ctx.get("outcome_direction")
    unit = h.get("unit")
    unit_kind = h.get("unit_kind", "unknown")

    if effect is None or not np.isfinite(effect):
        return (
            f"We were unable to produce a finite estimate of {treatment}'s "
            f"effect on {outcome}. Inspect the data and model specification."
        )

    verb = _direction_verb(effect, outcome_direction)
    magnitude = _format_value(abs(effect), unit, unit_kind)
    lo = h.get("ci_lower")
    hi = h.get("ci_upper")
    ci_str = ""
    if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
        lo_s = _format_value(lo, unit, unit_kind)
        hi_s = _format_value(hi, unit, unit_kind)
        ci_str = f" ({h.get('ci_level', 95)}% CI: {lo_s} to {hi_s})"
    by_clause = f" by {magnitude}" if effect != 0 else ""
    return f"{treatment.capitalize()} {verb} {outcome}{by_clause}{ci_str}."


def _render_summary(schema: Dict[str, Any]) -> str:
    """Render the short-form stakeholder summary paragraph."""
    sentences: List[str] = []
    ctx = schema.get("context", {})
    question = ctx.get("business_question")
    if question:
        sentences.append(f"Question: {question}")

    # Headline sentence with significance phrase.
    sentences.append(_render_headline_sentence(schema))
    h = schema.get("headline", {})
    p = h.get("p_value")
    alpha = ctx.get("alpha", 0.05)
    if p is not None and np.isfinite(p):
        sig = _significance_phrase(p, alpha)
        sentences.append(f"Statistically, {sig}.")
        if h.get("near_significance_threshold"):
            sentences.append(
                "The p-value is close to the conventional threshold; "
                "small changes to the sample could move it either way."
            )

    # Pre-trends + power-aware phrasing.
    pt = schema.get("pre_trends", {}) or {}
    if pt.get("status") == "computed":
        jp = pt.get("joint_p_value")
        verdict = pt.get("verdict")
        tier = pt.get("power_tier")
        method = pt.get("method")
        # ``compute_pretrends_power`` currently falls back to ``np.diag(ses**2)``
        # for CS / SA / ImputationDiD / Stacked / etc., even when the full
        # ``event_study_vcov`` is available. Downgrade any "well_powered" tier
        # to "moderately_powered" when we know the diagonal approximation was
        # the only input — a diagonal-only MDV can be optimistic because it
        # ignores correlations across pre-periods.
        cov_source = pt.get("power_covariance_source")
        if tier == "well_powered" and cov_source == "diag_fallback_available_full_vcov_unused":
            tier = "moderately_powered"
        subject = _pt_method_subject(method)
        stat_label = _pt_method_stat_label(method)
        jp_phrase = (
            f" ({stat_label} = {jp:.3g})" if isinstance(jp, (int, float)) and stat_label else ""
        )
        # Only point to "the sensitivity analysis below" when a
        # sensitivity block actually ran. For estimators that route to
        # native diagnostics (SDiD / TROP) or fits where sensitivity was
        # skipped / not applicable, the clause would mislead (round-12
        # CI review on PR #318).
        sens_ran = (schema.get("sensitivity", {}) or {}).get("status") == "computed"
        sens_tail_major = " pending the sensitivity analysis below" if sens_ran else ""
        sens_tail_alongside = " alongside the sensitivity analysis below" if sens_ran else ""
        sens_tail_see_bounded = (
            " See the sensitivity analysis below for bounded-violation guarantees."
            if sens_ran
            else ""
        )
        sens_tail_see_reliable = " See the sensitivity analysis below." if sens_ran else ""
        if verdict == "clear_violation":
            sentences.append(
                f"{subject} clearly reject parallel trends{jp_phrase}; the "
                "headline should be treated as tentative" + sens_tail_major + "."
            )
        elif verdict == "some_evidence_against":
            sentences.append(
                f"{subject} show some evidence against parallel trends"
                f"{jp_phrase}; interpret the headline"
                + (sens_tail_alongside if sens_ran else " with caution")
                + "."
            )
        elif verdict == "no_detected_violation":
            if tier == "well_powered":
                sentences.append(
                    f"{subject} are consistent with parallel trends, and "
                    "the test is well-powered (the minimum-detectable "
                    "violation is small relative to the estimated effect)."
                )
            elif tier == "moderately_powered":
                sentences.append(
                    f"{subject} do not reject parallel trends; the test is "
                    "moderately informative." + sens_tail_see_bounded
                )
            else:
                sentences.append(
                    f"{subject} do not reject parallel trends, but the test "
                    "has limited power — a non-rejection does not prove the "
                    "assumption." + sens_tail_see_reliable
                )
        elif verdict == "design_enforced_pt":
            sentences.append(
                "The synthetic control is designed to match the treated "
                "group's pre-period trajectory (SDiD's weighted-parallel-"
                "trends analogue)."
            )

    # Sensitivity. A ``single_M_precomputed`` sensitivity block has
    # ``breakdown_M=None`` by construction because only one M was evaluated;
    # narrate it as a point check, NOT as grid-wide robustness.
    sens = schema.get("sensitivity", {}) or {}
    if sens.get("status") == "computed":
        bkd = sens.get("breakdown_M")
        conclusion = sens.get("conclusion")
        if conclusion == "single_M_precomputed":
            grid_points = sens.get("grid") or []
            point = grid_points[0] if grid_points else {}
            m_val = point.get("M")
            robust = point.get("robust_to_zero")
            if isinstance(m_val, (int, float)):
                if robust:
                    sentences.append(
                        f"HonestDiD (single point checked): at M = {m_val:.2g}, "
                        f"the robust confidence interval excludes zero. This is "
                        f"a point check, not a breakdown analysis — run "
                        f"HonestDiD.sensitivity() across a grid of M values "
                        f"for a full robustness claim."
                    )
                else:
                    sentences.append(
                        f"HonestDiD (single point checked): at M = {m_val:.2g}, "
                        f"the robust confidence interval includes zero. Run "
                        f"HonestDiD.sensitivity() across a grid to find the "
                        f"breakdown value."
                    )
        elif bkd is None:
            sentences.append(
                "HonestDiD: the result remains significant across the "
                "full grid — robust to plausible parallel-trends violations."
            )
        elif isinstance(bkd, (int, float)) and bkd >= 1.0:
            sentences.append(
                f"HonestDiD: the result remains significant under "
                f"parallel-trends violations up to {bkd:.2g}x the observed "
                f"pre-period variation."
            )
        elif isinstance(bkd, (int, float)):
            sentences.append(
                f"HonestDiD: the result is fragile — the confidence interval "
                f"includes zero once violations reach {bkd:.2g}x the "
                f"pre-period variation."
            )

    # Sample sentence. For CS ``control_group="not_yet_treated"`` the
    # fixed control count is suppressed because the comparison group is
    # dynamic; narrate the mode explicitly rather than misreporting a
    # never-treated-only tally as "control" (round-13 CI review).
    sample = schema.get("sample", {}) or {}
    n_obs = sample.get("n_obs")
    n_t = sample.get("n_treated")
    n_c = sample.get("n_control")
    n_nt = sample.get("n_never_treated")
    control_mode = sample.get("control_group")
    if isinstance(n_obs, int):
        if isinstance(n_t, int) and isinstance(n_c, int):
            sentences.append(f"Sample: {n_obs:,} observations ({n_t:,} treated, {n_c:,} control).")
        elif control_mode == "not_yet_treated" and isinstance(n_t, int):
            extra = (
                f"; {n_nt:,} never-treated units are also present"
                if isinstance(n_nt, int) and n_nt > 0
                else ""
            )
            sentences.append(
                f"Sample: {n_obs:,} observations ({n_t:,} treated) with a "
                "dynamic not-yet-treated comparison group (the control set "
                f"varies by cohort and period){extra}."
            )
        else:
            sentences.append(f"Sample: {n_obs:,} observations.")
        survey = sample.get("survey")
        if survey and not survey.get("is_trivial"):
            deff = survey.get("design_effect")
            eff_n = survey.get("effective_n")
            if isinstance(deff, (int, float)) and isinstance(eff_n, (int, float)):
                sentences.append(
                    f"Survey design reduces effective sample size to "
                    f"~{eff_n:,.0f} (DEFF = {deff:.2g})."
                )

    # Highest-severity caveat (if any).
    caveats = schema.get("caveats", [])
    warning_caveats = [c for c in caveats if c.get("severity") == "warning"]
    if warning_caveats:
        top = warning_caveats[0]
        sentences.append(f"Caveat: {top.get('message')}")

    return " ".join(s for s in sentences if s)


def _render_full_report(schema: Dict[str, Any]) -> str:
    """Render the structured multi-section markdown report."""
    ctx = schema.get("context", {})
    h = schema.get("headline", {})
    sample = schema.get("sample", {})
    pt = schema.get("pre_trends", {}) or {}
    sens = schema.get("sensitivity", {}) or {}
    assumption = schema.get("assumption", {})
    het = schema.get("heterogeneity")
    caveats = schema.get("caveats", [])
    references = schema.get("references", [])
    next_steps = schema.get("next_steps", [])

    lines: List[str] = []
    lines.append(f"# Business Report: {ctx.get('outcome_label', 'Outcome')}")
    lines.append("")
    if ctx.get("business_question"):
        lines.append(f"**Question**: {ctx['business_question']}")
        lines.append("")
    lines.append(f"**Estimator**: `{schema.get('estimator', {}).get('class_name')}`")
    lines.append("")

    # Headline
    lines.append("## Headline")
    lines.append("")
    lines.append(_render_headline_sentence(schema))
    p = h.get("p_value")
    alpha = ctx.get("alpha", 0.05)
    if isinstance(p, (int, float)):
        lines.append("")
        lines.append(f"Statistically, {_significance_phrase(p, alpha)}.")
    lines.append("")

    # Identifying assumption
    lines.append("## Identifying Assumption")
    lines.append("")
    lines.append(assumption.get("description", "") or "Standard DiD parallel-trends assumption.")
    lines.append("")

    # Pre-trends
    lines.append("## Pre-Trends")
    lines.append("")
    if pt.get("status") == "computed":
        jp = pt.get("joint_p_value")
        verdict = pt.get("verdict")
        tier = pt.get("power_tier")
        jp_str = f"joint p = {jp:.3g}" if isinstance(jp, (int, float)) else "joint p unavailable"
        lines.append(f"- Verdict: `{verdict}` ({jp_str})")
        if tier:
            lines.append(f"- Power tier: `{tier}`")
        mdv = pt.get("mdv")
        ratio = pt.get("mdv_share_of_att")
        if isinstance(mdv, (int, float)):
            lines.append(f"- Minimum detectable violation (MDV): {mdv:.3g}")
        if isinstance(ratio, (int, float)):
            lines.append(f"- MDV / |ATT|: {ratio:.2g}")
    else:
        lines.append(f"- Pre-trends not computed: {pt.get('reason', 'unavailable')}")
    lines.append("")

    # Sensitivity. A single-M HonestDiDResults passthrough has
    # breakdown_M=None by construction because only one M was evaluated;
    # the "robust across full grid" phrasing is reserved for genuine
    # grid-over-M SensitivityResults.
    lines.append("## Sensitivity (HonestDiD)")
    lines.append("")
    if sens.get("status") == "computed":
        bkd = sens.get("breakdown_M")
        concl = sens.get("conclusion")
        lines.append(f"- Method: `{sens.get('method')}`")
        if concl == "single_M_precomputed":
            grid_points = sens.get("grid") or []
            point = grid_points[0] if grid_points else {}
            m_val = point.get("M")
            robust = point.get("robust_to_zero")
            if isinstance(m_val, (int, float)):
                lines.append(f"- Single point checked: M = {m_val:.3g}")
                lines.append(
                    f"- Robust CI at M = {m_val:.3g}: "
                    f"{'excludes zero' if robust else 'includes zero'}"
                )
                lines.append(
                    "- Run `HonestDiD.sensitivity()` across a grid of M "
                    "values to find the breakdown value."
                )
            else:
                lines.append("- Single-M passthrough (breakdown not available)")
        elif isinstance(bkd, (int, float)):
            lines.append(f"- Breakdown M: {bkd:.3g}")
        else:
            lines.append("- Breakdown M: robust across full grid (no breakdown)")
        lines.append(f"- Conclusion: `{concl}`")
    else:
        lines.append(f"- Sensitivity not computed: {sens.get('reason', 'unavailable')}")
    lines.append("")

    # Sample
    lines.append("## Sample")
    lines.append("")
    if isinstance(sample.get("n_obs"), int):
        lines.append(f"- Observations: {sample['n_obs']:,}")
    if isinstance(sample.get("n_treated"), int):
        lines.append(f"- Treated: {sample['n_treated']:,}")
    # ``n_control`` is only populated for estimators whose control set
    # is a fixed tally. For CS ``control_group="not_yet_treated"`` the
    # comparison group is dynamic per (g, t); report the never-treated
    # count (when non-zero) and the dynamic-comparison mode explicitly.
    if isinstance(sample.get("n_control"), int):
        lines.append(f"- Control: {sample['n_control']:,}")
    elif sample.get("control_group") == "not_yet_treated":
        if isinstance(sample.get("n_never_treated"), int) and sample["n_never_treated"] > 0:
            lines.append(
                f"- Never-treated units present in the panel: {sample['n_never_treated']:,}"
            )
        lines.append(
            "- Comparison group: dynamic not-yet-treated units "
            "(varies by cohort and period; no fixed control count)"
        )
    survey = sample.get("survey")
    if survey:
        if survey.get("is_trivial"):
            lines.append("- Survey design: trivial DEFF (~1.0)")
        else:
            deff = survey.get("design_effect")
            eff_n = survey.get("effective_n")
            if isinstance(deff, (int, float)):
                lines.append(f"- Survey DEFF: {deff:.2g}")
            if isinstance(eff_n, (int, float)):
                lines.append(f"- Effective N: {eff_n:,.0f}")
    lines.append("")

    # Heterogeneity
    if het:
        lines.append("## Heterogeneity")
        lines.append("")
        lines.append(f"- Source: `{het.get('source')}`")
        lines.append(f"- N effects: {het.get('n_effects')}")
        mn = het.get("min")
        mx = het.get("max")
        if isinstance(mn, (int, float)) and isinstance(mx, (int, float)):
            lines.append(f"- Range: {mn:.3g} to {mx:.3g}")
        cv = het.get("cv")
        if isinstance(cv, (int, float)):
            lines.append(f"- CV: {cv:.3g}")
        lines.append(f"- Sign consistent: {het.get('sign_consistent')}")
        lines.append("")

    # Caveats
    if caveats:
        lines.append("## Caveats")
        lines.append("")
        for c in caveats:
            sev = c.get("severity", "info")
            lines.append(f"- **{sev.upper()}** — {c.get('message')}")
        lines.append("")

    # Next steps
    if next_steps:
        lines.append("## Next Steps")
        lines.append("")
        for s in next_steps:
            if s.get("label"):
                lines.append(f"- {s['label']}")
                if s.get("why"):
                    lines.append(f"  - _why_: {s['why']}")
        lines.append("")

    # References
    if references:
        lines.append("## References")
        lines.append("")
        for ref in references:
            lines.append(f"- {ref.get('citation')}")
        lines.append("")

    return "\n".join(lines)
