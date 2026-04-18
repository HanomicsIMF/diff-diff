"""
DiagnosticReport — unified, plain-English validity assessment for diff-diff results.

Orchestrates the library's existing diagnostic functions (parallel trends,
pre-trends power, HonestDiD sensitivity, Goodman-Bacon, design-effect
diagnostics, EPV, heterogeneity, and estimator-native checks for SDiD/TROP)
into a single report with a stable AI-legible schema.

Design principles:

- No hard pass/fail gates. Severity is conveyed by natural-language phrasing,
  not a traffic-light enum. See ``docs/methodology/REPORTING.md``.
- No new statistical computation. Every reported number is either read from
  ``results`` or computed by an existing diff-diff utility function.
- Lazy evaluation. ``DiagnosticReport(results, ...)`` is free; ``run_all()``
  triggers compute and caches.
- Never prove a null. Pre-trends phrasing uses power information from
  ``compute_pretrends_power`` to distinguish well-powered from underpowered
  non-violations.

The ``to_dict()`` surface is an AI-legible contract. See the schema reference
in ``docs/methodology/REPORTING.md`` and the ``DIAGNOSTIC_REPORT_SCHEMA_VERSION``
constant below. The schema is marked experimental in v3.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import pandas as pd

DIAGNOSTIC_REPORT_SCHEMA_VERSION = "1.0"

__all__ = [
    "DiagnosticReport",
    "DiagnosticReportResults",
    "DIAGNOSTIC_REPORT_SCHEMA_VERSION",
]


# ---------------------------------------------------------------------------
# Canonical check names and per-type applicability
# ---------------------------------------------------------------------------
# The set of check names that ``DiagnosticReport`` supports.
_CHECK_NAMES: Tuple[str, ...] = (
    "parallel_trends",
    "pretrends_power",
    "sensitivity",
    "bacon",
    "design_effect",
    "heterogeneity",
    "epv",
    "estimator_native",
    "placebo",
)

# Type-level applicability: which checks are *ever* applicable for each of the
# 16 result types. Instance-level applicability further filters by whether
# required attributes are present (e.g. ``survey_metadata`` for DEFF) and by
# whether the user disabled a check via ``run_*=False``.
# See ``docs/methodology/REPORTING.md`` for the full matrix and rationale.
#
# Implementation note: The keys are result-class names looked up via
# ``type(results).__name__``. This string-based dispatch mirrors the
# ``_HANDLERS`` pattern in ``diff_diff/practitioner.py`` and avoids circular
# imports across the 16 result modules. Renaming or aliasing any result class
# requires updating both this table and ``_PT_METHOD`` below; the
# applicability-matrix test parametrized over all result types serves as the
# regression guard.
# ``pretrends_power`` is restricted to the result families for which
# ``compute_pretrends_power`` has an explicit adapter — see
# ``diff_diff/pretrends.py`` around the result-type dispatch. Expanding
# beyond this set (Imputation / Stacked / TwoStage / EfficientDiD /
# StaggeredTripleDiff / Wooldridge / dCDH) would cause the helper to
# raise ``TypeError("Unsupported results type ...")`` and mark the check
# as ``error``, so the narrower set is the right contract.
#
# ``sensitivity`` is restricted to families with a ``HonestDiD``
# adapter: MultiPeriod, CS, dCDH (via ``placebo_event_study``). SDiD
# and TROP use their own native paths (``estimator_native``) instead
# of HonestDiD.
_APPLICABILITY: Dict[str, FrozenSet[str]] = {
    "DiDResults": frozenset({"parallel_trends", "design_effect"}),
    "MultiPeriodDiDResults": frozenset(
        {"parallel_trends", "pretrends_power", "sensitivity", "bacon", "design_effect"}
    ),
    "CallawaySantAnnaResults": frozenset(
        {
            "parallel_trends",
            "pretrends_power",
            "sensitivity",
            "bacon",
            "design_effect",
            "heterogeneity",
            "epv",
        }
    ),
    "SunAbrahamResults": frozenset(
        {
            "parallel_trends",
            "pretrends_power",
            "bacon",
            "design_effect",
            "heterogeneity",
        }
    ),
    "ImputationDiDResults": frozenset(
        {
            "parallel_trends",
            "bacon",
            "design_effect",
            "heterogeneity",
        }
    ),
    "TwoStageDiDResults": frozenset(
        {
            "parallel_trends",
            "bacon",
            "design_effect",
            "heterogeneity",
        }
    ),
    "StackedDiDResults": frozenset(
        {
            "parallel_trends",
            "bacon",
            "design_effect",
            "heterogeneity",
        }
    ),
    "SyntheticDiDResults": frozenset(
        {"parallel_trends", "sensitivity", "design_effect", "estimator_native"}
    ),
    "TROPResults": frozenset(
        {
            "parallel_trends",
            "sensitivity",
            "design_effect",
            "heterogeneity",
            "estimator_native",
        }
    ),
    "EfficientDiDResults": frozenset(
        {
            "parallel_trends",
            "bacon",
            "design_effect",
            "heterogeneity",
            "epv",
        }
    ),
    "ContinuousDiDResults": frozenset({"design_effect", "heterogeneity"}),
    "TripleDifferenceResults": frozenset({"design_effect", "epv"}),
    "StaggeredTripleDiffResults": frozenset({"parallel_trends", "design_effect"}),
    "WooldridgeDiDResults": frozenset(
        {
            "parallel_trends",
            "bacon",
            "design_effect",
            "heterogeneity",
        }
    ),
    "ChaisemartinDHaultfoeuilleResults": frozenset(
        {
            "parallel_trends",
            "sensitivity",
            "bacon",
            "design_effect",
        }
    ),
    "BaconDecompositionResults": frozenset({"bacon"}),
}

# Per-type parallel-trends method. The PT check dispatches internally on this.
# Values:
#   "two_x_two"      — uses utils.check_parallel_trends (requires ``data``)
#   "event_study"    — joint Wald on pre-period event-study coefficients
#   "hausman"        — EfficientDiD.hausman_pretest (native PT-All vs PT-Post)
#   "synthetic_fit"  — SDiD weighted pre-treatment fit (surfaces pre_treatment_fit)
#   "factor"         — TROP factor-model identification (no PT; renders "N/A" prose)
_PT_METHOD: Dict[str, str] = {
    "DiDResults": "two_x_two",
    "MultiPeriodDiDResults": "event_study",
    "CallawaySantAnnaResults": "event_study",
    "SunAbrahamResults": "event_study",
    "ImputationDiDResults": "event_study",
    "TwoStageDiDResults": "event_study",
    "StackedDiDResults": "event_study",
    "EfficientDiDResults": "hausman",
    "ContinuousDiDResults": "event_study",
    "StaggeredTripleDiffResults": "event_study",
    "WooldridgeDiDResults": "event_study",
    "ChaisemartinDHaultfoeuilleResults": "event_study",
    "SyntheticDiDResults": "synthetic_fit",
    "TROPResults": "factor",
}


@dataclass(frozen=True)
class DiagnosticReportResults:
    """Frozen container holding the outcome of a ``DiagnosticReport.run_all()`` call.

    Attributes
    ----------
    schema : dict
        The AI-legible structured schema (also returned by ``to_dict()``).
    interpretation : str
        The ``overall_interpretation`` paragraph synthesizing findings across
        checks.
    applicable_checks : tuple of str
        The names of checks that applied to this estimator + options.
    skipped_checks : dict of str -> str
        Mapping from skipped-check name to plain-English reason.
    warnings : tuple of str
        Warnings captured while running the underlying diagnostic functions.
    """

    schema: Dict[str, Any]
    interpretation: str
    applicable_checks: Tuple[str, ...]
    skipped_checks: Dict[str, str] = field(default_factory=dict)
    warnings: Tuple[str, ...] = ()


class DiagnosticReport:
    """Run the standard diff-diff diagnostic battery on a fitted result.

    Parameters
    ----------
    results : Any
        A fitted diff-diff results object (e.g. ``CallawaySantAnnaResults``,
        ``DiDResults``, ``SyntheticDiDResults``). Any of the 16 result types
        in the library is accepted.
    data : pandas.DataFrame, optional
        The underlying panel. Required for checks that need raw data
        (2x2 parallel-trends check on ``DiDResults``; Bacon-from-scratch when
        ``results`` is not itself a Bacon fit; the opt-in placebo battery).
    outcome, treatment, time, unit, first_treat : str, optional
        Column names identifying the panel structure.
    pre_periods, post_periods : list, optional
        Explicit pre- and post-treatment period labels.
    run_parallel_trends, run_sensitivity, run_placebo, run_bacon,
    run_design_effect, run_heterogeneity, run_epv, run_pretrends_power : bool
        Per-check opt-in flags. ``run_placebo`` defaults to ``False`` (opt-in,
        expensive, currently not implemented — placebo key remains reserved
        as ``skipped`` in the schema). All other checks default to ``True``
        and are further gated by estimator-type and instance-level
        applicability (see ``docs/methodology/REPORTING.md``).
    sensitivity_M_grid : tuple of float, default (0.5, 1.0, 1.5, 2.0)
        Grid of M values passed to ``HonestDiD.sensitivity``. Yields a
        ``SensitivityResults`` object with ``breakdown_M`` populated.
    sensitivity_method : str, default "relative_magnitude"
        HonestDiD restriction type.
    alpha : float, default 0.05
        Significance level used across checks.
    precomputed : dict, optional
        Map of check name to a pre-computed result object. Accepted keys
        (this is the full implemented list; unsupported keys raise
        ``ValueError``):

        - ``"parallel_trends"`` — a dict returned by
          ``utils.check_parallel_trends`` (adapted into the schema shape).
        - ``"sensitivity"`` — a ``SensitivityResults`` (grid) or
          ``HonestDiDResults`` (single-M) object; used verbatim and no
          ``HonestDiD.sensitivity_analysis`` call is made.
        - ``"pretrends_power"`` — a ``PreTrendsPowerResults`` object.
        - ``"bacon"`` — a ``BaconDecompositionResults`` object.

        Other sections (``design_effect``, ``heterogeneity``, ``epv``) are
        read directly from the fitted result object and do not currently
        accept precomputed values — there is no expensive call to bypass.
        ``placebo`` is reserved in the schema but opt-in / deferred in MVP.
    outcome_label, treatment_label : str, optional
        Plain-English labels used in prose rendering.
    """

    def __init__(
        self,
        results: Any,
        *,
        data: Optional[pd.DataFrame] = None,
        outcome: Optional[str] = None,
        treatment: Optional[str] = None,
        time: Optional[str] = None,
        unit: Optional[str] = None,
        first_treat: Optional[str] = None,
        pre_periods: Optional[List[Any]] = None,
        post_periods: Optional[List[Any]] = None,
        run_parallel_trends: bool = True,
        run_sensitivity: bool = True,
        run_placebo: bool = False,
        run_bacon: bool = True,
        run_design_effect: bool = True,
        run_heterogeneity: bool = True,
        run_epv: bool = True,
        run_pretrends_power: bool = True,
        sensitivity_M_grid: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
        sensitivity_method: str = "relative_magnitude",
        alpha: float = 0.05,
        precomputed: Optional[Dict[str, Any]] = None,
        outcome_label: Optional[str] = None,
        treatment_label: Optional[str] = None,
    ):
        self._results = results
        self._data = data
        self._outcome = outcome
        self._treatment = treatment
        self._time = time
        self._unit = unit
        self._first_treat = first_treat
        self._pre_periods = pre_periods
        self._post_periods = post_periods
        self._run_flags: Dict[str, bool] = {
            "parallel_trends": run_parallel_trends,
            "pretrends_power": run_pretrends_power,
            "sensitivity": run_sensitivity,
            "bacon": run_bacon,
            "design_effect": run_design_effect,
            "heterogeneity": run_heterogeneity,
            "epv": run_epv,
            "placebo": run_placebo,
            "estimator_native": True,
        }
        self._sensitivity_M_grid = tuple(sensitivity_M_grid)
        self._sensitivity_method = sensitivity_method
        self._alpha = float(alpha)
        self._precomputed = dict(precomputed or {})
        # Validate precomputed keys against the actually-implemented passthrough
        # set so advertised contracts do not silently diverge from behavior.
        _supported_precomputed = {"parallel_trends", "sensitivity", "pretrends_power", "bacon"}
        _unsupported = set(self._precomputed) - _supported_precomputed
        if _unsupported:
            raise ValueError(
                "precomputed= contains keys that are not implemented: "
                f"{sorted(_unsupported)}. Supported keys: "
                f"{sorted(_supported_precomputed)}. ``design_effect``, "
                "``heterogeneity``, and ``epv`` are read directly from the "
                "fitted result and do not accept precomputed overrides."
            )
        self._outcome_label = outcome_label
        self._treatment_label = treatment_label
        self._cached: Optional[DiagnosticReportResults] = None

    # -- Public API ---------------------------------------------------------

    def run_all(self) -> DiagnosticReportResults:
        """Run all applicable diagnostics. Idempotent; caches on first call."""
        if self._cached is None:
            self._cached = self._execute()
        return self._cached

    def to_dict(self) -> Dict[str, Any]:
        """Return the AI-legible structured schema."""
        return self.run_all().schema

    def summary(self) -> str:
        """Return a short plain-English paragraph."""
        return self.run_all().interpretation

    def full_report(self) -> str:
        """Return the multi-section markdown report."""
        return _render_dr_full_report(self.run_all())

    def export_markdown(self) -> str:
        """Alias for ``full_report()``."""
        return self.full_report()

    def to_dataframe(self) -> pd.DataFrame:
        """Return one row per check with status and headline metric."""
        schema = self.to_dict()
        rows = []
        for check in _CHECK_NAMES:
            section_key = "estimator_native_diagnostics" if check == "estimator_native" else check
            section = schema.get(section_key, {})
            rows.append(
                {
                    "check": check,
                    "status": section.get("status"),
                    "headline": _check_headline(check, section),
                    "reason": section.get("reason"),
                }
            )
        return pd.DataFrame(rows)

    @property
    def applicable_checks(self) -> Tuple[str, ...]:
        """Names of checks that will run, given estimator + instance + options.

        No compute is triggered; this reflects only the applicability matrix
        filtered by instance state (survey_metadata, epv_diagnostics, vcov)
        and the user's ``run_*`` flags.
        """
        return tuple(sorted(self._compute_applicable_checks()[0]))

    @property
    def skipped_checks(self) -> Dict[str, str]:
        """Mapping of skipped check -> plain-English reason. Requires ``run_all()``."""
        return dict(self.run_all().skipped_checks)

    # -- Implementation detail ---------------------------------------------

    def _compute_applicable_checks(self) -> Tuple[set, Dict[str, str]]:
        """Compute the applicable-check set + per-check skipped reasons.

        Returns
        -------
        applicable : set of str
            Checks that will run.
        skipped : dict
            Mapping from check name -> plain-English reason for any check
            that is type-applicable but skipped for this instance or by user
            opt-out. Checks that are not type-applicable for this estimator
            are omitted from both sets (not surfaced as "skipped").
        """
        type_name = type(self._results).__name__
        type_level = set(_APPLICABILITY.get(type_name, frozenset()))
        applicable: set = set()
        skipped: Dict[str, str] = {}

        for check in type_level:
            # Per-check user opt-out
            if not self._run_flags.get(check, True):
                skipped[check] = f"run_{check}=False (user opted out)"
                continue
            # Instance-level gating
            reason = self._instance_skip_reason(check)
            if reason is not None:
                skipped[check] = reason
                continue
            applicable.add(check)

        # Placebo is reserved for every result type in MVP so the schema
        # shape is stable: ``schema["placebo"]["status"] == "skipped"``
        # always holds regardless of estimator. The opt-in execution path
        # is deferred to a follow-up; ``REPORTING.md`` documents this.
        skipped.setdefault(
            "placebo",
            "Placebo battery runs on opt-in only; not yet implemented in MVP. "
            "Reserved in the schema for forward compatibility.",
        )

        return applicable, skipped

    def _instance_skip_reason(self, check: str) -> Optional[str]:
        """Return a plain-English reason this check cannot run on this instance, or None."""
        r = self._results
        name = type(r).__name__
        if check == "design_effect":
            if getattr(r, "survey_metadata", None) is None:
                return "No survey design attached to results.survey_metadata."
            return None
        if check == "epv":
            if getattr(r, "epv_diagnostics", None) is None:
                return "Estimator did not produce results.epv_diagnostics for this fit."
            return None
        if check == "parallel_trends":
            method = _PT_METHOD.get(name)
            if method == "two_x_two" and self._data is None:
                return (
                    "2x2 parallel-trends check needs raw panel data; "
                    "pass data=<DataFrame> with outcome / time / treatment columns."
                )
            if method == "event_study":
                pre_coefs = _collect_pre_period_coefs(r)
                if not pre_coefs:
                    return (
                        "No pre-period event-study coefficients are exposed on "
                        "this fit. For staggered estimators, re-fit with "
                        "aggregate='event_study' to populate event-study output."
                    )
                # vcov is optional for the Bonferroni fallback.
            return None
        if check == "pretrends_power":
            # ``compute_pretrends_power`` handles CS / SA / ImputationDiD
            # event-study results by reading ``event_study_effects``
            # directly, so we accept either a top-level ``vcov`` OR a
            # populated event-study surface. Precomputed overrides also
            # bypass this gate.
            if "pretrends_power" in self._precomputed:
                return None
            has_vcov = getattr(r, "vcov", None) is not None
            has_event_vcov = getattr(r, "event_study_vcov", None) is not None
            has_event_es = getattr(r, "event_study_effects", None) is not None
            if not (has_vcov or has_event_vcov or has_event_es):
                return (
                    "Pre-trends power needs either results.vcov or "
                    "event_study_effects (from aggregate='event_study' on "
                    "staggered estimators); neither available."
                )
            pre_coefs = _collect_pre_period_coefs(r)
            if len(pre_coefs) < 2:
                return "Pre-trends power needs >= 2 pre-treatment periods."
            return None
        if check == "sensitivity":
            # Native SDiD/TROP paths substitute for HonestDiD.
            if name in {"SyntheticDiDResults", "TROPResults"}:
                return None
            # Precomputed sensitivity always unlocks this check.
            if "sensitivity" in self._precomputed:
                return None
            # dCDH uses ``placebo_event_study`` as its pre-period surface,
            # which HonestDiD consumes via a dedicated branch. Accept the
            # fit when that attribute is populated.
            if name == "ChaisemartinDHaultfoeuilleResults":
                pes = getattr(r, "placebo_event_study", None)
                if pes is None:
                    return (
                        "HonestDiD on dCDH requires results.placebo_event_study "
                        "(re-fit with a placebo-producing configuration)."
                    )
                return None
            # MultiPeriod / CS path: ``HonestDiD.sensitivity_analysis``
            # consumes ``event_study_effects`` plus either ``vcov`` +
            # ``interaction_indices`` (MultiPeriod) or ``event_study_vcov``
            # + ``event_study_vcov_index`` (CS), with a per-SE diagonal
            # fallback otherwise.
            has_vcov = getattr(r, "vcov", None) is not None
            has_event_vcov = getattr(r, "event_study_vcov", None) is not None
            has_event_es = getattr(r, "event_study_effects", None) is not None
            if not (has_vcov or has_event_vcov or has_event_es):
                return (
                    "HonestDiD needs either results.vcov, event_study_vcov, "
                    "or event_study_effects; none available."
                )
            pre_coefs = _collect_pre_period_coefs(r)
            if len(pre_coefs) < 1:
                return "HonestDiD requires at least one pre-period coefficient."
            return None
        if check == "bacon":
            # Can run if results is itself Bacon, or if data + first_treat supplied.
            if name == "BaconDecompositionResults":
                return None
            if self._data is None or self._first_treat is None:
                return (
                    "Bacon decomposition needs panel data + first_treat column; "
                    "pass data=<DataFrame> and first_treat=<column name>."
                )
            return None
        if check == "heterogeneity":
            # Needs multiple group or event-study effects. Use len() rather than
            # truthiness because some estimators expose these as DataFrames,
            # which raise on bool() conversion.
            for attr in (
                "group_effects",
                "event_study_effects",
                "treatment_effects",  # TROP per-(unit, time)
                "group_time_effects",  # CS default aggregation
                "period_effects",  # MultiPeriod
            ):
                val = getattr(r, attr, None)
                if val is None:
                    continue
                try:
                    if len(val) > 0:
                        return None
                except TypeError:
                    continue
            return "No group/event-study effects available to compute heterogeneity."
        if check == "estimator_native":
            if name not in {"SyntheticDiDResults", "TROPResults"}:
                return f"{name} does not expose native validation methods."
            return None
        return None

    def _execute(self) -> DiagnosticReportResults:
        """Run the diagnostic battery and assemble the schema."""
        applicable, skipped = self._compute_applicable_checks()

        # Initialize all schema sections to either "ran"/"skipped"/"not_applicable".
        sections: Dict[str, Dict[str, Any]] = {}
        for check in _CHECK_NAMES:
            if check in applicable:
                sections[check] = {"status": "not_run", "reason": "pending implementation"}
            elif check in skipped:
                sections[check] = {"status": "skipped", "reason": skipped[check]}
            else:
                sections[check] = {
                    "status": "not_applicable",
                    "reason": f"{check} is not applicable to " f"{type(self._results).__name__}.",
                }

        # Run the checks that are applicable. Each returns a schema-section dict
        # that replaces the placeholder above.
        if "parallel_trends" in applicable:
            sections["parallel_trends"] = self._check_parallel_trends()
        if "pretrends_power" in applicable:
            sections["pretrends_power"] = self._check_pretrends_power()
        if "sensitivity" in applicable:
            sections["sensitivity"] = self._check_sensitivity()
        if "bacon" in applicable:
            sections["bacon"] = self._check_bacon()
        if "design_effect" in applicable:
            sections["design_effect"] = self._check_design_effect()
        if "heterogeneity" in applicable:
            sections["heterogeneity"] = self._check_heterogeneity()
        if "epv" in applicable:
            sections["epv"] = self._check_epv()
        if "estimator_native" in applicable:
            sections["estimator_native"] = self._check_estimator_native()

        # Estimator-native placeholder: SDiD/TROP diagnostics come in a later task.
        if "estimator_native" not in applicable and "estimator_native" not in skipped:
            sections["estimator_native"] = {
                "status": "not_applicable",
                "reason": f"{type(self._results).__name__} does not expose native "
                "validation methods beyond what's captured above.",
            }

        # Headline metric — best-effort across estimator types.
        headline = self._extract_headline_metric()

        # Pull suggested next steps from the practitioner workflow.
        next_steps = self._collect_next_steps(applicable)

        # Populate schema-level warnings for every section that ended in "error",
        # so users and agents do not have to scan each section dict to discover
        # that a diagnostic failed. Preserves provenance per the "no silent
        # failures" convention.
        top_warnings: List[str] = []
        for check in _CHECK_NAMES:
            section_key = "estimator_native" if check == "estimator_native" else check
            section = sections.get(section_key, {})
            if section.get("status") == "error":
                reason = section.get("reason") or "diagnostic raised an exception"
                top_warnings.append(f"{check}: {reason}")

        schema: Dict[str, Any] = {
            "schema_version": DIAGNOSTIC_REPORT_SCHEMA_VERSION,
            "estimator": type(self._results).__name__,
            "headline_metric": headline,
            "parallel_trends": sections["parallel_trends"],
            "pretrends_power": sections["pretrends_power"],
            "sensitivity": sections["sensitivity"],
            "placebo": sections["placebo"],
            "bacon": sections["bacon"],
            "design_effect": sections["design_effect"],
            "heterogeneity": sections["heterogeneity"],
            "epv": sections["epv"],
            "estimator_native_diagnostics": sections["estimator_native"],
            "skipped": {k: v for k, v in skipped.items()},
            "warnings": top_warnings,
            "overall_interpretation": "",
            "next_steps": next_steps,
        }
        interpretation = _render_overall_interpretation(schema, self._context_labels())
        schema["overall_interpretation"] = interpretation

        return DiagnosticReportResults(
            schema=schema,
            interpretation=interpretation,
            applicable_checks=tuple(sorted(applicable)),
            skipped_checks=skipped,
            warnings=tuple(top_warnings),
        )

    def _context_labels(self) -> Dict[str, str]:
        """Return plain-English labels used in prose rendering."""
        return {
            "outcome_label": self._outcome_label or "the outcome",
            "treatment_label": self._treatment_label or "the treatment",
        }

    def _collect_next_steps(self, applicable: set) -> List[Dict[str, Any]]:
        """Pull and filter practitioner_next_steps, marking DR-covered steps complete."""
        try:
            from diff_diff.practitioner import practitioner_next_steps

            completed = []
            if "parallel_trends" in applicable:
                completed.append("parallel_trends")
            if "sensitivity" in applicable:
                completed.append("sensitivity")
            if "heterogeneity" in applicable:
                completed.append("heterogeneity")
            ns = practitioner_next_steps(
                self._results,
                completed_steps=completed,
                verbose=False,
            )
            return [
                {
                    "label": s.get("label"),
                    "why": s.get("why"),
                    "code": s.get("code"),
                    "priority": s.get("priority"),
                    "baker_step": s.get("baker_step"),
                }
                for s in ns.get("next_steps", [])[:5]
            ]
        except Exception:  # noqa: BLE001
            return []

    # -- Per-check runners --------------------------------------------------

    def _check_parallel_trends(self) -> Dict[str, Any]:
        """Run the parallel-trends check. Dispatches on PT method for this type."""
        if "parallel_trends" in self._precomputed:
            return self._format_precomputed_pt(self._precomputed["parallel_trends"])

        method = _PT_METHOD.get(type(self._results).__name__)
        if method == "two_x_two":
            return self._pt_two_x_two()
        if method == "event_study":
            return self._pt_event_study()
        if method == "hausman":
            return self._pt_hausman()
        if method == "synthetic_fit":
            return self._pt_synthetic_fit()
        if method == "factor":
            return self._pt_factor()
        return {
            "status": "not_applicable",
            "reason": f"No parallel-trends method registered for "
            f"{type(self._results).__name__}.",
        }

    def _pt_two_x_two(self) -> Dict[str, Any]:
        """Simple two-period PT check via ``utils.check_parallel_trends``."""
        from diff_diff.utils import check_parallel_trends

        if self._data is None or self._outcome is None or self._time is None:
            return {
                "status": "skipped",
                "reason": "Requires data=, outcome=, time=, and a treatment-group "
                "column; not supplied.",
            }
        treatment_group = self._treatment
        if treatment_group is None:
            return {
                "status": "skipped",
                "reason": "Requires treatment=<column name> identifying the "
                "treated-group indicator; not supplied.",
            }
        try:
            raw = check_parallel_trends(
                self._data,
                outcome=self._outcome,
                time=self._time,
                treatment_group=treatment_group,
                pre_periods=self._pre_periods,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "reason": f"check_parallel_trends raised {type(exc).__name__}: {exc}",
            }
        p_value = _to_python_float(raw.get("p_value"))
        return {
            "status": "ran",
            "method": "slope_difference",
            "joint_p_value": p_value,
            "treated_trend": _to_python_float(raw.get("treated_trend")),
            "control_trend": _to_python_float(raw.get("control_trend")),
            "trend_difference": _to_python_float(raw.get("trend_difference")),
            "t_statistic": _to_python_float(raw.get("t_statistic")),
            "verdict": _pt_verdict(p_value),
        }

    def _pt_event_study(self) -> Dict[str, Any]:
        """Event-study joint Wald (or Bonferroni fallback) on pre-period coefficients.

        Works with either ``pre_period_effects`` (``MultiPeriodDiDResults`` style,
        dict of ``PeriodEffect`` objects) or ``event_study_effects`` (CS / SA /
        ImputationDiD style, dict of dicts with ``effect``/``se``/``p_value`` keys).
        """
        r = self._results
        pre_coefs = _collect_pre_period_coefs(r)
        if not pre_coefs:
            return {
                "status": "skipped",
                "reason": "No pre-period event-study coefficients available.",
            }
        interaction_indices = getattr(r, "interaction_indices", None)
        vcov = getattr(r, "vcov", None)

        # pre_coefs is a sorted list of (key, effect, se, p_value) tuples.
        per_period = [
            {
                "period": _to_python_scalar(k),
                "coef": _to_python_float(eff),
                "se": _to_python_float(se),
                "p_value": _to_python_float(p),
            }
            for (k, eff, se, p) in pre_coefs
        ]

        joint_p: Optional[float] = None
        test_statistic: Optional[float] = None
        df = len(pre_coefs)
        method = "bonferroni"
        # Joint-Wald pathway is taken only when EVERY pre-period key is present
        # in the relevant index mapping (required len == df guard below). This
        # protects against estimators whose event-study keys use a different
        # namespace than the vcov indexing: if any key is missing, we fall back
        # to Bonferroni rather than risk indexing into the wrong vcov rows.
        # The schema's ``method`` field exposes which path ran so agents and
        # tests can distinguish the two unambiguously.
        #
        # Two covariance sources are supported:
        #   1. ``interaction_indices`` + ``vcov`` — the MultiPeriodDiDResults
        #      convention, where ``vcov`` is the full regression covariance
        #      matrix and ``interaction_indices`` maps period labels to rows.
        #   2. ``event_study_vcov_index`` + ``event_study_vcov`` — the
        #      CallawaySantAnnaResults convention, where the event-study
        #      covariance is stored separately from the full regression vcov.
        vcov_for_wald: Optional[Any] = None
        idx_map_for_wald: Optional[Any] = None
        vcov_method_tag = "joint_wald"
        if vcov is not None and interaction_indices is not None:
            vcov_for_wald = vcov
            idx_map_for_wald = interaction_indices
        else:
            es_vcov = getattr(r, "event_study_vcov", None)
            es_vcov_index = getattr(r, "event_study_vcov_index", None)
            if es_vcov is not None and es_vcov_index is not None:
                vcov_for_wald = es_vcov
                # ``event_study_vcov_index`` is an ordered list of relative-time
                # keys; convert it into a dict mapping key -> position.
                try:
                    idx_map_for_wald = {k: i for i, k in enumerate(es_vcov_index)}
                    vcov_method_tag = "joint_wald_event_study"
                except TypeError:
                    idx_map_for_wald = None
        if vcov_for_wald is not None and idx_map_for_wald is not None and df > 0:
            try:
                keys_in_vcov = [k for (k, _, _, _) in pre_coefs if k in idx_map_for_wald]
                if len(keys_in_vcov) == df:
                    idx = [idx_map_for_wald[k] for k in keys_in_vcov]
                    beta_map = {k: eff for (k, eff, _, _) in pre_coefs}
                    beta = np.array([beta_map[k] for k in keys_in_vcov], dtype=float)
                    v_sub = np.asarray(vcov_for_wald)[np.ix_(idx, idx)]
                    stat = float(beta @ np.linalg.solve(v_sub, beta))
                    from scipy.stats import chi2

                    joint_p = float(1.0 - chi2.cdf(stat, df=df))
                    test_statistic = stat
                    method = vcov_method_tag
            except Exception:  # noqa: BLE001
                joint_p = None
                test_statistic = None
                method = "bonferroni"

        if joint_p is None:
            # Bonferroni: min per-period p-value scaled by count, capped at 1.
            # NaN p-values are excluded — a non-finite p-value means the
            # per-period test was undefined (zero SE, reference marker that
            # slipped through, etc.) and must not be treated as clean
            # evidence. If no valid p-values remain, joint_p stays None and
            # the verdict will be ``inconclusive``.
            ps = [
                p["p_value"]
                for p in per_period
                if isinstance(p["p_value"], (int, float)) and np.isfinite(p["p_value"])
            ]
            if ps:
                joint_p = min(1.0, min(ps) * len(ps))

        return {
            "status": "ran",
            "method": method,
            "joint_p_value": joint_p,
            "test_statistic": test_statistic,
            "df": df,
            "n_pre_periods": df,
            "per_period": per_period,
            "verdict": _pt_verdict(joint_p),
        }

    def _check_pretrends_power(self) -> Dict[str, Any]:
        """Compute pre-trends power (MDV) via ``compute_pretrends_power``.

        Feeds the ``mdv_share_of_att`` ratio used by ``BusinessReport`` to select
        the power-aware phrasing tier for the ``no_detected_violation`` verdict.
        """
        if "pretrends_power" in self._precomputed:
            return self._format_precomputed_pretrends_power(self._precomputed["pretrends_power"])

        from diff_diff.pretrends import compute_pretrends_power

        try:
            pp = compute_pretrends_power(
                self._results,
                alpha=self._alpha,
                target_power=0.80,
                violation_type="linear",
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "reason": f"compute_pretrends_power raised " f"{type(exc).__name__}: {exc}",
            }

        # Build the schema section and compute the MDV/|ATT| ratio for BR.
        headline_metric = self._extract_headline_metric()
        att = headline_metric.get("value") if headline_metric else None
        mdv = _to_python_float(getattr(pp, "mdv", None))
        ratio: Optional[float] = None
        if (
            mdv is not None
            and att is not None
            and np.isfinite(att)
            and abs(att) > 0
            and np.isfinite(mdv)
        ):
            ratio = mdv / abs(att)

        # Annotate whether ``compute_pretrends_power`` had access to the full
        # pre-period covariance (CS / SA / ImputationDiD currently fall back to
        # ``np.diag(ses**2)`` inside ``pretrends.py``, even when
        # ``event_study_vcov`` is available). BR uses this field to downgrade
        # power-tier prose when only the diagonal approximation was used.
        r = self._results
        has_full_es_vcov = (
            getattr(r, "event_study_vcov", None) is not None
            and getattr(r, "event_study_vcov_index", None) is not None
        )
        is_event_study_type = type(r).__name__ in {
            "CallawaySantAnnaResults",
            "SunAbrahamResults",
            "ImputationDiDResults",
            "StackedDiDResults",
            "StaggeredTripleDiffResults",
            "WooldridgeDiDResults",
            "ChaisemartinDHaultfoeuilleResults",
            "EfficientDiDResults",
            "TwoStageDiDResults",
        }
        if is_event_study_type and has_full_es_vcov:
            # ``compute_pretrends_power`` does not currently consume
            # ``event_study_vcov`` for these result types (see the reviewer's
            # note on pretrends.py). Flag the diagonal fallback explicitly so
            # the prose layer can hedge.
            cov_source = "diag_fallback_available_full_vcov_unused"
        elif is_event_study_type:
            cov_source = "diag_fallback"
        else:
            cov_source = "full_pre_period_vcov"

        tier = _power_tier(ratio)
        return {
            "status": "ran",
            "method": "compute_pretrends_power",
            "violation_type": getattr(pp, "violation_type", "linear"),
            "alpha": _to_python_float(getattr(pp, "alpha", self._alpha)),
            "target_power": _to_python_float(getattr(pp, "target_power", 0.80)),
            "mdv": mdv,
            "mdv_share_of_att": ratio,
            "power_at_M_1": _to_python_float(getattr(pp, "power", None)),
            "n_pre_periods": int(getattr(pp, "n_pre_periods", 0) or 0),
            "tier": tier,
            "covariance_source": cov_source,
        }

    def _format_precomputed_pretrends_power(self, obj: Any) -> Dict[str, Any]:
        """Adapt a pre-computed ``PreTrendsPowerResults`` to the schema shape."""
        mdv = _to_python_float(getattr(obj, "mdv", None))
        hm = self._extract_headline_metric()
        att = hm.get("value") if hm else None
        ratio: Optional[float] = None
        if mdv is not None and att is not None and np.isfinite(att) and abs(att) > 0:
            ratio = mdv / abs(att)
        return {
            "status": "ran",
            "method": "precomputed",
            "violation_type": getattr(obj, "violation_type", "linear"),
            "alpha": _to_python_float(getattr(obj, "alpha", self._alpha)),
            "target_power": _to_python_float(getattr(obj, "target_power", 0.80)),
            "mdv": mdv,
            "mdv_share_of_att": ratio,
            "power_at_M_1": _to_python_float(getattr(obj, "power", None)),
            "n_pre_periods": int(getattr(obj, "n_pre_periods", 0) or 0),
            "tier": _power_tier(ratio),
            "precomputed": True,
        }

    def _check_sensitivity(self) -> Dict[str, Any]:
        """Run HonestDiD over the M grid. Uses ``SensitivityResults.breakdown_M``.

        The standard path calls ``HonestDiD(method=..., M_grid=...).sensitivity_analysis()``.
        SDiD and TROP route to estimator-native sensitivity in
        ``estimator_native_diagnostics`` and emit a pointer here.
        """
        if "sensitivity" in self._precomputed:
            return self._format_precomputed_sensitivity(self._precomputed["sensitivity"])

        name = type(self._results).__name__
        if name in {"SyntheticDiDResults", "TROPResults"}:
            return {
                "status": "skipped",
                "reason": "Estimator uses native sensitivity (see "
                "estimator_native_diagnostics).",
                "method": "estimator_native",
            }

        try:
            from typing import cast

            from diff_diff.honest_did import HonestDiD

            # The sensitivity_method string is validated at runtime by
            # HonestDiD; the Literal annotation is for static typing only.
            honest = HonestDiD(
                method=cast(Any, self._sensitivity_method),
                alpha=self._alpha,
            )
            sens = honest.sensitivity_analysis(
                self._results,
                M_grid=list(self._sensitivity_M_grid),
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "method": self._sensitivity_method,
                "reason": f"HonestDiD.sensitivity_analysis raised " f"{type(exc).__name__}: {exc}",
            }

        return self._format_sensitivity_results(sens)

    def _format_sensitivity_results(self, sens: Any) -> Dict[str, Any]:
        grid = []
        raw_M = getattr(sens, "M_values", None)
        raw_cis = getattr(sens, "robust_cis", None)
        raw_bounds = getattr(sens, "bounds", None)
        M_values: List[Any] = list(raw_M) if raw_M is not None else []
        cis: List[Any] = list(raw_cis) if raw_cis is not None else []
        bounds: List[Any] = list(raw_bounds) if raw_bounds is not None else []
        for i, M in enumerate(M_values):
            ci = cis[i] if i < len(cis) else (None, None)
            bd = bounds[i] if i < len(bounds) else (None, None)
            lo = _to_python_float(ci[0])
            hi = _to_python_float(ci[1])
            robust_to_zero = lo is not None and hi is not None and (lo > 0 or hi < 0)
            grid.append(
                {
                    "M": _to_python_float(M),
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "bound_lower": _to_python_float(bd[0]),
                    "bound_upper": _to_python_float(bd[1]),
                    "robust_to_zero": robust_to_zero,
                }
            )
        bkd = _to_python_float(getattr(sens, "breakdown_M", None))
        if bkd is None:
            conclusion = "robust_over_grid"
        elif bkd >= 1.0:
            conclusion = f"robust_to_M_{bkd:.2f}"
        else:
            conclusion = "fragile"
        return {
            "status": "ran",
            "method": getattr(sens, "method", self._sensitivity_method),
            "grid": grid,
            "breakdown_M": bkd,
            "original_estimate": _to_python_float(getattr(sens, "original_estimate", None)),
            "original_se": _to_python_float(getattr(sens, "original_se", None)),
            "conclusion": conclusion,
        }

    def _format_precomputed_sensitivity(self, obj: Any) -> Dict[str, Any]:
        """Accept either ``SensitivityResults`` (grid) or ``HonestDiDResults`` (single M)."""
        if hasattr(obj, "M_values") and hasattr(obj, "breakdown_M"):
            formatted = self._format_sensitivity_results(obj)
            formatted["precomputed"] = True
            return formatted
        # Single-M HonestDiDResults: adapt with no breakdown_M.
        ci_lb = _to_python_float(getattr(obj, "ci_lb", None))
        ci_ub = _to_python_float(getattr(obj, "ci_ub", None))
        return {
            "status": "ran",
            "method": getattr(obj, "method", self._sensitivity_method),
            "grid": [
                {
                    "M": _to_python_float(getattr(obj, "M", None)),
                    "ci_lower": ci_lb,
                    "ci_upper": ci_ub,
                    "bound_lower": _to_python_float(getattr(obj, "lb", None)),
                    "bound_upper": _to_python_float(getattr(obj, "ub", None)),
                    "robust_to_zero": (
                        ci_lb is not None and ci_ub is not None and (ci_lb > 0 or ci_ub < 0)
                    ),
                }
            ],
            "breakdown_M": None,
            "conclusion": "single_M_precomputed",
            "precomputed": True,
        }

    def _check_bacon(self) -> Dict[str, Any]:
        """Surface Bacon decomposition: read-out when applicable, else skip.

        If ``results`` is itself a ``BaconDecompositionResults``, read fields.
        If ``data`` + ``first_treat`` are supplied, call ``bacon_decompose``.
        Otherwise, skip with a helpful reason.
        """
        if "bacon" in self._precomputed:
            return self._format_bacon(self._precomputed["bacon"])

        r = self._results
        name = type(r).__name__
        if name == "BaconDecompositionResults":
            return self._format_bacon(r)

        data = self._data
        outcome = self._outcome
        unit = self._unit
        time = self._time
        first_treat = self._first_treat
        if data is None or outcome is None or unit is None or time is None or first_treat is None:
            return {
                "status": "skipped",
                "reason": "Bacon decomposition requires data + outcome + unit + time "
                "+ first_treat on DiagnosticReport; not all supplied.",
            }

        try:
            from diff_diff.bacon import bacon_decompose

            bacon = bacon_decompose(
                data,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "reason": f"bacon_decompose raised {type(exc).__name__}: {exc}",
            }
        return self._format_bacon(bacon)

    def _format_bacon(self, bacon: Any) -> Dict[str, Any]:
        treated_vs_never = _to_python_float(getattr(bacon, "total_weight_treated_vs_never", None))
        earlier_vs_later = _to_python_float(getattr(bacon, "total_weight_earlier_vs_later", None))
        later_vs_earlier = _to_python_float(getattr(bacon, "total_weight_later_vs_earlier", None))
        twfe = _to_python_float(getattr(bacon, "twfe_estimate", None))
        forbidden = later_vs_earlier if later_vs_earlier is not None else 0.0
        if forbidden > 0.10:
            verdict = "materially_contaminated"
        elif forbidden > 0.01:
            verdict = "minor_forbidden_weight"
        else:
            verdict = "clean"
        return {
            "status": "ran",
            "twfe_estimate": twfe,
            "weight_by_type": {
                "treated_vs_never": treated_vs_never,
                "earlier_vs_later": earlier_vs_later,
                "later_vs_earlier": later_vs_earlier,
            },
            "forbidden_weight": later_vs_earlier,
            "verdict": verdict,
            "n_timing_groups": _to_python_scalar(getattr(bacon, "n_timing_groups", None)),
        }

    def _check_design_effect(self) -> Dict[str, Any]:
        """Read survey design-effect from ``results.survey_metadata``."""
        sm = getattr(self._results, "survey_metadata", None)
        if sm is None:
            return {
                "status": "skipped",
                "reason": "No survey_metadata attached to results.",
            }
        deff = _to_python_float(getattr(sm, "design_effect", None))
        eff_n = _to_python_float(getattr(sm, "effective_n", None))
        is_trivial = deff is not None and 0.95 <= deff <= 1.05
        return {
            "status": "ran",
            "deff": deff,
            "effective_n": eff_n,
            "weight_type": getattr(sm, "weight_type", None),
            "n_strata": _to_python_scalar(getattr(sm, "n_strata", None)),
            "n_psu": _to_python_scalar(getattr(sm, "n_psu", None)),
            "df_survey": _to_python_scalar(getattr(sm, "df_survey", None)),
            "replicate_method": getattr(sm, "replicate_method", None),
            "is_trivial": is_trivial,
        }

    def _check_heterogeneity(self) -> Dict[str, Any]:
        """Compute effect-stability metrics (CV, range, sign consistency)."""
        effects = self._collect_effect_scalars()
        if not effects:
            return {
                "status": "skipped",
                "reason": "No group / event-study / period effects available.",
            }
        vals = np.array(effects, dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            return {
                "status": "skipped",
                "reason": "All effect values are non-finite.",
            }
        mean = float(np.mean(finite))
        sd = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        mn = float(np.min(finite))
        mx = float(np.max(finite))
        cv = sd / abs(mean) if abs(mean) > 0.1 * sd and abs(mean) > 0 else None
        sign_consistent = bool(np.all(finite >= 0) or np.all(finite <= 0))
        return {
            "status": "ran",
            "source": self._heterogeneity_source(),
            "n_effects": int(finite.size),
            "min": mn,
            "max": mx,
            "mean": mean,
            "sd": sd,
            "range": mx - mn,
            "cv": cv,
            "sign_consistent": sign_consistent,
        }

    def _check_epv(self) -> Dict[str, Any]:
        """Read EPV diagnostics from ``results.epv_diagnostics``.

        The diff-diff convention (see ``diff_diff/staggered.py`` around the
        low-EPV summary warning) is that ``epv_diagnostics`` is a dict keyed
        by cell identifier (e.g. ``(g, t)`` for staggered) whose values are
        per-cell dicts with ``is_low`` (bool) and ``epv`` (float). The
        threshold lives on ``results.epv_threshold`` (default 10) rather
        than being hardcoded.
        """
        r = self._results
        epv = getattr(r, "epv_diagnostics", None)
        if epv is None:
            return {
                "status": "skipped",
                "reason": "Estimator did not produce results.epv_diagnostics for this fit.",
            }
        threshold = _to_python_float(getattr(r, "epv_threshold", 10)) or 10.0

        if isinstance(epv, dict):
            low_cells = [k for k, v in epv.items() if isinstance(v, dict) and v.get("is_low")]
            epv_floats: List[float] = []
            for v in epv.values():
                if not isinstance(v, dict):
                    continue
                raw = v.get("epv")
                if raw is None:
                    continue
                converted = _to_python_float(raw)
                if converted is not None:
                    epv_floats.append(converted)
            min_epv: Optional[float] = min(epv_floats) if epv_floats else None
            return {
                "status": "ran",
                "threshold": threshold,
                "n_cells_low": len(low_cells),
                "n_cells_total": len(epv),
                "min_epv": min_epv,
                "affected_cohorts": [_to_python_scalar(c) for c in low_cells],
            }

        # Legacy object-shaped fallback (not currently emitted by the library
        # but kept so custom subclasses that mirror the old shape still work).
        low_cells_attr = getattr(epv, "low_epv_cells", None) or []
        return {
            "status": "ran",
            "threshold": threshold,
            "n_cells_low": int(len(low_cells_attr)),
            "n_cells_total": _to_python_scalar(getattr(epv, "n_cells_total", None)),
            "min_epv": _to_python_float(getattr(epv, "min_epv", None)),
            "affected_cohorts": [_to_python_scalar(c) for c in low_cells_attr],
        }

    def _check_estimator_native(self) -> Dict[str, Any]:
        """SDiD / TROP native validation surfaces.

        SDiD: ``pre_treatment_fit`` (weighted-PT analogue), weight
        concentration (``get_weight_concentration``), ``in_time_placebo``
        (placebo-timing sweep), and ``sensitivity_to_zeta_omega``
        (regularization sensitivity).

        TROP: factor-model fit metrics (``effective_rank``, ``loocv_score``,
        selected ``lambda_*``).
        """
        r = self._results
        name = type(r).__name__
        if name == "SyntheticDiDResults":
            return self._sdid_native(r)
        if name == "TROPResults":
            return self._trop_native(r)
        return {
            "status": "not_applicable",
            "reason": f"{name} does not expose native validation methods.",
        }

    def _sdid_native(self, r: Any) -> Dict[str, Any]:
        """Populate SDiD-native diagnostics section."""
        out: Dict[str, Any] = {"status": "ran", "estimator": "SyntheticDiD"}
        out["pre_treatment_fit"] = _to_python_float(getattr(r, "pre_treatment_fit", None))
        # Weight concentration via the public method on SyntheticDiDResults.
        try:
            wc = r.get_weight_concentration(top_k=5)
            out["weight_concentration"] = {
                "effective_n": _to_python_float(wc.get("effective_n")),
                "herfindahl": _to_python_float(wc.get("herfindahl")),
                "top_k": _to_python_scalar(wc.get("top_k")),
                "top_k_share": _to_python_float(wc.get("top_k_share")),
            }
        except Exception as exc:  # noqa: BLE001
            out["weight_concentration"] = {
                "status": "error",
                "reason": f"get_weight_concentration raised " f"{type(exc).__name__}: {exc}",
            }
        # In-time placebo — runs only when the fit snapshot is available.
        try:
            placebo_df = r.in_time_placebo()
            out["in_time_placebo"] = {
                "n_placebos": int(len(placebo_df)),
                "max_abs_effect": _to_python_float(
                    placebo_df["att"].abs().max() if len(placebo_df) > 0 else None
                ),
                "mean_abs_effect": _to_python_float(
                    placebo_df["att"].abs().mean() if len(placebo_df) > 0 else None
                ),
            }
        except Exception as exc:  # noqa: BLE001
            out["in_time_placebo"] = {
                "status": "skipped",
                "reason": f"in_time_placebo unavailable: " f"{type(exc).__name__}: {exc}",
            }
        # Zeta-omega sensitivity.
        try:
            zeta_df = r.sensitivity_to_zeta_omega()
            atts = zeta_df["att"].astype(float).tolist() if len(zeta_df) > 0 else []
            out["zeta_sensitivity"] = {
                "grid": [
                    {
                        "multiplier": _to_python_float(row.get("multiplier")),
                        "att": _to_python_float(row.get("att")),
                        "pre_fit_rmse": _to_python_float(row.get("pre_fit_rmse")),
                        "effective_n": _to_python_float(row.get("effective_n")),
                    }
                    for row in zeta_df.to_dict(orient="records")
                ],
                "att_range": ([min(atts), max(atts)] if atts else None),
            }
        except Exception as exc:  # noqa: BLE001
            out["zeta_sensitivity"] = {
                "status": "skipped",
                "reason": f"sensitivity_to_zeta_omega unavailable: " f"{type(exc).__name__}: {exc}",
            }
        return out

    def _trop_native(self, r: Any) -> Dict[str, Any]:
        """Populate TROP-native factor-model diagnostics section."""
        return {
            "status": "ran",
            "estimator": "TROP",
            "factor_model": {
                "effective_rank": _to_python_float(getattr(r, "effective_rank", None)),
                "loocv_score": _to_python_float(getattr(r, "loocv_score", None)),
                "lambda_time": _to_python_float(getattr(r, "lambda_time", None)),
                "lambda_unit": _to_python_float(getattr(r, "lambda_unit", None)),
                "lambda_nn": _to_python_float(getattr(r, "lambda_nn", None)),
                "n_pre_periods": _to_python_scalar(getattr(r, "n_pre_periods", None)),
                "n_post_periods": _to_python_scalar(getattr(r, "n_post_periods", None)),
            },
        }

    # -- Heterogeneity helpers --------------------------------------------

    def _collect_effect_scalars(self) -> List[float]:
        """Collect scalar effect values across group / event-study / TROP sources.

        Returns an empty list if no recognized effect container is present.
        Never raises on unexpected shapes; unrecognized entries are skipped.
        """
        r = self._results
        # 1. group_effects: dict keyed by cohort -> dict with 'effect' or float
        ge = getattr(r, "group_effects", None)
        if ge is not None:
            return self._scalars_from_mapping(ge)
        # 2. event_study_effects: dict keyed by relative time -> dict with 'effect'
        es = getattr(r, "event_study_effects", None)
        if es is not None:
            return self._scalars_from_mapping(es)
        # 3. TROP: treatment_effects dict keyed by (unit, time) -> float
        te = getattr(r, "treatment_effects", None)
        if te is not None:
            return self._scalars_from_mapping(te)
        # 4. CS default: group_time_effects dict keyed by (g, t) -> dict
        gte = getattr(r, "group_time_effects", None)
        if gte is not None:
            return self._scalars_from_mapping(gte)
        # 5. MultiPeriod: period_effects dict keyed by period -> PeriodEffect
        pe = getattr(r, "period_effects", None)
        if pe is not None:
            return self._scalars_from_mapping(pe)
        return []

    @staticmethod
    def _scalars_from_mapping(mapping: Any) -> List[float]:
        """Extract scalar effect values from various result-mapping shapes."""
        out: List[float] = []
        values: List[Any]
        values_fn = getattr(mapping, "values", None)
        if callable(values_fn):
            try:
                values = list(values_fn())
            except Exception:  # noqa: BLE001
                return []
        else:
            try:
                values = list(mapping)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                return []
        for val in values:
            eff = _extract_scalar_effect(val)
            if eff is not None:
                out.append(eff)
        return out

    def _heterogeneity_source(self) -> str:
        """Name the attribute that produced the scalars (for the schema)."""
        for attr in (
            "group_effects",
            "event_study_effects",
            "treatment_effects",
            "group_time_effects",
            "period_effects",
        ):
            if getattr(self._results, attr, None) is not None:
                return attr
        return "unknown"

    def _pt_hausman(self) -> Dict[str, Any]:
        """EfficientDiD native PT check via ``EfficientDiD.hausman_pretest``.

        This is the correct PT check for EfficientDiD (PT-All vs PT-Post); the
        generic event-study approach is inappropriate for this estimator per
        ``practitioner._parallel_trends_step`` guidance.
        """
        data = self._data
        outcome = self._outcome
        unit = self._unit
        time = self._time
        first_treat = self._first_treat
        missing = [
            name
            for name, val in (
                ("data", data),
                ("outcome", outcome),
                ("unit", unit),
                ("time", time),
                ("first_treat", first_treat),
            )
            if val is None
        ]
        if (
            missing
            or data is None
            or outcome is None
            or unit is None
            or time is None
            or first_treat is None
        ):
            return {
                "status": "skipped",
                "method": "hausman",
                "reason": (
                    "EfficientDiD.hausman_pretest requires data + outcome + unit + "
                    f"time + first_treat kwargs on DiagnosticReport; missing: "
                    f"{', '.join(missing)}."
                ),
            }

        try:
            from diff_diff.efficient_did import EfficientDiD

            pt = EfficientDiD.hausman_pretest(
                data,
                outcome=outcome,
                unit=unit,
                time=time,
                first_treat=first_treat,
                alpha=self._alpha,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "method": "hausman",
                "reason": f"hausman_pretest raised {type(exc).__name__}: {exc}",
            }

        p_value = _to_python_float(getattr(pt, "p_value", None))
        return {
            "status": "ran",
            "method": "hausman",
            "joint_p_value": p_value,
            "test_statistic": _to_python_float(getattr(pt, "test_statistic", None)),
            "df": _to_python_scalar(getattr(pt, "df", None)),
            "verdict": _pt_verdict(p_value),
        }

    def _pt_synthetic_fit(self) -> Dict[str, Any]:
        """SDiD weighted pre-treatment-fit PT analogue.

        SDiD's design-enforced fit quality substitutes for a standard PT test:
        the synthetic control is explicitly constructed to match the treated
        group's pre-period trajectory, so small ``pre_treatment_fit`` RMSE
        means the weighted-PT analogue is satisfied.
        """
        r = self._results
        fit = _to_python_float(getattr(r, "pre_treatment_fit", None))
        if fit is None:
            return {
                "status": "skipped",
                "method": "synthetic_fit",
                "reason": "SyntheticDiDResults.pre_treatment_fit is not populated " "on this fit.",
            }
        # Proxy verdict: unlike a classical PT p-value, this is a fit-quality
        # metric. Classify conservatively — phrasing in BR will explain that
        # this is SDiD's design-enforced analogue, not a PT hypothesis test.
        return {
            "status": "ran",
            "method": "synthetic_fit",
            "pre_treatment_fit_rmse": fit,
            "verdict": "design_enforced_pt",
        }

    def _pt_factor(self) -> Dict[str, Any]:
        """TROP has no PT concept — its identification is factor-model-based."""
        return {
            "status": "not_applicable",
            "reason": "TROP uses factor-model identification; parallel trends is "
            "not applicable. See estimator_native_diagnostics for the "
            "factor-model fit metrics.",
            "method": "factor",
        }

    def _format_precomputed_pt(self, obj: Any) -> Dict[str, Any]:
        """Adapt a pre-computed PT result (from utils.check_parallel_trends) to schema shape."""
        if not isinstance(obj, dict):
            return {
                "status": "error",
                "reason": "precomputed['parallel_trends'] must be a dict returned by "
                "check_parallel_trends or compatible shape.",
            }
        p_value = _to_python_float(obj.get("p_value"))
        return {
            "status": "ran",
            "method": obj.get("method", "precomputed"),
            "joint_p_value": p_value,
            "verdict": _pt_verdict(p_value),
            "precomputed": True,
        }

    # -- Headline metric extraction ----------------------------------------

    def _extract_headline_metric(self) -> Optional[Dict[str, Any]]:
        """Best-effort extraction of the scalar headline metric from the result."""
        extracted = _extract_scalar_headline(self._results, fallback_alpha=self._alpha)
        if extracted is None:
            return None
        name, value, se, p, ci, alpha = extracted
        return {
            "name": name,
            "value": value,
            "se": se,
            "p_value": p,
            "conf_int": ci,
            "alpha": alpha,
        }


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------
def _extract_scalar_headline(
    results: Any,
    fallback_alpha: float = 0.05,
) -> Optional[
    Tuple[
        str,
        Optional[float],
        Optional[float],
        Optional[float],
        Optional[List[float]],
        Optional[float],
    ]
]:
    """Extract ``(name, value, se, p_value, conf_int, alpha)`` from a fitted result.

    Centralizes the scalar-headline mapping shared by both ``BusinessReport``
    and ``DiagnosticReport`` so schema drift (e.g. ``ContinuousDiDResults``
    using ``overall_att_se`` / ``overall_att_p_value`` /
    ``overall_att_conf_int`` instead of the ``overall_att`` stem) is handled
    in one place.

    Each row in the attribute-alias table below is tried in priority order.
    The first point-estimate attribute that resolves to a non-None value
    wins; the companion SE / p-value / CI attributes are then resolved from
    the same row, taking the first alias that exists on the result object.
    """
    # (name, [se aliases], [p-value aliases], [ci aliases])
    alias_table: List[Tuple[str, List[str], List[str], List[str]]] = [
        # Staggered / multi-period aggregations
        (
            "overall_att",
            ["overall_se", "overall_att_se"],
            ["overall_p_value", "overall_att_p_value"],
            ["overall_conf_int", "overall_att_conf_int"],
        ),
        # MultiPeriodDiDResults
        ("avg_att", ["avg_se"], ["avg_p_value"], ["avg_conf_int"]),
        # Simple DiDResults / SyntheticDiDResults / TROPResults / TripleDifferenceResults
        ("att", ["se"], ["p_value"], ["conf_int"]),
    ]
    for name, se_aliases, p_aliases, ci_aliases in alias_table:
        val = getattr(results, name, None)
        if val is None:
            continue
        se = next(
            (
                _to_python_float(getattr(results, a, None))
                for a in se_aliases
                if getattr(results, a, None) is not None
            ),
            None,
        )
        p = next(
            (
                _to_python_float(getattr(results, a, None))
                for a in p_aliases
                if getattr(results, a, None) is not None
            ),
            None,
        )
        ci = next(
            (
                _to_python_ci(getattr(results, a, None))
                for a in ci_aliases
                if getattr(results, a, None) is not None
            ),
            None,
        )
        alpha = _to_python_float(getattr(results, "alpha", fallback_alpha))
        return (name, _to_python_float(val), se, p, ci, alpha)
    return None


def _extract_scalar_effect(val: Any) -> Optional[float]:
    """Pull a scalar ``effect`` out of the many shapes results expose.

    Handles: ``PeriodEffect`` / ``GroupTimeEffect`` objects (``.effect`` attr),
    dicts with an ``"effect"`` key, and bare scalars.
    """
    if isinstance(val, dict):
        eff = val.get("effect")
        if eff is None:
            return None
        try:
            return float(eff)
        except (TypeError, ValueError):
            return None
    eff_attr = getattr(val, "effect", None)
    if eff_attr is not None:
        try:
            return float(eff_attr)
        except (TypeError, ValueError):
            return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _power_tier(ratio: Optional[float]) -> str:
    """Map ``mdv / |att|`` to a phrasing tier used by ``BusinessReport``.

    Tiers per ``docs/methodology/REPORTING.md``:
      * ``well_powered``:         ratio < 0.25
      * ``moderately_powered``:  0.25 <= ratio < 1.0
      * ``underpowered``:         ratio >= 1.0
      * ``unknown``:              ratio is None or non-finite
    """
    if ratio is None or not np.isfinite(ratio):
        return "unknown"
    if ratio < 0.25:
        return "well_powered"
    if ratio < 1.0:
        return "moderately_powered"
    return "underpowered"


def _collect_pre_period_coefs(results: Any) -> List[Tuple[Any, float, float, Optional[float]]]:
    """Return a sorted list of ``(key, effect, se, p_value)`` for pre-period coefficients.

    Handles two shapes:
      * ``pre_period_effects``: dict-of-``PeriodEffect`` on ``MultiPeriodDiDResults``.
      * ``event_study_effects``: dict-of-dict (with ``effect`` / ``se`` / ``p_value`` keys)
        on the staggered estimators (CS / SA / ImputationDiD / Stacked / EDiD / etc.).
        Pre-period entries are those with negative relative-time keys.

    Filtering rules (critical for methodology-safe PT tests):

    * Entries marked as reference markers (``n_groups == 0`` on the CS / SA /
      ImputationDiD / Stacked event-study shape) are excluded. These are
      synthetic ``effect=0, se=NaN`` rows injected for universal-base
      normalization; treating them as real pre-period evidence would inflate
      the Bonferroni denominator and produce bogus zero-deviation entries.
    * Entries whose ``effect`` or ``se`` is non-finite (NaN / inf) are
      excluded. A NaN SE means inference is undefined — feeding it into
      Bonferroni or Wald would produce a false-clean PT verdict.

    Returns an empty list when neither source provides valid pre-period entries.
    """
    results_list: List[Tuple[Any, float, float, Optional[float]]] = []
    pre = getattr(results, "pre_period_effects", None)
    if pre:
        for k, pe in pre.items():
            eff = getattr(pe, "effect", None)
            se = getattr(pe, "se", None)
            p = getattr(pe, "p_value", None)
            if eff is None or se is None:
                continue
            try:
                eff_f = float(eff)
                se_f = float(se)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(eff_f) and np.isfinite(se_f)):
                continue
            results_list.append((k, eff_f, se_f, _to_python_float(p)))
    else:
        es = getattr(results, "event_study_effects", None) or {}
        for k, entry in es.items():
            # Pre-period relative-time keys are negative (convention: e=-1, -2, ...).
            try:
                rel = int(k)
            except (TypeError, ValueError):
                continue
            if rel >= 0:
                continue
            if not isinstance(entry, dict):
                continue
            # Drop universal-base reference markers. Different estimator
            # aggregations use different flags for the synthetic marker row
            # (all of which carry NaN SE and p-value):
            #   * CS / SA: ``n_groups == 0``
            #   * Stacked / TwoStage / Imputation: ``n_obs == 0``
            # Treat either as a disqualifier so the Bonferroni denominator
            # and joint-Wald index are not inflated by non-informative rows.
            if entry.get("n_groups") == 0 or entry.get("n_obs") == 0:
                continue
            eff = entry.get("effect")
            se = entry.get("se")
            p = entry.get("p_value")
            if eff is None or se is None:
                continue
            try:
                eff_f = float(eff)
                se_f = float(se)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(eff_f) and np.isfinite(se_f)):
                continue
            results_list.append((k, eff_f, se_f, _to_python_float(p)))
    results_list.sort(key=lambda t: t[0] if isinstance(t[0], (int, float)) else str(t[0]))
    return results_list


def _pt_verdict(p: Optional[float]) -> str:
    """Map a pre-trends joint p-value to the three-bin verdict enum.

    Verdicts per ``docs/methodology/REPORTING.md``:
      - p >= 0.30  -> ``no_detected_violation`` (phrasing hedges on power
        unless DR also reports that the test is well-powered via
        ``compute_pretrends_power``).
      - 0.05 <= p < 0.30  -> ``some_evidence_against``.
      - p < 0.05  -> ``clear_violation``.
    """
    if p is None or not np.isfinite(p):
        return "inconclusive"
    if p < 0.05:
        return "clear_violation"
    if p < 0.30:
        return "some_evidence_against"
    return "no_detected_violation"


def _to_python_float(value: Any) -> Optional[float]:
    """Convert numpy scalars to built-in ``float``; preserve None; return None on failure."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f


def _to_python_scalar(value: Any) -> Any:
    """Convert numpy scalars to built-in Python types where possible; pass through otherwise."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_python_ci(ci: Any) -> Optional[List[float]]:
    """Convert a 2-tuple CI to ``[float, float]``; return None when malformed."""
    if ci is None:
        return None
    try:
        lo, hi = ci
    except (TypeError, ValueError):
        return None
    lo_f = _to_python_float(lo)
    hi_f = _to_python_float(hi)
    if lo_f is None or hi_f is None:
        return None
    return [lo_f, hi_f]


# ---------------------------------------------------------------------------
# Prose rendering helpers
# ---------------------------------------------------------------------------
def _check_headline(check: str, section: Dict[str, Any]) -> Optional[Any]:
    """Return the most descriptive scalar for the per-check row in to_dataframe()."""
    if section.get("status") != "ran":
        return None
    if check == "parallel_trends":
        return section.get("joint_p_value")
    if check == "pretrends_power":
        return section.get("mdv_share_of_att")
    if check == "sensitivity":
        return section.get("breakdown_M")
    if check == "bacon":
        return section.get("forbidden_weight")
    if check == "design_effect":
        return section.get("deff")
    if check == "heterogeneity":
        return section.get("cv")
    if check == "epv":
        return section.get("min_epv")
    if check == "estimator_native":
        return section.get("pre_treatment_fit")
    return None


def _render_overall_interpretation(schema: Dict[str, Any], labels: Dict[str, str]) -> str:
    """Synthesize a plain-English paragraph across DR checks.

    The paragraph names the headline effect, the dominant validity concern
    (typically parallel trends or sensitivity), secondary caveats
    (heterogeneity, design effect, Bacon), and one concrete next action.
    Never produces traffic-light verdicts — severity is conveyed by natural
    language per ``docs/methodology/REPORTING.md``.
    """
    sentences: List[str] = []
    headline = schema.get("headline_metric") or {}
    est = schema.get("estimator", "the estimator")
    outcome = labels.get("outcome_label", "the outcome")
    treatment = labels.get("treatment_label", "the treatment")

    # Sentence 1: headline
    val = headline.get("value") if isinstance(headline, dict) else None
    ci = headline.get("conf_int") if isinstance(headline, dict) else None
    p = headline.get("p_value") if isinstance(headline, dict) else None
    if val is not None:
        direction = "increased" if val > 0 else "decreased" if val < 0 else "did not change"
        # Use the headline's own alpha rather than hardcoding 95 so prose
        # stays consistent with the rendered interval when alpha != 0.05.
        headline_alpha = headline.get("alpha") if isinstance(headline, dict) else None
        if isinstance(headline_alpha, (int, float)) and 0 < headline_alpha < 1:
            ci_level = int(round((1.0 - headline_alpha) * 100))
        else:
            ci_level = 95
        ci_str = (
            f" ({ci_level}% CI: {ci[0]:.3g} to {ci[1]:.3g})"
            if isinstance(ci, (list, tuple)) and len(ci) == 2 and None not in ci
            else ""
        )
        p_str = f", p = {p:.3g}" if isinstance(p, (int, float)) else ""
        sentences.append(
            f"On {est}, {treatment} {direction} {outcome} by {val:.3g}{ci_str}{p_str}."
        )

    # Sentence 2: parallel trends + power
    pt = schema.get("parallel_trends") or {}
    pp = schema.get("pretrends_power") or {}
    if pt.get("status") == "ran":
        verdict = pt.get("verdict")
        jp = pt.get("joint_p_value")
        jp_str = f" (joint p = {jp:.3g})" if isinstance(jp, (int, float)) else ""
        if verdict == "clear_violation":
            sentences.append(
                f"Pre-treatment event-study coefficients clearly reject parallel "
                f"trends{jp_str}. The headline estimate should be treated as "
                f"tentative pending sensitivity analysis."
            )
        elif verdict == "some_evidence_against":
            sentences.append(
                f"Pre-treatment data show some evidence of diverging trends"
                f"{jp_str}. Interpret the headline alongside the sensitivity "
                f"analysis below."
            )
        elif verdict == "no_detected_violation":
            tier = pp.get("tier") if pp.get("status") == "ran" else "unknown"
            if tier == "well_powered":
                sentences.append(
                    f"Pre-treatment data are consistent with parallel trends"
                    f"{jp_str} and the test is well-powered (MDV is a small "
                    f"share of the estimated effect), so a material pre-trend "
                    f"would likely have been detected."
                )
            elif tier == "moderately_powered":
                sentences.append(
                    f"Pre-treatment data do not reject parallel trends"
                    f"{jp_str}; the test is moderately informative. See the "
                    f"sensitivity analysis below for bounded-violation "
                    f"guarantees."
                )
            else:
                sentences.append(
                    f"Pre-treatment data do not reject parallel trends"
                    f"{jp_str}, but the test has limited power — a non-rejection "
                    f"does not prove the assumption. See the HonestDiD "
                    f"sensitivity analysis below for a more reliable signal."
                )
        elif verdict == "design_enforced_pt":
            rmse = pt.get("pre_treatment_fit_rmse")
            sentences.append(
                f"The synthetic control matches the treated group's "
                f"pre-period trajectory with RMSE = "
                f"{rmse:.3g} (SDiD's design-enforced analogue of parallel "
                f"trends)."
                if isinstance(rmse, (int, float))
                else "SDiD's synthetic control is designed to satisfy the "
                "weighted parallel-trends analogue."
            )

    # Sentence 3: sensitivity. The "robust across the grid" phrasing is reserved
    # for genuine SensitivityResults grids; a precomputed single-M HonestDiDResults
    # is narrated as a point check ("at M=<value>") even though breakdown_M is None.
    sens = schema.get("sensitivity") or {}
    if sens.get("status") == "ran":
        bkd = sens.get("breakdown_M")
        conclusion = sens.get("conclusion")
        if conclusion == "single_M_precomputed":
            grid = sens.get("grid") or []
            point = grid[0] if grid else {}
            m_val = point.get("M")
            robust = point.get("robust_to_zero")
            if isinstance(m_val, (int, float)):
                if robust:
                    sentences.append(
                        f"HonestDiD sensitivity (single point checked): "
                        f"at M = {m_val:.2g}, the robust CI excludes zero. "
                        f"This is a point check, not a grid — use "
                        f"HonestDiD.sensitivity() for a breakdown value."
                    )
                else:
                    sentences.append(
                        f"HonestDiD sensitivity (single point checked): "
                        f"at M = {m_val:.2g}, the robust CI includes zero. "
                        f"Run HonestDiD.sensitivity() across a grid to find "
                        f"the breakdown value."
                    )
        elif bkd is None:
            sentences.append(
                "The effect remains significant across the entire HonestDiD "
                "grid — robust to plausible parallel-trends violations."
            )
        elif isinstance(bkd, (int, float)) and bkd >= 1.0:
            sentences.append(
                f"HonestDiD sensitivity: the result remains significant under "
                f"parallel-trends violations up to {bkd:.2g}x the observed "
                f"pre-period variation."
            )
        else:
            sentences.append(
                f"HonestDiD sensitivity: the result is fragile — the "
                f"confidence interval includes zero once violations reach "
                f"{bkd:.2g}x the pre-period variation."
                if isinstance(bkd, (int, float))
                else ""
            )

    # Sentence 4: one secondary caveat if present.
    bacon = schema.get("bacon") or {}
    if bacon.get("status") == "ran" and bacon.get("verdict") == "materially_contaminated":
        fw = bacon.get("forbidden_weight")
        if isinstance(fw, (int, float)):
            sentences.append(
                f"Goodman-Bacon decomposition flags {fw:.0%} of TWFE weight on "
                f"'forbidden' later-vs-earlier comparisons — consider a "
                f"heterogeneity-robust estimator (CS / SA / BJS / Gardner) if "
                f"not already in use."
            )
    deff = schema.get("design_effect") or {}
    if deff.get("status") == "ran" and not deff.get("is_trivial"):
        d = deff.get("deff")
        eff_n = deff.get("effective_n")
        if isinstance(d, (int, float)) and d >= 1.05:
            eff_str = f", effective n = {eff_n:.0f}" if isinstance(eff_n, (int, float)) else ""
            sentences.append(
                f"Survey design effect is {d:.2g} (variance inflation relative "
                f"to simple random sampling{eff_str})."
            )

    # Sentence 5: next step
    next_steps = schema.get("next_steps") or []
    if next_steps:
        top = next_steps[0]
        if top.get("label"):
            sentences.append(f"Next step: {top['label']}.")

    if not sentences:
        return ""
    return " ".join(s for s in sentences if s)


def _render_dr_full_report(results: "DiagnosticReportResults") -> str:
    """Render a markdown report from a populated ``DiagnosticReportResults``."""
    schema = results.schema
    lines: List[str] = []
    lines.append("# Diagnostic Report")
    lines.append("")
    lines.append(f"**Estimator**: `{schema.get('estimator')}`")
    headline = schema.get("headline_metric")
    if headline:
        lines.append(
            f"**Headline**: {headline.get('name')} = "
            f"{headline.get('value')} "
            f"(SE {headline.get('se')}, p = {headline.get('p_value')})"
        )
    lines.append("")
    lines.append("## Overall Interpretation")
    lines.append("")
    lines.append(schema.get("overall_interpretation", "") or "_No synthesis available._")
    lines.append("")

    section_order = [
        ("Parallel trends", "parallel_trends"),
        ("Pre-trends power", "pretrends_power"),
        ("HonestDiD sensitivity", "sensitivity"),
        ("Goodman-Bacon decomposition", "bacon"),
        ("Effect-stability / heterogeneity", "heterogeneity"),
        ("Survey design effect", "design_effect"),
        ("Propensity-score EPV", "epv"),
        ("Estimator-native diagnostics", "estimator_native_diagnostics"),
        ("Placebo battery", "placebo"),
    ]
    for title, key in section_order:
        section = schema.get(key) or {}
        status = section.get("status", "not_run")
        lines.append(f"## {title}")
        lines.append(f"- status: `{status}`")
        if status == "skipped" or status == "not_applicable":
            reason = section.get("reason")
            if reason:
                lines.append(f"- reason: {reason}")
        else:
            for k, v in section.items():
                if k in ("status", "reason"):
                    continue
                if isinstance(v, (dict, list)):
                    continue
                lines.append(f"- {k}: `{v}`")
        lines.append("")

    if schema.get("next_steps"):
        lines.append("## Next Steps")
        for s in schema["next_steps"]:
            if s.get("label"):
                lines.append(f"- {s['label']}")
                if s.get("why"):
                    lines.append(f"  - why: {s['why']}")
    return "\n".join(lines)
