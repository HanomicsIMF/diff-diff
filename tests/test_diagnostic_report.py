"""Tests for ``diff_diff.diagnostic_report.DiagnosticReport``.

Covers:
- Schema contract: every top-level key always present, stable enum values.
- Applicability matrix: per-estimator ``applicable_checks`` property.
- JSON round-trip.
- ``precomputed=`` passthrough (sensitivity).
- Pre-trends verdict thresholds (three bins).
- Power-aware tier thresholds (three bins + fallback).
- DEFF reads from ``survey_metadata`` when present.
- EfficientDiD ``hausman_pretest`` pathway.
- SDiD / TROP native diagnostics.
- Error-doesn't-break-report (diagnostic raises -> section records error).
"""

from __future__ import annotations

import json
import warnings
from unittest.mock import patch

import numpy as np
import pytest

import diff_diff as dd
from diff_diff import (
    CallawaySantAnna,
    DiagnosticReport,
    DiagnosticReportResults,
    DifferenceInDifferences,
    EfficientDiD,
    MultiPeriodDiD,
    SyntheticDiD,
    generate_did_data,
    generate_factor_data,
    generate_staggered_data,
)
from diff_diff.diagnostic_report import (
    DIAGNOSTIC_REPORT_SCHEMA_VERSION,
    _power_tier,
    _pt_verdict,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
_TOP_LEVEL_KEYS = {
    "schema_version",
    "estimator",
    "headline_metric",
    "parallel_trends",
    "pretrends_power",
    "sensitivity",
    "placebo",
    "bacon",
    "design_effect",
    "heterogeneity",
    "epv",
    "estimator_native_diagnostics",
    "skipped",
    "warnings",
    "overall_interpretation",
    "next_steps",
}

_STATUS_ENUM = {
    "ran",
    "skipped",
    "error",
    "not_applicable",
    "not_run",
    "computed",
}


@pytest.fixture(scope="module")
def did_fit():
    warnings.filterwarnings("ignore")
    df = generate_did_data(n_units=80, n_periods=4, treatment_effect=1.5, seed=7)
    did = DifferenceInDifferences().fit(df, outcome="outcome", treatment="treated", time="post")
    return did, df


@pytest.fixture(scope="module")
def multi_period_fit():
    warnings.filterwarnings("ignore")
    df = generate_did_data(n_units=80, n_periods=8, treatment_effect=1.5, seed=7)
    es = MultiPeriodDiD().fit(
        df,
        outcome="outcome",
        treatment="treated",
        time="period",
        unit="unit",
        reference_period=3,
    )
    return es, df


@pytest.fixture(scope="module")
def cs_fit():
    warnings.filterwarnings("ignore")
    sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
    cs = CallawaySantAnna().fit(
        sdf,
        outcome="outcome",
        unit="unit",
        time="period",
        first_treat="first_treat",
        aggregate="event_study",
    )
    return cs, sdf


@pytest.fixture(scope="module")
def edid_fit():
    warnings.filterwarnings("ignore")
    sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
    edid = EfficientDiD().fit(
        sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
    )
    return edid, sdf


@pytest.fixture(scope="module")
def sdid_fit():
    warnings.filterwarnings("ignore")
    fdf = generate_factor_data(n_units=25, n_pre=8, n_post=4, n_treated=4, seed=11)
    sdid = SyntheticDiD().fit(fdf, outcome="outcome", unit="unit", time="period", treatment="treat")
    return sdid, fdf


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------
class TestSchemaContract:
    """The AI-legible schema is the public promise. These tests lock it down."""

    def test_every_top_level_key_present_did(self, did_fit):
        fit, df = did_fit
        dr = DiagnosticReport(fit, data=df, outcome="outcome", treatment="treated", time="post")
        schema = dr.to_dict()
        assert set(schema.keys()) == _TOP_LEVEL_KEYS

    def test_every_top_level_key_present_multiperiod(self, multi_period_fit):
        fit, _ = multi_period_fit
        schema = DiagnosticReport(fit).to_dict()
        assert set(schema.keys()) == _TOP_LEVEL_KEYS

    def test_every_top_level_key_present_cs(self, cs_fit):
        fit, sdf = cs_fit
        schema = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        ).to_dict()
        assert set(schema.keys()) == _TOP_LEVEL_KEYS

    def test_every_top_level_key_present_sdid(self, sdid_fit):
        fit, _ = sdid_fit
        schema = DiagnosticReport(fit).to_dict()
        assert set(schema.keys()) == _TOP_LEVEL_KEYS

    def test_schema_version_constant(self, multi_period_fit):
        fit, _ = multi_period_fit
        schema = DiagnosticReport(fit).to_dict()
        assert schema["schema_version"] == DIAGNOSTIC_REPORT_SCHEMA_VERSION
        assert DIAGNOSTIC_REPORT_SCHEMA_VERSION == "1.0"

    def test_all_statuses_use_closed_enum(self, cs_fit):
        fit, sdf = cs_fit
        schema = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        ).to_dict()
        for key in [
            "parallel_trends",
            "pretrends_power",
            "sensitivity",
            "placebo",
            "bacon",
            "design_effect",
            "heterogeneity",
            "epv",
            "estimator_native_diagnostics",
        ]:
            section = schema.get(key)
            assert isinstance(section, dict), f"{key} missing"
            assert (
                section.get("status") in _STATUS_ENUM
            ), f"{key}.status = {section.get('status')!r} not in {_STATUS_ENUM}"

    def test_json_round_trip_multiperiod(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit)
        dumped = json.dumps(dr.to_dict())
        assert len(dumped) > 0
        round = json.loads(dumped)
        assert round["schema_version"] == DIAGNOSTIC_REPORT_SCHEMA_VERSION

    def test_json_round_trip_sdid(self, sdid_fit):
        fit, _ = sdid_fit
        dumped = json.dumps(DiagnosticReport(fit).to_dict())
        assert len(dumped) > 0


# ---------------------------------------------------------------------------
# Applicability matrix
# ---------------------------------------------------------------------------
class TestApplicabilityMatrix:
    """Per-estimator applicability set filtered by instance state + options."""

    def test_did_without_data_skips_pt(self, did_fit):
        fit, _ = did_fit
        dr = DiagnosticReport(fit)  # no data
        assert "parallel_trends" not in dr.applicable_checks
        assert "parallel_trends" in dr.skipped_checks
        reason = dr.skipped_checks["parallel_trends"]
        assert "data" in reason.lower()

    def test_did_with_data_runs_pt(self, did_fit):
        fit, df = did_fit
        dr = DiagnosticReport(fit, data=df, outcome="outcome", treatment="treated", time="post")
        assert "parallel_trends" in dr.applicable_checks

    def test_multiperiod_runs_pt_and_power_and_sensitivity(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit)
        applicable = set(dr.applicable_checks)
        assert "parallel_trends" in applicable
        assert "pretrends_power" in applicable
        assert "sensitivity" in applicable

    def test_cs_runs_heterogeneity(self, cs_fit):
        fit, sdf = cs_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        applicable = set(dr.applicable_checks)
        assert "heterogeneity" in applicable
        assert "bacon" in applicable
        assert "parallel_trends" in applicable

    def test_sdid_has_estimator_native(self, sdid_fit):
        fit, _ = sdid_fit
        dr = DiagnosticReport(fit)
        assert "estimator_native" in dr.applicable_checks

    def test_run_opt_outs_move_checks_to_skipped(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit, run_sensitivity=False)
        assert "sensitivity" not in dr.applicable_checks
        assert dr.skipped_checks["sensitivity"].startswith("run_sensitivity=False")

    def test_placebo_is_reserved_and_skipped(self, did_fit):
        """Placebo is always in _CHECK_NAMES, always skipped in MVP."""
        fit, df = did_fit
        dr = DiagnosticReport(fit, data=df, outcome="outcome", treatment="treated", time="post")
        placebo_section = dr.to_dict()["placebo"]
        assert placebo_section["status"] in {"skipped", "not_applicable"}


# ---------------------------------------------------------------------------
# Precomputed passthrough
# ---------------------------------------------------------------------------
class TestPrecomputed:
    def test_precomputed_sensitivity_is_used_verbatim(self, multi_period_fit):
        fit, _ = multi_period_fit

        # Construct a minimal SensitivityResults-shaped object the formatter recognizes.
        class _FakeSens:
            M_values = np.array([0.5, 1.0])
            bounds = [(0.1, 2.0), (-0.2, 2.5)]
            robust_cis = [(0.05, 2.1), (-0.3, 2.6)]
            breakdown_M = 0.75
            method = "relative_magnitude"
            original_estimate = 1.0
            original_se = 0.2
            alpha = 0.05

        fake = _FakeSens()
        with patch("diff_diff.honest_did.HonestDiD.sensitivity_analysis") as mock:
            dr = DiagnosticReport(fit, precomputed={"sensitivity": fake})
            dr.to_dict()
            mock.assert_not_called()
        schema = dr.to_dict()
        assert schema["sensitivity"]["status"] == "ran"
        assert schema["sensitivity"]["breakdown_M"] == 0.75


# ---------------------------------------------------------------------------
# Verdict / tier helpers
# ---------------------------------------------------------------------------
class TestVerdictsAndTiers:
    def test_pt_verdict_three_bins(self):
        assert _pt_verdict(0.001) == "clear_violation"
        assert _pt_verdict(0.049) == "clear_violation"
        assert _pt_verdict(0.10) == "some_evidence_against"
        assert _pt_verdict(0.29) == "some_evidence_against"
        assert _pt_verdict(0.30) == "no_detected_violation"
        assert _pt_verdict(0.99) == "no_detected_violation"
        assert _pt_verdict(None) == "inconclusive"
        assert _pt_verdict(float("nan")) == "inconclusive"

    def test_power_tier_three_bins_plus_unknown(self):
        assert _power_tier(0.1) == "well_powered"
        assert _power_tier(0.24) == "well_powered"
        assert _power_tier(0.25) == "moderately_powered"
        assert _power_tier(0.99) == "moderately_powered"
        assert _power_tier(1.0) == "underpowered"
        assert _power_tier(5.0) == "underpowered"
        assert _power_tier(None) == "unknown"
        assert _power_tier(float("nan")) == "unknown"


# ---------------------------------------------------------------------------
# EfficientDiD hausman pathway
# ---------------------------------------------------------------------------
class TestEfficientDiDHausman:
    def test_hausman_pretest_runs_with_data_kwargs(self, edid_fit):
        fit, sdf = edid_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "ran"
        assert pt["method"] == "hausman"

    def test_hausman_skipped_without_data_kwargs(self, edid_fit):
        fit, _ = edid_fit
        dr = DiagnosticReport(fit)
        pt = dr.to_dict()["parallel_trends"]
        assert pt["status"] == "skipped"
        assert pt["method"] == "hausman"


# ---------------------------------------------------------------------------
# SDiD native
# ---------------------------------------------------------------------------
class TestSDiDNative:
    def test_sdid_pt_uses_synthetic_fit_method(self, sdid_fit):
        fit, _ = sdid_fit
        pt = DiagnosticReport(fit).to_dict()["parallel_trends"]
        assert pt["method"] == "synthetic_fit"
        assert pt["verdict"] == "design_enforced_pt"
        assert isinstance(pt.get("pre_treatment_fit_rmse"), float)

    def test_sdid_native_section_populated(self, sdid_fit):
        fit, _ = sdid_fit
        native = DiagnosticReport(fit).to_dict()["estimator_native_diagnostics"]
        assert native["status"] == "ran"
        assert native["estimator"] == "SyntheticDiD"
        assert "weight_concentration" in native
        assert "in_time_placebo" in native
        assert "zeta_sensitivity" in native

    def test_sdid_does_not_call_honest_did(self, sdid_fit):
        """HonestDiD sensitivity should NOT run on SDiD (native path used instead)."""
        fit, _ = sdid_fit
        with patch("diff_diff.honest_did.HonestDiD.sensitivity_analysis") as mock:
            DiagnosticReport(fit).to_dict()
            mock.assert_not_called()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_sensitivity_error_does_not_break_report(self, multi_period_fit):
        """A failing diagnostic records its error in the section; the report still renders."""
        fit, _ = multi_period_fit

        def _raise(*args, **kwargs):
            raise RuntimeError("synthetic test failure")

        with patch("diff_diff.honest_did.HonestDiD.sensitivity_analysis", side_effect=_raise):
            dr = DiagnosticReport(fit)
            schema = dr.to_dict()
            sens = schema["sensitivity"]
            assert sens["status"] == "error"
            assert "synthetic test failure" in sens["reason"]
            # Other sections still ran.
            assert schema["parallel_trends"]["status"] == "ran"


# ---------------------------------------------------------------------------
# Overall prose
# ---------------------------------------------------------------------------
class TestOverallInterpretation:
    def test_overall_interpretation_nonempty_for_fit(self, cs_fit):
        fit, sdf = cs_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        prose = dr.summary()
        assert isinstance(prose, str)
        assert len(prose) > 50  # a real paragraph

    def test_full_report_has_headers(self, cs_fit):
        fit, sdf = cs_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        md = dr.full_report()
        assert "# Diagnostic Report" in md
        assert "## Overall Interpretation" in md
        assert "## Parallel trends" in md
        assert "## HonestDiD sensitivity" in md


# ---------------------------------------------------------------------------
# Public result class
# ---------------------------------------------------------------------------
class TestDiagnosticReportResults:
    def test_run_all_returns_dataclass(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit)
        results = dr.run_all()
        assert isinstance(results, DiagnosticReportResults)
        assert isinstance(results.applicable_checks, tuple)
        assert isinstance(results.schema, dict)

    def test_run_all_is_idempotent(self, multi_period_fit):
        fit, _ = multi_period_fit
        dr = DiagnosticReport(fit)
        a = dr.run_all()
        b = dr.run_all()
        assert a is b  # cached


# ---------------------------------------------------------------------------
# Public API exposure
# ---------------------------------------------------------------------------
def test_public_api_exports():
    for name in ("DiagnosticReport", "DiagnosticReportResults", "DIAGNOSTIC_REPORT_SCHEMA_VERSION"):
        assert hasattr(dd, name), f"diff_diff must export {name}"
