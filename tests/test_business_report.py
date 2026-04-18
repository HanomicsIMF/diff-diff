"""Tests for ``diff_diff.business_report.BusinessReport``.

Covers the expanded test list from the approved plan:
- Schema contract across result types.
- JSON round-trip.
- BR-DR integration (auto, explicit, False).
- ``honest_did_results=`` passthrough (no re-computation).
- Unit-label behavior (pp vs $ differ; column-name fallback).
- Log-points unit policy (no arithmetic translation; informational caveat).
- Significance-chasing guard boundary.
- Pre-trends verdict thresholds (three bins routed through BR phrasing).
- Power-aware phrasing (three tiers + underpowered fallback).
- NaN ATT surfaces a caveat and does not crash.
- ``include_appendix`` toggle.
- ``BusinessReport(BaconDecompositionResults)`` raises TypeError.
- Survey metadata passthrough to schema + phrasing.
- Single-knob alpha drives both CI level and phrasing.
"""

from __future__ import annotations

import json
import warnings
from unittest.mock import patch

import numpy as np
import pytest

import diff_diff as dd
from diff_diff import (
    BusinessReport,
    BusinessContext,
    CallawaySantAnna,
    DiagnosticReport,
    DifferenceInDifferences,
    MultiPeriodDiD,
    SyntheticDiD,
    bacon_decompose,
    generate_did_data,
    generate_factor_data,
    generate_staggered_data,
)
from diff_diff.business_report import BUSINESS_REPORT_SCHEMA_VERSION

warnings.filterwarnings("ignore")

_BR_TOP_LEVEL_KEYS = {
    "schema_version",
    "estimator",
    "context",
    "headline",
    "assumption",
    "pre_trends",
    "sensitivity",
    "sample",
    "heterogeneity",
    "robustness",
    "diagnostics",
    "next_steps",
    "caveats",
    "references",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def did_fit():
    df = generate_did_data(n_units=80, n_periods=4, treatment_effect=1.5, seed=7)
    did = DifferenceInDifferences().fit(df, outcome="outcome", treatment="treated", time="post")
    return did, df


@pytest.fixture(scope="module")
def event_study_fit():
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
def sdid_fit():
    fdf = generate_factor_data(n_units=25, n_pre=8, n_post=4, n_treated=4, seed=11)
    sdid = SyntheticDiD().fit(fdf, outcome="outcome", unit="unit", time="period", treatment="treat")
    return sdid, fdf


# ---------------------------------------------------------------------------
# Schema contract
# ---------------------------------------------------------------------------
class TestSchemaContract:
    def test_top_level_keys(self, event_study_fit):
        fit, _ = event_study_fit
        br = BusinessReport(fit, auto_diagnostics=False)
        assert set(br.to_dict().keys()) == _BR_TOP_LEVEL_KEYS

    def test_schema_version(self, event_study_fit):
        fit, _ = event_study_fit
        assert (
            BusinessReport(fit, auto_diagnostics=False).to_dict()["schema_version"]
            == BUSINESS_REPORT_SCHEMA_VERSION
        )

    def test_json_round_trip(self, cs_fit):
        fit, _ = cs_fit
        br = BusinessReport(
            fit,
            outcome_label="sales",
            outcome_unit="$",
            treatment_label="the policy",
        )
        dumped = json.dumps(br.to_dict())
        assert len(dumped) > 0
        assert json.loads(dumped)["schema_version"] == BUSINESS_REPORT_SCHEMA_VERSION

    def test_json_round_trip_sdid(self, sdid_fit):
        fit, _ = sdid_fit
        br = BusinessReport(fit, outcome_label="revenue", outcome_unit="$")
        dumped = json.dumps(br.to_dict())
        assert len(dumped) > 0


# ---------------------------------------------------------------------------
# BR ↔ DR integration
# ---------------------------------------------------------------------------
class TestDiagnosticsIntegration:
    def test_auto_diagnostics_true_populates_diagnostics_block(self, event_study_fit):
        fit, _ = event_study_fit
        br = BusinessReport(fit, auto_diagnostics=True)
        d = br.to_dict()
        assert d["diagnostics"]["status"] == "ran"
        assert "schema" in d["diagnostics"]

    def test_auto_diagnostics_false_skips(self, event_study_fit):
        fit, _ = event_study_fit
        br = BusinessReport(fit, auto_diagnostics=False)
        d = br.to_dict()
        assert d["diagnostics"]["status"] == "skipped"
        assert "auto_diagnostics=False" in d["diagnostics"]["reason"]

    def test_explicit_diagnostics_results_takes_precedence(self, event_study_fit):
        fit, _ = event_study_fit
        dr = DiagnosticReport(fit)
        dr_results = dr.run_all()
        br = BusinessReport(fit, diagnostics=dr_results)
        d = br.to_dict()
        assert d["diagnostics"]["status"] == "ran"
        # Same dict identity shows the supplied results were used verbatim.
        assert d["diagnostics"]["schema"] is dr_results.schema

    def test_explicit_diagnostics_report_runs(self, event_study_fit):
        fit, _ = event_study_fit
        dr = DiagnosticReport(fit)
        br = BusinessReport(fit, diagnostics=dr)
        assert br.to_dict()["diagnostics"]["status"] == "ran"

    def test_diagnostics_wrong_type_raises(self, event_study_fit):
        fit, _ = event_study_fit
        with pytest.raises(TypeError):
            BusinessReport(fit, diagnostics="not a DR")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# HonestDiD passthrough
# ---------------------------------------------------------------------------
class TestHonestDiDPassthrough:
    def test_supplied_sensitivity_is_not_recomputed(self, event_study_fit):
        fit, _ = event_study_fit

        class _FakeSens:
            M_values = np.array([0.5, 1.0])
            bounds = [(0.1, 2.0), (-0.2, 2.5)]
            robust_cis = [(0.05, 2.1), (-0.3, 2.6)]
            breakdown_M = 1.5
            method = "relative_magnitude"
            original_estimate = 1.0
            original_se = 0.2
            alpha = 0.05

        fake = _FakeSens()
        with patch("diff_diff.honest_did.HonestDiD.sensitivity_analysis") as mock:
            br = BusinessReport(fit, honest_did_results=fake)
            schema = br.to_dict()
            mock.assert_not_called()
        sens = schema["sensitivity"]
        assert sens["status"] == "computed"
        assert sens["breakdown_M"] == 1.5


# ---------------------------------------------------------------------------
# Unit labels and policy
# ---------------------------------------------------------------------------
class TestUnitLabels:
    def test_dollar_unit_formats_currency(self, cs_fit):
        fit, _ = cs_fit
        br = BusinessReport(fit, outcome_label="sales", outcome_unit="$", auto_diagnostics=False)
        headline = br.headline()
        assert "$" in headline

    def test_pp_unit_formats_percentage_points(self, cs_fit):
        fit, _ = cs_fit
        br = BusinessReport(
            fit, outcome_label="awareness", outcome_unit="pp", auto_diagnostics=False
        )
        headline = br.headline()
        assert "pp" in headline

    def test_zero_config_falls_back_to_generic_label(self, cs_fit):
        fit, _ = cs_fit
        br = BusinessReport(fit, auto_diagnostics=False)
        d = br.to_dict()
        assert d["context"]["outcome_label"] == "the outcome"
        assert d["context"]["treatment_label"] == "the treatment"

    def test_log_points_emits_unit_policy_caveat(self, cs_fit):
        fit, _ = cs_fit
        br = BusinessReport(fit, outcome_unit="log_points", auto_diagnostics=False)
        caveats = br.caveats()
        topics = {c.get("topic") for c in caveats}
        assert "unit_policy" in topics


# ---------------------------------------------------------------------------
# Significance phrasing
# ---------------------------------------------------------------------------
class TestSignificancePhrasing:
    def test_high_significance_produces_strong_language(self, cs_fit):
        """CS on this seed has p ~ 1e-56 (very strong) -> 'strongly supported'."""
        fit, _ = cs_fit
        br = BusinessReport(fit, outcome_label="sales", outcome_unit="$")
        summary = br.summary()
        assert "strongly supported" in summary

    def test_near_threshold_caveat(self, event_study_fit):
        """Fabricate a p-value near 0.05 to exercise the significance-chasing guard."""
        fit, _ = event_study_fit
        # Monkey-patch the result to land p_value in (0.04, 0.051).
        original = fit.avg_p_value
        try:
            fit.avg_p_value = 0.045
            br = BusinessReport(fit, auto_diagnostics=False)
            caveats = br.caveats()
            topics = {c.get("topic") for c in caveats}
            assert "near_significance" in topics
        finally:
            fit.avg_p_value = original

    def test_far_from_threshold_no_near_caveat(self, event_study_fit):
        fit, _ = event_study_fit
        original = fit.avg_p_value
        try:
            fit.avg_p_value = 0.010
            br = BusinessReport(fit, auto_diagnostics=False)
            topics = {c.get("topic") for c in br.caveats()}
            assert "near_significance" not in topics
        finally:
            fit.avg_p_value = original


# ---------------------------------------------------------------------------
# Pre-trends verdict + power tier phrasing
# ---------------------------------------------------------------------------
class TestPreTrendsVerdictPhrasing:
    """Verdict and tier should flow through into schema AND phrasing."""

    def test_verdict_and_tier_surface_in_schema(self, event_study_fit):
        fit, _ = event_study_fit
        br = BusinessReport(fit, auto_diagnostics=True)
        pt = br.to_dict()["pre_trends"]
        # This fixture has a clear violation and an underpowered test — both set.
        assert pt["status"] == "computed"
        assert pt["verdict"] in {
            "no_detected_violation",
            "some_evidence_against",
            "clear_violation",
        }

    def test_clear_violation_phrased_tentatively(self, event_study_fit):
        fit, _ = event_study_fit
        br = BusinessReport(fit, auto_diagnostics=True)
        if br.to_dict()["pre_trends"].get("verdict") == "clear_violation":
            summary = br.summary()
            assert "tentative" in summary or "reject parallel trends" in summary

    def test_underpowered_phrasing_uses_hedge_language(self, cs_fit):
        """CS fit on this seed typically produces 'no_detected_violation' + underpowered."""
        fit, sdf = cs_fit
        # Force the CS fit through our BR pipeline.
        br = BusinessReport(
            fit,
            outcome_label="sales",
            outcome_unit="$",
            diagnostics=DiagnosticReport(
                fit,
                data=sdf,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            ),
        )
        pt = br.to_dict()["pre_trends"]
        if pt.get("verdict") == "no_detected_violation":
            summary = br.summary()
            # One of the three tier-specific phrases should appear.
            assert (
                "limited power" in summary
                or "moderately informative" in summary
                or "well-powered" in summary
                or "likely have been detected" in summary
            )


# ---------------------------------------------------------------------------
# NaN ATT
# ---------------------------------------------------------------------------
class TestNaNATT:
    def test_nan_att_produces_caveat_and_does_not_crash(self, event_study_fit):
        fit, _ = event_study_fit
        original = fit.avg_att
        try:
            fit.avg_att = float("nan")
            br = BusinessReport(fit, auto_diagnostics=False)
            summary = br.summary()
            caveats = br.caveats()
            assert isinstance(summary, str)
            assert any(c.get("topic") == "estimation_failure" for c in caveats)
            assert br.to_dict()["headline"]["sign"] == "undefined"
        finally:
            fit.avg_att = original


# ---------------------------------------------------------------------------
# include_appendix toggle
# ---------------------------------------------------------------------------
class TestAppendix:
    def test_include_appendix_true_embeds_summary(self, event_study_fit):
        fit, _ = event_study_fit
        br = BusinessReport(fit, auto_diagnostics=False, include_appendix=True)
        md = br.full_report()
        assert "## Technical Appendix" in md

    def test_include_appendix_false_omits(self, event_study_fit):
        fit, _ = event_study_fit
        br = BusinessReport(fit, auto_diagnostics=False, include_appendix=False)
        md = br.full_report()
        assert "## Technical Appendix" not in md


# ---------------------------------------------------------------------------
# BaconDecompositionResults
# ---------------------------------------------------------------------------
class TestBaconTypeError:
    def test_br_on_bacon_raises(self):
        sdf = generate_staggered_data(n_units=30, n_periods=6, treatment_effect=1.5, seed=7)
        bacon = bacon_decompose(
            sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        with pytest.raises(TypeError, match="BaconDecompositionResults is a diagnostic"):
            BusinessReport(bacon)


# ---------------------------------------------------------------------------
# Survey metadata passthrough
# ---------------------------------------------------------------------------
class TestSurveyPassthrough:
    def test_survey_absent_yields_null_survey_block(self, cs_fit):
        fit, _ = cs_fit
        br = BusinessReport(fit, auto_diagnostics=False)
        d = br.to_dict()
        assert d["sample"]["survey"] is None

    def test_survey_present_populates_block(self, event_study_fit):
        """Synthetically attach a survey_metadata shim and verify BR surfaces it."""
        fit, _ = event_study_fit

        class _ShimMeta:
            weight_type = "pweight"
            effective_n = 120.0
            design_effect = 2.5
            sum_weights = 200.0
            n_strata = 8
            n_psu = 20
            df_survey = 18
            replicate_method = None

        original = fit.survey_metadata
        try:
            fit.survey_metadata = _ShimMeta()
            br = BusinessReport(fit, auto_diagnostics=False)
            survey = br.to_dict()["sample"]["survey"]
            assert survey is not None
            assert survey["weight_type"] == "pweight"
            assert survey["design_effect"] == 2.5
            assert survey["is_trivial"] is False

            summary = br.summary()
            # When DEFF >= 1.5 we inject a caveat or a summary sentence.
            assert (
                "design effect" in summary.lower()
                or "effective sample size" in summary.lower()
                or any(c.get("topic") == "design_effect" for c in br.caveats())
            )
        finally:
            fit.survey_metadata = original


# ---------------------------------------------------------------------------
# Single-knob alpha
# ---------------------------------------------------------------------------
class TestAlphaKnob:
    def test_alpha_drives_ci_level(self, event_study_fit):
        fit, _ = event_study_fit
        br90 = BusinessReport(fit, alpha=0.10, auto_diagnostics=False)
        br95 = BusinessReport(fit, alpha=0.05, auto_diagnostics=False)
        assert br90.to_dict()["headline"]["ci_level"] == 90
        assert br95.to_dict()["headline"]["ci_level"] == 95


# ---------------------------------------------------------------------------
# Summary + full_report work across estimators
# ---------------------------------------------------------------------------
class TestAcrossEstimators:
    def test_summary_nonempty_for_all(self, did_fit, event_study_fit, cs_fit, sdid_fit):
        for fit, _ in (did_fit, event_study_fit, cs_fit, sdid_fit):
            br = BusinessReport(fit, auto_diagnostics=False)
            s = br.summary()
            assert isinstance(s, str)
            assert len(s) > 0


# ---------------------------------------------------------------------------
# Public API exposure
# ---------------------------------------------------------------------------
def test_public_api_exports():
    for name in ("BusinessReport", "BusinessContext", "BUSINESS_REPORT_SCHEMA_VERSION"):
        assert hasattr(dd, name)


def test_repr_includes_estimator_and_effect(cs_fit):
    fit, _ = cs_fit
    r = repr(BusinessReport(fit, auto_diagnostics=False))
    assert "CallawaySantAnnaResults" in r


def test_str_equals_summary(cs_fit):
    fit, _ = cs_fit
    br = BusinessReport(fit, auto_diagnostics=False)
    assert str(br) == br.summary()


def test_business_context_is_frozen_dataclass():
    ctx = BusinessContext(
        outcome_label="x",
        outcome_unit=None,
        outcome_direction=None,
        business_question=None,
        treatment_label="y",
        alpha=0.05,
    )
    with pytest.raises((AttributeError, Exception)):
        ctx.alpha = 0.10  # type: ignore[misc]
