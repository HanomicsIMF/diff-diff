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
    BusinessContext,
    BusinessReport,
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
    # base_period='universal' so DR's sensitivity check can run without
    # hitting the round-5 methodology-critical skip (Rambachan-Roth bounds
    # are not interpretable on consecutive-comparison pre-periods).
    cs = CallawaySantAnna(base_period="universal").fit(
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


@pytest.fixture(scope="module")
def edid_fit():
    from diff_diff import EfficientDiD

    sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
    edid = EfficientDiD().fit(
        sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
    )
    return edid, sdf


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
class TestOutcomeDirection:
    """outcome_direction selects value-laden vs neutral verbs."""

    def test_higher_is_better_positive_effect_uses_lifted(self, cs_fit):
        fit, _ = cs_fit
        br = BusinessReport(
            fit,
            outcome_label="sales",
            outcome_unit="$",
            outcome_direction="higher_is_better",
            treatment_label="the policy",
            auto_diagnostics=False,
        )
        headline = br.headline()
        assert "lifted" in headline
        assert "increased" not in headline

    def test_lower_is_better_positive_effect_uses_worsened(self, cs_fit):
        fit, _ = cs_fit  # CS has a positive effect on this seed
        br = BusinessReport(
            fit,
            outcome_label="churn",
            outcome_unit="%",
            outcome_direction="lower_is_better",
            treatment_label="the change",
            auto_diagnostics=False,
        )
        headline = br.headline()
        assert "worsened" in headline

    def test_direction_none_uses_neutral_verb(self, cs_fit):
        fit, _ = cs_fit
        br = BusinessReport(
            fit,
            outcome_label="sales",
            outcome_unit="$",
            auto_diagnostics=False,
        )
        headline = br.headline()
        assert "increased" in headline
        assert "lifted" not in headline


class TestWarningsPassthrough:
    """Broad exception handling still records provenance in schema.warnings."""

    def test_diagnostic_error_surfaces_as_top_level_warning(self, event_study_fit):
        fit, _ = event_study_fit

        def _raise(*args, **kwargs):
            raise RuntimeError("synthetic test failure")

        with patch("diff_diff.honest_did.HonestDiD.sensitivity_analysis", side_effect=_raise):
            br = BusinessReport(fit, auto_diagnostics=True)
            schema = br.to_dict()
            inner = schema["diagnostics"]["schema"]
            # The error is recorded at the section level...
            assert inner["sensitivity"]["status"] == "error"
            # ...AND surfaced at the top level for quick scanning.
            assert any("sensitivity:" in w for w in inner["warnings"])
            assert any("synthetic test failure" in w for w in inner["warnings"])


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
    def test_alpha_equal_to_result_alpha_drives_ci_level(self, event_study_fit):
        """When caller's alpha matches the fit's native alpha, ``ci_level``
        reflects that alpha (e.g., alpha=0.05 -> 95% CI)."""
        fit, _ = event_study_fit
        br = BusinessReport(fit, alpha=0.05, auto_diagnostics=False)
        assert br.to_dict()["headline"]["ci_level"] == 95

    def test_alpha_mismatch_preserves_fitted_ci_at_native_level(self, event_study_fit):
        """Round-7 regression: a caller alpha that differs from the fit's
        native alpha must NOT recompute a z-based CI (the fit used t-based
        inference with a finite ``df`` that BR cannot reproduce from
        ``(att, se)`` alone). The displayed CI stays at the fit's native
        level, while significance phrasing uses the caller's alpha. A
        caveat records the override.
        """
        import math

        fit, _ = event_study_fit
        br95 = BusinessReport(fit, alpha=0.05, auto_diagnostics=False)
        br90 = BusinessReport(fit, alpha=0.10, auto_diagnostics=False)
        h95 = br95.to_dict()["headline"]
        h90 = br90.to_dict()["headline"]
        if h95["effect"] is not None and math.isfinite(h95["effect"]):
            # Bounds must match between the two: the alpha=0.10 call
            # preserves the fit's 95% CI rather than recomputing a 90% z-CI.
            assert h90["ci_lower"] == pytest.approx(h95["ci_lower"])
            assert h90["ci_upper"] == pytest.approx(h95["ci_upper"])
        # ``ci_level`` stays at the fit's native level in both cases.
        assert h95["ci_level"] == 95
        assert h90["ci_level"] == 95
        # Override is surfaced as an info-level caveat.
        topics = {c.get("topic") for c in br90.caveats()}
        assert "alpha_override_preserved" in topics, (
            "Alpha mismatch must surface a caveat documenting the preserved "
            "native CI level; topics seen: " + str(topics)
        )


class TestAlphaOverrideBootstrapAndFiniteDF:
    """Alpha override preserves the fitted CI on any inference contract
    that cannot be reproduced from point-estimate + SE alone (bootstrap /
    wild cluster bootstrap / percentile / jackknife / placebo / finite-df
    survey / undefined-d.f. replicate / analytical t-quantile). The
    displayed CI stays at the fit's native level; significance phrasing
    still uses the caller's alpha; an informational caveat records the
    override.
    """

    class _BootstrapResultStub:
        """Minimal stub shaped like a bootstrap-inferred result."""

        def __init__(self):
            self.att = 1.0
            self.se = 0.5
            self.p_value = 0.04
            # Original 95% CI from the bootstrap distribution.
            self.conf_int = (0.05, 1.95)
            self.alpha = 0.05
            self.n_obs = 100
            self.n_treated = 40
            self.n_control = 60
            self.inference_method = "bootstrap"
            self.survey_metadata = None
            # Presence of a bootstrap distribution triggers the preserve path.
            import numpy as np

            self.bootstrap_distribution = np.random.default_rng(0).normal(1.0, 0.5, 200)

    def test_bootstrap_fit_preserves_fitted_ci_on_alpha_mismatch(self):
        stub = self._BootstrapResultStub()
        br = BusinessReport(stub, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        # Native fit was at 95%; requested 90% should NOT be reflected in the label.
        assert h["ci_level"] == 95, (
            "Bootstrap fit must preserve fitted CI level (95) when caller "
            f"requests a different alpha; got {h['ci_level']}"
        )
        # Bounds should match the stored bootstrap interval, not a normal-z
        # recomputation at 90%.
        assert h["ci_lower"] == pytest.approx(0.05)
        assert h["ci_upper"] == pytest.approx(1.95)
        # A caveat records the override.
        caveat_topics = {c.get("topic") for c in br.caveats()}
        assert "alpha_override_preserved" in caveat_topics

    class _FiniteDfSurveyStub:
        def __init__(self):
            from types import SimpleNamespace

            self.att = 2.0
            self.se = 0.4
            self.p_value = 0.001
            self.conf_int = (1.22, 2.78)  # 95% via survey t-quantile
            self.alpha = 0.05
            self.n_obs = 120
            self.n_treated = 50
            self.n_control = 70
            self.inference_method = "analytical"
            # Finite survey d.f. triggers the preserve path — normal approx
            # would widen / narrow incorrectly.
            self.survey_metadata = SimpleNamespace(
                weight_type="pweight",
                effective_n=110.0,
                design_effect=1.2,
                sum_weights=120.0,
                n_strata=4,
                n_psu=12,
                df_survey=8,
                replicate_method=None,
            )

    def test_finite_df_fit_preserves_fitted_ci_on_alpha_mismatch(self):
        stub = self._FiniteDfSurveyStub()
        br = BusinessReport(stub, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        assert h["ci_level"] == 95
        assert h["ci_lower"] == pytest.approx(1.22)
        assert h["ci_upper"] == pytest.approx(2.78)
        caveat_topics = {c.get("topic") for c in br.caveats()}
        assert "alpha_override_preserved" in caveat_topics


class TestWildBootstrapAlphaOverride:
    """Regression for the round-4 P0 finding that ``inference='wild_bootstrap'``
    results were falling through to a normal-approximation recomputation."""

    def test_wild_bootstrap_preserves_fitted_ci(self):
        class _WildBootstrapStub:
            def __init__(self):
                self.att = 1.0
                self.se = 0.5
                self.p_value = 0.04
                # 95% CI produced by the wild cluster bootstrap surface.
                self.conf_int = (0.10, 1.90)
                self.alpha = 0.05
                self.n_obs = 100
                self.n_treated = 40
                self.n_control = 60
                self.inference_method = "wild_bootstrap"
                self.survey_metadata = None
                # Wild-boot fits don't necessarily carry a raw distribution;
                # the inference_method string alone must be enough.
                self.bootstrap_distribution = None

        stub = _WildBootstrapStub()
        br = BusinessReport(stub, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        assert h["ci_level"] == 95, (
            "Wild cluster bootstrap must preserve fitted CI level on alpha "
            f"mismatch; got {h['ci_level']}"
        )
        assert h["ci_lower"] == pytest.approx(0.10)
        assert h["ci_upper"] == pytest.approx(1.90)
        caveats = br.caveats()
        assert any(c.get("topic") == "alpha_override_preserved" for c in caveats)
        # Caveat message should call out wild cluster bootstrap specifically.
        preserved_msg = next(
            c["message"] for c in caveats if c.get("topic") == "alpha_override_preserved"
        )
        assert "wild cluster bootstrap" in preserved_msg


class TestAssumptionBlockSourceFaithful:
    """Regression for the round-4 P1 finding that ``_describe_assumption``
    was producing generic DiD PT text for ContinuousDiD, TripleDifference,
    and StaggeredTripleDifference — all of which have different identifying
    logic per the Methodology Registry."""

    def _stub(self, class_name):
        cls = type(class_name, (), {})
        obj = cls()
        obj.att = 1.0
        obj.se = 0.1
        obj.p_value = 0.001
        obj.conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 100
        obj.n_treated = 40
        obj.n_control = 60
        obj.survey_metadata = None
        obj.event_study_effects = None
        obj.inference_method = "analytical"
        return obj

    def test_continuous_did_assumption_uses_two_level_pt(self):
        br = BusinessReport(self._stub("ContinuousDiDResults"), auto_diagnostics=False)
        assumption = br.to_dict()["assumption"]
        assert assumption["parallel_trends_variant"] == "dose_pt_or_strong_pt"
        desc = assumption["description"]
        # Registry-backed language: PT vs Strong PT + ACRT mention.
        assert "Strong Parallel Trends" in desc or "SPT" in desc
        assert "ATT(d" in desc or "ACRT" in desc
        assert "Callaway" in desc  # attribution to CGBS 2024

    def test_triple_difference_assumption_uses_ddd_decomposition(self):
        class TripleDifferenceResults:
            pass

        obj = TripleDifferenceResults()
        obj.att = 1.0
        obj.se = 0.1
        obj.p_value = 0.001
        obj.conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 100
        obj.n_treated = 40
        obj.n_control = 60
        obj.survey_metadata = None
        obj.inference_method = "analytical"

        br = BusinessReport(obj, auto_diagnostics=False)
        assumption = br.to_dict()["assumption"]
        assert assumption["parallel_trends_variant"] == "triple_difference_cancellation"
        desc = assumption["description"]
        assert "DDD" in desc
        assert "Ortiz-Villavicencio" in desc or "2025" in desc

    def test_staggered_triple_diff_assumption_uses_ddd_not_generic_pt(self):
        class StaggeredTripleDiffResults:
            pass

        obj = StaggeredTripleDiffResults()
        obj.overall_att = 1.0
        obj.overall_se = 0.1
        obj.overall_p_value = 0.001
        obj.overall_conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 100
        obj.n_treated = 40
        obj.n_control = 60
        obj.survey_metadata = None
        obj.event_study_effects = None
        obj.inference_method = "analytical"

        br = BusinessReport(obj, auto_diagnostics=False)
        assumption = br.to_dict()["assumption"]
        assert assumption["parallel_trends_variant"] == "triple_difference_cancellation"
        desc = assumption["description"]
        assert "triple-difference" in desc.lower() or "DDD" in desc
        # Must NOT be the generic group-time PT text.
        assert "group-time ATT" not in desc


class TestEfficientDiDAssumptionPtAllPtPost:
    """Round-8 regression: EfficientDiD has two distinct PT regimes
    (PT-All and PT-Post, per Chen-Sant'Anna-Xie 2025 Corollary 3.2 and
    Lemma 2.1). The old generic group-time PT text was source-unfaithful;
    the assumption block must now read ``results.pt_assumption`` and
    branch on it.
    """

    def _stub(self, pt_assumption: str, control_group: str = "never_treated"):
        """Build an EfficientDiD-shaped stub. ``control_group`` defaults to
        ``"never_treated"`` (the estimator's actual default); the only other
        accepted value is ``"last_cohort"`` (pseudo-never-treated). The
        earlier ``"not_yet_treated"`` default was invalid for this estimator
        and was flagged in round-10 CI review."""

        class EfficientDiDResults:
            pass

        stub = EfficientDiDResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.2
        stub.overall_p_value = 0.001
        stub.overall_conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.survey_metadata = None
        stub.event_study_effects = None
        stub.inference_method = "analytical"
        stub.pt_assumption = pt_assumption
        stub.control_group = control_group
        return stub

    def test_pt_all_uses_pt_all_language(self):
        br = BusinessReport(self._stub("all"), auto_diagnostics=False)
        a = br.to_dict()["assumption"]
        assert a["parallel_trends_variant"] == "pt_all"
        assert "PT-All" in a["description"]
        assert "Hausman" in a["description"]
        # Must NOT be the old generic group-time PT text.
        assert "group-time ATT" not in a["description"]

    def test_pt_post_uses_pt_post_language(self):
        br = BusinessReport(self._stub("post"), auto_diagnostics=False)
        a = br.to_dict()["assumption"]
        assert a["parallel_trends_variant"] == "pt_post"
        assert "PT-Post" in a["description"]
        assert "Corollary 3.2" in a["description"] or "single-baseline" in a["description"]

    def test_pt_post_never_treated_names_never_treated(self):
        """Default control_group: description must say never-treated."""
        br = BusinessReport(self._stub("post", "never_treated"), auto_diagnostics=False)
        desc = br.to_dict()["assumption"]["description"]
        assert "never-treated" in desc
        assert "latest treated cohort" not in desc

    def test_pt_post_last_cohort_branch_describes_pseudo_control(self):
        """Round-10 regression: ``control_group='last_cohort'`` must not be
        narrated with generic never-treated language. The description must
        describe the pseudo-never-treated latest-cohort design (REGISTRY.md
        §EfficientDiD line 908)."""
        br = BusinessReport(self._stub("post", "last_cohort"), auto_diagnostics=False)
        desc = br.to_dict()["assumption"]["description"]
        assert "latest treated cohort" in desc
        assert "pseudo-never-treated" in desc
        assert "dropped" in desc

    def test_pt_all_last_cohort_branch_describes_pseudo_control(self):
        br = BusinessReport(self._stub("all", "last_cohort"), auto_diagnostics=False)
        desc = br.to_dict()["assumption"]["description"]
        assert "latest treated cohort" in desc
        assert "pseudo-never-treated" in desc

    def test_control_group_is_reflected_in_block(self):
        br = BusinessReport(self._stub("all", "last_cohort"), auto_diagnostics=False)
        a = br.to_dict()["assumption"]
        assert a.get("control_group") == "last_cohort"


class TestMethodAwarePTProse:
    """Round-8 regression: BR and DR summary prose must branch on the
    ``parallel_trends.method`` field. Generic "pre-treatment event-study
    coefficients" wording is wrong for the 2x2 ``slope_difference`` path
    and for EfficientDiD's ``hausman`` PT-All vs PT-Post pretest.
    """

    def test_br_summary_uses_slope_difference_wording_for_simple_did(self):
        """Use a stub DR schema with a known slope_difference verdict so
        the test is deterministic across pre-period counts. The real
        2x2 fit can produce NaN verdicts when there is only one
        pre-period, so we don't rely on a real DR here."""

        class DiDResults:
            pass

        stub = DiDResults()
        stub.att = 1.0
        stub.se = 0.2
        stub.p_value = 0.001
        stub.conf_int = (0.6, 1.4)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.survey_metadata = None
        stub.inference_method = "analytical"

        # Hand-crafted DR schema with ``method = "slope_difference"``.
        from diff_diff.diagnostic_report import DiagnosticReportResults

        fake_schema = {
            "schema_version": "1.0",
            "estimator": "DiDResults",
            "headline_metric": {"name": "att", "value": 1.0},
            "parallel_trends": {
                "status": "ran",
                "method": "slope_difference",
                "joint_p_value": 0.40,
                "verdict": "no_detected_violation",
            },
            "pretrends_power": {"status": "not_applicable"},
            "sensitivity": {"status": "not_applicable"},
            "placebo": {"status": "skipped", "reason": "opt-in"},
            "bacon": {"status": "not_applicable"},
            "design_effect": {"status": "not_applicable"},
            "heterogeneity": {"status": "not_applicable"},
            "epv": {"status": "not_applicable"},
            "estimator_native_diagnostics": {"status": "not_applicable"},
            "skipped": {},
            "warnings": [],
            "overall_interpretation": "",
            "next_steps": [],
        }
        fake_dr_results = DiagnosticReportResults(
            schema=fake_schema,
            interpretation="",
            applicable_checks=("parallel_trends",),
            skipped_checks={},
            warnings=(),
        )
        br = BusinessReport(stub, diagnostics=fake_dr_results)
        summary = br.summary()
        pt_method = br.to_dict()["pre_trends"].get("method")
        assert pt_method == "slope_difference"
        # Must NOT use the generic event-study wording.
        assert "event-study coefficients" not in summary
        # Must use the slope-difference subject phrase.
        assert "slope-difference" in summary

    def test_dr_summary_uses_hausman_wording_for_efficient_did(self, edid_fit):
        from diff_diff import DiagnosticReport

        fit, sdf = edid_fit
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        summary = dr.summary()
        pt = dr.to_dict()["parallel_trends"]
        # EfficientDiD's PT check routes through hausman_pretest.
        assert pt.get("method") == "hausman"
        # The generic event-study wording must not appear for this path.
        assert "event-study coefficients" not in summary


class TestHausmanPretestPropagatesFitDesign:
    """Round-9 regression: ``_pt_hausman`` must propagate the fitted
    result's ``control_group`` and ``anticipation`` into
    ``EfficientDiD.hausman_pretest`` so the pretest diagnoses the same
    design as the estimate being summarized. Rerunning with defaults
    would silently change the identification regime.
    """

    def _real_edid_fit(self):
        from diff_diff import EfficientDiD

        sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
        edid = EfficientDiD().fit(
            sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        # Force non-default design knobs on the result so the regression
        # exercises propagation even when the constructor used defaults.
        edid.control_group = "last_cohort"
        edid.anticipation = 1
        return edid, sdf

    def test_hausman_pretest_receives_control_group_and_anticipation(self):
        from diff_diff import DiagnosticReport

        fit, sdf = self._real_edid_fit()
        captured: dict = {}

        def _fake_hausman(*args, **kwargs):
            captured.update(kwargs)

            class _Result:
                statistic = 0.0
                p_value = 0.5
                df = 1

            return _Result()

        with patch(
            "diff_diff.efficient_did.EfficientDiD.hausman_pretest",
            side_effect=_fake_hausman,
        ):
            DiagnosticReport(
                fit,
                data=sdf,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            ).run_all()

        assert (
            captured.get("control_group") == "last_cohort"
        ), f"control_group must propagate from the fit; got {captured}"
        assert (
            captured.get("anticipation") == 1
        ), f"anticipation must propagate from the fit; got {captured}"


class TestHausmanFitFaithfulSkip:
    """Round-10 regression: DR / survey-weighted EfficientDiD fits cannot
    replay the Hausman pretest from ``(data, outcome, unit, time,
    first_treat)`` alone because the result does not expose ``covariates``,
    ``cluster``, nuisance kwargs, or the full survey design. DR must skip
    with an explicit reason rather than rerunning defaults.
    """

    def _make_fit(self, *, estimation_path="nocov", survey_metadata=None):
        from diff_diff import EfficientDiD

        sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
        edid = EfficientDiD().fit(
            sdf, outcome="outcome", unit="unit", time="period", first_treat="first_treat"
        )
        edid.estimation_path = estimation_path
        edid.survey_metadata = survey_metadata
        return edid, sdf

    def test_dr_covariate_path_skipped_with_reason(self):
        from diff_diff import DiagnosticReport

        fit, sdf = self._make_fit(estimation_path="dr")
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert "parallel_trends" not in dr.applicable_checks
        reason = dr.skipped_checks.get("parallel_trends", "")
        assert "doubly-robust" in reason

    def test_survey_weighted_fit_skipped_with_reason(self):
        from types import SimpleNamespace

        from diff_diff import DiagnosticReport

        fake_survey = SimpleNamespace(
            weight_type="pweight",
            effective_n=80.0,
            design_effect=1.25,
            sum_weights=100.0,
            n_strata=None,
            n_psu=None,
            df_survey=40,
        )
        fit, sdf = self._make_fit(survey_metadata=fake_survey)
        dr = DiagnosticReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        assert "parallel_trends" not in dr.applicable_checks
        reason = dr.skipped_checks.get("parallel_trends", "")
        assert "survey design" in reason


class TestHausmanPretestPropagatesCluster:
    """Round-11 regression: ``EfficientDiDResults`` now persists the
    ``cluster`` column used at fit time, and ``_pt_hausman`` forwards
    it to ``EfficientDiD.hausman_pretest``. Without this, clustered
    fits would be replayed under unclustered inference, silently
    publishing an H statistic / p-value for the wrong design.
    """

    def test_hausman_pretest_receives_cluster_kwarg(self):
        import pandas as pd

        from diff_diff import DiagnosticReport, EfficientDiD

        sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
        # Add a cluster column (e.g., region) to the panel.
        sdf = pd.DataFrame(sdf).copy()
        sdf["cluster_col"] = sdf["unit"] % 10

        edid = EfficientDiD(cluster="cluster_col").fit(
            sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        # Confirm persistence landed.
        assert getattr(edid, "cluster", None) == "cluster_col"

        captured: dict = {}

        def _fake_hausman(*args, **kwargs):
            captured.update(kwargs)

            class _Result:
                statistic = 0.0
                p_value = 0.5
                df = 1

            return _Result()

        with patch(
            "diff_diff.efficient_did.EfficientDiD.hausman_pretest",
            side_effect=_fake_hausman,
        ):
            DiagnosticReport(
                edid,
                data=sdf,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            ).run_all()

        assert (
            captured.get("cluster") == "cluster_col"
        ), f"cluster column must propagate from fit to Hausman pretest; got {captured}"


class TestCSNotYetTreatedControlGroupSemantics:
    """Round-13 P1 regression: ``BusinessReport`` must not relabel
    ``n_control_units`` as generic "control" for a
    ``CallawaySantAnna(control_group='not_yet_treated')`` fit — that
    field counts only never-treated units, while the actual comparison
    group is the dynamic not-yet-treated set at each (g, t) cell.
    """

    def test_not_yet_treated_fit_does_not_render_misleading_control_count(self):
        sdf = generate_staggered_data(n_units=100, n_periods=6, treatment_effect=1.5, seed=7)
        # Fit with the dynamic not-yet-treated comparison mode.
        cs = CallawaySantAnna(base_period="universal", control_group="not_yet_treated").fit(
            sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="event_study",
        )
        br = BusinessReport(cs, auto_diagnostics=False)
        sample = br.to_dict()["sample"]

        # Fixed ``n_control`` must NOT be populated — the comparison set
        # is dynamic per (g, t), not a fixed unit tally.
        assert (
            sample["n_control"] is None
        ), f"n_control must be None for not_yet_treated; got {sample['n_control']}"
        # The new fields surface the real semantics.
        assert sample["control_group"] == "not_yet_treated"
        assert sample["n_never_treated"] == getattr(cs, "n_control_units", None)

        # Both summary and full_report must describe the dynamic
        # comparison group rather than asserting a misleading "control"
        # count.
        summary = br.summary()
        # No "(N treated, N control)" phrasing on this path.
        assert " control)" not in summary
        assert "not-yet-treated" in summary or "dynamic" in summary

        full = br.full_report()
        assert "- Control:" not in full or "not-yet-treated" in full
        assert "dynamic not-yet-treated" in full or "not-yet-treated" in full

    def test_never_treated_fit_still_shows_fixed_control_count(self, cs_fit):
        """Default path (``control_group='never_treated'``) keeps the
        fixed ``n_control`` tally so existing prose is unchanged."""
        fit, _ = cs_fit  # default is never_treated
        br = BusinessReport(fit, auto_diagnostics=False)
        sample = br.to_dict()["sample"]
        assert isinstance(sample["n_control"], int)
        assert sample["control_group"] == "never_treated"


class TestBRDataKwargsPassthroughToAutoDR:
    """Round-12 regression: ``BusinessReport`` now accepts
    ``data`` / ``outcome`` / ``treatment`` / ``unit`` / ``time`` /
    ``first_treat`` kwargs and forwards them to the auto-constructed
    ``DiagnosticReport``. Without this, data-dependent checks (2x2 PT,
    Bacon, EfficientDiD Hausman) are silently skipped on the zero-
    config auto path even though the README markets one-call
    diagnostics from a fitted result.
    """

    def test_did_fit_gets_2x2_pt_via_passthrough(self, did_fit):
        fit, df = did_fit
        br = BusinessReport(
            fit,
            data=df,
            outcome="outcome",
            treatment="treated",
            time="post",
        )
        # Auto-DR received the kwargs and ran the 2x2 PT check.
        dr_schema = br.to_dict()["diagnostics"]["schema"]
        assert dr_schema["parallel_trends"]["status"] == "ran"
        assert dr_schema["parallel_trends"]["method"] == "slope_difference"

    def test_cs_fit_gets_bacon_via_passthrough(self, cs_fit):
        fit, sdf = cs_fit
        br = BusinessReport(
            fit,
            data=sdf,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
        )
        dr_schema = br.to_dict()["diagnostics"]["schema"]
        # Bacon needs data + outcome + time + unit + first_treat; before
        # the passthrough, the auto path skipped because only the
        # estimator result was available.
        assert dr_schema["bacon"]["status"] == "ran"

    def test_no_passthrough_still_works_and_skips_gracefully(self, did_fit):
        """Zero-config auto path must still produce a valid report; it
        just skips data-dependent checks."""
        fit, _ = did_fit
        br = BusinessReport(fit)  # no data kwargs
        dr_schema = br.to_dict()["diagnostics"]["schema"]
        # PT needs data for 2x2 and was gated out of applicable — section
        # is "skipped" rather than "ran".
        assert dr_schema["parallel_trends"]["status"] in {"skipped", "not_applicable"}


class TestSensitivityProseGuarding:
    """Round-12 regression: BR / DR summary prose must not promise a
    "sensitivity analysis below" sentence when no sensitivity block
    actually ran (e.g., SDiD / TROP routed to native diagnostics,
    single-M precomputed passthrough rendered separately, skipped
    sensitivity for varying-base CS).
    """

    def test_br_sdid_does_not_mention_sensitivity_below(self, sdid_fit):
        fit, _ = sdid_fit
        summary = BusinessReport(fit).summary()
        # SDiD routes to estimator-native diagnostics, not HonestDiD.
        # The PT verdict for SDiD is ``design_enforced_pt`` which does
        # not append any "see sensitivity below" clause, so the prose
        # should not mention it.
        assert "sensitivity analysis below" not in summary

    def test_dr_trop_does_not_mention_sensitivity_below(self, sdid_fit):
        # SDiD and TROP both skip HonestDiD. Use SDiD as proxy here
        # since it already has a fixture; the same guard covers TROP.
        from diff_diff import DiagnosticReport

        fit, _ = sdid_fit
        summary = DiagnosticReport(fit).summary()
        assert "sensitivity analysis below" not in summary


class TestHausmanTestStatisticPopulated:
    """Round-10 P3 regression: ``HausmanPretestResult`` exposes
    ``statistic`` (not ``test_statistic``); the DR schema was previously
    reading the wrong attribute and losing the H statistic."""

    def test_test_statistic_field_is_populated_on_success(self, edid_fit):
        from diff_diff import DiagnosticReport

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
        if pt["status"] == "ran":
            # Method-specific: only Hausman exposes a test_statistic.
            assert pt["method"] == "hausman"
            ts = pt["test_statistic"]
            assert (
                ts is not None and isinstance(ts, float) and np.isfinite(ts)
            ), f"Hausman H statistic must be populated on success; got {ts}"


class TestFullReportSingleM:
    """Regression: ``full_report()`` must not claim full-grid robustness for a
    single-M HonestDiDResults passthrough. The summary path was fixed earlier;
    the structured-markdown path had the same bug and now mirrors it."""

    @staticmethod
    def _fake_single_m(M=1.5, ci_lb=1.0, ci_ub=3.0):
        from types import SimpleNamespace

        return SimpleNamespace(
            M=M,
            lb=ci_lb,
            ub=ci_ub,
            ci_lb=ci_lb,
            ci_ub=ci_ub,
            method="relative_magnitude",
            alpha=0.05,
        )

    def test_full_report_does_not_claim_full_grid_for_single_m(self, event_study_fit):
        fit, _ = event_study_fit
        br = BusinessReport(fit, honest_did_results=self._fake_single_m())
        md = br.full_report()
        assert "robust across full grid" not in md
        assert "Single point checked" in md or "single point" in md.lower()


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


class TestBootstrapResultsAndNBootstrapDetection:
    """Regression for the round-5 P0 finding that ``_extract_headline``
    only preserved native CI surfaces when a result advertised
    ``inference_method`` / ``bootstrap_distribution`` / ``variance_method``
    / ``df_survey``.

    Several staggered / continuous / dCDH result classes copy bootstrap-
    derived se/p/conf_int into their top-level fields at fit time and
    expose the bootstrap only via a ``bootstrap_results`` sub-object or
    an ``n_bootstrap > 0`` attribute. An ``alpha`` override on such a
    fit would silently swap a percentile/multiplier bootstrap CI for a
    normal-approximation one. BR must now detect either marker and
    preserve the fitted CI at its native level.
    """

    def _base_stub(self):
        stub = type("Stub", (), {})()
        stub.att = 1.0
        stub.se = 0.5
        stub.p_value = 0.04
        stub.conf_int = (0.05, 1.95)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.survey_metadata = None
        # Crucially NOT exposing inference_method / bootstrap_distribution
        # / variance_method / df_survey: exactly the surface the reviewer
        # flagged as silently falling through.
        return stub

    def test_bootstrap_results_object_alone_preserves_fit_ci(self):
        stub = self._base_stub()
        stub.bootstrap_results = type("BootSub", (), {"n_bootstrap": 199})()
        br = BusinessReport(stub, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        assert h["ci_level"] == 95, (
            "Result carrying bootstrap_results must preserve fitted CI "
            "level on alpha mismatch; got " + str(h["ci_level"])
        )
        assert h["ci_lower"] == pytest.approx(0.05)
        assert h["ci_upper"] == pytest.approx(1.95)
        topics = {c.get("topic") for c in br.caveats()}
        assert "alpha_override_preserved" in topics

    def test_n_bootstrap_positive_alone_preserves_fit_ci(self):
        """ContinuousDiDResults-style: ``n_bootstrap`` field, no bootstrap_results."""
        stub = self._base_stub()
        stub.n_bootstrap = 499
        br = BusinessReport(stub, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        assert h["ci_level"] == 95
        assert h["ci_lower"] == pytest.approx(0.05)
        assert h["ci_upper"] == pytest.approx(1.95)
        topics = {c.get("topic") for c in br.caveats()}
        assert "alpha_override_preserved" in topics

    def test_n_bootstrap_zero_still_preserves_on_alpha_mismatch(self):
        """Analytic fits (``n_bootstrap = 0``) also preserve the fitted CI
        on alpha mismatch — BR cannot reproduce a ``DiDResults`` /
        ``MultiPeriodDiDResults`` / TROP t-quantile CI without the fit's
        finite ``df``, which is not exposed uniformly. Round-7 regression.
        """
        stub = self._base_stub()
        stub.n_bootstrap = 0
        br = BusinessReport(stub, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        # Analytic fit's native 95% CI is preserved at 95% on 90% override.
        assert h["ci_level"] == 95
        assert h["ci_lower"] == pytest.approx(0.05)
        assert h["ci_upper"] == pytest.approx(1.95)
        topics = {c.get("topic") for c in br.caveats()}
        assert "alpha_override_preserved" in topics

    def test_dcdh_shaped_bootstrap_stub_preserves_fit_ci(self):
        """dCDH copies bootstrap se/p/conf_int into top-level fields without
        ``inference_method``. The reviewer called this out specifically."""

        class ChaisemartinDHaultfoeuilleResults:  # name-keyed dispatch
            pass

        stub = ChaisemartinDHaultfoeuilleResults()
        stub.att = 1.5
        stub.se = 0.4
        stub.p_value = 0.02
        stub.conf_int = (0.72, 2.28)
        stub.alpha = 0.05
        stub.n_obs = 200
        stub.n_treated = 80
        stub.n_control = 120
        stub.survey_metadata = None
        stub.event_study_effects = None
        stub.placebo_event_study = None
        # dCDH carries bootstrap via a sub-object; top-level fields are
        # the bootstrap-derived values, not analytic.
        stub.bootstrap_results = type("DCDHBoot", (), {"n_bootstrap": 499})()

        br = BusinessReport(stub, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        assert h["ci_level"] == 95
        assert h["ci_lower"] == pytest.approx(0.72)
        assert h["ci_upper"] == pytest.approx(2.28)


class TestAnalyticalFiniteDfAlphaOverride:
    """Round-7 regressions for the P0 finding that
    ``_extract_headline`` was recomputing a normal-z CI on alpha
    mismatch for analytical fits whose native inference used a finite
    ``df`` (``DifferenceInDifferences`` / ``MultiPeriodDiD`` / TROP)
    that BR cannot reproduce from ``(att, se)`` alone. The fix is to
    always preserve the fitted CI on alpha mismatch.
    """

    def test_analytical_did_result_preserves_native_ci(self):
        from diff_diff import DifferenceInDifferences, generate_did_data

        df = generate_did_data(n_units=80, n_periods=4, treatment_effect=1.5, seed=7)
        fit = DifferenceInDifferences().fit(df, outcome="outcome", treatment="treated", time="post")
        native_lo, native_hi = fit.conf_int

        br = BusinessReport(fit, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        # Native 95% CI preserved — no z-based recomputation.
        assert h["ci_level"] == 95
        assert h["ci_lower"] == pytest.approx(native_lo)
        assert h["ci_upper"] == pytest.approx(native_hi)
        topics = {c.get("topic") for c in br.caveats()}
        assert "alpha_override_preserved" in topics

    def test_multiperiod_preserves_native_ci_on_alpha_override(self):
        from diff_diff import MultiPeriodDiD, generate_did_data

        df = generate_did_data(n_units=80, n_periods=8, treatment_effect=1.5, seed=7)
        fit = MultiPeriodDiD().fit(
            df,
            outcome="outcome",
            treatment="treated",
            time="period",
            unit="unit",
            reference_period=3,
        )
        native_lo, native_hi = fit.avg_conf_int

        br = BusinessReport(fit, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        assert h["ci_level"] == 95
        assert h["ci_lower"] == pytest.approx(native_lo)
        assert h["ci_upper"] == pytest.approx(native_hi)

    def test_undefined_df_survey_stub_does_not_invent_finite_ci(self):
        """When the fit's native inference returned NaN (rank-deficient
        replicate design: ``df_survey = 0``), BR must not recompute a
        finite interval — the NaN signal must propagate through."""
        from types import SimpleNamespace

        class _UndefinedDfStub:
            pass

        stub = _UndefinedDfStub()
        stub.att = 1.0
        stub.se = float("nan")
        stub.p_value = float("nan")
        stub.conf_int = (float("nan"), float("nan"))
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.inference_method = "analytical"
        stub.survey_metadata = SimpleNamespace(
            weight_type="replicate",
            replicate_method="JK1",
            effective_n=80.0,
            design_effect=1.25,
            sum_weights=100.0,
            n_strata=None,
            n_psu=None,
            df_survey=0,
        )

        br = BusinessReport(stub, alpha=0.10, auto_diagnostics=False)
        h = br.to_dict()["headline"]
        # NaN bounds must propagate — BR must not invent a finite CI.
        lo, hi = h["ci_lower"], h["ci_upper"]
        assert lo is None or not np.isfinite(lo), f"ci_lower should be NaN/None, got {lo}"
        assert hi is None or not np.isfinite(hi), f"ci_upper should be NaN/None, got {hi}"


class TestDCDHAssumptionTransitionBased:
    """Regression for the round-5 P1 finding that
    ``ChaisemartinDHaultfoeuilleResults`` was narrated with generic group-
    time PT text instead of source-backed transition-based identification.
    """

    def test_dcdh_uses_transition_based_language(self):
        class ChaisemartinDHaultfoeuilleResults:
            pass

        obj = ChaisemartinDHaultfoeuilleResults()
        obj.att = 1.0
        obj.se = 0.1
        obj.p_value = 0.001
        obj.conf_int = (0.8, 1.2)
        obj.alpha = 0.05
        obj.n_obs = 100
        obj.n_treated = 40
        obj.n_control = 60
        obj.survey_metadata = None
        obj.event_study_effects = None
        obj.placebo_event_study = None
        obj.inference_method = "analytical"

        br = BusinessReport(obj, auto_diagnostics=False)
        assumption = br.to_dict()["assumption"]
        assert assumption["parallel_trends_variant"] == "transition_based"
        desc = assumption["description"]
        # Source-faithful: joiners/leavers/stable-control, dCDH paper attribution.
        assert "joiner" in desc.lower()
        assert "leaver" in desc.lower()
        assert "Chaisemartin" in desc or "D'Haultfoeuille" in desc
        # Must NOT open with the generic group-time PT framing. The text
        # may reference it inside a contrast clause ("not a single
        # group-time ATT PT"), which is fine and intended.
        assert not desc.startswith("Identification relies on parallel trends")


class TestCSVaryingBaseSensitivitySkipped:
    """Regression for the round-5 P1 finding that DR would narrate HonestDiD
    bounds as robust sensitivity for a CallawaySantAnna fit with
    ``base_period='varying'`` (the CS default). The HonestDiD helper
    explicitly warns that those bounds are not valid for interpretation;
    DR must preemptively skip and surface the reason."""

    def test_cs_varying_base_skips_sensitivity_with_reason(self):
        class CallawaySantAnnaResults:
            pass

        stub = CallawaySantAnnaResults()
        stub.overall_att = 1.0
        stub.overall_se = 0.3
        stub.overall_p_value = 0.01
        stub.overall_conf_int = (0.4, 1.6)
        stub.alpha = 0.05
        stub.n_obs = 100
        stub.n_treated = 40
        stub.n_control = 60
        stub.survey_metadata = None
        stub.event_study_effects = None
        stub.event_study_vcov = None
        stub.event_study_vcov_index = None
        stub.vcov = None
        stub.interaction_indices = None
        stub.base_period = "varying"
        stub.inference_method = "analytical"

        from diff_diff import DiagnosticReport

        dr = DiagnosticReport(stub).run_all()
        sens = dr.schema["sensitivity"]
        assert sens["status"] == "skipped"
        reason = sens["reason"]
        assert "base_period" in reason and "universal" in reason
        # And BR must surface this as a warning-severity caveat.
        br = BusinessReport(stub, diagnostics=dr)
        caveats = br.caveats()
        topics = {c.get("topic") for c in caveats}
        assert "sensitivity_skipped" in topics, (
            "BR must surface varying-base sensitivity skip as a caveat; " f"got topics {topics}"
        )
