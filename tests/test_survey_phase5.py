"""Tests for Phase 5 survey support: SyntheticDiD and TROP.

Covers: pweight-only survey integration for both estimators, including
point estimate weighting, bootstrap/placebo SE threading, survey_metadata
in results, error guards for unsupported designs, and scale invariance.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff import SurveyDesign, SyntheticDiD
from diff_diff.trop import TROP, trop

# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def sdid_survey_data():
    """Balanced panel for SDID with survey design columns.

    20 units (5 treated, 15 control), 10 periods, block treatment at period 6.
    Unit-constant weight column that varies across units.
    """
    np.random.seed(42)
    n_units = 20
    n_periods = 10
    n_treated = 5

    units = list(range(n_units))
    periods = list(range(n_periods))

    rows = []
    for u in units:
        is_treated = 1 if u < n_treated else 0
        base = np.random.randn() * 2
        for t in periods:
            y = base + 0.5 * t + np.random.randn() * 0.5
            if is_treated and t >= 6:
                y += 2.0  # treatment effect
            rows.append({"unit": u, "time": t, "outcome": y, "treated": is_treated})

    data = pd.DataFrame(rows)

    # Unit-constant survey columns
    unit_weight = 1.0 + np.arange(n_units) * 0.1  # [1.0, 1.1, ..., 2.9]
    unit_stratum = np.arange(n_units) // 10
    unit_psu = np.arange(n_units) // 5
    unit_map = {u: i for i, u in enumerate(units)}
    idx = data["unit"].map(unit_map).values

    data["weight"] = unit_weight[idx]
    data["stratum"] = unit_stratum[idx]
    data["psu"] = unit_psu[idx]

    return data


@pytest.fixture
def trop_survey_data():
    """Panel data for TROP with absorbing-state D and survey columns.

    20 units (5 treated starting at period 5), 10 periods.
    """
    np.random.seed(123)
    n_units = 20
    n_periods = 10
    n_treated = 5

    units = list(range(n_units))
    periods = list(range(n_periods))

    rows = []
    for u in units:
        is_treated_unit = u < n_treated
        base = np.random.randn() * 2
        for t in periods:
            y = base + 0.3 * t + np.random.randn() * 0.5
            # Absorbing state: D=1 for t >= 5 if treated unit
            d = 1 if (is_treated_unit and t >= 5) else 0
            if d == 1:
                y += 1.5  # treatment effect
            rows.append({"unit": u, "time": t, "outcome": y, "D": d})

    data = pd.DataFrame(rows)

    # Unit-constant survey columns
    unit_weight = 1.0 + np.arange(n_units) * 0.15
    unit_stratum = np.arange(n_units) // 10
    unit_psu = np.arange(n_units) // 5
    unit_map = {u: i for i, u in enumerate(units)}
    idx = data["unit"].map(unit_map).values

    data["weight"] = unit_weight[idx]
    data["stratum"] = unit_stratum[idx]
    data["psu"] = unit_psu[idx]

    return data


@pytest.fixture
def survey_design_weights():
    return SurveyDesign(weights="weight")


@pytest.fixture
def survey_design_full():
    return SurveyDesign(weights="weight", strata="stratum", psu="psu")


# =============================================================================
# SyntheticDiD Survey Tests
# =============================================================================


class TestSyntheticDiDSurvey:
    """Survey support tests for SyntheticDiD."""

    def test_smoke_weights_only(self, sdid_survey_data, survey_design_weights):
        """Fit completes and survey_metadata is populated."""
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        assert result.survey_metadata is not None
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)

    def test_uniform_weights_match_unweighted(self, sdid_survey_data):
        """Uniform weights (all 1.0) produce same ATT as unweighted."""
        sdid_survey_data = sdid_survey_data.copy()
        sdid_survey_data["uniform_w"] = 1.0

        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        result_no_survey = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
        )
        result_uniform = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=SurveyDesign(weights="uniform_w"),
        )
        assert result_uniform.att == pytest.approx(result_no_survey.att, abs=1e-10)

    def test_survey_metadata_fields(self, sdid_survey_data, survey_design_weights):
        """Metadata has correct weight_type, effective_n, design_effect."""
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        sm = result.survey_metadata
        assert sm.weight_type == "pweight"
        assert sm.effective_n > 0
        assert sm.design_effect > 0

    def test_strata_psu_fpc_raises(self, sdid_survey_data, survey_design_full):
        """Full design raises NotImplementedError."""
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        with pytest.raises(NotImplementedError, match="strata/PSU/FPC"):
            est.fit(
                sdid_survey_data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=survey_design_full,
            )

    def test_fweight_aweight_raises(self, sdid_survey_data):
        """Non-pweight raises ValueError."""
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        sd = SurveyDesign(weights="weight", weight_type="fweight")
        with pytest.raises(ValueError, match="pweight"):
            est.fit(
                sdid_survey_data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=sd,
            )

    def test_weighted_att_differs(self, sdid_survey_data, survey_design_weights):
        """Non-uniform weights produce different ATT than unweighted."""
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        result_no_survey = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
        )
        result_survey = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        # ATTs should differ since weights are non-uniform
        assert result_survey.att != pytest.approx(result_no_survey.att, abs=1e-6)

    def test_summary_includes_survey(self, sdid_survey_data, survey_design_weights):
        """summary() output contains Survey Design section."""
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        summary = result.summary()
        assert "Survey Design" in summary
        assert "pweight" in summary

    def test_bootstrap_with_survey(self, sdid_survey_data, survey_design_weights):
        """variance_method='bootstrap' completes with survey weights."""
        est = SyntheticDiD(variance_method="bootstrap", n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        assert np.isfinite(result.se)
        assert result.se > 0

    def test_placebo_with_survey(self, sdid_survey_data, survey_design_weights):
        """variance_method='placebo' completes with survey weights."""
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        assert np.isfinite(result.se)
        assert result.se > 0

    def test_weight_scale_invariance(self, sdid_survey_data, survey_design_weights):
        """Multiplying all weights by constant produces same ATT."""
        sdid_survey_data = sdid_survey_data.copy()
        sdid_survey_data["weight_2x"] = sdid_survey_data["weight"] * 2.0

        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        result_1x = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        result_2x = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=SurveyDesign(weights="weight_2x"),
        )
        assert result_2x.att == pytest.approx(result_1x.att, rel=1e-6)

    def test_unit_varying_survey_raises(self, sdid_survey_data):
        """Time-varying weight column raises ValueError."""
        sdid_survey_data = sdid_survey_data.copy()
        sdid_survey_data["bad_weight"] = sdid_survey_data["weight"] + sdid_survey_data["time"] * 0.1
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        with pytest.raises(ValueError):
            est.fit(
                sdid_survey_data,
                outcome="outcome",
                treatment="treated",
                unit="unit",
                time="time",
                post_periods=[6, 7, 8, 9],
                survey_design=SurveyDesign(weights="bad_weight"),
            )

    def test_to_dict_includes_survey(self, sdid_survey_data, survey_design_weights):
        """to_dict() output includes survey metadata fields."""
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        d = result.to_dict()
        assert "weight_type" in d
        assert d["weight_type"] == "pweight"

    def test_covariates_with_survey(self, sdid_survey_data, survey_design_weights):
        """Covariates + survey_design smoke test (WLS residualization)."""
        sdid_survey_data = sdid_survey_data.copy()
        sdid_survey_data["x1"] = np.random.randn(len(sdid_survey_data))
        est = SyntheticDiD(n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            covariates=["x1"],
            survey_design=survey_design_weights,
        )
        assert np.isfinite(result.att)
        assert result.survey_metadata is not None

    def test_effective_weights_returned(self, sdid_survey_data, survey_design_weights):
        """unit_weights returns composed ω_eff (not raw ω) under survey weighting."""
        est = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        result = est.fit(
            sdid_survey_data,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=survey_design_weights,
        )
        weights = result.unit_weights
        # Effective weights should sum to 1 (renormalized)
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-10)
        # With non-uniform survey weights, effective weights should differ
        # from what uniform survey weights would produce
        sdid_survey_data_u = sdid_survey_data.copy()
        sdid_survey_data_u["uniform_w"] = 1.0
        result_u = est.fit(
            sdid_survey_data_u,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="time",
            post_periods=[6, 7, 8, 9],
            survey_design=SurveyDesign(weights="uniform_w"),
        )
        # Non-uniform weights should change the returned weight distribution
        eff_vals = sorted(weights.values(), reverse=True)
        uni_vals = sorted(result_u.unit_weights.values(), reverse=True)
        assert eff_vals != pytest.approx(uni_vals, abs=1e-6)


# =============================================================================
# TROP Survey Tests
# =============================================================================


class TestTROPSurvey:
    """Survey support tests for TROP (local and global methods)."""

    def test_smoke_local_weights_only(self, trop_survey_data, survey_design_weights):
        """Local method completes with survey weights."""
        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        assert result.survey_metadata is not None
        assert np.isfinite(result.att)

    def test_smoke_global_weights_only(self, trop_survey_data, survey_design_weights):
        """Global method completes with survey weights."""
        est = TROP(method="global", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        assert result.survey_metadata is not None
        assert np.isfinite(result.att)

    def test_uniform_weights_match_local(self, trop_survey_data):
        """Uniform weights produce same ATT as unweighted (local)."""
        trop_survey_data = trop_survey_data.copy()
        trop_survey_data["uniform_w"] = 1.0

        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result_no_survey = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
        )
        result_uniform = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=SurveyDesign(weights="uniform_w"),
        )
        assert result_uniform.att == pytest.approx(result_no_survey.att, abs=1e-10)

    def test_uniform_weights_match_global(self, trop_survey_data):
        """Uniform weights produce same ATT as unweighted (global)."""
        trop_survey_data = trop_survey_data.copy()
        trop_survey_data["uniform_w"] = 1.0

        est = TROP(method="global", n_bootstrap=10, seed=42, max_iter=5)
        result_no_survey = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
        )
        result_uniform = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=SurveyDesign(weights="uniform_w"),
        )
        assert result_uniform.att == pytest.approx(result_no_survey.att, abs=1e-10)

    def test_survey_metadata_fields(self, trop_survey_data, survey_design_weights):
        """Metadata has correct fields."""
        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        sm = result.survey_metadata
        assert sm.weight_type == "pweight"
        assert sm.effective_n > 0

    def test_strata_psu_fpc_raises(self, trop_survey_data, survey_design_full):
        """Full design raises NotImplementedError."""
        est = TROP(method="local", n_bootstrap=10, seed=42)
        with pytest.raises(NotImplementedError, match="strata/PSU/FPC"):
            est.fit(
                trop_survey_data,
                outcome="outcome",
                treatment="D",
                unit="unit",
                time="time",
                survey_design=survey_design_full,
            )

    def test_fweight_aweight_raises(self, trop_survey_data):
        """Non-pweight raises ValueError."""
        est = TROP(method="local", n_bootstrap=10, seed=42)
        sd = SurveyDesign(weights="weight", weight_type="aweight")
        with pytest.raises(ValueError, match="pweight"):
            est.fit(
                trop_survey_data,
                outcome="outcome",
                treatment="D",
                unit="unit",
                time="time",
                survey_design=sd,
            )

    def test_weighted_att_differs(self, trop_survey_data, survey_design_weights):
        """Non-uniform weights change ATT."""
        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result_no = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
        )
        result_survey = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        assert result_survey.att != pytest.approx(result_no.att, abs=1e-6)

    def test_summary_includes_survey(self, trop_survey_data, survey_design_weights):
        """summary() contains survey section."""
        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        summary = result.summary()
        assert "Survey Design" in summary

    def test_weight_scale_invariance(self, trop_survey_data, survey_design_weights):
        """Scale invariance: 2x weights produce same ATT."""
        trop_survey_data = trop_survey_data.copy()
        trop_survey_data["weight_3x"] = trop_survey_data["weight"] * 3.0

        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result_1x = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        result_3x = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=SurveyDesign(weights="weight_3x"),
        )
        assert result_3x.att == pytest.approx(result_1x.att, rel=1e-6)

    def test_unit_varying_survey_raises(self, trop_survey_data):
        """Validation catches time-varying weights."""
        trop_survey_data = trop_survey_data.copy()
        trop_survey_data["bad_weight"] = trop_survey_data["weight"] + trop_survey_data["time"] * 0.1
        est = TROP(method="local", n_bootstrap=10, seed=42)
        with pytest.raises(ValueError):
            est.fit(
                trop_survey_data,
                outcome="outcome",
                treatment="D",
                unit="unit",
                time="time",
                survey_design=SurveyDesign(weights="bad_weight"),
            )

    def test_convenience_function_with_survey(self, trop_survey_data, survey_design_weights):
        """trop() convenience function accepts survey_design."""
        result = trop(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
            n_bootstrap=10,
            seed=42,
            max_iter=5,
        )
        assert result.survey_metadata is not None

    def test_to_dict_includes_survey(self, trop_survey_data, survey_design_weights):
        """to_dict() includes survey metadata fields."""
        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        d = result.to_dict()
        assert "weight_type" in d
        assert d["weight_type"] == "pweight"

    def test_local_bootstrap_nan_treated_outcomes(self, trop_survey_data):
        """Bootstrap handles NaN treated outcomes without poisoning SE."""
        trop_survey_data = trop_survey_data.copy()
        # Set some treated post-treatment outcomes to NaN
        mask = (trop_survey_data["D"] == 1) & (trop_survey_data["time"] == 7)
        trop_survey_data.loc[mask, "outcome"] = np.nan

        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
        )
        # Point estimate should use finite cells only
        assert np.isfinite(result.att)
        # SE should remain finite (not poisoned by NaN)
        assert np.isfinite(result.se)

    def test_local_bootstrap_nan_with_survey(self, trop_survey_data, survey_design_weights):
        """Bootstrap + survey handles NaN treated outcomes correctly."""
        trop_survey_data = trop_survey_data.copy()
        mask = (trop_survey_data["D"] == 1) & (trop_survey_data["time"] == 8)
        trop_survey_data.loc[mask, "outcome"] = np.nan

        est = TROP(method="local", n_bootstrap=10, seed=42, max_iter=5)
        result = est.fit(
            trop_survey_data,
            outcome="outcome",
            treatment="D",
            unit="unit",
            time="time",
            survey_design=survey_design_weights,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
