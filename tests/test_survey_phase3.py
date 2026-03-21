"""Tests for Phase 3 survey support: OLS-based standalone estimators.

Covers: StackedDiD, SunAbraham, BaconDecomposition, TripleDifference,
ContinuousDiD, EfficientDiD.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff import SurveyDesign

# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def staggered_survey_data():
    """Staggered treatment panel with survey design columns.

    60 units, 8 periods, 2 treatment cohorts (t=4, t=6), 20 never-treated.
    5 strata, 12 PSUs, FPC, and sampling weights included.
    """
    np.random.seed(42)
    n_units = 60
    n_periods = 8
    rows = []
    for unit in range(n_units):
        if unit < 20:
            ft = 4  # Early cohort
        elif unit < 40:
            ft = 6  # Late cohort
        else:
            ft = 0  # Never treated

        stratum = unit // 12  # 5 strata
        psu = unit // 5  # 12 PSUs
        fpc_val = 120.0  # Population per stratum
        wt = 1.0 + 0.3 * stratum

        for t in range(1, n_periods + 1):
            y = 10.0 + unit * 0.05 + t * 0.2
            if ft > 0 and t >= ft:
                y += 2.0  # Treatment effect
            y += np.random.normal(0, 0.5)

            rows.append(
                {
                    "unit": unit,
                    "time": t,
                    "first_treat": ft,
                    "outcome": y,
                    "weight": wt,
                    "stratum": stratum,
                    "psu": psu,
                    "fpc": fpc_val,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def ddd_survey_data():
    """Cross-sectional DDD data with survey columns."""
    np.random.seed(42)
    n = 400
    data = pd.DataFrame(
        {
            "outcome": np.random.randn(n) + 0.5,
            "group": np.random.choice([0, 1], n),
            "partition": np.random.choice([0, 1], n),
            "time": np.random.choice([0, 1], n),
            "weight": np.random.uniform(0.5, 2.0, n),
            "stratum": np.random.choice([1, 2, 3], n),
        }
    )
    # Add treatment effect for treated+eligible+post
    mask = (data["group"] == 1) & (data["partition"] == 1) & (data["time"] == 1)
    data.loc[mask, "outcome"] += 1.5
    return data


@pytest.fixture
def continuous_survey_data():
    """Panel data for continuous DiD with survey columns."""
    np.random.seed(42)
    n_u, n_t = 80, 4
    units = np.repeat(range(n_u), n_t)
    times = np.tile(range(1, n_t + 1), n_u)
    ft = np.repeat(np.where(np.arange(n_u) < 40, 3, 0), n_t)
    dose_per_unit = np.where(np.arange(n_u) < 40, np.random.uniform(0.5, 2.0, n_u), 0.0)
    dose = np.repeat(dose_per_unit, n_t)
    y = np.random.randn(len(units)) + 0.5 * dose * (times >= ft) * (ft > 0)
    w = np.repeat(np.random.uniform(0.5, 2.0, n_u), n_t)  # constant within unit
    strata = np.repeat(np.where(np.arange(n_u) < 40, 1, 2), n_t)

    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "first_treat": ft,
            "dose": dose,
            "outcome": y,
            "weight": w,
            "stratum": strata,
        }
    )


# =============================================================================
# SunAbraham
# =============================================================================


class TestSunAbrahamSurvey:
    """Survey design support for SunAbraham."""

    def test_smoke_weights_only(self, staggered_survey_data):
        """SunAbraham runs with weights-only survey design."""
        from diff_diff import SunAbraham

        sd = SurveyDesign(weights="weight")
        est = SunAbraham()
        result = est.fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None

    def test_uniform_weights_match_unweighted(self, staggered_survey_data):
        """Uniform survey weights should match unweighted result."""
        from diff_diff import SunAbraham

        staggered_survey_data["uniform_w"] = 1.0
        sd = SurveyDesign(weights="uniform_w")

        r_unw = SunAbraham().fit(staggered_survey_data, "outcome", "unit", "time", "first_treat")
        r_w = SunAbraham().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert abs(r_unw.overall_att - r_w.overall_att) < 1e-10

    def test_survey_metadata_fields(self, staggered_survey_data):
        """survey_metadata has correct fields with full design."""
        from diff_diff import SunAbraham

        sd = SurveyDesign(weights="weight", strata="stratum", psu="psu", fpc="fpc", nest=True)
        result = SunAbraham().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        sm = result.survey_metadata
        assert sm is not None
        assert sm.weight_type == "pweight"
        assert sm.effective_n > 0
        assert sm.design_effect > 0
        assert sm.n_strata is not None
        assert sm.n_psu is not None

    def test_se_differs_with_design(self, staggered_survey_data):
        """SEs should differ between weights-only and full design."""
        from diff_diff import SunAbraham

        sd_w = SurveyDesign(weights="weight")
        sd_full = SurveyDesign(weights="weight", strata="stratum", psu="psu", fpc="fpc", nest=True)
        r_w = SunAbraham().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd_w,
        )
        r_full = SunAbraham().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd_full,
        )
        # ATTs should be the same (same weights)
        assert abs(r_w.overall_att - r_full.overall_att) < 1e-10
        # SEs should differ due to different variance estimators
        assert r_w.overall_se != r_full.overall_se

    def test_bootstrap_survey_raises(self, staggered_survey_data):
        """Bootstrap + survey should raise NotImplementedError."""
        from diff_diff import SunAbraham

        sd = SurveyDesign(weights="weight")
        with pytest.raises(NotImplementedError, match="Bootstrap"):
            SunAbraham(n_bootstrap=99).fit(
                staggered_survey_data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                survey_design=sd,
            )

    def test_summary_includes_survey(self, staggered_survey_data):
        """Summary output should include survey design section."""
        from diff_diff import SunAbraham

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = SunAbraham().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        summary = result.summary()
        assert "Survey Design" in summary
        assert "pweight" in summary

    def test_no_survey_metadata_is_none(self, staggered_survey_data):
        """Without survey, survey_metadata should be None."""
        from diff_diff import SunAbraham

        result = SunAbraham().fit(staggered_survey_data, "outcome", "unit", "time", "first_treat")
        assert result.survey_metadata is None


# =============================================================================
# StackedDiD
# =============================================================================


class TestStackedDiDSurvey:
    """Survey design support for StackedDiD."""

    def test_smoke_weights_only(self, staggered_survey_data):
        """StackedDiD runs with weights-only survey design."""
        from diff_diff import StackedDiD

        sd = SurveyDesign(weights="weight")
        result = StackedDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None

    def test_survey_metadata_present(self, staggered_survey_data):
        """survey_metadata populated with full design."""
        from diff_diff import StackedDiD

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = StackedDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert result.survey_metadata is not None
        assert result.survey_metadata.weight_type == "pweight"

    def test_q_weight_composition(self, staggered_survey_data):
        """Survey weights should change results vs unweighted."""
        from diff_diff import StackedDiD

        r_unw = StackedDiD().fit(staggered_survey_data, "outcome", "unit", "time", "first_treat")
        sd = SurveyDesign(weights="weight")
        r_w = StackedDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        # ATT should differ (non-uniform weights)
        assert r_unw.overall_att != r_w.overall_att

    def test_convenience_function(self, staggered_survey_data):
        """stacked_did() convenience function threads survey_design."""
        from diff_diff.stacked_did import stacked_did

        sd = SurveyDesign(weights="weight")
        result = stacked_did(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert result.survey_metadata is not None

    def test_summary_includes_survey(self, staggered_survey_data):
        """Summary includes survey design section."""
        from diff_diff import StackedDiD

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = StackedDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert "Survey Design" in result.summary()


# =============================================================================
# BaconDecomposition
# =============================================================================


class TestBaconDecompositionSurvey:
    """Survey design support for BaconDecomposition."""

    def test_smoke_weights_only(self, staggered_survey_data):
        """BaconDecomposition runs with weights-only survey design."""
        from diff_diff import BaconDecomposition

        sd = SurveyDesign(weights="weight")
        result = BaconDecomposition().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.twfe_estimate)
        assert len(result.comparisons) > 0
        assert result.survey_metadata is not None

    def test_weighted_changes_twfe(self, staggered_survey_data):
        """Survey weights should change TWFE estimate."""
        from diff_diff import BaconDecomposition

        r_unw = BaconDecomposition().fit(
            staggered_survey_data, "outcome", "unit", "time", "first_treat"
        )
        sd = SurveyDesign(weights="weight")
        r_w = BaconDecomposition().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert r_unw.twfe_estimate != r_w.twfe_estimate

    def test_summary_includes_survey(self, staggered_survey_data):
        """Summary includes survey design section."""
        from diff_diff import BaconDecomposition

        sd = SurveyDesign(weights="weight")
        result = BaconDecomposition().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert "Survey Design" in result.summary()


# =============================================================================
# TripleDifference
# =============================================================================


class TestTripleDifferenceSurvey:
    """Survey design support for TripleDifference (reg method only)."""

    def test_smoke_reg_method(self, ddd_survey_data):
        """TripleDifference reg method runs with survey design."""
        from diff_diff import TripleDifference

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = TripleDifference(estimation_method="reg").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd,
        )
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.survey_metadata is not None

    def test_ipw_survey_raises(self, ddd_survey_data):
        """IPW + survey should raise NotImplementedError."""
        from diff_diff import TripleDifference

        sd = SurveyDesign(weights="weight")
        with pytest.raises(NotImplementedError, match="IPW"):
            TripleDifference(estimation_method="ipw").fit(
                ddd_survey_data,
                "outcome",
                "group",
                "partition",
                "time",
                survey_design=sd,
            )

    def test_dr_survey_raises(self, ddd_survey_data):
        """DR + survey should raise NotImplementedError."""
        from diff_diff import TripleDifference

        sd = SurveyDesign(weights="weight")
        with pytest.raises(NotImplementedError, match="doubly robust"):
            TripleDifference(estimation_method="dr").fit(
                ddd_survey_data,
                "outcome",
                "group",
                "partition",
                "time",
                survey_design=sd,
            )

    def test_weighted_changes_att(self, ddd_survey_data):
        """Survey weights should change ATT."""
        from diff_diff import TripleDifference

        r_unw = TripleDifference(estimation_method="reg").fit(
            ddd_survey_data, "outcome", "group", "partition", "time"
        )
        sd = SurveyDesign(weights="weight")
        r_w = TripleDifference(estimation_method="reg").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd,
        )
        assert r_unw.att != r_w.att

    def test_survey_metadata_in_to_dict(self, ddd_survey_data):
        """to_dict() includes survey metadata fields."""
        from diff_diff import TripleDifference

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = TripleDifference(estimation_method="reg").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd,
        )
        d = result.to_dict()
        assert "weight_type" in d
        assert "effective_n" in d
        assert "design_effect" in d

    def test_summary_includes_survey(self, ddd_survey_data):
        """Summary includes survey design section."""
        from diff_diff import TripleDifference

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = TripleDifference(estimation_method="reg").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd,
        )
        assert "Survey Design" in result.summary()


# =============================================================================
# ContinuousDiD
# =============================================================================


class TestContinuousDiDSurvey:
    """Survey design support for ContinuousDiD."""

    def test_smoke_weights_only(self, continuous_survey_data):
        """ContinuousDiD runs with survey design (analytical SEs)."""
        from diff_diff import ContinuousDiD

        sd = SurveyDesign(weights="weight")
        result = ContinuousDiD(n_bootstrap=0).fit(
            continuous_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            "dose",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert result.survey_metadata is not None

    def test_bootstrap_survey_raises(self, continuous_survey_data):
        """Bootstrap + survey should raise NotImplementedError."""
        from diff_diff import ContinuousDiD

        sd = SurveyDesign(weights="weight")
        with pytest.raises(NotImplementedError, match="bootstrap"):
            ContinuousDiD(n_bootstrap=99).fit(
                continuous_survey_data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                "dose",
                survey_design=sd,
            )

    def test_summary_includes_survey(self, continuous_survey_data):
        """Summary includes survey design section."""
        from diff_diff import ContinuousDiD

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = ContinuousDiD(n_bootstrap=0).fit(
            continuous_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            "dose",
            survey_design=sd,
        )
        assert "Survey Design" in result.summary()


# =============================================================================
# EfficientDiD
# =============================================================================


class TestEfficientDiDSurvey:
    """Survey design support for EfficientDiD."""

    def test_smoke_weights_only(self, staggered_survey_data):
        """EfficientDiD runs with weights-only survey design."""
        from diff_diff import EfficientDiD

        sd = SurveyDesign(weights="weight")
        result = EfficientDiD(n_bootstrap=0).fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None

    def test_bootstrap_survey_raises(self, staggered_survey_data):
        """Bootstrap + survey should raise NotImplementedError."""
        from diff_diff import EfficientDiD

        sd = SurveyDesign(weights="weight")
        with pytest.raises(NotImplementedError, match="bootstrap"):
            EfficientDiD(n_bootstrap=99).fit(
                staggered_survey_data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                survey_design=sd,
            )

    def test_survey_metadata_fields(self, staggered_survey_data):
        """survey_metadata has correct fields."""
        from diff_diff import EfficientDiD

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = EfficientDiD(n_bootstrap=0).fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        sm = result.survey_metadata
        assert sm is not None
        assert sm.weight_type == "pweight"
        assert sm.effective_n > 0

    def test_summary_includes_survey(self, staggered_survey_data):
        """Summary includes survey design section."""
        from diff_diff import EfficientDiD

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = EfficientDiD(n_bootstrap=0).fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert "Survey Design" in result.summary()

    def test_no_survey_metadata_is_none(self, staggered_survey_data):
        """Without survey, survey_metadata should be None."""
        from diff_diff import EfficientDiD

        result = EfficientDiD(n_bootstrap=0).fit(
            staggered_survey_data, "outcome", "unit", "time", "first_treat"
        )
        assert result.survey_metadata is None


# =============================================================================
# Scale Invariance (applies to all estimators)
# =============================================================================


class TestScaleInvariance:
    """Multiplying all survey weights by a constant should not change ATT or SE."""

    def test_sun_abraham_scale_invariance(self, staggered_survey_data):
        from diff_diff import SunAbraham

        sd1 = SurveyDesign(weights="weight")
        r1 = SunAbraham().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd1,
        )

        staggered_survey_data["weight_x10"] = staggered_survey_data["weight"] * 10.0
        sd2 = SurveyDesign(weights="weight_x10")
        r2 = SunAbraham().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd2,
        )

        assert abs(r1.overall_att - r2.overall_att) < 1e-10
        assert abs(r1.overall_se - r2.overall_se) < 1e-8

    def test_efficient_did_scale_invariance(self, staggered_survey_data):
        from diff_diff import EfficientDiD

        sd1 = SurveyDesign(weights="weight")
        r1 = EfficientDiD(n_bootstrap=0).fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd1,
        )

        staggered_survey_data["weight_x10"] = staggered_survey_data["weight"] * 10.0
        sd2 = SurveyDesign(weights="weight_x10")
        r2 = EfficientDiD(n_bootstrap=0).fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd2,
        )

        assert abs(r1.overall_att - r2.overall_att) < 1e-10
        assert abs(r1.overall_se - r2.overall_se) < 1e-8


# =============================================================================
# Regression Tests (PR #226 review feedback)
# =============================================================================


class TestReviewRegressions:
    """Targeted tests for issues found in PR #226 review."""

    def test_stacked_did_no_weight_survey(self, staggered_survey_data):
        """StackedDiD should handle SurveyDesign without weights column."""
        from diff_diff import StackedDiD

        sd = SurveyDesign(strata="stratum")  # No weights
        result = StackedDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert result.survey_metadata is not None

    def test_triple_diff_survey_df(self, ddd_survey_data):
        """TripleDifference should use survey df for p-values when survey active."""
        from diff_diff import TripleDifference

        sd = SurveyDesign(weights="weight", strata="stratum")
        r_survey = TripleDifference(estimation_method="reg").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd,
        )
        r_nosurv = TripleDifference(estimation_method="reg").fit(
            ddd_survey_data, "outcome", "group", "partition", "time"
        )
        # P-values should differ (different df)
        if np.isfinite(r_survey.p_value) and np.isfinite(r_nosurv.p_value):
            assert r_survey.p_value != r_nosurv.p_value

    def test_efficient_did_weights_only_se(self, staggered_survey_data):
        """EfficientDiD weights-only survey SE should be reasonable (not tiny)."""
        from diff_diff import EfficientDiD

        sd = SurveyDesign(weights="weight")
        r = EfficientDiD(n_bootstrap=0).fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(r.overall_se)
        assert r.overall_se > 0
        assert r.overall_se > 0.01  # Not artificially tiny

    def test_continuous_did_eventstudy_survey(self, continuous_survey_data):
        """ContinuousDiD aggregate=eventstudy should work with survey design."""
        from diff_diff import ContinuousDiD

        sd = SurveyDesign(weights="weight", strata="stratum")
        result = ContinuousDiD(n_bootstrap=0).fit(
            continuous_survey_data,
            "outcome",
            "unit",
            "time",
            "first_treat",
            "dose",
            aggregate="eventstudy",
            survey_design=sd,
        )
        assert result.event_study_effects is not None
        assert result.survey_metadata is not None
        for e, eff in result.event_study_effects.items():
            if np.isfinite(eff["effect"]):
                assert np.isfinite(eff["se"]), f"Non-finite SE for e={e}"

    def test_within_unit_varying_weights_rejected(self):
        """Time-varying survey weights within units should be rejected."""
        from diff_diff import ContinuousDiD

        np.random.seed(42)
        n_u, n_t = 20, 4
        data = pd.DataFrame(
            {
                "unit": np.repeat(range(n_u), n_t),
                "time": np.tile(range(1, n_t + 1), n_u),
                "first_treat": np.repeat(np.where(np.arange(n_u) < 10, 3, 0), n_t),
                "dose": np.repeat(np.where(np.arange(n_u) < 10, 1.0, 0.0), n_t),
                "outcome": np.random.randn(n_u * n_t),
                "w": np.random.uniform(0.5, 2.0, n_u * n_t),
            }
        )
        sd = SurveyDesign(weights="w")
        with pytest.raises(ValueError, match="varies within units"):
            ContinuousDiD(n_bootstrap=0).fit(
                data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                "dose",
                survey_design=sd,
            )

    def test_within_unit_varying_strata_rejected(self):
        """Time-varying strata within units should be rejected."""
        from diff_diff import EfficientDiD

        np.random.seed(42)
        n_u, n_t = 20, 4
        data = pd.DataFrame(
            {
                "unit": np.repeat(range(n_u), n_t),
                "time": np.tile(range(1, n_t + 1), n_u),
                "first_treat": np.repeat(np.where(np.arange(n_u) < 10, 3, 0), n_t),
                "outcome": np.random.randn(n_u * n_t),
                "w": np.repeat(np.ones(n_u), n_t),
                "strat": np.tile([1, 2, 1, 2], n_u),
            }
        )
        sd = SurveyDesign(weights="w", strata="strat")
        with pytest.raises(ValueError, match="varies within units"):
            EfficientDiD(n_bootstrap=0).fit(
                data,
                "outcome",
                "unit",
                "time",
                "first_treat",
                survey_design=sd,
            )
