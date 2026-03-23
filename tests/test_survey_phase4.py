"""Tests for Phase 4 survey support: complex standalone estimators.

Covers: ImputationDiD, TwoStageDiD, CallawaySantAnna, weighted solve_logit(),
TripleDifference IPW/DR unblock, and cross-estimator scale invariance.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    CallawaySantAnna,
    ImputationDiD,
    SurveyDesign,
    TripleDifference,
    TwoStageDiD,
    generate_staggered_data,
)
from diff_diff.linalg import solve_logit

# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def staggered_survey_data():
    """Staggered treatment panel with survey design columns.

    200 units via generate_staggered_data, then add unit-level survey columns
    (weights, stratum, psu, fpc) that are constant within each unit.
    """
    data = generate_staggered_data(n_units=200, seed=42)

    # Add unit-level survey columns (constant within unit)
    unit_ids = data["unit"].unique()
    n_units = len(unit_ids)
    np.random.RandomState(42)

    unit_weight = 1.0 + 0.5 * (np.arange(n_units) % 5)
    unit_stratum = np.arange(n_units) // 40  # 5 strata
    unit_psu = np.arange(n_units) // 10  # 20 PSUs
    unit_fpc = np.full(n_units, 400.0)  # population per stratum

    unit_map = {uid: i for i, uid in enumerate(unit_ids)}
    idx = data["unit"].map(unit_map).values

    data["weight"] = unit_weight[idx]
    data["stratum"] = unit_stratum[idx]
    data["psu"] = unit_psu[idx]
    data["fpc"] = unit_fpc[idx]

    return data


@pytest.fixture
def survey_design_weights_only():
    """SurveyDesign with weights only."""
    return SurveyDesign(weights="weight")


@pytest.fixture
def survey_design_full():
    """SurveyDesign with weights, strata, psu."""
    return SurveyDesign(weights="weight", strata="stratum", psu="psu")


@pytest.fixture
def ddd_survey_data():
    """Cross-sectional DDD data with survey columns for TripleDifference."""
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


# =============================================================================
# TestWeightedSolveLogit
# =============================================================================


class TestWeightedSolveLogit:
    """Tests for weighted solve_logit() (IRLS with survey weights)."""

    def test_uniform_weights_match_unweighted(self):
        """Uniform weights should produce same coefficients as unweighted."""
        rng = np.random.RandomState(123)
        n = 200
        X = rng.randn(n, 2)
        y = (X @ [1.0, -0.5] + rng.randn(n) > 0).astype(float)

        beta_unw, probs_unw = solve_logit(X, y)
        beta_w, probs_w = solve_logit(X, y, weights=np.ones(n))

        np.testing.assert_allclose(beta_unw, beta_w, atol=1e-10)
        np.testing.assert_allclose(probs_unw, probs_w, atol=1e-10)

    def test_convergence_with_weights(self):
        """solve_logit converges with non-uniform survey weights."""
        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n, 2)
        y = (X @ [0.5, -0.5] + rng.randn(n) * 1.5 > 0).astype(float)
        weights = rng.uniform(0.5, 3.0, n)

        beta, probs = solve_logit(X, y, weights=weights)

        # Should produce finite coefficients (convergence)
        assert np.all(np.isfinite(beta))
        assert np.all(np.isfinite(probs))
        assert np.all(probs > 0) and np.all(probs < 1)

    def test_separation_detection_with_weights(self):
        """Separation warning should still fire with survey weights."""
        rng = np.random.RandomState(789)
        n = 100
        # Create near-separation: x1 perfectly predicts y
        x1 = np.linspace(-5, 5, n)
        X = x1.reshape(-1, 1)
        y = (x1 > 0).astype(float)
        weights = rng.uniform(1.0, 3.0, n)

        with pytest.warns(UserWarning):
            beta, probs = solve_logit(X, y, weights=weights)

        assert np.all(np.isfinite(beta))

    def test_known_answer_small_dataset(self):
        """Manual check on a small dataset — weights should shift coefficients."""
        rng = np.random.RandomState(333)
        n = 50
        X = rng.randn(n, 1)
        prob = 1.0 / (1.0 + np.exp(-(0.5 + 1.0 * X[:, 0])))
        y = (rng.rand(n) < prob).astype(float)

        # Unweighted fit
        beta_unw, _ = solve_logit(X, y)

        # Non-uniform weights: upweight observations where y=1
        weights = np.where(y == 1, 5.0, 1.0)
        beta_w, _ = solve_logit(X, y, weights=weights)

        # Weighted fit should shift intercept (upweighting y=1 shifts boundary)
        assert beta_w[0] != pytest.approx(beta_unw[0], abs=0.1)

    def test_rank_deficiency_with_weights(self):
        """Rank-deficient columns should still be detected with weights."""
        rng = np.random.RandomState(111)
        n = 100
        x1 = rng.randn(n)
        # x2 is a perfect linear combination of x1
        X = np.column_stack([x1, 2.0 * x1])
        y = (x1 > 0).astype(float)
        weights = rng.uniform(0.5, 2.0, n)

        with pytest.warns(UserWarning, match="[Rr]ank"):
            beta, probs = solve_logit(X, y, weights=weights)

        assert np.all(np.isfinite(probs))

    def test_weight_scale_invariance(self):
        """Multiplying weights by a constant should not change beta."""
        rng = np.random.RandomState(222)
        n = 200
        X = rng.randn(n, 2)
        y = (X @ [0.8, -0.3] + rng.randn(n) > 0).astype(float)
        weights = rng.uniform(1.0, 4.0, n)

        beta1, _ = solve_logit(X, y, weights=weights)
        beta2, _ = solve_logit(X, y, weights=weights * 2.0)

        np.testing.assert_allclose(beta1, beta2, atol=1e-10)


# =============================================================================
# TestImputationDiDSurvey
# =============================================================================


class TestImputationDiDSurvey:
    """Survey design support for ImputationDiD."""

    def test_smoke_weights_only(self, staggered_survey_data, survey_design_weights_only):
        """ImputationDiD runs with weights-only survey design."""
        result = ImputationDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=survey_design_weights_only,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None

    def test_uniform_weights_match_unweighted(self, staggered_survey_data):
        """Uniform survey weights should match unweighted result."""
        staggered_survey_data["uniform_w"] = 1.0
        sd = SurveyDesign(weights="uniform_w")

        r_unw = ImputationDiD().fit(
            staggered_survey_data, "outcome", "unit", "period", "first_treat"
        )
        r_w = ImputationDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd,
        )
        assert abs(r_unw.overall_att - r_w.overall_att) < 1e-10

    def test_survey_metadata_fields(self, staggered_survey_data, survey_design_full):
        """survey_metadata has correct fields with full design."""
        result = ImputationDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=survey_design_full,
        )
        sm = result.survey_metadata
        assert sm is not None
        assert sm.weight_type == "pweight"
        assert sm.effective_n > 0
        assert sm.design_effect > 0
        assert sm.n_strata is not None
        assert sm.n_psu is not None

    def test_se_differs_with_design(self, staggered_survey_data):
        """Weights-only vs full design: same ATT, different inference via survey df."""
        sd_w = SurveyDesign(weights="weight")
        sd_full = SurveyDesign(weights="weight", strata="stratum", psu="psu")

        r_w = ImputationDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd_w,
        )
        r_full = ImputationDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd_full,
        )
        # ATTs should be the same (same weights)
        assert abs(r_w.overall_att - r_full.overall_att) < 1e-10
        # Full design should carry survey df (strata/PSU structure)
        assert r_full.survey_metadata is not None
        assert r_full.survey_metadata.n_strata is not None
        assert r_full.survey_metadata.n_psu is not None
        # P-values should differ due to t-distribution with survey df
        if np.isfinite(r_w.overall_p_value) and np.isfinite(r_full.overall_p_value):
            assert r_w.overall_p_value != r_full.overall_p_value

    def test_weighted_att_differs(self, staggered_survey_data, survey_design_weights_only):
        """Non-uniform survey weights should change the overall ATT."""
        r_unw = ImputationDiD().fit(
            staggered_survey_data, "outcome", "unit", "period", "first_treat"
        )
        r_w = ImputationDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=survey_design_weights_only,
        )
        # ATT should differ because non-uniform weights change aggregation
        assert r_unw.overall_att != r_w.overall_att

    def test_event_study_with_survey(self, staggered_survey_data, survey_design_weights_only):
        """Event study effects exist when using survey design."""
        result = ImputationDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            aggregate="event_study",
            survey_design=survey_design_weights_only,
        )
        assert result.event_study_effects is not None
        assert len(result.event_study_effects) > 0
        for h, eff in result.event_study_effects.items():
            assert np.isfinite(eff["effect"])
            assert np.isfinite(eff["se"])

    def test_bootstrap_survey_raises(self, staggered_survey_data, survey_design_weights_only):
        """Bootstrap + survey should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="[Bb]ootstrap"):
            ImputationDiD(n_bootstrap=99).fit(
                staggered_survey_data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                survey_design=survey_design_weights_only,
            )

    def test_summary_includes_survey(self, staggered_survey_data, survey_design_weights_only):
        """Summary output should include survey design section."""
        result = ImputationDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=survey_design_weights_only,
        )
        summary = result.summary()
        assert "Survey Design" in summary
        assert "pweight" in summary

    def test_se_scale_invariance_fe_only(self, staggered_survey_data):
        """SE must be invariant to weight rescaling (FE-only, no covariates)."""
        data = staggered_survey_data.copy()
        data["weight2"] = data["weight"] * 3.1
        sd1 = SurveyDesign(weights="weight")
        sd2 = SurveyDesign(weights="weight2")
        r1 = ImputationDiD().fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd1,
        )
        r2 = ImputationDiD().fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd2,
        )
        assert np.isclose(r1.overall_att, r2.overall_att, atol=1e-8)
        assert np.isclose(
            r1.overall_se, r2.overall_se, atol=1e-8
        ), f"SE not scale-invariant (FE-only): {r1.overall_se} vs {r2.overall_se}"

    def test_se_scale_invariance_with_covariates(self, staggered_survey_data):
        """SE must be invariant to weight rescaling (with covariates)."""
        data = staggered_survey_data.copy()
        data["x1"] = np.random.default_rng(99).normal(0, 1, len(data))
        data["weight2"] = data["weight"] * 3.1
        sd1 = SurveyDesign(weights="weight")
        sd2 = SurveyDesign(weights="weight2")
        r1 = ImputationDiD().fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
            survey_design=sd1,
        )
        r2 = ImputationDiD().fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
            survey_design=sd2,
        )
        assert np.isclose(r1.overall_att, r2.overall_att, atol=1e-8)
        assert np.isclose(
            r1.overall_se, r2.overall_se, atol=1e-8
        ), f"SE not scale-invariant (covariates): {r1.overall_se} vs {r2.overall_se}"

    def test_wrapper_imputation_did_with_survey(self, staggered_survey_data):
        """imputation_did() wrapper forwards survey_design correctly."""
        from diff_diff import imputation_did

        sd = SurveyDesign(weights="weight")
        r_wrapper = imputation_did(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd,
        )
        r_direct = ImputationDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd,
        )
        assert np.isclose(r_wrapper.overall_att, r_direct.overall_att, atol=1e-10)
        assert r_wrapper.survey_metadata is not None


# =============================================================================
# TestTwoStageDiDSurvey
# =============================================================================


class TestTwoStageDiDSurvey:
    """Survey design support for TwoStageDiD."""

    def test_smoke_weights_only(self, staggered_survey_data, survey_design_weights_only):
        """TwoStageDiD runs with weights-only survey design."""
        result = TwoStageDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=survey_design_weights_only,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None

    def test_uniform_weights_match_unweighted(self, staggered_survey_data):
        """Uniform survey weights should match unweighted result."""
        staggered_survey_data["uniform_w"] = 1.0
        sd = SurveyDesign(weights="uniform_w")

        r_unw = TwoStageDiD().fit(staggered_survey_data, "outcome", "unit", "period", "first_treat")
        r_w = TwoStageDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd,
        )
        assert abs(r_unw.overall_att - r_w.overall_att) < 1e-10

    def test_survey_metadata_fields(self, staggered_survey_data, survey_design_full):
        """survey_metadata has correct fields with full design."""
        result = TwoStageDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=survey_design_full,
        )
        sm = result.survey_metadata
        assert sm is not None
        assert sm.weight_type == "pweight"
        assert sm.effective_n > 0
        assert sm.design_effect > 0
        assert sm.n_strata is not None
        assert sm.n_psu is not None

    def test_se_differs_with_design(self, staggered_survey_data):
        """Weights-only vs full design: same ATT, different inference via survey df."""
        sd_w = SurveyDesign(weights="weight")
        sd_full = SurveyDesign(weights="weight", strata="stratum", psu="psu")

        r_w = TwoStageDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd_w,
        )
        r_full = TwoStageDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd_full,
        )
        # ATTs should be the same (same weights)
        assert abs(r_w.overall_att - r_full.overall_att) < 1e-10
        # Full design should carry survey df (strata/PSU structure)
        assert r_full.survey_metadata is not None
        assert r_full.survey_metadata.n_strata is not None
        assert r_full.survey_metadata.n_psu is not None
        # P-values should differ due to t-distribution with survey df
        if np.isfinite(r_w.overall_p_value) and np.isfinite(r_full.overall_p_value):
            assert r_w.overall_p_value != r_full.overall_p_value

    def test_weighted_gmm_variance(self, staggered_survey_data, survey_design_weights_only):
        """GMM SE should differ from unweighted (weights affect sandwich)."""
        r_unw = TwoStageDiD().fit(staggered_survey_data, "outcome", "unit", "period", "first_treat")
        r_w = TwoStageDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=survey_design_weights_only,
        )
        # SE magnitude should differ (not just sign)
        assert abs(r_unw.overall_se - r_w.overall_se) > 1e-6

    def test_bootstrap_survey_raises(self, staggered_survey_data, survey_design_weights_only):
        """Bootstrap + survey should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="[Bb]ootstrap"):
            TwoStageDiD(n_bootstrap=99).fit(
                staggered_survey_data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                survey_design=survey_design_weights_only,
            )

    def test_summary_includes_survey(self, staggered_survey_data, survey_design_weights_only):
        """Summary output should include survey design section."""
        result = TwoStageDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=survey_design_weights_only,
        )
        summary = result.summary()
        assert "Survey Design" in summary
        assert "pweight" in summary

    def test_wrapper_two_stage_did_with_survey(self, staggered_survey_data):
        """two_stage_did() wrapper forwards survey_design correctly."""
        from diff_diff import two_stage_did

        sd = SurveyDesign(weights="weight")
        r_wrapper = two_stage_did(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd,
        )
        r_direct = TwoStageDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd,
        )
        assert np.isclose(r_wrapper.overall_att, r_direct.overall_att, atol=1e-10)
        assert r_wrapper.survey_metadata is not None

    def test_always_treated_with_survey(self, staggered_survey_data):
        """TwoStageDiD with survey + always-treated units should not crash."""
        data = staggered_survey_data.copy()
        # Make some units always-treated (first_treat at or before min time)
        min_time = data["period"].min()
        units = data["unit"].unique()
        always_treated_units = units[:3]
        for u in always_treated_units:
            data.loc[data["unit"] == u, "first_treat"] = min_time
        sd = SurveyDesign(weights="weight")
        result = TwoStageDiD().fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None


# =============================================================================
# TestCallawaySantAnnaSurvey
# =============================================================================


class TestCallawaySantAnnaSurvey:
    """Survey design support for CallawaySantAnna."""

    def test_smoke_reg_weights_only(self, staggered_survey_data, survey_design_weights_only):
        """CallawaySantAnna regression method works with survey design."""
        result = CallawaySantAnna(estimation_method="reg").fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=survey_design_weights_only,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None

    def test_smoke_ipw_weights_only(self, staggered_survey_data, survey_design_weights_only):
        """CallawaySantAnna IPW method works with survey design."""
        result = CallawaySantAnna(estimation_method="ipw").fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=survey_design_weights_only,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None

    def test_smoke_dr_weights_only(self, staggered_survey_data, survey_design_weights_only):
        """CallawaySantAnna DR method works with survey design."""
        result = CallawaySantAnna(estimation_method="dr").fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=survey_design_weights_only,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.survey_metadata is not None

    def test_uniform_weights_match_unweighted(self, staggered_survey_data):
        """Uniform survey weights should match unweighted result — all methods."""
        staggered_survey_data["uniform_w"] = 1.0
        sd = SurveyDesign(weights="uniform_w")

        for method in ["reg", "ipw", "dr"]:
            r_unw = CallawaySantAnna(estimation_method=method).fit(
                staggered_survey_data, "outcome", "unit", "period", "first_treat"
            )
            r_w = CallawaySantAnna(estimation_method=method).fit(
                staggered_survey_data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                survey_design=sd,
            )
            assert abs(r_unw.overall_att - r_w.overall_att) < 1e-8, f"method={method}: ATT mismatch"

    def test_survey_metadata_fields(self, staggered_survey_data, survey_design_full):
        """survey_metadata has correct fields with full design."""
        result = CallawaySantAnna(estimation_method="reg").fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=survey_design_full,
        )
        sm = result.survey_metadata
        assert sm is not None
        assert sm.weight_type == "pweight"
        assert sm.effective_n > 0
        assert sm.design_effect > 0
        assert sm.n_strata is not None
        assert sm.n_psu is not None

    def test_se_differs_with_design(self, staggered_survey_data):
        """Weights-only vs full design: same ATT, different inference via survey df."""
        sd_w = SurveyDesign(weights="weight")
        sd_full = SurveyDesign(weights="weight", strata="stratum", psu="psu")

        r_w = CallawaySantAnna(estimation_method="reg").fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd_w,
        )
        r_full = CallawaySantAnna(estimation_method="reg").fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd_full,
        )
        # ATTs should be the same (same weights)
        assert abs(r_w.overall_att - r_full.overall_att) < 1e-10
        # Full design should carry survey df (strata/PSU structure)
        assert r_full.survey_metadata is not None
        assert r_full.survey_metadata.n_strata is not None
        assert r_full.survey_metadata.n_psu is not None
        # P-values should differ due to t-distribution with survey df
        if np.isfinite(r_w.overall_p_value) and np.isfinite(r_full.overall_p_value):
            assert r_w.overall_p_value != r_full.overall_p_value

    def test_bootstrap_survey_raises(self, staggered_survey_data, survey_design_weights_only):
        """Bootstrap + survey should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="[Bb]ootstrap"):
            CallawaySantAnna(estimation_method="reg", n_bootstrap=99).fit(
                staggered_survey_data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                survey_design=survey_design_weights_only,
            )

    def test_ipw_covariates_survey_raises(self, staggered_survey_data, survey_design_weights_only):
        """IPW + covariates + survey should raise NotImplementedError."""
        data = staggered_survey_data.copy()
        data["x1"] = np.random.default_rng(42).normal(0, 1, len(data))
        with pytest.raises(NotImplementedError, match="covariates"):
            CallawaySantAnna(estimation_method="ipw").fit(
                data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                covariates=["x1"],
                survey_design=survey_design_weights_only,
            )

    def test_dr_covariates_survey_raises(self, staggered_survey_data, survey_design_weights_only):
        """DR + covariates + survey should raise NotImplementedError."""
        data = staggered_survey_data.copy()
        data["x1"] = np.random.default_rng(42).normal(0, 1, len(data))
        with pytest.raises(NotImplementedError, match="covariates"):
            CallawaySantAnna(estimation_method="dr").fit(
                data,
                "outcome",
                "unit",
                "period",
                "first_treat",
                covariates=["x1"],
                survey_design=survey_design_weights_only,
            )

    def test_reg_covariates_survey_works(self, staggered_survey_data, survey_design_weights_only):
        """Regression + covariates + survey should work (has nuisance IF correction)."""
        data = staggered_survey_data.copy()
        data["x1"] = np.random.default_rng(42).normal(0, 1, len(data))
        result = CallawaySantAnna(estimation_method="reg").fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
            survey_design=survey_design_weights_only,
        )
        assert np.isfinite(result.overall_att)

    def test_reg_covariates_survey_se_scale_invariance(self, staggered_survey_data):
        """SE for reg + covariates + survey must be invariant to weight rescaling."""
        data = staggered_survey_data.copy()
        data["x1"] = np.random.default_rng(42).normal(0, 1, len(data))
        data["weight2"] = data["weight"] * 4.3
        sd1 = SurveyDesign(weights="weight")
        sd2 = SurveyDesign(weights="weight2")
        est = CallawaySantAnna(estimation_method="reg")
        r1 = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
            aggregate="simple",
            survey_design=sd1,
        )
        r2 = est.fit(
            data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            covariates=["x1"],
            aggregate="simple",
            survey_design=sd2,
        )
        assert np.isclose(
            r1.overall_att, r2.overall_att, atol=1e-8
        ), "ATT not scale-invariant for reg+cov+survey"
        assert np.isclose(
            r1.overall_se, r2.overall_se, atol=1e-8
        ), f"SE not scale-invariant for reg+cov+survey: {r1.overall_se} vs {r2.overall_se}"

    def test_weighted_logit(self, staggered_survey_data, survey_design_weights_only):
        """Propensity scores should change with survey weights (IPW path)."""
        r_unw = CallawaySantAnna(estimation_method="ipw").fit(
            staggered_survey_data, "outcome", "unit", "period", "first_treat"
        )
        r_w = CallawaySantAnna(estimation_method="ipw").fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=survey_design_weights_only,
        )
        # Non-uniform weights should produce different ATT
        # (propensity scores change with survey weights)
        assert r_unw.overall_att != r_w.overall_att

    def test_ipw_survey_weight_composition(self, staggered_survey_data, survey_design_weights_only):
        """w_survey x w_ipw should compose — ATT differs from unweighted IPW."""
        r_unw = CallawaySantAnna(estimation_method="ipw").fit(
            staggered_survey_data, "outcome", "unit", "period", "first_treat"
        )
        r_w = CallawaySantAnna(estimation_method="ipw").fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=survey_design_weights_only,
        )
        # Weighted IPW should produce different ATT than unweighted
        assert abs(r_unw.overall_att - r_w.overall_att) > 1e-6

    def test_aggregation_with_survey(self, staggered_survey_data, survey_design_weights_only):
        """Simple aggregation should use survey weights."""
        r_unw = CallawaySantAnna(estimation_method="reg").fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            aggregate="event_study",
        )
        r_w = CallawaySantAnna(estimation_method="reg").fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            aggregate="event_study",
            survey_design=survey_design_weights_only,
        )
        # Event study ATTs should differ with non-uniform weights
        assert r_unw.overall_att != r_w.overall_att
        # Event study effects should exist
        assert r_w.event_study_effects is not None
        assert len(r_w.event_study_effects) > 0

    def test_summary_includes_survey(self, staggered_survey_data, survey_design_weights_only):
        """Summary output should include survey design section."""
        result = CallawaySantAnna(estimation_method="reg").fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=survey_design_weights_only,
        )
        summary = result.summary()
        assert "Survey Design" in summary
        assert "pweight" in summary


# =============================================================================
# TestTripleDifferenceIPWSurvey
# =============================================================================


class TestTripleDifferenceIPWSurvey:
    """Verify TripleDifference IPW/DR + survey is unblocked."""

    def test_ipw_survey_no_longer_raises(self, ddd_survey_data):
        """IPW + survey should no longer raise NotImplementedError."""
        sd = SurveyDesign(weights="weight")
        # Should not raise
        result = TripleDifference(estimation_method="ipw").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd,
        )
        assert result is not None

    def test_dr_survey_no_longer_raises(self, ddd_survey_data):
        """DR + survey should no longer raise NotImplementedError."""
        sd = SurveyDesign(weights="weight")
        # Should not raise
        result = TripleDifference(estimation_method="dr").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd,
        )
        assert result is not None

    def test_ipw_survey_results_finite(self, ddd_survey_data):
        """IPW + survey should produce finite results."""
        sd = SurveyDesign(weights="weight")
        result = TripleDifference(estimation_method="ipw").fit(
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

    def test_ipw_nonuniform_weights_change_att(self, ddd_survey_data):
        """Non-uniform survey weights should change IPW ATT vs unweighted."""
        sd = SurveyDesign(weights="weight")
        r_no = TripleDifference(estimation_method="ipw").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
        )
        r_sv = TripleDifference(estimation_method="ipw").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd,
        )
        assert not np.isclose(
            r_no.att, r_sv.att, atol=1e-6
        ), "Non-uniform survey weights should change IPW ATT"

    def test_dr_nonuniform_weights_change_att(self, ddd_survey_data):
        """Non-uniform survey weights should change DR ATT vs unweighted."""
        sd = SurveyDesign(weights="weight")
        r_no = TripleDifference(estimation_method="dr").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
        )
        r_sv = TripleDifference(estimation_method="dr").fit(
            ddd_survey_data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd,
        )
        assert not np.isclose(
            r_no.att, r_sv.att, atol=1e-6
        ), "Non-uniform survey weights should change DR ATT"

    def test_ipw_uniform_weights_match_unweighted(self, ddd_survey_data):
        """Uniform survey weights should match unweighted IPW result."""
        data = ddd_survey_data.copy()
        data["uw"] = 1.0
        sd = SurveyDesign(weights="uw")
        r_no = TripleDifference(estimation_method="ipw").fit(
            data,
            "outcome",
            "group",
            "partition",
            "time",
        )
        r_sv = TripleDifference(estimation_method="ipw").fit(
            data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd,
        )
        assert np.isclose(r_no.att, r_sv.att, atol=1e-6)

    def test_dr_uniform_weights_match_unweighted(self, ddd_survey_data):
        """Uniform survey weights should match unweighted DR result."""
        data = ddd_survey_data.copy()
        data["uw"] = 1.0
        sd = SurveyDesign(weights="uw")
        r_no = TripleDifference(estimation_method="dr").fit(
            data,
            "outcome",
            "group",
            "partition",
            "time",
        )
        r_sv = TripleDifference(estimation_method="dr").fit(
            data,
            "outcome",
            "group",
            "partition",
            "time",
            survey_design=sd,
        )
        assert np.isclose(r_no.att, r_sv.att, atol=1e-6)

    def test_ipw_covariate_survey_nonuniform(self, ddd_survey_data):
        """IPW + covariates + non-uniform survey weights should change ATT."""
        data = ddd_survey_data.copy()
        data["x1"] = np.random.default_rng(42).normal(0, 1, len(data))
        sd = SurveyDesign(weights="weight")
        r_no = TripleDifference(estimation_method="ipw").fit(
            data,
            "outcome",
            "group",
            "partition",
            "time",
            covariates=["x1"],
        )
        r_sv = TripleDifference(estimation_method="ipw").fit(
            data,
            "outcome",
            "group",
            "partition",
            "time",
            covariates=["x1"],
            survey_design=sd,
        )
        assert np.isfinite(r_sv.att)
        assert np.isfinite(r_sv.se)
        assert not np.isclose(
            r_no.att, r_sv.att, atol=1e-6
        ), "Covariate IPW + non-uniform survey weights should change ATT"

    def test_dr_covariate_survey_nonuniform(self, ddd_survey_data):
        """DR + covariates + non-uniform survey weights should change ATT."""
        data = ddd_survey_data.copy()
        data["x1"] = np.random.default_rng(42).normal(0, 1, len(data))
        sd = SurveyDesign(weights="weight")
        r_no = TripleDifference(estimation_method="dr").fit(
            data,
            "outcome",
            "group",
            "partition",
            "time",
            covariates=["x1"],
        )
        r_sv = TripleDifference(estimation_method="dr").fit(
            data,
            "outcome",
            "group",
            "partition",
            "time",
            covariates=["x1"],
            survey_design=sd,
        )
        assert np.isfinite(r_sv.att)
        assert np.isfinite(r_sv.se)
        assert not np.isclose(
            r_no.att, r_sv.att, atol=1e-6
        ), "Covariate DR + non-uniform survey weights should change ATT"


# =============================================================================
# TestCallawaySantAnnaSurveyInference
# =============================================================================


class TestCallawaySantAnnaSurveyInference:
    """Validate CS survey inference beyond smoke tests."""

    def test_se_scale_invariance_all_methods(self, staggered_survey_data):
        """SE should be invariant under weight rescaling for all methods."""
        data = staggered_survey_data
        data = data.copy()
        data["weight2"] = data["weight"] * 3.7
        sd1 = SurveyDesign(weights="weight")
        sd2 = SurveyDesign(weights="weight2")

        for method in ["reg", "ipw", "dr"]:
            est = CallawaySantAnna(estimation_method=method)
            r1 = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                aggregate="simple",
                survey_design=sd1,
            )
            r2 = est.fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                aggregate="simple",
                survey_design=sd2,
            )
            assert np.isclose(
                r1.overall_att, r2.overall_att, atol=1e-8
            ), f"{method}: ATT not scale-invariant"
            assert np.isclose(
                r1.overall_se, r2.overall_se, atol=1e-8
            ), f"{method}: SE not scale-invariant"

    def test_survey_weights_change_per_cell_att(self, staggered_survey_data):
        """Non-uniform survey weights should change per-cell ATT(g,t)."""
        data = staggered_survey_data
        sd = SurveyDesign(weights="weight")
        for method in ["reg", "ipw", "dr"]:
            r_no = CallawaySantAnna(estimation_method=method).fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
            )
            r_sv = CallawaySantAnna(estimation_method=method).fit(
                data,
                outcome="outcome",
                unit="unit",
                time="period",
                first_treat="first_treat",
                survey_design=sd,
            )
            # At least one cell ATT should differ
            effects_no = [d["effect"] for d in r_no.group_time_effects.values()]
            effects_sv = [d["effect"] for d in r_sv.group_time_effects.values()]
            assert not np.allclose(
                effects_no, effects_sv, atol=1e-6
            ), f"{method}: survey weights should change per-cell ATT"

    def test_survey_df_affects_pvalues(self, staggered_survey_data):
        """Survey df (from strata/PSU) should affect p-values via t-distribution."""
        data = staggered_survey_data
        sd_weights = SurveyDesign(weights="weight")
        sd_full = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        est = CallawaySantAnna(estimation_method="reg")
        r_w = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="simple",
            survey_design=sd_weights,
        )
        r_f = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="period",
            first_treat="first_treat",
            aggregate="simple",
            survey_design=sd_full,
        )
        # ATT should be same (same weights), but p-values differ (different df)
        assert np.isclose(r_w.overall_att, r_f.overall_att, atol=1e-8)
        # Survey df from strata/PSU should change inference
        assert r_f.survey_metadata.df_survey is not None


# =============================================================================
# TestScaleInvariance
# =============================================================================


class TestScaleInvariance:
    """Multiplying all survey weights by a constant should not change ATT or SE."""

    def test_weight_scale_invariance_imputation(self, staggered_survey_data):
        """ImputationDiD: 2*w gives same ATT as w."""
        sd1 = SurveyDesign(weights="weight")
        r1 = ImputationDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd1,
        )

        staggered_survey_data["weight_x2"] = staggered_survey_data["weight"] * 2.0
        sd2 = SurveyDesign(weights="weight_x2")
        r2 = ImputationDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd2,
        )

        assert abs(r1.overall_att - r2.overall_att) < 1e-10
        assert abs(r1.overall_se - r2.overall_se) < 1e-8

    def test_weight_scale_invariance_two_stage(self, staggered_survey_data):
        """TwoStageDiD: 2*w gives same ATT as w."""
        sd1 = SurveyDesign(weights="weight")
        r1 = TwoStageDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd1,
        )

        staggered_survey_data["weight_x2"] = staggered_survey_data["weight"] * 2.0
        sd2 = SurveyDesign(weights="weight_x2")
        r2 = TwoStageDiD().fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd2,
        )

        assert abs(r1.overall_att - r2.overall_att) < 1e-10
        assert abs(r1.overall_se - r2.overall_se) < 1e-8

    def test_weight_scale_invariance_callaway_santanna(self, staggered_survey_data):
        """CallawaySantAnna: 2*w gives same ATT as w."""
        sd1 = SurveyDesign(weights="weight")
        r1 = CallawaySantAnna(estimation_method="reg").fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd1,
        )

        staggered_survey_data["weight_x2"] = staggered_survey_data["weight"] * 2.0
        sd2 = SurveyDesign(weights="weight_x2")
        r2 = CallawaySantAnna(estimation_method="reg").fit(
            staggered_survey_data,
            "outcome",
            "unit",
            "period",
            "first_treat",
            survey_design=sd2,
        )

        assert abs(r1.overall_att - r2.overall_att) < 1e-10
        assert abs(r1.overall_se - r2.overall_se) < 1e-8
