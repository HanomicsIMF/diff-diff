"""
Tests for pretrends=True feature in ImputationDiD and TwoStageDiD.

Tests that pre-period event study coefficients are computed correctly
when pretrends=True is set on the estimator.
"""

import numpy as np
import pandas as pd
import pytest

from diff_diff.imputation import ImputationDiD
from diff_diff.two_stage import TwoStageDiD


def generate_test_data(
    n_units: int = 100,
    n_periods: int = 10,
    treatment_effect: float = 2.0,
    never_treated_frac: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate staggered adoption data for pretrends testing."""
    rng = np.random.default_rng(seed)

    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(n_periods), n_units)

    n_never = int(n_units * never_treated_frac)
    n_treated = n_units - n_never

    cohort_periods = np.array([3, 5, 7])
    first_treat = np.zeros(n_units, dtype=int)
    if n_treated > 0:
        cohort_assignments = rng.choice(len(cohort_periods), size=n_treated)
        first_treat[n_never:] = cohort_periods[cohort_assignments]

    first_treat_expanded = np.repeat(first_treat, n_periods)

    unit_fe = rng.standard_normal(n_units) * 2.0
    time_fe = np.linspace(0, 1, n_periods)

    unit_fe_expanded = np.repeat(unit_fe, n_periods)
    time_fe_expanded = np.tile(time_fe, n_units)

    post = (times >= first_treat_expanded) & (first_treat_expanded > 0)
    relative_time = times - first_treat_expanded
    dynamic_mult = 1 + 0.1 * np.maximum(relative_time, 0)
    effect = treatment_effect * dynamic_mult

    outcomes = (
        unit_fe_expanded
        + time_fe_expanded
        + effect * post
        + rng.standard_normal(len(units)) * 0.5
    )

    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "outcome": outcomes,
            "first_treat": first_treat_expanded,
        }
    )


# =============================================================================
# ImputationDiD pretrends tests
# =============================================================================


class TestImputationPretrends:
    """Tests for ImputationDiD pretrends feature."""

    def test_pretrends_includes_negative_horizons(self):
        """Pre-period horizons appear in event study when pretrends=True."""
        data = generate_test_data()
        est = ImputationDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results.event_study_effects is not None
        horizons = sorted(results.event_study_effects.keys())
        negative = [h for h in horizons if h < 0]
        positive = [h for h in horizons if h >= 0]
        assert len(negative) > 1, "Should have pre-period horizons"
        assert len(positive) > 0, "Should have post-treatment horizons"

    def test_pretrends_coefficients_near_zero(self):
        """Pre-period effects ~0 under parallel trends (no violation)."""
        data = generate_test_data(treatment_effect=2.0, seed=99)
        est = ImputationDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        ref_period = -1
        for h, eff in results.event_study_effects.items():
            if h >= 0 or h == ref_period:
                continue
            assert np.isfinite(eff["effect"]), f"h={h}: effect not finite"
            assert abs(eff["effect"]) < 3 * eff["se"] + 0.5, (
                f"h={h}: pre-period effect {eff['effect']:.3f} too large"
            )

    def test_pretrends_se_finite_positive(self):
        """All pre-period horizons have finite, positive SEs."""
        data = generate_test_data()
        est = ImputationDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        ref_period = -1
        for h, eff in results.event_study_effects.items():
            if h == ref_period:
                continue
            if eff["n_obs"] == 0:
                continue
            assert np.isfinite(eff["se"]), f"h={h}: SE not finite"
            assert eff["se"] > 0, f"h={h}: SE not positive"

    def test_reference_period_correct(self):
        """Reference period h=-1 normalized to 0."""
        data = generate_test_data()
        est = ImputationDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert -1 in results.event_study_effects
        ref = results.event_study_effects[-1]
        assert ref["effect"] == 0.0
        assert ref["n_obs"] == 0

    def test_backward_compatibility(self):
        """pretrends=False (default) gives identical results."""
        data = generate_test_data()

        results_default = ImputationDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        results_false = ImputationDiD(pretrends=False).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results_default.overall_att == results_false.overall_att
        assert results_default.overall_se == results_false.overall_se
        assert set(results_default.event_study_effects.keys()) == set(
            results_false.event_study_effects.keys()
        )

    def test_post_treatment_invariance(self):
        """Post-treatment effects identical with pretrends=True vs False."""
        data = generate_test_data()

        results_off = ImputationDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        results_on = ImputationDiD(pretrends=True).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Overall ATT unchanged
        assert results_on.overall_att == results_off.overall_att
        assert results_on.overall_se == results_off.overall_se

        # Post-treatment event study effects unchanged
        for h in results_off.event_study_effects:
            assert h in results_on.event_study_effects
            eff_off = results_off.event_study_effects[h]
            eff_on = results_on.event_study_effects[h]
            np.testing.assert_allclose(
                eff_off["effect"], eff_on["effect"], rtol=1e-10
            )
            np.testing.assert_allclose(eff_off["se"], eff_on["se"], rtol=1e-10)

    def test_horizon_max_interaction(self):
        """horizon_max limits both pre and post horizons."""
        data = generate_test_data()
        est = ImputationDiD(pretrends=True, horizon_max=2)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        for h in results.event_study_effects:
            assert abs(h) <= 2, f"h={h} exceeds horizon_max=2"

    def test_anticipation_interaction(self):
        """With anticipation=1, reference shifts to h=-2."""
        data = generate_test_data()
        est = ImputationDiD(pretrends=True, anticipation=1)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert -2 in results.event_study_effects
        ref = results.event_study_effects[-2]
        assert ref["effect"] == 0.0
        assert ref["n_obs"] == 0

    def test_get_params_includes_pretrends(self):
        """get_params includes pretrends parameter."""
        est = ImputationDiD(pretrends=True)
        params = est.get_params()
        assert "pretrends" in params
        assert params["pretrends"] is True

        est2 = ImputationDiD()
        assert est2.get_params()["pretrends"] is False

    def test_no_pretreatment_obs_graceful(self):
        """All units treated at t=1 with pretrends=True: no error."""
        rng = np.random.default_rng(42)
        n_units = 20
        n_periods = 5
        data = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(n_units), n_periods),
                "time": np.tile(np.arange(n_periods), n_units),
                "outcome": rng.standard_normal(n_units * n_periods),
                "first_treat": np.repeat(
                    np.ones(n_units, dtype=int), n_periods
                ),
            }
        )

        est = ImputationDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        assert results.event_study_effects is not None


# =============================================================================
# TwoStageDiD pretrends tests
# =============================================================================


class TestTwoStagePretrends:
    """Tests for TwoStageDiD pretrends feature."""

    def test_pretrends_includes_negative_horizons(self):
        """Pre-period horizons appear in event study when pretrends=True."""
        data = generate_test_data()
        est = TwoStageDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert results.event_study_effects is not None
        horizons = sorted(results.event_study_effects.keys())
        negative = [h for h in horizons if h < 0]
        positive = [h for h in horizons if h >= 0]
        assert len(negative) > 1, "Should have pre-period horizons"
        assert len(positive) > 0, "Should have post-treatment horizons"

    def test_pretrends_coefficients_near_zero(self):
        """Pre-period effects ~0 under parallel trends."""
        data = generate_test_data(treatment_effect=2.0, seed=99)
        est = TwoStageDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        ref_period = -1
        for h, eff in results.event_study_effects.items():
            if h >= 0 or h == ref_period:
                continue
            assert np.isfinite(eff["effect"]), f"h={h}: effect not finite"
            assert abs(eff["effect"]) < 3 * eff["se"] + 0.5, (
                f"h={h}: pre-period effect {eff['effect']:.3f} too large"
            )

    def test_pretrends_se_finite_positive(self):
        """All pre-period horizons have finite, positive SEs."""
        data = generate_test_data()
        est = TwoStageDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        ref_period = -1
        for h, eff in results.event_study_effects.items():
            if h == ref_period:
                continue
            if eff["n_obs"] == 0:
                continue
            assert np.isfinite(eff["se"]), f"h={h}: SE not finite"
            assert eff["se"] > 0, f"h={h}: SE not positive"

    def test_post_treatment_invariance(self):
        """Post-treatment effects identical with pretrends=True vs False."""
        data = generate_test_data()

        results_off = TwoStageDiD().fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        results_on = TwoStageDiD(pretrends=True).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Overall ATT unchanged
        assert results_on.overall_att == results_off.overall_att
        assert results_on.overall_se == results_off.overall_se

        # Post-treatment event study effects unchanged
        for h in results_off.event_study_effects:
            assert h in results_on.event_study_effects
            eff_off = results_off.event_study_effects[h]
            eff_on = results_on.event_study_effects[h]
            np.testing.assert_allclose(
                eff_off["effect"], eff_on["effect"], rtol=1e-10
            )
            np.testing.assert_allclose(eff_off["se"], eff_on["se"], rtol=1e-10)

    def test_get_params_includes_pretrends(self):
        """get_params includes pretrends parameter."""
        est = TwoStageDiD(pretrends=True)
        params = est.get_params()
        assert "pretrends" in params
        assert params["pretrends"] is True

    def test_horizon_max_interaction(self):
        """horizon_max limits both pre and post horizons."""
        data = generate_test_data()
        est = TwoStageDiD(pretrends=True, horizon_max=2)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        for h in results.event_study_effects:
            assert abs(h) <= 2, f"h={h} exceeds horizon_max=2"

    def test_anticipation_interaction(self):
        """With anticipation=1, reference shifts to h=-2."""
        data = generate_test_data()
        est = TwoStageDiD(pretrends=True, anticipation=1)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        assert -2 in results.event_study_effects
        ref = results.event_study_effects[-2]
        assert ref["effect"] == 0.0
        assert ref["n_obs"] == 0


# =============================================================================
# Cross-estimator consistency
# =============================================================================


class TestPretrendsCrossEstimator:
    """Verify ImputationDiD and TwoStageDiD produce consistent pre-period effects."""

    def test_point_estimates_close(self):
        """ImputationDiD and TwoStageDiD pre-period effects should be similar."""
        data = generate_test_data(seed=77)

        imp = ImputationDiD(pretrends=True).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )
        ts = TwoStageDiD(pretrends=True).fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
        )

        # Both should have negative horizons
        imp_neg = {
            h: v["effect"]
            for h, v in imp.event_study_effects.items()
            if h < -1 and v["n_obs"] > 0
        }
        ts_neg = {
            h: v["effect"]
            for h, v in ts.event_study_effects.items()
            if h < -1 and v["n_obs"] > 0
        }

        common_h = set(imp_neg.keys()) & set(ts_neg.keys())
        assert len(common_h) > 0, "No common pre-period horizons"

        for h in common_h:
            # Point estimates should be numerically identical (same imputation)
            np.testing.assert_allclose(
                imp_neg[h],
                ts_neg[h],
                atol=1e-6,
                err_msg=f"h={h}: point estimates differ",
            )


# =============================================================================
# Regression tests for P0/P1 review findings
# =============================================================================


class TestPretrends_Regressions:
    """Regression tests for bugs found in AI code review."""

    def test_imputation_survey_weighted_no_covariates(self):
        """ImputationDiD with survey weights, no covariates, no pretrends.

        Regression for P1: untreated_units/untreated_times uninitialized
        in the survey-weighted FE-only variance path.
        """
        from diff_diff.survey import SurveyDesign

        data = generate_test_data(seed=42)
        rng = np.random.default_rng(42)
        n_units = data["unit"].nunique()
        unit_weights = rng.uniform(0.5, 2.0, n_units)
        data["weight"] = data["unit"].map(
            dict(enumerate(unit_weights))
        )

        sd = SurveyDesign(weights="weight")
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )

        assert np.isfinite(results.overall_att)
        assert np.isfinite(results.overall_se)
        assert results.overall_se > 0

    def test_imputation_survey_weighted_event_study(self):
        """ImputationDiD with survey weights and event study aggregation."""
        from diff_diff.survey import SurveyDesign

        data = generate_test_data(seed=42)
        rng = np.random.default_rng(42)
        n_units = data["unit"].nunique()
        unit_weights = rng.uniform(0.5, 2.0, n_units)
        data["weight"] = data["unit"].map(
            dict(enumerate(unit_weights))
        )

        sd = SurveyDesign(weights="weight")
        est = ImputationDiD()
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
            survey_design=sd,
        )

        for h, eff in results.event_study_effects.items():
            if eff["n_obs"] == 0:
                continue
            assert np.isfinite(eff["se"]), f"h={h}: SE not finite"

    def test_imputation_pretrends_with_covariates(self):
        """ImputationDiD pretrends=True with covariates.

        Tests _compute_v_untreated_with_covariates_preperiod().
        """
        data = generate_test_data(seed=42)
        rng = np.random.default_rng(42)
        data["x1"] = rng.standard_normal(len(data))

        est = ImputationDiD(pretrends=True)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            covariates=["x1"],
            aggregate="event_study",
        )

        negative = {
            h: v
            for h, v in results.event_study_effects.items()
            if h < -1 and v["n_obs"] > 0
        }
        assert len(negative) > 0, "Should have pre-period horizons"
        for h, eff in negative.items():
            assert np.isfinite(eff["se"]), f"h={h}: SE not finite with covariates"
            assert eff["se"] > 0, f"h={h}: SE not positive with covariates"

    def test_imputation_pretrends_balance_e_bootstrap(self):
        """ImputationDiD pretrends=True + balance_e + bootstrap.

        Regression for P0: bootstrap must use the same balance_e cohort
        filter (union of pre + post horizons) as the analytical path.
        """
        data = generate_test_data(seed=42)
        est = ImputationDiD(pretrends=True, n_bootstrap=50, seed=42)
        results = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            aggregate="event_study",
            balance_e=1,
        )

        assert results.event_study_effects is not None
        # Verify pre-period horizons have bootstrap inference
        negative = {
            h: v
            for h, v in results.event_study_effects.items()
            if h < -1 and v["n_obs"] > 0
        }
        for h, eff in negative.items():
            assert np.isfinite(eff["se"]), f"h={h}: SE not finite"
            assert eff["se"] > 0, f"h={h}: SE not positive"
