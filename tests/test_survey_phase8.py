"""Tests for Phase 8 Survey Maturity: SDR replicate method + FPC support."""

import numpy as np
import pandas as pd
import pytest

from diff_diff.survey import (
    SurveyDesign,
    _replicate_variance_factor,
    compute_replicate_if_variance,
    compute_replicate_vcov,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_replicate_data(n=200, k=20, seed=123):
    """Data with full-sample + replicate weight columns."""
    np.random.seed(seed)
    x = np.random.randn(n)
    eps = np.random.randn(n) * 0.5
    y = 1.0 + 2.0 * x + eps
    w = 1.0 + np.random.exponential(0.5, n)
    data = pd.DataFrame({"x": x, "y": y, "weight": w})

    # BRR-style replicates: w * (1 + 0.5 * sign) with random signs per cluster
    rng = np.random.RandomState(seed + 1)
    cluster_size = n // k
    rep_cols = []
    for r in range(k):
        signs = rng.choice([-1, 1], size=k)
        perturbation = np.repeat(signs, cluster_size)[:n]
        w_r = w * (1.0 + 0.5 * perturbation)
        w_r = np.maximum(w_r, 0.0)
        col = f"rep_{r}"
        data[col] = w_r
        rep_cols.append(col)
    return data, rep_cols


@pytest.fixture
def replicate_data():
    return _make_replicate_data()


def _make_staggered_panel(n_units=50, n_periods=8, seed=456):
    """Multi-period staggered panel for ImputationDiD/TwoStageDiD tests."""
    np.random.seed(seed)
    rows = []
    for i in range(n_units):
        if i < 15:
            ft = 4  # cohort 1
        elif i < 30:
            ft = 6  # cohort 2
        else:
            ft = 0  # never-treated
        # Assign to strata and PSUs for survey design
        stratum = i % 3
        psu = i  # one PSU per unit
        wt = 1.0 + 0.3 * (i % 5)
        for t in range(1, n_periods + 1):
            y = 10.0 + i * 0.03 + t * 0.2
            if ft > 0 and t >= ft:
                y += 2.0
            y += np.random.normal(0, 0.4)
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "first_treat": ft,
                    "outcome": y,
                    "weight": wt,
                    "stratum": stratum,
                    "psu": psu,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def staggered_data():
    return _make_staggered_panel()


# ===========================================================================
# 8a: SDR Replicate Method Tests
# ===========================================================================


class TestSDR:
    """Tests for Successive Difference Replication (SDR) method."""

    def test_sdr_accepted(self, replicate_data):
        """SurveyDesign with replicate_method='SDR' resolves without error."""
        data, rep_cols = replicate_data
        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="SDR",
        )
        resolved = sd.resolve(data)
        assert resolved.replicate_method == "SDR"
        assert resolved.n_replicates == len(rep_cols)

    def test_sdr_variance_factor(self):
        """SDR variance factor is 4/R."""
        assert _replicate_variance_factor("SDR", 80, 0.0) == pytest.approx(4.0 / 80)
        assert _replicate_variance_factor("SDR", 20, 0.0) == pytest.approx(4.0 / 20)
        assert _replicate_variance_factor("SDR", 1, 0.0) == pytest.approx(4.0)

    def test_sdr_4x_brr_scaling(self, replicate_data):
        """SDR variance == 4x BRR variance on same data (4/R vs 1/R)."""
        from diff_diff.linalg import solve_ols

        data, rep_cols = replicate_data
        y = data["y"].values
        X = np.column_stack([np.ones(len(data)), data["x"].values])

        sd_brr = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="BRR",
        )
        resolved_brr = sd_brr.resolve(data)
        coef, _, _ = solve_ols(X, y, weights=resolved_brr.weights, weight_type="pweight")
        vcov_brr, _ = compute_replicate_vcov(X, y, coef, resolved_brr)

        sd_sdr = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="SDR",
        )
        resolved_sdr = sd_sdr.resolve(data)
        vcov_sdr, _ = compute_replicate_vcov(X, y, coef, resolved_sdr)

        # SDR uses 4/R, BRR uses 1/R => SDR = 4 * BRR
        np.testing.assert_allclose(np.diag(vcov_sdr), 4.0 * np.diag(vcov_brr))

    def test_sdr_vcov_finite(self, replicate_data):
        """compute_replicate_vcov produces finite results with SDR."""
        from diff_diff.linalg import solve_ols

        data, rep_cols = replicate_data
        y = data["y"].values
        X = np.column_stack([np.ones(len(data)), data["x"].values])

        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="SDR",
        )
        resolved = sd.resolve(data)
        coef, _, _ = solve_ols(X, y, weights=resolved.weights, weight_type="pweight")
        vcov, n_valid = compute_replicate_vcov(X, y, coef, resolved)
        assert np.all(np.isfinite(np.diag(vcov)))
        assert np.all(np.diag(vcov) > 0)
        assert n_valid == len(rep_cols)

    def test_sdr_if_variance_finite(self, replicate_data):
        """compute_replicate_if_variance produces finite results with SDR."""
        data, rep_cols = replicate_data
        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="SDR",
        )
        resolved = sd.resolve(data)
        psi = np.random.randn(len(data)) * 0.1
        var, n_valid = compute_replicate_if_variance(psi, resolved)
        assert np.isfinite(var)
        assert var >= 0
        assert n_valid == len(rep_cols)

    def test_sdr_if_4x_brr(self, replicate_data):
        """SDR IF variance == 4x BRR IF variance."""
        data, rep_cols = replicate_data
        np.random.seed(99)
        psi = np.random.randn(len(data)) * 0.1

        sd_brr = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="BRR",
        )
        resolved_brr = sd_brr.resolve(data)
        var_brr, _ = compute_replicate_if_variance(psi, resolved_brr)

        sd_sdr = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="SDR",
        )
        resolved_sdr = sd_sdr.resolve(data)
        var_sdr, _ = compute_replicate_if_variance(psi, resolved_sdr)

        assert var_sdr == pytest.approx(4.0 * var_brr)

    def test_sdr_rejects_fay_rho(self, replicate_data):
        """SDR with fay_rho > 0 raises ValueError (generic validation)."""
        data, rep_cols = replicate_data
        with pytest.raises(ValueError, match="fay_rho must be 0"):
            SurveyDesign(
                weights="weight",
                replicate_weights=rep_cols,
                replicate_method="SDR",
                fay_rho=0.3,
            )

    def test_sdr_ignores_custom_scales(self, replicate_data):
        """SDR warns and ignores custom replicate_scale/replicate_rscales."""
        from diff_diff.linalg import solve_ols

        data, rep_cols = replicate_data
        y = data["y"].values
        X = np.column_stack([np.ones(len(data)), data["x"].values])

        sd_plain = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="SDR",
        )
        resolved_plain = sd_plain.resolve(data)
        coef, _, _ = solve_ols(X, y, weights=resolved_plain.weights, weight_type="pweight")
        vcov_plain, _ = compute_replicate_vcov(X, y, coef, resolved_plain)

        sd_scale = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="SDR",
            replicate_scale=99.0,
        )
        resolved_scale = sd_scale.resolve(data)
        with pytest.warns(UserWarning, match="ignored for SDR"):
            vcov_scale, _ = compute_replicate_vcov(X, y, coef, resolved_scale)

        # Custom scale should be ignored — results identical
        np.testing.assert_allclose(np.diag(vcov_scale), np.diag(vcov_plain))

    def test_sdr_end_to_end_callaway_santanna(self):
        """CallawaySantAnna + SDR end-to-end smoke test."""
        from diff_diff import CallawaySantAnna

        np.random.seed(42)
        n_units, n_periods = 40, 6
        rows = []
        for i in range(n_units):
            ft = 4 if i < 15 else (0 if i >= 30 else 6)
            wt = 1.0 + 0.2 * (i % 5)
            for t in range(1, n_periods + 1):
                y = 5.0 + i * 0.02 + t * 0.1
                if ft > 0 and t >= ft:
                    y += 1.5
                y += np.random.normal(0, 0.3)
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "first_treat": ft,
                        "outcome": y,
                        "weight": wt,
                    }
                )
        data = pd.DataFrame(rows)

        # Add BRR-style replicate weights (re-used as SDR)
        rng = np.random.RandomState(100)
        n_rep = 16
        rep_cols = []
        units = sorted(data["unit"].unique())
        for r in range(n_rep):
            signs = rng.choice([-1, 1], size=len(units))
            sign_map = dict(zip(units, signs))
            perturbation = data["unit"].map(sign_map).values.astype(float)
            w_r = data["weight"].values * (1.0 + 0.5 * perturbation)
            w_r = np.maximum(w_r, 0.0)
            col = f"sdr_{r}"
            data[col] = w_r
            rep_cols.append(col)

        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="SDR",
        )

        est = CallawaySantAnna()
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0


# ===========================================================================
# 8b: FPC in ImputationDiD and TwoStageDiD
# ===========================================================================


class TestImputationDiDFPC:
    """FPC support in ImputationDiD."""

    def test_imputation_fpc_no_raise(self, staggered_data):
        """ImputationDiD with FPC runs without error."""
        from diff_diff import ImputationDiD

        data = staggered_data
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc_col",
        )
        # Add FPC column: population size per stratum (larger than sample)
        n_units = data["unit"].nunique()
        data["fpc_col"] = data["stratum"].map({0: n_units * 3, 1: n_units * 3, 2: n_units * 3})

        est = ImputationDiD(horizon_max=3)
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0

    def test_imputation_fpc_reduces_se(self, staggered_data):
        """FPC SE should be <= no-FPC SE (by definition)."""
        from diff_diff import ImputationDiD

        data = staggered_data
        # Large FPC (small correction)
        data["fpc_col"] = data["stratum"].map({0: 500, 1: 500, 2: 500})

        sd_no_fpc = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        sd_fpc = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc_col",
        )

        est = ImputationDiD(horizon_max=3)
        result_no_fpc = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd_no_fpc,
        )
        result_fpc = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd_fpc,
        )
        assert result_fpc.overall_se <= result_no_fpc.overall_se + 1e-10

    def test_imputation_fpc_event_study(self, staggered_data):
        """FPC works with aggregate='event_study'."""
        from diff_diff import ImputationDiD

        data = staggered_data
        data["fpc_col"] = data["stratum"].map({0: 500, 1: 500, 2: 500})

        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc_col",
        )
        est = ImputationDiD(horizon_max=3)
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
            aggregate="event_study",
        )
        assert result.event_study_effects is not None
        for h, eff in result.event_study_effects.items():
            assert np.isfinite(eff["se"]), f"Non-finite SE at horizon {h}"

    def test_imputation_fpc_group(self, staggered_data):
        """FPC works with aggregate='group'."""
        from diff_diff import ImputationDiD

        data = staggered_data
        data["fpc_col"] = data["stratum"].map({0: 500, 1: 500, 2: 500})

        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc_col",
        )
        est = ImputationDiD(horizon_max=3)
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
            aggregate="group",
        )
        assert result.group_effects is not None
        for g, eff in result.group_effects.items():
            assert np.isfinite(eff["se"]), f"Non-finite SE for group {g}"


class TestTwoStageDiDFPC:
    """FPC support in TwoStageDiD."""

    def test_two_stage_fpc_no_raise(self, staggered_data):
        """TwoStageDiD with FPC runs without error."""
        from diff_diff import TwoStageDiD

        data = staggered_data
        data["fpc_col"] = data["stratum"].map({0: 500, 1: 500, 2: 500})

        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc_col",
        )
        est = TwoStageDiD()
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0

    def test_two_stage_fpc_reduces_se(self, staggered_data):
        """FPC SE should be <= no-FPC SE."""
        from diff_diff import TwoStageDiD

        data = staggered_data
        data["fpc_col"] = data["stratum"].map({0: 500, 1: 500, 2: 500})

        sd_no_fpc = SurveyDesign(weights="weight", strata="stratum", psu="psu")
        sd_fpc = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc_col",
        )

        est = TwoStageDiD()
        result_no_fpc = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd_no_fpc,
        )
        result_fpc = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd_fpc,
        )
        assert result_fpc.overall_se <= result_no_fpc.overall_se + 1e-10

    def test_two_stage_fpc_event_study(self, staggered_data):
        """FPC works with aggregate='event_study'."""
        from diff_diff import TwoStageDiD

        data = staggered_data
        data["fpc_col"] = data["stratum"].map({0: 500, 1: 500, 2: 500})

        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc_col",
        )
        est = TwoStageDiD()
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
            aggregate="event_study",
        )
        assert result.event_study_effects is not None
        for h, eff in result.event_study_effects.items():
            assert np.isfinite(eff["se"]), f"Non-finite SE at horizon {h}"

    def test_two_stage_fpc_group(self, staggered_data):
        """FPC works with aggregate='group'."""
        from diff_diff import TwoStageDiD

        data = staggered_data
        data["fpc_col"] = data["stratum"].map({0: 500, 1: 500, 2: 500})

        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc_col",
        )
        est = TwoStageDiD()
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
            aggregate="group",
        )
        assert result.group_effects is not None
        for g, eff in result.group_effects.items():
            assert np.isfinite(eff["se"]), f"Non-finite SE for group {g}"


class TestUnstratifiedFPC:
    """Regression tests for FPC without explicit strata (P1 fix)."""

    def test_imputation_unstratified_fpc(self, staggered_data):
        """ImputationDiD with weights + psu + fpc (no strata) runs."""
        from diff_diff import ImputationDiD

        data = staggered_data
        data["fpc_col"] = 500  # constant FPC for all

        sd = SurveyDesign(weights="weight", psu="psu", fpc="fpc_col")
        est = ImputationDiD(horizon_max=3)
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0

    def test_two_stage_unstratified_fpc(self, staggered_data):
        """TwoStageDiD with weights + psu + fpc (no strata) runs."""
        from diff_diff import TwoStageDiD

        data = staggered_data
        data["fpc_col"] = 500

        sd = SurveyDesign(weights="weight", psu="psu", fpc="fpc_col")
        est = TwoStageDiD()
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
        )
        assert np.isfinite(result.overall_att)
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0

    def test_two_stage_unstratified_fpc_event_study(self, staggered_data):
        """TwoStageDiD unstratified FPC with event study."""
        from diff_diff import TwoStageDiD

        data = staggered_data
        data["fpc_col"] = 500

        sd = SurveyDesign(weights="weight", psu="psu", fpc="fpc_col")
        est = TwoStageDiD()
        result = est.fit(
            data,
            outcome="outcome",
            unit="unit",
            time="time",
            first_treat="first_treat",
            survey_design=sd,
            aggregate="event_study",
        )
        assert result.event_study_effects is not None
        for h, eff in result.event_study_effects.items():
            assert np.isfinite(eff["se"]), f"Non-finite SE at horizon {h}"


class TestPSUMeatHelper:
    """Unit tests for _compute_stratified_meat_from_psu_scores."""

    def test_no_strata_matches_simple(self):
        """Without strata/FPC, helper should match S.T @ S."""
        from diff_diff.survey import _compute_stratified_meat_from_psu_scores

        np.random.seed(42)
        G, k = 20, 3
        S = np.random.randn(G, k) * 0.1
        # All in one stratum, no FPC
        psu_strata = np.zeros(G, dtype=int)
        meat, var_computed, legit_zero = _compute_stratified_meat_from_psu_scores(S, psu_strata)

        # Without FPC: adjustment = (1 - 0) * (G / (G-1)) = G/(G-1)
        expected = (G / (G - 1)) * (S - S.mean(axis=0)).T @ (S - S.mean(axis=0))
        np.testing.assert_allclose(meat, expected, rtol=1e-10)
        assert var_computed is True

    def test_fpc_reduces_meat(self):
        """FPC should reduce the meat matrix vs no-FPC."""
        from diff_diff.survey import _compute_stratified_meat_from_psu_scores

        np.random.seed(42)
        G, k = 20, 3
        S = np.random.randn(G, k) * 0.1
        psu_strata = np.zeros(G, dtype=int)

        meat_no_fpc, _, _ = _compute_stratified_meat_from_psu_scores(S, psu_strata)
        fpc = np.full(G, 100.0)  # 20 out of 100 sampled
        meat_fpc, _, _ = _compute_stratified_meat_from_psu_scores(
            S,
            psu_strata,
            fpc_per_psu=fpc,
        )

        # FPC reduces variance
        assert np.all(np.diag(meat_fpc) <= np.diag(meat_no_fpc) + 1e-15)
        # Not zero (f = 20/100 = 0.2, so 1-f = 0.8)
        assert np.all(np.diag(meat_fpc) > 0)

    def test_lonely_psu_adjust(self):
        """Lonely PSU with adjust contributes nonzero variance."""
        from diff_diff.survey import _compute_stratified_meat_from_psu_scores

        np.random.seed(42)
        G, k = 5, 2
        S = np.random.randn(G, k)
        # Stratum 0 has 1 PSU (lonely), stratum 1 has 4 PSUs
        psu_strata = np.array([0, 1, 1, 1, 1])

        meat_remove, vc_remove, _ = _compute_stratified_meat_from_psu_scores(
            S,
            psu_strata,
            lonely_psu="remove",
        )
        meat_adjust, vc_adjust, _ = _compute_stratified_meat_from_psu_scores(
            S,
            psu_strata,
            lonely_psu="adjust",
        )

        # "adjust" should produce larger variance (lonely PSU contributes)
        assert np.all(np.diag(meat_adjust) >= np.diag(meat_remove) - 1e-15)
        assert vc_adjust is True

    def test_all_singleton_remove_unidentified(self):
        """All singleton strata with remove: variance unidentified."""
        from diff_diff.survey import _compute_stratified_meat_from_psu_scores

        np.random.seed(42)
        G, k = 5, 2
        S = np.random.randn(G, k)
        # Every PSU is in its own stratum (all singletons)
        psu_strata = np.arange(G)

        meat, var_computed, legit_zero = _compute_stratified_meat_from_psu_scores(
            S,
            psu_strata,
            lonely_psu="remove",
        )
        # No variance was computed, no legitimate zeros
        assert var_computed is False
        assert legit_zero == 0
        # Meat is zero matrix (unidentified — caller should produce NaN)
        assert np.all(meat == 0.0)
