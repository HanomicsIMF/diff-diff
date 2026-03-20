"""Tests for survey data support (diff_diff.survey)."""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    DifferenceInDifferences,
    MultiPeriodDiD,
    SurveyDesign,
    SurveyMetadata,
)
from diff_diff.linalg import LinearRegression, solve_ols
from diff_diff.survey import (
    ResolvedSurveyDesign,
    compute_survey_metadata,
    compute_survey_vcov,
)
from diff_diff.twfe import TwoWayFixedEffects
from diff_diff.utils import within_transform

# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def survey_2x2_data():
    """2x2 DiD panel with survey design columns.

    100 units, 2 periods, known ATT = 3.0.
    Strata, PSU, FPC, and sampling weights are included.
    """
    np.random.seed(42)
    n_units = 100
    n_treated = 50
    rows = []
    for unit in range(n_units):
        is_treated = unit < n_treated
        stratum = unit % 5  # 5 strata
        psu = unit // 5  # 20 PSUs (4 per stratum)
        fpc_val = 200.0  # Population size per stratum
        # Sampling weight proportional to stratum population / sample count
        wt = 1.0 + 0.5 * stratum

        for period in [0, 1]:
            y = 10.0 + unit * 0.1
            if period == 1:
                y += 5.0
            if is_treated and period == 1:
                y += 3.0
            y += np.random.normal(0, 0.5)

            rows.append(
                {
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "post": period,
                    "outcome": y,
                    "stratum": stratum,
                    "psu": psu,
                    "fpc": fpc_val,
                    "weight": wt,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def twfe_panel_data():
    """Balanced panel for TWFE: 50 units, 2 periods, survey columns."""
    np.random.seed(123)
    n_units = 50
    rows = []
    for unit in range(n_units):
        is_treated = unit < 25
        stratum = unit % 5
        psu = unit // 5
        wt = 1.0 + 0.3 * stratum

        for period in [0, 1]:
            y = 5.0 + unit * 0.05
            if period == 1:
                y += 2.0
            if is_treated and period == 1:
                y += 4.0
            y += np.random.normal(0, 0.3)

            rows.append(
                {
                    "unit": unit,
                    "period": period,
                    "treated": int(is_treated),
                    "post": period,
                    "outcome": y,
                    "stratum": stratum,
                    "psu": psu,
                    "fpc": 100.0,
                    "weight": wt,
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def multiperiod_data():
    """Multi-period panel: 60 units, 5 periods, treatment at period 3."""
    np.random.seed(99)
    n_units = 60
    n_treated = 30
    periods = [1, 2, 3, 4, 5]
    rows = []
    for unit in range(n_units):
        is_treated = unit < n_treated
        stratum = unit % 3
        psu = unit // 3
        wt = 1.0 + 0.4 * stratum

        for t in periods:
            y = 8.0 + unit * 0.03 + t * 0.5
            if is_treated and t >= 3:
                y += 2.5  # True ATT = 2.5
            y += np.random.normal(0, 0.4)

            rows.append(
                {
                    "unit": unit,
                    "period": t,
                    "treated": int(is_treated),
                    "outcome": y,
                    "stratum": stratum,
                    "psu": psu,
                    "fpc": 200.0,
                    "weight": wt,
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# Tier 1: Analytical Verification (exact match, tol <= 1e-10)
# =============================================================================


class TestAnalyticalVerification:
    """Exact-match tests against manual computation."""

    def test_wls_coefficients_manual(self):
        """WLS coefficients match manual (X'WX)^{-1} X'Wy computation."""
        np.random.seed(10)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        beta_true = np.array([2.0, 3.0])
        y = X @ beta_true + np.random.randn(n) * 0.5
        weights = np.abs(np.random.randn(n)) + 0.5

        # Normalise pweights to mean=1
        w = weights * (n / np.sum(weights))

        # Manual WLS: (X'WX)^{-1} X'Wy
        W = np.diag(w)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        beta_manual = np.linalg.solve(XtWX, XtWy)

        # Via solve_ols
        coef, _resid, _vcov = solve_ols(X, y, weights=w, weight_type="pweight")

        np.testing.assert_allclose(coef, beta_manual, atol=1e-10)

    def test_uniform_weights_equals_unweighted(self):
        """Uniform weights (all ones) give identical results to no weights."""
        np.random.seed(20)
        n = 80
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 1.0 + 2.0 * X[:, 1] + np.random.randn(n) * 0.3

        coef_unw, resid_unw, vcov_unw = solve_ols(X, y)
        coef_w, resid_w, vcov_w = solve_ols(X, y, weights=np.ones(n), weight_type="pweight")

        np.testing.assert_allclose(coef_w, coef_unw, atol=1e-10)
        np.testing.assert_allclose(resid_w, resid_unw, atol=1e-10)
        np.testing.assert_allclose(vcov_w, vcov_unw, atol=1e-10)

    def test_weight_normalization(self):
        """resolve() normalises pweights to mean=1; fweights are unchanged."""
        n = 60
        df = pd.DataFrame(
            {
                "y": np.random.randn(n),
                "pw": np.random.uniform(1, 5, n),
                "fw": np.random.choice([1, 2, 3], n).astype(float),
            }
        )

        # pweight normalisation: mean should be 1 (sum should be n)
        sd_pw = SurveyDesign(weights="pw", weight_type="pweight")
        resolved_pw = sd_pw.resolve(df)
        np.testing.assert_allclose(np.mean(resolved_pw.weights), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.sum(resolved_pw.weights), n, atol=1e-10)

        # fweight: unchanged
        sd_fw = SurveyDesign(weights="fw", weight_type="fweight")
        resolved_fw = sd_fw.resolve(df)
        np.testing.assert_allclose(resolved_fw.weights, df["fw"].values, atol=1e-12)

    def test_scale_invariance(self):
        """Scaling all pweights by a constant c does not change coefs or SEs."""
        np.random.seed(30)
        n = 60
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 5.0 + X[:, 1] * 2.0 + np.random.randn(n) * 0.5
        base_weights = np.abs(np.random.randn(n)) + 0.5

        # After normalisation, scale is irrelevant, but let's check via
        # full DiD pipeline with DataFrames.
        df = pd.DataFrame(
            {
                "outcome": y,
                "treated": np.random.randint(0, 2, n),
                "post": np.random.randint(0, 2, n),
                "w1": base_weights,
                "w2": base_weights * 100.0,
            }
        )

        did = DifferenceInDifferences()
        sd1 = SurveyDesign(weights="w1", weight_type="pweight")
        sd2 = SurveyDesign(weights="w2", weight_type="pweight")
        r1 = did.fit(df, outcome="outcome", treatment="treated", time="post", survey_design=sd1)
        r2 = did.fit(df, outcome="outcome", treatment="treated", time="post", survey_design=sd2)

        np.testing.assert_allclose(r1.att, r2.att, atol=1e-10)
        np.testing.assert_allclose(r1.se, r2.se, atol=1e-10)

    def test_weighted_residuals(self):
        """Residuals are on the original (unweighted) scale: y - X @ beta."""
        np.random.seed(40)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 3.0 + X[:, 1] * 1.5 + np.random.randn(n) * 0.3
        w = np.abs(np.random.randn(n)) + 0.5
        w = w * (n / np.sum(w))  # normalise

        coef, resid, _vcov = solve_ols(X, y, weights=w, weight_type="pweight")

        expected_resid = y - X @ coef
        np.testing.assert_allclose(resid, expected_resid, atol=1e-10)


# =============================================================================
# Tier 2: Exact Manual Oracle Tests
# =============================================================================


class TestReferenceValues:
    """Tier 2: Exact manual oracle tests (replaces placeholder R cross-validation)."""

    def test_wls_coefficients_exact_oracle(self):
        """Verify WLS coefficients against hand-computed (X'WX)^{-1}X'Wy for n=4."""
        # 2x2 DiD design with 2 obs per cell (n=8 > k=4)
        X = np.array(
            [
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]
        )
        y = np.array([10.0, 10.5, 12.0, 12.5, 11.0, 11.5, 16.0, 16.5])
        w = np.array([1.0, 1.5, 2.0, 1.0, 1.0, 1.5, 2.0, 1.0])

        # Hand-compute: beta = (X'WX)^{-1} X'Wy
        W = np.diag(w)
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y
        beta_expected = np.linalg.solve(XtWX, XtWy)

        coef, resid, vcov = solve_ols(X, y, weights=w, weight_type="pweight")
        np.testing.assert_allclose(coef, beta_expected, atol=1e-10)

    def test_weighted_hc1_vcov_exact_oracle(self):
        """Verify weighted HC1 vcov against hand-computed sandwich formula."""
        np.random.seed(42)
        X = np.column_stack([np.ones(8), np.random.randn(8)])
        y = 1.0 + 2.0 * X[:, 1] + np.random.randn(8) * 0.5
        w = np.array([1.0, 2.0, 1.5, 0.5, 2.0, 1.0, 1.5, 0.5])

        # Normalize weights to sum=n
        w_norm = w * len(w) / np.sum(w)

        # Hand-compute WLS
        W = np.diag(w_norm)
        XtWX = X.T @ W @ X
        beta = np.linalg.solve(XtWX, X.T @ W @ y)
        u = y - X @ beta

        # Hand-compute weighted HC1 sandwich: (X'WX)^{-1} X'diag(w*u²)X (X'WX)^{-1}
        n, k = X.shape
        meat = X.T @ np.diag(w_norm * u**2) @ X
        bread_inv = np.linalg.inv(XtWX)
        adjustment = n / (n - k)
        vcov_expected = adjustment * bread_inv @ meat @ bread_inv

        coef, resid, vcov = solve_ols(X, y, weights=w_norm, weight_type="pweight")
        np.testing.assert_allclose(vcov, vcov_expected, atol=1e-10)
        np.testing.assert_allclose(coef, beta, atol=1e-10)


# =============================================================================
# Tier 3: Consistency and Invariance Tests
# =============================================================================


class TestConsistencyInvariance:
    """Statistical properties and invariance checks."""

    def test_fpc_monotonicity(self):
        """Increasing FPC (larger population) weakly increases SEs.

        FPC correction = 1 - n_h/N_h. As N_h grows, the correction gets
        closer to 1 (less reduction), so SEs should weakly increase.
        """
        np.random.seed(50)
        n = 60
        strata = np.repeat(np.arange(3), 20)
        psu = np.arange(n)  # each obs is own PSU
        weights = np.ones(n)

        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 1.0 + X[:, 1] * 0.5 + np.random.randn(n) * 0.3

        coef, resid, _ = solve_ols(X, y, weights=weights, weight_type="pweight")

        ses = []
        fpc_values = [25.0, 50.0, 200.0, 10000.0]
        for fpc_val in fpc_values:
            fpc_arr = np.full(n, fpc_val)
            resolved = ResolvedSurveyDesign(
                weights=weights,
                weight_type="pweight",
                strata=strata,
                psu=psu,
                fpc=fpc_arr,
                n_strata=3,
                n_psu=n,
                lonely_psu="remove",
            )
            vcov = compute_survey_vcov(X, resid, resolved)
            ses.append(np.sqrt(np.diag(vcov)))

        # Monotonically non-decreasing as FPC grows (less correction)
        for i in range(len(ses) - 1):
            assert np.all(ses[i + 1] >= ses[i] - 1e-12), (
                f"SEs did not weakly increase from FPC={fpc_values[i]} " f"to FPC={fpc_values[i+1]}"
            )

    def test_stratification_reduces_variance(self):
        """Stratification reduces variance when strata have different means."""
        np.random.seed(60)
        n = 120

        # Create data where strata have very different means
        strata = np.repeat(np.arange(3), 40)
        strata_means = np.array([0.0, 5.0, 10.0])
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = strata_means[strata] + X[:, 1] * 0.5 + np.random.randn(n) * 0.3
        psu = np.arange(n)
        weights = np.ones(n)

        coef, resid, _ = solve_ols(X, y, weights=weights, weight_type="pweight")

        # With stratification
        resolved_strat = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=strata,
            psu=psu,
            fpc=None,
            n_strata=3,
            n_psu=n,
            lonely_psu="remove",
        )
        vcov_strat = compute_survey_vcov(X, resid, resolved_strat)

        # Without stratification (single implicit stratum)
        resolved_nostrat = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=psu,
            fpc=None,
            n_strata=0,
            n_psu=n,
            lonely_psu="remove",
        )
        vcov_nostrat = compute_survey_vcov(X, resid, resolved_nostrat)

        se_strat = np.sqrt(np.diag(vcov_strat))
        se_nostrat = np.sqrt(np.diag(vcov_nostrat))

        # Stratification should reduce (or not increase) the SE of the intercept
        # because strata capture between-stratum variance
        assert se_strat[0] <= se_nostrat[0] + 1e-6

    def test_psu_equals_individual(self):
        """When each observation is its own PSU, TSL reduces to weighted HC1."""
        np.random.seed(70)
        n = 50
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 2.0 + X[:, 1] + np.random.randn(n) * 0.5
        weights = np.abs(np.random.randn(n)) + 0.5
        weights = weights * (n / np.sum(weights))
        psu = np.arange(n)  # each obs is its own PSU

        coef, resid, hc1_vcov = solve_ols(X, y, weights=weights, weight_type="pweight")

        # TSL with individual-level PSUs, no strata
        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=psu,
            fpc=None,
            n_strata=0,
            n_psu=n,
            lonely_psu="remove",
        )
        tsl_vcov = compute_survey_vcov(X, resid, resolved)

        # Both should be finite and positive-definite
        assert np.all(np.diag(tsl_vcov) > 0)
        assert np.all(np.diag(hc1_vcov) > 0)

        # Ratio should be close to 1 (not exact due to small-sample adjustments)
        ratio = np.sqrt(np.diag(tsl_vcov)) / np.sqrt(np.diag(hc1_vcov))
        np.testing.assert_allclose(ratio, 1.0, atol=0.3)

    def test_no_strata_degeneracy(self):
        """No strata + multiple PSUs: survey vcov matches hand-computed cluster-robust sandwich."""
        np.random.seed(80)
        n = 100
        n_psu = 20
        psu = np.repeat(np.arange(n_psu), 5)
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 1.0 + X[:, 1] + np.random.randn(n) * 0.5
        weights = np.ones(n)

        coef, resid, _ = solve_ols(X, y, weights=weights, weight_type="pweight")

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=psu,
            fpc=None,
            n_strata=0,
            n_psu=n_psu,
            lonely_psu="remove",
        )
        survey_vcov = compute_survey_vcov(X, resid, resolved)

        # Hand-compute cluster-robust sandwich at PSU level with weights
        # Bread: (X'WX)^{-1}
        XtWX = X.T @ (X * weights[:, np.newaxis])
        XtWX_inv = np.linalg.inv(XtWX)
        # Weighted scores: w_i * X_i * e_i
        scores = X * (weights * resid)[:, np.newaxis]
        # Aggregate to PSU level
        psu_scores = np.zeros((n_psu, X.shape[1]))
        for g in range(n_psu):
            psu_scores[g] = scores[psu == g].sum(axis=0)
        # Center and compute meat with HC1-like adjustment
        psu_mean = psu_scores.mean(axis=0, keepdims=True)
        centered = psu_scores - psu_mean
        adjustment = n_psu / (n_psu - 1)
        meat = adjustment * (centered.T @ centered)
        oracle_vcov = XtWX_inv @ meat @ XtWX_inv

        np.testing.assert_allclose(survey_vcov, oracle_vcov, atol=1e-12)

    def test_weights_only_oracle(self):
        """Weights-only design: survey vcov matches hand-computed weighted HC1."""
        np.random.seed(81)
        n = 60
        raw_weights = np.random.uniform(0.5, 3.0, n)
        weights = raw_weights * (n / np.sum(raw_weights))
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 2.0 + X[:, 1] * 0.5 + np.random.randn(n) * 0.3

        coef, resid, _ = solve_ols(X, y, weights=weights, weight_type="pweight")

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )
        survey_vcov = compute_survey_vcov(X, resid, resolved)

        # Correct weighted HC1: (X'WX)^{-1} * X' diag(w * e²) X * n/(n-k) * (X'WX)^{-1}
        k = X.shape[1]
        XtWX = X.T @ (X * weights[:, np.newaxis])
        XtWX_inv = np.linalg.inv(XtWX)
        meat = np.dot(X.T, X * (weights * resid**2)[:, np.newaxis])
        meat *= n / (n - k)
        oracle_vcov = XtWX_inv @ meat @ XtWX_inv

        np.testing.assert_allclose(survey_vcov, oracle_vcov, atol=1e-12)

    def test_fweight_expansion_equivalence(self):
        """fweight=k gives same coefficients as duplicating each row k times."""
        np.random.seed(90)
        n = 30
        X_base = np.column_stack([np.ones(n), np.random.randn(n)])
        y_base = 2.0 + X_base[:, 1] * 1.5 + np.random.randn(n) * 0.3
        freq = np.random.choice([1, 2, 3], n).astype(float)

        # WLS with fweights
        coef_fw, _, _ = solve_ols(X_base, y_base, weights=freq, weight_type="fweight")

        # Expanded dataset
        X_exp = np.repeat(X_base, freq.astype(int), axis=0)
        y_exp = np.repeat(y_base, freq.astype(int))
        coef_exp, _, _ = solve_ols(X_exp, y_exp)

        np.testing.assert_allclose(coef_fw, coef_exp, atol=1e-10)

    def test_kish_effective_n(self):
        """Verify effective N = (sum w)^2 / sum(w^2) (Kish formula)."""
        raw_weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        n = len(raw_weights)
        dummy_resolved = ResolvedSurveyDesign(
            weights=raw_weights * (n / np.sum(raw_weights)),
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )

        meta = compute_survey_metadata(dummy_resolved, raw_weights)

        expected_eff_n = np.sum(raw_weights) ** 2 / np.sum(raw_weights**2)
        np.testing.assert_allclose(meta.effective_n, expected_eff_n, atol=1e-12)

        expected_deff = n * np.sum(raw_weights**2) / np.sum(raw_weights) ** 2
        np.testing.assert_allclose(meta.design_effect, expected_deff, atol=1e-12)


# =============================================================================
# Tier 4: Integration Tests
# =============================================================================


class TestIntegration:
    """End-to-end integration with estimators."""

    def test_did_with_survey_design(self, survey_2x2_data):
        """Full pipeline: DifferenceInDifferences with SurveyDesign."""
        did = DifferenceInDifferences()
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            fpc="fpc",
            weight_type="pweight",
        )
        result = did.fit(
            survey_2x2_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            survey_design=sd,
        )

        assert isinstance(result.att, float)
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.se > 0
        assert np.isfinite(result.t_stat)
        assert np.isfinite(result.p_value)

        # ATT should be close to 3.0 (the true effect)
        assert abs(result.att - 3.0) < 1.0

        # Survey metadata should be populated
        assert result.survey_metadata is not None
        assert isinstance(result.survey_metadata, SurveyMetadata)
        assert result.survey_metadata.weight_type == "pweight"
        assert result.survey_metadata.n_strata == 5
        assert result.survey_metadata.n_psu == 20

    def test_psu_overrides_cluster(self, survey_2x2_data):
        """When both PSU and cluster specified, PSU is used with a warning."""
        did = DifferenceInDifferences(cluster="unit")
        sd = SurveyDesign(
            weights="weight",
            psu="psu",
            weight_type="pweight",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = did.fit(
                survey_2x2_data,
                outcome="outcome",
                treatment="treated",
                time="post",
                survey_design=sd,
            )
            psu_warnings = [
                x
                for x in w
                if "PSU will be used" in str(x.message) or "survey_design.psu" in str(x.message)
            ]
            assert len(psu_warnings) >= 1

        assert np.isfinite(result.att)
        assert result.se > 0

    def test_absorb_with_weights(self, survey_2x2_data):
        """Weighted demeaning produces different results from unweighted."""
        did_w = DifferenceInDifferences()
        sd = SurveyDesign(weights="weight", weight_type="pweight")
        result_w = did_w.fit(
            survey_2x2_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            absorb=["stratum"],
            survey_design=sd,
        )

        did_uw = DifferenceInDifferences()
        result_uw = did_uw.fit(
            survey_2x2_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            absorb=["stratum"],
        )

        # ATT should be similar (both near 3.0) but not identical
        assert abs(result_w.att - 3.0) < 1.5
        assert abs(result_uw.att - 3.0) < 1.5
        # SEs should differ because of weighting
        assert result_w.se != result_uw.se

    def test_summary_output(self, survey_2x2_data):
        """Survey Design block appears in summary()."""
        did = DifferenceInDifferences()
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            weight_type="pweight",
        )
        result = did.fit(
            survey_2x2_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            survey_design=sd,
        )
        summary_text = result.summary()

        assert "Survey Design" in summary_text
        assert "pweight" in summary_text
        assert "Strata:" in summary_text
        assert "PSU/Cluster:" in summary_text
        assert "Effective sample size:" in summary_text
        assert "Design effect (DEFF):" in summary_text

    def test_to_dict_survey_fields(self, survey_2x2_data):
        """Survey metadata fields appear in to_dict()."""
        did = DifferenceInDifferences()
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            weight_type="pweight",
        )
        result = did.fit(
            survey_2x2_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            survey_design=sd,
        )
        d = result.to_dict()

        assert "weight_type" in d
        assert d["weight_type"] == "pweight"
        assert "effective_n" in d
        assert d["effective_n"] > 0
        assert "design_effect" in d
        assert d["design_effect"] > 0
        assert "sum_weights" in d
        assert "n_strata" in d
        assert d["n_strata"] == 5
        assert "n_psu" in d
        assert d["n_psu"] == 20
        assert "df_survey" in d

    def test_wild_bootstrap_survey_guard(self, survey_2x2_data):
        """NotImplementedError raised when wild bootstrap + survey combined."""
        did = DifferenceInDifferences(inference="wild_bootstrap")
        sd = SurveyDesign(weights="weight", weight_type="pweight")

        with pytest.raises(NotImplementedError, match="Wild bootstrap"):
            did.fit(
                survey_2x2_data,
                outcome="outcome",
                treatment="treated",
                time="post",
                survey_design=sd,
            )

    def test_survey_design_validation_bad_weight_type(self):
        """Invalid weight_type raises ValueError."""
        with pytest.raises(ValueError, match="weight_type"):
            SurveyDesign(weights="w", weight_type="invalid")

    def test_survey_design_validation_bad_lonely_psu(self):
        """Invalid lonely_psu raises ValueError."""
        with pytest.raises(ValueError, match="lonely_psu"):
            SurveyDesign(lonely_psu="invalid")

    def test_survey_design_validation_missing_column(self):
        """Missing column in data raises ValueError."""
        df = pd.DataFrame({"y": [1, 2, 3], "w": [1.0, 1.0, 1.0]})
        sd = SurveyDesign(weights="missing_col")
        with pytest.raises(ValueError, match="not found"):
            sd.resolve(df)

    def test_survey_design_validation_zero_weights(self):
        """Zero weights raise ValueError."""
        df = pd.DataFrame({"y": [1, 2, 3], "w": [1.0, 0.0, 1.0]})
        sd = SurveyDesign(weights="w")
        with pytest.raises(ValueError, match="strictly positive"):
            sd.resolve(df)

    def test_survey_design_validation_nan_weights(self):
        """NaN weights raise ValueError."""
        df = pd.DataFrame({"y": [1, 2, 3], "w": [1.0, np.nan, 1.0]})
        sd = SurveyDesign(weights="w")
        with pytest.raises(ValueError, match="NaN"):
            sd.resolve(df)

    def test_survey_design_validation_inf_weights(self):
        """Inf weights raise ValueError."""
        df = pd.DataFrame({"y": [1, 2, 3], "w": [1.0, np.inf, 1.0]})
        sd = SurveyDesign(weights="w")
        with pytest.raises(ValueError, match="Inf"):
            sd.resolve(df)

    def test_survey_design_validation_missing_strata_column(self):
        """Missing strata column raises ValueError."""
        df = pd.DataFrame({"y": [1, 2, 3], "w": [1.0, 1.0, 1.0]})
        sd = SurveyDesign(weights="w", strata="missing")
        with pytest.raises(ValueError, match="not found"):
            sd.resolve(df)

    def test_survey_design_validation_missing_psu_column(self):
        """Missing PSU column raises ValueError."""
        df = pd.DataFrame({"y": [1, 2, 3], "w": [1.0, 1.0, 1.0]})
        sd = SurveyDesign(weights="w", psu="missing")
        with pytest.raises(ValueError, match="not found"):
            sd.resolve(df)

    def test_survey_design_validation_missing_fpc_column(self):
        """Missing FPC column raises ValueError."""
        df = pd.DataFrame({"y": [1, 2, 3], "w": [1.0, 1.0, 1.0]})
        sd = SurveyDesign(weights="w", fpc="missing")
        with pytest.raises(ValueError, match="not found"):
            sd.resolve(df)

    def test_survey_design_validation_nan_fpc(self):
        """NaN FPC values raise ValueError."""
        df = pd.DataFrame(
            {
                "y": [1, 2, 3],
                "w": [1.0, 1.0, 1.0],
                "fpc": [100.0, np.nan, 100.0],
            }
        )
        sd = SurveyDesign(weights="w", fpc="fpc")
        with pytest.raises(ValueError, match="finite"):
            sd.resolve(df)

    def test_survey_design_validation_fpc_too_small(self):
        """FPC smaller than sample size raises ValueError."""
        n = 20
        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "w": np.ones(n),
                "s": np.repeat([0, 1], 10),
                "psu": np.arange(n),
                "fpc": np.full(n, 5.0),  # FPC=5 but n_h=10 per stratum
            }
        )
        sd = SurveyDesign(weights="w", strata="s", psu="psu", fpc="fpc")
        with pytest.raises(ValueError, match="FPC"):
            sd.resolve(df)

    def test_unstratified_fpc_must_be_constant(self):
        """Row-varying FPC without strata raises ValueError."""
        n = 20
        psu = np.repeat(np.arange(4), 5)
        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "w": np.ones(n),
                "psu": psu,
                # FPC varies across rows (some 100, some 200)
                "fpc": np.where(psu < 2, 100.0, 200.0),
            }
        )
        sd = SurveyDesign(weights="w", psu="psu", fpc="fpc")
        with pytest.raises(ValueError, match="constant"):
            sd.resolve(df)

    def test_survey_design_type_error(self, survey_2x2_data):
        """Passing non-SurveyDesign object raises TypeError."""
        did = DifferenceInDifferences()
        with pytest.raises(TypeError, match="SurveyDesign"):
            did.fit(
                survey_2x2_data,
                outcome="outcome",
                treatment="treated",
                time="post",
                survey_design={"weights": "weight"},  # dict instead of SurveyDesign
            )

    def test_nest_true(self):
        """nest=True makes repeated PSU IDs unique across strata."""
        n = 40
        strata = np.repeat([0, 1], 20)
        # PSU IDs repeat across strata: 0..9 in each stratum
        psu_raw = np.tile(np.arange(10), 4)[:n]

        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "w": np.ones(n),
                "s": strata,
                "psu": psu_raw,
            }
        )

        # Without nest: PSU 0 in stratum 0 == PSU 0 in stratum 1
        sd_no_nest = SurveyDesign(weights="w", strata="s", psu="psu", nest=False)
        resolved_no_nest = sd_no_nest.resolve(df)

        # With nest: PSU 0 in stratum 0 != PSU 0 in stratum 1
        sd_nest = SurveyDesign(weights="w", strata="s", psu="psu", nest=True)
        resolved_nest = sd_nest.resolve(df)

        # nest=True should produce more unique PSUs
        assert resolved_nest.n_psu > resolved_no_nest.n_psu
        # Specifically: 10 PSUs repeated across 2 strata -> 20 unique with nest
        assert resolved_nest.n_psu == 20
        assert resolved_no_nest.n_psu == 10

    def test_twfe_with_survey_design(self, twfe_panel_data):
        """TwoWayFixedEffects accepts and uses survey_design."""
        twfe = TwoWayFixedEffects()
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            weight_type="pweight",
        )
        result = twfe.fit(
            twfe_panel_data,
            outcome="outcome",
            treatment="treated",
            time="post",
            unit="unit",
            survey_design=sd,
        )

        assert isinstance(result.att, float)
        assert np.isfinite(result.att)
        assert np.isfinite(result.se)
        assert result.se > 0

        # ATT should be close to 4.0
        assert abs(result.att - 4.0) < 1.5

        # Survey metadata present
        assert result.survey_metadata is not None
        assert result.survey_metadata.weight_type == "pweight"

    def test_multiperiod_with_survey_design(self, multiperiod_data):
        """MultiPeriodDiD accepts and uses survey_design."""
        mpd = MultiPeriodDiD()
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            weight_type="pweight",
        )
        result = mpd.fit(
            multiperiod_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5],
            survey_design=sd,
        )

        assert np.isfinite(result.avg_att)
        assert np.isfinite(result.avg_se)
        assert result.avg_se > 0

        # Average ATT should be close to 2.5
        assert abs(result.avg_att - 2.5) < 1.5

    def test_fweight_warning_for_fractional(self):
        """Fractional fweights emit a UserWarning."""
        df = pd.DataFrame(
            {
                "y": [1, 2, 3],
                "w": [1.5, 2.0, 3.0],  # 1.5 is fractional
            }
        )
        sd = SurveyDesign(weights="w", weight_type="fweight")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sd.resolve(df)
            fweight_warnings = [x for x in w if "Frequency weights" in str(x.message)]
            assert len(fweight_warnings) >= 1

    def test_lonely_psu_remove_warning(self):
        """Singleton stratum with lonely_psu='remove' emits warning."""
        n = 30
        # Stratum 2 has only 1 PSU
        strata = np.array([0] * 10 + [1] * 10 + [2] * 10)
        psu = np.array(
            list(range(5)) * 2  # stratum 0: 5 PSUs
            + list(range(5, 10)) * 2  # stratum 1: 5 PSUs
            + [10] * 10  # stratum 2: 1 PSU (singleton)
        )
        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "w": np.ones(n),
                "s": strata,
                "psu": psu,
            }
        )
        sd = SurveyDesign(weights="w", strata="s", psu="psu", lonely_psu="remove")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sd.resolve(df)
            lonely_warnings = [
                x for x in w if "only" in str(x.message).lower() and "PSU" in str(x.message)
            ]
            assert len(lonely_warnings) >= 1

    def test_aweight_type(self):
        """aweight type resolves and normalises correctly."""
        n = 40
        df = pd.DataFrame(
            {
                "y": np.random.randn(n),
                "w": np.random.uniform(0.5, 3.0, n),
            }
        )
        sd = SurveyDesign(weights="w", weight_type="aweight")
        resolved = sd.resolve(df)

        assert resolved.weight_type == "aweight"
        # aweights are normalised to mean=1
        np.testing.assert_allclose(np.mean(resolved.weights), 1.0, atol=1e-12)

    def test_survey_metadata_df_survey(self):
        """df_survey = n_psu - n_strata when both present."""
        n = 60
        strata = np.repeat(np.arange(3), 20)
        psu = np.repeat(np.arange(12), 5)
        weights = np.ones(n)
        weights_norm = weights * (n / np.sum(weights))

        resolved = ResolvedSurveyDesign(
            weights=weights_norm,
            weight_type="pweight",
            strata=strata,
            psu=psu,
            fpc=None,
            n_strata=3,
            n_psu=12,
            lonely_psu="remove",
        )
        meta = compute_survey_metadata(resolved, weights)

        assert meta.df_survey == 12 - 3  # n_psu - n_strata = 9

    def test_survey_metadata_df_survey_no_strata(self):
        """df_survey = n_psu - 1 when no strata."""
        n = 50
        psu = np.repeat(np.arange(10), 5)
        weights = np.ones(n)

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=psu,
            fpc=None,
            n_strata=0,
            n_psu=10,
            lonely_psu="remove",
        )
        meta = compute_survey_metadata(resolved, weights)

        assert meta.df_survey == 10 - 1  # n_psu - 1 = 9

    def test_survey_metadata_weight_range(self):
        """weight_range reports (min, max) of raw weights."""
        raw_w = np.array([1.0, 3.0, 5.0, 2.0, 4.0])
        n = len(raw_w)
        resolved = ResolvedSurveyDesign(
            weights=raw_w * (n / np.sum(raw_w)),
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )
        meta = compute_survey_metadata(resolved, raw_w)

        assert meta.weight_range == (1.0, 5.0)

    def test_no_weights_survey_design(self):
        """SurveyDesign without weights resolves to uniform weights."""
        n = 20
        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "s": np.repeat([0, 1], 10),
            }
        )
        sd = SurveyDesign(strata="s")  # no weights
        resolved = sd.resolve(df)

        np.testing.assert_allclose(resolved.weights, np.ones(n), atol=1e-12)


# =============================================================================
# Tier 5: Monte Carlo (slow)
# =============================================================================


class TestMonteCarlo:
    """Monte Carlo coverage tests."""

    @pytest.mark.slow
    def test_survey_coverage_monte_carlo(self, ci_params):
        """95% CI coverage should be between 92% and 98% over 500 simulations.

        DGP: weighted 2x2 DiD with known ATT = 2.0, stratified sampling.
        """
        n_sims = ci_params.bootstrap(500, min_n=199)
        true_att = 2.0
        covers = 0

        for sim in range(n_sims):
            np.random.seed(sim + 1000)
            n = 80
            treated = np.array([1] * 40 + [0] * 40)
            post = np.tile([0, 1], 40)
            strata = np.repeat(np.arange(4), 20)
            psu = np.repeat(np.arange(16), 5)
            weights = 1.0 + 0.5 * strata.astype(float)

            y = (
                5.0
                + 1.0 * treated
                + 2.0 * post
                + true_att * treated * post
                + np.random.randn(n) * 1.0
            )

            df = pd.DataFrame(
                {
                    "outcome": y,
                    "treated": treated,
                    "post": post,
                    "stratum": strata,
                    "psu": psu,
                    "weight": weights,
                }
            )

            sd = SurveyDesign(
                weights="weight",
                strata="stratum",
                psu="psu",
                weight_type="pweight",
            )
            did = DifferenceInDifferences()
            try:
                result = did.fit(
                    df,
                    outcome="outcome",
                    treatment="treated",
                    time="post",
                    survey_design=sd,
                )
                if result.conf_int[0] <= true_att <= result.conf_int[1]:
                    covers += 1
            except Exception:
                # Skip failed simulations (rank deficiency, etc.)
                n_sims -= 1
                continue

        if n_sims < 30:
            pytest.skip("Too few valid simulations for coverage test")

        coverage = covers / n_sims
        # Adjust tolerance for small n_sims (pure Python mode)
        tol_low = 0.88 if n_sims < 200 else 0.92
        tol_high = 0.99 if n_sims < 200 else 0.98
        assert tol_low <= coverage <= tol_high, (
            f"95% CI coverage = {coverage:.3f} ({covers}/{n_sims}) "
            f"outside [{tol_low}, {tol_high}]"
        )


# =============================================================================
# Tier 6: P0/P1 Fix Verification Tests
# =============================================================================


class TestP0P1Fixes:
    """Verification tests for P0 and P1 fixes."""

    def test_all_singleton_strata_nan_inference(self):
        """All-singleton strata design returns NaN inference (P0-3)."""
        np.random.seed(200)
        n = 30
        # Every stratum has exactly 1 PSU -> no within-stratum variance
        strata = np.arange(n)
        psu = np.arange(n)
        weights = np.ones(n)

        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 1.0 + X[:, 1] * 0.5 + np.random.randn(n) * 0.3

        coef, resid, _ = solve_ols(X, y, weights=weights, weight_type="pweight")

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=strata,
            psu=psu,
            fpc=None,
            n_strata=n,
            n_psu=n,
            lonely_psu="remove",
        )
        vcov = compute_survey_vcov(X, resid, resolved)

        # All singleton strata -> all removed -> NaN vcov
        assert np.all(np.isnan(vcov))

    def test_fpc_validates_against_psu_count(self):
        """FPC >= n_PSU passes validation even when n_obs > FPC (P1)."""
        n = 40
        # 2 strata, 4 PSUs per stratum, 5 obs per PSU
        strata = np.repeat([0, 1], 20)
        psu = np.repeat(np.arange(8), 5)
        # FPC = 10 (>= 4 PSUs per stratum, but < 20 obs per stratum)
        fpc_arr = np.full(n, 10.0)

        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "w": np.ones(n),
                "s": strata,
                "psu": psu,
                "fpc": fpc_arr,
            }
        )
        sd = SurveyDesign(weights="w", strata="s", psu="psu", fpc="fpc")
        # Should not raise: FPC=10 >= n_PSU=4 per stratum
        resolved = sd.resolve(df)
        assert resolved.fpc is not None

    def test_fpc_nonconstant_within_stratum(self):
        """Non-constant FPC within a stratum raises ValueError (P1)."""
        n = 20
        strata = np.repeat([0, 1], 10)
        psu = np.repeat(np.arange(4), 5)
        # FPC varies within stratum 0
        fpc_arr = np.array([100.0] * 5 + [200.0] * 5 + [150.0] * 10)

        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "w": np.ones(n),
                "s": strata,
                "psu": psu,
                "fpc": fpc_arr,
            }
        )
        sd = SurveyDesign(weights="w", strata="s", psu="psu", fpc="fpc")
        with pytest.raises(ValueError, match="constant within each stratum"):
            sd.resolve(df)

    def test_fpc_only_without_psu_raises(self):
        """FPC without psu or strata raises ValueError (P1)."""
        n = 10
        df = pd.DataFrame(
            {
                "y": np.ones(n),
                "w": np.ones(n),
                "fpc": np.full(n, 100.0),
            }
        )
        sd = SurveyDesign(weights="w", fpc="fpc")
        with pytest.raises(ValueError, match="FPC requires either psu or strata"):
            sd.resolve(df)

    def test_weighted_within_transform_matches_explicit_wls(self):
        """Weighted within_transform + OLS matches explicit WLS with dummies (P0-2).

        For a balanced panel with 2-way FE, the within-transformed WLS
        should yield the same treatment coefficient as WLS with explicit
        unit + time dummies.
        """
        np.random.seed(300)
        n_units = 10
        n_periods = 3
        rows = []
        for u in range(n_units):
            for t in range(n_periods):
                treated = 1 if u < 5 else 0
                post = 1 if t >= 2 else 0
                y = 5.0 + u * 0.1 + t * 0.5 + 3.0 * treated * post
                y += np.random.randn() * 0.1
                rows.append(
                    {
                        "unit": u,
                        "time": t,
                        "treated": treated,
                        "post": post,
                        "outcome": y,
                    }
                )
        df = pd.DataFrame(rows)
        weights = np.array([1.0 + 0.5 * (i % 3) for i in range(len(df))])
        weights = weights * len(df) / np.sum(weights)

        # Method 1: Within-transform + OLS on treatment*post
        df["treat_post"] = df["treated"] * df["post"]
        df_wt = within_transform(df, ["outcome", "treat_post"], "unit", "time", weights=weights)
        y_wt = df_wt["outcome_demeaned"].values
        X_wt = np.column_stack([np.ones(len(df)), df_wt["treat_post_demeaned"].values])
        coef_wt, _, _ = solve_ols(X_wt, y_wt, weights=weights, weight_type="pweight")

        # Method 2: Explicit WLS with unit + time dummies
        unit_dummies = pd.get_dummies(df["unit"], prefix="u", drop_first=True)
        time_dummies = pd.get_dummies(df["time"], prefix="t", drop_first=True)
        X_full = np.column_stack(
            [
                np.ones(len(df)),
                df["treat_post"].values,
                unit_dummies.values,
                time_dummies.values,
            ]
        )
        y_full = df["outcome"].values
        coef_full, _, _ = solve_ols(X_full, y_full, weights=weights, weight_type="pweight")

        # Treatment coefficient should match
        np.testing.assert_allclose(coef_wt[1], coef_full[1], atol=1e-6)

    def test_multiperiod_survey_metadata_populated(self, multiperiod_data):
        """MultiPeriodDiD populates survey_metadata in results (P2-1)."""
        mpd = MultiPeriodDiD()
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            weight_type="pweight",
        )
        result = mpd.fit(
            multiperiod_data,
            outcome="outcome",
            treatment="treated",
            time="period",
            post_periods=[3, 4, 5],
            survey_design=sd,
        )

        assert result.survey_metadata is not None
        assert isinstance(result.survey_metadata, SurveyMetadata)
        assert result.survey_metadata.weight_type == "pweight"
        assert result.survey_metadata.n_strata == 3
        assert result.survey_metadata.n_psu == 20

        # Survey info should appear in summary
        summary_text = result.summary()
        assert "Survey Design" in summary_text
        assert "pweight" in summary_text

        # Survey info should appear in to_dict
        d = result.to_dict()
        assert "weight_type" in d
        assert d["weight_type"] == "pweight"
        assert "effective_n" in d


# =============================================================================
# P1-1 / P1-2 regression tests (PR #218 review round 3)
# =============================================================================


class TestFweightInference:
    """Verify fweight df = sum(w) - k throughout the stack."""

    def test_fweight_se_matches_expanded_oracle(self):
        """Fweight SEs must match unweighted OLS on frequency-expanded data."""
        np.random.seed(42)
        n = 30
        x1 = np.random.randn(n)
        X = np.column_stack([np.ones(n), x1])
        y = 2.0 + 1.5 * x1 + np.random.randn(n) * 0.5
        fw = np.random.choice([1, 2, 3], size=n)

        # Oracle: expand each row fw[i] times, run unweighted OLS
        X_exp = np.repeat(X, fw, axis=0)
        y_exp = np.repeat(y, fw)
        coef_exp, _, vcov_exp = solve_ols(X_exp, y_exp)
        se_exp = np.sqrt(np.diag(vcov_exp))

        # Fweight path: compressed data with integer weights
        coef_fw, _, vcov_fw = solve_ols(
            X, y, weights=fw.astype(float), weight_type="fweight"
        )
        se_fw = np.sqrt(np.diag(vcov_fw))

        np.testing.assert_allclose(coef_fw, coef_exp, atol=1e-10)
        np.testing.assert_allclose(se_fw, se_exp, atol=1e-10)

    def test_linear_regression_fweight_df(self):
        """LinearRegression with fweight must set df_ = sum(w) - k."""
        np.random.seed(42)
        n = 30
        x1 = np.random.randn(n)
        X = np.column_stack([np.ones(n), x1])
        y = 2.0 + 1.5 * x1 + np.random.randn(n) * 0.5
        fw = np.random.choice([1, 2, 3], size=n)

        model = LinearRegression(
            weights=fw.astype(float),
            weight_type="fweight",
            include_intercept=False,
        )
        model.fit(X, y)

        k = X.shape[1]
        expected_df = int(np.sum(fw)) - k
        assert model.df_ == expected_df


class TestWeightedRankDeficiency:
    """Weighted rank-deficient fits must not produce all-NaN residuals."""

    def test_weighted_rank_deficient_solver_finite_residuals(self):
        """solve_ols with weights + duplicate column: residuals must be finite."""
        np.random.seed(42)
        n = 50
        x1 = np.random.randn(n)
        # Duplicate column -> rank-deficient
        X = np.column_stack([np.ones(n), x1, x1])
        y = 2.0 + 1.5 * x1 + np.random.randn(n) * 0.3
        pw = np.random.uniform(0.5, 3.0, size=n)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            coef, resid, vcov = solve_ols(
                X,
                y,
                weights=pw,
                weight_type="pweight",
                rank_deficient_action="warn",
            )

        # Exactly one coefficient should be NaN (dropped duplicate)
        assert np.sum(np.isnan(coef)) == 1

        # Residuals must all be finite (not all-NaN)
        assert np.all(np.isfinite(resid)), "Residuals must not be all-NaN"

        # Vcov: NaN rows/cols only for the dropped column
        nan_col = np.where(np.isnan(coef))[0][0]
        assert np.all(np.isnan(vcov[nan_col, :])), "Dropped col vcov row should be NaN"
        assert np.all(np.isnan(vcov[:, nan_col])), "Dropped col vcov col should be NaN"

        # Identified coefficients should have positive, finite SEs
        kept = np.where(~np.isnan(coef))[0]
        for i in kept:
            assert np.isfinite(vcov[i, i]) and vcov[i, i] > 0

    def test_linear_regression_weighted_rank_deficient_robust(self):
        """LinearRegression with weights + robust + rank deficiency: finite residuals."""
        np.random.seed(42)
        n = 50
        x1 = np.random.randn(n)
        X = np.column_stack([np.ones(n), x1, x1])
        y = 2.0 + 1.5 * x1 + np.random.randn(n) * 0.3
        pw = np.random.uniform(0.5, 3.0, size=n)

        model = LinearRegression(
            weights=pw,
            weight_type="pweight",
            robust=True,
            include_intercept=False,
            rank_deficient_action="warn",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(X, y)

        coef = model.coefficients_
        resid = model.residuals_
        vcov = model.vcov_

        # One dropped coefficient
        assert np.sum(np.isnan(coef)) == 1

        # Residuals all finite
        assert np.all(np.isfinite(resid))

        # Identified coefficients have positive, finite SEs
        kept = np.where(~np.isnan(coef))[0]
        for i in kept:
            assert np.isfinite(vcov[i, i]) and vcov[i, i] > 0

    def test_fweight_survey_oracle(self):
        """fweight SurveyDesign: survey vcov matches expanded-data unweighted HC1."""
        np.random.seed(55)
        n = 30
        X_base = np.column_stack([np.ones(n), np.random.randn(n)])
        y_base = 2.0 + X_base[:, 1] * 1.5 + np.random.randn(n) * 0.3
        freq = np.random.choice([1, 2, 3], n).astype(float)

        # WLS with fweights via survey
        coef_fw, resid_fw, _ = solve_ols(
            X_base, y_base, weights=freq, weight_type="fweight"
        )
        resolved = ResolvedSurveyDesign(
            weights=freq,
            weight_type="fweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )
        survey_vcov = compute_survey_vcov(X_base, resid_fw, resolved)

        # Oracle: expand data and compute unweighted HC1
        X_exp = np.repeat(X_base, freq.astype(int), axis=0)
        y_exp = np.repeat(y_base, freq.astype(int))
        coef_exp, resid_exp, _ = solve_ols(X_exp, y_exp)
        n_exp = X_exp.shape[0]
        k = X_exp.shape[1]
        XtX = X_exp.T @ X_exp
        XtX_inv = np.linalg.inv(XtX)
        meat = np.dot(X_exp.T, X_exp * (resid_exp**2)[:, np.newaxis])
        meat *= n_exp / (n_exp - k)
        oracle_vcov = XtX_inv @ meat @ XtX_inv

        np.testing.assert_allclose(survey_vcov, oracle_vcov, atol=1e-10)

    def test_survey_rank_deficient_with_psu(self):
        """LinearRegression + survey design (PSU) + rank deficiency: no crash."""
        np.random.seed(43)
        n = 50
        x1 = np.random.randn(n)
        X = np.column_stack([np.ones(n), x1, x1])  # duplicate col
        y = 2.0 + 1.5 * x1 + np.random.randn(n) * 0.3
        pw = np.random.uniform(0.5, 3.0, size=n)
        psu = np.arange(n)  # each obs is its own PSU

        resolved = ResolvedSurveyDesign(
            weights=pw,
            weight_type="pweight",
            strata=None,
            psu=psu,
            fpc=None,
            n_strata=0,
            n_psu=n,
            lonely_psu="remove",
        )

        model = LinearRegression(
            survey_design=resolved,
            include_intercept=False,
            rank_deficient_action="warn",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(X, y)

        coef = model.coefficients_
        resid = model.residuals_
        vcov = model.vcov_

        # One dropped coefficient
        assert np.sum(np.isnan(coef)) == 1

        # Residuals all finite
        assert np.all(np.isfinite(resid))

        # Identified coefficients have positive, finite SEs
        kept = np.where(~np.isnan(coef))[0]
        for i in kept:
            assert np.isfinite(vcov[i, i]) and vcov[i, i] > 0

        # Dropped column has NaN vcov
        dropped = np.where(np.isnan(coef))[0]
        for i in dropped:
            assert np.all(np.isnan(vcov[i, :]))
            assert np.all(np.isnan(vcov[:, i]))

    def test_survey_rank_deficient_weights_only(self):
        """Weights-only survey + rank deficiency: no crash, correct NaN pattern."""
        np.random.seed(44)
        n = 50
        x1 = np.random.randn(n)
        X = np.column_stack([np.ones(n), x1, x1])  # duplicate col
        y = 2.0 + 1.5 * x1 + np.random.randn(n) * 0.3
        pw = np.random.uniform(0.5, 3.0, size=n)

        resolved = ResolvedSurveyDesign(
            weights=pw,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )

        model = LinearRegression(
            survey_design=resolved,
            include_intercept=False,
            rank_deficient_action="warn",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(X, y)

        coef = model.coefficients_
        resid = model.residuals_
        vcov = model.vcov_

        # One dropped coefficient
        assert np.sum(np.isnan(coef)) == 1

        # Residuals all finite
        assert np.all(np.isfinite(resid))

        # Identified coefficients have positive, finite SEs
        kept = np.where(~np.isnan(coef))[0]
        for i in kept:
            assert np.isfinite(vcov[i, i]) and vcov[i, i] > 0

        # Dropped column has NaN vcov
        dropped = np.where(np.isnan(coef))[0]
        for i in dropped:
            assert np.all(np.isnan(vcov[i, :]))
            assert np.all(np.isnan(vcov[:, i]))

    def test_linear_regression_weighted_rank_deficient_classical(self):
        """LinearRegression with weights + classical vcov + rank deficiency."""
        np.random.seed(42)
        n = 50
        x1 = np.random.randn(n)
        X = np.column_stack([np.ones(n), x1, x1])
        y = 2.0 + 1.5 * x1 + np.random.randn(n) * 0.3
        pw = np.random.uniform(0.5, 3.0, size=n)

        model = LinearRegression(
            weights=pw,
            weight_type="pweight",
            robust=False,
            include_intercept=False,
            rank_deficient_action="warn",
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model.fit(X, y)

        coef = model.coefficients_
        resid = model.residuals_
        vcov = model.vcov_

        # One dropped coefficient
        assert np.sum(np.isnan(coef)) == 1

        # Residuals all finite
        assert np.all(np.isfinite(resid))

        # Identified coefficients have positive, finite SEs
        kept = np.where(~np.isnan(coef))[0]
        for i in kept:
            assert np.isfinite(vcov[i, i]) and vcov[i, i] > 0


# =============================================================================
# Round 5 Fixes (PR #218)
# =============================================================================


class TestRound5Fixes:
    """Tests for P1-A (implicit PSU df), P1-B (lonely_psu unstratified),
    P1-C (LinearRegression weight auto-derivation)."""

    def test_weights_only_survey_df(self):
        """P1-A: weights-only design uses n_obs - 1 for df."""
        n = 50
        weights = np.random.uniform(0.5, 3.0, size=n)
        weights = weights * (n / np.sum(weights))  # normalize

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=None,
            fpc=None,
            n_strata=0,
            n_psu=0,
            lonely_psu="remove",
        )

        assert resolved.df_survey == n - 1

        # Metadata should also report implicit n_psu
        raw_w = np.random.uniform(0.5, 3.0, size=n)
        meta = compute_survey_metadata(resolved, raw_w)
        assert meta.n_psu == n

    def test_stratified_no_psu_survey_df(self):
        """P1-A: stratified-no-PSU design uses n_obs - n_strata for df."""
        n = 60
        n_strata = 3
        weights = np.ones(n, dtype=np.float64)
        strata = np.repeat(np.arange(n_strata), n // n_strata)

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=strata,
            psu=None,
            fpc=None,
            n_strata=n_strata,
            n_psu=0,
            lonely_psu="remove",
        )

        assert resolved.df_survey == n - n_strata

    def test_single_psu_unstratified_lonely_psu_remove(self):
        """P1-B: single PSU unstratified, lonely_psu='remove' -> NaN vcov."""
        np.random.seed(42)
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 1.0 + 0.5 * X[:, 1] + np.random.randn(n) * 0.3
        weights = np.ones(n, dtype=np.float64)
        residuals = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=np.zeros(n, dtype=int),  # all same PSU
            fpc=None,
            n_strata=0,
            n_psu=1,
            lonely_psu="remove",
        )

        vcov = compute_survey_vcov(X, residuals, resolved)
        assert np.all(np.isnan(vcov))

    def test_single_psu_unstratified_lonely_psu_certainty(self):
        """P1-B: single PSU unstratified, lonely_psu='certainty' -> NaN vcov."""
        np.random.seed(42)
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 1.0 + 0.5 * X[:, 1] + np.random.randn(n) * 0.3
        weights = np.ones(n, dtype=np.float64)
        residuals = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=np.zeros(n, dtype=int),
            fpc=None,
            n_strata=0,
            n_psu=1,
            lonely_psu="certainty",
        )

        vcov = compute_survey_vcov(X, residuals, resolved)
        assert np.all(np.isnan(vcov))

    def test_single_psu_unstratified_lonely_psu_adjust(self):
        """P1-B: single PSU unstratified, lonely_psu='adjust' -> NaN vcov."""
        np.random.seed(42)
        n = 20
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        y = 1.0 + 0.5 * X[:, 1] + np.random.randn(n) * 0.3
        weights = np.ones(n, dtype=np.float64)
        residuals = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]

        resolved = ResolvedSurveyDesign(
            weights=weights,
            weight_type="pweight",
            strata=None,
            psu=np.zeros(n, dtype=int),
            fpc=None,
            n_strata=0,
            n_psu=1,
            lonely_psu="adjust",
        )

        vcov = compute_survey_vcov(X, residuals, resolved)
        assert np.all(np.isnan(vcov))

    def test_linear_regression_auto_derives_weights_from_survey(self):
        """P1-C: LinearRegression auto-derives weights from survey_design."""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        y = 2.0 + 1.5 * x1 + np.random.randn(n) * 0.5
        weights = np.random.uniform(0.5, 3.0, size=n)
        weights_norm = weights * (n / np.sum(weights))

        psu = np.repeat(np.arange(10), n // 10)

        resolved = ResolvedSurveyDesign(
            weights=weights_norm,
            weight_type="pweight",
            strata=None,
            psu=psu,
            fpc=None,
            n_strata=0,
            n_psu=10,
            lonely_psu="remove",
        )

        # Explicit weights path
        model_explicit = LinearRegression(
            weights=weights_norm,
            weight_type="pweight",
            robust=True,
            survey_design=resolved,
        )
        X = np.column_stack([np.ones(n), x1])
        model_explicit.fit(X, y)

        # Auto-derive path: no explicit weights
        model_auto = LinearRegression(
            robust=True,
            survey_design=resolved,
        )
        model_auto.fit(X, y)

        # Weights should be populated after fit
        assert model_auto.weights is not None

        # Coefficients should match
        np.testing.assert_allclose(
            model_auto.coefficients_, model_explicit.coefficients_, rtol=1e-10
        )

        # Vcov should match
        np.testing.assert_allclose(
            model_auto.vcov_, model_explicit.vcov_, rtol=1e-10
        )

    def test_resolve_warns_single_psu_unstratified(self):
        """P1-B: SurveyDesign.resolve() warns for single PSU unstratified."""
        n = 10
        df = pd.DataFrame(
            {
                "w": np.ones(n),
                "psu_col": np.zeros(n, dtype=int),
                "y": np.random.randn(n),
            }
        )

        for mode in ["remove", "certainty", "adjust"]:
            design = SurveyDesign(
                weights="w",
                psu="psu_col",
                lonely_psu=mode,
            )
            with pytest.warns(UserWarning, match=r"Only 1 PSU"):
                design.resolve(df)
