"""
Methodology tests for Synthetic Difference-in-Differences (SDID).

Tests verify the implementation matches Arkhangelsky et al. (2021) and
R's synthdid package behavior: Frank-Wolfe solver, collapsed form,
auto-regularization, sparsification, and correct bootstrap/placebo SE.
"""

import warnings
from unittest.mock import patch

import numpy as np
import pytest

import pandas as pd

from diff_diff.synthetic_did import SyntheticDiD
from diff_diff.utils import (
    _compute_noise_level,
    _compute_noise_level_numpy,
    _compute_regularization,
    _fw_step,
    _sc_weight_fw,
    _sparsify,
    _sum_normalize,
    compute_sdid_estimator,
    compute_sdid_unit_weights,
    compute_time_weights,
)


# =============================================================================
# Test Helpers
# =============================================================================


def _make_panel(n_control=20, n_treated=3, n_pre=5, n_post=3,
                att=5.0, seed=42):
    """Create a simple panel dataset for testing."""
    rng = np.random.default_rng(seed)
    data = []
    for unit in range(n_control + n_treated):
        is_treated = unit >= n_control
        unit_fe = rng.normal(0, 2)
        for t in range(n_pre + n_post):
            y = 10.0 + unit_fe + t * 0.3 + rng.normal(0, 0.5)
            if is_treated and t >= n_pre:
                y += att
            data.append({
                "unit": unit,
                "period": t,
                "treated": int(is_treated),
                "outcome": y,
            })
    import pandas as pd
    return pd.DataFrame(data)


# =============================================================================
# Phase A: Noise Level and Regularization
# =============================================================================


class TestNoiseLevel:
    """Verify _compute_noise_level matches hand-computed first-diff sd."""

    def test_known_values(self):
        """Test with a simple matrix where first-diffs are known."""
        # 3 time periods, 2 control units
        # Unit 0: [1, 3, 6] -> diffs: [2, 3]
        # Unit 1: [2, 2, 5] -> diffs: [0, 3]
        # All diffs: [2, 3, 0, 3], sd(ddof=1) = std([2,3,0,3], ddof=1)
        Y = np.array([[1.0, 2.0],
                       [3.0, 2.0],
                       [6.0, 5.0]])
        expected = np.std([2.0, 3.0, 0.0, 3.0], ddof=1)
        result = _compute_noise_level(Y)
        assert abs(result - expected) < 1e-10

    def test_single_period(self):
        """Single period -> no diffs possible -> noise level = 0."""
        Y = np.array([[1.0, 2.0, 3.0]])
        assert _compute_noise_level(Y) == 0.0

    def test_two_periods(self):
        """Two periods -> one diff per unit."""
        Y = np.array([[1.0, 4.0],
                       [3.0, 7.0]])
        # Diffs: [2.0, 3.0], sd(ddof=1)
        expected = np.std([2.0, 3.0], ddof=1)
        assert abs(_compute_noise_level(Y) - expected) < 1e-10


class TestRegularization:
    """Verify _compute_regularization formula with known inputs."""

    def test_formula(self):
        """Check zeta_omega = (N1*T1)^0.25 * sigma, zeta_lambda = 1e-6 * sigma."""
        # Use a simple Y_pre_control where sigma is easy to compute
        Y = np.array([[1.0, 2.0],
                       [3.0, 4.0],
                       [6.0, 7.0]])
        sigma = _compute_noise_level(Y)
        n_treated, n_post = 2, 3

        zeta_omega, zeta_lambda = _compute_regularization(Y, n_treated, n_post)

        expected_omega = (n_treated * n_post) ** 0.25 * sigma
        expected_lambda = 1e-6 * sigma

        assert abs(zeta_omega - expected_omega) < 1e-10
        assert abs(zeta_lambda - expected_lambda) < 1e-15

    def test_zero_noise(self):
        """Constant outcomes -> zero noise -> zero regularization."""
        Y = np.array([[5.0, 5.0],
                       [5.0, 5.0],
                       [5.0, 5.0]])
        zo, zl = _compute_regularization(Y, 2, 3)
        assert zo == 0.0
        assert zl == 0.0


# =============================================================================
# Phase B: Frank-Wolfe Solver
# =============================================================================


class TestFrankWolfe:
    """Verify Frank-Wolfe step and solver behavior."""

    def test_fw_step_descent(self):
        """A single FW step should not increase the half-gradient objective.

        The FW step minimizes the linearized objective at the current point.
        After the step with exact line search, the true objective should
        not increase when measured correctly.
        """
        rng = np.random.default_rng(42)
        N, T0 = 10, 5
        A = rng.standard_normal((N, T0))
        b = rng.standard_normal(N)
        eta = N * 0.1 ** 2  # eta = N * zeta^2 matching R's formulation

        x = np.ones(T0) / T0

        # Run a few steps to get away from the initial uniform point
        # (first step from uniform can have numerical issues)
        for _ in range(5):
            x = _fw_step(A, x, b, eta)

        def objective(lam):
            err = A @ lam - b
            return (eta / N) * np.sum(lam**2) + np.sum(err**2) / N

        obj_before = objective(x)
        x_new = _fw_step(A, x, b, eta)
        obj_after = objective(x_new)

        assert obj_after <= obj_before + 1e-8

    def test_fw_step_on_simplex(self):
        """FW step should return a vector on the simplex."""
        rng = np.random.default_rng(42)
        A = rng.standard_normal((8, 4))
        b = rng.standard_normal(8)
        x = np.array([0.25, 0.25, 0.25, 0.25])

        x_new = _fw_step(A, x, b, eta=1.0)
        assert np.all(x_new >= -1e-10)
        assert abs(np.sum(x_new) - 1.0) < 1e-10

    def test_sc_weight_fw_converges(self):
        """Full FW solver should converge on a known QP."""
        rng = np.random.default_rng(42)
        N, T0 = 15, 6
        Y = rng.standard_normal((N, T0 + 1))  # last col is target

        lam = _sc_weight_fw(Y, zeta=0.1, max_iter=1000)
        assert np.all(lam >= -1e-10)
        assert abs(np.sum(lam) - 1.0) < 1e-6

    def test_sc_weight_fw_max_iter_zero(self):
        """max_iter=0 should return initial uniform weights."""
        Y = np.random.randn(5, 4)  # (N, T0+1) with T0=3
        lam = _sc_weight_fw(Y, zeta=0.1, max_iter=0)
        expected = np.ones(3) / 3
        np.testing.assert_allclose(lam, expected, atol=1e-10)

    def test_sc_weight_fw_max_iter_one(self):
        """max_iter=1 should return weights after one step (still on simplex)."""
        rng = np.random.default_rng(99)
        Y = rng.standard_normal((8, 5))
        lam = _sc_weight_fw(Y, zeta=0.1, max_iter=1)
        assert np.all(lam >= -1e-10)
        assert abs(np.sum(lam) - 1.0) < 1e-10

    def test_intercept_centering(self):
        """With intercept=True, column-centering should occur."""
        rng = np.random.default_rng(42)
        Y = rng.standard_normal((10, 5)) + 100  # large offset
        lam_intercept = _sc_weight_fw(Y, zeta=0.1, intercept=True)
        lam_no_intercept = _sc_weight_fw(Y, zeta=0.1, intercept=False)
        # Both should be on simplex but may differ
        assert abs(np.sum(lam_intercept) - 1.0) < 1e-6
        assert abs(np.sum(lam_no_intercept) - 1.0) < 1e-6
        # They should be different because centering matters
        assert not np.allclose(lam_intercept, lam_no_intercept, atol=1e-3)


class TestSparsify:
    """Verify sparsification behavior."""

    def test_basic(self):
        """Weights below max/4 should be zeroed."""
        v = np.array([0.8, 0.1, 0.05, 0.05])
        result = _sparsify(v)
        assert result[0] > 0
        # 0.1 < 0.8/4 = 0.2, so should be zeroed
        assert result[1] == 0.0
        assert result[2] == 0.0
        assert result[3] == 0.0
        assert abs(np.sum(result) - 1.0) < 1e-10

    def test_all_zero(self):
        """All-zero input should return uniform weights."""
        v = np.zeros(5)
        result = _sparsify(v)
        np.testing.assert_allclose(result, np.ones(5) / 5)

    def test_single_nonzero(self):
        """Single nonzero element -> that element becomes 1.0."""
        v = np.array([0.0, 0.0, 0.5, 0.0])
        result = _sparsify(v)
        assert result[2] == 1.0
        assert np.sum(result) == 1.0

    def test_equal_weights(self):
        """Equal weights: all equal max, so max/4 threshold keeps them all."""
        v = np.array([0.25, 0.25, 0.25, 0.25])
        result = _sparsify(v)
        # 0.25 > 0.25/4 = 0.0625, so all kept
        np.testing.assert_allclose(result, v, atol=1e-10)


class TestSumNormalize:
    """Verify _sum_normalize helper."""

    def test_basic(self):
        v = np.array([2.0, 3.0, 5.0])
        result = _sum_normalize(v)
        np.testing.assert_allclose(result, [0.2, 0.3, 0.5])

    def test_zero_sum(self):
        """Zero-sum vector -> uniform weights."""
        v = np.array([0.0, 0.0, 0.0])
        result = _sum_normalize(v)
        np.testing.assert_allclose(result, [1.0/3, 1.0/3, 1.0/3])


# =============================================================================
# Phase C/D: Unit and Time Weights
# =============================================================================


class TestUnitWeights:
    """Verify compute_sdid_unit_weights behavior."""

    def test_simplex_constraint(self):
        """Weights should be on the simplex (sum to 1, non-negative)."""
        rng = np.random.default_rng(42)
        Y_pre = rng.standard_normal((8, 15))
        Y_pre_treated_mean = rng.standard_normal(8)

        omega = compute_sdid_unit_weights(Y_pre, Y_pre_treated_mean, zeta_omega=1.0)

        assert np.all(omega >= -1e-10)
        assert abs(np.sum(omega) - 1.0) < 1e-6

    def test_single_control(self):
        """Single control unit -> weight = [1.0]."""
        Y_pre = np.array([[1.0], [2.0], [3.0]])
        Y_pre_treated_mean = np.array([1.5, 2.5, 3.5])

        omega = compute_sdid_unit_weights(Y_pre, Y_pre_treated_mean, zeta_omega=1.0)

        assert len(omega) == 1
        assert abs(omega[0] - 1.0) < 1e-10

    def test_empty_control(self):
        """No control units -> empty array."""
        Y_pre = np.zeros((5, 0))
        Y_pre_treated_mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        omega = compute_sdid_unit_weights(Y_pre, Y_pre_treated_mean, zeta_omega=1.0)

        assert len(omega) == 0

    def test_sparsification_occurs(self):
        """With enough controls, some weights should be zeroed by sparsification."""
        rng = np.random.default_rng(42)
        # Many controls with varied patterns — expect some to get zeroed
        Y_pre = rng.standard_normal((10, 50))
        Y_pre_treated_mean = rng.standard_normal(10)

        omega = compute_sdid_unit_weights(Y_pre, Y_pre_treated_mean, zeta_omega=0.5)

        # At least some weights should be exactly zero after sparsification
        assert np.sum(omega == 0) > 0


class TestTimeWeights:
    """Verify compute_time_weights behavior."""

    def test_simplex_constraint(self):
        """Weights should be on the simplex."""
        rng = np.random.default_rng(42)
        Y_pre = rng.standard_normal((8, 15))
        Y_post = rng.standard_normal((3, 15))

        lam = compute_time_weights(Y_pre, Y_post, zeta_lambda=0.01)

        assert np.all(lam >= -1e-10)
        assert abs(np.sum(lam) - 1.0) < 1e-6

    def test_single_pre_period(self):
        """Single pre-period -> weight = [1.0]."""
        Y_pre = np.array([[1.0, 2.0, 3.0]])
        Y_post = np.array([[4.0, 5.0, 6.0]])

        lam = compute_time_weights(Y_pre, Y_post, zeta_lambda=0.01)

        assert len(lam) == 1
        assert abs(lam[0] - 1.0) < 1e-10

    def test_collapsed_form_correctness(self):
        """Verify the collapsed form matrix is built correctly.

        The time weight optimization solves on (N_co, T_pre+1) where the
        last column is the per-control post-period mean.
        """
        rng = np.random.default_rng(42)
        Y_pre = rng.standard_normal((4, 3))  # 4 pre-periods, 3 controls
        Y_post = rng.standard_normal((2, 3))  # 2 post-periods, 3 controls

        lam = compute_time_weights(Y_pre, Y_post, zeta_lambda=0.01)

        # Should have 4 weights (one per pre-period)
        assert len(lam) == 4
        assert abs(np.sum(lam) - 1.0) < 1e-6


# =============================================================================
# Full Pipeline
# =============================================================================


class TestATTFullPipeline:
    """Test full SDID estimation pipeline."""

    def test_estimation_produces_reasonable_att(self, ci_params):
        """Full estimation on canonical data should produce reasonable ATT."""
        df = _make_panel(n_control=20, n_treated=3, n_pre=6, n_post=3,
                         att=5.0, seed=42)
        n_boot = ci_params.bootstrap(50)
        sdid = SyntheticDiD(n_bootstrap=n_boot, seed=42, variance_method="placebo")
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(6, 9)),
        )

        # ATT should be positive and reasonably close to true value
        assert results.att > 0
        assert abs(results.att - 5.0) < 3.0

    def test_results_have_regularization_info(self):
        """Results should include noise_level, zeta_omega, zeta_lambda."""
        df = _make_panel(seed=42)
        sdid = SyntheticDiD(variance_method="placebo", seed=42)
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
        )

        assert results.noise_level is not None
        assert results.noise_level >= 0
        assert results.zeta_omega is not None
        assert results.zeta_omega >= 0
        assert results.zeta_lambda is not None
        assert results.zeta_lambda >= 0

    def test_user_override_regularization(self):
        """User-specified zeta_omega/zeta_lambda should be used instead of auto."""
        df = _make_panel(seed=42)
        sdid = SyntheticDiD(
            zeta_omega=99.0, zeta_lambda=0.5,
            variance_method="placebo", seed=42
        )
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
        )

        assert results.zeta_omega == 99.0
        assert results.zeta_lambda == 0.5


# =============================================================================
# Placebo SE
# =============================================================================


class TestPlaceboSE:
    """Verify placebo variance formula."""

    def test_placebo_se_formula(self):
        """SE should be sqrt((r-1)/r) * sd(estimates, ddof=1)."""
        df = _make_panel(n_control=15, n_treated=2, seed=42)
        sdid = SyntheticDiD(
            variance_method="placebo", n_bootstrap=100, seed=42
        )
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
        )

        assert results.se > 0
        assert results.variance_method == "placebo"

        # Verify the formula: se = sqrt((r-1)/r) * sd(placebo_estimates)
        if results.placebo_effects is not None:
            r = len(results.placebo_effects)
            expected_se = np.sqrt((r - 1) / r) * np.std(results.placebo_effects, ddof=1)
            assert abs(results.se - expected_se) < 1e-10


# =============================================================================
# Bootstrap SE
# =============================================================================


class TestBootstrapSE:
    """Verify bootstrap SE with fixed weights."""

    def test_bootstrap_se_positive(self, ci_params):
        """Bootstrap SE should be positive."""
        df = _make_panel(n_control=20, n_treated=3, seed=42)
        n_boot = ci_params.bootstrap(50)
        sdid = SyntheticDiD(
            variance_method="bootstrap", n_bootstrap=n_boot, seed=42
        )
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
        )

        assert results.se > 0
        assert results.variance_method == "bootstrap"

    def test_bootstrap_with_zeta_overrides(self, ci_params):
        """Bootstrap SE should work with user-specified zeta overrides."""
        df = _make_panel(n_control=20, n_treated=3, seed=42)
        n_boot = ci_params.bootstrap(50)
        sdid = SyntheticDiD(
            variance_method="bootstrap", n_bootstrap=n_boot,
            zeta_omega=99.0, zeta_lambda=0.5, seed=42,
        )
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
        )

        assert results.zeta_omega == 99.0
        assert results.zeta_lambda == 0.5
        assert results.variance_method == "bootstrap"
        assert results.se > 0


# =============================================================================
# Jackknife SE
# =============================================================================


class TestJackknifeSE:
    """Verify jackknife SE with fixed weights (Algorithm 3)."""

    def test_jackknife_se_positive(self):
        """Jackknife SE should be positive for well-specified data."""
        df = _make_panel(n_control=20, n_treated=3, seed=42)
        sdid = SyntheticDiD(variance_method="jackknife", seed=42)
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
        )
        assert results.se > 0
        assert results.variance_method == "jackknife"

    def test_jackknife_deterministic(self):
        """Jackknife should produce identical results regardless of seed."""
        df = _make_panel(n_control=15, n_treated=3, seed=42)
        results1 = SyntheticDiD(variance_method="jackknife", seed=1).fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
        )
        results2 = SyntheticDiD(variance_method="jackknife", seed=999).fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
        )
        # ATT should be identical (same weights, no randomness in point est)
        assert results1.att == results2.att
        # SE should be identical (jackknife is deterministic)
        assert results1.se == results2.se

    def test_jackknife_se_formula(self):
        """Verify SE matches sqrt((n-1)/n * sum((u - ubar)^2))."""
        df = _make_panel(n_control=15, n_treated=3, seed=42)
        sdid = SyntheticDiD(variance_method="jackknife", seed=42)
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
        )
        assert results.placebo_effects is not None
        u = results.placebo_effects
        n = len(u)
        u_bar = np.mean(u)
        expected_se = np.sqrt((n - 1) / n * np.sum((u - u_bar) ** 2))
        assert abs(results.se - expected_se) < 1e-10

    def test_jackknife_n_iterations(self):
        """Number of jackknife estimates = n_control + n_treated."""
        n_co, n_tr = 15, 3
        df = _make_panel(n_control=n_co, n_treated=n_tr, seed=42)
        sdid = SyntheticDiD(variance_method="jackknife", seed=42)
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
        )
        assert results.placebo_effects is not None
        assert len(results.placebo_effects) == n_co + n_tr

    def test_jackknife_single_treated_nan(self):
        """Single treated unit -> NaN SE (matches R's NA)."""
        df = _make_panel(n_control=15, n_treated=1, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sdid = SyntheticDiD(variance_method="jackknife", seed=42)
            results = sdid.fit(
                df, outcome="outcome", treatment="treated",
                unit="unit", time="period",
                post_periods=list(range(5, 7)),
            )
        assert np.isnan(results.se)
        assert np.isnan(results.t_stat)
        assert np.isnan(results.p_value)

    def test_jackknife_analytical_pvalue(self):
        """Jackknife should use analytical p-value, not empirical."""
        from scipy.stats import norm

        df = _make_panel(n_control=20, n_treated=3, seed=42)
        sdid = SyntheticDiD(variance_method="jackknife", seed=42)
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
        )
        if np.isfinite(results.t_stat):
            expected_p = 2 * (1 - norm.cdf(abs(results.t_stat)))
            assert abs(results.p_value - expected_p) < 1e-10

    def test_jackknife_same_att_as_placebo(self):
        """Jackknife should produce the same point estimate as placebo."""
        df = _make_panel(n_control=15, n_treated=3, seed=42)
        res_jk = SyntheticDiD(variance_method="jackknife", seed=42).fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
        )
        res_pl = SyntheticDiD(variance_method="placebo", seed=42, n_bootstrap=50).fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
        )
        assert abs(res_jk.att - res_pl.att) < 1e-10

    def test_jackknife_n_bootstrap_ignored(self):
        """n_bootstrap=1 should not raise for jackknife (it's ignored)."""
        sdid = SyntheticDiD(variance_method="jackknife", n_bootstrap=1)
        assert sdid.n_bootstrap == 1
        assert sdid.variance_method == "jackknife"

    def test_jackknife_n_bootstrap_none_in_results(self):
        """Results should have n_bootstrap=None for jackknife."""
        df = _make_panel(n_control=15, n_treated=3, seed=42)
        sdid = SyntheticDiD(variance_method="jackknife", seed=42)
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
        )
        assert results.n_bootstrap is None

    def test_jackknife_with_pweights(self):
        """Jackknife should produce finite SE with survey pweights."""
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=15, n_treated=3, seed=42)
        # Add unit-constant survey weights
        unit_weights = {u: 1.0 + u * 0.1 for u in df["unit"].unique()}
        df["weight"] = df["unit"].map(unit_weights)

        sdid = SyntheticDiD(variance_method="jackknife", seed=42)
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=list(range(5, 8)),
            survey_design=SurveyDesign(weights="weight"),
        )
        assert results.se > 0
        assert np.isfinite(results.se)
        assert results.variance_method == "jackknife"

    def test_jackknife_zero_effective_control_nan(self):
        """Zero-weight controls after composition -> NaN SE."""
        from diff_diff.survey import SurveyDesign

        # 3 controls, 2 treated. Set all but 1 control survey weight to 0
        # so effective support <= 1.
        df = _make_panel(n_control=3, n_treated=2, seed=42)
        weights = {}
        control_units = sorted(df.loc[df["treated"] == 0, "unit"].unique())
        treated_units = sorted(df.loc[df["treated"] == 1, "unit"].unique())
        # Only first control gets positive weight
        for i, u in enumerate(control_units):
            weights[u] = 1.0 if i == 0 else 0.0
        for u in treated_units:
            weights[u] = 1.0
        df["weight"] = df["unit"].map(weights)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sdid = SyntheticDiD(variance_method="jackknife", seed=42)
            results = sdid.fit(
                df, outcome="outcome", treatment="treated",
                unit="unit", time="period",
                post_periods=list(range(5, 7)),
                survey_design=SurveyDesign(weights="weight"),
            )
        assert np.isnan(results.se)

    def test_jackknife_zero_treated_weight_nan(self):
        """Single positive-weight treated unit with survey -> NaN SE."""
        from diff_diff.survey import SurveyDesign

        df = _make_panel(n_control=10, n_treated=2, seed=42)
        weights = {}
        treated_units = sorted(df.loc[df["treated"] == 1, "unit"].unique())
        control_units = sorted(df.loc[df["treated"] == 0, "unit"].unique())
        for u in control_units:
            weights[u] = 1.0
        # Only first treated unit gets positive weight
        weights[treated_units[0]] = 1.0
        weights[treated_units[1]] = 0.0
        df["weight"] = df["unit"].map(weights)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sdid = SyntheticDiD(variance_method="jackknife", seed=42)
            results = sdid.fit(
                df, outcome="outcome", treatment="treated",
                unit="unit", time="period",
                post_periods=list(range(5, 7)),
                survey_design=SurveyDesign(weights="weight"),
            )
        assert np.isnan(results.se)


# =============================================================================
# Jackknife SE - R Golden Value Parity
# =============================================================================


class TestJackknifeSERParity:
    """Verify jackknife SE matches R's synthdid::vcov(method='jackknife').

    Golden values generated with R 4.5.2, synthdid package:

        library(synthdid)
        set.seed(42)
        N0 <- 20; N1 <- 3; T0 <- 5; T1 <- 3
        N <- N0 + N1; T <- T0 + T1
        Y <- matrix(0, nrow=N, ncol=T)
        for (i in 1:N) {
          unit_fe <- rnorm(1, sd=2)
          for (t in 1:T) {
            Y[i,t] <- 10 + unit_fe + (t-1)*0.3 + rnorm(1, sd=0.5)
            if (i > N0 && t > T0) Y[i,t] <- Y[i,t] + 5.0
          }
        }
        tau_hat <- synthdid_estimate(Y, N0, T0)
        se_jk <- sqrt(vcov(tau_hat, method="jackknife")[1,1])
    """

    # R's Y matrix (23 units x 8 periods), row-major
    Y_FLAT = [
        12.459567808595292, 13.223481099962006, 13.658348196773856,
        13.844051055863837, 13.888854636247594, 14.997677893012806,
        14.494587375086788, 15.851128751231856, 10.527006629006900,
        11.317894498245712, 9.780141451338988, 10.635177418486473,
        11.007911133698329, 11.692547000930196, 11.532445341187122,
        10.646344091442769, 5.779122815714058, 5.265746845809725,
        4.828411925858962, 5.933107464969151, 6.926403492435262,
        7.566662873481445, 6.703831577045862, 7.090431451464497,
        6.703722507026075, 6.453676391630379, 7.301398891231049,
        7.726092498224848, 8.191225590595401, 7.669210641906834,
        8.526151391259425, 7.715169490073769, 8.005628186152748,
        7.523978158267692, 9.049143286687135, 9.434081283341134,
        9.450553333966674, 10.310163601090766, 9.867729569702721,
        9.846941461031360, 10.459939463684098, 11.887686682638062,
        11.249912950470762, 12.093459993478538, 12.226598684379407,
        11.973716581337246, 13.453499811673423, 13.287085704636093,
        10.317796666844943, 10.819165701226847, 10.824437736488752,
        9.582976251622744, 11.521962769964540, 11.495903971828724,
        12.072136575632017, 12.570433156881965, 12.435827624848123,
        13.750744970607428, 13.567397714461393, 14.218726703934166,
        14.459837938730677, 14.659912736018788, 14.077914185301429,
        14.854380461280002, 10.770274645112915, 11.275621916712160,
        12.137534572839927, 12.531125692916383, 12.678920118269170,
        12.304148175294246, 12.497145874675160, 14.103389828901550,
        10.560062989643855, 10.755394606294518, 10.518678427483797,
        11.721841324084256, 11.607272952190801, 11.924464521898100,
        12.782516039349641, 13.026729430318186, 12.546145790341205,
        13.409407032231695, 14.079787980063543, 13.128838312144593,
        13.553836458429620, 13.718363411441658, 13.854625752117343,
        14.924224028489123, 11.906891367097627, 12.128784222882244,
        11.404804355878456, 13.130649630134753, 12.173021974919472,
        12.859165585526416, 12.895280738363951, 13.345233593320895,
        10.435966548001499, 10.663839793569295, 11.030422432974012,
        11.033668451079661, 11.324277503659044, 11.045836529045589,
        11.985219205566086, 12.220060940064094, 14.722723885094736,
        15.772410109968900, 15.256969467031452, 15.568564129971197,
        16.666133193788099, 16.405462433247578, 17.202870693537243,
        17.289652559976691, 7.760317864391456, 8.460282811921017,
        9.462415007659978, 9.956467084312777, 9.726218110324272,
        10.272688229133685, 11.134101608790994, 11.592584658589104,
        7.747112683063268, 8.706521663648207, 8.170907672905205,
        8.679537720718859, 8.962718814069811, 8.861932954235140,
        9.383430460745986, 9.891050023644237, 9.728955313568255,
        9.231765881057163, 9.555677785583788, 10.420693590160205,
        9.844078095298698, 10.651913064308546, 10.196489890710358,
        11.855847076501993, 9.218785934915712, 9.133582433258733,
        10.048827580363175, 9.952567508276010, 10.385962432276619,
        11.596546220044132, 11.164945662130776, 11.016817405176500,
        10.145044557120791, 10.921420538928436, 11.642624728800259,
        10.730067509380019, 11.753738913724906, 11.868862794274008,
        12.574196556067037, 12.311524695461632, 10.800710206252880,
        12.817967597577915, 12.705627126180516, 12.497850142478354,
        12.148734571851643, 13.494742486942219, 13.714835068828613,
        13.770060323710533, 10.010857300549947, 10.787315152039971,
        11.050238955584605, 11.063282099053561, 10.834793458278272,
        17.153286194944865, 17.380010096861866, 16.984758489324143,
        6.913302966281331, 6.938279687001069, 7.537129527669741,
        7.063822443245238, 7.531238453797332, 13.853711102827464,
        13.812711128345372, 14.204067444347162, 13.694867606609098,
        12.929992273442151, 14.397345491024691, 15.116119455987304,
        15.860226513457558, 19.442026093187646, 19.855029109494353,
        20.377546194927845,
    ]
    N0, N1, T0, T1 = 20, 3, 5, 3
    R_ATT = 4.980848860060929
    R_JACKKNIFE_SE = 0.613846670319004

    @pytest.fixture
    def r_panel_df(self):
        """Reconstruct the R panel as a pandas DataFrame."""
        N = self.N0 + self.N1
        T = self.T0 + self.T1
        Y = np.array(self.Y_FLAT).reshape(N, T)
        rows = []
        for i in range(N):
            for t in range(T):
                rows.append({
                    "unit": i,
                    "time": t,
                    "outcome": Y[i, t],
                    "treated": int(i >= self.N0),
                })
        return pd.DataFrame(rows)

    def test_att_matches_r(self, r_panel_df):
        """ATT should match R's synthdid_estimate to machine precision."""
        sdid = SyntheticDiD(variance_method="jackknife", seed=42)
        results = sdid.fit(
            r_panel_df, outcome="outcome", treatment="treated",
            unit="unit", time="time",
            post_periods=[5, 6, 7],
        )
        assert abs(results.att - self.R_ATT) < 1e-10

    def test_jackknife_se_matches_r(self, r_panel_df):
        """Jackknife SE should match R's vcov(method='jackknife')."""
        sdid = SyntheticDiD(variance_method="jackknife", seed=42)
        results = sdid.fit(
            r_panel_df, outcome="outcome", treatment="treated",
            unit="unit", time="time",
            post_periods=[5, 6, 7],
        )
        assert abs(results.se - self.R_JACKKNIFE_SE) < 1e-10


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case handling."""

    def test_n_bootstrap_validation(self):
        """n_bootstrap < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="n_bootstrap must be >= 2"):
            SyntheticDiD(n_bootstrap=0)
        with pytest.raises(ValueError, match="n_bootstrap must be >= 2"):
            SyntheticDiD(n_bootstrap=1)
        # n_bootstrap=2 should be accepted
        sdid = SyntheticDiD(n_bootstrap=2)
        assert sdid.n_bootstrap == 2

    def test_single_treated_unit(self, ci_params):
        """Estimation should work with a single treated unit."""
        df = _make_panel(n_control=10, n_treated=1, n_pre=5, n_post=2,
                         att=3.0, seed=42)
        n_boot = ci_params.bootstrap(30)
        sdid = SyntheticDiD(n_bootstrap=n_boot, seed=42)
        results = sdid.fit(
            df, outcome="outcome", treatment="treated",
            unit="unit", time="period",
            post_periods=[5, 6],
        )
        assert np.isfinite(results.att)

    def test_insufficient_controls_for_placebo(self):
        """Placebo with n_control <= n_treated should warn and return SE=0."""
        df = _make_panel(n_control=2, n_treated=3, n_pre=5, n_post=2, seed=42)
        sdid = SyntheticDiD(variance_method="placebo", seed=42)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = sdid.fit(
                df, outcome="outcome", treatment="treated",
                unit="unit", time="period",
                post_periods=[5, 6],
            )
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            # Should warn about insufficient controls for placebo
            assert len(user_warnings) > 0

        assert results.se == 0.0

    def test_se_zero_propagation(self):
        """When SE=0, t_stat and p_value should be NaN, CI should be NaN."""
        df = _make_panel(n_control=2, n_treated=3, n_pre=5, n_post=2, seed=42)
        sdid = SyntheticDiD(variance_method="placebo", seed=42)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = sdid.fit(
                df, outcome="outcome", treatment="treated",
                unit="unit", time="period",
                post_periods=[5, 6],
            )

        if results.se == 0.0:
            assert np.isnan(results.t_stat)
            assert np.isnan(results.p_value)
            assert np.isnan(results.conf_int[0])
            assert np.isnan(results.conf_int[1])

    def test_nonfinite_tau_filtered_in_bootstrap(self):
        """Non-finite tau values are filtered in Python bootstrap path (matches Rust)."""
        call_count = [0]

        def mock_estimator(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                return np.inf
            return 1.0 + call_count[0] * 0.01

        rng = np.random.default_rng(42)
        n_pre, n_post, n_control, n_treated = 5, 2, 10, 3
        Y_pre_c = rng.normal(size=(n_pre, n_control))
        Y_post_c = rng.normal(size=(n_post, n_control))
        Y_pre_t = rng.normal(size=(n_pre, n_treated))
        Y_post_t = rng.normal(size=(n_post, n_treated))
        unit_weights = np.ones(n_control) / n_control
        time_weights = np.ones(n_pre) / n_pre

        sdid = SyntheticDiD(n_bootstrap=20, seed=42)

        with patch('diff_diff.synthetic_did.compute_sdid_estimator',
                   side_effect=mock_estimator), \
             patch('diff_diff._backend.HAS_RUST_BACKEND', False):
            se, estimates = sdid._bootstrap_se(
                Y_pre_c, Y_post_c, Y_pre_t, Y_post_t,
                unit_weights, time_weights,
            )

        # All retained estimates must be finite (non-finite filtered out)
        assert np.all(np.isfinite(estimates)), "Non-finite tau leaked into bootstrap estimates"
        # Some estimates should have been filtered (every 3rd call returns inf)
        assert len(estimates) < 20

    def test_nonfinite_tau_filtered_in_placebo(self):
        """Non-finite tau values are filtered in Python placebo path (matches Rust)."""
        call_count = [0]

        def mock_estimator(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                return np.nan
            return 2.0 + call_count[0] * 0.01

        rng = np.random.default_rng(42)
        n_pre, n_post, n_control, n_treated = 5, 2, 15, 3
        Y_pre_c = rng.normal(size=(n_pre, n_control))
        Y_post_c = rng.normal(size=(n_post, n_control))
        Y_pre_t_mean = rng.normal(size=(n_pre,))
        Y_post_t_mean = rng.normal(size=(n_post,))

        sdid = SyntheticDiD(seed=42)

        with patch('diff_diff.synthetic_did.compute_sdid_estimator',
                   side_effect=mock_estimator), \
             patch('diff_diff.synthetic_did.compute_sdid_unit_weights',
                   return_value=np.ones(n_control - n_treated) / (n_control - n_treated)), \
             patch('diff_diff.synthetic_did.compute_time_weights',
                   return_value=np.ones(n_pre) / n_pre), \
             patch('diff_diff._backend.HAS_RUST_BACKEND', False):
            se, estimates = sdid._placebo_variance_se(
                Y_pre_c, Y_post_c, Y_pre_t_mean, Y_post_t_mean,
                n_treated=n_treated,
                replications=20,
            )

        # All retained estimates must be finite (non-finite filtered out)
        assert np.all(np.isfinite(estimates)), "Non-finite tau leaked into placebo estimates"
        # Some estimates should have been filtered (every 3rd call returns nan)
        assert len(estimates) < 20

    def test_inf_se_produces_nan_inference(self):
        """SE=inf should produce NaN t_stat, p_value, and CI."""
        df = _make_panel(n_control=10, n_treated=3, n_pre=5, n_post=2, seed=42)
        sdid = SyntheticDiD(variance_method="bootstrap", n_bootstrap=10, seed=42)

        with patch.object(
            SyntheticDiD, '_bootstrap_se',
            return_value=(np.inf, np.array([1.0, 2.0, 3.0])),
        ):
            results = sdid.fit(
                df, outcome="outcome", treatment="treated",
                unit="unit", time="period",
                post_periods=[5, 6],
            )

        assert results.se == np.inf
        assert np.isnan(results.t_stat)
        assert np.isnan(results.p_value)
        assert np.isnan(results.conf_int[0])
        assert np.isnan(results.conf_int[1])


# =============================================================================
# get_params / set_params
# =============================================================================


class TestGetSetParams:
    """Verify parameter accessors."""

    def test_get_params_includes_new_names(self):
        """get_params should include zeta_omega/zeta_lambda."""
        sdid = SyntheticDiD(zeta_omega=1.0, zeta_lambda=0.5)
        params = sdid.get_params()
        assert "zeta_omega" in params
        assert "zeta_lambda" in params
        assert params["zeta_omega"] == 1.0
        assert params["zeta_lambda"] == 0.5

    def test_get_params_excludes_old_names(self):
        """get_params should NOT include lambda_reg or zeta."""
        sdid = SyntheticDiD()
        params = sdid.get_params()
        assert "lambda_reg" not in params
        assert "zeta" not in params

    def test_set_params_new_names(self):
        """set_params with new names should work."""
        sdid = SyntheticDiD()
        sdid.set_params(zeta_omega=2.0, zeta_lambda=0.1)
        assert sdid.zeta_omega == 2.0
        assert sdid.zeta_lambda == 0.1

    def test_set_params_deprecated_names_warn(self):
        """set_params with old names should emit DeprecationWarning."""
        sdid = SyntheticDiD()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sdid.set_params(lambda_reg=1.0)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 1

    def test_set_params_unknown_raises(self):
        """set_params with unknown name should raise ValueError."""
        sdid = SyntheticDiD()
        with pytest.raises(ValueError, match="Unknown parameter"):
            sdid.set_params(nonexistent_param=1.0)


class TestDeprecatedParams:
    """Test deprecated parameter handling in __init__."""

    def test_lambda_reg_warns(self):
        """SyntheticDiD(lambda_reg=...) emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sdid = SyntheticDiD(lambda_reg=0.1)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) == 1
            assert "lambda_reg" in str(dep[0].message)

        # Deprecated param is ignored — auto-computed used
        assert sdid.zeta_omega is None

    def test_zeta_warns(self):
        """SyntheticDiD(zeta=...) emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sdid = SyntheticDiD(zeta=2.0)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) == 1
            assert "zeta" in str(dep[0].message)

        assert sdid.zeta_lambda is None

    def test_both_deprecated_params(self):
        """Both deprecated params at once should emit two warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SyntheticDiD(lambda_reg=0.5, zeta=1.5)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) == 2

    def test_default_variance_method_is_placebo(self):
        """Default variance_method should be 'placebo' (matching R)."""
        sdid = SyntheticDiD()
        assert sdid.variance_method == "placebo"


class TestNoiseLevelEdgeCases:
    """Edge case tests for _compute_noise_level_numpy."""

    def test_noise_level_single_control_two_periods(self):
        """noise_level returns 0.0 (not NaN) for 1 control, 2 pre-periods.

        With shape (2, 1), first_diffs has size=1, and np.std([x], ddof=1)
        would divide by zero → NaN. Guard ensures 0.0 is returned instead,
        matching the Rust backend behavior.
        """
        Y = np.array([[1.0], [2.0]])  # (2, 1)
        result = _compute_noise_level_numpy(Y)
        assert result == 0.0
        assert not np.isnan(result)

    def test_noise_level_single_element_returns_zero(self):
        """noise_level returns 0.0 when first_diffs has exactly 1 element."""
        # (2, 1) → diff → (1, 1) → size=1 → return 0.0
        Y = np.array([[5.0], [10.0]])
        result = _compute_noise_level_numpy(Y)
        assert result == 0.0

    def test_noise_level_empty_returns_zero(self):
        """noise_level returns 0.0 for single time period (no diffs possible)."""
        Y = np.array([[1.0, 2.0, 3.0]])  # (1, 3)
        result = _compute_noise_level_numpy(Y)
        assert result == 0.0


class TestPlaceboReestimation:
    """Tests verifying placebo variance re-estimates weights (not fixed)."""

    def test_placebo_reestimates_weights_not_fixed(self):
        """Placebo variance re-estimates omega/lambda per replication (matching R).

        Verifies the methodology choice: R's vcov(method='placebo') passes
        update.omega=TRUE, update.lambda=TRUE, so weights are re-estimated
        via Frank-Wolfe on each permutation — NOT renormalized from originals.

        We verify this by comparing the actual placebo SE against a manual
        fixed-weight computation; if they differ, re-estimation is happening.
        """
        # Need enough controls for placebo to work (n_control > n_treated)
        # and enough variation for weights to differ between re-estimation
        # and renormalization.
        df = _make_panel(n_control=15, n_treated=2, n_pre=6, n_post=3,
                         att=5.0, seed=123)
        post_periods = list(range(6, 9))

        # Fit SDID to get original weights and matrices
        sdid = SyntheticDiD(variance_method="placebo", n_bootstrap=50, seed=42)
        results = sdid.fit(
            df,
            outcome="outcome",
            treatment="treated",
            unit="unit",
            time="period",
            post_periods=post_periods,
        )
        actual_se = results.se

        # Now compute a "fixed weight" placebo manually:
        # permute controls, renormalize original omega (no Frank-Wolfe),
        # keep original lambda unchanged.
        rng = np.random.default_rng(42)

        # Build the outcome matrix (T, N) as the estimator does
        pivot = df.pivot(index="period", columns="unit", values="outcome")
        control_units = sorted(results.unit_weights.keys())
        treated_mask = df.groupby("unit")["treated"].max().values.astype(bool)
        control_idx = np.where(~treated_mask)[0]
        treated_idx = np.where(treated_mask)[0]
        Y = pivot.values  # (T, N)
        pre_periods_arr = np.array(post_periods)
        pre_mask = ~np.isin(pivot.index.values, pre_periods_arr)

        Y_pre_control = Y[np.ix_(pre_mask, control_idx)]
        Y_post_control = Y[np.ix_(~pre_mask, control_idx)]

        # Extract numpy arrays from result dicts (ordered by control unit)
        unit_weights_arr = np.array([results.unit_weights[u] for u in control_units])
        time_weights_arr = np.array([results.time_weights[t]
                                     for t in sorted(results.time_weights.keys())])

        n_control = len(control_idx)
        n_treated_count = len(treated_idx)
        n_pseudo_control = n_control - n_treated_count

        fixed_estimates = []
        for _ in range(50):
            perm = rng.permutation(n_control)
            pc_idx = perm[:n_pseudo_control]
            pt_idx = perm[n_pseudo_control:]

            # Fixed weights: renormalize original omega for pseudo-controls
            fixed_omega = _sum_normalize(unit_weights_arr[pc_idx])
            fixed_lambda = time_weights_arr  # unchanged

            Y_pre_pc = Y_pre_control[:, pc_idx]
            Y_post_pc = Y_post_control[:, pc_idx]
            Y_pre_pt_mean = np.mean(Y_pre_control[:, pt_idx], axis=1)
            Y_post_pt_mean = np.mean(Y_post_control[:, pt_idx], axis=1)

            try:
                tau = compute_sdid_estimator(
                    Y_pre_pc, Y_post_pc,
                    Y_pre_pt_mean, Y_post_pt_mean,
                    fixed_omega, fixed_lambda,
                )
                fixed_estimates.append(tau)
            except (ValueError, np.linalg.LinAlgError):
                continue

        if len(fixed_estimates) >= 2:
            n_s = len(fixed_estimates)
            fixed_se = (np.sqrt((n_s - 1) / n_s)
                        * np.std(fixed_estimates, ddof=1))
            # The two SEs should differ because re-estimation produces
            # different weights than renormalization
            assert actual_se != pytest.approx(fixed_se, rel=0.01), (
                f"Placebo SE ({actual_se:.6f}) matches fixed-weight SE "
                f"({fixed_se:.6f}), suggesting weights are NOT being "
                f"re-estimated as R's synthdid does."
            )


# =============================================================================
# Treatment Validation
# =============================================================================


class TestTreatmentValidation:
    """Test that SDID rejects time-varying treatment (staggered designs)."""

    def test_varying_treatment_within_unit_raises(self):
        """Unit whose treatment switches over time should raise ValueError."""
        np.random.seed(42)
        data = pd.DataFrame({
            "unit": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "time": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4],
            "outcome": np.random.randn(12),
            # Unit 1: treatment turns on at time 3 (staggered)
            "treated": [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        })
        sdid = SyntheticDiD()
        with pytest.raises(ValueError, match="Treatment indicator varies within"):
            sdid.fit(
                data, outcome="outcome", treatment="treated",
                unit="unit", time="time", post_periods=[3, 4],
            )

    def test_constant_treatment_passes(self):
        """Normal block-treatment data should pass validation."""
        np.random.seed(42)
        n_units, n_periods = 10, 8
        rows = []
        for u in range(n_units):
            is_treated = 1 if u < 3 else 0
            for t in range(n_periods):
                rows.append({
                    "unit": u, "time": t,
                    "outcome": np.random.randn() + (2.0 if is_treated and t >= 5 else 0),
                    "treated": is_treated,
                })
        data = pd.DataFrame(rows)
        sdid = SyntheticDiD()
        result = sdid.fit(
            data, outcome="outcome", treatment="treated",
            unit="unit", time="time", post_periods=[5, 6, 7],
        )
        assert result is not None


# =============================================================================
# Balanced Panel Validation
# =============================================================================


class TestBalancedPanelValidation:
    """Test that SDID rejects unbalanced panels."""

    def test_unbalanced_panel_raises(self):
        """Unit missing a period should raise ValueError."""
        np.random.seed(42)
        rows = []
        for u in range(6):
            is_treated = 1 if u < 2 else 0
            for t in range(5):
                rows.append({
                    "unit": u, "time": t,
                    "outcome": np.random.randn(),
                    "treated": is_treated,
                })
        data = pd.DataFrame(rows)
        # Drop one observation to make panel unbalanced
        data = data[~((data["unit"] == 3) & (data["time"] == 2))].reset_index(drop=True)

        sdid = SyntheticDiD()
        with pytest.raises(ValueError, match="Panel is not balanced"):
            sdid.fit(
                data, outcome="outcome", treatment="treated",
                unit="unit", time="time", post_periods=[3, 4],
            )

    def test_balanced_panel_passes(self):
        """Fully balanced panel should pass validation."""
        np.random.seed(42)
        rows = []
        for u in range(8):
            is_treated = 1 if u < 2 else 0
            for t in range(6):
                rows.append({
                    "unit": u, "time": t,
                    "outcome": np.random.randn() + (1.5 if is_treated and t >= 4 else 0),
                    "treated": is_treated,
                })
        data = pd.DataFrame(rows)
        sdid = SyntheticDiD()
        result = sdid.fit(
            data, outcome="outcome", treatment="treated",
            unit="unit", time="time", post_periods=[4, 5],
        )
        assert result is not None


# =============================================================================
# Pre-treatment Fit Warning
# =============================================================================


class TestPreTreatmentFitWarning:
    """Test that poor pre-treatment fit emits a warning."""

    def test_poor_fit_emits_warning(self):
        """Treated units at very different level from controls should warn."""
        np.random.seed(42)
        rows = []
        for u in range(10):
            is_treated = 1 if u < 2 else 0
            # Large level difference: treated ~100, control ~10
            level = 100.0 if is_treated else 10.0
            for t in range(8):
                rows.append({
                    "unit": u, "time": t,
                    "outcome": level + np.random.randn() * 0.5,
                    "treated": is_treated,
                })
        data = pd.DataFrame(rows)
        sdid = SyntheticDiD()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sdid.fit(
                data, outcome="outcome", treatment="treated",
                unit="unit", time="time", post_periods=[6, 7],
            )
            fit_warnings = [x for x in w if "Pre-treatment fit is poor" in str(x.message)]
            assert len(fit_warnings) >= 1, (
                "Expected warning about poor pre-treatment fit but none was raised"
            )

    def test_good_fit_no_warning(self):
        """Parallel trends data with similar levels should not warn."""
        np.random.seed(42)
        rows = []
        for u in range(10):
            is_treated = 1 if u < 3 else 0
            for t in range(8):
                # Same level, parallel trends, treatment effect only in post
                rows.append({
                    "unit": u, "time": t,
                    "outcome": t + np.random.randn() * 0.3 + (2.0 if is_treated and t >= 5 else 0),
                    "treated": is_treated,
                })
        data = pd.DataFrame(rows)
        sdid = SyntheticDiD()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sdid.fit(
                data, outcome="outcome", treatment="treated",
                unit="unit", time="time", post_periods=[5, 6, 7],
            )
            fit_warnings = [x for x in w if "Pre-treatment fit is poor" in str(x.message)]
            assert len(fit_warnings) == 0, (
                f"Unexpected pre-treatment fit warning: {fit_warnings[0].message}"
            )


class TestMinDecreaseFloor:
    """Test min_decrease floor when noise_level == 0."""

    def test_floor_equivalence(self):
        """min_decrease=1e-5 floor vs 0.0 gives same weights on zero-noise data."""
        # Build zero-noise collapsed-form matrix (N_co=5, T_pre+1=4)
        # All rows identical → noise_level would be 0
        np.random.seed(42)
        Y = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), (5, 1))

        w_floor = _sc_weight_fw(Y, zeta=0.0, intercept=True,
                                min_decrease=1e-5, max_iter=10000)
        w_zero = _sc_weight_fw(Y, zeta=0.0, intercept=True,
                               min_decrease=0.0, max_iter=10000)

        np.testing.assert_allclose(w_floor, w_zero, atol=1e-10,
                                   err_msg="Floor min_decrease should give same weights as 0.0")
        # Both should be valid simplex weights
        assert np.all(w_floor >= -1e-12), "Weights should be non-negative"
        assert abs(w_floor.sum() - 1.0) < 1e-10, "Weights should sum to 1"


class TestBackendSEConsistency:
    """Test that pure Python and Rust-accelerated backends produce identical SEs."""

    def test_placebo_se_matches_across_backends(self):
        """SyntheticDiD placebo SE is identical regardless of backend.

        After removing the Rust outer-loop variance fast paths, both backends
        use the same Python loop for placebo/bootstrap variance. The only
        difference is that inner Frank-Wolfe weight calls may dispatch to Rust
        (faer) vs NumPy. With the same seed, RNG sequences are identical, so
        SEs should match within floating-point tolerance (rtol=1e-4 allows for
        accumulated faer vs NumPy differences across hundreds of iterations).
        """
        import sys

        df = _make_panel(n_control=20, n_treated=3, n_pre=5, n_post=3,
                         att=5.0, seed=42)
        post_periods = list(range(5, 8))

        # Run with default backend (Rust-accelerated inner calls if available)
        sdid_default = SyntheticDiD(
            variance_method="placebo", n_bootstrap=50, seed=42
        )
        results_default = sdid_default.fit(
            df, "outcome", "treated", "unit", "period", post_periods
        )

        # Run with pure Python backend
        utils_mod = sys.modules["diff_diff.utils"]
        with patch.object(utils_mod, "HAS_RUST_BACKEND", False):
            sdid_py = SyntheticDiD(
                variance_method="placebo", n_bootstrap=50, seed=42
            )
            results_py = sdid_py.fit(
                df.copy(), "outcome", "treated", "unit", "period", post_periods
            )

        # ATT must be identical (same data, same algorithm)
        np.testing.assert_allclose(
            results_default.att, results_py.att, rtol=1e-6,
            err_msg="ATT should match between backends"
        )

        # SE must match within tolerance (faer vs NumPy in FW inner loop)
        np.testing.assert_allclose(
            results_default.se, results_py.se, rtol=1e-4,
            err_msg="Placebo SE should match between backends"
        )

    def test_bootstrap_se_matches_across_backends(self):
        """SyntheticDiD bootstrap SE is identical regardless of backend."""
        import sys

        df = _make_panel(n_control=20, n_treated=3, n_pre=5, n_post=3,
                         att=5.0, seed=42)
        post_periods = list(range(5, 8))

        # Run with default backend
        sdid_default = SyntheticDiD(
            variance_method="bootstrap", n_bootstrap=50, seed=42
        )
        results_default = sdid_default.fit(
            df, "outcome", "treated", "unit", "period", post_periods
        )

        # Run with pure Python backend
        utils_mod = sys.modules["diff_diff.utils"]
        with patch.object(utils_mod, "HAS_RUST_BACKEND", False):
            sdid_py = SyntheticDiD(
                variance_method="bootstrap", n_bootstrap=50, seed=42
            )
            results_py = sdid_py.fit(
                df.copy(), "outcome", "treated", "unit", "period", post_periods
            )

        # ATT must match
        np.testing.assert_allclose(
            results_default.att, results_py.att, rtol=1e-6,
            err_msg="ATT should match between backends"
        )

        # Bootstrap SE must match (same RNG, same loop, only inner FW differs)
        np.testing.assert_allclose(
            results_default.se, results_py.se, rtol=1e-4,
            err_msg="Bootstrap SE should match between backends"
        )


class TestEmptyPostGuard:
    """Test compute_time_weights guard for empty Y_post_control."""

    def test_empty_post_raises(self):
        """compute_time_weights raises ValueError when Y_post_control has 0 rows."""
        Y_pre = np.random.randn(4, 5)   # 4 pre-periods, 5 controls
        Y_post = np.empty((0, 5))        # 0 post-periods

        with pytest.raises(ValueError, match="Y_post_control has no rows"):
            compute_time_weights(Y_pre, Y_post, zeta_lambda=0.0)
