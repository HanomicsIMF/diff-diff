"""
Methodology verification tests for Honest DiD (Rambachan & Roth, 2023).

These tests verify the corrected implementation against the paper's
equations, known analytical cases, and expected mathematical properties.
"""

import numpy as np
import pytest

from diff_diff.honest_did import (
    HonestDiD,
    _compute_flci,
    _compute_optimal_flci,
    _compute_pre_first_differences,
    _construct_A_sd,
    _construct_constraints_rm_component,
    _construct_constraints_sd,
    _cv_alpha,
    _solve_bounds_lp,
    _solve_rm_bounds_union,
)


# =============================================================================
# TestDeltaSDConstraintMatrix
# =============================================================================


class TestDeltaSDConstraintMatrix:
    """Verify DeltaSD constraint matrix accounts for delta_0 = 0 boundary."""

    def test_row_count(self):
        """T+Tbar-1 rows, not T+Tbar-2 (accounts for delta_0 = 0)."""
        for T, Tbar in [(2, 2), (3, 3), (4, 2), (1, 1), (3, 1), (1, 3)]:
            A = _construct_A_sd(T, Tbar)
            expected_rows = T + Tbar - 1
            assert A.shape == (expected_rows, T + Tbar), (
                f"T={T}, Tbar={Tbar}: expected {expected_rows} rows, got {A.shape[0]}"
            )

    def test_2pre_2post_hand_computed(self):
        """Hand-computed matrix for 2 pre + 2 post periods."""
        # delta = [d_{-2}, d_{-1}, d_1, d_2]
        A = _construct_A_sd(2, 2)
        expected = np.array([
            [1, -2, 0, 0],   # t=-1: d_{-2} - 2*d_{-1} + 0
            [0,  1, 1, 0],   # t= 0: d_{-1} + d_1 (bridge)
            [0,  0, -2, 1],  # t= 1: 0 - 2*d_1 + d_2
        ])
        np.testing.assert_array_equal(A, expected)

    def test_bridge_constraint_present(self):
        """The bridge constraint delta_{-1} + delta_1 is always present."""
        for T, Tbar in [(1, 1), (2, 2), (4, 3)]:
            A = _construct_A_sd(T, Tbar)
            # Find the bridge row: non-zero only at positions T-1 and T
            bridge_found = False
            for row in A:
                if row[T - 1] != 0 and row[T] != 0:
                    # This should be [0, ..., 1, 1, ..., 0]
                    assert row[T - 1] == 1, f"Bridge row should have 1 at delta_{{-1}}"
                    assert row[T] == 1, f"Bridge row should have 1 at delta_1"
                    bridge_found = True
            assert bridge_found, f"Bridge constraint not found for T={T}, Tbar={Tbar}"

    def test_constraints_span_all_periods(self):
        """Constraints involve both pre and post periods (not pre-only)."""
        A = _construct_A_sd(3, 3)
        # Some rows should have non-zero entries in post-period columns
        post_cols = A[:, 3:]  # columns for delta_1, delta_2, delta_3
        assert np.any(post_cols != 0), "No constraints involve post-period deltas"


# =============================================================================
# TestIdentifiedSetLP
# =============================================================================


class TestIdentifiedSetLP:
    """Verify identified set LP pins delta_pre = beta_pre."""

    def test_m0_linear_extrapolation(self):
        """M=0 with linear pre-trends gives finite point-identified bounds."""
        # Pre-trends: linear decline with slope -0.1
        beta_pre = np.array([0.3, 0.2, 0.1])
        beta_post = np.array([2.0])
        l_vec = np.array([1.0])

        A, b = _construct_constraints_sd(3, 1, M=0.0)
        lb, ub = _solve_bounds_lp(beta_pre, beta_post, l_vec, A, b, 3)

        # Linear extrapolation: slope = -0.1, so delta_1 = 0 - 0.1 = -0.1
        # theta = beta_post - delta_post = 2.0 - (-0.1) = 2.1
        assert np.isfinite(lb), "M=0 should give finite lower bound"
        assert np.isfinite(ub), "M=0 should give finite upper bound"
        np.testing.assert_allclose(lb, 2.1, atol=1e-6)
        np.testing.assert_allclose(ub, 2.1, atol=1e-6)

    def test_bounds_widen_with_m(self):
        """Identified set widens monotonically with M."""
        beta_pre = np.array([0.3, 0.2, 0.1])
        beta_post = np.array([2.0])
        l_vec = np.array([1.0])

        prev_width = 0
        for M in [0.0, 0.1, 0.5, 1.0]:
            A, b = _construct_constraints_sd(3, 1, M=M)
            lb, ub = _solve_bounds_lp(beta_pre, beta_post, l_vec, A, b, 3)
            width = ub - lb
            assert width >= prev_width - 1e-10, (
                f"Width should increase: M={M}, width={width}, prev={prev_width}"
            )
            prev_width = width

    def test_three_period_analytical(self):
        """Paper Section 2.3: three-period example (T=1, Tbar=1)."""
        # delta = [d_{-1}, d_1], with delta_0 = 0
        # DeltaSD(M): |d_1 + d_{-1}| <= M (bridge constraint only)
        # With d_{-1} = beta_{-1} pinned:
        #   d_1 in [-(beta_{-1} + M), -(beta_{-1} - M)] = [-beta_{-1} - M, -beta_{-1} + M]
        # theta = beta_1 - d_1
        #   lb = beta_1 - (-beta_{-1} + M) = beta_1 + beta_{-1} - M
        #   ub = beta_1 - (-beta_{-1} - M) = beta_1 + beta_{-1} + M
        beta_pre = np.array([0.5])
        beta_post = np.array([3.0])

        for M in [0.0, 0.2, 1.0]:
            A, b = _construct_constraints_sd(1, 1, M=M)
            lb, ub = _solve_bounds_lp(beta_pre, beta_post, np.array([1.0]), A, b, 1)
            expected_lb = 3.0 + 0.5 - M
            expected_ub = 3.0 + 0.5 + M
            np.testing.assert_allclose(lb, expected_lb, atol=1e-6,
                                       err_msg=f"M={M}: lb mismatch")
            np.testing.assert_allclose(ub, expected_ub, atol=1e-6,
                                       err_msg=f"M={M}: ub mismatch")


# =============================================================================
# TestDeltaRMFirstDifferences
# =============================================================================


class TestDeltaRMFirstDifferences:
    """Verify DeltaRM constrains first differences, not levels."""

    def test_pre_first_differences_computation(self):
        """Pre-period first differences include delta_0=0 boundary."""
        beta_pre = np.array([0.3, 0.2, 0.1])
        diffs = _compute_pre_first_differences(beta_pre)

        # Interior: |0.2-0.3|=0.1, |0.1-0.2|=0.1
        # Boundary: |0 - 0.1| = 0.1
        np.testing.assert_allclose(diffs, [0.1, 0.1, 0.1], atol=1e-10)

    def test_pre_first_differences_boundary(self):
        """The boundary term |0 - beta_{-1}| is included."""
        beta_pre = np.array([0.0, 0.0, 0.5])
        diffs = _compute_pre_first_differences(beta_pre)

        # Interior: |0-0|=0, |0.5-0|=0.5
        # Boundary: |0 - 0.5| = 0.5
        np.testing.assert_allclose(diffs, [0.0, 0.5, 0.5], atol=1e-10)

    def test_rm_constraints_are_first_differences(self):
        """RM constraint matrix constrains consecutive differences, not levels."""
        A, b = _construct_constraints_rm_component(2, 3, Mbar=1.0, max_pre_first_diff=0.1)

        # 3 post-period first diffs: |d_1|, |d_2-d_1|, |d_3-d_2|
        # Each needs pos/neg constraint = 6 rows total
        assert A.shape[0] == 6
        assert A.shape[1] == 5  # 2 pre + 3 post

        # First pair: d_1 <= 0.1 and -d_1 <= 0.1
        assert A[0, 2] == 1   # d_1
        assert A[1, 2] == -1  # -d_1

        # Second pair: d_2 - d_1 <= 0.1
        assert A[2, 3] == 1 and A[2, 2] == -1   # d_2 - d_1
        assert A[3, 3] == -1 and A[3, 2] == 1   # -(d_2 - d_1)

    def test_mbar0_gives_point_estimate(self):
        """Mbar=0: all post first diffs = 0, theta = l'beta_post."""
        beta_pre = np.array([0.3, 0.2, 0.1])
        beta_post = np.array([2.0, 2.5])
        l_vec = np.array([0.5, 0.5])

        lb, ub = _solve_rm_bounds_union(beta_pre, beta_post, l_vec, 3, Mbar=0.0)

        theta = np.dot(l_vec, beta_post)
        np.testing.assert_allclose(lb, theta, atol=1e-6)
        np.testing.assert_allclose(ub, theta, atol=1e-6)

    def test_rm_bounds_widen_with_mbar(self):
        """Identified set widens monotonically with Mbar."""
        beta_pre = np.array([0.3, 0.2, 0.1])
        beta_post = np.array([2.0, 2.5])
        l_vec = np.array([0.5, 0.5])

        prev_width = 0
        for Mbar in [0.0, 0.5, 1.0, 2.0]:
            lb, ub = _solve_rm_bounds_union(beta_pre, beta_post, l_vec, 3, Mbar)
            width = ub - lb
            assert width >= prev_width - 1e-10, f"Mbar={Mbar}: width decreased"
            prev_width = width


# =============================================================================
# TestOptimalFLCI
# =============================================================================


class TestOptimalFLCI:
    """Verify optimal FLCI properties."""

    def test_cv_alpha_at_zero(self):
        """cv_alpha(0, alpha) = z_{alpha/2} (standard normal quantile)."""
        from scipy.stats import norm
        np.testing.assert_allclose(_cv_alpha(0, 0.05), norm.ppf(0.975), atol=1e-4)
        np.testing.assert_allclose(_cv_alpha(0, 0.01), norm.ppf(0.995), atol=1e-4)

    def test_cv_alpha_monotonic(self):
        """cv_alpha(t) increases with |t| (more bias -> wider CI)."""
        cvs = [_cv_alpha(t, 0.05) for t in [0, 0.5, 1.0, 2.0, 5.0]]
        assert all(cvs[i] <= cvs[i + 1] + 1e-10 for i in range(len(cvs) - 1))

    def test_optimal_flci_narrower_than_naive(self):
        """Optimal FLCI should be no wider than naive FLCI."""
        beta_pre = np.array([0.3, 0.2, 0.1])
        beta_post = np.array([2.0])
        sigma = np.eye(4) * 0.01
        l_vec = np.array([1.0])

        ci_lb_opt, ci_ub_opt = _compute_optimal_flci(
            beta_pre, beta_post, sigma, l_vec, 3, 1, M=0.5, alpha=0.05
        )

        # Naive FLCI
        A, b = _construct_constraints_sd(3, 1, 0.5)
        lb, ub = _solve_bounds_lp(beta_pre, beta_post, l_vec, A, b, 3)
        se = np.sqrt(l_vec @ sigma[3:, 3:] @ l_vec)
        ci_lb_naive, ci_ub_naive = _compute_flci(lb, ub, se, 0.05)

        opt_width = ci_ub_opt - ci_lb_opt
        naive_width = ci_ub_naive - ci_lb_naive
        assert opt_width <= naive_width + 1e-6, (
            f"Optimal ({opt_width:.4f}) wider than naive ({naive_width:.4f})"
        )

    def test_m0_short_circuit(self):
        """M=0 should use standard CI without optimization."""
        beta_pre = np.array([0.3, 0.2, 0.1])
        beta_post = np.array([2.0])
        sigma = np.eye(4) * 0.01
        l_vec = np.array([1.0])

        import time
        t0 = time.time()
        _compute_optimal_flci(beta_pre, beta_post, sigma, l_vec, 3, 1, M=0.0)
        elapsed = time.time() - t0

        assert elapsed < 0.1, f"M=0 should be instant, took {elapsed:.2f}s"

    def test_optimal_flci_width_increases_with_m(self):
        """Regression for P0: smoothness CI width must increase with M."""
        beta_pre = np.array([0.3, 0.2, 0.1])
        beta_post = np.array([2.0])
        sigma = np.eye(4) * 0.01

        widths = []
        for M in [0.0, 0.1, 0.5, 1.0]:
            ci_lb, ci_ub = _compute_optimal_flci(
                beta_pre, beta_post, sigma, np.array([1.0]), 3, 1, M=M
            )
            widths.append(ci_ub - ci_lb)

        for i in range(len(widths) - 1):
            assert widths[i + 1] >= widths[i] - 1e-6, (
                f"CI width must increase with M: M[{i}]={widths[i]:.4f}, "
                f"M[{i+1}]={widths[i+1]:.4f}"
            )

    def test_optimal_flci_bias_nonzero_for_nonzero_m(self):
        """Regression for P0: bias should be nonzero when M > 0."""
        from diff_diff.honest_did import _compute_worst_case_bias

        # Build a v with nonzero pre-weights (as optimizer would choose)
        v = np.array([0.1, 0.05, 0.0, 1.0])  # 3 pre + 1 post, v_post = l
        l_vec = np.array([1.0])
        A, b = _construct_constraints_sd(3, 1, M=0.5)

        bias = _compute_worst_case_bias(v, l_vec, A, b, 3, 1)
        assert bias > 0, f"Bias should be nonzero for M>0 with nonzero v_pre, got {bias}"

    def test_infeasible_lp_returns_nan(self):
        """Regression for P1: infeasible LP should return NaN, not [-inf, inf]."""
        # Non-linear pre-trends that are inconsistent with M=0 smoothness
        beta_pre = np.array([1.0, 0.0, 1.0])  # quadratic, not linear
        beta_post = np.array([2.0])
        A, b = _construct_constraints_sd(3, 1, M=0.0)

        lb, ub = _solve_bounds_lp(beta_pre, beta_post, np.array([1.0]), A, b, 3)
        # M=0 with non-linear pre-trends: should be infeasible
        assert np.isnan(lb) and np.isnan(ub), (
            f"Infeasible LP should return NaN, got [{lb}, {ub}]"
        )


# =============================================================================
# TestBreakdownValueMethodology
# =============================================================================


class TestBreakdownValueMethodology:
    """Verify breakdown value properties."""

    def test_breakdown_monotonicity(self):
        """If significant at M=k, should be significant at all M < k."""
        from diff_diff.results import MultiPeriodDiDResults, PeriodEffect

        # Use a weak effect so breakdown is reachable at moderate M
        period_effects = {
            1: PeriodEffect(period=1, effect=0.1, se=0.05, t_stat=2.0,
                           p_value=0.05, conf_int=(0.0, 0.2)),
            2: PeriodEffect(period=2, effect=0.05, se=0.05, t_stat=1.0,
                           p_value=0.32, conf_int=(-0.05, 0.15)),
            4: PeriodEffect(period=4, effect=0.15, se=0.05, t_stat=3.0,
                           p_value=0.003, conf_int=(0.05, 0.25)),
        }
        results = MultiPeriodDiDResults(
            avg_att=0.15, avg_se=0.05, avg_t_stat=3.0, avg_p_value=0.003,
            avg_conf_int=(0.05, 0.25), n_obs=500, n_treated=250, n_control=250,
            period_effects=period_effects, pre_periods=[1, 2], post_periods=[4],
            vcov=np.eye(3) * 0.0025,
            interaction_indices={1: 0, 2: 1, 4: 2},
        )

        honest = HonestDiD(method="smoothness")
        # Check that CI at M=0 does not include zero
        r0 = honest.fit(results, M=0.0)
        assert r0.ci_lb > 0, "Should be significant at M=0"

        # At sufficiently large M, CI should include zero.
        # The optimal FLCI is efficient, so need large M for a weak effect.
        r_large = honest.fit(results, M=20.0)
        assert r_large.ci_lb <= 0 <= r_large.ci_ub, "Should lose significance at large M"
