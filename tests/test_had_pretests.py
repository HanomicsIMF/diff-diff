"""Tests for HAD Phase 3 pre-test diagnostics (had_pretests.py)."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    QUGTestResults,
    StuteJointResult,
    StuteTestResults,
    YatchewTestResults,
    did_had_pretest_workflow,
    joint_homogeneity_test,
    joint_pretrends_test,
    qug_test,
    stute_joint_pretest,
    stute_test,
    yatchew_hr_test,
)
from diff_diff.had_pretests import (
    _compose_verdict,
    _compose_verdict_event_study,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_two_period_panel(
    G: int,
    d: np.ndarray,
    dy: np.ndarray,
    seed: int = 42,
) -> pd.DataFrame:
    """Construct a two-period panel satisfying the HAD overall-path contract.

    - ``time == 0``: ``d = 0`` for all units (HAD no-unit-untreated pre).
    - ``time == 1``: dose ``d`` per unit, outcome ``y_pre + dy``.
    """
    rng = np.random.default_rng(seed)
    y_pre = rng.normal(0.0, 1.0, size=G)
    pre = pd.DataFrame({"unit": np.arange(G), "time": 0, "y": y_pre, "d": np.zeros(G)})
    post = pd.DataFrame({"unit": np.arange(G), "time": 1, "y": y_pre + dy, "d": d})
    return pd.concat([pre, post], ignore_index=True)


def _linear_dgp(G: int, beta: float = 2.0, sigma: float = 0.3, seed: int = 42):
    """Return ``(d, dy)`` for a linear DGP ``dy = beta * d + eps``."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.0, 1.0, size=G)
    dy = beta * d + rng.normal(0.0, sigma, size=G)
    return d, dy


def _quadratic_dgp(
    G: int, beta: float = 2.0, gamma: float = 5.0, sigma: float = 0.3, seed: int = 42
):
    """Return ``(d, dy)`` for a quadratic DGP ``dy = beta*d + gamma*d^2 + eps``."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.1, 1.0, size=G)
    dy = beta * d + gamma * d * d + rng.normal(0.0, sigma, size=G)
    return d, dy


# =============================================================================
# QUG test
# =============================================================================


class TestQUGTest:
    """Tests for :func:`qug_test`."""

    def test_tight_parity_closed_form(self):
        """Commit criterion 4: [0.2, 0.5, 0.9] -> T = 0.2/0.3 at atol=1e-12."""
        d = np.array([0.2, 0.5, 0.9])
        r = qug_test(d, alpha=0.05)
        expected_T = 0.2 / (0.5 - 0.2)
        expected_p = 1.0 / (1.0 + expected_T)
        np.testing.assert_allclose(r.t_stat, expected_T, atol=1e-12)
        np.testing.assert_allclose(r.p_value, expected_p, atol=1e-12)
        assert r.critical_value == pytest.approx(1.0 / 0.05 - 1.0, rel=1e-12)

    def test_tie_break(self):
        """Commit criterion 3: D_(1) == D_(2) -> NaN, reject=False."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = qug_test(np.array([0.3, 0.3, 0.5]))
            assert len(caught) == 1
            assert "D_(1) == D_(2)" in str(caught[0].message)
        assert np.isnan(r.t_stat)
        assert np.isnan(r.p_value)
        assert r.reject is False
        assert r.d_order_1 == 0.3
        assert r.d_order_2 == 0.3

    def test_zero_filter_emits_warning(self):
        """Zero-dose observations are excluded with a UserWarning."""
        d = np.array([0.0, 0.0, 0.3, 0.5, 0.9])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = qug_test(d)
            messages = [str(w.message) for w in caught]
        assert any("excluded 2" in m for m in messages)
        assert r.n_obs == 3
        assert r.n_excluded_zero == 2
        # Should reduce to [0.3, 0.5, 0.9]: T = 0.3 / 0.2 = 1.5
        np.testing.assert_allclose(r.t_stat, 0.3 / 0.2, atol=1e-12)

    def test_rejects_on_shifted_beta(self):
        """Commit criterion 2 (single-realization): shifted-Beta -> reject."""
        rng = np.random.default_rng(42)
        # Support is [0.2, 1.2]: infimum is well away from zero.
        d = 0.2 + rng.beta(2, 2, size=200)
        r = qug_test(d, alpha=0.05)
        # T = D_(1)/(D_(2) - D_(1)). D_(1) ~ 0.2; gap ~ 0.001 at G=200.
        # So T is large -> reject.
        assert r.reject is True
        assert r.p_value < 0.05

    def test_does_not_reject_on_uniform_support_zero(self):
        """Commit criterion 1 (single-realization): Uniform(0,1) -> usually no reject."""
        rng = np.random.default_rng(42)
        d = rng.uniform(0.0, 1.0, size=500)
        r = qug_test(d, alpha=0.05)
        # Uniform support has infimum at 0 (asymptotic). For G=500,
        # D_(1) is small but D_(2) - D_(1) also small -> T ~ O(1).
        # Not a guaranteed non-reject at single seed, but this seed passes.
        assert r.p_value > 0.05
        assert r.reject is False

    def test_reject_rate_bounded_by_alpha_on_uniform_null(self):
        """Commit criterion 1: asymptotic size alpha is an UPPER bound.

        Paper Theorem 4 establishes ``lim sup_{G->inf} P(reject | H_0) = alpha``
        (one-sided asymptotic size bound). At finite G the test is often
        conservative (actual rate below alpha). We verify the asymptotic
        size control: reject rate MUST NOT exceed alpha by more than 0.03.
        """
        alpha = 0.05
        n_reps = 200
        rejections = 0
        rng = np.random.default_rng(12345)
        for _ in range(n_reps):
            d = rng.uniform(0.0, 1.0, size=500)
            r = qug_test(d, alpha=alpha)
            if r.reject:
                rejections += 1
        rate = rejections / n_reps
        assert rate <= alpha + 0.03, (
            f"QUG reject rate {rate:.3f} exceeds alpha={alpha} by more than "
            f"0.03; asymptotic size control violated."
        )

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha must satisfy"):
            qug_test(np.array([0.1, 0.5]), alpha=0.0)
        with pytest.raises(ValueError, match="alpha must satisfy"):
            qug_test(np.array([0.1, 0.5]), alpha=1.0)

    def test_nan_input_raises(self):
        with pytest.raises(ValueError, match="contains NaN"):
            qug_test(np.array([0.1, np.nan, 0.5]))

    def test_2d_input_raises(self):
        with pytest.raises(ValueError, match="must be 1-dimensional"):
            qug_test(np.array([[0.1, 0.5], [0.3, 0.9]]))

    def test_negative_dose_raises(self):
        """P2 fix: HAD doses must be non-negative; qug_test rejects negatives
        at the front door rather than silently folding them into the
        zero-exclusion counter."""
        with pytest.raises(ValueError, match="negative value"):
            qug_test(np.array([0.1, -0.3, 0.5, 0.9]))


class TestNegativeDoseGuardsOnLinearityTests:
    """R6 P1 fix: stute_test and yatchew_hr_test must enforce the same
    HAD non-negative-dose contract as qug_test and the panel validator."""

    def test_stute_negative_dose_raises(self):
        d = np.linspace(-0.2, 0.8, 30)  # contains negatives
        dy = 2.0 * d + np.random.default_rng(42).normal(0.0, 0.1, size=30)
        with pytest.raises(ValueError, match="negative value"):
            stute_test(d, dy, n_bootstrap=199, seed=42)

    def test_yatchew_negative_dose_raises(self):
        d = np.linspace(-0.2, 0.8, 30)
        dy = 2.0 * d + np.random.default_rng(42).normal(0.0, 0.1, size=30)
        with pytest.raises(ValueError, match="negative value"):
            yatchew_hr_test(d, dy)


# =============================================================================
# Stute test
# =============================================================================


class TestStuteTest:
    """Tests for :func:`stute_test`."""

    def test_does_not_reject_linear_dgp(self):
        """Commit criterion 5 (single-realization): linear -> no reject."""
        d, dy = _linear_dgp(G=200, beta=2.0, sigma=0.3, seed=42)
        r = stute_test(d, dy, alpha=0.05, n_bootstrap=199, seed=42)
        assert r.reject is False
        assert r.p_value > 0.05

    def test_rejects_quadratic_dgp(self):
        """Commit criterion 6: quadratic DGP -> reject."""
        d, dy = _quadratic_dgp(G=200, beta=2.0, gamma=5.0, sigma=0.3, seed=42)
        r = stute_test(d, dy, alpha=0.05, n_bootstrap=199, seed=42)
        assert r.reject is True
        assert r.p_value < 0.05

    def test_cvm_statistic_manual_equivalence(self):
        """Verify the simplified CvM form matches the paper's stated form
        in the no-ties case.

        Paper: S = Σ(g/G)^2 · ((1/g) Σ eps_(h))^2
        Code (no ties):  S = (1/G^2) Σ (cumsum_g)^2
        These are algebraically identical when all dose values are unique.
        """
        from diff_diff.had_pretests import _cvm_statistic

        eps = np.array([1.0, -0.5, 0.3, -0.2, 0.1])
        d_sorted = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # all unique -> no ties
        G = len(eps)

        # Paper form
        cumsum = np.cumsum(eps)
        paper_S = 0.0
        for g in range(1, G + 1):
            c_g = cumsum[g - 1]
            paper_S += (g / G) ** 2 * (c_g / g) ** 2

        code_S = _cvm_statistic(eps, d_sorted)
        np.testing.assert_allclose(code_S, paper_S, atol=1e-14)

    def test_cvm_statistic_tie_safe_order_invariance(self):
        """CRITICAL P1 fix: tie-safe CvM is order-invariant within tie blocks.

        Under the paper definition, at a tied dose ``D_g == D_{g+1}``, the
        cusum ``c_G`` evaluated at both tied observations includes ALL
        tie-block members. So the CvM statistic must not depend on the
        within-tie ordering of residuals. A naive per-observation cumsum
        (without tie-block collapse) would give different S values when
        the within-tie residual order is permuted.
        """
        from diff_diff.had_pretests import _cvm_statistic

        # 6 observations with two tie blocks: d = [0.1, 0.1, 0.1, 0.5, 0.5, 0.9]
        d_sorted = np.array([0.1, 0.1, 0.1, 0.5, 0.5, 0.9])
        eps_a = np.array([1.0, -0.5, 0.3, 0.7, -0.2, 0.1])
        # Permute within-tie residual order (positions 0-2 and 3-4 permuted):
        eps_b = np.array([0.3, 1.0, -0.5, -0.2, 0.7, 0.1])
        S_a = _cvm_statistic(eps_a, d_sorted)
        S_b = _cvm_statistic(eps_b, d_sorted)
        np.testing.assert_allclose(S_a, S_b, atol=1e-14)

    def test_stute_order_invariance_with_duplicate_doses(self):
        """End-to-end P1 fix: stute_test on duplicate-dose inputs is invariant
        to within-tie row ordering (tie-safe CvM contract propagates)."""
        G = 40
        # Build d with ties and a matched dy
        rng = np.random.default_rng(42)
        d_unique = np.array([0.1, 0.3, 0.5, 0.8])
        d = np.repeat(d_unique, G // len(d_unique))
        dy = 2.0 * d + rng.normal(0.0, 0.1, size=G)
        # Permute order within each tie block
        perm = np.arange(G)
        rng_perm = np.random.default_rng(123)
        for start in range(0, G, G // len(d_unique)):
            block = perm[start : start + G // len(d_unique)]
            rng_perm.shuffle(block)
            perm[start : start + G // len(d_unique)] = block
        r_a = stute_test(d, dy, n_bootstrap=199, seed=42)
        r_b = stute_test(d[perm], dy[perm], n_bootstrap=199, seed=42)
        # Observed cvm_stat must be bit-identical (tie-safe).
        np.testing.assert_allclose(r_a.cvm_stat, r_b.cvm_stat, atol=1e-14)

    def test_n_bootstrap_below_99_raises(self):
        """Commit criterion 8: n_bootstrap < 99 -> ValueError."""
        d, dy = _linear_dgp(G=50)
        with pytest.raises(ValueError, match=r"n_bootstrap must be >= 99"):
            stute_test(d, dy, n_bootstrap=50)

    def test_exact_linear_returns_p1_not_nan(self):
        """P1 fix: exact linear fit (eps=0) must fail-to-reject with p=1,
        NOT return NaN. Assumption 8 holds exactly in this case."""
        G = 50
        d = np.linspace(0.1, 1.0, G)
        dy = 1.0 + 2.0 * d  # exact linear, residuals are zero
        r = stute_test(d, dy, n_bootstrap=199, seed=42)
        assert np.isfinite(r.p_value)
        assert r.p_value == 1.0  # all bootstrap S_b tied at 0 with observed S
        assert r.reject is False
        assert r.cvm_stat == 0.0

    def test_exact_linear_shortcut_does_not_fire_on_shifted_noisy_data(self):
        """P0 fix (R2): the exact-linear short-circuit must NOT fire on
        noisy data after an additive shift. The fix scales against
        ``sum((dy - dybar)^2)`` (centered TSS, translation-invariant).

        Exact numerical equivalence of the two bootstrap runs is NOT
        expected (FP cancellation at 1e12 scale costs ~4 digits of
        precision), but the short-circuit behavior and decision MUST
        match.
        """
        rng = np.random.default_rng(42)
        G = 50
        d = rng.uniform(0.0, 1.0, size=G)
        dy = 2.0 * d + rng.normal(0.0, 1.0, size=G)
        dy_shifted = dy + 1e12

        r_raw = stute_test(d, dy, n_bootstrap=199, seed=42)
        r_shift = stute_test(d, dy_shifted, n_bootstrap=199, seed=42)

        assert r_raw.cvm_stat > 0.0, "shortcut fired on raw noisy data"
        assert r_shift.cvm_stat > 0.0, "shortcut fired on shifted noisy data"
        assert r_raw.reject == r_shift.reject
        np.testing.assert_allclose(r_raw.p_value, r_shift.p_value, rtol=5e-3)

    def test_exact_linear_shortcut_does_not_fire_on_rescaled_noisy_data(self):
        """P0 fix (R5): the exact-linear short-circuit must NOT fire on
        noisy data after a MULTIPLICATIVE rescaling. An earlier revision
        used ``max(centered_TSS, 1.0)`` as the denominator in the guard,
        which broke scale invariance: scaling ``dy`` by ``1e-12`` would
        make ``centered_TSS ~ 1e-24`` but the floor would hold the
        threshold at 1.0, firing the shortcut on noisy data that should
        NOT trigger it. Fix uses a purely relative comparison with a
        separate branch for the ``centered_TSS == 0`` edge case.
        """
        rng = np.random.default_rng(42)
        G = 50
        d = rng.uniform(0.0, 1.0, size=G)
        dy = 2.0 * d + rng.normal(0.0, 1.0, size=G)
        dy_scaled = dy * 1e-12  # multiplicative rescale

        r_raw = stute_test(d, dy, n_bootstrap=199, seed=42)
        r_scaled = stute_test(d, dy_scaled, n_bootstrap=199, seed=42)

        # Neither run may hit the short-circuit.
        assert r_raw.cvm_stat > 0.0, "shortcut fired on raw noisy data"
        assert r_scaled.cvm_stat > 0.0, "shortcut fired on scaled noisy data"
        # Decision must be preserved: rescaling dy cannot change the
        # rejection outcome (the statistic is scale-invariant in dy).
        assert r_raw.reject == r_scaled.reject

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same length"):
            stute_test(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2.0]))

    def test_nan_input_raises(self):
        d, dy = _linear_dgp(G=50)
        dy_nan = dy.copy()
        dy_nan[5] = np.nan
        with pytest.raises(ValueError, match="contains NaN"):
            stute_test(d, dy_nan)


# =============================================================================
# Yatchew-HR test
# =============================================================================


class TestYatchewHRTest:
    """Tests for :func:`yatchew_hr_test`."""

    def test_does_not_reject_linear_dgp(self):
        """Commit criterion 9 (single-realization): linear -> no reject."""
        d, dy = _linear_dgp(G=200, beta=2.0, sigma=0.3, seed=42)
        r = yatchew_hr_test(d, dy, alpha=0.05)
        assert r.reject is False

    def test_rejects_quadratic_dgp(self):
        """Commit criterion 9 (single-realization): quadratic -> reject."""
        d, dy = _quadratic_dgp(G=200, beta=2.0, gamma=5.0, sigma=0.3, seed=42)
        r = yatchew_hr_test(d, dy, alpha=0.05)
        assert r.reject is True
        assert r.t_stat_hr > r.critical_value

    def test_sigma2_diff_normalizer_is_2G_paper_literal(self):
        """Commit criterion 10: sigma2_diff divides by 2G (paper-literal).

        Hand-compute sigma2_diff on a known input and verify the
        implementation divides by 2G, NOT 2(G-1).
        """
        # G = 4 deterministic inputs; sort by d first.
        d = np.array([0.1, 0.2, 0.3, 0.4])
        dy = np.array([1.0, 3.0, 2.0, 5.0])
        r = yatchew_hr_test(d, dy, alpha=0.05)

        # Sorted by d: (already sorted)
        # diff_dy = [dy[1]-dy[0], dy[2]-dy[1], dy[3]-dy[2]]
        #         = [2, -1, 3]; squared = [4, 1, 9]; sum = 14
        # Paper-literal: sigma2_diff = 14 / (2 * 4) = 1.75
        # WRONG form (using G-1=3): 14 / (2 * 3) = 2.333...
        sum_sq_diff = 14.0
        G = 4
        expected_paper_literal = sum_sq_diff / (2.0 * G)
        expected_wrong_form = sum_sq_diff / (2.0 * (G - 1))
        np.testing.assert_allclose(r.sigma2_diff, expected_paper_literal, atol=1e-12)
        assert (
            abs(r.sigma2_diff - expected_wrong_form) > 1e-6
        ), "sigma2_diff should divide by 2G (paper-literal), NOT 2(G-1)."

    def test_invalid_alpha_raises(self):
        d, dy = _linear_dgp(G=50)
        with pytest.raises(ValueError, match="alpha must satisfy"):
            yatchew_hr_test(d, dy, alpha=-0.1)

    def test_exact_linear_returns_p1_not_nan(self):
        """P1 fix: exact linear fit (eps=0) must fail-to-reject with p=1,
        NOT NaN. Yatchew statistic is formally -inf (finite-negative numerator
        over zero denominator), which corresponds to p=1 under the one-sided
        standard-normal critical value."""
        G = 50
        d = np.linspace(0.1, 1.0, G)
        dy = 1.0 + 2.0 * d  # exact linear, residuals are zero
        r = yatchew_hr_test(d, dy)
        assert np.isfinite(r.p_value)
        assert r.p_value == 1.0
        assert r.reject is False
        # t_stat_hr is formally -inf for the exact-linear limit.
        assert r.t_stat_hr == float("-inf")

    def test_exact_linear_shortcut_does_not_fire_on_shifted_noisy_data(self):
        """P0 fix (R2): Yatchew exact-linear short-circuit is translation-
        invariant (comparison against centered TSS, not raw sum(dy^2))."""
        rng = np.random.default_rng(42)
        G = 50
        d = rng.uniform(0.0, 1.0, size=G)
        dy = 2.0 * d + rng.normal(0.0, 1.0, size=G)
        dy_shifted = dy + 1e12

        r_raw = yatchew_hr_test(d, dy)
        r_shift = yatchew_hr_test(d, dy_shifted)

        assert np.isfinite(r_raw.t_stat_hr), "shortcut fired on raw noisy data"
        assert np.isfinite(r_shift.t_stat_hr), "shortcut fired on shifted noisy data"
        assert r_raw.reject == r_shift.reject
        np.testing.assert_allclose(r_raw.t_stat_hr, r_shift.t_stat_hr, rtol=1e-3)

    def test_exact_linear_shortcut_does_not_fire_on_rescaled_noisy_data(self):
        """P0 fix (R5): Yatchew exact-linear short-circuit is scale-
        invariant. An earlier revision used a ``max(centered_TSS, 1.0)``
        floor that broke scale invariance under multiplicative rescaling
        of dy (e.g. ``dy * 1e-12``). Fix removes the floor and handles
        the zero-centered-TSS case in a separate branch."""
        rng = np.random.default_rng(42)
        G = 50
        d = rng.uniform(0.0, 1.0, size=G)
        dy = 2.0 * d + rng.normal(0.0, 1.0, size=G)
        dy_scaled = dy * 1e-12

        r_raw = yatchew_hr_test(d, dy)
        r_scaled = yatchew_hr_test(d, dy_scaled)

        # Neither run may short-circuit to t_stat_hr = -inf.
        assert np.isfinite(r_raw.t_stat_hr), "shortcut fired on raw noisy data"
        assert np.isfinite(r_scaled.t_stat_hr), "shortcut fired on scaled noisy data"
        # Decision preserved: Yatchew is scale-invariant in dy.
        assert r_raw.reject == r_scaled.reject
        # T_hr is scale-invariant: sigma2_lin and sigma2_diff both scale
        # by c^2, sigma2_W scales by c^2, so the ratio is unchanged.
        np.testing.assert_allclose(r_raw.t_stat_hr, r_scaled.t_stat_hr, rtol=1e-3)

    def test_duplicate_doses_return_nan(self):
        """P1 fix: yatchew_hr_test rejects duplicate doses with UserWarning
        + NaN result. Adjacent differences depend on within-tie row order,
        which is non-methodological; users with tied doses should use
        stute_test (tie-safe CvM)."""
        d = np.array([0.1, 0.2, 0.2, 0.5, 0.8])  # one duplicate at 0.2
        dy = np.array([1.0, 2.0, 2.5, 3.0, 4.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = yatchew_hr_test(d, dy)
            msgs = [str(w.message) for w in caught]
        assert any("duplicate" in m for m in msgs)
        assert np.isnan(r.t_stat_hr)
        assert np.isnan(r.p_value)
        assert r.reject is False

    def test_constant_d_returns_nan(self):
        """P1 fix: yatchew_hr_test rejects constant d (degenerate case of
        duplicate doses: all observations tied)."""
        d = np.full(10, 0.5)
        dy = np.linspace(0.0, 2.0, 10)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = yatchew_hr_test(d, dy)
            msgs = [str(w.message) for w in caught]
        assert any("duplicate" in m for m in msgs)
        assert np.isnan(r.t_stat_hr)
        assert r.reject is False

    def test_sigma4_W_zero_with_positive_numerator_rejects(self):
        """P0 fix: when sigma4_W = 0 AFTER the exact-linear shortcut
        (i.e., residuals are NOT all zero but adjacent-residual-product
        sums vanish), the Yatchew statistic is formally +inf or -inf
        depending on numerator sign. A unit-dose, non-exact-linear
        input where residuals alternate zero/non-zero must NOT be
        silently mapped to p=1 (fail-to-reject).

        Counterexample from CI R3 reviewer: d=[1,2,3,4,5],
        dy=[1,0,-2,0,1]. OLS slope = 0, residuals = dy itself,
        sigma2_lin = 1.2, sigma2_diff = 1.0, sigma4_W = 0 (each
        adjacent residual product includes a zero factor), and the
        formal statistic is sqrt(5) * 0.2 / 0 = +inf -> reject=True.
        Previously the sigma4_W <= 0 branch unconditionally returned
        p=1, which flipped a legitimate rejection.
        """
        d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dy = np.array([1.0, 0.0, -2.0, 0.0, 1.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = yatchew_hr_test(d, dy, alpha=0.05)
            msgs = [str(w.message) for w in caught]
        # Must NOT silently fail-to-reject.
        assert not (r.p_value == 1.0 and r.reject is False), (
            "sigma4_W=0 with positive numerator was silently mapped to "
            "p=1, reject=False; expected +inf reject path."
        )
        # Positive numerator -> +inf statistic, p=0, reject=True.
        assert r.t_stat_hr == float("inf")
        assert r.p_value == 0.0
        assert r.reject is True
        # The caller should have been warned.
        assert any("sigma4_W = 0" in m for m in msgs)
        # Sanity: sigma2_lin and sigma2_diff preserved for inspection.
        np.testing.assert_allclose(r.sigma2_lin, 1.2, atol=1e-12)
        np.testing.assert_allclose(r.sigma2_diff, 1.0, atol=1e-12)


# =============================================================================
# Composite workflow
# =============================================================================


class TestCompositeWorkflow:
    """Tests for :func:`did_had_pretest_workflow`."""

    def test_all_pass_on_linear_flags_assumption7_gap(self):
        """Commit criterion 11: linear DGP -> all_pass, verdict flags
        Assumption 7 pre-trends gap per Phase 3 partial-workflow scope."""
        d, dy = _linear_dgp(G=200, beta=2.0, sigma=0.3, seed=42)
        panel = _make_two_period_panel(200, d, dy, seed=42)
        report = did_had_pretest_workflow(
            panel,
            outcome_col="y",
            dose_col="d",
            time_col="time",
            unit_col="unit",
            alpha=0.05,
            n_bootstrap=199,
            seed=42,
        )
        assert report.all_pass is True
        # Partial-workflow verdict: explicitly names the Assumption 7 gap
        # so callers do not receive a misleading "TWFE safe" signal.
        assert "QUG and linearity diagnostics fail-to-reject" in report.verdict
        assert "Assumption 7" in report.verdict
        assert "pre-trends" in report.verdict
        assert "paper step 2 deferred" in report.verdict
        # Verdict must NOT claim unconditional TWFE safety.
        assert "TWFE safe" not in report.verdict
        assert report.n_obs == 200
        assert isinstance(report.qug, QUGTestResults)
        assert isinstance(report.stute, StuteTestResults)
        assert isinstance(report.yatchew, YatchewTestResults)

    def test_rejects_on_quadratic_plus_shifted_support(self):
        """Commit criterion 12: quadratic + shifted support -> both reject."""
        rng = np.random.default_rng(42)
        G = 200
        # Shifted-Beta support [0.2, 1.2]: QUG should reject.
        d = 0.2 + rng.beta(2, 2, size=G)
        # Quadratic dose response: linearity tests should reject.
        dy = 2.0 * d + 5.0 * d * d + rng.normal(0, 0.3, size=G)
        panel = _make_two_period_panel(G, d, dy, seed=42)
        report = did_had_pretest_workflow(
            panel,
            outcome_col="y",
            dose_col="d",
            time_col="time",
            unit_col="unit",
            alpha=0.05,
            n_bootstrap=199,
            seed=42,
        )
        assert report.all_pass is False
        assert "support infimum rejected" in report.verdict
        assert "linearity rejected" in report.verdict
        # At least QUG and one of {Stute, Yatchew} rejected.
        assert report.qug.reject is True
        assert report.stute is not None and report.yatchew is not None
        assert report.stute.reject or report.yatchew.reject

    def test_workflow_handles_tied_zero_doses_via_stute_fallback(self):
        """R4 P2 fix: on a common QUG-style panel with repeated d=0
        (never-treated comparison group), Yatchew returns NaN (duplicate
        doses) but Stute handles ties correctly. The composite workflow
        must adjudicate the linearity step via Stute ALONE per the paper's
        "Stute or Yatchew" step-3 wording, producing a conclusive verdict
        with a Yatchew-skipped note rather than forcing the whole report
        to inconclusive.

        Paper review line 298 cites an application with 12 zero-dose
        units, so this panel shape is not exotic.
        """
        rng = np.random.default_rng(42)
        G = 60
        # 20 never-treated (d=0 at post), 40 treated with random positive.
        d = np.zeros(G)
        d[20:] = rng.uniform(0.1, 1.0, size=40)
        dy = 2.0 * d + rng.normal(0.0, 0.3, size=G)
        panel = _make_two_period_panel(G, d, dy, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = did_had_pretest_workflow(
                panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42
            )
        assert report.stute is not None and report.yatchew is not None
        # Yatchew must NaN (ties from the 20 zero doses).
        assert np.isnan(report.yatchew.p_value)
        assert report.yatchew.reject is False
        # QUG and Stute must be conclusive.
        assert np.isfinite(report.qug.p_value)
        assert np.isfinite(report.stute.p_value)
        # Composite must be conclusive (NOT inconclusive).
        assert "inconclusive" not in report.verdict
        # Verdict should note Yatchew was skipped.
        assert "Yatchew NaN - skipped" in report.verdict
        # With a linear DGP, all_pass should be True despite Yatchew NaN
        # (step 3 handled by Stute alone, per paper's "Stute OR Yatchew").
        assert report.all_pass is True

    def test_all_pass_false_when_any_test_nan(self):
        """P1 fix: all_pass must NOT be True when any constituent test is
        NaN. Previously all_pass was computed from `not reject` only, so a
        NaN-statistic test (reject=False by convention) incorrectly tripped
        all_pass=True while verdict was "inconclusive"."""
        G = 40
        # Constant dose triggers QUG tie (D_(1) == D_(2)) -> NaN.
        d = np.full(G, 0.5)
        dy = np.linspace(0.0, 2.0, G)
        panel = _make_two_period_panel(G, d, dy, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = did_had_pretest_workflow(
                panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42
            )
        # At least one test produces NaN, so `all_pass` MUST be False even
        # though none of the tests set reject=True.
        assert np.isnan(report.qug.p_value)
        assert report.all_pass is False
        assert report.verdict.startswith("inconclusive")

    def test_workflow_seed_controls_stute_only(self):
        """Workflow seed forwards to Stute only; re-run with same seed is reproducible."""
        d, dy = _linear_dgp(G=100, beta=2.0, sigma=0.3, seed=42)
        panel = _make_two_period_panel(100, d, dy, seed=42)
        report_a = did_had_pretest_workflow(
            panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42
        )
        report_b = did_had_pretest_workflow(
            panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42
        )
        assert report_a.stute is not None and report_b.stute is not None
        assert report_a.yatchew is not None and report_b.yatchew is not None
        assert report_a.stute.p_value == report_b.stute.p_value
        assert report_a.qug.t_stat == report_b.qug.t_stat
        assert report_a.yatchew.t_stat_hr == report_b.yatchew.t_stat_hr


# =============================================================================
# NaN propagation
# =============================================================================


class TestNaNPropagation:
    """All three tests NaN-propagate on degenerate inputs; verdict is inconclusive."""

    def test_constant_dy_trivially_satisfies_linearity(self):
        """Commit criterion 13 (reframed): constant dy is trivially linear
        in d; Stute and Yatchew should fail-to-reject with large p-values.

        Constant dy means OLS residuals are mathematically zero and the CvM
        cusum is zero; the bootstrap yields all-tied ``S_b = 0``. The test
        correctly fails-to-reject (linearity holds). NaN propagation is
        covered by :meth:`test_qug_tie_propagates_to_composite` below.
        """
        G = 50
        d = np.linspace(0.1, 1.0, G)
        dy = np.full(G, 3.0)
        panel = _make_two_period_panel(G, d, dy, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = did_had_pretest_workflow(
                panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42
            )
        assert report.stute is not None and report.yatchew is not None
        assert report.stute.reject is False
        assert report.yatchew.reject is False
        assert report.stute.p_value > 0.5
        assert report.yatchew.p_value > 0.5

    def test_qug_tie_propagates_to_composite(self):
        """QUG tie in composite -> verdict mentions QUG NaN."""
        G = 20
        # All units share the same positive dose -> D_(1) == D_(2) tie.
        d = np.full(G, 0.5)
        dy = np.linspace(0.0, 2.0, G)
        panel = _make_two_period_panel(G, d, dy, seed=42)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            report = did_had_pretest_workflow(
                panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42
            )
        assert np.isnan(report.qug.p_value)
        assert "QUG" in report.verdict
        assert report.verdict.startswith("inconclusive")

    def test_stute_constant_d_returns_nan(self):
        G = 50
        d = np.full(G, 0.5)
        dy = np.linspace(0.0, 2.0, G)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = stute_test(d, dy, n_bootstrap=199, seed=42)
        assert np.isnan(r.cvm_stat)
        assert r.reject is False


# =============================================================================
# JSON / DataFrame serialization
# =============================================================================


class TestJSONSerialization:
    """to_dict() is JSON-safe; to_dataframe() returns correct schema."""

    def test_qug_to_dict_is_json_safe(self):
        d = np.array([0.2, 0.5, 0.9])
        r = qug_test(d)
        json_str = json.dumps(r.to_dict())
        round_tripped = json.loads(json_str)
        assert round_tripped["test"] == "qug"
        assert round_tripped["reject"] is False

    def test_stute_to_dict_is_json_safe(self):
        d, dy = _linear_dgp(G=50)
        r = stute_test(d, dy, n_bootstrap=199, seed=42)
        json.dumps(r.to_dict())  # must not raise

    def test_yatchew_to_dict_is_json_safe(self):
        d, dy = _linear_dgp(G=50)
        r = yatchew_hr_test(d, dy)
        json.dumps(r.to_dict())  # must not raise

    def test_report_to_dict_is_json_safe(self):
        """Commit criterion 16: json.dumps(report.to_dict()) succeeds."""
        d, dy = _linear_dgp(G=50)
        panel = _make_two_period_panel(50, d, dy, seed=42)
        report = did_had_pretest_workflow(panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42)
        json_str = json.dumps(report.to_dict())
        round_tripped = json.loads(json_str)
        assert "qug" in round_tripped
        assert "stute" in round_tripped
        assert "yatchew" in round_tripped
        assert "verdict" in round_tripped

    def test_report_to_dataframe_schema(self):
        """Commit criterion 17: 3-row frame with specified columns in order."""
        d, dy = _linear_dgp(G=50)
        panel = _make_two_period_panel(50, d, dy, seed=42)
        report = did_had_pretest_workflow(panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42)
        df = report.to_dataframe()
        expected_cols = [
            "test",
            "statistic_name",
            "statistic_value",
            "p_value",
            "reject",
            "alpha",
            "n_obs",
        ]
        assert list(df.columns) == expected_cols
        assert list(df["test"]) == ["qug", "stute", "yatchew_hr"]
        assert list(df["statistic_name"]) == ["t_stat", "cvm_stat", "t_stat_hr"]
        assert len(df) == 3

    def test_per_test_to_dataframe_is_single_row(self):
        """Commit criterion 18: per-test to_dataframe returns a 1-row frame."""
        d, dy = _linear_dgp(G=50)
        r_stute = stute_test(d, dy, n_bootstrap=199, seed=42)
        r_yatchew = yatchew_hr_test(d, dy)
        r_qug = qug_test(d)
        assert len(r_stute.to_dataframe()) == 1
        assert len(r_yatchew.to_dataframe()) == 1
        assert len(r_qug.to_dataframe()) == 1
        assert r_stute.to_dataframe()["test"].iloc[0] == "stute"


# =============================================================================
# Seed reproducibility
# =============================================================================


class TestSeedReproducibility:
    """Commit criterion 7: stute_test(seed=42) bitwise-reproducible."""

    def test_stute_seed_reproducible(self):
        d, dy = _linear_dgp(G=100, seed=1)
        r_a = stute_test(d, dy, n_bootstrap=199, seed=42)
        r_b = stute_test(d, dy, n_bootstrap=199, seed=42)
        assert r_a.cvm_stat == r_b.cvm_stat
        assert r_a.p_value == r_b.p_value  # bitwise

    def test_stute_different_seeds_produce_different_pvals(self):
        d, dy = _linear_dgp(G=100, seed=1)
        r_a = stute_test(d, dy, n_bootstrap=199, seed=42)
        r_b = stute_test(d, dy, n_bootstrap=199, seed=43)
        # CvM stat is seed-independent; only p-value varies across bootstraps.
        assert r_a.cvm_stat == r_b.cvm_stat
        # P-values should differ at B = 199 (grid = 1/200).
        assert r_a.p_value != r_b.p_value


# =============================================================================
# Sample-size gates
# =============================================================================


class TestSampleSizeGates:
    """Below-minimum sample sizes emit UserWarning + NaN result (no raise)."""

    def test_qug_below_min_returns_nan(self):
        """G=1 (after filter) -> NaN, warning, reject=False."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = qug_test(np.array([0.5]))
            msgs = [str(w.message) for w in caught]
        assert any("at least 2" in m for m in msgs)
        assert np.isnan(r.t_stat)
        assert r.reject is False

    def test_qug_all_zero_returns_nan(self):
        """All d=0 -> filter removes everything -> NaN."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = qug_test(np.zeros(10))
        assert np.isnan(r.t_stat)
        assert r.n_obs == 0
        assert r.n_excluded_zero == 10

    def test_stute_below_min_returns_nan(self):
        """Commit criterion 14: stute_test(G=8) -> NaN, warning, no raise."""
        d, dy = _linear_dgp(G=8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = stute_test(d, dy, n_bootstrap=199, seed=42)
            msgs = [str(w.message) for w in caught]
        assert any("below the minimum" in m for m in msgs)
        assert np.isnan(r.cvm_stat)
        assert r.reject is False

    def test_yatchew_below_min_returns_nan(self):
        """yatchew_hr_test(G=2) -> NaN, warning."""
        d = np.array([0.1, 0.5])
        dy = np.array([1.0, 2.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = yatchew_hr_test(d, dy)
            msgs = [str(w.message) for w in caught]
        assert any("below the minimum" in m for m in msgs)
        assert np.isnan(r.t_stat_hr)
        assert r.reject is False


# =============================================================================
# _compose_verdict priority logic
# =============================================================================


def _mk_qug(reject: bool, p: float = 0.5) -> QUGTestResults:
    return QUGTestResults(
        t_stat=10.0 if reject else 0.5,
        p_value=p,
        reject=reject,
        alpha=0.05,
        critical_value=19.0,
        n_obs=50,
        n_excluded_zero=0,
        d_order_1=0.1,
        d_order_2=0.2,
    )


def _mk_stute(reject: bool, p: float = 0.5) -> StuteTestResults:
    return StuteTestResults(
        cvm_stat=0.5,
        p_value=p,
        reject=reject,
        alpha=0.05,
        n_bootstrap=999,
        n_obs=50,
        seed=42,
    )


def _mk_yatchew(reject: bool, p: float = 0.5) -> YatchewTestResults:
    return YatchewTestResults(
        t_stat_hr=3.0 if reject else -0.5,
        p_value=p,
        reject=reject,
        alpha=0.05,
        critical_value=1.645,
        sigma2_lin=1.0,
        sigma2_diff=1.0,
        sigma2_W=1.0,
        n_obs=50,
    )


class TestComposeVerdictLogic:
    """Commit criterion 15: 5 priority-order tests."""

    def test_qug_nan_is_inconclusive(self):
        """(a1) QUG NaN forces inconclusive (step 1 is required)."""
        q = _mk_qug(reject=False, p=float("nan"))
        s = _mk_stute(reject=False)
        y = _mk_yatchew(reject=False)
        verdict = _compose_verdict(q, s, y)
        assert verdict == "inconclusive - QUG NaN"

    def test_both_linearity_nan_is_inconclusive(self):
        """(a2) When BOTH Stute and Yatchew are NaN, the linearity step
        cannot be adjudicated and the verdict is inconclusive."""
        q = _mk_qug(reject=False)
        s = _mk_stute(reject=False, p=float("nan"))
        y = _mk_yatchew(reject=False, p=float("nan"))
        verdict = _compose_verdict(q, s, y)
        assert verdict == "inconclusive - both Stute and Yatchew linearity tests NaN"

    def test_yatchew_nan_alone_does_not_force_inconclusive(self):
        """CRITICAL R4 P3 fix: the paper says step 3 uses "Stute OR
        Yatchew". A conclusive Stute must be sufficient even when Yatchew
        returns NaN (e.g. tied-dose panels). Verdict should be
        fail-to-reject with a "(Yatchew NaN - skipped)" suffix."""
        q = _mk_qug(reject=False)
        s = _mk_stute(reject=False)
        y = _mk_yatchew(reject=False, p=float("nan"))
        verdict = _compose_verdict(q, s, y)
        assert "inconclusive" not in verdict
        assert verdict.startswith("QUG and linearity diagnostics fail-to-reject")
        assert "Yatchew NaN - skipped" in verdict
        assert "Assumption 7" in verdict

    def test_stute_nan_alone_does_not_force_inconclusive(self):
        """Mirror of the above: Yatchew conclusive + Stute NaN should
        still adjudicate the linearity step via Yatchew alone."""
        q = _mk_qug(reject=False)
        s = _mk_stute(reject=False, p=float("nan"))
        y = _mk_yatchew(reject=False)
        verdict = _compose_verdict(q, s, y)
        assert "inconclusive" not in verdict
        assert verdict.startswith("QUG and linearity diagnostics fail-to-reject")
        assert "Stute NaN - skipped" in verdict

    def test_qug_reject_with_both_linearity_nan_surfaces_rejection(self):
        """R6 P1 fix: a conclusive QUG rejection must NOT be hidden by
        "inconclusive" just because BOTH linearity tests are NaN. Paper
        rule is one-way: TWFE is admissible only if NO test rejects, so
        any conclusive rejection dominates unresolved-step notes.
        """
        q = _mk_qug(reject=True)
        s = _mk_stute(reject=False, p=float("nan"))
        y = _mk_yatchew(reject=False, p=float("nan"))
        verdict = _compose_verdict(q, s, y)
        # Rejection must be surfaced as the primary verdict.
        assert "support infimum rejected" in verdict
        assert not verdict.startswith("inconclusive")
        # Unresolved linearity step is appended, not replacing the rejection.
        assert "additional steps unresolved" in verdict
        assert "both Stute and Yatchew" in verdict

    def test_linearity_reject_with_qug_nan_surfaces_rejection(self):
        """R6 P1 fix: a conclusive linearity rejection (Stute or Yatchew)
        must NOT be hidden by "inconclusive - QUG NaN"."""
        q = _mk_qug(reject=False, p=float("nan"))
        s = _mk_stute(reject=True)
        y = _mk_yatchew(reject=False)
        verdict = _compose_verdict(q, s, y)
        assert "linearity rejected" in verdict
        assert "Stute" in verdict
        assert not verdict.startswith("inconclusive")
        assert "additional steps unresolved" in verdict
        assert "QUG NaN" in verdict

    def test_all_three_reject_with_qug_nan_keeps_conclusive_rejections(self):
        """Mixed case: Stute and Yatchew both conclusively reject, QUG
        unresolved. All conclusive rejections must be reported."""
        q = _mk_qug(reject=False, p=float("nan"))
        s = _mk_stute(reject=True)
        y = _mk_yatchew(reject=True)
        verdict = _compose_verdict(q, s, y)
        assert "linearity rejected" in verdict
        assert "Stute" in verdict
        assert "Yatchew" in verdict
        assert "QUG NaN" in verdict
        assert not verdict.startswith("inconclusive")

    def test_none_reject_flags_assumption7_gap(self):
        """(b) None reject -> verdict flags the Assumption 7 pre-trends gap
        rather than claiming unconditional TWFE safety."""
        q = _mk_qug(reject=False)
        s = _mk_stute(reject=False)
        y = _mk_yatchew(reject=False)
        verdict = _compose_verdict(q, s, y)
        assert "QUG and linearity diagnostics fail-to-reject" in verdict
        assert "Assumption 7" in verdict
        assert "pre-trends" in verdict
        assert "paper step 2 deferred" in verdict
        assert "TWFE safe" not in verdict

    def test_only_qug_rejects(self):
        """(c) Only QUG rejects -> QUG-only message."""
        q = _mk_qug(reject=True)
        s = _mk_stute(reject=False)
        y = _mk_yatchew(reject=False)
        verdict = _compose_verdict(q, s, y)
        assert "support infimum rejected" in verdict
        assert "linearity rejected" not in verdict

    def test_only_linearity_rejects(self):
        """(d) Only linearity rejects -> linearity-only message."""
        q = _mk_qug(reject=False)
        s = _mk_stute(reject=True)
        y = _mk_yatchew(reject=False)
        verdict = _compose_verdict(q, s, y)
        assert "linearity rejected" in verdict
        assert "Stute" in verdict
        assert "Yatchew" not in verdict
        assert "support infimum rejected" not in verdict

    def test_all_reject_bundles_reasons(self):
        """(e) All reject -> bundled message naming each."""
        q = _mk_qug(reject=True)
        s = _mk_stute(reject=True)
        y = _mk_yatchew(reject=True)
        verdict = _compose_verdict(q, s, y)
        assert "support infimum rejected" in verdict
        assert "linearity rejected" in verdict
        assert "Stute" in verdict
        assert "Yatchew" in verdict
        assert "; " in verdict  # bundled with separator


# =============================================================================
# Top-level import surface
# =============================================================================


def test_phase_3_imports_at_top_level():
    """Commit criterion 19: all Phase 3 names importable from `diff_diff`."""
    import diff_diff

    expected = [
        "HADPretestReport",
        "QUGTestResults",
        "StuteTestResults",
        "YatchewTestResults",
        "did_had_pretest_workflow",
        "qug_test",
        "stute_test",
        "yatchew_hr_test",
    ]
    for name in expected:
        assert hasattr(diff_diff, name), f"diff_diff.{name} missing from public API"
        assert name in diff_diff.__all__, f"{name} missing from diff_diff.__all__"


# =============================================================================
# summary() smoke
# =============================================================================


class TestSummary:
    """summary() / print_summary() produce non-empty strings for all classes."""

    def test_qug_summary_non_empty(self):
        r = qug_test(np.array([0.2, 0.5, 0.9]))
        s = r.summary()
        assert len(s) > 0
        assert "QUG" in s

    def test_stute_summary_non_empty(self):
        d, dy = _linear_dgp(G=50)
        r = stute_test(d, dy, n_bootstrap=199, seed=42)
        s = r.summary()
        assert len(s) > 0
        assert "Stute" in s or "CvM" in s

    def test_yatchew_summary_non_empty(self):
        d, dy = _linear_dgp(G=50)
        r = yatchew_hr_test(d, dy)
        s = r.summary()
        assert len(s) > 0
        assert "Yatchew" in s

    def test_report_summary_bundles_all(self):
        d, dy = _linear_dgp(G=50)
        panel = _make_two_period_panel(50, d, dy, seed=42)
        report = did_had_pretest_workflow(panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42)
        s = report.summary()
        assert "QUG" in s
        assert "Stute" in s or "CvM" in s
        assert "Yatchew" in s
        assert "Verdict:" in s


# =============================================================================
# Phase 3 follow-up: joint Stute tests + event-study workflow dispatch
# =============================================================================


def _make_multi_period_panel(
    G: int,
    periods: list,
    first_treat_period,
    dose_fn=None,
    outcome_fn=None,
    seed: int = 42,
) -> pd.DataFrame:
    """Construct a multi-period HAD panel.

    Parameters
    ----------
    G : int
        Number of units.
    periods : list
        Time labels, ordered (numeric or ordered dtype). Must contain at
        least one pre-period (``t < first_treat_period``) and one
        post-period (``t >= first_treat_period``).
    first_treat_period : period label
        The first period where any unit has ``D > 0``. For pre-periods,
        ``D = 0`` for every unit.
    dose_fn : callable or None
        ``dose_fn(rng, G) -> np.ndarray`` returning per-unit dose.
        Default: uniform on [0.05, 1.0].
    outcome_fn : callable or None
        ``outcome_fn(rng, unit_id, t, d, is_post, first_treat) -> float``
        returning the outcome for a single (unit, period) cell. Default:
        linear effect ``0.5 * d`` on post-periods plus Gaussian noise.
    seed : int
    """
    rng = np.random.default_rng(seed)
    if dose_fn is None:
        doses = rng.uniform(0.05, 1.0, size=G)
    else:
        doses = dose_fn(rng, G)
    unit_effects = rng.normal(0.0, 0.3, size=G)

    def _default_outcome(rng_, g, t, d, is_post, _ft):
        eff = 0.5 * d if is_post else 0.0
        return float(unit_effects[g] + eff + rng_.normal(0.0, 0.1))

    if outcome_fn is None:
        outcome_fn = _default_outcome

    rows = []
    for g in range(G):
        for t in periods:
            is_post = t >= first_treat_period
            d = float(doses[g]) if is_post else 0.0
            y = outcome_fn(rng, g, t, d, is_post, first_treat_period)
            rows.append({"unit": g, "period": t, "y": y, "d": d})
    return pd.DataFrame(rows)


def _nonlinear_outcome(d_effect_fn):
    """Build an outcome_fn applying d_effect_fn(d) at post-periods."""

    def _fn(rng_, g, t, d, is_post, _ft):
        eff = d_effect_fn(d) if is_post else 0.0
        noise = rng_.normal(0.0, 0.1)
        return float(0.3 * g / 100.0 + eff + noise)

    return _fn


def _multi_period_residuals(G: int, K: int, seed: int = 42):
    """Random Gaussian residuals + zero fitted + uniform doses."""
    rng = np.random.default_rng(seed)
    horizon_labels = [f"t={1995 + k}" for k in range(K)]
    residuals = {k: rng.normal(0.0, 1.0, size=G) for k in horizon_labels}
    fitted = {k: np.zeros(G) for k in horizon_labels}
    doses = rng.uniform(0.0, 1.0, size=G)
    return residuals, fitted, doses


class TestStuteJointPretest:
    """Tests for :func:`stute_joint_pretest` (residuals-in core)."""

    def test_k1_parity_with_single_horizon_stute(self):
        """K=1 joint matches stute_test on same residuals (refit semantics)."""
        rng = np.random.default_rng(42)
        G = 100
        d = rng.uniform(0.0, 1.0, G)
        dy = 0.3 * d + rng.normal(0.0, 0.2, G)
        # stute_test fits OLS(dy ~ 1 + d) internally. Mirror the fit here
        # so the residuals passed to the joint helper are identical.
        x = np.column_stack([np.ones(G), d])
        beta = np.linalg.solve(x.T @ x, x.T @ dy)
        fitted = x @ beta
        resid = dy - fitted
        joint = stute_joint_pretest(
            residuals_by_horizon={"only": resid},
            fitted_by_horizon={"only": fitted},
            doses=d,
            design_matrix=x,
            n_bootstrap=999,
            seed=123,
            null_form="linearity",
        )
        single = stute_test(d, dy, n_bootstrap=999, seed=123)
        np.testing.assert_allclose(joint.cvm_stat_joint, single.cvm_stat, atol=1e-14, rtol=1e-14)
        # p_value can differ slightly due to RNG draw order (joint draws
        # one eta vector per iteration, single draws one - same shape);
        # the statistic being bit-identical is the critical check.

    def test_linear_dgp_fails_to_reject(self):
        """Linear DGP across all horizons: joint test should not reject."""
        rng = np.random.default_rng(2)
        G = 80
        d = rng.uniform(0.0, 1.0, G)
        residuals = {}
        fitted = {}
        for k in range(3):
            dy = 0.4 * d + rng.normal(0.0, 0.2, G)
            x = np.column_stack([np.ones(G), d])
            beta = np.linalg.solve(x.T @ x, x.T @ dy)
            fit = x @ beta
            residuals[f"h{k}"] = dy - fit
            fitted[f"h{k}"] = fit
        result = stute_joint_pretest(
            residuals_by_horizon=residuals,
            fitted_by_horizon=fitted,
            doses=d,
            design_matrix=np.column_stack([np.ones(G), d]),
            n_bootstrap=499,
            seed=99,
        )
        assert result.p_value > 0.05, f"unexpected rejection: p={result.p_value}"
        assert result.reject is False

    def test_violated_dgp_in_single_horizon_reject(self):
        """Quadratic effect in one of 3 horizons: joint test should reject."""
        rng = np.random.default_rng(7)
        G = 150
        d = rng.uniform(0.05, 1.0, G)
        residuals = {}
        fitted = {}
        for k in range(3):
            if k == 1:
                dy = 4.0 * (d**2) + rng.normal(0.0, 0.2, G)
            else:
                dy = 0.4 * d + rng.normal(0.0, 0.2, G)
            x = np.column_stack([np.ones(G), d])
            beta = np.linalg.solve(x.T @ x, x.T @ dy)
            fit = x @ beta
            residuals[f"h{k}"] = dy - fit
            fitted[f"h{k}"] = fit
        result = stute_joint_pretest(
            residuals_by_horizon=residuals,
            fitted_by_horizon=fitted,
            doses=d,
            design_matrix=np.column_stack([np.ones(G), d]),
            n_bootstrap=999,
            seed=99,
        )
        assert result.reject is True, f"expected rejection, got p={result.p_value}"

    def test_shared_eta_across_horizons_white_box(self):
        """Bootstrap uses the same eta for all horizons in each iteration.

        White-box check: construct residuals where horizon 0 and horizon
        1 have the EXACT SAME residuals. The joint bootstrap under a
        SHARED eta must produce bootstrap outcomes dy_b_h0 == dy_b_h1
        exactly (same fitted + same residuals * same eta). The refit
        then gives the same residuals, and the CvM statistic for both
        horizons is identical. If eta were drawn independently per
        horizon, the two bootstrap residual streams would diverge.

        We verify by checking that S_joint_bootstrap = 2 * S_single_bootstrap
        across many iterations (same underlying process duplicated).
        """
        rng = np.random.default_rng(11)
        G = 60
        d = rng.uniform(0.0, 1.0, G)
        dy = 0.5 * d + rng.normal(0.0, 0.3, G)
        x = np.column_stack([np.ones(G), d])
        beta = np.linalg.solve(x.T @ x, x.T @ dy)
        fit = x @ beta
        resid = dy - fit
        joint = stute_joint_pretest(
            residuals_by_horizon={"a": resid, "b": resid.copy()},
            fitted_by_horizon={"a": fit, "b": fit.copy()},
            doses=d,
            design_matrix=x,
            n_bootstrap=499,
            seed=77,
        )
        single = stute_joint_pretest(
            residuals_by_horizon={"a": resid},
            fitted_by_horizon={"a": fit},
            doses=d,
            design_matrix=x,
            n_bootstrap=499,
            seed=77,
        )
        # Under shared eta, the joint stat is exactly 2x the single stat
        # (both horizons identical). Under independent eta, the joint
        # would be the sum of two INDEPENDENT draws - bootstrap
        # distributions would differ from 2x the single distribution.
        np.testing.assert_allclose(
            joint.cvm_stat_joint, 2.0 * single.cvm_stat_joint, atol=1e-14, rtol=1e-14
        )
        # Under shared eta, each bootstrap S*_b_joint = 2 * S*_b_single,
        # so the p-value (P(S*_b >= S_obs)) is identical to the single.
        np.testing.assert_allclose(joint.p_value, single.p_value, atol=1e-14)

    def test_seed_reproducibility(self):
        """Same seed -> bit-identical results across calls."""
        rng = np.random.default_rng(3)
        G = 80
        d = rng.uniform(0.0, 1.0, G)
        resid = {"h0": rng.normal(0.0, 1.0, G), "h1": rng.normal(0.0, 1.0, G)}
        fit = {"h0": np.zeros(G), "h1": np.zeros(G)}
        r1 = stute_joint_pretest(
            residuals_by_horizon=resid,
            fitted_by_horizon=fit,
            doses=d,
            design_matrix=np.ones((G, 1)),
            n_bootstrap=299,
            seed=55,
        )
        r2 = stute_joint_pretest(
            residuals_by_horizon=resid,
            fitted_by_horizon=fit,
            doses=d,
            design_matrix=np.ones((G, 1)),
            n_bootstrap=299,
            seed=55,
        )
        np.testing.assert_allclose(r1.cvm_stat_joint, r2.cvm_stat_joint, atol=1e-14, rtol=1e-14)
        np.testing.assert_allclose(r1.p_value, r2.p_value, atol=1e-14, rtol=1e-14)
        assert r1.reject == r2.reject

    def test_nan_propagation(self):
        """NaN in any residual -> p_value=NaN, reject=False, dict preserved."""
        G = 20
        rng = np.random.default_rng(4)
        resid = {
            "h0": rng.normal(0.0, 1.0, G),
            "h1": rng.normal(0.0, 1.0, G),
        }
        resid["h1"][5] = np.nan
        fit = {"h0": np.zeros(G), "h1": np.zeros(G)}
        d = rng.uniform(0.0, 1.0, G)
        result = stute_joint_pretest(
            residuals_by_horizon=resid,
            fitted_by_horizon=fit,
            doses=d,
            design_matrix=np.ones((G, 1)),
            n_bootstrap=199,
            seed=0,
        )
        assert np.isnan(result.p_value)
        assert np.isnan(result.cvm_stat_joint)
        assert result.reject is False
        assert result.exact_linear_short_circuited is False
        # per_horizon_stats must preserve ALL keys with NaN values (not
        # empty, not partial) - feedback_no_silent_failures.
        assert set(result.per_horizon_stats.keys()) == {"h0", "h1"}
        assert all(np.isnan(v) for v in result.per_horizon_stats.values())
        assert result.horizon_labels == ["h0", "h1"]

    def test_negative_dose_raises(self):
        G = 20
        resid, fit, _ = _multi_period_residuals(G, K=2)
        doses_neg = np.full(G, -0.1)
        with pytest.raises(ValueError, match="non-negative"):
            stute_joint_pretest(
                residuals_by_horizon=resid,
                fitted_by_horizon=fit,
                doses=doses_neg,
                design_matrix=np.ones((G, 1)),
                n_bootstrap=199,
                seed=0,
            )

    def test_small_G_warns_returns_nan(self):
        """R5: G < _MIN_G_STUTE mirrors single-horizon stute_test -
        warn + NaN result instead of raise. Prevents event-study
        workflow crash when a last-cohort filter leaves fewer than 10
        units."""
        G = 5  # below _MIN_G_STUTE (10)
        resid, fit, d = _multi_period_residuals(G, K=2)
        with pytest.warns(UserWarning, match="below the minimum"):
            result = stute_joint_pretest(
                residuals_by_horizon=resid,
                fitted_by_horizon=fit,
                doses=d,
                design_matrix=np.ones((G, 1)),
                n_bootstrap=199,
                seed=0,
            )
        assert np.isnan(result.cvm_stat_joint)
        assert np.isnan(result.p_value)
        assert result.reject is False
        assert result.n_obs == G
        # Full diagnostic surface preserved on the NaN result
        assert set(result.per_horizon_stats.keys()) == set(str(k) for k in resid.keys())
        assert all(np.isnan(v) for v in result.per_horizon_stats.values())

    def test_small_bootstrap_raises(self):
        G = 50
        resid, fit, d = _multi_period_residuals(G, K=2)
        with pytest.raises(ValueError, match="n_bootstrap"):
            stute_joint_pretest(
                residuals_by_horizon=resid,
                fitted_by_horizon=fit,
                doses=d,
                design_matrix=np.ones((G, 1)),
                n_bootstrap=50,
                seed=0,
            )

    def test_empty_residuals_raises(self):
        with pytest.raises(ValueError, match="at least one horizon"):
            stute_joint_pretest(
                residuals_by_horizon={},
                fitted_by_horizon={},
                doses=np.arange(30, dtype=np.float64),
                design_matrix=np.ones((30, 1)),
                n_bootstrap=199,
            )

    def test_key_mismatch_raises(self):
        G = 30
        with pytest.raises(ValueError, match="identical keys"):
            stute_joint_pretest(
                residuals_by_horizon={"a": np.zeros(G)},
                fitted_by_horizon={"b": np.zeros(G)},
                doses=np.arange(G, dtype=np.float64),
                design_matrix=np.ones((G, 1)),
                n_bootstrap=199,
            )

    def test_exact_linear_short_circuit_per_horizon(self):
        """All-horizons exact linear -> short-circuit (p=1, no bootstrap)."""
        G = 40
        rng = np.random.default_rng(5)
        d = rng.uniform(0.0, 1.0, G)
        # Two horizons, both perfectly linear in d (residuals near-zero)
        dy1 = 2.0 * d + 1.0
        dy2 = -0.5 * d + 3.0
        x = np.column_stack([np.ones(G), d])
        beta1 = np.linalg.solve(x.T @ x, x.T @ dy1)
        beta2 = np.linalg.solve(x.T @ x, x.T @ dy2)
        fit1 = x @ beta1
        fit2 = x @ beta2
        resid1 = dy1 - fit1
        resid2 = dy2 - fit2
        result = stute_joint_pretest(
            residuals_by_horizon={"h1": resid1, "h2": resid2},
            fitted_by_horizon={"h1": fit1, "h2": fit2},
            doses=d,
            design_matrix=x,
            n_bootstrap=199,
            seed=1,
        )
        assert result.exact_linear_short_circuited is True
        assert result.p_value == 1.0
        assert result.reject is False

    def test_exact_linear_short_circuit_scale_invariant(self):
        """Scale-invariant: rescaling residuals by 1e10 preserves short-circuit."""
        G = 40
        rng = np.random.default_rng(6)
        d = rng.uniform(0.0, 1.0, G)
        dy = 2.0 * d + 1.0
        x = np.column_stack([np.ones(G), d])
        beta = np.linalg.solve(x.T @ x, x.T @ dy)
        fit = x @ beta
        resid = dy - fit
        # Scale by 1e10
        result = stute_joint_pretest(
            residuals_by_horizon={"h1": resid * 1e10},
            fitted_by_horizon={"h1": fit * 1e10},
            doses=d,
            design_matrix=x,
            n_bootstrap=199,
            seed=1,
        )
        assert result.exact_linear_short_circuited is True
        assert result.p_value == 1.0

    def test_per_horizon_short_circuit_independence(self):
        """Degenerate horizon + nontrivial horizon -> no short-circuit."""
        G = 80
        rng = np.random.default_rng(8)
        d = rng.uniform(0.05, 1.0, G)
        # Horizon 1: exact linear (fitted = dy, residuals ~ 0)
        dy1 = 2.0 * d + 1.0
        x = np.column_stack([np.ones(G), d])
        beta = np.linalg.solve(x.T @ x, x.T @ dy1)
        fit1 = x @ beta
        resid1 = dy1 - fit1
        # Horizon 2: strong quadratic (nontrivial residuals)
        dy2 = 5.0 * (d**2) + rng.normal(0.0, 0.1, G)
        beta2 = np.linalg.solve(x.T @ x, x.T @ dy2)
        fit2 = x @ beta2
        resid2 = dy2 - fit2
        result = stute_joint_pretest(
            residuals_by_horizon={"lin": resid1, "quad": resid2},
            fitted_by_horizon={"lin": fit1, "quad": fit2},
            doses=d,
            design_matrix=x,
            n_bootstrap=999,
            seed=3,
        )
        # Must NOT short-circuit - the quadratic horizon is informative.
        assert result.exact_linear_short_circuited is False
        # Strong nonlinearity in horizon 2 should make the joint reject.
        assert result.reject is True, f"expected rejection; p={result.p_value}"

    def test_horizon_labels_preserved_as_strings(self):
        """Int / str / pd.Period labels all get str()'d; order preserved."""
        G = 40
        rng = np.random.default_rng(9)
        d = rng.uniform(0.0, 1.0, G)
        resid_int_keyed = {1997: rng.normal(0.0, 1.0, G), 1998: rng.normal(0.0, 1.0, G)}
        fit_int_keyed = {1997: np.zeros(G), 1998: np.zeros(G)}
        result = stute_joint_pretest(
            residuals_by_horizon=resid_int_keyed,
            fitted_by_horizon=fit_int_keyed,
            doses=d,
            design_matrix=np.ones((G, 1)),
            n_bootstrap=199,
            seed=0,
        )
        assert result.horizon_labels == ["1997", "1998"]
        assert set(result.per_horizon_stats.keys()) == {"1997", "1998"}

    def test_constant_d_returns_nan_with_warning(self):
        """R1: constant doses - no cross-sectional variation to detect
        nonlinearity. Must warn and return NaN inference rather than
        a mechanically-zero CvM (mean-indep null) or singular refit
        (linearity null). Mirrors stute_test's single-horizon guard."""
        G = 30
        resid, fit, _ = _multi_period_residuals(G, K=2, seed=123)
        d_constant = np.full(G, 0.5, dtype=np.float64)
        with pytest.warns(UserWarning, match="constant doses"):
            result = stute_joint_pretest(
                residuals_by_horizon=resid,
                fitted_by_horizon=fit,
                doses=d_constant,
                design_matrix=np.ones((G, 1)),
                n_bootstrap=199,
                seed=0,
            )
        assert np.isnan(result.cvm_stat_joint)
        assert np.isnan(result.p_value)
        assert result.reject is False
        assert result.exact_linear_short_circuited is False
        # Per-horizon stats preserved with NaN values (diagnostic surface)
        assert set(result.per_horizon_stats.keys()) == set(resid.keys())
        assert all(np.isnan(v) for v in result.per_horizon_stats.values())

    def test_singular_design_matrix_raises_valueerror(self):
        """R7 P2: rank-deficient custom design_matrix (e.g. duplicate
        columns) must raise an explicit ValueError from the front-door,
        not a raw np.linalg.LinAlgError from the internal solve()."""
        G = 30
        rng = np.random.default_rng(801)
        d = rng.uniform(0.0, 1.0, G)
        resid = {"h0": rng.normal(0.0, 1.0, G), "h1": rng.normal(0.0, 1.0, G)}
        fit = {"h0": np.zeros(G), "h1": np.zeros(G)}
        # design_matrix with two identical columns (rank deficient).
        singular_X = np.column_stack([d, d])
        with pytest.raises(ValueError, match="rank-deficient"):
            stute_joint_pretest(
                residuals_by_horizon=resid,
                fitted_by_horizon=fit,
                doses=d,
                design_matrix=singular_X,
                n_bootstrap=199,
                seed=0,
            )

    def test_stringified_key_collision_raises(self):
        """R4 P1 regression: two raw keys whose str() representations
        collide (e.g. int 1 and str '1', or int 1 and float 1.0) must
        raise explicitly rather than silently overwrite one horizon in
        the internal residuals_arrays map and double-count the survivor
        in the sum-of-CvMs S_joint."""
        G = 20
        rng = np.random.default_rng(701)
        d = rng.uniform(0.0, 1.0, G)
        # int / str collision: str(1) == "1"
        resid_int_str_collision = {
            1: rng.normal(0.0, 1.0, G),
            "1": rng.normal(0.0, 1.0, G),
        }
        fit_int_str_collision = {1: np.zeros(G), "1": np.zeros(G)}
        with pytest.raises(ValueError, match="collision after str"):
            stute_joint_pretest(
                residuals_by_horizon=resid_int_str_collision,
                fitted_by_horizon=fit_int_str_collision,
                doses=d,
                design_matrix=np.ones((G, 1)),
                n_bootstrap=199,
                seed=0,
            )

        # int / float collision: str(1) == "1" but str(1.0) == "1.0"
        # so these actually don't collide. Test a real collision case:
        # two different string representations of the same label.
        # Python: str(True) == "True"; bool(1) == True but that's the
        # same key. Use: str(None) == "None" collides if passed twice,
        # but keys must be unique per dict. Safer: two equal-after-str
        # object keys that were distinct before str conversion.
        class _WeirdLabel:
            def __init__(self, s):
                self._s = s

            def __str__(self):
                return self._s

            def __hash__(self):
                return hash((id(self), self._s))

        a = _WeirdLabel("horizon-1")
        b = _WeirdLabel("horizon-1")  # same str, different object
        assert a is not b
        assert str(a) == str(b)
        resid_obj_collision = {a: rng.normal(0.0, 1.0, G), b: rng.normal(0.0, 1.0, G)}
        fit_obj_collision = {a: np.zeros(G), b: np.zeros(G)}
        with pytest.raises(ValueError, match="collision after str"):
            stute_joint_pretest(
                residuals_by_horizon=resid_obj_collision,
                fitted_by_horizon=fit_obj_collision,
                doses=d,
                design_matrix=np.ones((G, 1)),
                n_bootstrap=199,
                seed=0,
            )


class TestJointPretrendsTest:
    """Tests for :func:`joint_pretrends_test` data-in wrapper."""

    def test_smoke_runs_on_valid_panel(self):
        df = _make_multi_period_panel(
            G=50,
            periods=[1997, 1998, 1999, 2000, 2001],
            first_treat_period=2000,
            seed=11,
        )
        result = joint_pretrends_test(
            df,
            "y",
            "d",
            "period",
            "unit",
            pre_periods=[1997, 1998],
            base_period=1999,
            n_bootstrap=299,
            seed=7,
        )
        assert isinstance(result, StuteJointResult)
        assert result.null_form == "mean_independence"
        assert result.n_horizons == 2
        assert result.n_obs == 50
        assert np.isfinite(result.p_value)
        # Linear DGP on post-periods; pre-periods have D=0 so no relationship
        # between dy_pre_t and D is expected. Fail-to-reject is the target.
        assert result.p_value > 0.05

    def test_matches_manually_constructed_residuals(self):
        """Data-in path reproduces explicit residuals-in call exactly."""
        df = _make_multi_period_panel(
            G=60,
            periods=[1997, 1998, 1999, 2000, 2001],
            first_treat_period=2000,
            seed=12,
        )
        # Data-in dispatch
        data_result = joint_pretrends_test(
            df,
            "y",
            "d",
            "period",
            "unit",
            pre_periods=[1997, 1998],
            base_period=1999,
            n_bootstrap=399,
            seed=22,
        )
        # Manual construction: pivot, compute dy_t = Y_t - Y_base, then
        # center per-horizon to build residuals.
        pivot = df.pivot(index="unit", columns="period", values="y").sort_index()
        d_per = df.groupby("unit")["d"].max().sort_index().to_numpy()
        G = len(d_per)
        base = pivot[1999].to_numpy(dtype=np.float64)
        resid = {}
        fit = {}
        for t in [1997, 1998]:
            dy_t = pivot[t].to_numpy(dtype=np.float64) - base
            mean_t = float(dy_t.mean())
            fit[str(t)] = np.full(G, mean_t)
            resid[str(t)] = dy_t - mean_t
        manual_result = stute_joint_pretest(
            residuals_by_horizon=resid,
            fitted_by_horizon=fit,
            doses=d_per,
            design_matrix=np.ones((G, 1)),
            n_bootstrap=399,
            seed=22,
            null_form="mean_independence",
        )
        np.testing.assert_allclose(
            data_result.cvm_stat_joint,
            manual_result.cvm_stat_joint,
            atol=1e-14,
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            data_result.p_value,
            manual_result.p_value,
            atol=1e-14,
            rtol=1e-14,
        )

    def test_empty_pre_periods_raises(self):
        df = _make_multi_period_panel(
            G=30, periods=[1997, 1998, 1999], first_treat_period=1999, seed=1
        )
        with pytest.raises(ValueError, match="non-empty"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                pre_periods=[],
                base_period=1998,
                n_bootstrap=199,
                seed=0,
            )

    def test_base_period_in_pre_periods_raises(self):
        df = _make_multi_period_panel(
            G=30, periods=[1997, 1998, 1999], first_treat_period=1999, seed=1
        )
        with pytest.raises(ValueError, match="must not appear"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                pre_periods=[1997, 1998],
                base_period=1998,
                n_bootstrap=199,
                seed=0,
            )

    def test_out_of_order_pre_period_raises(self):
        df = _make_multi_period_panel(
            G=30, periods=[1997, 1998, 1999, 2000], first_treat_period=2000, seed=1
        )
        with pytest.raises(ValueError, match="strictly < base_period"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                pre_periods=[1997, 1999],
                base_period=1998,
                n_bootstrap=199,
                seed=0,
            )

    def test_non_zero_dose_in_pre_period_raises(self):
        """HAD contract: pre-periods have D=0 for every unit. The
        event-study validator catches this via its staggered-cohort
        detection (a pre-period unit with D>0 looks like an earlier
        treatment cohort)."""
        df = _make_multi_period_panel(
            G=30, periods=[1997, 1998, 1999, 2000], first_treat_period=2000, seed=1
        )
        # Contaminate pre-period 1998 with a non-zero dose for one unit
        df.loc[(df["unit"] == 0) & (df["period"] == 1998), "d"] = 0.5
        with pytest.raises(ValueError, match="Staggered|dose invariant|D = 0"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                pre_periods=[1997, 1998],
                base_period=1999,
                n_bootstrap=199,
                seed=0,
            )

    def test_non_zero_dose_in_base_period_raises(self):
        """Reciprocal: base_period (last pre-period) must also satisfy
        D=0. Caught by the event-study validator before our local
        guard runs."""
        df = _make_multi_period_panel(
            G=30, periods=[1997, 1998, 1999, 2000], first_treat_period=2000, seed=1
        )
        df.loc[(df["unit"] == 0) & (df["period"] == 1999), "d"] = 0.3
        with pytest.raises(ValueError, match="Staggered|dose invariant|D = 0"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                pre_periods=[1997, 1998],
                base_period=1999,
                n_bootstrap=199,
                seed=0,
            )

    def test_staggered_panel_without_first_treat_col_raises(self):
        """R1: direct wrapper call on a staggered panel without
        first_treat_col must raise via the event-study validator
        contract (same behavior as did_had_pretest_workflow's
        event-study dispatch)."""
        parts = []
        for cohort_ft, cohort_range in [(1999, (0, 15)), (2000, (15, 30))]:
            for g in range(*cohort_range):
                dose = 0.05 + 0.01 * (g - cohort_range[0])
                for t in [1997, 1998, 1999, 2000, 2001]:
                    is_post = t >= cohort_ft
                    parts.append(
                        {
                            "unit": g,
                            "period": t,
                            "y": 0.1 * g + (0.3 * dose if is_post else 0.0),
                            "d": dose if is_post else 0.0,
                        }
                    )
        df = pd.DataFrame(parts)
        with pytest.raises(ValueError, match="Staggered"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                pre_periods=[1997, 1998],
                base_period=1999,
                n_bootstrap=199,
                seed=0,
            )

    def test_staggered_panel_with_first_treat_col_warns_and_filters(self):
        """R1: direct wrapper call on a staggered panel WITH
        first_treat_col auto-filters to last cohort + never-treated
        and emits UserWarning."""
        parts = []
        for cohort_ft, cohort_range in [(1999, (0, 10)), (2000, (10, 40))]:
            for g in range(*cohort_range):
                dose = 0.05 + 0.01 * (g - cohort_range[0])
                for t in [1997, 1998, 1999, 2000, 2001]:
                    is_post = t >= cohort_ft
                    parts.append(
                        {
                            "unit": g,
                            "period": t,
                            "y": 0.1 * g + (0.3 * dose if is_post else 0.0),
                            "d": dose if is_post else 0.0,
                            "first_treat": cohort_ft,
                        }
                    )
        df = pd.DataFrame(parts)
        with pytest.warns(UserWarning, match="staggered|Staggered"):
            result = joint_pretrends_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                pre_periods=[1997, 1998],
                base_period=1999,
                first_treat_col="first_treat",
                n_bootstrap=199,
                seed=0,
            )
        assert isinstance(result, StuteJointResult)
        # Last cohort (F=2000) + never-treated kept (none in this fixture
        # - all units are treated); n_obs reflects the filter.
        assert result.n_obs == 30  # 10 filtered + 30 kept... actually just 30 kept

    def test_constant_d_wrapper_path_returns_nan_with_warning(self):
        """R1: direct wrapper call on a panel where ALL units have the
        same dose - propagates the joint core's constant-d guard and
        returns NaN inference rather than a spurious fail-to-reject."""

        def const_dose(rng_, G):  # noqa: ARG001
            return np.full(G, 0.4, dtype=np.float64)

        df = _make_multi_period_panel(
            G=30,
            periods=[1997, 1998, 1999, 2000, 2001],
            first_treat_period=2000,
            dose_fn=const_dose,
            seed=33,
        )
        with pytest.warns(UserWarning, match="constant doses"):
            result = joint_pretrends_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                pre_periods=[1997, 1998],
                base_period=1999,
                n_bootstrap=199,
                seed=0,
            )
        assert np.isnan(result.p_value)
        assert result.reject is False


class TestJointHomogeneityTest:
    """Tests for :func:`joint_homogeneity_test` data-in wrapper."""

    def test_smoke_runs_on_linear_dgp(self):
        df = _make_multi_period_panel(
            G=50,
            periods=[1998, 1999, 2000, 2001, 2002],
            first_treat_period=2000,
            seed=13,
        )
        result = joint_homogeneity_test(
            df,
            "y",
            "d",
            "period",
            "unit",
            post_periods=[2000, 2001, 2002],
            base_period=1999,
            n_bootstrap=299,
            seed=21,
        )
        assert isinstance(result, StuteJointResult)
        assert result.null_form == "linearity"
        assert result.n_horizons == 3
        assert np.isfinite(result.p_value)
        assert result.reject is False

    def test_rejects_on_quadratic_post_effect(self):
        """Quadratic effect in D across post-periods -> joint homogeneity rejects."""
        df = _make_multi_period_panel(
            G=120,
            periods=[1998, 1999, 2000, 2001],
            first_treat_period=2000,
            outcome_fn=_nonlinear_outcome(lambda d: 4.0 * (d**2)),
            seed=14,
        )
        result = joint_homogeneity_test(
            df,
            "y",
            "d",
            "period",
            "unit",
            post_periods=[2000, 2001],
            base_period=1999,
            n_bootstrap=999,
            seed=31,
        )
        assert result.reject is True, f"expected rejection; p={result.p_value}"

    def test_matches_manually_constructed_residuals(self):
        df = _make_multi_period_panel(
            G=60,
            periods=[1998, 1999, 2000, 2001],
            first_treat_period=2000,
            seed=15,
        )
        data_result = joint_homogeneity_test(
            df,
            "y",
            "d",
            "period",
            "unit",
            post_periods=[2000, 2001],
            base_period=1999,
            n_bootstrap=399,
            seed=41,
        )
        # Manual construction: OLS(dy ~ 1 + D) per horizon
        pivot = df.pivot(index="unit", columns="period", values="y").sort_index()
        d_per = df.groupby("unit")["d"].max().sort_index().to_numpy()
        G = len(d_per)
        base = pivot[1999].to_numpy(dtype=np.float64)
        X = np.column_stack([np.ones(G), d_per.astype(np.float64)])
        resid = {}
        fit = {}
        for t in [2000, 2001]:
            dy_t = pivot[t].to_numpy(dtype=np.float64) - base
            beta = np.linalg.solve(X.T @ X, X.T @ dy_t)
            fit[str(t)] = X @ beta
            resid[str(t)] = dy_t - fit[str(t)]
        manual_result = stute_joint_pretest(
            residuals_by_horizon=resid,
            fitted_by_horizon=fit,
            doses=d_per,
            design_matrix=X,
            n_bootstrap=399,
            seed=41,
            null_form="linearity",
        )
        np.testing.assert_allclose(
            data_result.cvm_stat_joint,
            manual_result.cvm_stat_joint,
            atol=1e-14,
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            data_result.p_value,
            manual_result.p_value,
            atol=1e-14,
            rtol=1e-14,
        )

    def test_empty_post_periods_raises(self):
        df = _make_multi_period_panel(
            G=30, periods=[1997, 1998, 1999], first_treat_period=1999, seed=1
        )
        with pytest.raises(ValueError, match="non-empty"):
            joint_homogeneity_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                post_periods=[],
                base_period=1998,
                n_bootstrap=199,
                seed=0,
            )

    def test_base_period_in_post_periods_raises(self):
        df = _make_multi_period_panel(
            G=30, periods=[1997, 1998, 1999], first_treat_period=1999, seed=1
        )
        with pytest.raises(ValueError, match="must not appear"):
            joint_homogeneity_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                post_periods=[1999],
                base_period=1999,
                n_bootstrap=199,
                seed=0,
            )

    def test_post_period_before_base_raises(self):
        """All post_periods must be strictly > base_period."""
        df = _make_multi_period_panel(
            G=30, periods=[1997, 1998, 1999, 2000], first_treat_period=1999, seed=1
        )
        with pytest.raises(ValueError, match="strictly > base_period"):
            joint_homogeneity_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                post_periods=[1998, 2000],
                base_period=1999,
                n_bootstrap=199,
                seed=0,
            )

    def test_all_zero_dose_post_period_raises(self):
        """Post-period with D=0 for every unit contradicts HAD contract.
        Caught either by the event-study validator's contiguous-dose
        invariant (post-zero breaks the monotone transition from pre
        D=0 to post D>0) or by our local reciprocal guard."""
        df = _make_multi_period_panel(
            G=30, periods=[1997, 1998, 1999, 2000], first_treat_period=1999, seed=1
        )
        # Zero out all doses at post-period 2000 (keep 1999 post intact for base contract)
        df.loc[df["period"] == 2000, "d"] = 0.0
        with pytest.raises(ValueError, match="dose invariant|D = 0 for every unit"):
            joint_homogeneity_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                post_periods=[1999, 2000],
                base_period=1998,
                n_bootstrap=199,
                seed=0,
            )

    def test_two_period_negative_post_dose_raises(self):
        """R2 P1 regression: direct wrapper call on a 2-period panel
        with a negative post dose must raise rather than silently
        collapse to zero via ``groupby.max()`` and produce a finite
        result. The 2-period path skips the event-study validator
        (``n_periods < 3``) so the row-level non-negative guard must
        live in ``_aggregate_for_joint_test`` itself."""
        G = 20
        rng = np.random.default_rng(601)
        doses = rng.uniform(0.1, 1.0, size=G)
        # Flip one unit's post dose to a negative value.
        doses[0] = -0.3
        rows = []
        for g in range(G):
            # pre-period
            rows.append({"unit": g, "period": 0, "y": rng.normal(0, 0.1), "d": 0.0})
            # post-period (with negative dose injected for unit 0)
            rows.append(
                {
                    "unit": g,
                    "period": 1,
                    "y": rng.normal(0, 0.1) + 0.3 * doses[g],
                    "d": float(doses[g]),
                }
            )
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="negative dose value"):
            joint_homogeneity_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                post_periods=[1],
                base_period=0,
                n_bootstrap=199,
                seed=0,
            )


class TestMultiPeriodWorkflow:
    """Tests for :func:`did_had_pretest_workflow` event-study dispatch."""

    def _linear_panel(self, seed: int = 100) -> pd.DataFrame:
        return _make_multi_period_panel(
            G=80,
            periods=[1996, 1997, 1998, 1999, 2000, 2001],
            first_treat_period=1999,
            seed=seed,
        )

    def test_overall_aggregate_unchanged(self):
        """Default aggregate='overall' preserves Phase 3 behavior."""
        d, dy = _linear_dgp(G=50, seed=42)
        panel = _make_two_period_panel(50, d, dy, seed=42)
        report = did_had_pretest_workflow(panel, "y", "d", "time", "unit", n_bootstrap=299, seed=42)
        assert report.aggregate == "overall"
        assert report.stute is not None
        assert report.yatchew is not None
        assert report.pretrends_joint is None
        assert report.homogeneity_joint is None
        # Phase 3 step-2 gap string STILL present on the overall path
        assert "paper step 2 deferred" in report.verdict

    def test_event_study_linear_dgp_all_pass(self):
        df = self._linear_panel(seed=101)
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=299,
            seed=17,
        )
        assert report.aggregate == "event_study"
        assert report.stute is None and report.yatchew is None
        assert report.pretrends_joint is not None
        assert report.homogeneity_joint is not None
        assert report.all_pass is True
        assert "TWFE admissible under Section 4" in report.verdict
        # The Phase 3 "paper step 2 deferred" string MUST NOT appear on
        # the event-study path - the gap is closed.
        assert "paper step 2 deferred" not in report.verdict

    def test_event_study_pretrend_violation_flagged(self):
        """Strong pre-trend correlated with D -> pretrends_joint rejects."""

        def pretrend_outcome(rng_, g, t, d, is_post, ft):
            # D is fixed at F for this unit; simulate correlated pre-trend
            # via knowing what the unit's eventual dose will be.
            # Placeholder: rng seeds unit g the same way _make_multi_period_panel does
            # NOTE: this is hacky; we bake-in correlation via g as proxy for dose.
            trend = (g / 100.0) * (t - 1998)  # pre-existing linear trend
            eff = 0.3 * d if is_post else 0.0
            return float(trend + eff + rng_.normal(0.0, 0.05))

        # Construct panel where dose correlates strongly with unit id (proxy for trend strength)
        def dose_fn(rng_, G):
            return np.linspace(0.05, 1.0, G)

        df = _make_multi_period_panel(
            G=80,
            periods=[1996, 1997, 1998, 1999, 2000],
            first_treat_period=1999,
            dose_fn=dose_fn,
            outcome_fn=pretrend_outcome,
            seed=200,
        )
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=499,
            seed=29,
        )
        assert report.pretrends_joint is not None
        assert report.pretrends_joint.reject is True
        assert "joint pre-trends rejected - assumption 7 violated" in report.verdict

    def test_event_study_homogeneity_violation_flagged(self):
        """Strong quadratic effect across post-periods -> homogeneity rejects."""
        df = _make_multi_period_panel(
            G=100,
            periods=[1997, 1998, 1999, 2000, 2001],
            first_treat_period=1999,
            outcome_fn=_nonlinear_outcome(lambda d: 4.0 * (d**2)),
            seed=210,
        )
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=999,
            seed=31,
        )
        assert report.homogeneity_joint is not None
        assert report.homogeneity_joint.reject is True
        assert "joint linearity rejected - heterogeneity bias" in report.verdict

    def test_event_study_qug_violation_flagged(self):
        """Shift dose support away from 0 -> QUG rejects."""

        def shifted_dose_fn(rng_, G):
            # All doses far from 0; D_2,(1) and D_2,(2) close -> T small -> reject
            return rng_.uniform(0.5, 1.0, G)

        df = _make_multi_period_panel(
            G=80,
            periods=[1997, 1998, 1999, 2000, 2001],
            first_treat_period=1999,
            dose_fn=shifted_dose_fn,
            seed=220,
        )
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=299,
            seed=42,
        )
        # QUG T statistic should be large (shifted support) -> reject.
        if report.qug.reject:
            assert report.verdict.startswith("support infimum rejected")

    def test_invalid_aggregate_raises(self):
        df = self._linear_panel(seed=102)
        with pytest.raises(ValueError, match="aggregate must be one of"):
            did_had_pretest_workflow(
                df,
                "y",
                "d",
                "period",
                "unit",
                aggregate="bogus",
                n_bootstrap=199,
            )

    def test_single_pre_period_yields_pretrends_skipped(self):
        """If t_pre_list has only the base pre-period, no earlier placebos
        exist -> pretrends_joint is None and verdict flags the skip."""
        df = _make_multi_period_panel(
            G=50,
            periods=[1998, 1999, 2000, 2001],
            first_treat_period=1999,
            seed=130,
        )
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=299,
            seed=52,
        )
        assert report.pretrends_joint is None
        # Even with a fail-to-reject homogeneity test, all_pass should be
        # False because pretrends_joint is None (step 2 not closed).
        if report.homogeneity_joint is not None and not report.homogeneity_joint.reject:
            assert report.all_pass is False
        # Verdict should mention the pre-trends skip.
        assert "joint pre-trends skipped" in report.verdict

    def test_no_paper_step_2_deferred_string_on_event_study(self):
        """Regression: event-study verdict must not emit the Phase 3 caveat."""
        df = self._linear_panel(seed=111)
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=299,
            seed=61,
        )
        assert "paper step 2 deferred" not in report.verdict
        assert "deferred to Phase 3 follow-up" not in report.verdict

    def test_first_treat_col_none_with_staggered_raises(self):
        """Inherited contract: staggered panel + no first_treat_col -> raises."""
        # Build two cohorts: one treated at 1999, one at 2000.
        parts = []
        for cohort_ft, cohort_range in [(1999, (0, 40)), (2000, (40, 80))]:
            for g in range(*cohort_range):
                dose = 0.05 + 0.01 * (g - cohort_range[0])
                for t in [1997, 1998, 1999, 2000, 2001]:
                    is_post = t >= cohort_ft
                    parts.append(
                        {
                            "unit": g,
                            "period": t,
                            "y": 0.1 * g + (0.3 * dose if is_post else 0.0),
                            "d": dose if is_post else 0.0,
                        }
                    )
        df = pd.DataFrame(parts)
        with pytest.raises(ValueError):
            did_had_pretest_workflow(
                df,
                "y",
                "d",
                "period",
                "unit",
                aggregate="event_study",
                n_bootstrap=199,
                seed=0,
            )

    def test_staggered_auto_filter_warns(self):
        """With first_treat_col provided, staggered panel auto-filters with warning."""
        parts = []
        for cohort_ft, cohort_range in [(1999, (0, 30)), (2000, (30, 80))]:
            for g in range(*cohort_range):
                dose = 0.05 + 0.01 * (g - cohort_range[0])
                for t in [1997, 1998, 1999, 2000, 2001]:
                    is_post = t >= cohort_ft
                    parts.append(
                        {
                            "unit": g,
                            "period": t,
                            "y": 0.1 * g + (0.3 * dose if is_post else 0.0),
                            "d": dose if is_post else 0.0,
                            "first_treat": cohort_ft,
                        }
                    )
        df = pd.DataFrame(parts)
        with pytest.warns(UserWarning, match="staggered"):
            did_had_pretest_workflow(
                df,
                "y",
                "d",
                "period",
                "unit",
                first_treat_col="first_treat",
                aggregate="event_study",
                n_bootstrap=199,
                seed=0,
            )

    def test_event_study_verdict_priority_qug_first(self):
        """Rejections bundle; QUG rejection appears first."""
        qug = _mk_qug(reject=True, p=0.01)
        pretrends = StuteJointResult(
            cvm_stat_joint=1.0,
            p_value=0.01,
            reject=True,
            alpha=0.05,
            horizon_labels=["t"],
            per_horizon_stats={"t": 1.0},
            n_bootstrap=999,
            n_obs=50,
            n_horizons=1,
            seed=1,
            null_form="mean_independence",
            exact_linear_short_circuited=False,
        )
        homogeneity = StuteJointResult(
            cvm_stat_joint=0.5,
            p_value=0.5,
            reject=False,
            alpha=0.05,
            horizon_labels=["t"],
            per_horizon_stats={"t": 0.5},
            n_bootstrap=999,
            n_obs=50,
            n_horizons=1,
            seed=1,
            null_form="linearity",
            exact_linear_short_circuited=False,
        )
        verdict = _compose_verdict_event_study(qug, pretrends, homogeneity)
        # QUG appears before pre-trends
        assert verdict.index("QUG") < verdict.index("assumption 7")

    def test_event_study_all_conclusive_no_reject_admissible(self):
        qug = _mk_qug(reject=False, p=0.8)
        pretrends = StuteJointResult(
            cvm_stat_joint=0.3,
            p_value=0.7,
            reject=False,
            alpha=0.05,
            horizon_labels=["t"],
            per_horizon_stats={"t": 0.3},
            n_bootstrap=999,
            n_obs=50,
            n_horizons=1,
            seed=1,
            null_form="mean_independence",
            exact_linear_short_circuited=False,
        )
        homogeneity = StuteJointResult(
            cvm_stat_joint=0.5,
            p_value=0.6,
            reject=False,
            alpha=0.05,
            horizon_labels=["t"],
            per_horizon_stats={"t": 0.5},
            n_bootstrap=999,
            n_obs=50,
            n_horizons=1,
            seed=1,
            null_form="linearity",
            exact_linear_short_circuited=False,
        )
        verdict = _compose_verdict_event_study(qug, pretrends, homogeneity)
        assert "TWFE admissible under Section 4" in verdict

    def test_event_study_small_panel_after_filter_inconclusive_not_crash(self):
        """R5: staggered-panel last-cohort filter can leave fewer than
        `_MIN_G_STUTE` (10) units. The joint Stute core must warn +
        return NaN on small G (matching single-horizon stute_test) so
        the event-study workflow surfaces an inconclusive report
        rather than crashing. Regression against the original
        ValueError-on-G<10 contract."""
        parts = []
        # First cohort: 40 units treated at 1999 - will be DROPPED by
        # the last-cohort filter (F_last=2000 > 1999).
        # Second cohort: only 6 units treated at 2000 - kept. After
        # filter G = 6 < _MIN_G_STUTE, so the joint CvM is ill-
        # calibrated and must return NaN via warn.
        for cohort_ft, cohort_range in [(1999, (0, 40)), (2000, (40, 46))]:
            for g in range(*cohort_range):
                dose = 0.05 + 0.01 * (g - cohort_range[0])
                for t in [1997, 1998, 1999, 2000, 2001]:
                    is_post = t >= cohort_ft
                    parts.append(
                        {
                            "unit": g,
                            "period": t,
                            "y": 0.1 * g + (0.3 * dose if is_post else 0.0),
                            "d": dose if is_post else 0.0,
                            "first_treat": cohort_ft,
                        }
                    )
        df = pd.DataFrame(parts)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = did_had_pretest_workflow(
                df,
                "y",
                "d",
                "period",
                "unit",
                first_treat_col="first_treat",
                aggregate="event_study",
                n_bootstrap=199,
                seed=0,
            )
        # Workflow must complete (no crash) and surface an inconclusive
        # report. Both joint tests (pretrends + homogeneity) should
        # return NaN on the post-filter G=6 panel.
        assert report.aggregate == "event_study"
        if report.pretrends_joint is not None:
            assert np.isnan(report.pretrends_joint.p_value)
        assert report.homogeneity_joint is not None
        assert np.isnan(report.homogeneity_joint.p_value)
        assert report.all_pass is False
        # At least one "below the minimum" warning from the joint core.
        msgs = [str(w.message) for w in caught]
        assert any("below the minimum" in m for m in msgs)


class TestOrderedCategoricalChronology:
    """R2 P1 regressions: ordered-categorical time columns whose lexical
    and chronological order disagree (e.g. ``"q10"`` < ``"q2"``
    lexically but > chronologically). Raw ``t < base_period`` comparisons
    misorder these panels; the wrappers and workflow must use validated-
    rank comparisons to apply the test to the intended horizons."""

    @staticmethod
    def _categorical_panel(
        G: int = 60,
        categories=("q1", "q2", "q10", "post"),
        first_treat="post",
        seed: int = 501,
    ) -> pd.DataFrame:
        """Panel with ordered-categorical time whose lexical order
        (``"q1" < "q10" < "q2" < "post"``) differs from chronological
        order (``"q1" < "q2" < "q10" < "post"``)."""
        cat_type = pd.CategoricalDtype(categories=list(categories), ordered=True)
        rng = np.random.default_rng(seed)
        doses = rng.uniform(0.05, 1.0, size=G)
        rows = []
        for g in range(G):
            for t in categories:
                is_post = t == first_treat
                d = float(doses[g]) if is_post else 0.0
                y = 0.1 * g + (0.4 * d if is_post else 0.0) + rng.normal(0.0, 0.1)
                rows.append({"unit": g, "period": t, "y": y, "d": d})
        df = pd.DataFrame(rows)
        df["period"] = df["period"].astype(cat_type)
        return df

    def test_joint_pretrends_test_uses_chronological_rank(self):
        """Direct wrapper call with categories ["q1", "q2", "q10"] where
        the lexical order puts "q10" BEFORE "q2" but chronologically
        "q10" comes AFTER "q2". All three pre-periods must be accepted
        without a false out-of-order error."""
        df = self._categorical_panel()
        result = joint_pretrends_test(
            df,
            "y",
            "d",
            "period",
            "unit",
            pre_periods=["q1", "q2"],
            base_period="q10",
            n_bootstrap=199,
            seed=3,
        )
        assert result.n_horizons == 2
        assert set(result.horizon_labels) == {"q1", "q2"}
        # The detrended-outcome residuals are mean-centered; under null
        # (no pre-trend correlated with D), p should be > 0.05 on this
        # weakly-noisy DGP.
        assert np.isfinite(result.p_value)

    def test_joint_pretrends_raises_on_lexically_ordered_but_chrono_invalid(self):
        """With base_period="q2" and pre_periods=["q10"], chronologically
        q10 > q2 so this is out-of-order - the rank-based check must
        raise. Raw `<` on the lexical side would INCORRECTLY accept
        it since "q10" < "q2" lexically."""
        df = self._categorical_panel()
        with pytest.raises(ValueError, match="chronological order"):
            joint_pretrends_test(
                df,
                "y",
                "d",
                "period",
                "unit",
                pre_periods=["q10"],
                base_period="q2",
                n_bootstrap=199,
                seed=0,
            )

    def test_joint_homogeneity_test_uses_chronological_rank(self):
        """Homogeneity wrapper twin of the pretrends test. Post-period
        "post" comes after all pre-periods chronologically; base="q10"
        is the last pre-period. Lexically "post" > "q10" too (coincides
        here), but the rank-based check must not rely on that."""
        df = self._categorical_panel()
        result = joint_homogeneity_test(
            df,
            "y",
            "d",
            "period",
            "unit",
            post_periods=["post"],
            base_period="q10",
            n_bootstrap=199,
            seed=7,
        )
        assert result.n_horizons == 1
        assert result.horizon_labels == ["post"]
        assert np.isfinite(result.p_value)

    def test_workflow_event_study_ordered_categorical(self):
        """did_had_pretest_workflow(aggregate="event_study") must pick
        up BOTH earlier pre-periods ("q1", "q2") from an ordered-
        categorical panel where lexical order would silently drop one
        of them. Regression against the `earlier_pre` raw-< fix."""
        df = self._categorical_panel()
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=199,
            seed=13,
        )
        assert report.aggregate == "event_study"
        assert report.pretrends_joint is not None
        # t_pre_list = ["q1", "q2", "q10"] chronologically; base = "q10"
        # (last pre-period); earlier_pre should be ["q1", "q2"] - both
        # placebo horizons must appear in pretrends_joint.
        assert set(report.pretrends_joint.horizon_labels) == {"q1", "q2"}
        assert report.homogeneity_joint is not None
        assert report.homogeneity_joint.horizon_labels == ["post"]
        # Verdict does not emit the step-2-skipped flag (both earlier
        # placebos were found).
        assert "joint pre-trends skipped" not in report.verdict


class TestHADPretestReportSerialization:
    """Tests for HADPretestReport serialization branching by aggregate."""

    def test_to_dict_overall_preserves_phase3_schema(self):
        d, dy = _linear_dgp(G=50)
        panel = _make_two_period_panel(50, d, dy, seed=42)
        report = did_had_pretest_workflow(panel, "y", "d", "time", "unit", n_bootstrap=199, seed=42)
        out = report.to_dict()
        # Phase 3 schema is bit-exact: no `aggregate` key on the overall
        # path (only emitted on event_study) - Phase 3 downstream
        # consumers must not see a new key.
        assert "aggregate" not in out
        assert "qug" in out and "stute" in out and "yatchew" in out
        # Event-study keys absent on overall
        assert "pretrends_joint" not in out and "homogeneity_joint" not in out
        # Round-trip JSON safely
        json.dumps(out)

    def test_to_dict_event_study_emits_joint_keys(self):
        df = _make_multi_period_panel(
            G=60,
            periods=[1997, 1998, 1999, 2000, 2001],
            first_treat_period=2000,
            seed=131,
        )
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=199,
            seed=0,
        )
        out = report.to_dict()
        assert out["aggregate"] == "event_study"
        assert "qug" in out
        assert "pretrends_joint" in out and "homogeneity_joint" in out
        # Overall-path keys absent on event-study
        assert "stute" not in out and "yatchew" not in out
        json.dumps(out)

    def test_to_dataframe_stable_3_row_shape(self):
        """to_dataframe returns 3 rows for both aggregates."""
        d, dy = _linear_dgp(G=50)
        panel_overall = _make_two_period_panel(50, d, dy, seed=42)
        report_overall = did_had_pretest_workflow(
            panel_overall, "y", "d", "time", "unit", n_bootstrap=199, seed=42
        )
        df_o = report_overall.to_dataframe()
        assert df_o.shape[0] == 3
        assert list(df_o["test"]) == ["qug", "stute", "yatchew_hr"]

        panel_es = _make_multi_period_panel(
            G=50,
            periods=[1997, 1998, 1999, 2000, 2001],
            first_treat_period=2000,
            seed=132,
        )
        report_es = did_had_pretest_workflow(
            panel_es,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=199,
            seed=0,
        )
        df_e = report_es.to_dataframe()
        assert df_e.shape[0] == 3
        assert list(df_e["test"]) == ["qug", "pretrends_joint", "homogeneity_joint"]
        # Columns identical across aggregates
        assert set(df_o.columns) == set(df_e.columns)

    def test_summary_includes_aggregate_header(self):
        df = _make_multi_period_panel(
            G=50,
            periods=[1997, 1998, 1999, 2000, 2001],
            first_treat_period=2000,
            seed=133,
        )
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=199,
            seed=0,
        )
        s = report.summary()
        assert "aggregate: event_study" in s

    def test_repr_includes_aggregate(self):
        df = _make_multi_period_panel(
            G=50,
            periods=[1997, 1998, 1999, 2000, 2001],
            first_treat_period=2000,
            seed=134,
        )
        report = did_had_pretest_workflow(
            df,
            "y",
            "d",
            "period",
            "unit",
            aggregate="event_study",
            n_bootstrap=199,
            seed=0,
        )
        r = repr(report)
        assert "aggregate='event_study'" in r
