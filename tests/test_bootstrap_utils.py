"""Tests for bootstrap utility edge cases (NaN propagation)."""

import warnings

import numpy as np
import pytest

from diff_diff.bootstrap_utils import (
    compute_effect_bootstrap_stats,
    compute_effect_bootstrap_stats_batch,
    warn_bootstrap_failure_rate,
)


class TestBootstrapStatsNaNPropagation:
    """Regression tests for compute_effect_bootstrap_stats NaN guard."""

    def test_bootstrap_stats_single_valid_sample(self):
        """Single valid sample: ddof=1 produces NaN SE -> all NaN."""
        boot_dist = np.array([1.5])
        with pytest.warns(RuntimeWarning, match="non-finite or zero"):
            se, ci, p_value = compute_effect_bootstrap_stats(
                original_effect=1.0, boot_dist=boot_dist
            )
        assert np.isnan(se)
        assert np.isnan(ci[0])
        assert np.isnan(ci[1])
        assert np.isnan(p_value)

    def test_bootstrap_stats_all_nonfinite(self):
        """All non-finite samples: fails 50% validity check -> all NaN."""
        boot_dist = np.array([np.nan, np.nan, np.inf])
        with pytest.warns(RuntimeWarning):
            se, ci, p_value = compute_effect_bootstrap_stats(
                original_effect=1.0, boot_dist=boot_dist
            )
        assert np.isnan(se)
        assert np.isnan(ci[0])
        assert np.isnan(ci[1])
        assert np.isnan(p_value)

    def test_bootstrap_stats_identical_values(self):
        """All identical values: se=0 -> all NaN."""
        boot_dist = np.array([2.0] * 100)
        with pytest.warns(RuntimeWarning, match="non-finite or zero"):
            se, ci, p_value = compute_effect_bootstrap_stats(
                original_effect=2.0, boot_dist=boot_dist
            )
        assert np.isnan(se)
        assert np.isnan(ci[0])
        assert np.isnan(ci[1])
        assert np.isnan(p_value)

    def test_bootstrap_stats_mostly_valid_but_identical(self):
        """67% valid (passes 50% check) but identical values: se=0 -> all NaN."""
        boot_dist = np.array([2.0, 2.0, np.nan])
        with pytest.warns(RuntimeWarning, match="non-finite or zero"):
            se, ci, p_value = compute_effect_bootstrap_stats(
                original_effect=2.0, boot_dist=boot_dist
            )
        assert np.isnan(se)
        assert np.isnan(ci[0])
        assert np.isnan(ci[1])
        assert np.isnan(p_value)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_nonfinite_original_effect_with_finite_boot_dist(self, bad_value):
        """Non-finite original_effect must return all-NaN even with finite boot_dist."""
        boot_dist = np.arange(100.0)
        se, ci, p_value = compute_effect_bootstrap_stats(
            original_effect=bad_value, boot_dist=boot_dist
        )
        assert np.isnan(se)
        assert np.isnan(ci[0]) and np.isnan(ci[1])
        assert np.isnan(p_value)

    def test_bootstrap_stats_normal_case(self):
        """Normal case with varied values: all fields finite."""
        boot_dist = np.arange(100.0)
        se, ci, p_value = compute_effect_bootstrap_stats(original_effect=50.0, boot_dist=boot_dist)
        assert np.isfinite(se)
        assert se > 0
        assert np.isfinite(ci[0])
        assert np.isfinite(ci[1])
        assert ci[0] < ci[1]
        assert np.isfinite(p_value)
        assert 0 < p_value <= 1


class TestBatchBootstrapStatsWarnings:
    """Tests for warning emission in compute_effect_bootstrap_stats_batch."""

    def test_batch_warns_insufficient_valid_samples(self):
        """Batch function should warn when >50% of bootstrap samples are NaN."""
        rng = np.random.default_rng(42)
        n_bootstrap = 100
        n_effects = 3
        # Column 1 has >50% NaN -> should trigger warning
        matrix = rng.normal(size=(n_bootstrap, n_effects))
        matrix[:60, 1] = np.nan  # 60% NaN

        effects = np.array([1.0, 2.0, 3.0])
        with pytest.warns(RuntimeWarning, match="too few valid"):
            ses, ci_lo, ci_hi, pvals = compute_effect_bootstrap_stats_batch(effects, matrix)
        # Effect 1 (index 1) should be NaN
        assert np.isnan(ses[1])
        # Other effects should be finite
        assert np.isfinite(ses[0])
        assert np.isfinite(ses[2])

    def test_batch_warns_zero_se(self):
        """Batch function should warn when bootstrap SE is zero (identical values)."""
        n_bootstrap = 100
        n_effects = 2
        matrix = np.ones((n_bootstrap, n_effects)) * 5.0  # All identical -> SE=0

        effects = np.array([5.0, 5.0])
        with pytest.warns(RuntimeWarning, match="non-finite or zero"):
            ses, ci_lo, ci_hi, pvals = compute_effect_bootstrap_stats_batch(effects, matrix)
        assert np.isnan(ses[0])
        assert np.isnan(ses[1])

    def test_batch_no_warning_for_normal_case(self):
        """Batch function should not warn when all values are normal."""
        rng = np.random.default_rng(42)
        n_bootstrap = 200
        n_effects = 3
        matrix = rng.normal(size=(n_bootstrap, n_effects))
        effects = np.array([0.5, -0.3, 1.0])

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            ses, ci_lo, ci_hi, pvals = compute_effect_bootstrap_stats_batch(effects, matrix)


class TestWarnBootstrapFailureRate:
    """Proportional failure-rate guard for replicate loops (axis-D)."""

    def test_warns_above_threshold(self):
        """11/200 successes = 94.5% failure rate — must warn."""
        with pytest.warns(UserWarning, match=r"11/200 bootstrap iterations"):
            warn_bootstrap_failure_rate(n_success=11, n_attempted=200, context="test case")

    def test_warning_message_includes_context(self):
        """Context label must appear verbatim in the warning."""
        with pytest.warns(UserWarning, match="TROP global bootstrap") as rec:
            warn_bootstrap_failure_rate(
                n_success=50,
                n_attempted=200,
                context="TROP global bootstrap",
            )
        assert len(rec) == 1
        assert "75.0% failure rate" in str(rec[0].message)

    def test_silent_below_threshold(self):
        """Default threshold=0.05 — 4% failure is below and must not warn."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warn_bootstrap_failure_rate(n_success=960, n_attempted=1000, context="test case")

    def test_silent_on_full_success(self):
        """No warning when every replicate succeeded."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warn_bootstrap_failure_rate(n_success=200, n_attempted=200, context="test case")

    def test_silent_when_n_attempted_zero(self):
        """Degenerate empty call must not divide by zero."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warn_bootstrap_failure_rate(n_success=0, n_attempted=0, context="test case")

    def test_custom_threshold(self):
        """Higher threshold suppresses the 50% case."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            warn_bootstrap_failure_rate(
                n_success=100,
                n_attempted=200,
                context="test case",
                threshold=0.75,
            )

        with pytest.warns(UserWarning, match="50.0% failure rate"):
            warn_bootstrap_failure_rate(
                n_success=100,
                n_attempted=200,
                context="test case",
                threshold=0.25,
            )

    def test_all_failed_warns(self):
        """0/N replicates succeeded — caller handles NaN return, but the warning fires."""
        with pytest.warns(UserWarning, match=r"0/50 bootstrap iterations"):
            warn_bootstrap_failure_rate(n_success=0, n_attempted=50, context="test case")
