"""Tests for HAD Phase 3 pre-test diagnostics (had_pretests.py)."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    QUGTestResults,
    StuteTestResults,
    YatchewTestResults,
    did_had_pretest_workflow,
    qug_test,
    stute_test,
    yatchew_hr_test,
)
from diff_diff.had_pretests import _compose_verdict

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
        """Verify the simplified CvM form matches the paper's stated form.

        Paper: S = Σ(g/G)^2 · ((1/g) Σ eps_(h))^2
        Code:  S = (1/G^2) Σ (cumsum_g)^2
        These are algebraically identical.
        """
        from diff_diff.had_pretests import _cvm_statistic

        eps = np.array([1.0, -0.5, 0.3, -0.2, 0.1])
        G = len(eps)

        # Paper form
        cumsum = np.cumsum(eps)
        paper_S = 0.0
        for g in range(1, G + 1):
            c_g = cumsum[g - 1]
            paper_S += (g / G) ** 2 * (c_g / g) ** 2

        # Simplified form (what the code uses)
        code_S = _cvm_statistic(eps)

        np.testing.assert_allclose(code_S, paper_S, atol=1e-14)

    def test_n_bootstrap_below_99_raises(self):
        """Commit criterion 8: n_bootstrap < 99 -> ValueError."""
        d, dy = _linear_dgp(G=50)
        with pytest.raises(ValueError, match=r"n_bootstrap must be >= 99"):
            stute_test(d, dy, n_bootstrap=50)

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
        assert report.stute.reject or report.yatchew.reject

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

    def test_nan_in_any_test_inconclusive(self):
        """(a) Any NaN p-value -> inconclusive."""
        q = _mk_qug(reject=False)
        s = _mk_stute(reject=False, p=float("nan"))
        y = _mk_yatchew(reject=False)
        verdict = _compose_verdict(q, s, y)
        assert verdict.startswith("inconclusive")
        assert "Stute" in verdict

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
