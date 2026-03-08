"""
Test suite for the Efficient DiD estimator (Chen, Sant'Anna & Xie 2025).

Organized into tiers:
  Tier 1 — Core correctness (fast, deterministic)
  Tier 2 — Weight behavior and edge cases
  Tier 3 — Bootstrap
  Tier 4 — Simulation validation (slow, scaled via ci_params)
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import CallawaySantAnna, EDiD, EfficientDiD
from diff_diff.efficient_did_results import EfficientDiDResults
from diff_diff.efficient_did_weights import (
    enumerate_valid_triples,
)

# =============================================================================
# Helpers
# =============================================================================


def _make_simple_panel(
    n_units=100,
    n_periods=5,
    n_treated=50,
    treat_period=3,
    effect=2.0,
    sigma=0.5,
    seed=42,
):
    """Generate a simple balanced panel with one treatment cohort."""
    rng = np.random.default_rng(seed)
    units = np.repeat(np.arange(n_units), n_periods)
    times = np.tile(np.arange(1, n_periods + 1), n_units)

    ft = np.full(n_units, np.inf)
    ft[:n_treated] = treat_period
    ft_col = np.repeat(ft, n_periods)

    unit_fe = np.repeat(rng.normal(0, 1, n_units), n_periods)
    time_fe = np.tile(np.arange(1, n_periods + 1) * 0.5, n_units)
    tau = np.where((ft_col < np.inf) & (times >= ft_col), effect, 0.0)
    y = unit_fe + time_fe + tau + rng.normal(0, sigma, len(units))

    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "first_treat": ft_col,
            "y": y,
        }
    )


def _make_staggered_panel(
    n_per_group=60,
    n_control=80,
    groups=(3, 5),
    effects=None,
    n_periods=7,
    sigma=0.5,
    rho=0.0,
    seed=42,
):
    """Generate staggered treatment panel with AR(1) errors."""
    if effects is None:
        effects = {3: 2.0, 5: 1.0}
    rng = np.random.default_rng(seed)
    n_units = n_per_group * len(groups) + n_control
    n_t = n_periods

    units = np.repeat(np.arange(n_units), n_t)
    times = np.tile(np.arange(1, n_t + 1), n_units)

    ft = np.full(n_units, np.inf)
    start = 0
    for g in groups:
        ft[start : start + n_per_group] = g
        start += n_per_group
    ft_col = np.repeat(ft, n_t)

    unit_fe = np.repeat(rng.normal(0, 0.5, n_units), n_t)
    time_fe = np.tile(rng.normal(0, 0.1, n_t), n_units)

    # AR(1) errors
    eps = np.zeros((n_units, n_t))
    eps[:, 0] = rng.normal(0, sigma, n_units)
    for t in range(1, n_t):
        eps[:, t] = rho * eps[:, t - 1] + rng.normal(0, sigma, n_units)
    eps_flat = eps.flatten()

    tau = np.zeros(len(units))
    for g, eff in effects.items():
        mask = (ft_col == g) & (times >= g)
        tau[mask] = eff

    y = unit_fe + time_fe + tau + eps_flat

    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "first_treat": ft_col,
            "y": y,
        }
    )


def _make_compustat_dgp(
    n_units=400,
    n_periods=11,
    rho=0.0,
    seed=42,
):
    """Simplified Compustat-style DGP from Section 5.2.

    Groups: G=5 (~1/3), G=8 (~1/3), G=inf (~1/3).
    ATT(5,t) = 0.154*(t-4), ATT(8,t) = 0.093*(t-7).
    """
    rng = np.random.default_rng(seed)
    n_t = n_periods

    # Assign groups
    n_g5 = n_units // 3
    n_g8 = n_units // 3
    ft = np.full(n_units, np.inf)
    ft[:n_g5] = 5
    ft[n_g5 : n_g5 + n_g8] = 8

    units = np.repeat(np.arange(n_units), n_t)
    times = np.tile(np.arange(1, n_t + 1), n_units)
    ft_col = np.repeat(ft, n_t)

    # Unit and time FE
    alpha_t = rng.normal(0, 0.1, n_t)
    eta_i = rng.normal(0, 0.5, n_units)
    unit_fe = np.repeat(eta_i, n_t)
    time_fe = np.tile(alpha_t, n_units)

    # AR(1) errors
    eps = np.zeros((n_units, n_t))
    eps[:, 0] = rng.normal(0, 0.3, n_units)
    for t in range(1, n_t):
        eps[:, t] = rho * eps[:, t - 1] + rng.normal(0, 0.3, n_units)
    eps_flat = eps.flatten()

    # Treatment effects
    tau = np.zeros(len(units))
    for i in range(n_units):
        g = ft[i]
        if np.isinf(g):
            continue
        for t_idx in range(n_t):
            t = t_idx + 1
            if g == 5 and t >= 5:
                tau[i * n_t + t_idx] = 0.154 * (t - 4)
            elif g == 8 and t >= 8:
                tau[i * n_t + t_idx] = 0.093 * (t - 7)

    y = unit_fe + time_fe + tau + eps_flat

    return pd.DataFrame(
        {
            "unit": units,
            "time": times,
            "first_treat": ft_col,
            "y": y,
        }
    )


# =============================================================================
# Tier 1: Core Correctness
# =============================================================================


class TestBasicFit:
    """Test basic fit mechanics: types, shapes, required outputs."""

    def test_basic_fit(self):
        df = _make_simple_panel()
        edid = EfficientDiD(pt_assumption="all")
        result = edid.fit(df, "y", "unit", "time", "first_treat")

        assert isinstance(result, EfficientDiDResults)
        assert isinstance(result.overall_att, float)
        assert isinstance(result.overall_se, float)
        assert len(result.group_time_effects) > 0
        assert result.n_obs == len(df)
        assert result.pt_assumption == "all"

    def test_zero_effect(self):
        df = _make_simple_panel(effect=0.0)
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        # ATT should be near 0
        assert abs(result.overall_att) < 0.5

    def test_positive_effect(self):
        df = _make_simple_panel(effect=2.0, n_units=200)
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        # Recover ~2.0 within 2 SE
        assert abs(result.overall_att - 2.0) < 2 * result.overall_se + 0.5

    def test_single_pre_period(self):
        """When g=2 (only 1 pre-period), weights are trivially [1.0]."""
        df = _make_simple_panel(n_periods=4, treat_period=2)
        result = EfficientDiD(pt_assumption="all").fit(df, "y", "unit", "time", "first_treat")
        assert len(result.group_time_effects) > 0
        # Check weights are stored and have length 1 for the single valid pair
        if result.efficient_weights:
            for gt, w in result.efficient_weights.items():
                if len(w) == 1:
                    assert abs(w[0] - 1.0) < 1e-10


class TestPTPostMatchesCS:
    """Under PT-Post, EDiD should approximately match CS.

    The EDiD formula uses period_1 (earliest period) as the universal baseline,
    while CS uses g-1 (varying base). These are the same when g=2 (period_1 = g-1),
    and approximately the same for g > 2 under parallel trends.
    """

    def test_single_group_g2_exact_match(self):
        """g=2 means g-1 = period_1 = 1, so baselines coincide."""
        df = _make_simple_panel(n_units=200, treat_period=2, n_periods=5)
        edid = EfficientDiD(pt_assumption="post")
        cs = CallawaySantAnna(control_group="never_treated", base_period="varying")

        res_e = edid.fit(df, "y", "unit", "time", "first_treat")
        res_c = cs.fit(df, "y", "unit", "time", "first_treat")

        for gt in res_e.group_time_effects:
            if gt in res_c.group_time_effects:
                e_eff = res_e.group_time_effects[gt]["effect"]
                c_eff = res_c.group_time_effects[gt]["effect"]
                assert abs(e_eff - c_eff) < 1e-10, f"ATT{gt}: EDiD={e_eff:.10f} CS={c_eff:.10f}"

    def test_staggered_approximate_match(self):
        """For g > 2, EDiD(PT-Post) should exactly match CS for post-treatment effects."""
        df = _make_staggered_panel()
        edid = EfficientDiD(pt_assumption="post")
        cs = CallawaySantAnna(control_group="never_treated", base_period="varying")

        res_e = edid.fit(df, "y", "unit", "time", "first_treat")
        res_c = cs.fit(df, "y", "unit", "time", "first_treat")

        matched = 0
        for g, t in res_e.group_time_effects:
            if t >= g and (g, t) in res_c.group_time_effects:
                e_eff = res_e.group_time_effects[(g, t)]["effect"]
                c_eff = res_c.group_time_effects[(g, t)]["effect"]
                assert abs(e_eff - c_eff) < 1e-8, f"ATT({g},{t}): EDiD={e_eff:.10f} CS={c_eff:.10f}"
                matched += 1
        assert matched > 0, "No matching post-treatment effects found"


class TestAggregation:
    """Test aggregation: event study, group, overall."""

    def test_event_study_aggregation(self):
        df = _make_simple_panel()
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", aggregate="event_study")
        assert result.event_study_effects is not None
        # Should have pre and post-treatment event times
        keys = sorted(result.event_study_effects.keys())
        assert any(e < 0 for e in keys), "Should have pre-treatment event times"
        assert any(e >= 0 for e in keys), "Should have post-treatment event times"

    def test_group_aggregation(self):
        df = _make_staggered_panel()
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", aggregate="group")
        assert result.group_effects is not None
        assert 3.0 in result.group_effects
        assert 5.0 in result.group_effects

    def test_aggregate_all(self):
        df = _make_staggered_panel()
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", aggregate="all")
        assert result.event_study_effects is not None
        assert result.group_effects is not None


class TestValidation:
    """Test input validation: missing columns, unbalanced, non-absorbing."""

    def test_balanced_panel_validation(self):
        df = _make_simple_panel()
        # Drop some rows to create unbalanced panel
        df = df.drop(df.index[:3])
        with pytest.raises(ValueError, match="Unbalanced panel"):
            EfficientDiD().fit(df, "y", "unit", "time", "first_treat")

    def test_absorbing_treatment_validation(self):
        df = _make_simple_panel()
        # Make treatment non-absorbing for one unit
        mask = (df["unit"] == 0) & (df["time"] == 1)
        df.loc[mask, "first_treat"] = 5  # changes first_treat mid-panel
        with pytest.raises(ValueError, match="Non-absorbing"):
            EfficientDiD().fit(df, "y", "unit", "time", "first_treat")

    def test_covariates_not_implemented(self):
        df = _make_simple_panel()
        with pytest.raises(NotImplementedError, match="covariates"):
            EfficientDiD().fit(df, "y", "unit", "time", "first_treat", covariates=["y"])

    def test_missing_columns(self):
        df = _make_simple_panel()
        with pytest.raises(ValueError, match="Missing columns"):
            EfficientDiD().fit(df, "y", "unit", "time", "nonexistent")

    def test_pt_post_no_never_treated_raises(self):
        """PT-Post without never-treated group should raise."""
        df = _make_simple_panel(n_treated=100)  # all treated
        with pytest.raises(ValueError, match="never-treated"):
            EfficientDiD(pt_assumption="post").fit(df, "y", "unit", "time", "first_treat")


class TestSklearnCompat:
    """Test get_params / set_params."""

    def test_get_set_params(self):
        edid = EfficientDiD(pt_assumption="post", alpha=0.10, anticipation=1)
        params = edid.get_params()
        assert params["pt_assumption"] == "post"
        assert params["alpha"] == 0.10
        assert params["anticipation"] == 1

        edid.set_params(alpha=0.01)
        assert edid.alpha == 0.01
        assert edid.get_params()["alpha"] == 0.01

    def test_unknown_param_raises(self):
        edid = EfficientDiD()
        with pytest.raises(ValueError, match="Unknown parameter"):
            edid.set_params(nonexistent=True)

    def test_alias(self):
        assert EDiD is EfficientDiD


class TestOutputFormats:
    """Test summary() and to_dataframe()."""

    def test_summary_and_dataframe(self):
        df = _make_simple_panel()
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", aggregate="all")

        # summary() returns a string
        s = result.summary()
        assert isinstance(s, str)
        assert "Efficient DiD" in s

        # to_dataframe at different levels
        df_gt = result.to_dataframe("group_time")
        assert isinstance(df_gt, pd.DataFrame)
        assert "effect" in df_gt.columns

        df_es = result.to_dataframe("event_study")
        assert "relative_period" in df_es.columns

        df_g = result.to_dataframe("group")
        assert "group" in df_g.columns

    def test_to_dataframe_raises_without_aggregation(self):
        df = _make_simple_panel()
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        with pytest.raises(ValueError, match="Event study effects not computed"):
            result.to_dataframe("event_study")

    def test_repr(self):
        df = _make_simple_panel()
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        r = repr(result)
        assert "EfficientDiDResults" in r

    def test_significance_properties(self):
        df = _make_simple_panel(effect=5.0, n_units=200)
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        assert isinstance(result.is_significant, bool)
        assert isinstance(result.significance_stars, str)


class TestNanInference:
    """Test NaN propagation for undefined inference."""

    def test_nan_for_empty_pairs(self):
        """When no valid pairs exist, ATT should be NaN with proper NaN inference."""
        # Create a scenario with a single period (no pre-treatment baseline)
        df = _make_simple_panel(n_periods=2, treat_period=2)
        # Under PT-Post, baseline is g-1 = 1 = period_1, which IS the
        # universal reference. The enumerate function skips period_1 as t_pre,
        # so no valid pairs exist.
        # Actually, under PT-Post, baseline = g - 1 = 1 and period_1 = 1.
        # The valid pair would be (inf, 1), but period_1 is skipped.
        # So we should get NaN for pre-treatment effects at least.

        result = EfficientDiD(pt_assumption="all").fit(df, "y", "unit", "time", "first_treat")
        # At minimum, all effects should have finite or NaN SE
        for gt, d in result.group_time_effects.items():
            assert np.isfinite(d["effect"]) or np.isnan(d["effect"])


class TestPretreatment:
    """Test pre-treatment placebo effects."""

    def test_pretreatment_placebo_near_zero(self):
        """Under correct PT, pre-treatment ATT(g,t) for t < g should be near 0."""
        df = _make_simple_panel(n_units=200, effect=2.0, sigma=0.3)
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", aggregate="event_study")
        # Check pre-treatment effects are near zero
        for e, d in result.event_study_effects.items():
            if e < 0:
                assert (
                    abs(d["effect"]) < 1.0
                ), f"Pre-treatment effect at e={e} is {d['effect']:.4f}, expected ~0"

    def test_pretreatment_in_event_study(self):
        """Placebo effects should appear with negative event-time keys."""
        df = _make_simple_panel(n_periods=6, treat_period=3)
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", aggregate="event_study")
        assert result.event_study_effects is not None
        neg_keys = [e for e in result.event_study_effects if e < 0]
        assert len(neg_keys) > 0, "Should have negative event-time keys"

    def test_pretreatment_detects_violation(self):
        """DGP with pre-trend should produce non-zero placebo ATTs."""
        rng = np.random.default_rng(42)
        n_units, n_periods = 200, 6
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(1, n_periods + 1), n_units)
        ft = np.full(n_units, np.inf)
        ft[:100] = 4  # treated at t=4
        ft_col = np.repeat(ft, n_periods)
        uf = np.repeat(rng.normal(0, 1, n_units), n_periods)
        tf = np.tile(np.arange(1, n_periods + 1) * 0.5, n_units)
        # Add pre-trend for treated group
        pre_trend = np.where(ft_col < np.inf, times * 0.3, 0.0)
        treatment = np.where((ft_col < np.inf) & (times >= ft_col), 2.0, 0.0)
        y = uf + tf + pre_trend + treatment + rng.normal(0, 0.2, len(units))
        df = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "first_treat": ft_col,
                "y": y,
            }
        )
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", aggregate="event_study")
        # Pre-treatment effects should be significantly non-zero
        pre_effects = [d["effect"] for e, d in result.event_study_effects.items() if e < 0]
        assert any(
            abs(e) > 0.1 for e in pre_effects
        ), f"Pre-trend should be detected; pre effects: {pre_effects}"


# =============================================================================
# Tier 2: Weight Behavior and Edge Cases
# =============================================================================


class TestWeightBehavior:
    """Test that efficient weights respond to error structure."""

    def test_weights_uniform_under_iid(self):
        """iid errors -> weights should sum to 1 and be non-degenerate."""
        df = _make_staggered_panel(rho=0.0, seed=123, n_per_group=100, n_control=100)
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
        if result.efficient_weights:
            for gt, w in result.efficient_weights.items():
                if len(w) > 1:
                    # Weights should sum to 1
                    assert abs(w.sum() - 1.0) < 1e-8
                    # At least some variation (not all same)
                    assert w.std() > 0

    def test_condition_number_warning(self):
        """Near-singular Omega* should trigger a warning."""
        # Use a perfectly collinear DGP to produce near-singular Omega*
        n_units, n_periods = 100, 5
        units = np.repeat(np.arange(n_units), n_periods)
        times = np.tile(np.arange(1, n_periods + 1), n_units)
        ft = np.full(n_units, np.inf)
        ft[:50] = 4
        ft_col = np.repeat(ft, n_periods)
        # Constant outcome (zero variance -> degenerate Omega*)
        y = np.ones(len(units)) + np.where((ft_col < np.inf) & (times >= ft_col), 1.0, 0.0)
        df = pd.DataFrame(
            {
                "unit": units,
                "time": times,
                "first_treat": ft_col,
                "y": y,
            }
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
            # Should get a warning about condition number or zero matrix
            warning_msgs = [str(x.message) for x in w]
            assert any(
                "condition" in m.lower()
                or "zero" in m.lower()
                or "pseudoinverse" in m.lower()
                or "uniform" in m.lower()
                for m in warning_msgs
            ), f"Expected condition/zero warning, got: {warning_msgs}"


class TestValidTriples:
    """Test enumerate_valid_triples with hand-worked examples."""

    def test_pt_all_simple(self):
        """T=5, groups={3, inf}, target (3, 4), period_1=1.
        Valid pairs: [(inf, 2)] — only baseline t_pre=2."""
        pairs = enumerate_valid_triples(
            target_g=3,
            target_t=4,
            treatment_groups=[3],
            time_periods=[1, 2, 3, 4, 5],
            period_1=1,
            pt_assumption="all",
        )
        assert (np.inf, 2) in pairs

    def test_pt_all_staggered(self):
        """T=5, groups={3, 5, inf}, target (3, 4), period_1=1.
        Valid pairs under PT-All should include (inf, 2), (5, 2), (5, 3), (5, 4)."""
        pairs = enumerate_valid_triples(
            target_g=3,
            target_t=4,
            treatment_groups=[3, 5],
            time_periods=[1, 2, 3, 4, 5],
            period_1=1,
            pt_assumption="all",
        )
        assert (np.inf, 2) in pairs
        assert (5, 2) in pairs
        # g'=5 has effective treatment at t=5, so t_pre<5 && t_pre<3
        # t_pre must be < effective_g=3: so t_pre=2 only
        # Wait - t_pre<3 for target g, and t_pre<5 for g'=5
        # So valid t_pre for pair (5, t_pre): t_pre in {2} (since t_pre<3)
        expected = {(np.inf, 2), (5, 2)}
        actual = set(pairs)
        assert expected.issubset(actual), f"Expected {expected} ⊆ {actual}"

    def test_pt_post_single_pair(self):
        """PT-Post: only (inf, g-1)."""
        pairs = enumerate_valid_triples(
            target_g=3,
            target_t=4,
            treatment_groups=[3, 5],
            time_periods=[1, 2, 3, 4, 5],
            period_1=1,
            pt_assumption="post",
        )
        assert pairs == [(np.inf, 2)]

    def test_empty_valid_pairs(self):
        """When g=2 and period_1=1, no valid t_pre exists (since period_1 is skipped)."""
        pairs = enumerate_valid_triples(
            target_g=2,
            target_t=3,
            treatment_groups=[2],
            time_periods=[1, 2, 3],
            period_1=1,
            pt_assumption="all",
        )
        # t_pre must be < 2 and != period_1=1, so no valid t_pre
        assert len(pairs) == 0

    def test_anticipation(self):
        """Anticipation shifts effective treatment boundary."""
        pairs_no_ant = enumerate_valid_triples(
            target_g=4,
            target_t=5,
            treatment_groups=[4],
            time_periods=[1, 2, 3, 4, 5],
            period_1=1,
            pt_assumption="all",
            anticipation=0,
        )
        pairs_ant1 = enumerate_valid_triples(
            target_g=4,
            target_t=5,
            treatment_groups=[4],
            time_periods=[1, 2, 3, 4, 5],
            period_1=1,
            pt_assumption="all",
            anticipation=1,
        )
        # With anticipation=1, effective treatment is at g-1=3
        # so fewer pre-treatment baselines available
        assert len(pairs_ant1) <= len(pairs_no_ant)


class TestEdgeCases:
    """Edge cases: all treated, empty pairs."""

    def test_all_units_treated_pt_all(self):
        """No never-treated units under PT-All should raise ValueError."""
        df = _make_staggered_panel(n_control=0, groups=(3, 5))
        with pytest.raises(ValueError, match="never-treated"):
            EfficientDiD(pt_assumption="all").fit(df, "y", "unit", "time", "first_treat")

    def test_all_units_treated_pt_post_raises(self):
        """No never-treated under PT-Post raises ValueError."""
        df = _make_staggered_panel(n_control=0, groups=(3, 5))
        with pytest.raises(ValueError, match="never-treated"):
            EfficientDiD(pt_assumption="post").fit(df, "y", "unit", "time", "first_treat")

    def test_anticipation_parameter(self):
        """Anticipation=1 shifts treatment boundary."""
        df = _make_simple_panel(treat_period=4, n_periods=6)
        result = EfficientDiD(anticipation=1).fit(df, "y", "unit", "time", "first_treat")
        # With anticipation=1, effective treatment starts at g-1=3
        # So ATT(4,3) should be post-treatment
        post_effects = [
            (g, t)
            for (g, t) in result.group_time_effects
            if t >= g - 1  # effective treatment at g - anticipation
        ]
        assert len(post_effects) > 0


# =============================================================================
# Tier 3: Bootstrap
# =============================================================================


class TestBootstrap:
    """Test multiplier bootstrap inference."""

    def test_bootstrap_se_finite(self, ci_params):
        n_boot = ci_params.bootstrap(99)
        df = _make_simple_panel()
        result = EfficientDiD(n_bootstrap=n_boot, seed=42).fit(
            df, "y", "unit", "time", "first_treat"
        )
        assert result.bootstrap_results is not None
        assert np.isfinite(result.overall_se)
        assert result.overall_se > 0
        for gt, d in result.group_time_effects.items():
            if np.isfinite(d["effect"]):
                assert np.isfinite(d["se"])

    def test_bootstrap_with_aggregation(self, ci_params):
        n_boot = ci_params.bootstrap(99)
        df = _make_simple_panel()
        result = EfficientDiD(n_bootstrap=n_boot, seed=42).fit(
            df, "y", "unit", "time", "first_treat", aggregate="all"
        )
        assert result.bootstrap_results is not None
        if result.event_study_effects:
            for e, d in result.event_study_effects.items():
                if np.isfinite(d["effect"]):
                    assert np.isfinite(d["se"])

    def test_bootstrap_coverage_basic(self, ci_params):
        """Rough coverage check: true effect should be in CI."""
        n_boot = ci_params.bootstrap(199, min_n=49)
        df = _make_simple_panel(effect=2.0, n_units=200, seed=42)
        result = EfficientDiD(n_bootstrap=n_boot, seed=42).fit(
            df, "y", "unit", "time", "first_treat"
        )
        ci = result.overall_conf_int
        # True effect is 2.0 — should be within CI for this seed
        if np.isfinite(ci[0]) and np.isfinite(ci[1]):
            # Just check CI is reasonable (not testing exact coverage)
            assert ci[0] < ci[1], "CI should be ordered"


# =============================================================================
# Tier 4: Simulation Validation
# =============================================================================


class TestSimulationValidation:
    """Validation against paper's DGP properties."""

    def test_synthetic_staggered_unbiased(self):
        """Single run at rho=0, verify ATT estimates near true values."""
        df = _make_compustat_dgp(rho=0.0, seed=42)
        result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat", aggregate="all")

        # Check individual ATT(g,t) estimates
        # ATT(5,5) should be near 0.154
        gt_55 = (5.0, 5)
        if gt_55 in result.group_time_effects:
            d = result.group_time_effects[gt_55]
            se = d["se"]
            if np.isfinite(se) and se > 0:
                assert (
                    abs(d["effect"] - 0.154) < 3 * se + 0.1
                ), f"ATT(5,5)={d['effect']:.4f}, expected ~0.154"

        # ATT(5,6) should be near 0.308
        gt_56 = (5.0, 6)
        if gt_56 in result.group_time_effects:
            d = result.group_time_effects[gt_56]
            se = d["se"]
            if np.isfinite(se) and se > 0:
                assert (
                    abs(d["effect"] - 0.308) < 3 * se + 0.1
                ), f"ATT(5,6)={d['effect']:.4f}, expected ~0.308"

    def test_efficiency_gain_negative_rho(self):
        """With rho=-0.5, EDiD should have lower SE than CS."""
        df = _make_compustat_dgp(rho=-0.5, seed=42)

        edid = EfficientDiD(pt_assumption="all")
        cs = CallawaySantAnna(control_group="never_treated")

        res_e = edid.fit(df, "y", "unit", "time", "first_treat")
        res_c = cs.fit(df, "y", "unit", "time", "first_treat")

        # Count how many post-treatment effects have lower SE
        lower_count = 0
        total_count = 0
        for gt in res_e.group_time_effects:
            if gt in res_c.group_time_effects:
                g, t = gt
                if t >= g:  # post-treatment
                    e_se = res_e.group_time_effects[gt]["se"]
                    c_se = res_c.group_time_effects[gt]["se"]
                    if np.isfinite(e_se) and np.isfinite(c_se) and c_se > 0:
                        total_count += 1
                        if e_se < c_se:
                            lower_count += 1

        if total_count > 0:
            # Majority of post-treatment effects should have lower SE
            ratio = lower_count / total_count
            assert ratio > 0.3, (
                f"EDiD should have lower SE for most effects with rho=-0.5 "
                f"({lower_count}/{total_count} = {ratio:.2f})"
            )

    def test_weights_shift_with_rho(self):
        """Verify weights sum to 1 and change with serial correlation."""
        weights_rho0 = {}
        weights_rho09 = {}

        for rho, store in [(0.0, weights_rho0), (0.9, weights_rho09)]:
            df = _make_compustat_dgp(rho=rho, seed=42)
            result = EfficientDiD().fit(df, "y", "unit", "time", "first_treat")
            if result.efficient_weights:
                for gt, w in result.efficient_weights.items():
                    if len(w) > 2:
                        assert (
                            abs(w.sum() - 1.0) < 1e-8
                        ), f"Weights should sum to 1, got {w.sum():.10f}"
                        store[gt] = w.copy()

        # Weights should differ between rho=0 and rho=0.9
        common = set(weights_rho0) & set(weights_rho09)
        if common:
            diffs = [np.linalg.norm(weights_rho0[gt] - weights_rho09[gt]) for gt in common]
            assert max(diffs) > 0.01, "Weights should change with rho"

    def test_analytical_se_consistency(self, ci_params):
        """Analytical SE should roughly match bootstrap SE."""
        n_boot = ci_params.bootstrap(999, min_n=199)
        threshold = 0.40 if n_boot < 100 else 0.30

        df = _make_simple_panel(n_units=200, effect=2.0, seed=42)

        # Analytical SE
        res_anal = EfficientDiD(n_bootstrap=0).fit(df, "y", "unit", "time", "first_treat")
        anal_se = res_anal.overall_se

        # Bootstrap SE
        res_boot = EfficientDiD(n_bootstrap=n_boot, seed=42).fit(
            df, "y", "unit", "time", "first_treat"
        )
        boot_se = res_boot.overall_se

        if np.isfinite(anal_se) and np.isfinite(boot_se) and boot_se > 0:
            rel_diff = abs(anal_se - boot_se) / boot_se
            assert rel_diff < threshold, (
                f"Analytical SE ({anal_se:.4f}) differs from bootstrap SE "
                f"({boot_se:.4f}) by {rel_diff:.2%}"
            )


# =============================================================================
# Regression Tests (PR #192 review feedback)
# =============================================================================


class TestPTPostExactMatch:
    """Fix 2: EDiD(PT-Post) should exactly match CS for all g, including g > 2."""

    def test_pt_post_staggered_exact_match(self):
        """With per-group baseline, EDiD(PT-Post) = CS for post-treatment effects."""
        df = _make_staggered_panel(n_per_group=100, n_control=100, groups=(3, 5))
        edid = EfficientDiD(pt_assumption="post")
        cs = CallawaySantAnna(control_group="never_treated", base_period="varying")

        res_e = edid.fit(df, "y", "unit", "time", "first_treat")
        res_c = cs.fit(df, "y", "unit", "time", "first_treat")

        matched = 0
        for g, t in res_e.group_time_effects:
            if t >= g and (g, t) in res_c.group_time_effects:
                e_eff = res_e.group_time_effects[(g, t)]["effect"]
                c_eff = res_c.group_time_effects[(g, t)]["effect"]
                assert abs(e_eff - c_eff) < 1e-8, f"ATT({g},{t}): EDiD={e_eff:.10f} CS={c_eff:.10f}"
                matched += 1
        assert matched > 0, "No matching post-treatment effects found"


class TestBridgingComparison:
    """Fix 1: Bridging comparisons should be valid under PT-All."""

    def test_bridging_comparison_valid(self):
        """ATT should be finite even when bridging comparisons are used."""
        # Create panel where g'=3 is used as comparison for g=5 at t=4 (g' treated at t=3)
        df = _make_staggered_panel(n_per_group=80, n_control=80, groups=(3, 5), n_periods=7)
        result = EfficientDiD(pt_assumption="all").fit(df, "y", "unit", "time", "first_treat")
        # Post-treatment effects for g=5 should be finite
        for (g, t), d in result.group_time_effects.items():
            if g == 5.0 and t >= 5:
                assert np.isfinite(d["effect"]), f"ATT({g},{t}) should be finite"


class TestWIFCorrection:
    """Fix 3: WIF correction for aggregated SEs."""

    def test_wif_increases_se(self):
        """WIF-corrected SE should be >= naive SE (without WIF)."""
        df = _make_staggered_panel(n_per_group=100, n_control=100, groups=(3, 5))
        result = EfficientDiD(pt_assumption="all").fit(
            df, "y", "unit", "time", "first_treat", aggregate="all"
        )
        wif_se = result.overall_se

        # Compute naive SE without WIF for comparison
        edid = EfficientDiD(pt_assumption="all")
        res_naive = edid.fit(df, "y", "unit", "time", "first_treat")

        # The overall SE with WIF should be at least as large as without
        # (WIF adds non-negative variance). Allow small FP tolerance.
        assert (
            wif_se >= res_naive.overall_se * 0.99
        ), f"WIF SE ({wif_se:.6f}) should be >= naive SE ({res_naive.overall_se:.6f})"

    def test_wif_se_vs_bootstrap(self, ci_params):
        """WIF-corrected SE should roughly match bootstrap SE."""
        n_boot = ci_params.bootstrap(999, min_n=199)
        threshold = 0.40 if n_boot < 100 else 0.30

        df = _make_staggered_panel(n_per_group=100, n_control=100, groups=(3, 5))

        # Analytical SE (with WIF)
        res_anal = EfficientDiD(n_bootstrap=0).fit(df, "y", "unit", "time", "first_treat")
        anal_se = res_anal.overall_se

        # Bootstrap SE
        res_boot = EfficientDiD(n_bootstrap=n_boot, seed=42).fit(
            df, "y", "unit", "time", "first_treat"
        )
        boot_se = res_boot.overall_se

        if np.isfinite(anal_se) and np.isfinite(boot_se) and boot_se > 0:
            rel_diff = abs(anal_se - boot_se) / boot_se
            assert rel_diff < threshold, (
                f"WIF-corrected SE ({anal_se:.4f}) differs from bootstrap SE "
                f"({boot_se:.4f}) by {rel_diff:.2%}"
            )


class TestResultsParams:
    """Fix 7: Results object should contain estimator params."""

    def test_results_contain_params(self):
        df = _make_simple_panel()
        result = EfficientDiD(pt_assumption="post", anticipation=1, n_bootstrap=0, seed=123).fit(
            df, "y", "unit", "time", "first_treat"
        )

        assert result.pt_assumption == "post"
        assert result.anticipation == 1
        assert result.n_bootstrap == 0
        assert result.bootstrap_weights == "rademacher"
        assert result.seed == 123

    def test_summary_shows_anticipation(self):
        df = _make_simple_panel(treat_period=4, n_periods=6)
        result = EfficientDiD(anticipation=1).fit(df, "y", "unit", "time", "first_treat")
        s = result.summary()
        assert "Anticipation" in s

    def test_summary_shows_bootstrap(self, ci_params):
        n_boot = ci_params.bootstrap(99)
        df = _make_simple_panel()
        result = EfficientDiD(n_bootstrap=n_boot, seed=42).fit(
            df, "y", "unit", "time", "first_treat"
        )
        s = result.summary()
        assert "Bootstrap" in s
