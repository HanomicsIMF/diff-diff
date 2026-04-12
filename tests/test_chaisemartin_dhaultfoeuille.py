"""
API and behavior tests for ``ChaisemartinDHaultfoeuille`` (dCDH) — Phase 1.

Covers basic API, validation, forward-compat NotImplementedError gates,
``drop_larger_lower``, A11 zero-retention, NaN handling, bootstrap
plumbing, and the results dataclass round-trip. Methodology validation
(hand-calculable arithmetic, cohort recentering correctness, parity
against R) lives in ``test_methodology_chaisemartin_dhaultfoeuille.py``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff import (
    DCDH,
    ChaisemartinDHaultfoeuille,
    ChaisemartinDHaultfoeuilleResults,
    DCDHBootstrapResults,
    chaisemartin_dhaultfoeuille,
    twowayfeweights,
)
from diff_diff.prep import generate_reversible_did_data

# =============================================================================
# Basic API
# =============================================================================


class TestChaisemartinDHaultfoeuilleBasicAPI:
    """Smoke tests for the basic happy path."""

    def test_fit_returns_results_object(self):
        data = generate_reversible_did_data(n_groups=40, n_periods=5, seed=1)
        est = ChaisemartinDHaultfoeuille()
        results = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        assert isinstance(results, ChaisemartinDHaultfoeuilleResults)
        assert est.is_fitted_ is True
        assert est.results_ is results

    def test_fit_recovers_homogeneous_effect_single_switch(self):
        # With seed and n=120, the analytical CI should bracket the truth
        data = generate_reversible_did_data(
            n_groups=120,
            n_periods=6,
            treatment_effect=2.0,
            seed=42,
        )
        est = ChaisemartinDHaultfoeuille()
        results = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        # CI should bracket the true effect of 2.0
        lo, hi = results.overall_conf_int
        assert lo <= 2.0 <= hi, f"95% CI [{lo:.3f}, {hi:.3f}] does not bracket true effect 2.0"

    def test_fit_with_joiners_only_pattern(self):
        # Use n_periods=10 so the random switch times don't saturate the
        # final period (which would zero the last period via A11 and bias
        # DID_M toward zero). 10 periods + 80 groups + uniform switch times
        # leaves enough late-period stable_0 controls.
        data = generate_reversible_did_data(
            n_groups=80,
            n_periods=10,
            pattern="joiners_only",
            treatment_effect=1.5,
            seed=2,
        )
        est = ChaisemartinDHaultfoeuille()
        results = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        # Joiners present, no leavers
        assert results.joiners_available is True
        assert results.leavers_available is False
        assert np.isnan(results.leavers_att)
        # CI brackets the truth (modulo conservative-CI noise)
        lo, hi = results.overall_conf_int
        assert lo <= 1.5 <= hi, (
            f"95% CI [{lo:.3f}, {hi:.3f}] does not bracket true effect 1.5; "
            f"DID_M = {results.overall_att:.3f}"
        )

    def test_fit_with_leavers_only_pattern(self):
        # Same n_periods rationale as the joiners_only test
        data = generate_reversible_did_data(
            n_groups=80,
            n_periods=10,
            pattern="leavers_only",
            treatment_effect=1.5,
            seed=3,
        )
        est = ChaisemartinDHaultfoeuille()
        results = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        assert results.joiners_available is False
        assert results.leavers_available is True
        assert np.isnan(results.joiners_att)

    def test_missing_column_raises_value_error(self):
        data = generate_reversible_did_data(n_groups=20, n_periods=4, seed=1)
        est = ChaisemartinDHaultfoeuille()
        with pytest.raises(ValueError, match="Missing columns"):
            est.fit(
                data,
                outcome="bogus",
                group="group",
                time="period",
                treatment="treatment",
            )

    def test_non_binary_treatment_raises_value_error(self):
        df = pd.DataFrame(
            {
                "group": [1, 1, 2, 2],
                "period": [0, 1, 0, 1],
                "outcome": [10.0, 11.0, 10.0, 12.0],
                "treatment": [0, 2, 0, 1],  # 2 is non-binary
            }
        )
        est = ChaisemartinDHaultfoeuille()
        with pytest.raises(ValueError, match="binary treatment"):
            est.fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

    def test_alias_DCDH_identity(self):
        assert DCDH is ChaisemartinDHaultfoeuille

    def test_get_set_params(self):
        est = ChaisemartinDHaultfoeuille(alpha=0.10, n_bootstrap=99, seed=7)
        params = est.get_params()
        assert params["alpha"] == 0.10
        assert params["n_bootstrap"] == 99
        assert params["seed"] == 7
        assert "drop_larger_lower" in params
        assert "twfe_diagnostic" in params
        assert "placebo" in params

        est.set_params(alpha=0.01, drop_larger_lower=False)
        assert est.alpha == 0.01
        assert est.drop_larger_lower is False

    def test_set_params_unknown_raises(self):
        est = ChaisemartinDHaultfoeuille()
        with pytest.raises(ValueError, match="Unknown parameter"):
            est.set_params(bogus_param=True)

    def test_convenience_function_matches_class(self):
        data = generate_reversible_did_data(n_groups=40, n_periods=5, seed=1)
        results_class = ChaisemartinDHaultfoeuille(seed=1).fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        results_fn = chaisemartin_dhaultfoeuille(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            seed=1,
        )
        # Same point estimate
        assert results_class.overall_att == pytest.approx(results_fn.overall_att)
        assert results_class.overall_se == pytest.approx(results_fn.overall_se)

    def test_minimal_computation_path(self):
        # Disable everything optional; verify still works
        data = generate_reversible_did_data(n_groups=30, n_periods=4, seed=1)
        est = ChaisemartinDHaultfoeuille(
            twfe_diagnostic=False,
            placebo=False,
            n_bootstrap=0,
        )
        results = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        # TWFE fields should be None
        assert results.twfe_weights is None
        assert results.twfe_beta_fe is None
        # Placebo should be NaN with available=False
        assert results.placebo_available is False
        assert np.isnan(results.placebo_effect)
        # Bootstrap should be None
        assert results.bootstrap_results is None
        # Main estimate should still be finite
        assert np.isfinite(results.overall_att)


# =============================================================================
# Forward-compat NotImplementedError gates
# =============================================================================


class TestForwardCompatGates:
    """Each Phase 2/3/deferred parameter must raise NotImplementedError."""

    @pytest.fixture
    def data(self):
        return generate_reversible_did_data(n_groups=20, n_periods=4, seed=1)

    def _est(self):
        return ChaisemartinDHaultfoeuille()

    def test_aggregate_simple_raises_not_implemented(self, data):
        # Per MEDIUM #1: even "simple" must be rejected; require aggregate=None exactly
        with pytest.raises(NotImplementedError, match="Phase 2"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                aggregate="simple",
            )

    def test_aggregate_event_study_raises_not_implemented(self, data):
        with pytest.raises(NotImplementedError, match="Phase 2"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                aggregate="event_study",
            )

    def test_L_max_raises_not_implemented(self, data):
        with pytest.raises(NotImplementedError, match="Phase 2"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                L_max=4,
            )

    def test_controls_raises_not_implemented(self, data):
        with pytest.raises(NotImplementedError, match="Phase 3"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                controls=["x"],
            )

    def test_trends_linear_raises_not_implemented(self, data):
        with pytest.raises(NotImplementedError, match="Phase 3"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                trends_linear=True,
            )

    def test_trends_nonparam_raises_not_implemented(self, data):
        with pytest.raises(NotImplementedError, match="Phase 3"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                trends_nonparam="state",
            )

    def test_honest_did_raises_not_implemented(self, data):
        with pytest.raises(NotImplementedError, match="Phase 3"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                honest_did=True,
            )

    def test_survey_design_raises_not_implemented(self, data):
        with pytest.raises(NotImplementedError, match="separate effort"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=object(),
            )

    def test_cluster_parameter_raises_not_implemented(self, data):
        """
        Per Phase 1 cluster contract: dCDH always clusters at the
        group level via the cohort-recentered influence function
        (analytical SEs) and the multiplier bootstrap (also grouped at
        the group column). Custom clustering is not supported in
        Phase 1.

        The reviewer flagged that ``cluster`` was previously accepted
        on ``__init__`` and stored on ``self.cluster`` but never
        actually read by ``fit()`` or ``_compute_dcdh_bootstrap()``,
        making it a silent no-op. This test pins the new contract: any
        non-None cluster value raises ``NotImplementedError`` at
        construction time with a message naming the offending value
        and pointing at the Phase 1 reservation. The same gate fires
        from ``set_params``.

        See REGISTRY.md ``Note (Phase 1 cluster contract)``.
        """
        # __init__ rejects any non-None cluster
        with pytest.raises(NotImplementedError, match=r"cluster.*Phase 1"):
            ChaisemartinDHaultfoeuille(cluster="state")
        with pytest.raises(NotImplementedError, match=r"cluster.*Phase 1"):
            ChaisemartinDHaultfoeuille(cluster="unit")

        # set_params after construction also rejects
        est = ChaisemartinDHaultfoeuille()
        with pytest.raises(NotImplementedError, match=r"cluster.*Phase 1"):
            est.set_params(cluster="state")

        # cluster=None still works (the only supported value)
        est_default = ChaisemartinDHaultfoeuille(cluster=None)
        assert est_default.cluster is None
        assert est_default.get_params()["cluster"] is None

        # The convenience function also rejects (forward-compat gate
        # propagates through the wrapper at __init__ time)
        with pytest.raises(NotImplementedError, match=r"cluster.*Phase 1"):
            chaisemartin_dhaultfoeuille(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                cluster="state",
            )

    def test_rank_deficient_action_error_raises_on_fitted_twfe(self):
        """
        Per Round 13: rank_deficient_action="error" must be honored on
        the fitted TWFE diagnostic path, not swallowed by the blanket
        try/except. The standalone twowayfeweights() always honors it;
        the fitted path must too.

        Uses a minimal panel (1 joiner group + 1 control group, 3
        periods, 1 obs per cell = 6 cells total) where the FE design
        has more columns than cells and triggers the underdetermined-
        system ValueError from solve_ols.
        """
        # 2 groups, 3 periods: 6 cells but the FE design has
        # (2-1) + (3-1) + 1 = 4 columns. That's fine.
        # To trigger rank-deficient: use a panel so small that the
        # number of cells equals the number of FE dummies.
        # With 3 groups, 3 periods: 9 cells, (3-1) + (3-1) + 1 = 5 columns. Not rank-deficient.
        # With 2 groups, 2 periods: 4 cells, (2-1) + (2-1) + 1 = 3 columns. Not rank-deficient.
        # Trigger via an unbalanced panel: 3 groups, 3 periods, but
        # group 3 only has period 0 (terminal missingness), giving
        # 7 cells with 3+3-1 = 5 columns. Not rank-deficient.
        #
        # Simplest route: a single-group joiner panel (1 group, 2
        # periods = 2 cells, but group+time dummies need 3 columns).
        # This also needs a control group. Use 2 groups, but one
        # is a singleton-period (contributing 1 cell to 1 period only).
        # Actually, the easiest verified trigger: 1 group, 2 periods.
        # solve_ols raises "Fewer observations (2) than parameters (3)."
        # But fit() will also raise for missing-baseline or insufficient
        # groups BEFORE reaching the TWFE diagnostic — so the TWFE
        # diagnostic must run first (it does: Step 5a).
        #
        # Use the confirmed trigger: 1 group, 2 periods, which has
        # 2 cells < 3 columns in the FE design.
        df = pd.DataFrame(
            {
                "group": [1, 1],
                "period": [0, 1],
                "treatment": [0, 1],
                "outcome": [10.0, 12.0],
            }
        )
        # rank_deficient_action="error" should propagate through
        est = ChaisemartinDHaultfoeuille(twfe_diagnostic=True, rank_deficient_action="error")
        with pytest.raises(ValueError, match="Fewer observations"):
            est.fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

        # rank_deficient_action="warn" should NOT raise on the same panel
        # (the diagnostic fails gracefully and main estimation continues)
        est_warn = ChaisemartinDHaultfoeuille(twfe_diagnostic=True, rank_deficient_action="warn")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            # The estimation may still raise for other reasons (e.g.,
            # no switching cells after the 1-group panel has no controls).
            # What we're testing is that the TWFE diagnostic does NOT
            # raise. If the main estimation raises, that's fine — the
            # test goal is that rank_deficient_action="warn" doesn't
            # propagate the ValueError.
            try:
                est_warn.fit(
                    df,
                    outcome="outcome",
                    group="group",
                    time="period",
                    treatment="treatment",
                )
            except ValueError as exc:
                # Acceptable if the error is from main estimation
                # (not from the TWFE diagnostic)
                assert "Fewer observations" not in str(exc), (
                    "rank_deficient_action='warn' should not raise the "
                    "TWFE rank-deficiency error"
                )


# =============================================================================
# drop_larger_lower (Critical #1)
# =============================================================================


class TestDropLargerLower:
    """Multi-switch group filtering matches R DIDmultiplegtDYN behavior."""

    def test_default_drops_a5_violators_with_warning(self):
        # Mix of single-switch groups and one explicit multi-switch group
        data = generate_reversible_did_data(
            n_groups=40,
            n_periods=4,
            pattern="single_switch",
            seed=1,
        )
        # Inject a multi-switch group: switch 0 -> 1 -> 0
        multi_switch = pd.DataFrame(
            {
                "group": [9999] * 4,
                "period": [0, 1, 2, 3],
                "treatment": [0, 1, 1, 0],
                "outcome": [10.0, 13.0, 14.0, 11.0],
                "true_effect": [0.0, 2.0, 2.0, 0.0],
                "d_lag": [np.nan, 0.0, 1.0, 1.0],
                "switcher_type": ["initial", "joiner", "stable_1", "leaver"],
            }
        )
        data = pd.concat([data, multi_switch], ignore_index=True)

        est = ChaisemartinDHaultfoeuille()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )
        # The multi-switch group should be dropped
        assert results.n_groups_dropped_crossers >= 1
        assert 9999 not in results.groups
        # A drop_larger_lower warning should fire
        assert any("drop_larger_lower" in str(wi.message) for wi in w)

    def test_drop_larger_lower_false_emits_inconsistency_warning(self):
        data = generate_reversible_did_data(
            n_groups=40,
            n_periods=4,
            pattern="single_switch",
            seed=1,
        )
        est = ChaisemartinDHaultfoeuille(drop_larger_lower=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )
        # Inconsistency warning should fire
        assert any("drop_larger_lower=False" in str(wi.message) for wi in w)

    def test_drop_larger_lower_true_no_op_on_single_switch_data(self):
        data = generate_reversible_did_data(
            n_groups=40,
            n_periods=5,
            pattern="single_switch",
            seed=1,
        )
        est = ChaisemartinDHaultfoeuille()
        results = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        assert results.n_groups_dropped_crossers == 0

    def test_singleton_baseline_filter_variance_only(self):
        # Build a panel where one group has a unique baseline (e.g., only group
        # with D_{g,0}=1). This is the footnote-15 condition.
        #
        # Per the variance-only filter (the dCDH Round 2 fix), the singleton-
        # baseline group is identified, counted in
        # n_groups_dropped_singleton_baseline, and excluded from the cohort-
        # recentered VARIANCE. But it remains in the point-estimate sample
        # as a period-based stable control (matching Python's documented
        # period-vs-cohort stable-control interpretation).
        data = generate_reversible_did_data(
            n_groups=20,
            n_periods=4,
            pattern="joiners_only",
            seed=1,
        )
        # Inject a single leaver group (unique baseline=1)
        leaver = pd.DataFrame(
            {
                "group": [9999] * 4,
                "period": [0, 1, 2, 3],
                "treatment": [1, 0, 0, 0],
                "outcome": [10.0, 9.0, 8.0, 7.0],
                "true_effect": [0.0, 0.0, 0.0, 0.0],
                "d_lag": [np.nan, 1.0, 0.0, 0.0],
                "switcher_type": ["initial", "leaver", "stable_0", "stable_0"],
            }
        )
        data = pd.concat([data, leaver], ignore_index=True)

        est = ChaisemartinDHaultfoeuille()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )
        # The leaver has a unique baseline (D=1) -> excluded from variance.
        assert results.n_groups_dropped_singleton_baseline >= 1
        # Per the variance-only filter, the group is RETAINED in the
        # point-estimate sample (it can serve as a period-based stable
        # control), so it appears in results.groups.
        assert 9999 in results.groups
        # The warning text mentions the variance-only scope.
        assert any("Singleton-baseline" in str(wi.message) for wi in w)
        assert any(
            "VARIANCE computation only" in str(wi.message) for wi in w
        ), "Warning text should clarify the filter is variance-only"

    def test_missing_baseline_period_raises_value_error(self):
        """
        Per fit() Step 5b: groups missing the first global period have
        an undefined baseline D_{g,1} and must be rejected with a clear
        error rather than crashing the cohort enumeration with NaN.
        """
        data = generate_reversible_did_data(n_groups=10, n_periods=5, seed=1)
        # Drop period 0 for group 5 (a "late-entry" group)
        data = data[~((data["group"] == 5) & (data["period"] == 0))].reset_index(drop=True)
        est = ChaisemartinDHaultfoeuille()
        with pytest.raises(ValueError, match="missing this baseline"):
            est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

    def test_interior_gap_drops_group_with_warning(self):
        """
        Per fit() Step 5b: groups with missing intermediate periods
        (interior gaps between their first and last observed period)
        are dropped with an explicit warning. The cohort/variance path
        requires consecutive observed periods to detect first switches
        unambiguously.
        """
        data = generate_reversible_did_data(n_groups=10, n_periods=5, seed=1)
        # Drop period 2 for group 3 (interior gap: g=3 has periods 0, 1, 3, 4)
        data = data[~((data["group"] == 3) & (data["period"] == 2))].reset_index(drop=True)
        est = ChaisemartinDHaultfoeuille()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )
        # Group 3 was dropped from the post-filter sample
        assert 3 not in results.groups
        # The interior-gap warning fired
        assert any("interior period gaps" in str(wi.message) for wi in w)
        # Other groups still present
        assert len(results.groups) == 9

    def test_terminal_missingness_retained(self):
        """
        Per fit() Step 5b contract: groups observed at the baseline but
        missing one or more LATER periods (terminal missingness / early
        exit / right-censoring) are RETAINED. The group contributes from
        its observed periods only, masked out of missing transitions by
        the per-period ``present = (N_mat[:, t] > 0) & (N_mat[:, t-1] > 0)``
        guard at three sites in the variance computation
        (``_compute_per_period_dids``, ``_compute_full_per_group_contributions``,
        ``_compute_cohort_recentered_inputs``). NaN never propagates into
        the arithmetic because ``D_mat[g, t]`` and ``Y_mat[g, t]`` are
        never read without first checking ``N_mat[g, t] > 0``.

        This pins the remaining unspoken branch of the ragged-panel
        contract that fit() validates: missing baseline -> ValueError;
        interior gap -> drop with warning; terminal missingness -> retained.
        See REGISTRY.md ``Note (deviation from R DIDmultiplegtDYN)`` for
        the documented contract and the rationale for supporting only
        terminal missingness in Phase 1.
        """
        data = generate_reversible_did_data(n_groups=10, n_periods=5, seed=1)
        # Group 5 has periods 0, 1, 2 only (terminal missingness: missing 3, 4)
        data = data[~((data["group"] == 5) & (data["period"].isin([3, 4])))].reset_index(drop=True)
        est = ChaisemartinDHaultfoeuille()
        # The fit completes without error
        results = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        # Group 5 is RETAINED in the post-filter sample (NOT dropped)
        assert 5 in results.groups
        # All 10 groups remain
        assert len(results.groups) == 10
        # The point estimate is well-defined (not NaN)
        assert np.isfinite(results.overall_att)
        # Per-period DIDs were computed (the structure of per_period_effects
        # depends on the panel's switch pattern; assert at least one entry
        # was populated rather than asserting specific counts)
        assert len(results.per_period_effects) > 0

    def test_global_period_gap_treated_as_adjacent(self):
        """
        Per the REGISTRY.md period-index semantics contract: the
        estimator operates on sorted period indices, not calendar dates.
        A panel with periods [0, 1, 3] (period 2 missing for ALL groups)
        is treated as a valid 3-period panel where period 3 is the
        immediate successor of period 1. No error, no warning, no
        imputation. This is consistent with the AER 2020 paper's
        Theorem 3 (adjacent sorted periods) and R DIDmultiplegtDYN.

        This test pins the contract so a future change doesn't
        accidentally start rejecting or warning on globally missing
        calendar periods.
        """
        # 4 groups × 3 periods [0, 1, 3] — all groups present at all
        # three periods, no interior gaps, just a global calendar gap
        df = pd.DataFrame(
            {
                "group": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                "period": [0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3],
                "treatment": [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1],
                "outcome": [
                    10,
                    11,
                    15,
                    10,
                    11,
                    14,
                    10,
                    11,
                    12,
                    12,
                    13,
                    14,
                ],
            }
        )
        est = ChaisemartinDHaultfoeuille()
        # The fit completes without error
        results = est.fit(
            df,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        # All 4 groups present
        assert len(results.groups) == 4
        # Point estimate is finite
        assert np.isfinite(results.overall_att)
        # Per-period effects include the transition at t=3 (treated as
        # the successor of t=1)
        assert len(results.per_period_effects) > 0

    def test_cell_count_weighting_unbalanced_input(self):
        """
        Regression test: dCDH must use cell counts (paper-literal),
        not within-cell observation counts, as the Theorem 3 N_{a,b,t}
        weights.

        Constructed with two joiner groups whose (g, t) cells contain
        very different numbers of original observations (group 1 has
        100 obs/cell, group 2 has 1 obs/cell). Both joiners have the
        same true effect under the cell-weighted formula.

        Under cell weighting (paper-literal, the correct behavior),
        each cell contributes equally and the result equals the simple
        average of cell-level effects (~5.0). Under the bug behavior
        (sample-size weighting), group 1 dominates by 100x because its
        cells contribute 100x the weight.

        On a noiseless DGP both formulas would give 5.0; we add a
        deliberate per-cell perturbation to group 1 so that the bug
        would be visible: under sample-size weighting the result
        would shift toward group 1's cell mean (which is perturbed),
        while under cell weighting group 2's pristine effect would
        anchor the average.
        """
        records = []
        # Group 1: 100 obs per cell, joiner at t=2, but with a +0.5
        # perturbation to its post-treatment cell mean (so its cell
        # effect is 5.5, not 5.0)
        for t in [0, 1, 2]:
            for i in range(100):
                d = 1 if t == 2 else 0
                base = 10.0
                noise = 0.0  # noiseless within cell
                if t == 2:
                    y = base + 5.5 + noise  # perturbed post effect
                else:
                    y = base + noise
                records.append({"group": 1, "period": t, "treatment": d, "outcome": y})
        # Group 2: 1 obs per cell, joiner at t=2, clean effect of 5.0
        for t in [0, 1, 2]:
            d = 1 if t == 2 else 0
            y = 10.0 + (5.0 if d == 1 else 0)
            records.append({"group": 2, "period": t, "treatment": d, "outcome": y})
        # Stable controls
        for g in [3, 4]:
            for t in [0, 1, 2]:
                records.append(
                    {
                        "group": g,
                        "period": t,
                        "treatment": 0,
                        "outcome": 10.0,
                    }
                )

        df = pd.DataFrame(records)
        est = ChaisemartinDHaultfoeuille()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = est.fit(
                df, outcome="outcome", group="group", time="period", treatment="treatment"
            )

        # Expected under CELL weighting:
        #   DID_+,2 = avg over joiner cells - avg over stable_0 cells
        #         = avg(5.5, 5.0) - avg(0, 0) = 5.25
        # Expected under SAMPLE-SIZE weighting (the bug):
        #   DID_+,2 = (100*5.5 + 1*5.0) / 101 - 0 = 5.495
        # The two differ by ~0.25, so we can detect the bug at 0.05 tolerance.
        assert abs(results.overall_att - 5.25) < 0.05, (
            f"Expected DID_M ≈ 5.25 under cell weighting, got "
            f"{results.overall_att:.4f}. If you see ~5.495 the estimator "
            f"is using sample-size weighting (the bug)."
        )
        # n_switcher_cells should be 2 (one cell per joiner group at t=2),
        # NOT 101 (the total observation count)
        assert results.n_switcher_cells == 2, (
            f"n_switcher_cells should be 2 (cell count), got "
            f"{results.n_switcher_cells}. If you see 101 the estimator "
            f"is using sample-size weighting (the bug)."
        )


# =============================================================================
# A11 zero-retention (Critical #2)
# =============================================================================


class TestA11Handling:
    """Assumption 11 violations are zeroed in numerator, retained in denominator."""

    def test_a11_violation_zero_in_numerator_retain_in_denominator(self):
        # 4-group, 3-period panel where at t=2 there are joiners (g=1, g=2)
        # but no stable_0 controls. Both baselines (0, 1) are non-singleton
        # (2 groups each), so the singleton-baseline filter is a no-op.
        df = pd.DataFrame(
            {
                "group": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                "period": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                "treatment": [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                "outcome": [10.0, 11.0, 14.0, 10.0, 11.0, 14.0, 10.0, 11.0, 12.0, 10.0, 11.0, 12.0],
            }
        )
        # At t=2: joiners = {g=1, g=2}; stable_1 = {g=3, g=4}; NO stable_0 -> A11 violated
        est = ChaisemartinDHaultfoeuille()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = est.fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )
        # A11 warning should fire
        assert any("Assumption 11" in str(wi.message) for wi in w)
        # Per-period decomposition: t=2 should be A11-zeroed for joiners
        cell_t2 = results.per_period_effects[2]
        assert cell_t2["did_plus_t"] == 0.0
        assert cell_t2["did_plus_t_a11_zeroed"] is True
        # The joiner count is retained in N_S
        assert cell_t2["n_10_t"] == 2

    def test_placebo_a11_violation_emits_warning(self):
        """
        Mirror of the main A11 contract for the placebo:
        when placebo joiners exist (3-period stable D=0 history then
        switch) but no group provides a 3-period stable_0 control,
        the affected placebo period contribution is zeroed AND a
        consolidated ``Placebo (DID_M^pl) Assumption 11 violations``
        warning fires from ``fit()``.

        Construct: 4-group T=3 panel with two D=[0,0,1] joiners (also
        placebo joiners at t=2) and two always-treated controls. No
        group has D=[0,0,0], so the placebo joiner side has no
        stable_0 control. The main path also has an A11 violation
        on the same panel (its own warning fires too); this test
        asserts the PLACEBO warning specifically.
        """
        df = pd.DataFrame(
            {
                "group": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                "period": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                "treatment": [0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                "outcome": [
                    10.0,
                    11.0,
                    15.0,
                    10.0,
                    11.0,
                    16.0,
                    12.0,
                    13.0,
                    14.0,
                    12.0,
                    13.0,
                    14.0,
                ],
            }
        )
        est = ChaisemartinDHaultfoeuille()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = est.fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )
        # Placebo was computed (T >= 3 + qualifying cells) and is available
        assert results.placebo_available
        # The placebo A11 warning fired (text contains "Placebo" + "Assumption 11")
        placebo_a11_warnings = [
            wi for wi in w if "Placebo" in str(wi.message) and "Assumption 11" in str(wi.message)
        ]
        assert len(placebo_a11_warnings) >= 1, (
            "Expected the placebo A11 warning to fire on a panel where placebo "
            "joiners exist but no 3-period stable_0 controls exist. Got warnings: "
            f"{[str(wi.message) for wi in w]}"
        )
        # The warning should mention the affected placebo period
        assert "stable_0" in str(placebo_a11_warnings[0].message)

    def test_a11_natural_zero_no_switchers_does_not_zero_flag(self):
        data = generate_reversible_did_data(
            n_groups=20,
            n_periods=4,
            pattern="joiners_only",
            seed=1,
        )
        est = ChaisemartinDHaultfoeuille()
        results = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        # No leavers in joiners_only, so leaver A11 flag is always False
        for t, cell in results.per_period_effects.items():
            if cell["n_01_t"] == 0:
                assert cell["did_minus_t_a11_zeroed"] is False


# =============================================================================
# NaN handling
# =============================================================================


class TestNaNHandling:
    def test_empty_dataframe_raises(self):
        df = pd.DataFrame(columns=["group", "period", "treatment", "outcome"])
        est = ChaisemartinDHaultfoeuille()
        with pytest.raises((ValueError, KeyError)):
            est.fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

    def test_no_switchers_raises(self):
        # All groups stable -> dCDH cannot estimate. The exact error path
        # depends on which filter fires first (singleton-baseline vs
        # no-switching-cells), so accept either message.
        df = pd.DataFrame(
            {
                "group": [1, 1, 1, 2, 2, 2],
                "period": [0, 1, 2, 0, 1, 2],
                "treatment": [0, 0, 0, 1, 1, 1],
                "outcome": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            }
        )
        est = ChaisemartinDHaultfoeuille()
        with pytest.raises(ValueError, match=r"(No switching cells|no groups remain)"):
            est.fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )


# =============================================================================
# Bootstrap inference
# =============================================================================


class TestBootstrap:
    @pytest.fixture
    def data(self):
        return generate_reversible_did_data(n_groups=80, n_periods=5, seed=1)

    def test_bootstrap_zero_uses_analytical(self, data):
        est = ChaisemartinDHaultfoeuille(n_bootstrap=0)
        results = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        assert results.bootstrap_results is None
        assert np.isfinite(results.overall_se)

    def test_bootstrap_rademacher(self, data, ci_params):
        n_boot = ci_params.bootstrap(199)
        est = ChaisemartinDHaultfoeuille(
            n_bootstrap=n_boot,
            bootstrap_weights="rademacher",
            seed=42,
        )
        results = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        assert results.bootstrap_results is not None
        assert isinstance(results.bootstrap_results, DCDHBootstrapResults)
        assert results.bootstrap_results.n_bootstrap == n_boot
        assert results.bootstrap_results.weight_type == "rademacher"
        assert np.isfinite(results.bootstrap_results.overall_se)
        assert results.bootstrap_results.overall_se > 0

    def test_bootstrap_mammen(self, data, ci_params):
        n_boot = ci_params.bootstrap(199)
        est = ChaisemartinDHaultfoeuille(
            n_bootstrap=n_boot,
            bootstrap_weights="mammen",
            seed=42,
        )
        results = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        assert results.bootstrap_results is not None
        assert results.bootstrap_results.weight_type == "mammen"

    def test_bootstrap_webb(self, data, ci_params):
        n_boot = ci_params.bootstrap(199)
        est = ChaisemartinDHaultfoeuille(
            n_bootstrap=n_boot,
            bootstrap_weights="webb",
            seed=42,
        )
        results = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        assert results.bootstrap_results is not None
        assert results.bootstrap_results.weight_type == "webb"

    def test_placebo_bootstrap_unavailable_in_phase_1(self, data, ci_params):
        """
        Phase 1 commitment: the placebo SE is intentionally NaN even when
        ``n_bootstrap > 0``. The dynamic companion paper Section 3.7.3
        derives the cohort-recentered analytical variance for ``DID_l``
        only — the placebo's influence-function machinery is deferred to
        Phase 2. The bootstrap path covers ``DID_M``, ``DID_+``, and
        ``DID_-`` only.

        This test pins down the contract so that future contributors do
        not silently widen the bootstrap surface to include the placebo
        without also wiring up the documented Phase 2 derivation. If
        Phase 2 implements the placebo bootstrap, this test should be
        updated (not deleted) to assert finite placebo bootstrap fields.
        """
        n_boot = ci_params.bootstrap(199)
        est = ChaisemartinDHaultfoeuille(
            n_bootstrap=n_boot,
            placebo=True,
            seed=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

        # Bootstrap is populated for the three implemented targets
        assert results.bootstrap_results is not None
        assert np.isfinite(results.bootstrap_results.overall_se)

        # Placebo bootstrap fields are explicitly None (not populated)
        assert results.bootstrap_results.placebo_se is None
        assert results.bootstrap_results.placebo_ci is None
        assert results.bootstrap_results.placebo_p_value is None

        # Placebo inference fields on the main results stay NaN-consistent
        assert np.isnan(results.placebo_se)
        assert np.isnan(results.placebo_t_stat)
        assert np.isnan(results.placebo_p_value)
        assert np.isnan(results.placebo_conf_int[0])
        assert np.isnan(results.placebo_conf_int[1])

        # The placebo point estimate itself is still computed and finite
        # (the deferral is purely about inference, not the point estimate)
        if results.placebo_available:
            assert np.isfinite(results.placebo_effect)

    def test_bootstrap_p_value_and_ci_propagated_to_top_level(self, data, ci_params):
        """
        Per the bootstrap inference surface contract: when
        ``n_bootstrap > 0``, the top-level ``results.overall_*`` /
        ``joiners_*`` / ``leavers_*`` p-value and CI fields hold the
        percentile-based bootstrap inference computed by the
        multiplier bootstrap, NOT normal-theory recomputations from
        the bootstrap SE. The t-stat is still computed from the SE
        (project anti-pattern rule: never compute t = effect/se
        inline).

        Pre-Round-10, the dCDH ``fit()`` body silently called
        ``safe_inference(overall_att, br.overall_se)`` and stored its
        normal-theory p/CI on the top-level fields, which made the
        public inference surface a hybrid (bootstrap SE + normal-
        theory p/CI). Library precedent for the propagation:
        ``imputation.py:790-805``, ``two_stage.py:778-787``,
        ``efficient_did.py:1009-1013``. This test pins the new
        contract.

        See REGISTRY.md ``ChaisemartinDHaultfoeuille`` ``Note
        (bootstrap inference surface)``.
        """
        n_boot = ci_params.bootstrap(199)
        est = ChaisemartinDHaultfoeuille(
            n_bootstrap=n_boot,
            bootstrap_weights="rademacher",
            seed=42,
        )
        results = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        br = results.bootstrap_results
        assert br is not None

        # Overall DID_M: top-level p-value and CI come from bootstrap
        assert results.overall_p_value == pytest.approx(br.overall_p_value)
        assert results.overall_conf_int == pytest.approx(br.overall_ci)
        # The t-stat is computed from the SE (effect / se), not from
        # a percentile distribution
        assert np.isfinite(results.overall_t_stat)
        expected_t = results.overall_att / results.overall_se
        assert results.overall_t_stat == pytest.approx(expected_t)

        # Joiners
        if results.joiners_available and br.joiners_p_value is not None:
            assert results.joiners_p_value == pytest.approx(br.joiners_p_value)
            assert results.joiners_conf_int == pytest.approx(br.joiners_ci)

        # Leavers
        if results.leavers_available and br.leavers_p_value is not None:
            assert results.leavers_p_value == pytest.approx(br.leavers_p_value)
            assert results.leavers_conf_int == pytest.approx(br.leavers_ci)

        # event_study_effects[1] mirrors the top-level overall fields,
        # so it should also reflect the bootstrap inference
        assert results.event_study_effects is not None
        assert 1 in results.event_study_effects
        es = results.event_study_effects[1]
        assert es["p_value"] == pytest.approx(br.overall_p_value)
        assert es["conf_int"] == pytest.approx(br.overall_ci)

        # summary() and to_dataframe() chain off the top-level fields,
        # so they automatically reflect the bootstrap inference. Smoke
        # test that they don't crash and that the rendered values match
        # the bootstrap output.
        summary_text = results.summary()
        assert "DID_M" in summary_text
        # The summary footer should mention bootstrap inference, NOT
        # the analytical-CI conservativeness note (which only applies
        # when n_bootstrap=0). This pins the P2 fix from Round 11.
        assert "multiplier-bootstrap percentile inference" in summary_text
        assert "analytical CI is conservative" not in summary_text
        df_overall = results.to_dataframe(level="overall")
        assert df_overall.iloc[0]["p_value"] == pytest.approx(br.overall_p_value)
        assert df_overall.iloc[0]["conf_int_lower"] == pytest.approx(br.overall_ci[0])
        assert df_overall.iloc[0]["conf_int_upper"] == pytest.approx(br.overall_ci[1])

    def test_bootstrap_seed_reproducibility(self, data, ci_params):
        n_boot = ci_params.bootstrap(99)
        r1 = ChaisemartinDHaultfoeuille(n_bootstrap=n_boot, seed=42).fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        r2 = ChaisemartinDHaultfoeuille(n_bootstrap=n_boot, seed=42).fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        assert r1.overall_se == r2.overall_se


# =============================================================================
# Results dataclass round-trip
# =============================================================================


class TestResultsDataclass:
    @pytest.fixture
    def results(self):
        data = generate_reversible_did_data(n_groups=40, n_periods=5, seed=1)
        return ChaisemartinDHaultfoeuille().fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )

    def test_summary_formats_without_error(self, results):
        out = results.summary()
        assert isinstance(out, str)
        assert "DID_M" in out
        assert "DID_+" in out
        assert "DID_-" in out
        # Analytical mode (n_bootstrap=0) shows the conservative-CI note
        assert "analytical CI is conservative" in out
        assert "multiplier-bootstrap" not in out

    def test_print_summary(self, results, capsys):
        results.print_summary()
        captured = capsys.readouterr()
        assert "DID_M" in captured.out

    def test_to_dataframe_overall(self, results):
        df = results.to_dataframe("overall")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert list(df.columns) == [
            "estimand",
            "effect",
            "se",
            "t_stat",
            "p_value",
            "conf_int_lower",
            "conf_int_upper",
        ]
        assert df.iloc[0]["estimand"] == "DID_M"

    def test_to_dataframe_joiners_leavers(self, results):
        df = results.to_dataframe("joiners_leavers")
        assert len(df) == 3
        assert set(df["estimand"].tolist()) == {"DID_M", "DID_+", "DID_-"}
        # Round 4: n_cells and n_obs are separate columns with consistent
        # units across all rows. n_cells counts switching (g, t) cells,
        # n_obs sums raw observation counts over the same cells. The DID_M
        # row uses the union of joiner + leaver cells.
        assert "n_cells" in df.columns
        assert "n_obs" in df.columns
        # On balanced 1-obs-per-cell test data, n_cells == n_obs everywhere
        for _, row in df.iterrows():
            assert row["n_cells"] == row["n_obs"], (
                f"On balanced data n_cells should equal n_obs for row "
                f"{row['estimand']}, got n_cells={row['n_cells']}, "
                f"n_obs={row['n_obs']}"
            )
        # The DID_M row's count is the sum of the DID_+ and DID_- rows'
        did_m_row = df[df["estimand"] == "DID_M"].iloc[0]
        did_plus_row = df[df["estimand"] == "DID_+"].iloc[0]
        did_minus_row = df[df["estimand"] == "DID_-"].iloc[0]
        assert did_m_row["n_cells"] == did_plus_row["n_cells"] + did_minus_row["n_cells"]

    def test_to_dataframe_per_period(self, results):
        df = results.to_dataframe("per_period")
        assert isinstance(df, pd.DataFrame)
        assert "period" in df.columns
        assert "did_plus_t" in df.columns
        assert "did_plus_t_a11_zeroed" in df.columns

    def test_to_dataframe_twfe_weights(self, results):
        df = results.to_dataframe("twfe_weights")
        assert isinstance(df, pd.DataFrame)
        assert "weight" in df.columns

    def test_to_dataframe_unknown_level_raises(self, results):
        with pytest.raises(ValueError, match="Unknown level"):
            results.to_dataframe("bogus")

    def test_event_study_effects_populated_at_l1(self, results):
        # Per review MEDIUM #5: in Phase 1, event_study_effects should not be
        # None — it should hold a single key 1 with the same effect as overall_att
        assert results.event_study_effects is not None
        assert 1 in results.event_study_effects
        es1 = results.event_study_effects[1]
        assert es1["effect"] == pytest.approx(results.overall_att)
        assert es1["se"] == pytest.approx(results.overall_se)

    def test_is_significant_property(self, results):
        # Boolean reflects whether p-value < alpha
        expected = results.overall_p_value < results.alpha
        assert results.is_significant is expected

    def test_coef_var_nan_safe_on_non_finite_se(self):
        # coef_var = SE / |ATT|. When SE is non-finite (NaN or Inf), the
        # property must return NaN (not propagate the bad value). When SE
        # is exactly 0, coef_var = 0 is correct (zero variance).
        from diff_diff.chaisemartin_dhaultfoeuille_results import (
            ChaisemartinDHaultfoeuilleResults,
        )

        r_nan = ChaisemartinDHaultfoeuilleResults(
            overall_att=2.0,
            overall_se=float("nan"),
            overall_t_stat=float("nan"),
            overall_p_value=float("nan"),
            overall_conf_int=(float("nan"), float("nan")),
            joiners_att=float("nan"),
            joiners_se=float("nan"),
            joiners_t_stat=float("nan"),
            joiners_p_value=float("nan"),
            joiners_conf_int=(float("nan"), float("nan")),
            n_joiner_cells=0,
            n_joiner_obs=0,
            joiners_available=False,
            leavers_att=float("nan"),
            leavers_se=float("nan"),
            leavers_t_stat=float("nan"),
            leavers_p_value=float("nan"),
            leavers_conf_int=(float("nan"), float("nan")),
            n_leaver_cells=0,
            n_leaver_obs=0,
            leavers_available=False,
            placebo_effect=float("nan"),
            placebo_se=float("nan"),
            placebo_t_stat=float("nan"),
            placebo_p_value=float("nan"),
            placebo_conf_int=(float("nan"), float("nan")),
            placebo_available=False,
            per_period_effects={},
            groups=[1],
            time_periods=[0, 1],
            n_obs=2,
            n_treated_obs=1,
            n_switcher_cells=0,
            n_cohorts=0,
            n_groups_dropped_crossers=0,
            n_groups_dropped_singleton_baseline=0,
            n_groups_dropped_never_switching=0,
        )
        assert np.isnan(r_nan.coef_var)

        # Independently verify: with finite SE > 0, coef_var equals SE/|ATT|
        r_finite = ChaisemartinDHaultfoeuilleResults(
            overall_att=2.0,
            overall_se=0.5,
            overall_t_stat=4.0,
            overall_p_value=0.01,
            overall_conf_int=(1.0, 3.0),
            joiners_att=float("nan"),
            joiners_se=float("nan"),
            joiners_t_stat=float("nan"),
            joiners_p_value=float("nan"),
            joiners_conf_int=(float("nan"), float("nan")),
            n_joiner_cells=0,
            n_joiner_obs=0,
            joiners_available=False,
            leavers_att=float("nan"),
            leavers_se=float("nan"),
            leavers_t_stat=float("nan"),
            leavers_p_value=float("nan"),
            leavers_conf_int=(float("nan"), float("nan")),
            n_leaver_cells=0,
            n_leaver_obs=0,
            leavers_available=False,
            placebo_effect=float("nan"),
            placebo_se=float("nan"),
            placebo_t_stat=float("nan"),
            placebo_p_value=float("nan"),
            placebo_conf_int=(float("nan"), float("nan")),
            placebo_available=False,
            per_period_effects={},
            groups=[1],
            time_periods=[0, 1],
            n_obs=2,
            n_treated_obs=1,
            n_switcher_cells=0,
            n_cohorts=0,
            n_groups_dropped_crossers=0,
            n_groups_dropped_singleton_baseline=0,
            n_groups_dropped_never_switching=0,
        )
        assert r_finite.coef_var == pytest.approx(0.25)


# =============================================================================
# Standalone twowayfeweights helper
# =============================================================================


class TestTwowayFeweightsHelper:
    def test_standalone_function_runs(self):
        data = generate_reversible_did_data(n_groups=30, n_periods=5, seed=1)
        result = twowayfeweights(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        # Returns a TWFEWeightsResult
        assert hasattr(result, "weights")
        assert hasattr(result, "fraction_negative")
        assert hasattr(result, "sigma_fe")
        assert hasattr(result, "beta_fe")
        assert isinstance(result.weights, pd.DataFrame)

    def test_standalone_function_equals_fitted_diagnostic(self):
        data = generate_reversible_did_data(n_groups=30, n_periods=5, seed=1)
        # Standalone
        standalone = twowayfeweights(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        # Fitted (twfe_diagnostic=True by default)
        results = ChaisemartinDHaultfoeuille().fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )
        # Both APIs run on the FULL pre-filter cell sample per the
        # documented TWFE diagnostic sample contract. On clean
        # single-switch data with no crossers, no filters fire and
        # both should produce identical results. The more interesting
        # filter-divergence cases are pinned in
        # test_twfe_pre_filter_contract_with_interior_gap_drop and
        # test_twfe_pre_filter_contract_with_multi_switch_drop. See
        # REGISTRY.md ChaisemartinDHaultfoeuille
        # `Note (TWFE diagnostic sample contract)`.
        assert results.twfe_beta_fe == pytest.approx(standalone.beta_fe)
        assert results.twfe_fraction_negative == pytest.approx(standalone.fraction_negative)

    def test_twfe_pre_filter_contract_with_interior_gap_drop(self):
        """
        Per the TWFE diagnostic sample contract: when fit() drops a
        group via Step 5b's interior-gap filter, results.twfe_*
        continues to describe the FULL pre-filter cell sample (matching
        the standalone twowayfeweights() output), and a divergence
        warning fires. The fitted twfe_* and overall_att now describe
        DIFFERENT samples by design.

        See REGISTRY.md ChaisemartinDHaultfoeuille `Note (TWFE
        diagnostic sample contract)`.
        """
        data = generate_reversible_did_data(n_groups=10, n_periods=5, seed=1)
        # Drop period 2 for group 3 (interior gap)
        data = data[~((data["group"] == 3) & (data["period"] == 2))].reset_index(drop=True)

        # Standalone TWFE on full input
        standalone = twowayfeweights(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )

        # Fitted estimator
        est = ChaisemartinDHaultfoeuille()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

        # The fitted twfe_* matches the standalone (both pre-filter)
        assert results.twfe_beta_fe == pytest.approx(standalone.beta_fe)
        assert results.twfe_fraction_negative == pytest.approx(standalone.fraction_negative)

        # The estimation sample is smaller (group 3 was dropped)
        assert 3 not in results.groups
        assert len(results.groups) == 9

        # The divergence warning fired with the expected counts
        div_warnings = [
            wi for wi in w if "TWFE diagnostic sample-contract notice" in str(wi.message)
        ]
        assert len(div_warnings) == 1, "exactly one divergence warning expected"
        assert "1 interior-gap group(s)" in str(div_warnings[0].message)
        assert "0 multi-switch group(s)" in str(div_warnings[0].message)

    def test_twfe_pre_filter_contract_with_multi_switch_drop(self):
        """
        Per the TWFE diagnostic sample contract: when fit() drops a
        group via Step 6's drop_larger_lower (multi-switch) filter,
        results.twfe_* continues to describe the FULL pre-filter cell
        sample, and a divergence warning fires.

        See REGISTRY.md ChaisemartinDHaultfoeuille `Note (TWFE
        diagnostic sample contract)`.
        """
        # Build a panel where one group is a clear multi-switch crosser
        data = generate_reversible_did_data(
            n_groups=20,
            n_periods=4,
            pattern="single_switch",
            seed=1,
        )
        # Inject a multi-switch group: D = [0, 1, 0, 1]
        crosser = pd.DataFrame(
            {
                "group": [9999] * 4,
                "period": [0, 1, 2, 3],
                "treatment": [0, 1, 0, 1],
                "outcome": [10.0, 12.0, 11.0, 13.0],
                "true_effect": [0.0, 0.0, 0.0, 0.0],
                "d_lag": [np.nan, 0.0, 1.0, 0.0],
                "switcher_type": ["initial", "joiner", "leaver", "joiner"],
            }
        )
        data = pd.concat([data, crosser], ignore_index=True)

        # Standalone TWFE on full input (including the crosser)
        standalone = twowayfeweights(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
        )

        # Fitted estimator (drop_larger_lower=True default drops the crosser)
        est = ChaisemartinDHaultfoeuille()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

        # The fitted twfe_* matches the standalone (both pre-filter,
        # both include the crosser)
        assert results.twfe_beta_fe == pytest.approx(standalone.beta_fe)
        assert results.twfe_fraction_negative == pytest.approx(standalone.fraction_negative)

        # The estimation sample dropped the crosser
        assert 9999 not in results.groups
        assert results.n_groups_dropped_crossers >= 1

        # The divergence warning fired with the expected counts
        div_warnings = [
            wi for wi in w if "TWFE diagnostic sample-contract notice" in str(wi.message)
        ]
        assert len(div_warnings) == 1, "exactly one divergence warning expected"
        assert "0 interior-gap group(s)" in str(div_warnings[0].message)
        assert "1 multi-switch group(s)" in str(div_warnings[0].message)

    def test_twfe_no_divergence_warning_on_clean_panel(self):
        """
        Negative test for the TWFE diagnostic sample contract: on a
        clean panel where no filters fire, the divergence warning must
        NOT fire. The fitted twfe_* and overall_att describe the same
        sample, so there is no divergence to warn about.

        Hard-codes ``pattern="single_switch"`` so a future change to
        ``generate_reversible_did_data`` defaults can't silently
        introduce multi-switch crossers and start firing the warning.
        """
        data = generate_reversible_did_data(
            n_groups=20, n_periods=4, pattern="single_switch", seed=42
        )
        est = ChaisemartinDHaultfoeuille()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            results = est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

        # No filter drops on a clean panel
        assert results.n_groups_dropped_crossers == 0
        assert len(results.groups) == 20

        # The divergence warning did NOT fire
        div_warnings = [
            wi for wi in w if "TWFE diagnostic sample-contract notice" in str(wi.message)
        ]
        assert (
            len(div_warnings) == 0
        ), "Divergence warning should not fire on clean panels where filters do not drop groups"

    # The four tests below pin the contract that twowayfeweights() and
    # ChaisemartinDHaultfoeuille.fit() share the same validation rules
    # via the _validate_and_aggregate_to_cells helper. Without this
    # contract, the standalone helper could silently mishandle malformed
    # input (drop NaN rows in groupby, threshold non-binary treatment,
    # round within-cell varying treatment without warning).

    def test_twowayfeweights_rejects_nan_treatment(self):
        data = generate_reversible_did_data(n_groups=20, n_periods=4, seed=1)
        data.loc[data.index[0], "treatment"] = float("nan")
        with pytest.raises(ValueError, match="Treatment column.*NaN"):
            twowayfeweights(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

    def test_twowayfeweights_rejects_nan_outcome(self):
        data = generate_reversible_did_data(n_groups=20, n_periods=4, seed=1)
        data.loc[data.index[0], "outcome"] = float("nan")
        with pytest.raises(ValueError, match="Outcome column.*NaN"):
            twowayfeweights(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

    def test_twowayfeweights_rejects_non_binary_treatment(self):
        data = generate_reversible_did_data(n_groups=20, n_periods=4, seed=1)
        data.loc[data.index[0], "treatment"] = 2  # non-binary
        with pytest.raises(ValueError, match="binary treatment"):
            twowayfeweights(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

    def test_twowayfeweights_rejects_nan_group(self):
        data = generate_reversible_did_data(n_groups=20, n_periods=4, seed=1)
        data.loc[data.index[0], "group"] = float("nan")
        with pytest.raises(ValueError, match="Group column.*NaN"):
            twowayfeweights(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

    def test_twowayfeweights_rejects_nan_time(self):
        data = generate_reversible_did_data(n_groups=20, n_periods=4, seed=1)
        data.loc[data.index[0], "period"] = float("nan")
        with pytest.raises(ValueError, match="Time column.*NaN"):
            twowayfeweights(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

    def test_fit_rejects_nan_group(self):
        data = generate_reversible_did_data(n_groups=20, n_periods=4, seed=1)
        data.loc[data.index[0], "group"] = float("nan")
        est = ChaisemartinDHaultfoeuille()
        with pytest.raises(ValueError, match="Group column.*NaN"):
            est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

    def test_fit_rejects_nan_time(self):
        data = generate_reversible_did_data(n_groups=20, n_periods=4, seed=1)
        data.loc[data.index[0], "period"] = float("nan")
        est = ChaisemartinDHaultfoeuille()
        with pytest.raises(ValueError, match="Time column.*NaN"):
            est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

    def test_twowayfeweights_rejects_empty_input(self):
        df = pd.DataFrame(columns=["group", "period", "treatment", "outcome"])
        with pytest.raises(ValueError):
            twowayfeweights(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

    def test_twowayfeweights_rejects_within_cell_varying_treatment(self):
        # Construct a panel with two original rows per (group, period) cell
        # where the treatment values disagree within a cell. The helper
        # should raise ValueError (not silently round to majority).
        rows = []
        for g in [1, 2, 3, 4]:
            for t in [0, 1, 2]:
                # Two observations per cell with mixed treatment at t=2 for g=1
                if g == 1 and t == 2:
                    rows.append({"group": g, "period": t, "treatment": 1, "outcome": 10.0})
                    rows.append({"group": g, "period": t, "treatment": 0, "outcome": 11.0})
                else:
                    base_treat = 1 if (g <= 2 and t == 2) else 0
                    rows.append({"group": g, "period": t, "treatment": base_treat, "outcome": 10.0})
                    rows.append({"group": g, "period": t, "treatment": base_treat, "outcome": 10.5})
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="Within-cell-varying treatment"):
            twowayfeweights(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

    def test_fit_rejects_within_cell_varying_treatment(self):
        # Same rejection test via fit() entry point
        rows = []
        for g in [1, 2, 3, 4]:
            for t in [0, 1, 2]:
                if g == 1 and t == 2:
                    rows.append({"group": g, "period": t, "treatment": 1, "outcome": 10.0})
                    rows.append({"group": g, "period": t, "treatment": 0, "outcome": 11.0})
                else:
                    base_treat = 1 if (g <= 2 and t == 2) else 0
                    rows.append({"group": g, "period": t, "treatment": base_treat, "outcome": 10.0})
                    rows.append({"group": g, "period": t, "treatment": base_treat, "outcome": 10.5})
        df = pd.DataFrame(rows)
        est = ChaisemartinDHaultfoeuille()
        with pytest.raises(ValueError, match="Within-cell-varying treatment"):
            est.fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )
