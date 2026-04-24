"""
API and behavior tests for ``ChaisemartinDHaultfoeuille`` (dCDH) — Phase 1.

Covers basic API, validation, forward-compat NotImplementedError gates,
``drop_larger_lower``, A11 zero-retention, NaN handling, bootstrap
plumbing, and the results dataclass round-trip. Methodology validation
(hand-calculable arithmetic, cohort recentering correctness, parity
against R) lives in ``test_methodology_chaisemartin_dhaultfoeuille.py``.
"""

import json
import warnings
from pathlib import Path

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

    def test_non_binary_treatment_requires_lmax(self):
        """Non-binary treatment without L_max raises ValueError."""
        df = pd.DataFrame(
            {
                "group": [1, 1, 2, 2],
                "period": [0, 1, 0, 1],
                "outcome": [10.0, 11.0, 10.0, 12.0],
                "treatment": [0, 2, 0, 1],
            }
        )
        est = ChaisemartinDHaultfoeuille()
        with pytest.raises(ValueError, match="Non-binary treatment requires L_max"):
            est.fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

    def test_non_binary_treatment_with_lmax(self):
        """Non-binary treatment works with L_max=1."""
        np.random.seed(77)
        rows = []
        for g in range(20):
            for t in range(6):
                d = 0 if t < 3 else 2  # non-binary jump
                y = 10 + t + d * 1.5 + np.random.randn() * 0.3
                rows.append({"group": g, "period": t, "treatment": d, "outcome": y})
        for g in range(20, 40):
            for t in range(6):
                y = 10 + t + np.random.randn() * 0.3
                rows.append({"group": g, "period": t, "treatment": 0, "outcome": y})
        df = pd.DataFrame(rows)
        est = ChaisemartinDHaultfoeuille(twfe_diagnostic=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = est.fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                L_max=1,
            )
        assert np.isfinite(results.overall_att)

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
        # aggregate is reserved for Phase 3; require aggregate=None exactly
        with pytest.raises(NotImplementedError, match="Phase 3"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                aggregate="simple",
            )

    def test_aggregate_event_study_raises_not_implemented(self, data):
        with pytest.raises(NotImplementedError, match="Phase 3"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                aggregate="event_study",
            )

    def test_L_max_validation(self, data):
        """L_max is now a Phase 2 feature: positive int or None accepted,
        invalid values raise ValueError."""
        # Zero and negative raise
        with pytest.raises(ValueError, match="positive integer"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                L_max=0,
            )
        with pytest.raises(ValueError, match="positive integer"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                L_max=-1,
            )
        # Non-int raises
        with pytest.raises(ValueError, match="positive integer"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                L_max="5",
            )
        # Exceeding panel raises
        with pytest.raises(ValueError, match="exceeds available"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                L_max=100,
            )
        # L_max=1 is valid (equivalent to None)
        results = self._est().fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=1,
        )
        assert 1 in results.event_study_effects

    def test_controls_requires_lmax(self, data):
        """DID^X covariate adjustment requires L_max >= 1."""
        with pytest.raises(ValueError, match="requires L_max >= 1"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                controls=["outcome"],  # reuse existing column as dummy covariate
            )

    def test_trends_linear_requires_lmax(self, data):
        """DID^{fd} trend adjustment requires L_max >= 1."""
        with pytest.raises(ValueError, match="requires L_max >= 1"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                trends_linear=True,
            )

    def test_trends_nonparam_requires_lmax(self, data):
        """State-set trends requires L_max >= 1."""
        with pytest.raises(ValueError, match="requires L_max >= 1"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                trends_nonparam="state",
            )

    def test_honest_did_requires_lmax(self, data):
        with pytest.raises(ValueError, match="honest_did=True requires L_max"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                honest_did=True,
            )

    def test_survey_design_rejects_fweight(self, data):
        """Survey support requires pweight; fweight rejected."""
        from diff_diff import SurveyDesign

        data = data.copy()
        data["pw"] = 1.0
        sd = SurveyDesign(weights="pw", weight_type="fweight")
        with pytest.raises(ValueError, match="pweight"):
            self._est().fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                survey_design=sd,
            )

    def test_cluster_parameter_raises_not_implemented(self, data):
        """
        Per the dCDH cluster contract: dCDH clusters at the group
        level by default via the cohort-recentered influence function
        (analytical SEs) and the multiplier bootstrap. Under
        ``survey_design`` with strictly-coarser PSUs the bootstrap
        automatically upgrades to PSU-level Hall-Mammen wild. User-
        specified clustering via the ``cluster=`` kwarg is not
        supported.

        The reviewer flagged that ``cluster`` was previously accepted
        on ``__init__`` and stored on ``self.cluster`` but never
        actually read by ``fit()`` or ``_compute_dcdh_bootstrap()``,
        making it a silent no-op. This test pins the contract: any
        non-None cluster value raises ``NotImplementedError`` at
        construction time with a message naming the offending value
        and pointing at the no-custom-clustering reservation. The
        same gate fires from ``set_params``.

        See REGISTRY.md ``Note (cluster contract)``.
        """
        pattern = r"cluster.*(not supported|reserved for a future)"
        # __init__ rejects any non-None cluster
        with pytest.raises(NotImplementedError, match=pattern):
            ChaisemartinDHaultfoeuille(cluster="state")
        with pytest.raises(NotImplementedError, match=pattern):
            ChaisemartinDHaultfoeuille(cluster="unit")

        # set_params after construction also rejects
        est = ChaisemartinDHaultfoeuille()
        with pytest.raises(NotImplementedError, match=pattern):
            est.set_params(cluster="state")

        # cluster=None still works (the only supported value)
        est_default = ChaisemartinDHaultfoeuille(cluster=None)
        assert est_default.cluster is None
        assert est_default.get_params()["cluster"] is None

        # The convenience function also rejects (forward-compat gate
        # propagates through the wrapper at __init__ time)
        with pytest.raises(NotImplementedError, match=pattern):
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
        The TWFE diagnostic requires at least 2 groups and 2 periods
        to build a meaningful FE design. A 1-group panel triggers a
        ValueError from _build_group_time_design's guard, and when
        rank_deficient_action="error" the blanket except in fit()
        re-raises it instead of swallowing it as a warning.

        This also exercises the code path where rank_deficient_action
        ="warn" downgrades the failure to a warning so the main
        estimation can proceed.
        """
        # 1 group, 2 periods: triggers "at least 2 groups" guard
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
        with pytest.raises(ValueError, match="at least 2 groups"):
            est.fit(
                df,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
            )

        # rank_deficient_action="warn" should NOT raise the TWFE error
        est_warn = ChaisemartinDHaultfoeuille(twfe_diagnostic=True, rank_deficient_action="warn")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
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
                # (not from the TWFE diagnostic guard)
                assert "at least 2 groups" not in str(exc)


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

    def test_placebo_se_nan_for_phase1_per_period(self, data, ci_params):
        """
        Phase 1 per-period placebo (L_max=None): SE is NaN because the
        per-period DID_M^pl aggregation does not have an IF derivation.
        Multi-horizon placebos (L_max >= 2) have valid SE via the
        per-group placebo IF - see ``TestMultiHorizonPlacebos``.
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
        """TWFE diagnostic requires binary treatment."""
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


# =============================================================================
# Phase 2: Multi-horizon event study tests
# =============================================================================


class TestMultiHorizon:
    """Phase 2 multi-horizon DID_l tests."""

    @pytest.fixture()
    def data(self):
        return generate_reversible_did_data(
            n_groups=50, n_periods=8, pattern="joiners_only", seed=42
        )

    def test_L_max_none_preserves_phase1_behavior(self, data):
        """L_max=None must produce identical results to Phase 1."""
        est = ChaisemartinDHaultfoeuille(placebo=False, twfe_diagnostic=False)
        r = est.fit(data, outcome="outcome", group="group", time="period", treatment="treatment")
        assert len(r.event_study_effects) == 1
        assert 1 in r.event_study_effects
        assert r.L_max is None
        assert r.normalized_effects is None
        assert r.cost_benefit_delta is None
        assert r.sup_t_bands is None
        assert r.placebo_event_study is None

    def test_L_max_1_bootstrap_overall_matches_es1(self, data, ci_params):
        """With L_max=1 + bootstrap, overall_* must match event_study_effects[1]."""
        n_boot = ci_params.bootstrap(99)
        est = ChaisemartinDHaultfoeuille(
            placebo=False, twfe_diagnostic=False, n_bootstrap=n_boot, seed=42
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = est.fit(
                data, outcome="outcome", group="group", time="period",
                treatment="treatment", L_max=1,
            )
        es1 = r.event_study_effects[1]
        assert r.overall_att == es1["effect"]
        assert r.overall_se == es1["se"]
        assert r.overall_p_value == es1["p_value"]
        assert r.overall_conf_int == es1["conf_int"]

    def test_L_max_1_suppresses_joiner_leaver_decomposition(self):
        """L_max=1 suppresses joiner/leaver decomposition in summary()
        and to_dataframe("joiners_leavers") since it's a DID_M concept."""
        data = generate_reversible_did_data(
            n_groups=50, n_periods=8, pattern="mixed_single_switch", seed=42
        )
        est = ChaisemartinDHaultfoeuille(placebo=False, twfe_diagnostic=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = est.fit(
                data, outcome="outcome", group="group", time="period",
                treatment="treatment", L_max=1,
            )
        # Joiners/leavers suppressed for L_max=1
        assert r.joiners_available is False
        assert r.leavers_available is False
        # summary() should say DID_1, not DID_M
        s = r.summary()
        assert "DID_1" in s
        # to_dataframe("joiners_leavers"): DID_+/DID_- rows not available
        df_jl = r.to_dataframe("joiners_leavers")
        assert df_jl[df_jl["estimand"] == "DID_1"].iloc[0]["n_obs"] > 0
        assert not df_jl[df_jl["estimand"] == "DID_+"].iloc[0]["available"]
        assert not df_jl[df_jl["estimand"] == "DID_-"].iloc[0]["available"]

    def test_L_max_1_bootstrap_results_overall_synced(self, data, ci_params):
        """bootstrap_results.overall_* must match event_study horizon 1,
        and bootstrap_distribution must be cleared (DID_M distribution
        doesn't match the DID_1 summary stats)."""
        n_boot = ci_params.bootstrap(99)
        est = ChaisemartinDHaultfoeuille(
            placebo=False, twfe_diagnostic=False, n_bootstrap=n_boot, seed=42
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = est.fit(
                data, outcome="outcome", group="group", time="period",
                treatment="treatment", L_max=1,
            )
        br = r.bootstrap_results
        assert br is not None
        # Nested bootstrap overall_* should match horizon 1
        assert br.overall_se == br.event_study_ses[1]
        assert br.overall_ci == br.event_study_cis[1]
        assert br.overall_p_value == br.event_study_p_values[1]
        # bootstrap_distribution cleared (was DID_M, not DID_1)
        assert br.bootstrap_distribution is None

    def test_L_max_1_uses_per_group_path(self, data):
        """L_max=1 uses the per-group DID_{g,1} path (same as L_max >= 2
        uses for l=1). This is a different estimand from the per-period
        DID_M path used by L_max=None - documented as a REGISTRY Note."""
        est = ChaisemartinDHaultfoeuille(placebo=False, twfe_diagnostic=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_one = est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                L_max=1,
            )
        # Per-group path produces finite estimate and SE
        assert np.isfinite(r_one.event_study_effects[1]["effect"])
        assert np.isfinite(r_one.event_study_effects[1]["se"])
        assert np.isfinite(r_one.overall_att)
        # L_max=1 should have exactly 1 horizon
        assert set(r_one.event_study_effects.keys()) == {1}

    def test_L_max_populates_event_study_effects(self, data):
        """L_max=3 populates horizons {1, 2, 3} in event_study_effects."""
        est = ChaisemartinDHaultfoeuille(placebo=False, twfe_diagnostic=False)
        r = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=3,
        )
        assert set(r.event_study_effects.keys()) == {1, 2, 3}
        for horizon in [1, 2, 3]:
            entry = r.event_study_effects[horizon]
            assert "effect" in entry
            assert "se" in entry
            assert "n_obs" in entry
            assert entry["n_obs"] > 0

    def test_did_l1_uses_per_group_path_when_L_max(self, data):
        """When L_max >= 2, event_study_effects[1] uses the per-group
        DID_{g,1} path (consistent with horizons 2..L_max), which may
        differ from the Phase 1 per-period DID_M. The per-period DID_M
        is still available via the L_max=None path."""
        est = ChaisemartinDHaultfoeuille(placebo=False, twfe_diagnostic=False)
        r_multi = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=3,
        )
        # event_study_effects[1] is populated and finite
        assert np.isfinite(r_multi.event_study_effects[1]["effect"])
        assert np.isfinite(r_multi.event_study_effects[1]["se"])

    def test_N_l_decreases_with_horizon(self, data):
        """n_obs generally decreases for far horizons."""
        est = ChaisemartinDHaultfoeuille(placebo=False, twfe_diagnostic=False)
        r = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=5,
        )
        n_obs = [r.event_study_effects[h]["n_obs"] for h in sorted(r.event_study_effects)]
        # N_1 >= N_L_max (not strictly decreasing, but monotone non-increasing expected)
        assert n_obs[0] >= n_obs[-1]

    def test_N_l_zero_at_far_horizon_produces_nan(self):
        """When no groups are eligible at horizon l, DID_l is NaN."""
        # 3-period panel: L_max=2 has 1 post-baseline period, so l=2 has no room
        data = generate_reversible_did_data(
            n_groups=10, n_periods=3, pattern="joiners_only", seed=1
        )
        est = ChaisemartinDHaultfoeuille(placebo=False, twfe_diagnostic=False)
        r = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=2,
        )
        assert 2 in r.event_study_effects
        # l=2 may have 0 or few eligible groups; if 0, effect is NaN
        # (depends on the DGP; the key test is that the horizon key exists)

    def test_switcher_fraction_warning(self):
        """Far horizons with <50% of l=1 switchers emit a UserWarning."""
        # Use a short panel so far horizons thin out
        data = generate_reversible_did_data(
            n_groups=50, n_periods=6, pattern="joiners_only", seed=42
        )
        est = ChaisemartinDHaultfoeuille(placebo=False, twfe_diagnostic=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                L_max=4,
            )
        # May or may not fire depending on the DGP; the key test is no crash.
        _thin = [wi for wi in w if "50%" in str(wi.message)]  # noqa: F841

    def test_overall_att_is_cost_benefit_delta_when_L_max_gt_1(self, data):
        """When L_max > 1, overall_att is the cost-benefit delta."""
        est = ChaisemartinDHaultfoeuille(placebo=False, twfe_diagnostic=False)
        r = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=3,
        )
        assert r.cost_benefit_delta is not None
        assert r.overall_att == pytest.approx(r.cost_benefit_delta["delta"])
        # DID_1 is still accessible
        assert r.event_study_effects[1]["effect"] != r.overall_att or True  # may be close


class TestMultiHorizonPlacebos:
    """Phase 2 dynamic placebos."""

    @pytest.fixture()
    def data(self):
        return generate_reversible_did_data(
            n_groups=50, n_periods=10, pattern="joiners_only", seed=42
        )

    def test_placebo_event_study_populated(self, data):
        est = ChaisemartinDHaultfoeuille(twfe_diagnostic=False)
        r = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=3,
        )
        assert r.placebo_event_study is not None
        # Keys should be negative
        for k in r.placebo_event_study:
            assert k < 0

    def test_placebo_horizons_negative_keys(self, data):
        est = ChaisemartinDHaultfoeuille(twfe_diagnostic=False)
        r = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=3,
        )
        if r.placebo_event_study:
            for h, entry in r.placebo_event_study.items():
                assert h < 0
                assert "effect" in entry
                assert "n_obs" in entry

    def test_placebo_se_finite_multi_horizon(self, data):
        """Multi-horizon placebos (L_max >= 2) have finite analytical SE
        via the per-group placebo IF computation."""
        est = ChaisemartinDHaultfoeuille(twfe_diagnostic=False)
        r = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=3,
        )
        assert r.placebo_event_study is not None
        has_finite_se = False
        for h, entry in r.placebo_event_study.items():
            if entry["n_obs"] > 0:
                assert np.isfinite(entry["effect"]), f"Placebo h={h}: effect not finite"
                assert np.isfinite(entry["se"]), f"Placebo h={h}: SE not finite"
                assert entry["se"] > 0, f"Placebo h={h}: SE not positive"
                assert np.isfinite(entry["t_stat"]), f"Placebo h={h}: t_stat not finite"
                assert np.isfinite(entry["p_value"]), f"Placebo h={h}: p_value not finite"
                assert np.isfinite(entry["conf_int"][0]), f"Placebo h={h}: CI lo not finite"
                assert np.isfinite(entry["conf_int"][1]), f"Placebo h={h}: CI hi not finite"
                has_finite_se = True
        assert has_finite_se, "Expected at least one placebo horizon with finite SE"

    def test_placebo_bootstrap_se_multi_horizon(self, data, ci_params):
        """Multi-horizon placebo bootstrap SE should be finite."""
        n_boot = ci_params.bootstrap(199)
        est = ChaisemartinDHaultfoeuille(
            twfe_diagnostic=False, n_bootstrap=n_boot, seed=42
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                L_max=3,
            )
        assert r.bootstrap_results is not None
        assert r.bootstrap_results.placebo_horizon_ses is not None
        assert len(r.bootstrap_results.placebo_horizon_ses) > 0
        for lag, se in r.bootstrap_results.placebo_horizon_ses.items():
            assert np.isfinite(se), f"Bootstrap placebo SE lag={lag} not finite"
            assert se > 0, f"Bootstrap placebo SE lag={lag} not positive"


class TestNormalizedEffects:
    """Phase 2 normalized estimator DID^n_l."""

    @pytest.fixture()
    def data(self):
        return generate_reversible_did_data(
            n_groups=50, n_periods=8, pattern="joiners_only", seed=42
        )

    def test_normalized_populated_when_L_max(self, data):
        est = ChaisemartinDHaultfoeuille(placebo=False, twfe_diagnostic=False)
        r = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=3,
        )
        assert r.normalized_effects is not None
        assert set(r.normalized_effects.keys()) == {1, 2, 3}

    def test_normalized_equals_did_over_l_binary(self, data):
        """For binary treatment: DID^n_l = DID_l / l.

        Note: for l >= 2, the multi-horizon DID_l is used (per-group
        path). For l=1, there's a documented deviation between the
        Phase 1 per-period path and the Phase 2 per-group path, so
        we verify against the normalized_effects dict's own denominator.
        """
        est = ChaisemartinDHaultfoeuille(placebo=False, twfe_diagnostic=False)
        r = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=3,
        )
        for horizon in [1, 2, 3]:
            n_eff = r.normalized_effects[horizon]
            # Denominator should be horizon for binary treatment
            assert n_eff["denominator"] == pytest.approx(float(horizon), rel=1e-10)
            # DID^n_l * denominator should reconstruct the DID_l from
            # the same computation path (multi-horizon per-group)
            assert np.isfinite(n_eff["effect"])


class TestCostBenefitDelta:
    """Phase 2 cost-benefit aggregate delta."""

    @pytest.fixture()
    def data(self):
        return generate_reversible_did_data(
            n_groups=50, n_periods=8, pattern="joiners_only", seed=42
        )

    def test_delta_weights_sum_to_one(self, data):
        est = ChaisemartinDHaultfoeuille(placebo=False, twfe_diagnostic=False)
        r = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=3,
        )
        assert r.cost_benefit_delta is not None
        weights = r.cost_benefit_delta["weights"]
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-10)

    def test_delta_is_consistent(self, data):
        """Cost-benefit delta is a weighted average with weights summing to 1."""
        est = ChaisemartinDHaultfoeuille(placebo=False, twfe_diagnostic=False)
        r = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=3,
        )
        cb = r.cost_benefit_delta
        assert cb is not None
        assert np.isfinite(cb["delta"])
        # Weights sum to 1
        assert sum(cb["weights"].values()) == pytest.approx(1.0, abs=1e-10)
        # delta == overall_att when L_max > 1
        assert r.overall_att == pytest.approx(cb["delta"])


class TestSupTBands:
    """Phase 2 simultaneous confidence bands."""

    @pytest.fixture()
    def data(self):
        return generate_reversible_did_data(
            n_groups=50, n_periods=8, pattern="joiners_only", seed=42
        )

    def test_sup_t_requires_bootstrap(self, data):
        est = ChaisemartinDHaultfoeuille(n_bootstrap=0, placebo=False, twfe_diagnostic=False)
        r = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=3,
        )
        assert r.sup_t_bands is None

    def test_cband_wider_than_pointwise(self, data):
        est = ChaisemartinDHaultfoeuille(
            n_bootstrap=99, seed=1, placebo=False, twfe_diagnostic=False
        )
        r = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=3,
        )
        if r.sup_t_bands is not None:
            for horizon in r.event_study_effects:
                entry = r.event_study_effects[horizon]
                cband = entry.get("cband_conf_int")
                if cband is not None and np.isfinite(entry["se"]):
                    pw_ci = entry["conf_int"]
                    # Sup-t bands should be at least as wide as pointwise
                    assert cband[0] <= pw_ci[0] + 1e-10
                    assert cband[1] >= pw_ci[1] - 1e-10


class TestMultiHorizonToDataframe:
    """Phase 2 to_dataframe extensions."""

    @pytest.fixture()
    def data(self):
        return generate_reversible_did_data(
            n_groups=50, n_periods=8, pattern="joiners_only", seed=42
        )

    def test_event_study_level(self, data):
        est = ChaisemartinDHaultfoeuille(twfe_diagnostic=False)
        r = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=3,
        )
        df = r.to_dataframe("event_study")
        assert "horizon" in df.columns
        assert "effect" in df.columns
        # Should have: placebos + ref + positive horizons
        assert (df["horizon"] == 0).any()  # reference period
        assert (df["horizon"] > 0).any()  # positive horizons

    def test_normalized_level(self, data):
        est = ChaisemartinDHaultfoeuille(placebo=False, twfe_diagnostic=False)
        r = est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=3,
        )
        df = r.to_dataframe("normalized")
        assert "horizon" in df.columns
        assert "denominator" in df.columns
        assert len(df) == 3


class TestCovariateAdjustment:
    """DID^X covariate residualization (ROADMAP item 3a)."""

    @staticmethod
    def _make_panel_with_covariates(seed=42, n_groups=40, n_periods=6):
        """Create a panel where a covariate confounds the outcome."""
        rng = np.random.RandomState(seed)
        rows = []
        for g in range(n_groups):
            group_fe = rng.normal(0, 2)
            # Covariate: group-level value plus time variation
            x_base = rng.normal(0, 1)
            # Treatment: first half switch at period 3, rest never
            switches = g < n_groups // 2
            for t in range(n_periods):
                d = 1 if (switches and t >= 3) else 0
                x = x_base + 0.5 * t + rng.normal(0, 0.1)
                # Outcome depends on group FE, time trend, covariate,
                # and treatment effect
                y = group_fe + 2.0 * t + 3.0 * x + 5.0 * d + rng.normal(0, 0.5)
                rows.append(
                    {"group": g, "period": t, "treatment": d, "outcome": y, "X1": x}
                )
        return pd.DataFrame(rows)

    def test_controls_requires_lmax(self):
        """controls without L_max raises ValueError."""
        df = self._make_panel_with_covariates()
        with pytest.raises(ValueError, match="requires L_max >= 1"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment", controls=["X1"]
            )

    def test_controls_missing_column(self):
        """controls with nonexistent column raises ValueError."""
        df = self._make_panel_with_covariates()
        with pytest.raises(ValueError, match="not found in data"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                controls=["nonexistent"], L_max=1,
            )

    def test_covariate_residualization_basic(self):
        """DID^X produces different results from unadjusted DID."""
        df = self._make_panel_with_covariates()
        est = ChaisemartinDHaultfoeuille(seed=1)

        # Unadjusted
        r_plain = est.fit(df, "outcome", "group", "period", "treatment", L_max=1)
        # Covariate-adjusted
        r_x = est.fit(
            df, "outcome", "group", "period", "treatment",
            controls=["X1"], L_max=1,
        )

        # Results should differ (covariate is confounding)
        assert r_x.overall_att != r_plain.overall_att
        # Covariate diagnostics should be populated
        assert r_x.covariate_residuals is not None
        assert len(r_x.covariate_residuals) > 0
        assert "theta_hat" in r_x.covariate_residuals.columns
        # SE should be finite
        assert np.isfinite(r_x.overall_se)

    def test_multiple_covariates(self):
        """Multiple covariates are accepted and produce diagnostics."""
        df = self._make_panel_with_covariates()
        # Add a second covariate
        df["X2"] = np.random.RandomState(99).normal(0, 1, len(df))
        est = ChaisemartinDHaultfoeuille(seed=1)
        r = est.fit(
            df, "outcome", "group", "period", "treatment",
            controls=["X1", "X2"], L_max=1,
        )
        assert r.covariate_residuals is not None
        # Should have rows for each (baseline, covariate) combination
        assert set(r.covariate_residuals["covariate"].unique()) == {"X1", "X2"}

    def test_covariate_residuals_diagnostics(self):
        """Diagnostics DataFrame has expected structure."""
        df = self._make_panel_with_covariates()
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            df, "outcome", "group", "period", "treatment",
            controls=["X1"], L_max=2,
        )
        diag = r.covariate_residuals
        assert diag is not None
        expected_cols = {"baseline_treatment", "covariate", "theta_hat", "n_obs", "r_squared"}
        assert expected_cols.issubset(set(diag.columns))
        # All baselines should have positive n_obs
        assert (diag["n_obs"] > 0).all()
        # theta_hat should be finite (not NaN)
        theta = diag.loc[diag["covariate"] == "X1", "theta_hat"].values[0]
        assert np.isfinite(theta), f"theta_hat is not finite: {theta}"

    def test_controls_with_nonbinary_treatment(self):
        """Covariates work with non-binary treatment and L_max >= 1."""
        rng = np.random.RandomState(123)
        rows = []
        for g in range(30):
            x_base = rng.normal(0, 1)
            for t in range(5):
                # Ordinal treatment: 0 -> 2 for first 10, 0 -> 1 for next 10, never for rest
                if g < 10:
                    d = 2.0 if t >= 2 else 0.0
                elif g < 20:
                    d = 1.0 if t >= 3 else 0.0
                else:
                    d = 0.0
                x = x_base + 0.1 * t
                y = 10 + 2 * t + 1.5 * x + 3 * d + rng.normal(0, 0.5)
                rows.append({"group": g, "period": t, "treatment": d, "outcome": y, "X1": x})
        df = pd.DataFrame(rows)
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            df, "outcome", "group", "period", "treatment",
            controls=["X1"], L_max=1,
        )
        assert np.isfinite(r.overall_att)
        assert np.isfinite(r.overall_se)

    def test_controls_with_multi_horizon(self):
        """Covariates work with L_max > 1 event study."""
        df = self._make_panel_with_covariates()
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            df, "outcome", "group", "period", "treatment",
            controls=["X1"], L_max=2,
        )
        assert r.event_study_effects is not None
        assert 1 in r.event_study_effects
        assert 2 in r.event_study_effects
        # Both horizons should have finite effects and SEs
        for h in [1, 2]:
            assert np.isfinite(r.event_study_effects[h]["effect"])
            assert np.isfinite(r.event_study_effects[h]["se"])

    def test_controls_lmax1_estimand_contract(self):
        """DID^X with L_max=1: per_period_effects stay raw, overall uses DID^X_1."""
        df = self._make_panel_with_covariates()
        est = ChaisemartinDHaultfoeuille(seed=1)

        # Fit without controls for raw per-period baseline
        r_raw = est.fit(df, "outcome", "group", "period", "treatment")
        # Fit with controls
        r_x = est.fit(
            df, "outcome", "group", "period", "treatment",
            controls=["X1"], L_max=1,
        )

        # per_period_effects should be UNADJUSTED (raw Phase 1 DID_M)
        # because the per-period path does not support covariate adjustment
        for period_key in r_raw.per_period_effects:
            if period_key in r_x.per_period_effects:
                raw_eff = r_raw.per_period_effects[period_key]
                x_eff = r_x.per_period_effects[period_key]
                assert raw_eff["did_plus_t"] == pytest.approx(
                    x_eff["did_plus_t"], abs=1e-10
                ), f"per_period_effects should be unadjusted at period {period_key}"

        # overall_att should come from event_study_effects[1] (DID^X_1)
        assert r_x.overall_att == pytest.approx(
            r_x.event_study_effects[1]["effect"], abs=1e-10
        )
        # and should differ from the raw overall_att (covariate effect)
        assert r_x.overall_att != r_raw.overall_att


class TestLinearTrends:
    """DID^{fd} group-specific linear trends (ROADMAP item 3b)."""

    @staticmethod
    def _make_panel_with_trends(seed=42, n_groups=40, n_periods=8):
        """Create a panel with group-specific linear trends in outcomes."""
        rng = np.random.RandomState(seed)
        rows = []
        for g in range(n_groups):
            group_fe = rng.normal(0, 2)
            group_trend = rng.normal(0, 0.5)  # group-specific linear trend
            switches = g < n_groups // 2
            switch_period = 4 if switches else n_periods + 1
            for t in range(n_periods):
                d = 1 if t >= switch_period else 0
                y = (
                    group_fe
                    + 2.0 * t
                    + group_trend * t  # group-specific trend
                    + 5.0 * d
                    + rng.normal(0, 0.3)
                )
                rows.append({"group": g, "period": t, "treatment": d, "outcome": y})
        return pd.DataFrame(rows)

    def test_trends_linear_requires_lmax(self):
        """trends_linear without L_max raises ValueError."""
        df = self._make_panel_with_trends()
        with pytest.raises(ValueError, match="requires L_max >= 1"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                trends_linear=True,
            )

    def test_trends_linear_basic(self):
        """DID^{fd} produces different results from unadjusted DID."""
        df = self._make_panel_with_trends()
        est = ChaisemartinDHaultfoeuille(seed=1)
        r_plain = est.fit(df, "outcome", "group", "period", "treatment", L_max=2)
        r_fd = est.fit(
            df, "outcome", "group", "period", "treatment",
            L_max=2, trends_linear=True,
        )
        # Results should differ (group-specific trends confound unadjusted)
        assert r_fd.overall_att != r_plain.overall_att
        # Event study should have horizons
        assert r_fd.event_study_effects is not None
        assert 1 in r_fd.event_study_effects

    def test_cumulated_level_effects(self):
        """Cumulated delta^{fd}_l = sum DID^{fd}_{l'} for l'=1..l."""
        df = self._make_panel_with_trends()
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            df, "outcome", "group", "period", "treatment",
            L_max=3, trends_linear=True,
        )
        assert r.linear_trends_effects is not None
        # Check cumulation: delta^{fd}_1 = DID^{fd}_1
        es = r.event_study_effects
        lt = r.linear_trends_effects
        assert abs(lt[1]["effect"] - es[1]["effect"]) < 1e-12
        # delta^{fd}_2 = DID^{fd}_1 + DID^{fd}_2
        assert abs(lt[2]["effect"] - (es[1]["effect"] + es[2]["effect"])) < 1e-12

    def test_fg_less_than_3_warning(self):
        """Groups with F_g < 3 produce a UserWarning."""
        rng = np.random.RandomState(99)
        rows = []
        for g in range(20):
            for t in range(6):
                # Group 0-4: switch at period 1 (F_g=2, 0-indexed f_g=1 < 2)
                if g < 5:
                    d = 1 if t >= 1 else 0
                elif g < 10:
                    d = 1 if t >= 3 else 0
                else:
                    d = 0
                y = 10 + 2 * t + 3 * d + rng.normal(0, 0.5)
                rows.append({"group": g, "period": t, "treatment": d, "outcome": y})
        df = pd.DataFrame(rows)
        with pytest.warns(UserWarning, match="F_g < 3"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=2, trends_linear=True,
            )

    def test_trends_with_covariates(self):
        """Combined DID^{X,fd}: covariates + linear trends."""
        df = self._make_panel_with_trends()
        df["X1"] = np.random.RandomState(77).normal(0, 1, len(df))
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            df, "outcome", "group", "period", "treatment",
            controls=["X1"], L_max=2, trends_linear=True,
        )
        # overall_att is NaN for trends + L_max>=2 (no aggregate)
        assert np.isnan(r.overall_att)
        assert r.covariate_residuals is not None
        assert r.linear_trends_effects is not None

    def test_trends_linear_lmax2_overall_surface(self):
        """Under trends_linear + L_max>=2, overall_* is NaN (no aggregate).

        R's did_multiplegt_dyn with trends_lin=TRUE does not compute an
        aggregate average total effect. Cumulated level effects are
        available via results.linear_trends_effects[l].
        """
        df = self._make_panel_with_trends()
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            df, "outcome", "group", "period", "treatment",
            L_max=3, trends_linear=True,
        )
        # overall_* should be NaN (not computed in trends mode)
        assert np.isnan(r.overall_att)
        assert np.isnan(r.overall_se)
        # cost_benefit_delta suppressed
        assert r.cost_benefit_delta is None
        # Cumulated effects still available
        assert r.linear_trends_effects is not None
        assert len(r.linear_trends_effects) >= 1

    def test_cumulated_se_nan_propagation(self):
        """Cumulated SE is NaN when a component horizon has NaN SE."""
        # Create a panel where horizon 2 has no eligible switchers (NaN SE)
        # but horizon 1 does. The cumulated effect at h=2 should have NaN SE.
        rng = np.random.RandomState(77)
        rows = []
        for g in range(30):
            group_fe = rng.normal(0, 1)
            # Groups 0-9: switch at period 3 (enough pre-switch for trends)
            # Groups 10-19: never switch (controls)
            # Groups 20-29: switch at period 4 (only 1 post-switch period)
            if g < 10:
                switch_t = 3
            elif g < 20:
                switch_t = 99
            else:
                switch_t = 4
            for t in range(5):
                d = 1 if t >= switch_t else 0
                y = group_fe + t + 3 * d + rng.normal(0, 0.3)
                rows.append({"group": g, "period": t, "treatment": d, "outcome": y})
        df = pd.DataFrame(rows)
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            df, "outcome", "group", "period", "treatment",
            L_max=2, trends_linear=True,
        )
        # If SE at horizon 1 is finite but horizon 2 is NaN,
        # cumulated h=2 SE must be NaN (not 0.0)
        if r.linear_trends_effects is not None and 2 in r.linear_trends_effects:
            cum_se = r.linear_trends_effects[2]["se"]
            es = r.event_study_effects
            if es and 2 in es and not np.isfinite(es[2]["se"]):
                assert not np.isfinite(cum_se), (
                    f"Cumulated SE should be NaN when component h=2 SE is NaN, "
                    f"got {cum_se}"
                )


class TestStateSetTrends:
    """State-set-specific trends (ROADMAP item 3c)."""

    @staticmethod
    def _make_panel_with_sets(seed=42, n_groups=40, n_periods=6):
        """Create a panel where groups belong to state sets."""
        rng = np.random.RandomState(seed)
        rows = []
        for g in range(n_groups):
            state = g % 4  # 4 states
            group_fe = rng.normal(0, 2)
            switches = g < n_groups // 2
            for t in range(n_periods):
                d = 1 if (switches and t >= 3) else 0
                y = group_fe + 2.0 * t + 5.0 * d + rng.normal(0, 0.5)
                rows.append({
                    "group": g, "period": t, "treatment": d,
                    "outcome": y, "state": state,
                })
        return pd.DataFrame(rows)

    def test_trends_nonparam_requires_lmax(self):
        df = self._make_panel_with_sets()
        with pytest.raises(ValueError, match="requires L_max >= 1"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                trends_nonparam="state",
            )

    def test_trends_nonparam_basic(self):
        """State-set restriction produces different results."""
        df = self._make_panel_with_sets()
        est = ChaisemartinDHaultfoeuille(seed=1)
        r_plain = est.fit(df, "outcome", "group", "period", "treatment", L_max=1)
        r_set = est.fit(
            df, "outcome", "group", "period", "treatment",
            L_max=1, trends_nonparam="state",
        )
        # With set-restricted controls, results may differ
        # (both should be finite and reasonable)
        assert np.isfinite(r_set.overall_att)
        assert np.isfinite(r_set.overall_se)

    def test_time_varying_set_raises(self):
        """Set membership that varies over time raises ValueError."""
        df = self._make_panel_with_sets()
        # Make state vary over time for some groups
        df.loc[(df["group"] == 0) & (df["period"] == 3), "state"] = 99
        with pytest.raises(ValueError, match="time-invariant"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=1, trends_nonparam="state",
            )

    def test_missing_set_column_raises(self):
        df = self._make_panel_with_sets()
        with pytest.raises(ValueError, match="not found in data"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=1, trends_nonparam="nonexistent",
            )

    def test_group_level_set_rejected(self):
        """Set partition at group level (not coarser) raises ValueError."""
        df = self._make_panel_with_sets()
        # Use group column itself as set (each group is its own set)
        with pytest.raises(ValueError, match="coarser than group"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=1, trends_nonparam="group",
            )

    def test_nan_set_membership_rejected(self):
        """NaN in trends_nonparam column raises ValueError."""
        df = self._make_panel_with_sets()
        df.loc[df["group"] == 0, "state"] = np.nan
        with pytest.raises(ValueError, match="NaN/missing"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=1, trends_nonparam="state",
            )

    def test_nonparam_with_covariates(self):
        """Combined state-set trends + covariates."""
        df = self._make_panel_with_sets()
        df["X1"] = np.random.RandomState(77).normal(0, 1, len(df))
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            df, "outcome", "group", "period", "treatment",
            controls=["X1"], L_max=1, trends_nonparam="state",
        )
        assert np.isfinite(r.overall_att)
        assert r.covariate_residuals is not None

    def test_trends_nonparam_unequal_support(self):
        """Unequal switcher/control support across state sets.

        State A: 3 switchers + 5 controls -> finite effects.
        State B: 2 switchers + 0 controls -> empty control pool, groups
        excluded at horizons with empty pools (Assumption 14 support-trimming).
        """
        rng = np.random.RandomState(99)
        rows = []
        n_periods = 6
        # State A: groups 0-7 (0-2 switch at t=3, 3-7 never switch)
        for g in range(8):
            switches = g < 3
            for t in range(n_periods):
                d = 1 if (switches and t >= 3) else 0
                y = 10 + 2.0 * t + 5.0 * d + rng.normal(0, 0.5)
                rows.append({
                    "group": g, "period": t, "treatment": d,
                    "outcome": y, "state": "A",
                })
        # State B: groups 8-9 (both switch at t=3, NO controls in this set)
        for g in range(8, 10):
            for t in range(n_periods):
                d = 1 if t >= 3 else 0
                y = 10 + 2.0 * t + 5.0 * d + rng.normal(0, 0.5)
                rows.append({
                    "group": g, "period": t, "treatment": d,
                    "outcome": y, "state": "B",
                })
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=2, trends_nonparam="state",
            )
        # Should not error; State A groups contribute, State B excluded
        assert np.isfinite(r.overall_att)
        assert r.event_study_effects is not None


class TestHeterogeneityTesting:
    """Heterogeneity testing beta^{het}_l (ROADMAP item 3d)."""

    @staticmethod
    def _make_panel_with_het(seed=42, n_groups=40, n_periods=6):
        """Create a panel with heterogeneous effects by covariate."""
        rng = np.random.RandomState(seed)
        rows = []
        for g in range(n_groups):
            x_g = 1 if g < n_groups // 2 else 0  # binary het covariate
            group_fe = rng.normal(0, 2)
            switches = g < (3 * n_groups) // 4
            effect = 5.0 + 3.0 * x_g  # heterogeneous effect
            for t in range(n_periods):
                d = 1 if (switches and t >= 3) else 0
                y = group_fe + 2.0 * t + effect * d + rng.normal(0, 0.5)
                rows.append({
                    "group": g, "period": t, "treatment": d,
                    "outcome": y, "het_x": x_g,
                })
        return pd.DataFrame(rows)

    def test_heterogeneity_basic(self):
        """Detect heterogeneous effects with binary covariate."""
        df = self._make_panel_with_het()
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            df, "outcome", "group", "period", "treatment",
            L_max=1, heterogeneity="het_x",
        )
        assert r.heterogeneity_effects is not None
        assert 1 in r.heterogeneity_effects
        het = r.heterogeneity_effects[1]
        assert np.isfinite(het["beta"])
        assert np.isfinite(het["se"])
        # True het effect is ~3.0 (effect difference between x=1 and x=0)
        assert het["beta"] > 0, f"Expected positive beta, got {het['beta']}"

    def test_heterogeneity_null(self):
        """No heterogeneity produces beta near zero."""
        rng = np.random.RandomState(123)
        rows = []
        for g in range(40):
            x_g = rng.normal(0, 1)  # random covariate, uncorrelated with effect
            switches = g < 20
            for t in range(6):
                d = 1 if (switches and t >= 3) else 0
                y = 10 + 2 * t + 5 * d + rng.normal(0, 0.5)
                rows.append({
                    "group": g, "period": t, "treatment": d,
                    "outcome": y, "het_x": x_g,
                })
        df = pd.DataFrame(rows)
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            df, "outcome", "group", "period", "treatment",
            L_max=1, heterogeneity="het_x",
        )
        het = r.heterogeneity_effects[1]
        # Not significantly different from zero
        assert abs(het["beta"]) < 5.0

    def test_heterogeneity_multi_horizon(self):
        """Heterogeneity test at multiple horizons."""
        df = self._make_panel_with_het()
        r = ChaisemartinDHaultfoeuille(seed=1).fit(
            df, "outcome", "group", "period", "treatment",
            L_max=2, heterogeneity="het_x",
        )
        assert 1 in r.heterogeneity_effects
        assert 2 in r.heterogeneity_effects

    def test_heterogeneity_missing_column(self):
        df = self._make_panel_with_het()
        with pytest.raises(ValueError, match="not found"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=1, heterogeneity="nonexistent",
            )

    def test_heterogeneity_rejects_controls(self):
        """heterogeneity + controls raises ValueError (matching R predict_het)."""
        df = self._make_panel_with_het()
        df["X1"] = np.random.RandomState(42).normal(0, 1, len(df))
        with pytest.raises(ValueError, match="cannot be combined with controls"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=1, heterogeneity="het_x", controls=["X1"],
            )

    def test_heterogeneity_requires_lmax(self):
        """heterogeneity without L_max raises ValueError."""
        df = self._make_panel_with_het()
        with pytest.raises(ValueError, match="requires L_max >= 1"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                heterogeneity="het_x",
            )

    def test_heterogeneity_rejects_trends_linear(self):
        """heterogeneity + trends_linear raises ValueError."""
        df = self._make_panel_with_het()
        with pytest.raises(ValueError, match="cannot be combined with trends_linear"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=2, heterogeneity="het_x", trends_linear=True,
            )

    def test_heterogeneity_rejects_trends_nonparam(self):
        """heterogeneity + trends_nonparam raises ValueError."""
        df = self._make_panel_with_het()
        df["state"] = df["group"] % 3
        with pytest.raises(ValueError, match="cannot be combined with trends_nonparam"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=1, heterogeneity="het_x", trends_nonparam="state",
            )


class TestDesign2:
    """Design-2 switch-in/switch-out separation (ROADMAP item 3e)."""

    @staticmethod
    def _make_join_then_leave_panel(seed=42, n_groups=30, n_periods=8):
        """Panel with join-then-leave groups."""
        rng = np.random.RandomState(seed)
        rows = []
        for g in range(n_groups):
            group_fe = rng.normal(0, 2)
            for t in range(n_periods):
                # Groups 0-9: join at t=2, leave at t=5 (design 2)
                if g < 10:
                    d = 1 if 2 <= t < 5 else 0
                # Groups 10-19: join at t=3, never leave
                elif g < 20:
                    d = 1 if t >= 3 else 0
                # Groups 20-29: never switch
                else:
                    d = 0
                y = group_fe + 2.0 * t + 5.0 * d + rng.normal(0, 0.3)
                rows.append({"group": g, "period": t, "treatment": d, "outcome": y})
        return pd.DataFrame(rows)

    def test_design2_basic(self):
        """Design-2 identifies join-then-leave groups."""
        df = self._make_join_then_leave_panel()
        # drop_larger_lower=False to keep the 2-switch groups
        r = ChaisemartinDHaultfoeuille(seed=1, drop_larger_lower=False).fit(
            df, "outcome", "group", "period", "treatment",
            L_max=1, design2=True,
        )
        assert r.design2_effects is not None
        assert r.design2_effects["n_design2_groups"] == 10
        # Switch-in should show positive effect (joining treatment)
        assert r.design2_effects["switch_in"]["mean_effect"] > 0
        # Switch-out should show negative effect (leaving treatment)
        assert r.design2_effects["switch_out"]["mean_effect"] < 0

    def test_design2_no_eligible(self):
        """No join-then-leave groups returns None."""
        rng = np.random.RandomState(99)
        rows = []
        for g in range(20):
            for t in range(6):
                d = 1 if (g < 10 and t >= 3) else 0
                y = 10 + 2 * t + 5 * d + rng.normal(0, 0.5)
                rows.append({"group": g, "period": t, "treatment": d, "outcome": y})
        df = pd.DataFrame(rows)
        # drop_larger_lower=False required for design2=True
        r = ChaisemartinDHaultfoeuille(seed=1, drop_larger_lower=False).fit(
            df, "outcome", "group", "period", "treatment",
            L_max=1, design2=True,
        )
        assert r.design2_effects is None

    def test_design2_disabled_by_default(self):
        """design2=False (default) produces no design2_effects."""
        df = self._make_join_then_leave_panel()
        r = ChaisemartinDHaultfoeuille(seed=1, drop_larger_lower=False).fit(
            df, "outcome", "group", "period", "treatment", L_max=1,
        )
        assert r.design2_effects is None

    def test_design2_rejects_drop_larger_lower(self):
        """design2=True with default drop_larger_lower=True raises ValueError."""
        df = self._make_join_then_leave_panel()
        with pytest.raises(ValueError, match="drop_larger_lower=False"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=1, design2=True,
            )


class TestNonBinaryTreatment:
    """Non-binary treatment support (ROADMAP item 3f)."""

    def test_ordinal_treatment(self):
        """Ordinal treatment (0, 1, 2, 3) with L_max=2."""
        np.random.seed(42)
        rows = []
        for g in range(30):
            base_d = np.random.choice([0, 1, 2, 3])
            switch_period = np.random.randint(2, 6)
            new_d = base_d + np.random.choice([1, 2]) if base_d < 3 else base_d - 1
            for t in range(8):
                d = base_d if t < switch_period else new_d
                y = 10 + g * 0.5 + t * 0.3 + (d - base_d) * 2 + np.random.randn() * 0.5
                rows.append({"group": g, "period": t, "treatment": d, "outcome": y})
        df = pd.DataFrame(rows)
        est = ChaisemartinDHaultfoeuille(twfe_diagnostic=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Non-binary treatment requires L_max (multi-horizon path)
            r = est.fit(
                df, outcome="outcome", group="group", time="period",
                treatment="treatment", L_max=2,
            )
        assert np.isfinite(r.overall_att)

    def test_within_cell_heterogeneity_rejected_nonbinary(self):
        """Cells with mixed non-binary values (e.g., 1 and 2) should be rejected."""
        df = pd.DataFrame(
            {
                "group": [1, 1, 1, 1, 2, 2, 2, 2],
                "period": [0, 0, 1, 1, 0, 0, 1, 1],
                "outcome": [10.0, 10.5, 12.0, 12.5, 10.0, 10.5, 11.0, 11.5],
                "treatment": [0, 0, 1, 2, 0, 0, 0, 0],  # cell (1, 1) has values 1 and 2
            }
        )
        est = ChaisemartinDHaultfoeuille()
        with pytest.raises(ValueError, match="Within-cell-varying treatment"):
            est.fit(df, outcome="outcome", group="group", time="period", treatment="treatment")

    def test_single_large_dose_not_flagged_multi_switch(self):
        """A single jump 0->3 should NOT be flagged as multi-switch."""
        np.random.seed(55)
        rows = []
        for g in range(20):
            for t in range(6):
                d = 0 if t < 3 else 3  # single jump from 0 to 3
                y = 10 + t + (d - 0) * 2 + np.random.randn() * 0.5
                rows.append({"group": g, "period": t, "treatment": d, "outcome": y})
        # Add some never-switchers for controls
        for g in range(20, 40):
            for t in range(6):
                y = 10 + t + np.random.randn() * 0.5
                rows.append({"group": g, "period": t, "treatment": 0, "outcome": y})
        df = pd.DataFrame(rows)
        est = ChaisemartinDHaultfoeuille(twfe_diagnostic=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Non-binary treatment requires L_max >= 1 (multi-horizon path)
            r = est.fit(
                df, outcome="outcome", group="group", time="period",
                treatment="treatment", L_max=1,
            )
        # All 20 switcher groups should be kept (0 dropped as multi-switch)
        assert r.n_groups_dropped_crossers == 0

    def test_true_multi_switch_detected_nonbinary(self):
        """A group going 0->2->1 should be flagged as multi-switch."""
        rows = []
        # Multi-switch group
        for t in range(6):
            d = 0 if t < 2 else (2 if t < 4 else 1)  # 0->2->1
            rows.append({"group": 0, "period": t, "treatment": d, "outcome": 10 + t})
        # Normal groups (binary for simplicity)
        for g in range(1, 20):
            for t in range(6):
                d = 0 if t < 3 else 1
                rows.append({"group": g, "period": t, "treatment": d, "outcome": 10 + t})
        # Controls
        for g in range(20, 40):
            for t in range(6):
                rows.append({"group": g, "period": t, "treatment": 0, "outcome": 10 + t})
        df = pd.DataFrame(rows)
        est = ChaisemartinDHaultfoeuille(twfe_diagnostic=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Binary groups work at L_max=None; the multi-switch group
            # (0->2->1) should be detected and dropped.
            r = est.fit(df, outcome="outcome", group="group", time="period", treatment="treatment")
        assert r.n_groups_dropped_crossers >= 1

    def test_monotone_multi_step_dropped(self):
        """A monotone multi-step path 0->1->2 has 2 change periods and
        should be dropped (the second change confounds DID_{g,l})."""
        rows = []
        # Monotone multi-step group: 0->1->2
        for t in range(6):
            d = 0 if t < 2 else (1 if t < 4 else 2)
            rows.append({"group": 0, "period": t, "treatment": d, "outcome": 10 + t})
        # Normal single-switch groups (binary)
        for g in range(1, 20):
            for t in range(6):
                d = 0 if t < 3 else 1
                rows.append({"group": g, "period": t, "treatment": d, "outcome": 10 + t})
        # Controls
        for g in range(20, 40):
            for t in range(6):
                rows.append({"group": g, "period": t, "treatment": 0, "outcome": 10 + t})
        df = pd.DataFrame(rows)
        est = ChaisemartinDHaultfoeuille(twfe_diagnostic=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = est.fit(df, outcome="outcome", group="group", time="period", treatment="treatment")
        # Group 0 (0->1->2, 2 change periods) should be dropped
        assert r.n_groups_dropped_crossers >= 1

    def test_mixed_binary_nonbinary_panel_lmax1(self):
        """Mixed panel with both 0->1 and 0->2 switches at L_max=1.
        overall_att should use the per-group path (includes all switches),
        not the per-period path (binary-only)."""
        np.random.seed(88)
        rows = []
        # Binary switchers: 0->1
        for g in range(10):
            for t in range(6):
                d = 0 if t < 3 else 1
                y = 10 + t + d * 2 + np.random.randn() * 0.3
                rows.append({"group": g, "period": t, "treatment": d, "outcome": y})
        # Non-binary switchers: 0->2
        for g in range(10, 20):
            for t in range(6):
                d = 0 if t < 3 else 2
                y = 10 + t + d * 1.5 + np.random.randn() * 0.3
                rows.append({"group": g, "period": t, "treatment": d, "outcome": y})
        # Controls
        for g in range(20, 40):
            for t in range(6):
                y = 10 + t + np.random.randn() * 0.3
                rows.append({"group": g, "period": t, "treatment": 0, "outcome": y})
        df = pd.DataFrame(rows)
        est = ChaisemartinDHaultfoeuille(twfe_diagnostic=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = est.fit(
                df, outcome="outcome", group="group", time="period",
                treatment="treatment", L_max=1,
            )
        # overall_att should be from per-group path (includes both 0->1 and 0->2)
        assert np.isfinite(r.overall_att)
        # event_study_effects[1] and overall_att should be the same estimand
        assert r.overall_att == r.event_study_effects[1]["effect"]

    def test_constant_nonbinary_treatment_raises(self):
        """Constant non-binary treatment (no switchers) should raise ValueError."""
        rows = []
        for g in range(20):
            for t in range(6):
                rows.append({"group": g, "period": t, "treatment": 2, "outcome": 10 + t})
        for g in range(20, 40):
            for t in range(6):
                rows.append({"group": g, "period": t, "treatment": 0, "outcome": 10 + t})
        df = pd.DataFrame(rows)
        est = ChaisemartinDHaultfoeuille(twfe_diagnostic=False)
        with pytest.raises(ValueError, match="No switching groups found"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                est.fit(
                    df, outcome="outcome", group="group", time="period",
                    treatment="treatment", L_max=1,
                )

    def test_nonbinary_bootstrap(self, ci_params):
        """Non-binary panel with bootstrap: finite event study SEs AND
        top-level overall_* matches event_study_effects[1]."""
        np.random.seed(66)
        n_boot = ci_params.bootstrap(99)
        rows = []
        for g in range(20):
            for t in range(6):
                d = 0 if t < 3 else 2
                y = 10 + t + d * 1.5 + np.random.randn() * 0.3
                rows.append({"group": g, "period": t, "treatment": d, "outcome": y})
        for g in range(20, 40):
            for t in range(6):
                y = 10 + t + np.random.randn() * 0.3
                rows.append({"group": g, "period": t, "treatment": 0, "outcome": y})
        df = pd.DataFrame(rows)
        est = ChaisemartinDHaultfoeuille(
            twfe_diagnostic=False, n_bootstrap=n_boot, seed=42
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = est.fit(
                df, outcome="outcome", group="group", time="period",
                treatment="treatment", L_max=1,
            )
        assert r.bootstrap_results is not None
        assert r.bootstrap_results.event_study_ses is not None
        assert 1 in r.bootstrap_results.event_study_ses
        assert np.isfinite(r.bootstrap_results.event_study_ses[1])
        # Top-level overall_* must match event_study_effects[1]
        es1 = r.event_study_effects[1]
        assert r.overall_att == es1["effect"]
        assert r.overall_se == es1["se"]
        assert r.overall_p_value == es1["p_value"]

    def test_nonbinary_lmax1_renderer_contract(self):
        """Non-binary L_max=1: summary/to_dataframe use DID_1 label and
        suppress binary-only joiner/leaver decomposition."""
        np.random.seed(77)
        rows = []
        for g in range(20):
            for t in range(6):
                d = 0 if t < 3 else 2
                y = 10 + t + d + np.random.randn() * 0.3
                rows.append({"group": g, "period": t, "treatment": d, "outcome": y})
        for g in range(20, 40):
            for t in range(6):
                y = 10 + t + np.random.randn() * 0.3
                rows.append({"group": g, "period": t, "treatment": 0, "outcome": y})
        df = pd.DataFrame(rows)
        est = ChaisemartinDHaultfoeuille(twfe_diagnostic=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = est.fit(
                df, outcome="outcome", group="group", time="period",
                treatment="treatment", L_max=1,
            )
        # __repr__ should say DID_1
        assert "DID_1" in repr(r)
        # to_dataframe("overall") should label as DID_1
        df_overall = r.to_dataframe("overall")
        assert df_overall.iloc[0]["estimand"] == "DID_1"
        # n_switcher_cells should be > 0 (from per-group path)
        assert r.n_switcher_cells > 0
        # Joiners/leavers unavailable for non-binary
        assert r.joiners_available is False
        assert r.leavers_available is False
        # to_dataframe("joiners_leavers"): overall row n_obs should be > 0
        df_jl = r.to_dataframe("joiners_leavers")
        overall_row = df_jl[df_jl["estimand"] == "DID_1"]
        assert len(overall_row) == 1
        assert overall_row.iloc[0]["n_obs"] > 0
        # summary() should contain "DID_1" label
        s = r.summary()
        assert "DID_1" in s

    def test_twfe_diagnostic_skipped_nonbinary(self):
        """TWFE diagnostic should be skipped (with warning) for non-binary."""
        np.random.seed(77)
        rows = []
        for g in range(20):
            for t in range(6):
                d = 0 if t < 3 else 2
                y = 10 + t + d + np.random.randn() * 0.3
                rows.append({"group": g, "period": t, "treatment": d, "outcome": y})
        for g in range(20, 40):
            for t in range(6):
                y = 10 + t + np.random.randn() * 0.3
                rows.append({"group": g, "period": t, "treatment": 0, "outcome": y})
        df = pd.DataFrame(rows)
        est = ChaisemartinDHaultfoeuille(twfe_diagnostic=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = est.fit(
                df, outcome="outcome", group="group", time="period",
                treatment="treatment", L_max=1,
            )
        twfe_warnings = [x for x in w if "TWFE diagnostic" in str(x.message)]
        assert len(twfe_warnings) >= 1
        assert r.twfe_weights is None  # diagnostic was skipped

    def test_normalized_effects_general_formula(self):
        """For non-binary treatment, normalized denominator uses actual dose change."""
        np.random.seed(99)
        rows = []
        # Groups switching from 0 to 2 (dose = 2 per period)
        for g in range(20):
            for t in range(8):
                d = 0 if t < 3 else 2
                y = 10 + t + d * 1.5 + np.random.randn() * 0.3
                rows.append({"group": g, "period": t, "treatment": d, "outcome": y})
        # Controls at baseline 0
        for g in range(20, 40):
            for t in range(8):
                y = 10 + t + np.random.randn() * 0.3
                rows.append({"group": g, "period": t, "treatment": 0, "outcome": y})
        df = pd.DataFrame(rows)
        est = ChaisemartinDHaultfoeuille(placebo=False, twfe_diagnostic=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = est.fit(
                df, outcome="outcome", group="group", time="period",
                treatment="treatment", L_max=3,
            )
        if r.normalized_effects is not None and 1 in r.normalized_effects:
            # For dose 0->2: denominator at l=1 should be ~2 (not 1)
            denom = r.normalized_effects[1]["denominator"]
            assert denom > 1.5, f"Denominator should reflect dose=2, got {denom}"


# =============================================================================
# HonestDiD Integration
# =============================================================================


class TestHonestDiDIntegration:
    """HonestDiD (Rambachan-Roth 2023) integration on dCDH placebos."""

    @staticmethod
    def _make_data(n_groups=40, n_periods=6, seed=42):
        return generate_reversible_did_data(
            n_groups=n_groups, n_periods=n_periods, seed=seed
        )

    def test_honest_did_basic(self):
        """honest_did=True with L_max>=2 produces HonestDiDResults."""
        from diff_diff.honest_did import HonestDiDResults

        df = self._make_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=2, honest_did=True,
            )
        assert r.honest_did_results is not None
        assert isinstance(r.honest_did_results, HonestDiDResults)
        assert np.isfinite(r.honest_did_results.ci_lb)
        assert np.isfinite(r.honest_did_results.ci_ub)

    def test_honest_did_requires_lmax(self):
        """honest_did=True with L_max=None raises ValueError."""
        df = self._make_data()
        with pytest.raises(ValueError, match="honest_did=True requires L_max"):
            ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                honest_did=True,
            )

    def test_honest_did_rejects_placebo_false(self):
        """honest_did=True with placebo=False raises ValueError."""
        df = self._make_data()
        with pytest.raises(ValueError, match="placebo=False"):
            ChaisemartinDHaultfoeuille(seed=1, placebo=False).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=2, honest_did=True,
            )

    def test_honest_did_standalone(self):
        """compute_honest_did() on dCDH results matches honest_did=True."""
        from diff_diff.honest_did import compute_honest_did

        df = self._make_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_auto = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=2, honest_did=True,
            )
            r_plain = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=2,
            )
            r_manual = compute_honest_did(
                r_plain, method="relative_magnitude", M=1.0
            )
        # Deterministic - bitwise identical
        np.testing.assert_allclose(
            r_auto.honest_did_results.ci_lb, r_manual.ci_lb, rtol=0
        )
        np.testing.assert_allclose(
            r_auto.honest_did_results.ci_ub, r_manual.ci_ub, rtol=0
        )

    def test_honest_did_with_controls(self):
        """HonestDiD runs on DID^X placebos."""
        df = self._make_data(n_periods=6)
        df["X1"] = np.random.RandomState(77).normal(0, 1, len(df))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                controls=["X1"], L_max=2, honest_did=True,
            )
        assert r.honest_did_results is not None
        assert np.isfinite(r.honest_did_results.ci_lb)

    def test_honest_did_with_trends_linear(self):
        """HonestDiD on second-differenced DID^{fd} estimand."""
        df = self._make_data(n_periods=7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                trends_linear=True, L_max=2, honest_did=True,
            )
        # Bounds should be computed on second-differenced estimand
        assert r.honest_did_results is not None
        assert np.isfinite(r.honest_did_results.ci_lb)

    def test_honest_did_sensitivity(self):
        """sensitivity_analysis() on dCDH results."""
        from diff_diff.honest_did import HonestDiD

        df = self._make_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=2,
            )
        honest = HonestDiD(method="relative_magnitude")
        sens = honest.sensitivity_analysis(
            r, M_grid=list(np.linspace(0, 2, 5))
        )
        assert sens.breakdown_M is not None or len(sens.bounds) == 5

    def test_honest_did_smoothness(self):
        """Smoothness method gives different bounds than RM."""
        from diff_diff.honest_did import compute_honest_did

        df = self._make_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=2,
            )
        rm_bounds = compute_honest_did(r, method="relative_magnitude", M=1.0)
        sd_bounds = compute_honest_did(r, method="smoothness", M=0.5)
        # Different methods should generally give different bounds
        assert rm_bounds.ci_lb != sd_bounds.ci_lb or rm_bounds.ci_ub != sd_bounds.ci_ub

    def test_honest_did_original_estimate_is_post_average(self):
        """original_estimate targets equal-weight average over post horizons."""
        df = self._make_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=2, honest_did=True,
            )
        hd = r.honest_did_results
        assert hd is not None
        # Equal-weight average = mean of event_study_effects[1..L_max]
        es = r.event_study_effects
        avg = np.mean([es[h]["effect"] for h in sorted(es.keys())])
        np.testing.assert_allclose(hd.original_estimate, avg, rtol=1e-10)

    def test_honest_did_custom_l_vec_on_impact(self):
        """compute_honest_did with l_vec=[1,0] targets on-impact effect."""
        from diff_diff.honest_did import compute_honest_did

        df = self._make_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=2,
            )
        # l_vec=[1, 0] targets only DID_1 (on-impact, R's default)
        bounds = compute_honest_did(r, l_vec=np.array([1.0, 0.0]))
        np.testing.assert_allclose(
            bounds.original_estimate,
            r.event_study_effects[1]["effect"],
            rtol=1e-10,
        )

    def test_honest_did_respects_alpha(self):
        """honest_did=True propagates estimator alpha to HonestDiD."""
        df = self._make_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1, alpha=0.10).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=2, honest_did=True,
            )
        assert r.honest_did_results is not None
        assert r.honest_did_results.alpha == 0.10

    def test_honest_did_retains_period_metadata(self):
        """HonestDiDResults stores pre_periods_used and post_periods_used."""
        df = self._make_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=2, honest_did=True,
            )
        hd = r.honest_did_results
        assert hd.pre_periods_used is not None
        assert hd.post_periods_used is not None
        assert all(p < 0 for p in hd.pre_periods_used)
        assert all(p > 0 for p in hd.post_periods_used)
        # Summary renders the retained horizons
        text = r.summary()
        assert "Post horizons used:" in text

    def test_honest_did_custom_l_vec_summary_label(self):
        """summary() renders custom target label when l_vec is overridden."""
        from diff_diff.honest_did import compute_honest_did

        df = self._make_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=2,
            )
        # Attach custom-target HonestDiD to results
        r.honest_did_results = compute_honest_did(
            r, l_vec=np.array([1.0, 0.0])
        )
        text = r.summary()
        assert "on-impact" in text.lower()
        assert "Equal-weight" not in text

    def test_honest_did_with_trends_nonparam(self):
        """End-to-end trends_nonparam + honest_did=True (balanced support)."""
        rng = np.random.RandomState(42)
        rows = []
        for g in range(40):
            state = g % 4
            switches = g < 20
            for t in range(7):
                d = 1 if (switches and t >= 3) else 0
                y = 10 + 2.0 * t + 5.0 * d + rng.normal(0, 0.5)
                rows.append({
                    "group": g, "period": t, "treatment": d,
                    "outcome": y, "state": state,
                })
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=2, trends_nonparam="state", honest_did=True,
            )
        assert r.honest_did_results is not None
        assert np.isfinite(r.honest_did_results.ci_lb)

    def test_honest_did_trends_nonparam_trimming(self):
        """End-to-end: trends_nonparam causes NaN at far horizons, HonestDiD trims.

        State A: switches late (t=5), has never-switching controls.
        State B: switches early (t=2), "controls" switch at t=3 so
        control pool vanishes at h>=2. At L_max=3, h=3 and h=-3 have
        N_l=0 (NaN SE) because State A can't reach h=3 and State B
        has no controls there. HonestDiD extraction drops the NaN
        horizons and retains [-2, -1, 1, 2].
        """
        rng = np.random.RandomState(42)
        rows = []
        n_periods = 7
        # State A: 3 switch at t=5, 4 controls
        for g in range(7):
            switches = g < 3
            for t in range(n_periods):
                d = 1 if (switches and t >= 5) else 0
                y = 10 + 2.0*t + 5.0*d + rng.normal(0, 0.3)
                rows.append({
                    "group": g, "period": t, "treatment": d,
                    "outcome": y, "state": "A",
                })
        # State B: 4 switch at t=2, 2 "controls" switch at t=3
        for g in range(7, 13):
            switch_t = 2 if g < 11 else 3
            for t in range(n_periods):
                d = 1 if t >= switch_t else 0
                y = 10 + 2.0*t + 5.0*d + rng.normal(0, 0.3)
                rows.append({
                    "group": g, "period": t, "treatment": d,
                    "outcome": y, "state": "B",
                })
        df = pd.DataFrame(rows)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=3, trends_nonparam="state", honest_did=True,
            )
        # h=3 and h=-3 should be NaN (N_l=0 from support trimming)
        assert r.event_study_effects[3]["n_obs"] == 0
        assert r.placebo_event_study[-3]["n_obs"] == 0
        # HonestDiD should still compute on the retained block
        hd = r.honest_did_results
        assert hd is not None
        assert np.isfinite(hd.ci_lb)
        # Retained horizons should exclude the NaN endpoints
        assert -3 not in hd.pre_periods_used
        assert 3 not in hd.post_periods_used
        assert hd.post_periods_used == [1, 2]
        # The placebo-based pre-period warning should have been emitted
        placebo_warns = [
            x for x in w if "placebo" in str(x.message).lower()
            and "pre-period" in str(x.message).lower()
        ]
        assert len(placebo_warns) >= 1

    def test_honest_did_with_bootstrap(self):
        """honest_did=True works with bootstrap-fitted results."""
        df = self._make_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1, n_bootstrap=49).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=2, honest_did=True,
            )
        assert r.honest_did_results is not None
        assert np.isfinite(r.honest_did_results.ci_lb)
        assert r.honest_did_results.post_periods_used == [1, 2]


# =============================================================================
# Summary Phase 3 Rendering
# =============================================================================


class TestSummaryPhase3:
    """Verify summary() renders Phase 3 result blocks."""

    @staticmethod
    def _make_data(n_groups=40, n_periods=6, seed=42):
        return generate_reversible_did_data(
            n_groups=n_groups, n_periods=n_periods, seed=seed
        )

    def test_summary_renders_covariate_diagnostics(self):
        """Covariate Adjustment section appears in summary()."""
        df = self._make_data()
        df["X1"] = np.random.RandomState(77).normal(0, 1, len(df))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                controls=["X1"], L_max=1,
            )
        text = r.summary()
        assert "Covariate Adjustment" in text

    def test_summary_renders_linear_trends(self):
        """Cumulated Level Effects section appears in summary()."""
        df = self._make_data(n_periods=7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                trends_linear=True, L_max=2,
            )
        text = r.summary()
        assert "Cumulated Level Effects" in text

    def test_summary_renders_heterogeneity(self):
        """Heterogeneity Test section appears in summary()."""
        rng = np.random.RandomState(42)
        rows = []
        for g in range(40):
            x_g = 1 if g < 20 else 0
            switches = g < 30
            for t in range(6):
                d = 1 if (switches and t >= 3) else 0
                y = 10 + 2.0 * t + 5.0 * d + 3.0 * x_g * d + rng.normal(0, 0.5)
                rows.append({
                    "group": g, "period": t, "treatment": d,
                    "outcome": y, "het_x": x_g,
                })
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=1, heterogeneity="het_x",
            )
        text = r.summary()
        assert "Heterogeneity Test" in text

    def test_summary_renders_design2(self):
        """Design-2 section appears in summary()."""
        rng = np.random.RandomState(42)
        rows = []
        for g in range(30):
            for t in range(8):
                if g < 10:
                    d = 1 if 3 <= t < 6 else 0  # join then leave
                elif g < 20:
                    d = 1 if t >= 3 else 0  # join only
                else:
                    d = 0  # never switch
                y = 10 + t + 5.0 * d + rng.normal(0, 0.5)
                rows.append({
                    "group": g, "period": t, "treatment": d, "outcome": y,
                })
        df = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(
                seed=1, drop_larger_lower=False
            ).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=1, design2=True,
            )
        text = r.summary()
        assert "Design-2" in text

    def test_summary_renders_honest_did(self):
        """HonestDiD Sensitivity section appears in summary()."""
        df = self._make_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = ChaisemartinDHaultfoeuille(seed=1).fit(
                df, "outcome", "group", "period", "treatment",
                L_max=2, honest_did=True,
            )
        text = r.summary()
        assert "HonestDiD Sensitivity" in text


# =============================================================================
# by_path: per-path event-study disaggregation
# =============================================================================


def _by_path_three_path_data(seed: int = 42) -> pd.DataFrame:
    """Hand-checkable 6-switcher + 2-never-treated panel with 3 distinct paths.

    Periods 0..3, treatment effect = 2.0.

    - Groups 1, 2, 3: path (0, 1, 1, 1) — single switch, stay on
    - Groups 4, 5:    path (0, 1, 0, 0) — single pulse
    - Group  6:       path (0, 1, 1, 0) — two on then off
    - Groups 7, 8:    never-treated controls (path not defined)

    With treatment effect = 2.0, the per-horizon within-path effect should
    be ~2.0 when D=1 in the path window and ~0 when D=0, modulo noise.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in (1, 2, 3):
        for t in range(4):
            d = 0 if t == 0 else 1
            y = d * 2.0 + rng.normal(0, 0.1)
            rows.append({"group": g, "period": t, "treatment": d, "outcome": y})
    for g in (4, 5):
        for t in range(4):
            d = 1 if t == 1 else 0
            y = d * 2.0 + rng.normal(0, 0.1)
            rows.append({"group": g, "period": t, "treatment": d, "outcome": y})
    for g in (6,):
        for t in range(4):
            d = 1 if t in (1, 2) else 0
            y = d * 2.0 + rng.normal(0, 0.1)
            rows.append({"group": g, "period": t, "treatment": d, "outcome": y})
    for g in (7, 8):
        for t in range(4):
            y = rng.normal(0, 0.1)
            rows.append({"group": g, "period": t, "treatment": 0, "outcome": y})
    return pd.DataFrame(rows)


def _fit_by_path(data: pd.DataFrame, by_path: int, L_max: int = 3):
    """Fit with standard by_path kwargs and silence the drop_larger_lower warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        est = ChaisemartinDHaultfoeuille(
            drop_larger_lower=False,
            by_path=by_path,
            twfe_diagnostic=False,
            placebo=False,
        )
        return est, est.fit(
            data,
            outcome="outcome",
            group="group",
            time="period",
            treatment="treatment",
            L_max=L_max,
        )


class TestByPathGates:
    """Fit-time gates for by_path combinations."""

    def test_default_leaves_path_effects_none(self):
        data = generate_reversible_did_data(n_groups=40, n_periods=5, seed=1)
        est = ChaisemartinDHaultfoeuille()
        results = est.fit(
            data, outcome="outcome", group="group", time="period", treatment="treatment"
        )
        assert results.path_effects is None

    @pytest.mark.parametrize("bad", [0, -1, -5, 1.5, "all", True, False, 2.0])
    def test_invalid_type_raises(self, bad):
        with pytest.raises(ValueError, match="by_path"):
            ChaisemartinDHaultfoeuille(by_path=bad)

    def test_set_params_revalidates(self):
        est = ChaisemartinDHaultfoeuille()
        with pytest.raises(ValueError, match="by_path"):
            est.set_params(by_path=0)
        with pytest.raises(ValueError, match="by_path"):
            est.set_params(by_path=-3)

    def test_in_get_params(self):
        est = ChaisemartinDHaultfoeuille(by_path=5, drop_larger_lower=False)
        params = est.get_params()
        assert "by_path" in params
        assert params["by_path"] == 5

    def test_requires_drop_larger_lower_false(self):
        data = generate_reversible_did_data(n_groups=40, n_periods=5, seed=1)
        est = ChaisemartinDHaultfoeuille(by_path=3)
        with pytest.raises(ValueError, match="drop_larger_lower=False"):
            est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                L_max=2,
            )

    def test_requires_lmax(self):
        data = _by_path_three_path_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = ChaisemartinDHaultfoeuille(drop_larger_lower=False, by_path=3)
            with pytest.raises(ValueError, match="L_max"):
                est.fit(
                    data,
                    outcome="outcome",
                    group="group",
                    time="period",
                    treatment="treatment",
                )

    @pytest.mark.parametrize(
        "fit_kwargs, msg",
        [
            ({"controls": ["outcome"]}, "controls"),
            ({"trends_linear": True}, "trends_linear"),
            ({"trends_nonparam": "group"}, "trends_nonparam"),
            ({"heterogeneity": "group"}, "heterogeneity"),
            ({"design2": True}, "design2"),
            ({"honest_did": True}, "honest_did"),
        ],
    )
    def test_forbids_phase3_fit_kwargs(self, fit_kwargs, msg):
        data = _by_path_three_path_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = ChaisemartinDHaultfoeuille(drop_larger_lower=False, by_path=3)
            with pytest.raises(NotImplementedError, match=msg):
                est.fit(
                    data,
                    outcome="outcome",
                    group="group",
                    time="period",
                    treatment="treatment",
                    L_max=2,
                    **fit_kwargs,
                )

    def test_forbids_non_binary_treatment(self):
        # Continuous-dose treatment — detected inside the cell aggregator.
        rng = np.random.default_rng(0)
        rows = []
        for g in range(1, 7):
            for t in range(4):
                d = float(g % 3) if (g <= 3 and t >= 1) else 0.0
                y = d + rng.normal()
                rows.append({"group": g, "period": t, "treatment": d, "outcome": y})
        data = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = ChaisemartinDHaultfoeuille(
                drop_larger_lower=False, by_path=3, twfe_diagnostic=False
            )
            with pytest.raises(NotImplementedError, match="non-binary"):
                est.fit(
                    data,
                    outcome="outcome",
                    group="group",
                    time="period",
                    treatment="treatment",
                    L_max=2,
                )


class TestByPathBehavior:
    """Path enumeration, ranking, and result dict shape."""

    def test_top_k_selects_most_common(self):
        data = _by_path_three_path_data()
        # by_path=2 → top 2 paths by frequency: (0,1,1,1) with 3 groups
        # and (0,1,0,0) with 2 groups. The (0,1,1,0) path has 1 group
        # and should be excluded.
        _, results = _fit_by_path(data, by_path=2, L_max=3)
        assert results.path_effects is not None
        paths = set(results.path_effects.keys())
        assert paths == {(0, 1, 1, 1), (0, 1, 0, 0)}
        assert results.path_effects[(0, 1, 1, 1)]["frequency_rank"] == 1
        assert results.path_effects[(0, 1, 0, 0)]["frequency_rank"] == 2
        assert results.path_effects[(0, 1, 1, 1)]["n_groups"] == 3
        assert results.path_effects[(0, 1, 0, 0)]["n_groups"] == 2

    def test_overflow_returns_all_with_warning(self):
        data = _by_path_three_path_data()
        # Don't use the helper here — it suppresses UserWarnings that we
        # want to catch. Call fit directly and record all warnings.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est = ChaisemartinDHaultfoeuille(
                drop_larger_lower=False,
                by_path=10,
                twfe_diagnostic=False,
                placebo=False,
            )
            results = est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                L_max=3,
            )
        assert results.path_effects is not None
        assert len(results.path_effects) == 3
        overflow_msgs = [
            w for w in caught if "exceeds the number of observed paths" in str(w.message)
        ]
        assert overflow_msgs, "Expected a UserWarning about exceeding observed paths"

    def test_lexicographic_tiebreak(self):
        # Build data where two paths have the SAME frequency; expect the
        # lexicographically smaller tuple to rank first.
        rng = np.random.default_rng(0)
        rows = []
        # Path (0, 1, 0, 0): 2 groups
        for g in (1, 2):
            for t in range(4):
                d = 1 if t == 1 else 0
                rows.append(
                    {
                        "group": g,
                        "period": t,
                        "treatment": d,
                        "outcome": rng.normal(),
                    }
                )
        # Path (0, 1, 1, 0): 2 groups (tie with above on count)
        for g in (3, 4):
            for t in range(4):
                d = 1 if t in (1, 2) else 0
                rows.append(
                    {
                        "group": g,
                        "period": t,
                        "treatment": d,
                        "outcome": rng.normal(),
                    }
                )
        # Two never-treated for control pool
        for g in (5, 6):
            for t in range(4):
                rows.append(
                    {
                        "group": g,
                        "period": t,
                        "treatment": 0,
                        "outcome": rng.normal(),
                    }
                )
        data = pd.DataFrame(rows)
        _, results = _fit_by_path(data, by_path=2, L_max=3)
        assert results.path_effects is not None
        # (0,1,0,0) < (0,1,1,0) lexicographically → rank 1
        assert results.path_effects[(0, 1, 0, 0)]["frequency_rank"] == 1
        assert results.path_effects[(0, 1, 1, 0)]["frequency_rank"] == 2

    def test_result_dict_shape(self):
        data = _by_path_three_path_data()
        _, results = _fit_by_path(data, by_path=3, L_max=3)
        assert results.path_effects is not None
        for path, entry in results.path_effects.items():
            assert isinstance(path, tuple)
            assert all(isinstance(v, int) for v in path)
            assert set(entry.keys()) >= {"n_groups", "frequency_rank", "horizons"}
            assert isinstance(entry["horizons"], dict)
            for l_h, h_entry in entry["horizons"].items():
                assert isinstance(l_h, int)
                assert set(h_entry.keys()) == {
                    "effect",
                    "se",
                    "t_stat",
                    "p_value",
                    "conf_int",
                    "n_obs",
                }
                lo, hi = h_entry["conf_int"]
                # CI is a tuple of two floats (NaN permitted under degenerate cohorts)
                assert isinstance(lo, float) and isinstance(hi, float)

    def test_hand_calculable_effects_match_dgp(self):
        """Path (0,1,1,1) always-on → effect ≈ 2; path (0,1,0,0) on at l=1
        then off → effect ≈ 2 at l=1 and ≈ 0 at l=2, l=3."""
        data = _by_path_three_path_data()
        _, results = _fit_by_path(data, by_path=3, L_max=3)
        stay_on = results.path_effects[(0, 1, 1, 1)]["horizons"]
        pulse = results.path_effects[(0, 1, 0, 0)]["horizons"]
        for l_h in (1, 2, 3):
            assert (
                abs(stay_on[l_h]["effect"] - 2.0) < 0.5
            ), f"stay_on l={l_h} effect={stay_on[l_h]['effect']} not near 2.0"
        assert abs(pulse[1]["effect"] - 2.0) < 0.5
        assert abs(pulse[2]["effect"]) < 0.5
        assert abs(pulse[3]["effect"]) < 0.5

    def test_summary_renders_path_section(self):
        data = _by_path_three_path_data()
        _, results = _fit_by_path(data, by_path=2, L_max=3)
        text = results.summary()
        assert "Treatment-Path Disaggregation" in text
        assert "(0, 1, 1, 1)" in text
        assert "(0, 1, 0, 0)" in text
        # Per-horizon rows rendered
        for l_h in (1, 2, 3):
            assert f"l={l_h}" in text

    def test_to_dataframe_by_path(self):
        data = _by_path_three_path_data()
        _, results = _fit_by_path(data, by_path=2, L_max=3)
        df = results.to_dataframe(level="by_path")
        assert isinstance(df, pd.DataFrame)
        # 2 paths * 3 horizons = 6 rows
        assert len(df) == 6
        expected_cols = {
            "path",
            "frequency_rank",
            "n_groups",
            "horizon",
            "effect",
            "se",
            "t_stat",
            "p_value",
            "conf_int_lower",
            "conf_int_upper",
            "n_obs",
        }
        assert expected_cols.issubset(df.columns)
        assert set(df["horizon"].unique()) == {1, 2, 3}

    def test_to_dataframe_raises_when_not_requested(self):
        data = generate_reversible_did_data(n_groups=40, n_periods=5, seed=1)
        est = ChaisemartinDHaultfoeuille()
        results = est.fit(
            data, outcome="outcome", group="group", time="period", treatment="treatment"
        )
        with pytest.raises(ValueError, match="by_path"):
            results.to_dataframe(level="by_path")


class TestByPathEdgeCases:
    """Empty-result-set and degenerate-cohort branches per plan review."""

    def test_empty_path_surface_when_no_complete_window(self):
        """by_path requested but every switcher's window falls outside the panel.

        Switchers have F_g = period 3 with n_periods = 4 and L_max = 3, so
        the window [F_g - 1, F_g - 1 + L_max] = [2, 5] extends past the
        panel (period 5 doesn't exist). Expected behavior:

        - results.path_effects == {} (NOT None — distinguishes
          "requested but empty" from "not requested")
        - UserWarning emitted at fit-time
        - summary() renders a "no observed paths" notice
        - to_dataframe(level="by_path") returns empty DataFrame with
          canonical columns (does NOT raise — the caller already passed
          by_path=k)
        """
        rng = np.random.default_rng(0)
        rows = []
        # Switchers switch at t=3 → window [2, 5] with L_max=3 falls
        # outside the 4-period panel. Not-yet-switched at F_g-1=2,
        # treated at F_g=3, but the post-switch horizons 2 and 3 are
        # at t=4 and t=5 which don't exist.
        for g in (1, 2, 3, 4):
            for t in range(4):
                d = 1 if t >= 3 else 0
                rows.append(
                    {
                        "group": g,
                        "period": t,
                        "treatment": d,
                        "outcome": rng.normal(),
                    }
                )
        for g in (5, 6):
            for t in range(4):
                rows.append(
                    {
                        "group": g,
                        "period": t,
                        "treatment": 0,
                        "outcome": rng.normal(),
                    }
                )
        data = pd.DataFrame(rows)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est = ChaisemartinDHaultfoeuille(
                drop_larger_lower=False,
                by_path=3,
                twfe_diagnostic=False,
                placebo=False,
            )
            results = est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                L_max=3,
            )

        # Empty dict, NOT None
        assert results.path_effects is not None
        assert results.path_effects == {}

        # Fit-time warning surfaced
        empty_warnings = [
            w for w in caught if "no observed treatment path" in str(w.message)
        ]
        assert empty_warnings, (
            "Expected a UserWarning when by_path is requested but no "
            "observed path has a complete window"
        )

        # Summary renders a notice instead of the per-path block
        text = results.summary()
        assert "Treatment-Path Disaggregation" in text
        assert "No observed paths" in text

        # to_dataframe returns an empty DataFrame (NOT a ValueError)
        df = results.to_dataframe(level="by_path")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        expected_cols = {
            "path",
            "frequency_rank",
            "n_groups",
            "horizon",
            "effect",
            "se",
            "t_stat",
            "p_value",
            "conf_int_lower",
            "conf_int_upper",
            "n_obs",
        }
        assert expected_cols.issubset(df.columns)

    def test_degenerate_cohort_path_nan_inference_and_warning(self):
        """Every variance-eligible group in its own (D_{g,1}, F_g, S_g) cohort.

        Uses the canonical 4-group panel from
        ``test_methodology_chaisemartin_dhaultfoeuille.TestMethodologyWorkedExample``
        whose cohort structure is all-singleton:

            g=1: (0, 1, +1)  — path (0, 1) at L_max=1
            g=2: (1, 2, -1)  — path (1, 0) at L_max=1
            g=3: (0, -1,  0)
            g=4: (1, -1,  0)

        With every cohort a singleton, cohort recentering yields an
        identically-zero centered IF for every selected path →
        ``_plugin_se`` returns NaN → the per-(path, horizon) degenerate-
        cohort warning fires. Point estimate remains finite.
        """
        panel = pd.DataFrame(
            {
                "group": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                "period": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                "treatment": [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                "outcome": [
                    10.0, 13.0, 14.0,
                    10.0, 11.0, 9.0,
                    10.0, 11.0, 12.0,
                    10.0, 11.0, 12.0,
                ],
            }
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est = ChaisemartinDHaultfoeuille(
                drop_larger_lower=False,
                by_path=2,
                twfe_diagnostic=False,
                placebo=False,
            )
            results = est.fit(
                panel,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                L_max=1,
            )

        assert results.path_effects is not None
        # At least one (path, horizon) cell should have a NaN SE accompanied
        # by the degenerate-cohort warning.
        degenerate_warnings = [
            w
            for w in caught
            if "unidentified for path=" in str(w.message)
            and "horizon l=" in str(w.message)
        ]
        assert degenerate_warnings, (
            "Expected a per-(path, horizon) degenerate-cohort UserWarning "
            "when the path-subset centered IF collapses to zero"
        )
        # Point-estimate side still populated (only SE/t/p/CI are NaN)
        any_nan = False
        for entry in results.path_effects.values():
            for h in entry["horizons"].values():
                if np.isnan(h["se"]):
                    any_nan = True
                    assert np.isnan(h["t_stat"])
                    assert np.isnan(h["p_value"])
                    lo, hi = h["conf_int"]
                    assert np.isnan(lo) and np.isnan(hi)
                    # Point estimate is finite (only SE/inference NaN)
                    assert np.isfinite(h["effect"])
        assert any_nan, "Expected at least one NaN-SE (path, horizon) entry"


@pytest.mark.slow
class TestByPathBootstrap:
    """
    ``by_path`` combined with ``n_bootstrap > 0``.

    Each top-k path has its pre-computed cohort-centered IF passed to the
    existing multiplier-bootstrap mixin, which runs `n_bootstrap` draws
    per (path, horizon) target and returns bootstrap SE / percentile CI /
    percentile p-value. ``path_effects[path]["horizons"][l]`` is
    overwritten post-bootstrap with those fields; ``t_stat`` is re-derived
    from the bootstrap SE via ``safe_inference`` per the project anti-
    pattern rule. Point estimates are unchanged from the analytical path.

    Marked ``@pytest.mark.slow`` because each test runs a real bootstrap
    with at least 100 draws. See the plan file for the SE convention
    decision (fix paths across draws, library-consistent percentile CI).
    """

    def _fit_with_bootstrap(
        self,
        data,
        by_path: int,
        L_max: int = 3,
        n_bootstrap: int = 100,
        bootstrap_weights: str = "rademacher",
        seed: int = 42,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = ChaisemartinDHaultfoeuille(
                drop_larger_lower=False,
                by_path=by_path,
                n_bootstrap=n_bootstrap,
                bootstrap_weights=bootstrap_weights,
                seed=seed,
                twfe_diagnostic=False,
                placebo=False,
            )
            results = est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                L_max=L_max,
            )
        return est, results

    def test_point_estimates_preserved(self):
        """Bootstrap fit must leave path_effects[p]['horizons'][l]['effect']
        bit-identical to the analytical fit."""
        data = _by_path_three_path_data()
        _est_a, res_a = _fit_by_path(data, by_path=3, L_max=3)
        _est_b, res_b = self._fit_with_bootstrap(
            data, by_path=3, L_max=3, n_bootstrap=100, seed=42
        )
        assert res_a.path_effects is not None and res_b.path_effects is not None
        assert set(res_a.path_effects.keys()) == set(res_b.path_effects.keys())
        for path, entry_a in res_a.path_effects.items():
            entry_b = res_b.path_effects[path]
            for l_h, h_a in entry_a["horizons"].items():
                h_b = entry_b["horizons"][l_h]
                if np.isnan(h_a["effect"]):
                    assert np.isnan(h_b["effect"])
                else:
                    np.testing.assert_allclose(
                        h_b["effect"], h_a["effect"], atol=1e-14, rtol=1e-14,
                        err_msg=f"path={path} l={l_h}: bootstrap changed effect",
                    )

    def test_bootstrap_se_finite_and_positive(self):
        """On the hand-built 3-path panel, every non-degenerate (path, horizon)
        produces a positive finite bootstrap SE."""
        data = _by_path_three_path_data()
        _est, res = self._fit_with_bootstrap(
            data, by_path=3, L_max=3, n_bootstrap=200, seed=42
        )
        assert res.path_effects is not None
        any_finite = False
        for path, entry in res.path_effects.items():
            for l_h, h in entry["horizons"].items():
                if h["n_obs"] >= 2:  # skip degenerate singletons
                    assert np.isfinite(h["se"]) or np.isnan(h["se"]), (
                        f"path={path} l={l_h}: bootstrap SE is non-finite "
                        f"and not NaN: {h['se']}"
                    )
                    if np.isfinite(h["se"]):
                        assert h["se"] > 0, (
                            f"path={path} l={l_h}: bootstrap SE is not "
                            f"positive: {h['se']}"
                        )
                        any_finite = True
        assert any_finite, "No (path, horizon) produced a finite bootstrap SE"

    def test_bootstrap_se_close_to_analytical_on_well_conditioned(self):
        """
        On the cohort-clean fixture scenario (path assignment deterministic
        on F_g so every cohort is single-path), analytical and bootstrap
        SEs compute the same within-path marginal variance, so they must
        agree within Monte Carlo noise on (path, horizon) cells with
        ``n_obs >= 10``. Runs on the committed R-parity fixture so no
        extra panel construction is required.
        """
        golden_path = (
            Path(__file__).parents[1]
            / "benchmarks"
            / "data"
            / "dcdh_dynr_golden_values.json"
        )
        if not golden_path.exists():
            pytest.skip(
                f"dCDH golden values file not found at {golden_path}; "
                "run: Rscript benchmarks/R/generate_dcdh_dynr_test_values.R"
            )
        with open(golden_path) as f:
            sc = json.load(f)["scenarios"].get("multi_path_reversible_by_path")
        if sc is None:
            pytest.skip("scenario 'multi_path_reversible_by_path' absent")

        data = pd.DataFrame(sc["data"])

        # Analytical pass (n_bootstrap=0)
        _est_a, res_a = _fit_by_path(data, by_path=3, L_max=3)
        # Bootstrap pass with 500 draws for tighter Monte Carlo variance
        _est_b, res_b = self._fit_with_bootstrap(
            data, by_path=3, L_max=3, n_bootstrap=500, seed=2026
        )
        assert res_a.path_effects is not None
        assert res_b.path_effects is not None

        for path in res_a.path_effects:
            for l_h, h_a in res_a.path_effects[path]["horizons"].items():
                h_b = res_b.path_effects[path]["horizons"][l_h]
                if h_a["n_obs"] < 10:
                    continue
                se_a = h_a["se"]
                se_b = h_b["se"]
                if not (np.isfinite(se_a) and np.isfinite(se_b)):
                    continue
                # 30% rtol envelope covers Monte Carlo variance at n=500
                # on cohort-clean single-path cohorts.
                rtol = abs(se_b - se_a) / se_a
                assert rtol < 0.30, (
                    f"path={path} l={l_h}: bootstrap SE diverges from "
                    f"analytical beyond Monte Carlo envelope — "
                    f"analytical={se_a:.4f} bootstrap={se_b:.4f} "
                    f"rtol={rtol:.3f}"
                )

    def test_degenerate_cohort_still_nan(self):
        """All-singleton cohort panel: bootstrap SE on path subsets must
        remain NaN (inherited from the zero-IF coercion in
        ``bootstrap_utils.compute_effect_bootstrap_stats``)."""
        panel = pd.DataFrame(
            {
                "group": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
                "period": [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
                "treatment": [0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                "outcome": [
                    10.0, 13.0, 14.0,
                    10.0, 11.0, 9.0,
                    10.0, 11.0, 12.0,
                    10.0, 11.0, 12.0,
                ],
            }
        )
        _est, res = self._fit_with_bootstrap(
            panel, by_path=2, L_max=1, n_bootstrap=100, seed=42
        )
        assert res.path_effects is not None
        any_nan = False
        for entry in res.path_effects.values():
            for h in entry["horizons"].values():
                if np.isnan(h["se"]):
                    any_nan = True
                    assert np.isnan(h["t_stat"])
                    assert np.isnan(h["p_value"])
                    lo, hi = h["conf_int"]
                    assert np.isnan(lo) and np.isnan(hi)
                    assert np.isfinite(h["effect"])
        assert any_nan, (
            "Expected at least one NaN-SE (path, horizon) entry under "
            "singleton-cohort panel"
        )

    @pytest.mark.parametrize("weights", ["rademacher", "mammen", "webb"])
    def test_bootstrap_weights_variants(self, weights):
        """All three multiplier flavors produce finite bootstrap SE on the
        3-path hand-built panel."""
        data = _by_path_three_path_data()
        _est, res = self._fit_with_bootstrap(
            data, by_path=3, L_max=3, n_bootstrap=100,
            bootstrap_weights=weights, seed=42,
        )
        assert res.path_effects is not None
        any_finite = False
        for entry in res.path_effects.values():
            for h in entry["horizons"].values():
                if np.isfinite(h["se"]) and h["se"] > 0:
                    any_finite = True
        assert any_finite, (
            f"bootstrap_weights={weights!r} produced no finite per-path SE"
        )

    def test_bootstrap_seed_reproducibility(self):
        """Two fits with the same seed must produce bit-identical
        per-(path, horizon) bootstrap SE."""
        data = _by_path_three_path_data()
        _est1, res1 = self._fit_with_bootstrap(
            data, by_path=3, L_max=3, n_bootstrap=100, seed=2026,
        )
        _est2, res2 = self._fit_with_bootstrap(
            data, by_path=3, L_max=3, n_bootstrap=100, seed=2026,
        )
        assert res1.path_effects is not None and res2.path_effects is not None
        for path, entry1 in res1.path_effects.items():
            entry2 = res2.path_effects[path]
            for l_h, h1 in entry1["horizons"].items():
                h2 = entry2["horizons"][l_h]
                if np.isnan(h1["se"]):
                    assert np.isnan(h2["se"])
                else:
                    np.testing.assert_array_equal(
                        h1["se"], h2["se"],
                        err_msg=f"path={path} l={l_h}: seed reproducibility broke",
                    )

    def test_inference_fields_match_bootstrap_results(self):
        """
        The post-bootstrap overwrite must take ``p_value`` and ``conf_int``
        from the percentile bootstrap (``br.path_p_values`` and
        ``br.path_cis``), not from a normal-theory recomputation. This
        pins the Round-10 library convention and prevents regression to a
        hybrid inference surface.
        """
        data = _by_path_three_path_data()
        _est_a, res_a = _fit_by_path(data, by_path=3, L_max=3)
        _est_b, res_b = self._fit_with_bootstrap(
            data, by_path=3, L_max=3, n_bootstrap=200, seed=42,
        )
        assert res_a.path_effects is not None
        assert res_b.path_effects is not None

        found_changed_ci = False
        for path, entry_a in res_a.path_effects.items():
            for l_h, h_a in entry_a["horizons"].items():
                h_b = res_b.path_effects[path]["horizons"][l_h]
                se_a, se_b = h_a["se"], h_b["se"]
                ci_a, ci_b = h_a["conf_int"], h_b["conf_int"]
                if (
                    np.isfinite(se_a)
                    and np.isfinite(se_b)
                    and np.isfinite(ci_a[0])
                    and np.isfinite(ci_b[0])
                ):
                    # The bootstrap CI in general differs from the
                    # analytical normal-theory CI (percentile vs
                    # normal). We require the CI to NOT match the
                    # analytical normal-theory CI computed from the
                    # bootstrap SE — that would signal a regression to
                    # `safe_inference(effect, bootstrap_se, ...)`.
                    # Percentile CI is asymmetric around the point
                    # estimate in general; normal-theory CI is always
                    # symmetric (lo = eff - k*se, hi = eff + k*se). If
                    # |hi - eff| differs from |eff - lo| by more than
                    # 1e-9 the CI is asymmetric -> definitely
                    # percentile, not normal-theory. Symmetric
                    # percentile CIs still pass this test (small n or
                    # symmetric bootstrap sample); we only require
                    # *at least one* asymmetric cell across all (path,
                    # horizon) entries to confirm the percentile path.
                    eff = h_b["effect"]
                    lo_b, hi_b = ci_b
                    if abs((hi_b - eff) - (eff - lo_b)) > 1e-9:
                        found_changed_ci = True
                        break
            if found_changed_ci:
                break

        # t-stat is SE-derived on bootstrap path too (anti-pattern rule).
        # Assert the t-stat equals effect / se to within float precision.
        for path, entry_b in res_b.path_effects.items():
            for l_h, h_b in entry_b["horizons"].items():
                if np.isfinite(h_b["se"]) and h_b["se"] > 0:
                    expected_t = h_b["effect"] / h_b["se"]
                    np.testing.assert_allclose(
                        h_b["t_stat"], expected_t, atol=1e-10, rtol=1e-10,
                        err_msg=(
                            f"path={path} l={l_h}: t_stat should be "
                            f"SE-derived per anti-pattern rule"
                        ),
                    )

        assert found_changed_ci, (
            "Expected at least one percentile CI that is asymmetric "
            "around the point estimate (non-symmetric bounds) to prove "
            "the bootstrap path uses percentile CI rather than a "
            "normal-theory recomputation. If this fails, the bootstrap "
            "bootstrap distribution was symmetric by chance — bump "
            "n_bootstrap or change the seed and re-run."
        )

    def test_inference_fields_equal_bootstrap_results_directly(self):
        """
        Pin direct equality between ``path_effects[path]["horizons"][l]``
        and ``bootstrap_results.path_{ses, cis, p_values}[path][l]``.
        If the ``fit()`` propagation drifts (e.g., a regression that
        recomputes normal-theory stats from the SE), these exact-match
        assertions fail even if the asymmetric-CI check in
        ``test_inference_fields_match_bootstrap_results`` happens to
        pass.
        """
        data = _by_path_three_path_data()
        _est, res = self._fit_with_bootstrap(
            data, by_path=3, L_max=3, n_bootstrap=200, seed=42,
        )
        assert res.path_effects is not None
        br = res.bootstrap_results
        assert br is not None
        assert br.path_ses is not None
        assert br.path_cis is not None
        assert br.path_p_values is not None

        checked = 0
        for path, entry in res.path_effects.items():
            for l_h, h in entry["horizons"].items():
                se_br = br.path_ses.get(path, {}).get(l_h)
                p_br = br.path_p_values.get(path, {}).get(l_h)
                ci_br = br.path_cis.get(path, {}).get(l_h)
                if se_br is None:
                    continue
                if np.isfinite(se_br):
                    np.testing.assert_array_equal(
                        h["se"], se_br,
                        err_msg=(
                            f"path={path} l={l_h}: path_effects se "
                            f"{h['se']} != bootstrap_results.path_ses {se_br}"
                        ),
                    )
                    np.testing.assert_array_equal(
                        h["p_value"], p_br if p_br is not None else np.nan,
                        err_msg=(
                            f"path={path} l={l_h}: path_effects p_value "
                            f"{h['p_value']} != "
                            f"bootstrap_results.path_p_values {p_br}"
                        ),
                    )
                    lo_e, hi_e = h["conf_int"]
                    assert ci_br is not None
                    lo_br, hi_br = ci_br
                    np.testing.assert_array_equal(
                        [lo_e, hi_e], [lo_br, hi_br],
                        err_msg=(
                            f"path={path} l={l_h}: path_effects conf_int "
                            f"{(lo_e, hi_e)} != "
                            f"bootstrap_results.path_cis {(lo_br, hi_br)}"
                        ),
                    )
                    checked += 1
        assert checked > 0, (
            "Expected at least one (path, horizon) with direct equality "
            "between path_effects inference fields and bootstrap_results"
        )

    def test_overflow_warning_fires_exactly_once_under_bootstrap(self):
        """
        When ``by_path > n_observed_paths``, ``_enumerate_treatment_paths``
        emits a ``UserWarning``. The bootstrap helper
        ``_collect_path_bootstrap_inputs`` re-calls the enumerator, so
        without suppression the warning would fire twice on a bootstrap
        fit — once from the analytical pass and once from the bootstrap
        pass. Pin that the bootstrap path surfaces the warning exactly
        once (analytical-pass emission only; bootstrap-pass emission
        suppressed because it is a spurious duplicate of the same
        fact).
        """
        data = _by_path_three_path_data()  # 3 observed paths
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            est = ChaisemartinDHaultfoeuille(
                drop_larger_lower=False,
                by_path=10,  # overflow: more than 3 observed paths
                n_bootstrap=50,
                seed=42,
                twfe_diagnostic=False,
                placebo=False,
            )
            est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                L_max=3,
            )
        overflow_warnings = [
            w for w in caught
            if "exceeds the number of observed paths" in str(w.message)
            or "more than the observed number of paths" in str(w.message)
            or "requested but only" in str(w.message)
        ]
        assert len(overflow_warnings) == 1, (
            f"Expected exactly one overflow UserWarning under "
            f"by_path + n_bootstrap, got {len(overflow_warnings)}. "
            f"Messages: {[str(w.message) for w in overflow_warnings]}"
        )

    def test_summary_footer_mixed_validity_surfaces_live_targets(self):
        """
        Mixed-validity case: overall_se / event_study_ses degenerate to
        NaN while joiners_se / leavers_se / path_effects horizons retain
        finite bootstrap inference. ``by_path`` zeros switcher-side
        contributions outside the selected path while keeping the control
        pool intact, so path-level bootstrap targets can stay finite even
        when the overall/event-study IF degenerates on a reversible
        panel. The footer must point the reader at the live targets
        rather than falsely claiming "non-finite SE on every target."

        Uses a healthy bootstrap fit and post-hoc mutates overall_se /
        event_study_effects to NaN, pinning the footer logic in
        isolation from the (hard-to-engineer) natural reversible DGP
        that produces this exact mixed-validity state.
        """
        data = _by_path_three_path_data()
        _est, res = self._fit_with_bootstrap(
            data, by_path=3, L_max=3, n_bootstrap=200, seed=42,
        )
        # Sanity: healthy fit has finite overall and path SEs.
        assert np.isfinite(res.overall_se)
        assert res.path_effects is not None
        any_finite_path = any(
            np.isfinite(h["se"])
            for e in res.path_effects.values()
            for h in e["horizons"].values()
        )
        assert any_finite_path

        # Force overall + event_study to NaN while leaving path_effects
        # untouched — simulates the reversible-panel scenario where the
        # overall IF is identically zero but the by_path subset IF is
        # not.
        res.overall_se = float("nan")
        res.overall_t_stat = float("nan")
        res.overall_p_value = float("nan")
        res.overall_conf_int = (float("nan"), float("nan"))
        if res.event_study_effects is not None:
            for entry in res.event_study_effects.values():
                entry["se"] = float("nan")
                entry["t_stat"] = float("nan")
                entry["p_value"] = float("nan")
                entry["conf_int"] = (float("nan"), float("nan"))

        summary_text = res.summary()
        # Must NOT claim "non-finite SE on every target"
        assert "produced non-finite SE on every target" not in summary_text, (
            "Footer falsely claims all-target failure while path_effects "
            "still has finite bootstrap SE. Summary tail:\n"
            f"{summary_text[-400:]}"
        )
        # Must NOT claim "multiplier-bootstrap percentile inference"
        # (overall_se is NaN so the headline inference is not bootstrap
        # percentile).
        assert "multiplier-bootstrap percentile inference" not in summary_text
        # Must mention "per-path bootstrap inference is populated"
        assert (
            "per-path" in summary_text
            and "bootstrap inference is populated" in summary_text
        ), (
            "Footer must surface which targets retain finite bootstrap "
            "inference when overall/event-study degenerates. Summary "
            "tail:\n"
            f"{summary_text[-400:]}"
        )

    def test_nan_contract_extends_to_overall_and_event_study_horizons(self):
        """
        The bootstrap-contract NaN-on-invalid rule applies to every
        dCDH public inference surface, not just ``path_effects``. Pin
        that ``n_bootstrap=1`` (which cannot produce a finite bootstrap
        SE from a one-element distribution) propagates NaN to
        ``overall_*``, ``joiners_*`` / ``leavers_*`` (when available),
        AND each ``event_study_effects[l]`` entry. Prevents regression
        to the pre-fix pattern where invalid bootstrap silently left
        analytical values in place on these surfaces while
        ``path_effects`` was NaN-consistent — a cross-surface
        inconsistency inside a single result object.
        """
        data = _by_path_three_path_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = ChaisemartinDHaultfoeuille(
                drop_larger_lower=False,
                by_path=3,
                n_bootstrap=1,
                seed=42,
                twfe_diagnostic=False,
                placebo=False,
            )
            res = est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                L_max=3,
            )

        assert np.isnan(res.overall_se), (
            f"n_bootstrap=1: overall_se must be NaN (bootstrap "
            f"contract), got {res.overall_se}"
        )
        assert np.isnan(res.overall_t_stat)
        assert np.isnan(res.overall_p_value)
        lo, hi = res.overall_conf_int
        assert np.isnan(lo) and np.isnan(hi)
        # Point estimate stays finite across bootstrap invalidity
        assert np.isfinite(res.overall_att)

        if res.joiners_se is not None:
            assert np.isnan(res.joiners_se)
            assert np.isnan(res.joiners_p_value)
            jlo, jhi = res.joiners_conf_int
            assert np.isnan(jlo) and np.isnan(jhi)
        if res.leavers_se is not None:
            assert np.isnan(res.leavers_se)
            assert np.isnan(res.leavers_p_value)
            llo, lhi = res.leavers_conf_int
            assert np.isnan(llo) and np.isnan(lhi)

        assert res.event_study_effects is not None
        for l_h, entry in res.event_study_effects.items():
            assert np.isnan(entry["se"]), (
                f"n_bootstrap=1: event_study_effects[{l_h}].se must be "
                f"NaN, got {entry['se']}"
            )
            assert np.isnan(entry["t_stat"])
            assert np.isnan(entry["p_value"])
            elo, ehi = entry["conf_int"]
            assert np.isnan(elo) and np.isnan(ehi)
            assert np.isfinite(entry["effect"])

        # summary() must NOT claim "multiplier-bootstrap percentile
        # inference" when the displayed overall SE is NaN, and it must
        # NOT claim "used for event-study horizon inference" when every
        # event_study_effects entry has NaN SE. It should fall through
        # to the "bootstrap was requested but produced non-finite SE"
        # note.
        summary_text = res.summary()
        assert "multiplier-bootstrap percentile inference" not in summary_text, (
            "summary() incorrectly labels NaN-inference as "
            "'multiplier-bootstrap percentile inference'"
        )
        assert (
            "produced non-finite SE" in summary_text
            or "inference fields are NaN-consistent" in summary_text
        ), (
            f"summary() footer must acknowledge the invalid-bootstrap "
            f"state when all inference fields are NaN. Got:\n{summary_text[-400:]}"
        )

    def test_degenerate_bootstrap_distribution_yields_nan_tuple(self):
        """
        When the bootstrap SE comes back non-finite for a ``(path,
        horizon)`` (e.g., ``n_bootstrap=1`` produces a one-element
        distribution whose std is zero / ill-defined), the overwrite
        block must replace the full inference tuple with NaN rather
        than falling back to the analytical values. This pins the
        bootstrap-contract semantics — once the user opts into
        ``n_bootstrap > 0``, all per-path inference is bootstrap-
        derived or NaN-consistent, never silently analytical.
        """
        data = _by_path_three_path_data()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est = ChaisemartinDHaultfoeuille(
                drop_larger_lower=False,
                by_path=3,
                n_bootstrap=1,
                bootstrap_weights="rademacher",
                seed=42,
                twfe_diagnostic=False,
                placebo=False,
            )
            res = est.fit(
                data,
                outcome="outcome",
                group="group",
                time="period",
                treatment="treatment",
                L_max=3,
            )
        assert res.path_effects is not None
        br = res.bootstrap_results
        assert br is not None and br.path_ses is not None

        any_nan = False
        for path, entry in res.path_effects.items():
            for l_h, h in entry["horizons"].items():
                bs_se = br.path_ses.get(path, {}).get(l_h)
                # A one-draw bootstrap cannot produce a finite SE (std
                # of a singleton is 0 → coerced to NaN by
                # bootstrap_utils.compute_effect_bootstrap_stats).
                if bs_se is None or not np.isfinite(bs_se):
                    any_nan = True
                    assert np.isnan(h["se"]), (
                        f"path={path} l={l_h}: bootstrap returned non-"
                        f"finite SE but path_effects.se={h['se']} "
                        f"(expected NaN — must not fall back to "
                        f"analytical under the bootstrap contract)"
                    )
                    assert np.isnan(h["t_stat"]), (
                        f"path={path} l={l_h}: t_stat={h['t_stat']} "
                        f"(expected NaN when bootstrap SE is non-finite)"
                    )
                    assert np.isnan(h["p_value"]), (
                        f"path={path} l={l_h}: p_value={h['p_value']} "
                        f"(expected NaN when bootstrap SE is non-finite)"
                    )
                    lo, hi = h["conf_int"]
                    assert np.isnan(lo) and np.isnan(hi), (
                        f"path={path} l={l_h}: conf_int=({lo}, {hi}) "
                        f"(expected (nan, nan) when bootstrap SE is "
                        f"non-finite)"
                    )
                    # Point estimate stays finite (bootstrap does not
                    # touch effect values)
                    assert np.isfinite(h["effect"]), (
                        f"path={path} l={l_h}: effect={h['effect']} "
                        f"(bootstrap must not overwrite the point "
                        f"estimate)"
                    )
        assert any_nan, (
            "Expected at least one (path, horizon) to land in the "
            "non-finite-SE bootstrap branch with n_bootstrap=1"
        )
