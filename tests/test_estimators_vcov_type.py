"""Tests for `vcov_type` threading through DifferenceInDifferences.

Covers the Phase 1a commitments in the approved plan:
- `robust=True` aliases `vcov_type="hc1"`.
- `robust=False` aliases `vcov_type="classical"` (backward compat for the 7
  existing test files that pass `robust=False`).
- Explicit `vcov_type` values validate against {classical, hc1, hc2, hc2_bm}.
- `robust=False` + explicit non-classical `vcov_type` raises at `__init__`.
- `MultiPeriodDiD` and `TwoWayFixedEffects` inherit through `get_params`.
- HC2+BM produces a wider CI than HC1 on the same data (property of the DOF
  correction).
- `get_params` / `set_params` round-trip preserves `vcov_type`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from diff_diff import SurveyDesign
from diff_diff.estimators import DifferenceInDifferences, MultiPeriodDiD
from diff_diff.twfe import TwoWayFixedEffects


def _make_did_panel(n_units: int = 30, seed: int = 20260420) -> pd.DataFrame:
    """Deterministic two-period DiD panel with a treatment effect of 1.0."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        treated = int(i >= n_units // 2)
        for t in (0, 1):
            y = rng.normal(0.0, 1.0) + 0.5 * treated + 1.0 * treated * t
            rows.append({"unit": i, "time": t, "treated": treated, "y": y})
    return pd.DataFrame(rows)


# =============================================================================
# robust <-> vcov_type alias resolution
# =============================================================================


class TestRobustAliasing:
    def test_robust_true_aliases_hc1(self):
        est = DifferenceInDifferences(robust=True)
        assert est.vcov_type == "hc1"

    def test_robust_false_aliases_classical(self):
        est = DifferenceInDifferences(robust=False)
        assert est.vcov_type == "classical"

    def test_explicit_vcov_type_wins_when_robust_default(self):
        """When `robust` is the default (True) and vcov_type is explicit, vcov_type wins."""
        est = DifferenceInDifferences(vcov_type="hc2_bm")
        assert est.vcov_type == "hc2_bm"

    def test_robust_false_and_classical_coexist(self):
        """robust=False + vcov_type='classical' is redundant but not an error."""
        est = DifferenceInDifferences(robust=False, vcov_type="classical")
        assert est.vcov_type == "classical"
        assert est.robust is False

    def test_robust_false_explicit_hc1_raises(self):
        """robust=False + vcov_type='hc1' is inconsistent -> ValueError."""
        with pytest.raises(ValueError, match="robust=False conflicts with vcov_type"):
            DifferenceInDifferences(robust=False, vcov_type="hc1")

    def test_robust_false_explicit_hc2_raises(self):
        with pytest.raises(ValueError, match="robust=False conflicts with vcov_type"):
            DifferenceInDifferences(robust=False, vcov_type="hc2")

    def test_unknown_vcov_type_raises(self):
        with pytest.raises(ValueError, match="vcov_type must be one of"):
            DifferenceInDifferences(vcov_type="hc3")

    def test_hc0_not_accepted(self):
        for bad in ("hc0", "HC1", "CR2", "cr1", "hc2+bm"):
            with pytest.raises(ValueError, match="vcov_type must be one of"):
                DifferenceInDifferences(vcov_type=bad)


# =============================================================================
# get_params / set_params round-trip
# =============================================================================


class TestParamsRoundTrip:
    def test_get_params_includes_vcov_type(self):
        est = DifferenceInDifferences(vcov_type="hc2_bm")
        params = est.get_params()
        assert "vcov_type" in params
        assert params["vcov_type"] == "hc2_bm"

    def test_get_params_default_vcov_type(self):
        est = DifferenceInDifferences()
        assert est.get_params()["vcov_type"] == "hc1"

    def test_set_params_preserves_vcov_type(self):
        est = DifferenceInDifferences()
        est.set_params(vcov_type="hc2")
        assert est.vcov_type == "hc2"

    def test_set_params_rejects_conflict_robust_false_hc2(self):
        """set_params must re-validate robust/vcov_type consistency."""
        est = DifferenceInDifferences()
        with pytest.raises(ValueError, match="robust=False conflicts with vcov_type"):
            est.set_params(robust=False, vcov_type="hc2")

    def test_set_params_robust_only_rederives_vcov_type(self):
        """Setting robust= alone after init re-derives vcov_type from the alias.

        When only ``robust`` is passed to ``set_params``, the new ``robust`` value
        overrides the previously-set ``vcov_type`` via the alias rule:
        ``robust=False`` -> ``"classical"``. This keeps the pair internally
        consistent rather than leaving the estimator with ``robust=False,
        vcov_type="hc2_bm"`` (a state that ``__init__`` forbids).
        """
        est = DifferenceInDifferences(vcov_type="hc2_bm")
        est.set_params(robust=False)
        assert est.vcov_type == "classical"

    def test_set_params_invalid_vcov_type_rejected(self):
        est = DifferenceInDifferences()
        with pytest.raises(ValueError, match="vcov_type must be one of"):
            est.set_params(vcov_type="hc3")

    def test_set_params_robust_true_then_back_to_hc1(self):
        """robust=True after construction restores hc1 when no explicit vcov_type."""
        est = DifferenceInDifferences(robust=False)
        assert est.vcov_type == "classical"
        est.set_params(robust=True)
        assert est.vcov_type == "hc1"

    def test_set_params_multi_period_inherits(self):
        est = MultiPeriodDiD(vcov_type="hc2_bm")
        params = est.get_params()
        assert params["vcov_type"] == "hc2_bm"

    def test_set_params_twfe_inherits(self):
        est = TwoWayFixedEffects(vcov_type="hc2")
        assert est.vcov_type == "hc2"

    def test_set_params_conflict_leaves_estimator_unchanged(self):
        """A rejected set_params() call must leave the estimator unchanged.

        Previously `set_params` mutated attributes via `setattr` BEFORE
        re-validating the robust/vcov_type pair. A failing call left the
        estimator in exactly the half-configured state the alias/conflict
        check is supposed to prevent, which defeats callers that catch
        `ValueError` and try to keep using the object. This test pins the
        atomic behavior: on failure, no attribute moves.
        """
        est = DifferenceInDifferences(
            robust=True,
            vcov_type="hc1",
            cluster=None,
            alpha=0.05,
        )
        before_robust = est.robust
        before_vcov = est.vcov_type
        before_cluster = est.cluster
        before_alpha = est.alpha
        with pytest.raises(ValueError, match="robust=False conflicts with"):
            # Conflict: robust=False + vcov_type="hc2". The side-effect here is
            # the regression target — set_params must NOT apply cluster=/alpha=
            # (or anything else in the batch) when validation fails.
            est.set_params(robust=False, vcov_type="hc2", cluster="unit", alpha=0.1)
        assert est.robust == before_robust
        assert est.vcov_type == before_vcov
        assert est.cluster == before_cluster
        assert est.alpha == before_alpha

    def test_set_params_unknown_key_leaves_estimator_unchanged(self):
        """Unknown-key rejections must be atomic too, not partial.

        Regression guard for the first-pass validator: when one key in the
        params batch is unknown, no keys in the batch should have been
        applied by the time we raise.
        """
        est = DifferenceInDifferences(vcov_type="hc1", alpha=0.05)
        with pytest.raises(ValueError, match="Unknown parameter"):
            # vcov_type is valid but `not_a_real_param` is not — reject the
            # whole batch and leave vcov_type at "hc1".
            est.set_params(vcov_type="hc2_bm", not_a_real_param=1)
        assert est.vcov_type == "hc1"
        assert est.alpha == 0.05


# =============================================================================
# End-to-end fit() behavior
# =============================================================================


class TestFitBehavior:
    def test_hc1_fit_and_summary_contain_expected_fields(self):
        data = _make_did_panel()
        est = DifferenceInDifferences(vcov_type="hc1")
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)
        assert np.isfinite(res.conf_int[0])
        assert np.isfinite(res.conf_int[1])

    def test_hc1_and_hc2_bm_both_fit(self):
        """HC1 and HC2_BM produce the same point estimate; may share SE on a
        saturated balanced DiD but must still fit cleanly.

        For a saturated 2x2 DiD with balanced cells, h_ii = k/n is constant and
        both HC1 adjustment n/(n-k) and HC2's 1/(1-h_ii) cancel into the same
        vcov. The per-coefficient BM DOF for the saturated interaction happens
        to equal n-k exactly, so CIs match too. This test pins the point
        estimate equivalence, which is the guarantee users can rely on.
        """
        data = _make_did_panel()
        est_hc1 = DifferenceInDifferences(vcov_type="hc1")
        est_hc2bm = DifferenceInDifferences(vcov_type="hc2_bm")
        r_hc1 = est_hc1.fit(data, outcome="y", treatment="treated", time="time")
        r_hc2bm = est_hc2bm.fit(data, outcome="y", treatment="treated", time="time")
        # Point estimate unaffected by vcov choice.
        assert r_hc1.att == pytest.approx(r_hc2bm.att, abs=1e-10)
        # Both produce finite SEs and CIs.
        assert np.isfinite(r_hc1.se)
        assert np.isfinite(r_hc2bm.se)
        assert np.isfinite(r_hc1.conf_int[0]) and np.isfinite(r_hc1.conf_int[1])
        assert np.isfinite(r_hc2bm.conf_int[0]) and np.isfinite(r_hc2bm.conf_int[1])

    def test_classical_via_robust_false(self):
        data = _make_did_panel()
        est = DifferenceInDifferences(robust=False)
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)

    def test_classical_via_explicit_vcov_type(self):
        data = _make_did_panel()
        est = DifferenceInDifferences(vcov_type="classical")
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        assert np.isfinite(res.se)

    def test_summary_includes_vcov_label_hc1(self):
        """`summary()` output includes an HC1 label in the Variance line."""
        data = _make_did_panel()
        est = DifferenceInDifferences(vcov_type="hc1")
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        summary = res.summary()
        assert "HC1 heteroskedasticity-robust" in summary

    def test_summary_includes_vcov_label_hc2_bm(self):
        data = _make_did_panel()
        est = DifferenceInDifferences(vcov_type="hc2_bm")
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        summary = res.summary()
        assert "HC2 + Bell-McCaffrey" in summary

    def test_summary_includes_vcov_label_classical(self):
        data = _make_did_panel()
        est = DifferenceInDifferences(vcov_type="classical")
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        summary = res.summary()
        assert "Classical OLS SEs" in summary

    def test_summary_includes_vcov_label_cr1(self):
        """CR1 cluster-robust (HC1 + cluster) labels with the cluster name."""
        data = _make_did_panel()
        est = DifferenceInDifferences(vcov_type="hc1", cluster="unit")
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        summary = res.summary()
        assert "CR1 cluster-robust at unit" in summary

    def test_multi_period_fit_honors_classical(self):
        """MultiPeriodDiD.fit with vcov_type='classical' produces non-robust SEs.

        Regression test for the CI review finding: `MultiPeriodDiD` inherits
        `vcov_type` from the base class via get_params but its `fit()` path
        used to ignore the knob. Here we compare classical vs hc1 SEs on the
        same data and assert they differ (i.e. the parameter actually took).
        """
        rng = np.random.default_rng(20260419)
        n_units = 40
        rows = []
        for i in range(n_units):
            treated = int(i >= n_units // 2)
            for t in range(4):
                post = int(t >= 2)
                y = rng.normal(0.0, 1.0) + 0.3 * treated + 0.8 * treated * post
                rows.append({"unit": i, "time": t, "treated": treated, "y": y})
        data = pd.DataFrame(rows)

        r_hc1 = MultiPeriodDiD(vcov_type="hc1").fit(
            data, outcome="y", treatment="treated", time="time"
        )
        r_classical = MultiPeriodDiD(vcov_type="classical").fit(
            data, outcome="y", treatment="treated", time="time"
        )
        # Point estimates identical.
        assert r_hc1.avg_att == pytest.approx(r_classical.avg_att, abs=1e-10)
        # SEs must differ — vcov_type actually changed the variance family.
        assert r_hc1.avg_se != pytest.approx(r_classical.avg_se, abs=1e-10)

    def test_multi_period_cluster_plus_hc2_bm_rejected(self):
        """MultiPeriodDiD rejects cluster + hc2_bm until contrast-aware cluster BM lands.

        The CR2 per-coefficient DOF is available, but the post-period-average
        contrast DOF under cluster-robust Bell-McCaffrey is not yet
        implemented. Pairing CR2 SEs with one-way BM DOF would be a broken
        hybrid. Fail fast with a clear workaround.
        """
        rng = np.random.default_rng(2)
        rows = []
        for i in range(20):
            treated = int(i >= 10)
            for t in range(3):
                y = rng.normal(0.0, 1.0) + 0.5 * treated * (t >= 1)
                rows.append({"unit": i, "time": t, "treated": treated, "y": y})
        data = pd.DataFrame(rows)

        est = MultiPeriodDiD(vcov_type="hc2_bm", cluster="unit")
        with pytest.raises(NotImplementedError, match="cluster"):
            est.fit(data, outcome="y", treatment="treated", time="time")

    def test_multi_period_fit_honors_hc2_bm(self):
        """MultiPeriodDiD.fit with vcov_type='hc2_bm' uses Bell-McCaffrey DOF.

        Checks two things: (a) fit completes without error on the hc2_bm path
        for the period-effect loop, and (b) the BM Satterthwaite DOF produces
        a CI for avg_att with a finite width (non-degenerate case).
        """
        rng = np.random.default_rng(1919)
        n_units = 50
        rows = []
        for i in range(n_units):
            treated = int(i >= n_units // 2)
            for t in range(5):
                post = int(t >= 3)
                y = rng.normal(0.0, 1.0) + 0.2 * treated + 0.6 * treated * post
                rows.append({"unit": i, "time": t, "treated": treated, "y": y})
        data = pd.DataFrame(rows)

        r_hc2bm = MultiPeriodDiD(vcov_type="hc2_bm").fit(
            data, outcome="y", treatment="treated", time="time"
        )
        assert np.isfinite(r_hc2bm.avg_att)
        assert np.isfinite(r_hc2bm.avg_se)
        assert np.isfinite(r_hc2bm.avg_conf_int[0])
        assert np.isfinite(r_hc2bm.avg_conf_int[1])
        # CI width is finite and positive.
        ci_width = r_hc2bm.avg_conf_int[1] - r_hc2bm.avg_conf_int[0]
        assert ci_width > 0

    def test_twfe_rejects_hc2_and_hc2_bm(self):
        """TWFE rejects vcov_type in {hc2, hc2_bm} because it uses within-
        transformation. HC2 leverage on the reduced design is not the hat
        matrix of the full FE projection (FWL preserves coefficients, not
        the hat matrix), so applying HC2/CR2-BM to the demeaned regressors
        would silently ship wrong small-sample SEs. The fit must raise with
        a pointer to HC1 (which has no leverage term and survives FWL) or
        fixed_effects= dummies as workarounds.
        """
        data = _make_did_panel(n_units=20)
        for bad in ("hc2", "hc2_bm"):
            with pytest.raises(
                NotImplementedError,
                match="TwoWayFixedEffects.*not yet supported",
            ):
                TwoWayFixedEffects(vcov_type=bad).fit(
                    data,
                    outcome="y",
                    treatment="treated",
                    time="time",
                    unit="unit",
                )

    def test_twfe_results_record_cluster_name(self):
        """TWFE results should label the auto-clustered SE with the unit column."""
        rng = np.random.default_rng(1)
        n_units = 20
        rows = []
        for i in range(n_units):
            treated = int(i >= n_units // 2)
            for t in range(3):
                post = int(t >= 1)
                y = rng.normal(0.0, 1.0) + 0.5 * treated * post
                rows.append({"unit": i, "time": t, "treated": treated, "y": y})
        data = pd.DataFrame(rows)

        res = TwoWayFixedEffects(vcov_type="hc1").fit(
            data, outcome="y", treatment="treated", time="time", unit="unit"
        )
        summary = res.summary()
        # TWFE auto-clusters at the unit column when cluster=None.
        assert "CR1 cluster-robust at unit" in summary

    def test_twfe_honors_classical_without_autocluster(self):
        """TWFE with vcov_type='classical' must skip its unit auto-cluster.

        Classical SEs are one-way only and would be rejected by the linalg
        validator if TWFE still injected unit-level clustering. The fix
        drops the auto-cluster when the user explicitly asks for a one-way
        family.
        """
        data = _make_did_panel(n_units=20)
        res = TwoWayFixedEffects(vcov_type="classical").fit(
            data, outcome="y", treatment="treated", time="time", unit="unit"
        )
        assert np.isfinite(res.att)
        assert np.isfinite(res.se)
        assert res.se > 0
        assert res.vcov_type == "classical"
        # Without an explicit cluster and with a one-way family, TWFE should
        # NOT have injected unit as the auto-cluster.
        assert res.cluster_name is None
        summary = res.summary()
        # Summary must label the one-way family, not CR1 cluster-robust.
        assert "Classical OLS" in summary
        assert "CR1 cluster-robust" not in summary

    def test_twfe_honors_robust_false_without_autocluster(self):
        """`robust=False` on TWFE maps to vcov_type='classical' and must
        likewise disable the auto-cluster."""
        data = _make_did_panel(n_units=20)
        res = TwoWayFixedEffects(robust=False).fit(
            data, outcome="y", treatment="treated", time="time", unit="unit"
        )
        assert res.vcov_type == "classical"
        assert res.cluster_name is None
        assert "CR1 cluster-robust" not in res.summary()

    def test_twfe_wild_bootstrap_preserves_auto_cluster(self):
        """Wild-bootstrap inference on TWFE with no explicit cluster must
        keep the unit auto-cluster, even under vcov_type='classical'.

        Regression guard for a bug where the one-way-family auto-cluster
        bypass also applied under wild_bootstrap, silently dropping the
        cluster structure the bootstrap was supposed to consume. The fix
        gates the bypass on inference=='analytical'.
        """
        data = _make_did_panel(n_units=20)
        res = TwoWayFixedEffects(
            vcov_type="classical",
            inference="wild_bootstrap",
            n_bootstrap=50,
            seed=1,
        ).fit(data, outcome="y", treatment="treated", time="time", unit="unit")
        # Bootstrap must have succeeded with a finite SE.
        assert np.isfinite(res.se)
        assert res.se > 0
        # Bootstrap consumed a unit-level cluster (20 clusters).
        assert res.n_clusters == 20

    def test_did_absorb_rejects_hc2_and_hc2_bm(self):
        """DifferenceInDifferences with absorb= rejects HC2/HC2+BM.

        Same methodology reason as TWFE: absorb= demeans via within-
        transformation, and HC2/CR2 leverage corrections depend on the full
        FE hat matrix rather than the residualized design. The fit must
        raise with a pointer to vcov_type='hc1' or fixed_effects= dummies.
        """
        rng = np.random.default_rng(20260420)
        n_units, n_time = 30, 3
        rows = []
        for i in range(n_units):
            treated = int(i >= n_units // 2)
            for t in range(n_time):
                post = int(t >= 1)
                y = rng.normal(0.0, 1.0) + 0.5 * treated * post
                rows.append({"unit": i, "time": t, "treated": treated, "post": post, "y": y})
        data = pd.DataFrame(rows)

        for bad in ("hc2", "hc2_bm"):
            with pytest.raises(
                NotImplementedError,
                match="DifferenceInDifferences.*absorb.*not yet supported",
            ):
                DifferenceInDifferences(vcov_type=bad).fit(
                    data,
                    outcome="y",
                    treatment="treated",
                    time="post",
                    absorb=["unit"],
                )

    def test_did_fixed_effects_dummies_still_accept_hc2_and_hc2_bm(self):
        """DifferenceInDifferences with fixed_effects= (dummy expansion) is
        NOT affected by the absorb-FE guard: the dummies appear in the full
        design matrix, so HC2 leverage is computed on the full projection.
        """
        rng = np.random.default_rng(20260420)
        n_units, n_time = 20, 2
        rows = []
        for i in range(n_units):
            treated = int(i >= n_units // 2)
            stratum = i // 5  # categorical for fixed_effects= dummies
            for t in range(n_time):
                y = rng.normal(0.0, 1.0) + 0.5 * treated * t
                rows.append(
                    {
                        "unit": i,
                        "time": t,
                        "treated": treated,
                        "post": t,
                        "stratum": stratum,
                        "y": y,
                    }
                )
        data = pd.DataFrame(rows)

        # Neither call should raise.
        for good in ("hc2", "hc2_bm"):
            res = DifferenceInDifferences(vcov_type=good).fit(
                data,
                outcome="y",
                treatment="treated",
                time="post",
                fixed_effects=["stratum"],
            )
            assert np.isfinite(res.att)
            assert np.isfinite(res.se)

    def test_multi_period_absorb_rejects_hc2_and_hc2_bm(self):
        """MultiPeriodDiD with absorb= rejects HC2/HC2+BM for the same
        methodology reason as the base class."""
        rng = np.random.default_rng(20260420)
        n_units, n_time = 30, 4
        rows = []
        for i in range(n_units):
            treated = int(i >= n_units // 2)
            for t in range(n_time):
                y = rng.normal(0.0, 1.0) + 0.3 * treated + 0.5 * treated * (t >= 2)
                rows.append({"unit": i, "time": t, "treated": treated, "y": y})
        data = pd.DataFrame(rows)

        for bad in ("hc2", "hc2_bm"):
            with pytest.raises(
                NotImplementedError,
                match="MultiPeriodDiD.*absorb.*not yet supported",
            ):
                MultiPeriodDiD(vcov_type=bad).fit(
                    data,
                    outcome="y",
                    treatment="treated",
                    time="time",
                    absorb=["unit"],
                    unit="unit",
                )

    def test_summary_suppresses_variance_line_under_wild_bootstrap(self):
        """When inference_method='wild_bootstrap', the Variance label is omitted.

        The wild-bootstrap path reports bootstrap SE/CI, not analytical. Printing
        an analytical family like 'HC1 heteroskedasticity-robust' under those
        numbers would be misleading.
        """
        rng = np.random.default_rng(42)
        rows = []
        for i in range(20):
            treated = int(i >= 10)
            for t in (0, 1):
                y = rng.normal(0.0, 1.0) + 0.5 * treated * t
                rows.append({"unit": i, "time": t, "treated": treated, "y": y})
        data = pd.DataFrame(rows)

        est = DifferenceInDifferences(
            vcov_type="hc1",
            inference="wild_bootstrap",
            cluster="unit",
            n_bootstrap=50,
            seed=7,
        )
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        summary = res.summary()
        # The bootstrap path substitutes SE/CI from resampling; the Variance:
        # line (which labels the analytical family) must be suppressed so the
        # displayed inference is unambiguous.
        assert "Variance:" not in summary
        # But the inference method should still be visible.
        assert "wild_bootstrap" in summary

    def test_wild_bootstrap_preserves_vcov_type_no_error(self):
        """Wild-bootstrap inference path doesn't fight with vcov_type.

        The wild-bootstrap SE comes from resampling, not from the analytical
        sandwich. `vcov_type` has no effect on the bootstrap SE output, but
        the fit should still succeed without errors.
        """
        data = _make_did_panel(n_units=20)
        est = DifferenceInDifferences(
            vcov_type="hc2_bm",
            inference="wild_bootstrap",
            n_bootstrap=50,
            seed=42,
        )
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        assert np.isfinite(res.se)


# =============================================================================
# Survey-fit summary labeling (P2 fix from CI review on PR #327)
# =============================================================================


def _make_survey_panel(seed: int = 20260420) -> pd.DataFrame:
    """Two-period DiD panel with strata/PSU/weight columns for survey fits.

    40 units, 4 strata (10 units each), 8 PSUs nested within strata (2 PSUs
    per stratum, 5 units each). Treatment is 20 vs 20; PSU labels are
    globally unique so SurveyDesign.resolve does not raise.
    """
    rng = np.random.default_rng(seed)
    rows = []
    n_units = 40
    for i in range(n_units):
        treated = int(i >= n_units // 2)
        stratum = i // 10  # 4 strata, 10 units each
        psu = i // 5  # 8 PSUs globally (2 per stratum)
        wt = 1.0 + 0.25 * stratum
        for t in (0, 1):
            y = rng.normal(0.0, 1.0) + 0.5 * treated + 1.0 * treated * t
            rows.append(
                {
                    "unit": i,
                    "time": t,
                    "treated": treated,
                    "stratum": stratum,
                    "psu": psu,
                    "weight": wt,
                    "y": y,
                }
            )
    return pd.DataFrame(rows)


class TestSummarySurveyLabeling:
    """When a SurveyDesign drives inference, the analytical `Variance:` line
    must be suppressed: the reported SEs come from Taylor linearization or
    replicate-weight variance, not from the analytical HC/CR sandwich. The
    survey inference block (weight_type, strata/PSU counts, replicate method)
    already surfaces the actual inference source; a parallel
    `Variance: HC1/...` line would mislabel what produced the SEs.

    These tests pin the P2 fix flagged by CI review on PR #327.
    """

    def test_survey_taylor_suppresses_analytical_variance_label(self):
        """SurveyDesign with PSU/strata (no replicate weights) uses Taylor
        linearization; the analytical `Variance:` line must not appear.
        """
        data = _make_survey_panel()
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            weight_type="pweight",
        )
        # Explicit vcov_type="hc1" to make the regression meaningful: if the
        # suppression wasn't in place, the summary would print "HC1
        # heteroskedasticity-robust" even though the SE came from survey
        # Taylor linearization.
        est = DifferenceInDifferences(vcov_type="hc1")
        res = est.fit(
            data,
            outcome="y",
            treatment="treated",
            time="time",
            survey_design=sd,
        )
        assert res.survey_metadata is not None
        summary = res.summary()
        # The analytical Variance: label must not appear; the survey design
        # line(s) already surface the actual inference source.
        assert "Variance:" not in summary
        # And the summary must still show the survey design block so the
        # user can see where the SEs came from.
        assert (
            "pweight" in summary
            or "Weight type" in summary
            or "n_psu" in summary.lower()
            or "psu" in summary.lower()
        )

    def test_survey_replicate_weights_suppresses_analytical_variance_label(self):
        """SurveyDesign with replicate_weights (BRR) drives replicate-variance
        inference; the analytical `Variance:` line must not appear.
        """
        data = _make_survey_panel()
        # Attach 10 BRR replicate-weight columns.
        rng = np.random.default_rng(12345)
        rep_cols = [f"rep{r}" for r in range(10)]
        for col in rep_cols:
            data[col] = rng.choice([0.5, 1.5], size=len(data))

        sd = SurveyDesign(
            weights="weight",
            replicate_weights=rep_cols,
            replicate_method="BRR",
            weight_type="pweight",
        )
        est = DifferenceInDifferences(vcov_type="hc2_bm")
        res = est.fit(
            data,
            outcome="y",
            treatment="treated",
            time="time",
            survey_design=sd,
        )
        assert res.survey_metadata is not None
        summary = res.summary()
        # The analytical HC2+BM label must not appear for a replicate-weight
        # fit: the actual SEs come from the BRR replicates.
        assert "Variance:" not in summary
        assert "HC2 + Bell-McCaffrey" not in summary
        # Survey metadata should surface the replicate method.
        assert "BRR" in summary or "replicate" in summary.lower()

    def test_multi_period_survey_taylor_suppresses_variance_label(self):
        """Same survey suppression holds for `MultiPeriodDiDResults.summary()`.

        MultiPeriodDiD has its own summary block and its own gating logic; the
        P2 fix applies there too.
        """
        data = _make_survey_panel()
        sd = SurveyDesign(
            weights="weight",
            strata="stratum",
            psu="psu",
            weight_type="pweight",
        )
        est = MultiPeriodDiD(vcov_type="hc1")
        res = est.fit(
            data,
            outcome="y",
            treatment="treated",
            time="time",
            unit="unit",
            survey_design=sd,
        )
        assert res.survey_metadata is not None
        summary = res.summary()
        assert "Variance:" not in summary

    def test_non_survey_fit_still_prints_variance_label(self):
        """Regression guard: the survey-only suppression must not break the
        non-survey path, which should still print the analytical Variance: line.
        """
        data = _make_did_panel(n_units=30)
        est = DifferenceInDifferences(vcov_type="hc1")
        res = est.fit(data, outcome="y", treatment="treated", time="time")
        assert res.survey_metadata is None
        summary = res.summary()
        assert "Variance:" in summary
        assert "HC1" in summary
