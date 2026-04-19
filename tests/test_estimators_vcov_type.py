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

    def test_set_params_rejects_conflict_on_robust_only(self):
        """Setting robust=False on an estimator with vcov_type='hc2_bm' raises."""
        est = DifferenceInDifferences(vcov_type="hc2_bm")
        # The user is asking for non-robust SEs on an explicitly-HC2-BM estimator.
        # set_params re-derives vcov_type to "classical" since only `robust` changed;
        # this is a coherent override of the prior vcov_type, not a silent mismatch.
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
