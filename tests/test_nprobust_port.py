"""Unit tests for the private port in ``diff_diff._nprobust_port``.

These tests exercise the internal helpers directly (``kernel_W``,
``qrXXinv``, ``lprobust_bw``, etc.) so that when an end-to-end parity
test in ``tests/test_bandwidth_selector.py`` drifts, the root cause is
localised to a single helper.
"""

from __future__ import annotations

import numpy as np
import pytest

from diff_diff._nprobust_port import (
    NPROBUST_SHA,
    NPROBUST_VERSION,
    kernel_W,
    lprobust_bw,
    qrXXinv,
)

# =============================================================================
# Metadata
# =============================================================================


def test_pinned_nprobust_version():
    assert NPROBUST_VERSION == "0.5.0"
    assert NPROBUST_SHA == "36e4e532d2f7d23d4dc6e162575cca79e0927cda"


# =============================================================================
# kernel_W (W.fun port)
# =============================================================================


class TestKernelW:
    def test_epa_symmetric(self):
        u = np.array([-0.7, -0.3, 0.0, 0.3, 0.7])
        w = kernel_W(u, "epa")
        # Symmetric around 0.
        np.testing.assert_allclose(w, w[::-1], atol=1e-14, rtol=1e-14)
        # k(0) = 0.75.
        assert w[2] == pytest.approx(0.75)
        # Zero at |u| = 1.
        assert kernel_W(np.array([1.0, -1.0]), "epa")[0] == pytest.approx(0.0)
        # Zero outside.
        assert kernel_W(np.array([1.5, -1.5]), "epa")[0] == 0.0

    def test_epa_nonneg_matches_one_sided_phase1a(self):
        """For u >= 0, nprobust symmetric epa = Phase 1a one-sided epa."""
        from diff_diff.local_linear import epanechnikov_kernel

        u = np.linspace(0.0, 1.0, 20)
        assert np.allclose(kernel_W(u, "epa"), epanechnikov_kernel(u))

    def test_uniform(self):
        u = np.array([-0.9, 0.0, 0.5, 1.0, 1.1])
        w = kernel_W(u, "uni")
        assert w[0] == 0.5
        assert w[1] == 0.5
        assert w[2] == 0.5
        assert w[3] == 0.5  # |u|=1 is inside
        assert w[4] == 0.0  # |u|>1 is outside

    def test_triangular(self):
        u = np.array([-0.3, 0.0, 0.7, 1.0, 1.5])
        w = kernel_W(u, "tri")
        assert w[0] == pytest.approx(0.7)
        assert w[1] == pytest.approx(1.0)
        assert w[2] == pytest.approx(0.3)
        assert w[3] == pytest.approx(0.0)
        assert w[4] == 0.0

    def test_gaussian(self):
        from scipy.stats import norm

        u = np.array([-1.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(kernel_W(u, "gau"), norm.pdf(u), atol=1e-14)

    def test_unknown_kernel_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel"):
            kernel_W(np.array([0.5]), "cosine")


# =============================================================================
# qrXXinv (chol2inv(chol(crossprod(.))) port)
# =============================================================================


class TestQrXXinv:
    def test_matches_numpy_inverse(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 5))
        expected = np.linalg.inv(X.T @ X)
        actual = qrXXinv(X)
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)

    def test_symmetric_output(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(30, 3))
        A = qrXXinv(X)
        np.testing.assert_allclose(A, A.T, atol=1e-14, rtol=1e-14)

    def test_identity_for_orthonormal_input(self):
        # If X has orthonormal columns, X.T @ X = I, so inverse = I.
        X = np.eye(4)[:, :3]
        np.testing.assert_allclose(qrXXinv(X), np.eye(3), atol=1e-14, rtol=1e-14)


# =============================================================================
# lprobust_bw: single-stage parity against the golden JSON
# =============================================================================


def _precompute_for_test(x):
    """Helper that sorts and builds dups/dupsid."""
    from diff_diff._nprobust_port import _precompute_nn_duplicates

    order = np.argsort(x)
    x_sorted = x[order]
    dups, dupsid = _precompute_nn_duplicates(x_sorted)
    return order, x_sorted, dups, dupsid


class TestLprobustBwStageD1:
    """First Stage-2 call (C.d1 with o=q+1=3, nu=q+1=3, o_B=q+2=4).

    Uses the DGP 1 golden values as reference. The c_bw input must
    match R's clipped value, which is the raw rule-of-thumb bandwidth
    when bwcheck does not bind (true for DGP 1 at G=2000).
    """

    def _setup(self):
        import json
        from pathlib import Path

        golden_path = (
            Path(__file__).resolve().parents[1]
            / "benchmarks"
            / "data"
            / "nprobust_mse_dpi_golden.json"
        )
        if not golden_path.exists():
            pytest.skip(
                "Golden values file not found; "
                "run: Rscript benchmarks/R/generate_nprobust_golden.R"
            )
        with golden_path.open() as f:
            golden = json.load(f)
        dgp = golden["dgp1"]
        d = np.array(dgp["d"], dtype=np.float64)
        y = np.array(dgp["y"], dtype=np.float64)
        _, d_sorted, dups, dupsid = _precompute_for_test(d)
        order = np.argsort(d)
        y_sorted = y[order]
        c_bw = float(dgp["c_bw"])
        range_ = float(d_sorted.max() - d_sorted.min())
        return d_sorted, y_sorted, dups, dupsid, c_bw, range_, dgp["stage_d1"]

    def test_V_matches(self):
        d, y, dups, dupsid, c_bw, range_, stage = self._setup()
        C_d1 = lprobust_bw(
            y,
            d,
            None,
            0.0,
            o=3,
            nu=3,
            o_B=4,
            h_V=c_bw,
            h_B1=range_,
            h_B2=range_,
            scale=0.0,
            vce="nn",
            nnmatch=3,
            kernel="epa",
            dups=dups,
            dupsid=dupsid,
        )
        assert C_d1.V == pytest.approx(stage["V"], rel=0.01)

    def test_B1_matches(self):
        d, y, dups, dupsid, c_bw, range_, stage = self._setup()
        C_d1 = lprobust_bw(
            y,
            d,
            None,
            0.0,
            o=3,
            nu=3,
            o_B=4,
            h_V=c_bw,
            h_B1=range_,
            h_B2=range_,
            scale=0.0,
            vce="nn",
            nnmatch=3,
            kernel="epa",
            dups=dups,
            dupsid=dupsid,
        )
        assert C_d1.B1 == pytest.approx(stage["B1"], rel=0.01)

    def test_B2_matches(self):
        d, y, dups, dupsid, c_bw, range_, stage = self._setup()
        C_d1 = lprobust_bw(
            y,
            d,
            None,
            0.0,
            o=3,
            nu=3,
            o_B=4,
            h_V=c_bw,
            h_B1=range_,
            h_B2=range_,
            scale=0.0,
            vce="nn",
            nnmatch=3,
            kernel="epa",
            dups=dups,
            dupsid=dupsid,
        )
        assert C_d1.B2 == pytest.approx(stage["B2"], rel=0.01)

    def test_R_is_zero_when_scale_zero(self):
        d, y, dups, dupsid, c_bw, range_, _stage = self._setup()
        C_d1 = lprobust_bw(
            y,
            d,
            None,
            0.0,
            o=3,
            nu=3,
            o_B=4,
            h_V=c_bw,
            h_B1=range_,
            h_B2=range_,
            scale=0.0,
            vce="nn",
            nnmatch=3,
            kernel="epa",
            dups=dups,
            dupsid=dupsid,
        )
        # With scale=0, BWreg is never computed -> R stays 0.
        assert C_d1.R == 0.0


# =============================================================================
# lpbwselect_mse_dpi: input validation on the advanced-use entry point
# =============================================================================


class TestLpbwselectMseDpiValidation:
    """The public wrapper is restricted to the HAD surface; the port is
    the advertised advanced-use entry point. It must enforce its own
    shape / emptiness / finiteness contract -- silently truncating a
    longer y or cluster through sort-index reindexing would be a real
    bug."""

    def test_mismatched_shapes_raise(self):
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        x = np.array([0.1, 0.2, 0.3])
        y = np.array([1.0, 2.0, 3.0, 4.0])  # length 4 != 3
        with pytest.raises(ValueError, match="same 1-D shape"):
            lpbwselect_mse_dpi(y, x, eval_point=0.0)

    def test_longer_y_silent_truncation_rejected(self):
        """Regression: len(y) > len(x) previously got truncated via
        the sort-indexer under vce='nn'. Must now raise."""
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1.0, size=100)
        y = rng.normal(size=200)  # twice the length
        with pytest.raises(ValueError, match="same 1-D shape"):
            lpbwselect_mse_dpi(y, x, eval_point=0.0, vce="nn")

    def test_cluster_wrong_length_rejected(self):
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1.0, size=100)
        y = rng.normal(size=100)
        cluster = np.arange(50)  # wrong length
        with pytest.raises(ValueError, match="cluster must have"):
            lpbwselect_mse_dpi(y, x, cluster=cluster, eval_point=0.0)

    def test_empty_direct_port_input_rejected(self):
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        x = np.array([], dtype=np.float64)
        y = np.array([], dtype=np.float64)
        with pytest.raises(ValueError, match="non-empty"):
            lpbwselect_mse_dpi(y, x, eval_point=0.0)

    def test_non_finite_x_rejected(self):
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        x = np.array([0.1, np.nan, 0.3])
        y = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="x contains non-finite"):
            lpbwselect_mse_dpi(y, x, eval_point=0.0)

    def test_non_finite_y_rejected(self):
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        x = np.array([0.1, 0.2, 0.3])
        y = np.array([1.0, np.inf, 3.0])
        with pytest.raises(ValueError, match="y contains non-finite"):
            lpbwselect_mse_dpi(y, x, eval_point=0.0)

    def test_non_finite_eval_point_rejected(self):
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1.0, size=100)
        y = rng.normal(size=100)
        with pytest.raises(ValueError, match="eval_point"):
            lpbwselect_mse_dpi(y, x, eval_point=np.nan)

    def test_missing_cluster_id_rejected_nan(self):
        """Missing (NaN) cluster IDs must be rejected, not silently
        dropped from block grouping (cluster == c is False for NaN).
        Deviates from nprobust's complete-case filter -- documented
        in the module docstring."""
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1.0, size=100)
        y = rng.normal(size=100)
        cluster = np.arange(100, dtype=np.float64)
        cluster[7] = np.nan
        cluster[42] = np.nan
        with pytest.raises(ValueError, match="missing values"):
            lpbwselect_mse_dpi(y, x, cluster=cluster, eval_point=0.0)

    def test_missing_cluster_id_rejected_none(self):
        """``None`` sentinel in an object-dtype cluster array must
        also be rejected."""
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1.0, size=10)
        y = rng.normal(size=10)
        cluster = np.array(
            [1, 2, None, 3, 4, 5, 6, 7, 8, 9], dtype=object
        )
        with pytest.raises(ValueError, match="missing values"):
            lpbwselect_mse_dpi(y, x, cluster=cluster, eval_point=0.0)


# =============================================================================
# lprobust_vce clustered branch: port fidelity vs R source
# =============================================================================


class TestLprobustVceClustered:
    """Pin the clustered lprobust_vce result against the R source
    formula. R's lprobust.vce (npfunctions.R:165-185) returns M (the
    sum over clusters of outer products of sum-of-scores), NOT a
    finite-sample-scaled M. An earlier port mistakenly returned
    ``w * M``; this test is the regression anchor."""

    def test_clustered_meat_matches_unscaled_sum(self):
        from diff_diff._nprobust_port import lprobust_vce

        rng = np.random.default_rng(123)
        n, k = 40, 3
        RX = rng.normal(size=(n, k))
        res = rng.normal(size=(n, 1))
        cluster = np.repeat(np.arange(10), 4)

        # Manual computation: M = sum_g (X_g' r_g) outer (X_g' r_g),
        # with NO ((n-1)/(n-k))*(g/(g-1)) scaling.
        expected = np.zeros((k, k), dtype=np.float64)
        for c in np.unique(cluster):
            ind = cluster == c
            v = RX[ind].T @ res[ind].ravel()
            expected += np.outer(v, v)

        actual = lprobust_vce(RX, res, cluster)
        np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)

    def test_clustered_is_symmetric(self):
        """The meat is a sum of outer products of rank-1 vectors and
        must be symmetric."""
        from diff_diff._nprobust_port import lprobust_vce

        rng = np.random.default_rng(7)
        n, k = 30, 4
        RX = rng.normal(size=(n, k))
        res = rng.normal(size=(n, 1))
        cluster = np.array([0] * 10 + [1] * 10 + [2] * 10)
        M = lprobust_vce(RX, res, cluster)
        np.testing.assert_allclose(M, M.T, atol=1e-14, rtol=1e-14)

    def test_clustered_end_to_end_through_lprobust_bw(self):
        """Smoke test: a clustered call through lpbwselect_mse_dpi
        completes and returns finite stage bandwidths. Pins that the
        clustered DPI path exercises lprobust_vce -> lprobust_bw ->
        driver without rank / shape / indexing crashes."""
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        rng = np.random.default_rng(20260419)
        G = 500
        d = rng.uniform(0.0, 1.0, size=G)
        y = d + d**2 + rng.normal(0, 0.3, size=G)
        # 50 clusters, balanced.
        cluster = np.repeat(np.arange(50), G // 50)
        res = lpbwselect_mse_dpi(
            y, d, cluster=cluster, eval_point=0.0,
            p=1, deriv=0, kernel="epa", vce="nn",
        )
        assert np.isfinite(res.h_mse_dpi)
        assert res.h_mse_dpi > 0.0
        assert np.isfinite(res.c_bw)
        assert res.c_bw > 0.0
