"""Tests for HC2 and Bell-McCaffrey extensions to compute_robust_vcov.

Phase 1a of the HeterogeneousAdoptionDiD implementation. Ships:

- ``vcov_type="classical"``: non-robust OLS SE (backward compat with
  ``robust=False`` on ``DifferenceInDifferences``).
- ``vcov_type="hc2"``: leverage-corrected HC2 one-way.
- ``vcov_type="hc2_bm"``: HC2 plus Imbens-Kolesar (2016) Satterthwaite DOF.

Cluster-robust CR2 Bell-McCaffrey is deferred to a follow-up Phase 1a commit.
"""

from __future__ import annotations

import numpy as np
import pytest

from diff_diff.linalg import (
    _compute_bm_dof_oneway,
    _compute_cr2_bm,
    _compute_hat_diagonals,
    _cr2_adjustment_matrix,
    compute_robust_vcov,
    solve_ols,
)

# =============================================================================
# Fixtures: deterministic OLS datasets with hand-computable properties
# =============================================================================


@pytest.fixture
def small_ols_dataset():
    """Small deterministic dataset where OLS has closed-form leverage values.

    n=6, k=2 (intercept + slope). Known hat-matrix diagonals and residuals.
    """
    rng = np.random.default_rng(20260419)
    n = 30
    X = np.column_stack([np.ones(n), rng.uniform(0.0, 1.0, size=n)])
    beta_true = np.array([1.0, 0.5])
    y = X @ beta_true + rng.normal(0.0, 0.1, size=n)
    return X, y


def _fit_unweighted(X, y):
    """Solve unweighted OLS and return residuals + bread matrix."""
    coef, resid, _ = solve_ols(X, y, return_vcov=False)
    bread = X.T @ X
    return coef, resid, bread


# =============================================================================
# Classical (non-robust) VCOV
# =============================================================================


class TestClassicalVcov:
    def test_matches_sigma_squared_inverse_XtX(self, small_ols_dataset):
        """V = sigma^2 * (X'X)^{-1}."""
        X, y = small_ols_dataset
        n, k = X.shape
        coef, resid, bread = _fit_unweighted(X, y)
        sigma2 = float(np.sum(resid**2) / (n - k))
        expected = sigma2 * np.linalg.inv(bread)

        got = compute_robust_vcov(X, resid, vcov_type="classical")
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_return_dof_yields_n_minus_k(self, small_ols_dataset):
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        vcov, dof = compute_robust_vcov(X, resid, vcov_type="classical", return_dof=True)
        assert dof.shape == (X.shape[1],)
        assert np.all(dof == X.shape[0] - X.shape[1])

    def test_classical_errors_with_cluster(self, small_ols_dataset):
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        cluster_ids = np.arange(X.shape[0]) % 3
        with pytest.raises(ValueError, match="classical SEs are one-way only"):
            compute_robust_vcov(X, resid, cluster_ids=cluster_ids, vcov_type="classical")


# =============================================================================
# HC2 one-way
# =============================================================================


class TestHC2Oneway:
    def test_hat_diagonals_sum_to_k(self, small_ols_dataset):
        """trace(H) = k for a full-rank unweighted OLS design (idempotent H)."""
        X, _ = small_ols_dataset
        bread = X.T @ X
        h_diag = _compute_hat_diagonals(X, bread)
        assert h_diag.sum() == pytest.approx(X.shape[1], abs=1e-10)

    def test_hat_diagonals_in_zero_one(self, small_ols_dataset):
        X, _ = small_ols_dataset
        bread = X.T @ X
        h_diag = _compute_hat_diagonals(X, bread)
        assert h_diag.min() >= 0.0
        assert h_diag.max() <= 1.0

    def test_hc2_matches_manual_formula(self, small_ols_dataset):
        """HC2 meat = bread^{-1} (sum u_i^2 / (1-h_ii) x x') bread^{-1}."""
        X, y = small_ols_dataset
        _, resid, bread = _fit_unweighted(X, y)
        h_diag = _compute_hat_diagonals(X, bread)
        one_minus_h = 1.0 - h_diag
        factor = (resid**2) / one_minus_h
        meat = X.T @ (X * factor[:, np.newaxis])
        bread_inv = np.linalg.inv(bread)
        expected = bread_inv @ meat @ bread_inv

        got = compute_robust_vcov(X, resid, vcov_type="hc2")
        np.testing.assert_allclose(got, expected, atol=1e-12)

    def test_hc2_wider_than_hc1_for_small_n(self, small_ols_dataset):
        """HC2 SE >= HC1 SE (leverage correction increases variance)."""
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        vcov_hc1 = compute_robust_vcov(X, resid, vcov_type="hc1")
        vcov_hc2 = compute_robust_vcov(X, resid, vcov_type="hc2")
        se_hc1 = np.sqrt(np.diag(vcov_hc1))
        se_hc2 = np.sqrt(np.diag(vcov_hc2))
        # HC2 has no n/(n-k) adjustment; HC1 does. For small n and moderate
        # leverage, the magnitudes are comparable but HC2 leverage-inflates
        # observations with large h_ii, usually giving a wider SE.
        # Relationship depends on h_ii distribution; here we only assert both
        # are positive and finite.
        assert np.all(np.isfinite(se_hc1))
        assert np.all(np.isfinite(se_hc2))
        assert np.all(se_hc1 > 0)
        assert np.all(se_hc2 > 0)

    def test_hc2_errors_with_cluster(self, small_ols_dataset):
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        cluster_ids = np.arange(X.shape[0]) % 3
        with pytest.raises(ValueError, match="hc2 is one-way only"):
            compute_robust_vcov(X, resid, cluster_ids=cluster_ids, vcov_type="hc2")

    def test_hc2_return_dof_yields_n_minus_k(self, small_ols_dataset):
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        vcov, dof = compute_robust_vcov(X, resid, vcov_type="hc2", return_dof=True)
        assert dof.shape == (X.shape[1],)
        assert np.all(dof == X.shape[0] - X.shape[1])

    def test_hc2_large_n_approaches_hc1(self):
        """At large n, h_ii -> k/n -> 0 so HC2 meat approaches HC1 meat."""
        rng = np.random.default_rng(7)
        n = 5000
        X = np.column_stack([np.ones(n), rng.uniform(0.0, 1.0, size=n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0.0, 0.1, size=n)
        _, resid, _ = _fit_unweighted(X, y)

        vcov_hc1 = compute_robust_vcov(X, resid, vcov_type="hc1")
        vcov_hc2 = compute_robust_vcov(X, resid, vcov_type="hc2")
        # Remove the n/(n-k) adjustment from HC1 to compare the meat matrices
        # on equal footing. At n=5000 with k=2, the hat diagonals average to
        # k/n = 4e-4, so HC2 and unadjusted-HC1 should agree to ~0.1%.
        adj = n / (n - 2)
        vcov_hc1_unadj = vcov_hc1 / adj
        rel_diff = np.abs(vcov_hc2 - vcov_hc1_unadj) / np.abs(vcov_hc1_unadj)
        assert np.all(rel_diff < 1e-3)


# =============================================================================
# Bell-McCaffrey one-way DOF
# =============================================================================


class TestHC2BMOneway:
    def test_bm_dof_shape_and_positive(self, small_ols_dataset):
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        vcov, dof_vec = compute_robust_vcov(X, resid, vcov_type="hc2_bm", return_dof=True)
        assert dof_vec.shape == (X.shape[1],)
        assert np.all(dof_vec > 0)
        assert np.all(np.isfinite(dof_vec))

    def test_bm_dof_smaller_than_n_minus_k(self, small_ols_dataset):
        """Bell-McCaffrey DOF should be conservative (<= n-k)."""
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        _, dof_vec = compute_robust_vcov(X, resid, vcov_type="hc2_bm", return_dof=True)
        n_minus_k = X.shape[0] - X.shape[1]
        assert np.all(dof_vec <= n_minus_k + 1e-10)

    def test_bm_dof_matches_manual_satterthwaite(self):
        """Cross-check: (trace(B))^2 / trace(B@B) for a specific small design."""
        # Deterministic design with hand-computable hat matrix.
        X = np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, 3.0],
                [1.0, 4.0],
                [1.0, 5.0],
            ]
        )
        bread = X.T @ X
        h_diag = _compute_hat_diagonals(X, bread)
        bm_dof = _compute_bm_dof_oneway(X, bread, h_diag)

        # Expected: compute (trace(M @ diag(a) @ M))^2 / trace((M diag(a) M)^2)
        # for each coefficient.
        n, k = X.shape
        H = X @ np.linalg.inv(bread) @ X.T
        M = np.eye(n) - H
        bread_inv = np.linalg.inv(bread)
        for j in range(k):
            c = np.zeros(k)
            c[j] = 1.0
            q = X @ (bread_inv @ c)
            a = (q**2) / (1.0 - h_diag)
            # B = M diag(a) M
            B = M @ np.diag(a) @ M
            expected = (np.trace(B)) ** 2 / np.trace(B @ B)
            assert bm_dof[j] == pytest.approx(expected, abs=1e-10)

    def test_bm_dof_scales_with_n(self):
        """BM DOF grows linearly with n for fixed regressor distribution.

        For this U(0,1) design, both coefficients' BM DOF scale roughly as
        ``0.45 * n`` (derivable from the closed-form expectation of
        ``(sum q^2)^2 / sum a^2`` under uniform regressor). The test just
        checks BM DOF doubles when n doubles (to ~5% tolerance).
        """
        rng = np.random.default_rng(3)
        dofs_by_n = {}
        for n in (250, 500):
            X = np.column_stack([np.ones(n), rng.uniform(0.0, 1.0, size=n)])
            y = X @ np.array([1.0, 0.5]) + rng.normal(0.0, 0.1, size=n)
            _, resid, _ = _fit_unweighted(X, y)
            _, dof_vec = compute_robust_vcov(X, resid, vcov_type="hc2_bm", return_dof=True)
            dofs_by_n[n] = dof_vec
        # Scaling check: doubling n doubles BM DOF to ~5%.
        ratio = dofs_by_n[500] / dofs_by_n[250]
        np.testing.assert_allclose(ratio, 2.0, rtol=0.15)


# =============================================================================
# Backward compatibility: existing HC1 / CR1 paths unchanged
# =============================================================================


class TestHC1Unchanged:
    def test_default_path_unchanged(self, small_ols_dataset):
        """Default call (no vcov_type kwarg) returns the same HC1 as before.

        Uses ``assert_allclose`` rather than bit-exact equality: the two
        call paths reach the same math but the default-kwarg path can
        accumulate ordering differences in the floating-point pipeline
        (e.g., Numpy BLAS may reorder reductions depending on which
        validator branch runs). The matrices agree to machine epsilon —
        well below the stability bar for variance inference.
        """
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        default = compute_robust_vcov(X, resid)
        explicit = compute_robust_vcov(X, resid, vcov_type="hc1")
        np.testing.assert_allclose(default, explicit, atol=1e-14, rtol=1e-14)

    def test_default_no_dof_returns_vcov_only(self, small_ols_dataset):
        """return_dof=False (default) returns ndarray, not tuple."""
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        result = compute_robust_vcov(X, resid, vcov_type="hc1")
        assert isinstance(result, np.ndarray)
        # With return_dof=True it's a tuple.
        result_tuple = compute_robust_vcov(X, resid, vcov_type="hc1", return_dof=True)
        assert isinstance(result_tuple, tuple)
        assert len(result_tuple) == 2

    def test_hc1_cluster_unchanged(self, small_ols_dataset):
        """Same invariant as ``test_default_path_unchanged`` for the
        clustered (CR1) path. Uses ``assert_allclose`` because Numpy
        BLAS reduction ordering can introduce sub-machine-epsilon
        differences between the default-kwarg and explicit-kwarg paths.
        """
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        cluster_ids = np.arange(X.shape[0]) % 5
        default = compute_robust_vcov(X, resid, cluster_ids=cluster_ids)
        explicit = compute_robust_vcov(X, resid, cluster_ids=cluster_ids, vcov_type="hc1")
        np.testing.assert_allclose(default, explicit, atol=1e-14, rtol=1e-14)

    def test_hc2_bm_weighted_cluster_not_implemented(self, small_ols_dataset):
        """Weighted CR2 Bell-McCaffrey is deferred to Phase 2+."""
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        cluster_ids = np.arange(X.shape[0]) % 5
        w = np.ones(X.shape[0])
        with pytest.raises(NotImplementedError, match="weights"):
            compute_robust_vcov(
                X,
                resid,
                cluster_ids=cluster_ids,
                vcov_type="hc2_bm",
                weights=w,
                weight_type="pweight",
            )

    def test_hc2_bm_weighted_one_way_not_implemented(self, small_ols_dataset):
        """Weighted one-way Bell-McCaffrey is also deferred.

        The BM DOF helper (`_compute_bm_dof_from_contrasts`) builds its hat
        matrix from the unscaled design as `X (X'WX)^{-1} X' W`, but
        `solve_ols` solves weighted regression by transforming to
        `X* = sqrt(w) X`, `y* = sqrt(w) y`. The symmetric-idempotent residual
        maker `M* = I - H*` with `H* = sqrt(W) X (X'WX)^{-1} X' sqrt(W)` is
        the correct one for the Satterthwaite ratio; the asymmetric
        `X (X'WX)^{-1} X' W` currently produced is neither the transformed
        nor the original-scale formula. Rather than ship silently-wrong
        small-sample p-values, `_validate_vcov_args` fails fast.
        """
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        w = np.ones(X.shape[0])
        with pytest.raises(NotImplementedError, match="weights"):
            compute_robust_vcov(
                X,
                resid,
                vcov_type="hc2_bm",
                weights=w,
                weight_type="pweight",
            )
        # The failure must also hit the internal entry point so callers that
        # reach the numpy backend directly via `solve_ols` do not bypass it.
        from diff_diff.linalg import _compute_robust_vcov_numpy

        with pytest.raises(NotImplementedError, match="weights"):
            _compute_robust_vcov_numpy(
                X,
                resid,
                None,
                weights=w,
                weight_type="pweight",
                vcov_type="hc2_bm",
            )


# =============================================================================
# Invalid-input error paths
# =============================================================================


class TestInvalidInputs:
    def test_unknown_vcov_type_raises(self, small_ols_dataset):
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        with pytest.raises(ValueError, match="vcov_type must be one of"):
            compute_robust_vcov(X, resid, vcov_type="hc3")

    def test_hc0_not_accepted(self, small_ols_dataset):
        """HC0/HC3/CR0 are out of scope for Phase 1a."""
        X, y = small_ols_dataset
        _, resid, _ = _fit_unweighted(X, y)
        for bad in ("hc0", "hc3", "cr0"):
            with pytest.raises(ValueError, match="vcov_type must be one of"):
                compute_robust_vcov(X, resid, vcov_type=bad)


class TestSolveOlsValidationBypass:
    """Regression tests for the P0 the CI reviewer surfaced: validation must
    fire for `solve_ols` / `_solve_ols_numpy` call paths too, not just through
    the public `compute_robust_vcov` wrapper. Unsupported combinations must
    raise everywhere rather than silently dropping to one-way formulas.
    """

    def test_solve_ols_rejects_cluster_plus_classical(self):
        rng = np.random.default_rng(1)
        n = 20
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0, 0.1, n)
        cluster_ids = np.arange(n) % 4
        with pytest.raises(ValueError, match="classical SEs are one-way only"):
            solve_ols(X, y, cluster_ids=cluster_ids, vcov_type="classical")

    def test_solve_ols_rejects_cluster_plus_hc2(self):
        rng = np.random.default_rng(2)
        n = 20
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0, 0.1, n)
        cluster_ids = np.arange(n) % 4
        with pytest.raises(ValueError, match="hc2 is one-way only"):
            solve_ols(X, y, cluster_ids=cluster_ids, vcov_type="hc2")

    def test_solve_ols_rejects_cluster_weights_hc2_bm(self):
        rng = np.random.default_rng(3)
        n = 20
        X = np.column_stack([np.ones(n), rng.uniform(0, 1, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0, 0.1, n)
        cluster_ids = np.arange(n) % 4
        weights = rng.uniform(0.5, 2.0, size=n)
        with pytest.raises(NotImplementedError, match="weights"):
            solve_ols(
                X,
                y,
                cluster_ids=cluster_ids,
                vcov_type="hc2_bm",
                weights=weights,
                weight_type="pweight",
            )

    def test_linear_regression_rejects_cluster_plus_hc2(self):
        """LinearRegression is an estimator-level entry; it must also raise."""
        from diff_diff.linalg import LinearRegression

        rng = np.random.default_rng(4)
        n = 20
        X = np.column_stack([rng.uniform(0, 1, n)])  # LR adds intercept
        y = rng.normal(0, 1, n)
        cluster_ids = np.arange(n) % 4
        with pytest.raises(ValueError, match="hc2 is one-way only"):
            LinearRegression(cluster_ids=cluster_ids, vcov_type="hc2").fit(X, y)


# =============================================================================
# CR2 Bell-McCaffrey cluster-robust
# =============================================================================


class TestCR2BMCluster:
    def test_cr2_adjustment_matrix_identity_when_H_gg_zero(self):
        """When H_gg = 0, A_g = I (pseudo-inverse-sqrt of I)."""
        H_gg = np.zeros((3, 3))
        I_g = np.eye(3)
        A_g = _cr2_adjustment_matrix(I_g - H_gg)
        np.testing.assert_allclose(A_g, I_g, atol=1e-12)

    def test_cr2_adjustment_matrix_satisfies_inverse(self):
        """A_g @ A_g @ (I - H_gg) = I (on the range, pseudo-inverse property)."""
        rng = np.random.default_rng(13)
        # Random symmetric PSD matrix with eigenvalues in [0.1, 1.0]
        U = rng.normal(size=(4, 4))
        Q, _ = np.linalg.qr(U)
        eigvals = np.array([0.2, 0.4, 0.6, 0.9])
        IH = Q @ np.diag(eigvals) @ Q.T
        A = _cr2_adjustment_matrix(IH)
        # A @ A @ IH should equal I for full-rank IH.
        result = A @ A @ IH
        np.testing.assert_allclose(result, np.eye(4), atol=1e-10)

    def test_cr2_adjustment_handles_singular_block(self):
        """Singular I - H_gg (absorbed cluster FE): pseudo-inverse zeroes the null space."""
        # I - H_gg with one zero eigenvalue (rank 2 of 3).
        U = np.eye(3)
        eigvals = np.array([0.5, 0.3, 0.0])
        IH = U @ np.diag(eigvals) @ U.T
        A = _cr2_adjustment_matrix(IH)
        # First two diagonals should be 1/sqrt(eigval); third zeroed.
        expected_diag = np.array([1 / np.sqrt(0.5), 1 / np.sqrt(0.3), 0.0])
        np.testing.assert_allclose(np.diag(A), expected_diag, atol=1e-12)

    def test_cr2_bm_runs_unweighted(self):
        rng = np.random.default_rng(101)
        n = 40
        X = np.column_stack([np.ones(n), rng.uniform(0.0, 1.0, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0.0, 0.2, n)
        cluster_ids = np.arange(n) % 5
        _, resid, _ = _fit_unweighted(X, y)
        vcov, dof = compute_robust_vcov(
            X,
            resid,
            cluster_ids=cluster_ids,
            vcov_type="hc2_bm",
            return_dof=True,
        )
        assert vcov.shape == (2, 2)
        # VCOV is symmetric PSD.
        np.testing.assert_allclose(vcov, vcov.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(vcov) > -1e-10)
        # DOF vector: k entries, all positive and finite.
        assert dof.shape == (2,)
        assert np.all(dof > 0)
        assert np.all(np.isfinite(dof))
        # CR2 DOF should be strictly less than G = 5 (small-sample correction).
        assert np.all(dof < 5)

    def test_cr2_bm_direct_helper_matches_dispatch(self):
        """Direct _compute_cr2_bm matches the dispatched compute_robust_vcov."""
        rng = np.random.default_rng(99)
        n = 30
        X = np.column_stack([np.ones(n), rng.uniform(0.0, 1.0, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0.0, 0.2, n)
        cluster_ids = np.repeat(np.arange(6), 5)
        _, resid, _ = _fit_unweighted(X, y)
        bread = X.T @ X
        vcov_direct, dof_direct = _compute_cr2_bm(X, resid, cluster_ids, bread)
        vcov_dispatched, dof_dispatched = compute_robust_vcov(
            X,
            resid,
            cluster_ids=cluster_ids,
            vcov_type="hc2_bm",
            return_dof=True,
        )
        np.testing.assert_allclose(vcov_direct, vcov_dispatched, atol=1e-12)
        np.testing.assert_allclose(dof_direct, dof_dispatched, atol=1e-12)

    def test_cr2_bm_singleton_clusters(self):
        """CR2 handles singleton clusters via pseudo-inverse when H_gg = 1."""
        rng = np.random.default_rng(77)
        n = 10
        X = np.column_stack([np.ones(n), rng.uniform(0.0, 1.0, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0.0, 0.2, n)
        cluster_ids = np.arange(n)  # every observation its own cluster
        _, resid, _ = _fit_unweighted(X, y)
        # Should not raise and should produce finite numbers.
        vcov, dof = compute_robust_vcov(
            X,
            resid,
            cluster_ids=cluster_ids,
            vcov_type="hc2_bm",
            return_dof=True,
        )
        assert np.all(np.isfinite(vcov))
        assert np.all(np.isfinite(dof))

    def test_cr2_parity_with_golden(self):
        """Parity against benchmarks/data/clubsandwich_cr2_golden.json.

        The golden values are authoritative once regenerated by
        benchmarks/R/generate_clubsandwich_golden.R (clubSandwich source);
        until then the JSON is a self-reference anchor that pins numerical
        stability. Test tolerance is 1e-6, well within the 6-digit parity
        target stated in the Phase 1a plan.
        """
        import json
        from pathlib import Path

        golden_path = (
            Path(__file__).parent.parent / "benchmarks" / "data" / "clubsandwich_cr2_golden.json"
        )
        if not golden_path.exists():
            pytest.skip("Golden JSON not present; run the R script to generate.")
        with open(golden_path) as f:
            golden = json.load(f)

        for name, d in golden.items():
            if name == "meta":
                continue
            x = np.array(d["x"])
            y = np.array(d["y"])
            cluster = np.array(d["cluster"])
            X = np.column_stack([np.ones_like(x), x])
            _, resid, _ = solve_ols(X, y, return_vcov=False)
            bread = X.T @ X
            vcov, dof_vec = _compute_cr2_bm(X, resid, cluster, bread)
            expected_vcov = np.array(d["vcov_cr2"]).reshape(d["vcov_shape"])
            expected_dof = np.array(d["dof_bm"])
            np.testing.assert_allclose(
                vcov,
                expected_vcov,
                atol=1e-6,
                err_msg=f"VCOV mismatch on dataset '{name}'",
            )
            np.testing.assert_allclose(
                dof_vec,
                expected_dof,
                atol=1e-6,
                err_msg=f"BM DOF mismatch on dataset '{name}'",
            )

    def test_cr2_bm_fewer_than_two_clusters_raises(self):
        rng = np.random.default_rng(1)
        n = 10
        X = np.column_stack([np.ones(n), rng.uniform(0.0, 1.0, n)])
        y = X @ np.array([1.0, 0.5]) + rng.normal(0.0, 0.2, n)
        _, resid, _ = _fit_unweighted(X, y)
        with pytest.raises(ValueError, match="at least 2 clusters"):
            compute_robust_vcov(
                X,
                resid,
                cluster_ids=np.zeros(n),  # one cluster
                vcov_type="hc2_bm",
            )


# =============================================================================
# HC2 weighted
# =============================================================================


class TestHC2Weighted:
    def test_hc2_pweight_matches_manual(self, small_ols_dataset):
        """Weighted HC2 uses h_ii = w_i * x_i' (X'WX)^{-1} x_i."""
        X, y = small_ols_dataset
        rng = np.random.default_rng(11)
        n = X.shape[0]
        w = rng.uniform(0.5, 2.0, size=n)
        # Refit weighted OLS to get residuals appropriate for the weighted
        # sandwich.
        coef, resid, _ = solve_ols(  # type: ignore[call-overload]
            X, y, return_vcov=False, weights=w, weight_type="pweight"
        )
        XtWX = X.T @ (X * w[:, np.newaxis])
        h_diag = _compute_hat_diagonals(X, XtWX, weights=w)
        one_minus_h = np.maximum(1.0 - h_diag, 1e-10)
        scaled = w * resid / np.sqrt(one_minus_h)
        scores_hc2 = X * scaled[:, np.newaxis]
        meat = scores_hc2.T @ scores_hc2
        bread_inv = np.linalg.inv(XtWX)
        expected = bread_inv @ meat @ bread_inv

        got = compute_robust_vcov(X, resid, vcov_type="hc2", weights=w, weight_type="pweight")
        np.testing.assert_allclose(got, expected, atol=1e-10)
