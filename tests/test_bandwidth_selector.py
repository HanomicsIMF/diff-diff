"""End-to-end tests for ``diff_diff.local_linear.mse_optimal_bandwidth``.

Parity against nprobust 0.5.0 (SHA 36e4e53) is the primary test. Golden
values live at ``benchmarks/data/nprobust_mse_dpi_golden.json``; see
``benchmarks/R/generate_nprobust_golden.R`` for the generator.

Behavioural tests cover input validation, kernel dispatch, and
``return_diagnostics`` shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from diff_diff import BandwidthResult, local_linear_fit, mse_optimal_bandwidth

# =============================================================================
# Golden-value parity: per-stage at 1% relative tolerance (DGP 1 / 2 / 3)
# =============================================================================


GOLDEN_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "nprobust_mse_dpi_golden.json"
)

_PARITY_TOL = 0.01  # 1% relative error; commit criterion #4


@pytest.fixture(scope="module")
def golden():
    with GOLDEN_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="module", params=["dgp1", "dgp2", "dgp3"])
def dgp_case(request, golden):
    name = request.param
    d = np.array(golden[name]["d"], dtype=np.float64)
    y = np.array(golden[name]["y"], dtype=np.float64)
    return name, d, y, golden[name]


class TestNprobustParity:
    """1% per-stage parity against R nprobust::lpbwselect(bwselect="mse-dpi")."""

    def test_c_bw_parity(self, dgp_case):
        name, d, y, g = dgp_case
        br = mse_optimal_bandwidth(d, y, return_diagnostics=True)
        assert br.c_bw == pytest.approx(g["c_bw"], rel=_PARITY_TOL), name

    def test_bw_mp2_parity(self, dgp_case):
        name, d, y, g = dgp_case
        br = mse_optimal_bandwidth(d, y, return_diagnostics=True)
        assert br.bw_mp2 == pytest.approx(g["bw_mp2"], rel=_PARITY_TOL), name

    def test_bw_mp3_parity(self, dgp_case):
        name, d, y, g = dgp_case
        br = mse_optimal_bandwidth(d, y, return_diagnostics=True)
        assert br.bw_mp3 == pytest.approx(g["bw_mp3"], rel=_PARITY_TOL), name

    def test_b_mse_parity(self, dgp_case):
        name, d, y, g = dgp_case
        br = mse_optimal_bandwidth(d, y, return_diagnostics=True)
        assert br.b_mse == pytest.approx(g["b_mse_dpi"], rel=_PARITY_TOL), name

    def test_h_mse_parity(self, dgp_case):
        name, d, y, g = dgp_case
        br = mse_optimal_bandwidth(d, y, return_diagnostics=True)
        assert br.h_mse == pytest.approx(g["h_mse_dpi"], rel=_PARITY_TOL), name


class TestStageDiagnosticsParity:
    """Per-stage (V, B1, B2, R) parity at 1% each. Catches formula
    divergences that might cancel out in the final bandwidth."""

    @pytest.mark.parametrize(
        "stage",
        ["stage_d1", "stage_d2", "stage_b", "stage_h"],
    )
    def test_V_parity(self, dgp_case, stage):
        name, d, y, g = dgp_case
        br = mse_optimal_bandwidth(d, y, return_diagnostics=True)
        actual = getattr(br, f"{stage}_V")
        expected = g[stage]["V"]
        assert actual == pytest.approx(expected, rel=_PARITY_TOL), f"{name} {stage}"

    @pytest.mark.parametrize(
        "stage",
        ["stage_d1", "stage_d2", "stage_b", "stage_h"],
    )
    def test_B1_parity(self, dgp_case, stage):
        name, d, y, g = dgp_case
        br = mse_optimal_bandwidth(d, y, return_diagnostics=True)
        actual = getattr(br, f"{stage}_B1")
        expected = g[stage]["B1"]
        if expected == 0:
            assert actual == pytest.approx(0, abs=1e-10), f"{name} {stage}"
        else:
            assert actual == pytest.approx(expected, rel=_PARITY_TOL), f"{name} {stage}"

    @pytest.mark.parametrize(
        "stage",
        ["stage_d1", "stage_d2", "stage_b", "stage_h"],
    )
    def test_B2_parity(self, dgp_case, stage):
        name, d, y, g = dgp_case
        br = mse_optimal_bandwidth(d, y, return_diagnostics=True)
        actual = getattr(br, f"{stage}_B2")
        expected = g[stage]["B2"]
        if expected == 0:
            assert actual == pytest.approx(0, abs=1e-10), f"{name} {stage}"
        else:
            assert actual == pytest.approx(expected, rel=_PARITY_TOL), f"{name} {stage}"


# =============================================================================
# Behavioral tests
# =============================================================================


class TestReturnShape:
    def test_returns_float_by_default(self, dgp_case):
        _, d, y, _ = dgp_case
        h = mse_optimal_bandwidth(d, y)
        assert isinstance(h, float)
        assert h > 0.0

    def test_return_diagnostics_true_returns_dataclass(self, dgp_case):
        _, d, y, _ = dgp_case
        br = mse_optimal_bandwidth(d, y, return_diagnostics=True)
        assert isinstance(br, BandwidthResult)
        assert br.h_mse > 0.0
        assert br.c_bw > 0.0
        assert br.b_mse > 0.0

    def test_float_return_matches_h_mse(self, dgp_case):
        _, d, y, _ = dgp_case
        h = mse_optimal_bandwidth(d, y)
        br = mse_optimal_bandwidth(d, y, return_diagnostics=True)
        # Different code paths but same computation; should be bit-exact.
        assert h == br.h_mse


class TestInputValidation:
    def test_mismatched_shapes_raise(self):
        rng = np.random.default_rng(0)
        d = rng.uniform(size=100)
        y = rng.normal(size=50)
        with pytest.raises(ValueError, match="same shape"):
            mse_optimal_bandwidth(d, y)

    def test_non_finite_d_raises(self):
        d = np.array([0.1, np.nan, 0.3, 0.4])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="d contains non-finite"):
            mse_optimal_bandwidth(d, y)

    def test_non_finite_y_raises(self):
        d = np.array([0.1, 0.2, 0.3, 0.4])
        y = np.array([1.0, 2.0, np.inf, 4.0])
        with pytest.raises(ValueError, match="y contains non-finite"):
            mse_optimal_bandwidth(d, y)

    def test_non_finite_boundary_raises(self):
        d = np.array([0.1, 0.2, 0.3, 0.4])
        y = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="boundary"):
            mse_optimal_bandwidth(d, y, boundary=np.nan)

    def test_unknown_kernel_raises(self):
        rng = np.random.default_rng(0)
        d = rng.uniform(size=100)
        y = rng.normal(size=100)
        with pytest.raises(ValueError, match="Unknown kernel"):
            mse_optimal_bandwidth(d, y, kernel="gaussian")

    def test_weights_raises_not_implemented(self):
        rng = np.random.default_rng(0)
        d = rng.uniform(size=100)
        y = rng.normal(size=100)
        w = np.ones_like(d)
        with pytest.raises(NotImplementedError, match="weights"):
            mse_optimal_bandwidth(d, y, weights=w)

    def test_bwcheck_exceeds_sample_size_raises(self):
        """bwcheck > N would cause IndexError inside the selector; guard it."""
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        d = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="bwcheck"):
            # Default bwcheck=21 exceeds N=5.
            lpbwselect_mse_dpi(y, d, eval_point=0.0, bwcheck=21)

    def test_bwcheck_zero_raises(self):
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        rng = np.random.default_rng(0)
        d = rng.uniform(size=100)
        y = rng.normal(size=100)
        with pytest.raises(ValueError, match="bwcheck"):
            lpbwselect_mse_dpi(y, d, eval_point=0.0, bwcheck=0)

    def test_bwcheck_negative_raises(self):
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        rng = np.random.default_rng(0)
        d = rng.uniform(size=100)
        y = rng.normal(size=100)
        with pytest.raises(ValueError, match="bwcheck"):
            lpbwselect_mse_dpi(y, d, eval_point=0.0, bwcheck=-1)

    def test_public_wrapper_fixes_vce_nn_nnmatch_3(self):
        """Pin the public API scope restriction documented in
        REGISTRY.md and the mse_optimal_bandwidth docstring.

        The wrapper hard-codes vce='nn', nnmatch=3 for Phase 1b; users
        needing other variance modes must go through the private port.
        This test ensures the behavior is frozen: changing it would be
        a scope expansion that should update REGISTRY.md and the
        docstring in lockstep.
        """
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        rng = np.random.default_rng(20260419)
        G = 2000
        d = rng.uniform(0, 1, size=G)
        y = d + d**2 + rng.normal(0, 0.5, size=G)
        via_wrapper = mse_optimal_bandwidth(
            d, y, kernel="epanechnikov", return_diagnostics=True
        )
        via_port_nn = lpbwselect_mse_dpi(
            y,
            d,
            eval_point=0.0,
            p=1,
            q=2,
            deriv=0,
            kernel="epa",
            bwcheck=21,
            bwregul=1.0,
            vce="nn",
            nnmatch=3,
            interior=False,
        )
        # vce/nnmatch/interior/p/deriv chosen by the wrapper must match
        # what the port call explicitly sets.
        assert via_wrapper.h_mse == via_port_nn.h_mse_dpi
        assert via_wrapper.b_mse == via_port_nn.b_mse_dpi
        assert via_wrapper.c_bw == via_port_nn.c_bw

    def test_bwcheck_none_on_tiny_sample_raises_valueerror(self):
        """bwcheck=None on a tiny sample must raise a clear ValueError
        from the per-stage support/rank guard in lprobust_bw, NOT an
        opaque LinAlgError or IndexError."""
        from diff_diff._nprobust_port import lpbwselect_mse_dpi

        d = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        with pytest.raises(ValueError, match="lprobust_bw"):
            lpbwselect_mse_dpi(y, d, eval_point=0.0, bwcheck=None)

    def test_interior_boundary_rejected(self):
        """boundary strictly inside the support must be rejected by
        the Phase 1b wrapper. Running the boundary selector at an
        interior point would silently use a symmetric kernel and
        produce a bandwidth incompatible with the one-sided fitter."""
        rng = np.random.default_rng(2026)
        d = rng.uniform(0.0, 1.0, size=500)
        y = d + d**2 + rng.normal(0, 0.5, size=500)
        with pytest.raises(ValueError, match="boundary"):
            mse_optimal_bandwidth(d, y, boundary=0.5)

    def test_upper_boundary_rejected(self):
        """boundary at d.max() (upper support edge) must be rejected."""
        rng = np.random.default_rng(2026)
        d = rng.uniform(0.0, 1.0, size=500)
        y = d + d**2 + rng.normal(0, 0.5, size=500)
        with pytest.raises(ValueError, match="boundary"):
            mse_optimal_bandwidth(d, y, boundary=float(d.max()))

    def test_boundary_equal_to_min_d_accepted(self):
        """Design 1 continuous-near-d_lower uses boundary = min(d)
        exactly; this must pass the applicability check."""
        rng = np.random.default_rng(20260419)
        d = rng.uniform(1.0, 2.0, size=1500)
        y = 3.0 + 0.5 * (d - 1.0) ** 2 + rng.normal(0, 0.3, size=1500)
        h = mse_optimal_bandwidth(d, y, boundary=float(d.min()))
        assert np.isfinite(h)
        assert h > 0.0

    def test_boundary_below_min_d_accepted(self):
        """Design 1' uses boundary = 0 when all D_2 > 0; boundary
        below d.min() must pass."""
        rng = np.random.default_rng(20260419)
        d = rng.uniform(0.01, 1.0, size=1500)  # d.min() > 0
        y = d + d**2 + rng.normal(0, 0.5, size=1500)
        h = mse_optimal_bandwidth(d, y, boundary=0.0)
        assert np.isfinite(h)
        assert h > 0.0


class TestKernelDispatch:
    """Different kernels produce different bandwidths."""

    def test_kernel_epa_vs_uni_differ(self):
        rng = np.random.default_rng(42)
        G = 1000
        d = rng.uniform(0, 1, size=G)
        y = d + d**2 + rng.normal(0, 0.5, size=G)
        h_epa = mse_optimal_bandwidth(d, y, kernel="epanechnikov")
        h_uni = mse_optimal_bandwidth(d, y, kernel="uniform")
        # Different C_rot constants (2.34 vs 1.843) -> different c_bw -> cascade.
        assert h_epa != h_uni
        assert h_epa > 0
        assert h_uni > 0

    def test_kernel_epa_vs_tri_differ(self):
        rng = np.random.default_rng(42)
        d = rng.uniform(0, 1, size=1000)
        y = d + d**2 + rng.normal(0, 0.5, size=1000)
        h_epa = mse_optimal_bandwidth(d, y, kernel="epanechnikov")
        h_tri = mse_optimal_bandwidth(d, y, kernel="triangular")
        assert h_epa != h_tri


class TestBoundary:
    def test_nonzero_boundary(self):
        """Design 1 continuous-near-d_lower case: evaluate at a non-zero
        boundary d_0 = d_lower."""
        rng = np.random.default_rng(2026)
        d = rng.uniform(1.0, 2.0, size=1500)
        y = 3.0 + 0.5 * (d - 1.0) ** 2 + rng.normal(0, 0.3, size=1500)
        h = mse_optimal_bandwidth(d, y, boundary=1.0)
        assert np.isfinite(h)
        assert h > 0.0


class TestDownstreamIntegration:
    """End-to-end: bandwidth feeds into local_linear_fit without error."""

    def test_selector_feeds_local_linear_fit(self, dgp_case):
        name, d, y, _ = dgp_case
        h = mse_optimal_bandwidth(d, y)
        fit = local_linear_fit(d, y, bandwidth=h, boundary=0.0, kernel="epanechnikov")
        # Basic sanity -- fit returned finite intercept and slope.
        assert np.isfinite(fit.intercept), name
        assert np.isfinite(fit.slope), name
        assert fit.n_effective > 0, name


class TestRateScaling:
    """h* scales as G^{-1/5} in Monte Carlo (15% MC tolerance)."""

    def test_rate_scaling_across_g(self):
        rng = np.random.default_rng(20260419)
        ratios = []
        for G in [500, 2000, 8000]:
            d = rng.uniform(0, 1, size=G)
            y = d + d**2 + rng.normal(0, 0.5, size=G)
            h = mse_optimal_bandwidth(d, y)
            ratios.append((G, h))
        # h(G_2) / h(G_1) should be approx (G_1 / G_2)^{1/5}.
        g1, h1 = ratios[0]
        g2, h2 = ratios[-1]
        expected_ratio = (g1 / g2) ** (1.0 / 5.0)
        actual_ratio = h2 / h1
        # 15% MC tolerance; rate check, not parity.
        assert actual_ratio == pytest.approx(expected_ratio, rel=0.15)
