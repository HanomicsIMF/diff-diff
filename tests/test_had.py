"""Tests for :class:`diff_diff.had.HeterogeneousAdoptionDiD` (Phase 2a).

Covers the 12 plan commit criteria:

1. All three design paths produce a finite result on synthetic DGPs.
2. ``design="auto"`` resolves correctly on each DGP + two edge cases.
3. Beta-scale WAS estimator at atol=1e-14:
   - Design 1' / continuous_at_zero:
     ``att = (mean(ΔY) - tau_bc) / mean(D)``
   - Design 1 / continuous_near_d_lower:
     ``att = (mean(ΔY) - tau_bc) / mean(D - d_lower)``
   - CI endpoints reverse under subtraction:
     ``CI_lower(att) = (mean(ΔY) - CI_upper_boundary) / den``
4. Mass-point Wald-IV point estimate matches manual formula at
   ``atol=1e-14``.
5. Mass-point 2SLS SE parity against hand-coded sandwich at
   ``atol=1e-12`` for HC1, classical, and CR1 (cluster-robust).
6. Mass-point + ``vcov_type in {hc2, hc2_bm}`` raises
   ``NotImplementedError``.
7. Panel-contract violations raise targeted ``ValueError``s.
8. NaN propagation: constant-y and mass-point degenerate inputs produce
   all-NaN inference.
9. sklearn clone round-trip preserves raw ``design="auto"``; fit is
   idempotent.
10. Scaffolding (``aggregate="event_study"``, ``survey``, ``weights``)
    raises ``NotImplementedError`` with phase pointers.
11. ``get_params()`` keys match ``__init__`` signature.
12. REGISTRY ticks tested indirectly via parity with the paper rules.
"""

from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff.had import (
    HeterogeneousAdoptionDiD,
    HeterogeneousAdoptionDiDResults,
    _aggregate_first_difference,
    _detect_design,
    _fit_mass_point_2sls,
    _validate_had_panel,
)
from diff_diff.local_linear import bias_corrected_local_linear
from tests.conftest import assert_nan_inference

# =============================================================================
# DGP helpers
# =============================================================================


def _make_panel(d_post, delta_y, periods=(1, 2), extra_cols=None):
    """Build a balanced two-period panel with ``D_{g,1} = 0``.

    Parameters
    ----------
    d_post : np.ndarray, shape (G,)
        Unit-level post-period dose ``D_{g,2}``.
    delta_y : np.ndarray, shape (G,)
        Unit-level first-difference outcome ``Y_{g,2} - Y_{g,1}``.
    periods : tuple
        (t_pre, t_post).
    extra_cols : dict or None
        Additional unit-constant columns (e.g., cluster variable).
    """
    G = len(d_post)
    t_pre, t_post = periods
    units = np.arange(G)
    df = pd.DataFrame(
        {
            "unit": np.repeat(units, 2),
            "period": np.tile([t_pre, t_post], G),
            "dose": np.column_stack([np.zeros(G), d_post]).ravel(),
            # Set period-1 outcome to 0; period-2 outcome = delta_y so that
            # Y_{g,2} - Y_{g,1} == delta_y exactly.
            "outcome": np.column_stack([np.zeros(G), delta_y]).ravel(),
        }
    )
    if extra_cols:
        for col, vals in extra_cols.items():
            df[col] = np.repeat(vals, 2)
    return df


def _dgp_continuous_at_zero(G, seed):
    """Design 1' DGP: uniform dose on [0, 1] with exact zero in the sample."""
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.0, 1.0, G)
    d[0] = 0.0  # guarantee continuous_at_zero auto-detection
    dy = 0.3 * d + 0.1 * rng.standard_normal(G)
    return d, dy


def _dgp_continuous_near_d_lower(G, seed):
    """Design 1 continuous-near-d_lower DGP: Beta(2,2) shifted to [0.1, 1]."""
    rng = np.random.default_rng(seed)
    u = rng.beta(2, 2, G)
    d = 0.1 + 0.9 * u
    dy = 0.3 * d + 0.1 * rng.standard_normal(G)
    return d, dy


def _dgp_mass_point(G, seed, d_lower=0.5, mass_frac=0.3, beta=0.3):
    """Mass-point DGP: ``mass_frac`` at d_lower, rest Uniform(d_lower, 1)."""
    rng = np.random.default_rng(seed)
    mass_n = int(mass_frac * G)
    d = np.concatenate([np.full(mass_n, d_lower), rng.uniform(d_lower, 1.0, G - mass_n)])
    dy = beta * d + 0.1 * rng.standard_normal(G)
    return d, dy


# =============================================================================
# Criterion 1: Smoke tests - all 3 design paths produce finite output
# =============================================================================


class TestSmokeAllDesigns:
    def test_continuous_at_zero_finite(self):
        d, dy = _dgp_continuous_at_zero(500, seed=42)
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert np.isfinite(r.att)
        assert np.isfinite(r.se)
        assert r.se > 0

    def test_continuous_near_d_lower_finite(self):
        d, dy = _dgp_continuous_near_d_lower(500, seed=42)
        r = HeterogeneousAdoptionDiD(design="continuous_near_d_lower").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert np.isfinite(r.att)
        assert np.isfinite(r.se)
        assert r.se > 0

    def test_mass_point_finite(self):
        d, dy = _dgp_mass_point(500, seed=42)
        r = HeterogeneousAdoptionDiD(design="mass_point").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert np.isfinite(r.att)
        assert np.isfinite(r.se)
        assert r.se > 0

    def test_result_is_dataclass(self):
        d, dy = _dgp_continuous_at_zero(400, seed=0)
        r = HeterogeneousAdoptionDiD().fit(_make_panel(d, dy), "outcome", "dose", "period", "unit")
        assert isinstance(r, HeterogeneousAdoptionDiDResults)

    def test_continuous_populates_bandwidth_diagnostics(self):
        d, dy = _dgp_continuous_at_zero(400, seed=0)
        r = HeterogeneousAdoptionDiD().fit(_make_panel(d, dy), "outcome", "dose", "period", "unit")
        assert r.bandwidth_diagnostics is not None
        assert r.bias_corrected_fit is not None

    def test_mass_point_nulls_bandwidth_diagnostics(self):
        d, dy = _dgp_mass_point(400, seed=0)
        r = HeterogeneousAdoptionDiD(design="mass_point").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.bandwidth_diagnostics is None
        assert r.bias_corrected_fit is None
        assert r.n_mass_point is not None
        assert r.n_above_d_lower is not None

    def test_continuous_nulls_mass_point_counts(self):
        d, dy = _dgp_continuous_at_zero(400, seed=0)
        r = HeterogeneousAdoptionDiD().fit(_make_panel(d, dy), "outcome", "dose", "period", "unit")
        assert r.n_mass_point is None
        assert r.n_above_d_lower is None


# =============================================================================
# Criterion 2: design="auto" detection rule
# =============================================================================


class TestDesignAutoDetect:
    def test_detect_design_1_prime_exact_zero(self):
        d, _ = _dgp_continuous_at_zero(500, seed=0)
        assert _detect_design(d) == "continuous_at_zero"

    def test_detect_design_continuous_near_d_lower(self):
        d, _ = _dgp_continuous_near_d_lower(500, seed=0)
        assert _detect_design(d) == "continuous_near_d_lower"

    def test_detect_mass_point(self):
        d, _ = _dgp_mass_point(500, seed=0)
        assert _detect_design(d) == "mass_point"

    def test_edge_small_mass_at_zero_resolves_continuous_at_zero(self):
        """Plan criterion 2 edge-case (a): 3% at D=0 + 97% Uniform(0.5, 1)."""
        rng = np.random.default_rng(0)
        G = 1000
        mass_n = int(0.03 * G)
        d = np.concatenate([np.zeros(mass_n), rng.uniform(0.5, 1.0, G - mass_n)])
        assert _detect_design(d) == "continuous_at_zero"

    def test_edge_shifted_beta_not_small_enough_for_design_1_prime(self):
        """Plan criterion 2 edge-case (b): d.min/median ~ 0.03 > 0.01 threshold."""
        rng = np.random.default_rng(0)
        u = rng.beta(2, 2, 1000)
        d = 0.03 + u
        assert _detect_design(d) == "continuous_near_d_lower"

    def test_design_auto_dispatches_correctly_at_fit(self):
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        r = HeterogeneousAdoptionDiD(design="auto").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.design == "continuous_at_zero"

    def test_design_auto_mass_point_at_fit(self):
        d, dy = _dgp_mass_point(500, seed=0)
        r = HeterogeneousAdoptionDiD(design="auto").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.design == "mass_point"

    def test_auto_does_not_mutate_self_design(self):
        """Plan decision #14: self.design preserves raw 'auto' after fit."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        est = HeterogeneousAdoptionDiD(design="auto")
        _ = est.fit(_make_panel(d, dy), "outcome", "dose", "period", "unit")
        assert est.design == "auto"
        assert est.get_params()["design"] == "auto"


# =============================================================================
# Criterion 3: Beta-scale rescaling parity
# =============================================================================


class TestBetaScaleRescaling:
    """Plan commit criterion #3 + review P0: the continuous estimator is

        att = (mean(ΔY) - tau_bc) / den

    with ``den = mean(D)`` for Design 1' and ``den = mean(D - d_lower)``
    for Design 1 continuous-near-d_lower. SE is ``se_robust / |den|``.
    CI endpoints are computed via ``att +/- z * se`` (endpoints reverse
    relative to the boundary-limit CI because the numerator is
    ``ΔȲ - tau_bc``).
    """

    def test_att_design_1_prime(self):
        """att = (mean(ΔY) - tau_bc) / D_bar for Design 1' at atol=1e-14."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        bc = bias_corrected_local_linear(d=d, y=dy, boundary=0.0, alpha=0.05)
        d_bar = float(d.mean())
        dy_mean = float(dy.mean())
        expected = (dy_mean - float(bc.estimate_bias_corrected)) / d_bar
        assert abs(r.att - expected) < 1e-14

    def test_se_design_1_prime(self):
        """se = se_robust / |D_bar| for Design 1' at atol=1e-14."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        bc = bias_corrected_local_linear(d=d, y=dy, boundary=0.0, alpha=0.05)
        expected = float(bc.se_robust) / abs(float(d.mean()))
        assert abs(r.se - expected) < 1e-14

    def test_ci_endpoints_reverse_under_subtraction(self):
        """Because att = (ΔȲ - tau_bc)/D_bar, CI endpoints reverse:

        CI_lower(att) = (ΔȲ - CI_upper_boundary) / D_bar
        CI_upper(att) = (ΔȲ - CI_lower_boundary) / D_bar
        """
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        bc = bias_corrected_local_linear(d=d, y=dy, boundary=0.0, alpha=0.05)
        d_bar = float(d.mean())
        dy_mean = float(dy.mean())
        # CI bounds on the att scale, computed by endpoint reversal from
        # the boundary-limit CI.
        expected_lower = (dy_mean - float(bc.ci_high)) / d_bar
        expected_upper = (dy_mean - float(bc.ci_low)) / d_bar
        assert abs(r.conf_int[0] - expected_lower) < 1e-14
        assert abs(r.conf_int[1] - expected_upper) < 1e-14

    def test_att_design_1_continuous_near_d_lower(self):
        """att = (mean(ΔY) - tau_bc) / mean(D - d_lower) for Design 1 at atol=1e-14."""
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        panel = _make_panel(d, dy)
        d_lower_val = float(d.min())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = HeterogeneousAdoptionDiD(design="continuous_near_d_lower").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        d_reg = d - d_lower_val
        bc = bias_corrected_local_linear(d=d_reg, y=dy, boundary=0.0, alpha=0.05)
        den = float((d - d_lower_val).mean())
        dy_mean = float(dy.mean())
        expected = (dy_mean - float(bc.estimate_bias_corrected)) / den
        assert abs(r.att - expected) < 1e-14

    def test_att_recovers_true_beta_design_1_prime(self):
        """Sanity: on a known DGP with beta=0.3, att should be close to 0.3."""
        rng = np.random.default_rng(0)
        G = 2000
        d = rng.uniform(0, 1, G)
        d[0] = 0.0
        dy = 0.3 * d + 0.05 * rng.standard_normal(G)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        # Asymptotic: expect att close to 0.3 at G=2000, n=4000 observations.
        assert abs(r.att - 0.3) < 0.1

    def test_att_recovers_true_beta_continuous_near_d_lower(self):
        """Sanity: Design 1 DGP with beta_d_lower=0.3 recovers beta at scale."""
        rng = np.random.default_rng(0)
        G = 2000
        u = rng.beta(2, 2, G)
        d = 0.1 + 0.9 * u  # d_lower ~ 0.1
        # True WAS_{d_lower} = 0.3 since dy = 0.3 * (d - d_lower) + noise
        dy = 0.3 * (d - 0.1) + 0.05 * rng.standard_normal(G)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = HeterogeneousAdoptionDiD(design="continuous_near_d_lower").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        assert abs(r.att - 0.3) < 0.1

    def test_dose_mean_stored_on_result(self):
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD().fit(panel, "outcome", "dose", "period", "unit")
        assert abs(r.dose_mean - float(d.mean())) < 1e-14


# =============================================================================
# Criterion 4: Mass-point Wald-IV point estimate parity
# =============================================================================


class TestMassPointWaldIV:
    def test_wald_iv_point_estimate(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="mass_point").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        Z = (d > 0.5).astype(float)
        expected = (dy[Z == 1].mean() - dy[Z == 0].mean()) / (d[Z == 1].mean() - d[Z == 0].mean())
        assert abs(r.att - expected) < 1e-14

    def test_wald_iv_equals_2sls(self):
        """Sanity: Wald-IV is exactly 2SLS for binary instrument."""
        d, dy = _dgp_mass_point(500, seed=7)
        Z = (d > 0.5).astype(float).reshape(-1, 1)
        # 2SLS via Z'X invert: beta = [(Z'X)^-1 Z'y][1]
        X = np.column_stack([np.ones_like(d), d])
        Zd = np.column_stack([np.ones_like(d), Z.ravel()])
        beta_2sls = np.linalg.inv(Zd.T @ X) @ (Zd.T @ dy)
        beta_wald = (dy[Z.ravel() == 1].mean() - dy[Z.ravel() == 0].mean()) / (
            d[Z.ravel() == 1].mean() - d[Z.ravel() == 0].mean()
        )
        assert abs(float(beta_2sls[1]) - beta_wald) < 1e-12

    def test_mass_point_n_counts_populated(self):
        d, dy = _dgp_mass_point(500, seed=0, d_lower=0.5, mass_frac=0.3)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="mass_point").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        assert r.n_mass_point == int(0.3 * 500)
        assert r.n_above_d_lower == 500 - int(0.3 * 500)
        assert r.n_treated == r.n_above_d_lower
        assert r.n_control == r.n_mass_point


# =============================================================================
# Criterion 5: Mass-point 2SLS SE sandwich parity
# =============================================================================


def _manual_2sls_sandwich_se(d, dy, d_lower, vcov_type, cluster=None):
    """Hand-coded textbook 2SLS sandwich using structural residuals.

    Returns se_beta for the coefficient on d. Mirrors the helper in had.py
    but computed from scratch to serve as the parity reference.
    """
    n = len(d)
    Z = (d > d_lower).astype(np.float64)
    dose_gap = d[Z == 1].mean() - d[Z == 0].mean()
    dy_gap = dy[Z == 1].mean() - dy[Z == 0].mean()
    beta = dy_gap / dose_gap
    alpha_hat = dy.mean() - beta * d.mean()
    u = dy - alpha_hat - beta * d  # STRUCTURAL residuals
    X = np.column_stack([np.ones(n), d])
    Zd = np.column_stack([np.ones(n), Z])
    ZtX_inv = np.linalg.inv(Zd.T @ X)

    if cluster is not None:
        Omega = np.zeros((2, 2))
        clusters = pd.unique(cluster)
        G = len(clusters)
        for c in clusters:
            idx = cluster == c
            s = Zd[idx].T @ u[idx]
            Omega += np.outer(s, s)
        Omega *= (G / (G - 1)) * ((n - 1) / (n - 2))
    elif vcov_type == "classical":
        sigma2 = (u * u).sum() / (n - 2)
        Omega = sigma2 * (Zd.T @ Zd)
    elif vcov_type == "hc1":
        Omega = (n / (n - 2)) * (Zd.T @ ((u * u)[:, None] * Zd))
    else:
        raise ValueError(f"unknown vcov_type={vcov_type}")

    V = ZtX_inv @ Omega @ ZtX_inv.T
    return float(np.sqrt(V[1, 1]))


class TestMassPointSEParity:
    def test_classical_parity(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="classical").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        expected = _manual_2sls_sandwich_se(d, dy, 0.5, "classical")
        assert abs(r.se - expected) < 1e-12

    def test_hc1_parity(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="hc1").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        expected = _manual_2sls_sandwich_se(d, dy, 0.5, "hc1")
        assert abs(r.se - expected) < 1e-12

    def test_cr1_cluster_robust_parity(self):
        d, dy = _dgp_mass_point(500, seed=0)
        cluster_ids = np.tile(np.arange(50), 10)  # 50 clusters of 10 units
        panel = _make_panel(d, dy, extra_cols={"state": cluster_ids})
        r = HeterogeneousAdoptionDiD(design="mass_point", cluster="state").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        expected = _manual_2sls_sandwich_se(d, dy, 0.5, "hc1", cluster=cluster_ids)
        assert abs(r.se - expected) < 1e-12

    def test_robust_alias_maps_to_hc1(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        r_robust = HeterogeneousAdoptionDiD(design="mass_point", robust=True).fit(
            panel, "outcome", "dose", "period", "unit"
        )
        r_hc1 = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="hc1").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        assert r_robust.se == r_hc1.se

    def test_robust_false_maps_to_classical(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        r_robust = HeterogeneousAdoptionDiD(design="mass_point", robust=False).fit(
            panel, "outcome", "dose", "period", "unit"
        )
        r_classical = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="classical").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        assert r_robust.se == r_classical.se

    def test_vcov_type_explicit_overrides_robust(self):
        """When vcov_type is explicit, robust is ignored."""
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="classical", robust=True).fit(
            panel, "outcome", "dose", "period", "unit"
        )
        assert r.vcov_type == "classical"


# =============================================================================
# Criterion 6: hc2 / hc2_bm raise NotImplementedError
# =============================================================================


class TestMassPointUnsupportedVcov:
    def test_hc2_raises(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="hc2")
        with pytest.raises(NotImplementedError, match="HC2"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_hc2_bm_raises(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="hc2_bm")
        with pytest.raises(NotImplementedError, match="HC2"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_hc2_pointer_references_followup_pr(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="hc2")
        with pytest.raises(NotImplementedError, match="follow-up"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_vcov_type_ignored_on_continuous(self):
        """hc2 passed with continuous design emits warning, does not raise."""
        d, dy = _dgp_continuous_at_zero(300, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", vcov_type="hc2")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = est.fit(panel, "outcome", "dose", "period", "unit")
            assert any("ignored" in str(warn.message).lower() for warn in w)
        assert np.isfinite(r.att)

    def test_robust_true_ignored_on_continuous_warns(self):
        """Review P2 round 9: robust=True on continuous path must warn.

        The continuous designs use the CCT-2014 robust SE unconditionally;
        robust= is a mass-point-only backward-compat alias for vcov_type.
        Passing robust=True on a continuous path has no effect on the
        computed SE, so the user must get a warning that the flag was
        ignored.
        """
        d, dy = _dgp_continuous_at_zero(300, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", robust=True)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = est.fit(panel, "outcome", "dose", "period", "unit")
            robust_warnings = [warn for warn in w if "robust" in str(warn.message).lower()]
            assert len(robust_warnings) >= 1
        assert np.isfinite(r.att)

    def test_robust_false_silent_on_continuous(self):
        """robust=False (the default) on continuous path emits no robust-warn."""
        d, dy = _dgp_continuous_at_zero(300, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", robust=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = est.fit(panel, "outcome", "dose", "period", "unit")
            robust_warnings = [warn for warn in w if "robust=True is ignored" in str(warn.message)]
            assert len(robust_warnings) == 0
        assert np.isfinite(r.att)


# =============================================================================
# Criterion 7: Panel-contract violations
# =============================================================================


class TestPanelContract:
    def test_missing_outcome_col_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="column"):
            est.fit(panel, "missing", "dose", "period", "unit")

    def test_missing_dose_col_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="column"):
            est.fit(panel, "outcome", "missing", "period", "unit")

    def test_missing_time_col_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="column"):
            est.fit(panel, "outcome", "dose", "missing", "unit")

    def test_missing_unit_col_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="column"):
            est.fit(panel, "outcome", "dose", "period", "missing")

    def test_nonzero_pre_period_dose_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        panel.loc[panel["period"] == 1, "dose"] = 0.5
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match=r"D_\{g,1\}|pre-period"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_unbalanced_panel_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy).iloc[:-1]  # drop one row
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match=r"[Uu]nbalanced|[Bb]alanced"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_three_periods_without_first_treat_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel2 = _make_panel(d, dy)
        panel3 = pd.concat([panel2, panel2.assign(period=3)])
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match=r"two time periods|Phase 2b"):
            est.fit(panel3, "outcome", "dose", "period", "unit")

    def test_three_periods_with_first_treat_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel2 = _make_panel(d, dy)
        panel3 = pd.concat([panel2, panel2.assign(period=3)])
        panel3["ft"] = 2  # arbitrary first_treat
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match=r"two time periods|Phase 2b"):
            est.fit(
                panel3,
                "outcome",
                "dose",
                "period",
                "unit",
                first_treat_col="ft",
            )

    def test_single_period_raises(self):
        d, _ = _dgp_continuous_at_zero(200, seed=0)
        panel = pd.DataFrame(
            {
                "unit": np.arange(200),
                "period": 2,
                "dose": d,
                "outcome": np.zeros(200),
            }
        )
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="two-period"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_nan_outcome_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        panel.loc[0, "outcome"] = np.nan
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="NaN"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_nan_dose_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        panel.loc[3, "dose"] = np.nan
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="NaN"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_duplicate_unit_period_raises(self):
        """Two observations of the same unit-period cell."""
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        panel = pd.concat([panel, panel.iloc[[0]]])  # duplicate first row
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match=r"[Uu]nbalanced|observation"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_first_treat_col_invalid_cohort_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        # Set first_treat values to {0, 5, 2} where 5 is not t_post.
        ft_unit = np.where(np.arange(200) % 2 == 0, 0, 5)
        panel["ft"] = np.repeat(ft_unit, 2)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match=r"first_treat_col"):
            est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                first_treat_col="ft",
            )

    def test_first_treat_col_mixed_row_nan_raises(self):
        """Review P2 round 8: per-unit rows like [valid, NaN] must be rejected.

        `groupby().first()` silently skips NaNs; a unit with [0, NaN]
        collapses to first_treat=0 and a unit-level NaN check would
        pass. Row-level validation must catch the NaN on the bad row.
        """
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        # Unit-level first_treat all zero (never-treated); inject a NaN on
        # exactly the second row of unit 0 (t_post row).
        panel["ft"] = 0.0
        unit0_post_idx = panel[(panel["unit"] == 0) & (panel["period"] == 2)].index[0]
        panel.loc[unit0_post_idx, "ft"] = np.nan
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="NaN"):
            est.fit(panel, "outcome", "dose", "period", "unit", first_treat_col="ft")

    def test_first_treat_col_mixed_row_invalid_value_raises(self):
        """Per-unit rows like [valid, invalid_value] must be rejected."""
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        panel["ft"] = 0.0
        # Inject an out-of-domain value on unit 0's post-period row.
        unit0_post_idx = panel[(panel["unit"] == 0) & (panel["period"] == 2)].index[0]
        panel.loc[unit0_post_idx, "ft"] = 999.0
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match=r"first_treat_col.*999"):
            est.fit(panel, "outcome", "dose", "period", "unit", first_treat_col="ft")


# =============================================================================
# Criterion 8: NaN propagation
# =============================================================================


class TestNaNPropagation:
    def test_constant_y_produces_nan_inference(self):
        """Constant outcome -> zero residuals -> NaN via safe_inference."""
        d, _ = _dgp_continuous_at_zero(500, seed=0)
        dy_zero = np.zeros_like(d)
        panel = _make_panel(d, dy_zero)
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        # All inference fields NaN when SE is non-finite.
        assert_nan_inference(
            {
                "se": r.se,
                "t_stat": r.t_stat,
                "p_value": r.p_value,
                "conf_int": r.conf_int,
            }
        )

    def test_mass_point_all_at_d_lower_nan(self):
        """Degenerate mass-point: all units at d_lower -> NaN."""
        rng = np.random.default_rng(0)
        G = 500
        d = np.full(G, 0.5)  # all at 0.5
        dy = 0.1 * rng.standard_normal(G)
        panel = _make_panel(d, dy)
        # Avoid triggering pre-period D=0 check by starting at 0.5 at t2.
        r = HeterogeneousAdoptionDiD(design="mass_point", d_lower=0.5).fit(
            panel, "outcome", "dose", "period", "unit"
        )
        assert np.isnan(r.att)
        assert_nan_inference(
            {
                "se": r.se,
                "t_stat": r.t_stat,
                "p_value": r.p_value,
                "conf_int": r.conf_int,
            }
        )

    def test_helper_returns_nan_on_empty_z_one(self):
        """_fit_mass_point_2sls returns NaN when no units above d_lower."""
        d = np.full(50, 0.5)
        dy = np.random.default_rng(0).standard_normal(50)
        beta, se = _fit_mass_point_2sls(d, dy, 0.5, None, "hc1")
        assert np.isnan(beta)
        assert np.isnan(se)

    def test_helper_returns_nan_on_empty_z_zero(self):
        """_fit_mass_point_2sls returns NaN when no units at d_lower."""
        d = np.full(50, 0.6)  # all strictly above d_lower=0.5
        dy = np.random.default_rng(0).standard_normal(50)
        beta, se = _fit_mass_point_2sls(d, dy, 0.5, None, "hc1")
        assert np.isnan(beta)
        assert np.isnan(se)

    def test_single_cluster_cr1_returns_nan(self):
        """CR1 with only 1 cluster is undefined -> NaN."""
        rng = np.random.default_rng(0)
        G = 100
        d = np.concatenate([np.full(30, 0.5), rng.uniform(0.5, 1.0, G - 30)])
        dy = 0.3 * d + 0.1 * rng.standard_normal(G)
        panel = _make_panel(d, dy, extra_cols={"state": np.zeros(G, dtype=int)})  # single cluster
        r = HeterogeneousAdoptionDiD(design="mass_point", cluster="state").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        assert np.isnan(r.se)


# =============================================================================
# Criterion 9: sklearn clone round-trip + fit idempotence
# =============================================================================


class TestSklearnCompat:
    def test_get_params_returns_all_constructor_args(self):
        est = HeterogeneousAdoptionDiD(
            design="continuous_near_d_lower",
            d_lower=0.3,
            kernel="triangular",
            alpha=0.1,
            vcov_type="hc1",
            robust=True,
            cluster="state",
        )
        params = est.get_params()
        assert params == {
            "design": "continuous_near_d_lower",
            "d_lower": 0.3,
            "kernel": "triangular",
            "alpha": 0.1,
            "vcov_type": "hc1",
            "robust": True,
            "cluster": "state",
        }

    def test_clone_round_trip(self):
        est = HeterogeneousAdoptionDiD(design="auto", alpha=0.1, kernel="triangular")
        est2 = HeterogeneousAdoptionDiD(**est.get_params())
        assert est.get_params() == est2.get_params()

    def test_fit_idempotent_same_att(self):
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD()
        r1 = est.fit(panel, "outcome", "dose", "period", "unit")
        r2 = est.fit(panel, "outcome", "dose", "period", "unit")
        assert r1.att == r2.att
        assert r1.se == r2.se
        assert r1.conf_int == r2.conf_int

    def test_set_params_updates_and_returns_self(self):
        est = HeterogeneousAdoptionDiD()
        ret = est.set_params(alpha=0.1, design="continuous_at_zero")
        assert ret is est
        assert est.alpha == 0.1
        assert est.design == "continuous_at_zero"

    def test_set_params_invalid_key_raises(self):
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="Invalid parameter"):
            est.set_params(not_a_param=True)

    def test_set_params_rejects_method_names(self):
        """Review P1 round 10: set_params must restrict to constructor keys,
        not any hasattr-able name. Method names like 'fit' must raise,
        else they would silently overwrite the method.
        """
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="Invalid parameter"):
            est.set_params(fit="not_a_method")
        # sanity: fit is still callable on the class
        assert callable(est.fit)

    def test_set_params_rejects_private_attrs(self):
        """Internal-looking attribute names must also raise."""
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="Invalid parameter"):
            est.set_params(_internal=42)

    def test_get_params_accepts_deep_keyword(self):
        """Review P1 round 10: get_params must match sklearn's signature.

        sklearn.base.BaseEstimator.get_params(deep=True). This estimator
        has no nested sub-estimators, so deep=True and deep=False return
        the same dict, but the keyword must be accepted.
        """
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", alpha=0.1)
        params_default = est.get_params()
        params_deep_true = est.get_params(deep=True)
        params_deep_false = est.get_params(deep=False)
        assert params_default == params_deep_true == params_deep_false

    def test_sklearn_clone_round_trip_if_available(self):
        """If sklearn is installed, sklearn.base.clone round-trips the estimator."""
        sklearn_base = pytest.importorskip("sklearn.base")
        est = HeterogeneousAdoptionDiD(design="auto", alpha=0.1, kernel="triangular")
        cloned = sklearn_base.clone(est)
        assert cloned.get_params() == est.get_params()
        assert cloned is not est
        # clone produces a fresh instance of the same class.
        assert type(cloned) is type(est)

    def test_set_params_invalid_design_raises(self):
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="design"):
            est.set_params(design="made_up")

    def test_set_params_rollback_on_failure(self):
        """Review P2 round 11: set_params must be ATOMIC.

        A failing call (valid key but value violates constructor
        constraints) must leave the estimator unchanged so the caller
        can catch the ValueError and reuse the object.
        """
        est = HeterogeneousAdoptionDiD(alpha=0.05, design="continuous_at_zero")
        baseline = est.get_params()
        # Multi-key call where alpha is valid but design is invalid.
        # The old (non-atomic) code would have set alpha before raising
        # on design, leaving the estimator half-mutated.
        with pytest.raises(ValueError):
            est.set_params(alpha=0.1, design="garbage_design")
        assert est.get_params() == baseline

    def test_set_params_rollback_on_invalid_key(self):
        """Rejecting an unknown key must leave self unchanged."""
        est = HeterogeneousAdoptionDiD(alpha=0.05)
        baseline = est.get_params()
        with pytest.raises(ValueError):
            est.set_params(alpha=0.1, not_a_param=True)
        assert est.get_params() == baseline

    def test_set_params_rollback_on_invalid_alpha(self):
        """alpha outside (0, 1) must leave self unchanged."""
        est = HeterogeneousAdoptionDiD(alpha=0.05, design="continuous_at_zero")
        baseline = est.get_params()
        with pytest.raises(ValueError):
            est.set_params(alpha=1.5, kernel="triangular")
        assert est.get_params() == baseline


# =============================================================================
# Criterion 10: Scaffolding raises
# =============================================================================


class TestScaffoldingRejections:
    def test_aggregate_event_study_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(NotImplementedError, match="2b"):
            est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                aggregate="event_study",
            )

    def test_aggregate_invalid_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match="Invalid aggregate"):
            est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                aggregate="garbage",
            )

    def test_survey_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(NotImplementedError, match="survey"):
            est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                survey="anything",
            )

    def test_weights_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(NotImplementedError, match="weights"):
            est.fit(
                panel,
                "outcome",
                "dose",
                "period",
                "unit",
                weights=np.ones(200),
            )


# =============================================================================
# Criterion 11: get_params signature enumeration
# =============================================================================


class TestGetParamsContract:
    def test_get_params_matches_init_signature(self):
        sig_params = set(inspect.signature(HeterogeneousAdoptionDiD.__init__).parameters.keys()) - {
            "self"
        }
        gp_params = set(HeterogeneousAdoptionDiD().get_params().keys())
        assert sig_params == gp_params

    def test_set_params_covers_all_init_params(self):
        """Every __init__ param must be settable via set_params."""
        est = HeterogeneousAdoptionDiD()
        params = est.get_params()
        # Round-trip via set_params
        new_est = HeterogeneousAdoptionDiD()
        new_est.set_params(**params)
        assert new_est.get_params() == params


# =============================================================================
# Result class methods
# =============================================================================


class TestResultMethods:
    def _result(self):
        d, dy = _dgp_continuous_at_zero(400, seed=0)
        panel = _make_panel(d, dy)
        return HeterogeneousAdoptionDiD().fit(panel, "outcome", "dose", "period", "unit")

    def test_summary_returns_string(self):
        r = self._result()
        s = r.summary()
        assert isinstance(s, str)
        assert "HeterogeneousAdoptionDiD" in s
        assert "WAS" in s
        assert "Confidence Interval" in s

    def test_summary_uses_target_parameter_for_row_label(self):
        """Review P2: the estimate row must render target_parameter (WAS or
        WAS_d_lower), not hardcoded 'WAS'.
        """
        # Design 1' -> target_parameter = "WAS"
        d, dy = _dgp_continuous_at_zero(400, seed=0)
        panel = _make_panel(d, dy)
        r_d1p = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        s_d1p = r_d1p.summary()
        assert r_d1p.target_parameter == "WAS"
        assert "WAS" in s_d1p

        # Design 1 continuous-near-d_lower -> target_parameter = "WAS_d_lower"
        d, dy = _dgp_continuous_near_d_lower(400, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_d1 = HeterogeneousAdoptionDiD(design="continuous_near_d_lower").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        assert r_d1.target_parameter == "WAS_d_lower"
        assert "WAS_d_lower" in r_d1.summary()

        # Design 1 mass-point -> target_parameter = "WAS_d_lower"
        d, dy = _dgp_mass_point(400, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_mp = HeterogeneousAdoptionDiD(design="mass_point").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        assert r_mp.target_parameter == "WAS_d_lower"
        assert "WAS_d_lower" in r_mp.summary()

    def test_print_summary_executes(self, capsys):
        r = self._result()
        r.print_summary()
        captured = capsys.readouterr()
        assert "HeterogeneousAdoptionDiD" in captured.out

    def test_to_dict_populated(self):
        r = self._result()
        d = r.to_dict()
        assert "att" in d
        assert "se" in d
        assert "design" in d
        assert "target_parameter" in d
        assert "d_lower" in d
        assert "dose_mean" in d
        assert "n_obs" in d
        assert d["design"] == "continuous_at_zero"

    def test_to_dataframe_populated(self):
        r = self._result()
        df = r.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "att" in df.columns

    def test_repr_concise(self):
        r = self._result()
        s = repr(r)
        assert "HeterogeneousAdoptionDiDResults" in s
        assert "att=" in s
        assert "design=" in s

    def test_mass_point_summary_includes_mass_count(self):
        d, dy = _dgp_mass_point(400, seed=0)
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="mass_point").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        s = r.summary()
        assert "mass point" in s.lower() or "At d_lower" in s


# =============================================================================
# Design metadata
# =============================================================================


class TestDesignMetadata:
    def test_target_parameter_design_1_prime(self):
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.target_parameter == "WAS"

    def test_target_parameter_design_1(self):
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        r = HeterogeneousAdoptionDiD(design="continuous_near_d_lower").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.target_parameter == "WAS_d_lower"

    def test_target_parameter_mass_point(self):
        d, dy = _dgp_mass_point(500, seed=0)
        r = HeterogeneousAdoptionDiD(design="mass_point").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.target_parameter == "WAS_d_lower"

    def test_d_lower_zero_for_design_1_prime(self):
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.d_lower == 0.0

    def test_d_lower_from_data_for_continuous_near(self):
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        r = HeterogeneousAdoptionDiD(design="continuous_near_d_lower").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert abs(r.d_lower - float(d.min())) < 1e-14

    def test_d_lower_explicit_override(self):
        """d_lower override must satisfy d.min() >= d_lower (else negative shifted doses)."""
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        # d.min() is around 0.1 + epsilon for this DGP; override within that.
        d_lower_user = float(d.min())  # explicit but equal to default
        r = HeterogeneousAdoptionDiD(design="continuous_near_d_lower", d_lower=d_lower_user).fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert abs(r.d_lower - d_lower_user) < 1e-14

    def test_inference_method_continuous(self):
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        r = HeterogeneousAdoptionDiD().fit(_make_panel(d, dy), "outcome", "dose", "period", "unit")
        assert r.inference_method == "analytical_nonparametric"

    def test_inference_method_mass_point(self):
        d, dy = _dgp_mass_point(500, seed=0)
        r = HeterogeneousAdoptionDiD(design="mass_point").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.inference_method == "analytical_2sls"

    def test_survey_metadata_always_none_in_phase_2a(self):
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        r = HeterogeneousAdoptionDiD().fit(_make_panel(d, dy), "outcome", "dose", "period", "unit")
        assert r.survey_metadata is None

    def test_alpha_stored_on_result(self):
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        r = HeterogeneousAdoptionDiD(alpha=0.1).fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.alpha == 0.1


# =============================================================================
# Constructor validation
# =============================================================================


class TestConstructorValidation:
    def test_invalid_design_raises(self):
        with pytest.raises(ValueError, match="Invalid design"):
            HeterogeneousAdoptionDiD(design="random_garbage")

    def test_alpha_zero_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            HeterogeneousAdoptionDiD(alpha=0.0)

    def test_alpha_one_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            HeterogeneousAdoptionDiD(alpha=1.0)

    def test_alpha_negative_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            HeterogeneousAdoptionDiD(alpha=-0.05)

    def test_invalid_vcov_type_raises(self):
        with pytest.raises(ValueError, match="vcov_type"):
            HeterogeneousAdoptionDiD(vcov_type="garbage")

    def test_vcov_type_none_accepted(self):
        est = HeterogeneousAdoptionDiD(vcov_type=None)
        assert est.vcov_type is None

    def test_d_lower_none_accepted(self):
        est = HeterogeneousAdoptionDiD(d_lower=None)
        assert est.d_lower is None

    def test_d_lower_float_accepted(self):
        est = HeterogeneousAdoptionDiD(d_lower=0.3)
        assert est.d_lower == 0.3

    def test_d_lower_nan_raises(self):
        """Review P1 round 13: d_lower=NaN must be rejected in __init__."""
        with pytest.raises(ValueError, match=r"d_lower.*finite"):
            HeterogeneousAdoptionDiD(d_lower=float("nan"))

    def test_d_lower_posinf_raises(self):
        with pytest.raises(ValueError, match=r"d_lower.*finite"):
            HeterogeneousAdoptionDiD(d_lower=float("inf"))

    def test_d_lower_neginf_raises(self):
        with pytest.raises(ValueError, match=r"d_lower.*finite"):
            HeterogeneousAdoptionDiD(d_lower=float("-inf"))

    def test_d_lower_nan_via_set_params_raises(self):
        """d_lower=NaN through set_params must also raise (atomic rollback)."""
        est = HeterogeneousAdoptionDiD(d_lower=0.3)
        baseline = est.get_params()
        with pytest.raises(ValueError, match=r"d_lower.*finite"):
            est.set_params(d_lower=float("nan"))
        # Atomic rollback: d_lower unchanged after failure.
        assert est.get_params() == baseline


# =============================================================================
# Explicit design override (don't auto-reject)
# =============================================================================


class TestExplicitDesignOverrides:
    def test_force_continuous_at_zero_on_mass_point_data(self):
        """Forcing Design 1' on mass-point data should run (may produce wide CIs)."""
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        # Phase 1c's _validate_had_inputs would reject this (mass point),
        # so this will raise NotImplementedError from underneath, NOT from had.py.
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero")
        with pytest.raises(NotImplementedError, match="mass-point"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_force_mass_point_on_d_lower_zero_sample_raises(self):
        """Review P1 round 4: Design 1 paths require d_lower > 0.

        Paper Section 3.2 reserves the d_lower=0 regime for Design 1'
        (continuous_at_zero). Forcing `mass_point` on a sample with
        d.min()==0 must raise, pointing the user to continuous_at_zero
        or auto.
        """
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", d_lower=0.0)
        with pytest.raises(ValueError, match=r"d_lower > 0|Design 1'"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_force_continuous_near_d_lower_on_d_lower_zero_sample_raises(self):
        """Parallel: continuous_near_d_lower must also reject d_lower=0."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="continuous_near_d_lower")
        # d_lower auto-resolves to float(d.min()) == 0.0 on this DGP.
        with pytest.raises(ValueError, match=r"d_lower > 0|Design 1'"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_force_mass_point_d_lower_none_on_zero_sample_raises(self):
        """d_lower=None on a d.min()==0 sample resolves to 0; must still raise."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", d_lower=None)
        with pytest.raises(ValueError, match=r"d_lower > 0"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_continuous_at_zero_with_nonzero_d_lower_raises(self):
        """Review P1 round 12: continuous_at_zero must reject nonzero d_lower.

        Paper Section 3.2 Design 1' is defined at d_lower = 0; silently
        coercing a user-supplied d_lower=0.5 to zero would contradict
        the documented regime contract.
        """
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", d_lower=0.5)
        with pytest.raises(ValueError, match=r"d_lower == 0|Design 1'"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_continuous_at_zero_with_small_d_lower_raises(self):
        """Even a small nonzero d_lower should raise."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", d_lower=0.01)
        with pytest.raises(ValueError, match=r"d_lower == 0|Design 1'"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_continuous_at_zero_with_zero_d_lower_succeeds(self):
        """d_lower=0.0 exactly is fine (redundant but allowed)."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", d_lower=0.0)
        r = est.fit(panel, "outcome", "dose", "period", "unit")
        assert r.d_lower == 0.0
        assert np.isfinite(r.att)

    def test_auto_on_zero_sample_ignores_user_d_lower(self):
        """design='auto' resolving to continuous_at_zero must ALSO reject
        an explicit nonzero d_lower, not silently drop it.
        """
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="auto", d_lower=0.5)
        with pytest.raises(ValueError, match=r"d_lower == 0|Design 1'"):
            est.fit(panel, "outcome", "dose", "period", "unit")


# =============================================================================
# Design 1 d_lower contract enforcement (mass-point + continuous_near_d_lower)
# =============================================================================


class TestDesign1DLowerContract:
    """Paper Sections 3.2.2-3.2.4: Design 1 estimators identify at the support
    infimum. Both mass_point and continuous_near_d_lower require
    ``d_lower == float(d.min())`` within float tolerance; mismatched
    overrides raise.
    """

    def test_mass_point_d_lower_above_min_raises(self):
        d, dy = _dgp_mass_point(500, seed=0)  # d.min() == 0.5
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", d_lower=0.6)
        with pytest.raises(ValueError, match="support infimum"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_mass_point_d_lower_below_min_raises(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", d_lower=0.3)
        with pytest.raises(ValueError, match="support infimum"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_mass_point_d_lower_matches_succeeds(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", d_lower=0.5)
        r = est.fit(panel, "outcome", "dose", "period", "unit")
        assert r.d_lower == 0.5
        assert np.isfinite(r.att)

    def test_mass_point_d_lower_none_auto_resolves_to_min(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", d_lower=None)
        r = est.fit(panel, "outcome", "dose", "period", "unit")
        assert abs(r.d_lower - float(d.min())) < 1e-14

    def test_mass_point_d_lower_within_tolerance_succeeds(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        d_lower_user = float(d.min()) + 1e-15
        est = HeterogeneousAdoptionDiD(design="mass_point", d_lower=d_lower_user)
        r = est.fit(panel, "outcome", "dose", "period", "unit")
        assert np.isfinite(r.att)

    def test_mass_point_d_lower_below_min_within_tolerance_snaps(self):
        """Review P1 round 8: tolerance-accepted d_lower = d.min() - ε must
        be SNAPPED to d.min() so the instrument Z = d > d_lower matches
        the exact-minimum case; otherwise mass-point units would fall
        into Z=1 and empty the control group.
        """
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        d_lower_below = float(d.min()) - 1e-15
        r_below = HeterogeneousAdoptionDiD(design="mass_point", d_lower=d_lower_below).fit(
            panel, "outcome", "dose", "period", "unit"
        )
        r_exact = HeterogeneousAdoptionDiD(design="mass_point", d_lower=float(d.min())).fit(
            panel, "outcome", "dose", "period", "unit"
        )
        # Behavior must be identical within ULP (the snap collapses them).
        assert r_below.att == r_exact.att
        assert r_below.se == r_exact.se
        assert r_below.n_mass_point == r_exact.n_mass_point
        assert r_below.n_above_d_lower == r_exact.n_above_d_lower

    def test_continuous_near_d_lower_above_within_tolerance_snaps(self):
        """Review P1 round 8: tolerance-accepted d_lower = d.min() + ε on
        continuous_near_d_lower must be SNAPPED so the regressor shift
        `d - d_lower` does not produce negative doses and trip Phase 1c's
        _validate_had_inputs negative-dose guard.
        """
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        panel = _make_panel(d, dy)
        d_lower_above = float(d.min()) + 1e-15
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_above = HeterogeneousAdoptionDiD(
                design="continuous_near_d_lower", d_lower=d_lower_above
            ).fit(panel, "outcome", "dose", "period", "unit")
            r_exact = HeterogeneousAdoptionDiD(
                design="continuous_near_d_lower", d_lower=float(d.min())
            ).fit(panel, "outcome", "dose", "period", "unit")
        assert r_above.att == r_exact.att
        assert r_above.se == r_exact.se

    def test_continuous_near_d_lower_above_min_raises(self):
        """Review P1: continuous_near_d_lower must also enforce support infimum."""
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="continuous_near_d_lower", d_lower=0.3)
        with pytest.raises(ValueError, match="support infimum"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_continuous_near_d_lower_below_min_raises(self):
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        panel = _make_panel(d, dy)
        # d.min() for this Beta DGP is > 0.1 but setting d_lower=0.05 is below min.
        est = HeterogeneousAdoptionDiD(design="continuous_near_d_lower", d_lower=0.05)
        with pytest.raises(ValueError, match="support infimum"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_continuous_near_d_lower_matches_succeeds(self):
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            est = HeterogeneousAdoptionDiD(design="continuous_near_d_lower", d_lower=float(d.min()))
            r = est.fit(panel, "outcome", "dose", "period", "unit")
        assert np.isfinite(r.att)


# =============================================================================
# Post-period dose non-negative contract (review P1)
# =============================================================================


class TestPostPeriodDoseContract:
    """Paper Section 2 dose definition: D_{g,2} >= 0. _validate_had_panel
    rejects negative post-period dose front-door on the ORIGINAL scale
    (before the regressor shift) so the error references the user's
    dose column, not the Phase 1c shifted values.
    """

    def test_negative_post_dose_raises(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        # Inject a negative post-period dose on one unit.
        post_mask = panel["period"] == 2
        idx = panel[post_mask].index[0]
        panel.loc[idx, "dose"] = -0.1
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match=r"D_\{g,2\}|negative post"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_zero_post_dose_accepted(self):
        """D_{g,2} == 0 is the Design 1' no-treated-group case, always allowed."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        # Ensure d[0] == 0 exactly (no-treated unit) is accepted.
        assert d[0] == 0.0
        panel = _make_panel(d, dy)
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        assert np.isfinite(r.att)


# =============================================================================
# Design 1 Assumption 5/6 identification warning (review P1)
# =============================================================================


class TestAssumptionFiveSixWarning:
    """Paper Sections 3.2.2-3.2.4: Design 1 fits require Assumption 5 (sign
    identification) or Assumption 6 (point identification of WAS_{d_lower})
    beyond parallel trends. These extras are not pre-trend testable. A
    UserWarning surfaces the identification burden on Design 1 fits.
    """

    def test_continuous_near_d_lower_emits_assumption_warning(self):
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HeterogeneousAdoptionDiD(design="continuous_near_d_lower").fit(
                panel, "outcome", "dose", "period", "unit"
            )
            assumption_warnings = [warn for warn in w if "Assumption" in str(warn.message)]
            assert len(assumption_warnings) >= 1

    def test_mass_point_emits_assumption_warning(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HeterogeneousAdoptionDiD(design="mass_point").fit(
                panel, "outcome", "dose", "period", "unit"
            )
            assumption_warnings = [warn for warn in w if "Assumption" in str(warn.message)]
            assert len(assumption_warnings) >= 1

    def test_continuous_at_zero_does_not_emit_assumption_warning(self):
        """Design 1' (d_lower=0) is identified under Assumption 3 only; no warning."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
                panel, "outcome", "dose", "period", "unit"
            )
            assumption_warnings = [warn for warn in w if "Assumption 6" in str(warn.message)]
            assert len(assumption_warnings) == 0


# =============================================================================
# Cluster handling (unit-level aggregation)
# =============================================================================


class TestClusterHandling:
    def test_cluster_not_constant_within_unit_raises(self):
        d, dy = _dgp_mass_point(100, seed=0)
        panel = _make_panel(d, dy)
        # Make cluster vary within unit
        panel["state"] = np.arange(len(panel))
        est = HeterogeneousAdoptionDiD(design="mass_point", cluster="state")
        with pytest.raises(ValueError, match=r"constant within unit"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_missing_cluster_column_raises(self):
        d, dy = _dgp_mass_point(100, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point", cluster="missing")
        with pytest.raises(ValueError, match="cluster"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_nan_cluster_raises(self):
        d, dy = _dgp_mass_point(100, seed=0)
        # Unit-level cluster ids: 50 clusters, 2 units each, with NaN on unit 0.
        cluster_unit = np.repeat(np.arange(50).astype(float), 2)  # length 100
        cluster_unit[0] = np.nan
        panel = _make_panel(d, dy, extra_cols={"state": cluster_unit})
        est = HeterogeneousAdoptionDiD(design="mass_point", cluster="state")
        with pytest.raises(ValueError, match="NaN"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_mixed_row_nan_cluster_raises_on_mass_point(self):
        """Review P2 round 8: a unit with rows [valid, NaN] on mass-point
        must be rejected by row-level validation, not masked by
        `groupby().first()`.
        """
        d, dy = _dgp_mass_point(100, seed=0)
        cluster_unit = np.repeat(np.arange(50).astype(float), 2)  # all valid
        panel = _make_panel(d, dy, extra_cols={"state": cluster_unit})
        # Inject NaN only on the second row (t_post) of unit 0.
        unit0_post_idx = panel[(panel["unit"] == 0) & (panel["period"] == 2)].index[0]
        panel.loc[unit0_post_idx, "state"] = np.nan
        est = HeterogeneousAdoptionDiD(design="mass_point", cluster="state")
        with pytest.raises(ValueError, match="NaN"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_cluster_warns_on_continuous_path(self):
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        cluster_unit = np.repeat(np.arange(100), 2)  # length 200 unit-level
        panel = _make_panel(d, dy, extra_cols={"state": cluster_unit})
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", cluster="state")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = est.fit(panel, "outcome", "dose", "period", "unit")
            assert any("cluster" in str(warn.message).lower() for warn in w)
        assert np.isfinite(r.att)

    def test_cluster_name_populated_mass_point(self):
        d, dy = _dgp_mass_point(200, seed=0)
        cluster_unit = np.repeat(np.arange(50), 4)  # 50 clusters, 4 units each
        panel = _make_panel(d, dy, extra_cols={"state": cluster_unit})
        r = HeterogeneousAdoptionDiD(design="mass_point", cluster="state").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        assert r.cluster_name == "state"

    def test_cluster_name_none_without_cluster(self):
        d, dy = _dgp_mass_point(200, seed=0)
        r = HeterogeneousAdoptionDiD(design="mass_point").fit(
            _make_panel(d, dy), "outcome", "dose", "period", "unit"
        )
        assert r.cluster_name is None

    def test_missing_cluster_column_on_continuous_only_warns(self):
        """Review P1 round 7: irrelevant cluster on continuous path must not
        abort the fit. The cluster column doesn't even need to exist.
        """
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", cluster="does_not_exist")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = est.fit(panel, "outcome", "dose", "period", "unit")
            assert any("cluster" in str(warn.message).lower() for warn in w)
        assert np.isfinite(r.att)
        assert r.cluster_name is None

    def test_nan_cluster_on_continuous_only_warns(self):
        """NaN cluster IDs on continuous path must not abort the fit."""
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        cluster_unit = np.repeat(np.arange(100).astype(float), 2)
        cluster_unit[0] = np.nan
        panel = _make_panel(d, dy, extra_cols={"state": cluster_unit})
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", cluster="state")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = est.fit(panel, "outcome", "dose", "period", "unit")
            assert any("cluster" in str(warn.message).lower() for warn in w)
        assert np.isfinite(r.att)

    def test_within_unit_varying_cluster_on_continuous_only_warns(self):
        """Within-unit-varying cluster IDs on continuous path must not abort."""
        d, dy = _dgp_continuous_at_zero(200, seed=0)
        panel = _make_panel(d, dy)
        # Varies within unit (distinct value per row)
        panel["state"] = np.arange(len(panel))
        est = HeterogeneousAdoptionDiD(design="continuous_at_zero", cluster="state")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = est.fit(panel, "outcome", "dose", "period", "unit")
            assert any("cluster" in str(warn.message).lower() for warn in w)
        assert np.isfinite(r.att)

    def test_auto_design_ignores_irrelevant_cluster_on_continuous(self):
        """design='auto' resolving to a continuous path must also ignore cluster."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="auto", cluster="does_not_exist")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = est.fit(panel, "outcome", "dose", "period", "unit")
            assert any("cluster" in str(warn.message).lower() for warn in w)
        assert r.design == "continuous_at_zero"
        assert np.isfinite(r.att)


# =============================================================================
# First-difference aggregation helper
# =============================================================================


class TestFirstDifferenceAggregation:
    def test_aggregate_returns_sorted_unit_order(self):
        d, dy = _dgp_continuous_at_zero(100, seed=0)
        panel = _make_panel(d, dy)
        # Shuffle rows to test sort-invariance
        panel_shuffled = panel.sample(frac=1, random_state=42).reset_index(drop=True)
        d_arr, dy_arr, _, unit_ids = _aggregate_first_difference(
            panel_shuffled, "outcome", "dose", "period", "unit", 1, 2, None
        )
        # unit_ids sorted
        assert np.all(np.diff(unit_ids) >= 0)
        # Each dose matches the input dose for its unit
        for i, uid in enumerate(unit_ids):
            assert abs(d_arr[i] - d[uid]) < 1e-14
            assert abs(dy_arr[i] - dy[uid]) < 1e-14

    def test_aggregate_cluster_array_correct(self):
        d, dy = _dgp_mass_point(100, seed=0)
        cluster_unit = np.repeat(np.arange(25), 4)  # 25 clusters, 4 units each
        panel = _make_panel(d, dy, extra_cols={"state": cluster_unit})
        _, _, cluster_arr, unit_ids = _aggregate_first_difference(
            panel,
            "outcome",
            "dose",
            "period",
            "unit",
            1,
            2,
            "state",
        )
        assert cluster_arr is not None
        assert len(cluster_arr) == 100
        # Cluster_arr[i] should equal cluster_unit[unit_ids[i]]
        for i, uid in enumerate(unit_ids):
            assert cluster_arr[i] == cluster_unit[uid]

    def test_aggregate_no_cluster_returns_none(self):
        d, dy = _dgp_continuous_at_zero(50, seed=0)
        panel = _make_panel(d, dy)
        _, _, cluster_arr, _ = _aggregate_first_difference(
            panel, "outcome", "dose", "period", "unit", 1, 2, None
        )
        assert cluster_arr is None


# =============================================================================
# Auto-detect mass-point vs continuous-near at boundary
# =============================================================================


class TestAutoDetectEdges:
    def test_exactly_two_percent_modal_is_not_mass_point(self):
        """Threshold is strict >, not >=. 2% exactly should stay continuous."""
        rng = np.random.default_rng(0)
        G = 1000
        mass_n = 20  # exactly 2%
        d = np.concatenate([np.full(mass_n, 0.5), rng.uniform(0.5001, 1.0, G - mass_n)])
        # d.min() == 0.5, not 0, and modal fraction == 2% (not > 2%)
        assert _detect_design(d) == "continuous_near_d_lower"

    def test_slightly_over_two_percent_is_mass_point(self):
        rng = np.random.default_rng(0)
        G = 1000
        mass_n = 25  # 2.5%
        d = np.concatenate([np.full(mass_n, 0.5), rng.uniform(0.5001, 1.0, G - mass_n)])
        assert _detect_design(d) == "mass_point"

    def test_all_at_zero_resolves_continuous_at_zero(self):
        """Degenerate but well-defined: all zeros -> continuous_at_zero."""
        d = np.zeros(100)
        assert _detect_design(d) == "continuous_at_zero"


# =============================================================================
# Panel validator direct tests
# =============================================================================


class TestValidateHadPanel:
    def test_returns_period_pair(self):
        d, dy = _dgp_continuous_at_zero(100, seed=0)
        panel = _make_panel(d, dy, periods=(2020, 2021))
        t_pre, t_post = _validate_had_panel(panel, "outcome", "dose", "period", "unit", None)
        assert t_pre == 2020
        assert t_post == 2021

    def test_rejects_string_periods_gracefully(self):
        """String periods should still sort and validate."""
        d, dy = _dgp_continuous_at_zero(100, seed=0)
        panel = _make_panel(d, dy, periods=("A", "B"))
        # Should not raise - strings sort fine
        t_pre, t_post = _validate_had_panel(panel, "outcome", "dose", "period", "unit", None)
        assert t_pre == "A"
        assert t_post == "B"

    def test_first_treat_col_with_string_periods(self):
        """Review P1: first_treat_col validator must be dtype-agnostic.

        With string periods ("A", "B") and first_treat_col values in
        {0, "B"}, the validator must not attempt numeric coercion.
        """
        d, dy = _dgp_continuous_at_zero(100, seed=0)
        panel = _make_panel(d, dy, periods=("A", "B"))
        # 50 units never-treated (first_treat=0), 50 treated (first_treat="B")
        ft_unit = np.array([0 if i % 2 == 0 else "B" for i in range(100)], dtype=object)
        panel["ft"] = np.repeat(ft_unit, 2)
        t_pre, t_post = _validate_had_panel(panel, "outcome", "dose", "period", "unit", "ft")
        assert t_pre == "A"
        assert t_post == "B"

    def test_first_treat_col_dtype_agnostic_rejects_invalid_string(self):
        """Mix string periods + invalid first_treat_col string -> ValueError."""
        d, dy = _dgp_continuous_at_zero(100, seed=0)
        panel = _make_panel(d, dy, periods=("A", "B"))
        # Invalid: "Z" is neither 0 nor "B"
        ft_unit = np.array([0 if i % 2 == 0 else "Z" for i in range(100)], dtype=object)
        panel["ft"] = np.repeat(ft_unit, 2)
        with pytest.raises(ValueError, match="first_treat_col"):
            _validate_had_panel(panel, "outcome", "dose", "period", "unit", "ft")

    def test_semantic_pre_post_labels_not_lexicographic(self):
        """Review P1 round 3: pre/post inference must be dose-based.

        ("pre", "post") sorts alphabetically to ["post", "pre"], which
        previously flipped the pre/post labels and raised on a valid
        panel. The validator now infers pre from the all-zero-dose
        period.
        """
        d, dy = _dgp_continuous_at_zero(100, seed=0)
        panel = _make_panel(d, dy, periods=("pre", "post"))
        t_pre, t_post = _validate_had_panel(panel, "outcome", "dose", "period", "unit", None)
        assert t_pre == "pre"
        assert t_post == "post"

    def test_semantic_pre_post_with_first_treat_col(self):
        """Combined: string periods + first_treat_col in {0, 'post'}."""
        d, dy = _dgp_continuous_at_zero(100, seed=0)
        panel = _make_panel(d, dy, periods=("pre", "post"))
        ft_unit = np.array([0 if i % 2 == 0 else "post" for i in range(100)], dtype=object)
        panel["ft"] = np.repeat(ft_unit, 2)
        t_pre, t_post = _validate_had_panel(panel, "outcome", "dose", "period", "unit", "ft")
        assert t_pre == "pre"
        assert t_post == "post"

    def test_semantic_pre_post_fit_end_to_end(self):
        """End-to-end: fit() runs on ("pre","post")-labelled panel."""
        d, dy = _dgp_continuous_at_zero(500, seed=0)
        panel = _make_panel(d, dy, periods=("pre", "post"))
        r = HeterogeneousAdoptionDiD(design="continuous_at_zero").fit(
            panel, "outcome", "dose", "period", "unit"
        )
        assert np.isfinite(r.att)

    def test_before_after_labels(self):
        """("before","after") is also reversed alphabetically; must not fail."""
        d, dy = _dgp_continuous_at_zero(100, seed=0)
        panel = _make_panel(d, dy, periods=("before", "after"))
        t_pre, t_post = _validate_had_panel(panel, "outcome", "dose", "period", "unit", None)
        assert t_pre == "before"
        assert t_post == "after"

    def test_no_all_zero_period_raises(self):
        """If neither period has all-zero dose, HAD's D_{g,1}=0 contract fails."""
        d, dy = _dgp_continuous_at_zero(100, seed=0)
        panel = _make_panel(d, dy)
        # Inject nonzero dose into the pre period so neither period is all-zero.
        panel.loc[panel["period"] == 1, "dose"] = 0.5
        with pytest.raises(ValueError, match=r"D_\{g,1\}|pre-treatment"):
            _validate_had_panel(panel, "outcome", "dose", "period", "unit", None)

    def test_both_all_zero_periods_raises(self):
        """If both periods have all-zero dose, no treatment to estimate."""
        G = 100
        panel = pd.DataFrame(
            {
                "unit": np.repeat(np.arange(G), 2),
                "period": np.tile([1, 2], G),
                "dose": np.zeros(2 * G),
                "outcome": np.random.default_rng(0).standard_normal(2 * G),
            }
        )
        with pytest.raises(ValueError, match="variation"):
            _validate_had_panel(panel, "outcome", "dose", "period", "unit", None)

    def test_repeated_cross_section_raises(self):
        """Review P1 round 6: Phase 2a is panel-only. An RCS input (disjoint
        unit IDs across periods) must be rejected by the balanced-panel
        validator with the "unit(s) do not appear in both periods" error.
        """
        rng = np.random.default_rng(0)
        G = 100
        pre = pd.DataFrame(
            {
                "unit": np.arange(G),
                "period": 1,
                "dose": np.zeros(G),
                "outcome": rng.standard_normal(G),
            }
        )
        post = pd.DataFrame(
            {
                "unit": np.arange(G, 2 * G),
                "period": 2,
                "dose": rng.uniform(0, 1, G),
                "outcome": rng.standard_normal(G),
            }
        )
        rcs = pd.concat([pre, post], ignore_index=True)
        with pytest.raises(ValueError, match=r"both periods|[Uu]nbalanced"):
            _validate_had_panel(rcs, "outcome", "dose", "period", "unit", None)

    def test_repeated_cross_section_fit_raises(self):
        """End-to-end: fit() on an RCS panel raises ValueError."""
        rng = np.random.default_rng(0)
        G = 100
        pre = pd.DataFrame(
            {
                "unit": np.arange(G),
                "period": 1,
                "dose": np.zeros(G),
                "outcome": rng.standard_normal(G),
            }
        )
        post = pd.DataFrame(
            {
                "unit": np.arange(G, 2 * G),
                "period": 2,
                "dose": rng.uniform(0, 1, G),
                "outcome": rng.standard_normal(G),
            }
        )
        rcs = pd.concat([pre, post], ignore_index=True)
        est = HeterogeneousAdoptionDiD()
        with pytest.raises(ValueError, match=r"both periods|[Uu]nbalanced"):
            est.fit(rcs, "outcome", "dose", "period", "unit")


# =============================================================================
# Review P1: continuous_near_d_lower on a true mass-point sample rejects
# =============================================================================


class TestContinuousPathRejectsMassPoint:
    """Explicit override to continuous_near_d_lower on a mass-point sample
    must raise before the regressor shift, otherwise the Phase 1c
    mass-point guard (which fires only on d.min() > 0) is bypassed.
    """

    def test_continuous_near_on_mass_point_sample_raises(self):
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="continuous_near_d_lower")
        with pytest.raises(ValueError, match=r"mass-point sample|mass_point"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_continuous_near_on_continuous_sample_runs(self):
        """Sanity: the pre-shift check does NOT reject valid continuous samples."""
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = HeterogeneousAdoptionDiD(design="continuous_near_d_lower").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        assert np.isfinite(r.att)


class TestMassPointPathRejectsContinuousSample:
    """Review P1 round 5: reciprocal guard. Forcing design="mass_point" on a
    continuous-near-d_lower sample (modal fraction at d.min() <= 2%) must
    raise, otherwise 2SLS identifies the exact-d.min() cell rather than
    the paper's boundary-limit estimand.
    """

    def test_mass_point_on_continuous_near_sample_raises(self):
        d, dy = _dgp_continuous_near_d_lower(500, seed=0)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point")
        with pytest.raises(ValueError, match=r"modal mass|2SLS.*continuous"):
            est.fit(panel, "outcome", "dose", "period", "unit")

    def test_mass_point_on_true_mass_point_sample_runs(self):
        """Sanity: the reciprocal guard does NOT reject valid mass-point samples."""
        d, dy = _dgp_mass_point(500, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = HeterogeneousAdoptionDiD(design="mass_point").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        assert np.isfinite(r.att)

    def test_mass_point_modal_at_threshold_runs(self):
        """At exactly 2% + 1 unit, mass_point runs (strict > 0.02)."""
        rng = np.random.default_rng(0)
        G = 1000
        mass_n = 25  # 2.5% > threshold
        d = np.concatenate([np.full(mass_n, 0.5), rng.uniform(0.5001, 1.0, G - mass_n)])
        dy = 0.3 * d + 0.1 * rng.standard_normal(G)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = HeterogeneousAdoptionDiD(design="mass_point").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        assert np.isfinite(r.att)

    def test_mass_point_modal_exactly_two_percent_raises(self):
        """At exactly 2% (not strictly greater), mass_point must raise."""
        rng = np.random.default_rng(0)
        G = 1000
        mass_n = 20  # exactly 2% (not > 2%)
        d = np.concatenate([np.full(mass_n, 0.5), rng.uniform(0.5001, 1.0, G - mass_n)])
        dy = 0.3 * d + 0.1 * rng.standard_normal(G)
        panel = _make_panel(d, dy)
        est = HeterogeneousAdoptionDiD(design="mass_point")
        with pytest.raises(ValueError, match=r"modal mass"):
            est.fit(panel, "outcome", "dose", "period", "unit")


# =============================================================================
# Review P2: cluster-applied mass-point stores vcov_type="cr1"
# =============================================================================


class TestMassPointClusterLabel:
    def test_cluster_stores_cr1(self):
        """to_dict() / downstream consumers see 'cr1' not 'hc1' when clustered."""
        d, dy = _dgp_mass_point(200, seed=0)
        cluster_unit = np.repeat(np.arange(50), 4)
        panel = _make_panel(d, dy, extra_cols={"state": cluster_unit})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = HeterogeneousAdoptionDiD(design="mass_point", cluster="state").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        assert r.vcov_type == "cr1"
        assert r.cluster_name == "state"

    def test_no_cluster_stores_base_family(self):
        """Unclustered mass-point keeps 'hc1' or 'classical' label."""
        d, dy = _dgp_mass_point(200, seed=0)
        panel = _make_panel(d, dy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r_hc1 = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="hc1").fit(
                panel, "outcome", "dose", "period", "unit"
            )
            r_cl = HeterogeneousAdoptionDiD(design="mass_point", vcov_type="classical").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        assert r_hc1.vcov_type == "hc1"
        assert r_cl.vcov_type == "classical"

    def test_cluster_with_classical_collapses_to_cr1(self):
        """classical + cluster is CR1 in practice; label reflects that."""
        d, dy = _dgp_mass_point(200, seed=0)
        cluster_unit = np.repeat(np.arange(50), 4)
        panel = _make_panel(d, dy, extra_cols={"state": cluster_unit})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = HeterogeneousAdoptionDiD(
                design="mass_point", vcov_type="classical", cluster="state"
            ).fit(panel, "outcome", "dose", "period", "unit")
        assert r.vcov_type == "cr1"

    def test_to_dict_shows_effective_family(self):
        d, dy = _dgp_mass_point(200, seed=0)
        cluster_unit = np.repeat(np.arange(50), 4)
        panel = _make_panel(d, dy, extra_cols={"state": cluster_unit})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = HeterogeneousAdoptionDiD(design="mass_point", cluster="state").fit(
                panel, "outcome", "dose", "period", "unit"
            )
        result_dict = r.to_dict()
        assert result_dict["vcov_type"] == "cr1"
        assert result_dict["cluster_name"] == "state"
