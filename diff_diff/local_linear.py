"""
Kernels and univariate local-linear regression at a boundary.

Ships the foundational RDD infrastructure that downstream estimators compose:

- Bounded one-sided kernels (Epanechnikov, triangular, uniform) on ``[0, 1]``
  suitable for boundary-point nonparametric regression.
- Closed-form kernel-moment constants ``kappa_k := int_0^1 t^k * k(t) dt`` and the
  derived boundary-kernel constant ``C = (kappa_2^2 - kappa_1 kappa_3) /
  (kappa_0 kappa_2 - kappa_1^2)`` that appears in the asymptotic bias of
  local-linear at a boundary.
- A univariate local-linear regression fitter ``local_linear_fit`` that estimates
  the conditional mean ``m(d0) := E[Y | D = d0]`` at the boundary of ``D``'s
  support via kernel-weighted OLS.

This module is used by the :class:`HeterogeneousAdoptionDiD` phases:

- Phase 1a ships the kernels and fitter (this module).
- Phase 1b ships the MSE-optimal bandwidth selector
  ``mse_optimal_bandwidth`` / ``BandwidthResult`` (this module), a thin
  wrapper over the Calonico-Cattaneo-Farrell (2018) plug-in selector
  ported in ``diff_diff/_nprobust_port.py``.
- Phase 1c will add the bias-corrected confidence interval per Equation 8 of
  de Chaisemartin, Ciccia, D'Haultfoeuille & Knau (2026, arXiv:2405.04465v6).

References
----------
- de Chaisemartin, C., Ciccia, D., D'Haultfoeuille, X., & Knau, F. (2026).
  Difference-in-Differences Estimators When No Unit Remains Untreated.
  arXiv:2405.04465v6. Section 3.1.3 defines the kernel-moment constants used
  here.
- Calonico, S., Cattaneo, M. D., & Farrell, M. H. (2018). On the effect of bias
  estimation on coverage accuracy in nonparametric inference. Journal of the
  American Statistical Association, 113(522), 767-779.
- Fan, J., & Gijbels, I. (1996). Local Polynomial Modelling and Its
  Applications. Chapman & Hall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from scipy import integrate

from diff_diff.linalg import solve_ols

__all__ = [
    "BandwidthResult",
    "epanechnikov_kernel",
    "triangular_kernel",
    "uniform_kernel",
    "KERNELS",
    "kernel_moments",
    "LocalLinearFit",
    "local_linear_fit",
    "mse_optimal_bandwidth",
]


# =============================================================================
# Kernel functions
# =============================================================================
#
# Each kernel is defined on [0, 1] (one-sided, for boundary estimation where
# the running variable D is supported on [0, infinity) and the evaluation point
# is at d0 = 0). Kernels return 0 outside [0, 1].


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel on ``[0, 1]``.

    ``k(u) = (3/4)(1 - u^2)`` for ``u in [0, 1]``, zero elsewhere.

    Parameters
    ----------
    u : np.ndarray
        Points on the scaled domain ``u = (d - d0) / h``.

    Returns
    -------
    np.ndarray
        Kernel values, same shape as ``u``.
    """
    u = np.asarray(u, dtype=np.float64)
    inside = (u >= 0.0) & (u <= 1.0)
    return np.where(inside, 0.75 * (1.0 - u * u), 0.0)


def triangular_kernel(u: np.ndarray) -> np.ndarray:
    """Triangular kernel on ``[0, 1]``.

    ``k(u) = 1 - u`` for ``u in [0, 1]``, zero elsewhere.

    Using the convention ``int_0^1 k(u) du = 1/2`` to match Epanechnikov's
    one-sided normalization.
    """
    u = np.asarray(u, dtype=np.float64)
    inside = (u >= 0.0) & (u <= 1.0)
    return np.where(inside, 1.0 - u, 0.0)


def uniform_kernel(u: np.ndarray) -> np.ndarray:
    """Uniform (rectangular) kernel on ``[0, 1]``.

    ``k(u) = 1`` for ``u in [0, 1]``, zero elsewhere.

    Already normalized to ``int_0^1 k(u) du = 1`` (no factor of 1/2 like the
    other two).
    """
    u = np.asarray(u, dtype=np.float64)
    inside = (u >= 0.0) & (u <= 1.0)
    return np.where(inside, 1.0, 0.0)


KERNELS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "epanechnikov": epanechnikov_kernel,
    "triangular": triangular_kernel,
    "uniform": uniform_kernel,
}


# =============================================================================
# Closed-form kernel moments
# =============================================================================
#
# For each kernel k(t) on [0, 1], the moments
#
#     kappa_k := int_0^1 t^k * k(t) dt
#
# admit closed forms. These are the values from elementary integration; the
# test suite verifies each against a numerical scipy.integrate.quad call.
#
# The derived constant C from the paper's Section 3.1.3 is
#
#     C = (kappa_2^2 - kappa_1 * kappa_3) / (kappa_0 * kappa_2 - kappa_1^2)
#
# and can be negative depending on kernel shape (e.g. Epanechnikov C < 0).


_CLOSED_FORM_MOMENTS: Dict[str, Dict[str, float]] = {
    "epanechnikov": {
        "kappa_0": 1.0 / 2.0,
        "kappa_1": 3.0 / 16.0,
        "kappa_2": 1.0 / 10.0,
        "kappa_3": 1.0 / 16.0,
        "kappa_4": 3.0 / 70.0,
    },
    "triangular": {
        "kappa_0": 1.0 / 2.0,
        "kappa_1": 1.0 / 6.0,
        "kappa_2": 1.0 / 12.0,
        "kappa_3": 1.0 / 20.0,
        "kappa_4": 1.0 / 30.0,
    },
    "uniform": {
        "kappa_0": 1.0,
        "kappa_1": 1.0 / 2.0,
        "kappa_2": 1.0 / 3.0,
        "kappa_3": 1.0 / 4.0,
        "kappa_4": 1.0 / 5.0,
    },
}


def kernel_moments(kernel: str = "epanechnikov") -> Dict[str, float]:
    """Return kernel-moment constants used in boundary local-linear asymptotics.

    The returned dict contains five raw moments ``kappa_k`` for
    ``k in {0, 1, 2, 3, 4}``, plus two derived constants:

    - ``"C"``: the paper's boundary-kernel constant used in the asymptotic
      bias term ``h^2 * C * m''(0)``. Per de Chaisemartin, Ciccia,
      D'Haultfoeuille & Knau (2026, Section 3.1.3),
      ``C = (kappa_2^2 - kappa_1 * kappa_3) / (kappa_0 * kappa_2 - kappa_1^2)``.
    - ``"kstar_L2_norm"``: the asymptotic-variance constant
      ``int_0^1 k*(t)^2 dt`` where
      ``k*(t) = (kappa_2 - kappa_1 * t) / (kappa_0 * kappa_2 - kappa_1^2) * k(t)``
      is the equivalent kernel for local-linear at a boundary. Computed by
      numerical integration.

    Parameters
    ----------
    kernel : str
        One of ``"epanechnikov"``, ``"triangular"``, ``"uniform"``.

    Returns
    -------
    dict of {str: float}
        Keys ``kappa_0``, ``kappa_1``, ``kappa_2``, ``kappa_3``, ``kappa_4``,
        ``C``, ``kstar_L2_norm``.

    Raises
    ------
    ValueError
        If ``kernel`` is not a recognized name.
    """
    if kernel not in _CLOSED_FORM_MOMENTS:
        raise ValueError(
            f"Unknown kernel {kernel!r}. Expected one of " f"{sorted(_CLOSED_FORM_MOMENTS.keys())}."
        )

    kappas = dict(_CLOSED_FORM_MOMENTS[kernel])

    k0, k1, k2, k3 = kappas["kappa_0"], kappas["kappa_1"], kappas["kappa_2"], kappas["kappa_3"]
    denom = k0 * k2 - k1 * k1
    C = (k2 * k2 - k1 * k3) / denom
    kappas["C"] = C

    kfun = KERNELS[kernel]

    def _kstar_sq(t: float) -> float:
        kt = kfun(np.array([t]))[0]
        return ((k2 - k1 * t) / denom) ** 2 * kt * kt

    val, _ = integrate.quad(_kstar_sq, 0.0, 1.0, limit=200)
    kappas["kstar_L2_norm"] = float(val)

    return kappas


# =============================================================================
# Local-linear regression at a boundary
# =============================================================================


@dataclass
class LocalLinearFit:
    """Result of a local-linear regression at a boundary.

    Attributes
    ----------
    intercept : float
        Estimated conditional mean at the boundary, ``mu_hat_h(d0)``.
    slope : float
        Estimated slope of the local linear fit (coefficient on ``d - d0``).
    n_effective : int
        Count of observations with strictly positive kernel weight (within
        ``[d0, d0 + h]`` for the one-sided kernels shipped here).
    bandwidth : float
        Bandwidth ``h`` used.
    kernel : str
        Kernel name.
    boundary : float
        Evaluation point ``d0``.
    residuals : np.ndarray, shape (n_effective,)
        Residuals from the weighted OLS fit, in the order of the retained
        observations.
    kernel_weights : np.ndarray, shape (n_effective,)
        Kernel weights ``k((d_i - d0) / h)``. These are the pre-scaled weights;
        the ``1/h`` scaling cancels out of the weighted-OLS estimator (a
        constant factor on all weights does not change the point estimate).
    design_matrix : np.ndarray, shape (n_effective, 2)
        Design matrix ``X = [1, d_i - d0]`` used in the fit. Preserved for
        Phase 1c bias-correction machinery.
    """

    intercept: float
    slope: float
    n_effective: int
    bandwidth: float
    kernel: str
    boundary: float
    residuals: np.ndarray
    kernel_weights: np.ndarray
    design_matrix: np.ndarray


def local_linear_fit(
    d: np.ndarray,
    y: np.ndarray,
    bandwidth: float,
    boundary: float = 0.0,
    kernel: str = "epanechnikov",
    weights: Optional[np.ndarray] = None,
) -> LocalLinearFit:
    """Local-linear regression of ``y`` on ``d`` at a boundary.

    Fits ``y ~ a + b * (d - boundary)`` using kernel weights
    ``k((d - boundary) / h)`` on observations with ``d in [boundary,
    boundary + h]``. Returns the intercept (the boundary estimate ``mu_hat``)
    and slope.

    Parameters
    ----------
    d : np.ndarray, shape (n,)
        Regressor values. For the HAD application, ``d`` is the period-2 dose
        ``D_{g,2}`` and the boundary is 0.
    y : np.ndarray, shape (n,)
        Outcome values. For the HAD application, ``y`` is the first-difference
        ``Delta Y_g``.
    bandwidth : float
        Bandwidth ``h > 0``.
    boundary : float, default=0.0
        Evaluation point ``d0``. Observations with ``d < d0`` are excluded
        (one-sided boundary estimation).
    kernel : str, default="epanechnikov"
        One of ``"epanechnikov"``, ``"triangular"``, ``"uniform"``.
    weights : np.ndarray or None, optional
        Optional per-observation weights ``w_i >= 0`` multiplied into the
        kernel weights. Useful for survey weighting; when ``None``, treated as
        unit weights.

    Returns
    -------
    LocalLinearFit
        Named container with ``intercept``, ``slope``, ``n_effective``, and
        diagnostics needed by downstream bias-correction phases.

    Raises
    ------
    ValueError
        If ``bandwidth <= 0``, ``kernel`` is unknown, ``d`` and ``y`` differ
        in length, or the bandwidth window retains fewer than 2 observations.
    """
    if bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be positive; got {bandwidth}")
    if kernel not in KERNELS:
        raise ValueError(f"Unknown kernel {kernel!r}. Expected one of {sorted(KERNELS.keys())}.")

    d = np.asarray(d, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if d.shape != y.shape:
        raise ValueError(f"d and y must have the same shape; got {d.shape} and {y.shape}")

    # Explicit NaN / Inf validation at the API boundary so the caller gets a
    # targeted error rather than a downstream failure inside the kernel or OLS.
    if not np.all(np.isfinite(d)):
        raise ValueError("d contains non-finite values (NaN or Inf)")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values (NaN or Inf)")

    if weights is None:
        user_w = np.ones_like(d)
    else:
        user_w = np.asarray(weights, dtype=np.float64).ravel()
        if user_w.shape != d.shape:
            raise ValueError(
                f"weights must have the same shape as d; got " f"{user_w.shape} vs {d.shape}"
            )
        if not np.all(np.isfinite(user_w)):
            raise ValueError("weights contains non-finite values (NaN or Inf)")
        if np.any(user_w < 0):
            raise ValueError("weights must be nonnegative")

    # Kernel weights on the scaled domain u = (d - d0) / h.
    kfun = KERNELS[kernel]
    u = (d - boundary) / bandwidth
    k_weights_full = kfun(u)

    # Compose with user weights and restrict to the bandwidth window.
    combined = k_weights_full * user_w
    retain = combined > 0.0
    n_effective = int(retain.sum())
    if n_effective < 2:
        raise ValueError(
            f"bandwidth window retained {n_effective} observation(s); "
            f"need at least 2. Widen the bandwidth or move the boundary."
        )

    d_in = d[retain]
    y_in = y[retain]
    w_in = combined[retain]
    k_in = k_weights_full[retain]

    design = np.column_stack([np.ones_like(d_in), d_in - boundary])

    # Weighted OLS via solve_ols. "aweight" treats the weights as analytic
    # frequency weights so the unweighted-OLS formulas apply with w-scaled X.
    # We only need the coefficients and residuals, not a vcov for the fit
    # itself (Phase 1c will build its own bias-aware variance).
    coef, residuals, _ = solve_ols(
        design,
        y_in,
        cluster_ids=None,
        return_vcov=False,
        weights=w_in,
        weight_type="aweight",
    )

    return LocalLinearFit(
        intercept=float(coef[0]),
        slope=float(coef[1]),
        n_effective=n_effective,
        bandwidth=float(bandwidth),
        kernel=kernel,
        boundary=float(boundary),
        residuals=np.asarray(residuals, dtype=np.float64),
        kernel_weights=np.asarray(k_in, dtype=np.float64),
        design_matrix=np.asarray(design, dtype=np.float64),
    )


# =============================================================================
# MSE-optimal bandwidth selector (Phase 1b)
# =============================================================================
#
# Public wrapper around diff_diff._nprobust_port.lpbwselect_mse_dpi. The port
# is a faithful Python translation of nprobust::lpbwselect(bwselect="mse-dpi")
# (R package nprobust 0.5.0, SHA 36e4e53); see diff_diff/_nprobust_port.py for
# source mapping.
#
# The public API here is a thin wrapper that:
# 1. Accepts the diff-diff library's kernel naming convention (full words) and
#    translates to nprobust's short codes ("epa", "tri", "uni").
# 2. Converts the internal MseDpiStages dataclass to the user-facing
#    BandwidthResult dataclass.
# 3. Enforces input validation consistent with local_linear_fit.
# 4. Rejects unsupported combinations (e.g. weights=) with NotImplementedError
#    (nprobust has no weight support).


@dataclass
class BandwidthResult:
    """MSE-optimal bandwidth selector output plus per-stage diagnostics.

    Returned by ``mse_optimal_bandwidth(..., return_diagnostics=True)``.
    Mirrors the five-bandwidth + four-stage structure of
    ``nprobust::lpbwselect.mse.dpi``; see
    ``diff_diff/_nprobust_port.py`` for the source mapping.

    Attributes
    ----------
    h_mse : float
        Final MSE-optimal bandwidth ``h*`` for local-linear estimation at
        ``boundary``. The argument to pass to
        ``local_linear_fit(..., bandwidth=h_mse)``.
    b_mse : float
        Bias-correction bandwidth. Consumed by Phase 1c for the
        bias-corrected confidence interval (CCF 2018 Equation 8).
    c_bw : float
        Stage 1 preliminary bandwidth used as ``h.V`` in every
        ``lprobust.bw`` call downstream:
        ``C_kernel * min(sd(d), IQR(d)/1.349) * G^{-1/5}``.
        Kernel constants: ``epa=2.34``, ``uni=1.843``, ``tri=2.576``.
    bw_mp2 : float
        Stage 2 pilot bandwidth for the ``m^{(q+1)}`` derivative estimator.
    bw_mp3 : float
        Stage 2 pilot bandwidth for the ``m^{(q+2)}`` derivative estimator.
    stage_d1_V, stage_d1_B1, stage_d1_B2, stage_d1_R : float
        Variance and bias coefficients from the first Stage-2
        ``lprobust.bw`` call (order ``q+1``, reading the ``(q+1)``-th
        derivative). Parity-checked to 1% against R.
    stage_d2_V, stage_d2_B1, stage_d2_B2, stage_d2_R : float
        Same for the second Stage-2 call (order ``q+2``).
    stage_b_V, stage_b_B1, stage_b_B2, stage_b_R : float
        Same for the Stage-3 bias-bandwidth call (order ``q``, nu ``p+1``).
    stage_h_V, stage_h_B1, stage_h_B2, stage_h_R : float
        Same for the final Stage-3 main-bandwidth call (order ``p``,
        nu ``deriv``).
    n : int
        Sample size.
    kernel : str
        Kernel name as user supplied it ("epanechnikov" / "triangular" /
        "uniform").
    boundary : float
        Evaluation point ``d_0``.
    """

    h_mse: float
    b_mse: float
    c_bw: float
    bw_mp2: float
    bw_mp3: float
    stage_d1_V: float
    stage_d1_B1: float
    stage_d1_B2: float
    stage_d1_R: float
    stage_d2_V: float
    stage_d2_B1: float
    stage_d2_B2: float
    stage_d2_R: float
    stage_b_V: float
    stage_b_B1: float
    stage_b_B2: float
    stage_b_R: float
    stage_h_V: float
    stage_h_B1: float
    stage_h_B2: float
    stage_h_R: float
    n: int
    kernel: str
    boundary: float


# Mapping between diff-diff's full kernel names and nprobust's short codes.
_KERNEL_NAME_TO_NPROBUST = {
    "epanechnikov": "epa",
    "triangular": "tri",
    "uniform": "uni",
}


def mse_optimal_bandwidth(
    d: np.ndarray,
    y: np.ndarray,
    boundary: float = 0.0,
    kernel: str = "epanechnikov",
    weights: Optional[np.ndarray] = None,
    bwcheck: Optional[int] = 21,
    bwregul: float = 1.0,
    return_diagnostics: bool = False,
):
    """MSE-optimal bandwidth for local-linear regression at a boundary.

    Port of ``nprobust::lpbwselect(bwselect="mse-dpi")`` (R package
    ``nprobust`` 0.5.0, SHA ``36e4e53``). Implements the Calonico,
    Cattaneo, and Farrell (2018, JASA 113(522)) direct-plug-in DPI
    bandwidth selector for the local-linear boundary estimator used in
    Design 1' of de Chaisemartin et al. (2026) (HAD).

    The three-stage algorithm produces five bandwidths (``c_bw``,
    ``bw_mp2``, ``bw_mp3``, ``b_mse``, ``h_mse``); see
    ``BandwidthResult`` for full diagnostics.

    **Public API scope (Phase 1b of HAD).** This wrapper is intentionally
    restricted to the HAD configuration: ``p=1`` (local-linear),
    ``deriv=0`` (mean regression), ``interior=False`` (boundary eval
    point), ``vce="nn"`` (nearest-neighbor variance), and ``nnmatch=3``
    are hard-coded. The underlying ``diff_diff._nprobust_port`` supports
    additional ``vce`` modes (``hc0`` / ``hc1`` / ``hc2`` / ``hc3``),
    interior evaluation, and higher polynomial orders, but those paths
    are NOT parity-tested against ``nprobust`` and are deferred to Phase
    1c / Phase 2. Do not rely on this public wrapper for anything
    outside HAD Phase 1b; use the port directly if you need the broader
    surface and accept that parity has not been separately verified.

    Parameters
    ----------
    d : np.ndarray, shape (G,)
        Regressor values (the dose ``D_{g,2}`` in HAD).
    y : np.ndarray, shape (G,)
        Outcome values (the first-difference ``Delta Y_g`` in HAD).
    boundary : float, default=0.0
        Evaluation point ``d_0``. The Phase 1b wrapper accepts only
        two values (within float tolerance): ``boundary = 0`` for
        Design 1' or ``boundary = float(d.min())`` for Design 1
        continuous-near-``d_lower``. Use the sample minimum
        ``d.min()`` (not a known theoretical lower bound of the
        support), because the downstream selector operates on the
        realized data. Any other value -- including
        ``boundary < d.min()``, interior points, or
        ``boundary > d.min()`` -- raises ``ValueError``.
    kernel : str, default="epanechnikov"
        One of ``"epanechnikov"``, ``"triangular"``, ``"uniform"``.
    weights : np.ndarray or None, default=None
        Not supported in Phase 1b (raises ``NotImplementedError``).
        ``nprobust::lpbwselect`` has no weight argument and thus no
        parity anchor. Weighted-data support is queued for Phase 2+.
    bwcheck : int or None, default=21
        If set, clip ``c_bw`` (and all downstream bandwidths) below the
        distance to the ``bwcheck``-th nearest neighbor of ``boundary``.
        Matches ``nprobust::lpbwselect`` default.
    bwregul : float, default=1.0
        Bias-regularization scale used in Stage 3. ``bwregul=0`` disables
        the regularization term; ``bwregul=1`` matches ``nprobust``
        default.
    return_diagnostics : bool, default=False
        When ``False`` (default) return the final bandwidth as a
        ``float``. When ``True`` return a ``BandwidthResult`` with all
        five stage bandwidths plus per-stage ``(V, B1, B2, R)``
        diagnostics.

    Returns
    -------
    float or BandwidthResult
        The MSE-optimal bandwidth ``h*``, or (if
        ``return_diagnostics=True``) a ``BandwidthResult`` dataclass.

    Raises
    ------
    ValueError
        Raised on: shape mismatch between ``d`` and ``y``; non-finite
        values in ``d``, ``y``, or ``boundary``; unknown ``kernel``
        name; ``bwcheck`` outside ``[1, len(d)]``; ``boundary`` that
        is not approximately 0 or approximately ``d.min()`` (the only
        two supported HAD estimands in Phase 1b); or a rank-deficient
        / under-determined pilot fit inside the DPI port (surfaced
        from ``qrXXinv`` or the per-stage count guards in
        ``lprobust_bw``).
    NotImplementedError
        Raised on: ``weights=`` passed (no nprobust parity anchor);
        detected Design 1 mass-point design (``d.min() > 0`` and
        modal fraction at ``d.min()`` exceeds 2%, per the paper's
        Section 3.2.4 redirection to the 2SLS sample-average
        estimator, queued for Phase 2).

    Notes
    -----
    The port parity-tests at 1% relative error against R
    ``nprobust::lpbwselect(bwselect="mse-dpi", vce="nn")`` on three
    deterministic DGPs (see
    ``benchmarks/R/generate_nprobust_golden.R``). In practice the
    agreement is at machine precision (0.0000% on the golden tests);
    the 1% tolerance is the hard gate before a deviation is considered
    a regression.
    """
    if weights is not None:
        raise NotImplementedError(
            "weights= is not supported in Phase 1b of the MSE-optimal "
            "bandwidth selector. nprobust::lpbwselect has no weight "
            "argument, so there is no parity anchor. Weighted-data "
            "support is queued for Phase 2+ (survey-design adaptation)."
        )

    if kernel not in _KERNEL_NAME_TO_NPROBUST:
        raise ValueError(
            f"Unknown kernel {kernel!r}. Expected one of "
            f"{sorted(_KERNEL_NAME_TO_NPROBUST.keys())}."
        )
    nprobust_kernel = _KERNEL_NAME_TO_NPROBUST[kernel]

    d = np.asarray(d, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if d.shape != y.shape:
        raise ValueError(f"d and y must have the same shape; got {d.shape} and {y.shape}")
    if not np.all(np.isfinite(d)):
        raise ValueError("d contains non-finite values (NaN or Inf)")
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values (NaN or Inf)")
    if not np.isfinite(boundary):
        raise ValueError(f"boundary must be finite; got {boundary}")

    # Boundary-applicability check (Phase 1b scope).
    # The exported wrapper is scoped to the two documented HAD
    # nonparametric estimands:
    #   - Design 1' evaluates m(0) = lim_{d down 0} E[Delta Y | D_2 <= d]
    #     with ``boundary = 0``.
    #   - Design 1 continuous-near-d_lower evaluates m(d_lower) with
    #     ``boundary = d_lower = min(D_2)``.
    # Any other off-support boundary (interior, upper-boundary, or an
    # arbitrary value between 0 and d.min()) would silently target an
    # undocumented limit and is rejected.
    d_min = float(d.min())
    _boundary_tol = 1e-12 * max(1.0, abs(d_min), abs(boundary))
    _at_zero = abs(boundary) <= _boundary_tol
    _at_d_min = abs(boundary - d_min) <= _boundary_tol
    if not (_at_zero or _at_d_min):
        raise ValueError(
            f"boundary={boundary!r} is not at a supported HAD estimand. "
            f"The Phase 1b public wrapper accepts only boundary ~ 0 "
            f"(Design 1') or boundary ~ d.min()={d_min!r} (Design 1 "
            f"continuous-near-d_lower). Off-support values would "
            f"silently target an undocumented limit. For interior or "
            f"other boundary points, use "
            f"diff_diff._nprobust_port.lpbwselect_mse_dpi directly and "
            f"note that those paths are not separately parity-tested."
        )

    # Mass-point design check (paper Section 3.2.4, REGISTRY 2% rule).
    # When d_min > 0 and there is bunching at d_min, Design 1 requires
    # the 2SLS sample-average path (Phase 2), not the CCF nonparametric
    # selector. The check applies independently of the boundary the
    # user supplied: mass-point data is never appropriate for this
    # wrapper. The check explicitly excludes d_min ~ 0, which is the
    # Design 1' "untreated units present" subcase that the paper's
    # simulations and the Garrett et al. (2020) application accept.
    _MASS_POINT_THRESHOLD = 0.02  # REGISTRY rule: > 2% modal-min
    if d_min > _boundary_tol:
        eps_eq = 1e-12 * max(1.0, abs(d_min))
        at_d_min_mask = np.abs(d - d_min) <= eps_eq
        modal_fraction = float(np.mean(at_d_min_mask))
        if modal_fraction > _MASS_POINT_THRESHOLD:
            raise NotImplementedError(
                f"Detected mass-point design at d.min()={d_min!r} "
                f"(modal fraction {modal_fraction:.4f} > "
                f"{_MASS_POINT_THRESHOLD:.2f}). Per de Chaisemartin et "
                f"al. (2026) Section 3.2.4, Design 1 mass-point cases "
                f"require the 2SLS sample-average estimator with "
                f"instrument 1{{D_2 > d_lower}}, not the CCF "
                f"nonparametric selector. That path is queued for "
                f"Phase 2 (HeterogeneousAdoptionDiD). For continuous "
                f"near-d_lower designs (modal fraction <= "
                f"{_MASS_POINT_THRESHOLD:.2f}), this wrapper is "
                f"applicable."
            )

    # Defer heavy import to call time to avoid import-cycle risk.
    from diff_diff._nprobust_port import lpbwselect_mse_dpi

    stages = lpbwselect_mse_dpi(
        y=y,
        x=d,
        cluster=None,
        eval_point=float(boundary),
        p=1,
        q=2,
        deriv=0,
        kernel=nprobust_kernel,
        bwcheck=bwcheck,
        bwregul=bwregul,
        vce="nn",
        nnmatch=3,
        interior=False,
    )

    if not return_diagnostics:
        return stages.h_mse_dpi

    return BandwidthResult(
        h_mse=stages.h_mse_dpi,
        b_mse=stages.b_mse_dpi,
        c_bw=stages.c_bw,
        bw_mp2=stages.bw_mp2,
        bw_mp3=stages.bw_mp3,
        stage_d1_V=stages.stage_d1.V,
        stage_d1_B1=stages.stage_d1.B1,
        stage_d1_B2=stages.stage_d1.B2,
        stage_d1_R=stages.stage_d1.R,
        stage_d2_V=stages.stage_d2.V,
        stage_d2_B1=stages.stage_d2.B1,
        stage_d2_B2=stages.stage_d2.B2,
        stage_d2_R=stages.stage_d2.R,
        stage_b_V=stages.stage_b.V,
        stage_b_B1=stages.stage_b.B1,
        stage_b_B2=stages.stage_b.B2,
        stage_b_R=stages.stage_b.R,
        stage_h_V=stages.stage_h.V,
        stage_h_B1=stages.stage_h.B1,
        stage_h_B2=stages.stage_h.B2,
        stage_h_R=stages.stage_h.R,
        n=int(d.shape[0]),
        kernel=kernel,
        boundary=float(boundary),
    )
