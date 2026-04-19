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

This module is used by future :class:`HeterogeneousAdoptionDiD` phases:

- Phase 1a ships the kernels and fitter (this module).
- Phase 1b will add an MSE-optimal bandwidth selector (Calonico-Cattaneo-Farrell
  2018) built on top of the fitter.
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
    "epanechnikov_kernel",
    "triangular_kernel",
    "uniform_kernel",
    "KERNELS",
    "kernel_moments",
    "LocalLinearFit",
    "local_linear_fit",
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
            f"Unknown kernel {kernel!r}. Expected one of "
            f"{sorted(_CLOSED_FORM_MOMENTS.keys())}."
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
        raise ValueError(
            f"Unknown kernel {kernel!r}. Expected one of {sorted(KERNELS.keys())}."
        )

    d = np.asarray(d, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if d.shape != y.shape:
        raise ValueError(
            f"d and y must have the same shape; got {d.shape} and {y.shape}"
        )

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
                f"weights must have the same shape as d; got "
                f"{user_w.shape} vs {d.shape}"
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
