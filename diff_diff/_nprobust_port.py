"""In-house port of nprobust's MSE-DPI bandwidth selector.

Faithful Python translation of the mse-dpi branch of
``nprobust::lpbwselect`` from the R package
``nprobust`` 0.5.0 (SHA ``36e4e532d2f7d23d4dc6e162575cca79e0927cda``,
``github.com/nppackages/nprobust``). This module implements Calonico,
Cattaneo, and Farrell (2018, JASA 113(522)) plug-in bandwidth selection
exactly as the reference R code does, so that per-stage bandwidths and
per-stage bias/variance components parity-check to 1% on deterministic
seeds (see ``benchmarks/R/generate_nprobust_golden.R`` and
``tests/test_nprobust_port.py``).

Source mapping (every public function here pairs with one R function):

========================================  =========================================
Python                                    R (npfunctions.R)
========================================  =========================================
``kernel_W(u, kernel)``                   ``W.fun`` (npfunctions.R:1-7)
``qrXXinv(x)``                            ``qrXXinv`` (npfunctions.R:89-93)
``_precompute_nn_duplicates(x)``          npfunctions.R:518-529 (inline in mse.dpi)
``lprobust_res(...)``                     ``lprobust.res`` (npfunctions.R:131-162)
``lprobust_vce(...)``                     ``lprobust.vce`` (npfunctions.R:165-185)
``LprobustBwResult``                      return list of ``lprobust.bw``
``lprobust_bw(...)``                      ``lprobust.bw`` (npfunctions.R:187-288)
``lpbwselect_mse_dpi(...)``               ``lpbwselect.mse.dpi`` (npfunctions.R:498-607)
========================================  =========================================

This module is nprobust-internal logic; the public wrapper is
``diff_diff.local_linear.mse_optimal_bandwidth``. Phase 1c's robust
variance and bias estimator will also compose these helpers.

Deviations from nprobust (documented):

* ``weights=`` is not supported here or in the public wrapper
  (nprobust's ``lpbwselect`` has no weight argument, so Phase 1b has
  no parity anchor). Weighted-data support is queued for Phase 2+
  (survey-design adaptation). The public wrapper
  ``mse_optimal_bandwidth`` raises ``NotImplementedError`` when a
  ``weights`` array is passed.
* ``vce="nn"`` is the default and is fully ported. ``vce in
  {"hc0", "hc1", "hc2", "hc3"}`` is implemented in ``lprobust_res`` /
  ``lprobust_vce`` but has not been separately golden-tested; use at
  your own risk until Phase 1c.
* ``cluster=`` is supported in ``lprobust_vce`` and the ``lprobust_bw``
  wrapper but is only exercised by the HAD estimator via Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import optimize
from scipy.stats import norm as _scipy_norm

__all__ = [
    "NPROBUST_VERSION",
    "NPROBUST_SHA",
    "LprobustBwResult",
    "kernel_W",
    "qrXXinv",
    "lprobust_res",
    "lprobust_vce",
    "lprobust_bw",
    "lpbwselect_mse_dpi",
]

NPROBUST_VERSION = "0.5.0"
NPROBUST_SHA = "36e4e532d2f7d23d4dc6e162575cca79e0927cda"

_VALID_KERNELS = ("epa", "uni", "tri", "gau")
_VALID_VCE = ("nn", "hc0", "hc1", "hc2", "hc3")


# =============================================================================
# Kernel (W.fun, npfunctions.R:1-7)
# =============================================================================


def kernel_W(u: np.ndarray, kernel: str) -> np.ndarray:
    """Symmetric kernel evaluation matching ``nprobust::W.fun``.

    Parameters
    ----------
    u : np.ndarray
        Scaled argument ``(x - c) / h``.
    kernel : str
        One of "epa", "uni", "tri", "gau".

    Returns
    -------
    np.ndarray
        Kernel values, same shape as ``u``. Zero where ``|u| > 1`` for
        compact-support kernels.
    """
    u = np.asarray(u, dtype=np.float64)
    if kernel == "epa":
        return np.where(np.abs(u) <= 1.0, 0.75 * (1.0 - u * u), 0.0)
    if kernel == "uni":
        return np.where(np.abs(u) <= 1.0, 0.5, 0.0)
    if kernel == "tri":
        return np.where(np.abs(u) <= 1.0, 1.0 - np.abs(u), 0.0)
    if kernel == "gau":
        return _scipy_norm.pdf(u)
    raise ValueError(f"Unknown kernel {kernel!r}. Expected one of {_VALID_KERNELS}.")


# =============================================================================
# qrXXinv (npfunctions.R:89-93)
# =============================================================================


def qrXXinv(x: np.ndarray) -> np.ndarray:
    """Cholesky-based inverse of ``x.T @ x``.

    Mirrors ``chol2inv(chol(crossprod(x)))`` in R. ``x`` typically
    represents a design matrix already scaled by ``sqrt(weights)``.

    Parameters
    ----------
    x : np.ndarray, shape (n, k)

    Returns
    -------
    np.ndarray, shape (k, k)
        Inverse of ``x.T @ x``.

    Raises
    ------
    ValueError
        If ``x.T @ x`` is rank-deficient (Cholesky fails). Converts
        the raw ``np.linalg.LinAlgError`` into a targeted message so
        callers (``lprobust_bw``) can surface a clear failure reason
        instead of an opaque linear-algebra error.
    """
    xtx = x.T @ x
    k = xtx.shape[0]
    # Cholesky solve for the inverse. Matches R's chol2inv(chol(.)).
    try:
        L = np.linalg.cholesky(xtx)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"qrXXinv: Cholesky decomposition of X'X ({k}x{k}) failed. "
            f"The weighted design matrix is rank-deficient, likely "
            f"because the in-window support has fewer than {k} distinct "
            f"points. Increase sample size, widen the bandwidth, or pick "
            f"a boundary with more distinct values nearby. "
            f"(LinAlgError: {exc})"
        ) from exc
    Linv = np.linalg.solve(L, np.eye(k))
    return Linv.T @ Linv


# =============================================================================
# Nearest-neighbor duplicate precomputation (inlined in mse.dpi, R:518-529)
# =============================================================================


def _precompute_nn_duplicates(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute ``dups`` and ``dupsid`` arrays for NN residuals.

    ``x`` must already be sorted ascending.

    Mirrors npfunctions.R:518-529:

        for (j in 1:N) dups[j] = sum(x == x[j])
        j = 1
        while (j <= N) {
          dupsid[j:(j + dups[j] - 1)] = 1:dups[j]
          j = j + dups[j]
        }
    """
    n = x.shape[0]
    dups = np.empty(n, dtype=np.int64)
    for j in range(n):
        dups[j] = int(np.sum(x == x[j]))
    dupsid = np.empty(n, dtype=np.int64)
    j = 0
    while j < n:
        k = int(dups[j])
        # 1-indexed in R: dupsid[j:(j+dups[j]-1)] = 1:dups[j]
        dupsid[j : j + k] = np.arange(1, k + 1)
        j += k
    return dups, dupsid


# =============================================================================
# lprobust.res (npfunctions.R:131-162)
# =============================================================================


def lprobust_res(
    X: np.ndarray,
    y: np.ndarray,
    m: np.ndarray,
    hii: Optional[np.ndarray],
    vce: str,
    matches: int,
    dups: Optional[np.ndarray],
    dupsid: Optional[np.ndarray],
    d: int,
) -> np.ndarray:
    """Port of ``lprobust.res``.

    Parameters
    ----------
    X : np.ndarray, shape (n,)
        Regressor (one-column; the R code treats ``X`` as a scalar
        series for NN distance comparisons).
    y : np.ndarray, shape (n,)
        Outcome.
    m : np.ndarray, shape (n, 1) or (n,)
        Fitted values from the local-polynomial regression. Unused
        when ``vce="nn"``.
    hii : np.ndarray or None
        Hat-matrix diagonal, used by "hc2" / "hc3". Ignored for "nn",
        "hc0", "hc1".
    vce : str
        One of "nn", "hc0", "hc1", "hc2", "hc3".
    matches : int
        ``nnmatch``, target number of nearest neighbors per observation.
    dups, dupsid : np.ndarray or None
        Precomputed (see ``_precompute_nn_duplicates``). Required when
        ``vce="nn"``.
    d : int
        Polynomial-order-plus-one (``o + 1`` in lprobust.bw). Used for
        the HC1 degrees-of-freedom correction.

    Returns
    -------
    np.ndarray, shape (n, 1)
        Column vector of residuals.
    """
    if vce not in _VALID_VCE:
        raise ValueError(f"Unknown vce {vce!r}. Expected one of {_VALID_VCE}.")

    n = y.shape[0]
    res = np.empty((n, 1), dtype=np.float64)

    if vce == "nn":
        if dups is None or dupsid is None:
            raise ValueError("vce='nn' requires precomputed dups / dupsid")
        # Port of npfunctions.R:134-153. R uses 1-based indexing; Python is
        # 0-based, so every `pos`, `pos-lpos-1`, and `pos+rpos+1` translates
        # to subtracting one from the R expression to land in [0, n).
        for pos in range(n):
            rpos = int(dups[pos] - dupsid[pos])
            lpos = int(dupsid[pos] - 1)
            lim = min(matches, n - 1)
            while lpos + rpos < lim:
                # Guard conditions mirror R exactly; "pos-lpos-1" in R means
                # 1-indexed position, so "pos-lpos-1 <= 0" in R corresponds to
                # Python "(pos) - lpos - 1 < 0" where pos is 0-indexed.
                left_idx = pos - lpos - 1  # R: pos-lpos-1 (1-indexed)
                right_idx = pos + rpos + 1  # R: pos+rpos+1 (1-indexed)
                # In Python 0-indexed, "pos-lpos-1 <= 0" means no more room left.
                # Equivalent check: left_idx < 0 means index 0 already past.
                if left_idx < 0:
                    rpos = rpos + int(dups[right_idx])
                elif right_idx > n - 1:
                    lpos = lpos + int(dups[left_idx])
                elif (X[pos] - X[left_idx]) > (X[right_idx] - X[pos]):
                    rpos = rpos + int(dups[right_idx])
                elif (X[pos] - X[left_idx]) < (X[right_idx] - X[pos]):
                    lpos = lpos + int(dups[left_idx])
                else:
                    rpos = rpos + int(dups[right_idx])
                    lpos = lpos + int(dups[left_idx])
            # Indices of the neighbor group (inclusive bounds in R).
            ind_J_start = pos - lpos
            ind_J_end = min(n - 1, pos + rpos)  # inclusive
            y_J = float(np.sum(y[ind_J_start : ind_J_end + 1]) - y[pos])
            J_i = (ind_J_end - ind_J_start + 1) - 1
            res[pos, 0] = np.sqrt(J_i / (J_i + 1.0)) * (y[pos] - y_J / J_i)
        return res

    # HC variants (vce != "nn")
    m1 = m.reshape(-1) if m.ndim == 2 else m
    if vce == "hc0":
        w = np.ones(n, dtype=np.float64)
    elif vce == "hc1":
        w = np.full(n, np.sqrt(n / (n - d)), dtype=np.float64)
    elif vce == "hc2":
        if hii is None:
            raise ValueError("vce='hc2' requires hii")
        w = np.sqrt(1.0 / (1.0 - hii.reshape(-1)))
    else:  # hc3
        if hii is None:
            raise ValueError("vce='hc3' requires hii")
        w = 1.0 / (1.0 - hii.reshape(-1))
    res[:, 0] = w * (y - m1)
    return res


# =============================================================================
# lprobust.vce (npfunctions.R:165-185)
# =============================================================================


def lprobust_vce(
    RX: np.ndarray,
    res: np.ndarray,
    cluster: Optional[np.ndarray],
) -> np.ndarray:
    """Port of ``lprobust.vce``. Meat of the sandwich.

    Parameters
    ----------
    RX : np.ndarray, shape (n, k)
        Weighted design matrix ``R * eW`` from the caller.
    res : np.ndarray, shape (n, 1) or (n,)
        Residuals from ``lprobust_res``.
    cluster : np.ndarray or None
        Cluster identifier, same length as ``res``. ``None`` for
        unclustered.

    Returns
    -------
    np.ndarray, shape (k, k)
        Meat matrix.
    """
    n = RX.shape[0]
    k = RX.shape[1]
    r = res.reshape(-1)

    if cluster is None:
        rRX = (r[:, None]) * RX
        return rRX.T @ rRX

    clusters = np.unique(cluster)
    g = clusters.shape[0]
    w = ((n - 1) / (n - k)) * (g / (g - 1))
    M = np.zeros((k, k), dtype=np.float64)
    for c in clusters:
        ind = cluster == c
        Xi = RX[ind, :]
        ri = r[ind]
        # R: M = M + crossprod(t(crossprod(Xi,ri)),t(crossprod(Xi,ri)))
        # crossprod(Xi, ri) is a (k,) vector = Xi.T @ ri
        v = Xi.T @ ri
        M = M + np.outer(v, v)
    return w * M


# =============================================================================
# lprobust.bw (npfunctions.R:187-288)
# =============================================================================


@dataclass
class LprobustBwResult:
    """Return value of ``lprobust_bw``. Mirrors the R list.

    Attributes
    ----------
    V, B1, B2, R, r, rB, rV, bw : float
        See npfunctions.R:276-287 for the exact formulas.
    """

    V: float
    B1: float
    B2: float
    R: float
    r: float
    rB: float
    rV: float
    bw: float


def lprobust_bw(
    Y: np.ndarray,
    X: np.ndarray,
    cluster: Optional[np.ndarray],
    c: float,
    o: int,
    nu: int,
    o_B: int,
    h_V: float,
    h_B1: float,
    h_B2: float,
    scale: float,
    vce: str,
    nnmatch: int,
    kernel: str,
    dups: Optional[np.ndarray],
    dupsid: Optional[np.ndarray],
) -> LprobustBwResult:
    """Port of ``lprobust.bw`` (npfunctions.R:187-288).

    The heart of the 3-stage DPI: one call produces one stage's
    ``(V, B1, B2, R, bw)``. Called four times from ``lpbwselect_mse_dpi``.

    Parameters match the R signature argument-for-argument.

    Raises
    ------
    ValueError
        If any of the three local-polynomial fits has fewer in-window
        observations than its required column count. Catches opaque
        ``LinAlgError`` failures from downstream Cholesky inversion in
        tiny-sample or mispositioned-boundary settings and surfaces a
        targeted error naming the failing stage.
    """
    N = X.shape[0]
    eC: Optional[np.ndarray] = None

    # === Variance: local-poly fit of order o at bandwidth h_V ===
    u_V = (X - c) / h_V
    w = kernel_W(u_V, kernel) / h_V
    ind_V = w > 0.0
    eY = Y[ind_V]
    eX = X[ind_V]
    eW = w[ind_V]
    n_V = int(np.sum(ind_V))
    if n_V < o + 1:
        raise ValueError(
            f"lprobust_bw: variance stage has n_V={n_V} in-window "
            f"observations at h_V={h_V:.6g} (boundary={c}, kernel={kernel!r}), "
            f"but needs at least o+1={o + 1}. Increase sample size, choose "
            f"a valid lower boundary with sufficient data to the right, "
            f"or disable bwcheck=None-driven narrow windows."
        )
    # Design matrix R.V in R; rename to R_V to avoid Python builtin conflict.
    R_V = np.empty((n_V, o + 1), dtype=np.float64)
    for j in range(o + 1):
        R_V[:, j] = (eX - c) ** j
    sqrtW = np.sqrt(eW)
    invG_V = qrXXinv(R_V * sqrtW[:, None])
    beta_V = invG_V @ (R_V.T @ (eW * eY))

    if cluster is not None:
        eC = cluster[ind_V]

    dups_V = dupsid_V = None
    if vce == "nn":
        if dups is None or dupsid is None:
            raise ValueError("vce='nn' requires precomputed dups/dupsid")
        dups_V = dups[ind_V]
        dupsid_V = dupsid[ind_V]

    predicts_V = np.zeros(n_V, dtype=np.float64)
    hii: Optional[np.ndarray] = None
    if vce in ("hc0", "hc1", "hc2", "hc3"):
        predicts_V = R_V @ beta_V
        if vce in ("hc2", "hc3"):
            hii = np.empty(n_V, dtype=np.float64)
            RW = R_V * eW[:, None]
            for i in range(n_V):
                hii[i] = R_V[i, :] @ invG_V @ RW[i, :]

    res_V = lprobust_res(
        eX,
        eY,
        predicts_V.reshape(-1, 1),
        hii,
        vce,
        nnmatch,
        dups_V,
        dupsid_V,
        o + 1,
    )
    meat_V = lprobust_vce(R_V * eW[:, None], res_V, eC)
    V_V = float((invG_V @ meat_V @ invG_V)[nu, nu])

    # === Bias coefficient BConst1 / BConst2 ===
    # Hp (diag scaling in R). Hp[j] = h.V^((j-1)) for j=1..o+1, so at Python
    # index i in [0, o], Hp[i] = h_V ** i.
    Hp = np.array([h_V**j for j in range(o + 1)], dtype=np.float64)
    v1 = (R_V * eW[:, None]).T @ ((eX - c) / h_V) ** (o + 1)
    v2 = (R_V * eW[:, None]).T @ ((eX - c) / h_V) ** (o + 2)
    # (Hp * (invG.V %*% v1))[nu+1] in R == Python index nu.
    BConst1 = float((Hp * (invG_V @ v1))[nu])
    BConst2 = float((Hp * (invG_V @ v2))[nu])

    # === B1 via a separate fit at h.B1, order o.B ===
    u_B1 = (X - c) / h_B1
    w1 = kernel_W(u_B1, kernel)
    ind1 = w1 > 0.0
    eY1 = Y[ind1]
    eX1 = X[ind1]
    eW1 = w1[ind1]
    n_B1 = int(np.sum(ind1))
    if n_B1 < o_B + 1:
        raise ValueError(
            f"lprobust_bw: B1 stage has n_B1={n_B1} in-window observations "
            f"at h_B1={h_B1:.6g} (boundary={c}, kernel={kernel!r}), but "
            f"needs at least o_B+1={o_B + 1}. Increase sample size or "
            f"widen the pilot bandwidth."
        )
    R_B1 = np.empty((n_B1, o_B + 1), dtype=np.float64)
    for j in range(o_B + 1):
        R_B1[:, j] = (eX1 - c) ** j
    sqrtW1 = np.sqrt(eW1)
    invG_B1 = qrXXinv(R_B1 * sqrtW1[:, None])
    beta_B1 = invG_B1 @ (R_B1.T @ (eW1 * eY1))

    # === BWreg (only when scale > 0) ===
    BWreg = 0.0
    if scale > 0:
        eC1: Optional[np.ndarray] = None
        if cluster is not None:
            eC1 = cluster[ind1]
        dups_B = dupsid_B = None
        hii_B = None
        predicts_B = np.zeros(n_B1, dtype=np.float64)
        if vce == "nn":
            dups_B = dups[ind1] if dups is not None else None
            dupsid_B = dupsid[ind1] if dupsid is not None else None
        if vce in ("hc0", "hc1", "hc2", "hc3"):
            predicts_B = R_B1 @ beta_B1
            if vce in ("hc2", "hc3"):
                hii_B = np.empty(n_B1, dtype=np.float64)
                RW1 = R_B1 * eW1[:, None]
                for i in range(n_B1):
                    hii_B[i] = R_B1[i, :] @ invG_B1 @ RW1[i, :]
        res_B = lprobust_res(
            eX1,
            eY1,
            predicts_B.reshape(-1, 1),
            hii_B,
            vce,
            nnmatch,
            dups_B,
            dupsid_B,
            o_B + 1,
        )
        V_B = float(
            (invG_B1 @ lprobust_vce(R_B1 * eW1[:, None], res_B, eC1) @ invG_B1)[o + 1, o + 1]
        )
        BWreg = 3.0 * BConst1 * BConst1 * V_B

    # === B2 via a separate fit at h.B2, order o.B+1 ===
    u_B2 = (X - c) / h_B2
    w2 = kernel_W(u_B2, kernel)
    ind2 = w2 > 0.0
    eY2 = Y[ind2]
    eX2 = X[ind2]
    eW2 = w2[ind2]
    n_B2 = int(np.sum(ind2))
    if n_B2 < o_B + 2:
        raise ValueError(
            f"lprobust_bw: B2 stage has n_B2={n_B2} in-window observations "
            f"at h_B2={h_B2:.6g} (boundary={c}, kernel={kernel!r}), but "
            f"needs at least o_B+2={o_B + 2}. Increase sample size or "
            f"widen the pilot bandwidth."
        )
    R_B2 = np.empty((n_B2, o_B + 2), dtype=np.float64)
    for j in range(o_B + 2):
        R_B2[:, j] = (eX2 - c) ** j
    sqrtW2 = np.sqrt(eW2)
    invG_B2 = qrXXinv(R_B2 * sqrtW2[:, None])
    beta_B2 = invG_B2 @ (R_B2.T @ (eW2 * eY2))

    # === Compose final scalars (npfunctions.R:276-287) ===
    # R: B1 = BConst1 * beta.B1[o+2]  (1-indexed); Python: beta_B1[o+1]
    B1_val = float(BConst1 * beta_B1[o + 1])
    # R: B2 = BConst2 * beta.B2[o+3]  (1-indexed); Python: beta_B2[o + 2]
    B2_val = float(BConst2 * beta_B2[o + 2])
    V = float(N * h_V ** (2 * nu + 1) * V_V)
    R_reg = float(BWreg)
    r_val = 1.0 / (2.0 * o + 3.0)
    rB = float(2 * (o + 1 - nu))
    rV = float(2 * nu + 1)
    bw = (rV * V / (N * rB * (B1_val**2 + scale * R_reg))) ** r_val

    return LprobustBwResult(
        V=V,
        B1=B1_val,
        B2=B2_val,
        R=R_reg,
        r=r_val,
        rB=rB,
        rV=rV,
        bw=float(bw),
    )


# =============================================================================
# lpbwselect.mse.dpi (npfunctions.R:498-607)
# =============================================================================


@dataclass
class MseDpiStages:
    """Return value of ``lpbwselect_mse_dpi``. Mirrors the R list plus
    exposes per-stage diagnostics."""

    h_mse_dpi: float
    b_mse_dpi: float
    c_bw: float
    bw_mp2: float
    bw_mp3: float
    bw_max: float
    bw_min: Optional[float]
    stage_d1: LprobustBwResult
    stage_d2: LprobustBwResult
    stage_b: LprobustBwResult
    stage_h: LprobustBwResult


def lpbwselect_mse_dpi(
    y: np.ndarray,
    x: np.ndarray,
    cluster: Optional[np.ndarray] = None,
    eval_point: float = 0.0,
    p: int = 1,
    q: Optional[int] = None,
    deriv: int = 0,
    kernel: str = "epa",
    bwcheck: Optional[int] = 21,
    bwregul: float = 1.0,
    vce: str = "nn",
    nnmatch: int = 3,
    interior: bool = False,
) -> MseDpiStages:
    """Port of ``lpbwselect.mse.dpi`` (npfunctions.R:498-607).

    The R source computes ``even = (p - deriv) %% 2 == 0`` and dispatches
    each stage via ``if (even == FALSE | interior == TRUE) bw <- C$bw
    else bw <- optimize(...)$minimum``. For the HAD use case
    (``p=1, deriv=0``), ``(p - deriv) %% 2 == 1``, so ``even == FALSE`` and
    every stage bandwidth comes from the closed-form ``C$bw`` expression
    inside ``lprobust.bw``; the ``optimize()`` branch is taken only when
    ``(p - deriv)`` is even AND ``interior == FALSE``.

    Parameters match the R signature. See the R source comments for
    semantics.

    Raises
    ------
    ValueError
        If ``bwcheck`` is supplied and falls outside the valid range
        ``[1, len(x)]``.
    """
    N = x.shape[0] if hasattr(x, "shape") else len(x)
    if bwcheck is not None:
        if bwcheck < 1:
            raise ValueError(
                f"bwcheck must be a positive integer (>= 1); got {bwcheck}"
            )
        if bwcheck > N:
            raise ValueError(
                f"bwcheck={bwcheck} exceeds sample size N={N}. Either "
                f"reduce bwcheck or increase sample size; pass "
                f"bwcheck=None to skip the nearest-neighbor floor."
            )
    if kernel not in _VALID_KERNELS:
        raise ValueError(f"Unknown kernel {kernel!r}. Expected one of {_VALID_KERNELS}.")
    if vce not in _VALID_VCE:
        raise ValueError(f"Unknown vce {vce!r}. Expected one of {_VALID_VCE}.")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if cluster is not None:
        cluster = np.asarray(cluster)
    if q is None:
        q = p + 1

    even = (p - deriv) % 2 == 0
    N = x.shape[0]
    x_min = float(x.min())
    x_max = float(x.max())
    range_ = x_max - x_min
    x_iq = float(np.percentile(x, 75) - np.percentile(x, 25))

    C_c_table = {"epa": 2.34, "uni": 1.843, "tri": 2.576, "gau": 1.06}
    C_c = C_c_table[kernel]
    sd_x = float(np.std(x, ddof=1))
    c_bw = C_c * min(sd_x, x_iq / 1.349) * N ** (-1.0 / 5.0)
    bw_max = max(abs(eval_point - x_min), abs(eval_point - x_max))
    c_bw = min(c_bw, bw_max)

    # Sort and precompute NN structure (npfunctions.R:518-529).
    dups: Optional[np.ndarray] = None
    dupsid: Optional[np.ndarray] = None
    if vce == "nn":
        order_x = np.argsort(x)
        x = x[order_x]
        y = y[order_x]
        if cluster is not None:
            cluster = cluster[order_x]
        dups, dupsid = _precompute_nn_duplicates(x)

    bw_min: Optional[float] = None
    if bwcheck is not None:
        sorted_abs = np.sort(np.abs(x - eval_point))
        # R: bw.min <- sort(abs(x-eval))[bwcheck]. R is 1-indexed, so bwcheck-1 in Python.
        bw_min = float(sorted_abs[bwcheck - 1])
        c_bw = max(c_bw, bw_min)

    def _optimize_amse(
        C: LprobustBwResult,
        exp_bias: float,
        exp_var: float,
        scale: float,
    ) -> float:
        """Minimize ``|H^exp_bias * (B1 + H*B2 + scale*R)^2 + V / (N * H^exp_var)|``
        over ``H`` in ``(eps, range)``. Matches ``optimize()`` in R.

        Only called on the even-and-boundary branch of the original R
        conditional; for the HAD case (p=1, deriv=0) we take the
        closed-form ``C.bw`` branch instead.
        """

        def _fun(H: float) -> float:
            return abs(H**exp_bias * (C.B1 + H * C.B2 + scale * C.R) ** 2 + C.V / (N * H**exp_var))

        res = optimize.minimize_scalar(
            _fun,
            bounds=(np.finfo(float).eps, range_),
            method="bounded",
            options={"xatol": 1e-10},
        )
        return float(res.x)

    # Stage 2: C.d1 -> bw.mp2
    C_d1 = lprobust_bw(
        y,
        x,
        cluster,
        eval_point,
        o=q + 1,
        nu=q + 1,
        o_B=q + 2,
        h_V=c_bw,
        h_B1=range_,
        h_B2=range_,
        scale=0.0,
        vce=vce,
        nnmatch=nnmatch,
        kernel=kernel,
        dups=dups,
        dupsid=dupsid,
    )
    if (not even) or interior:
        bw_mp2 = C_d1.bw
    else:
        bw_mp2 = _optimize_amse(
            C_d1,
            exp_bias=2 * (q + 1) + 2 - 2 * (q + 1),  # = 2
            exp_var=1 + 2 * (q + 1),
            scale=0.0,
        )

    # Stage 2: C.d2 -> bw.mp3
    C_d2 = lprobust_bw(
        y,
        x,
        cluster,
        eval_point,
        o=q + 2,
        nu=q + 2,
        o_B=q + 3,
        h_V=c_bw,
        h_B1=range_,
        h_B2=range_,
        scale=0.0,
        vce=vce,
        nnmatch=nnmatch,
        kernel=kernel,
        dups=dups,
        dupsid=dupsid,
    )
    if (not even) or interior:
        bw_mp3 = C_d2.bw
    else:
        bw_mp3 = _optimize_amse(
            C_d2,
            exp_bias=2 * (q + 2) + 2 - 2 * (q + 2),  # = 2
            exp_var=1 + 2 * (q + 2),
            scale=0.0,
        )

    # Clipping (npfunctions.R:559-565)
    bw_mp2 = min(bw_mp2, bw_max)
    bw_mp3 = min(bw_mp3, bw_max)
    if bw_min is not None:
        bw_mp2 = max(bw_mp2, bw_min)
        bw_mp3 = max(bw_mp3, bw_min)

    # Stage 3: C.b -> b.mse.dpi
    C_b = lprobust_bw(
        y,
        x,
        cluster,
        eval_point,
        o=q,
        nu=p + 1,
        o_B=q + 1,
        h_V=c_bw,
        h_B1=bw_mp2,
        h_B2=bw_mp3,
        scale=bwregul,
        vce=vce,
        nnmatch=nnmatch,
        kernel=kernel,
        dups=dups,
        dupsid=dupsid,
    )
    if (not even) or interior:
        b_mse_dpi = C_b.bw
    else:
        b_mse_dpi = _optimize_amse(
            C_b,
            exp_bias=2 * q + 2 - 2 * (p + 1),
            exp_var=1 + 2 * (p + 1),
            scale=bwregul,
        )
    b_mse_dpi = min(b_mse_dpi, bw_max)
    if bw_min is not None:
        b_mse_dpi = max(b_mse_dpi, bw_min)

    # Stage 3 final: C.h -> h.mse.dpi
    C_h = lprobust_bw(
        y,
        x,
        cluster,
        eval_point,
        o=p,
        nu=deriv,
        o_B=q,
        h_V=c_bw,
        h_B1=b_mse_dpi,
        h_B2=bw_mp2,
        scale=bwregul,
        vce=vce,
        nnmatch=nnmatch,
        kernel=kernel,
        dups=dups,
        dupsid=dupsid,
    )
    if (not even) or interior:
        h_mse_dpi = C_h.bw
    else:
        h_mse_dpi = _optimize_amse(
            C_h,
            exp_bias=2 * p + 2 - 2 * deriv,
            exp_var=1 + 2 * deriv,
            scale=bwregul,
        )
    h_mse_dpi = min(h_mse_dpi, bw_max)
    if bw_min is not None:
        h_mse_dpi = max(h_mse_dpi, bw_min)

    return MseDpiStages(
        h_mse_dpi=float(h_mse_dpi),
        b_mse_dpi=float(b_mse_dpi),
        c_bw=float(c_bw),
        bw_mp2=float(bw_mp2),
        bw_mp3=float(bw_mp3),
        bw_max=float(bw_max),
        bw_min=bw_min,
        stage_d1=C_d1,
        stage_d2=C_d2,
        stage_b=C_b,
        stage_h=C_h,
    )
