"""Pre-test diagnostics for the HeterogeneousAdoptionDiD estimator.

Paper Section 4 (de Chaisemartin, Ciccia, D'Haultfoeuille, Knau 2026,
arXiv:2405.04465v6) prescribes a four-step pre-testing workflow for TWFE
validity in HADs. Phase 3 ships steps 1 and 3 of that workflow (step 2 is
deferred):

1. :func:`qug_test` - order-statistic ratio test of the support infimum
   ``H_0: d_lower = 0`` (paper Theorem 4). Closed-form, tuning-free.
2. :func:`stute_test` - Cramer-von Mises cusum test of linearity of
   ``E[ΔY | D_2]`` with Mammen (1993) wild bootstrap p-value (paper
   Appendix D).
3. :func:`yatchew_hr_test` - heteroskedasticity-robust variance-ratio
   linearity test (paper Theorem 7 / Equation 29). Feasible at
   ``G >= 100k``.

The composite :func:`did_had_pretest_workflow` runs the three implemented
tests in sequence on a two-period HAD panel and returns a
:class:`HADPretestReport` with a partial-workflow verdict. When all three
fail-to-reject, the verdict explicitly flags that **the paper's step 2
pre-trends test (Assumption 7) is NOT run** — callers do not receive an
unconditional "TWFE safe" signal; the Assumption 7 check must be performed
separately (e.g., via an event-study / placebo analysis) until the Phase 3
follow-up patch lands the joint Equation 18 cross-horizon Stute variant.

See ``docs/methodology/REGISTRY.md`` and ``TODO.md`` for the deferred items.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from diff_diff.had import (
    _aggregate_first_difference,
    _json_safe_scalar,
    _validate_had_panel,
)
from diff_diff.utils import _generate_mammen_weights

__all__ = [
    "QUGTestResults",
    "StuteTestResults",
    "YatchewTestResults",
    "HADPretestReport",
    "qug_test",
    "stute_test",
    "yatchew_hr_test",
    "did_had_pretest_workflow",
]


_MIN_G_QUG = 2
_MIN_G_STUTE = 10
_MIN_G_YATCHEW = 3
_MIN_N_BOOTSTRAP = 99
_STUTE_LARGE_G_THRESHOLD = 100_000

# Scale-invariant tolerance for detecting a numerically exact linear OLS fit.
# The ratio SSR / TSS = sum(eps^2) / sum((dy - dybar)^2) equals 1 - R^2
# and is BOTH TRANSLATION-INVARIANT (centering absorbs additive shifts)
# and SCALE-INVARIANT (the ratio is dimensionless under multiplicative
# rescaling of dy). Under exact Assumption 8, residuals are mathematically
# zero; in practice FP round-off leaves eps on the order of machine-epsilon
# (~1e-16). Squared that is ~1e-32. The threshold ~1e-24 leaves ~10^8
# accumulated FP operations of margin so genuinely-noisy data is never
# mis-classified.
#
# IMPORTANT: the comparison is purely ``eps^2 <= tol * dy_centered^2`` with
# NO additive floor (e.g. ``max(dy_centered^2, 1.0)`` would break scale
# invariance - scaling dy by 1e-12 would make dy_centered^2 ~ 1e-24 but
# the floor would hold the threshold at 1.0, firing the shortcut on
# noisy data that should not trigger it). The ``dy_centered^2 == 0``
# edge case (constant dy) is handled by a separate branch above the
# relative comparison, so the relative form is only applied when the
# denominator is genuinely positive.
_EXACT_LINEAR_RELATIVE_TOL = 1e-24


# =============================================================================
# Result dataclasses
# =============================================================================


@dataclass
class QUGTestResults:
    """Result of :func:`qug_test` (paper Theorem 4).

    The QUG test rejects ``H_0: d_lower = 0`` when the order-statistic
    ratio ``T = D_{(1)} / (D_{(2)} - D_{(1)})`` exceeds ``1/alpha - 1``.
    Under the null, the asymptotic limit law of ``T`` is the ratio of two
    independent Exp(1) random variables, with CDF ``F(t) = t / (1 + t)``,
    so ``p_value = 1 / (1 + T)``.

    Attributes
    ----------
    t_stat : float
        ``D_{(1)} / (D_{(2)} - D_{(1)})``. NaN when fewer than 2 non-zero
        observations remain or when the two smallest doses tie.
    p_value : float
        ``1 / (1 + t_stat)`` under the null. NaN when ``t_stat`` is NaN.
    reject : bool
        ``True`` iff ``t_stat > critical_value``. ``False`` on NaN statistic.
    alpha : float
        Significance level used.
    critical_value : float
        ``1 / alpha - 1``. Populated even when the statistic is NaN so
        downstream readers can inspect the decision threshold.
    n_obs : int
        Number of observations after filtering to ``d > 0``.
    n_excluded_zero : int
        Number of zero-dose observations excluded from the sample.
    d_order_1 : float
        Smallest positive dose ``D_{(1)}``. NaN when ``n_obs < 2``.
    d_order_2 : float
        Second-smallest positive dose ``D_{(2)}``. NaN when ``n_obs < 2``.
    """

    t_stat: float
    p_value: float
    reject: bool
    alpha: float
    critical_value: float
    n_obs: int
    n_excluded_zero: int
    d_order_1: float
    d_order_2: float

    def __repr__(self) -> str:
        return (
            f"QUGTestResults(t_stat={self.t_stat:.4f}, p_value={self.p_value:.4f}, "
            f"reject={self.reject}, alpha={self.alpha}, n_obs={self.n_obs})"
        )

    def summary(self) -> str:
        """Formatted summary table."""
        width = 64
        lines = [
            "=" * width,
            "QUG null test (H_0: d_lower = 0)".center(width),
            "=" * width,
            f"{'Statistic T:':<30} {self.t_stat:>20.4f}",
            f"{'p-value:':<30} {self.p_value:>20.4f}",
            f"{'Critical value (1/alpha-1):':<30} {self.critical_value:>20.4f}",
            f"{'Reject H_0:':<30} {str(self.reject):>20}",
            f"{'alpha:':<30} {self.alpha:>20.4f}",
            f"{'Observations:':<30} {self.n_obs:>20}",
            f"{'Excluded (d == 0):':<30} {self.n_excluded_zero:>20}",
            f"{'D_(1):':<30} {self.d_order_1:>20.4f}",
            f"{'D_(2):':<30} {self.d_order_2:>20.4f}",
            "=" * width,
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Return results as a JSON-safe dict."""
        return {
            "test": "qug",
            "t_stat": _json_safe_scalar(self.t_stat),
            "p_value": _json_safe_scalar(self.p_value),
            "reject": bool(self.reject),
            "alpha": float(self.alpha),
            "critical_value": _json_safe_scalar(self.critical_value),
            "n_obs": int(self.n_obs),
            "n_excluded_zero": int(self.n_excluded_zero),
            "d_order_1": _json_safe_scalar(self.d_order_1),
            "d_order_2": _json_safe_scalar(self.d_order_2),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a one-row DataFrame of the result dict."""
        return pd.DataFrame([self.to_dict()])


@dataclass
class StuteTestResults:
    """Result of :func:`stute_test` (paper Appendix D).

    The Stute test rejects the null that ``E[ΔY | D_2]`` is linear in
    ``D_2`` (paper Assumption 8) when the sorted-residual CvM statistic
    ``S = (1/G^2) Σ (Σ_{h=1}^g eps_{(h)})^2`` exceeds the Mammen wild
    bootstrap ``1 - alpha`` quantile.

    Attributes
    ----------
    cvm_stat : float
        CvM statistic. NaN when ``G < 10`` (below the threshold the
        statistic is not well-calibrated).
    p_value : float
        Bootstrap p-value ``(1 + sum(S_b >= S)) / (B + 1)``. NaN when
        the statistic is NaN.
    reject : bool
        ``True`` iff ``p_value <= alpha``. ``False`` on NaN.
    alpha : float
        Significance level used.
    n_bootstrap : int
        Number of Mammen wild bootstrap replications.
    n_obs : int
        Number of observations.
    seed : int or None
        Seed passed to ``np.random.default_rng``. ``None`` when unseeded.
    """

    cvm_stat: float
    p_value: float
    reject: bool
    alpha: float
    n_bootstrap: int
    n_obs: int
    seed: Optional[int]

    def __repr__(self) -> str:
        return (
            f"StuteTestResults(cvm_stat={self.cvm_stat:.4f}, "
            f"p_value={self.p_value:.4f}, reject={self.reject}, "
            f"alpha={self.alpha}, n_bootstrap={self.n_bootstrap}, "
            f"n_obs={self.n_obs})"
        )

    def summary(self) -> str:
        """Formatted summary table."""
        width = 64
        lines = [
            "=" * width,
            "Stute CvM linearity test (H_0: linear E[dY|D])".center(width),
            "=" * width,
            f"{'CvM statistic:':<30} {self.cvm_stat:>20.4f}",
            f"{'Bootstrap p-value:':<30} {self.p_value:>20.4f}",
            f"{'Reject H_0:':<30} {str(self.reject):>20}",
            f"{'alpha:':<30} {self.alpha:>20.4f}",
            f"{'Bootstrap replications:':<30} {self.n_bootstrap:>20}",
            f"{'Observations:':<30} {self.n_obs:>20}",
            f"{'Seed:':<30} {str(self.seed):>20}",
            "=" * width,
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Return results as a JSON-safe dict."""
        return {
            "test": "stute",
            "cvm_stat": _json_safe_scalar(self.cvm_stat),
            "p_value": _json_safe_scalar(self.p_value),
            "reject": bool(self.reject),
            "alpha": float(self.alpha),
            "n_bootstrap": int(self.n_bootstrap),
            "n_obs": int(self.n_obs),
            "seed": None if self.seed is None else int(self.seed),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a one-row DataFrame of the result dict."""
        return pd.DataFrame([self.to_dict()])


@dataclass
class YatchewTestResults:
    """Result of :func:`yatchew_hr_test` (paper Theorem 7 / Equation 29).

    Heteroskedasticity-robust test of the same linearity null as
    :func:`stute_test`, but using Yatchew's difference-based variance
    estimator. The test statistic
    ``T_hr = sqrt(G) * (sigma2_lin - sigma2_diff) / sigma2_W``
    is asymptotically N(0, 1) under H_0; rejection uses the one-sided
    standard-normal critical value.

    Attributes
    ----------
    t_stat_hr : float
        Test statistic ``T_hr`` from paper Equation 29. NaN when
        ``G < 3``.
    p_value : float
        ``1 - Phi(T_hr)``. NaN when the statistic is NaN.
    reject : bool
        ``True`` iff ``T_hr >= critical_value``. ``False`` on NaN.
    alpha : float
        Significance level used.
    critical_value : float
        One-sided standard-normal critical value ``z_{1 - alpha}``.
    sigma2_lin : float
        Residual variance from OLS of ``dy`` on ``d``.
    sigma2_diff : float
        Yatchew differencing variance
        ``(1 / (2G)) * sum((dy_{(g)} - dy_{(g-1)})^2)`` - divisor is ``2G``
        (paper-literal), NOT ``2(G-1)``.
    sigma2_W : float
        Heteroskedasticity-robust scale
        ``sqrt((1 / (G-1)) * sum(eps_{(g)}^2 * eps_{(g-1)}^2))``.
    n_obs : int
        Number of observations.
    """

    t_stat_hr: float
    p_value: float
    reject: bool
    alpha: float
    critical_value: float
    sigma2_lin: float
    sigma2_diff: float
    sigma2_W: float
    n_obs: int

    def __repr__(self) -> str:
        return (
            f"YatchewTestResults(t_stat_hr={self.t_stat_hr:.4f}, "
            f"p_value={self.p_value:.4f}, reject={self.reject}, "
            f"alpha={self.alpha}, n_obs={self.n_obs})"
        )

    def summary(self) -> str:
        """Formatted summary table."""
        width = 64
        lines = [
            "=" * width,
            "Yatchew-HR linearity test (H_0: linear E[dY|D])".center(width),
            "=" * width,
            f"{'T_hr statistic:':<30} {self.t_stat_hr:>20.4f}",
            f"{'p-value:':<30} {self.p_value:>20.4f}",
            f"{'Critical value (1-sided z):':<30} {self.critical_value:>20.4f}",
            f"{'Reject H_0:':<30} {str(self.reject):>20}",
            f"{'alpha:':<30} {self.alpha:>20.4f}",
            f"{'sigma^2_lin (OLS):':<30} {self.sigma2_lin:>20.4f}",
            f"{'sigma^2_diff (Yatchew):':<30} {self.sigma2_diff:>20.4f}",
            f"{'sigma^2_W (HR scale):':<30} {self.sigma2_W:>20.4f}",
            f"{'Observations:':<30} {self.n_obs:>20}",
            "=" * width,
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Return results as a JSON-safe dict."""
        return {
            "test": "yatchew_hr",
            "t_stat_hr": _json_safe_scalar(self.t_stat_hr),
            "p_value": _json_safe_scalar(self.p_value),
            "reject": bool(self.reject),
            "alpha": float(self.alpha),
            "critical_value": _json_safe_scalar(self.critical_value),
            "sigma2_lin": _json_safe_scalar(self.sigma2_lin),
            "sigma2_diff": _json_safe_scalar(self.sigma2_diff),
            "sigma2_W": _json_safe_scalar(self.sigma2_W),
            "n_obs": int(self.n_obs),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a one-row DataFrame of the result dict."""
        return pd.DataFrame([self.to_dict()])


@dataclass
class HADPretestReport:
    """Composite output of :func:`did_had_pretest_workflow`.

    Bundles the three individual tests with an overall verdict string.

    .. important::
        This report reflects a **partial** workflow: Phase 3 ships paper
        Sections 4.2-4.3 steps 1 (QUG) and 3 (linearity via Stute +
        Yatchew-HR), but **NOT** step 2 (Assumption 7 pre-trends test via
        Equation 18). Even when ``all_pass`` is ``True``, the paper's
        four-step certification for TWFE validity is incomplete — pre-
        trends testing is a separate diagnostic that must be run via the
        user's own event-study / placebo analysis until the Phase 3
        follow-up patch lands the joint Equation 18 Stute test.

    Attributes
    ----------
    qug : QUGTestResults
    stute : StuteTestResults
    yatchew : YatchewTestResults
    all_pass : bool
        ``True`` iff (a) QUG is conclusive (step 1), (b) at least ONE of
        Stute / Yatchew is conclusive (step 3 - paper's "Stute or
        Yatchew" wording), AND (c) no conclusive test rejects. This
        gating follows the paper's four-step workflow exactly: step 3
        accepts either linearity test, so a conclusive Stute is
        sufficient even when Yatchew returns NaN (e.g. tied-dose
        panels). Even when ``all_pass`` is ``True``, the report is a
        PARTIAL indicator: it does not certify Assumption 7 (pre-trends),
        which is not tested by Phase 3.
    verdict : str
        Human-readable classification. The paper's step 3 accepts either
        Stute OR Yatchew, so a conclusive Stute alone can adjudicate
        linearity even if Yatchew is NaN (e.g. tied-dose panels), and
        vice versa. Priority-ordered first-match:

        1. QUG NaN -> ``"inconclusive - QUG NaN"`` (step 1 is required).
        2. BOTH Stute AND Yatchew NaN -> ``"inconclusive - both Stute
           and Yatchew linearity tests NaN"`` (step 3 requires at least
           one).
        3. None of the CONCLUSIVE tests reject -> partial-workflow
           fail-to-reject verdict. Format: ``"QUG and linearity
           diagnostics fail-to-reject[ (Yatchew NaN - skipped)];
           Assumption 7 pre-trends test NOT run (paper step 2 deferred
           to Phase 3 follow-up)"``. The ``" (... - skipped)"`` suffix
           appears when Stute or Yatchew was NaN but the other was
           conclusive.
        4. At least one conclusive test rejects -> bundled string
           naming each failed assumption: ``"support infimum rejected
           - continuous_at_zero design invalid (QUG)"`` and/or
           ``"linearity rejected - heterogeneity bias
           ({Stute[,Yatchew]})"``.
    alpha : float
        Significance level shared across tests.
    n_obs : int
        Unit count after aggregation to the two-period first-difference.
    """

    qug: QUGTestResults
    stute: StuteTestResults
    yatchew: YatchewTestResults
    all_pass: bool
    verdict: str
    alpha: float
    n_obs: int

    def __repr__(self) -> str:
        return (
            f"HADPretestReport(all_pass={self.all_pass}, "
            f"verdict={self.verdict!r}, n_obs={self.n_obs})"
        )

    def summary(self) -> str:
        """Formatted summary of all three tests and the verdict."""
        width = 72
        parts = [
            "=" * width,
            "HAD pre-test workflow".center(width),
            "=" * width,
            self.qug.summary(),
            "",
            self.stute.summary(),
            "",
            self.yatchew.summary(),
            "",
            "=" * width,
            f"{'All pass:':<30} {str(self.all_pass):>40}",
            f"Verdict: {self.verdict}",
            "=" * width,
        ]
        return "\n".join(parts)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe nested dict of the full report."""
        return {
            "qug": self.qug.to_dict(),
            "stute": self.stute.to_dict(),
            "yatchew": self.yatchew.to_dict(),
            "all_pass": bool(self.all_pass),
            "verdict": str(self.verdict),
            "alpha": float(self.alpha),
            "n_obs": int(self.n_obs),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy 3-row DataFrame (one row per test).

        Columns (in order): ``[test, statistic_name, statistic_value,
        p_value, reject, alpha, n_obs]``.
        """
        rows = [
            {
                "test": "qug",
                "statistic_name": "t_stat",
                "statistic_value": _json_safe_scalar(self.qug.t_stat),
                "p_value": _json_safe_scalar(self.qug.p_value),
                "reject": bool(self.qug.reject),
                "alpha": float(self.qug.alpha),
                "n_obs": int(self.qug.n_obs),
            },
            {
                "test": "stute",
                "statistic_name": "cvm_stat",
                "statistic_value": _json_safe_scalar(self.stute.cvm_stat),
                "p_value": _json_safe_scalar(self.stute.p_value),
                "reject": bool(self.stute.reject),
                "alpha": float(self.stute.alpha),
                "n_obs": int(self.stute.n_obs),
            },
            {
                "test": "yatchew_hr",
                "statistic_name": "t_stat_hr",
                "statistic_value": _json_safe_scalar(self.yatchew.t_stat_hr),
                "p_value": _json_safe_scalar(self.yatchew.p_value),
                "reject": bool(self.yatchew.reject),
                "alpha": float(self.yatchew.alpha),
                "n_obs": int(self.yatchew.n_obs),
            },
        ]
        cols = [
            "test",
            "statistic_name",
            "statistic_value",
            "p_value",
            "reject",
            "alpha",
            "n_obs",
        ]
        return pd.DataFrame(rows).reindex(columns=cols)


# =============================================================================
# Private helpers
# =============================================================================


def _validate_1d_numeric(arr: np.ndarray, name: str) -> np.ndarray:
    """Return ``arr`` as a 1D float ndarray or raise ``ValueError``."""
    a = np.asarray(arr)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional, got shape {a.shape}.")
    a = a.astype(np.float64, copy=False)
    if np.isnan(a).any():
        raise ValueError(f"{name} contains NaN values.")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values (inf).")
    return a


def _fit_ols_intercept_slope(d: np.ndarray, dy: np.ndarray) -> "tuple[float, float, np.ndarray]":
    """Fit ``dy = a + b*d + eps`` via closed-form OLS.

    Returns ``(a_hat, b_hat, residuals)`` where ``residuals`` has the
    same length as ``d`` in the ORIGINAL input order (not sorted).
    """
    d_mean = d.mean()
    dy_mean = dy.mean()
    d_dev = d - d_mean
    var_d = np.dot(d_dev, d_dev)
    if var_d <= 0.0:
        # Degenerate case: all dose values equal. Slope undefined.
        # Caller is responsible for gating before we reach here; if we
        # do reach here, return (mean(dy), 0, dy - mean(dy)).
        return float(dy_mean), 0.0, dy - dy_mean
    b_hat = float(np.dot(d_dev, dy - dy_mean) / var_d)
    a_hat = float(dy_mean - b_hat * d_mean)
    residuals = dy - a_hat - b_hat * d
    return a_hat, b_hat, residuals


def _cvm_statistic(eps_sorted: np.ndarray, d_sorted: np.ndarray) -> float:
    """Compute the tie-safe Cramer-von Mises cusum statistic.

    Paper definition (Appendix D):

        c_G(d) := G^{-1/2} * sum_g 1{D_g <= d} * eps_g
        S := (1/G) * sum_g c_G^2(D_g) = (1/G^2) * sum_g (C_g)^2

    where ``C_g = sum_{h : D_h <= D_g} eps_h`` is the cumulative residual
    sum up to and including ALL observations with dose <= D_g. This
    definition is tie-safe: at a tied dose value ``D_g == D_{g+1}``, both
    c_G(D_g) and c_G(D_{g+1}) include all tie-block members, so the
    cumulative sum used at each tied observation is the cumulative sum
    through the END of the tie block.

    A naive per-observation ``cumsum`` on sorted residuals violates this
    at tie blocks (each tied observation sees a partial within-block
    cumulative sum). This implementation collapses each tie block to the
    post-tie cumulative sum before squaring, matching the paper definition.

    Parameters
    ----------
    eps_sorted : np.ndarray, shape (G,)
        Residuals sorted by ``d_sorted``.
    d_sorted : np.ndarray, shape (G,)
        Regressor values sorted ascending. Must be sorted consistently
        with ``eps_sorted``.

    Returns
    -------
    float
        ``S = (1 / G^2) * sum_g C_g^2``.
    """
    G = eps_sorted.shape[0]
    cumsum = np.cumsum(eps_sorted)
    # Tie-safe correction: replace within-tie-block values with the
    # cumulative sum at the END of each tie block. np.unique on the
    # already-sorted regressor gives per-unique-value counts; the last
    # index of each tie block is `cumsum(counts) - 1`, and np.repeat
    # expands that back to per-observation.
    _, counts = np.unique(d_sorted, return_counts=True)
    tie_end_idx = np.cumsum(counts) - 1
    cumsum_tie_safe = np.repeat(cumsum[tie_end_idx], counts)
    return float(np.sum(cumsum_tie_safe * cumsum_tie_safe) / (G * G))


def _compose_verdict(
    qug: QUGTestResults, stute: StuteTestResults, yatchew: YatchewTestResults
) -> str:
    """Build the :class:`HADPretestReport` verdict string.

    Paper Section 4.2-4.3 specifies a four-step workflow; Phase 3 ships
    step 1 (QUG) and step 3 (linearity, via ``stute_test`` OR
    ``yatchew_hr_test``). The linearity step accepts either test, so a
    conclusive Stute result alone suffices even when Yatchew is NaN
    (e.g. tied doses, which Yatchew rejects by contract).

    Priority-ordered first-match:

    1. QUG NaN -> ``"inconclusive - QUG NaN"`` (step 1 required).
    2. BOTH Stute AND Yatchew NaN -> ``"inconclusive - both Stute and
       Yatchew linearity tests NaN"`` (step 3 requires at least one).
    3. Otherwise, count rejections from CONCLUSIVE tests only:
       3a. None of the conclusive tests reject -> partial-workflow
           fail-to-reject verdict flagging the Assumption 7 gap, plus
           a ``" (Yatchew NaN - skipped)"`` suffix when applicable.
       3b. At least one conclusive test rejects -> bundle each
           rejection reason naming the failed assumption.
    """
    qug_ok = bool(np.isfinite(qug.p_value))
    stute_ok = bool(np.isfinite(stute.p_value))
    yatchew_ok = bool(np.isfinite(yatchew.p_value))

    if not qug_ok:
        return "inconclusive - QUG NaN"
    if not stute_ok and not yatchew_ok:
        return "inconclusive - both Stute and Yatchew linearity tests NaN"

    qug_rej = qug.reject
    stute_rej = bool(stute_ok and stute.reject)
    yatchew_rej = bool(yatchew_ok and yatchew.reject)

    if not (qug_rej or stute_rej or yatchew_rej):
        skipped = []
        if not stute_ok:
            skipped.append("Stute NaN")
        if not yatchew_ok:
            skipped.append("Yatchew NaN")
        skip_note = f" ({'; '.join(skipped)} - skipped)" if skipped else ""
        return (
            "QUG and linearity diagnostics fail-to-reject"
            f"{skip_note}; Assumption 7 pre-trends test NOT run "
            "(paper step 2 deferred to Phase 3 follow-up)"
        )

    reasons = []
    if qug_rej:
        reasons.append("support infimum rejected - continuous_at_zero design invalid (QUG)")
    if stute_rej or yatchew_rej:
        which = ",".join(
            name for name, rejected in (("Stute", stute_rej), ("Yatchew", yatchew_rej)) if rejected
        )
        reasons.append(f"linearity rejected - heterogeneity bias ({which})")
    return "; ".join(reasons)


# =============================================================================
# Public test functions
# =============================================================================


def qug_test(d: np.ndarray, alpha: float = 0.05) -> QUGTestResults:
    """Run the QUG null test for the support infimum (paper Theorem 4).

    Tests ``H_0: d_lower = 0`` using the order-statistic ratio
    ``T = D_{(1)} / (D_{(2)} - D_{(1)})``, rejecting when ``T > 1/alpha - 1``.
    Under the null, the asymptotic limit law of ``T`` is the ratio of two
    independent Exp(1) variables with CDF ``F(t) = t / (1 + t)``, so the
    one-sided p-value is ``1 / (1 + T)``.

    Zero-dose observations are filtered out (the test targets the infimum
    of the treated support). A ``UserWarning`` is emitted naming the
    exclusion count. When fewer than two positive doses remain, the test
    returns all-NaN inference with ``reject=False``.

    Parameters
    ----------
    d : np.ndarray, shape (G,)
        Post-period dose vector. Must be 1D numeric and contain no NaN.
    alpha : float, default 0.05
        One-sided significance level. Must satisfy ``0 < alpha < 1``.

    Returns
    -------
    QUGTestResults
        Result dataclass with ``t_stat``, ``p_value``, ``reject``, and
        sample metadata.

    Raises
    ------
    ValueError
        If ``d`` is not 1D numeric or contains NaN, or if ``alpha`` is
        not in ``(0, 1)``.

    Notes
    -----
    Tie-break: when ``D_{(1)} == D_{(2)}`` the statistic is undefined.
    The test returns ``t_stat=NaN, p_value=NaN, reject=False`` with a
    ``UserWarning`` rather than raising.

    References
    ----------
    de Chaisemartin, Ciccia, D'Haultfoeuille, Knau (2026, arXiv:2405.04465v6),
    Theorem 4 and Section 4.2.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha}.")

    d_arr = _validate_1d_numeric(d, "d")
    critical_value = 1.0 / alpha - 1.0

    # HAD support restriction: doses must be non-negative (paper Section 2).
    # Reject negative doses at the front door rather than silently filtering
    # them into the zero-exclusion counter.
    if (d_arr < 0).any():
        n_neg = int((d_arr < 0).sum())
        raise ValueError(
            f"qug_test: d contains {n_neg} negative value(s); HAD doses "
            f"must be non-negative (paper Section 2). Check your dose "
            f"column or pre-process before calling qug_test."
        )

    mask = d_arr > 0
    d_nz = d_arr[mask]
    n_excluded = int(d_arr.shape[0] - d_nz.shape[0])
    if n_excluded > 0:
        warnings.warn(
            f"qug_test: excluded {n_excluded} observation(s) with d == 0 "
            f"(the QUG null test targets the infimum of the treated-dose "
            f"support; zero-dose observations are not in scope).",
            UserWarning,
            stacklevel=2,
        )

    n_obs = int(d_nz.shape[0])
    if n_obs < _MIN_G_QUG:
        warnings.warn(
            f"qug_test: only {n_obs} positive-dose observation(s); need "
            f"at least {_MIN_G_QUG}. Returning NaN result.",
            UserWarning,
            stacklevel=2,
        )
        return QUGTestResults(
            t_stat=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=alpha,
            critical_value=critical_value,
            n_obs=n_obs,
            n_excluded_zero=n_excluded,
            d_order_1=float("nan"),
            d_order_2=float("nan"),
        )

    # Use np.partition for O(G) extraction of the two smallest positive
    # doses (faster than full O(G log G) sort). For k=1, np.partition
    # guarantees partitioned[0] <= partitioned[1] = D_{(2)} (the 2nd-smallest),
    # which implies partitioned[0] = D_{(1)} (the minimum).
    partitioned = np.partition(d_nz, 1)
    D1 = float(partitioned[0])
    D2 = float(partitioned[1])

    if D2 == D1:
        warnings.warn(
            "qug_test: D_(1) == D_(2); the test statistic is undefined "
            "(ties at the minimum positive dose). Returning NaN result.",
            UserWarning,
            stacklevel=2,
        )
        return QUGTestResults(
            t_stat=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=alpha,
            critical_value=critical_value,
            n_obs=n_obs,
            n_excluded_zero=n_excluded,
            d_order_1=D1,
            d_order_2=D2,
        )

    t_stat = D1 / (D2 - D1)
    p_value = 1.0 / (1.0 + t_stat)
    reject = t_stat > critical_value

    return QUGTestResults(
        t_stat=float(t_stat),
        p_value=float(p_value),
        reject=bool(reject),
        alpha=alpha,
        critical_value=critical_value,
        n_obs=n_obs,
        n_excluded_zero=n_excluded,
        d_order_1=D1,
        d_order_2=D2,
    )


def stute_test(
    d: np.ndarray,
    dy: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 999,
    seed: Optional[int] = None,
) -> StuteTestResults:
    """Run the Stute Cramer-von Mises linearity test (paper Appendix D).

    Tests ``H_0: E[ΔY | D_2]`` is linear in ``D_2`` (paper Assumption 8).
    The test statistic is the sorted-residual cusum CvM

        S = (1 / G^2) * sum_{g=1}^G (sum_{h=1}^g eps_(h))^2

    where ``eps_(h)`` is the ``h``-th OLS residual after sorting by ``d``.
    The p-value is the bootstrap tail probability
    ``(1 + sum(S_b >= S)) / (B + 1)`` under the Mammen (1993) two-point
    wild bootstrap; each bootstrap iteration refits OLS on
    ``dy_b = a_hat + b_hat * d + eps * eta`` with multiplier weights ``eta``.

    Parameters
    ----------
    d, dy : np.ndarray, shape (G,)
        Dose and first-difference outcome vectors.
    alpha : float, default 0.05
        Significance level. Must satisfy ``0 < alpha < 1``.
    n_bootstrap : int, default 999
        Number of Mammen wild bootstrap replications. Must be ``>= 99``
        (below which the discretised p-value grid is too coarse).
    seed : int or None, default None
        Seed for ``np.random.default_rng``. Pass an integer for
        reproducible results.

    Returns
    -------
    StuteTestResults

    Raises
    ------
    ValueError
        If ``d`` / ``dy`` are not 1D numeric, contain NaN, have unequal
        lengths, or if ``alpha`` is outside ``(0, 1)``, or if
        ``n_bootstrap < 99``.

    Notes
    -----
    Sample-size gate: below ``G = 10`` the CvM statistic is not
    well-calibrated. In that case the function emits ``UserWarning`` and
    returns all-NaN inference rather than raising.

    Large-G warning: at ``G > 100_000`` the per-iteration refit dominates
    runtime; the function emits a ``UserWarning`` pointing users to
    :func:`yatchew_hr_test`. Memory usage remains ``O(G)`` regardless
    (no G x G matrix).

    References
    ----------
    Stute, W. (1997). Nonparametric model checks for regression. Annals
    of Statistics 25, 613-641.
    Mammen, E. (1993). Bootstrap and wild bootstrap for high-dimensional
    linear models. Annals of Statistics 21, 255-285.
    de Chaisemartin et al. (2026), Appendix D.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha}.")
    if n_bootstrap < _MIN_N_BOOTSTRAP:
        raise ValueError(
            f"n_bootstrap must be >= {_MIN_N_BOOTSTRAP} (below this the "
            f"discretised p-value grid is too coarse to be meaningful). "
            f"Got n_bootstrap={n_bootstrap}."
        )

    d_arr = _validate_1d_numeric(d, "d")
    dy_arr = _validate_1d_numeric(dy, "dy")
    if d_arr.shape[0] != dy_arr.shape[0]:
        raise ValueError(
            f"d and dy must have the same length; got d.shape={d_arr.shape}, "
            f"dy.shape={dy_arr.shape}."
        )

    G = int(d_arr.shape[0])
    if G < _MIN_G_STUTE:
        warnings.warn(
            f"stute_test: G = {G} is below the minimum {_MIN_G_STUTE} for "
            f"the CvM statistic to be well-calibrated. Returning NaN result.",
            UserWarning,
            stacklevel=2,
        )
        return StuteTestResults(
            cvm_stat=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=alpha,
            n_bootstrap=int(n_bootstrap),
            n_obs=G,
            seed=seed,
        )
    if G > _STUTE_LARGE_G_THRESHOLD:
        warnings.warn(
            f"stute_test: G = {G} exceeds {_STUTE_LARGE_G_THRESHOLD}; the "
            f"per-iteration refit is O(G) per iteration so the "
            f"{n_bootstrap}-replication loop may take tens of seconds or "
            f"more. Consider yatchew_hr_test() instead (paper Theorem 7 "
            f"recommends Yatchew-HR at large G).",
            UserWarning,
            stacklevel=2,
        )

    a_hat, b_hat, eps = _fit_ols_intercept_slope(d_arr, dy_arr)
    # Genuine degeneracy: zero dose variation. The CvM cusum is defined
    # against the regressor, and constant d carries no signal to test
    # linearity against - emit NaN.
    if np.var(d_arr) <= 0.0:
        warnings.warn(
            "stute_test: constant d (zero dose variation); the Stute "
            "linearity test requires regressor variation. Returning NaN result.",
            UserWarning,
            stacklevel=2,
        )
        return StuteTestResults(
            cvm_stat=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=alpha,
            n_bootstrap=int(n_bootstrap),
            n_obs=G,
            seed=seed,
        )

    # Numerically exact linear fit: Assumption 8 holds to IEEE precision,
    # so the Stute CvM statistic is formally 0 and every bootstrap draw is
    # also 0. Short-circuit to p=1 to avoid FP-noise-driven bootstrap
    # comparisons where cvm_stat and S_b are both at machine-epsilon scale.
    # Comparison is purely relative against CENTERED TSS: both translation-
    # invariant (centering absorbs additive shifts) and scale-invariant
    # (ratio is dimensionless under multiplicative dy rescaling).
    eps_norm_sq = float(np.sum(eps * eps))
    dy_centered_sq = float(np.sum((dy_arr - dy_arr.mean()) ** 2))
    if dy_centered_sq <= 0.0:
        # Constant dy (zero centered TSS): trivially linear in d.
        # Return p = 1 without running the bootstrap.
        return StuteTestResults(
            cvm_stat=0.0,
            p_value=1.0,
            reject=False,
            alpha=alpha,
            n_bootstrap=int(n_bootstrap),
            n_obs=G,
            seed=seed,
        )
    if eps_norm_sq <= _EXACT_LINEAR_RELATIVE_TOL * dy_centered_sq:
        return StuteTestResults(
            cvm_stat=0.0,
            p_value=1.0,
            reject=False,
            alpha=alpha,
            n_bootstrap=int(n_bootstrap),
            n_obs=G,
            seed=seed,
        )

    idx = np.argsort(d_arr, kind="stable")
    d_sorted = d_arr[idx]
    S = _cvm_statistic(eps[idx], d_sorted)

    rng = np.random.default_rng(seed)
    bootstrap_S = np.empty(n_bootstrap, dtype=np.float64)
    fitted = a_hat + b_hat * d_arr  # baseline fitted values under H_0
    for b in range(n_bootstrap):
        eta = _generate_mammen_weights(G, rng)
        dy_b = fitted + eps * eta
        _, _, eps_b = _fit_ols_intercept_slope(d_arr, dy_b)
        bootstrap_S[b] = _cvm_statistic(eps_b[idx], d_sorted)

    p_value = float((1.0 + float(np.sum(bootstrap_S >= S))) / (n_bootstrap + 1.0))
    reject = p_value <= alpha

    return StuteTestResults(
        cvm_stat=float(S),
        p_value=p_value,
        reject=bool(reject),
        alpha=alpha,
        n_bootstrap=int(n_bootstrap),
        n_obs=G,
        seed=seed,
    )


def yatchew_hr_test(d: np.ndarray, dy: np.ndarray, alpha: float = 0.05) -> YatchewTestResults:
    """Run the Yatchew heteroskedasticity-robust linearity test.

    Tests ``H_0: E[ΔY | D_2]`` is linear in ``D_2`` (paper Assumption 8,
    Theorem 7) via the variance-ratio statistic

        T_hr = sqrt(G) * (sigma2_lin - sigma2_diff) / sigma2_W

    where

        sigma2_lin   = (1/G) * sum(eps^2)                        # OLS residuals
        sigma2_diff  = (1/(2G)) * sum((dy_{(g)} - dy_{(g-1)})^2) # Yatchew differencing
        sigma2_W     = sqrt((1/(G-1)) * sum(eps_{(g)}^2 * eps_{(g-1)}^2))

    and ``_{(g)}`` denotes sort by ``d``. Rejection uses the one-sided
    standard-normal critical value ``z_{1-alpha}``.

    Parameters
    ----------
    d, dy : np.ndarray, shape (G,)
        Dose and first-difference outcome vectors.
    alpha : float, default 0.05
        One-sided significance level.

    Returns
    -------
    YatchewTestResults

    Raises
    ------
    ValueError
        If ``d`` / ``dy`` are not 1D numeric, contain NaN, have unequal
        lengths, or if ``alpha`` is outside ``(0, 1)``.

    Notes
    -----
    Sample-size gate: below ``G = 3`` the difference-variance estimator
    is undefined; the function emits ``UserWarning`` and returns NaN
    rather than raising.

    Dose ties: REJECTED with ``UserWarning`` + all-NaN result. The
    difference-based variance estimator ``sigma2_diff`` and the
    heteroskedasticity-robust scale ``sigma4_W`` both use adjacent
    differences of quantities sorted by ``d``; under tied doses the
    within-tie row ordering is arbitrary (stable sort falls back to input
    order) so the statistic becomes order-dependent rather than
    data-dependent. Callers with tied doses (mass-point designs,
    discretised dose registers) should use :func:`stute_test` instead -
    its tie-safe Cramer-von Mises statistic collapses tie blocks to the
    post-tie cumulative sum and is provably order-invariant under
    within-tie permutations.

    Exact-linear short-circuit: when the OLS residual sum-of-squares is
    below IEEE precision relative to the centered total sum of squares
    (``sum(eps^2) <= 1e-24 * sum((dy - dybar)^2)``, i.e. essentially
    ``1 - R^2 == 0``), the test short-circuits to ``t_stat_hr=-inf,
    p_value=1.0, reject=False`` - Assumption 8 holds exactly, the formal
    statistic is ``-inf`` under the one-sided critical value, and the
    correct decision is fail-to-reject. This shortcut is translation-
    invariant because the comparison is against centered TSS (not raw
    ``sum(dy^2)``).

    Degenerate ``sigma4_W = 0`` with non-zero residuals: when the
    adjacent-residual-product sum vanishes AFTER the exact-linear
    shortcut is bypassed (e.g. residuals alternate zero/non-zero after
    sorting), the formal statistic is ``+inf`` or ``-inf`` depending on
    the sign of the numerator ``sigma2_lin - sigma2_diff``. The function
    returns the sign-aware limit (``p=0, reject=True`` for positive
    numerator; ``p=1, reject=False`` for negative; ``NaN`` for zero)
    with a ``UserWarning``, rather than unconditionally mapping this to
    ``p=1`` (which would flip a legitimate rejection).

    References
    ----------
    Yatchew, A. (1997). An elementary estimator of the partial linear
    model. Economics Letters 57, 135-143.
    de Chaisemartin et al. (2026), Theorem 7 / Equation 29.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha}.")

    d_arr = _validate_1d_numeric(d, "d")
    dy_arr = _validate_1d_numeric(dy, "dy")
    if d_arr.shape[0] != dy_arr.shape[0]:
        raise ValueError(
            f"d and dy must have the same length; got d.shape={d_arr.shape}, "
            f"dy.shape={dy_arr.shape}."
        )

    G = int(d_arr.shape[0])
    critical_value = float(stats.norm.ppf(1.0 - alpha))

    if G < _MIN_G_YATCHEW:
        warnings.warn(
            f"yatchew_hr_test: G = {G} is below the minimum {_MIN_G_YATCHEW} "
            f"(need at least 2 sorted differences). Returning NaN result.",
            UserWarning,
            stacklevel=2,
        )
        return YatchewTestResults(
            t_stat_hr=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=alpha,
            critical_value=critical_value,
            sigma2_lin=float("nan"),
            sigma2_diff=float("nan"),
            sigma2_W=float("nan"),
            n_obs=G,
        )

    # Tie / constant-dose front-door guard. Yatchew's difference-based
    # variance estimator uses adjacent differences of dy sorted by d;
    # under tied doses the within-tie ordering is arbitrary (stable sort
    # falls back to input row order), so the statistic becomes
    # non-methodological and order-dependent. Reject at the front door
    # with a UserWarning + NaN result rather than silently permuting.
    # Mass-point designs and other tied-dose panels should use
    # `stute_test` instead (its tie-safe CvM handles ties correctly).
    n_unique_d = int(np.unique(d_arr).shape[0])
    if n_unique_d < G:
        n_dups = G - n_unique_d
        warnings.warn(
            f"yatchew_hr_test: d contains {n_dups} duplicate value(s) "
            f"(only {n_unique_d} distinct dose values out of G={G}); "
            f"the difference-based variance estimator is not well-defined "
            f"under ties because adjacent-difference statistics depend on "
            f"arbitrary within-tie row ordering. Use stute_test() instead "
            f"(its tie-safe CvM handles ties correctly). Returning NaN result.",
            UserWarning,
            stacklevel=2,
        )
        return YatchewTestResults(
            t_stat_hr=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=alpha,
            critical_value=critical_value,
            sigma2_lin=float("nan"),
            sigma2_diff=float("nan"),
            sigma2_W=float("nan"),
            n_obs=G,
        )

    _, _, eps = _fit_ols_intercept_slope(d_arr, dy_arr)
    sigma2_lin = float(np.mean(eps * eps))

    # Numerically exact linear fit: same short-circuit as `stute_test`.
    # Assumption 8 holds to IEEE precision; the Yatchew statistic is
    # formally -inf (finite-negative numerator over zero denominator),
    # which maps to p = 1 under the one-sided standard-normal critical
    # value. Short-circuit so FP noise in ``sigma4_W`` cannot produce a
    # spuriously large finite ``T_hr``. Comparison is purely relative
    # against CENTERED TSS - translation- AND scale-invariant.
    eps_norm_sq = float(np.sum(eps * eps))
    dy_centered_sq = float(np.sum((dy_arr - dy_arr.mean()) ** 2))
    if dy_centered_sq <= 0.0 or eps_norm_sq <= _EXACT_LINEAR_RELATIVE_TOL * dy_centered_sq:
        # Exact-linear branch. Covers two cases:
        # - dy_centered_sq == 0: dy is constant (trivially linear).
        # - relative SSR below IEEE precision: near-exact OLS fit.
        # For reporting, compute sigma2_diff on the sorted dy (finite,
        # well-defined even in the exact-linear case).
        idx_early = np.argsort(d_arr, kind="stable")
        sigma2_diff_exact = float(np.sum(np.diff(dy_arr[idx_early]) ** 2) / (2.0 * G))
        return YatchewTestResults(
            t_stat_hr=float("-inf"),
            p_value=1.0,
            reject=False,
            alpha=alpha,
            critical_value=critical_value,
            sigma2_lin=sigma2_lin,
            sigma2_diff=sigma2_diff_exact,
            sigma2_W=0.0,
            n_obs=G,
        )

    idx = np.argsort(d_arr, kind="stable")
    dy_s = dy_arr[idx]
    eps_s = eps[idx]

    diff_dy = np.diff(dy_s)  # length G - 1
    # Paper-literal divisor: 2G (NOT 2(G-1)). This matches paper review
    # line 168: sigma2_diff := (1/(2G)) * sum((dy_{(g)} - dy_{(g-1)})^2).
    sigma2_diff = float(np.sum(diff_dy * diff_dy) / (2.0 * G))

    # sigma4_W = (1/(G-1)) * sum(eps_(g)^2 * eps_(g-1)^2) using np.mean
    # which divides by the length of the input (G-1 here). Matches paper
    # review line 171.
    sigma4_W = float(np.mean(eps_s[1:] ** 2 * eps_s[:-1] ** 2))
    if sigma4_W <= 0.0:
        # sigma4_W = 0 AFTER the exact-linear short-circuit means OLS
        # residuals are NOT zero (the shortcut already caught that case)
        # but every adjacent pair of sorted squared residuals contains a
        # zero (e.g. residuals alternate zero / nonzero after sort).
        # The formal test statistic is ±inf depending on the sign of the
        # numerator ``sigma2_lin - sigma2_diff``; mapping every such case
        # to p=1 (as an earlier revision did) can flip a legitimate
        # rejection into a fail-to-reject.
        warnings.warn(
            f"yatchew_hr_test: sigma4_W = 0 with non-zero residuals "
            f"(sigma2_lin = {sigma2_lin:.6g}, sigma2_diff = {sigma2_diff:.6g}); "
            f"the formal test statistic is infinite. Returning the "
            f"sign-aware limit decision.",
            UserWarning,
            stacklevel=2,
        )
        numerator = sigma2_lin - sigma2_diff
        if numerator > 0.0:
            # T_hr -> +inf: reject (far into right tail).
            t_stat_hr_val = float("inf")
            p_value_val = 0.0
            reject_val = True
        elif numerator < 0.0:
            # T_hr -> -inf: fail-to-reject.
            t_stat_hr_val = float("-inf")
            p_value_val = 1.0
            reject_val = False
        else:
            # 0/0: genuinely indeterminate.
            t_stat_hr_val = float("nan")
            p_value_val = float("nan")
            reject_val = False
        return YatchewTestResults(
            t_stat_hr=t_stat_hr_val,
            p_value=p_value_val,
            reject=reject_val,
            alpha=alpha,
            critical_value=critical_value,
            sigma2_lin=sigma2_lin,
            sigma2_diff=sigma2_diff,
            sigma2_W=0.0,
            n_obs=G,
        )
    sigma2_W = float(np.sqrt(sigma4_W))

    t_stat_hr = float(np.sqrt(G) * (sigma2_lin - sigma2_diff) / sigma2_W)
    p_value = float(1.0 - stats.norm.cdf(t_stat_hr))
    reject = t_stat_hr >= critical_value

    return YatchewTestResults(
        t_stat_hr=t_stat_hr,
        p_value=p_value,
        reject=bool(reject),
        alpha=alpha,
        critical_value=critical_value,
        sigma2_lin=sigma2_lin,
        sigma2_diff=sigma2_diff,
        sigma2_W=sigma2_W,
        n_obs=G,
    )


def did_had_pretest_workflow(
    data: pd.DataFrame,
    outcome_col: str,
    dose_col: str,
    time_col: str,
    unit_col: str,
    first_treat_col: Optional[str] = None,
    alpha: float = 0.05,
    n_bootstrap: int = 999,
    seed: Optional[int] = None,
) -> HADPretestReport:
    """Run a PARTIAL HAD pre-test workflow on a two-period panel
    (paper Section 4.2-4.3, steps 1 and 3 only; step 2 deferred).

    Phase 3 scope runs:

    - Step 1: :func:`qug_test` (``H_0: d_lower = 0``, Theorem 4).
    - Step 3: :func:`stute_test` and :func:`yatchew_hr_test` (linearity
      of ``E[ΔY | D_2]``, Assumption 8).

    Phase 3 does **NOT** run step 2 (Assumption 7 pre-trends test via
    paper Equation 18); that joint cross-horizon Stute variant is
    deferred to a follow-up patch. Users should continue to perform their
    own pre-trends / placebo analysis until the follow-up ships. The
    returned :class:`HADPretestReport` verdict explicitly flags the
    Assumption 7 gap when all implemented diagnostics fail-to-reject, so
    callers do not receive a misleading "TWFE safe" signal.

    The workflow reduces the panel to unit-level first differences using
    the Phase 2a validator + aggregator, then calls the three tests with
    shared ``alpha`` and a single-source seed passthrough (``seed`` is
    forwarded to :func:`stute_test` only; QUG and Yatchew are
    deterministic).

    Parameters
    ----------
    data : pd.DataFrame
        Balanced two-period HAD panel. The dose column must be 0 for all
        units at the pre-period (HAD no-unit-untreated pre-period
        contract).
    outcome_col, dose_col, time_col, unit_col : str
        Column names.
    first_treat_col : str or None, default None
        Optional first-treatment-period column for cross-validation
        (see :func:`HeterogeneousAdoptionDiD.fit`).
    alpha : float, default 0.05
    n_bootstrap : int, default 999
        Replication count for :func:`stute_test`.
    seed : int or None, default None
        Seed forwarded to :func:`stute_test` only.

    Returns
    -------
    HADPretestReport

    Notes
    -----
    Phase 3 scope is two-period overall-path only. For multi-period
    panels, slice to ``(F - 1, F)`` before calling. A future patch will
    add a multi-period dispatch and the joint Equation 18 pre-trend
    Stute test.

    References
    ----------
    de Chaisemartin et al. (2026), Section 4.2-4.3, Theorem 4, Appendix D,
    Theorem 7.
    """
    t_pre, t_post = _validate_had_panel(
        data, outcome_col, dose_col, time_col, unit_col, first_treat_col
    )
    d_arr, dy_arr, _, _ = _aggregate_first_difference(
        data,
        outcome_col,
        dose_col,
        time_col,
        unit_col,
        t_pre,
        t_post,
        cluster_col=None,  # pretests do not use cluster-robust SE
    )

    qug_res = qug_test(d_arr, alpha=alpha)
    stute_res = stute_test(d_arr, dy_arr, alpha=alpha, n_bootstrap=n_bootstrap, seed=seed)
    yatchew_res = yatchew_hr_test(d_arr, dy_arr, alpha=alpha)

    # `all_pass` must be conclusive under the paper's four-step workflow
    # (step 1 QUG + step 3 linearity via Stute OR Yatchew):
    #   - QUG must produce a finite p-value (step 1 is required).
    #   - At least ONE of Stute / Yatchew must produce a finite p-value
    #     (step 3 accepts either; the paper's wording is "Stute OR
    #     Yatchew"). This accommodates common QUG-style panels with
    #     repeated d=0 units, where Yatchew's duplicate-dose guard trips
    #     but Stute's tie-safe CvM still produces a conclusive result.
    #   - No conclusive test may reject. NaN-p tests have reject=False by
    #     convention, so the OR across `.reject` naturally counts only
    #     the conclusive rejections.
    qug_conclusive = bool(np.isfinite(qug_res.p_value))
    linearity_conclusive = bool(np.isfinite(stute_res.p_value) or np.isfinite(yatchew_res.p_value))
    any_reject = qug_res.reject or stute_res.reject or yatchew_res.reject
    all_pass = bool(qug_conclusive and linearity_conclusive and not any_reject)
    verdict = _compose_verdict(qug_res, stute_res, yatchew_res)

    return HADPretestReport(
        qug=qug_res,
        stute=stute_res,
        yatchew=yatchew_res,
        all_pass=all_pass,
        verdict=verdict,
        alpha=alpha,
        n_obs=int(d_arr.shape[0]),
    )
