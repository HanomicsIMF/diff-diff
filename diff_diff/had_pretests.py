"""Pre-test diagnostics for the HeterogeneousAdoptionDiD estimator.

Paper Section 4 (de Chaisemartin, Ciccia, D'Haultfoeuille, Knau 2026,
arXiv:2405.04465v6) prescribes a four-step pre-testing workflow for TWFE
validity in HADs. This module ships the tests and the composite workflow:

Single-horizon tests:

1. :func:`qug_test` - order-statistic ratio test of the support infimum
   ``H_0: d_lower = 0`` (paper Theorem 4). Closed-form, tuning-free.
2. :func:`stute_test` - Cramer-von Mises cusum test of linearity of
   ``E[ΔY | D_2]`` with Mammen (1993) wild bootstrap p-value (paper
   Appendix D).
3. :func:`yatchew_hr_test` - heteroskedasticity-robust variance-ratio
   linearity test (paper Theorem 7 / Equation 29). Feasible at
   ``G >= 100k``.

Joint / multi-period tests (Phase 3 follow-up):

4. :func:`stute_joint_pretest` - residuals-in core that generalizes the
   single-horizon Stute CvM to K horizons with shared-η wild bootstrap
   and sum-of-CvMs aggregation (Delgado 1993; Escanciano 2006).
5. :func:`joint_pretrends_test` - data-in wrapper for the mean-
   independence null (paper step 2 pre-trends across pre-period
   placebos, Section 4.2 footnote 6 + Section 4.3 paragraph 1).
6. :func:`joint_homogeneity_test` - data-in wrapper for the linearity
   null across post-periods (paper Section 4.3 joint extension,
   page 32).

Composite workflow:

:func:`did_had_pretest_workflow` has two dispatch modes:

- ``aggregate="overall"`` (default, two-period panel): runs steps 1 + 3
  via :func:`qug_test` + :func:`stute_test` + :func:`yatchew_hr_test`.
  Paper step 2 is NOT run on this path (a two-period panel has no pre-
  period placebo); the verdict explicitly flags the Assumption 7 gap
  via the ``"paper step 2 deferred"`` caveat so callers do not get an
  unconditional "TWFE safe" signal.
- ``aggregate="event_study"`` (multi-period panel, >= 3 periods): runs
  QUG at ``F`` + joint pre-trends Stute across earlier pre-periods +
  joint homogeneity-linearity Stute across post-periods. Closes the
  paper step-2 gap and does NOT emit the step-2-deferred caveat in the
  verdict when at least one earlier pre-period is available. The
  step-3 alternative (Yatchew-HR linearity) is subsumed by joint Stute
  on this path; the paper does not derive a joint Yatchew variant, so
  users who need Yatchew robustness under multi-period data can call
  :func:`yatchew_hr_test` on each ``(base, post)`` pair manually.
  (Step 4 in the paper's workflow is the decision itself - "use TWFE
  if none of the tests rejects" - not a separate test.)

Eq. 18 linear-trend detrending (paper Section 5.2 Pierce-Schott
application, published p=0.51) is the one remaining deferred item;
tracked in ``TODO.md`` and slated for Phase 4 alongside the replication
harness. See ``docs/methodology/REGISTRY.md`` for the full algorithm
narrative, invariants, and deviation notes.
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
    _validate_had_panel_event_study,
)
from diff_diff.utils import _generate_mammen_weights

__all__ = [
    "QUGTestResults",
    "StuteTestResults",
    "YatchewTestResults",
    "StuteJointResult",
    "HADPretestReport",
    "qug_test",
    "stute_test",
    "yatchew_hr_test",
    "stute_joint_pretest",
    "joint_pretrends_test",
    "joint_homogeneity_test",
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
class StuteJointResult:
    """Result of :func:`stute_joint_pretest` (joint Cramer-von Mises across horizons).

    Aggregates the per-horizon Stute (1997) CvM statistic into a joint
    specification test: ``S_joint = sum_k S_k``, where ``S_k`` is the
    single-horizon CvM on residuals ``eps_{g,k}``. Inference is via
    Mammen (1993) wild bootstrap with a **shared** multiplier ``eta_g``
    across horizons per unit (Delgado-Manteiga 2001; Hlavka-Huskova 2020)
    to preserve the unit-level dependence structure of the vector-valued
    empirical process.

    Two nulls are supported via the thin wrappers
    :func:`joint_pretrends_test` (mean-independence: ``E[Y_t - Y_base | D]
    = mu_t``, design matrix ``[1]``) and :func:`joint_homogeneity_test`
    (linearity: ``E[Y_t - Y_base | D_t] = beta_{0,t} + beta_{fe,t} * D``,
    design matrix ``[1, D]``). Eq 18 linear-trend detrending (paper
    Section 5.2 Pierce-Schott application) is a Phase 4 follow-up.

    Attributes
    ----------
    cvm_stat_joint : float
        Joint statistic ``S_joint = sum_k S_k``. NaN on NaN-propagation.
    p_value : float
        Bootstrap p-value ``(1 + sum(S*_b >= S_joint)) / (B + 1)``. NaN
        when the statistic is NaN. ``1.0`` when the per-horizon exact-
        linear short-circuit fires (all horizons machine-exact linear).
    reject : bool
        ``True`` iff ``p_value <= alpha``. Always ``False`` on NaN.
    alpha : float
        Significance level.
    horizon_labels : list of str
        Horizon identifiers as ``str(t)`` for each period. **String
        identity only** - NOT a chronological ordering key. Callers who
        need chronological order should preserve the original period
        values alongside (a downstream plotter sorting labels
        lexicographically will misorder e.g.
        ``["2003-Q10", "2003-Q2", ...]``).
    per_horizon_stats : dict[str, float]
        ``{label: S_k}`` diagnostic. Per-horizon p-values are NOT
        exposed (decomposing the joint bootstrap into K independent
        loops is a K-fold memory/time cost; deferred). Callers who need
        per-horizon p-values can call :func:`stute_test` separately on
        each (period, residual) pair.

        On NaN-propagation (any horizon has NaN input), this dict is
        preserved with ``{label: np.nan for label in horizon_labels}``,
        NOT an empty dict, NOT a partial dict: the keys carry diagnostic
        value (which horizons were attempted), the NaN values signal
        non-propagation.
    n_bootstrap : int
    n_obs : int
        Number of units ``G``.
    n_horizons : int
    seed : int or None
    null_form : str
        ``"mean_independence"`` (from :func:`joint_pretrends_test`) or
        ``"linearity"`` (from :func:`joint_homogeneity_test`).
        ``"custom"`` when called directly via :func:`stute_joint_pretest`
        without a wrapper.
    exact_linear_short_circuited : bool
        ``True`` when every horizon's residual SSR to centered TSS ratio
        is below :data:`_EXACT_LINEAR_RELATIVE_TOL`; bootstrap is
        skipped and ``p_value = 1.0``. The per-horizon check ensures a
        single degenerate horizon does not collapse the joint test when
        other horizons have nontrivial residuals.
    """

    cvm_stat_joint: float
    p_value: float
    reject: bool
    alpha: float
    horizon_labels: list
    per_horizon_stats: Dict[str, float]
    n_bootstrap: int
    n_obs: int
    n_horizons: int
    seed: Optional[int]
    null_form: str
    exact_linear_short_circuited: bool

    def __repr__(self) -> str:
        return (
            f"StuteJointResult(cvm_stat_joint={self.cvm_stat_joint:.4f}, "
            f"p_value={self.p_value:.4f}, reject={self.reject}, "
            f"n_horizons={self.n_horizons}, null_form={self.null_form!r}, "
            f"n_obs={self.n_obs})"
        )

    def summary(self) -> str:
        """Formatted summary table."""
        width = 64
        per_horizon_lines = [
            f"  {label:<20} {stat:>20.4f}" for label, stat in self.per_horizon_stats.items()
        ]
        null_label = {
            "mean_independence": "mean-independence (pre-trends)",
            "linearity": "linearity (post-homogeneity)",
        }.get(self.null_form, self.null_form)
        lines = [
            "=" * width,
            f"Joint Stute CvM test ({null_label})".center(width),
            "=" * width,
            f"{'Joint CvM statistic:':<30} {self.cvm_stat_joint:>20.4f}",
            f"{'Bootstrap p-value:':<30} {self.p_value:>20.4f}",
            f"{'Reject H_0:':<30} {str(self.reject):>20}",
            f"{'alpha:':<30} {self.alpha:>20.4f}",
            f"{'Bootstrap replications:':<30} {self.n_bootstrap:>20}",
            f"{'Horizons:':<30} {self.n_horizons:>20}",
            f"{'Observations:':<30} {self.n_obs:>20}",
            f"{'Seed:':<30} {str(self.seed):>20}",
            f"{'Exact-linear short-circuit:':<30} " f"{str(self.exact_linear_short_circuited):>20}",
            "-" * width,
            "Per-horizon statistics:",
            *per_horizon_lines,
            "=" * width,
        ]
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Return results as a JSON-safe dict."""
        return {
            "test": "stute_joint",
            "cvm_stat_joint": _json_safe_scalar(self.cvm_stat_joint),
            "p_value": _json_safe_scalar(self.p_value),
            "reject": bool(self.reject),
            "alpha": float(self.alpha),
            "horizon_labels": [str(label) for label in self.horizon_labels],
            "per_horizon_stats": {
                str(k): _json_safe_scalar(v) for k, v in self.per_horizon_stats.items()
            },
            "n_bootstrap": int(self.n_bootstrap),
            "n_obs": int(self.n_obs),
            "n_horizons": int(self.n_horizons),
            "seed": None if self.seed is None else int(self.seed),
            "null_form": str(self.null_form),
            "exact_linear_short_circuited": bool(self.exact_linear_short_circuited),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a one-row DataFrame of the top-level result fields."""
        return pd.DataFrame(
            [
                {
                    "test": "stute_joint",
                    "cvm_stat_joint": _json_safe_scalar(self.cvm_stat_joint),
                    "p_value": _json_safe_scalar(self.p_value),
                    "reject": bool(self.reject),
                    "alpha": float(self.alpha),
                    "n_bootstrap": int(self.n_bootstrap),
                    "n_obs": int(self.n_obs),
                    "n_horizons": int(self.n_horizons),
                    "null_form": str(self.null_form),
                }
            ]
        )


@dataclass
class HADPretestReport:
    """Composite output of :func:`did_had_pretest_workflow`.

    Two dispatch shapes, distinguished by :attr:`aggregate`:

    ``aggregate="overall"`` (default, two-period panel): bundles paper
    steps 1 (QUG) and 3 (linearity via Stute + Yatchew-HR) on a
    two-period first-differenced sample. Step 2 (Assumption 7 pre-trends)
    is NOT implemented on this path and is explicitly flagged in the
    verdict; callers must run pre-trends separately.

    ``aggregate="event_study"`` (multi-period panel, >= 3 periods):
    bundles QUG + joint pre-trends Stute + joint homogeneity-linearity
    Stute. The joint Stute variants close the paper step-2 gap; the
    event-study verdict does NOT emit the "paper step 2 deferred"
    caveat. Step 3 adjudication uses joint Stute only - no joint Yatchew
    variant exists because the paper does not derive one; users who need
    Yatchew robustness under multi-period data can run
    :func:`yatchew_hr_test` on each (base, post) pair manually.

    Attributes
    ----------
    qug : QUGTestResults
        Always populated.
    stute : StuteTestResults or None
        Populated when ``aggregate == "overall"``; ``None`` when
        ``aggregate == "event_study"``.
    yatchew : YatchewTestResults or None
        Populated when ``aggregate == "overall"``; ``None`` when
        ``aggregate == "event_study"``.
    pretrends_joint : StuteJointResult or None
        Populated when ``aggregate == "event_study"`` and at least one
        earlier pre-period exists; ``None`` on the overall path or when
        only the immediate base pre-period is available.
    homogeneity_joint : StuteJointResult or None
        Populated when ``aggregate == "event_study"``; ``None`` on the
        overall path.
    all_pass : bool
        On the overall path: same Phase 3 semantics - True iff QUG is
        conclusive AND at least one of Stute/Yatchew is conclusive AND
        no conclusive test rejects. On the event-study path: True iff
        ``np.isfinite(qug.p_value)``,
        ``pretrends_joint is not None and
        np.isfinite(pretrends_joint.p_value)``,
        ``np.isfinite(homogeneity_joint.p_value)``, AND none of the
        three rejects. Mirrors Phase 3's ``bool(np.isfinite(p_value))``
        convention - no ``.conclusive()`` helper on any result dataclass.
    verdict : str
        Human-readable classification. Paper rule applies symmetrically:
        TWFE is admissible only if NONE of the implemented tests
        rejects. Conclusive rejections are the primary verdict;
        unresolved steps append as ``"; additional steps unresolved:
        ..."`` rather than replacing the rejection.
    alpha : float
    n_obs : int
        Unit count. For overall: units after two-period first-difference
        aggregation. For event_study: units after balanced-panel
        validation and (if applicable) last-cohort auto-filter.
    aggregate : str
        ``"overall"`` or ``"event_study"``. Determines which component
        fields are populated and which branch of serialization methods
        to render.
    """

    qug: QUGTestResults
    stute: Optional[StuteTestResults]
    yatchew: Optional[YatchewTestResults]
    all_pass: bool
    verdict: str
    alpha: float
    n_obs: int
    pretrends_joint: Optional[StuteJointResult] = None
    homogeneity_joint: Optional[StuteJointResult] = None
    aggregate: str = "overall"

    def __repr__(self) -> str:
        # Preserve Phase 3 repr bit-exactly on the overall path. The
        # aggregate kwarg is only surfaced on the event-study path so
        # downstream consumers comparing repr strings on two-period
        # reports see identical output.
        if self.aggregate == "event_study":
            return (
                f"HADPretestReport(aggregate={self.aggregate!r}, "
                f"all_pass={self.all_pass}, "
                f"verdict={self.verdict!r}, n_obs={self.n_obs})"
            )
        return (
            f"HADPretestReport(all_pass={self.all_pass}, "
            f"verdict={self.verdict!r}, n_obs={self.n_obs})"
        )

    def summary(self) -> str:
        """Formatted summary of all tests and the verdict."""
        width = 72
        # Preserve Phase 3 summary bit-exactly on the overall path. The
        # `aggregate: ...` header line is only rendered on the event-
        # study path; two-period reports produce the Phase 3 layout.
        if self.aggregate == "event_study":
            header = [
                "=" * width,
                "HAD pre-test workflow".center(width),
                f"aggregate: {self.aggregate}".center(width),
                "=" * width,
                self.qug.summary(),
                "",
            ]
            if self.pretrends_joint is not None:
                body = [self.pretrends_joint.summary(), ""]
            else:
                body = [
                    "(joint pre-trends skipped - no earlier pre-period)",
                    "",
                ]
            if self.homogeneity_joint is not None:
                body += [self.homogeneity_joint.summary(), ""]
        else:
            # aggregate == "overall" - Phase 3 layout preserved.
            header = [
                "=" * width,
                "HAD pre-test workflow".center(width),
                "=" * width,
                self.qug.summary(),
                "",
            ]
            body = []
            if self.stute is not None:
                body += [self.stute.summary(), ""]
            if self.yatchew is not None:
                body += [self.yatchew.summary(), ""]
        footer = [
            "=" * width,
            f"{'All pass:':<30} {str(self.all_pass):>40}",
            f"Verdict: {self.verdict}",
            "=" * width,
        ]
        return "\n".join(header + body + footer)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-safe nested dict of the full report.

        On ``aggregate="overall"``, the output schema is bit-exact with
        Phase 3 (``{qug, stute, yatchew, all_pass, verdict, alpha,
        n_obs}``) - no new keys, no aggregate field. On
        ``aggregate="event_study"``, the output carries ``aggregate``,
        ``pretrends_joint``, ``homogeneity_joint`` and omits the
        ``None``-valued ``stute`` / ``yatchew`` keys entirely.
        """
        if self.aggregate == "event_study":
            return {
                "aggregate": str(self.aggregate),
                "qug": self.qug.to_dict(),
                "pretrends_joint": (
                    None if self.pretrends_joint is None else self.pretrends_joint.to_dict()
                ),
                "homogeneity_joint": (
                    None if self.homogeneity_joint is None else self.homogeneity_joint.to_dict()
                ),
                "all_pass": bool(self.all_pass),
                "verdict": str(self.verdict),
                "alpha": float(self.alpha),
                "n_obs": int(self.n_obs),
            }
        # aggregate == "overall" - Phase 3 schema preserved bit-exactly,
        # including key order and the absence of the aggregate field.
        return {
            "qug": self.qug.to_dict(),
            "stute": None if self.stute is None else self.stute.to_dict(),
            "yatchew": None if self.yatchew is None else self.yatchew.to_dict(),
            "all_pass": bool(self.all_pass),
            "verdict": str(self.verdict),
            "alpha": float(self.alpha),
            "n_obs": int(self.n_obs),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy 3-row DataFrame (one row per implemented test).

        Columns (stable across aggregates):
        ``[test, statistic_name, statistic_value, p_value, reject, alpha,
        n_obs]``. Row identifiers vary by aggregate:

        - ``aggregate="overall"``: rows are ``qug``, ``stute``,
          ``yatchew_hr`` (Phase 3 schema, unchanged).
        - ``aggregate="event_study"``: rows are ``qug``,
          ``pretrends_joint``, ``homogeneity_joint``.

        Rows for ``None``-valued components (e.g. ``pretrends_joint`` when
        no earlier pre-period exists) are emitted with NaN statistic
        values and ``reject=False`` to preserve the 3-row shape.
        """
        qug_row = {
            "test": "qug",
            "statistic_name": "t_stat",
            "statistic_value": _json_safe_scalar(self.qug.t_stat),
            "p_value": _json_safe_scalar(self.qug.p_value),
            "reject": bool(self.qug.reject),
            "alpha": float(self.qug.alpha),
            "n_obs": int(self.qug.n_obs),
        }
        if self.aggregate == "event_study":
            pre_row = self._joint_row_or_nan("pretrends_joint", self.pretrends_joint)
            hom_row = self._joint_row_or_nan("homogeneity_joint", self.homogeneity_joint)
            rows = [qug_row, pre_row, hom_row]
        else:
            stute_row = (
                {
                    "test": "stute",
                    "statistic_name": "cvm_stat",
                    "statistic_value": _json_safe_scalar(self.stute.cvm_stat),
                    "p_value": _json_safe_scalar(self.stute.p_value),
                    "reject": bool(self.stute.reject),
                    "alpha": float(self.stute.alpha),
                    "n_obs": int(self.stute.n_obs),
                }
                if self.stute is not None
                else {
                    "test": "stute",
                    "statistic_name": "cvm_stat",
                    "statistic_value": float("nan"),
                    "p_value": float("nan"),
                    "reject": False,
                    "alpha": float(self.alpha),
                    "n_obs": int(self.n_obs),
                }
            )
            yatchew_row = (
                {
                    "test": "yatchew_hr",
                    "statistic_name": "t_stat_hr",
                    "statistic_value": _json_safe_scalar(self.yatchew.t_stat_hr),
                    "p_value": _json_safe_scalar(self.yatchew.p_value),
                    "reject": bool(self.yatchew.reject),
                    "alpha": float(self.yatchew.alpha),
                    "n_obs": int(self.yatchew.n_obs),
                }
                if self.yatchew is not None
                else {
                    "test": "yatchew_hr",
                    "statistic_name": "t_stat_hr",
                    "statistic_value": float("nan"),
                    "p_value": float("nan"),
                    "reject": False,
                    "alpha": float(self.alpha),
                    "n_obs": int(self.n_obs),
                }
            )
            rows = [qug_row, stute_row, yatchew_row]
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

    def _joint_row_or_nan(
        self, test_label: str, joint: Optional[StuteJointResult]
    ) -> Dict[str, Any]:
        """Build a to_dataframe row for a joint-Stute component.

        When ``joint`` is ``None`` (e.g. pretrends_joint skipped because
        no earlier pre-period), emit a NaN row preserving the 3-row
        shape for downstream plotting.
        """
        if joint is None:
            return {
                "test": test_label,
                "statistic_name": "cvm_stat_joint",
                "statistic_value": float("nan"),
                "p_value": float("nan"),
                "reject": False,
                "alpha": float(self.alpha),
                "n_obs": int(self.n_obs),
            }
        return {
            "test": test_label,
            "statistic_name": "cvm_stat_joint",
            "statistic_value": _json_safe_scalar(joint.cvm_stat_joint),
            "p_value": _json_safe_scalar(joint.p_value),
            "reject": bool(joint.reject),
            "alpha": float(joint.alpha),
            "n_obs": int(joint.n_obs),
        }


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

    Paper logic: TWFE is admissible only if NONE of the implemented
    tests rejects. A conclusive rejection must therefore never be hidden
    by a purely-inconclusive verdict just because another step is NaN -
    it is reported as the primary outcome and any unresolved steps are
    appended as a suffix.

    Priority:

    1. Collect all rejection reasons from CONCLUSIVE tests. If any
       conclusive test rejected, that is the primary verdict. Unresolved
       steps (QUG NaN, or BOTH linearity tests NaN) are appended as
       ``"; additional steps unresolved: ..."`` rather than replacing
       the rejection.
    2. If no conclusive test rejected but a required step is unresolved,
       return a pure ``"inconclusive - ..."`` verdict naming the
       unresolved step(s).
    3. Otherwise (all required steps conclusive and none reject),
       return the partial-workflow fail-to-reject verdict flagging the
       Assumption 7 gap, with a ``" (Yatchew NaN - skipped)"`` suffix
       when ONE linearity test was NaN and the other was conclusive.
    """
    qug_ok = bool(np.isfinite(qug.p_value))
    stute_ok = bool(np.isfinite(stute.p_value))
    yatchew_ok = bool(np.isfinite(yatchew.p_value))

    # Rejections from conclusive tests only. NaN-p tests have reject=False
    # by convention, so the ``ok and reject`` guard is defensive.
    qug_rej = qug_ok and qug.reject
    stute_rej = stute_ok and stute.reject
    yatchew_rej = yatchew_ok and yatchew.reject

    reasons = []
    if qug_rej:
        reasons.append("support infimum rejected - continuous_at_zero design invalid (QUG)")
    if stute_rej or yatchew_rej:
        which = ",".join(
            name for name, rejected in (("Stute", stute_rej), ("Yatchew", yatchew_rej)) if rejected
        )
        reasons.append(f"linearity rejected - heterogeneity bias ({which})")

    # Unresolved steps: QUG is required; step 3 requires at least one
    # conclusive linearity test.
    unresolved = []
    if not qug_ok:
        unresolved.append("QUG NaN")
    if not stute_ok and not yatchew_ok:
        unresolved.append("both Stute and Yatchew linearity tests NaN")

    if reasons:
        # A conclusive rejection is the primary outcome. Append any
        # unresolved-step note rather than replacing the rejection.
        verdict = "; ".join(reasons)
        if unresolved:
            verdict += "; additional steps unresolved: " + "; ".join(unresolved)
        return verdict

    if unresolved:
        return "inconclusive - " + "; ".join(unresolved)

    # All required steps conclusive, none reject. Note any single skipped
    # linearity test (the OTHER linearity test was conclusive and
    # fail-to-reject, so step 3 IS resolved).
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


# =============================================================================
# Public test functions
# =============================================================================


def qug_test(
    d: np.ndarray,
    alpha: float = 0.05,
    *,
    survey: Any = None,
    weights: Optional[np.ndarray] = None,
) -> QUGTestResults:
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
    survey : SurveyDesign or None, keyword-only, default None
        Permanently rejected with ``NotImplementedError`` (Phase 4.5 C0
        decision gate). See *Notes -- Survey/weighted data*.
    weights : np.ndarray or None, keyword-only, default None
        Permanently rejected with ``NotImplementedError`` (Phase 4.5 C0
        decision gate). See *Notes -- Survey/weighted data*.

    Returns
    -------
    QUGTestResults
        Result dataclass with ``t_stat``, ``p_value``, ``reject``, and
        sample metadata.

    Raises
    ------
    ValueError
        If ``d`` is not 1D numeric or contains NaN, or if ``alpha`` is
        not in ``(0, 1)``, or if ``survey`` and ``weights`` are both
        non-None (mutex).
    NotImplementedError
        If ``survey`` or ``weights`` is non-None. See
        *Notes -- Survey/weighted data*.

    Notes
    -----
    Tie-break: when ``D_{(1)} == D_{(2)}`` the statistic is undefined.
    The test returns ``t_stat=NaN, p_value=NaN, reject=False`` with a
    ``UserWarning`` rather than raising.

    Survey/weighted data: QUG is permanently deferred under survey-weighted
    or pweight inputs (Phase 4.5 C0 decision gate, 2026-04). The test
    statistic uses extreme order statistics ``(D_{(1)}, D_{(2)})``, which
    are NOT smooth functionals of the empirical CDF -- standard survey
    machinery (Binder TSL linearization, Rao-Wu rescaled bootstrap) does
    not yield a calibrated test, and under cluster sampling the
    ``Exp(1)/Exp(1)`` limit law's independence assumption breaks. The
    extreme-value-theory-under-unequal-probability-sampling literature
    (Quintos et al. 2001, Beirlant et al.) addresses tail-index
    estimation, not boundary tests; no off-the-shelf survey-aware QUG
    exists. Use joint Stute via :func:`did_had_pretest_workflow`
    (``aggregate="event_study"``) for survey-aware HAD pretesting once
    Phase 4.5 C ships -- Stute tests a smooth empirical-CDF functional
    and admits a Rao-Wu rescaled bootstrap. See
    ``docs/methodology/REGISTRY.md`` § "QUG Null Test" for the full
    methodology note.

    References
    ----------
    de Chaisemartin, Ciccia, D'Haultfoeuille, Knau (2026, arXiv:2405.04465v6),
    Theorem 4 and Section 4.2.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha}.")

    # Mutex on survey/weights, mirroring HeterogeneousAdoptionDiD.fit()
    # at had.py:2890 so users get a consistent error across the HAD
    # surface area.
    if survey is not None and weights is not None:
        raise ValueError(
            "Pass survey=<SurveyDesign> OR weights=<array>, not both. "
            "qug_test does not yet accept either kwarg (Phase 4.5 C0 "
            "decision gate); see the NotImplementedError below for the "
            "methodology rationale."
        )

    # Phase 4.5 C0 decision gate: QUG-under-survey is permanently deferred.
    # Extreme-order-statistic functionals are not smooth in the empirical
    # CDF, so standard survey machinery (Binder TSL linearization, Rao-Wu
    # rescaled bootstrap) does not provide a calibrated test. See
    # REGISTRY.md § "QUG Null Test" for the full methodology note.
    if survey is not None or weights is not None:
        raise NotImplementedError(
            "qug_test does not support survey= / weights= kwargs.\n"
            "\n"
            "QUG (de Chaisemartin et al. 2026, Theorem 4) tests "
            "H_0: d_lower = 0 via the ratio of the two smallest order "
            "statistics, T = D_(1) / (D_(2) - D_(1)). "
            "Extreme-order-statistic functionals are not smooth in the "
            "empirical CDF, so standard survey machinery (Binder "
            "linearization, Rao-Wu rescaled bootstrap) does not provide "
            "a calibrated test. Under cluster sampling the Exp(1)/Exp(1) "
            "limit law's independence assumption breaks. The literature "
            "on extreme-value theory under unequal-probability sampling "
            "(Quintos et al. 2001, Beirlant et al.) addresses tail-index "
            "estimation, not boundary tests; no off-the-shelf "
            "survey-aware QUG exists.\n"
            "\n"
            "For survey-aware HAD pretesting, use joint Stute (Phase 4.5 "
            "C, planned) via did_had_pretest_workflow(..., survey=..., "
            "aggregate=...). Stute tests a smooth empirical-CDF "
            "functional and admits a Rao-Wu rescaled bootstrap. See "
            "docs/methodology/REGISTRY.md § 'QUG Null Test' for the "
            "full methodology note."
        )

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
        lengths, if any ``d`` value is negative (paper Section 2 HAD
        support restriction), if ``alpha`` is outside ``(0, 1)``, or if
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
    # HAD support restriction (paper Section 2): doses must be non-negative.
    # Mirror the front-door guard from qug_test / _validate_had_panel.
    if (d_arr < 0).any():
        n_neg = int((d_arr < 0).sum())
        raise ValueError(
            f"stute_test: d contains {n_neg} negative value(s); HAD doses "
            f"must be non-negative (paper Section 2). Check your dose "
            f"column or pre-process before calling stute_test."
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
        lengths, if any ``d`` value is negative (paper Section 2 HAD
        support restriction), or if ``alpha`` is outside ``(0, 1)``.

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
    # HAD support restriction (paper Section 2): doses must be non-negative.
    # Mirror the front-door guard from qug_test / _validate_had_panel.
    if (d_arr < 0).any():
        n_neg = int((d_arr < 0).sum())
        raise ValueError(
            f"yatchew_hr_test: d contains {n_neg} negative value(s); HAD "
            f"doses must be non-negative (paper Section 2). Check your "
            f"dose column or pre-process before calling yatchew_hr_test."
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


def _validate_multi_period_panel(
    data: pd.DataFrame,
    outcome_col: str,
    dose_col: str,
    time_col: str,
    unit_col: str,
    first_treat_col: Optional[str],
) -> "tuple[Any, list, list, pd.DataFrame, Optional[Dict[str, Any]]]":
    """Validate a multi-period HAD panel for joint pre-test dispatch.

    Thin wrapper over :func:`_validate_had_panel_event_study` (had.py) that
    inherits the full contract:

    - ``first_treat_col=None`` combined with a staggered panel → raises
      ``ValueError`` (the had.py helper does NOT silently accept; it
      requires an explicit first-treatment column to identify cohorts).
    - ``first_treat_col`` provided but identifies only one cohort → no
      auto-filter, proceeds.
    - ``first_treat_col`` provided with multiple cohorts → auto-filters
      to last-cohort + never-treated, emits ``UserWarning`` with
      ``filter_info`` summary.
    - Requires ≥ 3 time periods, balanced panel, ordered time dtype, and
      the pre-period D=0 invariant across all pre-periods.

    Additional guards on top of had.py:
    - ``len(t_pre_list) >= 1`` (need ≥ 1 pre-period for joint pre-trends
      infrastructure; had.py already enforces this).
    - ``len(t_post_list) >= 1`` (need ≥ 1 post-period for joint
      homogeneity; had.py already enforces this).

    Returns the same 5-tuple as the had.py helper:
    ``(F, t_pre_list, t_post_list, data_filtered, filter_info)``.
    """
    return _validate_had_panel_event_study(
        data,
        outcome_col=outcome_col,
        dose_col=dose_col,
        time_col=time_col,
        unit_col=unit_col,
        first_treat_col=first_treat_col,
    )


def _build_period_rank(data: pd.DataFrame, time_col: str) -> Dict[Any, int]:
    """Build a ``{period_label: chronological_rank}`` map.

    For ordered categorical time columns, uses the declared category
    order so that e.g. ``["q1", "q2", "q10"]`` ranks chronologically
    even though it sorts lexically in the opposite order. For numeric
    or datetime time columns, uses natural Python `sorted` order on
    the unique period labels. Object dtypes would fall back to
    lexicographic order - callers relying on chronology with object-
    dtype labels should convert to an ordered categorical first
    (this mirrors the contract in ``_validate_had_panel_event_study``).

    The rank map lets the joint-pretest wrappers compare period labels
    chronologically via ``rank[t1] < rank[t2]`` instead of raw Python
    ``t1 < t2``, which would silently misorder ordered-categorical
    panels (paper Appendix B.2 support contract).
    """
    time_dtype = data[time_col].dtype
    if isinstance(time_dtype, pd.CategoricalDtype) and time_dtype.ordered:
        return {c: i for i, c in enumerate(time_dtype.categories)}
    periods = sorted(data[time_col].unique())
    return {p: i for i, p in enumerate(periods)}


def _aggregate_for_joint_test(
    data: pd.DataFrame,
    outcome_col: str,
    dose_col: str,
    time_col: str,
    unit_col: str,
    horizons: list,
    base_period: Any,
) -> "tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]":
    """Aggregate a multi-period panel for a joint-Stute test.

    Builds per-horizon first differences ``dy_t = Y_{g,t} - Y_{g,base}``
    and the unit-level dose ``D_g`` for the joint-Stute test. All units
    must appear in every (horizon + base_period) period, matching the
    balanced-panel invariant of the single-period :func:`stute_test`.

    Dose extraction: ``D_g = max_t D_{g,t}`` under the HAD contract
    "once treated, stay treated with same dose". For pre-periods
    ``D_{g,t} = 0`` and for post-periods ``D_{g,t}`` is time-invariant
    per unit, so ``max`` recovers the realized post-period dose.

    Parameters
    ----------
    data : pd.DataFrame
    outcome_col, dose_col, time_col, unit_col : str
    horizons : list
        Non-empty list of period labels to build ``dy_t`` for.
        ``base_period`` must not be in ``horizons``. All ``horizons``
        and ``base_period`` must exist in the time column.
    base_period : period label
        The reference period for the first difference.

    Returns
    -------
    d_arr : np.ndarray, shape (G,)
    dy_by_horizon : dict[str, np.ndarray]
        Keys are ``str(t)`` per horizon, values are ``dy_t`` arrays of
        shape ``(G,)``. Insertion order follows ``horizons``.
    unit_ids : np.ndarray, shape (G,)
    """
    required = [outcome_col, dose_col, time_col, unit_col]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing column(s) in data: {missing}. Required: {required}.")
    if len(horizons) == 0:
        raise ValueError("horizons must be a non-empty list of period labels.")
    data_periods = set(data[time_col].unique())
    needed_periods = list(horizons) + [base_period]
    missing_periods = [t for t in needed_periods if t not in data_periods]
    if missing_periods:
        raise ValueError(
            f"Period(s) {missing_periods} not found in time_col "
            f"{time_col!r}. Available periods: "
            f"{sorted(data_periods, key=lambda x: (x is None, x))}."
        )
    if base_period in horizons:
        raise ValueError(
            f"base_period={base_period!r} must not appear in horizons " f"{list(horizons)!r}."
        )

    mask = data[time_col].isin(needed_periods)
    subset = data.loc[mask].copy()

    for col in [outcome_col, dose_col, unit_col]:
        col_series = subset[col]
        if bool(pd.isna(col_series).any()):
            n_nan = int(pd.isna(col_series).sum())
            raise ValueError(
                f"{n_nan} NaN value(s) found in column {col!r} across "
                f"periods {needed_periods}. Joint pre-test does not "
                f"silently drop rows; drop or impute before calling."
            )

    # Row-level non-negative-dose guard (paper Section 2 HAD support
    # restriction `D_{g,t} >= 0`). Must run BEFORE the groupby/max()
    # collapse below, otherwise a negative post dose would silently
    # become 0 in the per-unit dose vector (since `max(0, -d) = 0` for
    # positive d), letting the wrappers run on invalid data and
    # potentially return finite results. This is the direct-wrapper
    # equivalent of the row-level check inside
    # `_validate_had_panel_event_study`, centralized so both
    # `joint_pretrends_test` and `joint_homogeneity_test` inherit it on
    # the `n_periods < 3` fallback path that skips the validator.
    negative_dose_mask = subset[dose_col] < 0
    if bool(negative_dose_mask.any()):
        n_neg = int(negative_dose_mask.sum())
        raise ValueError(
            f"{n_neg} negative dose value(s) found in column "
            f"{dose_col!r} across periods {needed_periods}. HAD support "
            f"restriction (paper Section 2) requires D_{{g,t}} >= 0 "
            f"for every (unit, period)."
        )

    counts = subset.groupby(unit_col).size()
    n_needed = len(needed_periods)
    if (counts != n_needed).any():
        n_bad = int((counts != n_needed).sum())
        raise ValueError(
            f"Panel unbalanced across needed periods {needed_periods}: "
            f"{n_bad} unit(s) do not appear in all {n_needed} period(s). "
            f"Joint pre-test requires a balanced sub-panel."
        )

    wide_y = subset.pivot(index=unit_col, columns=time_col, values=outcome_col)
    wide_y = wide_y.sort_index()
    unit_ids = np.asarray(wide_y.index)

    base_y = wide_y[base_period].to_numpy(dtype=np.float64)
    dy_by_horizon: Dict[str, np.ndarray] = {}
    for t in horizons:
        y_t = wide_y[t].to_numpy(dtype=np.float64)
        dy_by_horizon[str(t)] = y_t - base_y

    # Dose per unit is the HAD time-invariant post-period dose:
    # D_g = max_t D_{g,t}. Critically, compute this over the FULL data,
    # not just the subset of needed_periods - for joint pre-trends,
    # needed_periods contains only pre-periods (all D=0), so taking max
    # over the subset would yield D_g = 0 for every unit and collapse
    # the CvM sort to arbitrary ties. Paper HAD convention: dose is
    # fixed per unit once treated; pre-period zero-dose is enforced by
    # the upstream validator.
    d_per_unit = data.groupby(unit_col)[dose_col].max().sort_index()
    # Align dose with the subset's unit ordering (pivot sort_index uses
    # natural unit_col order; groupby/sort_index on the full data gives
    # the same order).
    d_per_unit = d_per_unit.loc[unit_ids]
    d_arr = d_per_unit.to_numpy(dtype=np.float64)

    return d_arr, dy_by_horizon, unit_ids


def _compose_verdict_event_study(
    qug: QUGTestResults,
    pretrends_joint: Optional[StuteJointResult],
    homogeneity_joint: Optional[StuteJointResult],
) -> str:
    """Build the event-study :class:`HADPretestReport` verdict.

    Mirrors :func:`_compose_verdict` (two-period path) idiom verbatim:
    hyphen-separated ``"<concern> - <detail> (<source>)"`` reason
    strings, ``"; "`` join, ``"; additional steps unresolved: ..."``
    suffix for conclusive rejections that coexist with unresolved
    steps, lowercase concerns.

    Coverage:
    - Step 1 (QUG): always runs on the event-study path.
    - Step 2 (Assumption 7 pre-trends): runs via ``pretrends_joint``
      when at least one earlier pre-period is available. When skipped
      (only the immediate base pre-period), the verdict flags the skip
      but does NOT emit the Phase-3 "paper step 2 deferred to Phase 3
      follow-up" caveat - this PR closes that gap.
    - Step 3 (Assumption 8 linearity/homogeneity): runs via
      ``homogeneity_joint`` (joint Stute only; no joint Yatchew variant
      exists in the paper). The step-3 alternative Yatchew-HR test is
      subsumed by joint Stute on this path. (Paper step 4 is the
      decision itself - "use TWFE if none of the tests rejects" - not
      a separate diagnostic, so it has no code path here.)

    Priority:
    1. Any conclusive test rejecting → primary verdict bundles each
       rejection reason. Unresolved / skipped steps append as a suffix.
    2. No conclusive rejection but a required step unresolved →
       ``"inconclusive - ..."``.
    3. All required steps conclusive and none reject → admissible
       fail-to-reject string (Section 4 coverage).
    """
    qug_ok = bool(np.isfinite(qug.p_value))
    pretrends_ok = pretrends_joint is not None and bool(np.isfinite(pretrends_joint.p_value))
    homogeneity_ok = homogeneity_joint is not None and bool(np.isfinite(homogeneity_joint.p_value))

    qug_rej = qug_ok and qug.reject
    pretrends_rej = pretrends_joint is not None and pretrends_ok and bool(pretrends_joint.reject)
    homogeneity_rej = (
        homogeneity_joint is not None and homogeneity_ok and bool(homogeneity_joint.reject)
    )

    reasons = []
    if qug_rej:
        reasons.append("support infimum rejected - continuous_at_zero design invalid (QUG)")
    if pretrends_rej:
        reasons.append("joint pre-trends rejected - assumption 7 violated (joint Stute)")
    if homogeneity_rej:
        reasons.append("joint linearity rejected - heterogeneity bias (joint Stute)")

    unresolved = []
    if not qug_ok:
        unresolved.append("QUG NaN")
    if pretrends_joint is None:
        unresolved.append("joint pre-trends skipped (no earlier pre-period)")
    elif not pretrends_ok:
        unresolved.append("joint pre-trends NaN")
    if homogeneity_joint is None:
        unresolved.append("joint linearity skipped")
    elif not homogeneity_ok:
        unresolved.append("joint linearity NaN")

    if reasons:
        verdict = "; ".join(reasons)
        if unresolved:
            verdict += "; additional steps unresolved: " + "; ".join(unresolved)
        return verdict

    if unresolved:
        return "inconclusive - " + "; ".join(unresolved)

    return (
        "QUG, joint pre-trends, and joint linearity diagnostics "
        "fail-to-reject (TWFE admissible under Section 4 assumptions)"
    )


def stute_joint_pretest(
    residuals_by_horizon: Dict[Any, np.ndarray],
    fitted_by_horizon: Dict[Any, np.ndarray],
    doses: np.ndarray,
    design_matrix: np.ndarray,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 999,
    seed: Optional[int] = None,
    null_form: str = "custom",
) -> StuteJointResult:
    """Joint Cramer-von Mises pretest across multiple horizons.

    Generalizes :func:`stute_test` to K horizons with the joint
    statistic ``S_joint = sum_k S_k``, where ``S_k`` is the single-
    horizon CvM on residuals ``eps_{g,k}``. Inference is via Mammen wild
    bootstrap with a **shared** multiplier ``eta_g`` across horizons per
    unit to preserve the vector-valued empirical process's unit-level
    dependence.

    **Note:** sum-of-CvMs aggregation follows the standard joint
    specification-test construction (Delgado 1993; Escanciano 2006). The
    paper does not prescribe an aggregation; sum-of-CvMs balances power
    across diffuse vs concentrated alternatives and bootstraps cleanly
    with the shared-eta structure.

    Bootstrap uses the literal per-iteration OLS refit form (paper
    Appendix D) for consistency with Phase 3's :func:`stute_test`.
    ``XtX_inv_Xt`` is precomputed once (same design matrix each
    iteration), so the refit cost is O(Gp) per bootstrap draw and the
    overall loop is dominated by :func:`_cvm_statistic` across K
    horizons.

    Parameters
    ----------
    residuals_by_horizon : dict[str, np.ndarray]
        ``{label: eps_g}`` per horizon. All values must have identical
        length ``G`` and be unit-ordered consistently with ``doses``.
    fitted_by_horizon : dict[str, np.ndarray]
        ``{label: fitted_g}`` per horizon. Required to reconstruct
        bootstrap outcomes ``dy*_{g,k} = fitted_{g,k} + eps_{g,k} *
        eta_g`` under the null.
    doses : np.ndarray, shape (G,)
        Dose per unit. Shared across horizons (HAD contract: dose is
        time-invariant per unit). Must be finite and non-negative.
    design_matrix : np.ndarray, shape (G, p)
        Regression design used in the per-horizon bootstrap refit.
        Mean-independence: ``[1]`` (intercept only). Linearity:
        ``[1, doses]``. The matrix is identical across horizons.
    alpha, n_bootstrap, seed : see :func:`stute_test`.
    null_form : str
        Diagnostic label recorded on the result
        (``"mean_independence"`` | ``"linearity"`` | ``"custom"``).
        The wrappers :func:`joint_pretrends_test` and
        :func:`joint_homogeneity_test` set this automatically.

    Returns
    -------
    StuteJointResult
        On the common path, a populated result with bootstrap-based
        ``p_value`` and ``cvm_stat_joint``. On the small-sample branch
        (``G < _MIN_G_STUTE``), constant-dose branch
        (``np.ptp(doses) <= 0``), or any-NaN branch in the input
        residuals / fitted arrays, returns an all-NaN result (with
        ``reject=False`` and the full ``per_horizon_stats`` dict keyed
        by the validated horizon labels) and emits a ``UserWarning``
        for the first two branches. Mirrors the single-horizon
        :func:`stute_test` contract so event-study workflows on small
        or staggered-filtered panels surface an inconclusive report
        rather than crashing.

    Raises
    ------
    ValueError
        On empty input, key-mismatch, stringified-label collisions
        between distinct raw keys, shape-mismatch, ``doses`` containing
        negative values, ``n_bootstrap < _MIN_N_BOOTSTRAP``, or invalid
        ``alpha``. ``G < _MIN_G_STUTE`` does NOT raise; see Returns.
    """
    if not isinstance(residuals_by_horizon, dict) or not isinstance(fitted_by_horizon, dict):
        raise ValueError(
            "residuals_by_horizon and fitted_by_horizon must be dicts " "keyed by horizon label."
        )
    if len(residuals_by_horizon) == 0:
        raise ValueError("residuals_by_horizon must contain at least one horizon.")
    if set(residuals_by_horizon.keys()) != set(fitted_by_horizon.keys()):
        raise ValueError(
            "residuals_by_horizon and fitted_by_horizon must have "
            "identical keys. Got "
            f"residuals keys: {sorted(residuals_by_horizon.keys())!r}, "
            f"fitted keys: {sorted(fitted_by_horizon.keys())!r}."
        )

    doses_arr = _validate_1d_numeric(np.asarray(doses), "doses")
    G = doses_arr.shape[0]
    if np.any(doses_arr < 0):
        raise ValueError(
            "doses must be non-negative (HAD contract - paper Section 2). "
            f"Found {int(np.sum(doses_arr < 0))} negative value(s)."
        )

    # G < _MIN_G_STUTE (CvM statistic not well-calibrated): mirror the
    # single-horizon `stute_test` contract - warn + return NaN result
    # rather than raise, so callers (including the event-study workflow
    # on a staggered panel whose last-cohort filter leaves fewer than
    # 10 units) get an inconclusive diagnostic instead of a crash. The
    # NaN return still satisfies the workflow's `np.isfinite(p_value)`
    # gating, so `all_pass` becomes False downstream.
    # Note: the actual `warn + return` happens below after horizon
    # labels are validated and collision-checked, so the NaN result
    # carries full per-horizon diagnostic keys.
    if n_bootstrap < _MIN_N_BOOTSTRAP:
        raise ValueError(f"n_bootstrap must be >= {_MIN_N_BOOTSTRAP}; got " f"{n_bootstrap}.")
    if not isinstance(alpha, (int, float)) or not (0 < float(alpha) < 1):
        raise ValueError(f"alpha must be in (0, 1); got {alpha!r}.")

    X = np.asarray(design_matrix, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] != G:
        raise ValueError(f"design_matrix must have shape (G, p) with G={G}; got " f"{X.shape}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("design_matrix contains non-finite values (NaN/inf).")

    raw_horizon_labels = list(residuals_by_horizon.keys())
    K = len(raw_horizon_labels)

    # Stringified-label collision guard: distinct raw keys whose str()
    # representations collide (e.g. {1: ..., "1": ..., 1.0: ...}) would
    # overwrite each other in residuals_arrays / fitted_arrays, letting
    # the surviving horizon be double-counted in S_joint = sum of S_k
    # and leaving `n_horizons` inconsistent with the number of distinct
    # diagnostic statistics. Reject explicitly rather than silently
    # collapsing the test.
    str_labels = [str(k) for k in raw_horizon_labels]
    if len(set(str_labels)) != len(str_labels):
        from collections import Counter

        dup_strs = [s for s, c in Counter(str_labels).items() if c > 1]
        collisions = {s: [k for k in raw_horizon_labels if str(k) == s] for s in dup_strs}
        raise ValueError(
            f"Horizon label collision after str() stringification: "
            f"{collisions!r}. The joint Stute helpers index residuals "
            f"and fitted values by str(label); distinct raw keys whose "
            f"stringified form collides would silently overwrite each "
            f"other and double-count the surviving horizon in S_joint. "
            f"Use string-distinct horizon labels (e.g. 1997 and 1998 "
            f'as int, or "1997" and "1998" as str; not both).'
        )

    any_nan = False
    residuals_arrays: Dict[str, np.ndarray] = {}
    fitted_arrays: Dict[str, np.ndarray] = {}
    for k in raw_horizon_labels:
        eps_k = np.asarray(residuals_by_horizon[k], dtype=np.float64)
        fit_k = np.asarray(fitted_by_horizon[k], dtype=np.float64)
        if eps_k.shape != (G,) or fit_k.shape != (G,):
            raise ValueError(
                f"Horizon {k!r}: residuals shape {eps_k.shape} and "
                f"fitted shape {fit_k.shape} must both be ({G},) to "
                f"align with doses."
            )
        if not (np.all(np.isfinite(eps_k)) and np.all(np.isfinite(fit_k))):
            any_nan = True
        residuals_arrays[str(k)] = eps_k
        fitted_arrays[str(k)] = fit_k

    # Re-key to str labels consistently (wrappers already pass str; direct
    # callers may pass int/object). String identity per the documented
    # horizon_labels contract. The collision guard above ensures this
    # stringification is injective on the provided keys.
    horizon_labels = str_labels

    # Small-G NaN result (paired with the comment near the top of this
    # function): mirror the single-horizon stute_test contract so the
    # event-study workflow on a small or staggered-filtered panel gets
    # an inconclusive diagnostic rather than an exception. Positioned
    # AFTER the label-collision / shape-alignment guards so the NaN
    # result carries a consistent per-horizon diagnostic surface.
    if G < _MIN_G_STUTE:
        warnings.warn(
            f"stute_joint_pretest: G = {G} is below the minimum "
            f"{_MIN_G_STUTE} for the CvM statistic to be well-calibrated. "
            f"Returning NaN result.",
            UserWarning,
            stacklevel=2,
        )
        return StuteJointResult(
            cvm_stat_joint=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=float(alpha),
            horizon_labels=horizon_labels,
            per_horizon_stats={k: float("nan") for k in horizon_labels},
            n_bootstrap=int(n_bootstrap),
            n_obs=int(G),
            n_horizons=int(K),
            seed=None if seed is None else int(seed),
            null_form=str(null_form),
            exact_linear_short_circuited=False,
        )

    if any_nan:
        return StuteJointResult(
            cvm_stat_joint=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=float(alpha),
            horizon_labels=horizon_labels,
            per_horizon_stats={k: float("nan") for k in horizon_labels},
            n_bootstrap=int(n_bootstrap),
            n_obs=int(G),
            n_horizons=int(K),
            seed=None if seed is None else int(seed),
            null_form=str(null_form),
            exact_linear_short_circuited=False,
        )

    # Zero-variation-in-D degeneracy guard: mirrors stute_test's intent
    # (had_pretests.py:~1233). The CvM cusum is defined against the
    # dose regressor; constant d has no cross-sectional variation for
    # the test to detect nonlinearity. Under the mean-independence null
    # this yields a mechanically-zero statistic (bogus fail-to-reject);
    # under the linearity null a singular [1, d] design matrix crashes
    # the refit. Emit warning + NaN result instead.
    #
    # Uses ``ptp`` (peak-to-peak = max - min) rather than ``np.var`` for
    # the degeneracy check: ``np.var`` of a truly constant array returns
    # a small non-zero value (~1e-32) due to E[X^2] - E[X]^2 rounding
    # noise, so a ``<= 0`` comparison misses the degeneracy. ``ptp`` is
    # bit-exact for identical inputs.
    if float(np.ptp(doses_arr)) <= 0.0:
        warnings.warn(
            "stute_joint_pretest: constant doses (zero cross-sectional "
            "variation); the joint Stute CvM requires dose variation. "
            "Returning NaN result.",
            UserWarning,
            stacklevel=2,
        )
        return StuteJointResult(
            cvm_stat_joint=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=float(alpha),
            horizon_labels=horizon_labels,
            per_horizon_stats={k: float("nan") for k in horizon_labels},
            n_bootstrap=int(n_bootstrap),
            n_obs=int(G),
            n_horizons=int(K),
            seed=None if seed is None else int(seed),
            null_form=str(null_form),
            exact_linear_short_circuited=False,
        )

    idx = np.argsort(doses_arr, kind="stable")
    d_sorted = doses_arr[idx]

    per_horizon_stats: Dict[str, float] = {}
    for k in horizon_labels:
        per_horizon_stats[k] = _cvm_statistic(residuals_arrays[k][idx], d_sorted)
    S_joint = float(sum(per_horizon_stats.values()))

    # Per-horizon exact-linear short-circuit (scale- and translation-
    # invariant, matches Phase 3 invariant). A single degenerate horizon
    # does NOT collapse the joint test if other horizons have nontrivial
    # residuals.
    short_circuit = True
    for k in horizon_labels:
        eps_k = residuals_arrays[k]
        fit_k = fitted_arrays[k]
        dy_k = fit_k + eps_k
        tss_centered = float(np.sum((dy_k - dy_k.mean()) ** 2))
        if tss_centered == 0.0:
            # Outcome identically constant: treat as trivially linear for
            # this horizon (ratio = 0). Does not force short-circuit
            # because other horizons may still be nontrivial.
            ratio = 0.0
        else:
            ratio = float(np.sum(eps_k**2) / tss_centered)
        if ratio >= _EXACT_LINEAR_RELATIVE_TOL:
            short_circuit = False
            break

    if short_circuit:
        return StuteJointResult(
            cvm_stat_joint=S_joint,
            p_value=1.0,
            reject=False,
            alpha=float(alpha),
            horizon_labels=horizon_labels,
            per_horizon_stats=per_horizon_stats,
            n_bootstrap=int(n_bootstrap),
            n_obs=int(G),
            n_horizons=int(K),
            seed=None if seed is None else int(seed),
            null_form=str(null_form),
            exact_linear_short_circuited=True,
        )

    # Precompute OLS projection matrix once: same X per bootstrap draw,
    # so (X'X)^-1 X' is constant across iterations. Keeps refit O(Gp)
    # per draw without changing semantics from the literal paper form.
    # Catch rank-deficient designs explicitly rather than surfacing a
    # raw ``np.linalg.LinAlgError`` to direct callers of the public
    # residuals-in core; matches the front-door validation style of
    # the other guards in this function.
    try:
        XtX_inv_Xt = np.linalg.solve(X.T @ X, X.T)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"design_matrix is rank-deficient (singular X^T X); cannot "
            f"compute the OLS projection (X^T X)^-1 X^T for the "
            f"bootstrap refit. Check for duplicate or linearly-"
            f"dependent columns. shape={X.shape}."
        ) from exc

    rng = np.random.default_rng(seed)
    bootstrap_S = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        # SHARED eta across horizons - preserves unit-level dependence
        # in the vector-valued empirical process. Independent-per-horizon
        # draws would overstate precision.
        eta = _generate_mammen_weights(G, rng)
        S_b = 0.0
        for k in horizon_labels:
            dy_b = fitted_arrays[k] + residuals_arrays[k] * eta
            beta_b = XtX_inv_Xt @ dy_b
            eps_b = dy_b - X @ beta_b
            S_b += _cvm_statistic(eps_b[idx], d_sorted)
        bootstrap_S[b] = S_b

    p_value = float((1.0 + np.sum(bootstrap_S >= S_joint)) / (n_bootstrap + 1))
    reject = bool(p_value <= alpha)

    return StuteJointResult(
        cvm_stat_joint=S_joint,
        p_value=p_value,
        reject=reject,
        alpha=float(alpha),
        horizon_labels=horizon_labels,
        per_horizon_stats=per_horizon_stats,
        n_bootstrap=int(n_bootstrap),
        n_obs=int(G),
        n_horizons=int(K),
        seed=None if seed is None else int(seed),
        null_form=str(null_form),
        exact_linear_short_circuited=False,
    )


def joint_pretrends_test(
    data: pd.DataFrame,
    outcome_col: str,
    dose_col: str,
    time_col: str,
    unit_col: str,
    pre_periods: list,
    base_period: Any,
    first_treat_col: Optional[str] = None,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 999,
    seed: Optional[int] = None,
) -> StuteJointResult:
    """Joint Stute pre-trends test (paper Section 4.2 step 2).

    Data-in wrapper around :func:`stute_joint_pretest` for the
    mean-independence null
    ``E[Y_{g,t} - Y_{g,base} | D_{g,treat}] = mu_t``
    across multiple pre-period placebos. For each ``t in pre_periods``,
    residuals are the deviations of ``Y_{g,t} - Y_{g,base}`` from their
    cross-unit mean (an intercept-only OLS fit); the joint CvM tests
    that the conditional mean depends on ``D``.

    Use this wrapper to close the paper's step-2 pre-trends gap that
    :func:`did_had_pretest_workflow` otherwise flags. On a panel with
    at least one earlier pre-period, the
    ``aggregate="event_study"`` dispatch calls this wrapper internally.

    Parameters
    ----------
    data : pd.DataFrame
    outcome_col, dose_col, time_col, unit_col : str
    pre_periods : list
        Non-empty list of pre-period labels (all ``< base_period``, all
        with ``D = 0`` across every unit). Empty list raises; the
        workflow dispatch handles the "no earlier pre-period" case by
        setting ``pretrends_joint=None`` rather than calling this
        wrapper.
    base_period : period label
        The reference period. Must not be in ``pre_periods``. Must also
        satisfy ``D = 0`` across every unit (reciprocal of the pre-period
        HAD invariant - base is itself a pre-period in the four-step
        workflow).
    first_treat_col : str or None
        Forwarded to the underlying panel validator; matched cohort
        handling follows the HAD contract (staggered auto-filter warns
        and proceeds on last cohort; solo cohort proceeds).
    alpha, n_bootstrap, seed : as in :func:`stute_test`.

    Returns
    -------
    StuteJointResult with ``null_form = "mean_independence"``.
    """
    if len(pre_periods) == 0:
        raise ValueError(
            "pre_periods must be non-empty. Workflow dispatch handles "
            "the empty case by setting pretrends_joint=None; direct "
            "callers should not pass an empty list."
        )
    if base_period in pre_periods:
        raise ValueError(
            f"base_period={base_period!r} must not appear in " f"pre_periods {list(pre_periods)!r}."
        )

    # Ordering check: all pre_periods strictly < base_period in
    # chronological order. Uses `_build_period_rank` to handle ordered-
    # categorical time columns correctly (raw Python `<` would fail on
    # categories whose lexical order disagrees with chronology, e.g.
    # ["q1", "q2", "q10"]). Numeric / datetime dtypes get natural order.
    period_rank = _build_period_rank(data, time_col)
    if base_period not in period_rank:
        raise ValueError(
            f"base_period={base_period!r} not found in time_col "
            f"{time_col!r}. Available: "
            f"{sorted(period_rank.keys(), key=lambda t: period_rank[t])!r}."
        )
    missing_pre_in_data = [t for t in pre_periods if t not in period_rank]
    if missing_pre_in_data:
        raise ValueError(
            f"pre_periods entries {missing_pre_in_data!r} not found in "
            f"time_col {time_col!r}. Available: "
            f"{sorted(period_rank.keys(), key=lambda t: period_rank[t])!r}."
        )
    base_rank = period_rank[base_period]
    out_of_order = [t for t in pre_periods if period_rank[t] >= base_rank]
    if out_of_order:
        raise ValueError(
            f"All pre_periods must be strictly < base_period in "
            f"chronological order. Violators: {out_of_order!r} "
            f"(base_period={base_period!r})."
        )

    # Event-study validation contract (paper Appendix B.2):
    # When the panel has >= 3 distinct periods, always route through
    # `_validate_had_panel_event_study`. This enforces (a) balanced
    # panel, (b) ordered time dtype, (c) D = 0 across every pre-period,
    # (d) last-cohort auto-filter under staggered timing with
    # UserWarning, (e) constant post-treatment dose within unit. When
    # first_treat_col is None and the panel is staggered, the validator
    # RAISES - matching the workflow dispatch contract. For 2-period
    # panels the validator does not apply; skip and fall through to the
    # simpler balance/invariant guards in `_aggregate_for_joint_test`.
    n_periods = int(data[time_col].nunique())
    data_filtered: pd.DataFrame = data
    if n_periods >= 3:
        F_val, t_pre_list, _t_post_list, data_filtered, _filter_info = (
            _validate_had_panel_event_study(
                data,
                outcome_col=outcome_col,
                dose_col=dose_col,
                time_col=time_col,
                unit_col=unit_col,
                first_treat_col=first_treat_col,
            )
        )
        # `_validate_had_panel_event_study` already emits its own
        # `UserWarning` on the staggered-filter path; the wrapper
        # consumes `_filter_info` silently to avoid duplicated console
        # noise (R4 code-quality fix).
        # Subset invariants: the caller's base_period and pre_periods
        # must be pre-treatment periods under the validator's partition.
        if base_period not in t_pre_list:
            raise ValueError(
                f"base_period={base_period!r} is not in the validated "
                f"pre-period set {list(t_pre_list)!r} (periods before "
                f"first-treatment period F={F_val!r}). For the HAD "
                f"pre-trends workflow, base_period must be a pre-period "
                f"anchor (typically the last pre-period, F-1)."
            )
        not_pre = [t for t in pre_periods if t not in t_pre_list]
        if not_pre:
            raise ValueError(
                f"pre_periods must all be validated pre-treatment "
                f"periods. Not-pre entries: {not_pre!r}. Validator's "
                f"pre-period set: {list(t_pre_list)!r}."
            )

    d_arr, dy_by_horizon, _ = _aggregate_for_joint_test(
        data_filtered,
        outcome_col=outcome_col,
        dose_col=dose_col,
        time_col=time_col,
        unit_col=unit_col,
        horizons=list(pre_periods),
        base_period=base_period,
    )
    G = d_arr.shape[0]

    # HAD invariant: D_{g,t} = 0 for every g and every pre_period (and
    # for base_period - it is itself a pre-period relative to the
    # treatment onset). We check this on the passed-in panel subset.
    needed_all_zero = list(pre_periods) + [base_period]
    subset_zero_check = data_filtered[data_filtered[time_col].isin(needed_all_zero)]
    if (subset_zero_check[dose_col] != 0).any():
        n_nonzero = int((subset_zero_check[dose_col] != 0).sum())
        raise ValueError(
            f"Pre-trends test requires D = 0 in every pre-period "
            f"(including base_period). Found {n_nonzero} non-zero "
            f"dose observation(s) across periods "
            f"{needed_all_zero!r}. HAD contract (paper Section 2) and "
            f"pre-trends test design both require the zero-dose "
            f"invariant to hold in ALL periods used as placebo or "
            f"anchor."
        )

    residuals_by_horizon: Dict[str, np.ndarray] = {}
    fitted_by_horizon: Dict[str, np.ndarray] = {}
    for label, dy_t in dy_by_horizon.items():
        mean_t = float(dy_t.mean())
        fitted_t = np.full(G, mean_t, dtype=np.float64)
        residuals_t = dy_t - fitted_t
        residuals_by_horizon[label] = residuals_t
        fitted_by_horizon[label] = fitted_t

    design_matrix = np.ones((G, 1), dtype=np.float64)

    return stute_joint_pretest(
        residuals_by_horizon=residuals_by_horizon,
        fitted_by_horizon=fitted_by_horizon,
        doses=d_arr,
        design_matrix=design_matrix,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        seed=seed,
        null_form="mean_independence",
    )


def joint_homogeneity_test(
    data: pd.DataFrame,
    outcome_col: str,
    dose_col: str,
    time_col: str,
    unit_col: str,
    post_periods: list,
    base_period: Any,
    first_treat_col: Optional[str] = None,
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 999,
    seed: Optional[int] = None,
) -> StuteJointResult:
    """Joint Stute homogeneity-linearity test (paper Section 4.3 joint).

    Data-in wrapper around :func:`stute_joint_pretest` for the
    linearity null
    ``E[Y_{g,t} - Y_{g,base} | D_{g,t}] = beta_{0,t} + beta_{fe,t} * D_{g,t}``
    across multiple post-period horizons. For each ``t in post_periods``,
    residuals are from an OLS regression of ``Y_{g,t} - Y_{g,base}`` on
    ``[1, D_g]``; the joint CvM tests whether the conditional mean is
    nonlinear in ``D`` in any horizon.

    Used by :func:`did_had_pretest_workflow` with
    ``aggregate="event_study"`` as the step-3 test (no joint Yatchew
    variant exists - the paper does not derive one; users who need
    Yatchew-style adjacent-difference robustness can call
    :func:`yatchew_hr_test` on each (base, post) pair manually).

    Parameters
    ----------
    data : pd.DataFrame
    outcome_col, dose_col, time_col, unit_col : str
    post_periods : list
        Non-empty list of post-period labels (all strictly ``>
        base_period`` by chronological order; each with ``D > 0`` for
        some unit, i.e. at least one treated unit per horizon).
    base_period : period label
        The reference period (last pre-period in the event-study
        convention). Must not be in ``post_periods``.
    first_treat_col : str or None
        Forwarded to the underlying panel validator.
    alpha, n_bootstrap, seed : as in :func:`stute_test`.

    Returns
    -------
    StuteJointResult with ``null_form = "linearity"``.
    """
    if len(post_periods) == 0:
        raise ValueError(
            "post_periods must be non-empty. Workflow dispatch handles "
            "the empty case upstream; direct callers should not pass "
            "an empty list."
        )
    if base_period in post_periods:
        raise ValueError(
            f"base_period={base_period!r} must not appear in "
            f"post_periods {list(post_periods)!r}."
        )

    # Ordering: all post_periods strictly > base_period in
    # chronological order. Uses `_build_period_rank` for ordered-
    # categorical correctness (raw Python `>` would misorder e.g.
    # "q10" > "q2").
    period_rank = _build_period_rank(data, time_col)
    if base_period not in period_rank:
        raise ValueError(
            f"base_period={base_period!r} not found in time_col "
            f"{time_col!r}. Available: "
            f"{sorted(period_rank.keys(), key=lambda t: period_rank[t])!r}."
        )
    missing_post_in_data = [t for t in post_periods if t not in period_rank]
    if missing_post_in_data:
        raise ValueError(
            f"post_periods entries {missing_post_in_data!r} not found in "
            f"time_col {time_col!r}. Available: "
            f"{sorted(period_rank.keys(), key=lambda t: period_rank[t])!r}."
        )
    base_rank = period_rank[base_period]
    out_of_order = [t for t in post_periods if period_rank[t] <= base_rank]
    if out_of_order:
        raise ValueError(
            f"All post_periods must be strictly > base_period in "
            f"chronological order. Violators: {out_of_order!r} "
            f"(base_period={base_period!r})."
        )

    # Event-study validation contract (paper Appendix B.2) - twin of
    # `joint_pretrends_test`. Same gating by `n_periods >= 3`; same
    # subset-invariant checks; emits the staggered-filter UserWarning.
    # The validator also enforces constant post-treatment dose within
    # unit, which is critical for the homogeneity path because a
    # time-varying post-dose would make the per-horizon refit on
    # `[1, D_g]` misspecify the regressor.
    n_periods = int(data[time_col].nunique())
    data_filtered: pd.DataFrame = data
    if n_periods >= 3:
        F_val, t_pre_list, t_post_list, data_filtered, _filter_info = (
            _validate_had_panel_event_study(
                data,
                outcome_col=outcome_col,
                dose_col=dose_col,
                time_col=time_col,
                unit_col=unit_col,
                first_treat_col=first_treat_col,
            )
        )
        # `_validate_had_panel_event_study` already emits its own
        # `UserWarning` on the staggered-filter path; the wrapper
        # consumes `_filter_info` silently to avoid duplicated console
        # noise (R4 code-quality fix).
        if base_period not in t_pre_list:
            raise ValueError(
                f"base_period={base_period!r} is not in the validated "
                f"pre-period set {list(t_pre_list)!r} (periods before "
                f"first-treatment period F={F_val!r}). For the HAD "
                f"homogeneity workflow, base_period must be a pre-period "
                f"anchor (typically the last pre-period, F-1)."
            )
        not_post = [t for t in post_periods if t not in t_post_list]
        if not_post:
            raise ValueError(
                f"post_periods must all be validated post-treatment "
                f"periods. Not-post entries: {not_post!r}. Validator's "
                f"post-period set: {list(t_post_list)!r}."
            )

    d_arr, dy_by_horizon, _ = _aggregate_for_joint_test(
        data_filtered,
        outcome_col=outcome_col,
        dose_col=dose_col,
        time_col=time_col,
        unit_col=unit_col,
        horizons=list(post_periods),
        base_period=base_period,
    )
    G = d_arr.shape[0]

    # HAD invariant for the homogeneity path: base_period has D = 0
    # (last pre-period contract); each post_period has D > 0 for SOME
    # unit (existence) and is NOT identically zero across all units
    # (reciprocal twin of the pretrends guard - an all-zero post-period
    # contradicts the HAD treatment-onset contract).
    base_doses = data_filtered.loc[data_filtered[time_col] == base_period, dose_col]
    if (base_doses != 0).any():
        n_nonzero = int((base_doses != 0).sum())
        raise ValueError(
            f"base_period={base_period!r} must have D = 0 across every "
            f"unit (HAD last-pre-period invariant). Found {n_nonzero} "
            f"non-zero dose observation(s) in base_period."
        )
    for t in post_periods:
        post_doses = data_filtered.loc[data_filtered[time_col] == t, dose_col]
        if not (post_doses > 0).any():
            raise ValueError(
                f"post_period={t!r} has D = 0 for every unit. HAD "
                f"contract requires at least some unit to have D > 0 "
                f"in each post-period (reciprocal of the pre-period "
                f"zero-dose invariant)."
            )

    residuals_by_horizon: Dict[str, np.ndarray] = {}
    fitted_by_horizon: Dict[str, np.ndarray] = {}
    for label, dy_t in dy_by_horizon.items():
        a_hat, b_hat, residuals_t = _fit_ols_intercept_slope(d_arr, dy_t)
        fitted_t = a_hat + b_hat * d_arr
        residuals_by_horizon[label] = residuals_t
        fitted_by_horizon[label] = fitted_t

    design_matrix = np.column_stack([np.ones(G, dtype=np.float64), d_arr.astype(np.float64)])

    return stute_joint_pretest(
        residuals_by_horizon=residuals_by_horizon,
        fitted_by_horizon=fitted_by_horizon,
        doses=d_arr,
        design_matrix=design_matrix,
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        seed=seed,
        null_form="linearity",
    )


_VALID_AGGREGATES = ("overall", "event_study")


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
    *,
    aggregate: str = "overall",
    survey: Any = None,
    weights: Optional[np.ndarray] = None,
) -> HADPretestReport:
    """Run the HAD pre-test workflow (paper Section 4.2-4.3).

    Two dispatch modes via ``aggregate``:

    ``aggregate="overall"`` (default, two-period panel): runs paper
    steps 1 (:func:`qug_test`) and 3 (:func:`stute_test` +
    :func:`yatchew_hr_test`). Step 2 (Assumption 7 pre-trends) is NOT
    implemented on this path because a single-pre-period panel cannot
    support the joint Stute variant; the returned verdict flags the
    Assumption 7 gap explicitly so callers do not receive a misleading
    "TWFE safe" signal. For multi-period panels, pass
    ``aggregate="event_study"`` to close the step-2 gap.

    ``aggregate="event_study"`` (multi-period panel, >= 3 periods): runs
    QUG + joint pre-trends Stute + joint homogeneity-linearity Stute,
    covering paper Section 4 steps 1-3 together. Step 4 (Yatchew-style
    linearity as an alternative to Stute) is subsumed by the joint Stute
    in this path - the paper does not derive a joint Yatchew variant, so
    users who need Yatchew robustness under multi-period data should
    call :func:`yatchew_hr_test` on each (base, post) pair manually.

    Eq 18 linear-trend detrending (paper Section 5.2 Pierce-Schott
    application) is a Phase 4 follow-up; the event-study path here
    implements the simpler mean-independence / linearity nulls.

    Parameters
    ----------
    data : pd.DataFrame
        HAD panel. For ``aggregate="overall"``: balanced two-period
        panel with pre-period dose = 0 for every unit. For
        ``aggregate="event_study"``: balanced multi-period panel with
        >= 3 periods, an ordered time dtype (numeric, datetime, or
        ordered categorical), and the pre-period D=0 invariant across
        all pre-periods.
    outcome_col, dose_col, time_col, unit_col : str
    first_treat_col : str or None, default None
        Optional first-treatment-period column. Required on the
        ``aggregate="event_study"`` path when the panel is staggered
        (multi-cohort); the panel validator auto-filters to the last
        cohort and emits ``UserWarning``. The overall path uses this for
        cross-validation only.
    alpha : float, default 0.05
    n_bootstrap : int, default 999
        Replication count for the single-horizon Stute (overall) or
        joint Stute (event_study).
    seed : int or None, default None
        Seed forwarded to the Stute bootstrap. QUG / Yatchew are
        deterministic.
    aggregate : str, keyword-only, default ``"overall"``
        Dispatch mode. Invalid values raise ``ValueError``.
    survey : SurveyDesign or None, keyword-only, default None
        Currently rejected with ``NotImplementedError``. See
        *Notes -- Survey/weighted data*.
    weights : np.ndarray or None, keyword-only, default None
        Currently rejected with ``NotImplementedError``. See
        *Notes -- Survey/weighted data*.

    Returns
    -------
    HADPretestReport
        On the overall path: ``stute`` and ``yatchew`` populated,
        ``pretrends_joint`` / ``homogeneity_joint`` are ``None``. On the
        event-study path: ``pretrends_joint`` (``None`` if no earlier
        pre-period) and ``homogeneity_joint`` populated, ``stute`` /
        ``yatchew`` are ``None``. ``aggregate`` is recorded on the
        report for serialization dispatch.

    Raises
    ------
    ValueError
        On invalid ``aggregate``, ``survey`` and ``weights`` both
        non-None, or any downstream front-door failure (panel balance,
        dtype, dose invariant).
    NotImplementedError
        If ``survey`` or ``weights`` is non-None. See
        *Notes -- Survey/weighted data*.

    Notes
    -----
    Survey/weighted data: the workflow does not yet accept ``survey=`` /
    ``weights=`` kwargs. Two reasons:

    1. QUG-under-survey is **permanently deferred** (Phase 4.5 C0
       decision gate). Extreme-order-statistic tests are not smooth
       functionals of the empirical CDF and have no off-the-shelf
       survey-aware analog. See :func:`qug_test` Notes.
    2. Survey support for the linearity-family pretests (:func:`stute_test`,
       :func:`yatchew_hr_test`, :func:`stute_joint_pretest`,
       :func:`joint_pretrends_test`, :func:`joint_homogeneity_test`) is
       planned for Phase 4.5 C via Rao-Wu rescaled bootstrap. Until that
       ships those sister pretests still raise bare ``TypeError`` on
       ``survey=`` / ``weights=`` because their signatures are closed
       (no kwargs added) -- adding rejection-only kwargs in C0 then
       implementing in C is API churn for no user benefit.

    Until Phase 4.5 C ships, run the workflow without ``survey`` /
    ``weights`` kwargs and verify identification manually.

    References
    ----------
    de Chaisemartin et al. (2026), Section 4.2-4.3, Theorem 4, Appendix
    D, Theorem 7.
    """
    if aggregate not in _VALID_AGGREGATES:
        raise ValueError(
            f"aggregate must be one of {list(_VALID_AGGREGATES)!r}; " f"got {aggregate!r}."
        )

    # Mutex on survey/weights, mirroring HeterogeneousAdoptionDiD.fit()
    # at had.py:2890.
    if survey is not None and weights is not None:
        raise ValueError(
            "Pass survey=<SurveyDesign> OR weights=<array>, not both. "
            "did_had_pretest_workflow does not yet accept either kwarg "
            "(Phase 4.5 C0 + Phase 4.5 C); see the NotImplementedError "
            "below for the methodology rationale."
        )

    # Phase 4.5 C0 decision gate (workflow surface). QUG-under-survey is
    # permanently deferred; the linearity-family pretests are deferred to
    # Phase 4.5 C. Until C ships, the workflow has no survey-aware
    # dispatch and rejects the kwargs at the front door.
    if survey is not None or weights is not None:
        raise NotImplementedError(
            "did_had_pretest_workflow does not yet accept survey= / "
            "weights= kwargs.\n"
            "\n"
            "QUG-under-survey is permanently deferred (extreme-value "
            "theory under complex sampling is not a settled toolkit; see "
            "qug_test docstring for the methodology rationale). Survey "
            "support for stute_test, yatchew_hr_test, and joint variants "
            "is planned for Phase 4.5 C via Rao-Wu rescaled bootstrap. "
            "Until that ships, run the workflow without survey/weights "
            "kwargs and verify identification manually."
        )

    if aggregate == "event_study":
        F, t_pre_list, t_post_list, data_filtered, _filter_info = _validate_multi_period_panel(
            data,
            outcome_col=outcome_col,
            dose_col=dose_col,
            time_col=time_col,
            unit_col=unit_col,
            first_treat_col=first_treat_col,
        )
        # `_validate_multi_period_panel` delegates to
        # `_validate_had_panel_event_study`, which already emits its own
        # `UserWarning` on the staggered-filter path; we do NOT warn a
        # second time here (R4 code-quality fix - single emission point).

        # Base period for both joint tests is the last pre-period
        # (paper convention: anchor at F-1 under natural time order).
        # This is t_pre_list[-1] - NOT an arithmetic F-1, since the
        # time column may be non-integer (datetime, ordered categorical).
        base_period = t_pre_list[-1]

        # Step 1: QUG on dose distribution at F. Doses are
        # time-invariant in HAD, so D_g at F equals max_t D_{g,t}.
        doses_at_F = (
            data_filtered.loc[data_filtered[time_col] == F, [unit_col, dose_col]]
            .set_index(unit_col)
            .sort_index()[dose_col]
            .to_numpy(dtype=np.float64)
        )
        qug_res = qug_test(doses_at_F, alpha=alpha)

        # Step 2: joint pre-trends on earlier pre-periods (those
        # strictly before base_period). If only the base pre-period is
        # available (len(t_pre_list) == 1), there are no earlier
        # placebos; set pretrends_joint=None and flag in verdict.
        # ``t_pre_list`` is returned chronologically sorted by
        # ``_validate_had_panel_event_study`` (using the column's
        # ordered-categorical category order or the natural numeric /
        # datetime order), so taking everything but the last element
        # gives the earlier pre-periods regardless of dtype. Raw
        # ``t < base_period`` would misorder ordered-categorical labels
        # whose lexical and chronological order disagree (e.g. "q10" <
        # "q2" lexically but > chronologically).
        earlier_pre = list(t_pre_list[:-1])
        if len(earlier_pre) >= 1:
            pretrends_joint = joint_pretrends_test(
                data_filtered,
                outcome_col=outcome_col,
                dose_col=dose_col,
                time_col=time_col,
                unit_col=unit_col,
                pre_periods=earlier_pre,
                base_period=base_period,
                first_treat_col=first_treat_col,
                alpha=alpha,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
        else:
            pretrends_joint = None

        # Step 3: joint homogeneity-linearity on post-periods.
        homogeneity_joint = joint_homogeneity_test(
            data_filtered,
            outcome_col=outcome_col,
            dose_col=dose_col,
            time_col=time_col,
            unit_col=unit_col,
            post_periods=list(t_post_list),
            base_period=base_period,
            first_treat_col=first_treat_col,
            alpha=alpha,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )

        # Event-study `all_pass`: True iff every implemented step is
        # conclusive AND none reject. `pretrends_joint` must exist
        # (cannot be None) for the step-2 gap to be closed. Uses
        # `np.isfinite(p_value)` per Phase 3 convention (no
        # `.conclusive()` helper on result dataclasses).
        qug_ok = bool(np.isfinite(qug_res.p_value))
        pretrends_ok = pretrends_joint is not None and bool(np.isfinite(pretrends_joint.p_value))
        homogeneity_ok = bool(np.isfinite(homogeneity_joint.p_value))
        all_pass = bool(
            qug_ok
            and pretrends_ok
            and pretrends_joint is not None
            and not pretrends_joint.reject
            and homogeneity_ok
            and not homogeneity_joint.reject
            and not qug_res.reject
        )
        verdict = _compose_verdict_event_study(qug_res, pretrends_joint, homogeneity_joint)

        return HADPretestReport(
            qug=qug_res,
            stute=None,
            yatchew=None,
            all_pass=all_pass,
            verdict=verdict,
            alpha=alpha,
            n_obs=int(doses_at_F.shape[0]),
            pretrends_joint=pretrends_joint,
            homogeneity_joint=homogeneity_joint,
            aggregate="event_study",
        )

    # aggregate == "overall" - Phase 3 behavior, unchanged.
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
        aggregate="overall",
    )
