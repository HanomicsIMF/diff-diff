"""
Heterogeneous Adoption Difference-in-Differences (HAD) estimator (Phase 2a).

Implements the de Chaisemartin, Ciccia, D'Haultfoeuille, and Knau (2026)
Weighted-Average-Slope (WAS) estimator with three design-dispatch paths.
All three paths produce a beta-scale point estimate of the form
``(mean(Delta Y) - [boundary limit]) / [expected dose gap]`` (Design 1
family) or the Wald-IV ratio (mass-point), then route inference through
:func:`diff_diff.utils.safe_inference`.

1. Design 1' (``continuous_at_zero``): ``d_lower = 0``, boundary density
   continuous at zero, Assumption 3. Theorem 1 / Equation 3
   (identification); Equation 7 (sample estimator):

       beta = (E[Delta Y] - lim_{d v 0} E[Delta Y | D_2 <= d]) / E[D_2]

   The bias-corrected local-linear fit at ``boundary = 0`` estimates
   the boundary limit; ``E[D_2]`` and ``E[Delta Y]`` are plugin sample
   means.

2. Design 1 continuous-near-d_lower (``continuous_near_d_lower``):
   ``d_lower > 0``, continuous boundary density, Assumption 5 or 6.
   Theorem 3 / Equation 11 (``WAS_{d_lower}`` under Assumption 6;
   Theorem 4 is the QUG null test, not this estimand):

       beta = (E[Delta Y] - lim_{d v d_lower} E[Delta Y | D_2 <= d])
              / E[D_2 - d_lower]

   The local-linear fit is anchored at ``d_lower`` via the regressor
   shift ``D' = D - d_lower``, evaluated at ``boundary = 0`` on the
   shifted scale.

3. Design 1 mass-point (``mass_point``): ``d_lower > 0``, modal fraction
   at ``d.min()`` exceeds 2%. Sample-average 2SLS with instrument
   ``Z_g = 1{D_{g,2} > d_lower}`` (paper Section 3.2.4). Point estimate
   equals the Wald-IV ratio
   ``(Ybar_{Z=1} - Ybar_{Z=0}) / (Dbar_{Z=1} - Dbar_{Z=0})``. Standard
   error via the structural-residual 2SLS sandwich
   (see ``_fit_mass_point_2sls``).

Phase 2a ships the single-period WAS estimator (``aggregate="overall"``).
Phase 2b adds the multi-period event-study extension (paper Appendix B.2)
via ``aggregate="event_study"``: per-horizon WAS estimates with pointwise
CIs, including pre-period placebos, reusing the three Phase 2a design
paths on per-horizon first differences anchored at ``Y_{g, F-1}``.
Staggered-timing panels are auto-filtered to the last-treatment cohort
per paper Appendix B.2 prescription.

Infrastructure reused from Phase 1:
- ``diff_diff.local_linear.bias_corrected_local_linear`` (Phase 1c)
- ``diff_diff.local_linear.BiasCorrectedFit``, ``BandwidthResult``
- ``diff_diff.utils.safe_inference`` (NaN-safe CI gating)
- ``diff_diff.prep.balance_panel`` (panel validation)

References
----------
- de Chaisemartin, C., Ciccia, D., D'Haultfoeuille, X., & Knau, F. (2026).
  Difference-in-Differences Estimators When No Unit Remains Untreated.
  arXiv:2405.04465v6.
- Calonico, S., Cattaneo, M. D., & Titiunik, R. (2014). Robust
  nonparametric confidence intervals for regression-discontinuity designs.
  Econometrica, 82(6), 2295-2326.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.local_linear import (
    BandwidthResult,
    BiasCorrectedFit,
    bias_corrected_local_linear,
)
from diff_diff.utils import safe_inference

__all__ = [
    "HeterogeneousAdoptionDiD",
    "HeterogeneousAdoptionDiDResults",
    "HeterogeneousAdoptionDiDEventStudyResults",
]


# =============================================================================
# Module-level constants
# =============================================================================

# REGISTRY line 2320 auto-detect boundary: d.min() < 0.01 * median(|d|) resolves
# to Design 1' (continuous_at_zero). Distinct from Phase 1c's stricter
# _DESIGN_1_PRIME_RATIO = 0.05 in ``local_linear.py`` which is a plausibility
# guard on the nonparametric fit, not the auto-detect rule.
_CONTINUOUS_AT_ZERO_THRESHOLD = 0.01

# REGISTRY line 2320 mass-point rule: modal fraction at d.min() > 0.02 resolves
# to Design 1 mass-point. Mirrored (not imported) from
# ``diff_diff.local_linear._validate_had_inputs`` where the same constant
# enforces the upstream mass-point rejection on the nonparametric path.
_MASS_POINT_THRESHOLD = 0.02

_VALID_DESIGNS = (
    "auto",
    "continuous_at_zero",
    "continuous_near_d_lower",
    "mass_point",
)
_VALID_AGGREGATES = ("overall", "event_study")
# Mass-point 2SLS supports: classical, hc1 (HC0 is an unscaled variant not
# publicly exposed; hc2 and hc2_bm raise NotImplementedError pending a
# 2SLS-specific leverage derivation and R parity anchor).
_MASS_POINT_VCOV_SUPPORTED = ("classical", "hc1")
_MASS_POINT_VCOV_UNSUPPORTED = ("hc2", "hc2_bm")

# Target-parameter label per design. Design 1' targets the WAS (Assumption 3);
# Design 1 targets WAS_{d_lower} (Assumption 5 or 6), which also applies to
# the mass-point path (paper Section 3.2.4).
_TARGET_PARAMETER = {
    "continuous_at_zero": "WAS",
    "continuous_near_d_lower": "WAS_d_lower",
    "mass_point": "WAS_d_lower",
}


# =============================================================================
# JSON-serialization helpers
# =============================================================================


def _json_safe_scalar(x: Any) -> Any:
    """Coerce a scalar to a JSON-serializable type.

    - NumPy scalars (``np.int64``, ``np.float64``, ``np.bool_``) ->
      native Python via ``.item()``
    - ``pd.Timestamp`` / ``pd.Timedelta`` -> ISO 8601 string via
      ``.isoformat()``
    - Everything else returned as-is.

    The ``to_dict`` methods use this to keep the returned dict
    serializable via ``json.dumps`` regardless of the underlying
    pandas/numpy dtype of the time / first_treat columns.
    """
    if isinstance(x, (pd.Timestamp, pd.Timedelta)):
        return x.isoformat()
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return x.item()
        except (AttributeError, ValueError, TypeError):
            return x
    return x


def _json_safe_filter_info(
    filter_info: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Normalize a ``filter_info`` dict to JSON-safe scalars.

    Returns ``None`` unchanged; otherwise coerces ``F_last`` and each
    entry in ``dropped_cohorts`` via :func:`_json_safe_scalar`. Int
    counts are cast to ``int`` for stability.
    """
    if filter_info is None:
        return None
    return {
        "F_last": _json_safe_scalar(filter_info.get("F_last")),
        "n_kept": int(filter_info.get("n_kept", 0)),
        "n_dropped": int(filter_info.get("n_dropped", 0)),
        "dropped_cohorts": [_json_safe_scalar(c) for c in filter_info.get("dropped_cohorts", [])],
    }


# =============================================================================
# Results dataclass
# =============================================================================


@dataclass
class HeterogeneousAdoptionDiDResults:
    """Estimator output for :class:`HeterogeneousAdoptionDiD`.

    NaN-safe inference: the three downstream fields ``t_stat``,
    ``p_value``, and ``conf_int`` are routed through
    :func:`diff_diff.utils.safe_inference`, which returns NaN on all
    three whenever ``se`` is non-finite, zero, or negative. ``att`` and
    ``se`` themselves are RAW estimator outputs from the chosen fit
    path and are NOT gated by ``safe_inference``:

    - On the degenerate fit configurations (constant outcome on the
      continuous paths, all-units-at-d_lower / no-dose-variation on the
      mass-point path), the fit path explicitly returns
      ``(att=nan, se=nan)``, which combined with the safe-inference
      gate yields all five fields NaN together.
    - On the degenerate CR1 cluster configuration (mass-point path
      with a single cluster), ``_fit_mass_point_2sls`` returns
      ``(att=beta_hat, se=nan)`` - ``att`` stays finite because the
      Wald-IV ratio is well defined, but the cluster-robust SE is
      not, so ``se`` is NaN and the downstream triple becomes NaN
      via the safe-inference gate.

    So the guaranteed NaN coupling is on the downstream triple
    (``t_stat``, ``p_value``, ``conf_int``), not on ``att``. The
    ``assert_nan_inference`` fixture in ``tests/conftest.py`` checks
    the downstream triple against the gate contract and does not
    assume ``att`` is NaN.

    Attributes
    ----------
    att : float
        Point estimate of the WAS parameter on the beta-scale.

        - Design 1' (paper Theorem 1 / Equation 3 identification;
          Equation 7 sample estimator):
          ``att = (mean(ΔY) - tau_bc) / D_bar``
          where ``tau_bc`` is the bias-corrected local-linear estimate
          of ``lim_{d v 0} E[ΔY | D_2 <= d]`` and
          ``D_bar = (1/G) * sum(D_{g,2})``.
        - Design 1 continuous-near-d_lower (paper Theorem 3 /
          Equation 11, ``WAS_{d_lower}`` under Assumption 6):
          ``att = (mean(ΔY) - tau_bc) / mean(D_2 - d_lower)``
          where ``tau_bc`` is the bias-corrected local-linear estimate
          of ``lim_{d v d_lower} E[ΔY | D_2 <= d]``.
        - Mass-point (paper Section 3.2.4): the Wald-IV / 2SLS
          coefficient directly -
          ``(Ybar_{Z=1} - Ybar_{Z=0}) / (Dbar_{Z=1} - Dbar_{Z=0})``.
    se : float
        Standard error on the beta-scale. For continuous designs, the
        CCT-2014 robust SE from Phase 1c divided by ``|den|`` (the
        absolute denominator used in ``att``); the higher-order
        variance from ``mean(ΔY)`` is dominated by the nonparametric
        boundary estimate in large samples and is not included. For
        mass-point, the 2SLS structural-residual sandwich SE.
    t_stat, p_value, conf_int : inference fields
        Routed through ``safe_inference``; NaN when SE is non-finite.
    alpha : float
        CI level used at fit time (0.05 for a 95% CI).
    design : str
        Resolved design mode: ``"continuous_at_zero"``,
        ``"continuous_near_d_lower"``, or ``"mass_point"``. ``"auto"`` is
        resolved to one of the three concrete modes before storing.
    target_parameter : str
        Estimand label: ``"WAS"`` for Design 1', ``"WAS_d_lower"`` for the
        other two. Pins the estimand semantically even when two designs
        share the same divisor.
    d_lower : float
        Support infimum ``d_lower``. ``0.0`` for Design 1';
        ``float(d.min())`` for the other two.
    dose_mean : float
        ``D_bar = (1/G) * sum(D_{g,2})``.
    n_obs : int
        Number of units contributing to the estimator (post panel
        aggregation to unit-level first differences).
    n_treated : int
        Number of units with ``D_{g,2} > d_lower``.
    n_control : int
        Number of units at or below ``d_lower`` (the "not-treated" subset).
    n_mass_point : int or None
        Mass-point path only: number of units with ``D_{g,2} == d_lower``.
        ``None`` on continuous paths.
    n_above_d_lower : int or None
        Mass-point path only: number of units with ``D_{g,2} > d_lower``.
        ``None`` on continuous paths.
    inference_method : str
        ``"analytical_nonparametric"`` (continuous designs) or
        ``"analytical_2sls"`` (mass-point).
    vcov_type : str or None
        Effective variance-covariance family used. ``None`` on continuous
        paths (they use the CCT-2014 robust SE from Phase 1c, not the
        library's ``vcov_type`` enum). Mass-point: ``"classical"`` or
        ``"hc1"`` when ``cluster`` is not supplied, and ``"cr1"``
        whenever ``cluster`` is supplied (cluster-robust CR1 is computed
        regardless of the requested ``vcov_type`` because
        classical/hc1 + cluster collapses to the same CR1 sandwich).
        Downstream consumers reading ``result.to_dict()`` can inspect
        this field directly to determine the effective SE family.
    cluster_name : str or None
        Column name of the cluster variable on the mass-point path when
        cluster-robust SE is requested. ``None`` otherwise.
    survey_metadata : object or None
        Always ``None`` in Phase 2a. Field shape kept for future-compat
        with a planned survey integration PR.
    bandwidth_diagnostics : BandwidthResult or None
        Full Phase 1b MSE-DPI selector output on the continuous paths
        (when bandwidths were auto-selected). ``None`` on the mass-point
        path (parametric, no bandwidth).
    bias_corrected_fit : BiasCorrectedFit or None
        Full Phase 1c bias-corrected local-linear fit on the continuous
        paths. ``None`` on the mass-point path.
    """

    # Point estimate + inference (safe_inference-gated)
    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    alpha: float

    # Design metadata
    design: str
    target_parameter: str
    d_lower: float
    dose_mean: float

    # Sample counts
    n_obs: int
    n_treated: int
    n_control: int
    n_mass_point: Optional[int]
    n_above_d_lower: Optional[int]

    # Inference metadata
    inference_method: str
    vcov_type: Optional[str]
    cluster_name: Optional[str]
    survey_metadata: Optional[Any]

    # Nonparametric-only diagnostics
    bandwidth_diagnostics: Optional[BandwidthResult]
    bias_corrected_fit: Optional[BiasCorrectedFit]

    def __repr__(self) -> str:
        return (
            f"HeterogeneousAdoptionDiDResults("
            f"att={self.att:.4f}, se={self.se:.4f}, "
            f"design={self.design!r}, n_obs={self.n_obs})"
        )

    def summary(self) -> str:
        """Formatted summary table."""
        width = 72
        conf_level = int((1 - self.alpha) * 100)
        lines = [
            "=" * width,
            "HeterogeneousAdoptionDiD Estimation Results".center(width),
            "=" * width,
            "",
            f"{'Design:':<30} {self.design:>20}",
            f"{'Target parameter:':<30} {self.target_parameter:>20}",
            f"{'d_lower:':<30} {self.d_lower:>20.6g}",
            f"{'D_bar (dose mean):':<30} {self.dose_mean:>20.6g}",
            f"{'Observations (units):':<30} {self.n_obs:>20}",
            f"{'Above d_lower:':<30} {self.n_treated:>20}",
            f"{'At/below d_lower:':<30} {self.n_control:>20}",
        ]
        if self.n_mass_point is not None:
            lines.append(f"{'At d_lower (mass point):':<30} {self.n_mass_point:>20}")
        if self.n_above_d_lower is not None:
            lines.append(f"{'Strictly above d_lower:':<30} {self.n_above_d_lower:>20}")
        lines.append(f"{'Inference method:':<30} {self.inference_method:>20}")
        if self.vcov_type is not None:
            if self.cluster_name is not None:
                # Cluster-robust (CR1): the stored vcov_type is already "cr1",
                # but render with the cluster column for clarity.
                label = f"CR1 at {self.cluster_name}"
            else:
                label = self.vcov_type
            lines.append(f"{'Variance:':<30} {label:>20}")
        if self.bandwidth_diagnostics is not None:
            bw = self.bandwidth_diagnostics
            lines.append(f"{'Bandwidth h (MSE-DPI):':<30} {bw.h_mse:>20.6g}")
            lines.append(f"{'Bandwidth b (bias):':<30} {bw.b_mse:>20.6g}")
        if self.bias_corrected_fit is not None:
            bc = self.bias_corrected_fit
            lines.append(f"{'Bandwidth h used:':<30} {bc.h:>20.6g}")
            lines.append(f"{'Obs in window (n_used):':<30} {bc.n_used:>20}")
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            lines.append(f"{'Survey method:':<30} {sm.get('method', 'unknown'):>20}")
            if "effective_sample_size" in sm:
                ess = sm["effective_sample_size"]
                lines.append(f"{'Effective sample size:':<30} {ess:>20.6g}")
        param_label = self.target_parameter
        lines.extend(
            [
                "",
                "-" * width,
                (
                    f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} "
                    f"{'t-stat':>10} {'P>|t|':>10}"
                ),
                "-" * width,
                (
                    f"{param_label:<15} {self.att:>12.4f} {self.se:>12.4f} "
                    f"{self.t_stat:>10.3f} {self.p_value:>10.4f}"
                ),
                "-" * width,
                "",
                (
                    f"{conf_level}% Confidence Interval: "
                    f"[{self.conf_int[0]:.4f}, {self.conf_int[1]:.4f}]"
                ),
                "=" * width,
            ]
        )
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Return results as a dict of scalars + ``survey_metadata`` (dict
        or ``None``). When ``survey=`` / ``weights=`` is supplied to
        ``fit()``, ``survey_metadata`` carries the weighted-sample
        diagnostic (method, weight sum, effective sample size) so
        downstream consumers can inspect how the fit was weighted without
        digging into the estimator object."""
        return {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int[0],
            "conf_int_upper": self.conf_int[1],
            "alpha": self.alpha,
            "design": self.design,
            "target_parameter": self.target_parameter,
            "d_lower": self.d_lower,
            "dose_mean": self.dose_mean,
            "n_obs": self.n_obs,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "n_mass_point": self.n_mass_point,
            "n_above_d_lower": self.n_above_d_lower,
            "inference_method": self.inference_method,
            "vcov_type": self.vcov_type,
            "cluster_name": self.cluster_name,
            "survey_metadata": self.survey_metadata,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a one-row DataFrame of the result dict."""
        return pd.DataFrame([self.to_dict()])


@dataclass
class HeterogeneousAdoptionDiDEventStudyResults:
    """Event-study results for :class:`HeterogeneousAdoptionDiD` (Phase 2b).

    Per-horizon arrays align with ``event_times`` by index; all per-horizon
    arrays have shape ``(n_horizons,)``. The anchor horizon ``e = -1``
    (i.e., ``t = F - 1``) is NOT included because
    ``Y_{g, F-1} - Y_{g, F-1} = 0`` trivially and the WAS is not identified
    there.

    Per-horizon inference fields (``t_stat``, ``p_value``, ``conf_int_low``,
    ``conf_int_high``) are NaN-coupled to the per-horizon ``se`` via
    :func:`diff_diff.utils.safe_inference`; ``att`` and ``se`` themselves
    are raw estimator outputs from the chosen design path on each
    horizon's first differences.

    Design resolution is SHARED across horizons: the design, ``d_lower``,
    ``target_parameter``, and ``inference_method`` are single scalars
    determined once from the post-period dose distribution ``D_{g, F}``
    (paper Appendix B.2 convention — the dose regressor is invariant
    across event-time horizons).

    Attributes
    ----------
    event_times : np.ndarray, shape (n_horizons,)
        Integer event-time labels ``e = t - F``, sorted ascending.
        Excludes ``e = -1`` (the anchor). Post-period horizons have
        ``e >= 0``; pre-period placebos have ``e <= -2``.
    att : np.ndarray, shape (n_horizons,)
        Per-horizon WAS point estimate on the beta-scale (see
        :class:`HeterogeneousAdoptionDiDResults.att` for the per-design
        formula, applied to ``ΔY_t = Y_{g,t} - Y_{g,F-1}``).
    se : np.ndarray, shape (n_horizons,)
        Per-horizon standard error on the beta-scale. Each horizon uses
        the INDEPENDENT per-period sandwich from the chosen design path
        (continuous: CCT-2014 robust divided by ``|den|``; mass-point:
        structural-residual 2SLS sandwich). Pointwise CIs only — joint
        cross-horizon covariance is not computed in Phase 2b (paper
        reports pointwise CIs per Pierce-Schott).
    t_stat, p_value : np.ndarray, shape (n_horizons,)
        Per-horizon inference triple element.
    conf_int_low, conf_int_high : np.ndarray, shape (n_horizons,)
        Per-horizon CI endpoints at level ``alpha``.
    n_obs_per_horizon : np.ndarray, shape (n_horizons,)
        Per-horizon sample size (units contributing at that event time).
        In Phase 2b this equals ``n_units`` for every horizon because the
        validator rejects NaN in outcome / dose / unit columns upstream;
        tracked as a field for future flexibility (e.g., per-period
        missingness).
    alpha : float
        CI level used at fit time (0.05 for a 95% CI).
    design : str
        Resolved design mode, shared across horizons:
        ``"continuous_at_zero"``, ``"continuous_near_d_lower"``, or
        ``"mass_point"``.
    target_parameter : str
        Estimand label: ``"WAS"`` for Design 1' (continuous_at_zero),
        ``"WAS_d_lower"`` for the other two.
    d_lower : float
        Support infimum used for all horizons. ``0.0`` for Design 1';
        ``float(d.min())`` otherwise.
    dose_mean : float
        ``D_bar = (1/G) * sum(D_{g,F})`` computed on the fit sample (after
        the staggered last-cohort filter, if applied).
    F : object
        First-treatment period label (arbitrary dtype — int, str,
        datetime). Identified by the multi-period dose invariant from the
        fitted data.
    n_units : int
        Number of unique units contributing to the fit. After staggered
        auto-filter: last-cohort units PLUS never-treated (``first_treat = 0``)
        units retained as the untreated-group comparison per paper
        Appendix B.2. Only earlier-treated cohorts are dropped.
    inference_method : str
        ``"analytical_nonparametric"`` (continuous designs) or
        ``"analytical_2sls"`` (mass-point). Shared across horizons.
    vcov_type : str or None
        Effective variance-covariance family used on the mass-point path
        (``"classical"``, ``"hc1"``, or ``"cr1"`` when cluster supplied).
        ``None`` on the continuous paths (they use CCT-2014 robust SE).
    cluster_name : str or None
        Column name of the cluster variable when cluster-robust SE is
        requested. ``None`` otherwise.
    survey_metadata : object or None
        Always ``None`` in Phase 2b. Field shape kept for future-compat.
    bandwidth_diagnostics : list[BandwidthResult] or None
        Per-horizon bandwidth diagnostics on the continuous paths;
        ``None`` on the mass-point path. When non-None, aligned with
        ``event_times`` by index.
    bias_corrected_fit : list[BiasCorrectedFit] or None
        Per-horizon bias-corrected fit on the continuous paths; ``None``
        on the mass-point path. When non-None, aligned with
        ``event_times`` by index.
    filter_info : dict or None
        Populated when the staggered-timing last-cohort auto-filter fires.
        Keys: ``"F_last"`` (kept cohort label), ``"n_kept"`` (units
        retained), ``"n_dropped"`` (units dropped), ``"dropped_cohorts"``
        (list of dropped cohort labels). ``None`` when no filter was
        applied.
    """

    # Per-horizon arrays
    event_times: np.ndarray
    att: np.ndarray
    se: np.ndarray
    t_stat: np.ndarray
    p_value: np.ndarray
    conf_int_low: np.ndarray
    conf_int_high: np.ndarray
    n_obs_per_horizon: np.ndarray

    # Shared metadata
    alpha: float
    design: str
    target_parameter: str
    d_lower: float
    dose_mean: float
    F: Any
    n_units: int
    inference_method: str
    vcov_type: Optional[str]
    cluster_name: Optional[str]
    survey_metadata: Optional[Any]

    # Per-horizon diagnostics (lists, None on mass-point).
    # List entries may be None for horizons where the continuous-path fit
    # caught a degenerate bandwidth-selector failure (constant / perfectly-
    # linear outcome); att / se for those horizons are NaN as well.
    bandwidth_diagnostics: Optional[List[Optional[BandwidthResult]]]
    bias_corrected_fit: Optional[List[Optional[BiasCorrectedFit]]]

    # Staggered auto-filter metadata
    filter_info: Optional[Dict[str, Any]]

    def __repr__(self) -> str:
        return (
            f"HeterogeneousAdoptionDiDEventStudyResults("
            f"n_horizons={len(self.event_times)}, "
            f"design={self.design!r}, n_units={self.n_units})"
        )

    def summary(self) -> str:
        """Formatted per-horizon summary table."""
        width = 80
        conf_level = int((1 - self.alpha) * 100)
        lines = [
            "=" * width,
            "HeterogeneousAdoptionDiD Event-Study Results".center(width),
            "=" * width,
            "",
            f"{'Design:':<30} {self.design:>20}",
            f"{'Target parameter:':<30} {self.target_parameter:>20}",
            f"{'d_lower:':<30} {self.d_lower:>20.6g}",
            f"{'D_bar (dose mean):':<30} {self.dose_mean:>20.6g}",
            f"{'First-treatment period (F):':<30} {str(self.F):>20}",
            f"{'Observations (units):':<30} {self.n_units:>20}",
            f"{'Horizons:':<30} {len(self.event_times):>20}",
            f"{'Inference method:':<30} {self.inference_method:>20}",
        ]
        if self.vcov_type is not None:
            if self.cluster_name is not None:
                label = f"CR1 at {self.cluster_name}"
            else:
                label = self.vcov_type
            lines.append(f"{'Variance:':<30} {label:>20}")
        if self.filter_info is not None:
            lines.append(
                f"{'Last-cohort filter (F_last):':<30} "
                f"{str(self.filter_info.get('F_last')):>20}"
            )
            lines.append(
                f"{'  Units kept / dropped:':<30} "
                f"{self.filter_info.get('n_kept', 0)} / "
                f"{self.filter_info.get('n_dropped', 0):<8}".rjust(51)
            )
        lines.extend(
            [
                "",
                "-" * width,
                (
                    f"{'Event-time':>10} {'Estimate':>12} {'Std. Err.':>12} "
                    f"{'t-stat':>10} {'P>|t|':>10} "
                    f"{str(conf_level) + '% CI':>22}"
                ),
                "-" * width,
            ]
        )
        for i, e in enumerate(self.event_times):
            ci_str = f"[{self.conf_int_low[i]:.4f}, {self.conf_int_high[i]:.4f}]"
            # Default float formatting renders non-finite values as "nan";
            # we do not override this here since the column width is fixed
            # and lowercase "nan" is unambiguous.
            se_i = self.se[i]
            t_i = self.t_stat[i]
            p_i = self.p_value[i]
            lines.append(
                f"{int(e):>10} {self.att[i]:>12.4f} "
                f"{se_i:>12.4f} {t_i:>10.3f} {p_i:>10.4f} {ci_str:>22}"
            )
        lines.extend(
            [
                "-" * width,
                "",
                "=" * width,
            ]
        )
        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print the summary to stdout."""
        print(self.summary())

    def to_dict(self) -> Dict[str, Any]:
        """Return results as a dict with per-horizon arrays and scalars.

        Per-horizon arrays are converted to Python lists via
        ``ndarray.tolist()`` (which unwraps NumPy scalar elements to
        native ``int`` / ``float``); scalar fields are coerced to
        native Python types via ``_json_safe_scalar`` where relevant
        (NumPy scalars -> ``.item()``, pandas ``Timestamp`` -> ISO
        string, ``Timedelta`` -> ISO string). The returned dict is
        JSON-serializable directly via ``json.dumps``.
        """
        return {
            "event_times": self.event_times.tolist(),
            "att": self.att.tolist(),
            "se": self.se.tolist(),
            "t_stat": self.t_stat.tolist(),
            "p_value": self.p_value.tolist(),
            "conf_int_low": self.conf_int_low.tolist(),
            "conf_int_high": self.conf_int_high.tolist(),
            "n_obs_per_horizon": self.n_obs_per_horizon.tolist(),
            "alpha": float(self.alpha),
            "design": self.design,
            "target_parameter": self.target_parameter,
            "d_lower": float(self.d_lower),
            "dose_mean": float(self.dose_mean),
            "F": _json_safe_scalar(self.F),
            "n_units": int(self.n_units),
            "inference_method": self.inference_method,
            "vcov_type": self.vcov_type,
            "cluster_name": self.cluster_name,
            "filter_info": _json_safe_filter_info(self.filter_info),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a tidy per-horizon DataFrame.

        Columns: ``event_time, att, se, t_stat, p_value, conf_int_low,
        conf_int_high, n_obs``. One row per event-time horizon.
        """
        return pd.DataFrame(
            {
                "event_time": self.event_times,
                "att": self.att,
                "se": self.se,
                "t_stat": self.t_stat,
                "p_value": self.p_value,
                "conf_int_low": self.conf_int_low,
                "conf_int_high": self.conf_int_high,
                "n_obs": self.n_obs_per_horizon,
            }
        )


# =============================================================================
# Panel validation and aggregation
# =============================================================================


def _validate_had_panel(
    data: pd.DataFrame,
    outcome_col: str,
    dose_col: str,
    time_col: str,
    unit_col: str,
    first_treat_col: Optional[str],
) -> Tuple[int, int]:
    """Validate a HAD panel and return ``(t_pre, t_post)``.

    Enforces the Phase 2a panel contract:

    - All required columns present.
    - Exactly two distinct time periods. Staggered timing (``>2`` periods)
      with ``first_treat_col=None`` raises; with ``first_treat_col`` it
      also raises (multi-period reduction is Phase 2b).
    - Balanced panel (all units observed at both periods).
    - ``D_{g, t_pre} == 0`` for all units (HAD no-unit-untreated pre-period).
    - No NaN in outcome, dose, or unit columns.

    Parameters
    ----------
    data : pd.DataFrame
    outcome_col, dose_col, time_col, unit_col : str
    first_treat_col : str or None
        Optional column for cross-validation. Supplied column must contain
        ``0`` for never-treated and the post-period value for treated units.

    Returns
    -------
    tuple[Any, Any]
        ``(t_pre, t_post)`` - the two period identifiers identified by
        the HAD dose invariant (``t_pre`` is the period with dose == 0
        for all units; ``t_post`` is the other period). Supports
        arbitrary-dtype period labels (int, str, datetime, etc.) rather
        than relying on ordinal / lexicographic sort.

    Raises
    ------
    ValueError
    """
    required = [outcome_col, dose_col, time_col, unit_col]
    if first_treat_col is not None:
        required.append(first_treat_col)
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing column(s) in data: {missing}. Required: {required}.")

    periods_list = list(data[time_col].unique())
    if len(periods_list) < 2:
        raise ValueError(
            f"HAD requires a two-period panel; got {len(periods_list)} distinct "
            f"period(s) in column {time_col!r}."
        )
    if len(periods_list) > 2:
        raise ValueError(
            f"HAD with aggregate='overall' requires exactly two time "
            f"periods (got {len(periods_list)} in {time_col!r}). For "
            f"multi-period panels, pass aggregate='event_study' (paper "
            f"Appendix B.2 multi-period event-study extension) which "
            f"produces per-event-time WAS estimates."
        )

    # Balanced-panel check: every unit appears exactly once per period.
    counts = data.groupby([unit_col, time_col]).size()
    if (counts != 1).any():
        n_bad = int((counts != 1).sum())
        raise ValueError(
            f"Unbalanced panel: {n_bad} unit-period cells have != 1 "
            f"observation. HAD requires a balanced two-period panel "
            f"(each unit observed exactly once at each period)."
        )
    unit_counts = data.groupby(unit_col)[time_col].nunique()
    incomplete = unit_counts[unit_counts != 2]
    if len(incomplete) > 0:
        raise ValueError(
            f"Unbalanced panel: {len(incomplete)} unit(s) do not appear "
            f"in both periods. HAD requires a balanced two-period panel."
        )

    # NaN checks on key columns.
    for col in [outcome_col, dose_col, unit_col]:
        if bool(data[col].isna().any()):
            n_nan = int(data[col].isna().sum())
            raise ValueError(
                f"{n_nan} NaN value(s) found in column {col!r}. HAD "
                f"does not silently drop NaN rows; drop or impute before "
                f"calling fit()."
            )

    # Identify t_pre and t_post by the HAD invariant rather than by
    # lexicographic sort on the time labels: D_{g, t_pre} = 0 for all
    # units (paper Section 2 no-unit-untreated pre-period convention).
    # Sorting labels alphabetically reverses valid chronologies like
    # ("pre", "post") where ordering is semantic, not alphabetic.
    per_period_nonzero: Dict[Any, int] = {}
    for p in periods_list:
        p_doses = np.asarray(data.loc[data[time_col] == p, dose_col], dtype=np.float64)
        per_period_nonzero[p] = int((p_doses != 0).sum())
    all_zero_periods = [p for p, nz in per_period_nonzero.items() if nz == 0]
    if len(all_zero_periods) == 0:
        # Neither period has all-zero dose: HAD pre-period contract violated.
        stats_str = ", ".join(f"{p!r}: {nz} nonzero" for p, nz in per_period_nonzero.items())
        raise ValueError(
            f"HAD requires D_{{g,1}} = 0 for all units (pre-period "
            f"untreated). Neither period in column {time_col!r} has "
            f"all-zero dose ({stats_str}). Exactly one period must be "
            f"the pre-treatment period with D_{{g,1}} = 0 for every unit; "
            f"drop rows with nonzero pre-period dose or verify the dose "
            f"column."
        )
    if len(all_zero_periods) == 2:
        raise ValueError(
            f"HAD requires variation in D_{{g,2}} for estimation. Both "
            f"periods in column {time_col!r} have all-zero dose, so "
            f"there is no treatment assignment to estimate."
        )
    t_pre = all_zero_periods[0]
    t_post = [p for p in periods_list if p != t_pre][0]

    # Post-period nonnegative-dose check on the ORIGINAL (unshifted) dose
    # scale. Front-door rejection per paper Assumption (dose definition
    # Section 2) which treats D_{g,2} as nonnegative. Without this
    # check, negative original doses would only surface after the
    # regressor shift in ``_fit_continuous`` via Phase 1c's
    # ``_validate_had_inputs``, which references the shifted values
    # and would confuse users about which column is malformed.
    post_mask = data[time_col] == t_post
    post_doses = np.asarray(data.loc[post_mask, dose_col], dtype=np.float64)
    neg_post = post_doses < 0
    if neg_post.any():
        n_neg = int(neg_post.sum())
        min_neg = float(post_doses[neg_post].min())
        raise ValueError(
            f"HAD requires D_{{g,2}} >= 0 for all units (paper Section "
            f"2 dose definition). {n_neg} unit(s) have negative post-"
            f"period dose at t_post={t_post} (min={min_neg!r}). Drop "
            f"these units or verify the dose column."
        )

    # Optional value-domain validation via first_treat_col: if supplied,
    # every unit's first_treat value must be in {0, t_post} (0 = never
    # treated, t_post = treated in the second period). This is a value-
    # domain check that catches typos and staggered-timing mix-ups; it
    # does NOT cross-validate first_treat against post-period dose
    # (D_{g, t_post} remains the primary signal). Extended cross-checks
    # are queued for a follow-up PR. The check is DTYPE-AGNOSTIC: it uses
    # pd.isna() for missingness and raw-value membership against
    # {0, t_post} so that string-labelled periods (e.g., ("A", "B")) with
    # first_treat in {0, "B"} are supported.
    if first_treat_col is not None:
        # Row-level NaN check: `groupby().first()` skips NaNs silently, so a
        # unit with rows [valid, NaN] would pass a collapsed check. Validate
        # raw per-row values first, then verify per-unit constancy with
        # `nunique(dropna=False)` so within-unit NaN variation is caught.
        ft_raw = data[first_treat_col]
        if bool(ft_raw.isna().any()):
            n_nan = int(ft_raw.isna().sum())
            raise ValueError(
                f"first_treat_col={first_treat_col!r} contains "
                f"{n_nan} NaN value(s) at the row level. Use 0 for "
                f"never-treated units and t_post for treated, and drop "
                f"or impute any NaN rows before calling fit()."
            )
        # Row-level domain check: every row (not just the collapsed first())
        # must be in {0, t_post}. Catches mixed-row malformed inputs where
        # a unit has [valid, invalid].
        valid_values = {0, t_post}
        observed_raw = set(ft_raw.unique().tolist())
        bad = sorted(observed_raw - valid_values, key=lambda x: str(x))
        if bad:
            raise ValueError(
                f"first_treat_col={first_treat_col!r} contains value(s) "
                f"{bad} outside the allowed set {{0, {t_post!r}}} for a "
                f"two-period HAD panel. Staggered timing with multiple "
                f"cohorts is Phase 2b."
            )
        # Within-unit consistency: every unit must have a single
        # first_treat value across its rows. Uses dropna=False so a unit
        # with [value, NaN] counts as 2 unique values (caught above by
        # the NaN check anyway, but this is belt-and-suspenders).
        ft_per_unit_nunique = data.groupby(unit_col)[first_treat_col].nunique(dropna=False)
        if (ft_per_unit_nunique > 1).any():
            n_bad = int((ft_per_unit_nunique > 1).sum())
            raise ValueError(
                f"first_treat_col={first_treat_col!r} is not constant "
                f"within unit for {n_bad} unit(s). Each unit must have "
                f"a single first_treat value across both observed periods."
            )

    return t_pre, t_post


def _validate_had_panel_event_study(
    data: pd.DataFrame,
    outcome_col: str,
    dose_col: str,
    time_col: str,
    unit_col: str,
    first_treat_col: Optional[str],
) -> Tuple[Any, List[Any], List[Any], pd.DataFrame, Optional[Dict[str, Any]]]:
    """Validate a HAD panel for multi-period event-study mode (Phase 2b).

    Implements paper Appendix B.2 contract: a common treatment date ``F``
    where ``D_{g,t} = 0`` for all units at ``t < F`` and some units have
    ``D_{g,t} > 0`` for ``t >= F``. Requires ``len(periods) > 2`` with at
    least one pre-period (``t < F``, all D=0) and at least one post-period
    (``t >= F``, some D > 0).

    Staggered-timing handling: when ``first_treat_col`` is supplied and
    indicates more than one nonzero cohort, the panel is auto-filtered to
    the LAST cohort (``F_last = max(cohorts)``) per paper Appendix B.2
    prescription "did_had may be used only for the last treatment cohort
    in a staggered design". A ``UserWarning`` is emitted with drop-counts.

    Parameters
    ----------
    data, outcome_col, dose_col, time_col, unit_col, first_treat_col
        As in :func:`_validate_had_panel`.

    Returns
    -------
    F : period label
        First-treatment period (the earliest period where any unit has
        ``D > 0`` in the filtered data).
    t_pre_list : list
        Pre-period labels (``t < F``, all D=0), sorted by natural ordering
        on the column dtype.
    t_post_list : list
        Post-period labels (``t >= F``, some D > 0), sorted.
    data_filtered : pd.DataFrame
        Input with earlier cohorts (``first_treat`` in ``dropped_cohorts``)
        dropped if staggered; never-treated units (``first_treat = 0``)
        are RETAINED per paper Appendix B.2's "there must be an untreated
        group" requirement. Identical to input when no staggered filter
        applies.
    filter_info : dict or None
        Populated on staggered filter with keys ``F_last`` (kept cohort
        label), ``n_kept`` (last-cohort units PLUS never-treated units),
        ``n_dropped`` (earlier-cohort units removed), ``dropped_cohorts``
        (list of earlier cohort labels). ``None`` otherwise.

    Raises
    ------
    ValueError
        On missing columns, NaN in key columns, malformed panel, dose-
        invariant violations, or no-treatment detected.
    """
    required = [outcome_col, dose_col, time_col, unit_col]
    if first_treat_col is not None:
        required.append(first_treat_col)
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Missing column(s) in data: {missing}. Required: {required}.")

    periods_list = list(data[time_col].unique())
    if len(periods_list) < 3:
        raise ValueError(
            f"HAD with aggregate='event_study' requires more than two "
            f"time periods (got {len(periods_list)} in {time_col!r}). "
            f"For two-period panels, pass aggregate='overall' (Phase 2a "
            f"single-period WAS)."
        )

    # Ordered-time-type check. Paper Appendix B.2 event-time horizons
    # require chronological ordering of periods (anchor at F-1, horizons
    # e = t - F relative to F). Phase 2a two-period panels can use the
    # dose invariant alone to distinguish pre from post without needing
    # chronological order, so string labels ("pre", "post") work there.
    # For multi-period event-study, multiple pre-periods all have D=0
    # and multiple post-periods may both have D>0, so dose alone cannot
    # recover chronology: we must trust the time column's natural order.
    # Raw lexicographic sort on object/string labels silently misorders
    # panels like "pre1"/"pre2"/"post1"/"post2" or month-name labels.
    # Require an explicitly-ordered time representation.
    time_dtype = data[time_col].dtype
    if not (
        pd.api.types.is_numeric_dtype(time_dtype)
        or pd.api.types.is_datetime64_any_dtype(time_dtype)
        or (isinstance(time_dtype, pd.CategoricalDtype) and bool(time_dtype.ordered))
    ):
        raise ValueError(
            f"HAD aggregate='event_study' requires an ordered time "
            f"column. time_col={time_col!r} has dtype={time_dtype!r}, "
            f"which has no defined chronological order; raw sort would "
            f"fall back to lexicographic ordering and silently misindex "
            f"event-time horizons (e.g., 'pre1'/'pre2'/'post1'/'post2' "
            f"sorts lexicographically but not chronologically). "
            f"Convert time_col to numeric (e.g., integer year), "
            f"datetime, or ordered categorical "
            f"(``pd.Categorical(..., ordered=True, categories=[...])``) "
            f"before calling fit() with aggregate='event_study'."
        )

    # Construct the chronological sort key once, shared across every
    # downstream ordering: cohort ranking, pre/post period sorting, and
    # contiguity checks. Ordered categoricals use their declared
    # category index (``list(categorical)`` strips the ordering and
    # falls back to string comparison); numeric / datetime use natural
    # Python order. Reused by ``_aggregate_multi_period_first_differences``
    # via a parallel construction in that helper (both read the same
    # ``time_dtype``).
    if isinstance(time_dtype, pd.CategoricalDtype) and time_dtype.ordered:
        _cat_order = {c: i for i, c in enumerate(time_dtype.categories)}

        def _sort_key(x: Any) -> Tuple[bool, Any]:
            return (x is None, _cat_order.get(x, len(_cat_order)))

    else:

        def _sort_key(x: Any) -> Tuple[bool, Any]:
            return (x is None, x)

    # NaN checks on key columns (before any filter).
    for col in [outcome_col, dose_col, unit_col]:
        if bool(data[col].isna().any()):
            n_nan = int(data[col].isna().sum())
            raise ValueError(
                f"{n_nan} NaN value(s) found in column {col!r}. HAD "
                f"does not silently drop NaN rows; drop or impute before "
                f"calling fit()."
            )

    # Cohort detection and staggered-timing auto-filter.
    filter_info: Optional[Dict[str, Any]] = None
    data_filtered = data
    if first_treat_col is not None:
        ft_raw = data[first_treat_col]
        if bool(ft_raw.isna().any()):
            n_nan = int(ft_raw.isna().sum())
            raise ValueError(
                f"first_treat_col={first_treat_col!r} contains "
                f"{n_nan} NaN value(s) at the row level. Use 0 for "
                f"never-treated units and the treatment-start period "
                f"for treated units. Drop or impute any NaN rows "
                f"before calling fit()."
            )
        # Within-unit constancy check.
        ft_per_unit_nunique = data.groupby(unit_col)[first_treat_col].nunique(dropna=False)
        if (ft_per_unit_nunique > 1).any():
            n_bad = int((ft_per_unit_nunique > 1).sum())
            raise ValueError(
                f"first_treat_col={first_treat_col!r} is not constant "
                f"within unit for {n_bad} unit(s). Each unit must have "
                f"a single first_treat value across all observed periods."
            )
        # Cross-validate first_treat_col against observed first-positive-
        # dose period for every unit. A mislabeled cohort column would
        # otherwise silently select the wrong cohort as F_last and return
        # event-study estimates for the wrong units. Contract:
        #   - declared first_treat == 0: unit must have D == 0 at all t
        #     (never-treated)
        #   - declared first_treat == F_g > 0: unit's first period with
        #     D > 0 must equal F_g
        df_for_check = data.sort_values([unit_col, time_col])
        pos_rows = df_for_check.loc[df_for_check[dose_col] > 0]
        actual_first_pos = pos_rows.groupby(unit_col)[time_col].first()
        declared_ft = df_for_check.groupby(unit_col)[first_treat_col].first()
        n_mismatch = 0
        example_mismatch: Optional[Tuple[Any, Any, Any]] = None
        for u, declared in declared_ft.items():
            actual = actual_first_pos.get(u, None)
            if declared == 0:
                if actual is not None:
                    n_mismatch += 1
                    if example_mismatch is None:
                        example_mismatch = (u, declared, actual)
            else:
                if actual is None or actual != declared:
                    n_mismatch += 1
                    if example_mismatch is None:
                        example_mismatch = (u, declared, actual)
        if n_mismatch > 0:
            u, declared, actual = example_mismatch  # type: ignore[misc]
            raise ValueError(
                f"first_treat_col={first_treat_col!r} disagrees with the "
                f"observed dose path for {n_mismatch} unit(s). Example: "
                f"unit={u!r} declares first_treat={declared!r} but the "
                f"unit's first period with D>0 is {actual!r} "
                f"(None means never-treated). A mislabeled cohort column "
                f"would silently select the wrong cohort as F_last in the "
                f"last-cohort auto-filter. Fix the first_treat_col values "
                f"to equal each unit's first positive-dose period (or 0 "
                f"for never-treated) before calling fit()."
            )
        # Identify cohorts (nonzero first_treat values). Sort using
        # ``_sort_key`` (chronological order from ``time_dtype``), NOT
        # raw Python sort: first_treat values are period labels and
        # must rank chronologically so ``F_last = cohorts[-1]`` is the
        # chronologically latest cohort. Under ordered-categorical time
        # labels (e.g. month names), raw Python sort is lexicographic
        # and would silently pick the wrong ``F_last``.
        ft_unique = list(pd.unique(ft_raw))
        cohorts = sorted(
            [v for v in ft_unique if v != 0 and not (isinstance(v, float) and np.isnan(v))],
            key=_sort_key,
        )
        if len(cohorts) == 0:
            raise ValueError(
                f"first_treat_col={first_treat_col!r} has no nonzero "
                f"cohort values (all units appear never-treated). HAD "
                f"requires at least one treated cohort with "
                f"first_treat > 0 to identify a WAS effect."
            )
        if len(cohorts) > 1:
            F_last = cohorts[-1]
            dropped_cohorts = cohorts[:-1]
            # Filter: keep last-cohort AND never-treated (first_treat == 0).
            # Paper Appendix B.2: "in designs with variation in treatment
            # timing, there must be an untreated group, at least till the
            # period where the last cohort gets treated". Never-treated
            # units (first_treat=0) satisfy the dose invariant for every
            # period (D=0 throughout) and serve as the untreated-group
            # comparison at every pre-period horizon. Keeping them matches
            # the paper's "there must be an untreated group" language and
            # preserves Design 1' identifiability (boundary at 0) when the
            # last-cohort doses are uniformly positive. Only earlier-treated
            # cohorts (first_treat in dropped_cohorts) are dropped.
            keep_mask = (data[first_treat_col] == F_last) | (data[first_treat_col] == 0)
            dropped_unit_ids = set(data.loc[~keep_mask, unit_col].unique())
            kept_unit_ids = set(data.loc[keep_mask, unit_col].unique())
            data_filtered = data.loc[keep_mask].copy()
            n_dropped = len(dropped_unit_ids - kept_unit_ids)
            n_kept = len(kept_unit_ids)
            if n_kept == 0:
                raise ValueError(
                    f"Staggered auto-filter to last cohort "
                    f"(F_last={F_last!r}) left 0 units. Verify "
                    f"first_treat_col={first_treat_col!r} contains the "
                    f"expected cohort labels."
                )
            filter_info = {
                "F_last": F_last,
                "n_kept": n_kept,
                "n_dropped": n_dropped,
                "dropped_cohorts": dropped_cohorts,
            }
            warnings.warn(
                f"Staggered-timing panel detected: {len(cohorts)} distinct "
                f"nonzero cohorts in first_treat_col={first_treat_col!r} "
                f"({cohorts!r}). Auto-filtering to the last cohort "
                f"(F_last={F_last!r}) plus never-treated units "
                f"(first_treat=0): {n_kept} units kept, {n_dropped} "
                f"earlier-cohort units dropped (from cohorts "
                f"{dropped_cohorts!r}). HAD applies only to the last "
                f"treatment cohort in staggered designs (paper Appendix "
                f"B.2); never-treated units are retained as the untreated-"
                f"group comparison per the paper's \"there must be an "
                f'untreated group" requirement. For earlier-cohort '
                f"effects, use ChaisemartinDHaultfoeuille "
                f"(did_multiplegt_dyn).",
                UserWarning,
                stacklevel=3,
            )
            # After filter, re-read periods_list (cohort filter may have
            # dropped some periods if earlier cohorts contributed uniquely).
            periods_list = list(data_filtered[time_col].unique())
            if len(periods_list) < 3:
                raise ValueError(
                    f"After staggered auto-filter to last cohort "
                    f"(F_last={F_last!r}), only {len(periods_list)} "
                    f"distinct time periods remain in {time_col!r}. "
                    f"Event-study requires >2 periods; the filtered "
                    f"panel is too small. Pass aggregate='overall' on "
                    f"a two-period subset, or supply data with more "
                    f"pre- or post-periods for the last cohort."
                )

    # Balanced panel on the (possibly-filtered) data: every unit appears
    # exactly once per period. ``observed=True`` tells categorical
    # groupby to count only OBSERVED unit-period cells. Without it, a
    # time_col with an ordered-categorical dtype carrying extra unused
    # category levels (beyond the periods actually present in the data)
    # would expand to zero-count cells and the balance check would
    # falsely reject valid panels. The rest of the validator is keyed
    # to ``periods_list`` (observed unique values) so this stays
    # consistent.
    counts = data_filtered.groupby([unit_col, time_col], observed=True).size()
    if (counts != 1).any():
        n_bad = int((counts != 1).sum())
        raise ValueError(
            f"Unbalanced panel: {n_bad} unit-period cells have != 1 "
            f"observation. HAD requires a balanced panel (each unit "
            f"observed exactly once at each period)."
        )
    unit_counts = data_filtered.groupby(unit_col)[time_col].nunique()
    incomplete = unit_counts[unit_counts != len(periods_list)]
    if len(incomplete) > 0:
        raise ValueError(
            f"Unbalanced panel: {len(incomplete)} unit(s) do not appear "
            f"in all {len(periods_list)} periods. HAD requires a balanced "
            f"panel (each unit observed at every period)."
        )

    # Dose-invariant period classification on filtered data.
    per_period_nonzero: Dict[Any, int] = {}
    for p in periods_list:
        p_doses = np.asarray(
            data_filtered.loc[data_filtered[time_col] == p, dose_col], dtype=np.float64
        )
        per_period_nonzero[p] = int((p_doses != 0).sum())
    t_pre_list_unsorted = [p for p, nz in per_period_nonzero.items() if nz == 0]
    t_post_list_unsorted = [p for p, nz in per_period_nonzero.items() if nz > 0]

    if len(t_pre_list_unsorted) == 0:
        stats_str = ", ".join(f"{p!r}: {nz} nonzero" for p, nz in per_period_nonzero.items())
        raise ValueError(
            f"HAD requires D_{{g,t}} = 0 for all units in at least one "
            f"pre-period. No period in column {time_col!r} has all-zero "
            f"dose ({stats_str}). The panel has no identifiable baseline."
        )
    if len(t_post_list_unsorted) == 0:
        raise ValueError(
            f"HAD requires at least one period with nonzero dose for "
            f"some unit. All periods in column {time_col!r} have all-"
            f"zero dose; there is no treatment to estimate."
        )

    # Sort using the same ``_sort_key`` already constructed for cohorts
    # (ordered-categorical uses declared category order; numeric /
    # datetime use natural Python order).
    t_pre_list = sorted(t_pre_list_unsorted, key=_sort_key)
    t_post_list = sorted(t_post_list_unsorted, key=_sort_key)

    # Contiguity check: all pre < all post in the natural ordering.
    # The HAD dose invariant requires a single transition from all-zero
    # to any-nonzero; interleaved pre/post periods indicate a malformed
    # panel (e.g., dose going back to zero after treatment, or mixing
    # never-treated units with out-of-order labels). Uses ``_sort_key``
    # so ordered categoricals respect their declared category order.
    if t_pre_list and t_post_list:
        max_pre = t_pre_list[-1]
        min_post = t_post_list[0]
        contiguous = _sort_key(max_pre) < _sort_key(min_post)
        if not contiguous:
            raise ValueError(
                f"HAD dose invariant violated: pre-periods (all D=0) "
                f"and post-periods (some D>0) are not contiguous. "
                f"Pre-periods: {t_pre_list!r}; post-periods: "
                f"{t_post_list!r}. The dose sequence must transition "
                f"from all-zero to nonzero exactly once. For panels "
                f"where dose varies non-monotonically (e.g., reversed "
                f"treatment, switching), use "
                f"ChaisemartinDHaultfoeuille (did_multiplegt_dyn)."
            )

    F = t_post_list[0]  # earliest post-period

    # Post-period nonnegative-dose check on the filtered data.
    post_mask = data_filtered[time_col].isin(t_post_list)
    post_doses = np.asarray(data_filtered.loc[post_mask, dose_col], dtype=np.float64)
    neg_post = post_doses < 0
    if neg_post.any():
        n_neg = int(neg_post.sum())
        min_neg = float(post_doses[neg_post].min())
        raise ValueError(
            f"HAD requires D_{{g,t}} >= 0 for all units in post-periods "
            f"(paper Section 2 dose definition). {n_neg} unit-period "
            f"cell(s) have negative dose at t >= F={F!r} "
            f"(min={min_neg!r}). Drop these units or verify the dose "
            f"column."
        )

    # Staggered-without-``first_treat_col`` detection. When cohort metadata
    # is not supplied, the dose-invariant period classification still
    # declares t=F=min-post-period based on "any unit has nonzero dose".
    # That silently accepts staggered panels where units have DIFFERENT
    # first-positive-dose periods: the later-treated cohorts enter
    # ``d_arr`` as zero-dose "controls" at the inferred F, violating
    # paper Appendix B.2's last-cohort-only contract. Compute per-unit
    # first-positive-dose period directly from the dose path and raise
    # if multiple cohorts are present, directing users to pass
    # ``first_treat_col`` (which activates the last-cohort auto-filter)
    # or to use ChaisemartinDHaultfoeuille for full staggered support.
    if first_treat_col is None:
        df_sorted = data_filtered.sort_values([unit_col, time_col])
        # For each unit, the first period at which dose > 0.
        pos_mask_global = df_sorted[dose_col] > 0
        first_pos_per_unit = df_sorted.loc[pos_mask_global].groupby(unit_col)[time_col].first()
        cohort_labels = list(first_pos_per_unit.unique())
        if len(cohort_labels) > 1:
            # Sort chronologically via the validated time-column order.
            distinct_cohorts = sorted(cohort_labels, key=_sort_key)
            raise ValueError(
                f"Staggered-timing panel detected (first_treat_col is "
                f"None): {len(distinct_cohorts)} distinct first-positive-"
                f"dose periods {distinct_cohorts!r} across units. HAD's "
                f"last-cohort auto-filter (paper Appendix B.2) only runs "
                f"when first_treat_col is supplied so the estimator can "
                f"identify cohorts. Pass first_treat_col=<column> to "
                f"enable the auto-filter to the last cohort, or use "
                f"ChaisemartinDHaultfoeuille (did_multiplegt_dyn) for "
                f"full staggered support."
            )

    # Constant post-period dose check. Paper Appendix B.2 assumes
    # "once treated, stay treated with the same dose"; the event-study
    # aggregation uses ``D_{g, F}`` as the single regressor for every
    # event-time horizon. Panels where a unit's dose varies across
    # post-periods (e.g., phased adoption, dose changes after F) would
    # silently misattribute later-horizon effects to the period-F dose.
    # Reject front-door with a redirect to ChaisemartinDHaultfoeuille
    # for genuinely time-varying post-treatment doses.
    if len(t_post_list) > 1:
        post_data = data_filtered.loc[post_mask]
        dose_spread_per_unit = post_data.groupby(unit_col)[dose_col].agg(
            lambda x: float(x.max() - x.min())
        )
        abs_max_dose = float(np.max(np.abs(post_doses))) if post_doses.size else 0.0
        tol = 1e-12 * max(1.0, abs_max_dose)
        bad_mask = dose_spread_per_unit > tol
        if bool(bad_mask.any()):
            n_bad = int(bad_mask.sum())
            max_spread = float(dose_spread_per_unit.max())
            raise ValueError(
                f"HAD event-study requires constant dose within unit for "
                f"all post-treatment periods t >= F={F!r}. {n_bad} unit(s) "
                f"have time-varying doses across post-periods "
                f"{t_post_list!r} (max within-unit spread={max_spread!r}, "
                f"tolerance={tol!r}). The aggregation uses D_{{g, F}} as "
                f"the single regressor for every event-time horizon "
                f"(paper Appendix B.2 constant-dose convention), so "
                f"silently accepting time-varying post-treatment doses "
                f"would misattribute later-horizon effects. For genuinely "
                f"time-varying post-treatment doses use "
                f"ChaisemartinDHaultfoeuille (did_multiplegt_dyn)."
            )

    return F, t_pre_list, t_post_list, data_filtered, filter_info


def _aggregate_first_difference(
    data: pd.DataFrame,
    outcome_col: str,
    dose_col: str,
    time_col: str,
    unit_col: str,
    t_pre: Any,
    t_post: Any,
    cluster_col: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Reduce a balanced two-period panel to unit-level first differences.

    Returns
    -------
    d_arr : np.ndarray, shape (G,)
        Post-period dose ``D_{g,2}`` per unit, ordered by sorted unit id.
    dy_arr : np.ndarray, shape (G,)
        First-difference outcome ``Y_{g,2} - Y_{g,1}`` per unit.
    cluster_arr : np.ndarray or None, shape (G,)
        Cluster IDs per unit (must be unit-constant); ``None`` when
        ``cluster_col`` is ``None``.
    unit_ids : np.ndarray, shape (G,)
        Sorted unit identifiers (useful for debugging / test inspection).
    """
    df = data.sort_values([unit_col, time_col]).reset_index(drop=True)
    pre = df[df[time_col] == t_pre].set_index(unit_col).sort_index()
    post = df[df[time_col] == t_post].set_index(unit_col).sort_index()

    if bool(pre.index.equals(post.index)) is False:
        # Should not reach here after _validate_had_panel balanced-panel
        # check, but belt-and-suspenders:
        raise ValueError(
            "Internal error: pre and post period unit indices do not "
            "match after balanced-panel validation. Please report this."
        )

    unit_ids = np.asarray(pre.index)
    d_arr = post[dose_col].to_numpy(dtype=np.float64)
    dy_arr = post[outcome_col].to_numpy(dtype=np.float64) - pre[outcome_col].to_numpy(
        dtype=np.float64
    )

    cluster_arr: Optional[np.ndarray] = None
    if cluster_col is not None:
        if cluster_col not in data.columns:
            raise ValueError(f"cluster column {cluster_col!r} not found in data.")
        # Row-level NaN check: `groupby().first()` skips NaNs silently, so a
        # unit with rows [valid, NaN] would pass a collapsed check. Validate
        # raw per-row values first so any NaN in the cluster column is
        # rejected up-front, not masked by the per-unit collapse.
        cluster_raw = data[cluster_col]
        if bool(cluster_raw.isna().any()):
            n_nan = int(cluster_raw.isna().sum())
            raise ValueError(
                f"cluster column {cluster_col!r} contains {n_nan} NaN "
                f"value(s) at the row level. Silent row dropping is "
                f"disabled; drop or impute cluster ids before calling fit()."
            )
        # Cluster must be unit-constant. `nunique(dropna=False)` counts NaN
        # as a distinct value so rows like [value, NaN] register as 2
        # uniques (also caught by the row-level NaN check above).
        cluster_per_unit = df.groupby(unit_col)[cluster_col].nunique(dropna=False)
        if (cluster_per_unit > 1).any():
            n_bad = int((cluster_per_unit > 1).sum())
            raise ValueError(
                f"cluster column {cluster_col!r} is not constant within "
                f"unit for {n_bad} unit(s). Cluster must be unit-level "
                f"(e.g., cluster_col=unit_col, or a coarser grouping "
                f"where each unit belongs to a single cluster)."
            )
        cluster_arr = df.groupby(unit_col)[cluster_col].first().sort_index().to_numpy()

    return d_arr, dy_arr, cluster_arr, unit_ids


def _aggregate_unit_weights(
    data: pd.DataFrame,
    weights_arr: np.ndarray,
    unit_col: str,
) -> np.ndarray:
    """Aggregate per-row weights to per-unit, enforcing constant-within-unit.

    HAD's continuous path operates at the per-unit level (G rows, one per
    unit) so survey weights — which typically arrive per-row in a long
    panel — must collapse to a single value per unit. The paper's weighted
    extension assumes sampling weights assigned at the sampling-unit level
    (standard BRFSS/CPS/NHANES convention), so any within-unit weight
    variance is interpreted as user error and rejected front-door rather
    than silently mean-rolled (per ``feedback_no_silent_failures``).

    Parameters
    ----------
    data : pd.DataFrame
        Long panel; the same frame passed to ``_aggregate_first_difference``.
    weights_arr : np.ndarray, shape (n_rows,)
        Row-aligned weights.
    unit_col : str
        Unit identifier column.

    Returns
    -------
    w_unit : np.ndarray, shape (G,)
        Per-unit weights sorted by unit id to align with the d/dy arrays
        returned by ``_aggregate_first_difference``.

    Raises
    ------
    ValueError
        Shape mismatch, non-finite weights, negative weights, zero-sum
        weights, or weights that vary within a unit.
    """
    n_rows = int(data.shape[0])
    w = np.asarray(weights_arr, dtype=np.float64).ravel()
    if w.shape[0] != n_rows:
        raise ValueError(
            f"weights length ({w.shape[0]}) does not match number of "
            f"rows in data ({n_rows})."
        )
    if not np.all(np.isfinite(w)):
        raise ValueError("weights contains non-finite values (NaN or Inf).")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    if np.sum(w) <= 0:
        raise ValueError(
            "weights sum to zero — no observations have positive weight."
        )

    df = data.reset_index(drop=True).copy()
    df["_w_tmp__"] = w
    w_per_unit = df.groupby(unit_col)["_w_tmp__"].agg(
        lambda s: (float(s.min()), float(s.max()))
    )
    varying = w_per_unit.apply(lambda t: not np.isclose(t[0], t[1], rtol=1e-12, atol=0.0))
    if bool(varying.any()):
        n_bad = int(varying.sum())
        raise ValueError(
            f"weights vary within {n_bad} unit(s). HAD's continuous path "
            f"requires unit-level sampling weights (constant within each "
            f"unit). Either pre-aggregate your weights to the unit level "
            f"or pass a unit-constant weight column. Per-obs weights that "
            f"vary within unit would require an obs-level estimator not "
            f"available on this path."
        )
    w_unit = (
        df.groupby(unit_col)["_w_tmp__"].first().sort_index().to_numpy(dtype=np.float64)
    )
    return w_unit


def _aggregate_multi_period_first_differences(
    data: pd.DataFrame,
    outcome_col: str,
    dose_col: str,
    time_col: str,
    unit_col: str,
    F: Any,
    t_pre_list: List[Any],
    t_post_list: List[Any],
    cluster_col: Optional[str],
) -> Tuple[np.ndarray, Dict[int, np.ndarray], Optional[np.ndarray], np.ndarray, Any]:
    """Reduce a multi-period HAD panel to per-horizon first differences.

    For each period ``t`` other than the anchor ``t_anchor = F - 1`` (the
    last pre-period), computes the unit-level first difference
    ``ΔY_{g,t} = Y_{g,t} - Y_{g, t_anchor}`` and stores it under the event
    time ``e = rank(t) - rank(F)`` where ``rank`` is the natural ordering
    on the period column (so ``e = 0`` at ``t = F``, ``e = 1`` at the next
    post-period, etc., and ``e <= -2`` for pre-period placebos).

    The single dose regressor is ``D_{g, F}`` (the dose at the first
    treatment period), reused for every horizon. Paper Appendix B.2
    convention assumes "once treated, stay treated with same dose"; this
    helper uses the period-F dose for every horizon, so time-varying
    post-period dose is treated as the realized F-period dose.

    Parameters
    ----------
    data : pd.DataFrame
        Validated multi-period panel (already passed
        ``_validate_had_panel_event_study`` and any staggered filter).
    outcome_col, dose_col, time_col, unit_col, cluster_col : str
        Column names.
    F : period label
        First-treatment period (from the validator).
    t_pre_list, t_post_list : list
        Period labels partitioned by the dose invariant, sorted
        ascending.

    Returns
    -------
    d_arr : np.ndarray, shape (G,)
        Post-period dose ``D_{g, F}`` per unit.
    dy_dict : dict[int, np.ndarray]
        Maps event time ``e`` to first-difference outcome
        ``Y_{g, t} - Y_{g, F-1}`` per unit, for every ``t`` except the
        anchor. Keys cover all horizons EXCEPT ``e = -1`` (the anchor
        gives ``ΔY = 0`` trivially).
    cluster_arr : np.ndarray or None, shape (G,)
        Cluster IDs per unit when ``cluster_col`` is supplied.
    unit_ids : np.ndarray, shape (G,)
        Sorted unit identifiers.
    t_anchor : period label
        The anchor period used (``F - 1`` in the natural period ordering;
        equal to the LAST pre-period).
    """
    df = data.sort_values([unit_col, time_col]).reset_index(drop=True)
    # Period sort respects ordered categorical dtypes (matches
    # ``_validate_had_panel_event_study``). The validator already
    # enforces a numeric / datetime / ordered-categorical dtype on the
    # event-study path, so ``_sort_key`` lookups are well-defined here.
    time_dtype = data[time_col].dtype
    if isinstance(time_dtype, pd.CategoricalDtype) and time_dtype.ordered:
        _cat_order = {c: i for i, c in enumerate(time_dtype.categories)}

        def _sort_key(x: Any) -> Tuple[bool, Any]:
            return (x is None, _cat_order.get(x, len(_cat_order)))

    else:

        def _sort_key(x: Any) -> Tuple[bool, Any]:
            return (x is None, x)

    all_periods = sorted(t_pre_list + t_post_list, key=_sort_key)
    # Event-time mapping: natural rank of each period relative to F.
    F_idx = all_periods.index(F)
    period_to_event_time: Dict[Any, int] = {p: (i - F_idx) for i, p in enumerate(all_periods)}
    # Anchor = F-1 in natural rank (i.e., the last pre-period). By the
    # validator's contiguity guarantee, this IS t_pre_list[-1].
    if not t_pre_list:
        raise ValueError(
            "Internal error: event-study aggregation called with no "
            "pre-periods. _validate_had_panel_event_study should have "
            "rejected this upstream."
        )
    t_anchor = t_pre_list[-1]

    # Pivot to wide: units x periods.
    wide_y = df.pivot(index=unit_col, columns=time_col, values=outcome_col)
    wide_y = wide_y.sort_index()
    unit_ids = np.asarray(wide_y.index)

    # Dose at F (the single regressor for ALL horizons).
    d_at_F = (
        df[df[time_col] == F].set_index(unit_col).sort_index()[dose_col].to_numpy(dtype=np.float64)
    )

    dy_dict: Dict[int, np.ndarray] = {}
    anchor_y = wide_y[t_anchor].to_numpy(dtype=np.float64)
    for p in all_periods:
        if p == t_anchor:
            continue  # anchor gives ΔY = 0 trivially; skipped by contract
        e = period_to_event_time[p]
        # e = -1 corresponds to t_anchor, which we've already skipped.
        # All other periods get a horizon. The F-1 anchor / e=-1 coincidence
        # is preserved: event time -1 means "one period before F", which is
        # by definition the anchor.
        y_t = wide_y[p].to_numpy(dtype=np.float64)
        dy_dict[int(e)] = y_t - anchor_y

    cluster_arr: Optional[np.ndarray] = None
    if cluster_col is not None:
        if cluster_col not in data.columns:
            raise ValueError(f"cluster column {cluster_col!r} not found in data.")
        cluster_raw = data[cluster_col]
        if bool(cluster_raw.isna().any()):
            n_nan = int(cluster_raw.isna().sum())
            raise ValueError(
                f"cluster column {cluster_col!r} contains {n_nan} NaN "
                f"value(s) at the row level. Silent row dropping is "
                f"disabled; drop or impute cluster ids before calling fit()."
            )
        cluster_per_unit = df.groupby(unit_col)[cluster_col].nunique(dropna=False)
        if (cluster_per_unit > 1).any():
            n_bad = int((cluster_per_unit > 1).sum())
            raise ValueError(
                f"cluster column {cluster_col!r} is not constant within "
                f"unit for {n_bad} unit(s). Cluster must be unit-level."
            )
        cluster_arr = df.groupby(unit_col)[cluster_col].first().sort_index().to_numpy()

    return d_at_F, dy_dict, cluster_arr, unit_ids, t_anchor


# =============================================================================
# Design auto-detection
# =============================================================================


def _detect_design(d_arr: np.ndarray) -> str:
    """Resolve ``design="auto"`` to a concrete mode string.

    Implements the REGISTRY line-2320 rule:

    1. If ``d.min() == 0`` exactly, resolve unconditionally to
       ``continuous_at_zero``. (The modal-min check only runs when
       ``d.min() > 0`` since otherwise the "at d_lower" subset is always
       the "zero-dose" subset and the continuous-at-zero path is the
       paper-endorsed handling.)
    2. Otherwise, if ``d.min() < 0.01 * median(|d|)``, resolve to
       ``continuous_at_zero`` (small-share-of-treated samples that
       effectively satisfy Design 1').
    3. Else, if the modal fraction at ``d.min()`` exceeds 2%, resolve to
       ``mass_point``.
    4. Else, resolve to ``continuous_near_d_lower``.

    The rule is strict first-match: when both the ``<0.01 * median``
    threshold and the modal-fraction rule fire simultaneously (e.g., 3%
    of units at ``D = 0`` + 97% ``Uniform(0.5, 1)``), the resolution is
    ``continuous_at_zero``. This matches the paper-endorsed Design 1'
    handling of small-share-of-treated samples.

    Parameters
    ----------
    d_arr : np.ndarray, shape (G,)
        Unit-level post-period doses.

    Returns
    -------
    str
        One of ``"continuous_at_zero"``, ``"continuous_near_d_lower"``,
        ``"mass_point"``.
    """
    d_min = float(d_arr.min())

    # Tie-break: d.min() == 0 always resolves to Design 1'.
    # Use an absolute tolerance that scales with the range of the data.
    scale = max(1.0, float(np.max(np.abs(d_arr))))
    if abs(d_min) <= 1e-12 * scale:
        return "continuous_at_zero"

    median_abs = float(np.median(np.abs(d_arr)))
    # REGISTRY rule: d.min() < threshold * median(|d|) -> continuous_at_zero.
    # Guard against median_abs == 0 (all doses at zero; handled above).
    if median_abs > 0 and d_min < _CONTINUOUS_AT_ZERO_THRESHOLD * median_abs:
        return "continuous_at_zero"

    # Modal-fraction rule for mass-point detection.
    eps = 1e-12 * max(1.0, abs(d_min))
    at_d_min = np.abs(d_arr - d_min) <= eps
    modal_fraction = float(np.mean(at_d_min))
    if modal_fraction > _MASS_POINT_THRESHOLD:
        return "mass_point"

    return "continuous_near_d_lower"


# =============================================================================
# Mass-point 2SLS
# =============================================================================


def _fit_mass_point_2sls(
    d: np.ndarray,
    dy: np.ndarray,
    d_lower: float,
    cluster: Optional[np.ndarray],
    vcov_type: str,
) -> Tuple[float, float]:
    """Wald-IV point estimate and structural-residual 2SLS sandwich SE.

    The just-identified binary instrument ``Z_g = 1{D_{g,2} > d_lower}``
    gives a 2SLS estimator that collapses to the Wald-IV sample-average
    ratio
    ``beta_hat = (Ybar_{Z=1} - Ybar_{Z=0}) / (Dbar_{Z=1} - Dbar_{Z=0})``.

    The STANDARD ERROR is computed via the 2SLS sandwich
    ``V = [Z'X]^{-1} * Omega * [Z'X]^{-T}`` where ``Omega`` is built from
    the STRUCTURAL residuals
    ``u = dy - alpha_hat - beta_hat * d``
    (NOT the reduced-form residuals). This is the canonical 2SLS
    inference path and matches what ``AER::ivreg`` / ``ivreg2`` would
    produce. The "OLS-on-indicator + scale by dose-gap" shortcut gives
    reduced-form residuals, which diverge from structural residuals in
    finite samples and substantively under clustering.

    Supported ``vcov_type``:

    - ``"classical"``: constant variance ``sigma_hat^2 = sum(u^2) / (n-2)``.
    - ``"hc1"``: heteroskedasticity-robust with small-sample DOF scaling
      ``n / (n - 2)``. With ``cluster`` supplied, switches to CR1
      (Liang-Zeger) cluster-robust.

    ``"hc2"`` and ``"hc2_bm"`` raise ``NotImplementedError`` (2SLS-specific
    leverage derivation pending; queued for follow-up).

    Parameters
    ----------
    d : np.ndarray, shape (n,)
        Post-period doses ``D_{g,2}``.
    dy : np.ndarray, shape (n,)
        First-difference outcome ``Delta Y_g``.
    d_lower : float
        Support infimum / evaluation point.
    cluster : np.ndarray or None, shape (n,)
        Cluster ids per unit (``None`` for no clustering).
    vcov_type : str
        One of ``"classical"``, ``"hc1"``.

    Returns
    -------
    tuple[float, float]
        ``(beta_hat, se_beta)``. NaN for SE when the dose-gap vanishes
        (``Dbar_{Z=1} == Dbar_{Z=0}``) or the sandwich is singular.
    """
    d = np.asarray(d, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    n = d.shape[0]

    Z = (d > d_lower).astype(np.float64)
    n_above = int(Z.sum())
    n_at_or_below = n - n_above

    # Degeneracy checks: if either Z-subset is empty, Wald-IV is undefined.
    if n_above == 0 or n_at_or_below == 0:
        return float("nan"), float("nan")

    dose_gap = d[Z == 1].mean() - d[Z == 0].mean()
    if abs(dose_gap) < 1e-12 * max(1.0, abs(float(d.mean()))):
        # No dose variation around d_lower -> beta undefined.
        return float("nan"), float("nan")

    dy_gap = dy[Z == 1].mean() - dy[Z == 0].mean()
    beta_hat = float(dy_gap / dose_gap)
    alpha_hat = float(dy.mean() - beta_hat * d.mean())

    # STRUCTURAL residuals (plan-review CRITICAL #1): u = y - alpha - beta*x.
    # The Wald-IV/OLS-on-indicator shortcut would use reduced-form residuals
    # u_rf = dy - (alpha_rf + gamma * Z), which differ in finite samples.
    u = dy - alpha_hat - beta_hat * d

    # Design matrices: X = [1, d] (endogenous), Z_d = [1, Z] (instrument).
    X = np.column_stack([np.ones(n, dtype=np.float64), d])
    Zd = np.column_stack([np.ones(n, dtype=np.float64), Z])

    ZtX = Zd.T @ X  # (2, 2)
    try:
        ZtX_inv = np.linalg.inv(ZtX)
    except np.linalg.LinAlgError:
        # Z'X singular (e.g., no variation in Z, already handled above).
        return beta_hat, float("nan")

    vcov_type = vcov_type.lower()
    if vcov_type in _MASS_POINT_VCOV_UNSUPPORTED:
        raise NotImplementedError(
            f"vcov_type={vcov_type!r} is not supported on the "
            f"HeterogeneousAdoptionDiD mass-point path in Phase 2a. "
            f"HC2 / HC2-BM require a 2SLS-specific leverage derivation "
            f"`x_i' (Z'X)^{{-1}}(...)(X'Z)^{{-1}} x_i` that differs from "
            f"the OLS leverage `x_i' (X'X)^{{-1}} x_i`. Derivation + R "
            f"parity anchor are queued for the follow-up PR. Use "
            f"vcov_type='hc1' or 'classical' for now."
        )

    if cluster is not None:
        # CR1 (Liang-Zeger) cluster-robust sandwich.
        # Use pd.unique to match R's first-appearance order (stable for
        # cross-runtime reproducibility).
        clusters_unique = pd.unique(cluster)
        Omega = np.zeros((2, 2), dtype=np.float64)
        for c in clusters_unique:
            idx = cluster == c
            # score per cluster: s_c = Zd[idx]' @ u[idx]
            s = Zd[idx].T @ u[idx]
            Omega += np.outer(s, s)
        G = len(clusters_unique)
        k = 2
        if G < 2:
            # Cluster-robust SE undefined with a single cluster.
            return beta_hat, float("nan")
        Omega *= (G / (G - 1)) * ((n - 1) / (n - k))
    elif vcov_type == "classical":
        dof = n - 2
        if dof <= 0:
            return beta_hat, float("nan")
        sigma2 = float((u * u).sum()) / dof
        Omega = sigma2 * (Zd.T @ Zd)
    elif vcov_type == "hc1":
        dof = n - 2
        if dof <= 0:
            return beta_hat, float("nan")
        Omega = (n / dof) * (Zd.T @ ((u * u)[:, None] * Zd))
    else:
        raise ValueError(
            f"Unsupported vcov_type={vcov_type!r} on the HAD mass-point "
            f"path. Supported: {_MASS_POINT_VCOV_SUPPORTED} (plus "
            f"cluster-robust CR1 via cluster=)."
        )

    V = ZtX_inv @ Omega @ ZtX_inv.T
    var_beta = float(V[1, 1])
    if not np.isfinite(var_beta) or var_beta < 0:
        return beta_hat, float("nan")
    se_beta = float(np.sqrt(var_beta))
    return beta_hat, se_beta


# =============================================================================
# Main estimator class
# =============================================================================


class HeterogeneousAdoptionDiD:
    """Heterogeneous Adoption Difference-in-Differences estimator.

    Implements de Chaisemartin, Ciccia, D'Haultfoeuille, and Knau (2026)
    Weighted-Average-Slope (WAS) estimator with three design-dispatch
    paths: Design 1' (continuous-at-zero), Design 1 continuous-near-
    d_lower, and Design 1 mass-point (2SLS sample-average per paper
    Section 3.2.4). Two aggregation modes:

    - ``aggregate="overall"`` (Phase 2a, default) returns a single-period
      :class:`HeterogeneousAdoptionDiDResults` on a two-period panel.
    - ``aggregate="event_study"`` (Phase 2b, paper Appendix B.2) returns
      a :class:`HeterogeneousAdoptionDiDEventStudyResults` with per-
      event-time WAS estimates on a multi-period panel, using a uniform
      ``F-1`` anchor and pointwise CIs per horizon. Staggered-timing
      panels auto-filter to the last-treatment cohort plus never-treated
      units (paper Appendix B.2 prescription).

    Parameters
    ----------
    design : {"auto", "continuous_at_zero", "continuous_near_d_lower", "mass_point"}
        Design-dispatch strategy. Defaults to ``"auto"`` which resolves
        via the REGISTRY auto-detect rule on the fitted dose data
        (see :func:`_detect_design`).

        Explicit overrides are checked against the paper's
        regime-partition contract (Section 3.2) at fit time:

        - ``"continuous_at_zero"`` (Design 1'): paper requires the
          support infimum ``d_lower = 0``. Phase 1c's
          ``_validate_had_inputs`` rejects mass-point samples passed
          to this path.
        - ``"continuous_near_d_lower"`` (Design 1, continuous density
          near ``d_lower``): requires ``d_lower > 0`` and a
          non-mass-point sample (modal fraction at ``d.min()`` must be
          <= 2%). ``d_lower`` must equal ``float(d.min())`` within
          float tolerance; non-support-infimum thresholds are off-
          support and raise.
        - ``"mass_point"`` (Design 1 mass-point): requires
          ``d_lower > 0`` AND a mass-point sample (modal fraction at
          ``d.min()`` must be > 2%). ``d_lower`` must equal
          ``float(d.min())`` within float tolerance. Forcing this
          design on a ``d_lower = 0`` sample or on a continuous
          (non-mass-point) sample raises; in either case 2SLS
          identifies a different estimand than the paper's Design 1
          mass-point WAS.

        Mismatched overrides raise ``ValueError`` pointing at the
        correct design rather than silently identifying a different
        estimand.
    d_lower : float or None
        Support infimum ``d_lower``. ``None`` means use ``0.0`` on the
        Design 1' path and ``float(d.min())`` on the other two paths.
        On Design 1 paths (``continuous_near_d_lower`` and
        ``mass_point``), an explicit ``d_lower`` must equal
        ``float(d.min())`` within float tolerance AND must be strictly
        positive; zero-valued or mismatched thresholds raise.
    kernel : {"epanechnikov", "triangular", "uniform"}
        Forwarded to :func:`bias_corrected_local_linear` on the continuous
        paths. Ignored on the mass-point path.
    alpha : float
        CI level (0.05 for 95% CI).
    vcov_type : {"classical", "hc1"} or None
        Mass-point-path only. When ``None``, the effective family falls
        back to the ``robust`` flag: ``robust=True`` -> ``"hc1"``,
        ``robust=False`` -> ``"classical"`` (the default construction).
        Explicit ``"hc2"`` and ``"hc2_bm"`` raise ``NotImplementedError``
        pending a 2SLS-specific leverage derivation. Ignored on the
        continuous paths (which use the CCT-2014 robust SE from
        Phase 1c); passing a non-default ``vcov_type`` on a continuous
        path emits a ``UserWarning`` per fit call.
    robust : bool
        Backward-compat alias used only when ``vcov_type is None``:
        ``True`` -> ``"hc1"``, ``False`` -> ``"classical"``. Explicit
        ``vcov_type`` takes precedence (e.g.,
        ``vcov_type="classical", robust=True`` runs classical). Only
        the mass-point path consumes these; continuous paths ignore
        both with a warning.
    cluster : str or None
        Column name for cluster-robust SE on the mass-point path (CR1).
        Ignored with a ``UserWarning`` on the continuous paths in Phase
        2a (nonparametric cluster support exists on Phase 1c but is
        exposed separately via ``bias_corrected_local_linear``; the
        estimator-level knob is queued for a follow-up PR).

    Notes
    -----
    **Diagnostics coverage.** ``HeterogeneousAdoptionDiDResults.bandwidth_diagnostics``
    and ``.bias_corrected_fit`` are populated only on the continuous
    paths; both are ``None`` on the mass-point path (which is parametric
    and has no bandwidth). Conversely, ``.n_mass_point`` and
    ``.n_above_d_lower`` are populated only on the mass-point path.

    **Clone idempotence.** ``self.design`` stores the RAW user input
    (e.g., ``"auto"``); the resolved mode is stored on the result object
    at fit time. This mirrors Phase 1a's ``_vcov_type_arg`` pattern and
    keeps ``get_params()`` / ``sklearn.clone()`` round-trips exact.

    Examples
    --------
    Construct a two-period HAD panel by hand. Phase 2a requires exactly
    two periods with ``D_{g,1} = 0`` for every unit.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from diff_diff import HeterogeneousAdoptionDiD
    >>> rng = np.random.default_rng(42)  # doctest: +SKIP
    >>> G = 500  # doctest: +SKIP
    >>> dose_post = rng.uniform(0.0, 1.0, G)  # doctest: +SKIP
    >>> dose_post[0] = 0.0  # at least one zero-dose unit for Design 1'
    >>> delta_y = 0.3 * dose_post + 0.1 * rng.standard_normal(G)  # doctest: +SKIP
    >>> data = pd.DataFrame({  # doctest: +SKIP
    ...     "unit": np.repeat(np.arange(G), 2),
    ...     "period": np.tile([1, 2], G),
    ...     "dose": np.column_stack([np.zeros(G), dose_post]).ravel(),
    ...     "outcome": np.column_stack([np.zeros(G), delta_y]).ravel(),
    ... })
    >>> est = HeterogeneousAdoptionDiD(design="auto")  # doctest: +SKIP
    >>> result = est.fit(  # doctest: +SKIP
    ...     data, outcome_col="outcome", dose_col="dose",
    ...     time_col="period", unit_col="unit",
    ... )
    >>> result.design  # doctest: +SKIP
    'continuous_at_zero'
    """

    def __init__(
        self,
        design: str = "auto",
        d_lower: Optional[float] = None,
        kernel: str = "epanechnikov",
        alpha: float = 0.05,
        vcov_type: Optional[str] = None,
        robust: bool = False,
        cluster: Optional[str] = None,
    ) -> None:
        self.design = design
        self.d_lower = d_lower
        self.kernel = kernel
        self.alpha = alpha
        self.vcov_type = vcov_type
        self.robust = robust
        self.cluster = cluster
        self._validate_constructor_args()

    def _validate_constructor_args(self) -> None:
        if self.design not in _VALID_DESIGNS:
            raise ValueError(
                f"Invalid design={self.design!r}. Must be one of " f"{_VALID_DESIGNS}."
            )
        if not (0.0 < float(self.alpha) < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha!r}.")
        # d_lower must be None or a finite scalar. NaN / +/-inf would
        # bypass every downstream comparison-based guard (`>`, `<=`,
        # tolerance-abs-diff) because those return False for NaN, so we
        # front-door reject non-finite values here. This fires under
        # __init__ and set_params (which dry-runs through the
        # constructor), keeping the contract uniform across all entry
        # points.
        if self.d_lower is not None:
            d_lower_float = float(self.d_lower)
            if not np.isfinite(d_lower_float):
                raise ValueError(
                    f"d_lower must be None or a finite scalar; got "
                    f"d_lower={self.d_lower!r}. NaN and +/-inf are not "
                    f"valid support-infimum values."
                )
        if self.vcov_type is not None:
            if self.vcov_type.lower() in _MASS_POINT_VCOV_UNSUPPORTED:
                # Don't raise here — the raise happens at fit() time on
                # the mass-point path so users who set vcov_type=hc2 for
                # a continuous fit (which ignores the knob) don't get
                # blocked. But we also don't silently accept it at
                # construction if it's outside our documented enum.
                pass
            elif (
                self.vcov_type.lower()
                not in _MASS_POINT_VCOV_SUPPORTED + _MASS_POINT_VCOV_UNSUPPORTED
            ):
                raise ValueError(
                    f"Invalid vcov_type={self.vcov_type!r}. Must be one "
                    f"of {_MASS_POINT_VCOV_SUPPORTED + _MASS_POINT_VCOV_UNSUPPORTED}, "
                    f"or None."
                )

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return the raw constructor parameters (sklearn-compatible).

        Matches the :meth:`sklearn.base.BaseEstimator.get_params`
        signature. Preserves the user's original inputs - in particular,
        ``design`` returns ``"auto"`` when the user set it to ``"auto"``
        (even after fit), so ``sklearn.base.clone(est)`` round-trips
        exactly.

        Parameters
        ----------
        deep : bool, default=True
            Accepted for sklearn-contract compatibility. This estimator
            has no nested sub-estimator parameters, so ``deep=False``
            and ``deep=True`` return the same dict.
        """
        del deep  # accepted for compat; this estimator has no nested params
        return {
            "design": self.design,
            "d_lower": self.d_lower,
            "kernel": self.kernel,
            "alpha": self.alpha,
            "vcov_type": self.vcov_type,
            "robust": self.robust,
            "cluster": self.cluster,
        }

    def set_params(self, **params: Any) -> "HeterogeneousAdoptionDiD":
        """Set estimator parameters and return self (sklearn-compatible).

        Only keys returned by :meth:`get_params` are accepted. Passing
        any other attribute name (including method names like ``fit``)
        raises ``ValueError`` so the estimator cannot be silently
        corrupted by a mistyped or attacker-supplied key.

        Mutation is ATOMIC: validation runs on a proposed merged
        parameter dict before any attribute is overwritten. A failing
        call (invalid key, or an otherwise valid key whose value
        violates the constructor constraints) leaves ``self`` unchanged
        and safe to reuse.
        """
        valid_keys = set(self.get_params().keys())
        invalid = [k for k in params if k not in valid_keys]
        if invalid:
            raise ValueError(
                f"Invalid parameter: {invalid[0]!r}. Valid parameters: " f"{sorted(valid_keys)}."
            )
        # Dry-run validation by constructing a fresh instance with the
        # merged state. If the constructor raises, self is not mutated.
        merged = self.get_params()
        merged.update(params)
        type(self)(**merged)  # raises ValueError on invalid combination
        # All checks passed; apply atomically.
        for key, value in params.items():
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # Main fit entry point
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        dose_col: str,
        time_col: str,
        unit_col: str,
        first_treat_col: Optional[str] = None,
        aggregate: str = "overall",
        survey: Any = None,
        weights: Optional[np.ndarray] = None,
    ) -> HeterogeneousAdoptionDiDResults:
        """Fit the HAD estimator.

        ``aggregate="overall"`` (default) fits on a two-period panel and
        returns a :class:`HeterogeneousAdoptionDiDResults` with the
        single-period WAS estimate. ``aggregate="event_study"`` fits on
        a multi-period panel (``T > 2``) and returns a
        :class:`HeterogeneousAdoptionDiDEventStudyResults` with per-
        event-time WAS estimates using a uniform ``F-1`` anchor (paper
        Appendix B.2).

        Both the overall and event-study paths are **panel-only**: the paper
        (Section 2) defines HAD on panel or repeated-cross-section data,
        but this implementation requires a balanced panel with a unit
        identifier so that unit-level first differences
        ``ΔY_{g,t} = Y_{g,t} - Y_{g,t_anchor}`` can be formed.
        Repeated-cross-section inputs (disjoint unit IDs between
        periods) are rejected by the balanced-panel validator.
        Repeated-cross-section support is queued for a follow-up PR
        (tracked in ``TODO.md``); it requires a separate identification
        path based on pre/post cell means rather than unit-level
        differences.

        Parameters
        ----------
        data : pd.DataFrame
        outcome_col, dose_col, time_col, unit_col : str
            Column names.
        first_treat_col : str or None
            Optional first-treatment column (the period at which each
            unit first receives treatment; ``0`` for never-treated).
            Required on the event-study path when the panel has more
            than two distinct first-treat values (staggered timing):
            the estimator auto-filters to the last-treatment cohort
            with a ``UserWarning`` per paper Appendix B.2 prescription.
            For common-adoption panels the column is optional; when
            omitted, the event-study path infers the first-treatment
            period ``F`` from the dose invariant.
        aggregate : {"overall", "event_study"}
            ``"overall"`` (default): returns a single-period
            :class:`HeterogeneousAdoptionDiDResults` (Phase 2a). Requires
            exactly two time periods.
            ``"event_study"`` (Phase 2b): returns a
            :class:`HeterogeneousAdoptionDiDEventStudyResults` with per-
            event-time WAS estimates on the multi-period panel (paper
            Appendix B.2). Requires more than two time periods. Pointwise
            CIs per horizon; joint cross-horizon covariance is deferred
            to a follow-up PR. Staggered-timing panels are auto-filtered
            to the last-treatment cohort with a ``UserWarning``.
        survey : SurveyDesign or None
            Reserved for a follow-up survey-integration PR. Must be
            ``None`` in Phase 2a.
        weights : np.ndarray or None
            Reserved for a follow-up PR. Must be ``None`` in Phase 2a.

        Returns
        -------
        HeterogeneousAdoptionDiDResults
        """
        # ---- aggregate / survey / weights validation ----
        if aggregate not in _VALID_AGGREGATES:
            raise ValueError(
                f"Invalid aggregate={aggregate!r}. Must be one of " f"{_VALID_AGGREGATES}."
            )
        if survey is not None and weights is not None:
            raise ValueError(
                "Pass survey=<SurveyDesign> OR weights=<array>, not both. "
                "For SurveyDesign-composed inference (PSU, strata, FPC, "
                "replicate weights), use survey=. For a simple pweight-only "
                "shortcut, use weights=; it is internally equivalent to "
                "survey=SurveyDesign(weights=w)."
            )
        # Event-study + survey/weights: Phase 4.5 B deferral.
        if aggregate == "event_study" and (survey is not None or weights is not None):
            raise NotImplementedError(
                "survey= / weights= are not yet supported on "
                "aggregate='event_study' (deferred to Phase 4.5 B — "
                "event-study survey composition). The continuous-design "
                "overall path supports survey= and weights= as of this PR."
            )
        # Dispatch the event-study path to a dedicated method so the
        # single-period path stays unchanged (Phase 2a contract preserved).
        # Note: event_study returns HeterogeneousAdoptionDiDEventStudyResults
        # (distinct type from the overall path's HeterogeneousAdoptionDiDResults);
        # the static return-type annotation reflects the common "overall" case
        # to keep Phase 2a call-sites type-clean. Users explicitly passing
        # aggregate="event_study" should annotate the result as
        # HeterogeneousAdoptionDiDEventStudyResults.
        if aggregate == "event_study":
            return self._fit_event_study(  # type: ignore[return-value]
                data=data,
                outcome_col=outcome_col,
                dose_col=dose_col,
                time_col=time_col,
                unit_col=unit_col,
                first_treat_col=first_treat_col,
            )

        # ---- Resolve effective fit-time state (local vars only, per
        # feedback_fit_does_not_mutate_config; do not mutate self.*) ----
        design_arg = self.design
        d_lower_arg = self.d_lower
        vcov_type_arg = self.vcov_type
        robust_arg = self.robust
        cluster_arg = self.cluster

        # ---- Validate panel contract ----
        t_pre, t_post = _validate_had_panel(
            data, outcome_col, dose_col, time_col, unit_col, first_treat_col
        )

        # ---- Aggregate to unit-level first differences (no cluster yet) ----
        # Defer cluster validation/extraction until after the design is
        # resolved: the continuous paths ignore cluster= with a warning,
        # so a malformed or irrelevant cluster column must not abort a
        # valid continuous fit. Cluster extraction is re-run below only
        # when resolved_design == "mass_point".
        d_arr, dy_arr, _, _ = _aggregate_first_difference(
            data,
            outcome_col,
            dose_col,
            time_col,
            unit_col,
            t_pre,
            t_post,
            None,
        )

        # Resolve survey/weights into a per-unit weight array.
        # - `survey=SurveyDesign(weights=w)` → extract w, aggregate per-unit.
        # - `weights=w` → shorthand for SurveyDesign(weights=w).
        # - neither → unweighted.
        # The per-unit array is aligned with d_arr/dy_arr (sorted unit ids).
        # Non-trivial SurveyDesign fields (PSU/strata/FPC/replicate) are
        # routed through ``compute_survey_if_variance`` at the variance
        # step in _fit_continuous; this PR ships the ``weights=`` path;
        # the full PSU-composed variance extension is Phase 4.5 commit 3.
        weights_unit: Optional[np.ndarray] = None
        if weights is not None:
            weights_unit = _aggregate_unit_weights(data, weights, unit_col)
        elif survey is not None:
            # Minimal survey support: extract per-row weights from SurveyDesign,
            # aggregate to unit level, and apply via the weighted lprobust.
            # Full PSU/strata/FPC composition is queued for the next commit.
            if not hasattr(survey, "weights"):
                raise TypeError(
                    f"survey= must be a SurveyDesign-like object with a "
                    f".weights attribute; got {type(survey).__name__}. "
                    f"Construct a SurveyDesign via diff_diff.survey."
                )
            sd_weights = getattr(survey, "weights", None)
            if sd_weights is None:
                raise NotImplementedError(
                    "survey= without weights is not yet supported. Pass "
                    "survey=SurveyDesign(weights=...) with per-row weights; "
                    "full SurveyDesign integration (PSU, strata, FPC, "
                    "replicate weights) ships in the subsequent commit."
                )
            weights_unit = _aggregate_unit_weights(
                data, np.asarray(sd_weights, dtype=np.float64), unit_col
            )

        n_obs = int(d_arr.shape[0])
        if n_obs < 3:
            raise ValueError(
                f"HAD requires at least 3 units for inference; got "
                f"n_obs={n_obs} after aggregation."
            )

        # ---- Resolve design ----
        if design_arg == "auto":
            resolved_design = _detect_design(d_arr)
        else:
            resolved_design = design_arg

        # ---- Extract cluster IDs (mass-point path only) ----
        # Continuous paths ignore cluster= with a warning emitted later in
        # the dispatch block; the cluster column is not read for them. On
        # the mass-point path we now re-run the aggregation with
        # cluster_col so validation (missing column / NaN / within-unit
        # variance) fires only when cluster is actually going to be used.
        cluster_arr: Optional[np.ndarray] = None
        if resolved_design == "mass_point" and cluster_arg is not None:
            _, _, cluster_arr, _ = _aggregate_first_difference(
                data,
                outcome_col,
                dose_col,
                time_col,
                unit_col,
                t_pre,
                t_post,
                cluster_arg,
            )

        # ---- Resolve d_lower ----
        if resolved_design == "continuous_at_zero":
            # Design 1' regime (paper Section 3.2) is defined at d_lower = 0.
            # Reject explicit nonzero d_lower overrides front-door rather
            # than silently coerce to zero. Tolerance family matches the
            # Design 1 d_lower guards below.
            if d_lower_arg is not None:
                scale = max(1.0, float(np.max(np.abs(d_arr))))
                if abs(float(d_lower_arg)) > 1e-12 * scale:
                    raise ValueError(
                        f"design='continuous_at_zero' (Design 1') requires "
                        f"d_lower == 0 within float tolerance (paper Section "
                        f"3.2 Design 1' regime). Got d_lower="
                        f"{float(d_lower_arg)!r}. For d_lower > 0 use "
                        f"design='continuous_near_d_lower' (local-linear "
                        f"boundary-limit estimator) or design='mass_point' "
                        f"(2SLS) as appropriate, or design='auto' which "
                        f"auto-detects the correct path from the dose "
                        f"distribution."
                    )
            d_lower_val = 0.0
        elif d_lower_arg is None:
            d_lower_val = float(d_arr.min())
        else:
            d_lower_val = float(d_lower_arg)

        # ---- Regime partition: d_lower > 0 for Design 1 paths ----
        # Paper Section 3.2 partitions HAD into d_lower = 0 (Design 1',
        # continuous_at_zero) and d_lower > 0 (Design 1, continuous_near
        # _d_lower or mass_point). The auto-detect rule already enforces
        # this partition; explicit overrides must respect it too, otherwise
        # `design="mass_point", d_lower=0` returns a finite but
        # paper-incompatible 2SLS result and `design="continuous_near_d_lower"`
        # with d_lower=0 reduces to Design 1' algebra while mislabeling the
        # estimand as `WAS_d_lower` and emitting the wrong Assumption 5/6
        # warning. Use the same float-tolerance family as _detect_design's
        # d.min()==0 tie-break.
        if resolved_design in ("mass_point", "continuous_near_d_lower"):
            scale = max(1.0, float(np.max(np.abs(d_arr))))
            if abs(d_lower_val) <= 1e-12 * scale:
                raise ValueError(
                    f"design={resolved_design!r} requires d_lower > 0 (paper "
                    f"Section 3.2 reserves the d_lower=0 regime for Design 1' "
                    f"/ `continuous_at_zero`). Got d_lower={d_lower_val!r}. "
                    f"Use design='continuous_at_zero' (explicit) or "
                    f"design='auto' (auto-detect) for samples with support "
                    f"infimum at zero."
                )

        # ---- Original-scale modal-fraction / regime check ----
        # Paper Section 3.2 splits Design 1 into two REGIME-SPECIFIC
        # estimators by modal-min fraction:
        #   - continuous_near_d_lower: modal fraction at d.min() <= 2%
        #     (local-linear boundary-limit estimator).
        #   - mass_point: modal fraction at d.min() > 2%
        #     (Wald-IV / 2SLS identified by the instrument 1{D_2 > d_lower}).
        # The auto-detect rule already enforces this; explicit overrides
        # must too, otherwise the wrong estimand is returned silently.
        # Both guards are symmetric around the 2% threshold used in
        # _detect_design() and Phase 1c's _validate_had_inputs().
        if resolved_design in ("continuous_near_d_lower", "mass_point"):
            d_min_orig = float(d_arr.min())
            if d_min_orig > 0:
                eps_mp = 1e-12 * max(1.0, abs(d_min_orig))
                at_d_min_mask_orig = np.abs(d_arr - d_min_orig) <= eps_mp
                modal_fraction_orig = float(np.mean(at_d_min_mask_orig))
                if (
                    resolved_design == "continuous_near_d_lower"
                    and modal_fraction_orig > _MASS_POINT_THRESHOLD
                ):
                    raise ValueError(
                        f"design='continuous_near_d_lower' cannot be used on a "
                        f"mass-point sample (modal fraction {modal_fraction_orig:.4f} "
                        f"at d.min()={d_min_orig!r} exceeds the "
                        f"{_MASS_POINT_THRESHOLD:.2f} threshold from paper Section "
                        f"3.2.4). Use design='mass_point' (Wald-IV / 2SLS) or "
                        f"design='auto' which will auto-detect. Forcing the "
                        f"continuous path on a mass-point sample would produce "
                        f"the wrong estimand."
                    )
                if resolved_design == "mass_point" and modal_fraction_orig <= _MASS_POINT_THRESHOLD:
                    raise ValueError(
                        f"design='mass_point' requires a modal mass at d.min() "
                        f"exceeding the {_MASS_POINT_THRESHOLD:.2f} threshold "
                        f"(paper Section 3.2.4). Got modal fraction "
                        f"{modal_fraction_orig:.4f} at d.min()={d_min_orig!r}. "
                        f"For continuous-near-d_lower samples use "
                        f"design='continuous_near_d_lower' (local-linear "
                        f"boundary-limit estimator) or design='auto' which "
                        f"will auto-detect. Forcing 2SLS on a continuous "
                        f"sample identifies the exact-d.min() cell rather "
                        f"than the paper's boundary-limit estimand."
                    )

        # ---- d_lower contract for Design 1 paths ----
        # Paper Sections 3.2.2-3.2.4 define the Design 1 estimators at
        # d_lower = support infimum of D_{g,2}. For the mass-point path,
        # the instrument Z = 1{D_{g,2} > d_lower} requires d_lower to be
        # the lower-support mass point. For the continuous-near-d_lower
        # path, evaluating the local-linear fit after the regressor
        # shift (D - d_lower) at boundary=0 only makes sense when the
        # shift anchors to the realized sample minimum; otherwise the
        # boundary evaluation is off-support (no observations near zero
        # on the shifted scale) and Phase 1c's 5% plausibility heuristic
        # may fail to catch the mismatch. We enforce d_lower == d.min()
        # within float tolerance on both Design 1 paths; mismatched
        # overrides raise with a clear pointer to the unsupported
        # estimand.
        if resolved_design in ("mass_point", "continuous_near_d_lower") and d_lower_arg is not None:
            d_min = float(d_arr.min())
            tol = 1e-12 * max(1.0, abs(d_min))
            if abs(d_lower_val - d_min) > tol:
                raise ValueError(
                    f"design={resolved_design!r} requires d_lower to equal "
                    f"the support infimum float(d.min())={d_min!r}; got "
                    f"d_lower={d_lower_val!r}. The paper's Design 1 "
                    f"estimators (Sections 3.2.2-3.2.4) identify at the "
                    f"lower-support boundary, not at an arbitrary "
                    f"threshold. Pass d_lower=None to auto-resolve, or "
                    f"d_lower=float(d.min()) explicitly. Non-support-"
                    f"infimum thresholds identify a different (LATE-like "
                    f"for mass_point, off-support for continuous_near_"
                    f"d_lower) estimand that is out of Phase 2a scope."
                )
            # Snap tolerance-accepted overrides back to the exact support
            # infimum. Float-rounding drift matters downstream: on the
            # mass-point path, `Z = d > d_lower` with d_lower = d.min() - ε
            # puts the mass-point units into Z=1 (control group empties);
            # on the continuous-near-d_lower path, d_lower = d.min() + ε
            # makes `d - d_lower` negative and trips Phase 1c's
            # _validate_had_inputs negative-dose guard. Snapping preserves
            # the "within tolerance" contract while keeping downstream
            # algebra exact.
            d_lower_val = d_min

        # ---- Compute cohort counts ----
        if resolved_design == "mass_point":
            eps = 1e-12 * max(1.0, abs(d_lower_val))
            at_d_min_mask = np.abs(d_arr - d_lower_val) <= eps
            above_mask = d_arr > d_lower_val
            n_mass_point: Optional[int] = int(at_d_min_mask.sum())
            n_above_d_lower: Optional[int] = int(above_mask.sum())
            n_treated = n_above_d_lower
            n_control = int(at_d_min_mask.sum())
        else:
            n_mass_point = None
            n_above_d_lower = None
            above_mask = d_arr > d_lower_val
            n_treated = int(above_mask.sum())
            n_control = n_obs - n_treated

        dose_mean = float(d_arr.mean())

        # ---- Assumption 5/6 warning on Design 1 paths ----
        # Paper Sections 3.2.2-3.2.4: when d_lower > 0 (Design 1 family),
        # point identification of WAS_{d_lower} requires Assumption 6 in
        # addition to parallel trends (Assumption 1-3); Assumption 5 gives
        # only sign identification. These extra assumptions are NOT
        # testable via pre-trends. Surface this to the user front-door so
        # results are not silently interpreted as full point identification.
        if resolved_design in ("continuous_near_d_lower", "mass_point"):
            warnings.warn(
                f"design={resolved_design!r} (Design 1, d_lower > 0) requires "
                f"Assumption 6 from de Chaisemartin et al. (2026) for point "
                f"identification of WAS_{{d_lower}}, or Assumption 5 for "
                f"sign identification only. Neither is testable via "
                f"pre-trends. Confirm the extra assumption is defensible "
                f"for your setting before interpreting the returned "
                f"point estimate as the WAS.",
                UserWarning,
                stacklevel=2,
            )

        # ---- Dispatch ----
        if resolved_design in ("continuous_at_zero", "continuous_near_d_lower"):
            # Warn when the user set a mass-point-only knob that's ignored
            # on the continuous path. (Emitted per fit call; this is not
            # suppressed after the first call.)
            if vcov_type_arg is not None:
                warnings.warn(
                    f"vcov_type={vcov_type_arg!r} is ignored on the "
                    f"'{resolved_design}' path (the continuous designs "
                    f"use the CCT-2014 robust SE from Phase 1c). "
                    f"vcov_type applies only to design='mass_point'.",
                    UserWarning,
                    stacklevel=2,
                )
            if robust_arg:
                warnings.warn(
                    f"robust=True is ignored on the '{resolved_design}' "
                    f"path (the continuous designs use the CCT-2014 "
                    f"robust SE from Phase 1c unconditionally; the "
                    f"robust flag is a mass-point-only backward-compat "
                    f"alias for vcov_type).",
                    UserWarning,
                    stacklevel=2,
                )
            if cluster_arg is not None:
                warnings.warn(
                    f"cluster={cluster_arg!r} is ignored on the "
                    f"'{resolved_design}' path in Phase 2a. Cluster-"
                    f"robust SE on the nonparametric path is exposed "
                    f"via diff_diff.bias_corrected_local_linear directly "
                    f"but not yet threaded through the estimator-level "
                    f"knob.",
                    UserWarning,
                    stacklevel=2,
                )
            att, se, bc_fit, bw_diag = self._fit_continuous(
                d_arr,
                dy_arr,
                resolved_design,
                d_lower_val,
                weights_arr=weights_unit,
            )
            inference_method = "analytical_nonparametric"
            vcov_label: Optional[str] = None
            cluster_label: Optional[str] = None
        elif resolved_design == "mass_point":
            # Phase 4.5 B deferral: weighted 2SLS + survey composition on
            # the mass-point path is not yet wired. Reject explicitly
            # rather than silently ignoring survey/weights.
            if weights_unit is not None:
                raise NotImplementedError(
                    "survey= / weights= on design='mass_point' is deferred "
                    "to Phase 4.5 B (weighted 2SLS + sandwich variance). "
                    "This PR ships survey support only on the continuous-"
                    "dose paths (continuous_at_zero, continuous_near_d_lower)."
                )
            if vcov_type_arg is None:
                # Backward-compat: robust=True -> hc1, robust=False -> classical.
                vcov_requested = "hc1" if robust_arg else "classical"
            else:
                vcov_requested = vcov_type_arg.lower()
            att, se = _fit_mass_point_2sls(
                d_arr,
                dy_arr,
                d_lower_val,
                cluster_arr,
                vcov_requested,
            )
            bc_fit = None
            bw_diag = None
            inference_method = "analytical_2sls"
            # Store the EFFECTIVE variance family so downstream consumers
            # (to_dict, to_dataframe, summary) see the actual SE type that
            # was computed. When cluster is supplied, _fit_mass_point_2sls
            # unconditionally computes CR1 regardless of vcov_requested
            # (e.g. classical+cluster -> CR1), so we surface that here.
            vcov_label = "cr1" if cluster_arg is not None else vcov_requested
            cluster_label = cluster_arg if cluster_arg is not None else None
        else:
            raise ValueError(f"Internal error: unhandled design={resolved_design!r}.")

        # ---- Route all inference fields through safe_inference ----
        t_stat, p_value, conf_int = safe_inference(att, se, alpha=float(self.alpha))

        # Build survey metadata when weights/survey were supplied. Phase
        # 4.5 commit 2 ships weight composition only; PSU/strata/FPC +
        # Binder TSL composition lands in the subsequent commit.
        survey_metadata: Optional[Dict[str, Any]] = None
        if weights_unit is not None:
            w_sum = float(weights_unit.sum())
            w_sq_sum = float(np.dot(weights_unit, weights_unit))
            ess = (w_sum * w_sum / w_sq_sum) if w_sq_sum > 0 else float("nan")
            survey_metadata = {
                "method": "pweight" if survey is None else "survey_pweight",
                "source": "weights_arr" if survey is None else "SurveyDesign.weights",
                "n_units_weighted": int(weights_unit.shape[0]),
                "weight_sum": w_sum,
                "effective_sample_size": float(ess),
            }

        return HeterogeneousAdoptionDiDResults(
            att=float(att),
            se=float(se),
            t_stat=float(t_stat),
            p_value=float(p_value),
            conf_int=(float(conf_int[0]), float(conf_int[1])),
            alpha=float(self.alpha),
            design=resolved_design,
            target_parameter=_TARGET_PARAMETER[resolved_design],
            d_lower=d_lower_val,
            dose_mean=dose_mean,
            n_obs=n_obs,
            n_treated=n_treated,
            n_control=n_control,
            n_mass_point=n_mass_point,
            n_above_d_lower=n_above_d_lower,
            inference_method=inference_method,
            vcov_type=vcov_label,
            cluster_name=cluster_label,
            survey_metadata=survey_metadata,
            bandwidth_diagnostics=bw_diag,
            bias_corrected_fit=bc_fit,
        )

    # ------------------------------------------------------------------
    # Continuous-design dispatch (Design 1' + Design 1 continuous-near-d_lower)
    # ------------------------------------------------------------------

    def _fit_continuous(
        self,
        d_arr: np.ndarray,
        dy_arr: np.ndarray,
        resolved_design: str,
        d_lower_val: float,
        weights_arr: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, Optional[BiasCorrectedFit], Optional[BandwidthResult]]:
        """Fit Phase 1c ``bias_corrected_local_linear`` and form the WAS estimate.

        Implements de Chaisemartin, Ciccia, D'Haultfoeuille, and Knau
        (2026) continuous-design estimators:

        - Design 1' (``continuous_at_zero``), paper Theorem 1 /
          Equation 3 (identification); Equation 7 (sample estimator):

              beta = (E[Delta Y] - lim_{d v 0} E[Delta Y | D_2 <= d]) / E[D_2]

          Regressor passed to the local-linear boundary fit is
          ``d_arr``; the boundary is ``0``.

        - Design 1 (``continuous_near_d_lower``), paper Theorem 3 /
          Equation 11 (``WAS_{d_lower}`` under Assumption 6; note
          Theorem 4 is the QUG null test, not this estimand):

              beta = (E[Delta Y] - lim_{d v d_lower} E[Delta Y | D_2 <= d])
                     / E[D_2 - d_lower]

          Regressor passed to the local-linear fit is
          ``d_arr - d_lower`` so the boundary evaluation point is ``0``
          on the shifted scale; the numerator and denominator are
          computed on the original scale.

        The bias-corrected boundary estimate ``tau_bc`` from
        :func:`bias_corrected_local_linear` corresponds to the limit
        ``lim_{d v d_lower} E[Delta Y | D_2 <= d]``. The beta-scale
        estimator and standard error are then:

            att = (mean(dy) - tau_bc) / den
            se  = se_robust / |den|

        where ``den`` is the expectation in the denominator (``D_bar``
        for Design 1', ``mean(D_2 - d_lower)`` for Design 1). The
        confidence interval is ``att +/- z * se`` computed in
        :func:`diff_diff.utils.safe_inference`; the endpoints reverse
        relative to the boundary-limit CI because the numerator
        transformation is ``ΔȲ - tau_bc``.
        """
        # Weighted population moments (Phase 4.5). Under uniform weights,
        # `np.average(.., weights=np.ones(G)) == a.mean()` bit-exactly, so
        # the unweighted path is bit-identical when `weights_arr is None`.
        if resolved_design == "continuous_at_zero":
            d_reg = d_arr
            boundary = 0.0
            if weights_arr is None:
                den = float(d_arr.mean())
            else:
                den = float(np.average(d_arr, weights=weights_arr))
        elif resolved_design == "continuous_near_d_lower":
            d_reg = d_arr - d_lower_val
            boundary = 0.0
            if weights_arr is None:
                den = float(d_reg.mean())
            else:
                den = float(np.average(d_reg, weights=weights_arr))
        else:
            raise ValueError(
                f"_fit_continuous called with non-continuous " f"design={resolved_design!r}"
            )

        # Phase 1b/1c's bandwidth selector can hit degenerate ratios
        # (divide-by-zero in the bias estimator) when the outcome is
        # exactly constant or exactly linear in d (zero residuals).
        # The upstream fix lives in ``_nprobust_port.lpbwselect_mse_dpi``
        # and is deliberately out of Phase 2a scope (per plan "No changes
        # to local_linear.py / _nprobust_port.py"). Catch the symptomatic
        # exceptions here and surface a NaN result via ``safe_inference``
        # so users get a well-shaped ``HeterogeneousAdoptionDiDResults``
        # rather than a traceback.
        try:
            bc_fit = bias_corrected_local_linear(
                d=d_reg,
                y=dy_arr,
                boundary=boundary,
                kernel=self.kernel,
                alpha=float(self.alpha),
                weights=weights_arr,
                # No cluster / vce threading in Phase 2a (see UserWarning
                # in fit()).
            )
        except (ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
            return float("nan"), float("nan"), None, None

        # Guard against degenerate denominators: if all units are at
        # d_lower (continuous_near_d_lower) or if D_bar rounds to zero
        # (continuous_at_zero with a vanishing sample mean), the beta-
        # scale estimator is undefined. Fall through to NaN via
        # safe_inference in fit().
        if abs(den) < 1e-12:
            att = float("nan")
            se = float("nan")
        else:
            if weights_arr is None:
                dy_mean = float(dy_arr.mean())
            else:
                dy_mean = float(np.average(dy_arr, weights=weights_arr))
            tau_bc = float(bc_fit.estimate_bias_corrected)
            att = (dy_mean - tau_bc) / den
            se = float(bc_fit.se_robust) / abs(den)

        return att, se, bc_fit, bc_fit.bandwidth_diagnostics

    # ------------------------------------------------------------------
    # Event-study dispatch (Phase 2b, paper Appendix B.2)
    # ------------------------------------------------------------------

    def _fit_event_study(
        self,
        data: pd.DataFrame,
        outcome_col: str,
        dose_col: str,
        time_col: str,
        unit_col: str,
        first_treat_col: Optional[str],
    ) -> HeterogeneousAdoptionDiDEventStudyResults:
        """Multi-period event-study fit (paper Appendix B.2).

        Delegates to the multi-period panel validator (including staggered
        last-cohort auto-filter), aggregates per-horizon first differences
        against a common anchor ``Y_{g, F-1}``, resolves the design ONCE
        on the period-F dose distribution, and then fits the chosen design
        path independently on each event-time horizon's first differences.

        Per-horizon sandwich independence is the paper-faithful convention
        (Pierce-Schott Figure 2 pointwise CIs). Joint cross-horizon
        covariance is deferred to a follow-up PR.
        """
        # ---- Resolve effective fit-time state (local vars only,
        # feedback_fit_does_not_mutate_config). ----
        design_arg = self.design
        d_lower_arg = self.d_lower
        vcov_type_arg = self.vcov_type
        robust_arg = self.robust
        cluster_arg = self.cluster

        # ---- Validate multi-period panel and apply staggered filter ----
        F, t_pre_list, t_post_list, data_filtered, filter_info = _validate_had_panel_event_study(
            data, outcome_col, dose_col, time_col, unit_col, first_treat_col
        )

        # ---- Aggregate to per-horizon first differences ----
        # Cluster extraction is deferred until after design resolution.
        d_arr, dy_dict, _, _, _ = _aggregate_multi_period_first_differences(
            data_filtered,
            outcome_col,
            dose_col,
            time_col,
            unit_col,
            F,
            t_pre_list,
            t_post_list,
            None,
        )

        n_units = int(d_arr.shape[0])
        if n_units < 3:
            raise ValueError(
                f"HAD event-study requires at least 3 units for inference; "
                f"got n_units={n_units} after aggregation."
            )

        # ---- Resolve design (once, from D_{g, F} distribution) ----
        if design_arg == "auto":
            resolved_design = _detect_design(d_arr)
        else:
            resolved_design = design_arg

        # ---- Resolve d_lower ----
        if resolved_design == "continuous_at_zero":
            if d_lower_arg is not None:
                scale = max(1.0, float(np.max(np.abs(d_arr))))
                if abs(float(d_lower_arg)) > 1e-12 * scale:
                    raise ValueError(
                        f"design='continuous_at_zero' (Design 1') requires "
                        f"d_lower == 0 within float tolerance (paper Section "
                        f"3.2 Design 1' regime). Got d_lower="
                        f"{float(d_lower_arg)!r}. For d_lower > 0 use "
                        f"design='continuous_near_d_lower' or "
                        f"design='mass_point', or design='auto'."
                    )
            d_lower_val = 0.0
        elif d_lower_arg is None:
            d_lower_val = float(d_arr.min())
        else:
            d_lower_val = float(d_lower_arg)

        # ---- Regime-partition guards (mirror Phase 2a) ----
        if resolved_design in ("mass_point", "continuous_near_d_lower"):
            scale = max(1.0, float(np.max(np.abs(d_arr))))
            if abs(d_lower_val) <= 1e-12 * scale:
                raise ValueError(
                    f"design={resolved_design!r} requires d_lower > 0 (paper "
                    f"Section 3.2 reserves the d_lower=0 regime for Design 1' "
                    f"/ `continuous_at_zero`). Got d_lower={d_lower_val!r}."
                )

        if resolved_design in ("continuous_near_d_lower", "mass_point"):
            d_min_orig = float(d_arr.min())
            if d_min_orig > 0:
                eps_mp = 1e-12 * max(1.0, abs(d_min_orig))
                at_d_min_mask_orig = np.abs(d_arr - d_min_orig) <= eps_mp
                modal_fraction_orig = float(np.mean(at_d_min_mask_orig))
                if (
                    resolved_design == "continuous_near_d_lower"
                    and modal_fraction_orig > _MASS_POINT_THRESHOLD
                ):
                    raise ValueError(
                        f"design='continuous_near_d_lower' cannot be used on a "
                        f"mass-point sample (modal fraction {modal_fraction_orig:.4f} "
                        f"at d.min()={d_min_orig!r} exceeds the "
                        f"{_MASS_POINT_THRESHOLD:.2f} threshold)."
                    )
                if resolved_design == "mass_point" and modal_fraction_orig <= _MASS_POINT_THRESHOLD:
                    raise ValueError(
                        f"design='mass_point' requires a modal mass at d.min() "
                        f"exceeding the {_MASS_POINT_THRESHOLD:.2f} threshold "
                        f"(paper Section 3.2.4). Got modal fraction "
                        f"{modal_fraction_orig:.4f} at d.min()={d_min_orig!r}."
                    )

        if resolved_design in ("mass_point", "continuous_near_d_lower") and d_lower_arg is not None:
            d_min = float(d_arr.min())
            tol = 1e-12 * max(1.0, abs(d_min))
            if abs(d_lower_val - d_min) > tol:
                raise ValueError(
                    f"design={resolved_design!r} requires d_lower to equal "
                    f"the support infimum float(d.min())={d_min!r}; got "
                    f"d_lower={d_lower_val!r}."
                )
            d_lower_val = d_min  # snap

        dose_mean = float(d_arr.mean())

        # ---- Extract cluster IDs on mass-point path only ----
        cluster_arr: Optional[np.ndarray] = None
        if resolved_design == "mass_point" and cluster_arg is not None:
            _, _, cluster_arr, _, _ = _aggregate_multi_period_first_differences(
                data_filtered,
                outcome_col,
                dose_col,
                time_col,
                unit_col,
                F,
                t_pre_list,
                t_post_list,
                cluster_arg,
            )

        # ---- One-time warnings (per fit call, not per horizon) ----
        if resolved_design in ("continuous_near_d_lower", "mass_point"):
            warnings.warn(
                f"design={resolved_design!r} (Design 1, d_lower > 0) requires "
                f"Assumption 6 from de Chaisemartin et al. (2026) for point "
                f"identification of WAS_{{d_lower}}, or Assumption 5 for "
                f"sign identification only. Neither is testable via "
                f"pre-trends.",
                UserWarning,
                stacklevel=3,
            )

        if resolved_design in ("continuous_at_zero", "continuous_near_d_lower"):
            if vcov_type_arg is not None:
                warnings.warn(
                    f"vcov_type={vcov_type_arg!r} is ignored on the "
                    f"'{resolved_design}' path (continuous designs use the "
                    f"CCT-2014 robust SE from Phase 1c).",
                    UserWarning,
                    stacklevel=3,
                )
            if robust_arg:
                warnings.warn(
                    f"robust=True is ignored on the '{resolved_design}' " f"path.",
                    UserWarning,
                    stacklevel=3,
                )
            if cluster_arg is not None:
                warnings.warn(
                    f"cluster={cluster_arg!r} is ignored on the "
                    f"'{resolved_design}' path in Phase 2b (estimator-"
                    f"level cluster threading on the nonparametric path "
                    f"is queued for a follow-up PR).",
                    UserWarning,
                    stacklevel=3,
                )

        # ---- Resolve vcov label for mass-point ----
        if resolved_design == "mass_point":
            if vcov_type_arg is None:
                vcov_requested = "hc1" if robust_arg else "classical"
            else:
                vcov_requested = vcov_type_arg.lower()
            inference_method = "analytical_2sls"
            vcov_label: Optional[str] = "cr1" if cluster_arg is not None else vcov_requested
            cluster_label: Optional[str] = cluster_arg if cluster_arg is not None else None
        else:
            vcov_requested = ""
            inference_method = "analytical_nonparametric"
            vcov_label = None
            cluster_label = None

        # ---- Per-horizon loop ----
        event_times_sorted = sorted(dy_dict.keys())
        n_horizons = len(event_times_sorted)
        att_arr = np.full(n_horizons, np.nan, dtype=np.float64)
        se_arr = np.full(n_horizons, np.nan, dtype=np.float64)
        t_arr = np.full(n_horizons, np.nan, dtype=np.float64)
        p_arr = np.full(n_horizons, np.nan, dtype=np.float64)
        ci_lo_arr = np.full(n_horizons, np.nan, dtype=np.float64)
        ci_hi_arr = np.full(n_horizons, np.nan, dtype=np.float64)
        n_obs_arr = np.full(n_horizons, n_units, dtype=np.int64)

        # Collect per-horizon diagnostics on continuous paths. Entries may be
        # None for horizons where ``_fit_continuous`` caught a degenerate
        # bandwidth-selector failure (constant/perfectly-linear outcome).
        bc_fits: Optional[List[Optional[BiasCorrectedFit]]] = (
            [] if resolved_design in ("continuous_at_zero", "continuous_near_d_lower") else None
        )
        bw_diags: Optional[List[Optional[BandwidthResult]]] = (
            [] if resolved_design in ("continuous_at_zero", "continuous_near_d_lower") else None
        )

        for i, e in enumerate(event_times_sorted):
            dy_e = dy_dict[e]
            if resolved_design in ("continuous_at_zero", "continuous_near_d_lower"):
                att_e, se_e, bc_fit_e, bw_diag_e = self._fit_continuous(
                    d_arr, dy_e, resolved_design, d_lower_val
                )
                if bc_fits is not None:
                    bc_fits.append(bc_fit_e)
                if bw_diags is not None:
                    bw_diags.append(bw_diag_e)
            elif resolved_design == "mass_point":
                att_e, se_e = _fit_mass_point_2sls(
                    d_arr, dy_e, d_lower_val, cluster_arr, vcov_requested
                )
            else:
                raise ValueError(f"Internal error: unhandled design={resolved_design!r}.")

            t_stat_e, p_value_e, conf_int_e = safe_inference(att_e, se_e, alpha=float(self.alpha))
            att_arr[i] = float(att_e)
            se_arr[i] = float(se_e)
            t_arr[i] = float(t_stat_e)
            p_arr[i] = float(p_value_e)
            ci_lo_arr[i] = float(conf_int_e[0])
            ci_hi_arr[i] = float(conf_int_e[1])

        return HeterogeneousAdoptionDiDEventStudyResults(
            event_times=np.asarray(event_times_sorted, dtype=np.int64),
            att=att_arr,
            se=se_arr,
            t_stat=t_arr,
            p_value=p_arr,
            conf_int_low=ci_lo_arr,
            conf_int_high=ci_hi_arr,
            n_obs_per_horizon=n_obs_arr,
            alpha=float(self.alpha),
            design=resolved_design,
            target_parameter=_TARGET_PARAMETER[resolved_design],
            d_lower=d_lower_val,
            dose_mean=dose_mean,
            F=F,
            n_units=n_units,
            inference_method=inference_method,
            vcov_type=vcov_label,
            cluster_name=cluster_label,
            survey_metadata=None,
            bandwidth_diagnostics=bw_diags,
            bias_corrected_fit=bc_fits,
            filter_info=filter_info,
        )
