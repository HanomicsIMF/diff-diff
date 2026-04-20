"""
Heterogeneous Adoption Difference-in-Differences (HAD) estimator (Phase 2a).

Implements the de Chaisemartin, Ciccia, D'Haultfoeuille, and Knau (2026)
Weighted-Average-Slope (WAS) estimator with three design-dispatch paths:

1. Design 1' (``continuous_at_zero``): ``d_lower = 0``, boundary density
   continuous at zero. Evaluates the bias-corrected local-linear fit at
   ``boundary = 0`` and rescales to the beta-scale via Equation 8's divisor
   ``D_bar = (1/G) * sum(D_{g,2})``. Invokes Assumption 3.

2. Design 1 continuous-near-d_lower (``continuous_near_d_lower``):
   ``d_lower > 0``, continuous boundary density. Evaluates the bias-corrected
   local-linear fit at ``boundary = float(d.min())`` after the regressor
   shift ``D' = D - d_lower``. Same divisor, same CI path.
   Invokes Assumption 5 or 6.

3. Design 1 mass-point (``mass_point``): ``d_lower > 0``, modal fraction at
   ``d.min()`` exceeds 2%. Uses a sample-average 2SLS estimator with
   instrument ``Z_g = 1{D_{g,2} > d_lower}``. Point estimate reduces to the
   Wald-IV ratio
   ``(Ybar_{Z=1} - Ybar_{Z=0}) / (Dbar_{Z=1} - Dbar_{Z=0})``. Standard error
   via the structural-residual 2SLS sandwich (see ``_fit_mass_point_2sls``).

Phase 2a ships the single-period path only; the multi-period event-study
extension (paper Appendix B.2) is queued for Phase 2b.

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
from typing import Any, Dict, Optional, Tuple

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
# Results dataclass
# =============================================================================


@dataclass
class HeterogeneousAdoptionDiDResults:
    """Estimator output for :class:`HeterogeneousAdoptionDiD`.

    All inference fields (``att``, ``se``, ``t_stat``, ``p_value``,
    ``conf_int``) are routed through :func:`diff_diff.utils.safe_inference`
    at fit time; when SE is non-finite, zero, or negative, every inference
    field is NaN.

    Attributes
    ----------
    att : float
        Point estimate of the WAS parameter on the beta-scale. For the
        continuous designs: ``tau_bc / D_bar`` where ``tau_bc`` is the
        bias-corrected local-linear statistic on the mu-scale and
        ``D_bar = (1/G) * sum(D_{g,2})``. For the mass-point design:
        the 2SLS coefficient directly.
    se : float
        Standard error on the beta-scale. For continuous designs, the
        CCT-2014 robust SE divided by ``D_bar``. For mass-point, the 2SLS
        structural-residual sandwich SE.
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
        Variance-covariance family used. ``None`` on continuous paths
        (they use the CCT-2014 robust SE from Phase 1c, not the library's
        ``vcov_type`` enum). Mass-point: one of ``"classical"`` or
        ``"hc1"`` (CR1 when ``cluster`` is supplied).
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
            label = self.vcov_type
            if self.cluster_name is not None:
                label = f"CR1 at {self.cluster_name}"
            lines.append(f"{'Variance:':<30} {label:>20}")
        if self.bandwidth_diagnostics is not None:
            bw = self.bandwidth_diagnostics
            lines.append(f"{'Bandwidth h (MSE-DPI):':<30} {bw.h_mse:>20.6g}")
            lines.append(f"{'Bandwidth b (bias):':<30} {bw.b_mse:>20.6g}")
        if self.bias_corrected_fit is not None:
            bc = self.bias_corrected_fit
            lines.append(f"{'Bandwidth h used:':<30} {bc.h:>20.6g}")
            lines.append(f"{'Obs in window (n_used):':<30} {bc.n_used:>20}")
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
                    f"{'WAS':<15} {self.att:>12.4f} {self.se:>12.4f} "
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
        """Return results as a dict of scalars."""
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
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a one-row DataFrame of the result dict."""
        return pd.DataFrame([self.to_dict()])


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
    tuple[int, int]
        ``(t_pre, t_post)`` - the two period identifiers, min then max.

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

    periods = np.sort(np.asarray(data[time_col].unique()))
    if len(periods) < 2:
        raise ValueError(
            f"HAD requires a two-period panel; got {len(periods)} distinct "
            f"period(s) in column {time_col!r}."
        )
    if len(periods) > 2:
        if first_treat_col is None:
            raise ValueError(
                f"HAD Phase 2a requires exactly two time periods "
                f"(got {len(periods)} in {time_col!r}) when "
                f"first_treat_col=None. Multi-period / staggered adoption "
                f"support is queued for Phase 2b (Appendix B.2 event-study)."
            )
        raise ValueError(
            f"HAD Phase 2a requires exactly two time periods "
            f"(got {len(periods)} in {time_col!r}). Staggered adoption "
            f"reduction (first_treat_col supplied with >2 periods) is "
            f"queued for Phase 2b (Appendix B.2 event-study)."
        )

    t_pre = int(periods[0]) if np.issubdtype(periods.dtype, np.integer) else periods[0]
    t_post = int(periods[1]) if np.issubdtype(periods.dtype, np.integer) else periods[1]

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

    # Pre-period no-unit-untreated check.
    pre_mask = data[time_col] == t_pre
    pre_doses = np.asarray(data.loc[pre_mask, dose_col], dtype=np.float64)
    nonzero_pre = pre_doses != 0
    if nonzero_pre.any():
        n_bad = int(nonzero_pre.sum())
        raise ValueError(
            f"HAD requires D_{{g,1}} = 0 for all units (pre-period "
            f"untreated). {n_bad} unit(s) have nonzero dose at "
            f"t_pre={t_pre}. Drop these units or verify the dose column."
        )

    # Optional value-domain validation via first_treat_col: if supplied,
    # every unit's first_treat value must be in {0, t_post} (0 = never
    # treated, t_post = treated in the second period). This is a value-
    # domain check that catches typos and staggered-timing mix-ups; it
    # does NOT cross-validate first_treat against post-period dose
    # (D_{g, t_post} remains the primary signal). Extended cross-checks
    # are queued for a follow-up PR.
    if first_treat_col is not None:
        ft = data.groupby(unit_col)[first_treat_col].first().to_numpy()
        if np.any(np.isnan(ft.astype(np.float64, copy=False))):
            raise ValueError(
                f"first_treat_col={first_treat_col!r} contains NaN. "
                f"Use 0 for never-treated units and t_post for treated."
            )
        valid_values = {0, t_post}
        bad = [v for v in np.unique(ft) if int(v) not in valid_values]
        if bad:
            raise ValueError(
                f"first_treat_col={first_treat_col!r} contains value(s) "
                f"{bad} outside the allowed set {{0, {t_post}}} for a "
                f"two-period HAD panel. Staggered timing with multiple "
                f"cohorts is Phase 2b."
            )

    return t_pre, t_post


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
        # Cluster must be unit-constant; take the first cluster id per unit.
        cluster_per_unit = df.groupby(unit_col)[cluster_col].nunique()
        if (cluster_per_unit > 1).any():
            n_bad = int((cluster_per_unit > 1).sum())
            raise ValueError(
                f"cluster column {cluster_col!r} is not constant within "
                f"unit for {n_bad} unit(s). Cluster must be unit-level "
                f"(e.g., cluster_col=unit_col, or a coarser grouping "
                f"where each unit belongs to a single cluster)."
            )
        cluster_arr = df.groupby(unit_col)[cluster_col].first().sort_index().to_numpy()
        # NaN cluster ids are a silent-failure vector; reject front-door.
        if pd.isna(cluster_arr).any():
            n_nan = int(pd.isna(cluster_arr).sum())
            raise ValueError(
                f"{n_nan} unit(s) have NaN cluster id in "
                f"column {cluster_col!r}. Silent row dropping is "
                f"disabled; drop or impute cluster ids before calling fit()."
            )

    return d_arr, dy_arr, cluster_arr, unit_ids


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
    Section 3.2.4). Phase 2a ships the single-period path only; the
    multi-period event-study extension (Appendix B.2) is queued for
    Phase 2b.

    Parameters
    ----------
    design : {"auto", "continuous_at_zero", "continuous_near_d_lower", "mass_point"}
        Design-dispatch strategy. Defaults to ``"auto"`` which resolves
        via the REGISTRY auto-detect rule on the fitted dose data
        (see :func:`_detect_design`). Explicit overrides run the chosen
        path without auto-reject (e.g., forcing Design 1' on mass-point
        data runs the nonparametric fit even though the paper would
        counsel the 2SLS path).
    d_lower : float or None
        Support infimum ``d_lower``. ``None`` means use ``0.0`` on the
        Design 1' path and ``float(d.min())`` on the other two paths.
    kernel : {"epanechnikov", "triangular", "uniform"}
        Forwarded to :func:`bias_corrected_local_linear` on the continuous
        paths. Ignored on the mass-point path.
    alpha : float
        CI level (0.05 for 95% CI).
    vcov_type : {"classical", "hc1"} or None
        Mass-point-path only. ``None`` defaults to ``"hc1"``.
        ``"hc2"`` and ``"hc2_bm"`` raise ``NotImplementedError``. Ignored
        on the continuous paths (which use the CCT-2014 robust SE from
        Phase 1c); passing a non-default ``vcov_type`` on a continuous
        path emits a one-time ``UserWarning``.
    robust : bool
        Backward-compat alias: ``True`` maps to ``"hc1"``, ``False`` maps
        to ``"classical"``. Only relevant when ``vcov_type is None`` on
        the mass-point path.
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
    >>> import pandas as pd
    >>> from diff_diff import HeterogeneousAdoptionDiD, generate_continuous_did_data
    >>> data = generate_continuous_did_data(n_units=500, seed=42)  # doctest: +SKIP
    >>> est = HeterogeneousAdoptionDiD(design="auto")  # doctest: +SKIP
    >>> result = est.fit(data, outcome_col="outcome", dose_col="dose",
    ...                  time_col="period", unit_col="unit")  # doctest: +SKIP
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

    def get_params(self) -> Dict[str, Any]:
        """Return the raw constructor parameters (sklearn-compatible).

        Preserves the user's original inputs - in particular, ``design``
        returns ``"auto"`` when the user set it to ``"auto"`` (even after
        fit), so ``sklearn.base.clone(est)`` round-trips exactly.
        """
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
        """Set estimator parameters and return self (sklearn-compatible)."""
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(
                    f"Invalid parameter: {key}. Valid parameters: "
                    f"{list(self.get_params().keys())}."
                )
            setattr(self, key, value)
        self._validate_constructor_args()
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
        """Fit the HAD estimator on a two-period panel.

        Parameters
        ----------
        data : pd.DataFrame
        outcome_col, dose_col, time_col, unit_col : str
            Column names.
        first_treat_col : str or None
            Optional first-treatment column for cross-validation.
            Required when the panel has more than two periods (raises
            for >2 periods in Phase 2a; Phase 2b handles staggered).
        aggregate : {"overall"}
            ``"event_study"`` raises ``NotImplementedError`` (Phase 2b).
        survey : SurveyDesign or None
            Reserved for a follow-up survey-integration PR. Must be
            ``None`` in Phase 2a.
        weights : np.ndarray or None
            Reserved for a follow-up PR. Must be ``None`` in Phase 2a.

        Returns
        -------
        HeterogeneousAdoptionDiDResults
        """
        # ---- Phase 2a scaffolding rejections (before any work) ----
        if aggregate not in _VALID_AGGREGATES:
            raise ValueError(
                f"Invalid aggregate={aggregate!r}. Must be one of " f"{_VALID_AGGREGATES}."
            )
        if aggregate == "event_study":
            raise NotImplementedError(
                "aggregate='event_study' (multi-period event study per "
                "paper Appendix B.2) is queued for Phase 2b. Phase 2a "
                "supports aggregate='overall' (single-period WAS) only."
            )
        if survey is not None:
            raise NotImplementedError(
                "survey=<SurveyDesign> support on HeterogeneousAdoptionDiD "
                "is queued for a follow-up PR after Phase 2a. Pass "
                "survey=None for now."
            )
        if weights is not None:
            raise NotImplementedError(
                "weights=<array> support on HeterogeneousAdoptionDiD is "
                "queued for a follow-up PR (paired with survey "
                "integration). Pass weights=None for now."
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

        # ---- Aggregate to unit-level first differences ----
        d_arr, dy_arr, cluster_arr, _ = _aggregate_first_difference(
            data,
            outcome_col,
            dose_col,
            time_col,
            unit_col,
            t_pre,
            t_post,
            cluster_arg,
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

        # ---- Resolve d_lower ----
        if resolved_design == "continuous_at_zero":
            d_lower_val = 0.0
        elif d_lower_arg is None:
            d_lower_val = float(d_arr.min())
        else:
            d_lower_val = float(d_lower_arg)

        # ---- Mass-point contract: d_lower must equal the support infimum ----
        # Paper Section 3.2.4 defines the mass-point estimator with instrument
        # Z = 1{D_{g,2} > d_lower} where d_lower is the lower-support mass
        # point. A user-supplied d_lower that differs from float(d_arr.min())
        # redefines the instrument / control split and identifies a different
        # estimand. Phase 2a supports only the paper's lower-support mass-point
        # estimator, so we reject mismatched overrides front-door rather than
        # silently running an unsupported variant.
        if resolved_design == "mass_point" and d_lower_arg is not None:
            d_min = float(d_arr.min())
            tol = 1e-12 * max(1.0, abs(d_min))
            if abs(d_lower_val - d_min) > tol:
                raise ValueError(
                    f"design='mass_point' requires d_lower to equal the "
                    f"support infimum float(d.min())={d_min!r}; got "
                    f"d_lower={d_lower_val!r}. The paper's Design 1 mass-"
                    f"point estimator (Section 3.2.4) identifies at the "
                    f"lower-support mass point, not at an arbitrary "
                    f"threshold. Pass d_lower=None to auto-resolve, or "
                    f"d_lower=float(d.min()) explicitly. Non-support-"
                    f"infimum thresholds identify a different (LATE-like) "
                    f"estimand that is out of Phase 2a scope."
                )

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
            )
            inference_method = "analytical_nonparametric"
            vcov_label: Optional[str] = None
            cluster_label: Optional[str] = None
        elif resolved_design == "mass_point":
            if vcov_type_arg is None:
                # Backward-compat: robust=True -> hc1, robust=False -> classical.
                vcov_effective = "hc1" if robust_arg else "classical"
            else:
                vcov_effective = vcov_type_arg.lower()
            # Explicit cluster + hc1 becomes CR1 inside _fit_mass_point_2sls.
            att, se = _fit_mass_point_2sls(
                d_arr,
                dy_arr,
                d_lower_val,
                cluster_arr,
                vcov_effective,
            )
            bc_fit = None
            bw_diag = None
            inference_method = "analytical_2sls"
            vcov_label = vcov_effective
            cluster_label = cluster_arg if cluster_arg is not None else None
        else:
            raise ValueError(f"Internal error: unhandled design={resolved_design!r}.")

        # ---- Route all inference fields through safe_inference ----
        t_stat, p_value, conf_int = safe_inference(att, se, alpha=float(self.alpha))

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
            survey_metadata=None,  # Phase 2a: survey integration deferred.
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
    ) -> Tuple[float, float, Optional[BiasCorrectedFit], Optional[BandwidthResult]]:
        """Fit Phase 1c ``bias_corrected_local_linear`` and rescale to beta-scale.

        Both continuous designs share the same rescaling path - they
        differ only in the regressor shift and boundary:

        - ``continuous_at_zero``: regressor ``d_arr``, boundary ``0``.
        - ``continuous_near_d_lower``: regressor ``d_arr - d_lower``,
          boundary ``0``.

        The mu-scale output ``tau_bc`` from
        :func:`bias_corrected_local_linear` is divided by the
        beta-scale divisor ``D_bar = (1/G) * sum(D_{g,2})`` (Equation 8
        in the paper). The same divisor applies to the classical CI
        endpoints. The result of ``D_bar`` uses the ORIGINAL
        (unshifted) dose mean, matching paper convention.
        """
        if resolved_design == "continuous_at_zero":
            d_reg = d_arr
            boundary = 0.0
        elif resolved_design == "continuous_near_d_lower":
            d_reg = d_arr - d_lower_val
            boundary = 0.0
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
                # No cluster / vce / weights threading in Phase 2a (see
                # UserWarning in fit()).
            )
        except (ZeroDivisionError, FloatingPointError, np.linalg.LinAlgError):
            return float("nan"), float("nan"), None, None

        dose_mean = float(d_arr.mean())
        # Phase 1b / Phase 1c pre-checks enforce dose_mean > 0 for
        # Design 1-family samples, but defense-in-depth: if dose_mean
        # rounds to zero, fall through to NaN via safe_inference in fit().
        if abs(dose_mean) < 1e-12:
            att = float("nan")
            se = float("nan")
        else:
            att = float(bc_fit.estimate_bias_corrected) / dose_mean
            se = float(bc_fit.se_robust) / dose_mean

        return att, se, bc_fit, bc_fit.bandwidth_diagnostics
