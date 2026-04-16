"""
Results classes for difference-in-differences estimation.

Provides statsmodels-style output with a more Pythonic interface.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _format_survey_block(sm, width: int) -> list:
    """Format survey design metadata block for summary() output.

    Parameters
    ----------
    sm : SurveyMetadata
        Survey metadata from results object.
    width : int
        Total width for separator lines and centering.
    """
    label_width = 30 if width >= 80 else 25
    lines = [
        "",
        "-" * width,
        "Survey Design".center(width),
        "-" * width,
        f"{'Weight type:':<{label_width}} {sm.weight_type:>10}",
    ]
    if getattr(sm, "replicate_method", None) is not None:
        lines.append(f"{'Replicate method:':<{label_width}} {sm.replicate_method:>10}")
        if getattr(sm, "n_replicates", None) is not None:
            lines.append(f"{'Replicates:':<{label_width}} {sm.n_replicates:>10}")
    else:
        if sm.n_strata is not None:
            lines.append(f"{'Strata:':<{label_width}} {sm.n_strata:>10}")
        if sm.n_psu is not None:
            lines.append(f"{'PSU/Cluster:':<{label_width}} {sm.n_psu:>10}")
    lines.append(f"{'Effective sample size:':<{label_width}} {sm.effective_n:>10.1f}")
    lines.append(f"{'Kish DEFF (weights):':<{label_width}} {sm.design_effect:>10.2f}")
    if sm.df_survey is not None:
        lines.append(f"{'Survey d.f.:':<{label_width}} {sm.df_survey:>10}")
    lines.append("-" * width)
    return lines


@dataclass
class DiDResults:
    """
    Results from a Difference-in-Differences estimation.

    Provides easy access to coefficients, standard errors, confidence intervals,
    and summary statistics in a Pythonic way.

    Attributes
    ----------
    att : float
        Average Treatment effect on the Treated (ATT).
    se : float
        Standard error of the ATT estimate.
    t_stat : float
        T-statistic for the ATT estimate.
    p_value : float
        P-value for the null hypothesis that ATT = 0.
    conf_int : tuple[float, float]
        Confidence interval for the ATT.
    n_obs : int
        Number of observations used in estimation.
    n_treated : int
        Number of treated units/observations.
    n_control : int
        Number of control units/observations.
    """

    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    n_obs: int
    n_treated: int
    n_control: int
    alpha: float = 0.05
    coefficients: Optional[Dict[str, float]] = field(default=None)
    vcov: Optional[np.ndarray] = field(default=None)
    residuals: Optional[np.ndarray] = field(default=None)
    fitted_values: Optional[np.ndarray] = field(default=None)
    r_squared: Optional[float] = field(default=None)
    # Bootstrap inference fields
    inference_method: str = field(default="analytical")
    n_bootstrap: Optional[int] = field(default=None)
    n_clusters: Optional[int] = field(default=None)
    bootstrap_distribution: Optional[np.ndarray] = field(default=None, repr=False)
    # Survey design metadata (SurveyMetadata instance from diff_diff.survey)
    survey_metadata: Optional[Any] = field(default=None)

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"DiDResults(ATT={self.att:.4f}{self.significance_stars}, "
            f"SE={self.se:.4f}, "
            f"p={self.p_value:.4f})"
        )

    @property
    def coef_var(self) -> float:
        """Coefficient of variation: SE / |ATT|. NaN when ATT is 0 or SE non-finite."""
        if not (np.isfinite(self.se) and self.se >= 0):
            return np.nan
        if not np.isfinite(self.att) or self.att == 0:
            return np.nan
        return self.se / abs(self.att)

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate a formatted summary of the estimation results.

        Parameters
        ----------
        alpha : float, optional
            Significance level for confidence intervals. Defaults to the
            alpha used during estimation.

        Returns
        -------
        str
            Formatted summary table.
        """
        alpha = alpha or self.alpha
        conf_level = int((1 - alpha) * 100)

        lines = [
            "=" * 70,
            "Difference-in-Differences Estimation Results".center(70),
            "=" * 70,
            "",
            f"{'Observations:':<25} {self.n_obs:>10}",
            f"{'Treated:':<25} {self.n_treated:>10}",
            f"{'Control:':<25} {self.n_control:>10}",
        ]

        if self.r_squared is not None:
            lines.append(f"{'R-squared:':<25} {self.r_squared:>10.4f}")

        # Add survey design info
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            lines.extend(_format_survey_block(sm, 70))

        # Add inference method info
        if self.inference_method != "analytical":
            lines.append(f"{'Inference method:':<25} {self.inference_method:>10}")
            if self.n_bootstrap is not None:
                lines.append(f"{'Bootstrap replications:':<25} {self.n_bootstrap:>10}")
            if self.n_clusters is not None:
                lines.append(f"{'Number of clusters:':<25} {self.n_clusters:>10}")

        lines.extend(
            [
                "",
                "-" * 70,
                f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'':>5}",
                "-" * 70,
                f"{'ATT':<15} {self.att:>12.4f} {self.se:>12.4f} {self.t_stat:>10.3f} {self.p_value:>10.4f} {self.significance_stars:>5}",
                "-" * 70,
                "",
                f"{conf_level}% Confidence Interval: [{self.conf_int[0]:.4f}, {self.conf_int[1]:.4f}]",
            ]
        )

        cv = self.coef_var
        if np.isfinite(cv):
            lines.append(f"{'CV (SE/|ATT|):':<25} {cv:>10.4f}")

        # Add significance codes
        lines.extend(
            [
                "",
                "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
                "=" * 70,
            ]
        )

        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print the summary to stdout."""
        print(self.summary(alpha))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all estimation results.
        """
        result = {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int[0],
            "conf_int_upper": self.conf_int[1],
            "n_obs": self.n_obs,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "r_squared": self.r_squared,
            "inference_method": self.inference_method,
        }
        if self.n_bootstrap is not None:
            result["n_bootstrap"] = self.n_bootstrap
        if self.n_clusters is not None:
            result["n_clusters"] = self.n_clusters
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            result["weight_type"] = sm.weight_type
            result["effective_n"] = sm.effective_n
            result["design_effect"] = sm.design_effect
            result["sum_weights"] = sm.sum_weights
            result["n_strata"] = sm.n_strata
            result["n_psu"] = sm.n_psu
            result["df_survey"] = sm.df_survey
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with estimation results.
        """
        return pd.DataFrame([self.to_dict()])

    @property
    def is_significant(self) -> bool:
        """Check if the ATT is statistically significant at the alpha level."""
        return bool(self.p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        return _get_significance_stars(self.p_value)


def _get_significance_stars(p_value: float) -> str:
    """Return significance stars based on p-value.

    Returns empty string for NaN p-values (unidentified coefficients from
    rank-deficient matrices).
    """
    import numpy as np

    if np.isnan(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    elif p_value < 0.1:
        return "."
    return ""


@dataclass
class PeriodEffect:
    """
    Treatment effect for a single time period.

    Attributes
    ----------
    period : any
        The time period identifier.
    effect : float
        The treatment effect estimate for this period.
    se : float
        Standard error of the effect estimate.
    t_stat : float
        T-statistic for the effect estimate.
    p_value : float
        P-value for the null hypothesis that effect = 0.
    conf_int : tuple[float, float]
        Confidence interval for the effect.
    """

    period: Any
    effect: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.p_value)
        return (
            f"PeriodEffect(period={self.period}, effect={self.effect:.4f}{sig}, "
            f"SE={self.se:.4f}, p={self.p_value:.4f})"
        )

    @property
    def is_significant(self) -> bool:
        """Check if the effect is statistically significant at 0.05 level."""
        return bool(self.p_value < 0.05)

    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        return _get_significance_stars(self.p_value)


@dataclass
class MultiPeriodDiDResults:
    """
    Results from a Multi-Period Difference-in-Differences estimation.

    Provides access to period-specific treatment effects as well as
    an aggregate average treatment effect.

    Attributes
    ----------
    period_effects : dict[any, PeriodEffect]
        Dictionary mapping period identifiers to their PeriodEffect objects.
        Contains all estimated period effects (pre and post, excluding
        the reference period which is normalized to zero).
    avg_att : float
        Average Treatment effect on the Treated across post-periods only.
    avg_se : float
        Standard error of the average ATT.
    avg_t_stat : float
        T-statistic for the average ATT.
    avg_p_value : float
        P-value for the null hypothesis that average ATT = 0.
    avg_conf_int : tuple[float, float]
        Confidence interval for the average ATT.
    n_obs : int
        Number of observations used in estimation.
    n_treated : int
        Number of treated units/observations.
    n_control : int
        Number of control units/observations.
    pre_periods : list
        List of pre-treatment period identifiers.
    post_periods : list
        List of post-treatment period identifiers.
    reference_period : any, optional
        The reference (omitted) period. Its coefficient is zero by
        construction and it is excluded from ``period_effects``.
    interaction_indices : dict, optional
        Mapping from period identifier to column index in the full
        variance-covariance matrix. Used internally for sub-VCV
        extraction (e.g., by HonestDiD and PreTrendsPower).
    """

    period_effects: Dict[Any, PeriodEffect]
    avg_att: float
    avg_se: float
    avg_t_stat: float
    avg_p_value: float
    avg_conf_int: Tuple[float, float]
    n_obs: int
    n_treated: int
    n_control: int
    pre_periods: List[Any]
    post_periods: List[Any]
    alpha: float = 0.05
    coefficients: Optional[Dict[str, float]] = field(default=None)
    vcov: Optional[np.ndarray] = field(default=None)
    residuals: Optional[np.ndarray] = field(default=None)
    fitted_values: Optional[np.ndarray] = field(default=None)
    r_squared: Optional[float] = field(default=None)
    reference_period: Optional[Any] = field(default=None)
    interaction_indices: Optional[Dict[Any, int]] = field(default=None, repr=False)
    # Survey design metadata (SurveyMetadata instance from diff_diff.survey)
    survey_metadata: Optional[Any] = field(default=None)

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.avg_p_value)
        return (
            f"MultiPeriodDiDResults(avg_ATT={self.avg_att:.4f}{sig}, "
            f"SE={self.avg_se:.4f}, "
            f"n_post_periods={len(self.post_periods)})"
        )

    @property
    def pre_period_effects(self) -> Dict[Any, PeriodEffect]:
        """Pre-period effects only (for parallel trends assessment)."""
        return {p: pe for p, pe in self.period_effects.items() if p in self.pre_periods}

    @property
    def post_period_effects(self) -> Dict[Any, PeriodEffect]:
        """Post-period effects only."""
        return {p: pe for p, pe in self.period_effects.items() if p in self.post_periods}

    @property
    def coef_var(self) -> float:
        """Coefficient of variation: SE / |overall ATT|. NaN when ATT is 0 or SE non-finite."""
        if not (np.isfinite(self.avg_se) and self.avg_se >= 0):
            return np.nan
        if not np.isfinite(self.avg_att) or self.avg_att == 0:
            return np.nan
        return self.avg_se / abs(self.avg_att)

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate a formatted summary of the estimation results.

        Parameters
        ----------
        alpha : float, optional
            Significance level for confidence intervals. Defaults to the
            alpha used during estimation.

        Returns
        -------
        str
            Formatted summary table.
        """
        alpha = alpha or self.alpha
        conf_level = int((1 - alpha) * 100)

        lines = [
            "=" * 80,
            "Multi-Period Difference-in-Differences Estimation Results".center(80),
            "=" * 80,
            "",
            f"{'Observations:':<25} {self.n_obs:>10}",
            f"{'Treated observations:':<25} {self.n_treated:>10}",
            f"{'Control observations:':<25} {self.n_control:>10}",
            f"{'Pre-treatment periods:':<25} {len(self.pre_periods):>10}",
            f"{'Post-treatment periods:':<25} {len(self.post_periods):>10}",
        ]

        if self.r_squared is not None:
            lines.append(f"{'R-squared:':<25} {self.r_squared:>10.4f}")

        # Add survey design info
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            lines.extend(_format_survey_block(sm, 80))

        # Pre-period effects (parallel trends test)
        pre_effects = {p: pe for p, pe in self.period_effects.items() if p in self.pre_periods}
        if pre_effects:
            lines.extend(
                [
                    "",
                    "-" * 80,
                    "Pre-Period Effects (Parallel Trends Test)".center(80),
                    "-" * 80,
                    f"{'Period':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                    "-" * 80,
                ]
            )

            for period in self.pre_periods:
                if period in self.period_effects:
                    pe = self.period_effects[period]
                    stars = pe.significance_stars
                    lines.append(
                        f"{str(period):<15} {pe.effect:>12.4f} {pe.se:>12.4f} "
                        f"{pe.t_stat:>10.3f} {pe.p_value:>10.4f} {stars:>6}"
                    )

            # Show reference period
            if self.reference_period is not None:
                lines.append(
                    f"[ref: {self.reference_period}]"
                    f"{'0.0000':>21} {'---':>12} {'---':>10} {'---':>10} {'':>6}"
                )

            lines.append("-" * 80)

        # Post-period treatment effects
        lines.extend(
            [
                "",
                "-" * 80,
                "Post-Period Treatment Effects".center(80),
                "-" * 80,
                f"{'Period':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                "-" * 80,
            ]
        )

        for period in self.post_periods:
            pe = self.period_effects[period]
            stars = pe.significance_stars
            lines.append(
                f"{str(period):<15} {pe.effect:>12.4f} {pe.se:>12.4f} "
                f"{pe.t_stat:>10.3f} {pe.p_value:>10.4f} {stars:>6}"
            )

        # Average effect
        lines.extend(
            [
                "-" * 80,
                "",
                "-" * 80,
                "Average Treatment Effect (across post-periods)".center(80),
                "-" * 80,
                f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}",
                "-" * 80,
                f"{'Avg ATT':<15} {self.avg_att:>12.4f} {self.avg_se:>12.4f} "
                f"{self.avg_t_stat:>10.3f} {self.avg_p_value:>10.4f} {self.significance_stars:>6}",
                "-" * 80,
                "",
                f"{conf_level}% Confidence Interval: [{self.avg_conf_int[0]:.4f}, {self.avg_conf_int[1]:.4f}]",
            ]
        )

        cv = self.coef_var
        if np.isfinite(cv):
            lines.append(f"{'CV (SE/|ATT|):':<25} {cv:>10.4f}")

        # Add significance codes
        lines.extend(
            [
                "",
                "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
                "=" * 80,
            ]
        )

        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print the summary to stdout."""
        print(self.summary(alpha))

    def get_effect(self, period) -> PeriodEffect:
        """
        Get the treatment effect for a specific period.

        Parameters
        ----------
        period : any
            The period identifier.

        Returns
        -------
        PeriodEffect
            The treatment effect for the specified period.

        Raises
        ------
        KeyError
            If the period is not found in post-treatment periods.
        """
        if period not in self.period_effects:
            if hasattr(self, "reference_period") and period == self.reference_period:
                raise KeyError(
                    f"Period '{period}' is the reference period (coefficient "
                    f"normalized to zero by construction). Its effect is 0.0 with "
                    f"no associated uncertainty."
                )
            raise KeyError(
                f"Period '{period}' not found. "
                f"Available periods: {list(self.period_effects.keys())}"
            )
        return self.period_effects[period]

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all estimation results.
        """
        result: Dict[str, Any] = {
            "avg_att": self.avg_att,
            "avg_se": self.avg_se,
            "avg_t_stat": self.avg_t_stat,
            "avg_p_value": self.avg_p_value,
            "avg_conf_int_lower": self.avg_conf_int[0],
            "avg_conf_int_upper": self.avg_conf_int[1],
            "n_obs": self.n_obs,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "n_pre_periods": len(self.pre_periods),
            "n_post_periods": len(self.post_periods),
            "r_squared": self.r_squared,
            "reference_period": self.reference_period,
        }

        # Add period-specific effects
        for period, pe in self.period_effects.items():
            result[f"effect_period_{period}"] = pe.effect
            result[f"se_period_{period}"] = pe.se
            result[f"pval_period_{period}"] = pe.p_value

        # Add survey metadata if present
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            result["weight_type"] = sm.weight_type
            result["effective_n"] = sm.effective_n
            result["design_effect"] = sm.design_effect
            result["sum_weights"] = sm.sum_weights
            result["n_strata"] = sm.n_strata
            result["n_psu"] = sm.n_psu
            result["df_survey"] = sm.df_survey

        return result

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert period-specific effects to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per estimated period (pre and post).
        """
        rows = []
        for period, pe in self.period_effects.items():
            rows.append(
                {
                    "period": period,
                    "effect": pe.effect,
                    "se": pe.se,
                    "t_stat": pe.t_stat,
                    "p_value": pe.p_value,
                    "conf_int_lower": pe.conf_int[0],
                    "conf_int_upper": pe.conf_int[1],
                    "is_significant": pe.is_significant,
                    "is_post": period in self.post_periods,
                }
            )
        return pd.DataFrame(rows)

    @property
    def is_significant(self) -> bool:
        """Check if the average ATT is statistically significant at the alpha level."""
        return bool(self.avg_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Return significance stars for the average ATT based on p-value."""
        return _get_significance_stars(self.avg_p_value)


@dataclass
class _SyntheticDiDFitSnapshot:
    """Internal snapshot of fit() state retained on SyntheticDiDResults for
    post-hoc diagnostic methods (in-time placebo, regularization sensitivity).

    Not part of the public API. Arrays are marked read-only after
    construction to prevent accidental mutation by diagnostic methods.
    Memory scales as O(n_units * n_periods).
    """

    Y_pre_control: np.ndarray
    Y_post_control: np.ndarray
    Y_pre_treated: np.ndarray
    Y_post_treated: np.ndarray
    control_unit_ids: List[Any]
    treated_unit_ids: List[Any]
    pre_periods: List[Any]
    post_periods: List[Any]
    w_control: Optional[np.ndarray] = None
    w_treated: Optional[np.ndarray] = None

    def __post_init__(self):
        for arr in (
            self.Y_pre_control,
            self.Y_post_control,
            self.Y_pre_treated,
            self.Y_post_treated,
        ):
            arr.setflags(write=False)
        for arr in (self.w_control, self.w_treated):
            if arr is not None:
                arr.setflags(write=False)


@dataclass
class SyntheticDiDResults:
    """
    Results from a Synthetic Difference-in-Differences estimation.

    Combines DiD with synthetic control by re-weighting control units to match
    pre-treatment trends of treated units.

    Attributes
    ----------
    att : float
        Average Treatment effect on the Treated (ATT).
    se : float
        Standard error of the ATT estimate (bootstrap, jackknife, or placebo-based).
    t_stat : float
        T-statistic for the ATT estimate.
    p_value : float
        P-value for the null hypothesis that ATT = 0.
    conf_int : tuple[float, float]
        Confidence interval for the ATT.
    n_obs : int
        Number of observations used in estimation.
    n_treated : int
        Number of treated units/observations.
    n_control : int
        Number of control units/observations.
    unit_weights : dict
        Dictionary mapping control unit IDs to their synthetic weights.
        When survey weights are used, these are the composed effective
        weights (omega_eff = raw Frank-Wolfe * survey, renormalized) that
        were applied to produce the ATT, not the raw Frank-Wolfe solution.
    time_weights : dict
        Dictionary mapping pre-treatment periods to their time weights.
    pre_periods : list
        List of pre-treatment period identifiers.
    post_periods : list
        List of post-treatment period identifiers.
    variance_method : str
        Method used for variance estimation: "bootstrap", "jackknife", or "placebo".
    placebo_effects : np.ndarray, optional
        Method-specific per-iteration estimates: placebo treatment effects
        (for "placebo"), bootstrap ATT estimates (for "bootstrap"), or
        leave-one-out estimates (for "jackknife"). The ``variance_method``
        field disambiguates the contents.
    synthetic_pre_trajectory : np.ndarray, optional
        Synthetic control trajectory in pre-treatment periods, shape
        ``(n_pre,)``. Equal to ``Y_pre_control @ omega_eff`` where
        ``omega_eff`` is the composed effective weight vector.
    synthetic_post_trajectory : np.ndarray, optional
        Synthetic control trajectory in post-treatment periods, shape
        ``(n_post,)``.
    treated_pre_trajectory : np.ndarray, optional
        Treated-unit mean trajectory in pre-treatment periods, shape
        ``(n_pre,)``. Survey-weighted when the fit used survey weights.
    treated_post_trajectory : np.ndarray, optional
        Treated-unit mean trajectory in post-treatment periods, shape
        ``(n_post,)``.
    time_weights_array : np.ndarray, optional
        The Frank-Wolfe time weights as a 1-D array (same values as the
        ``time_weights`` dict but order-stable and usable for re-estimation
        by sensitivity methods). Shape ``(n_pre,)``.
    """

    att: float
    se: float
    t_stat: float
    p_value: float
    conf_int: Tuple[float, float]
    n_obs: int
    n_treated: int
    n_control: int
    unit_weights: Dict[Any, float]
    time_weights: Dict[Any, float]
    pre_periods: List[Any]
    post_periods: List[Any]
    alpha: float = 0.05
    variance_method: str = field(default="placebo")
    noise_level: Optional[float] = field(default=None)
    zeta_omega: Optional[float] = field(default=None)
    zeta_lambda: Optional[float] = field(default=None)
    pre_treatment_fit: Optional[float] = field(default=None)
    placebo_effects: Optional[np.ndarray] = field(default=None)
    n_bootstrap: Optional[int] = field(default=None)
    # Survey design metadata (SurveyMetadata instance from diff_diff.survey)
    survey_metadata: Optional[Any] = field(default=None)
    # Trajectory data for plotting / custom fit metrics
    synthetic_pre_trajectory: Optional[np.ndarray] = field(default=None)
    synthetic_post_trajectory: Optional[np.ndarray] = field(default=None)
    treated_pre_trajectory: Optional[np.ndarray] = field(default=None)
    treated_post_trajectory: Optional[np.ndarray] = field(default=None)
    # Frank-Wolfe time weights as ndarray, needed by sensitivity_to_zeta_omega
    # which holds time weights fixed
    time_weights_array: Optional[np.ndarray] = field(default=None)
    # Per-position unit ID / role arrays parallel to placebo_effects when
    # variance_method == "jackknife". Private — access via get_loo_effects_df().
    _loo_unit_ids: Optional[List[Any]] = field(default=None, repr=False)
    _loo_roles: Optional[List[str]] = field(default=None, repr=False)
    # Snapshot of fit() matrices and IDs for post-hoc re-estimation
    # (in_time_placebo, sensitivity_to_zeta_omega). Private — not part of API.
    _fit_snapshot: Optional[_SyntheticDiDFitSnapshot] = field(default=None, repr=False)

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.p_value)
        return (
            f"SyntheticDiDResults(ATT={self.att:.4f}{sig}, "
            f"SE={self.se:.4f}, "
            f"p={self.p_value:.4f})"
        )

    def __getstate__(self) -> Dict[str, Any]:
        """Exclude the internal fit snapshot from pickling.

        The snapshot retains outcome matrices, unit IDs, and survey weights
        to support post-hoc diagnostics (`in_time_placebo`,
        `sensitivity_to_zeta_omega`). Serialization would otherwise carry
        that panel state to wherever the pickle is sent, which is a privacy
        hazard for survey-weighted or sensitive fits.

        Unpickled results keep the public fields (ATT, weights, trajectories,
        etc.); calling a diagnostic method that needs the snapshot raises a
        ValueError directing the user to re-fit.
        """
        state = self.__dict__.copy()
        state["_fit_snapshot"] = None
        return state

    @property
    def coef_var(self) -> float:
        """Coefficient of variation: SE / |ATT|. NaN when ATT is 0 or SE non-finite."""
        if not (np.isfinite(self.se) and self.se >= 0):
            return np.nan
        if not np.isfinite(self.att) or self.att == 0:
            return np.nan
        return self.se / abs(self.att)

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate a formatted summary of the estimation results.

        Parameters
        ----------
        alpha : float, optional
            Significance level for confidence intervals. Defaults to the
            alpha used during estimation.

        Returns
        -------
        str
            Formatted summary table.
        """
        alpha = alpha or self.alpha
        conf_level = int((1 - alpha) * 100)

        lines = [
            "=" * 75,
            "Synthetic Difference-in-Differences Estimation Results".center(75),
            "=" * 75,
            "",
            f"{'Observations:':<25} {self.n_obs:>10}",
            f"{'Treated:':<25} {self.n_treated:>10}",
            f"{'Control:':<25} {self.n_control:>10}",
            f"{'Pre-treatment periods:':<25} {len(self.pre_periods):>10}",
            f"{'Post-treatment periods:':<25} {len(self.post_periods):>10}",
        ]

        if self.zeta_omega is not None:
            lines.append(f"{'Zeta (unit weights):':<25} {self.zeta_omega:>10.4f}")
        if self.zeta_lambda is not None:
            lines.append(f"{'Zeta (time weights):':<25} {self.zeta_lambda:>10.6f}")
        if self.noise_level is not None:
            lines.append(f"{'Noise level:':<25} {self.noise_level:>10.4f}")

        if self.pre_treatment_fit is not None:
            lines.append(f"{'Pre-treatment fit (RMSE):':<25} {self.pre_treatment_fit:>10.4f}")

        # Variance method info
        lines.append(f"{'Variance method:':<25} {self.variance_method:>10}")
        if self.variance_method == "bootstrap" and self.n_bootstrap is not None:
            lines.append(f"{'Bootstrap replications:':<25} {self.n_bootstrap:>10}")

        # Add survey design info
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            lines.extend(_format_survey_block(sm, 75))

        lines.extend(
            [
                "",
                "-" * 75,
                f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} {'t-stat':>10} {'P>|t|':>10} {'':>5}",
                "-" * 75,
                f"{'ATT':<15} {self.att:>12.4f} {self.se:>12.4f} {self.t_stat:>10.3f} {self.p_value:>10.4f} {self.significance_stars:>5}",
                "-" * 75,
                "",
                f"{conf_level}% Confidence Interval: [{self.conf_int[0]:.4f}, {self.conf_int[1]:.4f}]",
            ]
        )

        cv = self.coef_var
        if np.isfinite(cv):
            lines.append(f"{'CV (SE/|ATT|):':<25} {cv:>10.4f}")

        # Show top unit weights
        if self.unit_weights:
            sorted_weights = sorted(self.unit_weights.items(), key=lambda x: x[1], reverse=True)
            top_n = min(5, len(sorted_weights))
            lines.extend(
                [
                    "",
                    "-" * 75,
                    "Top Unit Weights (Synthetic Control)".center(75),
                    "-" * 75,
                ]
            )
            for unit, weight in sorted_weights[:top_n]:
                if weight > 0.001:  # Only show meaningful weights
                    lines.append(f"  Unit {unit}: {weight:.4f}")

            # Show how many units have non-trivial weight
            n_nonzero = sum(1 for w in self.unit_weights.values() if w > 0.001)
            lines.append(f"  ({n_nonzero} units with weight > 0.001)")

        # Add significance codes
        lines.extend(
            [
                "",
                "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
                "=" * 75,
            ]
        )

        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print the summary to stdout."""
        print(self.summary(alpha))

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing all estimation results.
        """
        result = {
            "att": self.att,
            "se": self.se,
            "t_stat": self.t_stat,
            "p_value": self.p_value,
            "conf_int_lower": self.conf_int[0],
            "conf_int_upper": self.conf_int[1],
            "n_obs": self.n_obs,
            "n_treated": self.n_treated,
            "n_control": self.n_control,
            "n_pre_periods": len(self.pre_periods),
            "n_post_periods": len(self.post_periods),
            "variance_method": self.variance_method,
            "noise_level": self.noise_level,
            "zeta_omega": self.zeta_omega,
            "zeta_lambda": self.zeta_lambda,
            "pre_treatment_fit": self.pre_treatment_fit,
        }
        if self.n_bootstrap is not None:
            result["n_bootstrap"] = self.n_bootstrap
        if self.survey_metadata is not None:
            sm = self.survey_metadata
            result["weight_type"] = sm.weight_type
            result["effective_n"] = sm.effective_n
            result["design_effect"] = sm.design_effect
            result["sum_weights"] = sm.sum_weights
            result["n_strata"] = sm.n_strata
            result["n_psu"] = sm.n_psu
            result["df_survey"] = sm.df_survey
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with estimation results.
        """
        return pd.DataFrame([self.to_dict()])

    def get_unit_weights_df(self) -> pd.DataFrame:
        """
        Get unit weights as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with unit IDs and their weights.
        """
        return pd.DataFrame(
            [{"unit": unit, "weight": weight} for unit, weight in self.unit_weights.items()]
        ).sort_values("weight", ascending=False)

    def get_time_weights_df(self) -> pd.DataFrame:
        """
        Get time weights as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with time periods and their weights.
        """
        return pd.DataFrame(
            [{"period": period, "weight": weight} for period, weight in self.time_weights.items()]
        )

    def get_loo_effects_df(self) -> pd.DataFrame:
        """
        Per-unit leave-one-out ATT from the jackknife variance pass.

        Requires ``variance_method='jackknife'``; raises ValueError otherwise.

        The underlying values come from the jackknife loops in
        ``SyntheticDiD._jackknife_se``: control LOO estimates fill the
        first ``n_control`` positions (in the order of the control units
        seen by fit), then treated LOO estimates fill the next
        ``n_treated`` positions. This method joins those estimates back
        to user-facing unit identities.

        ``att_loo`` is NaN when the fit hit the zero-sum weight guard for
        that unit (survey weights composed to zero once the unit was
        dropped). ``delta_from_full`` propagates NaN in that case.

        Returns
        -------
        pd.DataFrame
            Columns:
                - ``unit`` — user's unit ID
                - ``role`` — ``'control'`` or ``'treated'``
                - ``att_loo`` — ATT with this unit dropped
                - ``delta_from_full`` — ``att_loo - self.att``
            Sorted by ``|delta_from_full|`` descending, NaN rows at the end.
        """
        if self.variance_method != "jackknife":
            raise ValueError(
                "get_loo_effects_df() requires variance_method='jackknife'. "
                f"This result used variance_method='{self.variance_method}'. "
                "Re-fit with SyntheticDiD(variance_method='jackknife') to "
                "obtain per-unit leave-one-out estimates."
            )
        if (
            self._loo_unit_ids is None
            or self._loo_roles is None
            or self.placebo_effects is None
        ):
            raise ValueError(
                "Leave-one-out estimates are unavailable (jackknife returned "
                "NaN or an empty array). See prior warnings from fit() for the "
                "cause (e.g., single treated unit, all weight on one control)."
            )

        att_loo = np.asarray(self.placebo_effects, dtype=float)
        delta = att_loo - self.att
        df = pd.DataFrame(
            {
                "unit": self._loo_unit_ids,
                "role": self._loo_roles,
                "att_loo": att_loo,
                "delta_from_full": delta,
            }
        )
        # Sort by |delta| descending. NaN rows sort to the end so the most
        # influential real units appear first.
        df["_abs_delta"] = df["delta_from_full"].abs()
        df = df.sort_values(
            by="_abs_delta", ascending=False, na_position="last"
        ).drop(columns="_abs_delta")
        df = df.reset_index(drop=True)
        return df

    def in_time_placebo(
        self,
        fake_treatment_periods: Optional[List[Any]] = None,
        zeta_omega_override: Optional[float] = None,
        zeta_lambda_override: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Re-estimate the ATT on shifted fake treatment periods within the
        original pre-treatment window.

        A credible placebo should produce near-zero ATTs at every shifted
        date. Departures from zero signal that whatever the estimator
        picked up at the real treatment date is also present pre-treatment,
        weakening the causal interpretation.

        The post-treatment data is never used — only the pre-window is
        re-sliced. Regularization reuses ``self.zeta_omega`` and
        ``self.zeta_lambda`` from the original fit (R ``synthdid``
        convention), unless overrides are supplied.

        Parameters
        ----------
        fake_treatment_periods : list, optional
            Explicit pre-period values to test. If ``None`` (default),
            sweeps every feasible pre-period — every P in ``pre_periods``
            whose position ``i`` satisfies ``i >= 2`` (so at least 2
            pre-fake periods remain for weight estimation) and
            ``i <= n_pre - 1`` (so at least 1 post-fake period exists).
            Values not in ``pre_periods`` raise ValueError (a value in
            ``post_periods`` is explicitly not a placebo).
        zeta_omega_override : float, optional
            Override ``self.zeta_omega`` for the refit. Default reuses
            the original.
        zeta_lambda_override : float, optional
            Override ``self.zeta_lambda`` for the refit.

        Returns
        -------
        pd.DataFrame
            Columns:
                - ``fake_treatment_period`` — the shifted date
                - ``att`` — placebo ATT (ideally near 0)
                - ``pre_fit_rmse`` — RMSE on the fake pre-window
                - ``n_pre_fake`` — periods before the fake date
                - ``n_post_fake`` — periods from the fake date onward
            NaN is emitted only for dimensional infeasibility. Frank-Wolfe
            does not expose a mid-solver non-convergence signal; inspect
            ``pre_fit_rmse`` for poor refit quality.
        """
        if self._fit_snapshot is None:
            raise ValueError(
                "in_time_placebo() requires the fit snapshot on the results "
                "object. This result appears to have been loaded from "
                "serialization (which excludes the snapshot) or was produced "
                "by an older estimator version. Re-fit to enable."
            )
        from diff_diff.utils import (
            compute_sdid_estimator,
            compute_sdid_unit_weights,
            compute_time_weights,
        )

        snap = self._fit_snapshot
        pre_periods = snap.pre_periods
        n_pre = len(pre_periods)
        zeta_omega = (
            zeta_omega_override if zeta_omega_override is not None else self.zeta_omega
        )
        zeta_lambda = (
            zeta_lambda_override if zeta_lambda_override is not None else self.zeta_lambda
        )
        if zeta_omega is None or zeta_lambda is None:
            raise ValueError(
                "in_time_placebo() needs zeta_omega and zeta_lambda from the "
                "original fit. Expected on the results object but found None."
            )
        noise_level = self.noise_level if self.noise_level is not None else 0.0
        min_decrease = 1e-5 * noise_level if noise_level > 0 else 1e-5

        # Build the list of (fake_period, position) pairs to iterate.
        period_to_idx = {p: i for i, p in enumerate(pre_periods)}
        if fake_treatment_periods is None:
            positions = list(range(2, n_pre))
            fake_list = [(pre_periods[i], i) for i in positions]
        else:
            fake_list = []
            for p in fake_treatment_periods:
                if p in snap.post_periods:
                    raise ValueError(
                        f"fake_treatment_period={p!r} is in post_periods; a real "
                        "treatment date is not a placebo. Choose a value from "
                        "pre_periods."
                    )
                if p not in period_to_idx:
                    raise ValueError(
                        f"fake_treatment_period={p!r} not found in pre_periods "
                        f"({pre_periods!r})."
                    )
                fake_list.append((p, period_to_idx[p]))

        columns = [
            "fake_treatment_period",
            "att",
            "pre_fit_rmse",
            "n_pre_fake",
            "n_post_fake",
        ]
        if not fake_list:
            return pd.DataFrame(columns=columns)

        rows: List[Dict[str, Any]] = []
        for fake_period, i in fake_list:
            n_pre_fake = i
            n_post_fake = n_pre - i
            row: Dict[str, Any] = {
                "fake_treatment_period": fake_period,
                "att": float("nan"),
                "pre_fit_rmse": float("nan"),
                "n_pre_fake": n_pre_fake,
                "n_post_fake": n_post_fake,
            }
            # Dimensional infeasibility: Frank-Wolfe needs >=2 pre-fake
            # periods for unit weights; the estimator needs >=1 post-fake.
            if n_pre_fake < 2 or n_post_fake < 1:
                rows.append(row)
                continue

            Y_pre_c = snap.Y_pre_control[:i, :]
            Y_post_c = snap.Y_pre_control[i:, :]
            Y_pre_t = snap.Y_pre_treated[:i, :]
            Y_post_t = snap.Y_pre_treated[i:, :]

            if snap.w_treated is not None:
                w_t = snap.w_treated
                y_pre_t_mean = np.average(Y_pre_t, axis=1, weights=w_t)
                y_post_t_mean = np.average(Y_post_t, axis=1, weights=w_t)
            else:
                y_pre_t_mean = np.mean(Y_pre_t, axis=1)
                y_post_t_mean = np.mean(Y_post_t, axis=1)

            omega_fake = compute_sdid_unit_weights(
                Y_pre_c,
                y_pre_t_mean,
                zeta_omega=zeta_omega,
                min_decrease=min_decrease,
            )
            lambda_fake = compute_time_weights(
                Y_pre_c,
                Y_post_c,
                zeta_lambda=zeta_lambda,
                min_decrease=min_decrease,
            )

            if snap.w_control is not None:
                omega_eff_fake = omega_fake * snap.w_control
                denom = omega_eff_fake.sum()
                if denom == 0:
                    rows.append(row)
                    continue
                omega_eff_fake = omega_eff_fake / denom
            else:
                omega_eff_fake = omega_fake

            att_fake = compute_sdid_estimator(
                Y_pre_c,
                Y_post_c,
                y_pre_t_mean,
                y_post_t_mean,
                omega_eff_fake,
                lambda_fake,
            )
            synthetic_pre_fake = Y_pre_c @ omega_eff_fake
            pre_fit = float(
                np.sqrt(np.mean((y_pre_t_mean - synthetic_pre_fake) ** 2))
            )
            row["att"] = float(att_fake)
            row["pre_fit_rmse"] = pre_fit
            rows.append(row)

        return pd.DataFrame(rows)

    def sensitivity_to_zeta_omega(
        self,
        zeta_grid: Optional[List[float]] = None,
        multipliers: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
    ) -> pd.DataFrame:
        """
        Re-estimate the ATT across a grid of ``zeta_omega`` values to show
        how sensitive the estimate is to the unit-weight regularization.

        The Frank-Wolfe time weights computed during the original fit are
        held fixed here — this method isolates sensitivity to
        ``zeta_omega`` specifically. ``zeta_lambda`` and the time weights
        are not re-fit.

        Parameters
        ----------
        zeta_grid : list of float, optional
            Absolute ``zeta_omega`` values to evaluate. If ``None``
            (default), uses ``multipliers * self.zeta_omega`` — i.e. a
            5-point grid by default, spanning 16x from the smallest to
            the largest multiplier and symmetric in log space around 1.0.
        multipliers : tuple of float, default ``(0.25, 0.5, 1.0, 2.0, 4.0)``
            Multipliers on ``self.zeta_omega``. Ignored when
            ``zeta_grid`` is supplied.

        Returns
        -------
        pd.DataFrame
            Columns:
                - ``zeta_omega`` — the regularization value evaluated
                - ``att`` — resulting ATT
                - ``pre_fit_rmse`` — RMSE on the original pre-period
                - ``max_unit_weight`` — max element of the composed
                  ``omega_eff`` (sensitivity indicator: close to 1 means
                  near-one-hot solutions; close to ``1/n_control`` means
                  near-uniform)
                - ``effective_n`` — ``1 / sum(omega_eff**2)``

        Notes
        -----
        Extreme ``zeta_omega``: very small values push weights toward
        sparse one-hot solutions (few controls dominate); very large
        values push toward uniform weighting. The ``pre_fit_rmse`` column
        exposes the tradeoff.
        """
        if self._fit_snapshot is None:
            raise ValueError(
                "sensitivity_to_zeta_omega() requires the fit snapshot on the "
                "results object. This result appears to have been loaded from "
                "serialization (which excludes the snapshot) or was produced "
                "by an older estimator version. Re-fit to enable."
            )
        if self.time_weights_array is None:
            raise ValueError(
                "sensitivity_to_zeta_omega() needs the original time weights "
                "array. Expected on the results object but found None. Re-fit "
                "to populate."
            )
        from diff_diff.utils import compute_sdid_estimator, compute_sdid_unit_weights

        snap = self._fit_snapshot
        if zeta_grid is None:
            if self.zeta_omega is None:
                raise ValueError(
                    "Cannot build default zeta_grid: self.zeta_omega is None. "
                    "Supply zeta_grid explicitly."
                )
            zeta_values: List[float] = [float(m * self.zeta_omega) for m in multipliers]
        else:
            zeta_values = [float(z) for z in zeta_grid]

        noise_level = self.noise_level if self.noise_level is not None else 0.0
        min_decrease = 1e-5 * noise_level if noise_level > 0 else 1e-5

        if snap.w_treated is not None:
            y_pre_t_mean = np.average(
                snap.Y_pre_treated, axis=1, weights=snap.w_treated
            )
            y_post_t_mean = np.average(
                snap.Y_post_treated, axis=1, weights=snap.w_treated
            )
        else:
            y_pre_t_mean = np.mean(snap.Y_pre_treated, axis=1)
            y_post_t_mean = np.mean(snap.Y_post_treated, axis=1)

        columns = [
            "zeta_omega",
            "att",
            "pre_fit_rmse",
            "max_unit_weight",
            "effective_n",
        ]
        if not zeta_values:
            return pd.DataFrame(columns=columns)

        time_weights = np.asarray(self.time_weights_array, dtype=float)
        rows: List[Dict[str, Any]] = []
        for z in zeta_values:
            omega_fake = compute_sdid_unit_weights(
                snap.Y_pre_control,
                y_pre_t_mean,
                zeta_omega=z,
                min_decrease=min_decrease,
            )
            if snap.w_control is not None:
                omega_eff = omega_fake * snap.w_control
                denom = omega_eff.sum()
                if denom == 0:
                    rows.append(
                        {
                            "zeta_omega": z,
                            "att": float("nan"),
                            "pre_fit_rmse": float("nan"),
                            "max_unit_weight": float("nan"),
                            "effective_n": float("nan"),
                        }
                    )
                    continue
                omega_eff = omega_eff / denom
            else:
                omega_eff = omega_fake

            att = compute_sdid_estimator(
                snap.Y_pre_control,
                snap.Y_post_control,
                y_pre_t_mean,
                y_post_t_mean,
                omega_eff,
                time_weights,
            )
            synthetic_pre = snap.Y_pre_control @ omega_eff
            pre_fit = float(np.sqrt(np.mean((y_pre_t_mean - synthetic_pre) ** 2)))
            herf = float(np.sum(omega_eff ** 2))
            rows.append(
                {
                    "zeta_omega": z,
                    "att": float(att),
                    "pre_fit_rmse": pre_fit,
                    "max_unit_weight": float(np.max(omega_eff)),
                    "effective_n": float("nan") if herf == 0 else 1.0 / herf,
                }
            )
        return pd.DataFrame(rows, columns=columns)

    def get_weight_concentration(self, top_k: int = 5) -> Dict[str, Any]:
        """
        Concentration metrics for the control unit weights.

        Operates on ``self.unit_weights``, which for survey-weighted fits
        stores the composed effective weights
        (``omega_eff = raw_omega * w_control``, renormalized to sum to 1)
        that were applied to produce the ATT. For non-survey fits the
        values equal the raw Frank-Wolfe solution. Either way, the
        concentration reflects the distribution actually used by the
        estimator.

        Parameters
        ----------
        top_k : int, default 5
            Number of largest weights to sum for ``top_k_share``. Must be
            non-negative. Clamped to the available number of control units.

        Returns
        -------
        dict
            Keys:
                - ``effective_n`` — ``1 / sum(w**2)``, inverse Herfindahl
                - ``herfindahl`` — ``sum(w**2)``
                - ``top_k_share`` — sum of the ``top_k`` largest weights
                - ``top_k`` — the (possibly clamped) value used

        Raises
        ------
        ValueError
            If ``top_k`` is negative.
        """
        if top_k < 0:
            raise ValueError(
                f"top_k must be non-negative (got {top_k})."
            )
        weights = np.asarray(list(self.unit_weights.values()), dtype=float)
        if weights.size == 0:
            return {
                "effective_n": float("nan"),
                "herfindahl": float("nan"),
                "top_k_share": float("nan"),
                "top_k": 0,
            }
        herfindahl = float(np.sum(weights ** 2))
        effective_n = float("nan") if herfindahl == 0 else 1.0 / herfindahl
        k = min(int(top_k), weights.size)
        if k <= 0:
            top_k_share = 0.0
        else:
            top_k_share = float(np.sort(weights)[-k:].sum())
        return {
            "effective_n": effective_n,
            "herfindahl": herfindahl,
            "top_k_share": top_k_share,
            "top_k": k,
        }

    @property
    def is_significant(self) -> bool:
        """Check if the ATT is statistically significant at the alpha level."""
        return bool(self.p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Return significance stars based on p-value."""
        return _get_significance_stars(self.p_value)
