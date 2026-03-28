"""
Staggered Difference-in-Differences estimators.

Implements modern methods for DiD with variation in treatment timing,
including the Callaway-Sant'Anna (2021) estimator.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import linalg as scipy_linalg

from diff_diff.linalg import (
    _check_propensity_diagnostics,
    _detect_rank_deficiency,
    _format_dropped_columns,
    solve_logit,
    solve_ols,
)
from diff_diff.staggered_aggregation import (
    CallawaySantAnnaAggregationMixin,
)
from diff_diff.staggered_bootstrap import (
    CallawaySantAnnaBootstrapMixin,
    CSBootstrapResults,
)

# Import from split modules
from diff_diff.staggered_results import (
    CallawaySantAnnaResults,
    GroupTimeEffect,
)
from diff_diff.utils import safe_inference, safe_inference_batch

# Re-export for backward compatibility
__all__ = [
    "CallawaySantAnna",
    "CallawaySantAnnaResults",
    "CSBootstrapResults",
    "GroupTimeEffect",
]

# Type alias for pre-computed structures
PrecomputedData = Dict[str, Any]


def _linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    rank_deficient_action: str = "warn",
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit OLS regression.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features). Intercept added automatically.
    y : np.ndarray
        Outcome variable.
    rank_deficient_action : str, default "warn"
        Action when design matrix is rank-deficient:
        - "warn": Issue warning and drop linearly dependent columns (default)
        - "error": Raise ValueError
        - "silent": Drop columns silently without warning
    weights : np.ndarray, optional
        Observation weights for WLS. When None, OLS is used.

    Returns
    -------
    beta : np.ndarray
        Fitted coefficients (including intercept).
    residuals : np.ndarray
        Residuals from the fit.
    """
    n = X.shape[0]
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(n), X])

    # Use unified OLS backend (no vcov needed)
    beta, residuals, _ = solve_ols(
        X_with_intercept,
        y,
        return_vcov=False,
        rank_deficient_action=rank_deficient_action,
        weights=weights,
    )

    return beta, residuals


class CallawaySantAnna(
    CallawaySantAnnaBootstrapMixin,
    CallawaySantAnnaAggregationMixin,
):
    """
    Callaway-Sant'Anna (2021) estimator for staggered Difference-in-Differences.

    This estimator handles DiD designs with variation in treatment timing
    (staggered adoption) and heterogeneous treatment effects. It avoids the
    bias of traditional two-way fixed effects (TWFE) estimators by:

    1. Computing group-time average treatment effects ATT(g,t) for each
       cohort g (units first treated in period g) and time t.
    2. Aggregating these to summary measures (overall ATT, event study, etc.)
       using appropriate weights.

    Parameters
    ----------
    control_group : str, default="never_treated"
        Which units to use as controls:
        - "never_treated": Use only never-treated units (recommended)
        - "not_yet_treated": Use never-treated and not-yet-treated units
    anticipation : int, default=0
        Number of periods before treatment where effects may occur.
        Set to > 0 if treatment effects can begin before the official
        treatment date.
    estimation_method : str, default="dr"
        Estimation method:
        - "dr": Doubly robust (recommended)
        - "ipw": Inverse probability weighting
        - "reg": Outcome regression
    alpha : float, default=0.05
        Significance level for confidence intervals.
    cluster : str, optional
        Column name for cluster-robust standard errors.
        Defaults to unit-level clustering.
    n_bootstrap : int, default=0
        Number of bootstrap iterations for inference.
        If 0, uses analytical standard errors.
        Recommended: 999 or more for reliable inference.

        .. note:: Memory Usage
            The bootstrap stores all weights in memory as a (n_bootstrap, n_units)
            float64 array. For large datasets, this can be significant:
            - 1K bootstrap × 10K units = ~80 MB
            - 10K bootstrap × 100K units = ~8 GB
            Consider reducing n_bootstrap if memory is constrained.

    bootstrap_weights : str, default="rademacher"
        Type of weights for multiplier bootstrap:
        - "rademacher": +1/-1 with equal probability (standard choice)
        - "mammen": Two-point distribution (asymptotically valid, matches skewness)
        - "webb": Six-point distribution (recommended when n_clusters < 20)
    bootstrap_weight_type : str, optional
        .. deprecated:: 1.0.1
            Use ``bootstrap_weights`` instead. Will be removed in v3.0.
    seed : int, optional
        Random seed for reproducibility.
    rank_deficient_action : str, default="warn"
        Action when design matrix is rank-deficient (linearly dependent columns):
        - "warn": Issue warning and drop linearly dependent columns (default)
        - "error": Raise ValueError
        - "silent": Drop columns silently without warning
    base_period : str, default="varying"
        Method for selecting the base (reference) period for computing
        ATT(g,t). Options:
        - "varying": For pre-treatment periods (t < g - anticipation), use
          t-1 as base (consecutive comparisons). For post-treatment, use
          g-1-anticipation. Requires t-1 to exist in data.
        - "universal": Always use g-1-anticipation as base period.
        Both produce identical post-treatment effects. Matches R's
        did::att_gt() base_period parameter.
    cband : bool, default=True
        Whether to compute simultaneous confidence bands (sup-t) for
        event study aggregation. Requires ``n_bootstrap > 0``.
        When True, results include ``cband_crit_value`` and per-event-time
        ``cband_conf_int`` entries controlling family-wise error rate.
    pscore_trim : float, default=0.01
        Trimming bound for propensity scores. Scores are clipped to
        ``[pscore_trim, 1 - pscore_trim]`` before weight computation
        in IPW and DR estimation. Must be in ``(0, 0.5)``.

    Attributes
    ----------
    results_ : CallawaySantAnnaResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage:

    >>> import pandas as pd
    >>> from diff_diff import CallawaySantAnna
    >>>
    >>> # Panel data with staggered treatment
    >>> # 'first_treat' = period when unit was first treated (0 if never treated)
    >>> data = pd.DataFrame({
    ...     'unit': [...],
    ...     'time': [...],
    ...     'outcome': [...],
    ...     'first_treat': [...]  # 0 for never-treated, else first treatment period
    ... })
    >>>
    >>> cs = CallawaySantAnna()
    >>> results = cs.fit(data, outcome='outcome', unit='unit',
    ...                  time='time', first_treat='first_treat')
    >>>
    >>> results.print_summary()

    With event study aggregation:

    >>> cs = CallawaySantAnna()
    >>> results = cs.fit(data, outcome='outcome', unit='unit',
    ...                  time='time', first_treat='first_treat',
    ...                  aggregate='event_study')
    >>>
    >>> # Plot event study
    >>> from diff_diff import plot_event_study
    >>> plot_event_study(results)

    With covariate adjustment (conditional parallel trends):

    >>> # When parallel trends only holds conditional on covariates
    >>> cs = CallawaySantAnna(estimation_method='dr')  # doubly robust
    >>> results = cs.fit(data, outcome='outcome', unit='unit',
    ...                  time='time', first_treat='first_treat',
    ...                  covariates=['age', 'income'])
    >>>
    >>> # DR is recommended: consistent if either outcome model
    >>> # or propensity model is correctly specified

    Notes
    -----
    The key innovation of Callaway & Sant'Anna (2021) is the disaggregated
    approach: instead of estimating a single treatment effect, they estimate
    ATT(g,t) for each cohort-time pair. This avoids the "forbidden comparison"
    problem where already-treated units act as controls.

    The ATT(g,t) is identified under parallel trends conditional on covariates:

        E[Y(0)_t - Y(0)_g-1 | G=g] = E[Y(0)_t - Y(0)_g-1 | C=1]

    where G=g indicates treatment cohort g and C=1 indicates control units.
    This uses g-1 as the base period, which applies to post-treatment (t >= g).
    With base_period="varying" (default), pre-treatment uses t-1 as base for
    consecutive comparisons useful in parallel trends diagnostics.

    References
    ----------
    Callaway, B., & Sant'Anna, P. H. (2021). Difference-in-Differences with
    multiple time periods. Journal of Econometrics, 225(2), 200-230.
    """

    def __init__(
        self,
        control_group: str = "never_treated",
        anticipation: int = 0,
        estimation_method: str = "dr",
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        n_bootstrap: int = 0,
        bootstrap_weights: Optional[str] = None,
        bootstrap_weight_type: Optional[str] = None,
        seed: Optional[int] = None,
        rank_deficient_action: str = "warn",
        base_period: str = "varying",
        cband: bool = True,
        pscore_trim: float = 0.01,
    ):
        import warnings

        if control_group not in ["never_treated", "not_yet_treated"]:
            raise ValueError(
                f"control_group must be 'never_treated' or 'not_yet_treated', "
                f"got '{control_group}'"
            )
        if estimation_method not in ["dr", "ipw", "reg"]:
            raise ValueError(
                f"estimation_method must be 'dr', 'ipw', or 'reg', " f"got '{estimation_method}'"
            )
        if not (0 < pscore_trim < 0.5):
            raise ValueError(f"pscore_trim must be in (0, 0.5), got {pscore_trim}")

        # Handle bootstrap_weight_type deprecation
        if bootstrap_weight_type is not None:
            warnings.warn(
                "bootstrap_weight_type is deprecated and will be removed in v3.0. "
                "Use bootstrap_weights instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if bootstrap_weights is None:
                bootstrap_weights = bootstrap_weight_type

        # Default to rademacher if neither specified
        if bootstrap_weights is None:
            bootstrap_weights = "rademacher"

        if bootstrap_weights not in ["rademacher", "mammen", "webb"]:
            raise ValueError(
                f"bootstrap_weights must be 'rademacher', 'mammen', or 'webb', "
                f"got '{bootstrap_weights}'"
            )

        if rank_deficient_action not in ["warn", "error", "silent"]:
            raise ValueError(
                f"rank_deficient_action must be 'warn', 'error', or 'silent', "
                f"got '{rank_deficient_action}'"
            )

        if base_period not in ["varying", "universal"]:
            raise ValueError(
                f"base_period must be 'varying' or 'universal', " f"got '{base_period}'"
            )

        self.control_group = control_group
        self.anticipation = anticipation
        self.estimation_method = estimation_method
        self.alpha = alpha
        self.cluster = cluster
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        # Keep bootstrap_weight_type for backward compatibility
        self.bootstrap_weight_type = bootstrap_weights
        self.seed = seed
        self.rank_deficient_action = rank_deficient_action
        self.base_period = base_period

        self.cband = cband
        self.pscore_trim = pscore_trim

        self.is_fitted_ = False
        self.results_: Optional[CallawaySantAnnaResults] = None

    @staticmethod
    def _collapse_survey_to_unit_level(resolved_survey, df, unit_col, all_units):
        """Create unit-level ResolvedSurveyDesign for panel IF-based variance.

        Survey design columns are constant within units (validated upstream).
        This extracts one row per unit, aligned to ``all_units`` ordering.
        """
        from diff_diff.survey import ResolvedSurveyDesign

        n_units = len(all_units)
        # Use groupby().first() to get one value per unit, then reindex
        unit_groups = df.groupby(unit_col)

        weights_unit = (
            pd.Series(resolved_survey.weights, index=df.index)
            .groupby(df[unit_col])
            .first()
            .reindex(all_units)
            .values
        )

        strata_unit = None
        if resolved_survey.strata is not None:
            strata_unit = (
                pd.Series(resolved_survey.strata, index=df.index)
                .groupby(df[unit_col])
                .first()
                .reindex(all_units)
                .values
            )

        psu_unit = None
        if resolved_survey.psu is not None:
            psu_unit = (
                pd.Series(resolved_survey.psu, index=df.index)
                .groupby(df[unit_col])
                .first()
                .reindex(all_units)
                .values
            )

        fpc_unit = None
        if resolved_survey.fpc is not None:
            fpc_unit = (
                pd.Series(resolved_survey.fpc, index=df.index)
                .groupby(df[unit_col])
                .first()
                .reindex(all_units)
                .values
            )

        # Collapse replicate weights to unit level (same groupby pattern)
        rep_weights_unit = None
        if resolved_survey.replicate_weights is not None:
            R = resolved_survey.replicate_weights.shape[1]
            rep_weights_unit = np.zeros((n_units, R))
            for r in range(R):
                rep_weights_unit[:, r] = (
                    pd.Series(resolved_survey.replicate_weights[:, r], index=df.index)
                    .groupby(df[unit_col])
                    .first()
                    .reindex(all_units)
                    .values
                )

        return ResolvedSurveyDesign(
            weights=weights_unit.astype(np.float64),
            weight_type=resolved_survey.weight_type,
            strata=strata_unit,
            psu=psu_unit,
            fpc=fpc_unit,
            n_strata=resolved_survey.n_strata,
            n_psu=resolved_survey.n_psu,
            lonely_psu=resolved_survey.lonely_psu,
            replicate_weights=rep_weights_unit,
            replicate_method=resolved_survey.replicate_method,
            fay_rho=resolved_survey.fay_rho,
            n_replicates=resolved_survey.n_replicates,
            replicate_strata=resolved_survey.replicate_strata,
            combined_weights=resolved_survey.combined_weights,
            replicate_scale=resolved_survey.replicate_scale,
            replicate_rscales=resolved_survey.replicate_rscales,
            mse=resolved_survey.mse,
        )

    def _precompute_structures(
        self,
        df: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]],
        time_periods: List[Any],
        treatment_groups: List[Any],
        resolved_survey=None,
    ) -> PrecomputedData:
        """
        Pre-compute data structures for efficient ATT(g,t) computation.

        This pivots data to wide format and pre-computes:
        - Outcome matrix (units x time periods)
        - Covariate matrix (units x covariates) from base period
        - Unit cohort membership masks
        - Control unit masks

        Returns
        -------
        PrecomputedData
            Dictionary with pre-computed structures.
        """
        # Get unique units and their cohort assignments
        unit_info = df.groupby(unit)[first_treat].first()
        all_units = unit_info.index.values
        unit_cohorts = unit_info.values

        # Create unit index mapping for fast lookups
        unit_to_idx = {u: i for i, u in enumerate(all_units)}

        # Pivot outcome to wide format: rows = units, columns = time periods
        outcome_wide = df.pivot(index=unit, columns=time, values=outcome)
        # Reindex to ensure all units are present (handles unbalanced panels)
        outcome_wide = outcome_wide.reindex(all_units)
        outcome_matrix = outcome_wide.values  # Shape: (n_units, n_periods)
        period_to_col = {t: i for i, t in enumerate(outcome_wide.columns)}

        # Pre-compute cohort masks (boolean arrays)
        cohort_masks = {}
        for g in treatment_groups:
            cohort_masks[g] = unit_cohorts == g

        # Never-treated mask
        # np.inf was normalized to 0 in fit(), so the np.inf check is defensive only
        never_treated_mask = (unit_cohorts == 0) | (unit_cohorts == np.inf)

        # Pre-compute covariate matrices by time period if needed
        # (covariates are retrieved from the base period of each comparison)
        covariate_by_period = None
        if covariates:
            covariate_by_period = {}
            for t in time_periods:
                period_data = df[df[time] == t].set_index(unit)
                period_cov = period_data.reindex(all_units)[covariates]
                covariate_by_period[t] = period_cov.values  # Shape: (n_units, n_covariates)

        is_balanced = not np.any(np.isnan(outcome_matrix))

        # Extract per-unit survey weights (one weight per unit)
        if resolved_survey is not None:
            sw_by_unit = (
                pd.Series(resolved_survey.weights, index=df.index).groupby(df[unit]).first()
            )
            survey_weights_arr = sw_by_unit.reindex(all_units).values
        else:
            survey_weights_arr = None

        resolved_survey_unit = (
            self._collapse_survey_to_unit_level(resolved_survey, df, unit, all_units)
            if resolved_survey is not None
            else None
        )

        return {
            "all_units": all_units,
            "unit_to_idx": unit_to_idx,
            "unit_cohorts": unit_cohorts,
            "outcome_matrix": outcome_matrix,
            "period_to_col": period_to_col,
            "cohort_masks": cohort_masks,
            "never_treated_mask": never_treated_mask,
            "covariate_by_period": covariate_by_period,
            "time_periods": time_periods,
            "is_balanced": is_balanced,
            "survey_weights": survey_weights_arr,
            "resolved_survey": resolved_survey,
            "resolved_survey_unit": resolved_survey_unit,
            "df_survey": (
                resolved_survey_unit.df_survey if resolved_survey_unit is not None else None
            ),
        }

    def _compute_att_gt_fast(
        self,
        precomputed: PrecomputedData,
        g: Any,
        t: Any,
        covariates: Optional[List[str]],
        pscore_cache: Optional[Dict] = None,
        cho_cache: Optional[Dict] = None,
    ) -> Tuple[Optional[float], float, int, int, Optional[Dict[str, Any]], Optional[float]]:
        """
        Compute ATT(g,t) using pre-computed data structures (fast version).

        Uses vectorized numpy operations on pre-pivoted outcome matrix
        instead of repeated pandas filtering.

        Returns
        -------
        att_gt : float or None
        se_gt : float
        n_treated : int
        n_control : int
        inf_func_info : dict or None
        survey_weight_sum : float or None
            Sum of survey weights for treated units (for aggregation weighting).
        """
        period_to_col = precomputed["period_to_col"]
        outcome_matrix = precomputed["outcome_matrix"]
        cohort_masks = precomputed["cohort_masks"]
        never_treated_mask = precomputed["never_treated_mask"]
        unit_cohorts = precomputed["unit_cohorts"]
        covariate_by_period = precomputed["covariate_by_period"]

        # Base period selection based on mode
        if self.base_period == "universal":
            # Universal: always use g - 1 - anticipation
            base_period_val = g - 1 - self.anticipation
        else:  # varying
            if t < g - self.anticipation:
                # Pre-treatment: use t - 1 (consecutive comparison)
                base_period_val = t - 1
            else:
                # Post-treatment: use g - 1 - anticipation
                base_period_val = g - 1 - self.anticipation

        if base_period_val not in period_to_col:
            # Base period must exist; no fallback to maintain methodological consistency
            return None, 0.0, 0, 0, None, None

        # Check if periods exist in the data
        if base_period_val not in period_to_col or t not in period_to_col:
            return None, 0.0, 0, 0, None, None

        base_col = period_to_col[base_period_val]
        post_col = period_to_col[t]

        # Get treated units mask (cohort g)
        treated_mask = cohort_masks[g]

        # Get control units mask
        if self.control_group == "never_treated":
            control_mask = never_treated_mask
        else:  # not_yet_treated
            # Not yet treated at BOTH time t and the base period:
            # Controls must be untreated at whichever is later, otherwise
            # their outcome at the base period is contaminated by treatment.
            nyt_threshold = max(t, base_period_val) + self.anticipation
            control_mask = never_treated_mask | (
                (unit_cohorts > nyt_threshold) & (unit_cohorts != g)
            )

        # Extract outcomes for base and post periods
        y_base = outcome_matrix[:, base_col]
        y_post = outcome_matrix[:, post_col]

        # Compute outcome changes (vectorized)
        outcome_change = y_post - y_base

        # Filter to units with valid data (no NaN in either period)
        valid_mask = ~(np.isnan(y_base) | np.isnan(y_post))

        # Get treated and control with valid data
        treated_valid = treated_mask & valid_mask
        control_valid = control_mask & valid_mask

        n_treated = np.sum(treated_valid)
        n_control = np.sum(control_valid)

        if n_treated == 0 or n_control == 0:
            return None, 0.0, 0, 0, None, None

        # Extract outcome changes for treated and control
        treated_change = outcome_change[treated_valid]
        control_change = outcome_change[control_valid]

        # Extract survey weights for treated and control
        survey_w = precomputed.get("survey_weights")
        sw_treated = survey_w[treated_valid] if survey_w is not None else None
        sw_control = survey_w[control_valid] if survey_w is not None else None

        # Guard against zero effective mass after subpopulation filtering
        if sw_treated is not None and np.sum(sw_treated) <= 0:
            return np.nan, np.nan, 0, 0, None, None
        if sw_control is not None and np.sum(sw_control) <= 0:
            return np.nan, np.nan, 0, 0, None, None

        # Get covariates if specified (from the base period)
        X_treated = None
        X_control = None
        if covariates and covariate_by_period is not None:
            cov_matrix = covariate_by_period[base_period_val]
            X_treated = cov_matrix[treated_valid]
            X_control = cov_matrix[control_valid]

            # Check for missing values
            if np.any(np.isnan(X_treated)) or np.any(np.isnan(X_control)):
                warnings.warn(
                    f"Missing values in covariates for group {g}, time {t}. "
                    "Falling back to unconditional estimation.",
                    UserWarning,
                    stacklevel=3,
                )
                X_treated = None
                X_control = None

        # Compute cache key for propensity score reuse
        pscore_key = None
        if pscore_cache is not None and X_treated is not None:
            is_balanced = precomputed.get("is_balanced", False)
            if is_balanced and self.control_group == "never_treated":
                pscore_key = (g, base_period_val)
            else:
                pscore_key = (g, base_period_val, t)

        # Compute cache key for Cholesky reuse (DR outcome regression)
        cho_key = None
        if cho_cache is not None and X_control is not None:
            is_balanced = precomputed.get("is_balanced", False)
            if is_balanced and self.control_group == "never_treated":
                cho_key = base_period_val
            else:
                cho_key = (g, base_period_val, t)

        # Estimation method
        if self.estimation_method == "reg":
            att_gt, se_gt, inf_func = self._outcome_regression(
                treated_change,
                control_change,
                X_treated,
                X_control,
                sw_treated=sw_treated,
                sw_control=sw_control,
            )
        elif self.estimation_method == "ipw":
            sw_all = np.concatenate([sw_treated, sw_control]) if sw_treated is not None else None
            att_gt, se_gt, inf_func = self._ipw_estimation(
                treated_change,
                control_change,
                int(n_treated),
                int(n_control),
                X_treated,
                X_control,
                pscore_cache=pscore_cache,
                pscore_key=pscore_key,
                sw_treated=sw_treated,
                sw_control=sw_control,
                sw_all=sw_all,
            )
        else:  # doubly robust
            sw_all = np.concatenate([sw_treated, sw_control]) if sw_treated is not None else None
            att_gt, se_gt, inf_func = self._doubly_robust(
                treated_change,
                control_change,
                X_treated,
                X_control,
                pscore_cache=pscore_cache,
                pscore_key=pscore_key,
                cho_cache=cho_cache,
                cho_key=cho_key,
                sw_treated=sw_treated,
                sw_control=sw_control,
                sw_all=sw_all,
            )

        # Package influence function info with index arrays (positions into
        # precomputed['all_units']) for O(1) downstream lookups instead of
        # O(n) Python dict lookups.
        n_t = int(n_treated)
        all_units = precomputed["all_units"]
        treated_positions = np.where(treated_valid)[0]
        control_positions = np.where(control_valid)[0]
        inf_func_info = {
            "treated_idx": treated_positions,
            "control_idx": control_positions,
            "treated_units": all_units[treated_positions],
            "control_units": all_units[control_positions],
            "treated_inf": inf_func[:n_t],
            "control_inf": inf_func[n_t:],
        }

        sw_sum = float(np.sum(sw_treated)) if sw_treated is not None else None
        return att_gt, se_gt, int(n_treated), int(n_control), inf_func_info, sw_sum

    def _compute_all_att_gt_vectorized(
        self,
        precomputed: PrecomputedData,
        treatment_groups: List[Any],
        time_periods: List[Any],
        min_period: Any,
    ) -> Tuple[Dict, Dict]:
        """
        Vectorized computation of all ATT(g,t) for the no-covariates regression case.

        This inlines the simple difference-in-means path from _outcome_regression()
        and eliminates per-(g,t) Python function call overhead.

        Returns
        -------
        group_time_effects : dict
            Mapping (g, t) -> effect dict.
        influence_func_info : dict
            Mapping (g, t) -> influence function info dict.
        """
        period_to_col = precomputed["period_to_col"]
        outcome_matrix = precomputed["outcome_matrix"]
        cohort_masks = precomputed["cohort_masks"]
        never_treated_mask = precomputed["never_treated_mask"]
        unit_cohorts = precomputed["unit_cohorts"]
        survey_w = precomputed.get("survey_weights")

        group_time_effects = {}
        influence_func_info = {}

        # Collect all valid (g, t, base_col, post_col) tuples
        tasks = []
        for g in treatment_groups:
            if self.base_period == "universal":
                universal_base = g - 1 - self.anticipation
                valid_periods = [t for t in time_periods if t != universal_base]
            else:
                valid_periods = [
                    t for t in time_periods if t >= g - self.anticipation or t > min_period
                ]

            for t in valid_periods:
                # Base period selection
                if self.base_period == "universal":
                    base_period_val = g - 1 - self.anticipation
                else:
                    if t < g - self.anticipation:
                        base_period_val = t - 1
                    else:
                        base_period_val = g - 1 - self.anticipation

                if base_period_val not in period_to_col or t not in period_to_col:
                    continue

                tasks.append(
                    (g, t, period_to_col[base_period_val], period_to_col[t], base_period_val)
                )

        # Process all tasks
        atts = []
        ses = []
        task_keys = []

        for g, t, base_col, post_col, base_period_val in tasks:
            treated_mask = cohort_masks[g]

            if self.control_group == "never_treated":
                control_mask = never_treated_mask
            else:
                # Controls must be untreated at both t and base_period_val
                nyt_threshold = max(t, base_period_val) + self.anticipation
                control_mask = never_treated_mask | (
                    (unit_cohorts > nyt_threshold) & (unit_cohorts != g)
                )

            y_base = outcome_matrix[:, base_col]
            y_post = outcome_matrix[:, post_col]
            outcome_change = y_post - y_base
            valid_mask = ~(np.isnan(y_base) | np.isnan(y_post))

            treated_valid = treated_mask & valid_mask
            control_valid = control_mask & valid_mask

            n_treated = np.sum(treated_valid)
            n_control = np.sum(control_valid)

            if n_treated == 0 or n_control == 0:
                continue

            treated_change = outcome_change[treated_valid]
            control_change = outcome_change[control_valid]

            n_t = int(n_treated)
            n_c = int(n_control)

            # Inline no-covariates regression (difference in means)
            if survey_w is not None:
                sw_t = survey_w[treated_valid]
                sw_c = survey_w[control_valid]
                # Guard against zero effective mass
                if np.sum(sw_t) <= 0 or np.sum(sw_c) <= 0:
                    continue
                sw_t_norm = sw_t / np.sum(sw_t)
                sw_c_norm = sw_c / np.sum(sw_c)
                mu_t = float(np.sum(sw_t_norm * treated_change))
                mu_c = float(np.sum(sw_c_norm * control_change))
                att = mu_t - mu_c

                # Influence function (survey-weighted)
                inf_treated = sw_t_norm * (treated_change - mu_t)
                inf_control = -sw_c_norm * (control_change - mu_c)
                # SE derived from IF: sum(IF_i^2)
                se = (
                    float(np.sqrt(np.sum(inf_treated**2) + np.sum(inf_control**2)))
                    if (n_t > 0 and n_c > 0)
                    else 0.0
                )
                sw_sum = float(np.sum(sw_t))
            else:
                att = float(np.mean(treated_change) - np.mean(control_change))

                var_t = float(np.var(treated_change, ddof=1)) if n_t > 1 else 0.0
                var_c = float(np.var(control_change, ddof=1)) if n_c > 1 else 0.0
                se = float(np.sqrt(var_t / n_t + var_c / n_c)) if (n_t > 0 and n_c > 0) else 0.0

                # Influence function
                inf_treated = (treated_change - np.mean(treated_change)) / n_t
                inf_control = -(control_change - np.mean(control_change)) / n_c
                sw_sum = None

            gte_entry = {
                "effect": att,
                "se": se,
                # t_stat, p_value, conf_int filled by batch inference below
                "t_stat": np.nan,
                "p_value": np.nan,
                "conf_int": (np.nan, np.nan),
                "n_treated": n_t,
                "n_control": n_c,
            }
            if sw_sum is not None:
                gte_entry["survey_weight_sum"] = sw_sum
            group_time_effects[(g, t)] = gte_entry

            all_units = precomputed["all_units"]
            treated_positions = np.where(treated_valid)[0]
            control_positions = np.where(control_valid)[0]
            influence_func_info[(g, t)] = {
                "treated_idx": treated_positions,
                "control_idx": control_positions,
                "treated_units": all_units[treated_positions],
                "control_units": all_units[control_positions],
                "treated_inf": inf_treated,
                "control_inf": inf_control,
            }

            atts.append(att)
            ses.append(se)
            task_keys.append((g, t))

        # Batch inference for all (g,t) pairs at once
        if task_keys:
            df_survey_val = precomputed.get("df_survey")
            # Guard: replicate design with undefined df → NaN inference
            if (df_survey_val is None
                    and precomputed.get("resolved_survey_unit") is not None
                    and hasattr(precomputed["resolved_survey_unit"], 'uses_replicate_variance')
                    and precomputed["resolved_survey_unit"].uses_replicate_variance):
                df_survey_val = 0
            t_stats, p_values, ci_lowers, ci_uppers = safe_inference_batch(
                np.array(atts),
                np.array(ses),
                alpha=self.alpha,
                df=df_survey_val,
            )
            for idx, key in enumerate(task_keys):
                group_time_effects[key]["t_stat"] = float(t_stats[idx])
                group_time_effects[key]["p_value"] = float(p_values[idx])
                group_time_effects[key]["conf_int"] = (float(ci_lowers[idx]), float(ci_uppers[idx]))

        return group_time_effects, influence_func_info

    def _compute_all_att_gt_covariate_reg(
        self,
        precomputed: PrecomputedData,
        treatment_groups: List[Any],
        time_periods: List[Any],
        min_period: Any,
    ) -> Tuple[Dict, Dict]:
        """
        Optimized computation of all ATT(g,t) for the covariate regression case.

        Groups (g,t) pairs by their control regression key to reuse Cholesky
        factorizations of X^T X across pairs that share the same control design
        matrix.

        Returns
        -------
        group_time_effects : dict
            Mapping (g, t) -> effect dict.
        influence_func_info : dict
            Mapping (g, t) -> influence function info dict.
        """
        period_to_col = precomputed["period_to_col"]
        outcome_matrix = precomputed["outcome_matrix"]
        cohort_masks = precomputed["cohort_masks"]
        never_treated_mask = precomputed["never_treated_mask"]
        unit_cohorts = precomputed["unit_cohorts"]
        covariate_by_period = precomputed["covariate_by_period"]
        is_balanced = precomputed["is_balanced"]

        group_time_effects = {}
        influence_func_info = {}
        atts = []
        ses = []
        task_keys = []
        n_nan_cells = 0

        # Collect all valid (g, t) tasks with their base periods
        tasks_by_group = {}  # control_key -> list of (g, t, base_period_val, base_col, post_col)
        for g in treatment_groups:
            if self.base_period == "universal":
                universal_base = g - 1 - self.anticipation
                valid_periods = [t for t in time_periods if t != universal_base]
            else:
                valid_periods = [
                    t for t in time_periods if t >= g - self.anticipation or t > min_period
                ]

            for t in valid_periods:
                if self.base_period == "universal":
                    base_period_val = g - 1 - self.anticipation
                else:
                    if t < g - self.anticipation:
                        base_period_val = t - 1
                    else:
                        base_period_val = g - 1 - self.anticipation

                if base_period_val not in period_to_col or t not in period_to_col:
                    continue

                # Determine control regression grouping key.
                # For balanced panels with never_treated control, X_control depends
                # only on base_period_val (control mask is time-invariant).
                # For not_yet_treated, the control mask excludes cohort g, so include g.
                if is_balanced and self.control_group == "never_treated":
                    control_key = base_period_val
                else:
                    control_key = (g, base_period_val, t)

                tasks_by_group.setdefault(control_key, []).append(
                    (g, t, base_period_val, period_to_col[base_period_val], period_to_col[t])
                )

        # Process each group of tasks sharing the same control regression
        for control_key, tasks in tasks_by_group.items():
            # Use the first task to build X_control (same for all in the group)
            first_g, first_t, base_period_val, first_base_col, first_post_col = tasks[0]

            cov_matrix = covariate_by_period[base_period_val]

            # Build control mask (same for all tasks in this group)
            if self.control_group == "never_treated":
                control_mask = never_treated_mask
            else:
                # Controls must be untreated at both t and base_period_val
                nyt_threshold = max(first_t, base_period_val) + self.anticipation
                control_mask = never_treated_mask | (
                    (unit_cohorts > nyt_threshold) & (unit_cohorts != first_g)
                )

            # For balanced panels, valid_mask is all True so control_valid = control_mask
            if is_balanced:
                control_valid_base = control_mask
            else:
                y_base_first = outcome_matrix[:, first_base_col]
                y_post_first = outcome_matrix[:, first_post_col]
                valid_first = ~(np.isnan(y_base_first) | np.isnan(y_post_first))
                control_valid_base = control_mask & valid_first

            X_ctrl_raw = cov_matrix[control_valid_base]

            # Check for NaN in control covariates
            ctrl_has_nan = bool(np.any(np.isnan(X_ctrl_raw)))

            # Build X_ctrl with intercept
            n_c_base = int(np.sum(control_valid_base))
            if n_c_base == 0:
                continue

            X_ctrl = None
            cho = None
            kept_cols = None
            if not ctrl_has_nan:
                X_ctrl = np.column_stack([np.ones(n_c_base), X_ctrl_raw])

                # One-time rank check for this control group
                rank, dropped_cols, _ = _detect_rank_deficiency(X_ctrl)

                if len(dropped_cols) > 0:
                    # Rank-deficient: force lstsq for both "warn" and "silent".
                    # Cholesky on near-singular XtX could yield unstable coefficients.
                    if self.rank_deficient_action == "warn":
                        col_info = _format_dropped_columns(dropped_cols)
                        warnings.warn(
                            f"Rank-deficient covariate design (control_key={control_key}): "
                            f"dropped columns {col_info}. Rank {rank} < {X_ctrl.shape[1]}. "
                            "Using minimum-norm least-squares solution.",
                            UserWarning,
                            stacklevel=2,
                        )
                    cho = None  # Force lstsq path for ALL rank-deficient cases
                    kept_cols = np.array(
                        [i for i in range(X_ctrl.shape[1]) if i not in dropped_cols]
                    )
                else:
                    kept_cols = None  # Full rank — use all columns
                    with np.errstate(all="ignore"):
                        XtX = X_ctrl.T @ X_ctrl
                    try:
                        cho = scipy_linalg.cho_factor(XtX)
                    except np.linalg.LinAlgError:
                        cho = None

            # Process each (g, t) pair in this group
            for g, t, bp_val, base_col, post_col in tasks:
                treated_mask = cohort_masks[g]

                # Recompute control mask for not_yet_treated (varies by g, t)
                if self.control_group == "not_yet_treated":
                    # Controls must be untreated at both t and base period
                    nyt_threshold = max(t, bp_val) + self.anticipation
                    control_mask = never_treated_mask | (
                        (unit_cohorts > nyt_threshold) & (unit_cohorts != g)
                    )

                y_base = outcome_matrix[:, base_col]
                y_post = outcome_matrix[:, post_col]
                outcome_change = y_post - y_base

                if is_balanced:
                    valid_mask_pair = np.ones(len(y_base), dtype=bool)
                else:
                    valid_mask_pair = ~(np.isnan(y_base) | np.isnan(y_post))

                treated_valid = treated_mask & valid_mask_pair
                # For balanced + never_treated, control_valid is same as control_valid_base
                if is_balanced and self.control_group == "never_treated":
                    control_valid = control_valid_base
                else:
                    control_valid = control_mask & valid_mask_pair

                n_t = int(np.sum(treated_valid))
                n_c = int(np.sum(control_valid))

                if n_t == 0 or n_c == 0:
                    continue

                treated_change = outcome_change[treated_valid]
                control_change = outcome_change[control_valid]

                X_treated_pair = cov_matrix[treated_valid]
                X_control_pair = cov_matrix[control_valid]

                # Check for NaN in this pair's covariates
                if np.any(np.isnan(X_treated_pair)) or np.any(np.isnan(X_control_pair)):
                    # Fall back to unconditional (difference in means)
                    warnings.warn(
                        f"Missing values in covariates for group {g}, time {t}. "
                        "Falling back to unconditional estimation.",
                        UserWarning,
                        stacklevel=3,
                    )
                    att = float(np.mean(treated_change) - np.mean(control_change))
                    var_t = float(np.var(treated_change, ddof=1)) if n_t > 1 else 0.0
                    var_c = float(np.var(control_change, ddof=1)) if n_c > 1 else 0.0
                    se = float(np.sqrt(var_t / n_t + var_c / n_c))
                    inf_treated = (treated_change - np.mean(treated_change)) / n_t
                    inf_control = -(control_change - np.mean(control_change)) / n_c
                else:
                    # Build per-pair X_ctrl if control_valid differs from base
                    if is_balanced and self.control_group == "never_treated" and X_ctrl is not None:
                        pair_X_ctrl = X_ctrl
                        pair_n_c = n_c_base
                    else:
                        pair_X_ctrl = np.column_stack([np.ones(n_c), X_control_pair])
                        pair_n_c = n_c

                    # Solve for beta
                    beta = None
                    with np.errstate(all="ignore"):
                        if (
                            cho is not None
                            and is_balanced
                            and self.control_group == "never_treated"
                        ):
                            # Use cached Cholesky
                            Xty = pair_X_ctrl.T @ control_change
                            beta = scipy_linalg.cho_solve(cho, Xty)
                        else:
                            # Compute per-pair Cholesky or lstsq fallback
                            if kept_cols is not None:
                                # Rank-deficient: skip Cholesky, use reduced lstsq
                                pass
                            else:
                                pair_XtX = pair_X_ctrl.T @ pair_X_ctrl
                                try:
                                    pair_cho = scipy_linalg.cho_factor(pair_XtX)
                                    Xty = pair_X_ctrl.T @ control_change
                                    beta = scipy_linalg.cho_solve(pair_cho, Xty)
                                except np.linalg.LinAlgError:
                                    pass

                        if beta is None or np.any(~np.isfinite(beta)):
                            if kept_cols is not None:
                                # Reduced solve for rank-deficient design
                                result = scipy_linalg.lstsq(
                                    pair_X_ctrl[:, kept_cols],
                                    control_change,
                                    cond=1e-07,
                                )
                                beta = np.zeros(pair_X_ctrl.shape[1])
                                beta[kept_cols] = result[0]
                            else:
                                # Full-rank lstsq fallback (Cholesky numerical failure)
                                result = scipy_linalg.lstsq(
                                    pair_X_ctrl,
                                    control_change,
                                    cond=1e-07,
                                )
                                beta = result[0]

                    nan_cell = False

                    if beta is None or np.any(~np.isfinite(beta)):
                        nan_cell = True
                        n_nan_cells += 1

                    if not nan_cell:
                        X_treated_w_intercept = np.column_stack([np.ones(n_t), X_treated_pair])
                        with np.errstate(all="ignore"):
                            predicted_control = X_treated_w_intercept @ beta
                        treated_residuals = treated_change - predicted_control
                        if np.any(~np.isfinite(predicted_control)):
                            nan_cell = True
                            n_nan_cells += 1

                    if not nan_cell:
                        att = float(np.mean(treated_residuals))
                        with np.errstate(all="ignore"):
                            residuals = control_change - pair_X_ctrl @ beta
                        if np.any(~np.isfinite(residuals)):
                            nan_cell = True
                            n_nan_cells += 1

                    if nan_cell:
                        att = np.nan
                        se = np.nan
                        inf_treated = np.zeros(n_t)
                        inf_control = np.zeros(n_c)
                    else:
                        var_t = float(np.var(treated_residuals, ddof=1)) if n_t > 1 else 0.0
                        var_c = float(np.var(residuals, ddof=1)) if pair_n_c > 1 else 0.0
                        se = float(np.sqrt(var_t / n_t + var_c / pair_n_c))
                        inf_treated = (treated_residuals - np.mean(treated_residuals)) / n_t
                        inf_control = -residuals / pair_n_c

                group_time_effects[(g, t)] = {
                    "effect": att,
                    "se": se,
                    "t_stat": np.nan,
                    "p_value": np.nan,
                    "conf_int": (np.nan, np.nan),
                    "n_treated": n_t,
                    "n_control": n_c,
                }

                all_units = precomputed["all_units"]
                treated_positions = np.where(treated_valid)[0]
                control_positions = np.where(control_valid)[0]
                influence_func_info[(g, t)] = {
                    "treated_idx": treated_positions,
                    "control_idx": control_positions,
                    "treated_units": all_units[treated_positions],
                    "control_units": all_units[control_positions],
                    "treated_inf": inf_treated,
                    "control_inf": inf_control,
                }

                atts.append(att)
                ses.append(se)
                task_keys.append((g, t))

        if n_nan_cells > 0:
            warnings.warn(
                f"{n_nan_cells} group-time cell(s) have non-finite regression results "
                "(near-singular covariates). These cells are preserved with NaN inference.",
                UserWarning,
                stacklevel=2,
            )

        # Batch inference
        if task_keys:
            # Use survey df for replicate designs (propagated from precomputed)
            _ipw_dr_df = precomputed.get("df_survey") if precomputed is not None else None
            # Guard: replicate design with undefined df → NaN inference
            if (_ipw_dr_df is None and precomputed is not None
                    and precomputed.get("resolved_survey_unit") is not None
                    and hasattr(precomputed["resolved_survey_unit"], 'uses_replicate_variance')
                    and precomputed["resolved_survey_unit"].uses_replicate_variance):
                _ipw_dr_df = 0
            t_stats, p_values, ci_lowers, ci_uppers = safe_inference_batch(
                np.array(atts), np.array(ses), alpha=self.alpha, df=_ipw_dr_df
            )
            for idx, key in enumerate(task_keys):
                group_time_effects[key]["t_stat"] = float(t_stats[idx])
                group_time_effects[key]["p_value"] = float(p_values[idx])
                group_time_effects[key]["conf_int"] = (float(ci_lowers[idx]), float(ci_uppers[idx]))

        return group_time_effects, influence_func_info

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        first_treat: str,
        covariates: Optional[List[str]] = None,
        aggregate: Optional[str] = None,
        balance_e: Optional[int] = None,
        survey_design: object = None,
    ) -> CallawaySantAnnaResults:
        """
        Fit the Callaway-Sant'Anna estimator.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with unit and time identifiers.
        outcome : str
            Name of outcome variable column.
        unit : str
            Name of unit identifier column.
        time : str
            Name of time period column.
        first_treat : str
            Name of column indicating when unit was first treated.
            Use 0 (or np.inf) for never-treated units.
        covariates : list, optional
            List of covariate column names for conditional parallel trends.
        aggregate : str, optional
            How to aggregate group-time effects:
            - None: Only compute ATT(g,t) (default)
            - "simple": Simple weighted average (overall ATT)
            - "event_study": Aggregate by relative time (event study)
            - "group": Aggregate by treatment cohort
            - "all": Compute all aggregations
        balance_e : int, optional
            For event study, balance the panel at relative time e.
            Ensures all groups contribute to each relative period.
        survey_design : SurveyDesign, optional
            Survey design specification. Supports pweight with strata/PSU/FPC.
            Aggregated SEs (overall, event study, group) use design-based
            variance via compute_survey_if_variance().
            Covariates + IPW/DR + survey raises NotImplementedError.

        Returns
        -------
        CallawaySantAnnaResults
            Object containing all estimation results.

        Raises
        ------
        ValueError
            If required columns are missing or data validation fails.
        """
        # Validate pscore_trim (may have been changed via set_params)
        if not (0 < self.pscore_trim < 0.5):
            raise ValueError(f"pscore_trim must be in (0, 0.5), got {self.pscore_trim}")

        # Normalize empty covariates list to None
        if covariates is not None and len(covariates) == 0:
            covariates = None

        # Resolve survey design if provided
        from diff_diff.survey import (
            _resolve_survey_for_fit,
            _validate_unit_constant_survey,
        )

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, "analytical")
        )

        # Validate within-unit constancy for panel survey designs
        if resolved_survey is not None:
            _validate_unit_constant_survey(data, unit, survey_design)
            if resolved_survey.weight_type != "pweight":
                raise ValueError(
                    f"CallawaySantAnna survey support requires weight_type='pweight', "
                    f"got '{resolved_survey.weight_type}'. The survey variance math "
                    f"assumes probability weights (pweight)."
                )
        # Note: strata/PSU/FPC are now supported — aggregated SEs use
        # compute_survey_if_variance() for design-based inference.

        # Bootstrap + survey is now supported via PSU-level multiplier bootstrap.

        # Guard covariates + survey + IPW/DR (nuisance IF corrections not yet
        # implemented to match DRDID panel formula)
        if (
            resolved_survey is not None
            and covariates is not None
            and len(covariates) > 0
            and self.estimation_method in ("ipw", "dr")
        ):
            raise NotImplementedError(
                f"Survey weights with covariates and estimation_method="
                f"'{self.estimation_method}' is not yet supported for "
                f"CallawaySantAnna. The DRDID panel nuisance-estimation IF "
                f"corrections are not yet implemented. Use estimation_method='reg' "
                f"with covariates, or use any method without covariates."
            )

        # Validate inputs
        required_cols = [outcome, unit, time, first_treat]
        if covariates:
            required_cols.extend(covariates)

        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Create working copy
        df = data.copy()

        # Ensure numeric types
        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

        # Standardize the first_treat column name for internal use
        # This avoids hardcoding column names in internal methods
        df["first_treat"] = df[first_treat]

        # Never-treated indicator (must precede treatment_groups to exclude np.inf)
        df["_never_treated"] = (df[first_treat] == 0) | (df[first_treat] == np.inf)
        # Normalize np.inf → 0 so all downstream `> 0` checks exclude never-treated
        df.loc[df[first_treat] == np.inf, first_treat] = 0

        # Identify groups and time periods
        time_periods = sorted(df[time].unique())
        treatment_groups = sorted([g for g in df[first_treat].unique() if g > 0])

        # Get unique units
        unit_info = (
            df.groupby(unit).agg({first_treat: "first", "_never_treated": "first"}).reset_index()
        )

        n_treated_units = (unit_info[first_treat] > 0).sum()
        n_control_units = (unit_info["_never_treated"]).sum()

        if n_control_units == 0 and self.control_group == "never_treated":
            raise ValueError(
                "No never-treated units found. Check 'first_treat' column. "
                "Use control_group='not_yet_treated' if all units are eventually treated."
            )
        if n_control_units == 0 and self.control_group == "not_yet_treated":
            # With not_yet_treated, controls are units not yet treated at each
            # (g, t) pair — never-treated units are not required.
            if len(treatment_groups) < 2:
                raise ValueError(
                    "not_yet_treated control group requires at least 2 treatment "
                    "cohorts when there are no never-treated units."
                )

        # Note: CallawaySantAnna uses weights-only survey (strata/PSU/FPC
        # rejected above). We do NOT inject cluster-as-PSU here because CS
        # per-cell SEs use IF-based variance, not TSL. The user's cluster=
        # parameter is handled by the existing non-survey clustering path.
        # Pre-compute data structures for efficient ATT(g,t) computation
        precomputed = self._precompute_structures(
            df,
            outcome,
            unit,
            time,
            first_treat,
            covariates,
            time_periods,
            treatment_groups,
            resolved_survey=resolved_survey,
        )

        # Recompute survey metadata from the unit-level resolved survey so
        # that n_psu and df_survey reflect the actual survey design (explicit
        # PSU/strata) rather than hard-coding n_units.
        if resolved_survey is not None and survey_metadata is not None:
            resolved_survey_unit = precomputed.get("resolved_survey_unit")
            if resolved_survey_unit is not None:
                from diff_diff.survey import compute_survey_metadata

                unit_w = resolved_survey_unit.weights
                survey_metadata = compute_survey_metadata(resolved_survey_unit, unit_w)

        # Survey df for safe_inference calls — use the unit-level resolved
        # survey df computed in _precompute_structures for consistency.
        df_survey = precomputed.get("df_survey")
        # Guard: replicate design with undefined df (rank <= 1) → NaN inference
        if (df_survey is None and resolved_survey is not None
                and hasattr(resolved_survey, 'uses_replicate_variance')
                and resolved_survey.uses_replicate_variance):
            df_survey = 0

        # Compute ATT(g,t) for each group-time combination
        min_period = min(time_periods)
        has_survey = resolved_survey is not None

        if covariates is None and self.estimation_method == "reg":
            # Fast vectorized path for the common no-covariates regression case
            group_time_effects, influence_func_info = self._compute_all_att_gt_vectorized(
                precomputed, treatment_groups, time_periods, min_period
            )
        elif (
            covariates is not None
            and self.estimation_method == "reg"
            and self.rank_deficient_action != "error"
            and not has_survey  # Cholesky cache uses X'X; survey needs X'WX
        ):
            # Optimized covariate regression path with Cholesky caching
            group_time_effects, influence_func_info = self._compute_all_att_gt_covariate_reg(
                precomputed, treatment_groups, time_periods, min_period
            )
        else:
            # General path: IPW, DR, rank_deficient_action="error", or edge cases
            group_time_effects = {}
            influence_func_info = {}

            # Propensity score cache for IPW/DR with covariates
            pscore_cache = {} if (covariates and self.estimation_method in ("ipw", "dr")) else None
            # Cholesky cache for DR outcome regression component
            # Skip cache when survey weights present (X'WX differs from X'X)
            cho_cache = (
                {}
                if (
                    covariates
                    and self.estimation_method == "dr"
                    and self.rank_deficient_action != "error"
                    and not has_survey
                )
                else None
            )

            for g in treatment_groups:
                if self.base_period == "universal":
                    universal_base = g - 1 - self.anticipation
                    valid_periods = [t for t in time_periods if t != universal_base]
                else:
                    valid_periods = [
                        t for t in time_periods if t >= g - self.anticipation or t > min_period
                    ]

                for t in valid_periods:
                    att_gt, se_gt, n_treat, n_ctrl, inf_info, sw_sum = self._compute_att_gt_fast(
                        precomputed,
                        g,
                        t,
                        covariates,
                        pscore_cache=pscore_cache,
                        cho_cache=cho_cache,
                    )

                    if att_gt is not None:
                        t_stat, p_val, ci = safe_inference(
                            att_gt,
                            se_gt,
                            alpha=self.alpha,
                            df=df_survey,
                        )

                        gte_entry = {
                            "effect": att_gt,
                            "se": se_gt,
                            "t_stat": t_stat,
                            "p_value": p_val,
                            "conf_int": ci,
                            "n_treated": n_treat,
                            "n_control": n_ctrl,
                        }
                        if sw_sum is not None:
                            gte_entry["survey_weight_sum"] = sw_sum
                        group_time_effects[(g, t)] = gte_entry

                        if inf_info is not None:
                            influence_func_info[(g, t)] = inf_info

        if not group_time_effects:
            raise ValueError(
                "Could not estimate any group-time effects. "
                "Check that data has sufficient observations."
            )

        # Compute overall ATT (simple aggregation)
        overall_att, overall_se = self._aggregate_simple(
            group_time_effects, influence_func_info, df, unit, precomputed
        )
        # Re-read df_survey in case replicate aggregation updated it
        df_survey = precomputed.get("df_survey")
        # Propagate replicate df override to survey_metadata for display consistency
        if df_survey is not None and survey_metadata is not None:
            if survey_metadata.df_survey != df_survey:
                survey_metadata.df_survey = df_survey
        # Guard: replicate design with undefined df (rank <= 1) → NaN inference
        if (df_survey is None and resolved_survey is not None
                and hasattr(resolved_survey, 'uses_replicate_variance')
                and resolved_survey.uses_replicate_variance):
            df_survey = 0
        overall_t, overall_p, overall_ci = safe_inference(
            overall_att,
            overall_se,
            alpha=self.alpha,
            df=df_survey,
        )

        # Compute additional aggregations if requested
        event_study_effects = None
        group_effects = None

        if aggregate in ["event_study", "all"]:
            event_study_effects = self._aggregate_event_study(
                group_time_effects,
                influence_func_info,
                treatment_groups,
                time_periods,
                balance_e,
                df,
                unit,
                precomputed,
            )

        if aggregate in ["group", "all"]:
            group_effects = self._aggregate_by_group(
                group_time_effects,
                influence_func_info,
                treatment_groups,
                precomputed=precomputed,
                df=df,
                unit=unit,
            )

        # Reject replicate-weight designs for bootstrap — replicate variance
        # is an analytical alternative, not compatible with bootstrap
        if (
            self.n_bootstrap > 0
            and resolved_survey is not None
            and hasattr(resolved_survey, "uses_replicate_variance")
            and resolved_survey.uses_replicate_variance
        ):
            raise NotImplementedError(
                "CallawaySantAnna bootstrap (n_bootstrap > 0) is not supported "
                "with replicate-weight survey designs. Replicate weights provide "
                "analytical variance; use n_bootstrap=0 instead."
            )

        # Run bootstrap inference if requested
        bootstrap_results = None
        if self.n_bootstrap > 0 and influence_func_info:
            bootstrap_results = self._run_multiplier_bootstrap(
                group_time_effects=group_time_effects,
                influence_func_info=influence_func_info,
                aggregate=aggregate,
                balance_e=balance_e,
                treatment_groups=treatment_groups,
                time_periods=time_periods,
                df=df,
                unit=unit,
                precomputed=precomputed,
                cband=self.cband,
            )

            # Update estimates with bootstrap inference
            overall_se = bootstrap_results.overall_att_se
            overall_t = safe_inference(overall_att, overall_se, alpha=self.alpha)[0]
            overall_p = bootstrap_results.overall_att_p_value
            overall_ci = bootstrap_results.overall_att_ci

            # Update group-time effects with bootstrap SEs (batched)
            gt_keys = [gt for gt in group_time_effects if gt in bootstrap_results.group_time_ses]
            if gt_keys:
                gt_effects_arr = np.array(
                    [float(group_time_effects[gt]["effect"]) for gt in gt_keys]
                )
                gt_ses_arr = np.array(
                    [float(bootstrap_results.group_time_ses[gt]) for gt in gt_keys]
                )
                gt_t_stats, _, _, _ = safe_inference_batch(
                    gt_effects_arr, gt_ses_arr, alpha=self.alpha
                )
                for idx, gt in enumerate(gt_keys):
                    group_time_effects[gt]["se"] = bootstrap_results.group_time_ses[gt]
                    group_time_effects[gt]["conf_int"] = bootstrap_results.group_time_cis[gt]
                    group_time_effects[gt]["p_value"] = bootstrap_results.group_time_p_values[gt]
                    group_time_effects[gt]["t_stat"] = float(gt_t_stats[idx])

            # Update event study effects with bootstrap SEs (batched)
            if (
                event_study_effects is not None
                and bootstrap_results.event_study_ses is not None
                and bootstrap_results.event_study_cis is not None
                and bootstrap_results.event_study_p_values is not None
            ):
                es_keys = [e for e in event_study_effects if e in bootstrap_results.event_study_ses]
                if es_keys:
                    es_effects_arr = np.array(
                        [float(event_study_effects[e]["effect"]) for e in es_keys]
                    )
                    es_ses_arr = np.array(
                        [float(bootstrap_results.event_study_ses[e]) for e in es_keys]
                    )
                    es_t_stats, _, _, _ = safe_inference_batch(
                        es_effects_arr, es_ses_arr, alpha=self.alpha
                    )
                    for idx, e in enumerate(es_keys):
                        event_study_effects[e]["se"] = bootstrap_results.event_study_ses[e]
                        event_study_effects[e]["conf_int"] = bootstrap_results.event_study_cis[e]
                        event_study_effects[e]["p_value"] = bootstrap_results.event_study_p_values[
                            e
                        ]
                        event_study_effects[e]["t_stat"] = float(es_t_stats[idx])

            # Update group effects with bootstrap SEs (batched)
            if (
                group_effects is not None
                and bootstrap_results.group_effect_ses is not None
                and bootstrap_results.group_effect_cis is not None
                and bootstrap_results.group_effect_p_values is not None
            ):
                grp_keys = [g for g in group_effects if g in bootstrap_results.group_effect_ses]
                if grp_keys:
                    grp_effects_arr = np.array(
                        [float(group_effects[g]["effect"]) for g in grp_keys]
                    )
                    grp_ses_arr = np.array(
                        [float(bootstrap_results.group_effect_ses[g]) for g in grp_keys]
                    )
                    grp_t_stats, _, _, _ = safe_inference_batch(
                        grp_effects_arr, grp_ses_arr, alpha=self.alpha
                    )
                    for idx, g in enumerate(grp_keys):
                        group_effects[g]["se"] = bootstrap_results.group_effect_ses[g]
                        group_effects[g]["conf_int"] = bootstrap_results.group_effect_cis[g]
                        group_effects[g]["p_value"] = bootstrap_results.group_effect_p_values[g]
                        group_effects[g]["t_stat"] = float(grp_t_stats[idx])

        # Compute simultaneous confidence band CIs if cband is available
        cband_crit_value = None
        if bootstrap_results is not None:
            cband_crit_value = bootstrap_results.cband_crit_value

        if cband_crit_value is not None and event_study_effects is not None:
            for e, eff_data in event_study_effects.items():
                se_val = eff_data["se"]
                if np.isfinite(se_val) and se_val > 0:
                    eff_data["cband_conf_int"] = (
                        eff_data["effect"] - cband_crit_value * se_val,
                        eff_data["effect"] + cband_crit_value * se_val,
                    )

        # Store results
        self.results_ = CallawaySantAnnaResults(
            group_time_effects=group_time_effects,
            overall_att=overall_att,
            overall_se=overall_se,
            overall_t_stat=overall_t,
            overall_p_value=overall_p,
            overall_conf_int=overall_ci,
            groups=treatment_groups,
            time_periods=time_periods,
            n_obs=len(df),
            n_treated_units=n_treated_units,
            n_control_units=n_control_units,
            alpha=self.alpha,
            control_group=self.control_group,
            base_period=self.base_period,
            event_study_effects=event_study_effects,
            group_effects=group_effects,
            bootstrap_results=bootstrap_results,
            cband_crit_value=cband_crit_value,
            pscore_trim=self.pscore_trim,
            survey_metadata=survey_metadata,
        )

        self.is_fitted_ = True
        return self.results_

    def _outcome_regression(
        self,
        treated_change: np.ndarray,
        control_change: np.ndarray,
        X_treated: Optional[np.ndarray] = None,
        X_control: Optional[np.ndarray] = None,
        sw_treated: Optional[np.ndarray] = None,
        sw_control: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Estimate ATT using outcome regression.

        With covariates:
        1. Regress outcome changes on covariates for control group
        2. Predict counterfactual for treated using their covariates
        3. ATT = mean(treated_change) - mean(predicted_counterfactual)

        Without covariates:
        Simple difference in means.

        Parameters
        ----------
        sw_treated, sw_control : np.ndarray, optional
            Survey weights for treated and control units.
        """
        n_t = len(treated_change)
        n_c = len(control_change)

        if X_treated is not None and X_control is not None and X_treated.shape[1] > 0:
            # Covariate-adjusted outcome regression
            # Fit regression on control units: E[Delta Y | X, D=0]
            beta, residuals = _linear_regression(
                X_control,
                control_change,
                rank_deficient_action=self.rank_deficient_action,
                weights=sw_control,
            )

            # Zero NaN coefficients for prediction (dropped rank-deficient columns
            # contribute 0 to the column space projection, matching DR path convention)
            beta = np.where(np.isfinite(beta), beta, 0.0)

            # Predict counterfactual for treated units
            X_treated_with_intercept = np.column_stack([np.ones(n_t), X_treated])
            predicted_control = np.dot(X_treated_with_intercept, beta)

            # ATT: survey-weighted mean of treated residuals
            treated_residuals = treated_change - predicted_control

            if sw_treated is not None:
                sw_t_sum = float(np.sum(sw_treated))
                sw_c_sum = float(np.sum(sw_control))
                sw_t_norm = sw_treated / sw_t_sum
                sw_c_norm = sw_control / sw_c_sum
                att = float(np.sum(sw_t_norm * treated_residuals))

                # Survey-weighted OR influence function.
                # Mirrors unweighted: inf_treated = (resid-ATT)/n_t,
                # inf_control = -resid/n_c. Survey: w_i/sum(w_group).
                # WLS residuals are orthogonal to W*X by construction.
                X_c_int = np.column_stack([np.ones(n_c), X_control])
                resid_c = control_change - np.dot(X_c_int, beta)

                inf_treated = sw_t_norm * (treated_residuals - att)
                inf_control = -sw_c_norm * resid_c
                inf_func = np.concatenate([inf_treated, inf_control])

                # SE: survey-weighted variance matching unweighted var_t/n_t + var_c/n_c
                var_t = float(np.sum(sw_t_norm * (treated_residuals - att) ** 2))
                var_c = float(np.sum(sw_c_norm * resid_c**2))
                se = float(np.sqrt(var_t + var_c)) if (n_t > 0 and n_c > 0) else 0.0
            else:
                att = float(np.mean(treated_residuals))

                # Standard error using sandwich estimator
                var_t = np.var(treated_residuals, ddof=1) if n_t > 1 else 0.0
                var_c = np.var(residuals, ddof=1) if n_c > 1 else 0.0
                se = float(np.sqrt(var_t / n_t + var_c / n_c)) if (n_t > 0 and n_c > 0) else 0.0

                # Influence function
                inf_treated = (treated_residuals - np.mean(treated_residuals)) / n_t
                inf_control = -residuals / n_c
                inf_func = np.concatenate([inf_treated, inf_control])
        else:
            # Simple difference in means (no covariates)
            if sw_treated is not None:
                sw_t_norm = sw_treated / np.sum(sw_treated)
                sw_c_norm = sw_control / np.sum(sw_control)
                mu_t = float(np.sum(sw_t_norm * treated_change))
                mu_c = float(np.sum(sw_c_norm * control_change))
                att = mu_t - mu_c

                # Influence function (survey-weighted)
                inf_treated = sw_t_norm * (treated_change - mu_t)
                inf_control = -sw_c_norm * (control_change - mu_c)
                inf_func = np.concatenate([inf_treated, inf_control])

                # SE from influence function variance
                se = (
                    float(np.sqrt(np.sum(inf_treated**2) + np.sum(inf_control**2)))
                    if (n_t > 0 and n_c > 0)
                    else 0.0
                )
            else:
                att = float(np.mean(treated_change) - np.mean(control_change))

                var_t = np.var(treated_change, ddof=1) if n_t > 1 else 0.0
                var_c = np.var(control_change, ddof=1) if n_c > 1 else 0.0
                se = float(np.sqrt(var_t / n_t + var_c / n_c)) if (n_t > 0 and n_c > 0) else 0.0

                # Influence function (for aggregation)
                inf_treated = treated_change - np.mean(treated_change)
                inf_control = control_change - np.mean(control_change)
                inf_func = np.concatenate([inf_treated / n_t, -inf_control / n_c])

        return att, se, inf_func

    def _ipw_estimation(
        self,
        treated_change: np.ndarray,
        control_change: np.ndarray,
        n_treated: int,
        n_control: int,
        X_treated: Optional[np.ndarray] = None,
        X_control: Optional[np.ndarray] = None,
        pscore_cache: Optional[Dict] = None,
        pscore_key: Optional[Any] = None,
        sw_treated: Optional[np.ndarray] = None,
        sw_control: Optional[np.ndarray] = None,
        sw_all: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Estimate ATT using inverse probability weighting.

        With covariates:
        1. Estimate propensity score P(D=1|X) using logistic regression
        2. Reweight control units to match treated covariate distribution
        3. ATT = mean(treated) - weighted_mean(control)

        Without covariates:
        Simple difference in means with unconditional propensity weighting.

        Parameters
        ----------
        sw_treated, sw_control, sw_all : np.ndarray, optional
            Survey weights for treated, control, and all units.
        """
        n_t = len(treated_change)
        n_c = len(control_change)
        n_total = n_treated + n_control

        if X_treated is not None and X_control is not None and X_treated.shape[1] > 0:
            # Covariate-adjusted IPW estimation
            # Check propensity score cache
            cached_pscore = None
            if pscore_cache is not None and pscore_key is not None:
                cached_pscore = pscore_cache.get(pscore_key)

            if cached_pscore is not None:
                # Use cached propensity scores (beta coefficients)
                beta_logistic = cached_pscore
                X_all = np.vstack([X_treated, X_control])
                X_all_with_intercept = np.column_stack([np.ones(n_t + n_c), X_all])
                z = np.dot(X_all_with_intercept, beta_logistic)
                z = np.clip(z, -500, 500)
                pscore = 1 / (1 + np.exp(-z))
            else:
                # Stack covariates and create treatment indicator
                X_all = np.vstack([X_treated, X_control])
                D = np.concatenate([np.ones(n_t), np.zeros(n_c)])

                # Estimate propensity scores using IRLS logistic regression
                try:
                    beta_logistic, pscore = solve_logit(
                        X_all,
                        D,
                        rank_deficient_action=self.rank_deficient_action,
                        weights=sw_all,
                    )
                    _check_propensity_diagnostics(pscore, self.pscore_trim)
                    # Cache the fitted coefficients
                    if pscore_cache is not None and pscore_key is not None:
                        pscore_cache[pscore_key] = beta_logistic
                except (np.linalg.LinAlgError, ValueError):
                    if self.rank_deficient_action == "error":
                        raise
                    # Fallback to unconditional if logistic regression fails
                    warnings.warn(
                        "Propensity score estimation failed. "
                        "Falling back to unconditional estimation.",
                        UserWarning,
                        stacklevel=4,
                    )
                    pscore = np.full(len(D), n_t / (n_t + n_c))

            # Propensity scores for treated and control
            pscore_treated = pscore[:n_t]
            pscore_control = pscore[n_t:]

            # Clip propensity scores to avoid extreme weights
            pscore_control = np.clip(pscore_control, self.pscore_trim, 1 - self.pscore_trim)
            pscore_treated = np.clip(pscore_treated, self.pscore_trim, 1 - self.pscore_trim)

            if sw_treated is not None:
                # IPW weights compose with survey weights:
                # w_i = sw_i * p(X_i) / (1 - p(X_i))
                weights_control = sw_control * pscore_control / (1 - pscore_control)
                weights_control_norm = weights_control / np.sum(weights_control)

                # ATT: survey-weighted treated mean minus composite-weighted control mean
                sw_t_norm = sw_treated / np.sum(sw_treated)
                mu_t = float(np.sum(sw_t_norm * treated_change))
                att = mu_t - float(np.sum(weights_control_norm * control_change))

                # Influence function (survey-weighted)
                inf_treated = sw_t_norm * (treated_change - mu_t)
                inf_control = -weights_control_norm * (
                    control_change - np.sum(weights_control_norm * control_change)
                )
                inf_func = np.concatenate([inf_treated, inf_control])

                # Propensity score IF correction
                # Accounts for estimation uncertainty in logistic regression coefficients
                X_all_int = np.column_stack([np.ones(n_t + n_c), X_all])
                pscore_all = np.concatenate([pscore_treated, pscore_control])

                # Survey-weighted PS Hessian: sum(w_i * mu_i * (1-mu_i) * x_i * x_i')
                W_ps = pscore_all * (1 - pscore_all)
                if sw_all is not None:
                    W_ps = W_ps * sw_all
                H = X_all_int.T @ (W_ps[:, None] * X_all_int)
                try:
                    H_inv = np.linalg.solve(H, np.eye(H.shape[0]))
                except np.linalg.LinAlgError:
                    H_inv = np.linalg.lstsq(H, np.eye(H.shape[0]), rcond=None)[0]

                # PS score: w_i * (D_i - pi_i) * X_i
                D_all = np.concatenate([np.ones(n_t), np.zeros(n_c)])
                score_ps = (D_all - pscore_all)[:, None] * X_all_int
                if sw_all is not None:
                    score_ps = score_ps * sw_all[:, None]
                asy_lin_rep_ps = score_ps @ H_inv  # shape (n_t + n_c, p)

                # M2: gradient of ATT w.r.t. PS parameters
                att_control_weighted = np.sum(weights_control_norm * control_change)
                M2 = np.mean(
                    (weights_control_norm * (control_change - att_control_weighted))[:, None]
                    * X_all_int[n_t:],
                    axis=0,
                )

                # PS correction to influence function
                inf_ps_correction = asy_lin_rep_ps @ M2
                inf_func = inf_func + inf_ps_correction

                # SE from influence function variance
                var_psi = np.sum(inf_func**2)
                se = float(np.sqrt(var_psi)) if var_psi > 0 else 0.0
            else:
                # IPW weights for control units: p(X) / (1 - p(X))
                # This reweights controls to have same covariate distribution as treated
                weights_control = pscore_control / (1 - pscore_control)
                weights_control = weights_control / np.sum(weights_control)  # normalize

                # ATT = mean(treated) - weighted_mean(control)
                att = float(np.mean(treated_change) - np.sum(weights_control * control_change))

                # Compute standard error
                var_t = np.var(treated_change, ddof=1) if n_t > 1 else 0.0

                weighted_var_c = np.sum(
                    weights_control
                    * (control_change - np.sum(weights_control * control_change)) ** 2
                )

                se = float(np.sqrt(var_t / n_t + weighted_var_c)) if (n_t > 0 and n_c > 0) else 0.0

                # Influence function
                inf_treated = (treated_change - np.mean(treated_change)) / n_t
                inf_control = -weights_control * (
                    control_change - np.sum(weights_control * control_change)
                )
                inf_func = np.concatenate([inf_treated, inf_control])
        else:
            # Unconditional IPW (reduces to difference in means)
            if sw_treated is not None:
                # Survey-weighted difference in means
                sw_t_norm = sw_treated / np.sum(sw_treated)
                sw_c_norm = sw_control / np.sum(sw_control)
                mu_t = float(np.sum(sw_t_norm * treated_change))
                mu_c = float(np.sum(sw_c_norm * control_change))
                att = mu_t - mu_c

                inf_treated = sw_t_norm * (treated_change - mu_t)
                inf_control = -sw_c_norm * (control_change - mu_c)
                inf_func = np.concatenate([inf_treated, inf_control])

                se = (
                    float(np.sqrt(np.sum(inf_treated**2) + np.sum(inf_control**2)))
                    if (n_t > 0 and n_c > 0)
                    else 0.0
                )
            else:
                p_treat = n_treated / n_total  # unconditional propensity score

                att = float(np.mean(treated_change) - np.mean(control_change))

                var_t = np.var(treated_change, ddof=1) if n_t > 1 else 0.0
                var_c = np.var(control_change, ddof=1) if n_c > 1 else 0.0

                # Adjusted variance for IPW
                se = float(
                    np.sqrt(var_t / n_t + var_c * (1 - p_treat) / (n_c * p_treat))
                    if (n_t > 0 and n_c > 0 and p_treat > 0)
                    else 0.0
                )

                # Influence function (for aggregation)
                inf_treated = (treated_change - np.mean(treated_change)) / n_t
                inf_control = (control_change - np.mean(control_change)) / n_c
                inf_func = np.concatenate([inf_treated, -inf_control])

        return att, se, inf_func

    def _doubly_robust(
        self,
        treated_change: np.ndarray,
        control_change: np.ndarray,
        X_treated: Optional[np.ndarray] = None,
        X_control: Optional[np.ndarray] = None,
        pscore_cache: Optional[Dict] = None,
        pscore_key: Optional[Any] = None,
        cho_cache: Optional[Dict] = None,
        cho_key: Optional[Any] = None,
        sw_treated: Optional[np.ndarray] = None,
        sw_control: Optional[np.ndarray] = None,
        sw_all: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Estimate ATT using doubly robust estimation.

        With covariates:
        Combines outcome regression and IPW for double robustness.
        The estimator is consistent if either the outcome model OR
        the propensity model is correctly specified.

        ATT_DR = (1/n_t) * sum_i[D_i * (Y_i - m(X_i))]
               + (1/n_t) * sum_i[(1-D_i) * w_i * (m(X_i) - Y_i)]

        where m(X) is the outcome model and w_i are IPW weights.

        Without covariates:
        Reduces to simple difference in means.

        Parameters
        ----------
        sw_treated, sw_control, sw_all : np.ndarray, optional
            Survey weights for treated, control, and all units.
        """
        n_t = len(treated_change)
        n_c = len(control_change)

        if X_treated is not None and X_control is not None and X_treated.shape[1] > 0:
            # Doubly robust estimation with covariates
            # Step 1: Outcome regression - fit E[Delta Y | X] on control
            # Try Cholesky cache for outcome regression (disabled when survey weights present)
            beta = None
            X_control_with_intercept = np.column_stack([np.ones(n_c), X_control])
            if cho_cache is not None and cho_key is not None:
                cached_cho = cho_cache.get(cho_key)

                if cached_cho is False:
                    # Rank-deficient sentinel: skip Cholesky, fall through
                    pass
                elif cached_cho is not None:
                    Xty = X_control_with_intercept.T @ control_change
                    beta = scipy_linalg.cho_solve(cached_cho, Xty)
                    if np.any(~np.isfinite(beta)):
                        beta = None
                else:
                    # First time for this cho_key: check rank before Cholesky
                    rank_info = _detect_rank_deficiency(X_control_with_intercept)
                    if len(rank_info[1]) > 0:
                        cho_cache[cho_key] = False  # Sentinel
                    else:
                        XtX = X_control_with_intercept.T @ X_control_with_intercept
                        try:
                            cho_factor = scipy_linalg.cho_factor(XtX)
                            cho_cache[cho_key] = cho_factor
                            Xty = X_control_with_intercept.T @ control_change
                            beta = scipy_linalg.cho_solve(cho_factor, Xty)
                            if np.any(~np.isfinite(beta)):
                                beta = None
                        except np.linalg.LinAlgError:
                            pass

            if beta is None:
                beta, _ = _linear_regression(
                    X_control,
                    control_change,
                    rank_deficient_action=self.rank_deficient_action,
                    weights=sw_control,
                )
                # Zero NaN coefficients for prediction only — dropped columns
                # contribute 0 to the column space projection. Note: solve_ols
                # deliberately uses NaN (R's lm() convention) for inference, but
                # here we only need beta for prediction (m_treated, m_control).
                beta = np.where(np.isfinite(beta), beta, 0.0)

            # Predict counterfactual for both treated and control
            X_treated_with_intercept = np.column_stack([np.ones(n_t), X_treated])
            m_treated = np.dot(X_treated_with_intercept, beta)
            m_control = np.dot(X_control_with_intercept, beta)

            # Step 2: Propensity score estimation
            # Check propensity score cache
            cached_pscore = None
            if pscore_cache is not None and pscore_key is not None:
                cached_pscore = pscore_cache.get(pscore_key)

            if cached_pscore is not None:
                beta_logistic = cached_pscore
                X_all = np.vstack([X_treated, X_control])
                X_all_with_intercept = np.column_stack([np.ones(n_t + n_c), X_all])
                z = np.dot(X_all_with_intercept, beta_logistic)
                z = np.clip(z, -500, 500)
                pscore = 1 / (1 + np.exp(-z))
            else:
                X_all = np.vstack([X_treated, X_control])
                D = np.concatenate([np.ones(n_t), np.zeros(n_c)])

                try:
                    beta_logistic, pscore = solve_logit(
                        X_all,
                        D,
                        rank_deficient_action=self.rank_deficient_action,
                        weights=sw_all,
                    )
                    _check_propensity_diagnostics(pscore, self.pscore_trim)
                    if pscore_cache is not None and pscore_key is not None:
                        pscore_cache[pscore_key] = beta_logistic
                except (np.linalg.LinAlgError, ValueError):
                    if self.rank_deficient_action == "error":
                        raise
                    # Fallback to unconditional if logistic regression fails
                    warnings.warn(
                        "Propensity score estimation failed. "
                        "Falling back to unconditional estimation.",
                        UserWarning,
                        stacklevel=4,
                    )
                    pscore = np.full(len(D), n_t / (n_t + n_c))

            pscore_control = pscore[n_t:]

            # Clip propensity scores
            pscore_control = np.clip(pscore_control, self.pscore_trim, 1 - self.pscore_trim)

            if sw_treated is not None:
                # IPW weights compose with survey weights
                weights_control = sw_control * pscore_control / (1 - pscore_control)

                # Step 3: DR ATT (survey-weighted)
                sw_t_sum = np.sum(sw_treated)
                att_treated_part = float(
                    np.sum(sw_treated * (treated_change - m_treated)) / sw_t_sum
                )
                augmentation = float(
                    np.sum(weights_control * (m_control - control_change)) / sw_t_sum
                )
                att = att_treated_part + augmentation

                # Step 4: Influence function (survey-weighted DR)
                psi_treated = (sw_treated / sw_t_sum) * (treated_change - m_treated - att)
                psi_control = (weights_control / sw_t_sum) * (m_control - control_change)

                var_psi = np.sum(psi_treated**2) + np.sum(psi_control**2)
                se = float(np.sqrt(var_psi)) if var_psi > 0 else 0.0

                inf_func = np.concatenate([psi_treated, psi_control])
            else:
                # IPW weights for control: p(X) / (1 - p(X))
                weights_control = pscore_control / (1 - pscore_control)

                # Step 3: Doubly robust ATT
                att_treated_part = float(np.mean(treated_change - m_treated))
                augmentation = float(np.sum(weights_control * (m_control - control_change)) / n_t)
                att = att_treated_part + augmentation

                # Step 4: Standard error using influence function
                psi_treated = (treated_change - m_treated - att) / n_t
                psi_control = (weights_control * (m_control - control_change)) / n_t

                var_psi = np.sum(psi_treated**2) + np.sum(psi_control**2)
                se = float(np.sqrt(var_psi)) if var_psi > 0 else 0.0

                inf_func = np.concatenate([psi_treated, psi_control])
        else:
            # Without covariates, DR simplifies to difference in means
            if sw_treated is not None:
                sw_t_norm = sw_treated / np.sum(sw_treated)
                sw_c_norm = sw_control / np.sum(sw_control)
                mu_t = float(np.sum(sw_t_norm * treated_change))
                mu_c = float(np.sum(sw_c_norm * control_change))
                att = mu_t - mu_c

                inf_treated = sw_t_norm * (treated_change - mu_t)
                inf_control = -sw_c_norm * (control_change - mu_c)
                inf_func = np.concatenate([inf_treated, inf_control])

                se = (
                    float(np.sqrt(np.sum(inf_treated**2) + np.sum(inf_control**2)))
                    if (n_t > 0 and n_c > 0)
                    else 0.0
                )
            else:
                att = float(np.mean(treated_change) - np.mean(control_change))

                var_t = np.var(treated_change, ddof=1) if n_t > 1 else 0.0
                var_c = np.var(control_change, ddof=1) if n_c > 1 else 0.0

                se = float(np.sqrt(var_t / n_t + var_c / n_c)) if (n_t > 0 and n_c > 0) else 0.0

                # Influence function for DR estimator
                inf_treated = (treated_change - np.mean(treated_change)) / n_t
                inf_control = (control_change - np.mean(control_change)) / n_c
                inf_func = np.concatenate([inf_treated, -inf_control])

        return att, se, inf_func

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters (sklearn-compatible)."""
        return {
            "control_group": self.control_group,
            "anticipation": self.anticipation,
            "estimation_method": self.estimation_method,
            "alpha": self.alpha,
            "cluster": self.cluster,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_weights": self.bootstrap_weights,
            # Deprecated but kept for backward compatibility
            "bootstrap_weight_type": self.bootstrap_weight_type,
            "seed": self.seed,
            "rank_deficient_action": self.rank_deficient_action,
            "base_period": self.base_period,
            "cband": self.cband,
            "pscore_trim": self.pscore_trim,
        }

    def set_params(self, **params) -> "CallawaySantAnna":
        """Set estimator parameters (sklearn-compatible)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self

    def summary(self) -> str:
        """Get summary of estimation results."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling summary()")
        assert self.results_ is not None
        return self.results_.summary()

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())
