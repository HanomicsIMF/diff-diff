"""
Efficient Difference-in-Differences estimator.

Implements the ATT estimator from Chen, Sant'Anna & Xie (2025).
Without covariates, achieves the semiparametric efficiency bound via
closed-form within-group covariances.  With covariates, uses a doubly
robust path with OLS outcome regression, sieve propensity ratios, and
kernel-smoothed conditional Omega*(X) (see class docstring for caveats).

Under PT-All the model is overidentified and EDiD exploits this for
tighter inference; under PT-Post it reduces to the standard
single-baseline estimator (Callaway-Sant'Anna).
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.efficient_did_bootstrap import (
    EDiDBootstrapResults,
    EfficientDiDBootstrapMixin,
)
from diff_diff.efficient_did_covariates import (
    compute_eif_cov,
    compute_generated_outcomes_cov,
    compute_omega_star_conditional,
    compute_per_unit_weights,
    estimate_inverse_propensity_sieve,
    estimate_outcome_regression,
    estimate_propensity_ratio_sieve,
)
from diff_diff.efficient_did_results import EfficientDiDResults
from diff_diff.efficient_did_weights import (
    compute_efficient_weights,
    compute_eif_nocov,
    compute_generated_outcomes_nocov,
    compute_omega_star_nocov,
    enumerate_valid_triples,
)
from diff_diff.utils import safe_inference

# Re-export for convenience
__all__ = ["EfficientDiD", "EfficientDiDResults", "EDiDBootstrapResults"]


class EfficientDiD(EfficientDiDBootstrapMixin):
    """Efficient DiD estimator (Chen, Sant'Anna & Xie 2025).

    Without covariates, achieves the semiparametric efficiency bound for
    ATT(g,t) using a closed-form estimator based on within-group sample
    means and covariances.

    With covariates, uses a doubly robust path: sieve-based propensity
    score ratios (Eq 4.1-4.2), OLS outcome regression, sieve-estimated
    inverse propensities (algorithm step 4), and kernel-smoothed
    conditional Omega*(X) with per-unit efficient weights (Eq 3.12).
    The DR property ensures consistency if either the OLS outcome model
    or the sieve propensity ratio is correctly specified.  The OLS
    working model for outcome regressions does not generically guarantee
    the semiparametric efficiency bound (see REGISTRY.md).

    Parameters
    ----------
    pt_assumption : str, default ``"all"``
        Parallel trends variant: ``"all"`` (overidentified, uses all
        pre-treatment periods and comparison groups) or ``"post"``
        (just-identified, single baseline, equivalent to CS).
    alpha : float, default 0.05
        Significance level.
    cluster : str or None
        Column name for cluster-robust SEs (not yet implemented —
        currently only unit-level inference).
    n_bootstrap : int, default 0
        Number of multiplier bootstrap iterations (0 = analytical only).
    bootstrap_weights : str, default ``"rademacher"``
        Bootstrap weight distribution.
    seed : int or None
        Random seed for reproducibility.
    anticipation : int, default 0
        Number of anticipation periods (shifts the effective treatment
        boundary forward by this amount).
    sieve_k_max : int or None
        Maximum polynomial degree for sieve ratio estimation. None = auto
        (``min(floor(n_gp^{1/5}), 5)``). Only used with covariates.
    sieve_criterion : str, default ``"bic"``
        Information criterion for sieve degree selection: ``"aic"`` or ``"bic"``.
    ratio_clip : float, default 20.0
        Clip sieve propensity ratios to ``[1/ratio_clip, ratio_clip]``.
    kernel_bandwidth : float or None
        Bandwidth for Gaussian kernel in conditional Omega* estimation.
        None = Silverman's rule-of-thumb (automatic).

    Examples
    --------
    >>> from diff_diff import EfficientDiD
    >>> edid = EfficientDiD(pt_assumption="all")
    >>> results = edid.fit(data, outcome="y", unit="id", time="t",
    ...                    first_treat="first_treat", aggregate="all")
    >>> results.print_summary()
    """

    def __init__(
        self,
        pt_assumption: str = "all",
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        n_bootstrap: int = 0,
        bootstrap_weights: str = "rademacher",
        seed: Optional[int] = None,
        anticipation: int = 0,
        sieve_k_max: Optional[int] = None,
        sieve_criterion: str = "bic",
        ratio_clip: float = 20.0,
        kernel_bandwidth: Optional[float] = None,
    ):
        if cluster is not None:
            raise NotImplementedError(
                "Cluster-robust SEs are not yet implemented for EfficientDiD. "
                "Use n_bootstrap > 0 for bootstrap inference instead."
            )
        self.pt_assumption = pt_assumption
        self.alpha = alpha
        self.cluster = cluster
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = seed
        self.anticipation = anticipation
        self.sieve_k_max = sieve_k_max
        self.sieve_criterion = sieve_criterion
        self.ratio_clip = ratio_clip
        self.kernel_bandwidth = kernel_bandwidth
        self.is_fitted_ = False
        self.results_: Optional[EfficientDiDResults] = None
        self._validate_params()

    def _validate_params(self) -> None:
        """Validate constrained parameters."""
        if self.pt_assumption not in ("all", "post"):
            raise ValueError(f"pt_assumption must be 'all' or 'post', got '{self.pt_assumption}'")
        valid_weights = ("rademacher", "mammen", "webb")
        if self.bootstrap_weights not in valid_weights:
            raise ValueError(
                f"bootstrap_weights must be one of {valid_weights}, "
                f"got '{self.bootstrap_weights}'"
            )
        if self.sieve_criterion not in ("aic", "bic"):
            raise ValueError(
                f"sieve_criterion must be 'aic' or 'bic', got '{self.sieve_criterion}'"
            )
        if not (np.isfinite(self.ratio_clip) and self.ratio_clip > 1.0):
            raise ValueError(f"ratio_clip must be finite and > 1.0, got {self.ratio_clip}")
        if self.kernel_bandwidth is not None:
            if not (np.isfinite(self.kernel_bandwidth) and self.kernel_bandwidth > 0):
                raise ValueError(
                    f"kernel_bandwidth must be finite and > 0 (or None for auto), "
                    f"got {self.kernel_bandwidth}"
                )
        if self.sieve_k_max is not None:
            if not (isinstance(self.sieve_k_max, (int, np.integer)) and self.sieve_k_max > 0):
                raise ValueError(
                    f"sieve_k_max must be a positive integer (or None for auto), "
                    f"got {self.sieve_k_max}"
                )

    # -- sklearn compatibility ------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters (sklearn-compatible)."""
        return {
            "pt_assumption": self.pt_assumption,
            "anticipation": self.anticipation,
            "alpha": self.alpha,
            "cluster": self.cluster,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_weights": self.bootstrap_weights,
            "seed": self.seed,
            "sieve_k_max": self.sieve_k_max,
            "sieve_criterion": self.sieve_criterion,
            "ratio_clip": self.ratio_clip,
            "kernel_bandwidth": self.kernel_bandwidth,
        }

    def set_params(self, **params: Any) -> "EfficientDiD":
        """Set estimator parameters (sklearn-compatible)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        self._validate_params()
        return self

    # -- Main estimation ------------------------------------------------------

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
    ) -> EfficientDiDResults:
        """Fit the Efficient DiD estimator.

        Parameters
        ----------
        data : DataFrame
            Balanced panel data.
        outcome : str
            Outcome variable column name.
        unit : str
            Unit identifier column name.
        time : str
            Time period column name.
        first_treat : str
            Column indicating first treatment period.
            Use 0 or ``np.inf`` for never-treated units.
        covariates : list of str, optional
            Column names for time-invariant unit-level covariates.
            When provided, uses the doubly robust path (outcome regression
            + propensity score ratios).
        aggregate : str, optional
            ``None``, ``"simple"``, ``"event_study"``, ``"group"``, or
            ``"all"``.
        balance_e : int, optional
            Balance event study at this relative period.

        Returns
        -------
        EfficientDiDResults

        Raises
        ------
        ValueError
            Missing columns, unbalanced panel, non-absorbing treatment,
            or PT-Post without a never-treated group.
        """
        self._validate_params()

        # Normalize empty covariates list to None (use nocov path)
        if covariates is not None and len(covariates) == 0:
            covariates = None
        use_covariates = covariates is not None

        # ----- Validate inputs -----
        required_cols = [outcome, unit, time, first_treat]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = data.copy()
        df[time] = pd.to_numeric(df[time])
        df[first_treat] = pd.to_numeric(df[first_treat])

        # Normalize never-treated: inf -> 0 internally, keep track
        df["_never_treated"] = (df[first_treat] == 0) | (df[first_treat] == np.inf)
        df.loc[df[first_treat] == np.inf, first_treat] = 0

        time_periods = sorted(df[time].unique())
        treatment_groups = sorted([g for g in df[first_treat].unique() if g > 0])

        # Validate balanced panel
        unit_period_counts = df.groupby(unit)[time].nunique()
        n_periods = len(time_periods)
        if (unit_period_counts != n_periods).any():
            raise ValueError(
                "Unbalanced panel detected. EfficientDiD requires a balanced "
                "panel where every unit is observed in every time period."
            )

        # Reject non-finite outcomes (NaN/Inf corrupt Omega*/EIF calculations)
        non_finite_mask = ~np.isfinite(df[outcome])
        if non_finite_mask.any():
            n_bad = int(non_finite_mask.sum())
            raise ValueError(
                f"Found {n_bad} non-finite value(s) in outcome column '{outcome}'. "
                "EfficientDiD requires finite outcomes for all unit-period observations."
            )

        # Reject duplicate (unit, time) rows
        dup_mask = df.duplicated(subset=[unit, time], keep=False)
        if dup_mask.any():
            n_dups = int(dup_mask.sum())
            raise ValueError(
                f"Found {n_dups} duplicate ({unit}, {time}) rows. "
                "EfficientDiD requires exactly one observation per unit-period."
            )

        # Validate absorbing treatment (vectorized)
        ft_nunique = df.groupby(unit)[first_treat].nunique()
        bad_units = ft_nunique[ft_nunique > 1]
        if len(bad_units) > 0:
            uid = bad_units.index[0]
            raise ValueError(
                f"Non-absorbing treatment detected for unit {uid}: "
                "first_treat value changes over time."
            )

        # Unit info
        unit_info = (
            df.groupby(unit)
            .agg(
                {
                    first_treat: "first",
                    "_never_treated": "first",
                }
            )
            .reset_index()
        )
        n_treated_units = int((unit_info[first_treat] > 0).sum())
        n_control_units = int(unit_info["_never_treated"].sum())

        # Check for never-treated units — required for generated outcomes
        # (the formula's second term mean(Y_t - Y_{t_pre} | G=inf) needs G=inf)
        if n_control_units == 0:
            raise ValueError(
                "No never-treated units found. EfficientDiD Phase 1 requires a "
                "never-treated comparison group. The 'last cohort as control' "
                "fallback will be added in a future version."
            )

        # ----- Prepare data -----
        all_units = sorted(df[unit].unique())
        n_units = len(all_units)

        period_to_col = {p: i for i, p in enumerate(time_periods)}
        period_1 = time_periods[0]
        period_1_col = period_to_col[period_1]

        # Pivot outcome to wide matrix (n_units, n_periods)
        pivot = df.pivot(index=unit, columns=time, values=outcome)
        # Reindex to match all_units ordering and time_periods column order
        pivot = pivot.reindex(index=all_units, columns=time_periods)
        outcome_wide = pivot.values.astype(float)

        # Build cohort masks and fractions
        unit_info_indexed = unit_info.set_index(unit)
        unit_cohorts = unit_info_indexed.reindex(all_units)[first_treat].values.astype(
            float
        )  # 0 = never-treated

        cohort_masks: Dict[float, np.ndarray] = {}
        for g in treatment_groups:
            cohort_masks[g] = unit_cohorts == g
        never_treated_mask = unit_cohorts == 0
        cohort_masks[np.inf] = never_treated_mask  # also keyed by inf sentinel

        cohort_fractions: Dict[float, float] = {}
        for g in treatment_groups:
            cohort_fractions[g] = float(np.sum(cohort_masks[g])) / n_units
        cohort_fractions[np.inf] = float(np.sum(never_treated_mask)) / n_units

        # ----- Covariate preparation (if provided) -----
        covariate_matrix: Optional[np.ndarray] = None
        m_hat_cache: Dict[Tuple, np.ndarray] = {}
        r_hat_cache: Dict[Tuple[float, float], np.ndarray] = {}
        s_hat_cache: Dict[float, np.ndarray] = {}  # inverse propensities per group

        if use_covariates:
            assert covariates is not None  # for type narrowing

            # Validate covariate columns exist
            missing_cov = [c for c in covariates if c not in data.columns]
            if missing_cov:
                raise ValueError(f"Missing covariate columns: {missing_cov}")

            # Validate no NaN/Inf in covariates
            for col_name in covariates:
                non_finite_cov = ~np.isfinite(pd.to_numeric(df[col_name], errors="coerce"))
                if non_finite_cov.any():
                    n_bad = int(non_finite_cov.sum())
                    raise ValueError(
                        f"Found {n_bad} non-finite value(s) in covariate column "
                        f"'{col_name}'. Covariates must be finite."
                    )

            # Validate time-invariance: covariates must be constant within each unit
            for col_name in covariates:
                cov_nunique = df.groupby(unit)[col_name].nunique()
                varying = cov_nunique[cov_nunique > 1]
                if len(varying) > 0:
                    uid = varying.index[0]
                    raise ValueError(
                        f"Covariate '{col_name}' varies over time for unit {uid}. "
                        "EfficientDiD requires time-invariant covariates. "
                        "Extract base-period values before calling fit()."
                    )

            # Extract unit-level covariate matrix from period_1 observations
            base_df = df[df[time] == period_1].set_index(unit).reindex(all_units)
            covariate_matrix = base_df[list(covariates)].values.astype(float)

        # ----- Core estimation: ATT(g, t) for each target -----
        # Precompute per-group unit counts (avoid repeated np.sum in loop)
        n_treated_per_g = {g: int(np.sum(cohort_masks[g])) for g in treatment_groups}
        n_control_count = int(np.sum(never_treated_mask))

        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
        eif_by_gt: Dict[Tuple[Any, Any], np.ndarray] = {}
        stored_weights: Dict[Tuple[Any, Any], np.ndarray] = {}
        stored_cond: Dict[Tuple[Any, Any], float] = {}

        for g in treatment_groups:
            # Under PT-Post, use per-group baseline Y_{g-1-anticipation}
            # instead of the universal Y_1.  This implements the weaker
            # PT-Post assumption (parallel trends only from g-1 onward),
            # matching the Callaway-Sant'Anna estimator exactly.
            if self.pt_assumption == "post":
                effective_base = g - 1 - self.anticipation
                if effective_base not in period_to_col:
                    warnings.warn(
                        f"Cohort g={g} dropped: baseline period {effective_base} "
                        f"(g-1-anticipation) is not in the data.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                effective_p1_col = period_to_col[effective_base]
            else:
                effective_p1_col = period_1_col

            # Estimate all (g, t) cells including pre-treatment. Under PT-Post,
            # pre-treatment cells serve as placebo/pre-trend diagnostics, matching
            # the CallawaySantAnna implementation. Users filter to t >= g for
            # post-treatment effects; pre-treatment cells are clearly labeled by
            # their (g, t) coordinates in the results object.
            for t in time_periods:
                # Skip period_1 — it's the universal reference baseline,
                # not a target period
                if t == period_1:
                    continue

                # Enumerate valid comparison pairs
                pairs = enumerate_valid_triples(
                    target_g=g,
                    treatment_groups=treatment_groups,
                    time_periods=time_periods,
                    period_1=period_1,
                    pt_assumption=self.pt_assumption,
                    anticipation=self.anticipation,
                )

                if not pairs:
                    warnings.warn(
                        f"No valid comparison pairs for (g={g}, t={t}). " "ATT will be NaN.",
                        UserWarning,
                        stacklevel=2,
                    )
                    t_stat, p_val, ci = np.nan, np.nan, (np.nan, np.nan)
                    group_time_effects[(g, t)] = {
                        "effect": np.nan,
                        "se": np.nan,
                        "t_stat": t_stat,
                        "p_value": p_val,
                        "conf_int": ci,
                        "n_treated": n_treated_per_g[g],
                        "n_control": n_control_count,
                    }
                    eif_by_gt[(g, t)] = np.zeros(n_units)
                    continue

                if use_covariates:
                    assert covariate_matrix is not None
                    t_col_val = period_to_col[t]

                    # Lazily populate nuisance caches for this (g, t)
                    for gp, tpre in pairs:
                        tpre_col_val = period_to_col[tpre]
                        # m_{inf, t, tpre}(X)
                        key_inf_t = (np.inf, t_col_val, tpre_col_val)
                        if key_inf_t not in m_hat_cache:
                            m_hat_cache[key_inf_t] = estimate_outcome_regression(
                                outcome_wide,
                                covariate_matrix,
                                never_treated_mask,
                                t_col_val,
                                tpre_col_val,
                            )
                        # m_{g', tpre, 1}(X)
                        key_gp_tpre = (gp, tpre_col_val, effective_p1_col)
                        if key_gp_tpre not in m_hat_cache:
                            gp_mask_for_reg = (
                                never_treated_mask if np.isinf(gp) else cohort_masks[gp]
                            )
                            m_hat_cache[key_gp_tpre] = estimate_outcome_regression(
                                outcome_wide,
                                covariate_matrix,
                                gp_mask_for_reg,
                                tpre_col_val,
                                effective_p1_col,
                            )
                        # r_{g, inf}(X) and r_{g, g'}(X) via sieve (Eq 4.1-4.2)
                        for comp in {np.inf, gp}:
                            rkey = (g, comp)
                            if rkey not in r_hat_cache:
                                comp_mask = (
                                    never_treated_mask if np.isinf(comp) else cohort_masks[comp]
                                )
                                r_hat_cache[rkey] = estimate_propensity_ratio_sieve(
                                    covariate_matrix,
                                    cohort_masks[g],
                                    comp_mask,
                                    k_max=self.sieve_k_max,
                                    criterion=self.sieve_criterion,
                                    ratio_clip=self.ratio_clip,
                                )

                    # Per-unit DR generated outcomes: shape (n_units, H)
                    gen_out = compute_generated_outcomes_cov(
                        target_g=g,
                        target_t=t,
                        valid_pairs=pairs,
                        outcome_wide=outcome_wide,
                        cohort_masks=cohort_masks,
                        never_treated_mask=never_treated_mask,
                        period_to_col=period_to_col,
                        period_1_col=effective_p1_col,
                        cohort_fractions=cohort_fractions,
                        m_hat_cache=m_hat_cache,
                        r_hat_cache=r_hat_cache,
                    )

                    y_hat = np.mean(gen_out, axis=0)  # shape (H,)

                    # Inverse propensity estimation (algorithm step 4)
                    # s_hat_{g'}(X) = 1/p_{g'}(X) for Eq 3.12 scaling
                    for group_id in {g, np.inf} | {gp for gp, _ in pairs}:
                        if group_id not in s_hat_cache:
                            group_mask_s = (
                                never_treated_mask if np.isinf(group_id) else cohort_masks[group_id]
                            )
                            s_hat_cache[group_id] = estimate_inverse_propensity_sieve(
                                covariate_matrix,
                                group_mask_s,
                                k_max=self.sieve_k_max,
                                criterion=self.sieve_criterion,
                            )

                    # Conditional Omega*(X) with per-unit propensities (Eq 3.12)
                    omega_cond = compute_omega_star_conditional(
                        target_g=g,
                        target_t=t,
                        valid_pairs=pairs,
                        outcome_wide=outcome_wide,
                        cohort_masks=cohort_masks,
                        never_treated_mask=never_treated_mask,
                        period_to_col=period_to_col,
                        period_1_col=effective_p1_col,
                        cohort_fractions=cohort_fractions,
                        covariate_matrix=covariate_matrix,
                        s_hat_cache=s_hat_cache,
                        bandwidth=self.kernel_bandwidth,
                    )

                    # Per-unit weights: (n_units, H)
                    per_unit_w = compute_per_unit_weights(omega_cond)

                    # ATT = mean_i( w(X_i) @ gen_out[i] )
                    if per_unit_w.shape[1] > 0:
                        att_gt = float(np.mean(np.sum(per_unit_w * gen_out, axis=1)))
                    else:
                        att_gt = np.nan

                    # EIF with per-unit weights (Remark 4.2: plug-in valid)
                    # Center on scalar ATT, not per-pair means (ensures mean(EIF) ≈ 0)
                    eif_vals = compute_eif_cov(per_unit_w, gen_out, att_gt, n_units)
                    eif_by_gt[(g, t)] = eif_vals
                else:
                    # No-covariates path (closed-form)
                    omega = compute_omega_star_nocov(
                        target_g=g,
                        target_t=t,
                        valid_pairs=pairs,
                        outcome_wide=outcome_wide,
                        cohort_masks=cohort_masks,
                        never_treated_mask=never_treated_mask,
                        period_to_col=period_to_col,
                        period_1_col=effective_p1_col,
                        cohort_fractions=cohort_fractions,
                    )

                    weights, _, cond_num = compute_efficient_weights(omega)
                    stored_weights[(g, t)] = weights
                    if omega.size > 0:
                        stored_cond[(g, t)] = cond_num

                    y_hat = compute_generated_outcomes_nocov(
                        target_g=g,
                        target_t=t,
                        valid_pairs=pairs,
                        outcome_wide=outcome_wide,
                        cohort_masks=cohort_masks,
                        never_treated_mask=never_treated_mask,
                        period_to_col=period_to_col,
                        period_1_col=effective_p1_col,
                    )

                    att_gt = float(weights @ y_hat) if len(weights) > 0 else np.nan

                    eif_vals = compute_eif_nocov(
                        target_g=g,
                        target_t=t,
                        weights=weights,
                        valid_pairs=pairs,
                        outcome_wide=outcome_wide,
                        cohort_masks=cohort_masks,
                        never_treated_mask=never_treated_mask,
                        period_to_col=period_to_col,
                        period_1_col=effective_p1_col,
                        cohort_fractions=cohort_fractions,
                        n_units=n_units,
                    )
                    eif_by_gt[(g, t)] = eif_vals

                # Analytical SE = sqrt(mean(EIF^2) / n)  [paper p.21]
                se_gt = float(np.sqrt(np.mean(eif_vals**2) / n_units))

                t_stat, p_val, ci = safe_inference(att_gt, se_gt, alpha=self.alpha)

                group_time_effects[(g, t)] = {
                    "effect": att_gt,
                    "se": se_gt,
                    "t_stat": t_stat,
                    "p_value": p_val,
                    "conf_int": ci,
                    "n_treated": int(np.sum(cohort_masks[g])),
                    "n_control": int(np.sum(never_treated_mask)),
                }

        if not group_time_effects:
            raise ValueError(
                "Could not estimate any group-time effects. "
                "Check data has sufficient observations."
            )

        # ----- Aggregation -----
        overall_att, overall_se = self._aggregate_overall(
            group_time_effects, eif_by_gt, n_units, cohort_fractions, unit_cohorts
        )
        overall_t, overall_p, overall_ci = safe_inference(overall_att, overall_se, alpha=self.alpha)

        event_study_effects = None
        group_effects = None

        if aggregate in ("event_study", "all"):
            event_study_effects = self._aggregate_event_study(
                group_time_effects,
                eif_by_gt,
                n_units,
                cohort_fractions,
                treatment_groups,
                time_periods,
                balance_e,
                unit_cohorts=unit_cohorts,
            )
        if aggregate in ("group", "all"):
            group_effects = self._aggregate_by_group(
                group_time_effects,
                eif_by_gt,
                n_units,
                cohort_fractions,
                treatment_groups,
                unit_cohorts=unit_cohorts,
            )

        # ----- Bootstrap -----
        bootstrap_results = None
        if self.n_bootstrap > 0 and eif_by_gt:
            bootstrap_results = self._run_multiplier_bootstrap(
                group_time_effects=group_time_effects,
                eif_by_gt=eif_by_gt,
                n_units=n_units,
                aggregate=aggregate,
                balance_e=balance_e,
                treatment_groups=treatment_groups,
                cohort_fractions=cohort_fractions,
            )
            # Update estimates with bootstrap inference
            overall_se = bootstrap_results.overall_att_se
            overall_t = safe_inference(overall_att, overall_se, alpha=self.alpha)[0]
            overall_p = bootstrap_results.overall_att_p_value
            overall_ci = bootstrap_results.overall_att_ci

            for gt in group_time_effects:
                if gt in bootstrap_results.group_time_ses:
                    group_time_effects[gt]["se"] = bootstrap_results.group_time_ses[gt]
                    group_time_effects[gt]["conf_int"] = bootstrap_results.group_time_cis[gt]
                    group_time_effects[gt]["p_value"] = bootstrap_results.group_time_p_values[gt]
                    eff = float(group_time_effects[gt]["effect"])
                    se = float(group_time_effects[gt]["se"])
                    group_time_effects[gt]["t_stat"] = safe_inference(eff, se, alpha=self.alpha)[0]

            es_cis = bootstrap_results.event_study_cis
            es_pvs = bootstrap_results.event_study_p_values
            if (
                event_study_effects is not None
                and bootstrap_results.event_study_ses is not None
                and es_cis is not None
                and es_pvs is not None
            ):
                for e in event_study_effects:
                    if e in bootstrap_results.event_study_ses:
                        event_study_effects[e]["se"] = bootstrap_results.event_study_ses[e]
                        event_study_effects[e]["conf_int"] = es_cis[e]
                        event_study_effects[e]["p_value"] = es_pvs[e]
                        eff = float(event_study_effects[e]["effect"])
                        se = float(event_study_effects[e]["se"])
                        event_study_effects[e]["t_stat"] = safe_inference(
                            eff, se, alpha=self.alpha
                        )[0]

            g_cis = bootstrap_results.group_effect_cis
            g_pvs = bootstrap_results.group_effect_p_values
            if (
                group_effects is not None
                and bootstrap_results.group_effect_ses is not None
                and g_cis is not None
                and g_pvs is not None
            ):
                for g in group_effects:
                    if g in bootstrap_results.group_effect_ses:
                        group_effects[g]["se"] = bootstrap_results.group_effect_ses[g]
                        group_effects[g]["conf_int"] = g_cis[g]
                        group_effects[g]["p_value"] = g_pvs[g]
                        eff = float(group_effects[g]["effect"])
                        se = float(group_effects[g]["se"])
                        group_effects[g]["t_stat"] = safe_inference(eff, se, alpha=self.alpha)[0]

        # ----- Build results -----
        self.results_ = EfficientDiDResults(
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
            pt_assumption=self.pt_assumption,
            anticipation=self.anticipation,
            n_bootstrap=self.n_bootstrap,
            bootstrap_weights=self.bootstrap_weights,
            seed=self.seed,
            event_study_effects=event_study_effects,
            group_effects=group_effects,
            efficient_weights=stored_weights if stored_weights else None,
            omega_condition_numbers=stored_cond if stored_cond else None,
            influence_functions=None,  # can store full EIF matrix if needed
            bootstrap_results=bootstrap_results,
            estimation_path="dr" if use_covariates else "nocov",
            sieve_k_max=self.sieve_k_max,
            sieve_criterion=self.sieve_criterion,
            ratio_clip=self.ratio_clip,
            kernel_bandwidth=self.kernel_bandwidth,
        )
        self.is_fitted_ = True
        return self.results_

    # -- Aggregation helpers --------------------------------------------------

    def _compute_wif_contribution(
        self,
        keepers: List[Tuple],
        effects: np.ndarray,
        unit_cohorts: np.ndarray,
        cohort_fractions: Dict[float, float],
        n_units: int,
    ) -> np.ndarray:
        """Compute weight influence function correction (O(1) scale, matching EIF).

        This accounts for uncertainty in cohort-size aggregation weights.
        Matches R's ``did`` package WIF formula (staggered_aggregation.py:282-309),
        adapted to EDiD's EIF scale.

        Parameters
        ----------
        keepers : list of (g, t) tuples
            Post-treatment group-time pairs included in aggregation.
        effects : ndarray, shape (n_keepers,)
            ATT estimates for each keeper.
        unit_cohorts : ndarray, shape (n_units,)
            Cohort assignment for each unit (0 = never-treated).
        cohort_fractions : dict
            ``{cohort: n_cohort / n}`` for each cohort.
        n_units : int
            Total number of units.

        Returns
        -------
        ndarray, shape (n_units,)
            WIF contribution at O(1) scale, additive with ``agg_eif``.
        """
        groups_for_keepers = np.array([g for (g, t) in keepers])
        pg_keepers = np.array([cohort_fractions.get(g, 0.0) for g, t in keepers])
        sum_pg = pg_keepers.sum()
        if sum_pg == 0:
            return np.zeros(n_units)

        indicator = (unit_cohorts[:, None] == groups_for_keepers[None, :]).astype(float)
        indicator_sum = np.sum(indicator - pg_keepers, axis=1)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if1 = (indicator - pg_keepers) / sum_pg
            if2 = np.outer(indicator_sum, pg_keepers) / sum_pg**2
            wif_matrix = if1 - if2
            wif_contrib = wif_matrix @ effects
        return wif_contrib  # O(1) scale, same as agg_eif

    def _aggregate_overall(
        self,
        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]],
        eif_by_gt: Dict[Tuple[Any, Any], np.ndarray],
        n_units: int,
        cohort_fractions: Dict[float, float],
        unit_cohorts: np.ndarray,
    ) -> Tuple[float, float]:
        """Compute overall ATT with WIF-adjusted SE.

        Parameters
        ----------
        group_time_effects : dict
            Group-time ATT estimates.
        eif_by_gt : dict
            Per-unit EIF values for each (g, t).
        n_units : int
            Total number of units.
        cohort_fractions : dict
            Cohort size fractions.
        unit_cohorts : ndarray, shape (n_units,)
            Cohort assignment for each unit.
        """
        # Filter to post-treatment effects
        keepers = [
            (g, t)
            for (g, t) in group_time_effects
            if t >= g - self.anticipation and np.isfinite(group_time_effects[(g, t)]["effect"])
        ]
        if not keepers:
            return np.nan, np.nan

        # Cohort-size weights
        pg = np.array([cohort_fractions.get(g, 0.0) for (g, _) in keepers])
        total_pg = pg.sum()
        if total_pg == 0:
            return np.nan, np.nan
        w = pg / total_pg

        effects = np.array([group_time_effects[gt]["effect"] for gt in keepers])
        overall_att = float(np.sum(w * effects))

        # Aggregate EIF
        agg_eif = np.zeros(n_units)
        for k, gt in enumerate(keepers):
            agg_eif += w[k] * eif_by_gt[gt]

        # WIF correction: accounts for uncertainty in cohort-size weights
        wif = self._compute_wif_contribution(
            keepers, effects, unit_cohorts, cohort_fractions, n_units
        )
        agg_eif_total = agg_eif + wif  # both O(1) scale

        # SE = sqrt(mean(EIF^2) / n) — standard IF-based SE
        se = float(np.sqrt(np.mean(agg_eif_total**2) / n_units))

        return overall_att, se

    def _aggregate_event_study(
        self,
        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]],
        eif_by_gt: Dict[Tuple[Any, Any], np.ndarray],
        n_units: int,
        cohort_fractions: Dict[float, float],
        treatment_groups: List[Any],
        time_periods: List[Any],
        balance_e: Optional[int] = None,
        unit_cohorts: Optional[np.ndarray] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Aggregate ATT(g,t) by relative time e = t - g.

        Parameters
        ----------
        group_time_effects : dict
            Group-time ATT estimates.
        eif_by_gt : dict
            Per-unit EIF values for each (g, t).
        n_units : int
            Total number of units.
        cohort_fractions : dict
            Cohort size fractions.
        treatment_groups : list
            Treatment cohort identifiers.
        time_periods : list
            All time periods.
        balance_e : int, optional
            Balance event study at this relative period.
        unit_cohorts : ndarray, optional
            Cohort assignment for each unit (for WIF correction).
        """
        # Organize by relative time
        effects_by_e: Dict[int, List[Tuple[Tuple[Any, Any], float, float]]] = {}
        for (g, t), data in group_time_effects.items():
            if not np.isfinite(data["effect"]):
                continue
            e = int(t - g)
            if e not in effects_by_e:
                effects_by_e[e] = []
            effects_by_e[e].append(((g, t), data["effect"], cohort_fractions.get(g, 0.0)))

        # Balance if requested
        if balance_e is not None:
            groups_at_e = {gt[0] for gt, _, _ in effects_by_e.get(balance_e, [])}
            balanced: Dict[int, List[Tuple[Tuple[Any, Any], float, float]]] = {}
            for (g, t), data in group_time_effects.items():
                if not np.isfinite(data["effect"]):
                    continue
                if g in groups_at_e:
                    e = int(t - g)
                    if e not in balanced:
                        balanced[e] = []
                    balanced[e].append(((g, t), data["effect"], cohort_fractions.get(g, 0.0)))
            effects_by_e = balanced

        if balance_e is not None and not effects_by_e:
            warnings.warn(
                f"balance_e={balance_e}: no cohort has a finite effect at the "
                "anchor horizon. Event study will be empty.",
                UserWarning,
                stacklevel=2,
            )

        result: Dict[int, Dict[str, Any]] = {}
        for e, elist in sorted(effects_by_e.items()):
            gt_pairs = [x[0] for x in elist]
            effs = np.array([x[1] for x in elist])
            pgs = np.array([x[2] for x in elist])
            total_pg = pgs.sum()
            w = pgs / total_pg if total_pg > 0 else np.ones(len(pgs)) / len(pgs)

            agg_eff = float(np.sum(w * effs))

            # Aggregate EIF
            agg_eif = np.zeros(n_units)
            for k, gt in enumerate(gt_pairs):
                agg_eif += w[k] * eif_by_gt[gt]

            # WIF correction for event-study aggregation
            if unit_cohorts is not None:
                es_keepers = [(g, t) for (g, t) in gt_pairs]
                es_effects = effs
                wif = self._compute_wif_contribution(
                    es_keepers, es_effects, unit_cohorts, cohort_fractions, n_units
                )
                agg_eif = agg_eif + wif

            agg_se = float(np.sqrt(np.mean(agg_eif**2) / n_units))

            t_stat, p_val, ci = safe_inference(agg_eff, agg_se, alpha=self.alpha)
            result[e] = {
                "effect": agg_eff,
                "se": agg_se,
                "t_stat": t_stat,
                "p_value": p_val,
                "conf_int": ci,
                "n_groups": len(elist),
            }

        return result

    def _aggregate_by_group(
        self,
        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]],
        eif_by_gt: Dict[Tuple[Any, Any], np.ndarray],
        n_units: int,
        cohort_fractions: Dict[float, float],
        treatment_groups: List[Any],
        unit_cohorts: Optional[np.ndarray] = None,
    ) -> Dict[Any, Dict[str, Any]]:
        """Aggregate ATT(g,t) by treatment cohort.

        Parameters
        ----------
        group_time_effects : dict
            Group-time ATT estimates.
        eif_by_gt : dict
            Per-unit EIF values for each (g, t).
        n_units : int
            Total number of units.
        cohort_fractions : dict
            Cohort size fractions.
        treatment_groups : list
            Treatment cohort identifiers.
        unit_cohorts : ndarray, optional
            Cohort assignment for each unit (unused — group aggregation
            uses equal weights, not cohort-size weights).
        """
        result: Dict[Any, Dict[str, Any]] = {}
        for g in treatment_groups:
            g_gts = [
                (gg, t)
                for (gg, t) in group_time_effects
                if gg == g
                and t >= g - self.anticipation
                and np.isfinite(group_time_effects[(gg, t)]["effect"])
            ]
            if not g_gts:
                continue

            effs = np.array([group_time_effects[gt]["effect"] for gt in g_gts])
            w = np.ones(len(effs)) / len(effs)
            agg_eff = float(np.sum(w * effs))

            agg_eif = np.zeros(n_units)
            for k, gt in enumerate(g_gts):
                agg_eif += w[k] * eif_by_gt[gt]
            agg_se = float(np.sqrt(np.mean(agg_eif**2) / n_units))

            t_stat, p_val, ci = safe_inference(agg_eff, agg_se, alpha=self.alpha)
            result[g] = {
                "effect": agg_eff,
                "se": agg_se,
                "t_stat": t_stat,
                "p_value": p_val,
                "conf_int": ci,
                "n_periods": len(g_gts),
            }

        return result

    def summary(self) -> str:
        """Get summary of estimation results."""
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted before calling summary()")
        assert self.results_ is not None
        return self.results_.summary()

    def print_summary(self) -> None:
        """Print summary to stdout."""
        print(self.summary())
