"""WooldridgeDiD: Extended Two-Way Fixed Effects (ETWFE) estimator.

Implements Wooldridge (2025, 2023) ETWFE, faithful to the Stata jwdid package.

References
----------
Wooldridge (2025). Two-Way Fixed Effects, the Two-Way Mundlak Regression,
  and Difference-in-Differences Estimators. Empirical Economics, 69(5), 2545-2587.
Wooldridge (2023). Simple approaches to nonlinear difference-in-differences
  with panel data. The Econometrics Journal, 26(3), C31-C66.
Friosavila (2021). jwdid: Stata module. SSC s459114.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.linalg import compute_robust_vcov, solve_logit, solve_ols, solve_poisson
from diff_diff.utils import safe_inference, within_transform
from diff_diff.wooldridge_results import WooldridgeDiDResults

_VALID_METHODS = ("ols", "logit", "poisson")
_VALID_CONTROL_GROUPS = ("never_treated", "not_yet_treated")
_VALID_BOOTSTRAP_WEIGHTS = ("rademacher", "webb", "mammen")


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logistic_deriv(x: np.ndarray) -> np.ndarray:
    p = _logistic(x)
    return p * (1.0 - p)


def _compute_weighted_agg(
    gt_effects: Dict,
    gt_weights: Dict,
    gt_keys: List,
    gt_vcov: Optional[np.ndarray],
    alpha: float,
    df: Optional[int] = None,
) -> Dict:
    """Compute simple (overall) weighted average ATT and SE via delta method."""
    post_keys = [(g, t) for (g, t) in gt_keys if t >= g]
    w_total = sum(gt_weights.get(k, 0) for k in post_keys)
    if w_total == 0:
        att = float("nan")
        se = float("nan")
    else:
        att = (
            sum(gt_weights.get(k, 0) * gt_effects[k]["att"] for k in post_keys if k in gt_effects)
            / w_total
        )
        if gt_vcov is not None:
            w_vec = np.array(
                [gt_weights.get(k, 0) / w_total if k in post_keys else 0.0 for k in gt_keys]
            )
            var = float(w_vec @ gt_vcov @ w_vec)
            se = float(np.sqrt(max(var, 0.0)))
        else:
            se = float("nan")

    t_stat, p_value, conf_int = safe_inference(att, se, alpha=alpha, df=df)
    return {"att": att, "se": se, "t_stat": t_stat, "p_value": p_value, "conf_int": conf_int}


def _resolve_survey_for_wooldridge(survey_design, sample, cluster_ids, cluster_name):
    """Resolve survey design, inject cluster as PSU, recompute metadata.

    Shared helper for all three WooldridgeDiD sub-fitters.  Matches the
    resolution chain in DifferenceInDifferences.fit() (estimators.py:344-359).
    """
    from diff_diff.survey import (
        _resolve_survey_for_fit,
        _resolve_effective_cluster,
        _inject_cluster_as_psu,
        compute_survey_metadata,
    )

    resolved, survey_weights, survey_weight_type, survey_metadata = (
        _resolve_survey_for_fit(survey_design, sample)
    )
    if resolved is not None and resolved.uses_replicate_variance:
        raise NotImplementedError(
            "WooldridgeDiD does not yet support replicate-weight variance. "
            "Use TSL (strata/PSU/FPC) instead."
        )
    if resolved is not None and resolved.weight_type != "pweight":
        raise ValueError(
            f"WooldridgeDiD survey support requires weight_type='pweight', "
            f"got '{resolved.weight_type}'. The survey variance math "
            f"assumes probability weights (pweight)."
        )
    if resolved is not None:
        effective_cluster = _resolve_effective_cluster(
            resolved, cluster_ids, cluster_name
        )
        if effective_cluster is not None:
            resolved = _inject_cluster_as_psu(resolved, effective_cluster)
            if resolved.psu is not None and survey_metadata is not None:
                raw_w = (
                    sample[survey_design.weights].values.astype(np.float64)
                    if survey_design.weights
                    else np.ones(len(sample), dtype=np.float64)
                )
                survey_metadata = compute_survey_metadata(resolved, raw_w)
    df_inf = resolved.df_survey if resolved is not None else None
    return resolved, survey_weights, survey_weight_type, survey_metadata, df_inf


def _filter_sample(
    data: pd.DataFrame,
    unit: str,
    time: str,
    cohort: str,
    control_group: str,
    anticipation: int,
) -> pd.DataFrame:
    """Return the analysis sample following jwdid selection rules.

    All treated units keep ALL observations (pre- and post-treatment) for
    proper FE estimation. The control_group setting affects which additional
    control observations are included, AND the interaction matrix structure
    (see _build_interaction_matrix).
    """
    df = data.copy()
    # Normalise never-treated: fill NaN cohort with 0
    df[cohort] = df[cohort].fillna(0)

    treated_mask = df[cohort] > 0

    if control_group == "never_treated":
        control_mask = df[cohort] == 0
    else:  # not_yet_treated
        # Keep untreated-at-t observations for not-yet-treated units
        control_mask = (df[cohort] == 0) | (df[cohort] > df[time])

    return df[treated_mask | control_mask].copy()


def _build_interaction_matrix(
    data: pd.DataFrame,
    cohort: str,
    time: str,
    anticipation: int,
    control_group: str = "not_yet_treated",
    method: str = "ols",
) -> Tuple[np.ndarray, List[str], List[Tuple[Any, Any]]]:
    """Build the saturated cohort×time interaction design matrix.

    For ``not_yet_treated``: only post-treatment cells (t >= g - anticipation).
    Pre-treatment obs from treated units sit in the regression baseline alongside
    not-yet-treated controls.

    For ``never_treated`` + OLS: ALL (g, t) pairs for each treated cohort. This
    "absorbs" pre-treatment obs from treated units into their own indicators so
    they do not serve as implicit controls in the baseline. Only never-treated
    observations remain in the omitted category. Pre-treatment coefficients
    (t < g) serve as placebo/pre-trend tests.

    For ``never_treated`` + nonlinear (logit/Poisson): post-treatment cells only.
    Nonlinear paths use explicit cohort + time dummies (not within-transformation),
    so including all (g, t) cells would create exact collinearity between each
    cohort dummy and the sum of its cell indicators.

    Returns
    -------
    X_int : (n, n_cells) binary indicator matrix
    col_names : list of string labels "g{g}_t{t}"
    gt_keys : list of (g, t) tuples in same column order
    """
    groups = sorted(g for g in data[cohort].unique() if g > 0)
    times = sorted(data[time].unique())
    cohort_vals = data[cohort].values
    time_vals = data[time].values

    # OLS + never_treated: all (g,t) pairs (placebo via within-transform FE)
    # Nonlinear + never_treated: post-treatment only (avoids cohort dummy collinearity)
    # not_yet_treated: post-treatment only (always)
    include_pre = control_group == "never_treated" and method == "ols"

    cols = []
    col_names = []
    gt_keys = []

    for g in groups:
        for t in times:
            if include_pre or t >= g - anticipation:
                indicator = ((cohort_vals == g) & (time_vals == t)).astype(float)
                cols.append(indicator)
                col_names.append(f"g{g}_t{t}")
                gt_keys.append((g, t))

    if not cols:
        return np.empty((len(data), 0)), [], []
    return np.column_stack(cols), col_names, gt_keys


def _prepare_covariates(
    data: pd.DataFrame,
    exovar: Optional[List[str]],
    xtvar: Optional[List[str]],
    xgvar: Optional[List[str]],
    cohort: str,
    time: str,
    demean_covariates: bool,
    groups: List[Any],
) -> Optional[np.ndarray]:
    """Build covariate matrix following jwdid covariate type conventions.

    Returns None if no covariates, else (n, k) array.
    """
    parts = []

    if exovar:
        parts.append(data[exovar].values.astype(float))

    if xtvar:
        if demean_covariates:
            # Within-cohort×period demeaning
            grp_key = data[cohort].astype(str) + "_" + data[time].astype(str)
            tmp = data[xtvar].copy()
            for col in xtvar:
                tmp[col] = tmp[col] - tmp.groupby(grp_key)[col].transform("mean")
            parts.append(tmp.values.astype(float))
        else:
            parts.append(data[xtvar].values.astype(float))

    if xgvar:
        for g in groups:
            g_indicator = (data[cohort] == g).values.astype(float)
            for col in xgvar:
                parts.append((g_indicator * data[col].values).reshape(-1, 1))

    if not parts:
        return None
    return np.hstack([p if p.ndim == 2 else p.reshape(-1, 1) for p in parts])


class WooldridgeDiD:
    """Extended Two-Way Fixed Effects (ETWFE) DiD estimator.

    Implements the Wooldridge (2021) saturated cohort×time regression and
    Wooldridge (2023) nonlinear extensions (logit, Poisson).  Produces all
    four ``jwdid_estat`` aggregation types: simple, group, calendar, event.

    Parameters
    ----------
    method : {"ols", "logit", "poisson"}
        Estimation method. "ols" for continuous outcomes; "logit" for binary
        or fractional outcomes; "poisson" for count data.
    control_group : {"not_yet_treated", "never_treated"}
        Which units serve as the comparison group.  "not_yet_treated" (jwdid
        default) uses all untreated observations at each time period;
        "never_treated" uses only units never treated throughout the sample.
    anticipation : int
        Number of periods before treatment onset to include as treatment cells
        (anticipation effects).  0 means no anticipation.
    demean_covariates : bool
        If True (jwdid default), ``xtvar`` covariates are demeaned within each
        cohort×period cell before entering the regression.  Set to False to
        replicate jwdid's ``xasis`` option.
    alpha : float
        Significance level for confidence intervals.
    cluster : str or None
        Column name to use for cluster-robust SEs.  Defaults to the ``unit``
        identifier passed to ``fit()``.
    n_bootstrap : int
        Number of bootstrap replications.  0 disables bootstrap.
    bootstrap_weights : {"rademacher", "webb", "mammen"}
        Bootstrap weight distribution.
    seed : int or None
        Random seed for reproducibility.
    rank_deficient_action : {"warn", "error", "silent"}
        How to handle rank-deficient design matrices.
    """

    def __init__(
        self,
        method: str = "ols",
        control_group: str = "not_yet_treated",
        anticipation: int = 0,
        demean_covariates: bool = True,
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        n_bootstrap: int = 0,
        bootstrap_weights: str = "rademacher",
        seed: Optional[int] = None,
        rank_deficient_action: str = "warn",
    ) -> None:
        if method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
        if control_group not in _VALID_CONTROL_GROUPS:
            raise ValueError(
                f"control_group must be one of {_VALID_CONTROL_GROUPS}, got {control_group!r}"
            )
        if anticipation < 0:
            raise ValueError(f"anticipation must be >= 0, got {anticipation}")
        if bootstrap_weights not in _VALID_BOOTSTRAP_WEIGHTS:
            raise ValueError(
                f"bootstrap_weights must be one of {_VALID_BOOTSTRAP_WEIGHTS}, "
                f"got {bootstrap_weights!r}"
            )

        self.method = method
        self.control_group = control_group
        self.anticipation = anticipation
        self.demean_covariates = demean_covariates
        self.alpha = alpha
        self.cluster = cluster
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = seed
        self.rank_deficient_action = rank_deficient_action

        self.is_fitted_: bool = False
        self._results: Optional[WooldridgeDiDResults] = None

    @property
    def results_(self) -> WooldridgeDiDResults:
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before accessing results_")
        return self._results  # type: ignore[return-value]

    def get_params(self) -> Dict[str, Any]:
        """Return estimator parameters (sklearn-compatible)."""
        return {
            "method": self.method,
            "control_group": self.control_group,
            "anticipation": self.anticipation,
            "demean_covariates": self.demean_covariates,
            "alpha": self.alpha,
            "cluster": self.cluster,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_weights": self.bootstrap_weights,
            "seed": self.seed,
            "rank_deficient_action": self.rank_deficient_action,
        }

    def set_params(self, **params: Any) -> "WooldridgeDiD":
        """Set estimator parameters (sklearn-compatible). Returns self."""
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter: {key!r}")
            setattr(self, key, value)
        # Re-run validation after setting params
        if self.method not in _VALID_METHODS:
            raise ValueError(f"method must be one of {_VALID_METHODS}, got {self.method!r}")
        if self.control_group not in _VALID_CONTROL_GROUPS:
            raise ValueError(
                f"control_group must be one of {_VALID_CONTROL_GROUPS}, "
                f"got {self.control_group!r}"
            )
        if self.anticipation < 0:
            raise ValueError(f"anticipation must be >= 0, got {self.anticipation}")
        if self.bootstrap_weights not in _VALID_BOOTSTRAP_WEIGHTS:
            raise ValueError(
                f"bootstrap_weights must be one of {_VALID_BOOTSTRAP_WEIGHTS}, "
                f"got {self.bootstrap_weights!r}"
            )
        return self

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        cohort: str,
        exovar: Optional[List[str]] = None,
        xtvar: Optional[List[str]] = None,
        xgvar: Optional[List[str]] = None,
        survey_design=None,
    ) -> WooldridgeDiDResults:
        """Fit the ETWFE model.  See class docstring for parameter details.

        Parameters
        ----------
        data : DataFrame with panel data (long format)
        outcome : outcome column name
        unit : unit identifier column
        time : time period column
        cohort : first treatment period (0 or NaN = never treated)
        exovar : time-invariant covariates added without interaction/demeaning
        xtvar : time-varying covariates (demeaned within cohort×period cells
                when ``demean_covariates=True``)
        xgvar : covariates interacted with each cohort indicator
        survey_design : SurveyDesign, optional
            Survey design specification for complex survey data.  Supports
            stratified, clustered, and weighted designs via Taylor Series
            Linearization (TSL).  Replicate-weight designs raise
            ``NotImplementedError``.
        """
        df = data.copy()
        df[cohort] = df[cohort].fillna(0)

        # 0a. Validate cohort is time-invariant within unit
        cohort_per_unit = df.groupby(unit)[cohort].nunique()
        bad_units = cohort_per_unit[cohort_per_unit > 1]
        if len(bad_units) > 0:
            example = bad_units.index[0]
            raise ValueError(
                f"Cohort column '{cohort}' is not time-invariant within unit. "
                f"Unit {example!r} has {int(bad_units.iloc[0])} distinct cohort "
                f"values. The cohort column must be constant within each unit."
            )

        # 0b. Reject bootstrap for nonlinear methods (not implemented)
        if self.n_bootstrap > 0 and self.method != "ols":
            raise ValueError(
                f"Bootstrap inference is only supported for method='ols'. "
                f"Got method={self.method!r} with n_bootstrap={self.n_bootstrap}. "
                f"Set n_bootstrap=0 for analytic SEs."
            )

        # 0c. Reject bootstrap + survey (no survey-aware bootstrap variant)
        if self.n_bootstrap > 0 and survey_design is not None:
            raise ValueError(
                "Bootstrap inference is not supported with survey_design. "
                "Set n_bootstrap=0 for analytic survey SEs."
            )

        # 1. Filter to analysis sample
        sample = _filter_sample(df, unit, time, cohort, self.control_group, self.anticipation)

        # 1b. Identification checks
        groups = sorted(g for g in sample[cohort].unique() if g > 0)
        if len(groups) == 0:
            raise ValueError(
                "No treated cohorts found in data. Ensure the cohort column "
                "contains values > 0 for treated units."
            )
        if self.control_group == "never_treated" and not (sample[cohort] == 0).any():
            raise ValueError(
                "control_group='never_treated' but no never-treated units "
                "(cohort == 0) found. Use 'not_yet_treated' or add "
                "never-treated units."
            )
        if self.control_group == "not_yet_treated":
            # Verify at least some untreated comparison observations exist
            has_untreated = (sample[cohort] == 0).any() or (
                (sample[cohort] - self.anticipation) > sample[time]
            ).any()
            if not has_untreated:
                raise ValueError(
                    "control_group='not_yet_treated' but no untreated comparison "
                    "observations exist. All units are treated at all observed "
                    "time periods. Use 'never_treated' with a never-treated group."
                )

        # 2. Build interaction matrix
        X_int, int_col_names, gt_keys = _build_interaction_matrix(
            sample,
            cohort=cohort,
            time=time,
            anticipation=self.anticipation,
            control_group=self.control_group,
            method=self.method,
        )
        if X_int.shape[1] == 0:
            raise ValueError(
                "No valid treatment cells found. Check that treated units "
                "have post-treatment observations in the data."
            )

        # 3. Covariates
        X_cov = _prepare_covariates(
            sample,
            exovar=exovar,
            xtvar=xtvar,
            xgvar=xgvar,
            cohort=cohort,
            time=time,
            demean_covariates=self.demean_covariates,
            groups=groups,
        )

        all_regressors = int_col_names.copy()
        if X_cov is not None:
            # Build treatment × demeaned-covariate interactions (W2025 Eq. 5.3)
            # For each (g,t) cell indicator and each covariate, create the
            # moderating interaction: X_int[:, i] * x_hat[:, j]
            # This allows treatment effects to vary with covariates within cells.
            cov_names_list = list(exovar or []) + list(xtvar or []) + list(xgvar or [])
            # Compute cohort-demeaned covariates for interaction terms
            X_cov_demeaned = X_cov.copy()
            if self.demean_covariates:
                cohort_vals = sample[cohort].values
                for j in range(X_cov.shape[1]):
                    for g in groups:
                        mask = cohort_vals == g
                        if mask.any():
                            X_cov_demeaned[mask, j] -= X_cov[mask, j].mean()

            interact_cols = []
            interact_names = []
            for i, gt_name in enumerate(int_col_names):
                for j in range(X_cov_demeaned.shape[1]):
                    interact_cols.append(X_int[:, i] * X_cov_demeaned[:, j])
                    cov_label = cov_names_list[j] if j < len(cov_names_list) else f"cov{j}"
                    interact_names.append(f"{gt_name}_x_{cov_label}")

            # Cohort × covariate interactions (W2025 Eq. 5.3: D_g × X)
            # exovar/xtvar get automatic D_g × X; xgvar already has D_g × X
            cov_cols_for_dg = list(exovar or []) + list(xtvar or [])
            cohort_cov_cols = []
            cohort_cov_names = []
            if cov_cols_for_dg:
                cohort_vals_arr = sample[cohort].values
                for g in groups:
                    g_ind = (cohort_vals_arr == g).astype(float)
                    for col in cov_cols_for_dg:
                        cohort_cov_cols.append(g_ind * sample[col].values.astype(float))
                        cohort_cov_names.append(f"D{g}_x_{col}")

            # Time × covariate interactions (W2025 Eq. 5.3: f_t × X)
            # All covariates get f_t × X, drop first time for identification
            all_cov_cols = list(exovar or []) + list(xtvar or []) + list(xgvar or [])
            times_sorted = sorted(sample[time].unique())
            time_cov_cols = []
            time_cov_names = []
            time_vals_arr = sample[time].values
            for t in times_sorted[1:]:  # drop first
                t_ind = (time_vals_arr == t).astype(float)
                for col in all_cov_cols:
                    time_cov_cols.append(t_ind * sample[col].values.astype(float))
                    time_cov_names.append(f"ft{t}_x_{col}")

            # Assemble: [cell_indicators, cell×cov, D_g×X, f_t×X, raw_cov]
            blocks = [X_int]
            if interact_cols:
                blocks.append(np.column_stack(interact_cols))
                all_regressors.extend(interact_names)
            if cohort_cov_cols:
                blocks.append(np.column_stack(cohort_cov_cols))
                all_regressors.extend(cohort_cov_names)
            if time_cov_cols:
                blocks.append(np.column_stack(time_cov_cols))
                all_regressors.extend(time_cov_names)
            blocks.append(X_cov)
            for i in range(X_cov.shape[1]):
                all_regressors.append(f"_cov_{i}")
            X_design = np.hstack(blocks)
        else:
            X_design = X_int

        if self.method == "ols":
            results = self._fit_ols(
                sample,
                outcome,
                unit,
                time,
                cohort,
                X_design,
                all_regressors,
                gt_keys,
                int_col_names,
                groups,
                survey_design=survey_design,
            )
        elif self.method == "logit":
            n_cov_interact = X_cov.shape[1] if X_cov is not None else 0
            results = self._fit_logit(
                sample,
                outcome,
                unit,
                time,
                cohort,
                X_design,
                all_regressors,
                gt_keys,
                int_col_names,
                groups,
                n_cov_interact=n_cov_interact,
                survey_design=survey_design,
            )
        else:  # poisson
            n_cov_interact = X_cov.shape[1] if X_cov is not None else 0
            results = self._fit_poisson(
                sample,
                outcome,
                unit,
                time,
                cohort,
                X_design,
                all_regressors,
                gt_keys,
                int_col_names,
                groups,
                n_cov_interact=n_cov_interact,
                survey_design=survey_design,
            )

        self._results = results
        self.is_fitted_ = True
        return results

    def _count_control_units(self, sample: pd.DataFrame, unit: str, cohort: str, time: str) -> int:
        """Count control units consistent with control_group setting."""
        n_never = int(sample[sample[cohort] == 0][unit].nunique())
        if self.control_group == "not_yet_treated":
            # Also count future-treated units that contribute pre-anticipation obs
            nyt = sample[
                (sample[cohort] > 0) & (sample[time] < sample[cohort] - self.anticipation)
            ][unit].nunique()
            return n_never + int(nyt)
        return n_never

    def _fit_ols(
        self,
        sample: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        cohort: str,
        X_design: np.ndarray,
        col_names: List[str],
        gt_keys: List[Tuple],
        int_col_names: List[str],
        groups: List[Any],
        survey_design=None,
    ) -> WooldridgeDiDResults:
        """OLS path: within-transform FE, solve_ols, cluster SE."""
        # Reset index so numpy positional indexing matches pandas groupby
        sample = sample.reset_index(drop=True)
        # Cluster IDs (default: unit level) — needed before survey resolution
        cluster_col = self.cluster if self.cluster else unit
        cluster_ids = sample[cluster_col].values

        # Resolve survey design, inject cluster as PSU only when user explicitly set cluster=
        survey_cluster_ids = cluster_ids if self.cluster else None
        resolved, survey_weights, survey_weight_type, survey_metadata, df_inf = (
            _resolve_survey_for_wooldridge(survey_design, sample, survey_cluster_ids, self.cluster)
        )

        # 4. Within-transform: absorb unit + time FE
        all_vars = [outcome] + [f"_x{i}" for i in range(X_design.shape[1])]
        tmp = sample[[unit, time]].copy()
        tmp[outcome] = sample[outcome].values
        for i in range(X_design.shape[1]):
            tmp[f"_x{i}"] = X_design[:, i]

        # Use iterative alternating projections for demeaning (exact for
        # both balanced and unbalanced panels).  Survey weights change the
        # weighted FWL projection — all columns (treatment interactions +
        # covariates) are demeaned together.
        wt_weights = survey_weights if survey_weights is not None else np.ones(len(tmp))

        # Guard: zero-weight unit/time groups cause 0/0 in within_transform
        if survey_weights is not None and np.any(survey_weights == 0):
            sw_series = pd.Series(survey_weights, index=sample.index)
            for grp_col, grp_label in [(unit, "unit"), (time, "time period")]:
                grp_sums = sw_series.groupby(sample[grp_col]).sum()
                zero_grps = grp_sums[grp_sums == 0].index.tolist()
                if zero_grps:
                    raise ValueError(
                        f"Survey weights sum to zero for {grp_label}(s) "
                        f"{zero_grps[:3]}. Cannot compute weighted "
                        f"within-transformation. Remove zero-weight "
                        f"{grp_label}s or use non-zero weights."
                    )

        transformed = within_transform(
            tmp, all_vars, unit=unit, time=time, suffix="_demeaned",
            weights=wt_weights,
        )

        y = transformed[f"{outcome}_demeaned"].values
        X_cols = [f"_x{i}_demeaned" for i in range(X_design.shape[1])]
        X = transformed[X_cols].values

        # 6. Solve OLS (skip cluster-robust vcov when survey will provide TSL vcov)
        coefs, resids, vcov = solve_ols(
            X,
            y,
            cluster_ids=cluster_ids,
            return_vcov=(resolved is None),
            rank_deficient_action=self.rank_deficient_action,
            column_names=col_names,
            weights=survey_weights,
            weight_type=survey_weight_type,
        )

        # Survey TSL vcov replaces cluster-robust vcov
        if resolved is not None:
            from diff_diff.survey import compute_survey_vcov
            nan_mask_ols = np.isnan(coefs)
            if np.any(nan_mask_ols):
                kept = ~nan_mask_ols
                vcov_kept = compute_survey_vcov(X[:, kept], resids, resolved)
                vcov = np.full((len(coefs), len(coefs)), np.nan)
                kept_idx = np.where(kept)[0]
                vcov[np.ix_(kept_idx, kept_idx)] = vcov_kept
            else:
                vcov = compute_survey_vcov(X, resids, resolved)

        # 7. Extract β_{g,t} and build gt_effects dict
        gt_effects: Dict[Tuple, Dict] = {}
        gt_weights: Dict[Tuple, int] = {}
        for idx, (g, t) in enumerate(gt_keys):
            if idx >= len(coefs):
                break
            # Skip cells whose coefficient was dropped (rank deficiency)
            if np.isnan(coefs[idx]):
                continue
            att = float(coefs[idx])
            se = float(np.sqrt(max(vcov[idx, idx], 0.0))) if vcov is not None else float("nan")
            t_stat, p_value, conf_int = safe_inference(att, se, alpha=self.alpha, df=df_inf)
            gt_effects[(g, t)] = {
                "att": att,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
            }
            gt_weights[(g, t)] = int(((sample[cohort] == g) & (sample[time] == t)).sum())

        # Extract vcov submatrix for identified β_{g,t} only (skip NaN/dropped)
        gt_keys_ordered = list(gt_effects.keys())
        if vcov is not None and gt_keys_ordered:
            # Map from gt_keys_ordered to original indices in the coef vector
            orig_indices = [i for i, k in enumerate(gt_keys) if k in gt_effects]
            gt_vcov = vcov[np.ix_(orig_indices, orig_indices)]
        else:
            gt_vcov = None

        # 8. Simple aggregation (always computed)
        overall = _compute_weighted_agg(
            gt_effects, gt_weights, gt_keys_ordered, gt_vcov, self.alpha, df=df_inf
        )

        # Metadata
        n_treated = int(sample[sample[cohort] > 0][unit].nunique())
        n_control = self._count_control_units(sample, unit, cohort, time)
        all_times = sorted(sample[time].unique().tolist())

        results = WooldridgeDiDResults(
            group_time_effects=gt_effects,
            overall_att=overall["att"],
            overall_se=overall["se"],
            overall_t_stat=overall["t_stat"],
            overall_p_value=overall["p_value"],
            overall_conf_int=overall["conf_int"],
            method=self.method,
            control_group=self.control_group,
            groups=groups,
            time_periods=all_times,
            n_obs=len(sample),
            n_treated_units=n_treated,
            n_control_units=n_control,
            alpha=self.alpha,
            anticipation=self.anticipation,
            survey_metadata=survey_metadata,
            _gt_weights=gt_weights,
            _gt_vcov=gt_vcov,
            _gt_keys=gt_keys_ordered,
            _df_survey=df_inf,
        )

        # 9. Optional multiplier bootstrap (overrides analytic SE for overall ATT)
        if self.n_bootstrap > 0:
            rng = np.random.default_rng(self.seed)
            # Draw weights at the analytic cluster level (not always unit)
            unique_boot_clusters = np.unique(cluster_ids)
            n_boot_clusters = len(unique_boot_clusters)
            post_keys = [(g, t) for (g, t) in gt_keys_ordered if t >= g]
            w_total_b = sum(gt_weights.get(k, 0) for k in post_keys)
            boot_atts: List[float] = []
            for _ in range(self.n_bootstrap):
                if self.bootstrap_weights == "rademacher":
                    cl_weights = rng.choice([-1.0, 1.0], size=n_boot_clusters)
                elif self.bootstrap_weights == "webb":
                    cl_weights = rng.choice(
                        [-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)],
                        size=n_boot_clusters,
                    )
                else:  # mammen
                    phi = (1 + np.sqrt(5)) / 2
                    cl_weights = rng.choice(
                        [-(phi - 1), phi],
                        p=[phi / np.sqrt(5), (phi - 1) / np.sqrt(5)],
                        size=n_boot_clusters,
                    )
                obs_weights = cl_weights[np.searchsorted(unique_boot_clusters, cluster_ids)]
                y_boot = y + obs_weights * resids
                coefs_b, _, _ = solve_ols(
                    X,
                    y_boot,
                    cluster_ids=cluster_ids,
                    return_vcov=False,
                    rank_deficient_action="silent",
                )
                if w_total_b > 0:
                    att_b = (
                        sum(
                            gt_weights.get(k, 0) * float(coefs_b[i])
                            for i, k in enumerate(gt_keys)
                            if k in post_keys and i < len(coefs_b)
                        )
                        / w_total_b
                    )
                    boot_atts.append(att_b)
            if boot_atts:
                boot_se = float(np.std(boot_atts, ddof=1))
                t_stat_b, p_b, ci_b = safe_inference(results.overall_att, boot_se, alpha=self.alpha)
                results.overall_se = boot_se
                results.overall_t_stat = t_stat_b
                results.overall_p_value = p_b
                results.overall_conf_int = ci_b

        return results

    def _fit_logit(
        self,
        sample: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        cohort: str,
        X_int: np.ndarray,
        col_names: List[str],
        gt_keys: List[Tuple],
        int_col_names: List[str],
        groups: List[Any],
        n_cov_interact: int = 0,
        survey_design=None,
    ) -> WooldridgeDiDResults:
        """Logit path: cohort + time additive FEs + solve_logit + ASF ATT.

        Matches Stata jwdid method(logit): logit y [treatment_interactions]
        i.gvar i.tvar — cohort main effects + time main effects (additive),
        not cohort×time saturated group FEs.
        """
        n_int = len(int_col_names)

        # Design matrix: treatment interactions + cohort FEs + time FEs
        # This matches Stata's `i.gvar i.tvar` specification.
        cohort_dummies = pd.get_dummies(sample[cohort], drop_first=True).values.astype(float)
        time_dummies = pd.get_dummies(sample[time], drop_first=True).values.astype(float)
        X_full = np.hstack([X_int, cohort_dummies, time_dummies])

        y = sample[outcome].values.astype(float)
        if not np.all(np.isfinite(y)):
            raise ValueError("Outcome contains non-finite values (NaN/Inf).")
        if np.any(y < 0) or np.any(y > 1):
            raise ValueError(
                f"method='logit' requires outcomes in [0, 1]. "
                f"Got range [{y.min():.4f}, {y.max():.4f}]."
            )
        cluster_col = self.cluster if self.cluster else unit
        cluster_ids = sample[cluster_col].values

        # Resolve survey design, inject cluster as PSU only when user explicitly set cluster=
        survey_cluster_ids = cluster_ids if self.cluster else None
        resolved, survey_weights, survey_weight_type, survey_metadata, df_inf = (
            _resolve_survey_for_wooldridge(survey_design, sample, survey_cluster_ids, self.cluster)
        )
        _has_survey = resolved is not None

        beta, probs = solve_logit(
            X_full,
            y,
            rank_deficient_action=self.rank_deficient_action,
            weights=survey_weights,
        )
        # solve_logit prepends intercept — beta[0] is intercept, beta[1:] are X_full cols
        beta_int_cols = beta[1 : n_int + 1]  # treatment interaction coefficients

        # Handle rank-deficient designs: identify kept columns, compute vcov
        # on reduced design, then expand back
        nan_mask = np.isnan(beta)
        beta_clean = np.where(nan_mask, 0.0, beta)
        kept_beta = ~nan_mask

        # QMLE sandwich vcov
        resids = y - probs
        X_with_intercept = np.column_stack([np.ones(len(y)), X_full])

        if _has_survey:
            # X_tilde trick: transform design matrix so compute_survey_vcov
            # produces the correct QMLE sandwich for nonlinear models.
            # Bread: (X_tilde'WX_tilde)^{-1} = (X'diag(w*V)X)^{-1}
            # Scores: w*X_tilde*r_tilde = w*X*(y-mu)
            from diff_diff.survey import compute_survey_vcov
            V = probs * (1 - probs)
            sqrt_V = np.sqrt(np.clip(V, 1e-20, None))
            X_tilde = X_with_intercept * sqrt_V[:, None]
            r_tilde = resids / sqrt_V
            if np.any(nan_mask):
                X_tilde_r = X_tilde[:, kept_beta]
                vcov_reduced = compute_survey_vcov(X_tilde_r, r_tilde, resolved)
                k_full = len(beta)
                vcov_full = np.full((k_full, k_full), np.nan)
                kept_idx = np.where(kept_beta)[0]
                vcov_full[np.ix_(kept_idx, kept_idx)] = vcov_reduced
            else:
                vcov_full = compute_survey_vcov(X_tilde, r_tilde, resolved)
        else:
            # Cluster-robust QMLE sandwich (non-survey path)
            if np.any(nan_mask):
                X_reduced = X_with_intercept[:, kept_beta]
                vcov_reduced = compute_robust_vcov(
                    X_reduced,
                    resids,
                    cluster_ids=cluster_ids,
                    weights=probs * (1 - probs),
                    weight_type="aweight",
                )
                k_full = len(beta)
                vcov_full = np.full((k_full, k_full), np.nan)
                kept_idx = np.where(kept_beta)[0]
                vcov_full[np.ix_(kept_idx, kept_idx)] = vcov_reduced
            else:
                vcov_full = compute_robust_vcov(
                    X_with_intercept,
                    resids,
                    cluster_ids=cluster_ids,
                    weights=probs * (1 - probs),
                    weight_type="aweight",
                )
        beta = beta_clean

        # Survey-weighted averaging helpers for ASF computation
        def _avg(a, cell_mask):
            if survey_weights is not None:
                return float(np.average(a, weights=survey_weights[cell_mask]))
            return float(np.mean(a))

        def _avg_ax0(a, cell_mask):
            if survey_weights is not None:
                return np.average(a, weights=survey_weights[cell_mask], axis=0)
            return np.mean(a, axis=0)

        # ASF ATT(g,t) for treated units in each cell
        gt_effects: Dict[Tuple, Dict] = {}
        gt_weights: Dict[Tuple, int] = {}
        gt_grads: Dict[Tuple, np.ndarray] = {}  # store per-cell gradients for aggregate SE
        for idx, (g, t) in enumerate(gt_keys):
            if idx >= n_int:
                break
            cell_mask = (sample[cohort] == g) & (sample[time] == t)
            if cell_mask.sum() == 0:
                continue
            # Skip cells whose interaction coefficient was dropped (rank deficiency)
            # Skip cells where all survey weights are zero (non-estimable)
            if survey_weights is not None and np.sum(survey_weights[cell_mask]) == 0:
                continue
            delta = beta_int_cols[idx]
            if np.isnan(delta):
                continue
            eta_base = X_with_intercept[cell_mask] @ beta
            # Counterfactual: zero the FULL treatment block for cell (g,t).
            # This includes the scalar cell effect δ_{g,t} AND any cell ×
            # covariate interaction effects ξ_{g,t,j} * x_hat_j (W2023 Eq. 3.15).
            delta_total = np.full(cell_mask.sum(), float(delta))
            for j in range(n_cov_interact):
                coef_pos = 1 + n_int + idx * n_cov_interact + j
                if coef_pos < len(beta):
                    x_hat_j = X_with_intercept[cell_mask, coef_pos]
                    delta_total = delta_total + beta[coef_pos] * x_hat_j
            eta_0 = eta_base - delta_total
            att = _avg(_logistic(eta_base) - _logistic(eta_0), cell_mask)
            # Delta method gradient: d(ATT)/d(β)
            #   for nuisance p: mean_i[(Λ'(η_1) - Λ'(η_0)) * X_p]
            #   for cell intercept: mean_i[Λ'(η_1)]
            #   for cell × cov j: mean_i[Λ'(η_1) * x_hat_j]
            d_diff = _logistic_deriv(eta_base) - _logistic_deriv(eta_0)
            grad = _avg_ax0(X_with_intercept[cell_mask] * d_diff[:, None], cell_mask)
            grad[1 + idx] = _avg(_logistic_deriv(eta_base), cell_mask)
            for j in range(n_cov_interact):
                coef_pos = 1 + n_int + idx * n_cov_interact + j
                if coef_pos < len(beta):
                    x_hat_j = X_with_intercept[cell_mask, coef_pos]
                    grad[coef_pos] = _avg(_logistic_deriv(eta_base) * x_hat_j, cell_mask)
            # Compute SE in reduced parameter space if rank-deficient
            if np.any(nan_mask):
                grad_r = grad[kept_beta]
                se = float(np.sqrt(max(grad_r @ vcov_reduced @ grad_r, 0.0)))
            else:
                se = float(np.sqrt(max(grad @ vcov_full @ grad, 0.0)))
            t_stat, p_value, conf_int = safe_inference(att, se, alpha=self.alpha, df=df_inf)
            gt_effects[(g, t)] = {
                "att": att,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
            }
            gt_weights[(g, t)] = int(cell_mask.sum())
            # Store gradient in reduced space for aggregate SE
            gt_grads[(g, t)] = grad[kept_beta] if np.any(nan_mask) else grad

        gt_keys_ordered = [k for k in gt_keys if k in gt_effects]
        # Use reduced vcov for all downstream SE computations
        _vcov_se = vcov_reduced if np.any(nan_mask) else vcov_full
        # ATT-level covariance: J @ vcov @ J' where J rows are per-cell gradients
        if gt_keys_ordered:
            J = np.array([gt_grads[k] for k in gt_keys_ordered])
            gt_vcov = J @ _vcov_se @ J.T
        else:
            gt_vcov = None

        # Overall SE via joint delta method: ∇β(overall_att) = Σ w_k/w_total * grad_k
        post_keys = [(g, t) for (g, t) in gt_keys_ordered if t >= g]
        w_total = sum(gt_weights.get(k, 0) for k in post_keys)
        if w_total > 0 and post_keys:
            overall_att = sum(gt_weights[k] * gt_effects[k]["att"] for k in post_keys) / w_total
            agg_grad = sum((gt_weights[k] / w_total) * gt_grads[k] for k in post_keys)
            overall_se = float(np.sqrt(max(agg_grad @ _vcov_se @ agg_grad, 0.0)))
            t_stat, p_value, conf_int = safe_inference(overall_att, overall_se, alpha=self.alpha, df=df_inf)
            overall = {
                "att": overall_att,
                "se": overall_se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
            }
        else:
            overall = _compute_weighted_agg(
                gt_effects, gt_weights, gt_keys_ordered, None, self.alpha, df=df_inf
            )

        return WooldridgeDiDResults(
            group_time_effects=gt_effects,
            overall_att=overall["att"],
            overall_se=overall["se"],
            overall_t_stat=overall["t_stat"],
            overall_p_value=overall["p_value"],
            overall_conf_int=overall["conf_int"],
            method=self.method,
            control_group=self.control_group,
            groups=groups,
            time_periods=sorted(sample[time].unique().tolist()),
            n_obs=len(sample),
            n_treated_units=int(sample[sample[cohort] > 0][unit].nunique()),
            n_control_units=self._count_control_units(sample, unit, cohort, time),
            alpha=self.alpha,
            anticipation=self.anticipation,
            survey_metadata=survey_metadata,
            _gt_weights=gt_weights,
            _gt_vcov=gt_vcov,
            _gt_keys=gt_keys_ordered,
            _df_survey=df_inf,
        )

    def _fit_poisson(
        self,
        sample: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        cohort: str,
        X_int: np.ndarray,
        col_names: List[str],
        gt_keys: List[Tuple],
        int_col_names: List[str],
        groups: List[Any],
        n_cov_interact: int = 0,
        survey_design=None,
    ) -> WooldridgeDiDResults:
        """Poisson path: cohort + time additive FEs + solve_poisson + ASF ATT.

        Matches Stata jwdid method(poisson): poisson y [treatment_interactions]
        i.gvar i.tvar — cohort main effects + time main effects (additive),
        not cohort×time saturated group FEs.
        """
        n_int = len(int_col_names)

        # Design matrix: intercept + treatment interactions + cohort FEs + time FEs.
        # Matches Stata's `i.gvar i.tvar` + treatment interaction specification.
        # solve_poisson does not prepend an intercept, so we include one explicitly.
        intercept = np.ones((len(sample), 1))
        cohort_dummies = pd.get_dummies(sample[cohort], drop_first=True).values.astype(float)
        time_dummies = pd.get_dummies(sample[time], drop_first=True).values.astype(float)
        X_full = np.hstack([intercept, X_int, cohort_dummies, time_dummies])
        # Treatment interaction coefficients start at column index 1.

        y = sample[outcome].values.astype(float)
        if not np.all(np.isfinite(y)):
            raise ValueError("Outcome contains non-finite values (NaN/Inf).")
        if np.any(y < 0):
            raise ValueError(
                f"method='poisson' requires non-negative outcomes. "
                f"Got minimum value {y.min():.4f}."
            )
        cluster_col = self.cluster if self.cluster else unit
        cluster_ids = sample[cluster_col].values

        # Resolve survey design, inject cluster as PSU only when user explicitly set cluster=
        survey_cluster_ids = cluster_ids if self.cluster else None
        resolved, survey_weights, survey_weight_type, survey_metadata, df_inf = (
            _resolve_survey_for_wooldridge(survey_design, sample, survey_cluster_ids, self.cluster)
        )
        _has_survey = resolved is not None

        beta, mu_hat = solve_poisson(
            X_full, y,
            rank_deficient_action=self.rank_deficient_action,
            weights=survey_weights,
        )

        # Handle rank-deficient designs: compute vcov on reduced design.
        # Preserve raw interaction coefficients BEFORE zeroing NaN so the
        # NaN check in the ASF loop correctly skips dropped cells.
        nan_mask = np.isnan(beta)
        beta_int_raw = beta[1 : 1 + n_int].copy()  # before zeroing
        beta_clean = np.where(nan_mask, 0.0, beta)
        kept_beta = ~nan_mask

        # QMLE sandwich vcov
        resids = y - mu_hat

        if _has_survey:
            # X_tilde trick for nonlinear survey vcov (V = mu for Poisson)
            from diff_diff.survey import compute_survey_vcov
            sqrt_V = np.sqrt(np.clip(mu_hat, 1e-20, None))
            X_tilde = X_full * sqrt_V[:, None]
            r_tilde = resids / sqrt_V
            if np.any(nan_mask):
                X_tilde_r = X_tilde[:, kept_beta]
                vcov_reduced = compute_survey_vcov(X_tilde_r, r_tilde, resolved)
                k_full = len(beta)
                vcov_full = np.full((k_full, k_full), np.nan)
                kept_idx = np.where(kept_beta)[0]
                vcov_full[np.ix_(kept_idx, kept_idx)] = vcov_reduced
            else:
                vcov_full = compute_survey_vcov(X_tilde, r_tilde, resolved)
        else:
            # Cluster-robust QMLE sandwich (non-survey path)
            if np.any(nan_mask):
                X_reduced = X_full[:, kept_beta]
                vcov_reduced = compute_robust_vcov(
                    X_reduced,
                    resids,
                    cluster_ids=cluster_ids,
                    weights=mu_hat,
                    weight_type="aweight",
                )
                k_full = len(beta)
                vcov_full = np.full((k_full, k_full), np.nan)
                kept_idx = np.where(kept_beta)[0]
                vcov_full[np.ix_(kept_idx, kept_idx)] = vcov_reduced
            else:
                vcov_full = compute_robust_vcov(
                    X_full,
                    resids,
                    cluster_ids=cluster_ids,
                    weights=mu_hat,
                    weight_type="aweight",
                )
        beta = beta_clean

        # Treatment interaction coefficients (from cleaned beta for computation)
        beta_int = beta[1 : 1 + n_int]

        # Survey-weighted averaging helpers for ASF computation
        def _avg(a, cell_mask):
            if survey_weights is not None:
                return float(np.average(a, weights=survey_weights[cell_mask]))
            return float(np.mean(a))

        def _avg_ax0(a, cell_mask):
            if survey_weights is not None:
                return np.average(a, weights=survey_weights[cell_mask], axis=0)
            return np.mean(a, axis=0)

        # ASF ATT(g,t) for treated units in each cell.
        # eta_base = X_full @ beta already includes the treatment effect (D_{g,t}=1).
        # Counterfactual: eta_0 = eta_base - delta  (treatment switched off).
        # ATT = E[exp(η_1)] - E[exp(η_0)] = E[exp(η_base)] - E[exp(η_base - δ)]
        gt_effects: Dict[Tuple, Dict] = {}
        gt_weights: Dict[Tuple, int] = {}
        gt_grads: Dict[Tuple, np.ndarray] = {}  # per-cell gradients for aggregate SE
        for idx, (g, t) in enumerate(gt_keys):
            if idx >= n_int:
                break
            cell_mask = (sample[cohort] == g) & (sample[time] == t)
            if cell_mask.sum() == 0:
                continue
            # Skip cells whose interaction coefficient was dropped (rank deficiency).
            # Use raw coefficients (before NaN->0 zeroing) to detect dropped cells.
            if np.isnan(beta_int_raw[idx]):
                continue
            # Skip cells where all survey weights are zero (non-estimable)
            if survey_weights is not None and np.sum(survey_weights[cell_mask]) == 0:
                continue
            delta = beta_int[idx]
            if np.isnan(delta):
                continue
            eta_base = np.clip(X_full[cell_mask] @ beta, -500, 500)
            # Counterfactual: zero the FULL treatment block (W2023 Eq. 3.15)
            delta_total = np.full(cell_mask.sum(), float(delta))
            for j in range(n_cov_interact):
                coef_pos = 1 + n_int + idx * n_cov_interact + j
                if coef_pos < len(beta):
                    x_hat_j = X_full[cell_mask, coef_pos]
                    delta_total = delta_total + beta[coef_pos] * x_hat_j
            eta_0 = eta_base - delta_total
            mu_1 = np.exp(eta_base)
            mu_0 = np.exp(eta_0)
            att = _avg(mu_1 - mu_0, cell_mask)
            # Delta method gradient:
            #   for nuisance p: mean_i[(μ_1 - μ_0) * X_p]
            #   for cell intercept: mean_i[μ_1]
            #   for cell × cov j: mean_i[μ_1 * x_hat_j]
            diff_mu = mu_1 - mu_0
            grad = _avg_ax0(X_full[cell_mask] * diff_mu[:, None], cell_mask)
            grad[1 + idx] = _avg(mu_1, cell_mask)
            for j in range(n_cov_interact):
                coef_pos = 1 + n_int + idx * n_cov_interact + j
                if coef_pos < len(beta):
                    x_hat_j = X_full[cell_mask, coef_pos]
                    grad[coef_pos] = _avg(mu_1 * x_hat_j, cell_mask)
            # Compute SE in reduced parameter space if rank-deficient
            if np.any(nan_mask):
                grad_r = grad[kept_beta]
                se = float(np.sqrt(max(grad_r @ vcov_reduced @ grad_r, 0.0)))
            else:
                se = float(np.sqrt(max(grad @ vcov_full @ grad, 0.0)))
            t_stat, p_value, conf_int = safe_inference(att, se, alpha=self.alpha, df=df_inf)
            gt_effects[(g, t)] = {
                "att": att,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
            }
            gt_weights[(g, t)] = int(cell_mask.sum())
            gt_grads[(g, t)] = grad[kept_beta] if np.any(nan_mask) else grad

        gt_keys_ordered = [k for k in gt_keys if k in gt_effects]
        _vcov_se = vcov_reduced if np.any(nan_mask) else vcov_full
        # ATT-level covariance: J @ vcov @ J' where J rows are per-cell gradients
        if gt_keys_ordered:
            J = np.array([gt_grads[k] for k in gt_keys_ordered])
            gt_vcov = J @ _vcov_se @ J.T
        else:
            gt_vcov = None

        # Overall SE via joint delta method
        post_keys = [(g, t) for (g, t) in gt_keys_ordered if t >= g]
        w_total = sum(gt_weights.get(k, 0) for k in post_keys)
        if w_total > 0 and post_keys:
            overall_att = sum(gt_weights[k] * gt_effects[k]["att"] for k in post_keys) / w_total
            agg_grad = sum((gt_weights[k] / w_total) * gt_grads[k] for k in post_keys)
            overall_se = float(np.sqrt(max(agg_grad @ _vcov_se @ agg_grad, 0.0)))
            t_stat, p_value, conf_int = safe_inference(overall_att, overall_se, alpha=self.alpha, df=df_inf)
            overall = {
                "att": overall_att,
                "se": overall_se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
            }
        else:
            overall = _compute_weighted_agg(
                gt_effects, gt_weights, gt_keys_ordered, None, self.alpha, df=df_inf
            )

        return WooldridgeDiDResults(
            group_time_effects=gt_effects,
            overall_att=overall["att"],
            overall_se=overall["se"],
            overall_t_stat=overall["t_stat"],
            overall_p_value=overall["p_value"],
            overall_conf_int=overall["conf_int"],
            method=self.method,
            control_group=self.control_group,
            groups=groups,
            time_periods=sorted(sample[time].unique().tolist()),
            n_obs=len(sample),
            n_treated_units=int(sample[sample[cohort] > 0][unit].nunique()),
            n_control_units=self._count_control_units(sample, unit, cohort, time),
            alpha=self.alpha,
            anticipation=self.anticipation,
            survey_metadata=survey_metadata,
            _gt_weights=gt_weights,
            _gt_vcov=gt_vcov,
            _gt_keys=gt_keys_ordered,
            _df_survey=df_inf,
        )
