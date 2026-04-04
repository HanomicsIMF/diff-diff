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

    t_stat, p_value, conf_int = safe_inference(att, se, alpha=alpha)
    return {"att": att, "se": se, "t_stat": t_stat, "p_value": p_value, "conf_int": conf_int}


def _filter_sample(
    data: pd.DataFrame,
    unit: str,
    time: str,
    cohort: str,
    control_group: str,
    anticipation: int,
) -> pd.DataFrame:
    """Return the analysis sample following jwdid selection rules.

    For "not_yet_treated": keep all observations from treated units (pre- and
    post-treatment) plus all never-treated and not-yet-treated observations.
    For "never_treated": keep only post-treatment observations from treated
    units (t >= g - anticipation) plus all never-treated observations.
    Pre-treatment observations from treated units are excluded so they do not
    serve as implicit controls in the regression baseline.
    """
    df = data.copy()
    # Normalise never-treated: fill NaN cohort with 0
    df[cohort] = df[cohort].fillna(0)

    if control_group == "never_treated":
        # Post-treatment obs from treated units + all never-treated obs.
        # Pre-treatment obs from treated units are excluded so the
        # counterfactual is identified solely from never-treated units.
        treated_mask = (df[cohort] > 0) & (df[time] >= df[cohort] - anticipation)
        control_mask = df[cohort] == 0
    else:  # not_yet_treated
        # All treated-unit obs + never-treated + not-yet-treated obs
        treated_mask = df[cohort] > 0
        control_mask = (df[cohort] == 0) | (df[cohort] > df[time])

    return df[treated_mask | control_mask].copy()


def _build_interaction_matrix(
    data: pd.DataFrame,
    cohort: str,
    time: str,
    anticipation: int,
) -> Tuple[np.ndarray, List[str], List[Tuple[Any, Any]]]:
    """Build the saturated cohort×time interaction design matrix.

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

    cols = []
    col_names = []
    gt_keys = []

    for g in groups:
        for t in times:
            if t >= g - anticipation:
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
        """
        df = data.copy()
        df[cohort] = df[cohort].fillna(0)

        # 1. Filter to analysis sample
        sample = _filter_sample(df, unit, time, cohort, self.control_group, self.anticipation)

        # 2. Build interaction matrix
        X_int, int_col_names, gt_keys = _build_interaction_matrix(
            sample, cohort=cohort, time=time, anticipation=self.anticipation
        )

        # 3. Covariates
        groups = sorted(g for g in sample[cohort].unique() if g > 0)
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
            X_design = np.hstack([X_int, X_cov])
            for i in range(X_cov.shape[1]):
                all_regressors.append(f"_cov_{i}")
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
            )
        elif self.method == "logit":
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
            )
        else:  # poisson
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
            )

        self._results = results
        self.is_fitted_ = True
        return results

    def _count_control_units(self, sample: pd.DataFrame, unit: str, cohort: str, time: str) -> int:
        """Count control units consistent with control_group setting."""
        n_never = int(sample[sample[cohort] == 0][unit].nunique())
        if self.control_group == "not_yet_treated":
            # Also count future-treated units that contribute pre-treatment obs
            nyt = sample[(sample[cohort] > 0) & (sample[time] < sample[cohort])][unit].nunique()
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
    ) -> WooldridgeDiDResults:
        """OLS path: within-transform FE, solve_ols, cluster SE."""
        # 4. Within-transform: absorb unit + time FE
        all_vars = [outcome] + [f"_x{i}" for i in range(X_design.shape[1])]
        tmp = sample[[unit, time]].copy()
        tmp[outcome] = sample[outcome].values
        for i in range(X_design.shape[1]):
            tmp[f"_x{i}"] = X_design[:, i]

        transformed = within_transform(tmp, all_vars, unit=unit, time=time, suffix="_demeaned")

        y = transformed[f"{outcome}_demeaned"].values
        X_cols = [f"_x{i}_demeaned" for i in range(X_design.shape[1])]
        X = transformed[X_cols].values

        # 5. Cluster IDs (default: unit level)
        cluster_col = self.cluster if self.cluster else unit
        cluster_ids = sample[cluster_col].values

        # 6. Solve OLS
        coefs, resids, vcov = solve_ols(
            X,
            y,
            cluster_ids=cluster_ids,
            return_vcov=True,
            rank_deficient_action=self.rank_deficient_action,
            column_names=col_names,
        )

        # 7. Extract β_{g,t} and build gt_effects dict
        gt_effects: Dict[Tuple, Dict] = {}
        gt_weights: Dict[Tuple, int] = {}
        for idx, (g, t) in enumerate(gt_keys):
            if idx >= len(coefs):
                break
            att = float(coefs[idx])
            se = float(np.sqrt(max(vcov[idx, idx], 0.0))) if vcov is not None else float("nan")
            t_stat, p_value, conf_int = safe_inference(att, se, alpha=self.alpha)
            gt_effects[(g, t)] = {
                "att": att,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
            }
            gt_weights[(g, t)] = int(((sample[cohort] == g) & (sample[time] == t)).sum())

        # Extract vcov submatrix for beta_{g,t} only
        n_gt = len(gt_keys)
        gt_vcov = vcov[:n_gt, :n_gt] if vcov is not None else None
        gt_keys_ordered = list(gt_keys)

        # 8. Simple aggregation (always computed)
        overall = _compute_weighted_agg(
            gt_effects, gt_weights, gt_keys_ordered, gt_vcov, self.alpha
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
            _gt_weights=gt_weights,
            _gt_vcov=gt_vcov,
            _gt_keys=gt_keys_ordered,
        )

        # 9. Optional multiplier bootstrap (overrides analytic SE for overall ATT)
        if self.n_bootstrap > 0:
            rng = np.random.default_rng(self.seed)
            units_arr = sample[unit].values
            unique_units = np.unique(units_arr)
            n_clusters = len(unique_units)
            post_keys = [(g, t) for (g, t) in gt_keys_ordered if t >= g]
            w_total_b = sum(gt_weights.get(k, 0) for k in post_keys)
            boot_atts: List[float] = []
            for _ in range(self.n_bootstrap):
                if self.bootstrap_weights == "rademacher":
                    unit_weights = rng.choice([-1.0, 1.0], size=n_clusters)
                elif self.bootstrap_weights == "webb":
                    unit_weights = rng.choice(
                        [-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)],
                        size=n_clusters,
                    )
                else:  # mammen
                    phi = (1 + np.sqrt(5)) / 2
                    unit_weights = rng.choice(
                        [-(phi - 1), phi],
                        p=[phi / np.sqrt(5), (phi - 1) / np.sqrt(5)],
                        size=n_clusters,
                    )
                obs_weights = unit_weights[np.searchsorted(unique_units, units_arr)]
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
        cluster_col = self.cluster if self.cluster else unit
        cluster_ids = sample[cluster_col].values

        beta, probs = solve_logit(
            X_full,
            y,
            rank_deficient_action=self.rank_deficient_action,
        )
        # solve_logit prepends intercept — beta[0] is intercept, beta[1:] are X_full cols
        beta_int_cols = beta[1 : n_int + 1]  # treatment interaction coefficients

        # Handle rank-deficient designs: zero out NaN entries so downstream
        # matrix ops don't propagate NaN (dropped columns contribute nothing)
        nan_mask = np.isnan(beta)
        if np.any(nan_mask):
            beta = np.where(nan_mask, 0.0, beta)

        # QMLE sandwich vcov via shared linalg backend
        resids = y - probs
        X_with_intercept = np.column_stack([np.ones(len(y)), X_full])
        vcov_full = compute_robust_vcov(
            X_with_intercept,
            resids,
            cluster_ids=cluster_ids,
            weights=probs * (1 - probs),  # logit QMLE bread: (X'WX)^{-1}
            weight_type="aweight",  # unweighted scores for QMLE sandwich
        )

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
            delta = beta_int_cols[idx]
            if np.isnan(delta):
                continue
            eta_base = X_with_intercept[cell_mask] @ beta
            # eta_base already contains the treatment effect (D_{g,t}=1 in cell).
            # Counterfactual: eta_0 = eta_base - delta (treatment switched off).
            # ATT = E[Λ(η_1)] - E[Λ(η_0)] = E[Λ(η_base)] - E[Λ(η_base - δ)]
            eta_0 = eta_base - delta
            att = float(np.mean(_logistic(eta_base) - _logistic(eta_0)))
            # Delta method gradient: d(ATT)/d(β)
            #   for p ≠ int_idx: mean_i[(Λ'(η_1) - Λ'(η_0)) * X_p]
            #   for p = int_idx: mean_i[Λ'(η_1)]
            d_diff = _logistic_deriv(eta_base) - _logistic_deriv(eta_0)
            grad = np.mean(X_with_intercept[cell_mask] * d_diff[:, None], axis=0)
            grad[1 + idx] = float(np.mean(_logistic_deriv(eta_base)))
            se = float(np.sqrt(max(grad @ vcov_full @ grad, 0.0)))
            t_stat, p_value, conf_int = safe_inference(att, se, alpha=self.alpha)
            gt_effects[(g, t)] = {
                "att": att,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
                "_gradient": grad.copy(),
            }
            gt_weights[(g, t)] = int(cell_mask.sum())
            gt_grads[(g, t)] = grad

        gt_keys_ordered = [k for k in gt_keys if k in gt_effects]
        # ATT-level covariance: J @ vcov_full @ J' where J rows are per-cell gradients
        if gt_keys_ordered:
            J = np.array([gt_grads[k] for k in gt_keys_ordered])
            gt_vcov = J @ vcov_full @ J.T
        else:
            gt_vcov = None

        # Overall SE via joint delta method: ∇β(overall_att) = Σ w_k/w_total * grad_k
        post_keys = [(g, t) for (g, t) in gt_keys_ordered if t >= g]
        w_total = sum(gt_weights.get(k, 0) for k in post_keys)
        if w_total > 0 and post_keys:
            overall_att = sum(gt_weights[k] * gt_effects[k]["att"] for k in post_keys) / w_total
            agg_grad = sum((gt_weights[k] / w_total) * gt_grads[k] for k in post_keys)
            overall_se = float(np.sqrt(max(agg_grad @ vcov_full @ agg_grad, 0.0)))
            t_stat, p_value, conf_int = safe_inference(overall_att, overall_se, alpha=self.alpha)
            overall = {
                "att": overall_att,
                "se": overall_se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
            }
        else:
            overall = _compute_weighted_agg(
                gt_effects, gt_weights, gt_keys_ordered, None, self.alpha
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
            _gt_weights=gt_weights,
            _gt_vcov=gt_vcov,
            _gt_keys=gt_keys_ordered,
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
        cluster_col = self.cluster if self.cluster else unit
        cluster_ids = sample[cluster_col].values

        beta, mu_hat = solve_poisson(X_full, y)

        # QMLE sandwich vcov via shared linalg backend
        resids = y - mu_hat
        vcov_full = compute_robust_vcov(
            X_full,
            resids,
            cluster_ids=cluster_ids,
            weights=mu_hat,  # Poisson QMLE bread: (X'WX)^{-1}
            weight_type="aweight",  # unweighted scores for QMLE sandwich
        )

        # Treatment interaction coefficients: beta[1 : 1+n_int]
        beta_int = beta[1 : 1 + n_int]

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
            # Skip cells whose interaction coefficient was dropped (rank deficiency)
            delta = beta_int[idx]
            if np.isnan(delta):
                continue
            eta_base = np.clip(X_full[cell_mask] @ beta, -500, 500)
            eta_0 = eta_base - delta
            mu_1 = np.exp(eta_base)
            mu_0 = np.exp(eta_0)
            att = float(np.mean(mu_1 - mu_0))
            # Delta method gradient:
            #   for p ≠ int_idx: mean_i[(μ_1 - μ_0) * X_p]
            #   for p = int_idx: mean_i[μ_1]
            diff_mu = mu_1 - mu_0
            grad = np.mean(X_full[cell_mask] * diff_mu[:, None], axis=0)
            grad[1 + idx] = float(np.mean(mu_1))
            se = float(np.sqrt(max(grad @ vcov_full @ grad, 0.0)))
            t_stat, p_value, conf_int = safe_inference(att, se, alpha=self.alpha)
            gt_effects[(g, t)] = {
                "att": att,
                "se": se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
                "_gradient": grad.copy(),
            }
            gt_weights[(g, t)] = int(cell_mask.sum())
            gt_grads[(g, t)] = grad

        gt_keys_ordered = [k for k in gt_keys if k in gt_effects]
        # ATT-level covariance: J @ vcov_full @ J' where J rows are per-cell gradients
        if gt_keys_ordered:
            J = np.array([gt_grads[k] for k in gt_keys_ordered])
            gt_vcov = J @ vcov_full @ J.T
        else:
            gt_vcov = None

        # Overall SE via joint delta method
        post_keys = [(g, t) for (g, t) in gt_keys_ordered if t >= g]
        w_total = sum(gt_weights.get(k, 0) for k in post_keys)
        if w_total > 0 and post_keys:
            overall_att = sum(gt_weights[k] * gt_effects[k]["att"] for k in post_keys) / w_total
            agg_grad = sum((gt_weights[k] / w_total) * gt_grads[k] for k in post_keys)
            overall_se = float(np.sqrt(max(agg_grad @ vcov_full @ agg_grad, 0.0)))
            t_stat, p_value, conf_int = safe_inference(overall_att, overall_se, alpha=self.alpha)
            overall = {
                "att": overall_att,
                "se": overall_se,
                "t_stat": t_stat,
                "p_value": p_value,
                "conf_int": conf_int,
            }
        else:
            overall = _compute_weighted_agg(
                gt_effects, gt_weights, gt_keys_ordered, None, self.alpha
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
            _gt_weights=gt_weights,
            _gt_vcov=gt_vcov,
            _gt_keys=gt_keys_ordered,
        )
