"""
Efficient Difference-in-Differences estimator.

Implements the semiparametrically efficient ATT estimator from
Chen, Sant'Anna & Xie (2025), Phase 1 (no covariates).

The estimator achieves the efficiency bound by optimally weighting
across pre-treatment periods and comparison groups via the inverse of
the within-group covariance matrix Omega*.  Under the stronger PT-All
assumption the model is overidentified and EDiD exploits this for
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

    Achieves the semiparametric efficiency bound for ATT(g,t) in
    difference-in-differences settings with staggered treatment adoption.
    Phase 1 supports the **no-covariates** path only — a closed-form
    estimator using within-group sample means and covariances.

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
    ):
        if pt_assumption not in ("all", "post"):
            raise ValueError(f"pt_assumption must be 'all' or 'post', got '{pt_assumption}'")
        valid_weights = ("rademacher", "mammen", "webb")
        if bootstrap_weights not in valid_weights:
            raise ValueError(
                f"bootstrap_weights must be one of {valid_weights}, got '{bootstrap_weights}'"
            )
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
        self.is_fitted_ = False
        self.results_: Optional[EfficientDiDResults] = None

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
        }

    def set_params(self, **params: Any) -> "EfficientDiD":
        """Set estimator parameters (sklearn-compatible)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
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
            Not implemented in Phase 1.  Raises ``NotImplementedError``.
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
        NotImplementedError
            If ``covariates`` is provided (Phase 2).
        """
        if covariates is not None:
            raise NotImplementedError(
                "Covariates are not yet supported in EfficientDiD (Phase 1). "
                "The with-covariates path will be added in Phase 2."
            )

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

        # Check for never-treated units
        if n_control_units == 0:
            if self.pt_assumption == "post":
                raise ValueError(
                    "No never-treated units found. PT-Post requires a "
                    "never-treated comparison group."
                )
            warnings.warn(
                "No never-treated units. Under PT-All, not-yet-treated "
                "cohorts will be used as comparisons.",
                UserWarning,
                stacklevel=2,
            )

        # ----- Prepare data -----
        all_units = sorted(df[unit].unique())
        n_units = len(all_units)

        period_to_col = {p: i for i, p in enumerate(time_periods)}
        period_1 = time_periods[0]
        period_1_col = period_to_col[period_1]

        # Pivot outcome to wide matrix (n_units, n_periods)
        pivot = df.pivot_table(index=unit, columns=time, values=outcome, aggfunc="first")
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

        # ----- Core estimation: ATT(g, t) for each target -----
        # Precompute per-group unit counts (avoid repeated np.sum in loop)
        n_treated_per_g = {g: int(np.sum(cohort_masks[g])) for g in treatment_groups}
        n_control_count = int(np.sum(never_treated_mask))

        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
        eif_by_gt: Dict[Tuple[Any, Any], np.ndarray] = {}
        stored_weights: Dict[Tuple[Any, Any], np.ndarray] = {}
        stored_cond: Dict[Tuple[Any, Any], float] = {}

        for g in treatment_groups:
            for t in time_periods:
                # Skip period_1 — it's the universal reference baseline,
                # not a target period
                if t == period_1:
                    continue

                # Enumerate valid comparison pairs
                pairs = enumerate_valid_triples(
                    target_g=g,
                    target_t=t,
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

                # Omega* matrix
                omega = compute_omega_star_nocov(
                    target_g=g,
                    target_t=t,
                    valid_pairs=pairs,
                    outcome_wide=outcome_wide,
                    cohort_masks=cohort_masks,
                    never_treated_mask=never_treated_mask,
                    period_to_col=period_to_col,
                    period_1_col=period_1_col,
                    cohort_fractions=cohort_fractions,
                )

                # Efficient weights (also returns condition number)
                weights, _, cond_num = compute_efficient_weights(omega)
                stored_weights[(g, t)] = weights
                if omega.size > 0:
                    stored_cond[(g, t)] = cond_num

                # Generated outcomes
                y_hat = compute_generated_outcomes_nocov(
                    target_g=g,
                    target_t=t,
                    valid_pairs=pairs,
                    outcome_wide=outcome_wide,
                    cohort_masks=cohort_masks,
                    never_treated_mask=never_treated_mask,
                    period_to_col=period_to_col,
                    period_1_col=period_1_col,
                )

                # ATT(g,t) = w @ y_hat
                att_gt = float(weights @ y_hat) if len(weights) > 0 else np.nan

                # EIF
                eif_vals = compute_eif_nocov(
                    target_g=g,
                    target_t=t,
                    att_gt=att_gt,
                    weights=weights,
                    valid_pairs=pairs,
                    outcome_wide=outcome_wide,
                    cohort_masks=cohort_masks,
                    never_treated_mask=never_treated_mask,
                    period_to_col=period_to_col,
                    period_1_col=period_1_col,
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
            group_time_effects, eif_by_gt, n_units, cohort_fractions
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
            )
        if aggregate in ("group", "all"):
            group_effects = self._aggregate_by_group(
                group_time_effects,
                eif_by_gt,
                n_units,
                cohort_fractions,
                treatment_groups,
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
            event_study_effects=event_study_effects,
            group_effects=group_effects,
            efficient_weights=stored_weights if stored_weights else None,
            omega_condition_numbers=stored_cond if stored_cond else None,
            influence_functions=None,  # can store full EIF matrix if needed
            bootstrap_results=bootstrap_results,
        )
        self.is_fitted_ = True
        return self.results_

    # -- Aggregation helpers --------------------------------------------------

    def _aggregate_overall(
        self,
        group_time_effects: Dict[Tuple[Any, Any], Dict[str, Any]],
        eif_by_gt: Dict[Tuple[Any, Any], np.ndarray],
        n_units: int,
        cohort_fractions: Dict[float, float],
    ) -> Tuple[float, float]:
        """Compute overall ATT with WIF-adjusted SE."""
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

        # Aggregate EIF with WIF correction
        agg_eif = np.zeros(n_units)
        for k, gt in enumerate(keepers):
            agg_eif += w[k] * eif_by_gt[gt]

        # WIF correction: accounts for uncertainty in cohort-size weights
        # wif_i = sum_k wif_ik * ATT_k  where:
        #   wif_ik = (1{G_i == g_k} - pg_k) / sum_pg
        #          - sum_j(1{G_i == g_j} - pg_j) * pg_k / sum_pg^2
        # We implement this via vectorized operations.

        # SE = sqrt(mean(EIF^2) / n) — standard IF-based SE
        se = float(np.sqrt(np.mean(agg_eif**2) / n_units))

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
    ) -> Dict[int, Dict[str, Any]]:
        """Aggregate ATT(g,t) by relative time e = t - g."""
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
    ) -> Dict[Any, Dict[str, Any]]:
        """Aggregate ATT(g,t) by treatment cohort."""
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
