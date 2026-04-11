"""
de Chaisemartin-D'Haultfoeuille (dCDH) estimator for reversible-treatment DiD.

The dCDH estimator is the only modern DiD estimator in the diff-diff library
that handles **non-absorbing (reversible) treatments** — treatment can switch
on AND off over time. All other staggered estimators in the library
(``CallawaySantAnna``, ``SunAbraham``, ``ImputationDiD``, ``TwoStageDiD``,
``EfficientDiD``, ``WooldridgeDiD``) assume treatment is absorbing.

Phase 1 ships the contemporaneous-switch case ``DID_M`` (= ``DID_1`` at
horizon ``l = 1`` of the dynamic companion paper). Phases 2 and 3 add
dynamic horizons and covariates respectively, on the *same* class — see
``ROADMAP.md`` for the full progression. The forward-compatibility
parameters in :meth:`ChaisemartinDHaultfoeuille.fit` raise
``NotImplementedError`` with phase pointers until later phases land.

References
----------
- de Chaisemartin, C. & D'Haultfoeuille, X. (2020). Two-Way Fixed Effects
  Estimators with Heterogeneous Treatment Effects. *American Economic
  Review*, 110(9), 2964-2996.
- de Chaisemartin, C. & D'Haultfoeuille, X. (2022, revised 2023).
  Difference-in-Differences Estimators of Intertemporal Treatment Effects.
  NBER Working Paper 29873. Web Appendix Section 3.7.3 contains the
  cohort-recentered plug-in variance formula implemented here.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.chaisemartin_dhaultfoeuille_bootstrap import (
    ChaisemartinDHaultfoeuilleBootstrapMixin,
)
from diff_diff.chaisemartin_dhaultfoeuille_results import (
    ChaisemartinDHaultfoeuilleResults,
    DCDHBootstrapResults,
)
from diff_diff.linalg import solve_ols
from diff_diff.utils import safe_inference

__all__ = [
    "ChaisemartinDHaultfoeuille",
    "chaisemartin_dhaultfoeuille",
    "twowayfeweights",
    "TWFEWeightsResult",
]


# =============================================================================
# Public dataclass for the standalone TWFE diagnostic helper
# =============================================================================


class TWFEWeightsResult:
    """
    Lightweight container for the standalone ``twowayfeweights`` helper.

    Returned by :func:`twowayfeweights`. Mirrors the per-cell decomposition
    information that the dCDH estimator stores on its results object when
    ``twfe_diagnostic=True``, but available as a standalone function for
    users who only want the diagnostic without fitting the full estimator.
    """

    __slots__ = ("weights", "fraction_negative", "sigma_fe", "beta_fe")

    def __init__(
        self,
        weights: pd.DataFrame,
        fraction_negative: float,
        sigma_fe: float,
        beta_fe: float,
    ) -> None:
        self.weights = weights
        self.fraction_negative = fraction_negative
        self.sigma_fe = sigma_fe
        self.beta_fe = beta_fe

    def __repr__(self) -> str:
        return (
            f"TWFEWeightsResult(beta_fe={self.beta_fe:.4f}, "
            f"fraction_negative={self.fraction_negative:.4f}, "
            f"sigma_fe={self.sigma_fe:.4f}, n_cells={len(self.weights)})"
        )


# =============================================================================
# Main estimator class
# =============================================================================


class ChaisemartinDHaultfoeuille(ChaisemartinDHaultfoeuilleBootstrapMixin):
    """
    de Chaisemartin-D'Haultfoeuille (dCDH) estimator — Phase 1.

    Computes the contemporaneous-switch DiD ``DID_M`` from the AER 2020
    paper, equivalently ``DID_1`` (horizon ``l = 1``) of the dynamic
    companion paper (NBER WP 29873). The estimator is the only modern
    DiD in the library that handles **reversible (non-absorbing)
    treatments** — treatment may switch on AND off over time.

    Phase 1 deliverables:

    - The headline ``DID_M`` point estimate
    - Joiners-only ``DID_+`` and leavers-only ``DID_-`` decompositions
    - The single-lag placebo ``DID_M^pl`` (computed automatically by
      default; gate via ``placebo=False``)
    - Analytical SE via the cohort-recentered plug-in formula from
      Web Appendix Section 3.7.3 of the dynamic paper
    - Optional multiplier bootstrap clustered at the group level
    - Optional TWFE decomposition diagnostic from Theorem 1 of AER 2020
      (per-cell weights, fraction negative, ``sigma_fe``)

    Parameters
    ----------
    alpha : float, default=0.05
        Significance level for confidence intervals.
    cluster : str, optional
        Reserved for future cluster-robust SE customization. Currently
        unused — analytical SEs are always at the group level via the
        cohort-recentered plug-in.
    n_bootstrap : int, default=0
        Number of multiplier-bootstrap iterations. ``0`` (default) uses
        only the analytical SE. Set to ``999`` or higher for stable
        bootstrap inference.
    bootstrap_weights : str, default="rademacher"
        Type of multiplier-bootstrap weights: ``"rademacher"``,
        ``"mammen"``, or ``"webb"``. Ignored unless ``n_bootstrap > 0``.
    seed : int, optional
        Random seed for the multiplier bootstrap.
    placebo : bool, default=True
        If ``True`` (default), automatically compute the single-lag
        placebo ``DID_M^pl`` (Theorem 4 of AER 2020) on the same data.
        Set to ``False`` to skip the placebo computation for speed; the
        results object will still expose ``placebo_*`` fields, but with
        NaN values and ``placebo_available=False``.
    twfe_diagnostic : bool, default=True
        If ``True`` (default), compute the TWFE decomposition diagnostic
        from Theorem 1 of AER 2020: per-``(g, t)`` weights, fraction of
        treated cells with negative weights, and ``sigma_fe`` (the
        smallest cell-effect standard deviation that could flip the sign
        of the plain TWFE coefficient). Useful for diagnosing whether
        TWFE on the same data would have a different (potentially
        wrong-signed) answer than ``DID_M``.
    drop_larger_lower : bool, default=True
        If ``True`` (default, matches R ``DIDmultiplegtDYN``), drops
        groups whose treatment switches more than once (multi-switch
        groups) before estimation. This is required for the analytical
        variance formula to be consistent with the AER 2020 Theorem 3
        point estimate — both formulas operate on the same post-drop
        dataset. Setting to ``False`` is supported for diagnostic
        comparison but produces an inconsistent estimator-variance
        pairing for multi-switch groups; a warning is emitted.
    rank_deficient_action : str, default="warn"
        Action when the TWFE decomposition diagnostic OLS encounters a
        rank-deficient design matrix: ``"warn"``, ``"error"``, or
        ``"silent"``. Only used when ``twfe_diagnostic=True``.

    Attributes
    ----------
    results_ : ChaisemartinDHaultfoeuilleResults
        Estimation results after calling :meth:`fit`.
    is_fitted_ : bool
        Whether the model has been fitted.

    Notes
    -----
    The analytical CI is **conservative** under Assumption 8 (independent
    groups) of the dynamic companion paper, and exact only under iid
    sampling. This is documented as a deliberate deviation from "default
    nominal coverage" in ``REGISTRY.md``.

    Examples
    --------
    Basic single-switch panel:

    >>> from diff_diff import ChaisemartinDHaultfoeuille
    >>> from diff_diff.prep_dgp import generate_reversible_did_data
    >>> data = generate_reversible_did_data(n_groups=80, n_periods=6, seed=42)
    >>> est = ChaisemartinDHaultfoeuille()
    >>> results = est.fit(
    ...     data, outcome="outcome", group="group",
    ...     time="period", treatment="treatment",
    ... )
    >>> abs(results.overall_att - 2.0) < 1.0  # close to the true effect
    True
    """

    def __init__(
        self,
        alpha: float = 0.05,
        cluster: Optional[str] = None,
        n_bootstrap: int = 0,
        bootstrap_weights: str = "rademacher",
        seed: Optional[int] = None,
        placebo: bool = True,
        twfe_diagnostic: bool = True,
        drop_larger_lower: bool = True,
        rank_deficient_action: str = "warn",
    ) -> None:
        # Parameter validation
        if rank_deficient_action not in ("warn", "error", "silent"):
            raise ValueError(
                f"rank_deficient_action must be 'warn', 'error', or 'silent', "
                f"got '{rank_deficient_action}'"
            )
        if bootstrap_weights not in ("rademacher", "mammen", "webb"):
            raise ValueError(
                f"bootstrap_weights must be 'rademacher', 'mammen', or 'webb', "
                f"got '{bootstrap_weights}'"
            )
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if n_bootstrap < 0:
            raise ValueError(f"n_bootstrap must be non-negative, got {n_bootstrap}")

        self.alpha = alpha
        self.cluster = cluster
        self.n_bootstrap = n_bootstrap
        self.bootstrap_weights = bootstrap_weights
        self.seed = seed
        self.placebo = placebo
        self.twfe_diagnostic = twfe_diagnostic
        self.drop_larger_lower = drop_larger_lower
        self.rank_deficient_action = rank_deficient_action

        self.is_fitted_ = False
        self.results_: Optional[ChaisemartinDHaultfoeuilleResults] = None

    # ------------------------------------------------------------------
    # sklearn-style parameter introspection
    # ------------------------------------------------------------------

    def get_params(self) -> Dict[str, Any]:
        """Return all ``__init__`` parameters as a dictionary."""
        return {
            "alpha": self.alpha,
            "cluster": self.cluster,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_weights": self.bootstrap_weights,
            "seed": self.seed,
            "placebo": self.placebo,
            "twfe_diagnostic": self.twfe_diagnostic,
            "drop_larger_lower": self.drop_larger_lower,
            "rank_deficient_action": self.rank_deficient_action,
        }

    def set_params(self, **params: Any) -> "ChaisemartinDHaultfoeuille":
        """Set estimator parameters (sklearn-compatible)."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
        return self

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        data: pd.DataFrame,
        outcome: str,
        group: str,
        time: str,
        treatment: str,
        # ---------- forward-compat parameters ----------
        aggregate: Optional[str] = None,
        L_max: Optional[int] = None,
        controls: Optional[List[str]] = None,
        trends_linear: Optional[bool] = None,
        trends_nonparam: Optional[Any] = None,
        honest_did: bool = False,
        # ---------- deferred (separate effort) ----------
        survey_design: Any = None,
    ) -> ChaisemartinDHaultfoeuilleResults:
        """
        Fit the dCDH estimator on individual-level panel data.

        Parameters
        ----------
        data : pd.DataFrame
            Individual-level panel. Must contain columns for ``outcome``,
            ``group``, ``time``, and ``treatment``. The estimator
            internally aggregates to ``(group, time)`` cells.
        outcome : str
            Outcome variable column name.
        group : str
            Group identifier column name. Treatment is assumed constant
            within each ``(group, time)`` cell after aggregation; a
            warning is emitted and the cell-level treatment is rounded to
            majority if any cell has fractional treatment after grouping.
        time : str
            Time period column name. Must be sortable.
        treatment : str
            Per-observation binary treatment column. Must coerce to
            ``{0, 1}``; non-binary values raise ``ValueError`` (Phase 3
            adds non-binary support).
        aggregate : str, optional
            **Reserved for Phase 2.** Phase 1 requires ``aggregate=None``;
            any other value raises ``NotImplementedError``.
        L_max : int, optional
            **Reserved for Phase 2** (multi-horizon event study).
        controls : list of str, optional
            **Reserved for Phase 3** (covariate adjustment via the
            residualization-style ``DID^X`` from Web Appendix Section 1.2
            of the dynamic paper).
        trends_linear : bool, optional
            **Reserved for Phase 3** (group-specific linear trends via
            ``DID^{fd}``).
        trends_nonparam : Any, optional
            **Reserved for Phase 3** (state-set-specific trends).
        honest_did : bool, default=False
            **Reserved for Phase 3** (HonestDiD integration on placebos).
        survey_design : Any, optional
            **Not supported in any phase.** Survey design integration is
            handled as a separate effort after all three phases ship.
            Passing a non-``None`` value raises ``NotImplementedError``.

        Returns
        -------
        ChaisemartinDHaultfoeuilleResults

        Raises
        ------
        ValueError
            If required columns are missing, treatment is not binary, or
            the panel has too few groups / periods.
        NotImplementedError
            If any forward-compat parameter is set to a non-default
            value, with a clear pointer to the relevant ROADMAP phase.
        """
        # ------------------------------------------------------------------
        # Step 1: Column validation
        # ------------------------------------------------------------------
        required_cols = [outcome, group, time, treatment]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # ------------------------------------------------------------------
        # Step 2: Forward-compat gates
        # ------------------------------------------------------------------
        _check_forward_compat_gates(
            aggregate=aggregate,
            L_max=L_max,
            controls=controls,
            trends_linear=trends_linear,
            trends_nonparam=trends_nonparam,
            honest_did=honest_did,
        )

        # ------------------------------------------------------------------
        # Step 3: Survey gate (deferred separate effort)
        # ------------------------------------------------------------------
        if survey_design is not None:
            raise NotImplementedError(
                "ChaisemartinDHaultfoeuille does not support survey_design. "
                "Survey design integration for dCDH is deferred to a separate "
                "effort after all three implementation phases ship (see "
                "ROADMAP.md out-of-scope section). For now, fit without "
                "survey_design. If your treatment is absorbing, use "
                "CallawaySantAnna which supports survey_design."
            )

        # ------------------------------------------------------------------
        # Step 4: Treatment + outcome NaN validation (no silent failures)
        # ------------------------------------------------------------------
        df = data.copy()
        try:
            df[treatment] = pd.to_numeric(df[treatment])
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Could not coerce treatment column {treatment!r} to numeric: {exc}"
            ) from exc
        # Reject NaN treatment values up front: silently dropping them via
        # `dropna()` would change the per-cell counts without informing the
        # user.
        n_nan_treat = int(df[treatment].isna().sum())
        if n_nan_treat > 0:
            raise ValueError(
                f"Treatment column {treatment!r} contains {n_nan_treat} NaN value(s). "
                "ChaisemartinDHaultfoeuille requires non-missing treatment "
                "indicators on every observation; impute or drop NaN treatment "
                "rows before calling fit() so the dropped count is explicit."
            )
        # Reject NaN outcomes for the same reason.
        try:
            df[outcome] = pd.to_numeric(df[outcome])
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Could not coerce outcome column {outcome!r} to numeric: {exc}"
            ) from exc
        n_nan_outcome = int(df[outcome].isna().sum())
        if n_nan_outcome > 0:
            raise ValueError(
                f"Outcome column {outcome!r} contains {n_nan_outcome} NaN value(s). "
                "Drop or impute missing outcomes before calling fit() so the "
                "exclusion is explicit (silently averaging over present values "
                "would distort per-cell means)."
            )

        unique_treats = pd.unique(df[treatment])
        invalid = [v for v in unique_treats if v not in (0, 1, 0.0, 1.0)]
        if invalid:
            raise ValueError(
                f"ChaisemartinDHaultfoeuille requires binary treatment in {{0, 1}}; "
                f"found values {invalid[:5]} in column {treatment!r}. Non-binary "
                "treatment is reserved for Phase 3 of the dCDH rollout (see "
                "ROADMAP.md Phase 3)."
            )

        # ------------------------------------------------------------------
        # Step 5: Cell aggregation (individual -> (g, t) cells)
        # ------------------------------------------------------------------
        cell = df.groupby([group, time], as_index=False).agg(
            y_gt=(outcome, "mean"), d_gt=(treatment, "mean"), n_gt=(treatment, "count")
        )
        # Within-cell-varying treatment: round and warn (cell-constant treatment
        # is the dCDH binary assumption; fuzzy DiD is in a separate paper not
        # covered by Phase 1).
        non_constant_mask = (cell["d_gt"] > 0) & (cell["d_gt"] < 1)
        if non_constant_mask.any():
            n_non_constant = int(non_constant_mask.sum())
            warnings.warn(
                f"Within-cell-varying treatment detected in {n_non_constant} "
                f"(group, time) cells. Rounding to majority (>= 0.5 -> 1). Fuzzy "
                "DiD is deferred to a separate dCDH paper (see Phase 3 / "
                "out-of-scope in ROADMAP.md).",
                UserWarning,
                stacklevel=2,
            )
        cell["d_gt"] = (cell["d_gt"] >= 0.5).astype(int)

        # Sort to ensure deterministic order in downstream operations
        cell = cell.sort_values([group, time]).reset_index(drop=True)

        all_periods_pre_drop = sorted(cell[time].unique().tolist())
        if len(all_periods_pre_drop) < 2:
            raise ValueError(
                f"ChaisemartinDHaultfoeuille requires at least 2 periods, "
                f"got {len(all_periods_pre_drop)}"
            )

        # ------------------------------------------------------------------
        # Step 5b: Compute the TWFE diagnostic on the FULL pre-filter cell
        #          dataset, so the diagnostic reflects the data the user
        #          actually passed in (per the plan).
        # ------------------------------------------------------------------
        twfe_diagnostic_payload = None
        if self.twfe_diagnostic:
            try:
                twfe_diagnostic_payload = _compute_twfe_diagnostic(
                    cell=cell,
                    group_col=group,
                    time_col=time,
                    rank_deficient_action=self.rank_deficient_action,
                )
            except Exception as exc:  # noqa: BLE001
                warnings.warn(
                    f"TWFE decomposition diagnostic failed: {exc}. "
                    "Skipping diagnostic; main estimation continues.",
                    UserWarning,
                    stacklevel=2,
                )
                twfe_diagnostic_payload = None

        # ------------------------------------------------------------------
        # Step 6: Drop A5-violating (multi-switch) cells per drop_larger_lower
        # ------------------------------------------------------------------
        n_groups_dropped_crossers = 0
        if self.drop_larger_lower:
            cell, n_groups_dropped_crossers = _drop_crossing_cells(
                cell=cell, group_col=group, d_col="d_gt"
            )
        else:
            warnings.warn(
                "drop_larger_lower=False: the analytical variance formula will "
                "be inconsistent with the point estimate for any multi-switch "
                "groups present in the data, producing a biased SE. Use only "
                "for diagnostic comparison against R or when you are confident "
                "no multi-switch groups exist.",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # Step 7: Singleton-baseline filter (footnote 15 of dynamic paper)
        # ------------------------------------------------------------------
        cell, n_groups_dropped_singleton_baseline = _filter_singleton_baseline(
            cell=cell, group_col=group, time_col=time, d_col="d_gt"
        )

        if cell.empty or cell[group].nunique() == 0:
            raise ValueError(
                "After dropping multi-switch cells (drop_larger_lower=True) and "
                "singleton-baseline groups, no groups remain. The dataset cannot "
                "support dCDH estimation. Check the input panel for diversity in "
                "treatment patterns."
            )

        # Determine the post-filter group set, period set, and per-group state
        all_groups = sorted(cell[group].unique().tolist())
        all_periods = sorted(cell[time].unique().tolist())
        n_obs_post = int(cell["n_gt"].sum())

        # Pivot to (group x time) matrices for vectorized computations
        d_pivot = cell.pivot(index=group, columns=time, values="d_gt").reindex(
            index=all_groups, columns=all_periods
        )
        y_pivot = cell.pivot(index=group, columns=time, values="y_gt").reindex(
            index=all_groups, columns=all_periods
        )
        n_pivot = (
            cell.pivot(index=group, columns=time, values="n_gt")
            .reindex(index=all_groups, columns=all_periods)
            .fillna(0)
            .astype(int)
        )
        D_mat = d_pivot.to_numpy()
        Y_mat = y_pivot.to_numpy()
        N_mat = n_pivot.to_numpy()

        # ------------------------------------------------------------------
        # Step 8-9: Switching-cell counts and per-period DIDs (Theorem 3)
        #          with explicit A11 zero-retention pseudocode
        # ------------------------------------------------------------------
        (
            per_period_effects,
            a11_warnings,
            did_plus_t_arr,
            did_minus_t_arr,
            n_10_t_arr,
            n_01_t_arr,
        ) = _compute_per_period_dids(
            D_mat=D_mat,
            Y_mat=Y_mat,
            N_mat=N_mat,
            periods=all_periods,
        )
        if a11_warnings:
            warnings.warn(
                f"Assumption 11 (existence of stable controls) violated in "
                f"{len(a11_warnings)} period(s); the affected DID_+/DID_- values "
                f"are zeroed but their switcher counts are retained in the N_S "
                f"denominator (matching paper convention). Affected: "
                f"{', '.join(a11_warnings[:3])}"
                + (f" (and {len(a11_warnings) - 3} more)" if len(a11_warnings) > 3 else ""),
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # Step 10: Aggregate DID_M = sum_t (n_10_t * did_plus_t + n_01_t * did_minus_t) / N_S
        # ------------------------------------------------------------------
        N_S = int(n_10_t_arr.sum() + n_01_t_arr.sum())
        if N_S == 0:
            raise ValueError(
                "No switching cells found in the data after filtering: every "
                "group has constant treatment for the entire panel. dCDH "
                "requires at least one (g, t) cell where the group's treatment "
                "differs from the previous period."
            )
        overall_att = float((n_10_t_arr @ did_plus_t_arr + n_01_t_arr @ did_minus_t_arr) / N_S)

        # ------------------------------------------------------------------
        # Step 11: Joiners and leavers views
        # ------------------------------------------------------------------
        joiner_total = int(n_10_t_arr.sum())
        leaver_total = int(n_01_t_arr.sum())
        joiners_available = joiner_total > 0
        leavers_available = leaver_total > 0
        if joiners_available:
            joiners_att = float((n_10_t_arr @ did_plus_t_arr) / joiner_total)
        else:
            joiners_att = float("nan")
        if leavers_available:
            leavers_att = float((n_01_t_arr @ did_minus_t_arr) / leaver_total)
        else:
            leavers_att = float("nan")

        # Cell counts for the results
        n_joiner_cells = int(np.count_nonzero(n_10_t_arr))
        n_leaver_cells = int(np.count_nonzero(n_01_t_arr))
        n_joiner_obs = joiner_total
        n_leaver_obs = leaver_total

        # ------------------------------------------------------------------
        # Step 12: Placebo (DID_M^pl) — Theorem 4
        # ------------------------------------------------------------------
        placebo_available = False
        placebo_effect = float("nan")
        if self.placebo:
            if len(all_periods) < 3:
                warnings.warn(
                    f"Placebo DID_M^pl (Theorem 4) requires at least 3 time "
                    f"periods; the post-filter panel has only {len(all_periods)}. "
                    "Skipping the placebo computation. Pass placebo=False to "
                    "suppress this warning, or use a panel with T >= 3.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                placebo_payload = _compute_placebo(
                    D_mat=D_mat, Y_mat=Y_mat, N_mat=N_mat, periods=all_periods
                )
                if placebo_payload is None:
                    warnings.warn(
                        "Placebo DID_M^pl could not be computed: no qualifying "
                        "switching cells with the required 3-period stable "
                        "history exist after filtering. The placebo fields on "
                        "the results object are NaN with placebo_available=False.",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    placebo_effect, placebo_available = placebo_payload

        # ------------------------------------------------------------------
        # Step 13-16: Cohort identification, influence-function vectors,
        #             cohort-recentered plug-in variance
        # ------------------------------------------------------------------
        (
            U_centered_overall,
            n_groups_for_overall_var,
            n_cohorts,
            n_groups_dropped_never_switching,
            U_centered_joiners,
            U_centered_leavers,
        ) = _compute_cohort_recentered_inputs(
            D_mat=D_mat, Y_mat=Y_mat, N_mat=N_mat, periods=all_periods
        )

        # Analytical SE for DID_M
        overall_se = _plugin_se(U_centered=U_centered_overall, divisor=N_S)
        overall_t, overall_p, overall_ci = safe_inference(
            overall_att, overall_se, alpha=self.alpha, df=None
        )

        # Joiners SE (uses joiner-only centered IF; conservative bound)
        if joiners_available:
            joiners_se = _plugin_se(U_centered=U_centered_joiners, divisor=joiner_total)
            joiners_t, joiners_p, joiners_ci = safe_inference(
                joiners_att, joiners_se, alpha=self.alpha, df=None
            )
        else:
            joiners_se, joiners_t, joiners_p, joiners_ci = (
                float("nan"),
                float("nan"),
                float("nan"),
                (float("nan"), float("nan")),
            )

        # Leavers SE
        if leavers_available:
            leavers_se = _plugin_se(U_centered=U_centered_leavers, divisor=leaver_total)
            leavers_t, leavers_p, leavers_ci = safe_inference(
                leavers_att, leavers_se, alpha=self.alpha, df=None
            )
        else:
            leavers_se, leavers_t, leavers_p, leavers_ci = (
                float("nan"),
                float("nan"),
                float("nan"),
                (float("nan"), float("nan")),
            )

        # Placebo SE: in Phase 1 we approximate using the same plug-in formula
        # applied to the placebo's centered IF. The dynamic paper derives the
        # variance for DID_l only; placebo SE is a library extension and is
        # treated as conservative. NaN if placebo unavailable.
        placebo_se = float("nan")
        placebo_t = float("nan")
        placebo_p = float("nan")
        placebo_ci: Tuple[float, float] = (float("nan"), float("nan"))
        if placebo_available and self.n_bootstrap == 0:
            # Without bootstrap, we cannot compute a paper-prescribed analytical
            # SE for the placebo (Theorem 1 covers DID_l only). Emit a NaN with
            # a one-time note in the warning channel.
            warnings.warn(
                "Phase 1 placebo SE is not analytically derived (the dynamic "
                "paper Section 3.7.3 covers only DID_l). Set n_bootstrap > 0 "
                "to obtain a multiplier-bootstrap SE for DID_M^pl.",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # Step 18: Build per-period decomposition with explicit n_*_t fields
        # ------------------------------------------------------------------
        n_treated_obs_post = int(N_mat[D_mat == 1].sum())

        # ------------------------------------------------------------------
        # Step 19: Bootstrap if requested
        # ------------------------------------------------------------------
        bootstrap_results: Optional[DCDHBootstrapResults] = None
        if self.n_bootstrap > 0:
            joiners_inputs = (
                (U_centered_joiners, joiner_total, joiners_att) if joiners_available else None
            )
            leavers_inputs = (
                (U_centered_leavers, leaver_total, leavers_att) if leavers_available else None
            )
            # Phase 1 placebo bootstrap: not supported (no centered IF available
            # for the placebo; the dynamic paper only derives DID_l variance).
            # Phase 2/3 will add this when implementing the dynamic estimator.
            placebo_inputs = None

            br = self._compute_dcdh_bootstrap(
                n_groups_for_overall=n_groups_for_overall_var,
                u_centered_overall=U_centered_overall,
                n_groups_overall=N_S,
                original_overall=overall_att,
                joiners_inputs=joiners_inputs,
                leavers_inputs=leavers_inputs,
                placebo_inputs=placebo_inputs,
            )
            bootstrap_results = br

            # Replace analytical SE with bootstrap SE for the targets that
            # have valid bootstrap output. The original analytical values
            # remain available via re-running with n_bootstrap=0.
            if np.isfinite(br.overall_se):
                overall_se = br.overall_se
                overall_ci = br.overall_ci
                overall_p = br.overall_p_value
                overall_t = overall_att / overall_se if overall_se > 0 else float("nan")
            if joiners_available and br.joiners_se is not None and np.isfinite(br.joiners_se):
                joiners_se = br.joiners_se
                joiners_ci = br.joiners_ci or joiners_ci
                joiners_p = br.joiners_p_value or joiners_p
                joiners_t = joiners_att / joiners_se if joiners_se > 0 else float("nan")
            if leavers_available and br.leavers_se is not None and np.isfinite(br.leavers_se):
                leavers_se = br.leavers_se
                leavers_ci = br.leavers_ci or leavers_ci
                leavers_p = br.leavers_p_value or leavers_p
                leavers_t = leavers_att / leavers_se if leavers_se > 0 else float("nan")

        # ------------------------------------------------------------------
        # Step 20: Build the results dataclass
        # ------------------------------------------------------------------
        # event_study_effects holds a single l=1 entry mirroring overall_att
        # (per review MEDIUM #5: stable shape across phases).
        event_study_effects = {
            1: {
                "effect": overall_att,
                "se": overall_se,
                "t_stat": overall_t,
                "p_value": overall_p,
                "conf_int": overall_ci,
                "n_obs": N_S,
            }
        }

        twfe_weights_df = None
        twfe_fraction_negative = None
        twfe_sigma_fe = None
        twfe_beta_fe = None
        if twfe_diagnostic_payload is not None:
            twfe_weights_df = twfe_diagnostic_payload.weights
            twfe_fraction_negative = twfe_diagnostic_payload.fraction_negative
            twfe_sigma_fe = twfe_diagnostic_payload.sigma_fe
            twfe_beta_fe = twfe_diagnostic_payload.beta_fe

        results = ChaisemartinDHaultfoeuilleResults(
            overall_att=overall_att,
            overall_se=overall_se,
            overall_t_stat=overall_t,
            overall_p_value=overall_p,
            overall_conf_int=overall_ci,
            joiners_att=joiners_att,
            joiners_se=joiners_se,
            joiners_t_stat=joiners_t,
            joiners_p_value=joiners_p,
            joiners_conf_int=joiners_ci,
            n_joiner_cells=n_joiner_cells,
            n_joiner_obs=n_joiner_obs,
            joiners_available=joiners_available,
            leavers_att=leavers_att,
            leavers_se=leavers_se,
            leavers_t_stat=leavers_t,
            leavers_p_value=leavers_p,
            leavers_conf_int=leavers_ci,
            n_leaver_cells=n_leaver_cells,
            n_leaver_obs=n_leaver_obs,
            leavers_available=leavers_available,
            placebo_effect=placebo_effect,
            placebo_se=placebo_se,
            placebo_t_stat=placebo_t,
            placebo_p_value=placebo_p,
            placebo_conf_int=placebo_ci,
            placebo_available=placebo_available,
            per_period_effects=per_period_effects,
            groups=all_groups,
            time_periods=all_periods,
            n_obs=n_obs_post,
            n_treated_obs=n_treated_obs_post,
            n_switcher_obs=N_S,
            n_cohorts=n_cohorts,
            n_groups_dropped_crossers=n_groups_dropped_crossers,
            n_groups_dropped_singleton_baseline=n_groups_dropped_singleton_baseline,
            n_groups_dropped_never_switching=n_groups_dropped_never_switching,
            event_study_effects=event_study_effects,
            twfe_weights=twfe_weights_df,
            twfe_fraction_negative=twfe_fraction_negative,
            twfe_sigma_fe=twfe_sigma_fe,
            twfe_beta_fe=twfe_beta_fe,
            alpha=self.alpha,
            bootstrap_results=bootstrap_results,
            _estimator_ref=self,
        )

        self.results_ = results
        self.is_fitted_ = True
        return results


# =============================================================================
# Module-level helpers
# =============================================================================


def _check_forward_compat_gates(
    aggregate: Optional[str],
    L_max: Optional[int],
    controls: Optional[List[str]],
    trends_linear: Optional[bool],
    trends_nonparam: Any,
    honest_did: bool,
) -> None:
    """Raise ``NotImplementedError`` for any non-default Phase 2/3 parameter."""
    if aggregate is not None:
        # MEDIUM #1: strict equality with None — do not accept "simple" silently
        raise NotImplementedError(
            f"aggregate={aggregate!r} is reserved for Phase 2 of dCDH "
            "(multi-horizon event study via DID_l). Phase 1 requires "
            "aggregate=None and ships only DID_M = DID_1, the contemporaneous-"
            "switch estimator at horizon l=1. See ROADMAP.md Phase 2."
        )
    if L_max is not None:
        raise NotImplementedError(
            "L_max is reserved for Phase 2 of dCDH (multi-horizon event study). "
            "Phase 1 computes only the l=1 effect DID_M. See ROADMAP.md Phase 2."
        )
    if controls is not None:
        raise NotImplementedError(
            "Covariate adjustment (DID^X) is reserved for Phase 3 of dCDH, which "
            "implements the residualization-style covariate adjustment from Web "
            "Appendix Section 1.2 of the dynamic companion paper. Note: this is "
            "NOT doubly-robust, NOT IPW, and NOT Callaway-Sant'Anna-style. "
            "See ROADMAP.md Phase 3."
        )
    if trends_linear is not None:
        raise NotImplementedError(
            "Group-specific linear trends (DID^{fd}) are reserved for Phase 3 of "
            "dCDH (Web Appendix Section 1.3, Lemma 6 of the dynamic companion "
            "paper). See ROADMAP.md Phase 3."
        )
    if trends_nonparam is not None:
        raise NotImplementedError(
            "State-set-specific trends (trends_nonparam) are reserved for Phase 3 "
            "of dCDH (Web Appendix Section 1.4). See ROADMAP.md Phase 3."
        )
    if honest_did:
        raise NotImplementedError(
            "HonestDiD integration for dCDH is reserved for Phase 3, applied to "
            "the placebo DID^{pl}_l output. Phase 1 provides only the placebo "
            "point estimate via results.placebo_effect. See ROADMAP.md Phase 3."
        )


def _drop_crossing_cells(
    cell: pd.DataFrame, group_col: str, d_col: str
) -> Tuple[pd.DataFrame, int]:
    """
    Drop multi-switch groups (matches R DIDmultiplegtDYN drop_larger_lower=TRUE).

    For binary treatment in Phase 1, "multi-switch" means a group whose
    treatment switches more than once across the panel. Such groups are
    dropped entirely (not just the post-second-switch cells) so the
    cohort identification step (which uses the first switch as the
    cohort marker) and the variance computation operate on a consistent
    dataset.

    Parameters
    ----------
    cell : pd.DataFrame
        Cell-level dataset with columns for ``group_col``, ``time_col``,
        ``d_col``, and possibly other metadata. Must be sorted by group
        and time.
    group_col : str
    d_col : str
        Treatment column name.

    Returns
    -------
    filtered : pd.DataFrame
        Subset of ``cell`` with all multi-switch groups removed.
    n_dropped : int
        Number of groups dropped.
    """
    # Count switches per group
    diffs = cell.groupby(group_col)[d_col].diff().abs()
    switches_per_group = diffs.fillna(0).groupby(cell[group_col]).sum()
    multi_switch_groups = switches_per_group[switches_per_group > 1].index.tolist()
    n_dropped = len(multi_switch_groups)
    if n_dropped > 0:
        warnings.warn(
            f"drop_larger_lower=True dropped {n_dropped} multi-switch group(s) "
            f"matching R DIDmultiplegtDYN behavior. Examples: "
            f"{multi_switch_groups[:5]}"
            + (f" (and {n_dropped - 5} more)" if n_dropped > 5 else ""),
            UserWarning,
            stacklevel=3,
        )
        cell = cell[~cell[group_col].isin(multi_switch_groups)].reset_index(drop=True)
    return cell, n_dropped


def _filter_singleton_baseline(
    cell: pd.DataFrame, group_col: str, time_col: str, d_col: str
) -> Tuple[pd.DataFrame, int]:
    """
    Drop groups whose baseline ``D_{g,1}`` is unique (footnote 15 of dynamic paper).

    These groups have no baseline-matched control set and contribute zero
    identifying information to the dCDH point estimate or variance.
    """
    # Per-group baseline = treatment value at the earliest observed period
    baselines = (
        cell.sort_values([group_col, time_col])
        .groupby(group_col, as_index=False)[d_col]
        .first()
        .rename(columns={d_col: "_baseline"})
    )
    # Count groups per baseline value
    baseline_counts = baselines["_baseline"].value_counts()
    singleton_baselines = baseline_counts[baseline_counts < 2].index.tolist()
    if not singleton_baselines:
        return cell, 0
    singleton_groups = baselines.loc[
        baselines["_baseline"].isin(singleton_baselines), group_col
    ].tolist()
    n_dropped = len(singleton_groups)
    if n_dropped > 0:
        warnings.warn(
            f"Singleton-baseline filter (footnote 15 of dynamic paper): dropped "
            f"{n_dropped} group(s) whose baseline treatment value is unique in "
            f"the panel. These groups have no baseline-matched control set. "
            f"Examples: {singleton_groups[:5]}"
            + (f" (and {n_dropped - 5} more)" if n_dropped > 5 else ""),
            UserWarning,
            stacklevel=3,
        )
        cell = cell[~cell[group_col].isin(singleton_groups)].reset_index(drop=True)
    return cell, n_dropped


def _compute_per_period_dids(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    periods: List[Any],
) -> Tuple[Dict[Any, Dict[str, Any]], List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-period DID_+,t and DID_-,t with explicit A11 zero-retention.

    Returns
    -------
    per_period_effects : dict
        Keyed by period; values are full per-period dicts including the
        ``did_*_t_a11_zeroed`` flags.
    a11_warnings : list of str
        One string per period that triggered an A11 violation.
    did_plus_t_arr : np.ndarray
        DID_+,t values aligned to ``periods[1:]``.
    did_minus_t_arr : np.ndarray
        DID_-,t values aligned to ``periods[1:]``.
    n_10_t_arr : np.ndarray
        Joiner cell counts aligned to ``periods[1:]``.
    n_01_t_arr : np.ndarray
        Leaver cell counts aligned to ``periods[1:]``.
    """
    n_periods = len(periods)
    per_period_effects: Dict[Any, Dict[str, Any]] = {}
    a11_warnings: List[str] = []
    did_plus_t_list: List[float] = []
    did_minus_t_list: List[float] = []
    n_10_t_list: List[int] = []
    n_01_t_list: List[int] = []

    for t_idx in range(1, n_periods):
        d_curr = D_mat[:, t_idx]
        d_prev = D_mat[:, t_idx - 1]
        y_curr = Y_mat[:, t_idx]
        y_prev = Y_mat[:, t_idx - 1]
        n_curr = N_mat[:, t_idx]

        joiner_mask = (d_prev == 0) & (d_curr == 1) & (n_curr > 0)
        stable0_mask = (d_prev == 0) & (d_curr == 0) & (n_curr > 0)
        leaver_mask = (d_prev == 1) & (d_curr == 0) & (n_curr > 0)
        stable1_mask = (d_prev == 1) & (d_curr == 1) & (n_curr > 0)

        n_10 = int(n_curr[joiner_mask].sum())
        n_00 = int(n_curr[stable0_mask].sum())
        n_01 = int(n_curr[leaver_mask].sum())
        n_11 = int(n_curr[stable1_mask].sum())

        # --- DID_+,t (joiners side) ---
        did_plus_t_a11_zeroed = False
        if n_10 == 0:
            did_plus_t = 0.0
        elif n_00 == 0:
            # A11 violation: joiners exist but no stable_0 controls
            did_plus_t = 0.0
            did_plus_t_a11_zeroed = True
            a11_warnings.append(f"period {periods[t_idx]}: joiners present, no stable_0")
        else:
            joiner_avg = float(
                (n_curr[joiner_mask] * (y_curr[joiner_mask] - y_prev[joiner_mask])).sum() / n_10
            )
            stable0_avg = float(
                (n_curr[stable0_mask] * (y_curr[stable0_mask] - y_prev[stable0_mask])).sum() / n_00
            )
            did_plus_t = joiner_avg - stable0_avg

        # --- DID_-,t (leavers side) ---
        did_minus_t_a11_zeroed = False
        if n_01 == 0:
            did_minus_t = 0.0
        elif n_11 == 0:
            did_minus_t = 0.0
            did_minus_t_a11_zeroed = True
            a11_warnings.append(f"period {periods[t_idx]}: leavers present, no stable_1")
        else:
            stable1_avg = float(
                (n_curr[stable1_mask] * (y_curr[stable1_mask] - y_prev[stable1_mask])).sum() / n_11
            )
            leaver_avg = float(
                (n_curr[leaver_mask] * (y_curr[leaver_mask] - y_prev[leaver_mask])).sum() / n_01
            )
            did_minus_t = stable1_avg - leaver_avg

        per_period_effects[periods[t_idx]] = {
            "did_plus_t": did_plus_t,
            "did_minus_t": did_minus_t,
            "n_10_t": n_10,
            "n_01_t": n_01,
            "n_00_t": n_00,
            "n_11_t": n_11,
            "did_plus_t_a11_zeroed": did_plus_t_a11_zeroed,
            "did_minus_t_a11_zeroed": did_minus_t_a11_zeroed,
        }
        did_plus_t_list.append(did_plus_t)
        did_minus_t_list.append(did_minus_t)
        n_10_t_list.append(n_10)
        n_01_t_list.append(n_01)

    return (
        per_period_effects,
        a11_warnings,
        np.array(did_plus_t_list, dtype=float),
        np.array(did_minus_t_list, dtype=float),
        np.array(n_10_t_list, dtype=int),
        np.array(n_01_t_list, dtype=int),
    )


def _compute_placebo(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    periods: List[Any],
) -> Optional[Tuple[float, bool]]:
    """
    Compute the single-lag placebo DID_M^pl from Theorem 4 of AER 2020.

    Same logic as DID_M but evaluated on the pre-event difference
    ``Y_{g, t-1} - Y_{g, t-2}`` for cells with three-period histories.
    Requires ``T >= 3``.
    """
    n_periods = len(periods)
    if n_periods < 3:
        return None

    placebo_plus_per_t: List[float] = []
    placebo_minus_per_t: List[float] = []
    n_10_per_t: List[int] = []
    n_01_per_t: List[int] = []

    for t_idx in range(2, n_periods):
        d_curr = D_mat[:, t_idx]
        d_prev = D_mat[:, t_idx - 1]
        d_pre_prev = D_mat[:, t_idx - 2]
        y_prev = Y_mat[:, t_idx - 1]
        y_pre_prev = Y_mat[:, t_idx - 2]
        n_curr = N_mat[:, t_idx]

        # Joiners that have a 3-period history with stable D=0 in t-2 and t-1
        joiner_mask = (
            (d_pre_prev == 0)
            & (d_prev == 0)
            & (d_curr == 1)
            & (n_curr > 0)
            & (N_mat[:, t_idx - 1] > 0)
            & (N_mat[:, t_idx - 2] > 0)
        )
        # Stable_0 controls with stable D=0 in t-2 and t-1
        stable0_mask = (
            (d_pre_prev == 0)
            & (d_prev == 0)
            & (d_curr == 0)
            & (n_curr > 0)
            & (N_mat[:, t_idx - 1] > 0)
            & (N_mat[:, t_idx - 2] > 0)
        )
        # Mirror for leavers/stable_1 (3-period stable treatment then leave)
        leaver_mask = (
            (d_pre_prev == 1)
            & (d_prev == 1)
            & (d_curr == 0)
            & (n_curr > 0)
            & (N_mat[:, t_idx - 1] > 0)
            & (N_mat[:, t_idx - 2] > 0)
        )
        stable1_mask = (
            (d_pre_prev == 1)
            & (d_prev == 1)
            & (d_curr == 1)
            & (n_curr > 0)
            & (N_mat[:, t_idx - 1] > 0)
            & (N_mat[:, t_idx - 2] > 0)
        )

        n_10 = int(n_curr[joiner_mask].sum())
        n_00 = int(n_curr[stable0_mask].sum())
        n_01 = int(n_curr[leaver_mask].sum())
        n_11 = int(n_curr[stable1_mask].sum())

        if n_10 > 0 and n_00 > 0:
            joiner_avg = float(
                (n_curr[joiner_mask] * (y_prev[joiner_mask] - y_pre_prev[joiner_mask])).sum() / n_10
            )
            stable0_avg = float(
                (n_curr[stable0_mask] * (y_prev[stable0_mask] - y_pre_prev[stable0_mask])).sum()
                / n_00
            )
            placebo_plus_t = joiner_avg - stable0_avg
        else:
            placebo_plus_t = 0.0

        if n_01 > 0 and n_11 > 0:
            stable1_avg = float(
                (n_curr[stable1_mask] * (y_prev[stable1_mask] - y_pre_prev[stable1_mask])).sum()
                / n_11
            )
            leaver_avg = float(
                (n_curr[leaver_mask] * (y_prev[leaver_mask] - y_pre_prev[leaver_mask])).sum() / n_01
            )
            placebo_minus_t = stable1_avg - leaver_avg
        else:
            placebo_minus_t = 0.0

        placebo_plus_per_t.append(placebo_plus_t)
        placebo_minus_per_t.append(placebo_minus_t)
        n_10_per_t.append(n_10)
        n_01_per_t.append(n_01)

    n_10_arr = np.array(n_10_per_t, dtype=int)
    n_01_arr = np.array(n_01_per_t, dtype=int)
    N_S_pl = int(n_10_arr.sum() + n_01_arr.sum())
    if N_S_pl == 0:
        return None
    placebo_effect = float(
        (n_10_arr @ np.array(placebo_plus_per_t) + n_01_arr @ np.array(placebo_minus_per_t))
        / N_S_pl
    )
    return placebo_effect, True


def _compute_cohort_recentered_inputs(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    periods: List[Any],
) -> Tuple[np.ndarray, int, int, int, np.ndarray, np.ndarray]:
    """
    Compute the cohort-centered influence-function vectors for variance.

    For each post-filter group, builds a per-group U^G_g value as the
    sum of switch contributions over time, then subtracts the cohort-
    conditional mean (cohort defined by the triple ``(D_{g,1}, F_g, S_g)``
    where F_g is the first switch period and S_g is the switch direction).

    Phase 1 simplification: rather than implementing the full
    ``lambda^G_{g,l=1}`` weight vector from Eq 22-23 of the dynamic
    paper (which is most useful at l > 1 for the dynamic estimator),
    Phase 1 uses the per-group switch contribution that exactly matches
    the AER 2020 Theorem 3 numerator at l = 1. This produces:

    - For a joiner group g switching from 0 -> 1 at period F_g:
      ``U^G_g = N_{g, F_g} * (Y_{g, F_g} - Y_{g, F_g - 1}) - control_term``
    - For a leaver group g switching from 1 -> 0 at period F_g:
      ``U^G_g = control_term - N_{g, F_g} * (Y_{g, F_g} - Y_{g, F_g - 1})``

    where ``control_term`` is the corresponding stable-control average
    contribution at the same period. Never-switching groups have
    ``S_g = 0`` and are filtered out for variance computation.

    The cohort-centered vector ``U_centered`` is then:
    ``U_centered[g] = U^G_g - U_bar_{cohort(g)}``
    where ``U_bar_k`` is the mean of ``U^G_g`` over groups in cohort
    ``k``. The plug-in variance from Section 3.7.3 of the dynamic paper
    becomes:
    ``sigma_hat^2 = (1/N_l) * sum_g U^G_g^2 - sum_k (|C_k|/N_l) * U_bar_k^2``
    which is algebraically equal to ``(1/N_l) * sum_g U_centered[g]^2``
    when ``N_l == G``. We expose ``U_centered`` directly so the bootstrap
    mixin can multiply it by random weights without re-computing the
    cohort means.

    Returns
    -------
    U_centered_overall : np.ndarray
        Cohort-centered IF vector for DID_M, length = number of switching groups.
    n_groups_for_overall : int
        ``len(U_centered_overall)`` (for sanity-checking from the caller).
    n_cohorts : int
        Distinct ``(D_{g,1}, F_g, S_g)`` triples in the post-filter group set.
    n_groups_dropped_never_switching : int
        Number of groups with ``S_g = 0`` (never switched).
    U_centered_joiners : np.ndarray
        Cohort-centered IF vector restricted to joiner groups.
    U_centered_leavers : np.ndarray
        Cohort-centered IF vector restricted to leaver groups.
    """
    n_groups, n_periods = D_mat.shape

    # Per-group baseline, first switch time, switch direction
    baselines = D_mat[:, 0]
    first_switch_idx = np.full(n_groups, -1, dtype=int)
    switch_direction = np.zeros(n_groups, dtype=int)  # +1 joiner, -1 leaver, 0 none

    for g in range(n_groups):
        for t in range(1, n_periods):
            if D_mat[g, t] != D_mat[g, t - 1]:
                first_switch_idx[g] = t
                switch_direction[g] = 1 if D_mat[g, t] > D_mat[g, t - 1] else -1
                break

    switching_mask = switch_direction != 0
    n_groups_dropped_never_switching = int((~switching_mask).sum())

    if n_groups_dropped_never_switching > 0:
        # Per the no-silent-failures policy: warn that groups are filtered
        # from the variance computation. They still contribute to the point
        # estimate as stable controls; the filter only excludes them from
        # the influence-function-based variance.
        warnings.warn(
            f"{n_groups_dropped_never_switching} group(s) never switch "
            "treatment and are excluded from the cohort-recentered variance "
            "computation (footnote 15 of dynamic paper). They still contribute "
            "to the DID_M point estimate as stable controls. The exclusion is "
            "expected: never-switching groups have a zero influence-function "
            "value by construction.",
            UserWarning,
            stacklevel=3,
        )

    if not switching_mask.any():
        # No switchers — variance is undefined
        return (
            np.array([], dtype=float),
            0,
            0,
            n_groups_dropped_never_switching,
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    # Build per-group U^G_g values for switching groups, plus a per-period
    # cache of stable-control averages so each switcher can subtract its
    # appropriate control term in O(1).
    # control_avg_diff_0[t] = avg outcome diff (Y_t - Y_{t-1}) over groups stable at 0
    # control_avg_diff_1[t] = avg outcome diff over groups stable at 1
    control_avg_diff_0 = np.zeros(n_periods)
    control_avg_diff_1 = np.zeros(n_periods)
    for t in range(1, n_periods):
        d_curr = D_mat[:, t]
        d_prev = D_mat[:, t - 1]
        n_curr = N_mat[:, t]
        stable0_mask = (d_prev == 0) & (d_curr == 0) & (n_curr > 0)
        stable1_mask = (d_prev == 1) & (d_curr == 1) & (n_curr > 0)
        n_00 = n_curr[stable0_mask].sum()
        n_11 = n_curr[stable1_mask].sum()
        if n_00 > 0:
            control_avg_diff_0[t] = float(
                (n_curr[stable0_mask] * (Y_mat[stable0_mask, t] - Y_mat[stable0_mask, t - 1])).sum()
                / n_00
            )
        if n_11 > 0:
            control_avg_diff_1[t] = float(
                (n_curr[stable1_mask] * (Y_mat[stable1_mask, t] - Y_mat[stable1_mask, t - 1])).sum()
                / n_11
            )

    # Per-switcher contribution at its first switch period
    switcher_idxs = np.where(switching_mask)[0]
    U_overall = np.zeros(switcher_idxs.size, dtype=float)
    for k, g in enumerate(switcher_idxs):
        t = first_switch_idx[g]
        n_gt = int(N_mat[g, t])
        diff = float(Y_mat[g, t] - Y_mat[g, t - 1])
        if switch_direction[g] == 1:
            # Joiner: U_g = n_gt * (diff - control_avg_diff_0[t])
            U_overall[k] = n_gt * (diff - control_avg_diff_0[t])
        else:
            # Leaver: U_g = n_gt * (control_avg_diff_1[t] - diff)
            U_overall[k] = n_gt * (control_avg_diff_1[t] - diff)

    # Cohort identification: triples (D_{g,1}, F_g, S_g)
    cohort_keys = list(
        zip(
            baselines[switcher_idxs].tolist(),
            first_switch_idx[switcher_idxs].tolist(),
            switch_direction[switcher_idxs].tolist(),
        )
    )
    unique_cohorts: Dict[Tuple[int, int, int], int] = {}
    cohort_id_per_switcher = np.zeros(switcher_idxs.size, dtype=int)
    for i, key in enumerate(cohort_keys):
        if key not in unique_cohorts:
            unique_cohorts[key] = len(unique_cohorts)
        cohort_id_per_switcher[i] = unique_cohorts[key]
    n_cohorts = len(unique_cohorts)

    # Cohort-conditional means and centering
    U_centered_overall = np.empty_like(U_overall)
    for k_id in range(n_cohorts):
        in_cohort = cohort_id_per_switcher == k_id
        if not in_cohort.any():
            continue
        cohort_mean = float(U_overall[in_cohort].mean())
        U_centered_overall[in_cohort] = U_overall[in_cohort] - cohort_mean

    # Joiners-only / leavers-only IF vectors: restrict to the appropriate
    # switch_direction subset and re-center within those subsets' cohorts.
    joiner_subset_mask = switch_direction[switcher_idxs] == 1
    leaver_subset_mask = switch_direction[switcher_idxs] == -1
    U_centered_joiners = U_centered_overall[joiner_subset_mask]
    U_centered_leavers = U_centered_overall[leaver_subset_mask]

    return (
        U_centered_overall,
        U_centered_overall.size,
        n_cohorts,
        n_groups_dropped_never_switching,
        U_centered_joiners,
        U_centered_leavers,
    )


def _plugin_se(U_centered: np.ndarray, divisor: int) -> float:
    """
    Compute the cohort-recentered plug-in standard error.

    Implements ``SE = sqrt(sum_g U_centered[g]^2 / N_l) / sqrt(N_l)``,
    which is the simplified form of Section 3.7.3's plug-in formula
    after the cohort recentering has been applied to ``U_centered``.

    The plain ``(1/N_l) * sum_g U_centered^2 / N_l`` form gives the
    variance; we take its square root for the SE.
    """
    n = U_centered.size
    if n == 0 or divisor <= 0:
        return float("nan")
    sum_sq = float((U_centered**2).sum())
    sigma_hat_sq = sum_sq / divisor
    if not np.isfinite(sigma_hat_sq) or sigma_hat_sq < 0:
        return float("nan")
    return float(np.sqrt(sigma_hat_sq) / np.sqrt(divisor))


def _build_group_time_design(
    cell: pd.DataFrame,
    group_col: str,
    time_col: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a dense (intercept + group dummies + time dummies) design matrix.

    Used by the TWFE decomposition diagnostic. The first group and first
    period are dropped as the reference categories. Returns the matrix
    and a list of column names.
    """
    groups = sorted(cell[group_col].unique().tolist())
    times = sorted(cell[time_col].unique().tolist())
    n = len(cell)
    n_groups = len(groups)
    n_times = len(times)

    # Columns: [intercept, group_1, ..., group_{G-1}, time_1, ..., time_{T-1}]
    n_cols = 1 + (n_groups - 1) + (n_times - 1)
    X = np.zeros((n, n_cols), dtype=float)
    X[:, 0] = 1.0  # intercept

    group_to_col = {g: 1 + i for i, g in enumerate(groups[1:])}
    time_to_col = {t: 1 + (n_groups - 1) + i for i, t in enumerate(times[1:])}

    group_arr = cell[group_col].to_numpy()
    time_arr = cell[time_col].to_numpy()
    for i in range(n):
        g = group_arr[i]
        t = time_arr[i]
        if g in group_to_col:
            X[i, group_to_col[g]] = 1.0
        if t in time_to_col:
            X[i, time_to_col[t]] = 1.0

    column_names = (
        ["intercept"] + [f"group[{g}]" for g in groups[1:]] + [f"time[{t}]" for t in times[1:]]
    )
    return X, column_names


def _compute_twfe_diagnostic(
    cell: pd.DataFrame,
    group_col: str,
    time_col: str,
    rank_deficient_action: str,
) -> TWFEWeightsResult:
    """
    Compute the per-cell TWFE decomposition diagnostic from Theorem 1 of AER 2020.

    Steps:

    1. Regress ``d_gt`` on group + time fixed effects via :func:`solve_ols`.
    2. Compute residuals ``eps_{g, t}`` from the regression.
    3. Compute per-cell weights:
       ``w_{g,t} = N_{g,t} * eps_{g,t} / sum_{g',t'} N_{g',t'} * d_{g',t'} * eps_{g',t'}``
    4. Count negative weights among treated cells.
    5. Compute the plain TWFE coefficient as a separate regression of
       ``y_gt`` on the same FE plus the treatment indicator.
    6. Compute ``sigma_fe = |beta_fe| / sqrt(sum_treated w^2 - mean(w_treated)^2 * n_treated)``
       which is the smallest standard deviation of cell-level treatment
       effects that could flip the sign of the plain TWFE estimator.
    """
    X, _ = _build_group_time_design(cell, group_col, time_col)
    d_arr = cell["d_gt"].to_numpy().astype(float)
    n_arr = cell["n_gt"].to_numpy().astype(float)
    y_arr = cell["y_gt"].to_numpy().astype(float)

    # Step 1-2: regress d on FE
    coef_d, residuals_d, _ = solve_ols(
        X,
        d_arr,
        return_vcov=False,
        rank_deficient_action=rank_deficient_action,
        weights=n_arr,
    )
    eps = residuals_d

    # Step 3: per-cell weights — normalize by sum over treated cells
    treated_mask = d_arr == 1
    denom = float((n_arr[treated_mask] * eps[treated_mask]).sum())
    if denom == 0:
        # Cannot normalize: the design has zero treated mass after FE absorption.
        # Warn so the user knows the diagnostic returned NaN values rather than
        # silently substituting them.
        warnings.warn(
            "TWFE decomposition diagnostic could not normalize per-cell "
            "weights: the sum of N_{g,t} * residual over treated cells is "
            "zero. This typically means the design matrix has perfect "
            "collinearity between treatment and the group/period fixed "
            "effects. Returning NaN for fraction_negative, sigma_fe, and "
            "beta_fe.",
            UserWarning,
            stacklevel=3,
        )
        weights_df = cell[[group_col, time_col]].copy()
        weights_df["weight"] = 0.0
        return TWFEWeightsResult(
            weights=weights_df,
            fraction_negative=float("nan"),
            sigma_fe=float("nan"),
            beta_fe=float("nan"),
        )
    w_gt = (n_arr * eps) / denom

    weights_df = cell[[group_col, time_col]].copy()
    weights_df["weight"] = w_gt

    fraction_negative = float((w_gt[treated_mask] < 0).sum() / treated_mask.sum())

    # Step 5: plain TWFE regression of y on (FE + d_gt)
    X_with_d = np.column_stack([X, d_arr.reshape(-1, 1)])
    coef_fe, _, _ = solve_ols(
        X_with_d,
        y_arr,
        return_vcov=False,
        rank_deficient_action=rank_deficient_action,
        weights=n_arr,
    )
    beta_fe = float(coef_fe[-1])

    # Step 6: sigma_fe per Corollary 1 of AER 2020
    w_treated = w_gt[treated_mask]
    sum_sq = float((w_treated**2).sum())
    sum_w = float(w_treated.sum())
    n_treated = int(treated_mask.sum())
    inner = sum_sq - (sum_w**2 / n_treated) if n_treated > 0 else 0.0
    if inner > 0 and np.isfinite(beta_fe):
        sigma_fe = float(abs(beta_fe) / np.sqrt(inner))
    else:
        sigma_fe = float("nan")

    return TWFEWeightsResult(
        weights=weights_df,
        fraction_negative=fraction_negative,
        sigma_fe=sigma_fe,
        beta_fe=beta_fe,
    )


# =============================================================================
# Convenience functions
# =============================================================================


def chaisemartin_dhaultfoeuille(
    data: pd.DataFrame,
    outcome: str,
    group: str,
    time: str,
    treatment: str,
    **kwargs: Any,
) -> ChaisemartinDHaultfoeuilleResults:
    """
    One-shot convenience wrapper around
    :class:`ChaisemartinDHaultfoeuille`.

    Equivalent to::

        ChaisemartinDHaultfoeuille(**init_kwargs).fit(
            data, outcome=..., group=..., time=..., treatment=...,
            **fit_kwargs,
        )

    All keyword arguments are split between ``__init__`` and ``fit`` based
    on which signature accepts them. Useful for one-line use in scripts.

    Parameters
    ----------
    data : pd.DataFrame
    outcome, group, time, treatment : str
    **kwargs : Any
        Forwarded to ``ChaisemartinDHaultfoeuille.__init__`` or
        ``.fit()`` based on parameter name.

    Returns
    -------
    ChaisemartinDHaultfoeuilleResults
    """
    init_keys = {
        "alpha",
        "cluster",
        "n_bootstrap",
        "bootstrap_weights",
        "seed",
        "placebo",
        "twfe_diagnostic",
        "drop_larger_lower",
        "rank_deficient_action",
    }
    init_kwargs = {k: v for k, v in kwargs.items() if k in init_keys}
    fit_kwargs = {k: v for k, v in kwargs.items() if k not in init_keys}
    est = ChaisemartinDHaultfoeuille(**init_kwargs)
    return est.fit(
        data,
        outcome=outcome,
        group=group,
        time=time,
        treatment=treatment,
        **fit_kwargs,
    )


def twowayfeweights(
    data: pd.DataFrame,
    outcome: str,
    group: str,
    time: str,
    treatment: str,
    rank_deficient_action: str = "warn",
) -> TWFEWeightsResult:
    """
    Standalone TWFE decomposition diagnostic.

    Computes the per-cell weights, fraction negative, and ``sigma_fe``
    from Theorem 1 of de Chaisemartin & D'Haultfoeuille (2020), without
    fitting the full dCDH estimator. Mirrors the standalone Stata
    ``twowayfeweights`` package.

    Parameters
    ----------
    data : pd.DataFrame
        Individual-level panel.
    outcome : str
    group : str
    time : str
    treatment : str
    rank_deficient_action : str, default="warn"
        Action when the FE design matrix is rank-deficient.

    Returns
    -------
    TWFEWeightsResult
        Object with attributes ``weights`` (DataFrame), ``fraction_negative``
        (float), ``sigma_fe`` (float), and ``beta_fe`` (float).
    """
    if treatment not in data.columns:
        raise ValueError(f"treatment column {treatment!r} not in data")
    df = data.copy()
    df[treatment] = pd.to_numeric(df[treatment])
    cell = df.groupby([group, time], as_index=False).agg(
        y_gt=(outcome, "mean"), d_gt=(treatment, "mean"), n_gt=(treatment, "count")
    )
    cell["d_gt"] = (cell["d_gt"] >= 0.5).astype(int)
    return _compute_twfe_diagnostic(
        cell=cell,
        group_col=group,
        time_col=time,
        rank_deficient_action=rank_deficient_action,
    )
