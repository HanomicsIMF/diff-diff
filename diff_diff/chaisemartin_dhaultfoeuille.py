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
# Shared validation + cell aggregation helper
# =============================================================================


def _validate_and_aggregate_to_cells(
    data: pd.DataFrame,
    outcome: str,
    group: str,
    time: str,
    treatment: str,
) -> pd.DataFrame:
    """
    Validate input data and aggregate to ``(g, t)`` cells per the dCDH contract.

    Used by both :meth:`ChaisemartinDHaultfoeuille.fit` and
    :func:`twowayfeweights` so the validation rules and aggregation
    behavior are identical across the two public entry points.

    The contract (matching ``REGISTRY.md`` ``## ChaisemartinDHaultfoeuille``):

    1. **Required columns** ``outcome``, ``group``, ``time``, ``treatment``
       must all be present in ``data`` (raises ``ValueError`` listing
       any missing).
    2. **Treatment** must coerce to numeric and contain no ``NaN``
       (raises ``ValueError`` — silent dropping would change cell counts
       without informing the user).
    3. **Outcome** must coerce to numeric and contain no ``NaN`` (same
       reasoning).
    4. **Treatment** must be numeric. Both binary ``{0, 1}`` and
       non-binary (ordinal or continuous) treatment are supported.
       Non-binary treatment requires ``L_max >= 1`` in ``fit()`` because
       the per-period DID path uses binary joiner/leaver categorization.
    5. **Cell aggregation** via ``groupby([group, time]).agg(...)``
       producing ``y_gt`` (cell mean of ``outcome``), ``d_gt`` (cell
       mean of ``treatment``), and ``n_gt`` (count of original
       observations in the cell).
    6. **Within-cell-varying treatment** (any cell where ``d_min !=
       d_max``) raises ``ValueError``. Treatment must be constant
       within each ``(group, time)`` cell; fuzzy DiD is deferred to a
       separate dCdH 2018 paper. Pre-aggregate your data to constant
       cell-level treatment before calling ``fit()`` or
       ``twowayfeweights()``.

    Returns the aggregated cell DataFrame with columns
    ``[group, time, y_gt, d_gt, n_gt]``, sorted by ``[group, time]``
    with a fresh index.

    Raises
    ------
    ValueError
        On missing columns, NaN treatment / outcome values, non-numeric
        treatment / outcome that cannot be coerced, or non-binary raw
        treatment values.
    """
    # 1. Required columns
    missing = [c for c in (outcome, group, time, treatment) if c not in data.columns]
    if missing:
        raise ValueError(
            f"ChaisemartinDHaultfoeuille / twowayfeweights: column(s) {missing!r} "
            f"not found in data. Required columns: outcome, group, time, treatment."
        )

    df = data.copy()

    # 1b. Group and time NaN checks (before groupby, which silently drops NaN keys)
    n_nan_group = int(df[group].isna().sum())
    if n_nan_group > 0:
        raise ValueError(
            f"Group column {group!r} contains {n_nan_group} NaN value(s). "
            "groupby silently drops NaN keys, which would change the "
            "estimation sample without warning. Drop or impute NaN group "
            "values before calling fit() or twowayfeweights()."
        )
    n_nan_time = int(df[time].isna().sum())
    if n_nan_time > 0:
        raise ValueError(
            f"Time column {time!r} contains {n_nan_time} NaN value(s). "
            "groupby silently drops NaN keys, which would change the "
            "estimation sample without warning. Drop or impute NaN time "
            "values before calling fit() or twowayfeweights()."
        )

    # 2. Treatment numeric coercion + NaN check
    try:
        df[treatment] = pd.to_numeric(df[treatment])
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Could not coerce treatment column {treatment!r} to numeric: {exc}"
        ) from exc
    n_nan_treat = int(df[treatment].isna().sum())
    if n_nan_treat > 0:
        raise ValueError(
            f"Treatment column {treatment!r} contains {n_nan_treat} NaN value(s). "
            "ChaisemartinDHaultfoeuille requires non-missing treatment indicators "
            "on every observation; impute or drop NaN treatment rows before fitting "
            "so the dropped count is explicit."
        )

    # 3. Outcome numeric coercion + NaN check
    try:
        df[outcome] = pd.to_numeric(df[outcome])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Could not coerce outcome column {outcome!r} to numeric: {exc}") from exc
    n_nan_outcome = int(df[outcome].isna().sum())
    if n_nan_outcome > 0:
        raise ValueError(
            f"Outcome column {outcome!r} contains {n_nan_outcome} NaN value(s). "
            "Drop or impute missing outcomes before calling fit() so the "
            "exclusion is explicit (silently averaging over present values "
            "would distort per-cell means)."
        )

    # 4. Treatment must be numeric (binary or non-binary both accepted)
    # No longer enforces {0, 1} - non-binary and continuous treatment supported.

    # 5. Cell aggregation (compute min/max for within-cell check)
    cell = df.groupby([group, time], as_index=False).agg(
        y_gt=(outcome, "mean"),
        d_gt=(treatment, "mean"),
        d_min=(treatment, "min"),
        d_max=(treatment, "max"),
        n_gt=(treatment, "count"),
    )

    # 6. Within-cell-varying treatment rejection.
    # All observations in a cell must have the same treatment value
    # (for both binary and non-binary treatment). Detect by checking
    # that cell min equals cell max.
    non_constant_mask = cell["d_min"] != cell["d_max"]
    if non_constant_mask.any():
        n_non_constant = int(non_constant_mask.sum())
        example_cells = cell.loc[
            non_constant_mask, [group, time, "d_gt", "d_min", "d_max"]
        ].head(5)
        raise ValueError(
            f"Within-cell-varying treatment detected in {n_non_constant} "
            f"(group, time) cell(s). dCDH requires treatment to be "
            f"constant within each (group, time) cell. Cells where "
            f"d_min != d_max indicate that some units have different "
            f"treatment values. Pre-aggregate your data to constant "
            f"cell-level treatment before calling fit() or "
            f"twowayfeweights(). Fuzzy DiD is deferred to a separate "
            f"dCDH paper (see ROADMAP.md out-of-scope). Affected cells "
            f"(first 5):\n{example_cells}"
        )
    # Drop the min/max columns; keep d_gt as float (no int cast - supports
    # ordinal and continuous treatment).
    cell = cell.drop(columns=["d_min", "d_max"])

    # Sort to ensure deterministic order in downstream operations
    cell = cell.sort_values([group, time]).reset_index(drop=True)
    return cell


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
    cluster : str, optional, default=None
        **Phase 1 contract:** ``cluster`` must be ``None`` (the default).
        dCDH always clusters at the group level via the cohort-recentered
        influence-function plug-in (analytical SEs) and the multiplier
        bootstrap (also grouped at the ``group`` column). Passing any
        non-``None`` value raises ``NotImplementedError`` with a Phase 1
        pointer. Custom clustering at a coarser or finer level than the
        group is reserved for a future phase. See REGISTRY.md
        ``ChaisemartinDHaultfoeuille`` section for the full contract.
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
        placebo ``DID_M^pl`` (AER 2020 placebo specification) on the same data.
        Set to ``False`` to skip the placebo computation for speed; the
        results object will still expose ``placebo_*`` fields, but with
        NaN values and ``placebo_available=False``.
    twfe_diagnostic : bool, default=True
        If ``True`` (default), compute the TWFE decomposition diagnostic
        from Theorem 1 of AER 2020: per-``(g, t)`` weights, fraction of
        treated cells with negative weights, and ``sigma_fe`` (the
        smallest cell-effect standard deviation that could flip the sign
        of the plain TWFE coefficient). The diagnostic answers "what
        would the plain TWFE estimator say on the data you passed in?",
        so it runs on the **FULL pre-filter cell sample** (the same
        input as the standalone :func:`twowayfeweights` function), NOT
        on the post-filter estimation sample used by ``DID_M``. When
        the ragged-panel filter or ``drop_larger_lower`` drops groups,
        the fitted ``results.twfe_*`` values describe a LARGER sample
        (pre-filter) than ``results.overall_att`` and a ``UserWarning``
        is emitted to make the divergence explicit. See REGISTRY.md
        ``ChaisemartinDHaultfoeuille`` ``Note (TWFE diagnostic sample
        contract)`` for the full rationale.
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
        if cluster is not None:
            raise NotImplementedError(
                f"cluster={cluster!r}: custom clustering is not supported in "
                f"Phase 1 of ChaisemartinDHaultfoeuille. dCDH always clusters "
                f"at the group level via the cohort-recentered influence-"
                f"function plug-in (analytical SEs) and the multiplier "
                f"bootstrap (also grouped at the group column). To use the "
                f"supported group-level clustering, pass cluster=None (the "
                f"default). Custom clustering is reserved for a future "
                f"phase. See REGISTRY.md ChaisemartinDHaultfoeuille section "
                f"for the full contract."
            )

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
        """
        Set estimator parameters (sklearn-compatible).

        Re-runs the same validation rules as ``__init__`` so invalid
        parameter combinations cannot be introduced after construction.
        """
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter: {key}")
            setattr(self, key, value)

        # Re-run __init__ validation rules so the post-set state is valid.
        if self.rank_deficient_action not in ("warn", "error", "silent"):
            raise ValueError(
                f"rank_deficient_action must be 'warn', 'error', or 'silent', "
                f"got '{self.rank_deficient_action}'"
            )
        if self.bootstrap_weights not in ("rademacher", "mammen", "webb"):
            raise ValueError(
                f"bootstrap_weights must be 'rademacher', 'mammen', or 'webb', "
                f"got '{self.bootstrap_weights}'"
            )
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.n_bootstrap < 0:
            raise ValueError(f"n_bootstrap must be non-negative, got {self.n_bootstrap}")
        if self.cluster is not None:
            raise NotImplementedError(
                f"cluster={self.cluster!r}: custom clustering is not supported "
                f"in Phase 1 of ChaisemartinDHaultfoeuille. dCDH always clusters "
                f"at the group level. To use the supported group-level "
                f"clustering, pass cluster=None (the default). Custom clustering "
                f"is reserved for a future phase. See REGISTRY.md "
                f"ChaisemartinDHaultfoeuille section for the full contract."
            )
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
        # ---------- Phase 3 extensions ----------
        heterogeneity: Optional[str] = None,
        design2: bool = False,
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
            Group identifier column name. Treatment must be constant
            within each ``(group, time)`` cell after aggregation;
            ``ValueError`` is raised if any cell has fractional
            treatment after grouping (within-cell-varying treatment
            indicates a fuzzy design not supported in Phase 1).
        time : str
            Time period column name. Must be sortable.
        treatment : str
            Per-observation treatment column. Must be numeric and constant
            within each ``(group, time)`` cell. Both binary ``{0, 1}`` and
            non-binary (ordinal or continuous) treatment are supported.
            Non-binary treatment requires ``L_max >= 1``.
        aggregate : str, optional
            **Reserved for Phase 3.** Must be ``None``; any other value
            raises ``NotImplementedError``.
        L_max : int, optional
            Maximum event-study horizon. When set, computes ``DID_l``
            for ``l = 1, ..., L_max`` using the per-group building block
            from Equation 3 of the dynamic companion paper. When
            ``None`` (default), only the ``l = 1`` contemporaneous-
            switch estimator ``DID_M`` is computed (Phase 1 behavior).
            Must be a positive integer not exceeding the number of
            post-baseline periods in the panel.
        controls : list of str, optional
            Column names for covariate adjustment via residualization-style
            ``DID^X`` (Web Appendix Section 1.2). Requires ``L_max >= 1``.
            One ``theta_hat`` per baseline treatment value, estimated by
            OLS on not-yet-treated observations. NOT doubly-robust.
        trends_linear : bool, optional
            If ``True``, estimate group-specific linear trends via
            ``DID^{fd}`` (Web Appendix Section 1.3, Lemma 6). Requires
            ``L_max >= 1`` and at least 3 time periods.
        trends_nonparam : str, optional
            Column name for state-set membership. Restricts the control
            pool to groups in the same set (Web Appendix Section 1.4).
            Requires ``L_max >= 1`` and time-invariant values per group.
        honest_did : bool, default=False
            **Reserved for Phase 3** (HonestDiD integration on placebos).
        heterogeneity : str, optional
            Column name for a time-invariant covariate to test for
            heterogeneous effects (Web Appendix Section 1.5, Lemma 7).
            Partial implementation: post-treatment regressions only
            (no placebo regressions or joint null test). Cannot be
            combined with ``controls``, ``trends_linear``, or
            ``trends_nonparam``. Requires ``L_max >= 1``.
        design2 : bool, default=False
            If ``True``, identify and report switch-in/switch-out
            (Design-2) groups. Convenience wrapper (descriptive summary,
            not full paper re-estimation). Requires
            ``drop_larger_lower=False`` to retain 2-switch groups.
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

        # Design-2 precondition: requires drop_larger_lower=False
        if design2 and self.drop_larger_lower:
            raise ValueError(
                "design2=True requires drop_larger_lower=False because "
                "Design-2 groups have exactly 2 treatment changes (join "
                "then leave), which are dropped by the default "
                "drop_larger_lower=True filter. Construct the estimator "
                "with ChaisemartinDHaultfoeuille(drop_larger_lower=False)."
            )

        # ------------------------------------------------------------------
        # Step 4-5: Validate input + aggregate to (g, t) cells via the
        # shared helper used by both fit() and twowayfeweights(). The
        # helper enforces NaN/binary/within-cell-rounding rules from
        # REGISTRY.md and returns a sorted cell DataFrame with columns
        # [group, time, y_gt, d_gt, n_gt].
        # ------------------------------------------------------------------
        cell = _validate_and_aggregate_to_cells(
            data=data,
            outcome=outcome,
            group=group,
            time=time,
            treatment=treatment,
        )

        # ------------------------------------------------------------------
        # Step 4b: Covariate aggregation (DID^X, Web Appendix Section 1.2)
        # ------------------------------------------------------------------
        if controls is not None:
            if not controls:
                raise ValueError(
                    "controls must be a non-empty list of column names, "
                    "got an empty list. Pass controls=None to disable "
                    "covariate adjustment."
                )
            if L_max is None:
                raise ValueError(
                    "Covariate adjustment (DID^X) requires L_max >= 1. The "
                    "per-period DID path does not support covariate "
                    "residualization. Set L_max to use the per-group "
                    "DID_{g,l} path with covariate adjustment."
                )
            missing_controls = [c for c in controls if c not in data.columns]
            if missing_controls:
                raise ValueError(
                    f"Control column(s) {missing_controls!r} not found in "
                    f"data. Available columns: {list(data.columns)}"
                )
            # Work on a copy to avoid mutating the caller's DataFrame
            data_controls = data[controls].copy()
            for c in controls:
                try:
                    data_controls[c] = pd.to_numeric(data_controls[c])
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"Could not coerce control column {c!r} to numeric: {exc}"
                    ) from exc
                n_nan = int(data_controls[c].isna().sum())
                if n_nan > 0:
                    raise ValueError(
                        f"Control column {c!r} contains {n_nan} NaN value(s). "
                        "Drop or impute missing covariates before fitting."
                    )
                n_inf = int(np.isinf(data_controls[c].to_numpy()).sum())
                if n_inf > 0:
                    raise ValueError(
                        f"Control column {c!r} contains {n_inf} Inf value(s). "
                        "Remove or replace non-finite covariates before fitting."
                    )
            # Aggregate covariates to cell means (same groupby as treatment/outcome).
            # Use the coerced copy joined with group/time from original data.
            x_agg_input = data[[group, time]].copy()
            x_agg_input[controls] = data_controls[controls].values
            x_cell_agg = x_agg_input.groupby([group, time], as_index=False)[controls].mean()
            cell = cell.merge(x_cell_agg, on=[group, time], how="left")

        # ------------------------------------------------------------------
        # Step 5a: Compute the TWFE diagnostic on the FULL pre-filter cell
        #          dataset, so the diagnostic reflects the data the user
        #          actually passed in. This MUST run BEFORE Step 5b (the
        #          ragged-panel filter) so that the fitted diagnostic and
        #          the standalone twowayfeweights() function produce
        #          identical results on ragged panels — both operate on
        #          the same _validate_and_aggregate_to_cells() output.
        # ------------------------------------------------------------------
        twfe_diagnostic_payload = None
        # TWFE diagnostic assumes binary treatment (d_arr == 1 for
        # treated mask). Skip for non-binary data with a warning.
        is_binary_pre = set(cell["d_gt"].unique()).issubset({0.0, 1.0, 0, 1})
        if self.twfe_diagnostic and not is_binary_pre:
            warnings.warn(
                "TWFE diagnostic (twfe_diagnostic=True) is not supported for "
                "non-binary treatment. The diagnostic assumes binary {0, 1} "
                "treatment. Skipping TWFE diagnostic for this fit.",
                UserWarning,
                stacklevel=2,
            )
        elif self.twfe_diagnostic:
            try:
                twfe_diagnostic_payload = _compute_twfe_diagnostic(
                    cell=cell,
                    group_col=group,
                    time_col=time,
                    rank_deficient_action=self.rank_deficient_action,
                )
            except Exception as exc:  # noqa: BLE001
                # Honor rank_deficient_action="error": if the user
                # explicitly requested strict failure on rank-deficient
                # designs, re-raise instead of downgrading to a warning.
                # Only genuinely non-fatal failures (e.g., numerical
                # issues unrelated to rank deficiency) should be
                # swallowed as warnings.
                if self.rank_deficient_action == "error" and isinstance(exc, ValueError):
                    raise
                warnings.warn(
                    f"TWFE decomposition diagnostic failed: {exc}. "
                    "Skipping diagnostic; main estimation continues.",
                    UserWarning,
                    stacklevel=2,
                )
                twfe_diagnostic_payload = None

        # ------------------------------------------------------------------
        # Step 5b: Ragged panel validation
        #
        # The cohort/variance path treats D_{g,1} as the canonical
        # baseline and walks adjacent observed periods to detect first
        # switches. Ragged panels with missing baseline rows or interior
        # gaps would either crash the cohort enumeration (NaN -> int
        # cast) or silently misclassify cohorts. Two-tier handling:
        #
        # (a) Reject groups missing the FIRST GLOBAL period (the
        #     baseline) with a clear ValueError listing offenders.
        # (b) Drop groups with INTERIOR GAPS (missing intermediate
        #     periods between their first and last observed period)
        #     with an explicit UserWarning.
        # ------------------------------------------------------------------
        all_periods_pre_drop = sorted(cell[time].unique().tolist())
        if len(all_periods_pre_drop) < 2:
            raise ValueError(
                f"ChaisemartinDHaultfoeuille requires at least 2 distinct time "
                f"periods in the panel, got {len(all_periods_pre_drop)}."
            )
        first_global_period = all_periods_pre_drop[0]

        # (a) Reject groups missing the first global period
        groups_with_baseline = set(cell.loc[cell[time] == first_global_period, group].tolist())
        all_groups_pre_validation = set(cell[group].unique().tolist())
        groups_missing_baseline = sorted(all_groups_pre_validation - groups_with_baseline)
        if groups_missing_baseline:
            raise ValueError(
                f"ChaisemartinDHaultfoeuille requires every group to have an "
                f"observation at the first global period "
                f"(period={first_global_period!r}). "
                f"{len(groups_missing_baseline)} group(s) are missing this baseline. "
                f"Examples: {groups_missing_baseline[:5]}"
                + (
                    f" (and {len(groups_missing_baseline) - 5} more)"
                    if len(groups_missing_baseline) > 5
                    else ""
                )
                + ". Drop these groups or back-fill the baseline before fitting "
                "so the exclusion is explicit."
            )

        # (b) Drop groups with interior gaps
        period_index = {p: i for i, p in enumerate(all_periods_pre_drop)}
        groups_with_interior_gaps: List[Any] = []
        for g_id, sub in cell.groupby(group):
            g_periods = sub[time].tolist()
            g_min_idx = period_index[min(g_periods)]
            g_max_idx = period_index[max(g_periods)]
            expected_count = g_max_idx - g_min_idx + 1
            if len(g_periods) != expected_count:
                groups_with_interior_gaps.append(g_id)
        n_groups_dropped_interior_gap = len(groups_with_interior_gaps)
        if groups_with_interior_gaps:
            warnings.warn(
                f"Dropping {len(groups_with_interior_gaps)} group(s) with interior "
                f"period gaps (missing observations between their first and last "
                f"observed period). Examples: {groups_with_interior_gaps[:5]}"
                + (
                    f" (and {len(groups_with_interior_gaps) - 5} more)"
                    if len(groups_with_interior_gaps) > 5
                    else ""
                )
                + ". dCDH requires consecutive observed periods for the "
                "cohort/variance path; back-fill or interpolate the missing "
                "periods if you want these groups in the estimation.",
                UserWarning,
                stacklevel=2,
            )
            cell = cell[~cell[group].isin(groups_with_interior_gaps)].reset_index(drop=True)
            if cell.empty:
                raise ValueError(
                    "After dropping groups with interior period gaps, no groups "
                    "remain. Provide a balanced panel or back-fill missing periods."
                )

        all_periods_pre_drop = sorted(cell[time].unique().tolist())
        if len(all_periods_pre_drop) < 2:
            raise ValueError(
                f"ChaisemartinDHaultfoeuille requires at least 2 periods, "
                f"got {len(all_periods_pre_drop)}"
            )

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
        # Step 6b: TWFE diagnostic sample-contract notice
        #
        # The fitted twfe_* values (if the diagnostic succeeded in
        # Step 5a) were computed on the FULL pre-filter cell sample,
        # matching the standalone twowayfeweights() output. Steps 5b
        # and 6 may have dropped groups since then. When they did, the
        # fitted diagnostic and the dCDH point estimate describe
        # DIFFERENT samples, so we surface that divergence as a
        # UserWarning per the REGISTRY contract Note. Users see the
        # warning at fit time and can decide whether to pre-process
        # their data before re-fitting (or accept the documented
        # divergence).
        #
        # The warning fires whenever the user requested the diagnostic
        # AND filters dropped groups, even if _compute_twfe_diagnostic
        # itself failed (rank-deficient fallback) and
        # twfe_diagnostic_payload is None. The warning text uses "(if
        # the diagnostic succeeded)" to remain accurate in both cases.
        # ------------------------------------------------------------------
        if self.twfe_diagnostic and (n_groups_dropped_interior_gap + n_groups_dropped_crossers) > 0:
            warnings.warn(
                f"TWFE diagnostic sample-contract notice: the dCDH point "
                f"estimate, results.groups, and inference fields use a "
                f"POST-FILTER sample after Step 5b dropped "
                f"{n_groups_dropped_interior_gap} interior-gap group(s) "
                f"and Step 6 dropped {n_groups_dropped_crossers} multi-"
                f"switch group(s). The fitted results.twfe_* values (if "
                f"the diagnostic succeeded) were computed on the FULL "
                f"pre-filter cell sample, so they describe a LARGER "
                f"sample (pre-filter) than overall_att. The standalone "
                f"twowayfeweights() function also uses the pre-filter "
                f"sample. This is the documented Phase 1 contract — see "
                f"REGISTRY.md ChaisemartinDHaultfoeuille `Note (TWFE "
                f"diagnostic sample contract)` for the rationale. To "
                f"reproduce the dCDH estimation sample for an external "
                f"TWFE comparison, pre-process your data to drop the "
                f"{n_groups_dropped_interior_gap + n_groups_dropped_crossers} "
                f"flagged groups before re-fitting.",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # Step 7: Singleton-baseline identification (footnote 15 of dynamic paper)
        # ------------------------------------------------------------------
        # The singleton-baseline filter identifies groups whose baseline
        # treatment value D_{g,1} is unique in the panel. Per footnote 15
        # of the dynamic paper, these have no baseline-matched cohort peer
        # and contribute zero variance under the cohort framework.
        #
        # IMPORTANT: under Python's documented period-based stable-control
        # interpretation, a singleton-baseline group can STILL be a valid
        # stable_0 / stable_1 control for the point estimate, even though
        # it has no cohort peer. The filter is therefore applied at the
        # variance stage only — the cell DataFrame retains these groups
        # so they can serve as stable controls.
        # Use the validated first global period as the canonical baseline.
        # Step 5b guarantees every group has an observation at this period,
        # so we can read it directly without a groupby.first() that could
        # otherwise return a later observed period for late-entry groups.
        baselines_per_group = cell.loc[cell[time] == first_global_period, [group, "d_gt"]].rename(
            columns={"d_gt": "_baseline"}
        )
        baseline_counts = baselines_per_group["_baseline"].value_counts()
        singleton_baseline_values = baseline_counts[baseline_counts < 2].index.tolist()
        singleton_baseline_groups: List[Any] = (
            baselines_per_group.loc[
                baselines_per_group["_baseline"].isin(singleton_baseline_values), group
            ].tolist()
            if singleton_baseline_values
            else []
        )
        n_groups_dropped_singleton_baseline = len(singleton_baseline_groups)
        if n_groups_dropped_singleton_baseline > 0:
            warnings.warn(
                f"Singleton-baseline filter (footnote 15 of dynamic paper): "
                f"{n_groups_dropped_singleton_baseline} group(s) excluded from "
                f"the cohort-recentered VARIANCE computation only — they remain "
                f"in the point-estimate sample as period-based stable controls. "
                f"Examples: {singleton_baseline_groups[:5]}"
                + (
                    f" (and {n_groups_dropped_singleton_baseline - 5} more)"
                    if n_groups_dropped_singleton_baseline > 5
                    else ""
                ),
                UserWarning,
                stacklevel=2,
            )

        if cell.empty or cell[group].nunique() == 0:
            raise ValueError(
                "After dropping multi-switch cells (drop_larger_lower=True), no "
                "groups remain. The dataset cannot support dCDH estimation. "
                "Check the input panel for diversity in treatment patterns."
            )

        # Determine the post-filter group set, period set, and per-group state
        all_groups = sorted(cell[group].unique().tolist())
        all_periods = sorted(cell[time].unique().tolist())
        n_obs_post = int(cell["n_gt"].sum())

        # ------------------------------------------------------------------
        # L_max validation (Phase 2): must be a positive integer not
        # exceeding the number of post-baseline periods. Validated here
        # (after period detection) rather than in _check_forward_compat_gates
        # (which runs before data is processed).
        # ------------------------------------------------------------------
        if L_max is not None:
            if not isinstance(L_max, int) or L_max < 1:
                raise ValueError(f"L_max must be a positive integer or None, got {L_max!r}.")
            n_post_baseline = len(all_periods) - 1
            if L_max > n_post_baseline:
                raise ValueError(
                    f"L_max={L_max} exceeds available post-baseline periods "
                    f"({n_post_baseline}). Maximum L_max for this panel "
                    f"is {n_post_baseline}."
                )

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
        # Step 7b: Covariate residualization (DID^X)
        #
        # When controls are specified, residualize Y_mat by partialling
        # out covariate effects per baseline treatment group. This
        # transforms Y_mat in-place so ALL downstream DID computations
        # (per-period and per-group multi-horizon) automatically produce
        # covariate-adjusted estimates. See Web Appendix Section 1.2.
        # ------------------------------------------------------------------
        covariate_diagnostics: Optional[Dict[str, Any]] = None
        _switch_metadata_computed = False

        if controls is not None:
            # Pivot covariates to (n_groups, n_periods, n_covariates)
            X_pivots = []
            for c in controls:
                x_piv = cell.pivot(
                    index=group, columns=time, values=c
                ).reindex(index=all_groups, columns=all_periods)
                X_pivots.append(x_piv.to_numpy())
            X_cell = np.stack(X_pivots, axis=2)

            # Need switch metadata for residualization (baselines, F_g)
            baselines, first_switch_idx_arr, switch_direction_arr, T_g_arr = (
                _compute_group_switch_metadata(D_mat, N_mat)
            )
            _switch_metadata_computed = True

            Y_mat_residualized, covariate_diagnostics, _failed_baselines = (
                _compute_covariate_residualization(
                    Y_mat=Y_mat,
                    X_cell=X_cell,
                    N_mat=N_mat,
                    baselines=baselines,
                    first_switch_idx=first_switch_idx_arr,
                    rank_deficient_action=self.rank_deficient_action,
                )
            )
            # Zero out N_mat for failed-stratum groups so the downstream
            # eligibility checks (N_mat[g, idx] > 0) naturally exclude
            # them from all DID/IF/placebo computation.
            if _failed_baselines:
                for g_idx in range(len(baselines)):
                    if float(baselines[g_idx]) in _failed_baselines:
                        N_mat[g_idx, :] = 0
            # Keep raw Y_mat for the per-period DID path (which does not
            # support covariate residualization - it uses binary joiner/leaver
            # categorization). The residualized matrix is used only by the
            # per-group multi-horizon path (L_max >= 1).
            Y_mat_raw = Y_mat
            Y_mat = Y_mat_residualized

        # ------------------------------------------------------------------
        # Step 7c: First-differencing for linear trends (DID^{fd})
        #
        # When trends_linear=True, replace Y_mat with Z_mat (first-
        # differenced outcomes) so that DID_{g,l}(Z) = DID^{fd}_{g,l}.
        # N_mat is also adjusted: N_mat_fd marks which Z values are valid.
        # IMPORTANT: _compute_group_switch_metadata uses the ORIGINAL
        # N_mat (treatment path metadata), not N_mat_fd.
        # ------------------------------------------------------------------
        _is_trends_linear = trends_linear is True
        linear_trends_effects: Optional[Dict[int, Dict[str, Any]]] = None
        # N_mat_orig preserves observation counts for switch-metadata and
        # cohort-identification code that must NOT see the first-differenced
        # N_mat_fd. When trends_linear=False, N_mat_orig == N_mat.
        N_mat_orig = N_mat

        if _is_trends_linear:
            if L_max is None:
                raise ValueError(
                    "Group-specific linear trends (DID^{fd}) requires "
                    "L_max >= 1. Set L_max to use the per-group "
                    "DID_{g,l} path with trend adjustment."
                )
            if len(all_periods) < 3:
                raise ValueError(
                    "Group-specific linear trends (DID^{fd}) requires "
                    "at least 3 time periods (F_g >= 3 in the paper). "
                    f"Got {len(all_periods)} period(s)."
                )
            # Compute switch metadata on original N_mat if not done yet
            if not _switch_metadata_computed:
                baselines, first_switch_idx_arr, switch_direction_arr, T_g_arr = (
                    _compute_group_switch_metadata(D_mat, N_mat)
                )
                _switch_metadata_computed = True
            # Count and warn about excluded groups (F_g < 3 -> f_g < 2)
            n_excluded_fd = int(
                ((first_switch_idx_arr >= 0) & (first_switch_idx_arr < 2)).sum()
            )
            if n_excluded_fd > 0:
                warnings.warn(
                    f"DID^{{fd}} (trends_linear=True): {n_excluded_fd} "
                    f"switching group(s) have F_g < 3 (fewer than 2 "
                    f"pre-switch periods) and are excluded from the "
                    f"trend-adjusted estimation.",
                    UserWarning,
                    stacklevel=2,
                )
            N_mat_orig = N_mat.copy()
            Y_mat, N_mat = _compute_first_differenced_matrix(Y_mat, N_mat)

        # ------------------------------------------------------------------
        # Step 7d: State-set trends validation (trends_nonparam)
        #
        # When trends_nonparam is set (a column name), restrict the
        # control pool for each switcher to groups in the same set.
        # ------------------------------------------------------------------
        set_ids_arr: Optional[np.ndarray] = None

        if trends_nonparam is not None:
            if L_max is None:
                raise ValueError(
                    "State-set-specific trends (trends_nonparam) requires "
                    "L_max >= 1. Set L_max to use the per-group "
                    "DID_{g,l} path with state-set trends."
                )
            set_col = str(trends_nonparam)
            if set_col not in data.columns:
                raise ValueError(
                    f"trends_nonparam column {set_col!r} not found in "
                    f"data. Available columns: {list(data.columns)}"
                )
            # Reject NaN/missing set assignments
            n_na_set = int(data[set_col].isna().sum())
            if n_na_set > 0:
                raise ValueError(
                    f"trends_nonparam column {set_col!r} contains "
                    f"{n_na_set} NaN/missing value(s). All groups must "
                    f"have a valid set assignment."
                )
            # Aggregate set membership per group (must be time-invariant)
            set_per_group = data.groupby(group)[set_col].nunique()
            time_varying = set_per_group[set_per_group > 1]
            if len(time_varying) > 0:
                raise ValueError(
                    f"trends_nonparam column {set_col!r} must be "
                    f"time-invariant within each group. "
                    f"{len(time_varying)} group(s) have varying values. "
                    f"Examples: {time_varying.index.tolist()[:5]}"
                )
            # Set partition must be coarser than group (multiple groups
            # per set). A group-level partition creates singleton sets
            # with no within-set controls available.
            set_map_check = data.groupby(group)[set_col].first()
            n_sets = set_map_check.nunique()
            n_groups_total = len(set_map_check)
            if n_sets >= n_groups_total:
                raise ValueError(
                    f"trends_nonparam column {set_col!r} defines "
                    f"{n_sets} distinct sets for {n_groups_total} "
                    f"groups. The set partition must be coarser than "
                    f"group (multiple groups per set) to provide "
                    f"within-set controls."
                )
            # Extract set membership per group aligned with all_groups
            set_map = data.groupby(group)[set_col].first()
            set_ids_arr = np.array(
                [set_map.loc[g] for g in all_groups], dtype=object
            )

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
            n_00_t_arr,
            n_11_t_arr,
            a11_plus_zeroed_arr,
            a11_minus_zeroed_arr,
        ) = _compute_per_period_dids(
            D_mat=D_mat,
            # Use raw (unadjusted) outcomes for per-period DID. Covariate
            # residualization applies only to the per-group multi-horizon
            # path (L_max >= 1). The per-period path uses binary
            # joiner/leaver categorization and is not part of the DID^X
            # contract (Web Appendix Section 1.2).
            # Use raw outcomes for per-period DID when controls or
            # trends_linear is active (both transform Y_mat).
            Y_mat=Y_mat_raw if controls is not None else (y_pivot.to_numpy() if _is_trends_linear else Y_mat),
            N_mat=N_mat_orig,
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
        # For non-binary treatment, the per-period DID path may find N_S=0
        # because it uses binary joiner/leaver categorization. When L_max
        # is set, the multi-horizon path (which handles non-binary correctly
        # via per-group DID_{g,l}) will compute the effects. Only raise if
        # L_max is also None (i.e., no fallback path).
        is_binary = set(np.unique(D_mat[~np.isnan(D_mat)])).issubset({0.0, 1.0})
        if not is_binary and L_max is None:
            raise ValueError(
                "Non-binary treatment requires L_max >= 1. The per-period DID "
                "path uses binary joiner/leaver categorization; set L_max to "
                "use the per-group DID_{g,l} building block which handles "
                "non-binary treatment."
            )
        if N_S == 0 and (L_max is None or is_binary):
            raise ValueError(
                "No switching cells found in the data after filtering: every "
                "group has constant treatment for the entire panel. dCDH "
                "requires at least one (g, t) cell where the group's treatment "
                "differs from the previous period."
            )
        if N_S > 0:
            overall_att = float(
                (n_10_t_arr @ did_plus_t_arr + n_01_t_arr @ did_minus_t_arr) / N_S
            )
        else:
            # Non-binary treatment with L_max: per-period DID is not
            # applicable. The multi-horizon path will provide overall_att
            # via the cost-benefit delta.
            overall_att = float("nan")

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

        # Joiner / leaver sample-size metadata.
        # n_*_cells: total switching cells across all periods (sum of per-period
        #            cell counts; each (g, t) joiner/leaver cell counted once).
        # n_*_obs:   actual observation count (sum of n_gt over the same cells),
        #            which differs from cells when individual-level inputs have
        #            multiple original observations per (g, t).
        n_joiner_cells = int(n_10_t_arr.sum())
        n_leaver_cells = int(n_01_t_arr.sum())
        n_joiner_obs = 0
        n_leaver_obs = 0
        for t_idx in range(1, len(all_periods)):
            d_curr = D_mat[:, t_idx]
            d_prev = D_mat[:, t_idx - 1]
            n_curr = N_mat[:, t_idx]
            n_prev = N_mat[:, t_idx - 1]
            present = (n_curr > 0) & (n_prev > 0)
            joiner_mask_t = (d_prev == 0) & (d_curr == 1) & present
            leaver_mask_t = (d_prev == 1) & (d_curr == 0) & present
            n_joiner_obs += int(n_curr[joiner_mask_t].sum())
            n_leaver_obs += int(n_curr[leaver_mask_t].sum())

        # ------------------------------------------------------------------
        # Step 12: Placebo (DID_M^pl)
        # ------------------------------------------------------------------
        placebo_available = False
        placebo_effect = float("nan")
        if self.placebo:
            if len(all_periods) < 3:
                warnings.warn(
                    f"Placebo DID_M^pl requires at least 3 time "
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
                    placebo_effect, placebo_available, placebo_a11_warnings = placebo_payload
                    # Surface placebo A11 violations via a consolidated warning
                    # mirroring the main DID path's contract. The affected
                    # per-period placebo contributions are zeroed in the
                    # numerator with their switcher counts retained in the
                    # placebo N_S^pl denominator (placebo zero-retention).
                    if placebo_a11_warnings:
                        warnings.warn(
                            f"Placebo (DID_M^pl) Assumption 11 violations in "
                            f"{len(placebo_a11_warnings)} period(s); the affected "
                            f"placebo contributions are zeroed but their switcher "
                            f"counts are retained in the placebo N_S denominator "
                            f"(matching placebo paper convention). Affected: "
                            + ", ".join(placebo_a11_warnings[:3])
                            + (
                                f" (and {len(placebo_a11_warnings) - 3} more)"
                                if len(placebo_a11_warnings) > 3
                                else ""
                            ),
                            UserWarning,
                            stacklevel=2,
                        )

        # ------------------------------------------------------------------
        # Step 12b: Per-group switch metadata (shared by Phase 1 IF and
        #           Phase 2 multi-horizon). May already be computed by
        #           Step 7b (covariate residualization).
        # ------------------------------------------------------------------
        if not _switch_metadata_computed:
            baselines, first_switch_idx_arr, switch_direction_arr, T_g_arr = (
                _compute_group_switch_metadata(D_mat, N_mat_orig)
            )

        # ------------------------------------------------------------------
        # Step 12c: Multi-horizon per-group computation (L_max >= 1)
        # ------------------------------------------------------------------
        multi_horizon_dids: Optional[Dict[int, Dict[str, Any]]] = None
        multi_horizon_if: Optional[Dict[int, np.ndarray]] = None
        multi_horizon_se: Optional[Dict[int, float]] = None
        multi_horizon_inference: Optional[Dict[int, Dict[str, Any]]] = None

        if L_max is not None and L_max >= 1:
            multi_horizon_dids = _compute_multi_horizon_dids(
                D_mat=D_mat,
                Y_mat=Y_mat,
                N_mat=N_mat,
                baselines=baselines,
                first_switch_idx=first_switch_idx_arr,
                switch_direction=switch_direction_arr,
                T_g=T_g_arr,
                L_max=L_max,
                set_ids=set_ids_arr,
            )
            # Surface A11 warnings from multi-horizon computation
            mh_a11 = multi_horizon_dids.pop("_a11_warnings", None)
            if mh_a11:
                warnings.warn(
                    f"Multi-horizon control-availability violations in "
                    f"{len(mh_a11)} (group, horizon) pair(s): affected "
                    f"groups are excluded from N_l (no observed baseline-"
                    f"matched controls at the outcome period). Examples: "
                    + ", ".join(mh_a11[:3])
                    + (f" (and {len(mh_a11) - 3} more)" if len(mh_a11) > 3 else ""),
                    UserWarning,
                    stacklevel=2,
                )

            # Guard: if no eligible switchers at horizon 1 (e.g., all
            # groups have constant treatment), raise ValueError.
            if 1 in multi_horizon_dids and multi_horizon_dids[1]["N_l"] == 0:
                raise ValueError(
                    "No switching groups found at horizon 1 after filtering. "
                    "dCDH requires at least one group whose treatment changes "
                    "from the baseline period."
                )

            multi_horizon_if = _compute_per_group_if_multi_horizon(
                D_mat=D_mat,
                Y_mat=Y_mat,
                N_mat=N_mat,
                baselines=baselines,
                first_switch_idx=first_switch_idx_arr,
                switch_direction=switch_direction_arr,
                T_g=T_g_arr,
                L_max=L_max,
                set_ids=set_ids_arr,
            )

            # Per-horizon analytical SE via cohort recentering.
            # Reuse the singleton-baseline exclusion from Step 7 and
            # build cohort IDs per horizon.
            singleton_baseline_set = set(singleton_baseline_groups)
            eligible_mask_var = np.array(
                [g not in singleton_baseline_set for g in all_groups], dtype=bool
            )

            multi_horizon_se = {}
            multi_horizon_inference = {}
            # Compute inference for ALL horizons 1..L_max (including l=1)
            # so the event_study_effects dict uses a consistent estimand
            # (per-group DID_{g,l}) across all horizons.
            for l_h in range(1, L_max + 1):
                U_l = multi_horizon_if[l_h]
                # Cohort IDs for this horizon: (D_{g,1}, F_g, S_g) triples
                # are the same as Phase 1 (cohort identity depends on first
                # switch, not on the horizon). Filter to eligible.
                cohort_keys_l = [
                    (
                        float(baselines[g]),
                        int(first_switch_idx_arr[g]),
                        int(switch_direction_arr[g]),
                    )
                    for g in range(len(all_groups))
                ]
                unique_c: Dict[Tuple[float, int, int], int] = {}
                cid_l = np.zeros(len(all_groups), dtype=int)
                for g in range(len(all_groups)):
                    if not eligible_mask_var[g]:
                        cid_l[g] = -1
                        continue
                    key = cohort_keys_l[g]
                    if key not in unique_c:
                        unique_c[key] = len(unique_c)
                    cid_l[g] = unique_c[key]

                # Use the full variance-eligible group set (singleton-
                # baseline exclusion only). Do NOT intersect with
                # did_eligible — never-switchers and later-switching
                # controls can have non-zero IF mass via their control
                # roles, and dropping them understates the SE.
                U_l_elig = U_l[eligible_mask_var]
                cid_elig = cid_l[eligible_mask_var]
                U_centered_l = _cohort_recenter(U_l_elig, cid_elig)
                N_l_h = multi_horizon_dids[l_h]["N_l"]
                se_l = _plugin_se(U_centered=U_centered_l, divisor=N_l_h)
                multi_horizon_se[l_h] = se_l

                did_l_val = multi_horizon_dids[l_h]["did_l"]
                t_l, p_l, ci_l = safe_inference(did_l_val, se_l, alpha=self.alpha, df=None)
                multi_horizon_inference[l_h] = {
                    "effect": did_l_val,
                    "se": se_l,
                    "t_stat": t_l,
                    "p_value": p_l,
                    "conf_int": ci_l,
                    "n_obs": N_l_h,
                }

            # Emit <50% switcher warning for far horizons
            if multi_horizon_dids.get(1, {}).get("N_l", 0) > 0:
                N_1_ref = multi_horizon_dids[1]["N_l"]
                thin_horizons = [
                    l_h
                    for l_h in range(2, L_max + 1)
                    if multi_horizon_dids[l_h]["N_l"] < 0.5 * N_1_ref
                    and multi_horizon_dids[l_h]["N_l"] > 0
                ]
                if thin_horizons:
                    warnings.warn(
                        f"Fewer than 50% of l=1 switchers contribute at "
                        f"horizon(s) {thin_horizons}. Far-horizon estimates "
                        f"may be noisy. The paper recommends not reporting "
                        f"horizons where fewer than ~50% of switchers "
                        f"contribute (Favara-Imbs application, footnote 14).",
                        UserWarning,
                        stacklevel=2,
                    )

        # Phase 2: placebos, normalized effects, cost-benefit delta
        multi_horizon_placebos: Optional[Dict[int, Dict[str, Any]]] = None
        placebo_horizon_if: Optional[Dict[int, np.ndarray]] = None
        placebo_horizon_se: Optional[Dict[int, float]] = None
        placebo_horizon_inference: Optional[Dict[int, Dict[str, Any]]] = None
        normalized_effects_dict: Optional[Dict[int, Dict[str, Any]]] = None
        cost_benefit_result: Optional[Dict[str, Any]] = None

        if L_max is not None and L_max >= 1 and multi_horizon_dids is not None:
            # Dynamic placebos DID^{pl}_l
            if self.placebo:
                multi_horizon_placebos = _compute_multi_horizon_placebos(
                    D_mat=D_mat,
                    Y_mat=Y_mat,
                    N_mat=N_mat,
                    baselines=baselines,
                    first_switch_idx=first_switch_idx_arr,
                    switch_direction=switch_direction_arr,
                    T_g=T_g_arr,
                    L_max=L_max,
                    set_ids=set_ids_arr,
                )
                # Surface placebo A11 warnings
                pl_a11 = multi_horizon_placebos.pop("_a11_warnings", None)
                if pl_a11:
                    warnings.warn(
                        f"Multi-horizon placebo control-availability "
                        f"violations in {len(pl_a11)} (group, lag) pair(s): "
                        f"affected groups are excluded from N^{{pl}}_l "
                        f"(no observed controls). Examples: "
                        + ", ".join(pl_a11[:3])
                        + (f" (and {len(pl_a11) - 3} more)" if len(pl_a11) > 3 else ""),
                        UserWarning,
                        stacklevel=2,
                    )

            # Placebo IF computation + analytical SE
            if multi_horizon_placebos is not None:
                placebo_horizon_if = _compute_per_group_if_placebo_horizon(
                    D_mat=D_mat,
                    Y_mat=Y_mat,
                    N_mat=N_mat,
                    baselines=baselines,
                    first_switch_idx=first_switch_idx_arr,
                    switch_direction=switch_direction_arr,
                    T_g=T_g_arr,
                    L_max=L_max,
                    set_ids=set_ids_arr,
                )
                # Per-placebo-horizon analytical SE via cohort recentering
                # (same pattern as positive-horizon SE at Step 12c).
                placebo_horizon_se: Dict[int, float] = {}
                placebo_horizon_inference: Dict[int, Dict[str, Any]] = {}
                singleton_baseline_set_pl = set(singleton_baseline_groups)
                eligible_mask_pl = np.array(
                    [g not in singleton_baseline_set_pl for g in all_groups],
                    dtype=bool,
                )
                for lag_l in range(1, L_max + 1):
                    pl_data = multi_horizon_placebos.get(lag_l)
                    if pl_data is None or pl_data["N_pl_l"] == 0:
                        placebo_horizon_se[lag_l] = float("nan")
                        continue
                    U_pl = placebo_horizon_if[lag_l]
                    # Cohort IDs (same as positive horizons)
                    cohort_keys_pl = [
                        (
                            float(baselines[g]),
                            int(first_switch_idx_arr[g]),
                            int(switch_direction_arr[g]),
                        )
                        for g in range(len(all_groups))
                    ]
                    unique_cpl: Dict[Tuple[float, int, int], int] = {}
                    cid_pl = np.zeros(len(all_groups), dtype=int)
                    for g in range(len(all_groups)):
                        if not eligible_mask_pl[g]:
                            cid_pl[g] = -1
                            continue
                        key = cohort_keys_pl[g]
                        if key not in unique_cpl:
                            unique_cpl[key] = len(unique_cpl)
                        cid_pl[g] = unique_cpl[key]
                    U_pl_elig = U_pl[eligible_mask_pl]
                    cid_elig_pl = cid_pl[eligible_mask_pl]
                    U_centered_pl_l = _cohort_recenter(U_pl_elig, cid_elig_pl)
                    se_pl_l = _plugin_se(
                        U_centered=U_centered_pl_l, divisor=pl_data["N_pl_l"]
                    )
                    placebo_horizon_se[lag_l] = se_pl_l
                    pl_val = pl_data["placebo_l"]
                    t_pl_l, p_pl_l, ci_pl_l = safe_inference(
                        pl_val, se_pl_l, alpha=self.alpha, df=None
                    )
                    placebo_horizon_inference[lag_l] = {
                        "effect": pl_val,
                        "se": se_pl_l,
                        "t_stat": t_pl_l,
                        "p_value": p_pl_l,
                        "conf_int": ci_pl_l,
                        "n_obs": pl_data["N_pl_l"],
                    }

            # Normalized effects DID^n_l (suppressed under trends_linear
            # because event_study_effects holds second-differences DID^{fd}_l,
            # not level effects - normalizing second-differences is wrong)
            if not _is_trends_linear:
                normalized_effects_dict = _compute_normalized_effects(
                    multi_horizon_dids=multi_horizon_dids,
                    D_mat=D_mat,
                    baselines=baselines,
                    first_switch_idx=first_switch_idx_arr,
                    L_max=L_max,
                )

            # Cost-benefit delta (only meaningful when L_max >= 2)
            if L_max >= 2:
                cost_benefit_result = _compute_cost_benefit_delta(
                multi_horizon_dids=multi_horizon_dids,
                D_mat=D_mat,
                baselines=baselines,
                first_switch_idx=first_switch_idx_arr,
                switch_direction=switch_direction_arr,
                L_max=L_max,
            )
                if cost_benefit_result.get("has_leavers", False):
                    warnings.warn(
                        "Assumption 7 (D_{g,t} >= D_{g,1}) is violated: leavers "
                        "present. The cost-benefit delta is computed on the full "
                        "sample (both joiners and leavers); delta_joiners and "
                        "delta_leavers are available separately on "
                        "results.cost_benefit_delta.",
                        UserWarning,
                        stacklevel=2,
                    )

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
            D_mat=D_mat,
            # Phase 1 IF uses per-period structure: use raw outcomes
            # when controls or trends_linear transform Y_mat.
            Y_mat=Y_mat_raw if controls is not None else (y_pivot.to_numpy() if _is_trends_linear else Y_mat),
            N_mat=N_mat_orig,
            n_10_t_arr=n_10_t_arr,
            n_00_t_arr=n_00_t_arr,
            n_01_t_arr=n_01_t_arr,
            n_11_t_arr=n_11_t_arr,
            a11_plus_zeroed_arr=a11_plus_zeroed_arr,
            a11_minus_zeroed_arr=a11_minus_zeroed_arr,
            all_groups=all_groups,
            singleton_baseline_groups=singleton_baseline_groups,
        )

        # Analytical SE for DID_M
        overall_se = _plugin_se(U_centered=U_centered_overall, divisor=N_S)
        # Detect the degenerate-cohort case: every variance-eligible group
        # forms its own (D_{g,1}, F_g, S_g) cohort, so the centered
        # influence function is identically zero and `_plugin_se` returns
        # NaN. Surface this as a UserWarning so users see the variance is
        # unidentified rather than silently mistaking NaN for "missing
        # data" or 0.0 for infinite precision. The bootstrap path inherits
        # the same degeneracy on this panel because it multiplies the
        # same all-zero centered IF by random weights.
        if np.isnan(overall_se) and n_groups_for_overall_var > 0 and N_S > 0:
            warnings.warn(
                f"Cohort-recentered analytical variance is unidentified: "
                f"every variance-eligible group forms its own "
                f"(D_{{g,1}}, F_g, S_g) cohort "
                f"({n_groups_for_overall_var} groups across {n_cohorts} "
                f"cohorts), so the centered influence function vector is "
                f"identically zero. The DID_M point estimate is still "
                f"valid; SE / t_stat / p_value / conf_int are NaN-"
                f"consistent. To get a non-degenerate analytical SE, "
                f"include more groups so cohorts have peers (real-world "
                f"panels typically have G >> K). The bootstrap path "
                f"inherits the same degeneracy on this data.",
                UserWarning,
                stacklevel=2,
            )
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

        # Phase 1 per-period placebo (L_max=None): SE is NaN because the
        # per-period DID_M^pl aggregation path does not have an IF
        # derivation. Multi-horizon placebos (L_max >= 1) use the per-group
        # placebo IF computed above and have valid SE.
        placebo_se = float("nan")
        placebo_t = float("nan")
        placebo_p = float("nan")
        placebo_ci: Tuple[float, float] = (float("nan"), float("nan"))
        if placebo_available and L_max is None:
            warnings.warn(
                "Single-period placebo SE (L_max=None) is NaN. The "
                "per-period DID_M^pl aggregation path does not have an "
                "influence-function derivation. Use L_max >= 1 for "
                "multi-horizon placebos with valid SE. The placebo "
                "point estimate (results.placebo_effect) is still "
                "meaningful.",
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
            # Phase 1 placebo bootstrap: the Phase 1 per-period placebo
            # DID_M^pl still uses NaN SE (no IF derivation for the
            # per-period aggregation). The multi-horizon placebo bootstrap
            # below handles Phase 2+ placebos when placebo_horizon_if is
            # available.
            placebo_inputs = None

            # Phase 2: build placebo-horizon bootstrap inputs from the
            # cohort-centered placebo IF vectors.
            pl_boot_inputs = None
            if (
                placebo_horizon_if is not None
                and multi_horizon_placebos is not None
                and L_max is not None
                and L_max >= 1
            ):
                singleton_baseline_set_pl_b = set(singleton_baseline_groups)
                eligible_mask_pl_b = np.array(
                    [g not in singleton_baseline_set_pl_b for g in all_groups],
                    dtype=bool,
                )
                pl_boot_inputs = {}
                for lag_l in range(1, L_max + 1):
                    pl_data = multi_horizon_placebos.get(lag_l)
                    if pl_data is None or pl_data["N_pl_l"] == 0:
                        continue
                    U_pl_full = placebo_horizon_if[lag_l]
                    U_pl_elig_b = U_pl_full[eligible_mask_pl_b]
                    cohort_keys_pl_b = [
                        (
                            float(baselines[g]),
                            int(first_switch_idx_arr[g]),
                            int(switch_direction_arr[g]),
                        )
                        for g in range(len(all_groups))
                    ]
                    unique_cpl_b: Dict[Tuple[float, int, int], int] = {}
                    cid_pl_b = np.zeros(len(all_groups), dtype=int)
                    for g in range(len(all_groups)):
                        if not eligible_mask_pl_b[g]:
                            cid_pl_b[g] = -1
                            continue
                        key = cohort_keys_pl_b[g]
                        if key not in unique_cpl_b:
                            unique_cpl_b[key] = len(unique_cpl_b)
                        cid_pl_b[g] = unique_cpl_b[key]
                    cid_elig_pl_b = cid_pl_b[eligible_mask_pl_b]
                    U_centered_pl_b = _cohort_recenter(U_pl_elig_b, cid_elig_pl_b)
                    pl_boot_inputs[lag_l] = (
                        U_centered_pl_b,
                        pl_data["N_pl_l"],
                        pl_data["placebo_l"],
                    )

            # Phase 2: build multi-horizon bootstrap inputs from the
            # cohort-centered IF vectors computed in Step 12c.
            mh_boot_inputs = None
            if (
                multi_horizon_if is not None
                and multi_horizon_dids is not None
                and multi_horizon_se is not None
                and L_max is not None
                and L_max >= 1
            ):
                singleton_baseline_set_b = set(singleton_baseline_groups)
                eligible_mask_b = np.array(
                    [g not in singleton_baseline_set_b for g in all_groups], dtype=bool
                )
                mh_boot_inputs = {}
                # Include ALL horizons 1..L_max so the sup-t critical
                # value is calibrated over the same set that receives
                # cband_conf_int. For l=1, use the per-group IF (not
                # the Phase 1 per-period IF) so the bootstrap matches
                # the event_study_effects[1] estimand.
                for l_h in range(1, L_max + 1):
                    h_data = multi_horizon_dids.get(l_h)
                    if h_data is None or h_data["N_l"] == 0:
                        continue
                    U_l_full = multi_horizon_if[l_h]
                    # Full variance-eligible group set (matching
                    # analytical SE path: singleton-baseline only)
                    U_l_elig = U_l_full[eligible_mask_b]
                    # Use the same cohort IDs as the analytical SE path
                    cohort_keys_b = [
                        (
                            float(baselines[g]),
                            int(first_switch_idx_arr[g]),
                            int(switch_direction_arr[g]),
                        )
                        for g in range(len(all_groups))
                    ]
                    unique_cb: Dict[Tuple[float, int, int], int] = {}
                    cid_b = np.zeros(len(all_groups), dtype=int)
                    for g in range(len(all_groups)):
                        if not eligible_mask_b[g]:
                            cid_b[g] = -1
                            continue
                        key = cohort_keys_b[g]
                        if key not in unique_cb:
                            unique_cb[key] = len(unique_cb)
                        cid_b[g] = unique_cb[key]
                    cid_elig = cid_b[eligible_mask_b]
                    U_centered_h = _cohort_recenter(U_l_elig, cid_elig)
                    mh_boot_inputs[l_h] = (
                        U_centered_h,
                        h_data["N_l"],
                        h_data["did_l"],
                    )

            br = self._compute_dcdh_bootstrap(
                n_groups_for_overall=n_groups_for_overall_var,
                u_centered_overall=U_centered_overall,
                divisor_overall=N_S,
                original_overall=overall_att,
                joiners_inputs=joiners_inputs,
                leavers_inputs=leavers_inputs,
                placebo_inputs=placebo_inputs,
                multi_horizon_inputs=mh_boot_inputs,
                placebo_horizon_inputs=pl_boot_inputs,
            )
            bootstrap_results = br

            # Replace the analytical SE with the bootstrap SE for the
            # targets that have valid bootstrap output, AND propagate
            # the bootstrap percentile p-value and CI directly to the
            # top-level fields. The t-stat is computed from the SE via
            # safe_inference()[0] so the project anti-pattern rule
            # (never compute t_stat = effect / se inline) stays
            # satisfied — bootstrap does not define an alternative
            # t-stat semantic for percentile bootstrap, so the
            # SE-based t-stat is the natural choice.
            #
            # Library precedent: imputation.py:790-805,
            # two_stage.py:778-787, and efficient_did.py:1009-1013 all
            # propagate bootstrap p/CI to the public surface while
            # keeping a SE-derived t-stat. Round 10 brings dCDH in line
            # with that pattern (the prior code silently recomputed
            # normal-theory p/CI from the bootstrap SE, which made the
            # public inference surface a hybrid).
            #
            # See REGISTRY.md ChaisemartinDHaultfoeuille `Note
            # (bootstrap inference surface)` and the regression test
            # ``test_bootstrap_p_value_and_ci_propagated_to_top_level``.
            if np.isfinite(br.overall_se):
                overall_se = br.overall_se
                overall_p = br.overall_p_value if br.overall_p_value is not None else np.nan
                overall_ci = br.overall_ci if br.overall_ci is not None else (np.nan, np.nan)
                overall_t = safe_inference(overall_att, overall_se, alpha=self.alpha, df=None)[0]
            if joiners_available and br.joiners_se is not None and np.isfinite(br.joiners_se):
                joiners_se = br.joiners_se
                joiners_p = br.joiners_p_value if br.joiners_p_value is not None else np.nan
                joiners_ci = br.joiners_ci if br.joiners_ci is not None else (np.nan, np.nan)
                joiners_t = safe_inference(joiners_att, joiners_se, alpha=self.alpha, df=None)[0]
            if leavers_available and br.leavers_se is not None and np.isfinite(br.leavers_se):
                leavers_se = br.leavers_se
                leavers_p = br.leavers_p_value if br.leavers_p_value is not None else np.nan
                leavers_ci = br.leavers_ci if br.leavers_ci is not None else (np.nan, np.nan)
                leavers_t = safe_inference(leavers_att, leavers_se, alpha=self.alpha, df=None)[0]

        # ------------------------------------------------------------------
        # Step 20: Build the results dataclass
        # ------------------------------------------------------------------
        # event_study_effects: when L_max is None, l=1 mirrors Phase 1
        # DID_M (per-period path). When L_max >= 1, ALL horizons including
        # l=1 use the per-group DID_{g,l} path for a consistent estimand.
        if multi_horizon_inference is not None and 1 in multi_horizon_inference:
            # Per-group mode: use per-group path for all horizons.
            # When L_max >= 1, the per-group DID_{g,1} is the correct
            # estimand for overall_att (not the binary-only per-period
            # DID_M). This handles both pure non-binary (N_S=0) and
            # mixed binary/non-binary panels (N_S > 0 but incomplete).
            l1_inf = multi_horizon_inference[1]
            overall_att = l1_inf["effect"]
            overall_se = l1_inf["se"]
            overall_t = l1_inf["t_stat"]
            overall_p = l1_inf["p_value"]
            overall_ci = l1_inf["conf_int"]
            event_study_effects: Dict[int, Dict[str, Any]] = dict(multi_horizon_inference)
        else:
            # Phase 1 mode (L_max=None): l=1 from per-period path
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

        # Phase 2: propagate bootstrap results to event_study_effects
        if bootstrap_results is not None and bootstrap_results.event_study_ses:
            for l_h in bootstrap_results.event_study_ses:
                if l_h in event_study_effects:
                    bs_se = bootstrap_results.event_study_ses.get(l_h)
                    bs_ci = (
                        bootstrap_results.event_study_cis.get(l_h)
                        if bootstrap_results.event_study_cis
                        else None
                    )
                    bs_p = (
                        bootstrap_results.event_study_p_values.get(l_h)
                        if bootstrap_results.event_study_p_values
                        else None
                    )
                    if bs_se is not None and np.isfinite(bs_se):
                        eff = event_study_effects[l_h]["effect"]
                        event_study_effects[l_h]["se"] = bs_se
                        event_study_effects[l_h]["p_value"] = bs_p if bs_p is not None else np.nan
                        event_study_effects[l_h]["conf_int"] = (
                            bs_ci if bs_ci is not None else (np.nan, np.nan)
                        )
                        event_study_effects[l_h]["t_stat"] = safe_inference(
                            eff, bs_se, alpha=self.alpha, df=None
                        )[0]

            # Add sup-t bands to event_study_effects entries
            if bootstrap_results.cband_crit_value is not None:
                crit = bootstrap_results.cband_crit_value
                for l_h in event_study_effects:
                    se = event_study_effects[l_h]["se"]
                    eff = event_study_effects[l_h]["effect"]
                    if np.isfinite(se) and se > 0:
                        event_study_effects[l_h]["cband_conf_int"] = (
                            eff - crit * se,
                            eff + crit * se,
                        )

        # When L_max >= 1 and the per-group path is active, sync
        # overall_* from event_study_effects[1] AFTER bootstrap propagation
        # so that bootstrap SE/p/CI flow to the top-level surface.
        if (
            L_max is not None
            and L_max >= 1
            and 1 in event_study_effects
        ):
            es1 = event_study_effects[1]
            overall_att = es1["effect"]
            overall_se = es1["se"]
            overall_t = es1["t_stat"]
            overall_p = es1["p_value"]
            overall_ci = es1["conf_int"]
            # Sync nested bootstrap_results.overall_* to DID_1 only when
            # L_max == 1. When L_max >= 2, the cost-benefit delta overrides
            # overall_* later, so bootstrap_results.overall_* should stay
            # on the scalar DID_M bootstrap (or be overridden by delta logic).
            if (
                L_max == 1
                and bootstrap_results is not None
                and bootstrap_results.event_study_ses
                and 1 in bootstrap_results.event_study_ses
            ):
                bootstrap_results.overall_se = bootstrap_results.event_study_ses[1]
                bootstrap_results.overall_ci = (
                    bootstrap_results.event_study_cis[1]
                    if bootstrap_results.event_study_cis and 1 in bootstrap_results.event_study_cis
                    else (np.nan, np.nan)
                )
                # Clear the DID_M distribution - it doesn't match the
                # DID_1 summary statistics. The per-horizon bootstrap
                # stats are accessible via event_study_ses/cis/p_values.
                bootstrap_results.bootstrap_distribution = None
                bootstrap_results.overall_p_value = (
                    bootstrap_results.event_study_p_values[1]
                    if bootstrap_results.event_study_p_values
                    and 1 in bootstrap_results.event_study_p_values
                    else np.nan
                )

        # Phase 2: override overall_att with cost-benefit delta when L_max > 1
        effective_overall_att = overall_att
        effective_overall_se = overall_se
        effective_overall_t = overall_t
        effective_overall_p = overall_p
        effective_overall_ci = overall_ci
        if cost_benefit_result is not None and L_max is not None and L_max >= 2:
            delta_val = cost_benefit_result["delta"]
            if not np.isfinite(delta_val):
                # Delta is non-estimable (e.g., no eligible switchers at
                # any horizon). Set all overall_* to NaN rather than
                # silently falling back to the Phase 1 DID_M values,
                # since the results surface labels them as delta.
                effective_overall_att = float("nan")
                effective_overall_se = float("nan")
                effective_overall_t = float("nan")
                effective_overall_p = float("nan")
                effective_overall_ci = (float("nan"), float("nan"))
            else:
                effective_overall_att = delta_val
                # Cost-benefit delta SE: compute from per-horizon bootstrap
                # distributions if available (delta = sum w_l * DID_l, so
                # delta_b = sum w_l * DID_l_b for each bootstrap rep).
                # Delta-method SE: Var(delta) = sum w_l^2 * Var(DID_l)
                # (treating horizons as independent, conservative under
                # Assumption 8). Works on both analytical and bootstrap
                # SEs since event_study_effects[l]["se"] holds whichever
                # was propagated.
                # Require ALL positively-weighted horizons to have finite
                # SE. If any has NaN, delta SE is NaN (NaN-consistent
                # inference contract: no partial aggregation).
                weights = cost_benefit_result.get("weights", {})
                var_delta = 0.0
                all_finite = True
                for l_w, w_l in weights.items():
                    if w_l <= 0:
                        continue
                    se_l = event_study_effects.get(l_w, {}).get("se", float("nan"))
                    if not np.isfinite(se_l):
                        all_finite = False
                        break
                    var_delta += (w_l * se_l) ** 2
                delta_se = (
                    float(np.sqrt(var_delta)) if all_finite and var_delta > 0 else float("nan")
                )

                if np.isfinite(delta_se):
                    effective_overall_se = delta_se
                    effective_overall_t, effective_overall_p, effective_overall_ci = safe_inference(
                        delta_val, delta_se, alpha=self.alpha, df=None
                    )
                else:
                    effective_overall_se = float("nan")
                    effective_overall_t = float("nan")
                    effective_overall_p = float("nan")
                    effective_overall_ci = (float("nan"), float("nan"))

        # Phase 2: build placebo_event_study with negative keys.
        # Use analytical SE from placebo IF (computed above), with
        # bootstrap override when available.
        placebo_event_study_dict: Optional[Dict[int, Dict[str, Any]]] = None
        if multi_horizon_placebos is not None:
            placebo_event_study_dict = {}
            for lag_l, pl_data in multi_horizon_placebos.items():
                if pl_data["N_pl_l"] > 0:
                    # Pull analytical SE from placebo IF computation
                    if (
                        placebo_horizon_inference is not None
                        and lag_l in placebo_horizon_inference
                    ):
                        inf = placebo_horizon_inference[lag_l]
                        placebo_event_study_dict[-lag_l] = {
                            "effect": inf["effect"],
                            "se": inf["se"],
                            "t_stat": inf["t_stat"],
                            "p_value": inf["p_value"],
                            "conf_int": inf["conf_int"],
                            "n_obs": inf["n_obs"],
                        }
                    else:
                        # Fallback: NaN SE (Phase 1 path or missing IF)
                        pl_se = float("nan")
                        pl_t, pl_p, pl_ci = safe_inference(
                            pl_data["placebo_l"], pl_se, alpha=self.alpha, df=None
                        )
                        placebo_event_study_dict[-lag_l] = {
                            "effect": pl_data["placebo_l"],
                            "se": pl_se,
                            "t_stat": pl_t,
                            "p_value": pl_p,
                            "conf_int": pl_ci,
                            "n_obs": pl_data["N_pl_l"],
                        }
                else:
                    placebo_event_study_dict[-lag_l] = {
                        "effect": float("nan"),
                        "se": float("nan"),
                        "t_stat": float("nan"),
                        "p_value": float("nan"),
                        "conf_int": (float("nan"), float("nan")),
                        "n_obs": 0,
                    }

        # Propagate bootstrap results to placebo_event_study (must run
        # after placebo_event_study_dict is assembled above).
        if (
            bootstrap_results is not None
            and bootstrap_results.placebo_horizon_ses
            and placebo_event_study_dict is not None
        ):
            for lag_l in bootstrap_results.placebo_horizon_ses:
                neg_key = -lag_l
                if neg_key in placebo_event_study_dict:
                    bs_se = bootstrap_results.placebo_horizon_ses.get(lag_l)
                    bs_ci = (
                        bootstrap_results.placebo_horizon_cis.get(lag_l)
                        if bootstrap_results.placebo_horizon_cis
                        else None
                    )
                    bs_p = (
                        bootstrap_results.placebo_horizon_p_values.get(lag_l)
                        if bootstrap_results.placebo_horizon_p_values
                        else None
                    )
                    if bs_se is not None and np.isfinite(bs_se):
                        eff = placebo_event_study_dict[neg_key]["effect"]
                        placebo_event_study_dict[neg_key]["se"] = bs_se
                        placebo_event_study_dict[neg_key]["p_value"] = (
                            bs_p if bs_p is not None else np.nan
                        )
                        placebo_event_study_dict[neg_key]["conf_int"] = (
                            bs_ci if bs_ci is not None else (np.nan, np.nan)
                        )
                        placebo_event_study_dict[neg_key]["t_stat"] = safe_inference(
                            eff, bs_se, alpha=self.alpha, df=None
                        )[0]

        # Phase 2: build normalized_effects with SE
        normalized_effects_out: Optional[Dict[int, Dict[str, Any]]] = None
        if normalized_effects_dict is not None and multi_horizon_se is not None:
            normalized_effects_out = {}
            for l_h, n_data in normalized_effects_dict.items():
                denom = n_data["denominator"]
                eff = n_data["effect"]
                # SE via delta method: SE(DID^n_l) = SE(DID_l) / delta^D_l
                se_did_l = multi_horizon_se.get(l_h, float("nan"))
                se_norm = se_did_l / denom if np.isfinite(denom) and denom > 0 else float("nan")
                t_n, p_n, ci_n = safe_inference(eff, se_norm, alpha=self.alpha, df=None)
                normalized_effects_out[l_h] = {
                    "effect": eff,
                    "se": se_norm,
                    "t_stat": t_n,
                    "p_value": p_n,
                    "conf_int": ci_n,
                    "denominator": denom,
                }

        # ------------------------------------------------------------------
        # DID^{fd} cumulation: recover level effects from second-differences
        #
        # DID^{fd}_l identifies delta_{g,l} - delta_{g,l-1} (Lemma 6).
        # Cumulate per-group: for each group eligible at horizon l,
        # sum DID^{fd}_{g,l'} for l'=1..l, then average over that
        # eligible set. This matches R's did_multiplegt_dyn which
        # cumulates per-group then aggregates (NOT sum-of-aggregates,
        # which mixes different eligible populations).
        # ------------------------------------------------------------------
        if _is_trends_linear and multi_horizon_dids is not None:
            cumulated = {}
            n_groups_total = D_mat.shape[0]
            # Accumulate per-group running sum of DID^{fd}_{g,l'}
            running_per_group = np.zeros(n_groups_total)
            for l_h in range(1, (L_max or 0) + 1):
                if l_h not in multi_horizon_dids:
                    continue
                mh = multi_horizon_dids[l_h]
                did_g_l = mh["did_g_l"]         # (n_groups,) per-group DID
                eligible = mh["eligible_mask"]   # (n_groups,) bool
                N_l = mh["N_l"]
                if N_l == 0:
                    continue
                # Add this horizon's per-group DID to running sum
                # (NaN for ineligible groups; use 0 for accumulation)
                increment = np.where(np.isfinite(did_g_l), did_g_l, 0.0)
                running_per_group += increment
                # Average the cumulated sum over groups eligible at THIS horizon
                # Weight by S_g (switch direction) and divide by N_l
                S_arr = switch_direction_arr.astype(float)
                cum_effect = float(
                    np.sum(S_arr[eligible] * running_per_group[eligible]) / N_l
                )
                # SE: conservative upper bound (sum of per-horizon SEs).
                # NaN-consistency: if ANY component SE up to horizon l is
                # non-finite, the cumulated SE is NaN (not 0.0).
                if event_study_effects is not None:
                    component_ses = [
                        event_study_effects.get(ll, {}).get("se", np.nan)
                        for ll in range(1, l_h + 1)
                    ]
                    if all(np.isfinite(s) for s in component_ses):
                        running_se_ub = sum(component_ses)
                    else:
                        running_se_ub = float("nan")
                else:
                    running_se_ub = float("nan")
                cum_t, cum_p, cum_ci = safe_inference(
                    cum_effect, running_se_ub, alpha=self.alpha, df=None
                )
                cumulated[l_h] = {
                    "effect": cum_effect,
                    "se": running_se_ub,
                    "t_stat": cum_t,
                    "p_value": cum_p,
                    "conf_int": cum_ci,
                }
            linear_trends_effects = cumulated if cumulated else None

        # When trends_linear=True and L_max>=2, suppress cost_benefit_delta
        # and NaN out the overall_* surface. R's did_multiplegt_dyn with
        # trends_lin=TRUE does not compute an aggregate "average total
        # effect" - users should access cumulated level effects via
        # results.linear_trends_effects[l] instead.
        if _is_trends_linear and L_max is not None and L_max >= 2:
            cost_benefit_result = None
            effective_overall_att = float("nan")
            effective_overall_se = float("nan")
            effective_overall_t = float("nan")
            effective_overall_p = float("nan")
            effective_overall_ci = (float("nan"), float("nan"))

        # ------------------------------------------------------------------
        # Heterogeneity testing (Web Appendix Section 1.5, Lemma 7)
        # ------------------------------------------------------------------
        heterogeneity_effects: Optional[Dict[int, Dict[str, Any]]] = None
        if heterogeneity is not None:
            if L_max is None:
                raise ValueError(
                    "heterogeneity testing requires L_max >= 1. Set L_max "
                    "to use the per-group DID_{g,l} path."
                )
            het_col = str(heterogeneity)
            if het_col not in data.columns:
                raise ValueError(
                    f"heterogeneity column {het_col!r} not found in data."
                )
            # R's predict_het disallows controls; our partial implementation
            # follows this restriction to avoid inconsistent behavior.
            if controls is not None:
                raise ValueError(
                    "heterogeneity cannot be combined with controls. "
                    "R's did_multiplegt_dyn disallows predict_het with "
                    "controls; remove one of the two options."
                )
            if _is_trends_linear:
                raise ValueError(
                    "heterogeneity cannot be combined with trends_linear. "
                    "The heterogeneity test operates on level outcome "
                    "changes but trends_linear uses second-differenced "
                    "outcomes; the results would be inconsistent."
                )
            if trends_nonparam is not None:
                raise ValueError(
                    "heterogeneity cannot be combined with trends_nonparam. "
                    "The heterogeneity test does not thread state-set "
                    "control-pool restrictions; the results would be "
                    "inconsistent with the fitted estimator."
                )
            # Extract per-group covariate (must be time-invariant)
            het_per_group = data.groupby(group)[het_col].nunique()
            het_varying = het_per_group[het_per_group > 1]
            if len(het_varying) > 0:
                raise ValueError(
                    f"heterogeneity column {het_col!r} must be "
                    f"time-invariant within each group. "
                    f"{len(het_varying)} group(s) have varying values."
                )
            het_map = data.groupby(group)[het_col].first()
            X_het = np.array(
                [float(het_map.loc[g]) for g in all_groups]
            )
            # Use original Y_mat (not first-differenced) for heterogeneity
            # test, since it operates on level differences Y[out] - Y[ref].
            # When trends_linear, the DID^{fd} second-differences are in
            # event_study_effects but the het test uses level outcomes.
            Y_het = Y_mat if not _is_trends_linear else y_pivot.to_numpy()
            N_het = N_mat_orig
            heterogeneity_effects = _compute_heterogeneity_test(
                Y_mat=Y_het,
                N_mat=N_het,
                baselines=baselines,
                first_switch_idx=first_switch_idx_arr,
                switch_direction=switch_direction_arr,
                T_g=T_g_arr,
                X_het=X_het,
                L_max=L_max,
                alpha=self.alpha,
                rank_deficient_action=self.rank_deficient_action,
            )

        twfe_weights_df = None
        twfe_fraction_negative = None
        twfe_sigma_fe = None
        twfe_beta_fe = None
        if twfe_diagnostic_payload is not None:
            twfe_weights_df = twfe_diagnostic_payload.weights
            twfe_fraction_negative = twfe_diagnostic_payload.fraction_negative
            twfe_sigma_fe = twfe_diagnostic_payload.sigma_fe
            twfe_beta_fe = twfe_diagnostic_payload.beta_fe

        # When L_max >= 1, the overall estimand is per-group DID_1
        # (not per-period DID_M). The joiner/leaver decomposition is a
        # per-period DID_M concept and can differ from DID_1 on mixed
        # panels, so it's suppressed for all L_max >= 1 cases. N_S and
        # n_treated_obs are updated from the per-group path.
        effective_N_S = N_S
        effective_n_treated = n_treated_obs_post
        effective_joiners_available = joiners_available
        effective_leavers_available = leavers_available
        if (
            L_max is not None
            and L_max >= 1
            and multi_horizon_dids is not None
            and 1 in multi_horizon_dids
        ):
            # Use horizon-1 eligible switcher count as the effective N_S
            effective_N_S = multi_horizon_dids[1]["N_l"]
            if not is_binary:
                # For non-binary: count all observations where treatment
                # differs from baseline
                effective_n_treated = int(
                    N_mat[D_mat != D_mat[:, 0:1]].sum()
                ) if D_mat.shape[1] > 1 else 0
            if not is_binary:
                # Suppress binary-only Phase 1 artifacts on non-binary
                # panels: per_period_effects and single-period placebo
                # are DID_M concepts that don't apply to non-binary data.
                per_period_effects = {}
                placebo_effect = float("nan")
                placebo_se = float("nan")
                placebo_t = float("nan")
                placebo_p = float("nan")
                placebo_ci = (float("nan"), float("nan"))
                placebo_available = False
            # Suppress joiner/leaver decomposition for all L_max >= 1
            # (the decomposition is a per-period DID_M concept, not
            # applicable to the per-group DID_1 estimand)
            effective_joiners_available = False
            effective_leavers_available = False

        results = ChaisemartinDHaultfoeuilleResults(
            overall_att=effective_overall_att,
            overall_se=effective_overall_se,
            overall_t_stat=effective_overall_t,
            overall_p_value=effective_overall_p,
            overall_conf_int=effective_overall_ci,
            joiners_att=joiners_att if effective_joiners_available else float("nan"),
            joiners_se=joiners_se if effective_joiners_available else float("nan"),
            joiners_t_stat=joiners_t if effective_joiners_available else float("nan"),
            joiners_p_value=joiners_p if effective_joiners_available else float("nan"),
            joiners_conf_int=joiners_ci if effective_joiners_available else (float("nan"), float("nan")),
            n_joiner_cells=n_joiner_cells if effective_joiners_available else 0,
            n_joiner_obs=n_joiner_obs if effective_joiners_available else 0,
            joiners_available=effective_joiners_available,
            leavers_att=leavers_att if effective_leavers_available else float("nan"),
            leavers_se=leavers_se if effective_leavers_available else float("nan"),
            leavers_t_stat=leavers_t if effective_leavers_available else float("nan"),
            leavers_p_value=leavers_p if effective_leavers_available else float("nan"),
            leavers_conf_int=leavers_ci if effective_leavers_available else (float("nan"), float("nan")),
            n_leaver_cells=n_leaver_cells if effective_leavers_available else 0,
            n_leaver_obs=n_leaver_obs if effective_leavers_available else 0,
            leavers_available=effective_leavers_available,
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
            n_treated_obs=effective_n_treated,
            n_switcher_cells=effective_N_S,
            n_cohorts=n_cohorts,
            n_groups_dropped_crossers=n_groups_dropped_crossers,
            n_groups_dropped_singleton_baseline=n_groups_dropped_singleton_baseline,
            n_groups_dropped_never_switching=n_groups_dropped_never_switching,
            event_study_effects=event_study_effects,
            L_max=L_max,
            placebo_event_study=placebo_event_study_dict,
            twfe_weights=twfe_weights_df,
            twfe_fraction_negative=twfe_fraction_negative,
            twfe_sigma_fe=twfe_sigma_fe,
            twfe_beta_fe=twfe_beta_fe,
            alpha=self.alpha,
            normalized_effects=normalized_effects_out,
            cost_benefit_delta=cost_benefit_result,
            sup_t_bands=(
                {
                    "crit_value": bootstrap_results.cband_crit_value,
                    "alpha": self.alpha,
                    "n_bootstrap": self.n_bootstrap,
                    "method": "multiplier_bootstrap",
                }
                if bootstrap_results is not None and bootstrap_results.cband_crit_value is not None
                else None
            ),
            bootstrap_results=bootstrap_results,
            covariate_residuals=(
                _build_covariate_diagnostics_df(covariate_diagnostics, controls)
                if covariate_diagnostics is not None
                else None
            ),
            linear_trends_effects=linear_trends_effects,
            heterogeneity_effects=heterogeneity_effects,
            design2_effects=(
                _compute_design2_effects(
                    D_mat=D_mat,
                    # Design-2 always uses raw level outcomes (not residualized,
                    # not first-differenced). Use y_pivot as the canonical raw source.
                    Y_mat=y_pivot.to_numpy(),
                    N_mat=N_mat_orig,
                    baselines=baselines,
                    first_switch_idx=first_switch_idx_arr,
                    switch_direction=switch_direction_arr,
                    T_g=T_g_arr,
                    L_max=L_max if L_max is not None else 1,
                )
                if design2
                else None
            ),
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
    """Raise ``NotImplementedError`` for any non-default Phase 3 parameter.

    Phase 2 parameters (``L_max``) are validated inline in ``fit()``
    after period detection. The ``aggregate`` parameter is still
    reserved for Phase 3.
    """
    if aggregate is not None:
        raise NotImplementedError(
            f"aggregate={aggregate!r} is reserved for Phase 3 of dCDH. "
            "Multi-horizon event study effects are computed automatically "
            "when L_max is set. See ROADMAP.md Phase 3."
        )
    # L_max is validated inline in fit() after period detection (needs
    # the period count). Not gated here.
    # controls gate lifted — DID^X covariate residualization implemented.
    # Validation (L_max >= 1 required) is in fit() after L_max detection.
    # trends_linear gate lifted - DID^{fd} linear trends implemented.
    # Validation (L_max >= 1, n_periods >= 3 required) is in fit().
    # trends_nonparam gate lifted - state-set trends implemented.
    # Validation (L_max >= 1, column exists, time-invariant) is in fit().
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
    Drop groups with more than one treatment-change period.

    The dCDH estimator uses the **first treatment change** (``F_g``) as
    the cohort marker for both the per-group building block ``DID_{g,l}``
    and the variance computation. Groups with a second treatment change
    at a later period would confound the multi-horizon estimates because
    ``DID_{g,l}`` attributes the full outcome change from ``F_g-1`` to
    ``F_g-1+l`` to the first switch, while the second switch also
    contributes to that outcome change.

    For binary treatment, >1 change means a reversal (e.g., 0->1->0).
    For non-binary, >1 change includes both reversals (0->2->1) and
    monotone multi-step paths (0->1->2). Both are dropped because the
    dCDH framework requires a single treatment-change event per group.
    A single jump of any magnitude (e.g., 0->3->3->3) has exactly
    1 change period and is kept.

    Parameters
    ----------
    cell : pd.DataFrame
        Cell-level dataset with columns for ``group_col`` and ``d_col``.
        Must be sorted by group and time.
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
    # Count the number of periods with non-zero treatment changes per
    # group. A group with > 1 such period has changed treatment more
    # than once (multi-switch). This generalizes correctly to non-binary
    # treatment: a single jump 0->3 has 1 non-zero diff, while 0->1->0
    # has 2 non-zero diffs.
    diffs = cell.groupby(group_col)[d_col].diff().fillna(0)
    n_changes = (diffs != 0).groupby(cell[group_col]).sum()
    multi_switch_groups = n_changes[n_changes > 1].index.tolist()
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


def _compute_per_period_dids(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    periods: List[Any],
) -> Tuple[
    Dict[Any, Dict[str, Any]],
    List[str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
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
    n_00_t_arr : np.ndarray
        Stable-untreated cell counts aligned to ``periods[1:]``.
    n_11_t_arr : np.ndarray
        Stable-treated cell counts aligned to ``periods[1:]``.
    a11_plus_zeroed_arr : np.ndarray
        Boolean flags marking periods where DID_+,t was zeroed by the
        A11 convention (joiners present but no stable_0 controls).
    a11_minus_zeroed_arr : np.ndarray
        Mirror for DID_-,t.
    """
    n_periods = len(periods)
    per_period_effects: Dict[Any, Dict[str, Any]] = {}
    a11_warnings: List[str] = []
    did_plus_t_list: List[float] = []
    did_minus_t_list: List[float] = []
    n_10_t_list: List[int] = []
    n_01_t_list: List[int] = []
    n_00_t_list: List[int] = []
    n_11_t_list: List[int] = []
    a11_plus_zeroed_list: List[bool] = []
    a11_minus_zeroed_list: List[bool] = []

    for t_idx in range(1, n_periods):
        d_curr = D_mat[:, t_idx]
        d_prev = D_mat[:, t_idx - 1]
        y_curr = Y_mat[:, t_idx]
        y_prev = Y_mat[:, t_idx - 1]
        n_curr = N_mat[:, t_idx]

        # Cell-presence guard: a (g, t) cell only counts if BOTH t and t-1
        # were observed for that group (n_gt > 0 and n_{g,t-1} > 0).
        n_prev = N_mat[:, t_idx - 1]
        present = (n_curr > 0) & (n_prev > 0)

        joiner_mask = (d_prev == 0) & (d_curr == 1) & present
        stable0_mask = (d_prev == 0) & (d_curr == 0) & present
        leaver_mask = (d_prev == 1) & (d_curr == 0) & present
        stable1_mask = (d_prev == 1) & (d_curr == 1) & present

        # AER 2020 Theorem 3 N_{a,b,t} weights are CELL counts, not
        # within-cell observation sums. Each (g, t) cell contributes once
        # regardless of how many original observations fed into the
        # y_gt cell mean. See REGISTRY.md ChaisemartinDHaultfoeuille
        # estimator equations.
        n_10 = int(joiner_mask.sum())
        n_00 = int(stable0_mask.sum())
        n_01 = int(leaver_mask.sum())
        n_11 = int(stable1_mask.sum())

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
            # Unweighted means over cells (each cell contributes equally)
            joiner_avg = float((y_curr[joiner_mask] - y_prev[joiner_mask]).mean())
            stable0_avg = float((y_curr[stable0_mask] - y_prev[stable0_mask]).mean())
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
            stable1_avg = float((y_curr[stable1_mask] - y_prev[stable1_mask]).mean())
            leaver_avg = float((y_curr[leaver_mask] - y_prev[leaver_mask]).mean())
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
        n_00_t_list.append(n_00)
        n_11_t_list.append(n_11)
        a11_plus_zeroed_list.append(did_plus_t_a11_zeroed)
        a11_minus_zeroed_list.append(did_minus_t_a11_zeroed)

    return (
        per_period_effects,
        a11_warnings,
        np.array(did_plus_t_list, dtype=float),
        np.array(did_minus_t_list, dtype=float),
        np.array(n_10_t_list, dtype=int),
        np.array(n_01_t_list, dtype=int),
        np.array(n_00_t_list, dtype=int),
        np.array(n_11_t_list, dtype=int),
        np.array(a11_plus_zeroed_list, dtype=bool),
        np.array(a11_minus_zeroed_list, dtype=bool),
    )


def _compute_placebo(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    periods: List[Any],
) -> Optional[Tuple[float, bool, List[str]]]:
    """
    Compute the single-lag placebo DID_M^pl from AER 2020 placebo specification.

    Same logic as DID_M but evaluated on the pre-event difference
    ``Y_{g, t-1} - Y_{g, t-2}`` for cells with three-period histories.
    Requires ``T >= 3``.

    Mirrors the main path's A11 zero-retention machinery: when placebo
    joiners exist but no 3-period stable_0 controls do (or symmetric
    for leavers/stable_1), the affected per-period contribution is set
    to zero AND a warning string is appended to ``placebo_a11_warnings``.
    The caller is responsible for surfacing the consolidated warning.
    The zero-retention preserves the period's switcher count in the
    placebo ``N_S^pl`` denominator, biasing the placebo toward zero in
    the offending direction (matching placebo paper convention).

    Returns
    -------
    None if ``T < 3`` or no qualifying cells. Otherwise a tuple
    ``(placebo_effect, True, placebo_a11_warnings)`` where
    ``placebo_a11_warnings`` is a list of one string per period that
    triggered an A11 violation in the placebo numerator.
    """
    n_periods = len(periods)
    if n_periods < 3:
        return None

    placebo_plus_per_t: List[float] = []
    placebo_minus_per_t: List[float] = []
    n_10_per_t: List[int] = []
    n_01_per_t: List[int] = []
    placebo_a11_warnings: List[str] = []

    for t_idx in range(2, n_periods):
        d_curr = D_mat[:, t_idx]
        d_prev = D_mat[:, t_idx - 1]
        d_pre_prev = D_mat[:, t_idx - 2]
        y_prev = Y_mat[:, t_idx - 1]
        y_pre_prev = Y_mat[:, t_idx - 2]

        # Cell-presence guard: a (g, t) cell only counts if all three
        # consecutive periods (t-2, t-1, t) were observed for the group.
        present = (N_mat[:, t_idx] > 0) & (N_mat[:, t_idx - 1] > 0) & (N_mat[:, t_idx - 2] > 0)

        # Joiners that have a 3-period history with stable D=0 in t-2 and t-1
        joiner_mask = (d_pre_prev == 0) & (d_prev == 0) & (d_curr == 1) & present
        # Stable_0 controls with stable D=0 in t-2 and t-1
        stable0_mask = (d_pre_prev == 0) & (d_prev == 0) & (d_curr == 0) & present
        # Mirror for leavers/stable_1 (3-period stable treatment then leave)
        leaver_mask = (d_pre_prev == 1) & (d_prev == 1) & (d_curr == 0) & present
        stable1_mask = (d_pre_prev == 1) & (d_prev == 1) & (d_curr == 1) & present

        # Placebo weights are CELL counts (matching Theorem 3 convention)
        n_10 = int(joiner_mask.sum())
        n_00 = int(stable0_mask.sum())
        n_01 = int(leaver_mask.sum())
        n_11 = int(stable1_mask.sum())

        # Joiners side: distinguish "no joiners" (natural zero) from
        # "joiners but no stable_0" (A11 violation, flagged + warned)
        if n_10 == 0:
            placebo_plus_t = 0.0
        elif n_00 == 0:
            placebo_plus_t = 0.0
            placebo_a11_warnings.append(
                f"period {periods[t_idx]}: placebo joiners present, no stable_0"
            )
        else:
            joiner_avg = float((y_prev[joiner_mask] - y_pre_prev[joiner_mask]).mean())
            stable0_avg = float((y_prev[stable0_mask] - y_pre_prev[stable0_mask]).mean())
            placebo_plus_t = joiner_avg - stable0_avg

        # Leavers side: symmetric A11 distinction
        if n_01 == 0:
            placebo_minus_t = 0.0
        elif n_11 == 0:
            placebo_minus_t = 0.0
            placebo_a11_warnings.append(
                f"period {periods[t_idx]}: placebo leavers present, no stable_1"
            )
        else:
            stable1_avg = float((y_prev[stable1_mask] - y_pre_prev[stable1_mask]).mean())
            leaver_avg = float((y_prev[leaver_mask] - y_pre_prev[leaver_mask]).mean())
            placebo_minus_t = stable1_avg - leaver_avg

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
    return placebo_effect, True, placebo_a11_warnings


# ======================================================================
# Phase 3: Covariate residualization helpers
# ======================================================================


def _compute_covariate_residualization(
    Y_mat: np.ndarray,
    X_cell: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    rank_deficient_action: str = "warn",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Residualize outcomes by partialling out covariates per baseline treatment.

    Implements ``DID^X`` from Web Appendix Section 1.2 of de Chaisemartin &
    D'Haultfoeuille (2024). For each baseline treatment value *d*, estimates
    ``theta_hat_d`` via OLS of first-differenced outcomes on first-differenced
    covariates with time FEs, restricted to not-yet-treated observations.
    Then residualizes at levels: ``Y_tilde[g,t] = Y[g,t] - X[g,t] @ theta_hat_d``.

    The level-residualization is equivalent to difference-residualization by
    the Frisch-Waugh-Lovell theorem, so all downstream DID computations
    (which use ``Y[g, out] - Y[g, ref]``) automatically produce the correct
    covariate-adjusted estimates.

    Parameters
    ----------
    Y_mat : np.ndarray, shape (n_groups, n_periods)
        Cell-level outcome means.
    X_cell : np.ndarray, shape (n_groups, n_periods, n_covariates)
        Cell-level covariate means.
    N_mat : np.ndarray, shape (n_groups, n_periods)
        Observation counts per cell (>0 if observed).
    baselines : np.ndarray, shape (n_groups,)
        ``D_{g,1}`` baseline treatment values (float).
    first_switch_idx : np.ndarray, shape (n_groups,)
        Column index of first treatment change (-1 if never-switching).

    Returns
    -------
    Y_residualized : np.ndarray, shape (n_groups, n_periods)
        Outcome matrix with covariate effects removed.
    diagnostics : dict
        Keyed by baseline value (float). Each entry has ``theta_hat``
        (covariate coefficients), ``n_obs`` (OLS sample size), and
        ``r_squared`` (first-stage R-squared).
    """
    from diff_diff.linalg import solve_ols

    n_groups, n_periods = Y_mat.shape
    n_covariates = X_cell.shape[2]
    Y_resid = Y_mat.copy()
    diagnostics: Dict[str, Any] = {}
    failed_baselines: set = set()

    # Pre-compute observation validity masks for first-differencing.
    # both_observed[g, t] = True iff N_mat[g, t] > 0 AND N_mat[g, t-1] > 0
    both_observed = np.zeros((n_groups, n_periods), dtype=bool)
    both_observed[:, 1:] = (N_mat[:, 1:] > 0) & (N_mat[:, :-1] > 0)

    # not_yet_switched[g, t] = True iff group g has not switched by period t
    # (first_switch_idx[g] == -1 means never-switcher -> always True)
    t_indices = np.arange(n_periods)[np.newaxis, :]  # (1, n_periods)
    f_g_col = first_switch_idx[:, np.newaxis]  # (n_groups, 1)
    not_yet_switched = (f_g_col == -1) | (f_g_col > t_indices)

    for d_val in np.unique(baselines):
        d_mask = baselines == d_val  # (n_groups,)

        # Valid OLS observations: baseline matches, not-yet-treated, both
        # periods observed, t >= 1 (first-differencing needs t and t-1).
        valid = d_mask[:, np.newaxis] & not_yet_switched & both_observed
        valid_g, valid_t = np.where(valid)

        n_obs = len(valid_g)
        if n_obs == 0:
            diagnostics[float(d_val)] = {
                "theta_hat": np.full(n_covariates, np.nan),
                "n_obs": 0,
                "r_squared": np.nan,
            }
            # NaN out outcomes for failed strata so they're excluded
            # from downstream DID computation (don't mix raw + adjusted).
            group_indices = np.where(d_mask)[0]
            Y_resid[group_indices, :] = np.nan
            failed_baselines.add(float(d_val))
            warnings.warn(
                f"No not-yet-treated observations for baseline treatment "
                f"d={d_val}. Cannot estimate covariate slope theta_hat. "
                f"Groups with this baseline are excluded from the "
                f"covariate-adjusted estimation.",
                UserWarning,
                stacklevel=3,
            )
            continue

        # First-differenced outcomes and covariates
        dY = Y_mat[valid_g, valid_t] - Y_mat[valid_g, valid_t - 1]  # (n_obs,)
        dX = X_cell[valid_g, valid_t] - X_cell[valid_g, valid_t - 1]  # (n_obs, K)

        # Check for non-finite values (NaN from missing covariates/outcomes)
        finite_mask = np.isfinite(dY) & np.all(np.isfinite(dX), axis=1)
        if not finite_mask.all():
            dY = dY[finite_mask]
            dX = dX[finite_mask]
            n_obs = len(dY)
            if n_obs == 0:
                diagnostics[float(d_val)] = {
                    "theta_hat": np.full(n_covariates, np.nan),
                    "n_obs": 0,
                    "r_squared": np.nan,
                }
                continue
            valid_t_finite = valid_t[finite_mask]
        else:
            valid_t_finite = valid_t

        # Build design: [intercept, dX, time_dummies (reference dropped)]
        # The intercept is required when dropping one time dummy as
        # reference category; without it the omitted period's FE is
        # forced to zero, biasing theta_hat.
        intercept = np.ones((n_obs, 1))
        unique_t = np.unique(valid_t_finite)
        n_time_fe = len(unique_t) - 1
        if n_time_fe > 0:
            time_dummies = np.zeros((n_obs, n_time_fe))
            for i, t_val in enumerate(unique_t[1:]):
                time_dummies[:, i] = (valid_t_finite == t_val).astype(float)
            design = np.hstack([intercept, dX, time_dummies])
        else:
            design = np.hstack([intercept, dX])

        # Small-sample guard: skip if fewer obs than parameters
        n_params = design.shape[1]
        if n_obs < n_params:
            diagnostics[float(d_val)] = {
                "theta_hat": np.full(n_covariates, np.nan),
                "n_obs": n_obs,
                "r_squared": np.nan,
            }
            # NaN out outcomes for failed strata (don't mix raw + adjusted)
            group_indices_fail = np.where(d_mask)[0]
            Y_resid[group_indices_fail, :] = np.nan
            failed_baselines.add(float(d_val))
            warnings.warn(
                f"DID^X: baseline d={d_val} has {n_obs} not-yet-treated "
                f"observations but {n_params} regressors. Groups with "
                f"this baseline are excluded from covariate-adjusted "
                f"estimation.",
                UserWarning,
                stacklevel=3,
            )
            continue

        # OLS: dY = [dX, time_FE] @ beta + epsilon
        coefs, residuals, _vcov = solve_ols(
            design,
            dY,
            return_vcov=True,
            rank_deficient_action=rank_deficient_action,
        )

        # Extract covariate coefficients (indices 1..n_covariates;
        # index 0 is the intercept)
        theta_hat = coefs[1:1 + n_covariates]

        # R-squared of first-stage regression
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((dY - dY.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        diagnostics[float(d_val)] = {
            "theta_hat": theta_hat.copy(),
            "n_obs": n_obs,
            "r_squared": r_squared,
        }

        # Guard: if some control coefficients are NaN (rank-deficient
        # OLS dropped collinear controls), residualize with only the
        # finite subset. Replace NaN coefficients with 0 so einsum
        # only uses the identified controls.
        nan_mask = ~np.isfinite(theta_hat)
        if nan_mask.any():
            n_dropped = int(nan_mask.sum())
            warnings.warn(
                f"DID^X: rank-deficient first-stage OLS for baseline "
                f"d={d_val} dropped {n_dropped} collinear control(s). "
                f"Residualization uses the {n_covariates - n_dropped} "
                f"identified control(s).",
                UserWarning,
                stacklevel=3,
            )
            theta_hat = np.where(np.isfinite(theta_hat), theta_hat, 0.0)

        # Residualize Y at levels for all groups with this baseline.
        # Vectorized level residualization: Y_tilde[g, t] = Y[g, t] - X[g, t] @ theta_hat
        group_indices = np.where(d_mask)[0]
        if len(group_indices) > 0:
            # X_sub: (n_d_groups, n_periods, n_covariates), theta: (n_covariates,)
            X_sub = X_cell[group_indices]  # (n_d, T, K)
            adjustment = np.einsum("gtk,k->gt", X_sub, theta_hat)  # (n_d, T)
            # Mask: only adjust cells that are observed and have finite covariates
            valid = (N_mat[group_indices] > 0) & np.all(np.isfinite(X_sub), axis=2)
            Y_resid[group_indices] = np.where(
                valid, Y_mat[group_indices] - adjustment, Y_mat[group_indices]
            )

    return Y_resid, diagnostics, failed_baselines


def _compute_first_differenced_matrix(
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """First-difference the outcome matrix for ``DID^{fd}`` estimation.

    Transforms ``Y_mat`` into first-differences for the group-specific
    linear trends estimator (Web Appendix Section 1.3, Lemma 6). When
    passed to ``_compute_multi_horizon_dids()`` and the IF function,
    the standard ``DID_{g,l}`` formula on ``Z_mat`` produces
    ``DID^{fd}_{g,l}`` exactly.

    The ``F_g >= 3`` constraint (paper, 1-indexed) maps to
    ``first_switch_idx >= 2`` (0-indexed). This is enforced
    automatically: ``N_mat_fd[:, 0] = 0`` causes groups with
    ``first_switch_idx = 1`` to fail the ``N_mat > 0`` eligibility
    check at their reference period.

    Parameters
    ----------
    Y_mat : np.ndarray, shape (n_groups, n_periods)
        Cell-level outcome means (possibly already residualized).
    N_mat : np.ndarray, shape (n_groups, n_periods)
        Observation counts per cell.

    Returns
    -------
    Z_mat : np.ndarray, shape (n_groups, n_periods)
        First-differenced outcomes. ``Z[:, 0] = NaN``,
        ``Z[:, t] = Y[:, t] - Y[:, t-1]`` for ``t >= 1``.
    N_mat_fd : np.ndarray, shape (n_groups, n_periods)
        Adjusted observation counts. ``N_fd[:, 0] = 0``,
        ``N_fd[:, t] = min(N[:, t], N[:, t-1])`` for ``t >= 1``.
    """
    n_groups, n_periods = Y_mat.shape
    Z_mat = np.full((n_groups, n_periods), np.nan)
    Z_mat[:, 1:] = Y_mat[:, 1:] - Y_mat[:, :-1]

    N_mat_fd = np.zeros_like(N_mat)
    N_mat_fd[:, 1:] = np.minimum(N_mat[:, 1:], N_mat[:, :-1])

    return Z_mat, N_mat_fd


def _compute_heterogeneity_test(
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    T_g: np.ndarray,
    X_het: np.ndarray,
    L_max: int,
    alpha: float = 0.05,
    rank_deficient_action: str = "warn",
) -> Dict[int, Dict[str, Any]]:
    """Test for heterogeneous treatment effects (Web Appendix Section 1.5).

    Regresses ``S_g * (Y_{g, F_g-1+l} - Y_{g, F_g-1})`` on ``X_g`` plus
    cohort indicator dummies ``(D_{g,1}, F_g, S_g)``. Under Assumption 15
    (Lemma 7), the coefficient on ``X_g`` is an unbiased estimator of the
    variance-weighted average of effect differences. Standard OLS inference
    is valid - no need to account for DID estimation error.

    Parameters
    ----------
    Y_mat : np.ndarray, shape (n_groups, n_periods)
    N_mat : np.ndarray, shape (n_groups, n_periods)
    baselines, first_switch_idx, switch_direction, T_g : np.ndarray
    X_het : np.ndarray, shape (n_groups,)
        Time-invariant covariate to test for heterogeneity.
    L_max : int
    alpha : float

    Returns
    -------
    dict
        ``{l: {beta, se, t_stat, p_value, conf_int, n_obs}}`` per horizon.
    """
    from diff_diff.linalg import solve_ols
    from diff_diff.utils import safe_inference

    n_groups, n_periods = Y_mat.shape
    results: Dict[int, Dict[str, Any]] = {}

    for l_h in range(1, L_max + 1):
        # Eligible switchers at this horizon (same logic as multi-horizon DID)
        eligible = []
        dep_var = []
        x_vals = []
        cohort_keys = []

        for g in range(n_groups):
            f_g = first_switch_idx[g]
            if f_g < 0:
                continue  # never-switcher
            ref_idx = f_g - 1
            out_idx = f_g - 1 + l_h
            if out_idx >= n_periods:
                continue
            if ref_idx < 0:
                continue
            if N_mat[g, ref_idx] <= 0 or N_mat[g, out_idx] <= 0:
                continue
            if T_g[g] < out_idx:
                continue
            S_g = float(switch_direction[g])
            y_diff = Y_mat[g, out_idx] - Y_mat[g, ref_idx]
            eligible.append(g)
            dep_var.append(S_g * y_diff)
            x_vals.append(X_het[g])
            cohort_keys.append(
                (float(baselines[g]), int(f_g), int(switch_direction[g]))
            )

        n_obs = len(eligible)
        if n_obs < 3:
            results[l_h] = {
                "beta": float("nan"), "se": float("nan"),
                "t_stat": float("nan"), "p_value": float("nan"),
                "conf_int": (float("nan"), float("nan")),
                "n_obs": n_obs,
            }
            continue

        dep_arr = np.array(dep_var)
        x_arr = np.array(x_vals).reshape(-1, 1)

        # Design: [intercept, X_g, cohort_dummies (reference dropped)]
        # The intercept is required when dropping one cohort dummy as
        # reference; without it the omitted cohort's mean is forced to
        # zero, which biases beta^{het}_l.
        intercept = np.ones((n_obs, 1))
        unique_cohorts = sorted(set(cohort_keys))
        n_cohort_dummies = len(unique_cohorts) - 1
        if n_cohort_dummies > 0:
            cohort_map = {c: i for i, c in enumerate(unique_cohorts)}
            cohort_idx = np.array([cohort_map[c] for c in cohort_keys])
            cohort_dummies = np.zeros((n_obs, len(unique_cohorts)))
            cohort_dummies[np.arange(n_obs), cohort_idx] = 1.0
            # Drop first cohort as reference
            cohort_dummies = cohort_dummies[:, 1:]
            design = np.hstack([intercept, x_arr, cohort_dummies])
        else:
            design = np.hstack([intercept, x_arr])

        # Guard: need more observations than parameters
        n_params = design.shape[1]
        if n_obs <= n_params:
            results[l_h] = {
                "beta": float("nan"), "se": float("nan"),
                "t_stat": float("nan"), "p_value": float("nan"),
                "conf_int": (float("nan"), float("nan")),
                "n_obs": n_obs,
            }
            continue

        coefs, _residuals, vcov = solve_ols(
            design, dep_arr,
            return_vcov=True,
            rank_deficient_action=rank_deficient_action,
        )

        # beta_het is at index 1 (index 0 is intercept)
        beta_het = float(coefs[1])
        # NaN-safe: if vcov is None or target coefficient variance is NaN
        # (rank-deficient), all inference fields are NaN.
        se_het = float("nan")
        if vcov is not None and np.isfinite(vcov[1, 1]) and vcov[1, 1] > 0:
            se_het = float(np.sqrt(vcov[1, 1]))
        t_stat, p_val, ci = safe_inference(beta_het, se_het, alpha=alpha, df=None)

        results[l_h] = {
            "beta": beta_het,
            "se": se_het,
            "t_stat": t_stat,
            "p_value": p_val,
            "conf_int": ci,
            "n_obs": n_obs,
        }

    return results


def _compute_design2_effects(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    T_g: np.ndarray,
    L_max: int,
) -> Optional[Dict[str, Any]]:
    """Compute Design-2 switch-in/switch-out effects (Web Appendix Section 1.6).

    Identifies groups with exactly 2 treatment changes (join then leave),
    computes the exit period E_g, and provides delta^+ (post-join) and
    delta^- (post-leave) summaries.

    This is a convenience wrapper that reports descriptive statistics about
    the switch-in and switch-out subpopulations rather than a full
    re-estimation (which would require specialized control pools as
    described in the paper). See REGISTRY.md for documentation.

    Returns None if no join-then-leave groups exist.
    """
    n_groups, n_periods = D_mat.shape

    # Identify join-then-leave groups: exactly 2 treatment changes where
    # the first is a join (D increases) and the second is a leave (D decreases)
    design2_groups = []
    exit_periods = []

    for g in range(n_groups):
        changes = []
        for t in range(1, n_periods):
            if N_mat[g, t] <= 0 or N_mat[g, t - 1] <= 0:
                continue
            if D_mat[g, t] != D_mat[g, t - 1]:
                direction = 1 if D_mat[g, t] > D_mat[g, t - 1] else -1
                changes.append((t, direction))
        if len(changes) == 2 and changes[0][1] == 1 and changes[1][1] == -1:
            design2_groups.append(g)
            exit_periods.append(changes[1][0])

    if len(design2_groups) == 0:
        return None

    # Compute summary statistics for the switch-in/switch-out subpopulation
    switch_in_effects = []
    switch_out_effects = []

    for i, g in enumerate(design2_groups):
        f_g = first_switch_idx[g]
        e_g = exit_periods[i]
        ref_idx = f_g - 1

        # Switch-in: Y[g, f_g] - Y[g, f_g-1] (effect of joining)
        if ref_idx >= 0 and N_mat[g, f_g] > 0 and N_mat[g, ref_idx] > 0:
            switch_in = float(Y_mat[g, f_g] - Y_mat[g, ref_idx])
            switch_in_effects.append(switch_in)

        # Switch-out: Y[g, e_g] - Y[g, e_g-1] (effect of leaving)
        if e_g - 1 >= 0 and N_mat[g, e_g] > 0 and N_mat[g, e_g - 1] > 0:
            switch_out = float(Y_mat[g, e_g] - Y_mat[g, e_g - 1])
            switch_out_effects.append(switch_out)

    result: Dict[str, Any] = {
        "n_design2_groups": len(design2_groups),
        "switch_in": {
            "n_groups": len(switch_in_effects),
            "mean_effect": float(np.mean(switch_in_effects)) if switch_in_effects else np.nan,
        },
        "switch_out": {
            "n_groups": len(switch_out_effects),
            "mean_effect": float(np.mean(switch_out_effects)) if switch_out_effects else np.nan,
        },
    }
    return result


def _build_covariate_diagnostics_df(
    diagnostics: Dict[str, Any],
    control_names: List[str],
) -> pd.DataFrame:
    """Build a tidy DataFrame from the per-baseline residualization diagnostics."""
    rows = []
    for d_val, diag in sorted(diagnostics.items()):
        theta = diag["theta_hat"]
        for k, name in enumerate(control_names):
            rows.append(
                {
                    "baseline_treatment": d_val,
                    "covariate": name,
                    "theta_hat": float(theta[k]) if np.isfinite(theta[k]) else np.nan,
                    "n_obs": diag["n_obs"],
                    "r_squared": diag["r_squared"],
                }
            )
    return pd.DataFrame(rows)


# ======================================================================
# Phase 2: Multi-horizon helpers
# ======================================================================


def _compute_group_switch_metadata(
    D_mat: np.ndarray,
    N_mat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-group switch metadata from the pivoted panel matrices.

    For each group g, identifies the baseline treatment ``D_{g,1}``, the
    first-switch period index ``F_g`` (or -1 if never-switching), and the
    switch direction ``S_g`` (+1 joiner, -1 leaver, 0 never-switching).
    Also computes ``T_g`` - the last period index at which there is still
    a baseline-matched control that hasn't switched (needed for horizon
    eligibility).

    This helper is shared by Phase 1 (cohort-recentered IF in
    ``_compute_cohort_recentered_inputs``) and Phase 2 (multi-horizon
    ``DID_{g,l}`` computation).

    Parameters
    ----------
    D_mat : np.ndarray of shape (n_groups, n_periods)
        Pivoted treatment matrix (cell-level, binary in Phase 1).
    N_mat : np.ndarray of shape (n_groups, n_periods)
        Pivoted observation-count matrix. Zero means group g is missing
        at period t.

    Returns
    -------
    baselines : np.ndarray of shape (n_groups,), dtype int
        ``D_{g,1}`` for each group (treatment at the first global period).
    first_switch_idx : np.ndarray of shape (n_groups,), dtype int
        Period index of g's first treatment change (-1 if never-switching).
        This is ``F_g`` in the paper's notation, expressed as a column
        index into D_mat (0-based).
    switch_direction : np.ndarray of shape (n_groups,), dtype int
        ``S_g``: +1 if treatment increases at first switch (joiner),
        -1 if decreases (leaver), 0 if never-switching.
    T_g : np.ndarray of shape (n_groups,), dtype int
        For each group, the last period index at which a baseline-matched
        not-yet-switched control still exists. Groups whose baseline
        value has no other group that switches later get ``T_g = -1``
        (they have no valid control at any horizon). This is used for
        horizon eligibility: ``DID_{g,l}`` is computable iff
        ``first_switch_idx[g] - 1 + l <= T_g[g]``.

    Raises
    ------
    ValueError
        If any group is missing the first global period in N_mat (this
        should have been caught by fit() Step 5b validation).
    """
    n_groups, n_periods = D_mat.shape

    # Defensive: fit() Step 5b rejects groups missing the baseline.
    if N_mat.size > 0 and (N_mat[:, 0] <= 0).any():
        raise ValueError(
            "_compute_group_switch_metadata: at least one group is missing "
            "the first global period in N_mat. fit() Step 5b should have "
            "rejected this."
        )

    baselines = D_mat[:, 0].astype(float)
    first_switch_idx = np.full(n_groups, -1, dtype=int)
    switch_direction = np.zeros(n_groups, dtype=int)

    for g in range(n_groups):
        for t in range(1, n_periods):
            if N_mat[g, t] <= 0 or N_mat[g, t - 1] <= 0:
                continue
            if D_mat[g, t] != D_mat[g, t - 1]:
                first_switch_idx[g] = t
                switch_direction[g] = 1 if D_mat[g, t] > D_mat[g, t - 1] else -1
                break

    # T_g: for each group g, the last period at which there is still a
    # baseline-matched group whose treatment has NOT changed. This is
    # max_{g': D_{g',1} = D_{g,1}} (F_{g'} - 1), i.e., the period just
    # before the latest-switching control in g's baseline cohort.
    # Never-switching groups (F = -1) have F-1 = T (last period), so
    # they extend T_g to the panel end for their baseline cohort.
    unique_baselines = np.unique(baselines)
    max_control_period = {}  # baseline -> max period index with a valid control
    for d in unique_baselines:
        baseline_mask = baselines == d
        # For each group with this baseline, the last period at which it
        # can still serve as a not-yet-switched control is F_g - 1
        # (or n_periods - 1 if never-switching).
        f_vals = first_switch_idx[baseline_mask]
        control_last = np.where(f_vals == -1, n_periods - 1, f_vals - 1)
        max_control_period[float(d)] = int(control_last.max()) if control_last.size > 0 else -1

    T_g = np.array(
        [max_control_period.get(float(baselines[g]), -1) for g in range(n_groups)],
        dtype=int,
    )

    return baselines, first_switch_idx, switch_direction, T_g


def _compute_multi_horizon_dids(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    T_g: np.ndarray,
    L_max: int,
    set_ids: Optional[np.ndarray] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute the per-group building block ``DID_{g,l}`` and its aggregate
    ``DID_l`` for horizons ``l = 1, ..., L_max``.

    Implements Equation 3 and Equation 5 of the dynamic companion paper
    (NBER WP 29873). For each switching group g eligible at horizon l::

        DID_{g,l} = Y_{g, F_g-1+l} - Y_{g, F_g-1}
                    - mean_{g' in controls} (Y_{g', F_g-1+l} - Y_{g', F_g-1})

    where the control set is ``{g': D_{g',1} = D_{g,1}, F_{g'} > F_g-1+l}``.

    The aggregate is ``DID_l = (1/N_l) * sum S_g * DID_{g,l}`` over
    eligible groups.

    Parameters
    ----------
    D_mat, Y_mat, N_mat : np.ndarray of shape (n_groups, n_periods)
    baselines, first_switch_idx, switch_direction, T_g : np.ndarray
        From ``_compute_group_switch_metadata()``.
    L_max : int
        Maximum horizon to compute.

    Returns
    -------
    dict mapping horizon l -> {
        "did_l": float,           # aggregate DID_l (NaN if N_l=0)
        "N_l": int,               # count of eligible switching groups
        "did_g_l": np.ndarray,    # per-group DID_{g,l} (NaN for non-eligible)
        "eligible_mask": np.ndarray,  # boolean shape (n_groups,)
        "switcher_fraction": float,   # N_l / N_1 (NaN if N_1=0)
    }
    """
    n_groups, n_periods = D_mat.shape
    is_switcher = first_switch_idx >= 0

    # Pre-compute per-baseline lookup of (group_indices, first_switch_indices)
    # for efficient control-pool identification.
    unique_baselines = np.unique(baselines)
    baseline_groups: Dict[float, np.ndarray] = {}
    baseline_f: Dict[float, np.ndarray] = {}
    for d in unique_baselines:
        mask = baselines == d
        baseline_groups[float(d)] = np.where(mask)[0]
        baseline_f[float(d)] = first_switch_idx[mask]

    results: Dict[int, Dict[str, Any]] = {}
    a11_multi_warnings: List[str] = []
    N_1 = 0  # will be set at l=1 for switcher_fraction

    for l in range(1, L_max + 1):  # noqa: E741
        did_g_l = np.full(n_groups, np.nan)

        # Eligibility: switching group with F_g - 1 + l_h observable.
        # F_g is stored as a column index (0-based), so the outcome
        # period is first_switch_idx[g] - 1 + l. This must be a valid
        # column AND the group must be observed there (N_mat > 0).
        # Also, T_g[g] must be >= first_switch_idx[g] - 1 + l (controls
        # available at the outcome period).
        eligible = np.zeros(n_groups, dtype=bool)
        for g in range(n_groups):
            if not is_switcher[g]:
                continue
            f_g = first_switch_idx[g]
            ref_idx = f_g - 1  # period just before first switch
            out_idx = f_g - 1 + l  # outcome period for horizon l
            if ref_idx < 0 or out_idx >= n_periods:
                continue
            if N_mat[g, ref_idx] <= 0 or N_mat[g, out_idx] <= 0:
                continue
            if T_g[g] < out_idx:
                continue  # no baseline-matched control available
            eligible[g] = True

        N_l = int(eligible.sum())
        if l == 1:
            N_1 = N_l

        if N_l == 0:
            results[l] = {
                "did_l": float("nan"),
                "N_l": 0,
                "did_g_l": did_g_l,
                "eligible_mask": eligible,
                "switcher_fraction": float("nan"),
            }
            continue

        # Compute DID_{g,l} for each eligible group.
        for g in np.where(eligible)[0]:
            f_g = first_switch_idx[g]
            ref_idx = f_g - 1
            out_idx = f_g - 1 + l
            d_base = float(baselines[g])

            # Switcher's outcome change
            switcher_change = Y_mat[g, out_idx] - Y_mat[g, ref_idx]

            # Control pool: same baseline, not yet switched by out_idx.
            # F_{g'} > out_idx (hasn't switched yet) OR F_{g'} = -1
            # (never switches). Both must be observed at ref_idx and
            # out_idx.
            ctrl_indices = baseline_groups[d_base]
            ctrl_f = baseline_f[d_base]
            ctrl_mask = (
                ((ctrl_f > out_idx) | (ctrl_f == -1))
                & (N_mat[ctrl_indices, ref_idx] > 0)
                & (N_mat[ctrl_indices, out_idx] > 0)
            )
            # State-set trends: restrict controls to same set as switcher
            if set_ids is not None:
                ctrl_mask &= set_ids[ctrl_indices] == set_ids[g]
            ctrl_pool = ctrl_indices[ctrl_mask]

            if ctrl_pool.size == 0:
                # No observed controls at this horizon (may be terminal
                # missingness, not a true A11 violation). Exclude the
                # group from N_l rather than zero-retaining, so the
                # missing-data case doesn't bias DID_l toward zero.
                eligible[g] = False
                a11_multi_warnings.append(
                    f"horizon {l}, group_idx {g}: "
                    f"no baseline-matched controls at outcome period"
                )
                continue

            ctrl_changes = Y_mat[ctrl_pool, out_idx] - Y_mat[ctrl_pool, ref_idx]
            ctrl_avg = float(ctrl_changes.mean())
            did_g_l[g] = switcher_change - ctrl_avg

        # Recompute N_l after control-pool exclusions
        N_l = int(eligible.sum())
        if l == 1:
            N_1 = N_l
        if N_l == 0:
            results[l] = {
                "did_l": float("nan"),
                "N_l": 0,
                "did_g_l": did_g_l,
                "eligible_mask": eligible,
                "switcher_fraction": float("nan"),
            }
            continue

        # Aggregate: DID_l = (1/N_l) * sum S_g * DID_{g,l}
        S_eligible = switch_direction[eligible].astype(float)
        did_g_eligible = did_g_l[eligible]
        did_l = float((S_eligible * did_g_eligible).sum() / N_l)

        results[l] = {
            "did_l": did_l,
            "N_l": N_l,
            "did_g_l": did_g_l,
            "eligible_mask": eligible,
            "switcher_fraction": N_l / N_1 if N_1 > 0 else float("nan"),
        }

    # Attach A11 warnings to the results for the caller to surface
    if a11_multi_warnings:
        results["_a11_warnings"] = a11_multi_warnings  # type: ignore[assignment]

    return results


def _compute_per_group_if_multi_horizon(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    T_g: np.ndarray,
    L_max: int,
    set_ids: Optional[np.ndarray] = None,
) -> Dict[int, np.ndarray]:
    """
    Compute per-group influence function ``U^G_{g,l}`` for ``l = 1..L_max``.

    Each group g contributes to ``DID_l`` in two capacities:

    1. **As a switcher** (if g is eligible at horizon l): contributes
       ``S_g * (Y_{g, F_g-1+l} - Y_{g, F_g-1})`` to the numerator.
    2. **As a control** (if g serves as a not-yet-switched control for
       some other switcher g'): contributes
       ``-S_{g'} * (1/N^{g'}_{out}) * (Y_{g, out} - Y_{g, ref})``
       where ref/out are g's reference/outcome periods.

    The result satisfies ``sum(U_l) == N_l * DID_l``, which is verified
    as a sanity check.

    Parameters
    ----------
    D_mat, Y_mat, N_mat : np.ndarray of shape (n_groups, n_periods)
    baselines, first_switch_idx, switch_direction, T_g : np.ndarray
        From ``_compute_group_switch_metadata()``.
    L_max : int

    Returns
    -------
    dict mapping horizon l -> U_g_l array of shape (n_groups,)
        NOT cohort-centered. The caller applies ``_cohort_recenter()``
        before computing SE.
    """
    n_groups, n_periods = D_mat.shape
    is_switcher = first_switch_idx >= 0

    # Pre-compute per-baseline group indices for control-pool lookup.
    unique_baselines = np.unique(baselines)
    baseline_groups: Dict[float, np.ndarray] = {}
    baseline_f: Dict[float, np.ndarray] = {}
    for d in unique_baselines:
        mask = baselines == d
        baseline_groups[float(d)] = np.where(mask)[0]
        baseline_f[float(d)] = first_switch_idx[mask]

    results: Dict[int, np.ndarray] = {}

    for l in range(1, L_max + 1):  # noqa: E741
        U_l = np.zeros(n_groups, dtype=float)

        for g in range(n_groups):
            if not is_switcher[g]:
                continue
            f_g = first_switch_idx[g]
            ref_idx = f_g - 1
            out_idx = f_g - 1 + l
            if ref_idx < 0 or out_idx >= n_periods:
                continue
            if N_mat[g, ref_idx] <= 0 or N_mat[g, out_idx] <= 0:
                continue
            if T_g[g] < out_idx:
                continue

            d_base = float(baselines[g])
            S_g = float(switch_direction[g])

            # Control pool for this switcher at this horizon
            ctrl_indices = baseline_groups[d_base]
            ctrl_f = baseline_f[d_base]
            ctrl_mask = (
                ((ctrl_f > out_idx) | (ctrl_f == -1))
                & (N_mat[ctrl_indices, ref_idx] > 0)
                & (N_mat[ctrl_indices, out_idx] > 0)
            )
            # State-set trends: restrict controls to same set as switcher
            if set_ids is not None:
                ctrl_mask &= set_ids[ctrl_indices] == set_ids[g]
            ctrl_pool = ctrl_indices[ctrl_mask]
            n_ctrl = ctrl_pool.size

            if n_ctrl == 0:
                # No controls: A11-like, DID_{g,l} = 0. The switcher's
                # contribution to U_l is zero, but its count is in N_l.
                continue

            # Switcher contribution: +S_g * (Y_{g, out} - Y_{g, ref})
            switcher_change = Y_mat[g, out_idx] - Y_mat[g, ref_idx]
            U_l[g] += S_g * switcher_change

            # Control contributions: each control g' in the pool gets
            # -S_g * (1/n_ctrl) * (Y_{g', out} - Y_{g', ref})
            ctrl_changes = Y_mat[ctrl_pool, out_idx] - Y_mat[ctrl_pool, ref_idx]
            U_l[ctrl_pool] -= (S_g / n_ctrl) * ctrl_changes

        results[l] = U_l

    return results


def _compute_per_group_if_placebo_horizon(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    T_g: np.ndarray,
    L_max: int,
    set_ids: Optional[np.ndarray] = None,
) -> Dict[int, np.ndarray]:
    """
    Compute per-group influence function for placebo horizons.

    Mirrors ``_compute_per_group_if_multi_horizon`` but for backward
    horizons, matching ``_compute_multi_horizon_placebos`` eligibility
    and control-pool logic exactly.

    For placebo lag ``l``, switcher ``g``'s contribution uses the
    backward outcome change ``Y_{g, F_g-1-l} - Y_{g, F_g-1}`` (paper
    convention: pre-period minus reference). Controls are identified
    by the **positive**-horizon cutoff ``F_{g'} > F_g - 1 + l`` AND
    observation at ``ref_idx``, ``backward_idx``, AND ``forward_idx``
    (the terminal-missingness guard from Phase 2 Round 9).

    Returns
    -------
    dict mapping lag l (positive int) -> U_pl_l array of shape (n_groups,)
        NOT cohort-centered. The caller applies ``_cohort_recenter()``
        before computing SE.
    """
    n_groups, n_periods = D_mat.shape
    is_switcher = first_switch_idx >= 0

    unique_baselines = np.unique(baselines)
    baseline_groups: Dict[float, np.ndarray] = {}
    baseline_f: Dict[float, np.ndarray] = {}
    for d in unique_baselines:
        mask = baselines == d
        baseline_groups[float(d)] = np.where(mask)[0]
        baseline_f[float(d)] = first_switch_idx[mask]

    results: Dict[int, np.ndarray] = {}

    for l in range(1, L_max + 1):  # noqa: E741
        U_pl = np.zeros(n_groups, dtype=float)

        for g in range(n_groups):
            if not is_switcher[g]:
                continue
            f_g = first_switch_idx[g]
            ref_idx = f_g - 1
            backward_idx = ref_idx - l
            forward_idx = ref_idx + l

            # Dual eligibility (matches _compute_multi_horizon_placebos)
            if backward_idx < 0 or forward_idx >= n_periods:
                continue
            if N_mat[g, ref_idx] <= 0 or N_mat[g, backward_idx] <= 0:
                continue
            if T_g[g] < forward_idx:
                continue

            d_base = float(baselines[g])
            S_g = float(switch_direction[g])

            # Control pool: same baseline, not switched by forward_idx,
            # observed at ref, backward, AND forward (terminal-missingness
            # guard). Matches _compute_multi_horizon_placebos exactly.
            ctrl_indices = baseline_groups[d_base]
            ctrl_f = baseline_f[d_base]
            ctrl_mask = (
                ((ctrl_f > forward_idx) | (ctrl_f == -1))
                & (N_mat[ctrl_indices, ref_idx] > 0)
                & (N_mat[ctrl_indices, backward_idx] > 0)
                & (N_mat[ctrl_indices, forward_idx] > 0)
            )
            # State-set trends: restrict controls to same set
            if set_ids is not None:
                ctrl_mask &= set_ids[ctrl_indices] == set_ids[g]
            ctrl_pool = ctrl_indices[ctrl_mask]
            n_ctrl = ctrl_pool.size

            if n_ctrl == 0:
                continue

            # Switcher contribution: paper convention backward - ref
            switcher_change = Y_mat[g, backward_idx] - Y_mat[g, ref_idx]
            U_pl[g] += S_g * switcher_change

            # Control contributions
            ctrl_changes = Y_mat[ctrl_pool, backward_idx] - Y_mat[ctrl_pool, ref_idx]
            U_pl[ctrl_pool] -= (S_g / n_ctrl) * ctrl_changes

        results[l] = U_pl

    return results


def _compute_multi_horizon_placebos(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    T_g: np.ndarray,
    L_max: int,
    set_ids: Optional[np.ndarray] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute dynamic placebo estimators ``DID^{pl}_l`` for ``l = 1..L_pl_max``.

    Mirrors ``_compute_multi_horizon_dids`` but looks BACKWARD from
    each group's reference period (Web Appendix Section 1.1, Lemma 5).

    **Dual eligibility condition:** a group g is eligible for placebo
    lag l iff:
    - ``F_g - 1 - l >= 0`` (enough pre-treatment history), AND
    - ``F_g - 1 + l <= T_g`` (positive-horizon control pool exists)

    The control set uses the *positive*-horizon cutoff:
    ``{g': D_{g',1} = D_{g,1}, F_{g'} > F_g - 1 + l}``.

    Returns
    -------
    dict mapping lag l (positive int) -> {
        "placebo_l": float,
        "N_pl_l": int,
        "eligible_mask": np.ndarray,
    }
    """
    n_groups, n_periods = D_mat.shape
    is_switcher = first_switch_idx >= 0

    unique_baselines = np.unique(baselines)
    baseline_groups: Dict[float, np.ndarray] = {}
    baseline_f: Dict[float, np.ndarray] = {}
    for d in unique_baselines:
        mask = baselines == d
        baseline_groups[float(d)] = np.where(mask)[0]
        baseline_f[float(d)] = first_switch_idx[mask]

    results: Dict[int, Dict[str, Any]] = {}
    a11_placebo_warnings: List[str] = []

    for l in range(1, L_max + 1):  # noqa: E741
        eligible = np.zeros(n_groups, dtype=bool)
        pl_g_l = np.full(n_groups, np.nan)

        for g in range(n_groups):
            if not is_switcher[g]:
                continue
            f_g = first_switch_idx[g]
            ref_idx = f_g - 1
            backward_idx = ref_idx - l  # the pre-treatment outcome period
            forward_idx = ref_idx + l  # for control-pool eligibility

            # Dual eligibility: backward must be in range, forward must
            # have controls available
            if backward_idx < 0 or forward_idx >= n_periods:
                continue
            if N_mat[g, ref_idx] <= 0 or N_mat[g, backward_idx] <= 0:
                continue
            if T_g[g] < forward_idx:
                continue
            eligible[g] = True

        N_pl_l = int(eligible.sum())
        if N_pl_l == 0:
            results[l] = {
                "placebo_l": float("nan"),
                "N_pl_l": 0,
                "eligible_mask": eligible,
            }
            continue

        for g in np.where(eligible)[0]:
            f_g = first_switch_idx[g]
            ref_idx = f_g - 1
            backward_idx = ref_idx - l
            forward_idx = ref_idx + l
            d_base = float(baselines[g])

            # Switcher's backward outcome change: pre-period minus reference
            # (paper convention: Y_{F_g-1-l} - Y_{F_g-1})
            switcher_change = Y_mat[g, backward_idx] - Y_mat[g, ref_idx]

            # Control pool: same baseline, not switched by forward_idx,
            # AND observed at all three relevant periods (ref, backward,
            # AND forward - the last ensures terminally missing controls
            # don't leak into the placebo computation).
            ctrl_indices = baseline_groups[d_base]
            ctrl_f = baseline_f[d_base]
            ctrl_mask = (
                ((ctrl_f > forward_idx) | (ctrl_f == -1))
                & (N_mat[ctrl_indices, ref_idx] > 0)
                & (N_mat[ctrl_indices, backward_idx] > 0)
                & (N_mat[ctrl_indices, forward_idx] > 0)
            )
            # State-set trends: restrict controls to same set
            if set_ids is not None:
                ctrl_mask &= set_ids[ctrl_indices] == set_ids[g]
            ctrl_pool = ctrl_indices[ctrl_mask]

            if ctrl_pool.size == 0:
                eligible[g] = False
                a11_placebo_warnings.append(f"placebo lag {l}, group_idx {g}: no controls")
                continue

            ctrl_changes = Y_mat[ctrl_pool, backward_idx] - Y_mat[ctrl_pool, ref_idx]
            ctrl_avg = float(ctrl_changes.mean())
            pl_g_l[g] = switcher_change - ctrl_avg

        # Recompute N_pl_l after control-pool exclusions
        N_pl_l = int(eligible.sum())
        if N_pl_l == 0:
            results[l] = {
                "placebo_l": float("nan"),
                "N_pl_l": 0,
                "eligible_mask": eligible,
            }
            continue

        S_eligible = switch_direction[eligible].astype(float)
        pl_g_eligible = pl_g_l[eligible]
        placebo_l = float((S_eligible * pl_g_eligible).sum() / N_pl_l)

        results[l] = {
            "placebo_l": placebo_l,
            "N_pl_l": N_pl_l,
            "eligible_mask": eligible,
        }

    if a11_placebo_warnings:
        results["_a11_warnings"] = a11_placebo_warnings  # type: ignore[assignment]

    return results


def _compute_normalized_effects(
    multi_horizon_dids: Dict[int, Dict[str, Any]],
    D_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    L_max: int,
) -> Dict[int, Dict[str, Any]]:
    """
    Compute normalized event-study effects ``DID^n_l = DID_l / delta^D_l``.

    Uses the general formula (Eq 15) that works for both binary and
    non-binary treatment (future-proofing for Phase 3).

    For binary treatment: ``delta^D_{g,l} = l`` (joiners) or ``-l``
    (leavers), so ``|delta^D_{g,l}| = l`` and ``DID^n_l = DID_l / l``.

    Returns
    -------
    dict mapping l -> {effect, denominator}
    """
    n_groups = D_mat.shape[0]
    results: Dict[int, Dict[str, Any]] = {}

    for l in range(1, L_max + 1):  # noqa: E741
        h = multi_horizon_dids.get(l)
        if h is None or h["N_l"] == 0:
            results[l] = {"effect": float("nan"), "denominator": float("nan")}
            continue

        eligible = h["eligible_mask"]
        N_l = h["N_l"]
        did_l = h["did_l"]

        # Per-group incremental dose: delta^D_{g,l} = sum_{k=0}^{l-1} (D_{g,F_g+k} - D_{g,1})
        # General formula, works for non-binary treatment.
        delta_D_g = np.zeros(n_groups)
        for g in np.where(eligible)[0]:
            f_g = first_switch_idx[g]
            d_base = baselines[g]
            dose_sum = 0.0
            for k in range(l):
                col = f_g + k
                if col < D_mat.shape[1]:
                    dose_sum += D_mat[g, col] - d_base
            delta_D_g[g] = dose_sum

        # Aggregate dose denominator
        delta_D_l = float(np.abs(delta_D_g[eligible]).sum() / N_l)

        if delta_D_l <= 0:
            results[l] = {"effect": float("nan"), "denominator": 0.0}
            continue

        results[l] = {
            "effect": did_l / delta_D_l,
            "denominator": delta_D_l,
        }

    return results


def _compute_cost_benefit_delta(
    multi_horizon_dids: Dict[int, Dict[str, Any]],
    D_mat: np.ndarray,
    baselines: np.ndarray,
    first_switch_idx: np.ndarray,
    switch_direction: np.ndarray,
    L_max: int,
) -> Dict[str, Any]:
    """
    Compute the cost-benefit aggregate ``delta`` from Section 3.3, Lemma 4.

    ``delta = sum_l w_l * DID_l`` where
    ``w_l = N_l / sum_{g,l'} |D_{g,F_g-1+l'} - D_{g,1}|``.

    When leavers are present (Assumption 7 violated), also computes
    ``delta_joiners`` and ``delta_leavers`` separately.

    Returns
    -------
    dict with keys: delta, weights, has_leavers, delta_joiners, delta_leavers
    """

    # Per-horizon dose via Lemma 4: w_l uses the PER-PERIOD dose
    # D_{g,F_g-1+l} - D_{g,1} (NOT the cumulative delta^D_{g,l}).
    # For binary joiners this is 1 per (g,l) pair, so w_l = N_l / sum N_l'.
    total_dose = 0.0
    per_horizon_dose: Dict[int, float] = {}
    for l in range(1, L_max + 1):  # noqa: E741
        h = multi_horizon_dids.get(l)
        if h is None or h["N_l"] == 0:
            per_horizon_dose[l] = 0.0
            continue
        eligible = h["eligible_mask"]
        dose_l = 0.0
        for g in np.where(eligible)[0]:
            f_g = first_switch_idx[g]
            col = f_g - 1 + l
            if col < D_mat.shape[1]:
                dose_l += abs(float(D_mat[g, col] - baselines[g]))
        per_horizon_dose[l] = dose_l
        total_dose += dose_l

    if total_dose <= 0:
        return {
            "delta": float("nan"),
            "weights": {},
            "has_leavers": False,
            "delta_joiners": float("nan"),
            "delta_leavers": float("nan"),
        }

    # Horizon weights: w_l = N_l / total_dose (but using dose, not N_l)
    # Per Lemma 4: w_l = N_l * E[|delta^D_{g,l}|] / total_dose
    # which simplifies to per_horizon_dose[l] / total_dose
    weights: Dict[int, float] = {}
    delta = 0.0
    for l in range(1, L_max + 1):  # noqa: E741
        h = multi_horizon_dids.get(l)
        if h is None or h["N_l"] == 0:
            weights[l] = 0.0
            continue
        w_l = per_horizon_dose[l] / total_dose
        weights[l] = w_l
        delta += w_l * h["did_l"]

    # Check for leavers (Assumption 7 violation)
    has_leavers = bool(np.any(switch_direction < 0))

    delta_joiners = float("nan")
    delta_leavers = float("nan")
    if has_leavers:
        # Compute delta separately for joiners and leavers
        for direction, attr_name in [(1, "joiners"), (-1, "leavers")]:
            dir_dose = 0.0
            dir_horizon_dose: Dict[int, float] = {}
            for l in range(1, L_max + 1):  # noqa: E741
                h = multi_horizon_dids.get(l)
                if h is None or h["N_l"] == 0:
                    dir_horizon_dose[l] = 0.0
                    continue
                eligible = h["eligible_mask"]
                dose_l = 0.0
                for g in np.where(eligible)[0]:
                    if switch_direction[g] != direction:
                        continue
                    f_g = first_switch_idx[g]
                    col = f_g - 1 + l
                    if col < D_mat.shape[1]:
                        dose_l += abs(float(D_mat[g, col] - baselines[g]))
                dir_horizon_dose[l] = dose_l
                dir_dose += dose_l

            if dir_dose > 0:
                dir_delta = 0.0
                for l in range(1, L_max + 1):  # noqa: E741
                    h = multi_horizon_dids.get(l)
                    if h is None or h["N_l"] == 0:
                        continue
                    eligible = h["eligible_mask"]
                    # Per-direction DID_l
                    dir_eligible = eligible & (switch_direction == direction)
                    n_dir = int(dir_eligible.sum())
                    if n_dir == 0:
                        continue
                    did_g_l = h["did_g_l"]
                    S = switch_direction[dir_eligible].astype(float)
                    did_l_dir = float((S * did_g_l[dir_eligible]).sum() / n_dir)
                    w_dir = dir_horizon_dose[l] / dir_dose
                    dir_delta += w_dir * did_l_dir
                if attr_name == "joiners":
                    delta_joiners = dir_delta
                else:
                    delta_leavers = dir_delta

    return {
        "delta": delta,
        "weights": weights,
        "has_leavers": has_leavers,
        "delta_joiners": delta_joiners,
        "delta_leavers": delta_leavers,
    }


def _compute_full_per_group_contributions(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    n_10_t_arr: np.ndarray,
    n_00_t_arr: np.ndarray,
    n_01_t_arr: np.ndarray,
    n_11_t_arr: np.ndarray,
    a11_plus_zeroed_arr: np.ndarray,
    a11_minus_zeroed_arr: np.ndarray,
    side: str = "overall",
) -> np.ndarray:
    """
    Compute the per-group influence function ``U^G_g`` for ``DID_M``,
    ``DID_+``, or ``DID_-`` by summing role-weighted outcome differences
    across all periods (full ``Lambda^G_{g,l=1}`` from Section 3.7.2 of
    the dynamic companion paper, evaluated at horizon ``l = 1``).

    Decomposition (for ``side='overall'``)::

        N_S * DID_M = sum_t [
              sum_{g in joiners(t)}  (Y_{g,t} - Y_{g,t-1})
            - (n_10_t / n_00_t) * sum_{g in stable_0(t)} (Y_{g,t} - Y_{g,t-1})
            + (n_01_t / n_11_t) * sum_{g in stable_1(t)} (Y_{g,t} - Y_{g,t-1})
            - sum_{g in leavers(t)}  (Y_{g,t} - Y_{g,t-1})
        ]

    Each ``(g, t)`` cell contributes to ``U^G_g`` once per period, with
    the role weight determined by its ``(D_{g,t-1}, D_{g,t})`` transition.
    A switching group typically contributes from MULTIPLE periods (its
    own switch period + every period where it serves as a stable
    control); a never-switching group contributes only via its stable-
    control roles (which can be non-zero when it serves as a control
    for other cohorts' switches).

    Periods where ``DID_+,t`` or ``DID_-,t`` were zeroed under the A11
    convention contribute zero on the affected side, matching the
    point estimate.

    Parameters
    ----------
    D_mat, Y_mat, N_mat : np.ndarray of shape (n_groups, n_periods)
        Pivoted treatment, outcome, and observation-count matrices.
    n_10_t_arr, n_00_t_arr, n_01_t_arr, n_11_t_arr : np.ndarray
        Per-period CELL counts aligned to ``periods[1:]``.
    a11_plus_zeroed_arr, a11_minus_zeroed_arr : np.ndarray of bool
        Per-period A11-zeroing flags aligned to ``periods[1:]``.
    side : {"overall", "joiners", "leavers"}
        Which contribution to compute:

        - ``"overall"``: returns ``U^G_g`` such that ``U.sum() == N_S * DID_M``
        - ``"joiners"``: returns ``U^G_g`` such that ``U.sum() == joiner_total * DID_+``
          (only the joiners + stable_0 terms)
        - ``"leavers"``: returns ``U^G_g`` such that ``U.sum() == leaver_total * DID_-``
          (only the leavers + stable_1 terms, with the leavers side's sign convention)

    Returns
    -------
    U : np.ndarray of shape (n_groups,)
        Per-group contributions. NOT cohort-centered; the caller is
        responsible for centering before computing the SE.
    """
    if side not in ("overall", "joiners", "leavers"):
        raise ValueError(f"side must be one of overall/joiners/leavers, got {side!r}")

    n_groups, n_periods = D_mat.shape
    U = np.zeros(n_groups, dtype=float)

    include_joiners_side = side in ("overall", "joiners")
    include_leavers_side = side in ("overall", "leavers")

    for t_idx in range(1, n_periods):
        d_curr = D_mat[:, t_idx]
        d_prev = D_mat[:, t_idx - 1]
        y_diff = Y_mat[:, t_idx] - Y_mat[:, t_idx - 1]
        n_curr = N_mat[:, t_idx]
        n_prev = N_mat[:, t_idx - 1]
        present = (n_curr > 0) & (n_prev > 0)

        joiner_mask = (d_prev == 0) & (d_curr == 1) & present
        stable0_mask = (d_prev == 0) & (d_curr == 0) & present
        leaver_mask = (d_prev == 1) & (d_curr == 0) & present
        stable1_mask = (d_prev == 1) & (d_curr == 1) & present

        n_10_t = int(n_10_t_arr[t_idx - 1])
        n_00_t = int(n_00_t_arr[t_idx - 1])
        n_01_t = int(n_01_t_arr[t_idx - 1])
        n_11_t = int(n_11_t_arr[t_idx - 1])

        # Joiners side (+y_diff for joiners; -(n_10/n_00)*y_diff for stable_0)
        if (
            include_joiners_side
            and not bool(a11_plus_zeroed_arr[t_idx - 1])
            and n_10_t > 0
            and n_00_t > 0
        ):
            U[joiner_mask] += y_diff[joiner_mask]
            U[stable0_mask] -= (n_10_t / n_00_t) * y_diff[stable0_mask]

        # Leavers side (-y_diff for leavers; +(n_01/n_11)*y_diff for stable_1)
        if (
            include_leavers_side
            and not bool(a11_minus_zeroed_arr[t_idx - 1])
            and n_01_t > 0
            and n_11_t > 0
        ):
            U[leaver_mask] -= y_diff[leaver_mask]
            U[stable1_mask] += (n_01_t / n_11_t) * y_diff[stable1_mask]

    return U


def _cohort_recenter(
    U: np.ndarray,
    cohort_ids: np.ndarray,
) -> np.ndarray:
    """
    Subtract cohort-conditional means from U.

    For each cohort id, computes ``U_bar_k = mean(U[cohort==k])`` and
    returns ``U - U_bar_{cohort(g)}``. This is the per-group cohort-
    recentering step from Web Appendix Section 3.7.3 of the dynamic
    companion paper. Critical: subtracts the cohort mean, NOT a single
    grand mean — using a grand mean silently produces a smaller,
    incorrect variance.
    """
    U_centered = U.astype(float).copy()
    if U.size == 0:
        return U_centered
    unique_cohorts = np.unique(cohort_ids)
    for k in unique_cohorts:
        in_cohort = cohort_ids == k
        if in_cohort.any():
            U_centered[in_cohort] = U[in_cohort] - U[in_cohort].mean()
    return U_centered


def _compute_cohort_recentered_inputs(
    D_mat: np.ndarray,
    Y_mat: np.ndarray,
    N_mat: np.ndarray,
    n_10_t_arr: np.ndarray,
    n_00_t_arr: np.ndarray,
    n_01_t_arr: np.ndarray,
    n_11_t_arr: np.ndarray,
    a11_plus_zeroed_arr: np.ndarray,
    a11_minus_zeroed_arr: np.ndarray,
    all_groups: List[Any],
    singleton_baseline_groups: List[Any],
) -> Tuple[np.ndarray, int, int, int, np.ndarray, np.ndarray]:
    """
    Compute the cohort-centered influence-function vectors for variance.

    Implements the full ``Lambda^G_{g,l=1}`` weight vector from
    Section 3.7.2 of the dynamic companion paper (NBER WP 29873) at
    horizon ``l = 1``: each group's per-period role weights (joiner,
    stable_0, leaver, stable_1) sum to a per-group ``U^G_g`` value
    that, summed across groups, recovers ``N_S * DID_M``.

    Cohorts are defined by the triple ``(D_{g,1}, F_g, S_g)`` where
    ``F_g`` is the first switch period and ``S_g`` is the switch
    direction (+1 joiner, -1 leaver, 0 never-switching). Never-
    switching groups form their own cohorts indexed by baseline only.

    Per footnote 15 of the dynamic paper (passed in via
    ``singleton_baseline_groups``), groups whose baseline ``D_{g,1}``
    value is unique in the post-drop panel have no cohort peer and are
    excluded from the variance computation only. They remain in the
    point-estimate sample as period-based stable controls (this
    matches Python's documented period-vs-cohort stable-control
    interpretation; the cell DataFrame entering ``_compute_per_period_dids``
    retains them).

    Returns
    -------
    U_centered_overall : np.ndarray
        Cohort-centered IF vector for ``DID_M`` over the variance-
        eligible groups (post-singleton-filter).
    n_groups_for_overall : int
        ``U_centered_overall.size`` for sanity-checking by the caller.
    n_cohorts : int
        Distinct cohorts in the variance-eligible group set.
    n_groups_dropped_never_switching : int
        Count of never-switching groups for results metadata. (They
        ARE included in the variance computation under the full IF
        formula because they can have non-zero contributions when
        serving as stable controls; this count is reported for
        backwards compatibility with the existing results dataclass
        field but no longer represents an actual exclusion.)
    U_centered_joiners : np.ndarray
        Cohort-centered IF vector for ``DID_+`` (joiners-only side).
    U_centered_leavers : np.ndarray
        Cohort-centered IF vector for ``DID_-`` (leavers-only side).
    """
    n_groups, n_periods = D_mat.shape

    if n_groups == 0:
        return (
            np.array([], dtype=float),
            0,
            0,
            0,
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    # Per-group switch metadata via the shared helper (factored out in
    # Phase 2 so both the cohort-recentered IF path and the multi-
    # horizon DID_{g,l} path share the same computation).
    baselines, first_switch_idx, switch_direction, _T_g = _compute_group_switch_metadata(
        D_mat, N_mat
    )

    n_groups_dropped_never_switching = int((switch_direction == 0).sum())

    # Variance-eligibility mask: include all groups EXCEPT singleton-
    # baseline groups (footnote 15) which have no cohort peer.
    singleton_baseline_set = set(singleton_baseline_groups)
    eligible_mask = np.array([g not in singleton_baseline_set for g in all_groups], dtype=bool)

    # Cohort identification: (D_{g,1}, F_g, S_g) triples for the
    # variance-eligible group set. Never-switching groups (S_g = 0)
    # have F_g = -1 and form cohorts indexed by baseline alone.
    cohort_keys = [
        (float(baselines[g]), int(first_switch_idx[g]), int(switch_direction[g]))
        for g in range(n_groups)
    ]
    unique_cohorts: Dict[Tuple[float, int, int], int] = {}
    cohort_id = np.zeros(n_groups, dtype=int)
    for g in range(n_groups):
        if not eligible_mask[g]:
            cohort_id[g] = -1
            continue
        key = cohort_keys[g]
        if key not in unique_cohorts:
            unique_cohorts[key] = len(unique_cohorts)
        cohort_id[g] = unique_cohorts[key]
    n_cohorts = len(unique_cohorts)

    # Compute the full IF vectors via the new helper
    U_overall_full = _compute_full_per_group_contributions(
        D_mat=D_mat,
        Y_mat=Y_mat,
        N_mat=N_mat,
        n_10_t_arr=n_10_t_arr,
        n_00_t_arr=n_00_t_arr,
        n_01_t_arr=n_01_t_arr,
        n_11_t_arr=n_11_t_arr,
        a11_plus_zeroed_arr=a11_plus_zeroed_arr,
        a11_minus_zeroed_arr=a11_minus_zeroed_arr,
        side="overall",
    )
    U_joiners_full = _compute_full_per_group_contributions(
        D_mat=D_mat,
        Y_mat=Y_mat,
        N_mat=N_mat,
        n_10_t_arr=n_10_t_arr,
        n_00_t_arr=n_00_t_arr,
        n_01_t_arr=n_01_t_arr,
        n_11_t_arr=n_11_t_arr,
        a11_plus_zeroed_arr=a11_plus_zeroed_arr,
        a11_minus_zeroed_arr=a11_minus_zeroed_arr,
        side="joiners",
    )
    U_leavers_full = _compute_full_per_group_contributions(
        D_mat=D_mat,
        Y_mat=Y_mat,
        N_mat=N_mat,
        n_10_t_arr=n_10_t_arr,
        n_00_t_arr=n_00_t_arr,
        n_01_t_arr=n_01_t_arr,
        n_11_t_arr=n_11_t_arr,
        a11_plus_zeroed_arr=a11_plus_zeroed_arr,
        a11_minus_zeroed_arr=a11_minus_zeroed_arr,
        side="leavers",
    )

    # Restrict to variance-eligible groups (drop singleton-baseline groups)
    U_overall = U_overall_full[eligible_mask]
    U_joiners = U_joiners_full[eligible_mask]
    U_leavers = U_leavers_full[eligible_mask]
    cohort_id_eligible = cohort_id[eligible_mask]

    # Cohort-recenter each IF vector
    U_centered_overall = _cohort_recenter(U_overall, cohort_id_eligible)
    U_centered_joiners = _cohort_recenter(U_joiners, cohort_id_eligible)
    U_centered_leavers = _cohort_recenter(U_leavers, cohort_id_eligible)

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

    Returns ``NaN`` in three degenerate cases:

    1. ``U_centered`` is empty (no variance-eligible groups).
    2. ``divisor <= 0`` (no switching cells in N_S).
    3. ``sum(U_centered**2) <= 0`` — every cohort is a singleton, so
       cohort recentering produces an identically-zero centered IF
       vector and the variance is unidentified. The caller should
       detect this case (NaN return + non-empty input) and emit a
       user-facing warning explaining the degenerate-cohort condition.
       Returning ``NaN`` rather than ``0.0`` prevents the silently
       implies-infinite-precision failure mode.
    """
    n = U_centered.size
    if n == 0 or divisor <= 0:
        return float("nan")
    sum_sq = float((U_centered**2).sum())
    if sum_sq <= 0:
        # Degenerate-cohort case: every cohort is a singleton, so
        # cohort recentering produces all zeros. The variance is
        # unidentified — return NaN rather than 0.0 so downstream
        # inference is NaN-consistent and the caller surfaces a
        # warning. See the **Note** in REGISTRY.md
        # ChaisemartinDHaultfoeuille.
        return float("nan")
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
    if cell.empty:
        raise ValueError(
            "Cannot compute TWFE diagnostic on an empty cell DataFrame. "
            "Provide a panel with at least 2 groups and 2 time periods."
        )
    groups = sorted(cell[group_col].unique().tolist())
    times = sorted(cell[time_col].unique().tolist())
    n = len(cell)
    n_groups = len(groups)
    n_times = len(times)
    if n_groups < 2 or n_times < 2:
        raise ValueError(
            f"TWFE diagnostic requires at least 2 groups and 2 time periods, "
            f"got {n_groups} group(s) and {n_times} period(s)."
        )

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
    3. Compute per-cell **contribution weights** (the Theorem 1
       decomposition object):
       ``cw_{g,t} = N_{g,t} * eps_{g,t} / sum_{treated} N * eps``
       These are exported in the ``weights`` column of the returned
       ``TWFEWeightsResult``.
    4. Count negative contribution weights among treated cells.
    5. Compute the plain TWFE coefficient as a separate regression of
       ``y_gt`` on the same FE plus the treatment indicator.
    6. Compute ``sigma_fe`` from the Corollary 1 **paper weights**
       (a distinct object from the contribution weights):
       ``w_paper = eps / sum_treated(s * eps)`` where
       ``s = N_{g,t} / N_1`` are observation shares. The paper weight
       is centered at 1 under observation-share weighting. Then:
       ``sigma_fe = |beta_fe| / sqrt(sum_treated(s * (w_paper - 1)^2))``
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
    #
    # The paper defines w_{g,t} = eps_{g,t} / E_treated[eps], which
    # is DIFFERENT from the contribution weights w_gt exported in the
    # weights DataFrame (contribution_weight = s * w_paper). The paper
    # weight has the property that sum(s * w_paper) = 1 (centered at
    # 1 under observation-share weighting). sigma_fe uses the paper
    # weight:
    #
    #   w_paper = eps / sum_treated(s * eps)
    #   sigma(w) = sqrt(sum_treated(s * (w_paper - 1)^2))
    #   sigma_fe = |beta_fe| / sigma(w)
    #
    # where s_{g,t} = N_{g,t} / N_1 are observation shares.
    eps_treated = eps[treated_mask]
    n_treated_arr = n_arr[treated_mask]
    n1 = float(n_treated_arr.sum())  # total treated observations
    if n1 > 0:
        shares = n_treated_arr / n1  # s_{g,t} = N_{g,t} / N_1
        denom_paper = float((shares * eps_treated).sum())
        if abs(denom_paper) > 0:
            w_paper = eps_treated / denom_paper  # paper's w_{g,t}
            # Weighted variance around 1 (the weighted mean of w_paper is 1 by construction)
            var_w = float((shares * (w_paper - 1.0) ** 2).sum())
        else:
            var_w = 0.0
    else:
        var_w = 0.0
    if var_w > 0 and np.isfinite(beta_fe):
        sigma_fe = float(abs(beta_fe) / np.sqrt(var_w))
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
    # Validation + cell aggregation via the same helper used by
    # ChaisemartinDHaultfoeuille.fit() — enforces NaN/binary/within-cell
    # rules from REGISTRY.md so the standalone diagnostic does not
    # silently mishandle malformed input.
    cell = _validate_and_aggregate_to_cells(
        data=data,
        outcome=outcome,
        group=group,
        time=time,
        treatment=treatment,
    )
    # TWFE diagnostic assumes binary treatment (d_arr == 1 for treated mask).
    if not set(cell["d_gt"].unique()).issubset({0.0, 1.0, 0, 1}):
        raise ValueError(
            "twowayfeweights() requires binary treatment {0, 1}. "
            "Non-binary treatment is supported by fit() with L_max >= 1 "
            "but the TWFE diagnostic (Theorem 1 of AER 2020) assumes "
            "binary treatment."
        )
    return _compute_twfe_diagnostic(
        cell=cell,
        group_col=group,
        time_col=time,
        rank_deficient_action=rank_deficient_action,
    )
