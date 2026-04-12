"""
Result containers for the de Chaisemartin-D'Haultfoeuille (dCDH) estimator.

This module contains ``ChaisemartinDHaultfoeuilleResults`` and
``DCDHBootstrapResults`` dataclasses produced by the
``ChaisemartinDHaultfoeuille`` (alias ``DCDH``) estimator. The dCDH
estimator is the only modern DiD estimator in the library that handles
non-absorbing (reversible) treatments. Phase 1 ships the contemporaneous-
switch case ``DID_M`` (= ``DID_1`` of the dynamic companion paper).

References
----------
- de Chaisemartin, C. & D'Haultfoeuille, X. (2020). Two-Way Fixed Effects
  Estimators with Heterogeneous Treatment Effects. *American Economic
  Review*, 110(9), 2964-2996.
- de Chaisemartin, C. & D'Haultfoeuille, X. (2022, revised 2023).
  Difference-in-Differences Estimators of Intertemporal Treatment Effects.
  NBER Working Paper 29873.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.results import _get_significance_stars

__all__ = [
    "ChaisemartinDHaultfoeuilleResults",
    "DCDHBootstrapResults",
]


@dataclass
class DCDHBootstrapResults:
    """
    Results from ChaisemartinDHaultfoeuille (dCDH) multiplier bootstrap inference.

    The bootstrap is a library extension beyond the dCDH papers, which
    propose only the analytical cohort-recentered plug-in variance from
    Web Appendix Section 3.7.3 of the dynamic companion paper. Provided
    for consistency with CallawaySantAnna / ImputationDiD / TwoStageDiD.

    Per-target SE / CI / p-value are populated for the three scalar
    dCDH estimands implemented in Phase 1: overall (``DID_M``), joiners
    (``DID_+``), and leavers (``DID_-``). When a target is not available
    in the underlying data (e.g., no leavers), the matching fields are
    ``None``.

    **Phase 1 placebo bootstrap is intentionally NOT computed.** The
    dynamic companion paper Section 3.7.3 derives the cohort-recentered
    analytical variance for ``DID_l`` only, not for the placebo
    ``DID_M^pl``. The ``placebo_se`` / ``placebo_ci`` / ``placebo_p_value``
    fields below ALWAYS remain ``None`` in Phase 1, even when
    ``n_bootstrap > 0``. Phase 2 will add multiplier-bootstrap support
    for the placebo via the dynamic paper's machinery.

    Attributes
    ----------
    n_bootstrap : int
        Number of bootstrap iterations.
    weight_type : str
        Type of bootstrap weights: ``"rademacher"``, ``"mammen"``, or
        ``"webb"``.
    alpha : float
        Significance level used for confidence intervals.
    overall_se : float
        Bootstrap standard error for ``DID_M``.
    overall_ci : tuple of float
        Bootstrap confidence interval for ``DID_M``.
    overall_p_value : float
        Bootstrap p-value for ``DID_M``.
    joiners_se : float, optional
        Bootstrap SE for joiners-only ``DID_+`` (``None`` if no joiners).
    joiners_ci : tuple of float, optional
        Bootstrap CI for joiners-only ``DID_+``.
    joiners_p_value : float, optional
        Bootstrap p-value for joiners-only ``DID_+``.
    leavers_se : float, optional
        Bootstrap SE for leavers-only ``DID_-`` (``None`` if no leavers).
    leavers_ci : tuple of float, optional
        Bootstrap CI for leavers-only ``DID_-``.
    leavers_p_value : float, optional
        Bootstrap p-value for leavers-only ``DID_-``.
    placebo_se : float, optional
        **Always ``None`` in Phase 1** — placebo bootstrap is deferred
        to Phase 2 (see class docstring above).
    placebo_ci : tuple of float, optional
        **Always ``None`` in Phase 1** (see class docstring above).
    placebo_p_value : float, optional
        **Always ``None`` in Phase 1** (see class docstring above).
    bootstrap_distribution : np.ndarray, optional
        Full bootstrap distribution of the overall ``DID_M`` estimator
        (shape: ``(n_bootstrap,)``). Stored for advanced diagnostics;
        suppressed from ``__repr__``.
    """

    n_bootstrap: int
    weight_type: str
    alpha: float
    overall_se: float
    overall_ci: Tuple[float, float]
    overall_p_value: float
    joiners_se: Optional[float] = None
    joiners_ci: Optional[Tuple[float, float]] = None
    joiners_p_value: Optional[float] = None
    leavers_se: Optional[float] = None
    leavers_ci: Optional[Tuple[float, float]] = None
    leavers_p_value: Optional[float] = None
    placebo_se: Optional[float] = None
    placebo_ci: Optional[Tuple[float, float]] = None
    placebo_p_value: Optional[float] = None
    bootstrap_distribution: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class ChaisemartinDHaultfoeuilleResults:
    """
    Results from de Chaisemartin-D'Haultfoeuille (dCDH) Phase 1 estimation.

    Phase 1 ships the contemporaneous-switch estimator ``DID_M`` (= ``DID_1``
    at horizon ``l = 1`` of the dynamic companion paper) plus the joiners-
    only / leavers-only views, the single-lag placebo ``DID_M^pl``, and
    optionally the TWFE decomposition diagnostic (per-cell weights,
    fraction negative, ``sigma_fe``).

    Notes
    -----
    The analytical confidence interval is **conservative** under
    Assumption 8 (independent groups) of the dynamic companion paper, and
    exact only under iid sampling. This is documented as a deliberate
    deviation from "default nominal coverage" in the methodology registry.

    For binary treatment in Phase 1, multi-switch groups (i.e., groups
    that switch treatment more than once) are dropped before estimation
    when ``drop_larger_lower=True`` (the default), matching the R
    ``DIDmultiplegtDYN`` reference. The number of dropped groups is
    exposed via ``n_groups_dropped_crossers``.

    **Inference-method switch when bootstrap is enabled.** The
    ``overall_p_value`` / ``overall_conf_int`` (and joiners/leavers
    analogues) fields are populated by *normal-theory* inference from
    the cohort-recentered analytical SE when ``n_bootstrap=0`` (the
    default). When ``n_bootstrap > 0``, the same fields are populated
    by *percentile-based bootstrap inference* from the multiplier
    bootstrap distribution computed by ``_compute_dcdh_bootstrap()``.
    The t-stat (``overall_t_stat``, etc.) is computed from the SE in
    both cases, since percentile bootstrap does not define an
    alternative t-stat semantic. ``event_study_effects[1]``,
    ``summary()``, ``to_dataframe()``, ``is_significant``, and
    ``significance_stars`` all read from these top-level fields and
    therefore reflect the bootstrap inference automatically. The
    placebo path is unchanged: placebo bootstrap is deferred to Phase
    2, so ``placebo_p_value`` and ``placebo_conf_int`` stay NaN even
    when ``n_bootstrap > 0``. See the methodology registry
    ``Note (bootstrap inference surface)`` for the full contract and
    library precedent.

    Attributes
    ----------
    overall_att : float
        ``DID_M = DID_1``: the contemporaneous-switch dCDH point estimate.
    overall_se : float
        Standard error of ``DID_M``.
    overall_t_stat : float
    overall_p_value : float
    overall_conf_int : tuple of float
    joiners_att : float
        ``DID_+``: the joiners-only contribution. ``NaN`` when
        ``joiners_available`` is False.
    joiners_se : float
    joiners_t_stat : float
    joiners_p_value : float
    joiners_conf_int : tuple of float
    n_joiner_cells : int
        Total number of joiner switching ``(g, t)`` cells across all
        periods. Each cell counted once. Equals
        ``sum_t (#{g : D_{g,t-1}=0, D_{g,t}=1})``.
    n_joiner_obs : int
        Total raw observation count across joiner cells, summing
        ``n_gt`` over the same set of cells. For balanced
        one-observation-per-cell panels this equals ``n_joiner_cells``;
        for individual-level inputs with multiple observations per
        ``(g, t)`` it can be larger.
    joiners_available : bool
        ``True`` if at least one joiner switching cell exists.
    leavers_att : float
        ``DID_-``: the leavers-only contribution. ``NaN`` when
        ``leavers_available`` is False.
    leavers_se : float
    leavers_t_stat : float
    leavers_p_value : float
    leavers_conf_int : tuple of float
    n_leaver_cells : int
        Total number of leaver switching ``(g, t)`` cells (mirror of
        ``n_joiner_cells``).
    n_leaver_obs : int
        Total raw observation count across leaver cells (mirror of
        ``n_joiner_obs``).
    leavers_available : bool
    placebo_effect : float
        ``DID_M^pl``: the single-lag placebo. ``NaN`` when
        ``placebo_available`` is False.
    placebo_se : float
    placebo_t_stat : float
    placebo_p_value : float
    placebo_conf_int : tuple of float
    placebo_available : bool
        ``True`` when ``T >= 3`` and at least one qualifying placebo cell
        exists.
    per_period_effects : dict
        Per-period decomposition. Keys are period values; each value is a
        dict with the following keys:

        - ``"did_plus_t"`` (float): joiner effect at this period
          (``0.0`` if no joiners or A11 violation)
        - ``"did_minus_t"`` (float): leaver effect at this period
        - ``"n_10_t"`` (int): joiner cell count
        - ``"n_01_t"`` (int): leaver cell count
        - ``"n_00_t"`` (int): stable-untreated cell count
        - ``"n_11_t"`` (int): stable-treated cell count
        - ``"did_plus_t_a11_zeroed"`` (bool): True when joiners exist but
          no stable-untreated controls (Assumption 11 violation, period
          contributes 0 to numerator with non-zero weight in denominator)
        - ``"did_minus_t_a11_zeroed"`` (bool): mirror for leavers
    twfe_weights : pd.DataFrame, optional
        Per-cell TWFE decomposition weights from Theorem 1 of de
        Chaisemartin & D'Haultfoeuille (2020). Columns: ``group``,
        ``time``, ``weight``. Computed on the **FULL pre-filter cell
        sample** passed by the user (the same input the standalone
        :func:`twowayfeweights` function uses) — NOT the post-filter
        estimation sample described by ``overall_att`` and
        ``groups``. When ``fit()`` drops groups via the ragged-panel
        or ``drop_larger_lower`` filters, ``results.twfe_*`` and
        ``results.overall_att`` describe different samples and a
        ``UserWarning`` is emitted; see REGISTRY.md
        ``ChaisemartinDHaultfoeuille`` ``Note (TWFE diagnostic
        sample contract)`` for the rationale. Only populated when
        ``twfe_diagnostic=True``.
    twfe_fraction_negative : float, optional
        Fraction of treated-cell weights that are negative. ``> 0`` is
        the diagnostic for the heterogeneous-treatment-effect bias of
        the plain TWFE estimator on the **FULL pre-filter cell sample**
        (NOT the post-filter estimation sample). See the
        ``twfe_weights`` docstring above for the sample contract.
    twfe_sigma_fe : float, optional
        Smallest standard deviation of per-cell treatment effects that
        could flip the sign of the plain TWFE estimator (Corollary 1 of
        the AER 2020 paper). Computed on the **FULL pre-filter cell
        sample**.
    twfe_beta_fe : float, optional
        The plain TWFE coefficient computed on the **FULL pre-filter
        cell sample**, for comparison with ``overall_att``. Note that
        the two are computed on different samples when ``fit()``
        filters drop groups — see the ``twfe_weights`` docstring above
        for the sample contract.
    groups : list
        Group identifiers in the post-filter sample.
    time_periods : list
        Time periods in the panel.
    n_obs : int
        Total observations after filtering.
    n_treated_obs : int
        Treated observations in the post-filter sample.
    n_switcher_cells : int
        Number of switching ``(g, t)`` cells across periods. Equals
        ``sum_t (n_10_t + n_01_t)`` where each transition cell counts
        once regardless of how many original observations fed into it.
        This is the ``N_S`` denominator of ``DID_M`` per AER 2020
        Theorem 3 — cell counts, not within-cell observation counts.
    n_cohorts : int
        Distinct cohorts ``(D_{g,1}, F_g, S_g)`` after filtering.
    n_groups_dropped_crossers : int
        Number of groups dropped because they were multi-switch (matches
        R's ``drop_larger_lower=TRUE`` behavior). ``0`` when
        ``drop_larger_lower=False`` or no crossers exist.
    n_groups_dropped_singleton_baseline : int
        Number of groups whose baseline ``D_{g,1}`` is unique in the
        post-drop panel (footnote 15 of the dynamic paper). They are
        excluded from the cohort-recentered VARIANCE computation only —
        they remain in the point-estimate sample as period-based stable
        controls (see REGISTRY.md ``ChaisemartinDHaultfoeuille`` for the
        period-vs-cohort deviation that makes this distinction matter).
    n_groups_dropped_never_switching : int
        Number of groups with ``S_g = 0`` (never switched). **Reported
        for backwards compatibility only.** Per the Round 2 full
        influence-function fix, never-switching groups are NOT excluded
        from the variance: they contribute via their stable-control
        roles in the per-period IF formula. The field name retains
        "dropped" for API stability but no actual exclusion happens.
    alpha : float
        Significance level used for confidence intervals.
    event_study_effects : dict, optional
        In Phase 1 this is populated with a single entry for horizon
        ``1``, mirroring ``overall_att``. Keeping the field shape stable
        avoids API churn when Phase 2 adds entries for ``l = 2, ..., L``.
    normalized_effects : dict, optional
        Phase 2 placeholder (``DID^n_l``). Always ``None`` in Phase 1.
    cost_benefit_delta : dict, optional
        Phase 2 placeholder (cost-benefit aggregate ``delta``). Always
        ``None`` in Phase 1.
    sup_t_bands : dict, optional
        Phase 2 placeholder (sup-t simultaneous confidence bands).
    covariate_residuals : pd.DataFrame, optional
        Phase 3 placeholder (``DID^X`` residuals).
    linear_trends_effects : dict, optional
        Phase 3 placeholder (``DID^{fd}`` group-specific linear trends).
    honest_did_results : Any, optional
        Phase 3 placeholder (HonestDiD integration on placebos).
    survey_metadata : Any, optional
        Always ``None`` in Phase 1 — survey integration is deferred to a
        separate effort after all phases ship.
    bootstrap_results : DCDHBootstrapResults, optional
        Bootstrap inference results when ``n_bootstrap > 0``.
    """

    # --- Core: DID_M aggregate ---
    overall_att: float
    overall_se: float
    overall_t_stat: float
    overall_p_value: float
    overall_conf_int: Tuple[float, float]

    # --- Joiners-only view (DID_+) ---
    joiners_att: float
    joiners_se: float
    joiners_t_stat: float
    joiners_p_value: float
    joiners_conf_int: Tuple[float, float]
    n_joiner_cells: int
    n_joiner_obs: int
    joiners_available: bool

    # --- Leavers-only view (DID_-) ---
    leavers_att: float
    leavers_se: float
    leavers_t_stat: float
    leavers_p_value: float
    leavers_conf_int: Tuple[float, float]
    n_leaver_cells: int
    n_leaver_obs: int
    leavers_available: bool

    # --- Placebo (DID_M^pl) ---
    placebo_effect: float
    placebo_se: float
    placebo_t_stat: float
    placebo_p_value: float
    placebo_conf_int: Tuple[float, float]
    placebo_available: bool

    # --- Per-period decomposition ---
    per_period_effects: Dict[Any, Dict[str, Any]]

    # --- Metadata ---
    groups: List[Any]
    time_periods: List[Any]
    n_obs: int
    n_treated_obs: int
    n_switcher_cells: int
    n_cohorts: int
    n_groups_dropped_crossers: int
    n_groups_dropped_singleton_baseline: int
    n_groups_dropped_never_switching: int

    # --- Phase 1 event-study placeholder (populated with l=1 entry) ---
    # Stable shape across phases. In Phase 1, populated with a single
    # entry {1: {effect, se, t_stat, p_value, conf_int, n_obs}} mirroring
    # overall_att. Phase 2 extends with entries for l = 2, ..., L_max.
    event_study_effects: Optional[Dict[int, Dict[str, Any]]] = None

    # --- TWFE decomposition diagnostic (Theorem 1 of AER 2020) ---
    twfe_weights: Optional[pd.DataFrame] = field(default=None, repr=False)
    twfe_fraction_negative: Optional[float] = None
    twfe_sigma_fe: Optional[float] = None
    twfe_beta_fe: Optional[float] = None

    alpha: float = 0.05

    # --- Forward-compat placeholders (always None in Phase 1) ---
    normalized_effects: Optional[Dict[int, Dict[str, Any]]] = field(default=None, repr=False)
    cost_benefit_delta: Optional[Dict[str, Any]] = field(default=None, repr=False)
    sup_t_bands: Optional[Dict[str, Any]] = field(default=None, repr=False)
    covariate_residuals: Optional[pd.DataFrame] = field(default=None, repr=False)
    linear_trends_effects: Optional[Dict[int, Dict[str, Any]]] = field(default=None, repr=False)
    honest_did_results: Optional[Any] = field(default=None, repr=False)

    # --- Repr-suppressed metadata ---
    survey_metadata: Optional[Any] = field(default=None, repr=False)
    bootstrap_results: Optional[DCDHBootstrapResults] = field(default=None, repr=False)
    _estimator_ref: Optional[Any] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Repr / properties
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Concise string representation."""
        sig = _get_significance_stars(self.overall_p_value)
        return (
            f"ChaisemartinDHaultfoeuilleResults("
            f"DID_M={self.overall_att:.4f}{sig}, "
            f"SE={self.overall_se:.4f}, "
            f"n_groups={len(self.groups)}, "
            f"n_switcher_cells={self.n_switcher_cells})"
        )

    @property
    def coef_var(self) -> float:
        """SE / |DID_M|; NaN when DID_M is 0 or SE non-finite."""
        if not (np.isfinite(self.overall_se) and self.overall_se >= 0):
            return np.nan
        if not np.isfinite(self.overall_att) or self.overall_att == 0:
            return np.nan
        return self.overall_se / abs(self.overall_att)

    @property
    def is_significant(self) -> bool:
        """True iff overall ``DID_M`` p-value is below ``alpha``."""
        return bool(self.overall_p_value < self.alpha)

    @property
    def significance_stars(self) -> str:
        """Significance stars for the overall ``DID_M``."""
        return _get_significance_stars(self.overall_p_value)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate a formatted summary of dCDH estimation results.

        Parameters
        ----------
        alpha : float, optional
            Significance level for the confidence interval header. Defaults
            to ``self.alpha``.

        Returns
        -------
        str
            Formatted multi-block summary including overall ``DID_M``,
            joiners-only / leavers-only views, the placebo, the TWFE
            decomposition diagnostic, and a footer of significance codes.
        """
        alpha = alpha or self.alpha
        conf_level = int((1 - alpha) * 100)
        width = 85
        sep = "=" * width
        thin = "-" * width
        header_row = (
            f"{'Parameter':<15} {'Estimate':>12} {'Std. Err.':>12} "
            f"{'t-stat':>10} {'P>|t|':>10} {'Sig.':>6}"
        )

        lines = [
            sep,
            "de Chaisemartin-D'Haultfoeuille (dCDH) Estimator Results".center(width),
            sep,
            "",
            f"{'Total observations:':<35} {self.n_obs:>10}",
            f"{'Treated observations:':<35} {self.n_treated_obs:>10}",
            f"{'Switcher cells (N_S):':<35} {self.n_switcher_cells:>10}",
            f"{'Groups (post-filter):':<35} {len(self.groups):>10}",
            f"{'Cohorts:':<35} {self.n_cohorts:>10}",
            f"{'Time periods:':<35} {len(self.time_periods):>10}",
            "",
        ]

        # Filter counts (only show if any drops/exclusions happened).
        # After Round 2, never-switching groups participate in the variance
        # via stable-control roles and are NOT dropped — their count is
        # reported here for backwards compatibility only.
        if (
            self.n_groups_dropped_crossers
            + self.n_groups_dropped_singleton_baseline
            + self.n_groups_dropped_never_switching
            > 0
        ):
            lines.extend(
                [
                    "Group filter / metadata counts:",
                    f"{'  Multi-switch (dropped):':<42} " f"{self.n_groups_dropped_crossers:>10}",
                    f"{'  Singleton baseline (variance only):':<42} "
                    f"{self.n_groups_dropped_singleton_baseline:>10}",
                    f"{'  Never-switching (reported, not dropped):':<42} "
                    f"{self.n_groups_dropped_never_switching:>10}",
                    "",
                ]
            )

        # --- Overall DID_M ---
        lines.extend(
            [
                thin,
                "DID_M (Contemporaneous-Switch ATT)".center(width),
                thin,
                header_row,
                thin,
                _format_inference_row(
                    "DID_M",
                    self.overall_att,
                    self.overall_se,
                    self.overall_t_stat,
                    self.overall_p_value,
                ),
                thin,
                "",
                f"{conf_level}% Confidence Interval: "
                f"[{_fmt_float(self.overall_conf_int[0])}, "
                f"{_fmt_float(self.overall_conf_int[1])}]",
            ]
        )

        cv = self.coef_var
        if np.isfinite(cv):
            lines.append(f"{'CV (SE/|DID_M|):':<25} {cv:>10.4f}")

        lines.append("")
        lines.append(
            "Note: dCDH analytical CI is conservative under Assumption 8" " (independent groups);"
        )
        lines.append("      exact under iid sampling.")
        lines.append("")

        # --- Joiners and leavers ---
        lines.extend(
            [
                thin,
                "Decomposition: Joiners (DID_+) and Leavers (DID_-)".center(width),
                thin,
                header_row,
                thin,
            ]
        )

        if self.joiners_available:
            lines.append(
                _format_inference_row(
                    "DID_+",
                    self.joiners_att,
                    self.joiners_se,
                    self.joiners_t_stat,
                    self.joiners_p_value,
                )
            )
            lines.append(
                f"  ({self.n_joiner_cells} joiner cells, " f"{self.n_joiner_obs} observations)"
            )
        else:
            lines.append(
                f"{'DID_+':<15} {'(no joiners)':>12} " f"{'':>12} {'':>10} {'':>10} {'':>6}"
            )

        if self.leavers_available:
            lines.append(
                _format_inference_row(
                    "DID_-",
                    self.leavers_att,
                    self.leavers_se,
                    self.leavers_t_stat,
                    self.leavers_p_value,
                )
            )
            lines.append(
                f"  ({self.n_leaver_cells} leaver cells, " f"{self.n_leaver_obs} observations)"
            )
        else:
            lines.append(
                f"{'DID_-':<15} {'(no leavers)':>12} " f"{'':>12} {'':>10} {'':>10} {'':>6}"
            )

        lines.extend([thin, ""])

        # --- Placebo ---
        if self.placebo_available:
            lines.extend(
                [
                    thin,
                    "Single-Lag Placebo (DID_M^pl, Theorem 4)".center(width),
                    thin,
                    header_row,
                    thin,
                    _format_inference_row(
                        "DID_M^pl",
                        self.placebo_effect,
                        self.placebo_se,
                        self.placebo_t_stat,
                        self.placebo_p_value,
                    ),
                    thin,
                    "Under parallel trends, the placebo should be ~0.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    thin,
                    "Placebo not available (T < 3 or no qualifying cells)".center(width),
                    thin,
                    "",
                ]
            )

        # --- TWFE diagnostic ---
        if self.twfe_beta_fe is not None:
            lines.extend(
                [
                    thin,
                    "TWFE Decomposition Diagnostic (Theorem 1, AER 2020)".center(width),
                    thin,
                    f"{'Plain TWFE coefficient:':<35} {_fmt_float(self.twfe_beta_fe):>10}",
                ]
            )
            if self.twfe_fraction_negative is not None:
                lines.append(
                    f"{'Fraction of negative weights:':<35} "
                    f"{self.twfe_fraction_negative:>10.4f}"
                )
            if self.twfe_sigma_fe is not None and np.isfinite(self.twfe_sigma_fe):
                lines.append(
                    f"{'Sigma_fe (sign-flip threshold):':<35} " f"{self.twfe_sigma_fe:>10.4f}"
                )
            lines.extend(
                [
                    "",
                    "A positive fraction of negative weights signals that the plain",
                    "TWFE coefficient may have the wrong sign under heterogeneous",
                    "treatment effects. Sigma_fe is the smallest cell-level effect",
                    "standard deviation that could flip the sign of TWFE.",
                    thin,
                    "",
                ]
            )

        lines.extend(
            [
                "Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1",
                sep,
            ]
        )

        return "\n".join(lines)

    def print_summary(self, alpha: Optional[float] = None) -> None:
        """Print the formatted summary to stdout."""
        print(self.summary(alpha))

    # ------------------------------------------------------------------
    # to_dataframe
    # ------------------------------------------------------------------

    def to_dataframe(self, level: str = "overall") -> pd.DataFrame:
        """
        Convert results to a DataFrame at the requested level of aggregation.

        Parameters
        ----------
        level : str, default="overall"
            One of:

            - ``"overall"``: single-row table with the overall ``DID_M``
              point estimate, SE, t-stat, p-value, CI bounds.
            - ``"joiners_leavers"``: three rows for ``DID_M``, ``DID_+``,
              and ``DID_-``.
            - ``"per_period"``: one row per time period with
              ``did_plus_t``, ``did_minus_t``, switching cell counts, and
              the A11-zeroed flags.
            - ``"twfe_weights"``: per-(group, time) TWFE decomposition
              weights table. Only available when ``twfe_diagnostic=True``
              was passed to ``fit()``.

        Returns
        -------
        pd.DataFrame
        """
        if level == "overall":
            return pd.DataFrame(
                [
                    {
                        "estimand": "DID_M",
                        "effect": self.overall_att,
                        "se": self.overall_se,
                        "t_stat": self.overall_t_stat,
                        "p_value": self.overall_p_value,
                        "conf_int_lower": self.overall_conf_int[0],
                        "conf_int_upper": self.overall_conf_int[1],
                    }
                ]
            )

        elif level == "joiners_leavers":
            # Two separate count columns so each has consistent units
            # across all rows:
            #   n_cells: total switching cells (each (g, t) cell counted once)
            #   n_obs:   actual observation count summed over the same cells
            #            (equals n_cells on balanced 1-obs-per-cell panels;
            #            larger on individual-level inputs with multiple
            #            observations per cell).
            # For the DID_M row, both quantities use the overall switching
            # cell set: n_cells = sum of joiner + leaver cells, and n_obs
            # is the same sum of raw observation counts.
            rows = [
                {
                    "estimand": "DID_M",
                    "effect": self.overall_att,
                    "se": self.overall_se,
                    "t_stat": self.overall_t_stat,
                    "p_value": self.overall_p_value,
                    "conf_int_lower": self.overall_conf_int[0],
                    "conf_int_upper": self.overall_conf_int[1],
                    "n_cells": self.n_switcher_cells,
                    "n_obs": self.n_joiner_obs + self.n_leaver_obs,
                    "available": True,
                },
                {
                    "estimand": "DID_+",
                    "effect": self.joiners_att,
                    "se": self.joiners_se,
                    "t_stat": self.joiners_t_stat,
                    "p_value": self.joiners_p_value,
                    "conf_int_lower": self.joiners_conf_int[0],
                    "conf_int_upper": self.joiners_conf_int[1],
                    "n_cells": self.n_joiner_cells,
                    "n_obs": self.n_joiner_obs,
                    "available": self.joiners_available,
                },
                {
                    "estimand": "DID_-",
                    "effect": self.leavers_att,
                    "se": self.leavers_se,
                    "t_stat": self.leavers_t_stat,
                    "p_value": self.leavers_p_value,
                    "conf_int_lower": self.leavers_conf_int[0],
                    "conf_int_upper": self.leavers_conf_int[1],
                    "n_cells": self.n_leaver_cells,
                    "n_obs": self.n_leaver_obs,
                    "available": self.leavers_available,
                },
            ]
            return pd.DataFrame(rows)

        elif level == "per_period":
            if not self.per_period_effects:
                # Empty per-period table — return DataFrame with the
                # canonical column order so downstream code can rely on it.
                return pd.DataFrame(
                    {
                        "period": pd.Series(dtype="int64"),
                        "did_plus_t": pd.Series(dtype="float64"),
                        "did_minus_t": pd.Series(dtype="float64"),
                        "n_10_t": pd.Series(dtype="int64"),
                        "n_01_t": pd.Series(dtype="int64"),
                        "n_00_t": pd.Series(dtype="int64"),
                        "n_11_t": pd.Series(dtype="int64"),
                        "did_plus_t_a11_zeroed": pd.Series(dtype="bool"),
                        "did_minus_t_a11_zeroed": pd.Series(dtype="bool"),
                    }
                )
            rows = []
            for t in sorted(self.per_period_effects.keys()):
                cell = self.per_period_effects[t]
                rows.append({"period": t, **cell})
            return pd.DataFrame(rows)

        elif level == "twfe_weights":
            if self.twfe_weights is None:
                raise ValueError(
                    "TWFE decomposition weights not computed. Pass "
                    "twfe_diagnostic=True (the default) to ChaisemartinDHaultfoeuille()."
                )
            return self.twfe_weights.copy()

        else:
            raise ValueError(
                f"Unknown level: {level!r}. Use 'overall', 'joiners_leavers', "
                f"'per_period', or 'twfe_weights'."
            )


# =============================================================================
# Internal formatting helpers
# =============================================================================


def _fmt_float(x: float) -> str:
    """Format a float; render NaN/Inf as the string 'NaN'/'Inf'."""
    if not np.isfinite(x):
        return "NaN" if np.isnan(x) else ("Inf" if x > 0 else "-Inf")
    return f"{x:.4f}"


def _format_inference_row(
    label: str,
    effect: float,
    se: float,
    t_stat: float,
    p_value: float,
) -> str:
    """Format a single inference row for the summary table."""
    e_str = f"{_fmt_float(effect):>12}"
    s_str = f"{_fmt_float(se):>12}"
    t_str = f"{t_stat:>10.3f}" if np.isfinite(t_stat) else f"{'NaN':>10}"
    p_str = f"{p_value:>10.4f}" if np.isfinite(p_value) else f"{'NaN':>10}"
    sig = _get_significance_stars(p_value) if np.isfinite(p_value) else ""
    return f"{label:<15} {e_str} {s_str} {t_str} {p_str} {sig:>6}"
