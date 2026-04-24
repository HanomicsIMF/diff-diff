"""
Synthetic Difference-in-Differences estimator.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

from diff_diff.estimators import DifferenceInDifferences
from diff_diff.linalg import solve_ols
from diff_diff.results import SyntheticDiDResults, _SyntheticDiDFitSnapshot
from diff_diff.bootstrap_utils import generate_rao_wu_weights
from diff_diff.utils import (
    _compute_regularization,
    _sum_normalize,
    compute_sdid_estimator,
    compute_sdid_unit_weights,
    compute_sdid_unit_weights_survey,
    compute_time_weights,
    compute_time_weights_survey,
    safe_inference,
    validate_binary,
)


class SyntheticDiD(DifferenceInDifferences):
    """
    Synthetic Difference-in-Differences (SDID) estimator.

    Combines the strengths of Difference-in-Differences and Synthetic Control
    methods by re-weighting control units to better match treated units'
    pre-treatment trends.

    This method is particularly useful when:
    - You have few treated units (possibly just one)
    - Parallel trends assumption may be questionable
    - Control units are heterogeneous and need reweighting
    - You want robustness to pre-treatment differences

    Parameters
    ----------
    zeta_omega : float, optional
        Regularization for unit weights. If None (default), auto-computed
        from data as ``(N1 * T1)^(1/4) * noise_level`` matching R's synthdid.
    zeta_lambda : float, optional
        Regularization for time weights. If None (default), auto-computed
        from data as ``1e-6 * noise_level`` matching R's synthdid.
    alpha : float, default=0.05
        Significance level for confidence intervals.
    variance_method : str, default="placebo"
        Method for variance estimation:
        - "placebo": Placebo-based variance matching R's synthdid::vcov(method="placebo").
          Implements Algorithm 4 from Arkhangelsky et al. (2021). Library default
          (R's default is ``"bootstrap"``; we default to placebo because it is
          unconditionally available on pweight-only survey designs and avoids the
          ~5–30× slowdown of the refit bootstrap). See REGISTRY.md §SyntheticDiD
          ``Note (default variance_method deviation from R)`` for rationale.
        - "bootstrap": Paper-faithful pairs bootstrap — Arkhangelsky et al. (2021)
          Algorithm 2 step 2, also the behavior of R's default
          synthdid::vcov(method="bootstrap") (which rebinds ``attr(estimate, "opts")``
          with ``update.omega=TRUE``, so the renormalized ω is only Frank-Wolfe
          initialization). Re-estimates ω̂_b and λ̂_b via two-pass sparsified
          Frank-Wolfe on each bootstrap draw. **Survey support (PR #352):**
          pweight-only fits use the constant per-control survey weight as ``rw``;
          full-design fits (strata/PSU/FPC) use Rao-Wu rescaled weights per draw.
          Both compose with the **weighted Frank-Wolfe** kernel
          (``min ||A·diag(rw)·ω - b||² + ζ²·Σ rw_i ω_i²``); the FW returns ω on the
          standard simplex, then ``ω_eff = rw·ω/Σ(rw·ω)`` is composed for the SDID
          estimator. See REGISTRY.md §SyntheticDiD ``Note (survey + bootstrap
          composition)`` for the argmin-set caveat.
        - "jackknife": Jackknife variance matching R's synthdid::vcov(method="jackknife").
          Implements Algorithm 3 from Arkhangelsky et al. (2021). Deterministic
          (N_control + N_treated iterations), uses fixed weights (no re-estimation).
          The ``n_bootstrap`` parameter is ignored for this method.
    n_bootstrap : int, default=200
        Number of replications for variance estimation. Used for:
        - Bootstrap: Number of bootstrap samples
        - Placebo: Number of random permutations (matches R's `replications` argument)
        Ignored when ``variance_method="jackknife"``.
    seed : int, optional
        Random seed for reproducibility. If None (default), results
        will vary between runs.

    Attributes
    ----------
    results_ : SyntheticDiDResults
        Estimation results after calling fit().
    is_fitted_ : bool
        Whether the model has been fitted.

    Examples
    --------
    Basic usage with panel data:

    >>> import pandas as pd
    >>> from diff_diff import SyntheticDiD
    >>>
    >>> # Panel data with units observed over multiple time periods
    >>> # Treatment occurs at period 5 for treated units
    >>> data = pd.DataFrame({
    ...     'unit': [...],      # Unit identifier
    ...     'period': [...],    # Time period
    ...     'outcome': [...],   # Outcome variable
    ...     'treated': [...]    # 1 if unit is ever treated, 0 otherwise
    ... })
    >>>
    >>> # Fit SDID model
    >>> sdid = SyntheticDiD()
    >>> results = sdid.fit(
    ...     data,
    ...     outcome='outcome',
    ...     treatment='treated',
    ...     unit='unit',
    ...     time='period',
    ...     post_periods=[5, 6, 7, 8]
    ... )
    >>>
    >>> # View results
    >>> results.print_summary()
    >>> print(f"ATT: {results.att:.3f} (SE: {results.se:.3f})")
    >>>
    >>> # Examine unit weights
    >>> weights_df = results.get_unit_weights_df()
    >>> print(weights_df.head(10))

    Notes
    -----
    The SDID estimator (Arkhangelsky et al., 2021) computes:

        τ̂ = (Ȳ_treated,post - Σ_t λ_t * Y_treated,t)
            - Σ_j ω_j * (Ȳ_j,post - Σ_t λ_t * Y_j,t)

    Where:
    - ω_j are unit weights (sum to 1, non-negative)
    - λ_t are time weights (sum to 1, non-negative)

    Unit weights ω are chosen to match pre-treatment outcomes:
        min ||Σ_j ω_j * Y_j,pre - Y_treated,pre||²

    This interpolates between:
    - Standard DiD (uniform weights): ω_j = 1/N_control
    - Synthetic Control (exact matching): concentrated weights

    References
    ----------
    Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
    (2021). Synthetic Difference-in-Differences. American Economic Review,
    111(12), 4088-4118.
    """

    def __init__(
        self,
        zeta_omega: Optional[float] = None,
        zeta_lambda: Optional[float] = None,
        alpha: float = 0.05,
        variance_method: str = "placebo",
        n_bootstrap: int = 200,
        seed: Optional[int] = None,
        # Deprecated — accepted for backward compat, ignored with warning
        lambda_reg: Optional[float] = None,
        zeta: Optional[float] = None,
    ):
        if lambda_reg is not None:
            warnings.warn(
                "lambda_reg is deprecated and ignored. Regularization is now "
                "auto-computed from data. Use zeta_omega to override unit weight "
                "regularization. Will be removed in v4.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )
        if zeta is not None:
            warnings.warn(
                "zeta is deprecated and ignored. Use zeta_lambda to override "
                "time weight regularization. Will be removed in v4.0.0.",
                DeprecationWarning,
                stacklevel=2,
            )

        super().__init__(robust=True, cluster=None, alpha=alpha)
        self.zeta_omega = zeta_omega
        self.zeta_lambda = zeta_lambda
        self.variance_method = variance_method
        self.n_bootstrap = n_bootstrap
        self.seed = seed

        self._validate_config()

        self._unit_weights = None
        self._time_weights = None

    _VALID_VARIANCE_METHODS = ("bootstrap", "jackknife", "placebo")

    def _validate_config(self) -> None:
        """Validate ``variance_method`` and ``n_bootstrap`` on the current state.

        Called from both ``__init__`` and ``set_params`` so updates via the
        sklearn-style setter path enforce the same contract as construction.
        """
        if self.variance_method not in self._VALID_VARIANCE_METHODS:
            raise ValueError(
                f"variance_method must be one of {self._VALID_VARIANCE_METHODS}, "
                f"got '{self.variance_method}'"
            )
        if self.n_bootstrap < 2 and self.variance_method != "jackknife":
            raise ValueError(
                f"n_bootstrap must be >= 2 (got {self.n_bootstrap}). At least 2 "
                f"iterations are needed to estimate standard errors."
            )

    def fit(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        outcome: str,
        treatment: str,
        unit: str,
        time: str,
        post_periods: Optional[List[Any]] = None,
        covariates: Optional[List[str]] = None,
        survey_design=None,
    ) -> SyntheticDiDResults:
        """
        Fit the Synthetic Difference-in-Differences model.

        Parameters
        ----------
        data : pd.DataFrame
            Panel data with observations for multiple units over multiple
            time periods.
        outcome : str
            Name of the outcome variable column.
        treatment : str
            Name of the treatment group indicator column (0/1).
            Should be 1 for all observations of treated units
            (both pre and post treatment).
        unit : str
            Name of the unit identifier column.
        time : str
            Name of the time period column.
        post_periods : list, optional
            List of time period values that are post-treatment.
            If None, uses the last half of periods.
        covariates : list, optional
            List of covariate column names. Covariates are residualized
            out before computing the SDID estimator.
        survey_design : SurveyDesign, optional
            Survey design specification. Only pweight weight_type is
            supported. Support matrix (PR #352):

                method     pweight-only     strata/PSU/FPC
                bootstrap  ✓ weighted FW    ✓ weighted FW + Rao-Wu
                placebo    ✓                ✗ NotImplementedError
                jackknife  ✓                ✗ NotImplementedError

            The bootstrap path composes Rao-Wu rescaled weights per draw
            with the weighted-Frank-Wolfe kernel; see REGISTRY.md
            §SyntheticDiD ``Note (survey + bootstrap composition)``.
            ``placebo`` and ``jackknife`` still reject strata/PSU/FPC
            (separate methodology gap tracked in TODO.md).

        Returns
        -------
        SyntheticDiDResults
            Object containing the ATT estimate, standard error,
            unit weights, and time weights.

        Raises
        ------
        ValueError
            If required parameters are missing, data validation fails,
            or a non-pweight survey design is provided.
        NotImplementedError
            If ``survey_design`` with strata/PSU/FPC is provided with
            ``variance_method='placebo'`` or ``'jackknife'``. Bootstrap
            + any survey design (pweight-only or full design) is
            supported via PR #352's weighted-FW + Rao-Wu composition.
        """
        # Validate inputs
        if outcome is None or treatment is None or unit is None or time is None:
            raise ValueError("Must provide 'outcome', 'treatment', 'unit', and 'time'")

        # Check columns exist
        required_cols = [outcome, treatment, unit, time]
        if covariates:
            required_cols.extend(covariates)

        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # Resolve survey design
        from diff_diff.survey import (
            _extract_unit_survey_weights,
            _resolve_survey_for_fit,
            _validate_unit_constant_survey,
        )

        resolved_survey, survey_weights, survey_weight_type, survey_metadata = (
            _resolve_survey_for_fit(survey_design, data, "analytical")
        )
        # Reject replicate-weight designs — SyntheticDiD has no replicate-
        # weight variance path. Analytical (pweight / strata / PSU / FPC)
        # designs are supported per the PR #352 matrix (bootstrap covers
        # full design via weighted-FW + Rao-Wu; placebo / jackknife
        # accept pweight-only, reject strata/PSU/FPC).
        if resolved_survey is not None and resolved_survey.uses_replicate_variance:
            raise NotImplementedError(
                "SyntheticDiD does not support replicate-weight survey "
                "designs. Analytical survey designs are supported: "
                "variance_method='bootstrap' accepts both pweight-only "
                "and strata/PSU/FPC designs (PR #352), while "
                "variance_method='placebo' and 'jackknife' accept "
                "pweight-only. See docs/methodology/REGISTRY.md "
                "§SyntheticDiD for the full survey support matrix."
            )
        # Validate pweight only
        if resolved_survey is not None and resolved_survey.weight_type != "pweight":
            raise ValueError(
                "SyntheticDiD survey support requires weight_type='pweight'. "
                f"Got '{resolved_survey.weight_type}'."
            )

        # Strata/PSU/FPC support matrix (PR #352):
        #   bootstrap  → supported via weighted Frank-Wolfe + Rao-Wu rescaling
        #                (this PR; see _bootstrap_se Rao-Wu branch).
        #   placebo / jackknife → NotImplemented for full designs (separate
        #                methodology gap; resampling allocators differ between
        #                bootstrap pairs and placebo permutations / jackknife
        #                LOO). Tracked in TODO.md as a follow-up.
        if (
            resolved_survey is not None
            and (
                resolved_survey.strata is not None
                or resolved_survey.psu is not None
                or resolved_survey.fpc is not None
            )
            and self.variance_method in ("placebo", "jackknife")
        ):
            raise NotImplementedError(
                f"SyntheticDiD with variance_method='{self.variance_method}' "
                "does not yet support survey designs with strata/PSU/FPC. "
                "Pweight-only pseudo-population weights work with placebo / "
                "jackknife. Strata/PSU/FPC support requires per-method "
                "Rao-Wu / wild-bootstrap derivations on the placebo "
                "allocator and the jackknife LOO mass; tracked in TODO.md "
                "(SDID survey support follow-up). Use "
                "variance_method='bootstrap' for full survey designs."
            )

        # Validate treatment is binary
        validate_binary(data[treatment].values, "treatment")

        # Get all unique time periods
        all_periods = sorted(data[time].unique())

        if len(all_periods) < 2:
            raise ValueError("Need at least 2 time periods")

        # Determine pre and post periods
        if post_periods is None:
            mid = len(all_periods) // 2
            post_periods = list(all_periods[mid:])
            pre_periods = list(all_periods[:mid])
        else:
            post_periods = list(post_periods)
            pre_periods = [p for p in all_periods if p not in post_periods]

        if len(post_periods) == 0:
            raise ValueError("Must have at least one post-treatment period")
        if len(pre_periods) == 0:
            raise ValueError("Must have at least one pre-treatment period")

        # Validate post_periods are in data
        for p in post_periods:
            if p not in all_periods:
                raise ValueError(f"Post-period '{p}' not found in time column")

        # Identify treated and control units
        # Treatment indicator should be constant within unit
        unit_treatment = data.groupby(unit)[treatment].first()

        # Validate treatment is constant within unit (SDID requires block treatment)
        treatment_nunique = data.groupby(unit)[treatment].nunique()
        varying_units = treatment_nunique[treatment_nunique > 1]
        if len(varying_units) > 0:
            example_unit = varying_units.index[0]
            example_vals = sorted(data.loc[data[unit] == example_unit, treatment].unique())
            raise ValueError(
                f"Treatment indicator varies within {len(varying_units)} unit(s) "
                f"(e.g., unit '{example_unit}' has values {example_vals}). "
                f"SyntheticDiD requires 'block' treatment where treatment is "
                f"constant within each unit across all time periods. "
                f"For staggered adoption designs, use CallawaySantAnna or "
                f"ImputationDiD instead."
            )

        treated_units = unit_treatment[unit_treatment == 1].index.tolist()
        control_units = unit_treatment[unit_treatment == 0].index.tolist()

        if len(treated_units) == 0:
            raise ValueError("No treated units found")
        if len(control_units) == 0:
            raise ValueError("No control units found")

        # Validate balanced panel (SDID requires all units observed in all periods)
        periods_per_unit = data.groupby(unit)[time].nunique()
        expected_n_periods = len(all_periods)
        unbalanced_units = periods_per_unit[periods_per_unit != expected_n_periods]
        if len(unbalanced_units) > 0:
            example_unit = unbalanced_units.index[0]
            actual_count = unbalanced_units.iloc[0]
            raise ValueError(
                f"Panel is not balanced: {len(unbalanced_units)} unit(s) do not "
                f"have observations in all {expected_n_periods} periods "
                f"(e.g., unit '{example_unit}' has {actual_count} periods). "
                f"SyntheticDiD requires a balanced panel. Use "
                f"diff_diff.prep.balance_panel() to balance the panel first."
            )

        # Validate and extract survey weights. Pweight-only fits feed
        # placebo / jackknife / bootstrap via ``w_control`` directly.
        # Strata/PSU/FPC fits feed bootstrap via the unit-collapsed
        # ``resolved_survey_unit`` (PR #352) which Rao-Wu rescaling
        # consumes per draw.
        if resolved_survey is not None:
            _validate_unit_constant_survey(data, unit, survey_design)
            w_treated = _extract_unit_survey_weights(data, unit, survey_design, treated_units)
            w_control = _extract_unit_survey_weights(data, unit, survey_design, control_units)
            # Collapse to unit level for the bootstrap survey path. The
            # row order is [control_units..., treated_units...] so
            # boot_rw[:n_control] / boot_rw[n_control:] line up with the
            # bootstrap loop's column ordering. See
            # `collapse_survey_to_unit_level` in diff_diff/survey.py.
            from diff_diff.survey import collapse_survey_to_unit_level
            all_units_for_bootstrap = list(control_units) + list(treated_units)
            # Use `data` (not `working_data`) for the groupby — survey
            # design columns are unit-constant (validated above) and
            # covariate residualization doesn't shuffle row order, so the
            # collapse is invariant to which view we group on.
            resolved_survey_unit = collapse_survey_to_unit_level(
                resolved_survey, data, unit, all_units_for_bootstrap,
            )
        else:
            w_treated = None
            w_control = None
            resolved_survey_unit = None

        # Residualize covariates if provided
        working_data = data.copy()
        if covariates:
            working_data = self._residualize_covariates(
                working_data,
                outcome,
                covariates,
                unit,
                time,
                survey_weights=survey_weights,
                survey_weight_type=survey_weight_type,
            )

        # Create outcome matrices
        # Shape: (n_periods, n_units)
        Y_pre_control, Y_post_control, Y_pre_treated, Y_post_treated = (
            self._create_outcome_matrices(
                working_data,
                outcome,
                unit,
                time,
                pre_periods,
                post_periods,
                treated_units,
                control_units,
            )
        )

        # --- Y normalization ---------------------------------------------
        # τ is location-invariant and scale-equivariant in Y. Normalizing Y
        # once before weight optimization, the estimator, and the variance
        # procedures (and rescaling τ/SE/CI/effects by Y_scale at the end)
        # is mathematically a no-op but prevents ~6-digit precision loss in
        # the SDID double-difference when outcomes span millions-to-billions.
        # Normalization constants come from controls' pre-period only so the
        # reference is unaffected by treatment. See REGISTRY.md §SyntheticDiD
        # edge cases and synth-inference/synthdid#71 for R's version.
        Y_shift = float(np.mean(Y_pre_control))
        Y_scale_raw = float(np.std(Y_pre_control))
        # Relative floor: avoid amplifying roundoff when std is tiny but
        # nonzero (near-constant Y_pre_control). Fall back to 1.0 in that
        # case and on non-finite std.
        _scale_floor = 1e-12 * max(abs(Y_shift), 1.0)
        Y_scale = Y_scale_raw if np.isfinite(Y_scale_raw) and Y_scale_raw > _scale_floor else 1.0
        Y_pre_control_n = (Y_pre_control - Y_shift) / Y_scale
        Y_post_control_n = (Y_post_control - Y_shift) / Y_scale
        Y_pre_treated_n = (Y_pre_treated - Y_shift) / Y_scale
        Y_post_treated_n = (Y_post_treated - Y_shift) / Y_scale

        # Auto-regularization on normalized Y. FW's argmin is invariant under
        # (Y, ζ) -> (Y/s, ζ/s); auto-zetas computed on Y_n are already on the
        # normalized scale. User-supplied zetas are in original-Y units and
        # are divided by Y_scale for internal FW use. Original-scale values
        # are stored on results_ / self so diagnostic methods (in_time_placebo,
        # sensitivity_to_zeta_omega) — which operate on the stored original-
        # scale fit snapshot — see the same zeta the user specified.
        auto_zeta_omega_n, auto_zeta_lambda_n = _compute_regularization(
            Y_pre_control_n, len(treated_units), len(post_periods)
        )
        zeta_omega_n = (
            self.zeta_omega / Y_scale if self.zeta_omega is not None else auto_zeta_omega_n
        )
        zeta_lambda_n = (
            self.zeta_lambda / Y_scale if self.zeta_lambda is not None else auto_zeta_lambda_n
        )
        # Report the user-supplied value exactly (no roundoff from /*Y_scale
        # roundtrip); report auto-zeta rescaled to original Y units.
        zeta_omega = self.zeta_omega if self.zeta_omega is not None else auto_zeta_omega_n * Y_scale
        zeta_lambda = (
            self.zeta_lambda if self.zeta_lambda is not None else auto_zeta_lambda_n * Y_scale
        )

        # Store noise level for diagnostics (reported on original Y scale).
        from diff_diff.utils import _compute_noise_level

        noise_level_n = _compute_noise_level(Y_pre_control_n)
        noise_level = noise_level_n * Y_scale

        # Data-dependent convergence threshold (matches R's 1e-5 * noise.level),
        # evaluated on normalized Y since FW operates on normalized Y. Floor of
        # 1e-5 when noise is zero: R would use 0.0, causing FW to run all
        # max_iter iterations; the floor enables early stop on zero-variation
        # inputs without changing the optimum.
        min_decrease = 1e-5 * noise_level_n if noise_level_n > 0 else 1e-5

        # Compute unit weights (Frank-Wolfe with sparsification) on normalized Y.
        # Survey weights enter via the treated mean target.
        if w_treated is not None:
            Y_pre_treated_mean_n = np.average(Y_pre_treated_n, axis=1, weights=w_treated)
        else:
            Y_pre_treated_mean_n = np.mean(Y_pre_treated_n, axis=1)

        unit_weights = compute_sdid_unit_weights(
            Y_pre_control_n,
            Y_pre_treated_mean_n,
            zeta_omega=zeta_omega_n,
            min_decrease=min_decrease,
        )

        # Compute time weights (Frank-Wolfe on collapsed form) on normalized Y.
        time_weights = compute_time_weights(
            Y_pre_control_n,
            Y_post_control_n,
            zeta_lambda=zeta_lambda_n,
            min_decrease=min_decrease,
        )

        # Compose ω with control survey weights (WLS regression interpretation).
        # Frank-Wolfe finds best trajectory match; survey weights reweight by
        # population importance post-optimization.
        if w_control is not None:
            omega_eff = unit_weights * w_control
            omega_eff = omega_eff / omega_eff.sum()
        else:
            omega_eff = unit_weights

        # Compute SDID estimate on normalized Y, then rescale to original units.
        if w_treated is not None:
            Y_post_treated_mean_n = np.average(Y_post_treated_n, axis=1, weights=w_treated)
        else:
            Y_post_treated_mean_n = np.mean(Y_post_treated_n, axis=1)

        att_n = compute_sdid_estimator(
            Y_pre_control_n,
            Y_post_control_n,
            Y_pre_treated_mean_n,
            Y_post_treated_mean_n,
            omega_eff,
            time_weights,
        )
        att = att_n * Y_scale

        # Recover original-scale treated means for diagnostics / trajectories.
        Y_pre_treated_mean = Y_pre_treated_mean_n * Y_scale + Y_shift
        Y_post_treated_mean = Y_post_treated_mean_n * Y_scale + Y_shift

        # Compute pre-treatment fit (RMSE) using composed weights on the
        # original Y (user-visible scale). omega_eff is a simplex — applies
        # cleanly to any linear rescale of Y — so trajectories live on the
        # original outcome scale for plotting and the poor-fit warning.
        synthetic_pre_trajectory = Y_pre_control @ omega_eff
        synthetic_post_trajectory = Y_post_control @ omega_eff
        pre_fit_rmse = np.sqrt(np.mean((Y_pre_treated_mean - synthetic_pre_trajectory) ** 2))

        # Warn if pre-treatment fit is poor (Registry requirement).
        # Threshold: 1× SD of treated pre-treatment outcomes — a natural baseline
        # since RMSE exceeding natural variation indicates the synthetic control
        # fails to reproduce the treated series' level or trend.
        pre_treatment_sd = (
            np.std(Y_pre_treated_mean, ddof=1) if len(Y_pre_treated_mean) > 1 else 0.0
        )
        if pre_treatment_sd > 0 and pre_fit_rmse > pre_treatment_sd:
            warnings.warn(
                f"Pre-treatment fit is poor: RMSE ({pre_fit_rmse:.4f}) exceeds "
                f"the standard deviation of treated pre-treatment outcomes "
                f"({pre_treatment_sd:.4f}). The synthetic control may not "
                f"adequately reproduce treated unit trends. Consider adding "
                f"more control units or adjusting regularization.",
                UserWarning,
                stacklevel=2,
            )

        # Treated-unit trajectories (the pre/post means already computed above).
        treated_pre_trajectory = Y_pre_treated_mean
        treated_post_trajectory = Y_post_treated_mean

        # Compute standard errors on normalized Y, rescale to original units.
        # Variance procedures resample / permute indices (independent of Y
        # values) so RNG streams stay aligned across scales.
        if self.variance_method == "bootstrap":
            # Paper-faithful pairs bootstrap (Algorithm 2 step 2): re-estimate
            # ω̂_b and λ̂_b via Frank-Wolfe on each draw. With survey designs
            # the FW switches to the weighted-FW variant and Rao-Wu rescaling
            # supplies per-draw weights (PR #352). Pweight-only designs use
            # constant w_control across draws; full designs use Rao-Wu draws.
            # Determine which survey path the bootstrap should use:
            #   - resolved_survey_unit + strata/PSU/FPC → Rao-Wu rescaling
            #   - pweight-only (resolved_survey_unit but no strata/PSU) →
            #     pass w_control/w_treated as constant rw per draw (the
            #     bootstrap branch sets `_pweight_only` from `w_control`
            #     when resolved_survey is None).
            #   - non-survey → pass nothing (legacy path).
            full_design = (
                resolved_survey_unit is not None
                and (
                    resolved_survey_unit.strata is not None
                    or resolved_survey_unit.psu is not None
                    or resolved_survey_unit.fpc is not None
                )
            )
            _boot_resolved_survey = resolved_survey_unit if full_design else None
            _boot_w_control = w_control if not full_design else None
            _boot_w_treated = w_treated if not full_design else None

            se_n, bootstrap_estimates_n = self._bootstrap_se(
                Y_pre_control_n,
                Y_post_control_n,
                Y_pre_treated_n,
                Y_post_treated_n,
                unit_weights,
                time_weights,
                w_control=_boot_w_control,
                w_treated=_boot_w_treated,
                resolved_survey=_boot_resolved_survey,
                zeta_omega_n=zeta_omega_n,
                zeta_lambda_n=zeta_lambda_n,
                min_decrease=min_decrease,
            )
            se = se_n * Y_scale
            placebo_effects = np.asarray(bootstrap_estimates_n) * Y_scale
            inference_method = "bootstrap"
        elif self.variance_method == "jackknife":
            # Fixed-weight jackknife (R's synthdid Algorithm 3)
            se_n, jackknife_estimates_n = self._jackknife_se(
                Y_pre_control_n,
                Y_post_control_n,
                Y_pre_treated_n,
                Y_post_treated_n,
                unit_weights,
                time_weights,
                w_treated=w_treated,
                w_control=w_control,
            )
            se = se_n * Y_scale
            placebo_effects = np.asarray(jackknife_estimates_n) * Y_scale
            inference_method = "jackknife"
        else:
            # Use placebo-based variance (R's synthdid Algorithm 4).
            # Placebo re-estimates ω, λ inside the loop; it must receive the
            # normalized zetas and operate on normalized Y.
            se_n, placebo_effects_n = self._placebo_variance_se(
                Y_pre_control_n,
                Y_post_control_n,
                Y_pre_treated_mean_n,
                Y_post_treated_mean_n,
                n_treated=len(treated_units),
                zeta_omega=zeta_omega_n,
                zeta_lambda=zeta_lambda_n,
                min_decrease=min_decrease,
                replications=self.n_bootstrap,
                w_control=w_control,
            )
            se = se_n * Y_scale
            placebo_effects = np.asarray(placebo_effects_n) * Y_scale
            inference_method = "placebo"

        # Compute test statistics
        t_stat, p_value_analytical, conf_int = safe_inference(att, se, alpha=self.alpha)
        # Empirical p-value is valid only for placebo (Algorithm 4): control
        # permutations produce draws from the null distribution, centered on 0.
        # Bootstrap draws (Algorithm 2) are resampled units centered on τ̂
        # (sampling distribution, not null), and jackknife pseudo-values are not
        # null-distribution draws either. Both use the analytical p-value from
        # the bootstrap/jackknife SE.
        if inference_method == "placebo" and len(placebo_effects) > 0 and np.isfinite(t_stat):
            p_value = max(
                np.mean(np.abs(placebo_effects) >= np.abs(att)),
                1.0 / (len(placebo_effects) + 1),
            )
        else:
            p_value = p_value_analytical

        # Create weight dictionaries.  When survey weights are active, store
        # the effective (composed) weights that were actually used for the ATT
        # so that results.unit_weights matches the estimator.
        unit_weights_dict = {unit_id: w for unit_id, w in zip(control_units, omega_eff)}
        time_weights_dict = {period: w for period, w in zip(pre_periods, time_weights)}

        # Jackknife LOO ID/role arrays parallel to placebo_effects positions
        # (first n_control entries are control-LOO, next n_treated are treated-LOO;
        # see _jackknife_se docstring).
        loo_unit_ids: Optional[List[Any]]
        loo_roles: Optional[List[str]]
        if inference_method == "jackknife" and len(placebo_effects) > 0:
            loo_unit_ids = list(control_units) + list(treated_units)
            loo_roles = ["control"] * len(control_units) + ["treated"] * len(treated_units)
        else:
            loo_unit_ids = None
            loo_roles = None

        fit_snapshot = _SyntheticDiDFitSnapshot(
            Y_pre_control=Y_pre_control,
            Y_post_control=Y_post_control,
            Y_pre_treated=Y_pre_treated,
            Y_post_treated=Y_post_treated,
            control_unit_ids=list(control_units),
            treated_unit_ids=list(treated_units),
            pre_periods=list(pre_periods),
            post_periods=list(post_periods),
            w_control=w_control,
            w_treated=w_treated,
            Y_shift=Y_shift,
            Y_scale=Y_scale,
        )

        # Freeze the public diagnostic arrays so mutation via the results
        # object cannot silently invalidate later diagnostic calls.
        for _arr in (
            synthetic_pre_trajectory,
            synthetic_post_trajectory,
            treated_pre_trajectory,
            treated_post_trajectory,
            time_weights,
        ):
            _arr.setflags(write=False)

        # Store results. Internal diagnostic state (_loo_*, _fit_snapshot)
        # is attached as plain attributes after construction so that
        # dataclass-recursive serializers (dataclasses.asdict,
        # dataclasses.fields, dataclasses.replace) cannot reach retained
        # panel state or unit IDs.
        self.results_ = SyntheticDiDResults(
            att=att,
            se=se,
            t_stat=t_stat,
            p_value=p_value,
            conf_int=conf_int,
            n_obs=len(data),
            n_treated=len(treated_units),
            n_control=len(control_units),
            unit_weights=unit_weights_dict,
            time_weights=time_weights_dict,
            pre_periods=pre_periods,
            post_periods=post_periods,
            alpha=self.alpha,
            variance_method=inference_method,
            noise_level=noise_level,
            zeta_omega=zeta_omega,
            zeta_lambda=zeta_lambda,
            pre_treatment_fit=pre_fit_rmse,
            placebo_effects=placebo_effects if len(placebo_effects) > 0 else None,
            n_bootstrap=self.n_bootstrap if inference_method == "bootstrap" else None,
            survey_metadata=survey_metadata,
            synthetic_pre_trajectory=synthetic_pre_trajectory,
            synthetic_post_trajectory=synthetic_post_trajectory,
            treated_pre_trajectory=treated_pre_trajectory,
            treated_post_trajectory=treated_post_trajectory,
            time_weights_array=time_weights,
        )
        self.results_._loo_unit_ids = loo_unit_ids
        self.results_._loo_roles = loo_roles
        self.results_._fit_snapshot = fit_snapshot

        self._unit_weights = unit_weights
        self._time_weights = time_weights
        self.is_fitted_ = True

        return self.results_

    def _create_outcome_matrices(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        pre_periods: List[Any],
        post_periods: List[Any],
        treated_units: List[Any],
        control_units: List[Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create outcome matrices for SDID estimation.

        Returns
        -------
        tuple
            (Y_pre_control, Y_post_control, Y_pre_treated, Y_post_treated)
            Each is a 2D array with shape (n_periods, n_units)
        """
        # Pivot data to wide format
        pivot = data.pivot(index=time, columns=unit, values=outcome)

        # Extract submatrices
        Y_pre_control = pivot.loc[pre_periods, control_units].values
        Y_post_control = pivot.loc[post_periods, control_units].values
        Y_pre_treated = pivot.loc[pre_periods, treated_units].values
        Y_post_treated = pivot.loc[post_periods, treated_units].values

        return (
            Y_pre_control.astype(float),
            Y_post_control.astype(float),
            Y_pre_treated.astype(float),
            Y_post_treated.astype(float),
        )

    def _residualize_covariates(
        self,
        data: pd.DataFrame,
        outcome: str,
        covariates: List[str],
        unit: str,
        time: str,
        survey_weights=None,
        survey_weight_type=None,
    ) -> pd.DataFrame:
        """
        Residualize outcome by regressing out covariates.

        Uses two-way fixed effects to partial out covariates. When survey
        weights are provided, uses WLS for population-representative
        covariate removal.
        """
        data = data.copy()

        # Create design matrix with covariates
        X = data[covariates].values.astype(float)

        # Add unit and time dummies
        unit_dummies = pd.get_dummies(data[unit], prefix="u", drop_first=True)
        time_dummies = pd.get_dummies(data[time], prefix="t", drop_first=True)

        X_full = np.column_stack([np.ones(len(data)), X, unit_dummies.values, time_dummies.values])

        y = data[outcome].values.astype(float)

        # Fit and get residuals using unified backend
        coeffs, residuals, _ = solve_ols(
            X_full,
            y,
            return_vcov=False,
            weights=survey_weights,
            weight_type=survey_weight_type,
        )

        # Add back the mean for interpretability
        if survey_weights is not None:
            y_center = np.average(y, weights=survey_weights)
        else:
            y_center = np.mean(y)
        data[outcome] = residuals + y_center

        return data

    def _bootstrap_se(
        self,
        Y_pre_control: np.ndarray,
        Y_post_control: np.ndarray,
        Y_pre_treated: np.ndarray,
        Y_post_treated: np.ndarray,
        unit_weights: np.ndarray,
        time_weights: np.ndarray,
        zeta_omega_n: float = 0.0,
        zeta_lambda_n: float = 0.0,
        min_decrease: float = 1e-5,
        *,
        w_control: Optional[np.ndarray] = None,
        w_treated: Optional[np.ndarray] = None,
        resolved_survey=None,
    ) -> Tuple[float, np.ndarray]:
        """Compute pairs-bootstrap standard error for SDID (Algorithm 2 step 2).

        Paper-faithful refit bootstrap: resamples all units (control + treated)
        with replacement, then re-estimates ω̂_b and λ̂_b via two-pass sparsified
        Frank-Wolfe on each resampled panel (Arkhangelsky et al. 2021
        Algorithm 2 step 2). Also matches R's default
        ``synthdid::vcov(method="bootstrap")`` behavior: R rebinds
        ``attr(estimate, "opts")`` (including ``update.omega=TRUE``) back into
        ``synthdid_estimate`` on each draw, so the renormalized ω serves only
        as Frank-Wolfe initialization.

        Survey-bootstrap (PR #352): if ``w_control``/``w_treated`` (pweight-
        only) or ``resolved_survey`` (full design with strata/PSU/FPC) is
        passed, switches to weighted-FW per draw. Per-draw rescaled weights
        come from either the constant ``w_control`` (pweight-only) or
        ``generate_rao_wu_weights(resolved_survey, rng)`` sliced over the
        resampled units (full design). The weighted-FW solves
        ``min Σ_t (Σ_i rw_i·ω_i·Y_i,pre[t] - b_t)² + ζ²·Σ rw_i·ω_i²``;
        downstream ``ω_eff = rw·ω / Σ(rw·ω)`` is composed before the SDID
        estimator (see REGISTRY.md §SyntheticDiD ``Note (survey + bootstrap
        composition)`` for the argmin-set caveat).

        ``zeta_omega_n`` / ``zeta_lambda_n`` are the fit-time normalized-scale
        regularization parameters, used unchanged on each draw.

        ``unit_weights`` (fit-time ω) and ``time_weights`` (fit-time λ) are
        not reused as fixed estimator weights — every draw re-estimates ω̂_b
        and λ̂_b from scratch — but they ARE used as Frank-Wolfe warm-start
        initializations: ``_sum_normalize(unit_weights[boot_ctrl])`` seeds
        ω̂_b's first pass, ``time_weights`` seeds λ̂_b's. Matches R's
        ``vcov.R::bootstrap_sample`` shape.

        Retry-to-B: matches R's ``synthdid::bootstrap_sample`` while-loop;
        bounded attempt guard of ``20 * n_bootstrap``. Per-draw Frank-Wolfe
        non-convergence is tallied via the ``return_convergence`` flag from
        each helper and aggregated into one summary ``UserWarning`` if the
        rate exceeds 5% of valid draws.
        """
        rng = np.random.default_rng(self.seed)
        n_control = Y_pre_control.shape[1]
        n_treated = Y_pre_treated.shape[1]
        n_total = n_control + n_treated

        # Survey-bootstrap dispatch (PR #352). _use_rao_wu fires for any
        # survey design (the helper itself handles strata=None / psu=None
        # by treating each obs as its own PSU); _pweight_only fires when
        # we have constant per-control survey weights but no
        # resolved_survey object (e.g., a future caller path).
        _use_rao_wu = resolved_survey is not None
        _pweight_only = (w_control is not None) and (resolved_survey is None)

        # Single-PSU short-circuit: unstratified design with <2 PSUs has no
        # identified bootstrap distribution (resampling one PSU yields the
        # same subset every draw). Returns NaN SE — same shape as PR #351's
        # n_successful=0 raise but caught upstream as NaN. Recovered from
        # 91082e5:diff_diff/synthetic_did.py.
        if (
            _use_rao_wu
            and resolved_survey.psu is not None
            and resolved_survey.strata is None
        ):
            from numpy import unique as _unique
            n_psu = len(_unique(resolved_survey.psu))
            if n_psu < 2:
                return np.nan, np.array([])

        # Build full panel matrix: (n_pre+n_post, n_control+n_treated)
        Y_full = np.block([[Y_pre_control, Y_pre_treated], [Y_post_control, Y_post_treated]])
        n_pre = Y_pre_control.shape[0]

        bootstrap_estimates: List[float] = []

        # Retry-until-B-valid semantic: matches R's synthdid::bootstrap_sample.
        max_attempts = 20 * self.n_bootstrap
        attempts = 0
        # Tally draws with any Frank-Wolfe non-convergence; aggregated into
        # one summary warning after the loop.
        fw_nonconvergence_count = 0

        while len(bootstrap_estimates) < self.n_bootstrap and attempts < max_attempts:
            attempts += 1
            boot_idx = rng.choice(n_total, size=n_total, replace=True)

            # Split resampled units into control vs treated
            boot_is_control = boot_idx < n_control
            n_co_b = int(boot_is_control.sum())

            # Retry if no control or no treated units in bootstrap sample
            if n_co_b == 0 or n_co_b == n_total:
                continue

            try:
                # Per-draw rescaled weights (PR #352 survey path). For
                # Rao-Wu, generate_rao_wu_weights returns per-unit rescaled
                # weights (resolved_survey is unit-collapsed upstream in
                # fit(); see synthetic_did.py callers). For pweight-only,
                # rw is the constant per-control survey weight.
                if _use_rao_wu:
                    boot_rw = generate_rao_wu_weights(resolved_survey, rng)
                    rw_control_full = boot_rw[:n_control]
                    rw_treated_full = boot_rw[n_control:]
                    rw_control_draw = rw_control_full[boot_idx[boot_is_control]]
                    rw_treated_draw = rw_treated_full[boot_idx[~boot_is_control] - n_control]
                elif _pweight_only:
                    rw_control_draw = w_control[boot_idx[boot_is_control]]
                    rw_treated_draw = (
                        w_treated[boot_idx[~boot_is_control] - n_control]
                        if w_treated is not None
                        else None
                    )
                else:
                    rw_control_draw = None
                    rw_treated_draw = None

                # Degenerate-retry under ANY survey path: if a draw zeros
                # out the control or treated mass, retry. For Rao-Wu this
                # is expected behavior (PSUs not drawn get weight 0). For
                # pweight-only, zero-mass treated is reachable when at
                # least one unit has zero survey weight AND the
                # bootstrap resample happens to pick only zero-weight
                # treated units — silently falling back to an unweighted
                # mean would corrupt the bootstrap distribution because
                # fit-time ATT uses the survey-weighted mean (PR #355
                # R2 P0).
                if (_use_rao_wu or _pweight_only) and rw_treated_draw is not None and (
                    rw_control_draw.sum() == 0 or rw_treated_draw.sum() == 0
                ):
                    continue

                # Extract resampled outcome matrices
                Y_boot = Y_full[:, boot_idx]
                Y_boot_pre_c = Y_boot[:n_pre, boot_is_control]
                Y_boot_post_c = Y_boot[n_pre:, boot_is_control]
                Y_boot_pre_t = Y_boot[:n_pre, ~boot_is_control]
                Y_boot_post_t = Y_boot[n_pre:, ~boot_is_control]

                # Treated-unit mean: survey-weighted if rw_treated_draw is
                # set (PR #352), else unweighted.
                if rw_treated_draw is not None and rw_treated_draw.sum() > 0:
                    Y_boot_pre_t_mean = np.average(
                        Y_boot_pre_t, axis=1, weights=rw_treated_draw,
                    )
                    Y_boot_post_t_mean = np.average(
                        Y_boot_post_t, axis=1, weights=rw_treated_draw,
                    )
                else:
                    Y_boot_pre_t_mean = np.mean(Y_boot_pre_t, axis=1)
                    Y_boot_post_t_mean = np.mean(Y_boot_post_t, axis=1)

                # Warm-start init matching R's vcov.R::bootstrap_sample
                # shape. Under survey-bootstrap, scale the renormalized
                # init by rw before sum_normalize (matches the per-draw
                # weighted-FW geometry).
                boot_control_idx = boot_idx[boot_is_control]
                if rw_control_draw is not None:
                    boot_omega_init = _sum_normalize(
                        unit_weights[boot_control_idx] * rw_control_draw
                    )
                else:
                    boot_omega_init = _sum_normalize(unit_weights[boot_control_idx])
                boot_lambda_init = time_weights

                # Algorithm 2 step 2: re-estimate ω̂_b and λ̂_b via two-pass
                # sparsified Frank-Wolfe. Survey paths use the weighted-FW
                # helpers (PR #352 §1b); non-survey path unchanged.
                if rw_control_draw is not None:
                    boot_omega, omega_converged = compute_sdid_unit_weights_survey(
                        Y_boot_pre_c,
                        Y_boot_pre_t_mean,
                        rw_control_draw,
                        zeta_omega=zeta_omega_n,
                        min_decrease=min_decrease,
                        init_weights=boot_omega_init,
                        return_convergence=True,
                    )
                    boot_lambda, lambda_converged = compute_time_weights_survey(
                        Y_boot_pre_c,
                        Y_boot_post_c,
                        rw_control_draw,
                        zeta_lambda=zeta_lambda_n,
                        min_decrease=min_decrease,
                        init_weights=boot_lambda_init,
                        return_convergence=True,
                    )
                    # Compose ω_eff = rw·ω / Σ(rw·ω) for the SDID
                    # estimator (expects simplex weights). REGISTRY.md
                    # §SyntheticDiD covers the argmin-set caveat: the FW
                    # minimizes the weighted objective on ω, NOT the
                    # standard objective on ω_eff — intentional design
                    # choice validated by the stratified coverage MC.
                    omega_scaled = rw_control_draw * boot_omega
                    total = omega_scaled.sum()
                    if total <= 0:
                        # Degenerate: all mass on rw=0 controls
                        continue
                    boot_omega = omega_scaled / total
                else:
                    boot_omega, omega_converged = compute_sdid_unit_weights(
                        Y_boot_pre_c,
                        Y_boot_pre_t_mean,
                        zeta_omega=zeta_omega_n,
                        min_decrease=min_decrease,
                        init_weights=boot_omega_init,
                        return_convergence=True,
                    )
                    boot_lambda, lambda_converged = compute_time_weights(
                        Y_boot_pre_c,
                        Y_boot_post_c,
                        zeta_lambda=zeta_lambda_n,
                        min_decrease=min_decrease,
                        init_weights=boot_lambda_init,
                        return_convergence=True,
                    )

                tau = compute_sdid_estimator(
                    Y_boot_pre_c,
                    Y_boot_post_c,
                    Y_boot_pre_t_mean,
                    Y_boot_post_t_mean,
                    boot_omega,
                    boot_lambda,
                )
                if np.isfinite(tau):
                    bootstrap_estimates.append(float(tau))
                    # Count draws with ANY non-convergence (boolean per
                    # draw). Increment after the finite-τ gate so
                    # numerator and denominator (n_successful) match.
                    if not (omega_converged and lambda_converged):
                        fw_nonconvergence_count += 1

            except (ValueError, LinAlgError):
                continue

        bootstrap_estimates = np.array(bootstrap_estimates)

        # Check bootstrap success rate and handle failures.
        # n_successful should equal self.n_bootstrap under the retry contract;
        # it only falls short if max_attempts was exhausted on pathological data.
        n_successful = len(bootstrap_estimates)
        failure_rate = 1 - (n_successful / self.n_bootstrap)

        if n_successful == 0:
            raise ValueError(
                f"Could not produce any valid bootstrap draw in {attempts} "
                f"attempts (target {self.n_bootstrap}). This typically occurs when:\n"
                f"  - Sample size is too small for reliable resampling\n"
                f"  - Weight matrices are singular or near-singular\n"
                f"  - Insufficient pre-treatment periods for weight estimation\n"
                f"  - Too few control units relative to treated units\n"
                f"Consider using variance_method='placebo' or increasing "
                f"the regularization parameters (zeta_omega, zeta_lambda)."
            )
        if n_successful == 1:
            warnings.warn(
                f"Only 1/{self.n_bootstrap} valid bootstrap draw accumulated in "
                f"{attempts} attempts. Standard error cannot be computed reliably "
                f"(requires at least 2). Returning SE=0.0. Consider using "
                f"variance_method='placebo' or increasing the regularization "
                f"(zeta_omega, zeta_lambda).",
                UserWarning,
                stacklevel=2,
            )
            return 0.0, bootstrap_estimates

        if failure_rate > 0.05:
            warnings.warn(
                f"Bootstrap attempt budget exhausted: only {n_successful}/"
                f"{self.n_bootstrap} valid draws accumulated in {attempts} "
                f"attempts. Standard errors may be unreliable; pathological "
                f"inputs can produce this signal (e.g., extreme treated/control "
                f"imbalance or singular weight matrices).",
                UserWarning,
                stacklevel=2,
            )

        # Aggregate Frank-Wolfe non-convergence across bootstrap draws. Per-draw
        # convergence warnings from compute_sdid_unit_weights / compute_time_weights
        # are suppressed inside the loop; emit one summary here if the rate
        # exceeds the same 5% threshold used for retry exhaustion.
        if fw_nonconvergence_count > 0.05 * max(n_successful, 1):
            warnings.warn(
                f"Frank-Wolfe did not converge on {fw_nonconvergence_count} of "
                f"{n_successful} valid bootstrap draws "
                f"(variance_method='bootstrap'). SE is still reported from "
                f"the final iterate of each draw, but non-convergent draws may be "
                f"noisier; consider relaxing min_decrease or increasing pre-period "
                f"length if regularization is already moderate.",
                UserWarning,
                stacklevel=2,
            )

        # SE formula matches R's synthdid::vcov(method="bootstrap"):
        # sqrt((r-1)/r) * sd(ddof=1), equivalent to Arkhangelsky et al. (2021)
        # Algorithm 2's σ̂² = (1/r) Σ (τ_b - τ̄)². Uses n_successful for
        # consistency with _placebo_variance_se.
        se = float(np.sqrt((n_successful - 1) / n_successful) * np.std(bootstrap_estimates, ddof=1))

        return se, bootstrap_estimates

    def _placebo_variance_se(
        self,
        Y_pre_control: np.ndarray,
        Y_post_control: np.ndarray,
        Y_pre_treated_mean: np.ndarray,
        Y_post_treated_mean: np.ndarray,
        n_treated: int,
        zeta_omega: float = 0.0,
        zeta_lambda: float = 0.0,
        min_decrease: float = 1e-5,
        replications: int = 200,
        w_control=None,
    ) -> Tuple[float, np.ndarray]:
        """
        Compute placebo-based variance matching R's synthdid methodology.

        This implements Algorithm 4 from Arkhangelsky et al. (2021),
        matching R's synthdid::vcov(method = "placebo"):

        1. Randomly sample N₀ control indices (permutation)
        2. Designate last N₁ as pseudo-treated, first (N₀-N₁) as pseudo-controls
        3. Re-estimate both omega and lambda on the permuted data (from
           uniform initialization, fresh start), matching R's behavior where
           ``update.omega=TRUE, update.lambda=TRUE`` are passed via ``opts``
        4. Compute SDID estimate with re-estimated weights
        5. Repeat `replications` times
        6. SE = sqrt((r-1)/r) * sd(estimates)

        Parameters
        ----------
        Y_pre_control : np.ndarray
            Control outcomes in pre-treatment periods, shape (n_pre, n_control).
        Y_post_control : np.ndarray
            Control outcomes in post-treatment periods, shape (n_post, n_control).
        Y_pre_treated_mean : np.ndarray
            Mean treated outcomes in pre-treatment periods, shape (n_pre,).
        Y_post_treated_mean : np.ndarray
            Mean treated outcomes in post-treatment periods, shape (n_post,).
        n_treated : int
            Number of treated units in the original estimation.
        zeta_omega : float
            Regularization parameter for unit weights (for re-estimation).
        zeta_lambda : float
            Regularization parameter for time weights (for re-estimation).
        min_decrease : float
            Convergence threshold for Frank-Wolfe (for re-estimation).
        replications : int, default=200
            Number of placebo replications.

        Returns
        -------
        tuple
            (se, placebo_effects) where se is the standard error and
            placebo_effects is the array of placebo treatment effects.

        References
        ----------
        Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
        (2021). Synthetic Difference-in-Differences. American Economic Review,
        111(12), 4088-4118. Algorithm 4.
        """
        rng = np.random.default_rng(self.seed)
        n_pre, n_control = Y_pre_control.shape

        # Ensure we have enough controls for the split
        n_pseudo_control = n_control - n_treated
        if n_pseudo_control < 1:
            # Fallback guidance. Placebo and jackknife reject strata/PSU/FPC,
            # but bootstrap (PR #352) supports both pweight-only and
            # full-design surveys, so it's always a valid fallback.
            fallback = (
                "variance_method='bootstrap' (supports pweight-only and "
                "strata/PSU/FPC survey designs), variance_method='jackknife' "
                "(pweight-only only), or adding more control units"
                if w_control is not None
                else "variance_method='bootstrap', variance_method='jackknife', "
                "or adding more control units"
            )
            warnings.warn(
                f"Not enough control units ({n_control}) for placebo variance "
                f"estimation with {n_treated} treated units. "
                f"Consider using {fallback}.",
                UserWarning,
                stacklevel=3,
            )
            return 0.0, np.array([])

        placebo_estimates = []

        for _ in range(replications):
            try:
                # Random permutation of control indices (Algorithm 4, step 1)
                perm = rng.permutation(n_control)

                # Split into pseudo-controls and pseudo-treated (step 2)
                pseudo_control_idx = perm[:n_pseudo_control]
                pseudo_treated_idx = perm[n_pseudo_control:]

                # Get pseudo-control and pseudo-treated outcomes
                Y_pre_pseudo_control = Y_pre_control[:, pseudo_control_idx]
                Y_post_pseudo_control = Y_post_control[:, pseudo_control_idx]

                # Pseudo-treated means: survey-weighted when available
                if w_control is not None:
                    pseudo_w_tr = w_control[pseudo_treated_idx]
                    Y_pre_pseudo_treated_mean = np.average(
                        Y_pre_control[:, pseudo_treated_idx],
                        axis=1,
                        weights=pseudo_w_tr,
                    )
                    Y_post_pseudo_treated_mean = np.average(
                        Y_post_control[:, pseudo_treated_idx],
                        axis=1,
                        weights=pseudo_w_tr,
                    )
                else:
                    Y_pre_pseudo_treated_mean = np.mean(
                        Y_pre_control[:, pseudo_treated_idx], axis=1
                    )
                    Y_post_pseudo_treated_mean = np.mean(
                        Y_post_control[:, pseudo_treated_idx], axis=1
                    )

                # Re-estimate weights on permuted data (matching R's behavior)
                # R passes update.omega=TRUE, update.lambda=TRUE via opts,
                # re-estimating weights from uniform initialization (fresh start).
                # Unit weights: re-estimate on pseudo-control/pseudo-treated data
                pseudo_omega = compute_sdid_unit_weights(
                    Y_pre_pseudo_control,
                    Y_pre_pseudo_treated_mean,
                    zeta_omega=zeta_omega,
                    min_decrease=min_decrease,
                )

                # Compose pseudo_omega with control survey weights
                if w_control is not None:
                    pseudo_w_co = w_control[pseudo_control_idx]
                    pseudo_omega_eff = pseudo_omega * pseudo_w_co
                    pseudo_omega_eff = pseudo_omega_eff / pseudo_omega_eff.sum()
                else:
                    pseudo_omega_eff = pseudo_omega

                # Time weights: re-estimate on pseudo-control data
                pseudo_lambda = compute_time_weights(
                    Y_pre_pseudo_control,
                    Y_post_pseudo_control,
                    zeta_lambda=zeta_lambda,
                    min_decrease=min_decrease,
                )

                # Compute placebo SDID estimate (step 4)
                tau = compute_sdid_estimator(
                    Y_pre_pseudo_control,
                    Y_post_pseudo_control,
                    Y_pre_pseudo_treated_mean,
                    Y_post_pseudo_treated_mean,
                    pseudo_omega_eff,
                    pseudo_lambda,
                )
                if np.isfinite(tau):
                    placebo_estimates.append(tau)

            except (ValueError, LinAlgError, ZeroDivisionError):
                # Skip failed iterations
                continue

        placebo_estimates = np.array(placebo_estimates)
        n_successful = len(placebo_estimates)

        if n_successful < 2:
            # Same fallback guidance as the pre-replication guard above.
            # Bootstrap (PR #352) supports pweight-only + strata/PSU/FPC
            # survey designs, so it's always a valid fallback for survey
            # users even when placebo fails.
            fallback = (
                "variance_method='bootstrap' (supports pweight-only and "
                "strata/PSU/FPC survey designs), variance_method='jackknife' "
                "(pweight-only only), or increasing the number of control units"
                if w_control is not None
                else "variance_method='bootstrap' or variance_method='jackknife' "
                "or increasing the number of control units"
            )
            warnings.warn(
                f"Only {n_successful} placebo replications completed successfully. "
                f"Standard error cannot be estimated reliably. "
                f"Consider using {fallback}.",
                UserWarning,
                stacklevel=3,
            )
            return 0.0, placebo_estimates

        # Warn if many replications failed
        failure_rate = 1 - (n_successful / replications)
        if failure_rate > 0.05:
            warnings.warn(
                f"Only {n_successful}/{replications} placebo replications succeeded "
                f"({failure_rate:.1%} failure rate). Standard errors may be unreliable.",
                UserWarning,
                stacklevel=3,
            )

        # Compute SE using R's formula: sqrt((r-1)/r) * sd(estimates)
        # This matches synthdid::vcov.R exactly
        se = np.sqrt((n_successful - 1) / n_successful) * np.std(placebo_estimates, ddof=1)

        return se, placebo_estimates

    def _jackknife_se(
        self,
        Y_pre_control: np.ndarray,
        Y_post_control: np.ndarray,
        Y_pre_treated: np.ndarray,
        Y_post_treated: np.ndarray,
        unit_weights: np.ndarray,
        time_weights: np.ndarray,
        w_treated=None,
        w_control=None,
    ) -> Tuple[float, np.ndarray]:
        """Compute jackknife standard error matching R's synthdid Algorithm 3.

        Delete-1 jackknife over all units (control + treated) with **fixed**
        weights.  For each leave-one-out sample the original omega is subsetted
        and renormalized; lambda stays unchanged.  No Frank-Wolfe
        re-estimation, making this the fastest variance method.

        This matches R's ``synthdid::vcov(method="jackknife")`` which sets
        ``update.omega=FALSE, update.lambda=FALSE``.

        Parameters
        ----------
        Y_pre_control : np.ndarray
            Control outcomes in pre-treatment periods, shape (n_pre, n_control).
        Y_post_control : np.ndarray
            Control outcomes in post-treatment periods, shape (n_post, n_control).
        Y_pre_treated : np.ndarray
            Treated outcomes in pre-treatment periods, shape (n_pre, n_treated).
        Y_post_treated : np.ndarray
            Treated outcomes in post-treatment periods, shape (n_post, n_treated).
        unit_weights : np.ndarray
            Unit weights from Frank-Wolfe optimization, shape (n_control,).
        time_weights : np.ndarray
            Time weights from Frank-Wolfe optimization, shape (n_pre,).
        w_treated : np.ndarray, optional
            Survey probability weights for treated units.
        w_control : np.ndarray, optional
            Survey probability weights for control units.

        Returns
        -------
        tuple
            (se, jackknife_estimates) where se is the standard error and
            jackknife_estimates is a length-N array of leave-one-out estimates
            (first n_control entries are control-LOO, last n_treated are
            treated-LOO).

        References
        ----------
        Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W., & Wager, S.
        (2021). Synthetic Difference-in-Differences. American Economic Review,
        111(12), 4088-4118. Algorithm 3.
        """
        n_control = Y_pre_control.shape[1]
        n_treated = Y_pre_treated.shape[1]
        n = n_control + n_treated

        # --- Early-return NaN: matches R's NA conditions ---
        if n_treated <= 1:
            warnings.warn(
                "Jackknife variance requires more than 1 treated unit. "
                "Use variance_method='placebo' for single treated unit.",
                UserWarning,
                stacklevel=3,
            )
            return np.nan, np.array([])

        if np.sum(unit_weights > 0) <= 1:
            warnings.warn(
                "Jackknife variance requires more than 1 control unit with "
                "nonzero weight. Consider variance_method='placebo'.",
                UserWarning,
                stacklevel=3,
            )
            return np.nan, np.array([])

        # --- Effective-support guards for survey-weighted path ---
        if w_control is not None:
            effective_control = unit_weights * w_control
            if np.sum(effective_control > 0) <= 1:
                warnings.warn(
                    "Jackknife variance requires more than 1 control unit with "
                    "positive effective weight (omega * survey_weight). "
                    "Consider variance_method='placebo'.",
                    UserWarning,
                    stacklevel=3,
                )
                return np.nan, np.array([])

        if w_treated is not None and np.sum(w_treated > 0) <= 1:
            warnings.warn(
                "Jackknife variance requires more than 1 treated unit with "
                "positive survey weight. "
                "Consider variance_method='placebo'.",
                UserWarning,
                stacklevel=3,
            )
            return np.nan, np.array([])

        jackknife_estimates = np.empty(n)

        # --- Precompute treated means (constant across control-LOO) ---
        if w_treated is not None:
            treated_pre_mean = np.average(Y_pre_treated, axis=1, weights=w_treated)
            treated_post_mean = np.average(Y_post_treated, axis=1, weights=w_treated)
        else:
            treated_pre_mean = np.mean(Y_pre_treated, axis=1)
            treated_post_mean = np.mean(Y_post_treated, axis=1)

        # --- Precompute omega composed with survey weights (for treated-LOO) ---
        if w_control is not None:
            omega_eff_full = unit_weights * w_control
            omega_eff_full = omega_eff_full / omega_eff_full.sum()
        else:
            omega_eff_full = unit_weights

        # --- Leave-one-out over control units ---
        mask = np.ones(n_control, dtype=bool)
        for j in range(n_control):
            mask[j] = False

            # Subset and renormalize omega
            omega_jk = _sum_normalize(unit_weights[mask])

            # Compose with survey weights if present
            if w_control is not None:
                omega_jk = omega_jk * w_control[mask]
                if omega_jk.sum() == 0:
                    jackknife_estimates[j] = np.nan
                    mask[j] = True
                    continue
                omega_jk = omega_jk / omega_jk.sum()

            jackknife_estimates[j] = compute_sdid_estimator(
                Y_pre_control[:, mask],
                Y_post_control[:, mask],
                treated_pre_mean,
                treated_post_mean,
                omega_jk,
                time_weights,
            )

            mask[j] = True  # restore for next iteration

        # --- Leave-one-out over treated units ---
        mask = np.ones(n_treated, dtype=bool)
        for k in range(n_treated):
            mask[k] = False

            # Recompute treated means from remaining units
            if w_treated is not None:
                w_t_jk = w_treated[mask]
                if w_t_jk.sum() == 0:
                    jackknife_estimates[n_control + k] = np.nan
                    mask[k] = True
                    continue
                t_pre_mean = np.average(Y_pre_treated[:, mask], axis=1, weights=w_t_jk)
                t_post_mean = np.average(Y_post_treated[:, mask], axis=1, weights=w_t_jk)
            else:
                t_pre_mean = np.mean(Y_pre_treated[:, mask], axis=1)
                t_post_mean = np.mean(Y_post_treated[:, mask], axis=1)

            jackknife_estimates[n_control + k] = compute_sdid_estimator(
                Y_pre_control,
                Y_post_control,
                t_pre_mean,
                t_post_mean,
                omega_eff_full,
                time_weights,
            )

            mask[k] = True  # restore for next iteration

        # --- Check for non-finite estimates (propagate NaN like R's var()) ---
        if not np.all(np.isfinite(jackknife_estimates)):
            warnings.warn(
                "Some jackknife leave-one-out estimates are non-finite. "
                "Standard error cannot be computed.",
                UserWarning,
                stacklevel=3,
            )
            return np.nan, jackknife_estimates

        # --- Jackknife SE formula: sqrt((n-1)/n * sum((u - ubar)^2)) ---
        # Matches R's: sqrt(((n-1)/n) * (n-1) * var(u))
        u_bar = np.mean(jackknife_estimates)
        ss = np.sum((jackknife_estimates - u_bar) ** 2)
        se = np.sqrt((n - 1) / n * ss)

        return se, jackknife_estimates

    def get_params(self) -> Dict[str, Any]:
        """Get estimator parameters."""
        return {
            "zeta_omega": self.zeta_omega,
            "zeta_lambda": self.zeta_lambda,
            "alpha": self.alpha,
            "variance_method": self.variance_method,
            "n_bootstrap": self.n_bootstrap,
            "seed": self.seed,
        }

    def set_params(self, **params) -> "SyntheticDiD":
        """Set estimator parameters.

        Applies updates transactionally: if ``_validate_config()`` rejects the
        post-update state, the instance is rolled back to the pre-call values
        so a raised ``ValueError`` leaves the object consistent with its
        pre-call configuration.
        """
        # Deprecated parameter names — emit warning and ignore
        _deprecated = {"lambda_reg", "zeta"}
        # Snapshot original values for transactional rollback on validation failure.
        _rollback: Dict[str, Any] = {}
        for key in params:
            if key not in _deprecated and hasattr(self, key):
                _rollback[key] = getattr(self, key)
        try:
            for key, value in params.items():
                if key in _deprecated:
                    warnings.warn(
                        f"{key} is deprecated and ignored. Use zeta_omega/zeta_lambda "
                        f"instead. Will be removed in v4.0.0.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                elif hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise ValueError(f"Unknown parameter: {key}")
            self._validate_config()
        except (ValueError, TypeError):
            for key, prev in _rollback.items():
                setattr(self, key, prev)
            raise
        return self
