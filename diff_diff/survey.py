"""
Survey data support for diff-diff.

Provides SurveyDesign for specifying complex survey structures (stratification,
clustering, weights, FPC) and Taylor Series Linearization (TSL) variance
estimation for design-based inference.

References
----------
- Lumley (2004) "Analysis of Complex Survey Samples", JSS 9(8).
- Binder (1983) "On the Variances of Asymptotically Normal Estimators
  from Complex Surveys", International Statistical Review 51(3).
- Solon, Haider, & Wooldridge (2015) "What Are We Weighting For?",
  Journal of Human Resources 50(2).
"""

import warnings
from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from diff_diff.linalg import _factorize_cluster_ids


@dataclass
class SurveyDesign:
    """
    User-facing class specifying complex survey design structure.

    Column names are resolved against the DataFrame at fit-time via
    :meth:`resolve`.

    Parameters
    ----------
    weights : str, optional
        Column name for observation weights (sampling weights).
    strata : str, optional
        Column name for stratification variable.
    psu : str, optional
        Column name for primary sampling unit (cluster).
    fpc : str, optional
        Column name for finite population size (N_h per stratum).
    weight_type : str, default "pweight"
        Weight type: "pweight" (inverse selection probability),
        "fweight" (frequency/expansion), or "aweight" (inverse variance).
    nest : bool, default False
        Whether PSU IDs are nested within strata (i.e., PSU IDs may repeat
        across strata). If True, PSU IDs are made unique by combining with
        strata.
    lonely_psu : str, default "remove"
        How to handle singleton strata (strata with only one PSU):
        "remove" (skip, emit warning), "certainty" (set f_h=1, zero
        variance contribution), or "adjust" (center around grand mean).
    """

    weights: Optional[str] = None
    strata: Optional[str] = None
    psu: Optional[str] = None
    fpc: Optional[str] = None
    weight_type: str = "pweight"
    nest: bool = False
    lonely_psu: str = "remove"

    def __post_init__(self):
        valid_weight_types = {"pweight", "fweight", "aweight"}
        if self.weight_type not in valid_weight_types:
            raise ValueError(
                f"weight_type must be one of {valid_weight_types}, " f"got '{self.weight_type}'"
            )
        valid_lonely = {"remove", "certainty", "adjust"}
        if self.lonely_psu not in valid_lonely:
            raise ValueError(
                f"lonely_psu must be one of {valid_lonely}, " f"got '{self.lonely_psu}'"
            )

    def resolve(self, data: pd.DataFrame) -> "ResolvedSurveyDesign":
        """
        Validate column names and extract numpy arrays from DataFrame.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing survey design columns.

        Returns
        -------
        ResolvedSurveyDesign
            Internal representation with extracted numpy arrays.
        """
        n = len(data)

        # --- Weights ---
        if self.weights is not None:
            if self.weights not in data.columns:
                raise ValueError(f"Weight column '{self.weights}' not found in data")
            raw_weights = data[self.weights].values.astype(np.float64)

            # Validate weights
            if np.any(np.isnan(raw_weights)):
                raise ValueError("Weights contain NaN values")
            if np.any(~np.isfinite(raw_weights)):
                raise ValueError("Weights contain Inf values")
            if np.any(raw_weights <= 0):
                raise ValueError("Weights must be strictly positive")

            # fweight validation: must be positive integers
            if self.weight_type == "fweight":
                fractional = raw_weights - np.round(raw_weights)
                if np.any(np.abs(fractional) > 1e-10):
                    raise ValueError(
                        "Frequency weights (fweight) must be positive integers. "
                        "Fractional values detected. Use pweight for non-integer weights."
                    )

            # Normalize: pweights/aweights to sum=n (mean=1); fweights unchanged
            if self.weight_type in ("pweight", "aweight"):
                weights = raw_weights * (n / np.sum(raw_weights))
            else:
                weights = raw_weights.copy()
        else:
            weights = np.ones(n, dtype=np.float64)

        # --- Strata ---
        strata_arr = None
        n_strata = 0
        if self.strata is not None:
            if self.strata not in data.columns:
                raise ValueError(f"Strata column '{self.strata}' not found in data")
            strata_vals = data[self.strata].values
            if pd.isna(strata_vals).any():
                raise ValueError(
                    f"Strata column '{self.strata}' contains missing values. "
                    "All observations must have valid strata identifiers."
                )
            strata_arr = _factorize_cluster_ids(strata_vals)
            n_strata = len(np.unique(strata_arr))

        # --- PSU ---
        psu_arr = None
        n_psu = 0
        if self.psu is not None:
            if self.psu not in data.columns:
                raise ValueError(f"PSU column '{self.psu}' not found in data")
            psu_raw = data[self.psu].values
            if pd.isna(psu_raw).any():
                raise ValueError(
                    f"PSU column '{self.psu}' contains missing values. "
                    "All observations must have valid PSU identifiers."
                )

            if self.nest and strata_arr is not None:
                # Make PSU IDs unique within strata by combining
                combined = np.array([f"{s}_{p}" for s, p in zip(strata_arr, psu_raw)])
                psu_arr = _factorize_cluster_ids(combined)
            else:
                psu_arr = _factorize_cluster_ids(psu_raw)
                # Validate PSU labels are globally unique when nest=False
                # and strata are present. Repeated labels cause wrong n_psu,
                # df_survey, and lonely_psu="adjust" global mean.
                if strata_arr is not None:
                    seen_psus: set = set()
                    for h in np.unique(strata_arr):
                        psu_in_h = set(psu_raw[strata_arr == h])
                        overlap = seen_psus & psu_in_h
                        if overlap:
                            raise ValueError(
                                f"PSU labels {overlap} appear in multiple strata. "
                                "Set nest=True in SurveyDesign to make PSU IDs "
                                "unique within strata, or use globally unique "
                                "PSU labels."
                            )
                        seen_psus |= psu_in_h

            n_psu = len(np.unique(psu_arr))

        # --- FPC ---
        fpc_arr = None
        if self.fpc is not None:
            if self.fpc not in data.columns:
                raise ValueError(f"FPC column '{self.fpc}' not found in data")
            fpc_arr = data[self.fpc].values.astype(np.float64)

            if np.any(np.isnan(fpc_arr)) or np.any(~np.isfinite(fpc_arr)):
                raise ValueError("FPC values must be finite and non-NaN")

            # Validate FPC structure (constant within strata, positive).
            # FPC >= n_PSU validation is deferred to compute_survey_vcov()
            # where the final effective PSU structure is known (after
            # cluster-as-PSU injection and implicit per-obs PSU fallback).
            if strata_arr is not None:
                for h in np.unique(strata_arr):
                    mask_h = strata_arr == h
                    fpc_vals = fpc_arr[mask_h]
                    # Enforce FPC is constant within stratum
                    if len(np.unique(fpc_vals)) > 1:
                        raise ValueError(
                            f"FPC values must be constant within each stratum. "
                            f"Stratum {h} has values: {np.unique(fpc_vals)}"
                        )
                    fpc_h = fpc_vals[0]
                    # Validate FPC >= n_PSU when explicit PSU is declared
                    if psu_arr is not None:
                        n_psu_h = len(np.unique(psu_arr[mask_h]))
                        if fpc_h < n_psu_h:
                            raise ValueError(
                                f"FPC ({fpc_h}) is less than the number of PSUs "
                                f"({n_psu_h}) in stratum {h}. FPC must be >= n_PSU."
                            )
            else:
                # No strata: require FPC is a single constant value
                if len(np.unique(fpc_arr)) > 1:
                    raise ValueError(
                        "FPC values must be constant when no strata are specified. "
                        f"Found {len(np.unique(fpc_arr))} distinct values."
                    )
                # Validate FPC >= n_PSU when explicit PSU is declared
                if psu_arr is not None and fpc_arr[0] < n_psu:
                    raise ValueError(
                        f"FPC ({fpc_arr[0]}) is less than the number of PSUs "
                        f"({n_psu}). FPC must be >= number of PSUs."
                    )

        # --- Validate PSU counts per stratum ---
        if psu_arr is not None and strata_arr is not None:
            for h in np.unique(strata_arr):
                mask_h = strata_arr == h
                n_psu_h = len(np.unique(psu_arr[mask_h]))
                if n_psu_h < 2:
                    if self.lonely_psu == "remove":
                        warnings.warn(
                            f"Stratum {h} has only {n_psu_h} PSU(s). "
                            "It will be excluded from variance estimation "
                            "(lonely_psu='remove').",
                            UserWarning,
                            stacklevel=3,
                        )
                    elif self.lonely_psu == "certainty":
                        pass  # Handled in compute_survey_vcov
                    elif self.lonely_psu == "adjust":
                        pass  # Handled in compute_survey_vcov

        # Validate PSU count for unstratified designs
        if psu_arr is not None and strata_arr is None:
            if n_psu < 2:
                if self.lonely_psu == "remove":
                    msg = (
                        f"Only {n_psu} PSU(s) found (unstratified design). "
                        "Variance cannot be estimated (lonely_psu='remove')."
                    )
                elif self.lonely_psu == "certainty":
                    msg = (
                        f"Only {n_psu} PSU(s) found (unstratified design). "
                        "Treated as certainty PSU; zero variance contribution."
                    )
                else:
                    msg = (
                        f"Only {n_psu} PSU(s) found (unstratified design). "
                        "Cannot adjust with a single cluster and no strata; "
                        "variance will be NaN."
                    )
                warnings.warn(msg, UserWarning, stacklevel=3)

        return ResolvedSurveyDesign(
            weights=weights,
            weight_type=self.weight_type,
            strata=strata_arr,
            psu=psu_arr,
            fpc=fpc_arr,
            n_strata=n_strata,
            n_psu=n_psu,
            lonely_psu=self.lonely_psu,
        )


@dataclass
class ResolvedSurveyDesign:
    """
    Internal class with extracted numpy arrays from SurveyDesign.resolve().

    Not intended for direct construction by users.
    """

    weights: np.ndarray
    weight_type: str
    strata: Optional[np.ndarray]
    psu: Optional[np.ndarray]
    fpc: Optional[np.ndarray]
    n_strata: int
    n_psu: int
    lonely_psu: str

    @property
    def df_survey(self) -> Optional[int]:
        """Survey degrees of freedom: n_PSU - n_strata."""
        if self.psu is not None and self.n_psu > 0:
            if self.strata is not None and self.n_strata > 0:
                return self.n_psu - self.n_strata
            return self.n_psu - 1
        # Implicit PSU: each observation is its own PSU
        n_obs = len(self.weights)
        if self.strata is not None and self.n_strata > 0:
            return n_obs - self.n_strata
        return n_obs - 1

    @property
    def needs_survey_vcov(self) -> bool:
        """Whether survey vcov (not generic sandwich) should be used."""
        return True  # Any resolved survey design uses the survey vcov path


@dataclass
class SurveyMetadata:
    """
    Survey design metadata stored in results objects.

    Attributes
    ----------
    weight_type : str
        Type of weights used.
    effective_n : float
        Kish effective sample size: (sum(w))^2 / sum(w^2).
    design_effect : float
        DEFF: n * sum(w^2) / (sum(w))^2.
    sum_weights : float
        Sum of original (pre-normalization) weights.
    n_strata : int or None
        Number of strata (None if unstratified).
    n_psu : int or None
        Number of PSUs (None if no PSU specified).
    weight_range : tuple of (float, float)
        (min, max) of original weights.
    df_survey : int or None
        Survey degrees of freedom (n_psu - n_strata).
    """

    weight_type: str
    effective_n: float
    design_effect: float
    sum_weights: float
    n_strata: Optional[int] = None
    n_psu: Optional[int] = None
    weight_range: Tuple[float, float] = field(default=(0.0, 0.0))
    df_survey: Optional[int] = None


def compute_survey_metadata(
    resolved: "ResolvedSurveyDesign",
    raw_weights: np.ndarray,
) -> SurveyMetadata:
    """
    Compute survey metadata from resolved design.

    Parameters
    ----------
    resolved : ResolvedSurveyDesign
        Resolved survey design.
    raw_weights : np.ndarray
        Original (pre-normalization) weights.

    Returns
    -------
    SurveyMetadata
    """
    sum_w = float(np.sum(raw_weights))
    sum_w2 = float(np.sum(raw_weights**2))
    n = len(raw_weights)

    effective_n = sum_w**2 / sum_w2 if sum_w2 > 0 else float(n)
    design_effect = n * sum_w2 / (sum_w**2) if sum_w > 0 else 1.0

    n_strata = resolved.n_strata if resolved.strata is not None else None
    if resolved.psu is not None:
        n_psu = resolved.n_psu
    else:
        # Implicit PSU: each observation is its own PSU
        n_psu = len(resolved.weights)
    df_survey = resolved.df_survey

    return SurveyMetadata(
        weight_type=resolved.weight_type,
        effective_n=effective_n,
        design_effect=design_effect,
        sum_weights=sum_w,
        n_strata=n_strata,
        n_psu=n_psu,
        weight_range=(float(np.min(raw_weights)), float(np.max(raw_weights))),
        df_survey=df_survey,
    )


def _validate_unit_constant_survey(data, unit_col, survey_design):
    """Validate that survey design columns are constant within units.

    Panel estimators (ContinuousDiD, EfficientDiD) collapse panel-level
    survey info to one row per unit. This requires that survey columns
    do not vary across time periods within a unit.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    unit_col : str
        Unit identifier column name.
    survey_design : SurveyDesign
        Survey design specification (uses attribute names, not resolved arrays).

    Raises
    ------
    ValueError
        If any survey column varies within units.
    """
    cols_to_check = [
        survey_design.weights,
        survey_design.strata,
        survey_design.psu,
        survey_design.fpc,
    ]
    for col in cols_to_check:
        if col is not None and col in data.columns:
            n_unique = data.groupby(unit_col)[col].nunique()
            varying_units = n_unique[n_unique > 1]
            if len(varying_units) > 0:
                raise ValueError(
                    f"Survey column '{col}' varies within units "
                    f"(found {len(varying_units)} units with multiple values). "
                    f"Panel estimators require survey design columns to be "
                    f"constant within units."
                )


def _resolve_pweight_only(resolved_survey, estimator_name):
    """Guard: reject non-pweight and strata/PSU/FPC for pweight-only estimators.

    Parameters
    ----------
    resolved_survey : ResolvedSurveyDesign or None
        Resolved survey design. If None, returns immediately.
    estimator_name : str
        Estimator name for error messages.

    Raises
    ------
    ValueError
        If weight_type is not 'pweight'.
    NotImplementedError
        If strata, PSU, or FPC are present.
    """
    if resolved_survey is None:
        return
    if resolved_survey.weight_type != "pweight":
        raise ValueError(
            f"{estimator_name} survey support requires weight_type='pweight'. "
            f"Got '{resolved_survey.weight_type}'."
        )
    if (
        resolved_survey.strata is not None
        or resolved_survey.psu is not None
        or resolved_survey.fpc is not None
    ):
        raise NotImplementedError(
            f"{estimator_name} does not yet support strata/PSU/FPC in "
            "SurveyDesign. Use SurveyDesign(weights=...) only. Full "
            "design-based bootstrap is planned for the Bootstrap + "
            "Survey Interaction phase."
        )


def _extract_unit_survey_weights(data, unit_col, survey_design, unit_order):
    """Extract unit-level survey weights aligned to a given unit ordering.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data with survey weight column.
    unit_col : str
        Unit identifier column name.
    survey_design : SurveyDesign
        Survey design (uses ``weights`` column name).
    unit_order : array-like
        Ordered sequence of unit identifiers to align weights to.

    Returns
    -------
    np.ndarray
        Float64 array of unit-level weights, one per unit in ``unit_order``.
    """
    unit_w = data.groupby(unit_col)[survey_design.weights].first()
    return np.array([unit_w[u] for u in unit_order], dtype=np.float64)


def _resolve_survey_for_fit(survey_design, data, inference_mode="analytical"):
    """
    Shared helper: validate and resolve a SurveyDesign for an estimator fit() call.

    Returns (resolved, weights, weight_type, metadata) or all-None tuple if
    survey_design is None.
    """
    if survey_design is None:
        return None, None, "pweight", None

    if not isinstance(survey_design, SurveyDesign):
        raise TypeError("survey_design must be a SurveyDesign instance")

    if inference_mode == "wild_bootstrap":
        raise NotImplementedError(
            "Wild bootstrap with survey weights is not yet supported. "
            "Use inference='analytical' with survey_design, or see "
            "docs/survey-roadmap.md for planned Phase 5 support."
        )

    resolved = survey_design.resolve(data)
    raw_w = (
        data[survey_design.weights].values.astype(np.float64)
        if survey_design.weights
        else np.ones(len(data), dtype=np.float64)
    )
    metadata = compute_survey_metadata(resolved, raw_w)
    return resolved, resolved.weights, resolved.weight_type, metadata


def _resolve_effective_cluster(resolved_survey, cluster_ids, cluster_name=None):
    """
    Shared helper: determine effective cluster IDs for variance estimation.

    When survey PSU is present, it overrides the user-specified cluster.
    Warns if both are specified with different groupings.
    """
    if resolved_survey is None or resolved_survey.psu is None:
        return cluster_ids

    if cluster_ids is not None and cluster_name is not None:
        # Compare partition equivalence (not label equality)
        psu_codes, _ = pd.factorize(resolved_survey.psu)
        cluster_codes, _ = pd.factorize(cluster_ids)
        if not np.array_equal(psu_codes, cluster_codes):
            warnings.warn(
                f"Both survey_design.psu and cluster='{cluster_name}' specified "
                "with different groupings. PSU will be used for variance "
                "estimation (survey design-based inference).",
                UserWarning,
                stacklevel=3,
            )
    return resolved_survey.psu


def _inject_cluster_as_psu(resolved, cluster_ids):
    """
    When survey design has no PSU but cluster_ids are provided,
    inject cluster_ids as the effective PSU for TSL variance estimation.

    Returns a new ResolvedSurveyDesign (no mutation) or the original unchanged.
    """
    if resolved is None or cluster_ids is None:
        return resolved
    if resolved.psu is not None:
        return resolved  # PSU already present; _resolve_effective_cluster handles this

    # Validate no missing cluster IDs before factorization
    if pd.isna(cluster_ids).any():
        raise ValueError(
            "Cluster IDs contain missing values. "
            "All observations must have valid cluster identifiers "
            "when used as effective PSUs for survey variance estimation."
        )

    # When strata are present, make cluster IDs unique within strata
    # (same nesting logic as SurveyDesign.resolve() with nest=True)
    if resolved.strata is not None:
        combined = np.array([f"{s}_{c}" for s, c in zip(resolved.strata, cluster_ids)])
        codes, uniques = pd.factorize(combined)
    else:
        codes, uniques = pd.factorize(cluster_ids)
    n_clusters = len(uniques)

    return replace(resolved, psu=codes, n_psu=n_clusters)


def _compute_stratified_psu_meat(
    scores: np.ndarray,
    resolved: "ResolvedSurveyDesign",
) -> tuple:
    """Compute the stratified PSU-level meat matrix for TSL variance.

    This is the core computation shared by :func:`compute_survey_vcov`
    (which wraps it in a sandwich with the bread matrix) and
    :func:`compute_survey_if_variance` (which uses it directly for
    influence-function-based estimators).

    Parameters
    ----------
    scores : np.ndarray
        Score matrix of shape (n, k). For OLS-based estimators these are
        the weighted score contributions X_i * w_i * u_i.  For IF-based
        estimators these are the per-unit influence function values
        (reshaped to (n, 1) for scalar estimators).
    resolved : ResolvedSurveyDesign
        Resolved survey design with weights, strata, PSU arrays.

    Returns
    -------
    meat : np.ndarray
        Meat matrix of shape (k, k).
    variance_computed : bool
        Whether any actual variance computation happened.
    legitimate_zero_count : int
        Number of strata/sources that legitimately contribute zero variance.
    """
    n = scores.shape[0]
    k = scores.shape[1] if scores.ndim > 1 else 1
    if scores.ndim == 1:
        scores = scores[:, np.newaxis]

    strata = resolved.strata
    psu = resolved.psu

    legitimate_zero_count = 0
    _variance_computed = False

    if strata is None and psu is None:
        # No survey structure beyond weights — implicit per-observation PSUs
        psu_mean = scores.mean(axis=0, keepdims=True)
        centered = scores - psu_mean
        f_h = 0.0
        if resolved.fpc is not None:
            N_h = resolved.fpc[0]
            if N_h < n:
                raise ValueError(
                    f"FPC ({N_h}) is less than the number of observations "
                    f"({n}). FPC must be >= n_obs for implicit per-observation PSUs."
                )
            f_h = n / N_h
            if f_h >= 1.0:
                legitimate_zero_count += 1
        adjustment = (1.0 - f_h) * (n / (n - 1))
        meat = adjustment * (centered.T @ centered)
        _variance_computed = True
    elif strata is None and psu is not None:
        # No strata, but PSU present — single-stratum cluster-robust
        psu_scores = pd.DataFrame(scores).groupby(psu).sum().values
        n_psu = psu_scores.shape[0]

        if n_psu < 2:
            meat = np.zeros((k, k))
        else:
            psu_mean = psu_scores.mean(axis=0, keepdims=True)
            centered = psu_scores - psu_mean
            f_h = 0.0
            if resolved.fpc is not None:
                N_h = resolved.fpc[0]
                if N_h < n_psu:
                    raise ValueError(
                        f"FPC ({N_h}) is less than the number of effective PSUs "
                        f"({n_psu}). FPC must be >= n_PSU."
                    )
                f_h = n_psu / N_h
                if f_h >= 1.0:
                    legitimate_zero_count += 1
            adjustment = (1.0 - f_h) * (n_psu / (n_psu - 1))
            meat = adjustment * (centered.T @ centered)
            _variance_computed = True
    else:
        # Stratified with or without PSU
        unique_strata = np.unique(strata)
        meat = np.zeros((k, k))

        _global_psu_mean = None
        if resolved.lonely_psu == "adjust":
            if psu is not None:
                _global_psu_mean = (
                    pd.DataFrame(scores).groupby(psu).sum().values.mean(axis=0, keepdims=True)
                )
            else:
                _global_psu_mean = scores.mean(axis=0, keepdims=True)

        for h in unique_strata:
            mask_h = strata == h

            if psu is not None:
                psu_h = psu[mask_h]
                scores_h = scores[mask_h]
                psu_scores_h = pd.DataFrame(scores_h).groupby(psu_h).sum().values
                n_psu_h = psu_scores_h.shape[0]
            else:
                psu_scores_h = scores[mask_h]
                n_psu_h = psu_scores_h.shape[0]

            # Handle singleton strata
            if n_psu_h < 2:
                if resolved.lonely_psu == "remove":
                    continue
                elif resolved.lonely_psu == "certainty":
                    legitimate_zero_count += 1
                    continue
                elif resolved.lonely_psu == "adjust":
                    centered = psu_scores_h - _global_psu_mean
                    V_h = centered.T @ centered
                    meat += V_h
                    _variance_computed = True
                    continue

            # FPC
            f_h = 0.0
            if resolved.fpc is not None:
                N_h = resolved.fpc[mask_h][0]
                if N_h < n_psu_h:
                    raise ValueError(
                        f"FPC ({N_h}) is less than the number of effective PSUs "
                        f"({n_psu_h}) in stratum. FPC must be >= n_PSU."
                    )
                f_h = n_psu_h / N_h
                if f_h >= 1.0:
                    legitimate_zero_count += 1

            psu_mean_h = psu_scores_h.mean(axis=0, keepdims=True)
            centered = psu_scores_h - psu_mean_h

            adjustment = (1.0 - f_h) * (n_psu_h / (n_psu_h - 1))
            V_h = adjustment * (centered.T @ centered)
            meat += V_h
            _variance_computed = True

    return meat, _variance_computed, legitimate_zero_count


def compute_survey_vcov(
    X: np.ndarray,
    residuals: np.ndarray,
    resolved: "ResolvedSurveyDesign",
) -> np.ndarray:
    """
    Compute Taylor Series Linearization (TSL) variance-covariance matrix.

    Implements the stratified cluster sandwich estimator with optional
    finite population correction (FPC).

    V_TSL = (X'WX)^{-1} [sum_h V_h] (X'WX)^{-1}

    Parameters
    ----------
    X : np.ndarray
        Design matrix of shape (n, k).
    residuals : np.ndarray
        Residuals from WLS fit (y - X @ beta, on ORIGINAL scale).
    resolved : ResolvedSurveyDesign
        Resolved survey design with weights, strata, PSU arrays.

    Returns
    -------
    vcov : np.ndarray
        Variance-covariance matrix of shape (k, k).
    """
    n, k = X.shape
    weights = resolved.weights

    # Bread: (X'WX)^{-1}
    XtWX = X.T @ (X * weights[:, np.newaxis])

    # Compute weighted scores per observation: w_i * X_i * u_i
    if resolved.weight_type == "aweight":
        scores = X * residuals[:, np.newaxis]
    else:
        scores = X * (weights * residuals)[:, np.newaxis]

    meat, _variance_computed, legitimate_zero_count = _compute_stratified_psu_meat(scores, resolved)

    # Guard: if meat is zero, distinguish legitimate zero from unidentified variance
    if not np.any(meat != 0):
        if _variance_computed or legitimate_zero_count > 0:
            return np.zeros((k, k))
        return np.full((k, k), np.nan)

    # Sandwich: (X'WX)^{-1} meat (X'WX)^{-1}
    try:
        temp = np.linalg.solve(XtWX, meat)
        vcov = np.linalg.solve(XtWX, temp.T).T
    except np.linalg.LinAlgError as e:
        if "Singular" in str(e):
            raise ValueError(
                "Design matrix is rank-deficient (singular X'WX matrix). "
                "This indicates perfect multicollinearity."
            ) from e
        raise

    return vcov


def compute_survey_if_variance(
    psi: np.ndarray,
    resolved: "ResolvedSurveyDesign",
) -> float:
    """Compute design-based variance of a scalar estimator from IF values.

    For influence-function-based estimators (e.g., CallawaySantAnna),
    the per-unit influence function values ``psi_i`` capture each unit's
    contribution to the estimating equation.  Under simple random sampling
    the variance is ``sum(psi_i^2)``.  This function computes the
    design-based analogue accounting for PSU clustering, stratification,
    and finite population correction.

    V_design = sum_h (1-f_h) * (n_h/(n_h-1)) * sum_j (psi_hj - psi_h_bar)^2

    where psi_hj = sum_{i in PSU j, stratum h} psi_i.

    Parameters
    ----------
    psi : np.ndarray
        Per-unit influence function values, shape (n,).
    resolved : ResolvedSurveyDesign
        Resolved survey design.

    Returns
    -------
    float
        Design-based variance.  Returns ``np.nan`` when variance is
        unidentified (e.g., all strata removed by lonely_psu='remove').
    """
    psi = np.asarray(psi, dtype=np.float64).ravel()

    meat, _variance_computed, legitimate_zero_count = _compute_stratified_psu_meat(
        psi[:, np.newaxis], resolved
    )

    # meat is (1, 1) — extract scalar
    meat_scalar = float(meat[0, 0])

    if meat_scalar == 0.0:
        if _variance_computed or legitimate_zero_count > 0:
            return 0.0
        return np.nan

    return meat_scalar


def aggregate_to_psu(
    values: np.ndarray,
    resolved: "ResolvedSurveyDesign",
) -> tuple:
    """Sum values within PSUs for PSU-level bootstrap perturbation.

    Parameters
    ----------
    values : np.ndarray
        Per-observation values, shape (n,) or (n, k).
    resolved : ResolvedSurveyDesign
        Resolved survey design.

    Returns
    -------
    psu_sums : np.ndarray
        Aggregated values, shape (n_psu,) or (n_psu, k).
    psu_ids : np.ndarray
        Unique PSU identifiers in the same order as ``psu_sums``.
    """
    if resolved.psu is None:
        # Each observation is its own PSU — return as-is
        return values.copy(), np.arange(len(values))

    psu = resolved.psu
    unique_psu = np.unique(psu)
    if values.ndim == 1:
        psu_sums = np.array([values[psu == p].sum() for p in unique_psu])
    else:
        psu_sums = np.array([values[psu == p].sum(axis=0) for p in unique_psu])
    return psu_sums, unique_psu
