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

            # fweight validation: should be positive integers
            if self.weight_type == "fweight":
                fractional = raw_weights - np.round(raw_weights)
                if np.any(np.abs(fractional) > 1e-10):
                    warnings.warn(
                        "Frequency weights (fweight) should be positive integers. "
                        "Fractional values detected; rounding will not be applied.",
                        UserWarning,
                        stacklevel=2,
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
            strata_arr = _factorize_cluster_ids(data[self.strata].values)
            n_strata = len(np.unique(strata_arr))

        # --- PSU ---
        psu_arr = None
        n_psu = 0
        if self.psu is not None:
            if self.psu not in data.columns:
                raise ValueError(f"PSU column '{self.psu}' not found in data")
            psu_raw = data[self.psu].values

            if self.nest and strata_arr is not None:
                # Make PSU IDs unique within strata by combining
                combined = np.array([f"{s}_{p}" for s, p in zip(strata_arr, psu_raw)])
                psu_arr = _factorize_cluster_ids(combined)
            else:
                psu_arr = _factorize_cluster_ids(psu_raw)

            n_psu = len(np.unique(psu_arr))

        # --- FPC ---
        fpc_arr = None
        if self.fpc is not None:
            if self.fpc not in data.columns:
                raise ValueError(f"FPC column '{self.fpc}' not found in data")
            fpc_arr = data[self.fpc].values.astype(np.float64)

            if np.any(np.isnan(fpc_arr)) or np.any(~np.isfinite(fpc_arr)):
                raise ValueError("FPC values must be finite and non-NaN")

            # FPC requires survey structure (psu or strata)
            if self.psu is None and self.strata is None:
                raise ValueError(
                    "FPC requires either psu or strata to be specified. "
                    "FPC alone without survey structure is not supported."
                )

            # Validate FPC >= n_h per stratum
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
                    # Validate FPC >= number of sampled PSUs (not obs count)
                    if psu_arr is not None:
                        n_psu_h = len(np.unique(psu_arr[mask_h]))
                        if fpc_h < n_psu_h:
                            raise ValueError(
                                f"FPC ({fpc_h}) is less than the number of PSUs "
                                f"({n_psu_h}) in stratum {h}. FPC must be >= n_PSU."
                            )
                    else:
                        n_h = np.sum(mask_h)
                        if fpc_h < n_h:
                            raise ValueError(
                                f"FPC ({fpc_h}) is less than the number of observations "
                                f"({n_h}) in stratum {h}. FPC must be >= n_obs."
                            )
            elif psu_arr is not None:
                # No strata: require FPC is a single constant value
                if len(np.unique(fpc_arr)) > 1:
                    raise ValueError(
                        "FPC values must be constant when no strata are specified. "
                        f"Found {len(np.unique(fpc_arr))} distinct values."
                    )
                if fpc_arr[0] < n_psu:
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

    # Factorize cluster_ids for consistent integer encoding
    codes, uniques = pd.factorize(cluster_ids)
    n_clusters = len(uniques)

    return replace(resolved, psu=codes, n_psu=n_clusters)


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
        # aweights: no weight in meat (errors already homoskedastic after WLS)
        scores = X * residuals[:, np.newaxis]
    else:
        scores = X * (weights * residuals)[:, np.newaxis]

    # Determine strata and PSU structure
    strata = resolved.strata
    psu = resolved.psu

    if strata is None and psu is None:
        # No survey structure beyond weights — use implicit per-observation PSUs
        # so the TSL construction is consistent across all branches.
        # Each observation is its own PSU; scores are already per-obs.
        psu_mean = scores.mean(axis=0, keepdims=True)
        centered = scores - psu_mean
        adjustment = n / (n - 1)
        meat = adjustment * (centered.T @ centered)
    elif strata is None and psu is not None:
        # No strata, but PSU present — single-stratum cluster-robust
        psu_scores = pd.DataFrame(scores).groupby(psu).sum().values
        n_psu = psu_scores.shape[0]

        if n_psu < 2:
            # With only 1 PSU and no strata, variance estimation is impossible
            # regardless of lonely_psu mode. The "adjust" mode cannot help
            # because there is no global-vs-stratum distinction to exploit.
            meat = np.zeros((k, k))
        else:
            # Center around grand mean
            psu_mean = psu_scores.mean(axis=0, keepdims=True)
            centered = psu_scores - psu_mean
            f_h = 0.0  # No FPC
            if resolved.fpc is not None:
                N_h = resolved.fpc[0]
                f_h = n_psu / N_h
            adjustment = (1.0 - f_h) * (n_psu / (n_psu - 1))
            meat = adjustment * (centered.T @ centered)
    else:
        # Stratified with or without PSU
        unique_strata = np.unique(strata)
        meat = np.zeros((k, k))

        # Pre-compute global PSU scores for lonely_psu="adjust" (avoids
        # recomputing O(n) groupby inside the per-stratum loop)
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
                # Get PSU-level score totals within stratum h
                psu_h = psu[mask_h]
                scores_h = scores[mask_h]
                psu_scores_h = pd.DataFrame(scores_h).groupby(psu_h).sum().values
                n_psu_h = psu_scores_h.shape[0]
            else:
                # Each observation is its own PSU
                psu_scores_h = scores[mask_h]
                n_psu_h = psu_scores_h.shape[0]

            # Handle singleton strata
            if n_psu_h < 2:
                if resolved.lonely_psu == "remove":
                    continue  # Skip this stratum
                elif resolved.lonely_psu == "certainty":
                    continue  # f_h = 1, so (1-f_h) = 0, zero contribution
                elif resolved.lonely_psu == "adjust":
                    # Center around overall mean instead of stratum mean
                    centered = psu_scores_h - _global_psu_mean
                    V_h = centered.T @ centered
                    meat += V_h
                    continue

            # FPC
            f_h = 0.0
            if resolved.fpc is not None:
                N_h = resolved.fpc[mask_h][0]
                f_h = n_psu_h / N_h

            # Stratum mean of PSU scores
            psu_mean_h = psu_scores_h.mean(axis=0, keepdims=True)
            centered = psu_scores_h - psu_mean_h

            # V_h = (1 - f_h) * (n_h / (n_h - 1)) * sum (e_hi - e_bar_h)(...)^T
            adjustment = (1.0 - f_h) * (n_psu_h / (n_psu_h - 1))
            V_h = adjustment * (centered.T @ centered)
            meat += V_h

    # Guard: if no stratum contributed variance, return NaN vcov
    if not np.any(meat != 0):
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
