"""
Multiplier-bootstrap inference for the de Chaisemartin-D'Haultfoeuille (dCDH)
estimator.

The dCDH papers prescribe only the analytical cohort-recentered plug-in
variance from Web Appendix Section 3.7.3 of the dynamic companion paper.
This module adds an opt-in multiplier bootstrap clustered at the group
level, matching the inference convention used by ``CallawaySantAnna``,
``ImputationDiD``, and ``TwoStageDiD``. The bootstrap is a library
extension, not a paper requirement, and is documented as such in
``REGISTRY.md``.

The mixin operates on **pre-computed cohort-centered influence-function
values**: the main estimator class computes per-group ``U^G_g`` values
during the analytical variance calculation, recenters them by their
cohort means (using the ``(D_{g,1}, F_g, S_g)`` triple), and stores the
recentered vector. The bootstrap then multiplies this vector by random
multiplier weights (Rademacher / Mammen / Webb) and re-aggregates to
produce a bootstrap distribution per target.
"""

import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

from diff_diff.bootstrap_utils import (
    compute_effect_bootstrap_stats as _compute_effect_bootstrap_stats,
)
from diff_diff.bootstrap_utils import (
    generate_bootstrap_weights_batch as _generate_bootstrap_weights_batch,
)
from diff_diff.chaisemartin_dhaultfoeuille_results import DCDHBootstrapResults

__all__ = ["ChaisemartinDHaultfoeuilleBootstrapMixin"]


class ChaisemartinDHaultfoeuilleBootstrapMixin:
    """
    Bootstrap-inference mixin for ``ChaisemartinDHaultfoeuille``.

    Provides a single entry point ``_compute_dcdh_bootstrap`` that takes
    pre-computed centered influence-function values for each estimand
    target (overall ``DID_M``, joiners ``DID_+``, leavers ``DID_-``,
    placebo ``DID_M^pl``) and returns a populated
    :class:`DCDHBootstrapResults`.

    The mixin is pure (no instance state of its own); it only references
    instance attributes from the main class via ``TYPE_CHECKING`` hints.
    """

    # --- Type hints for attributes accessed from the main class ---
    n_bootstrap: int
    bootstrap_weights: str
    alpha: float
    seed: Optional[int]

    if TYPE_CHECKING:  # pragma: no cover

        def _placeholder(self) -> None: ...  # silences mypy "no attributes" warnings

    def _compute_dcdh_bootstrap(
        self,
        n_groups_for_overall: int,
        u_centered_overall: np.ndarray,
        n_groups_overall: int,
        original_overall: float,
        joiners_inputs: Optional[Tuple[np.ndarray, int, float]] = None,
        leavers_inputs: Optional[Tuple[np.ndarray, int, float]] = None,
        placebo_inputs: Optional[Tuple[np.ndarray, int, float]] = None,
    ) -> DCDHBootstrapResults:
        """
        Compute multiplier-bootstrap inference for all dCDH targets.

        Each target ``T`` is summarized by:

        - a centered influence-function vector of length equal to the
          number of groups contributing to ``T``
        - the count of contributing groups (used as the divisor when
          re-aggregating each bootstrap replicate)
        - the original point estimate of ``T`` (used as the centering
          point for the percentile p-value)

        For each target, this method:

        1. Generates an ``(n_bootstrap, n_groups_target)`` matrix of
           multiplier weights via
           :func:`~diff_diff.bootstrap_utils.generate_bootstrap_weights_batch`.
        2. Computes the bootstrap distribution as
           ``W @ u_centered / n_groups_target`` (one bootstrap replicate
           per row).
        3. Passes the distribution + the original point estimate through
           :func:`~diff_diff.bootstrap_utils.compute_effect_bootstrap_stats`
           to obtain ``(SE, CI, p_value)``.

        Parameters
        ----------
        n_groups_for_overall : int
            Number of groups contributing to the overall ``DID_M``
            (length of ``u_centered_overall``).
        u_centered_overall : np.ndarray
            Cohort-centered per-group influence-function values for
            ``DID_M``. Shape: ``(n_groups_for_overall,)``.
        n_groups_overall : int
            Divisor when re-aggregating each bootstrap replicate. For
            ``DID_M`` this is typically the count of switching groups
            ``N_S``-equivalent.
        original_overall : float
            The original point estimate of ``DID_M``. Used by
            :func:`compute_effect_bootstrap_stats` for the percentile
            p-value computation.
        joiners_inputs : tuple, optional
            ``(u_centered, n_groups, original_effect)`` triple for the
            joiners-only ``DID_+`` target. ``None`` when no joiners
            exist.
        leavers_inputs : tuple, optional
            Same triple for the leavers-only ``DID_-`` target.
        placebo_inputs : tuple, optional
            Same triple for the placebo ``DID_M^pl`` target.

        Returns
        -------
        DCDHBootstrapResults
            Populated bootstrap-results dataclass. Fields for unavailable
            targets (joiners / leavers / placebo) are ``None``.
        """
        if self.n_bootstrap <= 0:
            raise ValueError(
                f"_compute_dcdh_bootstrap called with n_bootstrap={self.n_bootstrap}; "
                "must be > 0."
            )
        if u_centered_overall.ndim != 1:
            raise ValueError(
                "u_centered_overall must be a 1-D array of per-group influence "
                f"function values, got shape {u_centered_overall.shape}"
            )
        if u_centered_overall.shape[0] != n_groups_for_overall:
            raise ValueError(
                f"u_centered_overall length ({u_centered_overall.shape[0]}) does not "
                f"match n_groups_for_overall ({n_groups_for_overall})"
            )
        if n_groups_overall <= 0:
            warnings.warn(
                f"_compute_dcdh_bootstrap: n_groups_overall={n_groups_overall} <= 0; "
                "returning all-NaN bootstrap results.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _empty_bootstrap_results(self.n_bootstrap, self.bootstrap_weights, self.alpha)

        rng = np.random.default_rng(self.seed)

        # --- Overall DID_M ---
        overall_se, overall_ci, overall_p, overall_dist = _bootstrap_one_target(
            u_centered=u_centered_overall,
            divisor=n_groups_overall,
            original=original_overall,
            n_bootstrap=self.n_bootstrap,
            weight_type=self.bootstrap_weights,
            alpha=self.alpha,
            rng=rng,
            context="dCDH overall DID_M bootstrap",
            return_distribution=True,
        )

        results = DCDHBootstrapResults(
            n_bootstrap=self.n_bootstrap,
            weight_type=self.bootstrap_weights,
            alpha=self.alpha,
            overall_se=overall_se,
            overall_ci=overall_ci,
            overall_p_value=overall_p,
            bootstrap_distribution=overall_dist,
        )

        # --- Joiners (DID_+) ---
        if joiners_inputs is not None:
            u_j, n_j, eff_j = joiners_inputs
            if u_j.size > 0 and n_j > 0:
                se_j, ci_j, p_j, _ = _bootstrap_one_target(
                    u_centered=u_j,
                    divisor=n_j,
                    original=eff_j,
                    n_bootstrap=self.n_bootstrap,
                    weight_type=self.bootstrap_weights,
                    alpha=self.alpha,
                    rng=rng,
                    context="dCDH joiners DID_+ bootstrap",
                    return_distribution=False,
                )
                results.joiners_se = se_j
                results.joiners_ci = ci_j
                results.joiners_p_value = p_j

        # --- Leavers (DID_-) ---
        if leavers_inputs is not None:
            u_l, n_l, eff_l = leavers_inputs
            if u_l.size > 0 and n_l > 0:
                se_l, ci_l, p_l, _ = _bootstrap_one_target(
                    u_centered=u_l,
                    divisor=n_l,
                    original=eff_l,
                    n_bootstrap=self.n_bootstrap,
                    weight_type=self.bootstrap_weights,
                    alpha=self.alpha,
                    rng=rng,
                    context="dCDH leavers DID_- bootstrap",
                    return_distribution=False,
                )
                results.leavers_se = se_l
                results.leavers_ci = ci_l
                results.leavers_p_value = p_l

        # --- Placebo (DID_M^pl) ---
        if placebo_inputs is not None:
            u_pl, n_pl, eff_pl = placebo_inputs
            if u_pl.size > 0 and n_pl > 0:
                se_pl, ci_pl, p_pl, _ = _bootstrap_one_target(
                    u_centered=u_pl,
                    divisor=n_pl,
                    original=eff_pl,
                    n_bootstrap=self.n_bootstrap,
                    weight_type=self.bootstrap_weights,
                    alpha=self.alpha,
                    rng=rng,
                    context="dCDH placebo DID_M^pl bootstrap",
                    return_distribution=False,
                )
                results.placebo_se = se_pl
                results.placebo_ci = ci_pl
                results.placebo_p_value = p_pl

        return results


# =============================================================================
# Internal helpers
# =============================================================================


def _bootstrap_one_target(
    u_centered: np.ndarray,
    divisor: int,
    original: float,
    n_bootstrap: int,
    weight_type: str,
    alpha: float,
    rng: np.random.Generator,
    context: str,
    return_distribution: bool,
) -> Tuple[float, Tuple[float, float], float, Optional[np.ndarray]]:
    """
    Run the multiplier bootstrap for a single dCDH target.

    Generates an ``(n_bootstrap, len(u_centered))`` matrix of multiplier
    weights, multiplies by ``u_centered``, and divides by ``divisor`` to
    get a bootstrap distribution. Returns
    ``(se, (ci_lo, ci_hi), p_value, distribution)``; ``distribution`` is
    ``None`` when ``return_distribution=False`` (saves memory for
    auxiliary targets).

    The "centered" naming is important: this function expects
    ``u_centered`` to already have its cohort means subtracted (so the
    sample mean of the bootstrap distribution should be approximately
    zero, not the original effect). The original effect is passed
    separately as the centering point for the percentile p-value.
    """
    n_groups_target = u_centered.shape[0]
    if n_groups_target == 0 or divisor == 0:
        return np.nan, (np.nan, np.nan), np.nan, None

    weight_matrix = _generate_bootstrap_weights_batch(
        n_bootstrap=n_bootstrap,
        n_units=n_groups_target,
        weight_type=weight_type,
        rng=rng,
    )

    # Each bootstrap replicate: (1 / divisor) * sum_g w_b[g] * u_centered[g]
    # The result is the bootstrap analog of the *deviation* from the original
    # effect; shift it by `original` so the bootstrap distribution is centered
    # at the original point estimate (which is what compute_effect_bootstrap_stats
    # expects when computing the percentile p-value).
    deviations = (weight_matrix @ u_centered) / divisor
    boot_dist = original + deviations

    se, ci, p_value = _compute_effect_bootstrap_stats(
        original_effect=original,
        boot_dist=boot_dist,
        alpha=alpha,
        context=context,
    )

    return se, ci, p_value, (boot_dist if return_distribution else None)


def _empty_bootstrap_results(
    n_bootstrap: int, weight_type: str, alpha: float
) -> DCDHBootstrapResults:
    """Return an all-NaN bootstrap results object as a graceful fallback."""
    return DCDHBootstrapResults(
        n_bootstrap=n_bootstrap,
        weight_type=weight_type,
        alpha=alpha,
        overall_se=np.nan,
        overall_ci=(np.nan, np.nan),
        overall_p_value=np.nan,
    )
