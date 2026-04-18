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

from typing import TYPE_CHECKING, Dict, Optional, Tuple

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
        divisor_overall: int,
        original_overall: float,
        joiners_inputs: Optional[Tuple[np.ndarray, int, float]] = None,
        leavers_inputs: Optional[Tuple[np.ndarray, int, float]] = None,
        placebo_inputs: Optional[Tuple[np.ndarray, int, float]] = None,
        # --- Phase 2: multi-horizon inputs ---
        multi_horizon_inputs: Optional[Dict[int, Tuple[np.ndarray, int, float]]] = None,
        placebo_horizon_inputs: Optional[Dict[int, Tuple[np.ndarray, int, float]]] = None,
        # --- Survey: PSU-level bootstrap under survey designs ---
        group_to_psu_map: Optional[np.ndarray] = None,
    ) -> DCDHBootstrapResults:
        """
        Compute multiplier-bootstrap inference for all dCDH targets.

        Each target ``T`` is summarized by:

        - a centered influence-function vector of length equal to the
          number of groups contributing to ``T``
        - a re-aggregation **divisor**, which is the *switching-cell*
          count from the Theorem 3 weighting formula (NOT a group
          count). For ``DID_M`` the divisor is ``N_S = sum_t (N_{1,0,t}
          + N_{0,1,t})``; for ``DID_+`` it is ``sum_t N_{1,0,t}``; for
          ``DID_-`` it is ``sum_t N_{0,1,t}``. See REGISTRY.md
          ``ChaisemartinDHaultfoeuille`` for the cell-count weighting
          contract.
        - the original point estimate of ``T`` (used as the centering
          point for the percentile p-value)

        For each target, this method:

        1. Generates an ``(n_bootstrap, n_groups_target)`` matrix of
           multiplier weights via
           :func:`~diff_diff.bootstrap_utils.generate_bootstrap_weights_batch`,
           where ``n_groups_target`` is the IF vector length (one
           weight per contributing group).
        2. Computes the bootstrap distribution as
           ``W @ u_centered / divisor`` (one bootstrap replicate per
           row), where ``divisor`` is the switching-cell count
           described above. Note: the weight matrix has one column per
           contributing group, but the divisor is a cell count — the
           two are different quantities (groups can contribute to
           multiple cells across periods).
        3. Passes the distribution + the original point estimate through
           :func:`~diff_diff.bootstrap_utils.compute_effect_bootstrap_stats`
           to obtain ``(SE, CI, p_value)``.

        Parameters
        ----------
        n_groups_for_overall : int
            Number of groups contributing to the overall ``DID_M``
            (length of ``u_centered_overall``). Used for shape
            validation and weight-matrix sizing.
        u_centered_overall : np.ndarray
            Cohort-centered per-group influence-function values for
            ``DID_M``. Shape: ``(n_groups_for_overall,)``.
        divisor_overall : int
            Re-aggregation **divisor** for ``DID_M`` — the switching-
            cell count ``N_S = sum_t (N_{1,0,t} + N_{0,1,t})`` from
            Theorem 3 of AER 2020. NOT a group count. For Phase 1
            this is the same value used in the analytical SE plug-in.
        original_overall : float
            The original point estimate of ``DID_M``. Used by
            :func:`compute_effect_bootstrap_stats` for the percentile
            p-value computation.
        joiners_inputs : tuple, optional
            ``(u_centered, divisor, original_effect)`` triple for the
            joiners-only ``DID_+`` target. The ``divisor`` is the
            joiner switching-cell total ``sum_t N_{1,0,t}``, NOT the
            joiner group count. ``None`` when no joiners exist.
        leavers_inputs : tuple, optional
            Same triple for the leavers-only ``DID_-`` target. The
            ``divisor`` is the leaver switching-cell total
            ``sum_t N_{0,1,t}``.
        placebo_inputs : tuple, optional
            Same triple for the Phase 1 per-period placebo ``DID_M^pl``.
            ``None`` when ``L_max=None`` (per-period placebo has no IF).

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
        rng = np.random.default_rng(self.seed)

        # --- Overall DID_M ---
        # Skip the scalar DID_M bootstrap when divisor_overall <= 0
        # (e.g., pure non-binary panels where N_S=0), but continue
        # to process multi_horizon_inputs and placebo_horizon_inputs.
        if divisor_overall > 0:
            overall_se, overall_ci, overall_p, overall_dist = _bootstrap_one_target(
                u_centered=u_centered_overall,
                divisor=divisor_overall,
                original=original_overall,
                n_bootstrap=self.n_bootstrap,
                weight_type=self.bootstrap_weights,
                alpha=self.alpha,
                rng=rng,
                context="dCDH overall DID_M bootstrap",
                return_distribution=True,
                group_to_psu_map=group_to_psu_map,
            )
        else:
            overall_se = np.nan
            overall_ci = (np.nan, np.nan)
            overall_p = np.nan
            overall_dist = None

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
                    group_to_psu_map=_slice_psu_map(group_to_psu_map, u_j.size),
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
                    group_to_psu_map=_slice_psu_map(group_to_psu_map, u_l.size),
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
                    group_to_psu_map=_slice_psu_map(group_to_psu_map, u_pl.size),
                )
                results.placebo_se = se_pl
                results.placebo_ci = ci_pl
                results.placebo_p_value = p_pl

        # --- Phase 2: Multi-horizon bootstrap with shared weight matrix ---
        # Generate ONE shared (n_bootstrap, n_groups) weight matrix so all
        # horizons use the same bootstrap draw, making the sup-t statistic
        # a valid joint multiplier-bootstrap band.
        if multi_horizon_inputs is not None:
            es_ses: Dict[int, float] = {}
            es_cis: Dict[int, Tuple[float, float]] = {}
            es_pvals: Dict[int, float] = {}
            es_dists: Dict[int, np.ndarray] = {}

            # Shared weight matrix sized for the group set. Under PSU-level
            # bootstrap (Hall-Mammen wild PSU), weights are drawn once per
            # PSU and broadcast to groups so all groups in the same PSU
            # share a multiplier within a single bootstrap replicate —
            # preserving the sup-t joint distribution across horizons.
            n_groups_mh = n_groups_for_overall
            shared_weights = _generate_psu_or_group_weights(
                n_bootstrap=self.n_bootstrap,
                n_groups_target=n_groups_mh,
                weight_type=self.bootstrap_weights,
                rng=rng,
                group_to_psu_map=group_to_psu_map,
            )

            for l_h, (u_h, n_h, eff_h) in sorted(multi_horizon_inputs.items()):
                if u_h.size > 0 and n_h > 0:
                    # Use the shared weight matrix truncated to u_h length
                    w_h = shared_weights[:, : u_h.size]
                    deviations = (w_h @ u_h) / n_h
                    dist_h = deviations + eff_h

                    se_h, ci_h, p_h = _compute_effect_bootstrap_stats(
                        original_effect=eff_h,
                        boot_dist=dist_h,
                        alpha=self.alpha,
                    )
                    es_ses[l_h] = se_h
                    es_cis[l_h] = ci_h
                    es_pvals[l_h] = p_h
                    es_dists[l_h] = dist_h

            results.event_study_ses = es_ses
            results.event_study_cis = es_cis
            results.event_study_p_values = es_pvals

            # Sup-t simultaneous confidence bands using the shared draws.
            valid_horizons = [
                l_h
                for l_h in es_dists
                if l_h in es_ses and np.isfinite(es_ses[l_h]) and es_ses[l_h] > 0
            ]
            if len(valid_horizons) >= 2:
                boot_matrix = np.array([es_dists[l_h] for l_h in valid_horizons])
                effects_vec = np.array([multi_horizon_inputs[l_h][2] for l_h in valid_horizons])
                ses_vec = np.array([es_ses[l_h] for l_h in valid_horizons])
                t_stats = np.abs((boot_matrix - effects_vec[:, None]) / ses_vec[:, None])
                sup_t_dist = np.max(t_stats, axis=0)
                finite_mask = np.isfinite(sup_t_dist)
                if finite_mask.sum() > 0.5 * self.n_bootstrap:
                    cband_crit = float(np.quantile(sup_t_dist[finite_mask], 1 - self.alpha))
                    results.cband_crit_value = cband_crit

        # --- Phase 2: Placebo horizon bootstrap ---
        if placebo_horizon_inputs is not None:
            pl_ses: Dict[int, float] = {}
            pl_cis: Dict[int, Tuple[float, float]] = {}
            pl_pvals: Dict[int, float] = {}

            for l_h, (u_h, n_h, eff_h) in sorted(placebo_horizon_inputs.items()):
                if u_h.size > 0 and n_h > 0:
                    se_h, ci_h, p_h, _ = _bootstrap_one_target(
                        u_centered=u_h,
                        divisor=n_h,
                        original=eff_h,
                        n_bootstrap=self.n_bootstrap,
                        weight_type=self.bootstrap_weights,
                        alpha=self.alpha,
                        rng=rng,
                        context=f"dCDH placebo l={l_h} bootstrap",
                        return_distribution=False,
                        group_to_psu_map=_slice_psu_map(group_to_psu_map, u_h.size),
                    )
                    pl_ses[l_h] = se_h
                    pl_cis[l_h] = ci_h
                    pl_pvals[l_h] = p_h

            results.placebo_horizon_ses = pl_ses
            results.placebo_horizon_cis = pl_cis
            results.placebo_horizon_p_values = pl_pvals

        return results


# =============================================================================
# Internal helpers
# =============================================================================


def _slice_psu_map(
    group_to_psu_map: Optional[np.ndarray],
    target_size: int,
) -> Optional[np.ndarray]:
    """Return a PSU map aligned to a subset bootstrap target.

    The dCDH bootstrap passes a single full-length ``group_to_psu_map``
    built in ``fit()`` from ``_eligible_group_ids`` (the variance-
    eligible group ordering). By construction all current bootstrap
    targets (overall, joiners, leavers, multi-horizon DID_l, placebo
    DID^pl_l) use that same ordering — each target's IF vector has the
    same length as the map and the entries correspond to the same
    groups in the same order. This invariant is enforced by requiring
    ``target_size == len(group_to_psu_map)``; if a future refactor
    introduces a target whose group subset differs from the overall
    variance-eligible ordering, this assertion fires loudly rather
    than silently misclustering the bootstrap draws.

    Returns ``None`` when no map is provided (plain multiplier-
    bootstrap path — identity across targets). Raises ``ValueError``
    when the target size does not match the map length: callers must
    supply a target-specific map (not a truncation) if they introduce
    non-aligned subsets.
    """
    if group_to_psu_map is None:
        return None
    if target_size == len(group_to_psu_map):
        return group_to_psu_map
    raise ValueError(
        f"PSU map length ({len(group_to_psu_map)}) does not match "
        f"bootstrap target size ({target_size}). dCDH's bootstrap contract "
        f"requires all targets to use the same variance-eligible group "
        f"ordering as `_eligible_group_ids`. If this target has a "
        f"different ordering, construct a target-specific map keyed by "
        f"its actual group IDs rather than truncating the full map."
    )


def _generate_psu_or_group_weights(
    n_bootstrap: int,
    n_groups_target: int,
    weight_type: str,
    rng: np.random.Generator,
    group_to_psu_map: Optional[np.ndarray],
) -> np.ndarray:
    """Generate a group-level weight matrix, possibly via PSU broadcasting.

    When ``group_to_psu_map`` is ``None`` or is the identity (each group
    is its own PSU), generates weights at the group level directly —
    bit-identical to the pre-PSU-bootstrap contract.

    When ``group_to_psu_map`` has fewer unique values than
    ``n_groups_target`` (strictly coarser PSU than group), generates
    weights at the PSU level and broadcasts to groups via the map.
    This is the Hall-Mammen wild PSU bootstrap.

    Parameters
    ----------
    n_bootstrap, weight_type, rng
        Passed through to generate_bootstrap_weights_batch.
    n_groups_target : int
        Number of groups contributing to the target's IF vector.
    group_to_psu_map : np.ndarray or None
        Dense integer PSU indices of shape ``(n_groups_target,)``.
        ``None`` triggers the group-level path.

    Returns
    -------
    np.ndarray
        Shape ``(n_bootstrap, n_groups_target)`` multiplier weights.
    """
    if group_to_psu_map is None:
        return _generate_bootstrap_weights_batch(
            n_bootstrap=n_bootstrap,
            n_units=n_groups_target,
            weight_type=weight_type,
            rng=rng,
        )
    if len(group_to_psu_map) != n_groups_target:
        raise ValueError(
            f"group_to_psu_map length ({len(group_to_psu_map)}) does not "
            f"match n_groups_target ({n_groups_target})."
        )
    n_psu = int(np.max(group_to_psu_map)) + 1 if group_to_psu_map.size > 0 else 0
    if n_psu >= n_groups_target:
        # Identity (each group its own PSU) — skip the broadcast for a
        # bit-identical fast path matching the pre-PSU-bootstrap behavior.
        return _generate_bootstrap_weights_batch(
            n_bootstrap=n_bootstrap,
            n_units=n_groups_target,
            weight_type=weight_type,
            rng=rng,
        )
    # Hall-Mammen wild PSU bootstrap: draw n_psu multipliers, broadcast
    # via the dense index map so all groups in the same PSU share a
    # multiplier. Preserves clustered sampling structure.
    psu_weights = _generate_bootstrap_weights_batch(
        n_bootstrap=n_bootstrap,
        n_units=n_psu,
        weight_type=weight_type,
        rng=rng,
    )
    return psu_weights[:, group_to_psu_map]


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
    group_to_psu_map: Optional[np.ndarray] = None,
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

    When ``group_to_psu_map`` is provided (length ``len(u_centered)``,
    dense integer PSU indices), multiplier weights are generated at the
    PSU level and broadcast to groups so all groups in the same PSU
    receive the same bootstrap multiplier. This is the Hall-Mammen wild
    PSU bootstrap; it reduces to the group-level bootstrap when each
    group is its own PSU (identity map).
    """
    n_groups_target = u_centered.shape[0]
    if n_groups_target == 0 or divisor == 0:
        return np.nan, (np.nan, np.nan), np.nan, None

    weight_matrix = _generate_psu_or_group_weights(
        n_bootstrap=n_bootstrap,
        n_groups_target=n_groups_target,
        weight_type=weight_type,
        rng=rng,
        group_to_psu_map=group_to_psu_map,
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


