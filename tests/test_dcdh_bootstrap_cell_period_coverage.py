"""Monte Carlo coverage simulation for the cell-level wild PSU bootstrap
in ChaisemartinDHaultfoeuille (PR 4).

Validates empirical coverage of the bootstrap confidence intervals under
a DGP with PSU that varies across cells of the same group — the regime
unlocked by PR 4's replacement of the group-level PSU map with a
cell-level map. Under PSU-within-group-constant the dispatcher routes
through the legacy bootstrap (covered by the pre-PR-4 test suite), so
the coverage check here exercises only the new cell-level code path.

Asserts coverage at TWO surfaces:

1. Overall DID_M bootstrap CI (`res.bootstrap_results.overall_ci`).
2. Event-study horizon CIs (`res.bootstrap_results.event_study_cis`) —
   this is the highest-risk surface per the PR 4 plan review's
   CRITICAL #2 (shared-PSU-weight matrix must be drawn once per
   multi-horizon block to preserve the sup-t joint distribution).
   Horizon-specific coverage regresses on any bug in the shared-
   weight machinery that a single-surface test would miss.

Marked ``slow`` and excluded from the default pytest run. To execute:

    pytest tests/test_dcdh_bootstrap_cell_period_coverage.py -m slow -v
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from diff_diff.chaisemartin_dhaultfoeuille import ChaisemartinDHaultfoeuille
from diff_diff.survey import SurveyDesign


def _simulate_panel(
    n_groups: int,
    n_periods: int,
    first_treated_period: int,
    tau: float,
    rng: np.random.Generator,
    psu_sigma: float = 0.5,
    obs_sigma: float = 1.0,
    group_sigma: float = 1.0,
) -> pd.DataFrame:
    """Generate a one-obs-per-cell panel with within-group-varying PSU
    (PSU = period parity per group) so the cell-level bootstrap is
    exercised. Matches the DGP shape of `test_dcdh_cell_period_coverage.py`.
    """
    groups = np.arange(n_groups)
    treated = groups < n_groups // 2
    group_fe = rng.normal(0.0, group_sigma, size=n_groups)
    # (group, psu) effect — constant within (g, parity), drives
    # within-group residual correlation that Binder variance + wild
    # PSU bootstrap should capture.
    psu_fe = rng.normal(0.0, psu_sigma, size=(n_groups, 2))

    rows = []
    for g in groups:
        for t in range(n_periods):
            parity = int(t % 2)
            # Per-(group, parity) PSU codes: 2 * n_groups distinct PSUs
            # so Binder / bootstrap see cell granularity rather than
            # two global PSUs reused across groups.
            psu_id = int(g) * 2 + parity
            d = 1 if (treated[g] and t >= first_treated_period) else 0
            y = (
                group_fe[g]
                + 0.1 * t
                + tau * d
                + psu_fe[g, parity]
                + rng.normal(0.0, obs_sigma)
            )
            rows.append({
                "group": int(g),
                "period": int(t),
                "treatment": int(d),
                "outcome": float(y),
                "psu": psu_id,
                "pw": 1.0,
            })
    return pd.DataFrame(rows)


@pytest.mark.slow
def test_bootstrap_cell_period_coverage_varying_psu():
    """Empirical 95% coverage for bootstrap CIs at BOTH the overall
    DID_M and per-horizon event-study surfaces on a DGP with
    within-group-varying PSU. Tolerance ±2.5pp mirrors the analytical
    coverage test; per-horizon coverage guards the shared-PSU-weight
    machinery from CRITICAL #2 of the PR 4 plan review.
    """
    n_reps = 500
    n_groups = 40
    n_periods = 6
    first_treated_period = 3
    tau_true = 2.0

    rng = np.random.default_rng(20260419)
    covered_overall = 0
    covered_h1 = 0
    failed = 0

    for r in range(n_reps):
        df = _simulate_panel(
            n_groups=n_groups,
            n_periods=n_periods,
            first_treated_period=first_treated_period,
            tau=tau_true,
            rng=rng,
        )
        sd = SurveyDesign(weights="pw", psu="psu")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # n_bootstrap=500 keeps internal bootstrap Monte-Carlo
                # noise well below the across-reps variance (at B=500
                # the percentile-CI endpoints are stable to ~0.3pp per
                # Efron-Tibshirani §13.3), so the across-reps coverage
                # mostly reflects the sampling-distribution / bootstrap-
                # consistency question rather than bootstrap MC noise.
                res = ChaisemartinDHaultfoeuille(
                    n_bootstrap=500, seed=r + 1,
                ).fit(
                    df,
                    outcome="outcome", group="group",
                    time="period", treatment="treatment",
                    survey_design=sd, L_max=1,
                )
        except Exception:
            failed += 1
            continue

        if res.bootstrap_results is None:
            failed += 1
            continue

        # Overall bootstrap CI.
        overall_ci = res.bootstrap_results.overall_ci
        if overall_ci is None or not all(np.isfinite(overall_ci)):
            failed += 1
            continue
        lo_o, hi_o = float(overall_ci[0]), float(overall_ci[1])
        if lo_o <= tau_true <= hi_o:
            covered_overall += 1

        # Horizon-1 bootstrap CI (guards the shared-PSU-weight path).
        es_cis = res.bootstrap_results.event_study_cis
        if es_cis is None or 1 not in es_cis:
            continue
        h1_ci = es_cis[1]
        if h1_ci is None or not all(np.isfinite(h1_ci)):
            continue
        lo_h, hi_h = float(h1_ci[0]), float(h1_ci[1])
        if lo_h <= tau_true <= hi_h:
            covered_h1 += 1

    completed = n_reps - failed
    assert completed >= int(0.95 * n_reps), (
        f"MC simulation had {failed}/{n_reps} fit failures, above "
        f"the 5% tolerance."
    )
    coverage_overall = covered_overall / completed
    coverage_h1 = covered_h1 / completed
    assert 0.925 <= coverage_overall <= 0.975, (
        f"Overall bootstrap CI coverage {coverage_overall:.3f} "
        f"(completed {completed}) outside [0.925, 0.975]; "
        f"tau_true={tau_true}, n_groups={n_groups}, n_periods={n_periods}, "
        f"n_reps={n_reps}."
    )
    assert 0.925 <= coverage_h1 <= 0.975, (
        f"Horizon-1 event-study bootstrap CI coverage "
        f"{coverage_h1:.3f} (completed {completed}) outside "
        f"[0.925, 0.975]; this is the shared-PSU-weight surface, "
        f"regression here likely indicates a bug in the multi-horizon "
        f"cell-level broadcast."
    )
