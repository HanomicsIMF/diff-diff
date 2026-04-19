"""
Scenario 1: Staggered marketing campaign.

CallawaySantAnna with covariates + bootstrap + aggregate='all', wrapped in
the 8-step Baker workflow: Bacon -> CS fit -> event-study pre-trend
inspection -> HonestDiD M-grid -> SunAbraham + ImputationDiD robustness
-> with/without-covariates refit -> practitioner_next_steps.

Data shape: 150 DMAs x 26 weekly periods, 2 staggered cohorts, 2 covariates.
"""

import numpy as np
import pandas as pd

from diff_diff import (
    BaconDecomposition,
    CallawaySantAnna,
    ImputationDiD,
    SunAbraham,
    compute_honest_did,
    practitioner_next_steps,
)
from diff_diff.prep import generate_staggered_data

from bench_shared import run_scenario


def build_data(seed=42):
    df = generate_staggered_data(
        n_units=150, n_periods=26, cohort_periods=[9, 14],
        never_treated_frac=0.3, treatment_effect=3.0,
        dynamic_effects=True, effect_growth=0.1, seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    unit_log_pop = pd.Series(
        rng.normal(0, 1, size=df["unit"].nunique()),
        index=sorted(df["unit"].unique()),
    )
    df["log_pop"] = df["unit"].map(unit_log_pop)
    df["baseline_spend"] = rng.normal(0, 1, size=len(df))
    return df


def main():
    data = build_data()
    covars = ["log_pop", "baseline_spend"]
    fit_kwargs = dict(
        data=data, outcome="outcome", unit="unit", time="period",
        first_treat="first_treat",
    )

    results = {}

    def bacon():
        results["bacon"] = BaconDecomposition().fit(
            data, outcome="outcome", unit="unit", time="period",
            first_treat="first_treat",
        )

    def cs_fit():
        cs = CallawaySantAnna(
            control_group="never_treated", estimation_method="dr",
            cluster="unit", n_bootstrap=999, seed=123,
        )
        results["cs"] = cs.fit(
            **fit_kwargs, covariates=covars, aggregate="all",
        )

    def inspect_pretrends():
        es = results["cs"].event_study_effects or {}
        results["pretrends"] = {
            rel_t: eff for rel_t, eff in es.items() if rel_t < 0
        }

    def honest_did_grid():
        out = {}
        for M in (0.5, 1.0, 1.5, 2.0):
            out[M] = compute_honest_did(
                results["cs"], method="relative_magnitude", M=M,
            )
        results["honest"] = out

    def sun_abraham():
        sa = SunAbraham(control_group="never_treated", cluster="unit")
        results["sa"] = sa.fit(**fit_kwargs)

    def imputation():
        bjs = ImputationDiD(cluster="unit")
        results["bjs"] = bjs.fit(**fit_kwargs, aggregate="event_study")

    def cs_no_covariates():
        cs = CallawaySantAnna(
            control_group="never_treated", estimation_method="reg",
            cluster="unit", n_bootstrap=199, seed=123,
        )
        results["cs_nocov"] = cs.fit(**fit_kwargs, aggregate="all")

    def next_steps():
        results["guidance"] = practitioner_next_steps(results["cs"])

    phases = [
        ("1_bacon_decomposition", bacon),
        ("2_cs_fit_with_covariates_bootstrap999", cs_fit),
        ("3_inspect_pretrends", inspect_pretrends),
        ("4_honest_did_M_grid", honest_did_grid),
        ("5_sun_abraham_robustness", sun_abraham),
        ("6_imputation_did_robustness", imputation),
        ("7_cs_without_covariates", cs_no_covariates),
        ("8_practitioner_next_steps", next_steps),
    ]

    run_scenario(
        "campaign_staggered",
        phases,
        metadata={
            "n_units": 150, "n_periods": 26, "n_cohorts": 2,
            "covariates": covars, "n_bootstrap": 999,
            "aggregate": "all", "estimation_method": "dr",
        },
    )


if __name__ == "__main__":
    main()
