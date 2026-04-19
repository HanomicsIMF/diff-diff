"""
Scenario 2: Brand awareness survey DiD — 2x2 with survey design.

DifferenceInDifferences + SurveyDesign under two variance paths:
  (a) analytical Taylor-series linearization (strata + PSU + FPC)
  (b) replicate-weight bootstrap (BRR-style, ~160 replicate columns)

Chains: naive fit (for SE-inflation comparison) -> TSL -> replicate -> multi-
outcome refit loop -> check_parallel_trends -> placebo -> HonestDiD grid.

Data shape: 40 regions x 8 quarters x ~100 respondents per cell =
~32K respondent rows, 10 strata, 4 PSUs/stratum.
"""

import numpy as np

from diff_diff import (
    DifferenceInDifferences,
    MultiPeriodDiD,
    SurveyDesign,
    check_parallel_trends,
    compute_honest_did,
)
from diff_diff.prep import generate_survey_did_data

from bench_shared import run_scenario


def build_data(seed=42):
    df = generate_survey_did_data(
        n_units=200, n_periods=12, cohort_periods=[7],
        never_treated_frac=0.5, treatment_effect=2.0,
        dynamic_effects=True, effect_growth=0.2,
        n_strata=10, psu_per_stratum=4,
        weight_variation="high", psu_re_sd=1.5,
        include_replicate_weights=True, panel=True, seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    df["consideration"] = df["outcome"] + rng.normal(0, 0.4, size=len(df))
    df["purchase_intent"] = df["outcome"] * 0.6 + rng.normal(0, 0.3, size=len(df))
    df["post"] = (df["period"] >= 7).astype(int)
    # Unit-level treatment indicator (for pre-period placebo and
    # parallel-trends check — `treated` is row-level and zero in the pre-
    # period, which those diagnostics can't use).
    df["treat_unit"] = (df["first_treat"] > 0).astype(int)
    return df


def main():
    data = build_data()
    rw_cols = [c for c in data.columns if c.startswith("rep_")]

    results = {}

    def naive_fit():
        did = DifferenceInDifferences(robust=True, cluster="psu")
        results["naive"] = did.fit(
            data, outcome="outcome", treatment="treat_unit", time="post",
        )

    def tsl_fit():
        sd = SurveyDesign(
            weights="weight", strata="stratum", psu="psu",
            fpc="fpc", nest=True,
        )
        did = DifferenceInDifferences(robust=True)
        results["tsl"] = did.fit(
            data, outcome="outcome", treatment="treat_unit", time="post",
            survey_design=sd,
        )

    def replicate_fit():
        if not rw_cols:
            raise RuntimeError("replicate weights not generated")
        sd = SurveyDesign(
            weights="weight", replicate_weights=rw_cols,
            replicate_method="BRR",
        )
        did = DifferenceInDifferences(robust=True)
        results["replicate"] = did.fit(
            data, outcome="outcome", treatment="treat_unit", time="post",
            survey_design=sd,
        )

    def multi_outcome_loop():
        sd = SurveyDesign(
            weights="weight", strata="stratum", psu="psu", nest=True,
        )
        out = {}
        for y in ("outcome", "consideration", "purchase_intent"):
            did = DifferenceInDifferences(robust=True)
            out[y] = did.fit(
                data, outcome=y, treatment="treat_unit", time="post",
                survey_design=sd,
            )
        results["multi_outcome"] = out

    def pretrends():
        results["pt"] = check_parallel_trends(
            data, outcome="outcome", time="period",
            treatment_group="treat_unit",
            pre_periods=list(range(1, 7)),
        )

    def placebo_refit():
        pre = data[data["period"] < 7].copy()
        pre["placebo_post"] = (pre["period"] >= 4).astype(int)
        sd = SurveyDesign(
            weights="weight", strata="stratum", psu="psu", nest=True,
        )
        did = DifferenceInDifferences(robust=True)
        results["placebo"] = did.fit(
            pre, outcome="outcome", treatment="treat_unit",
            time="placebo_post", survey_design=sd,
        )

    def honest_did_grid():
        sd = SurveyDesign(
            weights="weight", strata="stratum", psu="psu", nest=True,
        )
        es = MultiPeriodDiD()
        es_result = es.fit(
            data, outcome="outcome", treatment="treat_unit",
            time="period", unit="unit", reference_period=6,
            survey_design=sd,
        )
        results["event_study"] = es_result
        out = {}
        for M in (0.5, 1.0, 1.5):
            try:
                out[M] = compute_honest_did(
                    es_result, method="relative_magnitude", M=M,
                )
            except Exception as e:
                out[M] = f"{type(e).__name__}: {e}"
        results["honest"] = out

    phases = [
        ("1_naive_fit_no_survey_design", naive_fit),
        ("2_tsl_strata_psu_fpc", tsl_fit),
        ("3_replicate_weights_brr", replicate_fit),
        ("4_multi_outcome_loop_3_metrics", multi_outcome_loop),
        ("5_check_parallel_trends", pretrends),
        ("6_placebo_refit_pre_period", placebo_refit),
        ("7_event_study_plus_honest_did", honest_did_grid),
    ]

    run_scenario(
        "brand_awareness_survey",
        phases,
        metadata={
            "n_units": 200, "n_periods": 12, "n_obs": int(len(data)),
            "n_strata": 10, "n_psu_per_stratum": 4,
            "n_replicate_weights": len(rw_cols),
            "outcomes": ["outcome", "consideration", "purchase_intent"],
        },
    )


if __name__ == "__main__":
    main()
