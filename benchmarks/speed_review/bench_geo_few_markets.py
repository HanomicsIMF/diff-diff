"""
Scenario 4: Geo-experiment with few treated markets (SyntheticDiD).

Chains: SDiD with jackknife variance (80 LOO refits) -> SDiD with bootstrap
variance for SE comparison -> in_time_placebo -> get_loo_effects_df ->
sensitivity_to_zeta_omega -> weight-concentration diagnostic.

Data shape: 80 markets x 12 weekly periods (6 pre, 6 post), 5 treated,
2 latent factors. Matches Tutorial 18's geo-experiment walkthrough.
"""

from diff_diff import SyntheticDiD
from diff_diff.prep import generate_factor_data

from bench_shared import run_scenario


def build_data(seed=42):
    return generate_factor_data(
        n_units=80, n_pre=6, n_post=6, n_treated=5,
        n_factors=2, treatment_effect=2.0,
        factor_strength=1.0, treated_loading_shift=0.5,
        seed=seed,
    )


def main():
    data = build_data()
    # `treat` is the unit-level (block) indicator; `treated` is row-level.
    # SyntheticDiD requires block treatment, and post_periods identifies the
    # treatment window among treated units.
    post_periods = sorted(
        data.loc[(data["treat"] == 1) & (data["treated"] == 1),
                 "period"].unique().tolist(),
    )

    results = {}

    def sdid_jackknife():
        sdid = SyntheticDiD(variance_method="jackknife", seed=123)
        results["jk"] = sdid.fit(
            data, outcome="outcome", unit="unit", time="period",
            treatment="treat", post_periods=post_periods,
        )

    def sdid_bootstrap():
        sdid = SyntheticDiD(
            variance_method="bootstrap", n_bootstrap=200, seed=123,
        )
        results["bs"] = sdid.fit(
            data, outcome="outcome", unit="unit", time="period",
            treatment="treat", post_periods=post_periods,
        )

    def in_time_placebo():
        fn = getattr(results["jk"], "in_time_placebo", None)
        if fn is None:
            raise RuntimeError("in_time_placebo not available on results")
        results["in_time"] = fn()

    def loo_effects_df():
        fn = getattr(results["jk"], "get_loo_effects_df", None)
        if fn is None:
            raise RuntimeError("get_loo_effects_df not available")
        results["loo"] = fn()

    def sensitivity_zeta_omega():
        fn = getattr(results["jk"], "sensitivity_to_zeta_omega", None)
        if fn is None:
            raise RuntimeError("sensitivity_to_zeta_omega not available")
        results["zeta"] = fn()

    def weight_concentration():
        fn = getattr(results["jk"], "get_weight_concentration", None)
        if fn is None:
            raise RuntimeError("get_weight_concentration not available")
        results["wc"] = fn()

    phases = [
        ("1_sdid_jackknife_variance", sdid_jackknife),
        ("2_sdid_bootstrap_variance_200", sdid_bootstrap),
        ("3_in_time_placebo", in_time_placebo),
        ("4_get_loo_effects_df", loo_effects_df),
        ("5_sensitivity_to_zeta_omega", sensitivity_zeta_omega),
        ("6_weight_concentration", weight_concentration),
    ]

    run_scenario(
        "geo_few_markets",
        phases,
        metadata={
            "n_units": 80, "n_periods": 12, "n_treated": 5,
            "n_factors": 2,
        },
    )


if __name__ == "__main__":
    main()
