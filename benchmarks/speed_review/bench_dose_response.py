"""
Scenario 6: Pricing dose-response with ContinuousDiD cubic spline.

Chains: CDiD fit with aggregate='dose' (overall ATT + ACRT + dose-response
curves + bootstrap 199) -> dataframe extraction -> event-study pre-trend ->
binarized-DiD comparison -> spline sensitivity (degree=1, num_knots=2).

Data shape: 500 stores x 6 quarterly periods, 1 cohort at period 3,
log-normal dose. Matches Tutorial 14 scaled from 200 to 500 units.
"""

import numpy as np
import pandas as pd

from diff_diff import ContinuousDiD, DifferenceInDifferences
from diff_diff.prep import generate_continuous_did_data

from bench_shared import run_scenario


def build_data(seed=42):
    df = generate_continuous_did_data(
        n_units=500, n_periods=6, seed=seed,
    )
    # Set first_treat to period 3 for all treated; ContinuousDiD expects
    # staggered/first_treat format.
    if "first_treat" not in df.columns:
        treated_mask = df.get("dose", pd.Series(0.0, index=df.index)) > 0
        df["first_treat"] = np.where(treated_mask, 3, 0)
    return df


def main():
    data = build_data()

    results = {}
    fit_kwargs = dict(
        data=data, outcome="outcome", unit="unit", time="period",
        first_treat="first_treat", dose="dose",
    )

    def cdid_cubic_fit():
        cdid = ContinuousDiD(
            degree=3, num_knots=1, n_bootstrap=199, seed=123,
        )
        results["cubic"] = cdid.fit(**fit_kwargs, aggregate="dose")

    def extract_curves():
        # The cubic fit used aggregate="dose", so only dose-response and
        # group-time levels are available on the result. Event-study is
        # extracted separately in the dedicated pretrend phase below.
        # NB: ContinuousDiD uses 'eventstudy' for fit(aggregate=...) but
        # 'event_study' for to_dataframe(level=...). Two different
        # spellings within one estimator - flagged in performance-plan.md.
        r = results["cubic"]
        out = {}
        for level in ("dose_response", "group_time"):
            out[level] = r.to_dataframe(level=level)
        results["curves"] = out

    def cdid_event_study():
        cdid = ContinuousDiD(
            degree=3, num_knots=1, n_bootstrap=0, seed=123,
        )
        results["event_study"] = cdid.fit(
            **fit_kwargs, aggregate="eventstudy",
        )

    def binarized_comparison():
        data_bin = data.copy()
        data_bin["treated_any"] = (data_bin["dose"] > 0).astype(int)
        data_bin["post"] = (data_bin["period"] >= 3).astype(int)
        did = DifferenceInDifferences(robust=True)
        results["binarized"] = did.fit(
            data_bin, outcome="outcome", treatment="treated_any", time="post",
        )

    def spline_sensitivity_linear():
        cdid = ContinuousDiD(
            degree=1, num_knots=0, n_bootstrap=199, seed=123,
        )
        results["linear"] = cdid.fit(**fit_kwargs, aggregate="dose")

    def spline_sensitivity_more_knots():
        cdid = ContinuousDiD(
            degree=3, num_knots=2, n_bootstrap=199, seed=123,
        )
        results["many_knots"] = cdid.fit(**fit_kwargs, aggregate="dose")

    phases = [
        ("1_cdid_cubic_spline_bootstrap199", cdid_cubic_fit),
        ("2_extract_dose_response_dataframes", extract_curves),
        ("3_cdid_event_study_pretrend", cdid_event_study),
        ("4_binarized_did_comparison", binarized_comparison),
        ("5_spline_sensitivity_degree1", spline_sensitivity_linear),
        ("6_spline_sensitivity_num_knots2", spline_sensitivity_more_knots),
    ]

    run_scenario(
        "dose_response",
        phases,
        metadata={
            "n_units": 500, "n_periods": 6, "n_bootstrap": 199,
            "spline_configs": ["degree=3,k=1", "degree=1,k=0", "degree=3,k=2"],
        },
    )


if __name__ == "__main__":
    main()
