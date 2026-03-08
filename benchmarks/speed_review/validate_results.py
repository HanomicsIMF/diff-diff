"""
Validate that optimization changes produce identical results.

Usage:
    # Save baseline (run BEFORE code changes):
    python benchmarks/speed_review/validate_results.py --save

    # Validate (run AFTER code changes):
    python benchmarks/speed_review/validate_results.py --check
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from diff_diff import CallawaySantAnna


def generate_data(n_units=10_000, seed=42, n_covariates=0):
    """Generate deterministic test data."""
    rng = np.random.default_rng(seed)
    n_periods = 10
    treatment_periods = [3, 5, 7]

    cohort_assignment = rng.choice(
        [0] + treatment_periods,
        size=n_units,
        p=[0.25, 0.25, 0.25, 0.25],
    )

    rows = []
    for i in range(n_units):
        g = cohort_assignment[i]
        for t in range(1, n_periods + 1):
            treated = 1 if (g > 0 and t >= g) else 0
            y = rng.normal(0, 1) + 2.0 * treated
            rows.append((i, t, y, g))

    df = pd.DataFrame(rows, columns=["unit", "time", "outcome", "first_treat"])

    if n_covariates > 0:
        cov_rng = np.random.default_rng(seed + 1)
        for i in range(1, n_covariates + 1):
            df[f"x{i}"] = cov_rng.normal(size=len(df))

    return df


def run_estimator(df, estimation_method="reg", covariates=None, control_group="never_treated"):
    """Run estimator and extract key results."""
    cs = CallawaySantAnna(
        n_bootstrap=199,
        seed=42,
        estimation_method=estimation_method,
        control_group=control_group,
    )
    results = cs.fit(
        df,
        outcome="outcome",
        unit="unit",
        time="time",
        first_treat="first_treat",
        covariates=covariates,
        aggregate="all",
    )

    out = {
        "overall_att": float(results.overall_att),
        "overall_se": float(results.overall_se),
        "overall_p_value": float(results.overall_p_value),
        "overall_ci": [float(results.overall_conf_int[0]), float(results.overall_conf_int[1])],
    }

    # Group-time effects (sorted for determinism)
    gt_effects = {}
    for (g, t), data in sorted(results.group_time_effects.items()):
        key = f"{g},{t}"
        gt_effects[key] = {
            "effect": float(data["effect"]),
            "se": float(data["se"]),
        }
    out["group_time_effects"] = gt_effects

    # Event study
    if results.event_study_effects:
        es = {}
        for e, data in sorted(results.event_study_effects.items()):
            es[str(e)] = {
                "effect": float(data["effect"]),
                "se": float(data["se"]),
            }
        out["event_study"] = es

    return out


SCENARIOS = [
    {"name": "reg_nocov", "method": "reg", "n_cov": 0},
    {"name": "reg_2cov", "method": "reg", "n_cov": 2},
    {"name": "reg_10cov", "method": "reg", "n_cov": 10},
    {"name": "dr_2cov", "method": "dr", "n_cov": 2},
    {"name": "ipw_2cov", "method": "ipw", "n_cov": 2},
    {"name": "dr_2cov_nyt", "method": "dr", "n_cov": 2, "control_group": "not_yet_treated"},
    {"name": "reg_2cov_nyt", "method": "reg", "n_cov": 2, "control_group": "not_yet_treated"},
]


def save_baseline(path="benchmarks/speed_review/baseline_results.json"):
    """Save baseline results for all scenarios."""
    all_results = {}
    for scenario in SCENARIOS:
        name = scenario["name"]
        print(f"Running scenario: {name} ...")
        df = generate_data(n_covariates=scenario["n_cov"])
        covariates = [f"x{i}" for i in range(1, scenario["n_cov"] + 1)] if scenario["n_cov"] > 0 else None
        control_group = scenario.get("control_group", "never_treated")
        results = run_estimator(df, estimation_method=scenario["method"],
                                covariates=covariates, control_group=control_group)
        all_results[name] = results
        print(f"  Overall ATT: {results['overall_att']:.10f}")
        print(f"  N group-time effects: {len(results['group_time_effects'])}")

    with open(path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nBaseline saved to {path}")


def check_results(path="benchmarks/speed_review/baseline_results.json", tol=1e-12):
    """Check current results against baseline for all scenarios."""
    with open(path) as f:
        all_baseline = json.load(f)

    all_failures = []

    for scenario in SCENARIOS:
        name = scenario["name"]
        if name not in all_baseline:
            print(f"  Skipping {name} (no baseline)")
            continue

        baseline = all_baseline[name]
        df = generate_data(n_covariates=scenario["n_cov"])
        covariates = [f"x{i}" for i in range(1, scenario["n_cov"] + 1)] if scenario["n_cov"] > 0 else None
        control_group = scenario.get("control_group", "never_treated")

        # Use relaxed tolerance for covariate scenarios (Cholesky vs lstsq)
        scenario_tol = 1e-10 if scenario["n_cov"] > 0 else tol

        current = run_estimator(df, estimation_method=scenario["method"],
                                covariates=covariates, control_group=control_group)

        failures = []

        def compare(label, base_val, cur_val, t):
            if np.isnan(base_val) and np.isnan(cur_val):
                return
            diff = abs(base_val - cur_val)
            if diff > t:
                failures.append(f"  {label}: baseline={base_val:.15e}, current={cur_val:.15e}, diff={diff:.2e}")

        compare(f"{name}/overall_att", baseline["overall_att"], current["overall_att"], scenario_tol)
        compare(f"{name}/overall_se", baseline["overall_se"], current["overall_se"], scenario_tol)
        compare(f"{name}/overall_p_value", baseline["overall_p_value"], current["overall_p_value"], 0.02)

        # Compare overall CI values
        if "overall_ci" in baseline and "overall_ci" in current:
            for i, label in enumerate(["lower", "upper"]):
                compare(f"{name}/overall_ci.{label}",
                        baseline["overall_ci"][i], current["overall_ci"][i], scenario_tol)

        # Group-time SE tolerance: tight for covariate scenarios, relaxed for bootstrap
        gt_se_tol = 1e-8 if scenario["n_cov"] > 0 else 0.01

        for key in baseline["group_time_effects"]:
            b = baseline["group_time_effects"][key]
            c = current["group_time_effects"].get(key, {})
            if not c:
                failures.append(f"  {name}/Missing group-time effect: {key}")
                continue
            compare(f"{name}/gt[{key}].effect", b["effect"], c["effect"], scenario_tol)
            compare(f"{name}/gt[{key}].se", b["se"], c["se"], gt_se_tol)

        # Compare event study effects/SEs if present
        if "event_study" in baseline and "event_study" in current:
            for e_key in baseline["event_study"]:
                b_es = baseline["event_study"][e_key]
                c_es = current["event_study"].get(e_key, {})
                if not c_es:
                    failures.append(f"  {name}/Missing event study effect: e={e_key}")
                    continue
                compare(f"{name}/es[{e_key}].effect", b_es["effect"], c_es["effect"], scenario_tol)
                compare(f"{name}/es[{e_key}].se", b_es["se"], c_es["se"], gt_se_tol)

        if failures:
            all_failures.extend(failures)
            print(f"  {name}: FAILED ({len(failures)} mismatches)")
        else:
            print(f"  {name}: PASSED ({len(current['group_time_effects'])} effects checked)")

    if all_failures:
        print("\nVALIDATION FAILED:")
        for f in all_failures:
            print(f)
        sys.exit(1)
    else:
        print("\nALL SCENARIOS PASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save baseline results")
    parser.add_argument("--check", action="store_true", help="Check against baseline")
    parser.add_argument("--tol", type=float, default=1e-12, help="Tolerance for comparison")
    args = parser.parse_args()

    if args.save:
        save_baseline()
    elif args.check:
        check_results(tol=args.tol)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
