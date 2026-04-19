# Reporting

This document records the methodology choices embedded in
`BusinessReport` and `DiagnosticReport` — the convenience layer that
produces plain-English stakeholder narratives from any diff-diff result.

Methodology for estimators lives in `REGISTRY.md`. This file is the
single source for reporting-layer decisions; `REGISTRY.md` cross-links
here rather than duplicating content.

## Module

- `diff_diff/business_report.py` — `BusinessReport`, `BusinessContext`.
- `diff_diff/diagnostic_report.py` — `DiagnosticReport`,
  `DiagnosticReportResults`.

Both modules dispatch by `type(results).__name__` lookup to avoid
circular imports across the 16 result classes. They do no estimator
fitting and do not re-derive any variance from raw data; every effect,
SE, p-value, CI, and sensitivity bound is either read from the fitted
result or produced by an existing diff-diff utility
(`compute_honest_did`, `HonestDiD.sensitivity`, `bacon_decompose`,
`check_parallel_trends`, `compute_pretrends_power`). When the caller
passes the raw panel + column kwargs, `DiagnosticReport` may call
those utilities on the supplied data (2x2 PT via
`check_parallel_trends`, Goodman-Bacon decomposition via
`bacon_decompose`, and the EfficientDiD Hausman PT-All vs PT-Post
pretest via `EfficientDiD.hausman_pretest`).

The `design_effect` section of `DiagnosticReport.to_dict()` is a
read-only surface: it echoes `survey_metadata.design_effect` and
`effective_n` from the fitted result along with a plain-English band
label. It does not call `compute_deff_diagnostics` (that helper
needs per-fit internals the result objects do not expose). The report layer **does** compose a few
cross-period summary statistics from per-period inputs already
produced by the estimator — specifically the joint-Wald / Bonferroni
pre-trends p-value from pre-period event-study coefficients (see
`_pt_event_study`), the MDV-to-ATT ratio for power-tier selection,
and the heterogeneity dispersion block (CV / range / sign-
consistency over post-treatment group / event-study / group-time
effects, pre-period and reference-marker rows excluded). These are
reporting-layer aggregations of inputs already in the result object,
not new inference.

## Design deviations

- **Note:** No hard pass/fail gates. `DiagnosticReport` does not produce
  a traffic-light verdict. Severity is conveyed through natural-language
  phrasing ("robust", "fragile", "material share"). This is an explicit
  deviation from the strategy document's Gap 4 ("traffic-light
  assessment (green/yellow/red)"); the choice is motivated by the
  well-known risk of naive thresholds producing false confidence. A
  `ConservativeThresholds` opt-in layer remains available as a future
  addition if practitioner demand materialises.

- **Note:** Placebo battery is opt-in (`run_placebo=False` by default).
  `run_all_placebo_tests` on a typical panel (500 permutations times one
  DiD fit per permutation) adds tens of seconds of latency, which would
  be surprising as the default on a convenience wrapper. The schema
  reserves the `"placebo"` key; it is always rendered with
  `{"status": "skipped", "reason": "..."}` in MVP so agents parsing the
  schema see a stable shape.

- **Note:** `DiagnosticReport` does not call `check_parallel_trends` on
  event-study or staggered result objects. `check_parallel_trends` in
  `diff_diff/utils.py` assumes a single binary treatment with universal
  pre-periods; for staggered and event-study designs, DR reads the
  pre-period event-study coefficients directly and constructs a joint
  Wald statistic (or Bonferroni fallback when `vcov` is missing). This
  mirrors the guidance in `practitioner._parallel_trends_step(staggered=True)`.

- **Note:** Survey finite-df PT policy. When the fitted result carries
  a finite `survey_metadata.df_survey`, `_pt_event_study` computes
  `F = W / k` (numerator df = k pre-period coefficients) against an
  F(k, df_survey) reference distribution rather than chi-square(k).
  The design-based SE already reflects the effective sample size, so
  the chi-square reference would systematically over-reject under the
  finite-sample correction the SE captures. The schema surfaces the
  survey branch via the `method` suffix `_survey`
  (e.g., `joint_wald_survey`, `joint_wald_event_study_survey`) and
  exposes the denominator df as `df_denom`, so BR / DR prose can flag
  the finite-sample correction rather than silently presenting a
  chi-square-style result. Non-finite `df_survey` (NaN / inf /
  non-positive) falls back to the chi-square path.

- **Note:** Estimator-native validation surfaces are surfaced rather
  than duplicated. `SyntheticDiDResults` routes parallel-trends to
  `pre_treatment_fit` (the RMSE of the synthetic-control fit on the
  pre-period), and routes sensitivity to `in_time_placebo()` +
  `sensitivity_to_zeta_omega()`. `TROPResults` surfaces factor-model
  diagnostics (`effective_rank`, `loocv_score`, selected `lambda_*`)
  under `estimator_native_diagnostics`. `EfficientDiDResults` PT runs
  through `EfficientDiD.hausman_pretest` (the estimator's native
  PT-All vs PT-Post check).

- **Note:** Pre-trends verdict is a three-bin heuristic, not a field
  convention. DR maps the joint p-value as follows:

  - `joint_p >= 0.30` &rarr; `no_detected_violation`.
  - `0.05 <= joint_p < 0.30` &rarr; `some_evidence_against`.
  - `joint_p < 0.05` &rarr; `clear_violation`.

  These thresholds are diff-diff heuristics. The 0.30 upper bound draws
  on equivalence-testing intuition (Rambachan & Roth 2023 discuss the
  limitations of pre-tests). The `no_detected_violation` label
  deliberately avoids "parallel trends hold" language — the test did
  not detect a violation, but pre-trends tests are commonly
  underpowered. See the power-aware phrasing rule below.

- **Note:** Power-aware phrasing for `no_detected_violation`. DR calls
  `compute_pretrends_power(results, violation_type='linear',
  alpha=alpha, target_power=0.80)` for the estimator families that
  ship a `compute_pretrends_power` adapter: `MultiPeriodDiDResults`,
  `CallawaySantAnnaResults`, and `SunAbrahamResults` (see
  `_APPLICABILITY["pretrends_power"]` in
  `diff_diff/diagnostic_report.py`). Other staggered families with
  event-study output (`ImputationDiDResults`, `TwoStageDiDResults`,
  `StackedDiDResults`, `EfficientDiDResults`,
  `StaggeredTripleDiffResults`, `WooldridgeDiDResults`,
  `ChaisemartinDHaultfoeuilleResults`) do not yet have a power
  adapter and therefore render the `no_detected_violation` tier as
  `underpowered` with the fallback reason recorded in
  `schema["pre_trends"]["power_reason"]` (plain-English explanation)
  while `schema["pre_trends"]["power_status"]` carries the
  machine-readable enum (`"ran"` / `"skipped"` / `"error"` /
  `"not_applicable"`). BusinessReport then reads
  `mdv_share_of_att = mdv / abs(att)` and selects a tier:

  - `< 0.25` &rarr; `well_powered` &mdash; "the test has 80% power to
    detect a violation of magnitude M, which is only X% of the
    estimated effect; if a material pre-trend existed, this test would
    likely have caught it."
  - `>= 0.25 and < 1.0` &rarr; `moderately_powered` &mdash; "the test
    is informative but not definitive; see the sensitivity analysis
    below for bounded-violation guarantees."
  - `>= 1.0` &rarr; `underpowered` &mdash; "the test has limited
    power &mdash; a non-rejection does not prove the assumption. See
    the HonestDiD sensitivity analysis below for a more reliable
    signal."
  - Power analysis not runnable &rarr; fall back to `underpowered`
    phrasing; the fallback reason is recorded in
    `schema["pre_trends"]["power_reason"]` (plain-English explanation;
    `power_status` carries the enum).

  Rationale: always-hedging phrasing under-sells well-designed
  studies; always-confident phrasing over-sells underpowered ones.
  The library already ships `compute_pretrends_power()`, so using it
  is the honest default rather than hedging every non-violation.

- **Note:** Diagonal-covariance fallback for staggered-estimator power.
  `compute_pretrends_power()` currently drops to `np.diag(ses**2)` for
  CS / SA / ImputationDiD / Stacked / etc. even when the full
  `event_study_vcov` is attached on the result. The
  `DiagnosticReport.pretrends_power` block records
  `covariance_source: "diag_fallback_available_full_vcov_unused"` in
  that case, and `BusinessReport` downgrades a `well_powered` tier to
  `moderately_powered` before rendering prose. This is a known
  conservative deviation from the documented "use the full pre-period
  covariance" position — it prevents the diagonal approximation from
  producing an overly optimistic "well-powered" claim when correlated
  pre-period errors could tighten the MDV. The right long-term fix is
  to teach `compute_pretrends_power()` to consume `event_study_vcov`
  and `event_study_vcov_index`; until that lands this downgrade stays.

- **Note:** Unit-translation policy. BusinessReport does not
  arithmetically translate log-points to percents or level effects to
  log-points. The estimate is rendered in the scale the estimator
  produced; `outcome_unit="log_points"` emits an informational
  caveat. The policy avoids guessing the underlying model (no
  estimator in the library currently exports both log and level
  coefficients), which would be unsafe in the presence of non-linear
  link functions (Poisson QMLE, logit).

- **Note:** Single-knob `alpha` with preserved-native-CI fallback.
  BusinessReport exposes only `alpha` (defaults to `results.alpha`);
  there is no separate `significance_threshold` parameter. When the
  requested `alpha` matches the fit's native level, it drives both the
  CI level (`(1 - alpha) * 100`% interval) and the phrasing tier
  threshold ("statistically significant at the (1 - alpha) * 100%
  level"). When the requested `alpha` differs from the fit's native
  level (e.g., the user asks for `alpha=0.10` on a result fit with
  `alpha=0.05`), BusinessReport does NOT recompute the CI at the
  requested level, because the stored CI is the only quantile the
  underlying estimator supplied (bootstrap distributions and
  finite-df analytical variances are not always retained on the
  result). Instead, the schema preserves the fit's native CI (with its
  original level) and uses the requested `alpha` only for the
  significance-phrasing threshold, and emits an
  `alpha_override_preserved` caveat describing the mismatch. This is
  the conservative choice: it avoids silently recomputing CIs under
  assumptions the estimator may not support.

- **Note:** Schema stability policy for the AI-legible `to_dict()`
  surface. New top-level keys count as additive (no version bump); new
  values in any `status` enum count as breaking (agents doing
  exhaustive pattern match will break on unknown enums); renames and
  removals count as breaking. The `BUSINESS_REPORT_SCHEMA_VERSION`
  and `DIAGNOSTIC_REPORT_SCHEMA_VERSION` constants bump independently.
  The v3.2 CHANGELOG marks both schemas experimental so users do not
  anchor tooling on them prematurely; a formal deprecation policy will
  land within two subsequent PRs.

## Reference implementation(s)

The phrasing rules follow the guidance in:

- Baker, A. C., Callaway, B., Cunningham, S., Goodman-Bacon, A., &
  Sant'Anna, P. H. C. (2025). *Difference-in-Differences Designs: A
  Practitioner's Guide.* (The 8-step workflow enforced through
  `diff_diff/practitioner.py`.)
- Rambachan, A., & Roth, J. (2023). *A More Credible Approach to
  Parallel Trends.* Review of Economic Studies. (HonestDiD sensitivity;
  the pre-test power caveat directly shaped the three-tier power
  phrasing.)
- Roth, J. (2022). *Pretest with Caution: Event-study Estimates after
  Testing for Parallel Trends.* American Economic Review: Insights.
  (Motivates the power-aware phrasing tiers.)
