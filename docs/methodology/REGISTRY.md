# Methodology Registry

This document provides the academic foundations and key implementation requirements for each estimator in diff-diff. It serves as a reference for contributors and users who want to understand the theoretical basis of the methods.

## Table of Contents

1. [Core DiD Estimators](#core-did-estimators)
   - [DifferenceInDifferences](#differenceinifferences)
   - [MultiPeriodDiD](#multiperioddid)
   - [TwoWayFixedEffects](#twowayfixedeffects)
2. [Modern Staggered Estimators](#modern-staggered-estimators)
   - [CallawaySantAnna](#callawaysantanna)
   - [ChaisemartinDHaultfoeuille](#chaisemartindhaultfoeuille)
   - [ContinuousDiD](#continuousdid)
   - [SunAbraham](#sunabraham)
   - [ImputationDiD](#imputationdid)
   - [TwoStageDiD](#twostagedid)
   - [StackedDiD](#stackeddid)
   - [WooldridgeDiD (ETWFE)](#wooldridgedid-etwfe)
3. [Advanced Estimators](#advanced-estimators)
   - [SyntheticDiD](#syntheticdid)
   - [TripleDifference](#tripledifference)
   - [StaggeredTripleDifference](#staggeredtripledifference)
   - [TROP](#trop)
4. [Diagnostics & Sensitivity](#diagnostics--sensitivity)
   - [PlaceboTests](#placebotests)
   - [BaconDecomposition](#bacondecomposition)
   - [HonestDiD](#honestdid)
   - [PreTrendsPower](#pretrendspower)
   - [PowerAnalysis](#poweranalysis)

---

# Core DiD Estimators

## DifferenceInDifferences

**Primary source:** Canonical econometrics textbooks
- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed. MIT Press.
- Angrist, J.D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.

**Key implementation requirements:**

*Assumption checks / warnings:*
- Treatment and post indicators must be binary (0/1) with variation in both
- Warns if no treated units in pre-period or no control units in post-period
- Parallel trends assumption is untestable but can be assessed with pre-treatment data

*Estimator equation (as implemented):*
```
ATT = (Ȳ_{treated,post} - Ȳ_{treated,pre}) - (Ȳ_{control,post} - Ȳ_{control,pre})
    = E[Y(1) - Y(0) | D=1]
```

Regression form:
```
Y_it = α + β₁(Treated_i) + β₂(Post_t) + τ(Treated_i × Post_t) + X'γ + ε_it
```
where τ is the ATT.

*Standard errors:*
- Default: HC1 heteroskedasticity-robust
- Optional: Cluster-robust (specify `cluster` parameter)
- Optional: Wild cluster bootstrap for small number of clusters

*Edge cases:*
- Empty cells (e.g., no treated-pre observations) cause rank deficiency, handled per `rank_deficient_action` setting
  - With "warn" (default): emits warning, sets NaN for affected coefficients
  - With "error": raises ValueError
  - With "silent": continues silently with NaN coefficients
- Singleton clusters (one observation): included in variance estimation; contribute to meat matrix via u_i² X_i X_i' (same formula as larger clusters with n_g=1)
- Rank-deficient design matrix (collinearity): warns and sets NA for dropped coefficients (R-style, matches `lm()`)
  - Tolerance: `1e-07` (matches R's `qr()` default), relative to largest diagonal element of R in QR decomposition
  - Controllable via `rank_deficient_action` parameter: "warn" (default), "error", or "silent"

**Reference implementation(s):**
- R: `fixest::feols()` with interaction term
- Stata: `reghdfe` or manual regression with interaction

**Requirements checklist:**
- [x] Treatment and time indicators are binary 0/1 with variation
- [x] ATT equals coefficient on interaction term
- [x] Wild bootstrap supports Rademacher, Mammen, Webb weight distributions
- [x] Formula interface parses `y ~ treated * post` correctly

---

## MultiPeriodDiD

**Primary source:** Event study methodology
- Freyaldenhoven, S., Hansen, C., Pérez, J.P., & Shapiro, J.M. (2021). Visualization,
  identification, and estimation in the linear panel event-study design. NBER Working Paper 29170.
- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed.
  MIT Press, Ch. 10, 13.
- Angrist, J.D., & Pischke, J.-S. (2009). *Mostly Harmless Econometrics*. Princeton University Press.

**Scope:** Simultaneous adoption event study. All treated units receive treatment at the
same time. For staggered adoption (different units treated at different times), use
CallawaySantAnna or SunAbraham instead.

**Key implementation requirements:**

*Assumption checks / warnings:*
- Treatment indicator must be binary (0/1) with variation in both groups
- Requires at least 1 pre-treatment and 1 post-treatment period
- Warns when only 1 pre-period available (≥2 needed to test parallel trends;
  ATT is still valid but pre-trends assessment is not possible)
- Reference period defaults to last pre-treatment period (e=-1 convention)
- Treatment indicator should be time-invariant ever-treated (D_i);
  warns when time-varying D_it detected (requires `unit` parameter)
- Warns if treatment timing varies across units when `unit` is provided
  (suggests CallawaySantAnna or SunAbraham instead)
- Treatment must be an absorbing state (once treated, always treated)

*Estimator equation (target specification):*

With unit and time fixed effects absorbed:

```
Y_it = α_i + γ_t + Σ_{e≠-1} δ_e × D_i × 1(t = E + e) + X'β + ε_it
```

where:
- α_i = unit fixed effects (absorbed)
- γ_t = time fixed effects (absorbed)
- E = common treatment time (same for all treated units)
- D_i = treatment group indicator (1=treated, 0=control)
- e = t - E = event time (relative periods to treatment)
- δ_e = treatment effect at event time e
- δ_{-1} = 0 (reference period, omitted for identification)

For simultaneous treatment, this is equivalent to interacting treatment with
calendar-time indicators:

```
Y_it = α_i + γ_t + Σ_{t≠t_ref} δ_t × (D_i × Period_t) + X'β + ε_it
```

where interactions are included for ALL periods (pre and post), not just post-treatment.

Pre-treatment coefficients (e < -1) test the parallel trends assumption:
under H0 of parallel trends, δ_e = 0 for all e < 0.

Post-treatment coefficients (e ≥ 0) estimate dynamic treatment effects.

Average ATT over post-treatment periods:

```
ATT_avg = (1/|post|) × Σ_{e≥0} δ_e
```

with SE computed from the sub-VCV matrix:

```
Var(ATT_avg) = 1'V1 / |post|²
```

where V is the VCV sub-matrix for post-treatment δ_e coefficients.

*Standard errors:*
- Default: HC1 heteroskedasticity-robust (same as DifferenceInDifferences base class)
- Alternative: Cluster-robust at unit level via `cluster` parameter (recommended for panel data)
- Optional: Wild cluster bootstrap (complex for multi-coefficient testing;
  requires joint bootstrap distribution)
- Degrees of freedom adjusted for absorbed fixed effects

*Edge cases:*
- Reference period: omitted from design matrix; coefficient is zero by construction.
  Default is last pre-treatment period (e=-1). User can override via `reference_period`.
- Post-period reference: raises ValueError. Post-period references would exclude a
  post-treatment period from estimation, biasing avg_att and breaking downstream inference.
- Reference period default change: FutureWarning emitted when `reference_period` is not
  explicitly specified and ≥2 pre-periods exist, noting the default changed from first
  to last pre-period (e=-1 convention, matching fixest/did).
- Never-treated units: all event-time indicators are zero; they identify the time
  fixed effects and serve as comparison group.
- Endpoint binning: distant event times (e.g., e < -K or e > K) should be binned
  into endpoint indicators to avoid sparse cells. This prevents imprecise estimates
  at extreme leads/lags.
- Unbalanced panels: only uses observations where event-time is defined. Units
  not observed at all event times contribute to the periods they are present for.
- Rank-deficient design matrix (collinearity): warns and sets NA for dropped
  coefficients (R-style, matches `lm()`)
- Average ATT (`avg_att`) is NA if any post-period effect is unidentified
  (R-style NA propagation)
- NaN inference for undefined statistics:
  - t_stat: Uses NaN (not 0.0) when SE is non-finite or zero
  - p_value and CI: Also NaN when t_stat is NaN
  - avg_se: Checked for finiteness before computing avg_t_stat
  - **Note**: Defensive enhancement matching CallawaySantAnna NaN convention
- Treatment reversal: warns if any unit transitions from treated to untreated
  (non-absorbing treatment violates the simultaneous adoption assumption)
- Time-varying treatment (D_it): warns when `unit` parameter is provided and
  within-unit treatment variation is detected. Advises creating an ever-treated
  indicator. Without ever-treated D_i, pre-period interaction coefficients are
  unidentified.
- Pre-test of parallel trends: joint F-test on pre-treatment δ_e coefficients.
  Low power in pre-test does not validate parallel trends (Roth 2022).

**Reference implementation(s):**
- R: `fixest::feols(y ~ i(time, treatment, ref=ref_period) | unit + time, data, cluster=~unit)`
  or equivalently `feols(y ~ i(event_time, ref=-1) | unit + time, data, cluster=~unit)`
- Stata: `reghdfe y ib(-1).event_time#1.treatment, absorb(unit time) cluster(unit)`

**Requirements checklist:**

- [x] Event-time indicators for ALL periods (pre and post), not just post-treatment
- [x] Reference period coefficient is zero (normalized by omission from design matrix)
- [x] Pre-period coefficients available for parallel trends assessment
- [ ] Default cluster-robust SE at unit level (currently HC1; cluster-robust via `cluster` param)
- [ ] Supports unit and time FE via absorption
- [ ] Endpoint binning for distant event times
- [x] Average ATT correctly accounts for covariance between period effects
- [x] Returns PeriodEffect objects with confidence intervals
- [x] Supports both balanced and unbalanced panels

---

## TwoWayFixedEffects

**Primary source:** Panel data econometrics
- Wooldridge, J.M. (2010). *Econometric Analysis of Cross Section and Panel Data*, 2nd ed. MIT Press, Chapter 10.

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Staggered treatment warning**: If treatment timing varies across units, warns about potential bias from negative weights (Goodman-Bacon 2021, de Chaisemartin & D'Haultfœuille 2020)
- Requires sufficient within-unit and within-time variation
- Warns if any fixed effect is perfectly collinear with treatment

*Estimator equation (as implemented):*
```
Y_it = α_i + γ_t + τ(D_it) + X'β + ε_it
```
Estimated via within-transformation (demeaning):
```
Ỹ_it = τD̃_it + X̃'β + ε̃_it
```
where tildes denote demeaned variables.

**Note:** The interaction term `D_i × Post_t` is within-transformed (demeaned) alongside the
outcome and covariates before regression. This is required by the Frisch-Waugh-Lovell theorem:
all regressors must be projected out of the same fixed effects space as the dependent variable.
This matches the behavior of R's `fixest::feols()` with absorbed FE.

*Standard errors:*
- Default: Cluster-robust at unit level (accounts for serial correlation)
- Degrees of freedom adjusted for absorbed fixed effects: `df_adjustment = n_units + n_times - 2`

*Edge cases:*
- Singleton units/periods are automatically dropped
- Treatment perfectly collinear with FE raises error with informative message listing dropped columns
- Covariate collinearity emits warning but estimation continues (ATT still identified)
- Rank-deficient design matrix: warns and sets NA for dropped coefficients (R-style, matches `lm()`)
- Unbalanced panels handled via proper demeaning
- Multi-period `time` parameter: only binary (0/1) post indicator is recommended; multi-period values
  produce `treated × period_number` rather than `treated × post_indicator`. A `UserWarning` is
  emitted when `time` has >2 unique values, advising users to create a binary post column.
  Non-{0,1} binary time (e.g., {2020, 2021}) also emits a warning, though the ATT is mathematically
  correct — the within-transformation absorbs the scaling.
- Staggered warning limitation: requires `time` to have actual period values (not binary 0/1)
  so that different cohort first-treatment times can be distinguished. With binary `time="post"`,
  all treated units appear to start at `time=1`, making staggering undetectable. Users with
  staggered designs should use `decompose()` or `CallawaySantAnna` directly.

**Reference implementation(s):**
- R: `fixest::feols(y ~ treat:post | unit + post, data, cluster = ~unit)`
- Stata: `reghdfe y treat, absorb(unit time) cluster(unit)`

**Requirements checklist:**
- [ ] Staggered adoption detection warning (only fires when `time` has >2 unique values; with binary `time`, staggering is undetectable)
- [x] Multi-period time warning (fires when `time` has >2 unique values)
- [x] Auto-clusters standard errors at unit level
- [x] `decompose()` method returns BaconDecompositionResults
- [x] Within-transformation correctly handles unbalanced panels
- [x] Non-{0,1} binary time warning (fires when time has 2 unique values not in {0,1})
- [x] ATT invariance to time encoding (verified by test)

---

# Modern Staggered Estimators

## CallawaySantAnna

**Primary source:** [Callaway, B., & Sant'Anna, P.H.C. (2021). Difference-in-Differences with multiple time periods. *Journal of Econometrics*, 225(2), 200-230.](https://doi.org/10.1016/j.jeconom.2020.12.001)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires never-treated units as comparison group (identified by `first_treat=0` or `never_treated=True`)
- Warns if no never-treated units exist (suggests alternative comparison strategies)
- Limited pre-treatment periods reduce ability to test parallel trends

*Estimator equation (as implemented):*

Group-time average treatment effect:
```
ATT(g,t) = E[Y_t - Y_{g-1} | G_g=1] - E[Y_t - Y_{g-1} | C=1]
```
where G_g=1 indicates units first treated in period g, and C=1 indicates never-treated.

*Note:* This equation uses g-1 as the base period, which applies to post-treatment effects (t ≥ g) and `base_period="universal"`. With `base_period="varying"` (default), pre-treatment effects use t-1 as base for consecutive comparisons (see Base period selection in Edge cases).

With covariates (doubly robust):
```
ATT(g,t) = E[((G_g - p̂_g(X))/(1-p̂_g(X))) × (Y_t - Y_{g-1} - m̂_{0,g,t}(X) + m̂_{0,g,g-1}(X))] / E[G_g]
```

Aggregations:
- Simple: `ATT = Σ_{g,t} w_{g,t} × ATT(g,t)` weighted by group size
- Event-study: `ATT(e) = Σ_g w_g × ATT(g, g+e)` for event-time e
- Group: `ATT(g) = Σ_t ATT(g,t) / T_g` average over post-periods

*Standard errors:*
- Default: Analytical (influence function-based)
- All aggregation SEs (simple, event study) include the weight influence function (WIF)
  adjustment, matching R's `did::aggte()`. The WIF accounts for uncertainty in estimating
  group-size aggregation weights. Group aggregation uses equal time weights (deterministic),
  so WIF is zero.
- Bootstrap: Multiplier bootstrap with Rademacher, Mammen, or Webb weights. Bootstrap
  perturbs the combined influence function (standard IF + WIF) directly, not just fixed-weight
  re-aggregation. This correctly propagates weight estimation uncertainty.
- Block structure preserves within-unit correlation
- Simultaneous confidence bands (`cband=True`, default): Uses sup-t bootstrap to compute
  a uniform critical value across event times, controlling family-wise error rate. Matches
  R's `did::aggte(..., cband=TRUE)` default. Requires `n_bootstrap > 0`.

*Bootstrap weight distributions:*

The multiplier bootstrap uses random weights w_i with E[w]=0 and Var(w)=1:

| Weight Type | Values | Probabilities | Properties |
|-------------|--------|---------------|------------|
| Rademacher | ±1 | 1/2 each | Simplest; E[w³]=0 |
| Mammen | -(√5-1)/2, (√5+1)/2 | (√5+1)/(2√5), (√5-1)/(2√5) | E[w³]=1; better for skewed data |
| Webb | ±√(3/2), ±1, ±√(1/2) | 1/6 each | 6-point; recommended for few clusters |

**Webb distribution details:**
- Values: {-√(3/2), -1, -√(1/2), √(1/2), 1, √(3/2)} ≈ {-1.225, -1, -0.707, 0.707, 1, 1.225}
- Equal probabilities (1/6 each) giving E[w]=0, Var(w)=1
- Matches R's `did` package implementation
- **Verification**: Implementation matches `fwildclusterboot` R package
  ([C++ source](https://github.com/s3alfisc/fwildclusterboot/blob/master/src/wildboottest.cpp))
  which uses identical `sqrt(1.5)`, `1`, `sqrt(0.5)` values with equal 1/6 probabilities.
  Some documentation shows simplified values (±1.5, ±1, ±0.5) but actual implementations
  use square root values to achieve unit variance.
- Reference: Webb, M.D. (2023). Reworking Wild Bootstrap Based Inference for Clustered Errors.
  Queen's Economics Department Working Paper No. 1315. (Updated from Webb 2014)

*Edge cases:*
- Groups with single observation: included but may have high variance
- Missing group-time cells: omitted from `group_time_effects` with a consolidated warning listing skip reasons and counts
  - **Note:** Non-estimable cells (missing base/post period, zero treated/control, insufficient data) are omitted rather than stored as NaN. A consolidated UserWarning is emitted from `fit()` across all estimation paths. R's `did` package also omits these cells from `aggte()` results.
  - **Note:** When `balance_e` is specified, cohorts with NaN effects at the anchor horizon are excluded from the balanced panel
- Anticipation: `anticipation` parameter shifts reference period
  - Group aggregation includes periods t >= g - anticipation (not just t >= g)
  - Both analytical SE and bootstrap SE aggregation respect anticipation
  - Not-yet-treated + anticipation: control mask uses `G > max(t, base_period) + anticipation`
    to exclude cohorts treated at either the evaluation period or the base period.
    This prevents control contamination when `base_period="universal"` and the base
    period is later than the evaluation period (e.g., pre-treatment ATT with universal base)
- Rank-deficient design matrix (covariate collinearity):
  - Detection: Pivoted QR decomposition with tolerance `1e-07` (R's `qr()` default)
  - Handling: Warns and drops linearly dependent columns, sets NA for dropped coefficients (R-style, matches `lm()`)
  - Parameter: `rank_deficient_action` controls behavior: "warn" (default), "error", or "silent"
- Non-finite inference values:
  - Analytic SE: Returns NaN to signal invalid inference (not biased via zeroing)
  - Bootstrap: Drops non-finite samples, warns, and adjusts p-value floor accordingly. SE, CI, and p-value are all NaN if the original point estimate is non-finite, SE is non-finite or zero (e.g., n_valid=1 with ddof=1, or identical samples)
  - Threshold: Returns NaN if <50% of bootstrap samples are valid
  - Per-effect t_stat: Uses NaN (not 0.0) when SE is non-finite or zero (consistent with overall_t_stat)
  - **Note**: This is a defensive enhancement over reference implementations (R's `did::att_gt`, Stata's `csdid`) which may error or produce unhandled inf/nan in edge cases without informative warnings
- No post-treatment effects (all treatment occurs after data ends):
  - Overall ATT set to NaN (no post-treatment periods to aggregate)
  - All overall inference fields (SE, t-stat, p-value, CI) also set to NaN
  - Warning emitted: "No post-treatment effects for aggregation"
  - Individual pre-treatment ATT(g,t) are computed (for parallel trends assessment)
  - Bootstrap runs for per-effect SEs even without post-treatment; only overall statistics are NaN
  - **Principle**: NaN propagates consistently through overall inference fields; pre-treatment effects get full bootstrap inference
- Aggregated t_stat (event-study, group-level):
  - Uses NaN when SE is non-finite or zero (matches per-effect and overall t_stat behavior)
  - Previous behavior (0.0 default) was inconsistent and misleading
- Base period selection (`base_period` parameter):
  - "varying" (default): Pre-treatment uses t-1 as base (consecutive comparisons).
    Post-treatment uses long differences from g-1-anticipation. The parallel trends
    assumption for treatment effects is about long-run trends, but pre-treatment tests
    only check consecutive periods.
  - "universal": ALL effects (pre and post) are long differences from g-1-anticipation.
    The parallel trends assumption is about long-run trends. Pre-treatment coefficients
    test cumulative divergence from the base period.
  - Both produce identical post-treatment ATT(g,t); differ only pre-treatment
  - `anticipation` shifts the base period to g-1-anticipation, which moves it further
    from treatment and strengthens the parallel trends assumption.
  - Matches R `did::att_gt()` base_period parameter
  - **Event study output**: With "universal", includes reference period (e=-1-anticipation)
    with effect=0, se=NaN, conf_int=(NaN, NaN). Inference fields are NaN since this is
    a normalization constraint, not an estimated effect. Only added when real effects exist.
- Base period interaction with Sun-Abraham comparison:
  - CS with `base_period="varying"` produces different pre-treatment estimates than SA
  - This is expected: CS uses consecutive comparisons, SA uses fixed reference (e=-1-anticipation)
  - Use `base_period="universal"` for methodologically comparable pre-treatment effects
  - Post-treatment effects match regardless of base_period setting
- Propensity score estimation:
  - Algorithm: IRLS (Fisher scoring), matching R's `glm(family=binomial)` default
  - **Note:** Uses IRLS (Fisher scoring) for propensity score estimation, consistent
    with R's `did::att_gt()` which uses `glm(family=binomial)` internally
  - Near-separation detection: Warns when predicted probabilities are within 1e-5
    of 0 or 1, or when IRLS fails to converge
  - Trimming: Propensity scores clipped to `[pscore_trim, 1-pscore_trim]` (default
    0.01) before weight computation. Warning emitted when scores are trimmed.
  - **Events Per Variable (EPV) diagnostics:** Per-cohort EPV =
    min(n_treated, n_control) / n_covariates checked before IRLS.
    Default threshold: 10 (Peduzzi et al. 1996). Warns when EPV < threshold;
    errors when `rank_deficient_action="error"`. Pre-estimation check via
    `diagnose_propensity()`. Results stored in `results.epv_diagnostics`.
  - Fallback: Controlled by `pscore_fallback` parameter (default `"error"`).
    If IRLS fails entirely (LinAlgError/ValueError) and `pscore_fallback="error"`,
    the error is raised. If `pscore_fallback="unconditional"`, falls back to
    unconditional propensity score with warning. For IPW, this effectively
    drops all covariates. For DR, the propensity model is unconditional but
    the outcome-regression component still uses covariates.
  - **Note:** `pscore_fallback` default changed from unconditional to error.
    Set `pscore_fallback="unconditional"` for legacy behavior.
  - **Note:** When `pscore_fallback="unconditional"` triggers, the propensity-
    score influence function correction is skipped (constant pscore has zero
    estimation uncertainty). SEs reflect outcome-model uncertainty only.
- Control group with `control_group="not_yet_treated"`:
  - Always excludes cohort g from controls when computing ATT(g,t)
  - This applies to both pre-treatment (t < g) and post-treatment (t >= g) periods
  - For pre-treatment periods: even though cohort g hasn't been treated yet at time t, they are the treated group for this ATT(g,t) and cannot serve as their own controls
  - Control mask: `never_treated OR (first_treat > max(t, base_period) + anticipation AND first_treat != g)`
  - The `max(t, base_period)` ensures controls are untreated at both the evaluation period
    and the base period, preventing contamination when `base_period="universal"` uses
    a base period later than `t` (matching R's `did::att_gt()`)
  - Does not require never-treated units: when all units are eventually treated,
    not-yet-treated cohorts serve as controls for each other (requires ≥2 cohorts)
- **Note:** CallawaySantAnna survey support: weights, strata, PSU, and FPC are all supported for all estimation methods (reg, ipw, dr) with or without covariates. Analytical (`n_bootstrap=0`): aggregated SEs use design-based variance via `compute_survey_if_variance()`. Bootstrap (`n_bootstrap>0`): PSU-level multiplier weights replace analytical SEs for aggregated quantities. IPW and DR with covariates use DRDID panel nuisance IF corrections (Phase 7a: PS IF correction via survey-weighted Hessian/score, OR IF correction via WLS bread and gradient; Sant'Anna & Zhao 2020, Theorem 3.1). Survey weights compose with IPW weights multiplicatively. WIF in aggregation matches R's did::wif() formula. Per-unit survey weights are extracted via `groupby(unit).first()` from the panel-normalized pweight array; on unbalanced panels the pweight normalization (`w * n_obs / sum(w)`) preserves relative unit weights since all IF/WIF formulas use weight ratios (`sw_i / sum(sw)`) where the normalization constant cancels. Scale-invariance tests pass on both balanced and unbalanced panels.
- **Note (deviation from R):** Panel DR control augmentation is normalized by treated mass (`sw_t_sum` or `n_t`) rather than control IPW mass (`sum(w_cont)`). R's `DRDID::drdid_panel` uses `mean(w.cont)` as the control normalizer. Both are consistent asymptotically (under correct model specification, `E[w_cont] = E[D]` so the normalizers converge), but they differ in finite samples when IPW reweighting doesn't perfectly balance. The treated-mass normalization is simpler and matches the `did::att_gt` convention where ATT is defined per treated unit. Aligning to `DRDID::drdid_panel`'s exact `w.cont` normalization is deferred.
- **Note:** PS nuisance IF corrections follow DRDID's M-estimation convention: `asy_lin_rep_psi` is computed on O(1) psi scale (matching R's `asy.lin.rep.ps = score %*% Hessian.ps`), then the correction `asy_lin_rep_psi @ M2` is converted to the library's O(1/n) phi convention via a single `/n` division. OR corrections use the same phi-scale pattern via `solve(X'WX)` (unnormalized Hessian).
- **Note (deviation from R):** CallawaySantAnna survey reg+covariates per-cell SE uses a conservative plug-in IF based on WLS residuals. The treated IF is `inf_treated_i = (sw_i/sum(sw_treated)) * (resid_i - ATT)` (normalized by treated weight sum, matching unweighted `(resid-ATT)/n_t`). The control IF is `inf_control_i = -(sw_i/sum(sw_control)) * wls_resid_i` (normalized by control weight sum, matching unweighted `-resid/n_c`). SE is computed as `sqrt(sum(sw_t_norm * (resid_t - ATT)^2) + sum(sw_c_norm * resid_c^2))`, the weighted analogue of the unweighted `sqrt(var_t/n_t + var_c/n_c)`. This omits the semiparametrically efficient nuisance correction from DRDID's `reg_did_panel` — WLS residuals are orthogonal to the weighted design matrix by construction, so the first-order IF term is asymptotically valid but may be conservative. SEs pass weight-scale-invariance tests. The efficient DRDID correction is deferred to future work.
- **Note (deviation from R):** Per-cell ATT(g,t) SEs under survey weights use influence-function-based variance (matching R's `did::att_gt` analytical SE path) rather than full Taylor-series linearization. When strata/PSU/FPC are present, analytical aggregated SEs (`n_bootstrap=0`) use `compute_survey_if_variance()` on the combined IF/WIF; bootstrap aggregated SEs (`n_bootstrap>0`) use PSU-level multiplier weights.

- **Note:** Repeated cross-sections (`panel=False`, Phase 7b): supports surveys like BRFSS, ACS annual, and CPS monthly where units are not followed over time. Uses cross-sectional DRDID (Sant'Anna & Zhao 2020, Section 4): `reg` matches `DRDID::reg_did_rc` (Eq 2.2), `dr` matches `DRDID::drdid_rc` (locally efficient, Eq 3.3+3.4 with 4 OLS fits), `ipw` matches `DRDID::std_ipw_did_rc`. Per-observation influence functions instead of per-unit. All three estimation methods support covariates and survey weights.
- **Note:** Panel and RCS influence functions use the library-wide `phi_i = psi_i / n` convention (SE = `sqrt(sum(phi^2))`, algebraically equivalent to R's `sd(psi)*sqrt(n-1)/n`). Leading IF terms are computed on psi scale and divided by n; PS nuisance corrections are computed on psi scale (`score @ solve(Hessian)`) with a single `/n` conversion to phi.
- **Note:** Non-survey DR path also includes nuisance IF corrections (PS + OR), matching the survey path structure (Phase 7a). Previously used plug-in IF only.

**Reference implementation(s):**
- R: `did::att_gt()` (Callaway & Sant'Anna's official package)
- Stata: `csdid`

**Requirements checklist:**
- [ ] Requires never-treated units when `control_group="never_treated"` (default); not required for `"not_yet_treated"`
- [ ] Bootstrap weights support Rademacher, Mammen, Webb distributions
- [ ] Aggregations: simple, event_study, group all implemented
- [ ] Doubly robust estimation when covariates provided
- [ ] Multiplier bootstrap preserves panel structure
- [x] Repeated cross-sections (`panel=False`) for non-panel surveys (Phase 7b)

---

## ChaisemartinDHaultfoeuille

**Primary sources:**
- [de Chaisemartin, C. & D'Haultfœuille, X. (2020). Two-Way Fixed Effects Estimators with Heterogeneous Treatment Effects. *American Economic Review*, 110(9), 2964-2996.](https://doi.org/10.1257/aer.20181169)
- [de Chaisemartin, C. & D'Haultfœuille, X. (2022, revised 2024). Difference-in-Differences Estimators of Intertemporal Treatment Effects. NBER Working Paper 29873.](https://www.nber.org/papers/w29873) — Web Appendix Section 3.7.3 contains the cohort-recentered plug-in variance formula implemented here.

**Phase 1 scope:** Ships the contemporaneous-switch estimator `DID_M` from the AER 2020 paper, equivalently `DID_1` (horizon `l = 1`) of the dynamic companion paper. The full multi-phase rollout is in `ROADMAP.md`: Phase 2 adds dynamic horizons `DID_l` for `l > 1`, normalized estimators, cost-benefit aggregates, and sup-t bands; Phase 3 adds covariate adjustment (`DID^X`), group-specific linear trends (`DID^{fd}`), state-set-specific trends, and HonestDiD integration. Survey design support is deferred to a separate effort after all phases ship. **This is the only modern staggered estimator in the library that handles non-absorbing (reversible) treatments** — treatment can switch on AND off over time, making it the natural fit for marketing campaigns, seasonal promotions, on/off policy cycles.

**Key implementation requirements:**

*Assumption checks / warnings:*
- Treatment must be binary (0/1). Phase 3 will accept non-binary; Phase 1 raises `ValueError` for non-binary input.
- NaN values in `treatment` or `outcome` columns raise `ValueError` early in `fit()` (no silent drops).
- Treatment must be constant within each `(g, t)` cell. Within-cell-varying treatment (fractional `d_gt` after aggregation) raises `ValueError`. Pre-aggregate your data to constant binary cell-level treatment before fitting. Fuzzy DiD is deferred to a separate dCDH 2018 paper not covered by Phase 1.
- Multi-switch groups (those that switch treatment more than once across periods) are dropped before estimation when `drop_larger_lower=True` (the default, matching R `DIDmultiplegtDYN`). Each drop emits a warning with the count and example group IDs. See the multi-switch Note below.
- Singleton-baseline groups — groups whose `D_{g,1}` value is unique in the post-drop dataset — are excluded from the **variance computation only** (per footnote 15 of the dynamic paper, they have no cohort peer). They are **retained** in the point-estimate sample as period-based stable controls. Each emits a warning. See the singleton-baseline Note below.
- Never-switching groups (`S_g = 0`) participate in the variance computation when they serve as stable controls under the full influence function. The `n_groups_dropped_never_switching` results field is reported for backwards compatibility but the count no longer represents an actual exclusion.
- **Balanced-baseline panel required (deviation from R `DIDmultiplegtDYN`).** Every group must have an observation at the **first global period** (the panel's earliest time value); groups missing this baseline raise `ValueError` with the offending group IDs. Groups with **interior period gaps** (missing observations between their first and last observed period) are dropped with a `UserWarning`. **Terminal missingness** (groups observed at the baseline but missing one or more *later* periods) is **retained**: the group contributes from its observed periods only, masked out of the missing transitions by the per-period `present = (N_mat[:, t] > 0) & (N_mat[:, t-1] > 0)` guard. See the ragged-panel deviation Note below.
- **Period-index semantics.** The estimator operates on **sorted period indices**, not calendar dates. Per-period DIDs use `Y_{g,t} - Y_{g,t-1}` where `t-1` is the *previous observed period in the sorted panel*, not the previous calendar unit. A panel with periods `[2000, 2001, 2003]` (missing year 2002 for ALL groups) is treated as a valid 3-period panel where 2003 is the immediate successor of 2001. The estimator does NOT validate that periods are evenly spaced or that calendar gaps have been imputed. This matches the AER 2020 paper's Theorem 3, which defines transition sets by adjacent sorted periods without assuming calendar regularity, and is consistent with R `DIDmultiplegtDYN`'s behavior. If your data has calendar gaps that should be treated as missing periods rather than adjacent transitions, insert placeholder rows for the missing periods with the group's lagged treatment value and a reasonable imputed outcome (e.g., the group's last observed outcome), so the cell-aggregation step treats the gap as a stable-treatment period rather than a missing one. The validator rejects NaN in outcome and treatment columns, so placeholders must have finite values.
- Per-period Assumption 11 violations (joiners exist but no stable-untreated controls in some period, or leavers exist but no stable-treated controls) trigger zero-retention behavior with a consolidated warning. See the A11 Note below.

*Estimator equations (Theorem 3 of AER 2020 / Section 3.7.2 of the dynamic paper):*

Per-period DiDs at each switching period `t >= 2`:

```
DID_{+,t} = (1/N_{1,0,t}) * sum_{g in joiners(t)} (Y_{g,t} - Y_{g,t-1})
          - (1/N_{0,0,t}) * sum_{g in stable_0(t)} (Y_{g,t} - Y_{g,t-1})

DID_{-,t} = (1/N_{1,1,t}) * sum_{g in stable_1(t)} (Y_{g,t} - Y_{g,t-1})
          - (1/N_{0,1,t}) * sum_{g in leavers(t)} (Y_{g,t} - Y_{g,t-1})
```

where `joiners(t)` are groups switching from `D_{g,t-1}=0` to `D_{g,t}=1`, `leavers(t)` are groups switching `1->0`, `stable_0(t)` are groups with `D_{g,t-1}=D_{g,t}=0`, and `stable_1(t)` are groups with `D_{g,t-1}=D_{g,t}=1`. **`N_{a,b,t}` is the COUNT of `(g, t)` cells in each transition state — not the sum of within-cell observation counts.** Each `(g, t)` cell contributes once to its transition's count regardless of how many original observations fed into the cell mean. The cell mean `Y_{g,t}` is computed at the cell-aggregation step via `groupby([group, time]).agg(y_gt=mean)`; the per-period DIDs use these cell means directly without further sample-size weighting. This matches the AER 2020 paper's cell-level notation for `N_{a,b,t}` as a count of transition-state cells (the paper can also be read as using observation sums; the equal-cell interpretation is the one implemented here). **Note (deviation from R `DIDmultiplegtDYN`):** On individual-level inputs with uneven `(group, time)` cell sizes, Python gives each cell **equal weight** (paper-literal cell-count weighting). R `DIDmultiplegtDYN`, absent an explicit weight variable, weights estimation by the number of observations in each cell (cell-size weighting). The two agree exactly on cell-aggregated input where every cell has the same number of observations. The Python parity tests in `tests/test_chaisemartin_dhaultfoeuille_parity.py` use the `generate_reversible_did_data()` generator, which produces exactly one observation per cell, so parity holds. The regression test `test_cell_count_weighting_unbalanced_input` in `tests/test_chaisemartin_dhaultfoeuille.py` explicitly pins the equal-cell contract.

Aggregate `DID_M`:

```
N_S   = sum_{t>=2} (N_{1,0,t} + N_{0,1,t})
DID_M = (1/N_S) * sum_{t>=2} (N_{1,0,t} * DID_{+,t} + N_{0,1,t} * DID_{-,t})
```

Joiners-only and leavers-only views (each weighted by its own switcher count):

```
DID_+ = sum_{t>=2} (N_{1,0,t} / sum_{t} N_{1,0,t}) * DID_{+,t}
DID_- = sum_{t>=2} (N_{0,1,t} / sum_{t} N_{0,1,t}) * DID_{-,t}
```

Single-lag placebo (AER 2020 placebo specification, same section as Theorem 3) — applies the same Theorem 3 logic to `Y_{g,t-1} - Y_{g,t-2}` on cells with 3-period histories:

```
DID_M^pl = (1/N_S^pl) * sum_{t>=3} (
              N_{1,0,t} * [(Y_{g,t-1} - Y_{g,t-2})_{joiners} - ...] +
              N_{0,1,t} * [(Y_{g,t-1} - Y_{g,t-2})_{stable_1} - ...]
          )
```

*Standard errors (Web Appendix Section 3.7.3 of the dynamic companion paper):*

Default: cohort-recentered analytical plug-in variance, evaluated at horizon `l = 1`. Cohorts are defined by the triple `(D_{g,1}, F_g, S_g)` (baseline treatment, first-switch period, switch direction). Each group's per-period role weights (joiner, stable_0, leaver, stable_1) sum to a per-group `U^G_g` value via the full `Lambda^G_{g,l=1}` weight vector from Section 3.7.2 of the dynamic paper:

```
N_S * DID_M = sum_t [
      sum_{g in joiners(t)}  (Y_{g,t} - Y_{g,t-1})
    - (N_{1,0,t} / N_{0,0,t}) * sum_{g in stable_0(t)} (Y_{g,t} - Y_{g,t-1})
    + (N_{0,1,t} / N_{1,1,t}) * sum_{g in stable_1(t)} (Y_{g,t} - Y_{g,t-1})
    - sum_{g in leavers(t)}  (Y_{g,t} - Y_{g,t-1})
]
```

Reading off the coefficient on each `(Y_{g,t} - Y_{g,t-1})` gives the per-cell role weight, which sums across periods to:

```
U^G_g     = sum_t lambda^G_{g,t} * (Y_{g,t} - Y_{g,t-1})    # full IF
U_bar_k   = (1/|C_k|) * sum_{g in C_k} U^G_g                # cohort-conditional mean
sigma_hat^2 = sum_g (U^G_g - U_bar_{cohort(g)})^2 / N_l
SE         = sqrt(sigma_hat^2 / N_l)
```

Each switching group typically contributes from MULTIPLE periods: its own switch period plus every period where it serves as a stable control for another cohort's switch. Never-switching groups can also have non-zero `U^G_g` when they serve as stable controls. Singleton-baseline groups (footnote 15 of dynamic paper) are excluded from this sum because they have no cohort peer.

The cohort recentering is critical: subtracting cohort-conditional means is **not** the same as subtracting a single grand mean. The implementation has a dedicated regression test (`test_cohort_recentering_not_grand_mean`) that computes both formulas on a designed DGP and asserts they differ materially.

Alternative: Multiplier bootstrap clustered at group via the `n_bootstrap` parameter. Available weight distributions: `"rademacher"` (default), `"mammen"`, `"webb"`. The bootstrap is a library extension beyond the original papers and is provided for consistency with `CallawaySantAnna` / `ImputationDiD` / `TwoStageDiD`.

*Edge cases:*
- **No switchers in data** (after filtering): raises `ValueError` with a clear message indicating which filters dropped which groups.
- **No joiners** (only leavers in data): `joiners_available = False`, all `joiners_*` fields are `NaN`. Symmetric for `leavers_available = False`.
- **`T < 3`**: placebo cannot be computed; `placebo_available = False` with a `UserWarning`.
- **NaN inference**: `safe_inference()` produces NaN-consistent inference fields (t-stat, p-value, conf int) when SE is non-finite or zero. `assert_nan_inference()` is used in tests to enforce consistency.
- **TWFE diagnostic with zero denominator**: when `sum(d_gt - d_bar)^2 == 0` (e.g., all cells have identical treatment), the diagnostic returns NaN for `beta_fe` and `sigma_fe` with a `UserWarning`. The diagnostic is non-fatal — it does not block the main estimation.
- **`placebo=False`** (gating): the results object still exposes `placebo_*` fields, but with `NaN` values and `placebo_available = False`. This keeps the API surface stable.

- **Note:** The analytical CI is **conservative** under Assumption 8 (independent groups) of the dynamic companion paper, and exact only under iid sampling. This is documented as a deliberate deviation from "default nominal coverage". The bootstrap CI uses the same conservative weighting and is provided for users who want a non-asymptotic alternative.

- **Note:** Phase 1 placebo SE is intentionally `NaN` with a `UserWarning`. The dynamic companion paper Section 3.7.3 derives the cohort-recentered analytical variance for `DID_l` only — not for the placebo `DID_M^pl`. Phase 2 will add multiplier-bootstrap support for the placebo via the dynamic paper's machinery. Until then, the placebo point estimate is meaningful but its inference fields stay NaN-consistent **even when `n_bootstrap > 0`**: the bootstrap path computes SEs for `DID_M`, `DID_+`, and `DID_-`, but `placebo_se`, `placebo_t_stat`, `placebo_p_value`, and `placebo_conf_int` remain `NaN` because the placebo's influence function machinery is deferred to Phase 2.

- **Note:** When every variance-eligible group forms its own `(D_{g,1}, F_g, S_g)` cohort (a degenerate small-panel case where the cohort framework has zero degrees of freedom), the cohort-recentered plug-in formula is unidentified: cohort recentering subtracts the cohort mean from each group's `U^G_g`, and for singleton cohorts the centered value is exactly zero, so the centered influence function vector collapses to all zeros. The estimator returns `overall_se = NaN` with a `UserWarning` rather than silently collapsing to `0.0` (which would falsely imply infinite precision). The `DID_M` point estimate remains well-defined. The bootstrap path inherits the same degeneracy on these panels — the multiplier weights act on an all-zero vector, so the bootstrap distribution is also degenerate. **Deviation from R `DIDmultiplegtDYN`:** R returns a non-zero SE on the canonical 4-group worked example via small-sample sandwich machinery that Python does not implement. Both responses are valid for a degenerate case; Python's `NaN`+warning is the safer default. To get a non-degenerate SE, include more groups so cohorts have peers (real-world panels typically have `G >> K`).

- **Note (Phase 1 cluster contract):** `ChaisemartinDHaultfoeuille` always clusters at the group level. The cohort-recentered analytical SE plug-in operates on per-group influence-function values (one `U^G_g` per group); the multiplier bootstrap generates one weight per group; both inference paths cluster at the user's `group` column with no other option. The constructor accepts `cluster=None` (the default and only supported value); passing any non-`None` value raises `NotImplementedError` with a Phase 1 pointer at construction time (and the same gate fires from `set_params`). Custom clustering at a coarser or finer level than the group is reserved for a future phase. The matching test is `test_cluster_parameter_raises_not_implemented` in `tests/test_chaisemartin_dhaultfoeuille.py::TestForwardCompatGates`.

- **Note (bootstrap inference surface):** When `n_bootstrap > 0`, the top-level `results.overall_p_value` / `results.overall_conf_int` (and joiners/leavers analogues) hold **percentile-based bootstrap inference** computed by the multiplier bootstrap, NOT normal-theory recomputations from the bootstrap SE. The t-stat (`overall_t_stat`, etc.) is computed from the SE via `safe_inference()[0]` to satisfy the project's anti-pattern rule (never compute `t = effect / se` inline) — bootstrap does not define an alternative t-stat semantic for percentile bootstrap, so the SE-based t-stat is the natural choice. `event_study_effects[1]`, `summary()`, `to_dataframe()`, `is_significant`, and `significance_stars` all read from these top-level fields and therefore reflect the bootstrap inference automatically. The library precedent for this propagation is `imputation.py:790-805`, `two_stage.py:778-787`, and `efficient_did.py:1009-1013`. The placebo path is unchanged: placebo bootstrap is deferred to Phase 2 (see the placebo SE Note above), so `placebo_p_value` and `placebo_conf_int` stay NaN even when `n_bootstrap > 0`. The matching test is `test_bootstrap_p_value_and_ci_propagated_to_top_level` in `tests/test_chaisemartin_dhaultfoeuille.py::TestBootstrap`.

- **Note:** Placebo Assumption 11 violations (placebo joiners exist but no 3-period stable_0 controls, or symmetric for leavers/stable_1) trigger zero-retention in the placebo numerator AND emit a consolidated `Placebo (DID_M^pl) Assumption 11 violations` warning from `fit()`, mirroring the main DID path's contract documented above. The zeroed placebo periods retain their switcher counts in the placebo `N_S^pl` denominator, biasing `DID_M^pl` toward zero in the offending direction (matching the placebo paper convention).

- **Note (TWFE diagnostic sample contract):** The fitted `results.twfe_weights` / `results.twfe_fraction_negative` / `results.twfe_sigma_fe` / `results.twfe_beta_fe` are computed on the **FULL pre-filter cell sample** — the data the user passed in, after `_validate_and_aggregate_to_cells()` runs but **before** the ragged-panel validation (Step 5b) and the multi-switch filter (`drop_larger_lower`, Step 6). They do NOT describe the post-filter estimation sample used by `overall_att`, `results.groups`, and the inference fields. `fit()` has three sample-shaping filters in total: (1) interior-gap drops in Step 5b, (2) multi-switch drops in Step 6, and (3) the singleton-baseline filter in Step 7. Filters (1) and (2) actually shrink the point-estimate sample, so when either fires, the fitted TWFE diagnostic and `overall_att` describe **different samples** and the estimator emits a `UserWarning` explaining the divergence with explicit counts. Filter (3) is **variance-only** — singleton-baseline groups remain in the point-estimate sample as period-based stable controls (see the singleton-baseline Note above) — so it does NOT create a fitted-vs-`overall_att` mismatch and does NOT trigger the divergence warning. Rationale for the pre-filter design: the TWFE diagnostic answers "what would the plain TWFE estimator say on the data you passed in?" — not "what would TWFE say on the data dCDH actually used after filtering?" — so users comparing TWFE vs dCDH on a fixed input can do so without an interaction effect from the dCDH-specific filters. The standalone `twowayfeweights()` function uses the same pre-filter sample, so the fitted and standalone APIs always produce identical numbers on the same input. To reproduce the dCDH estimation sample for an external TWFE comparison, pre-process your data to drop the multi-switch and interior-gap groups before fitting (the warning lists offending IDs). The matching tests are `test_twfe_pre_filter_contract_with_interior_gap_drop` and `test_twfe_pre_filter_contract_with_multi_switch_drop` in `tests/test_chaisemartin_dhaultfoeuille.py`.

- **Note:** By default (`drop_larger_lower=True`), the estimator drops groups whose treatment switches more than once before estimation. This matches R `DIDmultiplegtDYN`'s default and is required for the analytical variance formula (Web Appendix Section 3.7.3 of the dynamic paper, which assumes Assumption 5 / no-crossing) to be consistent with the AER 2020 Theorem 3 point estimate. Both formulas operate on the same post-drop dataset. Setting `drop_larger_lower=False` is supported for diagnostic comparison but produces an inconsistent estimator-variance pairing for any multi-switch groups present, and emits an explicit warning.

- **Note:** When Assumption 11 (existence of stable controls) is violated for some period `t` — i.e., joiners exist but no stable-untreated controls, or leavers exist but no stable-treated controls — `DID_{+,t}` (or `DID_{-,t}`) is set to zero by paper convention, and the period's switcher count is **retained** in the `N_S` denominator. This means the affected period contributes a zero to the numerator with a non-zero weight in the denominator, biasing `DID_M` toward zero in the offending direction. Users can detect this by inspecting `results.per_period_effects[t]['did_plus_t_a11_zeroed']` (or `did_minus_t_a11_zeroed`) or the consolidated `fit()` warning. This matches the AER 2020 Theorem 3 paper convention and the worked example arithmetic.

- **Note:** Groups whose baseline treatment value `D_{g,1}` is unique in the post-drop panel (not shared by any other group) are excluded from the **variance computation only** per footnote 15 of the dynamic companion paper. They have no cohort peer for the cohort-recentered plug-in formula. They are **retained in the point-estimate sample** as period-based stable controls (Python's documented period-vs-cohort interpretation). The dropped count is stored on `results.n_groups_dropped_singleton_baseline`, a warning lists example group IDs, and the warning text explicitly states "VARIANCE computation only" so users know the filter does not change `DID_M`.

- **Note (deviation from R DIDmultiplegtDYN):** Python uses **period-based** stable-control sets — `stable_0(t)` is any cell with `D_{g,t-1} = D_{g,t} = 0` regardless of baseline `D_{g,1}`, and similarly for `stable_1(t)`. R `DIDmultiplegtDYN` uses **cohort-based** stable-control sets that additionally require `D_{g,1}` to match the side. Python's definition matches the AER 2020 Theorem 3 cell-count notation `N_{0,0,t}` and `N_{1,1,t}` literally; R's definition matches the dynamic companion paper's cohort `(D_{g,1}, F_g, S_g)` framework. The two definitions agree exactly on (a) panels containing only joiners, (b) panels containing only leavers, (c) the hand-calculable 4-group worked example, or (d) any panel where no joiner's post-switch state overlaps a period when leavers are switching. They disagree by O(1%) on the **point estimate** when both joiners and leavers exist AND some joiners' post-switch cells could serve as leavers' controls (or vice versa). After the Round 2 fix that implemented the full `Lambda^G_{g,l=1}` influence function, the **standard error** parity gap on pure-direction scenarios narrowed from ~18% to ~3%. The R parity tests in `tests/test_chaisemartin_dhaultfoeuille_parity.py` use a tight `1e-4` tolerance for pure-direction point estimates, a 5% rtol for pure-direction SEs, and a 2.5% tolerance for mixed-direction point estimates (with the SE check skipped on mixed scenarios because the period-vs-cohort point-estimate deviation cascades into the variance).

- **Note (deviation from R DIDmultiplegtDYN):** Phase 1 requires panels with a **balanced baseline** (every group observed at the first global period) and **no interior period gaps**. The Step 5b validation in `fit()` enforces this contract: groups missing the baseline raise `ValueError`; groups with interior gaps are dropped with a `UserWarning`; groups with **terminal missingness** (early exit / right-censoring — observed at the baseline but missing one or more later periods) are retained and contribute from their observed periods only. R `DIDmultiplegtDYN` accepts unbalanced panels with documented missing-treatment-before-first-switch handling. Python's restriction is a Phase 1 limitation: the cohort enumeration uses `D_{g,1}` as the canonical baseline (so the baseline observation must exist) and the first-switch detection walks adjacent observed periods (so interior gaps create ambiguous transition counts). Terminal missingness is supported because the per-period `present = (N_mat[:, t] > 0) & (N_mat[:, t-1] > 0)` guard appears at three sites in the variance computation (`_compute_per_period_dids`, `_compute_full_per_group_contributions`, `_compute_cohort_recentered_inputs`) and cleanly masks out missing transitions without propagating NaN into the arithmetic. **Workaround for unbalanced panels:** pre-process your data to back-fill the baseline (or drop late-entry groups before fitting), or use R `DIDmultiplegtDYN` until a future phase lifts the restriction. The Step 5b `ValueError` and `UserWarning` messages name the offending group IDs so you can locate them quickly.

**Reference implementation(s):**
- R: [`DIDmultiplegtDYN`](https://cran.r-project.org/package=DIDmultiplegtDYN) (CRAN, maintained by the paper authors). The Python implementation matches `did_multiplegt_dyn(..., effects=1)` at horizon `l = 1`. Parity tests live in `tests/test_chaisemartin_dhaultfoeuille_parity.py`.
- Stata: `did_multiplegt_dyn` (SSC, also maintained by the paper authors).

**Requirements checklist:**
- [x] Single class `ChaisemartinDHaultfoeuille` (alias `DCDH`); not a family
- [x] Forward-compat `fit()` signature with `NotImplementedError` gates for Phase 2/3 parameters (`aggregate`, `L_max`, `controls`, `trends_linear`, `trends_nonparam`, `honest_did`, `survey_design`)
- [x] `DID_M` point estimate with cohort-recentered analytical SE
- [x] Joiners-only `DID_+` and leavers-only `DID_-` decompositions with their own inference
- [x] Single-lag placebo `DID_M^pl` (point estimate; SE deferred to Phase 2)
- [x] TWFE decomposition diagnostic (Theorem 1 of AER 2020): per-cell weights, fraction negative, `sigma_fe`, `beta_fe`
- [x] Standalone `twowayfeweights()` helper for users who only want the TWFE diagnostic
- [x] Multiplier bootstrap with Rademacher / Mammen / Webb weights, clustered at group
- [x] `drop_larger_lower=True` default (matches R `DIDmultiplegtDYN`); `False` opt-in with explicit inconsistency warning
- [x] Singleton-baseline filter (footnote 15 of dynamic paper, variance computation only) with explicit warning
- [x] Never-switching groups participate in the variance via stable-control roles after the Round 2 full-IF fix; `n_groups_dropped_never_switching` field retained as backwards-compatibility metadata only
- [x] Balanced-baseline panel requirement: missing-baseline groups raise `ValueError`; interior-gap groups dropped with `UserWarning`; terminal missingness retained (deviation from R `DIDmultiplegtDYN` documented as a Note)
- [x] A11 zero-retention convention with per-period boolean flags (`did_plus_t_a11_zeroed` / `did_minus_t_a11_zeroed`) and consolidated warning
- [x] No silent failures: every drop / round / fallback emits a `warnings.warn()` or `ValueError`
- [x] Hand-calculable 4-group worked example: `DID_M = 2.5`, `DID_+ = 2.0`, `DID_- = 3.0` exactly
- [x] R `DIDmultiplegtDYN` parity tests at `l = 1` (fixture skips cleanly when R or `DIDmultiplegtDYN` is unavailable)

---

## ContinuousDiD

**Primary Source:** Callaway, Goodman-Bacon & Sant'Anna (2024), "Difference-in-Differences with a Continuous Treatment," NBER Working Paper 32117.

**R Reference:** `contdid` v0.1.0 (CRAN).

### Identification

Two levels of parallel trends (following CGBS 2024, Assumptions 1-2):

**Parallel Trends (PT):** for all doses d in D_+,
`E[Y_t(0) - Y_{t-1}(0) | D = d] = E[Y_t(0) - Y_{t-1}(0) | D = 0]`.
Untreated potential outcome paths are the same across all dose groups and the
untreated group. Stronger than binary PT because it conditions on specific dose values.
Identifies: `ATT(d|d)`, `ATT^{loc}`. Does NOT identify `ATT(d)`, `ACRT`, or cross-dose comparisons.

**Strong Parallel Trends (SPT):** additionally, for all d in D,
`E[Y_t(d) - Y_{t-1}(0) | D > 0] = E[Y_t(d) - Y_{t-1}(0) | D = d]`.
No selection into dose groups on the basis of treatment effects.
Implies `ATT(d|d) = ATT(d)` for all d.
Additionally identifies: `ATT(d)`, `ACRT(d)`, `ACRT^{glob}`, and cross-dose comparisons.

See `docs/methodology/continuous-did.md` Section 4 for full details.

### Key Equations

**Target parameters:**
- `ATT(d|d) = E[Y_t(d) - Y_t(0) | D = d]` — effect of dose d on units who received dose d (PT)
- `ATT(d) = E[Y_t(d) - Y_t(0) | D > 0]` — dose-response curve (SPT required)
- `ACRT(d) = dATT(d)/dd` — average causal response / marginal effect (SPT required)
- `ATT^{loc} = E[ATT(D|D) | D > 0] = E[Delta Y | D > 0] - E[Delta Y | D = 0]` — binarized ATT (PT); equals `ATT^{glob}` under SPT
- `ATT^{glob} = E[ATT(D) | D > 0]` — global average dose-response level (SPT required)
- `ACRT^{glob} = E[ACRT(D_i) | D > 0]` — plug-in average marginal effect (SPT required)

**Estimation via B-spline OLS:**
1. Compute `Delta_tilde_Y = (Y_t - Y_{t-1})_treated - mean((Y_t - Y_{t-1})_control)`
2. Build B-spline basis `Psi(D_i)` from treated doses
3. OLS: `beta = (Psi'Psi)^{-1} Psi' Delta_tilde_Y`
4. `ATT(d) = Psi(d)' beta`, `ACRT(d) = dPsi(d)/dd' beta`

### Edge Cases

- **No untreated group**: Remark 3.1 (lowest-dose-as-control) not implemented; requires P(D=0) > 0.
- **Discrete treatment**: Detect integer-valued dose and warn; saturated regression deferred.
- **All-same dose**: B-spline basis collapses; ACRT(d) = 0 everywhere.
- **Rank deficiency**: When n_treated <= n_basis, cell is skipped.
- **Balanced panel required**: Matches R `contdid` v0.1.0.
- **Anticipation + not-yet-treated**: Control mask uses `G > t + anticipation`
  (not just `G > t`) to exclude cohorts in the anticipation window from
  not-yet-treated controls. When `anticipation=0` (default), behavior is
  unchanged.
- **Boundary knots**: Knots are built once from all treated doses (global, not per-cell) to ensure a common basis across (g,t) cells for aggregation. Evaluation grid is clamped to training-dose boundary knots (`range(dose)`). R's `contdid` v0.1.0 has an inconsistency where `splines2::bSpline(dvals)` uses `range(dvals)` instead of `range(dose)`, which can produce extrapolation artifacts at dose grid extremes. Our approach avoids extrapolation and is methodologically sound.

### Implementation Checklist

- [x] B-spline basis construction matching R's `splines2::bSpline` (global knots from all treated doses; boundary knots use training-dose range; see deviation note above)
- [x] Multi-period (g,t) cell iteration with base period selection
- [x] Dose-response and event-study aggregation with group-proportional weights (n_treated/n_total per group, divided among post-treatment cells; R `ptetools` convention)
- [x] Multiplier bootstrap for inference
- [x] Analytical SEs via influence functions
- [x] Equation verification tests (linear, quadratic, multi-period)
- [ ] Covariate support (deferred, matching R v0.1.0)
- [ ] Discrete treatment saturated regression
- [ ] Lowest-dose-as-control (Remark 3.1)
- [x] Survey design support (Phase 3): weighted B-spline OLS, TSL on influence functions; bootstrap+survey supported (Phase 6)
- **Note:** ContinuousDiD bootstrap with survey weights supported (Phase 6) via PSU-level multiplier weights

---

## EfficientDiD

**Primary source:** Chen, X., Sant'Anna, P. H. C., & Xie, H. (2025). Efficient Difference-in-Differences and Event Study Estimators.

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Random Sampling (Assumption S)**: Data is a random sample of `(Y_{1}, ..., Y_{T}, X', G)'`
- **Overlap (Assumption O)**: For each group g, generalized propensity score `E[G_g | X]` must be in `(0, 1)` a.s. Near-zero propensity scores cause ratio `p_g(X)/p_{g'}(X)` to explode; warn on finite-sample instability
- **No-anticipation (Assumption NA)**: For all treated groups g and pre-treatment periods t < g: `E[Y_t(g) | G=g, X] = E[Y_t(infinity) | G=g, X]` a.s.
- **Parallel Trends -- two variants**:
  - **PT-Post** (weaker): PT holds only in post-treatment periods, comparison group = never-treated only, baseline = period g-1 only. Estimator is just-identified and reduces to standard single-baseline DiD (Corollary 3.2)
  - **PT-All** (stronger): PT holds for all groups and all periods. Enables using any not-yet-treated cohort and any pre-treatment period as baseline. Model is overidentified (Lemma 2.1); paper derives optimal combination weights
- **Absorbing treatment**: Binary treatment must be irreversible (once treated, stays treated)
- **Balanced panel**: Short balanced panel required ("large-n, fixed-T" regime). Does not handle unbalanced panels or repeated cross-sections
- Warn if treatment varies within units (non-absorbing treatment)
- Warn if propensity score estimates are near boundary values

*Estimator equation -- single treatment date (Equations 3.2, 3.5):*

Transformed outcome (Equation 3.2):
```
Y_tilde_{g,t,t_pre} = (1/pi_g) * (G_g - p_g(X)/p_inf(X) * G_inf) * (Y_t - Y_{t_pre} - m_{inf,t,t_pre}(X))
```

Efficient ATT estimand (Equation 3.5):
```
ATT(g, t) = E[ (1' V*_{gt}(X)^{-1} / (1' V*_{gt}(X)^{-1} 1)) * Y_tilde_{g,t} ]
```

where:
- `G_g = 1{G = g}` = indicator for belonging to treatment cohort g
- `G_inf = 1{G = infinity}` = indicator for never-treated
- `pi_g = P(G = g)` = population share of cohort g
- `p_g(X) = E[G_g | X]` = generalized propensity score
- `m_{inf,t,t_pre}(X) = E[Y_t - Y_{t_pre} | G = infinity, X]` = conditional mean outcome change for never-treated
- `V*_{gt}(X)` = `(g-1) x (g-1)` conditional covariance matrix with `(j,k)`-th element (Equation 3.4):
  ```
  (1/p_g(X)) Cov(Y_t - Y_j, Y_t - Y_k | G=g, X) + (1/(1-p_g(X))) Cov(Y_t - Y_j, Y_t - Y_k | G=inf, X)
  ```

*Estimator equation -- staggered adoption (Equations 3.9, 3.13, 4.3, 4.4):*

Generated outcome for each `(g', t_pre)` pair (Equation 3.9 / sample analog 4.4):
```
Y_hat^{att(g,t)}_{g',t_pre} = (G_g / pi_hat_g) * (Y_t - Y_1 - m_hat_{inf,t,t_pre}(X) - m_hat_{g',t_pre,1}(X))
    - r_hat_{g,inf}(X) * (G_inf / pi_hat_g) * (Y_t - Y_{t_pre} - m_hat_{inf,t,t_pre}(X))
    - r_hat_{g,g'}(X) * (G_{g'} / pi_hat_g) * (Y_{t_pre} - Y_1 - m_hat_{g',t_pre,1}(X))
```

where:
- `r_hat_{g,g'}(X) = p_g(X)/p_{g'}(X)` = estimated propensity score ratio
- `m_hat_{g',t,t_pre}(X) = E[Y_t - Y_{t_pre} | G = g', X]` = estimated conditional mean outcome change

Efficient ATT for staggered adoption (Equation 4.3):
```
ATT_hat_stg(g,t) = E_n[ (1' Omega_hat*_{gt}(X)^{-1}) / (1' Omega_hat*_{gt}(X)^{-1} 1) * Y_hat^{att(g,t)}_stg ]
```

where `Omega*_{gt}(X)` is the conditional covariance matrix with `(j,k)`-th element (Equation 3.12):
```
(1/p_g(X)) Cov(Y_t - Y_1, Y_t - Y_1 | G=g, X)
+ (1/p_inf(X)) Cov(Y_t - Y_{t'_j}, Y_t - Y_{t'_k} | G=inf, X)
- 1{g=g'_j}/p_g(X) * Cov(Y_t - Y_1, Y_{t'_j} - Y_1 | G=g, X)
- 1{g=g'_k}/p_g(X) * Cov(Y_t - Y_1, Y_{t'_k} - Y_1 | G=g, X)
+ 1{g_j=g'_k}/p_{g'_j}(X) * Cov(Y_{t'_j} - Y_1, Y_{t'_k} - Y_1 | G=g'_j, X)
```

*Event study aggregation (Equations 3.8, 3.14, 4.5):*

```
ES_hat(e) = sum_{g in G_{trt,e}}  (pi_hat_g / sum_{g' in G_{trt,e}} pi_hat_{g'})  * ATT_hat_stg(g, g+e)
```

where `G_{trt,e} = {g in G_trt : g + e <= T}` and weights are cohort relative size weights.

Overall average event-study parameter (Equation 2.3):
```
ES_avg = (1/N_E) * sum_{e in E} ES(e)
```

*With covariates / doubly robust:*

The estimator is doubly robust by construction. Consistency requires correct specification of either:
- Outcome regression: `m_{g',t,t_pre}(X) = E[Y_t - Y_{t_pre} | G = g', X]`, OR
- Propensity score ratio: `r_{g,g'}(X) = p_g(X)/p_{g'}(X)`

The Neyman orthogonality property (Remark 4.2) permits modern ML estimators (random forests, lasso, ridge, neural nets, boosted trees) for nuisance parameters without loss of efficiency.

*Without covariates (Section 4.1):*

Estimator simplifies to closed-form expressions using only within-group sample means and sample covariances. **No tuning parameters** are needed. The covariance matrix `Omega*_gt` uses unconditional within-group covariances with `pi_g` replacing `p_g(X)`.

*Standard errors (Theorem 4.1, Section 4):*
- Default: Analytical SE computed as the square root of the sample variance of estimated EIF values divided by n:
  ```
  SE_analytical = sqrt( (1/n^2) * sum_{i=1}^{n} EIF_hat_i^2 )
  ```
- Alternative: Cluster-robust SE at cross-sectional unit level (used in empirical application, page 34-35)
- Bootstrap: Nonparametric clustered bootstrap (resampling clusters with replacement); 300 replications recommended (page 23, footnote 16)
- **Small sample recommendation** (Section 5.1): Use cluster bootstrap SEs rather than analytical SEs when n is small (n <= 50). Analytical SEs are anticonservative with n=50 (coverage ~0.80) but perform well with n >= 200 (coverage ~0.94)
- Simultaneous confidence bands: Multiplier bootstrap procedure for multiple `(g,t)` pairs (footnote 13, referencing Callaway and Sant'Anna 2021, Theorems 2-3, Algorithm 1)
- **Implementation note**: Phase 1 uses multiplier bootstrap on EIF values (Rademacher/Mammen/Webb weights) rather than nonparametric clustered bootstrap. This is asymptotically equivalent and computationally cheaper, consistent with the CallawaySantAnna implementation pattern. Clustered resampling bootstrap may be added in a future version

*Efficient influence function for ATT(g,t) (Theorem 3.2):*
```
EIF^{att(g,t)}_stg = (1' Omega*_{gt}(X)^{-1}) / (1' Omega*_{gt}(X)^{-1} 1) * IF^{att(g,t)}_stg
```

*Efficient influence function for ES(e) (following Theorem 3.2, page 17):*
```
EIF^{es(e)}_stg = sum_{g in G_{trt,e}} ( q_{g,e} * EIF^{att(g,g+e)}_stg
    + ATT(g,g+e) / (sum_{g' in G_{trt,e}} pi_{g'}) * (G_g - pi_g)
    - q_{g,e} * sum_{s in G_{trt,e}} (G_s - pi_s) )
```
where `q_{g,e} = pi_g / sum_{g' in G_{trt,e}} pi_{g'}`.

*Edge cases:*
- **Single pre-treatment period (g=2)**: `V*_{gt}(X)` is 1x1, efficient weights are trivially 1, estimator collapses to standard DiD with single baseline
- **Rank deficiency in `V*_{gt}(X)` or `Omega*_{gt}(X)`**: Inverse does not exist if outcome changes are linearly dependent conditional on covariates. Detect via matrix condition number; fall back to pseudoinverse or standard estimator
- **Near-zero propensity scores**: Ratio `p_g(X)/p_{g'}(X)` explodes. Overlap assumption (O) rules this out in population; implement trimming or warn on finite-sample instability
- **Note:** When no sieve degree K succeeds for ratio estimation (basis dimension exceeds comparison group size, or all linear systems are singular), the estimator falls back to a constant ratio of 1 for all units with a UserWarning. The outcome regression adjustment remains active, so the generated outcomes (Eq 4.4) still incorporate covariate information via the m_hat terms. The DR property ensures consistency as long as the outcome regression is correctly specified.
- **Note:** When no sieve degree K succeeds for inverse propensity estimation (algorithm step 4), the estimator falls back to unconditional n/n_group scaling with a UserWarning, which reduces to the unconditional Omega* approximation for the affected group.
- **All units eventually treated**: Last cohort serves as "never-treated" by dropping time periods from the last cohort's treatment onset onward. Use `control_group="last_cohort"` to enable; default `"never_treated"` raises ValueError if no never-treated units exist
- **Negative weights**: Explicitly stated as harmless for bias and beneficial for precision; arise from efficiency optimization under overidentification (Section 5.2)
- **PT-Post regime (just-identified)**: Under PT-Post, EDiD automatically reduces to standard single-baseline estimator (Corollary 3.2). No downside to using EDiD -- it subsumes standard estimators
- **Duplicate rows**: Duplicate `(unit, time)` entries are rejected with `ValueError`. The estimator requires exactly one observation per unit-period
- **Note:** PT-All index set includes g'=∞ (never-treated) as a candidate comparison group and excludes period_1 for all g'. When g'=∞, the second and third Eq 3.9 terms telescope so all (∞, t_pre) moments produce the same 2x2 DiD value; these redundant moments are handled by Omega*'s pseudoinverse. When t_pre = period_1, the third term degenerates to E[Y_1 - Y_1 | G=g'] = 0 for any g', adding no information. Valid pairs require only t_pre < g' (pre-treatment for comparison group), not t_pre < g. Same-group pairs (g'=g) are valid and contribute overidentifying moments (Equation 3.9).
- **Note:** Bootstrap aggregation uses fixed cohort-size weights for overall/event-study reaggregation, matching the CallawaySantAnna bootstrap pattern (staggered_bootstrap.py:281 computes `bootstrap_overall = bootstrap_atts_gt[:, post_indices] @ weights`; L297 uses the same fixed-weight pattern for event study). The analytical path includes a WIF correction; fixed-weight bootstrap captures the same sampling variability through per-cell EIF perturbation without re-estimating aggregation weights, consistent with both the library's CS implementation and the R `did` package.
- **Overall ATT convention**: The library's `overall_att` uses cohort-size-weighted averaging of post-treatment (g,t) cells, matching the CallawaySantAnna simple aggregation. This differs from the paper's ES_avg (Eq 2.3), which uniformly averages over event-time horizons. ES_avg can be computed from event study output as `mean(event_study_effects[e]["effect"] for e >= 0)`

*Algorithm (two-step semiparametric estimation, Section 4):*

**Step 1: Estimate nuisance parameters**
1. Estimate outcome regressions `m_hat_{g',t,t_pre}(X)` using sieve regression, kernel smoothing, or ML methods (for each valid `(g', t_pre)` pair)
2. Estimate propensity score ratios `r_hat_{g,g'}(X) = p_g(X)/p_{g'}(X)` via convex minimization (Equation 4.1):
   ```
   r_{g,g'}(X) = arg min_{r} E[ r(X)^2 * G_{g'} - 2*r(X)*G_g ]
   ```
   Sieve estimator (Equation 4.2): `beta_hat_K = arg min_{beta_K} E_n[ G_{g'} * (psi^K(X)' beta_K)^2 - 2*G_g * (psi^K(X)' beta_K) ]`
3. Select sieve index K via information criterion: `K_hat = arg min_K { 2*loss(K) + C_n * K / n }` where `C_n = 2` (AIC) or `C_n = log(n)` (BIC)
4. Estimate `s_hat_{g'}(X) = 1/p_{g'}(X)` via analogous convex minimization
5. Estimate conditional covariance `Omega_hat*_{gt}(X)` using kernel smoothing with bandwidth h

**Step 2: Construct efficient estimator**
6. Compute generated outcomes `Y_hat^{att(g,t)}_{g',t_pre}` for each valid `(g', t_pre)` pair using Equation 4.4
7. Compute efficient weights `w(X) = 1' Omega_hat*_{gt}(X)^{-1} / (1' Omega_hat*_{gt}(X)^{-1} 1)`
8. Compute `ATT_hat_stg(g,t) = E_n[ w(X_i) * Y_hat^{att(g,t)}_stg ]` (Equation 4.3)
9. Aggregate to event-study: `ES_hat(e) = sum_g (pi_hat_g / sum pi_hat) * ATT_hat_stg(g, g+e)` (Equation 4.5)
10. Compute SE from sample variance of estimated EIF values

**Without covariates**: Steps 1-5 simplify to within-group sample means and sample covariances. No nuisance estimation or tuning needed.

**Reference implementation(s):**
- No specific software package named in the paper for the EDiD estimator
- Estimators compared against: Callaway-Sant'Anna (`did` R package), de Chaisemartin-D'Haultfoeuille (`DIDmultiplegt` R package / `did_multiplegt` Stata), Borusyak-Jaravel-Spiess / Gardner / Wooldridge imputation estimators
- Empirical replication: HRS data from Dobkin et al. (2018) following Sun and Abraham (2021) sample selection

**Requirements checklist:**
- [x] Implements two-step semiparametric estimator (Equation 4.3)
- [x] Supports both PT-Post (just-identified) and PT-All (overidentified) regimes
- [x] Computes efficient weights from conditional covariance matrix inverse
- [x] Doubly robust: consistent if either outcome regression or propensity score ratio is correct
- [x] No-covariates case uses closed-form sample means/covariances (no tuning)
- [x] With covariates: sieve-based propensity ratio estimation with AIC/BIC selection
- [x] Kernel-smoothed conditional covariance estimation
- [x] Analytical SE from EIF sample variance
- [x] Cluster-robust SE option (analytical from EIF + cluster-level multiplier bootstrap)
- [x] Event-study aggregation ES(e) with cohort-size weights
- [x] Hausman-type pre-test for PT-All vs PT-Post (Theorem A.1)
- [x] Each ATT(g,t) can be estimated independently (parallelizable)
- [x] Absorbing treatment validation
- [x] Overlap diagnostics for propensity score ratios
- [x] Survey design support (Phase 3): survey-weighted means/covariances in Omega*, TSL on EIF scores; bootstrap+survey supported (Phase 6)
- **Note:** Sieve ratio estimation uses polynomial basis functions (total degree up to K) with AIC/BIC model selection. The paper describes sieve estimators generally without specifying a particular basis family; polynomial sieves are a standard choice (Section 4, Eq 4.2). Negative sieve ratio predictions are clipped to a small positive value since the population ratio p_g(X)/p_{g'}(X) is non-negative.
- **Note:** Kernel-smoothed conditional covariance Omega*(X) uses Gaussian kernel with Silverman's rule-of-thumb bandwidth by default. The paper specifies kernel smoothing (step 5, Section 4) without mandating a particular kernel or bandwidth selection method.
- **Note:** Conditional covariance Omega*(X) scales each term by per-unit sieve-estimated inverse propensities s_hat_{g'}(X) = 1/p_{g'}(X) (algorithm step 4), matching Eq 3.12. The inverse propensity estimation uses the same polynomial sieve convex minimization as the ratio estimator. Estimated s_hat values are clipped to [1, n] with a UserWarning when clipping binds, mirroring the ratio path's overlap diagnostics.
- **Note:** Outcome regressions m_hat_{g',t,tpre}(X) use linear OLS working models. The paper's Section 4 describes flexible nonparametric nuisance estimation (sieve regression, kernel smoothing, or ML methods). The DR property ensures consistency if either the OLS outcome model or the sieve propensity ratio is correctly specified, but the linear OLS specification does not generically guarantee attainment of the semiparametric efficiency bound unless the conditional mean is linear in the covariates.
- **Note:** EfficientDiD bootstrap with survey weights supported (Phase 6) via PSU-level multiplier weights
- **Note:** EfficientDiD covariates (DR path) with survey weights supported — WLS outcome regression, weighted sieve normal equations for propensity ratios/inverse propensities, survey-weighted Nadaraya-Watson kernel for conditional Omega*(X), and survey-weighted ATT averaging. Silverman bandwidth uses unweighted statistics (survey-weighted bandwidth deferred as second-order refinement).
- **Note:** Cluster-robust SEs use the standard Liang-Zeger clustered sandwich estimator applied to EIF values: aggregate EIF within clusters, center, and compute variance with G/(G-1) small-sample correction. Cluster bootstrap generates multiplier weights at the cluster level (all units in a cluster share the same weight). Analytical clustered SEs are the default when `cluster` is set; cluster bootstrap is opt-in via `n_bootstrap > 0`.
- **Note:** Hausman pretest operates on the post-treatment event-study vector ES(e) per Theorem A.1. Both PT-All and PT-Post fits are aggregated to ES(e) using cohort-size weights before computing the test statistic H = delta' V^{-1} delta where delta = ES_post - ES_all and V = Cov(ES_post) - Cov(ES_all). Covariance is computed from aggregated ES(e)-level EIF values. The variance-difference matrix V is inverted via Moore-Penrose pseudoinverse to handle finite-sample non-positive-definiteness. Effective rank of V (number of positive eigenvalues) is used as degrees of freedom.
- **Note:** Last-cohort-as-control (`control_group="last_cohort"`) reclassifies the latest treatment cohort as pseudo-never-treated and drops time periods at/after that cohort's treatment start. This is distinct from CallawaySantAnna's `not_yet_treated` option which dynamically selects not-yet-treated units per (g,t) pair.

---

## SunAbraham

**Primary source:** [Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. *Journal of Econometrics*, 225(2), 175-199.](https://doi.org/10.1016/j.jeconom.2020.09.006)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires never-treated units as control group
- Warns if treatment effects may be heterogeneous across cohorts (which the method handles)
- Reference period: e=-1-anticipation (defaults to e=-1 when anticipation=0)

*Estimator equation (as implemented):*

Saturated regression with cohort-specific effects:
```
Y_it = α_i + γ_t + Σ_{g∈G} Σ_{e≠-1} δ_{g,e} × 1(G_i=g) × D^e_{it} + ε_it
```
where G_i is unit i's cohort (first treatment period), D^e_{it} = 1(t - G_i = e).

Interaction-weighted estimator:
```
δ̂_e = Σ_g ŵ_{g,e} × δ̂_{g,e}
```
where weights ŵ_{g,e} = n_{g,e} / Σ_g n_{g,e} (sample share of cohort g at event-time e).

*Standard errors:*
- Default: Cluster-robust at unit level
- Delta method for aggregated coefficients
- Optional: Pairs bootstrap for robustness

*Edge cases:*
- Single cohort: reduces to standard event study
- Cohorts with no observations at some event-times: weighted appropriately
- Extrapolation beyond observed event-times: not estimated
- Event-time range: no artificial cap (estimates all available relative times, matching R's `fixest::sunab()`)
- No post-treatment effects: returns `(NaN, NaN)` for overall ATT/SE; all inference fields (t_stat, p_value, conf_int) propagate NaN via `np.isfinite()` guards
- `min_pre_periods`/`min_post_periods` parameters: removed (previously deprecated with `FutureWarning`; callers passing these will now get `TypeError`)
- Variance fallback: when full weight vector cannot be constructed for overall ATT SE, uses simplified variance (ignores covariances between periods) with `UserWarning`
- Rank-deficient design matrix (covariate collinearity):
  - Detection: Pivoted QR decomposition with tolerance `1e-07` (R's `qr()` default)
  - Handling: Warns and drops linearly dependent columns, sets NA for dropped coefficients (R-style, matches `lm()`)
  - Parameter: `rank_deficient_action` controls behavior: "warn" (default), "error", or "silent"
- NaN inference for undefined statistics:
  - t_stat: Uses NaN (not 0.0) when SE is non-finite or zero
  - Analytical inference: p_value and CI also NaN when t_stat is NaN (NaN propagates through `compute_p_value` and `compute_confidence_interval`)
  - Bootstrap inference: p_value and CI computed from bootstrap distribution. SE, CI, and p-value are all NaN if the original point estimate is non-finite, SE is non-finite or zero, or if <50% of bootstrap samples are valid
  - Applies to overall ATT, per-effect event study, and aggregated event study
  - **Note**: Defensive enhancement matching CallawaySantAnna behavior; R's `fixest::sunab()` may produce Inf/NaN without warning
- Inference distribution:
  - Cohort-level p-values: t-distribution (via `LinearRegression.get_inference()`)
  - Aggregated event study and overall ATT p-values: normal distribution (via `compute_p_value()`)
  - This is asymptotically equivalent and standard for delta-method-aggregated quantities
  - **Deviation from R**: R's fixest uses t-distribution at all levels; aggregated p-values may differ slightly for small samples

**Reference implementation(s):**
- R: `fixest::sunab()` (Laurent Bergé's implementation)
- Stata: `eventstudyinteract`

**Requirements checklist:**
- [x] Never-treated units required as controls
- [x] Interaction weights sum to 1 within each relative time period
- [x] Reference period defaults to e=-1, coefficient normalized to zero
- [x] Cohort-specific effects recoverable from results
- [x] Cluster-robust SEs with delta method for aggregates
- [x] R comparison: ATT matches within machine precision (<1e-11)
- [x] R comparison: SE matches within 0.3% (well within 1% threshold)
- [x] R comparison: Event study effects match perfectly (correlation 1.0)
- [x] Survey design support (Phase 3): weighted within-transform, survey weights in LinearRegression with TSL vcov; bootstrap+survey supported (Phase 6) via Rao-Wu rescaled bootstrap. Replicate weights supported via estimator-level refit (see Replicate Weight Variance section); replicate+bootstrap rejected.

---

## ImputationDiD

**Primary source:** [Borusyak, K., Jaravel, X., & Spiess, J. (2024). Revisiting Event-Study Designs: Robust and Efficient Estimation. *Review of Economic Studies*, 91(6), 3253-3285.](https://doi.org/10.1093/restud/rdae007)

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Parallel trends (Assumption 1):** `E[Y_it(0)] = alpha_i + beta_t` for all observations. General form allows `E[Y_it(0)] = alpha_i + beta_t + X'_it * delta` with time-varying covariates.
- **No-anticipation effects (Assumption 2):** `Y_it = Y_it(0)` for all untreated observations. Adjustable via `anticipation` parameter.
- Treatment must be absorbing: `D_it` switches from 0 to 1 and stays at 1.
- Covariate space of treated observations must be spanned by untreated observations (rank condition). For unit/period FE case: every treated unit must have ≥1 untreated period; every post-treatment period must have ≥1 untreated unit.
- Without never-treated units, long-run effects at horizon `K_it >= H_bar` (where `H_bar = max(first_treat) - min(first_treat)`) are not identified (Proposition 5). Set to NaN with warning.

*Estimator equation (Theorem 2, as implemented):*

```
Step 1. Estimate counterfactual model on untreated observations only (it in Omega_0):
    Y_it = alpha_i + beta_t [+ X'_it * delta] + epsilon_it

Step 2. For each treated observation (it in Omega_1), impute:
    Y_hat_it(0) = alpha_hat_i + beta_hat_t [+ X'_it * delta_hat]
    tau_hat_it  = Y_it - Y_hat_it(0)

Step 3. Aggregate:
    tau_hat_w = sum_{it in Omega_1} w_it * tau_hat_it
```

where:
- `Omega_0 = {it : D_it = 0}` — all untreated observations (never-treated + not-yet-treated)
- `Omega_1 = {it : D_it = 1}` — all treated observations
- `w_it` = pre-specified weights (overall ATT: `w_it = 1/N_1`)

*Common estimation targets (weighting schemes):*
- Overall ATT: `w_it = 1/N_1` for all `it in Omega_1`
- Horizon-specific: `w_it = 1[K_it = h] / |Omega_{1,h}|` for `K_it = t - E_i`
- Group-specific: `w_it = 1[G_i = g] / |Omega_{1,g}|`

*Standard errors (Theorem 3, Equation 7):*

Conservative clustered variance estimator:
```
sigma_hat^2_w = sum_i ( sum_{t: it in Omega} v_it * epsilon_tilde_it )^2
```

Observation weights `v_it`:
- For treated `(i,t) in Omega_1`: `v_it = w_it` (the aggregation weight)
- For untreated `(i,t) in Omega_0` (FE-only case): `v_it = -(w_i./n_{0,i} + w_.t/n_{0,t} - w../N_0)`
  where `w_i. = sum of w over treated obs of unit i`, `n_{0,i} = untreated periods for unit i`, etc.
- For untreated with covariates: `v_untreated = -A_0 (A_0' A_0)^{-1} A_1' w_treated`
  where `A_0`, `A_1` are design matrices for untreated/treated observations.

**Note on v_it derivation:** The paper's Supplementary Proposition A3 provides the explicit formula for `v_it^*`, but was not in the extraction range for the paper review. The FE-only closed form above is reconstructed from Theorem 3's general form — it follows from the chain rule of the imputation estimator's dependence on the Step 1 OLS estimates. The covariate case uses the OLS projection matrix directly.

Auxiliary model residuals (Equation 8):
- Partition `Omega_1` into groups `G_g` (default: cohort × horizon)
- Compute `tau_tilde_g` for each group (weighted average within group)
- `epsilon_tilde_it = Y_it - alpha_hat_i - beta_hat_t [- X'delta_hat] - tau_tilde_g` (treated)
- `epsilon_tilde_it = Y_it - alpha_hat_i - beta_hat_t [- X'delta_hat]` (untreated, i.e., Step 1 residuals)

The `aux_partition` parameter controls the partition: `"cohort_horizon"` (default, tightest SEs), `"cohort"` (coarser, more conservative), `"horizon"` (groups by relative time only).

*Pre-trend test (Test 1, Equation 9):*
```
Y_it = alpha_i + beta_t [+ X'_it * delta] + W'_it * gamma + epsilon_it
```
- Estimate on untreated observations only
- Test `gamma = 0` via cluster-robust Wald F-test
- Independent of treatment effect estimation (Proposition 9)

*Pre-period event study coefficients (`pretrends=True`, Test 1 / Equation 9):*

Pre-period coefficients reuse the existing pre-trend test machinery (BJS Equation 9):
```
Y_it = alpha_i + beta_t [+ X'_it * delta] + sum_h gamma_h * W_it(h) + epsilon_it
```
where `W_it(h) = 1[K_it = h]` are lead indicators, estimated on `Omega_0` only.
- `gamma_h` are the pre-period event study coefficients (cluster-robust SEs by default; design-based survey VCV when analytical `survey_design` is present)
- Under parallel trends (Assumption 1), `gamma_h = 0` for all `h < -anticipation`
- Reference period `h = -1 - anticipation` is the omitted category (normalized to zero)
- SEs from cluster-robust Wald variance by default; design-based when survey present (consistent with `pretrend_test()`)
- Bootstrap does not update pre-period SEs (they are from the lead regression)
- When `balance_e` is set, lead indicators are restricted to balanced cohorts; the full Omega_0 sample (including never-treated) is kept for within-transformation
- Only affects event study aggregation; overall ATT and group aggregation unchanged
- **Note:** `pretrends=True` with analytical `survey_design` (strata/PSU/FPC) is supported. The lead regression uses survey-weighted demeaning, WLS point estimates, and `compute_survey_vcov()` for design-based VCV. The full survey design is preserved (subpopulation approach): Omega_0 scores are zero-padded back to full-panel length so PSU/strata structure is maintained for variance estimation. The F-test in `pretrend_test()` uses the full-design `df_survey` as denominator df. Replicate-weight survey designs raise `NotImplementedError` with `pretrends=True` because per-replicate lead regression refits are not yet implemented.

*Edge cases:*
- **Unbalanced panels:** FE estimated via iterative alternating projection (Gauss-Seidel), equivalent to OLS with unit+time dummies. Converges in O(max_iter) passes; typically 5-20 iterations for unbalanced panels, 1-2 for balanced. One-pass demeaning is only exact for balanced panels.
- **No never-treated units (Proposition 5):** Long-run effects at horizons `h >= H_bar` are not identified. Set to NaN with warning listing affected horizons.
- **Rank condition failure:** Every treated unit must have ≥1 untreated period; every post-treatment period must have ≥1 untreated unit. Behavior controlled by `rank_deficient_action`: "warn" (default), "error", or "silent". Missing FE produce NaN treatment effects for affected observations.
- **Always-treated units:** Units with `first_treat` at or before the earliest time period have no untreated observations. Warning emitted; these units are excluded from Step 1 OLS but their treated observations contribute to aggregation if imputation is possible.
- **NaN propagation:** If all `tau_hat` values for a given horizon or group are NaN, the aggregated effect and all inference fields (SE, t-stat, p-value, CI) are set to NaN. NaN in v*eps product (from missing FE) is zeroed for variance computation (matching R's did_imputation which drops unimputable obs).
- **NaN inference for undefined statistics:** t_stat uses NaN when SE is non-finite or zero; p_value and CI also NaN. Matches CallawaySantAnna NaN convention.
- **Pre-trend test:** Uses iterative demeaning (same as Step 1 FE) for exact within-transformation on unbalanced panels. One-pass demeaning is only exact for balanced panels.
- **Overall ATT variance:** Weights zero out non-finite tau_hat and renormalize, matching the ATT estimand (which averages only finite tau_hat). `_compute_conservative_variance` returns 0.0 for all-zeros weights, so the n_valid==0 guard is necessary to return NaN SE.
- **`balance_e` cohort filtering:** When `balance_e` is set, cohort balance is checked against the *full panel* (pre + post treatment) via `_build_cohort_rel_times()`, requiring observations at every relative time in `[-balance_e, max_h]`. Both analytical aggregation and bootstrap inference use the same `_compute_balanced_cohort_mask` with pre-computed cohort horizons.
- **Bootstrap clustering:** Multiplier bootstrap generates weights at `cluster_var` granularity (defaults to `unit` if `cluster` not specified). Invalid cluster column raises ValueError.
- **Non-constant `first_treat` within a unit:** Emits `UserWarning` identifying the count and example unit. The estimator proceeds using the first observed value per unit (via `.first()` aggregation), but results may be unreliable.
- **treatment_effects DataFrame weights:** `weight` column uses `1/n_valid` for finite tau_hat and 0 for NaN tau_hat, consistent with the ATT estimand (unweighted), or normalized survey weights `sw_i/sum(sw)` when `survey_design` is active.
- **Rank-deficient covariates in variance:** Covariates with NaN coefficients (dropped for rank deficiency in Step 1) are excluded from the variance design matrices `A_0`/`A_1`. Only covariates with finite coefficients participate in the `v_it` projection.
- **Sparse variance solver:** `_compute_v_untreated_with_covariates` uses `scipy.sparse.linalg.spsolve` to solve `(A_0'A_0) z = A_1'w` without densifying the normal equations matrix. Falls back to dense `lstsq` if the sparse solver fails.
- **Note:** Survey weights enter ImputationDiD via weighted iterative FE (Step 1), survey-weighted ATT aggregation (Step 3), and design-based variance via `compute_survey_if_variance()`. PSU clustering, stratification, and FPC are fully supported in the Theorem 3 variance path. When `resolved_survey` is present, the observation-level influence function (`v_it * epsilon_tilde_it`) is passed to `compute_survey_if_variance()` which applies the stratified PSU-level sandwich with FPC correction. Strata also enters survey df (n_PSU - n_strata) for t-distribution inference. Bootstrap + survey supported (Phase 6) via PSU-level multiplier weights.
- **Bootstrap inference:** Uses multiplier bootstrap on the Theorem 3 influence function: `psi_i = sum_t v_it * epsilon_tilde_it`. Cluster-level psi sums are pre-computed for each aggregation target (overall, per-horizon, per-group), then perturbed with multiplier weights (Rademacher by default; configurable via `bootstrap_weights` parameter to use Mammen or Webb weights, matching CallawaySantAnna). This is a library extension (not in the paper) consistent with CallawaySantAnna/SunAbraham bootstrap patterns.
- **Auxiliary residuals (Equation 8):** Uses v_it-weighted tau_tilde_g formula: `tau_tilde_g = sum(v_it * tau_hat_it) / sum(v_it)` within each partition group. Zero-weight groups (common in event-study SE computation) fall back to unweighted mean.

**Reference implementation(s):**
- Stata: `did_imputation` (Borusyak, Jaravel, Spiess; available from SSC)
- R: `didimputation` package (Kyle Butts)

**Requirements checklist:**
- [x] Step 1: OLS on untreated observations only (never-treated + not-yet-treated)
- [x] Step 2: Impute counterfactual `Y_hat_it(0)` for treated observations
- [x] Step 3: Aggregate with researcher-chosen weights `w_it`
- [x] Conservative clustered variance estimator (Theorem 3, Equation 7)
- [x] Auxiliary model for treated residuals (Equation 8) with configurable partition (`aux_partition`)
- [x] Supports unit FE, period FE, and time-varying covariates
- [x] Refuses to estimate unidentified estimands (Proposition 5) — sets NaN with warning
- [x] Pre-trend test uses only untreated observations (Test 1, Equation 9)
- [x] Supports balanced and unbalanced panels (iterative Gauss-Seidel demeaning for exact FE)
- [x] Event study and group aggregation

---

## TwoStageDiD

**Primary source:** [Gardner, J. (2022). Two-stage differences in differences. arXiv:2207.05943.](https://arxiv.org/abs/2207.05943)

**Key implementation requirements:**

*Assumption checks / warnings:*
- **Parallel trends (same as ImputationDiD):** `E[Y_it(0)] = alpha_i + beta_t` for all observations.
- **No-anticipation effects:** `Y_it = Y_it(0)` for all untreated observations.
- Treatment must be absorbing: `D_it` switches from 0 to 1 and stays at 1.
- Always-treated units (treated in all periods) are excluded with a warning, since they have no untreated observations for Stage 1 FE estimation.

*Estimator equation (two-stage procedure, as implemented):*

```
Stage 1. Estimate unit + time fixed effects on untreated observations only (it in Omega_0):
    Y_it = alpha_i + beta_t + epsilon_it
    Compute residuals: y_tilde_it = Y_it - alpha_hat_i - beta_hat_t  (for ALL observations)

Stage 2. Regress residualized outcomes on treatment indicators (on treated observations):
    y_tilde_it = tau * D_it + eta_it
    (or event-study specification with horizon indicators)
```

Point estimates are identical to ImputationDiD (Borusyak et al. 2024). The two-stage procedure is algebraically equivalent to the imputation approach: both estimate unit+time FE on untreated observations and recover treatment effects from the difference between observed and counterfactual outcomes.

*Variance: GMM sandwich (Newey & McFadden 1994 Theorem 6.1):*

The variance accounts for first-stage estimation error propagating into Stage 2, following the GMM framework:

```
V(tau_hat) = (D'D)^{-1} * Bread * (D'D)^{-1}

Bread = sum_c ( sum_{i in c} psi_i )( sum_{i in c} psi_i )'
```

where `psi_i` is the stacked influence function for unit i across all its observations, combining the Stage 2 score and the Stage 1 correction term.

**Note on Equation 6 discrepancy:** The paper's Equation 6 uses a per-cluster inverse `(D_c'D_c)^{-1}` when forming the influence function contribution. The R `did2s` implementation and our code use the GLOBAL inverse `(D'D)^{-1}` following standard GMM theory (Newey & McFadden 1994). We follow the R implementation, which is consistent with standard GMM sandwich variance estimation.

**No finite-sample adjustments:** The variance estimator uses the raw asymptotic sandwich without degrees-of-freedom corrections (no HC1-style `n/(n-k)` adjustment). This matches the R `did2s` implementation.

*Bootstrap:*

Our implementation uses multiplier bootstrap on the GMM influence function: cluster-level `psi` sums are pre-computed, then perturbed with multiplier weights (Rademacher by default; configurable via `bootstrap_weights` parameter to use Mammen or Webb weights, matching CallawaySantAnna). The R `did2s` package defaults to block bootstrap (resampling clusters with replacement). Both approaches are asymptotically valid; the multiplier bootstrap is computationally cheaper and consistent with the CallawaySantAnna/ImputationDiD bootstrap patterns in this library.

*Edge cases:*
- **Always-treated units:** Units treated in all observed periods have no untreated observations for Stage 1 FE estimation. These are excluded with a warning listing the affected unit IDs. Their treated observations do NOT contribute to Stage 2.
- **Rank condition violations:** If the Stage 1 design matrix (unit+time dummies on untreated obs) is rank-deficient, or if certain unit/time FE are unidentified (e.g., a unit with no untreated periods after excluding always-treated), the affected FE produce NaN. Behavior controlled by `rank_deficient_action`: "warn" (default), "error", or "silent".
- **NaN y_tilde handling:** When Stage 1 FE are unidentified for some observations, the residualized outcome `y_tilde` is NaN. These observations are zeroed out (excluded) from the Stage 2 regression and variance computation, matching the treatment of unimputable observations in ImputationDiD.
- **NaN inference for undefined statistics:** t_stat uses NaN when SE is non-finite or zero; p_value and CI also NaN. Matches CallawaySantAnna/ImputationDiD NaN convention.
- **Event study aggregation:** Horizon-specific effects use the same two-stage procedure with horizon indicator dummies in Stage 2. Unidentified horizons (e.g., long-run effects without never-treated units, per Proposition 5 of Borusyak et al. 2024) produce NaN.
- **Pre-period event study coefficients (`pretrends=True`):** When enabled, the Stage 2 design matrix `X_2` includes pre-period relative-time dummies. Pre-period observations have `y_tilde = Step 1 residual` by construction. The GMM sandwich variance accounts for Stage 1 estimation error (Gardner 2022, Theorem 1). Only affects event study aggregation; overall ATT unchanged.
- **balance_e with no qualifying cohorts:** If no cohorts have sufficient pre/post coverage for the requested `balance_e`, a warning is emitted and event study results contain only the reference period.
- **No never-treated units (Proposition 5):** When there are no never-treated units and multiple treatment cohorts, horizons h >= h_bar (where h_bar = max(groups) - min(groups)) are unidentified per Proposition 5 of Borusyak et al. (2024). These produce NaN inference with n_obs > 0 (treated observations exist but counterfactual is unidentified) and a warning listing affected horizons. Matches ImputationDiD behavior. Proposition 5 applies to event study horizons only, not cohort aggregation — a cohort whose treated obs all fall at Prop 5 horizons naturally gets n_obs=0 in group effects because all its y_tilde values are NaN.
- **Zero-observation horizons after filtering:** When `balance_e` or NaN `y_tilde` filtering results in zero observations for some non-Prop-5 event study horizons, those horizons produce NaN for all inference fields (effect, SE, t-stat, p-value, CI) with n_obs=0.
- **Zero-observation cohorts in group effects:** If all treated observations for a cohort have NaN `y_tilde` (excluded from estimation), that cohort's group effect is NaN with n_obs=0.
- **Note:** Survey weights in TwoStageDiD GMM sandwich via weighted cross-products: bread uses (X'_2 W X_2)^{-1}, gamma_hat uses (X'_{10} W X_{10})^{-1}(X'_1 W X_2), per-cluster scores multiply by survey weights. PSU clustering, stratification, and FPC are fully supported in the meat matrix via `_compute_stratified_meat_from_psu_scores()`. When strata or FPC are present, the meat computation replaces `S' S` with the stratified formula `sum_h (1 - f_h) * (n_h/(n_h-1)) * centered_h' centered_h`. Strata also enters survey df (n_PSU - n_strata) for t-distribution inference. Bootstrap + survey supported (Phase 6) via PSU-level multiplier weights.

**Reference implementation(s):**
- R: `did2s::did2s()` (Kyle Butts & John Gardner)

**Requirements checklist:**
- [x] Stage 1: OLS on untreated observations only for unit+time FE
- [x] Stage 2: Regress residualized outcomes on treatment indicators
- [x] Point estimates match ImputationDiD
- [x] GMM sandwich variance (Newey & McFadden 1994 Theorem 6.1)
- [x] Global `(D'D)^{-1}` in variance (matches R `did2s`, not paper Eq. 6)
- [x] No finite-sample adjustment (raw asymptotic sandwich)
- [x] Always-treated units excluded with warning
- [x] Multiplier bootstrap on GMM influence function
- [x] Event study and overall ATT aggregation

---

## StackedDiD

**Primary source:** Wing, C., Freedman, S. M., & Hollingsworth, A. (2024). Stacked Difference-in-Differences. NBER Working Paper 32054. http://www.nber.org/papers/w32054

**Key implementation requirements:**

*Assumption checks / warnings:*
- Assumption 1 (No Anticipation): ATT(a, a+e) = 0 for all e < 0
- Assumption 2 (Common Trends): E[Y_{s,a+e}(0) - Y_{s,a-1}(0) | A_s = a] = E[Y_{s,a+e}(0) - Y_{s,a-1}(0) | A_s > a + e]
- Clean controls must exist for each sub-experiment (IC2)
- Event window must fit within observed data range (IC1)

*Target parameter (Equation 2):*

    theta_kappa^e = sum_{a in Omega_kappa} ATT(a, a+e) * (N_a^D / N_Omega_kappa^D)

where:
- `theta_kappa^e` = trimmed aggregate ATT at event time e
- `Omega_kappa` = trimmed set of adoption events satisfying IC1 and IC2
- `N_a^D` = number of treated units in sub-experiment a
- `N_Omega_kappa^D` = total treated units across all sub-experiments in trimmed set

*Estimator equation (Equation 3 — weighted saturated event study, recommended):*

    Y_sae = alpha_0 + alpha_1 * D_sa + sum_{h != -1} [lambda_h * 1(e=h) + delta_h * D_sa * 1(e=h)] + U_sae

Estimated via WLS with Q-weights. The delta_h coefficients identify theta_kappa^e.

*Q-weights (Section 5.3, Table 1):*

    Q_sa = 1                                           if D_sa = 1 (treated)
    Q_sa = (N_a^D / N^D) / (N_a^C / N^C)             if D_sa = 0 (control, aggregate weighting)
    Q_sa = (Pop_a^D / Pop^D) / (N_a^C / N^C)         if D_sa = 0 (control, population weighting)
    Q_sa = ((N_a + N_a^C)/(N^D+N^C)) / (N_a^C/N^C)  if D_sa = 0 (control, sample share weighting)

*Standard errors (Section 5.4):*
- Default: Cluster-robust standard errors at the group (unit) level
- Alternative: Cluster at group x sub-experiment level
- Both approaches yield approximately correct coverage when clusters > 100 (Table 2)
- No special bootstrap procedure specified; standard cluster-robust SEs recommended
- For post-period average: delta method or `lincom`/`marginaleffects`

*Edge cases:*
- All events trimmed: `len(Omega_kappa) == 0` -> ValueError suggesting reduced kappa
- No clean controls for event a: IC2 check fails -> Trim event, warn user
- Single cohort in trimmed set: Valid — Q-weights simplify
- Duplicate observations: Same (unit, time) appears in multiple sub-experiments -> handled by clustering at unit level
- Constant treatment share across sub-exps: Unweighted FE recovers correct estimand (special case, Section 5.5)
- Anticipation > 0: Reference period shifts to e = -1 - anticipation. Post-treatment includes anticipation periods (e >= -anticipation). Window expands by anticipation pre-periods.
- Group aggregation: Not supported — pooled stacked regression cannot produce cohort-specific effects. Use CallawaySantAnna or ImputationDiD.

*Algorithm (Section 5):*
1. Choose kappa_pre, kappa_post event window
2. Apply IC1 (window fits in data) and IC2 (clean controls exist) to get Omega_kappa
3. For each a in Omega_kappa: build sub-experiment with treated (A_s = a), clean controls (A_s > a + kappa_post), time window [a - kappa_pre, a + kappa_post] (with anticipation: [a - kappa_pre - anticipation, a + kappa_post])
4. Stack all sub-experiments vertically
5. Compute Q-weights: aggregate weighting uses observation counts per (event_time, sub_exp), matching R reference. Population/sample_share use unit counts per sub_exp (paper notation).
6. Run WLS regression of Equation 3 with Q-weights
7. Extract delta_h coefficients as event-study ATTs
8. Compute cluster-robust SEs at unit level

*IC1 (Adoption Event Window, Section 3):*

    IC1_a = 1[a - kappa_pre >= T_min  AND  a + kappa_post <= T_max]

Note: Matches R reference implementation (`focalAdoptionTime - kappa_pre >= minTime`).
The reference period a-1 is included in the window [a-kappa_pre, a+kappa_post] when kappa_pre >= 1.
The paper text states a stricter bound (T_min + 1) but the R code by the co-author uses T_min.

*IC2 (Clean Controls Exist, Section 3):*

    IC2_a = 1[exists s with A_s > a + kappa_post]    (not_yet_treated)
    IC2_a = 1[exists s with A_s > a + kappa_post + kappa_pre]  (strict)
    IC2_a = 1[exists s with A_s = infinity]           (never_treated)

**Reference implementation(s):**
- R: https://github.com/hollina/stacked-did-weights (`create_sub_exp()`, `compute_weights()`)
- No Stata or Python package; Stata estimation via standard `reghdfe` with Q-weight column

**Requirements checklist:**
- [x] Sub-experiment construction with treated + clean controls + time window
- [x] IC1 and IC2 trimming with warnings
- [x] Q-weight computation for all three weighting schemes (Table 1)
- [x] WLS via sqrt(w) transformation
- [x] Event study regression (Equation 3) with reference period e=-1
- [x] Cluster-robust SEs at unit or unit x sub-exp level
- [x] Overall ATT as average of post-treatment delta_h with delta-method SE
- [x] Anticipation parameter support
- [x] Never-treated encoding (0 and inf)
- [x] Survey design support (Phase 3): Q-weights compose multiplicatively with survey weights; TSL vcov on composed weights; survey design columns propagated through sub-experiments. Replicate weights supported via estimator-level refit with Q-weight composition (see Replicate Weight Variance section).
- **Note:** Survey weights compose multiplicatively with Q-weights for StackedDiD; only `weight_type="pweight"` (default) is supported — `fweight` and `aweight` are rejected because Q-weight composition changes weight semantics (non-integer for fweight, non-inverse-variance for aweight)

---

## WooldridgeDiD (ETWFE)

**Primary source:** Wooldridge, J. M. (2025). Two-way fixed effects, the two-way Mundlak regression, and difference-in-differences estimators. *Empirical Economics*, 69(5), 2545–2587. (Published version of the 2021 SSRN working paper NBER WP 29154.)

**Secondary source:** Wooldridge, J. M. (2023). Simple approaches to nonlinear difference-in-differences with panel data. *The Econometrics Journal*, 26(3), C31–C66. https://doi.org/10.1093/ectj/utad016

**Application reference:** Nagengast, A. J., Rios-Avila, F., & Yotov, Y. V. (2026). The European single market and intra-EU trade: an assessment with heterogeneity-robust difference-in-differences methods. *Economica*, 93(369), 298–331.

**Reference implementation:** Stata: `jwdid` package (Rios-Avila, 2021). R: `etwfe` package (McDermott, 2023).

**Key implementation requirements:**

*Core estimand:*

    ATT(g, t) = E[Y_it(g) - Y_it(0) | G_i = g, T = t]    for t >= g

where `g` is cohort (first treatment period), `t` is calendar time.

*OLS design matrix (Wooldridge 2025, Section 5):*

The saturated ETWFE regression includes:
1. Unit fixed effects (absorbed via within-transformation or as dummies)
2. Time fixed effects (absorbed or as dummies)
3. Cohort×time treatment interactions: `I(G_i = g) * I(T = t)` for each post-treatment (g, t) cell
4. Additional covariates X_it interacted with cohort×time indicators (optional)

The interaction coefficient `δ_{g,t}` identifies `ATT(g, t)` under parallel trends.
- **Note:** OLS path uses iterative alternating-projection within-transformation (uniform weights) for exact FE absorption on both balanced and unbalanced panels. One-pass demeaning (`y - ȳ_i - ȳ_t + ȳ`) is only exact for balanced panels.

*Nonlinear extensions (Wooldridge 2023):*

For binary outcomes (logit) and count outcomes (Poisson), Wooldridge (2023) provides an
Average Structural Function (ASF) approach. For each treated cell (g, t):

    ATT(g, t) = mean_i[g(η_i + δ_{g,t}) - g(η_i)]   over units i in cell (g, t)

where `g(·)` is the link inverse (logistic or exp), `η_i` is the individual linear predictor
(fixed effects + controls), and `δ_{g,t}` is the interaction coefficient from the nonlinear model.

*Standard errors:*
- OLS: Cluster-robust sandwich estimator at the unit level (default)
- Logit/Poisson: QMLE sandwich `(X'WX)^{-1} meat (X'WX)^{-1}` via `compute_robust_vcov(..., weights=w, weight_type="aweight")` where `w = p_i(1-p_i)` for logit or `w = μ_i` for Poisson
- Delta-method SEs for ATT(g,t) from nonlinear models: `Var(ATT) = ∇θ' Σ_β ∇θ`
- Joint delta method for overall ATT: `agg_grad = Σ_k (w_k/w_total) * ∇θ_k`
- **Deviation from R:** R's `etwfe` package uses `fixest` for nonlinear paths; this implementation uses direct QMLE via `compute_robust_vcov` to avoid a statsmodels/fixest dependency.
- **Note:** QMLE sandwich uses `weight_type="aweight"` which applies `(G/(G-1)) * ((n-1)/(n-k))` small-sample adjustment. Stata `jwdid` uses `G/(G-1)` only. The `(n-1)/(n-k)` term is conservative (inflates SEs slightly). For typical ETWFE panels where n >> k, the difference is negligible.

*Aggregations (matching `jwdid_estat`):*
- `simple`: Weighted average across all post-treatment (g, t) cells with weights `n_{g,t}`:

      ATT_overall = Σ_{(g,t): t≥g} n_{g,t} · ATT(g,t) / Σ_{(g,t): t≥g} n_{g,t}

  Cell weight `n_{g,t}` = count of obs in cohort g at time t in estimation sample.
  - **Note:** Cell-level weighting (n_{g,t} observation counts) matches Stata `jwdid_estat` behavior. Differs from W2025 Eqs. 7.2-7.4 cohort-share weights that account for the number of post-treatment periods per cohort.

- `group`: Weighted average across t for each cohort g
- `calendar`: Weighted average across g for each calendar time t
- `event`: Weighted average across (g, t) cells by relative period k = t - g

*Covariates:*
- `exovar`: Time-invariant covariates, added without demeaning (corresponds to W2025 Eq. 5.2 `x_i`)
- `xtvar`: Time-varying covariates, demeaned within cohort×period cells when `demean_covariates=True` (corresponds to W2025 Eq. 10.2 `x_hat_itgs = x_it - x_bar_gs`)
- `xgvar`: Covariates interacted with each cohort indicator
- **Note:** Covariate-adjusted ETWFE includes the full W2025 Eq. 5.3 basis: raw X, cohort × X (D_g × X for treated cohorts, auto-generated for `exovar`/`xtvar`), time × X (f_t × X, drop first period), and cell × demeaned X (D_{g,t} × X̃). Variables in `xgvar` already contribute D_g × X via `_prepare_covariates`; `exovar`/`xtvar` get automatic D_g × X generation.
- **Note:** `xtvar` demeaning operates at the cohort×period level (W2025 Eq. 10.2), not the cohort level (W2025 Eq. 5.2). These are identical for time-constant covariates but differ for time-varying covariates.

*Control groups:*
- `not_yet_treated` (default): Control pool includes units not yet treated at time t (same as Callaway-Sant'Anna)
- `never_treated`: Control pool restricted to never-treated units only

*Edge cases:*
- Single cohort (no staggered adoption): Reduces to standard 2×2 DiD
- Missing cohorts: Only cohorts observed in the data are included in interactions
- Anticipation: When `anticipation > 0`, interactions include periods `t >= g - anticipation`
- **Note:** Aggregation (simple/group/calendar) uses `t >= g` as the post-treatment threshold regardless of `anticipation`. Anticipation-window cells (g - anticipation <= t < g) are estimated but treated as pre-treatment placebos in aggregation, not included in overall ATT. This matches the standard post-treatment ATT definition; users who want anticipation cells in the aggregate should compute custom weighted averages from `group_time_effects`.
- Never-treated control only: Pre-treatment periods still estimable as placebo ATTs
- **Note:** Poisson QMLE with cohort+time dummies (not unit dummies) is consistent even in short panels (Wooldridge 1999, JBES). The exponential mean function is unique in that incidental parameters from group dummies do not cause inconsistency.
- **Note:** Logit path uses cohort×time additive dummies (not unit dummies) to avoid incidental parameters bias — a standard limitation of logit FE in short panels. This matches Stata `jwdid method(logit)` which uses `i.gvar i.tvar`.
- **Note:** Nonlinear methods (logit, Poisson) with `control_group="never_treated"` restrict the interaction matrix to post-treatment cells only. Pre-treatment placebo cells are OLS-only (where within-transformation absorbs FE). Including all (g,t) cells in the nonlinear design creates exact collinearity between cohort dummies and cell indicator sums, leading to a data-dependent normalization via QR dropping.

*Algorithm:*
1. Identify cohorts G and time periods T from data
2. Build within-transformed design matrix (absorb unit + time FE)
3. Append cohort×time interaction columns for all post-treatment cells
4. Fit OLS/logit/Poisson
5. For nonlinear: compute ASF-based ATT(g,t) and delta-method SEs per cell
6. For OLS: extract δ_{g,t} coefficients directly as ATT(g,t)
7. Compute overall ATT as weighted average; store full vcov for aggregate SEs
8. Optionally run multiplier bootstrap for overall SE

**Requirements checklist:**
- [x] Saturated cohort×time interaction design matrix
- [x] Unit + time FE absorption (within-transformation)
- [x] OLS, logit (IRLS), and Poisson (IRLS) fitting methods
- [x] Cluster-robust SEs at unit level for all methods
- [x] ASF-based ATT for nonlinear methods with delta-method SEs
- [x] Joint delta-method SE for aggregate ATT in nonlinear models
- [x] Four aggregation types: simple, group, calendar, event
- [x] Both control groups: not_yet_treated, never_treated
- [x] Anticipation parameter support
- [x] Multiplier bootstrap (Rademacher/Webb/Mammen) for OLS overall SE
- [x] Survey design support (strata/PSU/FPC with TSL variance)

**Survey design notes:**
- **OLS path:** Survey-weighted within-transformation + WLS via `solve_ols(weights=...)` + TSL vcov via `compute_survey_vcov()`.
- **Logit/Poisson paths:** Survey-weighted IRLS via `solve_logit(weights=...)`/`solve_poisson(weights=...)` + X_tilde linearization trick for TSL vcov: `X_tilde = X * sqrt(V)`, `r_tilde = (y - mu) / sqrt(V)`, then `compute_survey_vcov(X_tilde, r_tilde, resolved)` gives correct QMLE sandwich. ASF means and gradients use survey-weighted averaging.
- **Note:** Only `pweight` (probability weights) are supported; `fweight`/`aweight` raise `ValueError` because the composed survey/QMLE weighting changes their semantics.
- **Note:** Replicate-weight variance is not yet supported (`NotImplementedError`). Use TSL (strata/PSU/FPC) instead.
- **Note:** Bootstrap inference (`n_bootstrap > 0`) cannot be combined with `survey_design` — no survey-aware bootstrap variant is implemented.

---

# Advanced Estimators

## SyntheticDiD

**Primary source:** [Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W., & Wager, S. (2021). Synthetic Difference-in-Differences. *American Economic Review*, 111(12), 4088-4118.](https://doi.org/10.1257/aer.20190159)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires balanced panel (same units observed in all periods)
- Warns if pre-treatment fit is poor (high RMSE)
- Treatment must be "block" structure: all treated units treated at same time

*Estimator equation (as implemented):*

```
τ̂^sdid = Σ_t λ_t (Ȳ_{tr,t} - Σ_j ω_j Y_{j,t})
```

where Ȳ_{tr,t} is the mean treated outcome at time t, ω_j are unit weights, and λ_t are time weights.

*Unit weights ω (Frank-Wolfe on collapsed form):*

Build collapsed-form matrix Y_unit of shape (T_pre, N_co + 1), where the last column is the mean treated pre-period outcomes. Solve via Frank-Wolfe on the simplex:

```
min_{ω on simplex}  ζ_ω² ||ω||₂² + (1/T_pre) ||A_centered ω - b_centered||₂²
```

where A = Y_unit[:, :N_co], b = Y_unit[:, N_co], and centering is column-wise (intercept=True).

**Two-pass sparsification procedure** (matches R's `synthdid::sc.weight.fw` + `sparsify_function`):
1. First pass: Run Frank-Wolfe for 100 iterations (max_iter_pre_sparsify) from uniform initialization
2. Sparsify: `v[v <= max(v)/4] = 0; v = v / sum(v)` (zero out small weights, renormalize)
3. Second pass: Run Frank-Wolfe for 10000 iterations (max_iter) starting from sparsified weights

The sparsification step concentrates weights on the most important control units, improving interpretability and stability.

*Time weights λ (Frank-Wolfe on collapsed form):*

Build collapsed-form matrix Y_time of shape (N_co, T_pre + 1), where the last column is the per-control post-period mean (averaged across post-periods for each control unit). Solve:

```
min_{λ on simplex}  ζ_λ² ||λ||₂² + (1/N_co) ||A_centered λ - b_centered||₂²
```

where A = Y_time[:, :T_pre], b = Y_time[:, T_pre], and centering is column-wise.

*Auto-regularization (matching R's synthdid):*
```
noise_level = sd(first_differences of control outcomes)   # pooled across units
zeta_omega  = (N_treated × T_post)^(1/4) × noise_level
zeta_lambda = 1e-6 × noise_level
```

The noise level is computed as the standard deviation (ddof=1) of `np.diff(Y_pre_control, axis=0)`,
which computes first-differences across time for each control unit and pools all values.
This matches R's `sd(apply(Y[1:N0, 1:T0], 1, diff))`.

*Frank-Wolfe step (matches R's `fw.step()`):*
```
half_grad = A' (A x - b) + η x        # η = N × ζ²
i = argmin(half_grad)                   # vertex selection (simplex corner)
d_x = e_i - x                          # direction toward vertex i
step = -(half_grad · d_x) / (||A d_x||² + η ||d_x||²)
step = clip(step, 0, 1)
x_new = x + step × d_x
```

Convergence criterion: stop when objective decrease < min_decrease² (default min_decrease = 1e-5 × noise_level, max_iter = 10000, max_iter_pre_sparsify = 100).

*Standard errors:*

- Default: Placebo variance estimator (Algorithm 4 in paper)
  1. Randomly permute control unit indices
  2. Split into pseudo-controls (first N_co - N_tr) and pseudo-treated (last N_tr)
  3. Re-estimate unit weights (Frank-Wolfe) on pseudo-control/pseudo-treated data
  4. Re-estimate time weights (Frank-Wolfe) on pseudo-control data
  5. Compute SDID estimate with re-estimated weights
  6. Repeat `replications` times (default 200)
  7. `SE = sqrt((r-1)/r) × sd(placebo_estimates)` where r = number of successful replications

  This matches R's `synthdid::vcov(method="placebo")` which passes `update.omega=TRUE, update.lambda=TRUE` via `opts`.

- Alternative: Bootstrap at unit level (matching R's `synthdid::vcov(method="bootstrap")`)
  1. Resample ALL units (control + treated) with replacement
  2. Identify which resampled units are control vs treated
  3. Renormalize original unit weights for resampled controls: `ω_boot = _sum_normalize(ω[boot_control_idx])`
  4. Time weights λ remain **unchanged** from original estimation (fixed weights)
  5. Compute SDID estimate with renormalized ω and original λ
  6. `SE = sd(bootstrap_estimates, ddof=1)`

*Edge cases:*
- **Frank-Wolfe non-convergence**: Returns current weights after max_iter iterations. No warning emitted; the convergence check `vals[t-1] - vals[t] < min_decrease²` simply does not trigger early exit, and the final iterate is returned.
- **`_sparsify` all-zero input**: If `max(v) <= 0`, returns uniform weights `ones(len(v)) / len(v)`.
- **Single control unit**: `compute_sdid_unit_weights` returns `[1.0]` immediately (short-circuit before Frank-Wolfe).
- **Zero control units**: `compute_sdid_unit_weights` returns empty array `[]`.
- **Single pre-period**: `compute_time_weights` returns `[1.0]` when `n_pre <= 1` (Frank-Wolfe on a 1-element simplex is trivial).
- **Bootstrap with 0 control or 0 treated in resample**: Skip iteration (`continue`). If ALL bootstrap iterations fail, raises `ValueError`. If only 1 succeeds, warns and returns SE=0.0. If >5% failure rate, warns about reliability.
- **Placebo with n_control <= n_treated**: Warns that not enough control units for placebo variance estimation, returns SE=0.0 and empty placebo effects array. The check is `n_control - n_treated < 1`.
- **Note:** Power analysis functions (`simulate_power`, `simulate_mde`, `simulate_sample_size`) raise `ValueError` for placebo variance when `n_control <= n_treated`. The registry path checks pre-generation using `n_units * treatment_fraction`; the custom-DGP path checks post-generation on the realized data (first iteration only, since treatment allocation is deterministic per `n_units`/`treatment_fraction`).
- **Negative weights attempted**: Frank-Wolfe operates on the simplex (non-negative, sum-to-1), so weights are always feasible by construction. The step size is clipped to [0, 1] and the move is toward a simplex vertex.
- **Perfect pre-treatment fit**: Regularization (ζ² ||ω||²) prevents overfitting by penalizing weight concentration.
- **Single treated unit**: Valid; placebo variance uses jackknife-style permutations of controls.
- **Noise level with < 2 pre-periods**: Returns 0.0, which makes both zeta_omega and zeta_lambda equal to 0.0 (no regularization). **Note (deviation from R):** `min_decrease` uses a `1e-5` floor when `noise_level == 0` to enable Frank-Wolfe early stopping. R would use `0.0`, causing FW to run all `max_iter` iterations; the result is equivalent since zero-noise data has no variation to optimize.
- **NaN inference for undefined statistics**: t_stat uses NaN when SE is zero or non-finite; p_value and CI also NaN. Matches CallawaySantAnna NaN convention.
- **Placebo p-value floor**: `p_value = max(empirical_p, 1/(n_replications + 1))` to avoid reporting exactly zero.
- **Varying treatment within unit**: Raises `ValueError`. SDID requires block treatment (constant within each unit). Suggests CallawaySantAnna or ImputationDiD for staggered adoption.
- **Unbalanced panel**: Raises `ValueError`. SDID requires all units observed in all periods. Suggests `balance_panel()`.
- **Poor pre-treatment fit**: Warns (`UserWarning`) when `pre_fit_rmse > std(treated_pre_outcomes, ddof=1)`. Diagnostic only; estimation proceeds.
- **Note:** Survey support: weights, strata, PSU, and FPC are all supported. Full-design surveys use Rao-Wu rescaled bootstrap (Phase 6); `variance_method="placebo"` requires weights-only (strata/PSU/FPC require bootstrap). Both sides weighted per WLS regression interpretation: treated-side means are survey-weighted (Frank-Wolfe target and ATT formula); control-side synthetic weights are composed with survey weights post-optimization (ω_eff = ω * w_co, renormalized). Frank-Wolfe optimization itself is unweighted — survey importance enters after trajectory-matching. Covariate residualization uses WLS with survey weights. Placebo and bootstrap SE preserve survey weights on both sides.

**Reference implementation(s):**
- R: `synthdid::synthdid_estimate()` (Arkhangelsky et al.'s official package)
- Key R functions matched: `sc.weight.fw()` (Frank-Wolfe), `sparsify_function` (sparsification), `vcov.synthdid_estimate()` (variance)

**Requirements checklist:**
- [x] Unit weights: Frank-Wolfe on collapsed form (T_pre, N_co+1), two-pass sparsification (100 iters -> sparsify -> 10000 iters)
- [x] Time weights: Frank-Wolfe on collapsed form (N_co, T_pre+1), last column = per-control post mean
- [x] Unit and time weights: sum to 1, non-negative (simplex constraint)
- [x] Auto-regularization: noise_level = sd(first_diffs), zeta_omega = (N1*T1)^0.25 * noise_level, zeta_lambda = 1e-6 * noise_level
- [x] Sparsification: v[v <= max(v)/4] = 0; v = v/sum(v)
- [x] Placebo SE formula: sqrt((r-1)/r) * sd(placebo_estimates)
- [x] Placebo SE: re-estimates omega and lambda per replication (matching R's update.omega=TRUE, update.lambda=TRUE)
- [x] Bootstrap: fixed weights (original lambda unchanged, omega renormalized for resampled controls)
- [x] Returns both unit and time weights for interpretation
- [x] Column centering (intercept=True) in Frank-Wolfe optimization

---

## TripleDifference

**Primary source:** [Ortiz-Villavicencio, M., & Sant'Anna, P.H.C. (2025). Better Understanding Triple Differences Estimators. arXiv:2505.09942.](https://arxiv.org/abs/2505.09942)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires all 8 cells of the 2×2×2 design: Group(0/1) × Period(0/1) × Treatment(0/1)
- Warns if any cell has fewer than threshold observations
- Propensity score overlap required for IPW/DR methods

*Estimator equation (as implemented):*

Three-DiD decomposition (matching R's `triplediff::ddd()`):
```
Subgroups: 4=G1P1, 3=G1P0, 2=G0P1, 1=G0P0
DDD = DiD_3 + DiD_2 - DiD_1
```
where `DiD_j` is a pairwise DiD comparing subgroup j vs subgroup 4 (reference).

Each pairwise DiD uses the selected estimation method (DR, IPW, or RA) with
repeated cross-section implementation (`panel=FALSE` in R).

Regression adjustment (RA): Separate OLS per subgroup-time cell within each
pairwise comparison, imputed counterfactual means.

IPW: Propensity score P(subgroup=4|X) within {j, 4} subset, Hajek normalization.

Doubly robust (DR): Combines outcome regression and IPW with efficiency correction
(OR bias correction term).

*Standard errors (all methods):*

Individual-level (default):
```
SE = std(w₃·IF₃ + w₂·IF₂ - w₁·IF₁, ddof=1) / sqrt(n)
```
where `w_j = n / n_j`, `n_j = |{subgroup=j}| + |{subgroup=4}|`, and `IF_j` is the
per-observation influence function for pairwise DiD j (padded to full n with zeros).

Cluster-robust (when `cluster` parameter is provided):
```
SE = sqrt( (G/(G-1)) * (1/n²) * Σ_c ψ_c² )
```
where `G` is the number of clusters, `ψ_c = Σ_{i∈c} IF_i` is the sum of the combined
influence function within cluster `c`, and the `G/(G-1)` factor is the Liang-Zeger
finite-sample adjustment.

Note: IF-based SEs are inherently heteroskedasticity-robust; the `robust` parameter
has no additional effect.

*Edge cases:*
- Propensity scores near 0/1: trimmed at `pscore_trim` (default 0.01)
- Empty cells: raises ValueError with diagnostic message
- Low cell counts: warns when any cell has fewer than 10 observations
- Cluster-robust SE: requires at least 2 clusters (raises `ValueError`)
- Cluster IDs: must not contain NaN (raises `ValueError`)
- Overlap warning: emitted when >5% of observations are trimmed at pscore bounds (IPW/DR only)
- Propensity score estimation failure: controlled by `pscore_fallback` parameter
  (default `"error"`). If `pscore_fallback="error"`, the error is raised. If
  `pscore_fallback="unconditional"`, falls back to unconditional probability
  P(subgroup=4), sets hessian=None (skipping PS correction in influence
  function), emits UserWarning. When `rank_deficient_action="error"`, errors
  are always re-raised regardless of `pscore_fallback`.
- **Events Per Variable (EPV) diagnostics:** Per-logit EPV =
  min(n_subgroup_j, n_subgroup_4) / n_covariates checked before IRLS.
  Default threshold: 10 (Peduzzi et al. 1996). Warns when EPV < threshold;
  errors when `rank_deficient_action="error"`.
- **Note:** `pscore_fallback` default changed from unconditional to error.
  Set `pscore_fallback="unconditional"` for legacy behavior.
- Collinear covariates: detected via pivoted QR in `solve_ols()`, action controlled by
  `rank_deficient_action` ("warn", "error", "silent")
- Non-finite influence function values (e.g., from extreme propensity scores in IPW/DR
  or near-singular design): warns and sets SE to NaN, propagated to t_stat/p_value/CI
  via safe_inference()
- NaN inference for undefined statistics:
  - t_stat: Uses NaN (not 0.0) when SE is non-finite or zero
  - p_value and CI: Also NaN when t_stat is NaN
  - **Note**: Defensive enhancement; reference implementation behavior not yet documented

**Reference implementation(s):**
- R `triplediff::ddd()` (v0.2.1, CRAN) — official companion by paper authors

**Requirements checklist:**
- [x] All 8 cells (G×P×T) must have observations
- [x] Propensity scores clipped at `pscore_trim` bounds
- [x] Doubly robust consistent if either propensity or outcome model correct
- [x] Returns cell means for diagnostic inspection
- [x] Supports RA, IPW, and DR estimation methods
- [x] Three-DiD decomposition: DDD = DiD_3 + DiD_2 - DiD_1 (matching R)
- [x] Influence function SE: std(w3·IF_3 + w2·IF_2 - w1·IF_1) / sqrt(n)
- [x] Cluster-robust SE via Liang-Zeger variance on influence function
- [x] ATT and SE match R within <0.001% for all methods and DGP types
- [x] Survey design support: all methods (reg, IPW, DR) with weighted OLS/logit + TSL on combined influence functions. Weighted solve_logit() for propensity scores in IPW/DR paths.
- **Note:** TripleDifference survey SE: for IPW/DR, pairwise IFs incorporate survey weights via weighted Riesz representers (`riesz *= weights`), so the combined IF is divided by per-observation survey weights (`inf / sw`) before passing to `compute_survey_vcov()` to prevent double-weighting. For regression (RA), pairwise IFs are already on the unweighted residual scale (WLS fits use weights internally but the IF is not Riesz-multiplied), so the combined IF passes directly to TSL without de-weighting. The OLS nuisance IF corrections in DR mode use weighted cross-products normalized by subgroup row count `n` (not `sum(weights)`).

---

## StaggeredTripleDifference

**Primary source:** [Ortiz-Villavicencio, M., & Sant'Anna, P.H.C. (2025). Better Understanding Triple Differences Estimators. arXiv:2505.09942.](https://arxiv.org/abs/2505.09942)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires balanced panel with enabling-group `S_i`, binary eligibility `Q_i` (time-invariant), and outcome `Y`
- Eligibility must be binary (0/1) — raises `ValueError` if not
- Eligibility must be time-invariant within each unit — raises `ValueError` if varying
- Requires both eligible (Q=1) and ineligible (Q=0) units
- Warns if any (S, Q) cell in a three-DiD comparison has < 5 units
- Warns if no valid comparison groups exist for a (g, t) pair (skips that pair)
- Propensity score overlap enforced by clipping at `pscore_trim` (default 0.01)
- **Events Per Variable (EPV) diagnostics:** Per-DiD EPV =
  min(n_subgroup_j, n_subgroup_4) / n_covariates checked before IRLS.
  Default threshold: 10 (Peduzzi et al. 1996). Warns when EPV < threshold;
  errors when `rank_deficient_action="error"`.
- **Note:** When multiple comparison cohorts `g_c` contribute to the same
  ATT(g,t) cell, `results.epv_diagnostics[(g,t)]` retains the worst-case
  (minimum EPV) across all contributing propensity fits, rather than per-fit
  diagnostics. This is a conservative cell-level summary.
- Propensity score estimation failure: controlled by `pscore_fallback` parameter
  (default `"error"`). If `pscore_fallback="error"`, the error is raised. If
  `pscore_fallback="unconditional"`, falls back to unconditional propensity with
  warning. When `rank_deficient_action="error"`, errors are always re-raised.
- **Note:** `pscore_fallback` default changed from unconditional to error.
  Set `pscore_fallback="unconditional"` for legacy behavior.
- Warns on singular GMM covariance matrix (falls back to pseudoinverse)

*Data structure:*

Balanced panel. Key variables:
- `S_i` (`first_treat`): enabling group — 0 or inf for never-enabled
- `Q_i` (`eligibility`): binary, time-invariant eligibility indicator
- Treatment: `D_{i,t} = 1{t >= S_i AND Q_i = 1}` (absorbing)
- Covariates `X_i`: time-invariant (first observation per unit used)

*Estimator equation (Equation 4.1 in paper, as implemented):*

Three-DiD decomposition for each (g, g_c, t) triple:

```
DDD(g, g_c, t) = DiD_A + DiD_B - DiD_C
```

where each pairwise DiD operates on panel outcome changes `delta_Y = Y_t - Y_b`:
- DiD_A: treated (S=g, Q=1) vs (S=g, Q=0)     [+1, paper Term 1]
- DiD_B: treated (S=g, Q=1) vs (S=g_c, Q=1)   [+1, paper Term 2]
- DiD_C: treated (S=g, Q=1) vs (S=g_c, Q=0)   [-1, paper Term 3]

This sign convention matches both the paper's Equation 4.1 and the existing
`TripleDifference` decomposition (DDD = DiD_3 + DiD_2 - DiD_1 with subgroups
4=G1P1, 3=G1P0, 2=G0P1, 1=G0P0).

Valid comparison groups: for `control_group="nevertreated"`, only the never-enabled
cohort (S=0). For `control_group="notyettreated"`, `G_c = {g_c : g_c > max(t, base_period)
+ anticipation}`, plus never-enabled.

- **Deviation from paper:** The paper's Section 4 defines admissible comparison cohorts
  as `g_c > max(g, t)`. The implementation follows the companion R package `triplediff`
  which uses `g_c > max(t, base_period) + anticipation`. These rules differ for
  pre-treatment cells (`t < g`) when a later cohort lies in `(t, g)`: the paper would
  exclude it, while the R package (and this implementation) may include it depending
  on the base period. The R-matching rule correctly accounts for the anticipation
  parameter and base-period selection in the comparison-group filter.

*With covariates / doubly robust (DR, recommended):*

Each pairwise DiD uses the CallawaySantAnna DR estimator on outcome changes:
1. Fit outcome regression `E[delta_Y | X]` on control units (OLS)
2. Estimate propensity score `P(treated | X)` within each 2-cell subset (logistic)
3. Combine: `ATT = mean(treated_change - m_hat) + sum(w_ipw * (m_hat - control_change)) / n_t`

*GMM-optimal combination across comparison groups (Equations 4.11-4.12):*

```
ATT_gmm(g,t) = w_gmm' @ [ATT_1, ..., ATT_k]
w_gmm = Omega^{-1} @ 1 / (1' @ Omega^{-1} @ 1)
```

where `Omega[j,l] = (1/n) * sum_i IF_j[i] * IF_l[i]` is estimated from influence
functions across comparison groups. Minimizes asymptotic variance subject to `sum(w) = 1`.

*Aggregation:*

Event study (Equation 4.13): cohort-share-weighted average across cohorts for each
relative time `e = t - g`. Reuses `CallawaySantAnnaAggregationMixin._aggregate_event_study()`.

Overall ATT: cohort-size-weighted average across post-treatment (g,t) pairs.
Reuses `CallawaySantAnnaAggregationMixin._aggregate_simple()`. Note: this is the
simple post-treatment aggregation, not the paper's Equation 4.14 (which averages
over event-study effects).

Group effects: average across post-treatment time periods for each cohort.
Reuses `CallawaySantAnnaAggregationMixin._aggregate_by_group()`.

All aggregation SEs include the WIF (Weight Influence Function) adjustment for
uncertainty in cohort-share weights, inherited from the CallawaySantAnna mixin.

- **Deviation from R:** Aggregation weights and WIF use the eligible-treated
  population `P(S=g, Q=1)` (matching the paper's Eq 4.13, where `G_i` is defined
  only for `Q=1` units). R's `agg_ddd()` uses `P(S=g)` (all units in the enabling
  group, including ineligible). This is implemented by setting `unit_cohorts=0` for
  ineligible units before calling the aggregation mixin.
- **Note:** Per-cohort group-effect SEs include WIF via the inherited mixin.
  R's `agg_ddd(type="group")` uses `wif=NULL` for per-cohort aggregation since
  within-cohort weights are fixed. This makes our per-cohort group-effect SEs
  slightly conservative relative to R.

*Standard errors:*

Individual (g,t) level:
```
SE(g,t) = std(IF_gmm, ddof=1) / sqrt(n)
```
where `IF_gmm = w_gmm' @ IF_matrix` is the GMM-combined unit-level influence function
(length n_units, zero-padded for non-participating units). Inherently
heteroskedasticity-robust via the influence function approach.

Aggregation SEs: via WIF-adjusted combined influence functions from the
CallawaySantAnna aggregation mixin.

Bootstrap: multiplier bootstrap (Algorithm 1 of Callaway & Sant'Anna 2021) via
`CallawaySantAnnaBootstrapMixin._run_multiplier_bootstrap()`. Supports
Rademacher, Mammen, and Webb weight distributions. Provides simultaneous
confidence bands (sup-t) for event study.

- **Note:** Matches R `triplediff` package `compute_did()` formulation:
  Hajek-normalized Riesz representers, separate M1/M3 OR corrections on
  treated/control IF components, PS correction via logistic Hessian and score
  function, hessian = (X'WX)^{-1} * n_pair. Three-DiD IF combination weights
  use `w_j = n_cell / n_pair_j` (matching R's att_dr). GMM Omega estimated via
  sample covariance (ddof=1). Per-(g,t) SE uses R's GMM formula
  `sqrt(1 / (n * sum(Omega_inv)))` for multiple comparison groups, or
  `sqrt(sum(IF^2) / n^2)` for single comparison group.
- **Deviation from R:** Propensity scores are clipped to `[pscore_trim, 1-pscore_trim]`
  (default 0.01). R's `triplediff` uses hard exclusion (`keep_ps`) for control units
  with `pscore >= 0.995` but does not apply a lower bound. The soft-clipping approach
  retains all observations with bounded weights, which is more conservative under
  moderate overlap violations.
- **Note:** The `cluster` parameter is accepted but not currently wired to the
  analytical SE computation. The multiplier bootstrap provides unit-level
  clustering. Full cluster-robust analytical SEs are deferred.
- **Note:** Full survey design support (pweight only). Survey weights enter
  propensity score estimation (weighted IRLS), outcome regression (WLS), and
  Riesz representer computation. IF combination weights (w1/w2/w3) use
  survey-weighted cell sizes. Aggregated SEs use `compute_survey_if_variance()`
  (TSL) or `compute_replicate_if_variance()` (replicate weights). Bootstrap
  uses PSU-level multiplier weights. The R `triplediff` package does not
  support survey weights.
- **Deviation from R:** Event-study and simple aggregation reuse
  `CallawaySantAnnaAggregationMixin` cohort-size weights (`n_treated` per cohort)
  instead of R's `agg_ddd()` group-probability weights (`pg = P(G=g)` over all
  units including ineligible). Group-time ATT(g,t) values are identical; only the
  weighted average across (g,t) pairs differs.

*Edge cases:*
- Single comparison group: GMM reduces to w=[1], no matrix inversion
- Zero valid comparison groups for a (g,t): skipped with warning
- Singular GMM covariance: falls back to pseudoinverse with warning
- Small cells (< 5 units): warns but proceeds
- Non-finite ATT from a comparison group: excluded from GMM combination
- Never-enabled encoded as inf: normalized to 0 internally
- No valid (g,t) pairs at all: raises `ValueError`

**Reference implementation(s):**
- R `triplediff` (companion package by paper authors) — not yet validated against

**Requirements checklist:**
- [x] Panel data with (unit, time, enabling-group S, eligibility Q, outcome Y)
- [x] Three comparison sub-groups per (g, g_c): (S=g, Q=0), (S=g_c, Q=1), (S=g_c, Q=0)
- [x] Individual comparison cohorts, never pooled — combined via GMM weights
- [x] Comparison groups satisfy g_c > max(t, base_period) + anticipation (notyettreated)
  or g_c = never-enabled only (nevertreated)
- [x] Doubly robust: consistent if either propensity or outcome model correct (per component)
- [x] GMM-optimal weighting via closed-form inverse-variance formula
- [x] Event-study aggregation with cohort-share weights (via CS mixin)
- [x] Pre-treatment event-study coefficients constructable
- [x] Influence-function-based SEs
- [x] Multiplier bootstrap for simultaneous confidence bands (via CS mixin)
- [ ] Cluster-robust analytical SEs (accepted but not wired — deferred)
- [x] Survey design support (pweight, strata/PSU/FPC, replicate weights)
- [x] Validation against R `triplediff` package: group-time ATT and SE match within
  0.001% across 10 scenarios (3 seeds, 3 methods, both control group modes).
  Aggregation (event study, overall ATT) uses CS mixin cohort-size weights which
  differ from R's `agg_ddd()` group-probability weights (within 25%); this is a
  documented weighting choice, not a specification violation.

---

## TROP

**Primary source:** [Athey, S., Imbens, G.W., Qu, Z., & Viviano, D. (2025). Triply Robust Panel Estimators. arXiv:2508.21536.](https://arxiv.org/abs/2508.21536)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires sufficient pre-treatment periods for factor estimation (at least 2)
- Warns if estimated rank seems too high/low relative to panel dimensions
- Unit weights can become degenerate if λ_unit too large
- Returns Q(λ) = ∞ if ANY LOOCV fit fails (Equation 5 compliance)

*Treatment indicator (D matrix) semantics:*

D must be an **ABSORBING STATE** indicator, not a treatment timing indicator:
- D[t, i] = 0 for all t < g_i (pre-treatment periods for unit i)
- D[t, i] = 1 for all t >= g_i (during and after treatment for unit i)

where g_i is the treatment start time for unit i.

For staggered adoption, different units have different treatment start times g_i.
The D matrix naturally handles this - distances use periods where BOTH units
have D=0, matching the paper's (1 - W_iu)(1 - W_ju) formula in Equation 3.

**Wrong D specification**: If user provides event-style D (only first treatment period
has D=1), ATT will be incorrect - document this clearly.

*ATT definition (Equation 1, Section 6.1):*
```
τ̂ = (1 / Σ_i Σ_t W_{it}) Σ_{i=1}^N Σ_{t=1}^T W_{it} τ̂_{it}(λ̂)
```
- ATT averages over ALL cells where D_it=1 (treatment indicator)
- No separate "post_periods" concept - D matrix is the sole input for treatment timing
- Supports general assignment patterns including staggered adoption

*Estimator equation (as implemented, Section 2.2):*

Working model (separating unit/time FE from regularized factor component):
```
Y_it(0) = α_i + β_t + L_it + ε_it,   E[ε_it | L] = 0
```
where α_i are unit fixed effects, β_t are time fixed effects, and L = UΣV' is a low-rank
factor structure. The FE are estimated separately from L because L is regularized but
the fixed effects are not.

Optimization (Equation 2):
```
(α̂, β̂, L̂) = argmin_{α,β,L} Σ_j Σ_s θ_s^{i,t} ω_j^{i,t} (1-W_js)(Y_js - α_j - β_s - L_js)² + λ_nn ||L||_*
```
Solved via alternating minimization. For α, β: weighted least squares (closed form).
The global solver adds an intercept μ and solves for (μ, α, β, L) on control data only,
extracting τ_it post-hoc as residuals (see Global section below).
For L: proximal gradient with step size η = 1/(2·max(W)):
```
Gradient step: G = L + (W/max(W)) ⊙ (R - L)
Proximal step: L = U × soft_threshold(Σ, η·λ_nn) × V'  (SVD of G = UΣV')
```
where R is the residual after removing fixed effects.
Both the local and global solvers use FISTA/Nesterov acceleration for the
inner L update (O(1/k²) convergence rate, up to 20 inner iterations per
outer alternating step).

Per-observation weights (Equation 3):
```
θ_s^{i,t}(λ) = exp(-λ_time × |t - s|)

ω_j^{i,t}(λ) = exp(-λ_unit × dist^unit_{-t}(j, i))

dist^unit_{-t}(j, i) = (Σ_u 1{u≠t}(1-W_iu)(1-W_ju)(Y_iu - Y_ju)² / Σ_u 1{u≠t}(1-W_iu)(1-W_ju))^{1/2}
```
Note: weights are per-(i,t) observation-specific. The distance formula excludes the
target period t and uses only periods where both units are untreated (W=0).

*Special cases (Section 2.2):*
- λ_nn=∞, ω_j=θ_s=1 (uniform weights) → recovers DID/TWFE
- ω_j=θ_s=1, λ_nn<∞ → recovers Matrix Completion (Athey et al. 2021)
- λ_nn=∞ with specific ω_j, θ_s → recovers SC/SDID

*LOOCV tuning parameter selection (Equation 5, Footnote 2):*
```
Q(λ) = Σ_{j,s: D_js=0} [τ̂_js^loocv(λ)]²
```
- Score is **SUM** of squared pseudo-treatment effects on control observations
- **Two-stage procedure** (per paper's footnote 2):
  - Stage 1: Univariate grid searches with extreme fixed values
    - λ_time search: fix λ_unit=0, λ_nn=∞ (disabled)
    - λ_nn search: fix λ_time=0 (uniform time weights), λ_unit=0
    - λ_unit search: fix λ_nn=∞, λ_time=0
  - Stage 2: Cycling (coordinate descent) until convergence
- **"Disabled" parameter semantics** (per paper Section 4.3, Table 5, Footnote 2):
  - `λ_time=0`: Uniform time weights (disabled), because exp(-0 × dist) = 1
  - `λ_unit=0`: Uniform unit weights (disabled), because exp(-0 × dist) = 1
  - `λ_nn=∞`: Factor model disabled (L=0), because infinite penalty; converted to `1e10` internally
  - **Note**: `λ_nn=0` means NO regularization (full-rank L), which is the OPPOSITE of "disabled"
  - **Validation**: `lambda_time_grid` and `lambda_unit_grid` must not contain inf. A `ValueError` is raised if they do, guiding users to use 0.0 for uniform weights per Eq. 3.
- **LOOCV failure handling** (Equation 5 compliance):
  - If ANY LOOCV fit fails for a parameter combination, Q(λ) = ∞
  - A warning is emitted on the first failure with the observation (t, i) and λ values
  - Subsequent failures for the same λ are not individually warned (early return)
  - This ensures λ selection only considers fully estimable combinations

*Standard errors:*
- Block bootstrap preserving panel structure (Algorithm 3)

*Edge cases:*
- Rank selection: automatic via cross-validation, information criterion, or elbow
- Zero singular values: handled by soft-thresholding
- Extreme distances: weights regularized to prevent degeneracy
- LOOCV fit failures: returns Q(λ) = ∞ on first failure (per Equation 5 requirement that Q sums over ALL control observations where D==0); if all parameter combinations fail, falls back to defaults (1.0, 1.0, 0.1)
- **λ_nn=∞ implementation**: Only λ_nn uses infinity (converted to 1e10 for computation):
  - λ_nn=∞ → 1e10 (large penalty → L≈0, factor model disabled)
  - Conversion applied to grid values during LOOCV (including Rust backend)
  - Conversion applied to selected values for point estimation
  - Conversion applied to selected values for variance estimation (ensures SE matches ATT)
  - **Results storage**: `TROPResults` stores *original* λ_nn value (inf), while computations use 1e10. λ_time and λ_unit store their selected values directly (0.0 = uniform).
- **Empty control observations**: If no valid control observations exist, returns Q(λ) = ∞ with warning. A score of 0.0 would incorrectly "win" over legitimate parameters.
- **Infinite LOOCV score handling**: If best LOOCV score is infinite, `best_lambda` is set to None, triggering defaults fallback
- Validation: requires at least 2 periods before first treatment
- **D matrix validation**: Treatment indicator must be an absorbing state (monotonic non-decreasing per unit)
  - Detection: `np.diff(D, axis=0) < 0` for any column indicates violation
  - Handling: Raises `ValueError` with list of violating unit IDs and remediation guidance
  - Error message includes: "convert to absorbing state: D[t, i] = 1 for all t >= first treatment period"
  - **Rationale**: Event-style D (0→1→0) silently biases ATT; runtime validation prevents misuse
  - **Unbalanced panels**: Missing unit-period observations are allowed. Monotonicity validation checks each unit's *observed* D sequence for monotonicity, which correctly catches 1→0 violations that span missing period gaps (e.g., D[2]=1, missing [3,4], D[5]=0 is detected as a violation even though the gap hides the transition in adjacent-period checks).
  - **n_post_periods metadata**: Counts periods where D=1 is actually observed (at least one unit has D=1), not calendar periods from first treatment. In unbalanced panels where treated units are missing in some post-treatment periods, only periods with observed D=1 values are counted.
- Wrong D specification: if user provides event-style D (only first treatment period),
  the absorbing-state validation will raise ValueError with helpful guidance
- **Bootstrap minimum**: `n_bootstrap` must be >= 2 (enforced via `ValueError`). TROP uses bootstrap for all variance estimation — there is no analytical SE formula.
- **LOOCV failure metadata**: When LOOCV fits fail in the Rust backend, the first failed observation coordinates (t, i) are returned to Python for informative warning messages
- **Inference CI distribution**: After `safe_inference()` migration, CI uses t-distribution (df = max(1, n_treated_obs - 1)), consistent with p_value. Previously CI used normal-distribution while p_value used t-distribution (inconsistent). This is a minor behavioral change; CIs may be slightly wider for small n_treated_obs.

**Reference implementation(s):**
- Authors' replication code (forthcoming)

**Requirements checklist:**
- [x] Factor matrix estimated via soft-threshold SVD
- [x] Unit weights: `exp(-λ_unit × distance)` (unnormalized, matching Eq. 2)
- [x] LOOCV implemented for tuning parameter selection
- [x] LOOCV uses SUM of squared errors per Equation 5
- [x] Multiple rank selection methods: cv, ic, elbow
- [x] Returns factor loadings and scores for interpretation
- [x] ATT averages over all D==1 cells (general assignment patterns)
- [x] No post_periods parameter (D matrix determines treatment timing)
- [x] D matrix semantics documented (absorbing state, not event indicator)
- [x] Unbalanced panels supported (missing observations don't trigger false violations)
- **Note:** Survey support: weights, strata, PSU, and FPC are all supported via Rao-Wu rescaled bootstrap with cross-classified pseudo-strata (Phase 6). Rust backend remains pweight-only; full-design surveys fall back to the Python bootstrap path. Survey weights enter ATT aggregation only — population-weighted average of per-observation treatment effects. Model fitting (kernel weights, LOOCV, nuclear norm regularization) stays unchanged. Rust and Python bootstrap paths both support survey-weighted ATT in each iteration.

### TROP Global Estimation Method

**Method**: `method="global"` in TROP estimator

**Approach**: Computationally efficient adaptation using the (1-W) masking
principle from Eq. 2. Fits a single global model on control data, then
extracts treatment effects as post-hoc residuals. For the paper's full
per-treated-cell estimator (Algorithm 2), use `method='local'`.

**Objective function** (Equation G1):
```
min_{μ, α, β, L}  Σ_{i,t} (1-W_{it}) × δ_{it} × (Y_{it} - μ - α_i - β_t - L_{it})² + λ_nn×||L||_*
```

where:
- (1-W_{it}) masks out treated observations — model is fit on control data only
- δ_{it} = δ_time(t) × δ_unit(i) are observation weights (product of time and unit weights)
- μ is the intercept
- α_i are unit fixed effects
- β_t are time fixed effects
- L_{it} is the low-rank factor component

**Post-hoc treatment effect extraction**:
```
τ̂_{it} = Y_{it} - μ̂ - α̂_i - β̂_t - L̂_{it}    for all (i,t) where W_{it} = 1
ATT = mean(τ̂_{it})  over all treated observations
```

Treatment effects are **heterogeneous** per-observation values. ATT is their mean.

**Weight computation** (differs from local):
- Time weights: δ_time(t) = exp(-λ_time × |t - center|) where center = T - treated_periods/2
- Unit weights: δ_unit(i) = exp(-λ_unit × RMSE(i, treated_avg))
  where RMSE is computed over pre-treatment periods comparing to average treated trajectory
- (1-W) masking applied after outer product: δ_{it} = 0 for all treated cells

**Implementation approach** (without CVXPY):

1. **Without low-rank (λ_nn = ∞)**: Standard weighted least squares
   - Build design matrix with unit/time dummies (no treatment indicator)
   - Solve via np.linalg.lstsq for (μ, α, β) using (1-W)-masked weights

2. **With low-rank (finite λ_nn)**: Alternating minimization
   - Alternate between:
     - Fix L, solve weighted LS for (μ, α, β)
     - Fix (μ, α, β), proximal gradient for L:
       - Lipschitz constant of ∇f is L_f = 2·max(δ)
       - Step size η = 1/L_f = 1/(2·max(δ))
       - Proximal operator: soft_threshold(gradient_step, η·λ_nn)
       - Inner solver uses FISTA/Nesterov acceleration (O(1/k²))
   - Continue until max(|L_new - L_old|) < tol

3. **Post-hoc**: Extract τ̂_{it} = Y_{it} - μ̂ - α̂_i - β̂_t - L̂_{it} for treated cells

**LOOCV parameter selection** (unified with local, Equation 5):
Following paper's Equation 5 and footnote 2:
```
Q(λ) = Σ_{j,s: D_js=0} [τ̂_js^loocv(λ)]²
```
where τ̂_js^loocv is the pseudo-treatment effect at control observation (j,s)
with that observation excluded from fitting.

For global method, LOOCV works as follows:
1. For each control observation (t, i):
   - Zero out weight δ_{ti} = 0 (exclude from weighted objective)
   - Fit global model on remaining data → obtain (μ̂, α̂, β̂, L̂)
   - Compute pseudo-treatment: τ̂_{ti} = Y_{ti} - μ̂ - α̂_i - β̂_t - L̂_{ti}
2. Score = Σ τ̂_{ti}² (sum of squared pseudo-treatment effects)
3. Select λ combination that minimizes Q(λ)

**Rust acceleration**: The LOOCV grid search is parallelized in Rust for 5-10x speedup.
- `loocv_grid_search_global()` - Parallel LOOCV across all λ combinations
- `bootstrap_trop_variance_global()` - Parallel bootstrap variance estimation

**Key differences from local method**:
- Global weights (distance to treated block center) vs. per-observation weights
- Single model fit per λ combination vs. N_treated fits
- Treatment effects are post-hoc residuals from a single global model (global)
  vs. post-hoc residuals from per-observation models (local)
- Both use (1-W) masking (control-only fitting)
- Faster computation for large panels

**Assumptions**:
- **Simultaneous adoption (enforced)**: The global method requires all treated units
  to receive treatment at the same time. A `ValueError` is raised if staggered
  adoption is detected (units first treated at different periods). Treatment timing is
  inferred once and held constant for bootstrap variance estimation.
  For staggered adoption designs, use `method="local"`.

**Reference**: Adapted from reference implementation. See also Athey et al. (2025).

**Edge Cases (treated NaN outcomes):**
- **Partial NaN**: When some treated outcomes Y_{it} are NaN/missing:
  - `_extract_posthoc_tau()` (global) skips these cells; only finite τ̂ values are averaged
  - Local loop skips NaN outcomes entirely (no model fit, no tau appended)
  - `n_treated_obs` in results reflects valid (finite) count, not total D==1 count
  - `df_trop = max(1, n_valid_treated - 1)` uses valid count
  - Warning issued when n_valid_treated < total treated count
- **All NaN**: When all treated outcomes are NaN:
  - ATT = NaN, warning issued
  - `n_treated_obs = 0`
- **Bootstrap SE with <2 draws**: Returns `se=NaN` (not 0.0) when zero bootstrap
  iterations succeed. `safe_inference()` propagates NaN downstream.

**Requirements checklist:**
- [x] Same LOOCV framework as local (Equation 5)
- [x] Global weight computation using treated block center
- [x] (1-W) masking for control-only fitting (per paper Eq. 2)
- [x] Alternating minimization for nuclear norm penalty
- [x] Returns ATT = mean of per-observation post-hoc τ̂_{it}
- [x] Rust acceleration for LOOCV and bootstrap

---

# Diagnostics & Sensitivity

## PlaceboTests

**Module:** `diff_diff/diagnostics.py`

*Edge cases:*
- NaN inference for undefined statistics:
  - `permutation_test`: t_stat is NaN when permutation SE is zero (all permutations produce identical estimates)
  - `leave_one_out_test`: t_stat, p_value, CI are NaN when LOO SE is zero (all LOO effects identical)
  - **Note**: Defensive enhancement matching CallawaySantAnna NaN convention

---

## BaconDecomposition

**Primary source:** [Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. *Journal of Econometrics*, 225(2), 254-277.](https://doi.org/10.1016/j.jeconom.2021.03.014)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires variation in treatment timing (staggered adoption)
- Warns if only one treatment cohort (decomposition not meaningful)
- Uses never-treated units as controls when present; falls back to timing-only comparisons otherwise

*Estimator equation (as implemented):*

TWFE decomposes as:
```
τ̂^TWFE = Σ_k s_k × τ̂_k
```
where k indexes 2×2 comparisons and s_k are Bacon weights.

Three comparison types:
1. **Treated vs. Never-treated** (if never-treated exist):
   ```
   τ̂_{T,U} = (Ȳ_{T,post} - Ȳ_{T,pre}) - (Ȳ_{U,post} - Ȳ_{U,pre})
   ```

2. **Earlier vs. Later-treated** (Earlier as treated, Later as control pre-treatment):
   ```
   τ̂_{k,l} = DiD using early-treated as treatment, late-treated as control
   ```

3. **Later vs. Earlier-treated** (problematic: uses post-treatment outcomes as control):
   ```
   τ̂_{l,k} = DiD using late-treated as treatment, early-treated (post) as control
   ```

Weights depend on group sizes and variance in treatment timing.

*Standard errors:*
- Not typically computed (decomposition is exact)
- Individual 2×2 estimates can have SEs

*Edge cases:*
- Continuous treatment: not supported, requires binary
- Weights may be negative for later-vs-earlier comparisons
- Single treatment time: no decomposition possible

**Reference implementation(s):**
- R: `bacondecomp::bacon()`
- Stata: `bacondecomp`

**Requirements checklist:**
- [ ] Three comparison types: treated_vs_never, earlier_vs_later, later_vs_earlier
- [ ] Weights sum to approximately 1 (numerical precision)
- [ ] TWFE coefficient ≈ weighted sum of 2×2 estimates
- [ ] Visualization shows weight vs. estimate by comparison type
- [x] Survey design support (Phase 3): weighted cell means, weighted within-transform, weighted group shares
- **Note:** Bacon decomposition with survey weights is diagnostic; exact-sum guarantee is approximate; `weights="exact"` requires within-unit-constant survey columns (approximate path accepts time-varying weights)

---

## HonestDiD

**Primary source:** [Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends. *Review of Economic Studies*, 90(5), 2555-2591.](https://doi.org/10.1093/restud/rdad018)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires event-study estimates with pre-treatment coefficients
- Warns if pre-treatment coefficients suggest parallel trends violation
- M=0 for Delta^SD: enforces linear trend extrapolation (not exact parallel trends)

*Restriction classes (Equations 8, Section 2.3):*

Delta^SD(M) — Smoothness (second differences, all periods):
```
Δ^SD(M) = {δ : |(δ_{t+1} − δ_t) − (δ_t − δ_{t-1})| ≤ M, for all t}
```
with δ_0 = 0 at the pre-post boundary. M=0 enforces linear trends.

Delta^RM(M̄) — Relative magnitudes (post-treatment first differences):
```
Δ^RM(M̄) = {δ : |δ_{t+1} − δ_t| ≤ M̄ × max_{s<0} |δ_{s+1} − δ_s|, for all t ≥ 0}
```
Post-treatment consecutive first differences bounded by M̄ × max pre-treatment first difference. Union of polyhedra (one per max location).

*Identified set (Equations 5-6):*
```
θ^lb = l'β_post − max{l'δ_post : δ ∈ Δ, δ_pre = β_pre}
θ^ub = l'β_post − min{l'δ_post : δ ∈ Δ, δ_pre = β_pre}
```
CRITICAL: δ_pre = β_pre pins pre-treatment violations to observed coefficients. Solved via LP (scipy.optimize.linprog).

*Inference (Sections 3.2, 4.1):*
- Delta^SD: Optimal FLCI — jointly optimizes affine estimator direction and half-length via folded normal quantile cv_α(bias/se) (Equation 18). When `df_survey` is present, uses folded non-central t (`scipy.stats.nct`) instead of folded normal; `df_survey=0` → NaN inference. For M=0, uses `_get_critical_value(alpha, df)` (standard t/z).
- Delta^RM: Paper recommends ARP conditional/hybrid confidence sets (Equations 14-15, κ = α/10). Currently uses **naive FLCI** unconditionally (conservative — wider CIs, valid coverage). ARP infrastructure exists but is disabled.
- **Note (deviation from R):** Delta^RM CIs use naive FLCI (`lb - z*se, ub + z*se`) instead of the paper's ARP hybrid. R's `HonestDiD` package implements full ARP conditional/hybrid. Our naive FLCI is conservative (wider, valid coverage) but does not adapt to the length of the identified set. ARP implementation deferred (see TODO.md).
- **Note:** `method="combined"` (Delta^SDRM) uses naive FLCI on the intersection of Delta^SD and Delta^RM bounds. The paper proves FLCI is NOT consistent for Delta^SDRM (Proposition 4.2). The paper recommends ARP hybrid for non-SD restriction classes. This is a known conservative approximation; a runtime UserWarning is emitted.

*Standard errors:*
- Inherits Σ̂ from underlying event-study estimation
- Sensitivity analysis reports identified set bounds and confidence sets

*Edge cases:*
- M=0 for Δ^SD: linear extrapolation, point identification, FLCI near-optimal
- M̄=0 for Δ^RM: post-treatment first differences = 0, point identification
- Breakdown point: smallest M where CI includes zero
- Negative M: not valid (constraints become infeasible)
- **Note:** Phase 7d: survey variance support. When input results carry `survey_metadata` with `df_survey`, Delta^SD smoothness uses folded non-central t critical values (`scipy.stats.nct`); Delta^RM and naive FLCI paths use `_get_critical_value(alpha, df)` (standard t-distribution). `df_survey=0` → NaN inference. CallawaySantAnnaResults stores `event_study_vcov` (full cross-event-time VCV from IF vectors), which HonestDiD uses instead of the diagonal fallback. For replicate-weight designs, the event-study VCV falls back to diagonal (multivariate replicate VCV deferred).
- **Note (deviation from R):** When HonestDiD receives bootstrap-fitted CallawaySantAnna results (`n_bootstrap > 0`), the full event-study covariance is unavailable (cleared to prevent mixing analytical VCV with bootstrap SEs). HonestDiD falls back to `diag(se^2)` from the bootstrap SEs with a UserWarning. R's `honest_did.AGGTEobj` computes a full covariance from the influence function matrix; implementing bootstrap event-study covariance is deferred. For full covariance structure in HonestDiD, use analytical SEs (`n_bootstrap=0`).
- **Note (deviation from R):** When CallawaySantAnna results are passed to HonestDiD, `base_period != "universal"` emits a warning but does not error. R's `honest_did::honest_did.AGGTEobj` requires universal base period. Our implementation warns because the varying-base pre-treatment coefficients use consecutive comparisons (not a common reference), which changes the parallel-trends restriction interpretation.

**Reference implementation(s):**
- R: `HonestDiD` package (Rambachan & Roth's official package)

**Requirements checklist:**
- [x] Δ^SD constrains second differences with δ_0 = 0 boundary handling
- [x] Δ^RM constrains first differences (not levels), union of polyhedra
- [x] Identified set LP pins δ_pre = β_pre (Equations 5-6)
- [x] Optimal FLCI for Δ^SD (convex optimization, folded normal quantile)
- [x] ARP hybrid framework for Δ^RM (vertex enumeration, truncated normal)
- [x] Sensitivity analysis over M/M̄ grid with breakdown value
- [x] M parameter must be ≥ 0
- [ ] ARP hybrid produces valid (non-degenerate) CIs for all test cases

---

## PreTrendsPower

**Primary source:** [Roth, J. (2022). Pretest with Caution: Event-Study Estimates after Testing for Parallel Trends. *American Economic Review: Insights*, 4(3), 305-322.](https://doi.org/10.1257/aeri.20210236)

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires specification of variance-covariance matrix of pre-treatment estimates
- Warns if pre-trends test has low power (uninformative)
- Different violation types have different power properties

*Estimator equation (as implemented):*

Pre-trends test statistic (Wald):
```
W = δ̂_pre' V̂_pre^{-1} δ̂_pre ~ χ²(k)
```

Power function:
```
Power(δ_true) = P(W > χ²_{α,k} | δ = δ_true)
```

Minimum detectable violation (MDV):
```
MDV(power=0.8) = min{|δ| : Power(δ) ≥ 0.8}
```

Violation types:
- **Linear**: δ_t = c × t (linear pre-trend)
- **Constant**: δ_t = c (level shift)
- **Last period**: δ_{-1} = c, others zero
- **Custom**: user-specified pattern

*Standard errors:*
- Power calculations are exact (no sampling variability)
- Uncertainty comes from estimated Σ

*Edge cases:*
- Perfect collinearity in pre-periods: test not well-defined
- Single pre-period: power calculation trivial
- Very high power: MDV approaches zero

**Reference implementation(s):**
- R: `pretrends` package (Roth's official package)

**Requirements checklist:**
- [ ] MDV = minimum detectable violation at target power level
- [ ] Violation types: linear, constant, last_period, custom all implemented
- [ ] Power curve plotting over violation magnitudes
- [ ] Integrates with HonestDiD for combined sensitivity analysis

---

## PowerAnalysis

**Primary source:**
- Bloom, H.S. (1995). Minimum Detectable Effects: A Simple Way to Report the Statistical Power of Experimental Designs. *Evaluation Review*, 19(5), 547-556. https://doi.org/10.1177/0193841X9501900504
- Burlig, F., Preonas, L., & Woerman, M. (2020). Panel Data and Experimental Design. *Journal of Development Economics*, 144, 102458.

**Key implementation requirements:**

*Assumption checks / warnings:*
- Requires specification of outcome variance and intraclass correlation
- Warns if power is very low (<0.5) or sample size insufficient
- Cluster randomization requires cluster-level parameters

*Estimator equation (as implemented):*

Minimum detectable effect (MDE):
```
MDE = (t_{α/2} + t_{1-κ}) × SE(τ̂)
```
where κ is target power (typically 0.8).

Standard error for DiD:
```
SE(τ̂) = σ × √(1/n_T + 1/n_C) × √(1 + ρ(m-1)) / √(1 - R²)
```
where:
- ρ = intraclass correlation
- m = cluster size
- R² = variance explained by covariates

Power function:
```
Power = Φ(|τ|/SE - z_{α/2})
```

Sample size for target power:
```
n = 2(t_{α/2} + t_{1-κ})² σ² / MDE²
```

*Standard errors:*
- Analytical formulas (no estimation uncertainty in power calculation)
- Simulation-based power accounts for finite-sample and model-specific factors

*Edge cases:*
- Very small effects: may require infeasibly large samples
- High ICC: dramatically reduces effective sample size
- Unequal allocation: optimal is often 50-50 but depends on costs
- **Note:** `data_generator_kwargs` keys that overlap with registry-managed simulation inputs (`treatment_effect`, `noise_sd`, `n_units`, `n_periods`, `treatment_fraction`, `treatment_period`, `n_pre`, `n_post`) are rejected with `ValueError` to prevent silent desync between the DGP and result metadata. `n_pre` and `n_post` are derived from `treatment_period` and `n_periods` in factor-model DGPs (SyntheticDiD, TROP); the 3-way intersection check naturally scopes the rejection to those estimators only. Use the corresponding `simulate_power()` parameters directly, or pass a custom `data_generator` to override the DGP entirely.
- **Note:** `simulate_sample_size()` rejects `n_per_cell` in `data_generator_kwargs` for `TripleDifference` because `n_per_cell` is derived from `n_units` (the search variable). A fixed override would freeze the effective sample size across bisection iterations, making the search degenerate. Use `simulate_power()` with a fixed `n_per_cell` override instead, or pass a custom `data_generator`.
- **Note:** The simulation-based power registry (`simulate_power`, `simulate_mde`, `simulate_sample_size`) uses a single-cohort staggered DGP by default. Estimators configured with `control_group="not_yet_treated"`, `clean_control="strict"`, or `anticipation>0` will receive a `UserWarning` because the default DGP does not match their identification strategy. Users must supply `data_generator_kwargs` (e.g., `cohort_periods=[2, 4]`, `never_treated_frac=0.0`) or a custom `data_generator` to match the estimator design.
- **Note:** The `TripleDifference` registry adapter uses `generate_ddd_data`, a fixed 2×2×2 factorial DGP (group × partition × time). The `n_periods`, `treatment_period`, and `treatment_fraction` parameters are ignored — DDD always simulates 2 periods with balanced groups. `n_units` is mapped to `n_per_cell = max(2, n_units // 8)` (effective total N = `n_per_cell × 8`), so non-multiples of 8 are rounded down and values below 16 are clamped to 16. A `UserWarning` is emitted when simulation inputs differ from the effective DDD design. When rounding occurs, all result objects (`SimulationPowerResults`, `SimulationMDEResults`, `SimulationSampleSizeResults`) set `effective_n_units` to the actual sample size used; it is `None` when no rounding occurred. `simulate_sample_size()` snaps bisection candidates to multiples of 8 so that `required_n` is always a realizable DDD sample size. Passing `n_per_cell` in `data_generator_kwargs` suppresses the effective-N rounding warning but not warnings for ignored parameters (`n_periods`, `treatment_period`, `treatment_fraction`).
- **Note:** The analytical power methods (`PowerAnalysis.power/mde/sample_size` and the `compute_power/compute_mde/compute_sample_size` convenience functions) accept a `deff` parameter (survey design effect, default 1.0). This inflates variance multiplicatively: `Var(ATT) *= deff`, and inflates required sample size: `n_total *= deff`. The `deff` parameter is **not redundant** with `rho` (intra-cluster correlation): `rho` models within-unit serial correlation in panel data via the Moulton factor `1 + (T-1)*rho`, while `deff` models the survey design effect from stratified multi-stage sampling (clustering + unequal weighting). A survey panel study may need both. Values `deff > 0` are accepted; `deff < 1.0` (net variance reduction, e.g., from stratification gain) emits a warning.
- **Note:** The simulation-based power functions (`simulate_power/simulate_mde/simulate_sample_size`) accept a `survey_config` parameter (`SurveyPowerConfig` dataclass). When set, the simulation loop uses `generate_survey_did_data` instead of the default registry DGP, and automatically injects `SurveyDesign(weights="weight", strata="stratum", psu="psu", fpc="fpc")` into the estimator's `fit()` call. Supported estimators: DifferenceInDifferences, TwoWayFixedEffects, MultiPeriodDiD, CallawaySantAnna, SunAbraham, ImputationDiD, TwoStageDiD, StackedDiD, EfficientDiD. Unsupported (raises `ValueError`): TROP, SyntheticDiD, TripleDifference (generate_survey_did_data produces staggered cohort data incompatible with factor-model/DDD DGPs). `survey_config` and `data_generator` are mutually exclusive. `data_generator_kwargs` may not contain keys managed by `SurveyPowerConfig` (n_strata, psu_per_stratum, etc.) but may contain passthrough DGP params (unit_fe_sd, add_covariates, strata_sizes). Repeated cross-section survey power (`panel=False`) is only supported for `CallawaySantAnna(panel=False)` with a matching `data_generator_kwargs={"panel": False}`; both mismatch directions are rejected. `estimator_kwargs` may not contain `survey_design` when `survey_config` is set (use `SurveyPowerConfig(survey_design=...)` instead). Estimator settings that require a multi-cohort DGP (`control_group="not_yet_treated"`, `clean_control="strict"`) are rejected because the survey DGP uses a single cohort; use the custom `data_generator` path for these configurations. `simulate_sample_size` raises the bisection floor to `n_strata * psu_per_stratum * 2` to ensure viable survey structure and rejects `strata_sizes` in `data_generator_kwargs` (it depends on `n_units` which varies during bisection).

**Reference implementation(s):**
- R: `pwr` package (general), `DeclareDesign` (simulation-based)
- Stata: `power` command

**Requirements checklist:**
- [ ] MDE calculation given sample size and variance parameters
- [ ] Power calculation given effect size and sample size
- [ ] Sample size calculation given MDE and target power
- [ ] Simulation-based power for complex designs
- [ ] Cluster adjustment for clustered designs

---

# Visualization

## Event Study Plotting (`plot_event_study`)

**Reference Period Normalization**

Normalization only occurs when `reference_period` is **explicitly specified** by the user:

- **Explicit `reference_period=X`**: Normalizes effects (subtracts ref effect), sets ref SE to NaN
  - Point estimates: `effect_normalized = effect - effect_ref`
  - Reference period SE → NaN (it's now a constraint, not an estimate)
  - Other periods' SEs unchanged (uncertainty relative to the constraint)
  - CIs recomputed from normalized effects and original SEs

- **Auto-inferred reference** (from CallawaySantAnna results): Hollow marker styling only, no normalization
  - Original effects are plotted unchanged
  - Reference period shown with hollow marker for visual indication
  - All periods retain their original SEs and error bars

This design prevents unintended normalization when the reference period isn't a true
identifying constraint (e.g., CallawaySantAnna with `base_period="varying"` where different
cohorts use different comparison periods).

The explicit-only normalization follows the `fixest` (R) convention where the omitted/reference
category is an identifying constraint with no associated uncertainty. Auto-inferred references
follow the `did` (R) package convention which does not normalize and reports full inference.

**Rationale**: When normalizing to a reference period, we're treating that period as an
identifying constraint (effect ≡ 0 by definition). The variance of a constant is zero,
but since it's a constraint rather than an estimated quantity, we report NaN rather than 0.
Auto-inferred references may not represent true identifying constraints, so normalization
should be a deliberate user choice.

**Edge Cases:**
- If `reference_period` not in data: No normalization applied
- If reference effect is NaN: No normalization applied
- Reference period CI becomes (NaN, NaN) after normalization (explicit only)
- Reference period is plotted with hollow marker (both explicit and auto-inferred)
- Reference period error bars: removed for explicit, retained for auto-inferred

**Reference implementation(s):**
- R: `fixest::coefplot()` with reference category shown at 0 with no CI
- R: `did::ggdid()` does not normalize; shows full inference for all periods

---

# Cross-Reference: Standard Errors Summary

| Estimator | Default SE | Alternatives |
|-----------|-----------|--------------|
| DifferenceInDifferences | HC1 robust | Cluster-robust, wild bootstrap |
| MultiPeriodDiD | HC1 robust | Cluster-robust (via `cluster` param), wild bootstrap |
| TwoWayFixedEffects | Cluster at unit | Wild bootstrap |
| CallawaySantAnna | Analytical (influence fn) | Multiplier bootstrap |
| SunAbraham | Cluster-robust + delta method | Pairs bootstrap |
| ImputationDiD | Conservative clustered (Thm 3) | Multiplier bootstrap (library extension; percentile CIs and empirical p-values, consistent with CS/SA) |
| TwoStageDiD | GMM sandwich (Newey & McFadden 1994) | Multiplier bootstrap on GMM influence function |
| SyntheticDiD | Placebo variance (Alg 4) | Unit-level bootstrap (fixed weights) |
| TripleDifference | Influence function (all methods) | SE = std(IF) / sqrt(n) |
| StackedDiD | Cluster-robust (unit) | Cluster at unit × sub-experiment |
| TROP | Block bootstrap | — |
| BaconDecomposition | N/A (exact decomposition) | Individual 2×2 SEs |
| HonestDiD | Inherited from event study | FLCI, C-LF |
| PreTrendsPower | Exact (analytical) | - |
| PowerAnalysis | Exact (analytical) | Simulation-based |

---

# Cross-Reference: R Package Equivalents

| diff-diff Estimator | R Package | Function |
|---------------------|-----------|----------|
| DifferenceInDifferences | fixest | `feols(y ~ treat:post, ...)` |
| MultiPeriodDiD | fixest | `feols(y ~ i(time, treat, ref=ref) \| unit + time)` |
| TwoWayFixedEffects | fixest | `feols(y ~ treat \| unit + time, ...)` |
| CallawaySantAnna | did | `att_gt()` |
| SunAbraham | fixest | `sunab()` |
| ImputationDiD | didimputation | `did_imputation()` |
| TwoStageDiD | did2s | `did2s()` |
| ContinuousDiD | contdid | `cont_did()` |
| SyntheticDiD | synthdid | `synthdid_estimate()` |
| TripleDifference | triplediff | `ddd()` |
| StackedDiD | stacked-did-weights | `create_sub_exp()` + `compute_weights()` |
| TROP | - | (forthcoming) |
| BaconDecomposition | bacondecomp | `bacon()` |
| HonestDiD | HonestDiD | `createSensitivityResults()` |
| PreTrendsPower | pretrends | `pretrends()` |
| PowerAnalysis | pwr / DeclareDesign | `pwr.t.test()` / simulation |

---

## Survey Data Support

Survey-weighted estimation allows correct population-level inference from data
collected via complex survey designs (multi-stage sampling, stratification,
unequal selection probabilities).

### Weighted Estimation

- **Reference**: Lumley (2004) "Analysis of Complex Survey Samples", Journal of
  Statistical Software 9(8). Solon, Haider, & Wooldridge (2015) "What Are We
  Weighting For?" Journal of Human Resources 50(2).
- **WLS formula**: `beta_WLS = (X'WX)^{-1} X'Wy` where `W = diag(w_i)`
- **Implementation**: Equivalent transformation via `sqrt(w)` scaling, then
  standard OLS. Residuals back-transformed to original scale.
- **Weight types**: pweight (inverse selection probability), fweight
  (frequency/expansion), aweight (inverse variance/precision)
- **Note:** Weight normalization uses `sum(w) = n` convention (DRDID/Stata), not
  raw weights (R `survey`). Coefficients are identical; SEs differ by constant
  factor.

### Taylor Series Linearization (TSL) Variance

- **Reference**: Binder (1983) "On the Variances of Asymptotically Normal
  Estimators from Complex Surveys", International Statistical Review 51(3).
  Lumley (2004).
- **Formula**: `V_TSL = (X'WX)^{-1} [sum_h V_h] (X'WX)^{-1}` with stratified
  PSU-level scores
- **Relationship to sandwich estimator**: TSL is a generalization of the
  Huber-White sandwich estimator that accounts for stratification and finite
  population correction
- **Deviation from R:** R `survey` defaults `lonely_psu` to "fail"; we default
  to "remove" with warning, matching common applied practice
- **Edge case**: Singleton strata (one PSU per stratum) — handled via
  `lonely_psu` parameter ("remove", "certainty", or "adjust")
- **Note:** For unstratified designs with a single PSU, all `lonely_psu` modes
  produce NaN variance. The "adjust" mode cannot center against a global mean
  when there is only one stratum (the single PSU is the entire sample).
- **Note:** Weights-only designs (no explicit PSU or strata) use implicit
  per-observation PSUs for the TSL meat computation, consistent with the
  stratified-no-PSU path. The adjustment factor is `n/(n-1)` (not HC1's
  `n/(n-k)`).

### Weight Type Effects on Inference

- **Note:** aweights use unweighted meat in the sandwich estimator (no `w` in
  `u^2` term). This matches Stata convention. Rationale: aweights model known
  heteroskedasticity; after WLS transformation, errors are approximately
  homoskedastic.
- **Note:** fweights affect degrees of freedom (`df = sum(w) - k`, not
  `n - k`). This matches Stata convention for frequency-expanded data.
- **Note:** pweight HC1 meat uses score outer products (Σ s_i s_i' where
  s_i = w_i x_i u_i), giving w² in the meat. fweight HC1 meat uses
  X'diag(w u²)X (one power of w), matching frequency-expanded HC1.
- **Note:** fweights must be non-negative integers; fractional values are
  rejected by `_validate_weights()`. All-zero vectors rejected at solver
  level. This matches Stata's convention.

### Absorbed Fixed Effects with Survey Weights

- **Note:** When `absorb` is used with a single variable in DiD/MultiPeriodDiD,
  all regressors (treatment, time, interactions, covariates) are within-transformed
  alongside the outcome per the FWL theorem. Regressors collinear with
  the absorbed FE (e.g., treatment after absorbing unit FE) are dropped
  via rank-deficiency handling. Multiple absorbed variables with survey weights
  are rejected (single-pass sequential demeaning is not the correct weighted
  FWL projection for N > 1 dimensions; iterative alternating projections are
  needed but not yet implemented).

### Survey Degrees of Freedom

- **Reference**: Korn & Graubard (1990) "Simultaneous Testing of Regression
  Coefficients with Complex Survey Data", JASA 85(409).
- **Formula**: `df = n_PSU - n_strata` (replaces `n - k` for t-distribution
  inference)
- **Deviation from R:** Some software uses Satterthwaite-type df approximation;
  we use the simpler and more common `n_PSU - n_strata` convention.
- **Note:** When no explicit PSU is specified (weights-only or stratified-no-PSU
  designs), each observation is treated as its own PSU for df purposes. Survey df
  becomes `n_obs - n_strata` (or `n_obs - 1` when unstratified).
- **Note:** When survey_design specifies weights only (no PSU) and cluster=
  is specified, cluster IDs are injected as effective PSUs for Taylor Series
  Linearization variance estimation, matching the R `survey` package
  convention that clusters are the primary sampling units.

### Survey Aggregation (`aggregate_survey`)

Aggregation of individual-level survey microdata to geographic-period cells with
design-based precision estimates, for use as a pre-processing step before panel
DiD estimation on repeated cross-section survey data.

- **Reference**: Lumley (2004) "Analysis of Complex Survey Samples", Journal of
  Statistical Software 9(8), Section 3.4 (domain estimation).
- **Cell mean**: Design-weighted mean `ȳ_g = Σ w_i y_i / Σ w_i` for each cell g
  defined by grouping columns (e.g., state × year).
- **Cell variance**: Each cell is treated as a subpopulation/domain of the full
  survey design (consistent with `SurveyDesign.subpopulation()` and the
  Subpopulation Analysis section below). The influence function
  `ψ_i = w_i (y_i - ȳ_g) / Σ w_j` is zero-padded outside the cell, preserving
  full strata/PSU structure for variance estimation via `compute_survey_if_variance()`
  (TSL) or `compute_replicate_if_variance()` (replicate designs).
- **Precision weight**: `1 / V(ȳ_g)` used as inverse-variance weight (aweight)
  in second-stage DiD estimation.
- **Note:** SRS fallback when design-based variance is unidentifiable (e.g., all
  strata contribute zero variance) or when the cell has fewer than `min_n` valid
  observations. Formula: `V_SRS = Σ w_i(y_i - ȳ)² / (Σ w_j)² × n/(n-1)`.
  Cells using SRS fallback are flagged via `srs_fallback` column.
- **Edge case**: Zero-variance cells (all observations identical) set precision to
  NaN to avoid infinite weights in second-stage WLS.

### Survey-Aware Bootstrap (Phase 6)

Two strategies for bootstrap variance under complex survey designs:

**Multiplier Bootstrap at PSU Level** (CallawaySantAnna, ImputationDiD, TwoStageDiD,
ContinuousDiD, EfficientDiD):

- **Reference**: Standard Taylor linearization bootstrap (Shao 2003, "Impact of the
  Bootstrap on Sample Surveys", Statistical Science 18(2))
- **Formula**: Generate multiplier weights independently within strata at the PSU level.
  Scale by `sqrt(1 - f_h)` for FPC. Perturbation:
  `ATT_boot[b] = ATT + w_b^T @ psi_psu` where `psi_psu` are PSU-aggregated IF sums.
- **Note:** When no strata/PSU/FPC, degenerates to standard unit-level multiplier bootstrap.

**Rao-Wu Rescaled Bootstrap** (SunAbraham, SyntheticDiD, TROP):

- **Reference**: Rao & Wu (1988) "Resampling Inference with Complex Survey Data",
  JASA 83(401); Rao, Wu & Yue (1992) "Some Recent Work on Resampling Methods for
  Complex Surveys", Survey Methodology 18(2), Section 3.
- **Formula**: Within each stratum *h* with *n_h* PSUs, draw `m_h` PSUs with replacement.
  Without FPC: `m_h = n_h - 1`. With FPC: `m_h = max(1, round((1 - f_h) * (n_h - 1)))`.
  Rescaled weight: `w*_i = w_i * (n_h / m_h) * r_hi` where `r_hi` = count of PSU *i* drawn.
- **Note:** FPC enters through the resample size `m_h`, not as a post-hoc scaling factor.
  When `f_h >= 1` (census stratum), observations keep original weights (zero variance).
- **Note:** Bootstrap paths support all three `lonely_psu` modes: `"remove"`, `"certainty"`,
  and `"adjust"`. For `"adjust"`, singleton PSUs from different strata are pooled into a
  combined pseudo-stratum and weights are generated for the pooled group. This is the
  bootstrap analogue of the TSL "adjust" behavior (centering around the global mean).
  Applies to both multiplier bootstrap (CallawaySantAnna, ImputationDiD, TwoStageDiD,
  ContinuousDiD, EfficientDiD) and Rao-Wu bootstrap (SunAbraham, SyntheticDiD, TROP).
  FPC scaling is skipped for pooled singletons (conservative). When only one singleton
  stratum exists total, pooling is not possible — the singleton contributes zero bootstrap
  variance (same as `remove`), with a `UserWarning` emitted. This is a library-specific
  documented fallback (R's analytical `adjust` uses grand-mean centering, but the bootstrap
  analogue for a single singleton is not defined in the literature). Reference: Rust & Rao (1996).
- **Deviation from R:** For the no-FPC case (`m_h = n_h - 1`), this matches R
  `survey::as.svrepdesign(type="subbootstrap")`. The FPC-adjusted resample size
  `m_h = round((1-f_h)*(n_h-1))` follows Rao, Wu & Yue (1992) Section 3.

**CallawaySantAnna Design-Based Aggregated SEs**:

- **Formula**: `V_design = sum_h (1-f_h) * (n_h/(n_h-1)) * sum_j (psi_hj - psi_h_bar)^2`
  where `psi_hj = sum_{i in PSU j} psi_i` and `psi_i` is the combined IF (standard + WIF).
- **Note:** Per-(g,t) cell SEs use the simpler IF-based formula `sqrt(sum(psi^2))` which
  already incorporates survey weights. Only aggregated SEs (overall, event study, group)
  use the full design-based variance.

**TROP Cross-Classified Strata**:

- **Note (deviation from R):** When survey strata and treatment groups both exist, TROP
  creates pseudo-strata as `(survey_stratum x treatment_group)` for Rao-Wu resampling.
  This preserves both survey variance structure and treatment ratio. Survey df computed
  from pseudo-strata structure.
- **Note:** When `survey_design.strata` is None but PSU/FPC trigger full-design bootstrap,
  TROP uses treatment group (treated vs control) as pseudo-strata for Rao-Wu resampling
  to preserve treatment ratio. FPC is applied within these pseudo-strata. This matches
  TROP's existing treatment-stratified resampling pattern.
- **Note (deviation from block bootstrap):** In Rao-Wu survey bootstrap, per-observation
  treatment effects tau_{it} are deterministic given (Y, D, lambda) because survey weights
  do not enter the kernel-weighted matrix completion. The Rao-Wu path therefore precomputes
  tau values once and only varies the ATT aggregation weights across draws. This is
  mathematically equivalent to refitting per draw and avoids redundant computation.

### Replicate Weight Variance (Phase 6)

Alternative to TSL: re-run WLS for each replicate weight column and compute
variance from the distribution of replicate estimates.

- **Reference**: Wolter (2007) "Introduction to Variance Estimation", 2nd ed.
  Rao & Wu (1988).
- **Supported methods**: BRR, Fay's BRR, JK1, JKn, SDR
- **Formulas**:
  - BRR: `V = (1/R) * sum_r (theta_r - theta)^2`
  - SDR: `V = (4/R) * sum_r (theta_r - theta)^2` (Fay & Train 1995)
  - Fay: `V = 1/(R*(1-rho)^2) * sum_r (theta_r - theta)^2`
  - JK1: `V = (R-1)/R * sum_r (theta_r - theta)^2`
  - JKn: `V = sum_h ((n_h-1)/n_h) * sum_{r in h} (theta_r - theta)^2`
- **Note:** SDR (Successive Difference Replication) uses variance factor 4/R, following Fay & Train (1995). Used by ACS PUMS (80 replicate columns). Treated identically to BRR for scaling purposes — no fay_rho, no replicate_strata, custom scale/rscales ignored.
- **IF-based replicate variance**: For influence-function estimators (CS
  aggregation, ContinuousDiD, EfficientDiD, TripleDifference), replicate
  contrasts are formed via weight-ratio rescaling:
  `theta_r = sum((w_r/w_full) * psi)` when `combined_weights=True`,
  `theta_r = sum(w_r * psi)` when `combined_weights=False`.
- **Survey df**: QR-rank of the analysis-weight matrix minus 1,
  matching R's `survey::degf()` which uses `qr(..., tol=1e-5)$rank`.
  For `combined_weights=True` (default), analysis weights are the raw
  replicate columns. For `combined_weights=False`, analysis weights are
  `replicate_weights * full_sample_weights`. Returns `None` (undefined)
  when rank <= 1, yielding NaN inference. Replaces `n_PSU - n_strata`.
- **Mutual exclusion**: Replicate weights cannot be combined with
  strata/psu/fpc (the replicates encode design structure implicitly)
- **Design parameters** (matching R `svrepdesign()`):
  - `combined_weights` (default True): replicate columns include full-sample
    weight. If False, replicate columns are perturbation factors multiplied
    by full-sample weight before WLS.
  - `replicate_scale`: overall variance multiplier, applied multiplicatively
    with `replicate_rscales` when both are provided (`scale * rscales`)
  - `replicate_rscales`: per-replicate scaling factors (vector of length R).
    BRR and Fay ignore custom `replicate_scale`/`replicate_rscales` with a
    warning (fixed scaling by design); JK1/JKn allow overrides.
  - `mse` (default False, matching R's `survey::svrepdesign()`): if True,
    center variance on full-sample estimate; if False, center on mean of
    replicate estimates. When `replicate_rscales` contains zero entries
    and `mse=False`, centering excludes zero-scaled replicates, matching
    R's `survey::svrVar()` convention.
- **Note:** Replicate columns are NOT normalized — raw values are preserved
  to maintain correct weight ratios in the IF path.
- **Note:** JKn requires explicit `replicate_strata` (per-replicate stratum
  assignment). Auto-derivation from weight patterns is not supported.
- **Note:** Invalid replicate solves (singular/degenerate) are dropped with
  a warning. Variance is computed from valid replicates only. Fewer than 2
  valid replicates returns NaN variance. The variance scaling factor
  (e.g., `1/R` for BRR, `(R-1)/R` for JK1) uses the original design's `R`,
  not the valid count — matching R's `survey` package convention where the
  design structure is fixed and dropped replicates contribute zero to the
  sum without changing the scale. Survey df uses `n_valid - 1` for
  t-based inference.
- **Note:** Replicate-weight support matrix (12 of 15 public estimators):
  - **Supported**: CallawaySantAnna (reg/ipw/dr with or without covariates,
    no bootstrap; IF-based replicate variance is covariate-agnostic),
    ContinuousDiD (no bootstrap), EfficientDiD (no bootstrap),
    TripleDifference (all methods), StaggeredTripleDifference (IF-based),
    DifferenceInDifferences (no-absorb via LinearRegression dispatch,
    absorb via estimator-level refit), MultiPeriodDiD (no-absorb via
    `compute_replicate_vcov`, absorb via estimator-level refit),
    TwoWayFixedEffects (estimator-level refit with within-transformation),
    SunAbraham (estimator-level refit, replaces `vcov_cohort`),
    StackedDiD (estimator-level refit with Q-weight composition),
    ImputationDiD (two-stage refit), TwoStageDiD (two-stage refit)
  - **Rejected with NotImplementedError**: SyntheticDiD, TROP
    (bootstrap-based variance), BaconDecomposition (diagnostic only)
  - Estimators with replicate support reject replicate + bootstrap
    (replicate weights provide analytical variance)
- **Note:** When invalid replicates are dropped in `compute_replicate_vcov`
  (OLS path), `n_valid` is returned and used for `df_survey = n_valid - 1`
  in `LinearRegression.fit()`. For IF-based replicate paths, replicates
  essentially never fail (weighted sums cannot be singular), so `n_valid`
  equals `R` in practice and df propagation is not needed.

### DEFF Diagnostics (Phase 6)

Per-coefficient design effect comparing survey variance to SRS variance.

- **Reference**: Kish (1965) "Survey Sampling", Wiley. Chapter 8.
- **Formula**: `DEFF_k = Var_survey(beta_k) / Var_SRS(beta_k)` where
  SRS baseline uses HC1 sandwich ignoring design structure
- **Effective n**: `n_eff_k = n / DEFF_k`
- **Display**: Existing weight-based DEFF labeled "Kish DEFF (weights)";
  per-coefficient DEFF available via `compute_deff_diagnostics()` or
  `LinearRegression.compute_deff()` post-fit
- **Note:** Opt-in computation — not run automatically. Users call standalone
  function or post-fit method when diagnostics are needed.

### Subpopulation Analysis (Phase 6)

Domain estimation preserving full design structure.

- **Reference**: Lumley (2004) Section 3.4. Stata `svy: subpop`.
- **Method**: `SurveyDesign.subpopulation(data, mask)` zeros out weights for
  excluded observations while retaining strata/PSU layout for correct
  variance estimation
- **Note:** Unlike naive subsetting, subpopulation analysis preserves design
  information (PSU structure, strata counts) that would be lost by dropping
  observations. This is the methodologically correct approach for domain
  estimation under complex survey designs.
- **Note:** Weight validation relaxed from "strictly positive" to
  "non-negative" to support zero-weight observations. Negative weights
  still rejected. All-zero weight vectors rejected at solver level.
- **Note:** Survey design df (`n_PSU - n_strata`) uses the full design
  structure (including zero-weight rows), ensuring variance estimation
  accounts for all strata and PSUs. The generic HC1/classical inference
  paths use positive-weight count for df adjustments, ensuring zero-weight
  padding is inference-invariant outside the survey vcov path. DEFF
  effective-n also uses positive-weight count.
- **Deviation from R:** `subpopulation()` preserves all strata in df
  computation even when a stratum has no positive-weight observations,
  while R's `subset()` drops empty strata from `survey::degf()`. For
  example, subsetting a 3-stratum design to one stratum gives df=n-3
  in diff-diff vs df=n-1 in R. Both ATT and SE match; only df (and
  therefore t-based CI width) differs. The diff-diff approach is
  conservative (more strata → lower df → wider CI) and preserves the
  full design structure per Lumley (2004) Section 3.4.
- **Note:** For replicate-weight designs, `subpopulation()` zeros out both
  full-sample and replicate weight columns for excluded observations,
  preserving all replicate metadata.
- **Note:** Estimator-level replicate refits (TWFE, SunAbraham, DiD/MultiPeriodDiD
  with `absorb`) drop zero-weight observations before weighted demeaning to
  prevent division-by-zero in within-transformation group means.  This matches
  R's `survey::withReplicates()` convention where zero-weight units are excluded
  from per-replicate estimation.  Replicates that fail despite this (e.g.,
  rank-deficient after unit deletion) are counted as invalid and excluded from
  variance computation.
- **Note:** Defensive enhancement: ContinuousDiD and TripleDifference
  validate the positive-weight effective sample size before WLS cell fits.
  After `subpopulation()` zeroes weights, raw row counts may exceed the
  regression rank requirement while the weighted effective sample does not.
  Underidentified cells are skipped (ContinuousDiD) or fall back to
  weighted means (TripleDifference).

---

# Practitioner Guide

The 8-step workflow in `docs/llms-practitioner.txt` is adapted from Baker et al. (2025)
"Difference-in-Differences Designs: A Practitioner's Guide" (arXiv:2503.13323), not a
1:1 mapping of the paper's forward-engineering framework.

- **Note:** The diff-diff canonical numbering is: 1-Define, 2-Assumptions, 3-Test PT,
  4-Choose estimator, 5-Estimate, 6-Sensitivity, 7-Heterogeneity, 8-Robustness.
  Paper's numbering: 1-Define, 2-Assumptions, 3-Estimation method, 4-Uncertainty,
  5-Estimate, 6-Sensitivity, 7-Heterogeneity, 8-Keep learning.
- **Note:** Parallel trends testing is a separate Step 3 (paper embeds it in Step 2),
  to ensure AI agents execute it as a distinct action.
- **Note:** Sources of uncertainty (paper's Step 4) is folded into Step 5 (Estimate)
  with an explicit cluster-count check directive (>= 50 clusters for asymptotic SEs,
  otherwise wild bootstrap). The 50-cluster threshold is a diff-diff convention.
- **Note:** Step 8 is "Robustness & Reporting" (compare estimators, report with/without
  covariates). Paper's Step 8 is "Keep learning." The mandatory with/without covariate
  comparison is a diff-diff convention.

### Survey DGP (`generate_survey_did_data`)

- **Note:** The `icc` parameter calibrates `psu_re_sd` using the full variance
  decomposition `Var(Y) = sigma²_psu * (1 + psu_period_factor²) + sigma²_unit +
  sigma²_noise + sigma²_cov`. When `add_covariates=True`, covariate variance
  `sigma²_cov = beta1² * Var(x1) + beta2² * Var(x2)` is included, where
  `(beta1, beta2)` defaults to `(0.5, 0.3)` but is configurable via
  `covariate_effects`.
- **Note:** When `informative_sampling=True` and `add_covariates=True`, covariate
  contributions are included in the Y(0) ranking used for weight assignment.
  Covariates are pre-drawn before the ranking step (panel: once before the loop;
  cross-section: each period) and reused in the outcome generation.

---

# Version History

- **v1.3** (2026-03-26): Added Replicate Weight Variance, DEFF Diagnostics,
  and Subpopulation Analysis sections (Phase 6 completion)
- **v1.2** (2026-03-24): Added Survey-Aware Bootstrap section (Phase 6)
- **v1.1** (2026-03-20): Added Survey Data Support section
- **v1.0** (2025-01-19): Initial registry with 12 estimators
