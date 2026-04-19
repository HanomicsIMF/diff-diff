# BR / DR canonical-dataset validation

Output of ``docs/validation/validate_br_dr_canonical.py``. Each section runs BusinessReport (and its auto-constructed DiagnosticReport) on a canonical DiD dataset and dumps summary + full_report + selected to_dict blocks. The purpose is to compare BR's prose output against published canonical interpretations and record divergences in ``br_dr_canonical_findings.md``.

This file is regenerable; do not hand-edit.

Datasets covered: Card-Krueger (1994), mpdta (Callaway-Sant'Anna 2021 benchmark), Castle Doctrine (Cheng-Hoekstra 2013, both CS and SA).

---

## Card & Krueger (1994): NJ/PA minimum wage
Data: NJ (treated, min wage $4.25 -> $5.05 on 1992-04-01) vs PA (control, $4.25 throughout). Outcome: full-time equivalent employment. N=310 stores.

Canonical interpretation: no significant disemployment effect of the minimum-wage increase; published ATT ~ +0.59 FTE (positive direction). The famous finding was that the CI included zero.

### BusinessReport.summary()
```
Question: Did the NJ minimum-wage increase reduce fast-food employment? The NJ minimum-wage increase lifted FTE employment by 1.47 FTE (95% CI: -2.32 FTE to 5.27 FTE). Statistically, the confidence interval includes zero; the data are consistent with no effect. Sample: 620 observations (462 treated, 158 control).
```
### BusinessReport.full_report()
```markdown
# Business Report: FTE employment

**Question**: Did the NJ minimum-wage increase reduce fast-food employment?

**Estimator**: `DiDResults`

## Headline

The NJ minimum-wage increase lifted FTE employment by 1.47 FTE (95% CI: -2.32 FTE to 5.27 FTE).

Statistically, the confidence interval includes zero; the data are consistent with no effect.

## Identifying Assumption

Identification relies on the standard DiD parallel-trends assumption plus no anticipation of treatment by either group.

## Pre-Trends

- Pre-trends not computed: auto_diagnostics=False

## Sensitivity (HonestDiD)

- Sensitivity not computed: auto_diagnostics=False

## Sample

- Observations: 620
- Treated: 462
- Control: 158

## References

- Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends. Review of Economic Studies.
- Baker, A. C., Callaway, B., Cunningham, S., Goodman-Bacon, A., & Sant'Anna, P. H. C. (2025). Difference-in-Differences Designs: A Practitioner's Guide.


## Technical Appendix

```
======================================================================
             Difference-in-Differences Estimation Results             
======================================================================

Observations:                    620
Treated:                         462
Control:                         158
R-squared:                    0.0036
Variance:                            HC1 heteroskedasticity-robust

----------------------------------------------------------------------
Parameter           Estimate    Std. Err.     t-stat      P>|t|      
----------------------------------------------------------------------
ATT                   1.4718       1.9320      0.762     0.4465      
----------------------------------------------------------------------

95% Confidence Interval: [-2.3224, 5.2660]
CV (SE/|ATT|):                1.3127

Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1
======================================================================
```
```
### BusinessReport.to_dict() - headline + assumption + caveats
```json
{
  "effect": 1.4718176338428604,
  "se": 1.9320362599811534,
  "ci_lower": -2.322358689575049,
  "ci_upper": 5.26599395726077,
  "alpha_was_honored": true,
  "alpha_override_caveat": null,
  "ci_level": 95,
  "p_value": 0.4464732839915416,
  "is_significant": false,
  "near_significance_threshold": false,
  "unit": "FTE",
  "unit_kind": "unknown",
  "sign": "positive",
  "breakdown_M": null
}
```
```json
{
  "parallel_trends_variant": "unconditional",
  "no_anticipation": true,
  "description": "Identification relies on the standard DiD parallel-trends assumption plus no anticipation of treatment by either group."
}
```
```json
[]
```

---
## Callaway-Sant'Anna benchmark (mpdta)
Data: simulated county-level panel from R `did` package (Callaway & Sant'Anna 2021), 2003-2007, staggered minimum-wage increases. Outcome: log employment (`lemp`).

Canonical interpretation: CS aggregate ATT ~ -0.04 to -0.05 (log points) on treated counties; group-specific ATT(g,t) negative across cohorts. See CS (2021) Figures 1-2.

### BusinessReport.summary()
```
Question: Did minimum-wage increases reduce county employment? The state-level minimum wage increase reduced Log employment by 0.0214 log-points (95% CI: -0.0251 log-points to -0.0178 log-points). Statistically, the direction of the effect is strongly supported by the data. Pre-treatment event-study coefficients do not reject parallel trends; the test is moderately informative. See the sensitivity analysis below for bounded-violation guarantees. HonestDiD: the result remains significant under parallel-trends violations up to 1.3x the observed pre-period variation. Sample: 2,500 observations (309 treated, 191 control). Caveat: Goodman-Bacon decomposition places 32% of TWFE weight on 'forbidden' later-vs-earlier comparisons. A TWFE benchmark on this rollout would be materially biased under heterogeneous effects; the displayed estimator is already heterogeneity-robust, so this is a statement about the rollout design (avoid reporting TWFE alongside this fit), not about the current result's validity.
```
### BusinessReport.full_report()
```markdown
# Business Report: Log employment

**Question**: Did minimum-wage increases reduce county employment?

**Estimator**: `CallawaySantAnnaResults`

## Headline

The state-level minimum wage increase reduced Log employment by 0.0214 log-points (95% CI: -0.0251 log-points to -0.0178 log-points).

Statistically, the direction of the effect is strongly supported by the data.

## Identifying Assumption

Identification relies on parallel trends across treatment cohorts and time periods (group-time ATT), plus no anticipation.

## Pre-Trends

- Verdict: `no_detected_violation` (joint p = 0.482)
- Power tier: `moderately_powered`
- Minimum detectable violation (MDV): 0.0105
- MDV / |ATT|: 0.49

## Sensitivity (HonestDiD)

- Method: `relative_magnitude`
- Breakdown M: 1.28
- Conclusion: `robust_to_M_1.28`

## Sample

- Observations: 2,500
- Treated: 309
- Control: 191

## Heterogeneity

- Source: `event_study_effects_post`
- N effects: 4
- Range: -0.0293 to -0.00305
- CV: 0.668
- Sign consistent: True

## Caveats

- **WARNING** — Goodman-Bacon decomposition places 32% of TWFE weight on 'forbidden' later-vs-earlier comparisons. A TWFE benchmark on this rollout would be materially biased under heterogeneous effects; the displayed estimator is already heterogeneity-robust, so this is a statement about the rollout design (avoid reporting TWFE alongside this fit), not about the current result's validity.
- **INFO** — The effect is reported in log-points as estimated; BusinessReport does not arithmetically translate log-points to percent or level changes. For small effects, log-points approximate percentage changes.

## Next Steps

- Define target parameter
  - _why_: State explicitly what causal effect you are estimating (ATT, ATT(g,t), weighted/unweighted) and what policy question it answers.
- State identification assumptions
  - _why_: Name the parallel trends variant you are invoking (unconditional, conditional, PT-GT-NYT, etc.), the no-anticipation assumption, and any overlap conditions.
- Compare with alternative estimators (SA, BJS, or Gardner)
  - _why_: Agreement across estimators with different assumptions strengthens conclusions. Disagreement reveals sensitivity.
- Report with and without covariates
  - _why_: Shows whether results are sensitive to covariate conditioning. Large shifts suggest covariates are driving identification.

## References

- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-Differences with multiple time periods. Journal of Econometrics.
- Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends. Review of Economic Studies.
- Baker, A. C., Callaway, B., Cunningham, S., Goodman-Bacon, A., & Sant'Anna, P. H. C. (2025). Difference-in-Differences Designs: A Practitioner's Guide.


## Technical Appendix

```
=====================================================================================
            Callaway-Sant'Anna Staggered Difference-in-Differences Results           
=====================================================================================

Total observations:                  2500
Treated units:                        309
Never-treated units:                  191
Treatment cohorts:                      3
Time periods:                           5
Control group:                 never_treated
Base period:                    universal

-------------------------------------------------------------------------------------
                   Overall Average Treatment Effect on the Treated                   
-------------------------------------------------------------------------------------
Parameter           Estimate    Std. Err.     t-stat      P>|t|   Sig.
-------------------------------------------------------------------------------------
ATT                  -0.0214       0.0019    -11.397     0.0000    ***
-------------------------------------------------------------------------------------

95% Confidence Interval: [-0.0251, -0.0178]
CV (SE/|ATT|):                0.0877

-------------------------------------------------------------------------------------
                            Event Study (Dynamic) Effects                            
-------------------------------------------------------------------------------------
Rel. Period         Estimate    Std. Err.     t-stat      P>|t|   Sig.
-------------------------------------------------------------------------------------
-4                    0.0023       0.0036      0.627     0.5309       
-3                   -0.0019       0.0023     -0.810     0.4179       
-2                   -0.0020       0.0022     -0.875     0.3818       
-1                    0.0000          nan        nan        nan       
0                    -0.0293       0.0019    -15.137     0.0000    ***
1                    -0.0235       0.0023    -10.111     0.0000    ***
2                    -0.0134       0.0031     -4.373     0.0000    ***
3                    -0.0031       0.0035     -0.884     0.3767       
-------------------------------------------------------------------------------------

Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1
=====================================================================================
```
```
### BusinessReport.to_dict() - headline + assumption + caveats
```json
{
  "effect": -0.021448663176265446,
  "se": 0.0018820025192546833,
  "ci_lower": -0.025137320332818274,
  "ci_upper": -0.01776000601971262,
  "alpha_was_honored": true,
  "alpha_override_caveat": null,
  "ci_level": 95,
  "p_value": 4.341504320370796e-30,
  "is_significant": true,
  "near_significance_threshold": false,
  "unit": "log_points",
  "unit_kind": "log_points",
  "sign": "negative",
  "breakdown_M": 1.2776496410369873
}
```
```json
{
  "parallel_trends_variant": "conditional_or_group_time",
  "no_anticipation": true,
  "description": "Identification relies on parallel trends across treatment cohorts and time periods (group-time ATT), plus no anticipation."
}
```
```json
{
  "status": "computed",
  "method": "joint_wald_event_study",
  "joint_p_value": 0.4816505473216015,
  "verdict": "no_detected_violation",
  "n_pre_periods": 3,
  "n_dropped_undefined": null,
  "reason": null,
  "df_denom": null,
  "power_status": "ran",
  "power_reason": null,
  "power_tier": "moderately_powered",
  "mdv": 0.010472079171705551,
  "mdv_share_of_att": 0.48823924762330606,
  "power_covariance_source": "diag_fallback_available_full_vcov_unused"
}
```
```json
{
  "status": "computed",
  "method": "relative_magnitude",
  "breakdown_M": 1.2776496410369873,
  "conclusion": "robust_to_M_1.28",
  "grid": [
    {
      "M": 0.5,
      "ci_lower": -0.026608074507223883,
      "ci_upper": -0.008013136465533054,
      "bound_lower": -0.022462755203290868,
      "bound_upper": -0.01215845576946607,
      "robust_to_zero": true
    },
    {
      "M": 1.0,
      "ci_lower": -0.03176022422413628,
      "ci_upper": -0.0028609867486206544,
      "bound_lower": -0.027614904920203267,
      "bound_upper": -0.00700630605255367,
      "robust_to_zero": true
    },
    {
      "M": 1.5,
      "ci_lower": -0.03691237394104868,
      "ci_upper": 0.002291162968291743,
      "bound_lower": -0.03276705463711566,
      "bound_upper": -0.0018541563356412726,
      "robust_to_zero": false
    },
    {
      "M": 2.0,
      "ci_lower": -0.04206452365796108,
      "ci_upper": 0.007443312685204144,
      "bound_lower": -0.03791920435402807,
      "bound_upper": 0.0032979933812711283,
      "robust_to_zero": false
    }
  ]
}
```
```json
[
  {
    "severity": "warning",
    "topic": "bacon_contamination",
    "message": "Goodman-Bacon decomposition places 32% of TWFE weight on 'forbidden' later-vs-earlier comparisons. A TWFE benchmark on this rollout would be materially biased under heterogeneous effects; the displayed estimator is already heterogeneity-robust, so this is a statement about the rollout design (avoid reporting TWFE alongside this fit), not about the current result's validity."
  },
  {
    "severity": "info",
    "topic": "unit_policy",
    "message": "The effect is reported in log-points as estimated; BusinessReport does not arithmetically translate log-points to percent or level changes. For small effects, log-points approximate percentage changes."
  }
]
```

---
## Cheng & Hoekstra (2013): Castle Doctrine laws
Data: state-year panel, staggered Castle Doctrine law adoption 2005-2009. Outcome: homicide rate per 100k population.

Canonical interpretation: Cheng & Hoekstra (2013) found ~8% increase in homicide rates in states that adopted Castle Doctrine (no deterrent effect; if anything, an escalation).

### BusinessReport.summary()
```
Question: Did Castle Doctrine law adoption change state homicide rates? Castle Doctrine law adoption worsened Homicide rate (per 100k) by 0.561 per 100k population (95% CI: 0.323 per 100k population to 0.799 per 100k population). Statistically, the direction of the effect is strongly supported by the data. Pre-treatment event-study coefficients clearly reject parallel trends (joint p = 0.00347); the headline should be treated as tentative pending the sensitivity analysis below. HonestDiD: the result is fragile — the confidence interval includes zero even at the smallest parallel-trends violations on the sensitivity grid. Sample: 539 observations (22 treated, 27 control). Caveat: Goodman-Bacon decomposition places 32% of TWFE weight on 'forbidden' later-vs-earlier comparisons. A TWFE benchmark on this rollout would be materially biased under heterogeneous effects; the displayed estimator is already heterogeneity-robust, so this is a statement about the rollout design (avoid reporting TWFE alongside this fit), not about the current result's validity.
```
### BusinessReport.full_report()
```markdown
# Business Report: Homicide rate (per 100k)

**Question**: Did Castle Doctrine law adoption change state homicide rates?

**Estimator**: `CallawaySantAnnaResults`

## Headline

Castle Doctrine law adoption worsened Homicide rate (per 100k) by 0.561 per 100k population (95% CI: 0.323 per 100k population to 0.799 per 100k population).

Statistically, the direction of the effect is strongly supported by the data.

## Identifying Assumption

Identification relies on parallel trends across treatment cohorts and time periods (group-time ATT), plus no anticipation.

## Pre-Trends

- Verdict: `clear_violation` (joint p = 0.00347)
- Power tier: `underpowered`
- Minimum detectable violation (MDV): 0.732
- MDV / |ATT|: 1.3

## Sensitivity (HonestDiD)

- Method: `relative_magnitude`
- Breakdown M: 0
- Conclusion: `fragile`

## Sample

- Observations: 539
- Treated: 22
- Control: 27

## Heterogeneity

- Source: `event_study_effects_post`
- N effects: 6
- Range: 0.237 to 0.764
- CV: 0.348
- Sign consistent: True

## Caveats

- **WARNING** — Goodman-Bacon decomposition places 32% of TWFE weight on 'forbidden' later-vs-earlier comparisons. A TWFE benchmark on this rollout would be materially biased under heterogeneous effects; the displayed estimator is already heterogeneity-robust, so this is a statement about the rollout design (avoid reporting TWFE alongside this fit), not about the current result's validity.
- **WARNING** — HonestDiD breakdown value is 0: the result's confidence interval includes zero once parallel-trends violations reach less than half the observed pre-period variation. Treat the headline as tentative.

## Next Steps

- Define target parameter
  - _why_: State explicitly what causal effect you are estimating (ATT, ATT(g,t), weighted/unweighted) and what policy question it answers.
- State identification assumptions
  - _why_: Name the parallel trends variant you are invoking (unconditional, conditional, PT-GT-NYT, etc.), the no-anticipation assumption, and any overlap conditions.
- Compare with alternative estimators (SA, BJS, or Gardner)
  - _why_: Agreement across estimators with different assumptions strengthens conclusions. Disagreement reveals sensitivity.
- Report with and without covariates
  - _why_: Shows whether results are sensitive to covariate conditioning. Large shifts suggest covariates are driving identification.

## References

- Callaway, B., & Sant'Anna, P. H. C. (2021). Difference-in-Differences with multiple time periods. Journal of Econometrics.
- Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends. Review of Economic Studies.
- Baker, A. C., Callaway, B., Cunningham, S., Goodman-Bacon, A., & Sant'Anna, P. H. C. (2025). Difference-in-Differences Designs: A Practitioner's Guide.


## Technical Appendix

```
=====================================================================================
            Callaway-Sant'Anna Staggered Difference-in-Differences Results           
=====================================================================================

Total observations:                   539
Treated units:                         22
Never-treated units:                   27
Treatment cohorts:                      6
Time periods:                          11
Control group:                 never_treated
Base period:                    universal

-------------------------------------------------------------------------------------
                   Overall Average Treatment Effect on the Treated                   
-------------------------------------------------------------------------------------
Parameter           Estimate    Std. Err.     t-stat      P>|t|   Sig.
-------------------------------------------------------------------------------------
ATT                   0.5608       0.1216      4.613     0.0000    ***
-------------------------------------------------------------------------------------

95% Confidence Interval: [0.3225, 0.7991]
CV (SE/|ATT|):                0.2168

-------------------------------------------------------------------------------------
                            Event Study (Dynamic) Effects                            
-------------------------------------------------------------------------------------
Rel. Period         Estimate    Std. Err.     t-stat      P>|t|   Sig.
-------------------------------------------------------------------------------------
-10                  -0.3415       0.1462     -2.336     0.0195      *
-9                    0.3406       0.5526      0.616     0.5377       
-8                   -0.1465       0.1794     -0.816     0.4143       
-7                    0.1393       0.3426      0.406     0.6844       
-6                    0.2611       0.1574      1.659     0.0972      .
-5                   -0.0466       0.1215     -0.383     0.7015       
-4                    0.1224       0.1511      0.810     0.4180       
-3                    0.0783       0.1505      0.520     0.6030       
-2                    0.1541       0.1085      1.420     0.1555       
-1                    0.0000          nan        nan        nan       
0                     0.4453       0.1606      2.772     0.0056     **
1                     0.7074       0.1957      3.614     0.0003    ***
2                     0.7642       0.1590      4.807     0.0000    ***
3                     0.5525       0.1582      3.492     0.0005    ***
4                     0.2367       0.1789      1.323     0.1859       
5                     0.6463       0.1227      5.269     0.0000    ***
-------------------------------------------------------------------------------------

Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1
=====================================================================================
```
```
### BusinessReport.to_dict() - headline + assumption + caveats
```json
{
  "effect": 0.5608256172839505,
  "se": 0.12157293428086259,
  "ci_lower": 0.32254704459860495,
  "ci_upper": 0.7991041899692961,
  "alpha_was_honored": true,
  "alpha_override_caveat": null,
  "ci_level": 95,
  "p_value": 3.967463629059167e-06,
  "is_significant": true,
  "near_significance_threshold": false,
  "unit": "per 100k population",
  "unit_kind": "unknown",
  "sign": "positive",
  "breakdown_M": 0.0
}
```
```json
{
  "parallel_trends_variant": "conditional_or_group_time",
  "no_anticipation": true,
  "description": "Identification relies on parallel trends across treatment cohorts and time periods (group-time ATT), plus no anticipation."
}
```
```json
{
  "status": "computed",
  "method": "joint_wald_event_study",
  "joint_p_value": 0.003469090217576798,
  "verdict": "clear_violation",
  "n_pre_periods": 9,
  "n_dropped_undefined": null,
  "reason": null,
  "df_denom": null,
  "power_status": "ran",
  "power_reason": null,
  "power_tier": "underpowered",
  "mdv": 0.7318611799799601,
  "mdv_share_of_att": 1.3049710238350487,
  "power_covariance_source": "diag_fallback_available_full_vcov_unused"
}
```
```json
{
  "status": "computed",
  "method": "relative_magnitude",
  "breakdown_M": 0.0,
  "conclusion": "fragile",
  "grid": [
    {
      "M": 0.5,
      "ci_lower": -0.84457211437162,
      "ci_upper": 1.9620045961559538,
      "bound_lower": -0.6348485739226502,
      "bound_upper": 1.752281055706984,
      "robust_to_zero": false
    },
    {
      "M": 1.0,
      "ci_lower": -2.038136929186437,
      "ci_upper": 3.1555694109707706,
      "bound_lower": -1.8284133887374672,
      "bound_upper": 2.9458458705218007,
      "robust_to_zero": false
    },
    {
      "M": 1.5,
      "ci_lower": -3.231701744001254,
      "ci_upper": 4.349134225785587,
      "bound_lower": -3.021978203552284,
      "bound_upper": 4.1394106853366175,
      "robust_to_zero": false
    },
    {
      "M": 2.0,
      "ci_lower": -4.4252665588160705,
      "ci_upper": 5.542699040600405,
      "bound_lower": -4.215543018367101,
      "bound_upper": 5.332975500151435,
      "robust_to_zero": false
    }
  ]
}
```
```json
[
  {
    "severity": "warning",
    "topic": "bacon_contamination",
    "message": "Goodman-Bacon decomposition places 32% of TWFE weight on 'forbidden' later-vs-earlier comparisons. A TWFE benchmark on this rollout would be materially biased under heterogeneous effects; the displayed estimator is already heterogeneity-robust, so this is a statement about the rollout design (avoid reporting TWFE alongside this fit), not about the current result's validity."
  },
  {
    "severity": "warning",
    "topic": "sensitivity_fragility",
    "message": "HonestDiD breakdown value is 0: the result's confidence interval includes zero once parallel-trends violations reach less than half the observed pre-period variation. Treat the headline as tentative."
  }
]
```

---
## Castle Doctrine under Sun-Abraham (2021)
Same dataset and research question; different estimator. Testing BR/DR cross-estimator narrative consistency.

### BusinessReport.summary()
```
Question: Did Castle Doctrine law adoption change state homicide rates? Castle Doctrine law adoption worsened Homicide rate (per 100k) by 0.561 per 100k population (95% CI: 0.324 per 100k population to 0.798 per 100k population). Statistically, the direction of the effect is strongly supported by the data. Pre-treatment event-study coefficients clearly reject parallel trends (joint p = 0.0128); the headline should be treated as tentative. Sample: 539 observations (22 treated, 27 control). Caveat: Goodman-Bacon decomposition places 32% of TWFE weight on 'forbidden' later-vs-earlier comparisons. A TWFE benchmark on this rollout would be materially biased under heterogeneous effects; the displayed estimator is already heterogeneity-robust, so this is a statement about the rollout design (avoid reporting TWFE alongside this fit), not about the current result's validity.
```
### BusinessReport.full_report()
```markdown
# Business Report: Homicide rate (per 100k)

**Question**: Did Castle Doctrine law adoption change state homicide rates?

**Estimator**: `SunAbrahamResults`

## Headline

Castle Doctrine law adoption worsened Homicide rate (per 100k) by 0.561 per 100k population (95% CI: 0.324 per 100k population to 0.798 per 100k population).

Statistically, the direction of the effect is strongly supported by the data.

## Identifying Assumption

Identification relies on parallel trends across treatment cohorts and time periods (group-time ATT), plus no anticipation.

## Pre-Trends

- Verdict: `clear_violation` (joint p = 0.0128)
- Power tier: `moderately_powered`
- Minimum detectable violation (MDV): 0.551
- MDV / |ATT|: 0.98

## Sensitivity (HonestDiD)

- Sensitivity not computed: sensitivity is not applicable to SunAbrahamResults.

## Sample

- Observations: 539
- Treated: 22
- Control: 27

## Heterogeneity

- Source: `event_study_effects_post`
- N effects: 6
- Range: 0.237 to 0.764
- CV: 0.348
- Sign consistent: True

## Caveats

- **WARNING** — Goodman-Bacon decomposition places 32% of TWFE weight on 'forbidden' later-vs-earlier comparisons. A TWFE benchmark on this rollout would be materially biased under heterogeneous effects; the displayed estimator is already heterogeneity-robust, so this is a statement about the rollout design (avoid reporting TWFE alongside this fit), not about the current result's validity.

## Next Steps

- Define target parameter
  - _why_: State explicitly what causal effect you are estimating (ATT, ATT(g,t), weighted/unweighted) and what policy question it answers.
- State identification assumptions
  - _why_: Name the parallel trends variant you are invoking (unconditional, conditional, PT-GT-NYT, etc.), the no-anticipation assumption, and any overlap conditions.
- Specification-based falsification
  - _why_: Compare results across control group definitions (never_treated vs not_yet_treated) and anticipation settings to assess robustness.
- Compare with alternative estimators (CS, BJS, or Gardner)
  - _why_: Agreement across estimators with different assumptions strengthens conclusions. Disagreement reveals sensitivity.
- Report with and without covariates
  - _why_: Shows whether results are sensitive to covariate conditioning. Large shifts suggest covariates are driving identification.

## References

- Sun, L., & Abraham, S. (2021). Estimating dynamic treatment effects in event studies. Journal of Econometrics.
- Rambachan, A., & Roth, J. (2023). A More Credible Approach to Parallel Trends. Review of Economic Studies.
- Baker, A. C., Callaway, B., Cunningham, S., Goodman-Bacon, A., & Sant'Anna, P. H. C. (2025). Difference-in-Differences Designs: A Practitioner's Guide.


## Technical Appendix

```
=====================================================================================
                  Sun-Abraham Interaction-Weighted Estimator Results                 
=====================================================================================

Total observations:                   539
Treated units:                         22
Control units:                         27
Treatment cohorts:                      6
Time periods:                          11
Control group:                 never_treated

-------------------------------------------------------------------------------------
                   Overall Average Treatment Effect on the Treated                   
-------------------------------------------------------------------------------------
Parameter           Estimate    Std. Err.     t-stat      P>|t|   Sig.
-------------------------------------------------------------------------------------
ATT                   0.5608       0.1208      4.642     0.0000    ***
-------------------------------------------------------------------------------------

95% Confidence Interval: [0.3240, 0.7976]
CV (SE/|ATT|):                0.2154

-------------------------------------------------------------------------------------
                            Event Study (Dynamic) Effects                            
-------------------------------------------------------------------------------------
Rel. Period         Estimate    Std. Err.     t-stat      P>|t|   Sig.
-------------------------------------------------------------------------------------
-10                  -0.3415       0.1566     -2.181     0.0292      *
-9                    0.3406       0.1067      3.191     0.0014     **
-8                   -0.1465       0.1379     -1.062     0.2882       
-7                    0.1393       0.3326      0.419     0.6754       
-6                    0.2611       0.1646      1.586     0.1128       
-5                   -0.0466       0.1181     -0.394     0.6933       
-4                    0.1224       0.1344      0.911     0.3625       
-3                    0.0783       0.1576      0.497     0.6194       
-2                    0.1541       0.0957      1.610     0.1075       
0                     0.4453       0.1627      2.737     0.0062     **
1                     0.7074       0.1903      3.716     0.0002    ***
2                     0.7642       0.1612      4.739     0.0000    ***
3                     0.5525       0.1646      3.357     0.0008    ***
4                     0.2367       0.1909      1.240     0.2150       
5                     0.6463       0.1313      4.921     0.0000    ***
-------------------------------------------------------------------------------------

Signif. codes: '***' 0.001, '**' 0.01, '*' 0.05, '.' 0.1
=====================================================================================
```
```
### BusinessReport.to_dict() - headline + assumption
```json
{
  "effect": 0.5608256172839505,
  "se": 0.12081241968043965,
  "ci_lower": 0.3240376258251508,
  "ci_upper": 0.7976136087427503,
  "alpha_was_honored": true,
  "alpha_override_caveat": null,
  "ci_level": 95,
  "p_value": 3.448543002483855e-06,
  "is_significant": true,
  "near_significance_threshold": false,
  "unit": "per 100k population",
  "unit_kind": "unknown",
  "sign": "positive",
  "breakdown_M": null
}
```
```json
{
  "parallel_trends_variant": "conditional_or_group_time",
  "no_anticipation": true,
  "description": "Identification relies on parallel trends across treatment cohorts and time periods (group-time ATT), plus no anticipation."
}
```

---
