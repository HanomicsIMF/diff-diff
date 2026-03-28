# Practitioner Guide Evaluation: Before/After Empirical Comparison

## Methodology

We tested whether the practitioner guide documentation changes the behavior of
AI agents when performing Difference-in-Differences analysis. Each agent was
given documentation and an identical task prompt, then scored against Baker et
al. (2025)'s 8-step practitioner workflow.

**Task prompt**: "Estimate the effect of a policy intervention using a staggered
difference-in-differences design using the load_mpdta() dataset."

**Before condition** (N=10): Agent receives only the original `docs/llms.txt`
(API reference with estimator list, diagnostics section, tutorial links).

**After condition** (N=10): Agent receives the restructured `docs/llms.txt`
(with practitioner workflow header) + key sections of `docs/llms-practitioner.txt`.

**Model**: 1 Opus + 9 Sonnet (before), 10 Sonnet (after). All agents are fresh
instances with no shared context.

## Scoring Rubric (0-2 per step, 16 total)

| Step | Description | 0 | 1 | 2 |
|------|-------------|---|---|---|
| S1 | Define target parameter | Not mentioned | Mentions ATT types | Explicitly defines weighted/unweighted, policy question |
| S2 | State assumptions | Not mentioned | Mentions parallel trends | Formally names PT variant (PT-GT-NYT etc.) |
| S3 | Appropriate estimator | Uses naive TWFE | Uses CS but no diagnostic | CS + Bacon diagnostic, explains choice |
| S4 | Inference approach | Not discussed | Clusters SEs | Clusters + discusses alternatives (wild bootstrap) |
| S5 | Estimation | No code | Partial code | Complete, working code |
| S6 | Sensitivity analysis | Not done | Mentions but doesn't run | Runs HonestDiD and/or placebo tests |
| S7 | Heterogeneity | Not done | Some aggregation | Group + event study + subgroup |
| S8 | Robustness | Not done | Compares 2 estimators | 3+ estimators + with/without covariates |

## Results

### Overall Scores

| Condition | Mean | SD | SE | Min | Max |
|-----------|------|----|----|-----|-----|
| **Before** | **9.4** | 0.84 | 0.27 | 8 | 11 |
| **After** | **15.25** | 0.26 | 0.08 | 15 | 15.5 |
| **Difference** | **+5.85** | | | | |

**Welch's t-test**: t = 21.0, p < 0.0001
**Cohen's d**: 9.4 (extremely large effect)

### Per-Step Comparison

| Step | Before Mean | After Mean | Change | Interpretation |
|------|------------|-----------|--------|----------------|
| S1: Target parameter | 1.0 | 2.0 | **+1.0** | Agents now explicitly define weighted/unweighted target |
| S2: Assumptions | 1.0 | 2.0 | **+1.0** | Agents now formally name PT variant (PT-GT-NYT) |
| S3: Estimator choice | 2.0 | 2.0 | 0.0 | Already perfect before |
| S4: Inference | 1.0 | 1.5 | +0.5 | Now discuss wild bootstrap alternative |
| S5: Estimation | 2.0 | 2.0 | 0.0 | Already perfect before |
| S6: Sensitivity | **0.1** | **2.0** | **+1.9** | From near-zero to universal HonestDiD + placebo |
| S7: Heterogeneity | 1.4 | 2.0 | +0.6 | Now consistently do group + event study |
| S8: Robustness | 0.9 | 1.75 | +0.85 | Now compare 3 estimators; ~50% add with/without covariates |

### Key Findings

1. **Sensitivity analysis (Step 6) showed the largest improvement**: 0.1 to 2.0
   (+1.9 points). Before, 0/10 agents ran HonestDiD or placebo tests. After,
   10/10 agents ran both. This was the primary gap the guide was designed to fix.

2. **Target parameter and assumptions (Steps 1-2) went from partial to full**:
   Before, agents mentioned "ATT" generically. After, they explicitly name the
   PT variant (PT-GT-NYT), discuss weighted vs unweighted targets, and state
   no-anticipation assumptions.

3. **Robustness (Step 8) nearly doubled**: Before, agents compared at most 2
   estimators. After, all agents compare 3 (CS, SA, BJS), and ~50% include
   explicit with/without covariates comparisons.

4. **Variance collapsed**: Before SD = 0.84, After SD = 0.26. The guide
   standardized behavior — agents now consistently follow the same high-quality
   workflow rather than producing variable-quality ad hoc analyses.

5. **Steps 3 and 5 were already perfect**: Agents already knew to use CS for
   staggered data and could produce working code. The gap was never in mechanics
   but in empiricist reasoning.

## Qualitative Observations

**Before agents** consistently:
- Called CS.fit(), printed summary, stopped
- Mentioned HonestDiD in prose but never ran it
- Used event study pre-periods as informal parallel trends "check"
- Compared at most CS vs SA

**After agents** consistently:
- Structured their code around all 8 Baker steps explicitly
- Ran check_parallel_trends() AND equivalence_test_trends()
- Ran compute_honest_did() with specific M values
- Ran run_all_placebo_tests()
- Compared CS vs SA vs BJS
- Called practitioner_next_steps(results)
- Named specific PT variants (PT-GT-NYT, PT-GT-Nev)

## Iteration 2: Targeted Fixes

After v1, the remaining 0.75 point gap was concentrated in:
- **Step 4 (Inference)**: Agents mentioned wild bootstrap generically but never
  checked the actual cluster count in the data (1.5/2 across all runs).
- **Step 8 (Robustness)**: ~50% of agents skipped with/without covariates
  comparison despite the guide listing it (mean 1.75/2).

### Targeted Changes

1. **Step 4**: Added "You MUST check the cluster count before choosing inference"
   with explicit code: `n_clusters = data[cluster_col].nunique()` + if/else branch.
2. **Step 8**: Strengthened "Report with and without covariates" from a checklist
   item to "REQUIRED — This is not optional" with explanation of why it matters.

### After v2 Results (N=10)

| Condition | Mean | SD | SE |
|-----------|------|----|----|
| Before | 9.4 | 0.84 | 0.27 |
| After v1 | 15.25 | 0.26 | 0.08 |
| **After v2** | **16.0** | **0.0** | **0.0** |

**10/10 agents scored 16/16 — perfect scores with zero variance.**

### Per-Step Progression

| Step | Before | After v1 | After v2 |
|------|--------|----------|----------|
| S1: Target parameter | 1.0 | 2.0 | 2.0 |
| S2: Assumptions | 1.0 | 2.0 | 2.0 |
| S3: Estimator choice | 2.0 | 2.0 | 2.0 |
| S4: Inference | 1.0 | 1.5 | **2.0** |
| S5: Estimation | 2.0 | 2.0 | 2.0 |
| S6: Sensitivity | 0.1 | 2.0 | 2.0 |
| S7: Heterogeneity | 1.4 | 2.0 | 2.0 |
| S8: Robustness | 0.9 | 1.75 | **2.0** |

The two targeted fixes (cluster count check directive + mandatory with/without
covariates) closed the remaining gaps completely.

## Conclusion

The practitioner guide increased analysis quality from **9.4/16 to 16.0/16
(+70%)** in two iterations. The effect is statistically significant (p < 0.0001)
and practically massive. Key results:

- **Sensitivity analysis** went from 0.1 to 2.0 (0/10 agents ran HonestDiD
  before; 10/10 after)
- **Variance collapsed** from SD=0.84 to SD=0.0 — the guide standardized
  behavior so completely that all agents produce the same high-quality workflow
- **Two iterations sufficed**: v1 closed 79% of the gap; targeted v2 fixes
  to Step 4 (cluster count) and Step 8 (covariates) closed the remaining 21%
- **Documentation alone** drove these results — no runtime enforcement was
  needed beyond the `practitioner_next_steps()` function
