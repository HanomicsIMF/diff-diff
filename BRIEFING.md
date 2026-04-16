# SDID Practitioner Validation Tooling - Briefing

## Problem

A data scientist runs `SyntheticDiD`, gets an ATT and a p-value, and then
faces the question: *should I trust this estimate?* The library gives them the
point estimate and inference, but the validation workflow - the steps between
"I got a number" and "I'm confident enough to present this" - is largely
left to the practitioner to assemble from scratch.

The standard validation workflow for synthetic control methods is well
understood in the econometrics literature (Arkhangelsky et al. 2021,
Abadie et al. 2010, Abadie 2021). The pieces include pre-treatment fit
assessment, weight diagnostics, placebo/falsification tests, sensitivity
analysis, and cross-estimator comparison. Our library provides some of the
raw ingredients (pre-treatment RMSE, weight dicts, placebo effects array)
but doesn't connect them into an accessible diagnostic workflow.

The gap is most visible in `practitioner.py`, where `_handle_synthetic`
recommends in-time placebos and leave-one-out analysis but provides only
comment-only pseudo-code. A practitioner following that guidance hits a wall.

## Current state

What we have today:

- `results.pre_treatment_fit` (RMSE) with a warning when it exceeds the
  treated pre-period SD
- `results.get_unit_weights_df()` and `results.get_time_weights_df()`
- Three variance methods: placebo (default), bootstrap, and jackknife (just
  landed in v3.1.1)
- `results.placebo_effects` - stores per-iteration estimates for all three
  variance methods, but for jackknife these are positional LOO estimates
  with no unit labels
- `results.summary()` shows top-5 unit weights and count of non-trivial weights
- `practitioner.py` guidance that names the right steps but can't point to
  runnable code for most of them

What the practitioner must currently build themselves:

- Mapping jackknife LOO estimates back to unit identities to answer "which
  unit, when dropped, changes my estimate the most?"
- In-time placebo tests (re-estimate with a fake treatment date)
- Any weight concentration metric beyond eyeballing the sorted list
- Any sense of whether their RMSE is "bad enough to worry about" beyond
  the binary warning
- Regularization sensitivity (does the ATT change if I perturb zeta?)
- Pre-treatment trajectory data for plotting (the Y matrices are internal
  to `fit()` and not returned)

## Context from prior discussion

The jackknife work created an interesting opportunity. The delete-one-re-estimate
loop already runs for SE computation. The per-unit ATT estimates are stored in
`results.placebo_effects`. The missing piece is a presentation layer that maps
those estimates to unit identities and surfaces the diagnostic interpretation
(which units are influential, how stable is the estimate to unit composition).

More broadly, the validation gaps fall into two categories:

1. **Low-marginal-cost additions** - things where the computation already
   exists and we just need to expose or label it (LOO diagnostic from
   jackknife, weight concentration metrics, trajectory data extraction)

2. **New functionality** - things that require new estimation loops or
   helpers (in-time placebo, regularization sensitivity sweep)

The practitioner guidance in `practitioner.py` should evolve alongside any
new tooling so that the recommended steps point to real, runnable code paths.

## What "done" looks like

A practitioner using SyntheticDiD should be able to follow a credible
validation workflow using library-provided tools and guidance, without
needing to reverse-engineer internals or write substantial boilerplate.
The validation steps recognized in the literature should either be directly
supported or have clear, concrete guidance for how to perform them with
the library's API.

This is not about adding visualization or plotting (that's a separate
concern). It's about making the computational and diagnostic building
blocks accessible and well-documented through the results API and
practitioner guidance.
