#!/usr/bin/env Rscript
# Generate golden values for de Chaisemartin-D'Haultfoeuille (dCDH)
# parity tests at horizon l=1.
#
# This script fits the R `DIDmultiplegtDYN` package's `did_multiplegt_dyn`
# function (the official R implementation of the dCDH dynamic-effects
# companion paper, NBER WP 29873) on a set of canonical reversible-treatment
# scenarios. At horizon l=1, did_multiplegt_dyn computes DID_1, which is
# numerically identical to DID_M of the AER 2020 paper. Phase 1 of the
# Python diff-diff dCDH implementation tests for parity against these
# golden values.
#
# Usage:
#   Rscript benchmarks/R/generate_dcdh_dynr_test_values.R
#
# Prerequisites:
#   install.packages("DIDmultiplegtDYN")  # CRAN v2.3.3+
#   install.packages("jsonlite")
#
# Output:
#   benchmarks/data/dcdh_dynr_golden_values.json
#
# Each scenario exports:
#   - data: the simulated dataset (so Python tests use identical data)
#   - params: the dCDH options used
#   - results: DID_1 point estimate, SE, CI, placebo, switcher counts

library(DIDmultiplegtDYN)
library(jsonlite)
suppressMessages(library(polars))  # required by DIDmultiplegtDYN >= 2.x

cat("Generating dCDH golden values via DIDmultiplegtDYN at l=1...\n")

output_path <- file.path("benchmarks", "data", "dcdh_dynr_golden_values.json")

# ---------------------------------------------------------------------------
# Helper: Python-mirror reversible-treatment generator.
# Mirrors generate_reversible_did_data() in diff_diff/prep_dgp.py at the
# STRUCTURAL level — the two implementations apply the same pattern logic
# (single_switch / joiners_only / leavers_only / mixed_single_switch) and
# the same fixed-effect / treatment-effect / time-trend / noise model. They
# do NOT produce bit-identical draws even with the same seed: R's set.seed
# and NumPy's default_rng use different RNGs and the parity tests don't
# rely on RNG identity. Instead, the parity tests load THIS R script's
# golden-value JSON output and pass the SAME data (group/period/treatment/
# outcome columns) to the Python estimator, so both sides operate on
# byte-identical input regardless of how it was originally generated.
# ---------------------------------------------------------------------------
gen_reversible <- function(n_groups, n_periods, pattern, seed,
                           p_switch = 0.2, initial_treat_frac = 0.3,
                           cycle_length = 2, treatment_effect = 2.0,
                           heterogeneous_effects = FALSE, effect_sd = 0.5,
                           group_fe_sd = 2.0, time_trend = 0.1, noise_sd = 0.5,
                           n_never_treated = 20, n_always_treated = 20) {
  # n_never_treated and n_always_treated add stable control cohorts so
  # both Python (AER 2020 zero-retention) and R DIDmultiplegtDYN (dynamic
  # paper, drop-cohort) implementations have controls available at every
  # period — eliminating the methodology divergence and giving a clean
  # parity comparison. The total returned panel has
  # n_groups + n_never_treated + n_always_treated groups.
  set.seed(seed)

  # --- Build the (n_groups, n_periods) treatment matrix ---
  D <- matrix(0L, nrow = n_groups, ncol = n_periods)

  if (pattern == "single_switch") {
    initial_treated <- runif(n_groups) < initial_treat_frac
    switch_times <- sample.int(n_periods - 1, n_groups, replace = TRUE)  # 1..(n_periods-1)
    for (g in seq_len(n_groups)) {
      st <- switch_times[g] + 1L  # convert to 1-indexed switch period
      if (initial_treated[g]) {
        D[g, seq_len(st - 1L)] <- 1L
        D[g, st:n_periods] <- 0L
      } else {
        D[g, seq_len(st - 1L)] <- 0L
        D[g, st:n_periods] <- 1L
      }
    }
  } else if (pattern == "joiners_only") {
    switch_times <- sample.int(n_periods - 1, n_groups, replace = TRUE)
    for (g in seq_len(n_groups)) {
      st <- switch_times[g] + 1L
      D[g, st:n_periods] <- 1L
    }
  } else if (pattern == "leavers_only") {
    switch_times <- sample.int(n_periods - 1, n_groups, replace = TRUE)
    for (g in seq_len(n_groups)) {
      st <- switch_times[g] + 1L
      D[g, seq_len(st - 1L)] <- 1L
    }
  } else if (pattern == "mixed_single_switch") {
    switch_times <- sample.int(n_periods - 1, n_groups, replace = TRUE)
    n_joiners <- n_groups %/% 2
    for (g in seq_len(n_groups)) {
      st <- switch_times[g] + 1L
      if (g <= n_joiners) {
        D[g, st:n_periods] <- 1L
      } else {
        D[g, seq_len(st - 1L)] <- 1L
      }
    }
  } else {
    stop(sprintf("Unknown pattern: %s", pattern))
  }

  # --- Append stable control cohorts ---
  if (n_never_treated > 0) {
    D <- rbind(D, matrix(0L, nrow = n_never_treated, ncol = n_periods))
  }
  if (n_always_treated > 0) {
    D <- rbind(D, matrix(1L, nrow = n_always_treated, ncol = n_periods))
  }
  n_groups <- nrow(D)

  # --- Generate fixed effects, true effects, outcomes ---
  group_fe <- rnorm(n_groups, mean = 0, sd = group_fe_sd)
  if (heterogeneous_effects) {
    true_effects <- matrix(rnorm(n_groups * n_periods, mean = treatment_effect, sd = effect_sd),
                           nrow = n_groups, ncol = n_periods)
  } else {
    true_effects <- matrix(treatment_effect, nrow = n_groups, ncol = n_periods)
  }
  true_effects[D == 0] <- 0.0

  period_arr <- 0:(n_periods - 1)
  noise <- matrix(rnorm(n_groups * n_periods, mean = 0, sd = noise_sd),
                  nrow = n_groups, ncol = n_periods)
  Y <- 10.0 + matrix(group_fe, nrow = n_groups, ncol = n_periods) +
       matrix(time_trend * period_arr, nrow = n_groups, ncol = n_periods, byrow = TRUE) +
       true_effects + noise

  # --- Build long-format data frame ---
  group_col <- rep(seq_len(n_groups) - 1L, each = n_periods)  # 0-indexed
  period_col <- rep(period_arr, n_groups)
  treatment_col <- as.vector(t(D))
  outcome_col <- as.vector(t(Y))

  data.frame(
    group = group_col,
    period = period_col,
    treatment = treatment_col,
    outcome = outcome_col
  )
}

# ---------------------------------------------------------------------------
# Helper: extract DID_1 (l=1) results from did_multiplegt_dyn output
# ---------------------------------------------------------------------------
extract_dcdh_l1 <- function(res) {
  # did_multiplegt_dyn returns a results object with $results$Effects matrix.
  # The Effects matrix has one row per effect (Effect_1, Effect_2, ...) and
  # columns: Estimate, SE, LB CI, UB CI, N, Switchers, N.w, Switchers.w. We
  # pull the row for "Effect_1" (the l=1 effect, == DID_M of AER 2020).
  effects <- res$results$Effects
  if (is.null(effects)) {
    stop("did_multiplegt_dyn returned no Effects; check the input data")
  }

  out <- list(
    overall_att = as.numeric(effects[1, "Estimate"]),
    overall_se = as.numeric(effects[1, "SE"]),
    overall_ci_lo = as.numeric(effects[1, "LB CI"]),
    overall_ci_hi = as.numeric(effects[1, "UB CI"]),
    n_switchers = as.numeric(effects[1, "N"])
  )

  # Placebo at lag 1 if available (Placebos has the same column layout)
  placebos <- res$results$Placebos
  if (!is.null(placebos) && nrow(placebos) >= 1) {
    out$placebo_effect <- as.numeric(placebos[1, "Estimate"])
    out$placebo_se <- as.numeric(placebos[1, "SE"])
    out$placebo_ci_lo <- as.numeric(placebos[1, "LB CI"])
    out$placebo_ci_hi <- as.numeric(placebos[1, "UB CI"])
  }

  out
}

# ---------------------------------------------------------------------------
# Helper: convert data frame to exportable list
# ---------------------------------------------------------------------------
export_data <- function(df) {
  list(
    group = as.numeric(df$group),
    period = as.numeric(df$period),
    treatment = as.numeric(df$treatment),
    outcome = as.numeric(df$outcome)
  )
}

scenarios <- list()

# Golden value datasets use n_groups=80 to keep the JSON file small (~50KB)
# while still being large enough to exercise inference.
N_GOLDEN <- 80

# ---------------------------------------------------------------------------
# Scenario 1: single_switch — mix of joiners and leavers
# ---------------------------------------------------------------------------
cat("  Scenario 1: single_switch_mixed\n")
d1 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 6,
                     pattern = "single_switch", seed = 101)
res1 <- did_multiplegt_dyn(
  df = d1, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 1, placebo = 1, ci_level = 95
)
scenarios$single_switch_mixed <- list(
  data = export_data(d1),
  params = list(pattern = "single_switch", n_groups = N_GOLDEN, n_periods = 6,
                seed = 101, effects = 1, placebo = 1, ci_level = 95),
  results = extract_dcdh_l1(res1)
)

# ---------------------------------------------------------------------------
# Scenario 2: joiners_only — pure staggered adoption
# ---------------------------------------------------------------------------
cat("  Scenario 2: joiners_only\n")
d2 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 6,
                     pattern = "joiners_only", seed = 102)
res2 <- did_multiplegt_dyn(
  df = d2, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 1, placebo = 1, ci_level = 95
)
scenarios$joiners_only <- list(
  data = export_data(d2),
  params = list(pattern = "joiners_only", n_groups = N_GOLDEN, n_periods = 6,
                seed = 102, effects = 1, placebo = 1, ci_level = 95),
  results = extract_dcdh_l1(res2)
)

# ---------------------------------------------------------------------------
# Scenario 3: leavers_only — pure staggered removal
# ---------------------------------------------------------------------------
cat("  Scenario 3: leavers_only\n")
d3 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 6,
                     pattern = "leavers_only", seed = 103)
res3 <- did_multiplegt_dyn(
  df = d3, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 1, placebo = 1, ci_level = 95
)
scenarios$leavers_only <- list(
  data = export_data(d3),
  params = list(pattern = "leavers_only", n_groups = N_GOLDEN, n_periods = 6,
                seed = 103, effects = 1, placebo = 1, ci_level = 95),
  results = extract_dcdh_l1(res3)
)

# ---------------------------------------------------------------------------
# Scenario 4: mixed_single_switch — deterministic 50/50 joiners/leavers
# ---------------------------------------------------------------------------
cat("  Scenario 4: mixed_single_switch\n")
d4 <- gen_reversible(n_groups = N_GOLDEN, n_periods = 6,
                     pattern = "mixed_single_switch", seed = 104)
res4 <- did_multiplegt_dyn(
  df = d4, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 1, placebo = 1, ci_level = 95
)
scenarios$mixed_single_switch <- list(
  data = export_data(d4),
  params = list(pattern = "mixed_single_switch", n_groups = N_GOLDEN, n_periods = 6,
                seed = 104, effects = 1, placebo = 1, ci_level = 95),
  results = extract_dcdh_l1(res4)
)

# ---------------------------------------------------------------------------
# Scenario 5: hand-calculable 4-group panel from the worked example.
# This is the panel used by tests/test_methodology_chaisemartin_dhaultfoeuille.py
# in test_hand_calculable_4group_3period_joiners_and_leavers. We capture R's
# answer here so the Python test can assert exact agreement.
# ---------------------------------------------------------------------------
cat("  Scenario 5: hand_calculable_worked_example\n")
d5 <- data.frame(
  group =     c(1, 1, 1,  2, 2, 2,  3, 3, 3,  4, 4, 4),
  period =    c(0, 1, 2,  0, 1, 2,  0, 1, 2,  0, 1, 2),
  treatment = c(0, 1, 1,  1, 1, 0,  0, 0, 0,  1, 1, 1),
  outcome   = c(10, 13, 14,  10, 11, 9,  10, 11, 12,  10, 11, 12)
)
res5 <- did_multiplegt_dyn(
  df = d5, outcome = "outcome", group = "group", time = "period",
  treatment = "treatment", effects = 1, placebo = 0, ci_level = 95
)
scenarios$hand_calculable_worked_example <- list(
  data = export_data(d5),
  params = list(description = "4-group hand-calculable panel from plan worked example",
                effects = 1, placebo = 0, ci_level = 95,
                expected_did_m = 2.5, expected_did_plus = 2.0, expected_did_minus = 3.0),
  results = extract_dcdh_l1(res5)
)

# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------
dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
writeLines(
  toJSON(
    list(scenarios = scenarios,
         generator = "generate_reversible_did_data v1",
         dcdh_package = paste0("DIDmultiplegtDYN ", utils::packageVersion("DIDmultiplegtDYN"))),
    auto_unbox = TRUE,
    digits = 10,
    pretty = TRUE
  ),
  output_path
)
cat(sprintf("dCDH golden values written to %s\n", output_path))
cat(sprintf("File size: %.1f KB\n", file.info(output_path)$size / 1024))
cat(sprintf("Scenarios: %d\n", length(scenarios)))
