# Generate CR2 Bell-McCaffrey golden values via R clubSandwich.
#
# This script is the parity source for CR2 Bell-McCaffrey cluster-robust
# inference implemented in diff_diff/linalg.py::_compute_cr2_bm.
#
# Usage:
#   Rscript benchmarks/R/generate_clubsandwich_golden.R
#
# Requirements:
#   clubSandwich (CRAN), jsonlite, readr
#
# Output:
#   benchmarks/data/clubsandwich_cr2_golden.json
#
# Phase 1a of the HeterogeneousAdoptionDiD implementation (de Chaisemartin,
# Ciccia, D'Haultfoeuille & Knau 2026, arXiv:2405.04465v6). The parity
# dataset below consists of three small deterministic designs; the Python
# test at tests/test_linalg_hc2_bm.py::TestCR2BMParityClubSandwich loads
# this JSON and checks agreement to 6 digits.

suppressPackageStartupMessages({
  library(clubSandwich)
  library(jsonlite)
})

set.seed(20260420)

# --- Three deterministic datasets ---------------------------------------------

make_dataset <- function(name, n_clusters, cluster_sizes, seed) {
  set.seed(seed)
  cluster_ids <- rep(seq_len(n_clusters), times = cluster_sizes)
  n <- length(cluster_ids)
  x <- runif(n, 0, 1)
  # Cluster-level shock to induce within-cluster correlation, plus idiosyncratic noise.
  shock <- rnorm(n_clusters, sd = 0.5)
  y <- 1 + 0.5 * x + shock[cluster_ids] + rnorm(n, sd = 0.2)
  data.frame(name = name, cluster = cluster_ids, x = x, y = y)
}

datasets <- list(
  balanced_small = make_dataset("balanced_small", 5, rep(6, 5), 101),
  unbalanced_medium = make_dataset("unbalanced_medium", 8, c(3, 4, 5, 6, 7, 8, 9, 10), 202),
  singletons_present = make_dataset("singletons_present", 10, c(1, 1, 2, 3, 4, 5, 6, 7, 8, 9), 303)
)

output <- list()

for (nm in names(datasets)) {
  d <- datasets[[nm]]
  fit <- lm(y ~ x, data = d)
  vcov_cr2 <- vcovCR(fit, cluster = d$cluster, type = "CR2")
  # Per-contrast Bell-McCaffrey DOF: one per coefficient via a unit contrast.
  coef_names <- names(coef(fit))
  dof_vec <- sapply(coef_names, function(nm_coef) {
    ctr <- setNames(as.numeric(names(coef(fit)) == nm_coef), names(coef(fit)))
    Wald_test(fit, constraints = matrix(ctr, 1), vcov = vcov_cr2, test = "Satterthwaite")$df
  })
  output[[nm]] <- list(
    x = d$x,
    y = d$y,
    cluster = d$cluster,
    coef = as.numeric(coef(fit)),
    coef_names = coef_names,
    vcov_cr2 = as.numeric(vcov_cr2),
    vcov_shape = dim(vcov_cr2),
    dof_bm = as.numeric(dof_vec),
    cluster_sizes = as.numeric(table(d$cluster))
  )
}

output$meta <- list(
  source = "clubSandwich",
  clubSandwich_version = as.character(packageVersion("clubSandwich")),
  R_version = R.version.string,
  generated_at = format(Sys.time(), tz = "UTC", usetz = TRUE),
  note = "CR2 Bell-McCaffrey cluster-robust parity target for diff_diff._compute_cr2_bm"
)

out_path <- file.path("benchmarks", "data", "clubsandwich_cr2_golden.json")
writeLines(toJSON(output, pretty = TRUE, digits = 15, auto_unbox = TRUE), out_path)
cat("Wrote", out_path, "\n")
