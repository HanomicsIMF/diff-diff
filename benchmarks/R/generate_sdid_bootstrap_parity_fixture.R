#!/usr/bin/env Rscript
# Generate an R-parity fixture for SyntheticDiD bootstrap SE.
#
# The fixture pins B non-degenerate bootstrap indices and the resulting
# R-computed bootstrap SE so that the Python R-parity test can feed the
# same indices through `_bootstrap_se` and expect bit-identical SE.
#
# Usage:
#   Rscript benchmarks/R/generate_sdid_bootstrap_parity_fixture.R
#
# Output:
#   tests/data/sdid_bootstrap_indices_r.json

library(synthdid)
library(jsonlite)

# Panel data — must match TestJackknifeSERParity in tests/test_methodology_sdid.py
Y_flat <- c(
  10.496714153011233, 9.861735698828815, 10.647688538100692,
  11.523029856408025, 9.765846625276664,  9.765275128435143,
  11.579212815507391, 10.767434729152909, 9.530525614065048,
  11.542560043585965, 10.241962271566207, 9.913380592302847,
  10.656348546859186, 10.708051752292503, 10.208339411407656,
  10.699376211546181, 10.005113145840244, 11.466240509800877,
  9.375411058822924,  9.704119846235497,  10.231216590181416,
  11.031599975853577, 11.057835140039923, 11.192136797527128,
  11.212700079956128, 11.291769039560822, 11.489765944538767,
  10.967023408812516, 11.178556285755873, 11.950749051590392,
  11.611488895752247, 11.051698965028522, 11.048729964911076,
  11.092181838374776, 11.180928177452678, 11.488853359306952,
  11.050238955584605, 11.063282099053561, 10.834793458278272,
  17.153286194944865, 17.380010096861866, 16.984758489324143,
  6.913302966281331,  6.938279687001069,  7.537129527669741,
  7.063822443245238,  7.531238453797332,  13.853711102827464,
  13.812711128345372, 14.204067444347162, 13.694867606609098,
  12.929992273442151, 14.397345491024691, 15.116119455987304,
  15.860226513457558, 19.442026093187646, 19.855029109494353,
  20.377546194927845
)

N0 <- 20L   # controls
N1 <- 3L    # treated
T0 <- 5L    # pre
T1 <- 3L    # post
N  <- N0 + N1
T  <- T0 + T1

Y <- matrix(Y_flat, nrow = N, ncol = T, byrow = TRUE)

# Fit once to obtain the omega / lambda that the bootstrap holds fixed
tau_hat <- synthdid_estimate(Y, N0, T0)
weights <- attr(tau_hat, "weights")

# Bootstrap loop — record indices and compute tau_b with fixed weights,
# mimicking synthdid::vcov(method="bootstrap") per the package source.
# Retry on degenerate draws (no controls or no treated) so the fixture
# contains B non-degenerate rows and Python's `_bootstrap_indices` seam
# consumes them all without skipping.
set.seed(42L)
B <- 200L

sum_normalize <- function(v) {
  s <- sum(v)
  if (s > 0) v / s else rep(1 / length(v), length(v))
}

indices_matrix <- matrix(0L, nrow = B, ncol = N)
tau_boot       <- numeric(B)
b <- 1L
while (b <= B) {
  ind <- sample(seq_len(N), replace = TRUE)
  n_co_b <- sum(ind <= N0)
  if (n_co_b == 0L || n_co_b == N) next   # degenerate — retry
  weights_boot <- weights
  weights_boot$omega <- sum_normalize(weights$omega[sort(ind[ind <= N0])])
  tau_boot_b <- synthdid_estimate(
    Y[sort(ind), ], sum(ind <= N0), T0,
    weights = weights_boot
  )
  indices_matrix[b, ] <- ind
  tau_boot[b] <- as.numeric(tau_boot_b)
  b <- b + 1L
}

se <- sqrt((B - 1) / B) * sd(tau_boot)

output_path <- file.path("tests", "data", "sdid_bootstrap_indices_r.json")
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write_json(
  list(
    indices      = indices_matrix,
    seed         = 42L,
    n_bootstrap  = B,
    se           = se,
    att          = as.numeric(tau_hat),
    metadata = list(
      r_version        = R.version.string,
      synthdid_version = as.character(packageVersion("synthdid")),
      panel_N          = N,
      panel_T          = T,
      N0 = N0, T0 = T0
    )
  ),
  output_path,
  pretty     = TRUE,
  auto_unbox = TRUE,
  digits     = NA   # preserve full float64 precision
)
cat(sprintf("Wrote %s (B=%d, SE=%.15g)\n", output_path, B, se))
