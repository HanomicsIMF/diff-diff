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

# Panel data — must match TestJackknifeSERParity.Y_FLAT in
# tests/test_methodology_sdid.py (23 units × 8 periods = 184 values).
Y_flat <- c(
  12.459567808595292, 13.223481099962006, 13.658348196773856,
  13.844051055863837, 13.888854636247594, 14.997677893012806,
  14.494587375086788, 15.851128751231856, 10.527006629006900,
  11.317894498245712, 9.780141451338988, 10.635177418486473,
  11.007911133698329, 11.692547000930196, 11.532445341187122,
  10.646344091442769, 5.779122815714058, 5.265746845809725,
  4.828411925858962, 5.933107464969151, 6.926403492435262,
  7.566662873481445, 6.703831577045862, 7.090431451464497,
  6.703722507026075, 6.453676391630379, 7.301398891231049,
  7.726092498224848, 8.191225590595401, 7.669210641906834,
  8.526151391259425, 7.715169490073769, 8.005628186152748,
  7.523978158267692, 9.049143286687135, 9.434081283341134,
  9.450553333966674, 10.310163601090766, 9.867729569702721,
  9.846941461031360, 10.459939463684098, 11.887686682638062,
  11.249912950470762, 12.093459993478538, 12.226598684379407,
  11.973716581337246, 13.453499811673423, 13.287085704636093,
  10.317796666844943, 10.819165701226847, 10.824437736488752,
  9.582976251622744, 11.521962769964540, 11.495903971828724,
  12.072136575632017, 12.570433156881965, 12.435827624848123,
  13.750744970607428, 13.567397714461393, 14.218726703934166,
  14.459837938730677, 14.659912736018788, 14.077914185301429,
  14.854380461280002, 10.770274645112915, 11.275621916712160,
  12.137534572839927, 12.531125692916383, 12.678920118269170,
  12.304148175294246, 12.497145874675160, 14.103389828901550,
  10.560062989643855, 10.755394606294518, 10.518678427483797,
  11.721841324084256, 11.607272952190801, 11.924464521898100,
  12.782516039349641, 13.026729430318186, 12.546145790341205,
  13.409407032231695, 14.079787980063543, 13.128838312144593,
  13.553836458429620, 13.718363411441658, 13.854625752117343,
  14.924224028489123, 11.906891367097627, 12.128784222882244,
  11.404804355878456, 13.130649630134753, 12.173021974919472,
  12.859165585526416, 12.895280738363951, 13.345233593320895,
  10.435966548001499, 10.663839793569295, 11.030422432974012,
  11.033668451079661, 11.324277503659044, 11.045836529045589,
  11.985219205566086, 12.220060940064094, 14.722723885094736,
  15.772410109968900, 15.256969467031452, 15.568564129971197,
  16.666133193788099, 16.405462433247578, 17.202870693537243,
  17.289652559976691, 7.760317864391456, 8.460282811921017,
  9.462415007659978, 9.956467084312777, 9.726218110324272,
  10.272688229133685, 11.134101608790994, 11.592584658589104,
  7.747112683063268, 8.706521663648207, 8.170907672905205,
  8.679537720718859, 8.962718814069811, 8.861932954235140,
  9.383430460745986, 9.891050023644237, 9.728955313568255,
  9.231765881057163, 9.555677785583788, 10.420693590160205,
  9.844078095298698, 10.651913064308546, 10.196489890710358,
  11.855847076501993, 9.218785934915712, 9.133582433258733,
  10.048827580363175, 9.952567508276010, 10.385962432276619,
  11.596546220044132, 11.164945662130776, 11.016817405176500,
  10.145044557120791, 10.921420538928436, 11.642624728800259,
  10.730067509380019, 11.753738913724906, 11.868862794274008,
  12.574196556067037, 12.311524695461632, 10.800710206252880,
  12.817967597577915, 12.705627126180516, 12.497850142478354,
  12.148734571851643, 13.494742486942219, 13.714835068828613,
  13.770060323710533, 10.010857300549947, 10.787315152039971,
  11.050238955584605, 11.063282099053561, 10.834793458278272,
  17.153286194944865, 17.380010096861866, 16.984758489324143,
  6.913302966281331, 6.938279687001069, 7.537129527669741,
  7.063822443245238, 7.531238453797332, 13.853711102827464,
  13.812711128345372, 14.204067444347162, 13.694867606609098,
  12.929992273442151, 14.397345491024691, 15.116119455987304,
  15.860226513457558, 19.442026093187646, 19.855029109494353,
  20.377546194927845
)
stopifnot(length(Y_flat) == 184L)

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

# Bootstrap loop — record indices and compute tau_b with FIXED weights.
#
# IMPORTANT: this is NOT the behavior of R's default
# synthdid::vcov(method="bootstrap"). The default vcov code path in vcov.R
# rebinds attr(tau_hat, "opts") (which includes update.omega = TRUE from
# the original fit) back into synthdid_estimate via do.call, so each
# bootstrap draw re-estimates ω and λ via Frank-Wolfe with the renormalized
# weights used only as initialization. The call below deliberately omits
# the opts rebind; because update.omega / update.lambda default to
# is.null(weights$*) and we pass non-null weights, this runs a manual
# fixed-weight bootstrap. We use this shape so that the 1e-10 Python-R
# parity test in tests/test_methodology_sdid.py anchors our fixed-weight
# variance_method="bootstrap" against a matching R invocation. Refit
# parity against the default vcov behavior belongs to a separate fixture
# (see variance_method="bootstrap_refit" and the Julia Synthdid.jl
# follow-up anchor).
#
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
