# Development TODO

Internal tracking for technical debt, known limitations, and maintenance tasks.

For the public feature roadmap, see [ROADMAP.md](ROADMAP.md).

---

## Known Limitations

Current limitations that may affect users:

| Issue | Location | Priority | Notes |
|-------|----------|----------|-------|
| MultiPeriodDiD wild bootstrap not supported | `estimators.py:778-784` | Low | Edge case |
| `predict()` raises NotImplementedError | `estimators.py:567-588` | Low | Rarely needed |

For survey-specific limitations (NotImplementedError paths), see the
[Current Limitations](docs/survey-roadmap.md#current-limitations) section
of survey-roadmap.md.

## Code Quality

### Large Module Files

Target: < 1000 lines per module for maintainability. Updated 2026-03-29.

| File | Lines | Action |
|------|-------|--------|
| `power.py` | 2588 | Consider splitting (power analysis + MDE + sample size) |
| `linalg.py` | 2289 | Monitor — unified backend, splitting would hurt cohesion |
| `staggered.py` | 2275 | Monitor — grew with survey support |
| `imputation.py` | 2009 | Monitor |
| `triple_diff.py` | 1921 | Monitor |
| `utils.py` | 1902 | Monitor |
| `two_stage.py` | 1708 | Monitor |
| `survey.py` | 1646 | Monitor — grew with Phase 6 features |
| `continuous_did.py` | 1626 | Monitor |
| `honest_did.py` | 1511 | Acceptable |
| `sun_abraham.py` | 1540 | Acceptable |
| `estimators.py` | 1357 | Acceptable |
| `trop_local.py` | 1261 | Acceptable |
| `trop_global.py` | 1251 | Acceptable |
| `prep.py` | 1225 | Acceptable |
| `pretrends.py` | 1105 | Acceptable |
| `trop.py` | 981 | Split done — trop_global.py + trop_local.py |
| `visualization/` | 4172 | Subpackage (split across 7 files) — OK |

---

### Tech Debt from Code Reviews

Deferred items from PR reviews that were not addressed before merge.

#### Methodology/Correctness

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| dCDH: Phase 1 per-period placebo DID_M^pl has NaN SE (no IF derivation for the per-period aggregation path). Multi-horizon placebos (L_max >= 1) have valid SE. | `chaisemartin_dhaultfoeuille.py` | #294 | Low |
| dCDH: Survey cell-period allocator's post-period attribution is a library convention, not derived from the observation-level survey linearization. MC coverage is empirically close to nominal on the test DGP; a formal derivation (or a covariance-aware two-cell alternative) is deferred. Documented in REGISTRY.md survey IF expansion Note. | `chaisemartin_dhaultfoeuille.py`, `docs/methodology/REGISTRY.md` | PR 2 | Medium |
| dCDH: Parity test SE/CI assertions only cover pure-direction scenarios; mixed-direction SE comparison is structurally apples-to-oranges (cell-count vs obs-count weighting). | `test_chaisemartin_dhaultfoeuille_parity.py` | #294 | Low |
| CallawaySantAnna: consider materializing NaN entries for non-estimable (g,t) cells in group_time_effects dict (currently omitted with consolidated warning); would require updating downstream consumers (event study, balance_e, aggregation) | `staggered.py` | #256 | Low |
| ImputationDiD dense `(A0'A0).toarray()` scales O((U+T+K)^2), OOM risk on large panels | `imputation.py` | #141 | Medium (deferred — only triggers when sparse solver fails) |
| Multi-absorb weighted demeaning needs iterative alternating projections for N > 1 absorbed FE with survey weights; unweighted multi-absorb also uses single-pass (pre-existing, exact only for balanced panels) | `estimators.py` | #218 | Medium |
| EfficientDiD `control_group="last_cohort"` trims at `last_g - anticipation` but REGISTRY says `t >= last_g`. With `anticipation=0` (default) these are identical. With `anticipation>0`, code is arguably more conservative (excludes anticipation-contaminated periods). Either align REGISTRY with code or change code to `t < last_g` — needs design decision. | `efficient_did.py` | #230 | Low |

| TripleDifference power: `generate_ddd_data` is a fixed 2×2×2 cross-sectional DGP — no multi-period or unbalanced-group support. Add a `generate_ddd_panel_data` for panel DDD power analysis. | `prep_dgp.py`, `power.py` | #208 | Low |
| Survey design resolution/collapse patterns are inconsistent across panel estimators — ContinuousDiD rebuilds unit-level design in SE code, EfficientDiD builds once in fit(), StackedDiD re-resolves on stacked data; extract shared helpers for panel-to-unit collapse, post-filter re-resolution, and metadata recomputation | `continuous_did.py`, `efficient_did.py`, `stacked_did.py` | #226 | Low |
| Survey-weighted Silverman bandwidth in EfficientDiD conditional Omega* — `_silverman_bandwidth()` uses unweighted mean/std for bandwidth selection; survey-weighted statistics would better reflect the population distribution but is a second-order refinement | `efficient_did_covariates.py` | — | Low |
| TROP: `fit()` and `_fit_global()` share ~150 lines of near-identical data setup (panel pivoting, absorbing-state validation, first-treatment detection, effective rank, NaN warnings). Both bootstrap methods also duplicate the stratified resampling loop. Extract shared helpers to eliminate cross-file sync risk. | `trop.py`, `trop_global.py`, `trop_local.py` | — | Low |
| StaggeredTripleDifference R cross-validation: CSV fixtures not committed (gitignored); tests skip without local R + triplediff. Commit fixtures or generate deterministically. | `tests/test_methodology_staggered_triple_diff.py` | #245 | Medium |
| StaggeredTripleDifference R parity: benchmark only tests no-covariate path (xformla=~1). Add covariate-adjusted scenarios and aggregation SE parity assertions. | `benchmarks/R/benchmark_staggered_triplediff.R` | #245 | Medium |
| StaggeredTripleDifference: per-cohort group-effect SEs include WIF (conservative vs R's wif=NULL). Documented in REGISTRY. Could override mixin for exact R match. | `staggered_triple_diff.py` | #245 | Low |
| HonestDiD Delta^RM: uses naive FLCI instead of paper's ARP conditional/hybrid confidence sets (Sections 3.2.1-3.2.2). ARP infrastructure exists but moment inequality transformation needs calibration. CIs are conservative (wider, valid coverage). | `honest_did.py` | #248 | Medium |
| Replicate weight tests use Fay-like BRR perturbations (0.5/1.5), not true half-sample BRR. Add true BRR regressions per estimator family. Existing `test_survey_phase6.py` covers true BRR at the helper level. | `tests/test_replicate_weight_expansion.py` | #253 | Low |
| WooldridgeDiD: QMLE sandwich uses `aweight` cluster-robust adjustment `(G/(G-1))*(n-1)/(n-k)` vs Stata's `G/(G-1)` only. Conservative (inflates SEs). Add `qmle` weight type if Stata golden values confirm material difference. | `wooldridge.py`, `linalg.py` | #216 | Medium |
| WooldridgeDiD: aggregation weights use cell-level n_{g,t} counts. Paper (W2025 Eqs. 7.2-7.4) defines cohort-share weights. Add optional `weights="cohort_share"` parameter to `aggregate()`. | `wooldridge_results.py` | #216 | Medium |
| WooldridgeDiD: canonical link requirement (W2023 Prop 3.1) not enforced — no warning if user applies wrong method to outcome type. Estimator is consistent regardless, but equivalence with imputation breaks. | `wooldridge.py` | #216 | Low |
| WooldridgeDiD: Stata `jwdid` golden value tests — add R/Stata reference script and `TestReferenceValues` class. | `tests/test_wooldridge.py` | #216 | Medium |
| Thread `vcov_type` (classical / hc1 / hc2 / hc2_bm) through the 8 standalone estimators that expose `cluster=`: `CallawaySantAnna`, `SunAbraham`, `ImputationDiD`, `TwoStageDiD`, `TripleDifference`, `StackedDiD`, `WooldridgeDiD`, `EfficientDiD`. Phase 1a added `vcov_type` to the `DifferenceInDifferences` inheritance chain only. | multiple | Phase 1a | Medium |
| Weighted one-way Bell-McCaffrey (`vcov_type="hc2_bm"` + `weights`, no cluster) currently raises `NotImplementedError`. `_compute_bm_dof_from_contrasts` builds its hat matrix from the unscaled design via `X (X'WX)^{-1} X' W`, but `solve_ols` solves the WLS problem by transforming to `X* = sqrt(w) X`, so the correct symmetric idempotent residual-maker is `M* = I - sqrt(W) X (X'WX)^{-1} X' sqrt(W)`. Rederive the Satterthwaite `(tr G)^2 / tr(G^2)` ratio on the transformed design and add weighted parity tests before lifting the guard. | `linalg.py::_compute_bm_dof_from_contrasts`, `linalg.py::_validate_vcov_args` | Phase 1a | Medium |
| HC2 / HC2 + Bell-McCaffrey on absorbed-FE fits currently raises `NotImplementedError` in three places: `TwoWayFixedEffects` unconditionally; `DifferenceInDifferences(absorb=..., vcov_type in {"hc2","hc2_bm"})`; `MultiPeriodDiD(absorb=..., vcov_type in {"hc2","hc2_bm"})`. Within-transformation preserves coefficients and residuals under FWL but not the hat matrix, so the reduced-design `h_ii` is not the diagonal of the full FE projection and CR2's block adjustment `A_g = (I - H_gg)^{-1/2}` is likewise wrong on absorbed cluster blocks. Lifting the guard needs HC2/CR2-BM computed from the full absorbed projection (unit/time FE dummies reconstructed internally, or a FE-aware hat-matrix formulation) and a parity harness against a full-dummy OLS run or R `fixest`/`clubSandwich`. HC1/CR1 are unaffected by this because they have no leverage term. | `twfe.py::fit`, `estimators.py::DifferenceInDifferences.fit`, `estimators.py::MultiPeriodDiD.fit` | Phase 1a | Medium |
| Weighted CR2 Bell-McCaffrey cluster-robust (`vcov_type="hc2_bm"` + `cluster_ids` + `weights`) currently raises `NotImplementedError`. Weighted hat matrix and residual rebalancing need threading per clubSandwich WLS handling. | `linalg.py::_compute_cr2_bm` | Phase 1a | Medium |
| Regenerate `benchmarks/data/clubsandwich_cr2_golden.json` from R (`Rscript benchmarks/R/generate_clubsandwich_golden.R`). Current JSON has `source: python_self_reference` as a stability anchor until an authoritative R run. | `benchmarks/R/generate_clubsandwich_golden.R` | Phase 1a | Medium |
| `honest_did.py:1907` `np.linalg.solve(A_sys, b_sys) / except LinAlgError: continue` is a silent basis-rejection in the vertex-enumeration loop that is algorithmically intentional (try the next basis). Consider surfacing a count of rejected bases as a diagnostic when ARP enumeration exhausts, so users see when the vertex search was heavily constrained. Not a silent failure in the sense of the Phase 2 audit (the algorithm is supposed to skip), but the diagnostic would help debug borderline cases. | `honest_did.py` | #334 | Low |
| TROP Rust vs Python bootstrap SE divergence under fixed seed: `seed=42` on a tiny panel produces ~28% bootstrap-SE gap. Root cause: Rust bootstrap uses its own RNG (`rand` crate) while Python uses `numpy.random.default_rng`; same seed value maps to different bytestreams across backends. Audit axis-H (RNG/seed) adjacent. `@pytest.mark.xfail(strict=True)` in `tests/test_rust_backend.py::TestTROPRustEdgeCaseParity::test_bootstrap_seed_reproducibility` baselines the gap. Unifying RNG (threading a numpy-generated seed-sequence into Rust, or porting Python to ChaCha) would close it. | `trop_global.py`, `rust/` | follow-up | Medium |
| `bias_corrected_local_linear`: extend golden parity to `kernel="triangular"` and `kernel="uniform"` (currently epa-only; all three kernels share `kernel_W` and the `lprobust` math, so parity is expected but not separately asserted). | `benchmarks/R/generate_nprobust_lprobust_golden.R`, `tests/test_bias_corrected_lprobust.py` | Phase 1c | Low |
| `bias_corrected_local_linear`: expose `vce in {"hc0", "hc1", "hc2", "hc3"}` on the public wrapper once R parity goldens exist (currently raises `NotImplementedError`). The port-level `lprobust` and `lprobust_res` already support all four; expanding the public surface requires a golden generator for each hc mode and a decision on hc2/hc3 q-fit leverage (R reuses p-fit `hii` for q-fit residuals; whether to match that or stage-match deserves a derivation before the wrapper advertises CCT-2014 conformance). | `diff_diff/local_linear.py::bias_corrected_local_linear`, `benchmarks/R/generate_nprobust_lprobust_golden.R`, `tests/test_bias_corrected_lprobust.py` | Phase 1c | Medium |
| `bias_corrected_local_linear`: support `weights=` once survey-design adaptation lands. nprobust's `lprobust` has no weight argument so there is no parity anchor; derivation needed. | `diff_diff/local_linear.py`, `diff_diff/_nprobust_port.py::lprobust` | Phase 1c | Medium |
| `bias_corrected_local_linear`: support multi-eval grid (`neval > 1`) with cross-covariance (`covgrid=TRUE` branch of `lprobust.R:253-378`). Not needed for HAD but useful for multi-dose diagnostics. | `diff_diff/_nprobust_port.py::lprobust` | Phase 1c | Low |
| Clustered-DGP parity: Phase 1c's DGP 4 uses manual `h=b=0.3` to sidestep an nprobust-internal singleton-cluster bug in `lpbwselect.mse.dpi`'s pilot fits. Once nprobust ships a fix (or we derive one independently), add a clustered-auto-bandwidth parity test. | `benchmarks/R/generate_nprobust_lprobust_golden.R` | Phase 1c | Low |
| `HeterogeneousAdoptionDiD` joint cross-horizon covariance on event study: per-horizon SEs use INDEPENDENT sandwiches in Phase 2b (paper-faithful pointwise CIs per Pierce-Schott Figure 2). A follow-up could derive an IF-based stacking of per-horizon scores for joint cross-horizon inference (needed for joint hypothesis tests across event-time horizons). Block-bootstrap is a reasonable alternative. | `diff_diff/had.py::_fit_event_study` | Phase 2b | Low |
| `HeterogeneousAdoptionDiD` event-study staggered-timing beyond last cohort: Phase 2b auto-filters staggered panels to the last cohort per paper Appendix B.2. Earlier-cohort treatment effects are not identified by HAD; redirecting to `ChaisemartinDHaultfoeuille` / `did_multiplegt_dyn` is the paper's prescription. A full staggered HAD would require a different identification path (out of paper scope). | `diff_diff/had.py::_validate_had_panel_event_study` | Phase 2b | Low |
| `HeterogeneousAdoptionDiD`: survey-design integration (`survey=SurveyDesign(...)`). Currently raises `NotImplementedError`. Requires Taylor-linearization of the β-scale rescaling and replicate-weight-compatible 2SLS variance on the mass-point path. | `diff_diff/had.py` | Phase 2a | Medium |
| `HeterogeneousAdoptionDiD`: `weights=` support. Deferred jointly with survey integration. nprobust's `lprobust` has no weight argument so the nonparametric continuous path needs a derivation; the 2SLS mass-point path needs weighted-sandwich parity. | `diff_diff/had.py` | Phase 2a | Medium |
| `HeterogeneousAdoptionDiD` mass-point: `vcov_type in {"hc2", "hc2_bm"}` raises `NotImplementedError` pending a 2SLS-specific leverage derivation. The OLS leverage `x_i' (X'X)^{-1} x_i` is wrong for 2SLS; the correct finite-sample correction uses `x_i' (Z'X)^{-1} (...) (X'Z)^{-1} x_i`. Needs derivation plus an R / Stata (`ivreg2 small robust`) parity anchor. | `diff_diff/had.py::_fit_mass_point_2sls` | Phase 2a | Medium |
| `HeterogeneousAdoptionDiD` continuous paths: thread `cluster=` through `bias_corrected_local_linear` (Phase 1c's wrapper already supports cluster; Phase 2a ignores it with a `UserWarning` on the continuous path to keep scope tight). | `diff_diff/had.py`, `diff_diff/local_linear.py` | Phase 2a | Low |
| `HeterogeneousAdoptionDiD` Phase 3: `qug_test()`, `stute_test()`, `yatchew_hr_test()` pre-test diagnostics (paper Section 3.3). Composite helper `did_had_pretest_workflow()`. Not part of Phase 2a scope. | `diff_diff/had.py`, new module | Phase 2a | Medium |
| `HeterogeneousAdoptionDiD` Phase 4: Pierce-Schott (2016) replication harness; reproduce paper Figure 2 values and Table 1 coverage rates. | `benchmarks/`, `tests/` | Phase 2a | Low |
| `HeterogeneousAdoptionDiD` Phase 5: `practitioner_next_steps()` integration, tutorial notebook, and `llms.txt` updates (preserving UTF-8 fingerprint). | `diff_diff/practitioner.py`, `tutorials/`, `diff_diff/guides/` | Phase 2a | Low |
| `HeterogeneousAdoptionDiD` time-varying dose on event study: Phase 2b REJECTS panels where `D_{g,t}` varies within a unit for `t >= F` (the aggregation uses `D_{g, F}` as the single regressor for all horizons, paper Appendix B.2 constant-dose convention). A follow-up PR could add a time-varying-dose estimator for these panels; current behavior is front-door rejection with a redirect to `ChaisemartinDHaultfoeuille`. | `diff_diff/had.py::_validate_had_panel_event_study` | Phase 2b | Low |
| `HeterogeneousAdoptionDiD` repeated-cross-section support: paper Section 2 defines HAD on panel OR repeated cross-section, but Phase 2a is panel-only. RCS inputs (disjoint unit IDs between periods) are rejected by the balanced-panel validator with the generic "unit(s) do not appear in both periods" error. A follow-up PR will add an RCS identification path based on pre/post cell means (rather than unit-level first differences), with its own validator and a distinct `data_mode` / API surface. | `diff_diff/had.py::_validate_had_panel`, `diff_diff/had.py::_aggregate_first_difference` | Phase 2a | Medium |
| SyntheticDiD: compose refit bootstrap with survey designs (Rao-Wu rescaled weights + Frank-Wolfe re-estimation on each draw). Currently raises `NotImplementedError` — paper has no survey support and R has no survey support, so the composition needs its own derivation. | `synthetic_did.py::fit` | follow-up | Low |
| SyntheticDiD: refit-bootstrap cross-language parity anchor against either R's default `synthdid::vcov(method="bootstrap")` (which is refit per vcov.R) or Julia `Synthdid.jl::src/vcov.jl::bootstrap_se`. Current R-parity fixture `test_bootstrap_se_matches_r` only covers the fixed-weight path via a manual `synthdid_estimate()` invocation that omits the opts rebind; the refit path has no direct cross-language anchor (only same-library placebo-SE-tracks-refit-SE + AER §6.3 MC truth). Julia is the cleanest target because its vcov runs refit by construction; R would require driving the default vcov path through `bootstrap_sample` rather than the manual fixed-weight shape. Tolerance target: 1e-6 on Monte Carlo samples (different BLAS + RNG paths preclude 1e-10). | `benchmarks/R/`, `benchmarks/julia/`, `tests/` | follow-up | Low |

#### Performance

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| ImputationDiD event-study SEs recompute full conservative variance per horizon (should cache A0/A1 factorization) | `imputation.py` | #141 | Low |
| Rust faer SVD ndarray-to-faer conversion overhead (minimal vs SVD cost) | `rust/src/linalg.rs:67` | #115 | Low |
| Unrelated label events (e.g., adding `bug` label) re-trigger CI workflows when `ready-for-ci` is already present; filter `labeled`/`unlabeled` events to only `ready-for-ci` transitions | `.github/workflows/rust-test.yml`, `notebooks.yml` | #269 | Low |
| `bread_inv` as a performance kwarg on `compute_robust_vcov` to avoid re-inverting `(X'WX)` when the caller already has it. Deferred from Phase 1a for scope. HC2 and HC2+BM both need the bread inverse, so a shared hint would save one `np.linalg.solve` per sandwich. | `linalg.py::compute_robust_vcov` | Phase 1a | Low |
| Rust-backend HC2 implementation. Current Rust path only supports HC1; HC2 and CR2 Bell-McCaffrey fall through to the NumPy backend. For large-n fits this is noticeable. | `rust/src/linalg.rs` | Phase 1a | Low |
| CR2 Bell-McCaffrey DOF uses a naive `O(n² k)` per-coefficient loop over cluster pairs. Pustejovsky-Tipton (2018) Appendix B has a scores-based formulation that avoids the full `n × n` `M` matrix. Switch when a user hits a large-`n` cluster-robust design. | `linalg.py::_compute_cr2_bm` | Phase 1a | Low |

#### Testing/Docs

| Issue | Location | PR | Priority |
|-------|----------|----|----------|
| R comparison tests spawn separate `Rscript` per test (slow CI) | `tests/test_methodology_twfe.py:294` | #139 | Low |
| CS R helpers hard-code `xformla = ~ 1`; no covariate-adjusted R benchmark for IRLS path | `tests/test_methodology_callaway.py` | #202 | Low |
| ~1583 `duplicate object description` Sphinx warnings — restructure `docs/api/*.rst` to avoid duplicate `:members:` + `autosummary` (count grew from ~376 as API surface expanded) | `docs/api/*.rst` | — | Low |
| Doc-snippet smoke tests only cover `.rst` files; `.txt` AI guides outside CI validation | `tests/test_doc_snippets.py` | #239 | Low |
| Add CI validation for `docs/doc-deps.yaml` integrity (stale paths, unmapped source files) | `docs/doc-deps.yaml` | #269 | Low |
| Sphinx autodoc fails to import 3 result members: `DiDResults.ci`, `MultiPeriodDiDResults.att`, `CallawaySantAnnaResults.aggregate` — investigate whether these are renamed/removed or just unresolvable from autosummary template | `docs/api/results.rst`, `docs/api/staggered.rst` | — | Medium |
| `EDiDBootstrapResults` cross-reference is ambiguous — class is exported from both `diff_diff` and `diff_diff.efficient_did_bootstrap`, producing 3 "more than one target found" warnings. Add `:noindex:` to one source or use full-path refs | `diff_diff/efficient_did_results.py`, `docs/api/efficient_did.rst` | — | Low |
| Tracked Sphinx autosummary stubs in `docs/api/_autosummary/*.rst` are stale — every sphinx build regenerates them with new attributes (e.g., `coef_var`, `survey_metadata`) that have been added to result classes. Either commit a refresh or move the directory to `.gitignore` and treat as build output. Also 6 untracked stubs exist for newer estimators (`WooldridgeDiD`, `SimulationMDEResults`, etc.) that have never been committed. | `docs/api/_autosummary/` | — | Low |
| HonestDiD `test_m0_short_circuit` uses wall-clock `elapsed < 0.5s` as a proxy for "short-circuit path taken" instead of calling the full optimizer. Replace with a direct correctness signal (mock/spy the optimizer or check a state flag) so the test doesn't depend on CI timing. Not flaky today at 500ms, but load-bearing correctness on a timing proxy is brittle. | `tests/test_methodology_honest_did.py:246` | — | Low |
| SyntheticDiD: rename internal `placebo_effects` variable to `null_or_bootstrap_effects` (or `variance_effects`). Misleading name across the placebo/bootstrap/bootstrap_refit/jackknife dispatch paths — now holds four different contents depending on variance method. Low-risk refactor; priority bumped since refit shipping made the misname more load-bearing. | `synthetic_did.py`, `synthetic_did_results.py` | follow-up | Medium |

---

### Standard Error Consistency

Different estimators compute SEs differently. Consider unified interface.

| Estimator | Default SE Type |
|-----------|-----------------|
| DifferenceInDifferences | HC1 or cluster-robust |
| TwoWayFixedEffects | Always cluster-robust (unit level) |
| CallawaySantAnna | Simple difference-in-means SE |
| SyntheticDiD | Bootstrap or placebo-based |

**Action**: Consider adding `se_type` parameter for consistency across estimators.

### Type Annotations

Mypy reports 0 errors. All mixin `attr-defined` errors resolved via
`TYPE_CHECKING`-guarded method stubs in bootstrap mixin classes.

## Deprecated Code

Deprecated parameters still present for backward compatibility:

- `lambda_reg` and `zeta` in `SyntheticDiD` (`synthetic_did.py`)
  - Deprecated in favor of `zeta_omega`/`zeta_lambda` parameters
  - Remove in v4.0.0 (SemVer-safe: public kwarg removal requires a major bump)

---

## Test Coverage

**Note**: 21 visualization tests are skipped when matplotlib unavailable—this is expected.

---

## Honest DiD Improvements

Enhancements for `honest_did.py`:

- [ ] Improved C-LF implementation with direct optimization instead of grid search
  (current implementation uses simplified FLCI approach with estimation uncertainty
  adjustment; see `honest_did.py:947`)
- [x] Support for CallawaySantAnnaResults (implemented in `honest_did.py:612-653`;
  requires `aggregate='event_study'` when calling `CallawaySantAnna.fit()`)
- [ ] Event-study-specific bounds for each post-period
- [ ] Hybrid inference methods
- [ ] Simulation-based power analysis for honest bounds

---

## CallawaySantAnna Bootstrap Improvements

- [ ] Consider aligning p-value computation with R `did` package (symmetric percentile method)

---

## RuntimeWarnings in Linear Algebra Operations

### Apple Silicon M4 BLAS Bug (numpy < 2.3)

Spurious RuntimeWarnings ("divide by zero", "overflow", "invalid value") are emitted by `np.matmul`/`@` on Apple Silicon M4 + macOS Sequoia with numpy < 2.3. The warnings appear for matrices with ≥260 rows but **do not affect result correctness** — coefficients and fitted values are valid (no NaN/Inf), and the design matrices are full rank.

**Root cause**: Apple's BLAS SME (Scalable Matrix Extension) kernels corrupt the floating-point status register, causing spurious FPE signals. Tracked in [numpy#28687](https://github.com/numpy/numpy/issues/28687) and [numpy#29820](https://github.com/numpy/numpy/issues/29820). Fixed in numpy ≥ 2.3 via [PR #29223](https://github.com/numpy/numpy/pull/29223).

**Not reproducible** on M3, Intel, or Linux.

- [ ] `linalg.py:162` - Warnings in fitted value computation (`X @ coefficients`)
  - Caused by M4 BLAS bug, not extreme coefficient values
  - Seen in test_prep.py during treatment effect recovery tests (n > 260)
- [ ] `triple_diff.py:307,323` - Warnings in propensity score computation
  - Occurs in IPW and DR estimation methods with covariates
  - Related to logistic regression overflow in edge cases (separate from BLAS bug)

- **Long-term:** Revert to `@` operator when numpy ≥ 2.3 becomes the minimum supported version.

---

## Feature Gaps (from R `did` package comparison)

Features in R's `did` package that block porting additional tests:

| Feature | R tests blocked | Priority | Status |
|---------|----------------|----------|--------|
| Calendar time aggregation | 1 test in test-att_gt.R | Low | |

---

## Performance Optimizations

Potential future optimizations:

- [ ] JIT compilation for bootstrap loops (numba)
- [ ] Sparse matrix handling for large fixed effects

### QR+SVD Redundancy in Rank Detection

**Background**: The current `solve_ols()` implementation performs both QR (for rank detection) and SVD (for solving) decompositions on rank-deficient matrices. This is technically redundant since SVD can determine rank directly.

**Current approach** (R-style, chosen for robustness):
1. QR with pivoting for rank detection (`_detect_rank_deficiency()`)
2. scipy's `lstsq` with 'gelsd' driver (SVD-based) for solving

**Why we use QR for rank detection**:
- QR with pivoting provides the canonical ordering of linearly dependent columns
- R's `lm()` uses this approach for consistent dropped-column reporting
- Ensures consistent column dropping across runs (SVD column selection can vary)

**Potential optimization** (future work):
- Skip QR when `rank_deficient_action="silent"` since we don't need column names
- Use SVD rank directly in the Rust backend (already implemented)
- Add `skip_rank_check` parameter for hot paths where matrix is known to be full-rank (implemented in v2.2.0)

**Priority**: Low - the QR overhead is minimal compared to SVD solve, and correctness is more important than micro-optimization.

### Incomplete `check_finite` Bypass

**Background**: The `solve_ols()` function accepts a `check_finite=False` parameter intended to skip NaN/Inf validation for performance in hot paths where data is known to be clean.

**Current limitation**: When `check_finite=False`, our explicit validation is skipped, but scipy's internal QR decomposition in `_detect_rank_deficiency()` still validates finite values. This means callers cannot fully bypass all finite checks.

**Impact**: Minimal - the scipy check is fast and only affects edge cases where users explicitly pass `check_finite=False` with non-finite data (which would be a bug in their code anyway).

**Potential fix** (future work):
- Pass `check_finite=False` through to scipy's QR call (requires scipy >= 1.9.0)
- Or skip `_detect_rank_deficiency()` entirely when `check_finite=False` and `_skip_rank_check=True`

**Priority**: Low - this is an edge case optimization that doesn't affect correctness.

