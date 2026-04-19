# Speed Review - Practitioner Workflow Benchmarks

Scenario-driven performance measurement for end-to-end practitioner chains,
as distinct from `benchmarks/run_benchmarks.py` which measures R-parity on
isolated `fit()` calls.

## Why these exist

See [`docs/performance-scenarios.md`](../../docs/performance-scenarios.md) for
the full methodology. Short version: the existing benchmarks measure
`fit()` in isolation on 200 x 8 synthetic panels, which does not reflect what
a practitioner running the 8-step Baker et al. (2025) workflow on a real
BRFSS or geo-experiment panel actually sees. These scripts measure the full
chain (Bacon -> fit -> HonestDiD -> cross-estimator robustness -> reporting)
at data shapes anchored to applied-econ conventions.

## Layout

```
benchmarks/speed_review/
├── README.md                           # this file
├── bench_shared.py                     # timing + pyinstrument + RSS harness
├── run_all.py                          # orchestrator (both backends)
├── bench_campaign_staggered.py         # Scenario 1: CS + 8-step chain
├── bench_brand_awareness_survey.py     # Scenario 2: DiD + SurveyDesign
├── bench_brfss_panel.py                # Scenario 3: aggregate_survey -> CS
├── bench_geo_few_markets.py            # Scenario 4: SDiD + jackknife
├── bench_reversible_dcdh.py            # Scenario 5: dCDH L_max + TSL
├── bench_dose_response.py              # Scenario 6: ContinuousDiD splines
├── mem_profile_brfss.py                # tracemalloc allocator attribution
│                                       #   for BRFSS-1M (standalone)
├── bench_callaway.py                   # pre-existing CS scaling sweep
├── baseline_results.json               # pre-existing CS baseline
└── baselines/                          # this effort's output
    ├── <scenario>_<backend>.json       # phase-level wall-clock + peak RSS
    ├── mem_profile_brfss_large_<backend>.txt   # tracemalloc top-N sites
    └── profiles/                       # flame HTMLs (gitignored)
        └── <scenario>_<backend>.html   # pyinstrument flame output
```

Each JSON baseline records both timing (per-phase wall-clock) and memory
(start/peak/growth from a psutil background sampler at 10 ms). The
`mem_profile_brfss.py` script does a separate tracemalloc pass on the
BRFSS-1M scenario - this is kept out of the main timing harness because
tracemalloc has 2-5x overhead and would contaminate wall-clock baselines.

**Note on profile HTMLs.** pyinstrument flames are ~500KB-1.2MB each and are
regenerated on every run; they live under `baselines/profiles/` which is
gitignored. The key hotspots identified from them are already captured in
the findings doc (top-5 hot phases per scenario); run a scenario locally
to regenerate the full flame when needed.

## Running

```bash
# One-time install
pip install pyinstrument

# All scenarios, both backends, all scales
python benchmarks/speed_review/run_all.py

# One scenario, one backend (the script runs its full scale sweep internally)
DIFF_DIFF_BACKEND=rust python benchmarks/speed_review/bench_campaign_staggered.py

# Subset
python benchmarks/speed_review/run_all.py --scenarios brfss_panel geo_few_markets
```

Multi-scale scenarios write per-scale outputs
(e.g. `campaign_staggered_small_rust.json`, `..._medium_rust.json`,
`..._large_rust.json`). Single-scale scenarios write the scale-free form
(e.g. `dose_response_rust.json`). Full runtime for all scales × both
backends is ~90 seconds on Apple Silicon M4.

## Where to look for findings

[`docs/performance-plan.md`](../../docs/performance-plan.md) - "Practitioner
Workflow Baseline (v3.1.3)" section holds per-scenario hot-phase rankings
and action recommendations. The scenarios here are the measurement surface;
the findings doc is the decision output.

## Adding a scenario

1. Add the scenario definition to `docs/performance-scenarios.md`
   (persona, data shape, operation chain, source anchor).
2. Add `bench_<name>.py` following the existing scripts: build data, define
   `phases` as a list of `(label, callable)` tuples, call `run_scenario`.
3. Register it in `run_all.py`'s `SCRIPTS` dict.
4. Run under both backends and commit the refreshed `baselines/*.json`.
   The `baselines/profiles/*.html` flame HTMLs are gitignored and
   regenerated per run - do not commit them.
5. Add a per-scenario finding paragraph to `docs/performance-plan.md`.
