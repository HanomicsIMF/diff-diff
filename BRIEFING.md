# dcdh-by-path — Briefing

## The ask

Clément de Chaisemartin (dCDH author) suggested implementing the `by_path`
option from R's `did_multiplegt_dyn`. It disaggregates the dynamic event-study
by observed treatment trajectory so practitioners can compare paths like:

- `(0,1,0,0)` — one pulse
- `(0,1,1,0)` — two periods on, then off
- `(0,1,1,1)` — three periods on, then off
- `(0,1,0,1)` vs `(0,1,1,0)` — sequencing

Use case: "is a single pulse enough, or do you need sustained exposure?"

## Where we stand today

`diff_diff/chaisemartin_dhaultfoeuille.py` implements `ChaisemartinDHaultfoeuille`.

- Supports reversible on/off treatments (the only estimator in the library
  that does)
- **Currently drops multi-switch groups by default** (`drop_larger_lower=True`) —
  exactly the groups `by_path` wants to keep and compare
- Stratifies by direction cohort (`DID_+`, `DID_-`, `S_g = sign(Δ)`) but not
  by trajectory
- No `by_path`, `treatment_path`, or path-enumeration code exists anywhere
- Not on ROADMAP.md; not in TODO.md

## Shape of the work

1. Parameter: likely `by_path: bool = False` (implies `drop_larger_lower=False`)
2. Enumerate unique treatment histories `(D_{g,1}, …, D_{g,T})` per group;
   optionally accept a user-specified subset of paths of interest
3. Per-path `DID_{g,l}` aggregation with influence-function SEs per path
4. Result container extension: `path_effects` dict keyed by trajectory tuple,
   each holding ATT + SE + CI vectors
5. Decide interaction with `drop_larger_lower`: probably forbid both being
   non-default simultaneously, or have `by_path` override
6. REGISTRY.md section on path-heterogeneity methodology + deviation notes
7. Methodology reference: `did_multiplegt_dyn` manual §on `by_path`; dCDH
   dynamic paper for the `DID_{g,l}` building block (already cited in REGISTRY)

## Open methodology questions (for plan mode)

- Which paths are enumerable? All observed, or user-specified subset only?
  R's default behavior on cardinality control is worth checking.
- How does path stratification interact with the current cohort pooling
  `(D_{g,1}, F_g, S_g)` used for variance recentering — does it still apply
  per path?
- Placebo and TWFE diagnostics: compute per-path or overall only?
- Bootstrap interaction: per-path bootstrap blocks vs single bootstrap with
  per-path aggregation

## Before starting

- Pull the R manual section on `by_path` for `did_multiplegt_dyn` — the option
  spec there is load-bearing; don't infer from usage examples alone
- Methodology changes: consult `docs/methodology/REGISTRY.md` first
- New estimator surface → budget ~12-20 CI review rounds
