# docs-refresh — Briefing

## The goal

Two-part documentation sweep, sequenced as one initiative across multiple PRs:

1. **README.md aggressive trim**
2. **RTD staleness audit + targeted fixes**

Tutorial work is OUT OF SCOPE — that's a separate worktree (`dcdh-tutorial`).

## Why now

Recent releases (3.0.x → 3.3.0) shipped a lot of new surface area without
proportional README/RTD updates:

- HeterogeneousAdoptionDiD (entirely new estimator, multi-phase)
- profile_panel() + llms-autonomous.txt
- dCDH by_path + R parity
- SDiD survey support across all three variance methods
- BR/DR target_parameter (schema 2.0)
- TROP backend parity

README is too long for skim consumption (SEO + first-impression problem).
RTD likely has stale pages, missing API references, and outdated examples.

## Sequencing

### PR 1 — README aggressive trim
Target a tight shape:
- One-line value prop
- Install (`pip install diff-diff`)
- Minimal working example (5-10 lines, one estimator)
- Estimator-list one-liner with link to RTD for full reference
- Citation + license

Aggressive cuts. Anything that belongs on RTD goes to RTD (or stays there if
already there). Don't try to be the docs.

Out of scope: rewriting RTD content that the README links to.

### PR 2+ — RTD staleness audit + fixes

Audit step (read-only):
- Walk `docs/` and identify pages missing post-3.0.x estimators / surfaces
- Cross-reference `docs/doc-deps.yaml` to surface known dependency drift
- Categorize: missing API page, stale example, broken link, outdated narrative

Then fix in scoped PRs (one PR per coherent batch — e.g., "Add HAD API reference
+ choosing-estimator entry", "Refresh practitioner decision tree for 3.3.0").

## What to read first

- `README.md` (current state, length)
- `docs/index.rst` (RTD entry point)
- `docs/doc-deps.yaml` (source-to-doc dependency map)
- `docs/api/` (API reference pages — what's missing)
- `docs/methodology/REGISTRY.md` (don't reformat; just cross-check it's
  referenced from RTD where appropriate)
- `CLAUDE.md` "Documenting Deviations" section (label patterns, don't violate)

## Memory rules to honor

- Hyphens, not em dashes (writing style)
- No competitor mentions in formal docs (ROADMAP / user-facing)
- No version numbers as RTD section headings
- diff-diff perspective (not neutral comparisons)
- Tutorial-scope discipline does NOT apply here — this is reference docs

## Out of scope

- New tutorials (separate `dcdh-tutorial` worktree owns DCDH; HAD tutorial queued after)
- ROADMAP.md restructuring (separate concern)
- BR/DR positioning beyond "experimental preview" framing (per memory)

## Cleanup note

This BRIEFING.md was accidentally committed to main from a prior worktree
session. Long-term, drop it from main and add to .gitignore so worktree
briefings stay local.
