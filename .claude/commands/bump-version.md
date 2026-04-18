---
description: Update version numbers across codebase and ensure CHANGELOG is populated
argument-hint: "<version> (e.g., 2.2.0)"
---

# Bump Version

Update version numbers across the codebase and ensure CHANGELOG is properly populated for a new release.

## Arguments

The user must provide a version number: `$ARGUMENTS`

- If empty or not provided: Ask the user for the target version
- Otherwise: Use the provided version (must match semver pattern X.Y.Z)

## Version Locations

Files that need updating:

| File | Format | Line |
|------|--------|------|
| `diff_diff/__init__.py` | `__version__ = "X.Y.Z"` | ~134 |
| `pyproject.toml` | `version = "X.Y.Z"` | ~7 |
| `rust/Cargo.toml` | `version = "X.Y.Z"` | ~3 |
| `CHANGELOG.md` | Section header + comparison link | Top + bottom |
| `diff_diff/guides/llms-full.txt` | `- Version: X.Y.Z` | ~5 |
| `CITATION.cff` | `version: "X.Y.Z"` + `date-released: "YYYY-MM-DD"` | ~10, ~11 |

## Instructions

1. **Parse and validate version**:
   - If no argument provided, use AskUserQuestion to get the target version
   - Validate format matches semver pattern `X.Y.Z` (e.g., `2.2.0`, `3.0.0`, `1.10.5`)
   - If invalid, ask user to provide a valid version

2. **Get current version**:
   - Read `diff_diff/__init__.py` and extract the current `__version__` value
   - Store as `OLD_VERSION` for comparison link generation

3. **Check CHANGELOG entry and resolve `RELEASE_DATE`**:
   - Search `CHANGELOG.md` for `## [NEW_VERSION]` section header.
   - If found with content (at least one `### Added/Changed/Fixed` subsection with
     bullet points):
     - **Parse the existing header date** (e.g., `## [3.1.3] - 2026-04-19` → `2026-04-19`).
       Store as `RELEASE_DATE` and skip to step 5.
     - If the header has no date (malformed), abort with: `Error: CHANGELOG header for
       [NEW_VERSION] is missing a date. Fix the header before re-running.`
   - If not found or empty: Set `RELEASE_DATE` to today's date in `YYYY-MM-DD` format,
     then continue to step 4.

   `RELEASE_DATE` is the single source of truth for the release date across every file
   touched in this bump. Do not recompute it downstream.

4. **Generate CHANGELOG from git** (only if needed):
   - Run: `git log v{OLD_VERSION}..HEAD --oneline`
   - If no tag exists, use: `git log --oneline -50`
   - Categorize commits using these heuristics:
     - **Added**: commits containing "add", "new", "implement", "introduce", "create"
     - **Changed**: commits containing "update", "change", "improve", "optimize", "refactor", "enhance"
     - **Fixed**: commits containing "fix", "bug", "correct", "repair", "resolve"
   - Use the `RELEASE_DATE` resolved in step 3 for the header.
   - Create CHANGELOG entry in this format:
     ```markdown
     ## [X.Y.Z] - YYYY-MM-DD

     ### Added
     - Feature description from commit message

     ### Changed
     - Change description from commit message

     ### Fixed
     - Fix description from commit message
     ```
   - Only include sections that have commits (omit empty sections)
   - Insert the new entry after the changelog header (after the "adheres to Semantic Versioning" line)

5. **Update version in all files**:
   Use the Edit tool to update each file:

   - `diff_diff/__init__.py`:
     Replace `__version__ = "OLD_VERSION"` with `__version__ = "NEW_VERSION"`

   - `pyproject.toml`:
     Replace `version = "OLD_VERSION"` with `version = "NEW_VERSION"`

   - `rust/Cargo.toml`:
     Replace `version = "OLD_VERSION"` (the first version line under [package]) with `version = "NEW_VERSION"`
     Note: Rust version may differ from Python version; always sync to the new version

   - `diff_diff/guides/llms-full.txt`:
     Replace `- Version: OLD_VERSION` with `- Version: NEW_VERSION`

   - `CITATION.cff`:
     Replace `version: "OLD_VERSION"` with `version: "NEW_VERSION"`.
     Also update `date-released: "OLD_DATE"` to `date-released: "{RELEASE_DATE}"`
     using the `RELEASE_DATE` resolved in step 3. Both fields are quoted strings;
     preserve the quoting style. `RELEASE_DATE` must match the CHANGELOG header
     date; never substitute a freshly computed "today" value here.

6. **Update CHANGELOG comparison links**:
   - Run `git remote get-url origin` to determine the repository's GitHub URL
     (strip `.git` suffix, convert SSH format to HTTPS if needed)
   - At the bottom of `CHANGELOG.md`, after `[OLD_VERSION]:`, add the new comparison link:
     ```
     [NEW_VERSION]: https://github.com/OWNER/REPO/compare/vOLD_VERSION...vNEW_VERSION
     ```
     using the owner/repo derived from the remote URL.

7. **Report summary**:
   Display a summary of all changes made:
   ```
   Version bump complete: OLD_VERSION -> NEW_VERSION

   Files updated:
   - diff_diff/__init__.py: __version__ = "NEW_VERSION"
   - pyproject.toml: version = "NEW_VERSION"
   - rust/Cargo.toml: version = "NEW_VERSION"
   - diff_diff/guides/llms-full.txt: Version: NEW_VERSION
   - CITATION.cff: version: NEW_VERSION, date-released: YYYY-MM-DD
   - CHANGELOG.md: Added/verified [NEW_VERSION] entry

   Next steps:
   1. Review changes: git diff
   2. Commit: git commit -am "Bump version to NEW_VERSION"
   3. Tag: git tag vNEW_VERSION
   4. Push: git push && git push --tags
   ```

## Notes

- The Rust version in `rust/Cargo.toml` is always synced to match the Python version
- If CHANGELOG already has the target version entry with content, it will not be overwritten
- Commit messages are cleaned up (prefixes like "feat:", "fix:" are removed) for CHANGELOG
- The comparison link format uses `v` prefix for tags (e.g., `v2.2.0`)
- `CITATION.cff` `date-released` and the `CHANGELOG.md` section header share a single
  `RELEASE_DATE` resolved in step 3: if the CHANGELOG entry was pre-populated, its
  existing header date wins (so pre-written changelog drafts don't silently drift
  from the CITATION date); otherwise today's date is used for both. If the release
  is cut on a different day than the bump, update both surfaces manually — drift
  causes auto-citation tools (Zenodo, GitHub's "cite this repository", reference
  managers) to report stale metadata.
