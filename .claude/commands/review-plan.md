---
description: Review a Claude Code plan file from a staff engineer perspective
argument-hint: "[--updated] [--pr <comment-url>] <path-to-plan-file>"
---

# Review Plan

Review a Claude Code plan file from a staff engineer perspective and provide structured feedback across 8 dimensions. Optionally, when given a `--pr <comment-url>`, also verify the plan covers all feedback items from a specific PR review comment (Dimension 9).

## Arguments

`$ARGUMENTS` may contain:
- **Plan file path** (required): Path to the plan file, e.g., `~/.claude/plans/dreamy-coalescing-brook.md`
- `--updated` (optional): Signal that the plan has been revised since a prior review. Forces a fresh full review and includes a delta assessment of what changed.
- `--pr <comment-url>` (optional): URL of the specific PR comment whose feedback
  the plan addresses. Accepts GitHub comment URLs in any of these formats:
    - `https://github.com/owner/repo/pull/123#issuecomment-456`
    - `https://github.com/owner/repo/pull/123#discussion_r789`
    - `https://github.com/owner/repo/pull/123#pullrequestreview-012`
  Enables branch verification and PR feedback coverage checking (Dimension 9).

Parse `$ARGUMENTS` to extract:
- **--updated**: Split `$ARGUMENTS` on whitespace and check if any token is exactly `--updated`. Remove that token to get the remaining text.
- **--pr**: Check if any token is exactly `--pr`. If found, take the next token as the comment URL and remove both tokens. If `--pr` is found with no following URL, use AskUserQuestion to request it:
```
What is the PR comment URL to check coverage against?

Supported formats:
1. PR-level comment: https://github.com/owner/repo/pull/42#issuecomment-123456
2. Inline review comment: https://github.com/owner/repo/pull/42#discussion_r789012
3. Full PR review: https://github.com/owner/repo/pull/42#pullrequestreview-345678
```
- **Plan file path**: The remaining non-flag tokens after removing `--updated` and `--pr <url>`, joined back together. All flags (`--updated`, `--pr <url>`) are position-independent relative to the path and to each other.
- If no path remains after stripping flags, use AskUserQuestion to request it:
```
Which plan file would you like me to review?

Options:
1. Enter the path (e.g., ~/.claude/plans/plan-name.md)
```

## Constraints

- **Read-only**: Do NOT create, edit, or delete any files. This skill only reads and reports.
- **Advisory-only**: Provide feedback and recommendations. Do not implement fixes.
- **No code changes**: Do not modify any source code, test files, or documentation.
- Use the Read tool for files and the Glob/Grep tools for searching. Do not use Write, Edit, NotebookEdit, or any file-modifying Bash commands.
- The `gh api` calls used with `--pr` are read-only API requests, consistent with the read-only constraint.

## Instructions

### Step 1: Read the Plan File

Read the plan file at the path provided in `$ARGUMENTS`.

If the file does not exist, report the error and stop:
```
Error: Plan file not found at <path>
```

### Step 1b: Handle Re-Review (if `--updated`)

If the `--updated` flag is present, this is a re-review of a revised plan.

**You MUST perform a complete fresh review** — do not skip or abbreviate any steps. Treat the plan file contents as the authoritative source, not your memory of a prior version.

After completing the standard 8-dimension review in Step 4, add a **Delta Assessment** section to the output (see Step 5 template for format). This section compares the revised plan against the prior review's feedback:
- Which previously-raised issues have been addressed?
- Which previously-raised issues remain unresolved?
- Are there any new issues introduced by the revisions?

If no prior review is available in conversation context (e.g., the user passed `--updated` on the first invocation, or the context was compressed), still include the Delta Assessment section but fill each subsection with: "Delta assessment unavailable — no prior review found in conversation context. Full fresh review performed."

### Step 2: Read CLAUDE.md for Project Context

Read the project's `CLAUDE.md` file to understand:
- Module structure and architecture
- Estimator inheritance map
- Development checklists (adding parameters, methodology-critical code, etc.)
- Test structure and conventions
- Design patterns (sklearn-like API, formula interface, results objects, etc.)

If the plan modifies estimator math, standard error formulas, inference logic, or edge-case handling, also read `docs/methodology/REGISTRY.md` to understand the academic foundations and reference implementations for the affected estimator(s).

### Step 2b: Parse Comment URL and Verify Branch (if `--pr`)

Only perform this step when `--pr <comment-url>` was provided. Otherwise skip to Step 3.

**Parse the URL:**
- Strip query parameters from the URL before parsing: remove the query string (the `?...` portion) while preserving the `#` fragment. For example, `https://github.com/o/r/pull/1?notification_referrer_id=abc#issuecomment-123` becomes `https://github.com/o/r/pull/1#issuecomment-123`. If the fragment itself contains `?` (e.g., `#discussion_r123?foo=bar`), strip the `?` and everything after it from the fragment before pattern matching, since GitHub fragments never contain `?` as meaningful data.
- Only `github.com` URLs are supported. If the URL host is not `github.com`, report an error and stop.
- Extract `owner`, `repo`, `pr_number` from the URL path. The `pr_number` is always the path segment immediately after `/pull/`.
- Extract comment type and ID from the fragment:

| Fragment | Type | `gh api` endpoint |
|---|---|---|
| `#issuecomment-{id}` | Issue comment | `repos/{owner}/{repo}/issues/comments/{id}` |
| `#discussion_r{id}` | Inline review comment | `repos/{owner}/{repo}/pulls/comments/{id}` |
| `#pullrequestreview-{id}` | PR review | `repos/{owner}/{repo}/pulls/{pr_number}/reviews/{id}` |

If the URL doesn't match any fragment pattern (including bare PR URLs without a fragment), report:
```
Error: Unrecognized PR comment URL format. Expected a GitHub PR comment URL like:
  https://github.com/owner/repo/pull/123#issuecomment-456
The URL must point to a specific comment, not a PR page.
```

**Verify `gh` CLI availability:**

Run `gh auth status 2>/dev/null` (suppress output on success). If it fails, report a hard error:
```
Error: The --pr flag requires the GitHub CLI (gh) to be installed and authenticated.
Run `gh auth login` to authenticate, then retry.
```

**Verify branch state:**

```bash
gh pr view <number> --repo <owner>/<repo> --json headRefName,baseRefName,title --jq '.'
```

Compare `headRefName` against `git branch --show-current`:
- **Match**: Note "Branch verified" in output.
- **Mismatch**: Emit a warning in the output and note under Dimension 2 (Codebase Correctness) that code references may be inaccurate. Recommend the user checkout the PR branch first (`git checkout <headRefName>`), but do not block the review.

### Step 2c: Fetch the Specific Comment (if `--pr`)

Only perform this step when `--pr` was provided. Otherwise skip to Step 3.

Fetch the comment using the `gh api` endpoint from the table in Step 2b.

**For `pullrequestreview-` URLs**, fetch BOTH the review body AND its inline comments:
```bash
# Review body
gh api repos/{owner}/{repo}/pulls/{pr_number}/reviews/{id} --jq '{body: .body, user: .user.login, state: .state}'

# All inline comments belonging to this review
gh api repos/{owner}/{repo}/pulls/{pr_number}/reviews/{id}/comments --paginate --jq '.[] | {body: .body, path: .path, line: .line, diff_hunk: .diff_hunk}'
```

**For other comment types**, fetch the single comment:

**Issue comment:**
```bash
gh api repos/{owner}/{repo}/issues/comments/{id} --jq '{body: .body, user: .user.login, created_at: .created_at}'
```

**Inline review comment:**
```bash
gh api repos/{owner}/{repo}/pulls/comments/{id} --jq '{body: .body, user: .user.login, path: .path, line: .line, diff_hunk: .diff_hunk}'
```

**Error handling:**
- **404**: `Error: Comment not found at <url>. It may have been deleted or the URL may be incorrect.`
- **403 / other API errors**: `Error: GitHub API returned <status>. You may not have access to this repository, or you may be rate-limited. Check 'gh auth status' and try again.`
- **Empty comment body** (and no inline comments for review types): report and skip Dimension 9:
  ```
  Note: No feedback text found in the comment at <url>.
  Skipping PR Feedback Coverage (Dimension 9). Reviewing plan without PR context.
  ```

The response includes: `body` (comment text), `user.login` (author), `created_at`, and for inline comments: `path` (file), `line` (line number in the file — use `line`, not `position` which is the diff offset, and not `original_line` which is the base branch line), `diff_hunk` (surrounding diff context).

**Extract discrete feedback items** from the comment body:
- For AI review comments (structured markdown with P0/P1/P2/P3 or Critical/Medium/Minor sections): parse each severity section and extract individual items with their labeled severity
- For human comments with numbered/bulleted lists: each list item is one feedback item
- For human comments that are a single paragraph or conversational: the entire comment is one feedback item
- For inline review comments: each comment is one item, with `path` and `line` as its file/line reference
- **Default severity**: when a feedback item has no severity label, treat it as Medium
- Process all feedback items regardless of count

Each feedback item tracks: severity (labeled or default Medium), description, file path (if available), and line reference (if available).

### Step 3: Read Referenced Files

Identify all files the plan references (file paths, module names, class names). When `--pr` was provided, also include files referenced in the feedback comment — inline comment `path` fields and file paths mentioned in the comment body (e.g., `path/to/file.py:L123`). Then read them to validate the plan's assumptions:

**Priority order for reading files:**
1. **Files the plan proposes to modify** — read ALL of these first
2. **Files referenced for context** (imports, call sites, existing patterns) — read selectively to verify specific claims

**Scope restriction:**
- Only read files that are within the project repository (the working directory tree).
- The plan file itself (the `$ARGUMENTS` input) is exempt — it can be anywhere (e.g., `~/.claude/plans/`).
- If the plan references paths outside the repo (home directory configs, SSH keys, `/etc/` files, etc.), do NOT read them. Instead, note in the review output under Dimension 2 (Codebase Correctness) that those external paths were not verified.

**What to verify:**
- File paths exist
- Class names, function signatures, and method names match what the plan describes
- Line numbers (if referenced) are accurate
- The plan's description of existing code matches reality

If the plan references more than ~15 files, use judgment: read all files slated for modification, then spot-check context files as needed rather than reading every one.

### Step 4: Evaluate Across 8 Dimensions

#### Dimension 1: Completeness & Executability

Could a fresh Claude Code session — with no access to the conversation history that produced this plan — execute it without asking clarifying questions?

Check for:
- Are all file paths explicit? (No "the relevant file" or "the test file")
- Are code changes described concretely? (Function signatures, parameter names, not just "add a method")
- Are decision points resolved, not deferred? ("We'll figure out the API later" is a red flag)
- Are there implicit assumptions that require conversation context to understand?

#### Dimension 2: Codebase Correctness

Do file paths, class names, function signatures, and line-number references in the plan match the actual codebase?

Use your findings from Step 3. Flag:
- File paths that don't exist
- Function/class names that are misspelled or don't exist
- Line numbers that point to the wrong code
- Descriptions of existing behavior that don't match reality

#### Dimension 3: Scope

Is the scope right — not too much, not too little?

Check for **missing related changes**:
- Tests for new/changed functionality
- `__init__.py` export updates
- `get_params()` / `set_params()` updates for new parameters
- Documentation updates (README, RST, tutorials, CLAUDE.md)
- For bug fixes: did the plan grep for ALL occurrences of the pattern, or just the one reported?

Check for **unnecessary additions**:
- Docstrings/comments/type annotations for untouched code
- Premature abstractions or over-engineering
- Feature flags or backward-compatibility shims when the code can just be changed

#### Dimension 4: Edge Cases & Failure Modes

For methodology-critical code:
- NaN propagation through ALL inference fields (SE, t-stat, p-value, CI)
- Empty inputs / empty result sets
- Boundary conditions (single observation, single group, etc.)

For all code:
- Error handling paths — are they tested with behavioral assertions (not just "runs without exception")?
- What happens when the feature interacts with other parameters/modes?

#### Dimension 5: Architecture & Patterns

Check against CLAUDE.md conventions:
- Does it respect the estimator inheritance map? (Adding a param to `DifferenceInDifferences` auto-propagates to `TwoWayFixedEffects` and `MultiPeriodDiD`; standalone estimators need individual updates)
- Does it use `linalg.py` for OLS/variance instead of reimplementing?
- Does it follow the sklearn-like `fit()` / results-object pattern?
- Is there a simpler alternative that avoids new abstraction?
- Does it match existing code patterns in the codebase?

#### Dimension 6: Plan Execution Risks

Plan-specific failure modes that wouldn't show up in a code review:

- **Ordering issues**: Does the plan propose changes in an order that would break things mid-implementation? (e.g., changing an import before the module it imports from exists, deleting a function before updating its callers)
- **Ambiguous decision points**: Does the plan defer decisions that should be made now? Vague phrases like "choose an appropriate approach" or "handle edge cases" without specifying which ones
- **Missing rollback path**: For risky changes (public API modifications, data format changes), does the plan consider what happens if something goes wrong?
- **Implicit dependencies**: Does step N assume step M was completed, but this ordering isn't stated?

#### Dimension 7: Backward Compatibility & API Risk

- Does the plan add, remove, or rename public API surface (parameters, methods, classes)?
- If so, does it acknowledge the breaking change and state the versioning decision (deprecation period vs clean removal)?
- Downstream effects on:
  - Convenience functions
  - Re-exports in `__init__.py`
  - Existing tutorials and documentation
  - User code that may depend on the current API

#### Dimension 8: Testing Strategy

- Are tests included in the plan? Do they cover the happy path AND the edge cases from Dimension 4?
- Are test assertions behavioral (checking outcomes) rather than just "runs without exception"?
- For bug fixes: does the plan fix all pattern instances and test all of them?
- Are there missing test scenarios? (Parameter interactions, error paths, boundary conditions)

#### Dimension 9: PR Feedback Coverage (only if `--pr` provided with non-empty comment)

Only evaluate this dimension when `--pr` was provided and a non-empty comment was fetched in Step 2c. For each feedback item extracted in Step 2c, assess:

- **Addressed**: Plan explicitly mentions the issue AND proposes a concrete fix
- **Partially addressed**: Plan touches the area but doesn't fully resolve the feedback
- **Not addressed**: Plan makes no mention of this feedback item
- **Dismissed with justification**: Plan acknowledges the feedback but explains why it won't be acted on (acceptable for Low/P3; flag for Critical/P0)

Use judgment, not just substring matching — the plan may use different words to describe the same fix.

**Verdict impact:**
- Unaddressed P0/P1/Critical items -> automatic "Needs revision"
- Unaddressed P2/Medium items count as Medium issues
- Unaddressed P3/Low items count as Low issues

### Step 5: Present Structured Feedback

Present the review in the following format. Do NOT skip any section — if a section has no findings, write "None." for that section. The Delta Assessment section is only included when the `--updated` flag was provided (see Step 1b). The PR Context and PR Feedback Coverage sections are only included when `--pr` was provided with a non-empty comment.

```
## Overall Assessment

[2-3 sentences: what the plan does, whether it's ready for implementation, and the biggest concern if any]

---

## PR Context (only include if `--pr` was provided with non-empty comment)

**PR**: #<number> - <title> (<owner>/<repo>)
**Branch**: <headRefName> -> <baseRefName>
**Comment**: <comment-url>
**Comment author**: <user.login>
**Feedback items extracted**: N
**Branch match**: Yes / No (warning: recommend `git checkout <headRefName>`)

---

## Issues

### CRITICAL
[Issues that would cause implementation failure, incorrect results, or breaking changes if not addressed. Each issue should include: file path and/or line number if applicable, what's wrong, and a concrete suggestion for fixing it.]

### MEDIUM
[Issues that should be addressed but won't block implementation. Missing test cases, incomplete documentation updates, scope gaps.]

### LOW
[Minor suggestions. Style consistency, optional improvements, things to consider.]

---

## Checklist Gaps

Cross-reference against the relevant CLAUDE.md checklists. List which checklist items are not addressed by the plan.

[Identify which CLAUDE.md checklist applies (e.g., "Adding a New Parameter to Estimators", "Implementing Methodology-Critical Code", "Fixing Bugs Across Multiple Locations") and list any items from that checklist that the plan doesn't cover.]

---

## PR Feedback Coverage (only include if `--pr` was provided with non-empty comment)

### Addressed
- [severity] <description> -- Plan step: <reference to plan section>

### Partially Addressed
- [severity] <description> -- Gap: <what's missing>

### Not Addressed
- [severity] <description>

### Dismissed
- [severity] <description> -- Plan's reason: "<quote>"

| Status | Count |
|--------|-------|
| Addressed | N |
| Partially addressed | N |
| Not addressed | N |
| Dismissed | N |

---

## Questions for the Author

[Ambiguities or missing information that should be clarified before implementation begins. Phrase as specific questions.]

---

## Delta Assessment (only include if `--updated` flag was provided)

### Addressed
[List prior issues that have been resolved in the revised plan]

### Unresolved
[List prior issues that remain. Include the original issue text for reference.]

### New Issues
[List any new issues introduced by the revisions, or "None."]

### PR Feedback Coverage Delta (only include if both `--updated` and `--pr` were provided)

The `--pr` URL must be the same across the initial review and the `--updated` re-review — this compares coverage of the same feedback comment. If the prior review's PR comment URL is no longer available in conversation context (e.g., context compressed), note: "PR coverage delta unavailable — prior PR context not found."

- **Newly addressed**: [list of feedback items now covered that were previously not addressed or partially addressed]
- **Still not addressed**: [list of feedback items still missing]

---

## Summary

| Category | Issues |
|----------|--------|
| Critical | [count] |
| Medium | [count] |
| Low | [count] |
| Checklist gaps | [count] |
| PR feedback gaps | [count of Not Addressed + Partially Addressed] (only if `--pr`) |
| Questions | [count] |

**Verdict**: [Ready / Ready with minor fixes / Needs revision]

- **Ready**: No critical issues, few or no medium issues
- **Ready with minor fixes**: No critical issues, some medium issues that are straightforward to address
- **Needs revision**: Has critical issues or many medium issues that require rethinking the approach
```

## Notes

- This skill is strictly read-only — it does not create, edit, or delete any files
- Plan files are typically located in `~/.claude/plans/`
- The review is displayed directly in the conversation, not saved to a file
- For best results, run this before implementing a plan to catch issues early
- The 8 dimensions are tuned for plan-specific failure modes, not generic code review
- Use `--updated` when re-reviewing a revised plan to get a delta assessment of what changed since the prior review
- Use `--pr <comment-url>` when the plan addresses a specific PR review comment.
  This fetches the comment, extracts feedback items, and checks that the plan
  covers each one. Pairs naturally with `/read-feedback-revise` which creates the plan.
- The `--pr` flag requires the `gh` CLI to be installed and authenticated.
- For best results, run this while on the PR branch so file contents and line
  numbers match what reviewers commented on.
- The comment URL can be copied from the GitHub web UI by right-clicking the
  timestamp on any PR comment and selecting "Copy link".
- For `pullrequestreview-` URLs, both the review body and its inline comments
  are fetched (matching `/read-feedback-revise` behavior).
- Verification of these changes should be done manually before merging the PR
  that adds this feature. See the Verification section of this plan.
