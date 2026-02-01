---
description: Remove a git worktree and optionally delete its branch
argument-hint: "<name>"
---

# Remove Git Worktree

Remove an existing worktree. Arguments: $ARGUMENTS

## Instructions

### 1. Parse Arguments

Extract **name** from `$ARGUMENTS`. If empty, abort:
```
Error: Name required. Usage: /worktree-rm <name>
Tip: Run /worktree-ls to see active worktrees.
```

Validate that **name** starts with a letter or digit, followed by `[a-zA-Z0-9._-]`.
If it starts with `-` or contains invalid characters, abort:
```
Error: Name must start with a letter or digit and contain only letters, digits, dots, hyphens, and underscores.
Got: <name>
```

### 2. Resolve Paths

```bash
MAIN_ROOT="$(git worktree list --porcelain | head -1 | sed 's/^worktree //')"
REPO_NAME="$(basename "$MAIN_ROOT")"
PARENT_DIR="$(dirname "$MAIN_ROOT")"
WORKTREE_PATH="${PARENT_DIR}/${REPO_NAME}-<name>"
```

### 3. Validate

```bash
git worktree list
```

If no worktree exists at `$WORKTREE_PATH`, abort:
```
Error: No worktree found at $WORKTREE_PATH
Active worktrees: <list them>
```

### 4. Check for Uncommitted Work

Capture the branch name first (needed for step 6, before the worktree is removed):

```bash
BRANCH=$(git -C "$WORKTREE_PATH" rev-parse --abbrev-ref HEAD)
```

Then check for uncommitted changes:

```bash
git -C "$WORKTREE_PATH" status --porcelain
```

If there are uncommitted changes, warn the user with AskUserQuestion:
- Option 1: "Abort — I have unsaved work"
- Option 2: "Remove anyway — discard changes"

**If the user chooses "Abort", stop immediately. Do NOT continue to step 5.**

### 5. Remove the Worktree

If the worktree had uncommitted changes and the user chose "Remove anyway":
```bash
git worktree remove "$WORKTREE_PATH" --force
```

If the worktree was clean (step 4 found no changes):
```bash
git worktree remove "$WORKTREE_PATH"
```

### 6. Try to Delete the Branch

Only attempt branch deletion if `$BRANCH` equals `<name>` (meaning we created it
via `/worktree-new <name>` without a base-ref). If the branch is something else
(e.g., `feature/existing-branch`), skip deletion — the user didn't create it.

#### 6a. Fetch origin (best-effort)

```bash
git fetch origin --quiet 2>/dev/null || true
```

This updates tracking refs so `git branch -d` has better merge detection. Scoped
to `origin` to avoid fetching all remotes. Fails silently if offline or no remote.

**If this step hangs** (network issues), Ctrl-C and continue — it is not required.

#### 6b. Check GitHub PR status

```bash
gh pr list --head "$BRANCH" --state merged --json number --jq '.[0].number' 2>/dev/null
```

Note: `gh pr list --head` queries by branch name, avoiding the `gh pr view` pitfall
where numeric branch names (e.g., `1234`) are interpreted as PR numbers.

Using `--state merged` directly queries for merged PRs, avoiding ambiguity when
multiple PRs share the same head branch (e.g., one open and one merged).

Interpret by checking the **exit code first**, then the output:

1. Command **exits non-zero** (gh not installed, auth/network error) → `PR_MERGED=false`
2. Command **exits zero** and output is **non-empty** (a merged PR exists) → `PR_MERGED=true`
3. Command **exits zero** and output is **empty** (no merged PR for this branch) → `PR_MERGED=false`

#### 6c. Delete the branch

- **`PR_MERGED=true`**: Check for unpushed local commits before force-deleting:
  ```bash
  UPSTREAM=$(git rev-parse --abbrev-ref "$BRANCH@{u}" 2>/dev/null)
  ```
  - If upstream exists, compute `AHEAD=$(git rev-list --count "$UPSTREAM..$BRANCH")`.
    - `AHEAD == 0`: Force-delete with `git branch -D -- "$BRANCH"`.
      Report: "Branch `$BRANCH` deleted (PR was merged on GitHub)."
    - `AHEAD > 0`: Safe-delete with `git branch -d -- "$BRANCH"`. If it fails,
      warn about unpushed local commits and suggest `git branch -D -- "$BRANCH"`.
  - If no upstream: Safe-delete with `git branch -d -- "$BRANCH"`. If it fails,
    warn about unpushed local commits and suggest `git branch -D -- "$BRANCH"`.
- **`PR_MERGED=false`**: Safe-delete with `git branch -d -- "$BRANCH"`.
  If it fails because the branch is not fully merged, relay the warning and
  suggest `git branch -D -- "$BRANCH"` if they want to force-delete.

### 7. Report

```
Removed worktree: $WORKTREE_PATH
[Branch status from step 6]
```
