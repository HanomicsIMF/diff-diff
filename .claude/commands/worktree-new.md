---
description: Create a new git worktree with full dev environment for parallel work
argument-hint: "<name> [base-branch]"
---

# Create Git Worktree

Create an isolated worktree for parallel development. Arguments: $ARGUMENTS

## Instructions

### 1. Parse Arguments

Parse `$ARGUMENTS` to extract:
- **name** (required): First argument — used as both directory suffix and branch name
- **base-ref** (optional): Second argument — existing branch or ref to check out
  instead of creating a new branch

If no name is provided, abort with:
```
Error: Name required. Usage: /worktree-new <name> [base-branch]
Example: /worktree-new feature-bacon-fix
```

Validate that **name** contains only `[a-zA-Z0-9._-]`. If it contains spaces,
slashes, or other shell metacharacters, abort:
```
Error: Name must contain only letters, digits, dots, hyphens, and underscores.
Got: <name>
```

### 2. Resolve Paths

Derive paths dynamically (do NOT hardcode the repo name):

```bash
MAIN_ROOT="$(git worktree list --porcelain | head -1 | sed 's/^worktree //')"
REPO_NAME="$(basename "$MAIN_ROOT")"
PARENT_DIR="$(dirname "$MAIN_ROOT")"
WORKTREE_PATH="${PARENT_DIR}/${REPO_NAME}-<name>"
```

Use `$WORKTREE_PATH` (the absolute path) for all subsequent commands.

### 3. Validate

```bash
git worktree list
```

- If a worktree already exists at `$WORKTREE_PATH`, abort with an error.
- If a branch named `<name>` already exists and no base-ref was given, ask the user
  whether to check out that existing branch or pick a different name.

### 4. Create the Worktree

```bash
# If base-ref provided:
git worktree add "$WORKTREE_PATH" <base-ref>

# If no base-ref (create new branch from current HEAD):
git worktree add -b <name> "$WORKTREE_PATH"
```

### 5. Set Up Python Environment

Note to user: dependency installation may take a moment on a fresh venv.

```bash
python3 -m venv "$WORKTREE_PATH/.venv"
"$WORKTREE_PATH/.venv/bin/pip" install --upgrade pip
"$WORKTREE_PATH/.venv/bin/pip" install -e "$WORKTREE_PATH[dev]"
```

Do NOT use `-q` — let pip output stream so the user sees progress.

### 6. Build Rust Backend (best-effort)

Use `--manifest-path` to avoid changing directories:

```bash
"$WORKTREE_PATH/.venv/bin/maturin" develop --manifest-path "$WORKTREE_PATH/Cargo.toml"
```

If maturin is not installed in the venv or the build fails, note that pure-Python
mode will be used and continue. This is not an error.

### 7. Report

Print:

```
Worktree ready: $WORKTREE_PATH
Branch: <branch>

To start working:
  cd $WORKTREE_PATH && source .venv/bin/activate && claude
```
