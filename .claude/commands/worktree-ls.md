---
description: List all active git worktrees with status
argument-hint: ""
---

# List Git Worktrees

## Instructions

### 1. Get Worktree List

```bash
git worktree list
```

### 2. Check Status of Each

For each worktree path returned, check for uncommitted changes:

```bash
git -C <path> status --porcelain | wc -l
```

### 3. Display Results

Show a table with:
- **Path**
- **Branch**
- **Commit** (short hash)
- **Status**: "clean" or "N uncommitted changes"

If there's only the main worktree, add:
```
No additional worktrees. Use /worktree-new <name> to create one.
```
