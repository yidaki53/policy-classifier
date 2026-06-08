---
_agent_frontmatter:
  id: ".githooks/README"
  purpose: "Repository markdown document."
  steward: "repo"
  edit_policy: "manual"
---

Repository git-hooks
===================

This directory contains repository-distributed git hooks. To use these hooks
locally, set the repository `core.hooksPath` to this directory (this is done
automatically by the helper script in this repository):

  git config core.hooksPath .githooks

The `pre-commit` hook enforces that commits use the canonical identity:

  user.name  = yidaki53
  user.email = robinoberg@live.com

If you intentionally need to commit with a different identity for a single
commit, use the `git -c user.name=... -c user.email=... commit ...` form.
