# Restore this research pass

The standalone ZIP contains the fresh-start research tree, including code,
protocols, raw results, figures, and checkpoints. Unzip and run the standard
library tests with `python -B -m unittest experiments.test_brain -v`.
It excludes `.git`, archived research, and downloaded corpus text.

The incremental Git bundle preserves the local research commits and requires
base commit `524a0b2abc8d5e2af97931fcd5354c10b12d9a06`. In a clone of `vrlda/soma`:

```sh
git fetch origin claude/youthful-ritchie-vq1bog
git bundle verify /path/to/soma-overnight-incremental.bundle
git fetch /path/to/soma-overnight-incremental.bundle refs/heads/codex/soma-local-learning-2026-10-10:refs/heads/codex/soma-local-learning-2026-10-10
git switch codex/soma-local-learning-2026-10-10
```

Alternatively, on a clean checkout of that base, inspect and apply the binary
patch with `git apply --check /path/to/soma-overnight.patch`, then
`git apply /path/to/soma-overnight.patch`. It contains the changes, not the
commit history. The artifact manifest records SHA-256 hashes and sizes.

GitHub denied writes in this session. Restoring locally does not publish the
branch; an account with repository write permission can publish it afterward.
