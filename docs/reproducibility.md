# R0 Reproducibility

- Baseline: `python3 -m unittest discover -s tests -v` passes 221/221 pre-R0 (229 post-R0 with new guard tests).
- Final gate: `python3 final_lifetime_benchmark.py` (v15, 6,656 interactions per seed, untouched seeds 32-39).
- Determinism: fixed seeds, independent RNG streams, no wall-clock in state, atomic JSON checkpoints.
- Perf fixture: ~0.210 ms/step, ~645 KB checkpoint after 1,000 steps, 14.3 ms save, 4.3 ms restore. Portable ceilings: 50 ms/step, 20 MiB checkpoint, 5 s restore.
- Archive: `reports/r0-baseline/` holds unittest output, env metadata, source hashes.

## Integrity manifest (`reports/r0-baseline/SHA256SUMS`)

- Policy: every tracked code, test, data, config, format, and report file
  is pinned. Prose (`*.md`) and repository metadata (`.gitignore`,
  `.gitattributes`, `.github/`) are not, so documentation edits never read
  as integrity failures.
- Check: `python3 scripts/verify_hashes.py` must print `missing=0 mismatch=0`.
  Install Git LFS and run `git lfs pull` first; LFS pointer files do not
  match their pinned reports.
- Re-baseline, in order: (1) re-run every frozen benchmark whose code
  changed and confirm its gates and metrics against the committed report;
  (2) run the full unit suite; (3) `python3 scripts/update_hashes.py`
  (it refuses LFS pointers); (4) commit the manifest together with a note
  of what was re-run.
