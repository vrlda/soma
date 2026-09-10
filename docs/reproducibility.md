# R0 Reproducibility

- Baseline: `python3 -m unittest discover -s tests -v` passes 221/221 pre-R0 (229 post-R0 with new guard tests).
- Final gate: `python3 final_lifetime_benchmark.py` (v15, 6,656 interactions per seed, untouched seeds 32-39).
- Determinism: fixed seeds, independent RNG streams, no wall-clock in state, atomic JSON checkpoints.
- Perf fixture: ~0.210 ms/step, ~645 KB checkpoint after 1,000 steps, 14.3 ms save, 4.3 ms restore. Portable ceilings: 50 ms/step, 20 MiB checkpoint, 5 s restore.
- Archive: `reports/r0-baseline/` holds unittest output, env metadata, source hashes.
