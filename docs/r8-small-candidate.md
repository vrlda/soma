# Small candidate brain: pre-registration

Protocol `r8-small-candidate-v1` (`r8_small_candidate_eval.py`). Master
plan §22 step 12, ADR 0006 item 2. **Status: pre-registered; not yet run (harness smoke test on a spent book in progress).**
The git history timestamps this file, the harness, the build script, and
the sealed book before the build.

## The candidate

- **Build:** `scripts/build_small_candidate.py`. It creates the canonical
  blank brain (`BrainStore.create`, seed 0, `--memory mixing`) and trains
  its engine memory on the E3-full acquisition: 233 books, 108.4 MB, in
  manifest order, one document each.
- **Budget:** 2^24 circuits with current defaults (retention policy
  adopted in [r6-retention-policy.md](r6-retention-policy.md)).
- **Size against the Small preset:** the preset's frozen RAM budget is
  8 GB. A 2^24 brain needs about 1.6 GB RSS and a 439 MB state. This
  exceeds the 32 MiB state ceiling of the `r6-tier-v2` harness, which was
  set for the old suffix memory, and the candidate does not claim that
  ceiling.
- **Lineage:** the brain manifest records the corpus manifest hash, the
  blank and trained memory hashes, and the engine config. Training is
  deterministic, so the memory hash is the reproducibility check. A second
  build saves after book 116 and resumes from the file.
- **Signing:** the `.soma` is signed with the owner's release key
  ([r8-signing.md](r8-signing.md)) after this evaluation. Here a
  throwaway key exercises the flow.

## Sealed book

`reports/e3u2-manifest.json`: **pg78576**, *The lost bride; vol. 2*
(Georgiana Chatterton), 206,961 bytes after stripping. It is the next book
in E3-full's seeded order after pg76222. pg76222 was used to decide the
retention policy, so it cannot certify this candidate. No E3-full
training book shares this book's title or author, and the same holds
for pg76222. Only its metadata has been printed.

## Gates

| Gate | Rule |
|---|---|
| H1 data | candidate < the same config trained on E2 (11.5 MB) on the sealed book |
| H2 conventional reference | candidate ≤ order-5 Witten-Bell byte n-gram trained on E2 − 0.02 (the n-gram saw less data; a pure-Python n-gram over 108 MB does not fit in memory) |
| P1 memory | engine peak RSS while serving chat ≤ 8192 MB (Small preset) |
| P2 cold turn | first chat turn, including loading the brain, ≤ 60 s |
| P3 warm turns | median of the next five turns ≤ 2.0 s |
| P4 resume | the split build's memory hash equals the straight build's |
| P5 signed round trip | sign, `brain-import-soma --require-signature`, identical memory |
| P6 replies | every chat reply is nonempty |

Reported, not gated: the byte unigram (E3-full) on the sealed book, the
artifact size and hash, every turn time, and the replies.

**Decision:** all gates pass → the candidate is ready for the owner to
sign and publish as a release asset. Any failure is recorded, and the
candidate is not published.

Before this pre-registration, chat on large brains was fixed (reuse the
live engine; link an unchanged memory instead of rewriting it): a turn
took 4–5 s on a 239 MB development brain and now takes 0.38 s. The P2/P3
thresholds were set after that measurement, on the development brain,
not on the candidate.
