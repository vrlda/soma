# Small candidate brain: pre-registration

Protocol `r8-small-candidate-v1` (`r8_small_candidate_eval.py`). Master
plan §22 step 12, ADR 0006 item 2. **Status: run 2026-10-01; all 8 gates pass. Ready for the owner to sign and publish.**
The git history timestamps this file, the harness, the build script, and
the sealed book before the build.

## The candidate

- **Build:** `research/scripts/build_small_candidate.py`. It creates the canonical
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
  ([r8-signing.md](../../docs/r8-signing.md)) after this evaluation. Here a
  throwaway key exercises the flow.

## Sealed book

`research/reports/e3u2-manifest.json`: **pg78576**, *The lost bride; vol. 2*
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

## Harness smoke test (before the run)

The smoke test used a development brain trained on E1 and the spent book pg76222. It exercised every
gate. The E2 control scored 0.2656 there, exactly the retention-policy
result for that configuration (a reproduction check). H1 failed in the
smoke test, as expected, because E1 is less data than E2.

## Result (`research/reports/r8-small-candidate.json`)

Sealed book pg78576, scored once per model (bits/bit):

| Model | Score |
|---|---|
| **Small candidate** (E3-full, 2^24) | **0.2161** (1.73 bits/byte) |
| same config trained on E2 | 0.2343 |
| order-5 Witten-Bell n-gram trained on E2 | 0.2574 |
| byte unigram (E3-full) | 0.5681 |

| Gate | Result | Value |
|---|---|---|
| H1 data | pass | −0.0182 against the E2-trained config |
| H2 n-gram reference | pass | −0.0413 (margin 0.02 required) |
| P1 memory | pass | 1.50 GB engine RSS while chatting (training peak 1.65 GB) |
| P2 cold turn | pass | 7.0 s |
| P3 warm turns | pass | median 0.49 s (0.41–0.67 s) |
| P4 resume | pass | both builds give memory `c4020919…1336e` |
| P5 signed round trip | pass | verified; imported memory identical |
| P6 replies | pass | e.g. "I am not afraid of him, " |

Build: 45.6 min on one core. 16.6M circuits, 402.7M created and 386.0M
reclaimed, over 866.9M bits. Artifact `soma-small-candidate.soma`:
466,466,017 bytes, SHA-256 `2db80d83…a7fd`. That hash covers this
build's creation time, so a rebuild matches on the memory hash and brain
identity, not on the `.soma` hash.

**What this does and does not show.** On a book never used by anyone,
the candidate predicts English 0.018 bits/bit better than the same memory
with a tenth of the data, and 0.041 better than a conventional order-5
n-gram. It runs chat in about half a second within the Small RAM budget,
and it reproduces exactly. Replies are fluent fragments of 19th-century
fiction. It is a statistical text predictor that speaks in the style of
its corpus. It is not an assistant that answers questions; that is R7's
job, and the human-rating floors (step 11) are still open.

**Publishing.** The artifact is not in git (466 MB). The owner signs it
with the release key, attaches it and its `.sig` to a GitHub release, and
adds the public key to `configs/trusted_keys.json`. Users then run
`brain-download <url> small`. Anyone can rebuild it with
`research/scripts/build_e3_full_corpus.py fetch` and
`research/scripts/build_small_candidate.py --root <dir>` and compare memory
hashes.
