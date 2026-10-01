# Support Runbooks

## Dirty marker on load (crash during save)

Symptom: operations continue normally (recovery is automatic).
Background: every save rotates the previous generation aside first.
Recovery: loads validate the current generation and fall back to the
previous one; the marker clears after. If recovery reports unrestorable,
restore from `brain-backup` output. Never delete `.prev` files by hand
before a successful load.

## Quota breach on teach/correct

Symptom: `ValueError: per-provenance episodic quota exceeded`.
Fix: forget stale entries (`brain-forget`), raise the quota where the
buffer is constructed, or split sources across brains.

## Out of memory on large corpora

Symptom: RSS climbs past several GB on 10 MB+ runs.
Fix: lower `--max-circuits` (E2-pressure precedent: 40k cap costs 0.002
bits/bit), split acquisition into staged runs, or wait for the R5 engine.
Do not raise the ceiling past physical RAM.

## Port already in use on serve

Symptom: serve fails to bind. Fix: `serve --port <free>` (default 8765);
localhost only unless `--host` is set explicitly.

## Corrupt download or import

Symptom: `soma chunk hash mismatch` / footer errors on import.
Fix: re-download; every byte is checksummed, partial files never load.
Verify with the published SHA-256 before importing.

## Wrong answers after teaching

Checklist: `brain-inspect` (entry present? uses counting?), provenance
ranks (a lower-trust teach cannot supersede — see conflicts via inspect
in a future release), interference (other rules with shared triggers?),
restart persistence (reload and re-ask). Roll back with `brain-restore`
from backup, then re-teach with higher provenance.

## Issue taxonomy

Classes: integrity (hash/validation failures), capacity (quota/ceiling/
memory), routing (recall misses, over-firing), calibration (NLL jumps),
policy (refusal/uncertainty disputes), performance (throughput, RSS).
File issues with: command, brain identity hash, `doctor` output, and a
`bundle` artifact (content excluded unless separately consented).
