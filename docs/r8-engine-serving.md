# Serving the language memory from the Rust engine

Master plan §22 step 8. Brains created with `--memory mixing` keep their
language memory (circuit mixing, ADR 0009) in a Rust engine process. Chat,
generation, teaching, quarantine, and persistence run through it.

## How it works

- `soma-mixer-serve` (`engine/soma-engine/src/bin/`) is a long-lived
  process speaking line-delimited JSON on stdin/stdout. It supports new,
  load, save (atomic rename), reset, observe (with trust weight and an
  optional prediction in the same reply), distribution, summary, and quit.
  A panic, for example on an invalid configuration, becomes an error reply
  rather than a crash.
- `soma.memory.engine.EngineMixingMemory` is the Python client, with the
  same interface as the reference. It buffers observed bits and sends them
  in one request when a prediction is needed, so a prompt costs one round
  trip.
- A brain directory gains `memory.somamix`, which `brain.json` references by
  relative name. Clone, backup, restore, and tar export copy it. `.soma`
  export/import carries it as an optional binary chunk; older `.soma` files
  still read. Crash-safe rotation covers it.
- CLI: `brain-create NAME [--memory auto|mixing|suffix] [--max-circuits N]`.
  The default `auto` picks circuit mixing when the engine is built.
  `install.sh` builds the engine when `cargo` is present (`SOMA_SKIP_ENGINE=1`
  opts out). `doctor` reports engine availability and a live round trip,
  and flags engine brains whose engine is missing.

## Verification

- `engine/differential_serve.py`: seeded random tapes of learning, frozen
  observation, trust weights, resets, and predictions. **0 mismatches in
  2,556 predictions** against the Python reference, byte-identical saved
  state, and identical reloads.
- **Frozen gates on the engine-served memory** (`--memory mixing`, product
  decoding settings):

  | Gate | Result | Note |
  |---|---|---|
  | R3C generation | 4/4 | continuation NLL 0.283 (old memory 0.413); horizon test at 12 bytes, beyond-horizon spread exactly 0; lesion = no learned circuits or weights; no self-training checked on the saved learned state, byte for byte |
  | R3D dialogue | 6/6 | controls still fail as required |
  | R7 instruction | 5/5 | |
  | R7 tools | 4/4 | |
  | R4 lifelong | 6/6 | Python reference memory (step 7) |
  | R9 red-team | 7/7 | Python reference memory (step 7) |

  Without the flag, all eight affected gate scripts reproduce their frozen
  reports byte-for-byte.
- `tests/test_r8_engine_brain.py`: client parity, error handling, state
  round trip, and the full brain lifecycle (create, teach, correct, chat,
  clone, backup, `.soma` export/import, crash recovery). It skips when the
  engine is not built and runs in `engine/tests.sh`.

## Latency (development container, one core)

| Operation | Time |
|---|---|
| condition on a 200-byte prompt | 1.2 ms |
| generate 24 bytes, bit by bit | 8.3 ms (about 43 µs per round trip) |
| teach the 151 KB E0 corpus | 1.2 s end to end |
| inference inside a chat turn (route, recall, 40-byte generation) | about 0.2 s |
| **full chat turn** | **0.4–0.9 s, median 0.70 s** (was 2.1–2.5 s) |

The rest of a turn is the turn-scoped dialogue tier, an order-256
`SequenceCircuitMemory` saved on every turn. It was 2.3 s and is now about
0.5 s, through four behavior-preserving changes:

- **Reclamation index.** A saturated 16K-circuit table rescanned and fully
  sorted every circuit about 900 times per turn. A min-heap of eligible
  circuits plus per-event buckets now yields exactly the same batches, in
  the same order. It is checked against a verbatim copy of the old scan
  under unobserve, chunk promotion, save/load, and direct count edits
  (`tests/test_sequence_memory_performance.py`). The heap is compacted by
  an exact rebuild when stale entries outnumber live circuits. A first
  version without compaction leaked memory on long runs: the R6
  reclamation benchmark was killed at 41 minutes. A test now bounds it.
- **State format v2.** Binary-symbol contexts are stored as bit strings,
  not integer lists, which halves `dialogue.json`. v1 states still load.
- **The C JSON encoder** (`json.dumps`; identical text), a cached
  protected-context set, and no duplicate validation.
- **No reload** after save in `chat_turn`.

All eight frozen gate reports that use this memory reproduce
byte-for-byte. So does the R6 reclamation benchmark, which runs this code
under constant forced eviction (`reports/r6-reclamation.json` unchanged).
It now finishes in 2,480 s instead of 4,538 s, with memory flat at about
200 MB.

## Chat quality fix found on the way

`respond()` handed generation to the turn-scoped dialogue table whenever it
matched 8 bits of context, and sampled at temperature 1 bit by bit. With
that table holding only taught facts and the current prompt, free
generation produced byte garbage on the old memory too. Engine brains now
use **evidence arbitration**, which prefers the dialogue table only when
its context is longer than the background memory's, plus deterministic
decoding. "Alice was beginning to get very" now continues "glad the door
mouthought to be and the c", where it used to emit random bytes. Both
switches are opt-in parameters, so the frozen gates keep their recorded
behavior.

## Open

- The engine is built from source by `install.sh`, or installed as a
  signed prebuilt package with `engine-install`
  ([r8-signing.md](r8-signing.md)).
- Generation is fluent locally and repetitive over longer spans ("the more
  the whole she was"). That is expected from a 12-byte-context predictor
  trained on 127 KB. Quality scales with corpus and budget (E3), not with
  serving.
