# ADR 0010 development log: long-range core

Development measurements on validation data only. The sealed test splits
(`wiki-test`, `squad-test`, `dialog-test`) have not been scored by
anything. Newest first.

## 2026-10-02: stage 3, answering from a passage (SQuAD v1.1 validation)

| Answerer | Training | Exact match | F1 |
|---|---|---|---|
| lexical baseline (no learning) | — | 0.0% | 6.0% |
| longmix greedy generation | 5k questions | 0.0% | 0.6% |
| longmix extractive, sum log-prob | 5k | 3.0% | 5.2% |
| longmix extractive, PMI | 5k | 1.0% | 6.2% |
| span reader, local delta rule | 10k | 30.6% | 37.7% |
| **span reader, local delta rule** | all 87.6k (2,000 validation questions) | **35.9%** | **41.6%** |

Measured on 300 questions, except the reader, which was measured on 500.

- **Free generation** ignores the question. It writes Wikipedia-style text and
  never stops at an answer.
- **Extractive scoring with the byte model** picks short or nearby
  spans. Count circuits learn co-occurrence, not answer types.
- **The span reader** (`research/benchmarks/l3_reader.py`) scores
  candidate spans with hashed evidence features: answer type × span
  shape, rarity-weighted question-word alignment, distance, the tokens
  around the span, and repetition. It learns online with a delta rule:
  each weight changes by its own feature's activation times the
  prediction error. Its typical remaining error is the entity type ("which
  team" → "National Football League"). For reference, the
  feature-based logistic regression in the SQuAD paper (Rajpurkar et al.
  2016) reached 40.4% EM and 51.0% F1, using parsers and POS tags that this
  reader does not have.

## 2026-10-02: stage 2, long-range circuits (Simple English Wikipedia)

20 MB of training text, scored on 1 MB of validation, in bits/byte. The
baseline is the same core with long-range parts off. It reproduces
`soma-mixer` exactly (2.09935 on E1).

| Config | 2^22 | 2^24 | Circuits reclaimed (2^24) |
|---|---|---|---|
| baseline (byte circuits) | 1.9873 | 1.8758 | 21M |
| + match model | **1.8769** | **1.7867** | 21M |
| + word, skip, topic, line circuits | 2.0458 | 1.9083 | 308M |
| + both | 1.9211 | 1.8153 | 308M |

- **The match model** (copying the continuation of the longest earlier
  repeat in the document) gains about 0.09–0.11 bits/byte at both
  budgets. That is well beyond the stage-2 gate of 0.02, though so far
  only on validation.
- **Word-level circuits hurt** at both budgets. They multiply circuit
  churn by about 15×, and gating arbitration by byte circuits with a
  zero-start weight did not fix it (2.0414 at 2^22). The likely cause is
  that their high-cardinality keys starve the byte circuits.
- Confirmed at 2^26: word circuits alone reach 1.8442, against 1.8758 for
  byte circuits at 2^24. The gain is 0.032, but it needs 6.4 GB of RAM
  and 66M circuits, and the match model alone (1.7867 at 2^24, 1.6 GB) is
  better. Word circuits need a cheaper representation (for example
  per-word rather than per-bit keys) before they earn their memory.
