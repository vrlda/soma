# ADR 0010: a long-range SOMA core for useful English and questions

- Status: accepted 2026-10-02 (owner decision: build a new SOMA core rather
  than retrieval-only or a conventional hybrid).
- Context: the circuit-mixing memory (ADR 0009) predicts the next bit from
  at most 12 preceding bytes. On 108 MB of fiction it reaches 1.73
  bits/byte on a sealed book, but it cannot answer a question. The question
  leaves its context after a dozen characters, and its corpus is 19th-century
  fiction. More data alone fixes neither problem.

## Decision

Build the core in measured stages. Each stage keeps SOMA's rules:

- No Transformer and no backpropagation through time.
- Learning stays local: counts, a delta rule, plastic arbitration.
- Structure grows from evidence under a hard budget.
- It stays one persistent model that learns online.

**Stage 1 — modern data.** A reproducible, licensed, hashed corpus:

| Source | License | Use |
|---|---|---|
| Simple English Wikipedia (2026 dump) | CC BY-SA 4.0 | knowledge and clear modern prose |
| SQuAD v1.1 | CC BY-SA 4.0 | passage-question-answer triples |
| OpenAssistant oasst1 | Apache 2.0 | human assistant dialogue |

Sealed test sets are split off by a fixed rule before any model sees them.

**Stage 2 — long-range circuits.** The model adds circuits keyed by
longer-timescale context:

- **Word contexts:** the last 1–3 whole words plus the current word prefix.
- **Sparse contexts:** pairs of salient words, for example a
  question word and the current answer position.
- **Recurrent trace state:** leaky traces of recent content words at
  several decay rates. This is fixed recurrent dynamics, so it needs no
  backpropagation through time.

All of them feed the existing plastic arbitration, which learns per
context which evidence to trust.

**Stage 3 — copy circuits.** A pointer into the active workspace (the
passage or conversation) aligns on the question's content words and
predicts the bytes that follow the aligned position there. This is how an
answer present in the input becomes reachable. Arbitration learns when to
trust it.

**Stage 4 — answering behaviour.** The model trains on formatted
passages, questions, answers, and dialogue turns, and decodes with an
answer-length stop.

## Gates (decided before results)

| Stage | Gate |
|---|---|
| 2 | On held-out modern text, beat the 12-byte circuit model trained on the same data at the same budget by at least 0.02 bits/byte |
| 3 | On held-out SQuAD questions (open book), beat a non-learning lexical-overlap baseline on exact match and F1 |
| 4 | In held-out conversations, produce answers rated useful against simple references (first automatic, then human: plan step 11) |

Each gate failure is recorded. No stage is relabelled as passing.

## Consequences

- Retrieval of whole documents is not the mechanism. The copy circuit
  reads only the active workspace, as ADR 0003 defines it.
- Word segmentation is adapter-level. Words are hashed byte spans split
  at whitespace and punctuation, so there is no fixed vocabulary.
- If stage 2 fails its gate, the trace and sparse designs are revisited
  before any scale-up.
