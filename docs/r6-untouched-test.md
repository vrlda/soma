# R6 untouched test: pre-registration

Master plan §22 step 2. **Status: pre-registered, not yet scored.** This
file was committed before any score existed for the chosen book. The git
history is the timestamp.

## Why

The circuit-mixing configuration in [r6-circuit-mixing.md](r6-circuit-mixing.md)
was tuned over about 20 runs with the Time Machine test score printed next
to validation. That makes Time Machine validation-grade for this model. A
clean claim needs a book that no manifest, experiment, or tuning run has
used, scored exactly once.

## Choice of book

**G. K. Chesterton, *The Man Who Was Thursday* (1908)**, Project Gutenberg
#1695, canonical UTF-8 edition
(`https://www.gutenberg.org/files/1695/1695-0.txt`).

- It is absent from the E1, E2, and E3 manifests and from `data/`. The
  guard checks both name and content hash.
- It was never used in any run of this project. Of the book, only the
  NLTK copy's format and size were inspected; no model has scored it.
- Its author differs from every acquisition book, while the register
  (early-20th-century English prose) matches.
- At about 320 KB, it is comparable to the old test book (181 KB).

**Rejected alternative:** the NLTK Gutenberg corpus copy of the same book
(reachable from this environment). It uses ASCII quotes, 3,217 of them
(2,660 `"`, 557 `'`). The 11.5 MB acquisition set contains zero `"` and two
`'`, so scoring that copy would measure novel-byte handling rather than
English modeling. `prepare` refuses such editions.

## Frozen procedure

1. `python3 r6_untouched_test.py prepare --raw 1695-0.txt --name thursday
   --source-url https://www.gutenberg.org/files/1695/1695-0.txt` converts
   CRLF to LF, strips the Gutenberg boilerplate, rejects any prior book,
   and writes `data/e2u/thursday.txt` plus `reports/e2u-manifest.json`: the
   E2 acquisition and validation books unchanged, with the new book as the
   only test partition.
2. `python3 r6_untouched_test.py score` runs every comparison below in one
   pass and writes `reports/r6-untouched-test.json`. It refuses to run if
   that report exists. The report is final whatever it says, and no
   configuration change may be made on its basis and then re-scored on
   this book.

Configuration: `FROZEN_CONFIG` in `r6_untouched_test.py`, written out in
full. It equals the defaults at commit 53b4404 (verified), including the
correction stage that step 3 may later remove.

## Comparisons (same 11.5 MB acquisition, frozen scoring, bits per bit)

- circuit mixing, full budget (2^24 circuits) and compact (2^22)
- prior E2 memory: `r3b_e1_benchmark.py`, protocol r3b-e1-v2, unchanged
- frozen Witten-Bell byte n-grams, orders 2 and 5
- acquisition-fitted byte unigram

## Hypotheses (reported pass or fail)

- **H1:** full circuit mixing beats the order-5 n-gram. (On Time Machine:
  0.2406 vs 0.2575.)
- **H2:** full circuit mixing beats the prior E2 memory. (0.2406 vs 0.3995.)
- **H3:** compact circuit mixing beats the order-5 n-gram. (0.2473 vs 0.2575.)

Absolute scores will differ from Time Machine because the book differs;
the hypotheses concern ordering on the same bytes.

## Blocker

This environment's network policy denies `www.gutenberg.org`. Allowing
that host (or supplying the file) is the only remaining prerequisite.
