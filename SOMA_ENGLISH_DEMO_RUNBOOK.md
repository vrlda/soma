# SOMA First English Brain Demonstration Runbook

**Status:** normative execution design; commands marked `TARGET INTERFACE` become executable as R2–R3 implement them  
**Scope:** train one blank universal SOMA descendant into a small English dialogue demonstration brain on the development MacBook  
**Not a claim:** this runbook does not assume that present v15 code can learn language or that the resulting demonstration will be consumer-grade

## 1. Demonstration objective

Starting from the canonical blank-brain genesis state, produce a downloadable `.soma` artifact that:

- learned through the universal event/outcome lifecycle;
- predicts unseen English better than declared baselines;
- acquired causally useful higher-level sequence circuits;
- generates valid, recognizably learned English sequences;
- performs elementary attributed dialogue and accepts corrections;
- retains earlier learning after later experience;
- continues the same lifetime after restart, backup, restore, and download;
- remains within the MacBook's frozen resource envelope;
- contains no language-specific branch in the universal learning core.

The run is successful only if behavioral, causal, persistence, and resource gates pass together.

## 2. Fixed terminology

| Term | Meaning |
|---|---|
| Blank brain | Canonical domain-neutral genesis state |
| Qualified generation | Immutable brain generation that passed a stage gate |
| Experimental branch | Mutable descendant used for one declared run |
| Recorded environment | Data source producing ordered external events and outcomes |
| Text transducer | Reversible UTF-8 byte/event boundary adapter |
| Brain | Persistent topology, numeric state, memories, evidence, workspace checkpoint, and lineage |
| Runtime | Separately installed code that executes the brain |
| Exposure | One ordered presentation of a declared acquisition partition |
| Untouched test | Data never used for selection, thresholds, debugging, or stopping |

## 3. Development hardware envelope

The initial reference machine is:

- Apple MacBook Pro `Mac15,6`;
- Apple M3 Pro;
- 11 CPU cores;
- 14 integrated GPU cores;
- 18 GB unified memory;
- 106 GB free storage measured when this runbook was created.

Initial hard budgets:

| Resource | Budget |
|---|---:|
| Resident brain state | 8 GB target, 10 GB hard ceiling |
| Runtime and active workspace | 3 GB target, 4 GB hard ceiling |
| Total process resident memory | 14 GB hard ceiling |
| Experiment artifacts | 40 GB hard ceiling |
| Full immutable qualified generations | Maximum 4 locally resident |
| Journal growth | Maximum 4 GB before required compaction |

The operator rechecks free disk before every stage. A run does not begin with less than twice its estimated new artifact requirement plus 15 GB operating-system headroom. Incremental content-addressed generations are mandatory before the 100 MB stage.

CPU is the semantic reference. Metal becomes eligible only after differential event-tape tests agree within frozen numerical tolerances and all discrete decisions match where determinism is required.

## 4. Required implementation state

The runbook may enter execution only when these capabilities exist and pass their own tests:

- versioned `Event`, `ActionEvent`, and `OutcomeEvent` schemas;
- stable channel identities and logical clocks;
- correlation of prediction/action with delayed outcome;
- generic discrete-event predictive distribution;
- bounded active workspace;
- multi-timescale local eligibility;
- calibrated circuit evidence and novelty;
- structural composition, reuse, consolidation, and reclamation;
- immutable qualified generations and mutable descendants;
- `.soma` snapshot/delta/journal persistence;
- exact restoration at arbitrary event boundaries;
- recorded-environment and live-interaction paths sharing one learner;
- per-event compute, memory, storage, and learning telemetry;
- learning-disabled and mechanism-disabled controls;
- UTF-8 text transducer passing round-trip tests for arbitrary byte sequences.

Failure of any prerequisite returns work to its owning milestone. Dataset acquisition does not begin as a workaround.

## 5. Workspace and artifact layout

The target command initializes an experiment outside the source package:

```bash
soma experiment init english-demo-v1 --root ./experiments/english-demo-v1
```

The resulting logical layout is:

```text
experiments/english-demo-v1/
  experiment.yaml
  hardware.json
  runtime-lock.json
  data/
    manifests/
    prepared/
    quarantine/
  brains/
    genesis/
    qualified/
    branches/
    releases/
  reports/
    development/
    validation/
    acceptance/
  logs/
  checksums/
```

Large raw downloads remain outside brain artifacts. No source document is copied into a release brain unless its manifest explicitly permits redistribution and the release policy requires it.

## 6. Data preparation contract

### 6.1 Source manifest

Every document record contains:

- content hash;
- stable source identifier;
- origin and retrieval date;
- license/public-domain assertion and evidence reference;
- redistribution permission for source text and derived artifact;
- language and encoding;
- document lineage/duplicate group;
- quality and safety classifications;
- partition assignment;
- removal status;
- preprocessing version and output hash.

Unknown rights or provenance place a document in quarantine. Quarantined data never reaches an acquisition, validation, test, or release run.

### 6.2 Processing

Preparation performs encoding validation, byte-preserving normalization where declared, exact and near-duplicate detection, document-boundary preservation, contamination comparison, malformed-input rejection, quality filtering, and partitioning by document lineage.

The canonical text stream is UTF-8 bytes plus generic document, paragraph, and attributed-source boundary events. Lowercasing, stemming, word segmentation, part-of-speech tagging, and semantic labeling are prohibited from the canonical path.

### 6.3 Partitioning

The default split is 80% acquisition, 10% validation, and 10% untouched acceptance by complete document lineage. Development fixtures use their own manifests. The untouched partition is sealed by hash before mechanism tuning at that stage.

### 6.4 Curriculum composition

The first English corpus favors clean, comprehensible material:

- short declarative and interrogative sentences;
- simple stories and dialogues;
- basic explanatory prose;
- repeated concepts expressed in varied wording;
- progressively longer documents;
- explicit but generic speaker/source attribution.

Raw web scale and indiscriminate scraping are excluded from the demonstration program.

## 7. Genesis and lineage

Create and verify one canonical blank brain:

```bash
soma brain create --blank --preset micro \
  experiments/english-demo-v1/brains/genesis/universal-v1.soma
soma brain verify \
  experiments/english-demo-v1/brains/genesis/universal-v1.soma
```

Record its brain identity, artifact hash, runtime hash, schema versions, resource budgets, initial graph summary, and deterministic seed manifest.

Every run uses a clone:

```bash
soma brain clone SOURCE.soma DESTINATION.soma --reason RUN_ID
```

The source of a development branch is never modified. Only a branch that passes every stage gate is compacted and promoted to `brains/qualified/`.

## 8. Universal sequence qualification

### 8.1 Fixtures

Use at least these recorded environments:

- periodic sequences with hidden periods;
- variable-length nested delimiters;
- deterministic long-gap copy dependencies;
- stochastic finite-state grammar;
- composition fixture in which reused subsequences improve prediction;
- one non-text encoding with equivalent dependency complexity.

### 8.2 Controls

Run the same manifests and seed tapes through:

- marginal event-frequency predictor;
- n-gram predictors at frozen orders;
- frozen SOMA;
- shuffled-credit SOMA;
- composition-disabled SOMA;
- structure-disabled SOMA;
- appropriately sized conventional recurrent learner.

### 8.3 Gate

The sequence-qualified brain must:

- outperform memoryless and declared n-gram controls on held-out long dependencies;
- show causal loss when learned composition circuits are disabled;
- retain the first fixture after learning later fixtures;
- restore byte/event-exactly at arbitrary boundaries;
- respect active-state, durable-storage, and mutation budgets;
- show equivalent learning semantics on the non-text fixture without core changes.

Promote the passing descendant as `sequence-qualified-v1.soma`.

## 9. English acquisition stages

Each stage follows the same lifecycle:

1. clone the latest qualified generation;
2. freeze run configuration, data manifest, controls, and validation gates;
3. measure pre-acquisition validation;
4. expose the branch to the acquisition stream;
5. create transactional snapshots at declared event intervals;
6. measure validation without learning;
7. run causal ablations on clones;
8. inspect retention and resource trends;
9. promote only if every gate passes.

### 9.1 Stage E0 — Kilobyte sanity

Purpose: establish English byte regularities and valid generation mechanics.

- Acquisition size: 100 KB–1 MB.
- Evaluation: bits/byte, calibration, valid UTF-8, fixed-prefix continuation.
- Gate: improvement over byte frequency and frozen SOMA on unseen documents, with exact restoration.

### 9.2 Stage E1 — Causal English fixture

Purpose: prove nonlocal English sequence learning.

- Acquisition size: 1–10 MB.
- Include dependencies that n-gram controls cannot solve reliably.
- Run frozen, shuffled-credit, structure-disabled, and composition-disabled ablations.
- Gate: held-out gain attributable to active learned circuitry and retained after an interference stream.

### 9.3 Stage E2 — Demonstration corpus

Purpose: acquire enough structure for recognizable short generation.

- Acquisition size: 10–50 MB.
- Gate: positive held-out scaling from E1, useful learned compositions, bounded active fraction, bounded durable growth, and predetermined fixed-prefix improvement.

### 9.4 Stage E3 — Scaling qualification

Purpose: determine whether additional data improves capability efficiently.

- Acquisition size: 100–500 MB.
- Measure quality per million bytes, quality per wall-clock hour, durable bytes per quality gain, and retention.
- Gate: positive validation trend without approaching total-brain execution, storage ceiling, or unacceptable forgetting.

### 9.5 Stage E4 — Extended local run

Purpose: attempt the largest justified MacBook demonstration brain.

- Acquisition ceiling: approximately 1 GB.
- Entry requires E3 passing all scaling gates.
- The exact size is frozen from measured time, memory, and disk projections.
- Gate: improvement over E3 large enough to justify resource cost and no regression of earlier causal evidence.

Failure at E3 ends data scaling and returns the project to mechanism diagnosis. E4 is not required for a valid smaller demonstration brain.

## 10. Target teaching invocation

The target interface for a qualified stage is:

```bash
soma teach BRANCH.soma \
  --transducer text-utf8 \
  --manifest DATA_MANIFEST.json \
  --partition acquisition \
  --checkpoint-events 1000000 \
  --evaluation-config EVALUATION_CONFIG.yaml \
  --resource-config MACBOOK_RESOURCE_CONFIG.yaml \
  --report REPORT_DIRECTORY
```

Teaching consumes the acquisition stream in recorded order. Repeated exposure must be declared. The default is one exposure because continual learning should extract value from a stream without depending on unlimited replay. Additional exposures are separate experiments and their effects on memorization and retention are reported.

The run aborts safely when it reaches a hard memory/storage budget, non-finite state, invariant failure, journal failure, sustained calibration failure, or a configured thermal/runtime safety condition. An aborted branch is retained for diagnosis and cannot be promoted.

## 11. Measurements

At every declared interval record:

- total external, predicted, generated, and outcome events;
- sustained and percentile events/second;
- bits/byte and next-event accuracy;
- negative log-likelihood, Brier score, and calibration error;
- active cells and connections/event;
- active fraction of total brain/event;
- circuit recruitment, reuse, consolidation, and reclamation;
- structural mutations/million events;
- durable bytes/million events;
- peak resident memory and Metal/CPU memory split;
- journal, snapshot, restore, and compaction duration;
- retained performance on all earlier qualified stages;
- fixed-prefix generations under frozen deterministic and sampled settings;
- causal degradation for each required ablation.

Estimated duration is:

```text
processed events × declared exposures ÷ measured sustained learning events/second
```

Planning estimates are recalculated after each smaller stage. No unmeasured throughput assumption authorizes a larger run.

## 12. Generation qualification

Generation uses an attributed output channel:

1. provide an external prefix through the text transducer;
2. request a distribution over the next byte or learned symbol event;
3. select under a frozen deterministic or sampling configuration;
4. emit through the UTF-8 effector;
5. return it to working state marked `model_emission`;
6. stop at a learned/declared boundary, invalid-sequence guard, or event budget.

Model emissions do not receive truth status and do not update protected knowledge without verified outcome or explicit approval.

The generation gate requires:

- valid UTF-8 at the frozen minimum rate;
- better fixed-prefix likelihood than declared controls;
- measurable dependence on supplied history;
- degradation under frozen-core and causal circuit ablations;
- bounded repetition and output length;
- exact deterministic reproduction after restoration;
- no unauthorized learning from self-emitted sequences.

Recognizable fragments qualify generation proof. They do not qualify consumer usefulness.

## 13. Dialogue acquisition

Dialogue is trained as attributed experience after English prediction and generation qualify. Generic channels distinguish participants and environment outcomes. The curriculum covers:

- alternating turns;
- question followed by relevant short answer;
- instruction followed by bounded response;
- response termination;
- explicit correction and retry;
- uncertainty when evidence is inadequate;
- reference to information earlier in the active interaction;
- retrieval of one approved prior episode after restart.

The dialogue gate requires successful behavior on untouched interaction templates, correction uptake within a frozen event budget, retention of base English metrics, persistence after restart, and no hardcoded question/answer routing in the universal core or text transducer.

## 14. Qualification and branching policy

The canonical lineage is:

```text
universal-v1
└── sequence-qualified-v1
    └── english-basic-v1
        └── english-generative-v1
            └── english-dialogue-v1
                └── soma-english-demo-v1
```

Multiple experiments may branch from a qualified generation. Promotion requires:

- complete configuration and manifest;
- passing development and validation gates;
- exact artifact and report hashes;
- no post-result gate changes;
- independent untouched acceptance run;
- successful integrity verification.

A failed branch remains labeled failed and is never used as the undocumented ancestor of a release.

## 15. Release preparation

Clone the accepted dialogue generation and perform:

```bash
soma brain compact RELEASE_CANDIDATE.soma
soma brain sanitize RELEASE_CANDIDATE.soma --policy demo-release-v1
soma brain verify RELEASE_CANDIDATE.soma --deep
soma brain export RELEASE_CANDIDATE.soma soma-english-demo-v1.soma
```

Sanitation removes private filesystem paths, host identifiers, secrets, development-only event journals, disallowed source content, private episodes, and nonredistributable cached material while preserving learned structural and numerical state permitted for release.

The release bundle includes:

- `.soma` artifact and checksum;
- runtime and transducer compatibility ranges;
- genesis and qualified lineage hashes;
- data-manifest identifiers and redistribution statement;
- resource requirements measured on the reference MacBook;
- evaluation configuration and complete acceptance report;
- limitations and intended-use statement;
- backup, restore, and continued-learning instructions.

## 16. Clean-install acceptance

On a clean supported installation:

1. install the signed runtime;
2. install the signed UTF-8 text transducer;
3. import and verify the demonstration brain;
4. reproduce fixed deterministic generations;
5. complete untouched short dialogues;
6. teach one approved new fact or pattern;
7. verify targeted uptake and retention;
8. stop and restart the runtime;
9. verify the learned change persists;
10. back up, restore into a separate brain identity, and compare behavior/state under the declared equivalence rules;
11. confirm all resource ceilings and integrity checks.

Only this passing artifact may be named `soma-english-demo-v1.soma`.

## 17. Completion levels

Reports use exactly these claims:

| Level | Minimum established result |
|---|---|
| Sequence proof | Nontrivial temporal dependencies learned causally |
| English acquisition proof | Untouched English prediction improves |
| Generation proof | Learned state causally produces valid recognizable English fragments |
| Dialogue proof | Basic attributed interactions and corrections work |
| Demonstration brain | Downloadable, reproducible, persistent, bounded artifact |
| Consumer English brain | Full SOMA Text v1 product gates pass |

Passing a lower level never implies a higher one.

## 18. Immediate implementation order

1. Complete R0 universal contracts.
2. Complete R1 calibrated evidence routing.
3. Complete R2 universal event-driven organism with cross-domain fixtures.
4. Implement experiment manifests, branching, and telemetry needed by this runbook.
5. Implement the reversible UTF-8 transducer and effector.
6. Execute universal sequence qualification.
7. Execute E0 and diagnose before increasing data.
8. Advance through E1–E4 only through their frozen gates.
9. Qualify generation.
10. Acquire and qualify elementary dialogue.
11. Produce and clean-install-test the demonstration release brain.

This runbook is revised only through versioned changes that explain why a gate, metric, resource limit, or procedure changed. Untouched acceptance data is replaced when a change has materially adapted the project to it.
