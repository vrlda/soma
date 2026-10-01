# SOMA Real Model Master Plan

**Status:** canonical post-v15 engineering plan. Live milestone status is in Section 16 and the ordered next work in Section 22 (last reviewed 2026-09-30).  
**Purpose:** take the validated SOMA learning kernel to a downloadable, trainable, continuously learning, prosumer-ready model that accepts human input and produces useful responses  
**Supersedes:** the scoped roadmap preserved in `SOMA_SYNTHETIC_KERNEL_HISTORY.md`; that file is evidence/history and is not an active plan

---

## 1. The correction

Language, perception, action, robotics, and real-world continual learning are not outside the purpose of SOMA. They are outside the claims justified by the current implementation and benchmarks.

That distinction matters:

- The project vision is a general continuously learning computational organism.
- The present implementation is a small, deterministic, bounded learning kernel operating on fixed-width numeric observations and scalar feedback.
- V15 completed validation of that kernel in its synthetic domain. It did not complete the product or the full model.
- The next program must convert the mechanism into a system that learns language, generates responses, operates at useful scale, survives real user data, and can be installed and used without code changes.

The v15 result is therefore the foundation, not the destination.

## 2. What “100% done and working” means

SOMA reaches product completion only when a person can:

1. Download an application and a signed base model artifact.
2. Start it with a documented command or desktop launcher on supported consumer hardware.
3. Give it text and files without writing integration code.
4. Receive coherent, relevant, useful generated responses.
5. Teach it through conversations, corrections, approved documents, and outcomes.
6. Close the application, reopen it, and continue the same model with its learned state intact.
7. Inspect, pause, constrain, back up, restore, export, and selectively forget learned material.
8. Run it for long periods without unbounded memory growth, silent corruption, catastrophic behavioral collapse, or mandatory full retraining.
9. Obtain equivalent documented behavior with default settings rather than architecture tuning.

“One model” means one persistent logical identity and checkpoint lineage. Internally, its artifact may contain an immutable base, mutable lifetime state, indexes, and recovery snapshots. Those are layers of one model, not separately selected task models.

“No retraining like a conventional LLM” means routine adaptation must happen online and incrementally. It does **not** mean a blank organism can speak without first acquiring language. A useful downloadable base checkpoint must be bootstrapped once from a substantial corpus. Users then continue that same model rather than periodically rebuilding it from scratch.

### 2.1 Foundational decisions

The following decisions define the project and must be treated as architectural constraints:

- SOMA's native abstraction is a time-ordered **event**, not a text token.
- Text tokens, image patches, audio frames, sensor readings, tool results, and actions are adapter-level encodings of events.
- The universal core contains no language-, vision-, audio-, robotics-, tool-, or image-generation policy.
- Domain adapters translate physical or digital interfaces; they may describe shape, timing, units, and safe bounds, but may not contain the intelligence that solves the domain.
- The installed runtime and the acquired brain are separate artifacts.
- A brain is a mutable structured state, not merely a Python program and not merely a fixed tensor file.
- SOMA still has numerical parameters such as connection strengths and thresholds; these are weight-like state, but they are only one part of the brain alongside topology, activity, evidence, memories, and lineage.
- The whole brain is persistent long-term state. It is not a context window. A bounded active workspace and selective retrieval determine what participates in a particular moment.
- A blank universal brain and an acquired trained brain are different distributions of the same architecture.
- Training and use share one lifecycle: events arrive, the brain predicts or acts, outcomes arrive, eligible state changes, and the new state persists.
- Sparse execution must make cost depend primarily on active circuitry, not total lifetime knowledge.

These constraints may be changed only through an explicit architecture decision record supported by experiments. Convenience in the text prototype is not sufficient reason to make text concepts fundamental to the organism.

## 3. Product and distribution targets

### 3.1 Universal blank SOMA

The foundational release is a domain-neutral organism definition and runtime. A newly created blank brain contains initial cells, generic plasticity and structural rules, generic event/action channels, resource limits, persistence machinery, and safety enforcement. It contains no acquired English vocabulary, visual categories, tool skills, robot policy, or other domain knowledge.

The blank brain is expected to be initially incapable. Universality means it can acquire different domains through compatible experience without changes to its learning core; it does not mean useful capabilities appear without data, interaction, transducers, and compute.

The universal distribution is intended for researchers, developers of new transducers and environments, organizations training new brains, and reproducibility. Its acceptance test is cross-domain acquisition with the same core—not immediate consumer usefulness.

### 3.2 First shippable trained brain: SOMA Text v1

The first prosumer release must be deliberately focused. Its complete user-facing capability set is:

- multi-turn text chat;
- local ingestion of plain text, Markdown, PDF text, and selected folders;
- persistent learning from explicitly approved conversations and documents;
- correction by the user with visible confirmation that the correction was stored;
- retrieval and use of learned personal knowledge with provenance;
- a safe “learning off” mode;
- streaming response generation;
- local command-line interface and local HTTP API;
- checkpoint backup, restore, export, import, and health inspection;
- sensible presets for supported hardware;
- no required Python coding, configuration editing, or model surgery.

Text v1 is not declared complete merely because it emits byte or symbol sequences. It must meet the language, continual-learning, reliability, safety, performance, and usability gates in this document.

Text v1 is an acquired descendant of the universal blank brain. English and interaction behavior live in its persisted learned state, not as special cases in the core runtime.

### 3.3 Destination releases

The architecture must leave clean interfaces for these subsequent capabilities:

- **Perception:** images first, then audio and continuous sensor streams.
- **Action:** schema-constrained tool calls, computer interaction, and environment actions.
- **Embodiment:** simulator-trained control adapters followed by hardware-specific robotics adapters.
- **Real-world continual learning:** long-lived learning across mixed modalities, changing tasks, delayed outcomes, conflicting information, and finite capacity.

These remain part of SOMA. They follow Text v1 because language provides the fastest path to a usable model and the strongest debugging surface for memory, routing, uncertainty, and correction. Building every modality simultaneously would make failures unidentifiable.

### 3.4 What users install and download

Distribution has three independent units:

1. **SOMA Runtime:** compiled CPU/GPU engine, persistence, adapter host, safety boundary, service, CLI, and optional desktop interface.
2. **SOMA Brain:** a mutable `.soma` artifact containing one organism's topology, numerical state, memories, evidence, consolidation, lineage, and configuration.
3. **SOMA Transducer Package:** a signed adapter that maps an external medium or device to and from the universal event protocol.

Installing the runtime does not create knowledge. Creating a blank brain produces a new untrained organism. Downloading an official or third-party trained brain provides acquired capabilities that can continue learning. Updating the runtime must not overwrite the brain; cloning a brain must create a distinct identity and lifetime lineage.

## 4. Current baseline and the actual gap

The "At v15" column is the starting point this plan was written against. "Now" is the state as of the last review. Details and evidence are in Section 16.

| Area | At v15 | Now (2026-09-30) | Required product state |
|---|---|---|---|
| Inputs | Fixed-width numeric vectors | Versioned event envelopes, channels, clocks; byte, glyph, and cart transducers | Universal timestamped event channels connected through transducers |
| Outputs | Small numeric action | Constrained UTF-8 generation, schema-constrained tool calls, cart actions | Generic prediction/action events interpreted by output transducers |
| Learning | Local online updates with delayed scalar reward | Online sequence acquisition, corrections, trust-ranked episodic rules, quarantine | Self-supervised acquisition, instruction behavior, explicit correction, and continual online adaptation |
| Context | Detector thresholds plus protected probes | Calibrated evidence router is the default | Calibrated circuit evidence with uncertainty and bounded active identification |
| Representation | Small graph-derived products | Bit suffix tables (service) and byte-context circuit mixing (R6 research, 1.92 bits/byte) | Scalable learned sequence, concept, episodic, and compositional representations |
| Capacity | Tens of cells and hundreds of synapses | Up to 16.8M budgeted circuits in the language memory; organism still small | Hardware-budgeted sparse substrate with millions or more effective parameters/connections |
| Working context | Current recurrent state in a small organism | Contract written (`docs/workspace-contract.md`); not yet enforced for language | Bounded active workspace plus selective retrieval from persistent brain state |
| Runtime | Standard-library Python reference | Python reference; Rust engine with parity for graph, forward, learning, evidence, detector, and circuit mixing; no GPU | Separately installed optimized CPU/GPU runtime and adapter host |
| Persistence | Atomic JSON checkpoint | Binary `.soma` v1 container with chunk hashes, journal, crash recovery | Downloadable mutable `.soma` brain with chunked, checksummed, migratable state |
| Evaluation | Synthetic numeric lifetimes | Frozen gates for R1–R12 first steps plus conventional reference baselines | Language quality, continual learning, safety, robustness, and user acceptance |
| Distribution | Source repository and CLI | CLI, local HTTP API, installer, doctor; unsigned | Signed packages, model downloader, guided setup, API, documentation, and diagnostics |

The first scientific risk is whether a common event-driven SOMA core can acquire representations across more than one domain without task-shaped code. The next is whether its local, structural, continual rules can learn useful language and generation at acceptable sample and compute efficiency. The plan treats both questions as early falsifiable gates rather than assuming success.

## 5. Architectural contract

The system must be separated into explicit universal and domain-boundary layers. Every layer has a stable interface, persisted-state rules, ablations, and metrics. No text-only assumption may leak into the universal layers.

### 5.1 Universal event protocol

The native input is a time-ordered event envelope, not a token or fixed observation vector. Its minimal schema contains:

- stable channel and source identity;
- monotonic event identity and timestamp or logical clock;
- typed bounded signal payload and declared shape;
- duration or validity interval when relevant;
- modality-neutral boundary and relationship metadata;
- provenance, trust, access scope, and learning permission;
- optional uncertainty and measurement quality;
- correlation identity for actions and delayed outcomes.

The core may learn recurring event compositions into higher-level circuits, but it may not assume that events are words, pixels, audio frames, or motor commands. Events with different rates are synchronized by explicit clocks and buffers rather than by pretending they share text positions.

Outputs use the same principle: SOMA emits predictions or proposed action events on declared channels. An effector converts them into bytes, images, tool calls, sound, or motor commands after the external safety boundary approves them.

### 5.2 Transducers and effectors

Transducers convert external data into universal events; effectors convert approved action events back into domain objects or physical outputs.

An adapter declares channel schemas, sampling or segmentation rules, units, clock behavior, reversible encoding where possible, hardware requirements, and hard safety bounds. It must never inject task identity, evaluation labels, hidden answers, future information, pretrained task solutions, or semantic categories unavailable in the raw interface.

Some preprocessing is physically necessary: decoding a file, sampling a microphone, reading camera pixels, or converting joint positions to normalized values. Learned modality encoders may be offered as explicitly identified acquired components and matched controls, but universality requires at least one minimally interpreted path through which the SOMA core can acquire structure itself.

### 5.3 Text transducer and learned symbolic composition

Tokens are optional compression and must not define the brain. The first canonical text transducer uses reversible UTF-8 bytes plus explicit stream and turn boundaries. This supports every Unicode language, contains no fixed English vocabulary, and gives the organism a neutral low-level alphabet.

SOMA must learn reusable compositions above bytes: character sequences, fragments, words, phrases, syntactic patterns, and concepts. A learned composition is accepted only when reuse improves prediction or action and causal ablation removes that improvement.

Three text encodings must be measured:

1. raw UTF-8 bytes as the architectural baseline;
2. a fixed reversible byte/subword compressor as an efficiency control;
3. learned SOMA chunking that promotes recurring byte circuits into higher-level events.

If embeddings or learned encoders are used, compare locally acquired, conventionally bootstrapped, and frozen-control versions. The release must disclose their training. A conventional Transformer hidden core may not be relabeled as SOMA.

### 5.4 Sparse recurrent SOMA substrate

The core receives embedded events and maintains persistent activity, eligibility, ownership, stability, utility, age, energy, and lineage. It must support:

- sparse recurrent execution;
- multiple temporal scales;
- bounded local plasticity;
- graph growth, reuse, consolidation, and reclamation;
- stable cell and connection identifiers;
- circuit-level uncertainty and ownership;
- delayed credit across event and action horizons;
- deterministic restoration at any event boundary;
- strict memory, compute, and growth budgets.

The existing Python organism remains the executable specification. The scalable implementation may change storage and kernels, but it must preserve behavioral invariants through differential tests.

### 5.5 Memory hierarchy and active workspace

A useful lifelong system cannot force all information into synaptic weights. It needs four distinct memory classes:

- **Active workspace:** the small set of current events, recurrent activations, goals, retrieved memories, and candidate circuits participating now; fast, strictly budgeted, and mostly disposable.
- **Episodic memory:** bounded records of interactions, outcomes, corrections, and provenance.
- **Semantic memory:** consolidated concepts and associations represented in learned circuits and/or a searchable store.
- **Procedural memory:** circuits for prediction, response style, tool use, and policies.

Retrieval is an internal action whose candidates and confidence are visible to the router. Stored text must not be presented as parametric knowledge. Provenance, deletion, expiry, trust level, and access scope are persisted for every external memory item.

SOMA therefore has no conventional stateless prompt context window. It still has finite working-state duration, active-circuit capacity, event-buffer size, retrieval bandwidth, and total storage. Evaluation must report those limits instead of claiming infinite context. Old information influences behavior only if it has been consolidated into circuitry, remains in memory and is retrieved, or is supplied again as an event.

### 5.6 Generic prediction and action heads

The core predicts future events and proposes actions over declared output channels. A head binds a generic brain channel to an effector schema; it does not own domain knowledge. The same learning contract must support text-byte prediction, sensor prediction, image events, tool actions, and motor commands.

For text, the effector produces an autoregressive distribution over the next byte or optional compressed symbol. It provides normalized probabilities, temperature and bounded sampling controls, deterministic mode, stop conditions, and streaming emission. For continuous outputs, the head provides a calibrated predictive distribution or bounded control value. For structured outputs, it proposes schema elements incrementally.

A byte output head is small but may require more events per sentence. Learned chunking may reduce sequence length. If a larger symbol vocabulary is used, measure dense projection first as the correctness baseline, then candidate shortlisting or hierarchical output structures. Any approximation must be evaluated for lost probability mass and generation degradation.

Every prediction head exposes likelihood, entropy, calibration, active-circuit attribution, and physical units. Fluent-looking samples or successful actions are not sufficient evidence of the claimed learning mechanism.

### 5.7 Product policy boundary

The generative model proposes text or structured actions. A separate deterministic boundary validates permissions, tool schemas, resource limits, file scopes, and user confirmation requirements. Learning cannot bypass this boundary.

## 6. Replace threshold-driven context with calibrated circuit evidence

This is the first implementation milestone after this document.

### 6.1 Required behavior

For every eligible circuit or motor module, maintain a local predictive distribution over outcomes conditioned only on information available to the organism: observation representation, proposed action, recent internal state, and delayed outcome history. Routing compares circuits using calibrated predictive evidence rather than a collection of hand-tuned switching thresholds.

The router must answer three different questions:

1. Which existing circuit best explains and handles the current stream?
2. Is the evidence sufficient to switch away from the incumbent?
3. Are all existing circuits inadequate enough to justify structural growth?

Those questions must not share one threshold.

### 6.2 Evidence model

Each circuit persists:

- predictive-model parameters and version;
- effective sample count with bounded decay;
- outcome mean, scale, and uncertainty estimates;
- accumulated log evidence for and against ownership;
- posterior routing probability;
- calibration statistics by confidence bin;
- last-used event, reuse count, stability, and protection state;
- evidence contributed by normal actions versus protected probes;
- pending delayed predictions keyed by action/event identity.

For scalar rewards, begin with a robust predictive likelihood such as a bounded Student-t or discretized outcome distribution. Compare the observed outcome under each circuit’s prediction and accumulate clipped log-likelihood ratios. Normalize across viable circuits with explicit prior mass reserved for “none of the above.” Forgetting or decay must operate on sufficient statistics, not erase evidence after one positive event.

For text-event prediction, evidence becomes sequence likelihood or locally normalized predictive surprise over a bounded interval. Length normalization prevents longer intervals from mechanically dominating. Confidence must reflect both likelihood separation and sample support.

### 6.3 Switching and hysteresis

A switch occurs only when the challenger’s posterior advantage is sustained, its lower confidence bound exceeds the incumbent’s upper confidence bound or a sequential test reaches its accept boundary, and the minimum evidence floor is met. Returning to the previous circuit uses a separate hysteresis boundary to prevent oscillation.

When evidence is ambiguous, the router keeps the safe incumbent and may initiate a bounded probe. When the incumbent is clearly poor but no challenger is credible, it enters novelty assessment. Structural growth requires stronger and independently repeated evidence than reversible recall.

### 6.4 Active identification

Probes select actions or predictions expected to maximize information gain between plausible circuits subject to a regret and safety budget. They are:

- deterministic under a recorded seed;
- bounded by event count, cumulative regret, and wall-clock budget;
- prohibited from changing circuit weights or structure while evidence is collected;
- checkpoint-exact, including pending delayed outcomes;
- terminated early when accept/reject confidence is reached;
- unavailable for unsafe real-world actions without explicit policy approval.

### 6.5 Calibration protocol

Calibration is measured on locked streams using negative log-likelihood, Brier score, expected calibration error, reliability diagrams, posterior coverage, false-switch rate, missed-switch rate, oscillation count, novelty false positives, time-to-detect, switching regret, and growth regret.

The rollout sequence is mandatory:

1. Implement the evidence router behind an opt-in mode.
2. Run it in shadow mode while the old router controls behavior.
3. Confirm checkpoint-exact evidence and delayed-credit association.
4. Compare decisions and diagnose disagreements without changing locked evaluation manifests.
5. Enable evidence decisions with growth disabled.
6. Enable bounded probing.
7. Enable growth only after recall and novelty calibration pass independently.
8. Remove the threshold router from default behavior only after all frozen v8–v15 regressions and the new calibration suite pass.

### 6.6 Acceptance gates

- Posterior probabilities are empirically calibrated within predeclared tolerances.
- One noisy reward cannot trigger a switch or allocation.
- Sustained mismatch is detected without a fixed long warm-up.
- Familiar returns choose their owner faster than fresh acquisition.
- “None of the above” separates novelty from ambiguous recall.
- Probe budgets and regret ceilings are never exceeded.
- All evidence, uncertainty, and pending outcomes restore exactly.
- The implementation improves or preserves v12–v15 behavioral gates without manifest or post-result threshold changes.

## 7. How a universal SOMA learns language

### 7.1 Bootstrapping phase

Begin with a newly created universal brain and attach the standard text transducer. Feed a licensed, deduplicated, quality-filtered text stream in document order. At every event, the model predicts future byte events, observes what occurs, calculates bounded local predictive error, and updates eligible local components. No English vocabulary, task labels, phase identities, or correct internal features are supplied.

The first experiment is intentionally small: byte prediction on tiny corpora. It establishes causality. Progress then moves to learned chunks, larger corpora, longer temporal dependencies, grounded or interactive data, and instruction experience. Scaling is allowed only when held-out loss and ablation evidence show that the SOMA substrate—not merely output frequency, adapter preprocessing, or a retrieval cache—is learning sequence structure.

The developmental progression is observable: repeated byte events form reusable local sequences; useful sequences consolidate into higher-level circuits; those circuits participate in phrases and relationships; episodic and semantic memories connect language to sources and outcomes. Promotion to a higher-level event must be reversible during development and justified by compression, predictive gain, reuse, and causal evidence.

### 7.2 Credit assignment

Language feedback arrives every byte or learned symbol event, but useful dependencies span many events. The substrate therefore needs multi-timescale eligibility traces, local predictive errors, and bounded credit routing. Compare:

- immediate three-factor event surprise;
- traces at logarithmically spaced horizons;
- circuit-specific predictive baselines;
- local synthetic gradients or target-propagation signals as explicit experimental variants.

Every variant must retain locality at the declared boundary and include a matched frozen-core and shuffled-credit control. If pure local credit cannot cross the dependency lengths required for useful language, that is a design failure to solve, not a result to conceal with retrieval.

### 7.3 Curriculum

The base acquisition curriculum advances through locked gates:

1. bytes, encoding regularities, boundaries, and short recurrence;
2. learned character, fragment, word, and phrase compositions;
3. long-form causal and referential dependencies;
4. factual and explanatory text with provenance-aware memory;
5. dialogue roles and instruction following;
6. corrections, preference signals, and safe refusal behavior;
7. mixed stationary and changing streams without resetting the model.

Curriculum stages may change sampling, never hidden targets or architecture in evaluation. Held-out corpora and prompts are frozen before each acceptance run.

### 7.4 Online lifetime phase

After base acquisition, normal usage generates four kinds of learning events:

- unsupervised future-event observations;
- explicit user corrections;
- approved durable knowledge ingestion;
- outcomes from tools or tasks.

Their trust and learning strengths differ. Arbitrary webpage text or model-generated text cannot silently rewrite high-stability behavior. Corrections receive provenance and can be reversed. Consolidation occurs under resource budgets during idle or low-load periods without requiring a full corpus replay.

### 7.5 Preventing self-contamination

Generated responses are marked as model emissions and are not automatically treated as ground truth. Learning from them requires an external outcome, user approval, or verified source. This prevents confident errors from recursively consolidating themselves.

### 7.6 Training is an operating mode, not a separate architecture

Batch corpus acquisition and live interaction call the same event/outcome lifecycle. The differences are event source, pacing, trust, feedback availability, and checkpoint cadence. A training orchestrator may parallelize independent experience or accelerate stable graph epochs, but it may not bypass the public learning semantics with an unrelated hidden model.

From a user's perspective:

- `soma brain create` creates a blank organism;
- `soma teach` exposes it to an approved recorded environment or corpus;
- `soma interact` connects live transducers and effectors;
- both operations change the same brain unless learning is explicitly paused;
- `soma clone` branches the current brain before risky or experimental teaching.

Teaching a blank brain English is expected to require substantial data and compute. Continuous learning eliminates routine full retraining after acquisition; it does not eliminate acquisition cost.

## 7A. First English Brain Demonstration Program

The first English brain is the operational bridge between the universal organism and a useful trained release. Its procedure is specified in `SOMA_ENGLISH_DEMO_RUNBOOK.md`. The runbook's commands are target interfaces until their corresponding milestones implement them.

### 7A.1 Brain identities

- **Universal blank brain:** a domain-neutral genesis state with no acquired English.
- **Sequence-qualified brain:** a descendant that passes generic temporal-learning gates.
- **Basic-English brain:** a descendant that improves on untouched English and retains useful learned compositions.
- **Dialogue brain:** a descendant that has acquired turn-taking and elementary response behavior.
- **Demonstration release brain:** a compacted, sanitized, verified, reproducible dialogue-brain generation.

Every transition creates a new immutable qualified generation. Experimental continuations branch from the latest qualified ancestor so a failed teaching run cannot destroy the only successful lineage.

### 7A.2 Entry requirements

English teaching may begin only after the universal core provides generic event/action channels, future-event prediction, calibrated evidence, multi-timescale credit, learned composition, bounded workspace and growth, `.soma` persistence, exact event-boundary restoration, and the same learning lifecycle for recorded and live environments.

### 7A.3 Recorded-environment contract

A corpus is a recorded text environment. The UTF-8 transducer emits attributed events; SOMA predicts; the environment reveals the next event; local prediction evidence updates the same persistent brain used during live interaction. Corpus teaching may accelerate event delivery but may not invoke a separate hidden learning architecture.

### 7A.4 Development ladder

Progression is gated in this order:

1. deterministic synthetic sequences;
2. small artificial grammars and delayed dependencies;
3. kilobytes of simple held-out English;
4. a 1–10 MB causal English fixture;
5. a 10–50 MB demonstration corpus;
6. a 100–500 MB scaling run;
7. a run approaching 1 GB only after positive quality, sparsity, storage, and retention trends;
8. attributed dialogue and correction experience after basic English prediction succeeds.

Data volume never substitutes for a failed lower-stage mechanism.

### 7A.5 Development-machine envelope

The initial machine is an Apple M3 Pro MacBook Pro with 11 CPU cores, 14 GPU cores, 18 GB unified memory, and approximately 106 GB free at the time this plan was revised. The provisional experimental budgets are 6–10 GB resident brain state, 2–4 GB runtime workspace, and 20–40 GB incremental brain/checkpoint storage. CPU semantic correctness precedes Metal acceleration. These are development constraints, not release requirements or capability promises.

### 7A.6 Mandatory measurements and controls

Every run records events/second, bits/byte, next-event calibration, active cells and connections/event, structural mutations and durable bytes/million events, peak memory, checkpoint/restore time, retention, fixed-prefix generations, and causal composition contribution. Duration is computed as processed events multiplied by exposures and divided by sustained learning events/second.

Matched controls are byte frequency, n-gram, frozen SOMA, shuffled credit, structure disabled, composition disabled, retrieval disabled, and an appropriately sized conventional recurrent learner. Generated samples are diagnostics, not substitutes for held-out metrics and causal controls.

### 7A.7 Generation and dialogue rules

External bytes are attributed observations. Predictions receive error when the corresponding external event arrives. Generated bytes remain attributed model emissions and are never promoted to truth without verified external outcome or explicit approval.

Dialogue is acquired after prose prediction. Speaker/source identity, turn-taking, response initiation and termination, questions, instructions, corrections, uncertainty, and conversational continuity are learned through generic attributed channels rather than language-specific core branches.

### 7A.8 Demonstration release gate

A demonstration brain must pass sequence, untouched-English, generation, dialogue, retention, sparse-resource, exact-restoration, and continued-learning gates. It is then compacted, sanitized of private/nonredistributable state, verified, loaded on a clean installation, and exported with lineage, transducer, data-manifest, runtime, and evaluation hashes.

## 8. Feasibility strategy: pure SOMA and pragmatic controls

The project must remain scientifically honest while pursuing a product.

### 8.1 Primary track

The primary track uses a SOMA recurrent/structural core, local credit, continual circuit creation and reuse, and no Transformer hidden stack. Success means the core produces measurable held-out sequence learning and continual adaptation.

### 8.2 Control track

A matched conventional small recurrent or Transformer model establishes the compute/quality floor. A hybrid SOMA model may use conventionally trained embeddings, decoder, or modality adapters around the SOMA core. It identifies whether failure lies in representation interfaces or the continual core.

### 8.3 Decision gates

At each scale, record quality per training token, quality per joule, inference latency, memory, forgetting, adaptation speed, and recovery. If the pure track misses a predeclared viability floor, stop scaling it and diagnose the mechanism. Do not spend large compute on an architecture that has not beaten trivial sequence controls at small scale.

A hybrid may ship only if it still provides demonstrated SOMA-specific continual-learning benefits and is described accurately. Product urgency cannot convert a failed experiment into a claim of a novel end-to-end learner.

## 9. Scalable implementation

### 9.1 Reference and production engines

Maintain two engines:

- **Reference engine:** readable Python, deterministic, small models, invariant oracle, migrations, and mechanism experiments.
- **Production engine:** Rust or C++ host with optimized CPU kernels and CUDA/Metal GPU backends, memory-mapped model storage, batched sparse operations, and Python bindings for evaluation.

The choice between Rust and C++ is made by a short prototype measuring dynamic sparse graph mutation, GPU integration, serialization safety, and packaging. File format and semantic tests must not depend on the host language.

### 9.2 Execution model

Do not allocate one object per production synapse. Use structure-of-arrays storage, compressed adjacency blocks, stable logical IDs mapped to compact physical slots, generation counters for reused slots, bounded free lists, and append-only mutation journals. Separate frequently accessed numeric state from metadata and provenance.

Growth and pruning happen at synchronization points. Event execution uses stable snapshots so graph mutation cannot invalidate active kernels. Compaction produces a new generation atomically and preserves logical identity.

### 9.3 Model tiers

Initial engineering tiers are measurement targets, not capability promises:

| Tier | Purpose | Hardware target | Constraint |
|---|---|---|---|
| Micro | correctness and CI | CPU, under 2 GB RAM | complete tests in minutes |
| Tiny | language feasibility | modern CPU or entry GPU, 8–16 GB | interactive generation and overnight experiments |
| Small | useful local alpha | consumer GPU, 16–24 GB VRAM or unified memory | sustained chat and local adaptation |
| Prosumer | release candidate | high-memory consumer workstation | best documented quality with one-command presets |

Exact cell, connection, context, and checkpoint budgets are frozen only after profiling. Every preset enforces hard ceilings and reports headroom.

The preliminary memory envelope must account for more than connection strength. A production connection may require endpoint references, strength, eligibility, stability, utility, age, flags, and allocator metadata. At 16–32 bytes per connection, one billion resident connections alone would occupy roughly 16–32 GB before cells, active workspace, episodic memory, indexes, runtime buffers, and snapshots. This estimate is a capacity warning, not a design target.

### 9.4 Performance gates

For each tier, publish ingestion events/second, output events/second, text bytes and characters/second where applicable, update overhead, peak resident memory, checkpoint size, checkpoint duration, restore duration, compaction duration, and energy where measurable. Online learning must not make interactive generation unusable. A read-only inference mode provides a performance and safety control.

Universal measurements are events/second, active cells/event, active connections/event, structural mutations/event, learning overhead, retrieval bandwidth, and bytes of durable growth per million events. Text reports bytes/second and user-visible characters or words/second in addition to any token-derived compatibility metric. Vision, audio, and robotics report their native real-time deadlines.

No final CPU, GPU, or RAM requirement may be promised before R2–R5 profiling. Sparse storage alone is not sufficient: the active fraction must remain bounded as the brain grows. A scaling run fails if compute approaches total-brain traversal per event or durable state grows without proportional retained capability.

## 10. Data and base-model training pipeline

### 10.1 Data requirements

Every source needs license status, provenance, collection date, language, quality score, duplication group, safety classification, and removal policy. Personal, secret, paywalled, and unlicensed data are excluded. Dataset manifests are versioned and content-addressed.

Processing includes exact and semantic deduplication, encoding repair, language identification, document-boundary preservation, PII filtering, contamination checks against evaluation sets, quality filtering, and train/validation/test partitioning by document lineage.

### 10.2 Reproducible training

The trainer must provide:

- declarative configuration with schema validation;
- deterministic seeds and independent data/model/sampling streams;
- resumable streaming shards;
- periodic atomic snapshots;
- metric and resource telemetry;
- divergence detection and automatic safe stop;
- retained best and latest recoverable checkpoints;
- complete artifact lineage from code commit, data manifest, transducer/codec, configuration, and hardware/software versions.

### 10.3 Training stages

- **Acquisition:** broad self-supervised language stream.
- **Instruction:** licensed prompt/response and task demonstrations.
- **Interaction:** corrections, preference pairs, and outcome-driven tasks with explicit trust labels.
- **Continual qualification:** changing streams, conflicts, returns, and long delays without resets.
- **Release stabilization:** learning-rate/metaplasticity calibration, safety behavior, quantization, and package validation.

Consolidation may revisit a bounded internal episodic sample, but the system may not depend on replaying the entire original training corpus during ordinary user learning.

## 11. Evaluation program

### 11.1 Language competence

Measure held-out bits-per-byte/perplexity, event-prediction calibration, long-history retrieval, compositional generalization, question answering, summarization, instruction following, factuality with provenance, dialogue consistency, correction uptake, and response preference. Use public suites where licenses permit and private locked suites to reduce benchmark shaping.

### 11.2 Continual learning

Evaluate streams containing new facts, changed facts, recurring skills, conflicting users, domain shifts, vocabulary growth, corrections, and long inactive intervals. Report acquisition time, return latency, forward transfer, backward transfer, forgetting, interference, capacity consumed, pruning damage, calibration, and recovery after erroneous teaching.

### 11.3 Causal evidence

Every claimed mechanism needs an opt-out or ablation: frozen core, no structural growth, no consolidation, no episodic retrieval, threshold router, shuffled delayed credit, no online learning, and matched replay/conventional models. Disable learned circuits and measure the predicted capability loss.

### 11.4 Safety and adversarial evaluation

Test prompt injection, poisoned documents, malicious corrections, privacy extraction, cross-user leakage, unsafe tool requests, untrusted brain imports, decompression bombs, malformed events and byte streams, checkpoint corruption, resource exhaustion, and model-generated misinformation fed back as training data.

### 11.5 Locked acceptance discipline

Manifests, metrics, controls, seed sets, and pass thresholds are frozen before final runs. Development results and untouched acceptance results are reported separately. A benchmark is retired or expanded when implementation choices have materially adapted to it.

## 12. Model artifact and persistence

Code is not the brain. The separately installed runtime defines execution and learning semantics; a `.soma` brain contains a particular organism's acquired and currently active persistent state. JSON is retained only for small reference fixtures. The production brain is a binary, versioned directory or single-file container with:

- signed manifest, brain identity, ancestor identity, and format/runtime compatibility versions;
- transducer schemas and hashes without requiring a particular modality;
- compact cell tables and stable logical-to-physical identity maps;
- compressed sparse graph blocks, connection strengths, and plasticity state;
- mutable circuit registry, ownership, structural lineage, utility, and consolidation;
- episodic/semantic stores and provenance indexes;
- router calibration and pending credit state;
- bounded active workspace and resumable event clocks when a live session is checkpointed;
- resource budgets and hardware profile;
- append-only write-ahead log;
- chunk checksums and Merkle/content hashes;
- migration history;
- safety policy/configuration version;
- user-visible model card and license.

Writes use journal, fsync, checksum verification, and atomic generation switch. Startup detects incomplete writes and restores the latest valid generation. Backups are incremental. Imports are parsed under resource limits and never execute embedded code.

Operationally, one logical brain may use three storage layers:

1. **Base snapshot:** a stable acquired starting state, possibly downloaded and shared.
2. **Lifetime delta:** private structural, numerical, and memory changes acquired by this descendant.
3. **Write-ahead journal:** recent transactional events awaiting validated consolidation.

Those layers are one brain lineage, not three task models. Compaction creates a new immutable generation and clears only journal entries proven incorporated. A shared downloaded base remains content-addressable, allowing many private descendants without copying every unchanged block.

Base updates and personal learning require a three-way migration: old base, private lifetime delta, and new base. If safe automatic reconciliation is impossible, preserve the existing brain and explain the conflict; never silently discard personal learning. Because structural identities may diverge, runtime upgrades are expected to be easier and safer than replacing a trained base underneath a living descendant.

Brain operations must include create, clone, branch, inspect, verify, compact, export, import, backup, restore, and retire. Sharing defaults to a sanitized base snapshot; episodic memories, secrets, user provenance, active sessions, and private deltas are excluded unless explicitly selected.

## 13. Prosumer experience

SOMA requires its own persistent interaction harness rather than a stateless prompt wrapper. The harness owns event scheduling, transducer lifecycles, delayed outcomes, learning permissions, active-workspace budgets, background consolidation, transactional persistence, provenance, rollback, and external action safety. Chat and OpenAI-compatible endpoints are convenience views over this lifecycle, not the architecture itself.

The default trained-brain workflow must be this small:

```bash
soma runtime install
soma brain pull soma-text-small
soma start soma-text-small
soma chat
soma learn ~/Documents/approved-notes
soma backup ~/Backups/soma
```

The blank universal workflow is distinct:

```bash
soma brain create --blank experiment.soma
soma transducer attach experiment.soma text-bytes
soma teach experiment.soma --source ./licensed-corpus
soma interact experiment.soma --text
```

The local API should provide an OpenAI-compatible chat endpoint where practical plus SOMA-specific endpoints for learning, evidence, provenance, health, backup, and forgetting. A later desktop interface wraps exactly the same service.

Required commands are:

- `soma runtime install|update` — hardware detection, signed engine installation, and backend selection;
- `soma brain create|pull|clone|list` — blank creation, trained-brain download, identity branching, and inventory;
- `soma transducer list|attach|detach` — signed external-interface management and schema validation;
- `soma start|stop|status` — lifecycle management;
- `soma interact` — connect declared live input/output channels;
- `soma chat` — text-interaction convenience interface;
- `soma teach` or `soma learn` — preview, approve, expose the brain to recorded experience, and report changes;
- `soma forget` — preview affected memories/circuits, snapshot, then remove or suppress selected knowledge;
- `soma inspect` — health, capacity, calibration, learning, and provenance summary;
- `soma backup|restore` — verified recovery operations;
- `soma export|import` — portable signed artifacts;
- `soma serve` — documented local API;
- `soma doctor` — hardware, runtime, integrity, and performance diagnostics.

Defaults must be safe. Learning from chat is opt-in per installation and visibly indicated. Data remains local unless the user configures a remote service. Error messages provide recovery actions rather than stack traces alone.

## 14. Safety, privacy, and control

Continual learning creates risks absent from static inference. Product release requires:

- trust levels for system data, signed base data, approved local sources, conversation corrections, untrusted retrieved text, and model emissions;
- quarantine before untrusted material can affect stable circuits;
- rate and magnitude limits on online updates;
- anomaly detection for sudden capability or policy shifts;
- automatic pre-learning snapshots and rollback;
- per-source provenance and deletion;
- encrypted local secrets and optional checkpoint encryption;
- strict separation between users/profiles;
- learning pause, read-only inference, factory clone, and safe recovery modes;
- explicit confirmations for destructive actions and high-impact tools;
- auditable structured action proposals;
- network and filesystem scopes enforced outside the model;
- signed releases and dependency/security scanning.

Selective forgetting must cover external memory immediately and circuit influence as far as technically verifiable. The UI must explain when exact removal from distributed learned state cannot be guaranteed and offer rollback to a snapshot predating ingestion.

## 15. Perception, tools, and robotics path

### 15.1 Vision and audio

Add modality transducers only after the universal event core and Text v1 path are viable. Each minimally interpreted adapter maps samples, pixels/patches, frames, and timing into the common event space without object labels or task solutions. Frozen conventional encoders are controls and optional efficiency scaffolds, not the definition of the modality path. Cross-modal alignment must be evaluated on new concepts acquired through paired experience and recalled after interference.

### 15.2 Tools

Tools use grammar-constrained structured decoding against declared schemas. The model receives tool results as attributed events. The external policy engine controls permissions, confirmation, timeout, network scope, and result size. Success/failure outcomes may train procedural circuits only after correct delayed association.

### 15.3 Robotics

Robotics begins in deterministic simulators with emergency-stop, action magnitude/rate limits, collision boundaries, and complete telemetry. Progression is offline replay evaluation, simulator closed loop, hardware-in-the-loop, supervised low-energy hardware, then bounded autonomous trials. A language checkpoint alone is never connected directly to unrestricted actuators.

Embodied completion requires sensor dropout handling, real-time deadlines, safe fallback control independent of SOMA, calibrated uncertainty, distribution-shift detection, and recovery after interrupted learning.

## 16. Milestones from v15 to the real model

Each milestone closes only when every exit gate passes in the same locked revision.

### R0 — Preserve the proven kernel ✅ COMPLETE

**Deliverables:** frozen v8–v15 artifacts, reproducible environment, clean dependency metadata, CI matrix, benchmark result archive, current checkpoint schema documentation, and architecture decision records for the universal event protocol, brain/runtime separation, active workspace, transducer boundary, and blank-versus-trained distribution.

**Exit gates:** all 221 existing tests and locked benchmarks pass; reference outputs are archived with hashes; the five universal contracts have versioned schemas and falsifiable acceptance tests; no product work silently changes historical claims.

### R1 — Calibrated evidence router ✅ COMPLETE

**Deliverables:** probabilistic per-circuit outcome models, posterior router, novelty mass, uncertainty, hysteresis, information-gain probes, checkpoint migration, shadow reports, and calibration tests.

**Exit gates:** all Section 6 gates pass; frozen v8–v15 remain green; threshold routing is no longer default.

### R2 — Universal event-driven organism ✅ COMPLETE

**Deliverables:** modality-neutral event/action API, channel registry, logical clock and synchronization rules, generic prediction heads, multi-timescale active workspace and credit, transducer SDK, persisted event state, and at least two minimal domain adapters.

**Exit gates:** the unchanged core learns locked temporal/compositional tasks in at least two differently encoded domains; adapter-only and shuffled-credit controls fail as predicted; task identity does not enter the brain; exact interruption/resume works at arbitrary event boundaries; active compute and storage remain bounded.

### R3A — Generic sequence acquisition ✅ COMPLETE

**Deliverables:** synthetic sequence and artificial-grammar environments, generic discrete prediction head, multi-timescale sequence credit, learned composition, baselines, and exact-restoration fixtures.

**Exit gates:** the universal core beats memoryless and n-gram controls on locked nontrivial dependencies; shuffled-credit, frozen, and composition-disabled ablations fail as predicted; learned circuits provide causal predictive gain; arbitrary-event restoration is exact.

### R3B — English developmental acquisition ✅ COMPLETE

**Deliverables:** reversible UTF-8 byte transducer, licensed document environment, learned symbolic composition, data ladder, fixed validation/test manifests, scaling curves, and MacBook resource profile. Fixed subword encoding remains an efficiency control.

**Exit gates:** the blank-descended brain improves held-out bits/byte and calibration across qualified stages; learned compositions are reused and retained; quality improves with data without total-brain execution or unbounded storage; no English-specific core code or evaluation leakage.

### R3C — Generative text ✅ COMPLETE

**Deliverables:** byte/symbol effector, bounded sampling, stopping, streaming output, fixed-prefix suite, self-emission attribution, and generation controls.

**Exit gates:** outputs are valid UTF-8 at the frozen rate; generations are measurably dependent on learned history; frozen-core and causal circuit ablations degrade generation; self-generated events do not silently train protected state.

### R3D — Basic dialogue demonstration ✅ COMPLETE

**Deliverables:** attributed participant channels, turn framing, elementary dialogue/correction curriculum, branch-qualified brain lineage, release sanitation, clean-machine loader, and downloadable demonstration `.soma` artifact.

**Exit gates:** the brain demonstrates basic turn-taking, relevant short responses, explicit correction uptake, retention after interference/restart, continued learning after download, exact backup/restore, and all release checks in Section 7A. This is a demonstration brain, not yet the consumer-ready Text v1 release.

### R4 — Continual language learning ✅ COMPLETE

**Deliverables:** correction protocol, trust-aware episodic/semantic memory, consolidation, knowledge conflict handling, source deletion, and lifelong text benchmark.

**Exit gates:** new approved knowledge is usable within a bounded interaction budget; old capabilities stay above retention floors; familiar knowledge is recalled rather than duplicated; poisoned/untrusted text does not update protected state.

### R5 — Scalable brain runtime — PARTIAL

**Status:** done: binary `.soma` v1, journal and crash recovery, profiler, budgets, tier ceilings, 2M-step soak, and Rust ports with differential parity for the graph, forward, learning, evidence, detector, and circuit-mixing kernels (`docs/r5-engine.md`, `engine/README.md`). Serving the language memory from the engine is done (`soma-mixer-serve`, [docs/r8-engine-serving.md](docs/r8-engine-serving.md)). Open: structural growth, fingerprints, and probes on the engine; GPU backend; 72-hour soak; Micro tier on the engine.

**Deliverables:** separately installed production host, CPU backend, first GPU backend, binary `.soma` format, adapter host, differential reference tests, profiler, journal/compactor, sparse activation instrumentation, and memory-budget enforcement.

**Exit gates:** numerical/decision agreement with reference tolerances; crash-safe persistence; target Tiny throughput and memory ceilings; 72-hour learning/inference soak without invariant failure or unbounded growth.

### R6 — Useful trained-brain acquisition — PARTIAL

**Status:** Small clears the 0.45 floor, factual/provenance 8/8, snapshot resume exact, and reclamation evidence 4/4 (`docs/adr/0006-r6-control-margins-evidence.md`). The old suffix memory was bit-identical to its n-gram control. The circuit-mixing memory is the first language result that clears conventional controls: 0.2411 bits/bit on E2 (current default) against 0.2575 for a frozen order-5 n-gram, 7/7 gates; metaplasticity (ADR 0008) cut cross-book forgetting 15% (`docs/r6-circuit-mixing.md`), confirmed on a pre-registered untouched book (0.2381 vs 0.2588; `docs/r6-untouched-test.md`). Open: see Section 22 (recency interference, E3 scale, Micro on the engine, signed candidate).

**Deliverables:** licensed corpus manifest, versioned text transducer, reproducible developmental trainer, Micro/Tiny/Small scaling runs, conventional controls, and signed candidate `.soma` brain descended from the canonical blank state.

**Exit gates:** scaling trends are positive; Small clears frozen language quality floors, factual/provenance gates, continual adaptation gates, and compute budget; independent reproduction resumes from intermediate snapshots.

### R7 — Instruction and interaction model — PARTIAL

**Status:** skills, calibrated uncertainty, refusal, and schema-constrained tool calls pass their frozen gates (`docs/r7-instruction.md`). Open: human preference floors; the preference harness exists but only has a synthetic rater.

**Deliverables:** dialogue/instruction curriculum, response policy, corrections, uncertainty expression, refusal behavior, and structured tool-call proposal head.

**Exit gates:** instruction-following and human preference floors pass; correction improves targeted behavior without unacceptable collateral regression; tool syntax validity meets the locked threshold; safety evaluations pass.

### R8 — Prosumer runtime — PARTIAL

**Status:** CLI, chat, local HTTP API, doctor, clone/backup/restore/export/import, installer, presets, and a clean-machine procedure exist (`docs/r8-service.md`). Engine brains (`brain-create --memory mixing`) serve the R6 memory. Open: signatures, downloader, updater, a recorded clean-machine run, and shipping the engine with the installer.

**Deliverables:** runtime installer/updater, blank-brain creator, trained-brain downloader, transducer manager, signatures, daemon, CLI, local API, hardware presets, clone/branch, backup/restore, import/export, doctor, documentation, and brain/model card.

**Exit gates:** fresh users complete install-chat-learn-restart-restore without code or configuration edits; supported systems pass clean-machine tests; interrupted install/update/checkpoint operations recover safely.

### R9 — Closed alpha and adversarial hardening — PARTIAL

**Status:** 7/7 red-team gates, trust ranks, quotas, quarantine, consent-gated telemetry, support bundles, and runbooks (`docs/r9-redteam.md`, `docs/runbooks.md`). Open: a real closed alpha with long-running brains.

**Deliverables:** consented telemetry option, local diagnostic bundles, issue taxonomy, red-team corpus, rollback tooling, migration rehearsal, and support runbooks.

**Exit gates:** no unresolved critical integrity, privacy, tool-boundary, or data-loss defect; long-running alpha models remain recoverable; quality and forgetting stay within declared envelopes.

### R10 — SOMA Text v1 release — OPEN

**Deliverables:** signed trained brain artifact(s), separately signed runtime and text transducer packages, reproducible release manifest, public evaluation report, limitations, licenses, upgrade policy, and support documentation.

**Exit gates:** every Text v1 definition-of-done item passes on every supported hardware class. This is the first point at which SOMA is called a consumer/prosumer-ready model.

### R11 — Multimodal perception — FIRST STEP

**Status:** 6×6 glyph vision through the unchanged event core, 7/7 gates (`docs/r11-vision.md`). Real images and audio are not started; they follow Text v1.

**Deliverables:** minimally interpreted vision then audio transducers, learned-adapter controls, shared universal event/checkpoint schema, cross-modal developmental training, continual cross-modal evaluation, and privacy controls.

**Exit gates:** perception contributes causal, retained capability beyond text-only and frozen-adapter controls; online adaptation remains bounded and recoverable.

### R12 — Safe action and embodied SOMA — FIRST STEP

**Status:** simulated 1-D cart with an independent safety controller, 5/5 gates (`docs/r12-embodied.md`). Multi-step planning is the recorded next step; it follows Text v1.

**Deliverables:** production tool boundary, simulator suite, real-time runtime, hardware adapter contract, independent safety controller, and staged robotics qualification.

**Exit gates:** tools and supported robots meet domain-specific success, uncertainty, latency, containment, and emergency-stop requirements. “Unrestricted” real-world operation is not a release criterion; increasing scope is earned through explicit safety envelopes.

## 17. Concrete repository evolution

The current package should evolve without destroying the reference implementation:

```text
soma/
  reference/              # preserved Python organism and compatibility layer
  events/                 # modality-neutral envelopes, channels, clocks, outcomes
  routing/                # calibrated evidence, novelty, probing, calibration
  transducers/            # signed adapter SDK, schemas, text and device boundaries
  effectors/              # approved generic action-event conversion
  text/                   # byte codec, learned chunks, sampling, chat convenience
  memory/                 # working, episodic, semantic, provenance, forgetting
  training/               # manifests, curriculum, trainer, metrics, resumption
  runtime/                # engine interface, scheduler, budgets, compaction
  persistence/            # manifest, chunks, journal, migrations, signatures
  safety/                 # trust, quarantine, policies, action validation
  api/                    # local service and compatibility endpoints
  cli/                    # install, chat, learn, inspect, backup, doctor
  evaluation/             # language, continual, calibration, safety, performance
engine/                   # separately installed optimized runtime implementation
formats/                  # versioned schemas and compatibility fixtures
configs/                  # validated model/training/runtime presets
tests/                    # unit, property, differential, integration, acceptance
docs/                     # architecture, operations, model cards, security
```

This is a target ownership map, not permission for a mass rewrite. Modules are extracted milestone by milestone while existing imports and frozen benchmarks remain operational.

## 18. Engineering quality requirements

- Property tests cover graph ownership, resource accounting, stable IDs, posterior normalization, credit uniqueness, and migration invariants.
- Fuzzers cover event envelopes, transducers, byte codecs, brain artifacts, manifests, API payloads, and imported documents.
- Differential tests compare reference and production engines on identical event tapes.
- Fault injection covers power loss during journal append, compaction, base update, and backup.
- Determinism is offered as a documented mode; faster nondeterministic kernels report their tolerance envelope.
- Every persisted field has schema, units, bounds, validation, migration, and corruption tests.
- Metrics distinguish development, validation, and untouched acceptance sets.
- Performance regressions have tier-specific budgets in CI or scheduled hardware jobs.
- Public claims map to a reproducible command, artifact hash, and report.

## 19. Risk register and response

| Risk | Early signal | Required response |
|---|---|---|
| The supposedly universal core depends on text assumptions | non-text adapters require changes to learning or memory semantics | stop domain work, move the assumption to a transducer, and repeat cross-domain R2 gates |
| Local learning cannot acquire language efficiently | Loss plateaus near n-gram controls | isolate credit/representation with tiny causal tasks; test hybrid interfaces; do not scale prematurely |
| Dynamic sparse execution is too slow | update overhead dominates token generation | profile layouts, batch stable graph epochs, optimize hot kernels, reduce mutation frequency without changing semantics |
| Router looks confident but is wrong | poor Brier/NLL despite good average reward | recalibrate likelihoods, uncertainty, priors, and dependence assumptions; block structural growth |
| Continual updates erase base competence | regression after small user stream | strengthen trust gating, stability, protected circuits, rollback, and bounded update magnitude |
| Model poisons itself | errors grow after learning from generated text | forbid unverified self-training; require source/outcome labels |
| Retrieval masks a weak learner | quality vanishes when retrieval is disabled | report separate parametric, memory-assisted, and causal-ablation results |
| Finite capacity fills with duplicates | growth rises faster than distinct capability | improve posterior recall, novelty tests, compression, dependency-aware reclamation |
| Replay/conventional models remain much stronger | persistent quality/sample-efficiency gap | treat as diagnostic evidence; revise mechanisms or ship an honestly described hybrid |
| Personal learning cannot migrate across bases | update loses or corrupts user behavior | version deltas, rehearse three-way migration, retain old model and rollback |
| Prosumer installation is fragile | clean-machine failures or manual fixes | signed bundles, hardware detection, one-command diagnostics, supported matrix |
| Robotics causes unsafe exploration | uncertainty or action limits violated | independent safety controller, simulator qualification, supervised staged deployment |
| “No context window” is mistaken for unlimited active memory | latency or active set grows with total history | enforce workspace/retrieval budgets and report finite operational limits |
| Brain artifacts become inseparable from one runtime build | migrations or independent engine reproduction fail | stabilize schemas, semantic conformance tests, and forward-compatible readers before distribution |

## 20. Stop/go rules

Continue to the next scale only when the current model shows causal learning, locked held-out improvement, checkpoint-exact continuation, and bounded resources. Pause scaling when additional compute improves training samples but not held-out quality, when controls explain the gain, or when calibration and safety regress.

Reject a mechanism when it requires evaluator-only information, task-shaped feature lists, post-acceptance threshold tuning, hidden resets, unlimited replay, or unbounded growth. Preserve negative results and comparator wins in the record.

Revisit architecture rather than packaging if R3 cannot demonstrate genuine sequence learning. Revisit learning rules rather than adding memory retrieval if R4 fails correction and retention. Revisit runtime structure rather than reducing tests if R5 misses practical performance.

## 21. Definition of done for SOMA Text v1

All boxes must be checked in one release candidate:

- [ ] The same domain-neutral core passes cross-domain R2 acquisition without task-specific learning code.
- [ ] A blank `.soma` brain can be created independently of all trained brains.
- [ ] Runtime, brain, and transducer packages are independently versioned, installed, and verified.
- [ ] Text is connected through the universal event protocol and is not fundamental to the core.
- [ ] The canonical text path can begin from UTF-8 bytes without a pretrained English vocabulary.
- [ ] A signed downloadable base model produces useful multi-turn text responses.
- [ ] Installation and first response require no coding or hand tuning.
- [ ] The same logical model learns approved facts and corrections online.
- [ ] Learned changes survive restart, backup, restore, and supported upgrades.
- [ ] Threshold-driven context selection has been replaced by calibrated circuit evidence.
- [ ] Routing uncertainty, novelty, probes, and growth are independently measured and bounded.
- [ ] The SOMA core has causal language contribution beyond retrieval and decoder controls. *(Interpretation: ADR 0009. Core = event core plus memory subsystems.)*
- [ ] Continual learning beats the frozen/no-learning model on locked changing streams.
- [ ] Retention, interference, and capacity remain inside published limits.
- [ ] The active workspace, retrieval bandwidth, total storage, and per-event active fraction have published finite limits.
- [ ] Model emissions cannot silently become training truth.
- [ ] Untrusted input cannot modify protected state without policy approval.
- [ ] Checkpoints are crash-safe, validated, migratable, and recoverable.
- [ ] CPU/GPU presets meet published speed, memory, and storage envelopes.
- [ ] Chat, learn, forget, inspect, backup, restore, and serve workflows are complete.
- [ ] Security, privacy, poisoning, and resource-exhaustion tests pass.
- [ ] A multi-day mixed-use soak passes with no invariant failure or unbounded growth.
- [ ] An untouched acceptance suite passes without post-result changes.
- [ ] The model card accurately states data, architecture, training, limitations, safety, and hardware requirements.

Perception and embodiment are checked separately under R11 and R12. They are not discarded; they build on the completed language-capable organism.

## 22. Exact next work on resume

This is the single ordered work list. Update it whenever a step closes. The original list (R0 contracts through R2) is complete; it remains in git history.

**Current goal:** make the R6 circuit-mixing memory a trustworthy, SOMA-native language substrate. Then answer the project's own question with it: does lifetime structural consolidation beat plain accumulation? Then put it behind chat. Scale comes after that.

### Phase A: make the new result trustworthy

1. ~~**Re-baseline integrity.**~~ **Done 2026-09-30.** Every frozen benchmark covering changed code was re-run, and the full suite passes (496 tests). Commit 3532e39 is behavior-preserving. The manifest now pins all 274 tracked evidence files and excludes prose; `verify_hashes.py` reports 0 mismatches with Git LFS present. Record: `reports/rebaseline-2026-09-30/README.md`. One carried finding: the v15 archive's learning-disabled control arm does not reproduce on Linux (the gate is unaffected); a macOS re-run could settle it.
2. ~~**Untouched test book.**~~ **Done 2026-09-30.** Pre-registered, then scored once on Chesterton's *The Man Who Was Thursday*: full circuit mixing 0.2381 bits/bit vs 0.2588 for the frozen order-5 n-gram and 0.4009 for the prior E2 memory; H1–H3 all supported ([docs/r6-untouched-test.md](docs/r6-untouched-test.md), `reports/r6-untouched-test.json`).
3. ~~**Remove or justify neutral mechanisms.**~~ **Done 2026-09-30** ([ADR 0007](docs/adr/0007-circuit-mixing-mechanism-audit.md)). The correction stage is off by default: it is redundant with partial-byte gating, and removing it also reduced recency interference. Every remaining mechanism has a positive validation ablation: arbitration 0.0202, growth gate 0.0032, partial-byte gating 0.0029.

### Phase B: the SOMA question on the language path

4. ~~**Measure retention.**~~ **Done 2026-09-30** ([docs/r6-retention.md](docs/r6-retention.md), `r6_retention_benchmark.py`). Baseline: mean forgetting 0.0060 bits/bit over four book orders, order spread 0.0053. Two thirds of the forgetting persists at full budget, so plastic drift, not reclamation, is the main cause.
5. ~~**Consolidation mechanism.**~~ **Done 2026-09-30** ([docs/r6-consolidation.md](docs/r6-consolidation.md), [ADR 0008](docs/adr/0008-metaplasticity-consolidation.md)). Per-weight-set metaplasticity (τ = 1e5) cut forgetting 15% and order spread 25%, and improved final validation in every book order. It halves weight drift. The other candidates tried were rejected; all are recorded. Still open: circuit loss under budget pressure.
6. ~~**Decide what "organism plasticity" means for language.**~~ **Done 2026-09-30** ([ADR 0009](docs/adr/0009-language-substrate.md)). The circuit memory is the organism's sequence-prediction memory subsystem and the Text v1 language substrate. The organism motor path scores 0.994 on E0, adds about 1e-6 when fused, and hit a recorded stop condition; the memory's SOMA mechanisms are each causal and domain-neutral. Re-entry for organism plasticity is a pre-declared experiment. First candidate: context-detector routing of arbitration banks.

### Phase C: make it usable

7. ~~**Feature parity for the product.**~~ **Done 2026-09-30** ([docs/r6-product-parity.md](docs/r6-product-parity.md)). The distribution interface, trust-weighted observation, organism ownership with quarantine and checkpoints, and a binary `SOMAMIX1` state that Python and Rust write byte-identically. R4 passes 6/6 and R9 7/7 on the new memory. Exact forgetting stays at the episodic layer, the only one the product forgets at; statistical unforgetting cannot be exact with capped counts and plastic weights.
8. ~~**Serve from the engine.**~~ **Done 2026-10-01** ([docs/r8-engine-serving.md](docs/r8-engine-serving.md)). `soma-mixer-serve` plus `EngineMixingMemory`: 0 mismatches against the reference on random tapes, a 1.2 ms prompt, about 43 µs per generated bit. Brains created with `--memory mixing` live on the engine through clone, backup, `.soma` export, and crash recovery. Gates on the engine-served memory: R3C 4/4, R3D 6/6, R7 instruction 5/5, R7 tools 4/4 (plus R4 6/6 and R9 7/7 in step 7). Chat now uses evidence arbitration and produces English where it used to produce byte garbage. A full turn takes a median of 0.70 s (was 2.3 s) after exact speedups to the dialogue tier.

### Phase D: scale and ship (R5–R10)

9. **E3 scaling.** Preview done: E3-lite (18 MB) brings full-budget test from 0.2406 to 0.2346, where the old memory gained nothing. Capacity is now binding, at 37.7M circuits reclaimed ([docs](docs/r6-circuit-mixing.md#scaling-preview-e3-lite-18-mb-27-books)). Run 100–500 MB through the Rust engine under the runbook's E3 gates (quality per MB, per hour, durable bytes, retention). The spend is now justified: the mechanism separates from controls.
10. **Micro tier on the engine** under frozen ceilings (ADR 0006 item 3). The GPU backend only if a measured workload needs it.
11. **R7 human preference floors** with real raters.
12. **R8 signatures, downloader, and updater;** a recorded clean-machine run; then the signed Small candidate (ADR 0006 item 2).
13. **R9 closed alpha** with long-running brains, then R10 against Section 21.

Stop/go (Section 20) applies at every step. If Phase B cannot show that consolidation helps, record the negative result before scaling.

## 23. Final project principle

SOMA is not finished when a benchmark says a small organism can adapt. It is finished as a model product when a non-developer can obtain one persistent organism, communicate with it, teach it safely, receive useful responses, and trust it to retain, revise, and recover its learned state within explicit resource and safety bounds.

The route to that outcome is not to deny the distance from v15. It is to preserve what v15 proved, define a genuinely universal event-driven organism, separate executable runtime from downloadable mutable brain, replace brittle routing with calibrated evidence, prove cross-domain and language acquisition causally, scale only mechanisms that remain sparse and bounded, and make continuous learning an operable product property rather than a benchmark description.
