# SOMA research notebook

Newest entries first. Each entry: date, question, setup, result, meaning.
Failures are recorded as carefully as successes.

---

## 2026-10-11: pass 2 merged, with a review

**What happened.** The overnight pass 2 (entries below) was merged unchanged.
32 tests pass, and the confirmation means in
`results/pass2_composition_confirmation.jsonl` were recomputed from the raw
records and match the report (candidate 95.5%, linear memory 91.3%, frozen
66.8%, shuffled 67.9%, Oja 50.1%, competitive pool 50.0%).

**Review against the vision.** This is the first hidden learning that earns
its place: predictor neurons learn by their own local next-input prediction
error, without labels, and that beats the frozen, shuffled, and matched
controls. Cautions:

- Every hidden unit is linear and the task is linearly separable, so the
  result is denoising (finding the predictable signal subspace), not
  composition. A supervised linear model nearly matches it.
- Observation neurons are fixed random projections, and growth and pruning
  never fired in these lifetimes: structure did not live and die here.
- `predictive.py` is a second, separate brain. The goal remains one brain;
  local prediction should become a rule inside a single substrate together
  with traces, recruitment, and protected pruning.
- The seed-11 diagnosis (aliasing during acquisition, not forgetting) is
  valuable. Retention tests must now gate on acquisition first.

**Next.** Pass 2's own step 1: a nonlinear task (for example XOR-like factor
interactions) that a linear model cannot solve, with nonlinear (dendritic)
predictive units.

---

## 2026-10-11: final reproducibility audit and a capacity correction

**Question.** Can every sampled record be replayed exactly, and were early
capacity controls actually yoked through their complete tests?

**Setup.** Twelve legacy current-code receipts; five pass-2 cases each replayed
with current code and an immutable source revision (b3b6d3f, 9bd06b5, 935aae1).
Raw receipts: `pass2_source_reproducibility.jsonl`. No results edited.

**Result.** All twelve legacy cases match current code exactly. All five
pass-2 pinned-source cases match exactly, including the confirmation candidate;
only two match the latest code exactly. The Oja addition regrouped floating
point multiplication in the new predictive kernel, changing tiny loss/state
values in older predictive runs. Early Oja testing also autonomously grew one
unit (seed 0 ends at nine), despite matched training capacity. Before
confirmation, yoking was explicitly extended through testing; all confirmed
new-module variants match eight units/424 weighted connections and the same
complete structural schedule. Earlier "zero test events" wording did not apply
to that exploratory Oja control. Keep that flawed control and correct its
interpretation rather than altering its record.

**Meaning.** Exact replay needs code and runtime, not just seed/parameters.
The versioned verifier reproduces the early data without inventing compatibility
switches or overwriting it. Original `Organism` code/defaults remain unchanged.
Current confirmation is source-frozen and exactly replayed. The exploratory
provenance/capacity limitations are explicit in the final report.

---

## 2026-10-11: uncertainty precursor, sensory surprise fails

**Question.** Do local hidden prediction errors forecast the brain's own errors?

**Setup.** Five learned checkpoint seeds, cloned per six preregistered conditions,
200 unlabelled episodes each (6000 total). Intrinsic learning continues; no
outcomes supplied. Within-condition/per-seed ROC AUC for residuals and entropy.

**Result.** Familiar/held-out accuracy 94.9%/95.4%, entropy AUC .954/.941.
Doubled/quadrupled nuisance accuracy 86.9%/79.5%, confidence 88.1%/87.3%,
entropy AUC .887/.761. Reduced signal 83.2%, entropy AUC .813. View residual
AUC only .487–.556 across the ordinary conditions; query residuals also fail.
Same inputs with reversed evaluator labels: 5.1% accuracy, unchanged 88.7%
confidence, entropy AUC .046. Every seed has exactly identical probabilities
and summaries in familiar/reversed conditions. Two small seed-0 states before
and after noise8 are saved; the 200-probe replay matches probabilities exactly.

**Meaning.** Outcome entropy ranks mistakes under the learned rule, but raw
sensory surprise is not outcome self-knowledge. Confidence is miscalibrated
under shift. An unobserved label-rule change has no input evidence; do not
pretend a network could infer it without feedback. No unknown-response policy,
thresholded abstention, or self-knowledge success claim was introduced.

---

## 2026-10-11: local predictive features confirmed on held-out linear factors

**Question.** Does the frozen local-prediction hypothesis beat required controls
on unused seed streams and combinations absent from all learning experience?

**Setup.** `SELECTED_PREDICTIVE.json`, confirmation seeds 10–19, same 600
learning episodes/160 held-out episodes, all seven variants, paired streams.
All 70 runs completed. No tuning after dispatch; source hashes at import.

**Result.** All registered required-control gates PASS. Held-out mean 95.50%,
range 90.625–99.375%; seen 93.917%. Frozen/matched random 66.813%, shuffled
67.938%, Oja 50.125%, original pool 50%. Four-probe initial transfer 90%,
versus frozen 75%, shuffled 65%, Oja/original 50%. Supplementary 160-probe
frozen-hidden transfer 95.313%; random frozen 67.75%. Every candidate seed
exceeded 90% on primary held-out combinations. All confirmed new-module
capacities and structural schedules match exactly. Eight predictors/eight
observation counterparts; weighted connections 424, no structural events.

**Meaning.** Hidden basal synapses learn useful denoising through local input
prediction without task labels, and their codes transfer to missing linear
combinations. The stronger linear-memory baseline scores 91.313% and ties
initial four-probe transfer at 90%, so the task does not establish hierarchy
or necessary deep credit assignment. Shuffled seed 15 scores 100%, better
than the full candidate's 98.75%; teacher alignment helps on average, not
necessarily every individual lifetime. Frozen/weight-scale/structural controls
and the linear baseline remain visible. Phase C can now run; no real text.

---

## 2026-10-11: predictive selection frozen before confirmation

**Question.** Does the eligible exploratory hypothesis survive unused streams?

**Setup.** All five original competitive runs completed on exactly the same
600-episode task: every withheld and initial-probe score was 50%. Freeze
`SELECTED_PREDICTIVE.json`: hidden step .02, local axon homeostasis, same
noise/exposure/gates. Confirmation 10–19, every required control plus linear
memory, two independent CPython workers. No tuning afterward.

**Result.** Exploration's full required-control gates pass. Supplementary
160 independent frozen-hidden probes per seed score 96.5% candidate, 68.625%
frozen, 66.25% shuffled, 56.375% Oja. They do not replace the original four-probe
gate. Structural events are now yoked through primary testing as well as
training; actual exploratory capacities were already identical (no events).
Confirmation is pending; do not call its outcomes before completion.

**Meaning.** A single frozen brain configuration is used across the task and
future unlabelled uncertainty probes. The earlier failed candidates remain
failed. A rejected PyPy runtime check preserved scores but changed numerical
state; retain CPython and exact historical reproducibility for the study.

---

## 2026-10-11: local axon homeostasis and learned predictive geometry

**Question.** Does input prediction create useful distributed features after
controlling the frozen features' smaller scale and generic Hebbian adaptation?

**Setup.** Preregistered local source-axon mean/variance normalization in every
new-module variant. Hidden step .02/.05, unchanged 600 episodes and task,
exploratory seeds 0–4. Added Oja with exactly matched population/connections and
candidate birth/death schedule. Environment-aware weight-span diagnostics never
teach the brain or produce its answers. Added a labelled linear-memory baseline.

**Result.** Step .02: withheld 97%, seen 95.167%, independent four-probe initial
transfer 95%; frozen/matched withheld 68%, initial 85%; shuffled withheld
66.625%, initial 70%; Oja withheld 53.25%, initial 55%. Step .05: withheld
95.5%, seen 92%, initial 90%, which fails the .10 initial-transfer advantage
against frozen 85%. The .02 setting is the eligible hypothesis; original-pool
control is still running before selection/confirmation. Exact matched controls
had zero births/deaths and eight units throughout.

The fraction of learned sensory weight energy in the true signal span was
69.4% for prediction, 8.1% frozen, 23.1% Oja, 47.8% shuffled. The second/first
signal eigenvalue ratio was .571 for prediction but .047 shuffled and .039 Oja:
shuffling can retain signal while collapsing its independent dimensions.
The stronger linear-memory baseline achieved 93.875% withheld and 97.5% seen.

**Meaning.** This is evidence of local self-supervised denoising, retaining two
independent signal directions; the full required-control/confirmation gate is
not yet established. The linear baseline shows the task does not require deep
hidden credit assignment. Even a positive confirmation will support useful
predictive features and transfer on this linear task, not hierarchy or general
nonlinear composition. The candidate's hidden map itself is affine. Preserve
these qualifications rather than calling the linear baseline a failure.

---

## 2026-10-11: seed 11 failed to acquire A before it learned B

**Question.** Is the 80% return caused by pruned old representations, damaged
outcome connections, or interference? Do not retune the original selection.

**Setup.** Replayed exact unchanged streams 10/11/12, comparing complete final
states with their saved checkpoints; all three matched exactly. Independent
boundary probes, actual query winners, and a focused seed-11 replay recording
last-400 winner counts and boundary synapse profiles. New raw diagnostic files
`pass2_retention_diagnostic.jsonl` and `pass2_retention_boundary_diagnostic.jsonl`.

**Result.** Seed 11's INITIAL A final-400 accuracy was only 73.5%, before B;
B was 100%, return final-400 75%, first-50 80%. Passing seeds 10/12 learned A
at 100%. In seed 11, cell 16 won both (0,1) and (1,1) throughout initial A's
last 400 trials (97 and 114 occurrences), which require opposite answers.
All ten first-50 return errors selected this same cell. Its seven sensory
connections and both prediction connections survived B, their strengths
exactly unchanged. First-factor afferents (.0505/.0576) blurred the small
older traces; the second-factor response was strong (.8000). It is a surviving
representation collision, not erased output weights or neuron deletion.

**Meaning.** The earlier report treated seed 11 as failed retention, but the
acquisition failure predates the context switch. The original gate omitted
initial-A acquisition. Keep its scores unchanged and correct the causal
interpretation: return accuracy is not a clean forgetting measure for a mapping
that never became reliable. A future retention protocol must require acquisition
first. Boundary probes with all learning disabled also disable variance
adaptation and suppress winners even in passing brains; these probes alone
cannot diagnose deletion. Actual uninterrupted trajectories provide the evidence.

---

## 2026-10-11: distributed future prediction, first composition gates failed

**Question.** Can self-supervised hidden prediction support combinations absent
from every training experience, beyond fixed random features?

**Setup.** Preregistered two independent four-valued factors, random sensory
directions plus independent orthogonal nuisance, three views and a query.
Twelve combinations in 600 learning episodes; four held out, 160 unlabelled
test episodes. Eight distributed next-input predictors learn their own
observation activity locally. Outcomes teach only the readout. Initial steps
.2, .02 and .05, seeds 0–4; frozen/shuffled/exact-capacity controls on the same
streams. No structural births/deaths occurred at registered costs. Competitive
control is running separately at the identical task and exposure.

**Result.** Withheld/seen accuracy: step .2 = 80.875%/80.833%; .02 =
82.75%/86.667%; .05 = 85.375%/88.5%. Every condition FAILS the unchanged
90% gates. Frozen/matched withheld 53.625%; shuffled about 50–52%.
All settings and attempted streams are retained in separate result files.

**Meaning.** Local hidden learning helps this task but is not yet a confirmed
composition result. Frozen random projections have smaller activity than
learned ones, confounding readout speed. Preregister local axon homeostasis
for every variant before attributing the difference solely to predictive
structure. Do not expose confirmation seeds until selection is frozen.

---

## 2026-10-10: overnight pass merged, with a review

**What happened.** A second agent built `brain/organism.py` and the controlled
experiments below in one overnight pass, without GitHub write access. Its commits
were merged unchanged; a follow-up commit removed generated files
(`SUMMARY.json`, SVG copies of the figures) and restore notes. Tests pass, and
`python -B -m experiments.report` regenerates `SUMMARY.md` byte-identically from
the raw results.

**Review against the vision.** The work respects the substrate rules (local
learning, no backpropagation, growth/pruning without size caps, lifelong
save/resume) and is honestly controlled. Cautions for whoever continues:

- The competitive hidden pool is winner-take-all prototype matching with
  recruitment (close to ART). On small finite tasks, success can mean one
  recruited neuron per memorized combination. Do not mistake it for learned
  structure; test generalization to unseen combinations.
- Only outcome synapses learn from error, and all tasks are labelled
  classification. The brain does not yet predict its own next input, and hidden
  neurons get no learning signal: credit assignment (VISION.md section 6.1) is untouched.
- Mechanisms accumulate as `Config` switches with per-task selections. That is
  right for ablations, but the goal is one brain with one rule set.

**Next.** Its own next step 2: local prediction between hidden populations,
tested on composition of unseen combinations, with frozen and shuffled controls.

---

## 2026-10-10: combined lifetime confirmation, with one failed return

**Question.** Do longer temporal representation and consolidation work together
on independent lifelong streams, without replay or reset?

**Setup.** Frozen `SELECTED_COMBINED.json`, gap 8, 1200 A / 1200 reversed B /
400 A episodes, confirmation seeds 10–19, paired protection-off control.
All 20 runs completed; final states saved in `checkpoints/combined_confirmation`.

**Result.** Both variants learned B at 100%. Candidate first-50 A return mean
98.0%, range 80–100%; protection-off 78.2%, range 64–90%. Nine candidate seeds
meet the 90% return gate; seed 11 fails at 80%. All ten candidate first return
responses were correct, versus five protection-off. Candidate final hidden
population mean 39.8, protection-off 35.5. No retuning after confirmation.

**Meaning.** Protection improves mean retention, but the longer lifetime is
not robust across all seeds. First-50 metrics include subsequent learning and
are not frozen recall. Preserve seed 11 as the next diagnostic, not an outlier
to discard. Gap-2 lifetime and gap-8 grammar remain separately confirmed.

**Pass record.** 1537 JSONL records, 1534 complete, three partial budget stops;
39 files including the empty aborted recruitment file. Twenty behavioral tests
pass; twelve representative historic runs reproduce exactly. Full final report:
`RESEARCH_REPORT.md`.

---

## 2026-10-10: local homeostasis extends temporal representation, confirmed

**Question.** Can a weaker old trace matter alongside a stronger recent input
without exposing output errors to hidden cells?

**Setup.** Preregistered source-variance normalization: each afferent maintains
its own incoming-trace mean/variance, rate .01, floor .0001. Competitive
mismatch divides by that variance. Recruitment, outcome learning, and the
finite grammar remain unchanged. Five exploratory seeds; frozen selection in
`SELECTED_HOMEOSTASIS.json`, then ten confirmation streams. Controls remove
normalization, freeze scale statistics, or remove recruitment.

**Result.** Confirmation candidate 100% at gaps 0/2/8. Gap-8 normalization-off
and scale-frozen controls average 49.825%; recruitment-off also stays near
chance. Initial gap-16 runs score 50.1%: that limit is still present. Normalized
representations cost about 30–35 cells versus roughly 14–17 without scaling.
The result supports scaling PLUS recruitment; it does not isolate a benefit
independent of the added representational capacity.

**Meaning.** Useful information was already in the traces. Local homeostasis
changes which differences the neurons represent. This is an experimentally
earned primitive, not global backpropagation or a task-specific parity feature.
The combined longer-contextual lifetime passed initial gates (100% new,
99.2% old return; protection-off 80.0%). Its final paired confirmation is
recorded separately when complete; do not infer it from separate component tests.

---

## 2026-10-10: confidence and developmental-history diagnostic

**Question.** Do internal prediction/evidence signals distinguish familiar
knowledge, novel contexts, and longer delays?

**Setup.** Completed combined-lifetime checkpoints, initial seeds 0–4, cloned
per condition. Forty episodes each, zero outcome labels supplied. Intrinsic
learning continues. Probe familiar contexts, unseen context values .5/-.5,
and familiar context at unseen gaps 16/32/64. No unknown-response rule or
abstention threshold exists in SOMA. `results/confidence_diagnostic.jsonl`.

**Result.** Familiar contexts: 100% accuracy, about 99% confidence. Novel
contexts: about 66–67% confidence (truth undefined, so no accuracy claim).
Transfer from the learned gap-8 lifetime to gap 16: 95.5% accuracy with 97.2%
confidence, without receiving any new outcomes. That is different from the
50.1% gap-16 result when starting from scratch. Gaps 32/64: about 51–52%
accuracy despite about 66% confidence.

**Meaning.** Developmental history matters: an already acquired representation
can transfer farther than a fresh brain discovers it. Confidence decreases on
novelty but is imperfect and overconfident on harder delays. This is not learned
self-knowledge, a calibrated ignorance detector, or an ability to explain why
something is unknown. Evidence and predictive entropy are candidate signals
for a future local metacognitive experiment, not a hard-coded answer policy.

---

## 2026-10-10: consolidated contextual lifetime, confirmed

**Question.** Can one growing/pruning brain keep an earlier temporal mapping
while learning its reversal in an observable new context?

**Setup.** `PROTOCOL.md` lifetime and consolidation entries. Two binary symbols
separated by two neutral events; seven primitive sensory channels include
context, two outcomes. 1200 original-context episodes, 1200 reversed-context
episodes, 400 original-context return episodes, no resets/replay. Score before
teaching. Select `SELECTED_LIFETIME.json` after five exploratory seeds, then
confirm ten independent streams (seeds 10–19). Protect pruning with synapse-local
match/use evidence decaying at .99999 per update; retain active representations
and permit missing prediction connections to regrow locally. Controls match
other settings and remove pruning protection or pruning itself.

**Result.** Confirmation: protected candidate 100% new-context final 400 and
100% old-context first 50; removing protection 100% new and 82.4% old return;
no pruning 100% on both. The first old response also precedes relearning;
its per-seed scores are recorded. Original pruning baseline returned at 86.4%
across five seeds, below the preregistered 90% gate. Full records/checkpoints:
`results/lifetime_confirmation.jsonl`, `checkpoints/lifetime_confirmation/`.

**Meaning.** Consolidation evidence earns its place in this small lifetime by
preventing structural forgetting. The test supports rare-path preservation,
not unlimited lifelong knowledge, general reasoning, or biological sleep.
A longer-gap combination with afferent normalization is being tested separately.

---

## 2026-10-10: temporal recruitment and its limits, confirmed

**Question.** Can local primitives form a representation that combines two
symbols across time, rather than just remembering one symbol?

**Setup.** Finite parity-terminal grammar; task rule exists only in experiments.
Sensory drive peak traces (.8 decay) feed competitive neurons. Start at one
cell, grow where persistent AND current input mismatch remains unmet, on every
experience rather than only labelled outcomes. Newborn afferents copy local
traces. Generic mismatch vigilance .03; smooth size cost, no size ceiling.
Selected settings frozen in `SELECTED_GRAMMAR.json`; ten confirmation seeds.

**Result.** 100% final-window accuracy at gaps 0 and 2, approximately 50% at
8. Removing recruitment: 51.3%, 49.4%, 49.4% at gaps 0/2/8. Removing input
traces: 49.8% at each gap. Freezing afferent weights ALSO scored 100% at 0/2:
recruitment and local outcome learning suffice here; afferent adaptation is
not established as necessary. End populations averaged 15.6 and 16.6 cells at
0/2, from one. Earlier fixed-16-cell candidate: 94.4%, 60.1%, 50.1% on initial
seeds. See `results/grammar_confirmation.jsonl` and earlier preserved results.

**Failure mechanism.** Query-only recruitment kept growing on mismatch from
blank/prefix experiences. The full run was interrupted for excessive cost;
a separate 120-episode reproduction reached 121 hidden cells and only 63.75%
accuracy. `results/recruitment_failure_diagnostic.json` is explicitly partial.
Aligning births with the currently unmet experience removed that failure.
Removing vigilance cost about 45 cells versus about 16 and scored only 89.4%
at gap 2 on initial seeds. Pruning preserved short-task accuracy but repeatedly
removed unlabelled prefix cells, producing substantial birth/death churn.

**Meaning.** A small population can acquire finite temporal conjunctions with
local rules and structural recruitment. It is not an emergent hierarchy or
compositional generalization. The surviving gap-8 failure remains a failure;
real text is deferred.

---

## 2026-10-10: what survives the longer delay (oracle diagnostic)

**Question.** Does grammar failure at longer gaps mean that the sensory trace
has lost the early symbol?

**Setup.** A pre-outcome observer copies traces; an explicitly hand-designed
ORACLE reads each sensory alphabet's larger trace and applies the environment's
parity rule. It does not teach or answer for SOMA. 1200 episodes, final 400,
five seeds per gap, selected grammar unchanged.

**Result.** Oracle 100% at gaps 8, 16, 64, while SOMA stayed near chance.
`results/trace_information.jsonl` includes pre-teaching trace samples.

**Meaning.** The information is still present. Ordinary Euclidean matching
can ignore the weaker older signal. A separately preregistered local variance
normalization experiment investigates representation resolution. The oracle
is an information upper bound, not an architectural shortcut inside SOMA.

---

## 2026-10-10: elementary memory and the mechanisms that did not earn a claim

**Question.** Which primitive rules explain elementary sequence, delay, context,
replay, and structural results?

**Setup.** Initial candidate and follow-ups in `PROTOCOL.md`, all seeds/configs
preserved in `results/`. Prediction always precedes outcome exposure. Five
initial seeds; selected event-trace and competitive settings then ten
confirmation streams.

**Results.** Initial sequence: 100% candidate, 25% disabled learning. Initial
averaging-trace delay candidate: 82.9%, 56.0%, 52.6%, 50.8% at gaps 0/4/16/64.
A full averaging-trace bank did not solve intervening distractors. Peak/event
traces did: selected confirmation 100% at 0/4/16/64; trace-only neurons also
scored 100% at 0/4/16. Removing all temporal paths scored 49.55%, 50.0%,
49.85% at 0/4/16. Conflicting-cue confirmation: selected candidate and trace-only
control both 100%. These are synaptic associations, not deep credit assignment.

Competitive context separation retained both meanings at 100%, whereas the
linear-only control's initial old-context probe was 19.5% in confirmation.
Frozen competitive afferents ALSO passed: do not call this necessary learned
organization. Replay labelled sleep and equal wake replay were numerically
identical; no sleep-specific consolidation was shown. Some initial replay
runs interfered with the new context rather than helping it.

**Matched structural control.** On random labels, prediction-error growth from
one cell made 150 births and 86 deaths in each seed, ending at 65 cells with
51.28% accuracy. Representation-novelty growth with the same initial size,
learning rule, interval, cost, pruning, and stream made 4–11 births; accuracy
remained chance. Novelty avoids interpreting irreducible label noise as an
endless need for neurons, but it does not solve unpredictable labels.

**Meaning.** Keep the evidence for local temporal state and representation
recruitment. Hidden Oja learning, outcome metaplasticity on elementary tasks,
and sleep-labelled replay have not independently established the larger
claims in the vision. Every failed candidate remains reproducible.

---

## 2026-10-10: initial local neural candidate (historical partial snapshot)

**Question.** Can local outcome learning plus leaky recurrent neurons and
averaging synaptic traces learn a sequence and retain a cue through distractors?

**Setup.** Registered `experiments/PROTOCOL.md` before execution. Standard-library
reference in `brain/organism.py`; predictions precede outcome exposure. Sparse
recurrent hidden neurons use local Oja updates. Prediction synapses use source
trace times their own target neuron's outcome error; no error propagates into
hidden cells. Five seeds, prequential metrics, controlled ablations. Six checks
passed, including exact checkpoint/RNG continuation and outcome causality.

**Partial results, not final.** All five candidate seeds scored 100% on the final
windows of the two four-symbol cycles; no-learning scored 25%. Removing hidden
plasticity also scored 100%: this task does not establish useful hidden learning.
The completed candidate runs at gaps 0, 4, 16 scored 82.9%, 56.0%, 52.6%.
The delayed-memory gate failed. Initial conflicting-cue runs also failed.
Full results and later confirmation will be recorded separately.

**Interpretation.** This is a functioning local readout, not deep credit
assignment, a learned hierarchy, or a working lifelong brain. A randomly
assigned averaging trace scales a one-off event by `(1 - decay)`: long time
constants preserve time but attenuate a brief cue. Randomly omitting particular
trace timescales also makes the primitive interface seed-dependent. Test that
mechanism explicitly next; do not conceal this failure by changing its gate.

---

## 2026-10-02: fresh start

**What happened.** The repository was reset around [`VISION.md`](VISION.md).
Everything earlier moved to `archive/` (last runnable commit `b6606cd`).
The earlier work drifted toward count tables, hand-built features, and
product engineering. Its lessons are in `VISION.md` section 8.

**Starting point for the new work:**

- Build a small, clean brain: neurons and synapses that hold numbers, with
  each kind of change as a separate, swappable rule (local learning, global
  modulators, growth and death, sleep).
- Test the first rules on tiny observable tasks: predict a repeating
  pattern, remember across a delay, learn a toy grammar, recall after
  sleep. Compare against trivial controls and ablations.
- Move to real text (the modern corpus in `data/modern/`) only once the
  small tasks are mastered.

**Baselines to beat eventually** (from the archive, Simple English
Wikipedia validation, bits per byte; lower is better): about 1.99 for
byte-context count tables at a 4M-entry budget, and about 1.88 for the same
plus copying from earlier in the document.
