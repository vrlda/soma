# SOMA: the vision

**This is the north star of the project. Read it fully before doing any work.
When in doubt, re-read it. If a plan, an idea, or a result contradicts it,
the plan is wrong, not the vision.**

Written 2026-10-02 from the owner's own words, after an agent had drifted away
from the idea. The rest of the repository serves this document.

---

## 1. What SOMA is

SOMA is an attempt to build an AI with a **synthetic brain**: a network of
neurons and synapses, instead of the architectures everyone else uses
(Transformers, LLMs, fixed neural networks trained by backpropagation).

It is **a research dream, not a product.** There is no deadline and nothing
to ship urgently. The goal is a **breakthrough**: to find the rules that let a
network of artificial neurons and synapses learn, think, and forget the way a
brain does.

We believe it is possible. It breaks no law of the universe, and every brain
proves it: neurons and synapses, learning only from local signals, with no
backpropagation and no Transformer, produce language, reasoning, and knowing
what you don't know, on about 20 watts. The question is not *whether* it can
be done, but *which principles* we are still missing.

## 2. The structure, in the owner's words

> A neuron is some kind of a point in the brain which holds different
> parameters, which are numbers, including plasticity and others. Each neuron
> has lots of connections to other neurons. These connections are synapses,
> and they have their own parameters as numbers, like the strength of the
> connection and others.
>
> All of these numbers in neurons and synapses change continuously: locally,
> globally, with new data received, or internally, rearranging themselves on
> the data they already have, like a brain rearranges and cleans itself up
> while the organism sleeps.
>
> When the numbers change, new synapses can be created or destroyed, and new
> neurons can be created or destroyed, without any particular limits, but
> with boundaries that mostly affect the speed of these changes: the speed of
> growth and destruction (learning and forgetting).
>
> SOMA has "parameters". It's just that these parameters have their own
> parameters. These parameters are neurons. SOMA copies the biological brain
> structure. It can learn, think, and forget. It can learn anything and has
> infinite possible capabilities with enough training. It "trains"
> continuously, throughout conversations or any work it does in any other
> modality.

So the architecture is simple to describe: **neurons and synapses, each
holding numbers, and rules for how those numbers change.** The hard part, and
the whole research problem, is finding the right rules.

## 3. Core principles (do not violate)

1. **Neurons and synapses are the substrate.** Knowledge lives in neurons
   and synapses and their numbers. A neuron holds state such as activity,
   threshold, plasticity, age, usefulness, and surprise. A synapse holds
   state such as strength, plasticity, a trace of recent use, age, and
   usefulness. Tables of counts, hashed feature lists, or external databases
   are not the brain. Some may serve as temporary scaffolding or baselines,
   but never as the answer.
2. **Learning is local and continuous.** A synapse changes using
   information available where it sits (the activity of its two neurons,
   its own trace and history), plus broadcast signals. There is no global
   backpropagation through the network, and no separate "training mode"
   versus "use mode". Every experience, including every conversation, is
   learning.
3. **Four ways the numbers change:**
   - *locally*: each synapse and neuron from its own activity and error;
   - *globally*: brain-wide signals, like the brain's neuromodulators
     (reward, surprise, attention, uncertainty), that gate *when* and *how
     much* to learn, not *what*;
   - *from new data*: every input is experience;
   - *internally, during sleep*: replaying and reorganizing what was
     already experienced, consolidating what matters, weakening and pruning
     the rest, freeing capacity.
4. **Structure lives and dies.** Synapses are created where neurons
   usefully work together and removed when useless. Neurons are born where
   the network keeps failing to represent something (persistent surprise)
   and die when they contribute nothing.
5. **No hard size limits; limits on speed.** Instead of a fixed number of
   neurons, every change has a cost and a rate: how fast synapses can grow,
   how fast plasticity drops as knowledge consolidates, how fast pruning
   happens. Size settles at a balance with experience, as in a brain.
6. **One lifelong brain.** It is never retrained from scratch. It is
   saved and resumed, and it keeps learning for its whole life. It forgets
   what is unimportant and keeps what is important.
7. **Biology is the guide, not a cage.** Copy the brain's principles
   (neurons, synapses, plasticity, neuromodulation, sleep, growth and
   pruning, dendrites, timing), not its chemistry. Simplify freely, but never
   replace the principle with a conventional shortcut.

## 4. What it should eventually be able to do

- **Talk like a person.** Conversational, human-like, and fluent, so that
  it feels like talking to a regular LLM. It gets there through its own
  neurons, not by borrowing one.
- **Know what it doesn't know.** When it lacks knowledge, it should
  recognize that and be able to reason about why ("I've never learned
  anything about that"). That awareness should come from its own internal
  signals: how certain its neurons are and how much evidence they hold. It
  should not be a hard-coded rule.
- **Learn anything, in any modality.** Text first, but the same neurons
  and rules should, in principle, learn vision, sound, and action.
- **Keep learning from use.** Teach it in conversation and it remembers.
  Correct it and it updates. Leave it alone and it consolidates during sleep.

These are the long-term direction, not near-term promises. Each step toward
them is earned by experiments.

## 5. What SOMA is NOT (anti-drift list)

An agent working on SOMA has already drifted once. These are the traps:

- ❌ **Not a Transformer, an LLM, or an LLM with a SOMA memory bolted on.**
  Do not propose using a pretrained language model as SOMA's "voice". That
  was proposed once and rejected.
- ❌ **Not backpropagation.** Gradient methods may appear only as clearly
  labelled *baselines* for comparison, never inside SOMA.
- ❌ **Not retrieval-plus-reader engineering.** Searching documents and
  extracting answers with hand-designed features can produce numbers fast,
  but it is not a brain learning.
- ❌ **Not count tables or n-gram statistics labelled as "circuits".** The
  2026 circuit-mixing memory was a strong text predictor, but it had no
  neurons or synapses. It remains useful only as a baseline and a lesson.
- ❌ **Not a product race.** Do not optimize for shipping, packaging,
  signatures, or release candidates until the brain itself works.
- ❌ **Not hand-designed solutions.** Hardcode only the primitive
  mechanisms that let structure develop (neurons, synapses, the rules).
  Never hardcode the final organization or task-specific intelligence.

If you find yourself reaching for one of these because the real thing is
hard, stop: **the hard part is the research.** Write down what failed and why,
and work on the rule that is missing.

## 6. Where the real problem is

The structure is easy. The breakthrough lies in **the rules**, the "physics"
by which the numbers change, so that neurons acting only on local information
make the whole network become intelligent. The known open problems:

1. **Credit assignment without backpropagation.** How does a synapse deep
   inside the network learn what helps, using only local information and
   broadcast signals? Promising leads: prediction-driven learning (each
   neuron predicts its input and learns from its own surprise; predictive
   coding), three-factor rules (local eligibility × global modulator),
   dendritic computation (branches carrying separate signals).
2. **Long-range time.** How do events many steps apart get connected?
   Leads: eligibility traces at several timescales, recurrent activity,
   persistent neuron states, replay during sleep.
3. **Hierarchy.** How do neurons form layers of meaning (letters → words →
   phrases → ideas) by themselves? Leads: competition between neurons,
   predicting the level below, growth where surprise persists.
4. **Stability versus plasticity.** How do you learn new things without
   erasing old ones? Leads: metaplasticity (consolidated synapses become
   harder to change), sleep consolidation, dormant structure.
5. **Self-knowledge.** How does the network know how certain it is? Leads:
   uncertainty as a neuromodulator, evidence carried in synapse state.

## 7. How we work

- **Research partners, not a feature factory.** The owner shares the dream;
  the agent researches, builds, measures, and reports honestly.
- **Small experiments first.** A mechanism must work on a tiny, fully
  observable problem (predicting a pattern, remembering across a delay, a
  toy grammar, recall after sleep) before it touches real text. If it can't
  do the small thing, scale won't save it.
- **Always a control.** Every claim is compared with a baseline (a trivial
  predictor, the same network with the mechanism removed, or a conventional
  method). An ablation must show that each mechanism earns its place.
- **Honest records.** Every experiment goes into `NOTEBOOK.md`, with the
  question, the setup, the result, and what it means, *especially failures*.
  Never relabel a failure as a success. Never move the goalposts after
  seeing results.
- **Observable brains.** Count neurons, synapses, births, deaths,
  plasticity, and activity over time. We should be able to watch the brain
  develop.
- **Reproducible.** Fixed seeds, saved checkpoints, recorded data.

## 8. What we already know (lessons from 2025–2026)

The repository's history (`archive/`) holds a large body of earlier work.
The useful lessons:

- **The kernel worked on small synthetic tasks.** The v1 organism, with
  stateful cells, sparse synapses, local plasticity, growth, pruning, and
  metaplasticity, passed continual-learning gates in synthetic worlds.
  Growth, pruning, and consolidation are buildable.
- **English through that organism was weak.** The rules did not discover
  deep structure or long-range dependencies in text. This is exactly
  problem 1 and problem 2 above, and the reason the project drifted to
  count tables.
- **Count-based "circuits" predict text well but are not a brain.** On
  modern Wikipedia text: about 1.88 bits/byte at a 4M-entry budget. Copying
  from earlier in the document (a "match" mechanism) gained about 0.1
  bits/byte. These are the numbers a neural SOMA should eventually beat.
- **Metaplasticity reduced forgetting** across books by 15%, and
  pressure-aware growth (grow only when evidence beats what would be
  evicted) helped under limited capacity. Both ideas should carry over to
  neurons and synapses.
- **More data helped only with more capacity.** Memory pressure, meaning what
  gets forgotten and when, mattered as much as the learning rule.
- **Benchmarks exist.** A modern corpus pipeline (Simple English Wikipedia,
  SQuAD questions, OpenAssistant dialogues, with sealed test splits) is in
  `scripts/build_modern_corpus.py`. A hand-engineered extractive reader
  reached 41.6% F1 on SQuAD validation. It is a baseline, not SOMA.

## 9. For any agent starting work

1. Read this file completely.
2. Read the last entries of `NOTEBOOK.md` to see where the research stands.
3. Before building, ask: *does this keep knowledge in neurons and synapses,
   learned by local rules, continuously?* If not, don't build it, or label
   it clearly as a baseline.
4. Work in small, measured experiments. Record every one in `NOTEBOOK.md`.
5. Tell the owner the truth about results, good or bad.
