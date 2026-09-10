"""R3B recorded English environment: licensed bytes in, bit events out.

Corpus manifests carry hash, license, partition. Streams are byte-preserving;
only declared document/chapter boundaries frame the events. Targets (next
bits) and rewards are computed outside the bridge.
"""

import hashlib
import math
import random

from ..events import Event, EventBridge
from ..events.envelope import OutcomeEvent
from ..organism import Organism
from ..synapses import Synapse
from ..transducers.text_bytes import (
    PhaseTextBytesTransducer,
    TextBytesTransducer,
    encode_bytes,
    validate_utf8,
)

E0_MARGIN_BITS = 0.05
SHUFFLE_LAG = 37

ADAPTERS = {
    "raw": TextBytesTransducer,
    "phase": PhaseTextBytesTransducer,
}


def load_corpus(path, strip_boilerplate=True):
    """Load bytes, validate UTF-8, strip Project Gutenberg framing."""
    with open(path, "rb") as handle:
        data = handle.read()
    validate_utf8(data)
    text = data.decode("utf-8", errors="strict")
    if strip_boilerplate:
        start = text.find("CHAPTER I.")
        end = text.rfind("End of the Project Gutenberg")
        if end < 0:
            end = text.rfind("*** END OF THE PROJECT GUTENBERG")
        if start > 0 and end > start:
            text = text[start:end]
        data = text.encode("utf-8")
    validate_utf8(data)
    return data


def split_chapters(data):
    """Split on chapter headings; each chapter is a document lineage unit."""
    text = data.decode("utf-8", errors="strict")
    chapters, current = [], []
    for line in text.splitlines(keepends=True):
        if line.startswith("CHAPTER ") and current:
            chapters.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        chapters.append("".join(current))
    return [chapter.encode("utf-8") for chapter in chapters if chapter.strip()]


def partition_documents(documents, seed=0):
    """80/10/10 split by document lineage. Untouched test sealed by hash."""
    rng = random.Random(seed)
    order = list(range(len(documents)))
    rng.shuffle(order)
    n = len(order)
    n_test = max(1, n // 10)
    n_valid = max(1, n // 10)
    test_idx = set(order[:n_test])
    valid_idx = set(order[n_test:n_test + n_valid])
    acquisition = b"".join(documents[i] for i in range(n) if i not in test_idx and i not in valid_idx)
    validation = b"".join(documents[i] for i in sorted(valid_idx))
    test = b"".join(documents[i] for i in sorted(test_idx))
    manifest = {
        "documents": n,
        "acquisition_bytes": len(acquisition),
        "validation_bytes": len(validation),
        "test_bytes": len(test),
        "test_sha256": hashlib.sha256(test).hexdigest(),
        "license": "public-domain",
    }
    return acquisition, validation, test, manifest


def make_english_organism(seed, frozen=False, input_size=3):
    organism = Organism.create_default(
        input_size=input_size, hidden_size=6, output_size=1, seed=seed)
    for synapse in organism.graph.iter_synapses():
        if synapse.destination in organism.output_ids:
            synapse.strength = 0.0
    for input_id in organism.input_ids:
        if not organism.graph.has(input_id, organism.output_ids[0]):
            organism.graph.add(Synapse(input_id, organism.output_ids[0], 0.0, plasticity=1.0))
    organism.resources.counters["synapses"] = len(organism.graph.synapses)
    organism.enable_context_modules(max_modules=4)
    organism.actor_learning_rate = 0.0 if frozen else 0.10
    if frozen:
        organism.learning_rate = 0.0
        organism.use_threshold_router()
    return organism


def bit_stream(data):
    """Flatten bytes to MSB-first bits plus byte-end flags."""
    rows = encode_bytes(data)
    bits, ends = [], []
    for row in rows:
        for position, bit in enumerate(row):
            bits.append(int(bit))
            ends.append(1 if position == 7 else 0)
    return bits, ends


def bits_per_bit(predictions, targets):
    """Mean binary NLL in bits, with epsilon clipping."""
    total = 0.0
    for prediction, target in zip(predictions, targets):
        probability = min(1.0 - 1e-6, max(1e-6, float(prediction)))
        if target:
            total -= math.log(probability, 2)
        else:
            total -= math.log(1.0 - probability, 2)
    return total / max(1, len(targets))


def byte_unigram_bits(data):
    """Byte-frequency baseline bits/byte on the same bytes."""
    counts = [0] * 256
    for byte in data:
        counts[byte] += 1
    total = len(data)
    bits = 0.0
    for byte in data:
        probability = counts[byte] / total
        bits -= math.log(probability, 2)
    return bits / max(1, total)


def bit_marginal_bits(train_data, test_data):
    """Unigram-bit baseline: train rate applied to held-out bits."""
    train_bits, _ = bit_stream(train_data)
    rate = sum(train_bits) / max(1, len(train_bits))
    test_bits, _ = bit_stream(test_data)
    return bits_per_bit([rate] * len(test_bits), test_bits)


def bit_bigram_bits(train_data, test_data):
    """Order-1 bit baseline: P(next|current) from train, applied held-out."""
    train_bits, _ = bit_stream(train_data)
    counts = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}
    for previous, current in zip(train_bits[:-1], train_bits[1:]):
        counts[(previous, current)] += 1
    test_bits, _ = bit_stream(test_data)
    predictions = []
    for index in range(1, len(test_bits)):
        previous = test_bits[index - 1]
        predictions.append(counts[(previous, 1)] / (counts[(previous, 0)] + counts[(previous, 1)]))
    return bits_per_bit(predictions, test_bits[1:])


def _step_envelopes(adapter_name, bits, ends, step, clock, counters):
    signals = {"bit": float(bits[step]), "boundary": float(ends[step])}
    if adapter_name == "phase":
        phase = step % 8
        for index in range(3):
            signals["phase%d" % index] = float((phase >> index) & 1)
    envelopes = []
    for channel, value in signals.items():
        key = "value" if channel == "bit" else ("byte_end" if channel == "boundary" else "on")
        envelopes.append(Event(channel, "corpus", counters[channel], clock, {key: value},
                               {"trust": "signed-base"}).to_dict())
        counters[channel] += 1
    return envelopes


def _stream_bits(bridge, adapter_name, data, counters, clock_offset, organism, mode, lagged):
    bits, ends = bit_stream(data)
    steps = len(bits) - 1
    predictions, targets, rewards, conditions = [], [], [], []
    for step in range(steps):
        clock = clock_offset + step
        envelopes = _step_envelopes(adapter_name, bits, ends, step, clock, counters)
        action = bridge.ingest(envelopes)
        prediction = action["proposal"]["value"]
        target = bits[step + 1]
        predictions.append(prediction)
        targets.append(target)
        conditions.append(ends[step])
        reward = 1.0 - abs(prediction - target)
        rewards.append(reward)
        if mode == "shuffled":
            lagged.append(reward)
            delivered = lagged.pop(0) if len(lagged) > SHUFFLE_LAG else 0.0
        else:
            delivered = reward
        bridge.outcome(OutcomeEvent(action["correlation_id"], clock, delivered, "corpus").to_dict())
    return predictions, targets, rewards, conditions


def _accuracy(predictions, targets):
    return sum(1.0 for p, t in zip(predictions, targets)
               if (p >= 0.5) == bool(t)) / max(1, len(targets))


def _summarize(seed, mode, predictions, targets, rewards, conditions=None):
    report = {
        "seed": seed,
        "mode": mode,
        "bits": len(predictions),
        "bits_per_bit": bits_per_bit(predictions, targets),
        "accuracy": _accuracy(predictions, targets),
        "mean_reward": sum(rewards) / max(1, len(rewards)),
    }
    if conditions is not None:
        for name, flag in (("msb", 1), ("intra", 0)):
            subset = [(p, t) for p, t, c in zip(predictions, targets, conditions) if c == flag]
            if subset:
                report["accuracy_%s" % name] = _accuracy(
                    [p for p, _ in subset], [t for _, t in subset])
    return report


def _counter_keys(adapter_name):
    keys = ["bit", "boundary"]
    if adapter_name == "phase":
        keys += ["phase%d" % index for index in range(3)]
    return keys


def run_english(data, seed, mode="learn", max_bits=None, adapter_name="raw"):
    """Stream bytes as bit events; predict each next bit. Returns metrics."""
    if mode not in ("learn", "frozen", "shuffled"):
        raise ValueError("unknown R3B mode: %s" % mode)
    if adapter_name not in ADAPTERS:
        raise ValueError("unknown English adapter: %s" % adapter_name)
    adapter = ADAPTERS[adapter_name]()
    organism = make_english_organism(seed, frozen=(mode == "frozen"),
                                     input_size=adapter.input_size)
    bridge = EventBridge(organism, adapter, novelty=0.10, exploration=0.20)
    if max_bits is not None:
        data = data[:max_bits // 8]
    bits, _ = bit_stream(data)
    tape = random.Random((seed + 1) * 1000033 + 3571)
    organism.set_exploration_tape(tuple(tape.uniform(-1.0, 1.0) for _ in range(len(bits))))
    counters = {key: 0 for key in _counter_keys(adapter_name)}
    predictions, targets, rewards, conditions = _stream_bits(
        bridge, adapter_name, data, counters, 0, organism, mode, [])
    bridge.close()
    return _summarize(seed, mode, predictions, targets, rewards, conditions)


def run_english_continued(partitions, seed, mode="learn", adapter_name="raw"):
    """One brain across ordered partitions (e.g. train then held-out).

    Returns per-partition metrics plus the final organism and bridge. Clocks
    and counters persist across partitions: one lifetime, exact resume.
    """
    if mode not in ("learn", "frozen", "shuffled"):
        raise ValueError("unknown R3B mode: %s" % mode)
    if adapter_name not in ADAPTERS:
        raise ValueError("unknown English adapter: %s" % adapter_name)
    adapter = ADAPTERS[adapter_name]()
    organism = make_english_organism(seed, frozen=(mode == "frozen"),
                                     input_size=adapter.input_size)
    bridge = EventBridge(organism, adapter, novelty=0.10, exploration=0.20)
    total_bits = sum(len(bit_stream(data)[0]) for data in partitions)
    tape = random.Random((seed + 1) * 1000033 + 3571)
    organism.set_exploration_tape(tuple(tape.uniform(-1.0, 1.0) for _ in range(total_bits)))
    counters = {key: 0 for key in _counter_keys(adapter_name)}
    clock_offset = 0
    lagged = []
    reports = []
    for data in partitions:
        predictions, targets, rewards, conditions = _stream_bits(
            bridge, adapter_name, data, counters, clock_offset, organism, mode, lagged)
        clock_offset += len(predictions)
        reports.append(_summarize(seed, mode, predictions, targets, rewards, conditions))
    bridge.close()
    return reports, organism, bridge
