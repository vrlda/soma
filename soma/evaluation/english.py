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
from ..memory import SequenceCircuitMemory
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


def strip_gutenberg_boilerplate(data):
    """Remove Project Gutenberg header/footer by marker lines.

    Returns (stripped_bytes, stripped_bool). Never fails: unknown layouts
    keep the full bytes and report stripped=False for the manifest.
    """
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return data, False
    lines = text.splitlines(keepends=True)
    start, end = 0, len(lines)
    found_start = found_end = False
    for index, line in enumerate(lines):
        upper = line.upper()
        if "START OF" in upper and "PROJECT GUTENBERG" in upper:
            start = index + 1
            found_start = True
            break
    for index in range(len(lines) - 1, -1, -1):
        upper = lines[index].upper()
        if ("END OF" in upper and "PROJECT GUTENBERG" in upper) or upper.startswith("END OF THE PROJECT"):
            end = index
            found_end = True
            break
    if not (found_start and found_end) or start >= end:
        return data, False
    stripped = "".join(lines[start:end]).encode("utf-8")
    validate_utf8(stripped)
    return stripped, True


def build_book_manifest(entries):
    """Manifest for a book-level corpus: per-book hash/license/partition.

    entries: list of (name, path, partition, license, source_url).
    """
    import hashlib as _hashlib
    import os as _os
    books = []
    for name, path, partition, license, source_url in entries:
        with open(path, "rb") as handle:
            raw = handle.read()
        stripped, did_strip = strip_gutenberg_boilerplate(raw)
        books.append({
            "name": name,
            "path": path,
            "partition": partition,
            "license": license,
            "source_url": source_url,
            "raw_bytes": len(raw),
            "raw_sha256": _hashlib.sha256(raw).hexdigest(),
            "bytes": len(stripped),
            "sha256": _hashlib.sha256(stripped).hexdigest(),
            "boilerplate_stripped": did_strip,
            "retrieved_utc": "2026-09-11",
        })
        if not _os.path.exists(path):
            raise ValueError("missing corpus file: %s" % path)
    manifest = {
        "books": books,
        "acquisition_bytes": sum(b["bytes"] for b in books if b["partition"] == "acquisition"),
        "validation_bytes": sum(b["bytes"] for b in books if b["partition"] == "validation"),
        "test_bytes": sum(b["bytes"] for b in books if b["partition"] == "test"),
    }
    return manifest


def load_book_corpus(manifest):
    """Load stripped bytes per partition from a book manifest."""
    parts = {"acquisition": [], "validation": [], "test": []}
    for book in manifest["books"]:
        with open(book["path"], "rb") as handle:
            raw = handle.read()
        stripped, _ = strip_gutenberg_boilerplate(raw)
        parts[book["partition"]].append((book["name"], stripped))
    return parts


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


def byte_vocabulary(data):
    """Set of byte values observed in training data (transducer-level fact)."""
    return frozenset(bytes(bytearray(data)))


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


def composite_probe_bytes(seed, n_bytes=12000):
    """Novel arrangements of familiar bytes: random lowercase words.

    Every byte is frequent in English training; almost no multi-byte
    sequence was ever observed. Memory backs off here while the motor
    framing rule still fires: the division-of-labor probe.
    """
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    out = bytearray()
    while len(out) < n_bytes:
        word = "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 9)))
        out.extend((word + " ").encode("utf-8"))
    return bytes(out[:n_bytes])


def run_english_fused(partitions, seed=0, adapter_name="phase", lesion_memory=False):
    """One brain, two learners: actor motor path plus owned sequence memory.

    The actor trains on its own motor error (independent fallback); memory
    learns by observation; readout fuses by memory concentration. With
    lesion_memory the memory leaves the decision path (motor-only control).
    """
    if adapter_name not in ADAPTERS:
        raise ValueError("unknown English adapter: %s" % adapter_name)
    from ..organism import Organism
    adapter = ADAPTERS[adapter_name]()
    organism = make_english_organism(seed, input_size=adapter.input_size)
    organism.enable_sequence_memory((0, 1), max_order=16, max_circuits=131072,
                                    min_support=2, prior=0.5)
    bridge = EventBridge(organism, adapter, novelty=0.10, exploration=0.20)
    total_bits = sum(len(bit_stream(data)[0]) for data in partitions)
    tape = random.Random((seed + 1) * 1000033 + 3571)
    organism.set_exploration_tape(tuple(tape.uniform(-1.0, 1.0) for _ in range(total_bits)))
    counters = {key: 0 for key in _counter_keys(adapter_name)}
    clock_offset = 0
    reports = []
    for partition_index, data in enumerate(partitions):
        bits, ends = bit_stream(data)
        organism.reset_sequence_history()
        predictions, targets, conditions, motor_only = [], [], [], []
        learn = partition_index == 0
        for step in range(max(0, len(bits) - 1)):
            clock = clock_offset + step
            action = bridge.ingest(_step_envelopes(
                adapter_name, bits, ends, step, clock, counters))
            motor_probability = action["proposal"]["value"]
            organism.observe_sequence_event(bits[step], learn=learn)
            if lesion_memory:
                prediction = motor_probability
            else:
                prediction = organism.fuse_with_memory(motor_probability, 1)
            predictions.append(prediction)
            targets.append(bits[step + 1])
            conditions.append(ends[step])
            motor_only.append(motor_probability)
            bridge.outcome(OutcomeEvent(
                action["correlation_id"], clock,
                1.0 - abs(motor_probability - bits[step + 1]), "corpus").to_dict())
        clock_offset += len(predictions)
        report = _summarize(seed, "fused_lesion" if lesion_memory else "fused",
                            predictions, targets,
                            [1.0 - abs(p - t) for p, t in zip(predictions, targets)],
                            conditions)
        report["partition"] = partition_index
        report["learning"] = learn
        report["circuits"] = len(organism.sequence_memory.circuits)
        subset = [(p, t) for p, t, c in zip(predictions, targets, conditions) if c == 1]
        report["msb_bits_per_bit"] = bits_per_bit(
            [p for p, _ in subset], [t for _, t in subset]) if subset else None
        reports.append(report)
    organism.validate()
    bridge.close()
    return reports, organism


def run_english_brain_memory(partitions, seed=0, max_order=16, max_circuits=131072,
                             min_support=2, prior=0.5, lesion=False):
    """Same protocol as run_english_sequence_memory, state owned by a brain.

    The SequenceCircuitMemory lives inside an Organism checkpoint: identical
    predictions to the standalone runner, plus lifecycle persistence, exact
    save/load resume, and validation. The lesion path is identical too.
    """
    from ..organism import Organism
    organism = Organism.create_default(input_size=2, hidden_size=2, output_size=1, seed=seed)
    organism.enable_sequence_memory((0, 1), max_order=max_order, max_circuits=max_circuits,
                                    min_support=min_support, prior=prior)
    reports = []
    for partition_index, data in enumerate(partitions):
        bits, ends = bit_stream(data)
        organism.reset_sequence_history()
        predictions, targets, conditions, orders = [], [], [], []
        learn = partition_index == 0
        for step in range(max(0, len(bits) - 1)):
            organism.observe_sequence_event(bits[step], learn=learn)
            distribution, order = organism.sequence_distribution()
            prediction = 0.5 if lesion else distribution[1]
            predictions.append(prediction)
            targets.append(bits[step + 1])
            conditions.append(ends[step])
            orders.append(order)
        rewards = [1.0 - abs(prediction - target)
                   for prediction, target in zip(predictions, targets)]
        report = _summarize(
            seed, "brain_memory_lesion" if lesion else "brain_memory",
            predictions, targets, rewards, conditions)
        report.update({
            "partition": partition_index,
            "learning": learn,
            "max_order": max_order,
            "mean_selected_order": sum(orders) / max(1, len(orders)),
            "circuits": len(organism.sequence_memory.circuits),
            "circuits_created": organism.sequence_memory.circuits_created,
            "circuits_reclaimed": organism.sequence_memory.circuits_reclaimed,
        })
        subset = [(p, t) for p, t, c in zip(predictions, targets, conditions) if c == 1]
        report["msb_bits_per_bit"] = bits_per_bit(
            [p for p, _ in subset], [t for _, t in subset]) if subset else None
        reports.append(report)
    organism.validate()
    return reports, organism


def run_english_sequence_memory(partitions, max_order=16, max_circuits=131072,
                                lesion=False):
    """Run the generic bounded sequence-circuit memory over byte-bit events.

    The first partition is acquisition. Later partitions are read-only tests:
    working history advances, while acquired circuit counts remain frozen.
    ``lesion`` disables circuit predictions but preserves the identical stream.
    """
    if not isinstance(partitions, (list, tuple)) or not partitions:
        raise ValueError("sequence-memory run requires at least one partition")
    memory = SequenceCircuitMemory(
        (0, 1), max_order=max_order, max_circuits=max_circuits,
        min_support=2, prior=0.5)
    reports = []
    for partition_index, data in enumerate(partitions):
        bits, ends = bit_stream(data)
        memory.reset_history()
        predictions, targets, conditions, orders = [], [], [], []
        learn = partition_index == 0
        for step in range(max(0, len(bits) - 1)):
            memory.observe(bits[step], learn=learn)
            distribution, order = memory.distribution()
            prediction = 0.5 if lesion else distribution[1]
            predictions.append(prediction)
            targets.append(bits[step + 1])
            conditions.append(ends[step])
            orders.append(order)
        rewards = [1.0 - abs(prediction - target)
                   for prediction, target in zip(predictions, targets)]
        report = _summarize(
            0, "sequence_memory_lesion" if lesion else "sequence_memory",
            predictions, targets, rewards, conditions)
        report.update({
            "partition": partition_index,
            "learning": learn,
            "max_order": max_order,
            "mean_selected_order": sum(orders) / max(1, len(orders)),
            "circuits": len(memory.circuits),
            "circuits_created": memory.circuits_created,
            "circuits_reclaimed": memory.circuits_reclaimed,
        })
        reports.append(report)
    memory.validate()
    return reports, memory
