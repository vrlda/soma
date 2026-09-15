"""Bounded local predictive circuits over ordered discrete events.

Each circuit represents one observed suffix and stores only local successor
evidence.  Circuits grow from experience, back off when evidence is sparse,
and are reclaimed deterministically under a hard budget.  The mechanism is
domain-neutral: symbols are declared by the caller and carry no text meaning.
"""

import math


class SequenceCircuitMemory(object):
    """Online variable-order prediction with bounded structural state."""

    VERSION = 1

    def __init__(self, symbols, max_order=16, max_circuits=131072,
                 min_support=2, prior=0.5):
        self.symbols = tuple(symbols)
        if len(self.symbols) < 2 or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("sequence memory needs at least two unique symbols")
        self.symbol_index = {symbol: index for index, symbol in enumerate(self.symbols)}
        self.max_order = int(max_order)
        self.max_circuits = int(max_circuits)
        self.min_support = int(min_support)
        self.prior = float(prior)
        if self.max_order < 0:
            raise ValueError("maximum order must be nonnegative")
        if self.max_circuits < self.max_order + 1:
            raise ValueError("circuit budget is too small for maximum order")
        if self.min_support < 1:
            raise ValueError("minimum support must be positive")
        if not math.isfinite(self.prior) or self.prior <= 0.0:
            raise ValueError("prior must be positive and finite")
        self.history = []
        self.circuits = {(): self._new_circuit(0)}
        self.events_seen = 0
        self.circuits_created = 1
        self.circuits_reclaimed = 0
        self.chunks = {}
        self.chunks_promoted = 0

    def _new_circuit(self, step):
        return {
            "counts": [0] * len(self.symbols),
            "last_used": int(step),
            "reuse": 0,
        }

    def reset_history(self):
        """Start a new stream without erasing acquired circuits."""
        self.history = []

    def _contexts(self):
        limit = min(self.max_order, len(self.history))
        return [tuple(self.history[-order:]) if order else ()
                for order in range(limit + 1)]

    def _reclaim(self):
        """Reclaim weak, old, specific circuits; root is permanent.

        Circuits younger than two full horizons keep a consolidation
        window: reclamation must not eat structures still being acquired
        within the current episode.
        """
        if len(self.circuits) < self.max_circuits:
            return
        protected = set(self._contexts())
        grace = 2 * max(1, self.max_order)
        candidates = []
        for context, circuit in self.circuits.items():
            if not context or context in protected:
                continue
            if self.events_seen - circuit["last_used"] < grace:
                continue
            support = sum(circuit["counts"])
            candidates.append((support, circuit["reuse"], circuit["last_used"],
                               -len(context), context))
        if not candidates:
            return
        candidates.sort()
        batch = max(1, min(1024, self.max_circuits // 100))
        for item in candidates[:batch]:
            del self.circuits[item[-1]]
            self.circuits_reclaimed += 1

    def distribution(self):
        """Predict from the longest sufficiently supported suffix."""
        chosen = self.circuits[()]
        chosen_order = 0
        for context in reversed(self._contexts()):
            circuit = self.circuits.get(context)
            if circuit is None or sum(circuit["counts"]) < self.min_support:
                continue
            chosen = circuit
            chosen_order = len(context)
            break
        denominator = sum(chosen["counts"]) + self.prior * len(self.symbols)
        probabilities = {
            symbol: (chosen["counts"][index] + self.prior) / denominator
            for index, symbol in enumerate(self.symbols)
        }
        return probabilities, chosen_order

    def probability(self, symbol):
        if symbol not in self.symbol_index:
            raise ValueError("unknown sequence symbol")
        return self.distribution()[0][symbol]

    def observe(self, symbol, learn=True, weight=1, min_weighted_order=0):
        """Observe one event and locally update its predecessor circuits.

        Weight scales the count increment: trusted sources (corrections,
        approved documents) learn faster than background chatter. Weights
        are positive integers; unobserve() reverses them exactly.
        min_weighted_order restricts full weight to specific (long)
        contexts; shorter shared contexts learn at weight 1. This is the
        episodic-vs-semantic distinction: single episodes must not rewrite
        generic statistics.
        """
        if symbol not in self.symbol_index:
            raise ValueError("unknown sequence symbol")
        if not isinstance(learn, bool):
            raise ValueError("learn flag must be boolean")
        if isinstance(weight, bool) or not isinstance(weight, int) or weight < 1:
            raise ValueError("weight must be a positive integer")
        if not isinstance(min_weighted_order, int) or min_weighted_order < 0:
            raise ValueError("min_weighted_order must be a nonnegative integer")
        if learn:
            symbol_index = self.symbol_index[symbol]
            for context in self._contexts():
                active_weight = weight if len(context) >= min_weighted_order else 1
                circuit = self.circuits.get(context)
                if circuit is None:
                    self._reclaim()
                    if len(self.circuits) >= self.max_circuits:
                        continue
                    circuit = self._new_circuit(self.events_seen)
                    self.circuits[context] = circuit
                    self.circuits_created += 1
                circuit["counts"][symbol_index] += active_weight
                circuit["last_used"] = self.events_seen
                circuit["reuse"] += 1
        self.history.append(symbol)
        if len(self.history) > self.max_order:
            if self.max_order <= 0:
                del self.history[:]
            else:
                del self.history[:-self.max_order]
        self.events_seen += 1

    def unobserve(self, symbols, preceding_history, weight=1, min_weighted_order=0):
        """Exactly reverse a previous weighted observation sequence.

        preceding_history is the history list as it stood before the taught
        sequence (snapshot it before teaching). Counts decrement by weight;
        circuits that hit all-zero counts are dropped. Raises when the
        reversal is impossible (never taught, or already removed).
        """
        symbols = list(symbols)
        base = list(preceding_history)
        if any(symbol not in self.symbol_index for symbol in symbols + base):
            raise ValueError("unknown sequence symbol in unobserve")
        if isinstance(weight, bool) or not isinstance(weight, int) or weight < 1:
            raise ValueError("weight must be a positive integer")
        for position, symbol in enumerate(symbols):
            window = base + symbols[:position]
            limit = min(self.max_order, len(window))
            for order in range(limit + 1):
                context = tuple(window[-order:]) if order else ()
                active_weight = weight if len(context) >= min_weighted_order else 1
                circuit = self.circuits.get(context)
                if circuit is None:
                    raise ValueError("unobserve found no circuit: teach first")
                index = self.symbol_index[symbol]
                circuit["counts"][index] -= active_weight
                if circuit["counts"][index] < 0:
                    raise ValueError("unobserve would drive counts negative")
            pruned = [context for context, circuit in self.circuits.items()
                      if context and sum(circuit["counts"]) <= 0]
            for context in pruned:
                del self.circuits[context]
        self.validate()

    def promote_chunks(self, min_order=8, min_reuse=10, min_concentration=0.9,
                       max_chunks=4096, max_merged_reuse=None):
        """Merge extension families into promoted chunk circuits.

        A circuit with a long, often-reused, concentrated context becomes a
        chunk: descendant extension circuits fold their counts into it and
        are removed. Prediction code is untouched; backoff simply finds
        richer aggregated counts. Returns the number of chunks promoted.
        """
        if min_order < 1 or min_reuse < 1 or max_chunks < 1:
            raise ValueError("chunk promotion bounds must be positive")
        if not 0.0 < float(min_concentration) <= 1.0:
            raise ValueError("chunk concentration must be in (0, 1]")
        candidates = []
        for context, circuit in self.circuits.items():
            if len(context) < min_order or context in self.chunks:
                continue
            total = sum(circuit["counts"])
            if circuit["reuse"] < min_reuse or total <= 0:
                continue
            concentration = max(circuit["counts"]) / total
            if concentration < float(min_concentration):
                continue
            candidates.append((circuit["reuse"], concentration, context))
        candidates.sort(key=lambda item: (-item[0], -item[1], repr(item[2])))
        promoted = 0
        for _, _, context in candidates:
            if len(self.chunks) >= max_chunks or context not in self.circuits:
                continue
            chunk = self.circuits[context]
            merged = 0
            for other in [c for c in self.circuits if len(c) > len(context) and c[-len(context):] == context]:
                if other in self.chunks:
                    continue
                if max_merged_reuse is not None and self.circuits[other]["reuse"] > max_merged_reuse:
                    continue
                descendant = self.circuits.pop(other)
                chunk["counts"] = [a + b for a, b in zip(chunk["counts"], descendant["counts"])]
                chunk["reuse"] += descendant["reuse"]
                chunk["last_used"] = max(chunk["last_used"], descendant["last_used"])
                merged += 1
            total = sum(chunk["counts"])
            self.chunks[context] = {
                "merged": merged,
                "reuse": chunk["reuse"],
                "concentration": (max(chunk["counts"]) / total) if total > 0 else 0.0,
            }
            self.chunks_promoted += 1
            promoted += 1
        return promoted


    def validate(self):
        if len(self.circuits) > self.max_circuits or () not in self.circuits:
            raise AssertionError("sequence circuit budget/root invariant failed")
        if len(self.history) > self.max_order:
            raise AssertionError("sequence history exceeds maximum order")
        for symbol in self.history:
            if symbol not in self.symbol_index:
                raise AssertionError("sequence history contains unknown symbol")
        if not isinstance(self.chunks, dict) or not isinstance(self.chunks_promoted, int):
            raise AssertionError("sequence chunk state is invalid")
        for context in self.chunks:
            if context not in self.circuits:
                raise AssertionError("sequence chunk has no circuit")
        for context, circuit in self.circuits.items():
            if len(context) > self.max_order or any(symbol not in self.symbol_index for symbol in context):
                raise AssertionError("sequence circuit context is invalid")
            counts = circuit.get("counts")
            if not isinstance(counts, list) or len(counts) != len(self.symbols):
                raise AssertionError("sequence circuit counts have invalid shape")
            if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
                raise AssertionError("sequence circuit counts must be nonnegative integers")
            for name in ("last_used", "reuse"):
                value = circuit.get(name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise AssertionError("sequence circuit metadata is invalid")
        return True

    def state_dict(self):
        self.validate()
        ordered = sorted(self.circuits.items(), key=lambda item: (len(item[0]), repr(item[0])))
        return {
            "version": self.VERSION,
            "symbols": list(self.symbols),
            "max_order": self.max_order,
            "max_circuits": self.max_circuits,
            "min_support": self.min_support,
            "prior": self.prior,
            "history": list(self.history),
            "events_seen": self.events_seen,
            "circuits_created": self.circuits_created,
            "circuits_reclaimed": self.circuits_reclaimed,
            "chunks_promoted": self.chunks_promoted,
            "chunks": [
                {"context": list(context),
                 "merged": chunk["merged"],
                 "reuse": chunk["reuse"],
                 "concentration": chunk["concentration"]}
                for context, chunk in sorted(self.chunks.items(), key=lambda item: repr(item[0]))
            ],
            "circuits": [
                {
                    "context": list(context),
                    "counts": list(circuit["counts"]),
                    "last_used": circuit["last_used"],
                    "reuse": circuit["reuse"],
                }
                for context, circuit in ordered
            ],
        }

    @classmethod
    def from_state_dict(cls, payload):
        if not isinstance(payload, dict) or payload.get("version") != cls.VERSION:
            raise ValueError("unsupported sequence memory state")
        memory = cls(
            payload["symbols"], payload["max_order"], payload["max_circuits"],
            payload["min_support"], payload["prior"])
        memory.history = list(payload["history"])
        memory.events_seen = int(payload["events_seen"])
        memory.circuits_created = int(payload["circuits_created"])
        memory.circuits_reclaimed = int(payload["circuits_reclaimed"])
        memory.chunks_promoted = int(payload.get("chunks_promoted", 0))
        memory.chunks = {}
        for item in payload.get("chunks", []):
            context = tuple(item["context"])
            if context in memory.chunks:
                raise ValueError("duplicate sequence chunk context")
            memory.chunks[context] = {
                "merged": int(item["merged"]),
                "reuse": int(item["reuse"]),
                "concentration": float(item["concentration"]),
            }
        memory.circuits = {}
        for item in payload["circuits"]:
            context = tuple(item["context"])
            if context in memory.circuits:
                raise ValueError("duplicate sequence circuit context")
            memory.circuits[context] = {
                "counts": [int(value) for value in item["counts"]],
                "last_used": int(item["last_used"]),
                "reuse": int(item["reuse"]),
            }
        try:
            memory.validate()
        except AssertionError as error:
            raise ValueError("invalid sequence memory state: %s" % error)
        return memory


class EpisodicBuffer(object):
    """Exact-match correction store: trigger patterns map to completions.

    Episodic entries override statistical prediction only on exact suffix
    match (longest trigger wins). They never alter table counts, so a
    correction cannot distort unrelated predictions: zero collateral by
    construction. Removal is exact deletion.
    """

    VERSION = 1

    # Provenance ranks: higher supersedes lower; equal ranks keep
    # latest-wins. Unknown labels default to user level (compat).
    PROVENANCE_RANKS = {"system": 3, "correction": 2, "user": 1,
                        "approved-notes": 1, "untrusted": 0}

    def __init__(self, max_entries=1024, max_per_provenance=None):
        self.max_entries = int(max_entries)
        if self.max_entries < 1:
            raise ValueError("episodic capacity must be positive")
        if max_per_provenance is not None and (not isinstance(max_per_provenance, int)
                                               or max_per_provenance < 1):
            raise ValueError("per-provenance quota must be a positive integer")
        self.max_per_provenance = max_per_provenance
        self.entries = {}
        self.next_id = 0
        self.hits = 0
        self.misses = 0
        self.conflicts = []
        self.blocked = []

    def add(self, trigger, completion, provenance="correction"):
        """Store trigger->completion bit patterns. Returns entry id."""
        trigger = tuple(int(bit) for bit in trigger)
        completion = tuple(int(bit) for bit in completion)
        if not trigger or not completion:
            raise ValueError("episodic trigger and completion must be nonempty")
        if any(bit not in (0, 1) for bit in trigger + completion):
            raise ValueError("episodic patterns carry bits only")
        rank = self.PROVENANCE_RANKS.get(str(provenance), 1)
        for existing in self.entries.values():
            if existing["trigger"] == trigger:
                old_rank = self.PROVENANCE_RANKS.get(existing["provenance"], 1)
                if existing["completion"] != completion:
                    self.conflicts.append({
                        "trigger": list(trigger),
                        "old_completion": list(existing["completion"]),
                        "new_completion": list(completion),
                        "old_provenance": existing["provenance"],
                        "new_provenance": str(provenance),
                        "resolution": "latest-wins" if rank >= old_rank else "blocked-lower-trust",
                    })
                    del self.conflicts[:-1024]
                if rank < old_rank:
                    if existing["id"] not in self.blocked:
                        self.blocked.append(existing["id"])
                    return existing["id"]
                existing["completion"] = completion
                existing["provenance"] = str(provenance)
                existing["superseded"] = existing.get("superseded", 0) + 1
                return existing["id"]
        if self.max_per_provenance is not None:
            owned = sum(1 for entry in self.entries.values()
                        if entry["provenance"] == str(provenance))
            if owned >= self.max_per_provenance:
                raise ValueError("per-provenance episodic quota exceeded")
        if len(self.entries) >= self.max_entries:
            victim = min(self.entries.values(),
                         key=lambda entry: (entry["uses"], entry["id"]))
            del self.entries[victim["id"]]
        entry_id = self.next_id
        self.next_id += 1
        self.entries[entry_id] = {
            "id": entry_id,
            "trigger": trigger,
            "completion": completion,
            "provenance": str(provenance),
            "uses": 0,
            "superseded": 0,
        }
        return entry_id

    def match(self, history):
        """Longest trigger that suffix-matches history, else None."""
        history = tuple(history)
        best = None
        for entry in self.entries.values():
            trigger = entry["trigger"]
            if len(trigger) > len(history):
                continue
            if tuple(history[-len(trigger):]) == trigger:
                if best is None or len(trigger) > len(best["trigger"]):
                    best = entry
        if best is None:
            self.misses += 1
            return None
        best["uses"] += 1
        self.hits += 1
        return best["completion"]

    def match_all(self, history, window=None):
        """Every entry whose trigger occurs in the recent window.

        Returns completions ordered by most-recent trigger end. Substring
        (not only suffix) matching enables systematic recombination: known
        parts in novel configurations. Window bounds the search and the
        false-positive surface; default is the full history.
        """
        history = tuple(history)
        if window is not None:
            history = history[-int(window):]
        hits = []
        for entry in self.entries.values():
            trigger = entry["trigger"]
            if not trigger or len(trigger) > len(history):
                continue
            for end in range(len(trigger), len(history) + 1):
                if tuple(history[end - len(trigger):end]) == trigger:
                    hits.append((end, len(trigger), entry))
                    break
        hits.sort(key=lambda item: (-item[0], -item[1]))
        accepted = []
        for end, length, entry in hits:
            start = end - length
            if any(other_start <= start and end <= other_end
                   for other_start, other_end, _ in accepted):
                continue
            accepted.append((start, end, entry))
        for _, _, entry in accepted:
            entry["uses"] += 1
        if accepted:
            self.hits += 1
        else:
            self.misses += 1
        return [(entry["id"], entry["completion"]) for _, _, entry in accepted]

    def remove(self, entry_id):
        if entry_id not in self.entries:
            raise ValueError("unknown episodic entry")
        del self.entries[entry_id]

    def consolidate(self, memory, min_uses=3, weight=50):
        """Fold high-reuse episodic rules into statistical table counts.

        Replays trigger+completion through the table at the given weight,
        then drops the consolidated entries. After consolidation the table
        alone recalls the fact: episodic-to-semantic transfer. Returns the
        list of consolidated entry ids. Raises on invalid input; the table
        validates itself on every observe.
        """
        if not isinstance(min_uses, int) or min_uses < 1:
            raise ValueError("min_uses must be a positive integer")
        if isinstance(weight, bool) or not isinstance(weight, int) or weight < 1:
            raise ValueError("weight must be a positive integer")
        consolidated = []
        for entry_id, entry in sorted(self.entries.items()):
            if entry["uses"] < min_uses:
                continue
            sequence = list(entry["trigger"]) + list(entry["completion"])
            memory.reset_history()
            for symbol in sequence:
                memory.observe(symbol, learn=True, weight=weight)
            consolidated.append(entry_id)
        for entry_id in consolidated:
            del self.entries[entry_id]
        return consolidated

    def validate(self):
        if len(self.entries) > self.max_entries:
            raise AssertionError("episodic buffer exceeds capacity")
        if not isinstance(self.conflicts, list):
            raise AssertionError("episodic conflicts must be a list")
        if self.max_per_provenance is not None and (
                not isinstance(self.max_per_provenance, int) or self.max_per_provenance < 1):
            raise AssertionError("episodic quota is invalid")
        if not isinstance(self.blocked, list):
            raise AssertionError("episodic blocked list is invalid")
        for entry_id, entry in self.entries.items():
            if entry_id != entry["id"]:
                raise AssertionError("episodic entry id mismatch")
            if not entry["trigger"] or not entry["completion"]:
                raise AssertionError("episodic entry patterns must be nonempty")
        return True

    def state_dict(self):
        self.validate()
        return {
            "version": self.VERSION,
            "max_entries": self.max_entries,
            "next_id": self.next_id,
            "hits": self.hits,
            "misses": self.misses,
            "max_per_provenance": self.max_per_provenance,
            "blocked": list(self.blocked),
            "conflicts": [dict(item) for item in self.conflicts],
            "entries": [
                {"id": entry["id"], "trigger": list(entry["trigger"]),
                 "completion": list(entry["completion"]),
                 "provenance": entry["provenance"], "uses": entry["uses"],
                 "superseded": entry.get("superseded", 0)}
                for _, entry in sorted(self.entries.items())
            ],
        }

    @classmethod
    def from_state_dict(cls, payload):
        if not isinstance(payload, dict) or payload.get("version") != cls.VERSION:
            raise ValueError("unsupported episodic buffer state")
        buffer = cls(payload.get("max_entries", 1024))
        buffer.next_id = int(payload.get("next_id", 0))
        buffer.hits = int(payload.get("hits", 0))
        buffer.misses = int(payload.get("misses", 0))
        quota = payload.get("max_per_provenance")
        buffer.max_per_provenance = None if quota is None else int(quota)
        buffer.blocked = [int(value) for value in payload.get("blocked", [])]
        buffer.conflicts = [dict(item) for item in payload.get("conflicts", [])]
        for item in payload.get("entries", []):
            entry_id = int(item["id"])
            buffer.entries[entry_id] = {
                "id": entry_id,
                "trigger": tuple(int(bit) for bit in item["trigger"]),
                "completion": tuple(int(bit) for bit in item["completion"]),
                "provenance": str(item.get("provenance", "")),
                "uses": int(item.get("uses", 0)),
                "superseded": int(item.get("superseded", 0)),
            }
        buffer.validate()
        return buffer
