"""R5 profiler: throughput, sparse activation, state growth, memory (stdlib)."""

import os
import random
import time

try:
    import resource as _resource
    import sys as _sys

    def _rss_mb():
        rss = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        return rss / (1024.0 * 1024.0) if _sys.platform == "darwin" else rss / 1024.0
except ImportError:  # pragma: no cover
    def _rss_mb():
        return -1.0


def _state_bytes(paths):
    total = 0
    for key in ("brain", "dialogue", "episodic", "manifest"):
        path = paths.get(key)
        if path is not None and os.path.exists(path):
            total += os.path.getsize(path)
    return total


def profile_brain(store, name, steps=2000, seed=0):
    """Step the stored brain's organism and sequence memory; report budgets."""
    from ..organism import Modulators
    organism, episodic, manifest, dialogue = store.load(name)
    paths = store._paths(name)
    bytes_before = _state_bytes(paths)
    rss_before = _rss_mb()
    rng = random.Random(seed)
    active_cells, active_synapses = [], []
    start = time.time()
    width = len(organism.input_ids)
    for _ in range(steps):
        inputs = [rng.uniform(-1.0, 1.0) for _ in range(width)]
        organism.step(inputs, Modulators(novelty=0.05, exploration=0.02))
        active_cells.append(sum(1 for cell in organism.cells.values()
                                if abs(cell.activation) > 0.05))
        active_synapses.append(len(organism.graph.synapses))
    wall = time.time() - start
    memory = organism.sequence_memory
    if memory is not None:
        bit_stream = [rng.randint(0, 1) for _ in range(steps)]
        start = time.time()
        for symbol in bit_stream:
            memory.observe(symbol)
            memory.distribution()
        memory_wall = time.time() - start
    else:
        memory_wall = 0.0
    store.save(name, organism, episodic, dialogue=dialogue)
    return {
        "brain": name,
        "steps": steps,
        "organism_us_per_step": wall / max(1, steps) * 1e6,
        "memory_us_per_event": memory_wall / max(1, steps) * 1e6,
        "mean_active_cells": sum(active_cells) / max(1, len(active_cells)),
        "max_active_cells": max(active_cells) if active_cells else 0,
        "total_cells": len(organism.cells),
        "synapses": active_synapses[-1] if active_synapses else 0,
        "state_bytes_before": bytes_before,
        "state_bytes_after": _state_bytes(paths),
        "rss_before_mb": rss_before,
        "rss_after_mb": _rss_mb(),
    }
