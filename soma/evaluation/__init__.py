"""R2 evaluation: locked cross-domain fixtures over the event bridge."""

from .english import (
    SHUFFLE_LAG,
    bit_bigram_bits,
    bit_marginal_bits,
    bit_stream,
    bits_per_bit,
    byte_unigram_bits,
    load_corpus,
    partition_documents,
    run_english,
    run_english_brain_memory,
    run_english_continued,
    split_chapters,
)
from .lineage import clone_brain, read_lineage, verify_clone
from .sequence import (
    R3A_ACCEPTANCE_SEEDS,
    run_sequence,
    run_sequence_benchmark,
)
from .temporal import (
    R2_ACCEPTANCE_SEEDS,
    run_event_context,
    run_event_context_benchmark,
)

__all__ = [
    "R2_ACCEPTANCE_SEEDS",
    "R3A_ACCEPTANCE_SEEDS",
    "SHUFFLE_LAG",
    "bit_bigram_bits",
    "bit_marginal_bits",
    "bit_stream",
    "bits_per_bit",
    "byte_unigram_bits",
    "clone_brain",
    "load_corpus",
    "partition_documents",
    "read_lineage",
    "run_english",
    "run_english_brain_memory",
    "run_english_continued",
    "run_event_context",
    "run_event_context_benchmark",
    "run_sequence",
    "run_sequence_benchmark",
    "split_chapters",
    "verify_clone",
]
