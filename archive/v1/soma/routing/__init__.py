"""R1 calibrated evidence routing: pure calculations, no behavior control yet."""

from .evidence import (
    CircuitEvidence,
    calibration_metrics,
    comparative_log_likelihood,
    normalize_posterior,
    probe_decision,
    probe_tally_posterior,
    should_switch,
    student_t_logpdf,
    update_evidence,
)

__all__ = [
    "CircuitEvidence",
    "calibration_metrics",
    "comparative_log_likelihood",
    "normalize_posterior",
    "probe_decision",
    "probe_tally_posterior",
    "should_switch",
    "student_t_logpdf",
    "update_evidence",
]
