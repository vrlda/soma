"""Public locked v10 variable-order benchmark API.

The implementation lives in :mod:`soma.lifetime` with the other lifetime
protocols so existing imports remain stable.  This module is intentionally a
thin export surface; it does not add learner logic or evaluator shortcuts.
"""

from soma.lifetime import (
    VARIABLE_ORDER_ACCEPTANCE_MANIFEST,
    VARIABLE_ORDER_CUBIC_FEATURES,
    VARIABLE_ORDER_PAIR_FEATURES,
    VARIABLE_ORDER_POLICY_SEQUENCE,
    VARIABLE_ORDER_TASK_MANIFEST,
    VariableOrderBenchmarkConfig,
    VariableOrderBenchmarkEnvironment,
    VariableOrderEnvironment,
    run_variable_order,
    run_variable_order_benchmark,
    variable_order_affine_proof,
    variable_order_feature_proof,
    variable_order_parity_proof,
    variable_order_phase_lengths_for_seed,
    variable_order_seed_manifest,
    variable_order_task_manifest,
)

__all__ = [
    "VARIABLE_ORDER_ACCEPTANCE_MANIFEST",
    "VARIABLE_ORDER_CUBIC_FEATURES",
    "VARIABLE_ORDER_PAIR_FEATURES",
    "VARIABLE_ORDER_POLICY_SEQUENCE",
    "VARIABLE_ORDER_TASK_MANIFEST",
    "VariableOrderBenchmarkConfig",
    "VariableOrderBenchmarkEnvironment",
    "VariableOrderEnvironment",
    "run_variable_order",
    "run_variable_order_benchmark",
    "variable_order_affine_proof",
    "variable_order_feature_proof",
    "variable_order_parity_proof",
    "variable_order_phase_lengths_for_seed",
    "variable_order_seed_manifest",
    "variable_order_task_manifest",
]


if __name__ == "__main__":
    import json

    print(json.dumps(run_variable_order_benchmark(VariableOrderBenchmarkConfig()), indent=2, sort_keys=True))
