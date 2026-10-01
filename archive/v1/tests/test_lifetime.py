import os
import copy
import json
import tempfile
import unittest

from soma import Modulators, Organism
from soma.lifetime import (
    AdaptiveDendriticBenchmarkConfig,
    AdaptiveDendriticEnvironment,
    LifetimeBenchmarkConfig,
    MultiContextBenchmarkConfig,
    NonlinearBenchmarkConfig,
    NonlinearMultiContextEnvironment,
    UnannouncedLifetimeEnvironment,
    UnannouncedMultiContextEnvironment,
    phase_lengths_for_seed,
    phase_windows_for_lengths,
    run_benchmark,
    run_lifetime,
    run_multicontext,
    run_multicontext_benchmark,
    run_nonlinear,
    run_nonlinear_benchmark,
    adaptive_dendritic_affine_proof,
    adaptive_dendritic_phase_lengths_for_seed,
    adaptive_dendritic_task_pairs,
    run_adaptive_dendritic,
    run_adaptive_dendritic_benchmark,
    VariableOrderBenchmarkConfig,
    VariableOrderBenchmarkEnvironment,
    VARIABLE_ORDER_POLICY_SEQUENCE,
    VARIABLE_ORDER_TASK_MANIFEST,
    variable_order_parity_proof,
    variable_order_phase_lengths_for_seed,
    variable_order_seed_manifest,
    run_variable_order,
    run_variable_order_benchmark,
    CompositionalBenchmarkConfig,
    CompositionalBenchmarkEnvironment,
    COMPOSITIONAL_POLICY_SEQUENCE,
    COMPOSITIONAL_TASK_MANIFEST,
    compositional_factorization_manifest,
    compositional_phase_lengths_for_seed,
    compositional_seed_manifest,
    compositional_walsh_proof,
    run_compositional,
    run_compositional_benchmark,
    StreamingCompositionalBenchmarkConfig,
    StreamingCompositionalBenchmarkEnvironment,
    streaming_compositional_schedule,
    run_streaming_compositional,
    run_streaming_compositional_benchmark,
    streaming_routing_diagnostic,
    _streaming_summary,
    _composition_install_matches_feature,
    _evaluate_composition_causal_ablation,
    _first_canonical_reactivation_lag,
    GeneralStructuralBenchmarkConfig,
    run_general_structural,
    run_general_structural_benchmark,
    ContinuousLifetimeBenchmarkConfig,
    ContinuousLifetimeEnvironment,
    run_continuous_lifetime,
    run_continuous_lifetime_benchmark,
    nonlinear_affine_proof,
    nonlinear_phase_lengths_for_seed,
    nonlinear_representation_seed,
    summarize,
    summarize_multicontext,
)
from soma.synapses import SparseDirectedGraph, Synapse


class LifetimeBenchmarkTests(unittest.TestCase):
    def test_phase_switch_is_hidden_from_observations(self):
        environment = UnannouncedLifetimeEnvironment(seed=17, phase_lengths=(3, 3, 3))
        observation = environment.reset()
        self.assertEqual(len(observation.values), 2)
        self.assertFalse(hasattr(observation, "phase"))
        self.assertNotIn("phase", observation.__dict__)
        observations = []
        for _ in range(environment.horizon):
            observations.append(observation.values)
            observation = environment.step(0.0)
        self.assertEqual(len(observations), environment.horizon)
        self.assertNotIn("phase", environment.state_dict())

    def test_reward_is_neutral_at_zero_and_penalizes_wrong_polarity(self):
        environment = UnannouncedLifetimeEnvironment(seed=17, phase_lengths=(3, 3, 3))
        observation = environment.reset()
        cue = observation.values[0]
        zero = environment.step(0.0).reward
        self.assertAlmostEqual(zero, 0.0)
        environment.reset()
        environment.step(-cue)
        self.assertLess(environment.last_reward, 0.0)

    def test_schedule_is_seeded_but_has_independent_hidden_lengths(self):
        self.assertEqual(phase_lengths_for_seed(4), phase_lengths_for_seed(4))
        self.assertNotEqual(phase_lengths_for_seed(4), phase_lengths_for_seed(5))

    def test_lifetime_run_is_deterministic_and_one_lineage(self):
        config = LifetimeBenchmarkConfig(phase_lengths=(20, 20, 20), seeds=(2,))
        first = run_lifetime(2, config)
        second = run_lifetime(2, config)
        self.assertEqual(first, second)
        self.assertEqual(first["organism_steps"], sum(phase_lengths_for_seed(2, config.phase_lengths)))
        self.assertTrue(set(first["initial_cell_ids"]).issubset(set(first["final_cell_ids"])))
        self.assertEqual(first["final_state"]["step_count"], first["organism_steps"])

    def test_context_modules_do_not_recruit_during_stationary_a(self):
        false_recruitments = 0
        false_alarms = 0
        warning_starts = 0
        for seed in range(24):
            environment = UnannouncedLifetimeEnvironment(seed=seed, phase_lengths=(300, 1, 1))
            organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=seed)
            organism.enable_context_modules()
            organism.actor_learning_rate = 0.1
            tape_rng = __import__("random").Random((seed + 1) * 1000033 + 3571)
            organism.set_exploration_tape(tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(environment.horizon)))
            observation = environment.reset()
            for _ in range(300):
                result = organism.step(observation.values, Modulators(reward=observation.reward, exploration=0.2))
                observation = environment.step(result.outputs[0])
            false_recruitments += len([event for event in organism.events if event["kind"] == "motor_module_recruited"]) > 1
            false_alarms += any(event["kind"] == "motor_warning_escalated" for event in organism.events)
            warning_starts += sum(event["kind"] == "motor_warning_started" for event in organism.events)
        self.assertLessEqual(false_recruitments, 4)
        self.assertLessEqual(false_alarms, 4)
        self.assertLessEqual(warning_starts, 4)

    def test_hidden_reversal_has_bounded_detection_and_reactivation_latency(self):
        detection_lags = []
        reactivation_lags = []
        for seed in range(24):
            config = LifetimeBenchmarkConfig(phase_lengths=(120, 120, 240), seeds=(seed,))
            result = run_lifetime(seed, config, learning_rate=0.1, mode="context_actor_no_generic_rewiring")
            b_start = result["phase_lengths"][0]
            a2_start = sum(result["phase_lengths"][:2])
            events = result["final_state"]["events"]
            detections = [event["step"] for event in events if event["kind"] == "motor_warning_started" and b_start <= event["step"] < a2_start]
            reactivations = [event["step"] for event in events if event["kind"] == "motor_module_reactivated" and event["module_id"] == "motor-module-000" and event["step"] >= a2_start]
            if detections:
                detection_lags.append(detections[0] - b_start)
            if reactivations:
                reactivation_lags.append(reactivations[0] - a2_start)
        self.assertGreaterEqual(len(detection_lags), 20)
        self.assertLessEqual(sorted(detection_lags)[len(detection_lags) // 2], 45)
        # Reactivation is censored when reward evidence never beats the
        # incumbent.  The canonical benchmark reports those runs explicitly;
        # this test checks latency only for observed reactivations.
        if reactivation_lags:
            self.assertLessEqual(sorted(reactivation_lags)[len(reactivation_lags) // 2], 45)

    def test_warning_is_deterministic_on_matched_tapes(self):
        def prepare(seed):
            organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=seed)
            organism.enable_context_modules(max_modules=2)
            active = organism.motor_modules[organism.active_motor_module]
            active.detector_count = organism.context_detector_min_evidence
            active.confidence = 1.0
            active.detector_cusum = organism.context_warning_threshold + 0.1
            organism.set_exploration_tape((0.25, -0.75, 0.5))
            organism._context_maybe_switch()
            return organism

        first = prepare(46)
        second = prepare(46)
        self.assertEqual(first.context_state, "warning")
        self.assertEqual(first.state_dict(), second.state_dict())
        first._propagate((0.8, -0.3))
        second._propagate((0.8, -0.3))
        self.assertEqual(first._context_output(Modulators(exploration=0.2)), second._context_output(Modulators(exploration=0.2)))
        self.assertEqual(first.state_dict(), second.state_dict())

    def test_warning_starts_protected_diagnostic_before_full_alarm(self):
        organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=41)
        organism.enable_context_modules(max_modules=3)
        active = organism.motor_modules[organism.active_motor_module]
        candidate = organism._recruit_motor_module()
        candidate.dormant = True
        active.detector_count = organism.context_detector_min_evidence
        active.confidence = 1.0
        active.detector_cusum = organism.context_warning_threshold + 0.1
        organism.cells[organism.input_ids[0]].activation = 0.8
        for synapse in organism._module_synapses(candidate):
            synapse.strength = 0.6
        organism.set_exploration_tape((0.9, 0.9, 0.9))

        organism._context_maybe_switch()

        self.assertEqual(organism.context_state, "warning")
        self.assertEqual(organism.context_probe_module, candidate.id)
        self.assertEqual(organism.context_probe_origin, "warning")
        self.assertLess(active.detector_cusum, organism.context_switch_threshold)
        action = organism._context_output(Modulators(exploration=0.8))[0]
        expected = organism._module_mean_from_current_inputs(candidate)
        self.assertAlmostEqual(action, expected)
        self.assertEqual(organism.exploration_cursor, 1)
        self.assertEqual(organism.context_pending_mode, "search")

    def test_warning_and_search_consume_tape_without_random_motor_noise(self):
        organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=42)
        organism.enable_context_modules()
        active = organism.motor_modules[organism.active_motor_module]
        active.detector_cusum = organism.context_warning_threshold + 0.1
        organism.cells[organism.input_ids[0]].activation = 0.7
        for synapse in organism._module_synapses(active):
            synapse.strength = 0.5
        organism.cells[active.cell_id].activation = 0.5
        organism.set_exploration_tape((1.0, -1.0))

        organism.context_state = "warning"
        warning_action = organism._context_output(Modulators(exploration=0.6))[0]
        organism.context_state = "search"
        search_action = organism._context_output(Modulators(exploration=0.6))[0]

        self.assertEqual(organism.exploration_cursor, 2)
        mean = organism.cells[active.cell_id].activation
        span = organism.context_switch_threshold - organism.context_warning_threshold
        attenuation = 0.5 + 0.5 * (1.0 - (active.detector_cusum - organism.context_warning_threshold) / span)
        self.assertAlmostEqual(warning_action, attenuation * mean)
        self.assertAlmostEqual(search_action, mean)

    def test_deterministic_probe_uses_base_information_threshold(self):
        organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=49)
        organism.enable_context_modules(max_modules=2)
        incumbent = organism.motor_modules[organism.active_motor_module]
        candidate = organism._recruit_motor_module()
        candidate.dormant = True
        organism.context_probe_module = candidate.id
        organism.context_probe_incumbent_module = incumbent.id
        organism.context_probe_remaining = 1
        organism.context_probe_candidate_mean = 0.10
        organism.context_probe_incumbent_mean = 0.00
        # Caller requests exploration, but _context_output suppresses its
        # motor noise during probes.  A .10 disagreement is informative above
        # the base .05 threshold and below the old 2*.20 threshold.
        organism.context_probe_exploration = 0.20
        organism.pending_motor_module = candidate.id
        organism.context_pending_mode = "search"

        organism._apply_context_reward(0.0)

        self.assertEqual(organism.context_probe_steps_seen, 1)
        self.assertEqual(organism.context_probe_informative_count, 1)
        self.assertEqual(organism.context_probe_count, 1)
        self.assertEqual(organism.context_probe_rewards, [0.0])

    def test_detector_accumulates_moderate_residuals_but_not_stationary_surprise(self):
        organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=50)
        organism.enable_context_modules(max_modules=2)
        module = organism.motor_modules[organism.active_motor_module]
        module.detector_frozen = True
        module.detector_count = organism.context_detector_min_evidence
        features = (1.0,) + (0.0,) * (organism._context_feature_dimension() - 1)

        def apply_residual(residual):
            organism.pending_motor_module = module.id
            organism.context_pending_mode = "normal"
            organism.context_pending_features = features
            organism.context_pending_prediction = 0.0
            organism.context_pending_scale = 1.0
            organism._apply_context_reward(residual)

        for _ in range(20):
            # -0.40 is informative with the new .35 tolerance, but was below
            # the old .50 tolerance and therefore contributed no CUSUM evidence.
            apply_residual(-0.40)
        self.assertGreater(module.detector_cusum, 0.5)

        module.detector_cusum = 0.0
        for _ in range(20):
            apply_residual(0.0)
        self.assertEqual(module.detector_cusum, 0.0)

    def test_warning_rejected_probe_returns_to_incumbent_without_recruitment(self):
        organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=43)
        organism.enable_context_modules(max_modules=3)
        active = organism.motor_modules[organism.active_motor_module]
        candidate = organism._recruit_motor_module()
        candidate.dormant = True
        active.detector_count = organism.context_detector_min_evidence
        active.confidence = 1.0
        active.detector_cusum = organism.context_warning_threshold + 0.1
        organism._context_maybe_switch()
        self.assertEqual(organism.context_probe_origin, "warning")
        organism.context_probe_remaining = 0
        organism.context_probe_steps_seen = organism.context_probe_min_steps
        organism.context_probe_count = organism.context_probe_min_steps
        organism.context_probe_informative_count = organism.context_probe_min_steps
        organism.context_probe_rewards = [-0.4, -0.3]
        organism.context_probe_sum = -0.7
        organism.context_probe_sumsq = 0.25
        before = len(organism.motor_modules)

        organism._context_maybe_switch()

        self.assertEqual(len(organism.motor_modules), before)
        self.assertEqual(organism.active_motor_module, active.id)
        self.assertEqual(organism.context_state, "warning")
        self.assertTrue(candidate.dormant)
        self.assertIsNone(organism.context_probe_origin)

    def test_warning_probe_checkpoint_preserves_origin_and_pending_action(self):
        organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=44)
        organism.enable_context_modules(max_modules=3)
        active = organism.motor_modules[organism.active_motor_module]
        candidate = organism._recruit_motor_module()
        candidate.dormant = True
        active.detector_count = organism.context_detector_min_evidence
        active.confidence = 1.0
        active.detector_cusum = organism.context_warning_threshold + 0.1
        organism.cells[organism.input_ids[0]].activation = 0.9
        organism.set_exploration_tape((0.2, -0.8, 0.4))
        organism._context_maybe_switch()
        organism._context_output(Modulators(exploration=0.9))

        resumed = Organism.from_state_dict(organism.state_dict())

        self.assertEqual(resumed.state_dict(), organism.state_dict())
        self.assertEqual(resumed.context_probe_origin, "warning")
        self.assertEqual(resumed.context_probe_module, candidate.id)
        self.assertEqual(resumed.exploration_cursor, organism.exploration_cursor)

    def test_stationary_warning_recovers_without_switching(self):
        organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=45)
        organism.enable_context_modules()
        active = organism.motor_modules[organism.active_motor_module]
        active.detector_count = organism.context_detector_min_evidence
        active.confidence = 1.0
        active.detector_cusum = organism.context_warning_threshold + 0.1
        organism._context_maybe_switch()
        self.assertEqual(organism.context_state, "warning")
        active.detector_cusum = organism.context_warning_threshold - 0.1

        organism._context_maybe_switch()

        self.assertEqual(organism.context_state, "normal")
        self.assertEqual(len(organism.motor_modules), 1)
        self.assertTrue(any(event["kind"] == "motor_warning_resolved" for event in organism.events))

    def test_checkpoint_mid_detector_wait_is_exact(self):
        environment = UnannouncedLifetimeEnvironment(seed=9, phase_lengths=(180, 180, 180))
        organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=9)
        organism.enable_context_modules()
        organism.actor_learning_rate = 0.1
        tape_rng = __import__("random").Random(9009)
        organism.set_exploration_tape(tuple(tape_rng.uniform(-1.0, 1.0) for _ in range(environment.horizon)))
        observation = environment.reset()
        for _ in range(20):
            result = organism.step(observation.values, Modulators(reward=observation.reward, exploration=0.2))
            observation = environment.step(result.outputs[0])
        resumed = Organism.from_state_dict(organism.state_dict())
        for _ in range(40):
            expected = organism.step(observation.values, Modulators(reward=observation.reward, exploration=0.2))
            actual = resumed.step(observation.values, Modulators(reward=observation.reward, exploration=0.2))
            self.assertEqual(expected.outputs, actual.outputs)
            self.assertEqual(organism.state_dict(), resumed.state_dict())
            observation = environment.step(expected.outputs[0])

    def test_reversal_reuses_original_module_after_probe(self):
        config = LifetimeBenchmarkConfig(phase_lengths=(180, 180, 180), seeds=(0,))
        result = run_lifetime(0, config, learning_rate=0.1)
        events = result["final_state"]["events"]
        self.assertTrue(any(event["kind"] == "motor_module_probe_started" for event in events))
        self.assertTrue(any(event["kind"] == "motor_module_reactivated" and event["module_id"] == "motor-module-000" for event in events))
        self.assertLessEqual(len(result["final_state"]["motor_modules"]), 4)

    def test_accepted_reactivation_clears_alarm_episode_preserving_learned_state(self):
        organism = Organism.create_default(input_size=2, hidden_size=3, output_size=1, seed=44)
        organism.enable_context_modules(max_modules=2)
        incumbent = organism.motor_modules[organism.active_motor_module]
        candidate = organism._recruit_motor_module()
        candidate.dormant = True
        candidate.detector_cusum = 4.75
        candidate.detector_frozen = True
        candidate.surprise_evidence = 4.75
        candidate.negative_surprise = 4.75
        candidate.negative_streak = 6
        candidate.confidence = 0.62
        candidate.baseline = 0.17
        candidate.detector_predictor = [0.1] * len(candidate.detector_predictor)
        candidate.detector_variance = 0.37
        candidate.detector_count = 13
        predictor = list(candidate.detector_predictor)
        policy = tuple(synapse.strength for synapse in organism._module_synapses(candidate))
        organism.context_probe_module = candidate.id
        organism.context_probe_incumbent_module = incumbent.id
        organism.context_probe_origin = "warning"
        organism.context_probe_remaining = 0
        organism.context_probe_steps_seen = 2
        organism.context_probe_informative_count = 2
        organism.context_probe_count = 2
        organism.context_probe_rewards = [0.5, 0.5]
        organism.context_probe_sum = 1.0
        organism.context_probe_sumsq = 0.5

        organism._context_maybe_switch()

        self.assertFalse(candidate.dormant)
        self.assertEqual(organism.active_motor_module, candidate.id)
        self.assertEqual(candidate.detector_cusum, 0.0)
        self.assertFalse(candidate.detector_frozen)
        self.assertEqual(candidate.surprise_evidence, 0.0)
        self.assertEqual(candidate.negative_surprise, 0.0)
        self.assertEqual(candidate.negative_streak, 0)
        self.assertEqual(candidate.baseline, 0.17)
        self.assertEqual(candidate.confidence, 0.0)
        self.assertEqual(candidate.detector_predictor, predictor)
        self.assertEqual(candidate.detector_variance, 0.37)
        self.assertEqual(candidate.detector_count, 13)
        self.assertEqual(tuple(synapse.strength for synapse in organism._module_synapses(candidate)), policy)

    def test_first_canonical_lag_ignores_later_reactivation_churn(self):
        events = [
            {"kind": "motor_module_reactivated", "module_id": "motor-module-001", "step": 105},
            {"kind": "motor_module_reactivated", "module_id": "motor-module-001", "step": 145},
            {"kind": "motor_module_reactivated", "module_id": "motor-module-002", "step": 110},
        ]
        self.assertEqual(_first_canonical_reactivation_lag(events, 100, 160, "motor-module-001"), (5,))
        self.assertEqual(_first_canonical_reactivation_lag(events, 100, 160, "motor-module-003"), ())

    def test_context_resources_and_module_weights_are_checkpointed(self):
        config = LifetimeBenchmarkConfig(phase_lengths=(40, 40, 40), seeds=(1,))
        first = run_lifetime(1, config, learning_rate=0.1)
        second = run_lifetime(1, config, learning_rate=0.1)
        self.assertEqual(first["final_state"], second["final_state"])
        resources = first["final_state"]["resources"]
        self.assertEqual(resources["counters"]["cells"], len(first["final_state"]["cells"]))
        self.assertEqual(resources["counters"]["synapses"], len(first["final_state"]["synapses"]))
        self.assertLessEqual(len(first["final_state"]["motor_modules"]), 4)

    def test_context_recruitment_cap_failure_is_atomic(self):
        organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=12, max_cells=9)
        organism.context_enabled = True
        organism.context_max_modules = 4
        before = organism.state_dict()
        with self.assertRaises(RuntimeError):
            organism._recruit_motor_module()
        self.assertEqual(before, organism.state_dict())

    def test_context_detector_parameters_roundtrip(self):
        organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=12)
        organism.enable_context_modules()
        organism.context_switch_threshold = 3.1
        organism.context_warning_threshold = 1.4
        organism.context_safe_action = 0.12
        organism.context_confidence_threshold = 0.77
        organism.context_negative_streak = 5
        organism.context_surprise_leak = 0.987
        organism.context_surprise_drift = 0.031
        organism.context_surprise_mean = -0.12
        organism.context_surprise_variance = 0.23
        organism.context_surprise_count = 17
        organism.context_detector_drift = 0.41
        organism.context_detector_recovery = 0.19
        organism.context_detector_predictor_rate = 0.33
        organism.context_detector_min_evidence = 29
        organism.context_detector_freeze_threshold = 2.6
        organism.context_probe_steps = 11
        organism.context_probe_min_steps = 7
        organism.context_probe_max_steps = 19
        organism.context_reactivation_margin = 0.014
        organism.context_probe_reward_floor = 0.021
        organism.context_probe_reference = 0.18
        organism.context_probe_failed_reference = 0.07
        organism.context_last_reward_value = -0.04
        restored = Organism.from_state_dict(organism.state_dict())
        self.assertEqual(organism.state_dict(), restored.state_dict())

    def test_cap_fallback_never_leaves_active_module_dormant(self):
        organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=13)
        organism.enable_context_modules(max_modules=1)
        module = organism.motor_modules[organism.active_motor_module]
        module.confidence = 1.0
        module.negative_surprise = 1.0
        module.negative_streak = 10
        organism._context_maybe_switch()
        self.assertFalse(module.dormant)
        organism.validate()

    def test_isolated_bad_outcome_warning_resolves_without_search(self):
        organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=31)
        organism.enable_context_modules(max_modules=2)
        module = organism.motor_modules[organism.active_motor_module]
        module.detector_count = organism.context_detector_min_evidence
        module.confidence = 1.0
        module.detector_cusum = organism.context_warning_threshold + 0.1
        module.detector_frozen = False
        before_weights = tuple(s.strength for s in organism._module_synapses(module))
        before_predictor = tuple(module.detector_predictor)
        organism._context_maybe_switch()
        self.assertEqual(organism.context_state, "warning")
        self.assertTrue(module.detector_frozen)
        warning_checkpoint = Organism.from_state_dict(organism.state_dict())
        organism._propagate((0.8, -0.3))
        warning_checkpoint._propagate((0.8, -0.3))
        result = organism._context_output(Modulators(exploration=0.2))
        resumed_result = warning_checkpoint._context_output(Modulators(exploration=0.2))
        self.assertEqual(result.outputs if hasattr(result, "outputs") else result, resumed_result.outputs if hasattr(resumed_result, "outputs") else resumed_result)
        self.assertEqual(organism.state_dict(), warning_checkpoint.state_dict())
        self.assertLess(abs(result[0]), 1.0)
        self.assertEqual(tuple(s.strength for s in organism._module_synapses(module)), before_weights)
        self.assertEqual(tuple(module.detector_predictor), before_predictor)
        module.detector_cusum = organism.context_warning_threshold - 1.0
        organism._context_maybe_switch()
        self.assertEqual(organism.context_state, "normal")
        self.assertEqual(len(organism.motor_modules), 1)
        self.assertTrue(any(event["kind"] == "motor_warning_resolved" for event in organism.events))

    def test_generic_structure_edits_exclude_motor_module_cells(self):
        organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=14)
        organism.enable_context_modules()
        organism.step_count = organism.structural_interval
        for cell in organism.cells.values():
            cell.activation = 1.0
        changes = organism._structural_adaptation(Modulators(novelty=1.0))
        self.assertTrue(all("motor-module" not in str(change.get("destination", "")) and "motor-module" not in str(change.get("source", "")) for change in changes))

    def test_dormant_module_weights_remain_frozen_until_probe(self):
        organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=47)
        organism.enable_context_modules(max_modules=3)
        candidate = organism._recruit_motor_module()
        candidate.dormant = True
        candidate_cell = organism.cells[candidate.cell_id]
        candidate_cell.activation = 0.31
        candidate_cell.adaptation = 0.22
        captured = tuple(s.strength for s in organism._module_synapses(candidate))
        captured_predictor = tuple(candidate.detector_predictor)
        captured_cell_state = (candidate_cell.activation, candidate_cell.adaptation, candidate_cell.threshold)
        for values in ((0.8, -0.3), (-0.4, 0.7), (0.2, 0.1)):
            organism._propagate(values)
            organism._homeostasis()
            organism._context_output(Modulators(exploration=0.2))
        self.assertEqual(captured, tuple(s.strength for s in organism._module_synapses(candidate)))
        self.assertEqual(captured_predictor, tuple(candidate.detector_predictor))
        self.assertEqual(captured_cell_state, (candidate_cell.activation, candidate_cell.adaptation, candidate_cell.threshold))

    def test_checkpoint_inside_probe_preserves_context_search_exactly(self):
        environment = UnannouncedLifetimeEnvironment(seed=0, phase_lengths=(180, 180, 180))
        organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=48)
        organism.enable_context_modules(max_modules=3)
        organism.actor_learning_rate = 0.1
        active = organism.motor_modules[organism.active_motor_module]
        candidate = organism._recruit_motor_module()
        candidate.dormant = True
        active.detector_count = organism.context_detector_min_evidence
        active.confidence = 1.0
        active.detector_cusum = organism.context_warning_threshold + 0.1
        organism.cells[organism.input_ids[0]].activation = 0.8
        organism.cells[organism.input_ids[1]].activation = -0.3
        organism.set_exploration_tape(tuple(0.25 for _ in range(environment.horizon)))
        organism._context_maybe_switch()
        observation = environment.reset()
        organism._context_output(Modulators(exploration=0.2))
        organism._pending_outcome = True
        self.assertIsNotNone(organism.context_probe_module)
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            path = handle.name
        try:
            organism.save_checkpoint(path, environment)
            expected = []
            for _ in range(10):
                result = organism.step(observation.values, Modulators(reward=observation.reward, exploration=0.2))
                observation = environment.step(result.outputs[0])
                expected.append((result.outputs, observation.values, observation.reward))
            resumed, resumed_environment = Organism.load_checkpoint(path)
            resumed_observation = ObservationAtCheckpoint(resumed_environment)
            actual = []
            for _ in range(10):
                result = resumed.step(resumed_observation.values, Modulators(reward=resumed_observation.reward, exploration=0.2))
                resumed_observation = ObservationAtCheckpoint(resumed_environment, result.outputs[0])
                actual.append((result.outputs, resumed_observation.values, resumed_observation.reward))
            self.assertEqual(expected, actual)
            self.assertEqual(organism.step_count, resumed.step_count)
            self.assertEqual(organism.state_dict()["motor_modules"], resumed.state_dict()["motor_modules"])
            self.assertEqual(organism.state_dict()["context_probe_module"], resumed.state_dict()["context_probe_module"])
            self.assertEqual(organism.resources.to_dict(), resumed.resources.to_dict())
        finally:
            os.unlink(path)

    def test_checkpoint_resume_at_hidden_phase_boundary(self):
        config = LifetimeBenchmarkConfig(phase_lengths=(20, 20, 20), seeds=(3,))
        lengths = phase_lengths_for_seed(3, config.phase_lengths)
        environment = UnannouncedLifetimeEnvironment(seed=3, phase_lengths=lengths)
        organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=3)
        organism.set_exploration_tape(tuple(0.25 for _ in range(environment.horizon)))
        observation = environment.reset()
        for _ in range(lengths[0]):
            result = organism.step(observation.values, Modulators(reward=observation.reward, novelty=config.novelty, exploration=config.exploration))
            observation = environment.step(result.outputs[0])
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            path = handle.name
        try:
            organism.save_checkpoint(path, environment)
            expected = []
            for _ in range(lengths[1]):
                result = organism.step(observation.values, Modulators(reward=observation.reward, novelty=config.novelty, exploration=config.exploration))
                observation = environment.step(result.outputs[0])
                expected.append((result.outputs, observation.values, observation.reward))
            resumed, resumed_environment = Organism.load_checkpoint(path)
            resumed_observation = ObservationAtCheckpoint(resumed_environment)
            actual = []
            for _ in range(lengths[1]):
                result = resumed.step(resumed_observation.values, Modulators(reward=resumed_observation.reward, novelty=config.novelty, exploration=config.exploration))
                resumed_observation = ObservationAtCheckpoint(resumed_environment, result.outputs[0])
                actual.append((result.outputs, resumed_observation.values, resumed_observation.reward))
            self.assertEqual(expected, actual)
            self.assertEqual(organism.step_count, resumed.step_count)
            self.assertEqual(organism.graph.to_dict(), resumed.graph.to_dict())
            self.assertEqual(organism.resources.to_dict(), resumed.resources.to_dict())
            self.assertEqual(len(organism.metrics_history), len(resumed.metrics_history))
            for expected_metric, actual_metric in zip(organism.metrics_history, resumed.metrics_history):
                for key in ("step", "cells", "synapses", "structural_events"):
                    self.assertEqual(expected_metric[key], actual_metric[key])
                for key in ("reward", "prediction_error", "mean_abs_activity", "energy_used"):
                    self.assertAlmostEqual(expected_metric[key], actual_metric[key], places=14)
        finally:
            os.unlink(path)

    def test_metric_windows_and_thresholds_are_evaluator_correct(self):
        config = LifetimeBenchmarkConfig()
        results = [
            {"initial_acquisition": 0.02, "b_adaptation": 0.03, "return_immediate_retention": 0.5, "return_relearning_steps": 10},
            {"initial_acquisition": 0.04, "b_adaptation": 0.04, "return_immediate_retention": 0.6, "return_relearning_steps": 20},
        ]
        summary = summarize(results, config)
        self.assertEqual(summary["seeds"], 2)
        self.assertAlmostEqual(summary["mean_initial_acquisition"], 0.03)
        self.assertTrue(summary["all_passed"])
        self.assertEqual(summary["pass_counts"], {"initial_acquisition": 2, "b_adaptation": 2, "return_retention": 2, "return_relearning": 2})
        self.assertEqual(summary["relearning_censored_count"], 0)
        self.assertEqual(phase_windows_for_lengths((20, 21, 22))["A_return"], (41, 63))

        metric = {name: {key: 0.0 for key in ("skill", "sign_accuracy", "early_skill", "tail_skill", "early_sign_accuracy", "tail_sign_accuracy")} for name in ("A_initial", "B_interference", "A_return")}
        metric["B_interference"]["tail_skill"] = -0.1
        negative_tail = [dict(item, phase_metrics=metric) for item in results]
        self.assertFalse(summarize(negative_tail, config)["passed"]["b_adaptation"])

    def test_benchmark_reports_failure_without_weakening_gates(self):
        config = LifetimeBenchmarkConfig(phase_lengths=(20, 20, 20), seeds=(0, 1))
        result = run_benchmark(config)
        self.assertIn(result["diagnostic"], ("PASS", "RETENTION_OR_ADAPTATION_NOT_ESTABLISHED"))
        self.assertGreaterEqual(result["learner"]["thresholds"]["return_retention_floor"], 0.20)
        self.assertEqual(result["protocol"]["hidden_phase_boundaries"], True)

    def test_multicontext_environment_hides_policy_and_randomizes_schedule(self):
        first = UnannouncedMultiContextEnvironment(seed=17, schedule_seed=701, phase_lengths=(40, 40, 40, 40, 40))
        second = UnannouncedMultiContextEnvironment(seed=17, schedule_seed=702)
        observation = first.reset()
        self.assertEqual(len(observation.values), 2)
        self.assertFalse(hasattr(observation, "phase"))
        self.assertNotIn("context", observation.__dict__)
        self.assertNotIn("phase", first.state_dict())
        self.assertNotEqual(first.phase_lengths, second.phase_lengths)
        self.assertEqual(first.policy_sequence, ("A", "B", "C", "A", "B"))

    def test_multicontext_policy_identity_follows_heldout_order(self):
        config = MultiContextBenchmarkConfig(phase_lengths=(50, 50, 50, 50, 50), seeds=(4,))
        result = run_multicontext(4, config, policy_sequence=("A", "C", "B", "C", "A"))
        self.assertEqual(result["policy_by_segment"], {"A_initial": "A", "B_interference": "C", "C_novel": "B", "A_return": "C", "B_return": "A"})
        self.assertEqual(result["first_occurrence_segment"], {"A": "A_initial", "C": "B_interference", "B": "C_novel"})
        self.assertEqual(result["return_policy_by_segment"], {"A_return": "C", "B_return": "A"})
        self.assertEqual(set(result["canonical_reuse"]), {"A_return", "B_return"})

    def test_heldout_policy_order_summary_uses_semantic_returns(self):
        config = MultiContextBenchmarkConfig(seeds=tuple(range(24)))
        result = run_multicontext_benchmark(config)
        for order, summary in result["heldout_policy_orders"].items():
            self.assertEqual(summary["return_policy_by_segment"], {"A_return": order[3], "B_return": order[4]})
            self.assertTrue(summary["passed"]["canonical_reuse"])
            self.assertTrue(summary["all_passed"])

    def test_multicontext_all_passed_matches_declared_summary_gates(self):
        names = ("A_initial", "B_interference", "C_novel", "A_return", "B_return")
        metrics = ("skill", "sign_accuracy", "early_skill", "tail_skill", "early_sign_accuracy", "tail_sign_accuracy")

        def result(seed, tail_skill):
            return {
                "seed": seed,
                "phase_metrics": {
                    name: {metric: (tail_skill if metric in ("skill", "tail_skill") else 1.0) for metric in metrics}
                    for name in names
                },
                "module_count": 3,
                "canonical_reuse": {"A_return": True, "B_return": True},
                "canonical_first_context": {"A": "motor-module-000", "B": "motor-module-001", "C": "motor-module-002"},
                "reactivation_lag_by_segment": {"A_return": (5,), "B_return": (5,)},
                "policy_by_segment": {"A_return": "A", "B_return": "B"},
            }

        results = [result(seed, 1.0) for seed in range(24)]
        results[0] = result(0, -1.0)
        summary = summarize_multicontext(results, MultiContextBenchmarkConfig())

        self.assertEqual(summary["positive_tail_counts"]["A_initial"], 23)
        self.assertTrue(all(summary["passed"].values()))
        self.assertTrue(summary["all_passed"])
        self.assertEqual(summary["all_passed"], all(summary["passed"].values()))

    def test_multicontext_benchmark_controls_and_module_reuse(self):
        config = MultiContextBenchmarkConfig(seeds=tuple(range(24)))
        result = run_multicontext_benchmark(config)
        actor = result["context_actor"]
        self.assertTrue(result["comparison"]["matched_seed_and_protocol"])
        self.assertTrue(result["comparison"]["matched_starting_structure"])
        self.assertGreaterEqual(actor["module_reuse_counts"]["A_return_original"], 8)
        self.assertGreaterEqual(actor["module_reuse_counts"]["B_return_original"], 8)
        self.assertEqual(actor["false_extra_recruitment_runs"], 0)
        self.assertLessEqual(actor["mean_module_count"], 3.0)
        self.assertEqual(result["no_context_single_engram"]["mean_module_count"], 1.0)
        self.assertTrue(float(result["comparison"]["a_return_tail_skill_delta_vs_no_actor"]) == float(result["comparison"]["a_return_tail_skill_delta_vs_no_actor"]))

    def test_multicontext_run_is_deterministic_and_preserves_lineage(self):
        config = MultiContextBenchmarkConfig(phase_lengths=(50, 50, 50, 50, 50), seeds=(3,))
        first = run_multicontext(3, config)
        second = run_multicontext(3, config)
        self.assertEqual(first, second)
        self.assertEqual(first["organism_steps"], sum(first["phase_lengths"]))
        self.assertTrue(set(first["initial_cell_ids"]).issubset(set(first["final_cell_ids"])))
        self.assertTrue(first["module_count"] <= 4)

    def test_multicontext_stationary_single_engram_companion_does_not_grow(self):
        config = MultiContextBenchmarkConfig(phase_lengths=(60, 60, 60, 60, 60), seeds=tuple(range(24)))
        runs = [run_multicontext(seed, config, mode="no_context_single_engram", policy_sequence=("A", "A", "A", "A", "A")) for seed in config.seeds]
        extra_recruitments = [sum(1 for event in run["final_state"]["events"] if event.get("kind") == "motor_module_recruited") - 1 for run in runs]
        warning_counts = [sum(1 for event in run["final_state"]["events"] if event.get("kind") == "motor_warning_started") for run in runs]
        self.assertEqual(sum(value > 0 for value in extra_recruitments), 0)
        self.assertLessEqual(sum(warning_counts), 4)
        self.assertTrue(all(run["module_count"] == 1 for run in runs))

    def test_multicontext_returned_warning_probe_checkpoint_is_exact(self):
        organism = Organism.create_default(input_size=2, hidden_size=6, output_size=1, seed=77)
        organism.enable_context_modules(max_modules=3)
        active = organism.motor_modules[organism.active_motor_module]
        candidate = organism._recruit_motor_module()
        candidate.dormant = True
        active.detector_count = organism.context_detector_min_evidence
        active.confidence = 1.0
        active.detector_cusum = organism.context_warning_threshold + 0.1
        organism.cells[organism.input_ids[0]].activation = 0.7
        organism.cells[organism.input_ids[1]].activation = -0.4
        organism.set_exploration_tape((0.8, -0.8, 0.2))
        organism._context_maybe_switch()
        organism._context_output(Modulators(exploration=0.9))
        organism._pending_outcome = True
        environment = UnannouncedMultiContextEnvironment(seed=91, phase_lengths=(40, 40, 40, 40, 40), schedule_seed=191)
        temporary = tempfile.NamedTemporaryFile(delete=False)
        path = temporary.name
        temporary.close()
        try:
            organism.save_checkpoint(path, environment)
            resumed, resumed_environment = Organism.load_checkpoint(path)
            self.assertEqual(resumed.state_dict(), organism.state_dict())
            self.assertEqual(resumed_environment.state_dict(), environment.state_dict())
            self.assertEqual(resumed.context_probe_origin, "warning")
            self.assertEqual(resumed.context_probe_module, candidate.id)
            self.assertEqual(resumed.exploration_cursor, 1)
            self.assertTrue(resumed._pending_outcome)
        finally:
            os.unlink(path)


class NonlinearRepresentationBenchmarkTests(unittest.TestCase):
    def test_nonlinear_observation_has_no_phase_or_target(self):
        environment = NonlinearMultiContextEnvironment(seed=17, phase_lengths=(8, 8, 8, 8, 8), schedule_seed=701)
        observation = environment.reset()
        self.assertEqual(len(observation.values), 3)
        self.assertFalse(hasattr(observation, "phase"))
        self.assertNotIn("phase", observation.__dict__)
        self.assertNotIn("target", observation.__dict__)
        self.assertNotIn("target", environment.state_dict())

    def test_nonlinear_states_are_exactly_balanced_per_block_and_phase(self):
        environment = NonlinearMultiContextEnvironment(seed=17, phase_lengths=(16, 24, 16, 24, 16), schedule_seed=701)
        environment.reset()
        expected = {tuple(float(1 if (mask >> index) & 1 else -1) for index in range(3)) for mask in range(8)}
        for start, end in zip((0, 16, 40, 56, 80), (16, 40, 56, 80, 96)):
            self.assertEqual((end - start) % 8, 0)
            self.assertEqual(set(environment.inputs[start:start + 8]), expected)
            self.assertEqual(set(environment.inputs[end - 8:end]), expected)
            for block_start in range(start, end, 8):
                self.assertEqual(set(environment.inputs[block_start:block_start + 8]), expected)

    def test_nonlinear_environment_checkpoint_and_seed_are_deterministic(self):
        first = NonlinearMultiContextEnvironment(seed=4, phase_lengths=nonlinear_phase_lengths_for_seed(4, (16, 16, 16, 16, 16)), schedule_seed=7103)
        second = NonlinearMultiContextEnvironment(seed=4, phase_lengths=first.phase_lengths, schedule_seed=7103)
        self.assertEqual(first.state_dict(), second.state_dict())
        first.reset()
        for _ in range(11):
            first.step(0.17)
        resumed = NonlinearMultiContextEnvironment.from_state_dict(first.state_dict())
        self.assertEqual(first.state_dict(), resumed.state_dict())
        self.assertEqual(first.target_at(), resumed.target_at())
        self.assertEqual(first.step(0.0), resumed.step(0.0))

    def test_nonlinear_input_stream_is_independent_of_schedule_stream(self):
        lengths = (16, 16, 16, 16, 16)
        first = NonlinearMultiContextEnvironment(seed=4, phase_lengths=lengths, schedule_seed=7103)
        second = NonlinearMultiContextEnvironment(seed=4, phase_lengths=lengths, schedule_seed=7109)
        different_input = NonlinearMultiContextEnvironment(seed=5, phase_lengths=lengths, schedule_seed=7103)
        self.assertEqual(first.inputs, second.inputs)
        self.assertNotEqual(first.inputs, different_input.inputs)
        states = {tuple(float(1 if (mask >> index) & 1 else -1) for index in range(3)) for mask in range(8)}
        for environment in (first, second, different_input):
            for start in range(0, environment.horizon, 8):
                self.assertEqual(set(environment.inputs[start:start + 8]), states)

    def test_nonlinear_tail_fraction_is_validated(self):
        with self.assertRaises(ValueError):
            NonlinearBenchmarkConfig(tail_fraction=0.0).validate()
        with self.assertRaises(ValueError):
            NonlinearBenchmarkConfig(tail_fraction=1.1).validate()

    def test_raw_affine_policy_is_orthogonal_to_all_nonlinear_targets(self):
        proof = nonlinear_affine_proof()
        self.assertEqual(proof["max_abs_projection"], 0.0)
        self.assertEqual(proof["optimal_affine_skill"], {"A": 0.0, "B": 0.0, "C": 0.0})
        self.assertEqual(proof["coefficients"], {"A": (0.0, 0.0, 0.0, 0.0), "B": (0.0, 0.0, 0.0, 0.0), "C": (0.0, 0.0, 0.0, 0.0)})

    def test_red_benchmark_never_reports_representation_success(self):
        config = NonlinearBenchmarkConfig(phase_lengths=(16, 16, 16, 16, 16), seeds=(0, 1))
        result = run_nonlinear_benchmark(config)
        self.assertEqual(result["diagnostic"], "NONLINEAR_REPRESENTATION_NOT_ESTABLISHED")
        self.assertFalse(result["raw_linear_modules"]["all_passed"])
        self.assertFalse(result["no_actor"]["all_passed"])
        self.assertFalse(result["frozen"]["all_passed"])
        self.assertEqual(result["protocol"]["linear_proof"]["optimal_affine_skill"], {"A": 0.0, "B": 0.0, "C": 0.0})

    def test_hidden_representation_modes_have_hidden_only_motor_edges(self):
        config = NonlinearBenchmarkConfig(phase_lengths=(16, 16, 16, 16, 16), seeds=(0,))
        result = run_nonlinear(0, config, mode="hidden_node_perturbation")
        self.assertEqual(result["representation_seed"], nonlinear_representation_seed(0))
        self.assertEqual(result["direct_input_motor_edges"], ())
        self.assertTrue(all(payload["afferent_kind"] == "hidden" for payload in result["final_state"]["motor_modules"].values()))
        self.assertGreater(result["final_state"]["representation_cursor"], 0)

    def test_hidden_no_node_and_no_actor_controls_are_reported(self):
        config = NonlinearBenchmarkConfig(phase_lengths=(16, 16, 16, 16, 16), seeds=(0,))
        no_node = run_nonlinear(0, config, mode="hidden_no_node_perturbation")
        no_actor = run_nonlinear(0, config, mode="hidden_no_actor")
        self.assertEqual(no_node["direct_input_motor_edges"], ())
        self.assertEqual(no_actor["direct_input_motor_edges"], ())
        self.assertFalse(no_node["final_state"]["representation_learning_enabled"])
        self.assertTrue(no_actor["final_state"]["representation_learning_enabled"])

    def test_raw_context_control_has_direct_edges_and_is_matched(self):
        config = NonlinearBenchmarkConfig(phase_lengths=(16, 16, 16, 16, 16), seeds=(0,))
        raw = run_nonlinear(0, config, mode="raw_linear_modules")
        no_actor = run_nonlinear(0, config, mode="no_actor")
        frozen = run_nonlinear(0, config, mode="frozen")
        self.assertTrue(raw["direct_input_motor_edges"])
        self.assertEqual(raw["initial_cell_ids"], no_actor["initial_cell_ids"])
        self.assertEqual(raw["initial_synapse_ids"], frozen["initial_synapse_ids"])
        self.assertEqual(raw["phase_lengths"], no_actor["phase_lengths"])

    def test_dendritic_main_is_hidden_only_and_stationary_safety_is_reported(self):
        config = NonlinearBenchmarkConfig(
            phase_lengths=(16, 16, 16, 16, 16),
            seeds=(0,),
            stationary_safety_min_runs=1,
        )
        result = run_nonlinear(0, config, mode="dendritic_representation")
        product_cells = [
            payload for payload in result["final_state"]["cells"].values()
            if payload["kind"] == "hidden"
        ]
        self.assertTrue(product_cells)
        self.assertTrue(all(payload["activation_type"] == "dendritic_product" for payload in product_cells))
        self.assertEqual(result["direct_input_motor_edges"], ())
        self.assertTrue(all(payload["afferent_kind"] == "hidden" for payload in result["final_state"]["motor_modules"].values()))
        self.assertIn("stationary_safety", result)
        self.assertIn("safe", result["stationary_safety"])

    def test_dendritic_controls_and_heldouts_cannot_change_red_diagnostic(self):
        config = NonlinearBenchmarkConfig(
            phase_lengths=(16, 16, 16, 16, 16),
            seeds=(0, 1),
            stationary_safety_min_runs=1,
        )
        result = run_nonlinear_benchmark(config)
        self.assertIn("dendritic_representation", result)
        self.assertEqual(result["heldout_diagnostics"]["declared_primary"], False)
        self.assertTrue(result["heldout_diagnostics"]["sensor_pair_permutation"]["positive_runs"] >= 1)
        self.assertTrue(result["heldout_diagnostics"]["continuous_product"]["positive_runs"] >= 1)
        self.assertEqual(result["diagnostic"], "NONLINEAR_REPRESENTATION_NOT_ESTABLISHED")
        self.assertFalse(result["dendritic_representation"]["all_passed"])
        self.assertTrue(result["comparison"]["main_beats_raw_by_margin"])
        self.assertTrue(result["comparison"]["main_beats_hidden_no_node_by_margin"])


class AdaptiveDendriticProtocolTests(unittest.TestCase):
    def test_adaptive_environment_balances_64_states_without_task_leak(self):
        environment = AdaptiveDendriticEnvironment(seed=8203, phase_lengths=(64, 128, 64, 128, 64), schedule_seed=10211, task_seed=16223)
        observation = environment.reset()
        expected = {tuple(float(1 if (mask >> index) & 1 else -1) for index in range(6)) for mask in range(64)}
        self.assertEqual(len(observation.values), 6)
        self.assertFalse(hasattr(observation, "phase"))
        self.assertFalse(hasattr(observation, "target"))
        self.assertFalse(hasattr(observation, "task_pairs"))
        for start in range(0, environment.horizon, 64):
            self.assertEqual(set(environment.inputs[start:start + 64]), expected)

    def test_adaptive_streams_are_independent(self):
        lengths = (64, 64, 64, 64, 64)
        first = AdaptiveDendriticEnvironment(seed=8203, phase_lengths=lengths, schedule_seed=10211, task_seed=16223)
        schedule = AdaptiveDendriticEnvironment(seed=8203, phase_lengths=lengths, schedule_seed=10217, task_seed=16223)
        task = AdaptiveDendriticEnvironment(seed=8203, phase_lengths=lengths, schedule_seed=10211, task_seed=16224)
        inputs = AdaptiveDendriticEnvironment(seed=8209, phase_lengths=lengths, schedule_seed=10211, task_seed=16223)
        self.assertEqual(first.inputs, schedule.inputs)
        self.assertEqual(first.inputs, task.inputs)
        self.assertNotEqual(first.inputs, inputs.inputs)
        self.assertNotEqual(first.task_pairs, task.task_pairs)
        self.assertEqual(first.task_pairs, adaptive_dendritic_task_pairs(16223))

    def test_adaptive_affine_proof_is_exactly_zero(self):
        proof = adaptive_dendritic_affine_proof()
        self.assertEqual(proof["max_abs_projection"], 0.0)
        self.assertTrue(all(value == 0.0 for values in proof["coefficients"].values() for value in values))
        self.assertTrue(all(value == 0.0 for value in proof["optimal_affine_skill"].values()))

    def test_adaptive_environment_checkpoint_and_phase_validation(self):
        config = AdaptiveDendriticBenchmarkConfig(phase_lengths=(64, 64, 64, 64, 64), seeds=(0, 1), stationary_safety_min_runs=1)
        first = AdaptiveDendriticEnvironment(seed=8203, phase_lengths=config.phase_lengths, schedule_seed=10211, task_seed=16223)
        first.reset()
        for _ in range(7):
            first.step(0.13)
        resumed = AdaptiveDendriticEnvironment.from_state_dict(first.state_dict())
        self.assertEqual(first.state_dict(), resumed.state_dict())
        self.assertEqual(first.step(-0.2), resumed.step(-0.2))
        with self.assertRaises(ValueError):
            AdaptiveDendriticBenchmarkConfig(phase_lengths=(64, 65, 64, 64, 64)).validate()
        with self.assertRaises(ValueError):
            AdaptiveDendriticBenchmarkConfig(tail_fraction=0.0).validate()
        with self.assertRaises(ValueError):
            AdaptiveDendriticBenchmarkConfig(tail_skill_floor=float("nan")).validate()
        with self.assertRaises(ValueError):
            AdaptiveDendriticBenchmarkConfig(comparison_margin=float("inf")).validate()
        with self.assertRaises(ValueError):
            AdaptiveDendriticBenchmarkConfig(reactivation_lag_limit=-1).validate()

    def test_adaptive_phase_length_stream_is_seeded_and_valid(self):
        first = adaptive_dendritic_phase_lengths_for_seed(10211)
        same = adaptive_dendritic_phase_lengths_for_seed(10211)
        other = adaptive_dendritic_phase_lengths_for_seed(10217)
        self.assertEqual(first, same)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 5)
        self.assertTrue(all(length >= 64 and length % 64 == 0 for length in first))

    def test_adaptive_environment_rejects_invalid_checkpoint_state(self):
        environment = AdaptiveDendriticEnvironment(seed=8203, phase_lengths=(64,) * 5)
        invalid_t = environment.state_dict()
        invalid_t["t"] = environment.horizon + 1
        with self.assertRaises(ValueError):
            AdaptiveDendriticEnvironment.from_state_dict(invalid_t)
        invalid_reward = environment.state_dict()
        invalid_reward["last_reward"] = float("inf")
        with self.assertRaises(ValueError):
            AdaptiveDendriticEnvironment.from_state_dict(invalid_reward)
        invalid_input = environment.state_dict()
        invalid_input["inputs"][0][0] = 0.0
        with self.assertRaises(ValueError):
            AdaptiveDendriticEnvironment.from_state_dict(invalid_input)

    def test_adaptive_environment_checkpoint_load_and_resume_is_exact(self):
        organism = Organism.create_default(input_size=6, hidden_size=4, output_size=1, seed=81)
        environment = AdaptiveDendriticEnvironment(seed=8203, phase_lengths=(64,) * 5)
        observation = environment.reset()
        for _ in range(3):
            result = organism.step(observation.values, Modulators(exploration=0.2))
            observation = environment.step(result.outputs[0])
        with tempfile.NamedTemporaryFile(suffix=".json") as handle:
            organism.save_checkpoint(handle.name, environment)
            resumed, resumed_environment = Organism.load_checkpoint(handle.name)
            self.assertEqual(organism.state_dict(), resumed.state_dict())
            self.assertEqual(environment.state_dict(), resumed_environment.state_dict())
            result = organism.step(observation.values, Modulators(reward=observation.reward, exploration=0.2))
            resumed_result = resumed.step(observation.values, Modulators(reward=resumed_environment.last_reward, exploration=0.2))
            self.assertEqual(result, resumed_result)

    def test_fixed_random_pair_bank_is_organism_seeded_and_task_independent(self):
        def pairs(organism):
            return tuple(
                organism.cells[identifier].dendritic_sources
                for identifier in sorted(organism.cells)
                if organism.cells[identifier].kind == "hidden"
            )

        first = Organism.create_random_dendritic_pair_bank(input_size=6, hidden_size=8, output_size=1, seed=101)
        same = Organism.create_random_dendritic_pair_bank(input_size=6, hidden_size=8, output_size=1, seed=101)
        other = Organism.create_random_dendritic_pair_bank(input_size=6, hidden_size=8, output_size=1, seed=102)
        first_pairs = pairs(first)
        self.assertEqual(first_pairs, pairs(same))
        self.assertNotEqual(first_pairs, pairs(other))
        self.assertEqual(len(first_pairs), 8)
        self.assertEqual(len(set(first_pairs)), 8)
        self.assertTrue(all(len(pair) == 2 and pair[0] != pair[1] for pair in first_pairs))
        self.assertTrue(all(source in first.input_ids for pair in first_pairs for source in pair))
        first_environment = AdaptiveDendriticEnvironment(seed=8203, phase_lengths=(64,) * 5, task_seed=16223)
        other_environment = AdaptiveDendriticEnvironment(seed=8203, phase_lengths=(64,) * 5, task_seed=16224)
        self.assertNotEqual(first_environment.task_pairs, other_environment.task_pairs)
        self.assertEqual(first_pairs, pairs(Organism.create_random_dendritic_pair_bank(input_size=6, hidden_size=8, output_size=1, seed=101)))

    def test_adaptive_benchmark_is_red_and_controls_are_matched(self):
        config = AdaptiveDendriticBenchmarkConfig(phase_lengths=(64, 64, 64, 64, 64), seeds=(0, 1), stationary_safety_min_runs=1)
        result = run_adaptive_dendritic_benchmark(config)
        self.assertEqual(result["diagnostic"], "ADAPTIVE_DENDRITIC_NOT_ESTABLISHED")
        self.assertFalse(result["adaptive_dendritic"]["all_passed"])
        self.assertTrue(result["comparison"]["matched_seed_and_protocol"])
        self.assertTrue(result["comparison"]["matched_starting_structure"])
        self.assertIn("main_return_tail_skill_minus_additive_hidden", result["comparison"])
        self.assertIn("main_beats_additive_hidden_by_margin", result["comparison"])
        self.assertEqual(result["protocol"]["balanced_block_size"], 64)
        self.assertFalse(result["protocol"]["cubic_heldout_declared_primary"])
        self.assertIn("adaptive_dendritic_counts", result["adaptive_dendritic"])
        self.assertTrue(all(name in result["adaptive_dendritic"]["adaptive_dendritic_counts"] for name in ("proposals", "accepts", "rejects")))
        self.assertIn("adaptive_learning_rule", result)
        self.assertEqual(result["adaptive_learning_rule"]["min_evidence"], 3)
        self.assertEqual(set(result["adaptive_dendritic"]["adaptive_pair_acquisitions_by_policy"]), {"A", "B", "C"})

        main = run_adaptive_dendritic(0, config, mode="adaptive_dendritic")
        self.assertEqual(main["direct_input_motor_edges"], ())
        self.assertNotIn("task_pairs", main["final_state"])
        self.assertEqual(set(main["target_pair_overlap"]), {"A", "B", "C"})


class VariableOrderBenchmarkTests(unittest.TestCase):
    def _config(self, seeds=(0,)):
        return VariableOrderBenchmarkConfig(phase_lengths=(64,) * 8, seeds=tuple(seeds), stationary_safety_min_runs=1)

    def test_balanced_blocks_streams_and_schedule_are_exact(self):
        environment = VariableOrderBenchmarkEnvironment(17, (64,) * 8, schedule_seed=23, task_seed=5)
        for start in range(0, environment.horizon, 64):
            self.assertEqual(set(environment.inputs[start:start + 64]), {
                tuple(float(1 if (mask >> index) & 1 else -1) for index in range(6)) for mask in range(64)
            })
        self.assertEqual(variable_order_phase_lengths_for_seed(3, (64,) * 8), variable_order_phase_lengths_for_seed(3, (64,) * 8))
        self.assertNotEqual(variable_order_phase_lengths_for_seed(3, (64,) * 8), variable_order_phase_lengths_for_seed(4, (64,) * 8))
        self.assertTrue(all(length % 64 == 0 for length in variable_order_phase_lengths_for_seed(3, (256,) * 8)))
        manifest = variable_order_seed_manifest(0)
        self.assertEqual(len(manifest), 6)
        self.assertEqual(len(set(manifest)), 6)
        self.assertEqual(manifest, variable_order_seed_manifest(0))

    def test_task_manifest_and_environment_checkpoint_are_evaluator_only(self):
        environment = VariableOrderBenchmarkEnvironment(7, (64,) * 8, schedule_seed=11, task_seed=13)
        checkpoint = json.loads(json.dumps(environment.state_dict()))
        replay = VariableOrderBenchmarkEnvironment.from_state_dict(checkpoint)
        self.assertEqual(replay.state_dict(), environment.state_dict())
        missing = dict(checkpoint)
        del missing["task_features"]
        with self.assertRaises(ValueError):
            VariableOrderBenchmarkEnvironment.from_state_dict(missing)
        missing_marker = dict(checkpoint)
        del missing_marker["custom_task_features"]
        with self.assertRaises(ValueError):
            VariableOrderBenchmarkEnvironment.from_state_dict(missing_marker)
        canonical_mismatch = dict(checkpoint, task_features=[list(feature) for feature in VARIABLE_ORDER_TASK_MANIFEST[0]])
        with self.assertRaises(ValueError):
            VariableOrderBenchmarkEnvironment.from_state_dict(canonical_mismatch)
        custom_features = ((0, 1), (0, 1, 2), (0, 1), (0, 1, 2), (0, 1), (0, 1, 2))
        custom = VariableOrderBenchmarkEnvironment(7, (64,) * 8, schedule_seed=11, task_seed=13, task_features=custom_features, custom_task_features=True)
        custom_replay = VariableOrderBenchmarkEnvironment.from_state_dict(json.loads(json.dumps(custom.state_dict())))
        self.assertEqual(custom_replay.state_dict(), custom.state_dict())
        self.assertTrue(custom_replay.state_dict()["custom_task_features"])
        corrupted = dict(checkpoint)
        corrupted["inputs"] = list(corrupted["inputs"])
        corrupted["inputs"][1] = corrupted["inputs"][0]
        with self.assertRaises(ValueError):
            VariableOrderBenchmarkEnvironment.from_state_dict(corrupted)
        with self.assertRaises(ValueError):
            VariableOrderBenchmarkEnvironment.from_state_dict(dict(checkpoint, policy_sequence=["leak"]))
        result = run_variable_order(0, self._config())
        self.assertNotIn("task_features", result["final_state"])
        self.assertIn("task_features", result["environment_state"])
        self.assertEqual(tuple(result["environment_state"]["policy_sequence"]), VARIABLE_ORDER_POLICY_SEQUENCE)

    def test_config_rejects_gate_weakening_values(self):
        for field in ("pair_tail_skill_floor", "cubic_tail_skill_floor", "return_early_skill_floor", "return_tail_skill_floor", "comparison_margin"):
            with self.assertRaises(ValueError):
                VariableOrderBenchmarkConfig(**{field: -0.01}).validate()
        for field, value in (("pair_tail_sign_accuracy_floor", -0.01), ("cubic_tail_sign_accuracy_floor", 1.01), ("stationary_safety_min_runs", 0)):
            with self.assertRaises(ValueError):
                VariableOrderBenchmarkConfig(**{field: value}).validate()

    def test_exact_pair_and_cubic_projection_certificates(self):
        proof = variable_order_parity_proof()
        self.assertEqual(len(proof["states"]), 64)
        self.assertEqual(proof["max_abs_pair_affine_projection"], 0)
        self.assertEqual(proof["max_abs_pair_own_basis_error"], 0)
        self.assertEqual(proof["max_abs_pair_other_degree2_projection"], 0)
        self.assertEqual(set(proof["pair_degree2_own_basis"].values()), {64})
        self.assertEqual(proof["max_abs_cubic_degree2_projection"], 0)
        self.assertEqual(len(VARIABLE_ORDER_TASK_MANIFEST), 24)

    def test_red_report_schema_and_matched_additive_control(self):
        report = run_variable_order_benchmark(self._config())
        self.assertEqual(report["diagnostic"], "VARIABLE_ORDER_REPRESENTATION_NOT_ESTABLISHED")
        self.assertFalse(report["variable_order"]["all_passed"])
        self.assertEqual(set(report["runs"]), {"variable_order", "additive_hidden", "raw_linear_modules", "fixed_random_order_bank", "no_actor", "frozen"})
        self.assertTrue(report["comparison"]["matched_starting_structure"])
        self.assertTrue(report["comparison"]["matched_exploration_tape"])
        self.assertTrue(report["comparison"]["matched_input_schedule_task_manifest"])
        self.assertTrue(report["comparison"]["matched_phase_lengths"])
        self.assertTrue(report["comparison"]["matched_task_features"])
        self.assertTrue(report["comparison"]["matched_input_stream"])
        self.assertEqual(report["protocol"]["stream_names"], ["organism", "input", "schedule", "exploration", "representation", "task"])
        self.assertEqual(len(report["protocol"]["representation_seeds"]), 1)
        self.assertIsNot(report["runs"]["variable_order"], report["runs"]["additive_hidden"])
        self.assertTrue(all("stationary_safety_by_seed" in report[mode] for mode in ("variable_order", "additive_hidden")))
        self.assertIn("safe", report["variable_order"]["stationary_safety_by_seed"]["0"])
        self.assertIn("recruitments", report["variable_order"]["stationary_safety_by_seed"]["0"])
        self.assertTrue(all(name in report["variable_order"]["passed"] for name in ("initial_pair_tail_skill_all", "initial_pair_tail_sign_all", "initial_cubic_tail_skill_all", "initial_cubic_tail_sign_all", "margin_pair_vs_raw", "margin_cubic_vs_raw", "margin_pair_vs_additive", "margin_cubic_vs_additive")))
        self.assertEqual(report["protocol"]["policy_sequence"], list(VARIABLE_ORDER_POLICY_SEQUENCE))
        self.assertEqual(report["fixed_random_order_bank"]["seeds"], 1)
        self.assertEqual(report["runs"]["fixed_random_order_bank"][0]["fixed_order_bank_status"], "deferred_order3_schema")

    def test_primary_main_has_no_direct_input_motor_and_is_resource_bounded(self):
        result = run_variable_order(0, self._config())
        self.assertEqual(result["direct_input_motor_edges"], ())
        self.assertLessEqual(result["module_count"], 6)
        self.assertNotIn("phase", result["final_state"])

    def test_v10_learner_opt_out_preserves_stage1_red(self):
        config = self._config()
        config.variable_order_learning_enabled = False
        report = run_variable_order_benchmark(config)
        self.assertEqual(report["diagnostic"], "VARIABLE_ORDER_REPRESENTATION_NOT_ESTABLISHED")
        self.assertFalse(report["protocol"]["variable_order_learning_enabled"])
        self.assertFalse(report["variable_order"]["passed"]["variable_order_learning_enabled"])


class CompositionalBenchmarkTests(unittest.TestCase):
    def _config(self, seeds=(0,)):
        return CompositionalBenchmarkConfig(
            phase_lengths=(64,) * 12,
            seeds=tuple(seeds),
            stationary_safety_min_runs=1,
            canonical_reuse_min_runs=1,
            composed_install_min_runs=1,
            closure_min_runs=1,
            composition_learning_enabled=False,
        )

    def test_balanced_blocks_streams_and_phase_schedule_are_exact(self):
        environment = CompositionalBenchmarkEnvironment(17, (64,) * 12, schedule_seed=23, task_seed=5)
        expected = {
            tuple(float(1 if (mask >> index) & 1 else -1) for index in range(6))
            for mask in range(64)
        }
        for start in range(0, environment.horizon, 64):
            self.assertEqual(set(environment.inputs[start:start + 64]), expected)
        self.assertEqual(environment.inputs, CompositionalBenchmarkEnvironment(17, (64,) * 12, schedule_seed=23, task_seed=5).inputs)
        self.assertNotEqual(
            compositional_phase_lengths_for_seed(3, (256,) * 12),
            compositional_phase_lengths_for_seed(4, (256,) * 12),
        )
        manifest = compositional_seed_manifest(0)
        self.assertEqual(len(manifest), 7)
        self.assertEqual(len(set(manifest)), 7)
        self.assertEqual(manifest, compositional_seed_manifest(0))

    def test_factorization_and_exact_walsh_certificates(self):
        for row in COMPOSITIONAL_TASK_MANIFEST:
            p0, p1, q0, _, p2, p3, q1, _ = row
            self.assertTrue(set(p0).isdisjoint(p1))
            self.assertEqual(tuple(sorted(p0 + p1)), q0)
            self.assertTrue(set(p2).isdisjoint(p3))
            self.assertEqual(tuple(sorted(p2 + p3)), q1)
        self.assertTrue(all(item["Q0_union_exact"] and item["Q1_union_exact"] for item in compositional_factorization_manifest()))
        proof = compositional_walsh_proof()
        self.assertEqual(proof["max_abs_lower_projection"], 0)
        self.assertEqual(proof["max_abs_other_quartic_projection"], 0)
        self.assertEqual(set(proof["own_quartic_basis"].values()), {64})
        self.assertTrue(proof["integer_certificate"])

    def test_checkpoint_replay_strict_schema_and_custom_marker(self):
        environment = CompositionalBenchmarkEnvironment(7, (64,) * 12, schedule_seed=11, task_seed=13)
        checkpoint = json.loads(json.dumps(environment.state_dict()))
        replay = CompositionalBenchmarkEnvironment.from_state_dict(checkpoint)
        self.assertEqual(replay.state_dict(), environment.state_dict())
        action = 0.25
        self.assertEqual(replay.step(action), environment.step(action))
        for field in ("task_features", "custom_task_features", "inputs"):
            missing = dict(checkpoint)
            del missing[field]
            with self.assertRaises(ValueError):
                CompositionalBenchmarkEnvironment.from_state_dict(missing)
        corrupted = dict(checkpoint, inputs=[list(values) for values in checkpoint["inputs"]])
        corrupted["inputs"][1] = corrupted["inputs"][0]
        with self.assertRaises(ValueError):
            CompositionalBenchmarkEnvironment.from_state_dict(corrupted)
        mismatch = dict(checkpoint, task_features=[list(feature) for feature in COMPOSITIONAL_TASK_MANIFEST[0]])
        with self.assertRaises(ValueError):
            CompositionalBenchmarkEnvironment.from_state_dict(mismatch)
        custom = CompositionalBenchmarkEnvironment(
            7, (64,) * 12, schedule_seed=11, task_seed=13,
            task_features=COMPOSITIONAL_TASK_MANIFEST[0], custom_task_features=True,
        )
        custom_replay = CompositionalBenchmarkEnvironment.from_state_dict(json.loads(json.dumps(custom.state_dict())))
        self.assertEqual(custom_replay.state_dict(), custom.state_dict())
        self.assertTrue(custom_replay.state_dict()["custom_task_features"])

    def test_red_report_controls_are_matched_and_schema_is_explicit(self):
        report = run_compositional_benchmark(self._config())
        self.assertEqual(report["diagnostic"], "COMPOSITIONAL_REPRESENTATION_NOT_ESTABLISHED")
        self.assertFalse(report["compositional"]["all_passed"])
        self.assertEqual(
            set(report["runs"]),
            {"compositional", "composition_disabled_v10", "additive_hidden", "raw_linear_modules", "no_actor", "frozen"},
        )
        self.assertTrue(report["comparison"]["matched_starting_structure"])
        self.assertTrue(report["comparison"]["matched_exploration_tape"])
        self.assertTrue(report["comparison"]["matched_phase_lengths"])
        self.assertTrue(report["comparison"]["matched_task_features"])
        self.assertEqual(report["protocol"]["stream_names"], ["organism", "input", "schedule", "exploration", "representation", "composition", "task"])
        self.assertEqual(set(report["protocol"]["stream_seeds"]["0"]), set(report["protocol"]["stream_names"]))
        self.assertEqual(len(set(report["protocol"]["stream_seeds"]["0"].values())), 7)
        self.assertEqual(report["protocol"]["policy_sequence"], list(COMPOSITIONAL_POLICY_SEQUENCE))
        self.assertEqual(report["protocol"]["fingerprint_window"], 16)
        self.assertEqual(report["protocol"]["fixed_random_order4_status"], "deferred_non_gating")
        self.assertFalse(report["protocol"]["composition_learning_enabled"])
        self.assertEqual(report["protocol"]["composition_substrate_status"], "seeded_substrate_only")
        self.assertEqual(report["runs"]["compositional"][0]["composition_substrate_status"], "seeded_substrate_only")
        self.assertEqual(report["runs"]["composition_disabled_v10"][0]["composition_substrate_status"], "disabled")
        safety = report["compositional"]["stationary_safety_by_seed"]["0"]
        self.assertTrue(safety["composition_substrate_enabled"])
        self.assertEqual(safety["composition_substrate_status"], "seeded_substrate_only")
        for gate in (
            "initial_pair_tail_skill_all", "initial_pair_tail_sign_all", "initial_cubic_tail_skill_all",
            "initial_cubic_tail_sign_all", "initial_composed_tail_skill_all", "initial_composed_tail_sign_all",
            "returned_P0_early", "returned_Q0_tail", "canonical_reuse_all_returns",
            "composed_feature_installed_before_Q_tail", "valid_four_distinct_leaf_closures",
            "Q_return_margin_vs_raw", "Q_return_margin_vs_composition_disabled", "composition_ablation_Q_drop",
        ):
            self.assertIn(gate, report["red_report_gates"])

    def test_main_run_has_no_evaluator_metadata_or_direct_input_motor_edges(self):
        result = run_compositional(0, self._config())
        self.assertEqual(result["direct_input_motor_edges"], ())
        self.assertLessEqual(result["module_count"], 8)
        self.assertNotIn("task_features", result["final_state"])
        self.assertNotIn("target", result["final_state"])
        self.assertTrue(result["target_blind"])

    def test_default_stage3_enablement_and_real_causal_ablation(self):
        self.assertTrue(CompositionalBenchmarkConfig().composition_learning_enabled)
        config = self._config()
        config.composition_learning_enabled = True
        result = run_compositional(0, config)
        ablation = _evaluate_composition_causal_ablation(result, config)
        self.assertTrue(ablation["available"])
        self.assertGreater(ablation["q_drop"], 0.20)
        self.assertLessEqual(ablation["direct_degradation"], 0.10)
        self.assertTrue(ablation["original_final_state_preserved"])
        self.assertTrue(ablation["ablated_only_edge_mutation"])
        self.assertTrue(ablation["ablated_untouched_synapses_preserved"])
        self.assertTrue(ablation["ablated_edge_ids"])
        self.assertEqual(set(ablation["ablated_edges_before"]), set(ablation["ablated_edges_after"]))
        self.assertTrue(all(payload["strength"] != 0.0 for payload in ablation["ablated_edges_before"].values()))
        self.assertTrue(all(payload["strength"] == 0.0 for payload in ablation["ablated_edges_after"].values()))
        self.assertNotEqual(result["phase_metrics"]["Q0_return"]["tail_skill"], ablation["q0"]["ablated"]["tail_skill"])

    def test_composition_control_difference_is_separate_from_causal_ablation(self):
        report = run_compositional_benchmark(self._config())
        self.assertIn("composition_control_Q_difference", report["comparison"])
        self.assertIn("causal_Q_ablation_drop", report["comparison"])
        self.assertNotEqual(report["comparison"]["composition_control_Q_difference"], report["comparison"]["causal_Q_ablation_drop"])
        self.assertIsNone(report["comparison"]["causal_Q_ablation_drop"])
        self.assertFalse(report["comparison"]["causal_Q_ablation_drop_gate"])
        self.assertFalse(report["comparison"]["causal_direct_degradation_gate"])
        self.assertIn("causal_ablation", report["runs"]["compositional"][0])

    def test_composition_install_timing_is_reported_against_Q_tails(self):
        config = self._config()
        config.composition_learning_enabled = True
        result = run_compositional(0, config)
        timing = result["composition_install_timing"]
        self.assertIn("Q0", timing)
        self.assertIn("Q1", timing)
        self.assertEqual(result["composed_feature_installed_before_Q_tail"], timing["Q0"]["passed"] and timing["Q1"]["passed"])
        self.assertTrue(all(event["step"] < timing["Q0"]["tail_start"] for event in timing["Q0"]["events_before_tail"]))

    def test_q0_install_cannot_satisfy_q1_timing_gate(self):
        config = self._config()
        config.composition_learning_enabled = True
        result = run_compositional(0, config)
        final = Organism.from_state_dict(result["final_state"])
        q0_event = result["composition_install_timing"]["Q0"]["events_before_tail"][0]
        self.assertTrue(_composition_install_matches_feature(final, q0_event, result["task_features"][2]))
        self.assertFalse(_composition_install_matches_feature(final, q0_event, result["task_features"][6]))
        self.assertTrue(result["composition_install_timing"]["Q1"]["passed"])


class StreamingCompositionalBenchmarkTests(unittest.TestCase):
    def _config(self, **kwargs):
        values = {
            "seeds": (0,), "segment_length_range": (97, 127),
            "canonical_reuse_min_runs": 1, "stationary_safety_min_runs": 1,
            "composed_install_min_runs": 1, "closure_min_runs": 1,
        }
        values.update(kwargs)
        return StreamingCompositionalBenchmarkConfig(**values)

    def test_schedule_and_global_stream_are_deterministic_and_irregular(self):
        first = streaming_compositional_schedule(17, length_range=(97, 127))
        second = streaming_compositional_schedule(17, length_range=(97, 127))
        self.assertEqual(first, second)
        environment = StreamingCompositionalBenchmarkEnvironment(101, 17, 303, self._config())
        expected = set(environment._states())
        self.assertTrue(all(set(environment.inputs[start:start + 64]) == expected for start in range(0, environment.horizon, 64)))
        self.assertTrue(all(length % 64 for length in environment.segment_lengths))
        self.assertTrue(all(start % 64 for start in environment.segment_starts[1:]))
        self.assertEqual(len(environment.task_sequence), 24)
        self.assertTrue(all(environment.task_sequence.count(name) == 3 for name in ("P0", "P1", "P2", "P3", "Q0", "Q1", "C0", "C1")))

    def test_environment_checkpoint_replays_mid_segment_direct_and_json(self):
        environment = StreamingCompositionalBenchmarkEnvironment(101, 17, 303, self._config())
        observation = environment.reset()
        for _ in range(73):
            observation = environment.step(0.125)
        checkpoint = environment.state_dict()
        direct = StreamingCompositionalBenchmarkEnvironment.from_state_dict(checkpoint)
        encoded = StreamingCompositionalBenchmarkEnvironment.from_state_dict(json.loads(json.dumps(checkpoint)))
        self.assertEqual(direct.state_dict(), checkpoint)
        self.assertEqual(encoded.state_dict(), checkpoint)
        for _ in range(13):
            expected = environment.step(-0.25)
            self.assertEqual(direct.step(-0.25), expected)
            self.assertEqual(encoded.step(-0.25), expected)
            self.assertEqual(encoded.state_dict(), direct.state_dict())

    def test_streaming_run_is_target_blind_and_reports_integrity(self):
        result = run_streaming_compositional(0, self._config())
        self.assertTrue(result["target_blind"])
        self.assertNotIn("task_features", result["final_state"])
        self.assertNotIn("target", result["final_state"])
        self.assertTrue(result["stream_integrity"]["complete_global_64_blocks"])
        self.assertTrue(result["stream_integrity"]["boundaries_mid_superblock"])
        self.assertTrue(result["stream_integrity"]["segment_lengths_non_multiple_64"])
        self.assertEqual({name: result["task_sequence"].count(name) for name in ("P0", "P1", "P2", "P3", "Q0", "Q1", "C0", "C1")}, {name: 3 for name in ("P0", "P1", "P2", "P3", "Q0", "Q1", "C0", "C1")})
        self.assertEqual(len(result["reactivation_lag_by_segment"]["Q0"]), 2)
        self.assertEqual(set(result["valid_four_leaf_closure_by_task"]), {"Q0", "Q1"})

    def test_routing_diagnostic_is_evaluator_only_deterministic_and_non_gating(self):
        config = self._config()
        result = run_streaming_compositional(0, config)
        checkpoint = copy.deepcopy(result["final_state"])
        first = streaming_routing_diagnostic(result, config)
        second = streaming_routing_diagnostic(result, config)
        self.assertEqual(first, second)
        self.assertTrue(first["evaluator_only"])
        self.assertFalse(first["acceptance_gating"])
        self.assertTrue(first["organism_checkpoint_unchanged"])
        self.assertEqual(result["final_state"], checkpoint)
        self.assertEqual(len(first["return_segments"]), 16)
        self.assertNotIn("routing_diagnostic", result)
        self.assertNotIn("task_features", result["final_state"])
        self.assertEqual(
            set(first["mean_scores"]),
            {"learned", "oracle_boundary_learned_selection", "learned_detection_oracle_selection", "oracle_detection_oracle_selection", "perfect_policy"},
        )

    def test_owner_evidence_shadow_does_not_change_stream_behavior(self):
        enabled = run_streaming_compositional(0, self._config(owner_evidence_shadow_enabled=True, owner_probe_enabled=False))
        disabled = run_streaming_compositional(0, self._config(owner_evidence_shadow_enabled=False, owner_probe_enabled=False))
        for key in ("actions", "rewards", "canonical_by_segment", "canonical_reuse", "reactivation_lag_by_segment"):
            self.assertEqual(enabled[key], disabled[key])
        self.assertTrue(enabled["final_state"]["variable_order_owner_evidence_enabled"])
        self.assertFalse(disabled["final_state"]["variable_order_owner_evidence_enabled"])

    def test_temporal_refresh_discards_boundary_straddling_false_feature(self):
        result = run_streaming_compositional(22, StreamingCompositionalBenchmarkConfig(seeds=(22,)))
        final = result["final_state"]
        kinds = [event.get("kind") for event in final["events"]]
        self.assertIn("variable_order_temporal_fingerprint_refreshed", kinds)
        self.assertIn("k3:input-002|input-004|input-005", final["variable_order_feature_owners"])
        self.assertNotIn("k2:input-003|input-004", final["variable_order_feature_owners"])
        self.assertEqual(result["module_count"], 8)

    def test_each_return_occurrence_is_gated_not_just_the_average(self):
        config = self._config()
        result = run_streaming_compositional(0, config)
        passing = copy.deepcopy(result)
        for metrics in passing["phase_metrics"].values():
            for name in metrics:
                metrics[name] = 1.0
        self.assertTrue(_streaming_summary([passing], config)["passed"]["returns_early_all"])
        failing = copy.deepcopy(passing)
        middle_key = next("Q0_%d" % index for index, task in enumerate(failing["task_sequence"]) if task == "Q0" and index != 2)
        failing["phase_metrics"][middle_key]["early_skill"] = 0.0
        summary = _streaming_summary([failing], config)
        self.assertGreater(summary["mean_phase_metrics"]["Q0_return"]["early_skill"], config.return_early_skill_floor)
        self.assertFalse(summary["passed"]["returns_early_all"])

    def test_controls_match_stream_schedule_tape_and_starting_structure(self):
        config = self._config()
        main = run_streaming_compositional(0, config, "compositional")
        control = run_streaming_compositional(0, config, "composition_disabled_v10")
        raw = run_streaming_compositional(0, config, "raw_linear_modules")
        for other in (control, raw):
            self.assertEqual(main["environment_state"]["inputs"], other["environment_state"]["inputs"])
            self.assertEqual(main["task_sequence"], other["task_sequence"])
            self.assertEqual(main["segment_lengths"], other["segment_lengths"])
            self.assertEqual(main["exploration_tape_signature"], other["exploration_tape_signature"])
        self.assertEqual(main["initial_cell_ids"], control["initial_cell_ids"])
        self.assertEqual(main["initial_synapse_ids"], control["initial_synapse_ids"])

    def test_delayed_reward_alignment_and_config_validation(self):
        environment = StreamingCompositionalBenchmarkEnvironment(101, 17, 303, self._config())
        observation = environment.reset()
        target = environment.target_at()
        next_observation = environment.step(0.25)
        expected = max(-1.0, min(1.0, (target * target - (0.25 - target) ** 2) / 4.0))
        self.assertEqual(next_observation.reward, expected)
        self.assertEqual(environment.t, 1)
        with self.assertRaises(ValueError):
            StreamingCompositionalBenchmarkConfig(segment_length_range=(64, 127)).validate()
        with self.assertRaises(ValueError):
            self._config(owner_evidence_shadow_enabled=False, owner_probe_enabled=True).validate()
        with self.assertRaises(ValueError):
            StreamingCompositionalBenchmarkConfig(segment_length_range=(97.5, 127)).validate()
        with self.assertRaises(ValueError):
            StreamingCompositionalBenchmarkConfig(segment_length_range=(True, 127)).validate()
        with self.assertRaises(ValueError):
            StreamingCompositionalBenchmarkConfig(seeds=(24,)).validate()
        checkpoint = environment.state_dict()
        del checkpoint["segment_lengths"]
        with self.assertRaises(ValueError):
            StreamingCompositionalBenchmarkEnvironment.from_state_dict(checkpoint)
        checkpoint = environment.state_dict()
        checkpoint["unexpected"] = 1
        with self.assertRaises(ValueError):
            StreamingCompositionalBenchmarkEnvironment.from_state_dict(checkpoint)
        checkpoint = environment.state_dict()
        checkpoint["t"] = "1"
        with self.assertRaises(ValueError):
            StreamingCompositionalBenchmarkEnvironment.from_state_dict(checkpoint)
        checkpoint = environment.state_dict()
        checkpoint["inputs"] = [[1.0] * 5]
        with self.assertRaises(ValueError):
            StreamingCompositionalBenchmarkEnvironment.from_state_dict(checkpoint)

    def test_causal_ablation_is_same_lineage_and_opt_out_stays_red(self):
        config = self._config()
        result = run_streaming_compositional(0, config)
        ablation = result["causal_ablation"]
        self.assertIn("original_final_state_preserved", ablation)
        self.assertTrue(ablation["original_final_state_preserved"])
        if ablation["available"]:
            self.assertTrue(ablation["ablated_edge_ids"])
            self.assertTrue(ablation["ablated_only_edge_mutation"])
            self.assertTrue(ablation["ablated_untouched_synapses_preserved"])
        disabled = self._config(composition_learning_enabled=False)
        report = run_streaming_compositional_benchmark(disabled)
        self.assertEqual(report["diagnostic"], "STREAMING_COMPOSITION_NOT_ESTABLISHED")
        self.assertFalse(report["streaming_compositional"]["passed"]["composition_learning_enabled"])
        self.assertFalse(report["streaming_compositional"]["passed"]["causal_ablation_all_runs"])


class ActorEligibilityTests(unittest.TestCase):
    def _motor_organism(self, seed):
        organism = Organism.create_default(input_size=1, hidden_size=1, output_size=1, seed=seed)
        graph = SparseDirectedGraph()
        graph.add(Synapse("input-000", "hidden-000", 1.0))
        graph.add(Synapse("hidden-000", "output-000", 0.0))
        organism.graph = graph
        organism.resources.counters["synapses"] = 2
        organism.structural_plasticity_enabled = False
        organism.legacy_learning_enabled = False
        organism.actor_learning_rate = 0.02
        return organism

    def _learn_polarity(self, target_sign, seed, misalign=False):
        organism = self._motor_organism(seed)
        tape_rng = __import__("random").Random(seed + 99)
        organism.set_exploration_tape(tuple(tape_rng.choice((-1.0, 1.0)) for _ in range(260)))
        reward = 0.0
        for _ in range(200):
            result = organism.step((0.8,), Modulators(reward=reward, exploration=0.2))
            if misalign:
                motor = organism.graph.get("hidden-000", "output-000")
                motor.actor_eligibility_trace *= -1.0
            action = result.outputs[0]
            target = target_sign * 0.8
            reward = (target * target - (action - target) ** 2) / 4.0
        organism.apply_outcome(reward)
        return organism.graph.get("hidden-000", "output-000").strength

    def test_motor_perturbation_escapes_zero_mean_and_updates_once(self):
        organism = self._motor_organism(5)
        organism.set_exploration_tape((1.0, 1.0, 1.0))
        first = organism.step((1.0,), Modulators(exploration=0.2))
        motor = organism.graph.get("hidden-000", "output-000")
        self.assertEqual(organism.cells["output-000"].activation, 0.0)
        self.assertNotEqual(motor.actor_eligibility_trace, 0.0)
        before = motor.strength
        previous_trace = motor.actor_eligibility_trace
        reward = (1.0 - (first.outputs[0] - 1.0) ** 2) / 4.0
        organism.step((1.0,), Modulators(reward=reward, exploration=0.2))
        changed = motor.strength
        self.assertAlmostEqual(changed, before + organism.actor_learning_rate * previous_trace * reward, places=14)
        self.assertNotEqual(motor.actor_eligibility_trace, 0.0)
        organism.apply_outcome(0.0)
        self.assertEqual(motor.actor_eligibility_trace, 0.0)
        with self.assertRaises(ValueError):
            organism.apply_outcome(0.0)

    def test_aligned_actor_trace_moves_mapping_in_both_polarities(self):
        for seed in range(6):
            self.assertGreater(self._learn_polarity(1.0, seed), 0.10)
            self.assertLess(self._learn_polarity(-1.0, seed), -0.10)

    def test_misaligned_perturbations_do_not_create_the_same_learning_signal(self):
        aligned = [self._learn_polarity(1.0, seed) for seed in range(6)]
        shuffled = [self._learn_polarity(1.0, seed, misalign=True) for seed in range(6)]
        self.assertGreater(sum(aligned) / len(aligned), 0.10)
        self.assertLess(sum(shuffled) / len(shuffled), sum(aligned) / len(aligned) - 0.10)


class GeneralStructuralBenchmarkTests(unittest.TestCase):
    def test_frozen_general_structure_manifest_passes_without_task_metadata(self):
        config = GeneralStructuralBenchmarkConfig(
            seeds=(0, 1), growth_min_runs=2, reuse_min_runs=2, pruning_min_runs=2,
        )
        report = run_general_structural_benchmark(config)
        self.assertEqual(report["diagnostic"], "GENERAL_STRUCTURAL_PASS")
        self.assertTrue(report["general_structural"]["all_passed"])
        self.assertFalse(report["protocol"]["task_identity_visible"])
        self.assertEqual(set(report["runs"]), {"general_structural", "structure_disabled"})
        self.assertTrue(all(result["max_depth"] >= 2 for result in report["runs"]["general_structural"]))

    def test_general_structure_run_is_deterministic_and_checkpoint_is_json_exact(self):
        config = GeneralStructuralBenchmarkConfig(seeds=(0,), growth_min_runs=1, reuse_min_runs=1, pruning_min_runs=1)
        first = run_general_structural(0, config)
        second = run_general_structural(0, config)
        self.assertEqual(first, second)
        restored = Organism.from_state_dict(json.loads(json.dumps(first["final_state"])))
        self.assertEqual(restored.state_dict(), first["final_state"])


class ContinuousLifetimeBenchmarkTests(unittest.TestCase):
    def test_delayed_credit_buffer_is_bounded_and_checkpoint_exact(self):
        organism = Organism.create_default(input_size=2, hidden_size=3, output_size=1, seed=23)
        organism.enable_context_modules(max_modules=2, afferent_kind="hidden")
        organism.enable_motor_bootstrap()
        organism.enable_delayed_credit(3)
        organism.set_exploration_tape((0.25,) * 12)
        rewards = (0.0, 0.0, 0.0, 0.2)
        for index, reward in enumerate(rewards):
            organism.step((1.0, -1.0), Modulators(reward=reward, exploration=0.2))
            self.assertLessEqual(len(organism.delayed_credit_queue), 3)
        checkpoint = json.loads(json.dumps(organism.state_dict()))
        restored = Organism.from_state_dict(checkpoint)
        self.assertEqual(restored.state_dict(), organism.state_dict())
        for reward in (0.15, 0.05, -0.2):
            left = organism.step((-1.0, 1.0), Modulators(reward=reward, exploration=0.2))
            right = restored.step((-1.0, 1.0), Modulators(reward=reward, exploration=0.2))
            self.assertEqual(left, right)
        self.assertEqual(organism.state_dict(), restored.state_dict())

    def test_continuous_stream_is_phase_blind_noisy_delayed_and_checkpoint_exact(self):
        config = ContinuousLifetimeBenchmarkConfig(seeds=(0,))
        environment = ContinuousLifetimeEnvironment(17, 19, config)
        observation = environment.reset()
        self.assertEqual(observation.reward, 0.0)
        self.assertFalse(hasattr(observation, "phase"))
        target = environment.target_at()
        for _ in range(config.reward_delay):
            observation = environment.step(target)
            self.assertEqual(observation.reward, 0.0)
        checkpoint = json.loads(json.dumps(environment.state_dict()))
        restored = ContinuousLifetimeEnvironment.from_state_dict(checkpoint)
        self.assertEqual(restored.state_dict(), environment.state_dict())
        self.assertEqual(restored.step(0.25), environment.step(0.25))
        self.assertTrue(any(a != b for clean, noisy in zip(environment.clean_inputs, environment.inputs) for a, b in zip(clean, noisy)))

    def test_current_continuous_baseline_is_deterministic_and_bounded(self):
        config = ContinuousLifetimeBenchmarkConfig(seeds=(0,), growth_min_runs=1, reuse_min_runs=1, bounded_min_runs=1)
        first = run_continuous_lifetime(0, config)
        second = run_continuous_lifetime(0, config)
        self.assertEqual(first, second)
        self.assertTrue(first["resource_bounded"])
        self.assertEqual(len(first["phases"]), len(first["environment_state"]["sequence"]))

    def test_continuous_report_includes_matched_structure_control(self):
        config = ContinuousLifetimeBenchmarkConfig(seeds=(0,), growth_min_runs=1, reuse_min_runs=1, bounded_min_runs=1)
        report = run_continuous_lifetime_benchmark(config)
        self.assertEqual(set(report["runs"]), {"soma", "structure_disabled"})
        self.assertEqual(set(report["continuous_lifetime"]["phase_tail_skill"]), set(report["continuous_lifetime"]["control_phase_tail_skill"]))
        self.assertIn("novel_structure_control_margin", report["continuous_lifetime"]["passed"])


class ObservationAtCheckpoint:
    def __init__(self, environment, action=None):
        if action is None:
            self.values = (environment.cue, environment.distractor)
            self.reward = environment.last_reward
        else:
            observation = environment.step(action)
            self.values = observation.values
            self.reward = observation.reward


if __name__ == "__main__":
    unittest.main()
