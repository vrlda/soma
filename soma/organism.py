"""The SOMA organism: state, sparse graph, local learning, and structure."""

from dataclasses import asdict, dataclass, field
import copy
from itertools import combinations
import json
import math
import os
import random
import tempfile
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .cells import Cell
from .resources import ResourceBudget
from .synapses import SparseDirectedGraph, Synapse, synapse_to_dict


@dataclass
class Modulators:
    reward: float = 0.0
    novelty: float = 0.0
    uncertainty: float = 0.0
    salience: float = 0.0
    exploration: float = 0.0


@dataclass
class StepResult:
    outputs: Tuple[float, ...]
    reward: float
    energy_used: float
    structural_events: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class MotorModule:
    """A graph-native context-specific motor engram."""

    id: str
    cell_id: str
    baseline: float = 0.0
    reward_mean: float = 0.0
    reward_variance: float = 0.01
    surprise_evidence: float = 0.0
    confidence: float = 0.0
    negative_surprise: float = 0.0
    negative_streak: int = 0
    protected: bool = False
    dormant: bool = False
    reward_count: int = 0
    probe_count: int = 0
    total_reward: float = 0.0
    detector_mean: float = 0.0
    detector_variance: float = 0.01
    detector_count: int = 0
    detector_cusum: float = 0.0
    detector_frozen: bool = False
    detector_predictor: List[float] = field(default_factory=lambda: [0.0] * 6)
    afferent_kind: str = "input"


class Organism:
    """A deterministic, rate-based, recurrent organism with local plasticity."""

    VERSION = 12

    def __init__(
        self,
        cells: Mapping[str, Cell],
        graph: SparseDirectedGraph,
        input_ids: Sequence[str],
        output_ids: Sequence[str],
        seed: int = 0,
        resources: Optional[ResourceBudget] = None,
        structural_interval: int = 4,
    ) -> None:
        self.cells: Dict[str, Cell] = dict(cells)
        self.graph = graph
        self.input_ids = list(input_ids)
        self.output_ids = list(output_ids)
        self.seed = seed
        self.rng = random.Random(seed)
        # Action exploration has its own stream so matched benchmark
        # conditions consume the same noise even when structure diverges.
        self.exploration_rng = random.Random(seed + 32452843)
        self.exploration_tape: Optional[Tuple[float, ...]] = None
        self.exploration_cursor = 0
        self.representation_rng = random.Random(seed + 45989)
        self.representation_tape: Optional[Tuple[float, ...]] = None
        self.representation_cursor = 0
        self.resources = resources or ResourceBudget(max_cells=max(64, len(cells)), max_synapses=256)
        self.structural_interval = max(1, structural_interval)
        if not self.input_ids:
            raise ValueError("organism requires at least one sensor/input cell")
        if not self.output_ids:
            raise ValueError("organism requires at least one output cell")
        if len(set(self.input_ids)) != len(self.input_ids) or len(set(self.output_ids)) != len(self.output_ids):
            raise ValueError("input and output IDs must be unique")
        if set(self.input_ids) & set(self.output_ids):
            raise ValueError("input and output IDs must be disjoint")
        if not self.cells:
            raise ValueError("organism requires at least one cell")
        if any(identifier != cell.id for identifier, cell in self.cells.items()):
            raise ValueError("cell mapping keys must equal stable cell IDs")
        for identifier in self.input_ids + self.output_ids:
            if identifier not in self.cells:
                raise ValueError("endpoint ID is missing from cells: %s" % identifier)
        for identifier in self.input_ids:
            if self.cells[identifier].kind != "input":
                raise ValueError("input endpoint has non-input role: %s" % identifier)
        for identifier in self.output_ids:
            if self.cells[identifier].kind != "output":
                raise ValueError("output endpoint has non-output role: %s" % identifier)
        self.step_count = 0
        self.events: List[Dict[str, object]] = []
        self.learning_rate = 0.08
        self.actor_learning_rate = 0.01
        self.representation_learning_enabled = False
        self.representation_learning_rate = 0.0
        self.representation_noise = 0.0
        # Stage-2 graph substrate is opt-in and inert by default.  Binary
        # ``dendritic_product`` remains the frozen v8 representation; order-N
        # cells use a separate activation tag and currently cap at ternary.
        self.variable_order_enabled = False
        self.max_dendritic_order = 2
        self.variable_order_feature_owners: Dict[str, str] = {}
        # R3B: installed product cells feeding input-afferent motor modules.
        # Empty in every legacy regime; consulted only for input-kind owners.
        self.variable_order_extra_sources: Dict[str, List[str]] = {}
        # Sequence-circuit memory: domain-neutral ordered-event prediction
        # owned by the brain lifecycle. None until explicitly enabled; the
        # harness declares the symbol set (brain never sees task meaning).
        self.sequence_memory = None
        self.sequence_memory_symbols: Optional[tuple] = None
        # v11 compositional substrate.  It is inert until explicitly enabled
        # and owns a dedicated RNG so v10 routing/action streams are stable.
        self.compositional_substrate_enabled = False
        self.max_composed_leaves = 4
        self.composition_representation_seed: Optional[int] = None
        self.composition_rng = random.Random(seed + 1181789)
        self.composition_feature_owners: Dict[str, str] = {}
        self.composition_module_features: Dict[str, str] = {}
        self.composition_feature_lineage: Dict[str, Dict[str, object]] = {}
        self.composition_install_count = 0
        self.composition_signal_gain = 1.0
        # Stage-3 learner/router.  Composition is learned only from graph
        # resident direct-pair cells and delayed reward evidence collected
        # during the neutral variable-order fingerprint.  It is deliberately
        # separate from the v10 direct-feature maps for migration/isolation.
        self.composition_learning_enabled = False
        self.composition_fingerprint_evidence: Dict[str, Dict[str, float]] = {}
        self.composition_fingerprint_pending: Optional[Dict[str, object]] = None
        self.composition_direct_route_pending = False
        self.composition_verification_suppressed_until = 0
        self.composition_route_count = 0
        self.composition_candidate_count = 0
        # Adaptive dendritic conversion is an optional local structural rule.
        # It starts with additive hidden cells and persists one proposal until
        # its next reward decides whether the graph edit survives.
        self.adaptive_dendritic_enabled = False
        self.adaptive_dendritic_proposal_interval = 8
        self.adaptive_dendritic_proposal: Optional[Dict[str, object]] = None
        self.adaptive_dendritic_accept_count = 0
        self.adaptive_dendritic_reject_count = 0
        # Structural trials are sequential local evidence, not one-shot
        # rewards.  These are organism learning-rule parameters (never
        # evaluator gates) and are persisted for exact continuation.
        self.adaptive_dendritic_min_evidence = 3
        self.adaptive_dendritic_max_trial_steps = 8
        self.adaptive_dendritic_confidence_z = 1.96
        self.adaptive_dendritic_pairs_tried: Dict[str, List[Tuple[str, str]]] = {}
        self.adaptive_dendritic_module_accepted: Dict[str, bool] = {}
        # Stage-4 shadow scorer: this is separate from the legacy pending
        # structural-trial record above so v8/v9 checkpoint migrations remain
        # readable while the adaptive v9 runner uses the shadow path.
        self.adaptive_dendritic_shadow_enabled = False
        self.adaptive_dendritic_shadow_evidence: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.adaptive_dendritic_shadow_search_exhausted: Dict[str, bool] = {}
        self.adaptive_dendritic_shadow_pending: Optional[Dict[str, object]] = None
        self.adaptive_dendritic_shadow_action_count = 0
        self.adaptive_dendritic_shadow_install_count = 0
        self.adaptive_dendritic_shadow_checkpoints = (16, 32, 64)
        self.adaptive_dendritic_shadow_confidence_z = 1.96
        self.adaptive_dendritic_shadow_separation = 0.05
        # Shadow-mode context routing uses a short, neutral reward fingerprint
        # instead of serial dormant probes.  All state is organism-local and
        # checkpointed so an interrupted fingerprint resumes exactly.
        self.adaptive_dendritic_fingerprint_enabled = False
        self.adaptive_dendritic_fingerprint_window = 16
        self.adaptive_dendritic_fingerprint_active = False
        self.adaptive_dendritic_fingerprint_owner: Optional[str] = None
        self.adaptive_dendritic_fingerprint_incumbent: Optional[str] = None
        self.adaptive_dendritic_fingerprint_start_step = 0
        self.adaptive_dendritic_fingerprint_deadline = 0
        self.adaptive_dendritic_fingerprint_count = 0
        self.adaptive_dendritic_fingerprint_evidence: Dict[str, Dict[str, float]] = {}
        self.adaptive_dendritic_fingerprint_pending: Optional[Dict[str, object]] = None
        self.adaptive_dendritic_pair_owners: Dict[str, str] = {}
        self.adaptive_dendritic_detector_hold_until = 0
        # v10 variable-order learning is deliberately isolated from the v9
        # adaptive fields above.  These fields are inert unless the explicit
        # learner opt-in is called by the v10 benchmark.
        self.variable_order_learning_enabled = False
        self.variable_order_representation_seed: Optional[int] = None
        self.variable_order_rng = random.Random(seed + 735391)
        self.variable_order_feature_order: Tuple[Tuple[str, ...], ...] = ()
        self.variable_order_module_features: Dict[str, str] = {}
        self.variable_order_fingerprint_window = 16
        self.variable_order_fingerprint_active = False
        self.variable_order_fingerprint_owner: Optional[str] = None
        self.variable_order_fingerprint_incumbent: Optional[str] = None
        self.variable_order_fingerprint_start_step = 0
        self.variable_order_fingerprint_deadline = 0
        self.variable_order_fingerprint_count = 0
        self.variable_order_fingerprint_evidence: Dict[str, Dict[str, float]] = {}
        self.variable_order_fingerprint_pending: Optional[Dict[str, object]] = None
        self.variable_order_midpoint_existing_owner_enabled = False
        self.variable_order_midpoint_z_threshold = 3.0
        self.variable_order_midpoint_separation = 0.05
        self.variable_order_owner_evidence_enabled = False
        self.variable_order_owner_posterior: Dict[str, float] = {}
        self.variable_order_owner_evidence_count = 0
        self.variable_order_midpoint_novelty_probability = 0.0
        self.variable_order_temporal_refresh_used = False
        # M4 graph-derived structural substrate. Inert until explicitly
        # enabled, preserving every frozen v8-v12 execution path.
        self.general_structural_learning_enabled = False
        self.general_structural_max_depth = 4
        self.general_structural_max_leaves = 8
        self.general_feature_owners: Dict[str, str] = {}
        self.general_module_features: Dict[str, str] = {}
        self.general_feature_lineage: Dict[str, Dict[str, object]] = {}
        self.general_feature_reuse: Dict[str, int] = {}
        self.general_feature_last_used_step: Dict[str, int] = {}
        self.general_feature_prune_count = 0
        self.general_prune_reuse_ceiling = 0
        self.general_fingerprint_evidence: Dict[str, Dict[str, float]] = {}
        self.general_novelty_candidate: Optional[str] = None
        self.general_novelty_confirmations = 0
        self.general_required_confirmations = 2
        self.general_audit_interval = 128
        self.general_last_audit_step = 0
        self.delayed_credit_enabled = False
        self.delayed_credit_delay = 0
        self.delayed_credit_queue: List[Dict[str, object]] = []
        self.variable_order_owner_probe_enabled = False
        self.variable_order_owner_midpoint_probe_enabled = False
        self.variable_order_owner_midpoint_probe_min_probability = 0.0
        self.variable_order_owner_midpoint_probe_min_margin = 0.0
        self.variable_order_normal_window = 8
        self.variable_order_normal_evidence: Dict[str, Dict[str, float]] = {}
        self.variable_order_normal_pending: Optional[Dict[str, object]] = None
        self.variable_order_normal_count = 0
        self._variable_order_internal_transition = False
        self.variable_order_detector_hold_until = 0
        self.variable_order_install_count = 0
        self.variable_order_route_count = 0
        # Target-independent actor bootstrap for hidden-afferent modules.
        # Zero preserves every legacy constructor/control unless explicitly
        # enabled by a benchmark mode.
        self.motor_bootstrap_scale = 0.0
        self.legacy_learning_enabled = True
        self.trace_decay = 0.85
        self.homeostasis_rate = 0.02
        # Controls are explicit benchmark modes, not hidden changes to the
        # structural schedule.  It is persisted as part of the organism.
        self.structural_plasticity_enabled = True
        self.reward_baseline = 0.0
        self.reward_baseline_rate = 0.03
        self.metrics_history: List[Dict[str, object]] = []
        self._pending_outcome = False
        self.graph.validate(self.cells)
        if self.resources.baseline_cost(len(self.cells), len(self.graph.synapses)) > self.resources.energy_per_step + 1e-12:
            raise ValueError("organism baseline cell+synapse cost exceeds per-step energy budget")
        if self.resources.max_cells < len(self.cells):
            raise ValueError("resource max_cells is smaller than current cell count")
        if self.resources.max_synapses < len(self.graph.synapses):
            raise ValueError("resource max_synapses is smaller than current synapse count")
        self.resources.counters["cells"] = len(self.cells)
        self.resources.counters["synapses"] = len(self.graph.synapses)
        self._coactivity: Dict[Tuple[str, str], float] = {}
        self.context_enabled = False
        self.context_afferent_kind = "input"
        self.context_max_modules = 4
        self.motor_modules: Dict[str, MotorModule] = {}
        self.active_motor_module: Optional[str] = None
        self.pending_motor_module: Optional[str] = None
        self.context_probe_module: Optional[str] = None
        self.context_probe_queue: List[str] = []
        self.context_probe_remaining = 0
        self.context_probe_reference = 0.0
        self.context_probe_failed_reference = 0.0
        self.context_last_reward_value = 0.0
        self.context_probe_rewards: List[float] = []
        self.context_probe_sum = 0.0
        self.context_probe_sumsq = 0.0
        self.context_probe_count = 0
        self.context_probe_steps_seen = 0
        self.context_probe_informative_count = 0
        self.context_probe_candidate_mean = 0.0
        self.context_probe_incumbent_mean = 0.0
        self.context_probe_incumbent_module: Optional[str] = None
        self.context_probe_origin: Optional[str] = None
        self.context_pending_features: Tuple[float, ...] = ()
        self.context_pending_action = 0.0
        self.context_pending_mode = "normal"
        self.context_pending_prediction = 0.0
        self.context_pending_scale = 0.1
        self.context_pending_exploration_value = 0.0
        self.context_pending_exploration_sigma = 0.0
        # Ephemeral causal snapshot: presynaptic values used when each motor
        # module was computed on the final propagation pass.  Actor traces
        # materialize before checkpointing, so snapshot itself need not persist.
        self._motor_causal_presynaptic: Dict[str, Dict[str, float]] = {}
        self.context_state = "normal"
        self.context_warning_threshold = 2.5
        self.context_safe_action = 0.0
        self.context_switch_threshold = 5.0
        self.context_confidence_threshold = 0.35
        self.context_negative_streak = 3
        self.context_probe_steps = 12
        self.context_probe_min_steps = 2
        self.context_probe_max_steps = 16
        # A probe is accepted when its aggregate outcome is strictly better
        # than the incumbent reference; this keeps a genuinely improved
        # dormant engram from being discarded for an arbitrary fixed margin.
        self.context_reactivation_margin = 0.0
        self.context_probe_reward_floor = 0.005
        self.context_surprise_mean = 0.0
        self.context_surprise_variance = 0.01
        self.context_surprise_count = 0
        self.context_surprise_leak = 0.995
        self.context_surprise_drift = 0.05
        self.context_detector_min_evidence = 32
        self.context_detector_freeze_threshold = 2.5
        # Tolerance for standardized negative residuals.  Sustained moderate
        # degradation must accumulate detector evidence, while zero-surprise
        # stationary streams leave the CUSUM at zero.
        self.context_detector_drift = 0.35
        self.context_detector_recovery = 0.20
        self.context_detector_predictor_rate = 0.50
        self.context_probe_information_threshold = 0.05
        self.context_probe_exploration = 0.0
        # R1 calibrated evidence router is the default controller since R1
        # promotion.  `use_threshold_router` retains legacy behavior as a
        # named control.  Evidence state is organism-local and persisted.
        self.evidence_router_mode = "shadow"
        self.evidence_novelty_mass = 0.05
        self.evidence_min_evidence = 16
        self.evidence_enter_margin = 0.15
        self.evidence_exit_margin = 0.05
        self.circuit_evidence: Dict[str, Dict[str, object]] = {}
        self.evidence_posterior: Dict[str, float] = {}
        self.evidence_shadow_steps = 0
        # R1 step 13/17: evidence-driven recall is the default. Growth is
        # legacy-initiated with evidence confirmation (watch + revert/expire).
        self.evidence_decisions_enabled = True
        # R1 steps 14-15: novelty needs absolute poor fit everywhere, not
        # just a relative leader. A recency-weighted mean of the best
        # per-outcome comparative log-likelihood below this threshold
        # justifies structural growth. Lifetime means dilute too slowly;
        # recency-aware evidence is the documented requirement.
        self.evidence_novelty_fit_threshold = -2.0
        self.evidence_recent_best_ll = 0.0
        self.evidence_recent_count = 0
        self.evidence_recent_decay = 0.95
        # R1 step 15: recruited modules are watched, not trusted. Pending
        # growth confirms on sustained comparative lead or reverts at deadline.
        self.evidence_pending_growth = None

    @classmethod
    def create_default(
        cls,
        input_size: int = 2,
        hidden_size: int = 6,
        output_size: int = 1,
        seed: int = 0,
        max_cells: int = 64,
        max_synapses: int = 256,
        energy_per_step: float = 20.0,
    ) -> "Organism":
        cells: Dict[str, Cell] = {}
        input_ids = ["input-%03d" % i for i in range(input_size)]
        hidden_ids = ["hidden-%03d" % i for i in range(hidden_size)]
        output_ids = ["output-%03d" % i for i in range(output_size)]
        for identifier in input_ids:
            cells[identifier] = Cell(identifier, "input", target_activity=0.5)
        for identifier in hidden_ids:
            cells[identifier] = Cell(identifier, "hidden")
        for identifier in output_ids:
            cells[identifier] = Cell(identifier, "output", target_activity=0.5)
        graph = SparseDirectedGraph()
        rng = random.Random(seed)
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if output_size <= 0:
            raise ValueError("output_size must be positive")
        if max_cells < input_size + hidden_size + output_size:
            raise ValueError("max_cells is smaller than the requested initial organism")
        if max_synapses < input_size + output_size:
            raise ValueError("max_synapses must allow one sensor-to-output path per endpoint")
        def add_if_budget_allows(synapse: Synapse) -> None:
            if not graph.has(synapse.source, synapse.destination) and len(graph.synapses) < max_synapses:
                graph.add(synapse)
        # Reserve the minimum sensor-to-output scaffold before exploratory edges.
        for source in input_ids:
            add_if_budget_allows(Synapse(source, hidden_ids[0], 0.15, plasticity=0.8))
        for destination in output_ids:
            add_if_budget_allows(Synapse(hidden_ids[0], destination, 0.15, plasticity=0.7))
        # Seed a sparse scaffold while preserving deterministic IDs and order.
        for source in input_ids:
            for destination in hidden_ids:
                if rng.random() < 0.55:
                    add_if_budget_allows(Synapse(source, destination, rng.uniform(-0.35, 0.35), plasticity=0.8))
        for source in hidden_ids:
            for destination in hidden_ids + output_ids:
                if source != destination and rng.random() < 0.3:
                    add_if_budget_allows(Synapse(source, destination, rng.uniform(-0.35, 0.35), plasticity=0.7))
        if not any(graph.get(source, output_ids[0]) for source in hidden_ids):
            add_if_budget_allows(Synapse(hidden_ids[0], output_ids[0], 0.2, plasticity=0.7))
        budget = ResourceBudget(max_cells=max_cells, max_synapses=max_synapses,
                                energy_per_step=energy_per_step)
        return cls(cells, graph, input_ids, output_ids, seed, budget)

    @classmethod
    def create_dendritic_pair_bank(
        cls,
        input_size: int = 3,
        hidden_size: int = 8,
        output_size: int = 1,
        seed: int = 0,
        max_cells: int = 64,
        max_synapses: int = 256,
    ) -> "Organism":
        """Create a target-independent bank of graph-native product cells.

        Pair cells are an explicit second-order sensory inductive bias, not a
        claim of general representation learning.  Every unordered sensor
        pair is assigned once before deterministic organism-RNG repeats.
        Motor modules are deliberately not installed here; callers can use
        ``enable_context_modules(..., afferent_kind="hidden")`` afterwards.
        """
        organism = cls.create_default(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            seed=seed,
            max_cells=max_cells,
            max_synapses=max_synapses,
        )
        organism.configure_dendritic_pair_bank()
        return organism

    @classmethod
    def create_random_dendritic_pair_bank(
        cls,
        input_size: int = 3,
        hidden_size: int = 8,
        output_size: int = 1,
        seed: int = 0,
        max_cells: int = 64,
        max_synapses: int = 256,
    ) -> "Organism":
        """Create a fixed organism-seeded pair bank without task knowledge."""
        organism = cls.create_default(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            seed=seed,
            max_cells=max_cells,
            max_synapses=max_synapses,
        )
        organism.configure_dendritic_pair_bank(randomize=True)
        return organism

    def configure_dendritic_pair_bank(self, randomize: bool = False) -> Tuple[Tuple[str, str], ...]:
        """Replace hidden scaffold with fixed pairs; random mode samples without replacement.

        If hidden cells outnumber unique input pairs, deterministic seeded repeats
        preserve the requested cell/resource shape after every pair is covered.
        """
        if len(self.input_ids) < 2:
            raise ValueError("dendritic pair bank requires at least two inputs")
        hidden_ids = sorted(identifier for identifier, cell in self.cells.items() if cell.kind == "hidden")
        if not hidden_ids:
            raise ValueError("dendritic pair bank requires hidden cells")
        required_synapses = 2 * len(hidden_ids) + len(self.input_ids) * len(self.output_ids)
        if required_synapses > self.resources.max_synapses:
            raise ValueError("max_synapses is too small for the dendritic pair bank")
        available = list(combinations(self.input_ids, 2))
        if randomize:
            assignments = list(self.rng.sample(available, min(len(hidden_ids), len(available))))
        else:
            assignments = list(available[: min(len(hidden_ids), len(available))])
        while len(assignments) < len(hidden_ids):
            assignments.append(available[self.rng.randrange(len(available))])
        # Keep only pair-cell incoming edges and retain explicit zero sensor
        # to output paths so every sensor remains graph-reachable to output.
        for source, destination in list(self.graph.synapses):
            if destination in hidden_ids or source in hidden_ids:
                self.graph.remove(source, destination)
        for identifier, pair in zip(hidden_ids, assignments):
            cell = self.cells[identifier]
            cell.activation_type = "dendritic_product"
            cell.dendritic_sources = tuple(pair)
            cell.dendritic_normalizer = 1.0
            cell.metadata["feature_protected"] = "direct"
            for source in pair:
                self.graph.add(Synapse(source, identifier, 1.0, plasticity=0.0))
        for source in self.input_ids:
            for destination in self.output_ids:
                if not self.graph.has(source, destination):
                    self.graph.add(Synapse(source, destination, 0.0, plasticity=0.0))
        self.resources.counters["synapses"] = len(self.graph.synapses)
        self.validate()
        return tuple(assignments)

    def enable_variable_order(self, max_order: int = 3) -> None:
        """Enable order-N substrate only; no proposal/evidence rule is attached."""
        if isinstance(max_order, bool) or not isinstance(max_order, int) or max_order < 3 or max_order > 3:
            raise ValueError("variable-order substrate currently supports max_order=3")
        self.variable_order_enabled = True
        self.max_dendritic_order = max_order
        self.validate()

    # Descriptive alias for callers that distinguish substrate from learner.
    enable_variable_order_substrate = enable_variable_order

    def enable_compositional_substrate(self, composition_seed: int) -> None:
        """Enable the inert graph substrate for pair-of-pair features.

        This method only enables graph validation/installation and seeds a
        dedicated representation RNG.  It does not enable an evidence learner
        and never consumes the v10 action, organism, or variable-order RNGs.
        """
        if isinstance(composition_seed, bool) or not isinstance(composition_seed, int):
            raise ValueError("composition_seed must be an integer")
        self.compositional_substrate_enabled = True
        self.max_composed_leaves = 4
        self.composition_representation_seed = int(composition_seed)
        self.composition_rng = random.Random(int(composition_seed))
        self.validate()

    def enable_composition_learning(self) -> None:
        """Enable the target-blind graph-derived pair-of-pairs learner.

        The candidate bank is rebuilt from direct pair cells that are already
        present in the graph.  No sensor/task manifest is copied into the
        organism; composition can therefore only use features that the local
        direct learner has actually acquired.
        """
        if not self.compositional_substrate_enabled:
            raise ValueError("compositional substrate must be enabled before composition learning")
        self.composition_learning_enabled = True
        self.composition_fingerprint_evidence = {}
        self.composition_fingerprint_pending = None
        self.composition_direct_route_pending = False
        self.validate()

    def enable_sequence_memory(self, symbols, max_order=16, max_circuits=131072,
                               min_support=2, prior=0.5) -> None:
        """Enable brain-owned ordered-event prediction over declared symbols.

        Symbols are opaque labels declared by the harness (transducer domain);
        the brain learns their sequential structure and nothing else.  Inert
        until enabled: legacy regimes never carry sequence state.
        """
        from .memory.sequence import SequenceCircuitMemory
        if self.sequence_memory is not None:
            raise ValueError("sequence memory is already enabled")
        memory = SequenceCircuitMemory(tuple(symbols), max_order=max_order,
                                       max_circuits=max_circuits,
                                       min_support=min_support, prior=prior)
        self.sequence_memory = memory
        self.sequence_memory_symbols = tuple(symbols)
        self.validate()

    def observe_sequence_event(self, symbol, learn=True):
        """Record one ordered event in brain-owned sequence memory."""
        if self.sequence_memory is None:
            raise ValueError("sequence memory is not enabled")
        self.sequence_memory.observe(symbol, learn=bool(learn))

    def sequence_distribution(self):
        """Predictive distribution from brain-owned sequence memory."""
        if self.sequence_memory is None:
            raise ValueError("sequence memory is not enabled")
        return self.sequence_memory.distribution()

    def reset_sequence_history(self):
        """Start a new stream; acquired circuits persist (long-term memory)."""
        if self.sequence_memory is None:
            raise ValueError("sequence memory is not enabled")
        self.sequence_memory.reset_history()

    def enable_composition_signal_normalization(self, gain: float) -> None:
        """Compensate bounded activation loss across a composed path."""
        if not self.compositional_substrate_enabled:
            raise ValueError("compositional substrate must be enabled first")
        if not math.isfinite(float(gain)) or not 1.0 <= float(gain) <= 2.0:
            raise ValueError("composition signal gain must be in [1, 2]")
        if self.composition_feature_owners:
            raise ValueError("composition signal gain must be fixed before installation")
        self.composition_signal_gain = float(gain)
        self.validate()

    def _composition_feature_bank(self) -> Tuple[Tuple[str, str, str], ...]:
        """Return canonical graph-derived disjoint direct-pair candidates.

        Entries are ``(key, left_cell, right_cell)``.  Only direct product
        cells with a valid local owner are eligible, so the learner cannot
        compose an evaluator-declared feature or an unowned nursery cell.
        """
        if not self.compositional_substrate_enabled or not self.composition_learning_enabled:
            return ()
        direct = []
        for cell in self.cells.values():
            if cell.kind != "hidden" or cell.activation_type != "dendritic_product":
                continue
            leaves = tuple(str(value) for value in cell.dendritic_sources)
            if len(leaves) != 2 or tuple(sorted(leaves)) != leaves or len(set(leaves)) != 2:
                continue
            try:
                key = self._dendritic_feature_key(leaves)
                owner = self._variable_order_owner_for_key(key, leaves)
                self._composition_source_info(cell.id)
            except (TypeError, ValueError):
                continue
            if owner is not None:
                direct.append((cell.id, leaves))
        candidates = []
        for left_index, (left_id, left_leaves) in enumerate(sorted(direct)):
            for right_id, right_leaves in sorted(direct)[left_index + 1:]:
                if set(left_leaves) & set(right_leaves):
                    continue
                leaves = (left_leaves, right_leaves)
                key = self._composition_feature_key((left_id, right_id), leaves)
                candidates.append((key, left_id, right_id))
        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        self.composition_candidate_count = len(candidates)
        return tuple(candidates)

    def _composition_feature_from_key(self, key: str) -> Tuple[str, str]:
        text = str(key)
        if not text.startswith("c4:") or "::" not in text:
            raise ValueError("composition feature key is malformed")
        sources, leaves = text[3:].split("::", 1)
        source_ids = tuple(sources.split("|"))
        leaf_ids = tuple(leaves.split("|"))
        if len(source_ids) != 2 or len(set(source_ids)) != 2 or len(leaf_ids) != 4 or len(set(leaf_ids)) != 4:
            raise ValueError("composition feature key has invalid arity")
        return source_ids

    def _composition_feature_rank(self, key: str) -> int:
        for index, candidate in enumerate(self._composition_feature_bank()):
            if candidate[0] == key:
                return index
        return len(self._composition_feature_bank()) + 1

    def _composition_owner_for_key(self, key: str) -> Optional[str]:
        owner_id = self.composition_feature_owners.get(key)
        if owner_id not in self.motor_modules:
            return None
        try:
            source_ids = self._composition_feature_from_key(key)
        except ValueError:
            return None
        lineage = self.composition_feature_lineage.get(key, {})
        if tuple(sorted(str(value) for value in lineage.get("source_cells", ()))) != tuple(sorted(source_ids)):
            return None
        for cell in self.cells.values():
            if cell.activation_type == "dendritic_product_composed" and tuple(sorted(cell.dendritic_sources)) == tuple(sorted(source_ids)):
                if self.graph.get(cell.id, self.motor_modules[owner_id].cell_id) is not None:
                    return owner_id
        return None

    def _composition_record_fingerprint_pending(self) -> None:
        if not self.composition_learning_enabled or not self.variable_order_fingerprint_active:
            return
        if self.composition_fingerprint_pending is not None:
            return
        sigma = float(self.context_pending_exploration_sigma)
        xi = float(self.context_pending_exploration_value)
        entries = []
        if sigma > 1e-12:
            factor = xi / sigma
            for key, left_id, right_id in self._composition_feature_bank():
                left = self.cells[left_id].activation
                right = self.cells[right_id].activation
                entries.append({"key": key, "eligibility": math.tanh(left * right) * factor})
        self.composition_fingerprint_pending = {
            "owner_module": self.variable_order_fingerprint_owner,
            "step": self.step_count,
            "exploration_value": xi,
            "exploration_sigma": sigma,
            "eligibilities": entries,
        }

    def _composition_accumulate_fingerprint(self, reward: float) -> None:
        pending = self.composition_fingerprint_pending
        self.composition_fingerprint_pending = None
        if pending is None or not self.composition_learning_enabled:
            return
        raw_reward = float(reward)
        for payload in pending.get("eligibilities", []):
            key = str(payload["key"])
            value = self.composition_fingerprint_evidence.setdefault(key, {"sum": 0.0, "sumsq": 0.0, "count": 0.0})
            credit = raw_reward * float(payload["eligibility"])
            value["sum"] += credit
            value["sumsq"] += credit * credit
            value["count"] += 1.0

    def _composition_route(
        self,
        incumbent_id: Optional[str],
        direct_competitor: Optional[Tuple[float, str]] = None,
    ) -> Optional[Tuple[str, Tuple[object, ...]]]:
        """Route a credible composition winner, returning target/event data."""
        if not self.composition_learning_enabled or not self.composition_fingerprint_evidence:
            return None
        candidates = []
        for key, value in self.composition_fingerprint_evidence.items():
            count = int(value.get("count", 0.0))
            total = float(value.get("sum", 0.0))
            sumsq = float(value.get("sumsq", 0.0))
            if count <= 0:
                continue
            variance = max(0.0, sumsq / float(count) - (total / float(count)) ** 2)
            stderr = math.sqrt(variance / float(count))
            z = total / float(count) / stderr if stderr > 1e-12 else (math.inf if total else 0.0)
            score = abs(total) / math.sqrt(max(1e-12, sumsq))
            candidates.append((score, abs(z), key, total, count))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], -item[1], self._composition_feature_rank(item[2]), item[2]))
        winner = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else 0.0
        # Composition uses the same evidence credibility rule as v10 direct
        # routing.  The joint arbitration below prevents a merely plausible
        # composed candidate from stealing a stronger direct cubic context.
        credible = (
            abs(winner[3]) > 0.0 and winner[1] >= 2.5
            and winner[0] - second_score >= 0.05
        )
        if credible and direct_competitor is not None:
            direct_score, _direct_key = direct_competitor
            # Equal scores are deliberately not credible: the existing v10
            # separation margin must hold across both candidate families.
            credible = winner[0] - direct_score >= 0.05
        if not credible:
            return None
        owner_id = self._composition_owner_for_key(winner[2])
        installed = False
        if owner_id is None and incumbent_id in self.motor_modules:
            if not self.composition_feature_owners and not self.variable_order_module_features:
                owner_id = incumbent_id
            elif len(self.motor_modules) < self.context_max_modules:
                checkpoint = copy.deepcopy(self.__dict__)
                try:
                    owner = self._recruit_motor_module(afferent_kind="hidden")
                    self._variable_order_internal_transition = True
                    try:
                        source_ids = self._composition_feature_from_key(winner[2])
                        self.install_composed_feature(source_ids, owner.id, self._variable_order_credit_strength(winner[3], winner[4]))
                    finally:
                        self._variable_order_internal_transition = False
                    owner_id = owner.id
                    installed = True
                except Exception:
                    self.__dict__.clear()
                    self.__dict__.update(checkpoint)
                    owner_id = None
        if owner_id is None:
            return None
        self.composition_route_count += 1
        return owner_id, (winner[2], winner[3], winner[4], installed, winner[0])

    def _variable_order_best_candidate(self) -> Optional[Tuple[float, str]]:
        """Return the strongest raw v10 score/key without mutating state."""
        candidates = []
        for key, value in self.variable_order_fingerprint_evidence.items():
            count = int(value.get("count", 0.0))
            if count <= 0:
                continue
            total = float(value.get("sum", 0.0))
            sumsq = float(value.get("sumsq", 0.0))
            mean = total / float(count)
            variance = max(0.0, sumsq / float(count) - mean * mean)
            stderr = math.sqrt(variance / float(count))
            z = abs(mean / stderr) if stderr > 1e-12 else (math.inf if abs(mean) > 0.0 else 0.0)
            score = abs(total) / math.sqrt(max(1e-12, sumsq))
            feature = self._dendritic_feature_from_key(key)
            candidates.append((score, abs(z), feature, key, total, count))
        candidates.sort(key=lambda item: (-item[0], -item[1], self._variable_order_feature_rank(item[3])))
        if not candidates:
            return None
        winner = candidates[0]
        # Return the strongest direct competitor even when its own z/separation
        # is insufficient.  Global arbitration must not let two nearly tied
        # families both pass simply because the direct family is noisy.
        return winner[0], winner[3]

    def _variable_order_midpoint_candidates(self) -> Tuple[Tuple[object, ...], ...]:
        """Rank direct/composed midpoint candidates without changing state."""
        candidates = []
        for key, value in self.variable_order_fingerprint_evidence.items():
            count = int(value.get("count", 0.0))
            total = float(value.get("sum", 0.0))
            sumsq = float(value.get("sumsq", 0.0))
            if count <= 0:
                continue
            mean = total / float(count)
            variance = max(0.0, sumsq / float(count) - mean * mean)
            stderr = math.sqrt(variance / float(count))
            z = abs(mean / stderr) if stderr > 1e-12 else (math.inf if abs(mean) > 0.0 else 0.0)
            score = abs(total) / math.sqrt(max(1e-12, sumsq))
            feature = self._dendritic_feature_from_key(key)
            owner = self._variable_order_owner_for_key(key, feature)
            candidates.append(("direct", score, z, self._variable_order_feature_rank(key), key, total, count, owner))
        for key, value in self.composition_fingerprint_evidence.items():
            count = int(value.get("count", 0.0))
            total = float(value.get("sum", 0.0))
            sumsq = float(value.get("sumsq", 0.0))
            if count <= 0:
                continue
            mean = total / float(count)
            variance = max(0.0, sumsq / float(count) - mean * mean)
            stderr = math.sqrt(variance / float(count))
            z = abs(mean / stderr) if stderr > 1e-12 else (math.inf if abs(mean) > 0.0 else 0.0)
            score = abs(total) / math.sqrt(max(1e-12, sumsq))
            owner = self._composition_owner_for_key(key)
            candidates.append(("composition", score, z, self._composition_feature_rank(key), key, total, count, owner))
        candidates.sort(key=lambda item: (-float(item[1]), -float(item[2]), int(item[0] != "direct"), int(item[3]), str(item[4])))
        return tuple(candidates)

    def _variable_order_midpoint_resolve(self) -> bool:
        """Resolve midpoint only to an existing, globally credible owner."""
        if not self.variable_order_midpoint_existing_owner_enabled:
            return False
        midpoint = self.variable_order_fingerprint_window // 2
        if self.variable_order_fingerprint_count != midpoint or midpoint <= 0:
            return False
        candidates = self._variable_order_midpoint_candidates()
        if not candidates:
            return False
        winner = candidates[0]
        family, score, z, _rank, key, total, count, owner_id = winner
        runner_up = next((candidate for candidate in candidates if candidate[0] != family), None)
        family_runner = next((candidate for candidate in candidates[1:] if candidate[0] == family), None)
        family_separation = float(score) - (float(family_runner[1]) if family_runner is not None else 0.0)
        cross_separation = float(score) - (float(runner_up[1]) if runner_up is not None else 0.0)
        credible = (
            abs(float(total)) > 0.0
            and float(z) >= self.variable_order_midpoint_z_threshold
            and family_separation >= self.variable_order_midpoint_separation
            and cross_separation >= self.variable_order_midpoint_separation
            and owner_id in self.motor_modules
        )
        if not credible:
            return False
        previous_id = self.active_motor_module
        self.variable_order_route_count += 1
        for module in self.motor_modules.values():
            module.dormant = module.id != owner_id
            module.protected = False
        target = self.motor_modules[owner_id]
        target.dormant = False
        target.detector_cusum = 0.0
        target.detector_frozen = False
        target.surprise_evidence = 0.0
        target.negative_surprise = 0.0
        target.negative_streak = 0
        self.active_motor_module = owner_id
        self.context_state = "normal"
        self.context_probe_module = None
        self.context_probe_queue = []
        self.context_probe_remaining = 0
        self.context_probe_incumbent_module = None
        self.context_probe_origin = None
        self.variable_order_detector_hold_until = self.step_count + 64
        self.variable_order_fingerprint_active = False
        self.variable_order_fingerprint_owner = None
        self.variable_order_fingerprint_pending = None
        self.variable_order_fingerprint_evidence = {}
        self.variable_order_normal_evidence = {}
        self.variable_order_normal_pending = None
        self.variable_order_normal_count = 0
        self.composition_fingerprint_evidence = {}
        self.composition_fingerprint_pending = None
        direct_route = family == "direct" and bool(self.composition_feature_owners)
        if direct_route and self.step_count < self.composition_verification_suppressed_until:
            self.composition_direct_route_pending = False
            self.composition_verification_suppressed_until = self.variable_order_detector_hold_until
        else:
            self.composition_direct_route_pending = direct_route
        self.events.append({
            "step": self.step_count,
            "kind": "variable_order_midpoint_existing_owner_resolved",
            "family": family,
            "feature": key,
            "owner_module": owner_id,
            "previous_module": previous_id,
            "signed_sum": total,
            "evidence_count": count,
            "score": score,
            "z": z,
            "family_separation": family_separation,
            "cross_family_separation": cross_separation,
            "installed": False,
            "owner_posterior": dict(self.variable_order_owner_posterior),
        })
        if owner_id != previous_id:
            self.events.append({"step": self.step_count, "kind": "motor_module_reactivated", "module_id": owner_id, "fingerprint": True, "midpoint": True})
        return True

    @staticmethod
    def _composition_feature_key(source_cell_ids: Sequence[str], leaf_closures: Optional[Sequence[Sequence[str]]] = None) -> str:
        """Return a canonical key from source feature identities and leaves."""
        sources = tuple(sorted(str(value) for value in source_cell_ids))
        if len(sources) != 2 or len(set(sources)) != 2:
            raise ValueError("composed features require two distinct source cells")
        if leaf_closures is None:
            return "c2:%s" % "|".join(sources)
        leaves = tuple(sorted(tuple(sorted(str(value) for value in closure)) for closure in leaf_closures))
        flattened = tuple(value for closure in leaves for value in closure)
        if len(flattened) != 4 or len(set(flattened)) != 4:
            raise ValueError("composed feature leaves must be four distinct inputs")
        return "c4:%s::%s" % ("|".join(sources), "|".join(flattened))

    def _composition_source_info(self, source_cell_id: str) -> Tuple[Cell, str, Tuple[str, ...]]:
        source_id = str(source_cell_id)
        cell = self.cells.get(source_id)
        if cell is None or cell.kind != "hidden" or cell.activation_type != "dendritic_product":
            raise ValueError("composed sources must be learned direct binary product cells")
        leaves = tuple(str(value) for value in cell.dendritic_sources)
        if len(leaves) != 2 or tuple(sorted(leaves)) != leaves or len(set(leaves)) != 2 or any(value not in self.input_ids for value in leaves):
            raise ValueError("composed source cell must be a canonical direct pair")
        if set(self.graph.incoming.get(source_id, ())) != set(leaves) or any(
            self.graph.get(value, source_id) is None or self.graph.get(value, source_id).transmission_delay != 0
            for value in leaves
        ):
            raise ValueError("composed source pair must have exactly two zero-delay input afferents")
        key = self._dendritic_feature_key(leaves)
        owner_id = self.variable_order_feature_owners.get(key)
        if owner_id is None:
            owner_id = self.adaptive_dendritic_pair_owners.get(self._adaptive_dendritic_pair_key(leaves))
        if owner_id not in self.motor_modules:
            raise ValueError("composed source pair must have an existing owner")
        owner = self.motor_modules[owner_id]
        if owner.afferent_kind != "hidden" or self.graph.get(source_id, owner.cell_id) is None:
            raise ValueError("composed source pair owner must be a hidden-afferent motor module")
        return cell, owner_id, leaves

    def _protected_feature_cell_ids(self) -> set:
        protected = {
            cell.id for cell in self.cells.values()
            if cell.metadata.get("feature_protected") in ("direct", "composed")
        }
        for key in self.variable_order_feature_owners:
            try:
                feature = self._dendritic_feature_from_key(key)
            except (TypeError, ValueError):
                continue
            for cell in self.cells.values():
                if tuple(cell.dendritic_sources) == feature and cell.activation_type in ("dendritic_product", "dendritic_product_n"):
                    protected.add(cell.id)
        for key, lineage in self.composition_feature_lineage.items():
            for source_id in lineage.get("source_cells", ()):
                if str(source_id) in self.cells:
                    protected.add(str(source_id))
            for cell in self.cells.values():
                if cell.metadata.get("composition_key") == key:
                    protected.add(cell.id)
        for lineage in self.general_feature_lineage.values():
            if str(lineage.get("cell_id", "")) in self.cells:
                protected.add(str(lineage["cell_id"]))
        return protected

    def enable_general_structural_learning(self, max_depth: int = 4, max_leaves: int = 8) -> None:
        """Enable bounded feature construction from current graph paths."""
        if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 1:
            raise ValueError("general structural max depth must be a positive integer")
        if isinstance(max_leaves, bool) or not isinstance(max_leaves, int) or max_leaves < 2:
            raise ValueError("general structural max leaves must be at least two")
        self.general_structural_learning_enabled = True
        self.general_structural_max_depth = max_depth
        self.general_structural_max_leaves = max_leaves
        # Higher-order residuals are weaker than direct parity residuals; this
        # opt-in ceiling keeps sustained mismatch observable without changing
        # the frozen router when M4 is disabled.
        self.context_detector_drift = min(self.context_detector_drift, 0.25)
        self.validate()

    def set_general_prune_reuse_ceiling(self, ceiling: int) -> None:
        """Allow bounded reclamation of only weakly reused dormant features."""
        if isinstance(ceiling, bool) or not isinstance(ceiling, int) or ceiling < 0:
            raise ValueError("general prune reuse ceiling must be a nonnegative integer")
        self.general_prune_reuse_ceiling = ceiling
        self.validate()

    def enable_delayed_credit(self, delay: int) -> None:
        """Associate arriving rewards with exact persisted prior action state."""
        if isinstance(delay, bool) or not isinstance(delay, int) or delay < 1 or delay > 32:
            raise ValueError("delayed credit delay must be an integer in [1, 32]")
        if self._pending_outcome or self.delayed_credit_queue:
            raise ValueError("delayed credit must be enabled before the first action")
        self.delayed_credit_enabled = True
        self.delayed_credit_delay = delay
        if self.variable_order_fingerprint_active:
            self.variable_order_fingerprint_deadline += delay
        if self.adaptive_dendritic_fingerprint_active:
            self.adaptive_dendritic_fingerprint_deadline += delay
        self.validate()

    def _delayed_credit_snapshot(self) -> Dict[str, object]:
        return {
            "pending_motor_module": self.pending_motor_module,
            "context_pending_mode": self.context_pending_mode,
            "context_pending_features": list(self.context_pending_features),
            "context_pending_action": self.context_pending_action,
            "context_pending_prediction": self.context_pending_prediction,
            "context_pending_scale": self.context_pending_scale,
            "context_pending_exploration_value": self.context_pending_exploration_value,
            "context_pending_exploration_sigma": self.context_pending_exploration_sigma,
            "variable_order_fingerprint_pending": copy.deepcopy(self.variable_order_fingerprint_pending),
            "composition_fingerprint_pending": copy.deepcopy(self.composition_fingerprint_pending),
            "variable_order_normal_pending": copy.deepcopy(self.variable_order_normal_pending),
            "adaptive_dendritic_fingerprint_pending": copy.deepcopy(self.adaptive_dendritic_fingerprint_pending),
            "adaptive_dendritic_shadow_pending": copy.deepcopy(self.adaptive_dendritic_shadow_pending),
            "actor_traces": {edge.id: edge.actor_eligibility_trace for edge in self.graph.iter_synapses() if abs(edge.actor_eligibility_trace) > 0.0},
        }

    def _delayed_credit_clear_live_pending(self) -> None:
        self.variable_order_fingerprint_pending = None
        self.composition_fingerprint_pending = None
        self.variable_order_normal_pending = None
        self.adaptive_dendritic_fingerprint_pending = None
        self.adaptive_dendritic_shadow_pending = None
        for edge in self.graph.iter_synapses():
            edge.actor_eligibility_trace = 0.0
        self._pending_outcome = False

    def _delayed_credit_restore(self, snapshot: Mapping[str, object]) -> None:
        self.pending_motor_module = snapshot.get("pending_motor_module")
        self.context_pending_mode = str(snapshot.get("context_pending_mode", "normal"))
        self.context_pending_features = tuple(float(value) for value in snapshot.get("context_pending_features", ()))
        self.context_pending_action = float(snapshot.get("context_pending_action", 0.0))
        self.context_pending_prediction = float(snapshot.get("context_pending_prediction", 0.0))
        self.context_pending_scale = float(snapshot.get("context_pending_scale", 0.1))
        self.context_pending_exploration_value = float(snapshot.get("context_pending_exploration_value", 0.0))
        self.context_pending_exploration_sigma = float(snapshot.get("context_pending_exploration_sigma", 0.0))
        self.variable_order_fingerprint_pending = copy.deepcopy(snapshot.get("variable_order_fingerprint_pending"))
        self.composition_fingerprint_pending = copy.deepcopy(snapshot.get("composition_fingerprint_pending"))
        self.variable_order_normal_pending = copy.deepcopy(snapshot.get("variable_order_normal_pending"))
        self.adaptive_dendritic_fingerprint_pending = copy.deepcopy(snapshot.get("adaptive_dendritic_fingerprint_pending"))
        self.adaptive_dendritic_shadow_pending = copy.deepcopy(snapshot.get("adaptive_dendritic_shadow_pending"))
        traces = dict(snapshot.get("actor_traces", {}))
        for edge in self.graph.iter_synapses():
            edge.actor_eligibility_trace = float(traces.get(edge.id, 0.0))
        self._pending_outcome = True

    def _general_source_lineage(self, source_id: str) -> Optional[Tuple[Tuple[str, ...], int]]:
        source_id = str(source_id)
        if source_id in self.input_ids:
            return ((source_id,), 0)
        cell = self.cells.get(source_id)
        if cell is None or cell.kind != "hidden" or cell.activation_type not in (
            "dendritic_product", "dendritic_product_n", "dendritic_product_composed"
        ):
            return None
        leaves: List[str] = []
        depth = 0
        for source in cell.dendritic_sources:
            lineage = self._general_source_lineage(source)
            if lineage is None:
                return None
            leaves.extend(lineage[0])
            depth = max(depth, lineage[1] + 1)
        if len(set(leaves)) != len(leaves):
            return None
        return (tuple(sorted(leaves)), depth)

    def general_feature_candidates(self) -> Tuple[Dict[str, object], ...]:
        """Enumerate bounded proposals solely from graph-visible sources."""
        if not self.general_structural_learning_enabled:
            return ()
        sources = list(self.input_ids) + sorted(
            identifier for identifier, cell in self.cells.items()
            if cell.kind == "hidden" and cell.activation_type != "additive"
        )
        candidates = []
        for left, right in combinations(sources, 2):
            left_lineage = self._general_source_lineage(left)
            right_lineage = self._general_source_lineage(right)
            if left_lineage is None or right_lineage is None:
                continue
            leaves = tuple(sorted(left_lineage[0] + right_lineage[0]))
            depth = max(left_lineage[1], right_lineage[1]) + 1
            if len(set(leaves)) != len(leaves) or len(leaves) > self.general_structural_max_leaves or depth > self.general_structural_max_depth:
                continue
            source_pair = tuple(sorted((left, right)))
            key = "g%d:%s::%s" % (depth, "|".join(source_pair), "|".join(leaves))
            candidates.append({"key": key, "sources": source_pair, "leaves": leaves, "depth": depth})
        return tuple(sorted(candidates, key=lambda item: str(item["key"])))

    def install_general_feature(
        self, source_cell_ids: Sequence[str], owner_module_id: str,
        signed_local_credit: float, cell_id: Optional[str] = None,
    ) -> str:
        """Atomically install one graph-derived product with reversible lineage."""
        if not self.general_structural_learning_enabled:
            raise ValueError("general structural learning is disabled")
        sources = tuple(sorted(str(value) for value in source_cell_ids))
        if len(sources) != 2 or len(set(sources)) != 2:
            raise ValueError("general feature requires two distinct graph sources")
        candidate = next((item for item in self.general_feature_candidates() if item["sources"] == sources), None)
        if candidate is None:
            raise ValueError("sources are not a bounded graph-derived candidate")
        if not math.isfinite(float(signed_local_credit)) or not -1.0 <= float(signed_local_credit) <= 1.0:
            raise ValueError("signed local credit must be finite and in [-1, 1]")
        owner_id = str(owner_module_id)
        owner = self.motor_modules.get(owner_id)
        if owner is None or owner.afferent_kind != "hidden" or owner_id in self.general_module_features:
            raise ValueError("general feature owner must be an available hidden-afferent module")
        nursery_ids = [str(cell_id)] if cell_id is not None else sorted(
            identifier for identifier, cell in self.cells.items()
            if cell.kind == "hidden" and cell.activation_type == "additive"
        )
        if not nursery_ids or nursery_ids[0] not in self.cells:
            raise ValueError("general feature requires an additive nursery cell")
        nursery_id = nursery_ids[0]
        nursery = self.cells[nursery_id]
        if nursery.kind != "hidden" or nursery.activation_type != "additive":
            raise ValueError("general feature cannot overwrite learned structure")
        checkpoint = copy.deepcopy(self.__dict__)
        old_cell = copy.deepcopy(nursery.__dict__)
        old_incoming = [synapse_to_dict(edge) for edge in self.graph.iter_synapses() if edge.destination == nursery_id]
        motor_afferents = [
            synapse_to_dict(edge) for edge in self.graph.iter_synapses()
            if edge.source == nursery_id and self.cells[edge.destination].kind == "motor_module"
        ]
        owner_edge = self.graph.get(nursery_id, owner.cell_id)
        try:
            for payload in old_incoming:
                self.graph.remove(str(payload["source"]), nursery_id)
                self.resources.removed_synapse()
            nursery.activation_type = "dendritic_product_composed"
            nursery.dendritic_sources = sources
            # Nested bounded products attenuate their source magnitude. Use
            # the same fixed local compensation validated for v12 composed
            # paths; this depends on graph depth, never task identity.
            nursery.dendritic_normalizer = 1.72
            nursery.metadata["feature_protected"] = "general"
            nursery.metadata["general_feature_key"] = str(candidate["key"])
            for source in sources:
                if not self.resources.can_add_synapse():
                    raise RuntimeError("general feature exceeds synapse budget")
                self.graph.add(Synapse(source, nursery_id, 1.0, plasticity=0.0, stability=1.0))
                self.resources.added_synapse()
            for payload in motor_afferents:
                edge = self.graph.get(nursery_id, str(payload["destination"]))
                if edge is not None:
                    edge.strength = float(signed_local_credit) if edge.destination == owner.cell_id else 0.0
                    edge.plasticity = 0.0
                    edge.stability = 1.0
            added_owner_edge = owner_edge is None
            if added_owner_edge:
                if not self.resources.can_add_synapse():
                    raise RuntimeError("general feature owner edge exceeds synapse budget")
                self.graph.add(Synapse(nursery_id, owner.cell_id, float(signed_local_credit), plasticity=0.0, stability=1.0))
                self.resources.added_synapse()
            reachability_synapses = []
            for source in self.input_ids:
                reachable = {source}
                pending = [source]
                while pending:
                    current = pending.pop(0)
                    for destination in self.graph.outgoing.get(current, ()):
                        if destination not in reachable:
                            reachable.add(destination)
                            pending.append(destination)
                for output in self.output_ids:
                    if output in reachable or self.graph.has(source, output):
                        continue
                    if not self.resources.can_add_synapse():
                        raise RuntimeError("general feature reachability scaffold exceeds budget")
                    self.graph.add(Synapse(source, output, 0.0, plasticity=0.0, stability=1.0))
                    self.resources.added_synapse()
                    reachability_synapses.append(synapse_to_dict(self.graph.get(source, output)))
            key = str(candidate["key"])
            self.general_feature_owners[key] = owner_id
            self.general_module_features[owner_id] = key
            self.general_feature_reuse[key] = 0
            self.general_feature_last_used_step[key] = self.step_count
            self.general_feature_lineage[key] = {
                "cell_id": nursery_id, "source_cells": list(sources),
                "leaf_closure": list(candidate["leaves"]), "depth": int(candidate["depth"]),
                "old_cell": old_cell, "old_incoming": old_incoming,
                "motor_afferents": motor_afferents, "added_owner_edge": added_owner_edge,
                "reachability_synapses": reachability_synapses,
            }
            self.resources.counters["synapses"] = len(self.graph.synapses)
            self.validate()
            self.events.append({"step": self.step_count, "kind": "general_feature_installed", "feature_key": key, "cell_id": nursery_id, "owner_module": owner_id, "sources": list(sources), "leaves": list(candidate["leaves"]), "depth": int(candidate["depth"])})
            return nursery_id
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(checkpoint)
            raise

    def mark_general_feature_reused(self, feature_key: str) -> None:
        key = str(feature_key)
        if key not in self.general_feature_owners:
            raise ValueError("unknown general feature")
        self.general_feature_reuse[key] = self.general_feature_reuse.get(key, 0) + 1
        self.events.append({"step": self.step_count, "kind": "general_feature_reused", "feature_key": key, "reuse_count": self.general_feature_reuse[key]})

    def prune_general_feature(self, feature_key: str) -> bool:
        """Recover a redundant feature cell and its owner slot atomically."""
        key = str(feature_key)
        owner_id = self.general_feature_owners.get(key)
        lineage = self.general_feature_lineage.get(key)
        if owner_id is None or lineage is None or owner_id == self.active_motor_module:
            return False
        if any(
            str(lineage.get("cell_id")) in tuple(str(value) for value in other.get("source_cells", ()))
            for other_key, other in self.general_feature_lineage.items() if other_key != key
        ):
            return False
        owner = self.motor_modules.get(owner_id)
        if owner is None or owner.protected or not owner.dormant:
            return False
        checkpoint = copy.deepcopy(self.__dict__)
        try:
            cell_id = str(lineage["cell_id"])
            for source in list(self.graph.incoming.get(cell_id, ())):
                self.graph.remove(source, cell_id)
                self.resources.removed_synapse()
            if bool(lineage.get("added_owner_edge")) and self.graph.get(cell_id, owner.cell_id) is not None:
                self.graph.remove(cell_id, owner.cell_id)
                self.resources.removed_synapse()
            for payload in lineage.get("reachability_synapses", []):
                edge = self.graph.get(str(payload["source"]), str(payload["destination"]))
                if edge is not None:
                    self.graph.remove(edge.source, edge.destination)
                    self.resources.removed_synapse()
            old_cell = copy.deepcopy(dict(lineage["old_cell"]))
            if isinstance(old_cell.get("dendritic_sources"), list):
                old_cell["dendritic_sources"] = tuple(old_cell["dendritic_sources"])
            self.cells[cell_id].__dict__.clear()
            self.cells[cell_id].__dict__.update(old_cell)
            for payload in lineage.get("old_incoming", []):
                self._adaptive_dendritic_restore_synapse(payload)
                self.resources.restored_synapse()
            for payload in lineage.get("motor_afferents", []):
                edge = self.graph.get(str(payload["source"]), str(payload["destination"]))
                if edge is not None:
                    edge.__dict__.update(Synapse(**dict(payload)).__dict__)
            del self.general_feature_owners[key]
            del self.general_module_features[owner_id]
            del self.general_feature_lineage[key]
            self.general_feature_reuse.pop(key, None)
            self.general_feature_last_used_step.pop(key, None)
            self.general_feature_prune_count += 1
            self.resources.counters["synapses"] = len(self.graph.synapses)
            self.validate()
            self.events.append({"step": self.step_count, "kind": "general_feature_pruned", "feature_key": key, "cell_id": cell_id, "owner_module": owner_id, "capacity_recovered": True})
            return True
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(checkpoint)
            raise

    def install_composed_feature(
        self,
        source_cell_ids: Sequence[str],
        owner_module_id: str,
        signed_local_credit: float,
        cell_id: Optional[str] = None,
    ) -> str:
        """Atomically install a target-blind pair-of-pairs feature cell."""
        if not self.compositional_substrate_enabled:
            raise ValueError("compositional substrate is disabled")
        if len(tuple(source_cell_ids)) != 2:
            raise ValueError("composed feature requires exactly two source cells")
        if not math.isfinite(float(signed_local_credit)) or not -1.0 <= float(signed_local_credit) <= 1.0:
            raise ValueError("signed local credit must be finite and in [-1, 1]")
        source_ids = tuple(sorted(str(value) for value in source_cell_ids))
        if len(set(source_ids)) != 2:
            raise ValueError("composed source cells must be distinct")
        source_info = [self._composition_source_info(source_id) for source_id in source_ids]
        leaves = [info[2] for info in source_info]
        if set(leaves[0]) & set(leaves[1]) or len(set(leaves[0] + leaves[1])) != self.max_composed_leaves:
            raise ValueError("composed source leaf closures must be disjoint and cover four inputs")
        key = self._composition_feature_key(source_ids, leaves)
        if key in self.composition_feature_owners or key in self.composition_feature_lineage:
            raise ValueError("composed feature is already owned")
        owner_id = str(owner_module_id)
        owner = self.motor_modules.get(owner_id)
        if owner is None or owner.afferent_kind != "hidden" or owner.cell_id not in self.cells:
            raise ValueError("composed owner must be an existing hidden-afferent motor module")
        if owner_id in self.variable_order_module_features or owner_id in self.variable_order_feature_owners.values() or owner_id in self.composition_module_features or owner_id in self.adaptive_dendritic_pair_owners.values():
            raise ValueError("one direct or composed learned feature per module is required")
        candidate_ids = [str(cell_id)] if cell_id is not None else sorted(
            identifier for identifier, cell in self.cells.items()
            if cell.kind == "hidden" and cell.activation_type == "additive"
        )
        if not candidate_ids or candidate_ids[0] not in self.cells:
            raise ValueError("composed install requires an additive hidden nursery cell")
        nursery_id = candidate_ids[0]
        nursery = self.cells[nursery_id]
        if nursery.kind != "hidden" or nursery.activation_type != "additive":
            raise ValueError("composed install cannot overwrite an accepted feature cell")
        old_incoming = [synapse_to_dict(synapse) for synapse in self.graph.iter_synapses() if synapse.destination == nursery_id]
        candidate_motor_edges = [
            synapse_to_dict(synapse) for synapse in self.graph.iter_synapses()
            if synapse.source == nursery_id and self.cells[synapse.destination].kind == "motor_module"
        ]
        owner_edge = self.graph.get(nursery_id, owner.cell_id)
        add_owner_edge = owner_edge is None
        prospective_synapses = len(self.graph.synapses) - len(old_incoming) + 2 + int(add_owner_edge)
        prospective_energy = self.resources.energy_used + (2 + int(add_owner_edge)) * self.resources.synapse_cost
        if prospective_synapses > self.resources.max_synapses or prospective_energy > self.resources.energy_per_step + 1e-12:
            raise RuntimeError("composed feature install exceeds synapse or energy budget")
        snapshot = copy.deepcopy(self.__dict__)
        try:
            for payload in old_incoming:
                self.graph.remove(str(payload["source"]), str(payload["destination"]))
                self.resources.removed_synapse()
            nursery.activation_type = "dendritic_product_composed"
            nursery.dendritic_sources = source_ids
            nursery.dendritic_normalizer = self.composition_signal_gain
            nursery.metadata["feature_protected"] = "composed"
            nursery.metadata["composition_key"] = key
            for source_id in source_ids:
                self.graph.add(Synapse(source_id, nursery_id, 1.0, plasticity=0.0, stability=1.0))
                self.resources.added_synapse()
            owner_strength = max(-1.0, min(1.0, float(signed_local_credit)))
            for payload in candidate_motor_edges:
                current = self.graph.get(str(payload["source"]), str(payload["destination"]))
                if current is None:
                    continue
                current.strength = owner_strength if current.destination == owner.cell_id else 0.0
                current.plasticity = 0.0
                current.actor_eligibility_trace = 0.0
                current.stability = 1.0
            if owner_edge is None:
                self.graph.add(Synapse(nursery_id, owner.cell_id, owner_strength, plasticity=0.0, stability=1.0))
                self.resources.added_synapse()
            else:
                owner_edge.strength = owner_strength
                owner_edge.plasticity = 0.0
                owner_edge.actor_eligibility_trace = 0.0
                owner_edge.stability = 1.0
            # Replacing a nursery cell can remove the only path from a sensor
            # to the output.  Preserve the graph invariant with neutral
            # sensor-to-output scaffolds, charged through the same resource
            # budget and kept distinct from forbidden sensor-to-motor edges.
            for source in self.input_ids:
                reachable = {source}
                pending = [source]
                while pending:
                    current = pending.pop(0)
                    for destination in self.graph.outgoing.get(current, ()):
                        if destination not in reachable:
                            reachable.add(destination)
                            pending.append(destination)
                for output in self.output_ids:
                    if output in reachable or self.graph.has(source, output):
                        continue
                    if not self.resources.can_add_synapse():
                        raise RuntimeError("composed feature reachability scaffold exceeds resource budget")
                    self.graph.add(Synapse(source, output, 0.0, plasticity=0.0, stability=1.0))
                    self.resources.added_synapse()
            self.resources.counters["synapses"] = len(self.graph.synapses)
            self.graph.validate(self.cells)
            self.composition_feature_owners[key] = owner_id
            self.composition_module_features[owner_id] = key
            self.composition_feature_lineage[key] = {
                "source_cells": list(source_ids),
                "leaf_closure": [list(leaf) for leaf in leaves],
            }
            self.composition_install_count += 1
            self.validate()
            self.events.append({
                "step": self.step_count,
                "kind": "composed_feature_installed",
                "cell_id": nursery_id,
                "source_cells": list(source_ids),
                "leaf_closure": [list(leaf) for leaf in leaves],
                "owner_module": owner_id,
                "strength": owner_strength,
                "feature_key": key,
            })
            return nursery_id
        except Exception:
            self.__dict__.clear()
            self.__dict__.update(snapshot)
            raise

    install_compositional_feature = install_composed_feature

    def enable_variable_order_learning(self, representation_seed: int, max_order: int = 3) -> None:
        """Enable the target-blind v10 local variable-order router."""
        self.enable_variable_order(max_order)
        if isinstance(representation_seed, bool) or not isinstance(representation_seed, int):
            raise ValueError("representation_seed must be an integer")
        self.variable_order_learning_enabled = True
        self.variable_order_representation_seed = int(representation_seed)
        self.variable_order_rng = random.Random(int(representation_seed))
        features = [tuple(str(source) for source in pair) for pair in combinations(self.input_ids, 2)]
        if max_order >= 3:
            features.extend(tuple(str(source) for source in triple) for triple in combinations(self.input_ids, 3))
        self.variable_order_rng.shuffle(features)
        self.variable_order_feature_order = tuple(features)
        self.variable_order_fingerprint_window = 16
        self.variable_order_detector_hold_until = 0
        # v10 gets a short local detector horizon; v9's opt-in fields remain
        # untouched because this method never enables the v9 shadow path.
        self.context_detector_drift = 0.65
        self.context_detector_min_evidence = 8
        self.context_warning_threshold = 1.5
        self.context_switch_threshold = 3.0
        if self.context_enabled and self.active_motor_module is not None:
            self._variable_order_begin_fingerprint()
        self.validate()

    def set_variable_order_fingerprint_window(self, window: int) -> None:
        """Configure probe evidence length without leaving an active probe stale."""
        if isinstance(window, bool) or not isinstance(window, int) or window < 2:
            raise ValueError("variable-order fingerprint window must be an integer of at least two")
        if self.variable_order_fingerprint_active and self.variable_order_fingerprint_count:
            raise ValueError("cannot resize a variable-order fingerprint after evidence arrives")
        self.variable_order_fingerprint_window = window
        if self.variable_order_fingerprint_active:
            self.variable_order_fingerprint_deadline = self.variable_order_fingerprint_start_step + window + (self.delayed_credit_delay if self.delayed_credit_enabled else 0)
        self.validate()

    def enable_variable_order_midpoint_existing_owner_resolution(
        self,
        z_threshold: float = 3.0,
        separation: float = 0.05,
    ) -> None:
        """Opt into target-blind midpoint confirmation of existing owners.

        This never recruits or installs.  It only permits an exactly half-way
        neutral fingerprint to resolve to a graph-proven existing direct or
        composed owner when both evidence families are decisively separated.
        """
        if not self.variable_order_learning_enabled:
            raise ValueError("variable-order learning must be enabled first")
        if not math.isfinite(float(z_threshold)) or float(z_threshold) < 0.0:
            raise ValueError("midpoint z threshold must be finite and nonnegative")
        if not math.isfinite(float(separation)) or float(separation) < 0.0:
            raise ValueError("midpoint separation must be finite and nonnegative")
        self.variable_order_midpoint_existing_owner_enabled = True
        self.variable_order_midpoint_z_threshold = float(z_threshold)
        self.variable_order_midpoint_separation = float(separation)
        self.validate()

    def enable_evidence_router(self, mode="shadow"):
        """Opt into R1 evidence tracking. Modes: shadow (default since R1)."""
        if mode != "shadow":
            raise ValueError("only shadow mode is implemented")
        self.evidence_router_mode = "shadow"
        self.validate()

    def use_threshold_router(self):
        """Named control: legacy threshold routing with no evidence decisions."""
        self.evidence_router_mode = "threshold"
        self.evidence_decisions_enabled = False
        self.validate()

    def _evidence_conditional_likelihood(self, module, reward_value, features):
        """Clipped outcome likelihood under a circuit's conditional predictor."""
        from .routing.evidence import LLR_CLIP, student_t_logpdf
        import math as _math
        predictor = module.detector_predictor
        if features and len(predictor) == len(features):
            prediction = sum(weight * value for weight, value in zip(predictor, features))
        else:
            prediction = module.detector_mean
        scale = _math.sqrt(max(1e-5, module.detector_variance))
        likelihood = student_t_logpdf(float(reward_value), prediction, scale)
        return max(-LLR_CLIP, min(LLR_CLIP, likelihood))

    def _evidence_shadow_update(self, module_id, reward_value, from_probe=False):
        from .routing.evidence import (
            CircuitEvidence,
            normalize_posterior,
            update_evidence,
        )
        if self.evidence_router_mode not in ("shadow", "evidence"):
            return
        # Comparative CONDITIONAL scoring (Section 6.2): every outcome is
        # evaluated under each circuit's detector predictor given the shared
        # observation/action features. Marginal mean/var stay diagnostic only.
        # Probe outcomes accumulate into protected tallies with models frozen.
        # Marginal stats update only while settled in normal operation,
        # mirroring the legacy detector freeze.
        learn = (
            not from_probe
            and self.context_pending_mode == "normal"
            and self.context_state == "normal"
            and self.context_probe_module is None
        )
        features = self.context_pending_features or self._context_feature_values(self.context_pending_action)
        trial_pair = (module_id, self.context_probe_incumbent_module)
        best_likelihood = None
        current_likelihoods = {}
        for candidate_id, module in self.motor_modules.items():
            payload = self.circuit_evidence.get(candidate_id)
            item = CircuitEvidence.from_dict(payload) if payload is not None else CircuitEvidence(candidate_id)
            likelihood = self._evidence_conditional_likelihood(module, float(reward_value), features)
            best_likelihood = likelihood if best_likelihood is None else max(best_likelihood, likelihood)
            current_likelihoods[candidate_id] = likelihood
            if from_probe:
                if candidate_id in trial_pair:
                    item.probe_log_evidence += likelihood
                    item.probe_scored += 1
            else:
                item.log_evidence += likelihood
                item.scored_outcomes += 1
                if candidate_id == module_id:
                    if learn:
                        update_evidence(item, float(reward_value), from_probe=False, step=self.step_count, accumulate=False)
                    else:
                        item.normal_outcomes += 1
                        item.last_used_step = self.step_count
                        bin_index = min(9, max(0, int(item.posterior * 10.0)))
                        item.calibration_bins[bin_index] += 1
            self.circuit_evidence[candidate_id] = item.to_dict()
        scores = {key: float(value.get("log_evidence", 0.0)) for key, value in self.circuit_evidence.items()}
        self.evidence_posterior = normalize_posterior(scores, novelty_mass=self.evidence_novelty_mass)
        self.evidence_shadow_steps += 1
        if best_likelihood is not None:
            decay = float(self.evidence_recent_decay)
            self.evidence_recent_best_ll = decay * self.evidence_recent_best_ll + (1.0 - decay) * float(best_likelihood)
            self.evidence_recent_count += 1
        pending_watch = self.evidence_pending_growth
        if (
            pending_watch is not None
            and not from_probe
            and pending_watch.get("module_id") in current_likelihoods
            and pending_watch.get("incumbent_id") in current_likelihoods
        ):
            edge = current_likelihoods[pending_watch["module_id"]] - current_likelihoods[pending_watch["incumbent_id"]]
            decay = float(self.evidence_recent_decay)
            pending_watch["edge_ema"] = decay * float(pending_watch.get("edge_ema", 0.0)) + (1.0 - decay) * float(edge)
            pending_watch["edge_n"] = int(pending_watch.get("edge_n", 0)) + 1
        self._evidence_check_growth_watch()
        ordered = sorted(self.evidence_posterior.items(), key=lambda entry: (-entry[1], entry[0]))
        self.events.append({
            "step": self.step_count,
            "kind": "evidence_shadow",
            "module_id": module_id,
            "from_probe": bool(from_probe),
            "leader": ordered[0][0],
            "leader_posterior": ordered[0][1],
            "legacy_active": self.active_motor_module,
            "legacy_state": self.context_state,
        })

    def enable_evidence_decisions(self, enabled=True):
        """Opt into evidence-driven recall. Growth remains legacy-gated."""
        if not isinstance(enabled, bool):
            raise ValueError("evidence decisions flag must be boolean")
        if enabled and self.evidence_router_mode not in ("shadow", "evidence"):
            raise ValueError("enable shadow tracking before evidence decisions")
        self.evidence_decisions_enabled = enabled
        self.validate()

    def _evidence_recall_candidate(self):
        """Return dormant owner justified by calibrated evidence, else None."""
        from .routing.evidence import CircuitEvidence, should_switch
        if not self.evidence_decisions_enabled or not self.evidence_posterior:
            return None
        ordered = sorted(self.evidence_posterior.items(), key=lambda entry: (-entry[1], entry[0]))
        leader, leader_post = ordered[0]
        if leader == "__novelty__" or leader not in self.motor_modules:
            return None
        incumbent_id = self.active_motor_module
        if incumbent_id not in self.motor_modules:
            return None
        leader_item = CircuitEvidence.from_dict(self.circuit_evidence.get(
            leader, CircuitEvidence(leader).to_dict()))
        incumbent_item = CircuitEvidence.from_dict(self.circuit_evidence.get(
            incumbent_id, CircuitEvidence(incumbent_id).to_dict()))
        if should_switch(leader_post, self.evidence_posterior.get(incumbent_id, 0.0),
                          leader_item.normal_outcomes + leader_item.probe_outcomes,
                          incumbent_item.normal_outcomes + incumbent_item.probe_outcomes,
                          min_evidence=self.evidence_min_evidence,
                          enter_margin=self.evidence_enter_margin,
                          exit_margin=self.evidence_exit_margin):
            candidate = self.motor_modules[leader]
            if candidate.dormant and leader != incumbent_id:
                return leader
        return None

    def _evidence_probe_challenger(self):
        """Dormant owner worth a bounded probe, or None (pure selection)."""
        from .routing.evidence import CircuitEvidence
        if (
            not self.evidence_decisions_enabled
            or not self.evidence_posterior
            or self.context_state not in ("warning", "search")
            or self.context_probe_module is not None
            or self.variable_order_fingerprint_active
            or self.adaptive_dendritic_fingerprint_active
        ):
            return None
        incumbent_id = self.active_motor_module
        if incumbent_id not in self.motor_modules:
            return None
        best_id = None
        best_post = None
        for module_id, posterior in self.evidence_posterior.items():
            if module_id == "__novelty__" or module_id == incumbent_id:
                continue
            module = self.motor_modules.get(module_id)
            if module is None or not module.dormant:
                continue
            if best_post is None or posterior > best_post:
                best_id, best_post = module_id, posterior
        if best_id is None:
            return None
        if self.evidence_posterior.get(incumbent_id, 0.0) - best_post >= self.evidence_enter_margin:
            return None
        payload = self.circuit_evidence.get(incumbent_id)
        incumbent_item = CircuitEvidence.from_dict(payload) if payload is not None else CircuitEvidence(incumbent_id)
        if incumbent_item.normal_outcomes + incumbent_item.probe_outcomes < self.evidence_min_evidence:
            return None
        return best_id

    def _evidence_maybe_begin_probe(self):
        """Start a bounded information-gain probe. Returns True if started."""
        from .routing.evidence import CircuitEvidence
        challenger_id = self._evidence_probe_challenger()
        if challenger_id is None or challenger_id not in self.motor_modules:
            return False
        incumbent_id = self.active_motor_module
        if incumbent_id not in self.motor_modules:
            return False
        for module_id in (challenger_id, incumbent_id):
            payload = self.circuit_evidence.get(module_id)
            item = CircuitEvidence.from_dict(payload) if payload is not None else CircuitEvidence(module_id)
            item.probe_log_evidence = 0.0
            item.probe_scored = 0
            self.circuit_evidence[module_id] = item.to_dict()
        incumbent = self.motor_modules[incumbent_id]
        incumbent.dormant = True
        incumbent.protected = True
        self.context_probe_queue = [challenger_id]
        self.context_probe_incumbent_module = incumbent_id
        self.context_probe_origin = "evidence_probe"
        self.context_probe_failed_reference = -1.0
        self.context_state = "search"
        self._start_next_probe()
        self.events.append({
            "step": self.step_count,
            "kind": "evidence_probe_started",
            "module_id": challenger_id,
            "incumbent_module": incumbent_id,
            "posterior": dict(self.evidence_posterior),
        })
        return True

    def _evidence_resolve_probe(self, candidate):
        """Resolve an evidence probe from protected tallies. No model updates."""
        from .routing.evidence import CircuitEvidence, probe_decision
        incumbent = self.motor_modules.get(self.context_probe_incumbent_module)
        candidate_payload = self.circuit_evidence.get(candidate.id)
        candidate_item = CircuitEvidence.from_dict(candidate_payload) if candidate_payload is not None else CircuitEvidence(candidate.id)
        incumbent_id = incumbent.id if incumbent is not None else None
        incumbent_payload = self.circuit_evidence.get(incumbent_id) if incumbent_id is not None else None
        incumbent_item = CircuitEvidence.from_dict(incumbent_payload) if incumbent_payload is not None else CircuitEvidence(incumbent_id or "?")
        verdict = probe_decision(
            candidate_item.probe_log_evidence, candidate_item.probe_scored,
            incumbent_item.probe_log_evidence, incumbent_item.probe_scored,
            min_probe=self.context_probe_min_steps,
            enter_margin=self.evidence_enter_margin,
            exit_margin=self.evidence_exit_margin,
            novelty_mass=self.evidence_novelty_mass,
        )
        if verdict == "accept":
            candidate.dormant = False
            candidate.protected = False
            candidate.detector_cusum = 0.0
            candidate.detector_frozen = False
            candidate.surprise_evidence = 0.0
            candidate.negative_surprise = 0.0
            candidate.negative_streak = 0
            candidate.confidence = 0.0
            self.active_motor_module = candidate.id
            self.context_state = "normal"
            self.context_probe_module = None
            self.context_probe_queue = []
            self._clear_probe_state()
            self.events.append({
                "step": self.step_count,
                "kind": "evidence_probe_accepted",
                "module_id": candidate.id,
                "incumbent_module": incumbent_id,
                "probe_scored": candidate_item.probe_scored,
            })
        else:
            candidate.dormant = True
            self.events.append({
                "step": self.step_count,
                "kind": "evidence_probe_rejected",
                "module_id": candidate.id,
                "incumbent_module": incumbent_id,
                "verdict": verdict,
                "probe_scored": candidate_item.probe_scored,
            })
            self.context_probe_module = None
            if incumbent is not None:
                incumbent.dormant = False
                incumbent.protected = False
                self.active_motor_module = incumbent.id
            self.context_state = "normal"
            self._clear_probe_state()

    def _evidence_begin_growth_watch(self, module_id, incumbent_id=None):
        """Watch a fresh recruit: confirm by evidence or revert to incumbent."""
        if not self.evidence_decisions_enabled:
            return
        if module_id not in self.motor_modules or incumbent_id not in self.motor_modules:
            return
        self.evidence_pending_growth = {
            "module_id": module_id,
            "incumbent_id": incumbent_id,
            "deadline_step": self.step_count + 2 * self.evidence_min_evidence,
            "edge_ema": 0.0,
            "edge_n": 0,
        }
        self.events.append({
            "step": self.step_count,
            "kind": "evidence_growth_pending",
            "module_id": module_id,
            "incumbent_module": incumbent_id,
        })

    def _evidence_check_growth_watch(self):
        """Confirm a watched recruit on sustained lead, else revert at deadline."""
        from .routing.evidence import CircuitEvidence
        pending = self.evidence_pending_growth
        if not pending or not self.evidence_decisions_enabled:
            return
        module_id, incumbent_id = pending.get("module_id"), pending.get("incumbent_id")
        if module_id not in self.motor_modules or incumbent_id not in self.motor_modules:
            self.evidence_pending_growth = None
            return
        edge_ema = float(pending.get("edge_ema", 0.0))
        edge_n = int(pending.get("edge_n", 0))
        # Sustained per-outcome lead confirms; sustained deficit reverts.
        # Cumulative posteriors bias incumbents, so the watch uses the recent
        # edge only. Ambiguity at deadline expires without yanking control:
        # deadline yanks caused measured retention collapse.
        if self.evidence_novelty_justified() or (
            edge_n >= self.evidence_min_evidence and edge_ema >= 0.5
        ):
            self.evidence_pending_growth = None
            self.events.append({
                "step": self.step_count,
                "kind": "evidence_growth_confirmed",
                "module_id": module_id,
                "incumbent_module": incumbent_id,
            })
            return
        if self.step_count >= int(pending.get("deadline_step", 0)):
            if edge_n >= self.evidence_min_evidence and edge_ema <= -0.5:
                recruited = self.motor_modules[module_id]
                incumbent = self.motor_modules[incumbent_id]
                recruited.dormant = True
                recruited.protected = False
                incumbent.dormant = False
                incumbent.protected = False
                self.active_motor_module = incumbent_id
                self.context_state = "normal"
                self.context_probe_queue = []
                self._clear_probe_state()
                self.evidence_pending_growth = None
                self.events.append({
                    "step": self.step_count,
                    "kind": "evidence_growth_reverted",
                    "module_id": module_id,
                    "incumbent_module": incumbent_id,
                })
            else:
                self.evidence_pending_growth = None
                self.events.append({
                    "step": self.step_count,
                    "kind": "evidence_growth_expired",
                    "module_id": module_id,
                    "incumbent_module": incumbent_id,
                })

    def _evidence_growth_allowed(self):
        if not self.evidence_decisions_enabled:
            return True
        return self.evidence_novelty_justified()

    def evidence_novelty_justified(self):
        """Growth gate: recent best fit poor AND every circuit well-sampled."""
        from .routing.evidence import CircuitEvidence
        if not self.evidence_decisions_enabled or not self.circuit_evidence:
            return False
        if self.evidence_recent_count < self.evidence_min_evidence:
            return False
        if self.evidence_recent_best_ll >= float(self.evidence_novelty_fit_threshold):
            return False
        for payload in self.circuit_evidence.values():
            item = CircuitEvidence.from_dict(payload)
            if item.normal_outcomes + item.probe_outcomes < self.evidence_min_evidence:
                return False
        return True

    def evidence_report(self):
        """Shadow comparison: evidence leader vs legacy controller. No control."""
        return {
            "mode": self.evidence_router_mode,
            "posterior": dict(self.evidence_posterior),
            "shadow_steps": self.evidence_shadow_steps,
            "legacy_active": self.active_motor_module,
            "legacy_state": self.context_state,
        }

    def enable_variable_order_owner_evidence(self) -> None:
        """Track a normalized, target-blind owner/novelty posterior in shadow mode."""
        if not self.variable_order_learning_enabled:
            raise ValueError("variable-order learning must be enabled first")
        self.variable_order_owner_evidence_enabled = True
        self.variable_order_owner_posterior = {}
        self.variable_order_owner_evidence_count = 0
        self.variable_order_midpoint_novelty_probability = 0.0
        self.variable_order_temporal_refresh_used = False
        self.validate()

    def enable_variable_order_owner_probes(self, midpoint: bool = False, min_probability: float = 0.0, min_margin: float = 0.0) -> None:
        if not self.variable_order_owner_evidence_enabled:
            raise ValueError("owner evidence must be enabled before owner probes")
        if not isinstance(midpoint, bool):
            raise ValueError("owner midpoint probe flag must be boolean")
        if not all(math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0 for value in (min_probability, min_margin)):
            raise ValueError("owner midpoint probe thresholds must be in [0, 1]")
        self.variable_order_owner_probe_enabled = True
        self.variable_order_owner_midpoint_probe_enabled = midpoint
        self.variable_order_owner_midpoint_probe_min_probability = float(min_probability)
        self.variable_order_owner_midpoint_probe_min_margin = float(min_margin)
        self.validate()

    def _variable_order_begin_owner_probe(self, owner_id: str) -> bool:
        incumbent_id = self.variable_order_fingerprint_incumbent
        if owner_id not in self.motor_modules or owner_id == incumbent_id or incumbent_id not in self.motor_modules:
            return False
        self.variable_order_fingerprint_active = False
        self.variable_order_fingerprint_owner = None
        self.variable_order_fingerprint_pending = None
        self.composition_fingerprint_pending = None
        self.context_probe_queue = [owner_id]
        self.context_probe_incumbent_module = incumbent_id
        self.context_probe_origin = "owner_evidence"
        self.context_probe_failed_reference = -1.0
        self.context_state = "search"
        self.motor_modules[incumbent_id].dormant = True
        self.motor_modules[incumbent_id].protected = True
        self._start_next_probe()
        self.events.append({"step": self.step_count, "kind": "variable_order_owner_probe_started", "owner_module": owner_id, "incumbent_module": incumbent_id, "posterior": dict(self.variable_order_owner_posterior)})
        return True

    def _variable_order_maybe_begin_midpoint_owner_probe(self) -> bool:
        if not self.variable_order_owner_midpoint_probe_enabled or not self.variable_order_owner_posterior:
            return False
        midpoint = self.variable_order_fingerprint_window // 2
        if self.variable_order_fingerprint_count != midpoint:
            return False
        leader = max(self.variable_order_owner_posterior, key=lambda key: (self.variable_order_owner_posterior[key], key != "__novelty__", key))
        if leader == "__novelty__":
            return False
        ordered = sorted(self.variable_order_owner_posterior.values(), reverse=True)
        probability = self.variable_order_owner_posterior[leader]
        margin = probability - (ordered[1] if len(ordered) > 1 else 0.0)
        if probability < self.variable_order_owner_midpoint_probe_min_probability or margin < self.variable_order_owner_midpoint_probe_min_margin:
            return False
        started = self._variable_order_begin_owner_probe(leader)
        if started:
            self.context_probe_origin = "owner_evidence_midpoint"
        return started

    def _variable_order_refresh_owner_evidence(self) -> None:
        if not self.variable_order_owner_evidence_enabled:
            return
        scores = {module_id: 0.0 for module_id in self.motor_modules}
        scores["__novelty__"] = 0.0
        for candidate in self._variable_order_midpoint_candidates():
            owner_id = candidate[7]
            bucket = str(owner_id) if owner_id in self.motor_modules else "__novelty__"
            scores[bucket] = max(scores[bucket], float(candidate[1]))
        maximum = max(scores.values()) if scores else 0.0
        weights = {key: math.exp(value - maximum) for key, value in scores.items()}
        total = sum(weights.values())
        self.variable_order_owner_posterior = {key: weights[key] / total for key in sorted(weights)}
        self.variable_order_owner_evidence_count = self.variable_order_fingerprint_count
        ordered = sorted(self.variable_order_owner_posterior.items(), key=lambda item: (-item[1], item[0]))
        self.events.append({
            "step": self.step_count,
            "kind": "variable_order_owner_evidence_shadow",
            "evidence_count": self.variable_order_owner_evidence_count,
            "leader": ordered[0][0],
            "leader_probability": ordered[0][1],
            "margin": ordered[0][1] - (ordered[1][1] if len(ordered) > 1 else 0.0),
            "posterior": dict(self.variable_order_owner_posterior),
        })

    @staticmethod
    def _variable_order_key(sources: Sequence[str]) -> str:
        return Organism._dendritic_feature_key(tuple(str(source) for source in sources))

    def _variable_order_feature_bank(self) -> Tuple[Tuple[str, ...], ...]:
        if self.variable_order_feature_order:
            return self.variable_order_feature_order
        features = [tuple(str(source) for source in pair) for pair in combinations(self.input_ids, 2)]
        if self.max_dendritic_order >= 3:
            features.extend(tuple(str(source) for source in triple) for triple in combinations(self.input_ids, 3))
        return tuple(features)

    def _variable_order_begin_fingerprint(self) -> None:
        if not self.variable_order_learning_enabled or self.variable_order_fingerprint_active:
            return
        incumbent = self.motor_modules.get(self.active_motor_module)
        if incumbent is None:
            return
        self.variable_order_fingerprint_active = True
        self.variable_order_fingerprint_owner = incumbent.id
        self.variable_order_fingerprint_incumbent = incumbent.id
        self.variable_order_fingerprint_start_step = self.step_count
        self.variable_order_fingerprint_deadline = self.step_count + self.variable_order_fingerprint_window + (self.delayed_credit_delay if self.delayed_credit_enabled else 0)
        self.variable_order_fingerprint_count = 0
        self.variable_order_fingerprint_evidence = {}
        self.variable_order_owner_posterior = {}
        self.variable_order_owner_evidence_count = 0
        self.variable_order_fingerprint_pending = None
        self.variable_order_normal_evidence = {}
        self.variable_order_normal_pending = None
        self.composition_fingerprint_evidence = {}
        self.composition_fingerprint_pending = None
        self.general_fingerprint_evidence = {}
        self.variable_order_normal_count = 0
        self.composition_fingerprint_evidence = {}
        self.composition_fingerprint_pending = None
        self.context_probe_module = None
        self.context_probe_queue = []
        self.context_probe_remaining = 0
        self.context_probe_incumbent_module = None
        self.context_probe_origin = None
        incumbent.dormant = False
        incumbent.protected = True
        self.context_state = "fingerprint"
        self.events.append({
            "step": self.step_count,
            "kind": "variable_order_fingerprint_started",
            "module_id": incumbent.id,
            "deadline": self.variable_order_fingerprint_deadline,
        })

    def _variable_order_record_fingerprint_pending(self) -> None:
        if not self.variable_order_fingerprint_active or self.variable_order_fingerprint_pending is not None:
            return
        if self.context_pending_mode != "fingerprint":
            return
        sigma = float(self.context_pending_exploration_sigma)
        xi = float(self.context_pending_exploration_value)
        entries = []
        general_entries = []
        if sigma > 1e-12:
            inputs = {identifier: float(self.cells[identifier].activation) for identifier in self.input_ids}
            factor = xi / sigma
            for feature in self._variable_order_feature_bank():
                product = 1.0
                for source in feature:
                    product *= inputs[source]
                entries.append({"key": self._variable_order_key(feature), "eligibility": math.tanh(product) * factor})
            for candidate in self.general_feature_candidates():
                # Direct sensor-only candidates remain the responsibility of
                # the proven variable-order learner. M4 handles graph reuse.
                if all(source in self.input_ids for source in candidate["sources"]):
                    continue
                product = 1.0
                for source in candidate["sources"]:
                    product *= float(self.cells[source].activation)
                general_entries.append({"key": candidate["key"], "eligibility": math.tanh(product) * factor})
        self.variable_order_fingerprint_pending = {
            "owner_module": self.variable_order_fingerprint_owner,
            "step": self.step_count,
            "exploration_value": xi,
            "exploration_sigma": sigma,
            "eligibilities": entries,
            "general_eligibilities": general_entries,
        }

    def _variable_order_record_normal_pending(self) -> None:
        if not self.variable_order_learning_enabled or self.variable_order_fingerprint_active or self.variable_order_normal_pending is not None:
            return
        if self.context_pending_mode != "normal":
            return
        sigma = float(self.context_pending_exploration_sigma)
        xi = float(self.context_pending_exploration_value)
        entries = []
        if sigma > 1e-12:
            inputs = {identifier: float(self.cells[identifier].activation) for identifier in self.input_ids}
            mu = float(self.cells[self.motor_modules[self.pending_motor_module].cell_id].activation) if self.pending_motor_module in self.motor_modules else 0.0
            factor = (1.0 - mu * mu) * xi / sigma
            for feature in self._variable_order_feature_bank():
                product = 1.0
                for source in feature:
                    product *= inputs[source]
                entries.append({"key": self._variable_order_key(feature), "eligibility": math.tanh(product) * factor})
        self.variable_order_normal_pending = {
            "owner_module": self.pending_motor_module,
            "step": self.step_count,
            "eligibilities": entries,
        }

    def _variable_order_accumulate_normal(self, reward: float) -> None:
        pending = self.variable_order_normal_pending
        self.variable_order_normal_pending = None
        if pending is None or self.variable_order_fingerprint_active:
            return
        raw_reward = float(reward)
        for payload in pending.get("eligibilities", []):
            key = str(payload["key"])
            value = self.variable_order_normal_evidence.setdefault(key, {"sum": 0.0, "sumsq": 0.0, "count": 0.0})
            credit = raw_reward * float(payload["eligibility"])
            value["sum"] += credit
            value["sumsq"] += credit * credit
            value["count"] += 1.0
        self.variable_order_normal_count += 1
        if self.variable_order_normal_count < self.variable_order_normal_window:
            return
        candidates = []
        for key, value in self.variable_order_normal_evidence.items():
            count = int(value.get("count", 0.0))
            total = float(value.get("sum", 0.0))
            sumsq = float(value.get("sumsq", 0.0))
            if count < self.variable_order_normal_window:
                continue
            mean = total / float(count)
            variance = max(0.0, sumsq / float(count) - mean * mean)
            stderr = math.sqrt(variance / float(count))
            z = mean / stderr if stderr > 1e-12 else (math.inf if abs(mean) > 0.0 else 0.0)
            score = abs(total) / math.sqrt(max(1e-12, sumsq))
            candidates.append((score, abs(z), key, total))
        candidates.sort(key=lambda item: (-item[0], -item[1], self._variable_order_feature_rank(item[2])))
        active_key = self.variable_order_module_features.get(str(pending.get("owner_module")))
        active_score = next((item[0] for item in candidates if item[2] == active_key), 0.0)
        if (
            self.composition_direct_route_pending
            and self.composition_learning_enabled
            and self.composition_feature_owners
            and self.context_state == "normal"
            and self.context_probe_module is None
            and self.step_count <= self.variable_order_detector_hold_until
        ):
            active_candidate = next((item for item in candidates if item[2] == active_key), None)
            active_z = active_candidate[1] if active_candidate is not None else 0.0
            direct_owner = self.variable_order_feature_owners.get(active_key) if active_key is not None else None
            direct_route = (
                active_candidate is not None
                and isinstance(active_key, str)
                and active_key.startswith(("k2:", "k3:"))
                and direct_owner == pending.get("owner_module")
            )
            if direct_route and active_z < 2.5:
                self._variable_order_begin_fingerprint()
                self.events.append({
                    "step": self.step_count,
                    "kind": "composition_normal_verification_retry_started",
                    "owner_module": pending.get("owner_module"),
                    "active_feature": active_key,
                    "active_z": active_z,
                })
                self.variable_order_normal_evidence = {}
                self.variable_order_normal_count = 0
                self.composition_direct_route_pending = False
                self.composition_verification_suppressed_until = max(
                    self.composition_verification_suppressed_until,
                    self.variable_order_detector_hold_until,
                )
                return
            self.composition_direct_route_pending = False
        if len(candidates) >= 2:
            best, second = candidates[0], candidates[1]
            best_feature = self._dendritic_feature_from_key(best[2])
            best_owner = self._variable_order_owner_for_key(best[2], best_feature)
            if active_score < 0.20 and best[2] != active_key and best_owner != pending.get("owner_module") and abs(best[3]) > 0.0 and best[0] >= 2.70 and best[1] >= 3.0 and best[0] - second[0] >= 0.15:
                self._variable_order_begin_fingerprint()
        if (
            self.general_structural_learning_enabled
            and not self.variable_order_fingerprint_active
            and self.step_count - self.general_last_audit_step >= self.general_audit_interval
        ):
            self.general_last_audit_step = self.step_count
            self._variable_order_begin_fingerprint()
            self.events.append({"step": self.step_count, "kind": "general_structural_audit_started"})
        self.variable_order_normal_evidence = {}
        self.variable_order_normal_count = 0

    def _variable_order_credit_strength(self, signed_sum: float, count: int) -> float:
        # Fixed local transform: four times the signed sample mean, bounded to
        # the graph's synaptic range.  It is independent of task metadata.
        return max(-1.0, min(1.0, 4.0 * float(signed_sum) / float(max(1, count))))

    def _variable_order_feature_rank(self, key: str) -> int:
        """Return persisted representation-bank rank used for exact ties."""
        try:
            feature = self._dendritic_feature_from_key(key)
            return self.variable_order_feature_order.index(feature)
        except (ValueError, AttributeError):
            return len(self.variable_order_feature_order) + 1

    def _variable_order_existing_owner(self, feature: Tuple[str, ...], owner_id: str) -> bool:
        """Check graph evidence before importing a legacy/generic owner map."""
        owner = self.motor_modules.get(str(owner_id))
        if owner is None or owner.afferent_kind != "hidden":
            return False
        expected_type = "dendritic_product" if len(feature) == 2 else "dendritic_product_n"
        for cell in self.cells.values():
            if cell.activation_type != expected_type or tuple(cell.dendritic_sources) != tuple(feature):
                continue
            edge = self.graph.get(cell.id, owner.cell_id)
            if edge is not None:
                return True
        return False

    def _variable_order_owner_for_key(self, key: str, feature: Tuple[str, ...]) -> Optional[str]:
        owner_id = self.variable_order_feature_owners.get(key)
        if owner_id in self.motor_modules and self._variable_order_existing_owner(feature, owner_id):
            return owner_id
        if len(feature) == 2:
            legacy = self.adaptive_dendritic_pair_owners.get(self._adaptive_dendritic_pair_key(feature))
            if legacy in self.motor_modules and self._variable_order_existing_owner(feature, legacy):
                return legacy
        return None

    def _general_structural_route(self, direct_competitor: Optional[Tuple[float, str]] = None) -> bool:
        """Resolve, confirm, or install the best graph-derived candidate."""
        if not self.general_structural_learning_enabled or not self.general_fingerprint_evidence:
            return False
        candidates = []
        for key, value in self.general_fingerprint_evidence.items():
            count = int(value.get("count", 0.0))
            total = float(value.get("sum", 0.0))
            sumsq = float(value.get("sumsq", 0.0))
            if count <= 0:
                continue
            mean = total / count
            variance = max(0.0, sumsq / count - mean * mean)
            stderr = math.sqrt(variance / count)
            z = mean / stderr if stderr > 1e-12 else (math.inf if abs(mean) > 0.0 else 0.0)
            score = abs(total) / math.sqrt(max(1e-12, sumsq))
            candidates.append((score, abs(z), key, total, count))
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        if not candidates:
            return False
        winner = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else 0.0
        direct_score = direct_competitor[0] if direct_competitor is not None else 0.0
        if not (abs(winner[3]) > 0.0 and winner[1] >= 2.5 and winner[0] - second_score >= 0.05 and winner[0] >= direct_score + 0.05):
            return False
        key = winner[2]
        owner_id = self.general_feature_owners.get(key)
        installed = False
        if owner_id not in self.motor_modules:
            if self.general_novelty_candidate == key:
                self.general_novelty_confirmations += 1
            else:
                self.general_novelty_candidate = key
                self.general_novelty_confirmations = 1
            if self.general_novelty_confirmations < self.general_required_confirmations:
                self.variable_order_fingerprint_start_step = self.step_count
                self.variable_order_fingerprint_deadline = self.step_count + self.variable_order_fingerprint_window + (self.delayed_credit_delay if self.delayed_credit_enabled else 0)
                self.variable_order_fingerprint_count = 0
                self.variable_order_fingerprint_evidence = {}
                self.composition_fingerprint_evidence = {}
                self.general_fingerprint_evidence = {}
                self.variable_order_fingerprint_pending = None
                self.composition_fingerprint_pending = None
                self.events.append({"step": self.step_count, "kind": "general_feature_confirmation_started", "feature_key": key, "confirmation": self.general_novelty_confirmations, "required": self.general_required_confirmations})
                return True
            proposal = next((item for item in self.general_feature_candidates() if item["key"] == key), None)
            if proposal is None:
                return False
            if len(self.motor_modules) >= self.context_max_modules:
                pruneable = [
                    old_key for old_key, old_owner in self.general_feature_owners.items()
                    if self.general_feature_reuse.get(old_key, 0) <= self.general_prune_reuse_ceiling
                    and self.step_count - self.general_feature_last_used_step.get(old_key, 0) >= self.general_audit_interval
                    and old_owner != self.active_motor_module
                    and self.motor_modules[old_owner].dormant
                ]
                pruneable.sort(key=lambda old_key: (
                    self.general_feature_reuse.get(old_key, 0),
                    -self.general_feature_last_used_step.get(old_key, 0),
                    old_key,
                ))
                if pruneable:
                    self.general_fingerprint_evidence = {}
                    self._variable_order_internal_transition = True
                    try:
                        self.prune_general_feature(pruneable[0])
                    finally:
                        self._variable_order_internal_transition = False
            occupied = set(self.variable_order_module_features) | set(self.composition_module_features) | set(self.general_module_features) | set(self.adaptive_dendritic_pair_owners.values())
            reusable = next((module for module in self.motor_modules.values() if module.id not in occupied and module.id != self.active_motor_module and module.dormant), None)
            if reusable is not None or len(self.motor_modules) < self.context_max_modules:
                checkpoint = copy.deepcopy(self.__dict__)
                try:
                    owner = reusable or self._recruit_motor_module(afferent_kind="hidden")
                    self._variable_order_internal_transition = True
                    try:
                        self.install_general_feature(proposal["sources"], owner.id, self._variable_order_credit_strength(winner[3], winner[4]))
                    finally:
                        self._variable_order_internal_transition = False
                    owner_id = owner.id
                    installed = True
                except Exception:
                    self.__dict__.clear()
                    self.__dict__.update(checkpoint)
                    owner_id = None
            self.general_novelty_candidate = None
            self.general_novelty_confirmations = 0
        else:
            if self.step_count - self.general_feature_last_used_step.get(key, self.step_count) >= 128:
                self.mark_general_feature_reused(key)
            self.general_feature_last_used_step[key] = self.step_count
            self.general_novelty_candidate = None
            self.general_novelty_confirmations = 0
        if owner_id not in self.motor_modules:
            return False
        previous_id = self.active_motor_module
        for module in self.motor_modules.values():
            module.dormant = module.id != owner_id
            module.protected = False
        self.motor_modules[owner_id].dormant = False
        self.active_motor_module = owner_id
        self.context_state = "normal"
        self.context_probe_module = None
        self.context_probe_queue = []
        self.context_probe_remaining = 0
        self.context_probe_incumbent_module = None
        self.context_probe_origin = None
        self.variable_order_detector_hold_until = self.step_count + 64
        self.variable_order_fingerprint_active = False
        self.variable_order_fingerprint_owner = None
        self.variable_order_fingerprint_pending = None
        self.variable_order_normal_evidence = {}
        self.variable_order_normal_pending = None
        self.variable_order_normal_count = 0
        self.composition_fingerprint_evidence = {}
        self.composition_fingerprint_pending = None
        self.general_fingerprint_evidence = {}
        self.events.append({"step": self.step_count, "kind": "general_feature_fingerprint_resolved", "feature_key": key, "owner_module": owner_id, "previous_module": previous_id, "installed": installed, "score": winner[0], "evidence_count": winner[4]})
        return True

    def _variable_order_route(self) -> None:
        if not self.variable_order_fingerprint_active:
            return
        self.variable_order_route_count += 1
        if self.variable_order_owner_probe_enabled and self.variable_order_owner_posterior:
            leader = max(self.variable_order_owner_posterior, key=lambda key: (self.variable_order_owner_posterior[key], key != "__novelty__", key))
            if leader != "__novelty__" and self._variable_order_begin_owner_probe(leader):
                return
        direct_competitor = self._variable_order_best_candidate()
        if self._general_structural_route(direct_competitor):
            return
        composition_result = self._composition_route(self.variable_order_fingerprint_incumbent, direct_competitor)
        if composition_result is not None:
            target_id, composition_info = composition_result
            previous_id = self.active_motor_module
            for module in self.motor_modules.values():
                module.dormant = module.id != target_id
                module.protected = False
            target = self.motor_modules[target_id]
            target.dormant = False
            target.detector_cusum = 0.0
            target.detector_frozen = False
            target.surprise_evidence = 0.0
            target.negative_surprise = 0.0
            target.negative_streak = 0
            self.active_motor_module = target_id
            self.context_state = "normal"
            self.context_probe_module = None
            self.context_probe_queue = []
            self.context_probe_remaining = 0
            self.context_probe_incumbent_module = None
            self.context_probe_origin = None
            self.variable_order_detector_hold_until = self.step_count + 64
            self.variable_order_fingerprint_active = False
            self.variable_order_fingerprint_owner = None
            self.variable_order_fingerprint_pending = None
            self.variable_order_normal_evidence = {}
            self.variable_order_normal_pending = None
            self.variable_order_normal_count = 0
            self.composition_fingerprint_evidence = {}
            self.composition_fingerprint_pending = None
            self.events.append({
                "step": self.step_count, "kind": "composition_fingerprint_resolved",
                "module_id": target_id, "feature": composition_info[0],
                "owner_module": target_id, "previous_module": previous_id,
                "signed_sum": composition_info[1], "evidence_count": composition_info[2],
                "installed": composition_info[3], "score": composition_info[4],
                "owner_posterior": dict(self.variable_order_owner_posterior),
            })
            if target_id != previous_id:
                self.events.append({"step": self.step_count, "kind": "motor_module_reactivated", "module_id": target_id, "fingerprint": True, "composition": True})
            return
        candidates = []
        for key, value in self.variable_order_fingerprint_evidence.items():
            count = int(value.get("count", 0.0))
            if count <= 0:
                continue
            total = float(value.get("sum", 0.0))
            sumsq = float(value.get("sumsq", 0.0))
            mean = total / float(count)
            variance = max(0.0, sumsq / float(count) - mean * mean)
            stderr = math.sqrt(variance / float(count))
            z = mean / stderr if stderr > 1e-12 else (math.inf if abs(mean) > 0.0 else 0.0)
            score = abs(total) / math.sqrt(max(1e-12, sumsq))
            feature = self._dendritic_feature_from_key(key)
            candidates.append((score, abs(z), feature, key, total, count))
        candidates.sort(key=lambda item: (-item[0], -item[1], self._variable_order_feature_rank(item[3])))
        incumbent_id = self.variable_order_fingerprint_incumbent
        winner = candidates[0] if candidates else None
        second_score = candidates[1][0] if len(candidates) > 1 else 0.0
        credible = bool(
            winner is not None
            and abs(winner[4]) > 0.0
            and winner[1] >= 2.5
            and winner[0] - second_score >= 0.05
        )
        owner_id = self._variable_order_owner_for_key(winner[3], winner[2]) if credible else None
        installed = False
        final_novelty = self.variable_order_owner_posterior.get("__novelty__", 0.0)
        if (
            credible
            and owner_id is None
            and self.variable_order_owner_probe_enabled
            and incumbent_id in self.variable_order_module_features
            and not self.variable_order_temporal_refresh_used
            and final_novelty - self.variable_order_midpoint_novelty_probability >= 0.25
        ):
            novelty_rise = final_novelty - self.variable_order_midpoint_novelty_probability
            self.variable_order_fingerprint_start_step = self.step_count
            self.variable_order_fingerprint_deadline = self.step_count + self.variable_order_fingerprint_window + (self.delayed_credit_delay if self.delayed_credit_enabled else 0)
            self.variable_order_fingerprint_count = 0
            self.variable_order_fingerprint_evidence = {}
            self.variable_order_owner_posterior = {}
            self.variable_order_owner_evidence_count = 0
            self.variable_order_midpoint_novelty_probability = 0.0
            self.variable_order_temporal_refresh_used = True
            self.composition_fingerprint_evidence = {}
            self.composition_fingerprint_pending = None
            self.events.append({
                "step": self.step_count,
                "kind": "variable_order_temporal_fingerprint_refreshed",
                "module_id": incumbent_id,
                "novelty_rise": novelty_rise,
                "deadline": self.variable_order_fingerprint_deadline,
            })
            return
        if credible and owner_id is None and incumbent_id in self.motor_modules:
            # Bootstrap the first feature on the initial module; every later
            # unowned feature gets a distinct module up to the hard cap.
            if not self.variable_order_module_features:
                owner_id = incumbent_id
            elif len(self.motor_modules) < self.context_max_modules:
                checkpoint = copy.deepcopy(self.__dict__)
                try:
                    incumbent_module = self.motor_modules.get(incumbent_id)
                    owner = self._recruit_motor_module(
                        afferent_kind=incumbent_module.afferent_kind
                        if incumbent_module is not None else "hidden")
                    if owner.id in self.variable_order_module_features:
                        raise RuntimeError("variable-order module already owns a feature")
                    self._variable_order_internal_transition = True
                    try:
                        self.install_dendritic_feature(
                            winner[2], owner.id,
                            self._variable_order_credit_strength(winner[4], winner[5]),
                        )
                    finally:
                        self._variable_order_internal_transition = False
                    owner_id = owner.id
                    installed = True
                except Exception:
                    self.__dict__.clear()
                    self.__dict__.update(checkpoint)
                    owner_id = None
            else:
                owner_id = None
        if credible and owner_id is not None and winner is not None:
            if owner_id not in self.variable_order_module_features:
                existing = self._variable_order_existing_owner(winner[2], owner_id)
                if owner_id == incumbent_id and not installed and not existing:
                    self._variable_order_internal_transition = True
                    try:
                        self.install_dendritic_feature(
                            winner[2], owner_id,
                            self._variable_order_credit_strength(winner[4], winner[5]),
                        )
                    finally:
                        self._variable_order_internal_transition = False
                    installed = True
                if installed or existing or self._variable_order_owner_for_key(winner[3], winner[2]) == owner_id:
                    self.variable_order_module_features[owner_id] = winner[3]
                    self.variable_order_feature_owners[winner[3]] = owner_id
                    self.variable_order_install_count += int(installed)
            elif self.variable_order_module_features[owner_id] != winner[3]:
                owner_id = None
        target_id = owner_id if owner_id in self.motor_modules else incumbent_id
        if target_id not in self.motor_modules:
            target_id = self.active_motor_module
        previous_id = self.active_motor_module
        for module in self.motor_modules.values():
            module.dormant = module.id != target_id
            module.protected = False
        if target_id in self.motor_modules:
            target = self.motor_modules[target_id]
            target.dormant = False
            target.detector_cusum = 0.0
            target.detector_frozen = False
            target.surprise_evidence = 0.0
            target.negative_surprise = 0.0
            target.negative_streak = 0
            self.active_motor_module = target_id
        self.context_state = "normal"
        self.context_probe_module = None
        self.context_probe_queue = []
        self.context_probe_remaining = 0
        self.context_probe_incumbent_module = None
        self.context_probe_origin = None
        self.variable_order_detector_hold_until = self.step_count + 64
        self.variable_order_fingerprint_active = False
        self.variable_order_fingerprint_owner = None
        self.variable_order_fingerprint_pending = None
        self.variable_order_normal_evidence = {}
        self.variable_order_normal_pending = None
        self.variable_order_normal_count = 0
        self.composition_fingerprint_evidence = {}
        self.composition_fingerprint_pending = None
        self.events.append({
            "step": self.step_count,
            "kind": "variable_order_fingerprint_resolved",
            "module_id": target_id,
            "feature": winner[3] if credible and winner is not None else None,
            "owner_module": owner_id,
            "previous_module": previous_id,
            "signed_sum": winner[4] if credible and winner is not None else 0.0,
            "score": winner[0] if credible and winner is not None else 0.0,
            "evidence_count": winner[5] if credible and winner is not None else 0,
            "installed": installed,
            "owner_posterior": dict(self.variable_order_owner_posterior),
        })
        if target_id != previous_id and target_id in self.motor_modules:
            self.events.append({"step": self.step_count, "kind": "motor_module_reactivated", "module_id": target_id, "fingerprint": True})
        direct_route = bool(
            self.composition_feature_owners and credible and winner is not None
            and owner_id in self.motor_modules and target_id == owner_id
        )
        if direct_route and self.step_count < self.composition_verification_suppressed_until:
            self.composition_direct_route_pending = False
            self.composition_verification_suppressed_until = self.variable_order_detector_hold_until
        else:
            self.composition_direct_route_pending = direct_route

    def _variable_order_accumulate_fingerprint(self, reward: float) -> None:
        pending = self.variable_order_fingerprint_pending
        self.variable_order_fingerprint_pending = None
        if pending is None or not self.variable_order_fingerprint_active:
            return
        raw_reward = float(reward)
        for payload in pending.get("eligibilities", []):
            key = str(payload["key"])
            value = self.variable_order_fingerprint_evidence.setdefault(key, {"sum": 0.0, "sumsq": 0.0, "count": 0.0})
            credit = raw_reward * float(payload["eligibility"])
            value["sum"] += credit
            value["sumsq"] += credit * credit
            value["count"] += 1.0
        for payload in pending.get("general_eligibilities", []):
            key = str(payload["key"])
            value = self.general_fingerprint_evidence.setdefault(key, {"sum": 0.0, "sumsq": 0.0, "count": 0.0})
            credit = raw_reward * float(payload["eligibility"])
            value["sum"] += credit
            value["sumsq"] += credit * credit
            value["count"] += 1.0
        self.variable_order_fingerprint_count += 1
        self._variable_order_refresh_owner_evidence()
        if self.variable_order_fingerprint_count == self.variable_order_fingerprint_window // 2:
            self.variable_order_midpoint_novelty_probability = self.variable_order_owner_posterior.get("__novelty__", 0.0)
        if self._variable_order_midpoint_resolve():
            return
        if self._variable_order_maybe_begin_midpoint_owner_probe():
            return
        if self.variable_order_fingerprint_count >= self.variable_order_fingerprint_window:
            self._variable_order_route()

    def _validate_variable_order_pending(self, pending: Optional[Mapping[str, object]], mode: str) -> None:
        if pending is None:
            return
        if not isinstance(pending, Mapping):
            raise AssertionError("variable-order pending record must be a mapping")
        owner_id = pending.get("owner_module")
        if not isinstance(owner_id, str) or owner_id not in self.motor_modules:
            raise AssertionError("variable-order pending owner is unknown")
        if not self._pending_outcome:
            raise AssertionError("variable-order pending record requires a pending outcome")
        step = pending.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step != self.step_count:
            raise AssertionError("variable-order pending step must equal current action step")
        if mode == "fingerprint":
            if not self.variable_order_fingerprint_active or owner_id != self.variable_order_fingerprint_owner:
                raise AssertionError("variable-order fingerprint pending mode/owner is invalid")
            if step < self.variable_order_fingerprint_start_step or step > self.variable_order_fingerprint_deadline:
                raise AssertionError("variable-order fingerprint pending timing is invalid")
        elif mode == "normal":
            if not self.variable_order_learning_enabled or self.variable_order_fingerprint_active:
                raise AssertionError("variable-order normal pending mode is invalid")
            if owner_id != self.pending_motor_module:
                raise AssertionError("variable-order normal pending owner is invalid")
            if self.variable_order_normal_count < 0 or self.variable_order_normal_count >= self.variable_order_normal_window:
                raise AssertionError("variable-order normal pending count is invalid")
        else:
            raise AssertionError("unknown variable-order pending mode")
        entries = pending.get("eligibilities")
        if not isinstance(entries, list):
            raise AssertionError("variable-order pending eligibilities must be a list")
        seen = set()
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise AssertionError("variable-order pending entry must be a mapping")
            key = entry.get("key")
            if not isinstance(key, str):
                raise AssertionError("variable-order pending feature key is invalid")
            feature = self._dendritic_feature_from_key(key)
            if key != self._dendritic_feature_key(feature) or len(feature) > self.max_dendritic_order or any(source not in self.input_ids for source in feature):
                raise AssertionError("variable-order pending feature is invalid")
            if key in seen or not math.isfinite(float(entry.get("eligibility", 0.0))):
                raise AssertionError("variable-order pending eligibility is invalid")
            seen.add(key)

    @staticmethod
    def _dendritic_feature_key(sources: Sequence[str]) -> str:
        values = tuple(sorted(str(source) for source in sources))
        if len(values) < 2 or len(set(values)) != len(values):
            raise ValueError("dendritic feature sources must be distinct")
        return "k%d:%s" % (len(values), "|".join(values))

    @staticmethod
    def _dendritic_feature_from_key(key: str) -> Tuple[str, ...]:
        text = str(key)
        if ":" not in text:
            raise ValueError("dendritic feature key is missing arity")
        prefix, encoded = text.split(":", 1)
        if not prefix.startswith("k") or not prefix[1:].isdigit() or not encoded:
            raise ValueError("dendritic feature key is malformed")
        arity = int(prefix[1:])
        values = tuple(encoded.split("|"))
        if arity < 2 or len(values) != arity or any(not value for value in values) or len(set(values)) != arity or tuple(sorted(values)) != values:
            raise ValueError("dendritic feature key has invalid arity or sources")
        return values

    def install_dendritic_feature(
        self,
        sources: Sequence[str],
        owner_module_id: str,
        signed_local_credit: float,
        cell_id: Optional[str] = None,
    ) -> str:
        """Atomically install one unowned order-N feature on a hidden nursery cell.

        This is graph substrate only: caller supplies local signed credit, while
        no evaluator target or task metadata enters the organism.
        """
        values = tuple(sorted(str(source) for source in sources))
        if not self.variable_order_enabled or self.max_dendritic_order < len(values):
            raise ValueError("variable-order substrate is disabled or feature order exceeds cap")
        if len(values) not in (2, 3) or len(set(values)) != len(values) or any(source not in self.input_ids for source in values):
            raise ValueError("only distinct canonical input pairs or triples are supported")
        if not math.isfinite(float(signed_local_credit)) or not -1.0 <= float(signed_local_credit) <= 1.0:
            raise ValueError("signed local credit must be finite and in [-1, 1]")
        key = self._dendritic_feature_key(values)
        if key in self.variable_order_feature_owners:
            raise ValueError("dendritic feature is already owned")
        owner = self.motor_modules.get(str(owner_module_id))
        if owner is None or owner.afferent_kind not in ("input", "hidden") or owner.cell_id not in self.cells or self.cells[owner.cell_id].kind != "motor_module":
            raise ValueError("feature owner must be an existing motor module")
        candidate_ids = [str(cell_id)] if cell_id is not None else sorted(
            identifier for identifier, cell in self.cells.items() if cell.kind == "hidden" and cell.activation_type == "additive"
        )
        if not candidate_ids or candidate_ids[0] not in self.cells:
            raise ValueError("feature install requires an additive hidden nursery cell")
        nursery_id = candidate_ids[0]
        nursery = self.cells[nursery_id]
        if nursery.kind != "hidden" or nursery.activation_type != "additive":
            raise ValueError("feature install cannot overwrite a product cell")
        for existing in self.cells.values():
            if existing.activation_type in ("dendritic_product", "dendritic_product_n") and tuple(existing.dendritic_sources) == values:
                raise ValueError("feature already exists in graph")
        owner_cell_id = owner.cell_id
        old_incoming = [synapse_to_dict(synapse) for synapse in self.graph.iter_synapses() if synapse.destination == nursery_id]
        # A converted nursery cell may already feed several hidden-afferent
        # motor modules.  The install owns exactly one of those routes: the
        # candidate's non-owner motor afferents must be silenced, while every
        # other hidden->motor afferent remains byte-for-byte untouched.
        candidate_motor_afferents = [
            synapse_to_dict(synapse)
            for synapse in self.graph.iter_synapses()
            if synapse.source == nursery_id and self.cells[synapse.destination].kind == "motor_module"
        ]
        owner_edge = self.graph.get(nursery_id, owner_cell_id)
        add_owner_edge = owner_edge is None
        # A converted nursery can be the only sensor-to-output path.  Any
        # missing endpoint scaffold is added below under the same atomic
        # rollback; avoid charging a worst-case scaffold that may not be
        # needed for this graph.
        max_reachability_edges = 0
        prospective_synapses = len(self.graph.synapses) - len(old_incoming) + len(values) + (1 if add_owner_edge else 0) + max_reachability_edges
        prospective_energy = self.resources.energy_used + (len(values) + (1 if add_owner_edge else 0) + max_reachability_edges) * self.resources.synapse_cost
        if prospective_synapses > self.resources.max_synapses or prospective_energy > self.resources.energy_per_step + 1e-12:
            raise RuntimeError("feature install exceeds synapse or energy budget")
        graph_snapshot = {
            "synapses": [synapse_to_dict(synapse) for synapse in self.graph.iter_synapses()],
            "next_synapse_index": self.graph.next_synapse_index,
            "retired_ids": set(self.graph.retired_ids),
        }
        cell_snapshot = copy.deepcopy(nursery.__dict__)
        resource_snapshot = self.resources.to_dict()
        rng_snapshot = self.rng.getstate()
        owners_snapshot = dict(self.variable_order_feature_owners)
        extra_snapshot = {key: list(value) for key, value in self.variable_order_extra_sources.items()}
        try:
            for payload in old_incoming:
                self.graph.remove(str(payload["source"]), str(payload["destination"]))
                self.resources.removed_synapse()
            nursery.activation_type = "dendritic_product" if len(values) == 2 else "dendritic_product_n"
            nursery.dendritic_sources = values
            nursery.dendritic_normalizer = 1.0
            nursery.metadata["feature_protected"] = "direct"
            for source in values:
                self.graph.add(Synapse(source, nursery_id, 1.0, plasticity=0.0))
                self.resources.added_synapse()
            product_strength = max(-1.0, min(1.0, float(signed_local_credit)))
            for payload in candidate_motor_afferents:
                current = self.graph.get(str(payload["source"]), str(payload["destination"]))
                if current is None:
                    continue
                current.strength = product_strength if current.destination == owner_cell_id else 0.0
                # Product routes are structural, not actor-plastic edges.
                current.plasticity = 0.0
                current.actor_eligibility_trace = 0.0
            if owner_edge is None:
                self.graph.add(Synapse(nursery_id, owner_cell_id, product_strength, plasticity=0.0))
                self.resources.added_synapse()
            else:
                owner_edge.strength = product_strength
                owner_edge.plasticity = 0.0
            if owner.afferent_kind == "input":
                extra = self.variable_order_extra_sources.setdefault(owner.id, [])
                if nursery_id not in extra:
                    extra.append(nursery_id)
            for source in self.input_ids:
                reachable = {source}
                pending = [source]
                while pending:
                    current_source = pending.pop(0)
                    for destination in self.graph.outgoing.get(current_source, []):
                        if destination not in reachable:
                            reachable.add(destination)
                            pending.append(destination)
                if all(output not in reachable for output in self.output_ids):
                    for output in self.output_ids:
                        if self.graph.has(source, output):
                            continue
                        if not self.resources.can_add_synapse():
                            raise RuntimeError("variable-order reachability scaffold exceeds resource budget")
                        self.graph.add(Synapse(source, output, 0.0, plasticity=0.0))
                        self.resources.added_synapse()
            self.resources.counters["synapses"] = len(self.graph.synapses)
            # First validate graph/resource invariants without ownership; only
            # then publish the feature-owner mapping and validate that link.
            self.validate()
            self.variable_order_feature_owners[key] = owner.id
            self.validate()
            return nursery_id
        except Exception:
            self.graph.synapses.clear()
            self.graph.outgoing.clear()
            self.graph.incoming.clear()
            self.graph.next_synapse_index = int(graph_snapshot["next_synapse_index"])
            self.graph.retired_ids = set()
            for payload in graph_snapshot["synapses"]:
                self.graph.add(Synapse(**dict(payload)))
            self.graph.next_synapse_index = int(graph_snapshot["next_synapse_index"])
            self.graph.retired_ids = set(graph_snapshot["retired_ids"])
            nursery.__dict__.clear()
            nursery.__dict__.update(cell_snapshot)
            for payload in candidate_motor_afferents:
                current = self.graph.get(str(payload["source"]), str(payload["destination"]))
                if current is not None:
                    current.__dict__.update(Synapse(**dict(payload)).__dict__)
            for name, value in resource_snapshot.items():
                if name == "counters":
                    self.resources.counters = dict(value)
                else:
                    setattr(self.resources, name, value)
            self.rng.setstate(rng_snapshot)
            self.variable_order_feature_owners = owners_snapshot
            self.variable_order_extra_sources = extra_snapshot
            raise

    install_dendritic_product_n = install_dendritic_feature
    install_variable_order_feature = install_dendritic_feature

    def enable_context_modules(self, max_modules: int = 4, afferent_kind: str = "input") -> None:
        """Enable reward-inferred motor engrams and recruit the first module."""
        if max_modules < 1:
            raise ValueError("max_modules must be positive")
        if afferent_kind not in ("input", "hidden"):
            raise ValueError("afferent_kind must be input or hidden")
        if self.motor_modules and any(module.afferent_kind != afferent_kind for module in self.motor_modules.values()):
            raise ValueError("existing motor modules use a different afferent kind")
        self.context_enabled = True
        self.context_max_modules = int(max_modules)
        self.context_afferent_kind = afferent_kind
        # Context milestone intentionally freezes the legacy hidden/recurrent
        # learner; callers can explicitly re-enable it for old experiments.
        self.legacy_learning_enabled = False
        if not self.motor_modules:
            self.active_motor_module = self._recruit_motor_module().id

    def enable_motor_bootstrap(self, scale: float = 1.0) -> None:
        """Initialize hidden-afferent actor weights from organism RNG only."""
        if not math.isfinite(float(scale)) or not 0.0 <= scale <= 1.0:
            raise ValueError("motor bootstrap scale must be finite and in [0, 1]")
        self.motor_bootstrap_scale = float(scale)

    def enable_adaptive_dendritic_learning(self, proposal_interval: int = 8) -> None:
        """Enable reward-gated, target-blind hidden pair conversion."""
        if isinstance(proposal_interval, bool) or not isinstance(proposal_interval, int) or proposal_interval < 1:
            raise ValueError("adaptive dendritic proposal interval must be a positive integer")
        self.adaptive_dendritic_enabled = True
        self.adaptive_dendritic_proposal_interval = proposal_interval
        for cell in self.cells.values():
            if cell.kind == "hidden":
                cell.threshold = 0.0

    def _adaptive_dendritic_module_search(self, module_id: str) -> Tuple[List[Tuple[str, str]], bool]:
        tried = self.adaptive_dendritic_pairs_tried.setdefault(module_id, [])
        accepted = bool(self.adaptive_dendritic_module_accepted.get(module_id, False))
        normalized = []
        for pair in tried:
            value = tuple(str(item) for item in pair)
            if len(value) == 2 and value not in normalized:
                normalized.append(value)
        if normalized != tried:
            self.adaptive_dendritic_pairs_tried[module_id] = normalized
            tried = normalized
        return tried, accepted

    def enable_adaptive_dendritic_shadow_learning(self) -> None:
        """Enable delayed sign-symmetric local pair scoring for adaptive v9."""
        self.adaptive_dendritic_shadow_enabled = True
        # Shadow learning starts from the first normal action and owns the
        # same bounded checkpoints on every organism; these are not evaluator
        # acceptance thresholds.
        self.adaptive_dendritic_shadow_pending = None
        self.adaptive_dendritic_fingerprint_enabled = True
        self.adaptive_dendritic_fingerprint_window = 16
        self.context_detector_drift = 0.65
        self.context_detector_min_evidence = 64
        self.adaptive_dendritic_detector_hold_until = 0

    @staticmethod
    def _adaptive_dendritic_pair_key(pair: Sequence[str]) -> str:
        values = tuple(str(value) for value in pair)
        if len(values) != 2:
            raise ValueError("adaptive dendritic pair must contain two sources")
        return "%s|%s" % values

    @staticmethod
    def _adaptive_dendritic_pair_from_key(key: str) -> Tuple[str, str]:
        values = tuple(str(value) for value in str(key).split("|"))
        if len(values) != 2 or values[0] == values[1]:
            raise ValueError("adaptive dendritic pair key is invalid")
        return values

    def _adaptive_dendritic_restore_synapse(self, payload: Mapping[str, object]) -> None:
        """Restore a removed synapse with its original stable ID."""
        synapse = Synapse(**dict(payload))
        key = (synapse.source, synapse.destination)
        if key in self.graph.synapses:
            raise AssertionError("cannot restore occupied synapse: %s>%s" % key)
        self.graph.synapses[key] = synapse
        self.graph.outgoing.setdefault(synapse.source, []).append(synapse.destination)
        self.graph.incoming.setdefault(synapse.destination, []).append(synapse.source)
        self.graph.outgoing[synapse.source].sort()
        self.graph.incoming[synapse.destination].sort()
        self.graph.retired_ids.discard(synapse.id)

    def _adaptive_dendritic_rollback(self, proposal: Mapping[str, object]) -> None:
        """Restore proposal structure/policy state; retain new-ID tombstones."""
        for payload in list(proposal.get("new_synapses", [])) + list(proposal.get("reachability_synapses", [])):
            current = self.graph.get(str(payload["source"]), str(payload["destination"]))
            if current is not None:
                self.graph.remove(current.source, current.destination)
                self.resources.removed_synapse()
        for payload in proposal.get("old_synapses", []):
            self._adaptive_dendritic_restore_synapse(payload)
            self.resources.restored_synapse()
        cell = self.cells[str(proposal["cell_id"])]
        old_cell = dict(proposal["old_cell"])
        if isinstance(old_cell.get("dendritic_sources"), list):
            old_cell["dendritic_sources"] = tuple(old_cell["dendritic_sources"])
        cell.__dict__.update(old_cell)
        self.graph.next_synapse_index = max(self.graph.next_synapse_index, int(proposal["old_next_synapse_index"]))
        for payload in proposal.get("motor_afferents", []):
            current = self.graph.get(str(payload["source"]), str(payload["destination"]))
            if current is not None:
                current.__dict__.update(Synapse(**dict(payload)).__dict__)
        self.resources.counters["synapses"] = len(self.graph.synapses)
        self.validate()

    def _adaptive_dendritic_maybe_propose(self) -> None:
        """Try one atomic random hidden-cell pair conversion before propagation."""
        if self.adaptive_dendritic_shadow_enabled:
            return
        if not self.adaptive_dendritic_enabled or self.adaptive_dendritic_proposal is not None:
            return
        if self.step_count % self.adaptive_dendritic_proposal_interval:
            return
        # Search/probe actions are deliberately protected from structural
        # credit.  Existing proposals remain pending and are settled only by
        # their already-materialized local eligibility at the next outcome.
        if not self.context_enabled or self.context_state != "normal" or self.context_probe_module is not None:
            return
        # Accepted product cells are protected representations for now.  Only
        # additive cells may be converted, preventing an accepted feature from
        # being overwritten by a later random proposal.
        owner_module = self.motor_modules.get(self.active_motor_module)
        if owner_module is None:
            return
        tried, accepted = self._adaptive_dendritic_module_search(owner_module.id)
        pairs = list(combinations(self.input_ids, 2))
        if accepted or len(tried) >= len(pairs):
            return
        hidden_ids = [
            identifier for identifier in self._representation_hidden_ids()
            if self.cells[identifier].activation_type == "additive"
        ]
        if not hidden_ids or len(self.input_ids) < 2:
            return
        proposal_rng_state = self.rng.getstate()
        cell_id = self.rng.choice(hidden_ids)
        cell = self.cells[cell_id]
        occupied_pairs = {
            tuple(cell_value.dendritic_sources)
            for cell_value in self.cells.values()
            if cell_value.activation_type == "dendritic_product"
        }
        candidates = [pair for pair in pairs if tuple(pair) not in occupied_pairs and tuple(pair) not in tried]
        if not candidates:
            return
        new_sources = self.rng.choice(candidates)
        old_synapses = [synapse_to_dict(synapse) for synapse in self.graph.iter_synapses() if synapse.destination == cell_id]
        old_cell = dict(cell.__dict__)
        old_next_synapse_index = self.graph.next_synapse_index
        # At most one zero-strength input->output reachability scaffold per
        # sensor may be needed when the only existing path ran through the
        # additive candidate.  These are not motor-module edges and carry no
        # policy signal, but keep the graph's endpoint invariant intact.
        prospective_count = len(self.graph.synapses) - len(old_synapses) + 2 + len(self.input_ids)
        prospective_energy = self.resources.energy_used + (2 + len(self.input_ids)) * self.resources.synapse_cost
        if prospective_count > self.resources.max_synapses or prospective_energy > self.resources.energy_per_step + 1e-12:
            self.rng.setstate(proposal_rng_state)
            return
        proposal = {
            "cell_id": cell_id,
            "old_cell": old_cell,
            "old_synapses": old_synapses,
            "old_next_synapse_index": old_next_synapse_index,
            "motor_afferents": [
                synapse_to_dict(synapse)
                for synapse in self.graph.iter_synapses()
                if self.cells[synapse.destination].kind == "motor_module"
            ],
            "owner_module": owner_module.id,
            "min_trial_steps": self.adaptive_dendritic_min_evidence,
            "max_trial_steps": self.adaptive_dendritic_max_trial_steps,
            "trial_min_step": self.step_count + 1,
            "trial_max_step": self.step_count + self.adaptive_dendritic_max_trial_steps,
            "trial_step_count": 0,
            "evidence_sum": 0.0,
            "evidence_sumsq": 0.0,
            "evidence_count": 0,
            "trial_eligibilities": [],
            "trial_evidence": [],
            "pending_local_eligibility": 0.0,
            "last_outcome_step": None,
            "new_sources": list(new_sources),
            "new_synapses": [],
            "reachability_synapses": [],
            "old_activation": cell.activation,
            "new_activation": cell.activation,
            "counterfactual_activation": cell.activation,
            "counterfactual_drive": 0.0,
            "counterfactual_sources": {},
            "local_actor_eligibility": 0.0,
            "local_node_eligibility": 0.0,
            "local_eligibility": 0.0,
            "proposal_step": self.step_count,
            "outcome_applied": False,
        }
        try:
            # Insulate dormant/non-owner policies while this candidate is
            # explored.  The owner receives only target-blind RNG bootstrap.
            for payload in proposal["motor_afferents"]:
                current = self.graph.get(str(payload["source"]), str(payload["destination"]))
                if current is None:
                    continue
                if str(payload["destination"]) == owner_module.cell_id and self.motor_bootstrap_scale > 0.0:
                    current.strength = self.motor_bootstrap_scale if self.rng.random() < 0.5 else -self.motor_bootstrap_scale
                else:
                    current.strength = 0.0
                current.actor_eligibility_trace = 0.0
            for payload in old_synapses:
                self.graph.remove(str(payload["source"]), str(payload["destination"]))
                self.resources.removed_synapse()
            cell.activation_type = "dendritic_product"
            cell.dendritic_sources = tuple(new_sources)
            for source in new_sources:
                if not self.resources.can_add_synapse():
                    raise RuntimeError("adaptive dendritic synapse cost exceeds resource budget")
                self.graph.add(Synapse(source, cell_id, 1.0, plasticity=0.0))
                self.resources.added_synapse()
                proposal["new_synapses"].append(synapse_to_dict(self.graph.get(source, cell_id)))
            for source in self.input_ids:
                reachable = {source}
                pending = [source]
                while pending:
                    current_source = pending.pop(0)
                    for destination in self.graph.outgoing.get(current_source, []):
                        if destination not in reachable:
                            reachable.add(destination)
                            pending.append(destination)
                if all(output not in reachable for output in self.output_ids):
                    for output in self.output_ids:
                        if self.graph.has(source, output):
                            continue
                        if not self.resources.can_add_synapse():
                            raise RuntimeError("adaptive dendritic reachability scaffold exceeds resource budget")
                        self.graph.add(Synapse(source, output, 0.0, plasticity=0.0))
                        self.resources.added_synapse()
                        proposal["reachability_synapses"].append(synapse_to_dict(self.graph.get(source, output)))
            self.resources.counters["synapses"] = len(self.graph.synapses)
            self.validate()
        except (AssertionError, KeyError, RuntimeError, ValueError):
            self._adaptive_dendritic_rollback(proposal)
            self.rng.setstate(proposal_rng_state)
            return
        tried.append(tuple(new_sources))
        self.adaptive_dendritic_proposal = proposal
        self.events.append({
            "step": self.step_count,
            "kind": "adaptive_dendritic_proposed",
            "cell_id": cell_id,
            "new_sources": list(new_sources),
        })

    def _adaptive_dendritic_record_action_credit(self) -> None:
        """Materialize local actor/node evidence before any later outcome."""
        proposal = self.adaptive_dendritic_proposal
        if proposal is None:
            return
        cell_id = str(proposal["cell_id"])
        cell = self.cells[cell_id]
        counterfactual_activation = float(proposal["counterfactual_activation"])
        delta = cell.activation - counterfactual_activation
        actor_signal = 0.0
        owner = self.motor_modules.get(str(proposal.get("owner_module", "")))
        if owner is not None:
            for synapse in self._module_synapses(owner):
                if synapse.source != cell_id:
                    continue
                pre = self._motor_causal_presynaptic.get(owner.id, {}).get(cell_id, 0.0)
                if abs(pre) > 1e-12:
                    score_factor = synapse.actor_eligibility_trace / pre
                    actor_signal += synapse.strength * delta * score_factor
        node_signal = sum(
            self.graph.get(source, cell_id).node_eligibility_trace
            for source in self.graph.incoming.get(cell_id, [])
            if self.graph.get(source, cell_id) is not None
        )
        proposal["new_activation"] = cell.activation
        proposal["local_actor_eligibility"] = actor_signal
        proposal["local_node_eligibility"] = node_signal
        proposal["local_eligibility"] = actor_signal + 0.25 * node_signal
        proposal["pending_local_eligibility"] = proposal["local_eligibility"]

    def _apply_adaptive_dendritic_outcome(self, reward_prediction_error: float) -> None:
        """Accumulate one delayed local outcome and settle by confidence bounds."""
        proposal = self.adaptive_dendritic_proposal
        if proposal is None or bool(proposal.get("outcome_applied", False)):
            return
        # One organism outcome can reach this hook only once.  Retain a
        # step-token guard as well so direct/reentrant callers cannot count a
        # delayed reward twice.
        if proposal.get("last_outcome_step") == self.step_count:
            return
        proposal["last_outcome_step"] = self.step_count
        local_eligibility = float(proposal.get("pending_local_eligibility", proposal.get("local_eligibility", 0.0)))
        credit = float(reward_prediction_error) * local_eligibility
        proposal["trial_step_count"] = int(proposal.get("trial_step_count", 0)) + 1
        proposal["evidence_sum"] = float(proposal.get("evidence_sum", 0.0)) + credit
        proposal["evidence_sumsq"] = float(proposal.get("evidence_sumsq", 0.0)) + credit * credit
        proposal["evidence_count"] = int(proposal.get("evidence_count", 0)) + 1
        proposal.setdefault("trial_eligibilities", []).append(local_eligibility)
        proposal.setdefault("trial_evidence", []).append(credit)
        proposal["pending_local_eligibility"] = 0.0
        count = int(proposal["evidence_count"])
        mean = float(proposal["evidence_sum"]) / float(count)
        variance = max(0.0, float(proposal["evidence_sumsq"]) / float(count) - mean * mean)
        standard_error = math.sqrt(variance / float(max(1, count)))
        margin = self.adaptive_dendritic_confidence_z * standard_error
        lower = mean - margin
        upper = mean + margin
        accept = count >= self.adaptive_dendritic_min_evidence and lower > 0.0
        reject = count >= self.adaptive_dendritic_min_evidence and upper < 0.0
        if self.step_count >= int(proposal["trial_max_step"]):
            accept = lower > 0.0
            reject = not accept
        if accept:
            proposal["outcome_applied"] = True
            self.adaptive_dendritic_module_accepted[str(proposal["owner_module"])] = True
            self.adaptive_dendritic_accept_count += 1
            self.events.append({
                "step": self.step_count,
                "kind": "adaptive_dendritic_accepted",
                "cell_id": proposal["cell_id"],
                "new_sources": list(proposal["new_sources"]),
                "credit": credit,
                "evidence_count": count,
                "evidence_mean": mean,
                "evidence_lower": lower,
                "evidence_upper": upper,
            })
            self.adaptive_dendritic_proposal = None
        elif reject:
            cell_id = str(proposal["cell_id"])
            new_sources = list(proposal["new_sources"])
            # Clear ownership before validation: rollback restores additive
            # state, so the pending proposal must no longer describe it.
            self.adaptive_dendritic_proposal = None
            self._adaptive_dendritic_rollback(proposal)
            self.adaptive_dendritic_reject_count += 1
            self.events.append({
                "step": self.step_count,
                "kind": "adaptive_dendritic_reverted",
                "cell_id": cell_id,
                "new_sources": new_sources,
                "credit": credit,
                "evidence_count": count,
                "evidence_mean": mean,
                "evidence_lower": lower,
                "evidence_upper": upper,
            })
        else:
            self.events.append({
                "step": self.step_count,
                "kind": "adaptive_dendritic_trial_evidence",
                "cell_id": proposal["cell_id"],
                "evidence_count": count,
                "evidence_mean": mean,
                "evidence_lower": lower,
                "evidence_upper": upper,
            })

    def _module_synapses(self, module: MotorModule) -> List[Synapse]:
        if module.afferent_kind == "input":
            extra = self.variable_order_extra_sources.get(module.id)
            sources = self.input_ids if not extra else list(self.input_ids) + [
                source for source in extra if source not in self.input_ids]
        else:
            sources = [
                identifier for identifier, cell in sorted(self.cells.items()) if cell.kind == "hidden"
            ]
        return [
            synapse for synapse in self.graph.iter_synapses()
            if synapse.destination == module.cell_id and synapse.source in sources
        ]

    def _module_source_ids(self, afferent_kind: str) -> List[str]:
        if afferent_kind == "input":
            return list(self.input_ids)
        if afferent_kind == "hidden":
            return sorted(identifier for identifier, cell in self.cells.items() if cell.kind == "hidden")
        raise ValueError("afferent_kind must be input or hidden")

    def _module_mean_from_current_inputs(self, module: MotorModule) -> float:
        """Evaluate a motor engram without touching its cell state."""
        drive = sum(self.cells[synapse.source].activation * synapse.strength for synapse in self._module_synapses(module))
        cell = self.cells[module.cell_id]
        centered_drive = drive - cell.threshold
        return math.tanh(centered_drive / (1.0 + cell.adaptation))

    def _context_feature_values(self, action: float) -> Tuple[float, ...]:
        """Build detector features from every current sensor coordinate.

        The original scalar-cue milestone used six values.  Keeping feature
        construction here, rather than indexing a particular cue in the
        detector, lets multi-context policies use the independent distractor
        sensor without exposing context or changing the learning rule.
        """
        inputs = tuple(float(self.cells[identifier].activation) for identifier in self.input_ids)
        action = float(action)
        return (1.0,) + inputs + (action,) + tuple(value * action for value in inputs) + tuple(value * value for value in inputs) + (action * action,)

    def _context_feature_dimension(self) -> int:
        return 3 + 3 * len(self.input_ids)

    def _recruit_motor_module(self, afferent_kind: Optional[str] = None) -> MotorModule:
        if len(self.motor_modules) >= self.context_max_modules:
            raise RuntimeError("motor module cap reached")
        index = len(self.motor_modules)
        module_id = "motor-module-%03d" % index
        cell_id = module_id
        afferent_kind = afferent_kind or self.context_afferent_kind
        source_ids = self._module_source_ids(afferent_kind)
        module = MotorModule(module_id, cell_id, afferent_kind=afferent_kind)
        module.detector_predictor = [0.0] * self._context_feature_dimension()
        # Creation is charged against the current structural budget.  Before
        # the first decision there is no step energy yet, so seed the current
        # baseline explicitly; subsequent recruitment occurs inside a step.
        required_synapses = len(source_ids)
        current_baseline = self.resources.baseline_cost(len(self.cells), len(self.graph.synapses))
        prospective_baseline = self.resources.baseline_cost(len(self.cells) + 1, len(self.graph.synapses) + required_synapses)
        if (
            current_baseline + self.resources.cell_cost > self.resources.energy_per_step + 1e-12
            or self.resources.counters.get("synapses", 0) + required_synapses > self.resources.max_synapses
            or len(self.cells) + 1 > self.resources.max_cells
            or prospective_baseline > self.resources.energy_per_step + 1e-12
        ):
            raise RuntimeError("resource budget cannot recruit motor module")
        self.resources.energy_used = current_baseline
        self.cells[cell_id] = Cell(cell_id, "motor_module", target_activity=0.25)
        self.resources.added_cell()
        for source in source_ids:
            strength = 0.0
            if afferent_kind == "hidden" and self.motor_bootstrap_scale > 0.0:
                strength = self.motor_bootstrap_scale if self.rng.random() < 0.5 else -self.motor_bootstrap_scale
            self.graph.add(Synapse(source, cell_id, strength, plasticity=1.0))
            self.resources.added_synapse()
        self.motor_modules[module_id] = module
        self.adaptive_dendritic_pairs_tried.setdefault(module_id, [])
        self.adaptive_dendritic_module_accepted.setdefault(module_id, False)
        self.adaptive_dendritic_shadow_evidence.setdefault(module_id, {})
        self.adaptive_dendritic_shadow_search_exhausted.setdefault(module_id, False)
        self.events.append({"step": self.step_count, "kind": "motor_module_recruited", "module_id": module_id, "cell_id": cell_id})
        return module

    def _context_output(self, modulators: Modulators) -> Tuple[float, ...]:
        module = self.motor_modules[self.active_motor_module]
        probing = self.context_probe_module == module.id
        fingerprinting = self.context_state == "fingerprint" and (
            self.adaptive_dendritic_fingerprint_active or self.variable_order_fingerprint_active
        )
        # Dormant candidates remain dormant throughout a probe.  Their
        # proposed motor mean is computed directly from current sensor
        # activity, so probing cannot advance their cell dynamics.
        mu = self._module_mean_from_current_inputs(module) if probing else (
            0.0 if fingerprinting else self.cells[module.cell_id].activation
        )
        mode = "fingerprint" if fingerprinting else ("search" if probing or self.context_state == "search" else self.context_state)
        if self.context_state == "warning" and not probing:
            span = max(1e-9, self.context_switch_threshold - self.context_warning_threshold)
            # Preserve at least half of the declared deterministic motor
            # command and smoothly attenuate toward that safety cap as
            # evidence rises.  Stronger evidence is handled by escalation
            # and protected probing rather than by further motor distortion.
            attenuation = 0.5 + 0.5 * max(0.0, min(1.0, 1.0 - (module.detector_cusum - self.context_warning_threshold) / span))
            mu = self.context_safe_action + (mu - self.context_safe_action) * attenuation
            self.events.append({
                "step": self.step_count,
                "kind": "motor_safe_action_attenuated",
                "module_id": module.id,
                "attenuation": attenuation,
                "detector_cusum": module.detector_cusum,
            })
        xi = self._exploration_value()
        exploration_sigma = float(modulators.exploration)
        self.context_pending_exploration_value = xi
        self.context_pending_exploration_sigma = exploration_sigma
        sigma = exploration_sigma
        if self.context_state == "warning" or probing or self.context_state == "search":
            # Consume the evaluator tape for alignment, but do not inject
            # random motor noise while hedging or diagnostically probing.
            sigma = 0.0
        for candidate in self.motor_modules.values():
            for synapse in self._module_synapses(candidate):
                synapse.actor_eligibility_trace = 0.0
        if sigma > 1e-12 and mode == "normal":
            for synapse in self._module_synapses(module):
                if module.afferent_kind == "hidden":
                    # Hidden modules were driven on final propagation pass by
                    # penultimate ``previous`` activations.  Credit that exact
                    # causal snapshot, not hidden cells after final dynamics.
                    pre = self._motor_causal_presynaptic.get(module.id, {}).get(synapse.source, 0.0)
                else:
                    # Preserve input-afferent actor behavior byte-for-byte.
                    pre = self.cells[synapse.source].activation
                synapse.actor_eligibility_trace = pre * (1.0 - mu * mu) * xi / sigma
        self.pending_motor_module = module.id
        self.context_pending_mode = mode
        action = max(-1.0, min(1.0, mu + sigma * xi))
        self.context_pending_action = action
        self.context_pending_features = self._context_feature_values(action)
        self.context_pending_prediction = sum(weight * value for weight, value in zip(module.detector_predictor, self.context_pending_features))
        self.context_pending_scale = math.sqrt(max(1e-5, module.detector_variance))
        if probing:
            incumbent = self.motor_modules.get(self.context_probe_incumbent_module) if self.context_probe_incumbent_module is not None else None
            incumbent_mean = self._module_mean_from_current_inputs(incumbent) if incumbent is not None else 0.0
            self.context_probe_candidate_mean = mu
            self.context_probe_incumbent_mean = incumbent_mean
            self.context_probe_exploration = exploration_sigma
        return (action,)

    def _adaptive_dendritic_cancel(self, reason: str) -> None:
        """Cancel a trial before an unrelated context reward can settle it."""
        proposal = self.adaptive_dendritic_proposal
        if proposal is None:
            return
        cell_id = str(proposal["cell_id"])
        owner_id = str(proposal.get("owner_module", ""))
        self.adaptive_dendritic_proposal = None
        self._adaptive_dendritic_rollback(proposal)
        self.adaptive_dendritic_reject_count += 1
        self.events.append({
            "step": self.step_count,
            "kind": "adaptive_dendritic_cancelled",
            "cell_id": cell_id,
            "owner_module": owner_id,
            "reason": str(reason),
        })

    def _adaptive_dendritic_shadow_record_pending(self) -> None:
        """Record pair eligibilities for the just-produced normal action."""
        if not self.adaptive_dendritic_shadow_enabled or self.adaptive_dendritic_shadow_pending is not None:
            return
        if not self.context_enabled or self.context_state != "normal" or self.context_probe_module is not None:
            return
        owner_id = self.pending_motor_module
        owner = self.motor_modules.get(owner_id) if owner_id is not None else None
        sigma = float(getattr(self, "context_pending_exploration_sigma", 0.0))
        if owner is None or self.context_pending_mode != "normal" or sigma <= 1e-12:
            return
        occupied_pairs = {
            tuple(cell.dendritic_sources)
            for cell in self.cells.values()
            if cell.activation_type == "dendritic_product"
        }
        mu = self.cells[owner.cell_id].activation
        xi = float(getattr(self, "context_pending_exploration_value", 0.0))
        factor = (1.0 - mu * mu) * xi / sigma
        entries = []
        for left, right in combinations(self.input_ids, 2):
            pair = (str(left), str(right))
            if pair in occupied_pairs:
                continue
            z = math.tanh(self.cells[left].activation * self.cells[right].activation)
            entries.append({"pair": list(pair), "eligibility": z * factor})
        self.adaptive_dendritic_shadow_pending = {
            "owner_module": owner.id,
            "step": self.step_count,
            "module_mean": mu,
            "exploration_value": xi,
            "exploration_sigma": sigma,
            "eligibilities": entries,
        }
        self.adaptive_dendritic_shadow_action_count += 1

    def _adaptive_dendritic_shadow_install(self, owner: MotorModule, pair: Tuple[str, str], signed_sum: float, score: float, count: int) -> bool:
        """Atomically convert the lowest-utility additive nursery cell."""
        hidden_ids = [
            identifier for identifier in self._representation_hidden_ids()
            if self.cells[identifier].activation_type == "additive"
        ]
        if not hidden_ids:
            return False
        def utility(identifier: str) -> Tuple[float, str]:
            synapse = self.graph.get(identifier, owner.cell_id)
            return (synapse.utility if synapse is not None else 0.0, identifier)
        cell_id = min(hidden_ids, key=utility)
        cell = self.cells[cell_id]
        old_synapses = [synapse_to_dict(synapse) for synapse in self.graph.iter_synapses() if synapse.destination == cell_id]
        old_cell = dict(cell.__dict__)
        owner_strength = max(-1.0, min(1.0, self.actor_learning_rate * signed_sum))
        proposal = {
            "cell_id": cell_id,
            "old_cell": old_cell,
            "old_synapses": old_synapses,
            "old_next_synapse_index": self.graph.next_synapse_index,
            "motor_afferents": [
                synapse_to_dict(synapse)
                for synapse in self.graph.iter_synapses()
                if self.cells[synapse.destination].kind == "motor_module"
            ],
            "new_sources": list(pair),
            "new_synapses": [],
            "reachability_synapses": [],
        }
        prospective_count = len(self.graph.synapses) - len(old_synapses) + 2 + len(self.input_ids)
        prospective_energy = self.resources.energy_used + (2 + len(self.input_ids)) * self.resources.synapse_cost
        if prospective_count > self.resources.max_synapses or prospective_energy > self.resources.energy_per_step + 1e-12:
            return False
        try:
            for payload in proposal["motor_afferents"]:
                source = str(payload["source"])
                destination = str(payload["destination"])
                # Installing a product changes only the product feature's
                # route.  Preserve every unrelated motor afferent, including
                # its eligibility trace and other synapse metadata.
                if source != cell_id:
                    continue
                current = self.graph.get(source, destination)
                if current is None:
                    continue
                current.strength = owner_strength if destination == owner.cell_id else 0.0
                current.actor_eligibility_trace = 0.0
            for payload in old_synapses:
                self.graph.remove(str(payload["source"]), str(payload["destination"]))
                self.resources.removed_synapse()
            cell.activation_type = "dendritic_product"
            cell.dendritic_sources = tuple(pair)
            for source in pair:
                if not self.resources.can_add_synapse():
                    raise RuntimeError("adaptive dendritic synapse cost exceeds resource budget")
                self.graph.add(Synapse(source, cell_id, 1.0, plasticity=0.0))
                self.resources.added_synapse()
                proposal["new_synapses"].append(synapse_to_dict(self.graph.get(source, cell_id)))
            for source in self.input_ids:
                reachable = {source}
                pending = [source]
                while pending:
                    current_source = pending.pop(0)
                    for destination in self.graph.outgoing.get(current_source, []):
                        if destination not in reachable:
                            reachable.add(destination)
                            pending.append(destination)
                if all(output not in reachable for output in self.output_ids):
                    for output in self.output_ids:
                        if self.graph.has(source, output):
                            continue
                        if not self.resources.can_add_synapse():
                            raise RuntimeError("adaptive dendritic reachability scaffold exceeds resource budget")
                        self.graph.add(Synapse(source, output, 0.0, plasticity=0.0))
                        self.resources.added_synapse()
                        proposal["reachability_synapses"].append(synapse_to_dict(self.graph.get(source, output)))
            self.resources.counters["synapses"] = len(self.graph.synapses)
            self.validate()
        except (AssertionError, KeyError, RuntimeError, ValueError):
            self._adaptive_dendritic_rollback(proposal)
            return False
        tried = self.adaptive_dendritic_pairs_tried.setdefault(owner.id, [])
        if tuple(pair) not in tried:
            tried.append(tuple(pair))
        self.adaptive_dendritic_module_accepted[owner.id] = True
        self.adaptive_dendritic_shadow_install_count += 1
        self.adaptive_dendritic_accept_count += 1
        self.adaptive_dendritic_pair_owners[self._adaptive_dendritic_pair_key(pair)] = owner.id
        owner.detector_cusum = 0.0
        owner.detector_frozen = False
        owner.surprise_evidence = 0.0
        owner.negative_surprise = 0.0
        owner.negative_streak = 0
        owner.confidence = 0.0
        self.adaptive_dendritic_detector_hold_until = max(self.adaptive_dendritic_detector_hold_until, self.step_count + 64)
        self.events.append({
            "step": self.step_count,
            "kind": "adaptive_dendritic_shadow_installed",
            "cell_id": cell_id,
            "owner_module": owner.id,
            "new_sources": list(pair),
            "signed_sum": signed_sum,
            "score": score,
            "evidence_count": count,
            "owner_strength": owner_strength,
        })
        return True

    def _adaptive_dendritic_begin_fingerprint(self) -> None:
        """Enter a neutral, fixed-length context fingerprint episode."""
        if not self.adaptive_dendritic_fingerprint_enabled or self.adaptive_dendritic_fingerprint_active:
            return
        incumbent = self.motor_modules.get(self.active_motor_module)
        if incumbent is None:
            return
        self.adaptive_dendritic_fingerprint_active = True
        self.adaptive_dendritic_fingerprint_owner = incumbent.id
        self.adaptive_dendritic_fingerprint_incumbent = incumbent.id
        self.adaptive_dendritic_fingerprint_start_step = self.step_count
        self.adaptive_dendritic_fingerprint_deadline = self.step_count + self.adaptive_dendritic_fingerprint_window + (self.delayed_credit_delay if self.delayed_credit_enabled else 0)
        self.adaptive_dendritic_fingerprint_count = 0
        self.adaptive_dendritic_fingerprint_evidence = {}
        self.adaptive_dendritic_fingerprint_pending = None
        self.context_probe_module = None
        self.context_probe_queue = []
        self.context_probe_remaining = 0
        self.context_probe_incumbent_module = None
        self.context_probe_origin = None
        incumbent.dormant = False
        incumbent.protected = True
        self.context_state = "fingerprint"
        self.events.append({
            "step": self.step_count,
            "kind": "adaptive_dendritic_fingerprint_started",
            "module_id": incumbent.id,
            "deadline": self.adaptive_dendritic_fingerprint_deadline,
        })

    def _adaptive_dendritic_fingerprint_record_pending(self) -> None:
        """Record all pair eligibilities for one neutral diagnostic action."""
        if not self.adaptive_dendritic_fingerprint_active or self.adaptive_dendritic_fingerprint_pending is not None:
            return
        owner = self.adaptive_dendritic_fingerprint_owner
        sigma = float(self.context_pending_exploration_sigma)
        if owner is None or self.context_pending_mode != "fingerprint":
            return
        xi = float(self.context_pending_exploration_value)
        entries = []
        if sigma > 1e-12:
            for left, right in combinations(self.input_ids, 2):
                z = math.tanh(self.cells[left].activation * self.cells[right].activation)
                entries.append({
                    "pair": [str(left), str(right)],
                    "eligibility": z * xi / sigma,
                })
        # With no exploration there is no local perturbation credit.  Still
        # record a neutral, empty fingerprint outcome so the fixed 16-outcome
        # episode terminates rather than remaining active forever or inventing
        # a pair from zero evidence.
        self.adaptive_dendritic_fingerprint_pending = {
            "owner_module": owner,
            "step": self.step_count,
            "exploration_value": xi,
            "exploration_sigma": sigma,
            "eligibilities": entries,
        }

    def _adaptive_dendritic_fingerprint_route(self) -> None:
        """Resolve a completed fingerprint using only installed pair owners."""
        if not self.adaptive_dendritic_fingerprint_active:
            return
        candidates = []
        for key, value in self.adaptive_dendritic_fingerprint_evidence.items():
            total = float(value.get("sum", 0.0))
            sumsq = float(value.get("sumsq", 0.0))
            count = int(value.get("count", 0.0))
            if count <= 0:
                continue
            score = abs(total) / math.sqrt(max(1e-12, sumsq))
            candidates.append((score, self._adaptive_dendritic_pair_from_key(key), total, count))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        pair = candidates[0][1] if candidates else None
        signed_sum = candidates[0][2] if candidates else 0.0
        score = candidates[0][0] if candidates else 0.0
        count = candidates[0][3] if candidates else 0
        owner_id = self.adaptive_dendritic_pair_owners.get(self._adaptive_dendritic_pair_key(pair)) if pair else None
        installed = False
        incumbent_id = self.adaptive_dendritic_fingerprint_incumbent
        if owner_id not in self.motor_modules and pair is not None and incumbent_id in self.motor_modules:
            # An unowned feature gets a fresh local engram.  Installing it on
            # the incumbent would silently merge contexts and defeat the
            # purpose of representation-addressed routing.
            owner = None
            recruit_checkpoint = self.state_dict()
            if len(self.motor_modules) < self.context_max_modules:
                try:
                    owner = self._recruit_motor_module()
                except RuntimeError:
                    owner = None
            if owner is not None:
                installed = self._adaptive_dendritic_shadow_install(owner, pair, signed_sum, score, count)
                if not installed:
                    # Recruitment and product installation form one logical
                    # operation. Restore the pre-recruit checkpoint so a
                    # failed product edit cannot consume a module/cell slot
                    # or leave resource counters permanently inflated.
                    restored = type(self).from_state_dict(recruit_checkpoint)
                    self.__dict__.clear()
                    self.__dict__.update(restored.__dict__)
                    owner = None
                owner_id = self.adaptive_dendritic_pair_owners.get(self._adaptive_dendritic_pair_key(pair))
        target_id = owner_id if owner_id in self.motor_modules else incumbent_id
        if target_id not in self.motor_modules:
            target_id = self.active_motor_module
        previous_id = self.active_motor_module
        for module in self.motor_modules.values():
            module.dormant = module.id != target_id
            module.protected = False
        target = self.motor_modules[target_id]
        target.dormant = False
        target.detector_cusum = 0.0
        target.detector_frozen = False
        target.surprise_evidence = 0.0
        target.negative_surprise = 0.0
        target.negative_streak = 0
        # Confidence is learned policy state, not part of the transient alarm
        # episode. Preserve it for incumbent and dormant pair owners alike.
        self.active_motor_module = target_id
        self.context_state = "normal"
        self.context_probe_module = None
        self.context_probe_queue = []
        self.context_probe_remaining = 0
        self.context_probe_incumbent_module = None
        self.context_probe_origin = None
        self.adaptive_dendritic_detector_hold_until = self.step_count + 64
        self.adaptive_dendritic_fingerprint_active = False
        self.adaptive_dendritic_fingerprint_owner = None
        self.adaptive_dendritic_fingerprint_pending = None
        event = {
            "step": self.step_count,
            "kind": "adaptive_dendritic_fingerprint_resolved",
            "module_id": target_id,
            "pair": list(pair) if pair else None,
            "owner_module": owner_id,
            "incumbent_module": incumbent_id,
            "previous_module": previous_id,
            "signed_sum": signed_sum,
            "score": score,
            "evidence_count": count,
            "installed": installed,
        }
        self.events.append(event)
        if target_id != previous_id or owner_id == target_id:
            self.events.append({
                "step": self.step_count,
                "kind": "motor_module_reactivated",
                "module_id": target_id,
                "fingerprint": True,
            })

    def _adaptive_dendritic_fingerprint_accumulate(self, reward: float) -> None:
        """Consume one delayed raw-reward fingerprint outcome exactly once."""
        pending = self.adaptive_dendritic_fingerprint_pending
        self.adaptive_dendritic_fingerprint_pending = None
        if pending is None or not self.adaptive_dendritic_fingerprint_active:
            return
        reward = float(reward)
        for payload in pending.get("eligibilities", []):
            pair = tuple(str(value) for value in payload["pair"])
            key = self._adaptive_dendritic_pair_key(pair)
            value = self.adaptive_dendritic_fingerprint_evidence.setdefault(key, {"sum": 0.0, "sumsq": 0.0, "count": 0.0})
            credit = reward * float(payload["eligibility"])
            value["sum"] += credit
            value["sumsq"] += credit * credit
            value["count"] += 1.0
        self.adaptive_dendritic_fingerprint_count += 1
        self.events.append({
            "step": self.step_count,
            "kind": "adaptive_dendritic_fingerprint_evidence",
            "count": self.adaptive_dendritic_fingerprint_count,
            "owner_module": pending.get("owner_module"),
        })
        if self.adaptive_dendritic_fingerprint_count >= self.adaptive_dendritic_fingerprint_window:
            self._adaptive_dendritic_fingerprint_route()

    def _adaptive_dendritic_shadow_accumulate(self, module_rpe: float) -> None:
        """Consume one pending action's pair eligibilities exactly once."""
        pending = self.adaptive_dendritic_shadow_pending
        self.adaptive_dendritic_shadow_pending = None
        if pending is None:
            return
        owner_id = str(pending["owner_module"])
        evidence = self.adaptive_dendritic_shadow_evidence.setdefault(owner_id, {})
        for payload in pending.get("eligibilities", []):
            pair = tuple(str(value) for value in payload["pair"])
            key = self._adaptive_dendritic_pair_key(pair)
            cell = evidence.setdefault(key, {"sum": 0.0, "sumsq": 0.0, "count": 0.0})
            credit = float(module_rpe) * float(payload["eligibility"])
            cell["sum"] += credit
            cell["sumsq"] += credit * credit
            cell["count"] += 1.0
        owner = self.motor_modules.get(owner_id)
        if owner is None or self.adaptive_dendritic_module_accepted.get(owner_id, False):
            return
        if self.adaptive_dendritic_shadow_search_exhausted.get(owner_id, False):
            return
        occupied_pairs = {
            tuple(cell.dendritic_sources)
            for cell in self.cells.values()
            if cell.activation_type == "dendritic_product"
        }
        candidates = []
        for key, cell in evidence.items():
            pair = self._adaptive_dendritic_pair_from_key(key)
            if pair in occupied_pairs or float(cell["count"]) < 1.0:
                continue
            count = int(cell["count"])
            mean = float(cell["sum"]) / count
            variance = max(0.0, float(cell["sumsq"]) / count - mean * mean)
            stderr = math.sqrt(variance / float(max(1, count)))
            z = math.inf if stderr <= 1e-12 and abs(mean) > 1e-12 else (abs(mean) / stderr if stderr > 1e-12 else 0.0)
            score = abs(float(cell["sum"])) / math.sqrt(max(1e-12, float(cell["sumsq"])))
            candidates.append((score, z, pair, float(cell["sum"]), count))
        if not candidates:
            return
        candidates.sort(key=lambda item: (-item[0], item[2]))
        best = candidates[0]
        second_score = candidates[1][0] if len(candidates) > 1 else 0.0
        credible = best[1] >= self.adaptive_dendritic_shadow_confidence_z and best[0] - second_score >= self.adaptive_dendritic_shadow_separation
        checkpoints = self.adaptive_dendritic_shadow_checkpoints
        checkpoint = max((value for value in checkpoints if best[4] >= value), default=None)
        if checkpoint is None:
            return
        if credible or best[4] >= max(checkpoints):
            if credible:
                self._adaptive_dendritic_shadow_install(owner, best[2], best[3], best[0], best[4])
            else:
                self.adaptive_dendritic_shadow_search_exhausted[owner_id] = True

    def _apply_context_reward(self, reward_prediction_error: float, structural_trial: bool = False) -> None:
        module_id = self.pending_motor_module
        if module_id is None:
            return
        module = self.motor_modules[module_id]
        action_mode = self.context_pending_mode
        fingerprinting = action_mode == "fingerprint"
        reward_value = reward_prediction_error + self.reward_baseline
        module_rpe = reward_value - module.baseline
        self._evidence_shadow_update(module_id, reward_value, from_probe=self.context_probe_module == module_id)
        for synapse in self._module_synapses(module):
            if self.context_probe_module != module_id and action_mode == "normal" and not fingerprinting:
                synapse.strength += self.actor_learning_rate * synapse.actor_eligibility_trace * module_rpe
            synapse.strength = max(-1.0, min(1.0, synapse.strength))
            synapse.actor_eligibility_trace = 0.0
        if structural_trial:
            # Structural candidate actions may train the owning actor, but
            # cannot alter detector statistics, module confidence/baseline,
            # or warning/search state.  This keeps context inference local to
            # settled normal policy outcomes.
            self.pending_motor_module = None
            self.context_pending_features = ()
            self.context_pending_action = 0.0
            self.context_pending_mode = "normal"
            self.context_pending_prediction = 0.0
            self.context_pending_scale = 0.1
            self.context_pending_exploration_value = 0.0
            self.context_pending_exploration_sigma = 0.0
            return
        self.context_last_reward_value = reward_value
        module.total_reward += reward_value
        module.reward_count += 1
        module.baseline += 0.05 * module_rpe
        is_probe = self.context_probe_module == module_id
        standardized_surprise = 0.0
        if not is_probe:
            # Page-Hinkley/CUSUM-style evidence is learned only while this
            # module is the normal active policy.  The slow expected reward
            # and variance are therefore not contaminated by candidate
            # probes from another context.
            features = self.context_pending_features or self._context_feature_values(self.context_pending_action)
            if len(module.detector_predictor) != len(features):
                raise ValueError("motor detector predictor feature dimension mismatch")
            if module.detector_count == 0:
                module.detector_predictor = [0.0] * len(features)
                module.detector_predictor[0] = reward_value
                module.detector_mean = reward_value
                module.detector_variance = 0.01
            prediction = self.context_pending_prediction if self.context_pending_features else sum(weight * value for weight, value in zip(module.detector_predictor, features))
            detector_delta = reward_value - prediction
            if not module.detector_frozen and action_mode == "normal":
                norm = 1.0 + sum(value * value for value in features)
                rate = self.context_detector_predictor_rate / norm
                module.detector_predictor = [weight + rate * detector_delta * value for weight, value in zip(module.detector_predictor, features)]
                module.detector_mean = sum(weight * value for weight, value in zip(module.detector_predictor, features))
                module.detector_variance = max(
                    1e-5,
                    self.context_surprise_leak * module.detector_variance
                    + (1.0 - self.context_surprise_leak) * detector_delta * detector_delta,
                )
            else:
                module.detector_mean = prediction
            module.detector_count += 1
            scale = self.context_pending_scale if self.context_pending_scale > 0.0 else math.sqrt(module.detector_variance)
            standardized_surprise = detector_delta / scale
            negative_deviation = max(0.0, -standardized_surprise - self.context_detector_drift)
            positive_recovery = max(0.0, standardized_surprise)
            module.detector_cusum = max(
                0.0,
                self.context_surprise_leak * module.detector_cusum
                + negative_deviation
                - self.context_detector_recovery * positive_recovery,
            )
            if module.detector_cusum >= self.context_detector_freeze_threshold:
                module.detector_frozen = True
            module.reward_mean = module.detector_mean
            module.reward_variance = module.detector_variance
            module.surprise_evidence = module.detector_cusum
            module.negative_surprise = module.detector_cusum
            module.negative_streak = module.negative_streak + 1 if standardized_surprise < -0.5 else max(0, module.negative_streak - 1)
            self.context_surprise_count += 1
            global_surprise_delta = reward_value - self.context_surprise_mean
            self.context_surprise_mean += self.context_surprise_drift * global_surprise_delta
            self.context_surprise_variance = max(
                1e-5,
                self.context_surprise_leak * self.context_surprise_variance
                + (1.0 - self.context_surprise_leak) * global_surprise_delta * global_surprise_delta,
            )
        module.confidence = max(0.0, min(1.0, 0.98 * module.confidence + 0.02 * (1.0 if module_rpe > 0.0 else 0.0)))
        if is_probe:
            module.probe_count += 1
            self.context_probe_steps_seen += 1
            disagreement = abs(self.context_probe_candidate_mean - self.context_probe_incumbent_mean)
            # Warning/search probes deliberately suppress motor noise (sigma=0)
            # while still receiving the caller's exploration modulation.  That
            # modulation is not an information-noise estimate, so it must not
            # raise the evidence threshold and discard valid deterministic
            # candidate/incumbent disagreements.
            informative_threshold = self.context_probe_information_threshold
            informative = disagreement >= informative_threshold or abs(self.context_probe_candidate_mean) >= informative_threshold
            if informative:
                self.context_probe_informative_count += 1
                self.context_probe_rewards.append(reward_value)
                self.context_probe_sum += reward_value
                self.context_probe_sumsq += reward_value * reward_value
                self.context_probe_count += 1
                self.events.append({
                    "step": self.step_count,
                    "kind": "motor_probe_evidence",
                    "module_id": module.id,
                    "candidate_mean": self.context_probe_candidate_mean,
                    "incumbent_mean": self.context_probe_incumbent_mean,
                    "reward": reward_value,
                    "informative_count": self.context_probe_informative_count,
                    "steps_seen": self.context_probe_steps_seen,
                })
            self.context_probe_remaining = max(0, self.context_probe_remaining - 1)
            if self.context_probe_count >= self.context_probe_min_steps:
                mean_probe = self.context_probe_sum / float(self.context_probe_count)
                variance = max(0.0, self.context_probe_sumsq / float(self.context_probe_count) - mean_probe * mean_probe)
                standard_error = math.sqrt(variance / float(max(1, self.context_probe_count)))
                lower = mean_probe - standard_error
                upper = mean_probe + standard_error
                reference = max(self.context_probe_reward_floor, self.context_probe_failed_reference + self.context_reactivation_margin)
                if lower > reference or upper < self.context_probe_reward_floor or self.context_probe_steps_seen >= self.context_probe_max_steps:
                    self.context_probe_remaining = 0
            elif self.context_probe_steps_seen >= self.context_probe_max_steps:
                self.context_probe_remaining = 0
            if self.context_probe_origin == "evidence_probe":
                from .routing.evidence import CircuitEvidence, probe_decision
                candidate_payload = self.circuit_evidence.get(module.id)
                candidate_item = CircuitEvidence.from_dict(candidate_payload) if candidate_payload is not None else CircuitEvidence(module.id)
                incumbent_payload = self.circuit_evidence.get(self.context_probe_incumbent_module)
                incumbent_item = CircuitEvidence.from_dict(incumbent_payload) if incumbent_payload is not None else CircuitEvidence(self.context_probe_incumbent_module or "?")
                if probe_decision(
                    candidate_item.probe_log_evidence, candidate_item.probe_scored,
                    incumbent_item.probe_log_evidence, incumbent_item.probe_scored,
                    min_probe=self.context_probe_min_steps,
                    enter_margin=self.evidence_enter_margin,
                    exit_margin=self.evidence_exit_margin,
                    novelty_mass=self.evidence_novelty_mass,
                ) != "continue":
                    self.context_probe_remaining = 0
        self.pending_motor_module = None
        self.context_pending_features = ()
        self.context_pending_action = 0.0
        self.context_pending_mode = "normal"
        self.context_pending_prediction = 0.0
        self.context_pending_scale = 0.1
        self.context_pending_exploration_value = 0.0
        self.context_pending_exploration_sigma = 0.0

    def _context_maybe_switch(self) -> None:
        if self.variable_order_learning_enabled and self.variable_order_fingerprint_active:
            return
        # R1 evidence recall is first resort: a dormant owner justified by
        # calibrated comparative evidence reactivates directly without a new
        # fingerprint or probe. Legacy path handles all other cases.
        if (
            self.evidence_decisions_enabled
            and self.context_state in ("warning", "search")
            and self.context_probe_module is None
        ):
            recall_id = self._evidence_recall_candidate()
            if recall_id is not None:
                incumbent = self.motor_modules.get(self.active_motor_module)
                candidate = self.motor_modules[recall_id]
                if incumbent is not None:
                    incumbent.dormant = True
                    incumbent.protected = True
                candidate.dormant = False
                candidate.protected = False
                candidate.detector_cusum = 0.0
                candidate.detector_frozen = False
                candidate.surprise_evidence = 0.0
                candidate.negative_surprise = 0.0
                candidate.negative_streak = 0
                candidate.confidence = 0.0
                self.active_motor_module = recall_id
                self.context_state = "normal"
                self.context_probe_queue = []
                self._clear_probe_state()
                self.events.append({
                    "step": self.step_count,
                    "kind": "evidence_recall",
                    "module_id": recall_id,
                    "incumbent_module": incumbent.id if incumbent is not None else None,
                    "posterior": dict(self.evidence_posterior),
                })
                return
        # Ambiguous evidence with a plausible dormant challenger earns one
        # bounded probe before any structural response. Legacy paths below
        # handle novelty-leaning and no-challenger cases.
        if (
            self.evidence_decisions_enabled
            and self.context_state in ("warning", "search")
            and self.context_probe_module is None
            and not self.variable_order_fingerprint_active
            and not self.adaptive_dendritic_fingerprint_active
        ):
            if self._evidence_maybe_begin_probe():
                return
        active_for_retry = self.motor_modules.get(self.active_motor_module) if self.active_motor_module is not None else None
        if (
            self.variable_order_learning_enabled
            and self.composition_learning_enabled
            and self.composition_feature_owners
            and self.context_state == "normal"
            and self.context_probe_module is None
            and active_for_retry is not None
            and active_for_retry.negative_streak >= self.context_negative_streak
            and self.step_count <= self.variable_order_detector_hold_until
        ):
            self._variable_order_begin_fingerprint()
            self.events.append({
                "step": self.step_count,
                "kind": "composition_fast_recall_retry_started",
                "module_id": active_for_retry.id,
                "negative_streak": active_for_retry.negative_streak,
                "detector_hold_until": self.variable_order_detector_hold_until,
            })
            return
        if self.variable_order_learning_enabled and self.step_count <= self.variable_order_detector_hold_until:
            return
        if self.variable_order_learning_enabled and self.context_state in ("warning", "search") and self.context_probe_module is None:
            self._variable_order_begin_fingerprint()
            return
        if self.adaptive_dendritic_fingerprint_enabled and self.adaptive_dendritic_fingerprint_active:
            return
        if self.adaptive_dendritic_fingerprint_enabled and self.step_count <= self.adaptive_dendritic_detector_hold_until:
            return
        if self.adaptive_dendritic_fingerprint_enabled and self.context_state in ("warning", "search") and self.context_probe_module is None:
            self._adaptive_dendritic_begin_fingerprint()
            return
        if self.context_probe_module is not None:
            if self.context_probe_remaining > 0:
                return
            candidate = self.motor_modules[self.context_probe_module]
            if self.context_probe_origin == "evidence_probe":
                self._evidence_resolve_probe(candidate)
                return
            mean_probe = self.context_probe_sum / float(self.context_probe_count) if self.context_probe_count else -1.0
            variance = 0.0
            if self.context_probe_count:
                variance = max(0.0, self.context_probe_sumsq / float(self.context_probe_count) - mean_probe * mean_probe)
            standard_error = math.sqrt(variance / float(max(1, self.context_probe_count)))
            reference = max(self.context_probe_reward_floor, self.context_probe_failed_reference + self.context_reactivation_margin)
            # A candidate is accepted only when the observed evidence is
            # positive with a modest uncertainty margin and beats the best
            # failed incumbent.  The probe may finish early, but its final
            # decision remains target/phase blind and reward-only.
            if mean_probe - standard_error > reference:
                candidate.dormant = False
                candidate.protected = False
                # A dormant engram carries the alarm episode that led to its
                # suspension.  Clear only that transient episode on accepted
                # reactivation; preserve detector predictor/variance/count,
                # baseline, confidence behavior, and all policy weights.
                candidate.detector_cusum = 0.0
                candidate.detector_frozen = False
                candidate.surprise_evidence = 0.0
                candidate.negative_surprise = 0.0
                candidate.negative_streak = 0
                # Re-establish confidence in the reactivated engram from its
                # new-context outcomes instead of immediately switching again
                # on one noisy surprise.
                candidate.confidence = 0.0
                self.active_motor_module = candidate.id
                self.context_state = "normal"
                self.events.append({"step": self.step_count, "kind": "motor_module_reactivated", "module_id": candidate.id, "probe_mean": mean_probe, "probe_origin": self.context_probe_origin})
                self.context_probe_module = None
                self.context_probe_queue = []
                self._clear_probe_state()
            else:
                candidate.dormant = True
                self.events.append({"step": self.step_count, "kind": "motor_probe_rejected", "module_id": candidate.id, "probe_mean": mean_probe, "probe_origin": self.context_probe_origin})
                self.context_probe_failed_reference = max(self.context_probe_failed_reference, mean_probe)
                self.context_probe_module = None
                if self.context_probe_queue:
                    self._start_next_probe()
                else:
                    # The candidate's detector state is intentionally not
                    # updated by a protected probe.  Use the incumbent's
                    # persisted evidence to decide whether this warning-only
                    # search may return to the fallback or has escalated to
                    # the normal recruitment path.
                    incumbent = self.motor_modules.get(self.context_probe_incumbent_module)
                    incumbent_cusum = incumbent.detector_cusum if incumbent is not None else self.context_switch_threshold
                    if self.context_probe_origin in ("owner_evidence", "owner_evidence_midpoint"):
                        restart_fingerprint = self.context_probe_origin == "owner_evidence_midpoint"
                        if incumbent is not None:
                            incumbent.dormant = False
                            incumbent.protected = False
                            self.active_motor_module = incumbent.id
                        self.context_state = "normal"
                        self._clear_probe_state()
                        if restart_fingerprint:
                            self._variable_order_begin_fingerprint()
                    elif self.context_probe_origin == "warning" and incumbent_cusum < self.context_switch_threshold:
                        if incumbent is not None:
                            incumbent.dormant = False
                            incumbent.protected = True
                            self.active_motor_module = incumbent.id
                        self.context_state = "warning"
                        self._clear_probe_state()
                    else:
                        if len(self.motor_modules) < self.context_max_modules:
                            try:
                                recruited = self._recruit_motor_module()
                            except RuntimeError:
                                # Recruitment is atomic, but the incumbent
                                # must remain a valid active fallback when a
                                # resource budget rejects the new module.
                                candidate.dormant = False
                                candidate.protected = False
                                self.active_motor_module = candidate.id
                                self.context_probe_queue = []
                                self._clear_probe_state()
                                self.context_state = "normal"
                            else:
                                self.active_motor_module = recruited.id
                                self.context_probe_queue = []
                                fallback_incumbent = self.context_probe_incumbent_module
                                self._clear_probe_state()
                                self.context_state = "normal"
                                self._evidence_begin_growth_watch(recruited.id, incumbent_id=fallback_incumbent)
                        else:
                            candidate.dormant = False
                            candidate.protected = False
                            self.active_motor_module = candidate.id
                            self.context_probe_queue = []
                            self._clear_probe_state()
                            self.context_state = "normal"
            return
        active = self.motor_modules[self.active_motor_module]
        # Once a graph-derived composed owner exists, a local surprise is
        # enough to begin the same neutral 16-outcome fingerprint immediately.
        # This is a learner-side recall acceleration: it uses only the motor
        # detector's target-blind surprise and never shortens the frozen
        # fingerprint window or consults task metadata.  It avoids paying the
        # full detector warm-up again when an already learned composition is
        # encountered after a context change.
        if (
            self.variable_order_learning_enabled
            and
            self.composition_learning_enabled
            and (self.composition_feature_owners or self._composition_feature_bank())
            and self.context_state == "normal"
            and active.negative_streak >= self.context_negative_streak
        ):
            self._variable_order_begin_fingerprint()
            self.events.append({
                "step": self.step_count,
                "kind": "composition_fast_recall_started",
                "module_id": active.id,
                "negative_streak": active.negative_streak,
                "surprise_evidence": active.surprise_evidence,
            })
            return
        if self.context_state == "normal":
            if active.detector_count < self.context_detector_min_evidence or active.confidence < self.context_confidence_threshold:
                return
            if active.detector_cusum < self.context_warning_threshold:
                return
            self.context_state = "warning"
            active.detector_frozen = True
            self.events.append({
                "step": self.step_count,
                "kind": "motor_warning_started",
                "module_id": active.id,
                "detector_count": active.detector_count,
                "detector_mean": active.detector_mean,
                "detector_variance": active.detector_variance,
                "detector_cusum": active.detector_cusum,
                "confidence": active.confidence,
            })
            if self.variable_order_learning_enabled:
                self._variable_order_begin_fingerprint()
                return
            dormant = [module for module in self.motor_modules.values() if module.id != active.id and module.dormant]
            dormant.sort(key=lambda module: (-module.baseline, module.id))
            if dormant:
                active.dormant = True
                active.protected = True
                self.context_probe_queue = [module.id for module in dormant]
                self.context_probe_reference = 0.0
                self.context_probe_incumbent_module = active.id
                self.context_probe_origin = "warning"
                self._start_next_probe()
                return
        if self.context_state == "warning":
            if active.detector_cusum < self.context_warning_threshold:
                self.context_state = "normal"
                active.detector_frozen = False
                self.events.append({"step": self.step_count, "kind": "motor_warning_resolved", "module_id": active.id, "detector_cusum": active.detector_cusum})
                return
            if active.detector_cusum < self.context_switch_threshold:
                return
            self.context_state = "search"
            self.events.append({
                "step": self.step_count,
                "kind": "motor_warning_escalated",
                "module_id": active.id,
                "detector_count": active.detector_count,
                "detector_cusum": active.detector_cusum,
            })
        if self.context_state != "search":
            return
        active.dormant = True
        active.protected = True
        dormant = [module for module in self.motor_modules.values() if module.id != active.id and module.dormant]
        dormant.sort(key=lambda module: (-module.baseline, module.id))
        if dormant:
            self.context_probe_queue = [module.id for module in dormant]
            self.context_probe_reference = 0.0
            self.context_probe_incumbent_module = active.id
            self.context_probe_origin = "alarm"
            self._start_next_probe()
        elif len(self.motor_modules) < self.context_max_modules:
            try:
                recruited = self._recruit_motor_module()
            except RuntimeError:
                # Do not leave the protected incumbent as the active module
                # if capacity/energy prevents recruitment.
                active.dormant = False
                active.protected = False
                self.active_motor_module = active.id
                self.context_state = "normal"
            else:
                self.active_motor_module = recruited.id
                self.context_state = "normal"
                self._evidence_begin_growth_watch(recruited.id, incumbent_id=active.id)
        else:
            active.dormant = False
            active.protected = False
            self.context_state = "normal"

    def _start_next_probe(self) -> None:
        if not self.context_probe_queue:
            return
        candidate_id = self.context_probe_queue.pop(0)
        candidate = self.motor_modules[candidate_id]
        # Keep candidate cell dynamics dormant; _context_output evaluates its
        # proposed mean directly from the current sensor activations.
        candidate.dormant = True
        self.context_probe_module = candidate.id
        self.context_probe_remaining = self.context_probe_max_steps
        self.context_probe_steps_seen = 0
        self.context_probe_informative_count = 0
        self.context_probe_rewards = []
        self.context_probe_sum = 0.0
        self.context_probe_sumsq = 0.0
        self.context_probe_count = 0
        self.context_probe_exploration = 0.0
        self.context_probe_candidate_mean = 0.0
        self.context_probe_incumbent_mean = 0.0
        self.active_motor_module = candidate.id
        self.events.append({"step": self.step_count, "kind": "motor_module_probe_started", "module_id": candidate.id, "reference": self.context_probe_reward_floor, "incumbent_module_id": self.context_probe_incumbent_module, "probe_origin": self.context_probe_origin})

    def _clear_probe_state(self) -> None:
        self.context_probe_module = None
        self.context_probe_remaining = 0
        self.context_probe_steps_seen = 0
        self.context_probe_informative_count = 0
        self.context_probe_rewards = []
        self.context_probe_sum = 0.0
        self.context_probe_sumsq = 0.0
        self.context_probe_count = 0
        self.context_probe_candidate_mean = 0.0
        self.context_probe_incumbent_mean = 0.0
        self.context_probe_exploration = 0.0
        self.context_probe_incumbent_module = None
        self.context_probe_origin = None

    def _activation_drive(self, destination: str, previous: Mapping[str, float]) -> float:
        drive = 0.0
        for source in self.graph.incoming.get(destination, []):
            synapse = self.graph.get(source, destination)
            if synapse is not None and synapse.transmission_delay == 0:
                drive += previous.get(source, 0.0) * synapse.strength
        return drive

    def enable_representation_learning(self, rate: float = 0.02, noise: float = 0.20) -> None:
        """Enable local reward-modulated hidden node perturbations."""
        if not math.isfinite(float(rate)) or rate < 0.0:
            raise ValueError("representation learning rate must be finite and nonnegative")
        if not math.isfinite(float(noise)) or not 0.0 <= noise <= 1.0:
            raise ValueError("representation noise must be finite and in [0, 1]")
        self.representation_learning_enabled = True
        self.representation_learning_rate = float(rate)
        self.representation_noise = float(noise)
        # Keep task-learned representation bias from being canceled by signed
        # threshold centering.  Magnitude adaptation still runs in propagate.
        for cell in self.cells.values():
            if cell.kind == "hidden":
                cell.threshold = 0.0

    def set_representation_tape(self, tape: Sequence[float]) -> None:
        """Use evaluator-supplied hidden perturbations for matched runs."""
        values = tuple(float(value) for value in tape)
        if not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in values):
            raise ValueError("representation tape values must be finite and in [-1, 1]")
        self.representation_tape = values
        self.representation_cursor = 0

    def _representation_value(self) -> float:
        if self.representation_tape is not None:
            if self.representation_cursor >= len(self.representation_tape):
                raise ValueError("representation tape exhausted")
            value = self.representation_tape[self.representation_cursor]
            self.representation_cursor += 1
            return value
        return self.representation_rng.uniform(-1.0, 1.0)

    def _representation_hidden_ids(self) -> List[str]:
        return sorted(identifier for identifier, cell in self.cells.items() if cell.kind == "hidden")

    def _record_node_eligibility(self, hidden_id: str, previous: Mapping[str, float], base: float, xi: float, sigma: float) -> None:
        if sigma <= 1e-12:
            return
        score = (1.0 - base * base) * xi / sigma
        cell = self.cells[hidden_id]
        cell.representation_bias_eligibility = self.trace_decay * cell.representation_bias_eligibility + score
        for source in self.graph.incoming.get(hidden_id, []):
            synapse = self.graph.get(source, hidden_id)
            if synapse is None or synapse.transmission_delay != 0:
                continue
            pre = previous.get(source, 0.0)
            synapse.node_eligibility_trace = self.trace_decay * synapse.node_eligibility_trace + pre * score

    def _apply_representation_reward(self, reward_prediction_error: float) -> None:
        """Apply delayed reward to local hidden node-perturbation traces once."""
        for cell in self.cells.values():
            if cell.kind != "hidden":
                continue
            if self.representation_learning_enabled:
                cell.representation_bias += self.representation_learning_rate * cell.representation_bias_eligibility * reward_prediction_error
                cell.representation_bias = max(-1.0, min(1.0, cell.representation_bias))
            cell.representation_bias_eligibility = 0.0
        for synapse in self.graph.iter_synapses():
            if self.cells[synapse.destination].kind == "hidden":
                if self.representation_learning_enabled:
                    synapse.strength += self.representation_learning_rate * synapse.node_eligibility_trace * reward_prediction_error
                    synapse.strength = max(-1.0, min(1.0, synapse.strength))
                synapse.node_eligibility_trace = 0.0

    def _propagation_steps(self) -> int:
        """Return the synchronous depth required by the current graph."""
        if self.general_feature_lineage:
            return max(2, 1 + max(int(value["depth"]) for value in self.general_feature_lineage.values()))
        if self.compositional_substrate_enabled and any(
            cell.activation_type == "dendritic_product_composed" for cell in self.cells.values()
        ):
            return 3
        return 2

    def _propagate(self, values: Sequence[float], propagation_steps: Optional[int] = None) -> None:
        for identifier, value in zip(self.input_ids, values):
            self.cells[identifier].observe(value)
        previous = {identifier: cell.activation for identifier, cell in self.cells.items()}
        total_steps = max(1, self._propagation_steps() if propagation_steps is None else int(propagation_steps))
        self._motor_causal_presynaptic = {}
        hidden_ids = self._representation_hidden_ids()
        suppress_representation_noise = self.context_state in ("warning", "search") or self.context_probe_module is not None
        for pass_index in range(total_steps):
            penultimate = pass_index == total_steps - 2
            current = dict(previous)
            for identifier in sorted(self.cells):
                cell = self.cells[identifier]
                if cell.kind == "input":
                    continue
                if self.context_enabled and cell.kind == "motor_module" and self.motor_modules.get(identifier) is not None and self.motor_modules[identifier].dormant:
                    # Dormant engrams are frozen memories.  Preserve all cell
                    # state needed for exact expression after reactivation;
                    # protected probes evaluate weights directly below.
                    current[identifier] = cell.activation
                    continue
                if cell.activation_type == "dendritic_product":
                    source_a, source_b = cell.dendritic_sources
                    synapse_a = self.graph.get(source_a, identifier)
                    synapse_b = self.graph.get(source_b, identifier)
                    if synapse_a is None or synapse_b is None:
                        raise AssertionError("product cell is missing a declared afferent")
                    drive = (
                        previous.get(source_a, 0.0) * synapse_a.strength
                        * previous.get(source_b, 0.0) * synapse_b.strength
                    )
                elif cell.activation_type == "dendritic_product_n":
                    drive = 1.0
                    for source in cell.dendritic_sources:
                        synapse = self.graph.get(source, identifier)
                        if synapse is None:
                            raise AssertionError("n-ary product cell is missing a declared afferent")
                        drive *= previous.get(source, 0.0) * synapse.strength
                elif cell.activation_type == "dendritic_product_composed":
                    drive = cell.dendritic_normalizer
                    for source in cell.dendritic_sources:
                        synapse = self.graph.get(source, identifier)
                        if synapse is None:
                            raise AssertionError("composed product cell is missing a declared afferent")
                        drive *= previous.get(source, 0.0) * synapse.strength
                else:
                    drive = self._activation_drive(identifier, previous)
                proposal = self.adaptive_dendritic_proposal
                if proposal is not None and proposal.get("cell_id") == identifier and pass_index == total_steps - 1:
                    old_cell = proposal["old_cell"]
                    old_drive = sum(
                        previous.get(str(payload["source"]), 0.0) * float(payload["strength"])
                        for payload in proposal.get("old_synapses", [])
                        if int(payload.get("transmission_delay", 0)) == 0
                    )
                    old_adaptation = float(old_cell.get("adaptation", 0.0))
                    old_threshold = float(old_cell.get("threshold", 0.0))
                    old_bias = float(old_cell.get("representation_bias", 0.0))
                    proposal["counterfactual_drive"] = old_drive
                    proposal["counterfactual_sources"] = {
                        str(payload["source"]): previous.get(str(payload["source"]), 0.0)
                        for payload in proposal.get("old_synapses", [])
                    }
                    proposal["counterfactual_activation"] = math.tanh(
                        (old_drive + old_bias - old_threshold) / (1.0 + old_adaptation)
                    )
                if pass_index == total_steps - 1 and cell.kind == "motor_module":
                    module = next((candidate for candidate in self.motor_modules.values() if candidate.cell_id == identifier), None)
                    if module is not None:
                        self._motor_causal_presynaptic[module.id] = {
                            synapse.source: previous.get(synapse.source, 0.0)
                            for synapse in self._module_synapses(module)
                        }
                # Adaptation is a symmetric excitability/gain change.  It is
                # magnitude-only state, never a signed subtractive bias.
                base = math.tanh((drive + cell.representation_bias - cell.threshold) / (1.0 + cell.adaptation))
                value = base
                if identifier in hidden_ids and penultimate and self.representation_learning_enabled:
                    xi = self._representation_value()
                    sigma = self.representation_noise
                    if sigma > 1e-12:
                        if not suppress_representation_noise:
                            self._record_node_eligibility(identifier, previous, base, xi, sigma)
                            value = max(-1.0, min(1.0, base + sigma * xi))
                current[identifier] = value
                cell.observe(value)
                cell.adaptation = 0.96 * cell.adaptation + 0.04 * abs(value)
            previous = current

    def _apply_reward(self, reward_prediction_error: float) -> None:
        """Apply delayed reward to traces created by the previous action."""
        if not self.legacy_learning_enabled:
            return
        for synapse in self.graph.iter_synapses():
            synapse.strength += self.learning_rate * synapse.plasticity * synapse.eligibility_trace * reward_prediction_error
            synapse.strength = max(-1.0, min(1.0, synapse.strength))

    def _apply_actor_reward(self, reward_prediction_error: float) -> None:
        """Credit the exact preceding motor perturbation once."""
        for synapse in self.graph.iter_synapses():
            if synapse.destination not in self.output_ids:
                continue
            synapse.strength += self.actor_learning_rate * synapse.actor_eligibility_trace * reward_prediction_error
            synapse.strength = max(-1.0, min(1.0, synapse.strength))
            synapse.actor_eligibility_trace = 0.0

    def _set_actor_eligibility(self, output_means: Mapping[str, float], perturbations: Mapping[str, float], sigma: float) -> None:
        """Snapshot a policy-gradient eligibility for this action.

        For a tanh motor mean ``mu`` and additive Gaussian-like perturbation
        ``sigma * xi``, the local score estimator is
        ``pre * (1 - mu**2) * xi / sigma``.  The benchmark supplies bounded
        deterministic xi values, but the same estimator remains valid.
        """
        for synapse in self.graph.iter_synapses():
            if synapse.destination not in self.output_ids or sigma <= 1e-12:
                synapse.actor_eligibility_trace = 0.0
                continue
            pre = self.cells[synapse.source].activation
            mu = max(-1.0, min(1.0, float(output_means[synapse.destination])))
            xi = float(perturbations[synapse.destination])
            synapse.actor_eligibility_trace = pre * (1.0 - mu * mu) * xi / sigma

    def _local_learning(self, modulators: Modulators) -> None:
        for synapse in self.graph.iter_synapses():
            if self.context_enabled and (synapse.source in self.motor_modules or synapse.destination in self.motor_modules):
                continue
            pre = self.cells[synapse.source].activation
            post = self.cells[synapse.destination].activation
            coactivity = pre * post
            self._coactivity[(synapse.source, synapse.destination)] = (
                0.9 * self._coactivity.get((synapse.source, synapse.destination), 0.0) + 0.1 * abs(coactivity)
            )
            synapse.record(coactivity, self.trace_decay)
            synapse.plasticity = max(0.02, min(1.0, synapse.plasticity * (0.999 if abs(coactivity) > 0.2 and modulators.reward > 0 else 1.0) + 0.01 * modulators.novelty))
            synapse.stability = min(1.0, synapse.stability + 0.005 * abs(coactivity))

    def _homeostasis(self) -> None:
        for cell in self.cells.values():
            if self.representation_learning_enabled and cell.kind == "hidden":
                # Hidden threshold remains neutral while representation_bias
                # carries task-specific offset; magnitude adaptation remains
                # updated by _propagate.
                continue
            if self.context_enabled and cell.kind == "motor_module" and cell.id != self.motor_modules[self.active_motor_module].cell_id:
                continue
            cell.update_homeostasis(self.homeostasis_rate)

    def _structural_adaptation(self, modulators: Modulators) -> List[Dict[str, object]]:
        if not self.structural_plasticity_enabled:
            return []
        if self.step_count % self.structural_interval:
            return []
        changes: List[Dict[str, object]] = []
        candidates: List[Tuple[float, str, str]] = []
        protected_features = self._protected_feature_cell_ids()
        sources = sorted(identifier for identifier, cell in self.cells.items() if cell.kind != "motor_module" and identifier not in protected_features and abs(cell.activation) > 0.2)
        destinations = sorted(identifier for identifier, cell in self.cells.items() if cell.kind not in ("input", "motor_module") and identifier not in protected_features and abs(cell.activation) > 0.05)
        for source in sources:
            for destination in destinations:
                if source == destination or self.graph.has(source, destination):
                    continue
                score = abs(self.cells[source].activation * self.cells[destination].activation) + 0.25 * modulators.novelty
                if score > 0.35:
                    candidates.append((score, source, destination))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        if candidates and self.resources.can_add_synapse():
            _, source, destination = candidates[0]
            strength = self.rng.uniform(-0.12, 0.12)
            self.graph.add(Synapse(source, destination, strength, plasticity=1.0))
            self.resources.added_synapse()
            changes.append({"step": self.step_count, "kind": "synapse_created", "synapse_id": self.graph.get(source, destination).id, "source": source, "destination": destination, "strength": strength})
        # Remove only aged, genuinely low-utility synapses. Keep a connected scaffold alive.
        removable = [s for s in self.graph.iter_synapses() if s.source not in protected_features and s.destination not in protected_features and self.cells[s.source].kind != "motor_module" and self.cells[s.destination].kind != "motor_module" and s.age >= self.structural_interval * 2 and s.utility < 0.002 and abs(s.strength) < 0.08]
        removable.sort(key=lambda s: (s.utility, s.age, s.id))
        for synapse in removable[:1]:
            if len(self.graph.synapses) <= len(self.output_ids):
                break
            if not self._preserves_sensor_output_reachability(synapse.source, synapse.destination):
                continue
            self.graph.remove(synapse.source, synapse.destination)
            self.resources.removed_synapse()
            changes.append({"step": self.step_count, "kind": "synapse_pruned", "synapse_id": synapse.id, "source": synapse.source, "destination": synapse.destination})
        self.events.extend(changes)
        return changes

    def _preserves_sensor_output_reachability(self, skip_source: Optional[str] = None, skip_destination: Optional[str] = None) -> bool:
        for source in self.input_ids:
            reachable = {source}
            pending = [source]
            while pending:
                current = pending.pop(0)
                for destination in self.graph.outgoing.get(current, []):
                    if current == skip_source and destination == skip_destination:
                        continue
                    if destination not in reachable:
                        reachable.add(destination)
                        pending.append(destination)
            if any(output not in reachable for output in self.output_ids):
                return False
        return True

    def _consume_outcome_immediate(self, reward: float) -> float:
        if not self._pending_outcome:
            if abs(reward) > 1e-12:
                raise ValueError("reward supplied without a pending organism action")
            return 0.0
        prediction_error = reward - self.reward_baseline
        fingerprint_owned = False
        if self.context_enabled:
            fingerprint_owned = (
                (self.adaptive_dendritic_fingerprint_active or self.variable_order_fingerprint_active)
                and self.context_state == "fingerprint"
                and self.pending_motor_module == (
                    self.adaptive_dendritic_fingerprint_owner
                    if self.adaptive_dendritic_fingerprint_active
                    else self.variable_order_fingerprint_owner
                )
                and self.context_pending_mode == "fingerprint"
            )
            # A delayed reward can arrive for one of the final diagnostic
            # actions after the fingerprint has already resolved.  That
            # action was intentionally learning-frozen; discard its stale
            # pending evidence instead of misclassifying it as normal control
            # evidence or leaving an unloadable checkpoint behind.
            if self.context_pending_mode == "fingerprint" and not fingerprint_owned:
                self.variable_order_fingerprint_pending = None
                self.composition_fingerprint_pending = None
                self.adaptive_dendritic_fingerprint_pending = None
                self._pending_outcome = False
                return prediction_error
            trial = self.adaptive_dendritic_proposal
            trial_owner = str(trial.get("owner_module", "")) if trial is not None else None
            trial_owned = (
                trial is not None
                and self.context_state == "normal"
                and self.context_probe_module is None
                and self.active_motor_module == trial_owner
                and self.pending_motor_module == trial_owner
                and self.context_pending_mode == "normal"
            )
            if trial is not None and not trial_owned:
                self._adaptive_dendritic_cancel("ownership_or_context_changed")
            shadow = self.adaptive_dendritic_shadow_pending
            shadow_owner = str(shadow.get("owner_module", "")) if shadow is not None else None
            shadow_owned = (
                shadow is not None
                and self.context_state == "normal"
                and self.context_probe_module is None
                and self.pending_motor_module == shadow_owner
                and self.context_pending_mode == "normal"
            )
            shadow_module = self.motor_modules.get(shadow_owner) if shadow_owner is not None else None
            shadow_local_rpe = (
                prediction_error + self.reward_baseline - shadow_module.baseline
                if shadow_module is not None else 0.0
            )
            self._apply_context_reward(prediction_error, structural_trial=trial_owned or fingerprint_owned)
            if fingerprint_owned:
                if self.variable_order_fingerprint_active:
                    self._composition_accumulate_fingerprint(reward)
                    self._variable_order_accumulate_fingerprint(reward)
                else:
                    self._adaptive_dendritic_fingerprint_accumulate(reward)
            elif self.variable_order_learning_enabled:
                self._variable_order_accumulate_normal(reward)
            if shadow is not None:
                if shadow_owned:
                    self._adaptive_dendritic_shadow_accumulate(shadow_local_rpe)
                else:
                    self.adaptive_dendritic_shadow_pending = None
            if not trial_owned and not fingerprint_owned:
                self._context_maybe_switch()
        else:
            self._apply_actor_reward(prediction_error)
        self._apply_adaptive_dendritic_outcome(prediction_error)
        self._apply_representation_reward(prediction_error)
        self._apply_reward(prediction_error)
        if not fingerprint_owned:
            self.reward_baseline += self.reward_baseline_rate * prediction_error
        self._pending_outcome = False
        return prediction_error

    def _consume_outcome(self, reward: float) -> float:
        if not self.delayed_credit_enabled:
            return self._consume_outcome_immediate(reward)
        if not self._pending_outcome:
            if abs(reward) > 1e-12:
                raise ValueError("reward supplied without a pending organism action")
            return 0.0
        self.delayed_credit_queue.append(self._delayed_credit_snapshot())
        self._delayed_credit_clear_live_pending()
        if len(self.delayed_credit_queue) <= self.delayed_credit_delay:
            return 0.0
        snapshot = self.delayed_credit_queue.pop(0)
        self._delayed_credit_restore(snapshot)
        result = self._consume_outcome_immediate(reward)
        self.events.append({"step": self.step_count, "kind": "delayed_credit_applied", "delay": self.delayed_credit_delay, "action_module": snapshot.get("pending_motor_module")})
        return result

    def apply_outcome(self, reward: float) -> None:
        """Apply a terminal outcome without generating another decision."""
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError("outcome reward must be finite")
        if not self._pending_outcome:
            raise ValueError("no pending organism action for outcome")
        prediction_error = self._consume_outcome(reward)
        if self.metrics_history:
            self.metrics_history[-1]["outcome_reward"] = reward
            self.metrics_history[-1]["outcome_prediction_error"] = prediction_error

    def set_exploration_tape(self, tape: Sequence[float]) -> None:
        """Use evaluator-supplied action noise for matched comparisons."""
        values = tuple(float(value) for value in tape)
        if not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in values):
            raise ValueError("exploration tape values must be finite and in [-1, 1]")
        self.exploration_tape = values
        self.exploration_cursor = 0

    def _exploration_value(self) -> float:
        if self.exploration_tape is not None:
            if self.exploration_cursor >= len(self.exploration_tape):
                raise ValueError("exploration tape exhausted")
            value = self.exploration_tape[self.exploration_cursor]
            self.exploration_cursor += 1
            return value
        return self.exploration_rng.uniform(-1.0, 1.0)

    def step(self, inputs: Sequence[float], modulators: Optional[Modulators] = None) -> StepResult:
        """Take one decision step; reward belongs to the preceding decision.

        The environment returns a reward after this method's output is applied.
        Passing that reward into the next call updates the eligibility trace made
        by this action before the next action is generated.
        """
        if len(inputs) != len(self.input_ids):
            raise ValueError("expected %d inputs, got %d" % (len(self.input_ids), len(inputs)))
        modulators = modulators or Modulators()
        values = [float(value) for value in inputs]
        if not all(math.isfinite(value) for value in values):
            raise ValueError("inputs must contain only finite numbers")
        for name in ("reward", "novelty", "uncertainty", "salience", "exploration"):
            if not math.isfinite(float(getattr(modulators, name))):
                raise ValueError("modulator %s must be finite" % name)
        self.step_count += 1
        self.resources.begin_step(len(self.cells), len(self.graph.synapses))
        event_start = len(self.events)
        prediction_error = self._consume_outcome(float(modulators.reward))
        self._adaptive_dendritic_maybe_propose()
        self._propagate(values)
        self._local_learning(modulators)
        self._homeostasis()
        changes = self._structural_adaptation(modulators)
        changes.extend(
            event for event in self.events[event_start:]
            if str(event.get("kind", "")).startswith("adaptive_dendritic_")
        )
        self._pending_outcome = True
        if self.context_enabled:
            outputs = self._context_output(modulators)
        else:
            output_means = {identifier: self.cells[identifier].activation for identifier in self.output_ids}
            perturbations = {identifier: self._exploration_value() for identifier in self.output_ids}
            sigma = float(modulators.exploration)
            self._set_actor_eligibility(output_means, perturbations, sigma)
            outputs = tuple(max(-1.0, min(1.0, output_means[identifier] + (perturbations[identifier] * sigma))) for identifier in self.output_ids)
        self._adaptive_dendritic_record_action_credit()
        if self.variable_order_fingerprint_active:
            self._variable_order_record_fingerprint_pending()
            self._composition_record_fingerprint_pending()
        elif self.adaptive_dendritic_fingerprint_active:
            self._adaptive_dendritic_fingerprint_record_pending()
        elif self.variable_order_learning_enabled:
            self._variable_order_record_normal_pending()
        else:
            self._adaptive_dendritic_shadow_record_pending()
        self.metrics_history.append({
            "step": self.step_count,
            "reward": float(modulators.reward),
            "prediction_error": prediction_error,
            "mean_abs_activity": sum(abs(self.cells[identifier].activation) for identifier in sorted(self.cells)) / float(len(self.cells)),
            "cells": len(self.cells),
            "synapses": len(self.graph.synapses),
            "energy_used": self.resources.energy_used,
            "structural_events": len(changes),
            "outcome_reward": None,
        })
        return StepResult(outputs, modulators.reward, self.resources.energy_used, changes)

    def validate(self) -> None:
        self.graph.validate(self.cells)
        if self.context_afferent_kind not in ("input", "hidden"):
            raise AssertionError("invalid context afferent kind")
        if not self.input_ids or not self.output_ids:
            raise AssertionError("organism must have nonzero sensor and output sizes")
        if not self._preserves_sensor_output_reachability():
            raise AssertionError("sensor-to-output reachability invariant violated")
        if len(self.cells) > self.resources.max_cells or len(self.graph.synapses) > self.resources.max_synapses:
            raise AssertionError("resource bound exceeded")
        if self.resources.baseline_cost(len(self.cells), len(self.graph.synapses)) > self.resources.energy_per_step + 1e-12:
            raise AssertionError("current baseline cost exceeds per-step energy budget")
        if self.resources.counters.get("cells") != len(self.cells) or self.resources.counters.get("synapses") != len(self.graph.synapses):
            raise AssertionError("resource counters do not match organism structure")
        if not isinstance(self.variable_order_enabled, bool):
            raise AssertionError("variable-order enabled flag is invalid")
        if isinstance(self.max_dendritic_order, bool) or not isinstance(self.max_dendritic_order, int) or self.max_dendritic_order < 2 or self.max_dendritic_order > 3:
            raise AssertionError("maximum dendritic order is invalid")
        if not self.variable_order_enabled and self.max_dendritic_order != 2:
            raise AssertionError("disabled variable-order substrate must retain binary maximum")
        if not isinstance(self.variable_order_feature_owners, dict):
            raise AssertionError("variable-order feature owners must be a mapping")
        if not isinstance(self.variable_order_learning_enabled, bool):
            raise AssertionError("variable-order learner flag is invalid")
        if isinstance(self.variable_order_fingerprint_window, bool) or self.variable_order_fingerprint_window < 1:
            raise AssertionError("variable-order fingerprint window is invalid")
        if isinstance(self.variable_order_normal_window, bool) or not isinstance(self.variable_order_normal_window, int) or self.variable_order_normal_window < 1:
            raise AssertionError("variable-order normal window is invalid")
        if self.variable_order_fingerprint_count < 0 or self.variable_order_fingerprint_count > self.variable_order_fingerprint_window:
            raise AssertionError("variable-order fingerprint count is invalid")
        if not isinstance(self.variable_order_midpoint_existing_owner_enabled, bool):
            raise AssertionError("variable-order midpoint resolution flag is invalid")
        if not isinstance(self.variable_order_owner_evidence_enabled, bool):
            raise AssertionError("variable-order owner evidence flag is invalid")
        if not isinstance(self.variable_order_owner_probe_enabled, bool):
            raise AssertionError("variable-order owner probe flag is invalid")
        if not isinstance(self.variable_order_owner_midpoint_probe_enabled, bool):
            raise AssertionError("variable-order owner midpoint probe flag is invalid")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in (self.variable_order_owner_midpoint_probe_min_probability, self.variable_order_owner_midpoint_probe_min_margin)):
            raise AssertionError("variable-order owner midpoint probe thresholds are invalid")
        if isinstance(self.variable_order_owner_evidence_count, bool) or not isinstance(self.variable_order_owner_evidence_count, int) or self.variable_order_owner_evidence_count < 0:
            raise AssertionError("variable-order owner evidence count is invalid")
        if not math.isfinite(self.variable_order_midpoint_novelty_probability) or not 0.0 <= self.variable_order_midpoint_novelty_probability <= 1.0:
            raise AssertionError("variable-order midpoint novelty probability is invalid")
        if not isinstance(self.variable_order_temporal_refresh_used, bool):
            raise AssertionError("variable-order temporal refresh flag is invalid")
        if self.evidence_router_mode not in ("threshold", "shadow"):
            raise AssertionError("evidence router mode is invalid")
        if not math.isfinite(self.evidence_novelty_mass) or not 0.0 < self.evidence_novelty_mass < 1.0:
            raise AssertionError("evidence novelty mass is invalid")
        if isinstance(self.evidence_min_evidence, bool) or self.evidence_min_evidence < 1:
            raise AssertionError("evidence minimum evidence is invalid")
        for value in (self.evidence_enter_margin, self.evidence_exit_margin):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise AssertionError("evidence hysteresis margin is invalid")
        if self.evidence_enter_margin < self.evidence_exit_margin:
            raise AssertionError("evidence enter margin must be >= exit margin")
        if any(key not in self.motor_modules and key != "__novelty__" for key in self.evidence_posterior):
            raise AssertionError("evidence posterior has an unknown owner")
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in self.evidence_posterior.values()):
            raise AssertionError("evidence posterior is invalid")
        if self.evidence_posterior and abs(sum(self.evidence_posterior.values()) - 1.0) > 1e-9:
            raise AssertionError("evidence posterior is not normalized")
        if isinstance(self.evidence_shadow_steps, bool) or self.evidence_shadow_steps < 0:
            raise AssertionError("evidence shadow steps are invalid")
        if not isinstance(self.evidence_decisions_enabled, bool):
            raise AssertionError("evidence decisions flag is invalid")
        if not math.isfinite(float(self.evidence_novelty_fit_threshold)):
            raise AssertionError("evidence novelty fit threshold is invalid")
        if not math.isfinite(float(self.evidence_recent_best_ll)):
            raise AssertionError("evidence recent fit is invalid")
        if isinstance(self.evidence_recent_count, bool) or self.evidence_recent_count < 0:
            raise AssertionError("evidence recent count is invalid")
        if not math.isfinite(float(self.evidence_recent_decay)) or not 0.0 < float(self.evidence_recent_decay) < 1.0:
            raise AssertionError("evidence recent decay is invalid")
        pending_growth = self.evidence_pending_growth
        if pending_growth is not None:
            if not isinstance(pending_growth, dict):
                raise AssertionError("evidence pending growth is invalid")
            if pending_growth.get("module_id") not in self.motor_modules:
                raise AssertionError("evidence pending growth has an unknown module")
            if pending_growth.get("incumbent_id") not in self.motor_modules:
                raise AssertionError("evidence pending growth has an unknown incumbent")
            if int(pending_growth.get("deadline_step", -1)) < 0:
                raise AssertionError("evidence pending growth deadline is invalid")
            if not math.isfinite(float(pending_growth.get("edge_ema", 0.0))):
                raise AssertionError("evidence pending growth edge is invalid")
            if int(pending_growth.get("edge_n", -1)) < 0:
                raise AssertionError("evidence pending growth edge count is invalid")
        if (self.sequence_memory is None) != (self.sequence_memory_symbols is None):
            raise AssertionError("sequence memory and symbols must be set together")
        if self.sequence_memory is not None:
            if tuple(self.sequence_memory.symbols) != tuple(self.sequence_memory_symbols):
                raise AssertionError("sequence memory symbols disagree with memory state")
            try:
                self.sequence_memory.validate()
            except AssertionError as error:
                raise AssertionError("sequence memory is invalid: %s" % error)
        if any(key != "__novelty__" and key not in self.motor_modules for key in self.variable_order_owner_posterior):
            raise AssertionError("variable-order owner posterior has an unknown owner")
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in self.variable_order_owner_posterior.values()):
            raise AssertionError("variable-order owner posterior is invalid")
        if self.variable_order_owner_posterior and abs(sum(self.variable_order_owner_posterior.values()) - 1.0) > 1e-9:
            raise AssertionError("variable-order owner posterior is not normalized")
        if not math.isfinite(float(self.variable_order_midpoint_z_threshold)) or self.variable_order_midpoint_z_threshold < 0.0:
            raise AssertionError("variable-order midpoint z threshold is invalid")
        if not math.isfinite(float(self.variable_order_midpoint_separation)) or self.variable_order_midpoint_separation < 0.0:
            raise AssertionError("variable-order midpoint separation is invalid")
        if self.variable_order_detector_hold_until < 0 or self.variable_order_install_count < 0 or self.variable_order_route_count < 0:
            raise AssertionError("variable-order learner counters are invalid")
        if self.variable_order_representation_seed is not None and isinstance(self.variable_order_representation_seed, bool):
            raise AssertionError("variable-order representation seed is invalid")
        for feature in self.variable_order_feature_order:
            if len(feature) not in (2, 3) or tuple(sorted(feature)) != tuple(feature) or len(set(feature)) != len(feature) or any(source not in self.input_ids for source in feature):
                raise AssertionError("variable-order feature order is invalid")
        if len(set(self.variable_order_feature_order)) != len(self.variable_order_feature_order):
            raise AssertionError("variable-order feature order contains duplicates")
        for module_id, key in self.variable_order_module_features.items():
            if module_id not in self.motor_modules:
                raise AssertionError("variable-order module feature has unknown module")
            feature = self._dendritic_feature_from_key(key)
            if self.variable_order_feature_owners.get(key) != module_id:
                raise AssertionError("variable-order module feature mapping is inconsistent")
        if self.variable_order_learning_enabled and not self._variable_order_internal_transition:
            direct_owner_values = list(self.variable_order_feature_owners.values())
            if len(set(direct_owner_values)) != len(direct_owner_values):
                raise AssertionError("variable-order learner cannot assign multiple direct features to one module")
            for key, module_id in self.variable_order_feature_owners.items():
                if self.variable_order_module_features.get(module_id) != key:
                    raise AssertionError("variable-order forward/reverse feature maps are inconsistent")
        if not isinstance(self.compositional_substrate_enabled, bool):
            raise AssertionError("compositional substrate enabled flag is invalid")
        if isinstance(self.max_composed_leaves, bool) or not isinstance(self.max_composed_leaves, int) or self.max_composed_leaves != 4:
            raise AssertionError("compositional substrate currently requires four leaves")
        if self.composition_representation_seed is not None and isinstance(self.composition_representation_seed, bool):
            raise AssertionError("composition representation seed is invalid")
        if self.composition_install_count < 0:
            raise AssertionError("composition install count is invalid")
        if not math.isfinite(self.composition_signal_gain) or not 1.0 <= self.composition_signal_gain <= 2.0:
            raise AssertionError("composition signal gain is invalid")
        if not isinstance(self.composition_learning_enabled, bool):
            raise AssertionError("composition learning flag is invalid")
        if not isinstance(self.composition_direct_route_pending, bool):
            raise AssertionError("composition direct-route verification flag is invalid")
        if isinstance(self.composition_verification_suppressed_until, bool) or self.composition_verification_suppressed_until < 0:
            raise AssertionError("composition verification suppression horizon is invalid")
        if self.composition_learning_enabled and not self.compositional_substrate_enabled:
            raise AssertionError("composition learner requires compositional substrate")
        if self.composition_route_count < 0 or self.composition_candidate_count < 0:
            raise AssertionError("composition learner counters are invalid")
        if not isinstance(self.composition_fingerprint_evidence, dict):
            raise AssertionError("composition fingerprint evidence must be a mapping")
        if not isinstance(self.composition_feature_owners, dict) or not isinstance(self.composition_module_features, dict) or not isinstance(self.composition_feature_lineage, dict):
            raise AssertionError("composition ownership state must be mappings")
        if not self.compositional_substrate_enabled and (
            self.composition_feature_owners or self.composition_module_features or self.composition_feature_lineage
        ):
            raise AssertionError("disabled compositional substrate cannot retain learned features")
        if len(self.composition_feature_owners) != len(self.composition_feature_lineage):
            raise AssertionError("composition owner and lineage maps must have equal keys")
        if set(self.composition_module_features.values()) != set(self.composition_feature_owners):
            raise AssertionError("composition reverse ownership map is inconsistent")
        if len(set(self.composition_module_features.values())) != len(self.composition_module_features):
            raise AssertionError("composition reverse ownership values must be unique")
        for module_id, key in self.composition_module_features.items():
            if module_id not in self.motor_modules or self.composition_feature_owners.get(key) != module_id:
                raise AssertionError("composition reverse ownership contains an unknown or aliased module")
        if set(self.variable_order_module_features) & set(self.composition_module_features):
            raise AssertionError("a module cannot own both direct and composed learned features")
        if set(self.variable_order_feature_owners.values()) & set(self.composition_module_features):
            raise AssertionError("a module cannot own both direct feature-map and composed learned features")
        if set(self.adaptive_dendritic_pair_owners.values()) & set(self.composition_module_features):
            raise AssertionError("a module cannot own both adaptive direct and composed learned features")
        for key, owner_id in self.composition_feature_owners.items():
            if not isinstance(key, str) or not key.startswith("c4:") or owner_id not in self.motor_modules:
                raise AssertionError("composition feature owner is invalid")
            owner = self.motor_modules[owner_id]
            if owner.afferent_kind != "hidden" or self.composition_module_features.get(owner_id) != key:
                raise AssertionError("composition feature owner module mapping is invalid")
            lineage = self.composition_feature_lineage.get(key)
            if not isinstance(lineage, Mapping):
                raise AssertionError("composition feature lineage is missing")
            source_ids = tuple(sorted(str(value) for value in lineage.get("source_cells", ())))
            raw_leaves = lineage.get("leaf_closure", ())
            if len(source_ids) != 2 or len(set(source_ids)) != 2 or not isinstance(raw_leaves, (list, tuple)) or len(raw_leaves) != 2:
                raise AssertionError("composition feature lineage shape is invalid")
            leaves = [tuple(sorted(str(value) for value in closure)) for closure in raw_leaves]
            if any(len(leaf) != 2 or len(set(leaf)) != 2 or any(value not in self.input_ids for value in leaf) for leaf in leaves):
                raise AssertionError("composition feature lineage leaves are invalid")
            if set(leaves[0]) & set(leaves[1]) or len(set(leaves[0] + leaves[1])) != 4:
                raise AssertionError("composition feature lineage leaves overlap")
            if self._composition_feature_key(source_ids, leaves) != key:
                raise AssertionError("composition feature key does not match lineage")
            for source_id, leaf in zip(source_ids, leaves):
                source = self.cells.get(source_id)
                if source is None or source.activation_type != "dendritic_product" or tuple(source.dendritic_sources) != leaf:
                    raise AssertionError("composition lineage source is not the declared direct pair")
            composed_cells = [
                cell for cell in self.cells.values()
                if cell.activation_type == "dendritic_product_composed" and tuple(cell.dendritic_sources) == source_ids
            ]
            if len(composed_cells) != 1 or self.graph.get(composed_cells[0].id, owner.cell_id) is None:
                raise AssertionError("composition owner lacks unique composed feature cell")
        if self.composition_install_count < len(self.composition_feature_owners):
            raise AssertionError("composition install count predates owned features")
        current_composition_keys = {candidate[0] for candidate in self._composition_feature_bank()}
        for key, value in self.composition_fingerprint_evidence.items():
            if key not in current_composition_keys:
                raise AssertionError("composition fingerprint evidence is not a current graph candidate")
            if not isinstance(value, Mapping) or any(not math.isfinite(float(value.get(name, 0.0))) for name in ("sum", "sumsq", "count")) or float(value.get("count", 0.0)) < 0.0 or float(value.get("sumsq", 0.0)) < 0.0:
                raise AssertionError("composition fingerprint evidence is invalid")
        pending_composition = self.composition_fingerprint_pending
        if pending_composition is not None:
            if not self.composition_learning_enabled or not isinstance(pending_composition, Mapping):
                raise AssertionError("composition fingerprint pending state is invalid")
            owner_id = pending_composition.get("owner_module")
            step = pending_composition.get("step")
            if owner_id != self.variable_order_fingerprint_owner or owner_id not in self.motor_modules:
                raise AssertionError("composition fingerprint pending owner is invalid")
            if isinstance(step, bool) or not isinstance(step, int) or step != self.step_count:
                raise AssertionError("composition fingerprint pending step is invalid")
            if not self.variable_order_fingerprint_active or step < self.variable_order_fingerprint_start_step or step > self.variable_order_fingerprint_deadline:
                raise AssertionError("composition fingerprint pending timing is invalid")
            entries = pending_composition.get("eligibilities")
            if not isinstance(entries, list):
                raise AssertionError("composition fingerprint pending eligibilities are invalid")
            seen = set()
            for entry in entries:
                key = entry.get("key") if isinstance(entry, Mapping) else None
                if key not in current_composition_keys or key in seen or not math.isfinite(float(entry.get("eligibility", 0.0))):
                    raise AssertionError("composition fingerprint pending candidate is invalid")
                seen.add(key)
        if self.variable_order_learning_enabled and not self.variable_order_enabled:
            raise AssertionError("variable-order learner requires enabled substrate")
        if not isinstance(self.general_structural_learning_enabled, bool):
            raise AssertionError("general structural learner flag is invalid")
        if self.general_structural_max_depth < 1 or self.general_structural_max_leaves < 2 or self.general_feature_prune_count < 0:
            raise AssertionError("general structural bounds are invalid")
        if isinstance(self.general_prune_reuse_ceiling, bool) or not isinstance(self.general_prune_reuse_ceiling, int) or self.general_prune_reuse_ceiling < 0:
            raise AssertionError("general prune reuse ceiling is invalid")
        if self.general_required_confirmations < 2 or self.general_novelty_confirmations < 0:
            raise AssertionError("general novelty confirmation state is invalid")
        if self.general_audit_interval < 16 or self.general_last_audit_step < 0:
            raise AssertionError("general structural audit state is invalid")
        if not isinstance(self.delayed_credit_enabled, bool) or self.delayed_credit_delay < 0 or self.delayed_credit_delay > 32:
            raise AssertionError("delayed credit configuration is invalid")
        if not self.delayed_credit_enabled and (self.delayed_credit_delay != 0 or self.delayed_credit_queue):
            raise AssertionError("disabled delayed credit retains state")
        if len(self.delayed_credit_queue) > self.delayed_credit_delay:
            raise AssertionError("delayed credit queue exceeds its bound")
        for snapshot in self.delayed_credit_queue:
            if not isinstance(snapshot, Mapping) or snapshot.get("pending_motor_module") not in self.motor_modules:
                raise AssertionError("delayed credit snapshot owner is invalid")
            if not isinstance(snapshot.get("actor_traces", {}), Mapping):
                raise AssertionError("delayed credit actor traces are invalid")
        if self.general_novelty_candidate is not None and not isinstance(self.general_novelty_candidate, str):
            raise AssertionError("general novelty candidate is invalid")
        if set(self.general_feature_owners) != set(self.general_feature_lineage) or set(self.general_module_features.values()) != set(self.general_feature_owners):
            raise AssertionError("general feature ownership and lineage are inconsistent")
        if not self.general_structural_learning_enabled and self.general_feature_owners:
            raise AssertionError("disabled general structural learner owns features")
        for key, owner_id in self.general_feature_owners.items():
            lineage = self.general_feature_lineage[key]
            cell_id = str(lineage.get("cell_id", ""))
            sources = tuple(str(value) for value in lineage.get("source_cells", ()))
            leaves = tuple(str(value) for value in lineage.get("leaf_closure", ()))
            depth = int(lineage.get("depth", 0))
            if owner_id not in self.motor_modules or self.general_module_features.get(owner_id) != key:
                raise AssertionError("general feature owner is invalid")
            if cell_id not in self.cells or self.cells[cell_id].metadata.get("general_feature_key") != key:
                raise AssertionError("general feature cell is invalid")
            if tuple(self.cells[cell_id].dendritic_sources) != sources or len(sources) != 2:
                raise AssertionError("general feature sources are invalid")
            derived = [self._general_source_lineage(source) for source in sources]
            if any(value is None for value in derived):
                raise AssertionError("general feature source lineage is invalid")
            actual_leaves = tuple(sorted(derived[0][0] + derived[1][0]))
            actual_depth = max(derived[0][1], derived[1][1]) + 1
            if leaves != actual_leaves or depth != actual_depth or len(leaves) > self.general_structural_max_leaves or depth > self.general_structural_max_depth:
                raise AssertionError("general feature closure exceeds persisted bounds")
            if self.general_feature_reuse.get(key, -1) < 0:
                raise AssertionError("general feature reuse count is invalid")
            if self.general_feature_last_used_step.get(key, -1) < 0:
                raise AssertionError("general feature last-use step is invalid")
        general_keys = {str(item["key"]) for item in self.general_feature_candidates()}
        for key, value in self.general_fingerprint_evidence.items():
            if key not in general_keys or any(not math.isfinite(float(value.get(name, 0.0))) for name in ("sum", "sumsq", "count")):
                raise AssertionError("general fingerprint evidence is invalid")
        if self.variable_order_fingerprint_active:
            if self.variable_order_fingerprint_owner not in self.motor_modules or self.variable_order_fingerprint_incumbent not in self.motor_modules:
                raise AssertionError("variable-order fingerprint owner is unknown")
        for key, value in self.variable_order_fingerprint_evidence.items():
            self._dendritic_feature_from_key(key)
            if any(not math.isfinite(float(value.get(name, 0.0))) for name in ("sum", "sumsq", "count")) or float(value.get("count", 0.0)) < 0.0 or float(value.get("sumsq", 0.0)) < 0.0:
                raise AssertionError("variable-order fingerprint evidence is invalid")
        if self.variable_order_normal_count < 0 or self.variable_order_normal_count > self.variable_order_normal_window:
            raise AssertionError("variable-order normal evidence count is invalid")
        for key, value in self.variable_order_normal_evidence.items():
            self._dendritic_feature_from_key(key)
            if any(not math.isfinite(float(value.get(name, 0.0))) for name in ("sum", "sumsq", "count")) or float(value.get("count", 0.0)) < 0.0 or float(value.get("sumsq", 0.0)) < 0.0:
                raise AssertionError("variable-order normal evidence is invalid")
        self._validate_variable_order_pending(self.variable_order_fingerprint_pending, "fingerprint")
        self._validate_variable_order_pending(self.variable_order_normal_pending, "normal")
        if self.variable_order_fingerprint_active and self._pending_outcome and self.variable_order_fingerprint_pending is None and not self._variable_order_internal_transition:
            raise AssertionError("active variable-order fingerprint lacks pending action record")
        if not math.isfinite(self.motor_bootstrap_scale) or not 0.0 <= self.motor_bootstrap_scale <= 1.0:
            raise AssertionError("motor bootstrap scale is invalid")
        if self.adaptive_dendritic_proposal_interval < 1:
            raise AssertionError("adaptive dendritic proposal interval must be positive")
        if isinstance(self.adaptive_dendritic_min_evidence, bool) or self.adaptive_dendritic_min_evidence < 1:
            raise AssertionError("adaptive dendritic minimum evidence must be positive")
        if isinstance(self.adaptive_dendritic_max_trial_steps, bool) or self.adaptive_dendritic_max_trial_steps < self.adaptive_dendritic_min_evidence:
            raise AssertionError("adaptive dendritic maximum trial duration is invalid")
        if not math.isfinite(float(self.adaptive_dendritic_confidence_z)) or self.adaptive_dendritic_confidence_z < 0.0:
            raise AssertionError("adaptive dendritic confidence multiplier is invalid")
        if not self.adaptive_dendritic_shadow_checkpoints or tuple(sorted(self.adaptive_dendritic_shadow_checkpoints)) != self.adaptive_dendritic_shadow_checkpoints or any(int(value) < 1 for value in self.adaptive_dendritic_shadow_checkpoints):
            raise AssertionError("adaptive dendritic shadow checkpoints are invalid")
        if not math.isfinite(float(self.adaptive_dendritic_shadow_confidence_z)) or self.adaptive_dendritic_shadow_confidence_z < 0.0:
            raise AssertionError("adaptive dendritic shadow confidence is invalid")
        if not math.isfinite(float(self.adaptive_dendritic_shadow_separation)) or self.adaptive_dendritic_shadow_separation < 0.0:
            raise AssertionError("adaptive dendritic shadow separation is invalid")
        for module_id, entries in self.adaptive_dendritic_shadow_evidence.items():
            if module_id not in self.motor_modules:
                raise AssertionError("adaptive shadow evidence has unknown module")
            for key, value in entries.items():
                pair = self._adaptive_dendritic_pair_from_key(key)
                if any(source not in self.input_ids for source in pair):
                    raise AssertionError("adaptive shadow evidence has unknown pair")
                if any(not math.isfinite(float(value.get(name, 0.0))) for name in ("sum", "sumsq", "count")) or float(value.get("count", 0.0)) < 0.0:
                    raise AssertionError("adaptive shadow evidence is invalid")
        if any(module_id not in self.motor_modules for module_id in self.adaptive_dendritic_shadow_search_exhausted):
            raise AssertionError("adaptive shadow exhaustion has unknown module")
        pending_shadow = self.adaptive_dendritic_shadow_pending
        if pending_shadow is not None:
            if str(pending_shadow.get("owner_module", "")) not in self.motor_modules or not isinstance(pending_shadow.get("step"), int):
                raise AssertionError("adaptive shadow pending owner/step is invalid")
            for name in ("module_mean", "exploration_value", "exploration_sigma"):
                if not math.isfinite(float(pending_shadow.get(name, 0.0))):
                    raise AssertionError("adaptive shadow pending value is invalid")
            for entry in pending_shadow.get("eligibilities", ()):
                pair = tuple(str(value) for value in entry.get("pair", ()))
                if len(pair) != 2 or any(source not in self.input_ids for source in pair) or not math.isfinite(float(entry.get("eligibility", 0.0))):
                    raise AssertionError("adaptive shadow pending pair is invalid")
        if self.adaptive_dendritic_fingerprint_window < 1 or self.adaptive_dendritic_fingerprint_count < 0 or self.adaptive_dendritic_fingerprint_count > self.adaptive_dendritic_fingerprint_window:
            raise AssertionError("adaptive fingerprint bounds are invalid")
        if self.adaptive_dendritic_fingerprint_start_step < 0 or self.adaptive_dendritic_fingerprint_deadline < self.adaptive_dendritic_fingerprint_start_step:
            raise AssertionError("adaptive fingerprint timing is invalid")
        if self.adaptive_dendritic_detector_hold_until < 0:
            raise AssertionError("adaptive detector hold timing is invalid")
        if self.adaptive_dendritic_fingerprint_owner is not None and self.adaptive_dendritic_fingerprint_owner not in self.motor_modules:
            raise AssertionError("adaptive fingerprint owner is unknown")
        if self.adaptive_dendritic_fingerprint_incumbent is not None and self.adaptive_dendritic_fingerprint_incumbent not in self.motor_modules:
            raise AssertionError("adaptive fingerprint incumbent is unknown")
        product_cells_by_pair: Dict[Tuple[str, str], List[str]] = {}
        for cell in self.cells.values():
            if cell.activation_type != "dendritic_product":
                continue
            pair = tuple(str(source) for source in cell.dendritic_sources)
            if len(pair) == 2 and all(source in self.input_ids for source in pair):
                product_cells_by_pair.setdefault(pair, []).append(cell.id)
        owners_seen: Dict[str, Tuple[str, str]] = {}
        for key, owner_id in self.adaptive_dendritic_pair_owners.items():
            pair = self._adaptive_dendritic_pair_from_key(key)
            if tuple(sorted(pair)) != pair or any(source not in self.input_ids for source in pair) or owner_id not in self.motor_modules:
                raise AssertionError("adaptive fingerprint pair owner is invalid")
            product_ids = product_cells_by_pair.get(pair, [])
            if len(product_ids) != 1:
                raise AssertionError("adaptive fingerprint pair owner has no unique product cell")
            previous_pair = owners_seen.get(owner_id)
            if previous_pair is not None and previous_pair != pair:
                raise AssertionError("adaptive fingerprint module owns multiple product pairs")
            owners_seen[owner_id] = pair
            owner_cell_id = self.motor_modules[owner_id].cell_id
            if self.graph.get(product_ids[0], owner_cell_id) is None:
                raise AssertionError("adaptive fingerprint owner is missing product motor afferent")
        for key, value in self.adaptive_dendritic_fingerprint_evidence.items():
            self._adaptive_dendritic_pair_from_key(key)
            if any(not math.isfinite(float(value.get(name, 0.0))) for name in ("sum", "sumsq", "count")) or float(value.get("count", 0.0)) < 0.0 or float(value.get("sumsq", 0.0)) < 0.0:
                raise AssertionError("adaptive fingerprint evidence is invalid")
        # One canonical feature has exactly one owner; an owner may retain
        # multiple distinct features.  Pair ownership remains in its frozen
        # v9 map and is validated separately above.
        product_cells_by_feature: Dict[Tuple[str, ...], List[str]] = {}
        for cell in self.cells.values():
            if cell.activation_type in ("dendritic_product", "dendritic_product_n"):
                product_cells_by_feature.setdefault(tuple(cell.dendritic_sources), []).append(cell.id)
        if any(
            len(product_ids) > 1
            and (len(feature) >= 3 or self._dendritic_feature_key(feature) in self.variable_order_feature_owners)
            for feature, product_ids in product_cells_by_feature.items()
        ):
            raise AssertionError("duplicate variable-order feature tuple")
        for key, owner_id in self.variable_order_feature_owners.items():
            feature = self._dendritic_feature_from_key(key)
            if len(feature) not in (2, 3) or len(feature) > self.max_dendritic_order or any(source not in self.input_ids for source in feature):
                raise AssertionError("variable-order feature owner has invalid feature")
            if owner_id not in self.motor_modules:
                raise AssertionError("variable-order feature owner is unknown")
            owner = self.motor_modules[owner_id]
            if owner.afferent_kind not in ("input", "hidden"):
                raise AssertionError("variable-order feature owner has invalid afferents")
            product_ids = product_cells_by_feature.get(feature, [])
            if len(product_ids) != 1 or self.graph.get(product_ids[0], owner.cell_id) is None:
                raise AssertionError("variable-order feature owner lacks unique product motor edge")
            product_cell = self.cells[product_ids[0]]
            expected_type = "dendritic_product" if len(feature) == 2 else "dendritic_product_n"
            if product_cell.activation_type != expected_type:
                raise AssertionError("variable-order feature owner has inconsistent product order")
        for pair_key, adaptive_owner in self.adaptive_dendritic_pair_owners.items():
            pair = self._adaptive_dendritic_pair_from_key(pair_key)
            generic_key = self._dendritic_feature_key(pair)
            generic_owner = self.variable_order_feature_owners.get(generic_key)
            if generic_owner is not None and generic_owner != adaptive_owner:
                raise AssertionError("binary pair has contradictory adaptive and variable-order owners")
        pending_fingerprint = self.adaptive_dendritic_fingerprint_pending
        if pending_fingerprint is not None:
            if pending_fingerprint.get("owner_module") not in self.motor_modules or not isinstance(pending_fingerprint.get("step"), int):
                raise AssertionError("adaptive fingerprint pending owner/step is invalid")
            for entry in pending_fingerprint.get("eligibilities", ()):
                pair = tuple(str(value) for value in entry.get("pair", ()))
                if len(pair) != 2 or any(source not in self.input_ids for source in pair) or not math.isfinite(float(entry.get("eligibility", 0.0))):
                    raise AssertionError("adaptive fingerprint pending pair is invalid")
        for module_id, pairs in self.adaptive_dendritic_pairs_tried.items():
            if module_id not in self.motor_modules or len(pairs) > len(list(combinations(self.input_ids, 2))):
                raise AssertionError("adaptive dendritic search state is invalid")
            if len(set(tuple(pair) for pair in pairs)) != len(pairs):
                raise AssertionError("adaptive dendritic search repeats a pair")
            if any(len(pair) != 2 or pair[0] not in self.input_ids or pair[1] not in self.input_ids or pair[0] == pair[1] for pair in pairs):
                raise AssertionError("adaptive dendritic search contains invalid pair")
        if any(module_id not in self.motor_modules for module_id in self.adaptive_dendritic_module_accepted):
            raise AssertionError("adaptive dendritic acceptance state has unknown module")
        if self.adaptive_dendritic_proposal is not None:
            proposal = self.adaptive_dendritic_proposal
            cell_id = str(proposal.get("cell_id"))
            if cell_id not in self.cells or self.cells[cell_id].kind != "hidden":
                raise AssertionError("adaptive proposal cell is invalid")
            cell = self.cells[cell_id]
            if cell.activation_type != "dendritic_product" or list(cell.dendritic_sources) != list(proposal.get("new_sources", ())):
                raise AssertionError("adaptive proposal does not match product cell")
            current_ids = {(synapse.source, synapse.destination) for synapse in self.graph.iter_synapses()}
            for payload in proposal.get("new_synapses", []):
                if (str(payload["source"]), str(payload["destination"])) not in current_ids:
                    raise AssertionError("adaptive proposal is missing new synapse")
            if str(proposal.get("owner_module", "")) not in self.motor_modules:
                raise AssertionError("adaptive proposal owner is invalid")
            if int(proposal.get("min_trial_steps", self.adaptive_dendritic_min_evidence)) != self.adaptive_dendritic_min_evidence or int(proposal.get("max_trial_steps", self.adaptive_dendritic_max_trial_steps)) != self.adaptive_dendritic_max_trial_steps:
                raise AssertionError("adaptive proposal trial parameters changed mid-trial")
            if int(proposal.get("trial_min_step", 0)) > int(proposal.get("trial_max_step", 0)):
                raise AssertionError("adaptive proposal trial bounds are invalid")
            if int(proposal.get("evidence_count", 0)) != len(proposal.get("trial_evidence", [])):
                raise AssertionError("adaptive proposal evidence count is inconsistent")
            if int(proposal.get("evidence_count", 0)) != len(proposal.get("trial_eligibilities", [])):
                raise AssertionError("adaptive proposal eligibility count is inconsistent")
            if any(not math.isfinite(float(value)) for value in proposal.get("trial_eligibilities", ()) + proposal.get("trial_evidence", ())):
                raise AssertionError("adaptive proposal trial evidence is not finite")
            for name in ("old_activation", "new_activation", "counterfactual_activation", "counterfactual_drive", "local_actor_eligibility", "local_node_eligibility", "local_eligibility", "pending_local_eligibility", "evidence_sum", "evidence_sumsq"):
                if not math.isfinite(float(proposal.get(name, 0.0))):
                    raise AssertionError("adaptive proposal value is not finite")
        for cell in self.cells.values():
            if cell.activation_type not in ("additive", "dendritic_product", "dendritic_product_n", "dendritic_product_composed"):
                raise AssertionError("invalid cell activation type: %s" % cell.id)
            if cell.activation_type == "additive":
                if cell.dendritic_sources:
                    raise AssertionError("additive cell has dendritic sources: %s" % cell.id)
            elif cell.activation_type == "dendritic_product":
                if cell.kind != "hidden" or len(cell.dendritic_sources) != 2 or len(set(cell.dendritic_sources)) != 2:
                    raise AssertionError("product cell must declare exactly two distinct sources: %s" % cell.id)
                if any(source not in self.cells for source in cell.dendritic_sources):
                    raise AssertionError("product cell has dangling source: %s" % cell.id)
                incoming = set(self.graph.incoming.get(cell.id, []))
                if incoming != set(cell.dendritic_sources):
                    raise AssertionError("product cell incoming edges do not match declared sources: %s" % cell.id)
                if any(self.graph.get(source, cell.id).transmission_delay != 0 for source in cell.dendritic_sources):
                    raise AssertionError("product cell afferents must have zero delay: %s" % cell.id)
            else:
                if cell.activation_type == "dendritic_product_composed":
                    general_key = cell.metadata.get("general_feature_key")
                    if general_key is not None:
                        if not self.general_structural_learning_enabled or general_key not in self.general_feature_lineage:
                            raise AssertionError("general product cell lacks enabled persisted lineage")
                        if cell.kind != "hidden" or tuple(sorted(cell.dendritic_sources)) != tuple(cell.dendritic_sources) or len(set(cell.dendritic_sources)) != 2:
                            raise AssertionError("general product cell must declare two canonical distinct sources")
                        if any(self._general_source_lineage(source) is None for source in cell.dendritic_sources):
                            raise AssertionError("general product cell source lacks graph lineage")
                        incoming = set(self.graph.incoming.get(cell.id, []))
                        if incoming != set(cell.dendritic_sources) or any(self.graph.get(source, cell.id).transmission_delay != 0 for source in cell.dendritic_sources):
                            raise AssertionError("general product incoming edges are invalid")
                    elif not self.compositional_substrate_enabled or len(cell.dendritic_sources) != 2:
                        raise AssertionError("composed product cell requires enabled substrate and two sources")
                    elif cell.kind != "hidden" or tuple(sorted(cell.dendritic_sources)) != tuple(cell.dendritic_sources) or len(set(cell.dendritic_sources)) != 2:
                        raise AssertionError("composed product cell must declare two canonical distinct sources")
                    source_cells = [self.cells.get(source) for source in cell.dendritic_sources]
                    if general_key is None and any(source is None or source.kind != "hidden" or source.activation_type != "dendritic_product" for source in source_cells):
                        raise AssertionError("composed product sources must be direct binary product cells")
                    leaves = [tuple(source.dendritic_sources) for source in source_cells if source is not None]
                    if general_key is None and (any(len(leaf) != 2 or any(value not in self.input_ids for value in leaf) for leaf in leaves) or set(leaves[0]) & set(leaves[1]) or len(set(leaves[0] + leaves[1])) != 4):
                        raise AssertionError("composed product leaves must be disjoint four-input closure")
                    incoming = set(self.graph.incoming.get(cell.id, []))
                    if incoming != set(cell.dendritic_sources) or any(self.graph.get(source, cell.id).transmission_delay != 0 for source in cell.dendritic_sources):
                        raise AssertionError("composed product incoming edges are invalid")
                else:
                    if not self.variable_order_enabled or len(cell.dendritic_sources) < 3 or len(cell.dendritic_sources) > self.max_dendritic_order:
                        raise AssertionError("n-ary product cell requires enabled variable-order substrate")
                    if cell.kind != "hidden" or tuple(sorted(cell.dendritic_sources)) != tuple(cell.dendritic_sources) or len(set(cell.dendritic_sources)) != len(cell.dendritic_sources):
                        raise AssertionError("n-ary product cell must declare canonical distinct sources")
                    if any(source not in self.input_ids for source in cell.dendritic_sources):
                        raise AssertionError("n-ary product cell sources must be direct input IDs")
                    incoming = set(self.graph.incoming.get(cell.id, []))
                    if incoming != set(cell.dendritic_sources):
                        raise AssertionError("n-ary product cell incoming edges do not match declared sources")
                    if any(self.graph.get(source, cell.id).transmission_delay != 0 for source in cell.dendritic_sources):
                        raise AssertionError("n-ary product cell afferents must have zero delay")
            normalizer_upper = 2.0 if cell.activation_type == "dendritic_product_composed" else 1.0
            for name, value, lower, upper in (("activation", cell.activation, -1.0, 1.0), ("plasticity", cell.plasticity, 0.0, 1.0), ("activity_ema", cell.activity_ema, 0.0, 1.0), ("signed_activity_ema", cell.signed_activity_ema, -1.0, 1.0), ("dendritic_normalizer", cell.dendritic_normalizer, 0.0, normalizer_upper)):
                if not math.isfinite(value) or not lower <= value <= upper:
                    raise AssertionError("cell %s out of bounds: %s" % (name, cell.id))
            for name, value, lower, upper in (("threshold", cell.threshold, -1.5, 1.5), ("adaptation", cell.adaptation, 0.0, 1.0), ("utility", cell.utility, 0.0, 1.0), ("energy_usage", cell.energy_usage, 0.0, float("inf")), ("representation_bias", cell.representation_bias, -1.0, 1.0), ("representation_bias_eligibility", cell.representation_bias_eligibility, -100.0, 100.0)):
                if not math.isfinite(value) or value < lower or value > upper:
                    raise AssertionError("cell %s out of bounds: %s" % (name, cell.id))
            if cell.age < 0 or cell.refractory < 0:
                raise AssertionError("cell age/refractory must be nonnegative: %s" % cell.id)
        for cell in self.cells.values():
            if cell.activation_type == "dendritic_product_composed":
                general_key = cell.metadata.get("general_feature_key")
                if general_key is not None:
                    if self.general_feature_owners.get(general_key) not in self.motor_modules or self.general_feature_lineage.get(general_key) is None:
                        raise AssertionError("general product cell lacks canonical ownership")
                    continue
                leaves = [tuple(self.cells[source].dendritic_sources) for source in cell.dendritic_sources]
                key = self._composition_feature_key(cell.dendritic_sources, leaves)
                if self.composition_feature_owners.get(key) not in self.motor_modules or self.composition_feature_lineage.get(key) is None:
                    raise AssertionError("composed product cell lacks canonical ownership")
        if not math.isfinite(self.reward_baseline) or not math.isfinite(self.reward_baseline_rate) or not 0.0 <= self.reward_baseline_rate <= 1.0:
            raise AssertionError("reward baseline state must be finite and bounded")
        if not math.isfinite(self.actor_learning_rate) or self.actor_learning_rate < 0.0:
            raise AssertionError("actor learning rate must be finite and nonnegative")
        if not math.isfinite(self.representation_learning_rate) or self.representation_learning_rate < 0.0 or not math.isfinite(self.representation_noise) or not 0.0 <= self.representation_noise <= 1.0:
            raise AssertionError("representation learning parameters are invalid")
        if self.representation_cursor < 0 or (self.representation_tape is not None and self.representation_cursor > len(self.representation_tape)):
            raise AssertionError("representation tape cursor is invalid")
        if self.representation_tape is not None and not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in self.representation_tape):
            raise AssertionError("representation tape values are invalid")
        if not math.isfinite(self.resources.energy_used) or self.resources.energy_used < 0.0:
            raise AssertionError("resource energy_used must be finite and nonnegative")
        if self.resources.energy_used > self.resources.energy_per_step + 1e-12:
            raise AssertionError("per-step energy budget exceeded")
        if not math.isfinite(self.resources.total_energy) or self.resources.total_energy < 0.0:
            raise AssertionError("resource total_energy must be finite and nonnegative")
        for synapse in self.graph.iter_synapses():
            if not -1.0 <= synapse.strength <= 1.0:
                raise AssertionError("strength out of bounds")
        if self.context_enabled:
            if not self.motor_modules or self.active_motor_module not in self.motor_modules:
                raise AssertionError("context organism requires one active motor module")
            if self.context_max_modules < 1 or len(self.motor_modules) > self.context_max_modules:
                raise AssertionError("motor module cap exceeded")
            if any(identifier not in self.motor_modules for identifier in self.context_probe_queue):
                raise AssertionError("probe queue contains unknown module")
            if len(set(self.context_probe_queue)) != len(self.context_probe_queue):
                raise AssertionError("probe queue contains duplicates")
            if self.context_probe_module is not None and self.context_probe_module not in self.motor_modules:
                raise AssertionError("active probe module is unknown")
            if self.context_probe_module is not None and self.context_probe_module in self.context_probe_queue:
                raise AssertionError("active probe module is still queued")
            if self.context_state not in ("normal", "warning", "search", "fingerprint"):
                raise AssertionError("invalid context state")
            if self.context_probe_origin not in (None, "warning", "alarm", "owner_evidence", "owner_evidence_midpoint", "evidence_probe"):
                raise AssertionError("invalid context probe origin")
            if self.context_probe_module is not None and self.context_probe_origin is None:
                raise AssertionError("active probe is missing its origin")
            if self.context_probe_module is None and self.context_probe_origin is not None:
                raise AssertionError("probe origin without an active probe")
            if self.context_pending_mode not in ("normal", "warning", "search", "fingerprint"):
                raise AssertionError("invalid pending context mode")
            if not math.isfinite(self.context_pending_action) or not math.isfinite(self.context_pending_prediction) or not math.isfinite(self.context_pending_scale) or self.context_pending_scale <= 0.0:
                raise AssertionError("invalid pending context record")
            expected_features = self._context_feature_dimension()
            if self.context_pending_features and (len(self.context_pending_features) != expected_features or not all(math.isfinite(float(value)) for value in self.context_pending_features)):
                raise AssertionError("invalid pending context features")
            for name in ("context_switch_threshold", "context_warning_threshold", "context_confidence_threshold", "context_safe_action", "context_reactivation_margin", "context_probe_reward_floor", "context_surprise_mean", "context_surprise_variance", "context_surprise_drift", "context_detector_freeze_threshold", "context_detector_drift", "context_detector_recovery", "context_detector_predictor_rate", "context_probe_information_threshold", "context_probe_exploration"):
                if not math.isfinite(getattr(self, name)):
                    raise AssertionError("context parameter must be finite: %s" % name)
            if self.context_switch_threshold <= self.context_warning_threshold or self.context_warning_threshold < 0.0 or self.context_confidence_threshold < 0.0 or not -1.0 <= self.context_safe_action <= 1.0 or self.context_probe_reward_floor < 0.0 or self.context_surprise_drift < 0.0 or self.context_detector_freeze_threshold < 0.0 or self.context_detector_drift < 0.0 or self.context_detector_recovery < 0.0 or self.context_detector_predictor_rate < 0.0 or self.context_probe_information_threshold < 0.0 or self.context_probe_exploration < 0.0:
                raise AssertionError("context thresholds/drift must be nonnegative")
            if not 0.0 < self.context_surprise_leak <= 1.0 or self.context_surprise_variance <= 0.0:
                raise AssertionError("context surprise parameters are invalid")
            if self.context_negative_streak < 1 or self.context_probe_steps < 1 or self.context_detector_min_evidence < 1 or self.context_probe_min_steps < 2 or self.context_probe_max_steps < self.context_probe_min_steps:
                raise AssertionError("context probe bounds are invalid")
            if self.context_probe_remaining < 0 or self.context_probe_count < 0 or len(self.context_probe_rewards) != self.context_probe_count:
                raise AssertionError("context probe state is invalid")
            if self.context_probe_steps_seen < 0 or self.context_probe_informative_count < 0 or self.context_probe_informative_count != self.context_probe_count:
                raise AssertionError("context probe evidence state is invalid")
            for name in ("context_probe_failed_reference", "context_probe_reference", "context_last_reward_value", "context_probe_sum", "context_probe_sumsq"):
                if not math.isfinite(getattr(self, name)):
                    raise AssertionError("context probe state must be finite: %s" % name)
            if self.context_probe_sumsq < -1e-12:
                raise AssertionError("context probe sumsq must be nonnegative")
            for identifier, module in self.motor_modules.items():
                if module.afferent_kind not in ("input", "hidden"):
                    raise AssertionError("invalid motor module afferent kind")
                if module.id != identifier or module.cell_id not in self.cells or self.cells[module.cell_id].kind != "motor_module":
                    raise AssertionError("motor module identity mismatch")
                allowed_sources = set(self._module_source_ids(module.afferent_kind))
                allowed_sources.update(self.variable_order_extra_sources.get(identifier, ()))
                if any(synapse.source not in allowed_sources for synapse in self.graph.iter_synapses() if synapse.destination == module.cell_id):
                    raise AssertionError("motor module has invalid afferent source")
        if not isinstance(self.variable_order_extra_sources, dict):
            raise AssertionError("variable-order extra sources must be a mapping")
        for module_id, cell_ids in self.variable_order_extra_sources.items():
            if module_id not in self.motor_modules:
                raise AssertionError("variable-order extra sources reference an unknown module")
            if not isinstance(cell_ids, list) or not cell_ids:
                raise AssertionError("variable-order extra sources must be nonempty lists")
            for cell_id in cell_ids:
                cell = self.cells.get(cell_id)
                if cell is None or cell.activation_type not in ("dendritic_product", "dendritic_product_n"):
                    raise AssertionError("variable-order extra source is not a product cell")
                if self.graph.get(cell_id, self.motor_modules[module_id].cell_id) is None:
                    raise AssertionError("variable-order extra source has no owner edge")
                for name in ("baseline", "reward_mean", "reward_variance", "surprise_evidence", "confidence", "negative_surprise", "total_reward", "detector_mean", "detector_variance", "detector_cusum"):
                    if not math.isfinite(float(getattr(module, name))):
                        raise AssertionError("motor module state must be finite: %s" % name)
                if len(module.detector_predictor) != expected_features or not all(math.isfinite(float(value)) for value in module.detector_predictor):
                    raise AssertionError("motor detector predictor state is invalid")
                if module.reward_variance <= 0.0 or module.detector_variance <= 0.0 or module.surprise_evidence < 0.0 or module.detector_cusum < 0.0 or not 0.0 <= module.confidence <= 1.0 or module.negative_surprise < 0.0:
                    raise AssertionError("motor module detector state is invalid")
                if module.reward_count < 0 or module.probe_count < 0 or module.negative_streak < 0 or module.detector_count < 0:
                    raise AssertionError("motor module counters must be nonnegative")
                if module.dormant and identifier == self.active_motor_module and self.context_probe_module is None:
                    raise AssertionError("dormant module cannot be active outside a probe")

    def state_dict(self) -> Dict[str, object]:
        return {
            "version": self.VERSION,
            "seed": self.seed,
            "step_count": self.step_count,
            "input_ids": list(self.input_ids),
            "output_ids": list(self.output_ids),
            "learning_rate": self.learning_rate,
            "trace_decay": self.trace_decay,
            "homeostasis_rate": self.homeostasis_rate,
            "actor_learning_rate": self.actor_learning_rate,
            "legacy_learning_enabled": self.legacy_learning_enabled,
            "structural_plasticity_enabled": self.structural_plasticity_enabled,
            "reward_baseline": self.reward_baseline,
            "reward_baseline_rate": self.reward_baseline_rate,
            "pending_outcome": self._pending_outcome,
            "structural_interval": self.structural_interval,
            "cells": {identifier: asdict(cell) for identifier, cell in sorted(self.cells.items())},
            "synapses": [synapse_to_dict(synapse) for synapse in self.graph.iter_synapses()],
            "next_synapse_index": self.graph.next_synapse_index,
            "retired_synapse_ids": sorted(self.graph.retired_ids),
            "resources": self.resources.to_dict(),
            "events": list(self.events),
            "coactivity": {"%s|%s" % key: value for key, value in sorted(self._coactivity.items())},
            "metrics_history": list(self.metrics_history),
            "rng_state": self.rng.getstate(),
            "exploration_rng_state": self.exploration_rng.getstate(),
            "exploration_tape": list(self.exploration_tape) if self.exploration_tape is not None else None,
            "exploration_cursor": self.exploration_cursor,
            "representation_learning_enabled": self.representation_learning_enabled,
            "representation_learning_rate": self.representation_learning_rate,
            "representation_noise": self.representation_noise,
            "representation_rng_state": self.representation_rng.getstate(),
            "representation_tape": list(self.representation_tape) if self.representation_tape is not None else None,
            "representation_cursor": self.representation_cursor,
            "variable_order_enabled": self.variable_order_enabled,
            "max_dendritic_order": self.max_dendritic_order,
            "variable_order_feature_owners": dict(sorted(self.variable_order_feature_owners.items())),
            "variable_order_extra_sources": {
                key: list(value) for key, value in sorted(self.variable_order_extra_sources.items())
            },
            "variable_order_learning_enabled": self.variable_order_learning_enabled,
            "variable_order_representation_seed": self.variable_order_representation_seed,
            "variable_order_rng_state": self.variable_order_rng.getstate(),
            "variable_order_feature_order": [list(feature) for feature in self.variable_order_feature_order],
            "variable_order_module_features": dict(sorted(self.variable_order_module_features.items())),
            "variable_order_fingerprint_window": self.variable_order_fingerprint_window,
            "variable_order_fingerprint_active": self.variable_order_fingerprint_active,
            "variable_order_fingerprint_owner": self.variable_order_fingerprint_owner,
            "variable_order_fingerprint_incumbent": self.variable_order_fingerprint_incumbent,
            "variable_order_fingerprint_start_step": self.variable_order_fingerprint_start_step,
            "variable_order_fingerprint_deadline": self.variable_order_fingerprint_deadline,
            "variable_order_fingerprint_count": self.variable_order_fingerprint_count,
            "variable_order_fingerprint_evidence": {
                key: dict(value) for key, value in sorted(self.variable_order_fingerprint_evidence.items())
            },
            "variable_order_fingerprint_pending": self.variable_order_fingerprint_pending,
            "variable_order_midpoint_existing_owner_enabled": self.variable_order_midpoint_existing_owner_enabled,
            "variable_order_midpoint_z_threshold": self.variable_order_midpoint_z_threshold,
            "variable_order_midpoint_separation": self.variable_order_midpoint_separation,
            "variable_order_owner_evidence_enabled": self.variable_order_owner_evidence_enabled,
            "variable_order_owner_posterior": dict(sorted(self.variable_order_owner_posterior.items())),
            "variable_order_owner_evidence_count": self.variable_order_owner_evidence_count,
            "variable_order_midpoint_novelty_probability": self.variable_order_midpoint_novelty_probability,
            "variable_order_temporal_refresh_used": self.variable_order_temporal_refresh_used,
            "general_structural_learning_enabled": self.general_structural_learning_enabled,
            "general_structural_max_depth": self.general_structural_max_depth,
            "general_structural_max_leaves": self.general_structural_max_leaves,
            "general_feature_owners": dict(sorted(self.general_feature_owners.items())),
            "general_module_features": dict(sorted(self.general_module_features.items())),
            "general_feature_lineage": copy.deepcopy(self.general_feature_lineage),
            "general_feature_reuse": dict(sorted(self.general_feature_reuse.items())),
            "general_feature_last_used_step": dict(sorted(self.general_feature_last_used_step.items())),
            "general_feature_prune_count": self.general_feature_prune_count,
            "general_prune_reuse_ceiling": self.general_prune_reuse_ceiling,
            "general_fingerprint_evidence": copy.deepcopy(self.general_fingerprint_evidence),
            "general_novelty_candidate": self.general_novelty_candidate,
            "general_novelty_confirmations": self.general_novelty_confirmations,
            "general_required_confirmations": self.general_required_confirmations,
            "general_audit_interval": self.general_audit_interval,
            "general_last_audit_step": self.general_last_audit_step,
            "delayed_credit_enabled": self.delayed_credit_enabled,
            "delayed_credit_delay": self.delayed_credit_delay,
            "delayed_credit_queue": copy.deepcopy(self.delayed_credit_queue),
            "variable_order_owner_probe_enabled": self.variable_order_owner_probe_enabled,
            "variable_order_owner_midpoint_probe_enabled": self.variable_order_owner_midpoint_probe_enabled,
            "variable_order_owner_midpoint_probe_min_probability": self.variable_order_owner_midpoint_probe_min_probability,
            "variable_order_owner_midpoint_probe_min_margin": self.variable_order_owner_midpoint_probe_min_margin,
            "variable_order_normal_window": self.variable_order_normal_window,
            "variable_order_normal_evidence": {
                key: dict(value) for key, value in sorted(self.variable_order_normal_evidence.items())
            },
            "variable_order_normal_pending": self.variable_order_normal_pending,
            "variable_order_normal_count": self.variable_order_normal_count,
            "variable_order_detector_hold_until": self.variable_order_detector_hold_until,
            "variable_order_install_count": self.variable_order_install_count,
            "variable_order_route_count": self.variable_order_route_count,
            "compositional_substrate_enabled": self.compositional_substrate_enabled,
            "max_composed_leaves": self.max_composed_leaves,
            "composition_representation_seed": self.composition_representation_seed,
            "composition_rng_state": self.composition_rng.getstate(),
            "composition_feature_owners": dict(sorted(self.composition_feature_owners.items())),
            "composition_module_features": dict(sorted(self.composition_module_features.items())),
            "composition_feature_lineage": {
                key: {
                    "source_cells": list(value.get("source_cells", ())),
                    "leaf_closure": [list(leaf) for leaf in value.get("leaf_closure", ())],
                }
                for key, value in sorted(self.composition_feature_lineage.items())
            },
            "composition_install_count": self.composition_install_count,
            "composition_signal_gain": self.composition_signal_gain,
            "composition_learning_enabled": self.composition_learning_enabled,
            "composition_direct_route_pending": self.composition_direct_route_pending,
            "composition_verification_suppressed_until": self.composition_verification_suppressed_until,
            "composition_fingerprint_evidence": {
                key: dict(value) for key, value in sorted(self.composition_fingerprint_evidence.items())
            },
            "composition_fingerprint_pending": self.composition_fingerprint_pending,
            "composition_route_count": self.composition_route_count,
            "composition_candidate_count": self.composition_candidate_count,
            "adaptive_dendritic_enabled": self.adaptive_dendritic_enabled,
            "adaptive_dendritic_proposal_interval": self.adaptive_dendritic_proposal_interval,
            "adaptive_dendritic_proposal": self.adaptive_dendritic_proposal,
            "adaptive_dendritic_accept_count": self.adaptive_dendritic_accept_count,
            "adaptive_dendritic_reject_count": self.adaptive_dendritic_reject_count,
            "adaptive_dendritic_min_evidence": self.adaptive_dendritic_min_evidence,
            "adaptive_dendritic_max_trial_steps": self.adaptive_dendritic_max_trial_steps,
            "adaptive_dendritic_confidence_z": self.adaptive_dendritic_confidence_z,
            "adaptive_dendritic_pairs_tried": {
                module_id: [list(pair) for pair in pairs]
                for module_id, pairs in sorted(self.adaptive_dendritic_pairs_tried.items())
            },
            "adaptive_dendritic_module_accepted": dict(sorted(self.adaptive_dendritic_module_accepted.items())),
            "adaptive_dendritic_shadow_enabled": self.adaptive_dendritic_shadow_enabled,
            "adaptive_dendritic_shadow_evidence": {
                module_id: {key: dict(value) for key, value in sorted(entries.items())}
                for module_id, entries in sorted(self.adaptive_dendritic_shadow_evidence.items())
            },
            "adaptive_dendritic_shadow_search_exhausted": dict(sorted(self.adaptive_dendritic_shadow_search_exhausted.items())),
            "adaptive_dendritic_shadow_pending": self.adaptive_dendritic_shadow_pending,
            "adaptive_dendritic_shadow_action_count": self.adaptive_dendritic_shadow_action_count,
            "adaptive_dendritic_shadow_install_count": self.adaptive_dendritic_shadow_install_count,
            "adaptive_dendritic_shadow_checkpoints": list(self.adaptive_dendritic_shadow_checkpoints),
            "adaptive_dendritic_shadow_confidence_z": self.adaptive_dendritic_shadow_confidence_z,
            "adaptive_dendritic_shadow_separation": self.adaptive_dendritic_shadow_separation,
            "adaptive_dendritic_fingerprint_enabled": self.adaptive_dendritic_fingerprint_enabled,
            "adaptive_dendritic_fingerprint_window": self.adaptive_dendritic_fingerprint_window,
            "adaptive_dendritic_fingerprint_active": self.adaptive_dendritic_fingerprint_active,
            "adaptive_dendritic_fingerprint_owner": self.adaptive_dendritic_fingerprint_owner,
            "adaptive_dendritic_fingerprint_incumbent": self.adaptive_dendritic_fingerprint_incumbent,
            "adaptive_dendritic_fingerprint_start_step": self.adaptive_dendritic_fingerprint_start_step,
            "adaptive_dendritic_fingerprint_deadline": self.adaptive_dendritic_fingerprint_deadline,
            "adaptive_dendritic_fingerprint_count": self.adaptive_dendritic_fingerprint_count,
            "adaptive_dendritic_fingerprint_evidence": {
                key: dict(value) for key, value in sorted(self.adaptive_dendritic_fingerprint_evidence.items())
            },
            "adaptive_dendritic_fingerprint_pending": self.adaptive_dendritic_fingerprint_pending,
            "adaptive_dendritic_pair_owners": dict(sorted(self.adaptive_dendritic_pair_owners.items())),
            "adaptive_dendritic_detector_hold_until": self.adaptive_dendritic_detector_hold_until,
            "motor_bootstrap_scale": self.motor_bootstrap_scale,
            "context_enabled": self.context_enabled,
            "context_max_modules": self.context_max_modules,
            "context_afferent_kind": self.context_afferent_kind,
            "motor_modules": {identifier: asdict(module) for identifier, module in sorted(self.motor_modules.items())},
            "active_motor_module": self.active_motor_module,
            "pending_motor_module": self.pending_motor_module,
            "context_pending_mode": self.context_pending_mode,
            "context_pending_prediction": self.context_pending_prediction,
            "context_pending_scale": self.context_pending_scale,
            "context_pending_exploration_value": self.context_pending_exploration_value,
            "context_pending_exploration_sigma": self.context_pending_exploration_sigma,
            "context_state": self.context_state,
            "context_warning_threshold": self.context_warning_threshold,
            "context_safe_action": self.context_safe_action,
            "context_probe_module": self.context_probe_module,
            "context_probe_queue": list(self.context_probe_queue),
            "context_probe_failed_reference": self.context_probe_failed_reference,
            "context_probe_remaining": self.context_probe_remaining,
            "context_probe_reference": self.context_probe_reference,
            "context_last_reward_value": self.context_last_reward_value,
            "context_probe_rewards": list(self.context_probe_rewards),
            "context_probe_sum": self.context_probe_sum,
            "context_probe_sumsq": self.context_probe_sumsq,
            "context_probe_count": self.context_probe_count,
            "context_probe_steps_seen": self.context_probe_steps_seen,
            "context_probe_informative_count": self.context_probe_informative_count,
            "context_probe_candidate_mean": self.context_probe_candidate_mean,
            "context_probe_incumbent_mean": self.context_probe_incumbent_mean,
            "context_probe_incumbent_module": self.context_probe_incumbent_module,
            "context_probe_origin": self.context_probe_origin,
            "context_probe_exploration": self.context_probe_exploration,
            "context_pending_features": list(self.context_pending_features),
            "context_pending_action": self.context_pending_action,
            "context_switch_threshold": self.context_switch_threshold,
            "context_confidence_threshold": self.context_confidence_threshold,
            "context_negative_streak": self.context_negative_streak,
            "context_probe_steps": self.context_probe_steps,
            "context_probe_min_steps": self.context_probe_min_steps,
            "context_probe_max_steps": self.context_probe_max_steps,
            "context_reactivation_margin": self.context_reactivation_margin,
            "context_probe_reward_floor": self.context_probe_reward_floor,
            "context_surprise_mean": self.context_surprise_mean,
            "context_surprise_variance": self.context_surprise_variance,
            "context_surprise_count": self.context_surprise_count,
            "context_surprise_leak": self.context_surprise_leak,
            "context_surprise_drift": self.context_surprise_drift,
            "context_detector_min_evidence": self.context_detector_min_evidence,
            "context_detector_freeze_threshold": self.context_detector_freeze_threshold,
            "context_detector_drift": self.context_detector_drift,
            "context_detector_recovery": self.context_detector_recovery,
            "context_detector_predictor_rate": self.context_detector_predictor_rate,
            "context_probe_information_threshold": self.context_probe_information_threshold,
            "evidence_router_mode": self.evidence_router_mode,
            "evidence_novelty_mass": self.evidence_novelty_mass,
            "evidence_min_evidence": self.evidence_min_evidence,
            "evidence_enter_margin": self.evidence_enter_margin,
            "evidence_exit_margin": self.evidence_exit_margin,
            "circuit_evidence": {key: dict(value) for key, value in sorted(self.circuit_evidence.items())},
            "evidence_posterior": dict(sorted(self.evidence_posterior.items())),
            "evidence_shadow_steps": self.evidence_shadow_steps,
            "evidence_decisions_enabled": self.evidence_decisions_enabled,
            "evidence_novelty_fit_threshold": self.evidence_novelty_fit_threshold,
            "evidence_recent_best_ll": self.evidence_recent_best_ll,
            "evidence_recent_count": self.evidence_recent_count,
            "evidence_recent_decay": self.evidence_recent_decay,
            "evidence_pending_growth": copy.deepcopy(self.evidence_pending_growth),
            "sequence_memory": None if self.sequence_memory is None else self.sequence_memory.state_dict(),
            "sequence_memory_symbols": None if self.sequence_memory_symbols is None else list(self.sequence_memory_symbols),
        }

    def save(self, path: str) -> None:
        _atomic_json_dump(path, self.state_dict())

    def save_checkpoint(self, path: str, environment) -> None:
        """Save organism and environment state at a decision boundary."""
        _atomic_json_dump(path, {
            "checkpoint_version": 1,
            "organism": self.state_dict(),
            "environment": environment.state_dict(),
            "environment_type": "%s.%s" % (environment.__class__.__module__, environment.__class__.__name__),
        })

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "Organism":
        if not isinstance(state, Mapping):
            raise ValueError("organism checkpoint must be a mapping")
        version = int(state.get("version", 0))
        if version not in (5, 6, 7, 8, 9, 10, 11, cls.VERSION):
            raise ValueError("unsupported checkpoint version")
        cells = {}
        for identifier, payload in state["cells"].items():
            cell_payload = dict(payload)
            # v5 had no signed EMA.  Start its new centering state neutral;
            # all legacy weights, thresholds, and execution state survive.
            cell_payload.setdefault("signed_activity_ema", 0.0)
            # v5-v7 cells are additive rate cells.  Product-cell metadata is
            # intentionally absent from those checkpoints and migrates to the
            # exact legacy behavior.
            cell_payload.setdefault("activation_type", "additive")
            cell_payload.setdefault("dendritic_sources", ())
            if isinstance(cell_payload.get("dendritic_sources"), list):
                cell_payload["dendritic_sources"] = tuple(cell_payload["dendritic_sources"])
            cells[identifier] = Cell(**cell_payload)
        graph = SparseDirectedGraph()
        for payload in state["synapses"]:
            graph.add(Synapse(**payload))
        graph.next_synapse_index = max(graph.next_synapse_index, int(state.get("next_synapse_index", graph.next_synapse_index)))
        graph.retired_ids = set(state.get("retired_synapse_ids", []))
        resource_payload = dict(state["resources"])
        persisted_counters = dict(resource_payload.get("counters", {}))
        if persisted_counters.get("cells") != len(cells) or persisted_counters.get("synapses") != len(graph.synapses):
            raise ValueError("checkpoint resource counters do not match structure")
        resources = ResourceBudget(**resource_payload)
        organism = cls(cells, graph, state["input_ids"], state["output_ids"], int(state["seed"]), resources, int(state["structural_interval"]))
        organism.step_count = int(state["step_count"])
        organism.learning_rate = float(state["learning_rate"])
        organism.trace_decay = float(state["trace_decay"])
        organism.homeostasis_rate = float(state["homeostasis_rate"])
        organism.actor_learning_rate = float(state.get("actor_learning_rate", 0.01))
        organism.legacy_learning_enabled = bool(state.get("legacy_learning_enabled", True))
        organism.structural_plasticity_enabled = bool(state.get("structural_plasticity_enabled", True))
        organism.reward_baseline = float(state.get("reward_baseline", 0.0))
        organism.reward_baseline_rate = float(state.get("reward_baseline_rate", 0.03))
        organism._pending_outcome = bool(state.get("pending_outcome", False))
        organism.events = list(state["events"])
        organism.metrics_history = list(state.get("metrics_history", []))
        organism._coactivity = {}
        for key, value in state.get("coactivity", {}).items():
            source, destination = key.split("|", 1)
            organism._coactivity[(source, destination)] = float(value)
        rng_state = state.get("rng_state")
        if rng_state is not None:
            organism.rng.setstate(_nested_tuple(rng_state))
        exploration_rng_state = state.get("exploration_rng_state")
        if exploration_rng_state is not None:
            organism.exploration_rng.setstate(_nested_tuple(exploration_rng_state))
        tape = state.get("exploration_tape")
        if tape is not None:
            organism.exploration_tape = tuple(float(value) for value in tape)
            organism.exploration_cursor = int(state.get("exploration_cursor", 0))
        organism.representation_learning_enabled = bool(state.get("representation_learning_enabled", False))
        organism.representation_learning_rate = float(state.get("representation_learning_rate", 0.0))
        organism.representation_noise = float(state.get("representation_noise", 0.0))
        representation_rng_state = state.get("representation_rng_state")
        if representation_rng_state is not None:
            organism.representation_rng.setstate(_nested_tuple(representation_rng_state))
        representation_tape = state.get("representation_tape")
        if representation_tape is not None:
            organism.representation_tape = tuple(float(value) for value in representation_tape)
            organism.representation_cursor = int(state.get("representation_cursor", 0))
        # v5-v8 checkpoints have no variable-order substrate.  Keep binary
        # product cells and every legacy ID/RNG/resource field untouched.
        organism.variable_order_enabled = bool(state.get("variable_order_enabled", False))
        organism.max_dendritic_order = int(state.get("max_dendritic_order", 2))
        organism.variable_order_feature_owners = {
            str(key): str(value) for key, value in dict(state.get("variable_order_feature_owners", {})).items()
        }
        organism.variable_order_extra_sources = {
            str(key): [str(cell_id) for cell_id in value]
            for key, value in dict(state.get("variable_order_extra_sources", {})).items()
        }
        organism.variable_order_learning_enabled = bool(state.get("variable_order_learning_enabled", False))
        representation_seed = state.get("variable_order_representation_seed")
        organism.variable_order_representation_seed = int(representation_seed) if representation_seed is not None else None
        variable_order_rng_state = state.get("variable_order_rng_state")
        if variable_order_rng_state is not None:
            organism.variable_order_rng.setstate(_nested_tuple(variable_order_rng_state))
        organism.variable_order_feature_order = tuple(
            tuple(str(source) for source in feature)
            for feature in state.get("variable_order_feature_order", ())
        )
        organism.variable_order_module_features = {
            str(module_id): str(key) for module_id, key in dict(state.get("variable_order_module_features", {})).items()
        }
        organism.variable_order_fingerprint_window = int(state.get("variable_order_fingerprint_window", 16))
        organism.variable_order_fingerprint_active = bool(state.get("variable_order_fingerprint_active", False))
        organism.variable_order_fingerprint_owner = state.get("variable_order_fingerprint_owner")
        organism.variable_order_fingerprint_incumbent = state.get("variable_order_fingerprint_incumbent")
        organism.variable_order_fingerprint_start_step = int(state.get("variable_order_fingerprint_start_step", 0))
        organism.variable_order_fingerprint_deadline = int(state.get("variable_order_fingerprint_deadline", 0))
        organism.variable_order_fingerprint_count = int(state.get("variable_order_fingerprint_count", 0))
        organism.variable_order_fingerprint_evidence = {
            str(key): {
                "sum": float(value.get("sum", 0.0)),
                "sumsq": float(value.get("sumsq", 0.0)),
                "count": float(value.get("count", 0.0)),
            }
            for key, value in dict(state.get("variable_order_fingerprint_evidence", {})).items()
        }
        organism.variable_order_fingerprint_pending = state.get("variable_order_fingerprint_pending")
        organism.variable_order_midpoint_existing_owner_enabled = bool(state.get("variable_order_midpoint_existing_owner_enabled", False))
        organism.variable_order_midpoint_z_threshold = float(state.get("variable_order_midpoint_z_threshold", 3.0))
        organism.variable_order_midpoint_separation = float(state.get("variable_order_midpoint_separation", 0.05))
        organism.variable_order_owner_evidence_enabled = bool(state.get("variable_order_owner_evidence_enabled", False))
        organism.variable_order_owner_posterior = {str(key): float(value) for key, value in dict(state.get("variable_order_owner_posterior", {})).items()}
        organism.variable_order_owner_evidence_count = int(state.get("variable_order_owner_evidence_count", 0))
        organism.variable_order_midpoint_novelty_probability = float(state.get("variable_order_midpoint_novelty_probability", 0.0))
        organism.variable_order_temporal_refresh_used = bool(state.get("variable_order_temporal_refresh_used", False))
        organism.general_structural_learning_enabled = bool(state.get("general_structural_learning_enabled", False))
        organism.general_structural_max_depth = int(state.get("general_structural_max_depth", 4))
        organism.general_structural_max_leaves = int(state.get("general_structural_max_leaves", 8))
        organism.general_feature_owners = {str(key): str(value) for key, value in dict(state.get("general_feature_owners", {})).items()}
        organism.general_module_features = {str(key): str(value) for key, value in dict(state.get("general_module_features", {})).items()}
        organism.general_feature_lineage = copy.deepcopy(dict(state.get("general_feature_lineage", {})))
        for lineage in organism.general_feature_lineage.values():
            old_cell = lineage.get("old_cell")
            if isinstance(old_cell, dict) and isinstance(old_cell.get("dendritic_sources"), list):
                old_cell["dendritic_sources"] = tuple(str(value) for value in old_cell["dendritic_sources"])
        organism.general_feature_reuse = {str(key): int(value) for key, value in dict(state.get("general_feature_reuse", {})).items()}
        organism.general_feature_last_used_step = {str(key): int(value) for key, value in dict(state.get("general_feature_last_used_step", {})).items()}
        for key in organism.general_feature_owners:
            organism.general_feature_last_used_step.setdefault(key, 0)
        organism.general_feature_prune_count = int(state.get("general_feature_prune_count", 0))
        organism.general_prune_reuse_ceiling = int(state.get("general_prune_reuse_ceiling", 0))
        organism.general_fingerprint_evidence = {
            str(key): {name: float(value.get(name, 0.0)) for name in ("sum", "sumsq", "count")}
            for key, value in dict(state.get("general_fingerprint_evidence", {})).items()
        }
        organism.general_novelty_candidate = state.get("general_novelty_candidate")
        organism.general_novelty_confirmations = int(state.get("general_novelty_confirmations", 0))
        organism.general_required_confirmations = int(state.get("general_required_confirmations", 2))
        organism.general_audit_interval = int(state.get("general_audit_interval", 128))
        organism.general_last_audit_step = int(state.get("general_last_audit_step", 0))
        organism.delayed_credit_enabled = bool(state.get("delayed_credit_enabled", False))
        organism.delayed_credit_delay = int(state.get("delayed_credit_delay", 0))
        organism.delayed_credit_queue = copy.deepcopy(list(state.get("delayed_credit_queue", [])))
        organism.variable_order_owner_probe_enabled = bool(state.get("variable_order_owner_probe_enabled", False))
        organism.variable_order_owner_midpoint_probe_enabled = bool(state.get("variable_order_owner_midpoint_probe_enabled", False))
        organism.variable_order_owner_midpoint_probe_min_probability = float(state.get("variable_order_owner_midpoint_probe_min_probability", 0.0))
        organism.variable_order_owner_midpoint_probe_min_margin = float(state.get("variable_order_owner_midpoint_probe_min_margin", 0.0))
        organism.variable_order_normal_window = int(state.get("variable_order_normal_window", 8))
        organism.variable_order_normal_evidence = {
            str(key): {
                "sum": float(value.get("sum", 0.0)),
                "sumsq": float(value.get("sumsq", 0.0)),
                "count": float(value.get("count", 0.0)),
            }
            for key, value in dict(state.get("variable_order_normal_evidence", {})).items()
        }
        organism.variable_order_normal_pending = state.get("variable_order_normal_pending")
        organism.variable_order_normal_count = int(state.get("variable_order_normal_count", 0))
        organism.variable_order_detector_hold_until = int(state.get("variable_order_detector_hold_until", 0))
        organism.variable_order_install_count = int(state.get("variable_order_install_count", 0))
        organism.variable_order_route_count = int(state.get("variable_order_route_count", 0))
        organism.compositional_substrate_enabled = bool(state.get("compositional_substrate_enabled", False))
        organism.max_composed_leaves = int(state.get("max_composed_leaves", 4))
        composition_seed = state.get("composition_representation_seed")
        organism.composition_representation_seed = int(composition_seed) if composition_seed is not None else None
        composition_rng_state = state.get("composition_rng_state")
        if composition_rng_state is not None:
            organism.composition_rng.setstate(_nested_tuple(composition_rng_state))
        organism.composition_feature_owners = {
            str(key): str(value) for key, value in dict(state.get("composition_feature_owners", {})).items()
        }
        organism.composition_module_features = {
            str(key): str(value) for key, value in dict(state.get("composition_module_features", {})).items()
        }
        organism.composition_feature_lineage = {}
        for key, payload in dict(state.get("composition_feature_lineage", {})).items():
            lineage = dict(payload)
            organism.composition_feature_lineage[str(key)] = {
                "source_cells": [str(value) for value in lineage.get("source_cells", ())],
                "leaf_closure": [[str(value) for value in leaf] for leaf in lineage.get("leaf_closure", ())],
            }
        organism.composition_install_count = int(state.get("composition_install_count", 0))
        organism.composition_signal_gain = float(state.get("composition_signal_gain", 1.0))
        organism.composition_learning_enabled = bool(state.get("composition_learning_enabled", False))
        organism.composition_direct_route_pending = bool(state.get("composition_direct_route_pending", False))
        organism.composition_verification_suppressed_until = int(state.get("composition_verification_suppressed_until", 0))
        organism.composition_fingerprint_evidence = {
            str(key): {
                "sum": float(value.get("sum", 0.0)),
                "sumsq": float(value.get("sumsq", 0.0)),
                "count": float(value.get("count", 0.0)),
            }
            for key, value in dict(state.get("composition_fingerprint_evidence", {})).items()
        }
        pending_composition = state.get("composition_fingerprint_pending")
        organism.composition_fingerprint_pending = None if pending_composition is None else dict(pending_composition)
        organism.composition_route_count = int(state.get("composition_route_count", 0))
        organism.composition_candidate_count = int(state.get("composition_candidate_count", 0))
        organism.adaptive_dendritic_enabled = bool(state.get("adaptive_dendritic_enabled", False))
        organism.adaptive_dendritic_proposal_interval = int(state.get("adaptive_dendritic_proposal_interval", 8))
        organism.adaptive_dendritic_proposal = state.get("adaptive_dendritic_proposal")
        if organism.adaptive_dendritic_proposal is not None:
            proposal = dict(organism.adaptive_dendritic_proposal)
            old_cell = dict(proposal.get("old_cell", {}))
            if isinstance(old_cell.get("dendritic_sources"), list):
                old_cell["dendritic_sources"] = tuple(old_cell["dendritic_sources"])
            proposal["old_cell"] = old_cell
            organism.adaptive_dendritic_proposal = proposal
        organism.adaptive_dendritic_accept_count = int(state.get("adaptive_dendritic_accept_count", 0))
        organism.adaptive_dendritic_reject_count = int(state.get("adaptive_dendritic_reject_count", 0))
        organism.adaptive_dendritic_min_evidence = int(state.get("adaptive_dendritic_min_evidence", 3))
        organism.adaptive_dendritic_max_trial_steps = int(state.get("adaptive_dendritic_max_trial_steps", 8))
        organism.adaptive_dendritic_confidence_z = float(state.get("adaptive_dendritic_confidence_z", 1.96))
        organism.adaptive_dendritic_pairs_tried = {
            str(module_id): [tuple(str(item) for item in pair) for pair in pairs]
            for module_id, pairs in dict(state.get("adaptive_dendritic_pairs_tried", {})).items()
        }
        organism.adaptive_dendritic_module_accepted = {
            str(module_id): bool(value)
            for module_id, value in dict(state.get("adaptive_dendritic_module_accepted", {})).items()
        }
        organism.adaptive_dendritic_shadow_enabled = bool(state.get("adaptive_dendritic_shadow_enabled", False))
        organism.adaptive_dendritic_shadow_evidence = {
            str(module_id): {
                str(key): {
                    "sum": float(value.get("sum", 0.0)),
                    "sumsq": float(value.get("sumsq", 0.0)),
                    "count": float(value.get("count", 0.0)),
                }
                for key, value in dict(entries).items()
            }
            for module_id, entries in dict(state.get("adaptive_dendritic_shadow_evidence", {})).items()
        }
        organism.adaptive_dendritic_shadow_search_exhausted = {
            str(module_id): bool(value)
            for module_id, value in dict(state.get("adaptive_dendritic_shadow_search_exhausted", {})).items()
        }
        organism.adaptive_dendritic_shadow_pending = state.get("adaptive_dendritic_shadow_pending")
        organism.adaptive_dendritic_shadow_action_count = int(state.get("adaptive_dendritic_shadow_action_count", 0))
        organism.adaptive_dendritic_shadow_install_count = int(state.get("adaptive_dendritic_shadow_install_count", 0))
        organism.adaptive_dendritic_shadow_checkpoints = tuple(int(value) for value in state.get("adaptive_dendritic_shadow_checkpoints", (16, 32, 64)))
        organism.adaptive_dendritic_shadow_confidence_z = float(state.get("adaptive_dendritic_shadow_confidence_z", 1.96))
        organism.adaptive_dendritic_shadow_separation = float(state.get("adaptive_dendritic_shadow_separation", 0.05))
        organism.adaptive_dendritic_fingerprint_enabled = bool(state.get("adaptive_dendritic_fingerprint_enabled", False))
        organism.adaptive_dendritic_fingerprint_window = int(state.get("adaptive_dendritic_fingerprint_window", 16))
        organism.adaptive_dendritic_fingerprint_active = bool(state.get("adaptive_dendritic_fingerprint_active", False))
        organism.adaptive_dendritic_fingerprint_owner = state.get("adaptive_dendritic_fingerprint_owner")
        organism.adaptive_dendritic_fingerprint_incumbent = state.get("adaptive_dendritic_fingerprint_incumbent")
        organism.adaptive_dendritic_fingerprint_start_step = int(state.get("adaptive_dendritic_fingerprint_start_step", 0))
        organism.adaptive_dendritic_fingerprint_deadline = int(state.get("adaptive_dendritic_fingerprint_deadline", 0))
        organism.adaptive_dendritic_fingerprint_count = int(state.get("adaptive_dendritic_fingerprint_count", 0))
        organism.adaptive_dendritic_fingerprint_evidence = {
            str(key): {
                "sum": float(value.get("sum", 0.0)),
                "sumsq": float(value.get("sumsq", 0.0)),
                "count": float(value.get("count", 0.0)),
            }
            for key, value in dict(state.get("adaptive_dendritic_fingerprint_evidence", {})).items()
        }
        organism.adaptive_dendritic_fingerprint_pending = state.get("adaptive_dendritic_fingerprint_pending")
        organism.adaptive_dendritic_pair_owners = {
            str(key): str(value) for key, value in dict(state.get("adaptive_dendritic_pair_owners", {})).items()
        }
        organism.adaptive_dendritic_detector_hold_until = int(state.get("adaptive_dendritic_detector_hold_until", 0))
        organism.motor_bootstrap_scale = float(state.get("motor_bootstrap_scale", 0.0))
        organism.context_enabled = bool(state.get("context_enabled", False))
        organism.context_max_modules = int(state.get("context_max_modules", 4))
        organism.context_afferent_kind = str(state.get("context_afferent_kind", "input"))
        organism.motor_modules = {}
        expected_features = organism._context_feature_dimension()
        for identifier, payload in state.get("motor_modules", {}).items():
            module_payload = dict(payload)
            module_payload.setdefault("afferent_kind", "input")
            predictor = list(module_payload.get("detector_predictor", []))
            if len(predictor) != expected_features:
                # Migrate the scalar-cue v5 detector into the generic sensor
                # feature layout without changing its learned coefficients.
                expanded = [0.0] * expected_features
                if len(predictor) == 6 and expected_features >= 6:
                    expanded[0] = predictor[0]
                    expanded[1] = predictor[1]
                    expanded[1 + len(organism.input_ids)] = predictor[2]
                    expanded[2 + len(organism.input_ids)] = predictor[3]
                    expanded[2 + 2 * len(organism.input_ids)] = predictor[4]
                    expanded[-1] = predictor[5]
                else:
                    expanded[:min(len(predictor), expected_features)] = predictor[:expected_features]
                module_payload["detector_predictor"] = expanded
            organism.motor_modules[identifier] = MotorModule(**module_payload)
        organism.active_motor_module = state.get("active_motor_module")
        organism.pending_motor_module = state.get("pending_motor_module")
        organism.context_probe_module = state.get("context_probe_module")
        organism.context_probe_queue = list(state.get("context_probe_queue", []))
        organism.context_probe_failed_reference = float(state.get("context_probe_failed_reference", 0.0))
        organism.context_probe_remaining = int(state.get("context_probe_remaining", 0))
        organism.context_probe_reference = float(state.get("context_probe_reference", 0.0))
        organism.context_last_reward_value = float(state.get("context_last_reward_value", 0.0))
        organism.context_probe_rewards = [float(value) for value in state.get("context_probe_rewards", [])]
        organism.context_probe_sum = float(state.get("context_probe_sum", 0.0))
        organism.context_probe_sumsq = float(state.get("context_probe_sumsq", 0.0))
        organism.context_probe_count = int(state.get("context_probe_count", len(organism.context_probe_rewards)))
        organism.context_probe_steps_seen = int(state.get("context_probe_steps_seen", organism.context_probe_count))
        organism.context_probe_informative_count = int(state.get("context_probe_informative_count", organism.context_probe_count))
        organism.context_probe_candidate_mean = float(state.get("context_probe_candidate_mean", 0.0))
        organism.context_probe_incumbent_mean = float(state.get("context_probe_incumbent_mean", 0.0))
        organism.context_probe_incumbent_module = state.get("context_probe_incumbent_module")
        origin = state.get("context_probe_origin")
        # Older v5 checkpoints could already contain an active probe before
        # probe origin was serialized; treat those in-flight searches as the
        # conservative alarm path while preserving new warning checkpoints.
        if origin is None and organism.context_probe_module is not None:
            origin = "alarm"
        organism.context_probe_origin = str(origin) if origin is not None else None
        organism.context_probe_exploration = float(state.get("context_probe_exploration", 0.0))
        organism.context_pending_features = tuple(float(value) for value in state.get("context_pending_features", ()))
        if len(organism.context_pending_features) == 6 and expected_features != 6:
            old = organism.context_pending_features
            inputs = [0.0] * len(organism.input_ids)
            inputs[0] = old[1]
            action = old[2]
            organism.context_pending_features = (1.0,) + tuple(inputs) + (action,) + tuple(value * action for value in inputs) + tuple(value * value for value in inputs) + (action * action,)
        organism.context_pending_action = float(state.get("context_pending_action", 0.0))
        organism.context_pending_mode = str(state.get("context_pending_mode", "normal"))
        organism.context_pending_prediction = float(state.get("context_pending_prediction", 0.0))
        organism.context_pending_scale = float(state.get("context_pending_scale", 0.1))
        organism.context_pending_exploration_value = float(state.get("context_pending_exploration_value", 0.0))
        organism.context_pending_exploration_sigma = float(state.get("context_pending_exploration_sigma", 0.0))
        organism.context_state = str(state.get("context_state", "normal"))
        for name in ("context_switch_threshold", "context_warning_threshold", "context_safe_action", "context_confidence_threshold", "context_reactivation_margin", "context_probe_reward_floor", "context_surprise_mean", "context_surprise_variance", "context_surprise_leak", "context_surprise_drift", "context_detector_freeze_threshold", "context_detector_drift", "context_detector_recovery", "context_detector_predictor_rate", "context_probe_information_threshold", "context_probe_candidate_mean", "context_probe_incumbent_mean", "context_probe_exploration"):
            if name in state:
                setattr(organism, name, float(state[name]))
        for name in ("context_negative_streak", "context_probe_steps", "context_probe_min_steps", "context_probe_max_steps", "context_surprise_count", "context_detector_min_evidence"):
            if name in state:
                setattr(organism, name, int(state[name]))
        organism.evidence_router_mode = str(state.get("evidence_router_mode", "shadow"))
        if organism.evidence_router_mode not in ("threshold", "shadow"):
            raise ValueError("evidence router mode is invalid")
        organism.evidence_novelty_mass = float(state.get("evidence_novelty_mass", 0.05))
        organism.evidence_min_evidence = int(state.get("evidence_min_evidence", 16))
        organism.evidence_enter_margin = float(state.get("evidence_enter_margin", 0.15))
        organism.evidence_exit_margin = float(state.get("evidence_exit_margin", 0.05))
        organism.circuit_evidence = {str(key): dict(value) for key, value in dict(state.get("circuit_evidence", {})).items()}
        organism.evidence_posterior = {str(key): float(value) for key, value in dict(state.get("evidence_posterior", {})).items()}
        organism.evidence_shadow_steps = int(state.get("evidence_shadow_steps", 0))
        organism.evidence_decisions_enabled = bool(state.get("evidence_decisions_enabled", True))
        organism.evidence_novelty_fit_threshold = float(state.get("evidence_novelty_fit_threshold", -2.0))
        organism.evidence_recent_best_ll = float(state.get("evidence_recent_best_ll", 0.0))
        organism.evidence_recent_count = int(state.get("evidence_recent_count", 0))
        organism.evidence_recent_decay = float(state.get("evidence_recent_decay", 0.95))
        pending_growth = state.get("evidence_pending_growth")
        organism.evidence_pending_growth = None if pending_growth is None else dict(pending_growth)
        sequence_payload = state.get("sequence_memory")
        if sequence_payload is None:
            organism.sequence_memory = None
            organism.sequence_memory_symbols = None
        else:
            from .memory.sequence import SequenceCircuitMemory
            organism.sequence_memory = SequenceCircuitMemory.from_state_dict(sequence_payload)
            symbols = state.get("sequence_memory_symbols")
            organism.sequence_memory_symbols = tuple(symbols) if symbols is not None else tuple(organism.sequence_memory.symbols)
            if tuple(organism.sequence_memory.symbols) != tuple(organism.sequence_memory_symbols):
                raise ValueError("sequence memory symbols disagree with memory state")
        organism.validate()
        return organism

    @classmethod
    def load(cls, path: str) -> "Organism":
        with open(path, "r") as handle:
            return cls.from_state_dict(json.load(handle))

    @classmethod
    def load_checkpoint(cls, path: str):
        from .environment import ContinuousTargetEnvironment
        from .lifetime import AdaptiveDendriticEnvironment, ContinuousLifetimeEnvironment, FinalLifetimeEnvironment, NonlinearMultiContextEnvironment, UnannouncedLifetimeEnvironment, UnannouncedMultiContextEnvironment
        with open(path, "r") as handle:
            payload = json.load(handle)
        if not isinstance(payload, Mapping) or "organism" not in payload or "environment" not in payload:
            raise ValueError("compound checkpoint schema is invalid")
        checkpoint_version = int(payload.get("checkpoint_version", 0))
        if checkpoint_version not in (0, 1):
            raise ValueError("unsupported compound checkpoint version")
        environment_type = payload.get("environment_type", "soma.environment.ContinuousTargetEnvironment")
        environment_classes = {
            "soma.environment.ContinuousTargetEnvironment": ContinuousTargetEnvironment,
            "soma.lifetime.UnannouncedLifetimeEnvironment": UnannouncedLifetimeEnvironment,
            "soma.lifetime.UnannouncedMultiContextEnvironment": UnannouncedMultiContextEnvironment,
            "soma.lifetime.NonlinearMultiContextEnvironment": NonlinearMultiContextEnvironment,
            "soma.lifetime.AdaptiveDendriticEnvironment": AdaptiveDendriticEnvironment,
            "soma.lifetime.ContinuousLifetimeEnvironment": ContinuousLifetimeEnvironment,
            "soma.lifetime.FinalLifetimeEnvironment": FinalLifetimeEnvironment,
        }
        environment_class = environment_classes.get(environment_type)
        if environment_class is None:
            raise ValueError("unsupported checkpoint environment type: %s" % environment_type)
        return cls.from_state_dict(payload["organism"]), environment_class.from_state_dict(payload["environment"])


def _atomic_json_dump(path: str, payload: Mapping[str, object]) -> None:
    """Write a JSON checkpoint atomically beside its final destination."""
    destination = os.path.abspath(os.fspath(path))
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".soma-", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _nested_tuple(value: object) -> object:
    if isinstance(value, list):
        return tuple(_nested_tuple(item) for item in value)
    return value
