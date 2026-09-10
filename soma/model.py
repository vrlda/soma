"""Stable public lifecycle facade for a continuously learning SOMA model."""

from typing import Dict, Sequence

from .organism import Modulators, Organism, StepResult


class SOMA:
    """One persistent model that learns on every rewarded interaction.

    ``reward`` passed to :meth:`step` belongs to an earlier action.  With
    ``reward_delay > 0`` the model retains the exact causal action state and
    associates the arriving reward with it automatically.
    """

    PROFILE = "continuous-v1"

    def __init__(self, organism: Organism) -> None:
        if not isinstance(organism, Organism):
            raise TypeError("organism must be an Organism")
        self.organism = organism
        self.organism.validate()

    @classmethod
    def create(
        cls,
        input_size: int,
        seed: int = 0,
        hidden_size: int = 6,
        output_size: int = 1,
        reward_delay: int = 0,
        max_cells: int = 32,
        max_synapses: int = 256,
        max_modules: int = 4,
        energy_per_step: float = 20.0,
        prune_reuse_ceiling: int = 0,
    ) -> "SOMA":
        organism = Organism.create_default(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            seed=seed,
            max_cells=max_cells,
            max_synapses=max_synapses,
        )
        if energy_per_step <= 0.0:
            raise ValueError("energy_per_step must be positive")
        organism.resources.energy_per_step = float(energy_per_step)
        organism.enable_context_modules(max_modules=max_modules, afferent_kind="hidden")
        organism.enable_motor_bootstrap()
        organism.actor_learning_rate = 0.40
        organism.enable_variable_order_learning(seed + 7919, max_order=3)
        organism.set_variable_order_fingerprint_window(32)
        organism.enable_variable_order_owner_evidence()
        organism.enable_variable_order_owner_probes(True)
        organism.context_detector_drift = 0.25
        organism.enable_general_structural_learning(max_depth=4, max_leaves=8)
        organism.set_general_prune_reuse_ceiling(prune_reuse_ceiling)
        if reward_delay:
            organism.enable_delayed_credit(reward_delay)
        return cls(organism)

    def step(
        self,
        inputs: Sequence[float],
        reward: float = 0.0,
        novelty: float = 0.10,
        exploration: float = 0.20,
        uncertainty: float = 0.0,
        salience: float = 0.0,
    ) -> StepResult:
        return self.organism.step(inputs, Modulators(
            reward=reward,
            novelty=novelty,
            exploration=exploration,
            uncertainty=uncertainty,
            salience=salience,
        ))

    def apply_outcome(self, reward: float) -> None:
        self.organism.apply_outcome(reward)

    def inspect(self, include_events: int = 10) -> Dict[str, object]:
        if isinstance(include_events, bool) or not isinstance(include_events, int) or include_events < 0:
            raise ValueError("include_events must be a nonnegative integer")
        resources = self.organism.resources
        modules = self.organism.motor_modules
        return {
            "profile": self.PROFILE,
            "checkpoint_version": self.organism.VERSION,
            "seed": self.organism.seed,
            "steps": self.organism.step_count,
            "pending_outcome": self.organism._pending_outcome,
            "delayed_credit": {
                "enabled": self.organism.delayed_credit_enabled,
                "delay": self.organism.delayed_credit_delay,
                "queued": len(self.organism.delayed_credit_queue),
            },
            "topology": {
                "cells": len(self.organism.cells),
                "synapses": len(self.organism.graph.synapses),
                "motor_modules": len(modules),
                "active_motor_module": self.organism.active_motor_module,
                "general_features": len(self.organism.general_feature_owners),
            },
            "learning": {
                "reward_baseline": self.organism.reward_baseline,
                "variable_order_installs": self.organism.variable_order_install_count,
                "variable_order_routes": self.organism.variable_order_route_count,
                "general_feature_reuse": dict(sorted(self.organism.general_feature_reuse.items())),
                "general_feature_prunes": self.organism.general_feature_prune_count,
            },
            "resources": {
                "energy_total": resources.total_energy,
                "max_cells": resources.max_cells,
                "max_synapses": resources.max_synapses,
                "counters": dict(resources.counters),
            },
            "recent_events": list(self.organism.events[-include_events:]) if include_events else [],
        }

    def save(self, path: str) -> None:
        self.organism.save(path)

    @classmethod
    def load(cls, path: str) -> "SOMA":
        return cls(Organism.load(path))
