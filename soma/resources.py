"""Explicit resource accounting for bounded lifetime adaptation."""

from dataclasses import dataclass, field
import math
from typing import Dict


@dataclass
class ResourceBudget:
    max_cells: int = 64
    max_synapses: int = 256
    energy_per_step: float = 20.0
    cell_cost: float = 1.0
    synapse_cost: float = 0.02
    energy_used: float = 0.0
    total_energy: float = 0.0
    structural_events: int = 0
    counters: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("max_cells", "max_synapses"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError("%s must be a nonnegative integer" % name)
        for name in ("energy_per_step", "cell_cost", "synapse_cost", "energy_used", "total_energy"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("%s must be finite and nonnegative" % name)

    def baseline_cost(self, cells: int, synapses: int) -> float:
        return self.cell_cost * cells + self.synapse_cost * synapses

    def begin_step(self, cells: int, synapses: int) -> None:
        self.energy_used = self.cell_cost * cells + self.synapse_cost * synapses
        if self.energy_used > self.energy_per_step + 1e-12:
            raise RuntimeError("organism baseline energy %.6f exceeds per-step budget %.6f" % (self.energy_used, self.energy_per_step))
        self.total_energy += self.energy_used
        self.counters["steps"] = self.counters.get("steps", 0) + 1

    def can_add_cell(self, count: int = 1) -> bool:
        return self.energy_used + self.cell_cost * count <= self.energy_per_step and self.counters.get("cells", 0) + count <= self.max_cells

    def can_add_synapse(self, count: int = 1) -> bool:
        return self.energy_used + self.synapse_cost * count <= self.energy_per_step and self.counters.get("synapses", 0) + count <= self.max_synapses

    def added_cell(self) -> None:
        if self.energy_used + self.cell_cost > self.energy_per_step + 1e-12:
            raise RuntimeError("cell structural cost exceeds per-step energy budget")
        self.energy_used += self.cell_cost
        self.total_energy += self.cell_cost
        self.counters["cells"] = self.counters.get("cells", 0) + 1
        self.structural_events += 1

    def removed_cell(self) -> None:
        self.counters["cells"] = max(0, self.counters.get("cells", 0) - 1)
        self.structural_events += 1

    def added_synapse(self) -> None:
        if self.energy_used + self.synapse_cost > self.energy_per_step + 1e-12:
            raise RuntimeError("synapse structural cost exceeds per-step energy budget")
        self.energy_used += self.synapse_cost
        self.total_energy += self.synapse_cost
        self.counters["synapses"] = self.counters.get("synapses", 0) + 1
        self.structural_events += 1

    def removed_synapse(self) -> None:
        self.counters["synapses"] = max(0, self.counters.get("synapses", 0) - 1)
        self.structural_events += 1

    def restored_synapse(self) -> None:
        """Restore a rolled-back synapse without refunding consumed energy."""
        self.counters["synapses"] = self.counters.get("synapses", 0) + 1
        self.structural_events += 1

    def to_dict(self) -> Dict[str, object]:
        return {
            "max_cells": self.max_cells,
            "max_synapses": self.max_synapses,
            "energy_per_step": self.energy_per_step,
            "cell_cost": self.cell_cost,
            "synapse_cost": self.synapse_cost,
            "energy_used": self.energy_used,
            "total_energy": self.total_energy,
            "structural_events": self.structural_events,
            "counters": dict(self.counters),
        }
