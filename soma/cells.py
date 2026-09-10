"""Stateful computational cells used by the organism."""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class Cell:
    """A rate-based cell with persistent state and local homeostasis."""

    id: str
    kind: str = "hidden"
    threshold: float = 0.0
    activation: float = 0.0
    adaptation: float = 0.0
    plasticity: float = 1.0
    age: int = 0
    utility: float = 0.0
    energy_usage: float = 0.0
    refractory: int = 0
    target_activity: float = 0.25
    activity_ema: float = 0.0
    # Signed activation mean drives threshold centering.  Keep it separate
    # from activity_ema, which tracks magnitude for homeostatic adaptation.
    signed_activity_ema: float = 0.0
    # Task-learned offset is distinct from homeostatic threshold centering.
    representation_bias: float = 0.0
    representation_bias_eligibility: float = 0.0
    # ``additive`` preserves the original rate-cell equation.  Binary product
    # cells retain the v8 equation; opt-in ``dendritic_product_n`` cells
    # multiply their declared direct-input afferents before the same bounded
    # activation equation.
    activation_type: str = "additive"
    dendritic_sources: Tuple[str, ...] = field(default_factory=tuple)
    # Bounded, persisted gain for composed product cells.  It defaults to one
    # so every legacy additive/binary cell keeps its exact equation.
    dendritic_normalizer: float = 1.0
    metadata: Dict[str, str] = field(default_factory=dict)

    def update_homeostasis(self, rate: float = 0.02) -> None:
        """Center signed activity without turning magnitude into a bias."""
        # Rate cells carry both polarities.  Using abs(activity) here makes
        # any active stream push threshold in one direction, eventually
        # suppressing the negative half of a signed policy.  Magnitude remains
        # available through activity_ema/adaptation; threshold only removes
        # persistent signed offset.
        self.threshold += rate * self.signed_activity_ema
        self.threshold = max(-1.5, min(1.5, self.threshold))

    def observe(self, value: float, ema_rate: float = 0.05) -> None:
        self.activation = max(-1.0, min(1.0, float(value)))
        self.activity_ema = (1.0 - ema_rate) * self.activity_ema + ema_rate * abs(self.activation)
        self.signed_activity_ema = (1.0 - ema_rate) * self.signed_activity_ema + ema_rate * self.activation
        self.utility = 0.995 * self.utility + 0.005 * abs(self.activation)
        self.age += 1
