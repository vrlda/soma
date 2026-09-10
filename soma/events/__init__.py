"""R2 universal event protocol: envelopes, channels, clocks, outcomes.

No organism imports. The core never sees task identity; adapters translate.
"""

from .bridge import EventBridge
from .channel import ChannelRegistry, ChannelSchema
from .clock import ChannelBuffer, LogicalClock
from .envelope import ActionEvent, Event, OutcomeEvent
from .heads import DiscreteHead, ScalarHead

__all__ = [
    "ActionEvent",
    "ChannelBuffer",
    "ChannelRegistry",
    "ChannelSchema",
    "DiscreteHead",
    "Event",
    "EventBridge",
    "LogicalClock",
    "OutcomeEvent",
    "ScalarHead",
]
