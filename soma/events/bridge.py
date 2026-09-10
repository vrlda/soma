"""Event bridge: envelopes in, action events out, outcomes correlated back.

The bridge owns scheduling, validation, clocks, buffers, correlation, and
persistence. It holds no task intelligence: vectors cross into the core,
targets and rewards are computed outside by the fixture/evaluator.
"""

import math

from ..organism import Modulators
from .channel import ChannelRegistry
from .clock import ChannelBuffer, LogicalClock
from .envelope import ActionEvent, Event, OutcomeEvent


class EventBridge(object):
    """Binds one organism to one transducer over declared channels."""

    VERSION = 1

    def __init__(self, organism, transducer, max_pending=1, max_buffered=64,
                 novelty=0.05, exploration=0.02):
        from ..transducers.sdk import Transducer
        if not isinstance(transducer, Transducer):
            raise ValueError("bridge requires a Transducer")
        self.organism = organism
        self.transducer = transducer
        self.novelty = float(novelty)
        self.exploration = float(exploration)
        for value in (self.novelty, self.exploration):
            import math as _math
            if not _math.isfinite(value) or value < 0.0:
                raise ValueError("bridge modulators must be nonnegative and finite")
        self.max_pending = int(max_pending)
        if self.max_pending < 1:
            raise ValueError("max_pending must be positive")
        self.registry = ChannelRegistry()
        for schema in transducer.spec.channels:
            self.registry.register(schema)
        self.registry.register(transducer.spec.action_schema)
        self.clock = LogicalClock()
        self.buffer = ChannelBuffer(
            [schema.name for schema in transducer.spec.channels], max_buffered=max_buffered)
        self.pending_reward = 0.0
        self.pending_action_id = None
        self.next_action_id = 0
        self.steps = 0
        self.closed = False

    def _check_vector(self, values):
        if len(values) != self.transducer.input_size:
            raise ValueError("adapter packed %d inputs, spec declares %d" % (
                len(values), self.transducer.input_size))
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError("core input vector must be finite numbers")
        return [float(value) for value in values]

    def ingest(self, envelopes):
        """Validate envelopes, join one frame, step the core, emit an action.

        Any outcome for the previous action must have been delivered first
        via `outcome()`; at most max_pending actions await attribution.
        """
        if self.closed:
            raise ValueError("bridge is closed")
        if not envelopes:
            raise ValueError("ingest requires at least one envelope")
        if self.pending_action_id is not None and self.pending_reward is None:
            raise ValueError("previous action %s has no outcome yet" % self.pending_action_id)
        events = [Event.from_dict(payload) if isinstance(payload, dict) else payload
                  for payload in envelopes]
        for event in events:
            self.registry.check_event(event)
            self.clock.observe(event)
            self.buffer.push(event)
        frame = self.buffer.pop_frame()
        if frame is None:
            return None
        vector = self._check_vector(self.transducer.pack_frame(frame))
        reward = self.pending_reward if self.pending_reward is not None else 0.0
        result = self.organism.step(
            vector, Modulators(reward=reward, novelty=self.novelty, exploration=self.exploration))
        self.pending_reward = None
        proposal = self.transducer.unpack_action(result.outputs)
        self.transducer.spec.action_schema.check_payload(proposal)
        action = ActionEvent(
            self.transducer.spec.action_schema.name, self.next_action_id,
            float(self.clock.tick), proposal, self.transducer.spec.name + ":action",
            correlation_id=self.next_action_id)
        self.pending_action_id = self.next_action_id
        self.next_action_id += 1
        self.steps += 1
        return action.to_dict()

    def outcome(self, payload):
        """Attribute one delayed outcome to its action. Never misattributes."""
        if self.closed:
            raise ValueError("bridge is closed")
        event = OutcomeEvent.from_dict(payload) if isinstance(payload, dict) else payload
        if self.pending_action_id is None:
            raise ValueError("outcome with no pending action")
        if int(event.correlation_id) != int(self.pending_action_id):
            raise ValueError("outcome correlation %s does not match pending action %s" % (
                event.correlation_id, self.pending_action_id))
        self.pending_reward = float(event.outcome)
        self.pending_action_id = None
        return True

    def close(self, final_outcome=None):
        """Close the lifetime, applying any terminal outcome without a new decision."""
        if self.closed:
            raise ValueError("bridge is closed")
        if final_outcome is not None:
            self.outcome(final_outcome)
        if self.pending_reward is not None and getattr(self.organism, "_pending_outcome", False):
            self.organism.apply_outcome(self.pending_reward)
            self.pending_reward = None
        self.closed = True
        return True

    def state_dict(self):
        return {
            "version": self.VERSION,
            "transducer": self.transducer.state_dict(),
            "registry": self.registry.to_dict(),
            "clock": self.clock.to_dict(),
            "buffer": self.buffer.to_dict(),
            "pending_reward": self.pending_reward,
            "pending_action_id": self.pending_action_id,
            "next_action_id": self.next_action_id,
            "steps": self.steps,
            "closed": self.closed,
            "max_pending": self.max_pending,
            "novelty": self.novelty,
            "exploration": self.exploration,
        }

    def load_state_dict(self, payload):
        if not isinstance(payload, dict) or payload.get("version") != self.VERSION:
            raise ValueError("unsupported bridge state version")
        self.transducer.load_state_dict(payload["transducer"])
        self.registry = ChannelRegistry.from_dict(payload["registry"])
        self.clock = LogicalClock.from_dict(payload["clock"])
        self.buffer = ChannelBuffer.from_dict(payload["buffer"])
        self.pending_reward = payload["pending_reward"]
        if self.pending_reward is not None:
            self.pending_reward = float(self.pending_reward)
        self.pending_action_id = payload["pending_action_id"]
        if self.pending_action_id is not None:
            self.pending_action_id = int(self.pending_action_id)
        self.next_action_id = int(payload["next_action_id"])
        self.steps = int(payload["steps"])
        self.closed = bool(payload["closed"])
        if int(payload.get("max_pending", 1)) != self.max_pending:
            raise ValueError("bridge max_pending mismatch on resume")
        if float(payload.get("novelty", self.novelty)) != self.novelty:
            raise ValueError("bridge novelty mismatch on resume")
        if float(payload.get("exploration", self.exploration)) != self.exploration:
            raise ValueError("bridge exploration mismatch on resume")
