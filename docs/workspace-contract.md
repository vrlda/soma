# Active workspace contract

- Holds: current events, recurrent activations, goals, retrieved memories, candidate circuits. Fast, strictly budgeted, mostly disposable.
- Limits: event-buffer size, active cells per event, active connections per event, retrieval bandwidth, wall-clock per event, total storage.
- Not knowledge: checkpoint stores resumable event clocks only; old info influences behavior only if consolidated, retrieved, or re-supplied.
- Measurement: events/second, active cells/event, active connections/event, structural mutations/event, retrieval bandwidth, durable bytes per million events.
