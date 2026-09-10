# R2 Universal Event-Driven Organism

## Contracts

- Envelopes: `soma/events/envelope.py` (protocol v2, schema `formats/event-protocol/v2.json`).
  R0 v1 remains valid; v2 adds shape/duration/boundary/scope/uncertainty.
- Channels: declared `ChannelSchema` set in `ChannelRegistry`; envelopes validate
  signals and bounds against it. Unregistered channels rejected.
- Clocks: `LogicalClock` enforces per-source event_id +1 sequencing and monotone
  source clocks; `ChannelBuffer` joins multi-rate channels with a hard bound
  (default 64) and emits only complete frames.
- Correlation: every `ActionEvent` carries an id; `OutcomeEvent` must match the
  pending id exactly or the bridge raises. No misattribution path exists.
- Heads: `ScalarHead` / `DiscreteHead` bind core outputs to event proposals.
  Fixed mappings, deterministic mode, no learning in heads.
- Bridge: `soma/events/bridge.py` owns validation, scheduling, correlation,
  persistence. Alternating discipline (one pending action max), closed-state
  enforcement, resumable state (clock, registry, buffer, transducer cursors,
  pending records).

## Transducer SDK

`soma/transducers/sdk.py`: `TransducerSpec` (channels, input width, action
schema, hardware, reversibility) + `Transducer` base (sequencing, finite checks,
recursive forbidden-key scan, pack/unpack interface, state round-trip).

Adapters: `scalar-stream-v1` (1 scalar channel, width 2), `symbol-bits-v1`
(3 bit channels, width 4, 3-bit action), `bit-scalar-v1` (3 bit channels in,
scalar prediction out). All reversible, stdlib-only, no task knowledge.

## Finite limits

Buffer bound 64/channel, one pending action, finite clocks/counters, bounded
workspace = joined frame + packed vector + core step. Bridge state size scales
with buffer occupancy, not lifetime length.

## Cross-domain proof (locked fixture `soma/evaluation/temporal.py`)

Latent-context polarity A -> B -> A (150/150/150, seeds 0-7) through `scalar`
and `bits` adapters into the unchanged scaffolded core:

- scalar learn tails 0.674/0.634/0.540 vs frozen 0.2, shuffled <=0.28.
- bits learn tails 0.820/0.797/0.618 vs frozen 0.2, shuffled <=0.19.
- Adapter-only persistence is perfect on stationary phases and fails the
  switched phase (-0.600) in both domains, as predicted.
- All 7 gates pass per adapter (learn beats frozen/shuffled everywhere,
  beats persistence on the switched phase, margin 0.10).
- Task identity absent from brain state (quoted-key scan test).
- Exact resume at step-137 boundary for both adapters (rewards + state identical).
