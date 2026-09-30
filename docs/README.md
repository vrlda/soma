# SOMA documentation index

Status and the ordered work list live in one place: Sections 16 and 22 of
[`SOMA_REAL_MODEL_MASTER_PLAN.md`](../SOMA_REAL_MODEL_MASTER_PLAN.md).
Files here are **records**: what was built, how it was gated, and where it
stops. When a record and the master plan disagree about status, the plan
wins; fix the record.

## Milestone records

| Milestone | Status | Record |
|---|---|---|
| R0 preserve kernel | ✅ | [reproducibility.md](reproducibility.md), [checkpoint-schema-v12.md](checkpoint-schema-v12.md), [genesis-state.md](genesis-state.md), [workspace-contract.md](workspace-contract.md), [soma-artifact-v1.md](soma-artifact-v1.md), [plan used](superpowers/plans/2026-09-10-r0-preserve-kernel.md) (historical) |
| R1 evidence router | ✅ | [r1-calibration.md](r1-calibration.md) |
| R2 event organism | ✅ | [r2-event-system.md](r2-event-system.md) |
| R3A sequences | ✅ | [r3a-sequence.md](r3a-sequence.md) |
| R3B English | ✅ | [r3b-english.md](r3b-english.md) (log, newest first), [r3b-reference-baselines.md](r3b-reference-baselines.md), [model-card-e2.md](model-card-e2.md) |
| R3D dialogue, systematicity | ✅ | [r3d-dialogue.md](r3d-dialogue.md), [r3d-systematicity.md](r3d-systematicity.md) |
| R4 continual | ✅ | [r4-lifelong.md](r4-lifelong.md) |
| R5 engine | partial | [r5-engine.md](r5-engine.md), [../engine/README.md](../engine/README.md) |
| R6 useful scale | partial | **[r6-circuit-mixing.md](r6-circuit-mixing.md)** (current language result), [r6-untouched-test.md](r6-untouched-test.md) (pre-registered untouched test), [r6-tier-qualification.md](r6-tier-qualification.md), [ADR 0006](adr/0006-r6-control-margins-evidence.md) |
| R7 instruction | partial | [r7-instruction.md](r7-instruction.md) |
| R8 service | partial | [r8-service.md](r8-service.md), [clean-machine.md](clean-machine.md) |
| R9 red-team, ops | partial | [r9-redteam.md](r9-redteam.md), [runbooks.md](runbooks.md) |
| R10 release | open | [release-checklist.md](release-checklist.md) (rehearsal) |
| R11 vision | first step | [r11-vision.md](r11-vision.md) |
| R12 embodied | first step | [r12-embodied.md](r12-embodied.md) |

R3C generation is recorded inside [r3b-english.md](r3b-english.md).

## R6 development pilots (non-gating)

These are development-only mechanism experiments. None of them qualifies
R6, and none changes the locked reports. They are relevant to master plan
§22 step 6 (whether organism-level plasticity belongs on the language
path).

- [r6-english-event-pilot.md](r6-english-event-pilot.md), [r6-english-lag-pilot.md](r6-english-lag-pilot.md): English through the organism event path.
- [r6-temporal-composition-pilot.md](r6-temporal-composition-pilot.md), [r6-temporal-composition-pilot-v4.md](r6-temporal-composition-pilot-v4.md): leave-one-combination-out temporal composition.
- [r6-text-temporal-transport-pilot.md](r6-text-temporal-transport-pilot.md), [r6-text-temporal-transport-2048.md](r6-text-temporal-transport-2048.md): the same composition through the real text transducer.
- [r6-compositional-byte-event.md](r6-compositional-byte-event.md): byte/event product discriminator.
- [r6-persistent-scout-calibration.md](r6-persistent-scout-calibration.md), [r6-persistent-scout-power-study.md](r6-persistent-scout-power-study.md): scout thresholds and statistical power.

## Architecture decision records

- [0001](adr/0001-runtime-brain-transducer.md) runtime, brain, and transducer are separate artifacts
- [0002](adr/0002-event-protocol.md) universal event protocol
- [0003](adr/0003-workspace-vs-memory.md) active workspace vs persistent memory
- [0004](adr/0004-transducer-boundary.md) transducers hold no task intelligence
- [0005](adr/0005-blank-vs-trained.md) blank vs trained brains
- [0006](adr/0006-r6-control-margins-evidence.md) R6 control-margin gates superseded by evidence

## Operations

[dependencies.md](dependencies.md) · [clean-machine.md](clean-machine.md) ·
[release-checklist.md](release-checklist.md) · [runbooks.md](runbooks.md)
