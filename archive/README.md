# Archive (reference only)

Everything SOMA built from 2025 to 2026-10-02, before the project
re-centred on the vision in [`../VISION.md`](../VISION.md). It is kept for its
lessons and numbers. **It is not maintained and is not runnable from this
location**: paths and imports assume the old layout.

To run any of it, check out the last commit with the old layout:

```bash
git checkout b6606cd   # "Long-range dev log ..." on claude/youthful-ritchie-vq1bog
```

| Folder | What it was |
|---|---|
| `v1/soma/` | the v1 Python package: the organism kernel (cells, synapses, plasticity, growth, pruning), the event system, the circuit-mixing language memory, and the brain service (CLI, chat, HTTP API, `.soma` files, signing) |
| `v1/engine/` | the Rust engine: graph kernels, `soma-mixer` (circuit mixing), `soma-longmix` (long-range circuits and the copy pointer) |
| `v1/tests/`, `v1/docs/`, `v1/configs/`, `v1/formats/` | product tests, product docs and ADRs 0001–0010, presets, the event-protocol spec |
| `v1/SOMA_REAL_MODEL_MASTER_PLAN.md` | the old engineering plan (superseded by `VISION.md`) |
| `v1/PROJECT.MD` | the founding research vision, still worth reading. `VISION.md` carries its spirit forward. |
| `research/` | benchmarks, pilots, frozen reports (large ones in Git LFS), per-milestone records R0–R12, and the 2026-10 long-range development log |

Key results worth remembering are summarized in `VISION.md` section 8.
