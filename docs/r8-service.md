# R8 Prosumer Service (`python -m soma.service.main`, 6 CLI tests green)

Named persistent brains under `~/.soma/brains` (override `SOMA_BRAINS`):
brain checkpoint owning sequence memory, turn-scoped dialogue memory,
episodic buffer, identity/ancestor manifest.

Commands: brain-create|list|clone|teach|correct|forget|inspect|backup|
restore|export|import, chat (REPL with /teach), serve (localhost HTTP with
OpenAI-compatible `/v1/chat/completions` plus `/learn`, `/inspect`,
`/health`), doctor, transducers. Verified end to end including a taught
fact recalled through the HTTP API.

Scope notes: dialogue memory budget can saturate on tiny teaches (bounded
by design); no authentication on the local API (localhost default).

Installer (`install.sh`): Python/hardware checks, layout creation, import
smoke, fast unit subset; fails loud. Presets (`configs/presets.json`):
Micro/Tiny/Small/Prosumer memory budgets. Clean-machine procedure:
`docs/clean-machine.md`. R6 quality floors: `research/scripts/check_floors.py`
(frozen bars over report JSONs; currently all passing).
