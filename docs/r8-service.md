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
by design); runtime install is local detection only (compiled engine is
R5); no authentication on the local API (localhost default).
