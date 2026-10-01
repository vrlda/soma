# R3D Dialogue Mechanics (`r3d_dialogue_benchmark.py`, all gates pass)

Three tiers, one decision path: exact episodic rules, turn-scoped order-256
dialogue table, frozen E0 background. Teaching exposes fact text to dialogue
memory and records a question->answer episodic rule; corrections supersede by
trigger identity with provenance. Emissions are UTF-8 constrained and never
train protected state.

Frozen gates: uptake, correction supersede, 3-turn interference, save/load
restart, table-alone and lesioned controls fail, release sanitize + reload +
continued learning of a new fact. Semantic relevance is limited to taught
facts (no comprehension claim); uncertainty expression and multi-skill
dialogue remain open.
