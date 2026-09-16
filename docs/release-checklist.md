# Release Checklist (Rehearsal — R10 Not Declared)

For any tagged release candidate, in order, same locked revision:

1. `python3 scripts/verify_hashes.py` — every tracked artifact matches.
2. `python3 -m unittest discover -s tests` — full suite green.
3. Frozen benchmark sweep: lifetime, final, r3b E0/E1/E2, fusion,
   generation, dialogue, systematicity, lifelong, instruction, tools,
   red-team. Reports land in reports/ with hashes appended.
4. `doctor` on a clean checkout (fresh clone, no caches): ok true.
5. Service smoke: create, teach, correct, ask, backup, restore, serve
   health (see demo_chat.py pattern).
6. Update CHANGELOG.md, VERSION, model card numbers, README board.
7. Tag `vX.Y.Z`, record tag hash; export demo brain `.soma` + SHA-256.

Upgrade policy: checkpoints migrate forward only (from_state_dict
defaults); old majors load read-only when migration is absent. Never
silently discard private lifetime deltas; conflicts surface, never merge.
