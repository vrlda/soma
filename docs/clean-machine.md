# Clean-Machine Test Procedure (R8 Rehearsal)

On a machine without this repo's caches or state:

1. Clone the repository; confirm `git status` is clean.
2. Run `bash install.sh` with an empty `SOMA_ROOT`. It must exit 0 and
   print the smoke lines. Any failure aborts the release.
3. `python3 research/scripts/verify_hashes.py` must report 0 missing, 0 mismatch.
4. `python3 -m unittest discover -s tests` must pass in full.
5. `SOMA_BRAINS=$SOMA_ROOT/brains python3 -m soma.service.main doctor`
   must report `"ok": true`.
6. Service smoke: create a brain, teach the fixture note, correct a fact,
   ask it back, back up, restore into a second root, compare inspect
   output (see `demo_chat.py` for the scripted form).
7. Record machine class, OS, Python version, timings, and peak RSS next
   to the release tag. Supported matrix starts with Apple Silicon +
   Python 3.10-3.14; anything else is best-effort until profiled.

Interrupted install/update/checkpoint operations recover via journal
rotation (tested); report any case that does not.
