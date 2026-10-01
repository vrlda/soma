# Systematic Composition (`r3d_systematicity_benchmark.py`, all gates pass)

Substring multi-trigger matching with fired-once sequential emission:
known parts recombine in novel configurations. Teach color->red,
shape->cube (+ unused size->big distractor); novel ask "KEY color and KEY
shape are " emits both completions ("cubered", valid UTF-8).

Frozen gates: composition (both parts present), precision (no distractor,
unrelated asks clean), single recall intact, legacy suffix-only matching
fires nothing on the combo, entry removal destroys recall. Ordering follows
trigger recency (documented, deterministic); semantic role structure is
explicitly not claimed.
