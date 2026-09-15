# R7 Instruction Behavior (`r7_instruction_benchmark.py`, all gates pass)

Router order: refusal policy, declared skill patterns (repeat, spell),
episodic recall, table generation, calibrated uncertainty. Skills are exact
deterministic transforms, disclosed as tools.

Frozen gates: skills 4/4 on novel inputs, taught-fact recall, 3/3
out-of-training uncertainty with byte-vocabulary gating, refusal with benign
neighbors answered, zero E0 collateral from corrections.

Honest scope: uncertainty fires on vocabulary novelty and sub-byte-order
backoff; fluent in-distribution nonsense is answered from marginals
(documented miscalibration). The instruct-level episodic check needs
background history covering the trigger; generation always checks episodic
on full dialogue history. Skill set is minimal (more tools later).
