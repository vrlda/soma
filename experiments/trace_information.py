"""An explicitly hand-designed ORACLE baseline, never part of SOMA."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random

from brain import Config, Organism
from experiments.run import metrics, onehot, score


def main():
    overrides = json.loads(Path("experiments/SELECTED_GRAMMAR.json").read_text())["overrides"]
    path = Path("experiments/results/trace_information.jsonl")
    if path.exists():
        raise SystemExit("Refusing to overwrite " + str(path))
    with path.open("w") as handle:
        for gap in (8, 16, 64):
            for seed in range(5):
                brain = Organism(Config(inputs=6, outputs=2, seed=seed, **overrides))
                stream = random.Random(seed + 60000)
                scores, oracle_scores, snapshots = [], [], []
                for episode in range(1200):
                    for _ in range(8):
                        brain.experience([0.0] * 6)
                    first, second = stream.randrange(2), stream.randrange(2)
                    brain.experience(onehot(first, 6))
                    for _ in range(gap):
                        brain.experience(onehot(4, 6))
                    brain.experience(onehot(2 + second, 6))
                    receipt = []
                    target = first ^ second
                    probabilities = brain.experience(onehot(5, 6), target, observer=receipt.append)
                    snapshot = receipt[0]
                    traces = {nid: 0.0 for nid in snapshot["sensory"]}
                    for source, _, kind, trace in snapshot["traces"]:
                        if kind == "drive" and source in traces:
                            traces[source] = max(traces[source], trace)
                    sensory = snapshot["sensory"]
                    # Task rule lives ONLY in this explicitly labelled oracle.
                    a = int(traces[sensory[1]] > traces[sensory[0]])
                    b = int(traces[sensory[3]] > traces[sensory[2]])
                    scores.append(score(probabilities, target))
                    oracle_scores.append(score(onehot(a ^ b, 2), target))
                    if episode % 100 == 99:
                        snapshots.append({"episode":episode + 1, "first_pair": [traces[sensory[0]], traces[sensory[1]]],
                                          "second_pair": [traces[sensory[2]], traces[sensory[3]]],
                                          "minimum_mismatch":brain.current_representation_surprise})
                row = dict(suite="trace_information", variant="candidate", condition=gap,
                           seed=seed, config=asdict(brain.config), metrics=metrics(scores[-400:]),
                           oracle=metrics(oracle_scores[-400:]), oracle_kind="hand-designed information upper bound",
                           brain=brain.summary(), trajectory=[], trace_samples=snapshots,
                           source_hashes={"brain/organism.py":hashlib.sha256(Path("brain/organism.py").read_bytes()).hexdigest(),
                                          "experiments/trace_information.py":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
                handle.write(json.dumps(row, allow_nan=False) + "\n"); handle.flush()
                print(json.dumps({"gap":gap,"seed":seed,"brain":row["metrics"],"oracle":row["oracle"]}), flush=True)


if __name__ == "__main__":
    main()
