"""Observe confidence on familiar/novel experiences; no abstention policy."""
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from brain import Organism
from experiments.run import metrics, score


def main():
    output = Path("experiments/results/confidence_diagnostic.jsonl")
    if output.exists():
        raise SystemExit("Refusing to overwrite " + str(output))
    conditions = (("known_A", 0, 8), ("known_B", 1, 8),
                  ("novel_context_half", .5, 8), ("novel_context_negative", -.5, 8),
                  ("new_gap_16", 0, 16), ("new_gap_32", 0, 32), ("new_gap_64", 0, 64))
    source_hashes = {"brain/organism.py": hashlib.sha256(Path("brain/organism.py").read_bytes()).hexdigest(),
                     "experiments/confidence.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with output.open("w") as handle:
        for seed in range(5):
            checkpoint = Path(f"experiments/checkpoints/combined_initial/lifetime_candidate_8_{seed}.json")
            for name, context, gap in conditions:
                brain = Organism.load(checkpoint)  # each probe is its own clone
                stream = random.Random(seed + 80000)
                scores, observations = [], []
                for episode in range(40):
                    def signal(index=None):
                        values = [0.0] * 7; values[6] = float(context)
                        if index is not None:
                            values[index] = 1.0
                        return values
                    for _ in range(8):
                        brain.experience(signal())
                    first, second = stream.randrange(2), stream.randrange(2)
                    brain.experience(signal(first))
                    for _ in range(gap):
                        brain.experience(signal(4))
                    brain.experience(signal(2 + second))
                    receipts = []
                    probabilities = brain.experience(signal(5), observer=receipts.append)
                    # No target is ever supplied in these probe lifetimes.
                    if context in (0, 1):
                        scores.append(score(probabilities, first ^ second ^ int(context)))
                    active = {nid for nid in brain.hidden if brain.neurons[nid].activity > .5}
                    evidence = [s.evidence for s in brain.synapses
                                if s.kind == "prediction" and s.source in active]
                    observations.append(dict(episode=episode + 1, confidence=max(probabilities),
                        entropy=-sum(p * math.log(max(p, 1e-300)) for p in probabilities),
                        query_evidence=statistics.mean(evidence) if evidence else 0.0,
                        mismatch=brain.current_representation_surprise))
                row = dict(suite="confidence_diagnostic", variant="candidate", condition=name,
                           seed=seed, config=brain.state()["config"], metrics=metrics(scores),
                           first_10=metrics(scores[:10]), brain=brain.summary(), trajectory=[],
                           gap=gap, context=context, labels_supplied=0,
                           confidence=statistics.mean(x["confidence"] for x in observations),
                           entropy=statistics.mean(x["entropy"] for x in observations),
                           query_evidence=statistics.mean(x["query_evidence"] for x in observations),
                           observations=observations, checkpoint=str(checkpoint),
                           checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                           source_hashes=source_hashes)
                handle.write(json.dumps(row, allow_nan=False) + "\n"); handle.flush()
                print(json.dumps({k:row[k] for k in ("condition", "seed", "metrics", "confidence", "query_evidence")}), flush=True)


if __name__ == "__main__":
    main()
