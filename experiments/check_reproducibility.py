"""Reproduce representative preserved runs using their recorded configurations."""
import json
from pathlib import Path

from experiments import run


FILES = (
    "sequence_initial.jsonl", "delay_initial.jsonl", "delay_ema_bank.jsonl",
    "delay_peak_bank.jsonl", "contextual_competitive.jsonl", "structure_novelty.jsonl",
    "growth_both.jsonl", "grammar_aligned_growth.jsonl", "grammar_pruning.jsonl",
    "lifetime_initial.jsonl", "lifetime_consolidation.jsonl", "lifetime_confirmation.jsonl",
)


def main():
    receipts = []
    for name in FILES:
        path = Path("experiments/results") / name
        row = json.loads(path.read_text().splitlines()[0])
        suite, seed, variant, overrides = row["suite"], row["seed"], row["variant"], row["config"]
        kwargs = {}
        if suite == "delay":
            kwargs["gap"] = row["condition"]
        elif suite == "contextual":
            kwargs["rehearsal"] = row["condition"]
        elif suite == "structure":
            kwargs["noise"] = row["condition"]
        elif suite == "grammar":
            kwargs["gap"] = row["condition"]
        result = getattr(run, suite)(seed, variant, overrides, **kwargs)
        checks = {"metrics": result["metrics"] == row["metrics"],
                  "brain": all(result["brain"][key] == value for key, value in row["brain"].items())}
        for key in ("phases", "recall_first_20", "recall_last_100"):
            if key in row:
                checks[key] = result[key] == row[key]
        receipt = {"file": name, "seed":seed, "variant":variant, "checks":checks,
                   "exact_match":all(checks.values())}
        receipts.append(receipt)
        print(json.dumps(receipt), flush=True)
    output = Path("experiments/report")
    output.mkdir(exist_ok=True)
    (output / "REPRODUCIBILITY.json").write_text(json.dumps(receipts, indent=2) + "\n")
    if not all(r["exact_match"] for r in receipts):
        raise SystemExit("Historical numerical behavior changed: inspect mismatched receipts")


if __name__ == "__main__":
    main()
