"""R7 preference harness: blind A/B pairs, agreement, win rates.

Format per pair: {id, prompt, A, B, order} where order records which system
produced A (blind). Raters record {id, choice: A/B/tie, rater}. Scoring
computes win rates with ties split, inter-rater agreement (Fleiss-style
pairwise match rate), and per-prompt breakdowns. Ships with a synthetic
demonstration rater only; human judgments arrive as JSON files.
"""

import json


def make_pair(pair_id, prompt, response_a, response_b, a_is_baseline):
    """Build a blinded pair. order=True means A is the baseline system."""
    return {
        "id": str(pair_id),
        "prompt": str(prompt),
        "A": str(response_a),
        "B": str(response_b),
        "order": bool(a_is_baseline),
    }


def validate_judgment(judgment, pair_ids):
    if not isinstance(judgment, dict):
        raise ValueError("judgment must be a mapping")
    if judgment.get("id") not in pair_ids:
        raise ValueError("judgment references unknown pair")
    if judgment.get("choice") not in ("A", "B", "tie"):
        raise ValueError("choice must be A, B, or tie")
    if not judgment.get("rater"):
        raise ValueError("judgment needs a rater id")
    return True


def score(pairs, judgments):
    """Win rates for the non-baseline system, ties split; agreement rate."""
    pair_ids = {pair["id"] for pair in pairs}
    by_pair = {}
    raters = set()
    for judgment in judgments:
        validate_judgment(judgment, pair_ids)
        raters.add(judgment["rater"])
        by_pair.setdefault(judgment["id"], []).append(judgment)
    wins = ties = losses = 0.0
    agreements, comparable = 0, 0
    for pair in pairs:
        votes = by_pair.get(pair["id"], [])
        for vote in votes:
            choice = vote["choice"]
            if choice == "tie":
                ties += 1
                continue
            baseline_picked = (choice == "A") == pair["order"]
            if baseline_picked:
                losses += 1
            else:
                wins += 1
        for index in range(len(votes)):
            for other in range(index + 1, len(votes)):
                comparable += 1
                if votes[index]["choice"] == votes[other]["choice"]:
                    agreements += 1
    total = wins + ties + losses
    return {
        "pairs": len(pairs),
        "judgments": len(judgments),
        "raters": sorted(raters),
        "win_rate": (wins + 0.5 * ties) / total if total else 0.0,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "agreement": agreements / comparable if comparable else None,
    }


def synthetic_rater(pair):
    """Demonstration rater only: prefers valid UTF-8, then shorter text."""
    def valid(text):
        try:
            text.encode("latin-1").decode("utf-8", errors="strict")
            return True
        except (UnicodeDecodeError, ValueError):
            return False
    valid_a, valid_b = valid(pair["A"]), valid(pair["B"])
    if valid_a != valid_b:
        return "A" if valid_a else "B"
    if len(pair["A"]) != len(pair["B"]):
        return "A" if len(pair["A"]) < len(pair["B"]) else "B"
    return "tie"


def load_json_list(path):
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as error:
        raise ValueError("cannot load preference file: %s" % error)
    if not isinstance(payload, list):
        raise ValueError("preference file must hold a JSON list")
    return payload
