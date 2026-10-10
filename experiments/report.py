"""Summarize preserved results and export scientific figures (Matplotlib)."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import statistics


ROOT = Path(__file__).resolve().parents[1]


def describe(values):
    if not values:
        return None
    return {"mean": statistics.mean(values), "min": min(values), "max": max(values),
            "sd": statistics.stdev(values) if len(values) > 1 else 0.0}


def read_results():
    rows, manifest = [], []
    for path in sorted((ROOT / "experiments/results").glob("*.jsonl")):
        content = path.read_bytes()
        records = [json.loads(line) for line in content.decode().splitlines()]
        for row in records:
            row["result_file"] = path.name
        rows.extend(records)
        manifest.append({"path": str(path.relative_to(ROOT)), "runs": len(records),
                         "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
    return rows, manifest


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["result_file"], r["variant"], str(r["condition"]))].append(r)
    summaries = []
    for (name, variant, condition), rr in sorted(groups.items()):
        complete = [r for r in rr if r.get("complete", True)]
        summaries.append(dict(file=name, variant=variant, condition=condition,
            runs=len(rr), complete_runs=len(complete), seeds=[r["seed"] for r in rr],
            accuracy=describe([r["metrics"]["accuracy"] for r in complete if r["metrics"]["accuracy"] is not None]),
            log_loss=describe([r["metrics"]["log_loss"] for r in complete if r["metrics"]["log_loss"] is not None]),
            new_context=describe([r["phases"][1]["last_200"]["accuracy"] for r in complete if r["suite"] == "contextual"]),
            old_probe=describe([r["recall_first_20"]["accuracy"] for r in complete if r["suite"] == "contextual"]),
            oracle=describe([r["oracle"]["accuracy"] for r in complete if "oracle" in r]),
            hidden=describe([r["brain"]["hidden"] for r in rr]),
            births=describe([r["brain"]["births"] for r in rr]),
            deaths=describe([r["brain"]["deaths"] for r in rr]),
            config=rr[0]["config"]))
    return summaries


def select(rows, file, variant, condition):
    return [r for r in rows if r["result_file"] == file and r["variant"] == variant
            and r["condition"] == condition and r.get("complete", True)]


def plots(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    gaps = (0, 4, 16, 64)
    for file, variant, label, color in (
        ("delay_initial.jsonl", "candidate", "Initial averaging traces (5 seeds)", "#bf4b45"),
        ("delay_confirmation.jsonl", "candidate", "Selected event traces (10 seeds)", "#177d76"),
        ("delay_confirmation.jsonl", "no_traces", "Temporal state removed (10 seeds)", "#667085"),
    ):
        x, mean, low, high = [], [], [], []
        for gap in gaps:
            rr = select(rows, "delay64_confirmation.jsonl" if gap == 64 and variant == "candidate" and "confirmation" in file else file, variant, gap)
            if not rr:
                continue
            values = [r["metrics"]["accuracy"] for r in rr]
            x.append(gap); mean.append(statistics.mean(values)); low.append(min(values)); high.append(max(values))
        axes[0].plot(x, mean, "o-", label=label, color=color)
        axes[0].fill_between(x, low, high, color=color, alpha=.12)
    axes[0].axhline(.5, color="#b3b3b3", linestyle="--", linewidth=1)
    axes[0].set(title="Remember one cue", xlabel="Intervening symbols", ylabel="Prequential accuracy", ylim=(0, 1.05))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(fontsize=8, loc="lower right")
    for i, (variant, label) in enumerate((("candidate", "Competitive"), ("no_hidden_plasticity", "Frozen cells"), ("no_hidden", "Linear only"))):
        rr = select(rows, "contextual_confirmation.jsonl", variant, "none")
        if rr:
            axes[1].bar(i-.18, statistics.mean(r["phases"][1]["last_200"]["accuracy"] for r in rr), .34, color="#177d76", label="New context, final 200" if i == 0 else None)
            axes[1].bar(i+.18, statistics.mean(r["recall_first_20"]["accuracy"] for r in rr), .34, color="#d18a32", label="Old context, first 20 (learning)" if i == 0 else None)
    axes[1].set_xticks(range(3), ("Competitive", "Frozen cells", "Linear only"))
    axes[1].set(title="Opposite mappings in two contexts", ylim=(0, 1.05))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(fontsize=8, loc="lower left")
    for file, label, color in (("structure_prediction_matched.jsonl", "Prediction-error growth", "#bf4b45"),
                               ("structure_novelty.jsonl", "Representation-novelty growth", "#177d76")):
        rr = select(rows, file, "structure", True)
        if rr:
            ticks = [t["tick"] for t in rr[0]["trajectory"]]
            for r in rr:
                axes[2].plot(ticks, [t["hidden"] for t in r["trajectory"]], color=color, alpha=.15)
            axes[2].plot(ticks, [statistics.mean(r["trajectory"][i]["hidden"] for r in rr) for i in range(len(ticks))], color=color, label=label)
    axes[2].set(title="Matched growth on random labels", xlabel="Experiences", ylabel="Hidden neurons")
    axes[2].legend(fontsize=8)
    fig.suptitle("SOMA: small-task evidence, not language or general intelligence", fontsize=13)
    for ext in ("svg", "png"):
        fig.savefig(output / ("first_results." + ext), dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    for file, variant, label, color in (
        ("grammar_initial.jsonl", "candidate", "Fixed 16 cells, local temporal input", "#bf4b45"),
        ("grammar_confirmation.jsonl", "candidate", "Recruited representations", "#177d76"),
        ("homeostasis_confirmation.jsonl", "candidate", "Local variance normalization", "#5a4eb2"),
        ("grammar_confirmation.jsonl", "no_growth", "Recruitment removed", "#667085"),
        ("grammar_confirmation.jsonl", "no_drive_memory", "Input traces removed", "#d18a32"),
    ):
        x, y, low, high = [], [], [], []
        for gap in (0, 2, 8):
            rr = select(rows, file, variant, gap)
            if rr:
                values = [r["metrics"]["accuracy"] for r in rr]
                x.append(gap); y.append(statistics.mean(values)); low.append(min(values)); high.append(max(values))
        ax.plot(x, y, "o-", label=label, color=color)
        ax.fill_between(x, low, high, color=color, alpha=.12)
    ax.axhline(.5, color="#b3b3b3", linestyle="--", linewidth=1)
    ax.set(title="Two-symbol grammar: terminal class depends on both symbols", xlabel="Neutral symbols between the two cues", ylabel="Prequential accuracy", ylim=(0, 1.05))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=9, loc="lower left")
    for ext in ("svg", "png"):
        fig.savefig(output / ("temporal_grammar." + ext), dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4.5), layout="constrained")
    cases = (("lifetime_initial.jsonl", "candidate", None, "Original pruning"),
             ("lifetime_confirmation.jsonl", "candidate", None, "Consolidation protected"),
             ("lifetime_confirmation.jsonl", "no_retention", None, "Protection removed"),
             ("lifetime_confirmation.jsonl", "no_pruning", None, "Pruning removed"))
    for i, (file, variant, condition, label) in enumerate(cases):
        rr = select(rows, file, variant, condition)
        if not rr:
            continue
        ax.bar(i-.18, statistics.mean(r["phases"][1]["last_400"]["accuracy"] for r in rr), .34,
               color="#177d76", label="New context, final 400" if i == 0 else None)
        ax.bar(i+.18, statistics.mean(r["metrics"]["accuracy"] for r in rr), .34,
               color="#d18a32", label="Old return, first 50 (learning)" if i == 0 else None)
        ax.plot(i+.18, statistics.mean(r["phases"][2]["first_1"]["accuracy"] for r in rr),
                "ko", markersize=5, label="First old response (before teaching)" if i == 0 else None)
    ax.set_xticks(range(len(cases)), [x[3] for x in cases])
    ax.set(title="One lifetime: original mapping → reversed context → original context",
           ylabel="Prequential accuracy", ylim=(0, 1.05))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(fontsize=9, loc="lower left")
    for ext in ("svg", "png"):
        fig.savefig(output / ("lifetime_retention." + ext), dpi=180)
    plt.close(fig)


def main():
    output = ROOT / "experiments/report"
    output.mkdir(exist_ok=True)
    rows, manifest = read_results()
    summaries = summarize(rows)
    data = dict(python=platform.python_version(), runs=len(rows),
                complete_runs=sum(r.get("complete", True) for r in rows),
                files=manifest, groups=summaries)
    (output / "SUMMARY.json").write_text(json.dumps(data, indent=2) + "\n")
    lines = ["# All preserved experiments", "", "Generated by `python -m experiments.report`.", "",
             "Accuracy is the mean over independent seeds; brackets show the observed seed range, not a confidence interval.",
             "Loss is natural-log loss in nats. Incomplete runs are excluded from accuracy summaries.",
             "Context recall uses the first 20 predictions of a learning checkpoint clone; it includes relearning.", "",
             "| Results file | Variant | Condition | Complete runs | Accuracy / new context | Old probe | Hidden cells | Births | Deaths |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    def percentage(x):
        return f"{100*x['mean']:.1f}% [{100*x['min']:.1f}, {100*x['max']:.1f}]" if x else "—"
    for g in summaries:
        lines.append(f"| {g['file']} | {g['variant']} | {g['condition']} | {g['complete_runs']}/{g['runs']} | {percentage(g['accuracy'] or g['new_context'])} | {percentage(g['old_probe'])} | {g['hidden']['mean']:.1f} | {g['births']['mean']:.1f} | {g['deaths']['mean']:.1f} |")
    lines.extend(("", "An empty `grammar_recruitment.jsonl` is the interrupted query-only growth run, not a missing success.",
                  "Its separately reproduced partial trajectory is in `results/recruitment_failure_diagnostic.json`.",
                  "Raw rows from early runs predate per-run source hashing; configurations and seeds are preserved.",
                  "Later rows include hashes of the source used. The notebook and protocol explain changes.", ""))
    (output / "SUMMARY.md").write_text("\n".join(lines))
    plots(rows, output)
    print(json.dumps({"runs":len(rows), "complete_runs":data["complete_runs"], "groups":len(summaries), "output":str(output)}))


if __name__ == "__main__":
    main()
