import argparse
import json
from pathlib import Path


def load_summaries(outputs_dir: Path):
    summaries = []
    for path in sorted(outputs_dir.glob("summary_*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        tag = data.get("experiment_tag")
        if not tag:
            tag = path.stem.replace("summary_", "")
        summaries.append((path, data, tag))
    return summaries


def build_rows(summaries):
    rows = []
    for path, data, tag in summaries:
        defense = data.get("defense", "")
        config = data.get("config", "")
        models = data.get("models", {})
        for model_name, stats in models.items():
            inj = stats.get("injection", {})
            ben = stats.get("benign", {})
            overall = stats.get("overall", {})
            rows.append(
                {
                    "experiment_tag": tag,
                    "summary_file": str(path),
                    "model": model_name,
                    "defense": defense,
                    "config": config,
                    "injection_success_rate": inj.get("attack_success_rate", 0.0),
                    "benign_refusal_rate": ben.get("refusal_rate", 0.0),
                    "avg_latency_s": overall.get("avg_latency_s", 0.0),
                }
            )
    return rows


def write_csv(rows, output_path: Path):
    if not rows:
        return
    cols = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")


def plot(rows, plot_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib not available, skipping plots: {exc}")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)
    models = sorted({r["model"] for r in rows})
    for model in models:
        r = [x for x in rows if x["model"] == model]
        labels = [x["experiment_tag"] for x in r]
        inj = [x["injection_success_rate"] for x in r]
        ben = [x["benign_refusal_rate"] for x in r]
        lat = [x["avg_latency_s"] for x in r]

        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        axes[0].bar(labels, inj)
        axes[0].set_ylabel("Injection success")
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, axis="y", linestyle="--", alpha=0.4)

        axes[1].bar(labels, ben)
        axes[1].set_ylabel("Benign refusal")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)

        axes[2].bar(labels, lat)
        axes[2].set_ylabel("Avg latency (s)")
        axes[2].grid(True, axis="y", linestyle="--", alpha=0.4)

        axes[2].tick_params(axis="x", rotation=30, labelsize=8)
        fig.suptitle(f"LLM Security Metrics: {model}")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{model}_summary.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--plot_dir", default="outputs/plots")
    parser.add_argument("--csv_path", default="outputs/summary_table.csv")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    summaries = load_summaries(outputs_dir)
    rows = build_rows(summaries)
    if not rows:
        print("No summary files found.")
        return
    write_csv(rows, Path(args.csv_path))
    plot(rows, Path(args.plot_dir))
    print(f"Wrote CSV: {args.csv_path}")
    print(f"Wrote plots to: {args.plot_dir}")


if __name__ == "__main__":
    main()
