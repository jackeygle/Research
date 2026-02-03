"""
Collect distillation experiment results and generate comparison plots.
"""

import argparse
import glob
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def parse_value(raw: str) -> Any:
    val = raw.strip()
    if val in {"None", ""}:
        return None
    if val.lower() in {"true", "false"}:
        return val.lower() == "true"
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val


def parse_results_file(path: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    with open(path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = parse_value(value)
    data["path"] = path
    return data


def collect_results(results_glob: str) -> List[Dict[str, Any]]:
    results = []
    for path in glob.glob(results_glob):
        results.append(parse_results_file(path))
    return results


def summarize_by_method(results: List[Dict[str, Any]], dataset: str) -> List[Dict[str, Any]]:
    filtered = [r for r in results if r.get("dataset") == dataset]
    summary = {}
    for r in filtered:
        method = "baseline" if not r.get("use_kd") else r.get("distill_method", "kd")
        best_acc = r.get("best_accuracy")
        if best_acc is None:
            continue
        if method not in summary or best_acc > summary[method]["best_accuracy"]:
            summary[method] = {
                "method": method,
                "best_accuracy": best_acc,
                "experiment": r.get("experiment"),
                "path": r.get("path"),
            }
    return list(summary.values())


def save_csv(summary: List[Dict[str, Any]], output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("method,best_accuracy,experiment,path\n")
        for row in summary:
            f.write(
                f"{row['method']},{row['best_accuracy']},"
                f"{row.get('experiment','')},{row.get('path','')}\n"
            )


def plot_summary(summary: List[Dict[str, Any]], output_path: str, title: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_sorted = sorted(summary, key=lambda x: x["best_accuracy"], reverse=True)
    methods = [s["method"] for s in summary_sorted]
    accs = [s["best_accuracy"] for s in summary_sorted]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(methods, accs, color="#3498db", edgecolor="black", linewidth=1.2)
    plt.ylabel("Best Accuracy (%)")
    plt.title(title)
    plt.ylim([min(accs) - 1.0, max(accs) + 1.0])

    for bar, acc in zip(bars, accs):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"{acc:.2f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare distillation methods")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--results-glob", type=str, default="./checkpoints/*_results.txt")
    parser.add_argument("--csv-out", type=str, default="./results/distill_method_comparison.csv")
    parser.add_argument("--fig-out", type=str, default="./results/figures/distill_methods_comparison.png")
    args = parser.parse_args()

    results = collect_results(args.results_glob)
    summary = summarize_by_method(results, dataset=args.dataset)
    if not summary:
        raise RuntimeError("No results found for the selected dataset.")

    save_csv(summary, args.csv_out)
    plot_summary(summary, args.fig_out, title=f"Distillation Method Comparison ({args.dataset})")
    print(f"Saved CSV: {args.csv_out}")
    print(f"Saved Figure: {args.fig_out}")


if __name__ == "__main__":
    main()



