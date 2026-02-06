from typing import Dict, List
from collections import defaultdict


def summarize_records(records: List[Dict]) -> Dict:
    by_model = defaultdict(list)
    for r in records:
        by_model[r["model_name"]].append(r)

    summary = {"models": {}}
    for model_name, recs in by_model.items():
        model_summary = _summarize(recs)
        summary["models"][model_name] = model_summary
    return summary


def _summarize(records: List[Dict]) -> Dict:
    def _agg(recs):
        total = len(recs)
        if total == 0:
            return {
                "total": 0,
                "refusal_rate": 0.0,
                "attack_success_rate": 0.0,
                "avg_latency_s": 0.0,
            }
        refusals = sum(1 for r in recs if r["refusal"])
        successes = sum(1 for r in recs if r["attack_success"])
        avg_latency = sum(r["latency_s"] for r in recs) / total
        return {
            "total": total,
            "refusal_rate": refusals / total,
            "attack_success_rate": successes / total,
            "avg_latency_s": avg_latency,
        }

    benign = [r for r in records if r["type"] == "benign"]
    injection = [r for r in records if r["type"] == "injection"]

    by_variant = {}
    variants = sorted({r.get("variant", "") for r in injection if r.get("variant")})
    for v in variants:
        by_variant[v] = _agg([r for r in injection if r.get("variant") == v])

    return {
        "overall": _agg(records),
        "benign": _agg(benign),
        "injection": _agg(injection),
        "injection_by_variant": by_variant,
    }
