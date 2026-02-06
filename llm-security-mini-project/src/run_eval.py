import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import yaml

from src.data import load_prompts
from src.defense import apply_input_defense, apply_output_defense, is_refusal, target_match
from src.metrics import summarize_records
from src.modeling import build_prompt, generate_response, load_model_and_tokenizer


def generate_mock_response(mode: str, prompt_record: Dict) -> str:
    if mode == "naive":
        if prompt_record["type"] == "injection" and prompt_record.get("target"):
            return prompt_record["target"]
        return "Here is a brief, safe response."
    if mode == "refusal":
        if prompt_record["type"] == "injection":
            return "I cannot comply with that request."
        return "Here is a brief, safe response."
    if mode == "echo":
        return prompt_record.get("prompt", "")
    return "Here is a brief, safe response."


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_model(
    model_name: str,
    model_id: str,
    prompts: List[Dict],
    system_prompt: str,
    decoding: Dict,
    defense_mode: str,
) -> List[Dict]:
    records: List[Dict] = []
    is_mock = model_id.startswith("mock:")
    mock_mode = ""
    model = tokenizer = None
    is_seq2seq = False
    if is_mock:
        mock_mode = model_id.split(":", 1)[1]
    else:
        model, tokenizer, is_seq2seq = load_model_and_tokenizer(model_id)

    for p in prompts:
        sys_prompt, user_prompt = apply_input_defense(system_prompt, p["prompt"], defense_mode)
        start = time.perf_counter()
        if is_mock:
            response = generate_mock_response(mock_mode, p)
        else:
            full_prompt = build_prompt(sys_prompt, user_prompt, tokenizer)
            response = generate_response(model, tokenizer, full_prompt, is_seq2seq, decoding)
        latency = time.perf_counter() - start

        response = apply_output_defense(
            response,
            is_injection=(p["type"] == "injection"),
            target=p.get("target", ""),
            mode=defense_mode,
        )

        refusal = is_refusal(response)
        attack_success = False
        if p["type"] == "injection":
            attack_success = target_match(response, p.get("target", ""))

        record = {
            "model_name": model_name,
            "model_id": model_id,
            "prompt_id": p["id"],
            "type": p["type"],
            "variant": p.get("variant", ""),
            "category": p.get("category", ""),
            "prompt": p["prompt"],
            "target": p.get("target", ""),
            "response": response,
            "refusal": refusal,
            "attack_success": attack_success,
            "latency_s": latency,
            "defense": defense_mode,
            "decoding": decoding,
        }
        records.append(record)
    return records


def write_jsonl(path: Path, records: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--defense", default="none", choices=["none", "filter_prefix", "output_guard", "combined"])
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--allow_cpu", action="store_true", default=False)
    args = parser.parse_args()

    config = load_config(args.config)
    models = config["models"]
    decoding = config.get("decoding", {})
    system_prompt = config["evaluation"]["system_prompt"]
    prompt_files = config["evaluation"]["prompt_files"]

    prompts = load_prompts(prompt_files)

    # Enforce GPU for real models unless explicitly allowed
    if not args.allow_cpu:
        requires_gpu = any(not m["id"].startswith("mock:") for m in models)
        if requires_gpu:
            try:
                import torch
            except Exception as exc:
                raise RuntimeError("Torch is required for GPU inference.") from exc
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for this run. Use --allow_cpu to override.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = os.environ.get("EXPERIMENT_TAG", "").strip()
    tag_prefix = f"{tag}_" if tag else ""
    run_path = out_dir / f"run_{tag_prefix}{ts}.jsonl"
    summary_path = out_dir / f"summary_{tag_prefix}{ts}.json"

    all_records: List[Dict] = []
    for m in models:
        records = evaluate_model(
            model_name=m["name"],
            model_id=m["id"],
            prompts=prompts,
            system_prompt=system_prompt,
            decoding=decoding,
            defense_mode=args.defense,
        )
        all_records.extend(records)

    write_jsonl(run_path, all_records)
    summary = summarize_records(all_records)
    summary["defense"] = args.defense
    summary["run_file"] = str(run_path)
    summary["config"] = args.config
    if tag:
        summary["experiment_tag"] = tag

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {run_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
