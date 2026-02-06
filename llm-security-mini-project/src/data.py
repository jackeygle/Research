import csv
from pathlib import Path
from typing import Dict, List


def _load_csv(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def load_prompts(prompt_files: List[str]) -> List[Dict]:
    prompts: List[Dict] = []
    for file_path in prompt_files:
        path = Path(file_path)
        rows = _load_csv(path)
        for row in rows:
            record = {
                "id": row.get("id", "").strip(),
                "prompt": row.get("prompt", "").strip(),
                "source": path.name,
            }
            target = row.get("target", "").strip()
            if target:
                record["type"] = "injection"
                record["variant"] = row.get("variant", "").strip()
                record["target"] = target
            else:
                record["type"] = "benign"
                record["category"] = row.get("category", "").strip()
                record["target"] = ""
            prompts.append(record)
    return prompts
