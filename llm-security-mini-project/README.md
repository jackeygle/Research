LLM Security Mini Project (Prompt Injection Evaluation)
========================================================

About
-----
This repository contains a small, safe prompt-injection evaluation suite for
open-source LLMs. It provides GPU-only inference scripts, metrics, and
visualizations to compare robustness and lightweight defenses in an
visualizations to compare robustness and lightweight defenses in a reproducible workflow.

Goal
----
Reproduce a safe evaluation of prompt-injection style attacks
against open-source LLMs, compare refusal/unsafe rates, and demonstrate a
lightweight defense layer. This project intentionally avoids harmful content
and uses only benign prompt-injection tests.

What this includes
------------------
- A small prompt suite (benign + safe injection attempts).
- An evaluation runner for two open-source LLMs using Hugging Face Transformers.
- A simple defense layer (input filter + output validator).
- A simple defense layer (input filter + output validator).

Quick start (GPU inference)
---------------------------
1) Run on a V100 16G node via Slurm:
   sbatch scripts/run_eval_v100.sh

2) Choose a config by overriding EVAL_CONFIG:
   - configs/eval_small_phi3.yaml (Qwen2 + TinyLlama + Phi-3)
   - configs/eval_small_phi3_sample.yaml (sampling decoding)
   - configs/eval_small_phi3_strong.yaml (strong system prompt)
   - configs/eval_small_phi3_expanded.yaml (expanded prompt suite)
   - configs/eval_phi3.yaml (Phi-3 only)
   - configs/eval.yaml (Llama 3 / Phi-3 if you have access)
   Example:
   sbatch --export=ALL,EVAL_CONFIG=configs/eval_small.yaml,DEFENSE=none scripts/run_eval_v100.sh

3) The outputs will appear in outputs/run_<timestamp>.jsonl and outputs/summary_<timestamp>.json

4) Run with a simple defense:
   sbatch --export=ALL,EVAL_CONFIG=configs/eval_small_phi3.yaml,DEFENSE=filter_prefix scripts/run_eval_v100.sh

5) Run the full experiment bundle (defense + decoding + prompt expansion):
   bash scripts/run_experiments_v100.sh

6) Visualize results (PNG + CSV):
   python scripts/plot_results.py

Notes for gated models
----------------------
Llama 3 is gated on Hugging Face. If you have access, submit with a token:
  sbatch --export=ALL,HF_TOKEN_OVERRIDE=YOUR_TOKEN,EVAL_CONFIG=configs/eval.yaml,DEFENSE=none scripts/run_eval_v100.sh

Outputs
-------
- Outputs are written to outputs/run_<timestamp>.jsonl
- A summary is written to outputs/summary_<timestamp>.json

Ethics and safety
----------------
This project deliberately avoids harmful prompts. Injection prompts are
benign (e.g., requesting a fixed harmless token) so you can measure instruction
hierarchy robustness without generating unsafe content.
