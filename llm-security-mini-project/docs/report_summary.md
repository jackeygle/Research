LLM Prompt-Injection Mini Project Report (Summary)
===================================================

Summary
-------
- Models: Qwen2-0.5B-Instruct, TinyLlama-1.1B-Chat
- Prompt suite: 6 benign + 6 benign injection prompts
- Defense tested: filter_prefix (input prefix)
- Key finding: naive prefix defense did not reduce injection success and added latency.

Setup
-----
- Inference: Hugging Face Transformers on V100 16G
- Decoding: max_new_tokens=96, temperature=0.2, top_p=0.9, do_sample=false
- Safety: Only benign injection targets (fixed tokens/strings)

Baseline results (defense = none)
---------------------------------
Qwen2-0.5B
- Injection success rate: 0.33 (2/6)
- Refusal rate (benign): 0.00
- Avg latency: 0.82 s
- Successful variants: override, roleplay

TinyLlama-1.1B
- Injection success rate: 0.33 (2/6)
- Refusal rate (benign): 0.00
- Avg latency: 1.56 s
- Successful variants: delimiter, override

Defense results (filter_prefix)
-------------------------------
Qwen2-0.5B
- Injection success rate: 0.50 (3/6)  (worse)
- Avg latency: 0.98 s

TinyLlama-1.1B
- Injection success rate: 0.33 (2/6)  (unchanged)
- Avg latency: 2.11 s

Interpretation
--------------
- Simple prefix defenses can be fragile; they may prime the model to follow the injected instruction.
- Small models showed low refusal rates and non-trivial injection success despite benign prompts.
- Defense overhead is measurable even with small prompt sets.

Artifacts
---------
- Baseline summary: outputs/summary_20260205_011844.json
- Defense summary: outputs/summary_20260205_012015.json
