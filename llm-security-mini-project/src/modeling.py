from typing import Dict, Tuple, Any


def load_model_and_tokenizer(model_id: str) -> Tuple[Any, Any, bool]:
    import torch
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        AutoModelForCausalLM,
        AutoModelForSeq2SeqLM,
    )

    config = AutoConfig.from_pretrained(model_id)
    is_seq2seq = bool(getattr(config, "is_encoder_decoder", False))
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    try:
        if is_seq2seq:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    except Exception:
        if is_seq2seq:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_id)

    model.to(device)
    model.eval()
    return model, tokenizer, is_seq2seq


def build_prompt(system_prompt: str, user_prompt: str, tokenizer) -> str:
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{system_prompt}\nUser: {user_prompt}\nAssistant:"


def generate_response(
    model,
    tokenizer,
    prompt: str,
    is_seq2seq: bool,
    decoding: Dict,
) -> str:
    import torch
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": int(decoding.get("max_new_tokens", 128)),
        "do_sample": bool(decoding.get("do_sample", False)),
        "top_p": float(decoding.get("top_p", 1.0)),
        "temperature": float(decoding.get("temperature", 0.2)),
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.inference_mode():
        outputs = model.generate(**inputs, **gen_kwargs)

    if is_seq2seq:
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decoded.strip()

    input_len = inputs["input_ids"].shape[-1]
    decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    return decoded.strip()
