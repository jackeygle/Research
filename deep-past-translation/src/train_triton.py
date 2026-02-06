"""
Deep Past Translation - Triton Training Script
Standalone training script for SLURM cluster.
"""

import os
import gc
import re
import argparse
import unicodedata
import inspect
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate


def normalize_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = str(text).strip()
    text = text.translate(str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'\.{3,}|…+', ' <big_gap> ', text)
    text = re.sub(r'xx+|\s+x\s+', ' <gap> ', text, flags=re.IGNORECASE)
    text = re.sub(r'[\[\]<>⌈⌋⌊]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def sentence_align(df):
    aligned = []
    for _, row in df.iterrows():
        src = str(row.get('transliteration', ''))
        tgt = str(row.get('translation', ''))
        tgt_sents = [t.strip() for t in re.split(r'(?<=[.!?])\s+', tgt) if t.strip()]
        src_lines = [s.strip() for s in src.split('\n') if s.strip()]
        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3:
                    aligned.append({'transliteration': normalize_text(s), 'translation': t.strip()})
        else:
            aligned.append({'transliteration': normalize_text(src), 'translation': tgt.strip()})
    return pd.DataFrame(aligned)


def create_bidirectional(df):
    fwd = df.copy()
    fwd['input_text'] = "translate Akkadian to English: " + fwd['transliteration'].astype(str)
    fwd['target_text'] = fwd['translation'].astype(str)
    bwd = df.copy()
    bwd['input_text'] = "translate English to Akkadian: " + bwd['translation'].astype(str)
    bwd['target_text'] = bwd['transliteration'].astype(str)
    return pd.concat([fwd, bwd], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)


def load_data(data_dir, use_bidir=False):
    data_path = Path(data_dir)
    df = pd.read_csv(data_path / 'train.csv')
    df = sentence_align(df)
    print(f"After alignment: {len(df)} samples")
    
    if use_bidir:
        df = create_bidirectional(df)
        print(f"After bidirectional: {len(df)} samples")
    else:
        df['input_text'] = "translate Akkadian to English: " + df['transliteration'].astype(str)
        df['target_text'] = df['translation'].astype(str)
    
    return df[['input_text', 'target_text']]


def train(args):
    print(f"\n{'='*60}")
    print(f"Training: {args.output_dir}")
    print(f"Model: {args.model_name}")
    print(f"Epochs: {args.epochs}, LR: {args.learning_rate}")
    print(f"Batch: {args.batch_size} x {args.gradient_accumulation}")
    print(
        f"Dataloader: workers={args.num_workers}, pin_memory={args.pin_memory}, "
        f"persistent_workers={args.persistent_workers}"
    )
    print(f"{'='*60}\n")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load data
    use_bidir = args.bidirectional.lower() == 'true'
    df = load_data(args.data_dir, use_bidir=use_bidir)
    dataset = Dataset.from_pandas(df)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    
    # Tokenize
    def preprocess(examples):
        inputs = [str(x) for x in examples['input_text']]
        targets = [str(x) for x in examples['target_text']]
        model_inputs = tokenizer(inputs, max_length=args.max_length, truncation=True)
        labels = tokenizer(targets, max_length=args.max_length, truncation=True)
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    train_tok = split['train'].map(preprocess, batched=True, remove_columns=split['train'].column_names)
    val_tok = split['test'].map(preprocess, batched=True, remove_columns=split['test'].column_names)
    
    # Metrics
    metric_chrf = evaluate.load('chrf')
    metric_bleu = evaluate.load('sacrebleu')
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        if hasattr(preds, 'ndim') and preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)
        preds = np.clip(preds.astype(np.int64), 0, tokenizer.vocab_size - 1)
        dec_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        dec_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        chrf = metric_chrf.compute(predictions=dec_preds, references=dec_labels)['score']
        bleu = metric_bleu.compute(predictions=dec_preds, references=[[x] for x in dec_labels])['score']
        geo_mean = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0
        return {'chrf': chrf, 'bleu': bleu, 'geo_mean': geo_mean}
    
    # Training arguments
    training_args_kwargs = {
        "output_dir": args.output_dir,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "learning_rate": float(args.learning_rate),
        "optim": "adafactor",
        "label_smoothing_factor": 0.2,
        "fp16": False,  # Prevent NaN with ByT5
        "per_device_train_batch_size": int(args.batch_size),
        "per_device_eval_batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation),
        "weight_decay": 0.01,
        "save_total_limit": 1,
        "num_train_epochs": int(args.epochs),
        "predict_with_generate": True,
        "logging_steps": 50,
        "report_to": "none",
        "load_best_model_at_end": True,
        "metric_for_best_model": "geo_mean",
        "greater_is_better": True,
    }

    ta_sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
    if "dataloader_num_workers" in ta_sig.parameters:
        training_args_kwargs["dataloader_num_workers"] = int(args.num_workers)
    if "dataloader_pin_memory" in ta_sig.parameters:
        training_args_kwargs["dataloader_pin_memory"] = bool(args.pin_memory)
    if "dataloader_persistent_workers" in ta_sig.parameters:
        training_args_kwargs["dataloader_persistent_workers"] = bool(
            args.persistent_workers and int(args.num_workers) > 0
        )
    if "dataloader_prefetch_factor" in ta_sig.parameters and int(args.num_workers) > 0:
        training_args_kwargs["dataloader_prefetch_factor"] = int(args.prefetch_factor)

    training_args = Seq2SeqTrainingArguments(**training_args_kwargs)
    
    # Trainer (handle transformers v4/v5 API differences)
    trainer_kwargs = {}
    sig = inspect.signature(Seq2SeqTrainer.__init__)
    if 'processing_class' in sig.parameters:
        trainer_kwargs['processing_class'] = tokenizer
    elif 'tokenizer' in sig.parameters:
        trainer_kwargs['tokenizer'] = tokenizer

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
        **trainer_kwargs,
    )
    
    # Train
    resume_path = args.resume_from if args.resume_from else None
    trainer.train(resume_from_checkpoint=resume_path)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"\nModel saved to {args.output_dir}")
    
    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=str, default='1e-4')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation', type=int, default=8)
    parser.add_argument('--bidirectional', type=str, default='false')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--prefetch_factor', type=int, default=2)
    parser.add_argument('--pin_memory', action='store_true', default=True)
    parser.add_argument('--no_pin_memory', dest='pin_memory', action='store_false')
    parser.add_argument('--persistent_workers', action='store_true', default=True)
    parser.add_argument('--no_persistent_workers', dest='persistent_workers', action='store_false')
    args = parser.parse_args()
    
    train(args)
