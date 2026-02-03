"""
Deep Past Translation - Training Script
Fine-tunes ByT5-small for Akkadian to English translation.
"""

import os
import gc
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import evaluate

from preprocessing import load_train_data


class Config:
    """Training configuration."""
    # Model
    MODEL_NAME = "google/byt5-small"
    MAX_LENGTH = 512
    
    # Training
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 1
    GRADIENT_ACCUMULATION = 8
    LABEL_SMOOTHING = 0.2
    WEIGHT_DECAY = 0.01
    
    # Data
    USE_SENTENCE_ALIGN = True
    USE_BIDIRECTIONAL = True
    USE_EXTRA_SENTENCES = False
    VAL_SIZE = 0.1
    
    # Output
    OUTPUT_DIR = "./byt5-akkadian-model"
    
    @classmethod
    def from_yaml(cls, path: str):
        """Load config from YAML file."""
        with open(path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        obj = cls()
        for key, value in cfg.items():
            if hasattr(obj, key.upper()):
                setattr(obj, key.upper(), value)
        return obj


def preprocess_function(examples, tokenizer, max_length):
    """Tokenize examples for Seq2Seq training."""
    inputs = [str(ex) for ex in examples["input_text"]]
    targets = [str(ex) for ex in examples["target_text"]]
    
    model_inputs = tokenizer(
        inputs,
        max_length=max_length,
        truncation=True
    )
    
    labels = tokenizer(
        targets,
        max_length=max_length,
        truncation=True
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_preds, tokenizer):
    """Compute chrF and BLEU metrics."""
    preds, labels = eval_preds
    
    # Handle tuple output
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # Convert logits to token ids if needed
    if hasattr(preds, "ndim") and preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)
    
    # Ensure valid token ids
    preds = preds.astype(np.int64)
    preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
    preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
    
    # Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Decode labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Compute metrics
    metric_chrf = evaluate.load("chrf")
    metric_bleu = evaluate.load("sacrebleu")
    
    chrf = metric_chrf.compute(predictions=decoded_preds, references=decoded_labels)["score"]
    bleu = metric_bleu.compute(predictions=decoded_preds, references=[[x] for x in decoded_labels])["score"]
    
    # Geometric mean (competition metric)
    geo_mean = (chrf * bleu) ** 0.5 if chrf > 0 and bleu > 0 else 0.0
    
    return {"chrf": chrf, "bleu": bleu, "geo_mean": geo_mean}


def train(config: Config, data_dir: str):
    """Main training function."""
    print("=" * 60)
    print(f"Training with config:")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Epochs: {config.EPOCHS}")
    print(f"  Learning Rate: {config.LEARNING_RATE}")
    print(f"  Batch Size: {config.BATCH_SIZE} x {config.GRADIENT_ACCUMULATION} = {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION}")
    print("=" * 60)
    
    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Load and preprocess data
    train_df = load_train_data(
        data_dir,
        use_sentence_align=config.USE_SENTENCE_ALIGN,
        use_bidirectional=config.USE_BIDIRECTIONAL,
        use_extra_sentences=config.USE_EXTRA_SENTENCES
    )
    
    # Create dataset
    dataset = Dataset.from_pandas(train_df)
    
    # Split train/validation
    split = dataset.train_test_split(test_size=config.VAL_SIZE, seed=42)
    train_dataset = split["train"]
    val_dataset = split["test"]
    
    print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME)
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config.MAX_LENGTH),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer, config.MAX_LENGTH),
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.LEARNING_RATE,
        optim="adafactor",
        label_smoothing_factor=config.LABEL_SMOOTHING,
        fp16=False,  # Prevent NaN errors with ByT5
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION,
        weight_decay=config.WEIGHT_DECAY,
        save_total_limit=2,
        num_train_epochs=config.EPOCHS,
        predict_with_generate=True,
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="geo_mean",
        greater_is_better=True,
    )
    
    # Create trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tokenizer),
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    print(f"\nModel saved to {config.OUTPUT_DIR}")
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description="Train ByT5 for Akkadian translation")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--data_dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--output_dir", type=str, help="Override output directory")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config()
    
    # Override from command line
    if args.output_dir:
        config.OUTPUT_DIR = args.output_dir
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.lr:
        config.LEARNING_RATE = args.lr
    
    # Train
    train(config, args.data_dir)


if __name__ == "__main__":
    main()
