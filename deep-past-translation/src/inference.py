"""
Deep Past Translation - Inference
Generate translations from trained model.
"""

import os
import torch
import pandas as pd
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset, DataLoader

from preprocessing import load_test_data, normalize_text


def build_memory_map(data_dir: str) -> dict:
    """Build a lookup of normalized transliteration -> translation."""
    train_path = Path(data_dir) / "train.csv"
    if not train_path.exists():
        return {}
    train_df = pd.read_csv(train_path)
    train_df["key"] = train_df["transliteration"].apply(normalize_text)
    # Use most frequent translation per key
    grouped = train_df.groupby("key")["translation"].agg(lambda x: x.value_counts().index[0])
    return grouped.to_dict()
from postprocess import postprocess_batch, create_submission


class TranslationDataset(Dataset):
    """Dataset for inference."""
    
    def __init__(self, texts: List[str]):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]


class TranslationModel:
    """Wrapper for translation inference."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize translation model.
        
        Args:
            model_path: Path to saved model directory
            device: Device to use (cuda/cpu)
            max_length: Maximum sequence length
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model = self.model.to(self.device).eval()
        
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {num_params:,} parameters on {self.device}")
    
    def translate(
        self,
        text: str,
        num_beams: int = 8,
        max_new_tokens: int = 512,
        min_new_tokens: int = 0,
        length_penalty: float = 1.1,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        adaptive_beams: bool = False,
        adaptive_min_beams: int = 4,
        adaptive_max_beams: int = 12,
        adaptive_length_threshold: int = 256,
        **kwargs
    ) -> str:
        """Translate a single text."""
        # Add prefix
        input_text = f"translate Akkadian to English: {normalize_text(text)}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            beam_size = num_beams
            if adaptive_beams:
                seq_len = int(inputs["attention_mask"].sum().item())
                beam_size = adaptive_max_beams if seq_len >= adaptive_length_threshold else adaptive_min_beams
            outputs = self.model.generate(
                **inputs,
                num_beams=beam_size,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                early_stopping=True,
                **kwargs
            )
        
        # Decode
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation
    
    def translate_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        num_beams: int = 8,
        max_new_tokens: int = 512,
        min_new_tokens: int = 0,
        length_penalty: float = 1.1,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        adaptive_beams: bool = False,
        adaptive_min_beams: int = 4,
        adaptive_max_beams: int = 12,
        adaptive_length_threshold: int = 256,
        show_progress: bool = True,
        **kwargs
    ) -> List[str]:
        """Translate a batch of texts."""
        # Prepare inputs
        input_texts = [f"translate Akkadian to English: {normalize_text(t)}" for t in texts]
        
        translations = []
        iterator = range(0, len(input_texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Translating")
        
        for i in iterator:
            batch = input_texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                beam_size = num_beams
                if adaptive_beams:
                    max_len = int(inputs["attention_mask"].sum(dim=1).max().item())
                    beam_size = adaptive_max_beams if max_len >= adaptive_length_threshold else adaptive_min_beams
                outputs = self.model.generate(
                    **inputs,
                    num_beams=beam_size,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    early_stopping=True,
                    **kwargs
                )
            
            # Decode
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translations.extend(decoded)
        
        return translations


def run_inference(
    model_path: str,
    data_dir: str,
    output_path: str,
    batch_size: int = 8,
    num_beams: int = 8,
    max_new_tokens: int = 512,
    min_new_tokens: int = 0,
    length_penalty: float = 1.1,
    no_repeat_ngram_size: int = 0,
    repetition_penalty: float = 1.0,
    do_sample: bool = False,
    top_p: float = 1.0,
    temperature: float = 1.0,
    adaptive_beams: bool = False,
    adaptive_min_beams: int = 4,
    adaptive_max_beams: int = 12,
    adaptive_length_threshold: int = 256,
    use_memory_map: bool = False,
    empty_fallback: str = "broken text",
    postprocess: bool = True,
    postprocess_aggressive: bool = True
):
    """
    Run inference on test data and create submission.
    
    Args:
        model_path: Path to trained model
        data_dir: Path to data directory
        output_path: Path to save submission
        batch_size: Inference batch size
        num_beams: Number of beams for generation
        postprocess: Apply postprocessing
    """
    # Load test data
    test_df = load_test_data(data_dir)
    print(f"Loaded {len(test_df)} test samples")
    
    # Initialize model
    model = TranslationModel(model_path)
    
    # Generate translations
    translations = model.translate_batch(
        test_df['transliteration'].tolist(),
        batch_size=batch_size,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
        top_p=top_p,
        temperature=temperature,
        adaptive_beams=adaptive_beams,
        adaptive_min_beams=adaptive_min_beams,
        adaptive_max_beams=adaptive_max_beams,
        adaptive_length_threshold=adaptive_length_threshold
    )
    
    if use_memory_map:
        memory_map = build_memory_map(data_dir)
        if memory_map:
            keys = test_df["transliteration"].apply(normalize_text).tolist()
            translations = [
                memory_map.get(k, t) for k, t in zip(keys, translations)
            ]

    # Postprocess and save
    if postprocess:
        translations = postprocess_batch(translations, aggressive=postprocess_aggressive)

    if empty_fallback:
        translations = [t if str(t).strip() else empty_fallback for t in translations]
    
    create_submission(test_df, translations, output_path)
    
    # Print samples
    print("\nSample translations:")
    for i in range(min(3, len(translations))):
        print(f"[{test_df.iloc[i]['id']}] {translations[i][:100]}...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output", type=str, default="submission.csv", help="Output file")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_beams", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--min_new_tokens", type=int, default=0)
    parser.add_argument("--length_penalty", type=float, default=1.1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--adaptive_beams", action="store_true")
    parser.add_argument("--adaptive_min_beams", type=int, default=4)
    parser.add_argument("--adaptive_max_beams", type=int, default=12)
    parser.add_argument("--adaptive_length_threshold", type=int, default=256)
    parser.add_argument("--use_memory_map", action="store_true")
    parser.add_argument("--empty_fallback", type=str, default="broken text")
    parser.add_argument("--postprocess", action="store_true", default=True)
    parser.add_argument("--no_postprocess", dest="postprocess", action="store_false")
    parser.add_argument("--postprocess_aggressive", action="store_true", default=True)
    parser.add_argument("--postprocess_light", dest="postprocess_aggressive", action="store_false")
    args = parser.parse_args()
    
    run_inference(
        args.model,
        args.data_dir,
        args.output,
        args.batch_size,
        args.num_beams,
        args.max_new_tokens,
        args.min_new_tokens,
        args.length_penalty,
        args.no_repeat_ngram_size,
        args.repetition_penalty,
        args.do_sample,
        args.top_p,
        args.temperature,
        args.adaptive_beams,
        args.adaptive_min_beams,
        args.adaptive_max_beams,
        args.adaptive_length_threshold,
        args.use_memory_map,
        args.empty_fallback,
        args.postprocess,
        args.postprocess_aggressive
    )
