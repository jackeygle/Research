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
        length_penalty: float = 1.1,
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
            outputs = self.model.generate(
                **inputs,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                length_penalty=length_penalty,
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
        length_penalty: float = 1.1,
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
                outputs = self.model.generate(
                    **inputs,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    length_penalty=length_penalty,
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
    postprocess: bool = True
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
        num_beams=num_beams
    )
    
    # Postprocess and save
    if postprocess:
        translations = postprocess_batch(translations)
    
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
    args = parser.parse_args()
    
    run_inference(
        args.model,
        args.data_dir,
        args.output,
        args.batch_size,
        args.num_beams
    )
