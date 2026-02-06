"""
Deep Past Translation - Model Ensemble
Combine multiple models using model soup or output blending.
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from preprocessing import load_test_data, normalize_text
from postprocess import postprocess_batch, create_submission


def build_memory_map(data_dir: str) -> dict:
    """Build a lookup of normalized transliteration -> translation."""
    train_path = Path(data_dir) / "train.csv"
    if not train_path.exists():
        return {}
    train_df = pd.read_csv(train_path)
    train_df["key"] = train_df["transliteration"].apply(normalize_text)
    grouped = train_df.groupby("key")["translation"].agg(lambda x: x.value_counts().index[0])
    return grouped.to_dict()


def load_model_state_dict(model_path: str) -> Dict:
    """Load model state dict from path."""
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    return model.state_dict()


def create_model_soup(
    model_paths: List[str],
    weights: Optional[List[float]] = None
) -> Dict:
    """
    Create model soup by averaging model weights.
    
    Args:
        model_paths: List of paths to model directories
        weights: Optional weights for each model (default: equal weights)
    
    Returns:
        Averaged state dict
    """
    if weights is None:
        weights = [1.0] * len(model_paths)
    
    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]
    
    print(f"Creating model soup from {len(model_paths)} models...")
    print(f"Weights: {weights}")
    
    # Load all state dicts
    state_dicts = []
    for path in tqdm(model_paths, desc="Loading models"):
        sd = load_model_state_dict(path)
        state_dicts.append(sd)
    
    # Average weights
    soup_sd = {}
    for key in tqdm(state_dicts[0].keys(), desc="Averaging weights"):
        weighted_sum = torch.zeros_like(state_dicts[0][key], dtype=torch.float32)
        weight_sum = 0.0
        
        for sd, weight in zip(state_dicts, weights):
            if key in sd:
                weighted_sum += weight * sd[key].float()
                weight_sum += weight
        
        soup_sd[key] = (weighted_sum / weight_sum).to(state_dicts[0][key].dtype)
    
    return soup_sd


def save_model_soup(
    model_paths: List[str],
    output_path: str,
    weights: Optional[List[float]] = None
):
    """
    Create and save model soup.
    
    Args:
        model_paths: List of model directories
        output_path: Where to save the soup model
        weights: Optional weights
    """
    # Create soup
    soup_sd = create_model_soup(model_paths, weights)
    
    # Load template model
    template_model = AutoModelForSeq2SeqLM.from_pretrained(model_paths[0])
    template_model.load_state_dict(soup_sd)
    
    # Save
    os.makedirs(output_path, exist_ok=True)
    template_model.save_pretrained(output_path)
    
    # Also save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_paths[0])
    tokenizer.save_pretrained(output_path)
    
    print(f"Model soup saved to {output_path}")


class EnsembleTranslator:
    """Ensemble multiple models for translation."""
    
    def __init__(
        self,
        model_paths: List[str],
        weights: Optional[List[float]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize ensemble.
        
        Args:
            model_paths: List of model directories
            weights: Weights for output blending
            device: Device to use
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = weights or [1.0] * len(model_paths)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        
        # Load models
        print(f"Loading {len(model_paths)} models for ensemble...")
        self.models = []
        self.tokenizers = []
        
        for path in tqdm(model_paths, desc="Loading"):
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSeq2SeqLM.from_pretrained(path)
            model = model.to(self.device).eval()
            
            self.models.append(model)
            self.tokenizers.append(tokenizer)
        
        print(f"Ensemble ready with {len(self.models)} models")
    
    def translate_single(
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
        strategy: str = "vote"
    ) -> str:
        """
        Translate using ensemble.
        
        Args:
            text: Input text
            num_beams: Beams for generation
            max_new_tokens: Max tokens to generate
            strategy: 'vote' or 'first'
        
        Returns:
            Best translation
        """
        input_text = f"translate Akkadian to English: {normalize_text(text)}"
        
        translations = []
        for model, tokenizer in zip(self.models, self.tokenizers):
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    num_beams=num_beams,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    length_penalty=length_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    repetition_penalty=repetition_penalty,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    early_stopping=True
                )
            
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translations.append(translation)
        
        # Select best
        if strategy == "vote":
            # Simple majority voting (works best when translations differ slightly)
            counter = Counter(translations)
            return counter.most_common(1)[0][0]
        else:
            # Return first model's output
            return translations[0]
    
    def translate_batch(
        self,
        texts: List[str],
        batch_size: int = 4,
        num_beams: int = 8,
        max_new_tokens: int = 512,
        min_new_tokens: int = 0,
        length_penalty: float = 1.1,
        no_repeat_ngram_size: int = 0,
        repetition_penalty: float = 1.0,
        do_sample: bool = False,
        top_p: float = 1.0,
        temperature: float = 1.0,
        strategy: str = "vote"
    ) -> List[str]:
        """Translate batch using ensemble."""
        results = []
        for text in tqdm(texts, desc="Translating"):
            result = self.translate_single(
                text,
                num_beams=num_beams,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                strategy=strategy
            )
            results.append(result)
        return results


def run_ensemble_inference(
    model_paths: List[str],
    data_dir: str,
    output_path: str,
    weights: Optional[List[float]] = None,
    use_soup: bool = True,
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
    postprocess_aggressive: bool = True,
    use_memory_map: bool = False,
    empty_fallback: str = "broken text"
):
    """
    Run ensemble inference.
    
    Args:
        model_paths: List of trained model paths
        data_dir: Data directory
        output_path: Submission output path
        weights: Model weights
        use_soup: Use model soup (faster) or output blending
        batch_size: Batch size
        num_beams: Number of beams
    """
    from inference import TranslationModel
    
    # Load test data
    test_df = load_test_data(data_dir)
    print(f"Loaded {len(test_df)} test samples")
    
    if use_soup:
        # Create soup and use single model inference
        soup_path = "./model_soup"
        save_model_soup(model_paths, soup_path, weights)
        
        model = TranslationModel(soup_path)
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
            temperature=temperature
        )
    else:
        # Use ensemble voting
        ensemble = EnsembleTranslator(model_paths, weights)
        translations = ensemble.translate_batch(
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
            temperature=temperature
        )
    
    # Postprocess and save
    translations = postprocess_batch(translations, aggressive=postprocess_aggressive)
    if use_memory_map:
        memory_map = build_memory_map(data_dir)
        if memory_map:
            keys = test_df["transliteration"].apply(normalize_text).tolist()
            translations = [
                memory_map.get(k, t) for k, t in zip(keys, translations)
            ]
    if empty_fallback:
        translations = [t if str(t).strip() else empty_fallback for t in translations]
    create_submission(test_df, translations, output_path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ensemble inference")
    parser.add_argument("--models", nargs="+", required=True, help="Model paths")
    parser.add_argument("--weights", nargs="+", type=float, help="Model weights")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--use_soup", action="store_true", help="Use model soup")
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
    parser.add_argument("--postprocess_aggressive", action="store_true", default=True)
    parser.add_argument("--postprocess_light", dest="postprocess_aggressive", action="store_false")
    parser.add_argument("--use_memory_map", action="store_true")
    parser.add_argument("--empty_fallback", type=str, default="broken text")
    args = parser.parse_args()
    
    run_ensemble_inference(
        args.models,
        args.data_dir,
        args.output,
        args.weights,
        args.use_soup,
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
        args.postprocess_aggressive,
        args.use_memory_map,
        args.empty_fallback
    )
