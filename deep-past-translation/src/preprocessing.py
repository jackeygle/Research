"""
Deep Past Translation - Data Preprocessing
Handles sentence alignment, bidirectional training, and text normalization.
"""

import re
import unicodedata
import pandas as pd
from typing import List, Tuple, Dict
from pathlib import Path


def normalize_text(text: str) -> str:
    """
    Normalize Akkadian transliteration text.
    - Remove subscript numbers (u₂ → u)
    - Normalize Unicode characters
    - Standardize GAP markers
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = str(text).strip()
    
    # Remove subscript digits (e.g., u₂ → u)
    subscript_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    text = text.translate(subscript_map)
    
    # Normalize Unicode (handle š, ṣ, ḫ etc.)
    text = unicodedata.normalize('NFKD', text)
    
    # Standardize GAP markers
    text = re.sub(r'\.{3,}|…+', ' <big_gap> ', text)
    text = re.sub(r'xx+|\s+x\s+', ' <gap> ', text, flags=re.IGNORECASE)
    
    # Remove editorial brackets but keep content
    text = re.sub(r'[\[\]<>⌈⌋⌊]', '', text)
    
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def sentence_align(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split multi-sentence documents into single sentence pairs.
    Doubles the training data when alignment is successful.
    
    Strategy:
    - Split English by sentence-ending punctuation
    - Split Akkadian by newlines
    - If counts match, create 1-to-1 pairs
    """
    aligned_data = []
    
    for _, row in df.iterrows():
        src = str(row.get('transliteration', ''))
        tgt = str(row.get('translation', ''))
        
        # Split English by sentence endings
        tgt_sents = [t.strip() for t in re.split(r'(?<=[.!?])\s+', tgt) if t.strip()]
        
        # Split Akkadian by newlines
        src_lines = [s.strip() for s in src.split('\n') if s.strip()]
        
        # If counts match and we have multiple sentences, use aligned pairs
        if len(tgt_sents) > 1 and len(tgt_sents) == len(src_lines):
            for s, t in zip(src_lines, tgt_sents):
                if len(s) > 3 and len(t) > 3:  # Filter very short/noisy data
                    aligned_data.append({
                        'transliteration': normalize_text(s),
                        'translation': t.strip()
                    })
        else:
            # Keep original document pair
            aligned_data.append({
                'transliteration': normalize_text(src),
                'translation': tgt.strip()
            })
    
    return pd.DataFrame(aligned_data)


def create_bidirectional_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create bidirectional training data.
    - Forward: Akkadian → English
    - Backward: English → Akkadian
    Doubles the training data.
    """
    forward = df.copy()
    forward['input_text'] = "translate Akkadian to English: " + forward['transliteration'].astype(str)
    forward['target_text'] = forward['translation'].astype(str)
    forward['direction'] = 'forward'
    
    backward = df.copy()
    backward['input_text'] = "translate English to Akkadian: " + backward['translation'].astype(str)
    backward['target_text'] = backward['transliteration'].astype(str)
    backward['direction'] = 'backward'
    
    combined = pd.concat([forward, backward], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return combined


def load_train_data(
    data_dir: str,
    use_sentence_align: bool = True,
    use_bidirectional: bool = True,
    use_extra_sentences: bool = False
) -> pd.DataFrame:
    """
    Load and preprocess training data.
    
    Args:
        data_dir: Path to data directory
        use_sentence_align: Apply sentence alignment augmentation
        use_bidirectional: Create bidirectional training pairs
        use_extra_sentences: Include OARE sentence data
    
    Returns:
        Preprocessed DataFrame with 'input_text' and 'target_text' columns
    """
    data_path = Path(data_dir)
    
    # Load main training data
    train_df = pd.read_csv(data_path / 'train.csv')
    print(f"Original train data: {len(train_df)} samples")
    
    # Apply sentence alignment
    if use_sentence_align:
        train_df = sentence_align(train_df)
        print(f"After sentence alignment: {len(train_df)} samples")
    else:
        train_df['transliteration'] = train_df['transliteration'].apply(normalize_text)
    
    # Add extra OARE sentence data
    if use_extra_sentences:
        oare_path = data_path / 'Sentences_Oare_FirstWord_LinNum.csv'
        if oare_path.exists():
            oare_df = pd.read_csv(oare_path)
            # Rename columns to match training format
            if 'transliteration' not in oare_df.columns:
                # Use first_word columns or available data
                extra = oare_df[['translation']].copy()
                extra['transliteration'] = oare_df.get('first_word_transcription', '')
                extra = extra.dropna()
                extra = extra[extra['transliteration'].str.len() > 3]
                extra = extra[extra['translation'].str.len() > 3]
                train_df = pd.concat([train_df, extra], ignore_index=True)
                print(f"After adding OARE data: {len(train_df)} samples")
    
    # Create training format
    if use_bidirectional:
        train_df = create_bidirectional_data(train_df)
        print(f"After bidirectional: {len(train_df)} samples")
    else:
        train_df['input_text'] = "translate Akkadian to English: " + train_df['transliteration'].astype(str)
        train_df['target_text'] = train_df['translation'].astype(str)
    
    return train_df[['input_text', 'target_text']]


def load_test_data(data_dir: str) -> pd.DataFrame:
    """Load and preprocess test data."""
    data_path = Path(data_dir)
    test_df = pd.read_csv(data_path / 'test.csv')
    
    # Normalize transliteration
    test_df['transliteration'] = test_df['transliteration'].apply(normalize_text)
    test_df['input_text'] = "translate Akkadian to English: " + test_df['transliteration'].astype(str)
    
    return test_df


if __name__ == "__main__":
    # Test the preprocessing
    data_dir = "/scratch/work/zhangx29/deep-past-translation/data"
    
    print("=" * 60)
    print("Testing data preprocessing...")
    print("=" * 60)
    
    # Test with all augmentations
    train_df = load_train_data(
        data_dir,
        use_sentence_align=True,
        use_bidirectional=True,
        use_extra_sentences=False
    )
    
    print(f"\nFinal training samples: {len(train_df)}")
    print("\nSample data:")
    print(train_df.head(3).to_string())
    
    # Test loading test data
    test_df = load_test_data(data_dir)
    print(f"\nTest samples: {len(test_df)}")
