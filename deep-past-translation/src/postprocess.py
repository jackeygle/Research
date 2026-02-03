"""
Deep Past Translation - Postprocessing
Cleans and normalizes model output for optimal BLEU/chrF scores.
"""

import re
import pandas as pd
from typing import List


# Character translation tables
SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
SPECIAL_CHARS_MAP = str.maketrans("ḫḪ", "hH")
# Note: Multi-byte Unicode chars handled in function, not maketrans
DASH_CHARS = {"–": "-", "—": "-", "−": "-"}
QUOTE_CHARS = {""": '"', """: '"', "'": "'", "'": "'"}
FORBIDDEN_CHARS = '!?()\"—–<>⌈⌋⌊[]+ʾ/;'


def postprocess_translation(text: str, aggressive: bool = True) -> str:
    """
    Postprocess a single translation for optimal metrics.
    
    Args:
        text: Raw model output
        aggressive: Apply aggressive cleaning (bracket removal, etc.)
    
    Returns:
        Cleaned translation text
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    text = str(text).strip()
    
    # 1. Subscript numbers → regular digits
    text = text.translate(SUBSCRIPT_MAP)
    
    # 2. Special characters normalization
    text = text.translate(SPECIAL_CHARS_MAP)
    # Handle multi-byte Unicode chars manually
    for old, new in DASH_CHARS.items():
        text = text.replace(old, new)
    for old, new in QUOTE_CHARS.items():
        text = text.replace(old, new)
    
    # 3. Handle GAP markers
    text = re.sub(r'(\[x\]|\(x\)|\bx\b)', '<gap>', text, flags=re.IGNORECASE)
    text = re.sub(r'(\.{3,}|…|\[\.+\])', '<big_gap>', text)
    text = re.sub(r'<gap>\s*<gap>', ' <big_gap> ', text)
    text = re.sub(r'<big_gap>\s*<big_gap>', ' <big_gap> ', text)
    
    # 4. Remove annotations like (fem), (plur), etc.
    text = re.sub(r'\((fem|plur|pl|sing|singular|plural|\?|!)\.?\s*\w*\)', '', text, flags=re.IGNORECASE)
    
    if aggressive:
        # 5. Remove forbidden characters
        text = text.translate(str.maketrans('', '', FORBIDDEN_CHARS))
        
        # 6. Remove GAP tokens for final output
        text = text.replace('<gap>', '...').replace('<big_gap>', '...')
    
    # 7. Handle fractions
    frac_replacements = [
        (r'(\d+)\.5\b', r'\1 ½'),
        (r'(\d+)\.25\b', r'\1 ¼'),
        (r'(\d+)\.75\b', r'\1 ¾'),
        (r'(\d+)\.33+\d*\b', r'\1 ⅓'),
        (r'(\d+)\.66+\d*\b', r'\1 ⅔'),
    ]
    for pattern, replacement in frac_replacements:
        text = re.sub(pattern, replacement, text)
    
    # 8. Remove repeated words
    text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text)
    
    # 9. Fix spacing around punctuation
    text = re.sub(r'\s+([.,:;!?])', r'\1', text)
    text = re.sub(r'([.,:])([A-Za-z])', r'\1 \2', text)
    
    # 10. Fix brackets if unbalanced
    if text.count('[') > text.count(']'):
        text += ']' * (text.count('[') - text.count(']'))
    if text.count('(') > text.count(')'):
        text += ')' * (text.count('(') - text.count(')'))
    
    # 11. Clean multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 12. Capitalize first letter
    if text and text[0].isalpha():
        text = text[0].upper() + text[1:]
    
    return text


def postprocess_batch(translations: List[str], aggressive: bool = True) -> List[str]:
    """Postprocess a batch of translations."""
    return [postprocess_translation(t, aggressive) for t in translations]


def create_submission(
    test_df: pd.DataFrame,
    translations: List[str],
    output_path: str
) -> pd.DataFrame:
    """
    Create submission file in Kaggle format.
    
    Args:
        test_df: Test dataframe with 'id' column
        translations: List of translated texts
        output_path: Path to save submission CSV
    
    Returns:
        Submission DataFrame
    """
    # Postprocess translations
    cleaned = postprocess_batch(translations, aggressive=True)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        'translation': cleaned
    })
    
    # Save
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    
    return submission


if __name__ == "__main__":
    # Test postprocessing
    test_cases = [
        "Šalim-Aššur made us approach Amur-Ištar.",
        "he owe 3 minas 3 minas of silver...",
        "the text [x] is damaged (fem)",
        "I₂ gave him 5.5 minas...",
    ]
    
    print("Testing postprocessing:")
    print("=" * 60)
    for text in test_cases:
        processed = postprocess_translation(text)
        print(f"Input:  {text}")
        print(f"Output: {processed}")
        print("-" * 60)
