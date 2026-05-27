# ============================================================
# SCRIPT 2: Quality Scoring & Categorization
# AI Prompt Quality Analyzer
# ============================================================
# INPUT : cleaned_prompts.csv
# OUTPUT: scored_prompts.csv
# ============================================================

import pandas as pd
import numpy as np
import re

print("=" * 50)
print("STEP 1: LOADING CLEANED DATA")
print("=" * 50)

df = pd.read_csv('cleaned_prompts.csv')
print(f"Loaded {len(df)} rows")

print("\n" + "=" * 50)
print("STEP 2: ENGINEERING TEXT FEATURES")
print("=" * 50)

df['word_count']       = df['prompt'].apply(lambda x: len(str(x).split()))
df['char_count']       = df['prompt'].apply(lambda x: len(str(x)))
df['sentence_count']   = df['prompt'].apply(
    lambda x: len([s for s in re.split(r'[.!?]+', str(x)) if len(s.strip()) > 2])
)
df['unique_word_ratio'] = df['prompt'].apply(
    lambda x: len(set(str(x).lower().split())) / len(str(x).split())
    if str(x).split() else 0
)
df['avg_word_length']  = df['prompt'].apply(
    lambda x: np.mean([len(w) for w in str(x).split()]) if str(x).split() else 0
)

print(df[['word_count', 'char_count', 'sentence_count', 'unique_word_ratio']].describe().round(2))

print("\n" + "=" * 50)
print("STEP 3: QUALITY SCORING (0-100)")
print("=" * 50)

# Markers that indicate a well-structured, specific prompt
QUALITY_MARKERS = [
    'you are', 'your task', 'i want you', 'please', 'do not',
    'must', 'should', 'will', 'when', 'if', 'provide', 'explain',
    'make sure', 'only', 'always', 'never', 'respond', 'act as'
]

def calculate_prompt_quality(text):
    """
    Score a prompt 0-100 across 4 dimensions:
    - Clarity (length)   : 0-25 pts
    - Structure          : 0-25 pts
    - Vocabulary         : 0-25 pts
    - Specificity        : 0-25 pts
    """
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 0

    score = 0
    text = str(text)

    # Dimension 1: Clarity via length (0-25 pts)
    wc = len(text.split())
    if wc >= 80:   score += 25
    elif wc >= 40: score += 20
    elif wc >= 20: score += 13
    elif wc >= 10: score += 7
    else:          score += 3

    # Dimension 2: Sentence structure (0-25 pts)
    sents = [s for s in re.split(r'[.!?]+', text) if len(s.strip()) > 2]
    sc = len(sents)
    if sc >= 5:   score += 25
    elif sc >= 3: score += 18
    elif sc == 2: score += 10
    else:         score += 4

    # Dimension 3: Vocabulary richness (0-25 pts)
    words = text.lower().split()
    if words:
        score += int((len(set(words)) / len(words)) * 25)

    # Dimension 4: Specificity markers (0-25 pts)
    text_lower = text.lower()
    hits = sum(1 for m in QUALITY_MARKERS if m in text_lower)
    score += min(hits * 4, 25)

    return min(score, 100)


def categorize_quality(score):
    if score >= 75: return 'High'
    elif score >= 50: return 'Medium'
    elif score >= 25: return 'Low'
    else: return 'Very Low'


df['quality_score']    = df['prompt'].apply(calculate_prompt_quality)
df['quality_category'] = df['quality_score'].apply(categorize_quality)

print(f"Score range : {df['quality_score'].min()} – {df['quality_score'].max()}")
print(f"Mean score  : {df['quality_score'].mean():.2f}")

print("\n" + "=" * 50)
print("STEP 4: ANALYSIS BY CATEGORY (act)")
print("=" * 50)

top_acts = df.groupby('act')['quality_score'].agg(['mean', 'count']).sort_values('mean', ascending=False)
print("\nTop 10 highest scoring prompt categories:")
print(top_acts.head(10).round(2))

print("\nQuality distribution:")
print(df['quality_category'].value_counts())

df.to_csv('scored_prompts.csv', index=False)
print("\n✅ Saved: scored_prompts.csv")
print("   → Run 03_visualization_dashboard.py next")
