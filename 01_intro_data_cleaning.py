# ============================================================
# SCRIPT 1: Data Loading, Exploration & Cleaning
# AI Prompt Quality Analyzer
# ============================================================
# INPUT : archive.csv  (ChatGPT Prompts dataset dari Kaggle)
# OUTPUT: cleaned_prompts.csv
# ============================================================

import pandas as pd
import os

print("=" * 50)
print("STEP 1: LOADING DATA")
print("=" * 50)

# Load dataset - coba beberapa nama file yang umum
possible_files = ['archive.csv', 'prompts.csv', 'chatgpt_prompts.csv']
df = None
for fname in possible_files:
    if os.path.exists(fname):
        df = pd.read_csv(fname)
        print(f"Loaded: {fname}")
        break

if df is None:
    raise FileNotFoundError("CSV file not found. Make sure the dataset CSV is in the same folder.")

print(f"\nShape      : {df.shape}")
print(f"Columns    : {df.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(df.head(3))

print("\n" + "=" * 50)
print("STEP 2: CHECKING DATA QUALITY")
print("=" * 50)

print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nData types:")
print(df.dtypes)

print(f"\nUnique 'act' categories: {df['act'].nunique()}")
print(f"Sample acts: {df['act'].head(10).tolist()}")

print("\n" + "=" * 50)
print("STEP 3: CLEANING")
print("=" * 50)

before = len(df)

# Drop rows with missing values
df = df.dropna(subset=['act', 'prompt'])

# Strip whitespace
df['act']    = df['act'].astype(str).str.strip()
df['prompt'] = df['prompt'].astype(str).str.strip()

# Remove empty rows
df = df[df['prompt'].str.len() > 5]

# Remove duplicates
df = df.drop_duplicates(subset=['prompt'])

# Reset index
df = df.reset_index(drop=True)

after = len(df)
print(f"Rows removed : {before - after}")
print(f"Final rows   : {after}")

print("\n" + "=" * 50)
print("STEP 4: SUMMARY")
print("=" * 50)

print(f"\nTop 10 most common 'act' categories:")
print(df['act'].value_counts().head(10))

print(f"\nAverage prompt length: {df['prompt'].apply(len).mean():.0f} characters")

# Save cleaned data
df.to_csv('cleaned_prompts.csv', index=False)
print("\n✅ Saved: cleaned_prompts.csv")
print("   → Run 02_intermediate_quality_scoring.py next")
