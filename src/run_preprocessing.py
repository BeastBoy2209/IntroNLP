"""
Batch Execution Script for Text Preprocessing.
Processes train.csv and test.csv using the production TextPreprocessor pipeline
and exports train_cleaned.csv and test_cleaned.csv for model training (Weeks 7–9).
"""
import os
import sys
import time
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import TextPreprocessor

def main():
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"
    train_out = "data/processed/train_cleaned.csv"
    test_out = "data/processed/test_cleaned.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: Missing {train_path} or {test_path}. Run src/dataset.py first.")
        return

    pipeline = TextPreprocessor(
        lower=True,
        expand_contractions=True,
        remove_html=True,
        replace_urls=True,
        normalize_elongations=True,
        remove_punctuation=True,
        remove_stopwords=True,
        preserve_negations=True,
        lemmatize=True
    )

    print("=== Commencing Batch Text Preprocessing ===")

    # Process Training Data
    print(f"Loading {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"Preprocessing {len(train_df):,} training texts...")
    t0 = time.time()
    train_df['cleaned_text'] = train_df['text'].apply(pipeline.transform)
    # Ensure no empty strings
    train_df['cleaned_text'] = train_df['cleaned_text'].fillna("").apply(lambda s: s if len(s.strip()) > 0 else "<EMPTY>")
    train_df.to_csv(train_out, index=False)
    print(f"Saved: {train_out} ({len(train_df):,} rows in {time.time() - t0:.1f}s)")

    # Process Testing Data
    print(f"\nLoading {test_path}...")
    test_df = pd.read_csv(test_path)
    print(f"Preprocessing {len(test_df):,} testing texts...")
    t1 = time.time()
    test_df['cleaned_text'] = test_df['text'].apply(pipeline.transform)
    test_df['cleaned_text'] = test_df['cleaned_text'].fillna("").apply(lambda s: s if len(s.strip()) > 0 else "<EMPTY>")
    test_df.to_csv(test_out, index=False)
    print(f"Saved: {test_out} ({len(test_df):,} rows in {time.time() - t1:.1f}s)")

    print("\n=== Preprocessing Execution Successfully Completed ===")

if __name__ == "__main__":
    main()
