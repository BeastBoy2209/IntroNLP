"""
Dataset Pre-processing and Stratified Splitting Module.
Unifies raw comments into three target classes ('Normal', 'Toxic', 'Spam')
and executes an 80/20 stratified train/test split.
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def prepare_multiclass_dataset(
    jigsaw_path: str = "data/raw/train.csv",
    spam_path: str = "data/raw/spam_spam.csv",
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42,
    sample_normal_limit: int = 40000,
    sample_toxic_limit: int = 40000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs a unified DataFrame with 3 target classes: 'Normal', 'Toxic', and 'Spam'.
    Performs stratified partitioning into Training (80%) and Test (20%) sets.
    """
    os.makedirs(output_dir, exist_ok=True)
    dfs = []

    # 1. Process Jigsaw Toxic Comment Classification
    if os.path.exists(jigsaw_path):
        print(f"Loading Jigsaw dataset from: {jigsaw_path}")
        df_jigsaw = pd.read_csv(jigsaw_path)
        
        # Identify toxic comments (positive flag in any toxic category)
        toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        available_toxic_cols = [c for c in toxic_cols if c in df_jigsaw.columns]
        
        if available_toxic_cols:
            is_toxic = (df_jigsaw[available_toxic_cols].sum(axis=1) > 0)
            df_toxic = df_jigsaw[is_toxic].copy()
            df_normal = df_jigsaw[~is_toxic].copy()
        else:
            if 'label' in df_jigsaw.columns:
                df_toxic = df_jigsaw[df_jigsaw['label'] == 1].copy()
                df_normal = df_jigsaw[df_jigsaw['label'] == 0].copy()
            else:
                raise ValueError("Expected toxicity columns not found in Jigsaw dataset.")

        df_toxic = df_toxic.rename(columns={'comment_text': 'text'})[['text']]
        df_toxic['label'] = 'Toxic'

        df_normal = df_normal.rename(columns={'comment_text': 'text'})[['text']]
        df_normal['label'] = 'Normal'

        # Manage class balance if requested
        if sample_toxic_limit and len(df_toxic) > sample_toxic_limit:
            df_toxic = df_toxic.sample(n=sample_toxic_limit, random_state=random_state)
        if sample_normal_limit and len(df_normal) > sample_normal_limit:
            df_normal = df_normal.sample(n=sample_normal_limit, random_state=random_state)

        dfs.extend([df_normal, df_toxic])
        print(f"Jigsaw parsed: {len(df_normal)} Normal, {len(df_toxic)} Toxic")
    else:
        print(f"Warning: File {jigsaw_path} not found. Ensure raw data is downloaded.")

    # 2. Process Spam dataset
    if os.path.exists(spam_path):
        print(f"Loading Spam dataset from: {spam_path}")
        try:
            df_spam_raw = pd.read_csv(spam_path, encoding='latin-1')
            if 'v1' in df_spam_raw.columns and 'v2' in df_spam_raw.columns:
                df_spam = df_spam_raw[df_spam_raw['v1'] == 'spam'].copy()
                df_spam = df_spam.rename(columns={'v2': 'text'})[['text']]
                df_spam['label'] = 'Spam'
                dfs.append(df_spam)
                print(f"Spam dataset parsed: {len(df_spam)} Spam samples")
            elif 'label' in df_spam_raw.columns and 'text' in df_spam_raw.columns:
                df_spam = df_spam_raw[df_spam_raw['label'].str.lower() == 'spam'].copy()
                df_spam['label'] = 'Spam'
                dfs.append(df_spam[['text', 'label']])
        except Exception as e:
            print(f"Unable to parse spam dataset: {e}")

    if not dfs:
        raise FileNotFoundError("No raw data files found for processing.")

    # Combine and de-duplicate
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.dropna(subset=['text']).drop_duplicates(subset=['text'])
    
    print("\nClass distribution prior to train/test split:")
    print(combined_df['label'].value_counts())
    print(f"Total instances: {len(combined_df)}")

    # 3. Stratified Split: Train (80%) and Test (20%)
    train_df, test_df = train_test_split(
        combined_df,
        test_size=test_size,
        stratify=combined_df['label'],
        random_state=random_state
    )

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nData partitioning complete:")
    print(f"Training split (80%): {len(train_df):,} instances -> {train_path}")
    print(f"Test split (20%):     {len(test_df):,} instances -> {test_path}")

    return train_df, test_df

if __name__ == "__main__":
    prepare_multiclass_dataset()
