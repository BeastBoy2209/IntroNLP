"""
Baseline Model Training Pipeline (Milestone: Week 7).
Extracts TF-IDF features (1-3 word N-grams + compositional litotes tokens)
and trains a Class-Weighted Logistic Regression classifier.
Demonstrates mitigation of veiled/implicit toxicity (e.g., 'not smart enough').
"""
import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import TextPreprocessor
from src.features import build_tfidf_vectorizer, get_veiled_toxicity_calibration_records

def train_and_save_baseline(
    train_path: str = "data/processed/train_cleaned.csv",
    test_path: str = "data/processed/test_cleaned.csv",
    models_dir: str = "models"
):
    os.makedirs(models_dir, exist_ok=True)
    print("=== Week 7: TF-IDF Feature Extraction & Logistic Regression Baseline ===")

    # 1. Load preprocessed datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 2. Append contrastive veiled-toxicity calibration records
    prep = TextPreprocessor(lemmatize=True, preserve_negations=True, resolve_veiled_toxicity=True)
    calib_df = get_veiled_toxicity_calibration_records()
    calib_df['cleaned_text'] = calib_df['text'].apply(prep.transform)

    train_full = pd.concat([train_df, calib_df], ignore_index=True)
    train_full['cleaned_text'] = train_full['cleaned_text'].fillna("")
    test_df['cleaned_text'] = test_df['cleaned_text'].fillna("")

    print(f"Training instances: {len(train_full):,} (including veiled-toxicity contrastive calibration)")
    print(f"Testing instances:  {len(test_df):,}")

    # 3. Fit TF-IDF Vectorizer (1 to 3 N-grams)
    print("\nFitting TF-IDF Vectorizer (ngram_range=(1, 3), max_features=50,000)...")
    t0 = time.time()
    vectorizer = build_tfidf_vectorizer(max_features=50000, ngram_range=(1, 3))
    X_train = vectorizer.fit_transform(train_full['cleaned_text'])
    X_test = vectorizer.transform(test_df['cleaned_text'])
    y_train = train_full['label']
    y_test = test_df['label']
    print(f"TF-IDF Matrix Shape: {X_train.shape} (extracted in {time.time() - t0:.2f}s)")

    # 4. Train Class-Weighted Logistic Regression Baseline
    print("\nTraining Class-Weighted Logistic Regression...")
    t1 = time.time()
    clf = LogisticRegression(
        C=2.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )
    clf.fit(X_train, y_train)
    print(f"Logistic Regression trained in {time.time() - t1:.2f}s")

    # 5. Preliminary Evaluation on Test Split
    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)

    print(f"\n--- Baseline Performance Summary (Test Set, N={len(test_df):,}) ---")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # 6. Veiled Toxicity Stress-Test (Professor's Feedback Verification)
    print("--- Veiled & Implicit Toxicity Stress-Test ---")
    stress_samples = [
        ("You are stupid and an idiot!", "Explicit Toxic"),
        ("Honestly, you are not smart enough to understand this.", "Veiled Toxic (Litotes)"),
        ("He is not very bright and clearly lacks intelligence.", "Veiled Toxic (Litotes)"),
        ("You have a room temperature IQ, stop posting.", "Veiled Toxic (Idiom)"),
        ("Stop acting like a st00pid id10t.", "Obfuscated Leetspeak Toxic"),
        ("This article is not long enough and needs more citations.", "Normal (Benign Negation)"),
        ("Thank you for updating the Wikipedia reference section!", "Normal (Constructive)"),
        ("URGENT! Claim your FREE £1000 cash prize now at http://win-free.com", "Spam (Commercial)"),
    ]

    for raw_text, category_note in stress_samples:
        cleaned = prep.transform(raw_text)
        vec = vectorizer.transform([cleaned])
        pred_label = clf.predict(vec)[0]
        probs = clf.predict_proba(vec)[0]
        class_idx = list(clf.classes_).index(pred_label)
        confidence = probs[class_idx] * 100
        print(f"[{category_note:26s}] -> Pred: {pred_label:6s} ({confidence:5.1f}%) | Text: \"{raw_text}\"")

    # 7. Save Serialized Artifacts
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    model_path = os.path.join(models_dir, "baseline_logreg.joblib")
    joblib.dump(vectorizer, vec_path)
    joblib.dump(clf, model_path)
    print(f"\nSaved Vectorizer -> {vec_path}")
    print(f"Saved Model      -> {model_path}")

if __name__ == "__main__":
    train_and_save_baseline()
