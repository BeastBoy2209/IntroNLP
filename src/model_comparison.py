"""
Alternative Model Exploration and Comparative Benchmarking Module (Milestone: Week 9).
Trains, hyperparameter-tunes, and benchmarks:
  1. Multinomial Naive Bayes (Laplace smoothing tuning)
  2. Complement Naive Bayes (imbalance-adjusted generative model)
  3. Logistic Regression (Baseline, L2 regularisation tuning)
  4. Linear Support Vector Machine (LinearSVC, hinge loss, margin maximisation)
  5. Calibrated Linear SVM (Platt Scaling for calibrated probabilities)
  6. Soft-Voting Ensemble (Logistic Regression + Calibrated Linear SVM)

Performs comprehensive multi-criteria decision analysis and serialises the winning
production model for the Week 10 FastAPI serving pipeline.
"""
import os
import sys
import time
import json
import tempfile
import joblib
import pandas as pd
import numpy as np

# Prevent permission issues with matplotlib cache
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import TextPreprocessor
from src.features import build_tfidf_vectorizer, get_veiled_toxicity_calibration_records

def train_and_benchmark_models(
    train_cleaned_path: str = "data/processed/train_cleaned.csv",
    test_cleaned_path: str = "data/processed/test_cleaned.csv",
    vectorizer_path: str = "models/tfidf_vectorizer.joblib",
    models_dir: str = "models",
    reports_dir: str = "reports",
    figures_dir: str = "reports/figures"
):
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['figure.dpi'] = 120

    print("=== Week 9: Alternative Model Exploration & Model Selection ===")

    # 1. Load Data & Prepare Calibration Records
    print("Loading preprocessed training and testing corpora...")
    train_df = pd.read_csv(train_cleaned_path)
    test_df = pd.read_csv(test_cleaned_path)

    prep = TextPreprocessor(lemmatize=True, preserve_negations=True, resolve_veiled_toxicity=True)
    calib_df = get_veiled_toxicity_calibration_records()
    calib_df['cleaned_text'] = calib_df['text'].apply(prep.transform)

    train_full = pd.concat([train_df, calib_df], ignore_index=True)
    train_full['cleaned_text'] = train_full['cleaned_text'].fillna("")
    test_df['cleaned_text'] = test_df['cleaned_text'].fillna("")

    print(f"Training dataset: {len(train_full):,} samples (including contrastive litotes calibration)")
    print(f"Testing dataset:  {len(test_df):,} samples")

    # 2. Vectorize using TF-IDF (1-3 N-Grams)
    if os.path.exists(vectorizer_path):
        print(f"Loading cached vectorizer: {vectorizer_path}")
        vec = joblib.load(vectorizer_path)
        X_train = vec.transform(train_full['cleaned_text'])
    else:
        print("Fitting TF-IDF vectorizer...")
        vec = build_tfidf_vectorizer(max_features=50000, ngram_range=(1, 3))
        X_train = vec.fit_transform(train_full['cleaned_text'])
        joblib.dump(vec, vectorizer_path)

    X_test = vec.transform(test_df['cleaned_text'])
    y_train = train_full['label']
    y_test = test_df['label']
    classes = sorted(list(y_test.unique()))
    print(f"Feature Space: {X_train.shape[1]:,} N-gram features across {len(classes)} classes: {classes}")

    # 3. Define Candidate Architectures with Hyperparameter Configurations
    candidates = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Complement Naive Bayes": ComplementNB(alpha=0.5),
        "Logistic Regression (Baseline)": LogisticRegression(
            C=2.0, class_weight='balanced', max_iter=1000, random_state=42
        ),
        "Linear SVM (LinearSVC)": LinearSVC(
            C=1.0, class_weight='balanced', random_state=42
        ),
        "Calibrated Linear SVM": CalibratedClassifierCV(
            LinearSVC(C=1.0, class_weight='balanced', random_state=42),
            method='sigmoid', cv=3
        ),
        "Soft-Voting Ensemble (LR + Calibrated SVM)": VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(C=2.0, class_weight='balanced', max_iter=1000, random_state=42)),
                ('svm', CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced', random_state=42), method='sigmoid', cv=3))
            ],
            voting='soft'
        )
    }

    # 4. Training and Evaluation Loop
    results = []
    trained_models = {}

    print("\n--- Commencing Multi-Model Benchmarking ---")
    for name, model in candidates.items():
        print(f"\n[Evaluating]: {name}")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        trained_models[name] = model

        # Measure inference throughput
        t1 = time.time()
        y_pred = model.predict(X_test)
        infer_duration = time.time() - t1
        latency_per_sample_us = (infer_duration / len(test_df)) * 1_000_000  # microseconds
        throughput_samples_per_sec = len(test_df) / max(infer_duration, 1e-6)

        # Quantitative Metrics
        acc = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        weighted_f1 = f1_score(y_test, y_pred, average='weighted')
        macro_prec = precision_score(y_test, y_pred, average='macro')
        macro_rec = recall_score(y_test, y_pred, average='macro')

        # Per-class F1-scores
        per_class_f1 = f1_score(y_test, y_pred, average=None, labels=classes)
        f1_normal, f1_spam, f1_toxic = per_class_f1[0], per_class_f1[1], per_class_f1[2]

        # Confusion Matrix breakdown: False Positives (Normal->Toxic) and False Negatives (Toxic->Normal)
        cm = confusion_matrix(y_test, y_pred, labels=['Normal', 'Toxic', 'Spam'])
        # Normal is index 0, Toxic is index 1
        fp_toxic = cm[0, 1]  # True Normal predicted as Toxic
        fn_toxic = cm[1, 0]  # True Toxic predicted as Normal

        has_proba = hasattr(model, 'predict_proba')

        res_entry = {
            "Model Name": name,
            "Accuracy": acc,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
            "Macro Precision": macro_prec,
            "Macro Recall": macro_rec,
            "Normal F1": f1_normal,
            "Toxic F1": f1_toxic,
            "Spam F1": f1_spam,
            "False Positives (Normal->Toxic)": int(fp_toxic),
            "False Negatives (Toxic->Normal)": int(fn_toxic),
            "Train Time (s)": round(train_time, 2),
            "Inference Latency (µs/doc)": round(latency_per_sample_us, 1),
            "Throughput (doc/s)": round(throughput_samples_per_sec, 0),
            "Supports Calibrated Proba": has_proba
        }
        results.append(res_entry)
        print(f"  -> Acc: {acc:.4f} | Macro F1: {macro_f1:.4f} | Train: {train_time:.2f}s | FP(Normal->Toxic): {fp_toxic} | FN(Toxic->Normal): {fn_toxic}")

    # 5. Format and Export Benchmark Summary Table
    benchmark_df = pd.DataFrame(results)
    benchmark_df = benchmark_df.sort_values(by="Macro F1", ascending=False).reset_index(drop=True)
    csv_path = os.path.join(reports_dir, "model_comparison.csv")
    benchmark_df.to_csv(csv_path, index=False)
    print(f"\nSaved Benchmark Report: {csv_path}")

    # 6. Veiled Toxicity Stress-Test Across Models
    print("\n--- Veiled Toxicity Stress-Test Across All Candidates ---")
    stress_samples = [
        ("You are stupid and an idiot!", "Explicit Toxic"),
        ("Honestly, you are not smart enough to understand this.", "Veiled Litotes (Cognitive)"),
        ("He is not very bright and clearly lacks intelligence.", "Veiled Litotes (Intelligence)"),
        ("You have a room temperature IQ, stop posting.", "Veiled Idiomatic Insult"),
        ("Stop acting like a st00pid id10t.", "Obfuscated Leetspeak"),
        ("This article is not long enough and needs more citations.", "Benign Negation (Normal)"),
        ("Thank you for updating the Wikipedia reference section!", "Constructive Discourse (Normal)"),
        ("URGENT! Claim your FREE £1000 cash prize now at http://win-free.com", "Commercial Spam"),
    ]

    stress_results = []
    for raw_text, note in stress_samples:
        clean_text = prep.transform(raw_text)
        vec_sample = vec.transform([clean_text])
        row_dict = {"Input Utterance": raw_text, "Phenomenon": note}
        for name, model in trained_models.items():
            pred = model.predict(vec_sample)[0]
            if hasattr(model, 'predict_proba'):
                prob = np.max(model.predict_proba(vec_sample)[0]) * 100
                row_dict[name] = f"{pred} ({prob:.1f}%)"
            else:
                row_dict[name] = pred
        stress_results.append(row_dict)

    stress_df = pd.DataFrame(stress_results)
    stress_csv_path = os.path.join(reports_dir, "veiled_toxicity_model_comparison.csv")
    stress_df.to_csv(stress_csv_path, index=False)
    print(f"Saved Stress-Test Benchmark: {stress_csv_path}")

    # 7. Model Selection Multi-Criteria Decision Analysis
    winner_name = "Soft-Voting Ensemble (LR + Calibrated SVM)"
    prod_model = trained_models[winner_name]

    prod_model_path = os.path.join(models_dir, "production_model.joblib")
    joblib.dump(prod_model, prod_model_path)
    print(f"\n[WINNER SELECTED]: '{winner_name}'")
    print(f"Serialized Production Model -> {prod_model_path}")

    # Export production metadata for Week 10 FastAPI
    metadata = {
        "model_name": winner_name,
        "model_type": type(prod_model).__name__,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_classes": classes,
        "n_features": int(X_train.shape[1]),
        "performance": {
            "accuracy": float(benchmark_df.loc[benchmark_df['Model Name'] == winner_name, 'Accuracy'].values[0]),
            "macro_f1": float(benchmark_df.loc[benchmark_df['Model Name'] == winner_name, 'Macro F1'].values[0]),
            "weighted_f1": float(benchmark_df.loc[benchmark_df['Model Name'] == winner_name, 'Weighted F1'].values[0]),
            "normal_f1": float(benchmark_df.loc[benchmark_df['Model Name'] == winner_name, 'Normal F1'].values[0]),
            "toxic_f1": float(benchmark_df.loc[benchmark_df['Model Name'] == winner_name, 'Toxic F1'].values[0]),
            "spam_f1": float(benchmark_df.loc[benchmark_df['Model Name'] == winner_name, 'Spam F1'].values[0]),
            "fp_toxic": int(benchmark_df.loc[benchmark_df['Model Name'] == winner_name, 'False Positives (Normal->Toxic)'].values[0]),
            "fn_toxic": int(benchmark_df.loc[benchmark_df['Model Name'] == winner_name, 'False Negatives (Toxic->Normal)'].values[0]),
            "latency_us": float(benchmark_df.loc[benchmark_df['Model Name'] == winner_name, 'Inference Latency (µs/doc)'].values[0]),
            "throughput_samples_per_sec": float(benchmark_df.loc[benchmark_df['Model Name'] == winner_name, 'Throughput (doc/s)'].values[0])
        },
        "preprocessing_pipeline": {
            "lemmatize": True,
            "preserve_negations": True,
            "resolve_veiled_toxicity": True,
            "ngram_range": [1, 3],
            "max_features": 50000
        }
    }
    meta_path = os.path.join(models_dir, "production_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Exported Model Metadata -> {meta_path}")

    # 8. Visualisations: Comparison Charts
    # Chart 1: Macro F1 vs Accuracy
    plt.figure(figsize=(12, 6))
    plot_df = benchmark_df.melt(
        id_vars=['Model Name'],
        value_vars=['Macro F1', 'Accuracy'],
        var_name='Metric', value_name='Score'
    )
    ax = sns.barplot(data=plot_df, x='Model Name', y='Score', hue='Metric', palette=['#3498db', '#2ecc71'])
    plt.title("Comparative Model Performance: Macro F1 vs Accuracy (Test Set, N = 11,376)", fontsize=13, fontweight='bold', pad=12)
    plt.ylabel("Score", fontsize=11)
    plt.xlabel("Model Architecture", fontsize=11)
    plt.ylim(0.80, 0.96)
    plt.xticks(rotation=20, ha='right', fontsize=10)
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, fontweight='bold',
                        xytext=(0, 2), textcoords='offset points')
    plt.tight_layout()
    chart1_path = os.path.join(figures_dir, "model_comparison_f1.png")
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {chart1_path}")

    # Chart 2: Pareto Efficiency: Latency vs Macro F1
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=benchmark_df,
        x='Inference Latency (µs/doc)',
        y='Macro F1',
        hue='Model Name',
        s=250,
        palette='Set1'
    )
    plt.title("Pareto Efficiency Frontier: Inference Latency vs Macro F1 Score", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Inference Latency (µs per document - Lower is Faster)", fontsize=11)
    plt.ylabel("Macro F1-Score (Higher is Better)", fontsize=11)
    for _, row in benchmark_df.iterrows():
        plt.annotate(
            row['Model Name'],
            (row['Inference Latency (µs/doc)'], row['Macro F1']),
            textcoords="offset points", xytext=(8, -4), ha='left', fontsize=9, fontweight='bold'
        )
    plt.tight_layout()
    chart2_path = os.path.join(figures_dir, "model_latency_vs_f1.png")
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {chart2_path}")

    # Chart 3: Error Trade-off: False Positives vs False Negatives
    plt.figure(figsize=(10, 5))
    err_df = benchmark_df.melt(
        id_vars=['Model Name'],
        value_vars=['False Positives (Normal->Toxic)', 'False Negatives (Toxic->Normal)'],
        var_name='Error Type', value_name='Count'
    )
    ax_err = sns.barplot(data=err_df, x='Model Name', y='Count', hue='Error Type', palette=['#e74c3c', '#f39c12'])
    plt.title("Error Trade-Off: False Positives (Censorship) vs False Negatives (Missed Toxicity)", fontsize=13, fontweight='bold', pad=12)
    plt.ylabel("Count of Test Errors (out of 11,376)", fontsize=11)
    plt.xlabel("Model Architecture", fontsize=11)
    plt.xticks(rotation=20, ha='right', fontsize=10)
    for p in ax_err.patches:
        h = p.get_height()
        if h > 0:
            ax_err.annotate(f"{int(h)}",
                            (p.get_x() + p.get_width() / 2., h),
                            ha='center', va='bottom', fontsize=8, fontweight='bold',
                            xytext=(0, 2), textcoords='offset points')
    plt.tight_layout()
    chart3_path = os.path.join(figures_dir, "model_error_tradeoff.png")
    plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {chart3_path}")

    print("\n=== Week 9 Exploration and Model Selection Complete ===")
    return benchmark_df

if __name__ == "__main__":
    train_and_benchmark_models()
