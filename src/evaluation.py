"""
Comprehensive Model Evaluation and Diagnostic Error Analysis Module (Milestone: Week 8).
Computes multi-class metrics (Accuracy, Precision, Recall, Macro/Weighted F1, ROC-AUC, PR-AUC),
generates Confusion Matrices, ROC/PR curves, and performs qualitative error audits.
"""
import os
import sys
import tempfile
import joblib
import pandas as pd
import numpy as np

# Configure matplotlib config dir to prevent permission issues on macOS
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

def run_evaluation(
    model_path: str = "models/baseline_logreg.joblib",
    vectorizer_path: str = "models/tfidf_vectorizer.joblib",
    test_cleaned_path: str = "data/processed/test_cleaned.csv",
    test_raw_path: str = "data/processed/test.csv",
    reports_dir: str = "reports",
    figures_dir: str = "reports/figures"
):
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams['figure.dpi'] = 120

    print("=== Week 8: Baseline Evaluation & Diagnostic Error Analysis ===")

    # 1. Load artifacts and test set
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model or Vectorizer artifacts missing in models/ directory.")

    print(f"Loading model: {model_path}")
    clf = joblib.load(model_path)
    print(f"Loading vectorizer: {vectorizer_path}")
    vec = joblib.load(vectorizer_path)

    test_df = pd.read_csv(test_cleaned_path)
    raw_test_df = pd.read_csv(test_raw_path) if os.path.exists(test_raw_path) else None
    
    # Align raw text for human-readable error analysis
    if raw_test_df is not None and len(raw_test_df) == len(test_df):
        test_df['raw_text'] = raw_test_df['text']
    else:
        test_df['raw_text'] = test_df['text'] if 'text' in test_df.columns else test_df['cleaned_text']

    classes = list(clf.classes_)
    print(f"Target Classes ({len(classes)}): {classes}")
    print(f"Evaluating on {len(test_df):,} test instances...")

    # 2. Vectorize and Predict
    X_test = vec.transform(test_df['cleaned_text'].fillna(""))
    y_true = test_df['label']
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    # 3. Quantitative Performance Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average='macro')
    macro_rec = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')

    # Binarize labels for multi-class ROC-AUC and PR-AUC
    y_true_bin = label_binarize(y_true, classes=classes)
    macro_roc_auc = roc_auc_score(y_true_bin, y_prob, average='macro', multi_class='ovr')
    weighted_roc_auc = roc_auc_score(y_true_bin, y_prob, average='weighted', multi_class='ovr')

    print("\n" + "=" * 50)
    print("      GLOBAL EVALUATION METRICS (TEST SET)      ")
    print("=" * 50)
    print(f"Overall Accuracy:                 {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro-Averaged Precision:         {macro_prec:.4f}")
    print(f"Macro-Averaged Recall:            {macro_rec:.4f}")
    print(f"Macro-Averaged F1-Score:          {macro_f1:.4f} ({macro_f1*100:.2f}%) [PRIMARY]")
    print(f"Weighted F1-Score:                {weighted_f1:.4f}")
    print(f"Multi-Class ROC-AUC (Macro OVR):   {macro_roc_auc:.4f}")
    print(f"Multi-Class ROC-AUC (Weighted):   {weighted_roc_auc:.4f}")
    print("=" * 50)

    print("\nDetailed Classification Report:")
    report_str = classification_report(y_true, y_pred, digits=4)
    print(report_str)

    # 4. Confusion Matrices (Raw Counts & Normalised)
    cm_raw = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = confusion_matrix(y_true, y_pred, labels=classes, normalize='true')

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Raw Confusion Matrix
    sns.heatmap(cm_raw, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=classes, yticklabels=classes, cbar=False)
    axes[0].set_title("Confusion Matrix (Absolute Counts)", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Ground Truth", fontsize=11)

    # Normalised Confusion Matrix (Recall per class along diagonal)
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='YlGnBu', ax=axes[1],
                xticklabels=classes, yticklabels=classes, cbar=True)
    axes[1].set_title("Normalised Confusion Matrix (Recall per True Class)", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Predicted Label", fontsize=11)
    axes[1].set_ylabel("True Ground Truth", fontsize=11)

    plt.tight_layout()
    cm_path = os.path.join(figures_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {cm_path}")

    # 5. One-vs-Rest ROC Curves
    plt.figure(figsize=(8, 6))
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    for idx, (cls, col) in enumerate(zip(classes, colors)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_prob[:, idx])
        auc_val = roc_auc_score(y_true_bin[:, idx], y_prob[:, idx])
        plt.plot(fpr, tpr, label=f"{cls} (AUC = {auc_val:.4f})", color=col, linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (AUC = 0.50)')
    plt.title(f"Multi-Class One-vs-Rest ROC Curves (Macro AUC = {macro_roc_auc:.4f})", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("False Positive Rate (FPR)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / TPR)", fontsize=11)
    plt.legend(loc='lower right', frameon=True)
    roc_path = os.path.join(figures_dir, "roc_curves.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {roc_path}")

    # 6. Precision-Recall Curves (Essential for Imbalanced Datasets)
    plt.figure(figsize=(8, 6))
    for idx, (cls, col) in enumerate(zip(classes, colors)):
        prec, rec, _ = precision_recall_curve(y_true_bin[:, idx], y_prob[:, idx])
        ap_val = average_precision_score(y_true_bin[:, idx], y_prob[:, idx])
        plt.plot(rec, prec, label=f"{cls} (Avg Precision = {ap_val:.4f})", color=col, linewidth=2)

    plt.title("Multi-Class Precision-Recall Curves", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Recall", fontsize=11)
    plt.ylabel("Precision", fontsize=11)
    plt.legend(loc='lower left', frameon=True)
    pr_path = os.path.join(figures_dir, "precision_recall_curves.png")
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {pr_path}")

    # 7. Diagnostic Qualitative Error Audit
    test_df['predicted_label'] = y_pred
    test_df['confidence'] = np.max(y_prob, axis=1)
    test_df['is_error'] = (test_df['label'] != test_df['predicted_label'])

    errors_df = test_df[test_df['is_error']].copy()
    errors_df['error_type'] = errors_df['label'] + " -> " + errors_df['predicted_label']

    print(f"\nTotal Misclassified Samples: {len(errors_df):,} out of {len(test_df):,} ({len(errors_df)/len(test_df)*100:.2f}%)")
    print("\nError Breakdown by Confusion Path:")
    print(errors_df['error_type'].value_counts())

    # High-confidence False Positives (Normal predicted as Toxic)
    fp_toxic = errors_df[errors_df['error_type'] == 'Normal -> Toxic'].sort_values(by='confidence', ascending=False)
    # High-confidence False Negatives (Toxic predicted as Normal)
    fn_toxic = errors_df[errors_df['error_type'] == 'Toxic -> Normal'].sort_values(by='confidence', ascending=False)

    print("\nTop 3 False Positives (Normal misclassified as Toxic with High Confidence):")
    for i, (_, row) in enumerate(fp_toxic.head(3).iterrows(), 1):
        print(f" {i}. Conf: {row['confidence']*100:.1f}% | Raw: \"{str(row['raw_text'])[:90]}...\"")

    print("\nTop 3 False Negatives (Toxic misclassified as Normal with High Confidence):")
    for i, (_, row) in enumerate(fn_toxic.head(3).iterrows(), 1):
        print(f" {i}. Conf: {row['confidence']*100:.1f}% | Raw: \"{str(row['raw_text'])[:90]}...\"")

    # Export Error Audit CSV
    error_csv_path = os.path.join(reports_dir, "error_analysis.csv")
    export_cols = ['raw_text', 'cleaned_text', 'label', 'predicted_label', 'confidence', 'error_type']
    errors_df[[c for c in export_cols if c in errors_df.columns]].to_csv(error_csv_path, index=False)
    print(f"\nSaved Error Audit Dataset: {error_csv_path}")

    print("\n=== Week 8 Evaluation and Error Analysis Completed ===")

if __name__ == "__main__":
    run_evaluation()
