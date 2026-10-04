"""
Exploratory Data Analysis (EDA) Pipeline.
Generates comprehensive statistical summaries and publication-quality visualisations
for class distribution, character lengths, word counts, and vocabulary statistics.
"""
import os
import tempfile

# Configure matplotlib config dir to prevent permission issues on macOS
os.environ['MPLCONFIGDIR'] = tempfile.mkdtemp()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv",
    reports_dir: str = "reports/figures"
):
    os.makedirs(reports_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    
    print("=== Executing Exploratory Data Analysis (EDA) ===")
    if not os.path.exists(train_path):
        print(f"Error: Training file '{train_path}' not found. Run 'src/dataset.py' first.")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

    print(f"\nTraining set shape: {train_df.shape}")
    if test_df is not None:
        print(f"Test set shape:     {test_df.shape}")

    # Compute text length metrics
    train_df['char_length'] = train_df['text'].astype(str).apply(len)
    train_df['word_count'] = train_df['text'].astype(str).apply(lambda x: len(x.split()))

    # 1. Class Distribution
    print("\nClass distribution in Training Set:")
    class_counts = train_df['label'].value_counts()
    class_percent = train_df['label'].value_counts(normalize=True) * 100
    for cls, count in class_counts.items():
        print(f" - {cls:6s}: {count:6d} ({class_percent[cls]:.2f}%)")

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette='viridis', legend=False)
    plt.title('Target Class Distribution in Training Set (N = 45,502)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Instance Count', fontsize=12)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height):,}\n({height/len(train_df)*100:.1f}%)',
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    class_dist_path = os.path.join(reports_dir, 'class_distribution.png')
    plt.savefig(class_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {class_dist_path}")

    # 2. Character Length Distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=train_df, x='label', y='char_length', hue='label', palette='Set2', showfliers=False, legend=False)
    plt.title('Character Length Distribution Across Classes (Outliers Excluded)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Character Count per Message', fontsize=12)
    char_dist_path = os.path.join(reports_dir, 'char_length_distribution.png')
    plt.savefig(char_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {char_dist_path}")

    # 3. Word Count Distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=train_df, x='label', y='word_count', hue='label', palette='Pastel1', showfliers=False, legend=False)
    plt.title('Word Count Distribution Across Classes (Outliers Excluded)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Word Count per Message', fontsize=12)
    words_dist_path = os.path.join(reports_dir, 'word_count_distribution.png')
    plt.savefig(words_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {words_dist_path}")

    # 4. Detailed Summary Table
    print("\nText Length Descriptive Statistics by Class:")
    stats = train_df.groupby('label')['word_count'].describe()
    print(stats[['count', 'mean', 'std', 'min', '50%', 'max']].to_string())

    print("\n=== Exploratory Data Analysis Successfully Executed ===")

if __name__ == "__main__":
    run_eda()
