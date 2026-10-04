"""
Скрипт для разведочного анализа данных (EDA).
Анализирует распределение классов, длину сообщений, строит графики и сохраняет отчет.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(
    train_path: str = "data/processed/train.csv",
    test_path: str = "data/processed/test.csv",
    reports_dir: str = "reports/figures"
):
    os.makedirs(reports_dir, exist_ok=True)
    
    print("=== Запуск EDA (Exploratory Data Analysis) ===")
    if not os.path.exists(train_path):
        print(f"Файл {train_path} не найден. Сначала выполните подготовку данных (src/dataset.py).")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else None

    print(f"\nРазмер обучающей выборки: {train_df.shape}")
    if test_df is not None:
        print(f"Размер тестовой выборки: {test_df.shape}")

    # Расчет характеристик текстов
    train_df['char_length'] = train_df['text'].astype(str).apply(len)
    train_df['word_count'] = train_df['text'].astype(str).apply(lambda x: len(x.split()))

    # 1. Распределение классов
    print("\nРаспределение классов в train:")
    class_counts = train_df['label'].value_counts()
    class_percent = train_df['label'].value_counts(normalize=True) * 100
    for cls, count in class_counts.items():
        print(f" - {cls}: {count} ({class_percent[cls]:.2f}%)")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=train_df, x='label', palette='viridis', order=class_counts.index)
    plt.title('Распределение классов в обучающей выборке (Train)')
    plt.xlabel('Класс')
    plt.ylabel('Количество')
    class_dist_path = os.path.join(reports_dir, 'class_distribution.png')
    plt.savefig(class_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График распределения классов сохранен: {class_dist_path}")

    # 2. Длина сообщений по классам (символы)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=train_df, x='label', y='char_length', palette='Set2', showfliers=False)
    plt.title('Распределение длины сообщений в символах (без выбросов)')
    plt.xlabel('Класс')
    plt.ylabel('Символов в сообщении')
    length_dist_path = os.path.join(reports_dir, 'char_length_distribution.png')
    plt.savefig(length_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График длины сообщений сохранен: {length_dist_path}")

    # 3. Длина сообщений в словах
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=train_df, x='label', y='word_count', palette='Pastel1', showfliers=False)
    plt.title('Распределение количества слов по классам (без выбросов)')
    plt.xlabel('Класс')
    plt.ylabel('Слов в сообщении')
    words_dist_path = os.path.join(reports_dir, 'word_count_distribution.png')
    plt.savefig(words_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"График количества слов сохранен: {words_dist_path}")

    # Статистика длины
    print("\nСводная статистика по количеству слов по классам:")
    stats = train_df.groupby('label')['word_count'].describe()
    print(stats[['count', 'mean', 'std', 'min', '50%', 'max']])

    print("\n=== EDA успешно завершен ===")

if __name__ == "__main__":
    run_eda()
