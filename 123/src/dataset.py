"""
Модуль подготовки, разметки и разделения датасета на train/test (80/20).
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
    random_state: float = 42,
    sample_normal_limit: int = 40000,
    sample_toxic_limit: int = 40000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Формирует объединенный датасет с 3 основными классами: 'Normal', 'Toxic', 'Spam'.
    Выполняет стратифицированное разделение на train (80%) и test (20%).
    """
    os.makedirs(output_dir, exist_ok=True)
    dfs = []

    # 1. Обработка Jigsaw Toxic Comment Classification
    if os.path.exists(jigsaw_path):
        print(f"Загрузка Jigsaw датасета из: {jigsaw_path}")
        df_jigsaw = pd.read_csv(jigsaw_path)
        
        # Определяем токсичные комментарии (если хотя бы один флаг равен 1)
        toxic_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        available_toxic_cols = [c for c in toxic_cols if c in df_jigsaw.columns]
        
        if available_toxic_cols:
            is_toxic = (df_jigsaw[available_toxic_cols].sum(axis=1) > 0)
            df_toxic = df_jigsaw[is_toxic].copy()
            df_normal = df_jigsaw[~is_toxic].copy()
        else:
            # Если датасет уже бинарный или имеет колонку label
            if 'label' in df_jigsaw.columns:
                df_toxic = df_jigsaw[df_jigsaw['label'] == 1].copy()
                df_normal = df_jigsaw[df_jigsaw['label'] == 0].copy()
            else:
                raise ValueError("Не найдены ожидаемые колонки токсичности в Jigsaw датасете.")

        df_toxic = df_toxic.rename(columns={'comment_text': 'text'})[['text']]
        df_toxic['label'] = 'Toxic'

        df_normal = df_normal.rename(columns={'comment_text': 'text'})[['text']]
        df_normal['label'] = 'Normal'

        # Для баланса классов можно сбалансировать выборку
        if sample_toxic_limit and len(df_toxic) > sample_toxic_limit:
            df_toxic = df_toxic.sample(n=sample_toxic_limit, random_state=random_state)
        if sample_normal_limit and len(df_normal) > sample_normal_limit:
            df_normal = df_normal.sample(n=sample_normal_limit, random_state=random_state)

        dfs.extend([df_normal, df_toxic])
        print(f"Jigsaw загружен: {len(df_normal)} Normal, {len(df_toxic)} Toxic")
    else:
        print(f"Внимание: Файл {jigsaw_path} не найден. Убедитесь, что датасет скачан.")

    # 2. Обработка Spam датасета (если доступен)
    if os.path.exists(spam_path):
        print(f"Загрузка Spam датасета из: {spam_path}")
        try:
            df_spam_raw = pd.read_csv(spam_path, encoding='latin-1')
            # Обычно в SMS spam датасете колонки v1 (метка) и v2 (текст)
            if 'v1' in df_spam_raw.columns and 'v2' in df_spam_raw.columns:
                df_spam = df_spam_raw[df_spam_raw['v1'] == 'spam'].copy()
                df_spam = df_spam.rename(columns={'v2': 'text'})[['text']]
                df_spam['label'] = 'Spam'
                dfs.append(df_spam)
                print(f"Spam датасет загружен: {len(df_spam)} записей Spam")
            elif 'label' in df_spam_raw.columns and 'text' in df_spam_raw.columns:
                df_spam = df_spam_raw[df_spam_raw['label'].str.lower() == 'spam'].copy()
                df_spam['label'] = 'Spam'
                dfs.append(df_spam[['text', 'label']])
        except Exception as e:
            print(f"Не удалось распарсить спам датасет: {e}")

    if not dfs:
        raise FileNotFoundError("Ни один файл данных не найден для обработки.")

    # Объединяем и перемешиваем
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.dropna(subset=['text']).drop_duplicates(subset=['text'])
    
    print("\nРаспределение классов перед разделением:")
    print(combined_df['label'].value_counts())
    print(f"Всего объектов: {len(combined_df)}")

    # 3. Разделение на train (80%) и test (20%) со стратификацией
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

    print(f"\nДанные успешно разделены:")
    print(f"Обучающая выборка (80%): {len(train_df)} строк -> {train_path}")
    print(f"Тестовая выборка (20%): {len(test_df)} строк -> {test_path}")

    return train_df, test_df

if __name__ == "__main__":
    prepare_multiclass_dataset()
