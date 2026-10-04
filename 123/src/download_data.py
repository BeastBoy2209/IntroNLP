"""
Модуль для загрузки датасетов (Jigsaw Toxic Comments и вспомогательных датасетов для Spam).
"""
import os
import shutil
import sys

def download_jigsaw_dataset(target_dir: str = "data/raw"):
    """
    Загрузка датасета Jigsaw Toxic Comment Classification с помощью kagglehub.
    """
    os.makedirs(target_dir, exist_ok=True)
    print("Попытка загрузки датасета Jigsaw через kagglehub...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("julian3833/jigsaw-toxic-comment-classification-challenge")
        print(f"Датасет успешно скачан в кэш: {path}")
        
        # Копируем основные файлы в target_dir
        for file_name in os.listdir(path):
            src_file = os.path.join(path, file_name)
            dst_file = os.path.join(target_dir, file_name)
            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                print(f"Скопирован: {file_name} -> {target_dir}")
        print(f"Все файлы перемещены в {target_dir}")
        return True
    except Exception as e:
        print(f"Ошибка при автоматической загрузке через kagglehub: {e}")
        print("Если у вас уже есть train.csv из соревнования Jigsaw, поместите его в папку data/raw/")
        return False

def download_spam_dataset(target_dir: str = "data/raw"):
    """
    Загрузка SMS Spam Collection для дополнения класса Spam.
    """
    os.makedirs(target_dir, exist_ok=True)
    try:
        import kagglehub
        path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
        print(f"Spam датасет скачан в: {path}")
        for file_name in os.listdir(path):
            src_file = os.path.join(path, file_name)
            dst_file = os.path.join(target_dir, f"spam_{file_name}")
            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                print(f"Скопирован spam файл: {file_name} -> {dst_file}")
        return True
    except Exception as e:
        print(f"Примечание по загрузке спам-датасета: {e}")
        return False

if __name__ == "__main__":
    download_jigsaw_dataset()
    download_spam_dataset()
