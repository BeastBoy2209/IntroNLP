"""
Data Ingestion Module.
Downloads the Jigsaw Toxic Comment Classification and supplementary Spam datasets via kagglehub.
"""
import os
import shutil
import sys

def download_jigsaw_dataset(target_dir: str = "data/raw") -> bool:
    """
    Downloads the Jigsaw Toxic Comment Classification challenge dataset using kagglehub.
    """
    os.makedirs(target_dir, exist_ok=True)
    print("Initiating Jigsaw dataset download via kagglehub...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("julian3833/jigsaw-toxic-comment-classification-challenge")
        print(f"Dataset cached successfully at: {path}")
        
        # Copy core files to target_dir
        for file_name in os.listdir(path):
            src_file = os.path.join(path, file_name)
            dst_file = os.path.join(target_dir, file_name)
            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                print(f"Transferred: {file_name} -> {target_dir}")
        print(f"All Jigsaw files successfully placed in {target_dir}")
        return True
    except Exception as e:
        print(f"Error downloading Jigsaw dataset via kagglehub: {e}")
        print("Note: If you have train.csv manually downloaded, place it directly in data/raw/")
        return False

def download_spam_dataset(target_dir: str = "data/raw") -> bool:
    """
    Downloads the SMS Spam Collection dataset to provide samples for the 'Spam' class.
    """
    os.makedirs(target_dir, exist_ok=True)
    print("Initiating SMS Spam dataset download via kagglehub...")
    try:
        import kagglehub
        path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")
        print(f"Spam dataset cached successfully at: {path}")
        for file_name in os.listdir(path):
            src_file = os.path.join(path, file_name)
            dst_file = os.path.join(target_dir, f"spam_{file_name}")
            if os.path.isfile(src_file) and not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)
                print(f"Transferred spam file: {file_name} -> {dst_file}")
        print(f"Spam dataset successfully placed in {target_dir}")
        return True
    except Exception as e:
        print(f"Notice regarding spam dataset acquisition: {e}")
        return False

if __name__ == "__main__":
    download_jigsaw_dataset()
    download_spam_dataset()
