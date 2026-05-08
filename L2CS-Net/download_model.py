"""
Download L2CS-Net pretrained model from Google Drive.
Model: L2CSNet_gaze360.pkl (trained on Gaze360 dataset)
Google Drive folder: https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd
"""

import os
import sys
import pathlib

MODEL_DIR = pathlib.Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "L2CSNet_gaze360.pkl"

# Direct file ID from the Google Drive folder
FILE_ID = "1E-HY1E2lHmyGsNSTFsQBHi2YzTYRj_0e"


def check_gdown():
    try:
        import gdown
        return True
    except ImportError:
        return False


def download_with_gdown():
    import gdown
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print(f"[INFO] Downloading model with gdown...")
    print(f"[INFO] URL: {url}")
    gdown.download(url, str(MODEL_PATH), quiet=False)


def download_with_wget():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?export=download&id={FILE_ID}"
    print(f"[INFO] Downloading model with wget (may fail for large files)...")
    ret = os.system(f'wget -q --show-progress --no-check-certificate "{url}" -O "{MODEL_PATH}"')
    if ret != 0:
        print("[WARN] wget failed, trying gdown...")
        return False
    return True


def main():
    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"[INFO] Model already exists: {MODEL_PATH} ({size_mb:.1f} MB)")
        if size_mb < 50:
            print("[WARN] File seems too small, re-downloading...")
        else:
            print("[INFO] Model is ready.")
            return

    print(f"[INFO] Model not found. Downloading to {MODEL_PATH} ...")

    if check_gdown():
        download_with_gdown()
    else:
        print("[INFO] gdown not installed, installing...")
        os.system(f"{sys.executable} -m pip install -q gdown")
        download_with_gdown()

    if MODEL_PATH.exists():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        if size_mb < 50:
            print(f"[ERROR] Downloaded file is too small ({size_mb:.1f} MB). "
                  "Google Drive may have blocked the download.\n"
                  "Please manually download from:\n"
                  "https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd\n"
                  f"and place it at: {MODEL_PATH}")
            sys.exit(1)
        else:
            print(f"[INFO] Download successful! ({size_mb:.1f} MB) -> {MODEL_PATH}")
    else:
        print(f"[ERROR] Download failed. Please manually download the model from:\n"
              "https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd\n"
              f"and place it at: {MODEL_PATH}")
        sys.exit(1)


if __name__ == "__main__":
    main()
