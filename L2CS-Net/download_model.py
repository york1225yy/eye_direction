"""
Download L2CS-Net pretrained model.
Download priority:
  1. HuggingFace Hub mirror (hf-mirror.com) — AutoDL / China accessible
  2. HuggingFace Hub official (huggingface.co)
  3. Google Drive (gdown) — blocked in mainland China
  4. Manual upload instructions

Model: L2CSNet_gaze360.pkl (~216 MB)
HuggingFace: https://huggingface.co/oraclex/L2CS-Net
Google Drive: https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd
"""

import os
import sys
import pathlib
import argparse

MODEL_DIR  = pathlib.Path(__file__).parent / "models"
MODEL_PATH = MODEL_DIR / "L2CSNet_gaze360.pkl"

# HuggingFace repo info
HF_REPO_ID   = "oraclex/L2CS-Net"
HF_FILENAME  = "L2CSNet_gaze360.pkl"

# Google Drive file ID (fallback)
GDRIVE_FILE_ID = "1E-HY1E2lHmyGsNSTFsQBHi2YzTYRj_0e"

# China-accessible HuggingFace mirror
HF_MIRROR = "https://hf-mirror.com"


# ─── helpers ────────────────────────────────────────────────────────────────

def _ensure_pkg(pkg_name: str, import_name: str = None):
    import_name = import_name or pkg_name
    try:
        __import__(import_name)
    except ImportError:
        print(f"[INFO] Installing {pkg_name} ...")
        os.system(f"{sys.executable} -m pip install -q {pkg_name}")


def _model_ok() -> bool:
    if not MODEL_PATH.exists():
        return False
    size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    return size_mb >= 50


def _check_net(host: str, port: int = 443, timeout: int = 5) -> bool:
    """Quick TCP reachability check."""
    import socket
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except OSError:
        return False


# ─── download strategies ────────────────────────────────────────────────────

def _download_hf_mirror() -> bool:
    """Download via hf-mirror.com (CN-accessible HuggingFace mirror)."""
    if not _check_net("hf-mirror.com"):
        print("[INFO] hf-mirror.com unreachable, skipping.")
        return False

    _ensure_pkg("huggingface_hub")
    try:
        from huggingface_hub import hf_hub_download
        print(f"[INFO] Trying HuggingFace mirror: {HF_MIRROR}")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            cache_dir=str(MODEL_DIR / ".cache"),
            local_dir=str(MODEL_DIR),
            endpoint=HF_MIRROR,
        )
        # hf_hub_download returns the actual path
        dest = pathlib.Path(downloaded)
        if dest.resolve() != MODEL_PATH.resolve() and dest.exists():
            import shutil
            shutil.copy2(str(dest), str(MODEL_PATH))
        return _model_ok()
    except Exception as e:
        print(f"[WARN] HuggingFace mirror failed: {e}")
        return False


def _download_hf_official() -> bool:
    """Download via huggingface.co (official)."""
    if not _check_net("huggingface.co"):
        print("[INFO] huggingface.co unreachable, skipping.")
        return False

    _ensure_pkg("huggingface_hub")
    try:
        from huggingface_hub import hf_hub_download
        print("[INFO] Trying HuggingFace official ...")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            cache_dir=str(MODEL_DIR / ".cache"),
            local_dir=str(MODEL_DIR),
        )
        dest = pathlib.Path(downloaded)
        if dest.resolve() != MODEL_PATH.resolve() and dest.exists():
            import shutil
            shutil.copy2(str(dest), str(MODEL_PATH))
        return _model_ok()
    except Exception as e:
        print(f"[WARN] HuggingFace official failed: {e}")
        return False


def _download_gdrive() -> bool:
    """Download via Google Drive (gdown). May be blocked in CN."""
    if not _check_net("drive.google.com"):
        print("[INFO] drive.google.com unreachable (expected in mainland China), skipping.")
        return False

    _ensure_pkg("gdown")
    try:
        import gdown
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        print(f"[INFO] Trying Google Drive: {url}")
        gdown.download(url, str(MODEL_PATH), quiet=False)
        return _model_ok()
    except Exception as e:
        print(f"[WARN] Google Drive download failed: {e}")
        return False


def _print_manual_instructions():
    print()
    print("=" * 60)
    print("  [ERROR] All automatic downloads failed.")
    print("  Please download the model manually:")
    print()
    print("  Option A — Local machine + AutoDL web upload:")
    print("    1. Download on your local PC:")
    print("       https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd")
    print("    2. In AutoDL console → 'File Upload' → upload to instance")
    print(f"    3. Move file to: {MODEL_PATH}")
    print()
    print("  Option B — AutoDL JupyterLab upload:")
    print("    1. Open JupyterLab on your AutoDL instance")
    print(f"    2. Navigate to: {MODEL_DIR}")
    print("    3. Use the upload button (↑) to upload L2CSNet_gaze360.pkl")
    print()
    print("  Option C — scp from local machine:")
    print("    scp L2CSNet_gaze360.pkl root@<autodl-ip>:<port>:" + str(MODEL_PATH))
    print("=" * 60)


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    global MODEL_PATH, MODEL_DIR

    parser = argparse.ArgumentParser(description="Download L2CS-Net pretrained model")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Custom output path (default: models/L2CSNet_gaze360.pkl)"
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Force re-download even if model already exists"
    )
    args = parser.parse_args()

    if args.output:
        MODEL_PATH = pathlib.Path(args.output)
        MODEL_DIR  = MODEL_PATH.parent

    # ── already exists? ──
    if not args.force and _model_ok():
        size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
        print(f"[INFO] Model already exists and is valid ({size_mb:.1f} MB): {MODEL_PATH}")
        return

    if MODEL_PATH.exists() and not _model_ok():
        print(f"[WARN] Existing file is incomplete, removing and re-downloading...")
        MODEL_PATH.unlink()

    print(f"[INFO] Downloading model to: {MODEL_PATH}")
    print("[INFO] Trying download sources in order...")

    strategies = [
        ("HuggingFace mirror (hf-mirror.com)", _download_hf_mirror),
        ("HuggingFace official",               _download_hf_official),
        ("Google Drive",                        _download_gdrive),
    ]

    for name, fn in strategies:
        print(f"\n[INFO] >>> {name}")
        try:
            if fn():
                size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
                print(f"\n[OK] Download succeeded via {name}")
                print(f"[OK] Model saved: {MODEL_PATH} ({size_mb:.1f} MB)")
                # cleanup hf cache
                import shutil
                cache_dir = MODEL_DIR / ".cache"
                if cache_dir.exists():
                    shutil.rmtree(cache_dir, ignore_errors=True)
                return
        except Exception as e:
            print(f"[WARN] {name} raised exception: {e}")

    _print_manual_instructions()
    sys.exit(1)


if __name__ == "__main__":
    main()
