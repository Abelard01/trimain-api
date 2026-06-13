"""
Downloads model weights from Google Drive if not present locally.
"""
import os
from pathlib import Path

MODEL_URL = "https://drive.google.com/uc?export=download&id=13ce38xSBoYoTgy5oUyEsoQBNUOfN2qEp"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best_baseline_cnn.pt"


def download_model():
    if MODEL_PATH.exists():
        print(f"Model already present at {MODEL_PATH}")
        return

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading model weights from Google Drive...")

    try:
        import gdown
        gdown.download(MODEL_URL, str(MODEL_PATH), quiet=False, fuzzy=True)
        print(f"Model downloaded to {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")


if __name__ == "__main__":
    download_model()
