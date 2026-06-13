"""
DIA'GNOS HAND — Configuration
"""
from pathlib import Path

# ── Paths ──
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_baseline_cnn.pt"

# ── Model ──
NUM_CLASSES = 2
IMG_SIZE = 224
DECISION_THRESHOLD = 0.35  # Seuil Youden optimisé (cf. notebook)

# ── Labels ──
# Mapping construit dynamiquement dans le notebook via sorted(df['target'].unique())
# Adapter ces valeurs aux labels exacts de votre dataset.
# Classe 0 = premier label alphabétique, classe 1 = second.
ID2LABEL = {
    0: "non", # urgentiste peut gérer
    1: "oui", # transfert chirurgien spécialisé
}

# ── ImageNet normalization (EfficientNetV2-S DEFAULT weights) ──
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGENET_STD = [0.26862954, 0.26130258, 0.27577711]
