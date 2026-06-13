"""
DIA'GNOS HAND — FastAPI backend
Receives a wound photograph, returns triage prediction.
"""
import io

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app.model import load_model
from app.predict import predict
from app.config import MODEL_PATH, ID2LABEL, DECISION_THRESHOLD
from app.download_model import download_model

# ── Device ──
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")  # Force CPU for compatibility and simplicity

# ── App ──
app = FastAPI(
    title="DIA'GNOS HAND API",
    description="IA d'aide au triage des plaies de la main — Hackathon Hacking Health 2026",
    version="0.1.0",
)

# ── CORS — allow the frontend to call the API ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model at startup (once) ──
model = None
tokenizer = None


@app.on_event("startup")
def startup():
    global model
    global tokenizer
    download_model()
    model, tokenizer = load_model(DEVICE)
    print(f"Model loaded on {DEVICE}")


# ── Routes ──

@app.get("/health")
def health():
    """Liveness / readiness check."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...), loc_recoded: str = Form(""), meca: str = Form("")):
    """
    Upload a wound photograph → receive triage prediction.

    **Returns**
    - `prediction`: 0 (suture simple) or 1 (exploration chirurgicale)
    - `label`: human-readable label
    - `confidence`: model confidence for the predicted class
    - `probability_exploration`: probability of class 1
    """
    print("==================================================")
    print("RECEIVED TRAFFIC: /predict")
    # Validate content type
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté : {file.content_type}. Envoyez JPEG, PNG ou WebP.",
        )

    # Read and decode image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire l'image.")

    # Inference
    result = predict(model, tokenizer, image, DEVICE, loc_recoded, meca)
    print(f" {result['label']} (confidence: {result['confidence']})")
    print(f"loc string: {loc_recoded} | meca string: {meca}")
    print("\n\n==================================================")
    return {
        "filename": file.filename,
        **result,
        "threshold_used": DECISION_THRESHOLD,
        "model": "EfficientNetV2-S",
        "loc_recoded": loc_recoded,
    }
