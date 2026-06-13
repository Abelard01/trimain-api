"""
DIA'GNOS HAND — Prediction pipeline
Handles image preprocessing and model inference.
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from app.config import IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD, DECISION_THRESHOLD, ID2LABEL


preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


@torch.no_grad()
def predict(model: torch.nn.Module, tokenizer, image: Image.Image, device: torch.device, loc_recoded: str = "", meca: str = "") -> dict:
    """
    Run inference on a single PIL image.

    Returns:
        {
            "prediction": 0 or 1,
            "label": human-readable label,
            "confidence": float probability of the predicted class,
            "probability_exploration": float probability of class 1,
        }
    """
    img_t = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(img_t)
    probs = F.softmax(logits, dim=-1).cpu().squeeze()
    prob_exploration = probs[1].item()
    prediction = int(prob_exploration > DECISION_THRESHOLD)

    return {
        "prediction": prediction,
        "label": ID2LABEL[prediction],
        "confidence": round(probs[prediction].item(), 4),
        "probability_exploration": round(prob_exploration, 4),
    }
