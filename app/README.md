# app/ — FastAPI Inference Backend

REST API that receives a hand wound photograph and returns a triage prediction.

## Structure

```
app/
├── __init__.py
├── config.py     # Paths, threshold, label mapping
├── model.py      # EfficientNetV2-S architecture definition
├── predict.py    # Image preprocessing + inference pipeline
└── main.py       # FastAPI app, routes, CORS, startup
```

## Running the API

From the project root:

```bash
# Install dependencies (first time)
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API loads `models/best_baseline_cnn.pt` at startup. If the file is missing, the server will fail to start with a clear error message.

Interactive docs: `http://localhost:8000/docs`

## Endpoints

### `GET /health`

Liveness check. Confirms the model has loaded.

```json
{ "status": "ok", "model_loaded": true, "device": "cpu" }
```

### `POST /predict`

Upload a wound photograph (`multipart/form-data`, field `file`). Returns triage prediction.

**Accepted formats:** JPEG, PNG, WebP

**Response:**
```json
{
  "filename": "wound.jpg",
  "prediction": 1,
  "label": "oui",
  "confidence": 0.8723,
  "probability_exploration": 0.8723,
  "threshold_used": 0.35,
  "model": "EfficientNetV2-S (baseline CNN)"
}
```

- `prediction`: 0 = suture simple, 1 = exploration chirurgicale
- `label`: `"non"` or `"oui"`
- `confidence`: probability of the predicted class
- `probability_exploration`: raw probability of class 1 (before threshold)

## Configuration — `config.py`

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/best_baseline_cnn.pt` | Path to model weights |
| `DECISION_THRESHOLD` | `0.35` | Youden-optimised classification threshold |
| `IMG_SIZE` | `384` | Input image size (pixels) |
| `NUM_CLASSES` | `2` | Number of output classes |
| `ID2LABEL` | `{0: "non", 1: "oui"}` | Class index → label mapping |

## Inference pipeline

1. Image is received as an uploaded file and decoded to PIL RGB.
2. Resized to 384 × 384 and normalised with ImageNet mean/std.
3. Forward pass through EfficientNetV2-S → logits → softmax probabilities.
4. `P(class=1)` is compared against `DECISION_THRESHOLD` to produce the binary prediction.

## Notes

- **CORS** is currently set to `allow_origins=["*"]`. Restrict this before any production deployment.
- **Device** is forced to CPU for portability. To use GPU, update the `DEVICE` variable in `main.py`.
- The `ID2LABEL` mapping must match `sorted(df['target'].unique())` from training. The current mapping (`non`=0, `oui`=1) is correct for the `metadata_vdef.csv` dataset.
