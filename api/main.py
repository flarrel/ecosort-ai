"""EcoSort AI inference API (Milestone 3).

Run from the project root:
    uvicorn api.main:app --reload
"""

from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

# Same order as Keras image_dataset_from_directory / train_ds.class_names
# in notebooks/01_baseline.ipynb (alphabetical folder names).
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_SIZE = (224, 224)

# Project root is one level above this file (api/main.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "ecosort_baseline.keras"

app = FastAPI(
    title="EcoSort AI",
    description="Upload a waste image and get a TrashNet class prediction.",
)

# Filled on first prediction (tests can also set this to a fake model).
_model = None


def get_model():
    """Load the trained Keras model once, then reuse it."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Trained model not found at {MODEL_PATH}. "
                    "Run notebooks/01_baseline.ipynb first."
                ),
            )
        _model = keras.models.load_model(MODEL_PATH)
    return _model


def prepare_image(image: Image.Image) -> np.ndarray:
    """Match the training input that goes into the saved model.

    Training did this:
      1. RGB images
      2. resize to 224x224 (Keras bilinear)
      3. pixel values in the 0–255 range
      4. MobileNetV2 preprocess_input *inside* the model

    We do steps 1–3 here and leave step 4 to the loaded model so we
    do not preprocess twice.
    """
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    batch = np.expand_dims(array, axis=0)
    return batch


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Classify one uploaded waste image."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image = Image.open(BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(
            status_code=400,
            detail="File is not a valid image. Please upload a JPEG or PNG.",
        )

    batch = prepare_image(image)
    model = get_model()
    probabilities = model.predict(batch, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = round(float(probabilities[predicted_index]), 4)
    scores = {
        class_name: round(float(score), 4)
        for class_name, score in zip(CLASS_NAMES, probabilities)
    }

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "scores": scores,
    }
