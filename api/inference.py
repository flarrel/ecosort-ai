"""Shared image preprocessing and model inference for EcoSort."""

import os
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

# Same alphabetical order produced by the training dataset in the notebook.
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_SIZE = (224, 224)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(
    os.getenv(
        "ECOSORT_MODEL_PATH",
        PROJECT_ROOT / "model" / "ecosort_baseline.keras",
    )
).expanduser()


class InvalidImageError(ValueError):
    """Raised when uploaded bytes cannot be decoded as an image."""


class ModelUnavailableError(RuntimeError):
    """Raised when the trained model cannot be loaded."""


class PredictionError(RuntimeError):
    """Raised when the loaded model cannot produce a valid prediction."""


_model = None


def get_model():
    """Load the trained Keras model once per process, then reuse it."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise ModelUnavailableError(
                f"Trained model not found at {MODEL_PATH}. "
                "Set ECOSORT_MODEL_PATH to an available model file."
            )
        try:
            _model = keras.models.load_model(MODEL_PATH)
        except Exception as error:
            raise ModelUnavailableError(
                "The trained model could not be loaded."
            ) from error
    return _model


def prepare_image(image: Image.Image) -> np.ndarray:
    """Prepare one image exactly as expected by the saved model."""
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_image(contents: bytes) -> dict:
    """Decode image bytes and return the class, confidence, and scores."""
    if not contents:
        raise InvalidImageError("Uploaded file is empty.")

    try:
        image = Image.open(BytesIO(contents))
        image.load()
    except (UnidentifiedImageError, OSError) as error:
        raise InvalidImageError(
            "File is not a valid image. Please upload a JPEG or PNG."
        ) from error

    batch = prepare_image(image)
    model = get_model()
    try:
        predictions = np.asarray(model.predict(batch, verbose=0))
        if predictions.shape != (1, len(CLASS_NAMES)):
            raise ValueError("Unexpected model output shape")
        probabilities = predictions[0]
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("Model output contains non-finite values")
    except Exception as error:
        raise PredictionError("Prediction failed. Please try again.") from error

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
