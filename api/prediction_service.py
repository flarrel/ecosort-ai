"""Shared prediction workflow used by FastAPI and direct Streamlit mode."""

from api import database, inference


def predict_and_save(contents: bytes, filename: str) -> dict:
    """Run one prediction and persist it only after inference succeeds."""
    result = inference.predict_image(contents)
    database.save_prediction(
        filename=filename or "unknown",
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
    )
    return result
