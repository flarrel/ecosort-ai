"""EcoSort AI inference API.

Run from the project root:
    uvicorn api.main:app --reload
"""

import sqlite3
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from api import database, inference, prediction_service

CLASS_NAMES = inference.CLASS_NAMES


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Create the SQLite database when the API starts."""
    database.initialize_database()
    yield


app = FastAPI(
    title="EcoSort AI",
    description="Upload a waste image and get a TrashNet class prediction.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Classify one uploaded waste image."""
    try:
        return prediction_service.predict_and_save(
            contents=await file.read(),
            filename=file.filename or "unknown",
        )
    except inference.InvalidImageError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error
    except inference.ModelUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except inference.PredictionError as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        ) from error
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=503,
            detail="Prediction succeeded, but it could not be saved.",
        ) from error


@app.get("/history")
def history(limit: int = Query(default=20, ge=1, le=100)):
    """Return recent successful predictions, newest first."""
    try:
        predictions = database.get_recent_predictions(limit)
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=503,
            detail="Prediction history is temporarily unavailable.",
        ) from error
    return {"predictions": predictions}


@app.get("/analytics")
def analytics():
    """Return prediction counts for every waste class."""
    try:
        stored_counts = database.get_prediction_counts()
    except sqlite3.Error as error:
        raise HTTPException(
            status_code=503,
            detail="Prediction analytics are temporarily unavailable.",
        ) from error
    counts = {class_name: stored_counts.get(class_name, 0) for class_name in CLASS_NAMES}
    return {
        "total_predictions": sum(counts.values()),
        "counts": counts,
    }
