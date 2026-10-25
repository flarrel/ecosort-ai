"""Checks for prediction, history, and analytics API behavior."""

import sqlite3
from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import database, inference
import api.main as api_main


def _png_bytes(size=(32, 32), color=(200, 40, 40)):
    """Make a tiny in-memory PNG so tests do not need a real photo."""
    image = Image.new("RGB", size, color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


class FakeModel:
    """Stand-in for the Keras model so tests do not load TensorFlow weights."""

    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 224, 224, 3)
        assert batch.dtype == np.float32
        # Highest score is plastic (index 4), matching CLASS_NAMES in api/main.py.
        return np.array([[0.05, 0.05, 0.05, 0.05, 0.70, 0.10]], dtype=np.float32)


class FailingModel:
    """Simulate an inference failure before a prediction can be saved."""

    def predict(self, batch, verbose=0):
        raise RuntimeError("Inference failed")


@pytest.fixture
def test_database_path(tmp_path, monkeypatch):
    """Point the database module at a disposable test location."""
    path = tmp_path / "test_ecosort.db"
    monkeypatch.setattr(database, "DB_PATH", path)
    return path


@pytest.fixture
def client(test_database_path):
    """Start the API with a fake model and temporary database."""
    previous_model = inference._model
    inference._model = FakeModel()
    with TestClient(api_main.app) as test_client:
        yield test_client
    inference._model = previous_model


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_database_is_created_in_temporary_directory(client, test_database_path):
    assert database.DB_PATH == test_database_path
    assert test_database_path.exists()
    assert test_database_path.name == "test_ecosort.db"


def test_predict_returns_class_and_scores(client):
    response = client.post(
        "/predict",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] == "plastic"
    assert body["confidence"] == 0.70
    assert set(body["scores"].keys()) == set(api_main.CLASS_NAMES)
    assert body["scores"]["plastic"] == 0.70


def test_successful_prediction_is_saved(client):
    client.post(
        "/predict",
        files={"file": ("bottle.png", _png_bytes(), "image/png")},
    )

    response = client.get("/history")
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 1
    assert predictions[0]["filename"] == "bottle.png"
    assert predictions[0]["predicted_class"] == "plastic"
    assert predictions[0]["confidence"] == 0.70
    assert predictions[0]["id"] == 1
    assert predictions[0]["timestamp"]


def test_predict_rejects_non_image_without_saving_it(client):
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"this is not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert "not a valid image" in response.json()["detail"]
    assert client.get("/history").json() == {"predictions": []}


def test_failed_prediction_is_not_saved(client):
    inference._model = FailingModel()

    response = client.post(
        "/predict",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Prediction failed. Please try again."
    assert client.get("/history").json() == {"predictions": []}


def test_missing_model_returns_clear_error_without_saving(client, tmp_path, monkeypatch):
    inference._model = None
    monkeypatch.setattr(inference, "MODEL_PATH", tmp_path / "missing.keras")

    response = client.post(
        "/predict",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert "Trained model not found" in response.json()["detail"]
    assert client.get("/history").json() == {"predictions": []}


def test_database_failure_returns_clear_error(client, monkeypatch):
    def fail_to_save(*args, **kwargs):
        raise sqlite3.OperationalError("database is unavailable")

    monkeypatch.setattr(database, "save_prediction", fail_to_save)
    response = client.post(
        "/predict",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Prediction succeeded, but it could not be saved."
    )


@pytest.mark.parametrize(
    ("endpoint", "database_function", "expected_detail"),
    [
        (
            "/history",
            "get_recent_predictions",
            "Prediction history is temporarily unavailable.",
        ),
        (
            "/analytics",
            "get_prediction_counts",
            "Prediction analytics are temporarily unavailable.",
        ),
    ],
)
def test_read_endpoints_return_clear_database_errors(
    client, monkeypatch, endpoint, database_function, expected_detail
):
    def fail_to_read(*args, **kwargs):
        raise sqlite3.OperationalError("database is unavailable")

    monkeypatch.setattr(database, database_function, fail_to_read)
    response = client.get(endpoint)

    assert response.status_code == 503
    assert response.json()["detail"] == expected_detail


def test_history_is_recent_first_and_respects_limit(client):
    for filename in ["first.png", "second.png", "third.png"]:
        client.post(
            "/predict",
            files={"file": (filename, _png_bytes(), "image/png")},
        )

    response = client.get("/history", params={"limit": 2})
    assert response.status_code == 200
    filenames = [item["filename"] for item in response.json()["predictions"]]
    assert filenames == ["third.png", "second.png"]


def test_analytics_returns_counts_for_all_classes(client):
    client.post(
        "/predict",
        files={"file": ("sample.png", _png_bytes(), "image/png")},
    )

    response = client.get("/analytics")
    assert response.status_code == 200
    body = response.json()
    assert body["total_predictions"] == 1
    assert set(body["counts"]) == set(api_main.CLASS_NAMES)
    assert body["counts"]["plastic"] == 1
    assert body["counts"]["glass"] == 0
