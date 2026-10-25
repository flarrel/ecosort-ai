"""Checks for prediction, history, and analytics API behavior."""

from io import BytesIO

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import database
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
def client(tmp_path, monkeypatch):
    """Use a fresh temporary SQLite database for every test."""
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test_ecosort.db")
    api_main._model = FakeModel()
    with TestClient(api_main.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
    api_main._model = FailingModel()

    with pytest.raises(RuntimeError, match="Inference failed"):
        client.post(
            "/predict",
            files={"file": ("sample.png", _png_bytes(), "image/png")},
        )

    assert client.get("/history").json() == {"predictions": []}


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
