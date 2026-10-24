"""Basic checks for the Milestone 3 inference API."""

from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

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


def test_health():
    client = TestClient(api_main.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_class_and_scores():
    api_main._model = FakeModel()
    client = TestClient(api_main.app)

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


def test_predict_rejects_non_image():
    client = TestClient(api_main.app)
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", b"this is not an image", "text/plain")},
    )
    assert response.status_code == 400
    assert "not a valid image" in response.json()["detail"]
