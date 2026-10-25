"""Checks for shared inference and Streamlit backend selection."""

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from api import database, inference
from app import backend


class FakeModel:
    def predict(self, batch, verbose=0):
        assert batch.shape == (1, 224, 224, 3)
        return np.array([[0.05, 0.05, 0.05, 0.05, 0.70, 0.10]], dtype=np.float32)


def _png_bytes() -> bytes:
    image = Image.new("RGB", (32, 32), (200, 40, 40))
    contents = BytesIO()
    image.save(contents, format="PNG")
    return contents.getvalue()


def test_streamlit_mode_defaults_to_api(monkeypatch):
    monkeypatch.delenv("ECOSORT_STREAMLIT_MODE", raising=False)
    assert backend.selected_mode() == backend.API_MODE
    assert isinstance(backend.get_backend(), backend.ApiBackend)


def test_streamlit_direct_mode_is_explicit(monkeypatch):
    monkeypatch.setenv("ECOSORT_STREAMLIT_MODE", "direct")
    assert backend.selected_mode() == backend.DIRECT_MODE
    assert isinstance(backend.get_backend(), backend.DirectBackend)


def test_invalid_streamlit_mode_is_rejected(monkeypatch):
    monkeypatch.setenv("ECOSORT_STREAMLIT_MODE", "automatic")
    with pytest.raises(ValueError, match="must be either"):
        backend.selected_mode()


def test_model_is_loaded_only_once_per_process(tmp_path, monkeypatch):
    model_path = tmp_path / "model.keras"
    model_path.touch()
    model = FakeModel()
    load_calls = []

    def fake_load_model(path):
        load_calls.append(path)
        return model

    monkeypatch.setattr(inference, "_model", None)
    monkeypatch.setattr(inference, "MODEL_PATH", model_path)
    monkeypatch.setattr(inference.keras.models, "load_model", fake_load_model)

    assert inference.get_model() is model
    assert inference.get_model() is model
    assert load_calls == [model_path]


def test_direct_mode_predicts_and_uses_shared_database(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "direct.db")
    monkeypatch.setattr(inference, "_model", FakeModel())
    direct_backend = backend.DirectBackend()

    result = direct_backend.predict(_png_bytes(), "direct.png", "image/png")
    predictions, counts = direct_backend.dashboard()

    assert result["predicted_class"] == "plastic"
    assert result["confidence"] == 0.70
    assert predictions[0]["filename"] == "direct.png"
    assert counts["plastic"] == 1
    assert set(counts) == set(inference.CLASS_NAMES)
