"""Data access modes for the Streamlit interface."""

import os
import sqlite3

import requests

API_MODE = "api"
DIRECT_MODE = "direct"


class FrontendBackendError(RuntimeError):
    """A readable error that the Streamlit page can show to the user."""


def selected_mode() -> str:
    """Return the explicitly configured Streamlit backend mode."""
    mode = os.getenv("ECOSORT_STREAMLIT_MODE", API_MODE).strip().lower()
    if mode not in {API_MODE, DIRECT_MODE}:
        raise ValueError("ECOSORT_STREAMLIT_MODE must be either 'api' or 'direct'.")
    return mode


def _api_error_detail(response) -> str:
    try:
        error_body = response.json()
    except ValueError:
        return "Unexpected API error"
    if isinstance(error_body, dict) and isinstance(error_body.get("detail"), str):
        return error_body["detail"]
    return "Unexpected API error"


class ApiBackend:
    """Call the existing FastAPI endpoints over HTTP."""

    def __init__(self, api_url: str | None = None):
        base_url = api_url or os.getenv("ECOSORT_API_URL", "http://127.0.0.1:8000")
        self.api_url = base_url.rstrip("/")

    def predict(self, contents: bytes, filename: str, content_type: str) -> dict:
        try:
            response = requests.post(
                f"{self.api_url}/predict",
                files={"file": (filename, contents, content_type)},
                timeout=30,
            )
        except requests.exceptions.ConnectionError as error:
            raise FrontendBackendError(
                "The API server is not running. From the project root, run: "
                "uvicorn api.main:app --reload"
            ) from error
        except requests.exceptions.Timeout as error:
            raise FrontendBackendError(
                "The API took too long to respond. Please try again."
            ) from error
        except requests.exceptions.RequestException as error:
            raise FrontendBackendError(
                "Could not reach the API. Please try again."
            ) from error

        if response.status_code != 200:
            raise FrontendBackendError(
                f"The API returned an error ({response.status_code}): "
                f"{_api_error_detail(response)}"
            )
        try:
            return response.json()
        except ValueError as error:
            raise FrontendBackendError(
                "The API returned an unexpected response. Please try again."
            ) from error

    def dashboard(self, limit: int = 10) -> tuple[list[dict], dict[str, int]]:
        try:
            history = requests.get(
                f"{self.api_url}/history", params={"limit": limit}, timeout=10
            )
            analytics = requests.get(f"{self.api_url}/analytics", timeout=10)
        except requests.exceptions.ConnectionError as error:
            raise FrontendBackendError(
                "Start the FastAPI server to view prediction history and analytics."
            ) from error
        except requests.exceptions.Timeout as error:
            raise FrontendBackendError(
                "The API took too long to load prediction history."
            ) from error
        except requests.exceptions.RequestException as error:
            raise FrontendBackendError(
                "Could not load prediction history. Please try again."
            ) from error

        if history.status_code != 200 or analytics.status_code != 200:
            raise FrontendBackendError(
                "The API could not load prediction history and analytics."
            )
        try:
            return history.json()["predictions"], analytics.json()["counts"]
        except (ValueError, KeyError, TypeError) as error:
            raise FrontendBackendError(
                "The API returned unexpected history or analytics data."
            ) from error


class DirectBackend:
    """Use the shared inference and SQLite layers without starting FastAPI."""

    def predict(self, contents: bytes, filename: str, _content_type: str) -> dict:
        from api import inference, prediction_service

        try:
            return prediction_service.predict_and_save(contents, filename)
        except (
            inference.InvalidImageError,
            inference.ModelUnavailableError,
            inference.PredictionError,
        ) as error:
            raise FrontendBackendError(str(error)) from error
        except sqlite3.Error as error:
            raise FrontendBackendError(
                "Prediction succeeded, but it could not be saved."
            ) from error

    def dashboard(self, limit: int = 10) -> tuple[list[dict], dict[str, int]]:
        from api import database, inference

        try:
            predictions = database.get_recent_predictions(limit)
            stored_counts = database.get_prediction_counts()
        except sqlite3.Error as error:
            raise FrontendBackendError(
                "Prediction history is temporarily unavailable."
            ) from error
        counts = {
            class_name: stored_counts.get(class_name, 0)
            for class_name in inference.CLASS_NAMES
        }
        return predictions, counts


def get_backend(mode: str | None = None):
    """Create the configured backend used by the Streamlit page."""
    mode = mode or selected_mode()
    if mode == API_MODE:
        return ApiBackend()
    if mode == DIRECT_MODE:
        return DirectBackend()
    raise ValueError("Streamlit backend mode must be either 'api' or 'direct'.")
