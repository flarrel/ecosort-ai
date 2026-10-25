"""EcoSort AI Streamlit frontend (Milestone 4).

Start the FastAPI backend first, then from the project root:
    streamlit run app/streamlit_app.py
"""

import os

import requests
import streamlit as st

# Docker Compose sets this to http://api:8000. Local runs use localhost.
API_URL = os.getenv("ECOSORT_API_URL", "http://127.0.0.1:8000").rstrip("/")
PREDICT_URL = f"{API_URL}/predict"
HISTORY_URL = f"{API_URL}/history"
ANALYTICS_URL = f"{API_URL}/analytics"


def api_error_detail(response):
    """Return a readable API error without displaying raw server output."""
    try:
        error_body = response.json()
    except ValueError:
        return "Unexpected API error"

    if isinstance(error_body, dict) and isinstance(error_body.get("detail"), str):
        return error_body["detail"]
    return "Unexpected API error"


st.set_page_config(page_title="EcoSort AI", layout="centered")

st.title("EcoSort AI")
st.write(
    "This app classifies a waste photo into one of six types: "
    "cardboard, glass, metal, paper, plastic, or trash. "
    "Upload an image and EcoSort will ask the FastAPI backend for a prediction."
)

uploaded_file = st.file_uploader(
    "Upload a JPG, JPEG, or PNG image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Choose an image to get started.")
else:
    if uploaded_file.size == 0:
        st.error("That file is empty. Please upload a valid JPG or PNG image.")
    else:
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

        if st.button("Classify image"):
            image_bytes = uploaded_file.getvalue()
            files = {
                "file": (
                    uploaded_file.name,
                    image_bytes,
                    uploaded_file.type or "application/octet-stream",
                )
            }

            try:
                response = requests.post(PREDICT_URL, files=files, timeout=30)
            except requests.exceptions.ConnectionError:
                st.error(
                    "The API server is not running. In another terminal, from the "
                    "project root, run: uvicorn api.main:app --reload"
                )
            except requests.exceptions.Timeout:
                st.error("The API took too long to respond. Please try again.")
            except requests.exceptions.RequestException:
                st.error("Could not reach the API. Please try again.")
            else:
                if response.status_code != 200:
                    detail = api_error_detail(response)
                    st.error(
                        f"The API returned an error ({response.status_code}): {detail}"
                    )
                else:
                    try:
                        result = response.json()
                        predicted_class = result["predicted_class"]
                        confidence = float(result["confidence"])
                        scores = result["scores"]
                        if not isinstance(scores, dict) or len(scores) != 6:
                            raise ValueError
                        scores = {
                            class_name: float(score)
                            for class_name, score in scores.items()
                        }
                    except (ValueError, KeyError, TypeError):
                        st.error(
                            "The API returned an unexpected response. Please try again."
                        )
                    else:
                        st.success(f"Predicted class: **{predicted_class}**")
                        st.write(f"Confidence: **{confidence:.2%}**")

                        st.subheader("Scores for all classes")
                        for class_name, score in scores.items():
                            st.write(f"- {class_name}: {score:.2%}")

st.divider()
st.header("Prediction history")

try:
    history_response = requests.get(HISTORY_URL, params={"limit": 10}, timeout=10)
    analytics_response = requests.get(ANALYTICS_URL, timeout=10)
except requests.exceptions.ConnectionError:
    st.info("Start the FastAPI server to view prediction history and analytics.")
except requests.exceptions.Timeout:
    st.error("The API took too long to load prediction history.")
except requests.exceptions.RequestException:
    st.error("Could not load prediction history. Please try again.")
else:
    if history_response.status_code != 200 or analytics_response.status_code != 200:
        st.error("The API could not load prediction history and analytics.")
    else:
        try:
            predictions = history_response.json()["predictions"]
            counts = analytics_response.json()["counts"]
            if not isinstance(predictions, list) or not isinstance(counts, dict):
                raise ValueError
            history_rows = [
                {
                    "Timestamp": item["timestamp"],
                    "Filename": item["filename"],
                    "Predicted class": item["predicted_class"],
                    "Confidence": f"{float(item['confidence']):.2%}",
                }
                for item in predictions
            ]
            chart_data = {
                "Waste class": list(counts.keys()),
                "Predictions": [int(count) for count in counts.values()],
            }
        except (ValueError, KeyError, TypeError):
            st.error("The API returned unexpected history or analytics data.")
        else:
            if predictions:
                st.dataframe(history_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No successful predictions have been saved yet.")

            st.subheader("Prediction counts by class")
            st.bar_chart(chart_data, x="Waste class", y="Predictions")
