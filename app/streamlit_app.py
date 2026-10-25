"""EcoSort AI Streamlit frontend (Milestone 4).

Start the FastAPI backend first, then from the project root:
    streamlit run app/streamlit_app.py
"""

import requests
import streamlit as st

# Must match the running FastAPI server (see README).
API_URL = "http://127.0.0.1:8000"
PREDICT_URL = f"{API_URL}/predict"

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
            except requests.exceptions.RequestException as error:
                st.error(f"Could not reach the API: {error}")
            else:
                if response.status_code != 200:
                    try:
                        error_body = response.json()
                    except ValueError:
                        detail = response.text or "Unknown error"
                    else:
                        if isinstance(error_body, dict):
                            detail = error_body.get("detail", response.text)
                        else:
                            detail = response.text or "Unknown error"
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
