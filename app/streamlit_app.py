"""EcoSort AI Streamlit frontend.

Local mode calls FastAPI by default:
    python -m streamlit run streamlit_app.py
"""

import streamlit as st

from app.backend import FrontendBackendError, get_backend, selected_mode


def _validated_result(result: dict) -> tuple[str, float, dict[str, float]]:
    """Validate prediction data before rendering it."""
    try:
        predicted_class = result["predicted_class"]
        confidence = float(result["confidence"])
        scores = result["scores"]
        if not isinstance(predicted_class, str) or not isinstance(scores, dict):
            raise ValueError
        scores = {name: float(score) for name, score in scores.items()}
        if len(scores) != 6:
            raise ValueError
    except (ValueError, KeyError, TypeError) as error:
        raise FrontendBackendError(
            "Prediction returned an unexpected result. Please try again."
        ) from error
    return predicted_class, confidence, scores


def render_app() -> None:
    """Render the same UI with either the API or direct backend."""
    st.set_page_config(page_title="EcoSort AI", layout="centered")
    backend = get_backend(selected_mode())

    st.title("EcoSort AI")
    st.write(
        "This app classifies a waste photo into one of six types: cardboard, "
        "glass, metal, paper, plastic, or trash. Upload an image to get a prediction."
    )

    uploaded_file = st.file_uploader(
        "Upload a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("Choose an image to get started.")
    elif uploaded_file.size == 0:
        st.error("That file is empty. Please upload a valid JPG or PNG image.")
    else:
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
        if st.button("Classify image"):
            try:
                result = backend.predict(
                    uploaded_file.getvalue(),
                    uploaded_file.name,
                    uploaded_file.type or "application/octet-stream",
                )
                predicted_class, confidence, scores = _validated_result(result)
            except FrontendBackendError as error:
                st.error(str(error))
            else:
                st.success(f"Predicted class: **{predicted_class}**")
                st.write(f"Confidence: **{confidence:.2%}**")
                st.subheader("Scores for all classes")
                for class_name, score in scores.items():
                    st.write(f"- {class_name}: {score:.2%}")

    st.divider()
    st.header("Prediction history")
    try:
        predictions, counts = backend.dashboard(limit=10)
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
    except FrontendBackendError as error:
        st.error(str(error))
    except (ValueError, KeyError, TypeError):
        st.error("Prediction history returned unexpected data.")
    else:
        if predictions:
            st.dataframe(history_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No successful predictions have been saved yet.")
        st.subheader("Prediction counts by class")
        st.bar_chart(chart_data, x="Waste class", y="Predictions")


if __name__ == "__main__":
    render_app()
