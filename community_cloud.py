"""Streamlit Community Cloud entrypoint using the direct shared backend."""

import os

os.environ["ECOSORT_STREAMLIT_MODE"] = "direct"

from app.streamlit_app import render_app  # noqa: E402


render_app()
