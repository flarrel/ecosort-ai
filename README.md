# EcoSort AI

A small university / portfolio project for **waste image classification**.

This repository currently covers **Milestone 4**: train a MobileNetV2 baseline on TrashNet, serve predictions with FastAPI, and classify images from a simple Streamlit UI.

## Requirements

- macOS
- Python 3.10 or 3.11 (recommended for TensorFlow)

## Setup

From the project root:

```bash
# 1. Create a virtual environment
python3 -m venv .venv

# 2. Activate it (macOS / zsh)
source .venv/bin/activate

# 3. Upgrade pip, then install packages
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To leave the environment later:

```bash
deactivate
```

## Dataset location

Put your images under `data/raw/`, **one folder per class**. Keras will treat each folder name as a class label.

TrashNet uses these 6 class folders:

```text
data/raw/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

Do **not** put images directly in `data/raw/` (they must sit inside class folders).

`data/processed/` is reserved for later cleaning or exported datasets. You can leave it empty for now.

## Baseline notebook

With the virtual environment activated:

```bash
jupyter notebook notebooks/01_baseline.ipynb
```

The notebook:

- loads images from `data/raw/` and resizes them to 224×224
- splits them into training and validation sets
- trains a frozen **MobileNetV2** (ImageNet) model with a small 6-class head
- plots accuracy/loss, prints a confusion matrix and classification report
- saves the model to `model/ecosort_baseline.keras`

First-time training will download ImageNet weights for MobileNetV2 (needs internet). A few epochs is enough for a baseline; expect several minutes on a Mac.

## Inference API

Train and save the model first (`model/ecosort_baseline.keras`). Then, from the project root with the virtual environment activated:

```bash
uvicorn api.main:app --reload
```

The API starts at http://127.0.0.1:8000 . Interactive docs: http://127.0.0.1:8000/docs

- `GET /health` — liveness check
- `POST /predict` — upload an image field named `file`

Example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@/path/to/waste_image.jpg"
```

The JSON response includes `predicted_class`, `confidence`, and `scores` for all six classes (`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`) in the same order used during training.

## Streamlit frontend

Keep the FastAPI server running in one terminal, then start the UI in a second terminal (project root, virtual environment activated):

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser. Upload a JPG/JPEG/PNG waste photo and click **Classify image**. The Streamlit app does not run the model itself; it sends the file to `POST /predict` and shows `predicted_class`, `confidence`, and the six class `scores`.

If the backend is not running, the UI will show an error instead of crashing.

## Tests

```bash
pytest tests/test_api.py
```
