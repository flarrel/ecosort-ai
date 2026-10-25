# EcoSort AI

EcoSort AI is an end-to-end waste image classification application. A user uploads a photo through a Streamlit interface, shared inference code classifies it with a MobileNetV2 model, and successful predictions are stored in SQLite for recent-history and class-count views. Local and Docker runs access that functionality through FastAPI.

The repository contains the complete workflow: dataset preparation, transfer learning, model evaluation, API inference, frontend integration, persistence, testing, and containerization.

## Problem

Waste images can contain visually similar materials that are difficult to categorize consistently. EcoSort accepts one image and predicts one of six TrashNet classes while exposing the model's confidence and all class scores. It is an image-classification tool, not a replacement for local disposal guidance or an industrial sorting system.

## Main features

- Classifies JPG, JPEG, and PNG images as cardboard, glass, metal, paper, plastic, or trash.
- Returns a predicted class, confidence, and scores for all six classes.
- Reuses one inference implementation across FastAPI and hosted Streamlit.
- Shows predictions through a simple Streamlit interface.
- Stores successful predictions in SQLite.
- Displays recent prediction history and prediction counts by class.
- Handles invalid uploads, unavailable services, and inference/database errors.
- Includes isolated API tests and a two-service Docker Compose setup.

## Tech stack

| Area | Technology |
| --- | --- |
| Model training and inference | TensorFlow/Keras, MobileNetV2, NumPy, Pillow |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit, Requests |
| Persistence | Python `sqlite3`, SQLite |
| Evaluation and development | scikit-learn, Matplotlib, pandas, Jupyter |
| Testing | pytest, FastAPI TestClient |
| Containers | Docker, Docker Compose |

## Architecture

```text
User uploads an image
        ↓
Streamlit frontend
        ├── local/Docker: HTTP → FastAPI
        └── Community Cloud: direct shared call
                            ↓
                 shared inference service
                            ↓
MobileNetV2 classifier
        ↓  successful prediction
SQLite history
        ↓
Frontend result, recent history, and class-count analytics
```

Local and Docker Streamlit runs call FastAPI for prediction, history, and analytics data. On Streamlit Community Cloud, a dedicated entrypoint calls the same shared inference and database layers directly, without starting another server. Both modes use identical preprocessing, model prediction, and persistence code.

## Dataset

The training notebook uses the TrashNet dataset: 2,527 images across six folders/classes:

```text
data/raw/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

Images are loaded at 224×224 with an 80/20 training-validation split. Dataset files under `data/raw/` are intentionally excluded from Git and are not needed for runtime inference.

## Model

The baseline is MobileNetV2 pretrained on ImageNet with its feature extractor frozen. The notebook adds global average pooling, dropout, and a six-class softmax output layer. MobileNetV2 preprocessing is included inside the saved model, preventing duplicate preprocessing in the API.

The included notebook trains for five epochs, evaluates with accuracy/loss plots, a confusion matrix, and a classification report, and achieved approximately **83.8% validation accuracy** in the recorded baseline run. It saves the trained model to:

```text
model/ecosort_baseline.keras
```

The trained model is tracked at this path so runtime environments can access it. Run the notebook again if the artifact needs to be regenerated.

## Project structure

```text
ecosort-ai/
├── api/
│   ├── database.py          # SQLite persistence helpers
│   ├── inference.py         # Shared preprocessing and model inference
│   ├── prediction_service.py # Shared prediction and persistence workflow
│   └── main.py              # FastAPI endpoints
├── app/
│   ├── backend.py           # API-backed and direct frontend adapters
│   └── streamlit_app.py     # Shared Streamlit interface
├── data/
│   ├── raw/                 # Local TrashNet dataset (Git-ignored)
│   └── ecosort.db           # Local runtime database (Git-ignored)
├── model/
│   └── ecosort_baseline.keras  # Trained inference model
├── notebooks/
│   └── 01_baseline.ipynb    # Training and evaluation workflow
├── tests/
│   └── test_api.py          # Isolated API and persistence tests
├── Dockerfile.api
├── Dockerfile.streamlit
├── community_cloud.py       # Community Cloud direct-mode entrypoint
├── docker-compose.yml
├── requirements-api.txt     # FastAPI container dependencies
├── requirements-streamlit.txt  # Streamlit container dependencies
├── requirements.txt
└── streamlit_app.py         # Local API-backed entrypoint
```

## Local setup (without Docker)

Use Python 3.12, matching the supported Community Cloud runtime and project dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the trained model is missing, place TrashNet in the class folders shown above and run:

```bash
jupyter notebook notebooks/01_baseline.ipynb
```

### Run FastAPI

From the project root with the virtual environment activated:

```bash
uvicorn api.main:app --reload
```

- API base URL: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs

### Run Streamlit

Keep FastAPI running and use a second terminal:

```bash
source .venv/bin/activate
python -m streamlit run streamlit_app.py
```

Open http://localhost:8501 and upload a JPG, JPEG, or PNG image.

## Docker setup

Docker requires the trained model at `model/ecosort_baseline.keras`. The raw training dataset is not copied into either image.

Build and start FastAPI and Streamlit together:

```bash
docker compose up --build
```

- Streamlit: http://localhost:8501
- FastAPI documentation: http://localhost:8000/docs

Inside Docker, Streamlit uses `http://api:8000`, where `api` is the Compose service name. The model directory is mounted read-only into FastAPI.

Stop the services while preserving prediction history:

```bash
docker compose down
```

To also delete the persisted Docker SQLite volume:

```bash
docker compose down -v
```

## Streamlit Community Cloud deployment

Community Cloud runs the dedicated entrypoint:

```text
community_cloud.py
```

This entrypoint explicitly selects direct mode. It does not start FastAPI or Uvicorn; instead, it calls the same inference, prediction, and SQLite modules used by FastAPI. The tracked `model/ecosort_baseline.keras` artifact and root `requirements.txt` provide the runtime model and dependencies.

In Community Cloud, select Python 3.12 and leave the app URL configuration unset. For local direct-mode verification, run `python -m streamlit run community_cloud.py`.

Hosted SQLite uses `data/ecosort.db`, but Community Cloud storage is ephemeral, so prediction history can be lost when the app restarts or is redeployed. Local Docker Compose persistence remains unchanged through the `ecosort_data` volume.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check whether the API is running. |
| `POST` | `/predict` | Upload one image in a multipart field named `file`. |
| `GET` | `/history?limit=20` | Return recent successful predictions, newest first. `limit` accepts 1–100. |
| `GET` | `/analytics` | Return total predictions and counts for all six classes. |

Example prediction request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@/path/to/waste_image.jpg"
```

## SQLite persistence

On a local run, the API automatically creates `data/ecosort.db`. Each successful prediction stores an ID, UTC timestamp, uploaded filename, predicted class, and confidence. Invalid uploads and failed predictions are not stored.

With Docker Compose, `/app/data` uses the named volume `ecosort_data`, so history survives container recreation and `docker compose down` unless the volume is explicitly removed with `-v`.

## Testing

Run the complete automated test suite from the project root:

```bash
pytest
```

Tests use a lightweight fake model and a fresh temporary SQLite database. They do not require the trained model and do not modify the local runtime database.

## Known limitations

- The model is trained on a small, controlled dataset and may perform poorly on cluttered backgrounds, unusual lighting, damaged objects, or materials unlike TrashNet.
- Classification is limited to six classes and one image-level label; the app does not detect or separate multiple objects.
- Confidence scores are model probabilities, not a guarantee that a disposal decision is correct.
- SQLite is suitable for this local single-application prototype, not a high-concurrency production system.
- The training dataset is not stored in Git; only the trained inference model is included.
- The application is not currently deployed and does not include authentication.

## Future improvements

- Deploy the existing application with suitable model access and persistent data storage.
- Evaluate with additional real-world waste images and document failure cases.
- Improve model quality only where evaluation shows a clear need, for example through careful fine-tuning or better data coverage.
- Complete final validation, cleanup, documentation verification, and release preparation.

## Current project status

Milestones 1–7 are complete and manually verified: dataset setup, MobileNetV2 training, FastAPI inference, Streamlit UI, SQLite history/analytics, integration testing, and Docker containerization.

Milestone 8—README and technical documentation cleanup—is implemented and awaiting manual verification. Deployment and final release preparation remain future milestones.
