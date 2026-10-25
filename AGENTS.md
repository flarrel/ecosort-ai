# EcoSort AI — Agent Context

## Project Purpose

EcoSort AI is a waste image classification application built around the TrashNet dataset and a MobileNetV2 transfer-learning model.

The application provides model inference through FastAPI, an upload and results interface through Streamlit, and SQLite-backed prediction history and analytics.

Keep the implementation coherent, reliable, and proportional to the problem it solves.

Favor:
- simplicity
- clear architecture
- beginner-readable code
- verifiable behavior
- features that fit naturally into the project

Avoid:
- unnecessary abstractions
- overengineering
- unrelated technologies
- adding tools without a clear technical need

---

## Current Architecture

The intended architecture is:

TrashNet image dataset  
→ MobileNetV2 transfer-learning classifier  
→ FastAPI inference backend  
→ Streamlit frontend  
→ SQLite prediction history and simple analytics

The Streamlit frontend must call the FastAPI backend.

Do not duplicate model inference logic inside the frontend.

---

## Current Project Status

### Milestone 1 — COMPLETE
Project setup and dataset loading.

Implemented:
- Python virtual environment
- project folder structure
- TrashNet dataset
- TensorFlow/Keras dataset loader
- train/validation split
- image visualization
- Git configuration
- dataset excluded from Git

The dataset is stored under:

`data/raw/`

The actual image dataset is intentionally ignored by Git.

---

### Milestone 2 — COMPLETE
MobileNetV2 transfer-learning baseline.

Implemented:
- MobileNetV2 pretrained on ImageNet
- 224×224 input images
- frozen pretrained feature extractor
- six TrashNet classes
- model training
- validation evaluation
- accuracy/loss plots
- confusion matrix
- classification report
- saved Keras model

Model path:

`model/ecosort_baseline.keras`

Use the exact preprocessing and class ordering established during training for all inference code.

Do not guess or redefine class order independently.

---

### Milestone 3 — COMPLETE
FastAPI inference backend.

Implemented:
- `GET /health`
- `POST /predict`
- image upload
- image preprocessing
- model inference
- predicted class
- confidence
- scores for all six classes
- invalid-upload handling
- API tests

Main backend file:

`api/main.py`

The backend can be started with:

`uvicorn api.main:app --reload`

Swagger documentation is available at:

`http://127.0.0.1:8000/docs`

---

### Milestone 4 — COMPLETE
Streamlit frontend.

Implemented:
- Streamlit UI titled `EcoSort AI`
- short project description
- JPG/JPEG/PNG upload
- display uploaded image
- send image to FastAPI `POST /predict`
- show predicted class
- show confidence
- show all six class scores
- handle backend being offline
- handle API errors cleanly
- do not perform model inference directly inside Streamlit

The Streamlit frontend has been manually tested successfully.

---

### Milestone 5 — COMPLETE
SQLite prediction history and simple analytics.

Implemented:
- automatic SQLite database creation using Python's built-in `sqlite3`
- successful prediction history with timestamp, filename, predicted class, and confidence
- recent prediction history API endpoint
- prediction counts by waste class API endpoint
- Streamlit history table and class-count chart
- temporary test databases that do not affect runtime data
- runtime database excluded from Git

Milestone 5 has been manually tested successfully.

---

### Milestone 6 — COMPLETE
Testing and integration cleanup.

Implemented:
- expanded API tests for health, prediction, persistence, history, and analytics
- deterministic tests using a fake model and temporary SQLite databases
- coverage ensuring invalid and failed predictions are not stored
- clear HTTP errors for model, inference, and database failures
- safer, readable Streamlit error messages without raw server output
- model output validation and integration cleanup

Milestone 6 has been manually tested successfully.

---

### Milestone 7 — COMPLETE
Docker/containerization.

Implemented:
- separate FastAPI and Streamlit Dockerfiles
- Docker Compose setup for both services
- internal Streamlit-to-FastAPI communication through the `api` service name
- read-only model bind mount for FastAPI inference
- persistent named volume for SQLite prediction history
- runtime-specific container dependencies
- Docker build-context exclusions for datasets, caches, tests, and local artifacts
- concise Docker usage documentation

Milestone 7 has been manually tested successfully.

---

## Remaining Roadmap

### Milestone 8 — NEXT
README and technical documentation cleanup.

### Milestone 9 — Deployment

Expected direction:
- deploy the existing application to SnapDeploy without adding product features
- configure model access and document SQLite's ephemeral hosted storage
- verify the API, Streamlit interface, history, and analytics in the deployed environment

### Milestone 10 — Final release

Expected direction:
- run final automated and manual validation
- verify documentation, commands, model results, and limitations
- add or refine an architecture visualization only if it improves understanding
- remove stale files or wording and prepare a clean software release

---

## GenAI / RAG Policy

LLM, RAG, LangChain, LlamaIndex, or similar features are optional.

Do not add them without a clear technical need.

They should only be added if:
- they remain simple,
- they fit the product naturally,
- and they materially improve the project.

---

## Git Rules

The repository should contain:
- source code
- notebooks
- tests
- configuration
- documentation

Do not commit:
- the full TrashNet dataset
- virtual environments
- temporary Jupyter files
- unnecessary large generated artifacts

Respect `.gitignore`.

Before committing a milestone:
1. inspect `git status`
2. inspect relevant diffs
3. manually test the feature
4. run tests
5. only then commit

At the end of each completed milestone, update this file’s Current Project Status and Remaining Roadmap. Do not mark a milestone complete until it has been manually tested successfully.

Do not commit automatically unless explicitly asked.

---

## Coding Style

Keep code:
- beginner-readable
- explicit
- lightly commented
- modular only where useful
- free from unnecessary abstractions

Prefer straightforward Python over elaborate design patterns.

Do not refactor working code merely to make it more sophisticated.

---

## Project-facing Documentation

Describe EcoSort AI in terms of the software itself: its technical purpose, implementation, behavior, results, architecture, setup, and limitations.

Project-facing files must not discuss university, coursework, assessments, lecturers, rubrics, portfolios, recruiting, employment, job searches, Sustainable Development Goals, academic reports or presentations, or any other external motivation for creating the project.

Do not replace that framing with labels such as personal project or passion project, and do not invent commercial motivation, users, adoption, impact, or business claims.

Keep documentation concise and proportional to the project. Include only material that helps someone understand, run, validate, or maintain the software.

---

## Instructions for Coding Agents

Before making changes:

1. Read this file.
2. Inspect the current repository.
3. Inspect existing uncommitted changes.
4. Understand the current milestone.
5. Preserve correct existing work.
6. Only modify what is necessary for the requested milestone.

When finishing a task:
- summarize files changed
- explain major implementation decisions
- mention any unresolved issues
- do not begin the next milestone unless explicitly requested

Never assume a milestone is complete merely because code exists.

A milestone is complete only after it has been manually tested successfully.
