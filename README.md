# EcoSort AI

A small university / portfolio project for **waste image classification**.

This repository currently covers **Milestone 1**: a Python machine-learning environment and a baseline notebook that loads and previews the dataset. Training, API, UI, and deployment come later.

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

# 3. Upgrade pip, then install Milestone 1 packages
python -m pip install --upgrade pip
pip install -r requirements.txt
```

To leave the environment later:

```bash
deactivate
```

## Dataset location

Put your images under `data/raw/`, **one folder per class**. Keras will treat each folder name as a class label.

Example:

```text
data/raw/
  plastic/
    bottle1.jpg
    bottle2.jpg
  paper/
    newspaper1.jpg
  organic/
    banana_peel1.jpg
```

Do **not** put images directly in `data/raw/` (they must sit inside class folders).

`data/processed/` is reserved for later cleaning or exported datasets. You can leave it empty for now.

## Baseline notebook

With the virtual environment activated:

```bash
jupyter notebook notebooks/01_baseline.ipynb
```

The notebook:

- loads images from `data/raw/`
- resizes them to 224×224
- splits them into training and validation sets
- prints class names, shows example images, and prints batch counts

It does **not** train a model yet.
