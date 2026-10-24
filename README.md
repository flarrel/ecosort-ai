# EcoSort AI

A small university / portfolio project for **waste image classification**.

This repository currently covers **Milestone 2**: load the TrashNet images, train a small MobileNetV2 transfer-learning baseline, evaluate it, and save the model. API, UI, and deployment come later.

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
