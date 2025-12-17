"""Configuration for skin lesion classifier."""

import torch
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Model
MODEL_NAME = "efficientnet_b0"
NUM_CLASSES = 7  # ISIC 2018 has 7 diagnostic categories
IMAGE_SIZE = 224

# Training
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
TRAIN_SPLIT = 0.8

# Device - MPS for Apple Silicon, CUDA for GPU, CPU fallback
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()

# Class names (ISIC 2018)
CLASS_NAMES = [
    "MEL",   # Melanoma
    "NV",    # Melanocytic nevus
    "BCC",   # Basal cell carcinoma
    "AKIEC", # Actinic keratosis
    "BKL",   # Benign keratosis
    "DF",    # Dermatofibroma
    "VASC"   # Vascular lesion
]

# MLflow
MLFLOW_EXPERIMENT_NAME = "skin-lesion-classifier"
MLFLOW_TRACKING_URI = "mlruns"
