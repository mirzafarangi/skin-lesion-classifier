# Skin Lesion Classifier

Medical image classification for dermatological conditions using PyTorch, with MLflow experiment tracking and GradCAM explainability.

## Overview

This project demonstrates a production-ready approach to medical image classification:

- **PyTorch** with MPS acceleration (Apple Silicon)
- **MLflow** for experiment tracking and model versioning
- **GradCAM** for model explainability and uncertainty visualization
- **FastAPI** for model serving

## Dataset

Uses the [ISIC 2018 Challenge](https://challenge.isic-archive.com/landing/2018/) dataset for skin lesion classification. The model classifies dermoscopic images into diagnostic categories.

## Project Structure

```
skin-lesion-classifier/
├── train.py              # Model training with MLflow logging
├── inference.py          # FastAPI inference endpoint
├── explainability.py     # GradCAM visualization
├── model.py              # Model architecture
├── data_loader.py        # Dataset handling
├── config.py             # Configuration
├── requirements.txt
└── models/               # Saved model weights
```

## Installation

```bash
# Create environment
conda create -n skin-classifier python=3.10
conda activate skin-classifier

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Train model with MLflow tracking
python train.py --epochs 10 --batch_size 16

# View MLflow UI
mlflow ui
```

### Inference API

```bash
# Start FastAPI server
uvicorn inference:app --reload

# Test endpoint
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_image.jpg"
```

### Explainability

```bash
# Generate GradCAM visualization
python explainability.py --image sample_image.jpg --output gradcam_output.png
```

## Model Architecture

Uses EfficientNet-B0 pretrained on ImageNet, fine-tuned for skin lesion classification:

- Input: 224x224 RGB dermoscopic images
- Output: Classification probabilities + uncertainty estimate
- Explainability: GradCAM heatmaps highlighting diagnostic regions

## MLflow Tracking

Experiments are logged with:
- Training/validation metrics (accuracy, loss, AUC)
- Model hyperparameters
- Model artifacts and checkpoints
- Confusion matrices

## Explainability

GradCAM visualizations show which image regions the model focuses on for its predictions, critical for medical AI transparency and regulatory compliance.

## Hardware

Optimized for Apple Silicon (M1/M2/M3) using MPS acceleration. Also supports CUDA GPUs and CPU fallback.

## License

MIT License
