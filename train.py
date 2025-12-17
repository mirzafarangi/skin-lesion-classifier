"""Training script with MLflow tracking."""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
from pathlib import Path

from model import SkinLesionClassifier
from data_loader import get_data_loaders
from config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    MODEL_DIR, MLFLOW_EXPERIMENT_NAME, CLASS_NAMES
)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    # Calculate AUC (one-vs-rest)
    try:
        all_probs = np.array(all_probs)
        all_labels_onehot = np.eye(len(CLASS_NAMES))[all_labels]
        auc = roc_auc_score(all_labels_onehot, all_probs, average='macro', multi_class='ovr')
    except:
        auc = 0.0
    
    return epoch_loss, epoch_acc, auc


def train(
    image_dir: str,
    labels_csv: str,
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE
):
    """Main training function with MLflow tracking."""
    
    print(f"Using device: {DEVICE}")
    
    # Create model directory
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Setup MLflow
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "model": "efficientnet_b0",
            "device": str(DEVICE)
        })
        
        # Load data
        train_loader, val_loader = get_data_loaders(
            image_dir=image_dir,
            labels_csv=labels_csv,
            batch_size=batch_size
        )
        
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        
        # Initialize model
        model = SkinLesionClassifier().to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            
            # Validate
            val_loss, val_acc, val_auc = validate(
                model, val_loader, criterion, DEVICE
            )
            
            # Update scheduler
            scheduler.step()
            
            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "learning_rate": scheduler.get_last_lr()[0]
            }, step=epoch)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_path = MODEL_DIR / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_auc': val_auc
                }, checkpoint_path)
                print(f"Saved best model with val_acc: {val_acc:.4f}")
        
        # Log final model to MLflow
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact(str(MODEL_DIR / "best_model.pth"))
        
        print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train skin lesion classifier")
    parser.add_argument("--image_dir", type=str, default="data/images",
                        help="Directory containing images")
    parser.add_argument("--labels_csv", type=str, default="data/labels.csv",
                        help="Path to labels CSV")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    
    args = parser.parse_args()
    
    train(
        image_dir=args.image_dir,
        labels_csv=args.labels_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
