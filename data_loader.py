"""Data loading utilities for skin lesion classification."""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import pandas as pd
from typing import Tuple, Optional

from config import IMAGE_SIZE, BATCH_SIZE, TRAIN_SPLIT, CLASS_NAMES


class SkinLesionDataset(Dataset):
    """Dataset for skin lesion images."""
    
    def __init__(
        self, 
        image_dir: str,
        labels_csv: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform or self.get_default_transform(is_train)
        
        # Load labels if provided
        if labels_csv:
            self.labels_df = pd.read_csv(labels_csv)
            self.image_ids = self.labels_df['image'].tolist()
        else:
            # Find all images in directory
            self.image_ids = [f.stem for f in self.image_dir.glob("*.jpg")]
            self.labels_df = None
    
    def __len__(self) -> int:
        return len(self.image_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_id = self.image_ids[idx]
        image_path = self.image_dir / f"{image_id}.jpg"
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # Get label
        if self.labels_df is not None:
            row = self.labels_df[self.labels_df['image'] == image_id].iloc[0]
            label = row[CLASS_NAMES].values.argmax()
        else:
            label = 0  # Dummy label for inference
        
        return image, label
    
    @staticmethod
    def get_default_transform(is_train: bool = True) -> transforms.Compose:
        """Get default image transforms."""
        if is_train:
            return transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])


def get_data_loaders(
    image_dir: str,
    labels_csv: str,
    batch_size: int = BATCH_SIZE,
    train_split: float = TRAIN_SPLIT
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders."""
    
    # Load full dataset
    full_dataset = SkinLesionDataset(
        image_dir=image_dir,
        labels_csv=labels_csv,
        is_train=True
    )
    
    # Split into train/val
    train_size = int(len(full_dataset) * train_split)
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Update transforms for validation
    val_dataset.dataset.transform = SkinLesionDataset.get_default_transform(is_train=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # MPS compatibility
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader


def get_inference_transform() -> transforms.Compose:
    """Get transform for inference."""
    return SkinLesionDataset.get_default_transform(is_train=False)
