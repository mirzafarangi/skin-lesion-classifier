"""Model architecture for skin lesion classification."""

import torch
import torch.nn as nn
from torchvision import models
from config import NUM_CLASSES, DEVICE


class SkinLesionClassifier(nn.Module):
    """EfficientNet-B0 based classifier for skin lesions with uncertainty estimation."""
    
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        
        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 10) -> tuple:
        """
        Monte Carlo Dropout for uncertainty estimation.
        
        Args:
            x: Input tensor
            n_samples: Number of forward passes for uncertainty estimation
            
        Returns:
            mean_probs: Mean predicted probabilities
            uncertainty: Predictive uncertainty (entropy)
        """
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs)
        
        # Stack predictions: (n_samples, batch, classes)
        predictions = torch.stack(predictions)
        
        # Mean prediction
        mean_probs = predictions.mean(dim=0)
        
        # Uncertainty as predictive entropy
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
        
        # Normalize entropy to [0, 1]
        max_entropy = torch.log(torch.tensor(NUM_CLASSES, dtype=torch.float32))
        uncertainty = entropy / max_entropy
        
        self.eval()
        return mean_probs, uncertainty


def load_model(model_path: str = None) -> SkinLesionClassifier:
    """Load model from checkpoint or create new."""
    model = SkinLesionClassifier()
    
    if model_path:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return model.to(DEVICE)
