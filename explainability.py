"""GradCAM explainability for skin lesion classification."""

import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import load_model
from data_loader import get_inference_transform
from config import DEVICE, CLASS_NAMES, MODEL_DIR, IMAGE_SIZE


def get_gradcam_visualization(
    image_path: str,
    model_path: str = None,
    target_class: int = None,
    output_path: str = None
) -> np.ndarray:
    """
    Generate GradCAM visualization for a skin lesion image.
    
    Args:
        image_path: Path to input image
        model_path: Path to model checkpoint
        target_class: Target class for GradCAM (None = predicted class)
        output_path: Path to save visualization
        
    Returns:
        GradCAM visualization as numpy array
    """
    # Load model
    if model_path:
        model = load_model(model_path)
    else:
        checkpoint_path = MODEL_DIR / "best_model.pth"
        if checkpoint_path.exists():
            model = load_model(str(checkpoint_path))
        else:
            model = load_model()
    
    model.eval()
    
    # Load and preprocess image
    original_image = Image.open(image_path).convert("RGB")
    original_image = original_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    original_np = np.array(original_image) / 255.0
    
    transform = get_inference_transform()
    input_tensor = transform(original_image).unsqueeze(0).to(DEVICE)
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = probs.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()
    
    # Use predicted class if target not specified
    if target_class is None:
        target_class = pred_class
    
    # Setup GradCAM
    # Target the last convolutional layer in EfficientNet
    target_layers = [model.backbone.features[-1]]
    
    cam = GradCAM(model=model, target_layers=target_layers)
    
    # Generate CAM
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Overlay on original image
    visualization = show_cam_on_image(original_np, grayscale_cam, use_rgb=True)
    
    # Create figure with original, CAM, and overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(grayscale_cam, cmap="jet")
    axes[1].set_title("GradCAM Heatmap")
    axes[1].axis("off")
    
    axes[2].imshow(visualization)
    axes[2].set_title(f"Prediction: {CLASS_NAMES[pred_class]} ({confidence:.2%})")
    axes[2].axis("off")
    
    plt.suptitle("Skin Lesion Classification - Model Explainability", fontsize=14)
    plt.tight_layout()
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return visualization


def generate_uncertainty_map(
    image_path: str,
    model_path: str = None,
    n_samples: int = 20,
    output_path: str = None
) -> dict:
    """
    Generate uncertainty visualization using Monte Carlo Dropout.
    
    Args:
        image_path: Path to input image
        model_path: Path to model checkpoint
        n_samples: Number of MC samples
        output_path: Path to save visualization
        
    Returns:
        Dictionary with prediction and uncertainty info
    """
    # Load model
    if model_path:
        model = load_model(model_path)
    else:
        checkpoint_path = MODEL_DIR / "best_model.pth"
        if checkpoint_path.exists():
            model = load_model(str(checkpoint_path))
        else:
            model = load_model()
    
    # Load image
    original_image = Image.open(image_path).convert("RGB")
    original_image = original_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    transform = get_inference_transform()
    input_tensor = transform(original_image).unsqueeze(0).to(DEVICE)
    
    # Get prediction with uncertainty
    mean_probs, uncertainty = model.predict_with_uncertainty(input_tensor, n_samples)
    
    probs = mean_probs[0].cpu().numpy()
    uncertainty_val = uncertainty[0].cpu().item()
    pred_class = probs.argmax()
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image with prediction
    axes[0].imshow(original_image)
    axes[0].set_title(f"Prediction: {CLASS_NAMES[pred_class]}\nConfidence: {probs[pred_class]:.2%}")
    axes[0].axis("off")
    
    # Probability distribution with uncertainty
    colors = ["red" if i == pred_class else "steelblue" for i in range(len(CLASS_NAMES))]
    bars = axes[1].bar(CLASS_NAMES, probs, color=colors)
    axes[1].set_ylabel("Probability")
    axes[1].set_title(f"Class Probabilities\nUncertainty: {uncertainty_val:.3f}")
    axes[1].set_ylim(0, 1)
    
    # Add uncertainty indicator
    uncertainty_color = "green" if uncertainty_val < 0.3 else "orange" if uncertainty_val < 0.6 else "red"
    axes[1].axhline(y=uncertainty_val, color=uncertainty_color, linestyle="--", 
                    label=f"Uncertainty: {uncertainty_val:.3f}")
    axes[1].legend()
    
    plt.suptitle("Skin Lesion Classification - Uncertainty Estimation", fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved uncertainty visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        "prediction": CLASS_NAMES[pred_class],
        "confidence": float(probs[pred_class]),
        "uncertainty": float(uncertainty_val),
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate explainability visualizations")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="gradcam_output.png", 
                        help="Output path for visualization")
    parser.add_argument("--mode", type=str, choices=["gradcam", "uncertainty"], 
                        default="gradcam", help="Visualization mode")
    
    args = parser.parse_args()
    
    if args.mode == "gradcam":
        get_gradcam_visualization(
            image_path=args.image,
            model_path=args.model,
            output_path=args.output
        )
    else:
        result = generate_uncertainty_map(
            image_path=args.image,
            model_path=args.model,
            output_path=args.output
        )
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Uncertainty: {result['uncertainty']:.3f}")
