"""FastAPI inference endpoint for skin lesion classification."""

import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from typing import Dict, Any

from model import load_model
from data_loader import get_inference_transform
from config import DEVICE, CLASS_NAMES, MODEL_DIR


app = FastAPI(
    title="Skin Lesion Classifier API",
    description="Medical image classification for dermatological conditions",
    version="1.0.0"
)

# Load model at startup
model = None
transform = get_inference_transform()


@app.on_event("startup")
async def load_model_on_startup():
    """Load model when server starts."""
    global model
    model_path = MODEL_DIR / "best_model.pth"
    
    if model_path.exists():
        model = load_model(str(model_path))
        print(f"Model loaded from {model_path}")
    else:
        model = load_model()
        print("Using pretrained model (no fine-tuned weights found)")
    
    model.eval()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "model": "skin-lesion-classifier"}


@app.get("/classes")
async def get_classes():
    """Get available classification classes."""
    return {
        "classes": CLASS_NAMES,
        "descriptions": {
            "MEL": "Melanoma",
            "NV": "Melanocytic nevus",
            "BCC": "Basal cell carcinoma",
            "AKIEC": "Actinic keratosis",
            "BKL": "Benign keratosis",
            "DF": "Dermatofibroma",
            "VASC": "Vascular lesion"
        }
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Classify a skin lesion image.
    
    Args:
        file: Image file (JPEG, PNG)
        
    Returns:
        Prediction with class probabilities and uncertainty
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        # Get prediction with uncertainty
        mean_probs, uncertainty = model.predict_with_uncertainty(image_tensor)
        
        # Get top prediction
        probs = mean_probs[0].cpu().numpy()
        pred_idx = probs.argmax()
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        
        # Build response
        response = {
            "prediction": {
                "class": pred_class,
                "class_name": get_class_description(pred_class),
                "confidence": round(confidence, 4),
                "uncertainty": round(float(uncertainty[0].cpu()), 4)
            },
            "probabilities": {
                CLASS_NAMES[i]: round(float(probs[i]), 4)
                for i in range(len(CLASS_NAMES))
            },
            "model_info": {
                "model": "efficientnet_b0",
                "uncertainty_method": "monte_carlo_dropout"
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: list[UploadFile] = File(...)) -> Dict[str, Any]:
    """Classify multiple skin lesion images."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            image_tensor = transform(image).unsqueeze(0).to(DEVICE)
            
            mean_probs, uncertainty = model.predict_with_uncertainty(image_tensor)
            probs = mean_probs[0].cpu().numpy()
            pred_idx = probs.argmax()
            
            results.append({
                "filename": file.filename,
                "prediction": CLASS_NAMES[pred_idx],
                "confidence": round(float(probs[pred_idx]), 4),
                "uncertainty": round(float(uncertainty[0].cpu()), 4)
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results}


def get_class_description(class_code: str) -> str:
    """Get human-readable class description."""
    descriptions = {
        "MEL": "Melanoma",
        "NV": "Melanocytic nevus",
        "BCC": "Basal cell carcinoma",
        "AKIEC": "Actinic keratosis",
        "BKL": "Benign keratosis",
        "DF": "Dermatofibroma",
        "VASC": "Vascular lesion"
    }
    return descriptions.get(class_code, class_code)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
