from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from model import load_model
from preprocessing import preprocess_image
import gradio as gr
import io
import requests
import os

# Configuration
MODEL_PATH = os.environ.get("MODEL_PATH", "efficientnet_imagenet100.pth")
CLASS_NAMES_PATH = os.environ.get("CLASS_NAMES_PATH", "class_names.txt")

#  FastAPI app
app = FastAPI(title="EfficientNet Image Classifier")


#  Load class names from the  text file
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]


with open("class_names_idx.txt", "r") as f:
    class_names_idx = [int(line.strip()) for line in f.readlines()]


NUM_CLASSES = len(class_names)

# Load model (detect device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(MODEL_PATH, NUM_CLASSES, device=device)
model.eval()


# Core predictions
def predict_core(tensor: torch.Tensor):
    """Run model on a preprocessed tensor and return (class_name, confidence).


    tensor: shape (1, C, H, W)
    """
    tensor = tensor.to(next(model.parameters()).device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return class_names[class_names_idx[predicted.item()]], confidence.item()


#  API endpoint
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    tensor = preprocess_image(image)
    pred_class, confidence = predict_core(tensor)
    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 4),
    }


#  Gradio UI


def predict_via_python(image: Image.Image):
    try:
        tensor = preprocess_image(image)
        pred_class, confidence = predict_core(tensor)
        return f"{pred_class} (confidence: {confidence:.2f})"
    except Exception as e:
        return f"Error: {str(e)}"


gr_interface = gr.Interface(
    fn=predict_via_python,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="EfficientNet Image Classifier",
    description="Upload an image to get prediction via FastAPI /predict endpoint (or use the Gradio UI).",
)


# mount gradio app inside FastAPI
app = gr.mount_gradio_app(app, gr_interface, path="/gradio")


# Root route
@app.get("/")
def root():
    return {"message": "Go to /gradio to use the UI or POST /predict for API."}
