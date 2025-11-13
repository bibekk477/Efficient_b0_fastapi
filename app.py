from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
from model import load_model
from preprocessing import preprocess_image
import gradio as gr
import io
import requests

#  FastAPI app
app = FastAPI(title="EfficientNet Image Classifier")


#  Load class names
# Read the text file
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
with open("class_names_idx.txt", "r") as f:
    class_names_idx = [int(line.strip()) for line in f.readlines()]


NUM_CLASSES = len(class_names)

#  Load model
model = load_model("efficientnet_imagenet100.pth", NUM_CLASSES)
model.eval()


# Core predictions
def predict_core(tensor):
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return class_names[class_names_idx[predicted.item()]], confidence.item()


# -------------------- API endpoint --------------------
@app.post("/predict")
async def predict_api(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    tensor = preprocess_image(image)
    pred_class, confidence = predict_core(tensor)
    return {"predicted_class": pred_class, "confidence": confidence}


# -------------------- Gradio wrapper that calls /predict --------------------
def predict_via_api(image: Image.Image):
    """
    Gradio function: sends image via HTTP POST to FastAPI /predict
    """
    # Convert PIL image to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    files = {"file": buffer}

    try:
        response = requests.post("http://127.0.0.1:8000/predict", files=files)
        response.raise_for_status()
        result = response.json()
        return f"{result['predicted_class']} (confidence: {result['confidence']:.2f})"
    except Exception as e:
        return f"Error: {str(e)}"


# -------------------- Gradio Interface --------------------
gr_interface = gr.Interface(
    fn=predict_via_api,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="EfficientNet Image Classifier",
    description="Upload an image to get prediction via FastAPI /predict endpoint",
)

app = gr.mount_gradio_app(
    app, gr_interface, path="/gradio"
)  # runs “in-process” with FastAPI


# -------------------- Root route --------------------
@app.get("/")
def root():
    return {"message": "Go to /gradio to use the UI or /predict for API."}
