# EfficientNet-B0 Image Classification API

A Docker-ready image classification API using EfficientNet-B0 trained on ImageNet-100 dataset with FastAPI backend and Gradio UI.

## Features

* EfficientNet-B0 model for efficient image classification
* **FastAPI REST API** for programmatic inference
* **Gradio Web UI** for interactive testing
* Docker support for easy deployment
* 100 ImageNet classes classification
* Modular architecture (preprocessing, model, API layers)
* Environment variable configuration

## Project Structure
```
├── app.py                          # FastAPI + Gradio application
├── model.py                        # Model definition and loading
├── preprocessing.py                # Image preprocessing pipeline
├── efficientnet_imagenet100.pth    # Pre-trained model weights
├── class_names.txt                 # Class labels
├── class_names_idx.txt             # Class index mapping
├── requirements.txt                # Dependencies
└── Dockerfile                      # Docker configuration
```

## Installation

### Local Setup
```bash
# Clone the repository
git clone https://github.com/bibekk477/Efficient_b0_fastapi.git
cd Efficient_b0_fastapi

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker Setup
```bash
# Build image
docker build -t efficientnet-api .

# Run container
docker run -p 8000:8000 -p 7860:7860 efficientnet-api
```

## Usage

### Access the Application

Once running, you can access:

* **Gradio UI**: http://localhost:7860/gradio (Interactive image upload)
* **FastAPI Docs**: http://localhost:8000/docs (API documentation)
* **Root**: http://localhost:8000/ (API info)

### 1. Gradio Web Interface

Simply navigate to `http://localhost:7860/gradio` and upload an image through the browser interface.

### 2. FastAPI Endpoint

**Endpoint:** `POST /predict`
