# Step 1: Base image
FROM python:3.11-slim

# Step 2: Environment setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Step 3: Set working directory
WORKDIR /app

# Step 4: Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Step 5: Copy application code
COPY . .

# Step 6: Expose Gradio port
EXPOSE 7860

# Step 7: Run FastAPI app (which also serves Gradio at /gradio)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
