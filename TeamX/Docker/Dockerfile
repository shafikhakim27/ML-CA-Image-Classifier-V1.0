# Multi-stage Dockerfile for Fruit Classification Model
# Stage 1: Builder (download pre-trained weights)
FROM python:3.9-slim as builder

WORKDIR /app
RUN pip install --no-cache-dir tensorflow==2.13.0
RUN python -c "from tensorflow.keras.applications import MobileNetV2; MobileNetV2(weights='imagenet')"

# Stage 2: Runtime (smaller final image)
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-downloaded weights from builder
COPY --from=builder /root/.keras /root/.keras

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY experiments/ ./experiments/
COPY standalone.py ./

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Default to prediction mode with standalone script
ENTRYPOINT ["python", "standalone.py"]
CMD ["--help"]
