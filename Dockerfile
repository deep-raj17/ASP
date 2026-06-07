# Production Dockerfile for MIMII Acoustic Anomaly Detection
# ─────────────────────────────────────────────────────────

FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS base

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# ─────────────────────────────────────────────────────────
# Build stage for dependencies
# ─────────────────────────────────────────────────────────
FROM base AS builder

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.1
RUN pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install -r requirements.txt

# Install production extras
RUN pip install \
    tensorrt \
    pycuda \
    onnx \
    onnxruntime-gpu \
    accelerate \
    triton

# ─────────────────────────────────────────────────────────
# Final production stage
# ─────────────────────────────────────────────────────────
FROM base AS production

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# PyTorch optimizations
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV TORCH_CUDNN_V8_API_ENABLED=1

# Copy application code
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/checkpoints /app/logs /app/exports

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('GPU:', torch.cuda.is_available())" || exit 1

# Expose port for Gradio
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
