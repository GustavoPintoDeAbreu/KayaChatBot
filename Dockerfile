FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Image version label
LABEL version="2.0.0" \
      description="KayaChatBot — RAG serving + data pipeline" \
      maintainer="GustavoPintoDeAbreu"

# Set working directory
WORKDIR /app

# Install system dependencies + Python 3.12 via the deadsnakes PPA so the image
# matches the local virtualenv (Python 3.12).
RUN apt-get update && apt-get install -y \
    software-properties-common \
    ca-certificates \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default `python` (run_full_pipeline.py and the compose
# commands invoke bare `python`) and install uv, which installs into that
# system interpreter — the image has no venv.
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

RUN uv pip install --python /usr/bin/python3.12 --no-cache \
    "setuptools==75.6.0" \
    "wheel==0.45.1" \
    "packaging==24.2" \
    "ninja==1.11.1.1"

# Install PyTorch with CUDA 12.4 wheels. Kept separate from requirements.txt
# because torch must come from the CUDA-specific index, not PyPI.
# torch 2.6.0 / cu124: matches the kaya_chatbot_env venv.
RUN uv pip install --python /usr/bin/python3.12 --no-cache \
    torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install all remaining dependencies from requirements.txt, the single source
# of truth for everything except torch.
COPY requirements.txt ./
RUN uv pip install --python /usr/bin/python3.12 --no-cache -r requirements.txt

# Set environment variables for GPU and caching
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models/.cache \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Copy project files
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY conftest.py /app/conftest.py
COPY config.yaml /app/config.yaml
COPY run_full_pipeline.py /app/

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/outputs /app/reports/benchmarks

# Health check: verify the Python environment is functional
HEALTHCHECK --interval=60s --timeout=30s --retries=3 \
    CMD python -c "import torch; import transformers; import chromadb; print('OK')" || exit 1

# Default command
CMD ["/bin/bash"]
