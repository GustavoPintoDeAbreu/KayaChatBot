FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Image version label
LABEL version="2.0.0" \
      description="KayaChatBot — fine-tuning + RAG pipeline" \
      maintainer="GustavoPintoDeAbreu"

# Set working directory
WORKDIR /app

# Install system dependencies + Python 3.12 via the deadsnakes PPA so the image
# matches the local virtualenv (Python 3.12) and the documented PEFT patch paths
# (kaya_chatbot_env/lib/python3.12/...).
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
# commands invoke bare `python`) and bootstrap pip for it.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && python -m pip install --no-cache-dir \
    "pip==24.3.1" \
    "setuptools==75.6.0" \
    "wheel==0.45.1" \
    "packaging==24.2" \
    "ninja==1.11.1.1"

# Install PyTorch with CUDA 12.8 wheels. Kept separate from requirements.txt
# because torch must come from the CUDA-specific index, not PyPI.
# torch 2.9.1: required exactly by xformers 0.0.33.post2; within unsloth-zoo's
# supported range (torch>=2.4.0,<2.11.0).
RUN python -m pip install --no-cache-dir \
    torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Install all remaining dependencies from requirements.txt — the single source
# of truth. This keeps the image in lockstep with the local environment and the
# version constraints in CLAUDE.md (transformers>=5.5.0, trl<=0.24.0,
# peft==0.19.0, unsloth>=2026.4.5). Previously these were a divergent inline
# list that violated those constraints.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# Build flash-attention from source (takes ~5-10 minutes). Requires torch to be
# already installed (--no-build-isolation); not pinned in requirements.txt
# because there is no prebuilt wheel for this CUDA/torch combination.
RUN python -m pip install --no-cache-dir flash-attn --no-build-isolation

# Set environment variables for GPU and caching
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models/.cache \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Copy project files
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY config.docker.yaml /app/config.yaml
COPY run_full_pipeline.py /app/
COPY configs/ /app/configs/

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/outputs /app/reports/benchmarks

# Health check: verify the Python environment is functional
HEALTHCHECK --interval=60s --timeout=30s --retries=3 \
    CMD python -c "import torch; import transformers; import chromadb; print('OK')" || exit 1

# Default command
CMD ["/bin/bash"]