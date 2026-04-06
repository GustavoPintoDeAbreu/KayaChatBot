FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Image version label
LABEL version="2.0.0" \
      description="KayaChatBot — fine-tuning + RAG pipeline" \
      maintainer="GustavoPintoDeAbreu"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip, setuptools, wheel (pinned for reproducibility)
RUN pip install --no-cache-dir \
    "pip==24.3.1" \
    "setuptools==75.6.0" \
    "wheel==0.45.1" \
    "packaging==24.2" \
    "ninja==1.11.1.1"

# Install PyTorch with CUDA 12.8 support (pinned for reproducibility)
# torch 2.9.1: required by xformers 0.0.33.post2; torchao 0.17.0 now works (register_constant added in torch 2.7+)
# unsloth-zoo 2026.4.2 requires torch>=2.4.0,<2.11.0 — torch 2.9.1 is within range
RUN pip install --no-cache-dir \
    torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Copy requirements file with pinned versions
COPY requirements.txt ./

# Install core ML dependencies (all versions pinned — see requirements.txt)
# transformers 4.57.6: latest stable 4.x; required >= 4.56.2 by trl 0.29.1
RUN pip install --no-cache-dir \
    transformers==4.57.6 \
    trl==0.29.1 \
    peft==0.15.2 \
    accelerate==1.5.2 \
    bitsandbytes==0.45.5 \
    datasets==3.6.0 \
    scikit-learn==1.6.1 \
    "PyYAML==6.0.2" \
    "numpy==1.26.4" \
    "pandas==2.2.3" \
    "scipy==1.14.1" \
    "sentencepiece==0.2.1" \
    "protobuf==5.29.6" \
    "tiktoken==0.9.0" \
    "python-dotenv==1.0.1" \
    "openai==1.109.1" \
    "xai-sdk==1.1.0" \
    "chromadb==1.0.0" \
    "sentence-transformers==3.4.1" \
    "tqdm==4.67.1" \
    "colorama==0.4.6" \
    "ipython==8.30.0"

# Install unsloth, unsloth-zoo and xformers
# unsloth 2026.4.2 adds Gemma 4 support
# xformers 0.0.33.post2 requires exactly torch==2.9.1
# torchao 0.17.0: requires torch 2.7+ (register_constant); works with torch 2.9.1
RUN pip install --no-cache-dir \
    unsloth==2026.4.2 \
    unsloth-zoo==2026.4.2 \
    xformers==0.0.33.post2 \
    torchao==0.17.0

# Build and install flash-attention from source (takes ~5-10 minutes)
# This ensures compatibility with the specific CUDA/PyTorch setup
RUN pip install --no-cache-dir flash-attn --no-build-isolation

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