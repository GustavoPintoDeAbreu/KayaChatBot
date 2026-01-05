FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

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

# Upgrade pip, setuptools, wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel packaging ninja

# Install PyTorch with CUDA 12.4 support
RUN pip install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Copy requirements files
COPY requirements.txt ./

# Install core ML dependencies (flexible versions to avoid conflicts)
RUN pip install --no-cache-dir \
    transformers \
    datasets \
    peft \
    trl \
    accelerate \
    bitsandbytes \
    scikit-learn \
    PyYAML \
    numpy \
    pandas \
    scipy \
    sentencepiece \
    protobuf \
    tiktoken \
    python-dotenv \
    openai \
    xai-sdk

# Install unsloth and unsloth-zoo
RUN pip install --no-cache-dir \
    unsloth==2025.12.7 \
    unsloth-zoo==2025.12.7 \
    xformers==0.0.27.post2

# FIX: Uninstall torchao to avoid torch.int1 error with torch 2.4.0
RUN pip uninstall -y torchao

# FIX: Patch unsloth_zoo for torch._inductor.config error
RUN sed -i "s/inductor_config_source = inspect.getsource(torch._inductor.config)/import torch._inductor.config; inductor_config_source = inspect.getsource(torch._inductor.config)/" /usr/local/lib/python3.10/dist-packages/unsloth_zoo/temporary_patches/common.py

# Build and install flash-attention from source (takes ~5-10 minutes)
# This ensures compatibility with the specific CUDA/PyTorch setup
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Install additional utilities
RUN pip install --no-cache-dir \
    ipython \
    tqdm \
    colorama

# Set environment variables for GPU and caching
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models/.cache \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Copy project files
COPY src/ /app/src/
COPY config.docker.yaml /app/config.yaml
COPY run_full_pipeline.py /app/
COPY test_pipeline.py /app/
COPY validate_pipeline.py /app/
COPY test_llm_cleaning.py /app/

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/outputs

# Default command
CMD ["/bin/bash"]