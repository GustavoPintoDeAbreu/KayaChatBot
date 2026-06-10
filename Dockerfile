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

# Install PyTorch with CUDA 12.4 wheels. Kept separate from requirements.txt
# because torch must come from the CUDA-specific index, not PyPI.
# torch 2.6.0 / cu124: matches the validated kaya_chatbot_env venv so the
# Gemma 4 inference behaviour in the image is identical to local.
RUN python -m pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install all remaining dependencies from requirements.txt — the single source
# of truth, pinned to the validated venv versions (transformers 5.5.0, trl
# 0.24.0, peft 0.19.0, unsloth 2026.4.5, xformers cu124).
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt

# PEFT 0.19.0 calls getattr(torch, "float8_e8m0fnu") which raises on torch 2.6
# (that dtype was added later). Mirror the venv patch: guard the lookup with
# hasattr so missing float8 dtypes are skipped. See CLAUDE.md.
RUN python - <<'PY'
import peft.tuners.tuners_utils as m
path = m.__file__
src = open(path).read()
needle = "    for name in UPCAST_DTYPES:\n        torch_dtype = getattr(torch, name)\n"
guard = "    for name in UPCAST_DTYPES:\n        if not hasattr(torch, name):\n            continue\n        torch_dtype = getattr(torch, name)\n"
if "if not hasattr(torch, name):" in src:
    print("PEFT float8 guard already present")
else:
    assert needle in src, "PEFT patch anchor not found — peft version changed?"
    open(path, "w").write(src.replace(needle, guard))
    print("Applied PEFT float8 guard patch")
PY

# flash-attention is intentionally NOT installed: it is absent from the validated
# venv and unsloth runs inference fine without it (falls back to SDPA / xformers).

# Set environment variables for GPU and caching
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/models/.cache \
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Copy project files
COPY src/ /app/src/
COPY tests/ /app/tests/
COPY config.yaml /app/config.yaml
COPY run_full_pipeline.py /app/
COPY configs/ /app/configs/

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/outputs /app/reports/benchmarks

# Health check: verify the Python environment is functional
HEALTHCHECK --interval=60s --timeout=30s --retries=3 \
    CMD python -c "import torch; import transformers; import chromadb; print('OK')" || exit 1

# Default command
CMD ["/bin/bash"]