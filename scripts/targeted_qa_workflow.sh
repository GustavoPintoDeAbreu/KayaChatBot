#!/usr/bin/env bash
# Helper script: generate targeted synthetic QA, append to main synthetic dataset, merge, and show commands to retrain.
# Run inside the 'kaya_chatbot' virtualenv: source kaya_chatbot_env/bin/activate

set -euo pipefail

# 1) Generate 60 targeted examples using configured provider (xai suggested)
python src/data/generate_synthetic_data.py \
  --mode count --provider xai \
  --output data/synthetic_targeted_qa.jsonl --count 60

# 2) Append to main synthetic dataset
cat data/synthetic_targeted_qa.jsonl >> data/synthetic_kaya.jsonl

# 3) Re-merge datasets (produce train/val)
python src/data/merge_datasets.py

# 4) Retrain (example command; ensure CUDA config and env active)
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/finetuning/train.py --profile gemma4-e4b --output-dir ./models/kaya_gemma4_synth_v5

# 5) Run benchmark after training (example)
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/testing/benchmark.py --model-dir ./models/kaya_gemma4_synth_v5 --profile gemma4-e4b --judge-provider xai --scenarios 5

# Notes: generation and training commands may take significant time and need correct env and API keys.
