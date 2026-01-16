"""
Export fine-tuned Kaya model to GGUF format for Ollama serving.

This script:
1. Loads the base model + LoRA adapters
2. Merges the LoRA weights into the base model
3. Exports the merged model to GGUF format (Q5_K_M quantization)
"""

import os
import yaml
from pathlib import Path
from unsloth import FastLanguageModel

def main():
    print("=" * 60)
    print("Kaya Model GGUF Export")
    print("=" * 60)

    # Load configuration
    config_path = Path(__file__).parent.parent.parent / "config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    model_dir = config['training']['output_dir']
    max_seq_length = config['model']['max_seq_length']

    # Check if model exists
    if not os.path.exists(model_dir):
        print(f"\n❌ Error: Model not found at {model_dir}")
        return

    print(f"\nLoading model from {model_dir}...")

    # Load model with LoRA adapters
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    print("✓ Model loaded with LoRA adapters")

    # Create export directory
    export_dir = Path(__file__).parent.parent.parent / "models"
    export_dir.mkdir(exist_ok=True)

    # Export to GGUF (Q5_K_M for good quality with reasonable size)
    gguf_path = export_dir / "kaya_v2_synthetic.gguf"

    print(f"\nExporting to GGUF format: {gguf_path}")
    print("This may take several minutes...")

    # Use Unsloth's GGUF export (includes LoRA merging automatically)
    model.save_pretrained_gguf(
        str(export_dir),
        tokenizer,
        quantization_method="q5_k_m",  # Good balance of quality/size for 24GB VRAM
    )

    print("✓ GGUF export completed!")
    print(f"📁 Model saved to: {gguf_path}")

    # Print file size
    if gguf_path.exists():
        size_gb = gguf_path.stat().st_size / (1024**3)
        print(f"📊 File size: {size_gb:.2f} GB")

if __name__ == "__main__":
    main()