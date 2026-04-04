"""
Fine-Tuning Script
Trains Qwen3-14B on WhatsApp chat data using LoRA and 4-bit quantization.
"""
import os
import sys
from pathlib import Path
import yaml
import torch
import builtins
import psutil

# Suppress HuggingFace cache deprecation warning
os.environ['HF_HOME'] = os.environ.get('HF_HOME', '/tmp/huggingface')

# Add project root to Python path so 'src' package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

builtins.psutil = psutil
from datasets import load_dataset

from src.finetuning.trainer import KayaTrainer


def main():
    print("=" * 60)
    print("Fine-Tuning Pipeline")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"\n\U0001f50d GPU Check:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        print(f"   GPU name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("   \u26a0\ufe0f  WARNING: No CUDA GPU detected!")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("   Exiting...")
            return
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
    print(f"\n1. Loading configuration from {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Check test mode
    test_mode = config['test_mode']['enabled']
    if test_mode:
        print("\n\u26a0\ufe0f  TEST MODE ENABLED - Using reduced parameters for quick validation")
        print("   Set test_mode.enabled: false in config.yaml for full training\n")
    
    # Extract settings
    model_id = config['model']['model_id']
    max_seq_length = config['model']['max_seq_length']
    
    # Use test or production output directory
    if test_mode:
        output_dir = config['training']['output_dir'] + "_test"
    else:
        output_dir = config['training']['output_dir']
    
    # Dataset paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    train_file = os.path.join(data_dir, "train_synthetic.jsonl")
    val_file = os.path.join(data_dir, "val_synthetic.jsonl")
    
    print(f"   \u2713 Model: {model_id}")
    print(f"   \u2713 Max sequence length: {max_seq_length}")
    print(f"   \u2713 Output directory: {output_dir}")
    
    # Check if datasets exist
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print(f"\n\u274c Error: Dataset files not found!")
        print(f"   Expected: {train_file}")
        print(f"   Expected: {val_file}")
        
        # Check what files exist
        existing_files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]
        if existing_files:
            print(f"\n   Found in data/: {', '.join(existing_files)}")
        
        print(f"\n   Please run these steps first:")
        print(f"   1. python src/data/extract_all_messages.py")
        print(f"   2. python src/data/generate_synthetic_data.py")
        print(f"   3. python src/data/prepare_portuguese_data.py")
        print(f"   4. python src/data/merge_datasets.py  \u2190 Creates train_synthetic.jsonl and val_synthetic.jsonl")
        print(f"\n   Or run: python validate_pipeline.py to check which steps are missing")
        return
    
    # Load datasets
    print(f"\n2. Loading datasets...")
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    val_dataset = load_dataset("json", data_files=val_file, split="train")
    print(f"   \u2713 Train samples: {len(train_dataset)}")
    print(f"   \u2713 Validation samples: {len(val_dataset)}")
    
    # Initialize trainer with LoRA config
    print(f"\n3. Initializing trainer with LoRA configuration...")
    lora_config = {
        "r": config['model']['lora_r'],
        "alpha": config['model']['lora_alpha'],
        "dropout": config['model']['lora_dropout'],
        "target_modules": config['model']['target_modules']
    }

    # Read incremental retraining config
    resume_from_checkpoint = config['training'].get('resume_from_checkpoint')

    trainer_wrapper = KayaTrainer(
        model_id=model_id,
        max_seq_length=max_seq_length,
        lora_config=lora_config,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    
    print(f"\n4. Loading model (this may take several minutes)...")
    print(f"   Downloading/loading {model_id}...")
    trainer_wrapper.load_model()
    
    # Prepare training config
    training_config = {
        "max_steps": config['training']['max_steps'],
        "per_device_train_batch_size": config['training']['per_device_train_batch_size'],
        "gradient_accumulation_steps": config['training']['gradient_accumulation_steps'],
        "learning_rate": config['training']['learning_rate'],
        "warmup_steps": config['training']['warmup_steps'],
        "weight_decay": config['training']['weight_decay'],
        "optim": config['training']['optim'],
        "lr_scheduler_type": config['training']['lr_scheduler_type'],
        "logging_steps": config['training']['logging_steps'],
        "save_steps": config['training']['save_steps'],
        "eval_steps": config['training']['eval_steps'],
        "seed": config['training']['seed'],
    }

    # Pass incremental params so the trainer can apply them when a checkpoint is loaded
    if resume_from_checkpoint:
        training_config["incremental_steps"] = config['training']['incremental_steps']
        training_config["incremental_learning_rate"] = config['training']['incremental_learning_rate']
        print(f"\n   \U0001f504 Incremental training enabled:")
        print(f"   \u2713 Checkpoint: {resume_from_checkpoint}")
        print(f"   \u2713 Steps: {config['training']['incremental_steps']} (vs full: {config['training']['max_steps']})")
        print(f"   \u2713 Learning rate: {config['training']['incremental_learning_rate']} (vs full: {config['training']['learning_rate']})")
    
    # Auto-detect latest checkpoint to resume from, but only when
    # resume_from_checkpoint is explicitly configured (non-null).
    # When it is null the intent is "start fresh" -- we must NOT
    # resume even if checkpoint-* directories happen to exist in the
    # output dir (e.g. because the models/ volume is mounted from a
    # previous local training run in the Docker pipeline).
    import glob as _glob
    resume_checkpoint = None
    if resume_from_checkpoint:
        checkpoints = sorted(_glob.glob(os.path.join(output_dir, "checkpoint-*")), key=lambda p: int(p.split("-")[-1]))
        resume_checkpoint = checkpoints[-1] if checkpoints else None
        if resume_checkpoint:
            print(f"\n   \u21a9\ufe0f  Resuming from checkpoint: {resume_checkpoint}")
    else:
        print(f"\n   \U0001f195 Starting fresh -- checkpoint auto-detection skipped (resume_from_checkpoint is null)")

    print(f"\n5. Starting training...")
    trainer, stats = trainer_wrapper.train(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        output_dir=output_dir,
        training_config=training_config,
        resume_from_checkpoint=resume_checkpoint,
    )
    
    # Save model
    print(f"\n6. Saving fine-tuned model...")
    trainer_wrapper.save_model(output_dir)
    
    print("\n" + "=" * 60)
    print("\u2705 Fine-tuning complete!")
    print(f"   Model saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
