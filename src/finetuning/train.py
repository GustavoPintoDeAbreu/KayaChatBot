"""
Fine-Tuning Script
Trains the active model profile on conversation data using LoRA and 4-bit quantization.

Uses a flat code path (direct SFTTrainer usage) to avoid OOM issues that occurred
with the KayaTrainer class wrapper during SFTTrainer initialization.
"""
import argparse
import gc
import glob
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import torch

# Use persistent HF cache
os.environ['HF_HOME'] = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Reset OOM score (VS Code systemd cgroup sets oom_score_adj=100)
try:
    with open('/proc/self/oom_score_adj', 'w') as _f:
        _f.write('0')
except OSError:
    pass

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def rss_gb():
    return psutil.Process().memory_info().rss / 1e9


def sys_ram():
    m = psutil.virtual_memory()
    return f"used={m.used/1e9:.1f}GB free={m.available/1e9:.1f}GB"


def main():
    print("=" * 60)
    print("Fine-Tuning Pipeline")
    print("=" * 60)
    print(f"[BASELINE] RSS: {rss_gb():.2f} GB | System: {sys_ram()}", flush=True)

    parser = argparse.ArgumentParser(description="KayaChatBot fine-tuning script.")
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Model profile name (overrides active_model_profile in config.yaml).",
    )
    args = parser.parse_args()

    # GPU check
    print(f"\n🔍 GPU Check:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("   ⚠️  No CUDA GPU detected!")
        if input("   Continue anyway? (y/n): ").lower() != 'y':
            return

    # Load configuration
    from src.config_loader import load_config

    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
    print(f"\n1. Loading configuration from {config_path}")
    config = load_config(config_path, profile_override=args.profile)

    model_id = config['model']['model_id']
    max_seq_length = config['model']['max_seq_length']
    lora_r = config['model']['lora_r']
    lora_alpha = config['model']['lora_alpha']
    lora_dropout = config['model']['lora_dropout']
    target_modules = config['model']['target_modules']
    tc = config['training']

    test_mode = config['test_mode']['enabled']
    if test_mode:
        output_dir = tc['output_dir'] + "_test"
        print("\n⚠️  TEST MODE ENABLED - Using reduced parameters for quick validation")
    else:
        output_dir = tc['output_dir']

    print(f"   ✓ Model: {model_id}")
    print(f"   ✓ Max sequence length: {max_seq_length}")
    print(f"   ✓ Output directory: {output_dir}")
    print(f"[AFTER CONFIG] RSS: {rss_gb():.2f} GB", flush=True)

    # Check datasets
    data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
    train_file = os.path.join(data_dir, "train_synthetic.jsonl")
    val_file = os.path.join(data_dir, "val_synthetic.jsonl")

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print(f"\n❌ Error: Dataset files not found!")
        print(f"   Expected: {train_file}")
        print(f"   Expected: {val_file}")
        print(f"\n   Run the data pipeline first:")
        print(f"   1. python src/data/extract_all_messages.py")
        print(f"   2. python src/data/generate_synthetic_data.py")
        print(f"   3. python src/data/prepare_portuguese_data.py")
        print(f"   4. python src/data/merge_datasets.py")
        return

    # Load model
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    print(f"\n2. Loading model: {model_id}...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        low_cpu_mem_usage=True,
    )
    gc.collect()
    print(f"   [MEM post-model] RSS={rss_gb():.2f}GB | VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    # Add LoRA adapters
    print(f"\n3. Adding LoRA adapters (r={lora_r}, alpha={lora_alpha})...")
    model = FastModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
        autocast_adapter_dtype=False,
    )
    tokenizer = get_chat_template(tokenizer, "gemma-4")
    gc.collect()
    print(f"   [MEM post-LoRA] RSS={rss_gb():.2f}GB | VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)
    model.print_trainable_parameters()

    # Load datasets
    from datasets import load_dataset

    print(f"\n4. Loading datasets...")
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    val_dataset = load_dataset("json", data_files=val_file, split="train")
    print(f"   ✓ Train samples: {len(train_dataset)}")
    print(f"   ✓ Validation samples: {len(val_dataset)}")
    print(f"[AFTER DATASETS] RSS: {rss_gb():.2f} GB", flush=True)

    # Create SFTTrainer
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback

    print(f"\n5. Creating SFTTrainer...")

    bos = tokenizer.bos_token or ""

    def formatting_func(examples):
        if isinstance(examples["formatted_text"], list):
            return [t.removeprefix(bos).strip() for t in examples["formatted_text"]]
        else:
            return [examples["formatted_text"].removeprefix(bos).strip()]

    class MemoryMonitorCallback(TrainerCallback):
        def __init__(self, log_every=10, rss_limit_gb=25.0):
            self.log_every = log_every
            self.rss_limit_gb = rss_limit_gb
            self._proc = psutil.Process(os.getpid())

        def _log(self, step):
            rss = self._proc.memory_info().rss / 1e9
            vram_a = torch.cuda.memory_allocated() / 1e9
            vram_r = torch.cuda.memory_reserved() / 1e9
            sf = psutil.virtual_memory().available / 1e9
            print(
                f"[MEM step={step}] RSS={rss:.2f}GB | VRAM alloc={vram_a:.2f}GB res={vram_r:.2f}GB | SysFree={sf:.1f}GB",
                flush=True,
            )
            if rss > self.rss_limit_gb:
                print(f"⚠️  RSS {rss:.1f}GB > limit — gc.collect()", flush=True)
                gc.collect()
                torch.cuda.empty_cache()

        def on_train_begin(self, args, state, control, **kwargs):
            self._log(0)

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step % self.log_every == 0:
                self._log(state.global_step)

    class ProgressCallback(TrainerCallback):
        def __init__(self):
            self.start_time = None

        def on_train_begin(self, args, state, control, **kwargs):
            self.start_time = time.time()
            print(
                f"\n{'='*60}\n🚀 Training Started - {datetime.now():%Y-%m-%d %H:%M:%S}\n{'='*60}",
                flush=True,
            )

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                step, total = state.global_step, state.max_steps
                elapsed = time.time() - self.start_time
                sps = step / elapsed if elapsed > 0 else 0
                eta = (total - step) / sps / 60 if sps > 0 else 0
                msg = f"\n[{datetime.now():%H:%M:%S}] 📊 Step {step}/{total} ({step/total*100:.1f}%)\n"
                msg += f"   ⚡ {sps:.2f} steps/sec | ETA: {eta:.1f} min\n"
                if "loss" in logs:
                    msg += f"   📉 Loss: {logs['loss']:.4f}\n"
                if "learning_rate" in logs:
                    msg += f"   📚 LR: {logs['learning_rate']:.2e}\n"
                if "eval_loss" in logs:
                    msg += f"   ✅ Val Loss: {logs['eval_loss']:.4f}\n"
                print(msg, end="", flush=True)

        def on_train_end(self, args, state, control, **kwargs):
            elapsed = time.time() - self.start_time
            print(
                f"\n{'='*60}\n✨ Training Completed - {datetime.now():%Y-%m-%d %H:%M:%S}"
                f"\n   Duration: {elapsed/60:.2f} min\n{'='*60}",
                flush=True,
            )

    # Incremental training: auto-detect checkpoint
    resume_from_checkpoint = tc.get('resume_from_checkpoint')
    resume_checkpoint = None
    max_steps = tc['max_steps']
    learning_rate = tc['learning_rate']

    if resume_from_checkpoint:
        checkpoints = sorted(
            glob.glob(os.path.join(output_dir, "checkpoint-*")),
            key=lambda p: int(p.split("-")[-1]),
        )
        resume_checkpoint = checkpoints[-1] if checkpoints else None
        max_steps = tc.get('incremental_steps', max_steps)
        learning_rate = tc.get('incremental_learning_rate', learning_rate)
        if resume_checkpoint:
            print(f"   ↩️  Resuming from checkpoint: {resume_checkpoint}")
        print(f"   🔄 Incremental: {max_steps} steps, LR={learning_rate}")
    else:
        print(f"   🆕 Starting fresh training")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        formatting_func=formatting_func,
        callbacks=[ProgressCallback(), MemoryMonitorCallback(log_every=10)],
        args=SFTConfig(
            dataset_text_field="formatted_text",
            max_length=max_seq_length,
            dataset_num_proc=tc.get("dataset_num_proc", 1),
            packing=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            per_device_train_batch_size=tc['per_device_train_batch_size'],
            per_device_eval_batch_size=tc.get('per_device_eval_batch_size', 1),
            gradient_accumulation_steps=tc['gradient_accumulation_steps'],
            warmup_steps=tc['warmup_steps'],
            max_steps=max_steps,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=tc['logging_steps'],
            optim=tc['optim'],
            weight_decay=tc['weight_decay'],
            lr_scheduler_type=tc['lr_scheduler_type'],
            seed=tc['seed'],
            output_dir=output_dir,
            save_steps=tc.get('save_steps', 100),
            eval_strategy="steps" if val_dataset else "no",
            eval_steps=tc.get('eval_steps', 50) if val_dataset else None,
            load_best_model_at_end=True if val_dataset else False,
            metric_for_best_model="eval_loss" if val_dataset else None,
            report_to="none",
            skip_memory_metrics=True,
        ),
    )
    gc.collect()
    print(f"   [MEM post-SFTTrainer] RSS={rss_gb():.2f}GB | VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB", flush=True)

    # Mask everything but assistant responses
    from unsloth.chat_templates import train_on_responses_only

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
        num_proc=1,
    )
    print(f"   [MEM post-response_masking] RSS={rss_gb():.2f}GB", flush=True)

    # Train
    print(f"\n6. Starting training ({max_steps} steps)...")
    trainer_stats = trainer.train(resume_from_checkpoint=resume_checkpoint)

    # Save
    print(f"\n7. Saving fine-tuned model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 60)
    print(f"✅ Fine-tuning complete!")
    print(f"   Model saved to: {output_dir}")
    print(f"   Final loss: {trainer_stats.training_loss:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
