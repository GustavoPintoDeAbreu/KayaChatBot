import gc
import os
import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
from datasets import Dataset
from typing import Optional
import time
from datetime import datetime
import psutil


class MemoryMonitorCallback(TrainerCallback):
    """Monitors CPU RSS and GPU VRAM every N steps. Aborts if RSS exceeds limit."""

    def __init__(self, log_every: int = 10, rss_limit_gb: float = 25.0):
        self.log_every = log_every
        self.rss_limit_gb = rss_limit_gb
        self._proc = psutil.Process(os.getpid())

    def _log_memory(self, step: int):
        rss_gb = self._proc.memory_info().rss / 1e9
        vram_alloc = torch.cuda.memory_allocated() / 1e9
        vram_reserved = torch.cuda.memory_reserved() / 1e9
        sys_free = psutil.virtual_memory().available / 1e9
        print(
            f"[MEM step={step}] RSS={rss_gb:.2f}GB | VRAM alloc={vram_alloc:.2f}GB res={vram_reserved:.2f}GB | SysFree={sys_free:.1f}GB",
            flush=True,
        )
        if rss_gb > self.rss_limit_gb:
            print(
                f"⚠️  RSS {rss_gb:.1f}GB exceeds limit {self.rss_limit_gb}GB — running gc.collect()",
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
            rss_gb = self._proc.memory_info().rss / 1e9
            print(f"   After GC: RSS={rss_gb:.2f}GB", flush=True)

    def on_train_begin(self, args, state, control, **kwargs):
        self._log_memory(0)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_every == 0:
            self._log_memory(state.global_step)


class ProgressCallback(TrainerCallback):
    """Custom callback for real-time training progress logging."""

    def __init__(self):
        self.start_time = None
        self.step_times = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = "\n" + "=" * 60 + f"\n🚀 Training Started - {timestamp}\n" + "=" * 60
        print(message, flush=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            total_steps = state.max_steps
            progress = (step / total_steps) * 100

            # Calculate speed
            elapsed = time.time() - self.start_time
            steps_per_sec = step / elapsed if elapsed > 0 else 0
            eta_seconds = (
                (total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
            )
            eta_mins = eta_seconds / 60

            timestamp = datetime.now().strftime("%H:%M:%S")
            message = f"\n[{timestamp}] 📊 Step {step}/{total_steps} ({progress:.1f}%)\n"
            message += f"   ⚡ Speed: {steps_per_sec:.2f} steps/sec | ETA: {eta_mins:.1f} min\n"

            if "loss" in logs:
                message += f"   📉 Loss: {logs['loss']:.4f}\n"
            if "learning_rate" in logs:
                message += f"   📚 LR: {logs['learning_rate']:.2e}\n"
            if "eval_loss" in logs:
                message += f"   ✅ Val Loss: {logs['eval_loss']:.4f}\n"

            print(message, end="", flush=True)

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = "\n" + "=" * 60 + "\n"
        message += f"✨ Training Completed - {timestamp}\n"
        message += f"   Duration: {elapsed/60:.2f} minutes\n"
        message += "=" * 60 + "\n"
        print(message, flush=True)


class KayaTrainer:
    def __init__(
        self,
        model_id: str,
        max_seq_length: int = 4096,
        lora_config: dict = None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        self.model_id = model_id
        self.max_seq_length = max_seq_length
        self.lora_config = lora_config or {}
        self.resume_from_checkpoint = resume_from_checkpoint
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Loads the model and tokenizer with 4-bit quantization and LoRA adapters.

        When ``resume_from_checkpoint`` is set on the trainer, the existing LoRA
        weights are loaded from that path (incremental training).  Otherwise a
        fresh set of LoRA adapters is applied to the base model (full training).
        """
        if self.resume_from_checkpoint:
            print(
                f"Loading existing LoRA checkpoint for incremental training: {self.resume_from_checkpoint}",
                flush=True,
            )
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=self.resume_from_checkpoint,
                max_seq_length=self.max_seq_length,
                dtype=None,  # Auto detection
                load_in_4bit=True,
                low_cpu_mem_usage=True,  # Load tensor-by-tensor to GPU; prevents ~28 GB CPU RAM spike
            )
            gc.collect()
            torch.cuda.empty_cache()
            print(
                f"✓ Loaded LoRA checkpoint from {self.resume_from_checkpoint}",
                flush=True,
            )
        else:
            print(f"Loading model: {self.model_id}", flush=True)
            _rss = lambda: psutil.Process(os.getpid()).memory_info().rss / 1e9
            print(f"   [MEM pre-from_pretrained] RSS={_rss():.2f}GB", flush=True)
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=self.model_id,
                max_seq_length=self.max_seq_length,
                dtype=None,  # Auto detection
                load_in_4bit=True,
                low_cpu_mem_usage=True,  # Load tensor-by-tensor to GPU; prevents ~28 GB CPU RAM spike
            )

            print(f"   [MEM post-from_pretrained] RSS={_rss():.2f}GB", flush=True)

            # Add LoRA adapters
            lora_r = self.lora_config.get("r", 16)
            lora_alpha = self.lora_config.get("alpha", 16)
            lora_dropout = self.lora_config.get("dropout", 0)
            target_modules = self.lora_config.get(
                "target_modules",
                [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            )

            self.model = FastModel.get_peft_model(
                self.model,
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

            print(f"   [MEM post-LoRA] RSS={_rss():.2f}GB", flush=True)

            # Free any temporary CPU allocations from model loading immediately
            gc.collect()
            torch.cuda.empty_cache()
            print(f"✓ Model loaded with LoRA (r={lora_r}, alpha={lora_alpha})", flush=True)

        # Apply Gemma 4 chat template to tokenizer
        self.tokenizer = get_chat_template(self.tokenizer, "gemma-4")

        print(f"✓ Trainable parameters: {self.model.print_trainable_parameters()}", flush=True)

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "outputs",
        training_config: dict = None,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Runs the training loop with validation support.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Optional validation dataset.
            output_dir: Directory to save checkpoints.
            training_config: Dictionary with training hyperparameters.
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Default config
        config = {
            "max_steps": 500,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "warmup_steps": 50,
            "weight_decay": 0.01,
            "optim": "adamw_8bit",
            "lr_scheduler_type": "cosine",
            "logging_steps": 10,
            "save_steps": 100,
            "eval_steps": 50,
            "seed": 3407,
            "dataset_num_proc": 1,
        }

        # Override with user config
        if training_config:
            config.update(training_config)

        # Apply incremental-training overrides when a checkpoint is loaded
        is_incremental = self.resume_from_checkpoint is not None
        if is_incremental:
            if "incremental_steps" in config:
                config["max_steps"] = config["incremental_steps"]
            if "incremental_learning_rate" in config:
                config["learning_rate"] = config["incremental_learning_rate"]

        # Log training mode
        if is_incremental:
            print(
                f"\n🔄 Mode: INCREMENTAL TRAINING (continuing from {self.resume_from_checkpoint})",
                flush=True,
            )
        else:
            print("\n🆕 Mode: FULL TRAINING (from scratch)", flush=True)

        print(f"\n📋 Training Configuration:", flush=True)
        print(f"   • Max steps: {config['max_steps']}", flush=True)
        print(
            f"   • Batch size: {config['per_device_train_batch_size']} (effective: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']})",
            flush=True
        )
        print(f"   • Learning rate: {config['learning_rate']}", flush=True)
        print(f"   • Train samples: {len(train_dataset)}", flush=True)
        if eval_dataset:
            print(f"   • Validation samples: {len(eval_dataset)}", flush=True)
        print(flush=True)

        _rss = lambda: psutil.Process(os.getpid()).memory_info().rss / 1e9
        print(f"   [MEM pre-SFTTrainer] RSS={_rss():.2f}GB", flush=True)

        # Formatting function to extract pre-formatted text from dataset
        def formatting_func(examples):
            """Extract the formatted_text field from batched examples, stripping BOS if present."""
            bos = self.tokenizer.bos_token or ""
            # Handle both single example and batched examples
            if isinstance(examples["formatted_text"], list):
                return [t.removeprefix(bos).strip() for t in examples["formatted_text"]]
            else:
                return [examples["formatted_text"].removeprefix(bos).strip()]

        trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            formatting_func=formatting_func,
            callbacks=[ProgressCallback(), MemoryMonitorCallback(log_every=10)],
            args=SFTConfig(
                dataset_text_field="formatted_text",
                max_length=self.max_seq_length,
                dataset_num_proc=config.get("dataset_num_proc", 1),
                packing=False,
                dataloader_pin_memory=False,  # Avoid page-locked RAM allocation (~1-2 GB savings)
                dataloader_num_workers=0,  # Prevent forking data loader workers (COW + 107GB vaddr = OOM)
                per_device_train_batch_size=config["per_device_train_batch_size"],
                per_device_eval_batch_size=config.get("per_device_eval_batch_size", 1),
                gradient_accumulation_steps=config["gradient_accumulation_steps"],
                warmup_steps=config["warmup_steps"],
                max_steps=config["max_steps"],
                learning_rate=config["learning_rate"],
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=config["logging_steps"],
                optim=config["optim"],
                weight_decay=config["weight_decay"],
                lr_scheduler_type=config["lr_scheduler_type"],
                seed=config["seed"],
                output_dir=output_dir,
                save_steps=config.get("save_steps", 100),
                eval_strategy="steps" if eval_dataset else "no",
                eval_steps=config.get("eval_steps", 50) if eval_dataset else None,
                load_best_model_at_end=True if eval_dataset else False,
                metric_for_best_model="eval_loss" if eval_dataset else None,
                report_to="none",  # Disable wandb/tensorboard by default
            ),
        )

        print(f"   [MEM post-SFTTrainer] RSS={_rss():.2f}GB", flush=True)

        # Mask everything but assistant responses — only compute loss on model turns
        # Use num_proc=1 to avoid forking 16+ COW child processes that trigger the
        # OOM killer on the parent's ~107 GB virtual address space.
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|turn>user\n",
            response_part="<|turn>model\n",
            num_proc=1,
        )

        print(f"   [MEM post-train_on_responses_only] RSS={_rss():.2f}GB", flush=True)

        trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return trainer, trainer_stats

    def save_model(self, output_dir: str = "outputs"):
        """Saves the LoRA adapters."""
        if self.model is None:
            raise ValueError("Model not loaded.")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"💾 Model saved to {output_dir}", flush=True)