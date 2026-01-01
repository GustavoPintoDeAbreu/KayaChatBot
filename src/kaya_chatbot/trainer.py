import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from datasets import Dataset
from typing import Optional
import time


class ProgressCallback(TrainerCallback):
    """Custom callback for real-time training progress logging."""

    def __init__(self):
        self.start_time = None
        self.step_times = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        print("\n" + "=" * 60)
        print("🚀 Training Started")
        print("=" * 60)

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

            print(f"\n📊 Step {step}/{total_steps} ({progress:.1f}%)")
            print(
                f"   ⚡ Speed: {steps_per_sec:.2f} steps/sec | ETA: {eta_mins:.1f} min"
            )

            if "loss" in logs:
                print(f"   📉 Loss: {logs['loss']:.4f}")
            if "learning_rate" in logs:
                print(f"   📚 LR: {logs['learning_rate']:.2e}")
            if "eval_loss" in logs:
                print(f"   ✅ Val Loss: {logs['eval_loss']:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self.start_time
        print("\n" + "=" * 60)
        print(f"✨ Training Completed in {elapsed/60:.2f} minutes")
        print("=" * 60 + "\n")


class KayaTrainer:
    def __init__(
        self, model_id: str, max_seq_length: int = 4096, lora_config: dict = None
    ):
        self.model_id = model_id
        self.max_seq_length = max_seq_length
        self.lora_config = lora_config or {}
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Loads the model and tokenizer with 4-bit quantization and LoRA adapters."""
        print(f"Loading model: {self.model_id}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_id,
            max_seq_length=self.max_seq_length,
            dtype=None,  # Auto detection
            load_in_4bit=True,
        )

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

        self.model = FastLanguageModel.get_peft_model(
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
        )

        print(f"✓ Model loaded with LoRA (r={lora_r}, alpha={lora_alpha})")
        print(f"✓ Trainable parameters: {self.model.print_trainable_parameters()}")

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "outputs",
        training_config: dict = None,
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
            "dataset_num_proc": 2,
        }

        # Override with user config
        if training_config:
            config.update(training_config)

        print(f"\n📋 Training Configuration:")
        print(f"   • Max steps: {config['max_steps']}")
        print(
            f"   • Batch size: {config['per_device_train_batch_size']} (effective: {config['per_device_train_batch_size'] * config['gradient_accumulation_steps']})"
        )
        print(f"   • Learning rate: {config['learning_rate']}")
        print(f"   • Train samples: {len(train_dataset)}")
        if eval_dataset:
            print(f"   • Validation samples: {len(eval_dataset)}")
        print()

        # Formatting function to extract pre-formatted text from dataset
        def formatting_func(examples):
            """Extract the formatted_text field from batched examples."""
            # Handle both single example and batched examples
            if isinstance(examples["formatted_text"], list):
                return examples["formatted_text"]
            else:
                return [examples["formatted_text"]]

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            formatting_func=formatting_func,
            dataset_text_field="formatted_text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=config.get("dataset_num_proc", 2),
            packing=False,
            callbacks=[ProgressCallback()],
            args=TrainingArguments(
                per_device_train_batch_size=config["per_device_train_batch_size"],
                per_device_eval_batch_size=config.get("per_device_eval_batch_size", 2),
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

        trainer_stats = trainer.train()
        return trainer, trainer_stats

    def save_model(self, output_dir: str = "outputs"):
        """Saves the LoRA adapters."""
        if self.model is None:
            raise ValueError("Model not loaded.")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"💾 Model saved to {output_dir}")
