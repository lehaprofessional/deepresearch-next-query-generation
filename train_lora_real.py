import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, EarlyStoppingCallback, set_seed
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_PATH = Path("data/lora/train_real_sft.jsonl")
VAL_PATH = Path("data/lora/val_real_sft.jsonl")

OUTPUT_DIR = Path("models/qwen2.5-1.5b-task4-lora-real")

MAX_LENGTH = 2048
NUM_EPOCHS = 5
SEED = 42


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    items = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {error}"
                ) from error

    return items


def convert_to_prompt_completion(item: dict) -> dict:
    messages = item["messages"]

    if not messages:
        raise ValueError("Empty conversation.")

    if messages[-1].get("role") != "assistant":
        raise ValueError(
            "The final message must have role='assistant'."
        )

    return {
        "prompt": messages[:-1],
        "completion": [messages[-1]],
        "research_question": item.get(
            "metadata", {}
        ).get("research_question", ""),
        "query_id": item.get(
            "metadata", {}
        ).get("query_id", ""),
    }


def print_gpu_status(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))

    if not torch.cuda.is_available():
        print("CUDA is unavailable.")
        return

    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "Allocated VRAM:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "GB",
    )
    print(
        "Reserved VRAM:",
        round(torch.cuda.memory_reserved() / 1024**3, 2),
        "GB",
    )
    print(
        "Peak allocated VRAM:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "GB",
    )


def main() -> None:
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Training must run on the GPU."
        )

    print("=" * 72)
    print("TASK 4 REAL-ONLY LORA TRAINING")
    print("=" * 72)
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
    print("Base model:", MODEL_NAME)
    print("Output:", OUTPUT_DIR)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    raw_train = load_jsonl(TRAIN_PATH)
    raw_val = load_jsonl(VAL_PATH)

    train_dataset = Dataset.from_list(
        [
            convert_to_prompt_completion(item)
            for item in raw_train
        ]
    )

    val_dataset = Dataset.from_list(
        [
            convert_to_prompt_completion(item)
            for item in raw_val
        ]
    )

    print()
    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(val_dataset))

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,

        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),

        num_train_epochs=NUM_EPOCHS,

        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        # Эффективный batch size = 1 × 8 = 8.
        gradient_accumulation_steps=8,

        bf16=True,
        fp16=False,
        tf32=True,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        },

        max_length=MAX_LENGTH,
        completion_only_loss=True,
        packing=False,

        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=5,
        logging_first_step=True,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        save_total_limit=2,

        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,

        report_to="none",

        seed=SEED,
        data_seed=SEED,

        model_init_kwargs={
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },
    )

    trainer = SFTTrainer(
        model=MODEL_NAME,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2,
                early_stopping_threshold=0.01,
            )
        ],
    )

    print()
    print("Trainable parameters:")
    trainer.model.print_trainable_parameters()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print_gpu_status("GPU BEFORE TRAINING")

    print()
    print("=" * 72)
    print("STARTING FULL REAL-ONLY TRAINING")
    print("=" * 72)

    train_result = trainer.train()

    print()
    print("=" * 72)
    print("TRAINING FINISHED")
    print("=" * 72)

    print()
    print("Training metrics:")

    for key, value in train_result.metrics.items():
        print(f"{key}: {value}")

    print_gpu_status("GPU AFTER TRAINING")

    print()
    print("Evaluating the best checkpoint...")

    eval_metrics = trainer.evaluate()

    print()
    print("Final validation metrics:")

    for key, value in eval_metrics.items():
        print(f"{key}: {value}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    metrics_path = OUTPUT_DIR / "final_metrics.json"

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "model_name": MODEL_NAME,
                "training_examples": len(train_dataset),
                "validation_examples": len(val_dataset),
                "epochs_requested": NUM_EPOCHS,
                "max_length": MAX_LENGTH,
                "train_metrics": train_result.metrics,
                "eval_metrics": eval_metrics,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("Best checkpoint:")
    print(trainer.state.best_model_checkpoint)

    print()
    print("Best validation metric:")
    print(trainer.state.best_metric)

    print()
    print("LoRA adapter saved to:")
    print(OUTPUT_DIR)

    print()
    print("Metrics saved to:")
    print(metrics_path)


if __name__ == "__main__":
    main()