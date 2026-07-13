import json
import random
from pathlib import Path
from statistics import mean, median

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_PATH = Path("data/lora/train_real_sft.jsonl")
VAL_PATH = Path("data/lora/val_real_sft.jsonl")

OUTPUT_DIR = Path("models/qwen2.5-1.5b-task4-lora-smoke")

MAX_LENGTH = 2048
SMOKE_TRAIN_EXAMPLES = 12
SMOKE_VAL_EXAMPLES = 4
SMOKE_STEPS = 8
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


def token_length(tokenizer, item: dict) -> int:
    formatted_text = tokenizer.apply_chat_template(
        item["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )

    encoded = tokenizer(
        formatted_text,
        add_special_tokens=False,
        truncation=False,
    )

    return len(encoded["input_ids"])


def print_length_statistics(
    name: str,
    tokenizer,
    items: list[dict],
) -> list[int]:
    lengths = [
        token_length(tokenizer, item)
        for item in items
    ]

    sorted_lengths = sorted(lengths)
    p90_index = max(
        0,
        min(
            len(sorted_lengths) - 1,
            int(len(sorted_lengths) * 0.9),
        ),
    )

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)
    print("Examples:", len(lengths))
    print(f"Average tokens: {mean(lengths):.1f}")
    print(f"Median tokens:  {median(lengths):.1f}")
    print("P90 tokens:", sorted_lengths[p90_index])
    print("Maximum tokens:", max(lengths))
    print(
        f"Examples over {MAX_LENGTH} tokens:",
        sum(length > MAX_LENGTH for length in lengths),
    )

    return lengths


def select_smoke_examples(
    tokenizer,
    items: list[dict],
    count: int,
    seed: int,
) -> list[dict]:
    suitable = [
        item
        for item in items
        if token_length(tokenizer, item) <= MAX_LENGTH
    ]

    if len(suitable) < count:
        raise ValueError(
            f"Only {len(suitable)} examples fit MAX_LENGTH={MAX_LENGTH}, "
            f"but {count} are required."
        )

    random_generator = random.Random(seed)
    return random_generator.sample(suitable, count)


def convert_to_prompt_completion(item: dict) -> dict:
    messages = item["messages"]

    if len(messages) < 2:
        raise ValueError("Conversation contains too few messages.")

    if messages[-1].get("role") != "assistant":
        raise ValueError(
            "The final message must have role='assistant'."
        )

    return {
        "prompt": messages[:-1],
        "completion": [messages[-1]],
    }


def print_gpu_status(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))

    if not torch.cuda.is_available():
        print("CUDA is not available.")
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


def generate_example(
    trainer: SFTTrainer,
    tokenizer,
    original_val_item: dict,
) -> None:
    prompt_messages = original_val_item["messages"][:-1]
    target = original_val_item["messages"][-1]["content"]

    inputs = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {
        key: value.to(trainer.model.device)
        for key, value in inputs.items()
    }

    trainer.model.eval()

    with torch.inference_mode():
        generated_ids = trainer.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]

    generated_text = tokenizer.decode(
        generated_ids[0][prompt_length:],
        skip_special_tokens=True,
    ).strip()

    print()
    print("=" * 72)
    print("SMOKE GENERATION")
    print("=" * 72)
    print("Target query:")
    print(target)
    print()
    print("Generated query:")
    print(generated_text)


def main() -> None:
    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. LoRA training must run on the GPU."
        )

    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())

    print()
    print("Loading tokenizer:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    train_items = load_jsonl(TRAIN_PATH)
    val_items = load_jsonl(VAL_PATH)

    print_length_statistics(
        "FULL TRAIN TOKEN LENGTHS",
        tokenizer,
        train_items,
    )

    print_length_statistics(
        "FULL VALIDATION TOKEN LENGTHS",
        tokenizer,
        val_items,
    )

    smoke_train_original = select_smoke_examples(
        tokenizer=tokenizer,
        items=train_items,
        count=SMOKE_TRAIN_EXAMPLES,
        seed=SEED,
    )

    smoke_val_original = select_smoke_examples(
        tokenizer=tokenizer,
        items=val_items,
        count=SMOKE_VAL_EXAMPLES,
        seed=SEED + 1,
    )

    smoke_train = Dataset.from_list(
        [
            convert_to_prompt_completion(item)
            for item in smoke_train_original
        ]
    )

    smoke_val = Dataset.from_list(
        [
            convert_to_prompt_completion(item)
            for item in smoke_val_original
        ]
    )

    print()
    print("Smoke train examples:", len(smoke_train))
    print("Smoke validation examples:", len(smoke_val))

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

        max_steps=SMOKE_STEPS,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=1,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,

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

        eval_strategy="steps",
        eval_steps=4,

        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,

        save_strategy="no",
        report_to="none",

        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,

        seed=SEED,
        data_seed=SEED,

        model_init_kwargs={
            "dtype": torch.bfloat16,
            "attn_implementation": "sdpa",
        },
    )

    print()
    print("Creating SFTTrainer...")
    print("The model may be downloaded on the first launch.")

    trainer = SFTTrainer(
        model=MODEL_NAME,
        args=training_config,
        train_dataset=smoke_train,
        eval_dataset=smoke_val,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print()
    print("Trainable parameters:")
    trainer.model.print_trainable_parameters()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print_gpu_status("GPU BEFORE TRAINING")

    print()
    print("=" * 72)
    print("STARTING SMOKE TRAINING")
    print("=" * 72)

    train_result = trainer.train()

    print()
    print("Training metrics:")
    for key, value in train_result.metrics.items():
        print(f"{key}: {value}")

    print_gpu_status("GPU AFTER TRAINING")

    print()
    print("Running final validation loss...")
    eval_metrics = trainer.evaluate()

    print()
    print("Validation metrics:")
    for key, value in eval_metrics.items():
        print(f"{key}: {value}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print()
    print("LoRA adapter saved to:")
    print(OUTPUT_DIR)

    generate_example(
        trainer=trainer,
        tokenizer=tokenizer,
        original_val_item=smoke_val_original[0],
    )


if __name__ == "__main__":
    main()