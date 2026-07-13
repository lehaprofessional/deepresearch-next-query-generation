from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required_sections = ("model", "data", "lora", "training")
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ValueError(f"Missing config sections: {missing}")

    return config


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    items: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {error}"
                ) from error

            messages = item.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(
                    f"Missing messages in {path}, line {line_number}"
                )

            if messages[-1].get("role") != "assistant":
                raise ValueError(
                    f"Last message must be assistant in {path}, "
                    f"line {line_number}"
                )

            items.append(item)

    if not items:
        raise ValueError(f"Dataset is empty: {path}")

    return items


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dtype(name: str) -> torch.dtype:
    normalized = name.lower().strip()

    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    if normalized not in mapping:
        raise ValueError(
            f"Unsupported dtype '{name}'. "
            f"Use one of: {sorted(mapping)}"
        )

    return mapping[normalized]


def render_chat(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except Exception as error:
            raise RuntimeError(
                "Tokenizer chat template failed. "
                "Use an instruction/chat backbone with a valid chat template."
            ) from error

    raise RuntimeError(
        "Tokenizer does not provide apply_chat_template(). "
        "A chat/instruction backbone is required for this pipeline."
    )


class ChatSFTDataset(Dataset):
    def __init__(
        self,
        items: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.examples: list[dict[str, list[int]]] = []
        self.target_token_lengths: list[int] = []
        self.total_token_lengths: list[int] = []

        for index, item in enumerate(items):
            messages = item["messages"]
            prompt_messages = messages[:-1]

            full_text = render_chat(
                tokenizer,
                messages,
                add_generation_prompt=False,
            )
            prompt_text = render_chat(
                tokenizer,
                prompt_messages,
                add_generation_prompt=True,
            )

            full_ids = tokenizer(
                full_text,
                add_special_tokens=False,
            )["input_ids"]

            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
            )["input_ids"]

            prompt_length = len(prompt_ids)

            # Preserve the assistant target when an example is too long.
            if len(full_ids) > max_length:
                overflow = len(full_ids) - max_length
                full_ids = full_ids[overflow:]
                prompt_length = max(0, prompt_length - overflow)

            labels = list(full_ids)
            labels[:prompt_length] = [-100] * prompt_length

            target_token_count = sum(
                label != -100 for label in labels
            )

            if target_token_count == 0:
                raise ValueError(
                    f"Example {index} has no trainable assistant tokens "
                    f"after truncation. Increase max_length."
                )

            self.examples.append(
                {
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                }
            )
            self.target_token_lengths.append(target_token_count)
            self.total_token_lengths.append(len(full_ids))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


@dataclass
class CausalLMCollator:
    pad_token_id: int
    pad_to_multiple_of: int | None = 8

    def __call__(
        self,
        features: list[dict[str, list[int]]],
    ) -> dict[str, torch.Tensor]:
        max_length = max(
            len(feature["input_ids"])
            for feature in features
        )

        if self.pad_to_multiple_of:
            multiple = self.pad_to_multiple_of
            max_length = (
                math.ceil(max_length / multiple) * multiple
            )

        batch_size = len(features)

        input_ids = torch.full(
            (batch_size, max_length),
            self.pad_token_id,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, max_length),
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, max_length),
            -100,
            dtype=torch.long,
        )

        for row, feature in enumerate(features):
            length = len(feature["input_ids"])

            input_ids[row, :length] = torch.tensor(
                feature["input_ids"],
                dtype=torch.long,
            )
            attention_mask[row, :length] = torch.tensor(
                feature["attention_mask"],
                dtype=torch.long,
            )
            labels[row, :length] = torch.tensor(
                feature["labels"],
                dtype=torch.long,
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def create_training_arguments(
    config: dict[str, Any],
    output_dir: Path,
) -> TrainingArguments:
    training = config["training"]

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(
            training.get("num_train_epochs", 5)
        ),
        "per_device_train_batch_size": int(
            training.get("train_batch_size", 1)
        ),
        "per_device_eval_batch_size": int(
            training.get("eval_batch_size", 1)
        ),
        "gradient_accumulation_steps": int(
            training.get("gradient_accumulation_steps", 8)
        ),
        "learning_rate": float(
            training.get("learning_rate", 1e-4)
        ),
        "weight_decay": float(
            training.get("weight_decay", 0.0)
        ),
        "warmup_ratio": float(
            training.get("warmup_ratio", 0.05)
        ),
        "logging_steps": int(
            training.get("logging_steps", 5)
        ),
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": int(
            training.get("save_total_limit", 2)
        ),
        "report_to": [],
        "remove_unused_columns": False,
        "seed": int(training.get("seed", 42)),
        "data_seed": int(training.get("seed", 42)),
        "bf16": bool(training.get("bf16", True)),
        "fp16": bool(training.get("fp16", False)),
        "gradient_checkpointing": bool(
            training.get("gradient_checkpointing", True)
        ),
        "optim": training.get("optim", "adamw_torch"),
        "lr_scheduler_type": training.get(
            "lr_scheduler_type",
            "linear",
        ),
        "dataloader_num_workers": int(
            training.get("dataloader_num_workers", 0)
        ),
    }

    signature = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    if "eval_strategy" in signature:
        kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in signature:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        raise RuntimeError(
            "Installed transformers version exposes neither "
            "eval_strategy nor evaluation_strategy."
        )

    if "gradient_checkpointing_kwargs" in signature:
        kwargs["gradient_checkpointing_kwargs"] = {
            "use_reentrant": False
        }

    return TrainingArguments(**kwargs)


def build_trainer(
    *,
    model: Any,
    tokenizer: Any,
    training_args: TrainingArguments,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    collator: CausalLMCollator,
    early_stopping_patience: int,
) -> Trainer:
    kwargs: dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
        "callbacks": [
            EarlyStoppingCallback(
                early_stopping_patience=(
                    early_stopping_patience
                )
            )
        ],
    }

    signature = inspect.signature(
        Trainer.__init__
    ).parameters

    if "processing_class" in signature:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in signature:
        kwargs["tokenizer"] = tokenizer

    return Trainer(**kwargs)


def count_parameters(model: Any) -> tuple[int, int]:
    total = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def dataset_stats(dataset: ChatSFTDataset) -> dict[str, Any]:
    total_lengths = dataset.total_token_lengths
    target_lengths = dataset.target_token_lengths

    return {
        "examples": len(dataset),
        "total_tokens": {
            "minimum": min(total_lengths),
            "mean": float(np.mean(total_lengths)),
            "median": float(np.median(total_lengths)),
            "maximum": max(total_lengths),
        },
        "assistant_target_tokens": {
            "minimum": min(target_lengths),
            "mean": float(np.mean(target_lengths)),
            "median": float(np.median(target_lengths)),
            "maximum": max(target_lengths),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Configurable LoRA training pipeline for Task 4."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a JSON experiment config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config["model"]
    data_config = config["data"]
    lora_config = config["lora"]
    training_config = config["training"]

    seed = int(training_config.get("seed", 42))
    set_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this training config."
        )

    output_dir = Path(training_config["output_dir"])
    metrics_output = Path(
        training_config.get(
            "metrics_output",
            f"results/{config.get('experiment_name', 'lora')}_metrics.json",
        )
    )

    overwrite_output_dir = bool(
        training_config.get("overwrite_output_dir", False)
    )

    if (
        output_dir.exists()
        and any(output_dir.iterdir())
        and not overwrite_output_dir
    ):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}\n"
            "Use a new directory or set "
            '"overwrite_output_dir": true in the config.'
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)

    train_path = Path(data_config["train_path"])
    validation_path = Path(
        data_config["validation_path"]
    )

    train_items = load_jsonl(train_path)
    validation_items = load_jsonl(validation_path)

    model_name = model_config["name"]
    max_length = int(
        model_config.get("max_length", 2048)
    )
    dtype = resolve_dtype(
        model_config.get("dtype", "bfloat16")
    )

    print("=" * 72)
    print("TASK 4 CONFIGURABLE LORA TRAINING")
    print("=" * 72)
    print(
        f"Experiment: "
        f"{config.get('experiment_name', 'unnamed')}"
    )
    print(f"Backbone: {model_name}")
    print(f"Train file: {train_path}")
    print(f"Validation file: {validation_path}")
    print(f"Train examples: {len(train_items)}")
    print(
        f"Validation examples: "
        f"{len(validation_items)}"
    )
    print(f"Maximum sequence length: {max_length}")
    print(f"Output directory: {output_dir}")
    print()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(
            model_config.get("trust_remote_code", False)
        ),
        use_fast=bool(
            model_config.get("use_fast_tokenizer", True)
        ),
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                "Tokenizer has neither pad_token_id nor eos_token_id."
            )
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    train_dataset = ChatSFTDataset(
        train_items,
        tokenizer,
        max_length,
    )
    validation_dataset = ChatSFTDataset(
        validation_items,
        tokenizer,
        max_length,
    )

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "trust_remote_code": bool(
            model_config.get("trust_remote_code", False)
        ),
        "low_cpu_mem_usage": True,
    }

    attention_implementation = model_config.get(
        "attn_implementation"
    )
    if attention_implementation:
        model_kwargs["attn_implementation"] = (
            attention_implementation
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs,
    )

    model.config.use_cache = False

    if bool(
        training_config.get(
            "gradient_checkpointing",
            True,
        )
    ):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={
                    "use_reentrant": False
                }
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=int(lora_config.get("rank", 16)),
        lora_alpha=int(
            lora_config.get("alpha", 32)
        ),
        lora_dropout=float(
            lora_config.get("dropout", 0.05)
        ),
        bias=lora_config.get("bias", "none"),
        target_modules=list(
            lora_config["target_modules"]
        ),
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)

    total_parameters, trainable_parameters = (
        count_parameters(model)
    )
    trainable_share = (
        trainable_parameters / total_parameters
        if total_parameters
        else 0.0
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )
    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )
    print(
        f"Trainable share: "
        f"{trainable_share:.3%}"
    )
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )
    print()

    training_args = create_training_arguments(
        config,
        output_dir,
    )

    collator = CausalLMCollator(
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=int(
            data_config.get(
                "pad_to_multiple_of",
                8,
            )
        ),
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        training_args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        collator=collator,
        early_stopping_patience=int(
            training_config.get(
                "early_stopping_patience",
                2,
            )
        ),
    )

    start_time = time.time()
    train_result = trainer.train(
        resume_from_checkpoint=training_config.get(
            "resume_from_checkpoint"
        )
    )
    runtime_seconds = time.time() - start_time

    final_eval_metrics = trainer.evaluate()

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    peak_vram_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    metrics = {
        "experiment_name": config.get(
            "experiment_name"
        ),
        "backbone": model_name,
        "train_path": str(train_path),
        "validation_path": str(validation_path),
        "train_examples": len(train_items),
        "validation_examples": len(
            validation_items
        ),
        "synthetic_data_used_for_validation": False,
        "max_length": max_length,
        "dtype": str(dtype),
        "lora": {
            "rank": peft_config.r,
            "alpha": peft_config.lora_alpha,
            "dropout": peft_config.lora_dropout,
            "target_modules": list(
                lora_config["target_modules"]
            ),
            "trainable_parameters": (
                trainable_parameters
            ),
            "total_parameters": total_parameters,
            "trainable_share": trainable_share,
        },
        "training": {
            "runtime_seconds": runtime_seconds,
            "peak_vram_gb": peak_vram_gb,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "global_step": trainer.state.global_step,
            "train_metrics": train_result.metrics,
            "final_eval_metrics": final_eval_metrics,
        },
        "dataset_stats": {
            "train": dataset_stats(train_dataset),
            "validation": dataset_stats(
                validation_dataset
            ),
        },
        "config": config,
    }

    with metrics_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print()
    print("=" * 72)
    print("TRAINING FINISHED")
    print("=" * 72)
    print(f"Runtime: {runtime_seconds:.1f} sec")
    print(f"Peak VRAM: {peak_vram_gb:.2f} GB")
    print(
        f"Best checkpoint: "
        f"{trainer.state.best_model_checkpoint}"
    )
    print(
        f"Best eval loss: "
        f"{trainer.state.best_metric}"
    )
    print(f"Saved adapter: {output_dir}")
    print(f"Saved metrics: {metrics_output}")


if __name__ == "__main__":
    main()
