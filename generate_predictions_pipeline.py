from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    for section in ("model", "data", "generation"):
        if section not in config:
            raise ValueError(f"Missing config section: {section}")

    return config


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

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


def save_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )


def clean_query(text: str) -> str:
    """
    Keep the exact cleaning behaviour used by the original
    generate_qwen_lora_predictions.py experiment.
    """
    text = text.strip()

    prefixes = [
        "Search query:",
        "Query:",
        "Next search query:",
        "Generated query:",
    ]

    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    text = text.strip("\"'` ")

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if lines:
        text = lines[0]

    return text


def generate_query(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    generation_config: dict[str, Any],
) -> str:
    """
    This intentionally mirrors the original verified inference code:
    - apply_chat_template(..., tokenize=True)
    - no manual text rendering
    - no implicit truncation
    - deterministic generation
    - repetition_penalty=1.05 by default
    """
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    max_input_tokens = generation_config.get(
        "max_input_tokens"
    )

    if max_input_tokens is not None:
        max_input_tokens = int(max_input_tokens)

        if encoded["input_ids"].shape[1] > max_input_tokens:
            encoded = {
                key: value[:, -max_input_tokens:]
                for key, value in encoded.items()
            }

    encoded = {
        key: value.to(model.device)
        for key, value in encoded.items()
    }

    prompt_length = encoded["input_ids"].shape[1]

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": int(
            generation_config.get("max_new_tokens", 64)
        ),
        "do_sample": bool(
            generation_config.get("do_sample", False)
        ),
        "repetition_penalty": float(
            generation_config.get(
                "repetition_penalty",
                1.05,
            )
        ),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if generation_kwargs["do_sample"]:
        generation_kwargs["temperature"] = float(
            generation_config.get("temperature", 1.0)
        )
        generation_kwargs["top_p"] = float(
            generation_config.get("top_p", 1.0)
        )

    with torch.inference_mode():
        generated_ids = model.generate(
            **encoded,
            **generation_kwargs,
        )

    new_tokens = generated_ids[0][prompt_length:]

    generated_text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    return clean_query(generated_text)


def load_model(
    *,
    model_name: str,
    adapter_path: Path | None,
    model_config: dict[str, Any],
) -> Any:
    dtype_name = str(
        model_config.get("dtype", "bfloat16")
    ).lower()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    if dtype_name not in dtype_map:
        raise ValueError(
            f"Unsupported dtype: {dtype_name}"
        )

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype_map[dtype_name],
        device_map=model_config.get(
            "device_map",
            "cuda",
        ),
        attn_implementation=model_config.get(
            "attn_implementation",
            "sdpa",
        ),
        trust_remote_code=bool(
            model_config.get(
                "trust_remote_code",
                False,
            )
        ),
    )

    if adapter_path is None:
        return base_model

    return PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Configurable Task 4 inference pipeline that reproduces "
            "the original verified Qwen/LoRA generation procedure."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="0 means all examples.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    config = load_json(args.config)
    model_config = config["model"]
    data_config = config["data"]
    generation_config = config["generation"]

    model_name = str(model_config["name"])
    adapter_value = model_config.get("adapter_path")
    adapter_path = (
        Path(adapter_value)
        if adapter_value
        else None
    )

    input_path = Path(data_config["input_path"])
    output_path = Path(data_config["output_path"])
    run_name = str(
        config.get("run_name", "task4_inference")
    )

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if adapter_path is not None and not adapter_path.exists():
        raise FileNotFoundError(adapter_path)

    items = load_jsonl(input_path)

    if args.max_examples > 0:
        items = items[:args.max_examples]

    print("=" * 72)
    print("TASK 4 CONFIGURABLE INFERENCE")
    print("=" * 72)
    print("Run name:", run_name)
    print("Backbone:", model_name)
    print(
        "LoRA adapter:",
        adapter_path if adapter_path else "none",
    )
    print("Input:", input_path)
    print("Output:", output_path)
    print("Examples:", len(items))
    print("GPU:", torch.cuda.get_device_name(0))
    print()

    # The original verified script always loaded the tokenizer
    # from the base model, not from the adapter directory.
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=bool(
            model_config.get("use_fast", True)
        ),
        trust_remote_code=bool(
            model_config.get(
                "trust_remote_code",
                False,
            )
        ),
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(
        model_name=model_name,
        adapter_path=adapter_path,
        model_config=model_config,
    )
    model.eval()

    results: list[dict[str, Any]] = []

    print()
    print("=" * 72)
    print("Generating:", run_name)
    print("=" * 72)

    for index, item in enumerate(items, start=1):
        messages = item["messages"]
        prompt_messages = messages[:-1]
        target_query = messages[-1]["content"].strip()
        metadata = item.get("metadata", {})

        generated_query = generate_query(
            model=model,
            tokenizer=tokenizer,
            messages=prompt_messages,
            generation_config=generation_config,
        )

        result = {
            "research_question": metadata.get(
                "research_question",
                "",
            ),
            "query_id": metadata.get(
                "query_id",
                "",
            ),
            "target_query": target_query,
            "generated_query": generated_query,
            "model": run_name,
        }

        results.append(result)

        print(
            f"[{index:02d}/{len(items)}] "
            f"query_id={result['query_id']} | "
            f"{generated_query}"
        )

    save_jsonl(output_path, results)

    print()
    print("=" * 72)
    print("INFERENCE FINISHED")
    print("=" * 72)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
