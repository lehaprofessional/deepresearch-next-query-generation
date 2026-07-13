import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_PATH = Path("models/qwen2.5-1.5b-task4-lora-mixed")

VAL_PATH = Path("data/lora/val_real_sft.jsonl")

BASE_OUTPUT = Path("runs/qwen2.5_1.5b_base_predictions.jsonl")
LORA_OUTPUT = Path("runs/qwen2.5_1.5b_lora_mixed_predictions.jsonl")

MAX_NEW_TOKENS = 64


def load_jsonl(path: Path) -> list[dict]:
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


def save_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )


def clean_query(text: str) -> str:
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

    # РќР°Рј РЅСѓР¶РµРЅ С‚РѕР»СЊРєРѕ РїРµСЂРІС‹Р№ СЃРѕРґРµСЂР¶Р°С‚РµР»СЊРЅС‹Р№ РѕС‚РІРµС‚.
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if lines:
        text = lines[0]

    return text


def generate_query(
    model,
    tokenizer,
    messages: list[dict],
) -> str:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    encoded = {
        key: value.to(model.device)
        for key, value in encoded.items()
    }

    prompt_length = encoded["input_ids"].shape[1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated_ids[0][prompt_length:]

    generated_text = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    return clean_query(generated_text)


def run_generation(
    model,
    tokenizer,
    val_items: list[dict],
    output_path: Path,
    run_name: str,
) -> None:
    results = []

    model.eval()

    print()
    print("=" * 72)
    print("Generating:", run_name)
    print("=" * 72)

    for index, item in enumerate(val_items, start=1):
        messages = item["messages"]

        prompt_messages = messages[:-1]
        target_query = messages[-1]["content"].strip()

        generated_query = generate_query(
            model=model,
            tokenizer=tokenizer,
            messages=prompt_messages,
        )

        metadata = item.get("metadata", {})

        result = {
            "research_question": metadata.get(
                "research_question",
                "",
            ),
            "query_id": metadata.get("query_id", ""),
            "target_query": target_query,
            "generated_query": generated_query,
            "model": run_name,
        }

        results.append(result)

        print(
            f"[{index:02d}/{len(val_items)}] "
            f"query_id={result['query_id']} | "
            f"{generated_query}"
        )

    save_jsonl(output_path, results)

    print()
    print("Saved:", output_path)


def load_base_model():
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    if not VAL_PATH.exists():
        raise FileNotFoundError(VAL_PATH)

    if not LORA_PATH.exists():
        raise FileNotFoundError(LORA_PATH)

    print("GPU:", torch.cuda.get_device_name(0))
    print("Base model:", BASE_MODEL_NAME)
    print("LoRA adapter:", LORA_PATH)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    val_items = load_jsonl(VAL_PATH)

    print("Validation examples:", len(val_items))

    # ---------------------------------------------------------
    # 1. РСЃС…РѕРґРЅР°СЏ Qwen Р±РµР· РѕР±СѓС‡РµРЅРёСЏ
    # ---------------------------------------------------------

    base_model = load_base_model()

    run_generation(
        model=base_model,
        tokenizer=tokenizer,
        val_items=val_items,
        output_path=BASE_OUTPUT,
        run_name="qwen2.5_1.5b_base",
    )

    del base_model
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 2. Qwen СЃ РѕР±СѓС‡РµРЅРЅС‹Рј LoRA-Р°РґР°РїС‚РµСЂРѕРј
    # ---------------------------------------------------------

    lora_base_model = load_base_model()

    lora_model = PeftModel.from_pretrained(
        lora_base_model,
        str(LORA_PATH),
    )

    run_generation(
        model=lora_model,
        tokenizer=tokenizer,
        val_items=val_items,
        output_path=LORA_OUTPUT,
        run_name="qwen2.5_1.5b_lora_mixed",
    )

    print()
    print("Generation completed.")
    print("Base predictions:", BASE_OUTPUT)
    print("LoRA predictions:", LORA_OUTPUT)


if __name__ == "__main__":
    main()
