import json
import shutil
from pathlib import Path


SYNTHETIC_PATH = Path(
    "data/synthetic/task4_synthetic_pilot.jsonl"
)
REAL_PATH = Path(
    "data/lora/train_real_sft.jsonl"
)
MIXED_PATH = Path(
    "data/lora/train_mixed_pilot_sft.jsonl"
)

SYNTHETIC_BACKUP = Path(
    "data/synthetic/task4_synthetic_pilot_before_cleanup.jsonl"
)
MIXED_BACKUP = Path(
    "data/lora/train_mixed_pilot_before_cleanup_sft.jsonl"
)


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
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


def normalize_candidate(value: object) -> str:
    if not isinstance(value, str):
        return ""

    text = value.strip()

    # Обрабатывает:
    # ["query"]
    # "query"
    # "[\"query\"]"
    for _ in range(4):
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            break

        if isinstance(parsed, str):
            text = parsed.strip()
            continue

        if (
            isinstance(parsed, list)
            and len(parsed) == 1
            and isinstance(parsed[0], str)
        ):
            text = parsed[0].strip()
            continue

        break

    return text.strip().strip('"').strip()


def find_assistant_message(item: dict) -> dict:
    messages = item.get("messages", [])

    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message

    raise ValueError("Item does not contain an assistant message")


def main() -> None:
    synthetic_items = load_jsonl(SYNTHETIC_PATH)
    real_items = load_jsonl(REAL_PATH)

    if not SYNTHETIC_BACKUP.exists():
        shutil.copy2(
            SYNTHETIC_PATH,
            SYNTHETIC_BACKUP,
        )

    if MIXED_PATH.exists() and not MIXED_BACKUP.exists():
        shutil.copy2(
            MIXED_PATH,
            MIXED_BACKUP,
        )

    changed = 0
    invalid = []

    for index, item in enumerate(synthetic_items, start=1):
        assistant_message = find_assistant_message(item)

        original_query = str(
            assistant_message.get("content", "")
        )
        cleaned_query = normalize_candidate(original_query)

        if cleaned_query != original_query:
            changed += 1

            metadata = item.setdefault("metadata", {})
            metadata["query_normalized_after_generation"] = True
            metadata["original_assistant_query"] = original_query

            assistant_message["content"] = cleaned_query

            print(f"\nChanged example {index}:")
            print(f"OLD: {original_query}")
            print(f"NEW: {cleaned_query}")

        token_count = len(cleaned_query.split())

        if (
            not cleaned_query
            or not 5 <= token_count <= 20
            or cleaned_query.startswith(("[", "{"))
            or cleaned_query.endswith(("]", "}"))
        ):
            invalid.append(
                {
                    "index": index,
                    "query": cleaned_query,
                    "token_count": token_count,
                }
            )

    if invalid:
        print("\nInvalid queries remain:")

        for problem in invalid:
            print(problem)

        raise RuntimeError(
            "Cleanup failed: invalid queries remain"
        )

    save_jsonl(
        SYNTHETIC_PATH,
        synthetic_items,
    )

    # Пересобираем mixed dataset:
    # 248 real + 100 cleaned synthetic.
    mixed_items = real_items + synthetic_items

    save_jsonl(
        MIXED_PATH,
        mixed_items,
    )

    print()
    print("=" * 72)
    print("SYNTHETIC QUERY CLEANUP FINISHED")
    print("=" * 72)
    print(f"Changed queries: {changed}")
    print(f"Synthetic examples: {len(synthetic_items)}")
    print(f"Real examples: {len(real_items)}")
    print(f"Mixed train size: {len(mixed_items)}")
    print(f"Invalid queries remaining: {len(invalid)}")
    print()
    print(f"Saved synthetic: {SYNTHETIC_PATH}")
    print(f"Saved mixed train: {MIXED_PATH}")
    print(f"Backup synthetic: {SYNTHETIC_BACKUP}")
    print(f"Backup mixed: {MIXED_BACKUP}")


if __name__ == "__main__":
    main()