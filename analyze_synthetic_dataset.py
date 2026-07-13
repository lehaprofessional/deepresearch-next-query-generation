import json
import random
import statistics
from collections import Counter
from pathlib import Path


SYNTHETIC_PATH = Path("data/synthetic/task4_synthetic_pilot.jsonl")
MIXED_PATH = Path("data/lora/train_mixed_pilot_sft.jsonl")
OUTPUT_PATH = Path("results/synthetic_dataset_audit.json")

RANDOM_SEED = 42
SAMPLE_SIZE = 20


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


def get_assistant_query(item: dict) -> str:
    for message in reversed(item.get("messages", [])):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()

    return ""


def get_research_question(item: dict) -> str:
    metadata = item.get("metadata", {})

    question = metadata.get("research_question")
    if question:
        return str(question).strip()

    return "UNKNOWN_RESEARCH_QUESTION"


def normalize_query(query: str) -> str:
    return " ".join(query.lower().split())


def main() -> None:
    synthetic_items = load_jsonl(SYNTHETIC_PATH)
    mixed_items = load_jsonl(MIXED_PATH)

    synthetic_queries = [
        get_assistant_query(item)
        for item in synthetic_items
    ]

    token_lengths = [
        len(query.split())
        for query in synthetic_queries
        if query
    ]

    question_counts = Counter(
        get_research_question(item)
        for item in synthetic_items
    )

    normalized_counts = Counter(
        normalize_query(query)
        for query in synthetic_queries
        if query
    )

    duplicate_queries = {
        query: count
        for query, count in normalized_counts.items()
        if count > 1
    }

    real_items = [
        item
        for item in mixed_items
        if item.get("metadata", {}).get("source") != "synthetic"
    ]

    real_queries = {
        normalize_query(get_assistant_query(item))
        for item in real_items
        if get_assistant_query(item)
    }

    synthetic_query_set = {
        normalize_query(query)
        for query in synthetic_queries
        if query
    }

    exact_real_overlaps = sorted(
        synthetic_query_set.intersection(real_queries)
    )

    invalid_length_queries = [
        query
        for query in synthetic_queries
        if query and not 5 <= len(query.split()) <= 20
    ]

    synthetic_in_mixed = sum(
        item.get("metadata", {}).get("source") == "synthetic"
        for item in mixed_items
    )

    print("=" * 72)
    print("TASK 4 SYNTHETIC DATASET AUDIT")
    print("=" * 72)

    print(f"Synthetic examples: {len(synthetic_items)}")
    print(f"Mixed train examples: {len(mixed_items)}")
    print(f"Real examples in mixed: {len(real_items)}")
    print(f"Synthetic examples in mixed: {synthetic_in_mixed}")

    synthetic_share = (
        synthetic_in_mixed / len(mixed_items)
        if mixed_items
        else 0.0
    )

    print(f"Synthetic share: {synthetic_share:.2%}")

    if token_lengths:
        print()
        print("Query token lengths:")
        print(f"- minimum: {min(token_lengths)}")
        print(f"- mean: {statistics.mean(token_lengths):.2f}")
        print(f"- median: {statistics.median(token_lengths):.2f}")
        print(f"- maximum: {max(token_lengths)}")

    print()
    print(f"Exact duplicate synthetic queries: {len(duplicate_queries)}")
    print(f"Exact overlaps with real targets: {len(exact_real_overlaps)}")
    print(f"Queries outside 5-20 tokens: {len(invalid_length_queries)}")

    print()
    print("Distribution by research question:")

    for question, count in question_counts.most_common():
        short_question = question.replace("\n", " ")[:100]
        print(f"- {count:3d} | {short_question}")

    random.seed(RANDOM_SEED)
    sample = random.sample(
        synthetic_items,
        min(SAMPLE_SIZE, len(synthetic_items)),
    )

    print()
    print("=" * 72)
    print(f"RANDOM SAMPLE: {len(sample)} EXAMPLES")
    print("=" * 72)

    sample_output = []

    for index, item in enumerate(sample, start=1):
        question = get_research_question(item)
        query = get_assistant_query(item)

        print()
        print(f"{index}. QUESTION: {question[:140]}")
        print(f"   QUERY: {query}")

        sample_output.append(
            {
                "research_question": question,
                "query": query,
                "token_length": len(query.split()),
                "parent_query_id": item.get(
                    "metadata", {}
                ).get("parent_query_id"),
            }
        )

    audit = {
        "synthetic_examples": len(synthetic_items),
        "mixed_train_examples": len(mixed_items),
        "real_examples_in_mixed": len(real_items),
        "synthetic_examples_in_mixed": synthetic_in_mixed,
        "synthetic_share": synthetic_share,
        "query_token_lengths": {
            "minimum": min(token_lengths) if token_lengths else None,
            "mean": (
                statistics.mean(token_lengths)
                if token_lengths
                else None
            ),
            "median": (
                statistics.median(token_lengths)
                if token_lengths
                else None
            ),
            "maximum": max(token_lengths) if token_lengths else None,
        },
        "exact_duplicate_synthetic_queries": duplicate_queries,
        "exact_overlaps_with_real_targets": exact_real_overlaps,
        "queries_outside_length_limit": invalid_length_queries,
        "distribution_by_research_question": dict(question_counts),
        "random_sample": sample_output,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            audit,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(f"Saved audit: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()