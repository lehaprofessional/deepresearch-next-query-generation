import json
from collections import defaultdict
from pathlib import Path


TRAIN_PATH = Path("data/task4_train.jsonl")
VAL_PATH = Path("data/task4_val.jsonl")
OUTPUT_PATH = Path("data/task4_val_v3_fewshot.jsonl")


def load_jsonl(path: Path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_jsonl(items, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def format_previous_queries(previous_queries: list[str]) -> str:
    if not previous_queries:
        return "No previous queries."

    return "\n".join(
        f"{i + 1}. {query}" for i, query in enumerate(previous_queries)
    )


def select_fewshot_examples(train_items: list[dict], max_examples: int = 6) -> list[dict]:
    """
    Берём few-shot примеры из разных research questions,
    чтобы не показывать модели только одну тему.
    """
    by_question = defaultdict(list)

    for item in train_items:
        by_question[item["research_question"]].append(item)

    examples = []

    # Сначала берём query_id=1 из разных вопросов
    for question, items in by_question.items():
        items = sorted(items, key=lambda x: x["query_id"])
        first = items[0]
        examples.append(first)

        if len(examples) >= max_examples:
            return examples

    return examples[:max_examples]


def build_fewshot_block(examples: list[dict]) -> str:
    blocks = []

    for i, ex in enumerate(examples, start=1):
        block = f"""Example {i}

Research question:
{ex["research_question"]}

Previous queries:
{format_previous_queries(ex["previous_queries"])}

Correct next search query:
{ex["target_query"]}"""
        blocks.append(block)

    return "\n\n".join(blocks)


def build_prompt(item: dict, fewshot_block: str) -> str:
    return f"""You are a search-query generator for a DeepResearch agent.

You will receive:
1. A complex research question.
2. Previous search queries already generated for this question.

Your task:
Generate the next search engine query in the same style as the examples.

The query should:
- be a compact search query, not a full sentence;
- use specific technical terms;
- explore a new aspect not already covered by previous queries;
- be suitable for Google/DuckDuckGo-style search;
- usually contain 6 to 14 words;
- return only one query.

Examples of the desired style:

{fewshot_block}

Now generate the next query.

Research question:
{item["research_question"]}

Current query step:
{item["query_id"]}

Previous queries:
{format_previous_queries(item["previous_queries"])}

Next search query:"""


def main():
    train_items = load_jsonl(TRAIN_PATH)
    val_items = load_jsonl(VAL_PATH)

    fewshot_examples = select_fewshot_examples(train_items, max_examples=6)
    fewshot_block = build_fewshot_block(fewshot_examples)

    output_items = []

    for item in val_items:
        new_item = dict(item)
        new_item["prompt"] = build_prompt(item, fewshot_block)
        output_items.append(new_item)

    save_jsonl(output_items, OUTPUT_PATH)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Validation prompts: {len(output_items)}")
    print(f"Few-shot examples: {len(fewshot_examples)}")

    print("\nFew-shot target queries:")
    for ex in fewshot_examples:
        print("-", ex["target_query"])

    print("\nExample validation prompt:")
    print(output_items[0]["prompt"][:4000])

    print("\nTarget query:")
    print(output_items[0]["target_query"])


if __name__ == "__main__":
    main()