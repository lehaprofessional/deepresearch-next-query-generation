import json
import re
from pathlib import Path
from collections import defaultdict


TRAIN_PATH = Path("data/task4_train.jsonl")
VAL_PATH = Path("data/task4_val.jsonl")
OUTPUT_PATH = Path("data/task4_val_v4_retrieval_fewshot.jsonl")

MAX_FEWSHOT_EXAMPLES = 6


STOPWORDS = {
    "what", "are", "the", "and", "how", "does", "of", "in", "a", "an", "to",
    "for", "with", "by", "from", "is", "be", "can", "used", "under", "between",
    "which", "when", "where", "why", "this", "that", "their", "into", "on",
    "most", "effective", "specific", "distinct", "using", "when", "through",
    "question", "research"
}


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


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9%°\-]+", text.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def text_for_retrieval(item: dict) -> str:
    previous = " ".join(item.get("previous_queries", []))
    return f"{item['research_question']} {previous} {item.get('target_query', '')}"


def similarity(a: str, b: str) -> float:
    a_set = set(tokenize(a))
    b_set = set(tokenize(b))

    if not a_set or not b_set:
        return 0.0

    return len(a_set & b_set) / len(a_set | b_set)


def format_previous_queries(previous_queries: list[str]) -> str:
    if not previous_queries:
        return "No previous queries."

    return "\n".join(
        f"{i + 1}. {query}" for i, query in enumerate(previous_queries)
    )


def select_relevant_examples(val_item: dict, train_items: list[dict]) -> list[dict]:
    val_text = text_for_retrieval(val_item)

    scored = []

    for train_item in train_items:
        train_text = text_for_retrieval(train_item)

        score = similarity(val_text, train_text)

        # Небольшой бонус, если query_id близкий:
        # стиль первого запроса часто отличается от стиля поздних запросов.
        diff = abs(int(val_item["query_id"]) - int(train_item["query_id"]))
        if diff <= 2:
            score += 0.05

        scored.append((score, train_item))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    per_question_count = defaultdict(int)

    for score, item in scored:
        question = item["research_question"]

        # Не берём слишком много примеров из одного и того же train-вопроса.
        if per_question_count[question] >= 2:
            continue

        selected.append(item)
        per_question_count[question] += 1

        if len(selected) >= MAX_FEWSHOT_EXAMPLES:
            break

    return selected


def build_fewshot_block(examples: list[dict]) -> str:
    blocks = []

    for i, ex in enumerate(examples, start=1):
        block = f"""Example {i}

Research question:
{ex["research_question"]}

Current query step:
{ex["query_id"]}

Previous queries:
{format_previous_queries(ex["previous_queries"])}

Correct next search query:
{ex["target_query"]}"""
        blocks.append(block)

    return "\n\n".join(blocks)


def build_prompt(item: dict, examples: list[dict]) -> str:
    fewshot_block = build_fewshot_block(examples)

    return f"""You are a search-query planner inside a DeepResearch agent.

Your task is to generate the next search engine query.

Use the examples below to learn the desired style:
- compact technical search terms;
- one query only;
- no explanations;
- not a full sentence;
- a new aspect compared with previous queries.

Relevant training examples:

{fewshot_block}

Now solve the actual case.

Research question:
{item["research_question"]}

Current query step:
{item["query_id"]}

Previous queries:
{format_previous_queries(item["previous_queries"])}

Instructions:
- Generate exactly one search query.
- The query must explore a new missing aspect.
- Do not repeat previous queries.
- Avoid broad restatement of the research question.
- Prefer concrete technical terms, process parameters, materials, mechanisms, trials, standards, or numerical conditions.
- Return only the query.

Next search query:"""


def main():
    train_items = load_jsonl(TRAIN_PATH)
    val_items = load_jsonl(VAL_PATH)

    output_items = []

    for item in val_items:
        examples = select_relevant_examples(item, train_items)

        new_item = dict(item)
        new_item["prompt"] = build_prompt(item, examples)
        new_item["fewshot_target_queries"] = [ex["target_query"] for ex in examples]

        output_items.append(new_item)

    save_jsonl(output_items, OUTPUT_PATH)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Validation prompts: {len(output_items)}")
    print(f"Few-shot examples per prompt: {MAX_FEWSHOT_EXAMPLES}")

    print("\nFirst prompt few-shot target queries:")
    for q in output_items[0]["fewshot_target_queries"]:
        print("-", q)

    print("\nExample prompt:")
    print(output_items[0]["prompt"][:5000])

    print("\nTarget query:")
    print(output_items[0]["target_query"])


if __name__ == "__main__":
    main()