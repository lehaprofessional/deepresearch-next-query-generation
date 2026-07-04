import json
import re
from pathlib import Path


INPUT_PATH = Path("data/task4_val.jsonl")
OUTPUT_PATH = Path("runs/simple_baseline_predictions.jsonl")


STOPWORDS = {
    "what", "are", "the", "and", "how", "does", "of", "in", "a", "an", "to",
    "for", "with", "by", "from", "is", "be", "can", "used", "under", "between",
    "which", "when", "where", "why", "this", "that", "their", "into", "on"
}


def load_jsonl(path: Path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def simple_query_from_question(question: str, max_words: int = 12) -> str:
    words = re.findall(r"[A-Za-z0-9%°\-]+", question.lower())

    filtered = []
    for word in words:
        if word in STOPWORDS:
            continue
        if len(word) <= 2:
            continue
        filtered.append(word)

    # Убираем повторы, сохраняя порядок
    unique_words = list(dict.fromkeys(filtered))

    return " ".join(unique_words[:max_words])


items = load_jsonl(INPUT_PATH)

OUTPUT_PATH.parent.mkdir(exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in items:
        generated_query = simple_query_from_question(item["research_question"])

        out = {
            "research_question": item["research_question"],
            "query_id": item["query_id"],
            "prompt": item["prompt"],
            "previous_queries": item["previous_queries"],
            "target_query": item["target_query"],
            "generated_query": generated_query,
        }

        f.write(json.dumps(out, ensure_ascii=False) + "\n")

print(f"Saved: {OUTPUT_PATH}")
print(f"Predictions: {len(items)}")

print("\nExample:")
print("TARGET:   ", items[0]["target_query"])
print("GENERATED:", simple_query_from_question(items[0]["research_question"]))