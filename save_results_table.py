import json
import re
import csv
from pathlib import Path
from statistics import mean


RUNS = {
    "simple_baseline": Path("runs/simple_baseline_predictions.jsonl"),
    "llm_v1": Path("runs/llm_baseline_predictions.jsonl"),
    "llm_v2": Path("runs/llm_prompt_v2_predictions.jsonl"),
    "llm_v3_fewshot": Path("runs/llm_prompt_v3_fewshot_predictions.jsonl"),
    "llm_v4_retrieval_fewshot": Path("runs/llm_prompt_v4_retrieval_fewshot_predictions.jsonl"),
}

OUTPUT_DIR = Path("results")
CSV_PATH = OUTPUT_DIR / "comparison_table.csv"
MD_PATH = OUTPUT_DIR / "comparison_table.md"


def load_jsonl(path: Path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9%°\\-]+", text.lower())


def token_f1(prediction: str, target: str) -> float:
    pred_tokens = tokenize(prediction)
    target_tokens = tokenize(target)

    if not pred_tokens or not target_tokens:
        return 0.0

    pred_counts = {}
    target_counts = {}

    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1

    for token in target_tokens:
        target_counts[token] = target_counts.get(token, 0) + 1

    overlap = 0
    for token in pred_counts:
        overlap += min(pred_counts.get(token, 0), target_counts.get(token, 0))

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(target_tokens)

    return 2 * precision * recall / (precision + recall)


def jaccard(prediction: str, target: str) -> float:
    pred_set = set(tokenize(prediction))
    target_set = set(tokenize(target))

    if not pred_set or not target_set:
        return 0.0

    return len(pred_set & target_set) / len(pred_set | target_set)


def exact_match(prediction: str, target: str) -> float:
    return float(prediction.strip().lower() == target.strip().lower())


def repeats_previous_query(prediction: str, previous_queries: list[str]) -> float:
    pred_norm = prediction.strip().lower()

    for query in previous_queries:
        if pred_norm == query.strip().lower():
            return 1.0

    return 0.0


def evaluate_run(path: Path):
    items = load_jsonl(path)

    exact_scores = []
    f1_scores = []
    jaccard_scores = []
    repeat_scores = []
    pred_lengths = []
    target_lengths = []

    for item in items:
        pred = item["generated_query"]
        target = item["target_query"]

        exact_scores.append(exact_match(pred, target))
        f1_scores.append(token_f1(pred, target))
        jaccard_scores.append(jaccard(pred, target))
        repeat_scores.append(repeats_previous_query(pred, item.get("previous_queries", [])))
        pred_lengths.append(len(tokenize(pred)))
        target_lengths.append(len(tokenize(target)))

    return {
        "examples": len(items),
        "exact_match": mean(exact_scores),
        "token_f1": mean(f1_scores),
        "jaccard": mean(jaccard_scores),
        "repeated_prev_query": mean(repeat_scores),
        "avg_generated_length": mean(pred_lengths),
        "avg_target_length": mean(target_lengths),
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    rows = []

    for name, path in RUNS.items():
        if not path.exists():
            print(f"Skip {name}: file not found {path}")
            continue

        metrics = evaluate_run(path)
        row = {
            "run": name,
            **metrics,
        }
        rows.append(row)

    fieldnames = [
        "run",
        "examples",
        "exact_match",
        "token_f1",
        "jaccard",
        "repeated_prev_query",
        "avg_generated_length",
        "avg_target_length",
    ]

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("| Method | Examples | Exact Match | Token F1 | Jaccard | Repeated Prev Query | Avg Generated Length | Avg Target Length |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")

        for row in rows:
            f.write(
                f"| {row['run']} "
                f"| {row['examples']} "
                f"| {row['exact_match']:.4f} "
                f"| {row['token_f1']:.4f} "
                f"| {row['jaccard']:.4f} "
                f"| {row['repeated_prev_query']:.4f} "
                f"| {row['avg_generated_length']:.2f} "
                f"| {row['avg_target_length']:.2f} |\n"
            )

    print(f"Saved CSV: {CSV_PATH}")
    print(f"Saved Markdown: {MD_PATH}")


if __name__ == "__main__":
    main()