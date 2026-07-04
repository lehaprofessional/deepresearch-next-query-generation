import json
import re
from pathlib import Path
from statistics import mean
from collections import defaultdict


RUNS = {
    "llm_v1": Path("runs/llm_baseline_predictions.jsonl"),
    "llm_v2": Path("runs/llm_prompt_v2_predictions.jsonl"),
    "llm_v3_fewshot": Path("runs/llm_prompt_v3_fewshot_predictions.jsonl"),
}


def load_jsonl(path: Path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9%°\-]+", text.lower())


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


def main():
    all_runs = {name: load_jsonl(path) for name, path in RUNS.items()}

    # Берём v1 как основной baseline для подробного анализа
    v1_items = all_runs["llm_v1"]
    v3_items = all_runs["llm_v3_fewshot"]

    rows = []

    for i, item in enumerate(v1_items):
        target = item["target_query"]
        pred_v1 = item["generated_query"]
        pred_v3 = v3_items[i]["generated_query"]

        f1_v1 = token_f1(pred_v1, target)
        f1_v3 = token_f1(pred_v3, target)

        rows.append({
            "idx": i,
            "research_question": item["research_question"],
            "query_id": item["query_id"],
            "target": target,
            "v1": pred_v1,
            "v3": pred_v3,
            "f1_v1": f1_v1,
            "f1_v3": f1_v3,
            "delta_v3_minus_v1": f1_v3 - f1_v1,
        })

    print("Worst examples for llm_v1")
    print("=" * 100)

    worst = sorted(rows, key=lambda x: x["f1_v1"])[:10]

    for row in worst:
        print("-" * 100)
        print("idx:", row["idx"], "query_id:", row["query_id"], "f1_v1:", round(row["f1_v1"], 4))
        print("QUESTION:", row["research_question"][:220])
        print("TARGET:  ", row["target"])
        print("V1:      ", row["v1"])
        print("V3:      ", row["v3"])

    print("\nWhere v3 improved most over v1")
    print("=" * 100)

    improved = sorted(rows, key=lambda x: x["delta_v3_minus_v1"], reverse=True)[:10]

    for row in improved:
        print("-" * 100)
        print(
            "idx:", row["idx"],
            "query_id:", row["query_id"],
            "v1:", round(row["f1_v1"], 4),
            "v3:", round(row["f1_v3"], 4),
            "delta:", round(row["delta_v3_minus_v1"], 4),
        )
        print("TARGET:", row["target"])
        print("V1:    ", row["v1"])
        print("V3:    ", row["v3"])

    print("\nAverage F1 by research question for llm_v1")
    print("=" * 100)

    by_question = defaultdict(list)

    for row in rows:
        by_question[row["research_question"]].append(row["f1_v1"])

    for question, scores in by_question.items():
        print("-" * 100)
        print("F1:", round(mean(scores), 4), "| examples:", len(scores))
        print(question[:220])


if __name__ == "__main__":
    main()