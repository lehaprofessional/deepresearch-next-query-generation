import json
import re
from pathlib import Path
from statistics import mean


RUNS = {
    "simple_baseline": Path("runs/simple_baseline_predictions.jsonl"),
    "llm_v1": Path("runs/llm_baseline_predictions.jsonl"),
    "llm_v2": Path("runs/llm_prompt_v2_predictions.jsonl"),
    "llm_v3_fewshot": Path("runs/llm_prompt_v3_fewshot_predictions.jsonl"),
    "llm_v4_retrieval_fewshot": Path("runs/llm_prompt_v4_retrieval_fewshot_predictions.jsonl"),
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
    print("Run comparison")
    print("-" * 100)
    print(f"{'run':<22} {'examples':>8} {'exact':>10} {'f1':>10} {'jaccard':>10} {'repeat':>10} {'gen_len':>10}")
    print("-" * 100)

    for name, path in RUNS.items():
        if not path.exists():
            print(f"{name:<22} FILE NOT FOUND: {path}")
            continue

        metrics = evaluate_run(path)

        print(
            f"{name:<22} "
            f"{metrics['examples']:>8} "
            f"{metrics['exact_match']:>10.4f} "
            f"{metrics['token_f1']:>10.4f} "
            f"{metrics['jaccard']:>10.4f} "
            f"{metrics['repeated_prev_query']:>10.4f} "
            f"{metrics['avg_generated_length']:>10.2f}"
        )


if __name__ == "__main__":
    main()