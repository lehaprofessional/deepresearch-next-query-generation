import json
import re
from pathlib import Path
from statistics import mean


PREDICTIONS_PATH = Path("runs/llm_prompt_v4_retrieval_fewshot_predictions.jsonl")


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


def main():
    items = load_jsonl(PREDICTIONS_PATH)

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

    print("Evaluation results")
    print("-" * 60)
    print("Examples:", len(items))
    print(f"Exact match:          {mean(exact_scores):.4f}")
    print(f"Token F1:             {mean(f1_scores):.4f}")
    print(f"Jaccard:              {mean(jaccard_scores):.4f}")
    print(f"Repeated prev query:  {mean(repeat_scores):.4f}")
    print(f"Avg generated length: {mean(pred_lengths):.2f} tokens")
    print(f"Avg target length:    {mean(target_lengths):.2f} tokens")

    print("\nExamples:")
    for item in items[:5]:
        print("-" * 60)
        print("QUESTION:", item["research_question"][:180])
        print("TARGET:  ", item["target_query"])
        print("PRED:    ", item["generated_query"])


if __name__ == "__main__":
    main()