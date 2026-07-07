import json
import re
import csv
from pathlib import Path
from statistics import mean
from collections import defaultdict


BASELINE_PATH = Path("runs/llm_context_v1_predictions.jsonl")
BEST_PATH = Path("runs/llm_context_prompt_v2_predictions.jsonl")
RETRIEVAL_PATH = Path("runs/llm_context_retrieval_fewshot_v2_predictions.jsonl")

OUTPUT_DIR = Path("results")
CSV_PATH = OUTPUT_DIR / "context_v2_error_analysis.csv"
MD_PATH = OUTPUT_DIR / "context_v2_error_analysis.md"


def load_jsonl(path: Path):
    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    return items


def tokenize(text: str):
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


def short(text: str, max_len=180):
    text = str(text).replace("\n", " ").strip()

    if len(text) <= max_len:
        return text

    return text[:max_len].rstrip() + "..."


def build_rows():
    baseline_items = load_jsonl(BASELINE_PATH)
    best_items = load_jsonl(BEST_PATH)
    retrieval_items = load_jsonl(RETRIEVAL_PATH)

    if not (len(baseline_items) == len(best_items) == len(retrieval_items)):
        raise ValueError("Prediction files have different number of examples.")

    rows = []

    for i, (base, best, retrieval) in enumerate(
        zip(baseline_items, best_items, retrieval_items)
    ):
        target = best["target_query"]

        base_pred = base["generated_query"]
        best_pred = best["generated_query"]
        retrieval_pred = retrieval["generated_query"]

        base_f1 = token_f1(base_pred, target)
        best_f1 = token_f1(best_pred, target)
        retrieval_f1 = token_f1(retrieval_pred, target)

        row = {
            "index": i,
            "research_question": best["research_question"],
            "query_id": best["query_id"],
            "target_query": target,

            "context_v1_prediction": base_pred,
            "context_v1_f1": base_f1,
            "context_v1_jaccard": jaccard(base_pred, target),

            "context_prompt_v2_prediction": best_pred,
            "context_prompt_v2_f1": best_f1,
            "context_prompt_v2_jaccard": jaccard(best_pred, target),

            "context_retrieval_fewshot_v2_prediction": retrieval_pred,
            "context_retrieval_fewshot_v2_f1": retrieval_f1,
            "context_retrieval_fewshot_v2_jaccard": jaccard(retrieval_pred, target),

            "delta_v2_minus_v1": best_f1 - base_f1,
            "delta_v2_minus_retrieval": best_f1 - retrieval_f1,
        }

        rows.append(row)

    return rows


def save_csv(rows):
    OUTPUT_DIR.mkdir(exist_ok=True)

    fieldnames = [
        "index",
        "research_question",
        "query_id",
        "target_query",
        "context_v1_prediction",
        "context_v1_f1",
        "context_v1_jaccard",
        "context_prompt_v2_prediction",
        "context_prompt_v2_f1",
        "context_prompt_v2_jaccard",
        "context_retrieval_fewshot_v2_prediction",
        "context_retrieval_fewshot_v2_f1",
        "context_retrieval_fewshot_v2_jaccard",
        "delta_v2_minus_v1",
        "delta_v2_minus_retrieval",
    ]

    with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_worst_examples(rows, n=8):
    print()
    print("=" * 80)
    print(f"Worst examples for context_prompt_v2 by Token F1, top {n}")
    print("=" * 80)

    worst = sorted(rows, key=lambda row: row["context_prompt_v2_f1"])[:n]

    for row in worst:
        print()
        print(f"Index: {row['index']} | query_id={row['query_id']} | F1={row['context_prompt_v2_f1']:.4f}")
        print("QUESTION:", short(row["research_question"], 220))
        print("TARGET:  ", row["target_query"])
        print("PRED:    ", row["context_prompt_v2_prediction"])


def print_best_improvements(rows, n=8):
    print()
    print("=" * 80)
    print(f"Where context_prompt_v2 improved over context_v1, top {n}")
    print("=" * 80)

    improved = sorted(rows, key=lambda row: row["delta_v2_minus_v1"], reverse=True)[:n]

    for row in improved:
        print()
        print(
            f"Index: {row['index']} | query_id={row['query_id']} | "
            f"v1={row['context_v1_f1']:.4f} → v2={row['context_prompt_v2_f1']:.4f} | "
            f"delta={row['delta_v2_minus_v1']:.4f}"
        )
        print("TARGET:  ", row["target_query"])
        print("V1 PRED: ", row["context_v1_prediction"])
        print("V2 PRED: ", row["context_prompt_v2_prediction"])


def print_biggest_regressions(rows, n=8):
    print()
    print("=" * 80)
    print(f"Where context_prompt_v2 became worse than context_v1, top {n}")
    print("=" * 80)

    regressed = sorted(rows, key=lambda row: row["delta_v2_minus_v1"])[:n]

    for row in regressed:
        print()
        print(
            f"Index: {row['index']} | query_id={row['query_id']} | "
            f"v1={row['context_v1_f1']:.4f} → v2={row['context_prompt_v2_f1']:.4f} | "
            f"delta={row['delta_v2_minus_v1']:.4f}"
        )
        print("TARGET:  ", row["target_query"])
        print("V1 PRED: ", row["context_v1_prediction"])
        print("V2 PRED: ", row["context_prompt_v2_prediction"])


def print_by_question(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["research_question"]].append(row)

    print()
    print("=" * 80)
    print("Average Token F1 by research question")
    print("=" * 80)

    for question, items in grouped.items():
        v1_avg = mean(row["context_v1_f1"] for row in items)
        v2_avg = mean(row["context_prompt_v2_f1"] for row in items)
        retrieval_avg = mean(row["context_retrieval_fewshot_v2_f1"] for row in items)

        print()
        print("QUESTION:", short(question, 220))
        print(f"Examples: {len(items)}")
        print(f"context_v1:                    {v1_avg:.4f}")
        print(f"context_prompt_v2:             {v2_avg:.4f}")
        print(f"context_retrieval_fewshot_v2:  {retrieval_avg:.4f}")


def save_markdown_report(rows):
    grouped = defaultdict(list)

    for row in rows:
        grouped[row["research_question"]].append(row)

    worst = sorted(rows, key=lambda row: row["context_prompt_v2_f1"])[:8]
    improved = sorted(rows, key=lambda row: row["delta_v2_minus_v1"], reverse=True)[:8]
    regressed = sorted(rows, key=lambda row: row["delta_v2_minus_v1"])[:8]

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("# Error analysis for context prompt v2\n\n")

        f.write("## Summary\n\n")
        f.write("This report compares three variants:\n\n")
        f.write("- `llm_context_v1`\n")
        f.write("- `llm_context_retrieval_fewshot_v2`\n")
        f.write("- `llm_context_prompt_v2`\n\n")

        f.write("The goal is to understand where `context_prompt_v2` improves over the previous context baseline and where it still fails.\n\n")

        f.write("## Average Token F1 by research question\n\n")
        f.write("| Research question | Examples | context_v1 | context_retrieval_fewshot_v2 | context_prompt_v2 |\n")
        f.write("|---|---:|---:|---:|---:|\n")

        for question, items in grouped.items():
            v1_avg = mean(row["context_v1_f1"] for row in items)
            v2_avg = mean(row["context_prompt_v2_f1"] for row in items)
            retrieval_avg = mean(row["context_retrieval_fewshot_v2_f1"] for row in items)

            f.write(
                f"| {short(question, 120)} "
                f"| {len(items)} "
                f"| {v1_avg:.4f} "
                f"| {retrieval_avg:.4f} "
                f"| {v2_avg:.4f} |\n"
            )

        f.write("\n## Worst examples for context_prompt_v2\n\n")
        f.write("| query_id | F1 | Target query | Generated query |\n")
        f.write("|---:|---:|---|---|\n")

        for row in worst:
            f.write(
                f"| {row['query_id']} "
                f"| {row['context_prompt_v2_f1']:.4f} "
                f"| {short(row['target_query'], 120)} "
                f"| {short(row['context_prompt_v2_prediction'], 120)} |\n"
            )

        f.write("\n## Best improvements over context_v1\n\n")
        f.write("| query_id | context_v1 F1 | context_prompt_v2 F1 | Delta | Target query | v2 generated query |\n")
        f.write("|---:|---:|---:|---:|---|---|\n")

        for row in improved:
            f.write(
                f"| {row['query_id']} "
                f"| {row['context_v1_f1']:.4f} "
                f"| {row['context_prompt_v2_f1']:.4f} "
                f"| {row['delta_v2_minus_v1']:.4f} "
                f"| {short(row['target_query'], 120)} "
                f"| {short(row['context_prompt_v2_prediction'], 120)} |\n"
            )

        f.write("\n## Biggest regressions compared with context_v1\n\n")
        f.write("| query_id | context_v1 F1 | context_prompt_v2 F1 | Delta | Target query | v2 generated query |\n")
        f.write("|---:|---:|---:|---:|---|---|\n")

        for row in regressed:
            f.write(
                f"| {row['query_id']} "
                f"| {row['context_v1_f1']:.4f} "
                f"| {row['context_prompt_v2_f1']:.4f} "
                f"| {row['delta_v2_minus_v1']:.4f} "
                f"| {short(row['target_query'], 120)} "
                f"| {short(row['context_prompt_v2_prediction'], 120)} |\n"
            )


def main():
    rows = build_rows()

    save_csv(rows)
    save_markdown_report(rows)

    print_worst_examples(rows)
    print_best_improvements(rows)
    print_biggest_regressions(rows)
    print_by_question(rows)

    print()
    print("Saved CSV:", CSV_PATH)
    print("Saved Markdown:", MD_PATH)


if __name__ == "__main__":
    main()