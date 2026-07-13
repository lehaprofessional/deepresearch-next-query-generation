import csv
import json
import os
from pathlib import Path
from statistics import mean

import torch
from bert_score import score


RUNS = {
    "simple_baseline": Path("runs/simple_baseline_predictions.jsonl"),
    "llm_v1": Path("runs/llm_baseline_predictions.jsonl"),
    "llm_v2": Path("runs/llm_prompt_v2_predictions.jsonl"),
    "llm_v3_fewshot": Path("runs/llm_prompt_v3_fewshot_predictions.jsonl"),
    "llm_v4_retrieval_fewshot": Path("runs/llm_prompt_v4_retrieval_fewshot_predictions.jsonl"),
    "llm_context_v1": Path("runs/llm_context_v1_predictions.jsonl"),
    "llm_context_retrieval_fewshot_v2": Path("runs/llm_context_retrieval_fewshot_v2_predictions.jsonl"),
    "llm_context_prompt_v2": Path("runs/llm_context_prompt_v2_predictions.jsonl"),
    "llm_context_hybrid_router": Path("runs/llm_context_hybrid_router_predictions.jsonl"),
    "qwen2.5_1.5b_base": Path(
    "runs/qwen2.5_1.5b_base_predictions.jsonl"
),
"qwen2.5_1.5b_lora_real": Path(
    "runs/qwen2.5_1.5b_lora_real_predictions.jsonl"
),
}

OUTPUT_DIR = Path("results")
SUMMARY_CSV_PATH = OUTPUT_DIR / "bertscore_comparison.csv"
SUMMARY_MD_PATH = OUTPUT_DIR / "bertscore_comparison.md"

# Можно поменять через переменную окружения.
# roberta-large качественнее, но тяжелее.
# distilroberta-base быстрее и легче.
MODEL_TYPE = os.getenv("BERTSCORE_MODEL", "roberta-large")

DEVICE = os.getenv(
    "BERTSCORE_DEVICE",
    "cuda" if torch.cuda.is_available() else "cpu"
)


def load_jsonl(path: Path):
    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    return items


def save_per_example_csv(run_name, items, precision, recall, f1):
    path = OUTPUT_DIR / f"bertscore_{run_name}.csv"

    fieldnames = [
        "research_question",
        "query_id",
        "target_query",
        "generated_query",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item, p, r, f1_value in zip(items, precision, recall, f1):
            writer.writerow(
                {
                    "research_question": item.get("research_question", ""),
                    "query_id": item.get("query_id", ""),
                    "target_query": item.get("target_query", ""),
                    "generated_query": item.get("generated_query", ""),
                    "bertscore_precision": p,
                    "bertscore_recall": r,
                    "bertscore_f1": f1_value,
                }
            )

    return path


def evaluate_run(run_name, path: Path):
    items = load_jsonl(path)

    candidates = [
        item["generated_query"]
        for item in items
    ]

    references = [
        item["target_query"]
        for item in items
    ]

    print()
    print("=" * 80)
    print(f"Run: {run_name}")
    print(f"Examples: {len(items)}")
    print(f"Model: {MODEL_TYPE}")
    print(f"Device: {DEVICE}")
    print("=" * 80)

    precision, recall, f1 = score(
        candidates,
        references,
        lang="en",
        model_type=MODEL_TYPE,
        device=DEVICE,
        verbose=True,
        rescale_with_baseline=False,
    )

    precision_values = [float(x) for x in precision.tolist()]
    recall_values = [float(x) for x in recall.tolist()]
    f1_values = [float(x) for x in f1.tolist()]

    per_example_path = save_per_example_csv(
        run_name,
        items,
        precision_values,
        recall_values,
        f1_values,
    )

    result = {
        "run": run_name,
        "examples": len(items),
        "bertscore_precision": mean(precision_values),
        "bertscore_recall": mean(recall_values),
        "bertscore_f1": mean(f1_values),
        "per_example_path": str(per_example_path),
    }

    print()
    print(f"Avg BERTScore Precision: {result['bertscore_precision']:.4f}")
    print(f"Avg BERTScore Recall:    {result['bertscore_recall']:.4f}")
    print(f"Avg BERTScore F1:        {result['bertscore_f1']:.4f}")
    print(f"Saved per-example CSV:   {per_example_path}")

    return result


def save_summary(results):
    OUTPUT_DIR.mkdir(exist_ok=True)

    fieldnames = [
        "run",
        "examples",
        "bertscore_precision",
        "bertscore_recall",
        "bertscore_f1",
        "per_example_path",
    ]

    with open(SUMMARY_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    with open(SUMMARY_MD_PATH, "w", encoding="utf-8") as f:
        f.write("| Method | Examples | BERTScore Precision | BERTScore Recall | BERTScore F1 |\n")
        f.write("|---|---:|---:|---:|---:|\n")

        for row in results:
            f.write(
                f"| {row['run']} "
                f"| {row['examples']} "
                f"| {row['bertscore_precision']:.4f} "
                f"| {row['bertscore_recall']:.4f} "
                f"| {row['bertscore_f1']:.4f} |\n"
            )

    print()
    print("Saved summary CSV:", SUMMARY_CSV_PATH)
    print("Saved summary Markdown:", SUMMARY_MD_PATH)


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    results = []

    for run_name, path in RUNS.items():
        if not path.exists():
            print(f"Skip {run_name}: file not found {path}")
            continue

        result = evaluate_run(run_name, path)
        results.append(result)

    save_summary(results)

    print()
    print("Final BERTScore comparison:")
    print("-" * 80)

    for row in results:
        print(
            f"{row['run']:<38} "
            f"P={row['bertscore_precision']:.4f} "
            f"R={row['bertscore_recall']:.4f} "
            f"F1={row['bertscore_f1']:.4f}"
        )


if __name__ == "__main__":
    main()