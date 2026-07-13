from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from bert_score import score


PREDICTION_KEYS = (
    "prediction",
    "generated_query",
    "pred",
    "output",
    "generated",
)

TARGET_KEYS = (
    "target",
    "target_query",
    "reference",
    "gold",
    "expected",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {error}"
                ) from error

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object in {path}, line {line_number}"
                )

            rows.append(row)

    if not rows:
        raise ValueError(f"No examples found in {path}")

    return rows


def find_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate one Task 4 prediction file with BERTScore."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="JSONL prediction file.",
    )
    parser.add_argument(
        "--model-type",
        default="roberta-large",
        help="BERTScore encoder. Keep identical across compared runs.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="BERTScore language code.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON metrics output.",
    )
    args = parser.parse_args()

    rows = load_jsonl(args.predictions)

    predictions: list[str] = []
    targets: list[str] = []

    for index, row in enumerate(rows, start=1):
        prediction = find_text(row, PREDICTION_KEYS)
        target = find_text(row, TARGET_KEYS)

        if not prediction or not target:
            raise ValueError(
                f"Could not find prediction/target in example {index}. "
                f"Available keys: {sorted(row.keys())}"
            )

        predictions.append(prediction)
        targets.append(target)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    precision, recall, f1 = score(
        predictions,
        targets,
        lang=args.lang,
        model_type=args.model_type,
        device=device,
        verbose=True,
        rescale_with_baseline=False,
    )

    metrics = {
        "predictions_file": str(args.predictions.resolve()),
        "examples": len(rows),
        "model_type": args.model_type,
        "language": args.lang,
        "device": device,
        "bertscore_precision": float(precision.mean().item()),
        "bertscore_recall": float(recall.mean().item()),
        "bertscore_f1": float(f1.mean().item()),
    }

    print()
    print("=" * 60)
    print("BERTSCORE RESULTS")
    print("=" * 60)
    print(f"Predictions: {args.predictions.resolve()}")
    print(f"Examples:    {len(rows)}")
    print(f"Model:       {args.model_type}")
    print(f"Device:      {device}")
    print(f"Precision:   {metrics['bertscore_precision']:.4f}")
    print(f"Recall:      {metrics['bertscore_recall']:.4f}")
    print(f"F1:          {metrics['bertscore_f1']:.4f}")

    output_path = args.output
    if output_path is None:
        output_path = Path("results") / (
            args.predictions.stem + "_bertscore.json"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    print(f"Saved:       {output_path}")


if __name__ == "__main__":
    main()
