from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {error}"
                ) from error
            if not isinstance(value, dict):
                raise ValueError(
                    f"Expected JSON object in {path}, line {line_number}"
                )
            rows.append(value)
    return rows


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build curated Kimi synthetic SFT examples."
    )
    parser.add_argument(
        "--source-sft",
        type=Path,
        default=Path("data/lora/train_real_sft.jsonl"),
    )
    parser.add_argument(
        "--map-file",
        type=Path,
        default=Path("synthetic_kimi_v3_ready/pilot_map_v3.jsonl"),
    )
    parser.add_argument(
        "--curated-queries",
        type=Path,
        default=Path("training_ready_semantic_dedup_20.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/lora/train_kimi_v3_curated_20_sft.jsonl"
        ),
    )
    args = parser.parse_args()

    source_rows = load_jsonl(args.source_sft)
    map_rows = load_jsonl(args.map_file)
    curated_rows = load_jsonl(args.curated_queries)

    map_by_id = {
        row["trajectory_id"]: row
        for row in map_rows
    }

    synthetic_rows: list[dict[str, Any]] = []
    missing_ids: list[str] = []

    for curated in curated_rows:
        trajectory_id = curated["trajectory_id"]
        map_row = map_by_id.get(trajectory_id)
        if map_row is None:
            missing_ids.append(trajectory_id)
            continue

        source_index = int(map_row["source_row_index"]) - 1
        if not 0 <= source_index < len(source_rows):
            raise IndexError(
                f"Invalid source index for {trajectory_id}: "
                f"{source_index + 1}"
            )

        item = copy.deepcopy(source_rows[source_index])
        messages = item.get("messages")
        if not isinstance(messages, list):
            raise ValueError(
                f"No messages array for {trajectory_id}"
            )

        assistant_index = None
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "assistant":
                assistant_index = index
                break

        if assistant_index is None:
            raise ValueError(
                f"No assistant target for {trajectory_id}"
            )

        messages[assistant_index]["content"] = curated[
            "synthetic_query"
        ]

        metadata = item.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            item["metadata"] = metadata

        metadata.update({
            "is_synthetic": True,
            "synthetic_source": "Kimi-K2.6",
            "synthetic_version": "3.1-curated",
            "synthetic_trajectory_id": trajectory_id,
            "curation_status": curated["curation_status"],
        })
        synthetic_rows.append(item)

    if missing_ids:
        raise ValueError(
            "Missing trajectory IDs in map file: "
            + ", ".join(missing_ids)
        )

    save_jsonl(args.output, synthetic_rows)

    print("=" * 72)
    print("CURATED KIMI SFT CREATED")
    print("=" * 72)
    print("Source SFT rows:", len(source_rows))
    print("Curated queries:", len(curated_rows))
    print("Output SFT rows:", len(synthetic_rows))
    print("Output:", args.output)


if __name__ == "__main__":
    main()
