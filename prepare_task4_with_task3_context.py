import ast
import json
from pathlib import Path

import pandas as pd


TASK3_PATH = Path("train_summerschool_task3.csv")
TASK4_PATH = Path("train_summerschool_task4.csv")

OUTPUT_PATH = Path("prepared_task4_with_context.jsonl")


def safe_parse_list(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []


def build_context_text(context_sources, max_sources=8, max_description_chars=700):
    parts = []

    for source in context_sources[:max_sources]:
        source_id = source["id"]
        link = source["link"]
        description = source["description"]

        if not isinstance(description, str):
            description = ""

        description = description.strip()
        if len(description) > max_description_chars:
            description = description[:max_description_chars].rstrip() + "..."

        parts.append(
            f"[Source id: {source_id}]\n"
            f"URL: {link}\n"
            f"Description: {description}"
        )

    return "\n\n".join(parts)


def main():
    if not TASK3_PATH.exists():
        raise FileNotFoundError(f"File not found: {TASK3_PATH}")

    if not TASK4_PATH.exists():
        raise FileNotFoundError(f"File not found: {TASK4_PATH}")

    task3 = pd.read_csv(TASK3_PATH, sep="\t")
    task4 = pd.read_csv(TASK4_PATH, sep="\t")

    if "Unnamed: 0" in task3.columns:
        task3 = task3.drop(columns=["Unnamed: 0"])

    if "Unnamed: 0" in task4.columns:
        task4 = task4.drop(columns=["Unnamed: 0"])

    print("Task3 shape:", task3.shape)
    print("Task3 columns:", list(task3.columns))
    print("Task4 shape:", task4.shape)
    print("Task4 columns:", list(task4.columns))

    required_task3_columns = {"id", "research_question", "link", "description"}
    required_task4_columns = {"task3_ids", "query", "query_ids", "research_questions"}

    missing_task3 = required_task3_columns - set(task3.columns)
    missing_task4 = required_task4_columns - set(task4.columns)

    if missing_task3:
        raise ValueError(f"Missing task3 columns: {missing_task3}")

    if missing_task4:
        raise ValueError(f"Missing task4 columns: {missing_task4}")

    task3_by_id = task3.set_index("id").to_dict(orient="index")

    prepared_items = []
    total_missing_ids = 0

    for research_question, group in task4.groupby("research_questions", sort=False):
        group = group.sort_values("query_ids")

        previous_queries = []

        for _, row in group.iterrows():
            task3_ids = safe_parse_list(row["task3_ids"])

            context_sources = []
            missing_ids = []

            for source_id in task3_ids:
                if source_id in task3_by_id:
                    source = task3_by_id[source_id]
                    context_sources.append(
                        {
                            "id": int(source_id),
                            "link": source.get("link", ""),
                            "description": source.get("description", ""),
                        }
                    )
                else:
                    missing_ids.append(source_id)

            total_missing_ids += len(missing_ids)

            item = {
                "research_question": research_question,
                "query_id": int(row["query_ids"]),
                "task3_ids": task3_ids,
                "context_sources": context_sources,
                "visited_context": build_context_text(context_sources),
                "previous_queries": previous_queries.copy(),
                "target_query": row["query"],
                "missing_task3_ids": missing_ids,
            }

            prepared_items.append(item)
            previous_queries.append(row["query"])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for item in prepared_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print()
    print("Saved:", OUTPUT_PATH)
    print("Prepared examples:", len(prepared_items))
    print("Unique research questions:", len(set(item["research_question"] for item in prepared_items)))
    print("Total missing task3 ids:", total_missing_ids)

    print()
    print("Example:")
    example = prepared_items[0]
    print("Research question:", example["research_question"][:300])
    print("Query id:", example["query_id"])
    print("Task3 ids:", example["task3_ids"])
    print("Context sources:", len(example["context_sources"]))
    print("Previous queries:", example["previous_queries"])
    print("Target query:", example["target_query"])
    print()
    print("Visited context preview:")
    print(example["visited_context"][:1200])


if __name__ == "__main__":
    main()