import ast
import json
from pathlib import Path

import pandas as pd


INPUT_PATH = "train_summerschool_task4.csv"
OUTPUT_PATH = "prepared_task4_query_generation.jsonl"


def safe_parse_list(value: str):
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


df = pd.read_csv(INPUT_PATH, sep="\t")

# Убираем лишнюю колонку, если она есть
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

print("Размер task4:", df.shape)
print("Колонки:", list(df.columns))
print()

# Нормализуем типы
df["research_questions"] = df["research_questions"].astype(str)
df["query"] = df["query"].astype(str)
df["query_ids"] = df["query_ids"].astype(int)
df["task3_ids_parsed"] = df["task3_ids"].astype(str).apply(safe_parse_list)

print("Количество уникальных research_questions:", df["research_questions"].nunique())
print("Количество строк:", len(df))
print("query_ids:")
print(df["query_ids"].value_counts().sort_index())
print()

prepared = []

# Группируем по исследовательскому вопросу
for question, group in df.groupby("research_questions"):
    group = group.sort_values("query_ids")

    previous_queries = []

    for _, row in group.iterrows():
        item = {
            "research_question": question,
            "query_id": int(row["query_ids"]),
            "task3_ids": row["task3_ids_parsed"],
            "previous_queries": previous_queries.copy(),
            "target_query": row["query"],
        }

        prepared.append(item)

        previous_queries.append(row["query"])

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in prepared:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Сохранено: {OUTPUT_PATH}")
print("Количество подготовленных примеров:", len(prepared))

print("\nПример:")
print(json.dumps(prepared[0], ensure_ascii=False, indent=2))