import ast
import json
import pandas as pd
from pathlib import Path

INPUT_PATH = "summerschool_task1.csv"
OUTPUT_PATH = "prepared_task1_query_generation.jsonl"

df = pd.read_csv(INPUT_PATH, sep="\t")

print("Размер исходного task1:", df.shape)
print("Колонки:", list(df.columns))

# Группируем все queries по одному research question
groups = []

for question, group in df.groupby("question"):
    queries = group["queries"].dropna().astype(str).tolist()

    # links в CSV лежат строкой вида "['url1', 'url2']"
    all_links = []
    for raw_links in group["links"].dropna().astype(str):
        try:
            parsed_links = ast.literal_eval(raw_links)
            if isinstance(parsed_links, list):
                all_links.extend(parsed_links)
        except Exception:
            pass

    # убираем дубли, сохраняя порядок
    unique_queries = list(dict.fromkeys(queries))
    unique_links = list(dict.fromkeys(all_links))

    groups.append({
        "question": question,
        "target_queries": unique_queries,
        "links": unique_links,
        "num_queries": len(unique_queries),
        "num_links": len(unique_links),
    })

print("Количество уникальных research questions:", len(groups))

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in groups:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Сохранено в {OUTPUT_PATH}")

print("\nПример первого объекта:")
print(json.dumps(groups[0], ensure_ascii=False, indent=2)[:3000])