import json
from pathlib import Path


INPUT_PATH = "prepared_task4_query_generation.jsonl"
OUTPUT_PATH = "task4_prompts_baseline.jsonl"


def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def build_prompt(item: dict) -> str:
    research_question = item["research_question"]
    query_id = item["query_id"]
    previous_queries = item["previous_queries"]

    if previous_queries:
        previous_block = "\n".join(
            f"{i + 1}. {query}" for i, query in enumerate(previous_queries)
        )
    else:
        previous_block = "No previous queries."

    prompt = f"""You are a query generation module for a DeepResearch agent.

Your task is to generate the next search engine query for a complex research question.

Research question:
{research_question}

Current query step:
{query_id}

Previous generated queries:
{previous_block}

Requirements:
- Generate exactly one search query.
- The query must be specific and useful for web search.
- Do not repeat previous queries.
- Prefer English technical terminology.
- Do not explain your answer.
- Return only the search query.

Next search query:"""

    return prompt


items = load_jsonl(INPUT_PATH)

output_items = []

for item in items:
    output_items.append({
        "prompt": build_prompt(item),
        "target_query": item["target_query"],
        "research_question": item["research_question"],
        "query_id": item["query_id"],
        "previous_queries": item["previous_queries"],
        "task3_ids": item["task3_ids"],
    })

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for item in output_items:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved: {OUTPUT_PATH}")
print(f"Number of prompts: {len(output_items)}")

print("\nExample prompt:")
print(output_items[0]["prompt"])

print("\nTarget query:")
print(output_items[0]["target_query"])