import json
from pathlib import Path


INPUT_PATH = "prepared_task4_query_generation.jsonl"
OUTPUT_PATH = "task4_prompts_v2.jsonl"


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

    prompt = f"""You are a search-query planner inside a DeepResearch agent.

Your goal is NOT to answer the research question.
Your goal is to generate the next useful web search query.

Research question:
{research_question}

Current query step:
{query_id}

Previous search queries:
{previous_block}

Think about what kind of information is still missing.
Generate a query that explores ONE new search direction.

A good query should focus on one of these:
- a specific mechanism
- a material or refractory composition
- an operational parameter
- an industrial practice
- a numerical range or condition
- a company product or supplier term
- a standard, patent, trial, or paper
- a failure mode or degradation mechanism

Rules:
- Generate exactly one search query.
- Do not answer the research question.
- Do not generate a broad restatement of the research question.
- Do not repeat previous queries.
- Do not use full sentences.
- Prefer compact English technical search terms.
- The query should be 6 to 14 words long.
- Return only the query.

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