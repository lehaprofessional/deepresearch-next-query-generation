import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

INPUT_PATH = Path(os.getenv("INPUT_PATH", "data/task4_val.jsonl"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", "runs/llm_baseline_predictions.jsonl"))

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("MODEL_NAME")

MAX_EXAMPLES = int(os.getenv("MAX_EXAMPLES", "0"))  # 0 = all examples


def load_jsonl(path: Path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def clean_query(text: str) -> str:
    text = text.strip()

    # Берём только первую непустую строку
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        text = lines[0]

    # Убираем возможные маркеры списка
    for prefix in ["- ", "* ", "1. ", "Query:", "Search query:", "Next search query:"]:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()

    # Убираем кавычки по краям
    text = text.strip("\"'` ")

    return text


def generate_query(client: OpenAI, prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You generate concise and precise web search queries for a DeepResearch agent.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
        max_tokens=80,
    )

    text = response.choices[0].message.content
    return clean_query(text)


def main():
    if not OPENAI_BASE_URL:
        raise ValueError("OPENAI_BASE_URL is not set. Add it to .env or PowerShell env variables.")

    if not MODEL_NAME:
        raise ValueError("MODEL_NAME is not set. Add it to .env or PowerShell env variables.")

    client = OpenAI(
        base_url=OPENAI_BASE_URL,
        api_key=OPENAI_API_KEY,
    )

    items = load_jsonl(INPUT_PATH)

    if MAX_EXAMPLES > 0:
        items = items[:MAX_EXAMPLES]

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    print("Model:", MODEL_NAME)
    print("Base URL:", OPENAI_BASE_URL)
    print("Examples:", len(items))
    print("Output:", OUTPUT_PATH)
    print()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for i, item in enumerate(items, start=1):
            print(f"[{i}/{len(items)}] query_id={item['query_id']}")

            try:
                generated_query = generate_query(client, item["prompt"])
                error = None
            except Exception as e:
                generated_query = ""
                error = str(e)

            out = {
                "research_question": item["research_question"],
                "query_id": item["query_id"],
                "prompt": item["prompt"],
                "previous_queries": item["previous_queries"],
                "target_query": item["target_query"],
                "generated_query": generated_query,
                "error": error,
            }

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

            print("TARGET:   ", item["target_query"])
            print("GENERATED:", generated_query)
            if error:
                print("ERROR:", error)
            print("-" * 80)

            time.sleep(0.2)

    print("Done.")


if __name__ == "__main__":
    main()