import json
from pathlib import Path


CONTEXT_V1_PATH = Path("runs/llm_context_v1_predictions.jsonl")
CONTEXT_RETRIEVAL_PATH = Path("runs/llm_context_retrieval_fewshot_v2_predictions.jsonl")
CONTEXT_PROMPT_V2_PATH = Path("runs/llm_context_prompt_v2_predictions.jsonl")

OUTPUT_PATH = Path("runs/llm_context_hybrid_router_predictions.jsonl")


def load_jsonl(path: Path):
    items = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    return items


def save_jsonl(path: Path, items):
    path.parent.mkdir(exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def select_prediction_source(question: str):
    question_lower = question.lower()

    if "ruhrstahl" in question_lower or "rh" in question_lower or "snorkel" in question_lower:
        return "context_v1"

    if "basic oxygen furnace" in question_lower or "bof" in question_lower:
        return "context_retrieval_fewshot_v2"

    if "ladle furnace" in question_lower or "slag line" in question_lower:
        return "context_prompt_v2"

    return "context_prompt_v2"


def main():
    context_v1 = load_jsonl(CONTEXT_V1_PATH)
    context_retrieval = load_jsonl(CONTEXT_RETRIEVAL_PATH)
    context_prompt_v2 = load_jsonl(CONTEXT_PROMPT_V2_PATH)

    if not (len(context_v1) == len(context_retrieval) == len(context_prompt_v2)):
        raise ValueError("Prediction files have different number of examples.")

    output_items = []

    source_counts = {
        "context_v1": 0,
        "context_retrieval_fewshot_v2": 0,
        "context_prompt_v2": 0,
    }

    for item_v1, item_retrieval, item_v2 in zip(
        context_v1,
        context_retrieval,
        context_prompt_v2,
    ):
        question = item_v2["research_question"]
        source = select_prediction_source(question)

        if source == "context_v1":
            selected = item_v1
        elif source == "context_retrieval_fewshot_v2":
            selected = item_retrieval
        else:
            selected = item_v2

        source_counts[source] += 1

        output_item = item_v2.copy()
        output_item["generated_query"] = selected["generated_query"]
        output_item["hybrid_selected_source"] = source

        output_items.append(output_item)

    save_jsonl(OUTPUT_PATH, output_items)

    print("Saved:", OUTPUT_PATH)
    print("Examples:", len(output_items))
    print("Source counts:")

    for source, count in source_counts.items():
        print(f"- {source}: {count}")


if __name__ == "__main__":
    main()