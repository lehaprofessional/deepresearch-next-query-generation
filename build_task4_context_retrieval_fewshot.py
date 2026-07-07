import json
import re
from pathlib import Path
from collections import defaultdict


TRAIN_INPUT_PATH = Path("data/task4_train_context.jsonl")
VAL_INPUT_PATH = Path("data/task4_val_context.jsonl")

OUTPUT_PATH = Path("data/task4_val_context_retrieval_fewshot.jsonl")


MAX_FEWSHOT_EXAMPLES = 4
MAX_FEWSHOT_CONTEXT_CHARS = 1200
MAX_ACTUAL_CONTEXT_CHARS = 6000


STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "from", "as", "at", "is", "are", "was", "were", "be", "been",
    "being", "this", "that", "these", "those", "it", "its", "into", "under",
    "over", "between", "through", "using", "use", "used", "how", "what",
    "which", "when", "where", "why", "does", "do", "did", "can", "could",
    "should", "would", "may", "might", "their", "there", "than", "then",
    "also", "such", "most", "more", "less", "very", "specific"
}


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


def tokenize(text: str):
    tokens = re.findall(r"[A-Za-z0-9%°\\-]+", text.lower())

    return {
        token
        for token in tokens
        if len(token) > 2 and token not in STOPWORDS
    }


def truncate_text(text: str, max_chars: int):
    if not isinstance(text, str):
        return ""

    text = text.strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "\n...[context truncated]"


def format_previous_queries(previous_queries):
    if not previous_queries:
        return "No previous queries."

    return "\n".join(
        f"{i + 1}. {query}"
        for i, query in enumerate(previous_queries)
    )


def input_text_for_retrieval(item):
    """
    Text available at inference time.
    Important: validation target_query is NOT used here.
    """
    parts = [
        item.get("research_question", ""),
        " ".join(item.get("previous_queries", [])),
        item.get("visited_context", ""),
    ]

    return "\n".join(parts)


def train_text_for_retrieval(item):
    """
    For train examples we can use the known target query,
    because these are examples available for few-shot prompting.
    """
    parts = [
        item.get("research_question", ""),
        " ".join(item.get("previous_queries", [])),
        item.get("visited_context", ""),
        item.get("target_query", ""),
    ]

    return "\n".join(parts)


def jaccard_similarity(tokens_a, tokens_b):
    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def select_fewshot_examples(val_item, train_items):
    val_tokens = tokenize(input_text_for_retrieval(val_item))

    scored = []

    for train_item in train_items:
        train_tokens = tokenize(train_text_for_retrieval(train_item))
        score = jaccard_similarity(val_tokens, train_tokens)

        # Небольшой бонус, если номера шагов поиска близки.
        if abs(int(val_item["query_id"]) - int(train_item["query_id"])) <= 2:
            score += 0.03

        scored.append((score, train_item))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    per_question_count = defaultdict(int)

    for score, train_item in scored:
        question = train_item["research_question"]

        # Не берём слишком много примеров из одного research_question.
        if per_question_count[question] >= 2:
            continue

        selected.append(train_item)
        per_question_count[question] += 1

        if len(selected) >= MAX_FEWSHOT_EXAMPLES:
            break

    return selected


def format_fewshot_example(example, index):
    context = truncate_text(
        example.get("visited_context", ""),
        MAX_FEWSHOT_CONTEXT_CHARS
    )

    previous_queries = format_previous_queries(
        example.get("previous_queries", [])
    )

    return f"""Example {index}

Research question:
{example["research_question"]}

Current query step:
{example["query_id"]}

Visited sources context:
{context}

Previous generated queries:
{previous_queries}

Correct next search query:
{example["target_query"]}"""


def build_prompt(val_item, fewshot_examples):
    fewshot_text = "\n\n---\n\n".join(
        format_fewshot_example(example, i + 1)
        for i, example in enumerate(fewshot_examples)
    )

    actual_context = truncate_text(
        val_item.get("visited_context", ""),
        MAX_ACTUAL_CONTEXT_CHARS
    )

    previous_queries = format_previous_queries(
        val_item.get("previous_queries", [])
    )

    prompt = f"""You are a search-query planner for a DeepResearch agent.

The agent investigates a complex research question step by step.
Your task is to generate exactly one next search query.

You are given several relevant training examples.
Use them only to understand the style and level of specificity of good search queries.

A good next query should:
- explore a useful missing aspect of the research question;
- use concrete technical terms from the visited sources when helpful;
- avoid repeating previous queries;
- avoid simply restating the research question;
- be suitable for web search or academic search;
- be concise, usually 7-18 words;
- return only the query, without explanation.

Relevant training examples:

{fewshot_text}

---

Now solve the actual case.

Research question:
{val_item["research_question"]}

Current query step:
{val_item["query_id"]}

Visited sources context:
{actual_context}

Previous generated queries:
{previous_queries}

Next search query:"""

    return prompt


def main():
    train_items = load_jsonl(TRAIN_INPUT_PATH)
    val_items = load_jsonl(VAL_INPUT_PATH)

    output_items = []

    for val_item in val_items:
        fewshot_examples = select_fewshot_examples(val_item, train_items)

        item = val_item.copy()
        item["prompt"] = build_prompt(val_item, fewshot_examples)
        item["fewshot_target_queries"] = [
            example["target_query"]
            for example in fewshot_examples
        ]
        item["fewshot_query_ids"] = [
            example["query_id"]
            for example in fewshot_examples
        ]

        output_items.append(item)

    save_jsonl(OUTPUT_PATH, output_items)

    print("Saved:", OUTPUT_PATH)
    print("Validation examples:", len(output_items))
    print("Few-shot examples per prompt:", MAX_FEWSHOT_EXAMPLES)

    print()
    print("Prompt preview:")
    print(output_items[0]["prompt"][:3500])

    print()
    print("Few-shot target queries for first example:")
    for query in output_items[0]["fewshot_target_queries"]:
        print("-", query)


if __name__ == "__main__":
    main()