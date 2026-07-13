import json
from pathlib import Path
from statistics import mean


TRAIN_INPUT = Path("data/task4_train_context_v2.jsonl")
VAL_INPUT = Path("data/task4_val_context_v2.jsonl")

OUTPUT_DIR = Path("data/lora")
TRAIN_OUTPUT = OUTPUT_DIR / "train_real_sft.jsonl"
VAL_OUTPUT = OUTPUT_DIR / "val_real_sft.jsonl"


SYSTEM_PROMPT = (
    "You are a search-query planner for a DeepResearch agent. "
    "Generate exactly one concise English web search query. "
    "The query must explore a useful new aspect of the research question, "
    "use information from visited website descriptions when relevant, "
    "and must not repeat previous queries. "
    "Return only the search query without explanations or quotation marks."
)


def load_jsonl(path: Path) -> list[dict]:
    items = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {error}"
                ) from error

    return items


def save_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def normalize_text(value) -> str:
    if value is None:
        return ""

    return str(value).strip()


def format_previous_queries(value) -> str:
    if not value:
        return "No previous queries."

    if isinstance(value, str):
        return value.strip() or "No previous queries."

    if isinstance(value, list):
        queries = [
            normalize_text(query)
            for query in value
            if normalize_text(query)
        ]

        if not queries:
            return "No previous queries."

        return "\n".join(
            f"{index}. {query}"
            for index, query in enumerate(queries, start=1)
        )

    return normalize_text(value)


def format_visited_context(item: dict) -> str:
    visited_context = item.get("visited_context")

    if isinstance(visited_context, str):
        text = visited_context.strip()
        return text or "No website descriptions are available."

    if isinstance(visited_context, list):
        descriptions = []

        for source in visited_context:
            if isinstance(source, dict):
                description = normalize_text(source.get("description"))
                link = normalize_text(source.get("link"))

                if description:
                    descriptions.append(description)
                elif link:
                    descriptions.append(link)
            else:
                text = normalize_text(source)

                if text:
                    descriptions.append(text)

        if descriptions:
            return "\n".join(
                f"- {description}"
                for description in descriptions
            )

    context_sources = item.get("context_sources")

    if isinstance(context_sources, list):
        descriptions = []

        for source in context_sources:
            if not isinstance(source, dict):
                continue

            description = normalize_text(source.get("description"))
            link = normalize_text(source.get("link"))

            if description:
                descriptions.append(description)
            elif link:
                descriptions.append(link)

        if descriptions:
            return "\n".join(
                f"- {description}"
                for description in descriptions
            )

    return "No website descriptions are available."


def build_user_prompt(item: dict) -> str:
    existing_prompt = normalize_text(item.get("prompt"))

    if existing_prompt:
        return existing_prompt

    research_question = normalize_text(
        item.get("research_question")
        or item.get("research_questions")
    )

    query_id = item.get("query_id", "")
    previous_queries = format_previous_queries(
        item.get("previous_queries")
    )
    visited_context = format_visited_context(item)

    return (
        f"Research question:\n"
        f"{research_question}\n\n"
        f"Current query step:\n"
        f"{query_id}\n\n"
        f"Previously generated queries:\n"
        f"{previous_queries}\n\n"
        f"Visited website descriptions:\n"
        f"{visited_context}\n\n"
        f"Generate the next search query."
    )


def convert_item(item: dict, split_name: str) -> dict:
    target_query = normalize_text(item.get("target_query"))

    if not target_query:
        raise ValueError(
            "Example does not contain a non-empty target_query: "
            f"{item}"
        )

    research_question = normalize_text(
        item.get("research_question")
        or item.get("research_questions")
    )

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": build_user_prompt(item),
            },
            {
                "role": "assistant",
                "content": target_query,
            },
        ],
        "metadata": {
            "split": split_name,
            "research_question": research_question,
            "query_id": item.get("query_id", ""),
        },
    }


def validate_dataset(
    train_items: list[dict],
    val_items: list[dict],
) -> None:
    train_questions = {
        item["metadata"]["research_question"]
        for item in train_items
        if item["metadata"]["research_question"]
    }

    val_questions = {
        item["metadata"]["research_question"]
        for item in val_items
        if item["metadata"]["research_question"]
    }

    overlap = train_questions & val_questions

    train_pairs = {
        (
            item["messages"][1]["content"],
            item["messages"][2]["content"],
        )
        for item in train_items
    }

    val_pairs = {
        (
            item["messages"][1]["content"],
            item["messages"][2]["content"],
        )
        for item in val_items
    }

    duplicate_pairs = train_pairs & val_pairs

    if overlap:
        print()
        print("WARNING: research-question leakage detected:")
        for question in sorted(overlap):
            print("-", question)

    if duplicate_pairs:
        print()
        print(
            "WARNING: exact input/target pairs occur "
            "in both train and validation:"
        )
        print(len(duplicate_pairs))

    if not overlap:
        print("Research-question leakage: 0")

    if not duplicate_pairs:
        print("Exact train/validation duplicate pairs: 0")


def print_statistics(
    name: str,
    items: list[dict],
) -> None:
    user_lengths = [
        len(item["messages"][1]["content"])
        for item in items
    ]

    target_lengths = [
        len(item["messages"][2]["content"])
        for item in items
    ]

    questions = {
        item["metadata"]["research_question"]
        for item in items
        if item["metadata"]["research_question"]
    }

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)
    print("Examples:", len(items))
    print("Research questions:", len(questions))
    print(f"Average input length: {mean(user_lengths):.1f} characters")
    print(f"Maximum input length: {max(user_lengths)} characters")
    print(f"Average target length: {mean(target_lengths):.1f} characters")
    print(f"Maximum target length: {max(target_lengths)} characters")


def main() -> None:
    if not TRAIN_INPUT.exists():
        raise FileNotFoundError(
            f"Train file not found: {TRAIN_INPUT}"
        )

    if not VAL_INPUT.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VAL_INPUT}"
        )

    raw_train = load_jsonl(TRAIN_INPUT)
    raw_val = load_jsonl(VAL_INPUT)

    train_sft = [
        convert_item(item, "train")
        for item in raw_train
    ]

    val_sft = [
        convert_item(item, "validation")
        for item in raw_val
    ]

    validate_dataset(train_sft, val_sft)

    save_jsonl(TRAIN_OUTPUT, train_sft)
    save_jsonl(VAL_OUTPUT, val_sft)

    print_statistics("REAL TRAIN SFT DATASET", train_sft)
    print_statistics("REAL VALIDATION SFT DATASET", val_sft)

    print()
    print("Saved:")
    print(TRAIN_OUTPUT)
    print(VAL_OUTPUT)

    print()
    print("First training example:")
    print(
        json.dumps(
            train_sft[0],
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()