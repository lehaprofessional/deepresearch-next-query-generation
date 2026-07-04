import json
import random
from collections import Counter, defaultdict
from pathlib import Path


INPUT_PATH = "task4_prompts_v2.jsonl"
OUTPUT_DIR = Path("data")
TRAIN_PATH = OUTPUT_DIR / "task4_train_v2.jsonl"
VAL_PATH = OUTPUT_DIR / "task4_val_v2.jsonl"

SEED = 42
VAL_QUESTIONS_COUNT = 3


def load_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def save_jsonl(items, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


items = load_jsonl(INPUT_PATH)

by_question = defaultdict(list)
for item in items:
    by_question[item["research_question"]].append(item)

questions = list(by_question.keys())

random.seed(SEED)
random.shuffle(questions)

val_questions = set(questions[:VAL_QUESTIONS_COUNT])
train_questions = set(questions[VAL_QUESTIONS_COUNT:])

train_items = []
val_items = []

for question, question_items in by_question.items():
    if question in val_questions:
        val_items.extend(question_items)
    else:
        train_items.extend(question_items)

OUTPUT_DIR.mkdir(exist_ok=True)

save_jsonl(train_items, TRAIN_PATH)
save_jsonl(val_items, VAL_PATH)

print("Всего примеров:", len(items))
print("Всего research questions:", len(questions))
print("Train examples:", len(train_items))
print("Validation examples:", len(val_items))

print("\nValidation research questions:")
for q in sorted(val_questions):
    print("-", q[:140] + ("..." if len(q) > 140 else ""))

print()
print(f"Saved train: {TRAIN_PATH}")
print(f"Saved val:   {VAL_PATH}")

print("\nРаспределение query_id в validation:")
print(Counter(item["query_id"] for item in val_items))