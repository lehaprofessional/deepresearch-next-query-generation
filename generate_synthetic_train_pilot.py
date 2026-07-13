from __future__ import annotations

import json
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


# -----------------------------------------------------------------------------
# Paths and settings
# -----------------------------------------------------------------------------

REAL_TRAIN_PATH = Path("data/lora/train_real_sft.jsonl")
SYNTHETIC_PATH = Path("data/synthetic/task4_synthetic_pilot.jsonl")
AUDIT_PATH = Path("data/synthetic/task4_synthetic_pilot_audit.jsonl")
MIXED_PATH = Path("data/lora/train_mixed_pilot_sft.jsonl")
SUMMARY_PATH = Path("results/synthetic_pilot_summary.json")

load_dotenv(override=True)

BASE_URL = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "")
TEACHER_MODEL = (
    os.getenv("SYNTHETIC_MODEL_NAME")
    or os.getenv("MODEL_NAME")
    or "deepseek/deepseek-chat-v3.1"
)

TARGET_SYNTHETIC_EXAMPLES = int(os.getenv("TARGET_SYNTHETIC_EXAMPLES", "100"))
MAX_SOURCE_EXAMPLES = int(os.getenv("MAX_SOURCE_EXAMPLES", "180"))
CANDIDATES_PER_EXAMPLE = int(os.getenv("CANDIDATES_PER_EXAMPLE", "3"))
SEED = int(os.getenv("SYNTHETIC_SEED", "42"))
MAX_RETRIES = int(os.getenv("MAX_API_RETRIES", "4"))

MIN_QUERY_TOKENS = int(os.getenv("MIN_QUERY_TOKENS", "5"))
MAX_QUERY_TOKENS = int(os.getenv("MAX_QUERY_TOKENS", "20"))
MIN_KEEP_RATIO = float(os.getenv("MIN_CONTEXT_KEEP_RATIO", "0.55"))
MAX_KEEP_RATIO = float(os.getenv("MAX_CONTEXT_KEEP_RATIO", "0.80"))
MAX_TARGET_JACCARD = float(os.getenv("MAX_TARGET_JACCARD", "0.85"))
MAX_PREVIOUS_JACCARD = float(os.getenv("MAX_PREVIOUS_JACCARD", "0.75"))
MIN_GROUNDED_TOKENS = int(os.getenv("MIN_GROUNDED_CONTENT_TOKENS", "2"))
TEMPERATURE = float(os.getenv("SYNTHETIC_TEMPERATURE", "0.7"))
MAX_COMPLETION_TOKENS = int(os.getenv("SYNTHETIC_MAX_TOKENS", "300"))


# -----------------------------------------------------------------------------
# Text helpers
# -----------------------------------------------------------------------------

CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")
NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:[-'][A-Za-zА-Яа-яЁё0-9]+)?")

VISITED_MARKERS = (
    "Visited sources context:",
    "Visited source context:",
    "Visited website descriptions:",
    "Visited sources:",
    "Source descriptions:",
)
PREVIOUS_MARKERS = (
    "Previous generated queries:",
    "Previous search queries:",
    "Previous queries:",
    "Past queries:",
)
QUESTION_MARKERS = ("Research question:", "Question:")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "between", "by", "can",
    "does", "for", "from", "how", "in", "into", "is", "it", "of", "on",
    "or", "that", "the", "their", "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with", "versus", "vs",
    "а", "без", "бы", "был", "была", "были", "было", "в", "во", "для",
    "до", "его", "ее", "её", "если", "же", "за", "и", "из", "или", "к",
    "как", "каким", "какой", "между", "на", "над", "не", "но", "о", "об",
    "от", "по", "под", "при", "с", "со", "то", "у", "что", "чем", "эта",
    "эти", "это",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}: {error}") from error
            if not isinstance(item, dict):
                raise ValueError(f"Expected JSON object in {path}, line {line_number}")
            items.append(item)
    return items


def save_jsonl(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(item, ensure_ascii=False) + "\n")


def backup(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.stem}_backup_{stamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    print(f"Backup: {backup_path}")


def get_message(item: dict[str, Any], role: str, reverse: bool = False) -> dict[str, Any] | None:
    messages = item.get("messages", [])
    iterable = reversed(messages) if reverse else messages
    for message in iterable:
        if message.get("role") == role:
            return message
    return None


def get_assistant_query(item: dict[str, Any]) -> str:
    message = get_message(item, "assistant", reverse=True)
    return str(message.get("content", "")).strip() if message else ""


def get_user_prompt(item: dict[str, Any]) -> str:
    message = get_message(item, "user", reverse=True)
    return str(message.get("content", "")).strip() if message else ""


def get_system_prompt(item: dict[str, Any]) -> str:
    message = get_message(item, "system")
    if message:
        return str(message.get("content", "")).strip()
    return "You generate concise technical search queries for a DeepResearch agent."


def get_research_question(item: dict[str, Any]) -> str:
    metadata = item.get("metadata", {})
    if metadata.get("research_question"):
        return str(metadata["research_question"]).strip()

    prompt = get_user_prompt(item)
    for marker in QUESTION_MARKERS:
        match = re.search(
            re.escape(marker) + r"\s*(.+?)(?=\n[A-ZА-Я][^\n]{0,50}:|\Z)",
            prompt,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return " ".join(match.group(1).split())
    return prompt[:500].strip()


def detect_query_language(real_target: str) -> str:
    """Use the language of the real parent target, not the question language."""
    return "Russian" if CYRILLIC_RE.search(real_target) else "English"


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in WORD_RE.finditer(text)]


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in tokenize(text)
        if token not in STOPWORDS and len(token) >= 3 and not token.isdigit()
    }


def jaccard(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def normalize_query(value: Any) -> str:
    """Unwrap malformed values such as [\"query\"] or \"[\\\"query\\\"]\"."""
    if not isinstance(value, str):
        return ""

    text = value.strip()
    for _ in range(5):
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            break
        if isinstance(parsed, str):
            text = parsed.strip()
            continue
        if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], str):
            text = parsed[0].strip()
            continue
        break

    text = text.strip().strip("`").strip().strip('"').strip("'").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip().strip('"').strip("'").strip()
    return " ".join(text.split())


def find_marker(text: str, markers: tuple[str, ...], start: int = 0) -> tuple[int, str] | None:
    lower = text.lower()
    matches = []
    for marker in markers:
        position = lower.find(marker.lower(), start)
        if position >= 0:
            matches.append((position, marker))
    return min(matches, key=lambda value: value[0]) if matches else None


def split_source_blocks(section: str) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", section.strip()) if block.strip()]
    if len(blocks) > 1:
        return blocks
    blocks = [
        block.strip()
        for block in re.split(
            r"(?=^\s*(?:Source|Link|Website)\s*\d*\s*:)",
            section.strip(),
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if block.strip()
    ]
    return blocks or ([section.strip()] if section.strip() else [])


def reduce_visited_sources(user_prompt: str, rng: random.Random) -> tuple[str, int, int]:
    visited = find_marker(user_prompt, VISITED_MARKERS)
    if not visited:
        return user_prompt, 0, 0

    visited_position, visited_label = visited
    section_start = visited_position + len(visited_label)
    previous = find_marker(user_prompt, PREVIOUS_MARKERS, start=section_start)
    section_end = previous[0] if previous else len(user_prompt)

    blocks = split_source_blocks(user_prompt[section_start:section_end])
    total = len(blocks)
    if total <= 1:
        return user_prompt, total, total

    keep_ratio = rng.uniform(MIN_KEEP_RATIO, MAX_KEEP_RATIO)
    keep_count = max(1, min(total, round(total * keep_ratio)))
    kept_indices = sorted(rng.sample(range(total), keep_count))
    reduced = "\n\n".join(blocks[index] for index in kept_indices)

    modified = (
        user_prompt[:section_start]
        + "\n"
        + reduced
        + "\n\n"
        + user_prompt[section_end:].lstrip()
    )
    return modified.strip(), total, keep_count


def extract_previous_queries(user_prompt: str) -> list[str]:
    previous = find_marker(user_prompt, PREVIOUS_MARKERS)
    if not previous:
        return []

    position, label = previous
    section = user_prompt[position + len(label):].strip()
    queries: list[str] = []
    for line in section.splitlines():
        line = re.sub(r"^\s*[-*•]\s*", "", line.strip())
        line = re.sub(r"^\s*\d+[.)]\s*", "", line)
        line = normalize_query(line)
        if not line:
            continue
        if re.match(r"^[A-Za-zА-Яа-яЁё][^:]{0,40}:$", line):
            break
        queries.append(line)
    return queries


def parse_candidates(response_text: str) -> list[str]:
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", text)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

    if isinstance(parsed, list):
        raw_candidates = parsed
    elif isinstance(parsed, str):
        raw_candidates = [parsed]
    else:
        raw_candidates = []
        for line in text.splitlines():
            cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if cleaned:
                raw_candidates.append(cleaned)

    candidates: list[str] = []
    seen: set[str] = set()
    for raw in raw_candidates:
        candidate = normalize_query(raw)
        normalized = candidate.lower()
        if candidate and normalized not in seen:
            seen.add(normalized)
            candidates.append(candidate)
    return candidates[:CANDIDATES_PER_EXAMPLE]


def language_is_valid(candidate: str, expected_language: str) -> bool:
    tokens = tokenize(candidate)
    if not tokens:
        return False
    cyrillic_count = sum(bool(CYRILLIC_RE.search(token)) for token in tokens)
    if expected_language == "Russian":
        return cyrillic_count >= 1
    return cyrillic_count / len(tokens) <= 0.25


def evaluate_candidate(
    candidate: str,
    expected_language: str,
    original_target: str,
    previous_queries: list[str],
    grounding_text: str,
    existing_synthetic: set[str],
) -> dict[str, Any]:
    candidate = normalize_query(candidate)
    normalized = candidate.lower()
    reasons: list[str] = []
    token_count = len(tokenize(candidate))

    if token_count < MIN_QUERY_TOKENS:
        reasons.append("too_short")
    if token_count > MAX_QUERY_TOKENS:
        reasons.append("too_long")
    if "\n" in candidate:
        reasons.append("multiline")
    if candidate.endswith((".", "!", "?", ";")):
        reasons.append("sentence_ending")
    if not language_is_valid(candidate, expected_language):
        reasons.append("wrong_language")
    if normalized in existing_synthetic:
        reasons.append("duplicate_synthetic")

    target_similarity = jaccard(candidate, original_target)
    if normalized == original_target.lower().strip():
        reasons.append("exact_original_target")
    elif target_similarity >= MAX_TARGET_JACCARD:
        reasons.append("too_similar_to_original_target")

    max_previous_similarity = max(
        (jaccard(candidate, previous) for previous in previous_queries),
        default=0.0,
    )
    if max_previous_similarity >= MAX_PREVIOUS_JACCARD:
        reasons.append("too_similar_to_previous_query")

    prompt_numbers = set(NUMBER_RE.findall(grounding_text))
    candidate_numbers = set(NUMBER_RE.findall(candidate))
    if not candidate_numbers.issubset(prompt_numbers):
        reasons.append("unsupported_numbers")

    grounded = sorted(content_tokens(candidate) & content_tokens(grounding_text))
    if len(grounded) < MIN_GROUNDED_TOKENS:
        reasons.append("weak_grounding")

    score = (
        len(grounded)
        - max_previous_similarity
        - 0.25 * target_similarity
        - 0.05 * abs(token_count - 11)
    )

    return {
        "candidate": candidate,
        "accepted": not reasons,
        "reasons": reasons,
        "token_count": token_count,
        "target_jaccard": target_similarity,
        "max_previous_jaccard": max_previous_similarity,
        "grounded_token_count": len(grounded),
        "grounded_tokens": grounded,
        "score": score,
        "expected_language": expected_language,
    }


# -----------------------------------------------------------------------------
# Teacher call
# -----------------------------------------------------------------------------


def build_teacher_prompt(state_prompt: str, expected_language: str) -> str:
    return f"""Generate {CANDIDATES_PER_EXAMPLE} alternative next search queries for
the DeepResearch state below.

The required output language is: {expected_language}.

Before producing the candidates, silently identify:
1. the central subject of the research question;
2. the requested metric, outcome, or comparison;
3. named entities and technical terms that must not be lost.

Requirements:
- Each candidate must be one concise {expected_language} technical search query.
- Use only {expected_language}, except for standard abbreviations, product names, formulas, and proper nouns.
- Keep each candidate between {MIN_QUERY_TOKENS} and {MAX_QUERY_TOKENS} tokens.
- Every candidate must remain directly useful for answering the original research question.
- Use concrete information from the research question and visited sources.
- Explore an aspect not already covered by previous queries.
- Preserve at least one distinctive anchor from the research question or an unambiguous technical synonym.
- Anchors may include named materials, additives, technologies, organisms, countries, metrics, processes, or historical events.
- For explicit comparison questions, preserve all central compared entities unless the candidate intentionally investigates one entity separately.
- When investigating one compared entity separately, name that entity explicitly and keep the original comparison outcome or metric.
- For questions about lifespan, fertility rate, wear rate, cost, population share, or another metric, preserve that metric or a direct synonym.
- Do not drift to a neighboring topic merely because it appears in visited-source descriptions.
- Do not invent numerical values absent from the supplied context.
- Do not repeat or closely paraphrase previous queries.
- Do not provide explanations.
- Return only a JSON array of strings.

DeepResearch state:
-------------------
{state_prompt}
"""


def call_teacher(client: OpenAI, prompt: str) -> tuple[list[str], str]:
    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=TEACHER_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise multilingual technical search-query planner. "
                            "Follow the requested output language and return valid JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_COMPLETION_TOKENS,
            )
            response_text = (response.choices[0].message.content or "").strip()
            return parse_candidates(response_text), response_text
        except Exception as error:
            last_error = error
            status_code = getattr(error, "status_code", None)
            print(f"API error, attempt {attempt}/{MAX_RETRIES}: {error}")
            if status_code == 403:
                raise RuntimeError(
                    "Teacher API returned 403. Keep the working VPN route enabled and check the API key."
                ) from error
            if attempt < MAX_RETRIES:
                time.sleep(min(2 ** attempt, 10))

    raise RuntimeError(f"Teacher API failed after {MAX_RETRIES} attempts") from last_error


# -----------------------------------------------------------------------------
# Dataset construction
# -----------------------------------------------------------------------------


def balanced_source_order(items: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        groups[get_research_question(item)].append(item)
    for group in groups.values():
        rng.shuffle(group)

    questions = list(groups)
    rng.shuffle(questions)
    ordered: list[dict[str, Any]] = []

    while questions:
        remaining: list[str] = []
        for question in questions:
            if groups[question]:
                ordered.append(groups[question].pop())
            if groups[question]:
                remaining.append(question)
        rng.shuffle(remaining)
        questions = remaining
    return ordered


def build_synthetic_item(
    source_item: dict[str, Any],
    modified_user_prompt: str,
    selected: dict[str, Any],
    expected_language: str,
    total_source_blocks: int,
    kept_source_blocks: int,
) -> dict[str, Any]:
    metadata = deepcopy(source_item.get("metadata", {}))
    parent_query_id = (
        metadata.get("query_id")
        or metadata.get("id")
        or metadata.get("parent_query_id")
    )
    metadata.update(
        {
            "source": "synthetic",
            "teacher_model": TEACHER_MODEL,
            "parent_query_id": parent_query_id,
            "target_language": expected_language,
            "context_source_count_original": total_source_blocks,
            "context_source_count_kept": kept_source_blocks,
            "quality_score": selected["score"],
            "quality_target_jaccard": selected["target_jaccard"],
            "quality_max_previous_jaccard": selected["max_previous_jaccard"],
            "quality_grounded_token_count": selected["grounded_token_count"],
            "quality_grounded_tokens": selected["grounded_tokens"],
        }
    )

    return {
        "messages": [
            {"role": "system", "content": get_system_prompt(source_item)},
            {"role": "user", "content": modified_user_prompt},
            {"role": "assistant", "content": selected["candidate"]},
        ],
        "metadata": metadata,
    }


def main() -> None:
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing from .env")
    if TARGET_SYNTHETIC_EXAMPLES <= 0:
        raise ValueError("TARGET_SYNTHETIC_EXAMPLES must be greater than zero")
    if MAX_SOURCE_EXAMPLES <= 0:
        raise ValueError("MAX_SOURCE_EXAMPLES must be greater than zero")
    if not REAL_TRAIN_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {REAL_TRAIN_PATH}")

    rng = random.Random(SEED)
    real_items = load_jsonl(REAL_TRAIN_PATH)
    source_items = balanced_source_order(real_items, rng)[:MAX_SOURCE_EXAMPLES]
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    print("=" * 72)
    print("TASK 4 SYNTHETIC TRAIN PILOT")
    print("=" * 72)
    print(f"Teacher model: {TEACHER_MODEL}")
    print(f"Real train examples: {len(real_items)}")
    print(f"Target synthetic examples: {TARGET_SYNTHETIC_EXAMPLES}")
    print(f"Maximum source examples: {MAX_SOURCE_EXAMPLES}")
    print("Language policy: same language as the real parent target")
    print()

    for path in (SYNTHETIC_PATH, AUDIT_PATH, MIXED_PATH, SUMMARY_PATH):
        backup(path)
    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text("", encoding="utf-8")

    accepted_items: list[dict[str, Any]] = []
    existing_synthetic: set[str] = set()
    rejection_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    api_failures = 0
    processed_sources = 0

    for source_index, source_item in enumerate(source_items, start=1):
        if len(accepted_items) >= TARGET_SYNTHETIC_EXAMPLES:
            break

        processed_sources += 1
        real_target = get_assistant_query(source_item)
        user_prompt = get_user_prompt(source_item)
        question = get_research_question(source_item)

        if not real_target or not user_prompt:
            rejection_counts["missing_user_or_target"] += 1
            append_jsonl(
                AUDIT_PATH,
                {
                    "source_index": source_index,
                    "research_question": question,
                    "accepted": False,
                    "source_rejection_reason": "missing_user_or_target",
                },
            )
            continue

        expected_language = detect_query_language(real_target)
        modified_prompt, total_blocks, kept_blocks = reduce_visited_sources(user_prompt, rng)
        previous_queries = extract_previous_queries(modified_prompt)
        teacher_prompt = build_teacher_prompt(modified_prompt, expected_language)

        try:
            candidates, raw_response = call_teacher(client, teacher_prompt)
        except Exception as error:
            api_failures += 1
            append_jsonl(
                AUDIT_PATH,
                {
                    "source_index": source_index,
                    "research_question": question,
                    "expected_language": expected_language,
                    "accepted": False,
                    "source_rejection_reason": "api_failure",
                    "error": str(error),
                },
            )
            if "403" in str(error):
                raise
            continue

        grounding_text = question + "\n" + modified_prompt
        evaluations = [
            evaluate_candidate(
                candidate,
                expected_language,
                real_target,
                previous_queries,
                grounding_text,
                existing_synthetic,
            )
            for candidate in candidates
        ]
        valid = [evaluation for evaluation in evaluations if evaluation["accepted"]]
        selected = max(valid, key=lambda value: value["score"]) if valid else None

        for evaluation in evaluations:
            for reason in evaluation["reasons"]:
                rejection_counts[reason] += 1

        append_jsonl(
            AUDIT_PATH,
            {
                "source_index": source_index,
                "research_question": question,
                "parent_target_language": expected_language,
                "context_source_count_original": total_blocks,
                "context_source_count_kept": kept_blocks,
                "previous_query_count": len(previous_queries),
                "candidate_count": len(candidates),
                "raw_teacher_response": raw_response,
                "candidates": evaluations,
                "accepted": selected is not None,
                "selected_candidate": selected["candidate"] if selected else None,
            },
        )

        if selected:
            accepted_items.append(
                build_synthetic_item(
                    source_item,
                    modified_prompt,
                    selected,
                    expected_language,
                    total_blocks,
                    kept_blocks,
                )
            )
            existing_synthetic.add(selected["candidate"].lower())
            language_counts[expected_language] += 1

        print(
            f"[{source_index:03d}/{len(source_items):03d}] "
            f"accepted={len(accepted_items):03d}/{TARGET_SYNTHETIC_EXAMPLES:03d} | "
            f"candidates={len(candidates)} | language={expected_language}"
        )

    mixed_items = real_items + accepted_items
    save_jsonl(SYNTHETIC_PATH, accepted_items)
    save_jsonl(MIXED_PATH, mixed_items)

    summary = {
        "teacher_model": TEACHER_MODEL,
        "seed": SEED,
        "real_train_examples": len(real_items),
        "target_synthetic_examples": TARGET_SYNTHETIC_EXAMPLES,
        "processed_source_examples": processed_sources,
        "accepted_synthetic_examples": len(accepted_items),
        "api_failures": api_failures,
        "mixed_train_examples": len(mixed_items),
        "language_policy": "same_as_parent_real_target",
        "accepted_language_distribution": dict(language_counts),
        "rejection_reasons": dict(rejection_counts.most_common()),
        "paths": {
            "synthetic": str(SYNTHETIC_PATH),
            "audit": str(AUDIT_PATH),
            "mixed": str(MIXED_PATH),
        },
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 72)
    print("SYNTHETIC PILOT FINISHED")
    print("=" * 72)
    print(f"Processed sources: {processed_sources}")
    print(f"Accepted synthetic: {len(accepted_items)}")
    print(f"API failures: {api_failures}")
    print(f"Mixed train size: {len(mixed_items)}")
    print("Accepted languages: " + ", ".join(f"{key}={value}" for key, value in language_counts.items()))

    print("\nTop rejection reasons:")
    if rejection_counts:
        for reason, count in rejection_counts.most_common():
            print(f"- {reason}: {count}")
    else:
        print("- none")

    print("\nSaved:")
    print(SYNTHETIC_PATH)
    print(AUDIT_PATH)
    print(MIXED_PATH)
    print(SUMMARY_PATH)

    if len(accepted_items) < TARGET_SYNTHETIC_EXAMPLES:
        print("\nWARNING: target count was not reached; increase MAX_SOURCE_EXAMPLES or inspect the audit.")


if __name__ == "__main__":
    main()
