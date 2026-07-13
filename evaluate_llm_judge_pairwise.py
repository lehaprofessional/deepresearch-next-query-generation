from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


PREDICTION_KEYS = (
    "prediction",
    "generated_query",
    "pred",
    "output",
    "generated",
)

TARGET_KEYS = (
    "target",
    "target_query",
    "reference",
    "gold",
    "expected",
)

SCORE_FIELDS = (
    "relevance",
    "novelty",
    "groundedness",
    "technical_specificity",
    "search_quality",
    "overall",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {error}"
                ) from error

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object in {path}, line {line_number}"
                )

            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {path}")

    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_text(
    row: dict[str, Any],
    keys: tuple[str, ...],
) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def get_prediction(row: dict[str, Any]) -> str:
    prediction = find_text(row, PREDICTION_KEYS)

    if not prediction:
        raise ValueError(
            "Prediction text was not found. "
            f"Available keys: {sorted(row.keys())}"
        )

    return prediction


def get_validation_fields(
    row: dict[str, Any],
) -> tuple[str, str, str]:
    messages = row.get("messages")

    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(
            "Validation row must contain a messages array."
        )

    user_messages = [
        message
        for message in messages
        if message.get("role") == "user"
    ]
    assistant_messages = [
        message
        for message in messages
        if message.get("role") == "assistant"
    ]

    if not user_messages or not assistant_messages:
        raise ValueError(
            "Validation row is missing user or assistant messages."
        )

    state_prompt = str(
        user_messages[-1].get("content", "")
    ).strip()
    target = str(
        assistant_messages[-1].get("content", "")
    ).strip()

    metadata = row.get("metadata", {})
    research_question = str(
        metadata.get("research_question", "")
    ).strip()

    if not research_question:
        match = re.search(
            r"Research question:\s*(.+?)(?=\n[A-ZА-Я][^\n]{0,60}:|\Z)",
            state_prompt,
            flags=re.IGNORECASE | re.DOTALL,
        )
        research_question = (
            " ".join(match.group(1).split())
            if match
            else state_prompt[:500]
        )

    return research_question, state_prompt, target


def strip_code_fence(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s*```$", "", text)

    return text.strip()


def validate_score_block(
    value: Any,
    label: str,
) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")

    normalized: dict[str, float] = {}

    for field in SCORE_FIELDS:
        score_value = value.get(field)

        if not isinstance(score_value, (int, float)):
            raise ValueError(
                f"{label}.{field} must be numeric"
            )

        numeric = float(score_value)

        if not 1.0 <= numeric <= 5.0:
            raise ValueError(
                f"{label}.{field} must be between 1 and 5"
            )

        normalized[field] = numeric

    return normalized


def parse_judgement(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(text)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            raise

        payload = json.loads(match.group(0))

    if not isinstance(payload, dict):
        raise ValueError("Judge response must be a JSON object")

    winner = str(payload.get("winner", "")).upper()

    if winner not in {"A", "B", "TIE"}:
        raise ValueError(
            "winner must be A, B, or TIE"
        )

    reason = str(payload.get("reason", "")).strip()

    return {
        "candidate_a": validate_score_block(
            payload.get("candidate_a"),
            "candidate_a",
        ),
        "candidate_b": validate_score_block(
            payload.get("candidate_b"),
            "candidate_b",
        ),
        "winner": winner,
        "reason": reason,
    }


def build_judge_prompt(
    *,
    state_prompt: str,
    reference_target: str,
    candidate_a: str,
    candidate_b: str,
) -> str:
    return f"""You are evaluating the next web-search query produced by a
DeepResearch agent.

The reference target is one valid next query, but it is not the only
acceptable answer. Judge semantic usefulness, not exact wording.

Evaluate Candidate A and Candidate B independently on a 1-5 scale:

- relevance: directly helps answer the research question;
- novelty: advances beyond the previous queries;
- groundedness: supported by the supplied state and avoids invented facts,
  unsupported numerical values, or incorrect process terminology;
- technical_specificity: contains useful technical entities and constraints;
- search_quality: concise, well-formed, and likely to retrieve useful sources;
- overall: overall suitability as the next search query.

Important:
- Do not reward a candidate merely for copying the reference.
- Penalize incorrect domain substitutions, such as confusing BOF with a
  blast furnace.
- Penalize fabricated numbers or long enumerations not justified by context.
- Ignore which candidate appears first.
- Return valid JSON only.

DeepResearch state:
-------------------
{state_prompt}

Reference target:
-----------------
{reference_target}

Candidate A:
------------
{candidate_a}

Candidate B:
------------
{candidate_b}

Return exactly this JSON schema:
{{
  "candidate_a": {{
    "relevance": 1,
    "novelty": 1,
    "groundedness": 1,
    "technical_specificity": 1,
    "search_quality": 1,
    "overall": 1
  }},
  "candidate_b": {{
    "relevance": 1,
    "novelty": 1,
    "groundedness": 1,
    "technical_specificity": 1,
    "search_quality": 1,
    "overall": 1
  }},
  "winner": "A",
  "reason": "Brief reason in no more than 45 words"
}}
"""


def call_judge(
    client: OpenAI,
    *,
    model: str,
    prompt: str,
    max_retries: int,
) -> tuple[dict[str, Any], str]:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict and impartial evaluator of "
                            "technical search queries. Return JSON only."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0,
                max_tokens=500,
            )

            raw = (
                response.choices[0].message.content or ""
            ).strip()

            return parse_judgement(raw), raw

        except Exception as error:
            last_error = error
            status_code = getattr(error, "status_code", None)

            print(
                f"Judge error, attempt {attempt}/{max_retries}: "
                f"{error}"
            )

            if status_code == 403:
                raise RuntimeError(
                    "Judge API returned 403. Keep the working VPN route "
                    "enabled and verify the API key."
                ) from error

            if attempt < max_retries:
                time.sleep(min(2 ** attempt, 10))

    raise RuntimeError(
        f"Judge failed after {max_retries} attempts"
    ) from last_error


def score_average(
    scores: dict[str, float],
) -> float:
    return sum(scores.values()) / len(scores)


def aggregate_results(
    records: list[dict[str, Any]],
    example_count: int,
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = {}

    for record in records:
        grouped.setdefault(
            int(record["example_index"]),
            [],
        ).append(record)

    example_outcomes: Counter[str] = Counter()
    call_outcomes: Counter[str] = Counter()
    real_scores: dict[str, list[float]] = {
        field: [] for field in SCORE_FIELDS
    }
    mixed_scores: dict[str, list[float]] = {
        field: [] for field in SCORE_FIELDS
    }

    per_example: list[dict[str, Any]] = []

    for example_index in sorted(grouped):
        example_records = grouped[example_index]

        real_votes = 0
        mixed_votes = 0
        tie_votes = 0

        for record in example_records:
            mapped_winner = record["mapped_winner"]
            call_outcomes[mapped_winner] += 1

            if mapped_winner == "real":
                real_votes += 1
            elif mapped_winner == "mixed":
                mixed_votes += 1
            else:
                tie_votes += 1

            for field in SCORE_FIELDS:
                real_scores[field].append(
                    float(record["real_scores"][field])
                )
                mixed_scores[field].append(
                    float(record["mixed_scores"][field])
                )

        if mixed_votes > real_votes:
            outcome = "mixed"
        elif real_votes > mixed_votes:
            outcome = "real"
        else:
            outcome = "tie"

        example_outcomes[outcome] += 1

        per_example.append(
            {
                "example_index": example_index,
                "real_votes": real_votes,
                "mixed_votes": mixed_votes,
                "tie_votes": tie_votes,
                "outcome": outcome,
            }
        )

    def means(
        values: dict[str, list[float]],
    ) -> dict[str, float]:
        return {
            field: (
                sum(field_values) / len(field_values)
                if field_values
                else 0.0
            )
            for field, field_values in values.items()
        }

    judged_examples = len(grouped)

    return {
        "expected_examples": example_count,
        "judged_examples": judged_examples,
        "judge_calls": len(records),
        "per_call_outcomes": dict(call_outcomes),
        "per_example_outcomes": dict(example_outcomes),
        "real_mean_scores": means(real_scores),
        "mixed_mean_scores": means(mixed_scores),
        "mixed_win_rate": (
            example_outcomes["mixed"] / judged_examples
            if judged_examples
            else 0.0
        ),
        "real_win_rate": (
            example_outcomes["real"] / judged_examples
            if judged_examples
            else 0.0
        ),
        "tie_rate": (
            example_outcomes["tie"] / judged_examples
            if judged_examples
            else 0.0
        ),
        "per_example": per_example,
    }


def main() -> None:
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        description=(
            "Blind pairwise LLM-as-a-judge evaluation of "
            "real-only versus mixed LoRA predictions."
        )
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/lora/val_real_sft.jsonl"),
    )
    parser.add_argument(
        "--real-predictions",
        type=Path,
        default=Path(
            "runs/qwen2.5_1.5b_lora_real_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--mixed-predictions",
        type=Path,
        default=Path(
            "runs/qwen2.5_1.5b_lora_mixed_predictions.jsonl"
        ),
    )
    parser.add_argument(
        "--details-output",
        type=Path,
        default=Path(
            "results/llm_judge_real_vs_mixed_details.jsonl"
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path(
            "results/llm_judge_real_vs_mixed_summary.json"
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=(
            os.getenv("JUDGE_MODEL")
            or os.getenv("SYNTHETIC_MODEL_NAME")
            or "deepseek/deepseek-chat-v3.1"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help=(
            "Judgements per example. With 2, candidate order is swapped "
            "to reduce position bias."
        ),
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="0 means all validation examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from an existing details JSONL file.",
    )
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv(
        "OPENAI_BASE_URL",
        "https://openrouter.ai/api/v1",
    )

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing from .env"
        )

    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")

    validation_rows = load_jsonl(args.validation)
    real_rows = load_jsonl(args.real_predictions)
    mixed_rows = load_jsonl(args.mixed_predictions)

    lengths = {
        len(validation_rows),
        len(real_rows),
        len(mixed_rows),
    }

    if len(lengths) != 1:
        raise ValueError(
            "Validation and prediction files have different lengths: "
            f"validation={len(validation_rows)}, "
            f"real={len(real_rows)}, mixed={len(mixed_rows)}"
        )

    total_examples = len(validation_rows)

    if args.max_examples > 0:
        total_examples = min(
            total_examples,
            args.max_examples,
        )

    completed_pairs: set[tuple[int, int]] = set()
    existing_records: list[dict[str, Any]] = []

    if args.resume and args.details_output.exists():
        existing_records = load_jsonl(
            args.details_output
        )

        for record in existing_records:
            completed_pairs.add(
                (
                    int(record["example_index"]),
                    int(record["repeat_index"]),
                )
            )
    else:
        args.details_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        args.details_output.write_text(
            "",
            encoding="utf-8",
        )

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    print("=" * 72)
    print("TASK 4 BLIND PAIRWISE LLM-AS-A-JUDGE")
    print("=" * 72)
    print(f"Judge model: {args.judge_model}")
    print(f"Real-only predictions: {args.real_predictions}")
    print(f"Mixed predictions: {args.mixed_predictions}")
    print(f"Real validation examples: {total_examples}")
    print(f"Judgements per example: {args.repeats}")
    print(
        f"Planned API calls: "
        f"{total_examples * args.repeats}"
    )
    print(
        "Synthetic examples used in evaluation: 0"
    )
    print()

    new_records: list[dict[str, Any]] = []

    for example_index in range(total_examples):
        (
            research_question,
            state_prompt,
            reference_target,
        ) = get_validation_fields(
            validation_rows[example_index]
        )

        real_prediction = get_prediction(
            real_rows[example_index]
        )
        mixed_prediction = get_prediction(
            mixed_rows[example_index]
        )

        rng = random.Random(
            args.seed + example_index
        )
        first_real_is_a = bool(
            rng.randint(0, 1)
        )

        for repeat_index in range(args.repeats):
            pair_key = (
                example_index + 1,
                repeat_index + 1,
            )

            if pair_key in completed_pairs:
                continue

            # Alternate order after the randomized first pass.
            real_is_a = (
                first_real_is_a
                if repeat_index % 2 == 0
                else not first_real_is_a
            )

            if real_is_a:
                candidate_a = real_prediction
                candidate_b = mixed_prediction
                label_a = "real"
                label_b = "mixed"
            else:
                candidate_a = mixed_prediction
                candidate_b = real_prediction
                label_a = "mixed"
                label_b = "real"

            prompt = build_judge_prompt(
                state_prompt=state_prompt,
                reference_target=reference_target,
                candidate_a=candidate_a,
                candidate_b=candidate_b,
            )

            judgement, raw_response = call_judge(
                client,
                model=args.judge_model,
                prompt=prompt,
                max_retries=args.max_retries,
            )

            if judgement["winner"] == "A":
                mapped_winner = label_a
            elif judgement["winner"] == "B":
                mapped_winner = label_b
            else:
                mapped_winner = "tie"

            scores_a = judgement["candidate_a"]
            scores_b = judgement["candidate_b"]

            real_scores = (
                scores_a if label_a == "real" else scores_b
            )
            mixed_scores = (
                scores_a if label_a == "mixed" else scores_b
            )

            record = {
                "example_index": example_index + 1,
                "repeat_index": repeat_index + 1,
                "research_question": research_question,
                "reference_target": reference_target,
                "real_prediction": real_prediction,
                "mixed_prediction": mixed_prediction,
                "candidate_a_source": label_a,
                "candidate_b_source": label_b,
                "judge_winner": judgement["winner"],
                "mapped_winner": mapped_winner,
                "real_scores": real_scores,
                "mixed_scores": mixed_scores,
                "real_score_average": score_average(
                    real_scores
                ),
                "mixed_score_average": score_average(
                    mixed_scores
                ),
                "reason": judgement["reason"],
                "raw_response": raw_response,
                "judge_model": args.judge_model,
            }

            append_jsonl(
                args.details_output,
                record,
            )
            new_records.append(record)

            print(
                f"[{example_index + 1:02d}/"
                f"{total_examples:02d}] "
                f"pass={repeat_index + 1}/"
                f"{args.repeats} | "
                f"winner={mapped_winner} | "
                f"real={record['real_score_average']:.2f} | "
                f"mixed={record['mixed_score_average']:.2f}"
            )

    all_records = existing_records + new_records
    summary = aggregate_results(
        all_records,
        total_examples,
    )

    summary.update(
        {
            "judge_model": args.judge_model,
            "validation_file": str(
                args.validation
            ),
            "real_predictions_file": str(
                args.real_predictions
            ),
            "mixed_predictions_file": str(
                args.mixed_predictions
            ),
            "repeats": args.repeats,
            "seed": args.seed,
            "synthetic_examples_in_evaluation": 0,
        }
    )

    args.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.summary_output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("=" * 72)
    print("LLM-AS-A-JUDGE FINISHED")
    print("=" * 72)
    print(
        f"Judged examples: "
        f"{summary['judged_examples']}"
    )
    print(
        f"Judge calls: "
        f"{summary['judge_calls']}"
    )
    print(
        f"Mixed wins: "
        f"{summary['per_example_outcomes'].get('mixed', 0)} "
        f"({summary['mixed_win_rate']:.2%})"
    )
    print(
        f"Real-only wins: "
        f"{summary['per_example_outcomes'].get('real', 0)} "
        f"({summary['real_win_rate']:.2%})"
    )
    print(
        f"Ties: "
        f"{summary['per_example_outcomes'].get('tie', 0)} "
        f"({summary['tie_rate']:.2%})"
    )
    print()
    print(
        "Real-only mean overall score: "
        f"{summary['real_mean_scores']['overall']:.3f}"
    )
    print(
        "Mixed mean overall score: "
        f"{summary['mixed_mean_scores']['overall']:.3f}"
    )
    print()
    print(f"Details: {args.details_output}")
    print(f"Summary: {args.summary_output}")


if __name__ == "__main__":
    main()
