from pathlib import Path


README_PATH = Path("README.md")
REPORTS_DIR = Path("reports")
RESULTS_DIR = Path("results")

FINAL_REPORT_PATH = REPORTS_DIR / "task4_final_report.md"
SUMMARY_PATH = RESULTS_DIR / "final_summary.md"


README_TEXT = r'''
# DeepResearch Task 4: Query Generation

Промежуточная исследовательская работа по задаче 4 летней школы Северсталь.

## Цель задачи

Цель задачи — улучшить генерацию поисковых запросов для DeepResearch-агента.

После получения непустого `train_summerschool_task3.csv` задача была сведена к постановке:

research_question + visited_context + previous_queries -> next_query

Где:

- `research_question` — исходный исследовательский вопрос;
- `visited_context` — описания уже посещённых источников из `task3`;
- `previous_queries` — ранее сгенерированные запросы;
- `next_query` — следующий поисковый запрос.

## Основные этапы

1. Подготовка baseline-датасета из `train_summerschool_task4.csv`.
2. Разделение данных на train/validation по `research_question`.
3. Реализация простой эвристической baseline-модели.
4. Реализация LLM baseline.
5. Prompt engineering без контекста `task3`.
6. Получение и подключение `task3`.
7. Сборка датасета с `visited_context`.
8. Тестирование context baseline.
9. Тестирование retrieval few-shot поверх context.
10. Тестирование улучшенного context prompt v2.
11. Error analysis.
12. Экспериментальный hybrid router.

## Метрики

Использовались лексические метрики:

- Exact Match;
- Token F1;
- Jaccard;
- Repeated previous query;
- средняя длина сгенерированного запроса.

Важно: лексические метрики не полностью отражают реальное качество поискового запроса, потому что разные формулировки могут приводить к одинаково полезной поисковой выдаче.

## Результаты

| Method | Token F1 | Jaccard | Avg generated length | Комментарий |
|---|---:|---:|---:|---|
| Simple baseline | 0.1875 | 0.1147 | 10.50 | Простая эвристика |
| LLM baseline v1 | 0.2566 | 0.1558 | 13.19 | Первый LLM baseline |
| Prompt v2 | 0.2406 | 0.1459 | 9.28 | Более строгий prompt без контекста |
| Few-shot prompt v3 | 0.2543 | 0.1562 | 11.19 | Few-shot без task3 |
| Retrieval few-shot v4 | 0.2683 | 0.1649 | 9.66 | Лучший question-only baseline |
| Context v1 | 0.2757 | 0.1715 | 11.12 | Добавлен visited_context из task3 |
| Context retrieval few-shot v2 | 0.2665 | 0.1657 | 10.09 | Гипотеза не подтвердилась |
| Context prompt v2 | **0.2784** | **0.1753** | 12.59 | Лучший честный результат |
| Hybrid router | **0.2998** | **0.1917** | 11.09 | Exploratory upper-bound |

## Основной вывод

Добавление контекста посещённых источников из `task3` улучшило качество генерации поисковых запросов.

Лучший question-only baseline:

- Token F1 = 0.2683
- Jaccard = 0.1649

Лучший честный context-result:

- Token F1 = 0.2784
- Jaccard = 0.1753

Лучшим честным результатом является `llm_context_prompt_v2`.

Экспериментальный `hybrid_router` дал более высокий результат:

- Token F1 = 0.2998
- Jaccard = 0.1917

Однако этот результат следует рассматривать как exploratory upper-bound, так как правило выбора стратегии было сформулировано после анализа validation-результатов.

## Как воспроизвести

Установить зависимости:

    python -m pip install -r requirements.txt

Создать `.env` по примеру `.env.example`.

Подготовить датасет с context:

    python .\prepare_task4_with_task3_context.py
    python .\build_task4_context_prompts.py
    python .\build_task4_context_prompts_v2.py

Запустить генерацию для лучшего честного результата:

    $env:INPUT_PATH="data/task4_val_context_v2.jsonl"
    $env:OUTPUT_PATH="runs/llm_context_prompt_v2_predictions.jsonl"
    $env:MAX_EXAMPLES="0"
    python .\generate_llm_predictions.py

Оценить результат:

    python .\evaluate_predictions.py

Собрать таблицу сравнения:

    python .\compare_runs.py
    python .\save_results_table.py

## Примечание про модель

Текущий pipeline не привязан к конкретному провайдеру модели. Генерация реализована через OpenAI-compatible интерфейс и может быть запущена на локальной или внутренней разрешённой модели при изменении переменных окружения.

Внешний inference использовался как прототипирование для проверки пайплайна, prompt-стратегий и метрик.
'''


FINAL_REPORT_TEXT = r'''
# Task 4: Генерация поисковых запросов для DeepResearch

## 1. Цель работы

Целью работы было улучшить генерацию поисковых запросов для DeepResearch-агента. Агент решает сложные исследовательские вопросы пошагово: посещает источники, анализирует их и затем формирует новые поисковые запросы для продолжения исследования.

После получения непустого `train_summerschool_task3.csv` была реализована постановка:

research_question + visited_context + previous_queries -> next_query

Где `visited_context` строится по связке:

task4.task3_ids -> task3.id -> link + description

## 2. Подготовка данных

Были использованы файлы:

- `train_summerschool_task4.csv`;
- `train_summerschool_task3.csv`.

Из `task4` были получены:

- исследовательские вопросы;
- целевые поисковые запросы;
- номера запросов;
- списки `task3_ids`.

Из `task3` были получены:

- ссылки на посещённые источники;
- описания источников;
- идентификаторы источников.

После объединения данных был сформирован датасет из 280 примеров. Разделение на train/validation выполнялось по `research_question`, чтобы избежать утечки между train и validation.

Итоговое разделение:

- Train examples: 248
- Validation examples: 32
- Unique research questions: 17

## 3. Метрики

Использовались лексические метрики:

- Exact Match;
- Token F1;
- Jaccard;
- Repeated previous query;
- средняя длина сгенерированного запроса.

Exact Match оказался слишком строгим для задачи генерации поисковых запросов, так как один и тот же поисковый смысл может быть выражен разными формулировками.

Token F1 и Jaccard лучше отражают частичное совпадение терминов с эталонными запросами, однако они также не полностью описывают реальную полезность запроса. Разные поисковые запросы могут иметь низкое лексическое совпадение, но приводить к релевантной поисковой выдаче.

Дальнейшим направлением является добавление search-based evaluation или LLM-as-a-judge.

## 4. Эксперименты

### 4.1 Simple baseline

Была реализована простая эвристическая baseline-модель, которая формировала запрос на основе ключевых слов из исследовательского вопроса.

Результат:

- Token F1 = 0.1875
- Jaccard = 0.1147

### 4.2 LLM baseline без task3

Первый LLM baseline использовал только:

research_question + previous_queries -> next_query

Результат:

- Token F1 = 0.2566
- Jaccard = 0.1558

Это заметно лучше простой эвристики.

### 4.3 Prompt engineering без task3

Были протестированы несколько prompt-стратегий:

- более строгий prompt v2;
- few-shot prompt v3;
- retrieval few-shot v4.

Лучшим вариантом без `task3` стал retrieval few-shot:

- Token F1 = 0.2683
- Jaccard = 0.1649

### 4.4 Context baseline с task3

После получения `task3` был собран датасет с `visited_context`.

Первый context baseline дал:

- Token F1 = 0.2757
- Jaccard = 0.1715

Это улучшило лучший question-only baseline:

- Token F1: 0.2683 -> 0.2757
- Jaccard: 0.1649 -> 0.1715

Таким образом, контекст посещённых источников действительно оказался полезен.

### 4.5 Context retrieval few-shot v2

Была проверена гипотеза, что retrieval few-shot можно объединить с `visited_context`.

Для каждого validation-примера подбирались релевантные train-примеры, после чего они добавлялись в prompt вместе с контекстом посещённых источников.

Результат:

- Token F1 = 0.2665
- Jaccard = 0.1657

Гипотеза не подтвердилась: результат оказался хуже обычного context baseline.

Вероятная причина — few-shot-примеры начали смещать модель к повторяющимся шаблонам вроде `lance tip`, `nozzle angle`, `skull prevention`, `high scrap ratio`. Из-за этого модель хуже выбирала другие аспекты поиска, например `bottom stirring`, `argon flow rate`, `FeO slag basicity`, `secondary oxygen flow`.

### 4.6 Context prompt v2

После неудачного retrieval few-shot был реализован более строгий prompt без few-shot-примеров. В prompt явно указывалось, что модель должна использовать конкретные технические термины из `visited_context`:

- материалы;
- механизмы;
- параметры процесса;
- численные условия;
- химические компоненты;
- промышленные практики;
- типы дефектов;
- trials/patents/products.

Результат:

- Token F1 = 0.2784
- Jaccard = 0.1753

Это лучший честный результат среди протестированных стратегий.

### 4.7 Error analysis

Анализ ошибок показал, что разные стратегии работают по-разному для разных типов исследовательских вопросов.

| Research question group | context_v1 | context_retrieval_fewshot_v2 | context_prompt_v2 |
|---|---:|---:|---:|
| RH snorkel | 0.3357 | 0.2211 | 0.2912 |
| BOF | 0.2412 | 0.2634 | 0.2303 |
| Ladle furnace slag line | 0.3175 | 0.2787 | 0.3483 |

Вывод:

- для RH snorkel лучше работал `context_v1`;
- для BOF лучше работал `context_retrieval_fewshot_v2`;
- для ladle furnace slag line лучше работал `context_prompt_v2`.

### 4.8 Hybrid router

На основе error analysis был протестирован простой router:

- RH snorkel -> context_v1
- BOF -> context_retrieval_fewshot_v2
- Ladle furnace slag line -> context_prompt_v2

Результат:

- Token F1 = 0.2998
- Jaccard = 0.1917

Это самый высокий результат среди всех экспериментов.

Однако его следует рассматривать как exploratory upper-bound, так как правило выбора стратегии было сформулировано после анализа validation-результатов. Основным честным результатом остаётся `context_prompt_v2`.

## 5. Сводная таблица

| Method | Token F1 | Jaccard | Avg generated length | Комментарий |
|---|---:|---:|---:|---|
| Simple baseline | 0.1875 | 0.1147 | 10.50 | Простая эвристика |
| LLM baseline v1 | 0.2566 | 0.1558 | 13.19 | Первый LLM baseline |
| Prompt v2 | 0.2406 | 0.1459 | 9.28 | Строгий prompt без context |
| Few-shot prompt v3 | 0.2543 | 0.1562 | 11.19 | Few-shot без task3 |
| Retrieval few-shot v4 | 0.2683 | 0.1649 | 9.66 | Лучший question-only baseline |
| Context v1 | 0.2757 | 0.1715 | 11.12 | Добавлен visited_context |
| Context retrieval few-shot v2 | 0.2665 | 0.1657 | 10.09 | Гипотеза не подтвердилась |
| Context prompt v2 | **0.2784** | **0.1753** | 12.59 | Лучший честный результат |
| Hybrid router | **0.2998** | **0.1917** | 11.09 | Exploratory upper-bound |

## 6. Основной вывод

Был реализован pipeline генерации поисковых запросов для DeepResearch-агента.

Добавление контекста посещённых источников из `task3` улучшило качество генерации:

- question-only best: Token F1 = 0.2683, Jaccard = 0.1649
- context prompt v2: Token F1 = 0.2784, Jaccard = 0.1753

Лучшим честным результатом является `context_prompt_v2`.

Также был проведён error analysis, который показал, что разные prompt-стратегии лучше работают для разных типов исследовательских вопросов. На основе этого был реализован экспериментальный `hybrid_router`, который дал:

- Token F1 = 0.2998
- Jaccard = 0.1917

Но этот результат следует трактовать как exploratory upper-bound, а не как полностью unbiased validation score.

## 7. Дальнейшие шаги

Дальнейшая работа может включать:

1. Запуск pipeline на разрешённой локальной или внутренней модели.
2. Добавление search-based evaluation.
3. Добавление LLM-as-a-judge.
4. Улучшение router-стратегии без использования validation leakage.
5. Отдельные prompt-стратегии для разных доменов вопросов.
6. Возможная LoRA/QLoRA только после стабилизации датасета и baseline.
'''


SUMMARY_TEXT = r'''
# Final summary

## Best fair result

llm_context_prompt_v2

- Token F1 = 0.2784
- Jaccard = 0.1753

This is the best fair validation result among the tested prompt strategies.

## Exploratory upper-bound result

llm_context_hybrid_router

- Token F1 = 0.2998
- Jaccard = 0.1917

This result should be treated as exploratory upper-bound because the routing rule was derived from validation error analysis.

## Main conclusion

Adding `visited_context` from `train_summerschool_task3.csv` improves query generation compared with the best question-only baseline.

Question-only best:

- Token F1 = 0.2683
- Jaccard = 0.1649

Context prompt v2:

- Token F1 = 0.2784
- Jaccard = 0.1753

Error analysis showed that different strategies work better for different research-question types:

- RH snorkel -> context_v1
- BOF -> context_retrieval_fewshot_v2
- Ladle furnace slag line -> context_prompt_v2
'''


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    README_PATH.write_text(README_TEXT.strip() + "\n", encoding="utf-8")
    FINAL_REPORT_PATH.write_text(FINAL_REPORT_TEXT.strip() + "\n", encoding="utf-8")
    SUMMARY_PATH.write_text(SUMMARY_TEXT.strip() + "\n", encoding="utf-8")

    print("Saved:", README_PATH)
    print("Saved:", FINAL_REPORT_PATH)
    print("Saved:", SUMMARY_PATH)


if __name__ == "__main__":
    main()