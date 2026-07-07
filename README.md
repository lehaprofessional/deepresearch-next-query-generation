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
