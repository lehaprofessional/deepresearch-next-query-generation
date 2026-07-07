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
