# DeepResearch Next-Query Generation Pipeline

A configurable training, inference, and evaluation pipeline for generating the **next web-search query** in a DeepResearch workflow.

The model receives:

- a research question;
- previously generated search queries;
- descriptions of already visited sources;

and produces one concise technical query that should advance the research process.

The implementation is designed around a replaceable language-model backbone. The same data preparation, LoRA training, inference, and evaluation logic can be reused with Qwen, Llama, Mistral, or another compatible instruction-tuned causal language model.

---

## Table of contents

1. [Task overview](#task-overview)
2. [Pipeline](#pipeline)
3. [Repository structure](#repository-structure)
4. [Environment setup](#environment-setup)
5. [Data format](#data-format)
6. [Preparing the real dataset](#preparing-the-real-dataset)
7. [Generating synthetic training data](#generating-synthetic-training-data)
8. [Training a LoRA adapter](#training-a-lora-adapter)
9. [Running inference](#running-inference)
10. [Evaluation](#evaluation)
11. [Experimental results](#experimental-results)
12. [Changing the backbone](#changing-the-backbone)
13. [Reproducibility](#reproducibility)
14. [Known limitations](#known-limitations)
15. [Troubleshooting](#troubleshooting)

---

## Task overview

The task is formulated as supervised next-query generation.

### Input

```text
Research question
+
Previous search queries
+
Descriptions of visited sources
```

### Output

```text
One concise technical search query
```

Example:

```text
Research question:
How does oxygen injection through the top lance affect refractory wear
in RH vacuum-degasser snorkels?

Previous queries:
- RH snorkel MgO-C refractory wear mechanisms
- oxygen lance RH degasser slag chemistry

Visited-source descriptions:
- Paper about oxidation and decarburization of MgO-C refractories
- Study of slag penetration under RH operating conditions

Target next query:
oxygen lance effect on MgO-C decarburization and slag penetration in RH snorkels
```

The desired query should be:

- relevant to the original research question;
- novel relative to the previous queries;
- grounded in the available context;
- technically specific;
- concise and suitable for a search engine.

---

## Pipeline

```text
Task 3 source descriptions
        +
Task 4 research states
        в†“
Context construction
        в†“
Real SFT dataset
        в†“
Optional synthetic augmentation
        в†“
Configurable LoRA training
        в†“
Configurable inference
        в†“
Token F1 / Jaccard / BERTScore / LLM-as-a-judge
```

The final evaluation is performed only on **real validation examples**. Synthetic examples are used only as additional training data.

---

## Repository structure

The repository contains both the final configurable pipeline and earlier experimental scripts.

### Core pipeline

| File | Purpose |
|---|---|
| `train_lora_pipeline.py` | Main configurable LoRA training entry point. Loads a backbone, applies LoRA, trains on a JSONL SFT dataset, evaluates on a real validation split, uses early stopping, and saves the best adapter and metrics. |
| `generate_predictions_pipeline.py` | Main configurable inference entry point. Loads a backbone with or without a LoRA adapter and generates predictions from a JSON configuration. |
| `evaluate_predictions.py` | Computes exact match, Token F1, Jaccard, repeated-query rate, and query-length statistics. The prediction path can be supplied through `PREDICTIONS_PATH`. |
| `evaluate_bertscore_single.py` | Computes BERTScore Precision, Recall, and F1 for one prediction file. |
| `evaluate_llm_judge_pairwise.py` | Runs blind pairwise LLM-as-a-judge evaluation between real-only and mixed-data models. Candidate order is swapped to reduce position bias. |
| `configs/` | JSON configurations for training and inference with different backbones, adapters, datasets, and output paths. |

### Data preparation

| File | Purpose |
|---|---|
| `prepare_task4_with_task3_context.py` | Joins Task 4 examples with Task 3 source descriptions through `task3_ids`. |
| `build_task4_context_prompts.py` | Builds an initial context-aware prompt representation. |
| `build_task4_context_prompts_v2.py` | Improved context-prompt construction used in later experiments. |
| `prepare_lora_sft_dataset.py` | Converts the prepared examples into chat-style SFT JSONL files for training and validation. |
| `data/lora/train_real_sft.jsonl` | Real-only training split: 248 examples. |
| `data/lora/val_real_sft.jsonl` | Real-only validation split: 32 examples. |
| `data/lora/train_mixed_pilot_sft.jsonl` | Mixed training split: 248 real + 100 synthetic examples. |

### Synthetic-data pipeline

| File | Purpose |
|---|---|
| `generate_synthetic_train_pilot.py` | Generates alternative next-query examples with an external teacher model, modifies the source context, filters candidates, and builds a mixed training dataset. |
| `clean_synthetic_queries.py` | Normalizes malformed teacher outputs such as `["query"]`, validates query lengths, and rebuilds the mixed dataset. |
| `analyze_synthetic_dataset.py` | Audits duplicates, real-target overlap, query length, topic distribution, and random qualitative samples. |
| `data/synthetic/task4_synthetic_pilot.jsonl` | Accepted synthetic examples. |
| `data/synthetic/task4_synthetic_pilot_audit.jsonl` | Accepted and rejected candidates with filtering reasons and quality metadata. |
| `results/synthetic_dataset_audit.json` | Summary of the final synthetic dataset. |
| `results/synthetic_pilot_summary.json` | Generation counts, rejection reasons, language distribution, and output paths. |

### Earlier experiments and diagnostic scripts

| File | Purpose |
|---|---|
| `generate_llm_predictions.py` | Generates predictions for prompt-based LLM experiments. |
| `compare_runs.py` | Compares multiple prediction runs. |
| `save_results_table.py` | Writes an aggregated experiment table. |
| `analyze_context_v2_errors.py` | Performs topic-level and qualitative error analysis for context-based prompts. |
| `make_hybrid_predictions.py` | Builds an exploratory topic router from several prompt strategies. This is an upper-bound experiment, not a clean held-out result. |
| `smoke_train_lora.py` | Minimal GPU and LoRA training smoke test. |
| `train_lora_real.py` | Original real-only LoRA experiment before the configurable training pipeline was introduced. |
| `generate_qwen_lora_predictions.py` | Original deterministic Qwen base/LoRA inference script used to establish the first reproducible results. |
| `generate_qwen_lora_mixed_predictions.py` | Temporary mixed-adapter variant of the original generator. The configurable inference pipeline supersedes it. |
| `evaluate_bertscore.py` | Earlier multi-run BERTScore evaluation script. |

### Outputs

| Directory | Contents |
|---|---|
| `models/` | Locally saved LoRA adapters and checkpoints. Usually excluded from Git because of size. |
| `runs/` | Generated prediction JSONL files. |
| `results/` | Training metrics, BERTScore results, synthetic-data audits, and LLM-judge outputs. |
| `reports/` | Experiment reports and result summaries. |

---

## Environment setup

### Tested environment

The experiments were run with:

```text
OS: Windows 10
GPU: NVIDIA GeForce RTX 5070 Ti, 16 GB VRAM
RAM: 32 GB
Python: virtual environment
PyTorch: 2.12.1+cu130
Transformers: 5.13.1
Datasets: 5.0.0
Accelerate: 1.14.0
PEFT: 0.19.1
TRL: 1.8.0
```

Other compatible versions may work, but exact reproducibility is best with the tested stack.

### 1. Create and activate a virtual environment

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Upgrade packaging tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install PyTorch

Install a PyTorch build compatible with your GPU and CUDA environment. Refer to the official PyTorch installation selector for the appropriate command.

Verify CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected on a CUDA machine:

```text
True
NVIDIA GeForce ...
```

### 4. Install project dependencies

```powershell
pip install `
  transformers `
  datasets `
  accelerate `
  peft `
  trl `
  sentencepiece `
  safetensors `
  bert-score `
  python-dotenv `
  openai `
  requests `
  numpy
```

### 5. Optional Hugging Face authentication

Public models can be downloaded without authentication, but Hugging Face applies lower rate limits to unauthenticated requests.

Set a token for the current PowerShell session:

```powershell
$env:HF_TOKEN="YOUR_HUGGINGFACE_TOKEN"
```

Never commit tokens to Git.

---

## Data format

Training and validation files use JSON Lines format. Each line contains a chat example:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You generate concise technical next-search queries."
    },
    {
      "role": "user",
      "content": "Research question: ...\nVisited sources context: ...\nPrevious generated queries: ..."
    },
    {
      "role": "assistant",
      "content": "target next search query"
    }
  ],
  "metadata": {
    "research_question": "...",
    "query_id": 1,
    "source": "real"
  }
}
```

For training, the system and user tokens are masked. Loss is computed only on the assistant target query.

### Current split

The split is grouped by research question to reduce leakage.

| Split | Examples | Unique research questions | Synthetic examples |
|---|---:|---:|---:|
| Real train | 248 | 14 | 0 |
| Mixed train | 348 | 14 | 100 |
| Real validation | 32 | 3 | 0 |

The final evaluation never uses synthetic examples.

---

## Preparing the real dataset

The repository already contains the prepared train and validation files. To rebuild them from the task data, run the preparation stages in order.

### 1. Join Task 4 with Task 3 descriptions

```powershell
python .\prepare_task4_with_task3_context.py
```

This stage joins Task 4 states with Task 3 rows through the stored `task3_ids`.

### 2. Build context-aware prompts

```powershell
python .\build_task4_context_prompts_v2.py
```

The prompt includes:

- the original research question;
- descriptions of previously visited sources;
- previously generated queries;
- the target next query.

### 3. Build SFT train and validation files

```powershell
python .\prepare_lora_sft_dataset.py
```

Expected outputs:

```text
data/lora/train_real_sft.jsonl
data/lora/val_real_sft.jsonl
```

The current prepared token statistics fit within a maximum sequence length of 2048:

```text
Train maximum: 2028 tokens
Validation maximum: 1344 tokens
```

---

## Generating synthetic training data

Synthetic data is optional. It is intended to expand the training set, not the validation set.

### Teacher API configuration

Create a local `.env` file:

```env
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=YOUR_OPENROUTER_API_KEY

MODEL_NAME=deepseek/deepseek-chat-v3.1
SYNTHETIC_MODEL_NAME=deepseek/deepseek-chat-v3.1
JUDGE_MODEL=deepseek/deepseek-chat-v3.1
```

Do not commit `.env`.

The API endpoint is OpenAI-compatible, so another compatible provider can be substituted.

### Synthetic-generation logic

For each selected real training example, the script:

1. keeps the real research question;
2. keeps the previous-query history;
3. randomly retains part of the visited-source context;
4. asks the teacher model for several alternative next queries;
5. does not expose the real target query to the teacher;
6. filters candidates;
7. selects the highest-scoring valid candidate;
8. stores accepted and rejected candidates in an audit file.

Candidate filtering includes:

- 5вЂ“20 token length;
- no duplicate synthetic query;
- no exact copy of the real target;
- limited similarity to the real target;
- limited similarity to previous queries;
- no unsupported numerical values;
- lexical grounding in the supplied context;
- preservation of the required output language;
- technical-anchor preservation.

The original real target is used only for post-generation filtering and is not shown to the teacher model.

### Run a small pilot

Keep the working API network route enabled:

```powershell
$env:TARGET_SYNTHETIC_EXAMPLES="10"
$env:MAX_SOURCE_EXAMPLES="30"

python .\generate_synthetic_train_pilot.py
```

### Audit the pilot

```powershell
python .\analyze_synthetic_dataset.py
```

Inspect the random examples and rejection statistics before scaling up.

### Generate 100 synthetic examples

```powershell
$env:TARGET_SYNTHETIC_EXAMPLES="100"
$env:MAX_SOURCE_EXAMPLES="180"

python .\generate_synthetic_train_pilot.py
```

### Normalize accepted examples and rebuild the mixed train set

```powershell
python .\clean_synthetic_queries.py
python .\analyze_synthetic_dataset.py
```

Expected mixed size:

```text
248 real + 100 synthetic = 348 examples
```

### Current synthetic audit

| Check | Result |
|---|---:|
| Accepted synthetic examples | 100 |
| Exact duplicate synthetic queries | 0 |
| Exact overlaps with real targets | 0 |
| Queries outside 5вЂ“20 tokens | 0 |
| Mean query length | 9.71 tokens |
| Median query length | 10 tokens |
| Synthetic share in mixed train | 28.74% |

The current generator preserves the language of the parent real target. The reported mixed-model experiment used the previously generated and manually audited 100-example synthetic set.

---

## Training a LoRA adapter

### Default mixed-data experiment

Configuration:

```text
configs/qwen2.5_1.5b_mixed.json
```

Run:

```powershell
python .\train_lora_pipeline.py `
  --config .\configs\qwen2.5_1.5b_mixed.json
```

### What the training pipeline does

`train_lora_pipeline.py`:

1. reads the JSON experiment configuration;
2. loads the tokenizer and backbone;
3. applies the model chat template;
4. masks all non-assistant tokens;
5. adds LoRA modules;
6. trains on the selected JSONL file;
7. evaluates only on `data/lora/val_real_sft.jsonl`;
8. saves one checkpoint per epoch;
9. selects the checkpoint with the lowest validation loss;
10. stops early when validation loss no longer improves;
11. saves the adapter, tokenizer, configuration, and metrics.

### Mixed-data configuration summary

```text
Backbone: Qwen/Qwen2.5-1.5B-Instruct
Train: data/lora/train_mixed_pilot_sft.jsonl
Validation: data/lora/val_real_sft.jsonl
Maximum sequence length: 2048
Precision: BF16
LoRA rank: 16
LoRA alpha: 32
LoRA dropout: 0.05
Learning rate: 1e-4
Micro-batch size: 1
Gradient accumulation: 8
Effective batch size: 8
Requested epochs: 5
Early-stopping patience: 2
Seed: 42
```

Target modules:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Expected output:

```text
models/qwen2.5-1.5b-task4-lora-mixed/
results/qwen2.5_1.5b_lora_mixed_metrics.json
```

### Training configuration example

```json
{
  "experiment_name": "qwen2.5-1.5b-task4-mixed",
  "model": {
    "name": "Qwen/Qwen2.5-1.5B-Instruct",
    "max_length": 2048,
    "dtype": "bfloat16"
  },
  "data": {
    "train_path": "data/lora/train_mixed_pilot_sft.jsonl",
    "validation_path": "data/lora/val_real_sft.jsonl"
  },
  "lora": {
    "rank": 16,
    "alpha": 32,
    "dropout": 0.05,
    "target_modules": [
      "q_proj",
      "k_proj",
      "v_proj",
      "o_proj",
      "gate_proj",
      "up_proj",
      "down_proj"
    ]
  },
  "training": {
    "output_dir": "models/qwen2.5-1.5b-task4-lora-mixed",
    "num_train_epochs": 5,
    "learning_rate": 0.0001,
    "train_batch_size": 1,
    "eval_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "early_stopping_patience": 2,
    "seed": 42
  }
}
```

---

## Running inference

The recommended entry point is:

```text
generate_predictions_pipeline.py
```

It supports:

- a base model without LoRA;
- a model with a LoRA adapter;
- configurable model name;
- configurable input and output paths;
- configurable dtype and attention implementation;
- deterministic or sampling-based generation;
- arbitrary compatible instruction-tuned causal-language-model backbones.

### Mixed LoRA inference

```powershell
python .\generate_predictions_pipeline.py `
  --config .\configs\qwen2.5_1.5b_mixed_inference.json
```

Expected output:

```text
runs/qwen2.5_1.5b_lora_mixed_repro_predictions.jsonl
```

### Base-model inference

```powershell
python .\generate_predictions_pipeline.py `
  --config .\configs\qwen2.5_1.5b_base_inference.json
```

### Real-only LoRA inference

```powershell
python .\generate_predictions_pipeline.py `
  --config .\configs\qwen2.5_1.5b_real_inference.json
```

### Smoke test on a subset

```powershell
python .\generate_predictions_pipeline.py `
  --config .\configs\qwen2.5_1.5b_mixed_inference.json `
  --max-examples 3
```

### Inference configuration example

```json
{
  "run_name": "qwen2.5_1.5b_lora_mixed_repro",
  "model": {
    "name": "Qwen/Qwen2.5-1.5B-Instruct",
    "adapter_path": "models/qwen2.5-1.5b-task4-lora-mixed",
    "dtype": "bfloat16",
    "device_map": "cuda",
    "attn_implementation": "sdpa"
  },
  "data": {
    "input_path": "data/lora/val_real_sft.jsonl",
    "output_path": "runs/qwen2.5_1.5b_lora_mixed_repro_predictions.jsonl"
  },
  "generation": {
    "max_new_tokens": 64,
    "do_sample": false,
    "repetition_penalty": 1.05,
    "max_input_tokens": null
  }
}
```

The current configurable inference implementation exactly reproduces the original deterministic mixed-model generator:

```text
Old examples: 32
Reproduced examples: 32
Mismatches: 0
```

---

## Evaluation

All final metrics are computed on:

```text
data/lora/val_real_sft.jsonl
```

This file contains 32 real examples from three held-out research questions.

### 1. Token metrics and length statistics

PowerShell:

```powershell
$env:PREDICTIONS_PATH="runs/qwen2.5_1.5b_lora_mixed_predictions.jsonl"
python .\evaluate_predictions.py
```

Metrics:

- exact match;
- Token F1;
- Jaccard similarity;
- repeated previous-query rate;
- average generated length;
- average target length.

### 2. BERTScore

```powershell
python .\evaluate_bertscore_single.py `
  --predictions .\runs\qwen2.5_1.5b_lora_mixed_predictions.jsonl `
  --output .\results\qwen2.5_1.5b_lora_mixed_bertscore.json
```

Default encoder:

```text
roberta-large
```

When comparing runs, keep the same BERTScore encoder and options.

The RoBERTa loading report may mention unused language-model-head parameters and a missing pooler. These warnings are expected for this encoder-based scoring use case.

### 3. Blind pairwise LLM-as-a-judge

The judge compares real-only and mixed-model outputs on the same real validation examples.

```powershell
python .\evaluate_llm_judge_pairwise.py `
  --repeats 2
```

The script:

1. anonymizes predictions as Candidate A and Candidate B;
2. evaluates each example twice;
3. swaps the A/B order on the second pass;
4. uses a different model family as the judge;
5. computes per-criterion scores and pairwise win rates.

Judge criteria:

- relevance;
- novelty;
- groundedness;
- technical specificity;
- search quality;
- overall quality.

Outputs:

```text
results/llm_judge_real_vs_mixed_details.jsonl
results/llm_judge_real_vs_mixed_summary.json
```

A short pilot can be run first:

```powershell
python .\evaluate_llm_judge_pairwise.py `
  --max-examples 3 `
  --repeats 2 `
  --details-output .\results\llm_judge_pilot_details.jsonl `
  --summary-output .\results\llm_judge_pilot_summary.json
```

---

## Experimental results

### Prompt-based baselines

| Run | Token F1 | Jaccard | BERTScore F1 |
|---|---:|---:|---:|
| Simple baseline | 0.1875 | 0.1147 | 0.8471 |
| LLM v1 | 0.2566 | 0.1558 | 0.8595 |
| Retrieval few-shot | 0.2683 | 0.1649 | 0.8603 |
| Context v1 | 0.2757 | 0.1715 | 0.8609 |
| Context prompt v2 | **0.2784** | **0.1753** | 0.8606 |

The exploratory `hybrid_router` reached Token F1 `0.2998`, Jaccard `0.1917`, and BERTScore F1 `0.8638`, but its routing rules were derived from validation error analysis. It should be treated as an upper-bound diagnostic rather than a fair held-out result.

### Qwen and LoRA experiments

| Model | Training data | Token F1 | Jaccard | BERTScore F1 | Average query length |
|---|---|---:|---:|---:|---:|
| Base Qwen 1.5B | None | 0.2092 | 0.1256 | 0.8433 | 17.03 |
| Real-only LoRA | 248 real | 0.2199 | 0.1346 | 0.8522 | 11.41 |
| Mixed LoRA | 248 real + 100 synthetic | **0.2296** | **0.1422** | **0.8552** | 13.03 |

Relative mixed-vs-real-only changes:

```text
Token F1:      +4.4%
Jaccard:       +5.6%
BERTScore F1:  +0.35%
```

### Mixed LoRA training behavior

| Epoch | Validation loss |
|---:|---:|
| 1 | 3.408 |
| 2 | **3.317** |
| 3 | 3.419 |
| 4 | 3.595 |

The best checkpoint was selected at epoch 2. The later epochs show overfitting.

### LLM-as-a-judge results

Evaluation:

```text
32 real validation examples
2 swapped-order judgements per example
64 judge calls
0 synthetic evaluation examples
```

Pairwise outcomes:

| Outcome | Count | Share |
|---|---:|---:|
| Real-only wins | 13 | 40.62% |
| Mixed wins | 12 | 37.50% |
| Ties | 7 | 21.88% |

Mean criterion scores:

| Criterion | Real-only | Mixed | Mixed delta |
|---|---:|---:|---:|
| Relevance | 3.406 | 3.375 | -0.031 |
| Novelty | 2.359 | 2.516 | **+0.156** |
| Groundedness | 3.781 | 3.391 | **-0.391** |
| Technical specificity | 3.062 | 3.312 | **+0.250** |
| Search quality | 3.516 | 3.281 | **-0.234** |
| Overall | 3.078 | 3.109 | +0.031 |

### Interpretation

Synthetic augmentation produced a small and consistent improvement in Token F1, Jaccard, and BERTScore on real validation examples.

However, the LLM judge did not identify a decisive pairwise winner. Mixed training improved novelty and technical specificity, but reduced groundedness and search quality.

Qualitative errors include:

- unsupported numerical parameters;
- long numerical enumerations;
- confusion between BOF and blast-furnace terminology;
- malformed quotation marks;
- overly narrow queries;
- context drift toward adjacent technical topics.

The synthetic set is therefore useful as moderate training augmentation, but it should be scaled only after improving grounding filters.

---

## Changing the backbone

The final solution is configuration-driven.

To train another backbone:

1. change `model.name`;
2. verify the tokenizer has a valid chat template;
3. configure architecture-compatible LoRA target modules;
4. choose a suitable sequence length and dtype;
5. use a new adapter output directory;
6. train a new adapter for that backbone.

Example:

```json
{
  "model": {
    "name": "Qwen/Qwen2.5-7B-Instruct",
    "max_length": 2048,
    "dtype": "bfloat16"
  },
  "training": {
    "output_dir": "models/qwen2.5-7b-task4-lora-mixed"
  }
}
```

Important:

> A LoRA adapter is architecture-specific. An adapter trained for Qwen 1.5B cannot be attached to Qwen 7B, Llama, or Mistral.

For another model family, update the LoRA target modules according to that architecture.

Examples of common projection-module names:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

Do not assume that every model exposes the same module names.

---

## Reproducibility

The project uses the following controls:

- grouped train/validation split by research question;
- no synthetic examples in validation;
- fixed seed `42`;
- deterministic inference with `do_sample=false`;
- fixed `repetition_penalty=1.05`;
- saved JSON configurations;
- saved prediction files;
- saved metric files;
- early stopping based on validation loss;
- exact comparison of the original and configurable inference outputs.

Verified inference reproduction:

```text
Old examples: 32
Reproduced examples: 32
Mismatches: 0
```

### Important ablation limitation

The real-only LoRA model was trained before the final configurable training pipeline was introduced.

It used:

- the same backbone;
- the same main LoRA hyperparameters;
- the same real training data;
- the same real validation split;

but a previous training-script implementation.

Therefore, the real-only vs. mixed-data comparison should be treated as a **preliminary ablation**, not as a perfectly controlled retraining experiment.

The inference procedure itself was later made configurable and verified to reproduce the original deterministic outputs exactly.

---

## Known limitations

1. **Small validation set**  
   The final validation set contains only 32 examples from three research questions.

2. **Limited topic coverage**  
   Topic-level conclusions may not generalize to all DeepResearch domains.

3. **Synthetic grounding**  
   Some generated training examples encourage unsupported numerical specificity.

4. **Teacher-model dependency**  
   Synthetic-data quality depends on the external teacher model and its provider.

5. **LLM-judge sensitivity**  
   Some decisions changed when Candidate A and Candidate B were swapped.

6. **Adapter portability**  
   LoRA adapters cannot be reused across incompatible model architectures or sizes.

7. **Model-size limitation**  
   Qwen2.5-1.5B is used as a low-cost test backbone. The pipeline is intended to be reproduced with larger backbones.

---

## Troubleshooting

### OpenRouter returns `403 Access denied by security policy`

This usually indicates a provider, account, IP, VPN, or network-security restriction rather than an invalid prompt.

Check the current key:

```powershell
python -c "from dotenv import load_dotenv; load_dotenv(override=True); import os,requests; r=requests.get('https://openrouter.ai/api/v1/key',headers={'Authorization':'Bearer '+os.environ['OPENAI_API_KEY']}); print(r.status_code); print(r.text)"
```

A working key and route should return:

```text
200
```

Possible actions:

- use the network route on which the API is accessible;
- check OpenRouter account and privacy settings;
- create a new API key;
- confirm that the selected model is available;
- do not retry a permanent `403` in a tight loop.

### Hugging Face unauthenticated warning

```text
Warning: You are sending unauthenticated requests to the HF Hub
```

This is not a training failure. Add `HF_TOKEN` for higher rate limits.

### CUDA is unavailable

Check:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Reinstall the correct CUDA-enabled PyTorch build if necessary.

### Out-of-memory error

Reduce one or more of:

```json
{
  "model": {
    "max_length": 1536
  },
  "training": {
    "train_batch_size": 1,
    "gradient_accumulation_steps": 8
  }
}
```

Other options:

- enable gradient checkpointing;
- use a smaller backbone;
- use quantized training if added to the pipeline;
- reduce evaluation batch size.

### Output directory is not empty

The training pipeline protects existing adapters by default.

Use a new output directory or explicitly enable overwrite in the configuration:

```json
{
  "training": {
    "overwrite_output_dir": true
  }
}
```

### BERTScore prints missing or unexpected RoBERTa keys

This is expected when loading the encoder from a checkpoint containing a language-model head. The metric can still be computed.

### Generated queries contain unsupported numbers

This is a known model-quality issue rather than an inference crash.

Recommended improvements:

- add numeric-grounding checks during synthetic generation;
- reject long numerical enumerations;
- add a post-generation validator;
- train with more grounded examples;
- include hard negatives with incorrect process terminology.

---

## Recommended end-to-end run order

For a clean reproduction:

```powershell
# 1. Activate the environment
.\.venv\Scripts\Activate.ps1

# 2. Prepare the real SFT data
python .\prepare_task4_with_task3_context.py
python .\build_task4_context_prompts_v2.py
python .\prepare_lora_sft_dataset.py

# 3. Optional: generate and audit synthetic training examples
$env:TARGET_SYNTHETIC_EXAMPLES="100"
$env:MAX_SOURCE_EXAMPLES="180"
python .\generate_synthetic_train_pilot.py
python .\clean_synthetic_queries.py
python .\analyze_synthetic_dataset.py

# 4. Train the mixed LoRA adapter
python .\train_lora_pipeline.py `
  --config .\configs\qwen2.5_1.5b_mixed.json

# 5. Run deterministic inference
python .\generate_predictions_pipeline.py `
  --config .\configs\qwen2.5_1.5b_mixed_inference.json

# 6. Compute lexical metrics
$env:PREDICTIONS_PATH="runs/qwen2.5_1.5b_lora_mixed_predictions.jsonl"
python .\evaluate_predictions.py

# 7. Compute BERTScore
python .\evaluate_bertscore_single.py `
  --predictions .\runs\qwen2.5_1.5b_lora_mixed_predictions.jsonl `
  --output .\results\qwen2.5_1.5b_lora_mixed_bertscore.json

# 8. Run pairwise LLM-as-a-judge
python .\evaluate_llm_judge_pairwise.py `
  --repeats 2
```

---

## Summary

The repository provides a complete configurable workflow for:

```text
context preparation
в†’ real and synthetic SFT construction
в†’ LoRA training
в†’ deterministic inference
в†’ lexical, semantic, and LLM-based evaluation
```

The current Qwen2.5-1.5B experiments demonstrate that synthetic augmentation can improve automatic metrics, while also showing that stronger grounding controls are required. The main contribution is the reproducible pipeline itself: a larger compatible backbone can be substituted through configuration without rewriting the full training and inference code.

