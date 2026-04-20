# Embedding Pipeline

## Data

Default data directory: `data/processed`

Required files:

- `train.csv` or `train.csv.gz`
- `test.csv` or `test.csv.gz`

Current repo files:

- `data/processed/train.csv.gz`
- `data/processed/test.csv`

## What Changed

- The original embedding baseline used `clean_posts`.
- `posts_list_pool` is newly added in the current embedding pipeline.
- `posts_list_pool` parses `posts_list`, masks MBTI words, encodes each post separately, then mean-pools them into one user embedding.

## Current Ablations

1. Model ablation
   - `MiniLM` vs `BGE-base`
   - fixed input mode: `posts_list_pool`
   - fixed `max_posts=16`

2. `max_posts` ablation
   - fixed model: `MiniLM`
   - fixed input mode: `posts_list_pool`
   - compare `max_posts=16`, `32`, `64`

## Input Modes

- `clean_posts`: old baseline, encode the full cleaned user text as one string
- `posts_list_pool`: new input mode, encode post-level text and mean-pool

## Install

```bash
pip install -r requirements.txt
```

## Run Commands

### Baseline: MiniLM + `clean_posts`

```bash
python models/embedding.py \
  --model-name minilm \
  --input-mode clean_posts \
  --results-path results/minilm_clean_posts.json
```

### Model Ablation: `posts_list_pool`, `max_posts=16`

```bash
python models/embedding.py \
  --model-name minilm \
  --input-mode posts_list_pool \
  --results-path results/minilm_posts_pool.json
```

```bash
python models/embedding.py \
  --model-name bge-base \
  --input-mode posts_list_pool \
  --results-path results/bge_posts_pool.json
```

### `max_posts` Ablation: MiniLM + `posts_list_pool`

```bash
python models/embedding.py \
  --model-name minilm \
  --input-mode posts_list_pool \
  --results-path results/minilm_posts_pool.json
```

```bash
python models/embedding.py \
  --model-name minilm \
  --input-mode posts_list_pool \
  --max-posts 32 \
  --results-path results/minilm_posts_pool_max32.json
```

```bash
python models/embedding.py \
  --model-name minilm \
  --input-mode posts_list_pool \
  --max-posts 64 \
  --results-path results/minilm_posts_pool_max64.json
```

## Result Files

- `results/minilm_clean_posts.json`
- `results/minilm_posts_pool.json`
- `results/bge_posts_pool.json`
- `results/minilm_posts_pool_max32.json`
- `results/minilm_posts_pool_max64.json`

Each JSON includes:

- model and input configuration
- train/test statistics
- training time
- centroid metrics
- logistic-regression metrics
- full classification report

## Minimal `load + predict`

```python
from models.embedding import EmbeddingPipeline

pipeline = EmbeddingPipeline(model_name="minilm")
pipeline.load("saved_models/v1")

text = "I like spending time alone to think through systems and ideas."
print(pipeline.predict(text, method="lr"))

user_posts = [
    "I usually recharge by myself after long meetings.",
    "I like organizing ideas before I speak.",
]
print(pipeline.predict([user_posts], method="lr")[0])
```
