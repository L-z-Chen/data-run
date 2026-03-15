# Data-Run

> **How few training tokens can you use to reach a target validation loss?**

A data efficiency benchmark for language model training. Same eval as [slowrun](https://github.com/qlabs-eng/slowrun) — but instead of fixed data and unlimited compute, it's fixed compute and unlimited data strategy. The competition is about **data quality**: curate, distill, or synthesize the best training data to hit the target loss in the fewest tokens.

## The competition

- **Eval data**: [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) validation set (identical to [slowrun](https://github.com/qlabs-eng/slowrun), hash-verified)
- **Metric**: Validation loss (mean cross-entropy = log PPL)
- **Target**: **val_loss <= 3.2**
- **Score**: **Number of unique training tokens** used to reach the target (lower is better)
- **Loss function**: Standard cross-entropy, same as [slowrun](https://github.com/qlabs-eng/slowrun)/[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)/[nanogpt](https://github.com/karpathy/nanoGPT)/[nanochat](https://github.com/karpathy/nanochat)

> **Note:** Training for multiple epochs does NOT increase your token count. If your dataset has 1M tokens and you train for 100 epochs, your score is still 1M. The score is purely the size of your training data.

### What you CAN do

- Use **any training data**: the eval data itself, [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), third-party datasets, distilled/synthetic data
- Modify `train.py`: model architecture, optimizer, hyperparameters, batch size, model size
- Use any sequence length during training (the eval sequence length is fixed at 2048)
- Train for as many epochs as you want (only unique tokens count toward your score)
- Bring your own training data in `.pt` format (see Data Format below)

### What is FIXED

- `prepare.py`: evaluation function, tokenizer ([GPT-2](https://huggingface.co/openai-community/gpt2)), eval sequence length (2048), val data
- The evaluation metric and target: **val_loss <= 3.2**

## Quick start

**Requirements:** A GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync

# 2. Download FineWeb data and prepare (one-time)
uv run prepare.py

# 3. Run baseline training
uv run train.py
```

## Project structure

```
prepare.py      — data prep + fixed evaluation (do not modify)
train.py        — model, optimizer, training loop (modify freely)
data/           — training and validation data (.pt files)
```

## Leaderboard

**Target: val_loss <= 3.2.** Score = unique training tokens (lower is better).

| # | Tokens | Description | Date | Script | Contributors |
|---|--------|-------------|------|--------|--------------|
| 1 | 5.9M | Subsample 2900 seqs from eval data, 304M params, 16 layers, Muon+AdamW | 03/15/26 | [Script](https://github.com/L-z-Chen/data-run/blob/892f5f2/train.py) | [@L-z-Chen](https://github.com/L-z-Chen) |
| 2 | 6.0M | Subsample 2950 seqs from eval data, 162M params, 12 layers, Muon+AdamW | 03/15/26 | [Script](https://github.com/L-z-Chen/data-run/blob/892f5f2/train.py) | [@L-z-Chen](https://github.com/L-z-Chen) |
| 3 | 6.1M | Subsample 3000 seqs from eval data, 162M params, 12 layers, Muon+AdamW | 03/15/26 | [Script](https://github.com/L-z-Chen/data-run/blob/892f5f2/train.py) | [@L-z-Chen](https://github.com/L-z-Chen) |
| 4 | 8.2M | Subsample 4000 seqs from eval data, 162M params, 12 layers, Muon+AdamW | 03/15/26 | [Script](https://github.com/L-z-Chen/data-run/blob/892f5f2/train.py) | [@L-z-Chen](https://github.com/L-z-Chen) |
| 5 | 10.0M | Train on full eval data (4880 seqs), 162M params, 12 layers, Muon+AdamW | 03/15/26 | [Script](https://github.com/L-z-Chen/data-run/blob/892f5f2/train.py) | [@L-z-Chen](https://github.com/L-z-Chen) |

## Scoring

Your score is the **number of unique tokens in your training dataset** to reach **val_loss <= 3.2**. Lower is better. Training for multiple epochs is free — only the dataset size counts.

## Custom training data

The default training data is 100M tokens from [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (same as [slowrun](https://github.com/qlabs-eng/slowrun)). To use custom data:

1. Create a `.pt` file in the standard format (see below)
2. In `train.py`, point the dataloader to your file:
   ```python
   train_loader = make_dataloader("path/to/your_data.pt", DEVICE_BATCH_SIZE)
   ```

You can also train directly on the eval data:
```python
train_loader = make_dataloader("val", DEVICE_BATCH_SIZE)
```

## Data format

The `.pt` files store pre-tokenized sequences using the [GPT-2](https://huggingface.co/openai-community/gpt2) tokenizer (vocab size 50,257):

```python
{
    'chunks': List[Tensor],     # each: (batch_size * sequence_size,) uint16 tokens
    'valid_counts': List[int],  # valid sequences per chunk
    'batch_size': int,          # sequences per chunk (default 16)
    'sequence_size': int,       # tokens per sequence (2049 = 2048 input + 1 target)
}
```

Each sequence is 2049 tokens. The dataloader splits each into input `[:2048]` and target `[1:]`.

## License

MIT
