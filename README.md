# NanoGPT Data-Run

> **How few training tokens can you use to reach a target validation loss?**

A data efficiency benchmark for language model training. Same eval as [slowrun](https://github.com/qlabs-eng/slowrun) — but instead of fixed data and unlimited compute, it's fixed compute and unlimited data strategy. The competition is about **data quality**: curate, distill, or synthesize the best training data to hit the target loss in the fewest tokens.

## The competition

- **Eval data**: FineWeb validation set (identical to slowrun, hash-verified)
- **Metric**: Validation loss (mean cross-entropy = log PPL)
- **Target**: val_loss <= 3.2
- **Score**: Total training tokens consumed to reach target (lower is better)
- **Loss function**: Standard cross-entropy, same as slowrun/nanochat

### What you CAN do

- Use **any training data**: the eval data itself, FineWeb, third-party datasets, distilled/synthetic data
- Modify `train.py`: model architecture, optimizer, hyperparameters, batch size, model size
- Use any sequence length during training (the eval sequence length is fixed at 2048)
- Bring your own training data in `.pt` format (see Data Format below)

### What is FIXED

- `prepare.py`: evaluation function, tokenizer (GPT-2), eval sequence length (2048), val data
- The evaluation metric and target loss

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

## Scoring

Your score is the total number of training tokens consumed to reach val_loss <= 3.2. Lower is better. If the target is not reached within MAX_STEPS, the run is unscored (DNF).

## Custom training data

The default training data is 100M tokens from FineWeb (same as slowrun). To use custom data:

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

The `.pt` files store pre-tokenized sequences using the GPT-2 tokenizer (vocab size 50,257):

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
