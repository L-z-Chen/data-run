# autoresearch

Autonomous LLM pretraining research, driven by AI agents.

The idea: give an AI agent a small but real LLM training setup and let it run experiments overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model.

This particular implementation is trying to be the least fancy baseline, but it's clear how one would adjust the `program.md` file to run more sophisticated research programs with more elaborate instructions. For example, the agent can actively do little experiments on research while the job is running.

The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat).

## How it works

The repo is deliberately small and only has a few files:

- **`constants.py`** — fixed rules: sequence length, time budget, eval tokens. Not modified.
- **`prepare.py`** — one-time data prep (downloads training data, trains a BPE tokenizer) and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc.
- **`program.md`** — instructions for the agent. Point your agent here and let it go.

Training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation). The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

## Quick start

**Requirements:** A single NVIDIA GPU (tested on H100), Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install dependencies
uv sync

# 2. Download data and train tokenizer (one-time, ~5 min)
uv run prepare.py

# 3. Run a single training experiment (5 min + startup)
uv run train.py
```

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo, then you can something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
constants.py    — fixed constants (do not modify)
prepare.py      — data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
spawn.sh        — multi-agent launcher
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes. This makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc).
- **BPB metric.** Bits per byte is independent of tokenizer vocabulary size, so the agent could in principle change the vocab size and still get a fair comparison.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## License

MIT
