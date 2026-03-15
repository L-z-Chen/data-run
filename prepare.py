"""
Data preparation and evaluation for data-run.

Downloads FineWeb data (same eval set as slowrun benchmark), provides
dataloader and fixed evaluation function.

Usage:
    python prepare.py                              # default: 100M train, 10M val
    python prepare.py --train_tokens 50_000_000    # custom train size

Validation data is identical to slowrun (hash-verified).
"""

import os
import math
import hashlib
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048
SEQUENCE_SIZE = MAX_SEQ_LEN + 1
VOCAB_SIZE = 50257  # GPT-2
TARGET_VAL_LOSS = 3.2

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

EXPECTED_HASHES = {
    "fineweb_val.pt": "80e7e430d3a7d10892c2ff32579370c5b65fbe833579d7ea5d55cbd0504c8462",
    "fineweb_train.pt": "e7e089aedbccb6865ce76c78453fa473c823969846fccd4000f5f13aef54e70e",
}

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _tokenize_documents(dataset_iter, encoder, total_tokens):
    """Tokenize documents until we have total_tokens tokens."""
    eot = encoder._special_tokens['<|endoftext|>']
    tokens = []
    pbar = tqdm(total=total_tokens, unit="tok")
    for doc in dataset_iter:
        doc_tokens = [eot] + encoder.encode_ordinary(doc["text"])
        tokens.extend(doc_tokens)
        pbar.update(len(doc_tokens))
        if len(tokens) >= total_tokens:
            tokens = tokens[:total_tokens]
            break
    pbar.close()
    return tokens


def _create_sequences(tokens):
    """Split flat token list into fixed-size sequences."""
    tokens = np.array(tokens, dtype=np.uint16)
    n = len(tokens) // SEQUENCE_SIZE
    return tokens[:n * SEQUENCE_SIZE].reshape(n, SEQUENCE_SIZE)


def _write_datafile(filename, sequences, batch_size=16):
    """Write sequences to chunked .pt file (slowrun-compatible format)."""
    seq_size = sequences.shape[1]
    n = len(sequences)
    full = n // batch_size
    leftover = n % batch_size
    chunks, valid_counts = [], []
    for i in range(full):
        chunk = sequences[i * batch_size:(i + 1) * batch_size].reshape(-1)
        chunks.append(torch.from_numpy(chunk.copy()))
        valid_counts.append(batch_size)
    if leftover > 0:
        pad = np.zeros((batch_size - leftover, seq_size), dtype=np.uint16)
        padded = np.concatenate([sequences[full * batch_size:], pad]).reshape(-1)
        chunks.append(torch.from_numpy(padded.copy()))
        valid_counts.append(leftover)
    torch.save({'chunks': chunks, 'valid_counts': valid_counts,
                'batch_size': batch_size, 'sequence_size': seq_size}, filename)
    print(f"  Wrote {len(chunks)} chunks ({n} sequences) to {filename}")


def _sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_hash(filepath):
    basename = os.path.basename(filepath)
    actual = _sha256(filepath)
    expected = EXPECTED_HASHES.get(basename)
    if expected and actual == expected:
        print(f"  Hash OK: {basename}")
        return True
    elif expected:
        print(f"  HASH MISMATCH: {basename}")
        print(f"    expected: {expected}")
        print(f"    actual:   {actual}")
        return False
    print(f"  Hash: {basename} = {actual}")
    return True


def prepare_data(train_tokens=100_000_000, val_tokens=10_000_000):
    """Download FineWeb and prepare train/val splits (identical to slowrun)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    encoder = tiktoken.get_encoding("gpt2")
    val_path = os.path.join(DATA_DIR, "fineweb_val.pt")
    train_path = os.path.join(DATA_DIR, "fineweb_train.pt")

    if os.path.exists(val_path) and os.path.exists(train_path):
        print("Data already prepared.")
        _verify_hash(val_path)
        _verify_hash(train_path)
        return

    print(f"Streaming FineWeb (val: {val_tokens:,}, train: {train_tokens:,} tokens)...")
    dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                           split="train", streaming=True)
    it = iter(dataset)

    # Val first (same order as slowrun for hash compatibility)
    print("Tokenizing val...")
    val_seqs = _create_sequences(_tokenize_documents(it, encoder, val_tokens))
    np.random.seed(42)
    np.random.shuffle(val_seqs)
    _write_datafile(val_path, val_seqs)
    _verify_hash(val_path)

    # Train (continues from same iterator — no document overlap with val)
    print("Tokenizing train...")
    train_seqs = _create_sequences(_tokenize_documents(it, encoder, train_tokens))
    np.random.seed(43)
    np.random.shuffle(train_seqs)
    _write_datafile(train_path, train_seqs)
    _verify_hash(train_path)

    print("Done!")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

def load_sequences(path):
    """Load a .pt data file -> (N, SEQUENCE_SIZE) long tensor."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    bs = data['batch_size']
    ss = data['sequence_size']
    parts = []
    for chunk, valid in zip(data['chunks'], data['valid_counts']):
        parts.append(chunk.reshape(bs, ss)[:valid])
    return torch.cat(parts, dim=0).long()


def make_dataloader(source, batch_size):
    """
    Infinite dataloader -> (inputs, targets, epoch) on GPU.
    source: "train", "val", or path to a .pt file.
    """
    if source in ("train", "val"):
        fname = "fineweb_val.pt" if source == "val" else "fineweb_train.pt"
        source = os.path.join(DATA_DIR, fname)
    sequences = load_sequences(source)
    n = len(sequences)
    epoch = 1
    while True:
        perm = torch.randperm(n)
        for i in range(0, n - batch_size + 1, batch_size):
            batch = sequences[perm[i:i + batch_size]].cuda()
            yield batch[:, :-1], batch[:, 1:], epoch
        epoch += 1


@torch.no_grad()
def evaluate_val_loss(model, batch_size):
    """
    Mean cross-entropy loss on the fixed validation set (= log PPL).
    This is the sole data-run metric. DO NOT MODIFY.
    """
    sequences = load_sequences(os.path.join(DATA_DIR, "fineweb_val.pt"))
    n = len(sequences)
    total_loss = 0.0
    total_tokens = 0
    was_training = model.training
    model.eval()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i in range(0, n - batch_size + 1, batch_size):
            batch = sequences[i:i + batch_size].cuda()
            x, y = batch[:, :-1], batch[:, 1:]
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.reshape(-1),
                                   reduction='sum')
            total_loss += loss.item()
            total_tokens += y.numel()
    if was_training:
        model.train()
    return total_loss / total_tokens

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare FineWeb data for data-run")
    parser.add_argument("--train_tokens", type=int, default=100_000_000)
    parser.add_argument("--val_tokens", type=int, default=10_000_000)
    args = parser.parse_args()
    prepare_data(args.train_tokens, args.val_tokens)
