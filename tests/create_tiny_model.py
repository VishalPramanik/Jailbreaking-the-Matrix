#!/usr/bin/env python3
# Copyright (c) 2025 Vishal Pramanik, University of Florida
# Licensed under the MIT License. See LICENSE for details.
"""
create_tiny_model.py — Generate a tiny GPT-2 model for offline testing.

Creates a 272K-parameter GPT-2 variant (4 layers, 4 heads, dim=64) with
a minimal BPE tokenizer. No internet access required after creation.

Usage:
    python tests/create_tiny_model.py
"""

import os
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers


def main():
    output_path = str(Path(__file__).resolve().parent / "tiny_gpt2")
    os.makedirs(output_path, exist_ok=True)

    # --- Model ---
    print("Creating tiny GPT-2 model...")
    config = GPT2Config(
        vocab_size=1000,
        n_positions=128,
        n_embd=64,
        n_layer=4,
        n_head=4,
        n_inner=256,
        bos_token_id=0,
        eos_token_id=1,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(output_path)

    # --- Tokenizer ---
    print("Creating minimal BPE tokenizer...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=1000,
        special_tokens=["<|endoftext|>", "<|pad|>"],
        min_frequency=1,
    )
    tokenizer.train_from_iterator(
        [
            "The quick brown fox jumps over the lazy dog. " * 100,
            "Hello world this is a test of the tokenizer. " * 100,
            "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z. " * 100,
            "The capital of France is Paris. The meaning of life is 42. " * 100,
        ],
        trainer=trainer,
    )
    tokenizer.save(os.path.join(output_path, "tokenizer.json"))

    fast_tok = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(output_path, "tokenizer.json"),
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    fast_tok.save_pretrained(output_path)

    # --- Verify ---
    encoded = fast_tok("Hello world", return_tensors="pt")
    with torch.no_grad():
        out = model(**encoded)

    print(f"\nModel saved to: {output_path}")
    print(f"  Layers: {config.n_layer}")
    print(f"  Heads:  {config.n_head}")
    print(f"  Dim:    {config.n_embd}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Vocab:  {fast_tok.vocab_size}")
    print(f"  Test forward shape: {out.logits.shape}")
    print("Done ✓")


if __name__ == "__main__":
    main()
