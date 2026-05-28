#!/usr/bin/env python3
# Copyright (c) 2025 Vishal Pramanik, University of Florida
# Licensed under the MIT License. See LICENSE for details.
"""
quick_start.py — Minimal Example for HMNS.

Demonstrates the core HMNS pipeline on GPT-2 (or any decoder-only LM).
This script shows how to:
  1. Load a model and tokenizer.
  2. Initialize the HMNS pipeline.
  3. Run the closed-loop intervention on a prompt.
  4. Inspect the results (completion, metrics, selected heads).
"""

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hmns import HMNSPipeline
from hmns.utils import setup_logging

setup_logging("INFO")


def main():
    # ----------------------------------------------------------------
    # Step 1: Load model and tokenizer
    # ----------------------------------------------------------------
    # Replace with target model for real experiments:
    #   "meta-llama/Llama-2-7b-chat-hf"
    #   "microsoft/Phi-3-medium-4k-instruct"
    #   "meta-llama/Llama-3.1-70B"
    #
    # For quick local testing, use the bundled tiny model:
    tiny_path = str(Path(__file__).resolve().parent.parent / "tests" / "tiny_gpt2")
    model_name = tiny_path if Path(tiny_path).exists() else "gpt2"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    # ----------------------------------------------------------------
    # Step 2: Initialize HMNS pipeline
    # ----------------------------------------------------------------
    pipeline = HMNSPipeline(
        model=model,
        tokenizer=tokenizer,
        top_k=4,              # Number of causal heads to select
        max_attempts=3,       # Closed-loop iterations (T_att)
        alpha=0.25,           # Initial steering coefficient
        alpha_schedule="linear",
        alpha_growth=0.1,
        scoring="kl",         # KL-divergence scoring (Eq. 4)
        proxy_preselect=False, # Disable proxy for small models
        max_new_tokens=50,
        temperature=0.7,
        top_p=0.95,
        seed=42,
    )

    # ----------------------------------------------------------------
    # Step 3: Run HMNS on a prompt
    # ----------------------------------------------------------------
    prompt = "The meaning of life is"
    print(f"\nPrompt: {prompt}")
    print("-" * 50)

    result = pipeline.run(prompt)

    # ----------------------------------------------------------------
    # Step 4: Inspect results
    # ----------------------------------------------------------------
    print(f"\n{'=' * 50}")
    print(f"  Completion:    {result.completion}")
    print(f"  Success:       {result.success}")
    print(f"  Attempt:       {result.attempt}")
    print(f"  ACQ:           {result.acq}")
    print(f"  IPC:           {result.ipc}")
    print(f"  Latency:       {result.latency_s:.2f}s")
    print(f"  Selected Heads: {result.selected_heads}")
    print(f"{'=' * 50}")

    print("\nAll completions:")
    for i, c in enumerate(result.all_completions):
        label = "baseline" if i == 0 else f"attempt {i}"
        print(f"  [{label}] {c[:80]}...")


if __name__ == "__main__":
    main()
