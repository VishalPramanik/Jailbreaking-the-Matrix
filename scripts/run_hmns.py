#!/usr/bin/env python3
# Copyright (c) 2025 Vishal Pramanik, University of Florida
# Licensed under the MIT License. See LICENSE for details.
"""
run_hmns.py — Main Entry Point for HMNS Experiments.

Runs the Head-Masked Nullspace Steering pipeline on a list of prompts
using a specified model and configuration.

Usage:
    python scripts/run_hmns.py \\
        --model meta-llama/Llama-2-7b-chat-hf \\
        --prompts prompts.txt \\
        --config configs/default.yaml \\
        --output results.json

    # Quick test with GPT-2:
    python scripts/run_hmns.py \\
        --model gpt2 \\
        --prompt "The meaning of life is" \\
        --config configs/test_gpt2.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hmns.pipeline import HMNSPipeline
from hmns.metrics import compute_metrics_from_result, aggregate_metrics
from hmns.utils import setup_logging, get_model_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="HMNS: Head-Masked Nullspace Steering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="HuggingFace model name or local path.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt string (for quick testing).",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="Path to a text file with one prompt per line.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results.json",
        help="Path for output JSON results.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Override: number of causal heads to select.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="Override: maximum closed-loop iterations.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override: initial steering coefficient.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (e.g., 'cuda', 'cpu'). Auto-detected if not set.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger("hmns.run")

    # --- Load config ---
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = load_config(
            str(Path(__file__).resolve().parent.parent / "configs" / "default.yaml")
        )

    # --- Apply CLI overrides ---
    model_name = args.model or cfg.get("model", {}).get("name", "gpt2")
    top_k = args.top_k or cfg.get("attribution", {}).get("top_k", 10)
    max_attempts = args.max_attempts or cfg.get("loop", {}).get("max_attempts", 10)
    alpha = args.alpha or cfg.get("intervention", {}).get("alpha", 0.25)
    seed = args.seed or cfg.get("seed", 42)

    # --- Load model and tokenizer ---
    logger.info(f"Loading model: {model_name}")
    dtype_str = cfg.get("model", {}).get("dtype", "float32")
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype_str, torch.float32)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.to("cpu")

    model.eval()
    logger.info(f"Model loaded on {device} ({torch_dtype})")

    # --- Collect prompts ---
    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts:
        with open(args.prompts, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default demo prompt
        prompts = ["The quick brown fox jumps over the"]

    logger.info(f"Processing {len(prompts)} prompt(s)")

    # --- Build pipeline ---
    pipeline = HMNSPipeline(
        model=model,
        tokenizer=tokenizer,
        top_k=top_k,
        max_attempts=max_attempts,
        alpha=alpha,
        alpha_schedule=cfg.get("intervention", {}).get("alpha_schedule", "linear"),
        alpha_growth=cfg.get("intervention", {}).get("alpha_growth", 0.1),
        scoring=cfg.get("attribution", {}).get("scoring", "kl"),
        proxy_preselect=cfg.get("attribution", {}).get("proxy_preselect", True),
        proxy_k=cfg.get("attribution", {}).get("proxy_k", 30),
        ortho_tol=cfg.get("nullspace", {}).get("ortho_tol", 1e-6),
        max_resample=cfg.get("nullspace", {}).get("max_resample", 3),
        temperature=cfg.get("decoding", {}).get("temperature", 0.7),
        top_p=cfg.get("decoding", {}).get("top_p", 0.95),
        max_new_tokens=cfg.get("decoding", {}).get("max_new_tokens", 128),
        seed=seed,
    )

    # --- Run ---
    results = pipeline.run_batch(prompts, verbose=True)

    # --- Compute metrics ---
    model_cfg = get_model_config(model)
    all_metrics = []
    for r in results:
        m = compute_metrics_from_result(r, model_cfg)
        all_metrics.append(m)

    agg = aggregate_metrics(all_metrics)

    # --- Output ---
    output_data = {
        "model": model_name,
        "config": cfg,
        "aggregate_metrics": agg,
        "results": [
            {
                "prompt": r.prompt,
                "completion": r.completion,
                "success": r.success,
                "attempt": r.attempt,
                "acq": r.acq,
                "ipc": r.ipc,
                "latency_s": r.latency_s,
                "all_completions": r.all_completions,
            }
            for r in results
        ],
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    logger.info(f"Results saved to {args.output}")

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("HMNS Results Summary")
    print("=" * 60)
    print(f"  Model:           {model_name}")
    print(f"  Prompts:         {agg.get('total_prompts', 0)}")
    print(f"  Successes:       {agg.get('successful_prompts', 0)}")
    print(f"  ASR:             {agg.get('asr', 0):.1f}%")
    print(f"  Mean ACQ:        {agg.get('mean_acq', 0):.2f}")
    print(f"  Mean IPC:        {agg.get('mean_ipc', 0):.1f}")
    print(f"  Mean FPS (T):    {agg.get('mean_fps', 0):.4f}")
    print(f"  Mean LPS (s):    {agg.get('mean_lps', 0):.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
