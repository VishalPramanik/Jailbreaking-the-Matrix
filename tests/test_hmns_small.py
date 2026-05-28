#!/usr/bin/env python3
# Copyright (c) 2025 Vishal Pramanik, University of Florida
# Licensed under the MIT License. See LICENSE for details.
"""
test_hmns_small.py — Smoke Test for HMNS Pipeline.

Verifies that the full HMNS pipeline (attribution → masking → nullspace
steering → generation) runs without errors on a tiny GPT-2 model.

This test validates the *mechanics* of HMNS (hook registration, column
masking, QR factorization, orthogonality checks, and closed-loop iteration)
rather than jailbreak effectiveness.

Usage:
    python tests/test_hmns_small.py
"""

import os
import sys
import logging
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hmns.utils import (
    setup_logging,
    set_seed,
    get_model_config,
    get_attention_layers,
    get_out_projection,
    rms_norm,
    compute_kl_divergence,
)
from hmns.attribution import CausalHeadAttributor
from hmns.masking import HeadMasker
from hmns.nullspace import NullspaceSteering
from hmns.intervention import HMNSIntervention
from hmns.pipeline import HMNSPipeline


setup_logging("INFO")
logger = logging.getLogger("test")


# ==========================================================================
# Fixtures
# ==========================================================================

TINY_MODEL_PATH = str(Path(__file__).resolve().parent / "tiny_gpt2")


def load_model():
    """Load the tiny GPT-2 model for testing (no internet required)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading tiny GPT-2 from {TINY_MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(TINY_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL_PATH, torch_dtype=torch.float32)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    cfg = model.config
    logger.info(
        f"  Loaded: {cfg.n_layer} layers, {cfg.n_head} heads, "
        f"dim={cfg.n_embd}, vocab={cfg.vocab_size}"
    )
    return model, tokenizer


# ==========================================================================
# Unit Tests
# ==========================================================================

def test_model_config(model):
    """Test model configuration extraction."""
    logger.info("--- test_model_config ---")
    cfg = get_model_config(model)
    assert cfg["num_layers"] == 4, f"Expected 4 layers, got {cfg['num_layers']}"
    assert cfg["num_heads"] == 4, f"Expected 4 heads, got {cfg['num_heads']}"
    assert cfg["hidden_dim"] == 64, f"Expected 64 hidden_dim, got {cfg['hidden_dim']}"
    assert cfg["head_dim"] == 16, f"Expected 16 head_dim, got {cfg['head_dim']}"
    logger.info(f"  Config: {cfg}")
    logger.info("  PASSED ✓")


def test_attention_layers(model):
    """Test attention layer extraction."""
    logger.info("--- test_attention_layers ---")
    layers = get_attention_layers(model)
    assert len(layers) == 4, f"Expected 4 layers, got {len(layers)}"
    for idx, mod in layers:
        wo = get_out_projection(mod)
        assert wo.weight.shape == (64, 64), (
            f"Layer {idx}: unexpected W^O shape {wo.weight.shape}"
        )
    logger.info(f"  Found {len(layers)} attention layers with correct W^O shapes")
    logger.info("  PASSED ✓")


def test_rms_and_kl():
    """Test utility functions."""
    logger.info("--- test_rms_and_kl ---")

    # RMS norm
    a = torch.randn(64)
    r = rms_norm(a)
    assert r > 0, "RMS should be positive"
    logger.info(f"  RMS({a.shape}) = {r:.4f}")

    # KL divergence
    p = torch.softmax(torch.randn(100), dim=-1)
    q = torch.softmax(torch.randn(100), dim=-1)
    kl = compute_kl_divergence(p, q)
    assert kl >= 0, "KL divergence should be non-negative"
    logger.info(f"  KL(p || q) = {kl:.4f}")

    # KL with self should be ~0
    kl_self = compute_kl_divergence(p, p)
    assert kl_self < 1e-5, f"KL(p || p) should be ~0, got {kl_self:.6f}"
    logger.info(f"  KL(p || p) = {kl_self:.6f} (≈ 0)")
    logger.info("  PASSED ✓")


def test_attribution(model, tokenizer):
    """Test causal head attribution (Eq. 3-4)."""
    logger.info("--- test_attribution ---")
    attributor = CausalHeadAttributor(
        model=model,
        tokenizer=tokenizer,
        top_k=3,
        scoring="kl",
        proxy_preselect=False,
    )
    input_ids = tokenizer("Hello world", return_tensors="pt")["input_ids"]
    heads, scores = attributor.attribute(input_ids)

    assert len(heads) == 3, f"Expected 3 heads, got {len(heads)}"
    assert all(isinstance(h, tuple) and len(h) == 2 for h in heads)
    assert all(v >= 0 for v in scores.values()), "KL scores should be non-negative"
    logger.info(f"  Top-3 heads: {heads}")
    logger.info(f"  Top score: {max(scores.values()):.4f}")
    logger.info("  PASSED ✓")


def test_masking(model, tokenizer):
    """Test head masking and write-matrix extraction (Eq. 3, 5)."""
    logger.info("--- test_masking ---")
    masker = HeadMasker(model)

    # Test write matrix extraction (Eq. 5)
    M = masker.get_write_matrix(0, [0, 1])
    d_h = 16
    expected_cols = 2 * d_h
    assert M.shape == (64, expected_cols), f"Expected (64, {expected_cols}), got {M.shape}"
    logger.info(f"  M_0 shape: {M.shape} (2 heads × {d_h} head_dim)")

    # Test context-managed masking
    wo = get_out_projection(get_attention_layers(model)[0][1])
    original_col = wo.weight.data[:, :d_h].clone()

    with masker.mask_heads([(0, 0)]) as layer_heads:
        assert torch.all(wo.weight.data[:, :d_h] == 0), (
            "Column should be zeroed inside mask context"
        )
        logger.info("  Inside mask context: columns zeroed ✓")

    assert torch.allclose(wo.weight.data[:, :d_h], original_col), (
        "Column should be restored after mask context"
    )
    logger.info("  Outside mask context: columns restored ✓")
    logger.info("  PASSED ✓")


def test_nullspace(model):
    """Test nullspace direction computation (Eq. 5-7)."""
    logger.info("--- test_nullspace ---")
    masker = HeadMasker(model)
    steerer = NullspaceSteering(masker, ortho_tol=1e-6, max_resample=3)

    # Compute direction for layer 0, heads [0, 1]
    u = steerer.compute_nullspace_direction(
        layer_idx=0, head_indices=[0, 1], device=torch.device("cpu")
    )
    assert u is not None, "Nullspace direction should not be None"
    assert u.shape == (64,), f"Expected (64,), got {u.shape}"

    # Verify unit norm
    norm = u.norm().item()
    assert abs(norm - 1.0) < 1e-4, f"Expected unit norm, got {norm:.6f}"
    logger.info(f"  ‖u‖ = {norm:.6f} (≈ 1.0)")

    # Verify orthogonality: ‖M^T u‖_∞ < δ  (Theorem 2)
    M = masker.get_write_matrix(0, [0, 1])
    orth_error = (M.T @ u.float()).abs().max().item()
    assert orth_error < 1e-6, f"Orthogonality violated: ‖M^T u‖_∞ = {orth_error:.2e}"
    logger.info(f"  ‖M^T u‖_∞ = {orth_error:.2e} (< δ = 1e-6)")
    logger.info("  PASSED ✓")


def test_perturbation(model):
    """Test perturbation scaling (Eq. 8)."""
    logger.info("--- test_perturbation ---")
    masker = HeadMasker(model)
    steerer = NullspaceSteering(masker)

    u = steerer.compute_nullspace_direction(0, [0, 1], torch.device("cpu"))
    activation = torch.randn(64)
    delta = steerer.compute_perturbation(u, activation, alpha=0.25)

    expected_scale = 0.25 * rms_norm(activation).item()
    actual_scale = delta.norm().item()
    assert abs(actual_scale - expected_scale) < 1e-3, (
        f"Expected δ scale ≈ {expected_scale:.4f}, got {actual_scale:.4f}"
    )
    logger.info(f"  α·RMS(a)·‖u‖ = {expected_scale:.4f}, ‖δ‖ = {actual_scale:.4f}")
    logger.info("  PASSED ✓")


def test_intervention(model, tokenizer):
    """Test the full intervention context manager."""
    logger.info("--- test_intervention ---")
    masker = HeadMasker(model)
    steerer = NullspaceSteering(masker)
    intervention = HMNSIntervention(model, masker, steerer, alpha=0.25)

    input_ids = tokenizer("Hello", return_tensors="pt")["input_ids"]
    heads = [(0, 0), (1, 2), (3, 1)]

    # Generate without intervention
    with torch.no_grad():
        out_baseline = model.generate(
            input_ids, max_new_tokens=10, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    baseline_text = tokenizer.decode(out_baseline[0], skip_special_tokens=True)

    # Generate with intervention
    with torch.no_grad():
        with intervention.intervene(heads, step=1):
            out_steered = model.generate(
                input_ids, max_new_tokens=10, do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    steered_text = tokenizer.decode(out_steered[0], skip_special_tokens=True)

    logger.info(f"  Baseline:  {repr(baseline_text[:60])}")
    logger.info(f"  Steered:   {repr(steered_text[:60])}")
    logger.info(f"  Outputs differ: {baseline_text != steered_text}")
    logger.info("  PASSED ✓")


def test_pipeline(model, tokenizer):
    """Test the full HMNS pipeline (Algorithm 2)."""
    logger.info("--- test_pipeline (full closed loop) ---")
    pipeline = HMNSPipeline(
        model=model,
        tokenizer=tokenizer,
        top_k=3,
        max_attempts=2,
        alpha=0.25,
        scoring="kl",
        proxy_preselect=False,
        temperature=0.7,
        top_p=0.95,
        max_new_tokens=15,
        seed=42,
    )

    result = pipeline.run("Hello world")

    logger.info(f"  Prompt:       {result.prompt}")
    logger.info(f"  Completion:   {repr(result.completion[:60])}")
    logger.info(f"  Success:      {result.success}")
    logger.info(f"  Attempt:      {result.attempt}")
    logger.info(f"  ACQ:          {result.acq}")
    logger.info(f"  IPC:          {result.ipc}")
    logger.info(f"  Latency:      {result.latency_s:.2f}s")
    logger.info(f"  Completions:  {len(result.all_completions)}")

    assert result.completion is not None, "Completion should not be None"
    assert result.acq >= 1, "ACQ should be at least 1"
    assert result.ipc >= 1, "IPC should be at least 1"
    assert result.latency_s > 0, "Latency should be positive"
    logger.info("  PASSED ✓")


# ==========================================================================
# Main
# ==========================================================================

def main():
    set_seed(42)

    if not os.path.exists(TINY_MODEL_PATH):
        print(f"ERROR: Tiny model not found at {TINY_MODEL_PATH}")
        print("Please run: python tests/create_tiny_model.py")
        sys.exit(1)

    model, tokenizer = load_model()

    print("\n" + "=" * 60)
    print("  HMNS Smoke Tests (Tiny GPT-2: 4 layers, 4 heads, dim=64)")
    print("=" * 60 + "\n")

    test_model_config(model)
    test_attention_layers(model)
    test_rms_and_kl()
    test_attribution(model, tokenizer)
    test_masking(model, tokenizer)
    test_nullspace(model)
    test_perturbation(model)
    test_intervention(model, tokenizer)
    test_pipeline(model, tokenizer)

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
