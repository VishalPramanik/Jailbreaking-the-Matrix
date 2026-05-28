# Copyright (c) 2025 Vishal Pramanik, University of Florida
# Licensed under the MIT License. See LICENSE for details.
"""
Compute-Normalized Evaluation Metrics for HMNS.

Implements the forward-equivalent pass (FEP) framework and associated
metrics from Section 4.3 and Appendix A3:

  - ACQ:  Average external query (decode) count.
  - IPC:  Internal Pass Count (forward-equivalent passes for attribution).
  - FPS:  FLOPs per Success (total floating-point operations).
  - LPS:  Latency per Success (wall-clock seconds to first success).
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


logger = logging.getLogger(__name__)


@dataclass
class ComputeMetrics:
    """Container for compute-normalized metrics (one prompt)."""

    acq: int           # External decodes to first success
    ipc: int           # Internal forward-equivalent passes
    fps: float         # FLOPs per success (×10^12)
    lps: float         # Latency per success (seconds)
    success: bool
    attempts: int


def estimate_forward_flops(
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    ff_dim: int,
    seq_len: int,
) -> float:
    """
    Estimate FLOPs for a single full forward pass (Eq. 18–19).

    Per-token, per-layer cost:
      F_attn(d, H, t) ≈ 4d² + 2H · t · d_h²
      F_mlp(d, d_ff)  ≈ 4 · d · d_ff

    Total decode FLOPs ≈ Σ_t Σ_ℓ [F_attn + F_mlp]

    Args:
        hidden_dim: Model hidden dimension d.
        num_layers: Number of transformer layers L.
        num_heads: Number of attention heads H.
        ff_dim: Feed-forward intermediate dimension d_ff.
        seq_len: Sequence length (prompt + generated tokens).

    Returns:
        Estimated FLOPs (floating point operations).
    """
    d = hidden_dim
    d_h = d // num_heads
    d_ff = ff_dim

    total = 0.0
    for t in range(1, seq_len + 1):
        per_layer = (4 * d * d + 2 * num_heads * t * d_h * d_h) + (4 * d * d_ff)
        total += num_layers * per_layer

    return total


def compute_metrics_from_result(result, model_config: Dict) -> ComputeMetrics:
    """
    Compute all metrics from an HMNSResult and model configuration.

    Args:
        result: An ``HMNSResult`` object.
        model_config: Dictionary with model params (from ``get_model_config``).

    Returns:
        ``ComputeMetrics`` dataclass.
    """
    ff_dim = getattr(model_config, "intermediate_size", None)
    if ff_dim is None:
        ff_dim = model_config.get("hidden_dim", 768) * 4  # common default

    # Rough FLOPs estimate
    flops_per_pass = estimate_forward_flops(
        hidden_dim=model_config.get("hidden_dim", 768),
        num_layers=model_config.get("num_layers", 12),
        num_heads=model_config.get("num_heads", 12),
        ff_dim=ff_dim if isinstance(ff_dim, int) else model_config.get("hidden_dim", 768) * 4,
        seq_len=128,  # approximate
    )

    fps = (result.ipc + result.acq) * flops_per_pass / 1e12

    return ComputeMetrics(
        acq=result.acq,
        ipc=result.ipc,
        fps=fps,
        lps=result.latency_s,
        success=result.success,
        attempts=result.total_attempts,
    )


def aggregate_metrics(metrics_list: List[ComputeMetrics]) -> Dict[str, float]:
    """
    Aggregate compute metrics across multiple prompts.

    Returns:
        Dictionary with mean values and success rate.
    """
    if not metrics_list:
        return {}

    n = len(metrics_list)
    successes = [m for m in metrics_list if m.success]
    n_success = len(successes)

    return {
        "asr": n_success / n * 100.0,
        "mean_acq": sum(m.acq for m in successes) / max(n_success, 1),
        "mean_ipc": sum(m.ipc for m in successes) / max(n_success, 1),
        "mean_fps": sum(m.fps for m in successes) / max(n_success, 1),
        "mean_lps": sum(m.lps for m in successes) / max(n_success, 1),
        "total_prompts": n,
        "successful_prompts": n_success,
    }
