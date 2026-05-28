# Copyright (c) 2025 Vishal Pramanik, University of Florida
# Licensed under the MIT License. See LICENSE for details.
"""
Utility functions for HMNS.

Provides helper routines for residual stream operations, activation
normalization, logging configuration, and device management.
"""

import logging
import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all backends."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


# ---------------------------------------------------------------------------
# Activation helpers (Eq. 8 in the paper)
# ---------------------------------------------------------------------------

def rms_norm(activation: torch.Tensor) -> torch.Tensor:
    """
    Compute the Root-Mean-Square of an activation vector.

    RMS(a) = sqrt(1/d * sum_i a_i^2)

    This is used to scale the steering perturbation to match the magnitude
    of the residual stream (Equation 8).

    Args:
        activation: Tensor of shape (..., d).

    Returns:
        Scalar RMS value.
    """
    return torch.sqrt(torch.mean(activation.float() ** 2))


def compute_kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Compute KL(P || Q) for next-token distributions (Equation 4).

    KL(P || Q) = sum_i P_i * log(P_i / Q_i)

    Args:
        p: Baseline distribution, shape (V,).
        q: Ablated distribution, shape (V,).
        eps: Small constant for numerical stability (log-domain clipping).

    Returns:
        Scalar KL divergence value.
    """
    p = p.float().clamp(min=eps)
    q = q.float().clamp(min=eps)
    return (p * (p.log() - q.log())).sum()


# ---------------------------------------------------------------------------
# Model introspection
# ---------------------------------------------------------------------------

def get_attention_layers(model) -> List[Tuple[int, torch.nn.Module]]:
    """
    Retrieve all self-attention layers from a decoder-only Transformer.

    Supports HuggingFace model architectures including LLaMA, Phi-3, Mistral,
    Qwen, and GPT-2/GPT-Neo families.

    Args:
        model: A HuggingFace pretrained causal LM.

    Returns:
        List of (layer_index, attention_module) tuples.
    """
    attention_layers = []

    # LLaMA / Mistral / Qwen / Yi family
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "self_attn"):
                attention_layers.append((idx, layer.self_attn))
    # Phi-3 family
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        for idx, layer in enumerate(model.model.layers):
            if hasattr(layer, "self_attn"):
                attention_layers.append((idx, layer.self_attn))
    # GPT-2 / GPT-Neo family
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        for idx, layer in enumerate(model.transformer.h):
            if hasattr(layer, "attn"):
                attention_layers.append((idx, layer.attn))
    else:
        raise ValueError(
            f"Unsupported model architecture: {type(model).__name__}. "
            "Please implement a custom layer extractor."
        )

    logger.info(f"Found {len(attention_layers)} attention layers")
    return attention_layers


def get_out_projection(attn_module: torch.nn.Module) -> torch.nn.Linear:
    """
    Retrieve the out-projection (W^O) from an attention module.

    W^O ∈ R^{d × (H * d_h)} maps concatenated head outputs back into the
    residual stream (Equation 2).

    Args:
        attn_module: A single attention layer module.

    Returns:
        The W^O linear layer.
    """
    # Common attribute names across architectures
    for attr_name in ["o_proj", "dense", "out_proj", "c_proj"]:
        if hasattr(attn_module, attr_name):
            return getattr(attn_module, attr_name)

    raise ValueError(
        f"Cannot locate out-projection in {type(attn_module).__name__}. "
        f"Available attributes: {[a for a in dir(attn_module) if not a.startswith('_')]}"
    )


def get_model_config(model) -> Dict:
    """
    Extract model configuration parameters needed for HMNS.

    Returns:
        Dictionary with keys: num_layers, num_heads, hidden_dim, head_dim.
    """
    config = model.config
    num_heads = getattr(config, "num_attention_heads", None)
    hidden_dim = getattr(config, "hidden_size", None)
    num_layers = getattr(config, "num_hidden_layers", None)

    if num_heads is None or hidden_dim is None or num_layers is None:
        # Fallback: try n_head / n_embd / n_layer (GPT-2 convention)
        num_heads = getattr(config, "n_head", num_heads)
        hidden_dim = getattr(config, "n_embd", hidden_dim)
        num_layers = getattr(config, "n_layer", num_layers)

    assert num_heads is not None, "Cannot determine num_attention_heads"
    assert hidden_dim is not None, "Cannot determine hidden_size"
    assert num_layers is not None, "Cannot determine num_hidden_layers"

    head_dim = hidden_dim // num_heads

    # Check for grouped-query attention (GQA)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)

    return {
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "hidden_dim": hidden_dim,
        "head_dim": head_dim,
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure logging for the HMNS pipeline."""
    fmt = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=getattr(logging, level), format=fmt, handlers=handlers)


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

def get_device(model) -> torch.device:
    """Infer the device a model resides on."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
