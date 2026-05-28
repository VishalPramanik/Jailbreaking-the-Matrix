# Copyright (c) 2025 Vishal Pramanik, University of Florida
# Licensed under the MIT License. See LICENSE for details.
"""
Dynamic Out-Projection Masking for HMNS.

Implements the column-zeroing of W^O for selected causal heads, effectively
removing their contribution to the residual stream during a single forward
pass (Section 3, Equation 3).

The masking is applied via a context manager that:
  1. Zeros the out-projection slices for all heads in the causal set S.
  2. Restores the original weights upon exit.
  3. Never modifies model weights on disk.
"""

import logging
from contextlib import contextmanager
from typing import Dict, List, Set, Tuple

import torch

from hmns.utils import get_attention_layers, get_model_config, get_out_projection


logger = logging.getLogger(__name__)


class HeadMasker:
    """
    Dynamically masks the out-projection columns of selected attention heads.

    For each layer ℓ with selected heads S_ℓ, the masking zeroes
    W^O_ℓ[:, h*d_h : (h+1)*d_h] for every h ∈ S_ℓ, effectively applying
    W̃^O_ℓ = W^O_ℓ (I − S_{ℓ,S}) during the current forward pass.

    Args:
        model: A HuggingFace causal language model.
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.config = get_model_config(model)
        self.attn_layers = get_attention_layers(model)
        self._attn_dict = {idx: mod for idx, mod in self.attn_layers}

    def _group_by_layer(
        self,
        heads: List[Tuple[int, int]],
    ) -> Dict[int, List[int]]:
        """Group (layer, head) pairs into a layer → [heads] mapping."""
        layer_heads: Dict[int, List[int]] = {}
        for layer_idx, head_idx in heads:
            layer_heads.setdefault(layer_idx, []).append(head_idx)
        return layer_heads

    @contextmanager
    def mask_heads(self, heads: List[Tuple[int, int]]):
        """
        Context manager that temporarily zeros the out-projection columns
        of the specified heads.

        The aggregated selector matrix S_{ℓ,S} zeroes all selected head
        slices simultaneously (Section 3, inference-time intervention).

        Args:
            heads: List of (layer_idx, head_idx) tuples.

        Yields:
            Dictionary mapping layer_idx → set of masked head indices (for
            downstream nullspace construction).

        Example::

            masker = HeadMasker(model)
            with masker.mask_heads([(0, 3), (2, 7)]) as masked_info:
                outputs = model(input_ids)
        """
        layer_heads = self._group_by_layer(heads)
        d_h = self.config["head_dim"]
        saved: Dict[int, Dict[int, torch.Tensor]] = {}

        # --- Zero selected columns ---
        for layer_idx, head_list in layer_heads.items():
            attn_module = self._attn_dict[layer_idx]
            wo = get_out_projection(attn_module)
            weight = wo.weight  # (d, H*d_h)
            saved[layer_idx] = {}
            for h in head_list:
                col_start = h * d_h
                col_end = (h + 1) * d_h
                saved[layer_idx][h] = weight.data[:, col_start:col_end].clone()
                weight.data[:, col_start:col_end] = 0.0

        logger.debug(
            f"Masked {sum(len(v) for v in layer_heads.values())} heads "
            f"across {len(layer_heads)} layers"
        )

        try:
            yield layer_heads
        finally:
            # --- Restore original columns ---
            for layer_idx, head_data in saved.items():
                attn_module = self._attn_dict[layer_idx]
                wo = get_out_projection(attn_module)
                weight = wo.weight
                for h, original_cols in head_data.items():
                    col_start = h * d_h
                    col_end = (h + 1) * d_h
                    weight.data[:, col_start:col_end] = original_cols

    def get_write_matrix(
        self,
        layer_idx: int,
        head_indices: List[int],
    ) -> torch.Tensor:
        """
        Construct the write matrix M_ℓ for a given layer (Eq. 5).

        M_ℓ = [W^O_ℓ[:, h*d_h : (h+1)*d_h]]_{h ∈ S_ℓ}  ∈  R^{d × (|S_ℓ|*d_h)}

        Args:
            layer_idx: Layer index.
            head_indices: Indices of the selected heads at this layer.

        Returns:
            M_ℓ tensor of shape (d, |S_ℓ| * d_h) in float32.
        """
        attn_module = self._attn_dict[layer_idx]
        wo = get_out_projection(attn_module)
        weight = wo.weight.data.float()  # float32 for QR stability
        d_h = self.config["head_dim"]

        slices = []
        for h in head_indices:
            col_start = h * d_h
            col_end = (h + 1) * d_h
            slices.append(weight[:, col_start:col_end])

        M = torch.cat(slices, dim=1)  # (d, |S_ℓ| * d_h)
        return M
