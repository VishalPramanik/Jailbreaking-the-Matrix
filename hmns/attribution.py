# Copyright (c) 2025 Vishal Pramanik, University of Florida
# Licensed under the MIT License. See LICENSE for details.
"""
Causal Head Attribution for HMNS.

Implements KL-divergence-based scoring of attention heads to identify the
top-K causally responsible heads for a model's continuation behaviour
(Section 3, Equations 3–4).

The attribution process:
  1. Run a clean forward pass to obtain baseline logits P = softmax(z).
  2. For each head (ℓ, h), zero its out-projection slice and recompute logits
     to obtain the ablated distribution P̃^{(ℓ,h)}.
  3. Score each head by Δ_{ℓ,h} = KL(P ∥ P̃^{(ℓ,h)}).
  4. Return the global top-K heads ranked by Δ.
"""

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn.functional as F

from hmns.utils import (
    compute_kl_divergence,
    get_attention_layers,
    get_device,
    get_model_config,
    get_out_projection,
)


logger = logging.getLogger(__name__)


class CausalHeadAttributor:
    """
    Identifies the most causally influential attention heads via
    counterfactual ablation and KL-divergence scoring (Eq. 4).

    Supports two scoring modes:
      - ``kl``: Full-distribution KL divergence (default; best ASR).
      - ``target_logit``: Drop in the target-token logit (faster but lower ASR).

    Args:
        model: A HuggingFace causal language model.
        tokenizer: The corresponding tokenizer.
        top_k: Number of heads to select globally (default: 10).
        scoring: Scoring function identifier (``"kl"`` or ``"target_logit"``).
        proxy_preselect: Whether to use lightweight proxy pre-selection before
            exact KL scoring (recommended for efficiency).
        proxy_k: Size of the proxy shortlist when ``proxy_preselect=True``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        top_k: int = 10,
        scoring: str = "kl",
        proxy_preselect: bool = True,
        proxy_k: int = 30,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.scoring = scoring
        self.proxy_preselect = proxy_preselect
        self.proxy_k = proxy_k

        self.device = get_device(model)
        self.config = get_model_config(model)
        self.attn_layers = get_attention_layers(model)

        # Validate
        total_heads = self.config["num_layers"] * self.config["num_heads"]
        assert top_k <= total_heads, (
            f"top_k={top_k} exceeds total heads={total_heads}"
        )

    # ------------------------------------------------------------------
    # Context manager for temporary column masking (Eq. 3)
    # ------------------------------------------------------------------

    @contextmanager
    def _mask_head(self, layer_idx: int, head_idx: int):
        """
        Temporarily zero the out-projection columns for head *head_idx*
        at layer *layer_idx*.

        Implements Eq. 3:  W̃^O_{ℓ,h} = W^O_ℓ (I − S_{ℓ,h})

        The original weight columns are restored on exit.
        """
        attn_module = self.attn_layers[layer_idx][1]
        wo = get_out_projection(attn_module)
        weight = wo.weight  # shape (d, H*d_h)  [transposed in nn.Linear]
        d_h = self.config["head_dim"]

        col_start = head_idx * d_h
        col_end = (head_idx + 1) * d_h

        # Save original columns
        original = weight.data[:, col_start:col_end].clone()

        # Zero the columns (masking)
        weight.data[:, col_start:col_end] = 0.0

        try:
            yield
        finally:
            # Restore original columns
            weight.data[:, col_start:col_end] = original

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return the final-position logits z ∈ R^V."""
        outputs = self.model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[:, -1, :]  # (1, V)
        return logits.squeeze(0)  # (V,)

    # ------------------------------------------------------------------
    # Proxy pre-selection (batched target-logit drop)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _proxy_preselect(
        self,
        input_ids: torch.Tensor,
        baseline_logits: torch.Tensor,
    ) -> List[Tuple[int, int]]:
        """
        Lightweight proxy scoring: measure the drop in the top predicted
        token's logit when each head is ablated. Returns the top-proxy_k
        heads by logit drop for exact scoring.
        """
        target_token = baseline_logits.argmax().item()
        baseline_val = baseline_logits[target_token].item()

        scores: List[Tuple[float, int, int]] = []

        for layer_idx, _attn in self.attn_layers:
            for head_idx in range(self.config["num_heads"]):
                with self._mask_head(layer_idx, head_idx):
                    ablated_logits = self._forward_logits(input_ids)
                drop = baseline_val - ablated_logits[target_token].item()
                scores.append((drop, layer_idx, head_idx))

        # Sort descending by drop magnitude
        scores.sort(key=lambda x: -x[0])
        shortlist = [(l, h) for _, l, h in scores[: self.proxy_k]]
        logger.debug(
            f"Proxy pre-selection: shortlisted {len(shortlist)} heads "
            f"(max drop={scores[0][0]:.4f})"
        )
        return shortlist

    # ------------------------------------------------------------------
    # Exact KL scoring (Eq. 4)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _score_kl(
        self,
        input_ids: torch.Tensor,
        baseline_dist: torch.Tensor,
        candidates: List[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], float]:
        """
        Score each candidate head via KL(P ∥ P̃^{(ℓ,h)}) (Eq. 4).

        Args:
            input_ids: Tokenized prompt.
            baseline_dist: P = softmax(z), shape (V,).
            candidates: List of (layer_idx, head_idx) to evaluate.

        Returns:
            Dictionary mapping (layer, head) → Δ_{ℓ,h}.
        """
        scores = {}
        for layer_idx, head_idx in candidates:
            with self._mask_head(layer_idx, head_idx):
                ablated_logits = self._forward_logits(input_ids)
            ablated_dist = F.softmax(ablated_logits, dim=-1)
            kl = compute_kl_divergence(baseline_dist, ablated_dist)
            scores[(layer_idx, head_idx)] = kl.item()
        return scores

    # ------------------------------------------------------------------
    # Target-logit scoring (simpler alternative)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _score_target_logit(
        self,
        input_ids: torch.Tensor,
        baseline_logits: torch.Tensor,
        candidates: List[Tuple[int, int]],
    ) -> Dict[Tuple[int, int], float]:
        """Score heads by drop in the target token's logit."""
        target_token = baseline_logits.argmax().item()
        baseline_val = baseline_logits[target_token].item()
        scores = {}
        for layer_idx, head_idx in candidates:
            with self._mask_head(layer_idx, head_idx):
                ablated_logits = self._forward_logits(input_ids)
            scores[(layer_idx, head_idx)] = (
                baseline_val - ablated_logits[target_token].item()
            )
        return scores

    # ------------------------------------------------------------------
    # Main attribution entry point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def attribute(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
        """
        Identify the global top-K causally responsible attention heads.

        Algorithm:
          1. Run baseline forward → logits z, distribution P.
          2. (Optional) proxy pre-selection to form a shortlist.
          3. Exact scoring on the shortlist via KL or target-logit.
          4. Select top-K heads globally.

        Args:
            input_ids: Tokenized prompt tensor, shape (1, T).

        Returns:
            Tuple of:
              - List of (layer_idx, head_idx) for the top-K causal heads.
              - Full scores dictionary for all evaluated candidates.
        """
        assert input_ids.dim() == 2 and input_ids.size(0) == 1

        # --- Step 1: Baseline forward ---
        baseline_logits = self._forward_logits(input_ids)
        baseline_dist = F.softmax(baseline_logits, dim=-1)

        # --- Step 2: Candidate set ---
        if self.proxy_preselect:
            candidates = self._proxy_preselect(input_ids, baseline_logits)
        else:
            candidates = [
                (l, h)
                for l, _ in self.attn_layers
                for h in range(self.config["num_heads"])
            ]

        # --- Step 3: Exact scoring ---
        if self.scoring == "kl":
            scores = self._score_kl(input_ids, baseline_dist, candidates)
        elif self.scoring == "target_logit":
            scores = self._score_target_logit(
                input_ids, baseline_logits, candidates
            )
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring}")

        # --- Step 4: Global top-K selection ---
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        top_k_heads = [head for head, _ in ranked[: self.top_k]]

        logger.info(
            f"Attribution complete: selected {len(top_k_heads)} heads "
            f"(top score={ranked[0][1]:.4f}, "
            f"bottom score={ranked[min(self.top_k - 1, len(ranked) - 1)][1]:.4f})"
        )

        return top_k_heads, scores
