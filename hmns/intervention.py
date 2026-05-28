# Copyright (c) 2025 Vishal Pramanik, University of Florida
# Licensed under the MIT License. See LICENSE for details.
"""
Inference-Time Intervention for HMNS.

Implements the two-part intervention at each decoding step (Section 3):
  1. Dynamic masking: zero out-projection columns of causal heads.
  2. Nullspace injection: add δ_ℓ = α · RMS(a_ℓ) · u_ℓ via forward hooks
     at the final token position.

Hooks are registered and removed in context-managed scopes to prevent
leakage between probes or attempts (Appendix A2.1).
"""

import logging
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from hmns.masking import HeadMasker
from hmns.nullspace import NullspaceSteering
from hmns.utils import get_attention_layers, get_model_config, get_out_projection, rms_norm


logger = logging.getLogger(__name__)


class HMNSIntervention:
    """
    Orchestrates the mask-and-steer intervention at inference time.

    Combines :class:`HeadMasker` (column zeroing) and
    :class:`NullspaceSteering` (orthogonal perturbation) into a single
    context manager that can be wrapped around any forward / generate call.

    Args:
        model: A HuggingFace causal language model.
        masker: ``HeadMasker`` instance.
        steerer: ``NullspaceSteering`` instance.
        alpha: Initial steering coefficient α_1.
        alpha_schedule: Schedule type (``"linear"`` or ``"cosine"``).
        alpha_growth: Growth rate for the linear schedule
            α_t = α * (1 + growth * (t − 1)).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        masker: HeadMasker,
        steerer: NullspaceSteering,
        alpha: float = 0.25,
        alpha_schedule: str = "linear",
        alpha_growth: float = 0.1,
    ):
        self.model = model
        self.masker = masker
        self.steerer = steerer
        self.alpha_init = alpha
        self.alpha_schedule = alpha_schedule
        self.alpha_growth = alpha_growth

        self.config = get_model_config(model)
        self.attn_layers = get_attention_layers(model)
        self._attn_dict = {idx: mod for idx, mod in self.attn_layers}

    def get_alpha(self, step: int) -> float:
        """
        Compute the steering coefficient at step t.

        Linear schedule (default, Section A2.1):
            α_t = α * (1 + 0.1 * (t − 1))

        Cosine schedule:
            α_t = α * (1 + 0.5 * (1 − cos(π * t / T_max)))

        Args:
            step: Current attempt index (1-indexed).

        Returns:
            Steering coefficient α_t.
        """
        if self.alpha_schedule == "linear":
            return self.alpha_init * (1.0 + self.alpha_growth * (step - 1))
        elif self.alpha_schedule == "cosine":
            import math
            return self.alpha_init * (1.0 + 0.5 * (1.0 - math.cos(math.pi * step / 10.0)))
        else:
            return self.alpha_init

    def _build_steering_hook(
        self,
        direction: torch.Tensor,
        alpha: float,
    ) -> Callable:
        """
        Build a forward hook that injects the nullspace perturbation
        at the final token position (Eq. 8).

        The hook intercepts the output of the attention out-projection
        and adds δ_ℓ = α · RMS(a_ℓ) · u_ℓ to the last position.

        Args:
            direction: Unit nullspace direction u_ℓ, shape (d,).
            alpha: Steering coefficient α_t.

        Returns:
            A hook function compatible with ``register_forward_hook``.
        """
        def hook_fn(module, input, output):
            # output is a tuple; first element is the hidden states
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            # Inject at the final token position only
            activation = hidden[:, -1, :]  # (batch, d)
            rms = rms_norm(activation)
            perturbation = alpha * rms * direction.to(
                dtype=hidden.dtype, device=hidden.device
            )

            # Clone to avoid in-place issues with autograd graph
            modified = hidden.clone()
            modified[:, -1, :] = modified[:, -1, :] + perturbation

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return hook_fn

    def _get_attn_output_module(self, layer_idx: int) -> nn.Module:
        """
        Get the module whose output we hook for steering injection.

        We hook the out-projection (W^O) so that the perturbation is
        applied after attention computation, as recommended in Table 12.
        """
        attn_module = self._attn_dict[layer_idx]
        return get_out_projection(attn_module)

    @contextmanager
    def intervene(
        self,
        heads: List[Tuple[int, int]],
        step: int = 1,
    ):
        """
        Context manager that applies the full HMNS intervention:
        masking + nullspace steering.

        Usage::

            intervention = HMNSIntervention(model, masker, steerer)
            with intervention.intervene(top_k_heads, step=t):
                output = model.generate(input_ids, ...)

        Args:
            heads: List of (layer_idx, head_idx) for the causal set S.
            step: Current closed-loop iteration (1-indexed).

        Yields:
            None. The intervention is applied via hooks and weight masking.
        """
        alpha = self.get_alpha(step)
        hooks = []

        # Use the masker context to zero selected columns
        with self.masker.mask_heads(heads) as layer_heads:
            # Compute nullspace directions for each intervened layer
            device = next(self.model.parameters()).device
            directions = self.steerer.compute_all_directions(layer_heads, device)

            # Register forward hooks for steering injection
            for layer_idx, u in directions.items():
                module = self._get_attn_output_module(layer_idx)
                hook = module.register_forward_hook(
                    self._build_steering_hook(u, alpha)
                )
                hooks.append(hook)

            logger.debug(
                f"Intervention active: {len(heads)} heads masked, "
                f"{len(hooks)} layers steered, α={alpha:.4f}"
            )

            try:
                yield
            finally:
                # Remove all hooks to prevent leakage
                for hook in hooks:
                    hook.remove()
