# Copyright (c) 2025 Vishal Pramanik, University of Florida
# Licensed under the MIT License. See LICENSE for details.
"""
Nullspace Steering for HMNS.

Computes orthogonal steering directions that lie in the nullspace of the
masked head write subspace (Section 3, Equations 5–8).

The procedure:
  1. Construct M_ℓ from selected out-projection slices (Eq. 5).
  2. Compute a thin QR factorization M_ℓ = Q_ℓ R_ℓ (Eq. 6).
  3. Sample r ~ N(0, I_d) and project into the orthogonal complement:
     u_ℓ = (I − Q_ℓ Q_ℓ^T) r / ‖…‖ (Eq. 7).
  4. Verify orthogonality: ‖M_ℓ^T u_ℓ‖_∞ < δ (resample if violated).
  5. Scale perturbation: δ_ℓ = α · RMS(a_ℓ) · u_ℓ (Eq. 8).
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

from hmns.masking import HeadMasker
from hmns.utils import rms_norm


logger = logging.getLogger(__name__)


class NullspaceSteering:
    """
    Computes geometry-constrained steering directions orthogonal to the
    masked head write subspace.

    By injecting directions in W_ℓ^⊥, the perturbation cannot be
    reconstructed or cancelled by the silenced heads (Theorem 2).

    Args:
        masker: A ``HeadMasker`` instance for write-matrix extraction.
        ortho_tol: Orthogonality tolerance δ (default: 1e-6).
        max_resample: Maximum resampling attempts if orthogonality fails.
        eps: Numerical stability constant ε for normalization.
    """

    def __init__(
        self,
        masker: HeadMasker,
        ortho_tol: float = 1e-6,
        max_resample: int = 3,
        eps: float = 1e-8,
    ):
        self.masker = masker
        self.ortho_tol = ortho_tol
        self.max_resample = max_resample
        self.eps = eps

    def compute_nullspace_direction(
        self,
        layer_idx: int,
        head_indices: List[int],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Compute a unit steering direction u_ℓ ∈ W_ℓ^⊥ (Eq. 7).

        Algorithm:
          1. Build M_ℓ ∈ R^{d × (|S_ℓ| · d_h)} from the selected head slices.
          2. Thin QR: M_ℓ = Q_ℓ R_ℓ.
          3. Sample r ~ N(0, I_d), project: v = (I − Q_ℓ Q_ℓ^T) r.
          4. Normalize: u_ℓ = v / (‖v‖_2 + ε).
          5. Verify: ‖M_ℓ^T u_ℓ‖_∞ < δ; resample r if violated.

        Args:
            layer_idx: Layer index ℓ.
            head_indices: Head indices S_ℓ selected for masking.
            device: Target device for the output tensor.

        Returns:
            Unit vector u_ℓ of shape (d,) in W_ℓ^⊥, or ``None`` if the
            nullspace is degenerate (rank(M_ℓ) = d).
        """
        # --- Step 1: Build write matrix M_ℓ (Eq. 5) ---
        M = self.masker.get_write_matrix(layer_idx, head_indices)  # (d, k*d_h)
        d = M.size(0)

        # Check that a non-trivial nullspace exists (Assumption in Sec. 3)
        if M.size(1) >= d:
            logger.warning(
                f"Layer {layer_idx}: write matrix has {M.size(1)} columns >= d={d}; "
                "nullspace may be trivial. Skipping this layer."
            )
            return None

        # --- Step 2: Thin QR factorization in float32 (Eq. 6) ---
        Q, R = torch.linalg.qr(M.float(), mode="reduced")  # Q: (d, r)

        # --- Step 3–5: Sample, project, verify ---
        for attempt in range(self.max_resample + 1):
            r = torch.randn(d, device=M.device, dtype=torch.float32)

            # Orthogonal complement projection: P^⊥_ℓ = I − Q Q^T
            projection = r - Q @ (Q.T @ r)  # (I − Q Q^T) r
            norm = projection.norm(p=2)

            if norm < self.eps:
                logger.debug(
                    f"Layer {layer_idx}, attempt {attempt + 1}: "
                    f"projected norm near zero ({norm:.2e}), resampling."
                )
                continue

            # Normalize (Eq. 7)
            u = projection / (norm + self.eps)

            # Orthogonality verification: ‖M^T u‖_∞ < δ
            orth_error = (M.T @ u).abs().max().item()

            if orth_error < self.ortho_tol:
                logger.debug(
                    f"Layer {layer_idx}: nullspace direction found "
                    f"(‖M^T u‖_∞ = {orth_error:.2e}, attempt {attempt + 1})"
                )
                return u.to(device=device)
            else:
                logger.debug(
                    f"Layer {layer_idx}, attempt {attempt + 1}: "
                    f"orthogonality violated (‖M^T u‖_∞ = {orth_error:.2e} >= δ={self.ortho_tol})"
                )

        logger.warning(
            f"Layer {layer_idx}: failed to find orthogonal direction after "
            f"{self.max_resample + 1} attempts. Skipping."
        )
        return None

    def compute_perturbation(
        self,
        u: torch.Tensor,
        activation: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Compute the scaled perturbation vector δ_ℓ (Eq. 8).

        δ_ℓ = α · RMS(a_ℓ) · u_ℓ

        Args:
            u: Unit nullspace direction u_ℓ, shape (d,).
            activation: Residual activation a_ℓ at the final token, shape (d,).
            alpha: Steering coefficient α.

        Returns:
            Perturbation δ_ℓ of shape (d,).
        """
        rms = rms_norm(activation)
        delta = alpha * rms * u.to(dtype=activation.dtype)
        return delta

    def compute_all_directions(
        self,
        layer_heads: Dict[int, List[int]],
        device: torch.device,
    ) -> Dict[int, torch.Tensor]:
        """
        Compute nullspace directions for all intervened layers.

        Args:
            layer_heads: Mapping of layer_idx → list of head indices.
            device: Target device.

        Returns:
            Dictionary mapping layer_idx → u_ℓ.
        """
        directions = {}
        for layer_idx, heads in layer_heads.items():
            u = self.compute_nullspace_direction(layer_idx, heads, device)
            if u is not None:
                directions[layer_idx] = u
        logger.info(
            f"Computed nullspace directions for {len(directions)}/{len(layer_heads)} layers"
        )
        return directions
