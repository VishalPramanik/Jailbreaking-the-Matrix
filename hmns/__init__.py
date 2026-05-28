# Copyright (c) 2025 Vishal Pramanik, University of Florida
# Licensed under the MIT License. See LICENSE for details.
"""
HMNS: Head-Masked Nullspace Steering for Controlled Model Subversion.

A circuit-level intervention method for decoder-only Transformer LLMs that
identifies causally responsible attention heads, suppresses their write paths,
and injects perturbations constrained to the orthogonal complement of the
muted subspace.

Reference:
    Pramanik, V., Maliha, M., Jha, S., & Jha, S. K. (2026).
    Jailbreaking the Matrix: Nullspace Steering for Controlled Model Subversion.
    Published as a conference paper at ICLR 2026.
"""

__version__ = "1.0.0"

from hmns.attribution import CausalHeadAttributor
from hmns.masking import HeadMasker
from hmns.nullspace import NullspaceSteering
from hmns.intervention import HMNSIntervention
from hmns.pipeline import HMNSPipeline

__all__ = [
    "CausalHeadAttributor",
    "HeadMasker",
    "NullspaceSteering",
    "HMNSIntervention",
    "HMNSPipeline",
]
