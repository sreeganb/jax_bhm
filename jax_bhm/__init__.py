"""
JAX-BHM: Protein Structure Sampling with BlackJax and Bayesian Inference

This package provides tools for protein structure sampling using JAX-based
scoring functions and BlackJax for Bayesian inference, similar to the
Integrative Modeling Platform (IMP).
"""

__version__ = "0.1.0"

from jax_bhm.scoring import (
    distance_restraint_score,
    angle_restraint_score,
    energy_function,
)
from jax_bhm.structure import ProteinStructure
from jax_bhm.sampling import sample_structure

__all__ = [
    "distance_restraint_score",
    "angle_restraint_score",
    "energy_function",
    "ProteinStructure",
    "sample_structure",
]
