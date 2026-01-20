"""Setup script for toy_model_problem package."""
from setuptools import setup, find_packages

setup(
    name="jax_bhm",
    version="0.1.0",
    description="Toy model problem for BHM sampling using JAX",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "numpy>=1.20",
    ],
    python_requires=">=3.8",
)
