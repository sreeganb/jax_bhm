# JAX-BHM: Protein Structure Sampling with Bayesian Inference

JAX-BHM is a Python package for protein structure sampling using JAX-based scoring functions and BlackJax for Bayesian inference, inspired by the Integrative Modeling Platform (IMP).

## Features

- **JAX-based scoring functions**: Fast, differentiable energy functions for protein structure evaluation
  - Distance restraints
  - Angle restraints
  - Excluded volume (soft-sphere repulsion)
- **BlackJax integration**: State-of-the-art MCMC sampling algorithms
  - NUTS (No-U-Turn Sampler)
  - HMC (Hamiltonian Monte Carlo)
  - Adaptive step size and mass matrix tuning
- **Protein structure representation**: Simple, JAX-compatible protein structure handling
- **Bayesian inference**: Sample protein conformations according to Boltzmann distribution

## Installation

```bash
# Clone the repository
git clone https://github.com/sreeganb/jax_bhm.git
cd jax_bhm

# Install the package
pip install -e .

# For development (includes matplotlib for visualization)
pip install -e ".[dev]"
```

## Quick Start

```python
import jax.numpy as jnp
from jax_bhm.structure import create_linear_chain
from jax_bhm.scoring import create_energy_fn
from jax_bhm.sampling import sample_structure

# Create a simple linear protein chain
structure = create_linear_chain(n_residues=10)

# Define distance restraints (atom_i, atom_j, target_distance, weight)
restraints = jnp.array([
    [0, 9, 10.0, 1.0],   # Bring chain ends closer
    [2, 7, 8.0, 1.0],    # Create a kink
])

# Create energy function
energy_fn = create_energy_fn(
    distance_restraints=restraints,
    use_excluded_volume=True
)

# Sample conformations using MCMC
samples, info = sample_structure(
    initial_structure=structure,
    energy_fn=energy_fn,
    n_samples=1000,
    n_warmup=500,
    algorithm="nuts"
)

print(f"Generated {samples.shape[0]} samples")
print(f"Acceptance rate: {info['acceptance_rate']:.2%}")
```

## Example Usage

Run the included example:

```bash
python examples/simple_sampling.py
```

This demonstrates:
1. Creating a protein structure
2. Defining restraints
3. Running MCMC sampling
4. Analyzing results

## Core Modules

### `jax_bhm.scoring`

Provides energy functions for structure evaluation:

- **`distance_restraint_score`**: Harmonic restraints on atom-atom distances
- **`angle_restraint_score`**: Harmonic restraints on three-atom angles
- **`excluded_volume_score`**: Soft-sphere repulsive potential
- **`energy_function`**: Combined scoring function
- **`create_energy_fn`**: Factory for creating energy functions

### `jax_bhm.structure`

Protein structure representation and manipulation:

- **`ProteinStructure`**: Main structure class with coordinates and metadata
- **`create_linear_chain`**: Create simple linear backbone
- **`create_random_chain`**: Create random walk backbone
- **`add_noise_to_structure`**: Add Gaussian noise to coordinates

### `jax_bhm.sampling`

MCMC sampling with BlackJax:

- **`sample_structure`**: Main sampling function (NUTS or HMC)
- **`sample_structure_gradient_based`**: Adaptive NUTS with window adaptation
- **`calculate_sampling_statistics`**: Compute statistics from samples
- **`get_best_structure`**: Find lowest energy structure

## How It Works

JAX-BHM uses Bayesian inference to sample protein structures according to:

```
P(structure | restraints) ∝ exp(-E(structure) / T)
```

Where:
- `E(structure)` is the energy function combining all restraints
- `T` is the temperature parameter
- MCMC (via BlackJax) samples from this probability distribution

This is similar to the Integrative Modeling Platform (IMP) approach, but:
- Uses JAX for automatic differentiation and GPU acceleration
- Leverages BlackJax's modern MCMC algorithms
- Provides a simpler, more focused API for protein sampling

## Comparison with IMP

| Feature | JAX-BHM | IMP |
|---------|---------|-----|
| Backend | JAX (GPU-capable) | C++/Python |
| Sampling | BlackJax (NUTS, HMC) | Various optimizers |
| Differentiation | Automatic (JAX) | Manual gradients |
| Focus | Simple protein sampling | Comprehensive integrative modeling |
| Use case | Research/education | Production modeling |

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run example
python examples/simple_sampling.py
```

## Dependencies

- JAX >= 0.4.20
- BlackJax >= 1.0.0
- NumPy >= 1.24.0
- SciPy >= 1.11.0

## License

MIT License - see LICENSE file for details.

## Citation

If you use JAX-BHM in your research, please cite:

```bibtex
@software{jax_bhm,
  author = {sreeganb},
  title = {JAX-BHM: Protein Structure Sampling with Bayesian Inference},
  year = {2026},
  url = {https://github.com/sreeganb/jax_bhm}
}
```

## Acknowledgments

- Inspired by the [Integrative Modeling Platform (IMP)](https://integrativemodeling.org/)
- Built with [JAX](https://github.com/google/jax) and [BlackJax](https://github.com/blackjax-devs/blackjax)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.