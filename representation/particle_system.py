"""
Particle system for integrative modeling with JAX.

This module provides a minimal data structure for representing particle systems,
ideal coordinates, and basic operations for MCMC sampling and scoring.
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class ParticleSystem:
    """
    Minimal particle system for MCMC.

    Attributes:
        types: Dict of particle types with radii and copy numbers.
        coords: JAX array of current coordinates (N, 3).
        ideal_coords: Dict of ideal coordinates per type.
    """
    types: Dict[str, Dict[str, float]]  # e.g., {'A': {'radius': 1.0, 'copy': 8}}
    coords: jnp.ndarray
    ideal_coords: Dict[str, jnp.ndarray]

    @classmethod
    def create(cls, types: Dict[str, Dict[str, float]], coords: jnp.ndarray) -> 'ParticleSystem':
        ideal_coords = get_ideal_coords()
        return cls(types, coords, ideal_coords)
    
    @property
    def identity_order(self) -> List[str]:
        return sorted(self.types.keys())

    def get_coords_by_type(self, identity: str) -> jnp.ndarray:
        """Get coordinates for a specific type."""
        # This assumes coords are sorted by type!
        start = sum(int(self.types[t]['copy']) for t in sorted(self.types) if t < identity)
        end = start + int(self.types[identity]['copy'])
        return self.coords[start:end]
        
    def unflatten(self, flat_coords: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """Convert flat array back to dictionary of coordinates."""
        return {k: self.get_coords_by_type_from_flat(flat_coords, k) for k in self.identity_order}
        
    def get_coords_by_type_from_flat(self, flat: jnp.ndarray, identity: str) -> jnp.ndarray:
        start = sum(int(self.types[t]['copy']) for t in sorted(self.types) if t < identity)
        end = start + int(self.types[identity]['copy'])
        return flat[start:end]

    def get_flat_radii(self) -> jnp.ndarray:
        """Get radii array corresponding to flat coordinates."""
        radii_list = []
        for k in self.identity_order:
            r = self.types[k]['radius']
            n = int(self.types[k]['copy'])
            radii_list.append(jnp.full((n,), r))
        return jnp.concatenate(radii_list, axis=0)

    def update_coords(self, new_coords: jnp.ndarray):
        """Update coordinates."""
        self.coords = new_coords


    def compute_score(self) -> float:
        """Placeholder for scoring function (e.g., distance-based)."""
        # Implement scoring logic here, e.g., sum of squared differences to ideal
        score = 0.0
        for identity in self.types:
            current = self.get_coords_by_type(identity)
            ideal = self.ideal_coords[identity]
            score += jnp.sum((current - ideal)**2)
        return float(score)


def get_ideal_coords() -> Dict[str, jnp.ndarray]:
    """Return ideal coordinates for particle types."""
    # Use numpy for initialization to avoid JAX Metal backend issues with list conversion
    array_A = np.array([
        [63., 0., 0.],
        [44.55, 44.55, 0.],
        [0., 63., 0.],
        [-44.55, 44.55, 0.],
        [-63., 0., 0.],
        [-44.55, -44.55, 0.],
        [0., -63., 0.],
        [44.55, -44.55, 0.]
    ])
    array_B = np.array([
        [63., 0., -38.5],
        [44.55, 44.55, -38.5],
        [0., 63., -38.5],
        [-44.55, 44.55, -38.5],
        [-63., 0., -38.5],
        [-44.55, -44.55, -38.5],
        [0., -63., -38.5],
        [44.55, -44.55, -38.5]
    ])
    array_C = np.array([
        [47.00, 0.00, -68.50],
        [79.00, 0.00, -68.50],
        [55.86, 55.86, -68.50],
        [33.23, 33.23, -68.50],
        [0.00, 47.00, -68.50],
        [0.00, 79.00, -68.50],
        [-55.86, 55.86, -68.50],
        [-33.23, 33.23, -68.50],
        [-47.00, 0.00, -68.50],
        [-79.00, 0.00, -68.50],
        [-55.86, -55.86, -68.50],
        [-33.23, -33.23, -68.50],
        [0.00, -47.00, -68.50],
        [0.00, -79.00, -68.50],
        [55.86, -55.86, -68.50],
        [33.23, -33.23, -68.50],
    ])
    return {
        'A': jnp.array(array_A),
        'B': jnp.array(array_B),
        'C': jnp.array(array_C)
    }


# Example usage
if __name__ == "__main__":
    types = {
        'A': {'radius': 1.0, 'copy': 8},
        'B': {'radius': 1.2, 'copy': 8},
        'C': {'radius': 0.8, 'copy': 16}
    }
    # Initialize with ideal coords
    ideal = get_ideal_coords()
    initial_coords = jnp.concatenate([ideal['A'], ideal['B'], ideal['C']])
    system = ParticleSystem.create(types, initial_coords)
    print(f"Total particles: {len(system.coords)}")
    print(f"Score: {system.compute_score()}")
