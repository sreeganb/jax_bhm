"""
Protein structure representation and manipulation.

This module provides classes and functions for representing and manipulating
protein structures in a way compatible with JAX operations.
"""

import jax.numpy as jnp
import jax
from typing import Optional, Tuple
import numpy as np


class ProteinStructure:
    """
    Representation of a protein structure.
    
    This class stores atomic coordinates and provides methods for
    structure manipulation and analysis.
    
    Attributes:
        positions: JAX array of shape (n_atoms, 3) with atomic coordinates
        atom_names: List of atom names (e.g., ['CA', 'CB', ...])
        residue_indices: Array of residue indices for each atom
    """
    
    def __init__(
        self,
        positions: jax.Array,
        atom_names: Optional[list] = None,
        residue_indices: Optional[jax.Array] = None
    ):
        """
        Initialize a protein structure.
        
        Args:
            positions: Array of shape (n_atoms, 3) with atomic coordinates
            atom_names: Optional list of atom names
            residue_indices: Optional array of residue indices
        """
        self.positions = jnp.array(positions)
        self.n_atoms = self.positions.shape[0]
        
        if atom_names is None:
            self.atom_names = [f"ATOM_{i}" for i in range(self.n_atoms)]
        else:
            self.atom_names = atom_names
            
        if residue_indices is None:
            self.residue_indices = jnp.arange(self.n_atoms)
        else:
            self.residue_indices = jnp.array(residue_indices)
    
    def update_positions(self, new_positions: jax.Array) -> 'ProteinStructure':
        """
        Create a new ProteinStructure with updated positions.
        
        Args:
            new_positions: New atomic coordinates
            
        Returns:
            New ProteinStructure instance
        """
        return ProteinStructure(
            positions=new_positions,
            atom_names=self.atom_names,
            residue_indices=self.residue_indices
        )
    
    def get_backbone_atoms(self) -> jax.Array:
        """
        Get indices of backbone atoms (CA atoms).
        
        Returns:
            Array of indices for backbone atoms
        """
        backbone_indices = [i for i, name in enumerate(self.atom_names) if name == 'CA']
        return jnp.array(backbone_indices)
    
    def center_of_mass(self) -> jax.Array:
        """
        Calculate the center of mass of the structure.
        
        Returns:
            Array of shape (3,) with center of mass coordinates
        """
        return jnp.mean(self.positions, axis=0)
    
    def rmsd(self, other: 'ProteinStructure') -> float:
        """
        Calculate RMSD to another structure.
        
        Args:
            other: Another ProteinStructure to compare to
            
        Returns:
            RMSD value
        """
        diff = self.positions - other.positions
        return jnp.sqrt(jnp.mean(jnp.sum(diff ** 2, axis=1)))
    
    def translate(self, translation: jax.Array) -> 'ProteinStructure':
        """
        Translate the structure.
        
        Args:
            translation: Translation vector of shape (3,)
            
        Returns:
            New ProteinStructure instance
        """
        new_positions = self.positions + translation
        return self.update_positions(new_positions)
    
    def rotate(self, rotation_matrix: jax.Array) -> 'ProteinStructure':
        """
        Rotate the structure.
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            New ProteinStructure instance
        """
        new_positions = jnp.dot(self.positions, rotation_matrix.T)
        return self.update_positions(new_positions)
    
    def to_pdb_string(self) -> str:
        """
        Convert structure to PDB format string.
        
        Returns:
            PDB format string
        """
        lines = []
        for i, (pos, atom_name, res_idx) in enumerate(
            zip(np.array(self.positions), self.atom_names, np.array(self.residue_indices))
        ):
            line = f"ATOM  {i+1:5d}  {atom_name:3s} ALA A{res_idx:4d}    "
            line += f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}"
            line += f"  1.00  0.00           C  "
            lines.append(line)
        return "\n".join(lines)


def create_linear_chain(n_residues: int, spacing: float = 3.8) -> ProteinStructure:
    """
    Create a linear chain of CA atoms.
    
    This creates a simple linear backbone useful for testing and demonstrations.
    
    Args:
        n_residues: Number of residues in the chain
        spacing: Distance between consecutive CA atoms (default 3.8 Å)
        
    Returns:
        ProteinStructure with linear chain
        
    Example:
        >>> structure = create_linear_chain(10)
        >>> structure.n_atoms
        10
    """
    positions = jnp.array([[i * spacing, 0.0, 0.0] for i in range(n_residues)])
    atom_names = ['CA'] * n_residues
    residue_indices = jnp.arange(n_residues)
    
    return ProteinStructure(
        positions=positions,
        atom_names=atom_names,
        residue_indices=residue_indices
    )


def create_random_chain(
    n_residues: int,
    bond_length: float = 3.8,
    key: Optional[jax.Array] = None
) -> ProteinStructure:
    """
    Create a random chain of CA atoms with bond length constraints.
    
    Args:
        n_residues: Number of residues in the chain
        bond_length: Distance between consecutive CA atoms
        key: JAX random key for reproducibility
        
    Returns:
        ProteinStructure with random chain
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    positions = [jnp.array([0.0, 0.0, 0.0])]
    
    for i in range(1, n_residues):
        # Generate random direction
        key, subkey = jax.random.split(key)
        direction = jax.random.normal(subkey, (3,))
        direction = direction / jnp.linalg.norm(direction)
        
        # New position at bond_length from previous
        new_pos = positions[-1] + direction * bond_length
        positions.append(new_pos)
    
    positions = jnp.stack(positions)
    atom_names = ['CA'] * n_residues
    residue_indices = jnp.arange(n_residues)
    
    return ProteinStructure(
        positions=positions,
        atom_names=atom_names,
        residue_indices=residue_indices
    )


def add_noise_to_structure(
    structure: ProteinStructure,
    noise_scale: float = 0.5,
    key: Optional[jax.Array] = None
) -> ProteinStructure:
    """
    Add Gaussian noise to structure coordinates.
    
    Args:
        structure: Input structure
        noise_scale: Standard deviation of noise
        key: JAX random key
        
    Returns:
        New structure with noise added
    """
    if key is None:
        key = jax.random.PRNGKey(42)
    
    noise = jax.random.normal(key, structure.positions.shape) * noise_scale
    new_positions = structure.positions + noise
    
    return structure.update_positions(new_positions)
