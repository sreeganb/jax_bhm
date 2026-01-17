"""Tests for protein structure representation."""

import jax.numpy as jnp
import pytest
from jax_bhm.structure import (
    ProteinStructure,
    create_linear_chain,
    create_random_chain,
    add_noise_to_structure,
)


def test_protein_structure_init():
    """Test ProteinStructure initialization."""
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])
    
    structure = ProteinStructure(positions)
    assert structure.n_atoms == 3
    assert structure.positions.shape == (3, 3)
    assert len(structure.atom_names) == 3


def test_protein_structure_with_names():
    """Test ProteinStructure with atom names."""
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    atom_names = ['CA', 'CB']
    
    structure = ProteinStructure(positions, atom_names=atom_names)
    assert structure.atom_names == atom_names


def test_update_positions():
    """Test updating positions."""
    positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    structure = ProteinStructure(positions)
    
    new_positions = jnp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    new_structure = structure.update_positions(new_positions)
    
    assert jnp.allclose(new_structure.positions, new_positions)
    assert new_structure.n_atoms == structure.n_atoms


def test_center_of_mass():
    """Test center of mass calculation."""
    positions = jnp.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    structure = ProteinStructure(positions)
    com = structure.center_of_mass()
    
    expected = jnp.array([2.0/3.0, 2.0/3.0, 0.0])
    assert jnp.allclose(com, expected)


def test_rmsd():
    """Test RMSD calculation."""
    positions1 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    positions2 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    
    structure1 = ProteinStructure(positions1)
    structure2 = ProteinStructure(positions2)
    
    rmsd = structure1.rmsd(structure2)
    assert jnp.isclose(rmsd, 0.0)
    
    # Test with different positions
    positions3 = jnp.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    structure3 = ProteinStructure(positions3)
    rmsd = structure1.rmsd(structure3)
    assert rmsd > 0.0


def test_translate():
    """Test translation."""
    positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    structure = ProteinStructure(positions)
    
    translation = jnp.array([1.0, 2.0, 3.0])
    translated = structure.translate(translation)
    
    expected = positions + translation
    assert jnp.allclose(translated.positions, expected)


def test_create_linear_chain():
    """Test creating a linear chain."""
    n_residues = 5
    structure = create_linear_chain(n_residues)
    
    assert structure.n_atoms == n_residues
    assert len(structure.atom_names) == n_residues
    assert all(name == 'CA' for name in structure.atom_names)
    
    # Check spacing
    for i in range(n_residues - 1):
        dist = jnp.linalg.norm(structure.positions[i+1] - structure.positions[i])
        assert jnp.isclose(dist, 3.8)


def test_create_linear_chain_custom_spacing():
    """Test creating a linear chain with custom spacing."""
    n_residues = 3
    spacing = 5.0
    structure = create_linear_chain(n_residues, spacing=spacing)
    
    dist = jnp.linalg.norm(structure.positions[1] - structure.positions[0])
    assert jnp.isclose(dist, spacing)


def test_create_random_chain():
    """Test creating a random chain."""
    n_residues = 5
    bond_length = 3.8
    structure = create_random_chain(n_residues, bond_length=bond_length)
    
    assert structure.n_atoms == n_residues
    
    # Check bond lengths
    for i in range(n_residues - 1):
        dist = jnp.linalg.norm(structure.positions[i+1] - structure.positions[i])
        assert jnp.isclose(dist, bond_length, atol=1e-5)


def test_add_noise_to_structure():
    """Test adding noise to structure."""
    positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    structure = ProteinStructure(positions)
    
    noisy_structure = add_noise_to_structure(structure, noise_scale=0.1)
    
    assert noisy_structure.n_atoms == structure.n_atoms
    # Positions should be different but close
    assert not jnp.allclose(noisy_structure.positions, structure.positions)
    rmsd = structure.rmsd(noisy_structure)
    assert rmsd > 0.0
    assert rmsd < 1.0  # Should be relatively small


def test_to_pdb_string():
    """Test PDB string generation."""
    positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    atom_names = ['CA', 'CB']
    structure = ProteinStructure(positions, atom_names=atom_names)
    
    pdb_string = structure.to_pdb_string()
    
    assert 'ATOM' in pdb_string
    assert 'CA' in pdb_string
    assert 'CB' in pdb_string
    lines = pdb_string.split('\n')
    assert len(lines) == 2
