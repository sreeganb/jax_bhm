"""
Test suite for the particle system module.
"""

import numpy as np
from particle_system import ParticleType, Particle, ParticleSystem, create_ideal_ground_truth_system


def test_particle_type_creation():
    """Test creating a particle type."""
    ptype = ParticleType('A', radius=1.0, copy_number=5)
    assert ptype.identity == 'A'
    assert ptype.radius == 1.0
    assert ptype.copy_number == 5
    print("✓ ParticleType creation test passed")


def test_particle_type_validation():
    """Test particle type validation."""
    try:
        ParticleType('AB', radius=1.0, copy_number=5)  # Should fail
        assert False, "Should have raised ValueError for multi-char identity"
    except ValueError:
        pass
    
    try:
        ParticleType('A', radius=-1.0, copy_number=5)  # Should fail
        assert False, "Should have raised ValueError for negative radius"
    except ValueError:
        pass
    
    try:
        ParticleType('A', radius=1.0, copy_number=-1)  # Should fail
        assert False, "Should have raised ValueError for negative copy_number"
    except ValueError:
        pass
    
    print("✓ ParticleType validation test passed")


def test_particle_creation():
    """Test creating a particle."""
    ptype = ParticleType('A', radius=1.0, copy_number=5)
    coords = np.array([1.0, 2.0, 3.0])
    particle = Particle(ptype, coords)
    
    assert particle.identity == 'A'
    assert particle.radius == 1.0
    assert particle.x == 1.0
    assert particle.y == 2.0
    assert particle.z == 3.0
    assert np.array_equal(particle.coordinates, coords)
    print("✓ Particle creation test passed")


def test_particle_system():
    """Test particle system operations."""
    system = ParticleSystem()
    
    # Add particle types
    ptype_a = system.add_particle_type('A', radius=1.0, copy_number=2)
    ptype_b = system.add_particle_type('B', radius=1.5, copy_number=3)
    
    assert len(system.particle_types) == 2
    assert 'A' in system.particle_types
    assert 'B' in system.particle_types
    
    # Add particles
    system.add_particle(ptype_a, np.array([0.0, 0.0, 0.0]))
    system.add_particle(ptype_a, np.array([1.0, 1.0, 1.0]))
    system.add_particle(ptype_b, np.array([2.0, 2.0, 2.0]))
    system.add_particle(ptype_b, np.array([3.0, 3.0, 3.0]))
    system.add_particle(ptype_b, np.array([4.0, 4.0, 4.0]))
    
    assert len(system.particles) == 5
    
    # Test get_particles_by_type
    particles_a = system.get_particles_by_type('A')
    assert len(particles_a) == 2
    
    particles_b = system.get_particles_by_type('B')
    assert len(particles_b) == 3
    
    # Test validate_copy_numbers
    assert system.validate_copy_numbers()
    
    # Test get_coordinates_array
    coords_array = system.get_coordinates_array()
    assert coords_array.shape == (5, 3)
    
    print("✓ ParticleSystem test passed")


def test_ideal_ground_truth_system():
    """Test the ideal ground truth system creation."""
    system = create_ideal_ground_truth_system()
    
    # Check particle types
    assert len(system.particle_types) == 3
    assert 'A' in system.particle_types
    assert 'B' in system.particle_types
    assert 'C' in system.particle_types
    
    # Check copy numbers
    assert system.particle_types['A'].copy_number == 8
    assert system.particle_types['B'].copy_number == 8
    assert system.particle_types['C'].copy_number == 16
    
    # Check radii
    assert system.particle_types['A'].radius == 1.0
    assert system.particle_types['B'].radius == 1.2
    assert system.particle_types['C'].radius == 0.8
    
    # Check total particles
    assert len(system.particles) == 32  # 8 + 8 + 16
    
    # Validate copy numbers match
    assert system.validate_copy_numbers()
    
    # Check particles by type
    particles_a = system.get_particles_by_type('A')
    particles_b = system.get_particles_by_type('B')
    particles_c = system.get_particles_by_type('C')
    
    assert len(particles_a) == 8
    assert len(particles_b) == 8
    assert len(particles_c) == 16
    
    # Check coordinates array shape
    coords = system.get_coordinates_array()
    assert coords.shape == (32, 3)
    
    print("✓ Ideal ground truth system test passed")


def test_empty_system():
    """Test empty particle system."""
    system = ParticleSystem()
    assert len(system.particles) == 0
    assert len(system.particle_types) == 0
    
    coords = system.get_coordinates_array()
    assert coords.shape == (0, 3)
    
    print("✓ Empty system test passed")


if __name__ == "__main__":
    print("Running particle system tests...\n")
    
    test_particle_type_creation()
    test_particle_type_validation()
    test_particle_creation()
    test_particle_system()
    test_ideal_ground_truth_system()
    test_empty_system()
    
    print("\n✅ All tests passed!")
