"""
Particle system definition for protein simulations.

This module defines a system of proteins represented as spheres in 3D space.
Each particle type has an identity (A, B, C, etc.), a radius, and a copy number.
Individual particles are defined by their 3D coordinates (x, y, z).
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class ParticleType:
    """
    Defines a type of particle with its properties.
    
    Attributes:
        identity: Single character identifier (e.g., 'A', 'B', 'C')
        radius: Radius of the particle sphere
        copy_number: Number of instances of this particle type
    """
    identity: str
    radius: float
    copy_number: int
    
    def __post_init__(self):
        if len(self.identity) != 1:
            raise ValueError(f"Identity must be a single character, got: {self.identity}")
        if self.radius <= 0:
            raise ValueError(f"Radius must be positive, got: {self.radius}")
        if self.copy_number < 0:
            raise ValueError(f"Copy number must be non-negative, got: {self.copy_number}")


@dataclass
class Particle:
    """
    Represents an individual particle instance in 3D space.
    
    Attributes:
        particle_type: The type of this particle
        coordinates: 3D coordinates (x, y, z) of the particle center
    """
    particle_type: ParticleType
    coordinates: np.ndarray
    
    def __post_init__(self):
        self.coordinates = np.asarray(self.coordinates, dtype=float)
        if self.coordinates.shape != (3,):
            raise ValueError(f"Coordinates must be 3D, got shape: {self.coordinates.shape}")
    
    @property
    def identity(self) -> str:
        return self.particle_type.identity
    
    @property
    def radius(self) -> float:
        return self.particle_type.radius
    
    @property
    def x(self) -> float:
        return self.coordinates[0]
    
    @property
    def y(self) -> float:
        return self.coordinates[1]
    
    @property
    def z(self) -> float:
        return self.coordinates[2]


@dataclass
class ParticleSystem:
    """
    Represents a complete system of particles.
    
    Attributes:
        particle_types: Dictionary mapping identity to ParticleType
        particles: List of all particle instances in the system
    """
    particle_types: Dict[str, ParticleType] = field(default_factory=dict)
    particles: List[Particle] = field(default_factory=list)
    
    def add_particle_type(self, identity: str, radius: float, copy_number: int):
        """Add a new particle type to the system."""
        particle_type = ParticleType(identity, radius, copy_number)
        self.particle_types[identity] = particle_type
        return particle_type
    
    def add_particle(self, particle_type: ParticleType, coordinates: np.ndarray):
        """Add a particle instance to the system."""
        particle = Particle(particle_type, coordinates)
        self.particles.append(particle)
        return particle
    
    def get_particles_by_type(self, identity: str) -> List[Particle]:
        """Get all particles of a specific type."""
        return [p for p in self.particles if p.identity == identity]
    
    def get_coordinates_array(self) -> np.ndarray:
        """Get all particle coordinates as an Nx3 numpy array."""
        if not self.particles:
            return np.array([]).reshape(0, 3)
        return np.array([p.coordinates for p in self.particles])
    
    def validate_copy_numbers(self) -> bool:
        """
        Validate that the number of particles matches the specified copy numbers.
        
        Returns:
            True if all copy numbers match, False otherwise
        """
        for identity, particle_type in self.particle_types.items():
            actual_count = len(self.get_particles_by_type(identity))
            if actual_count != particle_type.copy_number:
                return False
        return True
    
    def __str__(self) -> str:
        lines = ["Particle System:"]
        lines.append(f"  Particle Types: {len(self.particle_types)}")
        for identity, ptype in sorted(self.particle_types.items()):
            lines.append(f"    {identity}: radius={ptype.radius}, copy_number={ptype.copy_number}")
        lines.append(f"  Total Particles: {len(self.particles)}")
        return "\n".join(lines)


def create_ideal_ground_truth_system() -> ParticleSystem:
    """
    Create the ideal ground truth structure for a system with:
    - Particle A: copy number 8
    - Particle B: copy number 8
    - Particle C: copy number 16
    
    This is a reference configuration with predefined coordinates.
    The coordinates are arranged in a structured pattern for demonstration.
    
    Returns:
        ParticleSystem with the ideal configuration
    """
    system = ParticleSystem()
    
    # Define particle types with example radii
    particle_A = system.add_particle_type('A', radius=1.0, copy_number=8)
    particle_B = system.add_particle_type('B', radius=1.2, copy_number=8)
    particle_C = system.add_particle_type('C', radius=0.8, copy_number=16)
    
    # Define ideal coordinates for particle type A (8 particles)
    # Arranged in a cubic pattern
    coords_A = [
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [5.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
        [5.0, 0.0, 5.0],
        [0.0, 5.0, 5.0],
        [5.0, 5.0, 5.0],
    ]
    
    # Define ideal coordinates for particle type B (8 particles)
    # Arranged offset from A particles
    coords_B = [
        [2.5, 2.5, 0.0],
        [7.5, 2.5, 0.0],
        [2.5, 7.5, 0.0],
        [7.5, 7.5, 0.0],
        [2.5, 2.5, 5.0],
        [7.5, 2.5, 5.0],
        [2.5, 7.5, 5.0],
        [7.5, 7.5, 5.0],
    ]
    
    # Define ideal coordinates for particle type C (16 particles)
    # Arranged in a denser grid
    coords_C = [
        [1.0, 1.0, 2.5],
        [4.0, 1.0, 2.5],
        [7.0, 1.0, 2.5],
        [10.0, 1.0, 2.5],
        [1.0, 4.0, 2.5],
        [4.0, 4.0, 2.5],
        [7.0, 4.0, 2.5],
        [10.0, 4.0, 2.5],
        [1.0, 7.0, 2.5],
        [4.0, 7.0, 2.5],
        [7.0, 7.0, 2.5],
        [10.0, 7.0, 2.5],
        [1.0, 10.0, 2.5],
        [4.0, 10.0, 2.5],
        [7.0, 10.0, 2.5],
        [10.0, 10.0, 2.5],
    ]
    
    # Add all particles to the system
    for coord in coords_A:
        system.add_particle(particle_A, np.array(coord))
    
    for coord in coords_B:
        system.add_particle(particle_B, np.array(coord))
    
    for coord in coords_C:
        system.add_particle(particle_C, np.array(coord))
    
    return system


if __name__ == "__main__":
    # Create and display the ideal ground truth system
    ideal_system = create_ideal_ground_truth_system()
    print(ideal_system)
    print()
    
    # Validate copy numbers
    is_valid = ideal_system.validate_copy_numbers()
    print(f"Copy numbers valid: {is_valid}")
    print()
    
    # Display coordinates for each particle type
    for identity in sorted(ideal_system.particle_types.keys()):
        particles = ideal_system.get_particles_by_type(identity)
        print(f"\nParticle type {identity} ({len(particles)} particles):")
        for i, particle in enumerate(particles):
            print(f"  {i+1}. Position: ({particle.x:.2f}, {particle.y:.2f}, {particle.z:.2f}), "
                  f"Radius: {particle.radius:.2f}")
    
    # Get all coordinates as an array
    all_coords = ideal_system.get_coordinates_array()
    print(f"\nAll coordinates shape: {all_coords.shape}")
    print(f"Coordinate bounds:")
    print(f"  X: [{all_coords[:, 0].min():.2f}, {all_coords[:, 0].max():.2f}]")
    print(f"  Y: [{all_coords[:, 1].min():.2f}, {all_coords[:, 1].max():.2f}]")
    print(f"  Z: [{all_coords[:, 2].min():.2f}, {all_coords[:, 2].max():.2f}]")
