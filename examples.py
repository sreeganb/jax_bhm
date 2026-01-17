"""
Example demonstrating how to create custom particle systems with arbitrary
particle types, copy numbers, and configurations.
"""

import numpy as np
from particle_system import ParticleSystem


def create_custom_system_example():
    """
    Example of creating a custom particle system with different
    particle types and copy numbers.
    """
    print("=" * 60)
    print("Example 1: Custom System with 4 Particle Types")
    print("=" * 60)
    
    system = ParticleSystem()
    
    # Define 4 different particle types
    particle_A = system.add_particle_type('A', radius=2.0, copy_number=5)
    particle_B = system.add_particle_type('B', radius=1.5, copy_number=10)
    particle_C = system.add_particle_type('C', radius=1.0, copy_number=15)
    particle_D = system.add_particle_type('D', radius=0.5, copy_number=20)
    
    # Generate random coordinates for particle A
    np.random.seed(42)  # For reproducibility
    for i in range(5):
        coords = np.random.uniform(0, 10, size=3)
        system.add_particle(particle_A, coords)
    
    # Generate coordinates for particle B in a line
    for i in range(10):
        coords = np.array([i * 1.0, 0.0, 0.0])
        system.add_particle(particle_B, coords)
    
    # Generate coordinates for particle C in a grid
    count = 0
    for i in range(5):
        for j in range(3):
            coords = np.array([i * 2.0, j * 2.0, 5.0])
            system.add_particle(particle_C, coords)
            count += 1
    
    # Generate coordinates for particle D in a sphere
    phi = np.linspace(0, 2*np.pi, 20)
    for i in range(20):
        r = 5.0
        theta = np.pi / 4  # Fixed elevation
        x = r * np.cos(phi[i]) * np.sin(theta)
        y = r * np.sin(phi[i]) * np.sin(theta)
        z = r * np.cos(theta)
        system.add_particle(particle_D, np.array([x, y, z]))
    
    print(system)
    print(f"\nCopy numbers valid: {system.validate_copy_numbers()}")
    
    coords = system.get_coordinates_array()
    print(f"\nTotal particles: {len(system.particles)}")
    print(f"Coordinates array shape: {coords.shape}")
    
    return system


def create_minimal_system_example():
    """
    Example of creating a minimal system with just 2 particles.
    """
    print("\n" + "=" * 60)
    print("Example 2: Minimal System with 2 Particles")
    print("=" * 60)
    
    system = ParticleSystem()
    
    # Single particle type with 2 instances
    particle_X = system.add_particle_type('X', radius=3.0, copy_number=2)
    
    system.add_particle(particle_X, np.array([0.0, 0.0, 0.0]))
    system.add_particle(particle_X, np.array([10.0, 10.0, 10.0]))
    
    print(system)
    print(f"\nCopy numbers valid: {system.validate_copy_numbers()}")
    
    # Calculate distance between particles
    particles = system.particles
    if len(particles) == 2:
        dist = np.linalg.norm(particles[0].coordinates - particles[1].coordinates)
        print(f"Distance between particles: {dist:.2f}")
    
    return system


def demonstrate_particle_access():
    """
    Demonstrate different ways to access and analyze particles.
    """
    print("\n" + "=" * 60)
    print("Example 3: Accessing and Analyzing Particles")
    print("=" * 60)
    
    system = ParticleSystem()
    
    # Create a simple system
    ptype_A = system.add_particle_type('A', radius=1.0, copy_number=3)
    ptype_B = system.add_particle_type('B', radius=2.0, copy_number=2)
    
    system.add_particle(ptype_A, np.array([1.0, 0.0, 0.0]))
    system.add_particle(ptype_A, np.array([0.0, 1.0, 0.0]))
    system.add_particle(ptype_A, np.array([0.0, 0.0, 1.0]))
    system.add_particle(ptype_B, np.array([2.0, 2.0, 0.0]))
    system.add_particle(ptype_B, np.array([0.0, 2.0, 2.0]))
    
    print(system)
    
    # Access particles by type
    print("\nParticles of type A:")
    for i, particle in enumerate(system.get_particles_by_type('A')):
        print(f"  {i+1}. ({particle.x:.2f}, {particle.y:.2f}, {particle.z:.2f})")
    
    print("\nParticles of type B:")
    for i, particle in enumerate(system.get_particles_by_type('B')):
        print(f"  {i+1}. ({particle.x:.2f}, {particle.y:.2f}, {particle.z:.2f})")
    
    # Calculate center of mass for each type
    for identity in ['A', 'B']:
        particles = system.get_particles_by_type(identity)
        coords = np.array([p.coordinates for p in particles])
        center_of_mass = coords.mean(axis=0)
        print(f"\nCenter of mass for type {identity}: ({center_of_mass[0]:.2f}, "
              f"{center_of_mass[1]:.2f}, {center_of_mass[2]:.2f})")
    
    return system


def demonstrate_large_system():
    """
    Demonstrate handling a larger system with many particles.
    """
    print("\n" + "=" * 60)
    print("Example 4: Large System with 100+ Particles")
    print("=" * 60)
    
    system = ParticleSystem()
    
    # Create 5 particle types with varying copy numbers
    particle_types = []
    copy_numbers = [10, 20, 30, 25, 15]
    radii = [1.0, 0.8, 1.2, 0.9, 1.1]
    
    for i, (copy_num, radius) in enumerate(zip(copy_numbers, radii)):
        identity = chr(ord('A') + i)  # A, B, C, D, E
        ptype = system.add_particle_type(identity, radius=radius, copy_number=copy_num)
        particle_types.append(ptype)
    
    # Generate random coordinates for all particles
    np.random.seed(123)
    for ptype in particle_types:
        for _ in range(ptype.copy_number):
            coords = np.random.uniform(-10, 10, size=3)
            system.add_particle(ptype, coords)
    
    print(system)
    print(f"\nCopy numbers valid: {system.validate_copy_numbers()}")
    
    # Statistics
    coords = system.get_coordinates_array()
    print(f"\nCoordinate statistics:")
    print(f"  Mean: ({coords[:, 0].mean():.2f}, {coords[:, 1].mean():.2f}, {coords[:, 2].mean():.2f})")
    print(f"  Std:  ({coords[:, 0].std():.2f}, {coords[:, 1].std():.2f}, {coords[:, 2].std():.2f})")
    print(f"  Min:  ({coords[:, 0].min():.2f}, {coords[:, 1].min():.2f}, {coords[:, 2].min():.2f})")
    print(f"  Max:  ({coords[:, 0].max():.2f}, {coords[:, 1].max():.2f}, {coords[:, 2].max():.2f})")
    
    return system


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PARTICLE SYSTEM EXAMPLES")
    print("Demonstrating flexibility and extensibility")
    print("=" * 60)
    
    # Run all examples
    system1 = create_custom_system_example()
    system2 = create_minimal_system_example()
    system3 = demonstrate_particle_access()
    system4 = demonstrate_large_system()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
