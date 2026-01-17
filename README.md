# jax_bhm

A Python framework for simulating particle systems representing proteins as spheres in 3D space.

## Overview

This repository provides a flexible system for defining and managing protein particles in 3D space. Each particle type is identified by a single character (A, B, C, etc.) and has properties including:
- **Identity**: A single character identifier
- **Radius**: The radius of the spherical particle
- **Copy Number**: The number of instances of this particle type
- **Coordinates**: 3D position (x, y, z) for each particle instance

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Creating a Simple Particle System

```python
from particle_system import ParticleSystem
import numpy as np

# Create a new system
system = ParticleSystem()

# Add particle types
particle_A = system.add_particle_type('A', radius=1.0, copy_number=3)
particle_B = system.add_particle_type('B', radius=1.5, copy_number=2)

# Add particle instances with coordinates
system.add_particle(particle_A, np.array([0.0, 0.0, 0.0]))
system.add_particle(particle_A, np.array([2.0, 0.0, 0.0]))
system.add_particle(particle_A, np.array([4.0, 0.0, 0.0]))
system.add_particle(particle_B, np.array([1.0, 2.0, 0.0]))
system.add_particle(particle_B, np.array([3.0, 2.0, 0.0]))

# Validate and display
print(system)
print(f"Valid: {system.validate_copy_numbers()}")
```

### Using the Ideal Ground Truth System

The repository includes a predefined ideal ground truth structure with:
- **Particle A**: 8 instances with radius 1.0
- **Particle B**: 8 instances with radius 1.2
- **Particle C**: 16 instances with radius 0.8

```python
from particle_system import create_ideal_ground_truth_system

# Create the ideal system
ideal_system = create_ideal_ground_truth_system()

# Access particles by type
particles_A = ideal_system.get_particles_by_type('A')
print(f"Particle A count: {len(particles_A)}")

# Get all coordinates as a numpy array
all_coords = ideal_system.get_coordinates_array()
print(f"Coordinate matrix shape: {all_coords.shape}")
```

### Running the Demo

```bash
python particle_system.py
```

This will display the ideal ground truth system with all particle types and their coordinates.

## Testing

Run the test suite to verify functionality:

```bash
python test_particle_system.py
```

## Module Structure

- **`ParticleType`**: Defines a type of particle with identity, radius, and copy number
- **`Particle`**: Represents an individual particle instance with coordinates
- **`ParticleSystem`**: Manages collections of particle types and instances
- **`create_ideal_ground_truth_system()`**: Creates the reference configuration with A=8, B=8, C=16

## Key Features

- ✅ Flexible particle system definition
- ✅ Support for arbitrary particle types and copy numbers
- ✅ 3D coordinate management with NumPy arrays
- ✅ Validation of copy numbers against defined types
- ✅ Predefined ideal ground truth structure
- ✅ Easy access to particles by type
- ✅ Coordinate array extraction for analysis

## Requirements

- Python 3.7+
- NumPy >= 1.20.0