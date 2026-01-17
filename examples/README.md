# Examples

This directory contains example scripts demonstrating JAX-BHM usage.

## basic_sampling.py

A simple, educational example that demonstrates:
- Creating a linear protein chain
- Defining distance restraints
- Running MCMC sampling with NUTS
- Analyzing results

**Run it:**
```bash
python examples/basic_sampling.py
```

**Expected output:** Successfully converges to structures satisfying the distance restraint (10.0 Å between end atoms).

## simple_sampling.py

A more complex example with multiple restraints, demonstrating:
- Multiple distance restraints
- Excluded volume constraints
- More realistic protein folding scenario

**Run it:**
```bash
python examples/simple_sampling.py
```

## Key Concepts Demonstrated

### 1. Distance Restraints
Both examples show how to define distance restraints between atoms:
```python
restraints = jnp.array([
    [atom_i, atom_j, target_distance, weight]
])
```

### 2. Energy Functions
Create energy functions combining multiple scoring terms:
```python
energy_fn = create_energy_fn(
    distance_restraints=restraints,
    use_excluded_volume=True
)
```

### 3. MCMC Sampling
Sample conformations using BlackJax:
```python
samples, info = sample_structure(
    initial_structure=structure,
    energy_fn=energy_fn,
    n_samples=500,
    algorithm="nuts"
)
```

### 4. Analysis
Extract and analyze results:
```python
stats = calculate_sampling_statistics(samples, energy_fn)
best_structure, best_energy = get_best_structure(samples, energy_fn)
```

## Tips for Custom Examples

1. **Temperature**: Higher temperature (e.g., 10.0-100.0) helps exploration in complex energy landscapes
2. **Step size**: Start with 0.1-0.5; adjust based on acceptance rate
3. **Warmup samples**: Use 200-500 for adaptation
4. **Excluded volume**: Disable initially for simpler problems; add later for realism
5. **Force constants**: Lower values (1.0-10.0) for softer restraints; higher (100+) for stricter
