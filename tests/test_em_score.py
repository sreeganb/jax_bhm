#------------------------
# Test the EM score and make sure it computes to a small score for ideal structure 
#-------------------------
#!/usr/bin/env python
"""
Compute CCC between model density (ideal coords) and experimental density.
"""

import argparse
import numpy as np
import mrcfile
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

from representation.particle_system import get_ideal_coords
from scoring.em_score import create_em_config_from_mrcfile, calculate_ccc_score


def main():
    parser = argparse.ArgumentParser(description="CCC for ideal coords vs experimental density")
    parser.add_argument("--mrc", type=str, default="output/simulated_target_density.mrc",
                        help="Path to experimental density MRC file")
    parser.add_argument("--resolution", type=float, default=50.0,
                        help="Map resolution (Å)")
    args = parser.parse_args()

    # Particle types and radii (same as in your SMC setup)
    types_config = {
        'A': {'radius': 24.0, 'copy': 8},
        'B': {'radius': 14.0, 'copy': 8},
        'C': {'radius': 16.0, 'copy': 16},
    }

    # Ideal coordinates (dict of arrays)
    ideal = get_ideal_coords()
    
    do_perturb = True
    if do_perturb:
        # Slightly perturb ideal coords to avoid perfect score
        rng = np.random.default_rng(1234)
        for k in ideal:
            ideal[k] = ideal[k] + rng.normal(scale=0.0, size=ideal[k].shape)
    # 
    identity_order = sorted(types_config.keys())

    # Flatten coords into (N, 3) in consistent order
    coords = np.concatenate([np.array(ideal[k]) for k in identity_order], axis=0)

    # Radii array in same order
    radii = np.concatenate([
        np.full((types_config[k]['copy'],), types_config[k]['radius'])
        for k in identity_order
    ])

    # Load experimental density
    with mrcfile.open(args.mrc, mode='r') as mrc:
        em_config = create_em_config_from_mrcfile(mrc, resolution=args.resolution)

    # Compute CCC
    ccc = calculate_ccc_score(coords, radii, em_config)
    print(f"CCC (ideal vs experimental): {ccc:.6f}")


if __name__ == "__main__":
    main()
