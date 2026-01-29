#------------------
# Make sure that the pair score works as it should using the ideal system 
#-----------------
#!/usr/bin/env python
#------------------
# Make sure that the pair score works as it should using the ideal system 
#-----------------

import numpy as np
from representation.particle_system import get_ideal_coords


def compute_distances_between(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1) + 1e-8)


def gaussian_nll_matrix(dists: np.ndarray, target_dist: float, sigma: float) -> np.ndarray:
    return 0.5 * ((dists - target_dist) / sigma) ** 2 + 0.5 * np.log(2.0 * np.pi * sigma**2)


def union_argmin_pairs(score_matrix: np.ndarray, same_type: bool):
    """
    Returns:
        pair_mask (bool matrix), list of (i, j) pairs
    """
    n_a, n_b = score_matrix.shape
    INF = 1e12

    if same_type:
        tri_mask = np.triu(np.ones((n_a, n_b)), k=1)
        masked = np.where(tri_mask > 0, score_matrix, INF)

        row_best_j = np.argmin(masked, axis=1)
        row_best_vals = np.min(masked, axis=1)
        row_valid = row_best_vals < INF

        col_best_i = np.argmin(masked, axis=0)
        col_best_vals = np.min(masked, axis=0)
        col_valid = col_best_vals < INF

        row_sel = np.zeros((n_a, n_b), dtype=bool)
        row_sel[np.arange(n_a), row_best_j] = row_valid

        col_sel = np.zeros((n_a, n_b), dtype=bool)
        col_sel[col_best_i, np.arange(n_b)] = col_valid

        union_mask = row_sel | col_sel
    else:
        row_best_j = np.argmin(score_matrix, axis=1)
        row_valid = np.isfinite(np.min(score_matrix, axis=1))

        col_best_i = np.argmin(score_matrix, axis=0)
        col_valid = np.isfinite(np.min(score_matrix, axis=0))

        row_sel = np.zeros((n_a, n_b), dtype=bool)
        row_sel[np.arange(n_a), row_best_j] = row_valid

        col_sel = np.zeros((n_a, n_b), dtype=bool)
        col_sel[col_best_i, np.arange(n_b)] = col_valid

        union_mask = row_sel | col_sel

    pairs = list(zip(*np.where(union_mask)))
    return union_mask, pairs


def main():
    # Ideal coords
    coords_by_type = get_ideal_coords()
    coords_by_type = {k: np.array(v) for k, v in coords_by_type.items()}

    # Target distances and nuisance sigmas
    target_dists = {'AA': 48.2, 'AB': 38.5, 'BC': 34.0}
    nuisance_params = {'AA': 1.6, 'AB': 1.4, 'BC': 1.0}

    total_pair_score = 0.0

    for pair_key, target_dist in target_dists.items():
        type1, type2 = pair_key[0], pair_key[1]
        sigma = nuisance_params[pair_key]

        coords_a = coords_by_type[type1]
        coords_b = coords_by_type[type2]

        dists = compute_distances_between(coords_a, coords_b)
        nll_matrix = gaussian_nll_matrix(dists, target_dist, sigma)

        same_type = (type1 == type2)
        mask, pairs = union_argmin_pairs(nll_matrix, same_type)

        pair_score = np.sum(nll_matrix[mask])
        total_pair_score += pair_score

        print(f"\nPair type {pair_key}:")
        print(f"  Scored pairs: {len(pairs)}")
        print(f"  Pair score (NLL sum): {pair_score:.4f}")

        # Print which particles were scored
        for i, j in pairs:
            coord_i = coords_a[i]
            coord_j = coords_b[j]
            print(
                f"    {type1}{i+1} <-> {type2}{j+1} | "
                f"coords_i={coord_i} coords_j={coord_j} | "
                f"nll={nll_matrix[i, j]:.4f}"
            )

    print(f"\nTOTAL pair score (NLL sum): {total_pair_score:.4f}")


if __name__ == "__main__":
    main()