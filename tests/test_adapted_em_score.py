"""
Tests for scoring.adapted_em_score — adapted EM density scoring module.

All tests are self-contained (no MRC files needed).  They use the
particle system from representation.particle_system to build a
realistic synthetic target density.

Run:
    pytest tests/test_adapted_em_score.py -v
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from scoring.adapted_em_score import (
    resolution_to_sigma,
    pairwise_correlation_jax,
    calc_projection_jax,
    calculate_ccc_jax,
    calculate_ccc_score,
    create_em_config_from_coords,
    create_em_config_from_arrays,
    create_em_scoring_model,
    create_gaussian_ccc_log_likelihood,
    create_exponential_distance_log_prior,
    create_soft_box_log_prior,
    generate_density_map,
    diagnose_model,
    EMConfig,
)
from representation.particle_system import get_ideal_coords, ParticleSystem


# ── Fixtures ───────────────────────────────────────────────────────────

TYPES = {
    "A": {"radius": 24.0, "copy": 8},
    "B": {"radius": 14.0, "copy": 8},
    "C": {"radius": 16.0, "copy": 16},
}
RESOLUTION = 43.0
VOXEL_SIZE = 4.0
BOX_SIZE = 300.0


@pytest.fixture(scope="module")
def particle_system():
    """Create the canonical 32-particle system."""
    ideal = get_ideal_coords()
    return ParticleSystem.create(TYPES, ideal)


@pytest.fixture(scope="module")
def ideal_coords_and_radii(particle_system):
    """Flat coords (N, 3) and radii (N,) in identity order."""
    flat = particle_system.flatten(particle_system.ideal_coords)
    coords = flat.reshape(-1, 3)
    radii = particle_system.get_flat_radii()
    return np.array(coords), np.array(radii)


@pytest.fixture(scope="module")
def em_config(ideal_coords_and_radii):
    """EMConfig generated from ideal coordinates.

    Because target and model maps use the same ``calc_projection_jax``
    code path, self-CCC should be exactly 1.0.
    """
    coords, radii = ideal_coords_and_radii
    return create_em_config_from_coords(
        coords, radii, RESOLUTION, VOXEL_SIZE, BOX_SIZE
    )


# ── 1. resolution_to_sigma ────────────────────────────────────────────

class TestResolutionToSigma:
    def test_positive(self):
        """Sigma is positive for physical inputs."""
        assert resolution_to_sigma(43.0, 4.0) > 0

    def test_proportional_to_resolution(self):
        """Doubling resolution doubles sigma."""
        s1 = resolution_to_sigma(20.0, 4.0)
        s2 = resolution_to_sigma(40.0, 4.0)
        assert abs(s2 / s1 - 2.0) < 1e-6

    def test_inversely_proportional_to_pixel(self):
        """Doubling pixel size halves sigma (fewer pixels to cover)."""
        s1 = resolution_to_sigma(40.0, 2.0)
        s2 = resolution_to_sigma(40.0, 4.0)
        assert abs(s1 / s2 - 2.0) < 1e-6


# ── 2. Pearson correlation ────────────────────────────────────────────

class TestPairwiseCorrelation:
    def test_self_correlation_is_one(self):
        a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert float(pairwise_correlation_jax(a, a)) == pytest.approx(1.0, abs=1e-5)

    def test_anticorrelated(self):
        a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = jnp.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert float(pairwise_correlation_jax(a, b)) == pytest.approx(-1.0, abs=1e-5)

    def test_zero_vector_returns_zero(self):
        a = jnp.array([1.0, 2.0, 3.0])
        b = jnp.zeros(3)
        assert float(pairwise_correlation_jax(a, b)) == pytest.approx(0.0, abs=1e-8)

    def test_uncorrelated(self):
        rng = np.random.default_rng(42)
        a = jnp.array(rng.standard_normal(10_000))
        b = jnp.array(rng.standard_normal(10_000))
        assert abs(float(pairwise_correlation_jax(a, b))) < 0.05


# ── 3. Density projection ────────────────────────────────────────────

class TestDensityProjection:
    def test_output_shape(self, ideal_coords_and_radii):
        """Output shape is (nz, ny, nx)."""
        coords, radii = ideal_coords_and_radii
        grid_dim = 50
        half = grid_dim * VOXEL_SIZE / 2
        bins_1d = jnp.linspace(-half, half, grid_dim + 1)
        bins = (bins_1d, bins_1d, bins_1d)
        weights = jnp.array(radii) ** 3
        density = calc_projection_jax(jnp.array(coords), weights, bins, RESOLUTION)
        assert density.shape == (grid_dim, grid_dim, grid_dim)

    def test_total_mass_approximately_conserved(self, ideal_coords_and_radii):
        """After Gaussian blur, total mass ≈ sum of weights."""
        coords, radii = ideal_coords_and_radii
        grid_dim = 80
        half = grid_dim * VOXEL_SIZE / 2
        bins_1d = jnp.linspace(-half, half, grid_dim + 1)
        bins = (bins_1d, bins_1d, bins_1d)
        weights = jnp.array(radii) ** 3
        density = calc_projection_jax(jnp.array(coords), weights, bins, RESOLUTION)
        # fftconvolve preserves integral (kernel sums to 1)
        ratio = float(jnp.sum(density)) / float(jnp.sum(weights))
        assert ratio == pytest.approx(1.0, abs=0.15)

    def test_density_is_nonneg_near_particles(self, ideal_coords_and_radii):
        """Blurred density should be non-negative."""
        coords, radii = ideal_coords_and_radii
        grid_dim = 60
        half = grid_dim * VOXEL_SIZE / 2
        bins_1d = jnp.linspace(-half, half, grid_dim + 1)
        bins = (bins_1d, bins_1d, bins_1d)
        weights = jnp.array(radii) ** 3
        density = calc_projection_jax(jnp.array(coords), weights, bins, RESOLUTION)
        # Minor negative values from FFT ringing allowed, but not large ones
        assert float(jnp.min(density)) > -0.1 * float(jnp.max(density))


# ── 4. Self-CCC = 1.0 ────────────────────────────────────────────────

class TestSelfCCC:
    def test_self_ccc_from_coords(self, ideal_coords_and_radii, em_config):
        """Config generated from the same coords should yield CCC = 1.0."""
        coords, radii = ideal_coords_and_radii
        ccc = calculate_ccc_score(coords, radii, em_config)
        assert ccc == pytest.approx(1.0, abs=1e-4)

    def test_self_ccc_from_arrays(self, ideal_coords_and_radii):
        """Config from arrays (directly from generate_density_map)."""
        coords, radii = ideal_coords_and_radii
        density_np, bins = generate_density_map(
            coords, radii, RESOLUTION, VOXEL_SIZE, BOX_SIZE
        )
        config = create_em_config_from_arrays(
            density_np, VOXEL_SIZE, RESOLUTION,
        )
        ccc = calculate_ccc_score(coords, radii, config)
        assert ccc == pytest.approx(1.0, abs=1e-4)


# ── 5. Perturbation monotonicity ─────────────────────────────────────

class TestPerturbationMonotonicity:
    """Larger perturbations → lower CCC (monotone scoring signal)."""

    @pytest.mark.parametrize("sigma_perturb", [5.0, 20.0, 50.0])
    def test_monotone_decrease(self, ideal_coords_and_radii, em_config, sigma_perturb):
        coords, radii = ideal_coords_and_radii
        rng = np.random.default_rng(12)
        perturbed = coords + rng.normal(scale=sigma_perturb, size=coords.shape)
        ccc = calculate_ccc_score(perturbed, radii, em_config)
        assert ccc < 1.0, f"CCC should decrease for sigma={sigma_perturb}"

    def test_ordering_small_vs_large(self, ideal_coords_and_radii, em_config):
        coords, radii = ideal_coords_and_radii
        rng = np.random.default_rng(42)
        small_p = coords + rng.normal(scale=5.0, size=coords.shape)
        large_p = coords + rng.normal(scale=30.0, size=coords.shape)
        ccc_small = calculate_ccc_score(small_p, radii, em_config)
        ccc_large = calculate_ccc_score(large_p, radii, em_config)
        assert ccc_small > ccc_large, "Small perturbation should give higher CCC"


# ── 6. Mass weighting ────────────────────────────────────────────────

class TestMassWeighting:
    def test_weights_are_radius_cubed(self, ideal_coords_and_radii, em_config):
        """Verify the mass = r^3 convention inside calculate_ccc_jax."""
        coords, radii = ideal_coords_and_radii
        # Compute CCC using the public API (which does radii**3 internally)
        ccc_public = calculate_ccc_score(coords, radii, em_config)
        # Manually compute with explicit weights
        weights = jnp.array(radii) ** 3
        bins = (em_config.bins_x, em_config.bins_y, em_config.bins_z)
        proj = calc_projection_jax(jnp.array(coords), weights, bins, em_config.resolution)
        ccc_manual = float(pairwise_correlation_jax(
            proj.flatten(), em_config.target_data.flatten()
        ))
        assert abs(ccc_public - ccc_manual) < 1e-5


# ── 7. Likelihood ────────────────────────────────────────────────────

class TestLikelihood:
    def test_max_at_ideal(self, ideal_coords_and_radii, em_config):
        """Log-likelihood is maximised (= 0) at ideal coords."""
        coords, radii = ideal_coords_and_radii
        ll_fn = create_gaussian_ccc_log_likelihood(em_config, radii, sigma_ccc=0.3)
        ll = float(ll_fn(jnp.array(coords.flatten())))
        assert ll == pytest.approx(0.0, abs=1e-4)

    def test_decreases_with_perturbation(self, ideal_coords_and_radii, em_config):
        coords, radii = ideal_coords_and_radii
        ll_fn = create_gaussian_ccc_log_likelihood(em_config, radii, sigma_ccc=0.3)
        rng = np.random.default_rng(99)
        perturbed = coords + rng.normal(scale=20.0, size=coords.shape)
        ll_ideal = float(ll_fn(jnp.array(coords.flatten())))
        ll_perturbed = float(ll_fn(jnp.array(perturbed.flatten())))
        assert ll_ideal > ll_perturbed

    def test_sigma_controls_sharpness(self, ideal_coords_and_radii, em_config):
        coords, radii = ideal_coords_and_radii
        rng = np.random.default_rng(7)
        perturbed = coords + rng.normal(scale=15.0, size=coords.shape)
        flat_p = jnp.array(perturbed.flatten())
        ll_sharp = float(
            create_gaussian_ccc_log_likelihood(em_config, radii, sigma_ccc=0.1)(flat_p)
        )
        ll_broad = float(
            create_gaussian_ccc_log_likelihood(em_config, radii, sigma_ccc=1.0)(flat_p)
        )
        # Sharper sigma → more negative (penalises more)
        assert ll_sharp < ll_broad


# ── 8. Priors ────────────────────────────────────────────────────────

class TestPriors:
    def test_box_prior_zero_inside(self):
        """Box prior = 0 when all coords are inside."""
        box_fn = create_soft_box_log_prior(500.0, steepness=1.0)
        x = jnp.array([10.0, -10.0, 50.0, -50.0])
        assert float(box_fn(x)) == pytest.approx(0.0, abs=1e-8)

    def test_box_prior_negative_outside(self):
        box_fn = create_soft_box_log_prior(500.0, steepness=1.0)
        x = jnp.array([510.0, -600.0, 0.0])
        assert float(box_fn(x)) < 0.0

    def test_attract_prior_negative(self, em_config):
        """Attraction prior is always ≤ 0."""
        attract_fn = create_exponential_distance_log_prior(em_config, 0.01)
        x = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        assert float(attract_fn(x)) <= 0.0

    def test_attract_increases_closer_to_com(self, em_config):
        """Closer to COM → higher (less negative) prior."""
        attract_fn = create_exponential_distance_log_prior(em_config, 0.01)
        com = np.array(em_config.density_com)
        near = jnp.array(np.tile(com + 1.0, 2))
        far  = jnp.array(np.tile(com + 100.0, 2))
        assert float(attract_fn(near)) > float(attract_fn(far))


# ── 9. Combined scoring model ────────────────────────────────────────

class TestScoringModel:
    def test_returns_three_callables(self, ideal_coords_and_radii, em_config):
        coords, radii = ideal_coords_and_radii
        prior, lik, prob = create_em_scoring_model(em_config, radii)
        flat = jnp.array(coords.flatten())
        assert np.isfinite(float(prior(flat)))
        assert np.isfinite(float(lik(flat)))
        assert np.isfinite(float(prob(flat)))

    def test_prob_equals_prior_plus_lik(self, ideal_coords_and_radii, em_config):
        coords, radii = ideal_coords_and_radii
        prior_fn, lik_fn, prob_fn = create_em_scoring_model(em_config, radii)
        flat = jnp.array(coords.flatten())
        expected = float(prior_fn(flat)) + float(lik_fn(flat))
        got = float(prob_fn(flat))
        assert got == pytest.approx(expected, abs=1e-5)


# ── 10. Diagnostics ──────────────────────────────────────────────────

class TestDiagnose:
    def test_ideal_diagnostics(self, ideal_coords_and_radii, em_config):
        coords, radii = ideal_coords_and_radii
        diag = diagnose_model(
            jnp.array(coords.flatten()), em_config, radii,
            sigma_ccc=0.3, lambda_attract=0.001, box_size=500.0,
        )
        assert diag["ccc"] == pytest.approx(1.0, abs=1e-3)
        assert diag["mismatch"] == pytest.approx(0.0, abs=1e-3)
        assert diag["log_lik_ccc"] == pytest.approx(0.0, abs=1e-3)
        assert diag["log_prior_box"] == pytest.approx(0.0, abs=1e-8)
        assert np.isfinite(diag["log_posterior"])


# ── 11. EMConfig from arrays ─────────────────────────────────────────

class TestEMConfigFromArrays:
    def test_roundtrip(self, ideal_coords_and_radii):
        coords, radii = ideal_coords_and_radii
        density_np, bins = generate_density_map(
            coords, radii, RESOLUTION, VOXEL_SIZE, BOX_SIZE,
        )
        config = create_em_config_from_arrays(density_np, VOXEL_SIZE, RESOLUTION)
        assert config.target_data.shape == density_np.shape
        assert float(config.resolution) == RESOLUTION

    def test_density_com_finite(self, em_config):
        assert np.all(np.isfinite(np.array(em_config.density_com)))
