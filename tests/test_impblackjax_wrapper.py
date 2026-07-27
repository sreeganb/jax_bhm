import IMP
import IMP.atom
import IMP.core
import IMP.algebra
import IMP.pmi
import IMP.pmi.topology
import jax

from sampling.wrapper_imp_blackjax import (
	build_flexible_bead_rmh_wrapper,
	run_rmh_on_imp_system,
)


def main():
	m = IMP.Model()
	p1, p2 = IMP.Particle(m), IMP.Particle(m)
	IMP.core.XYZR.setup_particle(p1, IMP.algebra.Sphere3D(IMP.algebra.Vector3D(0, 0, 0), 12.0))
	IMP.core.XYZR.setup_particle(p2, IMP.algebra.Sphere3D(IMP.algebra.Vector3D(20, 0, 0), 6.0))
	ps = IMP.core.HarmonicDistancePairScore(25.0, 1.0)
	r = IMP.container.PairsRestraint(ps, IMP.container.ListPairContainer(m, [(p1.get_index(), p2.get_index())]))
	sf = IMP.core.RestraintsScoringFunction([r])
	parameter_space, log_posterior = build_flexible_bead_rmh_wrapper(
		model=m,
		scoring_function=sf,
		flexible_particle_indices=[int(p1.get_index()), int(p2.get_index())],
		temperature=1.0,
	)
	res = run_rmh_on_imp_system(
		log_prob_fn=log_posterior,
		initial_position=parameter_space.pack(),
		rng_key=jax.random.PRNGKey(0),
		n_steps=1000,
		sigma=3.0,
		sync_fn=lambda flat: parameter_space.unpack(flat),
		verbose=False,
	)
	print("final log posterior:", float(res.log_probs[-1]))
	print("best position:\n", res.positions[int(res.log_probs.argmax())])


if __name__ == "__main__":
	main()
