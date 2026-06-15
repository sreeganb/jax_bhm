import IMP
import jax

from sampling.wrapper_imp_blackjax import IMPDOFSpace, IMPSMCAdapter, run_smc_on_imp_system


def main():
	m = IMP.Model()
	p1, p2 = IMP.Particle(m), IMP.Particle(m)
	IMP.core.XYZR.setup_particle(p1, IMP.algebra.Sphere3D(IMP.algebra.Vector3D(0, 0, 0), 12.0))
	IMP.core.XYZR.setup_particle(p2, IMP.algebra.Sphere3D(IMP.algebra.Vector3D(20, 0, 0), 6.0))
	ps = IMP.core.HarmonicDistancePairScore(25.0, 1.0)
	r = IMP.container.PairsRestraint(ps, IMP.container.ListPairContainer(m, [(p1.get_index(), p2.get_index())]))
	sf = IMP.core.RestraintsScoringFunction([r])
	ji = sf._get_jax()
	adapter = IMPSMCAdapter(IMPDOFSpace.from_imp(None, ji, ji.get_jax_model()), ji.score_func, kT=1.0, box_half_width=50.0)
	state, _, best_pos, best_scores, _ = run_smc_on_imp_system(adapter, jax.random.PRNGKey(0), n_particles=32, n_temperature_steps=8, kernel="rmh", rmh_sigma=3.0, n_mcmc_steps=1000, init_from_prior=True, verbose=False)
	print("final score:", float(best_scores[-1]))
	print("best xyz:\n", adapter.decode_xyz(best_pos[-1]))


if __name__ == "__main__":
	main()
