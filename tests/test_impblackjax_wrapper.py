import IMP
import jax
import IMP.core

import IMP
import IMP.atom
import IMP.core
import IMP.algebra
import IMP.pmi
import IMP.pmi.topology

from sampling.wrapper_imp_blackjax import IMPDOFSpace, IMPSMCAdapter, run_smc_on_imp_system, run_rmh_on_imp_system


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
	state, _, best_pos, best_scores, _ = run_rmh_on_imp_system(adapter, sample_key=jax.random.PRNGKey(0),
                                                            rmh_sigma=3.0,
                                                            n_mcmc_steps=1000,
															imp_model=m,
                                                            save_rmf3_path="output.rmf3",
                                                            verbose=False)
	print("final score:", float(best_scores[-1]))
	print("best xyz:\n", adapter.decode_xyz(best_pos[-1]))


if __name__ == "__main__":
	main()
