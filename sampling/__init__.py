"""
Sampling module with BlackJAX wrappers.
"""
from .rmh import (
    create_rmh_kernel,
    run_rmh_sampling,
    run_annealed_rmh,
    run_parallel_rmh,
)
from .smc import (
    run_tempered_smc,
    get_smc_samples,
    get_best_sample,
)

"""
sampling
========

BlackJAX-based samplers and adapters for use with IMP / JAX scoring
functions.

Public API
----------
* :mod:`smc_base_sampler` -- generic BlackJAX base-SMC runners over
  flat parameter vectors.  Used directly by toy models.
* :mod:`imp_blackjax_adapter` -- adapter that bridges IMP's structured
  JAX model (``{'xyz', 'r', 'rigid_bodies', ...}``) onto the flat
  representation that ``smc_base_sampler`` expects.
"""

from .wrapper_imp_blackjax import (
    IMPDOFSpace,
    IMPSMCAdapter,
    run_smc_on_imp_system,
)

from .smc_base_sampler import (
    run_base_smc_rmh,
    run_base_smc_hmc,
    get_smc_samples,
    get_best_sample,
    SCHEDULE_REGISTRY,
)

__all__ = [
    'create_rmh_kernel',
    'run_rmh_sampling',
    'run_annealed_rmh',
    'run_parallel_rmh',
    'run_tempered_smc',
    'get_smc_samples',
    'get_best_sample',
    "IMPDOFSpace",
    "IMPSMCAdapter",
    "run_smc_on_imp_system",
    "run_base_smc_rmh",
    "run_base_smc_hmc",
    "SCHEDULE_REGISTRY",

]
