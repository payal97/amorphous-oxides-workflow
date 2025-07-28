from typing import Callable

import numpy as np
from agox.acquisitors import LowerConfidenceBoundAcquisitor
from agox.candidates import CandidateBaseClass
from agox.collectors import ParallelCollector
from agox.databases import Database
from agox.environments import Environment
from agox.evaluators import LocalOptimizationEvaluator
from agox.generators import RandomGenerator, RattleGenerator
from agox.main import AGOX
from agox.models import GPR
from agox.models.descriptors import Fingerprint, VoronoiSite
from agox.models.GPR.kernels import RBF
from agox.models.GPR.kernels import Constant as C
from agox.models.GPR.kernels import Noise
from agox.models.GPR.priors import Repulsive
from agox.postprocessors import (ParallelRelaxPostprocess,
                                 SurfaceCenteringPostprocess)
from agox.samplers import KMeansSampler
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator


def run_agox(symbols: str,
             num_iterations: int,
             target_calc: Calculator,
             check_callback : Callable[[CandidateBaseClass], None],
             template: Atoms,
             index: int):
    """Run global optimization using AGOX, as described in the "Global
    optimization" section of the manuscript.

    Parameters
    ----------
    symbols : str
        Stoichiometry to search for.
    num_iterations : int
        Number of AGOX iterations to run.
    target_calc : Calculator
        Target potential configuration.
    template : Atoms
        Cell to place atoms in.
    index : int
        Index of parallel run.
    """

    # --- AGOX loop setup ---
    # database
    database = Database(filename=f'db_{index:03d}.db', order=5)

    # environment
    confinement_cell = template.cell.copy()
    confinement_cell[2, 2] = 18.0
    confinement_corner = np.array([0, 0, 0])

    environment = Environment(template=template,
                              symbols=symbols,
                              confinement_cell=confinement_cell,
                              confinement_corner=confinement_corner,
                              box_constraint_pbc=[True, True, False])

    # model
    descriptor = Fingerprint(environment=environment)

    beta = 0.01
    sigma_noise = 0.01
    k0 = C(beta, (beta, beta)) * RBF()
    k1 = C(1 - beta, (1 - beta, 1 - beta)) * RBF()
    kernel = C(5000, (1, 1e5)) * (k0 + k1) + Noise(sigma_noise, (sigma_noise, sigma_noise))

    model = GPR(descriptor=descriptor,
                kernel=kernel,
                database=database,
                prior=Repulsive(),
                order=0)

    # sampler
    sampler = KMeansSampler(descriptor=descriptor,
                            database=database,
                            sample_size=10,
                            max_energy=5)

    # generators
    random_generator = RandomGenerator(**environment.get_confinement())
    rattle_generator = RattleGenerator(**environment.get_confinement())
    generators = [random_generator, rattle_generator]

    num_candidates = {0: [40, 0], 5: [40, 160]}

    collector = ParallelCollector(generators=generators,
                                  sampler=sampler,
                                  environment=environment,
                                  num_candidates=num_candidates,
                                  order=1)

    # acquisitor
    acquisitor = LowerConfidenceBoundAcquisitor(model=model,
                                                kappa=2.0,
                                                order=3)

    # relaxer
    relaxer = ParallelRelaxPostprocess(model=acquisitor.get_acquisition_calculator(),
                                       constraints=environment.get_constraints(),
                                       start_relax=5,
                                       optimizer_run_kwargs={'steps': 100, 'fmax': 0.05},
                                       order=2)

    # evaluator
    evaluator = LocalOptimizationEvaluator(target_calc,
                                           check_callback=check_callback,
                                           gets={"get_key": "prioritized_candidates"},
                                           optimizer_kwargs={"logfile": None},
                                           optimizer_run_kwargs={"fmax": 0.05, "steps": 1},
                                           constraints=environment.get_constraints(),
                                           store_trajectory=True,
                                           order=4)

    # run
    agox = AGOX(collector, relaxer, acquisitor, evaluator, database, seed=index)
    agox.run(N_iterations=num_iterations)
