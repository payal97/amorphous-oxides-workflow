import numpy as np
from agox.acquisitors import LCBPenaltyAcquisitor
from agox.collectors import ParallelCollector
from agox.databases import Database
from agox.environments import Environment
from agox.evaluators import SinglePointEvaluator
from agox.generators import RandomGenerator, RattleGenerator
from agox.main import AGOX
from agox.models import GPR
from agox.models.descriptors import Fingerprint, VoronoiSite
from agox.models.GPR.kernels import RBF
from agox.models.GPR.kernels import Constant as C
from agox.models.GPR.kernels import Noise
from agox.models.GPR.priors import Repulsive
from agox.postprocessors import (DisjointFilteringPostprocess,
                                 ParallelRelaxPostprocess,
                                 SurfaceCenteringPostprocess)
from agox.samplers import KMeansSampler
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator


def run_agox(nanocluster_stoichiometry: str,
             num_iterations: int,
             target_calc: Calculator,
             template: Atoms,
             template_size: tuple[int, int, int],
             index: int):
    """Run global optimization using AGOX, as described in the "Global
    optimization" section of the manuscript.

    Parameters
    ----------
    nanocluster_stoichiometry : str
        Stoichiometry of nanocluster to search for.
    num_iterations : int
        Number of AGOX iterations to run.
    target_calc : Calculator
        Target potential configuration.
    template : Atoms
        Metal surface to place nanocluster atoms on.
    template_size : tuple[int, int, int]
        Number of atoms (a, b, c) in the metal surface.
    index : int
        Index of parallel run.
    """

    # --- AGOX loop setup ---
    # database
    database = Database(filename=f'db_{index:03d}.db', order=8)

    # environment
    top_layer_indices = [atom.index for atom in template if atom.tag == 1]

    confinement_cell = template.get_cell() * np.array([0.7, 0.7, 0])
    confinement_cell[2, 2] = 7.0
    confinement_corner = np.dot(template.get_cell().T, np.array([0.15, 0.15, 0]))
    confinement_corner[2] = np.max(template.positions[:, 2]) - 0.5

    environment = Environment(template=template,
                              symbols=nanocluster_stoichiometry,
                              confinement_cell=confinement_cell,
                              confinement_corner=confinement_corner)

    # model
    descriptor = Fingerprint(environment=environment)

    beta = 0.01
    sigma_noise = 0.01
    k0 = C(beta, (beta, beta)) * RBF()
    k1 = C(1 - beta, (1 - beta, 1 - beta)) * RBF()
    kernel = C(5000, (1, 1e5)) * (k0 + k1) + Noise(sigma_noise, (sigma_noise, sigma_noise))

    model = GPR(descriptor=descriptor,
                kernel=kernel,
                prior=Repulsive(),
                order=0)
    model.attach_to_database(database)

    # sampler
    sampler = KMeansSampler(descriptor=descriptor,
                            sample_size=10,
                            max_energy=25,
                            order=1)
    sampler.attach_to_database(database)

    # generators
    random_generator = RandomGenerator(**environment.get_confinement())
    rattle_generator = RattleGenerator(**environment.get_confinement())
    generators = [random_generator, rattle_generator]
    num_candidates = {0: [16, 32]}

    collector = ParallelCollector(generators=generators,
                                  sampler=sampler,
                                  environment=environment,
                                  num_candidates=num_candidates,
                                  order=2)

    # acquisitor
    graph_indices = top_layer_indices + list(environment.get_missing_indices())
    graph_descriptor = VoronoiSite(site_mapping='fcc111',
                                   template=template,
                                   indices=graph_indices,
                                   n_points=0,
                                   environment=environment)

    acquisitor = LCBPenaltyAcquisitor(model=model,
                                      descriptor=graph_descriptor,
                                      penalty_scale=1.0,
                                      kappa=2.0,
                                      order=6)
    acquisitor.attach_to_database(database)

    # relaxer
    relaxer = ParallelRelaxPostprocess(model=acquisitor.get_acquisition_calculator(),
                                       constraints=environment.get_constraints(),
                                       start_relax=2,
                                       optimizer_run_kwargs={'steps': 100, 'fmax': 0.05},
                                       order=3)

    # evaluator
    evaluator = SinglePointEvaluator(target_calc,
                                     order=7)

    # postprocessors
    centerer = SurfaceCenteringPostprocess(template_size,
                                           order=4)

    filterer = DisjointFilteringPostprocess(order=5)

    # run
    agox = AGOX(collector, relaxer, centerer, filterer,
                acquisitor, evaluator, database, seed=index)
    agox.run(N_iterations=num_iterations)
