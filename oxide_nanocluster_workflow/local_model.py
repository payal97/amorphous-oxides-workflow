import json
import os
from contextlib import redirect_stdout
from os import PathLike
from pathlib import Path

import numpy as np
from agox.candidates import StandardCandidate
from agox.models.descriptors import SOAP
from agox.models.GPR import SparseGPR
from agox.models.GPR.kernels import RBF
from agox.models.GPR.kernels import Constant as C
from agox.models.GPR.priors.repulsive import Repulsive
from agox.postprocessors import ParallelRelaxPostprocess
from agox.utils.constraints.box_constraint import BoxConstraint
from agox.utils.sparsifiers import MBkmeans
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from scipy.optimize import minimize
from scipy.stats import gaussian_kde


def create_local_model(species: list[str]) -> SparseGPR:
    """Create an untrained local GPR surrogate model object.

    Parameters
    ----------
    species : list[str]
        List of species to include in the SOAP descriptor.

    Returns
    -------
    SparseGPR
        Untrained model object.
    """

    descriptor = SOAP.from_species(species=species,
                                   r_cut=5,
                                   nmax=3,
                                   lmax=2,
                                   sigma=1,
                                   weight=True,
                                   periodic=True)

    kernel = C(1, (1, 100)) * RBF(30, (20, 40))

    model = SparseGPR(
        descriptor=descriptor,
        kernel=kernel,
        noise=0.01,
        prior=Repulsive(),
        sparsifier=MBkmeans(m_points=1000),
        use_ray=False
    )

    return model


def train_local_model(structures: list[Atoms], species: list[str]) -> SparseGPR:
    """Create and train a local GPR surrogate model.

    Parameters
    ----------
    structures : list[Atoms]
        Training data.
    species : list[str]
        List of species to include in the SOAP descriptor.

    Returns
    -------
    SparseGPR
        Trained model object.
    """

    print(f'Training model on {len(structures)} structures...')

    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull):
            model = create_local_model(species)
            model.train(structures)

    print('Model trained.')

    return model


def load_local_model(model_parameters_file: PathLike, species: list[str]) -> SparseGPR:
    """Create a local GPR surrogate model and load its model parameters from a
    file.

    Parameters
    ----------
    model_parameters_file : PathLike
        Path to the model parameters file.
    species : list[str]
        List of species to include in the SOAP descriptor.

    Returns
    -------
    SparseGPR
        Trained model object.
    """

    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull):
            model = create_local_model(species)
            model.load(model_parameters_file)

    print('Model loaded.')

    return model


def find_best_model_parameters(paths: list[Path]) -> str:
    """Find the model that has the lowest overall RMSE.

    Parameters
    ----------
    paths : list[Path]
        List of paths to model info files.

    Returns
    -------
    dict
        Model info dictionary for the model with the lowest overall RMSE.
    """

    print('Overall RMSEs of trained models:')

    model_data: list[dict] = []
    for path in paths:
        with open(path, 'r') as f:
            data = json.load(f)
            print(f'* {data["overall_rmse"]:.6f} eV')
            model_data.append(data)

    best_model = min(model_data, key=lambda data: data['overall_rmse'])
    model_parameters_path = best_model['model_parameters_path']

    print(f'Best model: RMSE = {best_model["overall_rmse"]:.6f} eV '
          f'(parameters: {model_parameters_path})')

    return best_model


def relax_local_model(model: SparseGPR,
                      template: Atoms,
                      structures: list[Atoms]) -> list[Atoms]:
    """Relax structures with a trained local GPR surrogate model, and reset
    structures that relax into unphysically low energies to their unrelaxed
    geometries.

    Parameters
    ----------
    model : SparseGPR
        Trained local GPR model to use as potential.
    template : Atoms
        Metal surface which nanoclusters have been placed on.
    structures : list[Atoms]
        Structures to relax.

    Returns
    -------
    list[Atoms]
        Relaxed structures.
    """

    # cache DFT energies of structures
    dft_energies = [s.get_potential_energy() for s in structures]

    # cache model predictions of unrelaxed structures
    for structure in structures:
        structure.calc = model

    initial_predicted_energies = [s.get_potential_energy() for s in structures]

    # convert structures into candidates for AGOX relaxer
    candidates = [StandardCandidate.from_atoms(template, s) for s in structures]

    # set up constraints
    confinement_cell = template.get_cell() * np.array([1, 1, 0]).T
    confinement_cell[2, 2] = 7
    confinement_corner = np.dot(template.get_cell().T, np.array([0, 0, 0]))
    confinement_corner[2] = template.positions[:, 2].max() - 0.5

    constraints = [
        FixAtoms(indices=range(len(template))),
        BoxConstraint(
            confinement_cell=confinement_cell,
            confinement_corner=confinement_corner,
            indices=range(len(template), len(structures[0])),
            pbc=[True, True, False])
    ]

    # perform relaxation
    print(f'Relaxing {len(candidates)} structures...')

    relaxer = ParallelRelaxPostprocess(model,
                                       optimizer_run_kwargs={'fmax': 0.005, 'steps': 1000},
                                       fix_template=False,
                                       constraints=constraints)

    relaxed_candidates = relaxer.process_list(candidates)
    relaxed_structures = [Atoms(c) for c in relaxed_candidates]

    print('Relaxed all structures.')

    # filter relaxed structures with unphysical energy differences
    final_predicted_energies = [candidate.get_potential_energy() for candidate in candidates]

    predicted_energy_diffs = [ef - ei for ei, ef in zip(initial_predicted_energies, final_predicted_energies)]

    # use the shape of the distribution of energy differences to determine a
    # threshold by which to filter

    # absolute bandwidth, independent of the covariance of the sample
    bw = 0.1
    bw_factor = bw / np.sqrt(np.cov(predicted_energy_diffs))
    kde = gaussian_kde(predicted_energy_diffs, bw_method=bw_factor)

    # maximum of first (main) peak
    max_res = minimize(lambda x: -kde(x), np.max(predicted_energy_diffs))

    # valley between peaks
    x0 = max_res.x - bw
    min_res = minimize(kde, x0, bounds=((-np.inf, x0),))

    threshold = min_res.x
    accept = predicted_energy_diffs > threshold

    # replace filtered structures with original geometries, clear constraints,
    # and reset energies to cached DFT energies
    for i in range(len(relaxed_structures)):
        if not accept[i]:
            relaxed_structures[i] = structures[i]
        relaxed_structures[i].set_constraint()
        relaxed_structures[i].calc = SinglePointCalculator(relaxed_structures[i],
                                                           energy=dft_energies[i])

    print(f'Replaced {np.count_nonzero(~accept)} structures.')

    return relaxed_structures
