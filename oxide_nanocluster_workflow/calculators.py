from ase.atoms import Atoms
from ase.calculators.calculator import Calculator


def agox_target_calc(template: Atoms,
                     index: int) -> Calculator:
    """Return a Calculator object used as target potential in the AGOX
    structure searches.

    Parameters
    ----------
    template : Atoms
        Metal surface to place nanocluster atoms on.
    index : int
        Index of parallel run.

    Returns
    -------
    Calculator
        AGOX target potential.
    """

    from agox.helpers.gpaw_subprocess import SubprocessGPAW

    return SubprocessGPAW(
        log_directory=f'gpaw_logs_{index:03d}',
        mode='lcao',
        basis='dzp',
        convergence={'energy': 0.005,
                     'density': 0.001,
                     'eigenstates': 0.001,
                     'bands': 'occupied'},
        txt='gpaw.txt'
    )


def dft_relax_calc(structure: Atoms) -> Calculator:
    """Return a Calculator object used as potential to perform high-level DFT
    relaxations.

    Parameters
    ----------
    structure : Atoms
        Structure to perform relaxation on.

    Returns
    -------
    Calculator
        Calculator object.
    """

    from gpaw.calculator import GPAW
    from gpaw.utilities import h2gpts

    return GPAW(
        mode={'name': 'pw',
              'ecut': 400},
        xc='PBE',
        basis='dzp',
        convergence={'energy': 0.005,
                     'density': 1e-5,
                     'eigenstates': 4e-7,
                     'bands': 'occupied'
                     },
        occupations={'name': 'fermi-dirac',
                     'width': 0.1},
        gpts=h2gpts(0.20, structure.get_cell(), idiv=8),
        nbands='110%'
    )


def dft_refine_calc(calculator: Calculator,
                    structure: Atoms) -> Calculator:
    """Return a Calculator object used as potential to perform single-point DFT
    refinement in the high-level setup.

    Parameters
    ----------
    calculator : Calculator
        Calculator used to perform relaxations. Depending on the calculator
        implementation, modifying this calculator might be faster than defining
        a new calculator.
    structure : Atoms
        Structure to perform evaluation on.

    Returns
    -------
    Calculator
        Calculator object.
    """

    calculator.set(kpts=(2, 2, 1))
    return calculator
