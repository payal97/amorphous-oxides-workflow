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

    from ase.calculators.vasp import Vasp

    return Vasp(
        command='mpirun vasp_gam >> out',
        istart=0,
        icharg=2,
        xc="PBE",
        nsw=0,
        ibrion=-1,
        encut=400,
        prec="Normal",
        #algo="Fast",
        addgrid=True,
        isym=0,
        ismear=1,
        sigma=0.2,
        ediff=1E-4,
        ediffg=-0.05,
        npar=8,
        lplane=True,
        lscalu=False,
        nsim=32,
        lreal="Auto",
        kpts=(1,1,1),
        gamma=True,
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

    from ase.calculators.vasp import Vasp

    return Vasp(
        command='mpirun vasp_gam >> out',
        istart=1,
        icharg=1,
        xc="PBE",
        nsw=200,
        isif=2,
        ibrion=2,
        encut=400,
        prec="Accurate",
        #algo="Fast",
        ismear=1,
        sigma=0.2,
        ediff=1E-4,
        ediffg=-0.05,
        npar=8,
        lplane=True,
        lscalu=False,
        nsim=32,
        lreal="Auto",
        kpts=(1,1,1),
        gamma=True,
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

    calculator.set(
        command='mv CONTCAR POSCAR && rm WAVECAR && mpirun vasp_std >> out || true',
        kpts=(2, 2, 1), gamma=False
    )
    return calculator
