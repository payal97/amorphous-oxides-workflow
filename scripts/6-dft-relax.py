import os

from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize.bfgs import BFGS
from oxide_nanocluster_workflow.calculators import (dft_refine_calc,
                                                    dft_relax_calc)
from oxide_nanocluster_workflow.config import (SingleBulkStoichiometry,
                                               parse_args, parse_config)
from oxide_nanocluster_workflow.surface import create_surface, transfer_surface


def main():
    """Perform relaxation and refinement with the high-level DFT setup. Step 6
    as described in the "Dataset refinement" section of the manuscript.

    This script requires a configuration file for a single stoichiometry.

    This script can be run multiple times in parallel, as each instance only
    relaxes and refines a single structure. Provide the `--index` command-line
    argument when running in parallel.
    """

    (config_path, index) = parse_args()
    config = parse_config(config_path, SingleBulkStoichiometry)

    relax_run_dir = config.run_dir / f'dft_relax_{index:03d}'
    relax_run_dir.mkdir(parents=True, exist_ok=True)

    # trajectory indices start from 0, while job array indices start from 1
    index = index - 1
    structure = None
    try:
        structure = read(config.run_dir / 'graph_filtered_2.traj', index=index)
    except Exception as err:
        print(err)
    # if not found in graph_filtered_2, find in graph_filtered_1
    if structure is None:
        try:
            structure = read(config.run_dir / 'graph_filtered_1.traj', index=index)
        except Exception as err:
            print(err)

    if structure is not None:
        os.chdir(relax_run_dir)

        # initial relaxation
        calc = dft_relax_calc(structure)
        structure.calc = calc

        dyn = BFGS(structure, trajectory=str(f'traj_{index:03d}.traj'))
        dyn.run(fmax=0.05)

        # refinement
        calc = dft_refine_calc(calc, structure)
        structure.calc = calc

        structure.get_potential_energy()

        write(f'struc_{index:03d}.traj', structure)


if __name__ == '__main__':
    main()
