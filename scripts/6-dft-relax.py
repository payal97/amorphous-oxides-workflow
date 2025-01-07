from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize.bfgs import BFGS
from oxide_nanocluster_workflow.calculators import (dft_refine_calc,
                                                    dft_relax_calc)
from oxide_nanocluster_workflow.config import (SingleStoichiometry, parse_args,
                                               parse_config)
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
    config = parse_config(config_path, SingleStoichiometry)

    relax_run_dir = config.run_dir / 'dft_relax'
    relax_run_dir.mkdir(parents=True, exist_ok=True)

    structure = read(config.run_dir / 'graph_filtered_2.traj', index=index)

    # transfer structure to high-level surface
    templates = [create_surface(element=config.surface.element,
                                size=size,
                                a=config.surface.a,
                                vacuum=config.surface.vacuum)
                 for size in (config.surface.low_level_size, config.surface.high_level_size)]

    structure = transfer_surface(structure, templates[0], templates[1])

    structure.set_constraint(FixAtoms(mask=structure.get_tags() > 1))

    # initial relaxation
    calc = dft_relax_calc(structure)
    structure.calc = calc

    dyn = BFGS(structure, trajectory=str(relax_run_dir / f'traj_{index:03d}.traj'))
    dyn.run(fmax=0.05)

    # refinement
    calc = dft_refine_calc(calc, structure)
    structure.calc = calc

    structure.get_potential_energy()

    write(relax_run_dir / f'struc_{index:03d}.traj', structure)


if __name__ == '__main__':
    main()
