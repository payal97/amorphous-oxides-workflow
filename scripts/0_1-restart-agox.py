import os

from ase.atoms import Atoms
from ase.build import surface
from ase.cell import Cell
from oxide_nanocluster_workflow.calculators import agox_target_calc
from oxide_nanocluster_workflow.callback import vasp_callback
from oxide_nanocluster_workflow.config import (SingleBulkStoichiometry,
                                               parse_args, parse_config)
from oxide_nanocluster_workflow.restart_agox import restart_agox


def main():
    """Run global optimization using AGOX, as described in the "Global
    optimization" section of the manuscript.

    This script requires a configuration file for a single stoichiometry.

    This script can be run multiple times in parallel for independent AGOX
    global optimization runs. Provide the `--index` command-line argument when
    running in parallel.
    """

    (config_path, index) = parse_args()
    config = parse_config(config_path, SingleBulkStoichiometry)

    agox_run_dir = config.run_dir / 'agox_run'
    agox_run_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(agox_run_dir)

    # Make an empty template with periodic boundary conditions in all directions
    bulk = Atoms("",
                 cell=Cell.fromcellpar([config.bulk.a,
                                        config.bulk.b,
                                        config.bulk.c,
                                        config.bulk.alpha,
                                        config.bulk.beta,
                                        config.bulk.gamma]),
                 pbc=True)
    template = surface(bulk, (0, 0, 1), 1)
    template.center(vacuum=14, axis=2)
    template.pbc = True

    restart_agox(symbols=config.symbols,
             num_iterations=config.agox.num_iterations,
             target_calc=agox_target_calc(template, index),
             check_callback=vasp_callback,
             template=template,
             index=index)


if __name__ == '__main__':
    main()
