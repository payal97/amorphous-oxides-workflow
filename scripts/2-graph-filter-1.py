from ase.io import read, write
from ase.atoms import Atoms
from ase.build import surface
from ase.cell import Cell
from oxide_nanocluster_workflow.config import (SingleBulkStoichiometry,
                                               parse_args, parse_config)
from oxide_nanocluster_workflow.filters import graph_filter
from oxide_nanocluster_workflow.surface import create_surface


def main():
    """Identify structure groups by their graph fingerprint and retain the most
    stable structure from each group. Step 2 as described in the "Dataset
    refinement" section of the manuscript.

    This script requires a configuration file for a single stoichiometry.

    This script only needs to be run once for each stoichiometry.
    """

    (config_path, _) = parse_args()
    config = parse_config(config_path, SingleBulkStoichiometry)

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

    structures = read(config.run_dir / 'energy_filtered.traj', index=':')
    print(f'Loaded {len(structures)} structures.')

    structures = graph_filter(structures, template)
    print(f'Filtered to {len(structures)} structures.')

    write(config.run_dir / 'graph_filtered_1.traj', structures)


if __name__ == '__main__':
    main()
