from ase.io import read, write
from ase.atoms import Atoms
from ase.build import surface
from ase.cell import Cell
from oxide_nanocluster_workflow.config import (SingleBulkStoichiometry,
                                               parse_args, parse_config)
from oxide_nanocluster_workflow.filters import graph_filter, joined_filter
from oxide_nanocluster_workflow.surface import create_surface


def main():
    """Identify structure groups by their graph fingerprint and retain the most
    stable structure from each group. Then filter structures based on whether
    the adsorbed atoms form a single joined nanocluster. Steps 7 and 8 as
    described in the "Dataset refinement" section of the manuscript.

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

    structure_paths = sorted((config.run_dir).glob('dft_relax_*/struc_*.traj'))
    structures = [read(path) for path in structure_paths]
    print(f'Loaded {len(structures)} structures.')

    structures = graph_filter(structures, template)
    print(f'Filtered to {len(structures)} structural groups.')

    write(config.run_dir / 'final_structures.traj', structures)


if __name__ == '__main__':
    main()
