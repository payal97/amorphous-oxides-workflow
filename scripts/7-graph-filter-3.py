from ase.io import read, write
from oxide_nanocluster_workflow.config import (SingleStoichiometry, parse_args,
                                               parse_config)
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
    config = parse_config(config_path, SingleStoichiometry)

    template = create_surface(element=config.surface.element,
                              size=config.surface.high_level_size,
                              a=config.surface.a,
                              vacuum=config.surface.vacuum)

    structure_paths = sorted((config.run_dir / 'dft_relax').glob('struc_*.traj'))
    structures = [read(path) for path in structure_paths]
    print(f'Loaded {len(structures)} structures.')

    structures = graph_filter(structures, template)
    print(f'Filtered to {len(structures)} structural groups.')

    structures = joined_filter(structures, template)
    print(f'Filtered to {len(structures)} joined structures.')

    write('final_structures.traj', structures)


if __name__ == '__main__':
    main()
