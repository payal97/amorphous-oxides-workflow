from ase.io import write
from oxide_nanocluster_workflow.config import (SingleStoichiometry, parse_args,
                                               parse_config)
from oxide_nanocluster_workflow.filters import energy_filter
from oxide_nanocluster_workflow.utils import load_from_databases


def main():
    """Remove all structures with high potential energies. Step 1 as described
    in the "Dataset refinement" section of the manuscript.

    This script requires a configuration file for a single stoichiometry.

    This script only needs to be run once for each stoichiometry.
    """

    (config_path, _) = parse_args()
    config = parse_config(config_path, SingleStoichiometry)

    db_paths = sorted((config.run_dir / 'agox_run').glob('db_*.db'))
    structures = load_from_databases(db_paths)
    print(f'Loaded {len(structures)} structures.')

    structures = energy_filter(structures,
                               threshold=config.energy_filter.threshold)
    print(f'Filtered to {len(structures)} structures.')

    write(config.run_dir / 'energy_filtered.traj', structures)


if __name__ == '__main__':
    main()
