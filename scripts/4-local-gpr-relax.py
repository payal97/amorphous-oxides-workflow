from pathlib import Path

from ase.io import read, write
from oxide_nanocluster_workflow.config import (SingleStoichiometry, parse_args,
                                               parse_config)
from oxide_nanocluster_workflow.local_model import (find_best_model_parameters,
                                                    load_local_model,
                                                    relax_local_model)
from oxide_nanocluster_workflow.surface import create_surface


def main():
    """Relax structures with a trained local GPR surrogate model. Step 4 as
    described in the "Dataset refinement" section of the manuscript.

    This script requires a configuration file for a single stoichiometry.

    This script only needs to be run once for each stoichiometry.
    """

    (config_path, _) = parse_args()
    config = parse_config(config_path, SingleStoichiometry)

    local_model_paths = sorted(Path('local_model').glob('model_info_*.json'))
    best_model = find_best_model_parameters(local_model_paths)

    model = load_local_model(best_model['model_parameters_path'],
                             best_model['species'])

    template = create_surface(element=config.surface.element,
                              size=config.surface.low_level_size,
                              a=config.surface.a,
                              vacuum=config.surface.vacuum)

    structures = read(config.run_dir / 'graph_filtered_1.traj', index=':')

    structures = relax_local_model(model, template, structures)

    write(config.run_dir / 'local_gpr_relaxed.traj', structures)


if __name__ == '__main__':
    main()
