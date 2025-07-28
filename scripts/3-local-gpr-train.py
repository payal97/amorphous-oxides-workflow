import json
import random
from pathlib import Path

from ase.atoms import Atoms
from ase.io import read
from oxide_nanocluster_workflow.config import (LocalGPR, parse_args,
                                               parse_config)
from oxide_nanocluster_workflow.local_model import train_local_model
from oxide_nanocluster_workflow.utils import (get_subset, get_unique_species,
                                              rmse, split_data)


def main():
    """Train a local GPR surrogate model on the filtered structures. Step 3 as
    described in the "Dataset refinement" section of the manuscript.

    This script requires a configuration file for training a local GPR model.

    This script can be run multiple times in parallel for training multiple
    independent models. Provide the `--index` command-line argument when
    running in parallel.
    """

    (config_path, index) = parse_args()
    config = parse_config(config_path, LocalGPR)

    run_dir = Path('local_model')
    run_dir.mkdir(parents=True, exist_ok=True)

    model_parameters_path = run_dir / f'model_parameters_{index:03d}.h5'
    model_info_path = run_dir / f'model_info_{index:03d}.json'

    random.seed(index)

    structures: list[Atoms] = []
    for path in config.input_run_dirs:
        try:
            structures += read(path / 'graph_filtered_1.traj', index=':')
        except Exception as err:
            print(err)

    print(f'Loaded {len(structures)} structures.')

    species = get_unique_species(structures)

    train_indices, other_indices = split_data(len(structures), config.max_train_structures)
    train_structures = [structures[i] for i in train_indices]

    # train
    print('Training model...')

    model = train_local_model(train_structures, species)
    model.save(model_parameters_path)

    # evaluate
    in_structures = [structures[i] for i in
                     get_subset(train_indices, config.max_evaluate_structures)]
    out_structures = [structures[i] for i in
                      get_subset(other_indices, config.max_evaluate_structures)]

    print(f'Evaluating model on {len(in_structures)} in-sample structures and '
          f'{len(out_structures)} out-of-sample structures...')

    in_targets = [s.get_potential_energy() for s in in_structures]
    out_targets = [s.get_potential_energy() for s in out_structures]

    in_preds = [model.predict_energy(s) for s in in_structures]
    out_preds = [model.predict_energy(s) for s in out_structures]

    overall_rmse = rmse(in_targets + out_targets, in_preds + out_preds)
    in_rmse = rmse(in_targets, in_preds)
    out_rmse = rmse(out_targets, out_preds)

    # write info
    info = {
        'model_parameters_path': str(model_parameters_path),
        'overall_rmse': overall_rmse,
        'in_rmse': in_rmse,
        'out_rmse': out_rmse,
        'species': species
    }

    print(f'In-sample RMSE: {in_rmse:.6f} eV')
    print(f'Out-of-sample RMSE: {out_rmse:.6f} eV')
    print(f'Overall RMSE: {overall_rmse:.6f} eV')

    with open(model_info_path, 'w') as f:
        json.dump(info, f)


if __name__ == '__main__':
    main()
