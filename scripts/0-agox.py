import os

from oxide_nanocluster_workflow.calculators import agox_target_calc
from oxide_nanocluster_workflow.config import (SingleStoichiometry, parse_args,
                                               parse_config)
from oxide_nanocluster_workflow.run_agox import run_agox
from oxide_nanocluster_workflow.surface import create_surface


def main():
    """Run global optimization using AGOX, as described in the "Global
    optimization" section of the manuscript.

    This script requires a configuration file for a single stoichiometry.

    This script can be run multiple times in parallel for independent AGOX
    global optimization runs. Provide the `--index` command-line argument when
    running in parallel.
    """

    (config_path, index) = parse_args()
    config = parse_config(config_path, SingleStoichiometry)

    agox_run_dir = config.run_dir / 'agox_run'
    agox_run_dir.mkdir(parents=True, exist_ok=True)

    os.chdir(agox_run_dir)

    template = create_surface(element=config.surface.element,
                              size=config.surface.low_level_size,
                              a=config.surface.a,
                              vacuum=config.surface.vacuum)

    run_agox(nanocluster_stoichiometry=config.nanocluster_stoichiometry,
             num_iterations=config.agox.num_iterations,
             target_calc=agox_target_calc(template, index),
             template=template,
             template_size=config.surface.low_level_size,
             index=index)


if __name__ == '__main__':
    main()
