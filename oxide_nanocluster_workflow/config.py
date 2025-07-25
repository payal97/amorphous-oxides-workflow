import argparse
from os import PathLike
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel

C = TypeVar('C', bound=BaseModel)


class SurfaceSettings(BaseModel):
    element: str
    """Element to create the surface from."""

    a: float
    """Lattice parameter for the fcc(111) surface."""

    low_level_size: tuple[int, int, int]
    """Number of atoms (a, b, c) in the surface for low-level calculations."""

    high_level_size: tuple[int, int, int]
    """Number of atoms (a, b, c) in the surface for high-level calculations."""

    vacuum: float
    """Total vacuum size between top layer of surface atoms and bottom layer in
    next periodic copy (in Å)."""


class BulkSettings(BaseModel):
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    """Lattice parameters."""


class AGOXSettings(BaseModel):
    num_iterations: int
    """Number of AGOX iterations to run."""


class EnergyFilterSettings(BaseModel):
    threshold: float
    """Energy filtering threshold relative to the global minimum energy (in
    eV)."""


class SingleStoichiometry(BaseModel):
    run_dir: Path
    """Working directory for this stoichiometry."""

    nanocluster_stoichiometry: str
    """Stoichiometry of nanocluster to search for."""

    surface: SurfaceSettings
    agox: AGOXSettings
    energy_filter: EnergyFilterSettings

    def model_post_init(self, _):
        """Perform additional initialization.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)


class SingleBulkStoichiometry(BaseModel):
    run_dir: Path
    """Working directory for this stoichiometry."""

    symbols: str
    """Stoichiometry to search for."""

    bulk: BulkSettings
    agox: AGOXSettings
    energy_filter: EnergyFilterSettings

    def model_post_init(self, _):
        """Perform additional initialization.
        """
        self.run_dir.mkdir(parents=True, exist_ok=True)


class LocalGPR(BaseModel):
    input_run_dirs: list[Path]
    """List of run paths to take structures from for training the local GPR
    model."""

    max_train_structures: int
    """Maximum number of structures to train on (to limit memory usage)."""

    max_evaluate_structures: int
    """Maximum number of structures to evaluate models (to limit runtime)."""


def parse_config(config_path: PathLike, config_type: type[C]) -> C:
    """Parse a YAML configuration file into a dataclass.

    Parameters
    ----------
    config_path : PathLike
        Configuration file path.
    config_type : ConfigType
        Configuration class to parse data into.

    Returns
    -------
    ConfigType
        Parsed configuration.
    """

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config_type.model_validate(config)


def parse_args() -> tuple[str, int]:
    """Parse command-line arguments for configuration file path and parallel
    run index.

    Returns
    -------
    tuple[str, int]
        Configuration file path and parallel run index.
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('config',
                        type=str,
                        help='path to YAML config file')

    parser.add_argument('-i', '--index',
                        default=0,
                        type=int,
                        required=False,
                        help='index of parallel run')

    args = parser.parse_args()

    return args.config, args.index
