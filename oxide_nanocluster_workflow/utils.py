import random
from pathlib import Path
from typing import TypeVar

import numpy as np
from agox.databases import Database
from ase.atoms import Atoms

T = TypeVar('T')


def load_from_databases(db_paths: list[Path]) -> list[Atoms]:
    """Load structures from a list of AGOX databases.

    Parameters
    ----------
    db_paths : list[Path]
        List of paths to AGOX database files.

    Returns
    -------
    list[Atoms]
        List of structures read from all database files.
    """

    structures: list[Atoms] = []

    for db_path in db_paths:
        db = Database(db_path)
        structures += db.restore_to_trajectory()

    return structures


def get_unique_species(structures: list[Atoms]) -> list[str]:
    """Get the list of unique species occurring in a list of Atoms objects. The
    returned list is sorted alphabetically.

    Parameters
    ----------
    structures : list[Atoms]
        List of Atoms objects.

    Returns
    -------
    list[str]
        List of unique species.
    """

    species = set()
    for structure in structures:
        species.update(structure.get_chemical_symbols())
    return sorted(species)


def split_data(num_data: int,
               max_train: int) -> tuple[list[int], list[int]]:
    """Split input data into a training and validation set, with at least 10%
    of the data assigned for validation and a maximum number of data points
    assigned for training.

    Parameters
    ----------
    num_data : int
        Number of input data points.
    max_train : int
        Maximum number of data points to assign to the training set.

    Returns
    -------
    tuple[list[int], list[int]]
        Lists of indices that make up the training data and validation data,
        respectively.
    """

    indices = range(num_data)

    # ensure at least a 90/10 split
    n_train = min(int(round(0.9 * num_data)), max_train)

    train_indices = random.sample(indices, n_train)
    other_indices = list(set(indices) - set(train_indices))

    return train_indices, other_indices


def get_subset(data: list[T], n: int) -> list[T]:
    """Get a subset of a dataset if possible.

    Parameters
    ----------
    data : list[T]
        Dataset to get a subset from.
    n : int
        Number of elements to get.

    Returns
    -------
    list[T]
        Subset of data. If n >= len(data), data is returned.
    """

    if len(data) > n:
        return random.sample(data, n)
    else:
        return data


def rmse(target: list[float], pred: list[float]) -> float:
    """Calculate the root mean square error between target and predicted
    values.

    Parameters
    ----------
    target : list[float]
        List of target values.
    pred : list[float]
        List of predicted values.

    Returns
    -------
    float
        Root mean square error.
    """

    target = np.asarray(target)
    pred = np.asarray(pred)

    return np.sqrt(np.sum((target - pred) ** 2) / len(target))
