# Oxide nanocluster workflow

This repository contains a Python package, `oxide_nanocluster_workflow`,
implementing the methods used in the oxide nanocluster generation workflow as
outlined in
["Inverse catalysts: Tuning the composition and structure of oxide clusters through the metal support"](https://doi.org/10.1038/s41524-024-01507-z).
It also contains a set of scripts gluing the various methods together into the
actual workflow. These scripts are quite barebones so that they can be run
standalone or easily be adapted into your local job running system.

## Installation

To install the package, clone the Git repository and install via `pip`. It is
recommended to install the package
[in development mode ("editable install")](https://setuptools.pypa.io/en/latest/userguide/development_mode.html),
as it is likely you might want to make some changes to the package code to suit
your application.

```bash
$ git clone git@gitlab.com:lkkmpn/oxide-nanocluster-workflow.git
$ cd oxide-nanocluster-workflow

# in an activated virtual environment
$ pip install --editable .
```

## Usage

The standalone scripts can be found in the [`scripts`](scripts/) subdirectory.
These are listed in the order in which they need to be executed, i.e.,
`1-energy-filter.py` should be executed after `0-agox.py`. Some scripts are
meant to be executed in parallel, i.e., multiple AGOX searches can be performed
in parallel.

Basic script configuration is provided via YAML files; examples are included in
the [`config-examples`](config-examples/) subdirectory. Configuring DFT
potentials can be more complex, and these have therefore not been included in
the configuration files. Instead, the package file
[`calculators.py`](oxide_nanocluster_workflow/calculators.py) can be adapted to
your needs.

Each script takes the same command-line arguments:
```
usage: scriptname.py [-h] [-i INDEX] config

positional arguments:
  config                path to YAML config file

options:
  -h, --help            show this help message and exit
  -i INDEX, --index INDEX
                        index of parallel run
```
Note that the `index` argument is ignored when the respective script does not
require parallel execution; see the table below for more details.

Generally, each script operates on a set of structures with the same
stoichiometry, and [`config.yaml`](config-examples/config.yaml) therefore
defines a single stoichiometry. Independent parallel runs can be set up in
separate working directories (via the `run_dir` option in the configuration
file) to execute the workflow for multiple stoichiometries.

The local GPR training step (step 3) is an exception, as training of this model
can be done with multiple stoichiometries. Hence, this step should be run once
for a set of stoichiometries. It therefore also requires a different
configuration file,
[`local-gpr-config.yaml`](config-examples/local-gpr-config.yaml).

### Script overview

| File name                                              | Config type   | Parallel? (`--index`)        |
|--------------------------------------------------------|---------------|------------------------------|
| [`0-agox.py`](scripts/0-agox.py)                       | Single        | ✅ independent instances      |
| [`1-energy-filter.py`](scripts/1-energy-filter.py)     | Single        | ❎                            |
| [`2-graph-filter-1.py`](scripts/2-graph-filter-1.py)   | Single        | ❎                            |
| [`3-local-gpr-train.py`](scripts/3-local-gpr-train.py) | **Local GPR** | ✅ independent models         |
| [`4-local-gpr-relax.py`](scripts/4-local-gpr-relax.py) | Single        | ❎                            |
| [`5-graph-filter-2.py`](scripts/5-graph-filter-2.py)   | Single        | ❎                            |
| [`6-dft-relax.py`](scripts/6-dft-relax.py)             | Single        | ✅ one structure per instance |
| [`7-graph-filter-3.py`](scripts/7-graph-filter-3.py)   | Single        | ❎                            |

## Citation

If you use this workflow in a publication, please cite our work:

Kempen, L.H.E., Andersen, M. Inverse catalysts: tuning the composition and structure of oxide clusters through the metal support. _npj Computational Materials_ **11**, 8 (2025). https://doi.org/10.1038/s41524-024-01507-z

```
@article{kempen2025,
    title={Inverse catalysts: tuning the composition and structure of oxide clusters through the metal support},
    volume={11},
    ISSN={2057-3960},
    url={http://dx.doi.org/10.1038/s41524-024-01507-z},
    DOI={10.1038/s41524-024-01507-z},
    number={1},
    journal={npj Computational Materials},
    author={Kempen, Luuk H. E. and Andersen, Mie},
    year={2025},
    pages={8}
}
```

## License

All code in this repository is released under [the GPLv3 license.](LICENSE.md)
