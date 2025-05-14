# ExtraVentricular Drain (EVD) Planning Tool

This code is associated with the paper published in Neurosurgical Focus.
If you use this code, please cite the paper.

```bibtex
@article{de_Boer_Fully_Automatic_Anatomical_2025,
author = {de Boer, Mathijs and van Doormaal, Jesse A.M. and Köllen, Mare H. and Bartels, Lambertus W. and Robe, Pierre A.J.T. and van Doormaal, Tristan P.C.},
doi = {10.3171/2025.X.FOCUSXXXXX},
journal = {Neurosurgical Focus},
month = jul,
number = {7},
title = {{"Fully Automatic Anatomical Landmark Localization and Trajectory Planning for Navigated External Ventricular Drain Placement"}},
volume = {58},
year = {2025}
}
```

## Requirements

All requirements are documented in the `cargo.toml` and `pyproject.toml` files.
Building the project requires **both** the Rust compiler and the Python interpreter.

At time of writing, we used the following versions:

- Rust 1.84.0
- Python 3.12.8 (Initially 3.11)

## Installation

To install the project, clone the repository, open a terminal in the project directory and run the following command:

```bash
pip install .
```

We highly recommend using a virtual environment to avoid conflicts with other projects.

## Usage

We provide a set of CLI commands to interact with the main functionalities of the project.
Each command will start with `evd` followed by the subcommand.
Note that subcommands can have their own subcommands.

To see the list of available commands, run:

```bash
evd --help
```

This will also work for subcommands:

```bash
evd subcommand --help
```

We include `trogon` integration to run the commands in a TUI environment.

```bash
evd tui
```

Note that `trogon` is a fairly experimental library and may not work as expected.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
