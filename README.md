# Pysics Informed Neural Networks in Medicine
*Code by Jakob*

[![Python 3.9](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/downloads/release/python-3113/)
## Linters

Install the pre-commit hooks for linting:

```python
pre-commit install
```

To run all linters manually, use:

```python
pre-commit
```

Note: only changes added to git are included in linting when using the pre-commit command.

You can also run the single linters one at a time to apply the linter to unstaged files:

```bash
black --check .
flake8 .
isort --check-only .
```

## Installation and Setup

Most conventional packages can be installed by running
```bash
pip install -r requirements.txt
```

Additional packages concerning conventional means of solving PDEs (FEniCSx) can only be installed by running

```bash
conda install -c conda-forge fenics-dolfinx mpich pyvista
```
