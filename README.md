# PINNs_in_medicine
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
