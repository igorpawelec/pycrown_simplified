# Contributing to PyCrown Simplified

Thank you for your interest in contributing!

## How to contribute

1. **Fork** the repository and create a new branch from `main`.
2. **Install** the development environment:
   ```bash
   git clone https://github.com/<your-user>/pycrown_simplified.git
   cd pycrown_simplified
   pip install -e ".[dev]"
   ```
3. **Make your changes** — keep commits small and descriptive.
4. **Test** your changes:
   ```bash
   python -c "from pycrown import PyCrown; print('OK')"
   ```
5. **Open a Pull Request** against `main` with a clear description.

## Code style

- Follow PEP 8.
- Add docstrings (NumPy style) to all public functions and classes.
- Keep Numba-jitted functions in separate `_crown_*` modules.

## Reporting issues

Open an issue on GitHub with:
- A clear title and description
- Steps to reproduce (if applicable)
- Your Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the GNU GPLv3.
