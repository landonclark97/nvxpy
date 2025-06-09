# NVXPY

## Overview

NVXPY is a Python-based Domain Specific Language (DSL) designed for formulating and solving non-convex programs using a natural, mathematical API. It is designed to have as similar an interface to [CVXPY](https://github.com/cvxpy/cvxpy) as possible.

NVXPY is not a solver, it uses the solvers exposed by the `minimize` method in SciPy.

## Features

- **Problem Solving**: Define and solve non-convex problems using the `problem.py` module.
- **Variable Management**: Handle variables efficiently with `variable.py`.
- **Mathematical Constructs**: Utilize constructs like functions and expressions through the `constructs` and `expression.py` modules.
- **Atoms and Sets**: Work with mathematical atoms and sets using the `atoms` and `sets` directories.
- **Parser**: Parse and interpret expressions with `parser.py`.

## Installation

Ensure you have Python 3.11 or later. The project uses Poetry for package management and installation. You can install the required dependencies using:

```bash
poetry install
```

## Usage

[Provide examples of how to use the library. This could include code snippets demonstrating key functionalities.]

## Development

To contribute to `nvxpy`, clone the repository and install the development dependencies:

```bash
git clone [repository URL]
cd nvxpy
poetry install --with dev
```

### Running Tests

Tests are written using `pytest`. To run the tests, execute:

```bash
pytest
```

## Directory Structure

- **src/nvxpy**: Contains the main library code.
  - **problem.py**: Core problem-solving functionalities.
  - **variable.py**: Variable management.
  - **constructs/**: Mathematical constructs.
  - **atoms/**: Mathematical atoms like sum, maximum, etc.
  - **sets/**: Set operations and definitions.
- **tests/**: Contains test cases for the library.

## Dependencies

- `numpy` >= 2.3.0, < 3.0.0
- `scipy` >= 1.15.3, < 2.0.0
- `autograd` >= 1.8.0, < 2.0.0
- `matplotlib` >= 3.10.3, < 4.0.0

## License

[Specify the license under which the project is distributed.]

## Contact

For any inquiries or issues, please contact Landon Clark at [landonclark97@gmail.com](mailto:landonclark97@gmail.com).
