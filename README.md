# NVXPY

![Build](https://github.com/landonclark97/nvxpy/actions/workflows/test.yml/badge.svg)
[![codecov](https://codecov.io/gh/landonclark97/nvxpy/branch/main/graph/badge.svg)](https://codecov.io/gh/landonclark97/nvxpy)

## Overview

NVXPY is a Python-based Domain Specific Language (DSL) designed for formulating and solving non-convex programs using a natural, math-inspired API. It is designed to have as similar an interface to [CVXPY](https://github.com/cvxpy/cvxpy) as possible.

NVXPY is not a solver, it uses the solvers exposed by the `minimize` method in SciPy.


## Installation

NVXPY can be installed from PyPi using:

```bash
pip install nvxpy
```

and has the following dependencies:

* Python >= 3.11
* NumPy >= 2.3
* SciPy >= 1.15
* Autograd >= 1.8

## Usage

The following is a simple example to get started with NVXPY:

```python
import numpy as np
import nvxpy as nvx

x = nvx.Variable((3,))
x.value = np.array([-5.0, 0.0, 0.0]) # NLPs require a seed.

x_d = np.array([5.0, 0.0, 0.0])

obj = nvx.norm(x - x_d)
constraints = [nvx.norm(x) >= 1.0]

prob = nvx.Problem(nvx.Minimize(obj), constraints)
prob.solve(solver=nvx.SLSQP)

print(f'optimized value of x: {x.value}')
```

The above code will likely get stuck in a local optimum. To reach the globally optimal solution, we can adjust the seed and re-solve as follows:

```python
x.value = np.array([-5.0, 1.0, 1.0])
prob.solve(solver=nvx.SLSQP)

print(f'globally optimal value for x: {x.value}')
```


## Limitations

NVXPY is in early development. The most pressing issues are as follows:

* Slow evaluation of objective functions. Currently the objective function is always evaluated by parsing an expression tree. However, in an ideal world this expression tree would be "compiled" before solving.
* Only supports SciPy-based solvers. Ideally most other NLP solvers should be easy to add, such as IPOPT, which uses a similar API to the `minimize` function from SciPy.
* No plans to support integer programming any time soon. One potential solution is to make a SLP-inspired solver based on the MILP solver from SciPy to enforce integrality constraints. However, any custom solver would probably be out of the scope of this project.
* Small amount of atomic operations and sets.
* Unknown unknowns...


## Development

To contribute to NVXPY, clone the repository and install the development dependencies:

```bash
git clone https://github.com/landonclark97/nvxpy.git
cd nvxpy
poetry install --with dev
```

### Running Tests

Tests are written using `pytest`. To run the tests, execute:

```bash
poetry pytest
```

## License

[Apache 2.0](LICENSE)

## Contact

For any inquiries or issues, please contact Landon Clark at [landonclark97@gmail.com](mailto:landonclark97@gmail.com).
