# NVXPY

[![Build Status](https://github.com/landonclark97/nvxpy/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/landonclark97/nvxpy/actions/workflows/test.yaml)
[![codecov](https://codecov.io/gh/landonclark97/nvxpy/branch/main/graph/badge.svg)](https://codecov.io/gh/landonclark97/nvxpy)

## Overview

NVXPY is a Python-based Domain Specific Language (DSL) designed for formulating and solving non-convex programs using a natural, math-inspired API. It is designed to have as similar an interface to [CVXPY](https://www.cvxpy.org/) as possible.

NVXPY is not a solver, it uses solvers from other packages (such as SLSQP, IPOPT, etc.), and includes a built-in Branch-and-Bound solver for mixed-integer nonlinear programs (MINLP).


## Features

* Simple, concise, and math-inspired interface
* Codegen compiler for efficient evaluation (85%-99% as fast as native Python)
* Built-in Branch-and-Bound MINLP solver with discrete value constraints
* Graph constructs for clean MIP formulations (wraps [networkx](https://networkx.org/en/))
* Handles gradients seamlessly, even for custom functions


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
* NetworkX >= 3.0
* cyipopt (optional, for IPOPT solver)

## Usage

### Basic NLP Example

```python
import numpy as np
import nvxpy as nvx

x = nvx.Variable((3,))
x.value = np.array([-5.0, 0.0, 0.0])  # NLPs require a seed

x_d = np.array([5.0, 0.0, 0.0])

obj = nvx.norm(x - x_d)
constraints = [nvx.norm(x) >= 1.0]  # Non-convex!

prob = nvx.Problem(nvx.Minimize(obj), constraints)
prob.solve(solver=nvx.SLSQP)

print(f'optimized value of x: {x.value}')
```

### Mixed-Integer Programming

NVXPY supports integer and binary variables with a built-in Branch-and-Bound solver:

```python
import nvxpy as nvx

# Binary knapsack problem
x = nvx.Variable(integer=True, name="x")
y = nvx.Variable(integer=True, name="y")

prob = nvx.Problem(
    nvx.Maximize(10*x + 6*y),
    [
        5*x + 3*y <= 15,
        x ^ [0, 1],  # x in {0, 1}
        y ^ [0, 1],  # y in {0, 1}
    ]
)
prob.solve(solver=nvx.BNB)
```

### Discrete Value Constraints

Variables can be constrained to discrete sets of values:

```python
x = nvx.Variable(integer=True)

# x must be one of these values
prob = nvx.Problem(
    nvx.Minimize((x - 7)**2),
    [x ^ [1, 5, 10, 15]]  # x in {1, 5, 10, 15}
)
prob.solve(solver=nvx.BNB)  # Optimal: x = 5
```

### Graph-Based MIP Formulations

NVXPY provides `Graph` and `DiGraph` constructs for clean graph optimization problems:

```python
import networkx as nx
import nvxpy as nvx

# Maximum Independent Set
nxg = nx.petersen_graph()
G = nvx.Graph(nxg)
y = G.node_vars(binary=True)

prob = nvx.Problem(
    nvx.Maximize(nvx.sum([y[i] for i in G.nodes])),
    G.independent(y)  # y[i] + y[j] <= 1 for all edges
)
prob.solve(solver=nvx.BNB)
```

Available graph helpers:
- `G.edge_vars(binary=True)` / `G.node_vars(binary=True)` - Create decision variables
- `G.degree(x) == k` - Degree constraints (undirected)
- `G.in_degree(x)` / `G.out_degree(x)` - For directed graphs
- `G.independent(y)` - Independent set constraints
- `G.covers(x, y)` - Vertex cover constraints
- `G.total_weight(x)` - Sum of edge weights for objectives


### Custom Functions

Wrap arbitrary Python functions using the `@nvx.function` decorator:

```python
import autograd.numpy as np
import nvxpy as nvx

@nvx.function(jac="autograd")
def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

x = nvx.Variable(shape=(2,))
x.value = np.array([0.0, 0.0])

prob = nvx.Problem(nvx.Minimize(rosenbrock(x)))
prob.solve(solver=nvx.LBFGSB)
```

Options:
- `jac="numerical"` - Finite difference gradients (default)
- `jac="autograd"` - Automatic differentiation via autograd
- `jac=callable` - User-provided Jacobian function


### Compilation

Pass `compile=True` to `Problem` to compile expression trees into generated Python code, avoiding interpreter overhead on each evaluation:

```python
prob = nvx.Problem(nvx.Minimize(obj), constraints, compile=True)
prob.solve(solver=nvx.SLSQP)
```


## Available Solvers

### Gradient-Free

| Solver | Description |
|--------|-------------|
| `nvx.NELDER_MEAD` | Derivative-free simplex method |
| `nvx.POWELL` | Derivative-free conjugate direction method |
| `nvx.COBYLA` | Constrained Optimization BY Linear Approximation |
| `nvx.COBYQA` | Constrained Optimization BY Quadratic Approximation |

### Gradient-Based

| Solver | Description |
|--------|-------------|
| `nvx.CG` | Conjugate gradient method |
| `nvx.BFGS` | Quasi-Newton method |
| `nvx.LBFGSB` | Limited-memory BFGS with bounds |
| `nvx.TNC` | Truncated Newton method |
| `nvx.SLSQP` | Sequential Least Squares Programming (supports constraints) |

### Hessian-Based

| Solver | Description |
|--------|-------------|
| `nvx.NEWTON_CG` | Newton-CG trust region method |
| `nvx.DOGLEG` | Dogleg trust region method |
| `nvx.TRUST_NCG` | Trust-region Newton-CG |
| `nvx.TRUST_KRYLOV` | Trust-region with Krylov subspace |
| `nvx.TRUST_EXACT` | Trust-region with exact Hessian |
| `nvx.TRUST_CONSTR` | Trust-region constrained algorithm |

### Global Optimizers

| Solver | Description |
|--------|-------------|
| `nvx.DIFF_EVOLUTION` | Differential evolution |
| `nvx.DUAL_ANNEALING` | Dual annealing |
| `nvx.SHGO` | Simplicial homology global optimization |
| `nvx.BASINHOPPING` | Basin-hopping with local minimization |

### Other

| Solver | Description |
|--------|-------------|
| `nvx.IPOPT` | Interior Point Optimizer (requires cyipopt) |
| `nvx.BNB` | Built-in Branch-and-Bound MINLP solver |


## Limitations

NVXPY is in active development. Current limitations:

* Branch-and-Bound solver is basic and may struggle with large problems
* Limited set of atomic operations
* Some edge cases may be untested


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
poetry run pytest
```

## License

[Apache 2.0](LICENSE)

## Contact

For any inquiries or issues, please contact Landon Clark at [landonclark97@gmail.com](mailto:landonclark97@gmail.com).
