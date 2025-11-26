# NVXPY

[![Build Status](https://github.com/landonclark97/nvxpy/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/landonclark97/nvxpy/actions/workflows/test.yaml)
[![codecov](https://codecov.io/gh/landonclark97/nvxpy/branch/main/graph/badge.svg)](https://codecov.io/gh/landonclark97/nvxpy)

## Overview

NVXPY is a Python-based Domain Specific Language (DSL) designed for formulating and solving non-convex programs using a natural, math-inspired API. It is designed to have as similar an interface to [CVXPY](https://github.com/cvxpy/cvxpy) as possible.

NVXPY is not a solver, it uses solvers from other packages (such as SLSQP, IPOPT, etc.), and includes a built-in Branch-and-Bound solver for mixed-integer nonlinear programs (MINLP).


## Features

* Simple, concise, and math-inspired interface
* IR compiler for efficient evaluation (85%-99% as fast as native Python)
* Built-in Branch-and-Bound MINLP solver with discrete value constraints
* Graph constructs for clean MIP formulations (wraps networkx)
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


## Available Solvers

| Solver | Type | Description |
|--------|------|-------------|
| `nvx.SLSQP` | NLP | Sequential Least Squares Programming (SciPy) |
| `nvx.COBYLA` | NLP | Constrained Optimization BY Linear Approximation |
| `nvx.TRUST_CONSTR` | NLP | Trust-region constrained algorithm |
| `nvx.BFGS` | NLP | Unconstrained quasi-Newton method |
| `nvx.LBFGSB` | NLP | Limited-memory BFGS with bounds |
| `nvx.NELDER_MEAD` | NLP | Derivative-free simplex method |
| `nvx.TNC` | NLP | Truncated Newton method |
| `nvx.IPOPT` | NLP | Interior Point Optimizer (requires cyipopt) |
| `nvx.BNB` | MINLP | Built-in Branch-and-Bound solver |


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
