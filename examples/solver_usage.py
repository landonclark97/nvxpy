"""Examples demonstrating nvxpy with several solver backends.

Each helper builds a small optimization problem and solves it with a specific
solver. Execute this module directly to run every example in sequence, or
import the helper functions elsewhere to experiment interactively.
"""

from __future__ import annotations

import autograd.numpy as np  # type: ignore

from nvxpy import (  # type: ignore
    Minimize,
    Problem,
    Solver,
    SolverStatus,
    Variable,
)


def _unconstrained_quadratic(target: tuple[float, float]) -> Problem:
    x = Variable()
    y = Variable()
    tx, ty = target
    objective = Minimize((x - tx) ** 2 + (y - ty) ** 2)
    return Problem(objective, constraints=[])


def _constrained_quadratic() -> Problem:
    x = Variable()
    y = Variable()
    objective = Minimize((x - 1) ** 2 + (y - 2) ** 2)
    constraints = [
        x + y >= 1,
        x >= 0,
        y >= 0,
    ]
    return Problem(objective, constraints)


def _solve(problem: Problem, solver: Solver, *, solver_options=None, label: str = ""):
    try:
        result = problem.solve(solver=solver, solver_options=solver_options or {})
    except ValueError as exc:
        print(f"{label or solver.value}: failed -> {exc}")
        return

    status = result.status
    values = {var.name: var.value for var in problem.vars}
    print(f"{label or solver.value}: status={status}")
    for name, value in values.items():
        print(f"  {name} = {np.asarray(value).ravel()}")
    if status not in {SolverStatus.OPTIMAL, SolverStatus.SUBOPTIMAL}:
        print("  Warning: solver did not report optimality")


def solve_with_scipy_slsqp():
    _solve(_constrained_quadratic(), Solver.SLSQP, label="SLSQP")


def solve_with_scipy_cobyla():
    _solve(_constrained_quadratic(), Solver.COBYLA, label="COBYLA")


def solve_with_scipy_trust_constr():
    _solve(_constrained_quadratic(), Solver.TRUST_CONSTR, label="trust-constr")


def solve_with_scipy_nelder_mead():
    _solve(_unconstrained_quadratic((1.5, -0.5)), Solver.NELDER_MEAD, label="Nelder-Mead")


def solve_with_scipy_bfgs():
    _solve(_unconstrained_quadratic((2.0, -1.0)), Solver.BFGS, label="BFGS")


def solve_with_scipy_lbfgsb():
    _solve(_unconstrained_quadratic((0.5, 0.5)), Solver.LBFGSB, label="L-BFGS-B")


def solve_with_scipy_tnc():
    _solve(_unconstrained_quadratic((1.0, 1.0)), Solver.TNC, label="TNC")


def solve_with_ipopt():
    try:
        import cyipopt  # type: ignore  # noqa: F401
    except ImportError:  # pragma: no cover
        print("cyipopt is not installed; skipping IPOPT example.")
        return

    _solve(
        _constrained_quadratic(),
        Solver.IPOPT,
        label="IPOPT",
    )


SCIPY_EXAMPLES = [
    solve_with_scipy_slsqp,
    solve_with_scipy_cobyla,
    solve_with_scipy_trust_constr,
    solve_with_scipy_nelder_mead,
    solve_with_scipy_bfgs,
    solve_with_scipy_lbfgsb,
    solve_with_scipy_tnc,
]


IPOPT_EXAMPLES = [solve_with_ipopt]


def run_all_examples():
    print("--- SciPy Solvers ---")
    for example in SCIPY_EXAMPLES:
        example()
        print()

    print("--- IPOPT Solver ---")
    for example in IPOPT_EXAMPLES:
        example()
        print()


if __name__ == "__main__":  # pragma: no cover
    run_all_examples()

