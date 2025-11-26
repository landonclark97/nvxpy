from __future__ import annotations

from typing import Dict

from ..constants import Solver
from .base import (
    ConstraintData,
    ProblemData,
    SolverBackend,
    SolverResult,
    SolverStats,
    SolverStatus,
)
from .scipy_backend import ScipyBackend
from .ipopt_backend import IpoptBackend
from .bnb_backend import BranchAndBoundBackend


_SCIPY_BACKEND = ScipyBackend()
_IPOPT_BACKEND = IpoptBackend()
_BNB_BACKEND = BranchAndBoundBackend()


_SOLVER_BACKENDS: Dict[str, SolverBackend] = {
    Solver.SLSQP.value: _SCIPY_BACKEND,
    Solver.IPOPT.value: _IPOPT_BACKEND,
    Solver.COBYLA.value: _SCIPY_BACKEND,
    Solver.NELDER_MEAD.value: _SCIPY_BACKEND,
    Solver.BFGS.value: _SCIPY_BACKEND,
    Solver.LBFGSB.value: _SCIPY_BACKEND,
    Solver.TNC.value: _SCIPY_BACKEND,
    Solver.TRUST_CONSTR.value: _SCIPY_BACKEND,
    Solver.BNB.value: _BNB_BACKEND,
}


def register_solver_backend(solver_name: str, backend: SolverBackend) -> None:
    _SOLVER_BACKENDS[solver_name] = backend


def get_solver_backend(solver: Solver | str) -> SolverBackend:
    solver_name = solver.value if isinstance(solver, Solver) else str(solver)
    if solver_name not in _SOLVER_BACKENDS:
        raise ValueError(f"No solver backend registered for solver '{solver_name}'")
    return _SOLVER_BACKENDS[solver_name]


__all__ = [
    "ConstraintData",
    "ProblemData",
    "SolverBackend",
    "SolverResult",
    "SolverStats",
    "SolverStatus",
    "IpoptBackend",
    "get_solver_backend",
    "register_solver_backend",
]


