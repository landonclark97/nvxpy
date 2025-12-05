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
from .bnb import BranchAndBoundBackend
from .global_scipy_backend import GlobalScipyBackend


_SCIPY_BACKEND = ScipyBackend()
_IPOPT_BACKEND = IpoptBackend()
_BNB_BACKEND = BranchAndBoundBackend()
_GLOBAL_SCIPY_BACKEND = GlobalScipyBackend()


_SOLVER_BACKENDS: Dict[str, SolverBackend] = {
    # Gradient-free
    Solver.NELDER_MEAD.value: _SCIPY_BACKEND,
    Solver.POWELL.value: _SCIPY_BACKEND,
    Solver.COBYLA.value: _SCIPY_BACKEND,
    Solver.COBYQA.value: _SCIPY_BACKEND,
    # Gradient-based
    Solver.CG.value: _SCIPY_BACKEND,
    Solver.BFGS.value: _SCIPY_BACKEND,
    Solver.LBFGSB.value: _SCIPY_BACKEND,
    Solver.TNC.value: _SCIPY_BACKEND,
    Solver.SLSQP.value: _SCIPY_BACKEND,
    # Hessian-based
    Solver.NEWTON_CG.value: _SCIPY_BACKEND,
    Solver.DOGLEG.value: _SCIPY_BACKEND,
    Solver.TRUST_NCG.value: _SCIPY_BACKEND,
    Solver.TRUST_KRYLOV.value: _SCIPY_BACKEND,
    Solver.TRUST_EXACT.value: _SCIPY_BACKEND,
    Solver.TRUST_CONSTR.value: _SCIPY_BACKEND,
    # Global optimizers
    Solver.DIFF_EVOLUTION.value: _GLOBAL_SCIPY_BACKEND,
    Solver.DUAL_ANNEALING.value: _GLOBAL_SCIPY_BACKEND,
    Solver.SHGO.value: _GLOBAL_SCIPY_BACKEND,
    Solver.BASINHOPPING.value: _GLOBAL_SCIPY_BACKEND,
    # Other
    Solver.IPOPT.value: _IPOPT_BACKEND,
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

