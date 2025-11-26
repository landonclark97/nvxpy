from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import autograd.numpy as anp  # type: ignore


ArrayLike = anp.ndarray


class SolverStatus(StrEnum):
    OPTIMAL = "optimal"
    SUBOPTIMAL = "suboptimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    MAX_ITERATIONS = "max_iterations"
    NUMERICAL_ERROR = "numerical_error"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ConstraintData:
    type: str
    fun: Callable[[ArrayLike], ArrayLike]
    jac: Callable[[ArrayLike], ArrayLike]
    op: str


@dataclass
class ProblemData:
    x0: ArrayLike
    var_names: List[str]
    var_shapes: Dict[str, Tuple[int, ...]]
    var_slices: Dict[str, Tuple[int, int]]
    objective_expr: Any
    constraints: Sequence[Any]
    integer_vars: Sequence[str]
    projection_tolerance: float
    projection_maxiter: int
    presolve: bool
    compile: bool = False
    setup_time: float = 0.0

    def unpack(self, x: Sequence[float]) -> Dict[str, ArrayLike]:
        if isinstance(x, anp.ndarray):
            arr = x
        else:
            arr = anp.array(x)
        var_dict: Dict[str, ArrayLike] = {}
        for name in self.var_names:
            start, end = self.var_slices[name]
            shape = self.var_shapes[name]
            segment = arr[start:end]
            if shape:
                var_dict[name] = segment.reshape(shape)
            else:
                var_dict[name] = segment[0] if segment.size == 1 else segment
        return var_dict


@dataclass
class SolverStats:
    solver_name: str
    solve_time: Optional[float] = None
    setup_time: Optional[float] = None
    num_iters: Optional[int] = None


@dataclass
class SolverResult:
    x: ArrayLike
    status: SolverStatus
    stats: SolverStats
    raw_result: Optional[object] = None


class SolverBackend(Protocol):
    def solve(
        self,
        problem_data: ProblemData,
        solver: str,
        solver_options: Dict[str, object],
    ) -> SolverResult:
        ...


