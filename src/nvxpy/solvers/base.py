from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Dict, List, Protocol, Sequence, Tuple

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
    solve_time: float | None = None
    setup_time: float | None = None
    num_iters: int | None = None


@dataclass
class SolverResult:
    x: ArrayLike
    status: SolverStatus
    stats: SolverStats
    raw_result: object | None = None


class SolverBackend(Protocol):
    def solve(
        self,
        problem_data: ProblemData,
        solver: str,
        solver_options: Dict[str, object],
    ) -> SolverResult:
        ...


# =============================================================================
# Shared Solver Utilities
# =============================================================================

def build_objective(problem_data: ProblemData) -> Callable[[ArrayLike], float]:
    """
    Build an objective function from problem data.
    
    This is shared across all solver backends. When `problem_data.compile` is True,
    uses the codegen compiler for better performance. Otherwise uses the interpreter.
    
    Args:
        problem_data: The problem specification containing the objective expression
        
    Returns:
        A callable that takes a flat variable array and returns the objective value
    """
    from ..parser import eval_expression
    from ..compiler import compile_to_function
    
    if problem_data.compile:
        compiled_obj = compile_to_function(problem_data.objective_expr)

        def obj(x):
            var_dict = problem_data.unpack(x)
            return compiled_obj(var_dict)
    else:
        def obj(x):
            var_dict = problem_data.unpack(x)
            return eval_expression(problem_data.objective_expr, var_dict)

    return obj


def uses_projection(problem_data: ProblemData) -> bool:
    """
    Check if any constraint in the problem requires projection.
    
    Projection constraints use the "<-" operator and require a two-phase solve:
    first the main optimization, then a projection step to satisfy the constraint.
    
    Args:
        problem_data: The problem specification
        
    Returns:
        True if any constraint uses the "<-" projection operator
    """
    return any(getattr(c, "op", None) == "<-" for c in problem_data.constraints)


