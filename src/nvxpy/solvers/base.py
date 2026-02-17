from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Dict, List, Protocol, Sequence, Tuple, TYPE_CHECKING

import autograd.numpy as anp  # type: ignore

if TYPE_CHECKING:
    from ..constants import Curvature


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
class ConstraintFn:
    """A constraint function ready for use by solver backends.

    Constraint Residual Semantics:
    -----------------------------
    The `fun` callable returns a residual array where the sign indicates feasibility:

    For equality constraints (type="eq"):
        - residual == 0 means constraint is satisfied
        - Original form: left == right
        - Residual: left - right

    For inequality constraints (type="ineq"):
        - residual >= 0 means constraint is satisfied
        - Original forms are normalized to this convention:
            - left >= right  =>  residual = left - right  (>= 0 when satisfied)
            - left <= right  =>  residual = right - left  (>= 0 when satisfied)

    For semidefinite constraints (>> or <<):
        - residual contains eigenvalues of the matrix difference
        - residual >= 0 means all eigenvalues are non-negative (PSD satisfied)

    For projection constraints (<-):
        - residual = ||x - proj(x)|| (the norm of the difference)
        - Backends apply p_tol to get final constraint: p_tol - norm >= 0
        - Use eval_projection_constraint() helper to evaluate with tolerance

    Attributes:
        fun: Callable[[ArrayLike], ArrayLike] - Takes flat x array, returns residual.
        jac: Optional jacobian function. If None, backends compute via autograd.
        type: "eq" for equality constraints, "ineq" for inequality constraints.
        op: Original operator from the constraint (>=, <=, ==, >>, <<, <-).
        curvature: Curvature of the constraint (CONSTANT, AFFINE, CONVEX, etc.)
    """

    fun: Callable[[ArrayLike], ArrayLike]
    type: str  # "eq" or "ineq"
    op: str
    jac: Callable[[ArrayLike], ArrayLike] | None = None
    curvature: Curvature | None = None


@dataclass
class ProblemData:
    """Data passed to solver backends.

    The Problem class builds this with ready-to-use callables. Solver backends
    should not need to compile or interpret expressions.
    """

    x0: ArrayLike
    var_names: List[str]
    var_shapes: Dict[str, Tuple[int, ...]]
    var_slices: Dict[str, Tuple[int, int]]
    objective_fn: Callable[[ArrayLike], float]
    objective_grad: Callable[[ArrayLike], ArrayLike] | None  # Pre-computed gradient
    constraint_fns: List[ConstraintFn]
    integer_vars: Sequence[str]  # Variables that are integer (includes binary)
    binary_vars: Sequence[str]  # Variables that are binary (0 or 1)
    projection_tolerance: float
    projection_maxiter: int
    presolve: bool
    setup_time: float = 0.0
    # Original constraints kept for BnB branching logic
    constraints: Sequence[Any] | None = None
    # SOS1 groups: each group is a list of variable indices that must sum to 1
    # Used by rounding heuristics to maintain feasibility when rounding indicators
    sos1_groups: List[List[int]] | None = None

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
    ) -> SolverResult: ...


def uses_projection(problem_data: ProblemData) -> bool:
    """Check if any constraint uses the projection operator (<-)."""
    return any(c.op == "<-" for c in problem_data.constraint_fns)


# Standard scipy status codes (shared by scipy_backend and global_scipy_backend)
SCIPY_STATUS_MAP = {
    0: SolverStatus.OPTIMAL,
    1: SolverStatus.MAX_ITERATIONS,
    2: SolverStatus.INFEASIBLE,
    3: SolverStatus.UNBOUNDED,
    4: SolverStatus.NUMERICAL_ERROR,
}


def eval_projection_constraint(
    con_fn: ConstraintFn, x: ArrayLike, p_tol: float
) -> ArrayLike:
    """Evaluate a projection constraint with tolerance.

    For projection constraints (<-), the constraint function returns ||x - proj(x)||.
    This helper computes the final residual: p_tol - ||x - proj(x)||
    which is >= 0 when the constraint is satisfied.

    Args:
        con_fn: The constraint function (must have op == "<-")
        x: The point at which to evaluate
        p_tol: The projection tolerance

    Returns:
        Residual array where >= 0 means satisfied
    """
    norm = con_fn.fun(x)  # Returns ||x - proj(x)||
    return p_tol - norm


def wrap_projection_constraint(
    con_fn: ConstraintFn, p_tol: float
) -> Callable[[ArrayLike], ArrayLike]:
    """Wrap a projection constraint function to apply the tolerance.

    Returns a new function that computes: p_tol - ||x - proj(x)||

    Args:
        con_fn: The constraint function (must have op == "<-")
        p_tol: The projection tolerance

    Returns:
        Wrapped function suitable for scipy (returns >= 0 when satisfied)
    """

    def wrapped(x: ArrayLike) -> ArrayLike:
        norm = con_fn.fun(x)
        return p_tol - norm

    return wrapped


def downgrade_for_projection(
    base_status: SolverStatus,
    projection_result,
) -> SolverStatus:
    """Downgrade status to SUBOPTIMAL if projection failed.

    Args:
        base_status: The status from the main solve
        projection_result: Result from projection phase (or None)

    Returns:
        base_status if projection succeeded, SUBOPTIMAL if it failed
    """
    if projection_result is None:
        return base_status

    # Check scipy-style result
    if hasattr(projection_result, "success"):
        if not getattr(projection_result, "success", True):
            return SolverStatus.SUBOPTIMAL
    # Check dict-style result (IPOPT)
    elif isinstance(projection_result, dict):
        proj_status = projection_result.get("status", 0)
        # IPOPT success codes: 0 (Solve_Succeeded), 1 (Acceptable), 6 (Feasible)
        if proj_status not in [0, 1, 6]:
            return SolverStatus.SUBOPTIMAL

    return base_status


def extract_simple_bounds(problem_data: ProblemData) -> Dict[int, Tuple[float, float]]:
    """
    Extract simple variable bounds from constraints.

    Looks for constraints of the form:
    - var >= constant  (lower bound)
    - var <= constant  (upper bound)
    - constant <= var  (lower bound)
    - constant >= var  (upper bound)

    Returns a dict mapping flat index -> (lb, ub).
    """
    from ..variable import Variable

    var_bounds: Dict[int, Tuple[float, float]] = {}
    n_vars = len(problem_data.x0)

    # Initialize with no bounds
    for i in range(n_vars):
        var_bounds[i] = (float("-inf"), float("inf"))

    if problem_data.constraints is None:
        return {}

    for constraint in problem_data.constraints:
        if constraint.op not in (">=", "<=", "=="):
            continue

        # Check if this is a simple bound: var op constant or constant op var
        left = constraint.left
        right = constraint.right

        var = None
        const = None
        is_var_on_left = False

        if isinstance(left, Variable) and isinstance(right, (int, float)):
            var = left
            const = float(right)
            is_var_on_left = True
        elif isinstance(right, Variable) and isinstance(left, (int, float)):
            var = right
            const = float(left)
            is_var_on_left = False
        else:
            continue

        if var.name not in problem_data.var_slices:
            continue

        start, end = problem_data.var_slices[var.name]

        # Determine bound type based on operator and which side var is on
        for idx in range(start, end):
            current_lb, current_ub = var_bounds[idx]

            if constraint.op == ">=":
                if is_var_on_left:
                    # var >= const -> lower bound
                    current_lb = max(current_lb, const)
                else:
                    # const >= var -> upper bound
                    current_ub = min(current_ub, const)
            elif constraint.op == "<=":
                if is_var_on_left:
                    # var <= const -> upper bound
                    current_ub = min(current_ub, const)
                else:
                    # const <= var -> lower bound
                    current_lb = max(current_lb, const)
            elif constraint.op == "==":
                # var == const -> fixed
                current_lb = max(current_lb, const)
                current_ub = min(current_ub, const)

            var_bounds[idx] = (current_lb, current_ub)

    # Remove entries with no actual bounds (both inf)
    return {
        idx: bounds
        for idx, bounds in var_bounds.items()
        if bounds[0] > float("-inf") or bounds[1] < float("inf")
    }


def remove_redundant_equality_constraints(
    constraint_fns: List[ConstraintFn],
    x0: ArrayLike,
    tol: float = 1e-10,
) -> Tuple[List[ConstraintFn], int]:
    """
    Remove linearly dependent AFFINE equality constraints.

    Only processes constraints that are affine (constant Jacobian).
    Nonlinear equality constraints are never removed since their
    Jacobian varies with x, making point-wise redundancy checks invalid.

    Uses QR decomposition with column pivoting to identify and remove
    redundant affine constraints. This prevents singular Jacobian errors
    in solvers.

    This is common in graph problems where degree constraints are
    naturally redundant (e.g., sum of in-degrees = sum of out-degrees).

    Args:
        constraint_fns: List of ConstraintFn objects
        x0: Initial point for Jacobian evaluation
        tol: Tolerance for determining linear dependence

    Returns:
        Tuple of (filtered constraints, number removed)
    """
    from ..constants import Curvature

    # Separate constraints by type and curvature
    # Only affine equality constraints are candidates for redundancy removal
    affine_eq_cons = []
    other_cons = []

    for i, c in enumerate(constraint_fns):
        is_affine = (
            c.curvature in (Curvature.CONSTANT, Curvature.AFFINE)
            if c.curvature
            else False
        )

        if c.type == "eq" and is_affine:
            affine_eq_cons.append((i, c))
        else:
            other_cons.append(c)

    if len(affine_eq_cons) <= 1:
        return constraint_fns, 0

    # Build Jacobian matrix for affine equality constraints only
    try:
        from scipy.linalg import qr as scipy_qr
        from autograd import jacobian

        jac_rows = []
        eq_indices = []
        for orig_idx, c in affine_eq_cons:
            # Use pre-computed jacobian if available, otherwise compute
            jac_fn = c.jac if c.jac else jacobian(c.fun)
            jac = jac_fn(x0)
            if jac.ndim == 1:
                jac = jac.reshape(1, -1)
            for row in jac:
                jac_rows.append(row)
                eq_indices.append(orig_idx)

        if not jac_rows:
            return constraint_fns, 0

        J = anp.vstack(jac_rows)
        n_rows = J.shape[0]

        # Use QR decomposition with column pivoting on J^T to find row rank
        # pivoting=True returns permutation P where P[:rank] are independent rows
        _, R, P = scipy_qr(J.T, pivoting=True)

        # Find rank by counting significant diagonal elements of R
        diag_R = anp.abs(anp.diag(R))
        max_diag = anp.max(diag_R) if len(diag_R) > 0 else 0
        if max_diag > 0:
            rank = int(anp.sum(diag_R > tol * max_diag))
        else:
            rank = 0

        if rank == n_rows:
            # All affine equality constraints are independent
            return constraint_fns, 0

        # Keep only the first 'rank' rows (those corresponding to
        # linearly independent constraints)
        # P gives the permutation - P[:rank] are the indices of independent rows
        keep_row_indices = set(P[:rank])

        # Map back to original constraint indices
        keep_constraint_indices = set()
        for row_idx in keep_row_indices:
            keep_constraint_indices.add(eq_indices[row_idx])

        # Build filtered constraint list: independent affine eq + all others
        filtered_affine_eq = [
            c for orig_idx, c in affine_eq_cons if orig_idx in keep_constraint_indices
        ]

        n_removed = len(affine_eq_cons) - len(filtered_affine_eq)

        return filtered_affine_eq + other_cons, n_removed

    except Exception:
        # If anything fails, return original constraints
        return constraint_fns, 0
