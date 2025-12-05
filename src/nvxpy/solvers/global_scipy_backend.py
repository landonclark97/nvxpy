"""Backend for SciPy global optimizers.

This module provides a unified backend for stochastic and deterministic global
optimization algorithms from scipy.optimize, including:
- differential_evolution: Evolutionary optimization with constraint support
- dual_annealing: Generalized simulated annealing
- shgo: Simplicial homology global optimization with constraint support
- basinhopping: Basin-hopping with local minimization
"""

from __future__ import annotations

import time
import warnings
from typing import Dict, List, Tuple

import autograd.numpy as np
from autograd import grad, jacobian
from scipy.optimize import (
    basinhopping,
    differential_evolution,
    dual_annealing,
    NonlinearConstraint,
    shgo,
)

from ..compiler import compile_to_function
from ..constants import EPSILON, Solver
from ..parser import eval_expression
from .base import (
    build_objective,
    uses_projection,
    ProblemData,
    SolverResult,
    SolverStats,
    SolverStatus,
)
from .bnb.utils import extract_simple_bounds


def _constraint_bounds(op: str, for_projection: bool = False) -> Tuple[float, float]:
    """Get constraint bounds for NonlinearConstraint.

    All constraints are normalized to the form: lb <= g(x) <= ub
    where g(x) = lval - rval for most operators.

    Args:
        op: Constraint operator (==, >=, <=, >>, <<, <-)
        for_projection: If True, treat <- as equality constraint

    Returns:
        Tuple of (lower_bound, upper_bound) for the constraint
    """
    if op == "==" or (for_projection and op == "<-"):
        # lval == rval  =>  lval - rval == 0  =>  0 <= g(x) <= 0
        return 0.0, 0.0
    if op in (">=", ">>", "<-"):
        # lval >= rval  =>  lval - rval >= 0  =>  0 <= g(x) <= inf
        return 0.0, np.inf
    if op in ("<=", "<<"):
        # lval <= rval  =>  lval - rval <= 0  =>  -inf <= g(x) <= 0
        return -np.inf, 0.0
    raise ValueError(f"Unsupported constraint operator '{op}' for global optimizers")


def _constraint_residual(op: str, lval, rval, p_tol: float, for_projection: bool = False):
    """Compute constraint residual value g(x) = lval - rval.

    All constraints use the same residual formula (lval - rval).
    The constraint type is encoded in the bounds, not the residual direction.

    Args:
        op: Constraint operator
        lval: Left-hand side value
        rval: Right-hand side value
        p_tol: Projection tolerance
        for_projection: If True, treat <- as element-wise equality

    Returns:
        Flattened residual array
    """
    if op == "<-":
        if for_projection:
            # During projection, enforce element-wise equality
            res = lval - rval
        else:
            # During main solve, use smoothed norm constraint with epsilon
            diff = np.atleast_2d(lval - rval)
            res = p_tol - np.sqrt(np.sum(diff * diff) + EPSILON)
        return np.ravel(res)

    # All other operators use lval - rval as the residual
    # The constraint type (>=, <=, ==) is encoded in the bounds
    res = lval - rval

    if op in [">>", "<<"]:
        # PSD/NSD: check eigenvalues
        res = np.real(np.ravel(np.linalg.eig(res)[0]))

    return np.ravel(res)


def _build_nonlinear_constraints(
    problem_data: ProblemData,
    for_projection: bool = False,
) -> List[NonlinearConstraint]:
    """Build NonlinearConstraint objects for scipy global optimizers.

    Args:
        problem_data: Problem specification
        for_projection: If True, treat <- constraints as equality constraints

    Returns:
        List of NonlinearConstraint objects
    """
    constraints: List[NonlinearConstraint] = []
    p_tol = problem_data.projection_tolerance
    use_compile = problem_data.compile

    for constraint in problem_data.constraints:
        if constraint.op == "in":
            raise ValueError(
                "Global SciPy optimizers do not support discrete set membership. "
                "Use nvx.BNB for mixed-integer problems."
            )

        def make_con_fun(c, compile_exprs=use_compile, proj=for_projection):
            if compile_exprs:
                compiled_left = compile_to_function(c.left)
                compiled_right = compile_to_function(c.right)

                def con_fun(x):
                    var_dict = problem_data.unpack(x)
                    lval = compiled_left(var_dict)
                    rval = compiled_right(var_dict)
                    return _constraint_residual(c.op, lval, rval, p_tol, proj)
            else:

                def con_fun(x):
                    var_dict = problem_data.unpack(x)
                    lval = eval_expression(c.left, var_dict)
                    rval = eval_expression(c.right, var_dict)
                    return _constraint_residual(c.op, lval, rval, p_tol, proj)

            return con_fun

        con_fun = make_con_fun(constraint)
        lb, ub = _constraint_bounds(constraint.op, for_projection)
        con_jac = jacobian(con_fun)
        constraints.append(NonlinearConstraint(con_fun, lb, ub, jac=con_jac))

    return constraints


def _resolve_bounds(
    problem_data: ProblemData,
    bounds_opt,
    require_finite: bool = True,
) -> List[Tuple[float, float]]:
    """Resolve variable bounds from user options or constraints.

    Args:
        problem_data: Problem specification
        bounds_opt: User-provided bounds or None
        require_finite: If True, raise error for infinite bounds

    Returns:
        List of (lb, ub) tuples for each variable

    Raises:
        ValueError: If require_finite=True and any variable has infinite bounds
    """
    n_vars = len(problem_data.x0)

    # User-provided bounds take precedence
    if bounds_opt is not None:
        bounds_list = list(bounds_opt)
        if len(bounds_list) != n_vars:
            raise ValueError(f"bounds must have length {n_vars}, got {len(bounds_list)}")
        return bounds_list

    simple_bounds = extract_simple_bounds(problem_data)
    bounds: List[Tuple[float, float]] = []
    for i in range(n_vars):
        lb, ub = simple_bounds.get(i, (-np.inf, np.inf))
        bounds.append((lb, ub))

    if require_finite and any(np.isinf(lb) or np.isinf(ub) for lb, ub in bounds):
        raise ValueError(
            "This solver requires finite bounds for all variables. "
            "Pass `bounds=[(lb, ub), ...]` in solver_options or add bound constraints."
        )
    return bounds


def _has_nonbound_constraints(problem_data: ProblemData) -> bool:
    """Check if problem has constraints beyond simple variable bounds.

    Simple bound constraints (x >= c, x <= c, c >= x, c <= x) are handled
    via the bounds parameter, not as NonlinearConstraints.
    """
    from ..variable import Variable

    for constraint in problem_data.constraints:
        if constraint.op not in (">=", "<="):
            return True

        left = constraint.left
        right = constraint.right

        # Check if it's a simple bound: var op constant or constant op var
        is_simple = (isinstance(left, Variable) and isinstance(right, (int, float))) or (
            isinstance(right, Variable) and isinstance(left, (int, float))
        )
        if not is_simple:
            return True

    return False


def _is_feasible(x: np.ndarray, constraints: List[NonlinearConstraint], tol: float = 1e-6) -> bool:
    """Check feasibility against NonlinearConstraint objects."""
    if not constraints:
        return True
    for c in constraints:
        vals = c.fun(x)
        lb = np.broadcast_to(c.lb, vals.shape)
        ub = np.broadcast_to(c.ub, vals.shape)
        if np.any(vals < lb - tol) or np.any(vals > ub + tol):
            return False
    return True


class GlobalScipyBackend:
    """Backend for SciPy global optimizers (DE, dual annealing, SHGO, basinhopping).

    This backend provides access to stochastic and deterministic global optimization
    methods from scipy.optimize. These methods are useful for non-convex problems
    where local optimizers may get stuck in local minima.

    Attributes:
        SUPPORTED_METHODS: Set of supported solver names
        CONSTRAINT_AWARE: Subset of methods that support nonlinear constraints
    """

    SUPPORTED_METHODS = {
        Solver.DIFF_EVOLUTION.value,
        Solver.DUAL_ANNEALING.value,
        Solver.SHGO.value,
        Solver.BASINHOPPING.value,
    }

    CONSTRAINT_AWARE_METHODS = {
        Solver.DIFF_EVOLUTION.value,
        Solver.SHGO.value,
    }

    # Methods that require finite bounds on all variables
    REQUIRES_FINITE_BOUNDS = {
        Solver.DIFF_EVOLUTION.value,
        Solver.DUAL_ANNEALING.value,
        Solver.SHGO.value,
    }

    def solve(
        self,
        problem_data: ProblemData,
        solver: str,
        solver_options: Dict[str, object],
    ) -> SolverResult:
        method = str(solver)
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Solver '{method}' is not supported by the global SciPy backend. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_METHODS))}"
            )

        if problem_data.integer_vars:
            raise ValueError(
                "Global SciPy optimizers do not support integer variables. "
                "Use nvx.BNB for mixed-integer problems."
            )

        x0 = np.asarray(problem_data.x0, dtype=float)
        setup_time = problem_data.setup_time
        solve_time = 0.0

        # Build objective (uses compile flag internally)
        obj_func = build_objective(problem_data)
        obj_grad = grad(obj_func)

        options = dict(solver_options)
        bounds_opt = options.pop("bounds", None)

        compile_start = time.time()
        require_finite = method in self.REQUIRES_FINITE_BOUNDS
        bounds = _resolve_bounds(problem_data, bounds_opt, require_finite=require_finite)
        setup_time += time.time() - compile_start

        # Build constraints for main solve
        # Note: simple bound constraints (x >= c, x <= c) are already handled
        # via _resolve_bounds, so we only need NonlinearConstraints for others
        constraints: List[NonlinearConstraint] = []
        has_nonbound_constraints = _has_nonbound_constraints(problem_data)

        if has_nonbound_constraints:
            if method in self.CONSTRAINT_AWARE_METHODS:
                compile_start = time.time()
                constraints = _build_nonlinear_constraints(problem_data, for_projection=False)
                setup_time += time.time() - compile_start
            else:
                raise ValueError(
                    f"Solver '{method}' does not support constraints. "
                    f"Use one of: {', '.join(sorted(self.CONSTRAINT_AWARE_METHODS))}"
                )

        start_time = time.time()
        result = self._run_optimizer(method, obj_func, obj_grad, x0, bounds, constraints, options)
        solve_time += time.time() - start_time

        x_sol = getattr(result, "x", None)
        if x_sol is not None:
            x_sol = np.asarray(x_sol, dtype=float)

        # Projection phase for <- constraints
        projection_result = None
        if uses_projection(problem_data) and method in self.CONSTRAINT_AWARE_METHODS:
            compile_start = time.time()
            proj_constraints = _build_nonlinear_constraints(problem_data, for_projection=True)
            setup_time += time.time() - compile_start

            if proj_constraints and x_sol is not None:
                # Use a dummy objective to just find a feasible point
                def dummy_obj(_):
                    return 0.0

                start_time = time.time()
                projection_result = self._run_optimizer(
                    method, dummy_obj, lambda x: np.zeros_like(x),
                    x_sol, bounds, proj_constraints,
                    {**options, "maxiter": problem_data.projection_maxiter},
                )
                solve_time += time.time() - start_time

                if getattr(projection_result, "success", False):
                    x_sol = np.asarray(projection_result.x, dtype=float)
                else:
                    warnings.warn(
                        f"Projection step failed with status {getattr(projection_result, 'status', 'unknown')}"
                    )

        status = self._interpret_status(result, constraints, projection_result)
        stats = SolverStats(
            solver_name=method,
            solve_time=solve_time,
            setup_time=setup_time,
            num_iters=getattr(result, "nit", None),
        )

        raw_result = {
            "primary": result,
            "projection": projection_result,
        }

        return SolverResult(
            x=x_sol,
            status=status,
            stats=stats,
            raw_result=raw_result,
        )

    @staticmethod
    def _run_optimizer(
        method: str,
        obj_func,
        obj_grad,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]],
        constraints: List[NonlinearConstraint],
        options: Dict[str, object],
    ):
        """Run the appropriate scipy global optimizer.

        Args:
            method: Solver method name
            obj_func: Objective function
            obj_grad: Objective gradient (unused by most global methods)
            x0: Initial point
            bounds: Variable bounds
            constraints: NonlinearConstraint objects
            options: Solver-specific options

        Returns:
            OptimizeResult from scipy
        """
        if method == Solver.DIFF_EVOLUTION.value:
            return differential_evolution(
                obj_func,
                bounds=bounds,
                constraints=constraints if constraints else (),
                **options,
            )
        elif method == Solver.DUAL_ANNEALING.value:
            return dual_annealing(obj_func, bounds=bounds, x0=x0, **options)
        elif method == Solver.SHGO.value:
            return shgo(
                obj_func,
                bounds=bounds,
                constraints=constraints if constraints else None,
                **options,
            )
        elif method == Solver.BASINHOPPING.value:
            # basinhopping uses local minimizer; bounds are optional
            minimizer_kwargs = {"jac": obj_grad}
            has_finite_bounds = bounds and not any(
                np.isinf(lb) or np.isinf(ub) for lb, ub in bounds
            )
            if has_finite_bounds:
                minimizer_kwargs["bounds"] = bounds
                minimizer_kwargs["method"] = "L-BFGS-B"
            else:
                # CG (conjugate gradient) is faster than BFGS for large unconstrained problems
                minimizer_kwargs["method"] = "CG"
            return basinhopping(
                obj_func,
                x0=x0,
                minimizer_kwargs=minimizer_kwargs,
                **options,
            )
        else:
            raise ValueError(f"Unhandled solver '{method}' in global SciPy backend")

    @staticmethod
    def _interpret_status(
        result,
        constraints: List[NonlinearConstraint],
        projection_result=None,
    ) -> SolverStatus:
        """Interpret scipy OptimizeResult to SolverStatus.

        Args:
            result: Primary optimization result
            constraints: List of constraints for feasibility check
            projection_result: Optional projection phase result

        Returns:
            Appropriate SolverStatus
        """
        x = getattr(result, "x", None)
        success = bool(getattr(result, "success", False))

        # Feasibility check has priority when constraints exist
        if constraints and x is not None and not _is_feasible(np.asarray(x, dtype=float), constraints):
            return SolverStatus.INFEASIBLE

        if success:
            # Check projection result if present
            if projection_result is not None and not getattr(projection_result, "success", True):
                return SolverStatus.SUBOPTIMAL
            return SolverStatus.OPTIMAL

        # Check message for success/failure patterns
        # Note: basinhopping may report success=False from local minimizer but complete successfully
        message = getattr(result, "message", "")
        if isinstance(message, list):
            message = " ".join(message)
        message = str(message).lower()

        if "success" in message:
            if projection_result is not None and not getattr(projection_result, "success", True):
                return SolverStatus.SUBOPTIMAL
            return SolverStatus.OPTIMAL

        if "maximum" in message and ("iteration" in message or "evaluation" in message):
            return SolverStatus.MAX_ITERATIONS

        # Try status code if available (some optimizers like shgo use it)
        status_code = getattr(result, "status", None)
        if status_code is not None:
            status_map = {
                0: SolverStatus.OPTIMAL,
                1: SolverStatus.MAX_ITERATIONS,
                2: SolverStatus.MAX_ITERATIONS,
            }
            return status_map.get(status_code, SolverStatus.ERROR)

        return SolverStatus.ERROR
