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
from autograd import grad
from scipy.optimize import (
    basinhopping,
    differential_evolution,
    dual_annealing,
    NonlinearConstraint,
    shgo,
)

from ..constants import Solver, DEFAULT_FEASIBILITY_TOL
from .base import (
    uses_projection,
    downgrade_for_projection,
    wrap_projection_constraint,
    extract_simple_bounds,
    ConstraintFn,
    ProblemData,
    SolverResult,
    SolverStats,
    SolverStatus,
)


def _build_nonlinear_constraints(
    constraint_fns: List[ConstraintFn],
    for_projection: bool = False,
    projection_tolerance: float = 0.0,
) -> List[NonlinearConstraint]:
    """Build NonlinearConstraint objects for scipy global optimizers.

    For projection constraints (<-):
    - Main solve (for_projection=False): Apply p_tol so constraint is
      p_tol - ||x - proj(x)|| >= 0, i.e., ||x - proj(x)|| <= p_tol
    - Projection phase (for_projection=True): Enforce equality ||x - proj(x)|| == 0
    """
    constraints: List[NonlinearConstraint] = []

    for c in constraint_fns:
        from autograd import jacobian

        if c.op == "<-":
            if for_projection:
                # Projection phase: enforce equality (norm == 0)
                lb, ub = 0.0, 0.0
                con_fun = c.fun
            else:
                # Main solve: apply tolerance (p_tol - norm >= 0)
                lb, ub = 0.0, np.inf
                con_fun = wrap_projection_constraint(c, projection_tolerance)
        elif c.type == "eq":
            lb, ub = 0.0, 0.0
            con_fun = c.fun
        else:  # ineq: residual >= 0
            lb, ub = 0.0, np.inf
            con_fun = c.fun

        con_jac = jacobian(con_fun)
        constraints.append(NonlinearConstraint(con_fun, lb, ub, jac=con_jac))

    return constraints


def _resolve_bounds(
    problem_data: ProblemData,
    bounds_opt,
    require_finite: bool = True,
) -> List[Tuple[float, float]]:
    """Resolve variable bounds from user options or constraints."""
    n_vars = len(problem_data.x0)

    if bounds_opt is not None:
        bounds_list = list(bounds_opt)
        if len(bounds_list) != n_vars:
            raise ValueError(
                f"bounds must have length {n_vars}, got {len(bounds_list)}"
            )
        return bounds_list

    simple_bounds = extract_simple_bounds(problem_data)
    bounds: List[Tuple[float, float]] = []
    for i in range(n_vars):
        lb, ub = simple_bounds.get(i, (-np.inf, np.inf))
        bounds.append((lb, ub))

    if require_finite and any(np.isinf(lb) or np.isinf(ub) for lb, ub in bounds):
        raise ValueError(
            "This solver requires finite bounds for all variables. "
            "Add bound constraints for each variable: `x >= lower, x <= upper`."
        )
    return bounds


def _has_nonbound_constraints(problem_data: ProblemData) -> bool:
    """Check if problem has constraints beyond simple variable bounds."""
    from ..variable import Variable

    if problem_data.constraints is None:
        return bool(problem_data.constraint_fns)

    for constraint in problem_data.constraints:
        if constraint.op not in (">=", "<="):
            return True

        left = constraint.left
        right = constraint.right

        is_simple = (
            isinstance(left, Variable) and isinstance(right, (int, float))
        ) or (isinstance(right, Variable) and isinstance(left, (int, float)))
        if not is_simple:
            return True

    return False


def _is_feasible(
    x: np.ndarray,
    constraints: List[NonlinearConstraint],
    tol: float = DEFAULT_FEASIBILITY_TOL,
) -> bool:
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
    """Backend for SciPy global optimizers (DE, dual annealing, SHGO, basinhopping)."""

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

        obj_func = problem_data.objective_fn
        obj_grad = grad(obj_func)

        options = dict(solver_options)
        bounds_opt = options.pop("bounds", None)

        compile_start = time.time()
        require_finite = method in self.REQUIRES_FINITE_BOUNDS
        bounds = _resolve_bounds(
            problem_data, bounds_opt, require_finite=require_finite
        )
        setup_time += time.time() - compile_start

        constraints: List[NonlinearConstraint] = []
        has_nonbound_constraints = _has_nonbound_constraints(problem_data)

        if has_nonbound_constraints:
            if method in self.CONSTRAINT_AWARE_METHODS:
                compile_start = time.time()
                constraints = _build_nonlinear_constraints(
                    problem_data.constraint_fns,
                    for_projection=False,
                    projection_tolerance=problem_data.projection_tolerance,
                )
                setup_time += time.time() - compile_start
            else:
                raise ValueError(
                    f"Solver '{method}' does not support constraints. "
                    f"Use one of: {', '.join(sorted(self.CONSTRAINT_AWARE_METHODS))}"
                )

        start_time = time.time()
        result = self._run_optimizer(
            method, obj_func, obj_grad, x0, bounds, constraints, options
        )
        solve_time += time.time() - start_time

        x_sol = getattr(result, "x", None)
        if x_sol is not None:
            x_sol = np.asarray(x_sol, dtype=float)

        projection_result = None
        if uses_projection(problem_data) and method in self.CONSTRAINT_AWARE_METHODS:
            compile_start = time.time()
            proj_constraints = _build_nonlinear_constraints(
                problem_data.constraint_fns, for_projection=True
            )
            setup_time += time.time() - compile_start

            if proj_constraints and x_sol is not None:

                def dummy_obj(_):
                    return 0.0

                start_time = time.time()
                projection_result = self._run_optimizer(
                    method,
                    dummy_obj,
                    lambda x: np.zeros_like(x),
                    x_sol,
                    bounds,
                    proj_constraints,
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
        """Run the appropriate scipy global optimizer."""
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
            minimizer_kwargs = {"jac": obj_grad}
            has_finite_bounds = bounds and not any(
                np.isinf(lb) or np.isinf(ub) for lb, ub in bounds
            )
            if has_finite_bounds:
                minimizer_kwargs["bounds"] = bounds
                minimizer_kwargs["method"] = "L-BFGS-B"
            else:
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
        """Interpret scipy OptimizeResult to SolverStatus."""
        x = getattr(result, "x", None)
        success = bool(getattr(result, "success", False))

        if (
            constraints
            and x is not None
            and not _is_feasible(np.asarray(x, dtype=float), constraints)
        ):
            return SolverStatus.INFEASIBLE

        if success:
            return downgrade_for_projection(SolverStatus.OPTIMAL, projection_result)

        message = getattr(result, "message", "")
        if isinstance(message, list):
            message = " ".join(message)
        message = str(message).lower()

        if "success" in message:
            return downgrade_for_projection(SolverStatus.OPTIMAL, projection_result)

        if "maximum" in message and ("iteration" in message or "evaluation" in message):
            return SolverStatus.MAX_ITERATIONS

        status_code = getattr(result, "status", None)
        if status_code is not None:
            status_map = {
                0: SolverStatus.OPTIMAL,
                1: SolverStatus.MAX_ITERATIONS,
                2: SolverStatus.MAX_ITERATIONS,
            }
            return status_map.get(status_code, SolverStatus.ERROR)

        return SolverStatus.ERROR
