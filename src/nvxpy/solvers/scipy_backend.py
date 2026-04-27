from __future__ import annotations

import time
import warnings
from typing import Dict

import autograd.numpy as np
from autograd import grad, jacobian, hessian
from scipy.optimize import minimize

from .base import (
    uses_projection,
    downgrade_for_projection,
    wrap_projection_constraint,
    extract_simple_bounds,
    is_simple_bound_constraint,
    SCIPY_STATUS_MAP,
    ProblemData,
    SolverResult,
    SolverStats,
    SolverStatus,
)


class ScipyBackend:
    SUPPORTED_METHODS = {
        # Gradient-free
        "Nelder-Mead",
        "Powell",
        "COBYLA",
        "COBYQA",
        # Gradient-based
        "CG",
        "BFGS",
        "L-BFGS-B",
        "TNC",
        "SLSQP",
        # Hessian-based
        "Newton-CG",
        "dogleg",
        "trust-ncg",
        "trust-krylov",
        "trust-exact",
        "trust-constr",
    }

    GRADIENT_METHODS = {
        "CG",
        "BFGS",
        "L-BFGS-B",
        "TNC",
        "SLSQP",
        "Newton-CG",
        "dogleg",
        "trust-ncg",
        "trust-krylov",
        "trust-exact",
        "trust-constr",
    }

    HESSIAN_METHODS = {
        "Newton-CG",
        "dogleg",
        "trust-ncg",
        "trust-krylov",
        "trust-exact",
    }

    CONSTRAINED_METHODS = {
        "SLSQP",
        "COBYLA",
        "COBYQA",
        "trust-constr",
    }

    BOUND_METHODS = {
        "Nelder-Mead",
        "Powell",
        "L-BFGS-B",
        "TNC",
        "SLSQP",
        "COBYLA",
        "COBYQA",
        "trust-constr",
    }

    def solve(
        self,
        problem_data: ProblemData,
        solver: str,
        solver_options: Dict[str, object],
    ) -> SolverResult:
        method = str(solver)
        if method not in self.SUPPORTED_METHODS:
            raise ValueError(f"Solver '{method}' is not supported by the SciPy backend")

        if problem_data.integer_vars:
            raise ValueError(
                "SciPy backend does not support integer decision variables"
            )

        x0 = np.asarray(problem_data.x0, dtype=float)
        setup_time = problem_data.setup_time
        solve_time = 0.0

        obj_func = problem_data.objective_fn
        gradient = grad(obj_func)
        uses_hessian = method in self.HESSIAN_METHODS
        hess_func = hessian(obj_func) if uses_hessian else None

        options = dict(solver_options)
        explicit_bounds = options.pop("bounds", None)
        bounds = self._resolve_bounds(problem_data, explicit_bounds)
        has_bounds = any(lb is not None or ub is not None for lb, ub in bounds)
        uses_bounds = method in self.BOUND_METHODS and has_bounds

        simple_bound_constraint_indices = set()
        if uses_bounds and problem_data.constraints is not None:
            simple_bound_constraint_indices = {
                i
                for i, constraint in enumerate(problem_data.constraints)
                if is_simple_bound_constraint(constraint, problem_data)
            }

        # Build scipy constraint dicts from ConstraintFn objects
        # For projection constraints (<-), apply p_tol so the constraint becomes
        # p_tol - ||x - proj(x)|| >= 0, i.e., ||x - proj(x)|| <= p_tol
        cons = []
        for i, c in enumerate(problem_data.constraint_fns):
            if i in simple_bound_constraint_indices:
                continue
            if c.op == "<-":
                # Wrap projection constraint to apply tolerance
                wrapped_fun = wrap_projection_constraint(
                    c, problem_data.projection_tolerance
                )
                cons.append(
                    {
                        "type": "ineq",  # p_tol - norm >= 0
                        "fun": wrapped_fun,
                        "jac": jacobian(wrapped_fun),
                    }
                )
            else:
                cons.append(
                    {
                        "type": c.type,
                        "fun": c.fun,
                        "jac": jacobian(c.fun),
                    }
                )

        def dummy_func(_):
            return 0.0

        def dummy_jac(x):
            return np.zeros_like(x)

        uses_gradient = method in self.GRADIENT_METHODS
        uses_constraints = method in self.CONSTRAINED_METHODS

        if cons and not uses_constraints:
            raise ValueError(
                f"Solver '{method}' does not support constraints. "
                f"Use one of: {', '.join(sorted(self.CONSTRAINED_METHODS))}"
            )

        presolve_result = None
        if problem_data.presolve and cons:
            start_time = time.time()
            presolve_kwargs = {
                "method": method,
                "options": dict(options),
                "constraints": cons,
            }
            if uses_bounds:
                presolve_kwargs["bounds"] = bounds
            if uses_gradient:
                presolve_kwargs["jac"] = dummy_jac
            presolve_result = minimize(dummy_func, x0, **presolve_kwargs)
            x0 = presolve_result.x
            solve_time += time.time() - start_time

        start_time = time.time()
        if method == "dogleg":
            options.setdefault("initial_trust_radius", 0.1)
            options.setdefault("max_trust_radius", 1.0)

        minimize_kwargs = {
            "method": method,
            "options": options,
        }
        if uses_bounds:
            minimize_kwargs["bounds"] = bounds
        if uses_gradient:
            minimize_kwargs["jac"] = gradient
        if uses_hessian:
            minimize_kwargs["hess"] = hess_func
        if cons:
            minimize_kwargs["constraints"] = cons
        result = minimize(obj_func, x0, **minimize_kwargs)
        solve_time += time.time() - start_time
        x_sol = result.x

        projection_result = None
        if uses_projection(problem_data):
            # Build projection constraints (treat <- as equality)
            proj_cons = []
            for c in problem_data.constraint_fns:
                con_type = "eq" if c.op == "<-" else c.type
                proj_cons.append(
                    {
                        "type": con_type,
                        "fun": c.fun,
                        "jac": jacobian(c.fun),
                    }
                )

            if proj_cons:
                proj_options = dict(solver_options)
                proj_options.setdefault("maxiter", problem_data.projection_maxiter)

                start_time = time.time()
                proj_kwargs = {
                    "method": method,
                    "options": proj_options,
                    "constraints": proj_cons,
                }
                if uses_bounds:
                    proj_kwargs["bounds"] = bounds
                if uses_gradient:
                    proj_kwargs["jac"] = dummy_jac
                projection_result = minimize(dummy_func, x_sol, **proj_kwargs)
                solve_time += time.time() - start_time

                if projection_result.success:
                    x_sol = projection_result.x
                else:
                    warnings.warn(
                        f"Projection step failed with status {projection_result.status}"
                    )

        solver_status = self._interpret_status(result, projection_result)

        stats = SolverStats(
            solver_name=method,
            solve_time=solve_time,
            setup_time=setup_time,
            num_iters=getattr(result, "nit", None),
        )

        raw_result = {
            "primary": result,
            "presolve": presolve_result,
            "projection": projection_result,
        }

        return SolverResult(
            x=x_sol,
            status=solver_status,
            stats=stats,
            raw_result=raw_result,
        )

    @staticmethod
    def _interpret_status(result, projection_result) -> SolverStatus:
        status_code = getattr(result, "status", None)
        success = bool(getattr(result, "success", False))

        if success:
            return downgrade_for_projection(SolverStatus.OPTIMAL, projection_result)

        if status_code is None:
            return SolverStatus.UNKNOWN

        return SCIPY_STATUS_MAP.get(status_code, SolverStatus.ERROR)

    @staticmethod
    def _resolve_bounds(problem_data: ProblemData, bounds_opt) -> list:
        n_vars = len(problem_data.x0)

        if bounds_opt is not None:
            bounds = list(bounds_opt)
            if len(bounds) != n_vars:
                raise ValueError(
                    f"bounds must have length {n_vars}, got {len(bounds)}"
                )
            return bounds

        simple_bounds = extract_simple_bounds(problem_data)
        bounds = []
        for i in range(n_vars):
            lb, ub = simple_bounds.get(i, (float("-inf"), float("inf")))
            bounds.append(
                (
                    None if lb == float("-inf") else lb,
                    None if ub == float("inf") else ub,
                )
            )
        return bounds
