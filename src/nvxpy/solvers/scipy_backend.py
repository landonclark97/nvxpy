from __future__ import annotations

import time
import warnings
from typing import Dict, List

import autograd.numpy as np  # type: ignore
from autograd import grad, jacobian  # type: ignore
from scipy.optimize import minimize  # type: ignore

from ..parser import eval_expression
from ..compiler import compile_to_function
from .base import (
    ConstraintData,
    ProblemData,
    SolverResult,
    SolverStats,
    SolverStatus,
)


class ScipyBackend:
    SUPPORTED_METHODS = {
        "SLSQP",
        "COBYLA",
        "Nelder-Mead",
        "BFGS",
        "L-BFGS-B",
        "TNC",
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
            raise ValueError("SciPy backend does not support integer decision variables")

        x0 = np.asarray(problem_data.x0, dtype=float)
        setup_time = problem_data.setup_time
        solve_time = 0.0

        obj_func = self._build_objective(problem_data)
        gradient = grad(obj_func)

        compile_start = time.time()
        constraint_data = self._build_constraint_data(
            problem_data, for_projection=False
        )
        setup_time += time.time() - compile_start
        cons = [self._to_scipy_constraint(c) for c in constraint_data]

        def dummy_func(_):
            return 0.0

        def dummy_jac(x):
            return np.zeros_like(x)

        presolve_result = None
        if problem_data.presolve and cons:
            start_time = time.time()
            presolve_result = minimize(
                dummy_func,
                x0,
                jac=dummy_jac,
                constraints=cons,
                method=method,
                options=solver_options,
            )
            x0 = presolve_result.x
            solve_time += time.time() - start_time

        start_time = time.time()
        result = minimize(
            obj_func,
            x0,
            jac=gradient,
            constraints=cons,
            method=method,
            options=solver_options,
        )
        solve_time += time.time() - start_time
        x_sol = result.x

        projection_result = None
        if self._uses_projection(problem_data):
            compile_start = time.time()
            projection_data = self._build_constraint_data(
                problem_data, for_projection=True
            )
            setup_time += time.time() - compile_start
            proj_cons = [self._to_scipy_constraint(c) for c in projection_data]

            if proj_cons:
                proj_options = dict(solver_options)
                proj_options.setdefault("maxiter", problem_data.projection_maxiter)

                start_time = time.time()
                projection_result = minimize(
                    dummy_func,
                    x_sol,
                    jac=dummy_jac,
                    constraints=proj_cons,
                    method=method,
                    options=proj_options,
                )
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
    def _build_objective(problem_data: ProblemData):
        if problem_data.compile:
            # Use compiled evaluation for better performance
            compiled_obj = compile_to_function(problem_data.objective_expr)

            def obj(x):
                var_dict = problem_data.unpack(x)
                return compiled_obj(var_dict)
        else:
            def obj(x):
                var_dict = problem_data.unpack(x)
                return eval_expression(problem_data.objective_expr, var_dict)

        return obj

    @staticmethod
    def _build_constraint_data(
        problem_data: ProblemData, for_projection: bool
    ) -> List[ConstraintData]:
        constraints: List[ConstraintData] = []
        p_tol = problem_data.projection_tolerance
        use_compile = problem_data.compile

        for constraint in problem_data.constraints:

            def make_con_fun(c, compile_exprs=use_compile):
                if compile_exprs:
                    # Pre-compile constraint expressions
                    compiled_left = compile_to_function(c.left)
                    compiled_right = compile_to_function(c.right)

                    def con_fun(x):
                        var_dict = problem_data.unpack(x)
                        lval = compiled_left(var_dict)
                        rval = compiled_right(var_dict)
                        if c.op == "<-":
                            if for_projection:
                                res = lval - rval
                            else:
                                res = p_tol - np.linalg.norm(
                                    np.atleast_2d(lval - rval), ord="fro"
                                )
                        else:
                            res = (
                                lval - rval
                                if c.op in [">=", "==", ">>"]
                                else rval - lval
                            )
                        if c.op in [">>", "<<"]:
                            res = np.real(np.ravel(np.linalg.eig(res)[0]))
                        return np.ravel(res)
                else:
                    def con_fun(x):
                        var_dict = problem_data.unpack(x)
                        lval = eval_expression(c.left, var_dict)
                        rval = eval_expression(c.right, var_dict)
                        if c.op == "<-":
                            if for_projection:
                                res = lval - rval
                            else:
                                res = p_tol - np.linalg.norm(
                                    np.atleast_2d(lval - rval), ord="fro"
                                )
                        else:
                            res = (
                                lval - rval
                                if c.op in [">=", "==", ">>"]
                                else rval - lval
                            )
                        if c.op in [">>", "<<"]:
                            res = np.real(np.ravel(np.linalg.eig(res)[0]))
                        return np.ravel(res)

                return con_fun

            con_fun = make_con_fun(constraint)
            con_jac = jacobian(con_fun)
            constraint_type = (
                "eq"
                if (constraint.op == "==" or (for_projection and constraint.op == "<-"))
                else "ineq"
            )
            constraints.append(
                ConstraintData(
                    type=constraint_type,
                    fun=con_fun,
                    jac=con_jac,
                    op=constraint.op,
                )
            )

        return constraints

    @staticmethod
    def _to_scipy_constraint(constraint: ConstraintData) -> Dict[str, object]:
        return {"type": constraint.type, "fun": constraint.fun, "jac": constraint.jac}

    @staticmethod
    def _uses_projection(problem_data: ProblemData) -> bool:
        return any(getattr(c, "op", None) == "<-" for c in problem_data.constraints)

    @staticmethod
    def _interpret_status(result, projection_result) -> SolverStatus:
        status_code = getattr(result, "status", None)
        success = bool(getattr(result, "success", False))

        if success:
            if projection_result is not None and not getattr(
                projection_result, "success", True
            ):
                return SolverStatus.SUBOPTIMAL
            return SolverStatus.OPTIMAL

        status_map = {
            0: SolverStatus.OPTIMAL,
            1: SolverStatus.MAX_ITERATIONS,
            2: SolverStatus.INFEASIBLE,
            3: SolverStatus.UNBOUNDED,
            4: SolverStatus.NUMERICAL_ERROR,
        }

        if status_code is None:
            return SolverStatus.UNKNOWN

        return status_map.get(status_code, SolverStatus.ERROR)


