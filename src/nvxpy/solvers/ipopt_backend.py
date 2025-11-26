"""Direct IPOPT backend using cyipopt."""

from __future__ import annotations

import time
from typing import Dict

import autograd.numpy as np
from autograd import grad, jacobian

from ..parser import eval_expression
from ..compiler import compile_to_function
from .base import (
    ProblemData,
    SolverResult,
    SolverStats,
    SolverStatus,
)


class IpoptBackend:
    """Backend for IPOPT solver using cyipopt directly."""

    def solve(
        self,
        problem_data: ProblemData,
        solver: str,
        solver_options: Dict[str, object],
    ) -> SolverResult:
        try:
            import cyipopt
        except ImportError as exc:
            raise ImportError(
                "IPOPT backend requires 'cyipopt' to be installed. "
                "Install with: pip install cyipopt"
            ) from exc

        if problem_data.integer_vars:
            raise ValueError(
                "IPOPT does not support integer variables. "
                "Use nvx.BNB for mixed-integer problems."
            )

        if any(
            getattr(constraint, "op", None) in {">>", "<<", "<-"}
            for constraint in problem_data.constraints
        ):
            raise NotImplementedError(
                "IPOPT backend does not support PSD or projection constraints"
            )

        x0 = np.asarray(problem_data.x0, dtype=float)
        n = len(x0)
        setup_time = problem_data.setup_time

        # Build objective and gradient
        compile_start = time.time()
        obj_func = self._build_objective(problem_data)
        obj_grad = grad(obj_func)

        # Build constraints
        eq_constraints, ineq_constraints = self._build_constraints(problem_data)
        setup_time += time.time() - compile_start

        # Count constraint dimensions
        n_eq = sum(self._constraint_dim(c, x0) for c in eq_constraints)
        n_ineq = sum(self._constraint_dim(c, x0) for c in ineq_constraints)
        m = n_eq + n_ineq

        # Build combined constraint function and jacobian
        if m > 0:
            con_func, con_jac = self._build_combined_constraints(
                eq_constraints, ineq_constraints, n, m
            )
        else:
            def con_func(x):
                return np.array([])

            def con_jac(x):
                return np.zeros((0, n))

        # Constraint bounds: equality constraints have cl=cu=0, inequality have cl=0, cu=inf
        cl = np.zeros(m)
        cu = np.concatenate([
            np.zeros(n_eq),  # equality: g(x) = 0
            np.full(n_ineq, np.inf),  # inequality: g(x) >= 0
        ]) if m > 0 else np.array([])

        # Variable bounds (default: unbounded)
        lb = solver_options.get("lb", np.full(n, -1e20))
        ub = solver_options.get("ub", np.full(n, 1e20))
        if isinstance(lb, (int, float)):
            lb = np.full(n, lb)
        if isinstance(ub, (int, float)):
            ub = np.full(n, ub)

        # Create the problem
        nlp = cyipopt.Problem(
            n=n,
            m=m,
            problem_obj=_IpoptProblem(obj_func, obj_grad, con_func, con_jac),
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu,
        )

        # Set IPOPT options
        nlp.add_option("print_level", solver_options.get("print_level", 0))
        nlp.add_option("max_iter", solver_options.get("maxiter", 1000))
        nlp.add_option("tol", solver_options.get("tol", 1e-8))

        # Additional user options
        for key, value in solver_options.items():
            if key not in {"lb", "ub", "print_level", "maxiter", "tol"}:
                try:
                    nlp.add_option(key, value)
                except Exception:
                    pass  # Ignore invalid options

        # Solve
        start_time = time.time()
        x_sol, info = nlp.solve(x0)
        solve_time = time.time() - start_time

        # Interpret status
        status = self._interpret_status(info)

        stats = SolverStats(
            solver_name="IPOPT",
            solve_time=solve_time,
            setup_time=setup_time,
            num_iters=None,  # cyipopt doesn't expose iteration count easily
        )

        return SolverResult(
            x=x_sol,
            status=status,
            stats=stats,
            raw_result=info,
        )

    @staticmethod
    def _build_objective(problem_data: ProblemData):
        """Build objective function."""
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

    @staticmethod
    def _build_constraints(problem_data: ProblemData):
        """Build separate lists of equality and inequality constraint functions."""
        eq_constraints = []
        ineq_constraints = []
        use_compile = problem_data.compile

        for constraint in problem_data.constraints:
            if constraint.op == "in":
                # Discrete constraints not handled by IPOPT
                continue

            def make_con_fun(c, compile_exprs=use_compile):
                if compile_exprs:
                    compiled_left = compile_to_function(c.left)
                    compiled_right = compile_to_function(c.right)

                    def con_fun(x):
                        var_dict = problem_data.unpack(x)
                        lval = compiled_left(var_dict)
                        rval = compiled_right(var_dict)
                        res = lval - rval if c.op in [">=", "=="] else rval - lval
                        return np.ravel(res)
                else:
                    def con_fun(x):
                        var_dict = problem_data.unpack(x)
                        lval = eval_expression(c.left, var_dict)
                        rval = eval_expression(c.right, var_dict)
                        res = lval - rval if c.op in [">=", "=="] else rval - lval
                        return np.ravel(res)

                return con_fun

            con_fun = make_con_fun(constraint)

            if constraint.op == "==":
                eq_constraints.append(con_fun)
            else:  # >= or <=
                ineq_constraints.append(con_fun)

        return eq_constraints, ineq_constraints

    @staticmethod
    def _constraint_dim(con_func, x0):
        """Get the dimension of a constraint function."""
        try:
            return len(con_func(x0))
        except Exception:
            return 1

    @staticmethod
    def _build_combined_constraints(eq_constraints, ineq_constraints, n, m):
        """Build combined constraint function and jacobian."""
        all_constraints = eq_constraints + ineq_constraints

        def combined_con(x):
            results = []
            for con in all_constraints:
                results.append(np.atleast_1d(con(x)))
            return np.concatenate(results) if results else np.array([])

        combined_jac = jacobian(combined_con)

        return combined_con, combined_jac

    @staticmethod
    def _interpret_status(info: dict) -> SolverStatus:
        """Interpret IPOPT return status.

        IPOPT ApplicationReturnStatus codes:
            Success (>= 0):
                0: Solve_Succeeded
                1: Solved_To_Acceptable_Level
                2: Infeasible_Problem_Detected
                3: Search_Direction_Becomes_Too_Small
                4: Diverging_Iterates
                5: User_Requested_Stop
                6: Feasible_Point_Found

            Error (< 0):
                -1: Maximum_Iterations_Exceeded
                -2: Restoration_Failed
                -3: Error_In_Step_Computation
                -4: Maximum_CpuTime_Exceeded
                -5: Maximum_WallTime_Exceeded
                -10: Not_Enough_Degrees_Of_Freedom
                -11: Invalid_Problem_Definition
                -12: Invalid_Option
                -13: Invalid_Number_Detected
        """
        status = info.get("status", -100)

        if status == 0:  # Solve_Succeeded
            return SolverStatus.OPTIMAL
        elif status == 1:  # Solved_To_Acceptable_Level
            return SolverStatus.SUBOPTIMAL
        elif status == 2:  # Infeasible_Problem_Detected
            return SolverStatus.INFEASIBLE
        elif status == 3:  # Search_Direction_Becomes_Too_Small
            return SolverStatus.NUMERICAL_ERROR
        elif status == 4:  # Diverging_Iterates
            return SolverStatus.UNBOUNDED
        elif status == 5:  # User_Requested_Stop
            return SolverStatus.ERROR
        elif status == 6:  # Feasible_Point_Found (for square problems)
            return SolverStatus.OPTIMAL
        elif status == -1:  # Maximum_Iterations_Exceeded
            return SolverStatus.MAX_ITERATIONS
        elif status == -2:  # Restoration_Failed
            return SolverStatus.NUMERICAL_ERROR
        elif status == -3:  # Error_In_Step_Computation
            return SolverStatus.NUMERICAL_ERROR
        elif status == -4:  # Maximum_CpuTime_Exceeded
            return SolverStatus.MAX_ITERATIONS
        elif status == -5:  # Maximum_WallTime_Exceeded
            return SolverStatus.MAX_ITERATIONS
        elif status == -10:  # Not_Enough_Degrees_Of_Freedom
            return SolverStatus.ERROR
        elif status == -11:  # Invalid_Problem_Definition
            return SolverStatus.ERROR
        elif status == -12:  # Invalid_Option
            return SolverStatus.ERROR
        elif status == -13:  # Invalid_Number_Detected
            return SolverStatus.NUMERICAL_ERROR
        else:
            return SolverStatus.UNKNOWN


class _IpoptProblem:
    """Wrapper class that provides the interface expected by cyipopt."""

    def __init__(self, objective, gradient, constraints, jacobian):
        self._objective = objective
        self._gradient = gradient
        self._constraints = constraints
        self._jacobian = jacobian

    def objective(self, x):
        return self._objective(x)

    def gradient(self, x):
        return self._gradient(x)

    def constraints(self, x):
        return self._constraints(x)

    def jacobian(self, x):
        jac = self._jacobian(x)
        return jac.flatten()  # cyipopt expects flattened jacobian
