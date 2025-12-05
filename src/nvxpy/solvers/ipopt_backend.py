"""Direct IPOPT backend using cyipopt."""

from __future__ import annotations

import logging
import time
import warnings
from typing import Dict

import autograd.numpy as np
from autograd import grad, jacobian

from ..parser import eval_expression
from ..compiler import compile_to_function
from ..constants import EPSILON, DEFAULT_SOLVER_TOL

logger = logging.getLogger(__name__)
from .base import (
    build_objective,
    uses_projection,
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

        x0 = np.asarray(problem_data.x0, dtype=float)
        n = len(x0)
        setup_time = problem_data.setup_time

        # Build objective and gradient
        compile_start = time.time()
        obj_func = build_objective(problem_data)
        obj_grad = grad(obj_func)

        # Build constraints
        eq_constraints, ineq_constraints = self._build_constraints(
            problem_data, for_projection=False
        )
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
        nlp.add_option("tol", solver_options.get("tol", DEFAULT_SOLVER_TOL))

        # Additional user options
        for key, value in solver_options.items():
            if key not in {"lb", "ub", "print_level", "maxiter", "tol"}:
                try:
                    nlp.add_option(key, value)
                except Exception as e:
                    logger.warning(f"Invalid option '{key}={value}': {e}")

        # Solve
        start_time = time.time()
        x_sol, info = nlp.solve(x0)
        solve_time = time.time() - start_time

        # Projection phase for <- constraints
        projection_info = None
        if uses_projection(problem_data):
            compile_start = time.time()
            proj_eq, proj_ineq = self._build_constraints(
                problem_data, for_projection=True
            )
            setup_time += time.time() - compile_start

            n_proj_eq = sum(self._constraint_dim(c, x_sol) for c in proj_eq)
            n_proj_ineq = sum(self._constraint_dim(c, x_sol) for c in proj_ineq)
            m_proj = n_proj_eq + n_proj_ineq

            if m_proj > 0:
                proj_con_func, proj_con_jac = self._build_combined_constraints(
                    proj_eq, proj_ineq, n, m_proj
                )

                proj_cl = np.zeros(m_proj)
                proj_cu = np.concatenate([
                    np.zeros(n_proj_eq),
                    np.full(n_proj_ineq, np.inf),
                ]) if m_proj > 0 else np.array([])

                # Dummy objective for projection (just find feasible point)
                def dummy_obj(_):
                    return 0.0

                def dummy_grad(x):
                    return np.zeros_like(x)

                proj_nlp = cyipopt.Problem(
                    n=n,
                    m=m_proj,
                    problem_obj=_IpoptProblem(dummy_obj, dummy_grad, proj_con_func, proj_con_jac),
                    lb=lb,
                    ub=ub,
                    cl=proj_cl,
                    cu=proj_cu,
                )

                proj_nlp.add_option("print_level", 0)
                proj_nlp.add_option("max_iter", problem_data.projection_maxiter)
                proj_nlp.add_option("tol", solver_options.get("tol", DEFAULT_SOLVER_TOL))

                start_time = time.time()
                proj_x_sol, projection_info = proj_nlp.solve(x_sol)
                solve_time += time.time() - start_time

                if projection_info.get("status", -100) in [0, 1, 6]:
                    x_sol = proj_x_sol
                else:
                    warnings.warn(
                        f"Projection step failed with status {projection_info.get('status')}"
                    )

        # Interpret status
        status = self._interpret_status(info, projection_info)

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
            raw_result={"primary": info, "projection": projection_info},
        )

    @staticmethod
    def _build_constraints(problem_data: ProblemData, for_projection: bool = False):
        """Build separate lists of equality and inequality constraint functions.

        Handles all constraint types:
        - >= : Greater than or equal (inequality)
        - <= : Less than or equal (inequality)
        - == : Equality
        - >> : Positive semidefinite - eigenvalues must be >= 0
        - << : Negative semidefinite - eigenvalues must be <= 0
        - <- : Projection constraint (soft inequality during solve, equality during projection)
        """
        eq_constraints = []
        ineq_constraints = []
        use_compile = problem_data.compile
        p_tol = problem_data.projection_tolerance

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
                        if c.op == "<-":
                            if for_projection:
                                res = lval - rval
                            else:
                                # Use epsilon smoothing to avoid gradient singularity at zero
                                diff = np.atleast_2d(lval - rval)
                                res = p_tol - np.sqrt(np.sum(diff * diff) + EPSILON)
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
                                # Use epsilon smoothing to avoid gradient singularity at zero
                                diff = np.atleast_2d(lval - rval)
                                res = p_tol - np.sqrt(np.sum(diff * diff) + EPSILON)
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

            if constraint.op == "==" or (for_projection and constraint.op == "<-"):
                eq_constraints.append(con_fun)
            else:  # >=, <=, >>, <<, or <- (main solve)
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
    def _interpret_status(info: dict, projection_info: dict = None) -> SolverStatus:
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

        # Map status code to solver status
        if status == 0:  # Solve_Succeeded
            base_status = SolverStatus.OPTIMAL
        elif status == 1:  # Solved_To_Acceptable_Level
            base_status = SolverStatus.SUBOPTIMAL
        elif status == 2:  # Infeasible_Problem_Detected
            return SolverStatus.INFEASIBLE
        elif status == 3:  # Search_Direction_Becomes_Too_Small
            return SolverStatus.NUMERICAL_ERROR
        elif status == 4:  # Diverging_Iterates
            return SolverStatus.UNBOUNDED
        elif status == 5:  # User_Requested_Stop
            return SolverStatus.ERROR
        elif status == 6:  # Feasible_Point_Found (for square problems)
            base_status = SolverStatus.OPTIMAL
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

        # If projection was attempted and failed, downgrade to suboptimal
        if projection_info is not None:
            proj_status = projection_info.get("status", -100)
            if proj_status not in [0, 1, 6]:
                return SolverStatus.SUBOPTIMAL

        return base_status


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
