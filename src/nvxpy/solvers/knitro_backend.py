"""KNITRO backend using knitropy.

KNITRO is a commercial solver for nonlinear optimization that supports:
- Nonlinear programming (NLP)
- Mixed-integer nonlinear programming (MINLP) - native support
- Quadratic programming (QP/MIQP)
- Linear programming (LP/MILP)
- Multi-start global optimization
- Complementarity constraints (MPEC)

This backend provides an interface to KNITRO through the knitropy Python API.
"""

from __future__ import annotations

import logging
import time
import warnings
from typing import Dict, List

import autograd.numpy as np
from autograd import grad, jacobian

from ..constants import DEFAULT_SOLVER_TOL
from .base import (
    uses_projection,
    downgrade_for_projection,
    wrap_projection_constraint,
    extract_simple_bounds,
    remove_redundant_equality_constraints,
    ConstraintFn,
    ProblemData,
    SolverResult,
    SolverStats,
    SolverStatus,
)

logger = logging.getLogger(__name__)


class KnitroBackend:
    """Backend for KNITRO solver using knitropy.

    KNITRO is a powerful commercial solver that can handle:
    - Continuous nonlinear optimization (NLP)
    - Mixed-integer nonlinear optimization (MINLP) with native support
    - Problems with custom Python objective and constraint functions

    Unlike IPOPT, KNITRO has native MINLP support, so integer variables
    are handled directly without requiring a branch-and-bound wrapper.
    """

    def solve(
        self,
        problem_data: ProblemData,
        solver: str,
        solver_options: Dict[str, object],
    ) -> SolverResult:
        try:
            import knitro
        except ImportError as exc:
            raise ImportError(
                "KNITRO backend requires 'knitro' to be installed. "
                "Install the knitro Python interface and ensure you have a valid KNITRO license. "
                "See: https://www.artelys.com/solvers/knitro/"
            ) from exc

        x0 = np.asarray(problem_data.x0, dtype=float)
        n = len(x0)
        setup_time = problem_data.setup_time

        # Objective and gradient from problem_data (pre-computed in Problem)
        obj_func = problem_data.objective_fn
        obj_grad = (
            problem_data.objective_grad
            if problem_data.objective_grad
            else grad(obj_func)
        )

        # Remove redundant affine equality constraints using QR decomposition
        # This prevents "more equality constraints than variables" issues in KNITRO
        constraint_fns, n_removed = remove_redundant_equality_constraints(
            problem_data.constraint_fns, x0
        )
        if n_removed > 0:
            logger.info(f"Removed {n_removed} redundant affine equality constraint(s)")

        # Separate equality and inequality constraints
        # For projection constraints (<-), wrap with p_tol so constraint becomes
        # p_tol - ||x - proj(x)|| >= 0, i.e., ||x - proj(x)|| <= p_tol
        eq_constraints: List[ConstraintFn] = []
        ineq_constraints: List[ConstraintFn] = []

        for c in constraint_fns:
            if c.type == "eq":
                eq_constraints.append(c)
            elif c.type == "ineq":
                if c.op == "<-":
                    # Wrap projection constraint to apply tolerance
                    wrapped_fun = wrap_projection_constraint(
                        c, problem_data.projection_tolerance
                    )
                    ineq_constraints.append(
                        ConstraintFn(fun=wrapped_fun, type="ineq", op=c.op)
                    )
                else:
                    ineq_constraints.append(c)

        # Count constraint dimensions
        n_eq = sum(self._constraint_dim(c.fun, x0) for c in eq_constraints)
        n_ineq = sum(self._constraint_dim(c.fun, x0) for c in ineq_constraints)
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

        # Create KNITRO context
        kc = knitro.KN_new()

        try:
            # Add variables
            knitro.KN_add_vars(kc, n)

            # Set initial point
            knitro.KN_set_var_primal_init_values(kc, list(range(n)), x0)

            # Variable bounds - extract from constraints if not provided
            lb = solver_options.get("lb", None)
            ub = solver_options.get("ub", None)

            if lb is None or ub is None:
                # Extract bounds from constraints (e.g., var >= 0 from pos=True)
                simple_bounds = extract_simple_bounds(problem_data)
                if lb is None:
                    lb = np.full(n, -knitro.KN_INFINITY)
                    for idx, (lo, _) in simple_bounds.items():
                        if lo > float("-inf"):
                            lb[idx] = lo
                if ub is None:
                    ub = np.full(n, knitro.KN_INFINITY)
                    for idx, (_, hi) in simple_bounds.items():
                        if hi < float("inf"):
                            ub[idx] = hi

            if isinstance(lb, (int, float)):
                lb = np.full(n, lb)
            if isinstance(ub, (int, float)):
                ub = np.full(n, ub)

            knitro.KN_set_var_lobnds(kc, list(range(n)), lb)
            knitro.KN_set_var_upbnds(kc, list(range(n)), ub)

            # Set binary variables (must be done before general integers)
            if problem_data.binary_vars:
                bin_indices = self._get_binary_indices(problem_data)
                bin_types = [knitro.KN_VARTYPE_BINARY] * len(bin_indices)
                knitro.KN_set_var_types(kc, bin_indices, bin_types)
                # Set bounds [0, 1] for binary variables
                knitro.KN_set_var_lobnds(kc, bin_indices, [0.0] * len(bin_indices))
                knitro.KN_set_var_upbnds(kc, bin_indices, [1.0] * len(bin_indices))

            # Set integer variables (excluding binary, which are already set)
            if problem_data.integer_vars:
                # Get indices that are integer but not binary
                int_indices = self._get_integer_only_indices(problem_data)
                if int_indices:
                    int_types = [knitro.KN_VARTYPE_INTEGER] * len(int_indices)
                    knitro.KN_set_var_types(kc, int_indices, int_types)

            # Add constraints
            if m > 0:
                knitro.KN_add_cons(kc, m)

                # Set constraint bounds
                # Equality: cl = cu = 0
                # Inequality (>= 0): cl = 0, cu = inf
                cl = np.zeros(m)
                cu = (
                    np.concatenate(
                        [
                            np.zeros(n_eq),  # equality: g(x) = 0
                            np.full(
                                n_ineq, knitro.KN_INFINITY
                            ),  # inequality: g(x) >= 0
                        ]
                    )
                    if m > 0
                    else np.array([])
                )

                knitro.KN_set_con_lobnds(kc, list(range(m)), cl)
                knitro.KN_set_con_upbnds(kc, list(range(m)), cu)

            # Set up callback for objective and constraints
            cb = knitro.KN_add_eval_callback(
                kc,
                evalObj=True,
                indexCons=list(range(m)) if m > 0 else None,
                funcCallback=self._make_eval_callback(obj_func, con_func, m, n),
            )

            # Set up callback for gradients
            # Build dense jacobian indices (row-major: constraint i, variable j)
            jac_cons, jac_vars = [], []
            for i in range(m):
                for j in range(n):
                    jac_cons.append(i)
                    jac_vars.append(j)

            knitro.KN_set_cb_grad(
                kc,
                cb,
                objGradIndexVars=list(range(n)),
                jacIndexCons=jac_cons if m > 0 else None,
                jacIndexVars=jac_vars if m > 0 else None,
                gradCallback=self._make_grad_callback(obj_grad, con_jac, n, m),
            )

            # Set KNITRO options
            knitro.KN_set_int_param(
                kc, knitro.KN_PARAM_OUTLEV, solver_options.get("outlev", 0)
            )
            knitro.KN_set_int_param(
                kc, knitro.KN_PARAM_MAXIT, solver_options.get("maxiter", 1000)
            )
            knitro.KN_set_double_param(
                kc,
                knitro.KN_PARAM_FEASTOL,
                solver_options.get("feastol", DEFAULT_SOLVER_TOL),
            )
            knitro.KN_set_double_param(
                kc,
                knitro.KN_PARAM_OPTTOL,
                solver_options.get("opttol", DEFAULT_SOLVER_TOL),
            )

            # Algorithm selection (0 = auto, 1 = interior/direct, 2 = interior/CG,
            # 3 = active set, 4 = SQP, 5 = multi-start)
            if "algorithm" in solver_options:
                knitro.KN_set_int_param(
                    kc, knitro.KN_PARAM_ALGORITHM, solver_options["algorithm"]
                )

            # Multi-start options for global optimization
            if solver_options.get("multistart", False):
                knitro.KN_set_int_param(kc, knitro.KN_PARAM_MULTISTART, 1)
                knitro.KN_set_int_param(
                    kc,
                    knitro.KN_PARAM_MS_MAXSOLVES,
                    solver_options.get("ms_maxsolves", 10),
                )

            # MINLP options
            if problem_data.integer_vars:
                # MIP gap tolerances (relative and absolute)
                knitro.KN_set_double_param(
                    kc,
                    knitro.KN_PARAM_MIP_OPTGAPREL,
                    solver_options.get("mip_optgaprel", 1e-4),
                )
                knitro.KN_set_double_param(
                    kc,
                    knitro.KN_PARAM_MIP_OPTGAPABS,
                    solver_options.get("mip_optgapabs", 1e-6),
                )
                knitro.KN_set_int_param(
                    kc,
                    knitro.KN_PARAM_MIP_MAXNODES,
                    solver_options.get("mip_maxnodes", 100000),
                )
                knitro.KN_set_int_param(
                    kc,
                    knitro.KN_PARAM_MIP_NUMTHREADS,
                    solver_options.get("mip_numthreads", 1),
                )
                # Enable multistart for non-convex MINLPs (recommended by KNITRO docs)
                knitro.KN_set_int_param(
                    kc,
                    knitro.KN_PARAM_MIP_MULTISTART,
                    solver_options.get("mip_multistart", 0),
                )
                # Use extensive heuristics to find feasible solutions (2=advanced)
                knitro.KN_set_int_param(
                    kc,
                    knitro.KN_PARAM_MIP_HEUR_STRATEGY,
                    solver_options.get("mip_heuristic_strategy", 1),
                )

            # Additional user options - look up KN_PARAM_* constant dynamically
            for key, value in solver_options.items():
                if key not in {
                    "lb",
                    "ub",
                    "outlev",
                    "maxiter",
                    "feastol",
                    "opttol",
                    "algorithm",
                    "multistart",
                    "ms_maxsolves",
                    "mip_optgaprel",
                    "mip_optgapabs",
                    "mip_maxnodes",
                    "mip_numthreads",
                    "mip_multistart",
                    "mip_heuristic_strategy",
                }:
                    # Convert key to KNITRO constant name (e.g., "mip_debug" -> "KN_PARAM_MIP_DEBUG")
                    param_name = f"KN_PARAM_{key.upper()}"
                    param_const = getattr(knitro, param_name, None)
                    if param_const is None:
                        logger.warning(
                            f"Unknown KNITRO parameter '{key}' (no {param_name})"
                        )
                        continue
                    try:
                        if isinstance(value, int):
                            knitro.KN_set_int_param(kc, param_const, value)
                        elif isinstance(value, float):
                            knitro.KN_set_double_param(kc, param_const, value)
                        elif isinstance(value, str):
                            knitro.KN_set_char_param(kc, param_const, value)
                    except Exception as e:
                        logger.warning(f"Invalid KNITRO option '{key}={value}': {e}")

            # Solve
            start_time = time.time()
            nStatus = knitro.KN_solve(kc)
            solve_time = time.time() - start_time

            # Get solution
            x_sol = np.array(knitro.KN_get_var_primal_values(kc))

            # Projection phase for <- constraints
            projection_result = None
            if uses_projection(problem_data):
                projection_result = self._run_projection_phase(
                    kc,
                    knitro,
                    problem_data,
                    x_sol,
                    n,
                    eq_constraints,
                    ineq_constraints,
                    lb,
                    ub,
                )
                if projection_result is not None:
                    proj_status, proj_x = projection_result
                    if proj_status in [0, -100, -101, -102]:  # Success codes
                        x_sol = proj_x
                    else:
                        warnings.warn(
                            f"Projection step failed with status {proj_status}"
                        )

            # Interpret status
            status = self._interpret_status(nStatus, projection_result)

        finally:
            # Clean up KNITRO context
            knitro.KN_free(kc)

        stats = SolverStats(
            solver_name="KNITRO",
            solve_time=solve_time,
            setup_time=setup_time,
            num_iters=None,  # Could extract from KNITRO if needed
        )

        return SolverResult(
            x=x_sol,
            status=status,
            stats=stats,
            raw_result={"knitro_status": nStatus, "projection": projection_result},
        )

    def _run_projection_phase(
        self,
        kc_original,
        knitro,
        problem_data: ProblemData,
        x_sol: np.ndarray,
        n: int,
        eq_constraints: List[ConstraintFn],
        ineq_constraints: List[ConstraintFn],
        lb: np.ndarray,
        ub: np.ndarray,
    ):
        """Run projection phase to enforce exact projection constraints."""
        # For projection, treat <- as equality (norm == 0)
        proj_eq = list(eq_constraints)
        proj_ineq = []

        for c in problem_data.constraint_fns:
            if c.op == "<-":
                # Use original constraint (returns norm), treat as equality
                proj_eq.append(c)
            elif c.type == "ineq" and c.op != "<-":
                proj_ineq.append(c)

        n_proj_eq = sum(self._constraint_dim(c.fun, x_sol) for c in proj_eq)
        n_proj_ineq = sum(self._constraint_dim(c.fun, x_sol) for c in proj_ineq)
        m_proj = n_proj_eq + n_proj_ineq

        if m_proj == 0:
            return None

        proj_con_func, proj_con_jac = self._build_combined_constraints(
            proj_eq, proj_ineq, n, m_proj
        )

        # Create new KNITRO context for projection
        kc_proj = knitro.KN_new()

        try:
            knitro.KN_add_vars(kc_proj, n)
            knitro.KN_set_var_primal_init_values(kc_proj, list(range(n)), x_sol)
            knitro.KN_set_var_lobnds(kc_proj, list(range(n)), lb)
            knitro.KN_set_var_upbnds(kc_proj, list(range(n)), ub)

            knitro.KN_add_cons(kc_proj, m_proj)

            proj_cl = np.zeros(m_proj)
            proj_cu = (
                np.concatenate(
                    [
                        np.zeros(n_proj_eq),
                        np.full(n_proj_ineq, knitro.KN_INFINITY),
                    ]
                )
                if m_proj > 0
                else np.array([])
            )

            knitro.KN_set_con_lobnds(kc_proj, list(range(m_proj)), proj_cl)
            knitro.KN_set_con_upbnds(kc_proj, list(range(m_proj)), proj_cu)

            # Dummy objective for projection (just find feasible point)
            def dummy_obj(x):
                return 0.0

            def dummy_grad(x):
                return np.zeros(n)

            cb_proj = knitro.KN_add_eval_callback(
                kc_proj,
                evalObj=True,
                indexCons=list(range(m_proj)),
                funcCallback=self._make_eval_callback(
                    dummy_obj, proj_con_func, m_proj, n
                ),
            )

            # Build dense jacobian indices for projection
            proj_jac_cons, proj_jac_vars = [], []
            for i in range(m_proj):
                for j in range(n):
                    proj_jac_cons.append(i)
                    proj_jac_vars.append(j)

            knitro.KN_set_cb_grad(
                kc_proj,
                cb_proj,
                objGradIndexVars=list(range(n)),
                jacIndexCons=proj_jac_cons,
                jacIndexVars=proj_jac_vars,
                gradCallback=self._make_grad_callback(
                    dummy_grad, proj_con_jac, n, m_proj
                ),
            )

            knitro.KN_set_int_param(kc_proj, knitro.KN_PARAM_OUTLEV, 0)
            knitro.KN_set_int_param(
                kc_proj, knitro.KN_PARAM_MAXIT, problem_data.projection_maxiter
            )

            proj_status = knitro.KN_solve(kc_proj)

            proj_x = np.array(knitro.KN_get_var_primal_values(kc_proj))

            return (proj_status, proj_x)

        finally:
            knitro.KN_free(kc_proj)

    @staticmethod
    def _constraint_dim(con_func, x0):
        """Get the dimension of a constraint function."""
        try:
            return len(con_func(x0))
        except Exception:
            return 1

    @staticmethod
    def _build_combined_constraints(
        eq_constraints: List[ConstraintFn],
        ineq_constraints: List[ConstraintFn],
        n: int,
        m: int,  # noqa: ARG004 - kept for API consistency
    ):
        """Build combined constraint function and jacobian.

        Uses pre-computed jacobians from ConstraintFn when available,
        otherwise computes jacobian once at setup time using autograd.
        """
        all_constraints = list(eq_constraints) + list(ineq_constraints)
        all_fns = [c.fun for c in all_constraints]
        all_jacs = [c.jac for c in all_constraints]

        def combined_con(x):
            results = []
            for fn in all_fns:
                results.append(np.atleast_1d(fn(x)))
            return np.concatenate(results) if results else np.array([])

        # Use pre-computed jacobians if all constraints have them
        if all(jac is not None for jac in all_jacs):

            def combined_jac(x):
                jac_rows = []
                for jac_fn in all_jacs:
                    jac_rows.append(np.atleast_2d(jac_fn(x)))
                return np.vstack(jac_rows) if jac_rows else np.zeros((0, n))
        else:
            # Compute jacobian once at setup time (not in callback)
            combined_jac = jacobian(combined_con)

        return combined_con, combined_jac

    @staticmethod
    def _get_integer_indices(problem_data: ProblemData) -> List[int]:
        """Get indices of integer variables (includes binary)."""
        indices = []
        for var_name in problem_data.integer_vars:
            if var_name in problem_data.var_slices:
                start, end = problem_data.var_slices[var_name]
                indices.extend(range(start, end))
        return indices

    @staticmethod
    def _get_binary_indices(problem_data: ProblemData) -> List[int]:
        """Get indices of binary variables."""
        indices = []
        for var_name in problem_data.binary_vars:
            if var_name in problem_data.var_slices:
                start, end = problem_data.var_slices[var_name]
                indices.extend(range(start, end))
        return indices

    @staticmethod
    def _get_integer_only_indices(problem_data: ProblemData) -> List[int]:
        """Get indices of integer variables that are NOT binary."""
        binary_set = set(problem_data.binary_vars)
        indices = []
        for var_name in problem_data.integer_vars:
            if var_name not in binary_set and var_name in problem_data.var_slices:
                start, end = problem_data.var_slices[var_name]
                indices.extend(range(start, end))
        return indices

    @staticmethod
    def _make_eval_callback(obj_func, con_func, m, n):
        """Create evaluation callback for KNITRO."""

        def eval_callback(kc, cb, evalRequest, evalResult, userParams):
            if evalRequest.type != 1:  # KN_RC_EVALFC
                return -1

            # KNITRO may pass more elements than n (internal working space for MINLP)
            # We only use the first n elements
            x = np.array(evalRequest.x[:n], dtype=float)

            try:
                evalResult.obj = float(obj_func(x))

                if m > 0:
                    c_vals = np.array(con_func(x), dtype=float).ravel()
                    evalResult.c[:] = c_vals
            except Exception as e:
                logger.error(f"Error in eval_callback: {e}")
                return -1

            return 0

        return eval_callback

    @staticmethod
    def _make_grad_callback(obj_grad, con_jac, n, m):
        """Create gradient callback for KNITRO."""

        def grad_callback(kc, cb, evalRequest, evalResult, userParams):
            if evalRequest.type != 2:  # KN_RC_EVALGA
                return -1

            # KNITRO may pass more elements than n (internal working space for MINLP)
            # We only use the first n elements
            x = np.array(evalRequest.x[:n], dtype=float)

            try:
                g = np.array(obj_grad(x), dtype=float).ravel()
                evalResult.objGrad[:] = g

                if m > 0:
                    jac = np.array(con_jac(x), dtype=float)
                    jac_flat = jac.ravel(order="C")
                    evalResult.jac[:] = jac_flat
            except Exception as e:
                logger.error(f"Error in grad_callback: {e}")
                return -1

            return 0

        return grad_callback

    @staticmethod
    def _interpret_status(nStatus: int, projection_result=None) -> SolverStatus:
        """Interpret KNITRO return status."""
        # KNITRO status codes:
        # 0: KN_RC_OPTIMAL - Converged to optimality
        # -100: KN_RC_NEAR_OPT - Near optimal
        # -101 to -109: Various feasible point statuses
        # -200 to -209: Infeasible statuses
        # -300 to -309: Unbounded statuses
        # -400 to -419: Various error/limit statuses

        if nStatus == 0:  # KN_RC_OPTIMAL
            base_status = SolverStatus.OPTIMAL
        elif nStatus in [-100, -101, -102]:  # Near optimal / locally optimal
            base_status = SolverStatus.OPTIMAL
        elif nStatus in [-103, -104, -105, -106]:  # Feasible point found
            base_status = SolverStatus.SUBOPTIMAL
        elif nStatus in [-200, -201, -202, -203]:  # Infeasible
            return SolverStatus.INFEASIBLE
        elif nStatus in [-300, -301]:  # Unbounded
            return SolverStatus.UNBOUNDED
        elif nStatus in [-400, -401, -402, -403]:  # Iteration/time limit
            return SolverStatus.MAX_ITERATIONS
        elif nStatus in [-500, -501, -502, -503, -504, -505, -506]:  # Numerical errors
            return SolverStatus.NUMERICAL_ERROR
        elif nStatus < 0:  # Other errors
            return SolverStatus.ERROR
        else:
            return SolverStatus.UNKNOWN

        return downgrade_for_projection(base_status, projection_result)
