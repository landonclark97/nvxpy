from __future__ import annotations

import time
from typing import Callable, Dict, List, Tuple

import autograd.numpy as np
from autograd import grad, jacobian

from .constants import Solver, DEFAULT_PROJECTION_TOL
from .parser import collect_vars, eval_expression
from .compiler import compile_to_function
from .constraint import Constraint
from .sets.discrete_set import DiscreteSet, DiscreteRanges
from .solvers import ConstraintFn, ProblemData, get_solver_backend
from .variable import Variable
from .atoms import sum as nvx_sum


class Minimize:
    def __init__(self, expr):
        self.expr = expr


class Maximize:
    def __init__(self, expr):
        self.expr = -expr


class Problem:
    """An optimization problem with objective and constraints."""

    def __init__(
        self, objective: Minimize | Maximize, constraints=None, compile: bool = False
    ):
        if not isinstance(objective, (Minimize, Maximize)):
            raise TypeError(
                f"Objective must be Minimize or Maximize, got {type(objective).__name__}"
            )

        start_setup_time = time.time()

        self.objective = objective
        self.constraints = list(constraints) if constraints is not None else []

        # Reformulate discrete set constraints using indicator variables
        # This replaces x ^ [v0, v1, ...] with:
        #   x == v0*r0 + v1*r1 + ...
        #   sum(r) == 1
        #   r_i binary
        self._indicator_vars: Dict[str, Tuple[Variable, DiscreteSet]] = {}
        self.constraints, new_vars = self._reformulate_discrete_constraints(
            self.constraints
        )

        self.vars = []
        self.var_shapes = {}
        self.var_slices = {}
        self.total_size = 0

        all_vars = list(new_vars)  # Start with indicator variables
        collect_vars(objective.expr, all_vars)
        for c in self.constraints:
            collect_vars(c.left, all_vars)
            collect_vars(c.right, all_vars)

        # Deduplicate variables before collecting their embedded constraints
        self.vars = []
        for v in all_vars:
            if v.name not in self.var_shapes:
                self.var_shapes[v.name] = v.shape
                self.var_slices[v.name] = (self.total_size, self.total_size + v.size)
                self.total_size += v.size
                self.vars.append(v)

        # Now collect embedded constraints from unique variables only
        variable_constraints = []
        for v in self.vars:
            variable_constraints.extend(v.constraints)
        self.constraints += variable_constraints

        self.status = None
        self.solver_stats = None

        # Build callables for objective and constraints (with pre-computed gradients)
        self._compile = compile
        self._objective_fn, self._objective_grad = self._build_objective_fn()
        self._constraint_fns = self._build_constraint_fns()

        self._setup_time = time.time() - start_setup_time

        # Extract integer and binary variable names
        self._integer_vars = tuple(
            v.name for v in self.vars if getattr(v, "is_integer", False)
        )
        self._binary_vars = tuple(
            v.name for v in self.vars if getattr(v, "is_binary", False)
        )

    def _reformulate_discrete_constraints(
        self, constraints: List
    ) -> Tuple[List, List[Variable]]:
        """
        Reformulate DiscreteSet constraints using indicator variables.

        For a scalar constraint x ^ [v0, v1, v2, ...]:
        - Create binary indicator variable r with shape (n,) where n = len(values)
        - Add constraint: x == v0*r[0] + v1*r[1] + ... (convex combination)
        - Add constraint: sum(r) == 1 (exactly one indicator active)
        - The r variable is binary, so BnB will branch on it

        For an n-D constraint x ^ [[p0], [p1], ...] where x has size n:
        - Create binary indicator variable r with shape (num_points,)
        - For each dimension j: x[j] == sum(p_i[j] * r[i] for all points i)
        - Add constraint: sum(r) == 1 (exactly one point active)

        This gives a tighter LP relaxation than bound-based branching.

        Returns:
            Tuple of (new_constraints, new_indicator_variables)
        """
        new_constraints = []
        new_vars = []

        for c in constraints:
            if c.op == "in" and isinstance(c.right, DiscreteSet):
                expr = c.left  # Can be Variable or Expr
                discrete_set = c.right
                values = discrete_set.values
                point_dim = discrete_set.point_dim

                n_values = len(values)

                # Generate indicator name from expression
                if isinstance(expr, Variable):
                    indicator_name = f"_ind_{expr.name}"
                else:
                    indicator_name = f"_ind_{repr(expr)}"

                if point_dim == 1:
                    # Scalar mode: values are scalars
                    r = Variable(shape=(n_values,), name=indicator_name, binary=True)
                    new_vars.append(r)

                    # Build convex combination: expr == sum(v_i * r_i)
                    convex_combination = sum(values[i] * r[i] for i in range(n_values))

                    # Add equality constraint
                    new_constraints.append(Constraint(expr, "==", convex_combination))

                    # Add sum(r) == 1 constraint (SOS1)
                    new_constraints.append(Constraint(nvx_sum(r), "==", 1))
                else:
                    # n-D mode: values are tuples of coordinates
                    r = Variable(shape=(n_values,), name=indicator_name, binary=True)
                    new_vars.append(r)

                    # For each dimension, add convex combination constraint
                    for dim_idx in range(point_dim):
                        # expr[dim_idx] == sum(point[dim_idx] * r[i] for all points)
                        convex_combination = sum(
                            values[i][dim_idx] * r[i] for i in range(n_values)
                        )
                        expr_elem = expr.flatten()[dim_idx]
                        new_constraints.append(
                            Constraint(expr_elem, "==", convex_combination)
                        )

                    # Add sum(r) == 1 constraint (SOS1) - only once for all dimensions
                    new_constraints.append(Constraint(nvx_sum(r), "==", 1))

                # Store mapping for reference
                self._indicator_vars[indicator_name] = (r, discrete_set)

            elif c.op == "in" and isinstance(c.right, DiscreteRanges):
                expr = c.left  # Can be Variable or Expr
                discrete_ranges = c.right
                ranges = discrete_ranges.ranges

                n_ranges = len(ranges)

                # Generate indicator name from expression
                if isinstance(expr, Variable):
                    base_name = expr.name
                    indicator_name = f"_ind_{base_name}"
                    disagg_name = f"_disagg_{base_name}"
                else:
                    base_name = repr(expr)
                    indicator_name = f"_ind_{base_name}"
                    disagg_name = f"_disagg_{base_name}"

                # Create binary indicator variable for ranges
                # Initialize based on expression's current value
                r = Variable(shape=(n_ranges,), name=indicator_name, binary=True)

                # Get current value of expression to determine initial range
                expr_val = None
                if isinstance(expr, Variable) and expr.value is not None:
                    expr_val = float(np.ravel(expr.value)[0])

                # Find which range contains the current value, or use first range
                active_range = 0
                if expr_val is not None:
                    for i, rng in enumerate(ranges):
                        if rng.lb <= expr_val <= rng.ub:
                            active_range = i
                            break

                # Initialize r: active range = 1, others = 0
                r_init = np.zeros(n_ranges)
                r_init[active_range] = 1.0
                r.value = r_init
                new_vars.append(r)

                # Add sum(r) == 1 constraint (SOS1 - exactly one range active)
                new_constraints.append(Constraint(nvx_sum(r), "==", 1))

                # ============================================================
                # CONVEX HULL FORMULATION (Balas, 1985)
                # ============================================================
                # For disjunction: x in [lb_0, ub_0] OR x in [lb_1, ub_1] OR ...
                #
                # Introduce disaggregated variables y_i for each disjunct:
                #   x = sum(y_i)           -- x is the sum of disaggregated vars
                #   lb_i * r_i <= y_i      -- lower bound when r_i active
                #   y_i <= ub_i * r_i      -- upper bound when r_i active
                #   sum(r_i) = 1           -- exactly one disjunct active
                #
                # When r_i = 1: y_i in [lb_i, ub_i], all other y_j = 0, so x = y_i
                # When r_i = 0: y_i = 0
                #
                # This gives the CONVEX HULL of the disjunction, which is the
                # tightest possible LP relaxation.
                # ============================================================

                # Create disaggregated variables y_i for each range
                # Initialize: active range gets the expression value, others = 0
                y = Variable(shape=(n_ranges,), name=disagg_name)
                y_init = np.zeros(n_ranges)
                if (
                    expr_val is not None
                    and ranges[active_range].lb <= expr_val <= ranges[active_range].ub
                ):
                    y_init[active_range] = expr_val
                else:
                    # Use lower bound of active range
                    y_init[active_range] = ranges[active_range].lb
                y.value = y_init
                new_vars.append(y)

                # x == sum(y_i): original variable equals sum of disaggregated
                new_constraints.append(Constraint(expr, "==", nvx_sum(y)))

                # Compute global bounds for entire y vector
                # Each y_i can be in [lb_i, ub_i] when r_i=1, or 0 when r_i=0
                # So global bounds are: lower = min(0, min(lb_i)), upper = max(0, max(ub_i))
                global_lb = min(min(0.0, rng.lb) for rng in ranges)
                global_ub = max(max(0.0, rng.ub) for rng in ranges)
                # Add bounds on the entire y variable (detected by extract_simple_bounds)
                new_constraints.append(Constraint(y, ">=", global_lb))
                new_constraints.append(Constraint(y, "<=", global_ub))

                # For each range i: lb_i * r_i <= y_i <= ub_i * r_i
                for i in range(n_ranges):
                    lb_i = ranges[i].lb
                    ub_i = ranges[i].ub
                    # y_i >= lb_i * r_i
                    new_constraints.append(Constraint(y[i], ">=", lb_i * r[i]))
                    # y_i <= ub_i * r_i
                    new_constraints.append(Constraint(y[i], "<=", ub_i * r[i]))

                # Store mapping for reference
                self._indicator_vars[indicator_name] = (r, discrete_ranges)

            else:
                # Keep all other constraints as-is
                new_constraints.append(c)

        return new_constraints, new_vars

    def _unpack(self, x) -> Dict[str, np.ndarray]:
        """Unpack flat array x into variable dictionary."""
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        var_dict = {}
        for v in self.vars:
            start, end = self.var_slices[v.name]
            segment = x[start:end]
            if v.shape:
                var_dict[v.name] = segment.reshape(v.shape)
            else:
                var_dict[v.name] = segment[0] if segment.size == 1 else segment
        return var_dict

    def _build_objective_fn(
        self,
    ) -> Tuple[Callable[[np.ndarray], float], Callable[[np.ndarray], np.ndarray]]:
        """Build objective function and gradient: flat x -> scalar, flat x -> gradient."""
        if self._compile:
            compiled = compile_to_function(self.objective.expr)

            def obj_fn(x):
                return compiled(self._unpack(x))
        else:
            expr = self.objective.expr

            def obj_fn(x):
                return eval_expression(expr, self._unpack(x))

        # Pre-compute gradient function once
        obj_grad = grad(obj_fn)
        return obj_fn, obj_grad

    def _build_constraint_fns(self) -> List[ConstraintFn]:
        """Build constraint functions: each takes flat x -> residual array."""
        constraint_fns = []

        for c in self.constraints:
            if self._compile:
                left_fn = compile_to_function(c.left)
                right_fn = compile_to_function(c.right)

                def make_fn(lf=left_fn, rf=right_fn, op=c.op):
                    def con_fn(x):
                        var_dict = self._unpack(x)
                        lval = lf(var_dict)
                        rval = rf(var_dict)
                        return self._constraint_residual(lval, rval, op)

                    return con_fn
            else:
                left_expr = c.left
                right_expr = c.right

                def make_fn(le=left_expr, re=right_expr, op=c.op):
                    def con_fn(x):
                        var_dict = self._unpack(x)
                        lval = eval_expression(le, var_dict)
                        rval = eval_expression(re, var_dict)
                        return self._constraint_residual(lval, rval, op)

                    return con_fn

            con_fn = make_fn()
            con_type = "eq" if c.op == "==" else "ineq"
            # Pre-compute jacobian function once per constraint
            con_jac = jacobian(con_fn)
            # Get curvature from constraint (if it has one)
            con_curvature = getattr(c, "curvature", None)
            constraint_fns.append(
                ConstraintFn(
                    fun=con_fn,
                    type=con_type,
                    op=c.op,
                    jac=con_jac,
                    curvature=con_curvature,
                )
            )

        return constraint_fns

    @staticmethod
    def _constraint_residual(lval, rval, op) -> np.ndarray:
        """Compute constraint residual. For ineq: residual >= 0 means satisfied.

        For projection constraints (<-), returns ||x - proj(x)|| (the norm).
        The solver backend applies p_tol to get the final residual.
        """
        if op == "<-":
            # Projection constraint: return ||x - proj(x)||
            # Backend will compute: p_tol - norm (>= 0 when satisfied)
            diff = np.ravel(lval) - np.ravel(rval)
            norm = np.sqrt(np.sum(diff**2))
            return np.array([norm])
        elif op in [">=", "==", ">>"]:
            res = lval - rval
        else:  # <=, <<
            res = rval - lval

        if op in [">>", "<<"]:
            # PSD/NSD: check eigenvalues
            res = np.real(np.ravel(np.linalg.eig(res)[0]))

        return np.ravel(res)

    def _has_discrete_constraints(self) -> bool:
        """Check if any constraint involves discrete sets that weren't reformulated."""
        # Both DiscreteSet and DiscreteRanges constraints are now reformulated
        # to indicator variables, so they show up as integer variables.
        # This method returns True only for "in" constraints on expressions
        # (which can't be reformulated).
        for c in self.constraints:
            if c.op == "in":
                return True
        return False

    def _select_default_solver(
        self, has_integers: bool, has_discrete: bool, has_constraints: bool
    ) -> Solver:
        """Select the default solver based on problem characteristics."""
        if has_integers or has_discrete:
            return Solver.BNB
        if has_constraints:
            return Solver.SLSQP
        return Solver.LBFGSB

    def solve(self, solver=None, solver_options=None, presolve=False, verbose=False):
        """
        Solve the optimization problem.

        Args:
            solver: The solver to use. If None, automatically selects:
                    - BNB for problems with integer variables or discrete constraints
                    - SLSQP for problems with constraints
                    - L-BFGS-B for unconstrained problems
            solver_options: Options to pass to the solver
            presolve: Whether to run a presolve phase
            verbose: Whether to print solver progress. Sets the appropriate
                     verbosity option for the selected backend.

        Returns:
            SolverResult with solution and status
        """
        solver_options = solver_options or {}
        options = dict(solver_options)

        # Set verbose option for the backend (can be overridden by solver_options)
        if verbose and "verbose" not in options and "bb_verbose" not in options:
            options["verbose"] = True

        # Extract problem-level options
        p_tol = options.pop("p_tol", DEFAULT_PROJECTION_TOL)
        p_maxiter = options.pop("p_maxiter", 100)

        if p_tol <= 0:
            raise ValueError(f"p_tol must be positive, got {p_tol}")
        if p_maxiter <= 0:
            raise ValueError(f"p_maxiter must be positive, got {p_maxiter}")

        # Build x0 from current variable values
        x0 = np.zeros(self.total_size)
        for v in self.vars:
            v_start, v_end = self.var_slices[v.name]
            if v.value is None:
                x0[v_start:v_end] = np.zeros(v.size)
            else:
                x0[v_start:v_end] = np.ravel(v.value)

        # Build SOS1 groups from indicator variables
        # Each indicator variable from DiscreteSet/DiscreteRanges reformulation
        # forms an SOS1 group (exactly one indicator must be 1)
        sos1_groups = []
        for indicator_name, (indicator_var, _) in self._indicator_vars.items():
            start, end = self.var_slices[indicator_name]
            group_indices = list(range(start, end))
            sos1_groups.append(group_indices)

        # Build ProblemData with ready-to-use callables
        problem_data = ProblemData(
            x0=x0,
            var_names=[v.name for v in self.vars],
            var_shapes=dict(self.var_shapes),
            var_slices=dict(self.var_slices),
            objective_fn=self._objective_fn,
            objective_grad=self._objective_grad,
            constraint_fns=self._constraint_fns,
            integer_vars=self._integer_vars,
            binary_vars=self._binary_vars,
            projection_tolerance=p_tol,
            projection_maxiter=p_maxiter,
            presolve=presolve,
            setup_time=self._setup_time,
            constraints=self.constraints,
            sos1_groups=sos1_groups if sos1_groups else None,
        )

        # Auto-select solver if not specified
        if solver is None:
            has_integers = bool(self._integer_vars)
            has_discrete = self._has_discrete_constraints()
            has_constraints = bool(self.constraints)
            solver = self._select_default_solver(
                has_integers, has_discrete, has_constraints
            )

        backend = get_solver_backend(solver)
        solver_name = solver.value if isinstance(solver, Solver) else str(solver)

        has_discrete = self._has_discrete_constraints()
        # KNITRO has native MINLP support, BNB is our branch-and-bound solver
        minlp_solvers = {Solver.BNB.value, Solver.KNITRO.value}
        if (self._integer_vars or has_discrete) and solver_name not in minlp_solvers:
            raise ValueError(
                "Integer variables or discrete constraints detected; use the BnB solver (nvx.BNB) or KNITRO (nvx.KNITRO)."
            )

        result = backend.solve(problem_data, solver_name, options)

        self.status = result.status
        self.solver_stats = result.stats

        sol_vars = problem_data.unpack(result.x)

        for v in self.vars:
            v.value = sol_vars[v.name]

        return result
