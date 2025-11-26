from __future__ import annotations

import time
from typing import Dict

import autograd.numpy as np  # type: ignore

from .constants import Solver
from .parser import collect_vars
from .solvers import ProblemData, get_solver_backend


class Minimize:
    def __init__(self, expr):
        self.expr = expr


class Maximize:
    def __init__(self, expr):
        self.expr = -expr


class Problem:
    def __init__(self, objective, constraints=None):
        self.objective = objective
        self.constraints = list(constraints) if constraints is not None else []

        self.vars = []
        self.var_shapes = {}
        self.var_slices = {}
        self.total_size = 0

        all_vars = []
        collect_vars(objective.expr, all_vars)
        for c in self.constraints:
            collect_vars(c.left, all_vars)
            collect_vars(c.right, all_vars)

        variable_constraints = []
        for v in all_vars:
            variable_constraints.extend(v.constraints)
        self.constraints += variable_constraints

        self.vars = []
        for v in all_vars:
            if v.name not in self.var_shapes:
                self.var_shapes[v.name] = v.shape
                self.var_slices[v.name] = (self.total_size, self.total_size + v.size)
                self.total_size += v.size
                self.vars.append(v)

        self.status = None
        self.solver_stats = None

    def solve(self, solver=Solver.SLSQP, solver_options=None, presolve=False, compile=False):
        """
        Solve the optimization problem.

        Args:
            solver: The solver to use (default: SLSQP)
            solver_options: Options to pass to the solver
            presolve: Whether to run a presolve phase
            compile: If True, compile expressions for faster evaluation.
                     This can significantly speed up problems with complex
                     expressions that are evaluated many times.

        Returns:
            SolverResult with solution and status
        """
        solver_options = solver_options or {}

        problem_data, backend_options = self._compile_problem_data(
            solver_options, presolve, compile
        )

        backend = get_solver_backend(solver)
        solver_name = solver.value if isinstance(solver, Solver) else str(solver)

        if problem_data.integer_vars and solver_name != Solver.BNB.value:
            raise ValueError(
                "Integer variables detected; use the BnB solver (nvx.BNB)."
            )

        result = backend.solve(problem_data, solver_name, backend_options)

        self.status = result.status
        self.solver_stats = result.stats

        sol_vars = problem_data.unpack(result.x)

        for v in self.vars:
            v.value = sol_vars[v.name]

        return result

    def _compile_problem_data(
        self, solver_options: Dict[str, object], presolve: bool, compile: bool
    ) -> tuple[ProblemData, Dict[str, object]]:
        options = dict(solver_options)

        p_tol = options.pop("p_tol", 1e-6)
        p_maxiter = options.pop("p_maxiter", 100)

        start_setup_time = time.time()

        x0 = np.ones(self.total_size)
        var_names = [v.name for v in self.vars]
        var_shapes = dict(self.var_shapes)
        var_slices = dict(self.var_slices)
        integer_vars = tuple(v.name for v in self.vars if getattr(v, "is_integer", False))

        for v in self.vars:
            v_start, v_end = self.var_slices[v.name]
            if v.value is None:
                x0[v_start:v_end] = np.zeros(v.size)
            else:
                x0[v_start:v_end] = np.ravel(v.value)

        setup_time = time.time() - start_setup_time

        problem_data = ProblemData(
            x0=x0,
            var_names=var_names,
            var_shapes=var_shapes,
            var_slices=var_slices,
            objective_expr=self.objective.expr,
            constraints=tuple(self.constraints),
            integer_vars=integer_vars,
            projection_tolerance=p_tol,
            projection_maxiter=p_maxiter,
            presolve=presolve,
            compile=compile,
            setup_time=setup_time,
        )

        return problem_data, options
