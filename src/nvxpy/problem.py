from typing import Optional
from dataclasses import dataclass
import time

import autograd.numpy as np
from autograd import grad, jacobian
from scipy.optimize import minimize

from .constants import SLSQP
from .parser import collect_vars, eval_expression


@dataclass(frozen=True)
class SolverStats:
    solver_name: str
    solve_time: Optional[float] = None
    setup_time: Optional[float] = None
    num_iters: Optional[int] = None


class Minimize:
    def __init__(self, expr):
        self.expr = expr


class Maximize:
    def __init__(self, expr):
        self.expr = -expr


class Problem:
    def __init__(self, objective, constraints=[]):
        self.objective = objective
        self.constraints = constraints

        self.vars = []
        self.var_shapes = {}
        self.var_slices = {}
        self.total_size = 0

        all_vars = []
        collect_vars(objective.expr, all_vars)
        for c in constraints:
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

    def solve(self, solver=SLSQP, solver_options=None, presolve=False):
        start_setup_time = time.time()

        x0 = np.ones(self.total_size)
        var_names = [v.name for v in self.vars]
        var_shapes = self.var_shapes
        var_slices = self.var_slices

        for v in self.vars:
            v_start, v_end = self.var_slices[v.name]
            if v.value is None:
                x0[v_start:v_end] = np.zeros(v.size)
            else:
                x0[v_start:v_end] = np.ravel(v.value)

        def unpack(x):
            var_dict = {}
            for v in var_names:
                start, end = var_slices[v]
                shape = var_shapes[v]
                val = x[start:end]
                var_dict[v] = val if not shape else val.reshape(shape)
            return var_dict

        def obj_func(x):
            var_dict = unpack(x)
            return eval_expression(self.objective.expr, var_dict)

        jac_func = grad(obj_func)

        cons = []
        for c in self.constraints:

            def make_con_fun(c):
                def con_fun(x):
                    var_dict = unpack(x)
                    lval = eval_expression(c.left, var_dict)
                    rval = eval_expression(c.right, var_dict)
                    res = lval - rval if c.op in [">=", "==", ">>"] else rval - lval
                    if c.op in [">>", "<<"]:
                        return np.real(np.ravel(np.linalg.eig(res)[0]))
                    return np.ravel(res)

                return con_fun, jacobian(con_fun)

            ctype = "eq" if c.op == "==" else "ineq"
            con_fun, con_jac = make_con_fun(c)
            cons.append({"type": ctype, "fun": con_fun, "jac": con_jac})

        setup_time = time.time() - start_setup_time

        solve_time = 0.0
        if presolve:

            def dummy_func(_):
                return 0.0

            def dummy_jac(x):
                return np.zeros_like(x)

            start_time = time.time()
            results_constraints = minimize(
                dummy_func,
                x0,
                jac=dummy_jac,
                constraints=cons,
                method=solver,
                options=solver_options,
            )
            x0 = results_constraints.x
            solve_time += time.time() - start_time

        start_time = time.time()
        result = minimize(
            obj_func,
            x0,
            jac=jac_func,
            constraints=cons,
            method=solver,
            options=solver_options,
        )
        solve_time += time.time() - start_time

        self.status = result.status
        self.solver_stats = SolverStats(
            solver_name=solver,
            solve_time=solve_time,
            setup_time=setup_time,
            num_iters=result.nit,
        )

        sol_vars = unpack(result.x)

        for v in self.vars:
            v.value = sol_vars[v.name]
