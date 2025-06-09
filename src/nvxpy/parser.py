from typing import Callable, Iterable

import autograd.numpy as np

from .expression import Expr
from .variable import Variable
from .constructs.function import Function



def collect_vars(expr, vars):
    if isinstance(expr, Variable):
        vars.append(expr)
    elif isinstance(expr, Function):
        for arg in expr.args:
            collect_vars(arg, vars)
    elif isinstance(expr, Expr):
        collect_vars(expr.left, vars)
        if expr.right is not None:
            collect_vars(expr.right, vars)
    elif isinstance(expr, dict):
        for v in expr.values():
            collect_vars(v, vars)
    elif isinstance(expr, Iterable) and not isinstance(expr, np.ndarray):
        for e in expr:
            collect_vars(e, vars)


def eval_expression(expr, var_dict, use_value=False):
    if isinstance(expr, Variable):
        if use_value:
            return expr.value
        else:
            return var_dict[expr.name]

    elif isinstance(expr, Function):
        args = []
        for arg in expr.args:
            args.append(eval_expression(arg, var_dict, use_value))
        return expr.func(*args, **expr.kwargs)

    elif isinstance(expr, Expr):
        left_eval = eval_expression(expr.left, var_dict, use_value)
        right_eval = (
            eval_expression(expr.right, var_dict, use_value)
            if expr.right is not None
            else None
        )

        if expr.op == "add":
            return left_eval + right_eval
        elif expr.op == "sub":
            return left_eval - right_eval
        elif expr.op == "mul":
            return left_eval * right_eval
        elif expr.op == "div":
            return left_eval / right_eval
        elif expr.op == "pow":
            return left_eval**right_eval
        elif expr.op == "neg":
            return -left_eval
        elif expr.op == "matmul":
            return left_eval @ right_eval
        elif expr.op == "getitem":
            return left_eval[right_eval]
        elif expr.op == "transpose":
            return left_eval.T
        elif isinstance(expr, Callable):
            return expr(left_eval) if right_eval is None else expr(left_eval, right_eval)
        else:
            raise NotImplementedError(expr.op)
        
    elif isinstance(expr, dict):
        return {k: eval_expression(v, var_dict, use_value) for k, v in expr.items()}
    
    elif isinstance(expr, Iterable) and not isinstance(expr, np.ndarray):
        container = type(expr)
        return container([eval_expression(e, var_dict, use_value) for e in expr])

    else:
        return expr
