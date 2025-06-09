from typing import Callable

from .expression import Expr
from .variable import Variable


def collect_vars(expr, vars):
    if isinstance(expr, Variable):
        vars.append(expr)
    elif isinstance(expr, Expr):
        collect_vars(expr.left, vars)
        if expr.right is not None:
            collect_vars(expr.right, vars)


def eval_expression(expr, var_dict, use_value=False):
    if isinstance(expr, Variable):
        if use_value:
            return expr.value
        else:
            return var_dict[expr.name]

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

    else:
        return expr
