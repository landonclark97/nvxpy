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
        l = eval_expression(expr.left, var_dict, use_value)
        r = eval_expression(expr.right, var_dict, use_value) if expr.right is not None else None

        if expr.op == "add": return l + r
        elif expr.op == "sub": return l - r
        elif expr.op == "mul": return l * r
        elif expr.op == "div": return l / r
        elif expr.op == "pow": return l**r
        elif expr.op == "neg": return -l
        elif expr.op == "matmul": return l @ r
        elif expr.op == "getitem": return l[r]
        elif expr.op == "transpose": return l.T
        elif isinstance(expr, Callable): return expr(l) if r is None else expr(l, r)
        else: 
            raise NotImplementedError(expr.op)
    
    else:
        return expr
