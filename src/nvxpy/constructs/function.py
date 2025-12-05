import logging
from typing import Callable

import autograd.numpy as np

from autograd import jacobian
from autograd.extend import defvjp
from scipy.optimize import approx_fprime

from ..variable import Variable
from ..expression import BaseExpr, expr_to_str
from ..constants import Curvature as C

logger = logging.getLogger(__name__)


def function(func=None, *, jac="numerical", shape=None):
    """Decorator to create a Function from a Python callable.

    Can be used with or without arguments:

        @nvx.function
        def my_func(x):
            return x**2

        @nvx.function(jac="autograd", shape=(1,))
        def my_func(x):
            return x[0]**2 + np.sin(x[1])

    Args:
        func: The function to wrap (provided automatically when used without parens).
        jac: Differentiation method - "numerical", "autograd", or a callable.
        shape: Optional output shape hint.

    Returns:
        A Function instance wrapping the decorated function.
    """
    def decorator(f):
        return Function(f, jac=jac, shape=shape)

    if func is not None:
        # Called as @nvx.function (no parens)
        return decorator(func)
    # Called as @nvx.function(...) (with parens)
    return decorator


class Function(BaseExpr):
    """Wrapper for custom user-defined functions in optimization problems.

    Allows embedding arbitrary Python functions into nvxpy expressions while
    supporting automatic differentiation for gradient-based optimization.

    Args:
        func: The Python function to wrap.
        jac: Differentiation method. One of:
            - "numerical": Use finite differences (default, works for any function)
            - "autograd": Use autograd automatic differentiation
            - callable: A custom jacobian function
        shape: Optional output shape hint for the function.

    Example:
        def my_func(x):
            return x[0]**2 + np.sin(x[1])

        f = nvx.Function(my_func)
        x = nvx.Variable((2,))
        prob = nvx.Problem(nvx.Minimize(f(x)))
    """

    def __init__(
        self,
        func: Callable,
        jac: str | Callable = "numerical",
        shape: tuple[int, ...] | None = None,
    ) -> None:
        self.op = "func"
        self.func = func
        self.args = []
        self.kwargs = {}
        self._shape = shape

        if jac == "numerical":
            self.jac = self._numerical_diff
        elif jac == "autograd":
            self.jac = self._autograd_diff
        elif isinstance(jac, Callable):
            self.jac = jac
        else:
            raise ValueError(f"Invalid jacobian: {jac}")


    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        for key, arg in kwargs.items():
            if isinstance(arg, BaseExpr):
                raise TypeError(f"Decision variables cannot be passed as keyword arguments (got '{key}')")
        defvjp(self.func, *self.jac(self.func, *args))
        return self


    def _numerical_diff(self, func, *xs):
        def partial_grad(i):
            def grad_i(g):
                def f_i(xi):
                    x_copy = list(xs)
                    x_copy[i] = xi
                    return func(*x_copy, **self.kwargs)
                return approx_fprime(xs[i], f_i, epsilon=1e-8) * g
            return grad_i
        return [partial_grad(i) for i in range(len(xs))]


    def _autograd_diff(self, func, *xs):
        return [lambda g, i=i: jacobian(lambda *a: func(*a, **self.kwargs))( *xs )[i] * g
                for i in range(len(xs))]
    
    def __repr__(self) -> str:
        args_str = []
        for arg in self.args:
            arg_str = expr_to_str(arg)
            args_str.append(arg_str)
        return f"{self.op}({', '.join(args_str)})"

    def __hash__(self) -> int:
        return hash(str(self))

    @property
    def value(self):
        args_list = []
        for arg in self.args:
            if isinstance(arg, (BaseExpr, Variable)):
                args_list.append(arg.value)
            else:
                args_list.append(arg)
        return self.func(*args_list, **self.kwargs)

    @property
    def curvature(self):
        return C.UNKNOWN

    @property
    def shape(self) -> tuple[int, ...]:
        if self._shape is not None:
            return self._shape
        # Try to infer from first argument if available
        logger.warning("Function shape unknown - provide shape kwarg to Function()")
        if self.args:
            first_arg = self.args[0]
            if isinstance(first_arg, BaseExpr):
                return first_arg.shape
            return np.shape(first_arg) or (1,)
        return (1,)