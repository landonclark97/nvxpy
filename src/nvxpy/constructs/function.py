import logging
from typing import Callable

import autograd.numpy as np

from autograd import jacobian
from autograd.extend import defvjp
from scipy.optimize import approx_fprime

from ..variable import Variable
from ..expression import BaseExpr, expr_to_str
from ..constants import Curvature as C, DEFAULT_GRADIENT_EPS

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


class Function:
    """Factory for creating FunctionExpr instances from user-defined functions.

    This class wraps a Python callable and provides differentiation support.
    When called with arguments, it returns a FunctionExpr that can be used
    in optimization problems.

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

        # Can safely use same function multiple times:
        expr1 = f(x)
        expr2 = f(y)  # Creates separate FunctionExpr, doesn't overwrite expr1
    """

    def __init__(
        self,
        func: Callable,
        jac: str | Callable = "numerical",
        shape: tuple[int, ...] | None = None,
    ) -> None:
        self.func = func
        self._shape = shape
        self._vjp_registered = False

        if jac == "numerical":
            self.jac = self._numerical_diff
        elif jac == "autograd":
            self.jac = self._autograd_diff
        elif callable(jac):
            self.jac = jac
        else:
            raise ValueError(f"Invalid jacobian: {jac}")

    def __call__(self, *args, **kwargs) -> "FunctionExpr":
        """Create a FunctionExpr with the given arguments.

        Args:
            *args: Positional arguments (can include Variables/expressions)
            **kwargs: Keyword arguments (must not include Variables/expressions)

        Returns:
            A new FunctionExpr instance bound to these arguments.
        """
        for key, arg in kwargs.items():
            if isinstance(arg, BaseExpr):
                raise TypeError(
                    f"Decision variables cannot be passed as keyword arguments (got '{key}')"
                )

        # Register VJP once per function (not per call)
        if not self._vjp_registered:
            defvjp(self.func, *self.jac(self.func, *args, **kwargs))
            self._vjp_registered = True

        return FunctionExpr(self, args, kwargs)

    def _numerical_diff(self, func, *xs, **kwargs):
        """Compute jacobian using finite differences."""

        def partial_grad(i):
            def grad_i(g):
                def f_i(xi):
                    x_copy = list(xs)
                    x_copy[i] = xi
                    return func(*x_copy, **kwargs)

                return approx_fprime(xs[i], f_i, epsilon=DEFAULT_GRADIENT_EPS) * g

            return grad_i

        return [partial_grad(i) for i in range(len(xs))]

    def _autograd_diff(self, func, *xs, **kwargs):
        """Compute jacobian using autograd."""
        return [
            lambda g, i=i: jacobian(lambda *a: func(*a, **kwargs))(*xs)[i] * g
            for i in range(len(xs))
        ]


class FunctionExpr(BaseExpr):
    """Expression node representing a function call with specific arguments.

    This is created by calling a Function instance with arguments. Each call
    creates a new FunctionExpr, allowing the same Function to be used multiple
    times in a problem without interference.
    """

    def __init__(
        self,
        func_wrapper: Function,
        args: tuple,
        kwargs: dict,
    ) -> None:
        self.op = "func"
        self._func_wrapper = func_wrapper
        self.args = args
        self.kwargs = kwargs

    @property
    def func(self) -> Callable:
        """The underlying Python function."""
        return self._func_wrapper.func

    @property
    def jac(self) -> Callable:
        """The jacobian computation method."""
        return self._func_wrapper.jac

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
        """Evaluate the function with current argument values."""
        args_list = []
        for arg in self.args:
            if isinstance(arg, (BaseExpr, Variable)):
                args_list.append(arg.value)
            else:
                args_list.append(arg)
        return self.func(*args_list, **self.kwargs)

    @property
    def curvature(self):
        """User-defined functions have unknown curvature."""
        return C.UNKNOWN

    @property
    def shape(self) -> tuple[int, ...]:
        """Infer the output shape of the function."""
        if self._func_wrapper._shape is not None:
            return self._func_wrapper._shape
        # Try to infer from evaluation
        try:
            val = self.value
            if hasattr(val, "shape"):
                return val.shape
        except Exception:
            logger.warning(
                "Function shape unknown - either instantiate function "
                "or provide shape kwarg to Function()"
            )
        # Fall back to first argument shape
        if self.args:
            first_arg = self.args[0]
            if isinstance(first_arg, BaseExpr):
                return first_arg.shape
            return np.shape(first_arg) or (1,)
        return (1,)
