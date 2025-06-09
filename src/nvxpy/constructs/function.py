from typing import Callable

from autograd import jacobian
from autograd.extend import defvjp
from scipy.optimize import approx_fprime

from ..expression import Expr


class Function(Expr):
    def __init__(self, func: Callable, jac="numerical"):
        self.op = "func"
        self.func = func
        self.args = None

        if jac == "numerical":
            self.jac = self._numerical_diff
        elif jac == "autograd":
            self.jac = self._autograd_diff
        elif isinstance(jac, Callable):
            self.jac = jac
        else:
            raise ValueError(f"Invalid jacobian: {jac}")

        super().__init__("func", self)


    def __call__(self, *args, jac="numerical"):
        self.args = args
        defvjp(self.func, *self.jac(self.func, *args))

        return self


    def _numerical_diff(self, func, *xs):
        def partial_grad(i):
            def grad_i(g):
                def f_i(xi):
                    x_copy = list(xs)
                    x_copy[i] = xi
                    return func(*x_copy)
                return approx_fprime(xs[i], f_i, epsilon=1e-8) * g
            return grad_i
        return [partial_grad(i) for i in range(len(xs))]


    def _autograd_diff(self, func, *xs):
        return [lambda g, i=i: jacobian(lambda *a: func(*a))( *xs )[i] * g
                for i in range(len(xs))]