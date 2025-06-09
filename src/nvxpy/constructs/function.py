from typing import Callable

from autograd import grad
from autograd.extend import defvjp
from scipy.optimize import approx_fprime

from ..expression import Expr


class Function(Expr):

    def __init__(self, func, left, jac='numerical'):
        self.func = func
        
        if jac == 'numerical':
            self.jac = self._numerical_diff
        elif jac == 'autograd':
            self.jac = grad(self.__call__)
        elif isinstance(jac, Callable):
            self.jac = jac
        else:
            raise ValueError(f"Invalid jacobian: {jac}")
            
        defvjp(self.__call__, self.jac)

        super().__init__("func", left)

    
    def __call__(self, x):
        return self.func(x)
    

    def _numerical_diff(self, ans, x):
        return lambda g: approx_fprime(x, self.__call__, epsilon=1e-8) * g
