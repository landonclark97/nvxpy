import autograd.numpy as np

from ..expression import Expr



class sum(Expr):

    def __init__(self, left, axis=None):
        self.axis = axis
        super().__init__("sum", left)
    

    def __call__(self, x, axis=None):
        return np.sum(x, axis=axis)