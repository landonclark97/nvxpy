import autograd.numpy as np

from ..expression import Expr


class norm(Expr):
    def __init__(self, left, ord=2, axis=None):
        self.ord = ord
        self.axis = axis
        super().__init__("norm", left)

    def __call__(self, x):
        return np.linalg.norm(x, ord=self.ord, axis=self.axis)
