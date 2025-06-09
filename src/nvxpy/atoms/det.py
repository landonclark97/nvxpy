import autograd.numpy as np

from ..expression import Expr


class det(Expr):
    def __init__(self, left):
        super().__init__("det", left)

    def __call__(self, x):
        return np.linalg.det(x)
