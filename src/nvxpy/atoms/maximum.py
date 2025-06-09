import autograd.numpy as np

from ..expression import Expr


class maximum(Expr):
    def __init__(self, left, right):
        super().__init__("maximum", left, right)

    def __call__(self, x, y):
        return np.maximum(x, y)
