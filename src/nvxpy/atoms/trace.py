import autograd.numpy as np

from ..expression import Expr


class trace(Expr):
    def __init__(self, left, offset=0):
        self.offset = offset
        super().__init__("trace", left)

    def __call__(self, x):
        return np.trace(x, offset=self.offset)
