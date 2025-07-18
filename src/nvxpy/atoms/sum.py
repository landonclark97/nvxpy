import autograd.numpy as np

from ..expression import Expr, BaseExpr
from ..constants import Curvature as C


class sum(Expr):
    def __init__(self, left, axis=None):
        self.axis = axis
        super().__init__("sum", left)

    def __call__(self, x, axis=None):
        return np.sum(x, axis=axis)

    @property
    def curvature(self):
        arg = self.left
        arg_curv = arg.curvature if isinstance(arg, BaseExpr) else C.CONSTANT

        return arg_curv