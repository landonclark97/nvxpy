import autograd.numpy as np

from ..expression import Expr, BaseExpr
from ..constants import Curvature as C


class abs(Expr):
    def __init__(self, left):
        super().__init__("abs", left)

    def __call__(self, x):
        return np.abs(x)

    @property
    def curvature(self):
        arg = self.left
        arg_cvx = arg.curvature if isinstance(arg, BaseExpr) else C.CONSTANT

        if arg_cvx == C.CONSTANT:
            return C.CONSTANT

        if arg_cvx == C.AFFINE:
            return C.CONVEX
            
        return C.UNKNOWN