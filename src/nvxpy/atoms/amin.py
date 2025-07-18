import autograd.numpy as np

from ..expression import Expr, BaseExpr
from ..constants import Curvature as C


class amin(Expr):
    def __init__(self, left):
        super().__init__("amin", left)

    def __call__(self, x):
        return np.amin(x)

    @property
    def curvature(self):
        arg = self.left
        arg_cvx = arg.curvature if isinstance(arg, BaseExpr) else C.CONSTANT

        if arg_cvx == C.CONSTANT:
            return C.CONSTANT

        if arg_cvx in (C.AFFINE, C.CONCAVE):
            return C.CONCAVE
            
        return C.UNKNOWN