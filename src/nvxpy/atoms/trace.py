import autograd.numpy as np

from ..expression import Expr, BaseExpr
from ..constants import Curvature as C


class trace(Expr):
    def __init__(self, left, offset=0):
        self.offset = offset
        super().__init__("trace", left)

    def __call__(self, x):
        return np.trace(x, offset=self.offset)
    
    @property
    def curvature(self):
        arg = self.left
        arg_curv = arg.curvature if isinstance(arg, BaseExpr) else C.CONSTANT

        return arg_curv