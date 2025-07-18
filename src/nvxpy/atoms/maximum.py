import autograd.numpy as np

from ..expression import Expr, BaseExpr
from ..constants import Curvature as C


class maximum(Expr):
    def __init__(self, left, right):
        super().__init__("maximum", left, right)

    def __call__(self, x, y):
        return np.maximum(x, y)
    
    @property
    def curvature(self):
        if isinstance(self.left, BaseExpr):
            left = self.left.curvature
        else:
            left = C.CONSTANT

        if isinstance(self.right, BaseExpr):
            right = self.right.curvature
        else:
            right = C.CONSTANT

        if left == C.CONSTANT and right == C.CONSTANT:
            return C.CONSTANT

        if left == C.CONSTANT:
            left = C.AFFINE
        if right == C.CONSTANT:
            right = C.AFFINE

        if left in (C.CONVEX, C.AFFINE) and right in (C.CONVEX, C.AFFINE):
            return C.CONVEX
        return C.UNKNOWN