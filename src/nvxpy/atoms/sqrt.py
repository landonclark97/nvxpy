import autograd.numpy as np

from ..expression import Expr, BaseExpr
from ..constants import Curvature as C


class sqrt(Expr):
    """Square root.

    sqrt(x) is concave on x >= 0.

    Curvature rules:
    - sqrt(constant) -> constant
    - sqrt(affine) -> concave
    - sqrt(concave) -> concave (composition rule: concave + nondecreasing)
    - sqrt(convex) -> unknown
    """

    def __init__(self, left):
        super().__init__("sqrt", left)

    def __call__(self, x):
        return np.sqrt(x)

    @property
    def curvature(self):
        arg = self.left
        arg_cvx = arg.curvature if isinstance(arg, BaseExpr) else C.CONSTANT

        if arg_cvx == C.CONSTANT:
            return C.CONSTANT

        if arg_cvx in (C.AFFINE, C.CONCAVE):
            # sqrt is concave and nondecreasing, so sqrt(concave) is concave
            return C.CONCAVE

        return C.UNKNOWN
