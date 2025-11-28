import autograd.numpy as np

from ..expression import Expr, BaseExpr
from ..constants import Curvature as C


class log(Expr):
    """Natural logarithm.

    log(x) is concave on x > 0.

    Curvature rules:
    - log(constant) -> constant
    - log(affine) -> concave
    - log(concave) -> concave (composition rule: concave + nondecreasing)
    - log(convex) -> unknown
    """

    def __init__(self, left):
        super().__init__("log", left)

    def __call__(self, x):
        return np.log(x)

    @property
    def curvature(self):
        arg = self.left
        arg_cvx = arg.curvature if isinstance(arg, BaseExpr) else C.CONSTANT

        if arg_cvx == C.CONSTANT:
            return C.CONSTANT

        if arg_cvx in (C.AFFINE, C.CONCAVE):
            # log is concave and nondecreasing, so log(concave) is concave
            return C.CONCAVE

        return C.UNKNOWN
