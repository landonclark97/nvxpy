import autograd.numpy as np

from ..expression import Expr, BaseExpr
from ..constants import Curvature as C


class exp(Expr):
    """Exponential function.

    exp(x) is convex.

    Curvature rules:
    - exp(constant) -> constant
    - exp(affine) -> convex
    - exp(convex) -> convex (composition rule: convex + nondecreasing)
    - exp(concave) -> unknown
    """

    def __init__(self, left):
        super().__init__("exp", left)

    def __call__(self, x):
        return np.exp(x)

    @property
    def curvature(self):
        arg = self.left
        arg_cvx = arg.curvature if isinstance(arg, BaseExpr) else C.CONSTANT

        if arg_cvx == C.CONSTANT:
            return C.CONSTANT

        if arg_cvx in (C.AFFINE, C.CONVEX):
            # exp is convex and nondecreasing, so exp(convex) is convex
            return C.CONVEX

        return C.UNKNOWN
