import autograd.numpy as np

from ..expression import Expr, BaseExpr
from ..constants import Curvature as C


class norm(Expr):
    def __init__(self, left, ord=2, axis=None):
        self.ord = ord
        self.axis = axis
        super().__init__("norm", left)

    def __call__(self, x):
        return np.linalg.norm(x, ord=self.ord, axis=self.axis)

    @property
    def curvature(self):
        arg = self.left
        arg_cvx = (
            arg.curvature
            if isinstance(arg, BaseExpr)
            else C.CONSTANT
        )

        if arg_cvx == C.CONSTANT:
            arg_cvx = C.AFFINE

        o = self.ord
        is_convex_ord = False

        if o is None:
            is_convex_ord = True

        elif isinstance(o, (int, float)):
            if o >= 1 or np.isposinf(o):
                is_convex_ord = True

        elif isinstance(o, str):
            if o in ("fro", "nuc"):
                is_convex_ord = True

        if is_convex_ord and arg_cvx in (C.AFFINE, C.CONVEX):
            return C.CONVEX

        return C.UNKNOWN