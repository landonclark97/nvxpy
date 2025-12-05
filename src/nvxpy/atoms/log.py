from __future__ import annotations

import autograd.numpy as np

from ..expression import Expr, BaseExpr, ExprLike
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

    def __init__(self, left: ExprLike) -> None:
        super().__init__("log", left)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.log(x)

    @property
    def curvature(self) -> C:
        arg = self.left
        arg_cvx = arg.curvature if isinstance(arg, BaseExpr) else C.CONSTANT

        if arg_cvx == C.CONSTANT:
            return C.CONSTANT

        if arg_cvx in (C.AFFINE, C.CONCAVE):
            # log is concave and nondecreasing, so log(concave) is concave
            return C.CONCAVE

        return C.UNKNOWN

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.left, BaseExpr):
            return self.left.shape
        return np.shape(self.left) or (1,)
