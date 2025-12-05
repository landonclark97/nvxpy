from __future__ import annotations

import autograd.numpy as np

from ..expression import Expr, BaseExpr, ExprLike
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

    def __init__(self, left: ExprLike) -> None:
        super().__init__("exp", left)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x)

    @property
    def curvature(self) -> C:
        arg = self.left
        arg_cvx = arg.curvature if isinstance(arg, BaseExpr) else C.CONSTANT

        if arg_cvx == C.CONSTANT:
            return C.CONSTANT

        if arg_cvx in (C.AFFINE, C.CONVEX):
            # exp is convex and nondecreasing, so exp(convex) is convex
            return C.CONVEX

        return C.UNKNOWN

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.left, BaseExpr):
            return self.left.shape
        return np.shape(self.left) or (1,)
