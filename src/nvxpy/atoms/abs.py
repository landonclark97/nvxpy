from __future__ import annotations

import autograd.numpy as np

from ..expression import Expr, BaseExpr, ExprLike
from ..constants import Curvature as C


class abs(Expr):
    """Element-wise absolute value.

    abs(x) is convex for affine x.

    Curvature rules:
    - abs(constant) -> constant
    - abs(affine) -> convex
    - abs(convex) or abs(concave) -> unknown
    """

    def __init__(self, left: ExprLike) -> None:
        super().__init__("abs", left)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.abs(x)

    @property
    def curvature(self) -> C:
        arg = self.left
        arg_cvx = arg.curvature if isinstance(arg, BaseExpr) else C.CONSTANT

        if arg_cvx == C.CONSTANT:
            return C.CONSTANT

        if arg_cvx == C.AFFINE:
            return C.CONVEX

        return C.UNKNOWN

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.left, BaseExpr):
            return self.left.shape
        return np.shape(self.left) or (1,)