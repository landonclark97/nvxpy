from __future__ import annotations

import autograd.numpy as np

from ..expression import Expr, BaseExpr, ExprLike
from ..constants import Curvature as C


class sin(Expr):
    """Elementwise sine."""

    def __init__(self, left: ExprLike) -> None:
        super().__init__("sin", left)

    def __call__(self, x):
        return np.sin(x)

    @property
    def curvature(self) -> C:
        # Sine is non-convex in general
        return C.UNKNOWN

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.left, BaseExpr):
            return self.left.shape
        return np.shape(self.left) or (1,)
