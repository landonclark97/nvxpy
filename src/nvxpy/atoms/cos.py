from __future__ import annotations

import autograd.numpy as np

from ..expression import Expr, BaseExpr, ExprLike
from ..constants import Curvature as C


class cos(Expr):
    """Elementwise cosine."""

    def __init__(self, left: ExprLike) -> None:
        super().__init__("cos", left)

    def __call__(self, x):
        return np.cos(x)

    @property
    def curvature(self) -> C:
        # Cosine is non-convex in general
        return C.UNKNOWN

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.left, BaseExpr):
            return self.left.shape
        return np.shape(self.left) or (1,)
