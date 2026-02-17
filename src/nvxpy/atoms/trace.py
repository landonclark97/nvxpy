from __future__ import annotations

import autograd.numpy as np

from ..expression import Expr, BaseExpr, ExprLike
from ..constants import Curvature as C


class trace(Expr):
    """Sum along the diagonal of a matrix.

    trace(X) returns the sum of diagonal elements.
    Preserves curvature (affine operation).
    """

    def __init__(self, left: ExprLike, offset: int = 0) -> None:
        self.offset = offset
        super().__init__("trace", left)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.trace(x, offset=self.offset)

    @property
    def curvature(self) -> C:
        arg = self.left
        arg_curv = arg.curvature if isinstance(arg, BaseExpr) else C.CONSTANT

        return arg_curv

    @property
    def shape(self) -> tuple[int, ...]:
        return (1,)
