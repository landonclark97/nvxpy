from __future__ import annotations

import autograd.numpy as np

from ..expression import Expr, BaseExpr, broadcast_shapes, ExprLike
from ..constants import Curvature as C


class minimum(Expr):
    """Element-wise minimum of two arrays.

    minimum(x, y) is concave (pointwise min of concave functions is concave).

    Curvature rules:
    - minimum(constant, constant) -> constant
    - minimum(affine, affine) -> concave
    - minimum(concave, concave) -> concave
    - minimum(convex, _) -> unknown
    """

    def __init__(self, left: ExprLike, right: ExprLike) -> None:
        super().__init__("minimum", left, right)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.minimum(x, y)

    @property
    def curvature(self) -> C:
        if isinstance(self.left, BaseExpr):
            left = self.left.curvature
        else:
            left = C.CONSTANT

        if isinstance(self.right, BaseExpr):
            right = self.right.curvature
        else:
            right = C.CONSTANT

        if left == C.CONSTANT and right == C.CONSTANT:
            return C.CONSTANT

        if left == C.CONSTANT:
            left = C.AFFINE
        if right == C.CONSTANT:
            right = C.AFFINE

        if left in (C.CONCAVE, C.AFFINE) and right in (C.CONCAVE, C.AFFINE):
            return C.CONCAVE
        return C.UNKNOWN

    @property
    def shape(self) -> tuple[int, ...]:
        left_shape = self.left.shape if isinstance(self.left, BaseExpr) else np.shape(self.left)
        right_shape = self.right.shape if isinstance(self.right, BaseExpr) else np.shape(self.right)
        return broadcast_shapes(left_shape, right_shape)
