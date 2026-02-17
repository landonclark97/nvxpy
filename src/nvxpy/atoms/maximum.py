from __future__ import annotations

import autograd.numpy as np

from ..expression import Expr, BaseExpr, broadcast_shapes, ExprLike
from ..constants import Curvature as C


class maximum(Expr):
    """Element-wise maximum of two arrays.

    maximum(x, y) is convex (pointwise max of convex functions is convex).

    Curvature rules:
    - maximum(constant, constant) -> constant
    - maximum(affine, affine) -> convex
    - maximum(convex, convex) -> convex
    - maximum(concave, _) -> unknown
    """

    def __init__(self, left: ExprLike, right: ExprLike) -> None:
        super().__init__("maximum", left, right)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.maximum(x, y)

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

        if left in (C.CONVEX, C.AFFINE) and right in (C.CONVEX, C.AFFINE):
            return C.CONVEX
        return C.UNKNOWN

    @property
    def shape(self) -> tuple[int, ...]:
        left_shape = (
            self.left.shape if isinstance(self.left, BaseExpr) else np.shape(self.left)
        )
        right_shape = (
            self.right.shape
            if isinstance(self.right, BaseExpr)
            else np.shape(self.right)
        )
        return broadcast_shapes(left_shape, right_shape)
