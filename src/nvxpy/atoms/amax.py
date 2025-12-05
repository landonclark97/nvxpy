from __future__ import annotations

import autograd.numpy as np

from ..expression import Expr, BaseExpr, ExprLike
from ..constants import Curvature as C


class amax(Expr):
    """Maximum value along an axis.

    amax(x) returns the maximum element of x (scalar if axis=None).

    Curvature rules:
    - amax(constant) -> constant
    - amax(affine) -> convex
    - amax(convex) -> convex
    - amax(concave) -> unknown
    """

    def __init__(self, left: ExprLike, axis: int | None = None) -> None:
        self.axis = axis
        super().__init__("amax", left)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.amax(x, axis=self.axis)

    @property
    def curvature(self) -> C:
        arg = self.left
        arg_cvx = arg.curvature if isinstance(arg, BaseExpr) else C.CONSTANT

        if arg_cvx == C.CONSTANT:
            return C.CONSTANT

        if arg_cvx in (C.AFFINE, C.CONVEX):
            return C.CONVEX

        return C.UNKNOWN

    @property
    def shape(self) -> tuple[int, ...]:
        if self.axis is None:
            return (1,)
        arg_shape = self.left.shape if isinstance(self.left, BaseExpr) else np.shape(self.left)
        ndim = len(arg_shape)
        axis = self.axis if self.axis >= 0 else ndim + self.axis
        if axis < 0 or axis >= ndim:
            raise ValueError(f"axis {self.axis} is out of bounds for array of dimension {ndim}")
        result = list(arg_shape)
        result.pop(axis)
        return tuple(result) if result else (1,)