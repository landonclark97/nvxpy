from __future__ import annotations

import autograd.numpy as np

from ..expression import Expr, BaseExpr, ExprLike
from ..constants import Curvature as C


class sum(Expr):
    """Sum of array elements along an axis.

    sum(x) returns the sum of all elements (scalar if axis=None).
    Preserves curvature (affine operation).
    """

    def __init__(self, left: ExprLike, axis: int | None = None) -> None:
        self.axis = axis
        super().__init__("sum", left)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=self.axis)

    @property
    def curvature(self) -> C:
        arg = self.left
        arg_curv = arg.curvature if isinstance(arg, BaseExpr) else C.CONSTANT

        return arg_curv

    @property
    def shape(self) -> tuple[int, ...]:
        if self.axis is None:
            return ()  # Scalar result, consistent with NumPy
        arg_shape = (
            self.left.shape if isinstance(self.left, BaseExpr) else np.shape(self.left)
        )
        ndim = len(arg_shape)
        axis = self.axis if self.axis >= 0 else ndim + self.axis
        if axis < 0 or axis >= ndim:
            raise ValueError(
                f"axis {self.axis} is out of bounds for array of dimension {ndim}"
            )
        result = list(arg_shape)
        result.pop(axis)
        return tuple(result) if result else ()  # Scalar result if all dims summed
