from __future__ import annotations

import autograd.numpy as np

from ..expression import Expr, ExprLike
from ..constants import Curvature as C


class det(Expr):
    """Matrix determinant.

    det(A) returns the determinant of square matrix A.
    Generally non-convex.
    """

    def __init__(self, left: ExprLike) -> None:
        super().__init__("det", left)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.linalg.det(x)

    @property
    def curvature(self) -> C:
        return C.UNKNOWN

    @property
    def shape(self) -> tuple[int, ...]:
        return (1,)