from __future__ import annotations

import autograd.numpy as np

from ..expression import Expr, BaseExpr, ExprLike
from ..overrides import svd
from ..constants import Curvature as C


class PolarDecomposition(Expr):
    """Polar decomposition of a matrix.

    Returns the orthogonal factor R from X = R @ P where R is orthogonal
    and P is positive semidefinite. Uses SVD: R = U @ diag(S) @ Vt with
    sign correction to ensure det(R) > 0.

    Curvature is unknown (non-convex operation).
    """

    def __init__(self, left: ExprLike) -> None:
        super().__init__("polar_decomp", left)
        # Validate that input is 2D
        if isinstance(left, BaseExpr):
            if len(left.shape) != 2:
                raise ValueError("PolarDecomposition requires a 2D matrix input")
        elif hasattr(left, 'shape'):
            if len(left.shape) != 2:
                raise ValueError("PolarDecomposition requires a 2D matrix input")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        U, _, Vt = svd(x, full_matrices=False)
        det_UVt = np.linalg.det(U @ Vt)
        S = np.ones((x.shape[0],))
        S[-1] = np.sign(det_UVt)
        return U @ np.diag(S) @ Vt

    @property
    def curvature(self) -> C:
        return C.UNKNOWN

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.left, BaseExpr):
            return self.left.shape
        return np.shape(self.left) or (1,)