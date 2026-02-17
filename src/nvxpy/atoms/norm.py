from __future__ import annotations

import autograd.numpy as np

from ..expression import Expr, BaseExpr, ExprLike
from ..constants import Curvature as C, EPSILON


def _smooth_norm(
    x: np.ndarray, ord: int | float | str = 2, axis: int | None = None
) -> np.ndarray:
    """
    Compute norm with epsilon smoothing to avoid gradient singularity at zero.

    The gradient of ||x|| is x / ||x||, which is undefined when ||x|| = 0.
    We compute sqrt(||x||^2 + eps) instead, which has a well-defined gradient
    everywhere: x / sqrt(||x||^2 + eps).
    """
    if ord == 2 and axis is None:
        # L2 norm (default): sqrt(sum(x^2) + eps)
        return np.sqrt(np.sum(x * x) + EPSILON)
    elif ord == "fro":
        # Frobenius norm: sqrt(sum(x^2) + eps)
        return np.sqrt(np.sum(x * x) + EPSILON)
    elif ord == 1 and axis is None:
        # L1 norm: sum(|x|) - no singularity issue, but use smooth abs
        return np.sum(np.sqrt(x * x + EPSILON))
    elif ord == np.inf and axis is None:
        # Inf norm: max(|x|) - use logsumexp approximation for smoothness
        return np.max(np.sqrt(x * x + EPSILON))
    else:
        # Fall back to numpy for other norms, with post-hoc smoothing
        raw_norm = np.linalg.norm(x, ord=ord, axis=axis)
        return np.sqrt(raw_norm * raw_norm + EPSILON)


class norm(Expr):
    """Vector or matrix norm.

    norm(x, ord=2) computes the L2 norm by default.
    Supports ord=1, 2, inf, 'fro', 'nuc'.

    Curvature rules:
    - For ord >= 1: convex for affine/convex arguments
    - For ord < 1: generally non-convex
    """

    def __init__(
        self, left: ExprLike, ord: int | float | str = 2, axis: int | None = None
    ) -> None:
        self.ord = ord
        self.axis = axis
        super().__init__("norm", left)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return _smooth_norm(x, ord=self.ord, axis=self.axis)

    @property
    def curvature(self) -> C:
        arg = self.left
        arg_cvx = arg.curvature if isinstance(arg, BaseExpr) else C.CONSTANT

        if arg_cvx == C.CONSTANT:
            arg_cvx = C.AFFINE

        o = self.ord
        is_convex_ord = False

        if o is None:
            is_convex_ord = True

        elif isinstance(o, (int, float)):
            if o >= 1 or np.isposinf(o):
                is_convex_ord = True

        elif isinstance(o, str):
            if o in ("fro", "nuc"):
                is_convex_ord = True

        if is_convex_ord and arg_cvx in (C.AFFINE, C.CONVEX):
            return C.CONVEX

        return C.UNKNOWN

    @property
    def shape(self) -> tuple[int, ...]:
        if self.axis is None:
            return (1,)
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
        return tuple(result) if result else (1,)
