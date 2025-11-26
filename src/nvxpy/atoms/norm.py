import autograd.numpy as np

from ..expression import Expr, BaseExpr
from ..constants import Curvature as C

# Small epsilon to prevent gradient singularity when norm approaches zero
_NORM_EPS = 1e-12


def _smooth_norm(x, ord=2, axis=None):
    """
    Compute norm with epsilon smoothing to avoid gradient singularity at zero.

    The gradient of ||x|| is x / ||x||, which is undefined when ||x|| = 0.
    We compute sqrt(||x||^2 + eps) instead, which has a well-defined gradient
    everywhere: x / sqrt(||x||^2 + eps).
    """
    if ord == 2 and axis is None:
        # L2 norm (default): sqrt(sum(x^2) + eps)
        return np.sqrt(np.sum(x * x) + _NORM_EPS)
    elif ord == "fro":
        # Frobenius norm: sqrt(sum(x^2) + eps)
        return np.sqrt(np.sum(x * x) + _NORM_EPS)
    elif ord == 1 and axis is None:
        # L1 norm: sum(|x|) - no singularity issue, but use smooth abs
        return np.sum(np.sqrt(x * x + _NORM_EPS))
    elif ord == np.inf and axis is None:
        # Inf norm: max(|x|) - use logsumexp approximation for smoothness
        return np.max(np.sqrt(x * x + _NORM_EPS))
    else:
        # Fall back to numpy for other norms, with post-hoc smoothing
        raw_norm = np.linalg.norm(x, ord=ord, axis=axis)
        return np.sqrt(raw_norm * raw_norm + _NORM_EPS)


class norm(Expr):
    def __init__(self, left, ord=2, axis=None):
        self.ord = ord
        self.axis = axis
        super().__init__("norm", left)

    def __call__(self, x):
        return _smooth_norm(x, ord=self.ord, axis=self.axis)

    @property
    def curvature(self):
        arg = self.left
        arg_cvx = (
            arg.curvature
            if isinstance(arg, BaseExpr)
            else C.CONSTANT
        )

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