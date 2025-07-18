import autograd.numpy as np

from ..expression import Expr
from ..overrides import svd
from ..constants import Curvature as C


class PolarDecomposition(Expr):
    def __init__(self, left):
        super().__init__("polar_decomp", left)

    def __call__(self, x):
        U, _, Vt = svd(x, full_matrices=False)
        det_UVt = np.linalg.det(U @ Vt)
        S = np.ones((x.shape[0],))
        S[-1] = np.sign(det_UVt)
        return U @ np.diag(S) @ Vt

    @property
    def curvature(self):
        return C.UNKNOWN