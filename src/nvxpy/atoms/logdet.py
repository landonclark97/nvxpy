import autograd.numpy as np

from ..expression import Expr
from ..constants import Curvature as C


class logdet(Expr):
    def __init__(self, left):
        super().__init__("logdet", left)

    def __call__(self, x):
        return np.log(np.linalg.det(x))

    @property
    def curvature(self):
        return C.UNKNOWN