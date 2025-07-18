from .constants import Curvature as C


class Constraint:
    def __init__(self, left, op, right):
        assert op in [">=", "<=", "==", ">>", "<<", "<-"]
        self.left = left
        self.op = op
        self.right = right

    @property
    def curvature(self):
        res = self.right - self.left if self.op in [">=", "==", ">>", "<-"] else self.left - self.right
        curvature = res.curvature
        return curvature if curvature in (C.CONSTANT, C.AFFINE, C.CONVEX) else C.UNKNOWN
