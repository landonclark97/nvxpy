from ..constraint import Constraint
from ..set import Set
from ..constants import EPSILON


class PerspectiveCone(Set):
    def __init__(self, func, expr, p):
        super().__init__("PerspectiveCone")
        self.func = func
        self.expr = expr
        self.p = p

    def constrain(self, var):
        return Constraint(var, "==", self.p * self.func(self.expr / (self.p + EPSILON)))
