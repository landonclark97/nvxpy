from ..set import Set
from ..constraint import Constraint
from ..atoms.polar import PolarDecomposition


class SO(Set):
    def __init__(self, n):
        super().__init__(name=f"SO({n})")
        self.n = n

    def constrain(self, var):
        if var.shape != (self.n, self.n):
            raise ValueError(f"Variable shape {var.shape} does not match SO({self.n}), expected ({self.n}, {self.n})")
        return Constraint(var, "<-", PolarDecomposition(var))
