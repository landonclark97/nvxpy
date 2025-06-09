from ..constraint import Constraint
from ..set import Set
from ..atoms.polar import PolarDecomposition



class SO(Set):

    def __init__(self, n):
        super().__init__(f"SO({n})")
        self.n = n


    def constrain(self, var):
        assert var.shape == (self.n, self.n)
        return Constraint(var, "==", PolarDecomposition(var))