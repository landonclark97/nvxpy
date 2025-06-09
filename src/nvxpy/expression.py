from .constraint import Constraint
from .set import Set


class Expr:
    __array_priority__ = 100

    def __init__(self, op, left, right=None):
        self.op = op
        self.left = left
        self.right = right

    @property
    def value(self):
        from .parser import eval_expression

        return eval_expression(self, None, use_value=True)

    @property
    def T(self):
        return Expr("transpose", self)

    def __add__(self, other):
        return Expr("add", self, other)

    def __radd__(self, other):
        return Expr("add", other, self)

    def __sub__(self, other):
        return Expr("sub", self, other)

    def __rsub__(self, other):
        return Expr("sub", other, self)

    def __mul__(self, other):
        return Expr("mul", self, other)

    def __rmul__(self, other):
        return Expr("mul", other, self)

    def __matmul__(self, other):
        return Expr("matmul", self, other)

    def __rmatmul__(self, other):
        return Expr("matmul", other, self)

    def __truediv__(self, other):
        return Expr("div", self, other)

    def __pow__(self, other):
        return Expr("pow", self, other)

    def __neg__(self):
        return Expr("neg", self)

    def __getitem__(self, key):
        return Expr("getitem", self, key)

    def __ge__(self, other):
        return Constraint(self, ">=", other)

    def __le__(self, other):
        return Constraint(self, "<=", other)

    def __eq__(self, other):
        return Constraint(self, "==", other)

    def __rshift__(self, other):
        return Constraint(self, ">>", other)

    def __lshift__(self, other):
        return Constraint(self, "<<", other)

    def __xor__(self, other):
        assert isinstance(other, Set), "Set must be a Set object"
        return other.constrain(self)
