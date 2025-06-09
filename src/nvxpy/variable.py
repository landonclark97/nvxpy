import autograd.numpy as np

from .expression import Expr
from .constraint import Constraint
from .set import Set


class Variable:

    __array_priority__ = 100
    
    _ids = 0

    def __init__(
            self,
            shape=(1,),
            name=None,
            symmetric=False,
            PSD=False,
            NSD=False,
            pos=False,
            neg=False,
            integer=False,
        ):
        assert isinstance(shape, tuple), "Shape must be a tuple"
        assert len(shape) > 0, "Shape must be non-empty"
        assert all(isinstance(s, int) for s in shape), "Shape must be a tuple of integers"
        assert all(s > 0 for s in shape), "Shape must be a tuple of positive integers"
        assert len(shape) == 1 or len(shape) == 2, "Shape must be a tuple of length 1 or 2"

        self.name = name if name else f"x{Variable._ids}"
        self.shape = shape
        self.size = int(np.prod(shape)) if shape else 1
        self._value = None
        self._id = Variable._ids
        Variable._ids += 1

        self.constraints = []

        if symmetric:
            assert len(shape) == 2, "Symmetric variable must be a square matrix"
            assert shape[0] == shape[1], "Symmetric variable must be a square matrix"
            U_inds = np.triu_indices(shape[0], k=1)
            L_inds = np.tril_indices(shape[0], k=-1)
            self.constraints.append(Constraint(self[U_inds], "==", self[L_inds]))

        if PSD:
            assert len(shape) == 2, "PSD variable must be a square matrix"
            assert shape[0] == shape[1], "PSD variable must be a square matrix"
            assert not neg, "PSD variable cannot be negative"
            self.constraints.append(Constraint(self, ">>", 0))

        if NSD:
            assert len(shape) == 2, "NSD variable must be a square matrix"
            assert shape[0] == shape[1], "NSD variable must be a square matrix"
            assert not pos, "NSD variable cannot be positive"
            self.constraints.append(Constraint(self, "<<", 0))

        if pos:
            assert not neg and not NSD, "Positive variable cannot be NSD or negative"
            self.constraints.append(Constraint(self, ">=", 0))

        if neg:
            assert not PSD and not pos, "Negative variable cannot be PSD or positive"
            self.constraints.append(Constraint(self, "<=", 0))

        if integer:
            raise NotImplementedError("Integer variables are not supported yet")


    @property
    def value(self):
        return self._value


    @value.setter
    def value(self, val):
        self._value = np.array(val).reshape(self.shape)


    def __repr__(self):
        return f"Var({self.name}, shape={self.shape})"


    @property
    def T(self):
        return Expr("transpose", self)


    def __add__(self, other): return Expr("add", self, other)
    def __radd__(self, other): return Expr("add", other, self)
    def __sub__(self, other): return Expr("sub", self, other)
    def __rsub__(self, other): return Expr("sub", other, self)
    def __mul__(self, other): return Expr("mul", self, other)
    def __rmul__(self, other): return Expr("mul", other, self)
    def __matmul__(self, other): return Expr("matmul", self, other)
    def __rmatmul__(self, other): return Expr("matmul", other, self)
    def __truediv__(self, other): return Expr("div", self, other)
    def __pow__(self, other): return Expr("pow", self, other)
    def __neg__(self): return Expr("neg", self)

    def __getitem__(self, key): return Expr("getitem", self, key)

    def __ge__(self, other): return Constraint(self, ">=", other)
    def __le__(self, other): return Constraint(self, "<=", other)
    def __eq__(self, other): return Constraint(self, "==", other)
    def __rshift__(self, other): return Constraint(self, ">>", other)
    def __lshift__(self, other): return Constraint(self, "<<", other)
    
    def __xor__(self, other):
        assert isinstance(other, Set), "Set must be a Set object"
        return other.constrain(self)
