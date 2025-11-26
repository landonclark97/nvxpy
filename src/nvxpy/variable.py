import threading

import autograd.numpy as np

from .expression import BaseExpr
from .constraint import Constraint
from .constants import Curvature as C


# Thread-local storage for variable ID counter
_thread_local = threading.local()


def _get_next_id() -> int:
    """Get the next variable ID in a thread-safe manner."""
    if not hasattr(_thread_local, "var_id"):
        _thread_local.var_id = 0
    current = _thread_local.var_id
    _thread_local.var_id += 1
    return current


def reset_variable_ids():
    """Reset the variable ID counter for the current thread."""
    _thread_local.var_id = 0


class Variable(BaseExpr):
    __array_priority__ = 100

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
        assert all(isinstance(s, int) for s in shape), (
            "Shape must be a tuple of integers"
        )
        assert all(s > 0 for s in shape), "Shape must be a tuple of positive integers"
        assert len(shape) == 1 or len(shape) == 2, (
            "Shape must be a tuple of length 1 or 2"
        )

        var_id = _get_next_id()
        self.name = name if name else f"x{var_id}"
        self.shape = shape
        self.size = int(np.prod(shape)) if shape else 1
        self._value = None
        self._id = var_id

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

        self.is_integer = bool(integer)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, val):
        self._value = np.array(val).reshape(self.shape)

    def __repr__(self):
        return f"Var({self.name}, shape={self.shape})"
    
    def __hash__(self):
        return hash(str(self))
    
    @property
    def curvature(self):
        return C.AFFINE
